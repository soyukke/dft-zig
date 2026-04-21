//! Single-k-point q≠0 DFPT SCF solver.
//!
//! Solves the Sternheimer equation for one (atom, direction) perturbation
//! at a fixed q using potential mixing. Density mixing is unstable at
//! finite q because V_H(G=0) = 8π/|q|² diverges, so we mix V^(1)_SCF
//! (Hartree + XC + bare) directly via a Pulay mixer with restart logic.
//!
//! NOTE: This single-k entry point is not currently wired into the
//! top-level phonon band path (which calls the multi-k solver for every
//! case). It is retained as a reference implementation that also makes
//! the mixing + Sternheimer logic easy to read standalone.

const std = @import("std");
const math = @import("../../math/math.zig");
const scf_mod = @import("../../scf/scf.zig");
const plane_wave = @import("../../plane_wave/basis.zig");

const dfpt = @import("../dfpt.zig");
const perturbation = dfpt.perturbation;
const sternheimer = dfpt.sternheimer;
const GroundState = dfpt.GroundState;
const DfptConfig = dfpt.DfptConfig;
const PerturbationResult = dfpt.PerturbationResult;
const logDfpt = dfpt.logDfpt;

const cross_basis = @import("cross_basis.zig");
const applyV1PsiQCached = cross_basis.applyV1PsiQCached;
const computeRho1Q = cross_basis.computeRho1Q;
const computeRho1QCached = cross_basis.computeRho1QCached;
const complexRealToReciprocal = cross_basis.complexRealToReciprocal;

const dynmat_elem_q = @import("dynmat_elem_q.zig");
const computeElecDynmatElementQ = dynmat_elem_q.computeElecDynmatElementQ;

/// Solve DFPT perturbation at q≠0 for a single perturbation (atom, direction).
/// Uses **potential mixing** (mix V^(1)_SCF, not ρ^(1)) for stable convergence.
/// Density mixing is unstable at finite q because V_H(G=0) = 8π/|q|² diverges.
pub fn solvePerturbationQ(
    alloc: std.mem.Allocator,
    gs: GroundState,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
    q_cart: math.Vec3,
    gvecs_kq: []const plane_wave.GVector,
    apply_ctx_kq: *scf_mod.ApplyContext,
    map_kq: *const scf_mod.PwGridMap,
    occ_kq: []const []const math.Complex,
    n_occ_kq: usize,
) !PerturbationResult {
    const n_pw_k = gs.gvecs.len;
    const n_pw_kq = gvecs_kq.len;
    const total = gs.grid.count();
    const n_occ = gs.n_occ;
    const grid = gs.grid;

    // Build V_ext^(1)_q(G) for this perturbation (bare, fixed)
    const vloc1_g = try perturbation.buildLocalPerturbationQ(
        alloc,
        grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        q_cart,
        gs.local_cfg,
        gs.ff_tables,
    );
    defer alloc.free(vloc1_g);

    // Debug: verify vloc1_g depends on direction
    {
        var vloc_norm: f64 = 0.0;
        var vloc_sum_r: f64 = 0.0;
        var vloc_sum_i: f64 = 0.0;
        for (vloc1_g) |c| {
            vloc_norm += c.r * c.r + c.i * c.i;
            vloc_sum_r += c.r;
            vloc_sum_i += c.i;
        }
        logDfpt("dfptQ_vloc1: atom={d} dir={d} |vloc1_g|={e:.6} sum=({e:.6},{e:.6})\n", .{ atom_index, direction, @sqrt(vloc_norm), vloc_sum_r, vloc_sum_i });
        // Print first few G-point values
        const nshow = @min(vloc1_g.len, 5);
        for (0..nshow) |gi| {
            logDfpt("  vloc1_g[{d}]=({e:.8},{e:.8})\n", .{ gi, vloc1_g[gi].r, vloc1_g[gi].i });
        }
        // Print G=0 value (index where h=k=l=0)
        {
            const g0_h: usize = @intCast(-grid.min_h);
            const g0_k: usize = @intCast(-grid.min_k);
            const g0_l: usize = @intCast(-grid.min_l);
            const g0_idx = g0_l * grid.ny * grid.nx + g0_k * grid.nx + g0_h;
            logDfpt("  vloc1_g[G=0, idx={d}]=({e:.8},{e:.8}) grid=({d},{d},{d}) min=({d},{d},{d})\n", .{
                g0_idx,     vloc1_g[g0_idx].r, vloc1_g[g0_idx].i,
                grid.nx,    grid.ny,           grid.nz,
                grid.min_h, grid.min_k,        grid.min_l,
            });
        }
    }

    // Build ρ^(1)_core,q(G) (fixed, NLCC)
    const rho1_core_g = try perturbation.buildCorePerturbationQ(
        alloc,
        grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        q_cart,
        gs.rho_core_tables,
    );
    defer alloc.free(rho1_core_g);

    // IFFT ρ^(1)_core,q → real space (complex)
    const rho1_core_g_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_g_copy);
    @memcpy(rho1_core_g_copy, rho1_core_g);
    const rho1_core_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_r);
    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_g_copy, rho1_core_r, null);

    // Allocate first-order wavefunctions in k+q basis
    var psi1 = try alloc.alloc([]math.Complex, n_occ);
    for (0..n_occ) |n| {
        psi1[n] = try alloc.alloc(math.Complex, n_pw_kq);
        @memset(psi1[n], math.complex.init(0.0, 0.0));
    }

    // Map for k-basis
    const map_k_ptr: *const scf_mod.PwGridMap = &gs.apply_ctx.map;

    // Initialize V_SCF(G) = V_loc^(1)(G) [bare perturbation, no screening]
    var v_scf_g = try alloc.alloc(math.Complex, total);
    @memcpy(v_scf_g, vloc1_g);

    // Pre-compute ψ^(0)(r) for all occupied bands (invariant during SCF)
    const psi0_r_cache = try alloc.alloc([]math.Complex, n_occ);
    defer {
        for (psi0_r_cache) |band| alloc.free(band);
        alloc.free(psi0_r_cache);
    }
    for (0..n_occ) |n| {
        psi0_r_cache[n] = try alloc.alloc(math.Complex, total);
        const work = try alloc.alloc(math.Complex, total);
        defer alloc.free(work);
        @memset(work, math.complex.init(0.0, 0.0));
        map_k_ptr.scatter(gs.wavefunctions[n], work);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work, psi0_r_cache[n], null);
    }
    const psi0_r_const: []const []const math.Complex = @ptrCast(psi0_r_cache);

    // Pulay mixer for potential mixing
    var pulay = scf_mod.ComplexPulayMixer.init(alloc, cfg.pulay_history);
    defer pulay.deinit();

    // Pulay restart state
    var best_vresid: f64 = std.math.inf(f64);
    var best_v_scf: ?[]math.Complex = null;
    defer if (best_v_scf) |v| alloc.free(v);
    var pulay_active_since: usize = cfg.pulay_start;
    const restart_factor: f64 = 5.0;
    var force_converge: bool = false;

    // DFPT SCF loop — potential mixing
    var iter: usize = 0;
    while (iter < cfg.scf_max_iter) : (iter += 1) {
        // IFFT V_SCF(G) → V_SCF(r) [complex]
        const v_scf_g_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_scf_g_copy);
        @memcpy(v_scf_g_copy, v_scf_g);
        const v_scf_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_scf_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, v_scf_g_copy, v_scf_r, null);

        // Debug: on first iteration, print some v_scf values
        if (iter == 0) {
            logDfpt("dfptQ_vscf: atom={d} dir={d} iter=0 v_scf_r[0]=({e:.8},{e:.8}) v_scf_r[1]=({e:.8},{e:.8}) v_scf_r[100]=({e:.8},{e:.8})\n", .{
                atom_index,                                  direction,
                v_scf_r[0].r,                                v_scf_r[0].i,
                v_scf_r[1].r,                                v_scf_r[1].i,
                v_scf_r[@min(@as(usize, 100), total - 1)].r, v_scf_r[@min(@as(usize, 100), total - 1)].i,
            });
            logDfpt("dfptQ_vscf: v_scf_g[0]=({e:.8},{e:.8}) v_scf_g[1]=({e:.8},{e:.8}) v_scf_g[2]=({e:.8},{e:.8})\n", .{
                v_scf_g[0].r, v_scf_g[0].i,
                v_scf_g[1].r, v_scf_g[1].i,
                v_scf_g[2].r, v_scf_g[2].i,
            });
        }

        // Nonlocal contexts for V_nl^(1) (cross-basis: k → k+q)
        const nl_ctx_k_opt = gs.apply_ctx.nonlocal_ctx;
        const nl_ctx_kq_opt = apply_ctx_kq.nonlocal_ctx;

        // Solve Sternheimer for each occupied band
        for (0..n_occ) |n| {
            // RHS: -P_c^{k+q} × H^(1)|ψ^(0)_{n,k}⟩
            // H^(1)|ψ⟩ = V_SCF(r)|ψ(r)⟩ + V_nl^(1)|ψ⟩
            const rhs = try applyV1PsiQCached(alloc, grid, map_kq, v_scf_r, psi0_r_cache[n], n_pw_kq);
            defer alloc.free(rhs);

            // Add nonlocal perturbation: V_nl^(1)_{q}|ψ_k⟩ (cross-basis: k → k+q)
            if (nl_ctx_k_opt != null and nl_ctx_kq_opt != null) {
                const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
                defer alloc.free(nl_out);
                try perturbation.applyNonlocalPerturbationQ(
                    alloc,
                    gs.gvecs,
                    gvecs_kq,
                    gs.atoms,
                    nl_ctx_k_opt.?,
                    nl_ctx_kq_opt.?,
                    atom_index,
                    direction,
                    1.0 / grid.volume,
                    gs.wavefunctions[n],
                    nl_out,
                );
                for (0..n_pw_kq) |g| {
                    rhs[g] = math.complex.add(rhs[g], nl_out[g]);
                }
            }

            // Negate: rhs = -H^(1)|ψ⟩
            for (0..n_pw_kq) |g| {
                rhs[g] = math.complex.scale(rhs[g], -1.0);
            }

            // Project onto conduction band in k+q space
            sternheimer.projectConduction(rhs, occ_kq, n_occ_kq);

            // Solve: (H_{k+q} - ε_{n,k} + α·P_v^{k+q})|ψ^(1)⟩ = rhs
            const result = try sternheimer.solve(
                alloc,
                apply_ctx_kq,
                rhs,
                gs.eigenvalues[n],
                occ_kq,
                n_occ_kq,
                gvecs_kq,
                .{
                    .tol = cfg.sternheimer_tol,
                    .max_iter = cfg.sternheimer_max_iter,
                    .alpha_shift = cfg.alpha_shift,
                },
            );

            alloc.free(psi1[n]);
            psi1[n] = result.psi1;
        }

        // Compute ρ^(1)_{+q}(r) from cross-basis wavefunctions [weight=4×wtk/Ω]
        const psi1_const_view = try alloc.alloc([]const math.Complex, n_occ);
        defer alloc.free(psi1_const_view);
        for (0..n_occ) |n2| psi1_const_view[n2] = psi1[n2];
        const rho1_r = try computeRho1QCached(
            alloc,
            grid,
            map_kq,
            psi0_r_const,
            psi1_const_view,
            n_occ,
            1.0, // wtk=1.0 for single k-point backward compat
        );
        defer alloc.free(rho1_r);

        // FFT ρ^(1)(r) → ρ^(1)(G)
        const rho1_g = try complexRealToReciprocal(alloc, grid, rho1_r);
        defer alloc.free(rho1_g);

        // Diagnostic: ρ^(1) norm
        {
            var rho_norm: f64 = 0.0;
            for (0..total) |di| {
                rho_norm += rho1_g[di].r * rho1_g[di].r + rho1_g[di].i * rho1_g[di].i;
            }
            rho_norm = @sqrt(rho_norm);
            // D_elec from bare V^(1) and current ρ^(1)
            const d_elec_diag = computeElecDynmatElementQ(vloc1_g, rho1_g, grid.volume);
            logDfpt("dfptQ_diag: iter={d} |rho1|={e:.6} D_elec_bare(0)=({e:.6},{e:.6})\n", .{ iter, rho_norm, d_elec_diag.r, d_elec_diag.i });
        }

        // Build V_out(G) = V_loc^(1) + V_H^(1)[ρ] + V_xc^(1)[ρ]
        const vh1_g = try perturbation.buildHartreePerturbationQ(alloc, grid, rho1_g, q_cart);
        defer alloc.free(vh1_g);

        // V_xc^(1)(r) = f_xc(r) × ρ^(1)_total(r)  where ρ_total = ρ_val + ρ_core
        const rho1_g_copy2 = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_g_copy2);
        @memcpy(rho1_g_copy2, rho1_g);
        const rho1_val_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_val_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_g_copy2, rho1_val_r, null);

        const rho1_total_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_total_r);
        for (0..total) |i| {
            rho1_total_r[i] = math.complex.add(rho1_val_r[i], rho1_core_r[i]);
        }

        const vxc1_r = try perturbation.buildXcPerturbationFullComplex(alloc, gs, rho1_total_r);
        defer alloc.free(vxc1_r);

        // FFT V_xc^(1)(r) → V_xc^(1)(G)
        const vxc1_r_fft = try alloc.alloc(math.Complex, total);
        defer alloc.free(vxc1_r_fft);
        @memcpy(vxc1_r_fft, vxc1_r);
        const vxc1_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(vxc1_g);
        try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, vxc1_r_fft, vxc1_g, null);

        // V_out(G) = V_loc + V_H + V_xc
        const v_out_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_out_g);
        for (0..total) |i| {
            v_out_g[i] = math.complex.add(
                math.complex.add(vloc1_g[i], vh1_g[i]),
                vxc1_g[i],
            );
        }

        // Compute residual = V_out - V_SCF
        var residual_norm: f64 = 0.0;
        const residual = try alloc.alloc(math.Complex, total);
        for (0..total) |i| {
            residual[i] = math.complex.sub(v_out_g[i], v_scf_g[i]);
            residual_norm += residual[i].r * residual[i].r + residual[i].i * residual[i].i;
        }
        residual_norm = @sqrt(residual_norm);

        logDfpt("dfptQ_scf: iter={d} vresid={e:.6}\n", .{ iter, residual_norm });

        if (residual_norm < cfg.scf_tol or (force_converge and residual_norm < 10.0 * cfg.scf_tol)) {
            alloc.free(residual);
            logDfpt("dfptQ_scf: converged at iter={d} vresid={e:.6}\n", .{ iter, residual_norm });
            break;
        }

        // Track best residual and save corresponding V_SCF
        if (residual_norm < best_vresid) {
            best_vresid = residual_norm;
            if (best_v_scf == null) best_v_scf = try alloc.alloc(math.Complex, total);
            @memcpy(best_v_scf.?, v_scf_g);
        }

        // Pulay restart: if residual exceeds restart_factor × best, reset and restore
        if (iter >= pulay_active_since and residual_norm > restart_factor * best_vresid and best_vresid < 1.0) {
            if (best_v_scf) |v| @memcpy(v_scf_g, v);
            // If best is near convergence, force accept on next iteration
            if (best_vresid < 10.0 * cfg.scf_tol) {
                force_converge = true;
                logDfpt("dfptQ_scf: Pulay restart (near-converged) at iter={d} vresid={e:.6} best={e:.6}\n", .{ iter, residual_norm, best_vresid });
                alloc.free(residual);
                continue;
            }
            pulay.reset();
            pulay_active_since = iter + 1 + cfg.pulay_start;
            logDfpt("dfptQ_scf: Pulay restart at iter={d} vresid={e:.6} best={e:.6}\n", .{ iter, residual_norm, best_vresid });
            alloc.free(residual);
            continue;
        }

        // Mix V_SCF using Pulay (delayed start) or simple linear mixing
        if (cfg.pulay_history > 0 and iter >= pulay_active_since) {
            // Pulay/DIIS: ownership of residual transfers to mixer
            try pulay.mixWithResidual(v_scf_g, residual, cfg.mixing_beta);
        } else {
            // Simple linear mixing: V_SCF += β × residual
            const beta = cfg.mixing_beta;
            for (0..total) |i| {
                v_scf_g[i] = math.complex.add(v_scf_g[i], math.complex.scale(residual[i], beta));
            }
            alloc.free(residual);
        }
    }

    // Compute final ρ^(1)(G) from converged ψ^(1)
    const final_rho1_r = try computeRho1Q(
        alloc,
        grid,
        map_k_ptr,
        map_kq,
        gs.wavefunctions,
        psi1,
        n_occ,
        n_pw_k,
        n_pw_kq,
        1.0, // wtk=1.0 for single k-point backward compat
    );
    const final_rho1_g = try complexRealToReciprocal(alloc, grid, final_rho1_r);
    alloc.free(final_rho1_r);

    alloc.free(v_scf_g);

    return .{
        .rho1_g = final_rho1_g,
        .psi1 = psi1,
    };
}
