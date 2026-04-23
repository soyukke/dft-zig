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
const log_dfpt = dfpt.log_dfpt;
const Grid = scf_mod.Grid;

const cross_basis = @import("cross_basis.zig");
const apply_v1_psi_q_cached = cross_basis.apply_v1_psi_q_cached;
const compute_rho1_q = cross_basis.compute_rho1_q;
const compute_rho1_q_cached = cross_basis.compute_rho1_q_cached;
const complex_real_to_reciprocal = cross_basis.complex_real_to_reciprocal;

const dynmat_elem_q = @import("dynmat_elem_q.zig");
const compute_elec_dynmat_element_q = dynmat_elem_q.compute_elec_dynmat_element_q;

const SingleQSetup = struct {
    vloc1_g: []math.Complex,
    rho1_core_r: []math.Complex,
    psi1: [][]math.Complex,
    psi0_r_cache: [][]math.Complex,
    psi0_r_const: []const []const math.Complex,
    v_scf_g: []math.Complex,

    fn deinit(self: *SingleQSetup, alloc: std.mem.Allocator) void {
        if (self.vloc1_g.len > 0) alloc.free(self.vloc1_g);
        if (self.rho1_core_r.len > 0) alloc.free(self.rho1_core_r);
        for (self.psi1) |band| {
            if (band.len > 0) alloc.free(band);
        }
        if (self.psi1.len > 0) alloc.free(self.psi1);
        for (self.psi0_r_cache) |band| {
            if (band.len > 0) alloc.free(band);
        }
        if (self.psi0_r_cache.len > 0) alloc.free(self.psi0_r_cache);
        if (self.psi0_r_const.len > 0) alloc.free(self.psi0_r_const);
        if (self.v_scf_g.len > 0) alloc.free(self.v_scf_g);
    }
};

const SingleQMixState = struct {
    pulay: scf_mod.ComplexPulayMixer,
    best_vresid: f64 = std.math.inf(f64),
    best_v_scf: ?[]math.Complex = null,
    pulay_active_since: usize,
    force_converge: bool = false,

    fn init(alloc: std.mem.Allocator, cfg: DfptConfig) SingleQMixState {
        return .{
            .pulay = scf_mod.ComplexPulayMixer.init(alloc, cfg.pulay_history),
            .pulay_active_since = cfg.pulay_start,
        };
    }

    fn deinit(self: *SingleQMixState, alloc: std.mem.Allocator) void {
        self.pulay.deinit();
        if (self.best_v_scf) |v| alloc.free(v);
    }
};

fn log_local_perturbation(
    vloc1_g: []const math.Complex,
    atom_index: usize,
    direction: usize,
    grid: Grid,
) void {
    var vloc_norm: f64 = 0.0;
    var vloc_sum_r: f64 = 0.0;
    var vloc_sum_i: f64 = 0.0;
    for (vloc1_g) |c| {
        vloc_norm += c.r * c.r + c.i * c.i;
        vloc_sum_r += c.r;
        vloc_sum_i += c.i;
    }
    log_dfpt(
        "dfptQ_vloc1: atom={d} dir={d} |vloc1_g|={e:.6} sum=({e:.6},{e:.6})\n",
        .{ atom_index, direction, @sqrt(vloc_norm), vloc_sum_r, vloc_sum_i },
    );
    const nshow = @min(vloc1_g.len, 5);
    for (0..nshow) |gi| {
        log_dfpt("  vloc1_g[{d}]=({e:.8},{e:.8})\n", .{ gi, vloc1_g[gi].r, vloc1_g[gi].i });
    }

    const g0_h: usize = @intCast(-grid.min_h);
    const g0_k: usize = @intCast(-grid.min_k);
    const g0_l: usize = @intCast(-grid.min_l);
    const g0_idx = g0_l * grid.ny * grid.nx + g0_k * grid.nx + g0_h;
    log_dfpt(
        "  vloc1_g[G=0, idx={d}]=({e:.8},{e:.8})" ++
            " grid=({d},{d},{d}) min=({d},{d},{d})\n",
        .{
            g0_idx,     vloc1_g[g0_idx].r, vloc1_g[g0_idx].i,
            grid.nx,    grid.ny,           grid.nz,
            grid.min_h, grid.min_k,        grid.min_l,
        },
    );
}

fn build_core_perturbation_real_space(
    alloc: std.mem.Allocator,
    gs: *const GroundState,
    atom_index: usize,
    direction: usize,
    q_cart: math.Vec3,
) ![]math.Complex {
    const rho1_core_g = try perturbation.build_core_perturbation_q(
        alloc,
        gs.grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        q_cart,
        gs.rho_core_tables,
    );
    defer alloc.free(rho1_core_g);

    const rho1_core_g_copy = try alloc.alloc(math.Complex, gs.grid.count());
    defer alloc.free(rho1_core_g_copy);

    @memcpy(rho1_core_g_copy, rho1_core_g);

    const rho1_core_r = try alloc.alloc(math.Complex, gs.grid.count());
    try scf_mod.fft_reciprocal_to_complex_in_place(
        alloc,
        gs.grid,
        rho1_core_g_copy,
        rho1_core_r,
        null,
    );
    return rho1_core_r;
}

fn init_psi1_storage(
    alloc: std.mem.Allocator,
    n_occ: usize,
    n_pw_kq: usize,
) ![][]math.Complex {
    const psi1 = try alloc.alloc([]math.Complex, n_occ);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| alloc.free(psi1[i]);
        alloc.free(psi1);
    }

    for (0..n_occ) |n| {
        psi1[n] = try alloc.alloc(math.Complex, n_pw_kq);
        @memset(psi1[n], math.complex.init(0.0, 0.0));
        built = n + 1;
    }
    return psi1;
}

fn cache_psi0_real_space(
    alloc: std.mem.Allocator,
    gs: *const GroundState,
) !struct {
    psi0_r_cache: [][]math.Complex,
    psi0_r_const: []const []const math.Complex,
} {
    const psi0_r_cache = try alloc.alloc([]math.Complex, gs.n_occ);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| alloc.free(psi0_r_cache[i]);
        alloc.free(psi0_r_cache);
    }

    for (0..gs.n_occ) |n| {
        psi0_r_cache[n] = try alloc.alloc(math.Complex, gs.grid.count());
        const work = try alloc.alloc(math.Complex, gs.grid.count());
        defer alloc.free(work);

        @memset(work, math.complex.init(0.0, 0.0));
        gs.apply_ctx.map.scatter(gs.wavefunctions[n], work);
        try scf_mod.fft_reciprocal_to_complex_in_place(
            alloc,
            gs.grid,
            work,
            psi0_r_cache[n],
            null,
        );
        built = n + 1;
    }

    const psi0_r_const = try alloc.alloc([]const math.Complex, gs.n_occ);
    errdefer alloc.free(psi0_r_const);
    for (0..gs.n_occ) |n| {
        psi0_r_const[n] = psi0_r_cache[n];
    }
    return .{
        .psi0_r_cache = psi0_r_cache,
        .psi0_r_const = psi0_r_const,
    };
}

fn init_single_q_setup(
    alloc: std.mem.Allocator,
    gs: *const GroundState,
    atom_index: usize,
    direction: usize,
    q_cart: math.Vec3,
    n_pw_kq: usize,
) !SingleQSetup {
    const vloc1_g = try perturbation.build_local_perturbation_q(
        alloc,
        gs.grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        q_cart,
        gs.local_cfg,
        gs.ff_tables,
    );
    errdefer alloc.free(vloc1_g);
    log_local_perturbation(vloc1_g, atom_index, direction, gs.grid);

    const rho1_core_r = try build_core_perturbation_real_space(
        alloc,
        gs,
        atom_index,
        direction,
        q_cart,
    );
    errdefer alloc.free(rho1_core_r);

    const psi1 = try init_psi1_storage(alloc, gs.n_occ, n_pw_kq);
    errdefer {
        for (psi1) |band| alloc.free(band);
        alloc.free(psi1);
    }

    const cache = try cache_psi0_real_space(alloc, gs);
    errdefer {
        for (cache.psi0_r_cache) |band| alloc.free(band);
        alloc.free(cache.psi0_r_cache);
        alloc.free(cache.psi0_r_const);
    }

    const v_scf_g = try alloc.alloc(math.Complex, gs.grid.count());
    errdefer alloc.free(v_scf_g);
    @memcpy(v_scf_g, vloc1_g);

    return .{
        .vloc1_g = vloc1_g,
        .rho1_core_r = rho1_core_r,
        .psi1 = psi1,
        .psi0_r_cache = cache.psi0_r_cache,
        .psi0_r_const = cache.psi0_r_const,
        .v_scf_g = v_scf_g,
    };
}

fn reciprocal_to_complex_copy(
    alloc: std.mem.Allocator,
    grid: Grid,
    values_g: []const math.Complex,
) ![]math.Complex {
    const work = try alloc.alloc(math.Complex, values_g.len);
    defer alloc.free(work);

    @memcpy(work, values_g);

    const values_r = try alloc.alloc(math.Complex, values_g.len);
    try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, work, values_r, null);
    return values_r;
}

fn log_scf_potential_sample(
    v_scf_r: []const math.Complex,
    v_scf_g: []const math.Complex,
    atom_index: usize,
    direction: usize,
    total: usize,
) void {
    const v_scf_r_100 = v_scf_r[@min(@as(usize, 100), total - 1)];
    log_dfpt(
        "dfptQ_vscf: atom={d} dir={d} iter=0" ++
            " v_scf_r[0]=({e:.8},{e:.8})" ++
            " v_scf_r[1]=({e:.8},{e:.8})" ++
            " v_scf_r[100]=({e:.8},{e:.8})\n",
        .{
            atom_index,    direction,
            v_scf_r[0].r,  v_scf_r[0].i,
            v_scf_r[1].r,  v_scf_r[1].i,
            v_scf_r_100.r, v_scf_r_100.i,
        },
    );
    log_dfpt(
        "dfptQ_vscf: v_scf_g[0]=({e:.8},{e:.8})" ++
            " v_scf_g[1]=({e:.8},{e:.8})" ++
            " v_scf_g[2]=({e:.8},{e:.8})\n",
        .{
            v_scf_g[0].r, v_scf_g[0].i,
            v_scf_g[1].r, v_scf_g[1].i,
            v_scf_g[2].r, v_scf_g[2].i,
        },
    );
}

fn solve_single_q_bands(
    alloc: std.mem.Allocator,
    gs: *const GroundState,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
    gvecs_kq: []const plane_wave.GVector,
    apply_ctx_kq: *scf_mod.ApplyContext,
    map_kq: *const scf_mod.PwGridMap,
    occ_kq: []const []const math.Complex,
    n_occ_kq: usize,
    v_scf_r: []const math.Complex,
    psi0_r_cache: [][]math.Complex,
    psi1: [][]math.Complex,
) !void {
    const nl_ctx_k_opt = gs.apply_ctx.nonlocal_ctx;
    const nl_ctx_kq_opt = apply_ctx_kq.nonlocal_ctx;
    for (0..gs.n_occ) |n| {
        const rhs = try apply_v1_psi_q_cached(
            alloc,
            gs.grid,
            map_kq,
            v_scf_r,
            psi0_r_cache[n],
            gvecs_kq.len,
        );
        defer alloc.free(rhs);

        if (nl_ctx_k_opt != null and nl_ctx_kq_opt != null) {
            const nl_out = try alloc.alloc(math.Complex, gvecs_kq.len);
            defer alloc.free(nl_out);

            try perturbation.apply_nonlocal_perturbation_q(
                alloc,
                gs.gvecs,
                gvecs_kq,
                gs.atoms,
                nl_ctx_k_opt.?,
                nl_ctx_kq_opt.?,
                atom_index,
                direction,
                1.0 / gs.grid.volume,
                gs.wavefunctions[n],
                nl_out,
            );
            for (0..gvecs_kq.len) |g| {
                rhs[g] = math.complex.add(rhs[g], nl_out[g]);
            }
        }

        for (0..gvecs_kq.len) |g| rhs[g] = math.complex.scale(rhs[g], -1.0);
        sternheimer.project_conduction(rhs, occ_kq, n_occ_kq);

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
}

fn build_single_q_rho1_g(
    alloc: std.mem.Allocator,
    gs: *const GroundState,
    map_kq: *const scf_mod.PwGridMap,
    psi0_r_const: []const []const math.Complex,
    psi1: [][]math.Complex,
) ![]math.Complex {
    const psi1_const_view = try alloc.alloc([]const math.Complex, gs.n_occ);
    defer alloc.free(psi1_const_view);

    for (0..gs.n_occ) |n| psi1_const_view[n] = psi1[n];

    const rho1_r = try compute_rho1_q_cached(
        alloc,
        gs.grid,
        map_kq,
        psi0_r_const,
        psi1_const_view,
        gs.n_occ,
        1.0,
    );
    defer alloc.free(rho1_r);

    return complex_real_to_reciprocal(alloc, gs.grid, rho1_r);
}

fn log_single_q_rho_diagnostic(
    iter: usize,
    rho1_g: []const math.Complex,
    vloc1_g: []const math.Complex,
    volume: f64,
) void {
    var rho_norm: f64 = 0.0;
    for (rho1_g) |rho| {
        rho_norm += rho.r * rho.r + rho.i * rho.i;
    }
    rho_norm = @sqrt(rho_norm);
    const d_elec_diag = compute_elec_dynmat_element_q(vloc1_g, rho1_g, volume);
    log_dfpt(
        "dfptQ_diag: iter={d} |rho1|={e:.6} D_elec_bare(0)=({e:.6},{e:.6})\n",
        .{ iter, rho_norm, d_elec_diag.r, d_elec_diag.i },
    );
}

fn build_single_q_output_potential(
    alloc: std.mem.Allocator,
    gs: *const GroundState,
    q_cart: math.Vec3,
    vloc1_g: []const math.Complex,
    rho1_g: []const math.Complex,
    rho1_core_r: []const math.Complex,
) ![]math.Complex {
    const vh1_g = try perturbation.build_hartree_perturbation_q(alloc, gs.grid, rho1_g, q_cart);
    defer alloc.free(vh1_g);

    const rho1_val_r = try reciprocal_to_complex_copy(alloc, gs.grid, rho1_g);
    defer alloc.free(rho1_val_r);

    const rho1_total_r = try alloc.alloc(math.Complex, gs.grid.count());
    defer alloc.free(rho1_total_r);

    for (0..gs.grid.count()) |i| {
        rho1_total_r[i] = math.complex.add(rho1_val_r[i], rho1_core_r[i]);
    }

    const vxc1_r = try perturbation.build_xc_perturbation_full_complex(alloc, gs.*, rho1_total_r);
    defer alloc.free(vxc1_r);

    const vxc1_r_fft = try alloc.alloc(math.Complex, gs.grid.count());
    defer alloc.free(vxc1_r_fft);

    @memcpy(vxc1_r_fft, vxc1_r);

    const vxc1_g = try alloc.alloc(math.Complex, gs.grid.count());
    defer alloc.free(vxc1_g);

    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, gs.grid, vxc1_r_fft, vxc1_g, null);

    const v_out_g = try alloc.alloc(math.Complex, gs.grid.count());
    for (0..gs.grid.count()) |i| {
        v_out_g[i] = math.complex.add(
            math.complex.add(vloc1_g[i], vh1_g[i]),
            vxc1_g[i],
        );
    }
    return v_out_g;
}

fn mix_single_q_potential(
    alloc: std.mem.Allocator,
    cfg: DfptConfig,
    iter: usize,
    v_scf_g: []math.Complex,
    v_out_g: []const math.Complex,
    mix_state: *SingleQMixState,
) !bool {
    const residual = try alloc.alloc(math.Complex, v_out_g.len);
    var residual_norm: f64 = 0.0;
    for (0..v_out_g.len) |i| {
        residual[i] = math.complex.sub(v_out_g[i], v_scf_g[i]);
        residual_norm += residual[i].r * residual[i].r + residual[i].i * residual[i].i;
    }
    residual_norm = @sqrt(residual_norm);

    log_dfpt("dfptQ_scf: iter={d} vresid={e:.6}\n", .{ iter, residual_norm });
    const converged_tight = residual_norm < cfg.scf_tol;
    const converged_forced = mix_state.force_converge and residual_norm < 10.0 * cfg.scf_tol;
    if (converged_tight or converged_forced) {
        alloc.free(residual);
        log_dfpt("dfptQ_scf: converged at iter={d} vresid={e:.6}\n", .{ iter, residual_norm });
        return true;
    }

    if (residual_norm < mix_state.best_vresid) {
        mix_state.best_vresid = residual_norm;
        if (mix_state.best_v_scf == null) {
            mix_state.best_v_scf = try alloc.alloc(math.Complex, v_scf_g.len);
        }
        @memcpy(mix_state.best_v_scf.?, v_scf_g);
    }

    const restart_factor: f64 = 5.0;
    const pulay_ready = iter >= mix_state.pulay_active_since;
    const pulay_diverged = residual_norm > restart_factor * mix_state.best_vresid;
    if (pulay_ready and pulay_diverged and mix_state.best_vresid < 1.0) {
        if (mix_state.best_v_scf) |v| @memcpy(v_scf_g, v);
        if (mix_state.best_vresid < 10.0 * cfg.scf_tol) {
            mix_state.force_converge = true;
            log_dfpt(
                "dfptQ_scf: Pulay restart (near-converged)" ++
                    " at iter={d} vresid={e:.6} best={e:.6}\n",
                .{ iter, residual_norm, mix_state.best_vresid },
            );
            alloc.free(residual);
            return false;
        }
        mix_state.pulay.reset();
        mix_state.pulay_active_since = iter + 1 + cfg.pulay_start;
        log_dfpt(
            "dfptQ_scf: Pulay restart at iter={d} vresid={e:.6} best={e:.6}\n",
            .{ iter, residual_norm, mix_state.best_vresid },
        );
        alloc.free(residual);
        return false;
    }

    if (cfg.pulay_history > 0 and iter >= mix_state.pulay_active_since) {
        try mix_state.pulay.mix_with_residual(v_scf_g, residual, cfg.mixing_beta);
        return false;
    }

    for (0..v_out_g.len) |i| {
        v_scf_g[i] = math.complex.add(
            v_scf_g[i],
            math.complex.scale(residual[i], cfg.mixing_beta),
        );
    }
    alloc.free(residual);
    return false;
}

fn run_single_q_scf(
    alloc: std.mem.Allocator,
    gs: *const GroundState,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
    q_cart: math.Vec3,
    gvecs_kq: []const plane_wave.GVector,
    apply_ctx_kq: *scf_mod.ApplyContext,
    map_kq: *const scf_mod.PwGridMap,
    occ_kq: []const []const math.Complex,
    n_occ_kq: usize,
    setup: *SingleQSetup,
    mix_state: *SingleQMixState,
) !void {
    var iter: usize = 0;
    while (iter < cfg.scf_max_iter) : (iter += 1) {
        const v_scf_r = try reciprocal_to_complex_copy(alloc, gs.grid, setup.v_scf_g);
        defer alloc.free(v_scf_r);

        if (iter == 0) {
            log_scf_potential_sample(
                v_scf_r,
                setup.v_scf_g,
                atom_index,
                direction,
                gs.grid.count(),
            );
        }

        try solve_single_q_bands(
            alloc,
            gs,
            atom_index,
            direction,
            cfg,
            gvecs_kq,
            apply_ctx_kq,
            map_kq,
            occ_kq,
            n_occ_kq,
            v_scf_r,
            setup.psi0_r_cache,
            setup.psi1,
        );

        const rho1_g = try build_single_q_rho1_g(
            alloc,
            gs,
            map_kq,
            setup.psi0_r_const,
            setup.psi1,
        );
        defer alloc.free(rho1_g);

        log_single_q_rho_diagnostic(iter, rho1_g, setup.vloc1_g, gs.grid.volume);

        const v_out_g = try build_single_q_output_potential(
            alloc,
            gs,
            q_cart,
            setup.vloc1_g,
            rho1_g,
            setup.rho1_core_r,
        );
        defer alloc.free(v_out_g);

        if (try mix_single_q_potential(
            alloc,
            cfg,
            iter,
            setup.v_scf_g,
            v_out_g,
            mix_state,
        )) break;
    }
}

fn finalize_single_q_result(
    alloc: std.mem.Allocator,
    gs: *const GroundState,
    map_kq: *const scf_mod.PwGridMap,
    gvecs_kq: []const plane_wave.GVector,
    setup: *SingleQSetup,
) !PerturbationResult {
    const final_rho1_r = try compute_rho1_q(
        alloc,
        gs.grid,
        &gs.apply_ctx.map,
        map_kq,
        gs.wavefunctions,
        setup.psi1,
        gs.n_occ,
        gs.gvecs.len,
        gvecs_kq.len,
        1.0,
    );
    defer alloc.free(final_rho1_r);

    const final_rho1_g = try complex_real_to_reciprocal(alloc, gs.grid, final_rho1_r);
    const psi1 = setup.psi1;
    setup.psi1 = &[_][]math.Complex{};
    return .{
        .rho1_g = final_rho1_g,
        .psi1 = psi1,
    };
}

/// Solve DFPT perturbation at q≠0 for a single perturbation (atom, direction).
/// Uses **potential mixing** (mix V^(1)_SCF, not ρ^(1)) for stable convergence.
/// Density mixing is unstable at finite q because V_H(G=0) = 8π/|q|² diverges.
pub fn solve_perturbation_q(
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
    var setup = try init_single_q_setup(alloc, &gs, atom_index, direction, q_cart, gvecs_kq.len);
    defer setup.deinit(alloc);

    var mix_state = SingleQMixState.init(alloc, cfg);
    defer mix_state.deinit(alloc);

    try run_single_q_scf(
        alloc,
        &gs,
        atom_index,
        direction,
        cfg,
        q_cart,
        gvecs_kq,
        apply_ctx_kq,
        map_kq,
        occ_kq,
        n_occ_kq,
        &setup,
        &mix_state,
    );
    return finalize_single_q_result(alloc, &gs, map_kq, gvecs_kq, &setup);
}
