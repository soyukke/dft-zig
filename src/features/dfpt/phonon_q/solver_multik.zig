//! Multi-k-point q≠0 DFPT SCF solver.
//!
//! For each k-point in the full BZ, solves Sternheimer and accumulates
//! ρ^(1) with the k-point weight, then mixes V^(1)_SCF until self-consistency.
//!
//! Supports k-point threading via a worker pool driven by an atomic counter;
//! each worker owns a scratch arena and a thread-local ρ^(1) buffer, which
//! are summed into the shared ρ^(1) after the per-iteration barrier.

const std = @import("std");
const math = @import("../../math/math.zig");
const scf_mod = @import("../../scf/scf.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const form_factor = @import("../../pseudopotential/form_factor.zig");

const dfpt = @import("../dfpt.zig");
const perturbation = dfpt.perturbation;
const sternheimer = dfpt.sternheimer;
const GroundState = dfpt.GroundState;
const DfptConfig = dfpt.DfptConfig;
const logDfpt = dfpt.logDfpt;
const logDfptInfo = dfpt.logDfptInfo;

const kpt_dfpt = @import("kpt_dfpt.zig");
const KPointDfptData = kpt_dfpt.KPointDfptData;
const MultiKPertResult = kpt_dfpt.MultiKPertResult;

const cross_basis = @import("cross_basis.zig");
const applyV1PsiQCached = cross_basis.applyV1PsiQCached;
const computeRho1Q = cross_basis.computeRho1Q;
const computeRho1QCached = cross_basis.computeRho1QCached;
const complexRealToReciprocal = cross_basis.complexRealToReciprocal;

const dynmat_elem_q = @import("dynmat_elem_q.zig");
const computeElecDynmatElementQ = dynmat_elem_q.computeElecDynmatElementQ;

const Grid = scf_mod.Grid;

/// Shared data for parallel DFPT k-point processing within one SCF iteration.
const DfptKpointShared = struct {
    io: std.Io,
    kpts: []KPointDfptData,
    v_scf_r: []const math.Complex,
    vloc1_g: []const math.Complex,
    cfg: DfptConfig,
    atom_index: usize,
    direction: usize,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    /// Per-k-point ψ^(1) storage [n_kpts][n_occ][n_pw_kq] — each worker writes to its own ik
    psi1_per_k: [][][]math.Complex,
    /// Thread-local ρ^(1) buffers: rho1_locals[thread_index * total .. (thread_index+1) * total]
    rho1_locals: []math.Complex,
    total: usize,
    /// Pre-computed ψ^(0)_k(r) cache: [n_kpts][n_occ][total]
    psi0_r_cache: []const []const []const math.Complex,

    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

const DfptKpointWorker = struct {
    shared: *DfptKpointShared,
    thread_index: usize,
};

fn setDfptWorkerError(shared: *DfptKpointShared, e: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);
    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

fn dfptKpointWorkerFn(worker: *DfptKpointWorker) void {
    const shared = worker.shared;
    const thread_index = worker.thread_index;
    const total = shared.total;
    const start = thread_index * total;
    const rho1_local = shared.rho1_locals[start .. start + total];

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const ik = shared.next_index.fetchAdd(1, .acq_rel);
        if (ik >= shared.kpts.len) break;

        _ = arena.reset(.retain_capacity);
        const kalloc = arena.allocator();

        processOneKpointDfpt(
            kalloc,
            shared,
            ik,
            rho1_local,
        ) catch |e| {
            setDfptWorkerError(shared, e);
            shared.stop.store(1, .release);
            break;
        };
    }
}

/// Process a single k-point within one DFPT SCF iteration.
/// Solves Sternheimer for each occupied band, updates psi1_per_k[ik],
/// and accumulates ρ^(1) into the provided rho1_local buffer.
fn processOneKpointDfpt(
    alloc: std.mem.Allocator,
    shared: *DfptKpointShared,
    ik: usize,
    rho1_local: []math.Complex,
) !void {
    const kd = &shared.kpts[ik];
    const n_occ = kd.n_occ;
    const n_pw_kq = kd.n_pw_kq;
    const map_kq_ptr: *const scf_mod.PwGridMap = &kd.map_kq;
    const total = shared.total;

    // Nonlocal contexts
    const nl_ctx_k_opt = kd.apply_ctx_k.nonlocal_ctx;
    const nl_ctx_kq_opt = kd.apply_ctx_kq.nonlocal_ctx;

    // ψ^(0)(r) cache for this k-point
    const psi0_r_k = shared.psi0_r_cache[ik];

    // Solve Sternheimer for each occupied band at this k-point
    for (0..n_occ) |n| {
        // RHS: -P_c^{k+q} × H^(1)|ψ^(0)_{n,k}⟩
        const rhs = try applyV1PsiQCached(alloc, shared.grid, map_kq_ptr, shared.v_scf_r, psi0_r_k[n], n_pw_kq);
        defer alloc.free(rhs);

        // Add nonlocal perturbation
        if (nl_ctx_k_opt != null and nl_ctx_kq_opt != null) {
            const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
            defer alloc.free(nl_out);
            try perturbation.applyNonlocalPerturbationQ(
                alloc,
                kd.basis_k.gvecs,
                kd.basis_kq.gvecs,
                shared.atoms,
                nl_ctx_k_opt.?,
                nl_ctx_kq_opt.?,
                shared.atom_index,
                shared.direction,
                1.0 / shared.grid.volume,
                kd.wavefunctions_k_const[n],
                nl_out,
            );
            for (0..n_pw_kq) |g| {
                rhs[g] = math.complex.add(rhs[g], nl_out[g]);
            }
        }

        // Negate
        for (0..n_pw_kq) |g| {
            rhs[g] = math.complex.scale(rhs[g], -1.0);
        }

        // Project onto conduction band in k+q space
        sternheimer.projectConduction(rhs, kd.occ_kq_const, kd.n_occ_kq);

        // Solve Sternheimer
        const result = try sternheimer.solve(
            alloc,
            kd.apply_ctx_kq,
            rhs,
            kd.eigenvalues_k[n],
            kd.occ_kq_const,
            kd.n_occ_kq,
            kd.basis_kq.gvecs,
            .{
                .tol = shared.cfg.sternheimer_tol,
                .max_iter = shared.cfg.sternheimer_max_iter,
                .alpha_shift = shared.cfg.alpha_shift,
            },
        );

        // Copy result to psi1_per_k (fixed-size buffer, no reallocation needed)
        @memcpy(shared.psi1_per_k[ik][n], result.psi1);
        alloc.free(result.psi1);
    }

    // Compute ρ^(1) for this k-point (weighted by wtk)
    const psi1_const = try alloc.alloc([]const math.Complex, n_occ);
    defer alloc.free(psi1_const);
    for (0..n_occ) |n| psi1_const[n] = shared.psi1_per_k[ik][n];

    const rho1_k_r = try computeRho1QCached(
        alloc,
        shared.grid,
        map_kq_ptr,
        psi0_r_k,
        psi1_const,
        n_occ,
        kd.weight,
    );
    defer alloc.free(rho1_k_r);

    // Accumulate into thread-local buffer
    for (0..total) |i| {
        rho1_local[i] = math.complex.add(rho1_local[i], rho1_k_r[i]);
    }
}

/// Solve DFPT perturbation at q≠0 with multiple k-points.
/// For each k-point, solves Sternheimer equations and accumulates ρ^(1)
/// with the k-point weight. Uses potential mixing for stable convergence.
///
/// Returns the converged ρ^(1)(G) (summed over all k-points) and
/// per-k-point ψ^(1) wavefunctions.
pub fn solvePerturbationQMultiK(
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
    q_cart: math.Vec3,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) !MultiKPertResult {
    const total = grid.count();
    const n_kpts = kpts.len;

    // Build V_ext^(1)_q(G) for this perturbation (bare, fixed) — k-independent
    const vloc1_g = try perturbation.buildLocalPerturbationQ(
        alloc,
        grid,
        atoms[atom_index],
        species,
        direction,
        q_cart,
        gs.local_cfg,
        ff_tables,
    );
    defer alloc.free(vloc1_g);

    // Build ρ^(1)_core,q(G) (fixed, NLCC) — k-independent
    const rho1_core_g = try perturbation.buildCorePerturbationQ(
        alloc,
        grid,
        atoms[atom_index],
        species,
        direction,
        q_cart,
        rho_core_tables,
    );
    defer alloc.free(rho1_core_g);

    // IFFT ρ^(1)_core,q → real space (complex)
    const rho1_core_g_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_g_copy);
    @memcpy(rho1_core_g_copy, rho1_core_g);
    const rho1_core_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_r);
    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_g_copy, rho1_core_r, null);

    // Allocate per-k-point first-order wavefunctions
    var psi1_per_k = try alloc.alloc([][]math.Complex, n_kpts);
    for (0..n_kpts) |ik| {
        const n_occ = kpts[ik].n_occ;
        const n_pw_kq = kpts[ik].n_pw_kq;
        psi1_per_k[ik] = try alloc.alloc([]math.Complex, n_occ);
        for (0..n_occ) |n| {
            psi1_per_k[ik][n] = try alloc.alloc(math.Complex, n_pw_kq);
            @memset(psi1_per_k[ik][n], math.complex.init(0.0, 0.0));
        }
    }

    // Initialize V_SCF(G) = V_loc^(1)(G) [bare perturbation, no screening]
    var v_scf_g = try alloc.alloc(math.Complex, total);
    @memcpy(v_scf_g, vloc1_g);

    // Pre-compute ψ^(0)_k(r) for all k-points (invariant during SCF)
    const psi0_r_cache = try alloc.alloc([][]math.Complex, n_kpts);
    defer {
        for (psi0_r_cache) |kc| {
            for (kc) |band| alloc.free(band);
            alloc.free(kc);
        }
        alloc.free(psi0_r_cache);
    }
    for (kpts, 0..) |*kd, ik| {
        psi0_r_cache[ik] = try alloc.alloc([]math.Complex, kd.n_occ);
        for (0..kd.n_occ) |n| {
            psi0_r_cache[ik][n] = try alloc.alloc(math.Complex, total);
            const work = try alloc.alloc(math.Complex, total);
            defer alloc.free(work);
            @memset(work, math.complex.init(0.0, 0.0));
            kd.map_k.scatter(kd.wavefunctions_k_const[n], work);
            try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work, psi0_r_cache[ik][n], null);
        }
    }

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

    // DFPT SCF loop — potential mixing, multi-k
    var iter: usize = 0;
    while (iter < cfg.scf_max_iter) : (iter += 1) {
        // IFFT V_SCF(G) → V_SCF(r) [complex]
        const v_scf_g_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_scf_g_copy);
        @memcpy(v_scf_g_copy, v_scf_g);
        const v_scf_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_scf_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, v_scf_g_copy, v_scf_r, null);

        // Accumulate ρ^(1) over all k-points (parallel or sequential)
        const rho1_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_r);
        @memset(rho1_r, math.complex.init(0.0, 0.0));

        const thread_count = scf_mod.kpointThreadCount(n_kpts, cfg.kpoint_threads);

        if (thread_count <= 1) {
            // Sequential path — use cached ψ^(0)(r)
            for (kpts, 0..) |*kd, ik| {
                const n_occ = kd.n_occ;
                const n_pw_kq = kd.n_pw_kq;
                const map_kq_ptr: *const scf_mod.PwGridMap = &kd.map_kq;

                // Nonlocal contexts
                const nl_ctx_k_opt = kd.apply_ctx_k.nonlocal_ctx;
                const nl_ctx_kq_opt = kd.apply_ctx_kq.nonlocal_ctx;

                // Solve Sternheimer for each occupied band at this k-point
                for (0..n_occ) |n| {
                    // RHS: -P_c^{k+q} × H^(1)|ψ^(0)_{n,k}⟩
                    const rhs = try applyV1PsiQCached(alloc, grid, map_kq_ptr, v_scf_r, psi0_r_cache[ik][n], n_pw_kq);
                    defer alloc.free(rhs);

                    // Add nonlocal perturbation
                    if (nl_ctx_k_opt != null and nl_ctx_kq_opt != null) {
                        const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
                        defer alloc.free(nl_out);
                        try perturbation.applyNonlocalPerturbationQ(
                            alloc,
                            kd.basis_k.gvecs,
                            kd.basis_kq.gvecs,
                            atoms,
                            nl_ctx_k_opt.?,
                            nl_ctx_kq_opt.?,
                            atom_index,
                            direction,
                            1.0 / grid.volume,
                            kd.wavefunctions_k_const[n],
                            nl_out,
                        );
                        for (0..n_pw_kq) |g| {
                            rhs[g] = math.complex.add(rhs[g], nl_out[g]);
                        }
                    }

                    // Negate
                    for (0..n_pw_kq) |g| {
                        rhs[g] = math.complex.scale(rhs[g], -1.0);
                    }

                    // Project onto conduction band in k+q space
                    sternheimer.projectConduction(rhs, kd.occ_kq_const, kd.n_occ_kq);

                    // Solve Sternheimer
                    const result = try sternheimer.solve(
                        alloc,
                        kd.apply_ctx_kq,
                        rhs,
                        kd.eigenvalues_k[n],
                        kd.occ_kq_const,
                        kd.n_occ_kq,
                        kd.basis_kq.gvecs,
                        .{
                            .tol = cfg.sternheimer_tol,
                            .max_iter = cfg.sternheimer_max_iter,
                            .alpha_shift = cfg.alpha_shift,
                        },
                    );

                    @memcpy(psi1_per_k[ik][n], result.psi1);
                    alloc.free(result.psi1);
                }

                // Compute ρ^(1) for this k-point (weighted by wtk)
                const psi1_const = try alloc.alloc([]const math.Complex, n_occ);
                defer alloc.free(psi1_const);
                for (0..n_occ) |n| psi1_const[n] = psi1_per_k[ik][n];

                const psi0_r_k_const: []const []const math.Complex = @ptrCast(psi0_r_cache[ik]);
                const rho1_k_r = try computeRho1QCached(
                    alloc,
                    grid,
                    map_kq_ptr,
                    psi0_r_k_const,
                    psi1_const,
                    n_occ,
                    kd.weight,
                );
                defer alloc.free(rho1_k_r);

                // Accumulate
                for (0..total) |i| {
                    rho1_r[i] = math.complex.add(rho1_r[i], rho1_k_r[i]);
                }
            }
        } else {
            // Parallel path — spawn worker threads
            if (iter == 0) {
                logDfptInfo("dfptQ_mk: using {d} threads for {d} k-points\n", .{ thread_count, n_kpts });
            }

            // Allocate thread-local ρ^(1) buffers
            const rho1_locals = try alloc.alloc(math.Complex, total * thread_count);
            defer alloc.free(rho1_locals);
            @memset(rho1_locals, math.complex.init(0.0, 0.0));

            // Synchronization primitives
            var next_index = std.atomic.Value(usize).init(0);
            var stop = std.atomic.Value(u8).init(0);
            var worker_err: ?anyerror = null;
            var err_mutex = std.Io.Mutex.init;
            var log_mutex = std.Io.Mutex.init;

            // Build const view of psi0_r_cache for parallel workers
            const psi0_r_const_view = try alloc.alloc([]const []const math.Complex, n_kpts);
            defer alloc.free(psi0_r_const_view);
            for (0..n_kpts) |ik2| {
                psi0_r_const_view[ik2] = @ptrCast(psi0_r_cache[ik2]);
            }

            var shared = DfptKpointShared{
                .io = io,
                .kpts = kpts,
                .v_scf_r = v_scf_r,
                .vloc1_g = vloc1_g,
                .cfg = cfg,
                .atom_index = atom_index,
                .direction = direction,
                .grid = grid,
                .species = species,
                .atoms = atoms,
                .psi1_per_k = psi1_per_k,
                .rho1_locals = rho1_locals,
                .total = total,
                .psi0_r_cache = psi0_r_const_view,
                .next_index = &next_index,
                .stop = &stop,
                .err = &worker_err,
                .err_mutex = &err_mutex,
                .log_mutex = &log_mutex,
            };

            // Create workers (thread 0 runs on main thread)
            var workers = try alloc.alloc(DfptKpointWorker, thread_count);
            defer alloc.free(workers);
            var threads = try alloc.alloc(std.Thread, thread_count - 1);
            defer alloc.free(threads);

            for (0..thread_count) |ti| {
                workers[ti] = .{ .shared = &shared, .thread_index = ti };
            }

            // Spawn worker threads (skip thread 0, it runs on main)
            for (0..thread_count - 1) |ti| {
                threads[ti] = try std.Thread.spawn(.{}, dfptKpointWorkerFn, .{&workers[ti + 1]});
            }

            // Run thread 0 on main thread
            dfptKpointWorkerFn(&workers[0]);

            // Join all threads
            for (threads) |t| {
                t.join();
            }

            // Check for errors
            if (worker_err) |e| return e;

            // Sum thread-local ρ^(1) into rho1_r
            for (0..thread_count) |ti| {
                const local_start = ti * total;
                const local = rho1_locals[local_start .. local_start + total];
                for (0..total) |i| {
                    rho1_r[i] = math.complex.add(rho1_r[i], local[i]);
                }
            }
        }

        // FFT ρ^(1)(r) → ρ^(1)(G)
        const rho1_g = try complexRealToReciprocal(alloc, grid, rho1_r);
        defer alloc.free(rho1_g);

        // Diagnostic
        {
            var rho_norm: f64 = 0.0;
            for (0..total) |di| {
                rho_norm += rho1_g[di].r * rho1_g[di].r + rho1_g[di].i * rho1_g[di].i;
            }
            rho_norm = @sqrt(rho_norm);
            const d_elec_diag = computeElecDynmatElementQ(vloc1_g, rho1_g, grid.volume);
            logDfpt("dfptQ_mk: iter={d} |rho1|={e:.6} D_elec_bare=({e:.6},{e:.6}) nk={d}\n", .{ iter, rho_norm, d_elec_diag.r, d_elec_diag.i, n_kpts });
        }

        // Build V_out(G) = V_loc^(1) + V_H^(1)[ρ] + V_xc^(1)[ρ]
        const vh1_g = try perturbation.buildHartreePerturbationQ(alloc, grid, rho1_g, q_cart);
        defer alloc.free(vh1_g);

        // V_xc^(1)(r) = f_xc(r) × ρ^(1)_total(r)
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

        // Compute residual
        var residual_norm: f64 = 0.0;
        const residual = try alloc.alloc(math.Complex, total);
        for (0..total) |i| {
            residual[i] = math.complex.sub(v_out_g[i], v_scf_g[i]);
            residual_norm += residual[i].r * residual[i].r + residual[i].i * residual[i].i;
        }
        residual_norm = @sqrt(residual_norm);

        logDfpt("dfptQ_mk: iter={d} vresid={e:.6}\n", .{ iter, residual_norm });

        if (residual_norm < cfg.scf_tol or (force_converge and residual_norm < 10.0 * cfg.scf_tol)) {
            alloc.free(residual);
            logDfpt("dfptQ_mk: converged at iter={d} vresid={e:.6}\n", .{ iter, residual_norm });
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
                logDfpt("dfptQ_mk: Pulay restart (near-converged) at iter={d} vresid={e:.6} best={e:.6}\n", .{ iter, residual_norm, best_vresid });
                alloc.free(residual);
                continue;
            }
            pulay.reset();
            pulay_active_since = iter + 1 + cfg.pulay_start;
            logDfpt("dfptQ_mk: Pulay restart at iter={d} vresid={e:.6} best={e:.6}\n", .{ iter, residual_norm, best_vresid });
            alloc.free(residual);
            continue;
        }

        // Mix
        if (cfg.pulay_history > 0 and iter >= pulay_active_since) {
            try pulay.mixWithResidual(v_scf_g, residual, cfg.mixing_beta);
        } else {
            const beta = cfg.mixing_beta;
            for (0..total) |i| {
                v_scf_g[i] = math.complex.add(v_scf_g[i], math.complex.scale(residual[i], beta));
            }
            alloc.free(residual);
        }
    }

    // Compute final ρ^(1)(G) from converged ψ^(1) (sum over all k-points)
    const final_rho1_r = try alloc.alloc(math.Complex, total);
    @memset(final_rho1_r, math.complex.init(0.0, 0.0));

    for (kpts, 0..) |*kd, ik| {
        const psi1_const = try alloc.alloc([]const math.Complex, kd.n_occ);
        defer alloc.free(psi1_const);
        for (0..kd.n_occ) |n| psi1_const[n] = psi1_per_k[ik][n];

        const rho1_k_r = try computeRho1Q(
            alloc,
            grid,
            &kd.map_k,
            &kd.map_kq,
            kd.wavefunctions_k_const,
            psi1_const,
            kd.n_occ,
            kd.n_pw_k,
            kd.n_pw_kq,
            kd.weight,
        );
        defer alloc.free(rho1_k_r);

        for (0..total) |i| {
            final_rho1_r[i] = math.complex.add(final_rho1_r[i], rho1_k_r[i]);
        }
    }

    const final_rho1_g = try complexRealToReciprocal(alloc, grid, final_rho1_r);
    alloc.free(final_rho1_r);

    alloc.free(v_scf_g);

    // Build result with const views
    const psi1_result = try alloc.alloc([]const []math.Complex, n_kpts);
    for (0..n_kpts) |ik| {
        const psi1_const = try alloc.alloc([]math.Complex, kpts[ik].n_occ);
        for (0..kpts[ik].n_occ) |n| psi1_const[n] = psi1_per_k[ik][n];
        psi1_result[ik] = psi1_const;
    }
    // Free the per-k inner slices (ownership transferred to psi1_result above)
    for (0..n_kpts) |ik| {
        alloc.free(psi1_per_k[ik]);
    }
    alloc.free(psi1_per_k);

    return .{
        .rho1_g = final_rho1_g,
        .psi1_per_k = psi1_result,
    };
}
