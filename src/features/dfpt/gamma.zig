//! Γ-point DFPT phonon calculation.
//!
//! Contains the q=0 perturbation solver, dynmat construction,
//! diagnostics, and the top-level `runPhonon` entry point.

const std = @import("std");
const math = @import("../math/math.zig");
const xc = @import("../xc/xc.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../scf/scf.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const d3 = @import("../vdw/d3.zig");
const d3_params = @import("../vdw/d3_params.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const config_mod = @import("../config/config.zig");

const symmetry_mod = @import("../symmetry/symmetry.zig");

const dfpt = @import("dfpt.zig");
const perturbation = dfpt.perturbation;
const sternheimer = dfpt.sternheimer;
const ewald2 = dfpt.ewald2;
const dynmat_mod = dfpt.dynmat;
const dynmat_contrib = dfpt.dynmat_contrib;

const GroundState = dfpt.GroundState;
const PreparedGroundState = dfpt.PreparedGroundState;
const DfptConfig = dfpt.DfptConfig;
const IonicData = dfpt.IonicData;
const PerturbationResult = dfpt.PerturbationResult;
const logDfpt = dfpt.logDfpt;

const Grid = scf_mod.Grid;

/// Result of a full phonon calculation.
pub const PhononResult = struct {
    /// Phonon frequencies in cm⁻¹
    frequencies_cm1: []f64,
    /// Eigenvalues ω² in Ry/(bohr²·amu)
    omega2: []f64,
    /// Eigenvectors (column-major, dim×dim)
    eigenvectors: []f64,
    /// Dimension (3 × n_atoms)
    dim: usize,

    pub fn deinit(self: *PhononResult, alloc: std.mem.Allocator) void {
        if (self.frequencies_cm1.len > 0) alloc.free(self.frequencies_cm1);
        if (self.omega2.len > 0) alloc.free(self.omega2);
        if (self.eigenvectors.len > 0) alloc.free(self.eigenvectors);
    }
};

/// Run DFPT phonon calculation at the Γ-point.
/// Takes converged SCF results and returns phonon frequencies.
pub fn runPhonon(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
) !PhononResult {
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const grid = scf_result.grid;

    logDfpt("dfpt: starting phonon calculation ({d} atoms, dim={d})\n", .{ n_atoms, dim });

    // Prepare ground state (PW basis, eigenvalues, wavefunctions, NLCC, etc.)
    var prepared = try dfpt.prepareGroundState(alloc, io, cfg, scf_result, species, atoms, volume, recip);
    defer prepared.deinit();
    const gs = prepared.gs;

    // DFPT config
    const dfpt_cfg = DfptConfig.fromConfig(cfg);
    const pert_thread_count = dfpt.perturbationThreadCount(dim, dfpt_cfg.perturbation_threads);

    // Build symmetry operations and find irreducible atoms
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    defer alloc.free(symops);
    const sym_data = try dynmat_mod.buildIndsym(alloc, symops, atoms, recip, 1e-5);
    defer {
        for (sym_data.indsym) |row| alloc.free(row);
        alloc.free(sym_data.indsym);
        for (sym_data.tnons_shift) |row| alloc.free(row);
        alloc.free(sym_data.tnons_shift);
    }
    const q_zero = math.Vec3{ .x = 0, .y = 0, .z = 0 };
    var irr_info = try dynmat_mod.findIrreducibleAtoms(alloc, symops, sym_data.indsym, n_atoms, q_zero);
    defer irr_info.deinit(alloc);
    logDfpt("dfpt: {d} symops, {d}/{d} irreducible atoms\n", .{ symops.len, irr_info.n_irr_atoms, n_atoms });

    // Solve perturbations for each atom and direction
    // Store V_loc^(1)(G), ρ^(1)(G), and ρ^(1)_core(G) for dynmat construction
    var pert_results = try alloc.alloc(PerturbationResult, dim);
    var vloc1_gs = try alloc.alloc([]math.Complex, dim);
    var rho1_core_gs = try alloc.alloc([]math.Complex, dim);
    var pert_count: usize = 0;
    var vloc1_count: usize = 0;
    var rho1_core_count: usize = 0;
    defer {
        for (0..pert_count) |i| pert_results[i].deinit(alloc);
        alloc.free(pert_results);
        for (0..vloc1_count) |i| alloc.free(vloc1_gs[i]);
        alloc.free(vloc1_gs);
        for (0..rho1_core_count) |i| alloc.free(rho1_core_gs[i]);
        alloc.free(rho1_core_gs);
    }

    if (pert_thread_count <= 1) {
        // Phase 1: Build vloc1, rho1_core for ALL perturbations (cheap, analytic)
        for (0..n_atoms) |ia| {
            for (0..3) |dir| {
                const idx = 3 * ia + dir;
                vloc1_gs[idx] = try perturbation.buildLocalPerturbation(
                    alloc,
                    gs.grid,
                    gs.atoms[ia],
                    gs.species,
                    dir,
                    gs.local_cfg,
                    gs.ff_tables,
                );
                vloc1_count = idx + 1;

                rho1_core_gs[idx] = try perturbation.buildCorePerturbation(
                    alloc,
                    gs.grid,
                    gs.atoms[ia],
                    gs.species,
                    dir,
                    gs.rho_core_tables,
                );
                rho1_core_count = idx + 1;
            }
        }

        // Phase 2: Solve perturbation SCF for irreducible atoms only (expensive)
        // Initialize all pert_results to empty defaults first
        for (0..dim) |i| {
            pert_results[i] = .{ .rho1_g = &.{}, .psi1 = &.{} };
        }
        pert_count = dim;

        for (irr_info.irr_atom_indices) |ia| {
            for (0..3) |dir| {
                const idx = 3 * ia + dir;
                const dir_names = [_][]const u8{ "x", "y", "z" };
                logDfpt("dfpt: solving perturbation atom={d} dir={s} (irreducible)\n", .{ ia, dir_names[dir] });

                pert_results[idx] = try solvePerturbation(alloc, gs, ia, dir, dfpt_cfg);
            }
        }
    } else {
        // Parallel path — solve only irreducible perturbations concurrently
        const n_irr_perts = irr_info.n_irr_atoms * 3;
        logDfpt("dfpt: using {d} threads for {d} perturbations ({d} irreducible)\n", .{ pert_thread_count, dim, n_irr_perts });

        // Build irr_pert_indices: list of perturbation indices to solve
        const irr_pert_indices = try alloc.alloc(usize, n_irr_perts);
        defer alloc.free(irr_pert_indices);
        {
            var pi: usize = 0;
            for (irr_info.irr_atom_indices) |ia| {
                for (0..3) |dir| {
                    irr_pert_indices[pi] = 3 * ia + dir;
                    pi += 1;
                }
            }
        }

        // Initialize output arrays to safe defaults for cleanup
        for (0..dim) |i| {
            pert_results[i] = .{ .rho1_g = &.{}, .psi1 = &.{} };
            vloc1_gs[i] = &.{};
            rho1_core_gs[i] = &.{};
        }
        pert_count = dim;
        vloc1_count = dim;
        rho1_core_count = dim;

        // Build vloc1 and rho1_core for ALL perturbations (cheap, serial)
        for (0..n_atoms) |ia| {
            for (0..3) |dir| {
                const idx = 3 * ia + dir;
                vloc1_gs[idx] = try perturbation.buildLocalPerturbation(
                    alloc,
                    gs.grid,
                    gs.atoms[ia],
                    gs.species,
                    dir,
                    gs.local_cfg,
                    gs.ff_tables,
                );
                rho1_core_gs[idx] = try perturbation.buildCorePerturbation(
                    alloc,
                    gs.grid,
                    gs.atoms[ia],
                    gs.species,
                    dir,
                    gs.rho_core_tables,
                );
            }
        }

        var next_index = std.atomic.Value(usize).init(0);
        var stop = std.atomic.Value(u8).init(0);
        var worker_err: ?anyerror = null;
        var err_mutex = std.Io.Mutex.init;
        var log_mutex = std.Io.Mutex.init;

        var shared = GammaPertShared{
            .alloc = alloc,
            .io = io,
            .gs = &gs,
            .dfpt_cfg = &dfpt_cfg,
            .pert_results = pert_results,
            .vloc1_gs = vloc1_gs,
            .rho1_core_gs = rho1_core_gs,
            .dim = n_irr_perts,
            .irr_pert_indices = irr_pert_indices,
            .next_index = &next_index,
            .stop = &stop,
            .err = &worker_err,
            .err_mutex = &err_mutex,
            .log_mutex = &log_mutex,
        };

        var workers = try alloc.alloc(GammaPertWorker, pert_thread_count);
        defer alloc.free(workers);
        var threads = try alloc.alloc(std.Thread, pert_thread_count - 1);
        defer alloc.free(threads);

        for (0..pert_thread_count) |ti| {
            workers[ti] = .{ .shared = &shared, .thread_index = ti };
        }

        for (0..pert_thread_count - 1) |ti| {
            threads[ti] = try std.Thread.spawn(.{}, gammaPertWorkerFn, .{&workers[ti + 1]});
        }

        // Run thread 0 on main thread
        gammaPertWorkerFn(&workers[0]);

        for (threads) |t| {
            t.join();
        }

        if (worker_err) |e| return e;
    }

    // Run diagnostic tests (only when all perturbations are available)
    if (irr_info.n_irr_atoms == n_atoms) {
        try runDiagnostics(alloc, gs, pert_results, vloc1_gs, n_atoms);
    } else {
        logDfpt("dfpt: skipping diagnostics (symmetry-reduced: {d}/{d} irreducible atoms)\n", .{ irr_info.n_irr_atoms, n_atoms });
    }

    // Build ionic data and rho0_g for dynmat construction
    const ionic = try IonicData.init(alloc, species, atoms);
    defer ionic.deinit(alloc);

    const rho0_g = try scf_mod.realToReciprocal(alloc, grid, scf_result.density, false);
    defer alloc.free(rho0_g);

    // Build dynamical matrix from all contributions (only irreducible columns computed)
    const dyn = try buildGammaDynmat(alloc, gs, pert_results, vloc1_gs, rho1_core_gs, rho0_g, ionic.charges, ionic.positions, cell_bohr, recip, volume, cfg, irr_info);
    defer alloc.free(dyn);

    // Reconstruct non-irreducible columns from symmetry
    if (irr_info.n_irr_atoms < n_atoms) {
        dynmat_mod.reconstructDynmatColumnsReal(dyn, n_atoms, irr_info, symops, sym_data.indsym, cell_bohr);
    }

    // Print full Cartesian dynmat (Ry/bohr²) before ASR for comparison with ABINIT
    logDfpt("dfpt: full dynmat (Ry/bohr², before ASR):\n", .{});
    for (0..dim) |i| {
        const ia = i / 3;
        const da = i % 3;
        for (0..dim) |j| {
            const jb = j / 3;
            const db = j % 3;
            logDfpt("  D(atom{d},{d}, atom{d},{d}) = {e:.10}\n", .{ ia, da, jb, db, dyn[i * dim + j] });
        }
    }

    // Apply acoustic sum rule
    dynmat_mod.applyASR(dyn, n_atoms);

    logDfpt("dfpt: ASR applied\n", .{});

    // Mass-weight the dynamical matrix
    dynmat_mod.massWeight(dyn, n_atoms, ionic.masses);

    // Diagonalize
    const result = try dynmat_mod.diagonalize(alloc, dyn, dim);

    logDfpt("dfpt: phonon frequencies (cm⁻¹):\n", .{});
    for (result.frequencies_cm1) |f| {
        logDfpt("dfpt:   {d:.2}\n", .{f});
    }

    // Electric field response: dielectric tensor
    if (cfg.dfpt.compute_dielectric) {
        logDfpt("dfpt: computing dielectric tensor (ddk at all k-points)\n", .{});
        const electric = @import("electric.zig");

        const dielectric = try electric.computeDielectricAllK(
            alloc, io,
            cfg,
            &gs,
            prepared.local_r,
            gs.species,
            gs.atoms,
            cell_bohr,
            recip,
            volume,
        );
        logDfpt("dfpt: dielectric tensor ε∞:\n", .{});
        for (0..3) |i| {
            logDfpt("  {d:.6} {d:.6} {d:.6}\n", .{
                dielectric.epsilon[i][0],
                dielectric.epsilon[i][1],
                dielectric.epsilon[i][2],
            });
        }

        // Write results
        var out_dir = std.Io.Dir.cwd().openDir(io, cfg.out_dir, .{}) catch null;
        defer if (out_dir) |*d| d.close(io);
        if (out_dir) |od| {
            electric.writeElectricResults(io, od, dielectric) catch |err| {
                logDfpt("dfpt: warning: failed to write electric.dat: {}\n", .{err});
            };
        }
    }

    return PhononResult{
        .frequencies_cm1 = result.frequencies_cm1,
        .omega2 = result.omega2,
        .eigenvectors = result.eigenvectors,
        .dim = result.dim,
    };
}

/// Run DFPT SCF for a single perturbation (atom_index, direction).
/// Returns the converged first-order density and wavefunctions.
pub fn solvePerturbation(
    alloc: std.mem.Allocator,
    gs: GroundState,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
) !PerturbationResult {
    const n_pw = gs.gvecs.len;
    const total = gs.grid.count();
    const n_occ = gs.n_occ;

    // Build V_ext^(1)(G) = V_loc^(1)(G) for this perturbation
    const vloc1_g = try perturbation.buildLocalPerturbation(
        alloc,
        gs.grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        gs.local_cfg,
        gs.ff_tables,
    );
    defer alloc.free(vloc1_g);

    // Build ρ^(1)_core(G) for NLCC (fixed, does not change during SCF)
    const rho1_core_g = try perturbation.buildCorePerturbation(
        alloc,
        gs.grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        gs.rho_core_tables,
    );
    defer alloc.free(rho1_core_g);

    // Convert ρ^(1)_core(G) → ρ^(1)_core(r) for XC perturbation
    const rho1_core_g_for_ifft = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_g_for_ifft);
    @memcpy(rho1_core_g_for_ifft, rho1_core_g);
    const rho1_core_r = try scf_mod.reciprocalToReal(alloc, gs.grid, rho1_core_g_for_ifft);
    defer alloc.free(rho1_core_r);

    // Initialize ρ^(1)(G) = 0
    var rho1_g = try alloc.alloc(math.Complex, total);
    @memset(rho1_g, math.complex.init(0.0, 0.0));

    // Allocate first-order wavefunctions
    var psi1 = try alloc.alloc([]math.Complex, n_occ);
    for (0..n_occ) |n| {
        psi1[n] = try alloc.alloc(math.Complex, n_pw);
        @memset(psi1[n], math.complex.init(0.0, 0.0));
    }

    // DFPT SCF loop
    var iter: usize = 0;
    while (iter < cfg.scf_max_iter) : (iter += 1) {
        // Build V_SCF^(1)(G) = V_H^(1) + V_xc^(1)
        const vh1_g = try perturbation.buildHartreePerturbation(alloc, gs.grid, rho1_g);
        defer alloc.free(vh1_g);

        // V_xc^(1)(r) = f_xc(r) × [ρ^(1)_val(r) + ρ^(1)_core(r)]
        // First get ρ^(1)_val(r) from ρ^(1)(G) via inverse FFT
        const rho1_g_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_g_copy);
        @memcpy(rho1_g_copy, rho1_g);
        const rho1_r = try scf_mod.reciprocalToReal(alloc, gs.grid, rho1_g_copy);
        defer alloc.free(rho1_r);

        // Add ρ^(1)_core to get total density response for XC
        const rho1_total_r = try alloc.alloc(f64, total);
        defer alloc.free(rho1_total_r);
        for (0..total) |i| {
            rho1_total_r[i] = rho1_r[i] + rho1_core_r[i];
        }

        const vxc1_r = try perturbation.buildXcPerturbationFull(alloc, gs, rho1_total_r);
        defer alloc.free(vxc1_r);

        // FFT V_xc^(1)(r) → V_xc^(1)(G)
        const vxc1_g = try scf_mod.realToReciprocal(alloc, gs.grid, vxc1_r, false);
        defer alloc.free(vxc1_g);

        // Total first-order potential: V^(1)(G) = V_loc^(1) + V_H^(1) + V_xc^(1)
        var vtot1_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(vtot1_g);
        for (0..total) |i| {
            vtot1_g[i] = math.complex.add(
                math.complex.add(vloc1_g[i], vh1_g[i]),
                vxc1_g[i],
            );
        }

        // Convert V^(1)(G) to real space for H^(1)|ψ⟩ application
        const vtot1_g_for_ifft = try alloc.alloc(math.Complex, total);
        defer alloc.free(vtot1_g_for_ifft);
        @memcpy(vtot1_g_for_ifft, vtot1_g);
        const vtot1_r = try scf_mod.reciprocalToReal(alloc, gs.grid, vtot1_g_for_ifft);
        defer alloc.free(vtot1_r);

        // Solve Sternheimer for each occupied band
        var new_rho1_g = try alloc.alloc(math.Complex, total);
        @memset(new_rho1_g, math.complex.init(0.0, 0.0));

        // Allocate nonlocal work buffers once (reused across bands)
        const nl_ctx_opt = gs.apply_ctx.nonlocal_ctx;
        var nl_out_buf: ?[]math.Complex = null;
        var nl_phase_buf: ?[]math.Complex = null;
        var nl_xphase_buf: ?[]math.Complex = null;
        var nl_coeff_buf: ?[]math.Complex = null;
        var nl_coeff2_buf: ?[]math.Complex = null;
        if (nl_ctx_opt) |nl_ctx| {
            nl_out_buf = try alloc.alloc(math.Complex, n_pw);
            nl_phase_buf = try alloc.alloc(math.Complex, n_pw);
            nl_xphase_buf = try alloc.alloc(math.Complex, n_pw);
            nl_coeff_buf = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
            nl_coeff2_buf = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
        }
        defer {
            if (nl_out_buf) |b| alloc.free(b);
            if (nl_phase_buf) |b| alloc.free(b);
            if (nl_xphase_buf) |b| alloc.free(b);
            if (nl_coeff_buf) |b| alloc.free(b);
            if (nl_coeff2_buf) |b| alloc.free(b);
        }

        for (0..n_occ) |n| {
            // Build RHS: -P_c × H^(1)|ψ_n^(0)⟩
            // H^(1)|ψ⟩ = V^(1)_SCF(r)|ψ(r)⟩ + V_nl^(1)|ψ⟩
            const rhs = try applyV1Psi(alloc, gs.grid, gs.gvecs, vtot1_r, gs.wavefunctions[n], gs.apply_ctx);
            defer alloc.free(rhs);

            // Add nonlocal perturbation: V_nl^(1)|ψ⟩
            if (nl_ctx_opt) |nl_ctx| {
                perturbation.applyNonlocalPerturbation(
                    gs.gvecs,
                    gs.atoms,
                    nl_ctx,
                    atom_index,
                    direction,
                    1.0 / gs.grid.volume,
                    gs.wavefunctions[n],
                    nl_out_buf.?,
                    nl_phase_buf.?,
                    nl_xphase_buf.?,
                    nl_coeff_buf.?,
                    nl_coeff2_buf.?,
                );
                for (0..n_pw) |g| {
                    rhs[g] = math.complex.add(rhs[g], nl_out_buf.?[g]);
                }
            }

            // Negate: rhs = -H^(1)|ψ⟩
            for (0..n_pw) |g| {
                rhs[g] = math.complex.scale(rhs[g], -1.0);
            }

            // Project onto conduction band: P_c × rhs
            sternheimer.projectConduction(rhs, gs.wavefunctions, n_occ);

            // Solve Sternheimer equation
            const result = try sternheimer.solve(
                alloc,
                gs.apply_ctx,
                rhs,
                gs.eigenvalues[n],
                gs.wavefunctions,
                n_occ,
                gs.gvecs,
                .{
                    .tol = cfg.sternheimer_tol,
                    .max_iter = cfg.sternheimer_max_iter,
                    .alpha_shift = cfg.alpha_shift,
                },
            );

            // Update ψ^(1)_n (swap buffers)
            alloc.free(psi1[n]);
            psi1[n] = result.psi1;

            // Accumulate density response:
            // ρ^(1)(r) = 2 × Σ_n Re[ψ_n^(0)*(r) × ψ_n^(1)(r)]
            // In G-space: ρ^(1)(G) = 2 × Σ_n Σ_G' conj(ψ_n^(0)(G')) × ψ_n^(1)(G'+G)
            // Simplified: compute in real space and FFT back
        }

        // Compute new ρ^(1)(r) from wavefunctions
        const new_rho1_r = try computeRho1(alloc, gs.grid, gs.gvecs, gs.wavefunctions, psi1, n_occ, gs.apply_ctx);
        defer alloc.free(new_rho1_r);

        // FFT ρ^(1)(r) → ρ^(1)(G)
        alloc.free(new_rho1_g);
        new_rho1_g = try scf_mod.realToReciprocal(alloc, gs.grid, new_rho1_r, false);

        // Check convergence: ||ρ^(1)_new - ρ^(1)_old||
        var diff_norm: f64 = 0.0;
        for (0..total) |i| {
            const dr = new_rho1_g[i].r - rho1_g[i].r;
            const di = new_rho1_g[i].i - rho1_g[i].i;
            diff_norm += dr * dr + di * di;
        }
        diff_norm = @sqrt(diff_norm);

        // Mix: ρ^(1) = (1-β) ρ^(1)_old + β ρ^(1)_new
        const beta = cfg.mixing_beta;
        for (0..total) |i| {
            rho1_g[i] = math.complex.add(
                math.complex.scale(rho1_g[i], 1.0 - beta),
                math.complex.scale(new_rho1_g[i], beta),
            );
        }
        alloc.free(new_rho1_g);

        logDfpt("dfpt_scf: iter={d} diff_norm={e:.6}\n", .{ iter, diff_norm });

        if (diff_norm < cfg.scf_tol) {
            logDfpt("dfpt_scf: converged at iter={d}\n", .{iter});
            break;
        }
    }

    // Recompute ρ^(1) from final ψ^(1) for consistency
    // (The mixed rho1_g might differ slightly from ρ computed from psi1)
    const final_rho1_r = try computeRho1(alloc, gs.grid, gs.gvecs, gs.wavefunctions, psi1, n_occ, gs.apply_ctx);
    defer alloc.free(final_rho1_r);
    const final_rho1_g = try scf_mod.realToReciprocal(alloc, gs.grid, final_rho1_r, false);

    // Diagnostic: compare mixed vs recomputed density
    var rho1_norm: f64 = 0.0;
    var rho1_diff: f64 = 0.0;
    for (0..total) |i| {
        rho1_norm += final_rho1_g[i].r * final_rho1_g[i].r + final_rho1_g[i].i * final_rho1_g[i].i;
        const dr = final_rho1_g[i].r - rho1_g[i].r;
        const di = final_rho1_g[i].i - rho1_g[i].i;
        rho1_diff += dr * dr + di * di;
    }
    logDfpt("dfpt_scf: final |rho1_g|={e:.6} |rho1_mixed - rho1_psi1|={e:.6}\n", .{ @sqrt(rho1_norm), @sqrt(rho1_diff) });

    // Use the recomputed density (consistent with psi1)
    alloc.free(rho1_g);

    return .{
        .rho1_g = final_rho1_g,
        .psi1 = psi1,
    };
}

/// Apply V^(1)(r)|ψ(r)⟩ → result in PW basis.
/// Uses FFT: ψ(r) = IFFT[scatter(ψ_G)], multiply by V^(1)(r), FFT back, gather.
pub fn applyV1Psi(
    alloc: std.mem.Allocator,
    grid: Grid,
    gvecs: []const plane_wave.GVector,
    v1_r: []const f64,
    psi: []const math.Complex,
    ctx: *scf_mod.ApplyContext,
) ![]math.Complex {
    const n_pw = gvecs.len;
    const total = grid.count();

    // Scatter PW coefficients to full grid
    const work_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g);
    @memset(work_g, math.complex.init(0.0, 0.0));
    ctx.map.scatter(psi, work_g);

    // IFFT to real space
    const work_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r);
    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work_g, work_r, null);

    // Multiply by V^(1)(r)
    for (0..total) |i| {
        work_r[i] = math.complex.scale(work_r[i], v1_r[i]);
    }

    // FFT back to reciprocal space
    const work_g_out = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g_out);
    try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, work_r, work_g_out, null);

    // Gather back to PW basis
    const result = try alloc.alloc(math.Complex, n_pw);
    ctx.map.gather(work_g_out, result);

    return result;
}

/// Compute first-order density response from wavefunctions.
/// ρ^(1)(r) = (4/Ω) × Σ_n Re[ψ_n^(0)*(r) × ψ_n^(1)(r)]
/// Factor 4 = 2 (spin degeneracy) × 2 (derivative of |ψ|²).
/// Factor 1/Ω from the PW normalization convention (ψ^grid = √Ω × ψ^physical).
pub fn computeRho1(
    alloc: std.mem.Allocator,
    grid: Grid,
    gvecs: []const plane_wave.GVector,
    psi0: []const []const math.Complex,
    psi1: []const []const math.Complex,
    n_occ: usize,
    ctx: *scf_mod.ApplyContext,
) ![]f64 {
    const total = grid.count();
    const rho1_r = try alloc.alloc(f64, total);
    @memset(rho1_r, 0.0);

    const work_g0 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g0);
    const work_r0 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r0);
    const work_g1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g1);
    const work_r1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r1);

    // Weight: 4/Ω = 2(spin) × 2(d|ψ|²) × (1/Ω)(volume normalization)
    const weight = 4.0 / grid.volume;

    for (0..n_occ) |n| {
        // ψ_n^(0)(r) via IFFT
        @memset(work_g0, math.complex.init(0.0, 0.0));
        ctx.map.scatter(psi0[n], work_g0);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work_g0, work_r0, null);

        // ψ_n^(1)(r) via IFFT
        @memset(work_g1, math.complex.init(0.0, 0.0));
        ctx.map.scatter(psi1[n], work_g1);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work_g1, work_r1, null);

        // ρ^(1)(r) += (4/Ω) × Re[ψ_n^(0)*(r) × ψ_n^(1)(r)]
        for (0..total) |i| {
            const conj0 = math.complex.conj(work_r0[i]);
            const prod = math.complex.mul(conj0, work_r1[i]);
            rho1_r[i] += weight * prod.r;
        }
    }
    _ = gvecs;

    return rho1_r;
}

/// Compute the electronic contribution to the dynamical matrix element.
/// D^elec_{Iα,Jβ} = Σ_G conj(V^(1)_{Iα}(G)) × ρ^(1)_{Jβ}(G) × Ω
pub fn computeElecDynmatElement(
    vloc1_g: []const math.Complex,
    rho1_g: []const math.Complex,
    volume: f64,
) f64 {
    var sum = math.complex.init(0.0, 0.0);
    for (0..vloc1_g.len) |i| {
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(vloc1_g[i]), rho1_g[i]));
    }
    return sum.r * volume;
}

/// Compute the nonlocal response contribution to the dynamical matrix.
/// C_nl_{Iα,Jβ} = 4 × Σ_n Re[⟨ψ_n|V_nl^(1)_{Iα}|δψ_n,Jβ⟩]
///
/// This accounts for the nonlocal pseudopotential's contribution to the
/// force constant via the first-order wavefunctions.
pub fn computeNonlocalResponseDynmat(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]f64 {
    const dim = 3 * n_atoms;
    const n_pw = gs.gvecs.len;
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    const nl_ctx = gs.apply_ctx.nonlocal_ctx orelse return dyn;

    // Allocate work buffers
    const nl_out = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_out);
    const nl_phase = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_phase);
    const nl_xphase = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_xphase);
    const nl_coeff = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
    defer alloc.free(nl_coeff);
    const nl_coeff2 = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
    defer alloc.free(nl_coeff2);

    for (0..dim) |i| {
        const ia = i / 3;
        const dir_a = i % 3;
        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            var sum: f64 = 0.0;
            for (0..gs.n_occ) |n| {
                // Compute V_nl^(1)_{ia,dir_a} |δψ_n,j⟩
                perturbation.applyNonlocalPerturbation(
                    gs.gvecs,
                    gs.atoms,
                    nl_ctx,
                    ia,
                    dir_a,
                    1.0 / gs.grid.volume,
                    pert_results[j].psi1[n],
                    nl_out,
                    nl_phase,
                    nl_xphase,
                    nl_coeff,
                    nl_coeff2,
                );
                // ⟨ψ_n|V_nl^(1)|δψ_n⟩ = Σ_G ψ*_n(G) × nl_out(G)
                var ip = math.complex.init(0.0, 0.0);
                for (0..n_pw) |g| {
                    ip = math.complex.add(ip, math.complex.mul(
                        math.complex.conj(gs.wavefunctions[n][g]),
                        nl_out[g],
                    ));
                }
                sum += ip.r;
            }
            dyn[i * dim + j] = 4.0 * sum;
        }
    }

    return dyn;
}

/// Compute the NLCC cross-term contribution to the dynamical matrix.
///
/// D_NLCC_cross(Iα,Jβ) = ∫ V_xc^(1)[ρ^(1)_core,I](r) × ρ^(1)_total,J(r) dr
///
/// For LDA this reduces to ∫ f_xc(r) × ρ^(1)_core,I(r) × ρ^(1)_total,J(r) dr.
/// For GGA, V_xc^(1) includes gradient-dependent terms via buildXcPerturbationFull.
pub fn computeNlccCrossDynmat(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: GroundState,
    pert_results: []PerturbationResult,
    rho1_core_gs: []const []math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]f64 {
    const dim = 3 * n_atoms;
    const total = grid.count();
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    const dv = grid.volume / @as(f64, @floatFromInt(total));

    for (0..dim) |i| {
        // Get ρ^(1)_core,Iα(r)
        const rho1_core_i_g_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_core_i_g_copy);
        @memcpy(rho1_core_i_g_copy, rho1_core_gs[i]);
        const rho1_core_i_r = try scf_mod.reciprocalToReal(alloc, grid, rho1_core_i_g_copy);
        defer alloc.free(rho1_core_i_r);

        // Build V_xc^(1)[ρ^(1)_core,I] using full GGA-aware kernel
        const vxc1_core_i = try perturbation.buildXcPerturbationFull(alloc, gs, rho1_core_i_r);
        defer alloc.free(vxc1_core_i);

        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            // Get ρ^(1)_total,J(r) = ρ^(1)_val,J + ρ^(1)_core,J
            const rho1_val_g_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_val_g_copy);
            @memcpy(rho1_val_g_copy, pert_results[j].rho1_g);
            const rho1_val_r = try scf_mod.reciprocalToReal(alloc, grid, rho1_val_g_copy);
            defer alloc.free(rho1_val_r);

            const rho1_core_j_g_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_core_j_g_copy);
            @memcpy(rho1_core_j_g_copy, rho1_core_gs[j]);
            const rho1_core_j_r = try scf_mod.reciprocalToReal(alloc, grid, rho1_core_j_g_copy);
            defer alloc.free(rho1_core_j_r);

            // D_NLCC(I,J) = ∫ V_xc^(1)[ρ^(1)_core,I](r) × ρ^(1)_total,J(r) dr
            var sum: f64 = 0.0;
            for (0..total) |r| {
                const rho1_total_j = rho1_val_r[r] + rho1_core_j_r[r];
                sum += vxc1_core_i[r] * rho1_total_j;
            }
            dyn[i * dim + j] = sum * dv;
        }
    }

    return dyn;
}

/// Build the Γ-point dynamical matrix from all contributions:
/// electronic, Ewald, self-energy, nonlocal response, nonlocal self-energy,
/// NLCC cross, and NLCC self.
/// Returns an owned slice of size dim×dim (caller must free).
fn buildGammaDynmat(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    vloc1_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []math.Complex,
    charges: []const f64,
    positions: []const math.Vec3,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    cfg: config_mod.Config,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]f64 {
    const n_atoms = gs.atoms.len;
    const dim = 3 * n_atoms;
    const grid = gs.grid;

    var dyn = try alloc.alloc(f64, dim * dim);
    errdefer alloc.free(dyn);
    @memset(dyn, 0.0);

    // Electronic contribution (only irreducible columns j)
    for (0..dim) |i| {
        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            dyn[i * dim + j] = computeElecDynmatElement(
                vloc1_gs[i],
                pert_results[j].rho1_g,
                volume,
            );
        }
    }
    logDfpt("dfpt: electronic D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ dyn[0], dyn[3] });

    // Ewald (ionic) contribution — returned in Hartree/bohr², convert to Ry/bohr² (×2)
    // (full matrix, no symmetry reduction needed — cheap analytic computation)
    const ewald_dyn = try ewald2.ewaldDynmat(alloc, cell_bohr, recip, charges, positions);
    defer alloc.free(ewald_dyn);
    logDfpt("dfpt: ewald D(0x,0x)={e:.10} D(0x,1x)={e:.10} (Ha)\n", .{ ewald_dyn[0], ewald_dyn[3] });
    for (0..dim * dim) |i| {
        dyn[i] += ewald_dyn[i] * 2.0;
    }

    // Self-energy (non-variational) contribution: ∫ V^(2) × ρ^(0) dr
    // (full matrix, no symmetry reduction needed — cheap analytic computation)
    const self_dyn = try dynmat_contrib.computeSelfEnergyDynmat(alloc, grid, gs.species, gs.atoms, rho0_g, gs.local_cfg, gs.ff_tables);
    defer alloc.free(self_dyn);
    logDfpt("dfpt: self-energy D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ self_dyn[0], self_dyn[3] });
    for (0..dim * dim) |i| {
        dyn[i] += self_dyn[i];
    }

    // Nonlocal response contribution: C_nl = 4 × Σ_n Re⟨ψ|V_nl^(1)|δψ⟩
    // (only irreducible columns j — uses pert_results[j].psi1)
    const nl_resp_dyn = try computeNonlocalResponseDynmat(alloc, gs, pert_results, n_atoms, irr_info);
    defer alloc.free(nl_resp_dyn);
    logDfpt("dfpt: nl-response D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ nl_resp_dyn[0], nl_resp_dyn[3] });
    for (0..dim * dim) |i| {
        dyn[i] += nl_resp_dyn[i];
    }

    // Nonlocal self-energy contribution: ⟨ψ|V_nl^(2)|ψ⟩
    const nl_self_dyn = try dynmat_contrib.computeNonlocalSelfEnergyDynmat(alloc, gs, n_atoms);
    defer alloc.free(nl_self_dyn);
    logDfpt("dfpt: nl-self-energy D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ nl_self_dyn[0], nl_self_dyn[3] });
    for (0..dim * dim) |i| {
        dyn[i] += nl_self_dyn[i];
    }

    // NLCC contributions to dynamical matrix
    if (gs.rho_core != null) {
        // Cross-term: ∫ f_xc × ρ^(1)_core,Iα × ρ^(1)_total,Jβ dr
        // (only irreducible columns j — uses pert_results[j].rho1_g)
        const nlcc_cross_dyn = try computeNlccCrossDynmat(alloc, grid, gs, pert_results, rho1_core_gs, n_atoms, irr_info);
        defer alloc.free(nlcc_cross_dyn);
        logDfpt("dfpt: nlcc-cross D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ nlcc_cross_dyn[0], nlcc_cross_dyn[3] });
        for (0..dim * dim) |i| {
            dyn[i] += nlcc_cross_dyn[i];
        }

        // Self-energy term: Σ_G V_xc*(G) × (-G_αG_β) × ρ_core_form × exp(-iGτ)
        const vxc_r_slice = gs.vxc_r.?;
        const vxc_r_copy = try alloc.alloc(f64, vxc_r_slice.len);
        defer alloc.free(vxc_r_copy);
        @memcpy(vxc_r_copy, vxc_r_slice);
        const vxc_g = try scf_mod.realToReciprocal(alloc, grid, vxc_r_copy, false);
        defer alloc.free(vxc_g);

        const nlcc_self_dyn = try dynmat_contrib.computeNlccSelfDynmat(alloc, grid, gs.species, gs.atoms, vxc_g, gs.rho_core_tables);
        defer alloc.free(nlcc_self_dyn);
        logDfpt("dfpt: nlcc-self D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ nlcc_self_dyn[0], nlcc_self_dyn[3] });
        for (0..dim * dim) |i| {
            dyn[i] += nlcc_self_dyn[i];
        }
    }

    // D3 dispersion contribution to dynamical matrix
    if (cfg.vdw.enabled) {
        const atomic_numbers = try alloc.alloc(usize, n_atoms);
        defer alloc.free(atomic_numbers);
        const atom_positions = try alloc.alloc(math.Vec3, n_atoms);
        defer alloc.free(atom_positions);
        for (gs.atoms, 0..) |atom, idx| {
            atomic_numbers[idx] = d3_params.atomicNumber(gs.species[atom.species_index].symbol) orelse 0;
            atom_positions[idx] = atom.position;
        }
        var damping = d3_params.pbe_d3bj;
        if (cfg.vdw.s6) |v| damping.s6 = v;
        if (cfg.vdw.s8) |v| damping.s8 = v;
        if (cfg.vdw.a1) |v| damping.a1 = v;
        if (cfg.vdw.a2) |v| damping.a2 = v;
        const d3_dyn = try d3.computeDynmat(
            alloc,
            atomic_numbers,
            atom_positions,
            cell_bohr,
            damping,
            cfg.vdw.cutoff_radius,
            cfg.vdw.cn_cutoff,
        );
        defer alloc.free(d3_dyn);
        logDfpt("dfpt: d3-disp D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ d3_dyn[0], if (dim > 3) d3_dyn[3] else 0.0 });
        for (0..dim * dim) |i| {
            dyn[i] += d3_dyn[i];
        }
    }

    logDfpt("dfpt: total dynmat diagonal (Ry):", .{});
    for (0..dim) |i| {
        logDfpt(" {e:.6}", .{dyn[i * dim + i]});
    }
    logDfpt("\n", .{});

    return dyn;
}

/// Run diagnostic tests on the DFPT perturbation results at Γ-point.
/// This includes:
/// 1. ASR check: Σ_J ψ^(1)_{n,Jβ} vs P_c(-iG_β ψ^(0)_n)
/// 2. V_nl^(1) commutator test
/// 3. D_elec comparison via wavefunction inner product
fn runDiagnostics(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    vloc1_gs: []const []math.Complex,
    n_atoms: usize,
) !void {
    const n_pw = gs.gvecs.len;

    // 1. ASR diagnostic: check Σ_J ψ^(1)_{n,Jβ} vs P_c(-iG_β ψ^(0)_n)
    {
        for (0..3) |dir| {
            for (0..gs.n_occ) |n| {
                const sum_psi1 = try alloc.alloc(math.Complex, n_pw);
                defer alloc.free(sum_psi1);
                @memset(sum_psi1, math.complex.init(0.0, 0.0));
                for (0..n_atoms) |ia| {
                    const j_idx = 3 * ia + dir;
                    for (0..n_pw) |g| {
                        sum_psi1[g] = math.complex.add(sum_psi1[g], pert_results[j_idx].psi1[n][g]);
                    }
                }

                const expected = try alloc.alloc(math.Complex, n_pw);
                defer alloc.free(expected);
                for (0..n_pw) |g| {
                    const g_d = switch (dir) {
                        0 => gs.gvecs[g].cart.x,
                        1 => gs.gvecs[g].cart.y,
                        2 => gs.gvecs[g].cart.z,
                        else => 0.0,
                    };
                    const psi = gs.wavefunctions[n][g];
                    expected[g] = math.complex.init(psi.i * g_d, -psi.r * g_d);
                }
                sternheimer.projectConduction(expected, gs.wavefunctions, gs.n_occ);

                var diff_norm: f64 = 0.0;
                var exp_norm: f64 = 0.0;
                var sum_norm: f64 = 0.0;
                for (0..n_pw) |g| {
                    const dr = sum_psi1[g].r - expected[g].r;
                    const di = sum_psi1[g].i - expected[g].i;
                    diff_norm += dr * dr + di * di;
                    exp_norm += expected[g].r * expected[g].r + expected[g].i * expected[g].i;
                    sum_norm += sum_psi1[g].r * sum_psi1[g].r + sum_psi1[g].i * sum_psi1[g].i;
                }
                logDfpt("dfpt_asr: dir={d} band={d} |Σ_J ψ1|={e:.6} |expected|={e:.6} |diff|={e:.6} rel={e:.6}\n", .{
                    dir,                                                        n, @sqrt(sum_norm), @sqrt(exp_norm), @sqrt(diff_norm),
                    if (exp_norm > 1e-30) @sqrt(diff_norm / exp_norm) else 0.0,
                });
            }
        }
    }

    // 2. V_nl^(1) commutator test: Σ_J V_nl^(1)_J|ψ⟩ should equal V_nl|iGψ⟩ - iG(V_nl|ψ⟩)
    if (gs.apply_ctx.nonlocal_ctx != null) {
        const nl_out1 = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(nl_out1);
        const nl_out2 = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(nl_out2);
        const nl_out3 = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(nl_out3);
        const work_igpsi = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(work_igpsi);
        const nl_phase = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(nl_phase);
        const nl_xphase = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(nl_xphase);
        const nl_ctx = gs.apply_ctx.nonlocal_ctx.?;
        const nl_coeff = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
        defer alloc.free(nl_coeff);
        const nl_coeff2 = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
        defer alloc.free(nl_coeff2);

        for (0..1) |dir| { // just test x direction
            for (0..gs.n_occ) |n| {
                // Compute Σ_J V_nl^(1)_{J,dir}|ψ^(0)_n⟩
                @memset(nl_out1, math.complex.init(0.0, 0.0));
                for (0..n_atoms) |ia| {
                    perturbation.applyNonlocalPerturbation(
                        gs.gvecs,
                        gs.atoms,
                        nl_ctx,
                        ia,
                        dir,
                        1.0 / gs.grid.volume,
                        gs.wavefunctions[n],
                        nl_out2,
                        nl_phase,
                        nl_xphase,
                        nl_coeff,
                        nl_coeff2,
                    );
                    for (0..n_pw) |g| {
                        nl_out1[g] = math.complex.add(nl_out1[g], nl_out2[g]);
                    }
                }

                // Compute V_nl|iG_dir ψ^(0)_n⟩
                for (0..n_pw) |g| {
                    const g_d = switch (dir) {
                        0 => gs.gvecs[g].cart.x,
                        1 => gs.gvecs[g].cart.y,
                        2 => gs.gvecs[g].cart.z,
                        else => 0.0,
                    };
                    const psi = gs.wavefunctions[n][g];
                    work_igpsi[g] = math.complex.init(-psi.i * g_d, psi.r * g_d);
                }
                try scf_mod.applyNonlocalPotential(gs.apply_ctx, work_igpsi, nl_out2);

                // Compute iG_dir × (V_nl|ψ^(0)_n⟩)
                try scf_mod.applyNonlocalPotential(gs.apply_ctx, gs.wavefunctions[n], nl_out3);
                for (0..n_pw) |g| {
                    const g_d = switch (dir) {
                        0 => gs.gvecs[g].cart.x,
                        1 => gs.gvecs[g].cart.y,
                        2 => gs.gvecs[g].cart.z,
                        else => 0.0,
                    };
                    const v = nl_out3[g];
                    nl_out3[g] = math.complex.init(-v.i * g_d, v.r * g_d);
                }

                // Expected: Σ_J V^(1)_{nl,J}|ψ⟩ = V_nl|iGψ⟩ - iG(V_nl|ψ⟩) = nl_out2 - nl_out3
                var diff_norm: f64 = 0.0;
                var lhs_norm: f64 = 0.0;
                var rhs_norm: f64 = 0.0;
                for (0..n_pw) |g| {
                    const expected_g = math.complex.sub(nl_out2[g], nl_out3[g]);
                    const dr = nl_out1[g].r - expected_g.r;
                    const di = nl_out1[g].i - expected_g.i;
                    diff_norm += dr * dr + di * di;
                    lhs_norm += nl_out1[g].r * nl_out1[g].r + nl_out1[g].i * nl_out1[g].i;
                    rhs_norm += expected_g.r * expected_g.r + expected_g.i * expected_g.i;
                }
                logDfpt("dfpt_vnl_test: dir={d} band={d} |Σ V1_nl|ψ|={e:.6} |V_nl|iGψ⟩-iG V_nl|ψ⟩|={e:.6} |diff|={e:.6} rel={e:.6}\n", .{
                    dir,                                                        n, @sqrt(lhs_norm), @sqrt(rhs_norm), @sqrt(diff_norm),
                    if (rhs_norm > 1e-30) @sqrt(diff_norm / rhs_norm) else 0.0,
                });
            }
        }
    }

    // 3. D_elec comparison via wavefunction inner product
    // D_elec_wf_{Iα,Jβ} = 4 × Σ_n Re⟨ψ_n|V_loc^(1)_{Iα}|ψ_n^(1)_{Jβ}⟩
    {
        const vloc1_0x_g_copy = try alloc.alloc(math.Complex, gs.grid.count());
        defer alloc.free(vloc1_0x_g_copy);
        @memcpy(vloc1_0x_g_copy, vloc1_gs[0]);
        const vloc1_0x_r = try scf_mod.reciprocalToReal(alloc, gs.grid, vloc1_0x_g_copy);
        defer alloc.free(vloc1_0x_r);

        var d_wf_00: f64 = 0.0;
        var d_wf_03: f64 = 0.0;
        for (0..gs.n_occ) |n| {
            const vpsi_00 = try applyV1Psi(alloc, gs.grid, gs.gvecs, vloc1_0x_r, pert_results[0].psi1[n], gs.apply_ctx);
            defer alloc.free(vpsi_00);

            var ip_00 = math.complex.init(0.0, 0.0);
            for (0..n_pw) |g| {
                ip_00 = math.complex.add(ip_00, math.complex.mul(math.complex.conj(gs.wavefunctions[n][g]), vpsi_00[g]));
            }
            d_wf_00 += 4.0 * ip_00.r;

            const vpsi_03 = try applyV1Psi(alloc, gs.grid, gs.gvecs, vloc1_0x_r, pert_results[3].psi1[n], gs.apply_ctx);
            defer alloc.free(vpsi_03);

            var ip_03 = math.complex.init(0.0, 0.0);
            for (0..n_pw) |g| {
                ip_03 = math.complex.add(ip_03, math.complex.mul(math.complex.conj(gs.wavefunctions[n][g]), vpsi_03[g]));
            }
            d_wf_03 += 4.0 * ip_03.r;
        }
        logDfpt("dfpt: D_elec_wf D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ d_wf_00, d_wf_03 });
    }
}

// =====================================================================
// Perturbation parallelism
// =====================================================================

const GammaPertShared = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    gs: *const GroundState,
    dfpt_cfg: *const DfptConfig,
    pert_results: []PerturbationResult,
    vloc1_gs: [][]math.Complex,
    rho1_core_gs: [][]math.Complex,
    dim: usize,
    irr_pert_indices: ?[]const usize = null,

    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

const GammaPertWorker = struct {
    shared: *GammaPertShared,
    thread_index: usize,
};

fn setGammaPertError(shared: *GammaPertShared, e: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);
    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

fn gammaPertWorkerFn(worker: *GammaPertWorker) void {
    const shared = worker.shared;
    const gs = shared.gs.*;
    const alloc = shared.alloc;

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const work_idx = shared.next_index.fetchAdd(1, .acq_rel);
        if (work_idx >= shared.dim) break;

        // Map work index to actual perturbation index
        const idx = if (shared.irr_pert_indices) |indices| indices[work_idx] else work_idx;
        const ia = idx / 3;
        const dir = idx % 3;
        const dir_names = [_][]const u8{ "x", "y", "z" };

        {
            shared.log_mutex.lockUncancelable(shared.io);
            defer shared.log_mutex.unlock(shared.io);
            logDfpt("dfpt: [thread {d}] solving perturbation atom={d} dir={s} ({d}/{d})\n", .{ worker.thread_index, ia, dir_names[dir], work_idx + 1, shared.dim });
        }

        // Solve DFPT SCF (vloc1/rho1_core already built by caller)
        shared.pert_results[idx] = solvePerturbation(
            alloc,
            gs,
            ia,
            dir,
            shared.dfpt_cfg.*,
        ) catch |e| {
            setGammaPertError(shared, e);
            shared.stop.store(1, .release);
            break;
        };
    }
}
