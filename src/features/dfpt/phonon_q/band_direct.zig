//! Direct q-path phonon band structure (no IFC interpolation).
//!
//! Solves DFPT from scratch at every q-point along a high-symmetry path,
//! builds the complex dynamical matrix, applies symmetry reconstruction
//! and the acoustic sum rule at q=0, mass-weights, and diagonalizes
//! the Hermitian mass-weighted D(q) to produce the phonon frequencies.
//!
//! q-point-level perturbation parallelism lives in this file because it
//! is only used by runPhononBand — a per-q ThreadPool-like driver that
//! picks (atom, direction) work items for the multi-k solver.

const std = @import("std");
const math = @import("../../math/math.zig");
const scf_mod = @import("../../scf/scf.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const form_factor = @import("../../pseudopotential/form_factor.zig");
const config_mod = @import("../../config/config.zig");
const model_mod = @import("../../dft/model.zig");
const symmetry_mod = @import("../../symmetry/symmetry.zig");

const dfpt = @import("../dfpt.zig");
const perturbation = dfpt.perturbation;
const dynmat_mod = dfpt.dynmat;
const GroundState = dfpt.GroundState;
const DfptConfig = dfpt.DfptConfig;
const IonicData = dfpt.IonicData;
const logDfpt = dfpt.logDfpt;
const logDfptInfo = dfpt.logDfptInfo;

const qpath = @import("qpath.zig");
const generateFccQPath = qpath.generateFccQPath;

const kpt_gs = @import("kpt_gs.zig");
const prepareFullBZKpoints = kpt_gs.prepareFullBZKpoints;

const kpt_dfpt = @import("kpt_dfpt.zig");
const KPointDfptData = kpt_dfpt.KPointDfptData;
const MultiKPertResult = kpt_dfpt.MultiKPertResult;
const buildKPointDfptDataFromGS = kpt_dfpt.buildKPointDfptDataFromGS;

const solver_multik = @import("solver_multik.zig");
const solvePerturbationQMultiK = solver_multik.solvePerturbationQMultiK;

const dynmat_build = @import("dynmat_build.zig");
const buildQDynmatMultiK = dynmat_build.buildQDynmatMultiK;

const Grid = scf_mod.Grid;

/// Result of phonon band structure calculation.
pub const PhononBandResult = struct {
    /// q-path distances for plotting
    distances: []f64,
    /// Frequencies [n_q][n_modes] in cm⁻¹
    frequencies: [][]f64,
    /// Number of modes (3 × n_atoms)
    n_modes: usize,
    /// Number of q-points
    n_q: usize,
    /// Labels for high-symmetry points
    labels: [][]const u8,
    /// Label positions (indices into distances)
    label_positions: []usize,

    pub fn deinit(self: *PhononBandResult, alloc: std.mem.Allocator) void {
        for (self.frequencies) |f| alloc.free(f);
        alloc.free(self.frequencies);
        alloc.free(self.distances);
        alloc.free(self.labels);
        alloc.free(self.label_positions);
    }
};

/// Run DFPT phonon band structure calculation.
/// Computes phonon frequencies along a q-path for the FCC lattice.
pub fn runPhononBand(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    model: *const model_mod.Model,
    npoints_per_seg: usize,
) !PhononBandResult {
    const species = model.species;
    const atoms = model.atoms;
    const cell_bohr = model.cell_bohr;
    const recip = model.recip;
    const volume = model.volume_bohr;
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const grid = scf_result.grid;

    logDfptInfo("dfpt_band: starting phonon band calculation ({d} atoms)\n", .{n_atoms});

    // Prepare ground state (PW basis, eigenvalues, wavefunctions, NLCC, etc.)
    var prepared = try dfpt.prepareGroundState(
        alloc,
        io,
        cfg,
        scf_result,
        species,
        atoms,
        volume,
        recip,
    );
    defer prepared.deinit();
    const gs = prepared.gs;

    // Ionic data for dynmat construction
    const ionic = try IonicData.init(alloc, species, atoms);
    defer ionic.deinit(alloc);

    // Ground-state density in G-space
    const rho0_g = try scf_mod.realToReciprocal(alloc, grid, scf_result.density, false);
    defer alloc.free(rho0_g);

    // V_xc(G) for NLCC self-energy
    // (need mutable copy since realToReciprocal requires mutable input)
    var vxc_g: ?[]math.Complex = null;
    if (prepared.vxc_r) |v| {
        vxc_g = try scf_mod.realToReciprocal(alloc, grid, v, false);
    }
    defer if (vxc_g) |v| alloc.free(v);

    // Generate q-path
    const q_path_data = try generateFccQPath(alloc, recip, npoints_per_seg);
    defer alloc.free(q_path_data.q_points_frac);
    defer alloc.free(q_path_data.q_points_cart);

    const n_q = q_path_data.q_points_cart.len;
    logDfptInfo("dfpt_band: {d} q-points along path\n", .{n_q});

    // Build symmetry operations and atom mapping table for dynmat symmetrization
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    defer alloc.free(symops);
    logDfptInfo("dfpt_band: {d} symmetry operations found\n", .{symops.len});

    const sym_data = try dynmat_mod.buildIndsym(alloc, symops, atoms, recip, 1e-5);
    defer {
        for (sym_data.indsym) |row| alloc.free(row);
        alloc.free(sym_data.indsym);
        for (sym_data.tnons_shift) |row| alloc.free(row);
        alloc.free(sym_data.tnons_shift);
    }

    // Allocate result arrays
    var frequencies = try alloc.alloc([]f64, n_q);
    var freq_count: usize = 0;
    errdefer {
        for (0..freq_count) |i| alloc.free(frequencies[i]);
        alloc.free(frequencies);
    }

    const dfpt_cfg = DfptConfig.fromConfig(cfg);

    // ---------------------------------------------------------------
    // Prepare full-BZ k-point ground-state data (q-independent).
    // DFPT requires the full BZ sum over k-points. Even if SCF used
    // IBZ k-points, DFPT needs all k-points in the full BZ because
    // for q≠0, the function f(k, k+q) is not invariant under the
    // symmetry operations that map k to its star.
    // ---------------------------------------------------------------
    const kgs_data = try prepareFullBZKpoints(
        alloc,
        io,
        cfg,
        &gs,
        prepared.local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
    );
    defer {
        for (kgs_data) |*kg| kg.deinit(alloc);
        alloc.free(kgs_data);
    }
    const n_kpts = kgs_data.len;

    // q-point loop
    for (0..n_q) |iq| {
        const q_cart = q_path_data.q_points_cart[iq];
        const qf = q_path_data.q_points_frac[iq];
        const q_norm = math.Vec3.norm(q_cart);

        logDfpt(
            "dfpt_band: q[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n",
            .{ iq, qf.x, qf.y, qf.z, q_norm },
        );
        logDfpt("dfpt_band: q[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });

        // Find irreducible atoms for this q-point
        var irr_info = try dynmat_mod.findIrreducibleAtoms(
            alloc,
            symops,
            sym_data.indsym,
            n_atoms,
            qf,
        );
        defer irr_info.deinit(alloc);
        logDfpt(
            "dfpt_band: q[{d}] {d}/{d} irreducible atoms\n",
            .{ iq, irr_info.n_irr_atoms, n_atoms },
        );

        // Find irreducible perturbations (atom+direction) for this q-point
        // Note: direction-level perturbation reduction (findIrreduciblePerturbations) is available
        // in dynmat_mod but not yet used for SCF solving — buildQDynmatMultiK requires all 3
        // directions per irreducible atom. Log atom-level reduction only.

        // Build KPointDfptData from precomputed ground-state data + q-dependent k+q data
        const pert_thread_count = dfpt.perturbationThreadCount(dim, dfpt_cfg.perturbation_threads);
        const kpts = try buildKPointDfptDataFromGS(
            alloc,
            io,
            kgs_data,
            q_cart,
            q_norm,
            cfg,
            prepared.local_r,
            species,
            atoms,
            recip,
            volume,
            grid,
            pert_thread_count,
        );
        defer {
            for (kpts) |*kd| kd.deinitQOnly(alloc);
            alloc.free(kpts);
        }

        // Solve perturbations for each atom and direction using multi-k solver
        var pert_results_mk = try alloc.alloc(MultiKPertResult, dim);
        var vloc1_gs = try alloc.alloc([]math.Complex, dim);
        var rho1_core_gs = try alloc.alloc([]math.Complex, dim);
        var pert_count_local: usize = 0;
        var vloc1_count_local: usize = 0;
        var rho1_core_count_local: usize = 0;
        defer {
            for (0..pert_count_local) |i| pert_results_mk[i].deinit(alloc);
            alloc.free(pert_results_mk);
            for (0..vloc1_count_local) |i| alloc.free(vloc1_gs[i]);
            alloc.free(vloc1_gs);
            for (0..rho1_core_count_local) |i| alloc.free(rho1_core_gs[i]);
            alloc.free(rho1_core_gs);
        }

        if (pert_thread_count <= 1) {
            // Phase 1: Build vloc1, rho1_core for ALL perturbations (cheap, analytic)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.local_cfg,
                        gs.ff_tables,
                    );
                    vloc1_count_local = pidx + 1;

                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                    rho1_core_count_local = pidx + 1;
                }
            }

            // Phase 2: Solve perturbation SCF for irreducible atoms only (expensive)
            // Direction-level reduction is used for dynmat reconstruction, not for SCF solving,
            // because buildQDynmatMultiK requires all 3 directions per irreducible atom.
            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
            }
            pert_count_local = dim;

            for (irr_info.irr_atom_indices) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    pert_results_mk[pidx] = try solvePerturbationQMultiK(
                        alloc,
                        io,
                        kpts,
                        ia,
                        dir,
                        dfpt_cfg,
                        q_cart,
                        grid,
                        gs,
                        species,
                        atoms,
                        gs.ff_tables,
                        gs.rho_core_tables,
                    );

                    {
                        var rho1_norm: f64 = 0.0;
                        for (pert_results_mk[pidx].rho1_g) |c| {
                            rho1_norm += c.r * c.r + c.i * c.i;
                        }
                        logDfpt(
                            "dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n",
                            .{ ia, dir, @sqrt(rho1_norm) },
                        );
                    }
                }
            }
        } else {
            // Parallel path — solve only irreducible atoms' perturbations concurrently
            const n_irr_perts = irr_info.n_irr_atoms * 3;
            var pert_dfpt_cfg = dfpt_cfg;
            pert_dfpt_cfg.kpoint_threads = dfpt.kpointThreadsForPertParallel(
                pert_thread_count,
                dfpt_cfg.kpoint_threads,
            );

            logDfptInfo(
                "dfpt_band: using {d} pert threads × {d} kpt threads" ++
                    " for {d} perturbations ({d} irreducible)\n",
                .{
                    pert_thread_count,
                    pert_dfpt_cfg.kpoint_threads,
                    dim,
                    n_irr_perts,
                },
            );

            // Initialize output arrays to safe defaults for cleanup
            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
                vloc1_gs[i] = &.{};
                rho1_core_gs[i] = &.{};
            }
            pert_count_local = dim;
            vloc1_count_local = dim;
            rho1_core_count_local = dim;

            // Build vloc1 and rho1_core for ALL perturbations (cheap, serial)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;
                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.local_cfg,
                        gs.ff_tables,
                    );
                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                }
            }

            var next_index = std.atomic.Value(usize).init(0);
            var stop_flag = std.atomic.Value(u8).init(0);
            var worker_err: ?anyerror = null;
            var err_mutex = std.Io.Mutex.init;
            var log_mutex = std.Io.Mutex.init;

            var qshared = QPointPertShared{
                .alloc = alloc,
                .io = io,
                .kpts = kpts,
                .dfpt_cfg = &pert_dfpt_cfg,
                .q_cart = q_cart,
                .grid = grid,
                .gs = gs,
                .species = species,
                .atoms = atoms,
                .ff_tables = gs.ff_tables,
                .rho_core_tables = gs.rho_core_tables,
                .pert_results_mk = pert_results_mk,
                .vloc1_gs = vloc1_gs,
                .rho1_core_gs = rho1_core_gs,
                .dim = n_irr_perts,
                .irr_pert_indices = null,
                .next_index = &next_index,
                .stop = &stop_flag,
                .err = &worker_err,
                .err_mutex = &err_mutex,
                .log_mutex = &log_mutex,
            };

            // Build irr_pert_indices for atom-level reduction
            const irr_pert_indices = try alloc.alloc(usize, n_irr_perts);
            defer alloc.free(irr_pert_indices);
            {
                var pi: usize = 0;
                for (irr_info.irr_atom_indices) |ia| {
                    for (0..3) |dir_idx| {
                        irr_pert_indices[pi] = 3 * ia + dir_idx;
                        pi += 1;
                    }
                }
            }
            qshared.irr_pert_indices = irr_pert_indices;

            var workers = try alloc.alloc(QPointPertWorker, pert_thread_count);
            defer alloc.free(workers);
            var threads_arr = try alloc.alloc(std.Thread, pert_thread_count - 1);
            defer alloc.free(threads_arr);

            for (0..pert_thread_count) |ti| {
                workers[ti] = .{ .shared = &qshared, .thread_index = ti };
            }

            for (0..pert_thread_count - 1) |ti| {
                threads_arr[ti] = try std.Thread.spawn(
                    .{},
                    qpointPertWorkerFn,
                    .{&workers[ti + 1]},
                );
            }

            qpointPertWorkerFn(&workers[0]);

            for (threads_arr) |t| {
                t.join();
            }

            if (worker_err) |e| return e;
        }

        // Build complex dynamical matrix from all contributions (multi-k version)
        const dyn_q = try buildQDynmatMultiK(
            alloc,
            kpts,
            pert_results_mk,
            vloc1_gs,
            rho1_core_gs,
            rho0_g,
            gs,
            ionic.charges,
            ionic.positions,
            cell_bohr,
            recip,
            volume,
            q_cart,
            grid,
            species,
            atoms,
            gs.ff_tables,
            gs.rho_core_tables,
            gs.rho_core,
            vxc_g,
            cfg.vdw,
            irr_info,
        );
        defer alloc.free(dyn_q);

        // Reconstruct non-irreducible columns from symmetry
        if (irr_info.n_irr_atoms < n_atoms) {
            dynmat_mod.reconstructDynmatColumnsComplex(
                dyn_q,
                n_atoms,
                irr_info,
                symops,
                sym_data.indsym,
                sym_data.tnons_shift,
                cell_bohr,
                qf,
            );
        }

        // Apply acoustic sum rule at Γ-point (q=0)
        if (q_norm < 1e-10) {
            dynmat_mod.applyASRComplex(dyn_q, n_atoms);
        }

        // Mass-weight
        dynmat_mod.massWeightComplex(dyn_q, n_atoms, ionic.masses);

        // Diagonalize complex Hermitian matrix
        var result_q = try dynmat_mod.diagonalizeComplex(alloc, dyn_q, dim);
        defer result_q.deinit(alloc);

        frequencies[iq] = try alloc.alloc(f64, dim);
        @memcpy(frequencies[iq], result_q.frequencies_cm1);
        freq_count = iq + 1;

        logDfptInfo("dfpt_band: q[{d}] freqs:", .{iq});
        for (result_q.frequencies_cm1) |f| {
            logDfptInfo(" {d:.1}", .{f});
        }
        logDfptInfo("\n", .{});
    }

    return PhononBandResult{
        .distances = q_path_data.distances,
        .frequencies = frequencies,
        .n_modes = dim,
        .n_q = n_q,
        .labels = q_path_data.labels,
        .label_positions = q_path_data.label_positions,
    };
}

// =====================================================================
// Perturbation parallelism for q≠0
// =====================================================================

pub const QPointPertShared = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    dfpt_cfg: *const DfptConfig,
    q_cart: math.Vec3,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    pert_results_mk: []MultiKPertResult,
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

pub const QPointPertWorker = struct {
    shared: *QPointPertShared,
    thread_index: usize,
};

pub fn setQPointPertError(shared: *QPointPertShared, e: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);
    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

pub fn qpointPertWorkerFn(worker: *QPointPertWorker) void {
    const shared = worker.shared;
    const alloc = shared.alloc;

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const work_idx = shared.next_index.fetchAdd(1, .acq_rel);
        if (work_idx >= shared.dim) break;

        // Map work index to actual perturbation index
        const idx = if (shared.irr_pert_indices) |indices| indices[work_idx] else work_idx;
        const ia = idx / 3;
        const dir = idx % 3;

        {
            shared.log_mutex.lockUncancelable(shared.io);
            defer shared.log_mutex.unlock(shared.io);
            const dir_names = [_][]const u8{ "x", "y", "z" };
            logDfpt(
                "dfpt_band: [thread {d}] solving perturbation" ++
                    " atom={d} dir={s} ({d}/{d})\n",
                .{
                    worker.thread_index,
                    ia,
                    dir_names[dir],
                    work_idx + 1,
                    shared.dim,
                },
            );
        }

        // Solve perturbation SCF with all k-points (vloc1/rho1_core already built by caller)
        shared.pert_results_mk[idx] = solvePerturbationQMultiK(
            alloc,
            shared.io,
            shared.kpts,
            ia,
            dir,
            shared.dfpt_cfg.*,
            shared.q_cart,
            shared.grid,
            shared.gs,
            shared.species,
            shared.atoms,
            shared.ff_tables,
            shared.rho_core_tables,
        ) catch |e| {
            setQPointPertError(shared, e);
            shared.stop.store(1, .release);
            break;
        };

        // Debug: print rho1 norm
        {
            var rho1_norm: f64 = 0.0;
            for (shared.pert_results_mk[idx].rho1_g) |c| {
                rho1_norm += c.r * c.r + c.i * c.i;
            }
            shared.log_mutex.lockUncancelable(shared.io);
            defer shared.log_mutex.unlock(shared.io);
            logDfpt("dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n", .{ ia, dir, @sqrt(rho1_norm) });
        }
    }
}
