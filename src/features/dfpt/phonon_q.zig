//! Finite-q DFPT phonon band structure calculation.
//!
//! Contains the q≠0 perturbation solver, cross-basis operations,
//! complex dynamical matrix construction, q-path generation,
//! and the top-level `runPhononBand` entry point.

const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../scf/scf.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const d3 = @import("../vdw/d3.zig");
const d3_params = @import("../vdw/d3_params.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const config_mod = @import("../config/config.zig");
const model_mod = @import("../dft/model.zig");
const iterative = @import("../linalg/iterative.zig");
const symmetry_mod = @import("../symmetry/symmetry.zig");

const dfpt = @import("dfpt.zig");
const perturbation = dfpt.perturbation;
const sternheimer = dfpt.sternheimer;
const ewald2 = dfpt.ewald2;
const dynmat_mod = dfpt.dynmat;
const dynmat_contrib = dfpt.dynmat_contrib;
const gamma = dfpt.gamma;

const GroundState = dfpt.GroundState;
const PreparedGroundState = dfpt.PreparedGroundState;
const DfptConfig = dfpt.DfptConfig;
const IonicData = dfpt.IonicData;
const PerturbationResult = dfpt.PerturbationResult;
const logDfpt = dfpt.logDfpt;
const logDfptInfo = dfpt.logDfptInfo;

const kpoints_mod = @import("../kpoints/kpoints.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const mesh_mod = @import("../kpoints/mesh.zig");
const reduction = @import("../kpoints/reduction.zig");
const wfn_rot = @import("../symmetry/wavefunction_rotation.zig");

// Submodules (split out of phonon_q.zig)
const qpath_mod = @import("phonon_q/qpath.zig");
const cross_basis = @import("phonon_q/cross_basis.zig");
const dynmat_elem_q = @import("phonon_q/dynmat_elem_q.zig");
const kpt_gs_mod = @import("phonon_q/kpt_gs.zig");
const kpt_dfpt_mod = @import("phonon_q/kpt_dfpt.zig");
const solver_single_mod = @import("phonon_q/solver_single.zig");
const solver_multik_mod = @import("phonon_q/solver_multik.zig");
const dynmat_build_mod = @import("phonon_q/dynmat_build.zig");
const band_direct_mod = @import("phonon_q/band_direct.zig");

pub const generateFccQPath = qpath_mod.generateFccQPath;
pub const generateQPathFromConfig = qpath_mod.generateQPathFromConfig;

const applyV1PsiQ = cross_basis.applyV1PsiQ;
pub const applyV1PsiQCached = cross_basis.applyV1PsiQCached;
const computeRho1Q = cross_basis.computeRho1Q;
pub const computeRho1QCached = cross_basis.computeRho1QCached;
const complexRealToReciprocal = cross_basis.complexRealToReciprocal;

pub const computeElecDynmatElementQ = dynmat_elem_q.computeElecDynmatElementQ;
pub const computeNonlocalResponseDynmatQ = dynmat_elem_q.computeNonlocalResponseDynmatQ;
pub const computeNlccCrossDynmatQ = dynmat_elem_q.computeNlccCrossDynmatQ;

pub const KPointGsData = kpt_gs_mod.KPointGsData;
pub const prepareFullBZKpoints = kpt_gs_mod.prepareFullBZKpoints;
pub const prepareFullBZKpointsFromIBZ = kpt_gs_mod.prepareFullBZKpointsFromIBZ;

pub const KPointDfptData = kpt_dfpt_mod.KPointDfptData;
pub const MultiKPertResult = kpt_dfpt_mod.MultiKPertResult;
const buildKPointDfptDataFromGS = kpt_dfpt_mod.buildKPointDfptDataFromGS;

pub const solvePerturbationQ = solver_single_mod.solvePerturbationQ;
pub const solvePerturbationQMultiK = solver_multik_mod.solvePerturbationQMultiK;

pub const buildQDynmat = dynmat_build_mod.buildQDynmat;
const buildQDynmatMultiK = dynmat_build_mod.buildQDynmatMultiK;
pub const computeNonlocalResponseDynmatQMultiK = dynmat_build_mod.computeNonlocalResponseDynmatQMultiK;
pub const computeNonlocalSelfDynmatMultiK = dynmat_build_mod.computeNonlocalSelfDynmatMultiK;

pub const PhononBandResult = band_direct_mod.PhononBandResult;
pub const runPhononBand = band_direct_mod.runPhononBand;

// Re-exports used by IFC band path (§12 still lives in phonon_q.zig)
const QPointPertShared = band_direct_mod.QPointPertShared;
const QPointPertWorker = band_direct_mod.QPointPertWorker;
const qpointPertWorkerFn = band_direct_mod.qpointPertWorkerFn;

const Grid = scf_mod.Grid;


// =====================================================================
// K-point ground-state data (§1) — moved to phonon_q/kpt_gs.zig
// Multi-k-point DFPT data    (§2) — moved to phonon_q/kpt_dfpt.zig
// =====================================================================

// =====================================================================
// Cross-basis operations — moved to phonon_q/cross_basis.zig
// =====================================================================

// =====================================================================
// Complex dynmat element computations — moved to phonon_q/dynmat_elem_q.zig
// =====================================================================

// =====================================================================
// DFPT SCF solver for q≠0 (§5) — moved to phonon_q/solver_single.zig
// =====================================================================

// =====================================================================
// Multi-k-point DFPT perturbation solver (§6) — moved to phonon_q/solver_multik.zig
// =====================================================================
// Dynamical matrix construction (§7+§8) — moved to phonon_q/dynmat_build.zig
// =====================================================================
// =====================================================================
// Q-path generation — moved to phonon_q/qpath.zig (see re-exports above)
// =====================================================================

// =====================================================================
// =====================================================================
// Phonon band structure entry point (§10+§11) — moved to phonon_q/band_direct.zig
// =====================================================================
// =====================================================================
// IFC-interpolated phonon band structure
// =====================================================================

const ifc_mod = @import("ifc.zig");

/// Run DFPT phonon band structure using IFC interpolation.
/// 1. Compute D(q) on a coarse q-grid via DFPT
/// 2. Fourier transform to IFC: C(R)
/// 3. Interpolate D(q') at arbitrary q-path points
pub fn runPhononBandIFC(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    model: *const model_mod.Model,
) !PhononBandResult {
    const species = model.species;
    const atoms = model.atoms;
    const cell_bohr = model.cell_bohr;
    const recip = model.recip;
    const volume = model.volume_bohr;
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const grid = scf_result.grid;
    const qgrid = cfg.dfpt.qgrid orelse return error.MissingQgrid;

    logDfptInfo("dfpt_ifc: starting IFC phonon band ({d} atoms, qgrid={d}x{d}x{d})\n", .{ n_atoms, qgrid[0], qgrid[1], qgrid[2] });

    // Prepare ground state
    var prepared = try dfpt.prepareGroundState(alloc, io, cfg, scf_result, species, atoms, volume, recip);
    defer prepared.deinit();
    const gs = prepared.gs;

    // Ionic data for dynmat construction
    const ionic = try IonicData.init(alloc, species, atoms);
    defer ionic.deinit(alloc);

    // Ground-state density in G-space
    const rho0_g = try scf_mod.realToReciprocal(alloc, grid, scf_result.density, false);
    defer alloc.free(rho0_g);

    // V_xc(G) for NLCC self-energy
    var vxc_g: ?[]math.Complex = null;
    if (prepared.vxc_r) |v| {
        vxc_g = try scf_mod.realToReciprocal(alloc, grid, v, false);
    }
    defer if (vxc_g) |v| alloc.free(v);

    // Build symmetry operations for dynmat symmetrization
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    defer alloc.free(symops);
    logDfptInfo("dfpt_ifc: {d} symmetry operations found\n", .{symops.len});

    const sym_data = try dynmat_mod.buildIndsym(alloc, symops, atoms, recip, 1e-5);
    defer {
        for (sym_data.indsym) |row| alloc.free(row);
        alloc.free(sym_data.indsym);
        for (sym_data.tnons_shift) |row| alloc.free(row);
        alloc.free(sym_data.tnons_shift);
    }

    const dfpt_cfg = DfptConfig.fromConfig(cfg);

    // Prepare full-BZ k-point ground-state data
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

    // =============================================================
    // Phase 1: Compute D(q) on the coarse q-grid
    // =============================================================
    const shift_zero = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const qgrid_points = try mesh_mod.generateKmesh(alloc, qgrid, recip, shift_zero);
    defer alloc.free(qgrid_points);
    const n_qgrid = qgrid_points.len;
    logDfptInfo("dfpt_ifc: {d} q-grid points for DFPT\n", .{n_qgrid});

    // Store fractional q-points and dynamical matrices
    var q_frac_grid = try alloc.alloc(math.Vec3, n_qgrid);
    defer alloc.free(q_frac_grid);
    var dynmat_grid = try alloc.alloc([]math.Complex, n_qgrid);
    var dynmat_count: usize = 0;
    defer {
        for (0..dynmat_count) |i| alloc.free(dynmat_grid[i]);
        alloc.free(dynmat_grid);
    }

    for (0..n_qgrid) |iq| {
        const q_cart = qgrid_points[iq].k_cart;
        const qf = qgrid_points[iq].k_frac;
        const q_norm = math.Vec3.norm(q_cart);

        q_frac_grid[iq] = qf;

        logDfpt("dfpt_ifc: q_grid[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n", .{ iq, qf.x, qf.y, qf.z, q_norm });
        logDfpt("dfpt_ifc: q_grid[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });

        // Find irreducible atoms for this q-point
        var irr_info = try dynmat_mod.findIrreducibleAtoms(alloc, symops, sym_data.indsym, n_atoms, qf);
        defer irr_info.deinit(alloc);
        logDfpt("dfpt_ifc: q_grid[{d}] {d}/{d} irreducible atoms\n", .{ iq, irr_info.n_irr_atoms, n_atoms });

        // Find irreducible perturbations (atom+direction) for this q-point

        // Build k+q data for this q-point
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

        // Solve perturbations
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
            // Phase 1: Build vloc1, rho1_core for ALL perturbations (cheap)
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

            // Phase 2: Solve perturbation SCF for irreducible atoms only
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
                }
            }
        } else {
            const n_irr_perts = irr_info.n_irr_atoms * 3;
            var pert_dfpt_cfg = dfpt_cfg;
            pert_dfpt_cfg.kpoint_threads = dfpt.kpointThreadsForPertParallel(pert_thread_count, dfpt_cfg.kpoint_threads);

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
                threads_arr[ti] = try std.Thread.spawn(.{}, qpointPertWorkerFn, .{&workers[ti + 1]});
            }

            qpointPertWorkerFn(&workers[0]);

            for (threads_arr) |t| {
                t.join();
            }

            if (worker_err) |e| return e;
        }

        // Build dynamical matrix D(q) for this q-grid point
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

        // Reconstruct non-irreducible columns from symmetry
        if (irr_info.n_irr_atoms < n_atoms) {
            dynmat_mod.reconstructDynmatColumnsComplex(dyn_q, n_atoms, irr_info, symops, sym_data.indsym, sym_data.tnons_shift, cell_bohr, qf);
        }

        dynmat_grid[iq] = dyn_q;
        dynmat_count = iq + 1;

        logDfpt("dfpt_ifc: q_grid[{d}] D(q) computed\n", .{iq});
    }

    // =============================================================
    // Phase 2: Compute IFC: C(R) = FT[D(q)]
    // =============================================================
    logDfptInfo("dfpt_ifc: computing IFC from {d} q-grid points\n", .{n_qgrid});

    // Cast dynmat_grid to const slices for computeIFC
    var dynmat_const = try alloc.alloc([]const math.Complex, n_qgrid);
    defer alloc.free(dynmat_const);
    for (0..n_qgrid) |i| {
        dynmat_const[i] = dynmat_grid[i];
    }

    var ifc_data = try ifc_mod.computeIFC(alloc, dynmat_const, q_frac_grid, qgrid, n_atoms);
    defer ifc_data.deinit(alloc);

    // Apply ASR in IFC space
    ifc_mod.applyASR(&ifc_data);
    logDfptInfo("dfpt_ifc: IFC ASR applied\n", .{});

    // =============================================================
    // Phase 3: Generate q-path and interpolate
    // =============================================================
    const npoints_per_seg = cfg.dfpt.qpath_npoints;

    // Use custom q-path if specified, otherwise FCC default
    var q_points_frac: []math.Vec3 = undefined;
    var q_points_cart: []math.Vec3 = undefined;
    var qpath_distances: []f64 = undefined;
    var qpath_labels: [][]const u8 = undefined;
    var qpath_label_positions: []usize = undefined;

    if (cfg.dfpt.qpath.len >= 2) {
        const qpath = try generateQPathFromConfig(alloc, cfg.dfpt.qpath, npoints_per_seg, recip);
        q_points_frac = qpath.q_points_frac;
        q_points_cart = qpath.q_points_cart;
        qpath_distances = qpath.distances;
        qpath_labels = qpath.labels;
        qpath_label_positions = qpath.label_positions;
    } else {
        const qpath = try generateFccQPath(alloc, recip, npoints_per_seg);
        q_points_frac = qpath.q_points_frac;
        q_points_cart = qpath.q_points_cart;
        qpath_distances = qpath.distances;
        qpath_labels = qpath.labels;
        qpath_label_positions = qpath.label_positions;
    }
    defer alloc.free(q_points_frac);
    defer alloc.free(q_points_cart);

    const n_q = q_points_cart.len;
    logDfptInfo("dfpt_ifc: interpolating {d} q-path points\n", .{n_q});

    // Allocate result arrays
    var frequencies = try alloc.alloc([]f64, n_q);
    var freq_count: usize = 0;
    errdefer {
        for (0..freq_count) |i| alloc.free(frequencies[i]);
        alloc.free(frequencies);
    }

    for (0..n_q) |iq| {
        const qf = q_points_frac[iq];
        const q_norm = math.Vec3.norm(q_points_cart[iq]);

        // Interpolate D(q') from IFC
        const dyn_interp = try ifc_mod.interpolate(alloc, &ifc_data, qf);
        defer alloc.free(dyn_interp);

        // Apply ASR at Gamma
        if (q_norm < 1e-10) {
            dynmat_mod.applyASRComplex(dyn_interp, n_atoms);
        }

        // Mass-weight
        dynmat_mod.massWeightComplex(dyn_interp, n_atoms, ionic.masses);

        // Diagonalize
        var result_q = try dynmat_mod.diagonalizeComplex(alloc, dyn_interp, dim);
        defer result_q.deinit(alloc);

        frequencies[iq] = try alloc.alloc(f64, dim);
        @memcpy(frequencies[iq], result_q.frequencies_cm1);
        freq_count = iq + 1;

        if (iq % 10 == 0 or iq == n_q - 1) {
            logDfptInfo("dfpt_ifc: q[{d}] freqs:", .{iq});
            for (result_q.frequencies_cm1) |f| {
                logDfptInfo(" {d:.1}", .{f});
            }
            logDfptInfo("\n", .{});
        }
    }

    // =============================================================
    // Phase 4 (optional): Phonon DOS from IFC interpolation
    // =============================================================
    if (cfg.dfpt.dos_qmesh) |dos_qmesh| {
        const phonon_dos_mod = @import("phonon_dos.zig");
        logDfptInfo("dfpt_ifc: computing phonon DOS on {d}x{d}x{d} mesh\n", .{ dos_qmesh[0], dos_qmesh[1], dos_qmesh[2] });

        var pdos = try phonon_dos_mod.computePhononDos(
            alloc,
            &ifc_data,
            ionic.masses,
            n_atoms,
            dos_qmesh,
            cfg.dfpt.dos_sigma,
            cfg.dfpt.dos_nbin,
        );
        defer pdos.deinit(alloc);

        // Write to out_dir
        var out_dir = try std.Io.Dir.cwd().openDir(io, cfg.out_dir, .{});
        defer out_dir.close(io);
        try phonon_dos_mod.writePhononDosCsv(io, out_dir, pdos);
        logDfptInfo("dfpt_ifc: phonon DOS written to phonon_dos.csv\n", .{});
    }

    return PhononBandResult{
        .distances = qpath_distances,
        .frequencies = frequencies,
        .n_modes = dim,
        .n_q = n_q,
        .labels = qpath_labels,
        .label_positions = qpath_label_positions,
    };
}
