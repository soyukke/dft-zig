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

pub const BandGroundStateData = struct {
    prepared: dfpt.PreparedGroundState,
    ionic: IonicData,
    rho0_g: []math.Complex,
    vxc_g: ?[]math.Complex,

    fn deinit(self: *BandGroundStateData, alloc: std.mem.Allocator) void {
        if (self.vxc_g) |values| {
            alloc.free(values);
        }
        alloc.free(self.rho0_g);
        self.ionic.deinit(alloc);
        self.prepared.deinit();
    }
};

pub const BandSymmetryData = struct {
    symops: []symmetry_mod.SymOp,
    indsym: [][]usize,
    tnons_shift: [][]math.Vec3,

    fn deinit(self: *BandSymmetryData, alloc: std.mem.Allocator) void {
        deinitSymData(alloc, self.*);
        alloc.free(self.symops);
    }
};

pub const QPointPertBuffers = struct {
    pert_results_mk: []MultiKPertResult,
    vloc1_gs: [][]math.Complex,
    rho1_core_gs: [][]math.Complex,

    fn init(alloc: std.mem.Allocator, dim: usize) !QPointPertBuffers {
        const pert_results_mk = try alloc.alloc(MultiKPertResult, dim);
        errdefer alloc.free(pert_results_mk);
        const vloc1_gs = try alloc.alloc([]math.Complex, dim);
        errdefer alloc.free(vloc1_gs);
        const rho1_core_gs = try alloc.alloc([]math.Complex, dim);
        errdefer alloc.free(rho1_core_gs);

        for (0..dim) |i| {
            pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
            vloc1_gs[i] = &.{};
            rho1_core_gs[i] = &.{};
        }

        return .{
            .pert_results_mk = pert_results_mk,
            .vloc1_gs = vloc1_gs,
            .rho1_core_gs = rho1_core_gs,
        };
    }

    fn deinit(self: *QPointPertBuffers, alloc: std.mem.Allocator) void {
        for (self.pert_results_mk) |*result| {
            if (result.rho1_g.len > 0 or result.psi1_per_k.len > 0) {
                result.deinit(alloc);
            }
        }
        alloc.free(self.pert_results_mk);
        for (self.vloc1_gs) |values| {
            if (values.len > 0) alloc.free(values);
        }
        alloc.free(self.vloc1_gs);
        for (self.rho1_core_gs) |values| {
            if (values.len > 0) alloc.free(values);
        }
        alloc.free(self.rho1_core_gs);
    }
};

pub fn deinitKPointDfptData(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
) void {
    for (kpts) |*kd| kd.deinitQOnly(alloc);
    alloc.free(kpts);
}

pub fn deinitKPointGsData(
    alloc: std.mem.Allocator,
    kgs_data: []kpt_gs.KPointGsData,
) void {
    for (kgs_data) |*kg| kg.deinit(alloc);
    alloc.free(kgs_data);
}

pub fn deinitSymData(
    alloc: std.mem.Allocator,
    sym_data: anytype,
) void {
    for (sym_data.indsym) |row| alloc.free(row);
    alloc.free(sym_data.indsym);
    for (sym_data.tnons_shift) |row| alloc.free(row);
    alloc.free(sym_data.tnons_shift);
}

pub fn initBandGroundStateData(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    volume: f64,
    recip: math.Mat3,
    grid: Grid,
) !BandGroundStateData {
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
    errdefer prepared.deinit();

    const ionic = try IonicData.init(alloc, species, atoms);
    errdefer ionic.deinit(alloc);

    const rho0_g = try scf_mod.realToReciprocal(alloc, grid, scf_result.density, false);
    errdefer alloc.free(rho0_g);

    var vxc_g: ?[]math.Complex = null;
    if (prepared.vxc_r) |values| {
        vxc_g = try scf_mod.realToReciprocal(alloc, grid, values, false);
    }
    errdefer if (vxc_g) |values| alloc.free(values);

    return .{
        .prepared = prepared,
        .ionic = ionic,
        .rho0_g = rho0_g,
        .vxc_g = vxc_g,
    };
}

pub fn initBandSymmetryData(
    alloc: std.mem.Allocator,
    cell_bohr: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
) !BandSymmetryData {
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    errdefer alloc.free(symops);
    logDfptInfo("dfpt_band: {d} symmetry operations found\n", .{symops.len});

    const sym_data = try dynmat_mod.buildIndsym(alloc, symops, atoms, recip, 1e-5);
    errdefer deinitSymData(alloc, sym_data);

    return .{
        .symops = symops,
        .indsym = sym_data.indsym,
        .tnons_shift = sym_data.tnons_shift,
    };
}

pub fn prepareBandKgsData(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    gs: *const GroundState,
    local_r: []const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
) ![]kpt_gs.KPointGsData {
    return prepareFullBZKpoints(
        alloc,
        io,
        cfg,
        gs,
        local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
    );
}

fn buildAnalyticPerturbations(
    alloc: std.mem.Allocator,
    grid: Grid,
    atoms: []const hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
    q_cart: math.Vec3,
    gs: GroundState,
    buffers: *QPointPertBuffers,
) !void {
    for (0..atoms.len) |ia| {
        for (0..3) |dir| {
            const pidx = 3 * ia + dir;
            buffers.vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                alloc,
                grid,
                atoms[ia],
                species,
                dir,
                q_cart,
                gs.local_cfg,
                gs.ff_tables,
            );
            buffers.rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
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
}

fn logPerturbationRho1Norm(
    ia: usize,
    dir: usize,
    result: *const MultiKPertResult,
) void {
    var rho1_norm: f64 = 0.0;
    for (result.rho1_g) |c| {
        rho1_norm += c.r * c.r + c.i * c.i;
    }
    logDfpt(
        "dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n",
        .{ ia, dir, @sqrt(rho1_norm) },
    );
}

fn solveSequentialQPointPerturbations(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    q_cart: math.Vec3,
    dfpt_cfg: DfptConfig,
    kpts: []KPointDfptData,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    buffers: *QPointPertBuffers,
) !void {
    try buildAnalyticPerturbations(alloc, grid, atoms, species, q_cart, gs, buffers);
    for (irr_info.irr_atom_indices) |ia| {
        for (0..3) |dir| {
            const pidx = 3 * ia + dir;
            buffers.pert_results_mk[pidx] = try solvePerturbationQMultiK(
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
            logPerturbationRho1Norm(ia, dir, &buffers.pert_results_mk[pidx]);
        }
    }
}

fn logParallelPerturbationPlan(
    pert_thread_count: usize,
    kpoint_threads: usize,
    dim: usize,
    n_irr_perts: usize,
) void {
    logDfptInfo(
        "dfpt_band: using {d} pert threads × {d} kpt threads" ++
            " for {d} perturbations ({d} irreducible)\n",
        .{ pert_thread_count, kpoint_threads, dim, n_irr_perts },
    );
}

fn buildIrreduciblePerturbationIndices(
    alloc: std.mem.Allocator,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]usize {
    const irr_pert_indices = try alloc.alloc(usize, irr_info.n_irr_atoms * 3);
    var pi: usize = 0;
    for (irr_info.irr_atom_indices) |ia| {
        for (0..3) |dir_idx| {
            irr_pert_indices[pi] = 3 * ia + dir_idx;
            pi += 1;
        }
    }
    return irr_pert_indices;
}

fn initQPointShared(
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    dfpt_cfg: *const DfptConfig,
    q_cart: math.Vec3,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    buffers: *QPointPertBuffers,
    dim: usize,
    next_index: *std.atomic.Value(usize),
    stop_flag: *std.atomic.Value(u8),
    worker_err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
) QPointPertShared {
    return .{
        .alloc = alloc,
        .io = io,
        .kpts = kpts,
        .dfpt_cfg = dfpt_cfg,
        .q_cart = q_cart,
        .grid = grid,
        .gs = gs,
        .species = species,
        .atoms = atoms,
        .ff_tables = gs.ff_tables,
        .rho_core_tables = gs.rho_core_tables,
        .pert_results_mk = buffers.pert_results_mk,
        .vloc1_gs = buffers.vloc1_gs,
        .rho1_core_gs = buffers.rho1_core_gs,
        .dim = dim,
        .irr_pert_indices = null,
        .next_index = next_index,
        .stop = stop_flag,
        .err = worker_err,
        .err_mutex = err_mutex,
        .log_mutex = log_mutex,
    };
}

fn runParallelPerturbationWorkers(
    alloc: std.mem.Allocator,
    qshared: *QPointPertShared,
    pert_thread_count: usize,
) !void {
    var workers = try alloc.alloc(QPointPertWorker, pert_thread_count);
    defer alloc.free(workers);

    var threads_arr = try alloc.alloc(std.Thread, pert_thread_count - 1);
    defer alloc.free(threads_arr);

    for (0..pert_thread_count) |ti| {
        workers[ti] = .{ .shared = qshared, .thread_index = ti };
    }
    for (0..pert_thread_count - 1) |ti| {
        threads_arr[ti] = try std.Thread.spawn(.{}, qpointPertWorkerFn, .{&workers[ti + 1]});
    }
    qpointPertWorkerFn(&workers[0]);
    for (threads_arr) |t| t.join();
}

fn solveParallelQPointPerturbations(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    q_cart: math.Vec3,
    dfpt_cfg: DfptConfig,
    pert_thread_count: usize,
    kpts: []KPointDfptData,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    buffers: *QPointPertBuffers,
) !void {
    const dim = 3 * atoms.len;
    const n_irr_perts = irr_info.n_irr_atoms * 3;
    var pert_dfpt_cfg = dfpt_cfg;
    pert_dfpt_cfg.kpoint_threads = dfpt.kpointThreadsForPertParallel(
        pert_thread_count,
        dfpt_cfg.kpoint_threads,
    );
    logParallelPerturbationPlan(
        pert_thread_count,
        pert_dfpt_cfg.kpoint_threads,
        dim,
        n_irr_perts,
    );

    try buildAnalyticPerturbations(alloc, grid, atoms, species, q_cart, gs, buffers);

    var next_index = std.atomic.Value(usize).init(0);
    var stop_flag = std.atomic.Value(u8).init(0);
    var worker_err: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;
    var qshared = initQPointShared(
        alloc,
        io,
        kpts,
        &pert_dfpt_cfg,
        q_cart,
        grid,
        gs,
        species,
        atoms,
        buffers,
        n_irr_perts,
        &next_index,
        &stop_flag,
        &worker_err,
        &err_mutex,
        &log_mutex,
    );

    const irr_pert_indices = try buildIrreduciblePerturbationIndices(alloc, irr_info);
    defer alloc.free(irr_pert_indices);

    qshared.irr_pert_indices = irr_pert_indices;
    try runParallelPerturbationWorkers(alloc, &qshared, pert_thread_count);
    if (worker_err) |e| return e;
}

pub fn solveQPointPerturbations(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    q_cart: math.Vec3,
    dfpt_cfg: DfptConfig,
    pert_thread_count: usize,
    kpts: []KPointDfptData,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) !QPointPertBuffers {
    var buffers = try QPointPertBuffers.init(alloc, 3 * atoms.len);
    errdefer buffers.deinit(alloc);

    if (pert_thread_count <= 1) {
        try solveSequentialQPointPerturbations(
            alloc,
            io,
            grid,
            gs,
            species,
            atoms,
            q_cart,
            dfpt_cfg,
            kpts,
            irr_info,
            &buffers,
        );
        return buffers;
    }

    try solveParallelQPointPerturbations(
        alloc,
        io,
        grid,
        gs,
        species,
        atoms,
        q_cart,
        dfpt_cfg,
        pert_thread_count,
        kpts,
        irr_info,
        &buffers,
    );
    return buffers;
}

fn finalizeQPointFrequencies(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    gs: GroundState,
    ionic: *const IonicData,
    rho0_g: []const math.Complex,
    vxc_g: ?[]const math.Complex,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    q_cart: math.Vec3,
    qf: math.Vec3,
    q_norm: f64,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    symops: []const symmetry_mod.SymOp,
    indsym: []const []const usize,
    tnons_shift: []const []const math.Vec3,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    kpts: []KPointDfptData,
    buffers: *const QPointPertBuffers,
) ![]f64 {
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const dyn_q = try buildQDynmatMultiK(
        alloc,
        kpts,
        buffers.pert_results_mk,
        buffers.vloc1_gs,
        buffers.rho1_core_gs,
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

    if (irr_info.n_irr_atoms < n_atoms) {
        dynmat_mod.reconstructDynmatColumnsComplex(
            dyn_q,
            n_atoms,
            irr_info,
            symops,
            indsym,
            tnons_shift,
            cell_bohr,
            qf,
        );
    }
    if (q_norm < 1e-10) dynmat_mod.applyASRComplex(dyn_q, n_atoms);
    dynmat_mod.massWeightComplex(dyn_q, n_atoms, ionic.masses);

    var result_q = try dynmat_mod.diagonalizeComplex(alloc, dyn_q, dim);
    defer result_q.deinit(alloc);

    const frequencies = try alloc.alloc(f64, dim);
    @memcpy(frequencies, result_q.frequencies_cm1);

    return frequencies;
}

fn logQPointSummary(
    iq: usize,
    qf: math.Vec3,
    q_norm: f64,
    n_kpts: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    n_atoms: usize,
    frequencies: []const f64,
) void {
    logDfpt(
        "dfpt_band: q[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n",
        .{ iq, qf.x, qf.y, qf.z, q_norm },
    );
    logDfpt("dfpt_band: q[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });
    logDfpt(
        "dfpt_band: q[{d}] {d}/{d} irreducible atoms\n",
        .{ iq, irr_info.n_irr_atoms, n_atoms },
    );
    logDfptInfo("dfpt_band: q[{d}] freqs:", .{iq});
    for (frequencies) |f| {
        logDfptInfo(" {d:.1}", .{f});
    }
    logDfptInfo("\n", .{});
}

fn solveBandQPoint(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    dfpt_cfg: DfptConfig,
    gs: GroundState,
    local_r: []const f64,
    ionic: *const IonicData,
    rho0_g: []const math.Complex,
    vxc_g: ?[]const math.Complex,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    symops: []const symmetry_mod.SymOp,
    indsym: []const []const usize,
    tnons_shift: []const []const math.Vec3,
    kgs_data: []const kpt_gs.KPointGsData,
    iq: usize,
    q_cart: math.Vec3,
    qf: math.Vec3,
) ![]f64 {
    const n_atoms = atoms.len;
    const q_norm = math.Vec3.norm(q_cart);
    const irr_info = try dynmat_mod.findIrreducibleAtoms(
        alloc,
        symops,
        indsym,
        n_atoms,
        qf,
    );
    defer @constCast(&irr_info).deinit(alloc);

    const pert_thread_count = dfpt.perturbationThreadCount(
        3 * n_atoms,
        dfpt_cfg.perturbation_threads,
    );
    const kpts = try buildKPointDfptDataFromGS(
        alloc,
        io,
        kgs_data,
        q_cart,
        q_norm,
        cfg,
        local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
        pert_thread_count,
    );
    defer deinitKPointDfptData(alloc, kpts);

    var buffers = try solveQPointPerturbations(
        alloc,
        io,
        grid,
        gs,
        species,
        atoms,
        q_cart,
        dfpt_cfg,
        pert_thread_count,
        kpts,
        irr_info,
    );
    defer buffers.deinit(alloc);

    const frequencies = try finalizeQPointFrequencies(
        alloc,
        cfg,
        gs,
        ionic,
        rho0_g,
        vxc_g,
        cell_bohr,
        recip,
        volume,
        q_cart,
        qf,
        q_norm,
        grid,
        species,
        atoms,
        symops,
        indsym,
        tnons_shift,
        irr_info,
        kpts,
        &buffers,
    );
    logQPointSummary(iq, qf, q_norm, kgs_data.len, irr_info, n_atoms, frequencies);
    return frequencies;
}

fn solveBandQPoints(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    dfpt_cfg: DfptConfig,
    gs: GroundState,
    local_r: []const f64,
    ionic: *const IonicData,
    rho0_g: []const math.Complex,
    vxc_g: ?[]const math.Complex,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    sym_data: *const BandSymmetryData,
    kgs_data: []const kpt_gs.KPointGsData,
    q_points_cart: []const math.Vec3,
    q_points_frac: []const math.Vec3,
) ![][]f64 {
    const n_q = q_points_cart.len;
    var frequencies = try alloc.alloc([]f64, n_q);
    var freq_count: usize = 0;
    errdefer {
        for (0..freq_count) |i| alloc.free(frequencies[i]);
        alloc.free(frequencies);
    }

    for (0..n_q) |iq| {
        frequencies[iq] = try solveBandQPoint(
            alloc,
            io,
            cfg,
            dfpt_cfg,
            gs,
            local_r,
            ionic,
            rho0_g,
            vxc_g,
            species,
            atoms,
            cell_bohr,
            recip,
            volume,
            grid,
            sym_data.symops,
            sym_data.indsym,
            sym_data.tnons_shift,
            kgs_data,
            iq,
            q_points_cart[iq],
            q_points_frac[iq],
        );
        freq_count = iq + 1;
    }

    return frequencies;
}

fn solveBandPathFrequencies(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    dfpt_cfg: DfptConfig,
    model: *const model_mod.Model,
    grid: Grid,
    band_data: *const BandGroundStateData,
    sym_data: *const BandSymmetryData,
    kgs_data: []const kpt_gs.KPointGsData,
    q_points_cart: []const math.Vec3,
    q_points_frac: []const math.Vec3,
) ![][]f64 {
    return solveBandQPoints(
        alloc,
        io,
        cfg,
        dfpt_cfg,
        band_data.prepared.gs,
        band_data.prepared.local_r,
        &band_data.ionic,
        band_data.rho0_g,
        band_data.vxc_g,
        model.species,
        model.atoms,
        model.cell_bohr,
        model.recip,
        model.volume_bohr,
        grid,
        sym_data,
        kgs_data,
        q_points_cart,
        q_points_frac,
    );
}

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

    var band_data = try initBandGroundStateData(
        alloc,
        io,
        cfg,
        scf_result,
        species,
        atoms,
        volume,
        recip,
        grid,
    );
    defer band_data.deinit(alloc);

    const gs = band_data.prepared.gs;

    // Generate q-path
    const q_path_data = try generateFccQPath(alloc, recip, npoints_per_seg);
    errdefer {
        alloc.free(q_path_data.label_positions);
        alloc.free(q_path_data.labels);
        alloc.free(q_path_data.distances);
    }
    defer alloc.free(q_path_data.q_points_frac);
    defer alloc.free(q_path_data.q_points_cart);

    const n_q = q_path_data.q_points_cart.len;
    logDfptInfo("dfpt_band: {d} q-points along path\n", .{n_q});

    var sym_data = try initBandSymmetryData(alloc, cell_bohr, atoms, recip);
    defer sym_data.deinit(alloc);

    const dfpt_cfg = DfptConfig.fromConfig(cfg);

    // ---------------------------------------------------------------
    // Prepare full-BZ k-point ground-state data (q-independent).
    // DFPT requires the full BZ sum over k-points. Even if SCF used
    // IBZ k-points, DFPT needs all k-points in the full BZ because
    // for q≠0, the function f(k, k+q) is not invariant under the
    // symmetry operations that map k to its star.
    // ---------------------------------------------------------------
    const kgs_data = try prepareBandKgsData(
        alloc,
        io,
        cfg,
        &gs,
        band_data.prepared.local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
    );
    defer deinitKPointGsData(alloc, kgs_data);

    const frequencies = try solveBandPathFrequencies(
        alloc,
        io,
        cfg,
        dfpt_cfg,
        model,
        grid,
        &band_data,
        &sym_data,
        kgs_data,
        q_path_data.q_points_cart,
        q_path_data.q_points_frac,
    );

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
