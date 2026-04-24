//! Direct q-path phonon band structure (no IFC interpolation).
//!
//! Solves DFPT from scratch at every q-point along a high-symmetry path,
//! builds the complex dynamical matrix, applies symmetry reconstruction
//! and the acoustic sum rule at q=0, mass-weights, and diagonalizes
//! the Hermitian mass-weighted D(q) to produce the phonon frequencies.
//!
//! q-point-level perturbation parallelism lives in this file because it
//! is only used by run_phonon_band — a per-q ThreadPool-like driver that
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
const log_dfpt = dfpt.log_dfpt;
const log_dfpt_info = dfpt.log_dfpt_info;

const qpath = @import("qpath.zig");
const generate_fcc_q_path = qpath.generate_fcc_q_path;

const kpt_gs = @import("kpt_gs.zig");
const prepare_full_bz_kpoints = kpt_gs.prepare_full_bz_kpoints;

const kpt_dfpt = @import("kpt_dfpt.zig");
const KPointDfptData = kpt_dfpt.KPointDfptData;
const MultiKPertResult = kpt_dfpt.MultiKPertResult;
const build_k_point_dfpt_data_from_gs = kpt_dfpt.build_k_point_dfpt_data_from_gs;

const solver_multik = @import("solver_multik.zig");
const solve_perturbation_q_multi_k = solver_multik.solve_perturbation_q_multi_k;

const dynmat_build = @import("dynmat_build.zig");
const build_q_dynmat_multi_k = dynmat_build.build_q_dynmat_multi_k;

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

    pub fn deinit(self: *BandGroundStateData, alloc: std.mem.Allocator) void {
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

    pub fn deinit(self: *BandSymmetryData, alloc: std.mem.Allocator) void {
        deinit_sym_data(alloc, self.*);
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

    pub fn deinit(self: *QPointPertBuffers, alloc: std.mem.Allocator) void {
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

pub fn deinit_k_point_dfpt_data(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
) void {
    for (kpts) |*kd| kd.deinit_q_only(alloc);
    alloc.free(kpts);
}

pub fn deinit_k_point_gs_data(
    alloc: std.mem.Allocator,
    kgs_data: []kpt_gs.KPointGsData,
) void {
    for (kgs_data) |*kg| kg.deinit(alloc);
    alloc.free(kgs_data);
}

pub fn deinit_sym_data(
    alloc: std.mem.Allocator,
    sym_data: anytype,
) void {
    for (sym_data.indsym) |row| alloc.free(row);
    alloc.free(sym_data.indsym);
    for (sym_data.tnons_shift) |row| alloc.free(row);
    alloc.free(sym_data.tnons_shift);
}

pub fn init_band_ground_state_data(
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
    var prepared = try dfpt.prepare_ground_state(
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

    const rho0_g = try scf_mod.real_to_reciprocal(alloc, grid, scf_result.density, false);
    errdefer alloc.free(rho0_g);

    var vxc_g: ?[]math.Complex = null;
    if (prepared.vxc_r) |values| {
        vxc_g = try scf_mod.real_to_reciprocal(alloc, grid, values, false);
    }
    errdefer if (vxc_g) |values| alloc.free(values);

    return .{
        .prepared = prepared,
        .ionic = ionic,
        .rho0_g = rho0_g,
        .vxc_g = vxc_g,
    };
}

pub fn init_band_symmetry_data(
    alloc: std.mem.Allocator,
    cell_bohr: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
) !BandSymmetryData {
    const symops = try symmetry_mod.get_symmetry_ops(alloc, cell_bohr, atoms, 1e-5);
    errdefer alloc.free(symops);
    log_dfpt_info("dfpt_band: {d} symmetry operations found\n", .{symops.len});

    const sym_data = try dynmat_mod.build_indsym(alloc, symops, atoms, recip, 1e-5);
    errdefer deinit_sym_data(alloc, sym_data);

    return .{
        .symops = symops,
        .indsym = sym_data.indsym,
        .tnons_shift = sym_data.tnons_shift,
    };
}

pub fn prepare_band_kgs_data(
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
    return prepare_full_bz_kpoints(
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

fn build_analytic_perturbations(
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
            buffers.vloc1_gs[pidx] = try perturbation.build_local_perturbation_q(
                alloc,
                grid,
                atoms[ia],
                species,
                dir,
                q_cart,
                gs.local_cfg,
                gs.ff_tables,
            );
            buffers.rho1_core_gs[pidx] = try perturbation.build_core_perturbation_q(
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

fn log_perturbation_rho1_norm(
    ia: usize,
    dir: usize,
    result: *const MultiKPertResult,
) void {
    var rho1_norm: f64 = 0.0;
    for (result.rho1_g) |c| {
        rho1_norm += c.r * c.r + c.i * c.i;
    }
    log_dfpt(
        "dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n",
        .{ ia, dir, @sqrt(rho1_norm) },
    );
}

fn solve_sequential_q_point_perturbations(
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
    try build_analytic_perturbations(alloc, grid, atoms, species, q_cart, gs, buffers);
    for (irr_info.irr_atom_indices) |ia| {
        for (0..3) |dir| {
            const pidx = 3 * ia + dir;
            buffers.pert_results_mk[pidx] = try solve_perturbation_q_multi_k(
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
            log_perturbation_rho1_norm(ia, dir, &buffers.pert_results_mk[pidx]);
        }
    }
}

fn log_parallel_perturbation_plan(
    pert_thread_count: usize,
    kpoint_threads: usize,
    dim: usize,
    n_irr_perts: usize,
) void {
    log_dfpt_info(
        "dfpt_band: using {d} pert threads × {d} kpt threads" ++
            " for {d} perturbations ({d} irreducible)\n",
        .{ pert_thread_count, kpoint_threads, dim, n_irr_perts },
    );
}

fn build_irreducible_perturbation_indices(
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

fn init_q_point_shared(
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

fn run_parallel_perturbation_workers(
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
        threads_arr[ti] = try std.Thread.spawn(.{}, qpoint_pert_worker_fn, .{&workers[ti + 1]});
    }
    qpoint_pert_worker_fn(&workers[0]);
    for (threads_arr) |t| t.join();
}

fn solve_parallel_q_point_perturbations(
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
    pert_dfpt_cfg.kpoint_threads = dfpt.kpoint_threads_for_pert_parallel(
        pert_thread_count,
        dfpt_cfg.kpoint_threads,
    );
    log_parallel_perturbation_plan(
        pert_thread_count,
        pert_dfpt_cfg.kpoint_threads,
        dim,
        n_irr_perts,
    );

    try build_analytic_perturbations(alloc, grid, atoms, species, q_cart, gs, buffers);

    var next_index = std.atomic.Value(usize).init(0);
    var stop_flag = std.atomic.Value(u8).init(0);
    var worker_err: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;
    var qshared = init_q_point_shared(
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

    const irr_pert_indices = try build_irreducible_perturbation_indices(alloc, irr_info);
    defer alloc.free(irr_pert_indices);

    qshared.irr_pert_indices = irr_pert_indices;
    try run_parallel_perturbation_workers(alloc, &qshared, pert_thread_count);
    if (worker_err) |e| return e;
}

pub fn solve_q_point_perturbations(
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
        try solve_sequential_q_point_perturbations(
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

    try solve_parallel_q_point_perturbations(
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

fn finalize_q_point_frequencies(
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
    const dyn_q = try build_q_dynmat_multi_k(
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

    finalize_q_point_dynmat(
        dyn_q,
        n_atoms,
        irr_info,
        symops,
        indsym,
        tnons_shift,
        cell_bohr,
        qf,
        q_norm,
        ionic.masses,
    );
    var result_q = try dynmat_mod.diagonalize_complex(alloc, dyn_q, dim);
    defer result_q.deinit(alloc);

    const frequencies = try alloc.alloc(f64, dim);
    @memcpy(frequencies, result_q.frequencies_cm1);
    return frequencies;
}

fn finalize_q_point_dynmat(
    dyn_q: []math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    symops: []const symmetry_mod.SymOp,
    indsym: []const []const usize,
    tnons_shift: []const []const math.Vec3,
    cell_bohr: math.Mat3,
    qf: math.Vec3,
    q_norm: f64,
    masses: []const f64,
) void {
    if (irr_info.n_irr_atoms < n_atoms) {
        dynmat_mod.reconstruct_dynmat_columns_complex(
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
    if (q_norm < 1e-10) dynmat_mod.apply_asr_complex(dyn_q, n_atoms);
    dynmat_mod.mass_weight_complex(dyn_q, n_atoms, masses);
}

fn log_q_point_summary(
    iq: usize,
    qf: math.Vec3,
    q_norm: f64,
    n_kpts: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    n_atoms: usize,
    frequencies: []const f64,
) void {
    log_dfpt(
        "dfpt_band: q[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n",
        .{ iq, qf.x, qf.y, qf.z, q_norm },
    );
    log_dfpt("dfpt_band: q[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });
    log_dfpt(
        "dfpt_band: q[{d}] {d}/{d} irreducible atoms\n",
        .{ iq, irr_info.n_irr_atoms, n_atoms },
    );
    log_dfpt_info("dfpt_band: q[{d}] freqs:", .{iq});
    for (frequencies) |f| {
        log_dfpt_info(" {d:.1}", .{f});
    }
    log_dfpt_info("\n", .{});
}

const BandQPointInputs = struct {
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
};

fn solve_band_q_point(
    alloc: std.mem.Allocator,
    io: std.Io,
    inputs: BandQPointInputs,
    iq: usize,
    q_cart: math.Vec3,
    qf: math.Vec3,
) ![]f64 {
    const n_atoms = inputs.atoms.len;
    var context = try QPointSolveContext.init(
        alloc,
        io,
        inputs.cfg,
        inputs.dfpt_cfg,
        inputs.symops,
        inputs.indsym,
        n_atoms,
        qf,
        inputs.kgs_data,
        q_cart,
        inputs.local_r,
        inputs.species,
        inputs.atoms,
        inputs.recip,
        inputs.volume,
        inputs.grid,
    );
    var buffers = try solve_q_point_perturbations(
        alloc,
        io,
        inputs.grid,
        inputs.gs,
        inputs.species,
        inputs.atoms,
        q_cart,
        inputs.dfpt_cfg,
        context.pert_thread_count,
        context.kpts,
        context.irr_info,
    );
    defer buffers.deinit(alloc);
    defer context.deinit(alloc);

    return try finish_solved_q_point(
        alloc,
        inputs.cfg,
        inputs.gs,
        inputs.ionic,
        inputs.rho0_g,
        inputs.vxc_g,
        inputs.cell_bohr,
        inputs.recip,
        inputs.volume,
        q_cart,
        qf,
        math.Vec3.norm(q_cart),
        inputs.grid,
        inputs.species,
        inputs.atoms,
        inputs.symops,
        inputs.indsym,
        inputs.tnons_shift,
        iq,
        inputs.kgs_data.len,
        n_atoms,
        context.irr_info,
        &buffers,
        context.kpts,
    );
}

fn finish_solved_q_point(
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
    iq: usize,
    n_kpts: usize,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    buffers: *const QPointPertBuffers,
    kpts: []KPointDfptData,
) ![]f64 {
    const frequencies = try finalize_q_point_frequencies(
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
        buffers,
    );
    log_q_point_summary(iq, qf, q_norm, n_kpts, irr_info, n_atoms, frequencies);
    return frequencies;
}

const QPointSolveContext = struct {
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    kpts: []KPointDfptData,
    pert_thread_count: usize,

    fn init(
        alloc: std.mem.Allocator,
        io: std.Io,
        cfg: config_mod.Config,
        dfpt_cfg: DfptConfig,
        symops: []const symmetry_mod.SymOp,
        indsym: []const []const usize,
        n_atoms: usize,
        qf: math.Vec3,
        kgs_data: []const kpt_gs.KPointGsData,
        q_cart: math.Vec3,
        local_r: []const f64,
        species: []const hamiltonian.SpeciesEntry,
        atoms: []const hamiltonian.AtomData,
        recip: math.Mat3,
        volume: f64,
        grid: Grid,
    ) !QPointSolveContext {
        const irr_info = try dynmat_mod.find_irreducible_atoms(alloc, symops, indsym, n_atoms, qf);
        errdefer @constCast(&irr_info).deinit(alloc);

        const pert_thread_count = dfpt.perturbation_thread_count(
            3 * n_atoms,
            dfpt_cfg.perturbation_threads,
        );
        const q_norm = math.Vec3.norm(q_cart);
        const kpts = try build_k_point_dfpt_data_from_gs(
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
        return .{
            .irr_info = irr_info,
            .kpts = kpts,
            .pert_thread_count = pert_thread_count,
        };
    }

    fn deinit(self: *QPointSolveContext, alloc: std.mem.Allocator) void {
        deinit_k_point_dfpt_data(alloc, self.kpts);
        self.irr_info.deinit(alloc);
    }
};

fn solve_band_q_points(
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

    const q_point_inputs: BandQPointInputs = .{
        .cfg = cfg,
        .dfpt_cfg = dfpt_cfg,
        .gs = gs,
        .local_r = local_r,
        .ionic = ionic,
        .rho0_g = rho0_g,
        .vxc_g = vxc_g,
        .species = species,
        .atoms = atoms,
        .cell_bohr = cell_bohr,
        .recip = recip,
        .volume = volume,
        .grid = grid,
        .symops = sym_data.symops,
        .indsym = sym_data.indsym,
        .tnons_shift = sym_data.tnons_shift,
        .kgs_data = kgs_data,
    };
    for (0..n_q) |iq| {
        frequencies[iq] = try solve_band_q_point(
            alloc,
            io,
            q_point_inputs,
            iq,
            q_points_cart[iq],
            q_points_frac[iq],
        );
        freq_count = iq + 1;
    }

    return frequencies;
}

fn solve_band_path_frequencies(
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
    return solve_band_q_points(
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
pub fn run_phonon_band(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    model: *const model_mod.Model,
    npoints_per_seg: usize,
) !PhononBandResult {
    const dim = 3 * model.atoms.len;
    log_dfpt_info("dfpt_band: starting phonon band calculation ({d} atoms)\n", .{model.atoms.len});

    var context = try BandPhononContext.init(alloc, io, cfg, scf_result, model, npoints_per_seg);
    defer context.deinit(alloc);

    log_dfpt_info("dfpt_band: {d} q-points along path\n", .{context.q_path_data.q_points_cart.len});
    const frequencies = try solve_band_path_frequencies(
        alloc,
        io,
        cfg,
        context.dfpt_cfg,
        model,
        scf_result.grid,
        &context.band_data,
        &context.sym_data,
        context.kgs_data,
        context.q_path_data.q_points_cart,
        context.q_path_data.q_points_frac,
    );
    context.release_result_path_data();

    return PhononBandResult{
        .distances = context.q_path_data.distances,
        .frequencies = frequencies,
        .n_modes = dim,
        .n_q = context.q_path_data.q_points_cart.len,
        .labels = context.q_path_data.labels,
        .label_positions = context.q_path_data.label_positions,
    };
}

const BandPhononContext = struct {
    band_data: BandGroundStateData,
    q_path_data: qpath.GeneratedQPath,
    sym_data: BandSymmetryData,
    dfpt_cfg: DfptConfig,
    kgs_data: []kpt_gs.KPointGsData,
    owns_result_path_data: bool = true,

    fn init(
        alloc: std.mem.Allocator,
        io: std.Io,
        cfg: config_mod.Config,
        scf_result: *scf_mod.ScfResult,
        model: *const model_mod.Model,
        npoints_per_seg: usize,
    ) !BandPhononContext {
        var band_data = try init_band_ground_state_data(
            alloc,
            io,
            cfg,
            scf_result,
            model.species,
            model.atoms,
            model.volume_bohr,
            model.recip,
            scf_result.grid,
        );
        errdefer band_data.deinit(alloc);

        const q_path_data = try generate_fcc_q_path(alloc, model.recip, npoints_per_seg);
        errdefer {
            alloc.free(q_path_data.q_points_frac);
            alloc.free(q_path_data.q_points_cart);
            alloc.free(q_path_data.distances);
            alloc.free(q_path_data.labels);
            alloc.free(q_path_data.label_positions);
        }

        var sym_data = try init_band_symmetry_data(
            alloc,
            model.cell_bohr,
            model.atoms,
            model.recip,
        );
        errdefer sym_data.deinit(alloc);

        const kgs_data = try prepare_band_kgs_data(
            alloc,
            io,
            cfg,
            &band_data.prepared.gs,
            band_data.prepared.local_r,
            model.species,
            model.atoms,
            model.recip,
            model.volume_bohr,
            scf_result.grid,
        );
        return .{
            .band_data = band_data,
            .q_path_data = q_path_data,
            .sym_data = sym_data,
            .dfpt_cfg = DfptConfig.from_config(cfg),
            .kgs_data = kgs_data,
        };
    }

    fn release_result_path_data(self: *BandPhononContext) void {
        self.owns_result_path_data = false;
    }

    fn deinit(self: *BandPhononContext, alloc: std.mem.Allocator) void {
        deinit_k_point_gs_data(alloc, self.kgs_data);
        self.sym_data.deinit(alloc);
        alloc.free(self.q_path_data.q_points_frac);
        alloc.free(self.q_path_data.q_points_cart);
        if (self.owns_result_path_data) {
            alloc.free(self.q_path_data.distances);
            alloc.free(self.q_path_data.labels);
            alloc.free(self.q_path_data.label_positions);
        }
        self.band_data.deinit(alloc);
    }
};

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

pub fn set_q_point_pert_error(shared: *QPointPertShared, e: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);

    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

pub fn qpoint_pert_worker_fn(worker: *QPointPertWorker) void {
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
            log_dfpt(
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
        shared.pert_results_mk[idx] = solve_perturbation_q_multi_k(
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
            set_q_point_pert_error(shared, e);
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

            log_dfpt("dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n", .{ ia, dir, @sqrt(rho1_norm) });
        }
    }
}
