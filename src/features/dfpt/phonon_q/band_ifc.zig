//! IFC-interpolated phonon band structure.
//!
//! Three phases:
//!   1. Compute D(q) on a coarse q-grid via DFPT (Monkhorst-Pack).
//!   2. Fourier-transform D(q) → IFC C(R), apply ASR in real space.
//!   3. Interpolate D(q') along the band q-path, mass-weight, and
//!      diagonalize to obtain phonon frequencies.
//!
//! Optionally computes a phonon DOS on a dense mesh in phase 4.

const std = @import("std");
const math = @import("../../math/math.zig");
const scf_mod = @import("../../scf/scf.zig");
const config_mod = @import("../../config/config.zig");
const model_mod = @import("../../dft/model.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const mesh_mod = @import("../../kpoints/mesh.zig");

const dfpt = @import("../dfpt.zig");
const dynmat_mod = dfpt.dynmat;
const ifc_mod = @import("../ifc.zig");
const phonon_dos_mod = @import("../phonon_dos.zig");
const DfptConfig = dfpt.DfptConfig;
const IonicData = dfpt.IonicData;
const log_dfpt = dfpt.log_dfpt;
const log_dfpt_info = dfpt.log_dfpt_info;

const qpath_mod = @import("qpath.zig");
const generate_fcc_q_path = qpath_mod.generate_fcc_q_path;
const generate_q_path_from_config = qpath_mod.generate_q_path_from_config;
const GeneratedQPath = qpath_mod.GeneratedQPath;

const kpt_gs = @import("kpt_gs.zig");

const kpt_dfpt = @import("kpt_dfpt.zig");
const KPointDfptData = kpt_dfpt.KPointDfptData;
const build_k_point_dfpt_data_from_gs = kpt_dfpt.build_k_point_dfpt_data_from_gs;

const dynmat_build = @import("dynmat_build.zig");
const build_q_dynmat_multi_k = dynmat_build.build_q_dynmat_multi_k;

const band_direct = @import("band_direct.zig");
const BandGroundStateData = band_direct.BandGroundStateData;
const BandSymmetryData = band_direct.BandSymmetryData;
const PhononBandResult = band_direct.PhononBandResult;
const deinit_k_point_dfpt_data = band_direct.deinit_k_point_dfpt_data;
const deinit_k_point_gs_data = band_direct.deinit_k_point_gs_data;
const init_band_ground_state_data = band_direct.init_band_ground_state_data;
const init_band_symmetry_data = band_direct.init_band_symmetry_data;
const prepare_band_kgs_data = band_direct.prepare_band_kgs_data;
const solve_q_point_perturbations = band_direct.solve_q_point_perturbations;

const Grid = scf_mod.Grid;
const IFCData = ifc_mod.IFC;

const QGridDynmatData = struct {
    q_frac_grid: []math.Vec3,
    dynmat_grid: [][]math.Complex,

    fn deinit(self: *QGridDynmatData, alloc: std.mem.Allocator) void {
        for (self.dynmat_grid) |dyn_q| alloc.free(dyn_q);
        alloc.free(self.dynmat_grid);
        alloc.free(self.q_frac_grid);
    }
};

fn log_ifc_q_grid_point(
    iq: usize,
    qf: math.Vec3,
    q_norm: f64,
    n_kpts: usize,
    irr_atoms: usize,
    n_atoms: usize,
) void {
    log_dfpt(
        "dfpt_ifc: q_grid[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n",
        .{ iq, qf.x, qf.y, qf.z, q_norm },
    );
    log_dfpt("dfpt_ifc: q_grid[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });
    log_dfpt(
        "dfpt_ifc: q_grid[{d}] {d}/{d} irreducible atoms\n",
        .{ iq, irr_atoms, n_atoms },
    );
}

fn init_ifc_irreducible_atoms(
    alloc: std.mem.Allocator,
    sym_data: *const BandSymmetryData,
    n_atoms: usize,
    iq: usize,
    q_cart: math.Vec3,
    qf: math.Vec3,
    n_kpts: usize,
) !dynmat_mod.IrreducibleAtomInfo {
    const q_norm = math.Vec3.norm(q_cart);
    const irr_info = try dynmat_mod.find_irreducible_atoms(
        alloc,
        sym_data.symops,
        sym_data.indsym,
        n_atoms,
        qf,
    );
    log_ifc_q_grid_point(iq, qf, q_norm, n_kpts, irr_info.n_irr_atoms, n_atoms);
    return irr_info;
}

fn build_ifc_q_grid_k_points(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    band_data: *const BandGroundStateData,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    kgs_data: []const kpt_gs.KPointGsData,
    q_cart: math.Vec3,
    q_norm: f64,
    pert_thread_count: usize,
) ![]KPointDfptData {
    return build_k_point_dfpt_data_from_gs(
        alloc,
        io,
        kgs_data,
        q_cart,
        q_norm,
        cfg,
        band_data.prepared.local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
        pert_thread_count,
    );
}

fn compute_ifc_q_grid_point_response(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    dfpt_cfg: DfptConfig,
    band_data: *const BandGroundStateData,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    sym_data: *const BandSymmetryData,
    kpts: []KPointDfptData,
    q_cart: math.Vec3,
    qf: math.Vec3,
    iq: usize,
    pert_thread_count: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    var buffers = try solve_q_point_perturbations(
        alloc,
        io,
        grid,
        band_data.prepared.gs,
        species,
        atoms,
        q_cart,
        dfpt_cfg,
        pert_thread_count,
        kpts,
        irr_info,
    );
    defer buffers.deinit(alloc);

    const dyn_q = try build_q_dynmat_multi_k(
        alloc,
        kpts,
        buffers.pert_results_mk,
        buffers.vloc1_gs,
        buffers.rho1_core_gs,
        band_data.rho0_g,
        band_data.prepared.gs,
        band_data.ionic.charges,
        band_data.ionic.positions,
        cell_bohr,
        recip,
        volume,
        q_cart,
        grid,
        species,
        atoms,
        band_data.prepared.gs.ff_tables,
        band_data.prepared.gs.rho_core_tables,
        band_data.prepared.gs.rho_core,
        band_data.vxc_g,
        cfg.vdw,
        irr_info,
    );
    if (irr_info.n_irr_atoms < atoms.len) {
        dynmat_mod.reconstruct_dynmat_columns_complex(
            dyn_q,
            atoms.len,
            irr_info,
            sym_data.symops,
            sym_data.indsym,
            sym_data.tnons_shift,
            cell_bohr,
            qf,
        );
    }
    log_dfpt("dfpt_ifc: q_grid[{d}] D(q) computed\n", .{iq});
    return dyn_q;
}

fn solve_ifc_q_grid_point_dynmat(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    dfpt_cfg: DfptConfig,
    band_data: *const BandGroundStateData,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    sym_data: *const BandSymmetryData,
    kgs_data: []const kpt_gs.KPointGsData,
    iq: usize,
    q_cart: math.Vec3,
    qf: math.Vec3,
) ![]math.Complex {
    const n_atoms = atoms.len;
    const q_norm = math.Vec3.norm(q_cart);
    var irr_info = try init_ifc_irreducible_atoms(
        alloc,
        sym_data,
        n_atoms,
        iq,
        q_cart,
        qf,
        kgs_data.len,
    );
    defer irr_info.deinit(alloc);

    const pert_thread_count = dfpt.perturbation_thread_count(
        3 * n_atoms,
        dfpt_cfg.perturbation_threads,
    );
    const kpts = try build_ifc_q_grid_k_points(
        alloc,
        io,
        cfg,
        band_data,
        species,
        atoms,
        recip,
        volume,
        grid,
        kgs_data,
        q_cart,
        q_norm,
        pert_thread_count,
    );
    defer deinit_k_point_dfpt_data(alloc, kpts);

    return compute_ifc_q_grid_point_response(
        alloc,
        io,
        cfg,
        dfpt_cfg,
        band_data,
        species,
        atoms,
        cell_bohr,
        recip,
        volume,
        grid,
        sym_data,
        kpts,
        q_cart,
        qf,
        iq,
        pert_thread_count,
        irr_info,
    );
}

fn compute_ifc_q_grid_dynmat(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    dfpt_cfg: DfptConfig,
    band_data: *const BandGroundStateData,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    sym_data: *const BandSymmetryData,
    kgs_data: []const kpt_gs.KPointGsData,
    qgrid: [3]usize,
) !QGridDynmatData {
    const qgrid_points = try mesh_mod.generate_kmesh(
        alloc,
        qgrid,
        recip,
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
    );
    defer alloc.free(qgrid_points);

    log_dfpt_info("dfpt_ifc: {d} q-grid points for DFPT\n", .{qgrid_points.len});

    const q_frac_grid = try alloc.alloc(math.Vec3, qgrid_points.len);
    errdefer alloc.free(q_frac_grid);
    const dynmat_grid = try alloc.alloc([]math.Complex, qgrid_points.len);
    var dynmat_count: usize = 0;
    errdefer {
        for (0..dynmat_count) |i| alloc.free(dynmat_grid[i]);
        alloc.free(dynmat_grid);
    }

    for (qgrid_points, 0..) |qpoint, iq| {
        q_frac_grid[iq] = qpoint.k_frac;
        dynmat_grid[iq] = try solve_ifc_q_grid_point_dynmat(
            alloc,
            io,
            cfg,
            dfpt_cfg,
            band_data,
            species,
            atoms,
            cell_bohr,
            recip,
            volume,
            grid,
            sym_data,
            kgs_data,
            iq,
            qpoint.k_cart,
            qpoint.k_frac,
        );
        dynmat_count = iq + 1;
    }

    return .{
        .q_frac_grid = q_frac_grid,
        .dynmat_grid = dynmat_grid,
    };
}

fn init_ifc_q_path(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    recip: math.Mat3,
) !GeneratedQPath {
    const npoints_per_seg = cfg.dfpt.qpath_npoints;
    if (cfg.dfpt.qpath.len >= 2) {
        return generate_q_path_from_config(alloc, cfg.dfpt.qpath, npoints_per_seg, recip);
    }
    return generate_fcc_q_path(alloc, recip, npoints_per_seg);
}

fn interpolate_ifc_band_path(
    alloc: std.mem.Allocator,
    ifc_data: *const IFCData,
    ionic: *const IonicData,
    n_atoms: usize,
    q_points_frac: []const math.Vec3,
    q_points_cart: []const math.Vec3,
) ![][]f64 {
    const dim = 3 * n_atoms;
    var frequencies = try alloc.alloc([]f64, q_points_cart.len);
    var freq_count: usize = 0;
    errdefer {
        for (0..freq_count) |i| alloc.free(frequencies[i]);
        alloc.free(frequencies);
    }

    for (0..q_points_cart.len) |iq| {
        const dyn_interp = try ifc_mod.interpolate(alloc, ifc_data, q_points_frac[iq]);
        defer alloc.free(dyn_interp);

        if (math.Vec3.norm(q_points_cart[iq]) < 1e-10) {
            dynmat_mod.apply_asr_complex(dyn_interp, n_atoms);
        }
        dynmat_mod.mass_weight_complex(dyn_interp, n_atoms, ionic.masses);

        var result_q = try dynmat_mod.diagonalize_complex(alloc, dyn_interp, dim);
        defer result_q.deinit(alloc);

        frequencies[iq] = try alloc.alloc(f64, dim);
        @memcpy(frequencies[iq], result_q.frequencies_cm1);
        freq_count = iq + 1;

        if (iq % 10 == 0 or iq + 1 == q_points_cart.len) {
            log_dfpt_info("dfpt_ifc: q[{d}] freqs:", .{iq});
            for (result_q.frequencies_cm1) |f| log_dfpt_info(" {d:.1}", .{f});
            log_dfpt_info("\n", .{});
        }
    }
    return frequencies;
}

fn maybe_write_ifc_phonon_dos(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    ifc_data: *const IFCData,
    ionic: *const IonicData,
    n_atoms: usize,
) !void {
    const dos_qmesh = cfg.dfpt.dos_qmesh orelse return;
    log_dfpt_info(
        "dfpt_ifc: computing phonon DOS on {d}x{d}x{d} mesh\n",
        .{ dos_qmesh[0], dos_qmesh[1], dos_qmesh[2] },
    );

    var pdos = try phonon_dos_mod.compute_phonon_dos(
        alloc,
        ifc_data,
        ionic.masses,
        n_atoms,
        dos_qmesh,
        cfg.dfpt.dos_sigma,
        cfg.dfpt.dos_nbin,
    );
    defer pdos.deinit(alloc);

    var out_dir = try std.Io.Dir.cwd().openDir(io, cfg.out_dir, .{});
    defer out_dir.close(io);

    try phonon_dos_mod.write_phonon_dos_csv(io, out_dir, pdos);
    log_dfpt_info("dfpt_ifc: phonon DOS written to phonon_dos.csv\n", .{});
}

fn compute_ifc_from_q_grid(
    alloc: std.mem.Allocator,
    qgrid_data: *const QGridDynmatData,
    qgrid: [3]usize,
    n_atoms: usize,
) !IFCData {
    log_dfpt_info(
        "dfpt_ifc: computing IFC from {d} q-grid points\n",
        .{qgrid_data.dynmat_grid.len},
    );

    var dynmat_const = try alloc.alloc([]const math.Complex, qgrid_data.dynmat_grid.len);
    defer alloc.free(dynmat_const);

    for (0..qgrid_data.dynmat_grid.len) |i| {
        dynmat_const[i] = qgrid_data.dynmat_grid[i];
    }

    var ifc_data = try ifc_mod.compute_ifc(
        alloc,
        dynmat_const,
        qgrid_data.q_frac_grid,
        qgrid,
        n_atoms,
    );
    ifc_mod.apply_asr(&ifc_data);
    log_dfpt_info("dfpt_ifc: IFC ASR applied\n", .{});
    return ifc_data;
}

fn build_ifc_band_result(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    recip: math.Mat3,
    ifc_data: *const IFCData,
    ionic: *const IonicData,
    n_atoms: usize,
) !PhononBandResult {
    const qpath_data = try init_ifc_q_path(alloc, cfg, recip);
    errdefer {
        alloc.free(qpath_data.q_points_frac);
        alloc.free(qpath_data.q_points_cart);
        alloc.free(qpath_data.distances);
        alloc.free(qpath_data.labels);
        alloc.free(qpath_data.label_positions);
    }

    const n_q = qpath_data.q_points_cart.len;
    log_dfpt_info("dfpt_ifc: interpolating {d} q-path points\n", .{n_q});
    const frequencies = try interpolate_ifc_band_path(
        alloc,
        ifc_data,
        ionic,
        n_atoms,
        qpath_data.q_points_frac,
        qpath_data.q_points_cart,
    );
    errdefer {
        for (frequencies) |freq| alloc.free(freq);
        alloc.free(frequencies);
    }

    try maybe_write_ifc_phonon_dos(alloc, io, cfg, ifc_data, ionic, n_atoms);
    alloc.free(qpath_data.q_points_frac);
    alloc.free(qpath_data.q_points_cart);

    return .{
        .distances = qpath_data.distances,
        .frequencies = frequencies,
        .n_modes = 3 * n_atoms,
        .n_q = n_q,
        .labels = qpath_data.labels,
        .label_positions = qpath_data.label_positions,
    };
}

/// Run DFPT phonon band structure using IFC interpolation.
/// 1. Compute D(q) on a coarse q-grid via DFPT
/// 2. Fourier transform to IFC: C(R)
/// 3. Interpolate D(q') at arbitrary q-path points
pub fn run_phonon_band_ifc(
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
    const grid = scf_result.grid;
    const qgrid = cfg.dfpt.qgrid orelse return error.MissingQgrid;

    log_dfpt_info(
        "dfpt_ifc: starting IFC phonon band ({d} atoms, qgrid={d}x{d}x{d})\n",
        .{ n_atoms, qgrid[0], qgrid[1], qgrid[2] },
    );

    var band_data = try init_band_ground_state_data(
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

    const dfpt_cfg = DfptConfig.from_config(cfg);
    var sym_data = try init_band_symmetry_data(alloc, cell_bohr, atoms, recip);
    defer sym_data.deinit(alloc);

    const kgs_data = try prepare_band_kgs_data(
        alloc,
        io,
        cfg,
        &band_data.prepared.gs,
        band_data.prepared.local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
    );
    defer deinit_k_point_gs_data(alloc, kgs_data);

    var qgrid_data = try compute_ifc_q_grid_dynmat(
        alloc,
        io,
        cfg,
        dfpt_cfg,
        &band_data,
        species,
        atoms,
        cell_bohr,
        recip,
        volume,
        grid,
        &sym_data,
        kgs_data,
        qgrid,
    );
    defer qgrid_data.deinit(alloc);

    var ifc_data = try compute_ifc_from_q_grid(alloc, &qgrid_data, qgrid, n_atoms);
    defer ifc_data.deinit(alloc);

    return build_ifc_band_result(
        alloc,
        io,
        cfg,
        recip,
        &ifc_data,
        &band_data.ionic,
        n_atoms,
    );
}
