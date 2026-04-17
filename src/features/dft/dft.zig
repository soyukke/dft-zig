const std = @import("std");
const band = @import("../band/band.zig");
const config = @import("../config/config.zig");
const cube = @import("cube.zig");
const dos = @import("../dos/dos.zig");
const pdos_mod = @import("../dos/pdos.zig");
const scf = @import("../scf/scf.zig");
const kpath = @import("../kpath/kpath.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const linear_scaling = @import("../linear_scaling/linear_scaling.zig");
const math = @import("../math/math.zig");
const output = @import("output.zig");
const paw_mod = @import("../paw/paw.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const relax = @import("../relax/relax.zig");
const xyz = @import("../structure/xyz.zig");

fn logStep(io: std.Io, msg: []const u8) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    try out.print("{s}\n", .{msg});
    try out.flush();
}

fn nowNs(io: std.Io) u64 {
    const ts = std.Io.Clock.Timestamp.now(io, .awake);
    return @intCast(ts.raw.nanoseconds);
}

fn elapsedNs(io: std.Io, start_ns: u64) u64 {
    const now = nowNs(io);
    if (now <= start_ns) return 0;
    return now - start_ns;
}

const ActiveStructure = struct {
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
};

fn cellVolume(cell_bohr: math.Mat3) f64 {
    return @abs(math.Vec3.dot(cell_bohr.row(0), math.Vec3.cross(cell_bohr.row(1), cell_bohr.row(2))));
}

fn selectActiveStructure(
    initial_atoms: []const hamiltonian.AtomData,
    initial_cell_bohr: math.Mat3,
    initial_recip: math.Mat3,
    initial_volume_bohr: f64,
    relax_result: ?*const relax.RelaxResult,
) ActiveStructure {
    if (relax_result) |result| {
        const active_cell = result.final_cell orelse initial_cell_bohr;
        return .{
            .atoms = result.final_atoms,
            .cell_bohr = active_cell,
            .recip = result.final_recip orelse math.reciprocal(active_cell),
            .volume_bohr = result.final_volume orelse cellVolume(active_cell),
        };
    }
    return .{
        .atoms = initial_atoms,
        .cell_bohr = initial_cell_bohr,
        .recip = initial_recip,
        .volume_bohr = initial_volume_bohr,
    };
}

/// Run the current DFT workflow.
pub fn run(alloc: std.mem.Allocator, io: std.Io, cfg: config.Config, atoms: []xyz.Atom) !void {
    const total_start_ns = nowNs(io);
    var timing = output.Timing{};
    timing.cpu_start_us = output.Timing.getCpuTimeUs();

    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, cfg.out_dir);
    var out_dir = try cwd.openDir(io, cfg.out_dir, .{});
    defer out_dir.close(io);

    const setup_start_ns = nowNs(io);

    const unit_scale_bohr = math.unitsScaleToBohr(cfg.units);
    const cell_bohr = cfg.cell.scale(unit_scale_bohr);
    const recip = math.reciprocal(cell_bohr);
    const volume_bohr = cellVolume(cell_bohr);

    try logStep(io, "step: load pseudopotentials");
    const pseudo_data = try loadPseudopotentials(alloc, io, cfg.pseudopotentials);
    defer deinitPseudopotentials(alloc, pseudo_data);
    try logStep(io, "step: build species/atom data");
    const species = try hamiltonian.buildSpeciesEntries(alloc, pseudo_data);
    defer hamiltonian.deinitSpeciesEntries(alloc, species);
    const atom_data = try hamiltonian.buildAtomData(alloc, atoms, unit_scale_bohr, species);
    defer alloc.free(atom_data);

    timing.setup_ns = elapsedNs(io, setup_start_ns);

    // Structure relaxation
    var relax_result: ?relax.RelaxResult = null;
    defer if (relax_result) |*result| result.deinit(alloc);

    if (cfg.relax.enabled) {
        const relax_start_ns = nowNs(io);
        try logStep(io, "step: structure relaxation start");
        relax_result = try relax.run(alloc, io, cfg, species, atom_data, cell_bohr, recip, volume_bohr);

        // Write relaxation output files
        const bohr_to_ang = 0.529177; // 1 Bohr = 0.529177 Angstrom
        try relax.writeOutput(alloc, io, out_dir, &relax_result.?, species, bohr_to_ang);
        try relax.writeTrajectoryXyz(io, out_dir, &relax_result.?, species, cell_bohr, bohr_to_ang);

        try logStep(io, "step: structure relaxation done");
        timing.relax_ns = elapsedNs(io, relax_start_ns);
    }

    const active = selectActiveStructure(atom_data, cell_bohr, recip, volume_bohr, if (relax_result) |*rr| rr else null);
    const cell_ang = active.cell_bohr.scale(math.unitsScaleToAngstrom(.bohr));

    try logStep(io, "step: generate k-path");
    // Resolve path_string ("auto" or "G-X-W-K-G-L") to BandPathPoint array
    var resolved_band = cfg.band;
    var auto_path_result: ?kpath.auto_kpath.AutoKPathResult = null;
    defer if (auto_path_result) |*r| r.deinit(alloc);
    if (resolved_band.path_string) |ps| {
        auto_path_result = try kpath.auto_kpath.resolvePathString(alloc, ps, active.cell_bohr);
        resolved_band.path = auto_path_result.?.points;
    }
    var kpoints = try kpath.generate(alloc, resolved_band, active.recip);
    defer kpoints.deinit(alloc);

    var scf_result: ?scf.ScfResult = null;
    defer if (scf_result) |*result| result.deinit(alloc);

    // Skip post-relax SCF if we have the converged potential from relax
    const have_relax_potential = if (relax_result) |rr| rr.final_potential != null else false;
    const needs_reference = cfg.scf.reference_json != null or cfg.scf.compare_reference_json != null;

    if (cfg.scf.enabled and !(have_relax_potential and !needs_reference)) {
        const scf_start_ns = nowNs(io);
        try logStep(io, "step: scf start");
        // Use caches from relax if available
        // density and kpoint_cache are borrowed (scf.run copies internally)
        // apply_caches ownership is transferred to scf.run
        const init_density: ?[]const f64 = if (relax_result) |rr| rr.final_density else null;
        const init_kpoint_cache: ?[]scf.KpointCache = if (relax_result) |rr| rr.final_kpoint_cache else null;
        var init_apply_caches: ?[]scf.KpointApplyCache = null;
        if (relax_result) |*rr| {
            init_apply_caches = rr.final_apply_caches;
            rr.final_apply_caches = null; // transfer ownership
        }
        const result = try scf.run(.{
            .alloc = alloc,
            .io = io,
            .cfg = cfg,
            .species = species,
            .atoms = active.atoms,
            .recip = active.recip,
            .volume_bohr = active.volume_bohr,
            .initial_density = init_density,
            .initial_kpoint_cache = init_kpoint_cache,
            .initial_apply_caches = init_apply_caches,
        });
        scf_result = result;
        try logStep(io, "step: scf done");
        timing.scf_ns = elapsedNs(io, scf_start_ns);
    }

    // Stress tensor computation
    if (cfg.scf.compute_stress) {
        if (scf_result) |*result| {
            try logStep(io, "step: stress tensor start");
            const stress = @import("../stress/stress.zig");
            const stress_terms = try stress.computeStressFromScf(alloc, io, result, cfg, species, active.atoms);
            _ = stress_terms;
            try logStep(io, "step: stress tensor done");
        }
    }

    // DOS computation
    if (cfg.dos.enabled) {
        if (scf_result) |*result| {
            if (result.wavefunctions) |wf_data| {
                try logStep(io, "step: dos start");
                var dos_result = try dos.computeDos(
                    alloc,
                    wf_data,
                    cfg.dos.sigma,
                    cfg.dos.npoints,
                    cfg.dos.emin,
                    cfg.dos.emax,
                    cfg.scf.nspin,
                );
                defer dos_result.deinit(alloc);
                const dos_fermi = if (std.math.isNan(result.fermi_level)) wf_data.fermi_level else result.fermi_level;
                if (result.wavefunctions_down) |wf_down| {
                    try dos.writeDosCSVNamed(io, out_dir, dos_result, dos_fermi, "dos_up.csv");
                    var dos_down = try dos.computeDos(alloc, wf_down, cfg.dos.sigma, cfg.dos.npoints, cfg.dos.emin, cfg.dos.emax, 2);
                    defer dos_down.deinit(alloc);
                    try dos.writeDosCSVNamed(io, out_dir, dos_down, dos_fermi, "dos_down.csv");
                } else {
                    try dos.writeDosCSV(io, out_dir, dos_result, dos_fermi);
                }
                try logStep(io, "step: dos done");

                // PDOS computation
                if (cfg.dos.pdos) {
                    try logStep(io, "step: pdos start");
                    var pdos_result = try pdos_mod.computePdos(
                        alloc,
                        wf_data,
                        species,
                        active.atoms,
                        active.recip,
                        active.volume_bohr,
                        cfg.dos.sigma,
                        cfg.dos.npoints,
                        cfg.dos.emin,
                        cfg.dos.emax,
                        cfg.scf.nspin,
                    );
                    defer pdos_result.deinit(alloc);
                    try pdos_mod.writePdosCSV(io, out_dir, pdos_result, dos_fermi);
                    try logStep(io, "step: pdos done");
                }
            }
        }
    }

    // Cube output
    if (cfg.output.cube) {
        if (scf_result) |*result| {
            try logStep(io, "step: cube output start");
            try cube.writeCubeFile(
                io,
                out_dir,
                "density.cube",
                result.density,
                .{ result.grid.nx, result.grid.ny, result.grid.nz },
                active.cell_bohr,
                active.atoms,
                species,
            );
            try logStep(io, "step: cube output done");
        }
    }

    if (cfg.scf.reference_json != null or cfg.scf.compare_reference_json != null or cfg.scf.comparison_json != null or cfg.scf.compare_tolerance_json != null) {
        if (scf_result == null) return error.ScfDisabled;
    }

    if (scf_result) |*result| {
        if (cfg.scf.reference_json) |path| {
            try linear_scaling.writeReferenceFromScfResult(io, out_dir, path, result);
        }
        if (cfg.scf.compare_reference_json) |ref_path| {
            var reference = try linear_scaling.readReferenceJson(io, alloc, std.Io.Dir.cwd(), ref_path);
            defer reference.deinit(alloc);
            const report = try linear_scaling.compareReferenceToScfResult(&reference, result);
            if (cfg.scf.comparison_json) |out_path| {
                try linear_scaling.writeComparisonJson(io, out_dir, out_path, report);
            } else {
                return error.MissingComparisonOutput;
            }
            if (cfg.scf.compare_tolerance_json) |tol_path| {
                const tol_content = try cwd.readFileAlloc(io, tol_path, alloc, .limited(1024 * 1024));
                defer alloc.free(tol_content);
                var parsed_tol = try std.json.parseFromSlice(linear_scaling.ScfTolerance, alloc, tol_content, .{});
                defer parsed_tol.deinit();
                const tol_result = linear_scaling.withinScfTolerance(report, parsed_tol.value);
                if (!tol_result.all) {
                    return error.ComparisonToleranceFailed;
                }
            }
        } else if (cfg.scf.comparison_json != null or cfg.scf.compare_tolerance_json != null) {
            return error.MissingReferenceJson;
        }
    }

    // DFPT phonon calculation
    if (cfg.dfpt.enabled) {
        if (scf_result) |*result| {
            const dfpt_mod = @import("../dfpt/dfpt.zig");
            if (cfg.dfpt.qpath_npoints > 0) {
                try logStep(io, "step: dfpt phonon band start");
                var band_result = if (cfg.dfpt.qgrid != null)
                    try dfpt_mod.runPhononBandIFC(
                        alloc,
                        io,
                        cfg,
                        result,
                        species,
                        active.atoms,
                        active.cell_bohr,
                        active.recip,
                        active.volume_bohr,
                    )
                else
                    try dfpt_mod.runPhononBand(
                        alloc,
                        io,
                        cfg,
                        result,
                        species,
                        active.atoms,
                        active.cell_bohr,
                        active.recip,
                        active.volume_bohr,
                        cfg.dfpt.qpath_npoints,
                    );
                defer band_result.deinit(alloc);

                // Write phonon band CSV
                {
                    const csv_file = try out_dir.createFile(io, "phonon_band.csv", .{});
                    defer csv_file.close(io);
                    var csv_buf: [256]u8 = undefined;
                    var csv_writer = csv_file.writer(io, &csv_buf);
                    const csv = &csv_writer.interface;
                    // Header
                    try csv.print("distance", .{});
                    for (0..band_result.n_modes) |m| {
                        try csv.print(",mode_{d}", .{m});
                    }
                    try csv.print("\n", .{});
                    // Data
                    for (0..band_result.n_q) |iq| {
                        try csv.print("{d:.6}", .{band_result.distances[iq]});
                        for (0..band_result.n_modes) |m| {
                            try csv.print(",{d:.4}", .{band_result.frequencies[iq][m]});
                        }
                        try csv.print("\n", .{});
                    }
                    try csv.flush();
                }
                try logStep(io, "step: dfpt phonon band done");
            } else {
                try logStep(io, "step: dfpt phonon start");
                var phonon = try dfpt_mod.runPhonon(alloc, io, cfg, result, species, active.atoms, active.cell_bohr, active.recip, active.volume_bohr);
                defer phonon.deinit(alloc);
                try logStep(io, "step: dfpt phonon done");
                // Print frequencies
                {
                    var buffer: [256]u8 = undefined;
                    var writer = std.Io.File.stderr().writer(io, &buffer);
                    const out = &writer.interface;
                    try out.print("phonon frequencies (cm⁻¹):\n", .{});
                    for (phonon.frequencies_cm1) |f| {
                        try out.print("  {d:.2}\n", .{f});
                    }
                    try out.flush();
                }
            }
        } else {
            return error.ScfRequired;
        }
    }

    const extra_potential: ?*hamiltonian.PotentialGrid = blk: {
        if (relax_result) |*rr| {
            if (rr.final_potential) |*p| break :blk p;
        }
        if (scf_result) |*result| break :blk &result.potential;
        break :blk null;
    };
    const extra_potential_down: ?*hamiltonian.PotentialGrid = blk: {
        if (scf_result) |*result| {
            if (result.potential_down) |*p| break :blk p;
        }
        break :blk null;
    };

    try logStep(io, "step: write outputs");
    try output.writeRunInfo(io, out_dir, cfg, atoms, cell_ang);
    try output.writeAtomsFromAtomData(io, out_dir, active.atoms, species, math.unitsScaleToAngstrom(.bohr));
    try output.writeKpoints(io, out_dir, kpoints);
    try output.writePseudopotentials(io, out_dir, pseudo_data);

    if (kpoints.points.len > 0) {
        const band_start_ns = nowNs(io);
        try logStep(io, "step: band energies");
        const band_paw_tabs: ?[]const paw_mod.PawTab = if (scf_result) |r| r.paw_tabs else null;
        const band_paw_dij: ?[]const []const f64 = if (scf_result) |r| r.paw_dij else null;
        try band.writeBandEnergies(alloc, io, out_dir, cfg, kpoints, species, active.atoms, active.cell_bohr, active.recip, active.volume_bohr, extra_potential, extra_potential_down, band_paw_tabs, band_paw_dij);
        timing.band_ns = elapsedNs(io, band_start_ns);
    }

    timing.total_ns = elapsedNs(io, total_start_ns);
    timing.cpu_end_us = output.Timing.getCpuTimeUs();

    try output.writeStatus(io, out_dir, cfg, scf_result);
    try output.writeTiming(io, out_dir, timing, kpoints.points.len);
    try logStep(io, "step: done");
}

/// Load pseudopotentials from config list.
fn loadPseudopotentials(alloc: std.mem.Allocator, io: std.Io, specs: []pseudo.Spec) ![]pseudo.Parsed {
    if (specs.len == 0) {
        return &[_]pseudo.Parsed{};
    }
    var list: std.ArrayList(pseudo.Parsed) = .empty;
    errdefer {
        for (list.items) |*item| {
            item.deinit(alloc);
        }
        list.deinit(alloc);
    }
    for (specs) |spec| {
        try list.append(alloc, try pseudo.load(alloc, io, spec));
    }
    return try list.toOwnedSlice(alloc);
}

/// Free parsed pseudopotential list.
fn deinitPseudopotentials(alloc: std.mem.Allocator, items: []pseudo.Parsed) void {
    if (items.len == 0) return;
    for (items) |*item| {
        item.deinit(alloc);
    }
    alloc.free(items);
}

fn testOutputConfig() config.Config {
    return .{
        .title = @constCast("test"),
        .xyz_path = @constCast("test.xyz"),
        .out_dir = @constCast("out"),
        .units = .angstrom,
        .linalg_backend = .openblas,
        .threads = 0,
        .cell = math.Mat3.fromRows(
            .{ .x = 10.0, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = 10.0, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = 10.0 },
        ),
        .boundary = .periodic,
        .scf = .{
            .enabled = true,
            .solver = .iterative,
            .xc = .lda_pz,
            .smearing = .none,
            .smear_ry = 0.0,
            .ecut_ry = 30.0,
            .kmesh = .{ 4, 4, 4 },
            .kmesh_shift = .{ 0.0, 0.0, 0.0 },
            .grid = .{ 0, 0, 0 },
            .grid_scale = 1.0,
            .mixing_beta = 0.3,
            .max_iter = 50,
            .convergence = 1e-6,
            .convergence_metric = .density,
            .profile = false,
            .quiet = false,
            .debug_nonlocal = false,
            .debug_local = false,
            .debug_fermi = false,
            .enable_nonlocal = true,
            .local_potential = .short_range,
            .symmetry = true,
            .time_reversal = true,
            .kpoint_threads = 0,
            .iterative_max_iter = 20,
            .iterative_tol = 1e-4,
            .iterative_max_subspace = 0,
            .iterative_block_size = 0,
            .iterative_init_diagonal = false,
            .iterative_warmup_steps = 2,
            .iterative_warmup_max_iter = 10,
            .iterative_warmup_tol = 1e-3,
            .iterative_reuse_vectors = true,
            .kerker_q0 = 0.0,
            .diemac = 12.0,
            .dielng = 1.0,
            .pulay_history = 8,
            .pulay_start = 4,
            .mixing_mode = .potential,
            .use_rfft = false,
            .fft_backend = .fftw,
            .nspin = 1,
            .spinat = null,
            .compute_stress = false,
            .reference_json = null,
            .compare_reference_json = null,
            .comparison_json = null,
            .compare_tolerance_json = null,
        },
        .ewald = .{ .alpha = 0.0, .rcut = 0.0, .gcut = 0.0, .tol = 1e-8 },
        .vdw = .{},
        .band = .{
            .points_per_segment = 2,
            .nbands = 8,
            .path = &.{},
            .path_string = null,
            .solver = .dense,
            .iterative_max_iter = 40,
            .iterative_tol = 1e-6,
            .iterative_max_subspace = 0,
            .iterative_block_size = 0,
            .iterative_init_diagonal = false,
            .kpoint_threads = 0,
            .iterative_reuse_vectors = true,
            .use_symmetry = false,
            .lobpcg_parallel = false,
        },
        .relax = .{
            .enabled = false,
            .algorithm = .bfgs,
            .max_iter = 100,
            .force_tol = 1e-4,
            .max_step = 0.5,
            .output_trajectory = false,
        },
        .dfpt = .{
            .enabled = false,
            .sternheimer_tol = 1e-8,
            .sternheimer_max_iter = 200,
            .scf_tol = 1e-10,
            .scf_max_iter = 50,
            .mixing_beta = 0.3,
            .alpha_shift = 0.01,
            .qpath_npoints = 0,
            .pulay_history = 8,
            .pulay_start = 4,
            .kpoint_threads = 0,
            .perturbation_threads = 1,
            .qgrid = null,
            .qpath = &.{},
        },
        .dos = .{},
        .output = .{},
        .pseudopotentials = @constCast(&[_]pseudo.Spec{
            .{ .element = @constCast("Si"), .path = @constCast("Si.upf"), .format = .upf },
        }),
    };
}

test "selectActiveStructure prefers latest relax geometry and cell" {
    const initial_cell = math.Mat3.fromRows(
        .{ .x = 1.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 1.0 },
    );
    const final_cell = math.Mat3.fromRows(
        .{ .x = 2.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 2.0 },
    );
    const initial_recip = math.reciprocal(initial_cell);
    const final_recip = math.reciprocal(final_cell);
    var initial_atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 0.1, .y = 0.2, .z = 0.3 }, .species_index = 0 },
    };
    var final_atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 0.4, .y = 0.5, .z = 0.6 }, .species_index = 0 },
    };
    var relax_result = relax.RelaxResult{
        .final_atoms = final_atoms[0..],
        .final_energy = -1.0,
        .final_forces = &[_]math.Vec3{},
        .iterations = 2,
        .converged = true,
        .trajectory = null,
        .final_cell = final_cell,
        .final_recip = final_recip,
        .final_volume = 8.0,
    };

    const active = selectActiveStructure(initial_atoms[0..], initial_cell, initial_recip, 1.0, &relax_result);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), active.cell_bohr.m[0][0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, final_recip.m[0][0]), active.recip.m[0][0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), active.volume_bohr, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), active.atoms[0].position.x, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), active.atoms[0].position.y, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), active.atoms[0].position.z, 1e-12);
}

test "post-relax outputs use active cell and active atoms" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const cfg = testOutputConfig();
    var input_atoms = [_]xyz.Atom{
        .{
            .symbol = @constCast("Si"),
            .position = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        },
    };
    const species = [_]hamiltonian.SpeciesEntry{
        .{
            .symbol = "Si",
            .upf = undefined,
            .z_valence = 4.0,
            .epsatm_ry = 0.0,
            .local_mode = .short_range,
            .local_alpha = 0.0,
        },
    };

    const initial_cell = math.Mat3.fromRows(
        .{ .x = 5.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 5.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 5.0 },
    );
    const final_cell = math.Mat3.fromRows(
        .{ .x = 2.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 3.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 4.0 },
    );
    const initial_recip = math.reciprocal(initial_cell);
    const final_recip = math.reciprocal(final_cell);
    var initial_atom_data = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 1.0, .y = 1.0, .z = 1.0 }, .species_index = 0 },
    };
    var final_atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 2.0, .y = 3.0, .z = 4.0 }, .species_index = 0 },
    };
    var relax_result = relax.RelaxResult{
        .final_atoms = final_atoms[0..],
        .final_energy = -1.0,
        .final_forces = &[_]math.Vec3{},
        .iterations = 3,
        .converged = true,
        .trajectory = null,
        .final_cell = final_cell,
        .final_recip = final_recip,
        .final_volume = 24.0,
    };

    const active = selectActiveStructure(initial_atom_data[0..], initial_cell, initial_recip, 125.0, &relax_result);
    const cell_ang = active.cell_bohr.scale(math.unitsScaleToAngstrom(.bohr));

    try output.writeRunInfo(io, tmp.dir, cfg, input_atoms[0..], cell_ang);
    try output.writeAtomsFromAtomData(io, tmp.dir, active.atoms, species[0..], math.unitsScaleToAngstrom(.bohr));

    const run_info = try tmp.dir.readFileAlloc(io, "run_info.txt", alloc, .limited(1024 * 1024));
    defer alloc.free(run_info);
    const atoms_csv = try tmp.dir.readFileAlloc(io, "atoms.csv", alloc, .limited(1024 * 1024));
    defer alloc.free(atoms_csv);

    try std.testing.expect(std.mem.indexOf(u8, run_info, "cell_angstrom = [[1.05835442, 0.00000000, 0.00000000], [0.00000000, 1.58753163, 0.00000000], [0.00000000, 0.00000000, 2.11670884]]\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, run_info, "cell_angstrom = [[2.64588605, 0.00000000, 0.00000000], [0.00000000, 2.64588605, 0.00000000], [0.00000000, 0.00000000, 2.64588605]]\n") == null);
    try std.testing.expect(std.mem.indexOf(u8, atoms_csv, "Si,1.05835442,1.58753163,2.11670884\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, atoms_csv, "Si,0.52917721,0.52917721,0.52917721\n") == null);
}
