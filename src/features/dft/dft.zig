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
const model_mod = @import("model.zig");
const output = @import("output.zig");
const paw_mod = @import("../paw/paw.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const relax = @import("../relax/relax.zig");
const runtime_logging = @import("../runtime/logging.zig");
const xyz = @import("../structure/xyz.zig");

fn logStep(io: std.Io, msg: []const u8) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "{s}\n", .{msg});
}

fn logPhononFrequencies(io: std.Io, frequencies_cm1: []const f64) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "phonon frequencies (cm⁻¹):\n", .{});
    for (frequencies_cm1) |f| {
        try logger.print(.info, "  {d:.2}\n", .{f});
    }
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
    const a1 = cell_bohr.row(0);
    const a2 = cell_bohr.row(1);
    const a3 = cell_bohr.row(2);
    return @abs(math.Vec3.dot(a1, math.Vec3.cross(a2, a3)));
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

    const setup = try prepareRunSetup(alloc, io, cfg, atoms);
    defer setup.deinit(alloc);

    timing.setup_ns = setup.setup_ns;

    // Structure relaxation
    var relax_result: ?relax.RelaxResult = null;
    defer if (relax_result) |*result| result.deinit(alloc);

    try maybeRunRelax(
        alloc,
        io,
        out_dir,
        cfg,
        setup.species,
        setup.atom_data,
        setup.cell.cell_bohr,
        setup.cell.recip,
        setup.cell.volume_bohr,
        &relax_result,
        &timing,
    );

    const active = selectActiveStructure(
        setup.atom_data,
        setup.cell.cell_bohr,
        setup.cell.recip,
        setup.cell.volume_bohr,
        if (relax_result) |*rr| rr else null,
    );
    const cell_ang = active.cell_bohr.scale(math.unitsScaleToAngstrom(.bohr));

    try logStep(io, "step: generate k-path");
    var auto_path_result: ?kpath.auto_kpath.AutoKPathResult = null;
    defer if (auto_path_result) |*r| r.deinit(alloc);

    var kpoints = try generateBandKpath(alloc, cfg, active, &auto_path_result);
    defer kpoints.deinit(alloc);

    var scf_result: ?scf.ScfResult = null;
    defer if (scf_result) |*result| result.deinit(alloc);

    const active_model = buildActiveModel(setup.species, active);
    try maybeRunScf(alloc, io, cfg, &active_model, &relax_result, &scf_result, &timing);

    try runPostScfPipeline(
        alloc,
        io,
        out_dir,
        cwd,
        cfg,
        &active_model,
        setup.species,
        active,
        &scf_result,
    );

    try finalizeRunOutputs(
        alloc,
        io,
        out_dir,
        cfg,
        atoms,
        cell_ang,
        active,
        setup.species,
        kpoints,
        setup.pseudo_data,
        &active_model,
        &relax_result,
        &scf_result,
        &timing,
    );

    try finishRun(io, out_dir, cfg, scf_result, &timing, total_start_ns, kpoints.points.len);
}

/// Write all end-of-run outputs: run-info/atoms/kpoints/pseudos, and band
/// energies if a k-path was generated.
fn finalizeRunOutputs(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    atoms: []const xyz.Atom,
    cell_ang: math.Mat3,
    active: ActiveStructure,
    species: []const hamiltonian.SpeciesEntry,
    kpoints: kpath.KPath,
    pseudo_data: []const pseudo.Parsed,
    active_model: *const model_mod.Model,
    relax_result: *?relax.RelaxResult,
    scf_result: *?scf.ScfResult,
    timing: *output.Timing,
) !void {
    const extra_potential = selectExtraPotential(relax_result, scf_result);
    const extra_potential_down = selectExtraPotentialDown(scf_result);

    try logStep(io, "step: write outputs");
    try writeRunOutputs(
        io,
        out_dir,
        cfg,
        atoms,
        cell_ang,
        active.atoms,
        species,
        kpoints,
        pseudo_data,
    );

    if (kpoints.points.len > 0) {
        const band_start_ns = nowNs(io);
        try runBandEnergies(
            alloc,
            io,
            out_dir,
            cfg,
            kpoints,
            active_model,
            extra_potential,
            extra_potential_down,
            if (scf_result.*) |*r| r else null,
        );
        timing.band_ns = elapsedNs(io, band_start_ns);
    }
}

fn selectExtraPotential(
    relax_result: *?relax.RelaxResult,
    scf_result: *?scf.ScfResult,
) ?*hamiltonian.PotentialGrid {
    if (relax_result.*) |*rr| {
        if (rr.final_potential) |*p| return p;
    }
    if (scf_result.*) |*result| return &result.potential;
    return null;
}

fn selectExtraPotentialDown(scf_result: *?scf.ScfResult) ?*hamiltonian.PotentialGrid {
    if (scf_result.*) |*result| {
        if (result.potential_down) |*p| return p;
    }
    return null;
}

fn writeRunOutputs(
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    atoms: []const xyz.Atom,
    cell_ang: math.Mat3,
    active_atoms: []const hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
    kpoints: kpath.KPath,
    pseudo_data: []const pseudo.Parsed,
) !void {
    try output.writeRunInfo(io, out_dir, cfg, atoms, cell_ang);
    try output.writeAtomsFromAtomData(
        io,
        out_dir,
        active_atoms,
        species,
        math.unitsScaleToAngstrom(.bohr),
    );
    try output.writeKpoints(io, out_dir, kpoints);
    try output.writePseudopotentials(io, out_dir, pseudo_data);
}

fn runBandEnergies(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    kpoints: kpath.KPath,
    active_model: *const model_mod.Model,
    extra_potential: ?*hamiltonian.PotentialGrid,
    extra_potential_down: ?*hamiltonian.PotentialGrid,
    scf_result: ?*scf.ScfResult,
) !void {
    try logStep(io, "step: band energies");
    const band_paw_tabs: ?[]const paw_mod.PawTab = if (scf_result) |r| r.paw_tabs else null;
    const band_paw_dij: ?[]const []const f64 = if (scf_result) |r| r.paw_dij else null;
    try band.writeBandEnergies(
        alloc,
        io,
        out_dir,
        cfg,
        kpoints,
        active_model,
        extra_potential,
        extra_potential_down,
        band_paw_tabs,
        band_paw_dij,
    );
}

/// Load pseudopotentials from config list.
fn loadPseudopotentials(
    alloc: std.mem.Allocator,
    io: std.Io,
    specs: []pseudo.Spec,
) ![]pseudo.Parsed {
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

/// Unit conversion + derived quantities for the simulation cell.
const CellSetup = struct {
    unit_scale_bohr: f64,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
};

const RunSetup = struct {
    cell: CellSetup,
    pseudo_data: []pseudo.Parsed,
    species: []hamiltonian.SpeciesEntry,
    atom_data: []hamiltonian.AtomData,
    setup_ns: u64,

    fn deinit(self: *const RunSetup, alloc: std.mem.Allocator) void {
        alloc.free(self.atom_data);
        hamiltonian.deinitSpeciesEntries(alloc, self.species);
        deinitPseudopotentials(alloc, self.pseudo_data);
    }
};

fn buildCellSetup(cfg: config.Config) CellSetup {
    const unit_scale_bohr = math.unitsScaleToBohr(cfg.units);
    const cell_bohr = cfg.cell.scale(unit_scale_bohr);
    return .{
        .unit_scale_bohr = unit_scale_bohr,
        .cell_bohr = cell_bohr,
        .recip = math.reciprocal(cell_bohr),
        .volume_bohr = cellVolume(cell_bohr),
    };
}

fn prepareRunSetup(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    atoms: []const xyz.Atom,
) !RunSetup {
    const setup_start_ns = nowNs(io);
    const cell = buildCellSetup(cfg);

    try logStep(io, "step: load pseudopotentials");
    const pseudo_data = try loadPseudopotentials(alloc, io, cfg.pseudopotentials);
    errdefer deinitPseudopotentials(alloc, pseudo_data);

    try logStep(io, "step: build species/atom data");
    const species = try hamiltonian.buildSpeciesEntries(alloc, pseudo_data);
    errdefer hamiltonian.deinitSpeciesEntries(alloc, species);

    const atom_data = try hamiltonian.buildAtomData(
        alloc,
        atoms,
        cell.unit_scale_bohr,
        species,
    );
    errdefer alloc.free(atom_data);

    return .{
        .cell = cell,
        .pseudo_data = pseudo_data,
        .species = species,
        .atom_data = atom_data,
        .setup_ns = elapsedNs(io, setup_start_ns),
    };
}

fn finishRun(
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    scf_result: ?scf.ScfResult,
    timing: *output.Timing,
    total_start_ns: u64,
    kpoint_count: usize,
) !void {
    timing.total_ns = elapsedNs(io, total_start_ns);
    timing.cpu_end_us = output.Timing.getCpuTimeUs();
    try output.writeStatus(io, out_dir, cfg, scf_result);
    try output.writeTiming(io, out_dir, timing.*, kpoint_count);
    try logStep(io, "step: done");
}

fn buildActiveModel(
    species: []const hamiltonian.SpeciesEntry,
    active: ActiveStructure,
) model_mod.Model {
    return .{
        .species = species,
        .atoms = active.atoms,
        .cell_bohr = active.cell_bohr,
        .recip = active.recip,
        .volume_bohr = active.volume_bohr,
    };
}

/// Skip the post-relax SCF when the relax pass already produced a converged
/// potential AND no reference-JSON consumer needs the fresh result.
fn maybeRunScf(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    active_model: *const model_mod.Model,
    relax_result: *?relax.RelaxResult,
    scf_result: *?scf.ScfResult,
    timing: *output.Timing,
) !void {
    const have_relax_potential = if (relax_result.*) |rr| rr.final_potential != null else false;
    const needs_reference = cfg.scf.reference_json != null or
        cfg.scf.compare_reference_json != null;
    if (!(cfg.scf.enabled and !(have_relax_potential and !needs_reference))) return;

    const scf_start_ns = nowNs(io);
    scf_result.* = try runScfWithRelaxWarmStart(
        alloc,
        io,
        cfg,
        active_model,
        if (relax_result.*) |*rr| rr else null,
    );
    timing.scf_ns = elapsedNs(io, scf_start_ns);
}

/// Run all post-SCF analyses that depend on a converged SCF result:
/// stress tensor, DOS/PDOS, cube output, reference-JSON emission/compare,
/// and DFPT phonons.
fn runPostScfPipeline(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cwd: std.Io.Dir,
    cfg: config.Config,
    active_model: *const model_mod.Model,
    species: []const hamiltonian.SpeciesEntry,
    active: ActiveStructure,
    scf_result: *?scf.ScfResult,
) !void {
    // Stress tensor computation
    if (cfg.scf.compute_stress) {
        if (scf_result.*) |*result| {
            try runStressTensor(alloc, io, cfg, result, active_model);
        }
    }

    // DOS computation
    if (cfg.dos.enabled) {
        if (scf_result.*) |*result| {
            if (result.wavefunctions) |wf_data| {
                try maybeRunDos(alloc, io, out_dir, cfg, result, wf_data, species, active);
            }
        }
    }

    // Cube output
    if (cfg.output.cube) {
        if (scf_result.*) |*result| {
            try writeCubeOutput(io, out_dir, result, active, species);
        }
    }

    try handleReferenceJson(alloc, io, out_dir, cwd, cfg, scf_result.*);

    // DFPT phonon calculation
    if (cfg.dfpt.enabled) {
        if (scf_result.*) |*result| {
            try runDfpt(alloc, io, out_dir, cfg, result, active_model);
        } else {
            return error.ScfRequired;
        }
    }
}

/// Resolve the band k-path (auto-derived from the cell, or a named
/// "G-X-W-K-G-L" expression) and generate its k-point grid.
fn generateBandKpath(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    active: ActiveStructure,
    auto_path_result: *?kpath.auto_kpath.AutoKPathResult,
) !kpath.KPath {
    // Resolve path_string ("auto" or "G-X-W-K-G-L") to BandPathPoint array
    var resolved_band = cfg.band;
    if (resolved_band.path_string) |ps| {
        auto_path_result.* = try kpath.auto_kpath.resolvePathString(alloc, ps, active.cell_bohr);
        resolved_band.path = auto_path_result.*.?.points;
    }
    return kpath.generate(alloc, resolved_band, active.recip);
}

fn maybeRunRelax(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atom_data: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
    relax_result: *?relax.RelaxResult,
    timing: *output.Timing,
) !void {
    if (!cfg.relax.enabled) return;

    const relax_start_ns = nowNs(io);
    relax_result.* = try runRelaxAndWriteOutputs(
        alloc,
        io,
        out_dir,
        cfg,
        species,
        atom_data,
        cell_bohr,
        recip,
        volume_bohr,
    );
    timing.relax_ns = elapsedNs(io, relax_start_ns);
}

/// Run structural relaxation and dump its trajectory + final geometry files.
fn runRelaxAndWriteOutputs(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atom_data: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
) !relax.RelaxResult {
    try logStep(io, "step: structure relaxation start");
    var result = try relax.run(
        alloc,
        io,
        cfg,
        species,
        atom_data,
        cell_bohr,
        recip,
        volume_bohr,
    );
    errdefer result.deinit(alloc);

    // Write relaxation output files
    const bohr_to_ang = 0.529177; // 1 Bohr = 0.529177 Angstrom
    try relax.writeOutput(alloc, io, out_dir, &result, species, bohr_to_ang);
    try relax.writeTrajectoryXyz(io, out_dir, &result, species, cell_bohr, bohr_to_ang);

    try logStep(io, "step: structure relaxation done");
    return result;
}

/// Run the SCF cycle, forwarding cached density and LOBPCG workspaces from a
/// previous relax step if available (warm-start).
fn runScfWithRelaxWarmStart(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    active_model: *const model_mod.Model,
    relax_result: ?*relax.RelaxResult,
) !scf.ScfResult {
    try logStep(io, "step: scf start");
    // density and kpoint_cache are borrowed (scf.run copies internally)
    // apply_caches ownership is transferred to scf.run
    const init_density: ?[]const f64 = if (relax_result) |rr| rr.final_density else null;
    const init_kpoint_cache: ?[]scf.KpointCache = if (relax_result) |rr|
        rr.final_kpoint_cache
    else
        null;
    var init_apply_caches: ?[]scf.KpointApplyCache = null;
    if (relax_result) |rr| {
        init_apply_caches = rr.final_apply_caches;
        rr.final_apply_caches = null; // transfer ownership
    }
    const result = try scf.run(.{
        .alloc = alloc,
        .io = io,
        .cfg = cfg,
        .model = active_model,
        .initial_density = init_density,
        .initial_kpoint_cache = init_kpoint_cache,
        .initial_apply_caches = init_apply_caches,
    });
    try logStep(io, "step: scf done");
    return result;
}

fn runStressTensor(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    result: *scf.ScfResult,
    active_model: *const model_mod.Model,
) !void {
    try logStep(io, "step: stress tensor start");
    const stress = @import("../stress/stress.zig");
    const stress_terms = try stress.computeStressFromScf(
        alloc,
        io,
        result,
        cfg,
        active_model,
    );
    _ = stress_terms;
    try logStep(io, "step: stress tensor done");
}

fn writeCubeOutput(
    io: std.Io,
    out_dir: std.Io.Dir,
    result: *const scf.ScfResult,
    active: ActiveStructure,
    species: []const hamiltonian.SpeciesEntry,
) !void {
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

/// Compute DOS (and optionally PDOS) from converged wavefunctions and write
/// the CSV outputs into `out_dir`.
fn maybeRunDos(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    result: *scf.ScfResult,
    wf_data: scf.WavefunctionData,
    species: []const hamiltonian.SpeciesEntry,
    active: ActiveStructure,
) !void {
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

    const dos_fermi = if (std.math.isNan(result.fermi_level))
        wf_data.fermi_level
    else
        result.fermi_level;
    if (result.wavefunctions_down) |wf_down| {
        try dos.writeDosCSVNamed(io, out_dir, dos_result, dos_fermi, "dos_up.csv");
        var dos_down = try dos.computeDos(
            alloc,
            wf_down,
            cfg.dos.sigma,
            cfg.dos.npoints,
            cfg.dos.emin,
            cfg.dos.emax,
            2,
        );
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

/// Emit the converged SCF reference JSON, optionally comparing against a
/// pre-existing reference and a tolerance spec.
fn handleReferenceJson(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cwd: std.Io.Dir,
    cfg: config.Config,
    scf_result: ?scf.ScfResult,
) !void {
    if (cfg.scf.reference_json != null or
        cfg.scf.compare_reference_json != null or
        cfg.scf.comparison_json != null or
        cfg.scf.compare_tolerance_json != null)
    {
        if (scf_result == null) return error.ScfDisabled;
    }

    var maybe_result = scf_result;
    if (maybe_result) |*result| {
        if (cfg.scf.reference_json) |path| {
            try linear_scaling.writeReferenceFromScfResult(io, out_dir, path, result);
        }
        if (cfg.scf.compare_reference_json) |ref_path| {
            try compareToReferenceJson(alloc, io, out_dir, cwd, cfg, result, ref_path);
        } else if (cfg.scf.comparison_json != null or cfg.scf.compare_tolerance_json != null) {
            return error.MissingReferenceJson;
        }
    }
}

fn compareToReferenceJson(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cwd: std.Io.Dir,
    cfg: config.Config,
    result: *scf.ScfResult,
    ref_path: []const u8,
) !void {
    var reference = try linear_scaling.readReferenceJson(io, alloc, cwd, ref_path);
    defer reference.deinit(alloc);

    const report = try linear_scaling.compareReferenceToScfResult(&reference, result);
    if (cfg.scf.comparison_json) |out_path| {
        try linear_scaling.writeComparisonJson(io, out_dir, out_path, report);
    } else {
        return error.MissingComparisonOutput;
    }
    if (cfg.scf.compare_tolerance_json) |tol_path| {
        const tol_content = try cwd.readFileAlloc(
            io,
            tol_path,
            alloc,
            .limited(1024 * 1024),
        );
        defer alloc.free(tol_content);

        var parsed_tol = try std.json.parseFromSlice(
            linear_scaling.ScfTolerance,
            alloc,
            tol_content,
            .{},
        );
        defer parsed_tol.deinit();

        const tol_result = linear_scaling.withinScfTolerance(report, parsed_tol.value);
        if (!tol_result.all) {
            return error.ComparisonToleranceFailed;
        }
    }
}

/// Run DFPT (phonon-band-along-qpath or single phonon) and write its outputs.
fn runDfpt(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    result: *scf.ScfResult,
    active_model: *const model_mod.Model,
) !void {
    const dfpt_mod = @import("../dfpt/dfpt.zig");
    if (cfg.dfpt.qpath_npoints > 0) {
        try logStep(io, "step: dfpt phonon band start");
        var band_result = if (cfg.dfpt.qgrid != null)
            try dfpt_mod.runPhononBandIFC(alloc, io, cfg, result, active_model)
        else
            try dfpt_mod.runPhononBand(
                alloc,
                io,
                cfg,
                result,
                active_model,
                cfg.dfpt.qpath_npoints,
            );
        defer band_result.deinit(alloc);

        try writePhononBandCsv(io, out_dir, band_result);
        try logStep(io, "step: dfpt phonon band done");
    } else {
        try logStep(io, "step: dfpt phonon start");
        var phonon = try dfpt_mod.runPhonon(alloc, io, cfg, result, active_model);
        defer phonon.deinit(alloc);

        try logStep(io, "step: dfpt phonon done");
        try logPhononFrequencies(io, phonon.frequencies_cm1);
    }
}

fn writePhononBandCsv(
    io: std.Io,
    out_dir: std.Io.Dir,
    band_result: anytype,
) !void {
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

const output_test_config: config.Config = .{
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

fn testOutputConfig() config.Config {
    return output_test_config;
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

    const active = selectActiveStructure(
        initial_atoms[0..],
        initial_cell,
        initial_recip,
        1.0,
        &relax_result,
    );

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

    const active = selectActiveStructure(
        initial_atom_data[0..],
        initial_cell,
        initial_recip,
        125.0,
        &relax_result,
    );
    const cell_ang = active.cell_bohr.scale(math.unitsScaleToAngstrom(.bohr));

    try output.writeRunInfo(io, tmp.dir, cfg, input_atoms[0..], cell_ang);
    try output.writeAtomsFromAtomData(
        io,
        tmp.dir,
        active.atoms,
        species[0..],
        math.unitsScaleToAngstrom(.bohr),
    );

    const run_info = try tmp.dir.readFileAlloc(io, "run_info.txt", alloc, .limited(1024 * 1024));
    defer alloc.free(run_info);

    const atoms_csv = try tmp.dir.readFileAlloc(io, "atoms.csv", alloc, .limited(1024 * 1024));
    defer alloc.free(atoms_csv);

    try std.testing.expect(std.mem.indexOf(
        u8,
        run_info,
        "cell_angstrom = [[1.05835442, 0.00000000, 0.00000000]," ++
            " [0.00000000, 1.58753163, 0.00000000]," ++
            " [0.00000000, 0.00000000, 2.11670884]]\n",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        run_info,
        "cell_angstrom = [[2.64588605, 0.00000000, 0.00000000]," ++
            " [0.00000000, 2.64588605, 0.00000000]," ++
            " [0.00000000, 0.00000000, 2.64588605]]\n",
    ) == null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        atoms_csv,
        "Si,1.05835442,1.58753163,2.11670884\n",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        atoms_csv,
        "Si,0.52917721,0.52917721,0.52917721\n",
    ) == null);
}
