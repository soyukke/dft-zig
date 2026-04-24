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

fn log_step(io: std.Io, msg: []const u8) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "{s}\n", .{msg});
}

fn log_phonon_frequencies(io: std.Io, frequencies_cm1: []const f64) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "phonon frequencies (cm⁻¹):\n", .{});
    for (frequencies_cm1) |f| {
        try logger.print(.info, "  {d:.2}\n", .{f});
    }
}

fn now_ns(io: std.Io) u64 {
    const ts = std.Io.Clock.Timestamp.now(io, .awake);
    return @intCast(ts.raw.nanoseconds);
}

fn elapsed_ns(io: std.Io, start_ns: u64) u64 {
    const now = now_ns(io);
    if (now <= start_ns) return 0;
    return now - start_ns;
}

const ActiveStructure = struct {
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
};

const ResolvedActiveStructure = struct {
    active: ActiveStructure,
    cell_ang: math.Mat3,
};

fn cell_volume(cell_bohr: math.Mat3) f64 {
    const a1 = cell_bohr.row(0);
    const a2 = cell_bohr.row(1);
    const a3 = cell_bohr.row(2);
    return @abs(math.Vec3.dot(a1, math.Vec3.cross(a2, a3)));
}

fn select_active_structure(
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
            .volume_bohr = result.final_volume orelse cell_volume(active_cell),
        };
    }
    return .{
        .atoms = initial_atoms,
        .cell_bohr = initial_cell_bohr,
        .recip = initial_recip,
        .volume_bohr = initial_volume_bohr,
    };
}

fn open_run_out_dir(cwd: std.Io.Dir, io: std.Io, cfg: config.Config) !std.Io.Dir {
    try cwd.createDirPath(io, cfg.out_dir);
    return cwd.openDir(io, cfg.out_dir, .{});
}

fn resolve_active_run_structure(
    setup: *const RunSetup,
    relax_result: ?*const relax.RelaxResult,
) ResolvedActiveStructure {
    const active = select_active_structure(
        setup.atom_data,
        setup.cell.cell_bohr,
        setup.cell.recip,
        setup.cell.volume_bohr,
        relax_result,
    );
    return .{
        .active = active,
        .cell_ang = active.cell_bohr.scale(math.units_scale_to_angstrom(.bohr)),
    };
}

const ScfFlowState = struct {
    active_model: model_mod.Model,
    scf_result: ?scf.ScfResult = null,
};

const BandPathState = struct {
    auto_path_result: ?kpath.auto_kpath.AutoKPathResult = null,
    kpoints: kpath.KPath,

    fn deinit(self: *BandPathState, alloc: std.mem.Allocator) void {
        self.kpoints.deinit(alloc);
        if (self.auto_path_result) |*result| result.deinit(alloc);
    }
};

fn run_main_scf_flow(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cwd: std.Io.Dir,
    cfg: config.Config,
    species: []const hamiltonian.SpeciesEntry,
    active: ActiveStructure,
    relax_result: *?relax.RelaxResult,
    timing: *output.Timing,
) !ScfFlowState {
    var flow = ScfFlowState{
        .active_model = build_active_model(species, active),
    };
    try maybe_run_scf(
        alloc,
        io,
        cfg,
        &flow.active_model,
        relax_result,
        &flow.scf_result,
        timing,
    );
    try run_post_scf_pipeline(
        alloc,
        io,
        out_dir,
        cwd,
        cfg,
        &flow.active_model,
        species,
        active,
        &flow.scf_result,
    );
    return flow;
}

fn prepare_band_kpath(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    active: ActiveStructure,
) !BandPathState {
    try log_step(io, "step: generate k-path");
    var auto_path_result: ?kpath.auto_kpath.AutoKPathResult = null;
    return .{
        .auto_path_result = auto_path_result,
        .kpoints = try generate_band_kpath(alloc, cfg, active, &auto_path_result),
    };
}

fn finalize_and_finish_run(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    atoms: []const xyz.Atom,
    resolved: ResolvedActiveStructure,
    species: []const hamiltonian.SpeciesEntry,
    kpoints: kpath.KPath,
    pseudo_data: []const pseudo.Parsed,
    flow: *ScfFlowState,
    relax_result: *?relax.RelaxResult,
    timing: *output.Timing,
    total_start_ns: u64,
) !void {
    try finalize_run_outputs(
        alloc,
        io,
        out_dir,
        cfg,
        atoms,
        resolved.cell_ang,
        resolved.active,
        species,
        kpoints,
        pseudo_data,
        &flow.active_model,
        relax_result,
        &flow.scf_result,
        timing,
    );
    try finish_run(io, out_dir, cfg, flow.scf_result, timing, total_start_ns, kpoints.points.len);
}

/// Run the current DFT workflow.
pub fn run(alloc: std.mem.Allocator, io: std.Io, cfg: config.Config, atoms: []xyz.Atom) !void {
    const total_start_ns = now_ns(io);
    var timing = output.Timing{};
    timing.cpu_start_us = output.Timing.get_cpu_time_us();

    const cwd = std.Io.Dir.cwd();
    var out_dir = try open_run_out_dir(cwd, io, cfg);
    defer out_dir.close(io);

    const setup = try prepare_run_setup(alloc, io, cfg, atoms);
    defer setup.deinit(alloc);

    timing.setup_ns = setup.setup_ns;

    // Structure relaxation
    var relax_result: ?relax.RelaxResult = null;
    defer if (relax_result) |*result| result.deinit(alloc);

    try maybe_run_relax(
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

    const resolved = resolve_active_run_structure(
        &setup,
        if (relax_result) |*rr| rr else null,
    );
    const active = resolved.active;

    var band_path = try prepare_band_kpath(alloc, io, cfg, active);
    defer band_path.deinit(alloc);

    var flow = try run_main_scf_flow(
        alloc,
        io,
        out_dir,
        cwd,
        cfg,
        setup.species,
        active,
        &relax_result,
        &timing,
    );
    defer if (flow.scf_result) |*result| result.deinit(alloc);

    try finalize_and_finish_run(
        alloc,
        io,
        out_dir,
        cfg,
        atoms,
        resolved,
        setup.species,
        band_path.kpoints,
        setup.pseudo_data,
        &flow,
        &relax_result,
        &timing,
        total_start_ns,
    );
}

/// Write all end-of-run outputs: run-info/atoms/kpoints/pseudos, and band
/// energies if a k-path was generated.
fn finalize_run_outputs(
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
    const extra_potential = select_extra_potential(relax_result, scf_result);
    const extra_potential_down = select_extra_potential_down(scf_result);

    try log_step(io, "step: write outputs");
    try write_run_outputs(
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
        const band_start_ns = now_ns(io);
        try run_band_energies(
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
        timing.band_ns = elapsed_ns(io, band_start_ns);
    }
}

fn select_extra_potential(
    relax_result: *?relax.RelaxResult,
    scf_result: *?scf.ScfResult,
) ?*hamiltonian.PotentialGrid {
    if (relax_result.*) |*rr| {
        if (rr.final_potential) |*p| return p;
    }
    if (scf_result.*) |*result| return &result.potential;
    return null;
}

fn select_extra_potential_down(scf_result: *?scf.ScfResult) ?*hamiltonian.PotentialGrid {
    if (scf_result.*) |*result| {
        if (result.potential_down) |*p| return p;
    }
    return null;
}

fn write_run_outputs(
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
    try output.write_run_info(io, out_dir, cfg, atoms, cell_ang);
    try output.write_atoms_from_atom_data(
        io,
        out_dir,
        active_atoms,
        species,
        math.units_scale_to_angstrom(.bohr),
    );
    try output.write_kpoints(io, out_dir, kpoints);
    try output.write_pseudopotentials(io, out_dir, pseudo_data);
}

fn run_band_energies(
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
    try log_step(io, "step: band energies");
    const band_paw_tabs: ?[]const paw_mod.PawTab = if (scf_result) |r| r.paw_tabs else null;
    const band_paw_dij: ?[]const []const f64 = if (scf_result) |r| r.paw_dij else null;
    try band.write_band_energies(
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
fn load_pseudopotentials(
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
fn deinit_pseudopotentials(alloc: std.mem.Allocator, items: []pseudo.Parsed) void {
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
        hamiltonian.deinit_species_entries(alloc, self.species);
        deinit_pseudopotentials(alloc, self.pseudo_data);
    }
};

fn build_cell_setup(cfg: config.Config) CellSetup {
    const unit_scale_bohr = math.units_scale_to_bohr(cfg.units);
    const cell_bohr = cfg.cell.scale(unit_scale_bohr);
    return .{
        .unit_scale_bohr = unit_scale_bohr,
        .cell_bohr = cell_bohr,
        .recip = math.reciprocal(cell_bohr),
        .volume_bohr = cell_volume(cell_bohr),
    };
}

fn prepare_run_setup(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    atoms: []const xyz.Atom,
) !RunSetup {
    const setup_start_ns = now_ns(io);
    const cell = build_cell_setup(cfg);

    try log_step(io, "step: load pseudopotentials");
    const pseudo_data = try load_pseudopotentials(alloc, io, cfg.pseudopotentials);
    errdefer deinit_pseudopotentials(alloc, pseudo_data);

    try log_step(io, "step: build species/atom data");
    const species = try hamiltonian.build_species_entries(alloc, pseudo_data);
    errdefer hamiltonian.deinit_species_entries(alloc, species);

    const atom_data = try hamiltonian.build_atom_data(
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
        .setup_ns = elapsed_ns(io, setup_start_ns),
    };
}

fn finish_run(
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    scf_result: ?scf.ScfResult,
    timing: *output.Timing,
    total_start_ns: u64,
    kpoint_count: usize,
) !void {
    timing.total_ns = elapsed_ns(io, total_start_ns);
    timing.cpu_end_us = output.Timing.get_cpu_time_us();
    try output.write_status(io, out_dir, cfg, scf_result);
    try output.write_timing(io, out_dir, timing.*, kpoint_count);
    try log_step(io, "step: done");
}

fn build_active_model(
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
fn maybe_run_scf(
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

    const scf_start_ns = now_ns(io);
    scf_result.* = try run_scf_with_relax_warm_start(
        alloc,
        io,
        cfg,
        active_model,
        if (relax_result.*) |*rr| rr else null,
    );
    timing.scf_ns = elapsed_ns(io, scf_start_ns);
}

/// Run all post-SCF analyses that depend on a converged SCF result:
/// stress tensor, DOS/PDOS, cube output, reference-JSON emission/compare,
/// and DFPT phonons.
fn run_post_scf_pipeline(
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
            try run_stress_tensor(alloc, io, cfg, result, active_model);
        }
    }

    // DOS computation
    if (cfg.dos.enabled) {
        if (scf_result.*) |*result| {
            if (result.wavefunctions) |wf_data| {
                try maybe_run_dos(alloc, io, out_dir, cfg, result, wf_data, species, active);
            }
        }
    }

    // Cube output
    if (cfg.output.cube) {
        if (scf_result.*) |*result| {
            try write_cube_output(io, out_dir, result, active, species);
        }
    }

    try handle_reference_json(alloc, io, out_dir, cwd, cfg, scf_result.*);

    // DFPT phonon calculation
    if (cfg.dfpt.enabled) {
        if (scf_result.*) |*result| {
            try run_dfpt(alloc, io, out_dir, cfg, result, active_model);
        } else {
            return error.ScfRequired;
        }
    }
}

/// Resolve the band k-path (auto-derived from the cell, or a named
/// "G-X-W-K-G-L" expression) and generate its k-point grid.
fn generate_band_kpath(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    active: ActiveStructure,
    auto_path_result: *?kpath.auto_kpath.AutoKPathResult,
) !kpath.KPath {
    // Resolve path_string ("auto" or "G-X-W-K-G-L") to BandPathPoint array
    var resolved_band = cfg.band;
    if (resolved_band.path_string) |ps| {
        auto_path_result.* = try kpath.auto_kpath.resolve_path_string(alloc, ps, active.cell_bohr);
        resolved_band.path = auto_path_result.*.?.points;
    }
    return kpath.generate(alloc, resolved_band, active.recip);
}

fn maybe_run_relax(
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

    const relax_start_ns = now_ns(io);
    relax_result.* = try run_relax_and_write_outputs(
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
    timing.relax_ns = elapsed_ns(io, relax_start_ns);
}

/// Run structural relaxation and dump its trajectory + final geometry files.
fn run_relax_and_write_outputs(
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
    try log_step(io, "step: structure relaxation start");
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
    try relax.write_output(alloc, io, out_dir, &result, species, bohr_to_ang);
    try relax.write_trajectory_xyz(io, out_dir, &result, species, cell_bohr, bohr_to_ang);

    try log_step(io, "step: structure relaxation done");
    return result;
}

/// Run the SCF cycle, forwarding cached density and LOBPCG workspaces from a
/// previous relax step if available (warm-start).
fn run_scf_with_relax_warm_start(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    active_model: *const model_mod.Model,
    relax_result: ?*relax.RelaxResult,
) !scf.ScfResult {
    try log_step(io, "step: scf start");
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
    try log_step(io, "step: scf done");
    return result;
}

fn run_stress_tensor(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    result: *scf.ScfResult,
    active_model: *const model_mod.Model,
) !void {
    try log_step(io, "step: stress tensor start");
    const stress = @import("../stress/stress.zig");
    const stress_terms = try stress.compute_stress_from_scf(
        alloc,
        io,
        result,
        cfg,
        active_model,
    );
    _ = stress_terms;
    try log_step(io, "step: stress tensor done");
}

fn write_cube_output(
    io: std.Io,
    out_dir: std.Io.Dir,
    result: *const scf.ScfResult,
    active: ActiveStructure,
    species: []const hamiltonian.SpeciesEntry,
) !void {
    try log_step(io, "step: cube output start");
    try cube.write_cube_file(
        io,
        out_dir,
        "density.cube",
        result.density,
        .{ result.grid.nx, result.grid.ny, result.grid.nz },
        active.cell_bohr,
        active.atoms,
        species,
    );
    try log_step(io, "step: cube output done");
}

/// Compute DOS (and optionally PDOS) from converged wavefunctions and write
/// the CSV outputs into `out_dir`.
fn maybe_run_dos(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    result: *scf.ScfResult,
    wf_data: scf.WavefunctionData,
    species: []const hamiltonian.SpeciesEntry,
    active: ActiveStructure,
) !void {
    try log_step(io, "step: dos start");
    var dos_result = try dos.compute_dos(
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
        try dos.write_dos_csv_named(io, out_dir, dos_result, dos_fermi, "dos_up.csv");
        var dos_down = try dos.compute_dos(
            alloc,
            wf_down,
            cfg.dos.sigma,
            cfg.dos.npoints,
            cfg.dos.emin,
            cfg.dos.emax,
            2,
        );
        defer dos_down.deinit(alloc);

        try dos.write_dos_csv_named(io, out_dir, dos_down, dos_fermi, "dos_down.csv");
    } else {
        try dos.write_dos_csv(io, out_dir, dos_result, dos_fermi);
    }
    try log_step(io, "step: dos done");

    // PDOS computation
    if (cfg.dos.pdos) {
        try log_step(io, "step: pdos start");
        var pdos_result = try pdos_mod.compute_pdos(
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

        try pdos_mod.write_pdos_csv(io, out_dir, pdos_result, dos_fermi);
        try log_step(io, "step: pdos done");
    }
}

/// Emit the converged SCF reference JSON, optionally comparing against a
/// pre-existing reference and a tolerance spec.
fn handle_reference_json(
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
            try linear_scaling.write_reference_from_scf_result(io, out_dir, path, result);
        }
        if (cfg.scf.compare_reference_json) |ref_path| {
            try compare_to_reference_json(alloc, io, out_dir, cwd, cfg, result, ref_path);
        } else if (cfg.scf.comparison_json != null or cfg.scf.compare_tolerance_json != null) {
            return error.MissingReferenceJson;
        }
    }
}

fn compare_to_reference_json(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cwd: std.Io.Dir,
    cfg: config.Config,
    result: *scf.ScfResult,
    ref_path: []const u8,
) !void {
    var reference = try linear_scaling.read_reference_json(io, alloc, cwd, ref_path);
    defer reference.deinit(alloc);

    const report = try linear_scaling.compare_reference_to_scf_result(&reference, result);
    if (cfg.scf.comparison_json) |out_path| {
        try linear_scaling.write_comparison_json(io, out_dir, out_path, report);
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

        const tol_result = linear_scaling.within_scf_tolerance(report, parsed_tol.value);
        if (!tol_result.all) {
            return error.ComparisonToleranceFailed;
        }
    }
}

/// Run DFPT (phonon-band-along-qpath or single phonon) and write its outputs.
fn run_dfpt(
    alloc: std.mem.Allocator,
    io: std.Io,
    out_dir: std.Io.Dir,
    cfg: config.Config,
    result: *scf.ScfResult,
    active_model: *const model_mod.Model,
) !void {
    const dfpt_mod = @import("../dfpt/dfpt.zig");
    if (cfg.dfpt.qpath_npoints > 0) {
        try log_step(io, "step: dfpt phonon band start");
        var band_result = if (cfg.dfpt.qgrid != null)
            try dfpt_mod.run_phonon_band_ifc(alloc, io, cfg, result, active_model)
        else
            try dfpt_mod.run_phonon_band(
                alloc,
                io,
                cfg,
                result,
                active_model,
                cfg.dfpt.qpath_npoints,
            );
        defer band_result.deinit(alloc);

        try write_phonon_band_csv(io, out_dir, band_result);
        try log_step(io, "step: dfpt phonon band done");
    } else {
        try log_step(io, "step: dfpt phonon start");
        var phonon = try dfpt_mod.run_phonon(alloc, io, cfg, result, active_model);
        defer phonon.deinit(alloc);

        try log_step(io, "step: dfpt phonon done");
        try log_phonon_frequencies(io, phonon.frequencies_cm1);
    }
}

fn write_phonon_band_csv(
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
    .cell = math.Mat3.from_rows(
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

fn test_output_config() config.Config {
    return output_test_config;
}

test "select_active_structure prefers latest relax geometry and cell" {
    const initial_cell = math.Mat3.from_rows(
        .{ .x = 1.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 1.0 },
    );
    const final_cell = math.Mat3.from_rows(
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

    const active = select_active_structure(
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

    const cfg = test_output_config();
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

    const initial_cell = math.Mat3.from_rows(
        .{ .x = 5.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 5.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 5.0 },
    );
    const final_cell = math.Mat3.from_rows(
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

    const active = select_active_structure(
        initial_atom_data[0..],
        initial_cell,
        initial_recip,
        125.0,
        &relax_result,
    );
    const cell_ang = active.cell_bohr.scale(math.units_scale_to_angstrom(.bohr));

    try output.write_run_info(io, tmp.dir, cfg, input_atoms[0..], cell_ang);
    try output.write_atoms_from_atom_data(
        io,
        tmp.dir,
        active.atoms,
        species[0..],
        math.units_scale_to_angstrom(.bohr),
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
