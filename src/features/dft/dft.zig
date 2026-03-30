const std = @import("std");
const band = @import("../band/band.zig");
const config = @import("../config/config.zig");
const cube = @import("cube.zig");
const dos = @import("../dos/dos.zig");
const pdos_mod = @import("../dos/pdos.zig");
const scf = @import("../scf/scf.zig");
const apply = @import("../scf/apply.zig");
const kpath = @import("../kpath/kpath.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const linear_scaling = @import("../linear_scaling/linear_scaling.zig");
const math = @import("../math/math.zig");
const output = @import("output.zig");
const paw_mod = @import("../paw/paw.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const relax = @import("../relax/relax.zig");
const xyz = @import("../structure/xyz.zig");

fn logStep(msg: []const u8) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.fs.File.stderr().writer(&buffer);
    const out = &writer.interface;
    try out.print("{s}\n", .{msg});
    try out.flush();
}

/// Run the current DFT workflow.
pub fn run(alloc: std.mem.Allocator, cfg: config.Config, atoms: []xyz.Atom) !void {
    const total_start = std.time.Instant.now() catch null;
    var timing = output.Timing{};
    timing.cpu_start_us = output.Timing.getCpuTimeUs();

    try std.fs.cwd().makePath(cfg.out_dir);
    var out_dir = try std.fs.cwd().openDir(cfg.out_dir, .{});
    defer out_dir.close();

    const setup_start = std.time.Instant.now() catch null;

    const unit_scale_ang = math.unitsScaleToAngstrom(cfg.units);
    const unit_scale_bohr = math.unitsScaleToBohr(cfg.units);
    const cell_ang = cfg.cell.scale(unit_scale_ang);
    const cell_bohr = cfg.cell.scale(unit_scale_bohr);
    const recip = math.reciprocal(cell_bohr);
    const volume_bohr = @abs(math.Vec3.dot(cell_bohr.row(0), math.Vec3.cross(cell_bohr.row(1), cell_bohr.row(2))));

    try logStep("step: generate k-path");
    var kpoints = try kpath.generate(alloc, cfg.band, recip);
    defer kpoints.deinit(alloc);

    try logStep("step: load pseudopotentials");
    const pseudo_data = try loadPseudopotentials(alloc, cfg.pseudopotentials);
    defer deinitPseudopotentials(alloc, pseudo_data);
    try logStep("step: build species/atom data");
    const species = try hamiltonian.buildSpeciesEntries(alloc, pseudo_data);
    defer hamiltonian.deinitSpeciesEntries(alloc, species);
    var atom_data = try hamiltonian.buildAtomData(alloc, atoms, unit_scale_bohr, species);
    defer alloc.free(atom_data);

    if (setup_start) |t0| {
        if (std.time.Instant.now() catch null) |t1| {
            timing.setup_ns = t1.since(t0);
        }
    }

    // Structure relaxation
    var relax_result: ?relax.RelaxResult = null;
    defer if (relax_result) |*result| result.deinit(alloc);

    if (cfg.relax.enabled) {
        const relax_start = std.time.Instant.now() catch null;
        try logStep("step: structure relaxation start");
        relax_result = try relax.run(alloc, cfg, species, atom_data, cell_bohr, recip, volume_bohr);

        // Write relaxation output files
        const bohr_to_ang = 0.529177; // 1 Bohr = 0.529177 Angstrom
        try relax.writeOutput(alloc, out_dir, &relax_result.?, species, bohr_to_ang);
        try relax.writeTrajectoryXyz(out_dir, &relax_result.?, species, cell_bohr, bohr_to_ang);

        // Update atom_data with relaxed positions
        if (relax_result.?.converged) {
            alloc.free(atom_data);
            atom_data = relax_result.?.final_atoms;
            relax_result.?.final_atoms = &[_]hamiltonian.AtomData{}; // Transfer ownership
        }

        try logStep("step: structure relaxation done");
        if (relax_start) |t0| {
            if (std.time.Instant.now() catch null) |t1| {
                timing.relax_ns = t1.since(t0);
            }
        }
    }

    var scf_result: ?scf.ScfResult = null;
    defer if (scf_result) |*result| result.deinit(alloc);

    // Skip post-relax SCF if we have the converged potential from relax
    const have_relax_potential = if (relax_result) |rr| rr.final_potential != null else false;
    const needs_reference = cfg.scf.reference_json != null or cfg.scf.compare_reference_json != null;

    if (cfg.scf.enabled and !(have_relax_potential and !needs_reference)) {
        const scf_start = std.time.Instant.now() catch null;
        try logStep("step: scf start");
        // Use caches from relax if available
        // density and kpoint_cache are borrowed (scf.run copies internally)
        // apply_caches ownership is transferred to scf.run
        const init_density: ?[]const f64 = if (relax_result) |rr| rr.final_density else null;
        const init_kpoint_cache: ?[]scf.KpointCache = if (relax_result) |rr| rr.final_kpoint_cache else null;
        var init_apply_caches: ?[]apply.KpointApplyCache = null;
        if (relax_result) |*rr| {
            init_apply_caches = rr.final_apply_caches;
            rr.final_apply_caches = null; // transfer ownership
        }
        const result = try scf.run(.{
            .alloc = alloc,
            .cfg = cfg,
            .species = species,
            .atoms = atom_data,
            .recip = recip,
            .volume_bohr = volume_bohr,
            .initial_density = init_density,
            .initial_kpoint_cache = init_kpoint_cache,
            .initial_apply_caches = init_apply_caches,
        });
        scf_result = result;
        try logStep("step: scf done");
        if (scf_start) |t0| {
            if (std.time.Instant.now() catch null) |t1| {
                timing.scf_ns = t1.since(t0);
            }
        }
    }

    // Stress tensor computation
    if (cfg.scf.compute_stress) {
        if (scf_result) |*result| {
            try logStep("step: stress tensor start");
            const stress = @import("../stress/stress.zig");
            const stress_terms = try stress.computeStressFromScf(alloc, result, cfg, species, atom_data);
            _ = stress_terms;
            try logStep("step: stress tensor done");
        }
    }

    // DOS computation
    if (cfg.dos.enabled) {
        if (scf_result) |*result| {
            if (result.wavefunctions) |wf_data| {
                try logStep("step: dos start");
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
                    try dos.writeDosCSVNamed(out_dir, dos_result, dos_fermi, "dos_up.csv");
                    var dos_down = try dos.computeDos(alloc, wf_down, cfg.dos.sigma, cfg.dos.npoints, cfg.dos.emin, cfg.dos.emax, 2);
                    defer dos_down.deinit(alloc);
                    try dos.writeDosCSVNamed(out_dir, dos_down, dos_fermi, "dos_down.csv");
                } else {
                    try dos.writeDosCSV(out_dir, dos_result, dos_fermi);
                }
                try logStep("step: dos done");

                // PDOS computation
                if (cfg.dos.pdos) {
                    try logStep("step: pdos start");
                    var pdos_result = try pdos_mod.computePdos(
                        alloc,
                        wf_data,
                        species,
                        atom_data,
                        recip,
                        volume_bohr,
                        cfg.dos.sigma,
                        cfg.dos.npoints,
                        cfg.dos.emin,
                        cfg.dos.emax,
                        cfg.scf.nspin,
                    );
                    defer pdos_result.deinit(alloc);
                    try pdos_mod.writePdosCSV(out_dir, pdos_result, dos_fermi);
                    try logStep("step: pdos done");
                }
            }
        }
    }

    // Cube output
    if (cfg.output.cube) {
        if (scf_result) |*result| {
            try logStep("step: cube output start");
            try cube.writeCubeFile(
                out_dir,
                "density.cube",
                result.density,
                .{ result.grid.nx, result.grid.ny, result.grid.nz },
                cell_bohr,
                atom_data,
                species,
            );
            try logStep("step: cube output done");
        }
    }

    if (cfg.scf.reference_json != null or cfg.scf.compare_reference_json != null or cfg.scf.comparison_json != null or cfg.scf.compare_tolerance_json != null) {
        if (scf_result == null) return error.ScfDisabled;
    }

    if (scf_result) |*result| {
        if (cfg.scf.reference_json) |path| {
            try linear_scaling.writeReferenceFromScfResult(out_dir, path, result);
        }
        if (cfg.scf.compare_reference_json) |ref_path| {
            var reference = try linear_scaling.readReferenceJson(alloc, std.fs.cwd(), ref_path);
            defer reference.deinit(alloc);
            const report = try linear_scaling.compareReferenceToScfResult(&reference, result);
            if (cfg.scf.comparison_json) |out_path| {
                try linear_scaling.writeComparisonJson(out_dir, out_path, report);
            } else {
                return error.MissingComparisonOutput;
            }
            if (cfg.scf.compare_tolerance_json) |tol_path| {
                const tol_content = try std.fs.cwd().readFileAlloc(alloc, tol_path, 1024 * 1024);
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
                try logStep("step: dfpt phonon band start");
                var band_result = if (cfg.dfpt.qgrid != null)
                    try dfpt_mod.runPhononBandIFC(
                        alloc,
                        cfg,
                        result,
                        species,
                        atom_data,
                        cell_bohr,
                        recip,
                        volume_bohr,
                    )
                else
                    try dfpt_mod.runPhononBand(
                        alloc,
                        cfg,
                        result,
                        species,
                        atom_data,
                        cell_bohr,
                        recip,
                        volume_bohr,
                        cfg.dfpt.qpath_npoints,
                    );
                defer band_result.deinit(alloc);

                // Write phonon band CSV
                {
                    const csv_file = try out_dir.createFile("phonon_band.csv", .{});
                    defer csv_file.close();
                    var csv_buf: [256]u8 = undefined;
                    var csv_writer = csv_file.writer(&csv_buf);
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
                try logStep("step: dfpt phonon band done");
            } else {
                try logStep("step: dfpt phonon start");
                var phonon = try dfpt_mod.runPhonon(alloc, cfg, result, species, atom_data, cell_bohr, recip, volume_bohr);
                defer phonon.deinit(alloc);
                try logStep("step: dfpt phonon done");
                // Print frequencies
                {
                    var buffer: [256]u8 = undefined;
                    var writer = std.fs.File.stderr().writer(&buffer);
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

    try logStep("step: write outputs");
    try output.writeRunInfo(out_dir, cfg, atoms, cell_ang);
    try output.writeAtoms(out_dir, atoms, unit_scale_ang);
    try output.writeKpoints(out_dir, kpoints);
    try output.writePseudopotentials(out_dir, pseudo_data);

    if (kpoints.points.len > 0) {
        const band_start = std.time.Instant.now() catch null;
        try logStep("step: band energies");
        const band_paw_tabs: ?[]const paw_mod.PawTab = if (scf_result) |r| r.paw_tabs else null;
        const band_paw_dij: ?[]const []const f64 = if (scf_result) |r| r.paw_dij else null;
        try band.writeBandEnergies(alloc, out_dir, cfg, kpoints, species, atom_data, cell_bohr, recip, volume_bohr, extra_potential, extra_potential_down, band_paw_tabs, band_paw_dij);
        if (band_start) |t0| {
            if (std.time.Instant.now() catch null) |t1| {
                timing.band_ns = t1.since(t0);
            }
        }
    }

    if (total_start) |t0| {
        if (std.time.Instant.now() catch null) |t1| {
            timing.total_ns = t1.since(t0);
        }
    }
    timing.cpu_end_us = output.Timing.getCpuTimeUs();

    try output.writeStatus(out_dir, cfg, scf_result);
    try output.writeTiming(out_dir, timing, kpoints.points.len);
    try logStep("step: done");
}

/// Load pseudopotentials from config list.
fn loadPseudopotentials(alloc: std.mem.Allocator, specs: []pseudo.Spec) ![]pseudo.Parsed {
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
        try list.append(alloc, try pseudo.load(alloc, spec));
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
