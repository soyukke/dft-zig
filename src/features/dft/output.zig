const std = @import("std");
const config = @import("../config/config.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpath = @import("../kpath/kpath.zig");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const timing_mod = @import("../runtime/timing.zig");
const scf = @import("../scf/scf.zig");
const xyz = @import("../structure/xyz.zig");

pub const Timing = timing_mod.Timing;

/// Write input summary for reproducibility.
pub fn writeRunInfo(
    io: std.Io,
    dir: std.Io.Dir,
    cfg: config.Config,
    atoms: []xyz.Atom,
    cell_ang: math.Mat3,
) !void {
    var file = try dir.createFile(io, "run_info.txt", .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    try writeRunInfoHeader(out, cfg);
    try writeRunInfoScf(out, cfg);
    try writeRunInfoBand(out, cfg);
    try writeRunInfoAtoms(out, cfg, atoms, cell_ang);
    try out.flush();
}

fn writeRunInfoHeader(out: anytype, cfg: config.Config) !void {
    try out.print("title = {s}\n", .{cfg.title});
    try out.print("xyz = {s}\n", .{cfg.xyz_path});
    try out.print("out_dir = {s}\n", .{cfg.out_dir});
    try out.print("units = {s}\n", .{unitsName(cfg.units)});
    try out.print("linalg_backend = {s}\n", .{linalg.backendName(cfg.linalg_backend)});
    try out.print("threads = {d}\n", .{cfg.threads});
}

fn writeRunInfoScf(out: anytype, cfg: config.Config) !void {
    try out.print("scf_solver = {s}\n", .{config.scfSolverName(cfg.scf.solver)});
    try out.print("scf_xc = {s}\n", .{config.xcFunctionalName(cfg.scf.xc)});
    try out.print("scf_smearing = {s}\n", .{config.smearingName(cfg.scf.smearing)});
    try out.print("scf_smear_ry = {d:.6}\n", .{cfg.scf.smear_ry});
    try out.print("scf_symmetry = {s}\n", .{if (cfg.scf.symmetry) "true" else "false"});
    try out.print(
        "scf_time_reversal = {s}\n",
        .{if (cfg.scf.time_reversal) "true" else "false"},
    );
    try out.print("scf_kpoint_threads = {d}\n", .{cfg.scf.kpoint_threads});
    try out.print("scf_iterative_max_iter = {d}\n", .{cfg.scf.iterative_max_iter});
    try out.print("scf_iterative_tol = {d:.6}\n", .{cfg.scf.iterative_tol});
    try out.print("scf_iterative_max_subspace = {d}\n", .{cfg.scf.iterative_max_subspace});
    try out.print("scf_iterative_block_size = {d}\n", .{cfg.scf.iterative_block_size});
    try out.print(
        "scf_iterative_init_diagonal = {s}\n",
        .{if (cfg.scf.iterative_init_diagonal) "true" else "false"},
    );
    try out.print("scf_iterative_warmup_steps = {d}\n", .{cfg.scf.iterative_warmup_steps});
    try out.print("scf_iterative_warmup_max_iter = {d}\n", .{cfg.scf.iterative_warmup_max_iter});
    try out.print("scf_iterative_warmup_tol = {d:.6}\n", .{cfg.scf.iterative_warmup_tol});
    try out.print(
        "scf_iterative_reuse_vectors = {s}\n",
        .{if (cfg.scf.iterative_reuse_vectors) "true" else "false"},
    );
    try out.print("ecut_ry = {d:.6}\n", .{cfg.scf.ecut_ry});
    try out.print("ewald_alpha = {d:.6}\n", .{cfg.ewald.alpha});
    try out.print("ewald_rcut = {d:.6}\n", .{cfg.ewald.rcut});
    try out.print("ewald_gcut = {d:.6}\n", .{cfg.ewald.gcut});
    try out.print("ewald_tol = {d:.6}\n", .{cfg.ewald.tol});
}

fn writeRunInfoBand(out: anytype, cfg: config.Config) !void {
    try out.print("band_nbands = {d}\n", .{cfg.band.nbands});
    try out.print("band_solver = {s}\n", .{config.bandSolverName(cfg.band.solver)});
    try out.print("band_iterative_max_iter = {d}\n", .{cfg.band.iterative_max_iter});
    try out.print("band_iterative_tol = {d:.6}\n", .{cfg.band.iterative_tol});
    try out.print("band_iterative_max_subspace = {d}\n", .{cfg.band.iterative_max_subspace});
    try out.print("band_iterative_block_size = {d}\n", .{cfg.band.iterative_block_size});
    try out.print(
        "band_iterative_init_diagonal = {s}\n",
        .{if (cfg.band.iterative_init_diagonal) "true" else "false"},
    );
    try out.print("band_kpoint_threads = {d}\n", .{cfg.band.kpoint_threads});
    try out.print(
        "band_iterative_reuse_vectors = {s}\n",
        .{if (cfg.band.iterative_reuse_vectors) "true" else "false"},
    );
}

fn writeRunInfoAtoms(
    out: anytype,
    cfg: config.Config,
    atoms: []xyz.Atom,
    cell_ang: math.Mat3,
) !void {
    try out.print("atoms = {d}\n", .{atoms.len});
    try out.print("pseudopotentials = {d}\n", .{cfg.pseudopotentials.len});
    for (cfg.pseudopotentials, 0..) |p, idx| {
        try out.print(
            "pseudo[{d}] = {s},{s},{s}\n",
            .{ idx, p.element, pseudo.formatName(p.format), p.path },
        );
    }
    try out.print(
        "cell_angstrom = [[{d:.8}, {d:.8}, {d:.8}], " ++
            "[{d:.8}, {d:.8}, {d:.8}], " ++
            "[{d:.8}, {d:.8}, {d:.8}]]\n",
        .{
            cell_ang.m[0][0], cell_ang.m[0][1], cell_ang.m[0][2],
            cell_ang.m[1][0], cell_ang.m[1][1], cell_ang.m[1][2],
            cell_ang.m[2][0], cell_ang.m[2][1], cell_ang.m[2][2],
        },
    );
}

/// Write k-point path CSV.
pub fn writeKpoints(io: std.Io, dir: std.Io.Dir, path: kpath.KPath) !void {
    var file = try dir.createFile(io, "band_kpoints.csv", .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    try out.writeAll("index,label,kx,ky,kz,kx_frac,ky_frac,kz_frac,dist\n");
    for (path.points, 0..) |p, idx| {
        try out.print(
            "{d},{s},{d:.10},{d:.10},{d:.10},{d:.10},{d:.10},{d:.10},{d:.10}\n",
            .{
                idx,
                p.label,
                p.k_cart.x,
                p.k_cart.y,
                p.k_cart.z,
                p.k_frac.x,
                p.k_frac.y,
                p.k_frac.z,
                p.distance,
            },
        );
    }
    try out.flush();
}

/// Write atoms to CSV in angstrom.
pub fn writeAtoms(io: std.Io, dir: std.Io.Dir, atoms: []xyz.Atom, unit_scale: f64) !void {
    var file = try dir.createFile(io, "atoms.csv", .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    try out.writeAll("symbol,x,y,z\n");
    for (atoms) |atom| {
        const pos = math.Vec3.scale(atom.position, unit_scale);
        try out.print(
            "{s},{d:.8},{d:.8},{d:.8}\n",
            .{ atom.symbol, pos.x, pos.y, pos.z },
        );
    }
    try out.flush();
}

/// Write atom data to CSV in angstrom.
pub fn writeAtomsFromAtomData(
    io: std.Io,
    dir: std.Io.Dir,
    atoms: []const hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
    bohr_to_ang: f64,
) !void {
    var file = try dir.createFile(io, "atoms.csv", .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    try out.writeAll("symbol,x,y,z\n");
    for (atoms) |atom| {
        const symbol = species[atom.species_index].symbol;
        try out.print(
            "{s},{d:.8},{d:.8},{d:.8}\n",
            .{
                symbol,
                atom.position.x * bohr_to_ang,
                atom.position.y * bohr_to_ang,
                atom.position.z * bohr_to_ang,
            },
        );
    }
    try out.flush();
}

/// Write current feature status.
pub fn writeStatus(
    io: std.Io,
    dir: std.Io.Dir,
    cfg: config.Config,
    scf_result: ?scf.ScfResult,
) !void {
    var file = try dir.createFile(io, "status.txt", .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    if (scf_result) |result| {
        try out.print("scf_enabled = true\n", .{});
        try out.print("scf_solver = {s}\n", .{config.scfSolverName(cfg.scf.solver)});
        try out.print("scf_xc = {s}\n", .{config.xcFunctionalName(cfg.scf.xc)});
        try out.print("scf_smearing = {s}\n", .{config.smearingName(cfg.scf.smearing)});
        try out.print("scf_smear_ry = {d:.6}\n", .{cfg.scf.smear_ry});
        try out.print("scf_symmetry = {s}\n", .{if (cfg.scf.symmetry) "true" else "false"});
        try out.print(
            "scf_time_reversal = {s}\n",
            .{if (cfg.scf.time_reversal) "true" else "false"},
        );
        try out.print(
            "scf_convergence_metric = {s}\n",
            .{config.convergenceMetricName(cfg.scf.convergence_metric)},
        );
        try out.print("scf_converged = {s}\n", .{if (result.converged) "true" else "false"});
        try out.print("scf_iterations = {d}\n", .{result.iterations});
        try out.print("scf_energy_total = {d:.10}\n", .{result.energy.total});
        try out.print("scf_energy_band = {d:.10}\n", .{result.energy.band});
        try out.print("scf_energy_hartree = {d:.10}\n", .{result.energy.hartree});
        try out.print("scf_energy_vxc_rho = {d:.10}\n", .{result.energy.vxc_rho});
        try out.print("scf_energy_xc = {d:.10}\n", .{result.energy.xc});
        try out.print("scf_energy_ion_ion = {d:.10}\n", .{result.energy.ion_ion});
        try out.print("scf_energy_psp_core = {d:.10}\n", .{result.energy.psp_core});
        try out.print("scf_energy_double_counting = {d:.10}\n", .{result.energy.double_counting});
        try out.print("scf_energy_local_pseudo = {d:.10}\n", .{result.energy.local_pseudo});
        try out.print("scf_energy_nonlocal_pseudo = {d:.10}\n", .{result.energy.nonlocal_pseudo});
        try out.print("scf_energy_paw_onsite = {d:.10}\n", .{result.energy.paw_onsite});
        try out.print("scf_potential_residual_rms = {d:.10}\n", .{result.potential_residual});
        if (std.math.isFinite(result.fermi_level)) {
            try out.print("scf_fermi_level_ry = {d:.10}\n", .{result.fermi_level});
        }
        try out.print("nspin = {d}\n", .{cfg.scf.nspin});
        if (cfg.scf.nspin == 2) {
            try out.print("magnetization = {d:.10}\n", .{result.magnetization});
            if (std.math.isFinite(result.fermi_level_down)) {
                try out.print("scf_fermi_level_down_ry = {d:.10}\n", .{result.fermi_level_down});
            }
        }
        try out.print("band_status = local_nonlocal_qij_scf\n", .{});
    } else {
        try out.print("scf_enabled = false\n", .{});
        try out.print("scf_solver = {s}\n", .{config.scfSolverName(cfg.scf.solver)});
        try out.print("scf_xc = {s}\n", .{config.xcFunctionalName(cfg.scf.xc)});
        try out.print("scf_smearing = {s}\n", .{config.smearingName(cfg.scf.smearing)});
        try out.print("scf_smear_ry = {d:.6}\n", .{cfg.scf.smear_ry});
        try out.print("scf_symmetry = {s}\n", .{if (cfg.scf.symmetry) "true" else "false"});
        try out.print(
            "scf_time_reversal = {s}\n",
            .{if (cfg.scf.time_reversal) "true" else "false"},
        );
        try out.print("band_status = local_nonlocal_qij_no_scf\n", .{});
    }
    try out.print("linalg_backend = {s}\n", .{linalg.backendName(cfg.linalg_backend)});
    try out.flush();
}

/// Write parsed pseudopotential metadata.
pub fn writePseudopotentials(io: std.Io, dir: std.Io.Dir, items: []pseudo.Parsed) !void {
    var file = try dir.createFile(io, "pseudopotentials.csv", .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    try out.writeAll(
        "element,format,path,header_element,z_valence,l_max,mesh_size," ++
            "r_size,rab_size,vlocal_size,beta_count,dij_size,qij_size,nlcc_size\n",
    );
    for (items) |item| {
        const header_element = item.header.element orelse "";
        const z_valence = item.header.z_valence orelse 0.0;
        const l_max = item.header.l_max orelse 0;
        const mesh_size = item.header.mesh_size orelse 0;
        const r_size = if (item.upf) |data| data.r.len else 0;
        const rab_size = if (item.upf) |data| data.rab.len else 0;
        const vlocal_size = if (item.upf) |data| data.v_local.len else 0;
        const beta_count = if (item.upf) |data| data.beta.len else 0;
        const dij_size = if (item.upf) |data| data.dij.len else 0;
        const qij_size = if (item.upf) |data| data.qij.len else 0;
        const nlcc_size = if (item.upf) |data| data.nlcc.len else 0;
        try out.print(
            "{s},{s},{s},{s},{d:.6},{d},{d},{d},{d},{d},{d},{d},{d},{d}\n",
            .{
                item.spec.element,
                pseudo.formatName(item.spec.format),
                item.spec.path,
                header_element,
                z_valence,
                l_max,
                mesh_size,
                r_size,
                rab_size,
                vlocal_size,
                beta_count,
                dij_size,
                qij_size,
                nlcc_size,
            },
        );
    }
    try out.flush();
}

/// Write timing information.
pub fn writeTiming(io: std.Io, dir: std.Io.Dir, timing: Timing, band_kpoints: usize) !void {
    var file = try dir.createFile(io, "timing.txt", .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    try out.print("setup_sec = {d:.3}\n", .{Timing.toSeconds(timing.setup_ns)});
    try out.print("relax_sec = {d:.3}\n", .{Timing.toSeconds(timing.relax_ns)});
    try out.print("scf_sec = {d:.3}\n", .{Timing.toSeconds(timing.scf_ns)});
    try out.print("band_sec = {d:.3}\n", .{Timing.toSeconds(timing.band_ns)});
    try out.print("total_sec = {d:.3}\n", .{Timing.toSeconds(timing.total_ns)});
    try out.print("cpu_sec = {d:.3}\n", .{timing.cpuSeconds()});
    try out.print("band_kpoints = {d}\n", .{band_kpoints});
    if (band_kpoints > 0 and timing.band_ns > 0) {
        const per_kpoint = Timing.toSeconds(timing.band_ns) / @as(f64, @floatFromInt(band_kpoints));
        try out.print("band_sec_per_kpoint = {d:.6}\n", .{per_kpoint});
    }
    try out.flush();
}

/// Convert units enum to name.
fn unitsName(units: math.Units) []const u8 {
    return switch (units) {
        .angstrom => "angstrom",
        .bohr => "bohr",
    };
}

test "writeAtomsFromAtomData writes atom positions in angstrom" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const atoms = [_]hamiltonian.AtomData{
        .{
            .position = .{ .x = 1.0, .y = 2.0, .z = 3.0 },
            .species_index = 0,
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

    try writeAtomsFromAtomData(
        io,
        tmp.dir,
        atoms[0..],
        species[0..],
        math.unitsScaleToAngstrom(.bohr),
    );

    const content = try tmp.dir.readFileAlloc(io, "atoms.csv", alloc, .limited(1024 * 1024));
    defer alloc.free(content);

    try std.testing.expect(std.mem.indexOf(u8, content, "symbol,x,y,z\n") != null);
    const expected_line = "Si,0.52917721,1.05835442,1.58753163\n";
    try std.testing.expect(std.mem.indexOf(u8, content, expected_line) != null);
}
