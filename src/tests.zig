const std = @import("std");
const builtin = @import("builtin");
const math = @import("features/math/math.zig");
const hamiltonian = @import("features/hamiltonian/hamiltonian.zig");
const spacegroup = @import("features/symmetry/spacegroup.zig");
const pseudo = @import("features/pseudopotential/pseudopotential.zig");
const form_factor = @import("features/pseudopotential/form_factor.zig");
const nonlocal = @import("features/pseudopotential/nonlocal.zig");
const plane_wave = @import("features/plane_wave/basis.zig");
const ewald = @import("features/ewald/ewald.zig");
const config = @import("features/config/config.zig");
const scf = @import("features/scf/scf.zig");
const forces = @import("features/forces/forces.zig");
const fft = @import("features/fft/fft.zig");

const verbose_tests = false;
const scf_fd_verbose = false;
const scf_fd_enabled = true;

/// Skip test if a required file does not exist (e.g. pseudo/ files in CI).
fn requireFile(path: []const u8) !void {
    std.fs.cwd().access(path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("  [SKIP] file not found: {s}\n", .{path});
            return error.SkipZigTest;
        }
        return err;
    };
}

fn vprint(comptime fmt: []const u8, args: anytype) void {
    if (verbose_tests) std.debug.print(fmt, args);
}

fn scfPrint(comptime fmt: []const u8, args: anytype) void {
    if (scf_fd_verbose) std.debug.print(fmt, args);
}

test "spacegroup silicon conventional" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = 5.431;
    const half = a / 2.0;
    const quarter = a / 4.0;
    const three_quarter = 3.0 * a / 4.0;
    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = a, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = a },
    );

    const atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .species_index = 0 },
        .{ .position = .{ .x = 0.0, .y = half, .z = half }, .species_index = 0 },
        .{ .position = .{ .x = half, .y = 0.0, .z = half }, .species_index = 0 },
        .{ .position = .{ .x = half, .y = half, .z = 0.0 }, .species_index = 0 },
        .{ .position = .{ .x = quarter, .y = quarter, .z = quarter }, .species_index = 0 },
        .{ .position = .{ .x = quarter, .y = three_quarter, .z = three_quarter }, .species_index = 0 },
        .{ .position = .{ .x = three_quarter, .y = quarter, .z = three_quarter }, .species_index = 0 },
        .{ .position = .{ .x = three_quarter, .y = three_quarter, .z = quarter }, .species_index = 0 },
    };

    const info_opt = try spacegroup.detectSpaceGroupFromAtoms(alloc, cell, atoms[0..], 1e-5);
    try std.testing.expect(info_opt != null);
    var info = info_opt.?;
    defer info.deinit(alloc);
    try std.testing.expectEqual(@as(i32, 227), info.number);
    try std.testing.expect(std.mem.eql(u8, info.international_short, "Fd-3m"));
}

test "spacegroup silicon axis swap" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = 5.431;
    const half = a / 2.0;
    const quarter = a / 4.0;
    const three_quarter = 3.0 * a / 4.0;
    const cell = math.Mat3.fromRows(
        .{ .x = 0.0, .y = a, .z = 0.0 },
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = a },
    );

    const atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .species_index = 0 },
        .{ .position = .{ .x = 0.0, .y = half, .z = half }, .species_index = 0 },
        .{ .position = .{ .x = half, .y = 0.0, .z = half }, .species_index = 0 },
        .{ .position = .{ .x = half, .y = half, .z = 0.0 }, .species_index = 0 },
        .{ .position = .{ .x = quarter, .y = quarter, .z = quarter }, .species_index = 0 },
        .{ .position = .{ .x = quarter, .y = three_quarter, .z = three_quarter }, .species_index = 0 },
        .{ .position = .{ .x = three_quarter, .y = quarter, .z = three_quarter }, .species_index = 0 },
        .{ .position = .{ .x = three_quarter, .y = three_quarter, .z = quarter }, .species_index = 0 },
    };

    const info_opt = try spacegroup.detectSpaceGroupFromAtoms(alloc, cell, atoms[0..], 1e-5);
    try std.testing.expect(info_opt != null);
    var info = info_opt.?;
    defer info.deinit(alloc);
    try std.testing.expectEqual(@as(i32, 227), info.number);
    try std.testing.expect(std.mem.eql(u8, info.international_short, "Fd-3m"));
}

// Test local pseudopotential form factor
test "local pseudopotential V(q) for Carbon" {
    const allocator = std.testing.allocator;

    // Load C.upf using pseudo.load
    var element_buf: [2]u8 = .{ 'C', 0 };
    var path_buf: [20]u8 = undefined;
    const path_slice = "pseudo/C.upf";
    try requireFile(path_slice);
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..1],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(allocator, spec);
    defer parsed.deinit(allocator);

    const upf = parsed.upf orelse return error.NoUpfData;

    // Test V(q) at various q values, including typical smallest G values
    const q_values = [_]f64{ 0.1, 0.166, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0 };
    vprint("\n=== Local Pseudopotential V(q) ===\n", .{});
    for (q_values) |q| {
        const vq = form_factor.localVq(upf, q);
        vprint("V(q={d:.1}) = {d:.4} Ry\n", .{ q, vq });
    }

    // V(q) should be finite for q > 0
    const v_at_1 = form_factor.localVq(upf, 1.0);
    try std.testing.expect(std.math.isFinite(v_at_1));

    // Test with Coulomb tail correction
    // Note: In Rydberg units, the Coulomb potential is -2Z/r, so V_Coul(q) = -8πZ/q²
    vprint("\n=== V(q) with Coulomb tail correction (Rydberg units) ===\n", .{});
    const z_val: f64 = 4.0;
    for (q_values) |q| {
        const vq_raw = form_factor.localVq(upf, q);
        const vq_tail = form_factor.localVqWithTail(upf, z_val, q);
        // Coulomb in Rydberg: -8πZ/q² (factor of 2 for Ry vs Ha)
        const v_coulomb_ry = -8.0 * std.math.pi * z_val / (q * q);
        const vq_sr = vq_tail - v_coulomb_ry;
        vprint("q={d:.2}: V_raw={d:.1}, V_tail={d:.1}, V_Coul_Ry={d:.1}, V_SR={d:.1} Ry\n", .{ q, vq_raw, vq_tail, v_coulomb_ry, vq_sr });
    }

    // Test Ewald-compensated form factor
    // Use typical alpha = 5/L_min where L_min ≈ 4.65 Bohr for graphene
    const alpha: f64 = 1.07; // roughly 5/4.65
    vprint("\n=== V(q) with Ewald compensation (α={d:.2}) ===\n", .{alpha});
    for (q_values) |q| {
        const vq_ewald = form_factor.localVqEwald(upf, z_val, q, alpha);
        vprint("q={d:.1}: V_Ewald={d:.4} Ry\n", .{ q, vq_ewald });
    }
    // Test at q=0 (G=0 component - should be finite)
    const vq_ewald_0 = form_factor.localVqEwald(upf, z_val, 0.0, alpha);
    vprint("q=0: V_Ewald={d:.4} Ry (should be finite)\n", .{vq_ewald_0});

    // The Ewald form factor should be finite everywhere
    const vq_ewald_small = form_factor.localVqEwald(upf, z_val, 0.166, alpha);
    try std.testing.expect(std.math.isFinite(vq_ewald_small));
    // Note: The Ewald approach still gives large values due to pseudopotential structure
    vprint("V_Ewald(0.166) = {d:.2} Ry\n", .{vq_ewald_small});
}

fn indexToFreq(i: usize, n: usize) i32 {
    const half = (n - 1) / 2;
    return if (i <= half) @as(i32, @intCast(i)) else @as(i32, @intCast(i)) - @as(i32, @intCast(n));
}

fn densityToReciprocal(
    alloc: std.mem.Allocator,
    grid: forces.Grid,
    density: []const f64,
    fft_backend: config.FftBackend,
) ![]math.Complex {
    const nx = grid.nx;
    const ny = grid.ny;
    const nz = grid.nz;
    const total = nx * ny * nz;

    if (density.len != total) return error.InvalidDensitySize;

    var data = try alloc.alloc(math.Complex, total);
    defer alloc.free(data);
    for (density, 0..) |d, i| {
        data[i] = math.complex.init(d, 0.0);
    }

    var plan = try fft.Fft3dPlan.initWithBackend(alloc, nx, ny, nz, fft_backend);
    defer plan.deinit(alloc);
    plan.forward(data);

    const scale = 1.0 / @as(f64, @floatFromInt(total));
    var out = try alloc.alloc(math.Complex, total);

    var idx: usize = 0;
    var z: usize = 0;
    while (z < nz) : (z += 1) {
        var y: usize = 0;
        while (y < ny) : (y += 1) {
            var x: usize = 0;
            while (x < nx) : (x += 1) {
                const fh = indexToFreq(x, nx);
                const fk = indexToFreq(y, ny);
                const fl = indexToFreq(z, nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                const out_idx = th + nx * (tk + ny * tl);
                out[out_idx] = math.complex.scale(data[idx], scale);
                idx += 1;
            }
        }
    }

    return out;
}

fn makeScfConfig(alloc: std.mem.Allocator, out_dir: []const u8, cell: math.Mat3) !config.Config {
    const title = try alloc.dupe(u8, "scf_force_fd");
    errdefer alloc.free(title);
    const xyz_path = try alloc.dupe(u8, "unused.xyz");
    errdefer alloc.free(xyz_path);
    const out_dir_owned = try alloc.dupe(u8, out_dir);
    errdefer alloc.free(out_dir_owned);

    const empty_band_path = try alloc.alloc(config.BandPathPoint, 0);
    errdefer alloc.free(empty_band_path);
    const empty_pseudos = try alloc.alloc(pseudo.Spec, 0);
    errdefer alloc.free(empty_pseudos);

    const scf_cfg = config.ScfConfig{
        .enabled = true,
        .solver = .dense,
        .xc = .lda_pz,
        .smearing = .none,
        .smear_ry = 0.0,
        .ecut_ry = 10.0,
        .kmesh = .{ 1, 1, 1 },
        .kmesh_shift = .{ 0.0, 0.0, 0.0 },
        .grid = .{ 0, 0, 0 },
        .grid_scale = 1.0,
        .mixing_beta = 0.3,
        .max_iter = 80,
        .convergence = 1e-8,
        .convergence_metric = .density,
        .profile = false,
        .quiet = true,
        .debug_nonlocal = false,
        .debug_local = false,
        .debug_fermi = false,
        .enable_nonlocal = true,
        .local_potential = .short_range,
        .symmetry = false,
        .time_reversal = false,
        .kpoint_threads = 1,
        .iterative_max_iter = 20,
        .iterative_tol = 1e-4,
        .iterative_max_subspace = 0,
        .iterative_block_size = 0,
        .iterative_init_diagonal = false,
        .iterative_warmup_steps = 1,
        .iterative_warmup_max_iter = 8,
        .iterative_warmup_tol = 1e-3,
        .iterative_reuse_vectors = true,
        .kerker_q0 = 0.0,
        .diemac = 1.0,
        .dielng = 1.0,
        .pulay_history = 6,
        .pulay_start = 0,
        .mixing_mode = .potential,
        .use_rfft = false,
        .fft_backend = .zig,
        .compute_stress = false,
        .nspin = 1,
        .spinat = null,
        .reference_json = null,
        .compare_reference_json = null,
        .comparison_json = null,
        .compare_tolerance_json = null,
    };

    return config.Config{
        .title = title,
        .xyz_path = xyz_path,
        .out_dir = out_dir_owned,
        .units = .bohr,
        .linalg_backend = if (builtin.os.tag == .macos) .accelerate else .openblas,
        .threads = 1,
        .cell = cell,
        .scf = scf_cfg,
        .ewald = .{ .alpha = 0.0, .rcut = 0.0, .gcut = 0.0, .tol = 1e-8 },
        .relax = .{ .enabled = true, .algorithm = .bfgs, .max_iter = 0, .force_tol = 1e-6, .max_step = 0.1, .output_trajectory = false },
        .dos = .{},
        .output = .{},
        .dfpt = .{ .enabled = false, .sternheimer_tol = 1e-8, .sternheimer_max_iter = 200, .scf_tol = 1e-10, .scf_max_iter = 50, .mixing_beta = 0.3, .alpha_shift = 0.01, .qpath_npoints = 0, .pulay_history = 8, .pulay_start = 4, .kpoint_threads = 0, .perturbation_threads = 1, .qgrid = null, .qpath = &.{} },
        .band = .{
            .points_per_segment = 10,
            .nbands = 4,
            .path = empty_band_path,
            .solver = .dense,
            .iterative_max_iter = 20,
            .iterative_tol = 1e-4,
            .iterative_max_subspace = 0,
            .iterative_block_size = 0,
            .iterative_init_diagonal = false,
            .kpoint_threads = 1,
            .iterative_reuse_vectors = true,
            .use_symmetry = false,
            .lobpcg_parallel = false,
        },
        .vdw = .{},
        .pseudopotentials = empty_pseudos,
    };
}

test "scf total force finite difference" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var element_buf: [2]u8 = .{ 'S', 'i' };
    var path_buf: [24]u8 = undefined;
    const path_slice = "pseudo/Si.upf";
    try requireFile(path_slice);
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..2],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(alloc, spec);
    defer parsed.deinit(alloc);

    var parsed_items = [_]pseudo.Parsed{parsed};
    const species = try hamiltonian.buildSpeciesEntries(alloc, parsed_items[0..]);
    defer {
        for (species) |*entry| {
            entry.deinit();
        }
        alloc.free(species);
    }
    if (species.len > 0) {
        scfPrint("UPF qij size={d} nlcc size={d}\n", .{ species[0].upf.qij.len, species[0].upf.nlcc.len });
    }

    const a = 8.0;
    const cell = math.Mat3{ .m = .{
        .{ a, 0.0, 0.0 },
        .{ 0.0, a, 0.0 },
        .{ 0.0, 0.0, a },
    } };
    const recip = math.Mat3{ .m = .{
        .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
        .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
        .{ 0.0, 0.0, 2.0 * std.math.pi / a },
    } };
    const volume = a * a * a;

    const base_atoms = [_]hamiltonian.AtomData{
        .{ .position = math.Vec3{ .x = 0.3, .y = 0.4, .z = 0.2 }, .species_index = 0 },
        .{ .position = math.Vec3{ .x = 0.6, .y = 0.2, .z = 0.5 }, .species_index = 0 },
    };

    var cfg = try makeScfConfig(alloc, "zig-cache/tmp/scf_force_fd", cell);
    defer cfg.deinit(alloc);

    var atoms_base = base_atoms;
    var scf_result = try scf.run(.{ .alloc = alloc, .cfg = cfg, .species = species, .atoms = atoms_base[0..], .recip = recip, .volume_bohr = volume });
    defer scf_result.deinit(alloc);
    try testing.expect(scf_result.converged);

    if (scf_result.vresid) |vresid| {
        var sum_sq: f64 = 0.0;
        var max_abs: f64 = 0.0;
        for (vresid.values) |v| {
            const mag2 = v.r * v.r + v.i * v.i;
            sum_sq += mag2;
            const mag = std.math.sqrt(mag2);
            if (mag > max_abs) max_abs = mag;
        }
        const rms = std.math.sqrt(sum_sq / @as(f64, @floatFromInt(vresid.values.len)));
        scfPrint("SCF vresid rms={e:.6} max={e:.6}\n", .{ rms, max_abs });
    }

    const grid = forces.Grid{
        .nx = scf_result.potential.nx,
        .ny = scf_result.potential.ny,
        .nz = scf_result.potential.nz,
        .min_h = scf_result.potential.min_h,
        .min_k = scf_result.potential.min_k,
        .min_l = scf_result.potential.min_l,
        .cell = cell,
        .recip = recip,
    };

    const rho_g = try densityToReciprocal(alloc, grid, scf_result.density, cfg.scf.fft_backend);
    defer alloc.free(rho_g);

    var force_terms = try forces.computeForces(
        alloc,
        grid,
        rho_g,
        scf_result.potential.values,
        null,
        species,
        atoms_base[0..],
        cell,
        recip,
        volume,
        cfg.ewald.alpha,
        scf_result.wavefunctions,
        if (scf_result.vresid) |vresid| vresid.values else null,
        true,
        null,
        null,
        null,
        null,
        null,
        null,
        .{},
        null,
        null,
        null,
        null,
    );
    defer force_terms.deinit(alloc);

    const local_force_base = force_terms.local[0];

    const ewald_fd = blk: {
        const delta_ewald = 1e-4;
        var charges = try alloc.alloc(f64, atoms_base.len);
        defer alloc.free(charges);
        var positions = try alloc.alloc(math.Vec3, atoms_base.len);
        defer alloc.free(positions);
        for (atoms_base, 0..) |atom, i| {
            charges[i] = species[atom.species_index].z_valence;
            positions[i] = atom.position;
        }
        var positions_plus = try alloc.alloc(math.Vec3, positions.len);
        defer alloc.free(positions_plus);
        var positions_minus = try alloc.alloc(math.Vec3, positions.len);
        defer alloc.free(positions_minus);
        @memcpy(positions_plus, positions);
        @memcpy(positions_minus, positions);
        positions_plus[0].x += delta_ewald;
        positions_minus[0].x -= delta_ewald;
        const params = ewald.Params{ .alpha = cfg.ewald.alpha, .rcut = 0.0, .gcut = 0.0, .tol = 1e-8, .quiet = true };
        const e_plus = try ewald.ionIonEnergy(cell, recip, charges, positions_plus, params);
        const e_minus = try ewald.ionIonEnergy(cell, recip, charges, positions_minus, params);
        break :blk -(e_plus - e_minus) / (2.0 * delta_ewald);
    };
    const ewald_fd_ry = ewald_fd * 2.0;

    const localEnergyFixed = struct {
        fn eval(
            grid_local: forces.Grid,
            rho_g_local: []const math.Complex,
            species_entries: []hamiltonian.SpeciesEntry,
            atoms_local: []hamiltonian.AtomData,
            volume_local: f64,
        ) !f64 {
            const inv_volume = 1.0 / volume_local;
            const b1 = grid_local.recip.row(0);
            const b2 = grid_local.recip.row(1);
            const b3 = grid_local.recip.row(2);
            var sum: f64 = 0.0;
            var idx: usize = 0;
            var l: usize = 0;
            while (l < grid_local.nz) : (l += 1) {
                var k: usize = 0;
                while (k < grid_local.ny) : (k += 1) {
                    var h: usize = 0;
                    while (h < grid_local.nx) : (h += 1) {
                        const gh = grid_local.min_h + @as(i32, @intCast(h));
                        const gk = grid_local.min_k + @as(i32, @intCast(k));
                        const gl = grid_local.min_l + @as(i32, @intCast(l));
                        const gvec = math.Vec3.add(
                            math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                            math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                        );
                        const vloc = try hamiltonian.ionicLocalPotential(gvec, species_entries, atoms_local, inv_volume);
                        const rho = rho_g_local[idx];
                        sum += rho.r * vloc.r + rho.i * vloc.i;
                        idx += 1;
                    }
                }
            }
            return sum * volume_local;
        }
    };

    const local_fd_fixed = blk: {
        const delta_local = 1e-4;
        var atoms_plus = base_atoms;
        var atoms_minus = base_atoms;
        atoms_plus[0].position.x += delta_local;
        atoms_minus[0].position.x -= delta_local;
        const e_plus = try localEnergyFixed.eval(grid, rho_g, species, atoms_plus[0..], volume);
        const e_minus = try localEnergyFixed.eval(grid, rho_g, species, atoms_minus[0..], volume);
        break :blk -(e_plus - e_minus) / (2.0 * delta_local);
    };

    scfPrint("Local FD (fixed rho): {d:.6} | Force Local: {d:.6}\n", .{ local_fd_fixed, local_force_base.x });

    const nonlocalEnergy = struct {
        fn bandEnergy(n: usize, vnl: []math.Complex, vectors: []math.Complex, band: usize) f64 {
            var sum = math.complex.init(0.0, 0.0);
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const ci = vectors[i + band * n];
                var tmp = math.complex.init(0.0, 0.0);
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    const cj = vectors[j + band * n];
                    const hij = vnl[i + j * n];
                    tmp = math.complex.add(tmp, math.complex.mul(hij, cj));
                }
                sum = math.complex.add(sum, math.complex.mul(math.complex.conj(ci), tmp));
            }
            return sum.r;
        }

        fn eval(
            alloc_local: std.mem.Allocator,
            wf: scf.WavefunctionData,
            species_entries: []hamiltonian.SpeciesEntry,
            atoms_local: []hamiltonian.AtomData,
            recip_local: math.Mat3,
            volume_local: f64,
        ) !f64 {
            const inv_volume = 1.0 / volume_local;
            const spin_factor = 2.0;
            var total: f64 = 0.0;

            for (wf.kpoints) |kp_wf| {
                var basis = try plane_wave.generate(alloc_local, recip_local, wf.ecut_ry, kp_wf.k_cart);
                defer basis.deinit(alloc_local);
                if (basis.gvecs.len != kp_wf.basis_len) continue;
                const vnl = try hamiltonian.buildNonlocalMatrix(alloc_local, basis.gvecs, species_entries, atoms_local, inv_volume);
                defer alloc_local.free(vnl);

                var band: usize = 0;
                while (band < kp_wf.nbands) : (band += 1) {
                    const occ = kp_wf.occupations[band];
                    if (occ <= 0.0) continue;
                    const e_nl = bandEnergy(basis.gvecs.len, vnl, kp_wf.coefficients, band);
                    total += kp_wf.weight * occ * spin_factor * e_nl;
                }
            }

            return total;
        }
    };

    var nl_fx_fixed: ?f64 = null;
    if (scf_result.wavefunctions) |wf| {
        const nl_force = force_terms.nonlocal orelse return error.MissingNonlocalForces;
        const delta_nl = 1e-5;
        const nl_fx_num = blk: {
            var atoms_plus = atoms_base;
            var atoms_minus = atoms_base;
            atoms_plus[0].position.x += delta_nl;
            atoms_minus[0].position.x -= delta_nl;
            const e_plus = try nonlocalEnergy.eval(alloc, wf, species, atoms_plus[0..], recip, volume);
            const e_minus = try nonlocalEnergy.eval(alloc, wf, species, atoms_minus[0..], recip, volume);
            break :blk -(e_plus - e_minus) / (2.0 * delta_nl);
        };
        scfPrint("Nonlocal FD (fixed wf): {d:.6} | Force Nonlocal: {d:.6}\n", .{ nl_fx_num, nl_force[0].x });
        try testing.expectApproxEqAbs(nl_force[0].x, nl_fx_num, 5e-3);
        nl_fx_fixed = nl_fx_num;
    }

    if (nl_fx_fixed) |nl_fixed| {
        const fixed_total = ewald_fd_ry + local_fd_fixed + nl_fixed;
        const fixed_force = force_terms.ewald[0].x + local_force_base.x + (force_terms.nonlocal orelse return error.MissingNonlocalForces)[0].x;
        scfPrint("Fixed FD total: {d:.6} | Fixed Force total: {d:.6}\n", .{ fixed_total, fixed_force });
        try testing.expectApproxEqAbs(fixed_force, fixed_total, 5e-3);
    }
}

test "scf total force finite difference (self-consistent)" {
    if (!scf_fd_enabled) return;

    const testing = std.testing;
    const alloc = testing.allocator;

    var element_buf: [2]u8 = .{ 'S', 'i' };
    var path_buf: [24]u8 = undefined;
    const path_slice = "pseudo/Si.upf";
    try requireFile(path_slice);
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..2],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(alloc, spec);
    defer parsed.deinit(alloc);

    var parsed_items = [_]pseudo.Parsed{parsed};
    const species = try hamiltonian.buildSpeciesEntries(alloc, parsed_items[0..]);
    defer {
        for (species) |*entry| {
            entry.deinit();
        }
        alloc.free(species);
    }

    const a = 8.0;
    const cell = math.Mat3{ .m = .{
        .{ a, 0.0, 0.0 },
        .{ 0.0, a, 0.0 },
        .{ 0.0, 0.0, a },
    } };
    const recip = math.Mat3{ .m = .{
        .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
        .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
        .{ 0.0, 0.0, 2.0 * std.math.pi / a },
    } };
    const volume = a * a * a;

    const base_atoms = [_]hamiltonian.AtomData{
        .{ .position = math.Vec3{ .x = 0.3, .y = 0.4, .z = 0.2 }, .species_index = 0 },
        .{ .position = math.Vec3{ .x = 0.6, .y = 0.2, .z = 0.5 }, .species_index = 0 },
    };

    var cfg = try makeScfConfig(alloc, "zig-cache/tmp/scf_force_fd", cell);
    defer cfg.deinit(alloc);

    const scf_eval = struct {
        fn run(
            alloc_local: std.mem.Allocator,
            cfg_local: config.Config,
            species_entries: []hamiltonian.SpeciesEntry,
            atoms_local: []hamiltonian.AtomData,
            recip_local: math.Mat3,
            volume_local: f64,
        ) !scf.EnergyTerms {
            var scf_result = try scf.run(.{ .alloc = alloc_local, .cfg = cfg_local, .species = species_entries, .atoms = atoms_local, .recip = recip_local, .volume_bohr = volume_local });
            defer scf_result.deinit(alloc_local);
            try std.testing.expect(scf_result.converged);
            return scf_result.energy;
        }
    };

    var atoms_base = base_atoms;
    var scf_result = try scf.run(.{ .alloc = alloc, .cfg = cfg, .species = species, .atoms = atoms_base[0..], .recip = recip, .volume_bohr = volume });
    defer scf_result.deinit(alloc);
    try testing.expect(scf_result.converged);

    const grid = forces.Grid{
        .nx = scf_result.potential.nx,
        .ny = scf_result.potential.ny,
        .nz = scf_result.potential.nz,
        .min_h = scf_result.potential.min_h,
        .min_k = scf_result.potential.min_k,
        .min_l = scf_result.potential.min_l,
        .cell = cell,
        .recip = recip,
    };

    const rho_g = try densityToReciprocal(alloc, grid, scf_result.density, cfg.scf.fft_backend);
    defer alloc.free(rho_g);

    var force_terms = try forces.computeForces(
        alloc,
        grid,
        rho_g,
        scf_result.potential.values,
        null,
        species,
        atoms_base[0..],
        cell,
        recip,
        volume,
        cfg.ewald.alpha,
        scf_result.wavefunctions,
        if (scf_result.vresid) |vresid| vresid.values else null,
        true,
        null,
        null,
        null,
        null,
        null,
        null,
        .{},
        null,
        null,
        null,
        null,
    );
    defer force_terms.deinit(alloc);

    const base_force = force_terms.total[0];

    if (scf_result.wavefunctions) |wf| {
        var e_kin: f64 = 0.0;
        var norm_min: f64 = std.math.inf(f64);
        var norm_max: f64 = 0.0;
        const spin_factor = 2.0;
        for (wf.kpoints) |kp| {
            var basis = try plane_wave.generate(alloc, recip, wf.ecut_ry, kp.k_cart);
            defer basis.deinit(alloc);
            if (basis.gvecs.len != kp.basis_len) continue;
            var band: usize = 0;
            while (band < kp.nbands) : (band += 1) {
                const occ = kp.occupations[band];
                if (occ <= 0.0) continue;
                const coeffs = kp.coefficients[band * kp.basis_len .. (band + 1) * kp.basis_len];
                var sum: f64 = 0.0;
                var norm: f64 = 0.0;
                for (coeffs, 0..) |c, i| {
                    const mag2 = c.r * c.r + c.i * c.i;
                    norm += mag2;
                    sum += mag2 * basis.gvecs[i].kinetic;
                }
                norm_min = @min(norm_min, norm);
                norm_max = @max(norm_max, norm);
                e_kin += kp.weight * occ * spin_factor * sum;
            }
        }
        scfPrint("SCF coeff norm min={d:.6} max={d:.6}\n", .{ norm_min, norm_max });
        const e_loc = scf_result.energy.local_pseudo;
        const e_nl = scf_result.energy.nonlocal_pseudo;
        const e_h = scf_result.energy.hartree;
        const e_xc = scf_result.energy.xc;
        const e_vxc_rho = scf_result.energy.vxc_rho;
        const e_ion = scf_result.energy.ion_ion;
        const total_parts = e_kin + e_loc + e_nl + e_h + e_xc + e_ion;
        const band_expected = e_kin + e_loc + e_nl + 2.0 * e_h + e_vxc_rho;
        scfPrint(
            "SCF energy parts: Ekin={d:.6} Eloc={d:.6} Enl={d:.6} Eh={d:.6} Exc={d:.6} Vxc·rho={d:.6} Eion={d:.6}\n",
            .{ e_kin, e_loc, e_nl, e_h, e_xc, e_vxc_rho, e_ion },
        );
        scfPrint(
            "SCF energy totals: SumParts={d:.6} Band={d:.6} BandExpected={d:.6} Total={d:.6}\n",
            .{ total_parts, scf_result.energy.band, band_expected, scf_result.energy.total },
        );
    }

    const delta = 5e-4;
    const fx_num = blk: {
        var atoms_plus = base_atoms;
        var atoms_minus = base_atoms;
        atoms_plus[0].position.x += delta;
        atoms_minus[0].position.x -= delta;
        const e_plus = try scf_eval.run(alloc, cfg, species, atoms_plus[0..], recip, volume);
        const e_minus = try scf_eval.run(alloc, cfg, species, atoms_minus[0..], recip, volume);
        break :blk -(e_plus.total - e_minus.total) / (2.0 * delta);
    };

    const e_terms_plus = blk: {
        var atoms_plus = base_atoms;
        atoms_plus[0].position.x += delta;
        break :blk try scf_eval.run(alloc, cfg, species, atoms_plus[0..], recip, volume);
    };
    const e_terms_minus = blk: {
        var atoms_minus = base_atoms;
        atoms_minus[0].position.x -= delta;
        break :blk try scf_eval.run(alloc, cfg, species, atoms_minus[0..], recip, volume);
    };

    const band_fd = -(e_terms_plus.band - e_terms_minus.band) / (2.0 * delta);
    const hartree_fd = -(e_terms_plus.hartree - e_terms_minus.hartree) / (2.0 * delta);
    const xc_fd = -(e_terms_plus.xc - e_terms_minus.xc) / (2.0 * delta);
    const vxc_rho_fd = -(e_terms_plus.vxc_rho - e_terms_minus.vxc_rho) / (2.0 * delta);
    const double_count_fd = -(e_terms_plus.double_counting - e_terms_minus.double_counting) / (2.0 * delta);

    const ewald_fd_scf = -(e_terms_plus.ion_ion - e_terms_minus.ion_ion) / (2.0 * delta);
    const local_fd_scf = -(e_terms_plus.local_pseudo - e_terms_minus.local_pseudo) / (2.0 * delta);
    const nonlocal_fd_scf = -(e_terms_plus.nonlocal_pseudo - e_terms_minus.nonlocal_pseudo) / (2.0 * delta);
    const hf_fd = ewald_fd_scf + local_fd_scf + nonlocal_fd_scf;
    scfPrint(
        "SCF HF FD: Ewald={d:.6} Local={d:.6} Nonlocal={d:.6} Sum={d:.6}\n",
        .{ ewald_fd_scf, local_fd_scf, nonlocal_fd_scf, hf_fd },
    );
    scfPrint(
        "SCF Total FD terms: Band={d:.6} Hartree={d:.6} XC={d:.6} Vxc·rho={d:.6} DoubleCount={d:.6}\n",
        .{ band_fd, hartree_fd, xc_fd, vxc_rho_fd, double_count_fd },
    );

    const hf_force = force_terms.ewald[0].x + force_terms.local[0].x + (force_terms.nonlocal orelse return error.MissingNonlocalForces)[0].x;
    const resid_force = if (force_terms.residual) |resid| resid[0].x else 0.0;
    scfPrint("SCF Force: HF={d:.6} Resid={d:.6} Total={d:.6}\n", .{ hf_force, resid_force, base_force.x });

    scfPrint("SCF total force: {d:.6} | SCF FD: {d:.6}\n", .{ base_force.x, fx_num });
    try testing.expectApproxEqAbs(base_force.x, fx_num, 5e-3);
}

test "scf force FD FCC cell" {
    if (!scf_fd_enabled) return;

    const testing = std.testing;
    const alloc = testing.allocator;

    var element_buf: [2]u8 = .{ 'S', 'i' };
    var path_buf: [24]u8 = undefined;
    const path_slice = "pseudo/Si.upf";
    try requireFile(path_slice);
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..2],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(alloc, spec);
    defer parsed.deinit(alloc);

    var parsed_items = [_]pseudo.Parsed{parsed};
    const species = try hamiltonian.buildSpeciesEntries(alloc, parsed_items[0..]);
    defer {
        for (species) |*entry| {
            entry.deinit();
        }
        alloc.free(species);
    }

    // FCC cell for silicon (Bohr units)
    // a = 5.431 Å = 10.2631 Bohr
    const a_bohr = 10.2631;
    const half_a = a_bohr / 2.0;
    const cell = math.Mat3{ .m = .{
        .{ 0.0, half_a, half_a },
        .{ half_a, 0.0, half_a },
        .{ half_a, half_a, 0.0 },
    } };

    // Compute reciprocal lattice and volume
    const recip = math.reciprocal(cell);
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    const volume = @abs(math.Vec3.dot(a1, math.Vec3.cross(a2, a3)));

    // Displaced atom positions (Bohr)
    const base_atoms = [_]hamiltonian.AtomData{
        .{ .position = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 }, .species_index = 0 },
        .{ .position = math.Vec3{ .x = 2.66, .y = 2.47, .z = 2.62 }, .species_index = 0 },
    };

    var cfg = try makeScfConfig(alloc, "zig-cache/tmp/scf_force_fcc", cell);
    defer cfg.deinit(alloc);
    // Override for FCC test
    cfg.scf.kmesh = .{ 2, 2, 2 };
    cfg.scf.convergence = 1e-8;
    cfg.scf.max_iter = 200;
    cfg.scf.solver = .iterative;

    var atoms_base = base_atoms;
    var scf_result = try scf.run(.{ .alloc = alloc, .cfg = cfg, .species = species, .atoms = atoms_base[0..], .recip = recip, .volume_bohr = volume });
    defer scf_result.deinit(alloc);
    try testing.expect(scf_result.converged);

    const grid = forces.Grid{
        .nx = scf_result.potential.nx,
        .ny = scf_result.potential.ny,
        .nz = scf_result.potential.nz,
        .min_h = scf_result.potential.min_h,
        .min_k = scf_result.potential.min_k,
        .min_l = scf_result.potential.min_l,
        .cell = cell,
        .recip = recip,
    };

    const rho_g = try densityToReciprocal(alloc, grid, scf_result.density, cfg.scf.fft_backend);
    defer alloc.free(rho_g);

    var force_terms = try forces.computeForces(
        alloc,
        grid,
        rho_g,
        scf_result.potential.values,
        null,
        species,
        atoms_base[0..],
        cell,
        recip,
        volume,
        cfg.ewald.alpha,
        scf_result.wavefunctions,
        if (scf_result.vresid) |vresid| vresid.values else null,
        true,
        null,
        null,
        null,
        null,
        null,
        null,
        .{},
        null,
        null,
        null,
        null,
    );
    defer force_terms.deinit(alloc);

    const base_force = force_terms.total[0];
    scfPrint("FCC analytical force atom 0: ({d:.6},{d:.6},{d:.6})\n", .{ base_force.x, base_force.y, base_force.z });
    scfPrint("FCC ewald: ({d:.6},{d:.6},{d:.6})\n", .{ force_terms.ewald[0].x, force_terms.ewald[0].y, force_terms.ewald[0].z });
    scfPrint("FCC local: ({d:.6},{d:.6},{d:.6})\n", .{ force_terms.local[0].x, force_terms.local[0].y, force_terms.local[0].z });
    if (force_terms.nonlocal) |nl| {
        scfPrint("FCC nonlocal: ({d:.6},{d:.6},{d:.6})\n", .{ nl[0].x, nl[0].y, nl[0].z });
    }
    if (force_terms.nlcc) |nlcc| {
        scfPrint("FCC nlcc: ({d:.6},{d:.6},{d:.6})\n", .{ nlcc[0].x, nlcc[0].y, nlcc[0].z });
    }

    const scf_eval = struct {
        fn run_scf(
            alloc_local: std.mem.Allocator,
            cfg_local: config.Config,
            species_entries: []hamiltonian.SpeciesEntry,
            atoms_local: []hamiltonian.AtomData,
            recip_local: math.Mat3,
            volume_local: f64,
        ) !scf.EnergyTerms {
            var scf_result_local = try scf.run(.{ .alloc = alloc_local, .cfg = cfg_local, .species = species_entries, .atoms = atoms_local, .recip = recip_local, .volume_bohr = volume_local });
            defer scf_result_local.deinit(alloc_local);
            try std.testing.expect(scf_result_local.converged);
            return scf_result_local.energy;
        }
    };

    const delta = 5e-4;
    // FD in x direction for atom 0
    const fx_num = blk: {
        var atoms_plus = base_atoms;
        var atoms_minus = base_atoms;
        atoms_plus[0].position.x += delta;
        atoms_minus[0].position.x -= delta;
        const e_plus = try scf_eval.run_scf(alloc, cfg, species, atoms_plus[0..], recip, volume);
        const e_minus = try scf_eval.run_scf(alloc, cfg, species, atoms_minus[0..], recip, volume);
        std.debug.print("FCC E(+d)={d:.10} E(-d)={d:.10}\n", .{ e_plus.total, e_minus.total });
        std.debug.print("FCC E_local(+d)={d:.10} E_local(-d)={d:.10}\n", .{ e_plus.local_pseudo, e_minus.local_pseudo });
        std.debug.print("FCC E_nl(+d)={d:.10} E_nl(-d)={d:.10}\n", .{ e_plus.nonlocal_pseudo, e_minus.nonlocal_pseudo });
        std.debug.print("FCC E_ion(+d)={d:.10} E_ion(-d)={d:.10}\n", .{ e_plus.ion_ion, e_minus.ion_ion });
        break :blk -(e_plus.total - e_minus.total) / (2.0 * delta);
    };

    std.debug.print("FCC total force x: analytical={d:.6} FD={d:.6} diff={d:.6}\n", .{ base_force.x, fx_num, base_force.x - fx_num });
    try testing.expectApproxEqAbs(base_force.x, fx_num, 1e-2);
}

// Test kinetic energy calculation
test "kinetic energy |k+G|^2" {
    const allocator = std.testing.allocator;

    // Graphene cell (Bohr)
    const a = 4.6487262675; // 2.46 Å in Bohr
    const c = 18.8972613; // 10 Å in Bohr

    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * std.math.sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );

    const recip = math.reciprocal(cell);

    // Generate basis at Gamma (k=0)
    const ecut_ry: f64 = 10.0;
    const k_cart = math.Vec3{ .x = 0, .y = 0, .z = 0 };

    var basis = try plane_wave.generate(allocator, recip, ecut_ry, k_cart);
    defer basis.deinit(allocator);

    vprint("\n=== Kinetic Energy ===\n", .{});
    vprint("Number of plane waves: {d}\n", .{basis.gvecs.len});

    // Find G=0 component
    for (basis.gvecs) |gvec| {
        if (gvec.h == 0 and gvec.k == 0 and gvec.l == 0) {
            vprint("T(G=0) = {d:.6} Ry (should be 0)\n", .{gvec.kinetic});
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), gvec.kinetic, 1e-10);
            break;
        }
    }

    // Print some kinetic energies
    for (basis.gvecs[0..@min(5, basis.gvecs.len)]) |gvec| {
        vprint("G=({d},{d},{d}) T={d:.4} Ry\n", .{ gvec.h, gvec.k, gvec.l, gvec.kinetic });
    }
}

// Test smallest G vector magnitude
test "smallest G vector for large vacuum" {
    const allocator = std.testing.allocator;

    // Graphene cell with 20 Å vacuum (matches ABINIT)
    const a = 4.6487262675;
    const c = 37.7945225; // 20 Å in Bohr

    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * std.math.sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );

    const recip = math.reciprocal(cell);

    // Generate basis at Gamma with typical ecut
    const ecut_ry: f64 = 30.0; // 15 Ha like ABINIT
    const k_cart = math.Vec3{ .x = 0, .y = 0, .z = 0 };

    var basis = try plane_wave.generate(allocator, recip, ecut_ry, k_cart);
    defer basis.deinit(allocator);

    vprint("\n=== G vector analysis ===\n", .{});
    vprint("Number of plane waves: {d}\n", .{basis.gvecs.len});
    vprint("Reciprocal lattice spacing (z): 2π/c = {d:.4} bohr⁻¹\n", .{2.0 * std.math.pi / c});
    vprint("Reciprocal lattice spacing (xy): 2π/a ≈ {d:.4} bohr⁻¹\n", .{2.0 * std.math.pi / a});

    // Find smallest non-zero G
    var min_g: f64 = 1e10;
    for (basis.gvecs) |gvec| {
        const g_mag = std.math.sqrt(gvec.kinetic);
        if (g_mag > 1e-10 and g_mag < min_g) {
            min_g = g_mag;
        }
    }
    vprint("Smallest non-zero |G| = {d:.4} bohr⁻¹\n", .{min_g});

    // The smallest G along z is 2π/37.8 ≈ 0.166 bohr⁻¹
    const expected_min_gz = 2.0 * std.math.pi / c;
    try std.testing.expectApproxEqRel(expected_min_gz, min_g, 0.01);
}

test "ewald force finite difference" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const a = 8.0;
    const cell = math.Mat3{ .m = .{
        .{ a, 0.0, 0.0 },
        .{ 0.0, a, 0.0 },
        .{ 0.0, 0.0, a },
    } };
    const recip = math.Mat3{ .m = .{
        .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
        .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
        .{ 0.0, 0.0, 2.0 * std.math.pi / a },
    } };

    const charges = [_]f64{ 4.0, 4.0 };
    const positions = [_]math.Vec3{
        .{ .x = 0.3, .y = 0.4, .z = 0.2 },
        .{ .x = 0.6, .y = 0.2, .z = 0.5 },
    };

    const delta = 1e-3;
    const alpha = 5.0 / a;
    const real_params = ewald.Params{ .alpha = alpha, .rcut = 20.0, .gcut = 1e-6, .tol = 1e-8, .quiet = true };
    const quiet_params = ewald.Params{ .alpha = 0.0, .rcut = 0.0, .gcut = 0.0, .tol = 0.0, .quiet = true };
    const real_forces = try ewald.ionIonForces(alloc, cell, recip, charges[0..], positions[0..], real_params);
    defer alloc.free(real_forces);
    const fx_real_num = blk: {
        var positions_plus = positions;
        var positions_minus = positions;
        positions_plus[0].x += delta;
        positions_minus[0].x -= delta;
        const e_plus = try ewald.ionIonEnergy(cell, recip, charges[0..], positions_plus[0..], real_params);
        const e_minus = try ewald.ionIonEnergy(cell, recip, charges[0..], positions_minus[0..], real_params);
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };
    try testing.expectApproxEqAbs(real_forces[0].x, fx_real_num, 1e-3);

    const ewald_forces = try ewald.ionIonForces(alloc, cell, recip, charges[0..], positions[0..], quiet_params);
    defer alloc.free(ewald_forces);

    const fx_num = blk: {
        var positions_plus = positions;
        var positions_minus = positions;
        positions_plus[0].x += delta;
        positions_minus[0].x -= delta;
        const e_plus = try ewald.ionIonEnergy(cell, recip, charges[0..], positions_plus[0..], quiet_params);
        const e_minus = try ewald.ionIonEnergy(cell, recip, charges[0..], positions_minus[0..], quiet_params);
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };

    try testing.expectApproxEqAbs(ewald_forces[0].x, fx_num, 1e-3);
}

// Test nonlocal pseudopotential projectors
test "nonlocal projector radial integral" {
    const allocator = std.testing.allocator;

    // Load C.upf
    var element_buf: [2]u8 = .{ 'C', 0 };
    var path_buf: [20]u8 = undefined;
    const path_slice = "pseudo/C.upf";
    try requireFile(path_slice);
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..1],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(allocator, spec);
    defer parsed.deinit(allocator);

    const upf = parsed.upf orelse return error.NoUpfData;

    vprint("\n=== Nonlocal Projector Analysis ===\n", .{});
    vprint("Number of beta projectors: {d}\n", .{upf.beta.len});
    vprint("D_ij size: {d}\n", .{upf.dij.len});

    // Print D_ij matrix
    vprint("\nD_ij matrix (Rydberg):\n", .{});
    const nb = upf.beta.len;
    for (0..nb) |i| {
        for (0..nb) |j| {
            const dij = upf.dij[i * nb + j];
            vprint("  {d:8.4}", .{dij});
        }
        vprint("\n", .{});
    }

    // Test radial projector at G=0 (|k+G|=0)
    vprint("\nRadial projectors at |G|=0:\n", .{});
    for (upf.beta, 0..) |beta, idx| {
        const l_val = beta.l orelse 0;
        const rad = nonlocal.radialProjector(beta.values, upf.r, upf.rab, l_val, 0.0);
        vprint("  beta_{d} (l={d}): {d:.6}\n", .{ idx, l_val, rad });
    }

    // Test radial projector at typical G values
    const g_vals = [_]f64{ 0.5, 1.0, 2.0, 5.0 };
    vprint("\nRadial projectors at various |G|:\n", .{});
    for (g_vals) |g| {
        vprint("|G|={d:.1}:", .{g});
        for (upf.beta, 0..) |beta, idx| {
            const l_val = beta.l orelse 0;
            const rad = nonlocal.radialProjector(beta.values, upf.r, upf.rab, l_val, g);
            _ = idx;
            vprint("  {d:8.4}", .{rad});
        }
        vprint("\n", .{});
    }

    // Calculate V_nl(G=0, G=0) for s-channel (should be positive from D_11, D_22)
    // Angular factor at cos_gamma=1: 4π(2l+1)P_l(1) = 4π(2l+1)
    const angular_s = 4.0 * std.math.pi * 1.0; // l=0
    vprint("\nAngular factor for l=0: {d:.4}\n", .{angular_s});

    // s-channel contribution at G=0
    const r0_s1 = nonlocal.radialProjector(upf.beta[0].values, upf.r, upf.rab, 0, 0.0);
    const r0_s2 = nonlocal.radialProjector(upf.beta[1].values, upf.r, upf.rab, 0, 0.0);
    const d11 = upf.dij[0];
    const d22 = upf.dij[1 * nb + 1];
    const vnl_s_00 = angular_s * (d11 * r0_s1 * r0_s1 + d22 * r0_s2 * r0_s2);
    vprint("V_nl(G=0,G=0) s-channel: {d:.4} Ry\n", .{vnl_s_00});
}

// Test k-point fractional to Cartesian conversion
test "k-point fractional to Cartesian conversion" {
    // Graphene hexagonal cell (Bohr)
    const a = 4.6487262675;
    const c = 37.7945225;

    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * std.math.sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );

    const recip = math.reciprocal(cell);
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);

    vprint("\n=== K-point Conversion Test ===\n", .{});
    vprint("Reciprocal lattice vectors:\n", .{});
    vprint("  b1 = ({d:.6}, {d:.6}, {d:.6})\n", .{ b1.x, b1.y, b1.z });
    vprint("  b2 = ({d:.6}, {d:.6}, {d:.6})\n", .{ b2.x, b2.y, b2.z });
    vprint("  b3 = ({d:.6}, {d:.6}, {d:.6})\n", .{ b3.x, b3.y, b3.z });

    // K point in fractional coordinates: (1/3, 1/3, 0)
    const k_frac = math.Vec3{ .x = 1.0 / 3.0, .y = 1.0 / 3.0, .z = 0.0 };

    // CORRECT method using math.fracToCart
    const k_cart_correct = math.fracToCart(k_frac, recip);

    // Compare with explicit calculation
    const k_cart_explicit = math.Vec3.add(
        math.Vec3.add(
            math.Vec3.scale(b1, k_frac.x),
            math.Vec3.scale(b2, k_frac.y),
        ),
        math.Vec3.scale(b3, k_frac.z),
    );

    vprint("\nK point (1/3, 1/3, 0):\n", .{});
    vprint("  fracToCart k_cart = ({d:.6}, {d:.6}, {d:.6})\n", .{ k_cart_correct.x, k_cart_correct.y, k_cart_correct.z });
    vprint("  explicit k_cart   = ({d:.6}, {d:.6}, {d:.6})\n", .{ k_cart_explicit.x, k_cart_explicit.y, k_cart_explicit.z });
    vprint("  |k_cart| = {d:.6}\n", .{math.Vec3.norm(k_cart_correct)});

    // Check if fracToCart matches explicit calculation
    const diff = math.Vec3.sub(k_cart_correct, k_cart_explicit);
    const diff_norm = math.Vec3.norm(diff);
    vprint("  Difference = {d:.6}\n", .{diff_norm});

    // fracToCart should match explicit calculation exactly
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), diff_norm, 1e-10);
}

// Test Hamiltonian Hermiticity at M-Γ midpoint
test "Hamiltonian Hermiticity" {
    const allocator = std.testing.allocator;

    // Graphene cell (Bohr)
    const a = 4.6487262675;
    const c = 37.7945225;

    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * std.math.sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );

    const recip = math.reciprocal(cell);

    // M-Γ midpoint: k_frac = (0.25, 0, 0)
    const k_frac = math.Vec3{ .x = 0.25, .y = 0.0, .z = 0.0 };
    const k_cart = math.fracToCart(k_frac, recip);

    vprint("\n=== Hamiltonian Hermiticity Test ===\n", .{});
    vprint("k_frac = ({d:.4}, {d:.4}, {d:.4})\n", .{ k_frac.x, k_frac.y, k_frac.z });
    vprint("k_cart = ({d:.6}, {d:.6}, {d:.6})\n", .{ k_cart.x, k_cart.y, k_cart.z });

    // Generate basis with small cutoff for quick test
    const ecut_ry: f64 = 10.0;
    var basis = try plane_wave.generate(allocator, recip, ecut_ry, k_cart);
    defer basis.deinit(allocator);

    vprint("Number of plane waves: {d}\n", .{basis.gvecs.len});

    // Build Hamiltonian without pseudopotential (just kinetic + local)
    const n = basis.gvecs.len;
    const h = try allocator.alloc(math.Complex, n * n);
    defer allocator.free(h);

    // Fill with kinetic energy (diagonal)
    for (0..n) |j| {
        for (0..n) |i| {
            if (i == j) {
                h[i + j * n] = math.complex.init(basis.gvecs[i].kinetic, 0.0);
            } else {
                h[i + j * n] = math.complex.init(0.0, 0.0);
            }
        }
    }

    // Check Hermiticity: H[i,j] should equal conj(H[j,i])
    var max_asym: f64 = 0.0;
    for (0..n) |i| {
        for (i + 1..n) |j| {
            const hij = h[i + j * n];
            const hji = h[j + i * n];
            const diff_r = hij.r - hji.r;
            const diff_i = hij.i + hji.i;
            const asym = std.math.sqrt(diff_r * diff_r + diff_i * diff_i);
            if (asym > max_asym) {
                max_asym = asym;
            }
        }
    }

    vprint("Max |H[i,j] - conj(H[j,i])| = {e:.2}\n", .{max_asym});

    // Kinetic-only Hamiltonian should be perfectly Hermitian (diagonal real)
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), max_asym, 1e-14);
}

// Test Ewald ion-ion energy
test "Ewald ion-ion energy for graphene" {
    // Graphene cell (Bohr)
    const a = 4.6487262675;
    const c = 37.7945225; // 20 Å in Bohr

    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * std.math.sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );

    const recip = math.reciprocal(cell);

    const charges = [_]f64{ 4.0, 4.0 }; // Carbon Z_val = 4
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 3.0991508450, .y = 2.6839433578, .z = 0.0 },
    };

    const quiet_params = ewald.Params{ .alpha = 0.0, .rcut = 0.0, .gcut = 0.0, .tol = 0.0, .quiet = true };
    const e_ion = try ewald.ionIonEnergy(cell, recip, &charges, &positions, quiet_params);

    vprint("\n=== Ewald Energy ===\n", .{});
    vprint("E_ion (Hartree) = {d:.6}\n", .{e_ion});
    vprint("E_ion (Rydberg) = {d:.6}\n", .{e_ion * 2.0});
    vprint("ABINIT Ewald    = 50.1056 Ha\n", .{});

    // Check against ABINIT value (50.1056 Ha)
    try std.testing.expectApproxEqRel(@as(f64, 50.1056), e_ion, 0.01);
}

// Test local potential application: iterative vs dense
test "apply local potential vs dense" {
    const allocator = std.testing.allocator;
    const fft_mod = @import("features/fft/fft.zig");

    // Simple 4x4x4 grid
    const nx: usize = 4;
    const ny: usize = 4;
    const nz: usize = 4;
    const total = nx * ny * nz;

    // Create a simple potential in G-space
    // V(G=0) = 0.5, V(G=(1,0,0)) = 0.1, V(G=(-1,0,0)) = 0.1
    const v_g = try allocator.alloc(math.Complex, total);
    defer allocator.free(v_g);
    @memset(v_g, math.complex.init(0.0, 0.0));
    v_g[0] = math.complex.init(0.5, 0.0); // G=0
    v_g[1] = math.complex.init(0.1, 0.0); // G=(1,0,0)
    v_g[nx - 1] = math.complex.init(0.1, 0.0); // G=(-1,0,0)

    // Build local_r = N * IFFT(V_G) (this is what buildLocalPotentialReal does)
    const local_r = try allocator.alloc(f64, total);
    defer allocator.free(local_r);
    {
        const temp = try allocator.alloc(math.Complex, total);
        defer allocator.free(temp);
        // Apply scale = N before IFFT
        for (v_g, 0..) |v, i| {
            temp[i] = math.complex.scale(v, @as(f64, @floatFromInt(total)));
        }
        try fft_mod.fft3dInverseInPlace(allocator, temp, nx, ny, nz);
        for (temp, 0..) |v, i| {
            local_r[i] = v.r;
        }
    }

    vprint("\n=== Apply Local Potential Test ===\n", .{});
    vprint("local_r[0] = {d:.6} (should be N*V(G=0) after IFFT = {d:.6})\n", .{
        local_r[0],
        0.5 + 0.1 + 0.1, // V(G=0) + V(G=1) + V(G=-1) all contribute to r=0
    });

    // Test psi = delta at G=0: psi_G = [1, 0, 0, ...]
    const psi_g = try allocator.alloc(math.Complex, total);
    defer allocator.free(psi_g);
    @memset(psi_g, math.complex.init(0.0, 0.0));
    psi_g[0] = math.complex.init(1.0, 0.0);

    // Method 1: Dense matrix-vector product (V*psi)_G = Σ_G' V_{G-G'} * psi_G'
    // For psi = delta at G=0: (V*psi)_G = V_G
    const dense_result = try allocator.alloc(math.Complex, total);
    defer allocator.free(dense_result);
    @memcpy(dense_result, v_g); // (V*psi) = V for delta psi

    // Method 2: FFT-based (simulating applyLocalPotential)
    // Step 1: IFFT of psi with scale=N (like fftReciprocalToComplexInPlace)
    const psi_r = try allocator.alloc(math.Complex, total);
    defer allocator.free(psi_r);
    for (psi_g, 0..) |p, i| {
        psi_r[i] = math.complex.scale(p, @as(f64, @floatFromInt(total)));
    }
    try fft_mod.fft3dInverseInPlace(allocator, psi_r, nx, ny, nz);

    // Step 2: Multiply in real space: local_r * psi_r
    const vpsi_r = try allocator.alloc(math.Complex, total);
    defer allocator.free(vpsi_r);
    for (vpsi_r, 0..) |*v, i| {
        v.* = math.complex.scale(psi_r[i], local_r[i]);
    }

    // Step 3: FFT with 1/N (like fftComplexToReciprocalInPlace)
    const fft_result = try allocator.alloc(math.Complex, total);
    defer allocator.free(fft_result);
    @memcpy(fft_result, vpsi_r);
    try fft_mod.fft3dForwardInPlace(allocator, fft_result, nx, ny, nz);
    for (fft_result) |*v| {
        v.* = math.complex.scale(v.*, 1.0 / @as(f64, @floatFromInt(total)));
    }

    vprint("\nComparing results for psi = delta at G=0:\n", .{});
    var max_diff: f64 = 0.0;
    for (0..@min(total, 5)) |i| {
        const diff_r = dense_result[i].r - fft_result[i].r;
        const diff_i = dense_result[i].i - fft_result[i].i;
        const diff = @sqrt(diff_r * diff_r + diff_i * diff_i);
        if (diff > max_diff) max_diff = diff;
        vprint("G[{d}]: dense=({d:.6},{d:.6}) fft=({d:.6},{d:.6}) diff={d:.6}\n", .{
            i, dense_result[i].r, dense_result[i].i, fft_result[i].r, fft_result[i].i, diff,
        });
    }
    vprint("Max difference: {d:.6}\n", .{max_diff});

    // The results should match
    try std.testing.expect(max_diff < 1e-10);
}

// Test FFT convolution vs direct matrix-vector product for local potential
test "local potential FFT vs direct" {
    const allocator = std.testing.allocator;
    const fft_mod = @import("features/fft/fft.zig");

    // Simple 4x4x4 grid for testing
    const nx: usize = 4;
    const ny: usize = 4;
    const nz: usize = 4;
    const total = nx * ny * nz;

    // Create a simple test potential in G-space (just a cosine)
    const v_g = try allocator.alloc(math.Complex, total);
    defer allocator.free(v_g);
    @memset(v_g, math.complex.init(0.0, 0.0));
    // Set V(G=0) = 1.0, V(G=(1,0,0)) = 0.5, V(G=(-1,0,0)) = 0.5
    v_g[0] = math.complex.init(1.0, 0.0); // G=0
    v_g[1] = math.complex.init(0.5, 0.0); // G=(1,0,0)
    v_g[nx - 1] = math.complex.init(0.5, 0.0); // G=(-1,0,0)

    // Create test wavefunction in G-space
    const psi_g = try allocator.alloc(math.Complex, total);
    defer allocator.free(psi_g);
    @memset(psi_g, math.complex.init(0.0, 0.0));
    psi_g[0] = math.complex.init(1.0, 0.0); // ψ(G=0) = 1

    // Method 1: Direct convolution (V*ψ)_G = Σ_G' V_{G-G'} * ψ_G'
    const direct_result = try allocator.alloc(math.Complex, total);
    defer allocator.free(direct_result);
    @memset(direct_result, math.complex.init(0.0, 0.0));

    var gz: usize = 0;
    while (gz < nz) : (gz += 1) {
        var gy: usize = 0;
        while (gy < ny) : (gy += 1) {
            var gx: usize = 0;
            while (gx < nx) : (gx += 1) {
                const g_idx = gx + nx * (gy + ny * gz);
                var sum = math.complex.init(0.0, 0.0);

                // Sum over all G'
                var gpz: usize = 0;
                while (gpz < nz) : (gpz += 1) {
                    var gpy: usize = 0;
                    while (gpy < ny) : (gpy += 1) {
                        var gpx: usize = 0;
                        while (gpx < nx) : (gpx += 1) {
                            const gp_idx = gpx + nx * (gpy + ny * gpz);
                            // G - G' with wrapping
                            const dqx = (gx + nx - gpx) % nx;
                            const dqy = (gy + ny - gpy) % ny;
                            const dqz = (gz + nz - gpz) % nz;
                            const dq_idx = dqx + nx * (dqy + ny * dqz);

                            const v_q = v_g[dq_idx];
                            const psi_gp = psi_g[gp_idx];
                            sum = math.complex.add(sum, math.complex.mul(v_q, psi_gp));
                        }
                    }
                }
                direct_result[g_idx] = sum;
            }
        }
    }

    // Method 2: FFT-based convolution
    // Step 1: IFFT of V_G to get V_r (IFFT applies 1/N normalization)
    const v_temp = try allocator.alloc(math.Complex, total);
    defer allocator.free(v_temp);
    @memcpy(v_temp, v_g);
    try fft_mod.fft3dInverseInPlace(allocator, v_temp, nx, ny, nz);

    // Step 2: IFFT of ψ_G to get ψ_r
    const psi_temp = try allocator.alloc(math.Complex, total);
    defer allocator.free(psi_temp);
    @memcpy(psi_temp, psi_g);
    try fft_mod.fft3dInverseInPlace(allocator, psi_temp, nx, ny, nz);

    // Step 3: Pointwise multiply in real space: (V*ψ)_r
    const vpsi_r = try allocator.alloc(math.Complex, total);
    defer allocator.free(vpsi_r);
    for (vpsi_r, 0..) |*v, i| {
        v.* = math.complex.mul(v_temp[i], psi_temp[i]);
    }

    // Step 4: FFT to get (V*ψ)_G (FFT has no normalization)
    try fft_mod.fft3dForwardInPlace(allocator, vpsi_r, nx, ny, nz);

    // Compare results
    vprint("\n=== Local Potential FFT vs Direct ===\n", .{});
    vprint("Grid: {d}x{d}x{d} = {d} points\n", .{ nx, ny, nz, total });

    // Note: With IFFT(1/N) and FFT(1), the convolution theorem gives:
    // FFT(IFFT(V) * IFFT(ψ)) = FFT((1/N)V_r * (1/N)ψ_r) = (1/N²) * N * (V⊛ψ) = (1/N)(V⊛ψ)
    // So we expect fft_result = direct_result / N
    const scale = @as(f64, @floatFromInt(total));

    var max_diff: f64 = 0.0;
    var idx: usize = 0;
    while (idx < total) : (idx += 1) {
        const scaled_fft = math.complex.scale(vpsi_r[idx], scale);
        const diff_r = direct_result[idx].r - scaled_fft.r;
        const diff_i = direct_result[idx].i - scaled_fft.i;
        const diff = @sqrt(diff_r * diff_r + diff_i * diff_i);
        if (diff > max_diff) max_diff = diff;

        if (idx < 5) {
            vprint("G[{d}]: direct=({d:.6},{d:.6}) fft*N=({d:.6},{d:.6}) diff={d:.6}\n", .{
                idx,
                direct_result[idx].r,
                direct_result[idx].i,
                scaled_fft.r,
                scaled_fft.i,
                diff,
            });
        }
    }

    vprint("Max difference (after scaling by N): {d:.6}\n", .{max_diff});

    // After scaling by N, FFT result should match direct result
    try std.testing.expect(max_diff < 1e-10);
}

// test "iterative eigensolver nbands comparison" - disabled due to LAPACK linking in test env
// Run the actual DFT calculation with debug output instead

// Diagnose radial projector behavior at large |g|
test "radialProjector large g behavior" {
    vprint("\n=== Radial Projector Large g Test ===\n", .{});

    // Create simple mock beta function: gaussian-like r*exp(-r²)
    var r: [100]f64 = undefined;
    var rab: [100]f64 = undefined;
    var beta: [100]f64 = undefined;

    const dr = 0.1;
    for (0..100) |i| {
        const ri = @as(f64, @floatFromInt(i)) * dr;
        r[i] = ri;
        rab[i] = dr;
        // beta stores r*β(r), so we store r * gaussian = r * exp(-r²)
        beta[i] = ri * std.math.exp(-ri * ri);
    }

    // Test radialProjector at various g values
    vprint("Testing l=0 (s-wave):\n", .{});
    for ([_]f64{ 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0 }) |g| {
        const proj = nonlocal.radialProjector(&beta, &r, &rab, 0, g);
        vprint("  g={d:.1}: proj={d:.6}\n", .{ g, proj });
    }

    vprint("Testing l=1 (p-wave):\n", .{});
    for ([_]f64{ 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0 }) |g| {
        const proj = nonlocal.radialProjector(&beta, &r, &rab, 1, g);
        vprint("  g={d:.1}: proj={d:.6}\n", .{ g, proj });
    }

    // Projector should decrease for large g (decay due to oscillating j_l)
    const proj_small = nonlocal.radialProjector(&beta, &r, &rab, 0, 1.0);
    const proj_large = nonlocal.radialProjector(&beta, &r, &rab, 0, 10.0);
    vprint("proj(g=1)/proj(g=10) = {d:.2}\n", .{@abs(proj_small / proj_large)});

    // Large g projector should be smaller than small g projector
    try std.testing.expect(@abs(proj_large) < @abs(proj_small));
}

// Test phase factor calculation for graphene
test "graphene phase factor" {
    vprint("\n=== Graphene Phase Factor Test ===\n", .{});

    // Graphene cell (Bohr)
    const a = 4.6487262675;
    const sqrt3 = std.math.sqrt(3.0);

    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * sqrt3 / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 37.7945225 },
    );
    const recip = math.reciprocal(cell);

    // Two carbon atoms at (0,0,0) and (1/3,1/3,0) in fractional
    const pos1 = math.fracToCart(.{ .x = 0.0, .y = 0.0, .z = 0.0 }, cell);
    const pos2 = math.fracToCart(.{ .x = 1.0 / 3.0, .y = 1.0 / 3.0, .z = 0.0 }, cell);

    vprint("Atom positions (Bohr):\n", .{});
    vprint("  C1: ({d:.6}, {d:.6}, {d:.6})\n", .{ pos1.x, pos1.y, pos1.z });
    vprint("  C2: ({d:.6}, {d:.6}, {d:.6})\n", .{ pos2.x, pos2.y, pos2.z });

    const bond = math.Vec3.sub(pos2, pos1);
    const bond_length = math.Vec3.norm(bond);
    vprint("  Bond: ({d:.6}, {d:.6}, {d:.6}), |bond|={d:.6}\n", .{ bond.x, bond.y, bond.z, bond_length });
    vprint("  Expected ~2.68 Bohr, got {d:.4}\n", .{bond_length});

    vprint("Reciprocal lattice (1/Bohr):\n", .{});
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    vprint("  b1: ({d:.6}, {d:.6}, {d:.6})\n", .{ b1.x, b1.y, b1.z });
    vprint("  b2: ({d:.6}, {d:.6}, {d:.6})\n", .{ b2.x, b2.y, b2.z });

    // Phase for G = b1 - b2 should be 0 with this basis
    const g_test = math.Vec3.sub(b1, b2);
    const g_phase = math.Vec3.dot(g_test, bond);
    vprint("Phase for G = b1 - b2:\n", .{});
    vprint("  G·Δτ = {d:.6} (mod 2π = {d:.6})\n", .{ g_phase, @mod(g_phase, 2.0 * std.math.pi) });
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), g_phase, 1e-10);

    // Phase for K = (1/3, 2/3, 0) should be 2π/3 for Δτ = (1/3, 1/3, 0)
    const k_frac = math.Vec3{ .x = 1.0 / 3.0, .y = 2.0 / 3.0, .z = 0.0 };
    const k_cart = math.fracToCart(k_frac, recip);
    const k_phase = math.Vec3.dot(k_cart, bond);
    vprint("Phase for K = (1/3, 2/3, 0):\n", .{});
    vprint("  K·Δτ = {d:.6} (mod 2π = {d:.6})\n", .{ k_phase, @mod(k_phase, 2.0 * std.math.pi) });
    vprint("  Expected 2π/3 = {d:.6}\n", .{2.0 * std.math.pi / 3.0});

    const exp1 = math.complex.expi(0.0);
    const exp2 = math.complex.expi(-k_phase);
    const sum = math.complex.add(exp1, exp2);
    const sum_abs = std.math.sqrt(sum.r * sum.r + sum.i * sum.i);
    vprint("  |1 + exp(-iK·Δτ)| = {d:.6}\n", .{sum_abs});
    vprint("  Expected |sum| = 1.0 for phase_diff = 2π/3\n", .{});

    try std.testing.expect(bond_length > 2.5 and bond_length < 2.9);
    try std.testing.expectApproxEqAbs(2.0 * std.math.pi / 3.0, k_phase, 1e-8);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum_abs, 1e-8);
}

// Test angular factor values
test "angular factor values" {
    vprint("\n=== Angular Factor Test ===\n", .{});

    // Angular factor = 4π(2l+1)P_l(cos γ)
    // For l=0: 4π × 1 × 1 = 12.566 (cos γ = 1)
    // For l=1: 4π × 3 × cos γ

    vprint("l=0 (s-wave):\n", .{});
    for ([_]f64{ 1.0, 0.5, 0.0, -0.5, -1.0 }) |cos_g| {
        const af = nonlocal.angularFactor(0, cos_g);
        vprint("  cos_γ={d:.2}: angular={d:.6}\n", .{ cos_g, af });
    }

    vprint("l=1 (p-wave):\n", .{});
    for ([_]f64{ 1.0, 0.5, 0.0, -0.5, -1.0 }) |cos_g| {
        const af = nonlocal.angularFactor(1, cos_g);
        vprint("  cos_γ={d:.2}: angular={d:.6}\n", .{ cos_g, af });
    }

    // l=0: 4π × 1 × P_0 = 4π ≈ 12.566
    const af0 = nonlocal.angularFactor(0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 * std.math.pi), af0, 1e-10);

    // l=1: 4π × 3 × P_1(1) = 12π ≈ 37.699
    const af1 = nonlocal.angularFactor(1, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0 * std.math.pi), af1, 1e-10);
}

// Test spherical Bessel function accuracy (Miller's algorithm)
test "spherical Bessel Miller algorithm" {
    vprint("\n=== Spherical Bessel Miller Algorithm Test ===\n", .{});

    // Test against known exact values
    // j_0(x) = sin(x)/x
    // j_1(x) = sin(x)/x² - cos(x)/x
    // j_2(x) = (3/x² - 1)sin(x)/x - 3cos(x)/x²

    const test_x = [_]f64{ 0.1, 1.0, 2.0, 5.0, 10.0, 20.0 };

    vprint("j_0(x) = sin(x)/x:\n", .{});
    for (test_x) |x| {
        const computed = nonlocal.sphericalBessel(0, x);
        const expected = std.math.sin(x) / x;
        const err = @abs(computed - expected);
        vprint("  x={d:.1}: computed={d:.10}, expected={d:.10}, err={e:.2}\n", .{ x, computed, expected, err });
        try std.testing.expectApproxEqAbs(expected, computed, 1e-12);
    }

    vprint("j_1(x) = sin(x)/x² - cos(x)/x:\n", .{});
    for (test_x) |x| {
        const computed = nonlocal.sphericalBessel(1, x);
        const expected = std.math.sin(x) / (x * x) - std.math.cos(x) / x;
        const err = @abs(computed - expected);
        vprint("  x={d:.1}: computed={d:.10}, expected={d:.10}, err={e:.2}\n", .{ x, computed, expected, err });
        try std.testing.expectApproxEqAbs(expected, computed, 1e-12);
    }

    vprint("j_2(x) = (3/x² - 1)sin(x)/x - 3cos(x)/x²:\n", .{});
    for (test_x) |x| {
        const computed = nonlocal.sphericalBessel(2, x);
        const expected = (3.0 / (x * x) - 1.0) * std.math.sin(x) / x - 3.0 * std.math.cos(x) / (x * x);
        const err = @abs(computed - expected);
        vprint("  x={d:.1}: computed={d:.10}, expected={d:.10}, err={e:.2}\n", .{ x, computed, expected, err });
        try std.testing.expectApproxEqAbs(expected, computed, 1e-10);
    }

    // Test j_3(x) at specific values
    // j_3(x) = (15/x³ - 6/x)sin(x)/x - (15/x² - 1)cos(x)/x
    vprint("j_3(x) test:\n", .{});
    for (test_x) |x| {
        const computed = nonlocal.sphericalBessel(3, x);
        const expected = (15.0 / (x * x * x) - 6.0 / x) * std.math.sin(x) / x - (15.0 / (x * x) - 1.0) * std.math.cos(x) / x;
        const err = @abs(computed - expected);
        vprint("  x={d:.1}: computed={d:.10}, expected={d:.10}, err={e:.2}\n", .{ x, computed, expected, err });
        // Allow larger tolerance for l=3 due to numerical accumulation
        try std.testing.expectApproxEqAbs(expected, computed, 1e-8);
    }

    // Test decay at large x for l > x (should be very small)
    vprint("Decay test j_l(x) for l > x:\n", .{});
    const j2_at_1 = nonlocal.sphericalBessel(2, 1.0);
    const j3_at_1 = nonlocal.sphericalBessel(3, 1.0);
    vprint("  j_2(1) = {d:.10}\n", .{j2_at_1});
    vprint("  j_3(1) = {d:.10}\n", .{j3_at_1});
    // j_l(x) ~ x^l / (2l+1)!! for small x, so j_3(1) << j_2(1)
    try std.testing.expect(@abs(j3_at_1) < @abs(j2_at_1));
}
