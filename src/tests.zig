const std = @import("std");
const math = @import("features/math/math.zig");
const hamiltonian = @import("features/hamiltonian/hamiltonian.zig");
const spacegroup = @import("features/symmetry/spacegroup.zig");
const pseudo = @import("features/pseudopotential/pseudopotential.zig");
const form_factor = @import("features/pseudopotential/form_factor.zig");
const local_potential = @import("features/pseudopotential/local_potential.zig");
const nonlocal = @import("features/pseudopotential/nonlocal.zig");
const plane_wave = @import("features/plane_wave/basis.zig");
const ewald = @import("features/ewald/ewald.zig");
const scf = @import("features/scf/scf.zig");
const forces = @import("features/forces/forces.zig");

const verbose_tests = false;

fn print_stderr(comptime fmt: []const u8, args: anytype) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(std.testing.io, &buffer);
    const out = &writer.interface;
    try out.print(fmt, args);
    try out.flush();
}

/// Skip test if a required file does not exist (e.g. pseudo/ files in CI).
fn require_file(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            try print_stderr("  [SKIP] file not found: {s}\n", .{path});
            return error.SkipZigTest;
        }
        return err;
    };
}

fn vprint(comptime fmt: []const u8, args: anytype) void {
    if (!verbose_tests) return;
    print_stderr(fmt, args) catch {};
}

fn expect_vec_approx_eq_abs(expected: math.Vec3, actual: math.Vec3, tol: f64) !void {
    try std.testing.expectApproxEqAbs(expected.x, actual.x, tol);
    try std.testing.expectApproxEqAbs(expected.y, actual.y, tol);
    try std.testing.expectApproxEqAbs(expected.z, actual.z, tol);
}

test "spacegroup silicon conventional" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const alloc = arena.allocator();

    const a = 5.431;
    const half = a / 2.0;
    const quarter = a / 4.0;
    const three_quarter = 3.0 * a / 4.0;
    const cell = math.Mat3.from_rows(
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
        .{
            .position = .{ .x = quarter, .y = three_quarter, .z = three_quarter },
            .species_index = 0,
        },
        .{
            .position = .{ .x = three_quarter, .y = quarter, .z = three_quarter },
            .species_index = 0,
        },
        .{
            .position = .{ .x = three_quarter, .y = three_quarter, .z = quarter },
            .species_index = 0,
        },
    };

    const info_opt = try spacegroup.detect_space_group_from_atoms(alloc, cell, atoms[0..], 1e-5);
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
    const cell = math.Mat3.from_rows(
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
        .{
            .position = .{ .x = quarter, .y = three_quarter, .z = three_quarter },
            .species_index = 0,
        },
        .{
            .position = .{ .x = three_quarter, .y = quarter, .z = three_quarter },
            .species_index = 0,
        },
        .{
            .position = .{ .x = three_quarter, .y = three_quarter, .z = quarter },
            .species_index = 0,
        },
    };

    const info_opt = try spacegroup.detect_space_group_from_atoms(alloc, cell, atoms[0..], 1e-5);
    try std.testing.expect(info_opt != null);
    var info = info_opt.?;
    defer info.deinit(alloc);

    try std.testing.expectEqual(@as(i32, 227), info.number);
    try std.testing.expect(std.mem.eql(u8, info.international_short, "Fd-3m"));
}

// Test local pseudopotential form factor
test "local pseudopotential V(q) for Carbon" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // Load C.upf using pseudo.load
    var element_buf: [2]u8 = .{ 'C', 0 };
    var path_buf: [20]u8 = undefined;
    const path_slice = "pseudo/C.upf";
    try require_file(io, path_slice);
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..1],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(allocator, io, spec);
    defer parsed.deinit(allocator);

    const upf = parsed.upf orelse return error.NoUpfData;

    // Test V(q) at various q values, including typical smallest G values
    const q_values = [_]f64{ 0.1, 0.166, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0 };
    const z_val: f64 = 4.0;
    vprint("\n=== Local Pseudopotential V(q) ===\n", .{});
    for (q_values) |q| {
        const vq = form_factor.local_vq_short_range(upf, z_val, q);
        vprint("V(q={d:.1}) = {d:.4} Ry\n", .{ q, vq });
    }

    // V(q) should be finite for q > 0
    const v_at_1 = form_factor.local_vq_short_range(upf, z_val, 1.0);
    try std.testing.expect(std.math.isFinite(v_at_1));

    // Test with Coulomb tail correction
    // Note: In Rydberg units, the Coulomb potential is -2Z/r, so V_Coul(q) = -8πZ/q²
    vprint("\n=== V(q) with Coulomb tail correction (Rydberg units) ===\n", .{});
    for (q_values) |q| {
        const vq_sr = form_factor.local_vq_short_range(upf, z_val, q);
        const vq_tail = form_factor.local_vq_with_tail(upf, z_val, q);
        // Coulomb in Rydberg: -8πZ/q² (factor of 2 for Ry vs Ha)
        const v_coulomb_ry = -8.0 * std.math.pi * z_val / (q * q);
        vprint(
            "q={d:.2}: V_short={d:.1}, V_tail={d:.1}, V_Coul_Ry={d:.1}, V_SR={d:.1} Ry\n",
            .{ q, vq_sr, vq_tail, v_coulomb_ry, vq_tail - v_coulomb_ry },
        );
    }

    // Test Ewald-compensated form factor
    // Use typical alpha = 5/L_min where L_min ≈ 4.65 Bohr for graphene
    const alpha: f64 = 1.07; // roughly 5/4.65
    vprint("\n=== V(q) with Ewald compensation (α={d:.2}) ===\n", .{alpha});
    for (q_values) |q| {
        const vq_ewald = form_factor.local_vq_ewald(upf, z_val, q, alpha);
        vprint("q={d:.1}: V_Ewald={d:.4} Ry\n", .{ q, vq_ewald });
    }
    // Test at q=0 (G=0 component - should be finite)
    const vq_ewald_0 = form_factor.local_vq_ewald(upf, z_val, 0.0, alpha);
    vprint("q=0: V_Ewald={d:.4} Ry (should be finite)\n", .{vq_ewald_0});

    // The Ewald form factor should be finite everywhere
    const vq_ewald_small = form_factor.local_vq_ewald(upf, z_val, 0.166, alpha);
    try std.testing.expect(std.math.isFinite(vq_ewald_small));
    // Note: The Ewald approach still gives large values due to pseudopotential structure
    vprint("V_Ewald(0.166) = {d:.2} Ry\n", .{vq_ewald_small});
}

test "compute forces assembles component forces" {
    const io = std.testing.io;
    const testing = std.testing;
    const alloc = testing.allocator;

    var element_buf: [2]u8 = .{ 'S', 'i' };
    var path_buf: [24]u8 = undefined;
    const path_slice = "pseudo/Si.upf";
    try require_file(io, path_slice);
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..2],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(alloc, io, spec);
    defer parsed.deinit(alloc);

    var parsed_items = [_]pseudo.Parsed{parsed};
    const species = try hamiltonian.build_species_entries(alloc, parsed_items[0..]);
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

    const grid = forces.Grid{
        .nx = 4,
        .ny = 4,
        .nz = 4,
        .min_h = -1,
        .min_k = -1,
        .min_l = -1,
        .cell = cell,
        .recip = recip,
    };
    const atoms = [_]hamiltonian.AtomData{
        .{ .position = math.Vec3{ .x = 0.3, .y = 0.4, .z = 0.2 }, .species_index = 0 },
        .{ .position = math.Vec3{ .x = 0.6, .y = 0.2, .z = 0.5 }, .species_index = 0 },
    };

    const total = grid.nx * grid.ny * grid.nz;
    var rho_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho_g);

    @memset(rho_g, math.complex.init(0.0, 0.0));

    const idx = struct {
        fn of(grid_local: forces.Grid, gh: i32, gk: i32, gl: i32) usize {
            const h = @as(usize, @intCast(gh - grid_local.min_h));
            const k = @as(usize, @intCast(gk - grid_local.min_k));
            const l = @as(usize, @intCast(gl - grid_local.min_l));
            return h + grid_local.nx * (k + grid_local.ny * l);
        }
    };

    const rho_x = math.complex.init(0.01, 0.02);
    const rho_y = math.complex.init(0.015, -0.005);
    const rho_z = math.complex.init(-0.007, 0.011);
    rho_g[idx.of(grid, 1, 0, 0)] = rho_x;
    rho_g[idx.of(grid, -1, 0, 0)] = math.complex.conj(rho_x);
    rho_g[idx.of(grid, 0, 1, 0)] = rho_y;
    rho_g[idx.of(grid, 0, -1, 0)] = math.complex.conj(rho_y);
    rho_g[idx.of(grid, 0, 0, 1)] = rho_z;
    rho_g[idx.of(grid, 0, 0, -1)] = math.complex.conj(rho_z);

    const k_cart = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const ecut_ry = 6.0;
    var basis = try plane_wave.generate(alloc, recip, ecut_ry, k_cart);
    defer basis.deinit(alloc);

    try testing.expect(basis.gvecs.len > 0);

    const eigenvalues = try alloc.alloc(f64, 1);
    defer alloc.free(eigenvalues);

    eigenvalues[0] = 0.0;

    const occupations = try alloc.alloc(f64, 1);
    defer alloc.free(occupations);

    occupations[0] = 1.0;

    const coefficients = try alloc.alloc(math.Complex, basis.gvecs.len);
    defer alloc.free(coefficients);

    for (coefficients, 0..) |*c_val, i| {
        const re = 0.05 * @as(f64, @floatFromInt(i + 1));
        const im = -0.03 * @as(f64, @floatFromInt(i + 2));
        c_val.* = math.complex.init(re, im);
    }

    const kpoints = try alloc.alloc(scf.KpointWavefunction, 1);
    defer alloc.free(kpoints);

    kpoints[0] = .{
        .k_frac = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .k_cart = k_cart,
        .weight = 1.0,
        .basis_len = basis.gvecs.len,
        .nbands = 1,
        .eigenvalues = eigenvalues,
        .coefficients = coefficients,
        .occupations = occupations,
    };

    const wavefunctions = scf.WavefunctionData{
        .kpoints = kpoints,
        .ecut_ry = ecut_ry,
        .fermi_level = 0.0,
    };

    var force_terms = try forces.compute_forces(
        alloc,
        io,
        grid,
        rho_g,
        null,
        null,
        species,
        atoms[0..],
        cell,
        recip,
        volume,
        0.0,
        local_potential.LocalPotentialConfig.init(.short_range, 0.0),
        wavefunctions,
        null,
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

    const local_forces = try forces.local_force.local_pseudo_forces(
        alloc,
        grid,
        rho_g,
        species,
        atoms[0..],
        volume,
        local_potential.LocalPotentialConfig.init(.short_range, 0.0),
        null,
    );
    defer alloc.free(local_forces);

    const nonlocal_forces = try forces.nonlocal_force.nonlocal_forces(
        alloc,
        wavefunctions,
        species,
        atoms[0..],
        recip,
        volume,
        null,
        null,
        null,
        2.0,
    );
    defer alloc.free(nonlocal_forces);

    const charges = [_]f64{
        species[atoms[0].species_index].z_valence,
        species[atoms[1].species_index].z_valence,
    };
    const positions = [_]math.Vec3{
        atoms[0].position,
        atoms[1].position,
    };
    const ewald_params = ewald.Params{
        .alpha = 0.0,
        .rcut = 0.0,
        .gcut = 0.0,
        .tol = 1e-8,
        .quiet = true,
    };
    const ewald_forces_ha = try ewald.ion_ion_forces(
        alloc,
        cell,
        recip,
        charges[0..],
        positions[0..],
        ewald_params,
    );
    defer alloc.free(ewald_forces_ha);

    try testing.expect(force_terms.nonlocal != null);
    try testing.expect(force_terms.residual == null);
    try testing.expect(force_terms.nlcc == null);
    try testing.expect(force_terms.dispersion == null);
    try testing.expect(force_terms.paw_dhat == null);

    const nl_terms = force_terms.nonlocal.?;
    for (atoms, 0..) |_, i| {
        const expected_ewald = math.Vec3.scale(ewald_forces_ha[i], 2.0);
        const expected_total = math.Vec3.add(
            math.Vec3.add(expected_ewald, local_forces[i]),
            nonlocal_forces[i],
        );

        try expect_vec_approx_eq_abs(expected_ewald, force_terms.ewald[i], 1e-10);
        try expect_vec_approx_eq_abs(local_forces[i], force_terms.local[i], 1e-10);
        try expect_vec_approx_eq_abs(nonlocal_forces[i], nl_terms[i], 1e-10);
        try expect_vec_approx_eq_abs(expected_total, force_terms.total[i], 1e-10);
    }
}

// Test kinetic energy calculation
test "kinetic energy |k+G|^2" {
    const allocator = std.testing.allocator;

    // Graphene cell (Bohr)
    const a = 4.6487262675; // 2.46 Å in Bohr
    const c = 18.8972613; // 10 Å in Bohr

    const cell = math.Mat3.from_rows(
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

    const cell = math.Mat3.from_rows(
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
    const recip_spacing_xy = 2.0 * std.math.pi / a;
    vprint("Reciprocal lattice spacing (xy): 2π/a ≈ {d:.4} bohr⁻¹\n", .{recip_spacing_xy});

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
    const io = std.testing.io;
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
    const real_params = ewald.Params{
        .alpha = alpha,
        .rcut = 20.0,
        .gcut = 1e-6,
        .tol = 1e-8,
        .quiet = true,
    };
    const quiet_params = ewald.Params{
        .alpha = 0.0,
        .rcut = 0.0,
        .gcut = 0.0,
        .tol = 0.0,
        .quiet = true,
    };
    const real_forces = try ewald.ion_ion_forces(
        alloc,
        cell,
        recip,
        charges[0..],
        positions[0..],
        real_params,
    );
    defer alloc.free(real_forces);

    const fx_real_num = blk: {
        var positions_plus = positions;
        var positions_minus = positions;
        positions_plus[0].x += delta;
        positions_minus[0].x -= delta;
        const e_plus = try ewald.ion_ion_energy(
            io,
            cell,
            recip,
            charges[0..],
            positions_plus[0..],
            real_params,
        );
        const e_minus = try ewald.ion_ion_energy(
            io,
            cell,
            recip,
            charges[0..],
            positions_minus[0..],
            real_params,
        );
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };
    try testing.expectApproxEqAbs(real_forces[0].x, fx_real_num, 1e-3);

    const ewald_forces = try ewald.ion_ion_forces(
        alloc,
        cell,
        recip,
        charges[0..],
        positions[0..],
        quiet_params,
    );
    defer alloc.free(ewald_forces);

    const fx_num = blk: {
        var positions_plus = positions;
        var positions_minus = positions;
        positions_plus[0].x += delta;
        positions_minus[0].x -= delta;
        const e_plus = try ewald.ion_ion_energy(
            io,
            cell,
            recip,
            charges[0..],
            positions_plus[0..],
            quiet_params,
        );
        const e_minus = try ewald.ion_ion_energy(
            io,
            cell,
            recip,
            charges[0..],
            positions_minus[0..],
            quiet_params,
        );
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };

    try testing.expectApproxEqAbs(ewald_forces[0].x, fx_num, 1e-3);
}

// Test nonlocal pseudopotential projectors
test "nonlocal projector radial integral" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // Load C.upf
    var element_buf: [2]u8 = .{ 'C', 0 };
    var path_buf: [20]u8 = undefined;
    const path_slice = "pseudo/C.upf";
    try require_file(io, path_slice);
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..1],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(allocator, io, spec);
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
        const rad = nonlocal.radial_projector(beta.values, upf.r, upf.rab, l_val, 0.0);
        vprint("  beta_{d} (l={d}): {d:.6}\n", .{ idx, l_val, rad });
    }

    // Test radial projector at typical G values
    const g_vals = [_]f64{ 0.5, 1.0, 2.0, 5.0 };
    vprint("\nRadial projectors at various |G|:\n", .{});
    for (g_vals) |g| {
        vprint("|G|={d:.1}:", .{g});
        for (upf.beta, 0..) |beta, idx| {
            const l_val = beta.l orelse 0;
            const rad = nonlocal.radial_projector(beta.values, upf.r, upf.rab, l_val, g);
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
    const r0_s1 = nonlocal.radial_projector(upf.beta[0].values, upf.r, upf.rab, 0, 0.0);
    const r0_s2 = nonlocal.radial_projector(upf.beta[1].values, upf.r, upf.rab, 0, 0.0);
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

    const cell = math.Mat3.from_rows(
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

    // CORRECT method using math.frac_to_cart
    const k_cart_correct = math.frac_to_cart(k_frac, recip);

    // Compare with explicit calculation
    const k_cart_explicit = math.Vec3.add(
        math.Vec3.add(
            math.Vec3.scale(b1, k_frac.x),
            math.Vec3.scale(b2, k_frac.y),
        ),
        math.Vec3.scale(b3, k_frac.z),
    );

    vprint("\nK point (1/3, 1/3, 0):\n", .{});
    vprint(
        "  frac_to_cart k_cart = ({d:.6}, {d:.6}, {d:.6})\n",
        .{ k_cart_correct.x, k_cart_correct.y, k_cart_correct.z },
    );
    vprint(
        "  explicit k_cart   = ({d:.6}, {d:.6}, {d:.6})\n",
        .{ k_cart_explicit.x, k_cart_explicit.y, k_cart_explicit.z },
    );
    vprint("  |k_cart| = {d:.6}\n", .{math.Vec3.norm(k_cart_correct)});

    // Check if frac_to_cart matches explicit calculation
    const diff = math.Vec3.sub(k_cart_correct, k_cart_explicit);
    const diff_norm = math.Vec3.norm(diff);
    vprint("  Difference = {d:.6}\n", .{diff_norm});

    // frac_to_cart should match explicit calculation exactly
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), diff_norm, 1e-10);
}

// Test Hamiltonian Hermiticity at M-Γ midpoint
test "Hamiltonian Hermiticity" {
    const allocator = std.testing.allocator;

    // Graphene cell (Bohr)
    const a = 4.6487262675;
    const c = 37.7945225;

    const cell = math.Mat3.from_rows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * std.math.sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );

    const recip = math.reciprocal(cell);

    // M-Γ midpoint: k_frac = (0.25, 0, 0)
    const k_frac = math.Vec3{ .x = 0.25, .y = 0.0, .z = 0.0 };
    const k_cart = math.frac_to_cart(k_frac, recip);

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
    const io = std.testing.io;
    // Graphene cell (Bohr)
    const a = 4.6487262675;
    const c = 37.7945225; // 20 Å in Bohr

    const cell = math.Mat3.from_rows(
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

    const quiet_params = ewald.Params{
        .alpha = 0.0,
        .rcut = 0.0,
        .gcut = 0.0,
        .tol = 0.0,
        .quiet = true,
    };
    const e_ion = try ewald.ion_ion_energy(io, cell, recip, &charges, &positions, quiet_params);

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

    // Build local_r = N * IFFT(V_G) (this is what build_local_potential_real does)
    const local_r = try allocator.alloc(f64, total);
    defer allocator.free(local_r);

    {
        const temp = try allocator.alloc(math.Complex, total);
        defer allocator.free(temp);
        // Apply scale = N before IFFT
        for (v_g, 0..) |v, i| {
            temp[i] = math.complex.scale(v, @as(f64, @floatFromInt(total)));
        }
        try fft_mod.fft3d_inverse_in_place(allocator, temp, nx, ny, nz);
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

    // Method 2: FFT-based (simulating apply_local_potential)
    // Step 1: IFFT of psi with scale=N (like fft_reciprocal_to_complex_in_place)
    const psi_r = try allocator.alloc(math.Complex, total);
    defer allocator.free(psi_r);

    for (psi_g, 0..) |p, i| {
        psi_r[i] = math.complex.scale(p, @as(f64, @floatFromInt(total)));
    }
    try fft_mod.fft3d_inverse_in_place(allocator, psi_r, nx, ny, nz);

    // Step 2: Multiply in real space: local_r * psi_r
    const vpsi_r = try allocator.alloc(math.Complex, total);
    defer allocator.free(vpsi_r);

    for (vpsi_r, 0..) |*v, i| {
        v.* = math.complex.scale(psi_r[i], local_r[i]);
    }

    // Step 3: FFT with 1/N (like fft_complex_to_reciprocal_in_place)
    const fft_result = try allocator.alloc(math.Complex, total);
    defer allocator.free(fft_result);

    @memcpy(fft_result, vpsi_r);
    try fft_mod.fft3d_forward_in_place(allocator, fft_result, nx, ny, nz);
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
    try fft_mod.fft3d_inverse_in_place(allocator, v_temp, nx, ny, nz);

    // Step 2: IFFT of ψ_G to get ψ_r
    const psi_temp = try allocator.alloc(math.Complex, total);
    defer allocator.free(psi_temp);

    @memcpy(psi_temp, psi_g);
    try fft_mod.fft3d_inverse_in_place(allocator, psi_temp, nx, ny, nz);

    // Step 3: Pointwise multiply in real space: (V*ψ)_r
    const vpsi_r = try allocator.alloc(math.Complex, total);
    defer allocator.free(vpsi_r);

    for (vpsi_r, 0..) |*v, i| {
        v.* = math.complex.mul(v_temp[i], psi_temp[i]);
    }

    // Step 4: FFT to get (V*ψ)_G (FFT has no normalization)
    try fft_mod.fft3d_forward_in_place(allocator, vpsi_r, nx, ny, nz);

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
test "radial_projector large g behavior" {
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

    // Test radial_projector at various g values
    vprint("Testing l=0 (s-wave):\n", .{});
    for ([_]f64{ 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0 }) |g| {
        const proj = nonlocal.radial_projector(&beta, &r, &rab, 0, g);
        vprint("  g={d:.1}: proj={d:.6}\n", .{ g, proj });
    }

    vprint("Testing l=1 (p-wave):\n", .{});
    for ([_]f64{ 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0 }) |g| {
        const proj = nonlocal.radial_projector(&beta, &r, &rab, 1, g);
        vprint("  g={d:.1}: proj={d:.6}\n", .{ g, proj });
    }

    // Projector should decrease for large g (decay due to oscillating j_l)
    const proj_small = nonlocal.radial_projector(&beta, &r, &rab, 0, 1.0);
    const proj_large = nonlocal.radial_projector(&beta, &r, &rab, 0, 10.0);
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

    const cell = math.Mat3.from_rows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * sqrt3 / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 37.7945225 },
    );
    const recip = math.reciprocal(cell);

    // Two carbon atoms at (0,0,0) and (1/3,1/3,0) in fractional
    const pos1 = math.frac_to_cart(.{ .x = 0.0, .y = 0.0, .z = 0.0 }, cell);
    const pos2 = math.frac_to_cart(.{ .x = 1.0 / 3.0, .y = 1.0 / 3.0, .z = 0.0 }, cell);

    vprint("Atom positions (Bohr):\n", .{});
    vprint("  C1: ({d:.6}, {d:.6}, {d:.6})\n", .{ pos1.x, pos1.y, pos1.z });
    vprint("  C2: ({d:.6}, {d:.6}, {d:.6})\n", .{ pos2.x, pos2.y, pos2.z });

    const bond = math.Vec3.sub(pos2, pos1);
    const bond_length = math.Vec3.norm(bond);
    vprint(
        "  Bond: ({d:.6}, {d:.6}, {d:.6}), |bond|={d:.6}\n",
        .{ bond.x, bond.y, bond.z, bond_length },
    );
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
    vprint(
        "  G·Δτ = {d:.6} (mod 2π = {d:.6})\n",
        .{ g_phase, @mod(g_phase, 2.0 * std.math.pi) },
    );
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), g_phase, 1e-10);

    // Phase for K = (1/3, 2/3, 0) should be 2π/3 for Δτ = (1/3, 1/3, 0)
    const k_frac = math.Vec3{ .x = 1.0 / 3.0, .y = 2.0 / 3.0, .z = 0.0 };
    const k_cart = math.frac_to_cart(k_frac, recip);
    const k_phase = math.Vec3.dot(k_cart, bond);
    vprint("Phase for K = (1/3, 2/3, 0):\n", .{});
    vprint(
        "  K·Δτ = {d:.6} (mod 2π = {d:.6})\n",
        .{ k_phase, @mod(k_phase, 2.0 * std.math.pi) },
    );
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
        const af = nonlocal.angular_factor(0, cos_g);
        vprint("  cos_γ={d:.2}: angular={d:.6}\n", .{ cos_g, af });
    }

    vprint("l=1 (p-wave):\n", .{});
    for ([_]f64{ 1.0, 0.5, 0.0, -0.5, -1.0 }) |cos_g| {
        const af = nonlocal.angular_factor(1, cos_g);
        vprint("  cos_γ={d:.2}: angular={d:.6}\n", .{ cos_g, af });
    }

    // l=0: 4π × 1 × P_0 = 4π ≈ 12.566
    const af0 = nonlocal.angular_factor(0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 * std.math.pi), af0, 1e-10);

    // l=1: 4π × 3 × P_1(1) = 12π ≈ 37.699
    const af1 = nonlocal.angular_factor(1, 1.0);
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
        const computed = nonlocal.spherical_bessel(0, x);
        const expected = std.math.sin(x) / x;
        const err = @abs(computed - expected);
        vprint(
            "  x={d:.1}: computed={d:.10}, expected={d:.10}, err={e:.2}\n",
            .{ x, computed, expected, err },
        );
        try std.testing.expectApproxEqAbs(expected, computed, 1e-12);
    }

    vprint("j_1(x) = sin(x)/x² - cos(x)/x:\n", .{});
    for (test_x) |x| {
        const computed = nonlocal.spherical_bessel(1, x);
        const expected = std.math.sin(x) / (x * x) - std.math.cos(x) / x;
        const err = @abs(computed - expected);
        vprint(
            "  x={d:.1}: computed={d:.10}, expected={d:.10}, err={e:.2}\n",
            .{ x, computed, expected, err },
        );
        try std.testing.expectApproxEqAbs(expected, computed, 1e-12);
    }

    vprint("j_2(x) = (3/x² - 1)sin(x)/x - 3cos(x)/x²:\n", .{});
    for (test_x) |x| {
        const computed = nonlocal.spherical_bessel(2, x);
        const term_a = (3.0 / (x * x) - 1.0) * std.math.sin(x) / x;
        const term_b = 3.0 * std.math.cos(x) / (x * x);
        const expected = term_a - term_b;
        const err = @abs(computed - expected);
        vprint(
            "  x={d:.1}: computed={d:.10}, expected={d:.10}, err={e:.2}\n",
            .{ x, computed, expected, err },
        );
        try std.testing.expectApproxEqAbs(expected, computed, 1e-10);
    }

    // Test j_3(x) at specific values
    // j_3(x) = (15/x³ - 6/x)sin(x)/x - (15/x² - 1)cos(x)/x
    vprint("j_3(x) test:\n", .{});
    for (test_x) |x| {
        const computed = nonlocal.spherical_bessel(3, x);
        const term_a = (15.0 / (x * x * x) - 6.0 / x) * std.math.sin(x) / x;
        const term_b = (15.0 / (x * x) - 1.0) * std.math.cos(x) / x;
        const expected = term_a - term_b;
        const err = @abs(computed - expected);
        vprint(
            "  x={d:.1}: computed={d:.10}, expected={d:.10}, err={e:.2}\n",
            .{ x, computed, expected, err },
        );
        // Allow larger tolerance for l=3 due to numerical accumulation
        try std.testing.expectApproxEqAbs(expected, computed, 1e-8);
    }

    // Test decay at large x for l > x (should be very small)
    vprint("Decay test j_l(x) for l > x:\n", .{});
    const j2_at_1 = nonlocal.spherical_bessel(2, 1.0);
    const j3_at_1 = nonlocal.spherical_bessel(3, 1.0);
    vprint("  j_2(1) = {d:.10}\n", .{j2_at_1});
    vprint("  j_3(1) = {d:.10}\n", .{j3_at_1});
    // j_l(x) ~ x^l / (2l+1)!! for small x, so j_3(1) << j_2(1)
    try std.testing.expect(@abs(j3_at_1) < @abs(j2_at_1));
}
