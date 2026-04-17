//! Slow GTO regression tests that are intentionally excluded from the
//! day-to-day `zig build test-full` suite.

const std = @import("std");
const math = @import("../math/math.zig");
const Vec3 = math.Vec3;
const basis_mod = @import("../basis/basis.zig");
const ContractedShell = basis_mod.ContractedShell;
const sto3g = @import("../basis/sto3g.zig");
const aux_basis = @import("../basis/aux_basis.zig");
const basis631g = @import("../basis/basis631g.zig");
const basis631g_2dfp = @import("../basis/basis631g_2dfp.zig");
const integrals = @import("../integrals/integrals.zig");
const obara_saika = integrals.obara_saika;
const gto_scf = @import("gto_scf.zig");
const kohn_sham = @import("kohn_sham.zig");
const optimizer = @import("optimizer.zig");
const vibrational = @import("vibrational.zig");

fn buildSto3gShells(
    alloc: std.mem.Allocator,
    atomic_numbers: []const u8,
    positions: []const Vec3,
) ![]ContractedShell {
    var shells_list: std.ArrayListUnmanaged(ContractedShell) = .empty;
    errdefer shells_list.deinit(alloc);

    for (atomic_numbers, 0..) |z, i| {
        const center = positions[i];
        const n_shells = sto3g.numShellsForAtom(@intCast(z)) orelse return error.UnsupportedElement;
        const atom_shells = sto3g.buildAtomShells(@intCast(z), center) orelse return error.UnsupportedElement;
        for (0..n_shells) |s| {
            try shells_list.append(alloc, atom_shells[s]);
        }
    }

    return try shells_list.toOwnedSlice(alloc);
}

test "General RHF H2O 6-31G" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    const o_data = basis631g.buildAtomShells(8, nuc_positions[0]).?;
    const h1_data = basis631g.buildAtomShells(1, nuc_positions[1]).?;
    const h2_data = basis631g.buildAtomShells(1, nuc_positions[2]).?;

    var all_shells: [basis631g.MAX_SHELLS_PER_ATOM * 3]ContractedShell = undefined;
    var count: usize = 0;
    for (o_data.shells[0..o_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h1_data.shells[0..h1_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h2_data.shells[0..h2_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }

    const shells = all_shells[0..count];
    try testing.expectEqual(@as(usize, 13), obara_saika.totalBasisFunctions(shells));

    var result = try gto_scf.runGeneralRhfScf(alloc, shells, &nuc_positions, &nuc_charges, 10, .{});
    defer result.deinit(alloc);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(-75.9839484981, result.total_energy, 1e-4);
    try testing.expectApproxEqAbs(9.1882584178, result.nuclear_repulsion, 1e-6);
}

test "KS-DFT H2O STO-3G B3LYP" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };
    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    var result = try kohn_sham.runKohnShamScf(alloc, std.testing.io, &shells, &nuc_positions, &nuc_charges, 10, .{
        .xc_functional = .b3lyp,
        .n_radial = 99,
        .n_angular = 590,
        .prune = false,
    });
    defer result.deinit(alloc);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(-75.3125872072, result.total_energy, 1e-3);
}

test "QM9 validation: H2O B3LYP/6-31G(2df,p) vs PySCF" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.2217971552 },
        .{ .x = 0.0, .y = 1.4308250325, .z = -0.8871886210 },
        .{ .x = 0.0, .y = -1.4308250325, .z = -0.8871886210 },
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    const o_data = basis631g_2dfp.buildAtomShells(8, nuc_positions[0]).?;
    const h1_data = basis631g_2dfp.buildAtomShells(1, nuc_positions[1]).?;
    const h2_data = basis631g_2dfp.buildAtomShells(1, nuc_positions[2]).?;

    var all_shells: [basis631g_2dfp.MAX_SHELLS_PER_ATOM * 3]ContractedShell = undefined;
    var count: usize = 0;
    for (o_data.shells[0..o_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h1_data.shells[0..h1_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h2_data.shells[0..h2_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }

    var result = try kohn_sham.runKohnShamScf(
        alloc,
        std.testing.io,
        all_shells[0..count],
        &nuc_positions,
        &nuc_charges,
        10,
        .{
            .xc_functional = .b3lyp,
            .n_radial = 50,
            .n_angular = 302,
            .prune = false,
            .use_direct_scf = true,
            .schwarz_threshold = 1e-10,
        },
    );
    defer result.deinit(alloc);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(@as(f64, -76.425747169883), result.total_energy, 5e-3);
}

test "QM9 validation: CH4 B3LYP/6-31G(2df,p) vs PySCF" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.1888607200, .y = 1.1888607200, .z = 1.1888607200 },
        .{ .x = -1.1888607200, .y = -1.1888607200, .z = 1.1888607200 },
        .{ .x = -1.1888607200, .y = 1.1888607200, .z = -1.1888607200 },
        .{ .x = 1.1888607200, .y = -1.1888607200, .z = -1.1888607200 },
    };
    const nuc_charges = [_]f64{ 6.0, 1.0, 1.0, 1.0, 1.0 };

    const c_data = basis631g_2dfp.buildAtomShells(6, nuc_positions[0]).?;
    const h1_data = basis631g_2dfp.buildAtomShells(1, nuc_positions[1]).?;
    const h2_data = basis631g_2dfp.buildAtomShells(1, nuc_positions[2]).?;
    const h3_data = basis631g_2dfp.buildAtomShells(1, nuc_positions[3]).?;
    const h4_data = basis631g_2dfp.buildAtomShells(1, nuc_positions[4]).?;

    var all_shells: [basis631g_2dfp.MAX_SHELLS_PER_ATOM * 5]ContractedShell = undefined;
    var count: usize = 0;
    inline for (.{ c_data, h1_data, h2_data, h3_data, h4_data }) |atom_data| {
        for (atom_data.shells[0..atom_data.count]) |s| {
            all_shells[count] = s;
            count += 1;
        }
    }

    var result = try kohn_sham.runKohnShamScf(
        alloc,
        std.testing.io,
        all_shells[0..count],
        &nuc_positions,
        &nuc_charges,
        10,
        .{
            .xc_functional = .b3lyp,
            .n_radial = 50,
            .n_angular = 302,
            .prune = false,
            .use_direct_scf = true,
            .schwarz_threshold = 1e-10,
        },
    );
    defer result.deinit(alloc);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(@as(f64, -40.525553681791), result.total_energy, 5e-3);
}

test "QM9 validation: CH2O B3LYP/6-31G(2df,p) vs PySCF" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = -1.0000430651 },
        .{ .x = 0.0, .y = 0.0, .z = 1.2757541067 },
        .{ .x = 0.0, .y = 1.7680277621, .z = -2.0970290804 },
        .{ .x = 0.0, .y = -1.7680277621, .z = -2.0970290804 },
    };
    const nuc_charges = [_]f64{ 6.0, 8.0, 1.0, 1.0 };

    const c_data = basis631g_2dfp.buildAtomShells(6, nuc_positions[0]).?;
    const o_data = basis631g_2dfp.buildAtomShells(8, nuc_positions[1]).?;
    const h1_data = basis631g_2dfp.buildAtomShells(1, nuc_positions[2]).?;
    const h2_data = basis631g_2dfp.buildAtomShells(1, nuc_positions[3]).?;

    var all_shells: [basis631g_2dfp.MAX_SHELLS_PER_ATOM * 4]ContractedShell = undefined;
    var count: usize = 0;
    inline for (.{ c_data, o_data, h1_data, h2_data }) |atom_data| {
        for (atom_data.shells[0..atom_data.count]) |s| {
            all_shells[count] = s;
            count += 1;
        }
    }

    var result = try kohn_sham.runKohnShamScf(
        alloc,
        std.testing.io,
        all_shells[0..count],
        &nuc_positions,
        &nuc_charges,
        16,
        .{
            .xc_functional = .b3lyp,
            .n_radial = 50,
            .n_angular = 302,
            .prune = false,
            .use_direct_scf = true,
            .schwarz_threshold = 1e-10,
        },
    );
    defer result.deinit(alloc);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(@as(f64, -114.511830940393), result.total_energy, 5e-3);
}

test "DF KS-DFT H2O STO-3G LDA" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };
    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    const aux_o = aux_basis.buildDef2UniversalJkfit(8, nuc_positions[0]).?;
    const aux_h1 = aux_basis.buildDef2UniversalJkfit(1, nuc_positions[1]).?;
    const aux_h2 = aux_basis.buildDef2UniversalJkfit(1, nuc_positions[2]).?;

    var all_aux: [aux_basis.MAX_AUX_SHELLS * 3]ContractedShell = undefined;
    var aux_count: usize = 0;
    for (aux_o.shells[0..aux_o.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }
    for (aux_h1.shells[0..aux_h1.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }
    for (aux_h2.shells[0..aux_h2.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }

    var result_conv = try kohn_sham.runKohnShamScf(
        alloc,
        std.testing.io,
        &shells,
        &nuc_positions,
        &nuc_charges,
        10,
        .{ .xc_functional = .lda_svwn },
    );
    defer result_conv.deinit(alloc);

    var result_df = try kohn_sham.runKohnShamScf(
        alloc,
        std.testing.io,
        &shells,
        &nuc_positions,
        &nuc_charges,
        10,
        .{
            .xc_functional = .lda_svwn,
            .use_density_fitting = true,
            .aux_shells = all_aux[0..aux_count],
        },
    );
    defer result_df.deinit(alloc);

    try testing.expect(result_df.converged);
    try testing.expectApproxEqAbs(result_conv.total_energy, result_df.total_energy, 1e-3);
}

test "H2 STO-3G geometry optimization" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.8, .y = 0.0, .z = 0.0 },
    };
    const charges = [_]f64{ 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 1, 1 };

    const shells = try buildSto3gShells(alloc, &atomic_numbers, &positions);
    defer alloc.free(shells);

    var result = try optimizer.optimizeGeometry(
        alloc,
        shells,
        &positions,
        &charges,
        2,
        .{
            .max_steps = 50,
            .print_progress = false,
            .scf_params = .{
                .max_iter = 100,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
            },
        },
    );
    defer result.deinit(alloc);

    const dx = result.positions[1].x - result.positions[0].x;
    const dy = result.positions[1].y - result.positions[0].y;
    const dz = result.positions[1].z - result.positions[0].z;
    const bond_length = @sqrt(dx * dx + dy * dy + dz * dz);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.3459, bond_length, 0.01);
    try testing.expectApproxEqAbs(-1.1175, result.energy, 1e-3);
}

test "H2O STO-3G geometry optimization" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const r_oh: f64 = 1.809;
    const angle: f64 = 104.5 * std.math.pi / 180.0;
    const h1_y = r_oh * @sin(angle / 2.0);
    const h1_z = -r_oh * @cos(angle / 2.0);

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = h1_y, .z = h1_z },
        .{ .x = 0.0, .y = -h1_y, .z = h1_z },
    };
    const charges = [_]f64{ 8.0, 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 8, 1, 1 };

    const shells = try buildSto3gShells(alloc, &atomic_numbers, &positions);
    defer alloc.free(shells);

    var result = try optimizer.optimizeGeometry(
        alloc,
        shells,
        &positions,
        &charges,
        10,
        .{
            .max_steps = 50,
            .print_progress = false,
            .scf_params = .{
                .max_iter = 100,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
            },
        },
    );
    defer result.deinit(alloc);

    const oh1 = Vec3{
        .x = result.positions[1].x - result.positions[0].x,
        .y = result.positions[1].y - result.positions[0].y,
        .z = result.positions[1].z - result.positions[0].z,
    };
    const oh2 = Vec3{
        .x = result.positions[2].x - result.positions[0].x,
        .y = result.positions[2].y - result.positions[0].y,
        .z = result.positions[2].z - result.positions[0].z,
    };
    const r1 = oh1.norm();
    const r2 = oh2.norm();
    const cos_angle = oh1.dot(oh2) / (r1 * r2);
    const opt_angle = std.math.acos(cos_angle) * 180.0 / std.math.pi;

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.8697, r1, 0.02);
    try testing.expectApproxEqAbs(1.8697, r2, 0.02);
    try testing.expectApproxEqAbs(100.0, opt_angle, 2.0);
    try testing.expectApproxEqAbs(-74.9659, result.energy, 1e-3);
}

test "H2 LDA STO-3G geometry optimization" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.8, .y = 0.0, .z = 0.0 },
    };
    const charges = [_]f64{ 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 1, 1 };

    const shells = try buildSto3gShells(alloc, &atomic_numbers, &positions);
    defer alloc.free(shells);

    var result = try optimizer.optimizeKsDftGeometry(
        alloc,
        std.testing.io,
        shells,
        &positions,
        &charges,
        2,
        .{
            .max_steps = 50,
            .print_progress = false,
            .atomic_numbers = &atomic_numbers,
            .ks_params = .{
                .xc_functional = .lda_svwn,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
                .n_radial = 50,
                .n_angular = 194,
                .prune = false,
            },
        },
    );
    defer result.deinit(alloc);

    const dx = result.positions[1].x - result.positions[0].x;
    const dy = result.positions[1].y - result.positions[0].y;
    const dz = result.positions[1].z - result.positions[0].z;
    const bond_length = @sqrt(dx * dx + dy * dy + dz * dz);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.392, bond_length, 0.05);
    try testing.expectApproxEqAbs(-1.1212, result.energy, 5e-3);
}

test "H2O LDA STO-3G geometry optimization" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const r_oh: f64 = 1.809;
    const angle: f64 = 104.5 * std.math.pi / 180.0;
    const h1_y = r_oh * @sin(angle / 2.0);
    const h1_z = -r_oh * @cos(angle / 2.0);

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = h1_y, .z = h1_z },
        .{ .x = 0.0, .y = -h1_y, .z = h1_z },
    };
    const charges = [_]f64{ 8.0, 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 8, 1, 1 };

    const shells = try buildSto3gShells(alloc, &atomic_numbers, &positions);
    defer alloc.free(shells);

    var result = try optimizer.optimizeKsDftGeometry(
        alloc,
        std.testing.io,
        shells,
        &positions,
        &charges,
        10,
        .{
            .max_steps = 50,
            .print_progress = false,
            .atomic_numbers = &atomic_numbers,
            .ks_params = .{
                .xc_functional = .lda_svwn,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
                .n_radial = 50,
                .n_angular = 194,
                .prune = false,
            },
        },
    );
    defer result.deinit(alloc);

    const oh1 = Vec3{
        .x = result.positions[1].x - result.positions[0].x,
        .y = result.positions[1].y - result.positions[0].y,
        .z = result.positions[1].z - result.positions[0].z,
    };
    const oh2 = Vec3{
        .x = result.positions[2].x - result.positions[0].x,
        .y = result.positions[2].y - result.positions[0].y,
        .z = result.positions[2].z - result.positions[0].z,
    };
    const r1 = oh1.norm();
    const r2 = oh2.norm();
    const cos_angle = oh1.dot(oh2) / (r1 * r2);
    const opt_angle = std.math.acos(cos_angle) * 180.0 / std.math.pi;

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.948, r1, 0.05);
    try testing.expectApproxEqAbs(1.948, r2, 0.05);
    try testing.expectApproxEqAbs(96.7, opt_angle, 3.0);
    try testing.expectApproxEqAbs(-74.743, result.energy, 0.01);
}

test "KS-DFT B3LYP geometry optimization H2 STO-3G" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const charges = [_]f64{ 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 1, 1 };
    const init_shells = [_]ContractedShell{
        .{ .center = positions[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = positions[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    const shells = try alloc.alloc(ContractedShell, init_shells.len);
    @memcpy(shells, &init_shells);
    defer alloc.free(shells);

    var result = try optimizer.optimizeKsDftGeometry(
        alloc,
        std.testing.io,
        shells,
        &positions,
        &charges,
        2,
        .{
            .max_steps = 50,
            .print_progress = false,
            .atomic_numbers = &atomic_numbers,
            .ks_params = .{
                .xc_functional = .b3lyp,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
                .n_radial = 50,
                .n_angular = 194,
                .prune = false,
            },
        },
    );
    defer result.deinit(alloc);

    const r_hh = @sqrt(
        (result.positions[1].x - result.positions[0].x) * (result.positions[1].x - result.positions[0].x) +
            (result.positions[1].y - result.positions[0].y) * (result.positions[1].y - result.positions[0].y) +
            (result.positions[1].z - result.positions[0].z) * (result.positions[1].z - result.positions[0].z),
    );

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.3764, r_hh, 0.05);
    try testing.expectApproxEqAbs(-1.16554, result.energy, 0.005);
}

test "KS-DFT B3LYP geometry optimization H2O STO-3G" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };
    const charges = [_]f64{ 8.0, 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 8, 1, 1 };
    const init_shells = [_]ContractedShell{
        .{ .center = positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    const shells = try alloc.alloc(ContractedShell, init_shells.len);
    @memcpy(shells, &init_shells);
    defer alloc.free(shells);

    var result = try optimizer.optimizeKsDftGeometry(
        alloc,
        std.testing.io,
        shells,
        &positions,
        &charges,
        10,
        .{
            .max_steps = 50,
            .print_progress = false,
            .atomic_numbers = &atomic_numbers,
            .ks_params = .{
                .xc_functional = .b3lyp,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
                .n_radial = 80,
                .n_angular = 302,
                .prune = false,
            },
        },
    );
    defer result.deinit(alloc);

    const oh1 = Vec3{
        .x = result.positions[1].x - result.positions[0].x,
        .y = result.positions[1].y - result.positions[0].y,
        .z = result.positions[1].z - result.positions[0].z,
    };
    const oh2 = Vec3{
        .x = result.positions[2].x - result.positions[0].x,
        .y = result.positions[2].y - result.positions[0].y,
        .z = result.positions[2].z - result.positions[0].z,
    };
    const r1 = oh1.norm();
    const r2 = oh2.norm();
    const cos_angle = oh1.dot(oh2) / (r1 * r2);
    const opt_angle = std.math.acos(cos_angle) * 180.0 / std.math.pi;

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.941, r1, 0.05);
    try testing.expectApproxEqAbs(1.941, r2, 0.05);
    try testing.expectApproxEqAbs(97.2, opt_angle, 3.0);
    try testing.expectApproxEqAbs(-75.323, result.energy, 0.01);
}

test "H2 STO-3G vibrational frequency" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.3459, .y = 0.0, .z = 0.0 },
    };
    const charges = [_]f64{ 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 1, 1 };

    const shells = try buildSto3gShells(alloc, &atomic_numbers, &positions);
    defer alloc.free(shells);

    var result = try vibrational.vibrationalAnalysis(
        alloc,
        shells,
        &positions,
        &charges,
        &atomic_numbers,
        2,
        .{
            .displacement = 0.001,
            .print_progress = false,
            .use_integer_masses = true,
            .scf_params = .{
                .max_iter = 100,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
            },
        },
    );
    defer result.deinit(alloc);

    try testing.expect(result.vib_frequencies_cm1.len == 1);
    try testing.expectApproxEqAbs(5481.0, result.vib_frequencies_cm1[0], 30.0);
}

test "H2O STO-3G vibrational frequencies" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4326, .z = -1.2015 },
        .{ .x = 0.0, .y = -1.4326, .z = -1.2015 },
    };
    const charges = [_]f64{ 8.0, 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 8, 1, 1 };

    const shells = try buildSto3gShells(alloc, &atomic_numbers, &positions);
    defer alloc.free(shells);

    var result = try vibrational.vibrationalAnalysis(
        alloc,
        shells,
        &positions,
        &charges,
        &atomic_numbers,
        10,
        .{
            .displacement = 0.001,
            .print_progress = false,
            .use_integer_masses = true,
            .scf_params = .{
                .max_iter = 100,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
            },
        },
    );
    defer result.deinit(alloc);

    try testing.expect(result.vib_frequencies_cm1.len == 3);

    var sorted_freqs: [3]f64 = undefined;
    @memcpy(&sorted_freqs, result.vib_frequencies_cm1[0..3]);
    std.mem.sort(f64, &sorted_freqs, {}, std.sort.asc(f64));

    try testing.expectApproxEqAbs(2170.0, sorted_freqs[0], 150.0);
    try testing.expectApproxEqAbs(4139.0, sorted_freqs[1], 100.0);
    try testing.expectApproxEqAbs(4390.0, sorted_freqs[2], 100.0);
    try testing.expectApproxEqAbs(0.0244, result.zpve, 0.003);
}

test "H2O STO-3G thermodynamic properties" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0616 },
        .{ .x = 0.0, .y = 1.4325, .z = -1.1401 },
        .{ .x = 0.0, .y = -1.4325, .z = -1.1401 },
    };
    const charges = [_]f64{ 8.0, 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 8, 1, 1 };

    const shells = try buildSto3gShells(alloc, &atomic_numbers, &positions);
    defer alloc.free(shells);

    var vib_result = try vibrational.vibrationalAnalysis(
        alloc,
        shells,
        &positions,
        &charges,
        &atomic_numbers,
        10,
        .{
            .displacement = 0.001,
            .print_progress = false,
            .use_integer_masses = true,
            .scf_params = .{
                .max_iter = 100,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
            },
        },
    );
    defer vib_result.deinit(alloc);

    var scf_result = try gto_scf.runGeneralRhfScf(
        alloc,
        shells,
        &positions,
        &charges,
        10,
        .{
            .max_iter = 100,
            .energy_threshold = 1e-10,
            .density_threshold = 1e-8,
        },
    );
    defer scf_result.deinit(alloc);

    const thermo = vibrational.computeThermo(
        &vib_result,
        &positions,
        &atomic_numbers,
        scf_result.total_energy,
        298.15,
        101325,
        2,
        true,
    );

    try testing.expectApproxEqAbs(0.001416, thermo.e_trans, 1e-5);
    try testing.expectApproxEqAbs(5.515e-5, thermo.s_trans, 1e-6);
    try testing.expect(!thermo.is_linear);
    try testing.expectApproxEqAbs(0.001416, thermo.e_rot, 1e-5);
    try testing.expectApproxEqAbs(1.701e-5, thermo.s_rot, 1e-6);
    try testing.expectApproxEqAbs(698.1, thermo.rot_const_ghz[0], 10.0);
    try testing.expectApproxEqAbs(436.2, thermo.rot_const_ghz[1], 10.0);
    try testing.expectApproxEqAbs(268.5, thermo.rot_const_ghz[2], 10.0);
    try testing.expectApproxEqAbs(7.216e-5, thermo.s_tot, 2e-6);
    try testing.expectApproxEqAbs(-74.9377, thermo.h_tot, 0.03);
    try testing.expectApproxEqAbs(-74.9593, thermo.g_tot, 0.03);
}

test "H2 STO-3G thermodynamic properties" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.3459, .y = 0.0, .z = 0.0 },
    };
    const charges = [_]f64{ 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 1, 1 };

    const shells = try buildSto3gShells(alloc, &atomic_numbers, &positions);
    defer alloc.free(shells);

    var vib_result = try vibrational.vibrationalAnalysis(
        alloc,
        shells,
        &positions,
        &charges,
        &atomic_numbers,
        2,
        .{
            .displacement = 0.001,
            .print_progress = false,
            .use_integer_masses = true,
            .scf_params = .{
                .max_iter = 100,
                .energy_threshold = 1e-10,
                .density_threshold = 1e-8,
            },
        },
    );
    defer vib_result.deinit(alloc);

    var scf_result = try gto_scf.runGeneralRhfScf(
        alloc,
        shells,
        &positions,
        &charges,
        2,
        .{
            .max_iter = 100,
            .energy_threshold = 1e-10,
            .density_threshold = 1e-8,
        },
    );
    defer scf_result.deinit(alloc);

    const thermo = vibrational.computeThermo(
        &vib_result,
        &positions,
        &atomic_numbers,
        scf_result.total_energy,
        298.15,
        101325,
        2,
        true,
    );

    try testing.expect(thermo.is_linear);
    try testing.expectApproxEqAbs(0.001416, thermo.e_trans, 1e-5);
    try testing.expectApproxEqAbs(0.000944, thermo.e_rot, 1e-5);
    try testing.expectApproxEqAbs(vib_result.zpve, thermo.zpve, 1e-8);
}
