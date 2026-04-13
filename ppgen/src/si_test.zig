//! Si pseudopotential generation and validation test.
//!
//! Si: Z=14, [Ne] 3s² 3p²
//! Valence: 3s² 3p² (4 electrons)
//! Core: 1s² 2s² 2p⁶ (10 electrons)

const std = @import("std");
const dft_zig = @import("dft_zig");
const xc_mod = dft_zig.xc;
const pseudo_parser = dft_zig.pseudopotential;

const RadialGrid = @import("radial_grid.zig").RadialGrid;
const atomic_solver = @import("atomic_solver.zig");
const schrodinger = @import("schrodinger.zig");
const tm_generator = @import("tm_generator.zig");
const kb_projector = @import("kb_projector.zig");
const upf_writer = @import("upf_writer.zig");
const pipeline = @import("pipeline.zig");

test "Si all-electron atomic SCF" {
    // Si: Z=14, 1s² 2s² 2p⁶ 3s² 3p²
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 6000, 1e-9, 80.0);
    defer grid.deinit();

    const orbitals = [_]atomic_solver.OrbitalConfig{
        .{ .n = 1, .l = 0, .occupation = 2.0 }, // 1s
        .{ .n = 2, .l = 0, .occupation = 2.0 }, // 2s
        .{ .n = 2, .l = 1, .occupation = 6.0 }, // 2p
        .{ .n = 3, .l = 0, .occupation = 2.0 }, // 3s
        .{ .n = 3, .l = 1, .occupation = 2.0 }, // 3p
    };

    var result = try atomic_solver.solve(allocator, &grid, .{
        .z = 14,
        .orbitals = &orbitals,
        .xc = .lda_pz,
    }, 300, 0.3, 1e-10);
    defer result.deinit();

    std.debug.print("\n  === Si all-electron LDA ===\n", .{});
    const labels = [_][]const u8{ "1s", "2s", "2p", "3s", "3p" };
    for (0..5) |i| {
        std.debug.print("  ε({s}) = {d:12.6} Ry\n", .{ labels[i], result.eigenvalues[i] });
    }
    std.debug.print("  E_total = {d:12.6} Ry\n", .{result.total_energy});

    // Reference: Si LDA total energy ≈ -577.9 Ry (NIST)
    // Our spin-unpolarized value will be close but not exact for open-shell
    try std.testing.expect(result.total_energy < -570.0);
    try std.testing.expect(result.total_energy > -590.0);

    // 3s eigenvalue should be around -1.0 to -0.5 Ry
    try std.testing.expect(result.eigenvalues[3] < 0);
    try std.testing.expect(result.eigenvalues[3] > -2.0);

    // 3p eigenvalue should be around -0.4 to -0.1 Ry
    try std.testing.expect(result.eigenvalues[4] < 0);
    try std.testing.expect(result.eigenvalues[4] > -1.0);
}

test "Si pseudopotential generation: valence only" {
    const allocator = std.testing.allocator;

    // All-electron orbitals: 1s² 2s² 2p⁶ 3s² 3p²
    const all_orbs = [_]pipeline.OrbitalDef{
        .{ .n = 1, .l = 0, .occupation = 2.0 },
        .{ .n = 2, .l = 0, .occupation = 2.0 },
        .{ .n = 2, .l = 1, .occupation = 6.0 },
        .{ .n = 3, .l = 0, .occupation = 2.0 },
        .{ .n = 3, .l = 1, .occupation = 2.0 },
    };

    // Valence channels only: 3s, 3p
    const val_channels = [_]pipeline.ChannelConfig{
        .{ .n = 3, .l = 0, .occupation = 2.0, .rc = 1.8 }, // 3s
        .{ .n = 3, .l = 1, .occupation = 2.0, .rc = 2.0 }, // 3p
    };

    var buf: [2 * 1024 * 1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);

    try pipeline.generatePseudopotential(allocator, .{
        .z = 14,
        .element = "Si",
        .xc = .lda_pz,
        .all_orbitals = &all_orbs,
        .valence_channels = &val_channels,
        .l_local = 1, // p as local, s as nonlocal
    }, fbs.writer());

    const output = fbs.getWritten();
    std.debug.print("\n  Si UPF size: {d} bytes\n", .{output.len});

    // z_valence = 4 (3s² + 3p²)
    try std.testing.expect(std.mem.indexOf(u8, output, "z_valence=\"4.0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "element=\"Si\"") != null);

    // Write and parse back
    const tmp_path = "/tmp/Si_ppgen_val.upf";
    {
        const file = try std.fs.cwd().createFile(tmp_path, .{});
        defer file.close();
        try file.writeAll(output);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var parsed = try pseudo_parser.load(allocator, .{
        .element = "Si",
        .path = tmp_path,
        .format = .upf,
    });
    defer parsed.deinit(allocator);

    const upf = parsed.upf.?;
    std.debug.print("  Parsed: mesh={d}, n_beta={d}, dij_len={d}\n", .{ upf.r.len, upf.beta.len, upf.dij.len });

    // 1 KB projector (s-channel, since p is local)
    try std.testing.expectEqual(@as(usize, 1), upf.beta.len);
    try std.testing.expectEqual(@as(i32, 0), upf.beta[0].l.?);
    try std.testing.expectApproxEqAbs(4.0, parsed.header.z_valence.?, 0.01);

    std.debug.print("  z_valence = {d:.1}\n", .{parsed.header.z_valence.?});
    std.debug.print("  l_max = {d}\n", .{parsed.header.l_max.?});
    std.debug.print("  beta[0]: l={d}, D={d:.6} Ry\n", .{ upf.beta[0].l.?, upf.dij[0] });
}
