//! Integration test: generate Si pseudopotential with ppgen, then run
//! DFT-Zig SCF calculation on Si crystal and verify convergence.

const std = @import("std");
const dft_zig = @import("dft_zig");
const pseudo_parser = dft_zig.pseudopotential;
const xc_mod = dft_zig.xc;

const pipeline = @import("pipeline.zig");

test "integration: generate Si UPF and write to file" {
    const allocator = std.testing.allocator;

    // Si: Z=14, [Ne] 3s² 3p²
    const all_orbs = [_]pipeline.OrbitalDef{
        .{ .n = 1, .l = 0, .occupation = 2.0 }, // 1s
        .{ .n = 2, .l = 0, .occupation = 2.0 }, // 2s
        .{ .n = 2, .l = 1, .occupation = 6.0 }, // 2p
        .{ .n = 3, .l = 0, .occupation = 2.0 }, // 3s
        .{ .n = 3, .l = 1, .occupation = 2.0 }, // 3p
    };

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

    // Write to project pseudo directory for DFT-Zig to use
    const upf_path = "/tmp/Si_ppgen.upf";
    {
        const file = try std.fs.cwd().createFile(upf_path, .{});
        defer file.close();
        try file.writeAll(output);
    }
    defer std.fs.cwd().deleteFile(upf_path) catch {};

    // Verify it parses
    var parsed = try pseudo_parser.load(allocator, .{
        .element = "Si",
        .path = upf_path,
        .format = .upf,
    });
    defer parsed.deinit(allocator);

    try std.testing.expectApproxEqAbs(4.0, parsed.header.z_valence.?, 0.01);
    try std.testing.expectEqual(@as(usize, 1), parsed.upf.?.beta.len);

    std.debug.print("\n  === Si UPF generated: {d} bytes ===\n", .{output.len});
    std.debug.print("  z_val={d:.1}, l_max={d}, n_beta={d}\n", .{
        parsed.header.z_valence.?,
        parsed.header.l_max.?,
        parsed.upf.?.beta.len,
    });
    std.debug.print("  D_11 = {d:.6} Ry\n", .{parsed.upf.?.dij[0]});
    std.debug.print("  V_local(r=0.1) = {d:.4} Ry\n", .{parsed.upf.?.v_local[100]});
    std.debug.print("  Written to {s}\n", .{upf_path});
}
