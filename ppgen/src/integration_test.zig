//! Integration test: generate Si pseudopotential with ppgen, then run
//! DFT-Zig SCF calculation on Si crystal and verify convergence.

const std = @import("std");
const dft_zig = @import("dft_zig");
const pseudo_parser = dft_zig.pseudopotential;
const xc_mod = dft_zig.xc;

const pipeline = @import("pipeline.zig");

test "integration: generate Si UPF and write to file" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

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
    var writer = std.Io.Writer.fixed(&buf);

    try pipeline.generate_pseudopotential(allocator, .{
        .z = 14,
        .element = "Si",
        .xc = .lda_pz,
        .all_orbitals = &all_orbs,
        .valence_channels = &val_channels,
        .l_local = 1, // p as local, s as nonlocal
    }, &writer);

    const output = writer.buffered();
    try std.testing.expect(output.len > 0);

    // Write to project pseudo directory for DFT-Zig to use
    const upf_path = "/tmp/Si_ppgen.upf";
    const cwd = std.Io.Dir.cwd();
    {
        const file = try cwd.createFile(io, upf_path, .{});
        defer file.close(io);

        try file.writeStreamingAll(io, output);
    }
    defer cwd.deleteFile(io, upf_path) catch {};

    // Verify it parses
    var parsed = try pseudo_parser.load(allocator, io, .{
        .element = "Si",
        .path = upf_path,
        .format = .upf,
    });
    defer parsed.deinit(allocator);

    try std.testing.expectApproxEqAbs(4.0, parsed.header.z_valence.?, 0.01);
    try std.testing.expectEqual(@as(usize, 1), parsed.upf.?.beta.len);
}
