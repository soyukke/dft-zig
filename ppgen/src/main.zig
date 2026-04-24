const std = @import("std");
pub const radial_grid = @import("radial_grid.zig");
pub const schrodinger = @import("schrodinger.zig");
pub const atomic_solver = @import("atomic_solver.zig");
pub const diagnostics = @import("diagnostics.zig");
pub const tm_generator = @import("tm_generator.zig");
pub const kb_projector = @import("kb_projector.zig");
pub const upf_writer = @import("upf_writer.zig");
pub const pipeline = @import("pipeline.zig");
pub const si_test = @import("si_test.zig");
pub const integration_test = @import("integration_test.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    var stderr_buffer: [256]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;

    var args_iter = try init.minimal.args.iterateAllocator(allocator);
    defer args_iter.deinit();

    _ = args_iter.next();
    const output_path = args_iter.next() orelse {
        try stderr.writeAll("Usage: ppgen <output.upf>\n");
        try stderr.writeAll("  Generates Si LDA norm-conserving pseudopotential.\n");
        try stderr.flush();
        return;
    };

    // Si: Z=14, [Ne] 3s² 3p²
    const all_orbs = [_]pipeline.OrbitalDef{
        .{ .n = 1, .l = 0, .occupation = 2.0 },
        .{ .n = 2, .l = 0, .occupation = 2.0 },
        .{ .n = 2, .l = 1, .occupation = 6.0 },
        .{ .n = 3, .l = 0, .occupation = 2.0 },
        .{ .n = 3, .l = 1, .occupation = 2.0 },
    };

    const val_channels = [_]pipeline.ChannelConfig{
        .{ .n = 3, .l = 0, .occupation = 2.0, .rc = 1.8 },
        .{ .n = 3, .l = 1, .occupation = 2.0, .rc = 2.0 },
    };

    // Generate into memory buffer, then write to file
    var buf: [2 * 1024 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);

    try pipeline.generate_pseudopotential(allocator, .{
        .z = 14,
        .element = "Si",
        .xc = .lda_pz,
        .all_orbitals = &all_orbs,
        .valence_channels = &val_channels,
        .l_local = 1,
    }, &writer);

    const output = writer.buffered();
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, output_path, .{});
    defer file.close(io);

    try file.writeStreamingAll(io, output);

    try stdout.print("Written: {s}\n", .{output_path});
    try stdout.flush();
}

test {
    _ = radial_grid;
    _ = schrodinger;
    _ = atomic_solver;
    _ = diagnostics;
    _ = tm_generator;
    _ = kb_projector;
    _ = upf_writer;
    _ = pipeline;
    _ = si_test;
    _ = integration_test;
}
