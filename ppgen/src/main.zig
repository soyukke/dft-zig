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

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: ppgen <output.upf>\n", .{});
        std.debug.print("  Generates Si LDA norm-conserving pseudopotential.\n", .{});
        return;
    }

    const output_path = args[1];

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
    var fbs = std.io.fixedBufferStream(&buf);

    try pipeline.generatePseudopotential(allocator, .{
        .z = 14,
        .element = "Si",
        .xc = .lda_pz,
        .all_orbitals = &all_orbs,
        .valence_channels = &val_channels,
        .l_local = 1,
    }, fbs.writer());

    const output = fbs.getWritten();
    const file = try std.fs.cwd().createFile(output_path, .{});
    defer file.close();
    try file.writeAll(output);

    std.debug.print("Written: {s}\n", .{output_path});
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
