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

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: ppgen <config.toml>\n", .{});
        return;
    }

    std.debug.print("ppgen: pseudopotential generator (Phase 1: atomic solver)\n", .{});
    std.debug.print("Config file: {s}\n", .{args[1]});
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
}
