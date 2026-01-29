const std = @import("std");
const dft = @import("dft_zig");

/// Entry point: parse config and run workflow.
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const config_path = args[1];
    var cfg = try dft.config.load(alloc, config_path);
    defer cfg.deinit(alloc);

    var atoms = try dft.xyz.load(alloc, cfg.xyz_path);
    defer atoms.deinit(alloc);

    try dft.xyz.validateInCell(atoms.items, cfg.cell);

    try dft.dft.run(alloc, cfg, atoms.items);
}

/// Print CLI usage.
fn printUsage() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print(
        "Usage: dft_zig <config.toml>\n" ++
            "Example: zig build run -- examples/graphene.toml\n",
        .{},
    );
    try stdout.flush();
}
