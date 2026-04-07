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

    // Pre-flight file existence checks — intentional TOCTOU: we want to report
    // all missing files up front rather than failing one-by-one during loading.
    var has_file_errors = false;
    if (std.fs.cwd().statFile(cfg.xyz_path)) |_| {} else |_| {
        std.debug.print("[ERROR] [root.xyz] file not found: \"{s}\"\n", .{cfg.xyz_path});
        has_file_errors = true;
    }
    for (cfg.pseudopotentials) |pp| {
        if (std.fs.cwd().statFile(pp.path)) |_| {} else |_| {
            std.debug.print("[ERROR] [pseudopotential.path] file not found: \"{s}\" (element {s})\n", .{ pp.path, pp.element });
            has_file_errors = true;
        }
    }

    // Semantic validation
    var validation = try cfg.validate(alloc);
    defer validation.deinit();

    for (validation.issues) |issue| {
        const prefix: []const u8 = switch (issue.severity) {
            .err => "ERROR",
            .warning => "WARNING",
            .hint => "HINT",
        };
        const field_sep: []const u8 = if (issue.field.len > 0) "." else "";
        std.debug.print("[{s}] [{s}{s}{s}] {s}\n", .{ prefix, issue.section, field_sep, issue.field, issue.message });
    }

    if (has_file_errors or validation.hasErrors()) {
        std.debug.print("Config validation failed. Aborting.\n", .{});
        return;
    }

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
