const std = @import("std");
const dft = @import("dft_zig");

/// Entry point: parse config and run workflow.
pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;
    const logger = dft.runtime_logging.stderr(io, .info);

    var args_iter = try init.minimal.args.iterateAllocator(alloc);
    defer args_iter.deinit();
    _ = args_iter.next(); // program name
    const config_path_opt = args_iter.next();
    if (config_path_opt == null) {
        try printUsage(io);
        return;
    }
    const config_path = config_path_opt.?;

    var cfg = try dft.config.load(alloc, io, config_path);
    defer cfg.deinit(alloc);

    // Pre-flight file existence checks — intentional TOCTOU: we want to report
    // all missing files up front rather than failing one-by-one during loading.
    const cwd = std.Io.Dir.cwd();
    var has_file_errors = false;
    if (cwd.statFile(io, cfg.xyz_path, .{})) |_| {} else |_| {
        try logger.print(.err, "[ERROR] [root.xyz] file not found: \"{s}\"\n", .{cfg.xyz_path});
        has_file_errors = true;
    }
    for (cfg.pseudopotentials) |pp| {
        if (cwd.statFile(io, pp.path, .{})) |_| {} else |_| {
            try logger.print(
                .err,
                "[ERROR] [pseudopotential.path] file not found: \"{s}\" (element {s})\n",
                .{ pp.path, pp.element },
            );
            has_file_errors = true;
        }
    }

    // Semantic validation
    var validation = try cfg.validate(alloc);
    defer validation.deinit();

    try reportValidationIssues(logger, validation.issues);

    if (has_file_errors or validation.hasErrors()) {
        try logger.print(.err, "Config validation failed. Aborting.\n", .{});
        return;
    }

    var atoms = try dft.xyz.load(alloc, io, cfg.xyz_path);
    defer atoms.deinit(alloc);

    try dft.xyz.validateInCell(atoms.items, cfg.cell);

    try dft.dft.run(alloc, io, cfg, atoms.items);
}

fn reportValidationIssues(
    logger: dft.runtime_logging.Logger,
    issues: []const dft.config.ValidationIssue,
) !void {
    for (issues) |issue| {
        const level: dft.runtime_logging.Level = switch (issue.severity) {
            .err => .err,
            .warning => .warn,
            .hint => .info,
        };
        const prefix: []const u8 = switch (issue.severity) {
            .err => "ERROR",
            .warning => "WARNING",
            .hint => "HINT",
        };
        const field_sep: []const u8 = if (issue.field.len > 0) "." else "";
        try logger.print(
            level,
            "[{s}] [{s}{s}{s}] {s}\n",
            .{ prefix, issue.section, field_sep, issue.field, issue.message },
        );
    }
}

/// Print CLI usage.
fn printUsage(io: std.Io) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print(
        "Usage: dft_zig <config.toml>\n" ++
            "Example: zig build run -- examples/graphene.toml\n",
        .{},
    );
    try stdout.flush();
}
