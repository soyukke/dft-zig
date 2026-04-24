const std = @import("std");
const builtin = @import("builtin");

fn print_stderr(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    try out.print(fmt, args);
    try out.flush();
}

pub fn require_file(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            try print_stderr(io, "  [SKIP] file not found: {s}\n", .{path});
            return error.SkipZigTest;
        }
        return err;
    };
}

pub fn skip_on_github_actions_linux(reason: []const u8) !void {
    if (builtin.target.os.tag == .linux and std.posix.getenv("GITHUB_ACTIONS") != null) {
        try print_stderr(std.testing.io, "  [SKIP] {s}\n", .{reason});
        return error.SkipZigTest;
    }
}
