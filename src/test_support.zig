const std = @import("std");
const builtin = @import("builtin");

pub fn requireFile(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("  [SKIP] file not found: {s}\n", .{path});
            return error.SkipZigTest;
        }
        return err;
    };
}

pub fn skipOnGithubActionsLinux(reason: []const u8) !void {
    if (builtin.target.os.tag == .linux and std.posix.getenv("GITHUB_ACTIONS") != null) {
        std.debug.print("  [SKIP] {s}\n", .{reason});
        return error.SkipZigTest;
    }
}
