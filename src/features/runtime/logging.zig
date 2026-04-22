const std = @import("std");

pub const Level = enum(u8) {
    err = 0,
    warn = 1,
    info = 2,
    debug = 3,
};

pub fn name(level: Level) []const u8 {
    return switch (level) {
        .err => "error",
        .warn => "warn",
        .info => "info",
        .debug => "debug",
    };
}

pub fn parseLevel(value: []const u8) !Level {
    if (std.mem.eql(u8, value, "error")) return .err;
    if (std.mem.eql(u8, value, "warn") or
        std.mem.eql(u8, value, "warning") or
        std.mem.eql(u8, value, "quiet")) return .warn;
    if (std.mem.eql(u8, value, "info") or std.mem.eql(u8, value, "normal")) return .info;
    if (std.mem.eql(u8, value, "debug") or std.mem.eql(u8, value, "verbose")) return .debug;
    return error.InvalidLogLevel;
}

pub fn enabled(max_level: Level, level: Level) bool {
    return @intFromEnum(level) <= @intFromEnum(max_level);
}

pub const Logger = struct {
    io: std.Io,
    max_level: Level,

    pub fn init(io: std.Io, max_level: Level) Logger {
        return .{ .io = io, .max_level = max_level };
    }

    pub fn print(self: Logger, level: Level, comptime fmt: []const u8, args: anytype) !void {
        if (!enabled(self.max_level, level)) return;
        var buffer: [1024]u8 = undefined;
        var writer = std.Io.File.stderr().writer(self.io, &buffer);
        const out = &writer.interface;
        try out.print(fmt, args);
        try out.flush();
    }

    pub fn writeAll(self: Logger, level: Level, msg: []const u8) !void {
        if (!enabled(self.max_level, level)) return;
        var buffer: [1024]u8 = undefined;
        var writer = std.Io.File.stderr().writer(self.io, &buffer);
        const out = &writer.interface;
        try out.writeAll(msg);
        try out.flush();
    }
};

pub fn stderr(io: std.Io, max_level: Level) Logger {
    return Logger.init(io, max_level);
}

pub fn debugPrint(max_level: Level, level: Level, comptime fmt: []const u8, args: anytype) void {
    if (!enabled(max_level, level)) return;
    std.debug.print(fmt, args);
}

test "parseLevel supports common aliases" {
    try std.testing.expectEqual(.warn, try parseLevel("warn"));
    try std.testing.expectEqual(.warn, try parseLevel("quiet"));
    try std.testing.expectEqual(.info, try parseLevel("normal"));
    try std.testing.expectEqual(.debug, try parseLevel("verbose"));
}

test "enabled gates by max level" {
    try std.testing.expect(enabled(.info, .warn));
    try std.testing.expect(enabled(.debug, .debug));
    try std.testing.expect(!enabled(.warn, .info));
}
