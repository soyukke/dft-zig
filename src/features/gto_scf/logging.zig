const runtime_logging = @import("../runtime/logging.zig");

pub fn verbose(enabled: bool, comptime fmt: []const u8, args: anytype) void {
    if (!enabled) return;
    runtime_logging.debug_print(.debug, .debug, fmt, args);
}

pub fn progress(enabled: bool, comptime fmt: []const u8, args: anytype) void {
    if (!enabled) return;
    runtime_logging.debug_print(.info, .info, fmt, args);
}
