const runtime_logging = @import("../../runtime/logging.zig");

pub fn debug(enabled: bool, comptime fmt: []const u8, args: anytype) void {
    if (!enabled) return;
    runtime_logging.debugPrint(.debug, .debug, fmt, args);
}
