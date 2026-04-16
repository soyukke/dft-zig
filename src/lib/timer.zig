const std = @import("std");
const builtin = @import("builtin");

pub const Timer = struct {
    start_ns: u64,

    pub const Error = error{TimerUnsupported};

    pub fn start() Error!Timer {
        return .{ .start_ns = nowNs() };
    }

    pub fn read(self: *Timer) u64 {
        return nowNs() -% self.start_ns;
    }

    pub fn lap(self: *Timer) u64 {
        const now = nowNs();
        const elapsed = now -% self.start_ns;
        self.start_ns = now;
        return elapsed;
    }

    pub fn reset(self: *Timer) void {
        self.start_ns = nowNs();
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    const sec: u64 = @intCast(ts.sec);
    const nsec: u64 = @intCast(ts.nsec);
    return sec * std.time.ns_per_s + nsec;
}
