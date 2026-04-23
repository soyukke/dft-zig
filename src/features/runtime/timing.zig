const std = @import("std");

/// Timing information for each high-level workflow step.
pub const Timing = struct {
    total_ns: u64 = 0,
    setup_ns: u64 = 0,
    relax_ns: u64 = 0,
    scf_ns: u64 = 0,
    band_ns: u64 = 0,
    // CPU time (user + system) in microseconds
    cpu_start_us: i64 = 0,
    cpu_end_us: i64 = 0,

    pub fn to_seconds(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
    }

    pub fn cpu_seconds(self: Timing) f64 {
        const diff = self.cpu_end_us - self.cpu_start_us;
        if (diff <= 0) return 0.0;
        return @as(f64, @floatFromInt(diff)) / 1_000_000.0;
    }

    /// Get current CPU time (user + system) in microseconds using getrusage.
    pub fn get_cpu_time_us() i64 {
        const rusage = std.posix.getrusage(std.posix.rusage.SELF);
        const user_us = @as(i64, rusage.utime.sec) * 1_000_000 + rusage.utime.usec;
        const sys_us = @as(i64, rusage.stime.sec) * 1_000_000 + rusage.stime.usec;
        return user_us + sys_us;
    }
};
