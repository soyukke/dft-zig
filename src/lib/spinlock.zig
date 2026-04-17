const std = @import("std");

/// File-level mutex backed by POSIX pthread_mutex.
/// Provides blocking lock/unlock with kernel-managed waitqueues -
/// the same semantics as the pre-0.16 `std.Thread.Mutex` when it was
/// pthread-based. Unlike a naive spinlock, contended waiters park in
/// the kernel and free the CPU for lock holders.
pub const SpinLock = struct {
    m: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    pub fn lock(self: *SpinLock) void {
        _ = std.c.pthread_mutex_lock(&self.m);
    }

    pub fn unlock(self: *SpinLock) void {
        _ = std.c.pthread_mutex_unlock(&self.m);
    }
};
