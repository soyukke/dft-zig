const std = @import("std");

pub const SpinLock = struct {
    state: std.atomic.Value(bool) = .init(false),

    pub fn lock(self: *SpinLock) void {
        var spin: u32 = 0;
        while (self.state.cmpxchgStrong(false, true, .acquire, .monotonic) != null) {
            spin += 1;
            if (spin >= 32) {
                spin = 0;
                std.Thread.yield() catch std.atomic.spinLoopHint();
            } else {
                std.atomic.spinLoopHint();
            }
        }
    }

    pub fn unlock(self: *SpinLock) void {
        self.state.store(false, .release);
    }
};
