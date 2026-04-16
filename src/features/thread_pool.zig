//! Thread pool for parallel computation
//!
//! Provides a reusable thread pool that can be passed through
//! the computation pipeline (SCF -> LOBPCG -> op.apply).
//!
//! 0.16: std.Thread.Pool was removed. This implementation uses raw
//! std.Thread.spawn with std.atomic for work distribution and
//! std.Thread.Semaphore-like coordination via atomic counters.

const std = @import("std");

/// Thread pool wrapper with convenient parallel execution methods.
/// Threads are spawned on demand in each parallelFor call.
pub const ThreadPool = struct {
    num_threads: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_threads: usize) !ThreadPool {
        const actual_threads = if (num_threads == 0)
            std.Thread.getCpuCount() catch 4
        else
            num_threads;
        return .{
            .num_threads = actual_threads,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadPool) void {
        _ = self;
    }

    /// Execute a function for each index in range [0, count) in parallel.
    pub fn parallelFor(
        self: *ThreadPool,
        count: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) void,
    ) void {
        if (count == 0) return;

        const Context = @TypeOf(context);
        const Shared = struct {
            next: std.atomic.Value(usize) = .init(0),
            total: usize,
            ctx: Context,
        };
        var shared = Shared{ .total = count, .ctx = context };

        const n_threads = @min(self.num_threads, count);
        var threads = self.allocator.alloc(std.Thread, n_threads) catch {
            // Fallback: run serially
            var i: usize = 0;
            while (i < count) : (i += 1) func(context, i);
            return;
        };
        defer self.allocator.free(threads);

        const Worker = struct {
            fn run(s: *Shared) void {
                while (true) {
                    const idx = s.next.fetchAdd(1, .acq_rel);
                    if (idx >= s.total) break;
                    func(s.ctx, idx);
                }
            }
        };

        var spawned: usize = 0;
        for (threads) |*t| {
            t.* = std.Thread.spawn(.{}, Worker.run, .{&shared}) catch break;
            spawned += 1;
        }
        // Run remaining work on this thread too
        Worker.run(&shared);
        for (threads[0..spawned]) |t| t.join();
    }

    /// Execute a function for each index, allowing errors.
    pub fn parallelForWithError(
        self: *ThreadPool,
        count: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) anyerror!void,
    ) !void {
        if (count == 0) return;

        const Context = @TypeOf(context);
        const Shared = struct {
            next: std.atomic.Value(usize) = .init(0),
            total: usize,
            ctx: Context,
            err_flag: std.atomic.Value(bool) = .init(false),
            err: ?anyerror = null,
        };
        var shared = Shared{ .total = count, .ctx = context };

        const n_threads = @min(self.num_threads, count);
        var threads = try self.allocator.alloc(std.Thread, n_threads);
        defer self.allocator.free(threads);

        const Worker = struct {
            fn run(s: *Shared) void {
                while (true) {
                    const idx = s.next.fetchAdd(1, .acq_rel);
                    if (idx >= s.total) break;
                    func(s.ctx, idx) catch |e| {
                        if (!s.err_flag.swap(true, .acq_rel)) {
                            s.err = e;
                        }
                    };
                }
            }
        };

        var spawned: usize = 0;
        for (threads) |*t| {
            t.* = std.Thread.spawn(.{}, Worker.run, .{&shared}) catch break;
            spawned += 1;
        }
        Worker.run(&shared);
        for (threads[0..spawned]) |t| t.join();

        if (shared.err) |e| return e;
    }
};

pub const OptionalPool = ?*ThreadPool;

test "ThreadPool basic" {
    const allocator = std.testing.allocator;

    var pool = try ThreadPool.init(allocator, 2);
    defer pool.deinit();

    var results: [10]usize = undefined;
    @memset(&results, 0);

    const Ctx = struct {
        results: *[10]usize,
    };

    pool.parallelFor(10, Ctx{ .results = &results }, struct {
        fn run(ctx: Ctx, idx: usize) void {
            ctx.results[idx] = idx * 2;
        }
    }.run);

    for (0..10) |i| {
        try std.testing.expectEqual(results[i], i * 2);
    }
}
