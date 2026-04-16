//! Thread pool for parallel computation
//!
//! Provides a reusable thread pool that can be passed through
//! the computation pipeline (SCF -> LOBPCG -> op.apply).

const std = @import("std");

/// Thread pool wrapper with convenient parallel execution methods
pub const ThreadPool = struct {
    inner: *std.Thread.Pool,
    num_threads: usize,
    allocator: std.mem.Allocator,

    /// Initialize thread pool with specified number of threads
    /// If num_threads is 0, uses CPU count
    pub fn init(allocator: std.mem.Allocator, num_threads: usize) !ThreadPool {
        const actual_threads = if (num_threads == 0)
            std.Thread.getCpuCount() catch 4
        else
            num_threads;

        const pool = try allocator.create(std.Thread.Pool);
        errdefer allocator.destroy(pool);

        try pool.init(.{
            .allocator = allocator,
            .n_jobs = @intCast(actual_threads),
        });

        return .{
            .inner = pool,
            .num_threads = actual_threads,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadPool) void {
        self.inner.deinit();
        self.allocator.destroy(self.inner);
    }

    /// Execute a function for each index in range [0, count) in parallel
    /// The function receives the index as parameter
    pub fn parallelFor(
        self: *ThreadPool,
        count: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) void,
    ) void {
        if (count == 0) return;

        const Context = @TypeOf(context);
        const Wrapper = struct {
            ctx: Context,
            func_ptr: *const fn (Context, usize) void,

            fn run(wrapper: *@This(), idx: usize) void {
                wrapper.func_ptr(wrapper.ctx, idx);
            }
        };

        var wrapper = Wrapper{ .ctx = context, .func_ptr = func };

        // Use WaitGroup to wait for all tasks
        var wg = std.Thread.WaitGroup{};
        for (0..count) |i| {
            wg.start();
            self.inner.spawn(struct {
                fn task(w: *Wrapper, wg_ptr: *std.Thread.WaitGroup, idx: usize) void {
                    defer wg_ptr.finish();
                    w.run(idx);
                }
            }.task, .{ &wrapper, &wg, i }) catch {
                wg.finish();
                continue;
            };
        }
        wg.wait();
    }

    /// Execute a function for each index, allowing errors
    /// Returns first error encountered or success
    pub fn parallelForWithError(
        self: *ThreadPool,
        count: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) anyerror!void,
    ) !void {
        if (count == 0) return;

        const Context = @TypeOf(context);
        const Wrapper = struct {
            ctx: Context,
            err: ?anyerror = null,
            err_mutex: std.Io.Mutex = .init,

            fn run(wrapper: *@This(), idx: usize) void {
                func(wrapper.ctx, idx) catch |e| {
                    wrapper.err_mutex.lock();
                    defer wrapper.err_mutex.unlock();
                    if (wrapper.err == null) {
                        wrapper.err = e;
                    }
                };
            }
        };

        var wrapper = Wrapper{ .ctx = context };

        var wg = std.Thread.WaitGroup{};
        for (0..count) |i| {
            wg.start();
            self.inner.spawn(struct {
                fn task(w: *Wrapper, wg_ptr: *std.Thread.WaitGroup, idx: usize) void {
                    defer wg_ptr.finish();
                    w.run(idx);
                }
            }.task, .{ &wrapper, &wg, i }) catch {
                wg.finish();
                continue;
            };
        }
        wg.wait();

        if (wrapper.err) |e| return e;
    }
};

/// Optional thread pool reference
/// Allows passing thread pool as optional parameter
pub const OptionalPool = ?*ThreadPool;

// ============== Tests ==============

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
