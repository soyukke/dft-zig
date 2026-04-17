//! Thread pool abstraction for parallel computation.
//!
//! Wraps `std.Io.Group` which already provides a shared worker pool
//! inside the `std.Io` runtime (e.g. `std.Io.Threaded` maintains a
//! reusable worker pool spawned lazily up to `async_limit`).
//!
//! This keeps API compatibility with the previous `std.Thread.Pool`-based
//! implementation while delegating scheduling to the stdlib.

const std = @import("std");

/// Thread pool wrapper carrying an `io` reference. Scheduling is handled
/// by the underlying `std.Io.Group` implementation so worker threads are
/// reused across `parallelFor` calls, matching pre-0.16 performance.
pub const ThreadPool = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    num_threads: usize,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, num_threads: usize) !ThreadPool {
        const actual_threads = if (num_threads == 0)
            std.Thread.getCpuCount() catch 4
        else
            num_threads;
        return .{
            .io = io,
            .allocator = allocator,
            .num_threads = actual_threads,
        };
    }

    pub fn deinit(self: *ThreadPool) void {
        _ = self;
    }

    /// Execute `func(context, idx)` for each idx in [0, count) in parallel.
    /// Blocks until all tasks complete.
    pub fn parallelFor(
        self: *ThreadPool,
        count: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) void,
    ) void {
        if (count == 0) return;

        const Context = @TypeOf(context);
        const Wrapper = struct {
            fn task(ctx: Context, idx: usize) std.Io.Cancelable!void {
                func(ctx, idx);
            }
        };

        var group: std.Io.Group = .init;
        for (0..count) |i| {
            group.async(self.io, Wrapper.task, .{ context, i });
        }
        group.await(self.io) catch {};
    }

    /// Execute `func(context, idx)` for each idx in [0, count) in parallel,
    /// collecting the first error encountered. Blocks until all tasks finish.
    pub fn parallelForWithError(
        self: *ThreadPool,
        count: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) anyerror!void,
    ) !void {
        if (count == 0) return;

        const Context = @TypeOf(context);
        const Shared = struct {
            ctx: Context,
            err_flag: std.atomic.Value(bool) = .init(false),
            err: anyerror = error.Unexpected,
        };
        var shared = Shared{ .ctx = context };

        const Wrapper = struct {
            fn task(s: *Shared, idx: usize) std.Io.Cancelable!void {
                func(s.ctx, idx) catch |e| {
                    if (!s.err_flag.swap(true, .acq_rel)) s.err = e;
                };
            }
        };

        var group: std.Io.Group = .init;
        for (0..count) |i| {
            group.async(self.io, Wrapper.task, .{ &shared, i });
        }
        group.await(self.io) catch {};

        if (shared.err_flag.load(.acquire)) return shared.err;
    }
};

/// Optional thread pool reference.
pub const OptionalPool = ?*ThreadPool;

test "ThreadPool basic" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var pool = try ThreadPool.init(allocator, io, 2);
    defer pool.deinit();

    var results: [10]usize = undefined;
    @memset(&results, 0);

    const Ctx = struct { results: *[10]usize };

    pool.parallelFor(10, Ctx{ .results = &results }, struct {
        fn run(ctx: Ctx, idx: usize) void {
            ctx.results[idx] = idx * 2;
        }
    }.run);

    for (0..10) |i| try std.testing.expectEqual(results[i], i * 2);
}
