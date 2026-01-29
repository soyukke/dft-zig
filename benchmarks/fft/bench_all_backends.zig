//! Comprehensive 3D FFT Backend Benchmark
//!
//! Compares all available 3D FFT backends:
//! - zig (sequential)
//! - zig_parallel
//! - zig_transpose
//! - zig_comptime24 (24^3 only)
//! - vdsp (macOS Accelerate, power-of-2 only)
//! - rfft3d (real FFT)
//!
//! Usage:
//!   zig build bench-fft-all -Doptimize=ReleaseFast

const std = @import("std");
const builtin = @import("builtin");
const fft_lib = @import("fft_lib");
const Complex = fft_lib.Complex;

const WARMUP_ITERS = 5;

const BenchResult = struct {
    name: []const u8,
    time_per_iter_ms: f64,
    throughput_mpts: f64,
};

fn benchmarkPlan3d(allocator: std.mem.Allocator, grid: usize, iterations: usize) !BenchResult {
    const n = grid * grid * grid;
    const data = try allocator.alloc(Complex, n);
    defer allocator.free(data);
    const data_copy = try allocator.alloc(Complex, n);
    defer allocator.free(data_copy);

    for (data, 0..) |*c, i| {
        c.* = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
    }

    var plan = try fft_lib.Plan3d.init(allocator, grid, grid, grid);
    defer plan.deinit();

    // Warmup
    for (0..WARMUP_ITERS) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }

    // Benchmark
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const time_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const throughput = @as(f64, @floatFromInt(n)) / time_per_iter / 1000.0;

    return .{
        .name = "zig (seq)",
        .time_per_iter_ms = time_per_iter,
        .throughput_mpts = throughput,
    };
}

fn benchmarkParallel(allocator: std.mem.Allocator, grid: usize, iterations: usize) !BenchResult {
    const n = grid * grid * grid;
    const data = try allocator.alloc(Complex, n);
    defer allocator.free(data);
    const data_copy = try allocator.alloc(Complex, n);
    defer allocator.free(data_copy);

    for (data, 0..) |*c, i| {
        c.* = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
    }

    var plan = try fft_lib.parallel_fft.ParallelPlan3d.init(allocator, grid, grid, grid);
    defer plan.deinit();

    // Warmup
    for (0..WARMUP_ITERS) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }

    // Benchmark
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const time_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const throughput = @as(f64, @floatFromInt(n)) / time_per_iter / 1000.0;

    return .{
        .name = "zig_parallel",
        .time_per_iter_ms = time_per_iter,
        .throughput_mpts = throughput,
    };
}

fn benchmarkTranspose(allocator: std.mem.Allocator, grid: usize, iterations: usize) !BenchResult {
    const n = grid * grid * grid;
    const data = try allocator.alloc(Complex, n);
    defer allocator.free(data);
    const data_copy = try allocator.alloc(Complex, n);
    defer allocator.free(data_copy);

    for (data, 0..) |*c, i| {
        c.* = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
    }

    var plan = try fft_lib.parallel_fft_transpose.TransposePlan3d.init(allocator, grid, grid, grid);
    defer plan.deinit();

    // Warmup
    for (0..WARMUP_ITERS) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }

    // Benchmark
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const time_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const throughput = @as(f64, @floatFromInt(n)) / time_per_iter / 1000.0;

    return .{
        .name = "zig_transpose",
        .time_per_iter_ms = time_per_iter,
        .throughput_mpts = throughput,
    };
}

fn benchmarkComptime24(allocator: std.mem.Allocator, iterations: usize) !BenchResult {
    const grid: usize = 24;
    const n = grid * grid * grid;
    const data = try allocator.alloc(Complex, n);
    defer allocator.free(data);
    const data_copy = try allocator.alloc(Complex, n);
    defer allocator.free(data_copy);

    for (data, 0..) |*c, i| {
        c.* = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
    }

    var plan = try fft_lib.parallel_fft24.ParallelPlan3d24.init(allocator, grid, grid, grid);
    defer plan.deinit();

    // Warmup
    for (0..WARMUP_ITERS) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }

    // Benchmark
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const time_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const throughput = @as(f64, @floatFromInt(n)) / time_per_iter / 1000.0;

    return .{
        .name = "zig_comptime24",
        .time_per_iter_ms = time_per_iter,
        .throughput_mpts = throughput,
    };
}

fn benchmarkVdsp(allocator: std.mem.Allocator, grid: usize, iterations: usize) !BenchResult {
    const n = grid * grid * grid;
    const data = try allocator.alloc(Complex, n);
    defer allocator.free(data);
    const data_copy = try allocator.alloc(Complex, n);
    defer allocator.free(data_copy);

    for (data, 0..) |*c, i| {
        c.* = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
    }

    var plan = try fft_lib.vdsp_fft.VdspPlan3d.init(allocator, grid, grid, grid);
    defer plan.deinit();

    // Warmup
    for (0..WARMUP_ITERS) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }

    // Benchmark
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        @memcpy(data_copy, data);
        plan.forward(data_copy);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const time_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const throughput = @as(f64, @floatFromInt(n)) / time_per_iter / 1000.0;

    return .{
        .name = "vdsp",
        .time_per_iter_ms = time_per_iter,
        .throughput_mpts = throughput,
    };
}

fn benchmarkRfft3d(allocator: std.mem.Allocator, grid: usize, iterations: usize) !BenchResult {
    const n = grid * grid * grid;
    const nx_complex = grid / 2 + 1;
    const complex_size = nx_complex * grid * grid;

    const real_data = try allocator.alloc(f64, n);
    defer allocator.free(real_data);
    const complex_data = try allocator.alloc(Complex, complex_size);
    defer allocator.free(complex_data);

    for (real_data, 0..) |*v, i| {
        v.* = @floatFromInt(i % 17);
    }

    var plan = try fft_lib.rfft3d.RealPlan3d.init(allocator, grid, grid, grid);
    defer plan.deinit();

    // Warmup
    for (0..WARMUP_ITERS) |_| {
        plan.forward(real_data, complex_data);
    }

    // Benchmark
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        plan.forward(real_data, complex_data);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const time_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const throughput = @as(f64, @floatFromInt(n)) / time_per_iter / 1000.0;

    return .{
        .name = "rfft3d",
        .time_per_iter_ms = time_per_iter,
        .throughput_mpts = throughput,
    };
}

fn printResults(grid: usize, results: []const BenchResult) void {
    const n = grid * grid * grid;
    std.debug.print("\n=== 3D FFT Benchmark: {d}x{d}x{d} ({d} points) ===\n", .{ grid, grid, grid, n });
    std.debug.print("{s:<20} {s:>12} {s:>12} {s:>10}\n", .{ "Backend", "ms/iter", "Mpt/s", "vs best" });
    std.debug.print("-" ** 56 ++ "\n", .{});

    // Find best time
    var best_time: f64 = std.math.inf(f64);
    for (results) |r| {
        if (r.time_per_iter_ms < best_time) best_time = r.time_per_iter_ms;
    }

    for (results) |r| {
        const ratio = r.time_per_iter_ms / best_time;
        const marker: []const u8 = if (ratio < 1.01) " <-- best" else "";
        std.debug.print("{s:<20} {d:>12.3} {d:>12.1} {d:>9.2}x{s}\n", .{
            r.name, r.time_per_iter_ms, r.throughput_mpts, ratio, marker,
        });
    }
}

const ResultList = struct {
    items: [16]BenchResult = undefined,
    len: usize = 0,

    fn append(self: *ResultList, r: BenchResult) void {
        self.items[self.len] = r;
        self.len += 1;
    }

    fn slice(self: *const ResultList) []const BenchResult {
        return self.items[0..self.len];
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("Comprehensive 3D FFT Backend Benchmark\n", .{});
    std.debug.print("Platform: {s}\n", .{@tagName(builtin.os.tag)});
    std.debug.print("=" ** 60 ++ "\n", .{});

    // --- Grid 24^3 (DFT common size, smooth number) ---
    {
        const grid: usize = 24;
        const iters: usize = 200;
        var results = ResultList{};

        results.append(try benchmarkPlan3d(allocator, grid, iters));
        results.append(try benchmarkParallel(allocator, grid, iters));
        results.append(try benchmarkTranspose(allocator, grid, iters));
        results.append(try benchmarkComptime24(allocator, iters));
        if (grid % 2 == 0) {
            results.append(try benchmarkRfft3d(allocator, grid, iters));
        }

        printResults(grid, results.slice());
    }

    // --- Grid 32^3 (power-of-2) ---
    {
        const grid: usize = 32;
        const iters: usize = 200;
        var results = ResultList{};

        results.append(try benchmarkPlan3d(allocator, grid, iters));
        results.append(try benchmarkParallel(allocator, grid, iters));
        results.append(try benchmarkTranspose(allocator, grid, iters));
        if (comptime builtin.os.tag == .macos) {
            results.append(try benchmarkVdsp(allocator, grid, iters));
        }
        if (grid % 2 == 0) {
            results.append(try benchmarkRfft3d(allocator, grid, iters));
        }

        printResults(grid, results.slice());
    }

    // --- Grid 48^3 (smooth number, larger) ---
    {
        const grid: usize = 48;
        const iters: usize = 50;
        var results = ResultList{};

        results.append(try benchmarkPlan3d(allocator, grid, iters));
        results.append(try benchmarkParallel(allocator, grid, iters));
        results.append(try benchmarkTranspose(allocator, grid, iters));
        if (grid % 2 == 0) {
            results.append(try benchmarkRfft3d(allocator, grid, iters));
        }

        printResults(grid, results.slice());
    }

    // --- Grid 64^3 (power-of-2, larger) ---
    {
        const grid: usize = 64;
        const iters: usize = 50;
        var results = ResultList{};

        results.append(try benchmarkPlan3d(allocator, grid, iters));
        results.append(try benchmarkParallel(allocator, grid, iters));
        results.append(try benchmarkTranspose(allocator, grid, iters));
        if (comptime builtin.os.tag == .macos) {
            results.append(try benchmarkVdsp(allocator, grid, iters));
        }
        if (grid % 2 == 0) {
            results.append(try benchmarkRfft3d(allocator, grid, iters));
        }

        printResults(grid, results.slice());
    }

    // --- Grid 96^3 (smooth number, large) ---
    {
        const grid: usize = 96;
        const iters: usize = 10;
        var results = ResultList{};

        results.append(try benchmarkPlan3d(allocator, grid, iters));
        results.append(try benchmarkParallel(allocator, grid, iters));
        results.append(try benchmarkTranspose(allocator, grid, iters));
        if (grid % 2 == 0) {
            results.append(try benchmarkRfft3d(allocator, grid, iters));
        }

        printResults(grid, results.slice());
    }

    // --- Grid 128^3 (power-of-2, large) ---
    {
        const grid: usize = 128;
        const iters: usize = 10;
        var results = ResultList{};

        results.append(try benchmarkPlan3d(allocator, grid, iters));
        results.append(try benchmarkParallel(allocator, grid, iters));
        results.append(try benchmarkTranspose(allocator, grid, iters));
        if (comptime builtin.os.tag == .macos) {
            results.append(try benchmarkVdsp(allocator, grid, iters));
        }
        if (grid % 2 == 0) {
            results.append(try benchmarkRfft3d(allocator, grid, iters));
        }

        printResults(grid, results.slice());
    }

    std.debug.print("\n" ++ "=" ** 60 ++ "\n", .{});
    std.debug.print("Benchmark complete.\n", .{});
}
