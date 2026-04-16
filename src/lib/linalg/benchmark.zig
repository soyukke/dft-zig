//! Benchmark for complex vector operations
//!
//! Compare SIMD-optimized vs scalar implementations.

const std = @import("std");
const Timer = @import("../timer.zig").Timer;
const complex_vec = @import("complex_vec.zig");
const Complex = complex_vec.Complex;

/// Scalar reference implementation of inner product
fn innerProductScalar(a: []const Complex, b: []const Complex) Complex {
    var sum_re: f64 = 0.0;
    var sum_im: f64 = 0.0;
    const n = @min(a.len, b.len);
    for (0..n) |i| {
        // conj(a) * b
        sum_re += a[i].r * b[i].r + a[i].i * b[i].i;
        sum_im += a[i].r * b[i].i - a[i].i * b[i].r;
    }
    return Complex.init(sum_re, sum_im);
}

/// Scalar reference implementation of vectorNorm
fn vectorNormScalar(a: []const Complex) f64 {
    var sum: f64 = 0.0;
    for (a) |v| {
        sum += v.r * v.r + v.i * v.i;
    }
    return @sqrt(sum);
}

// Wrapper functions with noinline to prevent dead code elimination
noinline fn innerProductScalarNoInline(a: []const Complex, b: []const Complex) Complex {
    return innerProductScalar(a, b);
}

noinline fn innerProductSimdNoInline(a: []const Complex, b: []const Complex) Complex {
    return complex_vec.innerProduct(a, b);
}

noinline fn vectorNormScalarNoInline(a: []const Complex) f64 {
    return vectorNormScalar(a);
}

noinline fn vectorNormSimdNoInline(a: []const Complex) f64 {
    return complex_vec.vectorNorm(a);
}

/// Scalar reference implementation of axpy
fn axpyScalar(y: []Complex, x: []const Complex, alpha: f64) void {
    const n = @min(y.len, x.len);
    for (0..n) |i| {
        y[i].r += alpha * x[i].r;
        y[i].i += alpha * x[i].i;
    }
}

/// Volatile sink to prevent dead code elimination
var volatile_sink: f64 = 0;

fn benchmarkInnerProduct(comptime name: []const u8, comptime func: anytype, a: []const Complex, b: []const Complex, iterations: usize) f64 {
    var timer = Timer.start() catch return 0;
    var acc: f64 = 0;

    for (0..iterations) |_| {
        const result = func(a, b);
        acc += result.r + result.i;
    }

    const elapsed_ns = timer.read();
    volatile_sink = acc; // Prevent optimization
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    std.debug.print("{s}: {d:.3} ms ({d} iterations)\n", .{ name, elapsed_ms, iterations });
    return elapsed_ms;
}

fn benchmarkNorm(comptime name: []const u8, func: *const fn ([]const Complex) f64, a: []const Complex, iterations: usize) f64 {
    var timer = Timer.start() catch return 0;
    var acc: f64 = 0;

    for (0..iterations) |_| {
        acc += func(a);
    }

    const elapsed_ns = timer.read();
    volatile_sink = acc;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    std.debug.print("{s}: {d:.3} ms ({d} iterations)\n", .{ name, elapsed_ms, iterations });
    return elapsed_ms;
}

pub fn main() !void {
    const sizes = [_]usize{ 500, 1000, 2000, 5000 };
    const iterations = 50000;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("\n=== Complex Vector Operations Benchmark ===\n\n", .{});

    for (sizes) |size| {
        std.debug.print("--- Size: {d} ---\n", .{size});

        const a = try alloc.alloc(Complex, size);
        defer alloc.free(a);
        const b = try alloc.alloc(Complex, size);
        defer alloc.free(b);
        var y = try alloc.alloc(Complex, size);
        defer alloc.free(y);

        // Initialize with random-ish data
        for (0..size) |i| {
            const fi: f64 = @floatFromInt(i);
            a[i] = Complex.init(@sin(fi), @cos(fi));
            b[i] = Complex.init(@cos(fi * 1.5), @sin(fi * 0.7));
            y[i] = Complex.init(fi * 0.01, fi * 0.02);
        }

        // innerProduct benchmark
        const scalar_ip = benchmarkInnerProduct("innerProduct (scalar)", innerProductScalarNoInline, a, b, iterations);
        const simd_ip = benchmarkInnerProduct("innerProduct (SIMD)  ", innerProductSimdNoInline, a, b, iterations);
        std.debug.print("  Speedup: {d:.2}x\n\n", .{scalar_ip / simd_ip});

        // vectorNorm benchmark
        const scalar_norm = benchmarkNorm("vectorNorm (scalar)  ", vectorNormScalarNoInline, a, iterations);
        const simd_norm = benchmarkNorm("vectorNorm (SIMD)    ", vectorNormSimdNoInline, a, iterations);
        std.debug.print("  Speedup: {d:.2}x\n\n", .{scalar_norm / simd_norm});

        // Reset y for axpy benchmark
        for (0..size) |i| {
            const fi: f64 = @floatFromInt(i);
            y[i] = Complex.init(fi * 0.01, fi * 0.02);
        }

        // axpy benchmark (need to copy y each time to avoid accumulation)
        const y_copy = try alloc.alloc(Complex, size);
        defer alloc.free(y_copy);

        var timer_scalar = Timer.start() catch unreachable;
        for (0..iterations) |_| {
            @memcpy(y_copy, y);
            axpyScalar(y_copy, a, 0.5);
        }
        const scalar_axpy = @as(f64, @floatFromInt(timer_scalar.read())) / 1_000_000.0;

        var timer_simd = Timer.start() catch unreachable;
        for (0..iterations) |_| {
            @memcpy(y_copy, y);
            complex_vec.axpy(y_copy, a, 0.5);
        }
        const simd_axpy = @as(f64, @floatFromInt(timer_simd.read())) / 1_000_000.0;

        std.debug.print("axpy (scalar): {d:.3} ms\n", .{scalar_axpy});
        std.debug.print("axpy (SIMD)  : {d:.3} ms\n", .{simd_axpy});
        std.debug.print("  Speedup: {d:.2}x\n\n", .{scalar_axpy / simd_axpy});
    }
}
