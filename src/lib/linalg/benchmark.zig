//! Benchmark for complex vector operations
//!
//! Compare SIMD-optimized vs scalar implementations.

const std = @import("std");
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

fn benchmarkInnerProduct(
    io: std.Io,
    out: *std.Io.Writer,
    comptime name: []const u8,
    comptime func: anytype,
    a: []const Complex,
    b: []const Complex,
    iterations: usize,
) !f64 {
    const t_start = std.Io.Clock.Timestamp.now(io, .awake);
    var acc: f64 = 0;

    for (0..iterations) |_| {
        const result = func(a, b);
        acc += result.r + result.i;
    }

    const elapsed_ns: u64 = @intCast(t_start.untilNow(io).raw.nanoseconds);
    volatile_sink = acc; // Prevent optimization
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    try out.print("{s}: {d:.3} ms ({d} iterations)\n", .{ name, elapsed_ms, iterations });
    return elapsed_ms;
}

fn benchmarkNorm(
    io: std.Io,
    out: *std.Io.Writer,
    comptime name: []const u8,
    func: *const fn ([]const Complex) f64,
    a: []const Complex,
    iterations: usize,
) !f64 {
    const t_start = std.Io.Clock.Timestamp.now(io, .awake);
    var acc: f64 = 0;

    for (0..iterations) |_| {
        acc += func(a);
    }

    const elapsed_ns: u64 = @intCast(t_start.untilNow(io).raw.nanoseconds);
    volatile_sink = acc;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    try out.print("{s}: {d:.3} ms ({d} iterations)\n", .{ name, elapsed_ms, iterations });
    return elapsed_ms;
}

/// Run the axpy scalar-vs-SIMD timing and print the speedup ratio.
/// Copies `y_src` into `y_copy` each iteration to avoid accumulation drift.
fn benchmarkAxpy(
    io: std.Io,
    out: *std.Io.Writer,
    a: []const Complex,
    y_src: []const Complex,
    y_copy: []Complex,
    iterations: usize,
) !void {
    const t_scalar = std.Io.Clock.Timestamp.now(io, .awake);
    for (0..iterations) |_| {
        @memcpy(y_copy, y_src);
        axpyScalar(y_copy, a, 0.5);
    }
    const scalar_axpy_ns: u64 = @intCast(t_scalar.untilNow(io).raw.nanoseconds);
    const scalar_axpy = @as(f64, @floatFromInt(scalar_axpy_ns)) / 1_000_000.0;

    const t_simd = std.Io.Clock.Timestamp.now(io, .awake);
    for (0..iterations) |_| {
        @memcpy(y_copy, y_src);
        complex_vec.axpy(y_copy, a, 0.5);
    }
    const simd_axpy_ns: u64 = @intCast(t_simd.untilNow(io).raw.nanoseconds);
    const simd_axpy = @as(f64, @floatFromInt(simd_axpy_ns)) / 1_000_000.0;

    try out.print("axpy (scalar): {d:.3} ms\n", .{scalar_axpy});
    try out.print("axpy (SIMD)  : {d:.3} ms\n", .{simd_axpy});
    try out.print("  Speedup: {d:.2}x\n\n", .{scalar_axpy / simd_axpy});
}

/// Run inner-product scalar-vs-SIMD benchmarks and print the speedup ratio.
fn benchmarkInnerProductPair(
    io: std.Io,
    out: *std.Io.Writer,
    a: []const Complex,
    b: []const Complex,
    iterations: usize,
) !void {
    const ip_s = innerProductScalarNoInline;
    const ip_v = innerProductSimdNoInline;
    const scalar_ip = try benchmarkInnerProduct(
        io,
        out,
        "innerProduct (scalar)",
        ip_s,
        a,
        b,
        iterations,
    );
    const simd_ip = try benchmarkInnerProduct(
        io,
        out,
        "innerProduct (SIMD)  ",
        ip_v,
        a,
        b,
        iterations,
    );
    try out.print("  Speedup: {d:.2}x\n\n", .{scalar_ip / simd_ip});
}

/// Run vector-norm scalar-vs-SIMD benchmarks and print the speedup ratio.
fn benchmarkNormPair(
    io: std.Io,
    out: *std.Io.Writer,
    a: []const Complex,
    iterations: usize,
) !void {
    const vn_s = vectorNormScalarNoInline;
    const vn_v = vectorNormSimdNoInline;
    const scalar_norm = try benchmarkNorm(io, out, "vectorNorm (scalar)  ", vn_s, a, iterations);
    const simd_norm = try benchmarkNorm(io, out, "vectorNorm (SIMD)    ", vn_v, a, iterations);
    try out.print("  Speedup: {d:.2}x\n\n", .{scalar_norm / simd_norm});
}

fn runSizeBenchmark(
    alloc: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    size: usize,
    iterations: usize,
) !void {
    try out.print("--- Size: {d} ---\n", .{size});

    const a = try alloc.alloc(Complex, size);
    defer alloc.free(a);

    const b = try alloc.alloc(Complex, size);
    defer alloc.free(b);

    const y = try alloc.alloc(Complex, size);
    defer alloc.free(y);

    // Initialize with random-ish data
    for (0..size) |i| {
        const fi: f64 = @floatFromInt(i);
        a[i] = Complex.init(@sin(fi), @cos(fi));
        b[i] = Complex.init(@cos(fi * 1.5), @sin(fi * 0.7));
        y[i] = Complex.init(fi * 0.01, fi * 0.02);
    }

    try benchmarkInnerProductPair(io, out, a, b, iterations);
    try benchmarkNormPair(io, out, a, iterations);

    // Reset y for axpy benchmark
    for (0..size) |i| {
        const fi: f64 = @floatFromInt(i);
        y[i] = Complex.init(fi * 0.01, fi * 0.02);
    }

    // axpy benchmark (need to copy y each time to avoid accumulation)
    const y_copy = try alloc.alloc(Complex, size);
    defer alloc.free(y_copy);

    try benchmarkAxpy(io, out, a, y, y_copy, iterations);
}

pub fn main(init: std.process.Init) !void {
    const sizes = [_]usize{ 500, 1000, 2000, 5000 };
    const iterations = 50000;

    const alloc = init.gpa;
    const io = init.io;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const out = &stdout_writer.interface;

    try out.print("\n=== Complex Vector Operations Benchmark ===\n\n", .{});

    for (sizes) |size| {
        try runSizeBenchmark(alloc, io, out, size, iterations);
    }

    try out.flush();
}
