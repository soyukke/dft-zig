//! SIMD-optimized complex vector operations
//!
//! Provides high-performance implementations of common complex vector
//! operations used in iterative eigensolvers like LOBPCG.

const std = @import("std");

/// Complex number type - uses same layout as math.Complex (extern struct { r, i })
pub const Complex = extern struct {
    r: f64,
    i: f64,

    pub fn init(re: f64, im: f64) Complex {
        return .{ .r = re, .i = im };
    }
};

// SIMD vector width for f64
const simd_width = 4;
const F64Vec = @Vector(simd_width, f64);

/// Complex inner product: <a|b> = sum(conj(a[i]) * b[i])
/// SIMD optimized version processes 4 complex numbers at a time.
pub fn inner_product(a: []const Complex, b: []const Complex) Complex {
    const n = @min(a.len, b.len);
    if (n == 0) return Complex.init(0.0, 0.0);

    var sum_re: f64 = 0.0;
    var sum_im: f64 = 0.0;

    // SIMD loop: process 4 complex numbers at a time
    const simd_end = n - (n % simd_width);
    var i: usize = 0;

    while (i < simd_end) : (i += simd_width) {
        // Load 4 complex numbers from a and b
        var a_re: F64Vec = undefined;
        var a_im: F64Vec = undefined;
        var b_re: F64Vec = undefined;
        var b_im: F64Vec = undefined;

        inline for (0..simd_width) |j| {
            a_re[j] = a[i + j].r;
            a_im[j] = a[i + j].i;
            b_re[j] = b[i + j].r;
            b_im[j] = b[i + j].i;
        }

        // conj(a) * b = (a_re - i*a_im) * (b_re + i*b_im)
        //             = (a_re*b_re + a_im*b_im) + i*(a_re*b_im - a_im*b_re)
        const prod_re = a_re * b_re + a_im * b_im;
        const prod_im = a_re * b_im - a_im * b_re;

        // Horizontal sum
        sum_re += @reduce(.Add, prod_re);
        sum_im += @reduce(.Add, prod_im);
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        // conj(a) * b
        sum_re += a[i].r * b[i].r + a[i].i * b[i].i;
        sum_im += a[i].r * b[i].i - a[i].i * b[i].r;
    }

    return Complex.init(sum_re, sum_im);
}

/// Vector norm: sqrt(<a|a>) = sqrt(sum(|a[i]|^2))
/// SIMD optimized.
pub fn vector_norm(a: []const Complex) f64 {
    const n = a.len;
    if (n == 0) return 0.0;

    var sum: f64 = 0.0;

    // SIMD loop
    const simd_end = n - (n % simd_width);
    var i: usize = 0;

    while (i < simd_end) : (i += simd_width) {
        var re: F64Vec = undefined;
        var im: F64Vec = undefined;

        inline for (0..simd_width) |j| {
            re[j] = a[i + j].r;
            im[j] = a[i + j].i;
        }

        const norm_sq = re * re + im * im;
        sum += @reduce(.Add, norm_sq);
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        sum += a[i].r * a[i].r + a[i].i * a[i].i;
    }

    return @sqrt(@max(0.0, sum));
}

/// AXPY: y = y + alpha * x
/// SIMD optimized.
pub fn axpy(y: []Complex, x: []const Complex, alpha: f64) void {
    const n = @min(y.len, x.len);
    if (n == 0) return;

    const alpha_vec: F64Vec = @splat(alpha);

    // SIMD loop
    const simd_end = n - (n % simd_width);
    var i: usize = 0;

    while (i < simd_end) : (i += simd_width) {
        var y_re: F64Vec = undefined;
        var y_im: F64Vec = undefined;
        var x_re: F64Vec = undefined;
        var x_im: F64Vec = undefined;

        inline for (0..simd_width) |j| {
            y_re[j] = y[i + j].r;
            y_im[j] = y[i + j].i;
            x_re[j] = x[i + j].r;
            x_im[j] = x[i + j].i;
        }

        y_re = y_re + alpha_vec * x_re;
        y_im = y_im + alpha_vec * x_im;

        inline for (0..simd_width) |j| {
            y[i + j].r = y_re[j];
            y[i + j].i = y_im[j];
        }
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        y[i].r += alpha * x[i].r;
        y[i].i += alpha * x[i].i;
    }
}

/// Complex AXPY: y = y + alpha * x where alpha is complex
pub fn axpy_complex(y: []Complex, x: []const Complex, alpha: Complex) void {
    const n = @min(y.len, x.len);
    if (n == 0) return;

    const alpha_re: F64Vec = @splat(alpha.r);
    const alpha_im: F64Vec = @splat(alpha.i);

    // SIMD loop
    const simd_end = n - (n % simd_width);
    var i: usize = 0;

    while (i < simd_end) : (i += simd_width) {
        var y_re: F64Vec = undefined;
        var y_im: F64Vec = undefined;
        var x_re: F64Vec = undefined;
        var x_im: F64Vec = undefined;

        inline for (0..simd_width) |j| {
            y_re[j] = y[i + j].r;
            y_im[j] = y[i + j].i;
            x_re[j] = x[i + j].r;
            x_im[j] = x[i + j].i;
        }

        // y += alpha * x = (alpha_re + i*alpha_im) * (x_re + i*x_im)
        //                = (alpha_re*x_re - alpha_im*x_im) + i*(alpha_re*x_im + alpha_im*x_re)
        y_re = y_re + alpha_re * x_re - alpha_im * x_im;
        y_im = y_im + alpha_re * x_im + alpha_im * x_re;

        inline for (0..simd_width) |j| {
            y[i + j].r = y_re[j];
            y[i + j].i = y_im[j];
        }
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        const prod_re = alpha.r * x[i].r - alpha.i * x[i].i;
        const prod_im = alpha.r * x[i].i + alpha.i * x[i].r;
        y[i].r += prod_re;
        y[i].i += prod_im;
    }
}

/// Scale vector: y = alpha * x
pub fn scale(y: []Complex, x: []const Complex, alpha: f64) void {
    const n = @min(y.len, x.len);
    if (n == 0) return;

    const alpha_vec: F64Vec = @splat(alpha);

    // SIMD loop
    const simd_end = n - (n % simd_width);
    var i: usize = 0;

    while (i < simd_end) : (i += simd_width) {
        var x_re: F64Vec = undefined;
        var x_im: F64Vec = undefined;

        inline for (0..simd_width) |j| {
            x_re[j] = x[i + j].r;
            x_im[j] = x[i + j].i;
        }

        const y_re = alpha_vec * x_re;
        const y_im = alpha_vec * x_im;

        inline for (0..simd_width) |j| {
            y[i + j].r = y_re[j];
            y[i + j].i = y_im[j];
        }
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        y[i].r = alpha * x[i].r;
        y[i].i = alpha * x[i].i;
    }
}

/// Scale vector in-place: x *= alpha
pub fn scale_in_place(x: []Complex, alpha: f64) void {
    scale(x, x, alpha);
}

/// Zero vector
pub fn zero(x: []Complex) void {
    @memset(x, Complex.init(0.0, 0.0));
}

/// Copy vector: y = x
pub fn copy(y: []Complex, x: []const Complex) void {
    const n = @min(y.len, x.len);
    @memcpy(y[0..n], x[0..n]);
}

// ============== Tests ==============

test "inner_product basic" {
    const a = [_]Complex{
        Complex.init(1.0, 2.0),
        Complex.init(3.0, 4.0),
    };
    const b = [_]Complex{
        Complex.init(5.0, 6.0),
        Complex.init(7.0, 8.0),
    };

    // <a|b> = conj(a[0])*b[0] + conj(a[1])*b[1]
    //       = (1-2i)*(5+6i) + (3-4i)*(7+8i)
    //       = (5+6i-10i+12) + (21+24i-28i+32)
    //       = (17-4i) + (53-4i) = 70 - 8i
    const result = inner_product(&a, &b);
    try std.testing.expectApproxEqAbs(70.0, result.r, 1e-10);
    try std.testing.expectApproxEqAbs(-8.0, result.i, 1e-10);
}

test "inner_product SIMD" {
    var a: [100]Complex = undefined;
    var b: [100]Complex = undefined;

    for (0..100) |i| {
        const fi: f64 = @floatFromInt(i);
        a[i] = Complex.init(fi, fi * 0.5);
        b[i] = Complex.init(fi * 2.0, fi * 0.25);
    }

    // Reference scalar implementation
    var ref_re: f64 = 0.0;
    var ref_im: f64 = 0.0;
    for (0..100) |i| {
        ref_re += a[i].r * b[i].r + a[i].i * b[i].i;
        ref_im += a[i].r * b[i].i - a[i].i * b[i].r;
    }

    const result = inner_product(&a, &b);
    try std.testing.expectApproxEqAbs(ref_re, result.r, 1e-8);
    try std.testing.expectApproxEqAbs(ref_im, result.i, 1e-8);
}

test "vector_norm" {
    const a = [_]Complex{
        Complex.init(3.0, 4.0), // |a[0]|^2 = 25
        Complex.init(0.0, 0.0),
    };
    const norm = vector_norm(&a);
    try std.testing.expectApproxEqAbs(5.0, norm, 1e-10);
}

test "axpy" {
    var y = [_]Complex{
        Complex.init(1.0, 2.0),
        Complex.init(3.0, 4.0),
    };
    const x = [_]Complex{
        Complex.init(10.0, 20.0),
        Complex.init(30.0, 40.0),
    };

    axpy(&y, &x, 0.5);

    try std.testing.expectApproxEqAbs(6.0, y[0].r, 1e-10);
    try std.testing.expectApproxEqAbs(12.0, y[0].i, 1e-10);
    try std.testing.expectApproxEqAbs(18.0, y[1].r, 1e-10);
    try std.testing.expectApproxEqAbs(24.0, y[1].i, 1e-10);
}
