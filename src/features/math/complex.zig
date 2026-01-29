const std = @import("std");

pub const Complex = extern struct {
    r: f64,
    i: f64,
};

/// Construct a complex number from real and imag parts.
pub fn init(r: f64, i: f64) Complex {
    return .{ .r = r, .i = i };
}

/// Complex add.
pub fn add(a: Complex, b: Complex) Complex {
    return .{ .r = a.r + b.r, .i = a.i + b.i };
}

/// Complex subtract.
pub fn sub(a: Complex, b: Complex) Complex {
    return .{ .r = a.r - b.r, .i = a.i - b.i };
}

/// Complex multiply.
pub fn mul(a: Complex, b: Complex) Complex {
    return .{
        .r = a.r * b.r - a.i * b.i,
        .i = a.r * b.i + a.i * b.r,
    };
}

/// Multiply complex by real.
pub fn scale(a: Complex, s: f64) Complex {
    return .{ .r = a.r * s, .i = a.i * s };
}

/// Complex conjugate.
pub fn conj(a: Complex) Complex {
    return .{ .r = a.r, .i = -a.i };
}

/// Complex exponential of i*theta.
pub fn expi(theta: f64) Complex {
    return .{ .r = std.math.cos(theta), .i = std.math.sin(theta) };
}
