//! Radial quadrature grids for atomic integration.
//!
//! Implements the Treutler-Ahlrichs (M4) radial transformation
//! [JCP 102, 346 (1995)] which is the default in PySCF.
//!
//! Also implements Mura-Knowles radial grids as an alternative.

const std = @import("std");
const math = std.math;

/// Bragg-Slater atomic radii in Bohr.
/// Used for scaling the radial grid per atom.
/// Index by atomic number Z. Values from PySCF data.radii.BRAGG.
pub const bragg_slater_radii = [_]f64{
    0.0, // Z=0 placeholder
    0.661404, // H  (Z=1)
    0.661404, // He (Z=2) - same as H
    2.456710, // Li (Z=3)
    1.795306, // Be (Z=4)
    1.606334, // B  (Z=5)
    1.322808, // C  (Z=6)
    1.228322, // N  (Z=7)
    1.133836, // O  (Z=8)
    0.944863, // F  (Z=9)
    0.944863, // Ne (Z=10)
};

/// Returns the Bragg-Slater radius for a given atomic number.
/// Falls back to 2.0 Bohr for elements not in the table.
pub fn braggSlaterRadius(z: usize) f64 {
    if (z < bragg_slater_radii.len) {
        return bragg_slater_radii[z];
    }
    return 2.0; // default fallback
}

/// A radial grid point with position and integration weight.
pub const RadialPoint = struct {
    r: f64,
    w: f64,
};

/// Treutler-Ahlrichs (M4) radial grid.
///
/// The transformation maps the Gauss-Chebyshev points x_i in [-1, 1] to
/// radial points r_i in [0, infinity):
///
///   r(x) = -alpha / ln(2) * (1 + x)^0.6 * ln((1 - x) / 2)
///
/// where alpha is a scaling parameter (typically Bragg-Slater radius / ln(2)).
///
/// The weight includes the Jacobian dr/dx and the Gauss-Chebyshev weight pi/n.
///
/// Returns an array of n RadialPoint structs.
pub fn treutlerAhlrichs(allocator: std.mem.Allocator, n: usize, alpha_scale: f64) ![]RadialPoint {
    const points = try allocator.alloc(RadialPoint, n);
    errdefer allocator.free(points);

    const ln2 = @as(f64, math.ln2);
    // alpha = Bragg-Slater radius, but the actual implementation uses
    // alpha / ln(2) as the overall scale factor
    const alpha = alpha_scale / ln2;

    for (0..n) |i| {
        // Gauss-Chebyshev points: x_i = cos(pi * (i + 0.5) / n)
        const fi: f64 = @floatFromInt(i);
        const fn_: f64 = @floatFromInt(n);
        const x = @cos(math.pi * (fi + 0.5) / fn_);

        // Treutler M4 transformation
        // r(x) = -alpha * (1+x)^0.6 * ln((1-x)/2)
        const one_plus_x = 1.0 + x;
        const one_minus_x = 1.0 - x;
        const log_arg = one_minus_x / 2.0;

        // Avoid log(0) at x = 1
        if (log_arg < 1e-300) {
            points[i] = .{ .r = 0.0, .w = 0.0 };
            continue;
        }

        const ln_val = @log(log_arg);
        const pow06 = @exp(0.6 * @log(@max(one_plus_x, 1e-300)));

        const r = -alpha * pow06 * ln_val;

        // dr/dx = alpha * [ 0.6 * (1+x)^(-0.4) * (-ln((1-x)/2)) + (1+x)^0.6 / (1-x) ]
        const pow_neg04 = @exp(-0.4 * @log(@max(one_plus_x, 1e-300)));
        const dr_dx = alpha * (0.6 * pow_neg04 * (-ln_val) + pow06 / @max(one_minus_x, 1e-300));

        // Gauss-Chebyshev weight: pi/n * sqrt(1 - x^2)
        const gc_weight = math.pi / fn_ * @sqrt(@max(1.0 - x * x, 0.0));

        // Full weight includes r^2 for the spherical volume element
        // But we don't include r^2 here - the caller decides whether to include it
        // (typically: integral = sum_i f(r_i) * w_i * r_i^2)
        const w = gc_weight * dr_dx;

        points[i] = .{ .r = r, .w = w };
    }

    return points;
}

/// Mura-Knowles radial grid.
///
/// Uses the transformation r(x) = -alpha * ln(1 - x^3) where x_i are
/// uniformly spaced on (0, 1).
///
/// Reference: Mura & Knowles, JCP 104, 9848 (1996).
pub fn muraKnowles(allocator: std.mem.Allocator, n: usize, alpha_scale: f64) ![]RadialPoint {
    const points = try allocator.alloc(RadialPoint, n);
    errdefer allocator.free(points);

    for (0..n) |i| {
        const fi: f64 = @floatFromInt(i);
        const fn_: f64 = @floatFromInt(n);

        // Uniform grid on (0, 1): x_i = (i + 0.5) / n
        const x = (fi + 0.5) / fn_;
        const x3 = x * x * x;

        // r(x) = -alpha * ln(1 - x^3)
        const r = -alpha_scale * @log(1.0 - x3);

        // dr/dx = alpha * 3*x^2 / (1 - x^3)
        const dr_dx = alpha_scale * 3.0 * x * x / (1.0 - x3);

        // Uniform weight: 1/n
        const w = dr_dx / fn_;

        points[i] = .{ .r = r, .w = w };
    }

    return points;
}

/// Generates the default radial grid for an atom.
/// Uses Treutler-Ahlrichs with the Bragg-Slater radius scaling.
/// n_radial is the number of radial points (typically 75 for light atoms).
pub fn defaultRadialGrid(allocator: std.mem.Allocator, atomic_number: usize, n_radial: usize) ![]RadialPoint {
    const radius = braggSlaterRadius(atomic_number);
    return treutlerAhlrichs(allocator, n_radial, radius);
}

// --- Tests ---

test "treutler radial grid basic properties" {
    const allocator = std.testing.allocator;

    const grid = try treutlerAhlrichs(allocator, 50, 1.0);
    defer allocator.free(grid);

    // All r values should be positive
    for (grid) |pt| {
        try std.testing.expect(pt.r >= 0.0);
        try std.testing.expect(pt.w >= 0.0);
    }

    // Points should span a reasonable range
    var r_max: f64 = 0.0;
    var r_min: f64 = 1e30;
    for (grid) |pt| {
        if (pt.r > r_max) r_max = pt.r;
        if (pt.r > 0 and pt.r < r_min) r_min = pt.r;
    }

    // Should reach at least 10 Bohr
    try std.testing.expect(r_max > 10.0);
    // Should have points close to origin
    try std.testing.expect(r_min < 0.01);
}

test "mura-knowles radial grid basic properties" {
    const allocator = std.testing.allocator;

    const grid = try muraKnowles(allocator, 50, 1.0);
    defer allocator.free(grid);

    for (grid) |pt| {
        try std.testing.expect(pt.r > 0.0);
        try std.testing.expect(pt.w > 0.0);
    }
}

test "treutler grid integrates r^2 * exp(-r^2)" {
    // Integral of r^2 * exp(-r^2) * r^2 dr from 0 to inf (with r^2 volume element)
    // = integral of r^4 * exp(-r^2) dr = 3*sqrt(pi)/8
    const allocator = std.testing.allocator;

    const grid = try treutlerAhlrichs(allocator, 100, 1.0);
    defer allocator.free(grid);

    var integral: f64 = 0.0;
    for (grid) |pt| {
        const r = pt.r;
        const f = r * r * @exp(-r * r); // r^2 * exp(-r^2)
        integral += f * pt.w * r * r; // * r^2 for volume element
    }

    const expected = 3.0 * @sqrt(math.pi) / 8.0;
    // Allow some tolerance for numerical quadrature
    try std.testing.expectApproxEqAbs(expected, integral, 1e-8);
}

test "mura-knowles grid integrates r^2 * exp(-r^2)" {
    const allocator = std.testing.allocator;

    const grid = try muraKnowles(allocator, 100, 1.0);
    defer allocator.free(grid);

    var integral: f64 = 0.0;
    for (grid) |pt| {
        const r = pt.r;
        const f = r * r * @exp(-r * r);
        integral += f * pt.w * r * r;
    }

    const expected = 3.0 * @sqrt(math.pi) / 8.0;
    // Mura-Knowles with alpha=1.0 and 100 points has lower accuracy than Treutler
    try std.testing.expectApproxEqAbs(expected, integral, 1e-4);
}
