//! Logarithmic radial grid for atomic calculations.
//!
//! Grid points: r_i = a * (exp(b * i) - 1), i = 0..n-1
//! where a = r_min, b = ln(r_max/r_min + 1) / (n - 1)
//! This gives r_0 = 0, r_{n-1} = r_max.
//!
//! The Jacobian for integration: dr/di = a * b * exp(b * i) = rab_i

const std = @import("std");

pub const RadialGrid = struct {
    /// Grid points r_i (Bohr)
    r: []f64,
    /// Integration weights dr = rab_i (Bohr)
    rab: []f64,
    /// Number of grid points
    n: usize,
    /// Parameter a (determines smallest nonzero r)
    a: f64,
    /// Parameter b (logarithmic spacing)
    b: f64,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize, r_min: f64, r_max: f64) !RadialGrid {
        std.debug.assert(n >= 2);
        std.debug.assert(r_min > 0);
        std.debug.assert(r_max > r_min);

        const r = try allocator.alloc(f64, n);
        const rab = try allocator.alloc(f64, n);

        const a = r_min;
        const b_param = @log(r_max / r_min + 1.0) / @as(f64, @floatFromInt(n - 1));

        for (0..n) |i| {
            const fi: f64 = @floatFromInt(i);
            const exp_bi = @exp(b_param * fi);
            r[i] = a * (exp_bi - 1.0);
            rab[i] = a * b_param * exp_bi;
        }

        return .{
            .r = r,
            .rab = rab,
            .n = n,
            .a = a,
            .b = b_param,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RadialGrid) void {
        self.allocator.free(self.r);
        self.allocator.free(self.rab);
    }

    /// Integrate f(r) * r^2 dr over the grid using trapezoidal rule with
    /// Newton-Cotes endpoint correction (ctrap).
    pub fn integrate(self: *const RadialGrid, f: []const f64) f64 {
        std.debug.assert(f.len == self.n);
        var sum: f64 = 0;
        for (0..self.n) |i| {
            const w = ctrap_weight(i, self.n);
            sum += w * f[i] * self.r[i] * self.r[i] * self.rab[i];
        }
        return sum;
    }

    /// Integrate f(r) dr (no r^2 factor) over the grid.
    pub fn integrate_raw(self: *const RadialGrid, f: []const f64) f64 {
        std.debug.assert(f.len == self.n);
        var sum: f64 = 0;
        for (0..self.n) |i| {
            const w = ctrap_weight(i, self.n);
            sum += w * f[i] * self.rab[i];
        }
        return sum;
    }
};

/// Newton-Cotes 6th order endpoint correction weights (ABINIT convention).
/// Interior points have weight 1.0. First/last 5 points use corrected weights.
fn ctrap_weight(i: usize, n: usize) f64 {
    const endpoint_weights = [5]f64{
        23.75 / 72.0,
        95.10 / 72.0,
        55.20 / 72.0,
        79.30 / 72.0,
        70.65 / 72.0,
    };

    if (n < 10) {
        // Fall back to trapezoidal for very small grids
        if (i == 0 or i == n - 1) return 0.5;
        return 1.0;
    }

    // Left endpoint correction
    if (i < 5) return endpoint_weights[i];
    // Right endpoint correction
    if (i >= n - 5) return endpoint_weights[n - 1 - i];
    // Interior
    return 1.0;
}

// ============================================================================
// Tests
// ============================================================================

test "radial grid basic properties" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 500, 1e-6, 100.0);
    defer grid.deinit();

    // r[0] should be ~0 (a*(exp(0)-1) = 0)
    try std.testing.expectApproxEqAbs(0.0, grid.r[0], 1e-15);
    // r[n-1] should be r_max
    try std.testing.expectApproxEqAbs(100.0, grid.r[grid.n - 1], 1e-10);
    // Monotonically increasing
    for (1..grid.n) |i| {
        try std.testing.expect(grid.r[i] > grid.r[i - 1]);
    }
    // rab > 0 everywhere
    for (0..grid.n) |i| {
        try std.testing.expect(grid.rab[i] > 0);
    }
}

test "radial grid integration: gaussian" {
    // ∫_0^∞ exp(-r^2) * r^2 dr = sqrt(π)/4 ≈ 0.44311346
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 1000, 1e-7, 20.0);
    defer grid.deinit();

    const f = try allocator.alloc(f64, grid.n);
    defer allocator.free(f);

    for (0..grid.n) |i| {
        f[i] = @exp(-grid.r[i] * grid.r[i]);
    }

    const result = grid.integrate(f);
    const expected = @sqrt(std.math.pi) / 4.0;
    try std.testing.expectApproxEqAbs(expected, result, 1e-6);
}

test "radial grid integration: hydrogen 1s density normalization" {
    // ∫_0^∞ |R_1s(r)|^2 r^2 dr = 1, where R_1s = 2*exp(-r) (in Bohr)
    // ∫_0^∞ 4*exp(-2r) * r^2 dr = 4 * 2/8 = 1
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 1000, 1e-7, 50.0);
    defer grid.deinit();

    const f = try allocator.alloc(f64, grid.n);
    defer allocator.free(f);

    for (0..grid.n) |i| {
        f[i] = 4.0 * @exp(-2.0 * grid.r[i]);
    }

    const result = grid.integrate(f);
    try std.testing.expectApproxEqAbs(1.0, result, 1e-6);
}
