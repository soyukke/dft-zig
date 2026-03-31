const std = @import("std");

/// ABINIT-style ctrap endpoint weight for corrected trapezoidal integration.
/// Uses Newton-Cotes 5-point endpoint correction (6th order accuracy).
/// For an integral ∫f(r)dr ≈ Σ w(i) × f(r_i) × rab(i), where the first and
/// last 5 points receive modified weights instead of 1.0.
pub fn ctrapWeight(i: usize, n: usize) f64 {
    const w = [5]f64{ 23.75 / 72.0, 95.10 / 72.0, 55.20 / 72.0, 79.30 / 72.0, 70.65 / 72.0 };
    if (n < 10) return 1.0;
    if (i < 5) return w[i];
    if (i >= n - 5) return w[n - 1 - i];
    return 1.0;
}

test "ctrapWeight interior points are 1.0" {
    try std.testing.expectEqual(@as(f64, 1.0), ctrapWeight(5, 20));
    try std.testing.expectEqual(@as(f64, 1.0), ctrapWeight(10, 20));
    try std.testing.expectEqual(@as(f64, 1.0), ctrapWeight(14, 20));
}

test "ctrapWeight small grid returns 1.0" {
    // n < 10 => all weights are 1.0
    try std.testing.expectEqual(@as(f64, 1.0), ctrapWeight(0, 5));
    try std.testing.expectEqual(@as(f64, 1.0), ctrapWeight(3, 9));
}

test "ctrapWeight endpoint symmetry" {
    // w(i, n) == w(n-1-i, n) for endpoint points
    const n: usize = 20;
    for (0..5) |i| {
        try std.testing.expectEqual(ctrapWeight(i, n), ctrapWeight(n - 1 - i, n));
    }
}

test "ctrapWeight known values" {
    const tol = 1e-14;
    try std.testing.expectApproxEqAbs(@as(f64, 23.75 / 72.0), ctrapWeight(0, 20), tol);
    try std.testing.expectApproxEqAbs(@as(f64, 95.10 / 72.0), ctrapWeight(1, 20), tol);
    try std.testing.expectApproxEqAbs(@as(f64, 55.20 / 72.0), ctrapWeight(2, 20), tol);
    try std.testing.expectApproxEqAbs(@as(f64, 79.30 / 72.0), ctrapWeight(3, 20), tol);
    try std.testing.expectApproxEqAbs(@as(f64, 70.65 / 72.0), ctrapWeight(4, 20), tol);
}

test "ctrapWeight polynomial integration" {
    // Integrate f(x) = x^4 from 0 to 1 on uniform grid.
    // Exact = 0.2. Corrected trapezoid should be much closer than naive.
    const n: usize = 101;
    var sum: f64 = 0.0;
    const dx = 1.0 / @as(f64, @floatFromInt(n - 1));
    for (0..n) |i| {
        const x = @as(f64, @floatFromInt(i)) * dx;
        sum += x * x * x * x * dx * ctrapWeight(i, n);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), sum, 1e-8);
}
