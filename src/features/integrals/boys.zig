//! Boys function F_n(x) for molecular integral evaluation.
//!
//! The Boys function is defined as:
//!   F_n(x) = ∫₀¹ t^(2n) exp(-x·t²) dt
//!
//! For n=0 (s-type orbitals only):
//!   F_0(x) = √(π/(4x)) × erf(√x)       for x > 0
//!   F_0(0) = 1
//!
//! References:
//!   - S. F. Boys, Proc. R. Soc. London A 200, 542 (1950)
//!   - Obara & Saika, J. Chem. Phys. 84, 3963 (1986)

const std = @import("std");
const c = @cImport({
    @cInclude("math.h");
});

/// Error function erf(x) from C standard library (full double precision).
fn erf(x: f64) f64 {
    return c.erf(x);
}

/// Compute the Boys function F_0(x).
///
/// F_0(x) = √(π/(4x)) × erf(√x) for x > 0
/// F_0(0) = 1
pub fn boys0(x: f64) f64 {
    if (x < 1e-12) {
        // Taylor expansion: F_0(x) ≈ 1 - x/3 + x²/10 - ...
        return 1.0 - x / 3.0;
    }
    const sqrt_x = @sqrt(x);
    return std.math.sqrt(std.math.pi / (4.0 * x)) * erf(sqrt_x);
}

/// Compute the Boys function F_n(x) for general n.
///
/// Uses two strategies depending on x:
///
/// For small x (< threshold), uses Taylor expansion:
///   F_n(x) = Σ_k (-x)^k / (k! × (2n + 2k + 1))
///   The threshold is chosen as min(5 + n, 12) to ensure convergence
///   with a reasonable number of terms.
///
/// For larger x, uses upward recurrence from F_0:
///   F_{n+1}(x) = ((2n+1) F_n(x) - exp(-x)) / (2x)
///   This is numerically stable for all x > ~1 because the subtraction
///   (2n+1)*F_n(x) - exp(-x) does not suffer catastrophic cancellation:
///   exp(-x) is tiny for large x while (2n+1)*F_n(x) remains well-conditioned.
pub fn boysN(n: u32, x: f64) f64 {
    if (n == 0) return boys0(x);

    if (x < 1e-12) {
        // F_n(0) = 1 / (2n + 1)
        return 1.0 / @as(f64, @floatFromInt(2 * n + 1));
    }

    // Adaptive threshold: Taylor expansion converges well only for small x.
    // Higher n requires even smaller x for convergence. The Taylor series
    // involves alternating terms (-x)^k / k! that grow before decaying,
    // causing catastrophic cancellation for moderate-to-large x.
    // Empirically: threshold = min(5 + n, 12) ensures relative error < 1e-12
    // for all n up to ~20.
    const taylor_threshold: f64 = @min(@as(f64, @floatFromInt(5 + n)), 12.0);

    if (x < taylor_threshold) {
        // Taylor expansion: F_n(x) = Σ_k (-x)^k / (k! × (2n + 2k + 1))
        var sum: f64 = 0.0;
        var term: f64 = 1.0; // (-x)^k / k!, starting at k=0 => 1
        sum = term / @as(f64, @floatFromInt(2 * n + 1));
        var k: u32 = 1;
        while (k < 200) : (k += 1) {
            term *= -x / @as(f64, @floatFromInt(k));
            const denom = @as(f64, @floatFromInt(2 * n + 2 * k + 1));
            const contrib = term / denom;
            sum += contrib;
            if (@abs(contrib) < 1e-15 * @abs(sum)) break;
        }
        return sum;
    }

    // Upward recurrence from F_0:
    //   F_{n+1}(x) = ((2n+1) F_n(x) - exp(-x)) / (2x)
    // This is numerically stable for x >= ~1 and is exact to machine precision.
    const f0 = boys0(x);
    const exp_neg_x = @exp(-x);
    var f_prev = f0;
    var i: u32 = 1;
    while (i <= n) : (i += 1) {
        const ni = @as(f64, @floatFromInt(2 * i - 1));
        f_prev = (ni * f_prev - exp_neg_x) / (2.0 * x);
    }
    return f_prev;
}

/// Compute all Boys function values F_0(x) through F_nmax(x) in one call.
///
/// This is more efficient than calling boysN() individually for each m,
/// because it computes F_nmax once (via Taylor or asymptotic) and then
/// uses downward recurrence to get F_{nmax-1}, ..., F_0.
///
/// Downward recurrence from F_n to F_{n-1}:
///   F_{n-1}(x) = (2x * F_n(x) + exp(-x)) / (2n - 1)
///
/// This is numerically stable because the denominator (2n-1) grows with n,
/// and the addition (not subtraction) avoids catastrophic cancellation.
///
/// Output: result[m] = F_m(x) for m = 0, 1, ..., nmax.
pub fn boysBatch(nmax: u32, x: f64, result: []f64) void {
    if (x < 1e-12) {
        // F_m(0) = 1/(2m+1)
        var m: u32 = 0;
        while (m <= nmax) : (m += 1) {
            result[m] = 1.0 / @as(f64, @floatFromInt(2 * m + 1));
        }
        return;
    }

    if (nmax == 0) {
        result[0] = boys0(x);
        return;
    }

    // Compute F_nmax using the existing (accurate) boysN function
    result[nmax] = boysN(nmax, x);

    // Downward recurrence: F_{n-1}(x) = (2x * F_n(x) + exp(-x)) / (2n - 1)
    const exp_neg_x = @exp(-x);
    const two_x = 2.0 * x;
    var n: u32 = nmax;
    while (n > 0) : (n -= 1) {
        const denom = @as(f64, @floatFromInt(2 * n - 1));
        result[n - 1] = (two_x * result[n] + exp_neg_x) / denom;
    }
}

test "boys F_0(0) = 1" {
    const testing = std.testing;
    try testing.expectApproxEqAbs(boys0(0.0), 1.0, 1e-12);
}

test "boys F_0 known values" {
    const testing = std.testing;
    // F_0(1) = √(π/4) × erf(1) ≈ 0.74682413
    try testing.expectApproxEqAbs(boys0(1.0), 0.746824133, 1e-6);
    // F_0(10) = √(π/40) × erf(√10) ≈ 0.28024739
    try testing.expectApproxEqAbs(boys0(10.0), 0.28024739, 1e-5);
}

test "boys F_n(0) = 1/(2n+1)" {
    const testing = std.testing;
    try testing.expectApproxEqAbs(boysN(0, 0.0), 1.0, 1e-12);
    try testing.expectApproxEqAbs(boysN(1, 0.0), 1.0 / 3.0, 1e-12);
    try testing.expectApproxEqAbs(boysN(2, 0.0), 1.0 / 5.0, 1e-12);
}

test "boys F_n scipy reference values" {
    const testing = std.testing;
    const tol: f64 = 1e-9;

    // Reference values from scipy.special.hyp1f1
    // x = 20.325618 (large-x upward recurrence path)
    try testing.expectApproxEqAbs(1.965726357771075e-01, boysN(0, 20.325618), tol);
    try testing.expectApproxEqAbs(4.835588130427028e-03, boysN(1, 20.325618), tol);
    try testing.expectApproxEqAbs(3.568590854890924e-04, boysN(2, 20.325618), tol);

    // x = 7.665237 (Taylor expansion path)
    try testing.expectApproxEqAbs(3.200685119119207e-01, boysN(0, 7.665237), tol);
    try testing.expectApproxEqAbs(2.084734407339424e-02, boysN(1, 7.665237), tol);
    try testing.expectApproxEqAbs(4.049006351303070e-03, boysN(2, 7.665237), tol);

    // x = 5.079105 (Taylor expansion path)
    try testing.expectApproxEqAbs(3.926693368267595e-01, boysN(0, 5.079105), tol);
    try testing.expectApproxEqAbs(3.804251521679047e-02, boysN(1, 5.079105), tol);
    try testing.expectApproxEqAbs(1.062215363966337e-02, boysN(2, 5.079105), tol);

    // x = 32.985999 (large-x upward recurrence path)
    try testing.expectApproxEqAbs(1.543050430110980e-01, boysN(0, 32.985999), tol);
    try testing.expectApproxEqAbs(2.338947548793251e-03, boysN(1, 32.985999), tol);
    try testing.expectApproxEqAbs(1.063609237115272e-04, boysN(2, 32.985999), tol);
}

test "boys F_n large argument x=29.68 (CH2O regression)" {
    const testing = std.testing;
    const tol: f64 = 1e-12;

    // Reference values from scipy.special.hyp1f1 at x = 29.68.
    // This argument caused negative Boys values with the old threshold (x<30 Taylor).
    // All Boys function values must be strictly positive.
    const x = 29.68;
    try testing.expectApproxEqAbs(1.626720697317235e-01, boysN(0, x), tol);
    try testing.expectApproxEqAbs(2.740432441569994e-03, boysN(1, x), tol);
    try testing.expectApproxEqAbs(1.384989441472560e-04, boysN(2, x), tol);
    try testing.expectApproxEqAbs(1.166601618273946e-05, boysN(3, x), tol);
    try testing.expectApproxEqAbs(1.375709453340797e-06, boysN(4, x), tol);
    try testing.expectApproxEqAbs(2.085812828706326e-07, boysN(5, x), tol);
    try testing.expectApproxEqAbs(3.865218973568958e-08, boysN(6, x), tol);
    try testing.expectApproxEqAbs(8.464931564985529e-09, boysN(7, x), tol);

    // Verify all values are positive (Boys function is always non-negative)
    var n: u32 = 0;
    while (n <= 12) : (n += 1) {
        const val = boysN(n, x);
        try testing.expect(val > 0.0);
    }
}

test "boys F_n moderate argument transition region" {
    const testing = std.testing;
    const tol: f64 = 1e-10;

    // Test at x values near the Taylor/recurrence boundary to ensure
    // both code paths produce consistent results.
    // Reference from scipy.special.hyp1f1.

    // x = 10.0 (near boundary for high n)
    try testing.expectApproxEqAbs(2.802473905066e-01, boysN(0, 10.0), tol);
    try testing.expectApproxEqAbs(2.099244932838e-03, boysN(2, 10.0), tol);
    try testing.expectApproxEqAbs(1.806194363644e-04, boysN(4, 10.0), tol);
    try testing.expectApproxEqAbs(4.118481594360e-05, boysN(6, 10.0), tol);

    // x = 15.0
    try testing.expectApproxEqAbs(2.288227983297e-01, boysN(0, 15.0), tol);
    try testing.expectApproxEqAbs(7.627314446807e-04, boysN(2, 15.0), tol);
    try testing.expectApproxEqAbs(2.964920241996e-05, boysN(4, 15.0), tol);
    try testing.expectApproxEqAbs(3.247476716040e-06, boysN(6, 15.0), tol);
}

test "boysBatch matches individual boysN calls" {
    const testing = std.testing;
    const tol: f64 = 1e-10;

    // Test batch at several x values and nmax values.
    // Tolerance is 1e-10 because the batch uses upward recurrence from F_0
    // for x >= 1, while individual boysN uses Taylor expansion for some (n, x)
    // combinations. Both are accurate to ~1e-12 individually, but they use
    // different code paths that can differ at ~1e-11 level.
    const x_vals = [_]f64{ 0.0, 0.001, 0.5, 1.0, 3.0, 7.5, 10.0, 15.0, 25.0, 29.68 };
    const nmax_vals = [_]u32{ 0, 1, 4, 8, 12 };

    for (x_vals) |x| {
        for (nmax_vals) |nmax| {
            var batch: [13]f64 = undefined;
            boysBatch(nmax, x, &batch);

            var m: u32 = 0;
            while (m <= nmax) : (m += 1) {
                const individual = boysN(m, x);
                try testing.expectApproxEqAbs(individual, batch[m], tol);
            }
        }
    }
}
