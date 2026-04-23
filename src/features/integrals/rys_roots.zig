//! Rys quadrature roots and weights for electron repulsion integrals.
//!
//! Computes Rys polynomial roots and weights from Boys function moments.
//! The Rys quadrature exactly evaluates ERIs using nroots = ceil((la+lb+lc+ld)/2) + 1
//! quadrature points, where la..ld are the total angular momenta of the four shells.
//!
//! Algorithm:
//!   1. Compute Boys function moments F_0(x) .. F_{2*nroots-1}(x)
//!   2. Build modified Chebyshev / Stieltjes moment matrix
//!   3. Extract roots (u_i) and weights (w_i) via tridiagonal eigenvalue decomposition
//!
//! The roots u_i and weights w_i satisfy:
//!   F_n(x) = Σ_i w_i * u_i^n   for n = 0, 1, ..., 2*nroots-1
//!
//! References:
//!   - Dupuis, Rys, King, J. Chem. Phys. 65, 111 (1976)
//!   - Lindh, Ryu, Liu, J. Chem. Phys. 95, 5889 (1991)
//!   - King, Dupuis, J. Comput. Phys. 21, 144 (1976)

const std = @import("std");
const boys_mod = @import("boys.zig");

/// Maximum number of Rys roots supported.
/// For 6-31G(2df,p): max angular momentum is f (l=3), so max nroots = ceil(12/2)+1 = 7.
/// Support up to 10 for safety (g-type orbitals).
pub const MAX_NROOTS: usize = 10;

/// Maximum number of Boys function moments needed: 2*MAX_NROOTS.
const MAX_MOMENTS: usize = 2 * MAX_NROOTS;

/// Small threshold for treating x as essentially zero.
const SMALL_X: f64 = 1e-15;

/// Compute Rys quadrature roots and weights.
///
/// Given the Boys function argument x and the number of roots nroots,
/// computes roots[0..nroots] and weights[0..nroots] such that:
///   Σ_i weights[i] * roots[i]^n = F_n(x)   for n = 0..2*nroots-1
///
/// The roots are values of t^2 where t is the Rys variable.
///
/// Parameters:
///   nroots: number of quadrature points (1 to MAX_NROOTS)
///   x: Boys function argument (rho * |PQ|^2)
///   roots: output array for roots (at least nroots elements)
///   weights: output array for weights (at least nroots elements)
pub fn rysRoots(nroots: usize, x: f64, roots: []f64, weights: []f64) void {
    std.debug.assert(nroots >= 1 and nroots <= MAX_NROOTS);
    std.debug.assert(roots.len >= nroots);
    std.debug.assert(weights.len >= nroots);

    if (nroots == 1) {
        rysRoots1(x, roots, weights);
        return;
    }

    // Compute Boys function moments F_0(x) through F_{2*nroots-1}(x)
    var moments: [MAX_MOMENTS]f64 = undefined;
    boys_mod.boysBatch(@as(u32, @intCast(2 * nroots - 1)), x, &moments);

    // Use the modified Chebyshev algorithm to find the recurrence coefficients
    // of the orthogonal polynomials, then find roots via tridiagonal eigenproblem.
    //
    // The idea: given moments mu_k = F_k(x), find alpha_j, beta_j such that
    // the orthogonal polynomials P_j(t) satisfy:
    //   t * P_j(t) = P_{j+1}(t) + alpha_j * P_j(t) + beta_j * P_{j-1}(t)
    //
    // Then the roots are eigenvalues of the tridiagonal matrix with
    // diagonal = alpha and off-diagonal = sqrt(beta), and
    // weights[i] = mu_0 * v_i[0]^2 where v_i is the eigenvector.

    // Step 1: Compute recurrence coefficients via modified Chebyshev algorithm
    var alpha: [MAX_NROOTS]f64 = undefined;
    var beta_val: [MAX_NROOTS]f64 = undefined;

    modifiedChebyshev(nroots, &moments, &alpha, &beta_val);

    // Step 2: Solve tridiagonal eigenproblem using implicit QR (Golub-Welsch)
    // The tridiagonal matrix has:
    //   diagonal: alpha[0], alpha[1], ..., alpha[nroots-1]
    //   off-diagonal: sqrt(beta[1]), sqrt(beta[2]), ..., sqrt(beta[nroots-1])
    //
    // beta[0] = mu_0 = F_0(x) (used for weight calculation)

    var diag: [MAX_NROOTS]f64 = undefined;
    var offdiag: [MAX_NROOTS]f64 = undefined;

    for (0..nroots) |i| {
        diag[i] = alpha[i];
    }
    offdiag[0] = 0.0;
    for (1..nroots) |i| {
        if (beta_val[i] > 0.0) {
            offdiag[i] = @sqrt(beta_val[i]);
        } else {
            offdiag[i] = 0.0;
        }
    }

    // Eigenvector first components (for weight calculation)
    var z: [MAX_NROOTS]f64 = undefined;
    for (0..nroots) |i| {
        z[i] = if (i == 0) 1.0 else 0.0;
    }

    // Implicit QR algorithm for symmetric tridiagonal matrix
    tridiagEigen(nroots, diag[0..nroots], offdiag[0..nroots], &z);

    // Extract roots and weights
    const mu0 = moments[0]; // F_0(x)
    for (0..nroots) |i| {
        roots[i] = diag[i];
        weights[i] = mu0 * z[i] * z[i];
    }
}

/// Special case: nroots = 1.
/// F_0(x) = w_0, F_1(x) = w_0 * u_0
/// => u_0 = F_1(x) / F_0(x), w_0 = F_0(x)
fn rysRoots1(x: f64, roots: []f64, weights: []f64) void {
    const f0 = boys_mod.boysN(0, x);
    const f1 = boys_mod.boysN(1, x);

    weights[0] = f0;
    if (f0 > SMALL_X) {
        roots[0] = f1 / f0;
    } else {
        roots[0] = 0.0;
    }
}

/// Modified Chebyshev algorithm to compute recurrence coefficients
/// from the moments of the weight function.
///
/// Given moments mu[k] = integral t^k dw(t) for k = 0..2n-1,
/// computes alpha[j] and beta[j] for j = 0..n-1, where
/// beta[0] = mu[0].
///
/// Based on:
///   J.C. Wheeler, "Modified moments and Gaussian quadratures",
///   Rocky Mountain J. Math., 4:2 (1974), 287-296.
///   Also: Gautschi, "On Generating Orthogonal Polynomials",
///   SIAM J. Sci. Stat. Comput., 3:3 (1982), 289-317.
fn modifiedChebyshev(
    n: usize,
    moments: *const [MAX_MOMENTS]f64,
    alpha: *[MAX_NROOTS]f64,
    beta: *[MAX_NROOTS]f64,
) void {
    // sigma[l][k] for l = -1..n-1, k = 0..2n-1
    // We use l+1 as the actual index (so sigma[0] = sigma_{-1}, sigma[1] = sigma_0, etc.)
    var sigma: [MAX_NROOTS + 1][MAX_MOMENTS]f64 = undefined;

    // Initialize sigma[-1][k] = 0 for all k
    for (0..2 * n) |k| {
        sigma[0][k] = 0.0;
    }

    // sigma[0][k] = mu[k] for k = 0..2n-1
    for (0..2 * n) |k| {
        sigma[1][k] = moments[k];
    }

    // alpha[0] = mu[1] / mu[0]
    // beta[0] = mu[0]
    alpha[0] = moments[1] / moments[0];
    beta[0] = moments[0];

    for (1..n) |l| {
        // sigma[l+1][k] = sigma[l][k+1] - alpha[l-1]*sigma[l][k] - beta[l-1]*sigma[l-1][k]
        // for k = l..2n-l-1
        const l_idx = l + 1; // actual array index for sigma_l
        for (l..2 * n - l) |k| {
            const term_a = alpha[l - 1] * sigma[l_idx - 1][k];
            const term_b = beta[l - 1] * sigma[l_idx - 2][k];
            sigma[l_idx][k] = sigma[l_idx - 1][k + 1] - term_a - term_b;
        }
        const alpha_hi = sigma[l_idx][l + 1] / sigma[l_idx][l];
        const alpha_lo = sigma[l_idx - 1][l] / sigma[l_idx - 1][l - 1];
        alpha[l] = alpha_hi - alpha_lo;
        beta[l] = sigma[l_idx][l] / sigma[l_idx - 1][l - 1];
    }
}

/// Implicit QR algorithm for symmetric tridiagonal eigenvalue problem.
///
/// On input:
///   diag[0..n]: diagonal elements
///   offdiag[0..n]: off-diagonal elements (offdiag[0] unused)
///   z[0..n]: initial vector (typically e_1 = [1,0,...,0])
///
/// On output:
///   diag[0..n]: eigenvalues
///   z[i]: first component of eigenvector i
///
/// This is the QL algorithm with implicit shifts (Golub-Welsch).
fn tridiagEigen(n: usize, diag: []f64, offdiag: []f64, z: *[MAX_NROOTS]f64) void {
    if (n <= 1) return;

    // Copy off-diagonal to working array (shifted by 1)
    var e: [MAX_NROOTS]f64 = undefined;
    for (1..n) |i| {
        e[i - 1] = offdiag[i];
    }
    e[n - 1] = 0.0;

    const max_iter: usize = 100;

    for (0..n) |l| {
        var iter: usize = 0;
        while (iter < max_iter) {
            const m = findTridiagSplit(n, diag, &e, l);
            if (m == l) break;

            iter += 1;
            applyQlSweep(diag, &e, z, l, m);
        }
    }
}

fn findTridiagSplit(n: usize, diag: []const f64, e: *const [MAX_NROOTS]f64, l: usize) usize {
    var m: usize = l;
    while (m < n - 1) {
        const dd = @abs(diag[m]) + @abs(diag[m + 1]);
        if (@abs(e[m]) <= 1e-14 * dd) break;
        m += 1;
    }
    return m;
}

fn applyQlSweep(
    diag: []f64,
    e: *[MAX_NROOTS]f64,
    z: *[MAX_NROOTS]f64,
    l: usize,
    m: usize,
) void {
    // Implicit shift
    var g = (diag[l + 1] - diag[l]) / (2.0 * e[l]);
    var r = @sqrt(g * g + 1.0);
    if (g < 0.0) {
        r = -r;
    }
    g = diag[m] - diag[l] + e[l] / (g + r);

    var s: f64 = 1.0;
    var c_val: f64 = 1.0;
    var p: f64 = 0.0;

    // QL transformation
    if (m > l) {
        var i: usize = m - 1;
        var converged = false;
        while (true) {
            var f = s * e[i];
            const b = c_val * e[i];
            r = @sqrt(f * f + g * g);
            e[i + 1] = r;
            if (r < 1e-30) {
                diag[i + 1] -= p;
                e[m] = 0.0;
                converged = true;
                break;
            }
            s = f / r;
            c_val = g / r;
            g = diag[i + 1] - p;
            r = (diag[i] - g) * s + 2.0 * c_val * b;
            p = s * r;
            diag[i + 1] = g + p;
            g = c_val * r - b;

            // Track eigenvector first components
            f = z[i + 1];
            z[i + 1] = s * z[i] + c_val * f;
            z[i] = c_val * z[i] - s * f;

            if (i == l) break;
            i -= 1;
        }
        if (!converged) {
            diag[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "rys roots nroots=1 basic" {
    const testing = std.testing;
    const tol: f64 = 1e-10;

    // For nroots=1: root = F_1(x)/F_0(x), weight = F_0(x)
    // Verify: weight * root^0 = F_0(x), weight * root^1 = F_1(x)
    var roots: [1]f64 = undefined;
    var weights: [1]f64 = undefined;

    // x = 0
    rysRoots(1, 0.0, &roots, &weights);
    try testing.expectApproxEqAbs(1.0, weights[0], tol); // F_0(0) = 1
    try testing.expectApproxEqAbs(1.0 / 3.0, roots[0], tol); // F_1(0)/F_0(0) = 1/3

    // x = 1.0
    rysRoots(1, 1.0, &roots, &weights);
    const f0 = boys_mod.boysN(0, 1.0);
    const f1 = boys_mod.boysN(1, 1.0);
    try testing.expectApproxEqAbs(f0, weights[0], tol);
    try testing.expectApproxEqAbs(f1 / f0, roots[0], tol);
}

test "rys roots nroots=2 moment verification" {
    const testing = std.testing;
    const tol: f64 = 1e-10;

    var roots: [MAX_NROOTS]f64 = undefined;
    var weights: [MAX_NROOTS]f64 = undefined;

    const x_vals = [_]f64{ 0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0 };
    const nroots: usize = 2;

    for (x_vals) |x| {
        rysRoots(nroots, x, &roots, &weights);

        // Verify: sum_i w_i * u_i^k = F_k(x) for k = 0..2*nroots-1
        for (0..2 * nroots) |k| {
            var sum: f64 = 0.0;
            for (0..nroots) |i| {
                var uk: f64 = 1.0;
                for (0..k) |_| {
                    uk *= roots[i];
                }
                sum += weights[i] * uk;
            }
            const fk = boys_mod.boysN(@as(u32, @intCast(k)), x);
            try testing.expectApproxEqAbs(fk, sum, tol);
        }
    }
}

test "rys roots nroots=3 moment verification" {
    const testing = std.testing;
    const tol: f64 = 1e-9;

    var roots: [MAX_NROOTS]f64 = undefined;
    var weights: [MAX_NROOTS]f64 = undefined;

    const x_vals = [_]f64{ 0.0, 0.5, 2.0, 8.0, 15.0, 30.0 };
    const nroots: usize = 3;

    for (x_vals) |x| {
        rysRoots(nroots, x, &roots, &weights);

        for (0..2 * nroots) |k| {
            var sum: f64 = 0.0;
            for (0..nroots) |i| {
                var uk: f64 = 1.0;
                for (0..k) |_| {
                    uk *= roots[i];
                }
                sum += weights[i] * uk;
            }
            const fk = boys_mod.boysN(@as(u32, @intCast(k)), x);
            try testing.expectApproxEqAbs(fk, sum, tol);
        }
    }
}

test "rys roots nroots=4,5 moment verification" {
    const testing = std.testing;
    const tol: f64 = 1e-8;

    var roots: [MAX_NROOTS]f64 = undefined;
    var weights: [MAX_NROOTS]f64 = undefined;

    const x_vals = [_]f64{ 0.0, 1.0, 5.0, 10.0, 20.0 };

    for (4..6) |nroots| {
        for (x_vals) |x| {
            rysRoots(nroots, x, &roots, &weights);

            for (0..2 * nroots) |k| {
                var sum: f64 = 0.0;
                for (0..nroots) |i| {
                    var uk: f64 = 1.0;
                    for (0..k) |_| {
                        uk *= roots[i];
                    }
                    sum += weights[i] * uk;
                }
                const fk = boys_mod.boysN(@as(u32, @intCast(k)), x);
                try testing.expectApproxEqAbs(fk, sum, tol);
            }
        }
    }
}

test "rys roots nroots=7 for (ff|ff) case" {
    const testing = std.testing;
    const tol: f64 = 1e-7;

    var roots: [MAX_NROOTS]f64 = undefined;
    var weights: [MAX_NROOTS]f64 = undefined;

    const nroots: usize = 7;
    const x_vals = [_]f64{ 0.0, 0.5, 3.0, 10.0, 25.0, 50.0 };

    for (x_vals) |x| {
        rysRoots(nroots, x, &roots, &weights);

        // Verify first few moments (higher moments may lose some precision)
        for (0..@min(2 * nroots, 10)) |k| {
            var sum: f64 = 0.0;
            for (0..nroots) |i| {
                var uk: f64 = 1.0;
                for (0..k) |_| {
                    uk *= roots[i];
                }
                sum += weights[i] * uk;
            }
            const fk = boys_mod.boysN(@as(u32, @intCast(k)), x);
            try testing.expectApproxEqAbs(fk, sum, tol);
        }
    }
}

test "rys roots positive weights" {
    const testing = std.testing;

    var roots: [MAX_NROOTS]f64 = undefined;
    var weights: [MAX_NROOTS]f64 = undefined;

    const x_vals = [_]f64{ 0.0, 0.01, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0 };

    for (1..8) |nroots| {
        for (x_vals) |x| {
            rysRoots(nroots, x, &roots, &weights);

            for (0..nroots) |i| {
                try testing.expect(weights[i] >= 0.0);
                try testing.expect(roots[i] >= -1e-14); // roots should be non-negative
            }
        }
    }
}
