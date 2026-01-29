//! Density matrix construction for Restricted Hartree-Fock.
//!
//! For RHF with n_occ occupied orbitals (each doubly occupied):
//!   P_μν = 2 × Σ_{a=1}^{n_occ} C_μa × C_νa
//!
//! where C is the MO coefficient matrix from solving FC = SCε.
//! C is stored in column-major order (LAPACK convention):
//!   C[μ + a*n] = coefficient of basis function μ in MO a.

const std = @import("std");

/// Build the density matrix P from MO coefficients C.
///
/// Parameters:
///   alloc: allocator for the result
///   n: number of basis functions
///   n_occ: number of occupied orbitals (for RHF, n_electrons / 2)
///   c: MO coefficient matrix, column-major n×n (eigenvectors from LAPACK)
///
/// Returns: row-major n×n density matrix P where
///   P[μ*n + ν] = 2 × Σ_{a=0}^{n_occ-1} C[μ + a*n] × C[ν + a*n]
pub fn buildDensityMatrix(
    alloc: std.mem.Allocator,
    n: usize,
    n_occ: usize,
    c: []const f64,
) ![]f64 {
    std.debug.assert(c.len == n * n);
    std.debug.assert(n_occ <= n);

    const p = try alloc.alloc(f64, n * n);
    for (0..n) |mu| {
        for (0..n) |nu| {
            var sum: f64 = 0.0;
            for (0..n_occ) |a| {
                // Column-major: C[mu, a] = c[mu + a * n]
                sum += c[mu + a * n] * c[nu + a * n];
            }
            p[mu * n + nu] = 2.0 * sum;
        }
    }
    return p;
}

/// Update the density matrix in-place from MO coefficients.
/// Same as buildDensityMatrix but writes to an existing buffer.
pub fn updateDensityMatrix(
    n: usize,
    n_occ: usize,
    c: []const f64,
    p: []f64,
) void {
    std.debug.assert(c.len == n * n);
    std.debug.assert(p.len == n * n);
    std.debug.assert(n_occ <= n);

    for (0..n) |mu| {
        for (0..n) |nu| {
            var sum: f64 = 0.0;
            for (0..n_occ) |a| {
                sum += c[mu + a * n] * c[nu + a * n];
            }
            p[mu * n + nu] = 2.0 * sum;
        }
    }
}

/// Compute the RMS difference between two density matrices.
/// Used for SCF convergence check.
pub fn densityRmsDiff(n: usize, p_new: []const f64, p_old: []const f64) f64 {
    std.debug.assert(p_new.len == n * n);
    std.debug.assert(p_old.len == n * n);

    var sum: f64 = 0.0;
    for (0..n * n) |i| {
        const d = p_new[i] - p_old[i];
        sum += d * d;
    }
    return @sqrt(sum / @as(f64, @floatFromInt(n * n)));
}

test "density matrix 1x1" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // 1 basis function, 1 occupied orbital: P = 2 * C^2
    const c = [_]f64{0.8};
    const p = try buildDensityMatrix(alloc, 1, 1, &c);
    defer alloc.free(p);
    try testing.expectApproxEqAbs(p[0], 2.0 * 0.8 * 0.8, 1e-12);
}

test "density matrix 2x2 H2" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // 2 basis functions, 1 occupied orbital (2 electrons in H2)
    // Column-major C: eigenvector 0 is c[0], c[1]
    //                  eigenvector 1 is c[2], c[3]
    // Bonding orbital: (1/sqrt(2), 1/sqrt(2))
    const s = 1.0 / @sqrt(2.0);
    const c = [_]f64{ s, s, s, -s }; // col-major: col0 = (s,s), col1 = (s,-s)

    const p = try buildDensityMatrix(alloc, 2, 1, &c);
    defer alloc.free(p);

    // P = 2 * [s*s, s*s; s*s, s*s] = [1, 1; 1, 1]
    try testing.expectApproxEqAbs(p[0], 1.0, 1e-12); // P[0,0]
    try testing.expectApproxEqAbs(p[1], 1.0, 1e-12); // P[0,1]
    try testing.expectApproxEqAbs(p[2], 1.0, 1e-12); // P[1,0]
    try testing.expectApproxEqAbs(p[3], 1.0, 1e-12); // P[1,1]
}

test "density RMS diff" {
    const p1 = [_]f64{ 1.0, 0.5, 0.5, 1.0 };
    const p2 = [_]f64{ 1.1, 0.5, 0.5, 0.9 };
    const rms = densityRmsDiff(2, &p1, &p2);
    // diff = (0.1, 0, 0, 0.1), sum of squares = 0.02, rms = sqrt(0.02/4) = sqrt(0.005)
    try std.testing.expectApproxEqAbs(rms, @sqrt(0.005), 1e-12);
}
