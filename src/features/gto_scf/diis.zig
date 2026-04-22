//! DIIS (Direct Inversion in the Iterative Subspace) for GTO-based SCF.
//!
//! Implements Pulay's DIIS extrapolation on the Fock matrix. The error
//! vector is the commutator e = FPS - SPF, which vanishes at convergence
//! (when [F, P] = 0 in the AO basis weighted by S).
//!
//! The DIIS equations minimise |Σ_i c_i e_i|² subject to Σ_i c_i = 1,
//! then the extrapolated Fock matrix is F_opt = Σ_i c_i F_i.
//!
//! References:
//!   P. Pulay, Chem. Phys. Lett. 73, 393 (1980)
//!   P. Pulay, J. Comput. Chem. 3, 556 (1982)

const std = @import("std");

/// DIIS accelerator for Fock-matrix extrapolation.
pub const GtoDiis = struct {
    /// Maximum number of stored (F, e) pairs.
    max_history: usize,
    /// Basis dimension (n×n matrices).
    n: usize,
    /// Stored Fock matrices (each n*n f64).
    fock_history: std.ArrayList([]f64),
    /// Stored error vectors (each n*n f64).
    error_history: std.ArrayList([]f64),
    /// Allocator.
    alloc: std.mem.Allocator,

    /// Create a DIIS accelerator.
    ///
    /// Parameters:
    ///   alloc: memory allocator
    ///   n: number of basis functions
    ///   max_history: maximum number of (F, e) pairs to keep (typically 5-8)
    pub fn init(alloc: std.mem.Allocator, n: usize, max_history: usize) GtoDiis {
        return .{
            .max_history = max_history,
            .n = n,
            .fock_history = .empty,
            .error_history = .empty,
            .alloc = alloc,
        };
    }

    /// Release all memory owned by the DIIS accelerator.
    pub fn deinit(self: *GtoDiis) void {
        for (self.fock_history.items) |buf| {
            self.alloc.free(buf);
        }
        self.fock_history.deinit(self.alloc);
        for (self.error_history.items) |buf| {
            self.alloc.free(buf);
        }
        self.error_history.deinit(self.alloc);
    }

    /// Record the current Fock matrix and density matrix, compute the
    /// error vector e = FPS - SPF, store (F, e), and return the
    /// extrapolated Fock matrix written into `f_out`.
    ///
    /// Parameters:
    ///   f_mat: current Fock matrix (n×n, row-major) — not modified
    ///   p_mat: current density matrix (n×n, row-major)
    ///   s_mat: overlap matrix (n×n, row-major)
    ///   f_out: buffer to receive the extrapolated Fock matrix (n×n)
    ///
    /// If fewer than 2 history vectors are available, `f_out` is simply
    /// set to `f_mat` (no extrapolation).
    pub fn extrapolate(
        self: *GtoDiis,
        f_mat: []const f64,
        p_mat: []const f64,
        s_mat: []const f64,
        f_out: []f64,
    ) !void {
        const n = self.n;
        const nn = n * n;

        // Compute error vector: e = FPS - SPF
        const err_vec = try self.alloc.alloc(f64, nn);
        errdefer self.alloc.free(err_vec);

        // We need two temporary n×n matrices for FP and SP
        const fp = try self.alloc.alloc(f64, nn);
        defer self.alloc.free(fp);
        const sp = try self.alloc.alloc(f64, nn);
        defer self.alloc.free(sp);

        matMul(n, f_mat, p_mat, fp); // FP
        matMul(n, s_mat, p_mat, sp); // SP

        // e = (FP)S - (SP)F
        // e_ij = Σ_k (FP)_ik S_kj - Σ_k (SP)_ik F_kj
        for (0..n) |i| {
            for (0..n) |j| {
                var fps: f64 = 0.0;
                var spf: f64 = 0.0;
                for (0..n) |k| {
                    fps += fp[i * n + k] * s_mat[k * n + j];
                    spf += sp[i * n + k] * f_mat[k * n + j];
                }
                err_vec[i * n + j] = fps - spf;
            }
        }

        // Store a copy of the current Fock matrix
        const f_copy = try self.alloc.alloc(f64, nn);
        errdefer self.alloc.free(f_copy);
        @memcpy(f_copy, f_mat);

        // Evict oldest entry if at capacity
        if (self.fock_history.items.len >= self.max_history) {
            const old_f = self.fock_history.orderedRemove(0);
            self.alloc.free(old_f);
            const old_e = self.error_history.orderedRemove(0);
            self.alloc.free(old_e);
        }

        try self.fock_history.append(self.alloc, f_copy);
        try self.error_history.append(self.alloc, err_vec);

        const m = self.fock_history.items.len;

        // Need at least 2 vectors for DIIS; otherwise pass through
        if (m < 2) {
            @memcpy(f_out, f_mat);
            return;
        }

        // Build the DIIS B matrix of size (m+1)×(m+1):
        //   B[i][j] = <e_i | e_j>   for i,j < m
        //   B[i][m] = B[m][i] = -1   (Lagrange constraint row/col)
        //   B[m][m] = 0
        const dim = m + 1;
        const b_mat = try self.alloc.alloc(f64, dim * dim);
        defer self.alloc.free(b_mat);
        const rhs = try self.alloc.alloc(f64, dim);
        defer self.alloc.free(rhs);

        for (0..m) |i| {
            for (0..m) |j| {
                b_mat[i * dim + j] = dotProduct(
                    nn,
                    self.error_history.items[i],
                    self.error_history.items[j],
                );
            }
            b_mat[i * dim + m] = -1.0;
            b_mat[m * dim + i] = -1.0;
            rhs[i] = 0.0;
        }
        b_mat[m * dim + m] = 0.0;
        rhs[m] = -1.0;

        // Solve the linear system B c = rhs
        const coeffs = try solveDiisSystem(self.alloc, dim, b_mat, rhs);
        defer self.alloc.free(coeffs);

        // Extrapolate: F_opt = Σ_i c_i F_i
        @memset(f_out, 0.0);
        for (0..m) |i| {
            const ci = coeffs[i];
            const fi = self.fock_history.items[i];
            for (0..nn) |idx| {
                f_out[idx] += ci * fi[idx];
            }
        }
    }

    /// Reset the DIIS history (e.g., when restarting SCF).
    pub fn reset(self: *GtoDiis) void {
        for (self.fock_history.items) |buf| {
            self.alloc.free(buf);
        }
        self.fock_history.clearRetainingCapacity();
        for (self.error_history.items) |buf| {
            self.alloc.free(buf);
        }
        self.error_history.clearRetainingCapacity();
    }

    /// Return the RMS of the most recent error vector (for monitoring).
    pub fn lastErrorRms(self: *const GtoDiis) f64 {
        if (self.error_history.items.len == 0) return 0.0;
        const e = self.error_history.items[self.error_history.items.len - 1];
        var sum: f64 = 0.0;
        for (e) |v| {
            sum += v * v;
        }
        return @sqrt(sum / @as(f64, @floatFromInt(e.len)));
    }
};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Row-major n×n matrix multiply: C = A × B.
fn matMul(n: usize, a: []const f64, b: []const f64, c: []f64) void {
    for (0..n) |i| {
        for (0..n) |j| {
            var s: f64 = 0.0;
            for (0..n) |k| {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/// Dot product of two vectors of length len.
fn dotProduct(len: usize, a: []const f64, b: []const f64) f64 {
    var s: f64 = 0.0;
    for (0..len) |i| {
        s += a[i] * b[i];
    }
    return s;
}

/// Solve a small dense linear system Ax = b using partial-pivot Gaussian
/// elimination. Returns the solution vector x (caller owns the memory).
///
/// Falls back to equal-weight coefficients if the matrix is singular.
fn solveDiisSystem(
    alloc: std.mem.Allocator,
    dim: usize,
    a_in: []const f64,
    b_in: []const f64,
) ![]f64 {
    // Work on copies since we modify in place
    const a = try alloc.alloc(f64, dim * dim);
    defer alloc.free(a);
    @memcpy(a, a_in);

    const x = try alloc.alloc(f64, dim);
    @memcpy(x, b_in);

    // Forward elimination with partial pivoting
    for (0..dim) |col| {
        // Find pivot
        var max_val: f64 = @abs(a[col * dim + col]);
        var max_row: usize = col;
        for (col + 1..dim) |row| {
            const v = @abs(a[row * dim + col]);
            if (v > max_val) {
                max_val = v;
                max_row = row;
            }
        }

        if (max_val < 1e-14) {
            // Singular — return equal weights for DIIS coefficients
            const m = dim - 1; // number of Fock matrices
            const w = 1.0 / @as(f64, @floatFromInt(m));
            for (0..m) |i| {
                x[i] = w;
            }
            x[m] = 0.0; // Lagrange multiplier
            return x;
        }

        // Swap rows
        if (max_row != col) {
            for (0..dim) |j| {
                const tmp = a[col * dim + j];
                a[col * dim + j] = a[max_row * dim + j];
                a[max_row * dim + j] = tmp;
            }
            const tmp = x[col];
            x[col] = x[max_row];
            x[max_row] = tmp;
        }

        // Eliminate below
        const pivot = a[col * dim + col];
        for (col + 1..dim) |row| {
            const factor = a[row * dim + col] / pivot;
            for (col..dim) |j| {
                a[row * dim + j] -= factor * a[col * dim + j];
            }
            x[row] -= factor * x[col];
        }
    }

    // Back substitution
    var col_idx: usize = dim;
    while (col_idx > 0) {
        col_idx -= 1;
        var s: f64 = x[col_idx];
        for (col_idx + 1..dim) |j| {
            s -= a[col_idx * dim + j] * x[j];
        }
        x[col_idx] = s / a[col_idx * dim + col_idx];
    }

    return x;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "DIIS matMul identity" {
    const n: usize = 3;
    // Identity matrix
    const eye = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var c: [9]f64 = undefined;
    matMul(n, &a, &eye, &c);
    for (0..9) |i| {
        try std.testing.expectApproxEqAbs(a[i], c[i], 1e-12);
    }
}

test "DIIS solver 2x2" {
    const alloc = std.testing.allocator;
    // Solve: [2 1; 1 3] x = [5; 7] → x = [8/5, 9/5] = [1.6, 1.8]
    const a = [_]f64{ 2, 1, 1, 3 };
    const b = [_]f64{ 5, 7 };
    const x = try solveDiisSystem(alloc, 2, &a, &b);
    defer alloc.free(x);
    try std.testing.expectApproxEqAbs(1.6, x[0], 1e-12);
    try std.testing.expectApproxEqAbs(1.8, x[1], 1e-12);
}

test "DIIS extrapolation with known Fock matrices" {
    const alloc = std.testing.allocator;
    const n: usize = 2;

    var diis = GtoDiis.init(alloc, n, 6);
    defer diis.deinit();

    // Use identity overlap matrix for simplicity
    const s_mat = [_]f64{ 1, 0, 0, 1 };

    // For S = I, the error is e = FP - PF = [F, P].
    const f1 = [_]f64{ -2.0, 0.1, 0.1, -1.0 };
    const p1 = [_]f64{ 1.5, 0.2, 0.2, 0.5 };
    var f_out: [4]f64 = undefined;

    // First call: only 1 vector, should pass through
    try diis.extrapolate(&f1, &p1, &s_mat, &f_out);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(f1[i], f_out[i], 1e-12);
    }

    // Second call with slightly different F and P
    const f2 = [_]f64{ -2.01, 0.09, 0.09, -1.01 };
    const p2 = [_]f64{ 1.52, 0.19, 0.19, 0.48 };
    try diis.extrapolate(&f2, &p2, &s_mat, &f_out);

    // Now we have 2 vectors, DIIS should produce some extrapolation.
    // Just verify it's finite and not NaN.
    for (0..4) |i| {
        try std.testing.expect(!std.math.isNan(f_out[i]));
        try std.testing.expect(!std.math.isInf(f_out[i]));
    }

    // The error RMS should be > 0 since [F, P] != 0
    const err_rms = diis.lastErrorRms();
    try std.testing.expect(err_rms > 0.0);
}

test "DIIS reset clears history" {
    const alloc = std.testing.allocator;
    var diis = GtoDiis.init(alloc, 2, 6);
    defer diis.deinit();

    const s_mat = [_]f64{ 1, 0, 0, 1 };
    const f1 = [_]f64{ -2.0, 0.1, 0.1, -1.0 };
    const p1 = [_]f64{ 1.0, 0.0, 0.0, 1.0 };
    var f_out: [4]f64 = undefined;

    try diis.extrapolate(&f1, &p1, &s_mat, &f_out);
    try std.testing.expectEqual(@as(usize, 1), diis.fock_history.items.len);

    diis.reset();
    try std.testing.expectEqual(@as(usize, 0), diis.fock_history.items.len);
    try std.testing.expectEqual(@as(usize, 0), diis.error_history.items.len);
}
