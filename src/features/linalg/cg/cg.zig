//! Band-by-band Conjugate Gradient eigensolver
//!
//! Solves for the lowest nbands eigenvalues/eigenvectors of H*psi = lambda*psi
//! by optimizing one band at a time using CG with 3-term Rayleigh-Ritz.
//!
//! Uses a 3-term recurrence: at each CG step, the subspace {ψ, d, ψ_prev}
//! is used for Rayleigh-Ritz optimization (where ψ_prev is from the previous
//! CG step). This gives much faster convergence than 2x2 {ψ, d} alone.
//! Falls back to 2x2 when ψ_prev is linearly dependent with {ψ, d}.
//!
//! After all bands, a final subspace diagonalization rotates eigenvectors
//! to properly handle degenerate states (prevents SCF charge oscillation).

const std = @import("std");
const math = @import("../../math/math.zig");
const linalg = @import("../linalg.zig");
const common = @import("../lobpcg/common.zig");
const blas = @import("../../../lib/linalg/blas.zig");

pub fn solve(
    alloc: std.mem.Allocator,
    op: common.Operator,
    diag: []const f64,
    nbands: usize,
    opts: common.Options,
) !linalg.EigenDecomp {
    const n = op.n;
    if (nbands == 0 or n == 0) {
        return linalg.EigenDecomp{
            .values = try alloc.alloc(f64, 0),
            .vectors = try alloc.alloc(math.Complex, 0),
            .n = n,
        };
    }
    if (diag.len != n) return error.InvalidMatrixSize;

    const max_iter = if (opts.max_iter == 0) 20 else opts.max_iter;
    const tol = if (opts.tol == 0.0) 1e-6 else opts.tol;

    // Output arrays
    const values = try alloc.alloc(f64, nbands);
    errdefer alloc.free(values);
    const vectors = try alloc.alloc(math.Complex, n * nbands);
    errdefer alloc.free(vectors);

    // Work buffers (reused across bands)
    const h_psi = try alloc.alloc(math.Complex, n);
    defer alloc.free(h_psi);
    const residual = try alloc.alloc(math.Complex, n);
    defer alloc.free(residual);
    const precond_buf = try alloc.alloc(math.Complex, n);
    defer alloc.free(precond_buf);
    const direction = try alloc.alloc(math.Complex, n);
    defer alloc.free(direction);
    const h_dir = try alloc.alloc(math.Complex, n);
    defer alloc.free(h_dir);
    const psi_old = try alloc.alloc(math.Complex, n);
    defer alloc.free(psi_old);
    const h_psi_old = try alloc.alloc(math.Complex, n);
    defer alloc.free(h_psi_old);
    // 3-term work buffers
    const psi_prev = try alloc.alloc(math.Complex, n);
    defer alloc.free(psi_prev);
    const h_psi_prev = try alloc.alloc(math.Complex, n);
    defer alloc.free(h_psi_prev);

    var seed: u64 = 0x6f3d3cbe;

    // Initialize vectors
    if (opts.init_vectors) |init_vecs| {
        const cols = if (opts.init_vectors_cols == 0) nbands else opts.init_vectors_cols;
        const copy_cols = @min(nbands, cols);
        if (init_vecs.len >= n * copy_cols) {
            @memcpy(vectors[0 .. n * copy_cols], init_vecs[0 .. n * copy_cols]);
        }
        if (copy_cols < nbands) {
            common.initRandomVectors(n, vectors[n * copy_cols ..], nbands - copy_cols, &seed);
        }
    } else {
        common.initRandomVectors(n, vectors, nbands, &seed);
    }

    // Orthonormalize all initial vectors
    try common.orthonormalizeAll(n, vectors, nbands, &seed);

    // Solve band by band
    for (0..nbands) |band| {
        const psi = common.column(vectors, n, band);

        // Re-orthogonalize against converged bands (previous bands changed)
        if (band > 0) {
            _ = common.orthonormalizeVector(n, psi, vectors, band);
        }

        // Apply H to psi
        try op.apply(op.ctx, psi, h_psi);

        // lambda = Re(<psi|H|psi>)
        var lambda = common.innerProduct(n, psi, h_psi).r;

        var prev_zr: f64 = 0.0;
        var has_prev_direction = false;

        for (0..max_iter) |_| {
            // Residual: r = H*psi - lambda*psi
            @memcpy(residual, h_psi);
            common.axpy(n, residual, psi, -lambda);

            // Orthogonalize residual against converged bands (0..band-1)
            orthogonalize(n, residual, vectors, band);

            // Check residual convergence
            const res_norm = common.vectorNorm(n, residual);
            if (res_norm < tol) break;

            // Precondition: z[i] = r[i] / (diag[i] - lambda)
            common.precondition(n, diag, lambda, residual, precond_buf);

            // Orthogonalize z against converged bands + current band
            orthogonalize(n, precond_buf, vectors, band + 1);

            // CG direction: d = z + beta * d_prev (Polak-Ribiere)
            const zr = common.innerProduct(n, precond_buf, residual).r;
            if (has_prev_direction and prev_zr > 1e-30) {
                const beta = @max(0.0, zr / prev_zr);
                for (0..n) |i| {
                    direction[i] = math.complex.add(precond_buf[i], math.complex.scale(direction[i], beta));
                }
            } else {
                @memcpy(direction, precond_buf);
            }
            prev_zr = zr;

            // Orthogonalize direction against converged bands + current band, then normalize
            orthogonalize(n, direction, vectors, band + 1);
            const d_norm = common.vectorNorm(n, direction);
            if (d_norm < 1e-14) break;
            scaleVec(n, direction, 1.0 / d_norm);

            // Apply H to direction
            try op.apply(op.ctx, direction, h_dir);

            // ========== Rayleigh-Ritz (3x3 or 2x2) ==========
            var lambda_new: f64 = undefined;
            var c1: math.Complex = undefined;
            var c2: math.Complex = undefined;
            var c3 = math.complex.init(0.0, 0.0);
            var use_3term = false;

            // Try 3-term recurrence: include ψ_{k-1} from previous CG step
            if (has_prev_direction) {
                // psi_old = ψ_{k-1}, h_psi_old = H·ψ_{k-1}
                // Copy to work buffers for orthogonalization
                @memcpy(psi_prev, psi_old);
                @memcpy(h_psi_prev, h_psi_old);

                // Orthogonalize p against ψ (track H·p simultaneously)
                const ov_psi = common.innerProduct(n, psi, psi_prev);
                zaxpyComplex(n, psi_prev, psi, math.complex.init(-ov_psi.r, -ov_psi.i));
                zaxpyComplex(n, h_psi_prev, h_psi, math.complex.init(-ov_psi.r, -ov_psi.i));

                // Orthogonalize p against d (track H·p simultaneously)
                const ov_d = common.innerProduct(n, direction, psi_prev);
                zaxpyComplex(n, psi_prev, direction, math.complex.init(-ov_d.r, -ov_d.i));
                zaxpyComplex(n, h_psi_prev, h_dir, math.complex.init(-ov_d.r, -ov_d.i));

                // Normalize p and H·p
                const p_norm = common.vectorNorm(n, psi_prev);
                if (p_norm > 1e-6) {
                    scaleVec(n, psi_prev, 1.0 / p_norm);
                    scaleVec(n, h_psi_prev, 1.0 / p_norm);
                    use_3term = true;
                }
            }

            if (use_3term) rr3: {
                // 3x3 Rayleigh-Ritz in {ψ, d, p}
                const a11 = lambda;
                const a22 = common.innerProduct(n, direction, h_dir).r;
                const a33 = common.innerProduct(n, psi_prev, h_psi_prev).r;
                const a12 = common.innerProduct(n, psi, h_dir);
                const a13 = common.innerProduct(n, psi, h_psi_prev);
                const a23 = common.innerProduct(n, direction, h_psi_prev);

                // Build 3x3 Hermitian matrix (column-major)
                var h3: [9]math.Complex = undefined;
                h3[0] = math.complex.init(a11, 0.0); // (0,0)
                h3[1] = math.complex.init(a12.r, -a12.i); // (1,0) = conj(a12)
                h3[2] = math.complex.init(a13.r, -a13.i); // (2,0) = conj(a13)
                h3[3] = a12; // (0,1)
                h3[4] = math.complex.init(a22, 0.0); // (1,1)
                h3[5] = math.complex.init(a23.r, -a23.i); // (2,1) = conj(a23)
                h3[6] = a13; // (0,2)
                h3[7] = a23; // (1,2)
                h3[8] = math.complex.init(a33, 0.0); // (2,2)

                const sub_eig = common.hermitianEigenDecompSmall(alloc, 3, &h3) catch {
                    use_3term = false;
                    break :rr3;
                };
                defer alloc.free(sub_eig.values);
                defer alloc.free(sub_eig.vectors);

                lambda_new = sub_eig.values[0]; // smallest eigenvalue
                c1 = sub_eig.vectors[0]; // first column, row 0
                c2 = sub_eig.vectors[1]; // first column, row 1
                c3 = sub_eig.vectors[2]; // first column, row 2
            }

            if (!use_3term) {
                // 2x2 Rayleigh-Ritz in {ψ, d}
                const a11 = lambda;
                const a22 = common.innerProduct(n, direction, h_dir).r;
                const a12 = common.innerProduct(n, psi, h_dir);

                const avg = 0.5 * (a11 + a22);
                const diff = 0.5 * (a11 - a22);
                const off_sq = a12.r * a12.r + a12.i * a12.i;
                const disc = @sqrt(diff * diff + off_sq);
                lambda_new = avg - disc;

                if (off_sq > 1e-30) {
                    c1 = math.complex.init(-a12.r, -a12.i);
                    c2 = math.complex.init(a11 - lambda_new, 0.0);
                } else {
                    if (a11 <= a22) {
                        c1 = math.complex.init(1.0, 0.0);
                        c2 = math.complex.init(0.0, 0.0);
                    } else {
                        c1 = math.complex.init(0.0, 0.0);
                        c2 = math.complex.init(1.0, 0.0);
                    }
                }
                c3 = math.complex.init(0.0, 0.0);

                // Normalize coefficients
                const c_norm = @sqrt(c1.r * c1.r + c1.i * c1.i + c2.r * c2.r + c2.i * c2.i);
                if (c_norm > 1e-30) {
                    c1 = math.complex.scale(c1, 1.0 / c_norm);
                    c2 = math.complex.scale(c2, 1.0 / c_norm);
                }
            }

            // Save old psi and H*psi (for next iteration's 3-term and update formula)
            @memcpy(psi_old, psi);
            @memcpy(h_psi_old, h_psi);
            has_prev_direction = true;

            // Update: psi_new = c1*psi_old + c2*d + c3*p
            for (0..n) |i| {
                var v = math.complex.add(
                    math.complex.mul(c1, psi_old[i]),
                    math.complex.mul(c2, direction[i]),
                );
                if (use_3term) {
                    v = math.complex.add(v, math.complex.mul(c3, psi_prev[i]));
                }
                psi[i] = v;
            }

            // Update: H*psi_new = c1*H*psi_old + c2*H*d + c3*H*p
            for (0..n) |i| {
                var v = math.complex.add(
                    math.complex.mul(c1, h_psi_old[i]),
                    math.complex.mul(c2, h_dir[i]),
                );
                if (use_3term) {
                    v = math.complex.add(v, math.complex.mul(c3, h_psi_prev[i]));
                }
                h_psi[i] = v;
            }

            // Re-normalize psi (and H*psi accordingly)
            const psi_norm = common.vectorNorm(n, psi);
            if (psi_norm > 1e-14) {
                scaleVec(n, psi, 1.0 / psi_norm);
                scaleVec(n, h_psi, 1.0 / psi_norm);
            }

            // Check eigenvalue convergence (eigenvalue converges as ||r||^2/gap,
            // much faster than residual norm itself)
            const eval_change = @abs(lambda_new - lambda);
            lambda = lambda_new;
            if (eval_change < tol) break;
        }

        values[band] = lambda;

        // Re-orthogonalize final vector against previous bands
        // (eigenvalues will be corrected by subspace diagonalization at the end)
        if (band > 0) {
            _ = common.orthonormalizeVector(n, psi, vectors, band);
        }
    }

    // Subspace diagonalization: rotate all bands to diagonalize H in {ψ_0..ψ_{nbands-1}}
    // This is crucial for degenerate bands — band-by-band CG picks arbitrary linear
    // combinations each SCF step, causing density oscillation. The rotation fixes this.
    {
        // Compute H*ψ for all bands
        const h_vectors = try alloc.alloc(math.Complex, n * nbands);
        defer alloc.free(h_vectors);
        for (0..nbands) |b| {
            const psi_b = common.columnConst(vectors, n, b);
            const h_b = common.column(h_vectors, n, b);
            try op.apply(op.ctx, psi_b, h_b);
        }

        // Build projected Hamiltonian: H_sub[i,j] = <ψ_i|H|ψ_j>
        const h_sub = try alloc.alloc(math.Complex, nbands * nbands);
        defer alloc.free(h_sub);
        common.buildProjected(n, vectors, h_vectors, h_sub, nbands);

        // Diagonalize (returns eigenvalues sorted ascending and eigenvectors as columns)
        var sub_eig = try common.hermitianEigenDecompSmall(alloc, nbands, h_sub);
        defer sub_eig.deinit(alloc);

        // Rotate vectors: ψ_new = ψ_old * U (where U = sub_eig.vectors)
        const rotated = try alloc.alloc(math.Complex, n * nbands);
        defer alloc.free(rotated);
        for (0..nbands) |b| {
            const out_col = common.column(rotated, n, b);
            const coeffs = common.columnConst(sub_eig.vectors, nbands, b);
            common.combineColumns(n, vectors, nbands, coeffs, out_col);
        }
        @memcpy(vectors, rotated);

        // Update eigenvalues from subspace diagonalization
        @memcpy(values, sub_eig.values);
    }

    return linalg.EigenDecomp{
        .values = values,
        .vectors = vectors,
        .n = n,
    };
}

/// Orthogonalize v against columns 0..m-1 of basis (without normalization).
fn orthogonalize(n: usize, v: []math.Complex, basis: []const math.Complex, m: usize) void {
    const blas_v: []blas.Complex = @ptrCast(v[0..n]);
    for (0..m) |j| {
        const bj = common.columnConst(basis, n, j);
        const dot = common.innerProduct(n, bj, v);
        const neg_dot = blas.Complex.init(-dot.r, -dot.i);
        const blas_bj: []const blas.Complex = @ptrCast(bj);
        blas.zaxpy(neg_dot, blas_bj, blas_v);
    }
}

/// Complex AXPY: y += alpha * x (alpha is complex)
fn zaxpyComplex(n: usize, y: []math.Complex, x: []const math.Complex, alpha: math.Complex) void {
    if (n == 0) return;
    const blas_alpha = blas.Complex.init(alpha.r, alpha.i);
    const blas_x: []const blas.Complex = @ptrCast(x[0..n]);
    const blas_y: []blas.Complex = @ptrCast(y[0..n]);
    blas.zaxpy(blas_alpha, blas_x, blas_y);
}

/// Scale vector: v *= alpha
fn scaleVec(n: usize, v: []math.Complex, alpha: f64) void {
    if (n == 0) return;
    const blas_alpha = blas.Complex.init(alpha, 0.0);
    const blas_v: []blas.Complex = @ptrCast(v[0..n]);
    blas.zscal(blas_alpha, blas_v);
}

// ============== Tests ==============

test "CG basic 3x3 diagonal" {
    const allocator = std.testing.allocator;

    const TestCtx = struct {
        diag_vals: [3]f64,

        pub fn apply(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            for (0..3) |i| {
                y[i] = math.complex.scale(x[i], self.diag_vals[i]);
            }
        }
    };

    var ctx = TestCtx{ .diag_vals = .{ 1.0, 3.0, 2.0 } };
    const op = common.Operator{
        .n = 3,
        .ctx = @ptrCast(&ctx),
        .apply = TestCtx.apply,
    };

    // Use zero diagonal for preconditioner (identity preconditioning).
    // Using the exact matrix diagonal as preconditioner causes z = psi
    // (gradient cancellation), which is not an issue in real DFT where
    // the preconditioner (kinetic energy) differs from the full Hamiltonian.
    const zero_diag = [_]f64{ 0.0, 0.0, 0.0 };
    var result = try solve(
        allocator,
        op,
        &zero_diag,
        2,
        .{ .max_iter = 50, .tol = 1e-10 },
    );
    defer result.deinit(allocator);

    // Should find eigenvalues 1.0 and 2.0
    try std.testing.expectApproxEqAbs(result.values[0], 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(result.values[1], 2.0, 1e-6);
}

test "CG 5x5 non-diagonal" {
    const allocator = std.testing.allocator;

    const TestCtx = struct {
        pub fn apply(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
            _ = ctx_ptr;
            const nn = 5;
            // H = diag(1,2,3,4,5) + symmetric off-diagonal perturbation
            const h = [nn][nn]f64{
                .{ 1.0, 0.5, 0.0, 0.0, 0.0 },
                .{ 0.5, 2.0, 0.5, 0.0, 0.0 },
                .{ 0.0, 0.5, 3.0, 0.5, 0.0 },
                .{ 0.0, 0.0, 0.5, 4.0, 0.5 },
                .{ 0.0, 0.0, 0.0, 0.5, 5.0 },
            };
            for (0..nn) |i| {
                var sum = math.complex.init(0.0, 0.0);
                for (0..nn) |j| {
                    sum = math.complex.add(sum, math.complex.scale(x[j], h[i][j]));
                }
                y[i] = sum;
            }
        }
    };

    var ctx = TestCtx{};
    // Use zero diagonal (identity preconditioning) for robust convergence in tests
    const diag_arr = [_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const op = common.Operator{
        .n = 5,
        .ctx = @ptrCast(&ctx),
        .apply = TestCtx.apply,
    };

    var result = try solve(
        allocator,
        op,
        &diag_arr,
        3,
        .{ .max_iter = 100, .tol = 1e-10 },
    );
    defer result.deinit(allocator);

    // For tridiagonal with diag=[1..5] and off-diag=0.5:
    // exact eigenvalues: 0.7746, 1.9766, 3.0000, 4.0234, 5.2254
    try std.testing.expect(result.values[0] < result.values[1]);
    try std.testing.expect(result.values[1] < result.values[2]);
    try std.testing.expectApproxEqAbs(result.values[0], 0.7746, 1e-3);
    try std.testing.expectApproxEqAbs(result.values[1], 1.9766, 1e-3);
    try std.testing.expectApproxEqAbs(result.values[2], 3.0, 1e-3);
}
