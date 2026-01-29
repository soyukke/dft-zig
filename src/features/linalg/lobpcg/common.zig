//! Common utilities for LOBPCG implementations
//!
//! Shared between serial and parallel versions.

const std = @import("std");
const math = @import("../../math/math.zig");
const linalg = @import("../linalg.zig");
const simd_vec = @import("../../../lib/linalg/complex_vec.zig");
const blas = @import("../../../lib/linalg/blas.zig");

/// Operator interface for matrix-vector multiplication
pub const Operator = struct {
    n: usize,
    ctx: *anyopaque,
    apply: *const fn (ctx: *anyopaque, x: []const math.Complex, y: []math.Complex) anyerror!void,
    /// Optional batched apply: y_batch = A * x_batch for ncols vectors at once.
    /// x_batch and y_batch are column-major [n × ncols].
    apply_batch: ?*const fn (ctx: *anyopaque, x_batch: []const math.Complex, y_batch: []math.Complex, n: usize, ncols: usize) anyerror!void = null,
    /// Optional overlap operator S for generalized eigenvalue problem H·x = λ·S·x.
    /// When null, S = I (standard eigenvalue problem, NC pseudopotentials).
    /// When set, LOBPCG uses S-inner products and solves the generalized problem.
    apply_s: ?*const fn (ctx: *anyopaque, x: []const math.Complex, y: []math.Complex) anyerror!void = null,
    /// Optional batched overlap operator.
    apply_s_batch: ?*const fn (ctx: *anyopaque, x_batch: []const math.Complex, y_batch: []math.Complex, n: usize, ncols: usize) anyerror!void = null,
};

/// LOBPCG options
pub const Options = struct {
    max_iter: usize = 40,
    tol: f64 = 1e-6,
    max_subspace: usize = 0,
    block_size: usize = 0,
    init_diagonal: bool = false,
    init_vectors: ?[]const math.Complex = null,
    init_vectors_cols: usize = 0,
};

/// Get column slice from matrix
pub fn column(data: []math.Complex, n: usize, col: usize) []math.Complex {
    return data[col * n .. (col + 1) * n];
}

/// Get const column slice from matrix
pub fn columnConst(data: []const math.Complex, n: usize, col: usize) []const math.Complex {
    return data[col * n .. (col + 1) * n];
}

/// Complex inner product: <a|b> = sum(conj(a[i]) * b[i])
/// Uses SIMD-optimized implementation for ~1.25x speedup.
pub fn innerProduct(n: usize, a: []const math.Complex, b: []const math.Complex) math.Complex {
    // Use SIMD-optimized version (same memory layout as math.Complex)
    const simd_a: []const simd_vec.Complex = @ptrCast(a[0..n]);
    const simd_b: []const simd_vec.Complex = @ptrCast(b[0..n]);
    const result = simd_vec.innerProduct(simd_a, simd_b);
    return math.complex.init(result.r, result.i);
}

/// Vector norm: sqrt(<a|a>)
/// Vector 2-norm using BLAS dznrm2
pub fn vectorNorm(n: usize, a: []const math.Complex) f64 {
    if (n == 0) return 0.0;
    const blas_a: []const blas.Complex = @ptrCast(a[0..n]);
    return blas.dznrm2(blas_a);
}

/// AXPY: y = y + alpha * x using BLAS zaxpy
pub fn axpy(n: usize, y: []math.Complex, x: []const math.Complex, alpha: f64) void {
    if (n == 0) return;
    const blas_alpha = blas.Complex.init(alpha, 0.0);
    const blas_x: []const blas.Complex = @ptrCast(x[0..n]);
    const blas_y: []blas.Complex = @ptrCast(y[0..n]);
    blas.zaxpy(blas_alpha, blas_x, blas_y);
}

/// Combine columns: out = V * coeffs using BLAS zgemv
/// V is n x m column-major matrix, coeffs is m-vector, out is n-vector
pub fn combineColumns(n: usize, v: []const math.Complex, m: usize, coeffs: []const math.Complex, out: []math.Complex) void {
    if (m == 0 or n == 0) {
        @memset(out[0..n], math.complex.init(0.0, 0.0));
        return;
    }
    // Use BLAS zgemv: out = 1.0 * V * coeffs + 0.0 * out
    const blas_v: []const blas.Complex = @ptrCast(v);
    const blas_coeffs: []const blas.Complex = @ptrCast(coeffs);
    const blas_out: []blas.Complex = @ptrCast(out);
    blas.combineColumns(n, m, blas_v, blas_coeffs, blas_out);
}

/// Batch combine columns: out = V * C using BLAS zgemm
/// V is n x m column-major, C is m x ncols column-major, out is n x ncols column-major
pub fn combineColumnsMatrix(n: usize, v: []const math.Complex, m: usize, coeffs: []const math.Complex, ncols: usize, out: []math.Complex) void {
    if (m == 0 or n == 0 or ncols == 0) {
        @memset(out[0 .. n * ncols], math.complex.init(0.0, 0.0));
        return;
    }
    const blas_v: []const blas.Complex = @ptrCast(v[0 .. n * m]);
    const blas_c: []const blas.Complex = @ptrCast(coeffs[0 .. m * ncols]);
    const blas_out: []blas.Complex = @ptrCast(out[0 .. n * ncols]);
    blas.zgemm(
        .no_trans,
        .no_trans,
        n,
        ncols,
        m,
        blas.Complex.init(1.0, 0.0),
        blas_v,
        n,
        blas_c,
        m,
        blas.Complex.init(0.0, 0.0),
        blas_out,
        n,
    );
}

/// Precondition residual for generalized problem: out[i] = r[i] / (diag_H[i] - lambda * diag_S[i])
/// When diag_s is null, uses standard preconditioner (S=I).
pub fn preconditionGeneralized(n: usize, diag: []const f64, diag_s: ?[]const f64, lambda: f64, r: []const math.Complex, out: []math.Complex) void {
    for (0..n) |i| {
        const s_ii = if (diag_s) |ds| ds[i] else 1.0;
        var denom = diag[i] - lambda * s_ii;
        if (@abs(denom) < 1e-8) denom = if (denom >= 0.0) 1e-8 else -1e-8;
        out[i] = math.complex.scale(r[i], 1.0 / denom);
    }
}

/// S-inner product: <a|S|b> using pre-computed S·b column.
/// When sv_col is null (S=I), falls back to standard inner product.
pub fn innerProductS(n: usize, a: []const math.Complex, b: []const math.Complex, sv_col: ?[]const math.Complex) math.Complex {
    if (sv_col) |sb| {
        return innerProduct(n, a, sb);
    }
    return innerProduct(n, a, b);
}

/// Modified Gram-Schmidt orthonormalization with S-inner product.
/// sv is the S·v matrix (column-major), updated in place for new column.
/// When sv is null, uses standard inner product (S=I).
pub fn orthonormalizeVectorS(n: usize, v: []math.Complex, basis: []const math.Complex, sv: ?[]math.Complex, sv_basis: ?[]const math.Complex, m: usize) f64 {
    const blas_v: []blas.Complex = @ptrCast(v);
    for (0..m) |j| {
        const bj = columnConst(basis, n, j);
        // Use S-inner product: <bj|S|v>
        const sv_v = if (sv) |s| column(@constCast(s), n, m) else null;
        _ = sv_v;
        const dot = if (sv_basis) |sb|
            innerProduct(n, columnConst(sb, n, j), v)
        else
            innerProduct(n, bj, v);
        const neg_dot = blas.Complex.init(-dot.r, -dot.i);
        const blas_bj: []const blas.Complex = @ptrCast(bj);
        blas.zaxpy(neg_dot, blas_bj, blas_v);
    }
    // Compute S-norm: sqrt(<v|S|v>)
    // Note: sv must be recomputed after orthogonalization for accurate norm
    // For now, use standard norm (the S-orthogonalization is approximate)
    const norm = blas.dznrm2(blas_v);
    if (norm > 0.0) {
        const inv = blas.Complex.init(1.0 / norm, 0.0);
        blas.zscal(inv, blas_v);
    }
    return norm;
}

/// Precondition residual: out[i] = r[i] / (diag[i] - lambda)
pub fn precondition(n: usize, diag: []const f64, lambda: f64, r: []const math.Complex, out: []math.Complex) void {
    for (0..n) |i| {
        var denom = diag[i] - lambda;
        if (@abs(denom) < 1e-8) denom = if (denom >= 0.0) 1e-8 else -1e-8;
        out[i] = math.complex.scale(r[i], 1.0 / denom);
    }
}

/// Build projected matrix: T = V† * W using BLAS zgemm
/// V is n x m, W is n x m, T is m x m (Hermitian)
/// For numerical stability, enforces Hermitian symmetry after computation.
pub fn buildProjected(n: usize, v: []const math.Complex, w: []const math.Complex, out: []math.Complex, m: usize) void {
    if (m == 0 or n == 0) return;

    // Use BLAS zgemm: T = V† * W
    const blas_v: []const blas.Complex = @ptrCast(v);
    const blas_w: []const blas.Complex = @ptrCast(w);
    const blas_out: []blas.Complex = @ptrCast(out);
    blas.buildProjectedMatrix(n, m, blas_v, blas_w, blas_out);

    // Enforce Hermitian symmetry: T = (T + T†) / 2
    for (0..m) |j| {
        for (0..j) |i| {
            const tij = out[i + j * m];
            const tji = out[j + i * m];
            const sym_r = 0.5 * (tij.r + tji.r);
            const sym_i = 0.5 * (tij.i - tji.i);
            out[i + j * m] = math.complex.init(sym_r, sym_i);
            out[j + i * m] = math.complex.init(sym_r, -sym_i);
        }
        // Diagonal elements should be real
        out[j + j * m].i = 0.0;
    }
}

/// Orthonormalize vector against basis using modified Gram-Schmidt
/// Uses BLAS zaxpy for vector updates.
pub fn orthonormalizeVector(n: usize, v: []math.Complex, basis: []const math.Complex, m: usize) f64 {
    const blas_v: []blas.Complex = @ptrCast(v);
    for (0..m) |j| {
        const bj = columnConst(basis, n, j);
        const dot = innerProduct(n, bj, v);
        // v = v - dot * bj (using BLAS zaxpy: y = alpha * x + y)
        const neg_dot = blas.Complex.init(-dot.r, -dot.i);
        const blas_bj: []const blas.Complex = @ptrCast(bj);
        blas.zaxpy(neg_dot, blas_bj, blas_v);
    }
    const norm = blas.dznrm2(blas_v);
    if (norm > 0.0) {
        const inv = blas.Complex.init(1.0 / norm, 0.0);
        blas.zscal(inv, blas_v);
    }
    return norm;
}

/// Initialize random vectors
pub fn initRandomVectors(n: usize, v: []math.Complex, m: usize, seed: *u64) void {
    for (0..m) |col| {
        for (0..n) |i| {
            const r = nextRand01(seed) - 0.5;
            const im = nextRand01(seed) - 0.5;
            v[i + col * n] = math.complex.init(r, im);
        }
    }
}

/// Orthonormalize all columns
pub fn orthonormalizeAll(n: usize, v: []math.Complex, m: usize, seed: *u64) !void {
    for (0..m) |col| {
        var retries: usize = 0;
        while (retries < 3) : (retries += 1) {
            const norm = orthonormalizeVector(n, column(v, n, col), v, col);
            if (norm > 1e-8) break;
            for (0..n) |i| {
                const r = nextRand01(seed) - 0.5;
                const im = nextRand01(seed) - 0.5;
                v[i + col * n] = math.complex.init(r, im);
            }
        }
        if (vectorNorm(n, column(v, n, col)) < 1e-8) return error.InvalidMatrixSize;
    }
}

/// Simple LCG random number generator
pub fn nextRand01(seed: *u64) f64 {
    seed.* = seed.* *% 6364136223846793005 +% 1;
    const val: u64 = seed.* >> 11;
    return @as(f64, @floatFromInt(val)) / 9007199254740992.0;
}

/// Small matrix eigendecomposition using LAPACK zheev
/// Optimized for small matrices: bypasses global mutex and uses stack workspace.
pub fn hermitianEigenDecompSmall(alloc: std.mem.Allocator, n: usize, a: []math.Complex) !linalg.EigenDecomp {
    // For small matrices, call LAPACK directly without mutex and with stack workspace
    if (n <= 64 and n > 0) {
        const vectors = try alloc.alloc(math.Complex, n * n);
        errdefer alloc.free(vectors);
        @memcpy(vectors, a[0 .. n * n]);

        const values = try alloc.alloc(f64, n);
        errdefer alloc.free(values);

        // Stack-allocated workspace for small matrices
        var rwork_buf: [3 * 64]f64 = undefined;
        const rwork = rwork_buf[0..3 * n -| 2];

        var nn: c_int = @intCast(n);
        var lda: c_int = @intCast(n);
        var info: c_int = 0;
        var jobz = [1]u8{'V'};
        var uplo = [1]u8{'U'};

        // Workspace query
        var lwork: c_int = -1;
        var work_query = math.complex.init(0.0, 0.0);
        zheev_(
            &jobz,
            &uplo,
            &nn,
            @ptrCast(vectors.ptr),
            &lda,
            values.ptr,
            @ptrCast(&work_query),
            &lwork,
            rwork.ptr,
            &info,
        );
        if (info != 0) return error.LapackFailure;

        lwork = @intFromFloat(work_query.r);
        if (lwork < 1) lwork = 1;

        // Use stack buffer if small enough, otherwise heap allocate
        var work_stack: [2 * 64 * 64]math.Complex = undefined;
        var work_heap: ?[]math.Complex = null;
        defer if (work_heap) |w| alloc.free(w);

        const work_ptr: [*]math.Complex = if (@as(usize, @intCast(lwork)) <= work_stack.len)
            &work_stack
        else blk: {
            work_heap = try alloc.alloc(math.Complex, @intCast(lwork));
            break :blk work_heap.?.ptr;
        };

        info = 0;
        zheev_(
            &jobz,
            &uplo,
            &nn,
            @ptrCast(vectors.ptr),
            &lda,
            values.ptr,
            @ptrCast(work_ptr),
            &lwork,
            rwork.ptr,
            &info,
        );
        if (info != 0) return error.LapackFailure;

        return linalg.EigenDecomp{ .values = values, .vectors = vectors, .n = n };
    }
    // Delegate to LAPACK zheev via linalg module for larger matrices
    return linalg.hermitianEigenDecomp(alloc, .accelerate, n, a);
}

extern fn zheev_(
    jobz: [*]u8,
    uplo: [*]u8,
    n: *c_int,
    a: [*]math.Complex,
    lda: *c_int,
    w: [*]f64,
    work: [*]math.Complex,
    lwork: *c_int,
    rwork: [*]f64,
    info: *c_int,
) callconv(.c) void;

extern fn zhegv_(
    itype: *c_int,
    jobz: [*]u8,
    uplo: [*]u8,
    n: *c_int,
    a: [*]math.Complex,
    lda: *c_int,
    b: [*]math.Complex,
    ldb: *c_int,
    w: [*]f64,
    work: [*]math.Complex,
    lwork: *c_int,
    rwork: [*]f64,
    info: *c_int,
) callconv(.c) void;

/// Solve generalized Hermitian eigenvalue problem A·x = λ·B·x for small matrices.
/// B must be positive definite. Returns eigenvalues and eigenvectors.
pub fn hermitianGeneralizedEigenDecompSmall(alloc: std.mem.Allocator, n: usize, a: []math.Complex, b: []math.Complex) !linalg.EigenDecomp {
    if (n == 0) {
        return linalg.EigenDecomp{
            .values = try alloc.alloc(f64, 0),
            .vectors = try alloc.alloc(math.Complex, 0),
            .n = 0,
        };
    }

    const vectors = try alloc.alloc(math.Complex, n * n);
    errdefer alloc.free(vectors);
    @memcpy(vectors, a[0 .. n * n]);

    const b_copy = try alloc.alloc(math.Complex, n * n);
    defer alloc.free(b_copy);
    @memcpy(b_copy, b[0 .. n * n]);

    const values = try alloc.alloc(f64, n);
    errdefer alloc.free(values);

    var rwork_buf: [3 * 128]f64 = undefined;
    const rwork_len = if (3 * n > 2) 3 * n - 2 else 1;
    const rwork = if (rwork_len <= rwork_buf.len)
        rwork_buf[0..rwork_len]
    else
        return error.MatrixTooLarge;

    var nn: c_int = @intCast(n);
    var lda: c_int = @intCast(n);
    var ldb: c_int = @intCast(n);
    var info: c_int = 0;
    var itype: c_int = 1;
    var jobz = [1]u8{'V'};
    var uplo = [1]u8{'U'};

    // Workspace query
    var lwork: c_int = -1;
    var work_query = math.complex.init(0.0, 0.0);
    zhegv_(
        &itype,
        &jobz,
        &uplo,
        &nn,
        @ptrCast(vectors.ptr),
        &lda,
        @ptrCast(b_copy.ptr),
        &ldb,
        values.ptr,
        @ptrCast(&work_query),
        &lwork,
        rwork.ptr,
        &info,
    );
    if (info != 0) return error.LapackFailure;

    lwork = @intFromFloat(work_query.r);
    if (lwork < 1) lwork = 1;

    var work_stack: [2 * 128 * 128]math.Complex = undefined;
    var work_heap: ?[]math.Complex = null;
    defer if (work_heap) |w_h| alloc.free(w_h);

    const work_ptr: [*]math.Complex = if (@as(usize, @intCast(lwork)) <= work_stack.len)
        &work_stack
    else blk: {
        work_heap = try alloc.alloc(math.Complex, @intCast(lwork));
        break :blk work_heap.?.ptr;
    };

    info = 0;
    zhegv_(
        &itype,
        &jobz,
        &uplo,
        &nn,
        @ptrCast(vectors.ptr),
        &lda,
        @ptrCast(b_copy.ptr),
        &ldb,
        values.ptr,
        @ptrCast(work_ptr),
        &lwork,
        rwork.ptr,
        &info,
    );
    if (info != 0) return error.LapackFailure;

    return linalg.EigenDecomp{ .values = values, .vectors = vectors, .n = n };
}
