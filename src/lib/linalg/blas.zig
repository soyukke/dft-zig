//! BLAS wrappers for complex matrix operations
//!
//! Provides zgemm, zgemv and related operations using Accelerate framework.

const std = @import("std");

/// Complex number type (same layout as Accelerate's __CLPK_doublecomplex)
pub const Complex = extern struct {
    r: f64,
    i: f64,

    pub fn init(re: f64, im: f64) Complex {
        return .{ .r = re, .i = im };
    }
};

// BLAS function declarations (Accelerate framework)
extern fn cblas_zgemm(
    order: c_int,
    transA: c_int,
    transB: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex,
    a: [*]const Complex,
    lda: c_int,
    b: [*]const Complex,
    ldb: c_int,
    beta: *const Complex,
    c: [*]Complex,
    ldc: c_int,
) callconv(.c) void;

extern fn cblas_zgemv(
    order: c_int,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex,
    a: [*]const Complex,
    lda: c_int,
    x: [*]const Complex,
    incx: c_int,
    beta: *const Complex,
    y: [*]Complex,
    incy: c_int,
) callconv(.c) void;

extern fn cblas_zdotc_sub(
    n: c_int,
    x: [*]const Complex,
    incx: c_int,
    y: [*]const Complex,
    incy: c_int,
    result: *Complex,
) callconv(.c) void;

extern fn cblas_zaxpy(
    n: c_int,
    alpha: *const Complex,
    x: [*]const Complex,
    incx: c_int,
    y: [*]Complex,
    incy: c_int,
) callconv(.c) void;

extern fn cblas_zscal(
    n: c_int,
    alpha: *const Complex,
    x: [*]Complex,
    incx: c_int,
) callconv(.c) void;

extern fn cblas_dznrm2(
    n: c_int,
    x: [*]const Complex,
    incx: c_int,
) callconv(.c) f64;

// Real BLAS function declarations (Accelerate framework)
extern fn cblas_dgemm(
    order: c_int,
    transA: c_int,
    transB: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    b: [*]const f64,
    ldb: c_int,
    beta: f64,
    c: [*]f64,
    ldc: c_int,
) callconv(.c) void;

extern fn cblas_dgemv(
    order: c_int,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    x: [*]const f64,
    incx: c_int,
    beta: f64,
    y: [*]f64,
    incy: c_int,
) callconv(.c) void;

extern fn cblas_dtrsm(
    order: c_int,
    side: c_int,
    uplo: c_int,
    transA: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    b: [*]f64,
    ldb: c_int,
) callconv(.c) void;

extern fn cblas_dtrsv(
    order: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: [*]const f64,
    lda: c_int,
    x: [*]f64,
    incx: c_int,
) callconv(.c) void;

// LAPACK function declarations
extern fn dpotrf_(
    uplo: *const u8,
    n: *const c_int,
    a: [*]f64,
    lda: *const c_int,
    info: *c_int,
) callconv(.c) void;

// CBLAS constants
const CblasRowMajor: c_int = 101;
const CblasColMajor: c_int = 102;
const CblasNoTrans: c_int = 111;
const CblasTrans: c_int = 112;
const CblasConjTrans: c_int = 113;
const CblasLeft: c_int = 141;
const CblasRight: c_int = 142;
const CblasUpper: c_int = 121;
const CblasLower: c_int = 122;
const CblasNonUnit: c_int = 131;

/// Matrix-matrix multiply: C = alpha * op(A) * op(B) + beta * C
/// Uses column-major layout.
pub fn zgemm(
    trans_a: enum { no_trans, trans, conj_trans },
    trans_b: enum { no_trans, trans, conj_trans },
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex,
    a: []const Complex,
    lda: usize,
    b: []const Complex,
    ldb: usize,
    beta: Complex,
    c: []Complex,
    ldc: usize,
) void {
    const ta: c_int = switch (trans_a) {
        .no_trans => CblasNoTrans,
        .trans => CblasTrans,
        .conj_trans => CblasConjTrans,
    };
    const tb: c_int = switch (trans_b) {
        .no_trans => CblasNoTrans,
        .trans => CblasTrans,
        .conj_trans => CblasConjTrans,
    };

    cblas_zgemm(
        CblasColMajor,
        ta,
        tb,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        &alpha,
        a.ptr,
        @intCast(lda),
        b.ptr,
        @intCast(ldb),
        &beta,
        c.ptr,
        @intCast(ldc),
    );
}

/// Matrix-vector multiply: y = alpha * op(A) * x + beta * y
/// Uses column-major layout.
pub fn zgemv(
    trans: enum { no_trans, trans, conj_trans },
    m: usize,
    n: usize,
    alpha: Complex,
    a: []const Complex,
    lda: usize,
    x: []const Complex,
    beta: Complex,
    y: []Complex,
) void {
    const t: c_int = switch (trans) {
        .no_trans => CblasNoTrans,
        .trans => CblasTrans,
        .conj_trans => CblasConjTrans,
    };

    cblas_zgemv(
        CblasColMajor,
        t,
        @intCast(m),
        @intCast(n),
        &alpha,
        a.ptr,
        @intCast(lda),
        x.ptr,
        1,
        &beta,
        y.ptr,
        1,
    );
}

/// Complex dot product: result = conj(x) . y
pub fn zdotc(x: []const Complex, y: []const Complex) Complex {
    const n = @min(x.len, y.len);
    if (n == 0) return Complex.init(0.0, 0.0);

    var result = Complex.init(0.0, 0.0);
    cblas_zdotc_sub(
        @intCast(n),
        x.ptr,
        1,
        y.ptr,
        1,
        &result,
    );
    return result;
}

/// AXPY: y = alpha * x + y
pub fn zaxpy(alpha: Complex, x: []const Complex, y: []Complex) void {
    const n = @min(x.len, y.len);
    if (n == 0) return;

    cblas_zaxpy(
        @intCast(n),
        &alpha,
        x.ptr,
        1,
        y.ptr,
        1,
    );
}

/// Scale: x = alpha * x
pub fn zscal(alpha: Complex, x: []Complex) void {
    if (x.len == 0) return;

    cblas_zscal(
        @intCast(x.len),
        &alpha,
        x.ptr,
        1,
    );
}

/// Vector 2-norm: ||x||
pub fn dznrm2(x: []const Complex) f64 {
    if (x.len == 0) return 0.0;

    return cblas_dznrm2(
        @intCast(x.len),
        x.ptr,
        1,
    );
}

// ============== High-level operations ==============

// ============== Real BLAS wrappers ==============

/// Real matrix-matrix multiply: C = alpha * op(A) * op(B) + beta * C
/// Uses row-major layout (natural for Zig arrays).
///
/// For row-major: A is m×k, B is k×n, C is m×n.
/// A[i,j] = a[i * lda + j], B[i,j] = b[i * ldb + j], C[i,j] = c[i * ldc + j].
pub fn dgemm(
    trans_a: enum { no_trans, trans },
    trans_b: enum { no_trans, trans },
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: []const f64,
    lda: usize,
    b: []const f64,
    ldb: usize,
    beta: f64,
    c: []f64,
    ldc: usize,
) void {
    const ta: c_int = switch (trans_a) {
        .no_trans => CblasNoTrans,
        .trans => CblasTrans,
    };
    const tb: c_int = switch (trans_b) {
        .no_trans => CblasNoTrans,
        .trans => CblasTrans,
    };

    cblas_dgemm(
        CblasRowMajor,
        ta,
        tb,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        alpha,
        a.ptr,
        @intCast(lda),
        b.ptr,
        @intCast(ldb),
        beta,
        c.ptr,
        @intCast(ldc),
    );
}

// ============== Real BLAS additional wrappers ==============

/// Real matrix-vector multiply: y = alpha * op(A) * x + beta * y
/// Uses row-major layout.
/// A is m×n, x is n-vector (or m if trans), y is m-vector (or n if trans).
pub fn dgemv(
    trans: enum { no_trans, trans },
    m: usize,
    n: usize,
    alpha: f64,
    a: []const f64,
    lda: usize,
    x: []const f64,
    beta: f64,
    y: []f64,
) void {
    const t: c_int = switch (trans) {
        .no_trans => CblasNoTrans,
        .trans => CblasTrans,
    };

    cblas_dgemv(
        CblasRowMajor,
        t,
        @intCast(m),
        @intCast(n),
        alpha,
        a.ptr,
        @intCast(lda),
        x.ptr,
        1,
        beta,
        y.ptr,
        1,
    );
}

/// Cholesky factorization: A = L * L^T (lower triangular).
/// Input A is n×n symmetric positive definite in row-major.
/// On output, the lower triangle of A contains L.
/// Returns error if matrix is not positive definite.
pub fn dpotrf(n: usize, a: []f64) !void {
    // LAPACK dpotrf uses column-major. For row-major symmetric matrix,
    // using 'U' with column-major is equivalent to 'L' with row-major.
    // So we call with uplo='U' and the lower triangle in row-major becomes
    // the upper triangle in column-major (since A^T = A for symmetric).
    // Actually, for symmetric matrices stored in row-major, we use 'U'
    // because row-major lower = column-major upper.
    var nn: c_int = @intCast(n);
    var info: c_int = 0;
    const uplo: u8 = 'U'; // Row-major lower triangle
    dpotrf_(&uplo, &nn, a.ptr, &nn, &info);
    if (info != 0) return error.CholeskyFailed;
}

/// Triangular solve: op(A) * X = alpha * B  or  X * op(A) = alpha * B
/// A is triangular n×n, B is m×n (overwritten with solution X).
/// Uses row-major layout.
pub fn dtrsm(
    side: enum { left, right },
    uplo: enum { upper, lower },
    trans: enum { no_trans, trans },
    m: usize,
    n: usize,
    alpha: f64,
    a: []const f64,
    lda: usize,
    b: []f64,
    ldb: usize,
) void {
    const s: c_int = switch (side) {
        .left => CblasLeft,
        .right => CblasRight,
    };
    const u: c_int = switch (uplo) {
        .upper => CblasUpper,
        .lower => CblasLower,
    };
    const t: c_int = switch (trans) {
        .no_trans => CblasNoTrans,
        .trans => CblasTrans,
    };

    cblas_dtrsm(
        CblasRowMajor,
        s,
        u,
        t,
        CblasNonUnit,
        @intCast(m),
        @intCast(n),
        alpha,
        a.ptr,
        @intCast(lda),
        b.ptr,
        @intCast(ldb),
    );
}

/// Triangular solve for vector: op(A) * x = b (x overwrites b in-place).
/// A is triangular n×n in row-major.
pub fn dtrsv(
    uplo: enum { upper, lower },
    trans: enum { no_trans, trans },
    n: usize,
    a: []const f64,
    lda: usize,
    x: []f64,
) void {
    const u: c_int = switch (uplo) {
        .upper => CblasUpper,
        .lower => CblasLower,
    };
    const t: c_int = switch (trans) {
        .no_trans => CblasNoTrans,
        .trans => CblasTrans,
    };

    cblas_dtrsv(
        CblasRowMajor,
        u,
        t,
        CblasNonUnit,
        @intCast(n),
        a.ptr,
        @intCast(lda),
        x.ptr,
        1,
    );
}

// ============== Complex high-level operations ==============

/// Build projected matrix: T = V† * W
/// V is n x m, W is n x m, T is m x m
/// T[i,j] = <V[:,i] | W[:,j]> = conj(V[:,i])^T * W[:,j]
pub fn buildProjectedMatrix(
    n: usize,
    m: usize,
    v: []const Complex,
    w: []const Complex,
    t: []Complex,
) void {
    // T = V† * W using zgemm
    // V† is m x n (conj_trans of n x m)
    // W is n x m
    // Result T is m x m
    const alpha = Complex.init(1.0, 0.0);
    const beta = Complex.init(0.0, 0.0);

    zgemm(
        .conj_trans, // V†
        .no_trans, // W
        m, // rows of result
        m, // cols of result
        n, // inner dimension
        alpha,
        v,
        n, // lda = n (leading dimension of V)
        w,
        n, // ldb = n (leading dimension of W)
        beta,
        t,
        m, // ldc = m (leading dimension of T)
    );
}

/// Combine columns: out = V * coeffs
/// V is n x m matrix, coeffs is m-vector, out is n-vector
pub fn combineColumns(
    n: usize,
    m: usize,
    v: []const Complex,
    coeffs: []const Complex,
    out: []Complex,
) void {
    const alpha = Complex.init(1.0, 0.0);
    const beta = Complex.init(0.0, 0.0);

    zgemv(
        .no_trans,
        n,
        m,
        alpha,
        v,
        n, // lda
        coeffs,
        beta,
        out,
    );
}

// ============== Tests ==============

test "zgemm basic" {
    // 2x2 * 2x2 matrix multiply
    const a = [_]Complex{
        Complex.init(1.0, 0.0), Complex.init(2.0, 0.0),
        Complex.init(3.0, 0.0), Complex.init(4.0, 0.0),
    };
    const b = [_]Complex{
        Complex.init(5.0, 0.0), Complex.init(6.0, 0.0),
        Complex.init(7.0, 0.0), Complex.init(8.0, 0.0),
    };
    var c = [_]Complex{
        Complex.init(0.0, 0.0), Complex.init(0.0, 0.0),
        Complex.init(0.0, 0.0), Complex.init(0.0, 0.0),
    };

    zgemm(.no_trans, .no_trans, 2, 2, 2, Complex.init(1.0, 0.0), &a, 2, &b, 2, Complex.init(0.0, 0.0), &c, 2);

    // Column-major: A=[[1,3],[2,4]], B=[[5,7],[6,8]]
    // C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 3*6 = 23
    try std.testing.expectApproxEqAbs(23.0, c[0].r, 1e-10);
}

test "zdotc basic" {
    const x = [_]Complex{
        Complex.init(1.0, 2.0),
        Complex.init(3.0, 4.0),
    };
    const y = [_]Complex{
        Complex.init(5.0, 6.0),
        Complex.init(7.0, 8.0),
    };

    // conj(x) . y = (1-2i)*(5+6i) + (3-4i)*(7+8i)
    //             = (5+6i-10i+12) + (21+24i-28i+32)
    //             = 17-4i + 53-4i = 70-8i
    const result = zdotc(&x, &y);
    try std.testing.expectApproxEqAbs(70.0, result.r, 1e-10);
    try std.testing.expectApproxEqAbs(-8.0, result.i, 1e-10);
}

test "dznrm2 basic" {
    const x = [_]Complex{
        Complex.init(3.0, 4.0), // |x[0]| = 5
        Complex.init(0.0, 0.0),
    };
    const norm = dznrm2(&x);
    try std.testing.expectApproxEqAbs(5.0, norm, 1e-10);
}

test "dpotrf Cholesky decomposition" {
    // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, √2]]
    // Row-major: dpotrf with 'U' gives upper triangle in col-major = lower in row-major
    var a = [_]f64{ 4.0, 2.0, 2.0, 3.0 };
    try dpotrf(2, &a);

    // After dpotrf, lower triangle contains L (row-major)
    // L[0,0] = 2, L[1,0] = 1, L[1,1] = sqrt(2)
    try std.testing.expectApproxEqAbs(2.0, a[0], 1e-10); // L[0,0]
    try std.testing.expectApproxEqAbs(1.0, a[2], 1e-10); // L[1,0]
    try std.testing.expectApproxEqAbs(@sqrt(2.0), a[3], 1e-10); // L[1,1]

    // Verify L * L^T == A
    const l00 = a[0];
    const l10 = a[2];
    const l11 = a[3];
    try std.testing.expectApproxEqAbs(4.0, l00 * l00, 1e-10); // A[0,0]
    try std.testing.expectApproxEqAbs(2.0, l10 * l00, 1e-10); // A[1,0]
    try std.testing.expectApproxEqAbs(3.0, l10 * l10 + l11 * l11, 1e-10); // A[1,1]
}

test "dtrsm triangular solve" {
    // L = [[2, 0], [1, √2]], solve L * X = B where B = [[4], [4]]
    // L*x = b: 2*x0 = 4 → x0=2, x0 + √2*x1 = 4 → x1 = 2/√2 = √2
    const l = [_]f64{ 2.0, 0.0, 1.0, @sqrt(2.0) };
    var b = [_]f64{ 4.0, 4.0 };

    dtrsm(.left, .lower, .no_trans, 2, 1, 1.0, &l, 2, &b, 1);

    try std.testing.expectApproxEqAbs(2.0, b[0], 1e-10);
    try std.testing.expectApproxEqAbs(@sqrt(2.0), b[1], 1e-10);
}

test "dgemv matrix-vector multiply" {
    // A = [[1, 2], [3, 4]], x = [5, 6]
    // y = A * x = [1*5+2*6, 3*5+4*6] = [17, 39]
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const x = [_]f64{ 5.0, 6.0 };
    var y = [_]f64{ 0.0, 0.0 };

    dgemv(.no_trans, 2, 2, 1.0, &a, 2, &x, 0.0, &y);

    try std.testing.expectApproxEqAbs(17.0, y[0], 1e-10);
    try std.testing.expectApproxEqAbs(39.0, y[1], 1e-10);
}

test "dtrsv triangular vector solve" {
    // L = [[2, 0], [1, √2]], solve L * x = b where b = [4, 4]
    // 2*x0 = 4 → x0=2, x0 + √2*x1 = 4 → x1 = √2
    const l = [_]f64{ 2.0, 0.0, 1.0, @sqrt(2.0) };
    var x = [_]f64{ 4.0, 4.0 };

    dtrsv(.lower, .no_trans, 2, &l, 2, &x);

    try std.testing.expectApproxEqAbs(2.0, x[0], 1e-10);
    try std.testing.expectApproxEqAbs(@sqrt(2.0), x[1], 1e-10);
}
