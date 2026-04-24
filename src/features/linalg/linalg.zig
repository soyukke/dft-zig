const std = @import("std");
const builtin = @import("builtin");
const math = @import("../math/math.zig");

var lapack_mutex: @import("../../lib/spinlock.zig").SpinLock = .{};

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

extern fn dsygv_(
    itype: *c_int,
    jobz: [*]u8,
    uplo: [*]u8,
    n: *c_int,
    a: [*]f64,
    lda: *c_int,
    b: [*]f64,
    ldb: *c_int,
    w: [*]f64,
    work: [*]f64,
    lwork: *c_int,
    info: *c_int,
) callconv(.c) void;

extern fn dsyev_(
    jobz: [*]u8,
    uplo: [*]u8,
    n: *c_int,
    a: [*]f64,
    lda: *c_int,
    w: [*]f64,
    work: [*]f64,
    lwork: *c_int,
    info: *c_int,
) callconv(.c) void;

pub const Backend = enum {
    accelerate,
    openblas,
    zig,
};

pub const EigenDecomp = struct {
    values: []f64,
    vectors: []math.Complex,
    n: usize,

    /// Free eigenvalues and eigenvectors.
    pub fn deinit(self: *EigenDecomp, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
        if (self.vectors.len > 0) alloc.free(self.vectors);
    }
};

/// Real symmetric eigenvalue decomposition result.
pub const RealEigenDecomp = struct {
    values: []f64,
    vectors: []f64,
    n: usize,

    /// Free eigenvalues and eigenvectors.
    pub fn deinit(self: *RealEigenDecomp, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
        if (self.vectors.len > 0) alloc.free(self.vectors);
    }
};

fn empty_real_eigen_decomp(alloc: std.mem.Allocator) !RealEigenDecomp {
    return .{
        .values = try alloc.alloc(f64, 0),
        .vectors = try alloc.alloc(f64, 0),
        .n = 0,
    };
}

fn empty_eigen_decomp(alloc: std.mem.Allocator) !EigenDecomp {
    return EigenDecomp{
        .values = try alloc.alloc(f64, 0),
        .vectors = try alloc.alloc(math.Complex, 0),
        .n = 0,
    };
}

/// Parse backend string.
pub fn parse_backend(value: []const u8) !Backend {
    if (std.mem.eql(u8, value, "accelerate")) return .accelerate;
    if (std.mem.eql(u8, value, "openblas")) return .openblas;
    if (std.mem.eql(u8, value, "zig")) return .zig;
    return error.InvalidLinalgBackend;
}

/// Return backend name.
pub fn backend_name(backend: Backend) []const u8 {
    return switch (backend) {
        .accelerate => "accelerate",
        .openblas => "openblas",
        .zig => "zig",
    };
}

/// Compute Hermitian eigenvalues using selected backend.
pub fn hermitian_eigenvalues(
    alloc: std.mem.Allocator,
    backend: Backend,
    n: usize,
    a: []math.Complex,
) ![]f64 {
    return switch (backend) {
        .accelerate, .openblas => hermitian_eigenvalues_lapack(alloc, n, a),
        .zig => return error.UnsupportedLinalgBackend,
    };
}

/// Compute Hermitian eigenvalues and eigenvectors using selected backend.
pub fn hermitian_eigen_decomp(
    alloc: std.mem.Allocator,
    backend: Backend,
    n: usize,
    a: []math.Complex,
) !EigenDecomp {
    return switch (backend) {
        .accelerate, .openblas => hermitian_eigen_decomp_lapack(alloc, n, a),
        .zig => return error.UnsupportedLinalgBackend,
    };
}

/// Compute generalized Hermitian eigenvalues using selected backend.
pub fn hermitian_gen_eigenvalues(
    alloc: std.mem.Allocator,
    backend: Backend,
    n: usize,
    a: []math.Complex,
    b: []math.Complex,
) ![]f64 {
    return switch (backend) {
        .accelerate, .openblas => hermitian_gen_eigenvalues_lapack(alloc, n, a, b),
        .zig => return error.UnsupportedLinalgBackend,
    };
}

/// Compute generalized Hermitian eigenvalues and eigenvectors using selected backend.
pub fn hermitian_gen_eigen_decomp(
    alloc: std.mem.Allocator,
    backend: Backend,
    n: usize,
    a: []math.Complex,
    b: []math.Complex,
) !EigenDecomp {
    return switch (backend) {
        .accelerate, .openblas => hermitian_gen_eigen_decomp_lapack(alloc, n, a, b),
        .zig => return error.UnsupportedLinalgBackend,
    };
}

/// Compute generalized real symmetric eigenvalues and eigenvectors.
/// Solves A·x = λ·B·x where A and B are real symmetric, B is positive definite.
/// Input matrices are in row-major order. On return, eigenvectors are column-major
/// (LAPACK convention): eigenvector i is vectors[i*n .. (i+1)*n] after transpose.
/// Eigenvalues are returned in ascending order.
pub fn real_symmetric_gen_eigen_decomp(
    alloc: std.mem.Allocator,
    backend: Backend,
    n: usize,
    a: []f64,
    b: []f64,
) !RealEigenDecomp {
    return switch (backend) {
        .accelerate, .openblas => real_symmetric_gen_eigen_decomp_lapack(alloc, n, a, b),
        .zig => return error.UnsupportedLinalgBackend,
    };
}

/// Compute real symmetric eigenvalues and eigenvectors (standard problem).
/// Solves A·x = λ·x where A is real symmetric.
/// Input matrix is in row-major order (for symmetric, same as column-major).
/// Eigenvalues are returned in ascending order.
pub fn real_symmetric_eigen_decomp(
    alloc: std.mem.Allocator,
    backend: Backend,
    n: usize,
    a: []f64,
) !RealEigenDecomp {
    return switch (backend) {
        .accelerate, .openblas => real_symmetric_eigen_decomp_lapack(alloc, n, a),
        .zig => return error.UnsupportedLinalgBackend,
    };
}

fn query_dsygv_lwork(
    nn: c_int,
    matrix_a: []f64,
    matrix_b: []f64,
    w: []f64,
    itype: *c_int,
    jobz: *[1]u8,
    uplo: *[1]u8,
    lda: *const c_int,
    ldb: *const c_int,
) !c_int {
    var lwork: c_int = -1;
    var work_query: f64 = 0.0;
    var info: c_int = 0;
    dsygv_(
        itype,
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix_a.ptr)),
        @constCast(lda),
        @ptrCast(@constCast(matrix_b.ptr)),
        @constCast(ldb),
        w.ptr,
        @ptrCast(&work_query),
        &lwork,
        &info,
    );
    if (info != 0) return error.LapackFailure;

    lwork = @intFromFloat(work_query);
    return @max(lwork, 1);
}

/// Compute generalized real symmetric eigenvalues and eigenvectors using LAPACK.
/// LAPACK dsygv_ uses column-major order. Since A and B are symmetric,
/// row-major == column-major, so no transposition is needed on input.
/// On output, eigenvectors are stored as columns of A (column-major),
/// which means eigenvector j occupies elements a[j*n..j*n+n] in the flat array.
fn real_symmetric_gen_eigen_decomp_lapack(
    alloc: std.mem.Allocator,
    n: usize,
    a: []f64,
    b: []f64,
) !RealEigenDecomp {
    lapack_mutex.lock();
    defer lapack_mutex.unlock();

    if (a.len != n * n or b.len != n * n) return error.InvalidMatrixSize;
    if (n == 0) return empty_real_eigen_decomp(alloc);

    const matrix_a = try alloc.alloc(f64, n * n);
    errdefer alloc.free(matrix_a);
    @memcpy(matrix_a, a);
    const matrix_b = try alloc.alloc(f64, n * n);
    errdefer alloc.free(matrix_b);
    @memcpy(matrix_b, b);

    const lda: c_int = @intCast(n);
    const ldb: c_int = @intCast(n);
    const nn: c_int = @intCast(n);
    var itype: c_int = 1; // A*x = lambda*B*x
    var jobz: [1]u8 = .{'V'}; // compute eigenvectors
    var uplo: [1]u8 = .{'U'}; // upper triangle
    var info: c_int = 0;

    const w = try alloc.alloc(f64, n);
    errdefer alloc.free(w);

    const lwork = try query_dsygv_lwork(
        nn,
        matrix_a,
        matrix_b,
        w,
        &itype,
        &jobz,
        &uplo,
        &lda,
        &ldb,
    );
    const work = try alloc.alloc(f64, @intCast(lwork));
    errdefer alloc.free(work);

    info = 0;
    dsygv_(
        &itype,
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix_a.ptr)),
        @constCast(&lda),
        @ptrCast(@constCast(matrix_b.ptr)),
        @constCast(&ldb),
        w.ptr,
        @ptrCast(@constCast(work.ptr)),
        &lwork,
        &info,
    );
    alloc.free(work);
    alloc.free(matrix_b);
    if (info != 0) return error.LapackFailure;

    return RealEigenDecomp{ .values = w, .vectors = matrix_a, .n = n };
}

/// Compute real symmetric eigenvalues and eigenvectors using LAPACK (dsyev_).
/// Standard eigenvalue problem: A·x = λ·x.
fn real_symmetric_eigen_decomp_lapack(
    alloc: std.mem.Allocator,
    n: usize,
    a: []f64,
) !RealEigenDecomp {
    lapack_mutex.lock();
    defer lapack_mutex.unlock();

    if (a.len != n * n) return error.InvalidMatrixSize;
    if (n == 0) {
        return RealEigenDecomp{
            .values = try alloc.alloc(f64, 0),
            .vectors = try alloc.alloc(f64, 0),
            .n = 0,
        };
    }

    const matrix_a = try alloc.alloc(f64, n * n);
    errdefer alloc.free(matrix_a);
    @memcpy(matrix_a, a);

    const lda: c_int = @intCast(n);
    const nn: c_int = @intCast(n);
    var jobz: [1]u8 = .{'V'}; // compute eigenvectors
    var uplo: [1]u8 = .{'U'}; // upper triangle
    var info: c_int = 0;

    const w = try alloc.alloc(f64, n);
    errdefer alloc.free(w);

    // Workspace query
    var lwork: c_int = -1;
    var work_query: f64 = 0.0;

    dsyev_(
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix_a.ptr)),
        @constCast(&lda),
        w.ptr,
        @ptrCast(&work_query),
        &lwork,
        &info,
    );
    if (info != 0) return error.LapackFailure;

    lwork = @intFromFloat(work_query);
    if (lwork < 1) lwork = 1;
    const work = try alloc.alloc(f64, @intCast(lwork));
    errdefer alloc.free(work);

    info = 0;
    dsyev_(
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix_a.ptr)),
        @constCast(&lda),
        w.ptr,
        @ptrCast(@constCast(work.ptr)),
        &lwork,
        &info,
    );
    alloc.free(work);
    if (info != 0) return error.LapackFailure;

    return RealEigenDecomp{ .values = w, .vectors = matrix_a, .n = n };
}

/// Compute Hermitian eigenvalues using LAPACK.
fn hermitian_eigenvalues_lapack(alloc: std.mem.Allocator, n: usize, a: []math.Complex) ![]f64 {
    lapack_mutex.lock();
    defer lapack_mutex.unlock();

    if (a.len != n * n) return error.InvalidMatrixSize;
    if (n == 0) return try alloc.alloc(f64, 0);

    const matrix = try alloc.alloc(math.Complex, n * n);
    defer alloc.free(matrix);

    @memcpy(matrix, a);

    const lda: c_int = @intCast(n);
    const nn: c_int = @intCast(n);
    var jobz: [1]u8 = .{'N'};
    var uplo: [1]u8 = .{'U'};

    var info: c_int = 0;
    const w = try alloc.alloc(f64, n);

    var lwork: c_int = -1;
    var work_query = math.complex.init(0.0, 0.0);
    const rwork = try alloc.alloc(f64, @max(@as(usize, 1), 3 * n - 2));
    errdefer alloc.free(rwork);

    zheev_(
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix.ptr)),
        @constCast(&lda),
        w.ptr,
        @ptrCast(&work_query),
        &lwork,
        rwork.ptr,
        &info,
    );
    if (info != 0) {
        alloc.free(w);
        return error.LapackFailure;
    }

    lwork = @intFromFloat(work_query.r);
    if (lwork < 1) lwork = 1;
    const work = try alloc.alloc(math.Complex, @intCast(lwork));
    errdefer alloc.free(work);

    info = 0;
    zheev_(
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix.ptr)),
        @constCast(&lda),
        w.ptr,
        @ptrCast(@constCast(work.ptr)),
        &lwork,
        rwork.ptr,
        &info,
    );
    alloc.free(work);
    alloc.free(rwork);
    if (info != 0) {
        alloc.free(w);
        return error.LapackFailure;
    }

    return w;
}

/// Compute Hermitian eigenvalues and eigenvectors using LAPACK.
fn hermitian_eigen_decomp_lapack(
    alloc: std.mem.Allocator,
    n: usize,
    a: []math.Complex,
) !EigenDecomp {
    lapack_mutex.lock();
    defer lapack_mutex.unlock();

    if (a.len != n * n) return error.InvalidMatrixSize;
    if (n == 0) return empty_eigen_decomp(alloc);

    const matrix = try alloc.alloc(math.Complex, n * n);
    errdefer alloc.free(matrix);
    @memcpy(matrix, a);

    const lda: c_int = @intCast(n);
    const nn: c_int = @intCast(n);
    var jobz: [1]u8 = .{'V'};
    var uplo: [1]u8 = .{'U'};
    var info: c_int = 0;

    const w = try alloc.alloc(f64, n);
    errdefer alloc.free(w);

    var lwork: c_int = -1;
    var work_query = math.complex.init(0.0, 0.0);
    const rwork = try alloc.alloc(f64, @max(@as(usize, 1), 3 * n - 2));
    errdefer alloc.free(rwork);

    zheev_(
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix.ptr)),
        @constCast(&lda),
        w.ptr,
        @ptrCast(&work_query),
        &lwork,
        rwork.ptr,
        &info,
    );
    if (info != 0) return error.LapackFailure;

    lwork = @intFromFloat(work_query.r);
    if (lwork < 1) lwork = 1;
    const work = try alloc.alloc(math.Complex, @intCast(lwork));
    errdefer alloc.free(work);

    info = 0;
    zheev_(
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix.ptr)),
        @constCast(&lda),
        w.ptr,
        @ptrCast(@constCast(work.ptr)),
        &lwork,
        rwork.ptr,
        &info,
    );
    alloc.free(work);
    alloc.free(rwork);
    if (info != 0) return error.LapackFailure;

    return EigenDecomp{ .values = w, .vectors = matrix, .n = n };
}

/// Compute generalized Hermitian eigenvalues using LAPACK.
fn hermitian_gen_eigenvalues_lapack(
    alloc: std.mem.Allocator,
    n: usize,
    a: []math.Complex,
    b: []math.Complex,
) ![]f64 {
    lapack_mutex.lock();
    defer lapack_mutex.unlock();

    if (a.len != n * n or b.len != n * n) return error.InvalidMatrixSize;
    if (n == 0) return try alloc.alloc(f64, 0);

    const lda: c_int = @intCast(n);
    const ldb: c_int = @intCast(n);
    const nn: c_int = @intCast(n);
    var itype: c_int = 1;
    var jobz: [1]u8 = .{'N'};
    var uplo: [1]u8 = .{'U'};
    var info: c_int = 0;

    const w = try alloc.alloc(f64, n);
    errdefer alloc.free(w);

    var lwork: c_int = -1;
    var work_query = math.complex.init(0.0, 0.0);
    const rwork = try alloc.alloc(f64, @max(@as(usize, 1), 3 * n - 2));
    errdefer alloc.free(rwork);

    zhegv_(
        &itype,
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(a.ptr)),
        @constCast(&lda),
        @ptrCast(@constCast(b.ptr)),
        @constCast(&ldb),
        w.ptr,
        @ptrCast(&work_query),
        &lwork,
        rwork.ptr,
        &info,
    );
    if (info != 0) {
        alloc.free(rwork);
        alloc.free(w);
        return error.LapackFailure;
    }

    lwork = @intFromFloat(work_query.r);
    if (lwork < 1) lwork = 1;
    const work = try alloc.alloc(math.Complex, @intCast(lwork));
    errdefer alloc.free(work);

    info = 0;
    zhegv_(
        &itype,
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(a.ptr)),
        @constCast(&lda),
        @ptrCast(@constCast(b.ptr)),
        @constCast(&ldb),
        w.ptr,
        @ptrCast(@constCast(work.ptr)),
        &lwork,
        rwork.ptr,
        &info,
    );
    alloc.free(work);
    alloc.free(rwork);
    if (info != 0) {
        alloc.free(w);
        return error.LapackFailure;
    }

    return w;
}

/// Compute generalized Hermitian eigenvalues and eigenvectors using LAPACK.
fn hermitian_gen_eigen_decomp_lapack(
    alloc: std.mem.Allocator,
    n: usize,
    a: []math.Complex,
    b: []math.Complex,
) !EigenDecomp {
    lapack_mutex.lock();
    defer lapack_mutex.unlock();

    if (a.len != n * n or b.len != n * n) return error.InvalidMatrixSize;
    if (n == 0) {
        return EigenDecomp{
            .values = try alloc.alloc(f64, 0),
            .vectors = try alloc.alloc(math.Complex, 0),
            .n = 0,
        };
    }

    const matrix_a = try alloc.alloc(math.Complex, n * n);
    errdefer alloc.free(matrix_a);
    @memcpy(matrix_a, a);
    const matrix_b = try alloc.alloc(math.Complex, n * n);
    errdefer alloc.free(matrix_b);
    @memcpy(matrix_b, b);

    const lda: c_int = @intCast(n);
    const ldb: c_int = @intCast(n);
    const nn: c_int = @intCast(n);
    var itype: c_int = 1;
    var jobz: [1]u8 = .{'V'};
    var uplo: [1]u8 = .{'U'};
    var info: c_int = 0;

    const w = try alloc.alloc(f64, n);
    errdefer alloc.free(w);

    var lwork: c_int = -1;
    var work_query = math.complex.init(0.0, 0.0);
    const rwork = try alloc.alloc(f64, @max(@as(usize, 1), 3 * n - 2));
    errdefer alloc.free(rwork);

    zhegv_(
        &itype,
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix_a.ptr)),
        @constCast(&lda),
        @ptrCast(@constCast(matrix_b.ptr)),
        @constCast(&ldb),
        w.ptr,
        @ptrCast(&work_query),
        &lwork,
        rwork.ptr,
        &info,
    );
    if (info != 0) return error.LapackFailure;

    lwork = @intFromFloat(work_query.r);
    if (lwork < 1) lwork = 1;
    const work = try alloc.alloc(math.Complex, @intCast(lwork));
    errdefer alloc.free(work);

    info = 0;
    zhegv_(
        &itype,
        jobz[0..].ptr,
        uplo[0..].ptr,
        @constCast(&nn),
        @ptrCast(@constCast(matrix_a.ptr)),
        @constCast(&lda),
        @ptrCast(@constCast(matrix_b.ptr)),
        @constCast(&ldb),
        w.ptr,
        @ptrCast(@constCast(work.ptr)),
        &lwork,
        rwork.ptr,
        &info,
    );
    alloc.free(work);
    alloc.free(rwork);
    alloc.free(matrix_b);
    if (info != 0) return error.LapackFailure;

    return EigenDecomp{ .values = w, .vectors = matrix_a, .n = n };
}
