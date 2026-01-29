//! Iterative eigenvalue solvers
//!
//! Provides LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
//! for computing a few smallest eigenvalues of large sparse matrices.
//!
//! Supports both serial and parallel implementations.

const std = @import("std");
const math = @import("../math/math.zig");
const linalg = @import("linalg.zig");
const thread_pool_mod = @import("../thread_pool.zig");

// Import LOBPCG modules
pub const lobpcg_common = @import("lobpcg/common.zig");
pub const lobpcg_serial = @import("lobpcg/serial.zig");
pub const lobpcg_parallel = @import("lobpcg/parallel.zig");
pub const cg = @import("cg/cg.zig");

// Re-export common types for backward compatibility
pub const Operator = lobpcg_common.Operator;
pub const Options = lobpcg_common.Options;
pub const ThreadPool = thread_pool_mod.ThreadPool;

/// LOBPCG backend selection
pub const LobpcgBackend = enum {
    serial,
    parallel,
};

/// Extended options with backend selection and thread pool
pub const ExtendedOptions = struct {
    base: Options = .{},
    lobpcg_backend: LobpcgBackend = .serial,
    pool: ?*ThreadPool = null, // External thread pool for parallel backend
};

/// Solve eigenvalue problem using LOBPCG with specified backend
pub fn hermitianEigenDecompIterativeExt(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    op: Operator,
    diag: []const f64,
    nbands: usize,
    opts: ExtendedOptions,
) !linalg.EigenDecomp {
    return switch (opts.lobpcg_backend) {
        .serial => lobpcg_serial.solve(alloc, backend, op, diag, nbands, opts.base),
        .parallel => lobpcg_parallel.solve(alloc, backend, op, diag, nbands, .{
            .base = opts.base,
            .pool = opts.pool,
        }),
    };
}

/// Solve eigenvalue problem using band-by-band CG
pub fn hermitianEigenDecompCG(
    alloc: std.mem.Allocator,
    op: Operator,
    diag: []const f64,
    nbands: usize,
    opts: Options,
) !linalg.EigenDecomp {
    return cg.solve(alloc, op, diag, nbands, opts);
}

/// Solve eigenvalue problem using serial LOBPCG (backward compatible)
pub fn hermitianEigenDecompIterative(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    op: Operator,
    diag: []const f64,
    nbands: usize,
    opts: Options,
) !linalg.EigenDecomp {
    return lobpcg_serial.solve(alloc, backend, op, diag, nbands, opts);
}

// ============== Tests ==============

test "LOBPCG serial basic" {
    const allocator = std.testing.allocator;

    // Simple 3x3 diagonal matrix
    const TestCtx = struct {
        diag: [3]f64,

        pub fn apply(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            for (0..3) |i| {
                y[i] = math.complex.scale(x[i], self.diag[i]);
            }
        }
    };

    var ctx = TestCtx{ .diag = .{ 1.0, 2.0, 3.0 } };
    const op = Operator{
        .n = 3,
        .ctx = @ptrCast(&ctx),
        .apply = TestCtx.apply,
    };

    var result = try hermitianEigenDecompIterative(
        allocator,
        .accelerate,
        op,
        &ctx.diag,
        2,
        .{ .max_iter = 20, .tol = 1e-8 },
    );
    defer result.deinit(allocator);

    // Should find eigenvalues 1.0 and 2.0
    try std.testing.expectApproxEqAbs(result.values[0], 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(result.values[1], 2.0, 1e-6);
}

test "LOBPCG generalized eigenvalue (S != I)" {
    const allocator = std.testing.allocator;

    // 3x3 generalized problem: H·x = λ·S·x
    // H = diag(2, 6, 12), S = diag(1, 2, 3)
    // Eigenvalues: 2/1=2, 6/2=3, 12/3=4
    const TestCtx = struct {
        h_diag: [3]f64,
        s_diag: [3]f64,

        pub fn applyH(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            for (0..3) |i| {
                y[i] = math.complex.scale(x[i], self.h_diag[i]);
            }
        }

        pub fn applyS(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            for (0..3) |i| {
                y[i] = math.complex.scale(x[i], self.s_diag[i]);
            }
        }
    };

    var ctx = TestCtx{
        .h_diag = .{ 2.0, 6.0, 12.0 },
        .s_diag = .{ 1.0, 2.0, 3.0 },
    };
    const op = Operator{
        .n = 3,
        .ctx = @ptrCast(&ctx),
        .apply = TestCtx.applyH,
        .apply_s = TestCtx.applyS,
    };

    var result = try hermitianEigenDecompIterative(
        allocator,
        .accelerate,
        op,
        &ctx.h_diag,
        2,
        .{ .max_iter = 40, .tol = 1e-8 },
    );
    defer result.deinit(allocator);

    // Should find eigenvalues 2.0 and 3.0
    try std.testing.expectApproxEqAbs(result.values[0], 2.0, 1e-6);
    try std.testing.expectApproxEqAbs(result.values[1], 3.0, 1e-6);
}

test "LOBPCG with apply_s=null gives same results as standard" {
    const allocator = std.testing.allocator;

    // Standard problem with apply_s=null should give same results
    const TestCtx = struct {
        diag: [4]f64,

        pub fn apply(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            for (0..4) |i| {
                y[i] = math.complex.scale(x[i], self.diag[i]);
            }
        }
    };

    var ctx = TestCtx{ .diag = .{ 1.0, 3.0, 5.0, 7.0 } };
    const op = Operator{
        .n = 4,
        .ctx = @ptrCast(&ctx),
        .apply = TestCtx.apply,
        // apply_s = null (default) => S = I
    };

    var result = try hermitianEigenDecompIterative(
        allocator,
        .accelerate,
        op,
        &ctx.diag,
        2,
        .{ .max_iter = 20, .tol = 1e-8 },
    );
    defer result.deinit(allocator);

    try std.testing.expectApproxEqAbs(result.values[0], 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(result.values[1], 3.0, 1e-6);
}
