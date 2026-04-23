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

const SolveParams = struct {
    max_iter: usize,
    tol: f64,
};

const BandState = struct {
    lambda: f64,
    prev_zr: f64 = 0.0,
    has_prev_direction: bool = false,
};

const RitzResult = struct {
    lambda_new: f64,
    c1: math.Complex,
    c2: math.Complex,
    c3: math.Complex = math.complex.init(0.0, 0.0),
    use_3term: bool = false,
};

const CgWorkspace = struct {
    h_psi: []math.Complex = &[_]math.Complex{},
    residual: []math.Complex = &[_]math.Complex{},
    precond_buf: []math.Complex = &[_]math.Complex{},
    direction: []math.Complex = &[_]math.Complex{},
    h_dir: []math.Complex = &[_]math.Complex{},
    psi_old: []math.Complex = &[_]math.Complex{},
    h_psi_old: []math.Complex = &[_]math.Complex{},
    psi_prev: []math.Complex = &[_]math.Complex{},
    h_psi_prev: []math.Complex = &[_]math.Complex{},

    fn init(alloc: std.mem.Allocator, n: usize) !CgWorkspace {
        var workspace: CgWorkspace = .{};
        errdefer workspace.deinit(alloc);
        workspace.h_psi = try alloc.alloc(math.Complex, n);
        workspace.residual = try alloc.alloc(math.Complex, n);
        workspace.precond_buf = try alloc.alloc(math.Complex, n);
        workspace.direction = try alloc.alloc(math.Complex, n);
        workspace.h_dir = try alloc.alloc(math.Complex, n);
        workspace.psi_old = try alloc.alloc(math.Complex, n);
        workspace.h_psi_old = try alloc.alloc(math.Complex, n);
        workspace.psi_prev = try alloc.alloc(math.Complex, n);
        workspace.h_psi_prev = try alloc.alloc(math.Complex, n);
        return workspace;
    }

    fn deinit(self: *CgWorkspace, alloc: std.mem.Allocator) void {
        if (self.h_psi.len > 0) alloc.free(self.h_psi);
        if (self.residual.len > 0) alloc.free(self.residual);
        if (self.precond_buf.len > 0) alloc.free(self.precond_buf);
        if (self.direction.len > 0) alloc.free(self.direction);
        if (self.h_dir.len > 0) alloc.free(self.h_dir);
        if (self.psi_old.len > 0) alloc.free(self.psi_old);
        if (self.h_psi_old.len > 0) alloc.free(self.h_psi_old);
        if (self.psi_prev.len > 0) alloc.free(self.psi_prev);
        if (self.h_psi_prev.len > 0) alloc.free(self.h_psi_prev);
        self.* = .{};
    }
};

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

    const params = SolveParams{
        .max_iter = if (opts.max_iter == 0) 20 else opts.max_iter,
        .tol = if (opts.tol == 0.0) 1e-6 else opts.tol,
    };
    const values = try alloc.alloc(f64, nbands);
    errdefer alloc.free(values);
    const vectors = try alloc.alloc(math.Complex, n * nbands);
    errdefer alloc.free(vectors);
    var workspace = try CgWorkspace.init(alloc, n);
    defer workspace.deinit(alloc);

    try initialize_solve_vectors(n, vectors, nbands, opts);
    try solve_bands(alloc, op, diag, nbands, params, values, vectors, &workspace);
    try diagonalize_cg_subspace(alloc, op, n, nbands, values, vectors);

    return linalg.EigenDecomp{
        .values = values,
        .vectors = vectors,
        .n = n,
    };
}

fn initialize_solve_vectors(
    n: usize,
    vectors: []math.Complex,
    nbands: usize,
    opts: common.Options,
) !void {
    var seed: u64 = 0x6f3d3cbe;
    if (opts.init_vectors) |init_vecs| {
        const cols = if (opts.init_vectors_cols == 0) nbands else opts.init_vectors_cols;
        const copy_cols = @min(nbands, cols);
        if (init_vecs.len >= n * copy_cols) {
            @memcpy(vectors[0 .. n * copy_cols], init_vecs[0 .. n * copy_cols]);
        }
        if (copy_cols < nbands) {
            common.init_random_vectors(n, vectors[n * copy_cols ..], nbands - copy_cols, &seed);
        }
    } else {
        common.init_random_vectors(n, vectors, nbands, &seed);
    }
    try common.orthonormalize_all(n, vectors, nbands, &seed);
}

fn solve_bands(
    alloc: std.mem.Allocator,
    op: common.Operator,
    diag: []const f64,
    nbands: usize,
    params: SolveParams,
    values: []f64,
    vectors: []math.Complex,
    workspace: *CgWorkspace,
) !void {
    for (0..nbands) |band| {
        try solve_band(alloc, op, diag, band, params, values, vectors, workspace);
    }
}

fn solve_band(
    alloc: std.mem.Allocator,
    op: common.Operator,
    diag: []const f64,
    band: usize,
    params: SolveParams,
    values: []f64,
    vectors: []math.Complex,
    workspace: *CgWorkspace,
) !void {
    const n = op.n;
    const psi = common.column(vectors, n, band);
    if (band > 0) _ = common.orthonormalize_vector(n, psi, vectors, band);
    try op.apply(op.ctx, psi, workspace.h_psi);

    var state = BandState{ .lambda = common.inner_product(n, psi, workspace.h_psi).r };
    for (0..params.max_iter) |_| {
        const ready = try prepare_cg_iteration(
            op,
            diag,
            band,
            params.tol,
            vectors,
            psi,
            workspace,
            &state,
        );
        if (!ready) break;

        const ritz = try compute_ritz_result(
            alloc,
            n,
            state.lambda,
            psi,
            workspace,
            state.has_prev_direction,
        );
        store_previous_band_state(psi, workspace, &state);
        apply_ritz_update(n, psi, workspace, ritz);
        if (update_band_eigenvalue(&state.lambda, ritz.lambda_new, params.tol)) break;
    }

    values[band] = state.lambda;
    if (band > 0) _ = common.orthonormalize_vector(n, psi, vectors, band);
}

fn prepare_cg_iteration(
    op: common.Operator,
    diag: []const f64,
    band: usize,
    tol: f64,
    vectors: []const math.Complex,
    psi: []const math.Complex,
    workspace: *CgWorkspace,
    state: *BandState,
) !bool {
    const n = op.n;
    @memcpy(workspace.residual, workspace.h_psi);
    common.axpy(n, workspace.residual, psi, -state.lambda);
    orthogonalize(n, workspace.residual, vectors, band);
    if (common.vector_norm(n, workspace.residual) < tol) return false;

    common.precondition(n, diag, state.lambda, workspace.residual, workspace.precond_buf);
    orthogonalize(n, workspace.precond_buf, vectors, band + 1);
    update_search_direction(
        n,
        workspace.direction,
        workspace.precond_buf,
        workspace.residual,
        state,
    );

    orthogonalize(n, workspace.direction, vectors, band + 1);
    const d_norm = common.vector_norm(n, workspace.direction);
    if (d_norm < 1e-14) return false;
    scale_vec(n, workspace.direction, 1.0 / d_norm);
    try op.apply(op.ctx, workspace.direction, workspace.h_dir);
    return true;
}

fn update_search_direction(
    n: usize,
    direction: []math.Complex,
    precond_buf: []const math.Complex,
    residual: []const math.Complex,
    state: *BandState,
) void {
    const zr = common.inner_product(n, precond_buf, residual).r;
    if (state.has_prev_direction and state.prev_zr > 1e-30) {
        const beta = @max(0.0, zr / state.prev_zr);
        for (0..n) |i| {
            const scaled = math.complex.scale(direction[i], beta);
            direction[i] = math.complex.add(precond_buf[i], scaled);
        }
    } else {
        @memcpy(direction, precond_buf);
    }
    state.prev_zr = zr;
}

fn compute_ritz_result(
    alloc: std.mem.Allocator,
    n: usize,
    lambda: f64,
    psi: []const math.Complex,
    workspace: *CgWorkspace,
    has_prev_direction: bool,
) !RitzResult {
    if (has_prev_direction and prepare_three_term_subspace(n, psi, workspace)) {
        if (try compute_three_term_ritz_result(
            alloc,
            n,
            lambda,
            psi,
            workspace,
        )) |ritz| return ritz;
    }
    return compute_two_term_ritz_result(n, lambda, psi, workspace.direction, workspace.h_dir);
}

fn prepare_three_term_subspace(
    n: usize,
    psi: []const math.Complex,
    workspace: *CgWorkspace,
) bool {
    @memcpy(workspace.psi_prev, workspace.psi_old);
    @memcpy(workspace.h_psi_prev, workspace.h_psi_old);

    const ov_psi = common.inner_product(n, psi, workspace.psi_prev);
    const neg_ov_psi = math.complex.init(-ov_psi.r, -ov_psi.i);
    zaxpy_complex(n, workspace.psi_prev, psi, neg_ov_psi);
    zaxpy_complex(n, workspace.h_psi_prev, workspace.h_psi, neg_ov_psi);

    const ov_d = common.inner_product(n, workspace.direction, workspace.psi_prev);
    const neg_ov_d = math.complex.init(-ov_d.r, -ov_d.i);
    zaxpy_complex(n, workspace.psi_prev, workspace.direction, neg_ov_d);
    zaxpy_complex(n, workspace.h_psi_prev, workspace.h_dir, neg_ov_d);

    const p_norm = common.vector_norm(n, workspace.psi_prev);
    if (p_norm <= 1e-6) return false;
    scale_vec(n, workspace.psi_prev, 1.0 / p_norm);
    scale_vec(n, workspace.h_psi_prev, 1.0 / p_norm);
    return true;
}

fn compute_three_term_ritz_result(
    alloc: std.mem.Allocator,
    n: usize,
    lambda: f64,
    psi: []const math.Complex,
    workspace: *CgWorkspace,
) !?RitzResult {
    const a11 = lambda;
    const a22 = common.inner_product(n, workspace.direction, workspace.h_dir).r;
    const a33 = common.inner_product(n, workspace.psi_prev, workspace.h_psi_prev).r;
    const a12 = common.inner_product(n, psi, workspace.h_dir);
    const a13 = common.inner_product(n, psi, workspace.h_psi_prev);
    const a23 = common.inner_product(n, workspace.direction, workspace.h_psi_prev);

    var h3: [9]math.Complex = undefined;
    h3[0] = math.complex.init(a11, 0.0);
    h3[1] = math.complex.init(a12.r, -a12.i);
    h3[2] = math.complex.init(a13.r, -a13.i);
    h3[3] = a12;
    h3[4] = math.complex.init(a22, 0.0);
    h3[5] = math.complex.init(a23.r, -a23.i);
    h3[6] = a13;
    h3[7] = a23;
    h3[8] = math.complex.init(a33, 0.0);

    const sub_eig = common.hermitian_eigen_decomp_small(alloc, 3, &h3) catch return null;
    defer alloc.free(sub_eig.values);
    defer alloc.free(sub_eig.vectors);

    return .{
        .lambda_new = sub_eig.values[0],
        .c1 = sub_eig.vectors[0],
        .c2 = sub_eig.vectors[1],
        .c3 = sub_eig.vectors[2],
        .use_3term = true,
    };
}

fn compute_two_term_ritz_result(
    n: usize,
    lambda: f64,
    psi: []const math.Complex,
    direction: []const math.Complex,
    h_dir: []const math.Complex,
) RitzResult {
    const a11 = lambda;
    const a22 = common.inner_product(n, direction, h_dir).r;
    const a12 = common.inner_product(n, psi, h_dir);
    const avg = 0.5 * (a11 + a22);
    const diff = 0.5 * (a11 - a22);
    const off_sq = a12.r * a12.r + a12.i * a12.i;
    const disc = @sqrt(diff * diff + off_sq);
    var result = RitzResult{ .lambda_new = avg - disc, .c1 = undefined, .c2 = undefined };
    if (off_sq > 1e-30) {
        result.c1 = math.complex.init(-a12.r, -a12.i);
        result.c2 = math.complex.init(a11 - result.lambda_new, 0.0);
    } else if (a11 <= a22) {
        result.c1 = math.complex.init(1.0, 0.0);
        result.c2 = math.complex.init(0.0, 0.0);
    } else {
        result.c1 = math.complex.init(0.0, 0.0);
        result.c2 = math.complex.init(1.0, 0.0);
    }
    normalize_two_term_coefficients(&result.c1, &result.c2);
    return result;
}

fn normalize_two_term_coefficients(c1: *math.Complex, c2: *math.Complex) void {
    const c_norm = @sqrt(c1.r * c1.r + c1.i * c1.i + c2.r * c2.r + c2.i * c2.i);
    if (c_norm > 1e-30) {
        c1.* = math.complex.scale(c1.*, 1.0 / c_norm);
        c2.* = math.complex.scale(c2.*, 1.0 / c_norm);
    }
}

fn store_previous_band_state(
    psi: []const math.Complex,
    workspace: *CgWorkspace,
    state: *BandState,
) void {
    @memcpy(workspace.psi_old, psi);
    @memcpy(workspace.h_psi_old, workspace.h_psi);
    state.has_prev_direction = true;
}

fn apply_ritz_update(
    n: usize,
    psi: []math.Complex,
    workspace: *CgWorkspace,
    ritz: RitzResult,
) void {
    for (0..n) |i| {
        var value = math.complex.add(
            math.complex.mul(ritz.c1, workspace.psi_old[i]),
            math.complex.mul(ritz.c2, workspace.direction[i]),
        );
        if (ritz.use_3term) {
            value = math.complex.add(value, math.complex.mul(ritz.c3, workspace.psi_prev[i]));
        }
        psi[i] = value;
    }

    for (0..n) |i| {
        var value = math.complex.add(
            math.complex.mul(ritz.c1, workspace.h_psi_old[i]),
            math.complex.mul(ritz.c2, workspace.h_dir[i]),
        );
        if (ritz.use_3term) {
            value = math.complex.add(value, math.complex.mul(ritz.c3, workspace.h_psi_prev[i]));
        }
        workspace.h_psi[i] = value;
    }
    renormalize_band_state(n, psi, workspace.h_psi);
}

fn renormalize_band_state(n: usize, psi: []math.Complex, h_psi: []math.Complex) void {
    const psi_norm = common.vector_norm(n, psi);
    if (psi_norm > 1e-14) {
        scale_vec(n, psi, 1.0 / psi_norm);
        scale_vec(n, h_psi, 1.0 / psi_norm);
    }
}

fn update_band_eigenvalue(lambda: *f64, lambda_new: f64, tol: f64) bool {
    const eval_change = @abs(lambda_new - lambda.*);
    lambda.* = lambda_new;
    return eval_change < tol;
}

fn diagonalize_cg_subspace(
    alloc: std.mem.Allocator,
    op: common.Operator,
    n: usize,
    nbands: usize,
    values: []f64,
    vectors: []math.Complex,
) !void {
    const h_vectors = try alloc.alloc(math.Complex, n * nbands);
    defer alloc.free(h_vectors);

    for (0..nbands) |band| {
        try op.apply(
            op.ctx,
            common.column_const(vectors, n, band),
            common.column(h_vectors, n, band),
        );
    }

    const h_sub = try alloc.alloc(math.Complex, nbands * nbands);
    defer alloc.free(h_sub);

    common.build_projected(n, vectors, h_vectors, h_sub, nbands);

    var sub_eig = try common.hermitian_eigen_decomp_small(alloc, nbands, h_sub);
    defer sub_eig.deinit(alloc);

    const rotated = try alloc.alloc(math.Complex, n * nbands);
    defer alloc.free(rotated);

    for (0..nbands) |band| {
        common.combine_columns(
            n,
            vectors,
            nbands,
            common.column_const(sub_eig.vectors, nbands, band),
            common.column(rotated, n, band),
        );
    }
    @memcpy(vectors, rotated);
    @memcpy(values, sub_eig.values);
}

/// Orthogonalize v against columns 0..m-1 of basis (without normalization).
fn orthogonalize(n: usize, v: []math.Complex, basis: []const math.Complex, m: usize) void {
    const blas_v: []blas.Complex = @ptrCast(v[0..n]);
    for (0..m) |j| {
        const bj = common.column_const(basis, n, j);
        const dot = common.inner_product(n, bj, v);
        const neg_dot = blas.Complex.init(-dot.r, -dot.i);
        const blas_bj: []const blas.Complex = @ptrCast(bj);
        blas.zaxpy(neg_dot, blas_bj, blas_v);
    }
}

/// Complex AXPY: y += alpha * x (alpha is complex)
fn zaxpy_complex(n: usize, y: []math.Complex, x: []const math.Complex, alpha: math.Complex) void {
    if (n == 0) return;
    const blas_alpha = blas.Complex.init(alpha.r, alpha.i);
    const blas_x: []const blas.Complex = @ptrCast(x[0..n]);
    const blas_y: []blas.Complex = @ptrCast(y[0..n]);
    blas.zaxpy(blas_alpha, blas_x, blas_y);
}

/// Scale vector: v *= alpha
fn scale_vec(n: usize, v: []math.Complex, alpha: f64) void {
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
