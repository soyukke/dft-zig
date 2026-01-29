//! Serial (single-threaded) LOBPCG implementation
//!
//! Locally Optimal Block Preconditioned Conjugate Gradient method
//! for computing a few smallest eigenvalues of large sparse matrices.
//! Supports both standard (H·x = λ·x) and generalized (H·x = λ·S·x)
//! eigenvalue problems.

const std = @import("std");
const math = @import("../../math/math.zig");
const linalg = @import("../linalg.zig");
const common = @import("common.zig");

pub const Operator = common.Operator;
pub const Options = common.Options;

const debug_iterative = false;

/// Apply S operator to a single vector. If apply_s is null, y = x (identity).
fn applySingle(op: Operator, x: []const math.Complex, y: []math.Complex) !void {
    if (op.apply_s) |apply_s_fn| {
        try apply_s_fn(op.ctx, x, y);
    } else {
        @memcpy(y[0..op.n], x[0..op.n]);
    }
}

/// Apply S operator to multiple columns.
fn applySBatch(op: Operator, v: []const math.Complex, sv: []math.Complex, m: usize) !void {
    if (op.apply_s_batch) |batch_fn| {
        try batch_fn(op.ctx, v[0 .. op.n * m], sv[0 .. op.n * m], op.n, m);
    } else if (op.apply_s) |apply_s_fn| {
        for (0..m) |col| {
            try apply_s_fn(
                op.ctx,
                common.columnConst(v, op.n, col),
                common.column(sv, op.n, col),
            );
        }
    }
}

/// Apply H operator to multiple columns.
fn applyHBatch(op: Operator, v: []const math.Complex, w: []math.Complex, m: usize) !void {
    if (op.apply_batch) |batch_fn| {
        try batch_fn(op.ctx, v[0 .. op.n * m], w[0 .. op.n * m], op.n, m);
    } else {
        for (0..m) |col| {
            try op.apply(op.ctx, common.columnConst(v, op.n, col), common.column(w, op.n, col));
        }
    }
}

/// Serial LOBPCG eigenvalue solver
pub fn solve(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    op: Operator,
    diag: []const f64,
    nbands: usize,
    opts: Options,
) !linalg.EigenDecomp {
    if (nbands == 0 or op.n == 0) {
        return linalg.EigenDecomp{
            .values = try alloc.alloc(f64, 0),
            .vectors = try alloc.alloc(math.Complex, 0),
            .n = op.n,
        };
    }
    if (diag.len != op.n) return error.InvalidMatrixSize;

    const has_overlap = op.apply_s != null;

    // Subspace parameters
    const max_subspace = if (opts.max_subspace == 0)
        @min(op.n, 4 * nbands + 8)
    else
        @min(op.n, @max(opts.max_subspace, nbands));
    const block = if (opts.block_size == 0)
        @min(max_subspace, nbands + 4)
    else
        @min(max_subspace, @max(opts.block_size, nbands));

    if (debug_iterative) {
        std.debug.print("\n[LOBPCG] n={d} nbands={d} max_subspace={d} block={d} generalized={}\n", .{ op.n, nbands, max_subspace, block, has_overlap });
    }

    // Allocate workspace
    const v = try alloc.alloc(math.Complex, op.n * max_subspace);
    defer alloc.free(v);
    const w = try alloc.alloc(math.Complex, op.n * max_subspace);
    defer alloc.free(w);
    // S·v workspace (only for generalized problem)
    const sv = if (has_overlap) try alloc.alloc(math.Complex, op.n * max_subspace) else null;
    defer if (sv) |s| alloc.free(s);
    const ritz = try alloc.alloc(math.Complex, op.n * nbands);
    defer alloc.free(ritz);
    const residuals = try alloc.alloc(math.Complex, op.n * nbands);
    defer alloc.free(residuals);
    // S·ritz workspace for residual computation
    const s_ritz = if (has_overlap) try alloc.alloc(math.Complex, op.n * nbands) else null;
    defer if (s_ritz) |s| alloc.free(s);

    var seed: u64 = 0x6f3d3cbe;

    // Initialize vectors
    var m: usize = block;
    var init_cols: usize = 0;
    if (opts.init_vectors) |init_vecs| {
        const cols = if (opts.init_vectors_cols == 0) nbands else opts.init_vectors_cols;
        init_cols = @min(m, cols);
        if (init_vecs.len < op.n * init_cols) return error.InvalidMatrixSize;
    }

    if (opts.init_diagonal) {
        const order = try alloc.alloc(usize, op.n);
        defer alloc.free(order);
        for (order, 0..) |*idx, i| {
            idx.* = i;
        }
        std.sort.block(usize, order, diag, struct {
            fn lessThan(ctx: []const f64, a: usize, b: usize) bool {
                return ctx[a] < ctx[b];
            }
        }.lessThan);
        @memset(v[0 .. op.n * m], math.complex.init(0.0, 0.0));
        for (0..m) |col| {
            const idx = order[col];
            v[idx + col * op.n] = math.complex.init(1.0, 0.0);
        }
    } else {
        common.initRandomVectors(op.n, v, m, &seed);
    }

    if (opts.init_vectors) |init_vecs| {
        const count = op.n * init_cols;
        @memcpy(v[0..count], init_vecs[0..count]);
    }
    try common.orthonormalizeAll(op.n, v, m, &seed);

    // Apply H to initial vectors
    try applyHBatch(op, v, w, m);
    // Apply S to initial vectors (if generalized)
    if (has_overlap) try applySBatch(op, v, sv.?, m);

    const t = try alloc.alloc(math.Complex, max_subspace * max_subspace);
    defer alloc.free(t);
    // Overlap projected matrix T_S = V†·S·V (only for generalized)
    const ts = if (has_overlap) try alloc.alloc(math.Complex, max_subspace * max_subspace) else null;
    defer if (ts) |s| alloc.free(s);
    const last_values = try alloc.alloc(f64, nbands);
    defer alloc.free(last_values);

    // Main iteration loop
    var iter: usize = 0;
    while (iter < opts.max_iter) : (iter += 1) {
        // Build projected Hamiltonian T_H = V† * W
        common.buildProjected(op.n, v, w, t, m);

        var eig: linalg.EigenDecomp = undefined;
        if (has_overlap) {
            // Build projected overlap T_S = V† * S·V
            common.buildProjected(op.n, v, sv.?, ts.?, m);

            // Solve generalized eigenvalue problem T_H·c = λ·T_S·c
            const th_slice = t[0 .. m * m];
            const ts_slice = ts.?[0 .. m * m];
            eig = try common.hermitianGeneralizedEigenDecompSmall(alloc, m, th_slice, ts_slice);
        } else {
            // Standard eigenvalue problem
            const t_slice = t[0 .. m * m];
            eig = if (m <= 64)
                try common.hermitianEigenDecompSmall(alloc, m, t_slice)
            else
                try linalg.hermitianEigenDecomp(alloc, backend, m, t_slice);
        }
        defer eig.deinit(alloc);

        if (debug_iterative and iter == 0) {
            std.debug.print("[LOBPCG] iter=0 m={d} eigenvalues: ", .{m});
            for (0..@min(nbands + 2, eig.values.len)) |i| {
                std.debug.print("{d:.4} ", .{eig.values[i]});
            }
            std.debug.print("\n", .{});
        }

        @memcpy(last_values, eig.values[0..nbands]);

        // Compute Ritz vectors and residuals using batched zgemm
        common.combineColumnsMatrix(op.n, v, m, eig.vectors, nbands, ritz);
        common.combineColumnsMatrix(op.n, w, m, eig.vectors, nbands, residuals);
        var max_residual: f64 = 0.0;
        if (has_overlap) {
            // Residual: r = H·x - λ·S·x
            // Compute S·ritz
            common.combineColumnsMatrix(op.n, sv.?, m, eig.vectors, nbands, s_ritz.?);
            for (0..nbands) |b| {
                const lambda = eig.values[b];
                const sr = common.column(s_ritz.?, op.n, b);
                const r = common.column(residuals, op.n, b);
                // r = H·ritz - λ·S·ritz  (r already contains H·ritz via w projection)
                common.axpy(op.n, r, sr, -lambda);
                const norm = common.vectorNorm(op.n, r);
                if (norm > max_residual) max_residual = norm;
            }
        } else {
            for (0..nbands) |b| {
                const lambda = eig.values[b];
                const x = common.column(ritz, op.n, b);
                const r = common.column(residuals, op.n, b);
                common.axpy(op.n, r, x, -lambda);
                const norm = common.vectorNorm(op.n, r);
                if (norm > max_residual) max_residual = norm;
            }
        }

        if (debug_iterative) {
            std.debug.print("[LOBPCG] iter={d} m={d} max_residual={e:.4}\n", .{ iter, m, max_residual });
        }

        // Check convergence
        if (max_residual < opts.tol) {
            if (debug_iterative) {
                std.debug.print("[LOBPCG] Converged at iter={d}!\n", .{iter});
            }
            const values = try alloc.alloc(f64, nbands);
            const vectors = try alloc.alloc(math.Complex, op.n * nbands);
            @memcpy(values, eig.values[0..nbands]);
            @memcpy(vectors, ritz[0 .. op.n * nbands]);
            return linalg.EigenDecomp{ .values = values, .vectors = vectors, .n = op.n };
        }

        // Expand subspace with preconditioned residuals
        var added: usize = 0;
        for (0..nbands) |b| {
            if (m + added >= max_subspace) break;
            const r = common.column(residuals, op.n, b);
            const res_norm = common.vectorNorm(op.n, r);
            if (res_norm < opts.tol) continue;
            const lambda = eig.values[b];
            const q = common.column(v, op.n, m + added);
            common.precondition(op.n, diag, lambda, r, q);
            const norm = common.orthonormalizeVector(op.n, q, v, m + added);
            if (norm < 1e-8) continue;
            added += 1;
        }
        // Batch-apply operator to all newly added vectors
        if (added > 0) {
            try applyHBatch(op, v[op.n * m ..], w[op.n * m ..], added);
            if (has_overlap) try applySBatch(op, v[op.n * m ..], sv.?[op.n * m ..], added);
        }

        // If no vectors were added, all residuals are below tol individually.
        if (added == 0) {
            if (debug_iterative) {
                std.debug.print("[LOBPCG] All residuals below tol, treating as converged at iter={d}\n", .{iter});
            }
            const values = try alloc.alloc(f64, nbands);
            const vectors = try alloc.alloc(math.Complex, op.n * nbands);
            @memcpy(values, eig.values[0..nbands]);
            @memcpy(vectors, ritz[0 .. op.n * nbands]);
            return linalg.EigenDecomp{ .values = values, .vectors = vectors, .n = op.n };
        }

        // Restart if subspace is full
        if (m + added >= max_subspace) {
            @memcpy(v[0 .. op.n * nbands], ritz[0 .. op.n * nbands]);
            m = nbands;
            try common.orthonormalizeAll(op.n, v, m, &seed);
            try applyHBatch(op, v, w, m);
            if (has_overlap) try applySBatch(op, v, sv.?, m);
            continue;
        }

        m += added;
    }

    // Did not converge, return best estimate
    if (debug_iterative) {
        std.debug.print("[LOBPCG] Did not converge after {d} iterations\n", .{opts.max_iter});
    }
    const values = try alloc.alloc(f64, nbands);
    const vectors = try alloc.alloc(math.Complex, op.n * nbands);
    @memcpy(values, last_values);
    @memcpy(vectors, ritz[0 .. op.n * nbands]);
    return linalg.EigenDecomp{ .values = values, .vectors = vectors, .n = op.n };
}
