//! Parallel (multi-threaded) LOBPCG implementation
//!
//! Uses thread pool for:
//! - Applying operator to multiple vectors simultaneously
//! - Building projected matrix (inner products) - future
//!
//! Thread pool should be created at program start and passed through.
//! Supports both standard (H·x = λ·x) and generalized (H·x = λ·S·x)
//! eigenvalue problems.

const std = @import("std");

const math = @import("../../math/math.zig");
const linalg = @import("../linalg.zig");
const common = @import("common.zig");
const thread_pool = @import("../../thread_pool.zig");

pub const Operator = common.Operator;
pub const Options = common.Options;
pub const ThreadPool = thread_pool.ThreadPool;

const debug_parallel = false;

/// Extended options for parallel LOBPCG
pub const ParallelOptions = struct {
    base: Options = .{},
    pool: ?*ThreadPool = null, // External thread pool (preferred)
};

/// Parallel LOBPCG eigenvalue solver
pub fn solve(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    op: Operator,
    diag: []const f64,
    nbands: usize,
    opts: ParallelOptions,
) !linalg.EigenDecomp {
    // If no thread pool provided, fall back to serial
    const pool = opts.pool orelse {
        const serial = @import("serial.zig");
        return serial.solve(alloc, backend, op, diag, nbands, opts.base);
    };

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
    const max_subspace = if (opts.base.max_subspace == 0)
        @min(op.n, 4 * nbands + 8)
    else
        @min(op.n, @max(opts.base.max_subspace, nbands));
    const block = if (opts.base.block_size == 0)
        @min(max_subspace, nbands + 4)
    else
        @min(max_subspace, @max(opts.base.block_size, nbands));

    if (debug_parallel) {
        std.debug.print("\n[LOBPCG-PAR] n={d} nbands={d} threads={d} generalized={}\n", .{ op.n, nbands, pool.num_threads, has_overlap });
    }

    // Allocate workspace
    const v = try alloc.alloc(math.Complex, op.n * max_subspace);
    defer alloc.free(v);
    const w = try alloc.alloc(math.Complex, op.n * max_subspace);
    defer alloc.free(w);
    const sv = if (has_overlap) try alloc.alloc(math.Complex, op.n * max_subspace) else null;
    defer if (sv) |s| alloc.free(s);
    const ritz = try alloc.alloc(math.Complex, op.n * nbands);
    defer alloc.free(ritz);
    const residuals = try alloc.alloc(math.Complex, op.n * nbands);
    defer alloc.free(residuals);
    const s_ritz = if (has_overlap) try alloc.alloc(math.Complex, op.n * nbands) else null;
    defer if (s_ritz) |s| alloc.free(s);

    var seed: u64 = 0x6f3d3cbe;

    // Initialize vectors
    var m: usize = block;
    var init_cols: usize = 0;
    if (opts.base.init_vectors) |init_vecs| {
        const cols = if (opts.base.init_vectors_cols == 0) nbands else opts.base.init_vectors_cols;
        init_cols = @min(m, cols);
        if (init_vecs.len < op.n * init_cols) return error.InvalidMatrixSize;
    }

    if (opts.base.init_diagonal) {
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

    if (opts.base.init_vectors) |init_vecs| {
        const count = op.n * init_cols;
        @memcpy(v[0..count], init_vecs[0..count]);
    }
    try common.orthonormalizeAll(op.n, v, m, &seed);

    // Apply operator to initial vectors (PARALLEL)
    try applyOperatorParallel(pool, op, v, w, m);
    if (has_overlap) try applySParallel(pool, op, v, sv.?, m);

    const t = try alloc.alloc(math.Complex, max_subspace * max_subspace);
    defer alloc.free(t);
    const ts = if (has_overlap) try alloc.alloc(math.Complex, max_subspace * max_subspace) else null;
    defer if (ts) |s| alloc.free(s);
    const last_values = try alloc.alloc(f64, nbands);
    defer alloc.free(last_values);

    // Main iteration loop
    var iter: usize = 0;
    while (iter < opts.base.max_iter) : (iter += 1) {
        // Build projected matrix T = V† * W (PARALLEL)
        buildProjectedParallel(pool, op.n, v, w, t, m);

        var eig: linalg.EigenDecomp = undefined;
        if (has_overlap) {
            buildProjectedParallel(pool, op.n, v, sv.?, ts.?, m);
            const th_slice = t[0 .. m * m];
            const ts_slice = ts.?[0 .. m * m];
            eig = try common.hermitianGeneralizedEigenDecompSmall(alloc, m, th_slice, ts_slice);
        } else {
            const t_slice = t[0 .. m * m];
            eig = if (m <= 64)
                try common.hermitianEigenDecompSmall(alloc, m, t_slice)
            else
                try linalg.hermitianEigenDecomp(alloc, backend, m, t_slice);
        }
        defer eig.deinit(alloc);

        @memcpy(last_values, eig.values[0..nbands]);

        // Compute Ritz vectors and residuals
        var max_residual: f64 = 0.0;
        if (has_overlap) {
            common.combineColumnsMatrix(op.n, v, m, eig.vectors, nbands, ritz);
            common.combineColumnsMatrix(op.n, w, m, eig.vectors, nbands, residuals);
            common.combineColumnsMatrix(op.n, sv.?, m, eig.vectors, nbands, s_ritz.?);
            for (0..nbands) |b| {
                const lambda = eig.values[b];
                const sr = common.column(s_ritz.?, op.n, b);
                const r = common.column(residuals, op.n, b);
                common.axpy(op.n, r, sr, -lambda);
                const norm = common.vectorNorm(op.n, r);
                if (norm > max_residual) max_residual = norm;
            }
        } else {
            for (0..nbands) |b| {
                const lambda = eig.values[b];
                const y = common.columnConst(eig.vectors, m, b);
                const x = common.column(ritz, op.n, b);
                const r = common.column(residuals, op.n, b);
                common.combineColumns(op.n, v, m, y, x);
                common.combineColumns(op.n, w, m, y, r);
                common.axpy(op.n, r, x, -lambda);
                const norm = common.vectorNorm(op.n, r);
                if (norm > max_residual) max_residual = norm;
            }
        }

        if (debug_parallel) {
            std.debug.print("[LOBPCG-PAR] iter={d} m={d} max_residual={e:.4}\n", .{ iter, m, max_residual });
        }

        // Check convergence
        if (max_residual < opts.base.tol) {
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
            if (res_norm < opts.base.tol) continue;
            const lambda = eig.values[b];
            const q = common.column(v, op.n, m + added);
            common.precondition(op.n, diag, lambda, r, q);
            const norm = common.orthonormalizeVector(op.n, q, v, m + added);
            if (norm < 1e-8) continue;
            added += 1;
        }

        // Apply operator to new vectors (PARALLEL if multiple)
        if (added > 0) {
            try applyOperatorParallelRange(pool, op, v, w, m, added);
            if (has_overlap) try applySParallelRange(pool, op, v, sv.?, m, added);
        }

        // If no vectors were added, all residuals are below tol individually.
        if (added == 0) {
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
            try applyOperatorParallel(pool, op, v, w, m);
            if (has_overlap) try applySParallel(pool, op, v, sv.?, m);
            continue;
        }

        m += added;
    }

    // Did not converge
    const values = try alloc.alloc(f64, nbands);
    const vectors = try alloc.alloc(math.Complex, op.n * nbands);
    @memcpy(values, last_values);
    @memcpy(vectors, ritz[0 .. op.n * nbands]);
    return linalg.EigenDecomp{ .values = values, .vectors = vectors, .n = op.n };
}

/// Apply operator to columns [0, count) in parallel
fn applyOperatorParallel(
    pool: *ThreadPool,
    op: Operator,
    v: []const math.Complex,
    w: []math.Complex,
    count: usize,
) !void {
    if (count == 0) return;

    // For small count, sequential may be faster due to thread overhead
    if (count < 4) {
        for (0..count) |col| {
            try op.apply(op.ctx, common.columnConst(v, op.n, col), common.column(w, op.n, col));
        }
        return;
    }

    const Ctx = struct {
        op: Operator,
        v: []const math.Complex,
        w: []math.Complex,
    };
    const ctx = Ctx{ .op = op, .v = v, .w = w };

    try pool.parallelForWithError(count, ctx, struct {
        fn run(c: Ctx, col: usize) anyerror!void {
            const v_col = common.columnConst(c.v, c.op.n, col);
            const w_col = common.column(c.w, c.op.n, col);
            try c.op.apply(c.op.ctx, v_col, w_col);
        }
    }.run);
}

/// Apply S operator to columns [0, count) in parallel
fn applySParallel(
    pool: *ThreadPool,
    op: Operator,
    v: []const math.Complex,
    sv: []math.Complex,
    count: usize,
) !void {
    const apply_s_fn = op.apply_s orelse return;
    if (count == 0) return;

    if (count < 4) {
        for (0..count) |col| {
            try apply_s_fn(op.ctx, common.columnConst(v, op.n, col), common.column(sv, op.n, col));
        }
        return;
    }

    const Ctx = struct {
        apply_fn: *const fn (ctx: *anyopaque, x: []const math.Complex, y: []math.Complex) anyerror!void,
        ctx_ptr: *anyopaque,
        n: usize,
        v: []const math.Complex,
        sv: []math.Complex,
    };
    const ctx = Ctx{ .apply_fn = apply_s_fn, .ctx_ptr = op.ctx, .n = op.n, .v = v, .sv = sv };

    try pool.parallelForWithError(count, ctx, struct {
        fn run(c: Ctx, col: usize) anyerror!void {
            const v_col = common.columnConst(c.v, c.n, col);
            const sv_col = common.column(c.sv, c.n, col);
            try c.apply_fn(c.ctx_ptr, v_col, sv_col);
        }
    }.run);
}

/// Build projected matrix T[i,j] = <v[i]|w[j]> in parallel
fn buildProjectedParallel(
    pool: *ThreadPool,
    n: usize,
    v: []const math.Complex,
    w: []const math.Complex,
    out: []math.Complex,
    m: usize,
) void {
    // For small m, serial is faster
    if (m < 8) {
        common.buildProjected(n, v, w, out, m);
        return;
    }

    // Parallelize over columns j
    const Ctx = struct {
        n: usize,
        m: usize,
        v: []const math.Complex,
        w: []const math.Complex,
        out: []math.Complex,
    };
    const ctx = Ctx{ .n = n, .m = m, .v = v, .w = w, .out = out };

    pool.parallelFor(m, ctx, struct {
        fn run(c: Ctx, j: usize) void {
            for (0..j + 1) |i| {
                const val = common.innerProduct(c.n, common.columnConst(c.v, c.n, i), common.columnConst(c.w, c.n, j));
                c.out[i + j * c.m] = val;
                if (i != j) {
                    c.out[j + i * c.m] = math.complex.conj(val);
                }
            }
        }
    }.run);
}

/// Apply operator to columns [start, start+count) in parallel
fn applyOperatorParallelRange(
    pool: *ThreadPool,
    op: Operator,
    v: []const math.Complex,
    w: []math.Complex,
    start: usize,
    count: usize,
) !void {
    if (count == 0) return;

    // For small count, sequential
    if (count < 4) {
        for (0..count) |i| {
            const col = start + i;
            try op.apply(op.ctx, common.columnConst(v, op.n, col), common.column(w, op.n, col));
        }
        return;
    }

    const Ctx = struct {
        op: Operator,
        v: []const math.Complex,
        w: []math.Complex,
        start: usize,
    };
    const ctx = Ctx{ .op = op, .v = v, .w = w, .start = start };

    try pool.parallelForWithError(count, ctx, struct {
        fn run(c: Ctx, i: usize) anyerror!void {
            const col = c.start + i;
            const v_col = common.columnConst(c.v, c.op.n, col);
            const w_col = common.column(c.w, c.op.n, col);
            try c.op.apply(c.op.ctx, v_col, w_col);
        }
    }.run);
}

/// Apply S operator to columns [start, start+count) in parallel
fn applySParallelRange(
    pool: *ThreadPool,
    op: Operator,
    v: []const math.Complex,
    sv: []math.Complex,
    start: usize,
    count: usize,
) !void {
    const apply_s_fn = op.apply_s orelse return;
    if (count == 0) return;

    if (count < 4) {
        for (0..count) |i| {
            const col = start + i;
            try apply_s_fn(op.ctx, common.columnConst(v, op.n, col), common.column(sv, op.n, col));
        }
        return;
    }

    const Ctx = struct {
        apply_fn: *const fn (ctx: *anyopaque, x: []const math.Complex, y: []math.Complex) anyerror!void,
        ctx_ptr: *anyopaque,
        n: usize,
        v: []const math.Complex,
        sv: []math.Complex,
        start: usize,
    };
    const ctx = Ctx{ .apply_fn = apply_s_fn, .ctx_ptr = op.ctx, .n = op.n, .v = v, .sv = sv, .start = start };

    try pool.parallelForWithError(count, ctx, struct {
        fn run(c: Ctx, i: usize) anyerror!void {
            const col = c.start + i;
            const v_col = common.columnConst(c.v, c.n, col);
            const sv_col = common.column(c.sv, c.n, col);
            try c.apply_fn(c.ctx_ptr, v_col, sv_col);
        }
    }.run);
}
