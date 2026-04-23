const std = @import("std");

const sparse = @import("sparse.zig");

pub fn mcWeeny(
    alloc: std.mem.Allocator,
    initial: sparse.CsrMatrix,
    iterations: usize,
    threshold: f64,
) !sparse.CsrMatrix {
    var current = try sparse.clone(alloc, initial);
    errdefer current.deinit(alloc);

    var iter: usize = 0;
    while (iter < iterations) : (iter += 1) {
        var d2 = try sparse.mul(alloc, current, current, threshold);
        defer d2.deinit(alloc);

        var d3 = try sparse.mul(alloc, d2, current, threshold);
        defer d3.deinit(alloc);

        const next = try sparse.addScaled(alloc, d2, 3.0, d3, -2.0, threshold);
        current.deinit(alloc);
        current = next;
    }
    return current;
}

pub fn mcWeenyNonOrthogonal(
    alloc: std.mem.Allocator,
    initial: sparse.CsrMatrix,
    overlap: sparse.CsrMatrix,
    iterations: usize,
    threshold: f64,
) !sparse.CsrMatrix {
    if (initial.nrows != overlap.nrows or initial.ncols != overlap.ncols) return error.InvalidShape;
    var current = try sparse.clone(alloc, initial);
    errdefer current.deinit(alloc);

    var iter: usize = 0;
    while (iter < iterations) : (iter += 1) {
        var ps = try sparse.mul(alloc, current, overlap, threshold);
        defer ps.deinit(alloc);

        var psp = try sparse.mul(alloc, ps, current, threshold);
        defer psp.deinit(alloc);

        var psp_s = try sparse.mul(alloc, psp, overlap, threshold);
        defer psp_s.deinit(alloc);

        var psp_sp = try sparse.mul(alloc, psp_s, current, threshold);
        defer psp_sp.deinit(alloc);

        const next = try sparse.addScaled(alloc, psp, 3.0, psp_sp, -2.0, threshold);
        current.deinit(alloc);
        current = next;
    }
    return current;
}

pub fn traceOverlap(density: sparse.CsrMatrix, overlap: sparse.CsrMatrix) !f64 {
    return sparse.traceProduct(density, overlap);
}

pub fn normalizeTrace(matrix: *sparse.CsrMatrix, target: f64) !void {
    if (target <= 0.0) return error.InvalidTraceTarget;
    const current = sparse.trace(matrix.*);
    if (current == 0.0) return error.ZeroTrace;
    sparse.scaleInPlace(matrix, target / current);
}

pub fn normalizeTraceOverlap(
    matrix: *sparse.CsrMatrix,
    overlap: sparse.CsrMatrix,
    target: f64,
) !void {
    if (target <= 0.0) return error.InvalidTraceTarget;
    const current = try traceOverlap(matrix.*, overlap);
    if (current == 0.0) return error.ZeroTrace;
    sparse.scaleInPlace(matrix, target / current);
}

pub fn densityFromHamiltonian(
    alloc: std.mem.Allocator,
    hamiltonian: sparse.CsrMatrix,
    overlap: sparse.CsrMatrix,
    electrons: f64,
    iterations: usize,
    threshold: f64,
) !sparse.CsrMatrix {
    if (hamiltonian.nrows != overlap.nrows or hamiltonian.ncols != overlap.ncols)
        return error.InvalidShape;
    if (electrons <= 0.0) return error.InvalidTraceTarget;
    const diag = try sparse.diagonalValues(alloc, hamiltonian);
    defer alloc.free(diag);

    var min_val = diag[0];
    var max_val = diag[0];
    for (diag[1..]) |value| {
        min_val = @min(min_val, value);
        max_val = @max(max_val, value);
    }
    const denom = max_val - min_val;

    // If spectrum is flat (all diagonal elements equal), use uniform initial density
    if (denom <= 1e-12) {
        const n = hamiltonian.nrows;
        const fill = electrons / @as(f64, @floatFromInt(n));
        var density = try sparse.diagonal(alloc, n, fill);
        errdefer density.deinit(alloc);
        try normalizeTraceOverlap(&density, overlap, electrons);
        return density;
    }

    const shift = max_val / denom;
    const scale = -1.0 / denom;

    var diag_mat = try sparse.diagonal(alloc, hamiltonian.nrows, shift);
    defer diag_mat.deinit(alloc);

    var p0 = try sparse.addScaled(alloc, diag_mat, 1.0, hamiltonian, scale, threshold);
    defer p0.deinit(alloc);

    var density = try mcWeenyNonOrthogonal(alloc, p0, overlap, iterations, threshold);
    errdefer density.deinit(alloc);
    try normalizeTraceOverlap(&density, overlap, electrons);
    return density;
}

test "mcWeeny keeps identity" {
    const alloc = std.testing.allocator;
    const triplets = [_]sparse.Triplet{
        .{ .row = 0, .col = 0, .value = 1.0 },
        .{ .row = 1, .col = 1, .value = 1.0 },
    };
    var identity = try sparse.CsrMatrix.initFromTriplets(alloc, 2, 2, triplets[0..]);
    defer identity.deinit(alloc);

    var purified = try mcWeeny(alloc, identity, 2, 0.0);
    defer purified.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), purified.valueAt(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), purified.valueAt(1, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), purified.valueAt(0, 1), 1e-12);
}

test "mcWeenyNonOrthogonal matches orthogonal when overlap is identity" {
    const alloc = std.testing.allocator;
    var overlap = try sparse.diagonal(alloc, 2, 1.0);
    defer overlap.deinit(alloc);

    const triplets = [_]sparse.Triplet{
        .{ .row = 0, .col = 0, .value = 0.6 },
        .{ .row = 1, .col = 1, .value = 0.4 },
    };
    var initial = try sparse.CsrMatrix.initFromTriplets(alloc, 2, 2, triplets[0..]);
    defer initial.deinit(alloc);

    var ortho = try mcWeeny(alloc, initial, 1, 0.0);
    defer ortho.deinit(alloc);

    var non_ortho = try mcWeenyNonOrthogonal(alloc, initial, overlap, 1, 0.0);
    defer non_ortho.deinit(alloc);

    try std.testing.expectApproxEqAbs(ortho.valueAt(0, 0), non_ortho.valueAt(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(ortho.valueAt(1, 1), non_ortho.valueAt(1, 1), 1e-12);
}

test "normalizeTrace rescales density" {
    const alloc = std.testing.allocator;
    var diag = try sparse.diagonal(alloc, 2, 1.0);
    defer diag.deinit(alloc);

    try normalizeTrace(&diag, 4.0);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), sparse.trace(diag), 1e-12);
}

test "normalizeTraceOverlap rescales using overlap" {
    const alloc = std.testing.allocator;
    var density = try sparse.diagonal(alloc, 2, 1.0);
    defer density.deinit(alloc);

    var overlap = try sparse.diagonal(alloc, 2, 2.0);
    defer overlap.deinit(alloc);

    try normalizeTraceOverlap(&density, overlap, 2.0);
    const value = try traceOverlap(density, overlap);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), value, 1e-12);
}

test "densityFromHamiltonian normalizes trace" {
    const alloc = std.testing.allocator;
    const h_triplets = [_]sparse.Triplet{
        .{ .row = 0, .col = 0, .value = 0.0 },
        .{ .row = 1, .col = 1, .value = 1.0 },
    };
    var h = try sparse.CsrMatrix.initFromTriplets(alloc, 2, 2, h_triplets[0..]);
    defer h.deinit(alloc);

    var s = try sparse.diagonal(alloc, 2, 1.0);
    defer s.deinit(alloc);

    var density = try densityFromHamiltonian(alloc, h, s, 1.0, 2, 0.0);
    defer density.deinit(alloc);

    const trace_val = try traceOverlap(density, s);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), trace_val, 1e-10);
}
