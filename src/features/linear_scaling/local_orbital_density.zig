const std = @import("std");

const density_matrix = @import("density_matrix.zig");
const local_orbital = @import("local_orbital.zig");
const neighbor_list = @import("neighbor_list.zig");
const sparse = @import("sparse.zig");
const math = @import("../math/math.zig");

pub const DensityPipelineResult = struct {
    overlap: sparse.CsrMatrix,
    density: sparse.CsrMatrix,
    iterations: usize,

    pub fn deinit(self: *DensityPipelineResult, alloc: std.mem.Allocator) void {
        self.overlap.deinit(alloc);
        self.density.deinit(alloc);
        self.* = undefined;
    }
};

pub const DensityPipelineOptions = struct {
    sigma: f64,
    cutoff: f64,
    iterations: usize,
    threshold: f64,
    electrons: ?f64 = null,
};

pub fn buildDensityFromCenters(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: DensityPipelineOptions,
) !DensityPipelineResult {
    if (centers.len == 0) return error.InvalidShape;
    var overlap = try local_orbital.buildOverlapCsrFromCenters(
        alloc,
        centers,
        opts.sigma,
        opts.cutoff,
        pbc,
        cell,
    );
    errdefer overlap.deinit(alloc);
    var density = try density_matrix.mcWeenyNonOrthogonal(
        alloc,
        overlap,
        overlap,
        opts.iterations,
        opts.threshold,
    );
    errdefer density.deinit(alloc);
    if (opts.electrons) |target| {
        try density_matrix.normalizeTraceOverlap(&density, overlap, target);
    }
    return .{ .overlap = overlap, .density = density, .iterations = opts.iterations };
}

test "density pipeline preserves symmetry" {
    const alloc = std.testing.allocator;
    const centers = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 1.0 },
    };
    const cell = math.Mat3.fromRows(
        .{ .x = 3.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 3.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 3.0 },
    );
    const pbc = neighbor_list.Pbc{ .x = true, .y = true, .z = true };
    const opts = DensityPipelineOptions{
        .sigma = 0.5,
        .cutoff = 2.0,
        .iterations = 2,
        .threshold = 0.0,
        .electrons = 4.0,
    };
    var result = try buildDensityFromCenters(alloc, centers[0..], cell, pbc, opts);
    defer result.deinit(alloc);

    try std.testing.expectApproxEqAbs(
        result.density.valueAt(0, 1),
        result.density.valueAt(1, 0),
        1e-12,
    );
    const trace_val = try density_matrix.traceOverlap(result.density, result.overlap);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), trace_val, 1e-10);
}
