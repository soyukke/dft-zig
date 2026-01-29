const std = @import("std");

const local_orbital = @import("local_orbital.zig");
const neighbor_list = @import("neighbor_list.zig");
const sparse = @import("sparse.zig");
const math = @import("../math/math.zig");

pub const PotentialGrid = struct {
    cell: math.Mat3,
    dims: [3]usize,
    values: []const f64,

    pub fn count(self: PotentialGrid) usize {
        return self.dims[0] * self.dims[1] * self.dims[2];
    }

    pub fn weight(self: PotentialGrid) !f64 {
        const volume = cellVolume(self.cell);
        const npoints = self.count();
        if (npoints == 0) return error.InvalidGrid;
        return volume / @as(f64, @floatFromInt(npoints));
    }

    pub fn point(self: PotentialGrid, ix: usize, iy: usize, iz: usize) math.Vec3 {
        const nx = @as(f64, @floatFromInt(self.dims[0]));
        const ny = @as(f64, @floatFromInt(self.dims[1]));
        const nz = @as(f64, @floatFromInt(self.dims[2]));
        const frac = math.Vec3{
            .x = (@as(f64, @floatFromInt(ix)) + 0.5) / nx,
            .y = (@as(f64, @floatFromInt(iy)) + 0.5) / ny,
            .z = (@as(f64, @floatFromInt(iz)) + 0.5) / nz,
        };
        return math.fracToCart(frac, self.cell);
    }
};

pub fn buildLocalPotentialCsr(
    alloc: std.mem.Allocator,
    orbitals: []const local_orbital.Orbital,
    neighbors: neighbor_list.NeighborList,
    grid: PotentialGrid,
    pbc: neighbor_list.Pbc,
) !sparse.CsrMatrix {
    const count = orbitals.len;
    if (neighbors.offsets.len != count + 1) return error.MismatchedLength;
    if (grid.values.len != grid.count()) return error.InvalidGrid;
    if (grid.count() == 0) return error.InvalidGrid;

    const inv_cell = try invertCell(grid.cell);
    const weight = try grid.weight();

    var triplets: std.ArrayList(sparse.Triplet) = .empty;
    defer triplets.deinit(alloc);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const diag = try integratePair(orbitals[i], orbitals[i], grid, inv_cell, pbc);
        if (diag != 0.0) {
            try triplets.append(alloc, .{ .row = i, .col = i, .value = diag * weight });
        }
        for (neighbors.neighborsOf(i)) |j| {
            if (j <= i) continue;
            const value = try integratePair(orbitals[i], orbitals[j], grid, inv_cell, pbc);
            if (value == 0.0) continue;
            const scaled = value * weight;
            try triplets.append(alloc, .{ .row = i, .col = j, .value = scaled });
            try triplets.append(alloc, .{ .row = j, .col = i, .value = scaled });
        }
    }

    return sparse.CsrMatrix.initFromTriplets(alloc, count, count, triplets.items);
}

pub fn buildLocalPotentialCsrFromCenters(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    sigma: f64,
    cutoff: f64,
    pbc: neighbor_list.Pbc,
    grid: PotentialGrid,
) !sparse.CsrMatrix {
    if (centers.len == 0) return error.InvalidShape;
    if (sigma <= 0.0) return error.InvalidSigma;
    var list = try neighbor_list.NeighborList.init(alloc, grid.cell, pbc, centers, cutoff);
    defer list.deinit(alloc);

    var orbitals = try alloc.alloc(local_orbital.Orbital, centers.len);
    defer alloc.free(orbitals);
    const alpha = 1.0 / (sigma * sigma);
    for (centers, 0..) |center, idx| {
        orbitals[idx] = .{ .center = center, .alpha = alpha, .cutoff = cutoff };
    }
    return buildLocalPotentialCsr(alloc, orbitals, list, grid, pbc);
}

fn integratePair(
    a: local_orbital.Orbital,
    b: local_orbital.Orbital,
    grid: PotentialGrid,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
) !f64 {
    var sum: f64 = 0.0;
    var iz: usize = 0;
    while (iz < grid.dims[2]) : (iz += 1) {
        var iy: usize = 0;
        while (iy < grid.dims[1]) : (iy += 1) {
            var ix: usize = 0;
            while (ix < grid.dims[0]) : (ix += 1) {
                const index = ix + grid.dims[0] * (iy + grid.dims[1] * iz);
                const value = grid.values[index];
                if (value == 0.0) continue;
                const point = grid.point(ix, iy, iz);
                const phi_a = orbitalValueAt(a, point, grid.cell, inv_cell, pbc);
                if (phi_a == 0.0) continue;
                const phi_b = orbitalValueAt(b, point, grid.cell, inv_cell, pbc);
                if (phi_b == 0.0) continue;
                sum += value * phi_a * phi_b;
            }
        }
    }
    return sum;
}

pub fn orbitalValueAt(
    orbital: local_orbital.Orbital,
    point: math.Vec3,
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
) f64 {
    if (orbital.alpha <= 0.0) return 0.0;
    const delta = math.Vec3.sub(point, orbital.center);
    const dvec = minimumImageDelta(cell, inv_cell, pbc, delta);
    const r2 = math.Vec3.dot(dvec, dvec);
    if (r2 > orbital.cutoff * orbital.cutoff) return 0.0;
    const norm = local_orbital.gaussianNorm(orbital.alpha);
    return norm * std.math.exp(-orbital.alpha * r2);
}

pub fn cellVolume(cell: math.Mat3) f64 {
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    return @abs(math.Vec3.dot(a1, math.Vec3.cross(a2, a3)));
}

pub fn invertCell(cell: math.Mat3) !math.Mat3 {
    const r0 = cell.row(0);
    const r1 = cell.row(1);
    const r2 = cell.row(2);
    const c0 = math.Vec3.cross(r1, r2);
    const det = math.Vec3.dot(r0, c0);
    if (@abs(det) < 1e-12) return error.SingularCell;
    const inv_det = 1.0 / det;
    const c1 = math.Vec3.cross(r2, r0);
    const c2 = math.Vec3.cross(r0, r1);
    return math.Mat3.fromRows(
        math.Vec3.scale(c0, inv_det),
        math.Vec3.scale(c1, inv_det),
        math.Vec3.scale(c2, inv_det),
    );
}

fn wrapFrac(value: f64) f64 {
    return value - std.math.floor(value + 0.5);
}

fn minimumImageDelta(cell: math.Mat3, inv_cell: math.Mat3, pbc: neighbor_list.Pbc, delta: math.Vec3) math.Vec3 {
    var frac = inv_cell.mulVec(delta);
    if (pbc.x) frac.x = wrapFrac(frac.x);
    if (pbc.y) frac.y = wrapFrac(frac.y);
    if (pbc.z) frac.z = wrapFrac(frac.z);
    return cell.mulVec(frac);
}

test "local potential constant matches overlap scaling" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    );
    const dims = [3]usize{ 16, 16, 16 };
    const count = dims[0] * dims[1] * dims[2];
    const values = try alloc.alloc(f64, count);
    defer alloc.free(values);
    @memset(values, 0.7);
    const grid_full = PotentialGrid{ .cell = cell, .dims = dims, .values = values };
    const centers = [_]math.Vec3{
        .{ .x = 2.0, .y = 2.0, .z = 2.0 },
        .{ .x = 3.0, .y = 2.0, .z = 2.0 },
    };
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const sigma = 0.5;
    const cutoff = 3.0;
    var local = try buildLocalPotentialCsrFromCenters(alloc, centers[0..], sigma, cutoff, pbc, grid_full);
    defer local.deinit(alloc);

    const alpha = 1.0 / (sigma * sigma);
    const orbitals = [_]local_orbital.Orbital{
        .{ .center = centers[0], .alpha = alpha, .cutoff = cutoff },
        .{ .center = centers[1], .alpha = alpha, .cutoff = cutoff },
    };
    const s00 = local_orbital.overlapIntegral(orbitals[0], orbitals[0]);
    const s01 = local_orbital.overlapIntegral(orbitals[0], orbitals[1]);
    const expected00 = 0.7 * s00;
    const expected01 = 0.7 * s01;
    try std.testing.expectApproxEqAbs(expected00, local.valueAt(0, 0), 2e-2);
    try std.testing.expectApproxEqAbs(expected01, local.valueAt(0, 1), 2e-2);
}
