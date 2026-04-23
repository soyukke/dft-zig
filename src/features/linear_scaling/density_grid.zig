const std = @import("std");

const local_orbital = @import("local_orbital.zig");
const local_orbital_potential = @import("local_orbital_potential.zig");
const neighbor_list = @import("neighbor_list.zig");
const sparse = @import("sparse.zig");
const math = @import("../math/math.zig");

pub fn buildDensityGridFromCenters(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    density: sparse.CsrMatrix,
    sigma: f64,
    cutoff: f64,
    pbc: neighbor_list.Pbc,
    grid: local_orbital_potential.PotentialGrid,
) ![]f64 {
    if (centers.len == 0) return error.InvalidShape;
    if (density.nrows != centers.len or density.ncols != centers.len) return error.InvalidShape;
    if (sigma <= 0.0) return error.InvalidSigma;
    const count = grid.count();
    if (count == 0) return error.InvalidGrid;

    var orbitals = try alloc.alloc(local_orbital.Orbital, centers.len);
    defer alloc.free(orbitals);

    const alpha = 1.0 / (sigma * sigma);
    for (centers, 0..) |center, idx| {
        orbitals[idx] = .{ .center = center, .alpha = alpha, .cutoff = cutoff };
    }

    var cell_list = try OrbitalCellList.init(alloc, grid.cell, pbc, centers, cutoff);
    defer cell_list.deinit(alloc);

    const inv_cell = try local_orbital_potential.invertCell(grid.cell);
    const values = try alloc.alloc(f64, count);
    @memset(values, 0.0);

    var candidates: std.ArrayList(usize) = .empty;
    defer candidates.deinit(alloc);

    var phi_map = std.AutoHashMap(usize, f64).init(alloc);
    defer phi_map.deinit();

    var idx: usize = 0;
    var iz: usize = 0;
    while (iz < grid.dims[2]) : (iz += 1) {
        var iy: usize = 0;
        while (iy < grid.dims[1]) : (iy += 1) {
            var ix: usize = 0;
            while (ix < grid.dims[0]) : (ix += 1) {
                const point = grid.point(ix, iy, iz);
                try cell_list.collectCandidates(alloc, point, &candidates);
                phi_map.clearRetainingCapacity();
                for (candidates.items) |orb_idx| {
                    const phi = local_orbital_potential.orbitalValueAt(
                        orbitals[orb_idx],
                        point,
                        grid.cell,
                        inv_cell,
                        pbc,
                    );
                    if (phi == 0.0) continue;
                    try phi_map.put(orb_idx, phi);
                }

                var rho: f64 = 0.0;
                var it = phi_map.iterator();
                while (it.next()) |entry| {
                    const row = entry.key_ptr.*;
                    const phi_i = entry.value_ptr.*;
                    const start = density.row_ptr[row];
                    const end = density.row_ptr[row + 1];
                    var di: usize = start;
                    while (di < end) : (di += 1) {
                        const col = density.col_idx[di];
                        if (phi_map.get(col)) |phi_j| {
                            rho += density.values[di] * phi_i * phi_j;
                        }
                    }
                }

                values[idx] = rho;
                idx += 1;
            }
        }
    }
    return values;
}

const OrbitalCellList = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    heads: []i64,
    next: []i64,
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    cutoff: f64,

    pub fn init(
        alloc: std.mem.Allocator,
        cell: math.Mat3,
        pbc: neighbor_list.Pbc,
        positions: []const math.Vec3,
        cutoff: f64,
    ) !OrbitalCellList {
        if (cutoff <= 0.0) return error.InvalidCutoff;
        const inv_cell = try local_orbital_potential.invertCell(cell);
        const lengths = cellLengths(cell);
        var nx = @max(@as(usize, @intFromFloat(std.math.floor(lengths.x / cutoff))), 1);
        var ny = @max(@as(usize, @intFromFloat(std.math.floor(lengths.y / cutoff))), 1);
        var nz = @max(@as(usize, @intFromFloat(std.math.floor(lengths.z / cutoff))), 1);
        if (!pbc.x) nx = @max(nx, 1);
        if (!pbc.y) ny = @max(ny, 1);
        if (!pbc.z) nz = @max(nz, 1);
        const cell_count = nx * ny * nz;

        const heads = try alloc.alloc(i64, cell_count);
        errdefer alloc.free(heads);
        @memset(heads, -1);
        const next = try alloc.alloc(i64, positions.len);
        errdefer alloc.free(next);
        @memset(next, -1);

        var idx: usize = 0;
        while (idx < positions.len) : (idx += 1) {
            const frac = inv_cell.mulVec(positions[idx]);
            const ix = clampCellIndex(frac.x, nx);
            const iy = clampCellIndex(frac.y, ny);
            const iz = clampCellIndex(frac.z, nz);
            const cell_index = ix + nx * (iy + ny * iz);
            next[idx] = heads[cell_index];
            heads[cell_index] = @as(i64, @intCast(idx));
        }

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .heads = heads,
            .next = next,
            .cell = cell,
            .inv_cell = inv_cell,
            .pbc = pbc,
            .cutoff = cutoff,
        };
    }

    pub fn deinit(self: *OrbitalCellList, alloc: std.mem.Allocator) void {
        if (self.heads.len > 0) alloc.free(self.heads);
        if (self.next.len > 0) alloc.free(self.next);
        self.* = undefined;
    }

    pub fn collectCandidates(
        self: *OrbitalCellList,
        gpa: std.mem.Allocator,
        point: math.Vec3,
        list: *std.ArrayList(usize),
    ) !void {
        list.clearRetainingCapacity();
        const frac = self.inv_cell.mulVec(point);
        const ix = clampCellIndex(frac.x, self.nx);
        const iy = clampCellIndex(frac.y, self.ny);
        const iz = clampCellIndex(frac.z, self.nz);
        var dx: i32 = -1;
        while (dx <= 1) : (dx += 1) {
            var dy: i32 = -1;
            while (dy <= 1) : (dy += 1) {
                var dz: i32 = -1;
                while (dz <= 1) : (dz += 1) {
                    if (!self.pbc.x and (ix == 0 and dx < 0)) continue;
                    if (!self.pbc.x and (ix + 1 == self.nx and dx > 0)) continue;
                    if (!self.pbc.y and (iy == 0 and dy < 0)) continue;
                    if (!self.pbc.y and (iy + 1 == self.ny and dy > 0)) continue;
                    if (!self.pbc.z and (iz == 0 and dz < 0)) continue;
                    if (!self.pbc.z and (iz + 1 == self.nz and dz > 0)) continue;

                    const nx_i = wrapCellIndex(ix, self.nx, dx, self.pbc.x);
                    const ny_i = wrapCellIndex(iy, self.ny, dy, self.pbc.y);
                    const nz_i = wrapCellIndex(iz, self.nz, dz, self.pbc.z);
                    const cell_index = nx_i + self.nx * (ny_i + self.ny * nz_i);
                    var atom = self.heads[cell_index];
                    while (atom >= 0) : (atom = self.next[@as(usize, @intCast(atom))]) {
                        try list.append(gpa, @as(usize, @intCast(atom)));
                    }
                }
            }
        }
    }
};

fn cellLengths(cell: math.Mat3) math.Vec3 {
    return .{
        .x = math.Vec3.norm(cell.row(0)),
        .y = math.Vec3.norm(cell.row(1)),
        .z = math.Vec3.norm(cell.row(2)),
    };
}

fn clampCellIndex(frac: f64, n: usize) usize {
    var t = frac - std.math.floor(frac);
    if (t < 0.0) t += 1.0;
    if (t >= 1.0) t -= 1.0;
    const idx = @as(usize, @intFromFloat(std.math.floor(t * @as(f64, @floatFromInt(n)))));
    return if (idx >= n) n - 1 else idx;
}

fn wrapCellIndex(index: usize, n: usize, delta: i32, enabled: bool) usize {
    if (!enabled) return index;
    const ni = @as(i32, @intCast(n));
    const idx = @mod(@as(i32, @intCast(index)) + delta, ni);
    return @as(usize, @intCast(idx));
}

test "density grid integrates to one for normalized orbital" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 6.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 6.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 6.0 },
    );
    const dims = [3]usize{ 20, 20, 20 };
    const grid = local_orbital_potential.PotentialGrid{
        .cell = cell,
        .dims = dims,
        .values = &[_]f64{},
    };
    const centers = [_]math.Vec3{.{ .x = 3.0, .y = 3.0, .z = 3.0 }};
    const sigma = 0.8;
    const cutoff = 4.0;
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const triplets = [_]sparse.Triplet{.{ .row = 0, .col = 0, .value = 1.0 }};
    var density_matrix = try sparse.CsrMatrix.initFromTriplets(alloc, 1, 1, triplets[0..]);
    defer density_matrix.deinit(alloc);

    const rho = try buildDensityGridFromCenters(
        alloc,
        centers[0..],
        density_matrix,
        sigma,
        cutoff,
        pbc,
        grid,
    );
    defer alloc.free(rho);

    const dv = try grid.weight();
    var sum: f64 = 0.0;
    for (rho) |value| {
        sum += value * dv;
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 2e-2);
}
