const std = @import("std");

const math = @import("../math/math.zig");
const structures = @import("structures.zig");

pub const Pbc = struct {
    x: bool,
    y: bool,
    z: bool,
};

pub const NeighborList = struct {
    offsets: []usize,
    neighbors: []usize,
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: Pbc,
    cutoff: f64,

    pub fn init(
        alloc: std.mem.Allocator,
        cell: math.Mat3,
        pbc: Pbc,
        positions: []const math.Vec3,
        cutoff: f64,
    ) !NeighborList {
        if (cutoff <= 0.0) return error.InvalidCutoff;
        const inv_cell = try invertCell(cell);
        const count = positions.len;

        const lists = try alloc.alloc(std.ArrayList(usize), count);
        errdefer {
            for (lists) |*list| {
                list.deinit(alloc);
            }
            alloc.free(lists);
        }
        for (lists) |*list| {
            list.* = .empty;
        }

        var i: usize = 0;
        while (i < count) : (i += 1) {
            var j: usize = i + 1;
            while (j < count) : (j += 1) {
                const delta = math.Vec3.sub(positions[j], positions[i]);
                const dvec = minimumImageDelta(cell, inv_cell, pbc, delta);
                const dist2 = math.Vec3.dot(dvec, dvec);
                if (dist2 <= cutoff * cutoff) {
                    try lists[i].append(alloc, j);
                    try lists[j].append(alloc, i);
                }
            }
        }

        const offsets = try alloc.alloc(usize, count + 1);
        errdefer alloc.free(offsets);
        offsets[0] = 0;
        var total: usize = 0;
        i = 0;
        while (i < count) : (i += 1) {
            total += lists[i].items.len;
            offsets[i + 1] = total;
        }

        const neighbors = try alloc.alloc(usize, total);
        errdefer alloc.free(neighbors);
        var cursor: usize = 0;
        i = 0;
        while (i < count) : (i += 1) {
            for (lists[i].items) |index| {
                neighbors[cursor] = index;
                cursor += 1;
            }
        }

        for (lists) |*list| {
            list.deinit(alloc);
        }
        alloc.free(lists);

        return .{
            .offsets = offsets,
            .neighbors = neighbors,
            .cell = cell,
            .inv_cell = inv_cell,
            .pbc = pbc,
            .cutoff = cutoff,
        };
    }

    pub fn initCellList(
        alloc: std.mem.Allocator,
        cell: math.Mat3,
        pbc: Pbc,
        positions: []const math.Vec3,
        cutoff: f64,
    ) !NeighborList {
        if (cutoff <= 0.0) return error.InvalidCutoff;
        const inv_cell = try invertCell(cell);
        const count = positions.len;

        const lengths = cellLengths(cell);
        var nx = @max(@as(usize, @intFromFloat(std.math.floor(lengths.x / cutoff))), 1);
        var ny = @max(@as(usize, @intFromFloat(std.math.floor(lengths.y / cutoff))), 1);
        var nz = @max(@as(usize, @intFromFloat(std.math.floor(lengths.z / cutoff))), 1);
        if (!pbc.x) nx = @max(nx, 1);
        if (!pbc.y) ny = @max(ny, 1);
        if (!pbc.z) nz = @max(nz, 1);
        const cell_count = nx * ny * nz;

        const heads = try alloc.alloc(i64, cell_count);
        defer alloc.free(heads);
        @memset(heads, -1);
        const next = try alloc.alloc(i64, count);
        defer alloc.free(next);
        @memset(next, -1);

        var idx: usize = 0;
        while (idx < count) : (idx += 1) {
            const frac = fracFromCart(positions[idx], inv_cell);
            const ix = clampCellIndex(frac.x, nx);
            const iy = clampCellIndex(frac.y, ny);
            const iz = clampCellIndex(frac.z, nz);
            const cell_index = ix + nx * (iy + ny * iz);
            next[idx] = heads[cell_index];
            heads[cell_index] = @as(i64, @intCast(idx));
        }

        const lists = try alloc.alloc(std.ArrayList(usize), count);
        errdefer {
            for (lists) |*list| {
                list.deinit(alloc);
            }
            alloc.free(lists);
        }
        for (lists) |*list| {
            list.* = .empty;
        }

        var cx: usize = 0;
        while (cx < nx) : (cx += 1) {
            var cy: usize = 0;
            while (cy < ny) : (cy += 1) {
                var cz: usize = 0;
                while (cz < nz) : (cz += 1) {
                    const cell_index = cx + nx * (cy + ny * cz);
                    var i_atom = heads[cell_index];
                    while (i_atom >= 0) : (i_atom = next[@as(usize, @intCast(i_atom))]) {
                        var dx: i32 = -1;
                        while (dx <= 1) : (dx += 1) {
                            var dy: i32 = -1;
                            while (dy <= 1) : (dy += 1) {
                                var dz: i32 = -1;
                                while (dz <= 1) : (dz += 1) {
                                    if (!pbc.x and (cx == 0 and dx < 0)) continue;
                                    if (!pbc.x and (cx + 1 == nx and dx > 0)) continue;
                                    if (!pbc.y and (cy == 0 and dy < 0)) continue;
                                    if (!pbc.y and (cy + 1 == ny and dy > 0)) continue;
                                    if (!pbc.z and (cz == 0 and dz < 0)) continue;
                                    if (!pbc.z and (cz + 1 == nz and dz > 0)) continue;

                                    const nx_i = wrapCellIndex(cx, nx, dx, pbc.x);
                                    const ny_i = wrapCellIndex(cy, ny, dy, pbc.y);
                                    const nz_i = wrapCellIndex(cz, nz, dz, pbc.z);
                                    const neighbor_cell = nx_i + nx * (ny_i + ny * nz_i);
                                    var j_atom = heads[neighbor_cell];
                                    while (j_atom >= 0) : (j_atom = next[@as(usize, @intCast(j_atom))]) {
                                        const i_idx = @as(usize, @intCast(i_atom));
                                        const j_idx = @as(usize, @intCast(j_atom));
                                        if (j_idx <= i_idx) continue;
                                        const delta = math.Vec3.sub(positions[j_idx], positions[i_idx]);
                                        const dvec = minimumImageDelta(cell, inv_cell, pbc, delta);
                                        const dist2 = math.Vec3.dot(dvec, dvec);
                                        if (dist2 <= cutoff * cutoff) {
                                            try lists[i_idx].append(alloc, j_idx);
                                            try lists[j_idx].append(alloc, i_idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        const offsets = try alloc.alloc(usize, count + 1);
        errdefer alloc.free(offsets);
        offsets[0] = 0;
        var total: usize = 0;
        idx = 0;
        while (idx < count) : (idx += 1) {
            total += lists[idx].items.len;
            offsets[idx + 1] = total;
        }

        const neighbors = try alloc.alloc(usize, total);
        errdefer alloc.free(neighbors);
        var cursor: usize = 0;
        idx = 0;
        while (idx < count) : (idx += 1) {
            for (lists[idx].items) |index| {
                neighbors[cursor] = index;
                cursor += 1;
            }
        }

        for (lists) |*list| {
            list.deinit(alloc);
        }
        alloc.free(lists);

        return .{
            .offsets = offsets,
            .neighbors = neighbors,
            .cell = cell,
            .inv_cell = inv_cell,
            .pbc = pbc,
            .cutoff = cutoff,
        };
    }

    pub fn deinit(self: *NeighborList, alloc: std.mem.Allocator) void {
        if (self.offsets.len > 0) alloc.free(self.offsets);
        if (self.neighbors.len > 0) alloc.free(self.neighbors);
        self.* = undefined;
    }

    pub fn neighborsOf(self: *const NeighborList, index: usize) []const usize {
        const start = self.offsets[index];
        const end = self.offsets[index + 1];
        return self.neighbors[start..end];
    }
};

fn invertCell(cell: math.Mat3) !math.Mat3 {
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

fn minimumImageDelta(cell: math.Mat3, inv_cell: math.Mat3, pbc: Pbc, delta: math.Vec3) math.Vec3 {
    var frac = inv_cell.mulVec(delta);
    if (pbc.x) frac.x = wrapFrac(frac.x);
    if (pbc.y) frac.y = wrapFrac(frac.y);
    if (pbc.z) frac.z = wrapFrac(frac.z);
    return cell.mulVec(frac);
}

fn cellLengths(cell: math.Mat3) math.Vec3 {
    return .{
        .x = math.Vec3.norm(cell.row(0)),
        .y = math.Vec3.norm(cell.row(1)),
        .z = math.Vec3.norm(cell.row(2)),
    };
}

fn fracFromCart(cart: math.Vec3, inv_cell: math.Mat3) math.Vec3 {
    return inv_cell.mulVec(cart);
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

test "neighbor list PBC wraps across boundary" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    );
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 9.5, .y = 0.0, .z = 0.0 },
    };
    const pbc = Pbc{ .x = true, .y = true, .z = true };
    var list = try NeighborList.init(alloc, cell, pbc, positions[0..], 1.0);
    defer list.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), list.neighborsOf(0).len);
    try std.testing.expectEqual(@as(usize, 1), list.neighborsOf(1).len);
}

test "neighbor list without PBC ignores across boundary" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    );
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 9.5, .y = 0.0, .z = 0.0 },
    };
    const pbc = Pbc{ .x = false, .y = false, .z = false };
    var list = try NeighborList.init(alloc, cell, pbc, positions[0..], 1.0);
    defer list.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 0), list.neighborsOf(0).len);
    try std.testing.expectEqual(@as(usize, 0), list.neighborsOf(1).len);
}

test "neighbor list simple cubic 2x2x2" {
    const alloc = std.testing.allocator;
    const a = 2.0;
    const cell = math.Mat3.fromRows(
        .{ .x = 2.0 * a, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 2.0 * a, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 2.0 * a },
    );
    var positions: [8]math.Vec3 = undefined;
    var idx: usize = 0;
    var ix: usize = 0;
    while (ix < 2) : (ix += 1) {
        var iy: usize = 0;
        while (iy < 2) : (iy += 1) {
            var iz: usize = 0;
            while (iz < 2) : (iz += 1) {
                positions[idx] = .{
                    .x = a * @as(f64, @floatFromInt(ix)),
                    .y = a * @as(f64, @floatFromInt(iy)),
                    .z = a * @as(f64, @floatFromInt(iz)),
                };
                idx += 1;
            }
        }
    }
    const pbc = Pbc{ .x = true, .y = true, .z = true };
    var list = try NeighborList.init(alloc, cell, pbc, positions[0..], 2.01);
    defer list.deinit(alloc);

    var atom: usize = 0;
    while (atom < positions.len) : (atom += 1) {
        try std.testing.expectEqual(@as(usize, 3), list.neighborsOf(atom).len);
    }
}

test "neighbor list cell list matches naive" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 6.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 6.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 6.0 },
    );
    const positions = [_]math.Vec3{
        .{ .x = 0.5, .y = 0.5, .z = 0.5 },
        .{ .x = 1.4, .y = 0.6, .z = 0.5 },
        .{ .x = 4.9, .y = 0.5, .z = 0.5 },
        .{ .x = 3.0, .y = 3.0, .z = 3.0 },
    };
    const pbc = Pbc{ .x = true, .y = true, .z = true };
    const cutoff = 1.1;

    var naive = try NeighborList.init(alloc, cell, pbc, positions[0..], cutoff);
    defer naive.deinit(alloc);
    var cell_list = try NeighborList.initCellList(alloc, cell, pbc, positions[0..], cutoff);
    defer cell_list.deinit(alloc);

    var i: usize = 0;
    while (i < positions.len) : (i += 1) {
        try std.testing.expectEqual(naive.neighborsOf(i).len, cell_list.neighborsOf(i).len);
    }
}

test "neighbor list diamond conventional PBC" {
    const alloc = std.testing.allocator;
    const a = 10.0;
    const cell = structures.diamondConventionalCell(a);
    const positions = structures.diamondConventionalPositions(a);
    const pbc = Pbc{ .x = true, .y = true, .z = true };
    const cutoff = 0.45 * a;
    var list = try NeighborList.initCellList(alloc, cell, pbc, positions[0..], cutoff);
    defer list.deinit(alloc);
    var idx: usize = 0;
    while (idx < positions.len) : (idx += 1) {
        try std.testing.expectEqual(@as(usize, 4), list.neighborsOf(idx).len);
    }
}
