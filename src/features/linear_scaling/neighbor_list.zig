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
        const inv_cell = try invert_cell(cell);
        const count = positions.len;
        const lists = try init_neighbor_lists(alloc, count);
        defer deinit_neighbor_lists(alloc, lists);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            var j: usize = i + 1;
            while (j < count) : (j += 1) {
                const delta = math.Vec3.sub(positions[j], positions[i]);
                const dvec = minimum_image_delta(cell, inv_cell, pbc, delta);
                const dist2 = math.Vec3.dot(dvec, dvec);
                if (dist2 <= cutoff * cutoff) {
                    try append_neighbor_pair(alloc, lists, i, j);
                }
            }
        }
        return build_neighbor_list(alloc, lists, cell, inv_cell, pbc, cutoff);
    }

    pub fn init_cell_list(
        alloc: std.mem.Allocator,
        cell: math.Mat3,
        pbc: Pbc,
        positions: []const math.Vec3,
        cutoff: f64,
    ) !NeighborList {
        if (cutoff <= 0.0) return error.InvalidCutoff;
        var cell_grid = try init_cell_grid(alloc, cell, pbc, positions, cutoff);
        defer cell_grid.deinit(alloc);

        const lists = try init_neighbor_lists(alloc, positions.len);
        defer deinit_neighbor_lists(alloc, lists);

        try append_cell_list_pairs(alloc, lists, positions, cell, pbc, cutoff, &cell_grid);

        return build_neighbor_list(alloc, lists, cell, cell_grid.inv_cell, pbc, cutoff);
    }

    pub fn deinit(self: *NeighborList, alloc: std.mem.Allocator) void {
        if (self.offsets.len > 0) alloc.free(self.offsets);
        if (self.neighbors.len > 0) alloc.free(self.neighbors);
        self.* = undefined;
    }

    pub fn neighbors_of(self: *const NeighborList, index: usize) []const usize {
        const start = self.offsets[index];
        const end = self.offsets[index + 1];
        return self.neighbors[start..end];
    }
};

const NeighborStorage = struct {
    offsets: []usize,
    neighbors: []usize,
};

const CellGrid = struct {
    heads: []i64,
    next: []i64,
    inv_cell: math.Mat3,
    nx: usize,
    ny: usize,
    nz: usize,

    fn deinit(self: *CellGrid, alloc: std.mem.Allocator) void {
        alloc.free(self.heads);
        alloc.free(self.next);
    }
};

const neighbor_offsets = [_][3]i32{
    .{ -1, -1, -1 },
    .{ -1, -1, 0 },
    .{ -1, -1, 1 },
    .{ -1, 0, -1 },
    .{ -1, 0, 0 },
    .{ -1, 0, 1 },
    .{ -1, 1, -1 },
    .{ -1, 1, 0 },
    .{ -1, 1, 1 },
    .{ 0, -1, -1 },
    .{ 0, -1, 0 },
    .{ 0, -1, 1 },
    .{ 0, 0, -1 },
    .{ 0, 0, 0 },
    .{ 0, 0, 1 },
    .{ 0, 1, -1 },
    .{ 0, 1, 0 },
    .{ 0, 1, 1 },
    .{ 1, -1, -1 },
    .{ 1, -1, 0 },
    .{ 1, -1, 1 },
    .{ 1, 0, -1 },
    .{ 1, 0, 0 },
    .{ 1, 0, 1 },
    .{ 1, 1, -1 },
    .{ 1, 1, 0 },
    .{ 1, 1, 1 },
};

fn init_neighbor_lists(
    alloc: std.mem.Allocator,
    count: usize,
) ![]std.ArrayList(usize) {
    const lists = try alloc.alloc(std.ArrayList(usize), count);
    errdefer alloc.free(lists);
    for (lists) |*list| {
        list.* = .empty;
    }
    return lists;
}

fn deinit_neighbor_lists(
    alloc: std.mem.Allocator,
    lists: []std.ArrayList(usize),
) void {
    for (lists) |*list| {
        list.deinit(alloc);
    }
    alloc.free(lists);
}

fn append_neighbor_pair(
    alloc: std.mem.Allocator,
    lists: []std.ArrayList(usize),
    i: usize,
    j: usize,
) !void {
    try lists[i].append(alloc, j);
    try lists[j].append(alloc, i);
}

fn build_neighbor_storage(
    alloc: std.mem.Allocator,
    lists: []std.ArrayList(usize),
) !NeighborStorage {
    const offsets = try alloc.alloc(usize, lists.len + 1);
    errdefer alloc.free(offsets);
    offsets[0] = 0;

    var total: usize = 0;
    for (lists, 0..) |list, i| {
        total += list.items.len;
        offsets[i + 1] = total;
    }

    const neighbors = try alloc.alloc(usize, total);
    errdefer alloc.free(neighbors);
    var cursor: usize = 0;
    for (lists) |list| {
        for (list.items) |index| {
            neighbors[cursor] = index;
            cursor += 1;
        }
    }
    return .{ .offsets = offsets, .neighbors = neighbors };
}

fn build_neighbor_list(
    alloc: std.mem.Allocator,
    lists: []std.ArrayList(usize),
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: Pbc,
    cutoff: f64,
) !NeighborList {
    const storage = try build_neighbor_storage(alloc, lists);
    return .{
        .offsets = storage.offsets,
        .neighbors = storage.neighbors,
        .cell = cell,
        .inv_cell = inv_cell,
        .pbc = pbc,
        .cutoff = cutoff,
    };
}

fn cell_grid_dims(
    cell: math.Mat3,
    pbc: Pbc,
    cutoff: f64,
) struct { nx: usize, ny: usize, nz: usize } {
    const lengths = cell_lengths(cell);
    var nx = @max(@as(usize, @intFromFloat(std.math.floor(lengths.x / cutoff))), 1);
    var ny = @max(@as(usize, @intFromFloat(std.math.floor(lengths.y / cutoff))), 1);
    var nz = @max(@as(usize, @intFromFloat(std.math.floor(lengths.z / cutoff))), 1);
    if (!pbc.x) nx = @max(nx, 1);
    if (!pbc.y) ny = @max(ny, 1);
    if (!pbc.z) nz = @max(nz, 1);
    return .{ .nx = nx, .ny = ny, .nz = nz };
}

fn init_cell_grid(
    alloc: std.mem.Allocator,
    cell: math.Mat3,
    pbc: Pbc,
    positions: []const math.Vec3,
    cutoff: f64,
) !CellGrid {
    const inv_cell = try invert_cell(cell);
    const dims = cell_grid_dims(cell, pbc, cutoff);
    const cell_count = dims.nx * dims.ny * dims.nz;
    const heads = try alloc.alloc(i64, cell_count);
    errdefer alloc.free(heads);
    @memset(heads, -1);

    const next = try alloc.alloc(i64, positions.len);
    errdefer alloc.free(next);
    @memset(next, -1);

    for (positions, 0..) |position, idx| {
        const frac = frac_from_cart(position, inv_cell);
        const ix = clamp_cell_index(frac.x, dims.nx);
        const iy = clamp_cell_index(frac.y, dims.ny);
        const iz = clamp_cell_index(frac.z, dims.nz);
        const cell_index = ix + dims.nx * (iy + dims.ny * iz);
        next[idx] = heads[cell_index];
        heads[cell_index] = @as(i64, @intCast(idx));
    }

    return .{
        .heads = heads,
        .next = next,
        .inv_cell = inv_cell,
        .nx = dims.nx,
        .ny = dims.ny,
        .nz = dims.nz,
    };
}

fn neighbor_offset_allowed(
    pbc: Pbc,
    cx: usize,
    cy: usize,
    cz: usize,
    grid: *const CellGrid,
    offset: [3]i32,
) bool {
    if (!pbc.x and
        ((cx == 0 and offset[0] < 0) or (cx + 1 == grid.nx and offset[0] > 0))) return false;
    if (!pbc.y and
        ((cy == 0 and offset[1] < 0) or (cy + 1 == grid.ny and offset[1] > 0))) return false;
    if (!pbc.z and
        ((cz == 0 and offset[2] < 0) or (cz + 1 == grid.nz and offset[2] > 0))) return false;
    return true;
}

fn append_pairs_for_cell(
    alloc: std.mem.Allocator,
    lists: []std.ArrayList(usize),
    positions: []const math.Vec3,
    cell: math.Mat3,
    pbc: Pbc,
    cutoff: f64,
    grid: *const CellGrid,
    cx: usize,
    cy: usize,
    cz: usize,
) !void {
    const cell_index = cx + grid.nx * (cy + grid.ny * cz);
    var i_atom = grid.heads[cell_index];
    while (i_atom >= 0) : (i_atom = grid.next[@as(usize, @intCast(i_atom))]) {
        const i_idx = @as(usize, @intCast(i_atom));
        for (neighbor_offsets) |offset| {
            if (!neighbor_offset_allowed(pbc, cx, cy, cz, grid, offset)) continue;

            const nx_i = wrap_cell_index(cx, grid.nx, offset[0], pbc.x);
            const ny_i = wrap_cell_index(cy, grid.ny, offset[1], pbc.y);
            const nz_i = wrap_cell_index(cz, grid.nz, offset[2], pbc.z);
            const neighbor_cell = nx_i + grid.nx * (ny_i + grid.ny * nz_i);
            var j_atom = grid.heads[neighbor_cell];
            while (j_atom >= 0) : (j_atom = grid.next[@as(usize, @intCast(j_atom))]) {
                const j_idx = @as(usize, @intCast(j_atom));
                if (j_idx <= i_idx) continue;

                const delta = math.Vec3.sub(positions[j_idx], positions[i_idx]);
                const dvec = minimum_image_delta(cell, grid.inv_cell, pbc, delta);
                const dist2 = math.Vec3.dot(dvec, dvec);
                if (dist2 <= cutoff * cutoff) {
                    try append_neighbor_pair(alloc, lists, i_idx, j_idx);
                }
            }
        }
    }
}

fn append_cell_list_pairs(
    alloc: std.mem.Allocator,
    lists: []std.ArrayList(usize),
    positions: []const math.Vec3,
    cell: math.Mat3,
    pbc: Pbc,
    cutoff: f64,
    grid: *const CellGrid,
) !void {
    var cx: usize = 0;
    while (cx < grid.nx) : (cx += 1) {
        var cy: usize = 0;
        while (cy < grid.ny) : (cy += 1) {
            var cz: usize = 0;
            while (cz < grid.nz) : (cz += 1) {
                try append_pairs_for_cell(
                    alloc,
                    lists,
                    positions,
                    cell,
                    pbc,
                    cutoff,
                    grid,
                    cx,
                    cy,
                    cz,
                );
            }
        }
    }
}

fn invert_cell(cell: math.Mat3) !math.Mat3 {
    const r0 = cell.row(0);
    const r1 = cell.row(1);
    const r2 = cell.row(2);
    const c0 = math.Vec3.cross(r1, r2);
    const det = math.Vec3.dot(r0, c0);
    if (@abs(det) < 1e-12) return error.SingularCell;
    const inv_det = 1.0 / det;
    const c1 = math.Vec3.cross(r2, r0);
    const c2 = math.Vec3.cross(r0, r1);
    return math.Mat3.from_rows(
        math.Vec3.scale(c0, inv_det),
        math.Vec3.scale(c1, inv_det),
        math.Vec3.scale(c2, inv_det),
    );
}

fn wrap_frac(value: f64) f64 {
    return value - std.math.floor(value + 0.5);
}

fn minimum_image_delta(cell: math.Mat3, inv_cell: math.Mat3, pbc: Pbc, delta: math.Vec3) math.Vec3 {
    var frac = inv_cell.mul_vec(delta);
    if (pbc.x) frac.x = wrap_frac(frac.x);
    if (pbc.y) frac.y = wrap_frac(frac.y);
    if (pbc.z) frac.z = wrap_frac(frac.z);
    return cell.mul_vec(frac);
}

fn cell_lengths(cell: math.Mat3) math.Vec3 {
    return .{
        .x = math.Vec3.norm(cell.row(0)),
        .y = math.Vec3.norm(cell.row(1)),
        .z = math.Vec3.norm(cell.row(2)),
    };
}

fn frac_from_cart(cart: math.Vec3, inv_cell: math.Mat3) math.Vec3 {
    return inv_cell.mul_vec(cart);
}

fn clamp_cell_index(frac: f64, n: usize) usize {
    var t = frac - std.math.floor(frac);
    if (t < 0.0) t += 1.0;
    if (t >= 1.0) t -= 1.0;
    const idx = @as(usize, @intFromFloat(std.math.floor(t * @as(f64, @floatFromInt(n)))));
    return if (idx >= n) n - 1 else idx;
}

fn wrap_cell_index(index: usize, n: usize, delta: i32, enabled: bool) usize {
    if (!enabled) return index;
    const ni = @as(i32, @intCast(n));
    const idx = @mod(@as(i32, @intCast(index)) + delta, ni);
    return @as(usize, @intCast(idx));
}

test "neighbor list PBC wraps across boundary" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.from_rows(
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

    try std.testing.expectEqual(@as(usize, 1), list.neighbors_of(0).len);
    try std.testing.expectEqual(@as(usize, 1), list.neighbors_of(1).len);
}

test "neighbor list without PBC ignores across boundary" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.from_rows(
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

    try std.testing.expectEqual(@as(usize, 0), list.neighbors_of(0).len);
    try std.testing.expectEqual(@as(usize, 0), list.neighbors_of(1).len);
}

test "neighbor list simple cubic 2x2x2" {
    const alloc = std.testing.allocator;
    const a = 2.0;
    const cell = math.Mat3.from_rows(
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
        try std.testing.expectEqual(@as(usize, 3), list.neighbors_of(atom).len);
    }
}

test "neighbor list cell list matches naive" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.from_rows(
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

    var cell_list = try NeighborList.init_cell_list(alloc, cell, pbc, positions[0..], cutoff);
    defer cell_list.deinit(alloc);

    var i: usize = 0;
    while (i < positions.len) : (i += 1) {
        try std.testing.expectEqual(naive.neighbors_of(i).len, cell_list.neighbors_of(i).len);
    }
}

test "neighbor list diamond conventional PBC" {
    const alloc = std.testing.allocator;
    const a = 10.0;
    const cell = structures.diamond_conventional_cell(a);
    const positions = structures.diamond_conventional_positions(a);
    const pbc = Pbc{ .x = true, .y = true, .z = true };
    const cutoff = 0.45 * a;
    var list = try NeighborList.init_cell_list(alloc, cell, pbc, positions[0..], cutoff);
    defer list.deinit(alloc);

    var idx: usize = 0;
    while (idx < positions.len) : (idx += 1) {
        try std.testing.expectEqual(@as(usize, 4), list.neighbors_of(idx).len);
    }
}
