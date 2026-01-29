const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("symmetry.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const data = @import("spacegroup_data.zig");

pub const SpaceGroupInfo = struct {
    number: i32,
    hall_number: i32,
    hall_symbol: []u8,
    international: []u8,
    international_short: []u8,
    schoenflies: []u8,
    choice: []u8,

    pub fn deinit(self: *SpaceGroupInfo, alloc: std.mem.Allocator) void {
        if (self.hall_symbol.len > 0) alloc.free(self.hall_symbol);
        if (self.international.len > 0) alloc.free(self.international);
        if (self.international_short.len > 0) alloc.free(self.international_short);
        if (self.schoenflies.len > 0) alloc.free(self.schoenflies);
        if (self.choice.len > 0) alloc.free(self.choice);
    }
};

/// Detect space group and return its number and symbols.
pub fn detectSpaceGroup(
    alloc: std.mem.Allocator,
    cell: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    symprec: f64,
) !?SpaceGroupInfo {
    const ops = try symmetry.getSymmetryOps(alloc, cell, atoms, symprec);
    defer alloc.free(ops);
    const hall_number = try matchHallNumber(alloc, ops, symprec);
    if (hall_number == 0) return null;

    const spg = data.spacegroup_types[@as(usize, @intCast(hall_number))];
    return SpaceGroupInfo{
        .number = spg.number,
        .hall_number = hall_number,
        .hall_symbol = try copyHallSymbol(alloc, spg.hall_symbol),
        .international = try copyTrimmed(alloc, spg.international),
        .international_short = try copyTrimmed(alloc, spg.international_short),
        .schoenflies = try copyTrimmed(alloc, spg.schoenflies),
        .choice = try copyTrimmed(alloc, spg.choice),
    };
}

/// Detect space group using atoms only.
pub fn detectSpaceGroupFromAtoms(
    alloc: std.mem.Allocator,
    cell: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    symprec: f64,
) !?SpaceGroupInfo {
    return detectSpaceGroup(alloc, cell, atoms, symprec);
}

fn matchHallNumber(alloc: std.mem.Allocator, ops_in: []const symmetry.SymOp, tol: f64) !i32 {
    const bases = try orthogonalBasisTransforms(alloc);
    defer alloc.free(bases);
    const transformed = try alloc.alloc(symmetry.SymOp, ops_in.len);
    defer alloc.free(transformed);

    var hall: usize = 1;
    while (hall < data.spacegroup_types.len) : (hall += 1) {
        const db_ops = try getDatabaseOps(alloc, hall);
        defer alloc.free(db_ops);
        if (db_ops.len != ops_in.len) continue;
        var matched = false;
        for (bases) |basis| {
            transformOps(transformed, ops_in, basis);
            if (matchOperations(alloc, transformed, db_ops, tol)) {
                matched = true;
                break;
            }
        }
        if (matched) {
            return @intCast(hall);
        }
    }
    return 0;
}

fn matchOperations(
    alloc: std.mem.Allocator,
    ops_in: []const symmetry.SymOp,
    ops_db: []const symmetry.SymOp,
    tol: f64,
) bool {
    _ = tol;
    for (ops_db) |op| {
        if (!hasRotation(ops_in, op.rot)) return false;
    }

    const grid: i32 = 24;
    var set = std.AutoHashMap(u64, void).init(alloc);
    defer set.deinit();
    for (ops_in) |op| {
        const key = encodeOp(op.rot, op.trans, grid);
        set.put(key, {}) catch return false;
    }

    const inv_grid = 1.0 / @as(f64, @floatFromInt(grid));
    var ox: i32 = 0;
    while (ox < grid) : (ox += 1) {
        var oy: i32 = 0;
        while (oy < grid) : (oy += 1) {
            var oz: i32 = 0;
            while (oz < grid) : (oz += 1) {
                const origin = math.Vec3{
                    .x = @as(f64, @floatFromInt(ox)) * inv_grid,
                    .y = @as(f64, @floatFromInt(oy)) * inv_grid,
                    .z = @as(f64, @floatFromInt(oz)) * inv_grid,
                };
                var matched = true;
                for (ops_db) |op_db| {
                    const shift = math.Vec3.sub(origin, op_db.rot.mulVec(origin));
                    const trans = wrap01(math.Vec3.add(op_db.trans, shift));
                    const key = encodeOp(op_db.rot, trans, grid);
                    if (!set.contains(key)) {
                        matched = false;
                        break;
                    }
                }
                if (matched) return true;
            }
        }
    }
    return false;
}

fn hasRotation(ops: []const symmetry.SymOp, rot: symmetry.Mat3i) bool {
    for (ops) |op| {
        if (mat3iEqual(op.rot, rot)) return true;
    }
    return false;
}

fn orthogonalBasisTransforms(alloc: std.mem.Allocator) ![]symmetry.Mat3i {
    const perms = [_][3]u8{
        .{ 0, 1, 2 },
        .{ 0, 2, 1 },
        .{ 1, 0, 2 },
        .{ 1, 2, 0 },
        .{ 2, 0, 1 },
        .{ 2, 1, 0 },
    };
    var list: std.ArrayList(symmetry.Mat3i) = .empty;
    errdefer list.deinit(alloc);
    for (perms) |perm| {
        var sx: i32 = -1;
        while (sx <= 1) : (sx += 2) {
            var sy: i32 = -1;
            while (sy <= 1) : (sy += 2) {
                var sz: i32 = -1;
                while (sz <= 1) : (sz += 2) {
                    var m = [3][3]i32{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
                    m[0][perm[0]] = sx;
                    m[1][perm[1]] = sy;
                    m[2][perm[2]] = sz;
                    try list.append(alloc, symmetry.Mat3i{ .m = m });
                }
            }
        }
    }
    return try list.toOwnedSlice(alloc);
}

fn transformOps(out_ops: []symmetry.SymOp, ops: []const symmetry.SymOp, basis: symmetry.Mat3i) void {
    const inv = basis.inverse() orelse basis;
    var i: usize = 0;
    while (i < ops.len) : (i += 1) {
        const rot = mat3iMul(mat3iMul(inv, ops[i].rot), basis);
        const trans = wrap01(inv.mulVec(ops[i].trans));
        const inv_rot = rot.inverse() orelse rot;
        const k_rot = inv_rot.transpose();
        out_ops[i] = .{ .rot = rot, .k_rot = k_rot, .trans = trans };
    }
}

fn mat3iMul(a: symmetry.Mat3i, b: symmetry.Mat3i) symmetry.Mat3i {
    var out = [3][3]i32{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 3) : (j += 1) {
            var sum: i32 = 0;
            var k: usize = 0;
            while (k < 3) : (k += 1) {
                sum += a.m[i][k] * b.m[k][j];
            }
            out[i][j] = sum;
        }
    }
    return .{ .m = out };
}

fn mat3iEqual(a: symmetry.Mat3i, b: symmetry.Mat3i) bool {
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 3) : (j += 1) {
            if (a.m[i][j] != b.m[i][j]) return false;
        }
    }
    return true;
}

fn encodeOp(rot: symmetry.Mat3i, trans: math.Vec3, grid: i32) u64 {
    const r = encodeRotation(rot);
    const t = encodeTranslation(trans, grid);
    return (@as(u64, r) << 32) | @as(u64, t);
}

fn encodeRotation(rot: symmetry.Mat3i) u32 {
    var code: u32 = 0;
    var factor: u32 = 1;
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 3) : (j += 1) {
            const digit = @as(u32, @intCast(rot.m[i][j] + 1));
            code += digit * factor;
            factor *= 3;
        }
    }
    return code;
}

fn encodeTranslation(trans: math.Vec3, grid: i32) u32 {
    const t = wrap01(trans);
    const qx = quantizeCoord(t.x, grid);
    const qy = quantizeCoord(t.y, grid);
    const qz = quantizeCoord(t.z, grid);
    const grid_u = @as(u32, @intCast(grid));
    return @as(u32, @intCast(qx)) + grid_u * (@as(u32, @intCast(qy)) + grid_u * @as(u32, @intCast(qz)));
}

fn quantizeCoord(x: f64, grid: i32) i32 {
    const scaled = x * @as(f64, @floatFromInt(grid));
    var q = @as(i32, @intFromFloat(std.math.round(scaled)));
    if (q == grid) q = 0;
    if (q < 0) q += grid;
    return q;
}

fn getDatabaseOps(alloc: std.mem.Allocator, hall_number: usize) ![]symmetry.SymOp {
    const idx = data.symmetry_operation_index[hall_number];
    const count = idx[0];
    const start = idx[1];
    const list = try alloc.alloc(symmetry.SymOp, @as(usize, @intCast(count)));
    var i: usize = 0;
    while (i < @as(usize, @intCast(count))) : (i += 1) {
        const encoded = data.symmetry_operations[@as(usize, @intCast(start)) + i];
        const op = decodeOperation(encoded);
        list[i] = op;
    }
    return list;
}

fn decodeOperation(encoded: i32) symmetry.SymOp {
    var rot: [3][3]i32 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    var trans = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const r = @mod(encoded, 19683);
    var digit: i32 = 6561;
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 3) : (j += 1) {
            rot[i][j] = @intCast(@divTrunc(@mod(r, digit * 3), digit) - 1);
            digit = @divTrunc(digit, 3);
        }
    }

    const t = @divTrunc(encoded, 19683);
    digit = 144;
    var k: usize = 0;
    while (k < 3) : (k += 1) {
        const val = @divTrunc(@mod(t, digit * 12), digit);
        const fval = @as(f64, @floatFromInt(val)) / 12.0;
        switch (k) {
            0 => trans.x = fval,
            1 => trans.y = fval,
            else => trans.z = fval,
        }
        digit = @divTrunc(digit, 12);
    }

    const rot_mat = symmetry.Mat3i{ .m = rot };
    const inv = rot_mat.inverse() orelse rot_mat;
    const k_rot = inv.transpose();
    return .{ .rot = rot_mat, .k_rot = k_rot, .trans = trans };
}

fn wrap01(v: math.Vec3) math.Vec3 {
    return .{
        .x = v.x - std.math.floor(v.x),
        .y = v.y - std.math.floor(v.y),
        .z = v.z - std.math.floor(v.z),
    };
}

fn wrapCentered(v: math.Vec3) math.Vec3 {
    return .{
        .x = v.x - std.math.round(v.x),
        .y = v.y - std.math.round(v.y),
        .z = v.z - std.math.round(v.z),
    };
}

fn fracClose(a: math.Vec3, b: math.Vec3, tol: f64) bool {
    const d = wrapCentered(math.Vec3.sub(a, b));
    return @abs(d.x) < tol and @abs(d.y) < tol and @abs(d.z) < tol;
}

fn copyTrimmed(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    const trimmed = std.mem.trimRight(u8, input, " ");
    const out = try alloc.alloc(u8, trimmed.len);
    @memcpy(out, trimmed);
    return out;
}

fn copyHallSymbol(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    const trimmed = std.mem.trimRight(u8, input, " ");
    const out = try alloc.alloc(u8, trimmed.len);
    @memcpy(out, trimmed);
    for (out) |*c| {
        if (c.* == '=') c.* = '"';
    }
    return out;
}
