const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const mesh = @import("mesh.zig");

pub const Index3 = struct {
    x: i32,
    y: i32,
    z: i32,
};

pub const KmeshOp = struct {
    op: symmetry.SymOp,
    delta: Index3,
};

pub fn filterSymOpsForKmesh(
    alloc: std.mem.Allocator,
    ops: []const symmetry.SymOp,
    kmesh: [3]usize,
    shift: math.Vec3,
    tol: f64,
) ![]KmeshOp {
    const shift_grid = shiftInGridUnits(kmesh, shift);
    var list: std.ArrayList(KmeshOp) = .empty;
    errdefer list.deinit(alloc);

    for (ops) |op| {
        if (deltaForOp(op, shift_grid, tol)) |delta| {
            try list.append(alloc, .{ .op = op, .delta = delta });
        }
    }
    return try list.toOwnedSlice(alloc);
}

pub fn reduceKmesh(
    alloc: std.mem.Allocator,
    kmesh: [3]usize,
    shift: math.Vec3,
    ops: []const KmeshOp,
    recip: math.Mat3,
    time_reversal: bool,
) ![]symmetry.KPoint {
    const total = kmesh[0] * kmesh[1] * kmesh[2];
    if (total == 0) return error.InvalidKmesh;

    const shift_grid = shiftInGridUnits(kmesh, shift);
    const shift2 = math.Vec3.scale(shift_grid, 2.0);
    const shift2_int_opt = intVector(shift2, 1e-8);
    const shift2_int = shift2_int_opt orelse Index3{ .x = 0, .y = 0, .z = 0 };
    const can_time_reverse = time_reversal and shift2_int_opt != null;

    const used = try alloc.alloc(bool, total);
    defer alloc.free(used);
    @memset(used, false);

    var reduced: std.ArrayList(symmetry.KPoint) = .empty;
    errdefer reduced.deinit(alloc);

    var i: usize = 0;
    while (i < total) : (i += 1) {
        if (used[i]) continue;
        const seed = indexFromFlat(kmesh, i);
        const group = try alloc.alloc(bool, total);
        defer alloc.free(group);
        @memset(group, false);

        markIndex(group, kmesh, seed);
        for (ops) |op| {
            const mapped = mapIndex(op, seed, kmesh);
            markIndex(group, kmesh, mapped);
            if (can_time_reverse) {
                const tr = timeReversalIndex(mapped, kmesh, shift2_int);
                markIndex(group, kmesh, tr);
            }
        }

        var count: usize = 0;
        var j: usize = 0;
        while (j < total) : (j += 1) {
            if (!group[j]) continue;
            used[j] = true;
            count += 1;
        }

        const weight = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(total));
        const k_frac = math.Vec3{
            .x = mesh.fracFromIndex(seed.x, kmesh[0], shift.x),
            .y = mesh.fracFromIndex(seed.y, kmesh[1], shift.y),
            .z = mesh.fracFromIndex(seed.z, kmesh[2], shift.z),
        };
        reduced.append(alloc, .{
            .k_frac = k_frac,
            .k_cart = math.fracToCart(k_frac, recip),
            .weight = weight,
        }) catch return error.OutOfMemory;
    }

    for (used) |ok| {
        if (!ok) return error.InvalidKmeshReduction;
    }

    return try reduced.toOwnedSlice(alloc);
}

/// IBZ→full BZ mapping: which IBZ k-point and symmetry operation
/// produces each full-BZ k-point.
pub const KmeshMapping = struct {
    ibz_kpoints: []symmetry.KPoint, // IBZ k-points (with weights)
    full_to_ibz: []usize, // full_to_ibz[i_full] = IBZ representative index
    full_symop: []usize, // full_symop[i_full] = symop index: S*k_ibz = k_full
    full_time_reversed: []bool, // whether time reversal was applied
    n_full: usize,
    kmesh: [3]usize,
    shift: math.Vec3,

    pub fn deinit(self: *KmeshMapping, alloc: std.mem.Allocator) void {
        alloc.free(self.ibz_kpoints);
        alloc.free(self.full_to_ibz);
        alloc.free(self.full_symop);
        alloc.free(self.full_time_reversed);
    }
};

/// Like `reduceKmesh`, but also returns the mapping from each full-BZ k-point
/// to its IBZ representative and the symmetry operation that connects them.
pub fn reduceKmeshWithMapping(
    alloc: std.mem.Allocator,
    kmesh: [3]usize,
    shift: math.Vec3,
    ops: []const KmeshOp,
    recip: math.Mat3,
    time_reversal: bool,
) !KmeshMapping {
    const total = kmesh[0] * kmesh[1] * kmesh[2];
    if (total == 0) return error.InvalidKmesh;

    const shift_grid = shiftInGridUnits(kmesh, shift);
    const shift2 = math.Vec3.scale(shift_grid, 2.0);
    const shift2_int_opt = intVector(shift2, 1e-8);
    const shift2_int = shift2_int_opt orelse Index3{ .x = 0, .y = 0, .z = 0 };
    const can_time_reverse = time_reversal and shift2_int_opt != null;

    const used = try alloc.alloc(bool, total);
    defer alloc.free(used);
    @memset(used, false);

    // Per full-BZ k-point: which IBZ index and symop index
    const full_to_ibz = try alloc.alloc(usize, total);
    errdefer alloc.free(full_to_ibz);
    const full_symop = try alloc.alloc(usize, total);
    errdefer alloc.free(full_symop);
    const full_time_reversed = try alloc.alloc(bool, total);
    errdefer alloc.free(full_time_reversed);
    @memset(full_to_ibz, 0);
    @memset(full_symop, 0);
    @memset(full_time_reversed, false);

    var reduced: std.ArrayList(symmetry.KPoint) = .empty;
    errdefer reduced.deinit(alloc);

    var i: usize = 0;
    while (i < total) : (i += 1) {
        if (used[i]) continue;
        const seed = indexFromFlat(kmesh, i);
        const ibz_idx = reduced.items.len;

        const seed_flat = flatIndex(kmesh, seed);

        // Find identity operation index first.
        // The IBZ representative must use identity to preserve LOBPCG eigenvectors.
        var identity_idx: ?usize = null;
        for (ops, 0..) |op, op_idx| {
            const mapped = mapIndex(op, seed, kmesh);
            const mapped_flat = flatIndex(kmesh, mapped);
            if (mapped_flat == seed_flat) {
                // Check if this is the identity (rot = I, no translation needed for grid)
                if (isIdentityOp(op.op)) {
                    identity_idx = op_idx;
                    break;
                }
            }
        }

        // Mark seed with identity (must exist)
        if (identity_idx) |id_idx| {
            used[seed_flat] = true;
            full_to_ibz[seed_flat] = ibz_idx;
            full_symop[seed_flat] = id_idx;
            full_time_reversed[seed_flat] = false;
        } else {
            return error.InvalidKmeshReduction;
        }

        // Mark other k-points in the star
        for (ops, 0..) |op, op_idx| {
            const mapped = mapIndex(op, seed, kmesh);
            const mapped_flat = flatIndex(kmesh, mapped);
            if (!used[mapped_flat]) {
                used[mapped_flat] = true;
                full_to_ibz[mapped_flat] = ibz_idx;
                full_symop[mapped_flat] = op_idx;
                full_time_reversed[mapped_flat] = false;
            }
            if (can_time_reverse) {
                const tr = timeReversalIndex(mapped, kmesh, shift2_int);
                const tr_flat = flatIndex(kmesh, tr);
                if (!used[tr_flat]) {
                    used[tr_flat] = true;
                    full_to_ibz[tr_flat] = ibz_idx;
                    full_symop[tr_flat] = op_idx;
                    full_time_reversed[tr_flat] = true;
                }
            }
        }

        // Count star size for weight
        var count: usize = 0;
        var j: usize = 0;
        while (j < total) : (j += 1) {
            if (full_to_ibz[j] == ibz_idx and used[j]) {
                count += 1;
            }
        }

        const weight = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(total));
        const k_frac = math.Vec3{
            .x = mesh.fracFromIndex(seed.x, kmesh[0], shift.x),
            .y = mesh.fracFromIndex(seed.y, kmesh[1], shift.y),
            .z = mesh.fracFromIndex(seed.z, kmesh[2], shift.z),
        };
        reduced.append(alloc, .{
            .k_frac = k_frac,
            .k_cart = math.fracToCart(k_frac, recip),
            .weight = weight,
        }) catch return error.OutOfMemory;
    }

    for (used) |ok| {
        if (!ok) return error.InvalidKmeshReduction;
    }

    return KmeshMapping{
        .ibz_kpoints = try reduced.toOwnedSlice(alloc),
        .full_to_ibz = full_to_ibz,
        .full_symop = full_symop,
        .full_time_reversed = full_time_reversed,
        .n_full = total,
        .kmesh = kmesh,
        .shift = shift,
    };
}

pub fn verifyKmeshReduction(
    alloc: std.mem.Allocator,
    full: []const symmetry.KPoint,
    reduced: []const symmetry.KPoint,
    kmesh: [3]usize,
    shift: math.Vec3,
    ops: []const KmeshOp,
    time_reversal: bool,
    tol: f64,
) !bool {
    if (reduced.len == full.len) return true;
    if (full.len == 0) return reduced.len == 0;

    const total = full.len;
    const covered = try alloc.alloc(bool, total);
    defer alloc.free(covered);
    @memset(covered, false);

    const shift_grid = shiftInGridUnits(kmesh, shift);
    const shift2 = math.Vec3.scale(shift_grid, 2.0);
    const shift2_int_opt = intVector(shift2, tol);
    const shift2_int = shift2_int_opt orelse Index3{ .x = 0, .y = 0, .z = 0 };
    const can_time_reverse = time_reversal and shift2_int_opt != null;

    const weight_tol = 1e-8;
    for (reduced) |kp| {
        const seed = indexFromKpoint(kp, kmesh, shift, tol) orelse return false;
        const group = try alloc.alloc(bool, total);
        defer alloc.free(group);
        @memset(group, false);

        markIndex(group, kmesh, seed);
        for (ops) |op| {
            const mapped = mapIndex(op, seed, kmesh);
            markIndex(group, kmesh, mapped);
            if (can_time_reverse) {
                const tr = timeReversalIndex(mapped, kmesh, shift2_int);
                markIndex(group, kmesh, tr);
            }
        }

        var count: usize = 0;
        var j: usize = 0;
        while (j < total) : (j += 1) {
            if (!group[j]) continue;
            covered[j] = true;
            count += 1;
        }

        const weight = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(total));
        if (@abs(weight - kp.weight) > weight_tol) return false;
    }

    for (covered) |ok| {
        if (!ok) return false;
    }
    return true;
}

fn shiftInGridUnits(kmesh: [3]usize, shift: math.Vec3) math.Vec3 {
    return .{
        .x = shift.x * @as(f64, @floatFromInt(kmesh[0])),
        .y = shift.y * @as(f64, @floatFromInt(kmesh[1])),
        .z = shift.z * @as(f64, @floatFromInt(kmesh[2])),
    };
}

fn deltaForOp(op: symmetry.SymOp, shift_grid: math.Vec3, tol: f64) ?Index3 {
    const mapped = op.k_rot.mulVec(shift_grid);
    const delta = math.Vec3.sub(mapped, shift_grid);
    return intVector(delta, tol);
}

fn intVector(value: math.Vec3, tol: f64) ?Index3 {
    if (!closeToInt(value.x, tol)) return null;
    if (!closeToInt(value.y, tol)) return null;
    if (!closeToInt(value.z, tol)) return null;
    return .{
        .x = @as(i32, @intFromFloat(std.math.round(value.x))),
        .y = @as(i32, @intFromFloat(std.math.round(value.y))),
        .z = @as(i32, @intFromFloat(std.math.round(value.z))),
    };
}

fn closeToInt(value: f64, tol: f64) bool {
    return @abs(value - std.math.round(value)) < tol;
}

fn mapIndex(op: KmeshOp, index: Index3, kmesh: [3]usize) Index3 {
    const mapped = mulMatVec(op.op.k_rot, index);
    return wrapIndex(.{
        .x = mapped.x + op.delta.x,
        .y = mapped.y + op.delta.y,
        .z = mapped.z + op.delta.z,
    }, kmesh);
}

fn timeReversalIndex(index: Index3, kmesh: [3]usize, shift2: Index3) Index3 {
    return wrapIndex(.{
        .x = -index.x - shift2.x,
        .y = -index.y - shift2.y,
        .z = -index.z - shift2.z,
    }, kmesh);
}

fn wrapIndex(index: Index3, kmesh: [3]usize) Index3 {
    return .{
        .x = modIndex(index.x, kmesh[0]),
        .y = modIndex(index.y, kmesh[1]),
        .z = modIndex(index.z, kmesh[2]),
    };
}

fn modIndex(value: i32, n: usize) i32 {
    const ni = @as(i32, @intCast(n));
    return @as(i32, @intCast(@mod(value, ni)));
}

fn mulMatVec(mat: symmetry.Mat3i, vec: Index3) Index3 {
    return .{
        .x = mat.m[0][0] * vec.x + mat.m[0][1] * vec.y + mat.m[0][2] * vec.z,
        .y = mat.m[1][0] * vec.x + mat.m[1][1] * vec.y + mat.m[1][2] * vec.z,
        .z = mat.m[2][0] * vec.x + mat.m[2][1] * vec.y + mat.m[2][2] * vec.z,
    };
}

fn markIndex(group: []bool, kmesh: [3]usize, index: Index3) void {
    const flat = flatIndex(kmesh, index);
    group[flat] = true;
}

fn flatIndex(kmesh: [3]usize, index: Index3) usize {
    return @as(usize, @intCast(index.x)) + kmesh[0] * (@as(usize, @intCast(index.y)) + kmesh[1] * @as(usize, @intCast(index.z)));
}

fn indexFromFlat(kmesh: [3]usize, flat: usize) Index3 {
    const nx = kmesh[0];
    const ny = kmesh[1];
    const ix = flat % nx;
    const iy = (flat / nx) % ny;
    const iz = flat / (nx * ny);
    return .{ .x = @intCast(ix), .y = @intCast(iy), .z = @intCast(iz) };
}

fn indexFromKpoint(
    kp: symmetry.KPoint,
    kmesh: [3]usize,
    shift: math.Vec3,
    tol: f64,
) ?Index3 {
    const ix = matchIndex(kp.k_frac.x, kmesh[0], shift.x, tol) orelse return null;
    const iy = matchIndex(kp.k_frac.y, kmesh[1], shift.y, tol) orelse return null;
    const iz = matchIndex(kp.k_frac.z, kmesh[2], shift.z, tol) orelse return null;
    return .{ .x = ix, .y = iy, .z = iz };
}

fn matchIndex(value: f64, n: usize, shift: f64, tol: f64) ?i32 {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const frac = mesh.fracFromIndex(@as(i32, @intCast(i)), n, shift);
        if (fracClose(frac, value, tol)) return @intCast(i);
    }
    return null;
}

fn fracClose(a: f64, b: f64, tol: f64) bool {
    const d = a - b;
    const centered = d - std.math.round(d);
    return @abs(centered) < tol;
}

fn isIdentityOp(op: symmetry.SymOp) bool {
    return op.rot.m[0][0] == 1 and op.rot.m[0][1] == 0 and op.rot.m[0][2] == 0 and
        op.rot.m[1][0] == 0 and op.rot.m[1][1] == 1 and op.rot.m[1][2] == 0 and
        op.rot.m[2][0] == 0 and op.rot.m[2][1] == 0 and op.rot.m[2][2] == 1;
}
