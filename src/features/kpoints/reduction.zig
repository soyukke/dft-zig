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

pub fn filter_sym_ops_for_kmesh(
    alloc: std.mem.Allocator,
    ops: []const symmetry.SymOp,
    kmesh: [3]usize,
    shift: math.Vec3,
    tol: f64,
) ![]KmeshOp {
    const shift_grid = shift_in_grid_units(kmesh, shift);
    var list: std.ArrayList(KmeshOp) = .empty;
    errdefer list.deinit(alloc);

    for (ops) |op| {
        if (delta_for_op(op, shift_grid, tol)) |delta| {
            try list.append(alloc, .{ .op = op, .delta = delta });
        }
    }
    return try list.toOwnedSlice(alloc);
}

const TimeReversalSettings = struct {
    shift2_int: Index3,
    can_time_reverse: bool,
};

fn init_time_reversal_settings(
    kmesh: [3]usize,
    shift: math.Vec3,
    tol: f64,
    time_reversal: bool,
) TimeReversalSettings {
    const shift_grid = shift_in_grid_units(kmesh, shift);
    const shift2_int_opt = int_vector(math.Vec3.scale(shift_grid, 2.0), tol);
    return .{
        .shift2_int = shift2_int_opt orelse .{ .x = 0, .y = 0, .z = 0 },
        .can_time_reverse = time_reversal and shift2_int_opt != null,
    };
}

pub fn reduce_kmesh(
    alloc: std.mem.Allocator,
    kmesh: [3]usize,
    shift: math.Vec3,
    ops: []const KmeshOp,
    recip: math.Mat3,
    time_reversal: bool,
) ![]symmetry.KPoint {
    const total = kmesh[0] * kmesh[1] * kmesh[2];
    if (total == 0) return error.InvalidKmesh;

    const shift_grid = shift_in_grid_units(kmesh, shift);
    const shift2 = math.Vec3.scale(shift_grid, 2.0);
    const shift2_int_opt = int_vector(shift2, 1e-8);
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
        const seed = index_from_flat(kmesh, i);
        const group = try alloc.alloc(bool, total);
        defer alloc.free(group);

        @memset(group, false);

        mark_index(group, kmesh, seed);
        for (ops) |op| {
            const mapped = map_index(op, seed, kmesh);
            mark_index(group, kmesh, mapped);
            if (can_time_reverse) {
                const tr = time_reversal_index(mapped, kmesh, shift2_int);
                mark_index(group, kmesh, tr);
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
            .x = mesh.frac_from_index(seed.x, kmesh[0], shift.x),
            .y = mesh.frac_from_index(seed.y, kmesh[1], shift.y),
            .z = mesh.frac_from_index(seed.z, kmesh[2], shift.z),
        };
        reduced.append(alloc, .{
            .k_frac = k_frac,
            .k_cart = math.frac_to_cart(k_frac, recip),
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

/// Find the identity symmetry-op index (rot = I) that maps `seed` to itself.
/// The IBZ representative must use identity to preserve LOBPCG eigenvectors.
fn find_identity_op_index(
    ops: []const KmeshOp,
    seed: Index3,
    seed_flat: usize,
    kmesh: [3]usize,
) ?usize {
    for (ops, 0..) |op, op_idx| {
        const mapped = map_index(op, seed, kmesh);
        const mapped_flat = flat_index(kmesh, mapped);
        if (mapped_flat == seed_flat) {
            if (is_identity_op(op.op)) return op_idx;
        }
    }
    return null;
}

/// Mark every k-point in the star of `seed` (optionally including time-reversed
/// partners) as belonging to `ibz_idx`.
fn mark_star_from_seed(
    seed: Index3,
    kmesh: [3]usize,
    ops: []const KmeshOp,
    can_time_reverse: bool,
    shift2_int: Index3,
    ibz_idx: usize,
    used: []bool,
    full_to_ibz: []usize,
    full_symop: []usize,
    full_time_reversed: []bool,
) void {
    for (ops, 0..) |op, op_idx| {
        const mapped = map_index(op, seed, kmesh);
        const mapped_flat = flat_index(kmesh, mapped);
        if (!used[mapped_flat]) {
            used[mapped_flat] = true;
            full_to_ibz[mapped_flat] = ibz_idx;
            full_symop[mapped_flat] = op_idx;
            full_time_reversed[mapped_flat] = false;
        }
        if (can_time_reverse) {
            const tr = time_reversal_index(mapped, kmesh, shift2_int);
            const tr_flat = flat_index(kmesh, tr);
            if (!used[tr_flat]) {
                used[tr_flat] = true;
                full_to_ibz[tr_flat] = ibz_idx;
                full_symop[tr_flat] = op_idx;
                full_time_reversed[tr_flat] = true;
            }
        }
    }
}

/// Process one still-unvisited seed k-point: find identity op, expand its star,
/// accumulate weight, and push the IBZ representative into `reduced`.
fn process_seed_kpoint(
    alloc: std.mem.Allocator,
    seed_flat_start: usize,
    kmesh: [3]usize,
    shift: math.Vec3,
    ops: []const KmeshOp,
    recip: math.Mat3,
    can_time_reverse: bool,
    shift2_int: Index3,
    total: usize,
    used: []bool,
    full_to_ibz: []usize,
    full_symop: []usize,
    full_time_reversed: []bool,
    reduced: *std.ArrayList(symmetry.KPoint),
) !void {
    const seed = index_from_flat(kmesh, seed_flat_start);
    const ibz_idx = reduced.items.len;
    const seed_flat = flat_index(kmesh, seed);

    const identity_idx = find_identity_op_index(ops, seed, seed_flat, kmesh) orelse
        return error.InvalidKmeshReduction;

    // Mark seed with identity (must exist)
    used[seed_flat] = true;
    full_to_ibz[seed_flat] = ibz_idx;
    full_symop[seed_flat] = identity_idx;
    full_time_reversed[seed_flat] = false;

    mark_star_from_seed(
        seed,
        kmesh,
        ops,
        can_time_reverse,
        shift2_int,
        ibz_idx,
        used,
        full_to_ibz,
        full_symop,
        full_time_reversed,
    );

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
        .x = mesh.frac_from_index(seed.x, kmesh[0], shift.x),
        .y = mesh.frac_from_index(seed.y, kmesh[1], shift.y),
        .z = mesh.frac_from_index(seed.z, kmesh[2], shift.z),
    };
    reduced.append(alloc, .{
        .k_frac = k_frac,
        .k_cart = math.frac_to_cart(k_frac, recip),
        .weight = weight,
    }) catch return error.OutOfMemory;
}

/// Like `reduce_kmesh`, but also returns the mapping from each full-BZ k-point
/// to its IBZ representative and the symmetry operation that connects them.
pub fn reduce_kmesh_with_mapping(
    alloc: std.mem.Allocator,
    kmesh: [3]usize,
    shift: math.Vec3,
    ops: []const KmeshOp,
    recip: math.Mat3,
    time_reversal: bool,
) !KmeshMapping {
    const total = kmesh[0] * kmesh[1] * kmesh[2];
    if (total == 0) return error.InvalidKmesh;

    const tr_settings = init_time_reversal_settings(kmesh, shift, 1e-8, time_reversal);

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
        try process_seed_kpoint(
            alloc,
            i,
            kmesh,
            shift,
            ops,
            recip,
            tr_settings.can_time_reverse,
            tr_settings.shift2_int,
            total,
            used,
            full_to_ibz,
            full_symop,
            full_time_reversed,
            &reduced,
        );
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

pub fn verify_kmesh_reduction(
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

    const shift_grid = shift_in_grid_units(kmesh, shift);
    const shift2 = math.Vec3.scale(shift_grid, 2.0);
    const shift2_int_opt = int_vector(shift2, tol);
    const shift2_int = shift2_int_opt orelse Index3{ .x = 0, .y = 0, .z = 0 };
    const can_time_reverse = time_reversal and shift2_int_opt != null;

    const weight_tol = 1e-8;
    for (reduced) |kp| {
        const seed = index_from_kpoint(kp, kmesh, shift, tol) orelse return false;
        const group = try alloc.alloc(bool, total);
        defer alloc.free(group);

        @memset(group, false);

        mark_index(group, kmesh, seed);
        for (ops) |op| {
            const mapped = map_index(op, seed, kmesh);
            mark_index(group, kmesh, mapped);
            if (can_time_reverse) {
                const tr = time_reversal_index(mapped, kmesh, shift2_int);
                mark_index(group, kmesh, tr);
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

fn shift_in_grid_units(kmesh: [3]usize, shift: math.Vec3) math.Vec3 {
    return .{
        .x = shift.x * @as(f64, @floatFromInt(kmesh[0])),
        .y = shift.y * @as(f64, @floatFromInt(kmesh[1])),
        .z = shift.z * @as(f64, @floatFromInt(kmesh[2])),
    };
}

fn delta_for_op(op: symmetry.SymOp, shift_grid: math.Vec3, tol: f64) ?Index3 {
    const mapped = op.k_rot.mul_vec(shift_grid);
    const delta = math.Vec3.sub(mapped, shift_grid);
    return int_vector(delta, tol);
}

fn int_vector(value: math.Vec3, tol: f64) ?Index3 {
    if (!close_to_int(value.x, tol)) return null;
    if (!close_to_int(value.y, tol)) return null;
    if (!close_to_int(value.z, tol)) return null;
    return .{
        .x = @as(i32, @intFromFloat(std.math.round(value.x))),
        .y = @as(i32, @intFromFloat(std.math.round(value.y))),
        .z = @as(i32, @intFromFloat(std.math.round(value.z))),
    };
}

fn close_to_int(value: f64, tol: f64) bool {
    return @abs(value - std.math.round(value)) < tol;
}

fn map_index(op: KmeshOp, index: Index3, kmesh: [3]usize) Index3 {
    const mapped = mul_mat_vec(op.op.k_rot, index);
    return wrap_index(.{
        .x = mapped.x + op.delta.x,
        .y = mapped.y + op.delta.y,
        .z = mapped.z + op.delta.z,
    }, kmesh);
}

fn time_reversal_index(index: Index3, kmesh: [3]usize, shift2: Index3) Index3 {
    return wrap_index(.{
        .x = -index.x - shift2.x,
        .y = -index.y - shift2.y,
        .z = -index.z - shift2.z,
    }, kmesh);
}

fn wrap_index(index: Index3, kmesh: [3]usize) Index3 {
    return .{
        .x = mod_index(index.x, kmesh[0]),
        .y = mod_index(index.y, kmesh[1]),
        .z = mod_index(index.z, kmesh[2]),
    };
}

fn mod_index(value: i32, n: usize) i32 {
    const ni = @as(i32, @intCast(n));
    return @as(i32, @intCast(@mod(value, ni)));
}

fn mul_mat_vec(mat: symmetry.Mat3i, vec: Index3) Index3 {
    return .{
        .x = mat.m[0][0] * vec.x + mat.m[0][1] * vec.y + mat.m[0][2] * vec.z,
        .y = mat.m[1][0] * vec.x + mat.m[1][1] * vec.y + mat.m[1][2] * vec.z,
        .z = mat.m[2][0] * vec.x + mat.m[2][1] * vec.y + mat.m[2][2] * vec.z,
    };
}

fn mark_index(group: []bool, kmesh: [3]usize, index: Index3) void {
    const flat = flat_index(kmesh, index);
    group[flat] = true;
}

fn flat_index(kmesh: [3]usize, index: Index3) usize {
    const ix = @as(usize, @intCast(index.x));
    const iy = @as(usize, @intCast(index.y));
    const iz = @as(usize, @intCast(index.z));
    return ix + kmesh[0] * (iy + kmesh[1] * iz);
}

fn index_from_flat(kmesh: [3]usize, flat: usize) Index3 {
    const nx = kmesh[0];
    const ny = kmesh[1];
    const ix = flat % nx;
    const iy = (flat / nx) % ny;
    const iz = flat / (nx * ny);
    return .{ .x = @intCast(ix), .y = @intCast(iy), .z = @intCast(iz) };
}

fn index_from_kpoint(
    kp: symmetry.KPoint,
    kmesh: [3]usize,
    shift: math.Vec3,
    tol: f64,
) ?Index3 {
    const ix = match_index(kp.k_frac.x, kmesh[0], shift.x, tol) orelse return null;
    const iy = match_index(kp.k_frac.y, kmesh[1], shift.y, tol) orelse return null;
    const iz = match_index(kp.k_frac.z, kmesh[2], shift.z, tol) orelse return null;
    return .{ .x = ix, .y = iy, .z = iz };
}

fn match_index(value: f64, n: usize, shift: f64, tol: f64) ?i32 {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const frac = mesh.frac_from_index(@as(i32, @intCast(i)), n, shift);
        if (frac_close(frac, value, tol)) return @intCast(i);
    }
    return null;
}

fn frac_close(a: f64, b: f64, tol: f64) bool {
    const d = a - b;
    const centered = d - std.math.round(d);
    return @abs(centered) < tol;
}

fn is_identity_op(op: symmetry.SymOp) bool {
    return op.rot.m[0][0] == 1 and op.rot.m[0][1] == 0 and op.rot.m[0][2] == 0 and
        op.rot.m[1][0] == 0 and op.rot.m[1][1] == 1 and op.rot.m[1][2] == 0 and
        op.rot.m[2][0] == 0 and op.rot.m[2][1] == 0 and op.rot.m[2][2] == 1;
}
