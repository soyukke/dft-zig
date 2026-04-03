const std = @import("std");
const math = @import("../math/math.zig");
const grid_mod = @import("pw_grid.zig");
const Grid = grid_mod.Grid;

/// Single G-vector item yielded by the iterator.
pub const GVecItem = struct {
    gh: i32,
    gk: i32,
    gl: i32,
    gvec: math.Vec3,
    g2: f64,
    idx: usize,
};

/// Iterator over all G-vectors in an FFT grid.
/// Iteration order: l (outermost) → k → h (innermost), matching FFT memory layout.
/// idx increments sequentially from 0 to nx*ny*nz - 1.
pub const GVecIterator = struct {
    b1: math.Vec3,
    b2: math.Vec3,
    b3: math.Vec3,
    min_h: i32,
    min_k: i32,
    min_l: i32,
    nx: usize,
    ny: usize,
    nz: usize,
    // current position (h innermost, l outermost)
    h: usize,
    k: usize,
    l: usize,
    idx: usize,

    pub fn init(grid: anytype) GVecIterator {
        const T = @TypeOf(grid);
        // Unwrap pointer types so @hasField works for both Grid and *Grid / *const Grid.
        const Inner = switch (@typeInfo(T)) {
            .pointer => |p| p.child,
            else => T,
        };
        comptime {
            for (.{ "recip", "min_h", "min_k", "min_l", "nx", "ny", "nz" }) |field| {
                if (!@hasField(Inner, field))
                    @compileError("GVecIterator: grid type missing field '" ++ field ++ "'");
            }
        }
        return .{
            .b1 = grid.recip.row(0),
            .b2 = grid.recip.row(1),
            .b3 = grid.recip.row(2),
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .h = 0,
            .k = 0,
            .l = 0,
            .idx = 0,
        };
    }

    pub fn next(self: *GVecIterator) ?GVecItem {
        if (self.l >= self.nz) return null;

        const gh = self.min_h + @as(i32, @intCast(self.h));
        const gk = self.min_k + @as(i32, @intCast(self.k));
        const gl = self.min_l + @as(i32, @intCast(self.l));
        const gvec = math.Vec3.add(
            math.Vec3.add(
                math.Vec3.scale(self.b1, @as(f64, @floatFromInt(gh))),
                math.Vec3.scale(self.b2, @as(f64, @floatFromInt(gk))),
            ),
            math.Vec3.scale(self.b3, @as(f64, @floatFromInt(gl))),
        );

        const item = GVecItem{
            .gh = gh,
            .gk = gk,
            .gl = gl,
            .gvec = gvec,
            .g2 = math.Vec3.dot(gvec, gvec),
            .idx = self.idx,
        };

        // Advance: h (innermost) → k → l (outermost)
        self.idx += 1;
        self.h += 1;
        if (self.h >= self.nx) {
            self.h = 0;
            self.k += 1;
            if (self.k >= self.ny) {
                self.k = 0;
                self.l += 1;
            }
        }

        return item;
    }
};

// ============================================================================
// Tests
// ============================================================================

fn makeTestGrid(nx: usize, ny: usize, nz: usize) Grid {
    const twopi = 2.0 * std.math.pi;
    return .{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .min_h = -@as(i32, @intCast(nx / 2)),
        .min_k = -@as(i32, @intCast(ny / 2)),
        .min_l = -@as(i32, @intCast(nz / 2)),
        .cell = math.Mat3.fromRows(
            .{ .x = 1, .y = 0, .z = 0 },
            .{ .x = 0, .y = 1, .z = 0 },
            .{ .x = 0, .y = 0, .z = 1 },
        ),
        .recip = math.Mat3.fromRows(
            .{ .x = twopi, .y = 0, .z = 0 },
            .{ .x = 0, .y = twopi, .z = 0 },
            .{ .x = 0, .y = 0, .z = twopi },
        ),
        .volume = 1.0,
    };
}

test "GVecIterator yields correct count" {
    const grid = makeTestGrid(3, 4, 5);
    var it = GVecIterator.init(grid);
    var count: usize = 0;
    while (it.next()) |_| count += 1;
    try std.testing.expectEqual(@as(usize, 60), count);
}

test "GVecIterator idx is sequential" {
    const grid = makeTestGrid(3, 3, 3);
    var it = GVecIterator.init(grid);
    var expected_idx: usize = 0;
    while (it.next()) |item| {
        try std.testing.expectEqual(expected_idx, item.idx);
        expected_idx += 1;
    }
    try std.testing.expectEqual(@as(usize, 27), expected_idx);
}

test "GVecIterator G=0 present" {
    const grid = makeTestGrid(3, 3, 3);
    var it = GVecIterator.init(grid);
    var found_g0 = false;
    while (it.next()) |item| {
        if (item.gh == 0 and item.gk == 0 and item.gl == 0) {
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), item.g2, 1e-15);
            found_g0 = true;
        }
    }
    try std.testing.expect(found_g0);
}

test "GVecIterator loop order l-outer k h-inner" {
    // 2x2x2 grid: verify the first few items follow l→k→h order
    const grid = makeTestGrid(2, 2, 2);
    var it = GVecIterator.init(grid);

    // idx=0: l=0 (gl=min_l), k=0 (gk=min_k), h=0 (gh=min_h)
    const item0 = it.next().?;
    try std.testing.expectEqual(grid.min_h, item0.gh);
    try std.testing.expectEqual(grid.min_k, item0.gk);
    try std.testing.expectEqual(grid.min_l, item0.gl);
    try std.testing.expectEqual(@as(usize, 0), item0.idx);

    // idx=1: h advances (innermost)
    const item1 = it.next().?;
    try std.testing.expectEqual(grid.min_h + 1, item1.gh);
    try std.testing.expectEqual(grid.min_k, item1.gk);
    try std.testing.expectEqual(grid.min_l, item1.gl);
    try std.testing.expectEqual(@as(usize, 1), item1.idx);

    // idx=2: h wraps, k advances
    const item2 = it.next().?;
    try std.testing.expectEqual(grid.min_h, item2.gh);
    try std.testing.expectEqual(grid.min_k + 1, item2.gk);
    try std.testing.expectEqual(grid.min_l, item2.gl);
    try std.testing.expectEqual(@as(usize, 2), item2.idx);

    // idx=3: h advances again
    const item3 = it.next().?;
    try std.testing.expectEqual(grid.min_h + 1, item3.gh);
    try std.testing.expectEqual(grid.min_k + 1, item3.gk);
    try std.testing.expectEqual(grid.min_l, item3.gl);
    try std.testing.expectEqual(@as(usize, 3), item3.idx);

    // idx=4: h wraps, k wraps, l advances (outermost)
    const item4 = it.next().?;
    try std.testing.expectEqual(grid.min_h, item4.gh);
    try std.testing.expectEqual(grid.min_k, item4.gk);
    try std.testing.expectEqual(grid.min_l + 1, item4.gl);
    try std.testing.expectEqual(@as(usize, 4), item4.idx);
}

test "GVecIterator gvec matches manual calculation" {
    const twopi = 2.0 * std.math.pi;
    const grid = makeTestGrid(3, 1, 1);
    var it = GVecIterator.init(grid);

    // First item: gh=min_h=-1, gk=0, gl=0 => gvec = (-2pi, 0, 0)
    const first = it.next().?;
    try std.testing.expectEqual(@as(i32, -1), first.gh);
    try std.testing.expectApproxEqAbs(-twopi, first.gvec.x, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), first.gvec.y, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), first.gvec.z, 1e-12);
    try std.testing.expectApproxEqAbs(twopi * twopi, first.g2, 1e-10);
}

test "GVecIterator matches manual l-k-h loop" {
    // Verify iterator output exactly matches the manual triple loop used in energy.zig etc.
    const grid = makeTestGrid(4, 3, 5);
    var it = GVecIterator.init(grid);

    var manual_idx: usize = 0;
    var l: usize = 0;
    while (l < grid.nz) : (l += 1) {
        var k: usize = 0;
        while (k < grid.ny) : (k += 1) {
            var h: usize = 0;
            while (h < grid.nx) : (h += 1) {
                const gh = grid.min_h + @as(i32, @intCast(h));
                const gk = grid.min_k + @as(i32, @intCast(k));
                const gl = grid.min_l + @as(i32, @intCast(l));

                const item = it.next().?;
                try std.testing.expectEqual(gh, item.gh);
                try std.testing.expectEqual(gk, item.gk);
                try std.testing.expectEqual(gl, item.gl);
                try std.testing.expectEqual(manual_idx, item.idx);
                manual_idx += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(?GVecItem, null), it.next());
}
