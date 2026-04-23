const std = @import("std");

const math = @import("../math/math.zig");

pub fn diamond_conventional_cell(a: f64) math.Mat3 {
    return math.Mat3.from_rows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = a, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = a },
    );
}

pub fn diamond_conventional_fractional() [8]math.Vec3 {
    return .{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.5, .z = 0.5 },
        .{ .x = 0.5, .y = 0.0, .z = 0.5 },
        .{ .x = 0.5, .y = 0.5, .z = 0.0 },
        .{ .x = 0.25, .y = 0.25, .z = 0.25 },
        .{ .x = 0.25, .y = 0.75, .z = 0.75 },
        .{ .x = 0.75, .y = 0.25, .z = 0.75 },
        .{ .x = 0.75, .y = 0.75, .z = 0.25 },
    };
}

pub fn diamond_conventional_positions(a: f64) [8]math.Vec3 {
    const frac = diamond_conventional_fractional();
    var positions: [8]math.Vec3 = undefined;
    for (frac, 0..) |f, i| {
        positions[i] = .{ .x = f.x * a, .y = f.y * a, .z = f.z * a };
    }
    return positions;
}

pub fn diamond_conventional_supercell(
    alloc: std.mem.Allocator,
    a: f64,
    reps: [3]usize,
) ![]math.Vec3 {
    const base = diamond_conventional_fractional();
    const count = base.len * reps[0] * reps[1] * reps[2];
    const positions = try alloc.alloc(math.Vec3, count);
    var idx: usize = 0;
    var ix: usize = 0;
    while (ix < reps[0]) : (ix += 1) {
        var iy: usize = 0;
        while (iy < reps[1]) : (iy += 1) {
            var iz: usize = 0;
            while (iz < reps[2]) : (iz += 1) {
                const shift = math.Vec3{
                    .x = @as(f64, @floatFromInt(ix)),
                    .y = @as(f64, @floatFromInt(iy)),
                    .z = @as(f64, @floatFromInt(iz)),
                };
                for (base) |f| {
                    const frac = math.Vec3.add(f, shift);
                    positions[idx] = .{ .x = frac.x * a, .y = frac.y * a, .z = frac.z * a };
                    idx += 1;
                }
            }
        }
    }
    return positions;
}

test "diamond conventional positions count and bounds" {
    const a = 4.0;
    const positions = diamond_conventional_positions(a);
    try std.testing.expectEqual(@as(usize, 8), positions.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), positions[0].x, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), positions[7].y, 1e-12);
}

test "diamond supercell count" {
    const alloc = std.testing.allocator;
    const a = 5.0;
    const reps = [3]usize{ 2, 2, 2 };
    const positions = try diamond_conventional_supercell(alloc, a, reps);
    defer alloc.free(positions);

    try std.testing.expectEqual(@as(usize, 8 * 8), positions.len);
}
