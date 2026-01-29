const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("../symmetry/symmetry.zig");

pub fn generateKmesh(
    alloc: std.mem.Allocator,
    kmesh: [3]usize,
    recip: math.Mat3,
    shift: math.Vec3,
) ![]symmetry.KPoint {
    const total = kmesh[0] * kmesh[1] * kmesh[2];
    if (total == 0) return error.InvalidKmesh;

    var list = try alloc.alloc(symmetry.KPoint, total);
    var idx: usize = 0;
    var i: usize = 0;
    while (i < kmesh[0]) : (i += 1) {
        var j: usize = 0;
        while (j < kmesh[1]) : (j += 1) {
            var k: usize = 0;
            while (k < kmesh[2]) : (k += 1) {
                const fx = fracFromIndex(@as(i32, @intCast(i)), kmesh[0], shift.x);
                const fy = fracFromIndex(@as(i32, @intCast(j)), kmesh[1], shift.y);
                const fz = fracFromIndex(@as(i32, @intCast(k)), kmesh[2], shift.z);
                const k_frac = math.Vec3{ .x = fx, .y = fy, .z = fz };
                const k_cart = math.fracToCart(k_frac, recip);
                list[idx] = .{
                    .k_frac = k_frac,
                    .k_cart = k_cart,
                    .weight = 1.0 / @as(f64, @floatFromInt(total)),
                };
                idx += 1;
            }
        }
    }
    return list;
}

pub fn fracFromIndex(index: i32, n: usize, shift: f64) f64 {
    const nf = @as(f64, @floatFromInt(n));
    const value = (@as(f64, @floatFromInt(index)) + shift) / nf;
    return wrapCenteredScalar(value);
}

fn wrapCenteredScalar(value: f64) f64 {
    return value - std.math.round(value);
}
