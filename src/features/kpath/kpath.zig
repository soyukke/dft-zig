const std = @import("std");
const config = @import("../config/config.zig");
const math = @import("../math/math.zig");

pub const KPoint = struct {
    k_frac: math.Vec3,
    k_cart: math.Vec3,
    distance: f64,
    label: []const u8,
};

pub const KPath = struct {
    points: []KPoint,

    /// Free allocated k-point list.
    pub fn deinit(self: *KPath, alloc: std.mem.Allocator) void {
        if (self.points.len > 0) alloc.free(self.points);
    }
};

/// Generate interpolated k-points and cumulative distances.
pub fn generate(alloc: std.mem.Allocator, band: config.BandConfig, recip: math.Mat3) !KPath {
    if (band.path.len < 2) return KPath{ .points = &[_]KPoint{} };
    if (band.points_per_segment == 0) return error.InvalidBandPoints;

    var list: std.ArrayList(KPoint) = .empty;
    errdefer list.deinit(alloc);

    var distance: f64 = 0.0;
    var prev_cart: ?math.Vec3 = null;

    var i: usize = 0;
    while (i + 1 < band.path.len) : (i += 1) {
        const start = band.path[i];
        const end = band.path[i + 1];
        const delta = math.Vec3.sub(end.k, start.k);

        var step: usize = 0;
        while (step < band.points_per_segment) : (step += 1) {
            const t = @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(band.points_per_segment));
            const k_frac = math.Vec3.add(start.k, math.Vec3.scale(delta, t));
            const k_cart = math.fracToCart(k_frac, recip);
            if (prev_cart) |prev| {
                distance += math.Vec3.norm(math.Vec3.sub(k_cart, prev));
            }
            prev_cart = k_cart;

            const label = if (step == 0) start.label else "";
            try list.append(alloc, .{
                .k_frac = k_frac,
                .k_cart = k_cart,
                .distance = distance,
                .label = label,
            });
        }
    }

    const last = band.path[band.path.len - 1];
    const last_cart = math.fracToCart(last.k, recip);
    if (prev_cart) |prev| {
        distance += math.Vec3.norm(math.Vec3.sub(last_cart, prev));
    }
    try list.append(alloc, .{
        .k_frac = last.k,
        .k_cart = last_cart,
        .distance = distance,
        .label = last.label,
    });

    const points = try list.toOwnedSlice(alloc);
    return KPath{ .points = points };
}
