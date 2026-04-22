const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("../symmetry/symmetry.zig");

pub const GVector = struct {
    h: i32,
    k: i32,
    l: i32,
    cart: math.Vec3,
    kpg: math.Vec3,
    kinetic: f64,
};

pub const Basis = struct {
    gvecs: []GVector,

    /// Free allocated G-vectors.
    pub fn deinit(self: *Basis, alloc: std.mem.Allocator) void {
        if (self.gvecs.len > 0) {
            alloc.free(self.gvecs);
        }
    }
};

/// Generate plane-wave basis for a k-point within cutoff (Ry).
pub fn generate(
    alloc: std.mem.Allocator,
    recip: math.Mat3,
    ecut_ry: f64,
    k_cart: math.Vec3,
) !Basis {
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);

    const min_b = @min(@min(math.Vec3.norm(b1), math.Vec3.norm(b2)), math.Vec3.norm(b3));
    const gmax = std.math.sqrt(ecut_ry);
    const max_n = @as(i32, @intFromFloat(std.math.ceil(gmax / min_b))) + 1;

    var list: std.ArrayList(GVector) = .empty;
    errdefer list.deinit(alloc);

    var h: i32 = -max_n;
    while (h <= max_n) : (h += 1) {
        var k: i32 = -max_n;
        while (k <= max_n) : (k += 1) {
            var l: i32 = -max_n;
            while (l <= max_n) : (l += 1) {
                const g_cart = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(h))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(k))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(l))),
                );
                const kpg = math.Vec3.add(k_cart, g_cart);
                const kinetic = math.Vec3.dot(kpg, kpg);
                if (kinetic <= ecut_ry) {
                    try list.append(alloc, .{
                        .h = h,
                        .k = k,
                        .l = l,
                        .cart = g_cart,
                        .kpg = kpg,
                        .kinetic = kinetic,
                    });
                }
            }
        }
    }

    const gvecs = try list.toOwnedSlice(alloc);
    return Basis{ .gvecs = gvecs };
}

/// Build a mapping from basis_k G-vectors to basis_sk G-vectors
/// under rotation rot (reciprocal-space integer matrix).
///
/// G' = sign * (rot * G) + delta_hkl
///
/// where sign = -1 for time reversal (negate=true), +1 otherwise.
/// delta_hkl accounts for BZ folding: when Sk wraps around the BZ boundary,
/// G-vectors shift by a reciprocal lattice vector.
///
/// delta_hkl = round(k_full_frac - sign * k_rot * k_ibz_frac)
///
/// mapping[i] = index j in basis_sk such that basis_sk.gvecs[j].hkl == G',
/// or null if not found (cutoff boundary effect).
pub fn buildSymmetryMapping(
    alloc: std.mem.Allocator,
    basis_k: Basis,
    basis_sk: Basis,
    rot: symmetry.Mat3i,
    negate: bool,
    delta_hkl: [3]i32,
) ![]?usize {
    // Build hash map from (h,k,l) -> index for basis_sk
    var hkl_map = std.AutoHashMap([3]i32, usize).init(alloc);
    defer hkl_map.deinit();
    try hkl_map.ensureTotalCapacity(@intCast(basis_sk.gvecs.len));
    for (basis_sk.gvecs, 0..) |gv, j| {
        hkl_map.putAssumeCapacity(.{ gv.h, gv.k, gv.l }, j);
    }

    const sign: i32 = if (negate) -1 else 1;
    const mapping = try alloc.alloc(?usize, basis_k.gvecs.len);
    for (basis_k.gvecs, 0..) |gv, i| {
        const h0 = gv.h;
        const k0 = gv.k;
        const l0 = gv.l;
        const h1 = sign * (rot.m[0][0] * h0 + rot.m[0][1] * k0 + rot.m[0][2] * l0) + delta_hkl[0];
        const k1 = sign * (rot.m[1][0] * h0 + rot.m[1][1] * k0 + rot.m[1][2] * l0) + delta_hkl[1];
        const l1 = sign * (rot.m[2][0] * h0 + rot.m[2][1] * k0 + rot.m[2][2] * l0) + delta_hkl[2];
        mapping[i] = hkl_map.get(.{ h1, k1, l1 });
    }

    return mapping;
}
