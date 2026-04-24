const std = @import("std");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const lebedev = @import("../grid/lebedev.zig");

/// Precomputed Gaunt coefficient table for PAW multi-L support.
///
/// Gaunt coefficients are integrals of three real spherical harmonics:
///   G(l1,m1, l2,m2, L,M) = integral Y_{l1,m1}(Omega) Y_{l2,m2}(Omega) Y_{L,M}(Omega) dOmega
///
/// These appear in the PAW compensation charge formula:
///   n_hat(G) = sum_a sum_ij rho_ij sum_{L,M}
///              G(l_i,m_i, l_j,m_j, L,M) * Q^L_ij(|G|)
///              * Y_{L,M}(G_hat) * exp(-iGR) / Omega
///
/// and similarly in D^hat and on-site quantities.
///
/// Layout: flat array indexed by (lm1, lm2, LM) where lm = l*l + l + m.
/// First two indices run over projector channels (up to lmax_proj),
/// third index runs over augmentation channels (up to lmax_aug).
pub const GauntTable = struct {
    /// Gaunt coefficients: layout [n_lm_proj * n_lm_proj * n_lm_aug]
    values: []f64,
    /// Maximum l for projectors (l1, l2)
    lmax_proj: usize,
    /// Maximum L for augmentation
    lmax_aug: usize,
    /// n_lm for projectors = (lmax_proj+1)^2
    n_lm_proj: usize,
    /// n_LM for augmentation = (lmax_aug+1)^2
    n_lm_aug: usize,

    /// Convert (l, m) to flat lm index: lm = l*l + l + m.
    /// m ranges from -l to +l.
    pub fn lm_index(l: usize, m: i32) usize {
        return l * l + @as(usize, @intCast(@as(i64, @intCast(l)) + m));
    }

    /// Get Gaunt coefficient G(l1, m1, l2, m2, L, M).
    /// Returns 0.0 if any index is out of range.
    pub fn get(
        self: *const GauntTable,
        l1: usize,
        m1: i32,
        l2: usize,
        m2: i32,
        big_l: usize,
        big_m: i32,
    ) f64 {
        const lm1 = lm_index(l1, m1);
        const lm2 = lm_index(l2, m2);
        const lm3 = lm_index(big_l, big_m);
        if (lm1 >= self.n_lm_proj or lm2 >= self.n_lm_proj or lm3 >= self.n_lm_aug) return 0.0;
        return self.values[(lm1 * self.n_lm_proj + lm2) * self.n_lm_aug + lm3];
    }

    /// Iterate over all non-zero Gaunt coefficients for a given (l1, m1, l2, m2) pair.
    /// Calls callback with (L, M, gaunt_value) for each non-zero entry.
    /// This avoids iterating over all (L, M) when most are zero due to selection rules.
    pub fn iter_non_zero(
        self: *const GauntTable,
        l1: usize,
        m1: i32,
        l2: usize,
        m2: i32,
        context: anytype,
        comptime callback: fn (@TypeOf(context), usize, i32, f64) void,
    ) void {
        const lm1 = lm_index(l1, m1);
        const lm2 = lm_index(l2, m2);
        if (lm1 >= self.n_lm_proj or lm2 >= self.n_lm_proj) return;
        const base = (lm1 * self.n_lm_proj + lm2) * self.n_lm_aug;
        for (0..self.n_lm_aug) |lm3| {
            const val = self.values[base + lm3];
            if (val != 0.0) {
                // Recover (L, M) from flat index lm3
                // lm3 = L*L + L + M => L = floor(sqrt(lm3)), M = lm3 - L*L - L
                const big_l = l_from_lm_index(lm3);
                const big_m: i32 = @intCast(
                    @as(i64, @intCast(lm3)) - @as(i64, @intCast(big_l * big_l + big_l)),
                );
                callback(context, big_l, big_m, val);
            }
        }
    }

    /// Recover l from a flat lm index.
    /// lm = l^2 + l + m, so l = floor(sqrt(lm)).
    fn l_from_lm_index(lm: usize) usize {
        const s = @sqrt(@as(f64, @floatFromInt(lm)));
        return @intFromFloat(@floor(s));
    }

    /// Build Gaunt table using Lebedev quadrature.
    ///
    /// lmax_proj: maximum l for projector channels (l_i, l_j).
    ///   For Si PAW: lmax_proj = 2 (s, p, d projectors).
    /// lmax_aug: maximum L for augmentation channels.
    ///   Typically lmax_aug = 2 * lmax_proj (e.g., 4 for Si).
    ///
    /// The Lebedev grid is chosen to exactly integrate the product of three
    /// spherical harmonics of combined degree lmax_proj + lmax_proj + lmax_aug.
    pub fn init(alloc: std.mem.Allocator, lmax_proj: usize, lmax_aug: usize) !GauntTable {
        if (lmax_proj > 4) return error.LmaxTooLarge;
        if (lmax_aug > 4) return error.LmaxTooLarge;

        const n_lm_proj = (lmax_proj + 1) * (lmax_proj + 1);
        const n_lm_aug = (lmax_aug + 1) * (lmax_aug + 1);
        const n_total = n_lm_proj * n_lm_proj * n_lm_aug;

        const values = try alloc.alloc(f64, n_total);
        @memset(values, 0.0);
        accumulate_gaunt_over_sphere(values, lmax_proj, lmax_aug, n_lm_proj, n_lm_aug);
        for (values) |*v| {
            if (@abs(v.*) < 1e-14) v.* = 0.0;
        }
        return .{
            .values = values,
            .lmax_proj = lmax_proj,
            .lmax_aug = lmax_aug,
            .n_lm_proj = n_lm_proj,
            .n_lm_aug = n_lm_aug,
        };
    }

    pub fn deinit(self: *GauntTable, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
    }
};

fn accumulate_gaunt_over_sphere(
    values: []f64,
    lmax_proj: usize,
    lmax_aug: usize,
    n_lm_proj: usize,
    n_lm_aug: usize,
) void {
    const max_lm = 25;
    const grid = lebedev.get_lebedev_grid(302);
    for (grid) |pt| {
        const w = pt.w * 4.0 * std.math.pi;
        var ylm_proj: [max_lm]f64 = undefined;
        eval_all_real_ylm(lmax_proj, pt.x, pt.y, pt.z, &ylm_proj);
        var ylm_aug: [max_lm]f64 = undefined;
        eval_all_real_ylm(lmax_aug, pt.x, pt.y, pt.z, &ylm_aug);
        for (0..n_lm_proj) |lm1| {
            const y1 = ylm_proj[lm1];
            if (@abs(y1) < 1e-30) continue;
            for (0..n_lm_proj) |lm2| {
                const y2 = ylm_proj[lm2];
                if (@abs(y2) < 1e-30) continue;
                const y12 = y1 * y2 * w;
                const base = (lm1 * n_lm_proj + lm2) * n_lm_aug;
                for (0..n_lm_aug) |lm3| {
                    values[base + lm3] += y12 * ylm_aug[lm3];
                }
            }
        }
    }
}

fn eval_all_real_ylm(lmax: usize, x: f64, y: f64, z: f64, out: []f64) void {
    for (0..lmax + 1) |l| {
        const l_i32: i32 = @intCast(l);
        var m: i32 = -l_i32;
        while (m <= l_i32) : (m += 1) {
            out[GauntTable.lm_index(l, m)] = nonlocal.real_spherical_harmonic(l_i32, m, x, y, z);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "lm_index round-trip" {
    // l=0, m=0 => 0
    try std.testing.expectEqual(@as(usize, 0), GauntTable.lm_index(0, 0));
    // l=1, m=-1 => 1, m=0 => 2, m=1 => 3
    try std.testing.expectEqual(@as(usize, 1), GauntTable.lm_index(1, -1));
    try std.testing.expectEqual(@as(usize, 2), GauntTable.lm_index(1, 0));
    try std.testing.expectEqual(@as(usize, 3), GauntTable.lm_index(1, 1));
    // l=2, m=-2 => 4
    try std.testing.expectEqual(@as(usize, 4), GauntTable.lm_index(2, -2));
    // l=2, m=2 => 8
    try std.testing.expectEqual(@as(usize, 8), GauntTable.lm_index(2, 2));
}

test "Gaunt G(0,0, 0,0, 0,0) = 1/sqrt(4*pi)" {
    // integral Y_00 * Y_00 * Y_00 dOmega = (1/sqrt(4pi))^3 * 4pi = 1/sqrt(4pi)
    const alloc = std.testing.allocator;
    var table = try GauntTable.init(alloc, 0, 0);
    defer table.deinit(alloc);

    const expected = 1.0 / @sqrt(4.0 * std.math.pi);
    const got = table.get(0, 0, 0, 0, 0, 0);
    try std.testing.expectApproxEqAbs(expected, got, 1e-12);
}

test "Gaunt orthonormality: G(l1,m1, l2,m2, 0,0) = delta(l1,l2)*delta(m1,m2) / sqrt(4pi)" {
    // integral Y_{l1,m1} * Y_{l2,m2} * Y_{0,0} dOmega
    //   = Y_{0,0} * integral Y_{l1,m1} * Y_{l2,m2} dOmega
    //   = (1/sqrt(4pi)) * delta(l1,l2) * delta(m1,m2)
    const alloc = std.testing.allocator;
    var table = try GauntTable.init(alloc, 2, 4);
    defer table.deinit(alloc);

    const inv_sqrt_4pi = 1.0 / @sqrt(4.0 * std.math.pi);

    // Check diagonal: G(l,m, l,m, 0,0) = 1/sqrt(4pi)
    for (0..3) |l| {
        const l_i32: i32 = @intCast(l);
        var m: i32 = -l_i32;
        while (m <= l_i32) : (m += 1) {
            const got = table.get(l, m, l, m, 0, 0);
            try std.testing.expectApproxEqAbs(inv_sqrt_4pi, got, 1e-12);
        }
    }

    // Check off-diagonal: G(0,0, 1,0, 0,0) = 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), table.get(0, 0, 1, 0, 0, 0), 1e-14);
    // G(1,0, 1,1, 0,0) = 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), table.get(1, 0, 1, 1, 0, 0), 1e-14);
    // G(1,-1, 2,0, 0,0) = 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), table.get(1, -1, 2, 0, 0, 0), 1e-14);
}

test "Gaunt selection rule: l1+l2+L must be even" {
    // When l1+l2+L is odd, the Gaunt coefficient must be zero.
    const alloc = std.testing.allocator;
    var table = try GauntTable.init(alloc, 2, 4);
    defer table.deinit(alloc);

    // l1=0, l2=0, L=1 => 0+0+1=1 (odd) => should be 0
    for (0..3) |big_m_idx| {
        const big_m: i32 = @as(i32, @intCast(big_m_idx)) - 1;
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), table.get(0, 0, 0, 0, 1, big_m), 1e-14);
    }

    // l1=1, l2=1, L=1 => 1+1+1=3 (odd) => should be 0
    var m1: i32 = -1;
    while (m1 <= 1) : (m1 += 1) {
        var m2: i32 = -1;
        while (m2 <= 1) : (m2 += 1) {
            var big_m: i32 = -1;
            while (big_m <= 1) : (big_m += 1) {
                try std.testing.expectApproxEqAbs(
                    @as(f64, 0.0),
                    table.get(1, m1, 1, m2, 1, big_m),
                    1e-14,
                );
            }
        }
    }
}

test "Gaunt known value: G(1,0, 1,0, 2,0)" {
    // Analytic: integral Y_10 * Y_10 * Y_20 dOmega
    // Y_10 = sqrt(3/4pi) * cos(theta)
    // Y_20 = sqrt(5/16pi) * (3*cos^2(theta) - 1)
    // Integral = sqrt(3/4pi)^2 * sqrt(5/16pi) * integral (3*cos^4 - cos^2) sin dtheta dphi
    //          = (3/4pi) * sqrt(5/16pi) * 2*pi * (3*4/5 - 2/3) ... let me use the exact formula:
    // G(1,0,1,0,2,0) = sqrt(5/(4pi)) * C(1,1,2;0,0,0)^2 / sqrt(4pi)
    // From Clebsch-Gordan: C(1,0,1,0|2,0) = sqrt(2/3) for real harmonics
    // Exact value: 1/(2*sqrt(pi)) * sqrt(1/5) = ...
    // Let's just compute it numerically and verify it's non-zero and reasonable.
    const alloc = std.testing.allocator;
    var table = try GauntTable.init(alloc, 2, 4);
    defer table.deinit(alloc);

    const val = table.get(1, 0, 1, 0, 2, 0);

    // Numerical integration gives:
    //   sqrt(5)/(4*pi*sqrt(pi)) * 2*pi * integral_0^pi (3cos^4-cos^2) sin d theta
    // = sqrt(5)/(2*sqrt(pi)) * [3*2/5 - 2/3]
    // = sqrt(5)/(2*sqrt(pi)) * (6/5 - 2/3)
    // = sqrt(5)/(2*sqrt(pi)) * 8/15
    // But let's use the standard formula: G(1,0,1,0,2,0) = (1/sqrt(4pi)) * sqrt(5/pi) * (1/5)
    // Actually, more directly:
    // integral Y_10^2 Y_20 = (3/(4pi)) * sqrt(5/(16pi)) * 2pi
    //                      * integral_0^pi cos^2(t) (3cos^2(t)-1) sin(t) dt
    // = (3/(4pi)) * sqrt(5/(16pi)) * 2pi * [3*2/5 - 2/3]
    // = (3/2) * sqrt(5/(16pi)) * 8/15
    // = (3/2) * (8/15) * sqrt(5/(16pi))
    // = (4/5) * sqrt(5/(16pi))
    // = 4/(5*4*sqrt(pi/5))
    // = 1/(5*sqrt(pi/5))
    // = sqrt(5)/(5*sqrt(pi))
    // = 1/(sqrt(5)*sqrt(pi))
    // = 1/sqrt(5*pi)
    const expected = 1.0 / @sqrt(5.0 * std.math.pi);
    try std.testing.expectApproxEqAbs(expected, val, 1e-12);
}

test "Gaunt symmetry: G(l1,m1, l2,m2, L,M) = G(l2,m2, l1,m1, L,M)" {
    // The integral is symmetric in the first two harmonics.
    const alloc = std.testing.allocator;
    var table = try GauntTable.init(alloc, 2, 4);
    defer table.deinit(alloc);

    // Check a few representative cases
    for (0..3) |l1| {
        const l1_i32: i32 = @intCast(l1);
        var m1: i32 = -l1_i32;
        while (m1 <= l1_i32) : (m1 += 1) {
            for (0..3) |l2| {
                const l2_i32: i32 = @intCast(l2);
                var m2: i32 = -l2_i32;
                while (m2 <= l2_i32) : (m2 += 1) {
                    for (0..5) |big_l| {
                        const big_l_i32: i32 = @intCast(big_l);
                        var big_m: i32 = -big_l_i32;
                        while (big_m <= big_l_i32) : (big_m += 1) {
                            const g1 = table.get(l1, m1, l2, m2, big_l, big_m);
                            const g2 = table.get(l2, m2, l1, m1, big_l, big_m);
                            try std.testing.expectApproxEqAbs(g1, g2, 1e-14);
                        }
                    }
                }
            }
        }
    }
}

test "Gaunt L=0 sum rule: sum_M G(l,m, l,m, 0,0) * Y_00 = Y_00^2 integral Y_lm^2 = Y_00" {
    // This is really just a restatement of orthonormality.
    // sum over all (l,m): sum_{lm} G(lm,lm,00) should equal n_lm / sqrt(4pi)
    const alloc = std.testing.allocator;
    var table = try GauntTable.init(alloc, 2, 4);
    defer table.deinit(alloc);

    const inv_sqrt_4pi = 1.0 / @sqrt(4.0 * std.math.pi);
    var sum: f64 = 0.0;
    for (0..3) |l| {
        const l_i32: i32 = @intCast(l);
        var m: i32 = -l_i32;
        while (m <= l_i32) : (m += 1) {
            sum += table.get(l, m, l, m, 0, 0);
        }
    }
    // sum should be 9 * 1/sqrt(4pi) since there are 9 (l,m) channels for lmax=2
    try std.testing.expectApproxEqAbs(9.0 * inv_sqrt_4pi, sum, 1e-12);
}

test "Gaunt table with lmax_proj=2 lmax_aug=4 has correct dimensions" {
    const alloc = std.testing.allocator;
    var table = try GauntTable.init(alloc, 2, 4);
    defer table.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 9), table.n_lm_proj); // (2+1)^2 = 9
    try std.testing.expectEqual(@as(usize, 25), table.n_lm_aug); // (4+1)^2 = 25
    try std.testing.expectEqual(@as(usize, 9 * 9 * 25), table.values.len);
}

test "Gaunt out-of-range returns 0" {
    const alloc = std.testing.allocator;
    var table = try GauntTable.init(alloc, 1, 2);
    defer table.deinit(alloc);

    // l=2 is out of range for lmax_proj=1
    try std.testing.expectEqual(@as(f64, 0.0), table.get(2, 0, 0, 0, 0, 0));
    // L=3 is out of range for lmax_aug=2
    try std.testing.expectEqual(@as(f64, 0.0), table.get(0, 0, 0, 0, 3, 0));
}
