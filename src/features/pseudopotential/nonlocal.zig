const std = @import("std");

const ctrap_weight = @import("../math/math.zig").radial.ctrap_weight;

/// Compute radial projector integral for given l and |g|.
/// Note: UPF files store r*beta(r) in PP_BETA, so we use r*beta*jl*dr
/// which gives ∫ r² β(r) jl(gr) dr after accounting for the r factor.
pub fn radial_projector(beta: []const f64, r: []const f64, rab: []const f64, l: i32, g: f64) f64 {
    std.debug.assert(beta.len == r.len);
    std.debug.assert(r.len == rab.len);
    const n = beta.len;
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const x = g * r[i];
        const jl = spherical_bessel(l, x);
        // beta[i] is r*beta(r), so we multiply by r (not r²) to get r²*beta(r)
        sum += r[i] * beta[i] * jl * rab[i] * ctrap_weight(i, n);
    }
    return sum;
}

/// Compute angular factor 4pi(2l+1)P_l(cos).
pub fn angular_factor(l: i32, cos_gamma: f64) f64 {
    const p = legendre_p(l, cos_gamma);
    return 4.0 * std.math.pi * (2.0 * @as(f64, @floatFromInt(l)) + 1.0) * p;
}

/// Compute Legendre polynomial P_l(x) for l<=3.
pub fn legendre_p(l: i32, x: f64) f64 {
    switch (l) {
        0 => return 1.0,
        1 => return x,
        2 => return 0.5 * (3.0 * x * x - 1.0),
        3 => return 0.5 * (5.0 * x * x * x - 3.0 * x),
        else => return legendre_p_rec(l, x),
    }
}

/// Compute Legendre polynomial via recursion for l>=0.
fn legendre_p_rec(l: i32, x: f64) f64 {
    if (l == 0) return 1.0;
    if (l == 1) return x;
    var pnm1: f64 = x;
    var pnm2: f64 = 1.0;
    var n: i32 = 1;
    while (n < l) : (n += 1) {
        const nf = @as(f64, @floatFromInt(n));
        const pn = ((2.0 * nf + 1.0) * x * pnm1 - nf * pnm2) / (nf + 1.0);
        pnm2 = pnm1;
        pnm1 = pn;
    }
    return pnm1;
}

/// Compute spherical Bessel j_l(x) using Miller's algorithm (downward recursion).
/// This is numerically stable for all l and x, unlike forward recursion.
/// Based on ABINIT's sbf8 implementation.
pub fn spherical_bessel(l: i32, x: f64) f64 {
    const ax = @abs(x);
    if (ax < 1e-6) return spherical_bessel_small(l, x);

    // For l=0,1 use direct formulas (always stable)
    if (l == 0) return std.math.sin(x) / x;
    if (l == 1) return std.math.sin(x) / (x * x) - std.math.cos(x) / x;

    // For l>=2, use Miller's algorithm (downward recursion + normalization)
    return spherical_bessel_miller(l, x);
}

/// Miller's algorithm for spherical Bessel functions.
/// Uses downward recursion from a high l value, then normalizes using sum rule.
fn spherical_bessel_miller(l: i32, x: f64) f64 {
    const nm: usize = @intCast(l + 1);

    // Determine how high to start the downward recursion
    // Formula from ABINIT: nlim = nm + int(1.36*x) + 15 for x >= 1
    //                      nlim = nm + int(15*x) + 1 for x < 1
    const nlim: usize = if (x < 1.0)
        nm + @as(usize, @intFromFloat(15.0 * x)) + 1
    else
        nm + @as(usize, @intFromFloat(1.36 * x)) + 15;

    // Allocate working array on stack (max reasonable size)
    const max_nlim: usize = 256;
    if (nlim > max_nlim) {
        // Fallback to direct formula for extreme cases
        return spherical_bessel_direct(l, x);
    }

    var sb: [max_nlim + 2]f64 = undefined;

    // Start downward recursion from arbitrary small value
    const xi = 1.0 / x;
    sb[nlim + 1] = 0.0;
    sb[nlim] = 1.0e-18;

    // Downward recursion: j_{n-1} = (2n+1)/x * j_n - j_{n+1}
    var n: usize = nlim;
    while (n >= 1) : (n -= 1) {
        const nf = @as(f64, @floatFromInt(n));
        sb[n - 1] = (2.0 * nf + 1.0) * xi * sb[n] - sb[n + 1];
    }

    // Normalize using sum rule: sum_{n=0}^{inf} (2n+1) j_n^2(x) = 1
    // Actually: sum_{n=0}^{inf} (2n-1) j_n^2(x) = 1 (with j_{-1} = cos(x)/x)
    // We use: j_0(x) = sin(x)/x to normalize
    const j0_exact = std.math.sin(x) / x;
    const scale = j0_exact / sb[0];

    return sb[nm - 1] * scale;
}

/// Direct formula for spherical Bessel (used as fallback for extreme cases).
fn spherical_bessel_direct(l: i32, x: f64) f64 {
    // For very large l, the function is essentially zero
    const lf = @as(f64, @floatFromInt(l));
    if (lf > x + 50.0) return 0.0;

    // Use the general formula with better numerical properties
    // j_l(x) = sqrt(pi/(2x)) * J_{l+1/2}(x)
    // For now, use forward recursion but with double precision checks
    var jlm1 = std.math.sin(x) / x;
    var jl = std.math.sin(x) / (x * x) - std.math.cos(x) / x;
    var n: i32 = 1;
    while (n < l) : (n += 1) {
        const nf = @as(f64, @floatFromInt(n));
        const jlp1 = (2.0 * nf + 1.0) / x * jl - jlm1;
        jlm1 = jl;
        jl = jlp1;
        // Check for overflow/underflow
        if (!std.math.isFinite(jl)) return 0.0;
    }
    return jl;
}

/// Compute small-argument series for spherical Bessel.
fn spherical_bessel_small(l: i32, x: f64) f64 {
    const x2 = x * x;
    switch (l) {
        0 => return 1.0 - x2 / 6.0 + x2 * x2 / 120.0,
        1 => return x / 3.0 - x * x2 / 30.0,
        2 => return x2 / 15.0 - x2 * x2 / 210.0,
        3 => return x * x2 / 105.0,
        else => return 0.0,
    }
}

/// Compute real spherical harmonic Y_l^m(r̂) for unit vector r̂.
/// Uses tesseral harmonics (real combinations of complex Y_l^m).
/// The normalization satisfies: Σ_m Y_l^m(r̂) * Y_l^m(r̂') = (2l+1)/(4π) * P_l(cos_γ)
pub fn real_spherical_harmonic(l: i32, m: i32, x: f64, y: f64, z: f64) f64 {
    const r2 = x * x + y * y + z * z;
    if (r2 < 1e-30) {
        // At origin, only l=0 term is non-zero
        return if (l == 0) 1.0 / @sqrt(4.0 * std.math.pi) else 0.0;
    }
    const inv_r = 1.0 / @sqrt(r2);
    const nx = x * inv_r;
    const ny = y * inv_r;
    const nz = z * inv_r;

    return switch (l) {
        0 => real_ylm0(m),
        1 => real_ylm1(m, nx, ny, nz),
        2 => real_ylm2(m, nx, ny, nz),
        3 => real_ylm3(m, nx, ny, nz),
        4 => real_ylm4(m, nx, ny, nz),
        else => 0.0,
    };
}

/// Y_0^0 = 1/sqrt(4π)
fn real_ylm0(m: i32) f64 {
    return if (m == 0) 1.0 / @sqrt(4.0 * std.math.pi) else 0.0;
}

/// Real spherical harmonics for l=1
/// Y_1^{-1} = sqrt(3/4π) * y
/// Y_1^0 = sqrt(3/4π) * z
/// Y_1^1 = sqrt(3/4π) * x
fn real_ylm1(m: i32, nx: f64, ny: f64, nz: f64) f64 {
    const c = @sqrt(3.0 / (4.0 * std.math.pi));
    return switch (m) {
        -1 => c * ny,
        0 => c * nz,
        1 => c * nx,
        else => 0.0,
    };
}

/// Real spherical harmonics for l=2
fn real_ylm2(m: i32, nx: f64, ny: f64, nz: f64) f64 {
    const c0 = @sqrt(5.0 / (16.0 * std.math.pi)); // for m=0
    const c1 = @sqrt(15.0 / (4.0 * std.math.pi)); // for |m|=1
    const c2 = @sqrt(15.0 / (16.0 * std.math.pi)); // for |m|=2

    return switch (m) {
        -2 => c2 * 2.0 * nx * ny, // xy
        -1 => c1 * ny * nz, // yz
        0 => c0 * (3.0 * nz * nz - 1.0), // 3z²-1
        1 => c1 * nx * nz, // xz
        2 => c2 * (nx * nx - ny * ny), // x²-y²
        else => 0.0,
    };
}

/// Real spherical harmonics for l=3
fn real_ylm3(m: i32, nx: f64, ny: f64, nz: f64) f64 {
    const pi = std.math.pi;
    return switch (m) {
        -3 => @sqrt(35.0 / (32.0 * pi)) * (3.0 * nx * nx - ny * ny) * ny,
        -2 => @sqrt(105.0 / (4.0 * pi)) * nx * ny * nz,
        -1 => @sqrt(21.0 / (32.0 * pi)) * ny * (5.0 * nz * nz - 1.0),
        0 => @sqrt(7.0 / (16.0 * pi)) * (5.0 * nz * nz * nz - 3.0 * nz),
        1 => @sqrt(21.0 / (32.0 * pi)) * nx * (5.0 * nz * nz - 1.0),
        2 => @sqrt(105.0 / (16.0 * pi)) * (nx * nx - ny * ny) * nz,
        3 => @sqrt(35.0 / (32.0 * pi)) * (nx * nx - 3.0 * ny * ny) * nx,
        else => 0.0,
    };
}

/// Real spherical harmonics for l=4
fn real_ylm4(m: i32, nx: f64, ny: f64, nz: f64) f64 {
    const pi = std.math.pi;
    const nz2 = nz * nz;
    const nx2 = nx * nx;
    const ny2 = ny * ny;
    return switch (m) {
        -4 => @sqrt(315.0 / (256.0 * pi)) * 4.0 * nx * ny * (nx2 - ny2),
        -3 => @sqrt(315.0 / (32.0 * pi)) * ny * nz * (3.0 * nx2 - ny2),
        -2 => @sqrt(45.0 / (64.0 * pi)) * 2.0 * nx * ny * (7.0 * nz2 - 1.0),
        -1 => @sqrt(45.0 / (32.0 * pi)) * ny * nz * (7.0 * nz2 - 3.0),
        0 => @sqrt(9.0 / (256.0 * pi)) * (35.0 * nz2 * nz2 - 30.0 * nz2 + 3.0),
        1 => @sqrt(45.0 / (32.0 * pi)) * nx * nz * (7.0 * nz2 - 3.0),
        2 => @sqrt(45.0 / (64.0 * pi)) * (nx2 - ny2) * (7.0 * nz2 - 1.0),
        3 => @sqrt(315.0 / (32.0 * pi)) * nx * nz * (nx2 - 3.0 * ny2),
        4 => @sqrt(315.0 / (256.0 * pi)) * (nx2 * nx2 - 6.0 * nx2 * ny2 + ny2 * ny2),
        else => 0.0,
    };
}

/// Return number of m values for given l: 2l+1
pub fn num_m_values(l: i32) usize {
    return @intCast(2 * l + 1);
}

/// Pre-computed lookup table for radial_projector as a function of |g|.
/// Uses linear interpolation for O(1) evaluation instead of O(N_r) per call.
pub const RadialTable = struct {
    values: []f64,
    g_max: f64,
    dg: f64,
    n_points: usize,

    /// Build a lookup table for radial_projector(beta, r, rab, l, g)
    /// over g in [0, g_max] with n_points uniformly spaced points.
    pub fn init(
        alloc: std.mem.Allocator,
        beta: []const f64,
        r: []const f64,
        rab: []const f64,
        l: i32,
        g_max_val: f64,
        n_points: usize,
    ) !RadialTable {
        const n = @max(n_points, 2);
        const values = try alloc.alloc(f64, n);
        const dg = g_max_val / @as(f64, @floatFromInt(n - 1));
        for (0..n) |i| {
            const g = dg * @as(f64, @floatFromInt(i));
            values[i] = radial_projector(beta, r, rab, l, g);
        }
        return .{ .values = values, .g_max = g_max_val, .dg = dg, .n_points = n };
    }

    pub fn deinit(self: *RadialTable, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
        self.* = undefined;
    }

    /// Evaluate using linear interpolation. O(1) per call.
    pub fn eval(self: *const RadialTable, g: f64) f64 {
        if (g <= 0.0) return self.values[0];
        if (g >= self.g_max) return 0.0;
        const x = g / self.dg;
        const i = @as(usize, @intFromFloat(x));
        if (i + 1 >= self.n_points) return self.values[self.n_points - 1];
        const t = x - @as(f64, @floatFromInt(i));
        return self.values[i] * (1.0 - t) + self.values[i + 1] * t;
    }

    /// Numerical derivative dR/dg.
    pub fn eval_deriv(self: *const RadialTable, g: f64) f64 {
        if (g < self.dg) {
            return (self.eval(self.dg) - self.eval(0.0)) / self.dg;
        }
        return (self.eval(g + self.dg) - self.eval(g - self.dg)) / (2.0 * self.dg);
    }
};

/// Collection of radial tables for all beta projectors of a species.
pub const RadialTableSet = struct {
    tables: []RadialTable,

    pub fn init(
        alloc: std.mem.Allocator,
        upf_beta: []const @import("pseudopotential.zig").Beta,
        r: []const f64,
        rab: []const f64,
        g_max: f64,
    ) !RadialTableSet {
        const n_table: usize = 2048;
        const tables = try alloc.alloc(RadialTable, upf_beta.len);
        errdefer {
            for (tables) |*t| t.deinit(alloc);
            alloc.free(tables);
        }
        for (upf_beta, 0..) |beta, i| {
            tables[i] = try RadialTable.init(
                alloc,
                beta.values,
                r,
                rab,
                beta.l orelse 0,
                g_max,
                n_table,
            );
        }
        return .{ .tables = tables };
    }

    pub fn deinit(self: *RadialTableSet, alloc: std.mem.Allocator) void {
        for (self.tables) |*t| t.deinit(alloc);
        if (self.tables.len > 0) alloc.free(self.tables);
        self.* = undefined;
    }
};
