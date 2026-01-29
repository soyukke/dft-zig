const std = @import("std");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("grid.zig");
const math = @import("../math/math.zig");
const xc = @import("../xc/xc.zig");

pub const Grid = grid_mod.Grid;

const realToReciprocal = fft_grid.realToReciprocal;
const reciprocalToReal = fft_grid.reciprocalToReal;

pub const XcFields = struct {
    vxc: []f64,
    exc: []f64,
};

pub const XcFieldsSpin = struct {
    vxc_up: []f64,
    vxc_down: []f64,
    exc: []f64,

    pub fn deinit(self: *XcFieldsSpin, alloc: std.mem.Allocator) void {
        alloc.free(self.vxc_up);
        alloc.free(self.vxc_down);
        alloc.free(self.exc);
    }
};

pub const Gradient = struct {
    x: []f64,
    y: []f64,
    z: []f64,

    pub fn deinit(self: *Gradient, alloc: std.mem.Allocator) void {
        if (self.x.len > 0) alloc.free(self.x);
        if (self.y.len > 0) alloc.free(self.y);
        if (self.z.len > 0) alloc.free(self.z);
    }
};

pub fn gradientFromReal(
    alloc: std.mem.Allocator,
    grid: Grid,
    values: []const f64,
    use_rfft: bool,
) !Gradient {
    const values_g = try realToReciprocal(alloc, grid, values, use_rfft);
    defer alloc.free(values_g);

    const total = grid.count();
    const gx_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gx_g);
    const gy_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gy_g);
    const gz_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gz_g);

    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);
    const i_unit = math.complex.init(0.0, 1.0);
    var idx: usize = 0;
    var l: usize = 0;
    while (l < grid.nz) : (l += 1) {
        var k: usize = 0;
        while (k < grid.ny) : (k += 1) {
            var h: usize = 0;
            while (h < grid.nx) : (h += 1) {
                const gh = grid.min_h + @as(i32, @intCast(h));
                const gk = grid.min_k + @as(i32, @intCast(k));
                const gl = grid.min_l + @as(i32, @intCast(l));
                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const i_rho = math.complex.mul(values_g[idx], i_unit);
                gx_g[idx] = math.complex.scale(i_rho, gvec.x);
                gy_g[idx] = math.complex.scale(i_rho, gvec.y);
                gz_g[idx] = math.complex.scale(i_rho, gvec.z);
                idx += 1;
            }
        }
    }

    const gx = try reciprocalToReal(alloc, grid, gx_g);
    const gy = try reciprocalToReal(alloc, grid, gy_g);
    const gz = try reciprocalToReal(alloc, grid, gz_g);
    alloc.free(gx_g);
    alloc.free(gy_g);
    alloc.free(gz_g);
    return .{ .x = gx, .y = gy, .z = gz };
}

pub fn divergenceFromReal(
    alloc: std.mem.Allocator,
    grid: Grid,
    bx: []const f64,
    by: []const f64,
    bz: []const f64,
    use_rfft: bool,
) ![]f64 {
    const bx_g = try realToReciprocal(alloc, grid, bx, use_rfft);
    defer alloc.free(bx_g);
    const by_g = try realToReciprocal(alloc, grid, by, use_rfft);
    defer alloc.free(by_g);
    const bz_g = try realToReciprocal(alloc, grid, bz, use_rfft);
    defer alloc.free(bz_g);

    const total = grid.count();
    const div_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(div_g);
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);
    const i_unit = math.complex.init(0.0, 1.0);

    var idx: usize = 0;
    var l: usize = 0;
    while (l < grid.nz) : (l += 1) {
        var k: usize = 0;
        while (k < grid.ny) : (k += 1) {
            var h: usize = 0;
            while (h < grid.nx) : (h += 1) {
                const gh = grid.min_h + @as(i32, @intCast(h));
                const gk = grid.min_k + @as(i32, @intCast(k));
                const gl = grid.min_l + @as(i32, @intCast(l));
                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const sum = math.complex.add(
                    math.complex.add(math.complex.scale(bx_g[idx], gvec.x), math.complex.scale(by_g[idx], gvec.y)),
                    math.complex.scale(bz_g[idx], gvec.z),
                );
                div_g[idx] = math.complex.mul(sum, i_unit);
                idx += 1;
            }
        }
    }

    const div = try reciprocalToReal(alloc, grid, div_g);
    alloc.free(div_g);
    return div;
}

pub fn computeXcFields(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []const f64,
    rho_core: ?[]const f64,
    use_rfft: bool,
    xc_func: xc.Functional,
) !XcFields {
    const total = grid.count();
    const vxc = try alloc.alloc(f64, total);
    errdefer alloc.free(vxc);
    const exc = try alloc.alloc(f64, total);
    errdefer alloc.free(exc);

    if (xc_func == .lda_pz) {
        for (rho, 0..) |value, i| {
            const core = if (rho_core) |rc| rc[i] else 0.0;
            const n = value + core;
            const eval = xc.evalPoint(.lda_pz, n, 0.0);
            vxc[i] = eval.df_dn;
            exc[i] = eval.f;
        }
        return .{ .vxc = vxc, .exc = exc };
    }

    var rho_total: ?[]f64 = null;
    if (rho_core) |rc| {
        const total_rho = try alloc.alloc(f64, total);
        for (rho, 0..) |value, i| {
            total_rho[i] = value + rc[i];
        }
        rho_total = total_rho;
    }
    defer if (rho_total) |values| alloc.free(values);
    const density = rho_total orelse rho;

    var grad = try gradientFromReal(alloc, grid, density, use_rfft);
    defer grad.deinit(alloc);

    const bx = try alloc.alloc(f64, total);
    errdefer alloc.free(bx);
    const by = try alloc.alloc(f64, total);
    errdefer alloc.free(by);
    const bz = try alloc.alloc(f64, total);
    errdefer alloc.free(bz);

    var i: usize = 0;
    while (i < total) : (i += 1) {
        const gx = grad.x[i];
        const gy = grad.y[i];
        const gz = grad.z[i];
        const g2 = gx * gx + gy * gy + gz * gz;
        const eval = xc.evalPoint(xc_func, density[i], g2);
        vxc[i] = eval.df_dn;
        exc[i] = eval.f;
        const coeff = eval.df_dg2;
        bx[i] = coeff * gx;
        by[i] = coeff * gy;
        bz[i] = coeff * gz;
    }

    const div = try divergenceFromReal(alloc, grid, bx, by, bz, use_rfft);
    defer alloc.free(div);
    alloc.free(bx);
    alloc.free(by);
    alloc.free(bz);

    for (vxc, 0..) |*value, idx| {
        value.* -= 2.0 * div[idx];
    }

    return .{ .vxc = vxc, .exc = exc };
}

/// Compute spin-polarized XC fields (V_xc_up, V_xc_down, exc).
/// NLCC core density is split equally between up and down channels.
pub fn computeXcFieldsSpin(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_core: ?[]const f64,
    use_rfft: bool,
    xc_func: xc.Functional,
) !XcFieldsSpin {
    const total = grid.count();
    const vxc_up = try alloc.alloc(f64, total);
    errdefer alloc.free(vxc_up);
    const vxc_down = try alloc.alloc(f64, total);
    errdefer alloc.free(vxc_down);
    const exc = try alloc.alloc(f64, total);
    errdefer alloc.free(exc);

    if (xc_func == .lda_pz) {
        for (0..total) |i| {
            const core_half = if (rho_core) |rc| rc[i] / 2.0 else 0.0;
            const n_up = rho_up[i] + core_half;
            const n_down = rho_down[i] + core_half;
            const eval = xc.evalPointSpin(.lda_pz, n_up, n_down, 0.0, 0.0, 0.0);
            vxc_up[i] = eval.df_dn_up;
            vxc_down[i] = eval.df_dn_down;
            exc[i] = eval.f;
        }
        return .{ .vxc_up = vxc_up, .vxc_down = vxc_down, .exc = exc };
    }

    // GGA: need gradients for each spin channel
    // Build total density with core
    const density_up = try alloc.alloc(f64, total);
    defer alloc.free(density_up);
    const density_down = try alloc.alloc(f64, total);
    defer alloc.free(density_down);
    for (0..total) |i| {
        const core_half = if (rho_core) |rc| rc[i] / 2.0 else 0.0;
        density_up[i] = rho_up[i] + core_half;
        density_down[i] = rho_down[i] + core_half;
    }

    // Compute gradients for up and down channels
    var grad_up = try gradientFromReal(alloc, grid, density_up, use_rfft);
    defer grad_up.deinit(alloc);
    var grad_down = try gradientFromReal(alloc, grid, density_down, use_rfft);
    defer grad_down.deinit(alloc);

    // B vectors for divergence correction (6 components: bx_uu, by_uu, bz_uu, bx_dd, by_dd, bz_dd, + cross terms)
    // For PBE spin: V_xc_sigma -= 2 * div(df/dg2_ss * grad_sigma + df/dg2_ud * grad_sigma')
    const bx_up = try alloc.alloc(f64, total);
    errdefer alloc.free(bx_up);
    const by_up = try alloc.alloc(f64, total);
    errdefer alloc.free(by_up);
    const bz_up = try alloc.alloc(f64, total);
    errdefer alloc.free(bz_up);
    const bx_down = try alloc.alloc(f64, total);
    errdefer alloc.free(bx_down);
    const by_down = try alloc.alloc(f64, total);
    errdefer alloc.free(by_down);
    const bz_down = try alloc.alloc(f64, total);
    errdefer alloc.free(bz_down);

    for (0..total) |i| {
        const gux = grad_up.x[i];
        const guy = grad_up.y[i];
        const guz = grad_up.z[i];
        const gdx = grad_down.x[i];
        const gdy = grad_down.y[i];
        const gdz = grad_down.z[i];

        const g2_uu = gux * gux + guy * guy + guz * guz;
        const g2_dd = gdx * gdx + gdy * gdy + gdz * gdz;
        const g2_ud = gux * gdx + guy * gdy + guz * gdz;

        const eval = xc.evalPointSpin(xc_func, density_up[i], density_down[i], g2_uu, g2_dd, g2_ud);
        vxc_up[i] = eval.df_dn_up;
        vxc_down[i] = eval.df_dn_down;
        exc[i] = eval.f;

        // B_up = 2*df/dg2_uu * grad_up + df/dg2_ud * grad_down
        bx_up[i] = 2.0 * eval.df_dg2_uu * gux + eval.df_dg2_ud * gdx;
        by_up[i] = 2.0 * eval.df_dg2_uu * guy + eval.df_dg2_ud * gdy;
        bz_up[i] = 2.0 * eval.df_dg2_uu * guz + eval.df_dg2_ud * gdz;

        // B_down = 2*df/dg2_dd * grad_down + df/dg2_ud * grad_up
        bx_down[i] = 2.0 * eval.df_dg2_dd * gdx + eval.df_dg2_ud * gux;
        by_down[i] = 2.0 * eval.df_dg2_dd * gdy + eval.df_dg2_ud * guy;
        bz_down[i] = 2.0 * eval.df_dg2_dd * gdz + eval.df_dg2_ud * guz;
    }

    // Divergence corrections
    const div_up = try divergenceFromReal(alloc, grid, bx_up, by_up, bz_up, use_rfft);
    defer alloc.free(div_up);
    alloc.free(bx_up);
    alloc.free(by_up);
    alloc.free(bz_up);

    const div_down = try divergenceFromReal(alloc, grid, bx_down, by_down, bz_down, use_rfft);
    defer alloc.free(div_down);
    alloc.free(bx_down);
    alloc.free(by_down);
    alloc.free(bz_down);

    for (0..total) |i| {
        vxc_up[i] -= div_up[i];
        vxc_down[i] -= div_down[i];
    }

    return .{ .vxc_up = vxc_up, .vxc_down = vxc_down, .exc = exc };
}
