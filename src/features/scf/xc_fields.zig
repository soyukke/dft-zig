const std = @import("std");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const gvec_iter = @import("gvec_iter.zig");
const math = @import("../math/math.zig");
const xc = @import("../xc/xc.zig");

pub const Grid = grid_mod.Grid;

const real_to_reciprocal = fft_grid.real_to_reciprocal;
const reciprocal_to_real = fft_grid.reciprocal_to_real;

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

fn alloc_gradient(alloc: std.mem.Allocator, total: usize) !Gradient {
    const x = try alloc.alloc(f64, total);
    errdefer alloc.free(x);
    const y = try alloc.alloc(f64, total);
    errdefer alloc.free(y);
    const z = try alloc.alloc(f64, total);
    errdefer alloc.free(z);
    return .{ .x = x, .y = y, .z = z };
}

fn alloc_density_with_core(
    alloc: std.mem.Allocator,
    rho: []const f64,
    rho_core: ?[]const f64,
) !?[]f64 {
    const core = rho_core orelse return null;
    const density = try alloc.alloc(f64, rho.len);
    for (rho, 0..) |value, i| {
        density[i] = value + core[i];
    }
    return density;
}

fn compute_lda_xc_fields(
    alloc: std.mem.Allocator,
    rho: []const f64,
    rho_core: ?[]const f64,
) !XcFields {
    const total = rho.len;
    const vxc = try alloc.alloc(f64, total);
    errdefer alloc.free(vxc);
    const exc = try alloc.alloc(f64, total);
    errdefer alloc.free(exc);

    for (rho, 0..) |value, i| {
        const core = if (rho_core) |rc| rc[i] else 0.0;
        const n = value + core;
        const eval = xc.eval_point(.lda_pz, n, 0.0);
        vxc[i] = eval.df_dn;
        exc[i] = eval.f;
    }
    return .{ .vxc = vxc, .exc = exc };
}

fn fill_gga_xc_fields(
    density: []const f64,
    grad: Gradient,
    xc_func: xc.Functional,
    vxc: []f64,
    exc: []f64,
    b: *Gradient,
) void {
    for (density, 0..) |value, i| {
        const gx = grad.x[i];
        const gy = grad.y[i];
        const gz = grad.z[i];
        const g2 = gx * gx + gy * gy + gz * gz;
        const eval = xc.eval_point(xc_func, value, g2);
        vxc[i] = eval.df_dn;
        exc[i] = eval.f;
        b.x[i] = eval.df_dg2 * gx;
        b.y[i] = eval.df_dg2 * gy;
        b.z[i] = eval.df_dg2 * gz;
    }
}

fn compute_gga_xc_fields(
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

    const rho_total = try alloc_density_with_core(alloc, rho, rho_core);
    defer if (rho_total) |values| alloc.free(values);

    const density = rho_total orelse rho;

    var grad = try gradient_from_real(alloc, grid, density, use_rfft);
    defer grad.deinit(alloc);

    var b = try alloc_gradient(alloc, total);
    defer b.deinit(alloc);

    fill_gga_xc_fields(density, grad, xc_func, vxc, exc, &b);

    const div = try divergence_from_real(alloc, grid, b.x, b.y, b.z, use_rfft);
    defer alloc.free(div);

    for (vxc, 0..) |*value, idx| {
        value.* -= 2.0 * div[idx];
    }
    return .{ .vxc = vxc, .exc = exc };
}

fn compute_lda_xc_fields_spin(
    alloc: std.mem.Allocator,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_core: ?[]const f64,
) !XcFieldsSpin {
    const total = rho_up.len;
    const vxc_up = try alloc.alloc(f64, total);
    errdefer alloc.free(vxc_up);
    const vxc_down = try alloc.alloc(f64, total);
    errdefer alloc.free(vxc_down);
    const exc = try alloc.alloc(f64, total);
    errdefer alloc.free(exc);

    for (0..total) |i| {
        const core_half = if (rho_core) |rc| rc[i] / 2.0 else 0.0;
        const n_up = rho_up[i] + core_half;
        const n_down = rho_down[i] + core_half;
        const eval = xc.eval_point_spin(.lda_pz, n_up, n_down, 0.0, 0.0, 0.0);
        vxc_up[i] = eval.df_dn_up;
        vxc_down[i] = eval.df_dn_down;
        exc[i] = eval.f;
    }
    return .{ .vxc_up = vxc_up, .vxc_down = vxc_down, .exc = exc };
}

const SpinDensity = struct {
    up: []f64,
    down: []f64,

    fn deinit(self: *SpinDensity, alloc: std.mem.Allocator) void {
        alloc.free(self.up);
        alloc.free(self.down);
    }
};

fn build_spin_density_with_core(
    alloc: std.mem.Allocator,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_core: ?[]const f64,
) !SpinDensity {
    const total = rho_up.len;
    const density_up = try alloc.alloc(f64, total);
    errdefer alloc.free(density_up);
    const density_down = try alloc.alloc(f64, total);
    errdefer alloc.free(density_down);

    for (0..total) |i| {
        const core_half = if (rho_core) |rc| rc[i] / 2.0 else 0.0;
        density_up[i] = rho_up[i] + core_half;
        density_down[i] = rho_down[i] + core_half;
    }
    return .{ .up = density_up, .down = density_down };
}

fn fill_gga_spin_xc_fields(
    density: SpinDensity,
    grad_up: Gradient,
    grad_down: Gradient,
    xc_func: xc.Functional,
    vxc_up: []f64,
    vxc_down: []f64,
    exc: []f64,
    b_up: *Gradient,
    b_down: *Gradient,
) void {
    for (0..vxc_up.len) |i| {
        const gux = grad_up.x[i];
        const guy = grad_up.y[i];
        const guz = grad_up.z[i];
        const gdx = grad_down.x[i];
        const gdy = grad_down.y[i];
        const gdz = grad_down.z[i];
        const g2_uu = gux * gux + guy * guy + guz * guz;
        const g2_dd = gdx * gdx + gdy * gdy + gdz * gdz;
        const g2_ud = gux * gdx + guy * gdy + guz * gdz;
        const eval = xc.eval_point_spin(
            xc_func,
            density.up[i],
            density.down[i],
            g2_uu,
            g2_dd,
            g2_ud,
        );

        vxc_up[i] = eval.df_dn_up;
        vxc_down[i] = eval.df_dn_down;
        exc[i] = eval.f;
        b_up.x[i] = 2.0 * eval.df_dg2_uu * gux + eval.df_dg2_ud * gdx;
        b_up.y[i] = 2.0 * eval.df_dg2_uu * guy + eval.df_dg2_ud * gdy;
        b_up.z[i] = 2.0 * eval.df_dg2_uu * guz + eval.df_dg2_ud * gdz;
        b_down.x[i] = 2.0 * eval.df_dg2_dd * gdx + eval.df_dg2_ud * gux;
        b_down.y[i] = 2.0 * eval.df_dg2_dd * gdy + eval.df_dg2_ud * guy;
        b_down.z[i] = 2.0 * eval.df_dg2_dd * gdz + eval.df_dg2_ud * guz;
    }
}

fn compute_gga_xc_fields_spin(
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

    var density = try build_spin_density_with_core(alloc, rho_up, rho_down, rho_core);
    defer density.deinit(alloc);

    var grad_up = try gradient_from_real(alloc, grid, density.up, use_rfft);
    defer grad_up.deinit(alloc);

    var grad_down = try gradient_from_real(alloc, grid, density.down, use_rfft);
    defer grad_down.deinit(alloc);

    var b_up = try alloc_gradient(alloc, total);
    defer b_up.deinit(alloc);

    var b_down = try alloc_gradient(alloc, total);
    defer b_down.deinit(alloc);

    fill_gga_spin_xc_fields(
        density,
        grad_up,
        grad_down,
        xc_func,
        vxc_up,
        vxc_down,
        exc,
        &b_up,
        &b_down,
    );

    const div_up = try divergence_from_real(alloc, grid, b_up.x, b_up.y, b_up.z, use_rfft);
    defer alloc.free(div_up);

    const div_down = try divergence_from_real(
        alloc,
        grid,
        b_down.x,
        b_down.y,
        b_down.z,
        use_rfft,
    );
    defer alloc.free(div_down);

    for (0..total) |i| {
        vxc_up[i] -= div_up[i];
        vxc_down[i] -= div_down[i];
    }
    return .{ .vxc_up = vxc_up, .vxc_down = vxc_down, .exc = exc };
}

pub fn gradient_from_real(
    alloc: std.mem.Allocator,
    grid: Grid,
    values: []const f64,
    use_rfft: bool,
) !Gradient {
    const values_g = try real_to_reciprocal(alloc, grid, values, use_rfft);
    defer alloc.free(values_g);

    const total = grid.count();
    const gx_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gx_g);
    const gy_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gy_g);
    const gz_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gz_g);

    const i_unit = math.complex.init(0.0, 1.0);
    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        const i_rho = math.complex.mul(values_g[g.idx], i_unit);
        gx_g[g.idx] = math.complex.scale(i_rho, g.gvec.x);
        gy_g[g.idx] = math.complex.scale(i_rho, g.gvec.y);
        gz_g[g.idx] = math.complex.scale(i_rho, g.gvec.z);
    }

    const gx = try reciprocal_to_real(alloc, grid, gx_g);
    const gy = try reciprocal_to_real(alloc, grid, gy_g);
    const gz = try reciprocal_to_real(alloc, grid, gz_g);
    alloc.free(gx_g);
    alloc.free(gy_g);
    alloc.free(gz_g);
    return .{ .x = gx, .y = gy, .z = gz };
}

pub fn divergence_from_real(
    alloc: std.mem.Allocator,
    grid: Grid,
    bx: []const f64,
    by: []const f64,
    bz: []const f64,
    use_rfft: bool,
) ![]f64 {
    const bx_g = try real_to_reciprocal(alloc, grid, bx, use_rfft);
    defer alloc.free(bx_g);

    const by_g = try real_to_reciprocal(alloc, grid, by, use_rfft);
    defer alloc.free(by_g);

    const bz_g = try real_to_reciprocal(alloc, grid, bz, use_rfft);
    defer alloc.free(bz_g);

    const total = grid.count();
    const div_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(div_g);
    const i_unit = math.complex.init(0.0, 1.0);

    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        const bx_term = math.complex.scale(bx_g[g.idx], g.gvec.x);
        const by_term = math.complex.scale(by_g[g.idx], g.gvec.y);
        const bz_term = math.complex.scale(bz_g[g.idx], g.gvec.z);
        const sum = math.complex.add(math.complex.add(bx_term, by_term), bz_term);
        div_g[g.idx] = math.complex.mul(sum, i_unit);
    }

    const div = try reciprocal_to_real(alloc, grid, div_g);
    alloc.free(div_g);
    return div;
}

pub fn compute_xc_fields(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []const f64,
    rho_core: ?[]const f64,
    use_rfft: bool,
    xc_func: xc.Functional,
) !XcFields {
    if (xc_func == .lda_pz) return compute_lda_xc_fields(alloc, rho, rho_core);
    return compute_gga_xc_fields(alloc, grid, rho, rho_core, use_rfft, xc_func);
}

/// Compute spin-polarized XC fields (V_xc_up, V_xc_down, exc).
/// NLCC core density is split equally between up and down channels.
pub fn compute_xc_fields_spin(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_core: ?[]const f64,
    use_rfft: bool,
    xc_func: xc.Functional,
) !XcFieldsSpin {
    if (xc_func == .lda_pz) return compute_lda_xc_fields_spin(alloc, rho_up, rho_down, rho_core);
    return compute_gga_xc_fields_spin(alloc, grid, rho_up, rho_down, rho_core, use_rfft, xc_func);
}
