const std = @import("std");

const fft = @import("../fft/fft.zig");
const local_orbital_potential = @import("local_orbital_potential.zig");
const math = @import("../math/math.zig");
const xc = @import("../xc/xc.zig");

pub const HartreeXcResult = struct {
    hartree: []f64,
    vxc: []f64,
    exc: []f64,
    energy_hartree: f64,
    energy_xc: f64,
    energy_vxc_rho: f64,

    pub fn deinit(self: *HartreeXcResult, alloc: std.mem.Allocator) void {
        if (self.hartree.len > 0) alloc.free(self.hartree);
        if (self.vxc.len > 0) alloc.free(self.vxc);
        if (self.exc.len > 0) alloc.free(self.exc);
        self.* = undefined;
    }
};

pub const LocalPotentialResult = struct {
    values: []f64,
    energy_hartree: f64,
    energy_xc: f64,
    energy_vxc_rho: f64,

    pub fn deinit(self: *LocalPotentialResult, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
        self.* = undefined;
    }
};

pub fn buildHartreeXc(
    alloc: std.mem.Allocator,
    grid: local_orbital_potential.PotentialGrid,
    density: []const f64,
    xc_func: xc.Functional,
) !HartreeXcResult {
    const count = grid.count();
    if (density.len != count) return error.InvalidGrid;

    const vxc = try alloc.alloc(f64, count);
    errdefer alloc.free(vxc);
    const exc = try alloc.alloc(f64, count);
    errdefer alloc.free(exc);
    for (density, 0..) |value, idx| {
        const eval = xc.evalPoint(xc_func, value, 0.0);
        vxc[idx] = eval.df_dn;
        exc[idx] = eval.f;
    }

    const hartree = try computeHartreePotential(alloc, grid, density);
    errdefer alloc.free(hartree);

    const dv = try grid.weight();
    var energy_h: f64 = 0.0;
    var energy_xc: f64 = 0.0;
    var energy_vxc_rho: f64 = 0.0;
    for (density, 0..) |value, idx| {
        energy_h += value * hartree[idx] * dv;
        energy_xc += exc[idx] * dv;
        energy_vxc_rho += vxc[idx] * value * dv;
    }
    energy_h *= 0.5;

    return .{
        .hartree = hartree,
        .vxc = vxc,
        .exc = exc,
        .energy_hartree = energy_h,
        .energy_xc = energy_xc,
        .energy_vxc_rho = energy_vxc_rho,
    };
}

pub fn buildLocalPotential(
    alloc: std.mem.Allocator,
    ionic: ?[]const f64,
    hartree: []const f64,
    vxc: []const f64,
) ![]f64 {
    if (hartree.len != vxc.len) return error.InvalidGrid;
    if (ionic) |ion| {
        if (ion.len != hartree.len) return error.InvalidGrid;
    }
    const values = try alloc.alloc(f64, hartree.len);
    for (hartree, 0..) |value, idx| {
        var total = value + vxc[idx];
        if (ionic) |ion| {
            total += ion[idx];
        }
        values[idx] = total;
    }
    return values;
}

pub fn buildLocalPotentialGrid(
    alloc: std.mem.Allocator,
    grid: local_orbital_potential.PotentialGrid,
    density: []const f64,
    xc_func: xc.Functional,
    ionic: ?[]const f64,
) !LocalPotentialResult {
    var fields = try buildHartreeXc(alloc, grid, density, xc_func);
    defer fields.deinit(alloc);
    const values = try buildLocalPotential(alloc, ionic, fields.hartree, fields.vxc);
    return .{
        .values = values,
        .energy_hartree = fields.energy_hartree,
        .energy_xc = fields.energy_xc,
        .energy_vxc_rho = fields.energy_vxc_rho,
    };
}

fn computeHartreePotential(
    alloc: std.mem.Allocator,
    grid: local_orbital_potential.PotentialGrid,
    density: []const f64,
) ![]f64 {
    const total = grid.count();
    if (density.len != total) return error.InvalidGrid;

    var rho_g = try fftRealToReciprocal(alloc, grid, density);
    defer alloc.free(rho_g);

    const recip = math.reciprocal(grid.cell);
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);
    var idx: usize = 0;
    var iz: usize = 0;
    while (iz < grid.dims[2]) : (iz += 1) {
        var iy: usize = 0;
        while (iy < grid.dims[1]) : (iy += 1) {
            var ix: usize = 0;
            while (ix < grid.dims[0]) : (ix += 1) {
                const gh = indexToFreq(ix, grid.dims[0]);
                const gk = indexToFreq(iy, grid.dims[1]);
                const gl = indexToFreq(iz, grid.dims[2]);
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g2 = math.Vec3.dot(gvec, gvec);
                if (g2 > 1e-12) {
                    rho_g[idx] = math.complex.scale(rho_g[idx], 8.0 * std.math.pi / g2);
                } else {
                    rho_g[idx] = math.complex.init(0.0, 0.0);
                }
                idx += 1;
            }
        }
    }

    return fftReciprocalToReal(alloc, grid, rho_g);
}

fn fftRealToReciprocal(
    alloc: std.mem.Allocator,
    grid: local_orbital_potential.PotentialGrid,
    values: []const f64,
) ![]math.Complex {
    const total = grid.count();
    if (values.len != total) return error.InvalidGrid;
    const data = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(data);
    for (values, 0..) |v, i| {
        data[i] = math.complex.init(v, 0.0);
    }
    try fft.fft3dForwardInPlace(alloc, data, grid.dims[0], grid.dims[1], grid.dims[2]);
    const scale = 1.0 / @as(f64, @floatFromInt(total));
    for (data) |*v| {
        v.* = math.complex.scale(v.*, scale);
    }
    return data;
}

fn fftReciprocalToReal(
    alloc: std.mem.Allocator,
    grid: local_orbital_potential.PotentialGrid,
    values: []const math.Complex,
) ![]f64 {
    const total = grid.count();
    if (values.len != total) return error.InvalidGrid;
    const data = try alloc.alloc(math.Complex, total);
    defer alloc.free(data);
    const scale = @as(f64, @floatFromInt(total));
    for (values, 0..) |v, i| {
        data[i] = math.complex.scale(v, scale);
    }
    try fft.fft3dInverseInPlace(alloc, data, grid.dims[0], grid.dims[1], grid.dims[2]);
    const out = try alloc.alloc(f64, total);
    errdefer alloc.free(out);
    for (data, 0..) |v, i| {
        out[i] = v.r;
    }
    return out;
}

fn indexToFreq(index: usize, n: usize) i32 {
    const half = @as(i32, @intCast(n / 2));
    const idx = @as(i32, @intCast(index));
    return if (idx <= half) idx else idx - @as(i32, @intCast(n));
}

test "hartree potential of uniform density is zero" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 8.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 8.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 8.0 },
    );
    const dims = [3]usize{ 8, 8, 8 };
    const count = dims[0] * dims[1] * dims[2];
    const density = try alloc.alloc(f64, count);
    defer alloc.free(density);
    @memset(density, 0.5);
    const grid = local_orbital_potential.PotentialGrid{
        .cell = cell,
        .dims = dims,
        .values = &[_]f64{},
    };
    const hartree = try computeHartreePotential(alloc, grid, density);
    defer alloc.free(hartree);
    var max_abs: f64 = 0.0;
    for (hartree) |value| {
        max_abs = @max(max_abs, @abs(value));
    }
    try std.testing.expect(max_abs < 1e-8);
}

test "xc fields are uniform for uniform density" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 6.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 6.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 6.0 },
    );
    const dims = [3]usize{ 6, 6, 6 };
    const count = dims[0] * dims[1] * dims[2];
    const density = try alloc.alloc(f64, count);
    defer alloc.free(density);
    @memset(density, 0.2);
    const grid = local_orbital_potential.PotentialGrid{
        .cell = cell,
        .dims = dims,
        .values = &[_]f64{},
    };
    var fields = try buildHartreeXc(alloc, grid, density, .lda_pz);
    defer fields.deinit(alloc);
    const eval = xc.evalPoint(.lda_pz, 0.2, 0.0);
    try std.testing.expectApproxEqAbs(eval.df_dn, fields.vxc[0], 1e-12);
    var max_diff: f64 = 0.0;
    for (fields.vxc) |value| {
        max_diff = @max(max_diff, @abs(value - fields.vxc[0]));
    }
    try std.testing.expect(max_diff < 1e-12);
    const volume = local_orbital_potential.cellVolume(cell);
    try std.testing.expectApproxEqAbs(eval.f * volume, fields.energy_xc, 1e-8);
}

test "local potential combines ionic and xc" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 4.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 4.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 4.0 },
    );
    const dims = [3]usize{ 4, 4, 4 };
    const count = dims[0] * dims[1] * dims[2];
    const density = try alloc.alloc(f64, count);
    defer alloc.free(density);
    @memset(density, 0.1);
    const ionic = try alloc.alloc(f64, count);
    defer alloc.free(ionic);
    @memset(ionic, 0.05);
    const grid = local_orbital_potential.PotentialGrid{
        .cell = cell,
        .dims = dims,
        .values = &[_]f64{},
    };
    var fields = try buildHartreeXc(alloc, grid, density, .lda_pz);
    defer fields.deinit(alloc);
    const local = try buildLocalPotential(alloc, ionic, fields.hartree, fields.vxc);
    defer alloc.free(local);
    const expected = ionic[0] + fields.vxc[0] + fields.hartree[0];
    try std.testing.expectApproxEqAbs(expected, local[0], 1e-12);
}
