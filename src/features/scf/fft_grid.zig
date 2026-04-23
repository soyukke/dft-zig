const std = @import("std");
const fft = @import("../fft/fft.zig");
const math = @import("../math/math.zig");
const grid_mod = @import("pw_grid.zig");

pub const Grid = grid_mod.Grid;

/// Convert real grid to reciprocal grid using direct DFT.
pub fn realToReciprocal(
    alloc: std.mem.Allocator,
    grid: Grid,
    values: []const f64,
    use_rfft: bool,
) ![]math.Complex {
    if (use_rfft and grid.nx % 2 == 0) {
        return try fftRealToReciprocalRfft(alloc, grid, values);
    }
    return try fftRealToReciprocal(alloc, grid, values);
}

/// Convert reciprocal grid to real grid.
pub fn reciprocalToReal(alloc: std.mem.Allocator, grid: Grid, values: []math.Complex) ![]f64 {
    // FFT supports arbitrary sizes via Bluestein's algorithm
    return try fftReciprocalToReal(alloc, grid, values);
}

/// Convert real grid to reciprocal grid using direct DFT.
fn dftRealToReciprocalDirect(
    alloc: std.mem.Allocator,
    grid: Grid,
    values: []const f64,
) ![]math.Complex {
    const total = grid.count();
    if (values.len != total) return error.InvalidGrid;
    const out = try alloc.alloc(math.Complex, total);
    const scale = 1.0 / @as(f64, @floatFromInt(total));

    const a1 = grid.cell.row(0);
    const a2 = grid.cell.row(1);
    const a3 = grid.cell.row(2);
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

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
                const gh_f = @as(f64, @floatFromInt(gh));
                const gk_f = @as(f64, @floatFromInt(gk));
                const gl_f = @as(f64, @floatFromInt(gl));
                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, gh_f), math.Vec3.scale(b2, gk_f)),
                    math.Vec3.scale(b3, gl_f),
                );

                var sum = math.complex.init(0.0, 0.0);
                var iz: usize = 0;
                var ridx: usize = 0;
                const nx_f = @as(f64, @floatFromInt(grid.nx));
                const ny_f = @as(f64, @floatFromInt(grid.ny));
                const nz_f = @as(f64, @floatFromInt(grid.nz));
                while (iz < grid.nz) : (iz += 1) {
                    const fz = @as(f64, @floatFromInt(iz)) / nz_f;
                    var iy: usize = 0;
                    while (iy < grid.ny) : (iy += 1) {
                        const fy = @as(f64, @floatFromInt(iy)) / ny_f;
                        var ix: usize = 0;
                        while (ix < grid.nx) : (ix += 1) {
                            const fx = @as(f64, @floatFromInt(ix)) / nx_f;
                            const rvec = math.Vec3.add(
                                math.Vec3.add(math.Vec3.scale(a1, fx), math.Vec3.scale(a2, fy)),
                                math.Vec3.scale(a3, fz),
                            );
                            const phase = math.complex.expi(-math.Vec3.dot(gvec, rvec));
                            sum = math.complex.add(sum, math.complex.scale(phase, values[ridx]));
                            ridx += 1;
                        }
                    }
                }
                out[idx] = math.complex.scale(sum, scale);
                idx += 1;
            }
        }
    }
    return out;
}

/// Convert real grid to reciprocal grid using RFFT (faster for real data).
/// The output is expanded to full size using Hermitian symmetry: F[-k] = conj(F[k])
fn fftRealToReciprocalRfft(
    alloc: std.mem.Allocator,
    grid: Grid,
    values: []const f64,
) ![]math.Complex {
    const nx = grid.nx;
    const ny = grid.ny;
    const nz = grid.nz;
    const total = grid.count();
    if (values.len != total) return error.InvalidGrid;
    if (nx % 2 != 0) return error.InvalidGrid; // RFFT requires even nx

    // Perform RFFT
    var rfft_plan = try fft.RealFft3dPlan.init(alloc, nx, ny, nz);
    defer rfft_plan.deinit(alloc);

    const nx_c = nx / 2 + 1;
    const rfft_size = nx_c * ny * nz;
    const rfft_out = try alloc.alloc(fft.LibComplex, rfft_size);
    defer alloc.free(rfft_out);

    rfft_plan.forward(values, rfft_out);

    // Expand to full size using Hermitian symmetry
    const data = try alloc.alloc(math.Complex, total);
    defer alloc.free(data);

    // Copy RFFT output and expand using symmetry
    var z: usize = 0;
    while (z < nz) : (z += 1) {
        var y: usize = 0;
        while (y < ny) : (y += 1) {
            // Copy first half (0 to nx/2) directly from RFFT output
            var x: usize = 0;
            while (x < nx_c) : (x += 1) {
                const rfft_idx = x + nx_c * (y + ny * z);
                const fft_idx = x + nx * (y + ny * z);
                data[fft_idx] = math.complex.init(rfft_out[rfft_idx].re, rfft_out[rfft_idx].im);
            }
            // Second half (nx/2+1 to nx-1) from Hermitian symmetry:
            //   F[nx-x, ny-y, nz-z] = conj(F[x, y, z])
            x = nx_c;
            while (x < nx) : (x += 1) {
                const sym_x = nx - x; // Maps to 1, 2, ..., nx/2-1
                const sym_y = if (y == 0) 0 else ny - y;
                const sym_z = if (z == 0) 0 else nz - z;
                const rfft_idx = sym_x + nx_c * (sym_y + ny * sym_z);
                const fft_idx = x + nx * (y + ny * z);
                // Conjugate for Hermitian symmetry
                data[fft_idx] = math.complex.init(rfft_out[rfft_idx].re, -rfft_out[rfft_idx].im);
            }
        }
    }

    // Scale
    const scale = 1.0 / @as(f64, @floatFromInt(total));
    for (data) |*v| {
        v.* = math.complex.scale(v.*, scale);
    }

    // Remap indices (same as original)
    const out = try alloc.alloc(math.Complex, total);
    var idx: usize = 0;
    z = 0;
    while (z < nz) : (z += 1) {
        var y_idx: usize = 0;
        while (y_idx < ny) : (y_idx += 1) {
            var x_idx: usize = 0;
            while (x_idx < nx) : (x_idx += 1) {
                const fh = indexToFreq(x_idx, nx);
                const fk = indexToFreq(y_idx, ny);
                const fl = indexToFreq(z, nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                const out_idx = th + nx * (tk + ny * tl);
                out[out_idx] = data[idx];
                idx += 1;
            }
        }
    }
    return out;
}

/// Convert real grid to reciprocal grid using radix-2 FFT.
fn fftRealToReciprocal(alloc: std.mem.Allocator, grid: Grid, values: []const f64) ![]math.Complex {
    const total = grid.count();
    if (values.len != total) return error.InvalidGrid;

    const data = try alloc.alloc(math.Complex, total);
    defer alloc.free(data);

    for (values, 0..) |v, i| {
        data[i] = math.complex.init(v, 0.0);
    }

    try fft.fft3dForwardInPlace(alloc, data, grid.nx, grid.ny, grid.nz);

    const scale = 1.0 / @as(f64, @floatFromInt(total));
    for (data) |*v| {
        v.* = math.complex.scale(v.*, scale);
    }

    const out = try alloc.alloc(math.Complex, total);
    const nx = grid.nx;
    const ny = grid.ny;
    const nz = grid.nz;
    var idx: usize = 0;
    var z: usize = 0;
    while (z < nz) : (z += 1) {
        var y: usize = 0;
        while (y < ny) : (y += 1) {
            var x: usize = 0;
            while (x < nx) : (x += 1) {
                const fh = indexToFreq(x, nx);
                const fk = indexToFreq(y, ny);
                const fl = indexToFreq(z, nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                const out_idx = th + nx * (tk + ny * tl);
                out[out_idx] = data[idx];
                idx += 1;
            }
        }
    }
    return out;
}

/// Convert reciprocal grid to real using radix-2 FFT.
fn fftReciprocalToReal(alloc: std.mem.Allocator, grid: Grid, values: []math.Complex) ![]f64 {
    const total = grid.count();
    if (values.len != total) return error.InvalidGrid;

    const data = try alloc.alloc(math.Complex, total);
    defer alloc.free(data);

    const scale = @as(f64, @floatFromInt(total));
    var idx: usize = 0;
    var z: usize = 0;
    while (z < grid.nz) : (z += 1) {
        var y: usize = 0;
        while (y < grid.ny) : (y += 1) {
            var x: usize = 0;
            while (x < grid.nx) : (x += 1) {
                const fh = indexToFreq(x, grid.nx);
                const fk = indexToFreq(y, grid.ny);
                const fl = indexToFreq(z, grid.nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                const in_idx = th + grid.nx * (tk + grid.ny * tl);
                data[idx] = math.complex.scale(values[in_idx], scale);
                idx += 1;
            }
        }
    }

    try fft.fft3dInverseInPlace(alloc, data, grid.nx, grid.ny, grid.nz);

    const out = try alloc.alloc(f64, total);
    for (data, 0..) |v, i| {
        out[i] = v.r;
    }
    return out;
}

/// Convert reciprocal grid to complex real-space grid in-place.
pub fn fftReciprocalToComplexInPlace(
    alloc: std.mem.Allocator,
    grid: Grid,
    values: []const math.Complex,
    out: []math.Complex,
    plan: ?*fft.Fft3dPlan,
) !void {
    const total = grid.count();
    if (values.len != total or out.len != total) return error.InvalidGrid;

    const scale = @as(f64, @floatFromInt(total));
    var idx: usize = 0;
    var z: usize = 0;
    while (z < grid.nz) : (z += 1) {
        var y: usize = 0;
        while (y < grid.ny) : (y += 1) {
            var x: usize = 0;
            while (x < grid.nx) : (x += 1) {
                const fh = indexToFreq(x, grid.nx);
                const fk = indexToFreq(y, grid.ny);
                const fl = indexToFreq(z, grid.nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                const in_idx = th + grid.nx * (tk + grid.ny * tl);
                out[idx] = math.complex.scale(values[in_idx], scale);
                idx += 1;
            }
        }
    }

    if (plan) |p| {
        if (p.nx != grid.nx or p.ny != grid.ny or p.nz != grid.nz) return error.InvalidGrid;
        try fft.fft3dInverseInPlacePlan(p, out);
    } else {
        try fft.fft3dInverseInPlace(alloc, out, grid.nx, grid.ny, grid.nz);
    }
}

pub fn fftReciprocalToComplexInPlaceMapped(
    alloc: std.mem.Allocator,
    grid: Grid,
    map: []const usize,
    values: []const math.Complex,
    out: []math.Complex,
    plan: ?*fft.Fft3dPlan,
) !void {
    const total = grid.count();
    if (values.len != total or out.len != total or map.len != total) return error.InvalidGrid;

    const scale = @as(f64, @floatFromInt(total));
    var idx: usize = 0;
    while (idx < total) : (idx += 1) {
        out[idx] = math.complex.scale(values[map[idx]], scale);
    }

    if (plan) |p| {
        if (p.nx != grid.nx or p.ny != grid.ny or p.nz != grid.nz) return error.InvalidGrid;
        try fft.fft3dInverseInPlacePlan(p, out);
    } else {
        try fft.fft3dInverseInPlace(alloc, out, grid.nx, grid.ny, grid.nz);
    }
}

/// Convert complex real-space grid to reciprocal grid in-place.
pub fn fftComplexToReciprocalInPlace(
    alloc: std.mem.Allocator,
    grid: Grid,
    data: []math.Complex,
    out: []math.Complex,
    plan: ?*fft.Fft3dPlan,
) !void {
    const total = grid.count();
    if (data.len != total or out.len != total) return error.InvalidGrid;

    if (plan) |p| {
        if (p.nx != grid.nx or p.ny != grid.ny or p.nz != grid.nz) return error.InvalidGrid;
        try fft.fft3dForwardInPlacePlan(p, data);
    } else {
        try fft.fft3dForwardInPlace(alloc, data, grid.nx, grid.ny, grid.nz);
    }

    const scale = 1.0 / @as(f64, @floatFromInt(total));
    var idx: usize = 0;
    var z: usize = 0;
    while (z < grid.nz) : (z += 1) {
        var y: usize = 0;
        while (y < grid.ny) : (y += 1) {
            var x: usize = 0;
            while (x < grid.nx) : (x += 1) {
                const fh = indexToFreq(x, grid.nx);
                const fk = indexToFreq(y, grid.ny);
                const fl = indexToFreq(z, grid.nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                const out_idx = th + grid.nx * (tk + grid.ny * tl);
                out[out_idx] = math.complex.scale(data[idx], scale);
                idx += 1;
            }
        }
    }
}

pub fn fftComplexToReciprocalInPlaceMapped(
    alloc: std.mem.Allocator,
    grid: Grid,
    map: []const usize,
    data: []math.Complex,
    out: []math.Complex,
    plan: ?*fft.Fft3dPlan,
) !void {
    const total = grid.count();
    if (data.len != total or out.len != total or map.len != total) return error.InvalidGrid;

    if (plan) |p| {
        if (p.nx != grid.nx or p.ny != grid.ny or p.nz != grid.nz) return error.InvalidGrid;
        try fft.fft3dForwardInPlacePlan(p, data);
    } else {
        try fft.fft3dForwardInPlace(alloc, data, grid.nx, grid.ny, grid.nz);
    }

    const scale = 1.0 / @as(f64, @floatFromInt(total));
    var idx: usize = 0;
    while (idx < total) : (idx += 1) {
        out[map[idx]] = math.complex.scale(data[idx], scale);
    }
}

pub fn buildFftIndexMap(alloc: std.mem.Allocator, grid: Grid) ![]usize {
    const total = grid.count();
    const map = try alloc.alloc(usize, total);
    const nx = grid.nx;
    const ny = grid.ny;
    const nz = grid.nz;
    var idx: usize = 0;
    var z: usize = 0;
    while (z < nz) : (z += 1) {
        var y: usize = 0;
        while (y < ny) : (y += 1) {
            var x: usize = 0;
            while (x < nx) : (x += 1) {
                const fh = indexToFreq(x, nx);
                const fk = indexToFreq(y, ny);
                const fl = indexToFreq(z, nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                map[idx] = th + nx * (tk + ny * tl);
                idx += 1;
            }
        }
    }
    return map;
}

/// Convert FFT index to signed frequency.
pub fn indexToFreq(i: usize, n: usize) i32 {
    const half = (n - 1) / 2;
    return if (i <= half) @as(i32, @intCast(i)) else @as(i32, @intCast(i)) - @as(i32, @intCast(n));
}
