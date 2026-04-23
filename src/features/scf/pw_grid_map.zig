const std = @import("std");
const math = @import("../math/math.zig");
const grid_mod = @import("pw_grid.zig");
const plane_wave = @import("../plane_wave/basis.zig");

pub const Grid = grid_mod.Grid;

pub const PwGridMap = struct {
    indices: []usize,
    fft_indices: []usize, // PW -> FFT-order position (for fused scatter/gather)
    nx: usize,
    ny: usize,
    nz: usize,
    min_h: i32,
    min_k: i32,
    min_l: i32,

    pub fn init(
        alloc: std.mem.Allocator,
        gvecs: []const plane_wave.GVector,
        grid: Grid,
    ) !PwGridMap {
        const idxs = try alloc.alloc(usize, gvecs.len);
        errdefer alloc.free(idxs);
        for (gvecs, 0..) |g, i| {
            const hi = g.h - grid.min_h;
            const ki = g.k - grid.min_k;
            const li = g.l - grid.min_l;
            if (hi < 0 or ki < 0 or li < 0) return error.InvalidGrid;
            const nx_i = @as(i32, @intCast(grid.nx));
            const ny_i = @as(i32, @intCast(grid.ny));
            const nz_i = @as(i32, @intCast(grid.nz));
            if (hi >= nx_i or ki >= ny_i or li >= nz_i) {
                return error.InvalidGrid;
            }
            const h = @as(usize, @intCast(hi));
            const k = @as(usize, @intCast(ki));
            const l = @as(usize, @intCast(li));
            idxs[i] = h + grid.nx * (k + grid.ny * l);
        }
        return .{
            .indices = idxs,
            .fft_indices = &[_]usize{},
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
        };
    }

    pub fn deinit(self: *PwGridMap, alloc: std.mem.Allocator) void {
        if (self.indices.len > 0) alloc.free(self.indices);
        if (self.fft_indices.len > 0) alloc.free(self.fft_indices);
    }

    /// Build direct PW->FFT index mapping for fused scatter/gather.
    /// Eliminates the full-grid remap+scale loops (32K ops -> 2K ops).
    pub fn build_fft_indices(
        self: *PwGridMap,
        alloc: std.mem.Allocator,
        fft_index_map: []const usize,
    ) !void {
        const total = fft_index_map.len;
        // Build inverse map: recip_grid_pos -> fft_sequential_pos
        const inv_map = try alloc.alloc(usize, total);
        defer alloc.free(inv_map);

        for (fft_index_map, 0..) |recip_idx, fft_idx| {
            inv_map[recip_idx] = fft_idx;
        }
        // Build PW -> FFT indices
        const fft_idx = try alloc.alloc(usize, self.indices.len);
        for (self.indices, 0..) |recip_idx, i| {
            fft_idx[i] = inv_map[recip_idx];
        }
        self.fft_indices = fft_idx;
    }

    pub fn scatter(self: PwGridMap, coeffs: []const math.Complex, grid_data: []math.Complex) void {
        @memset(grid_data, math.complex.init(0.0, 0.0));
        for (self.indices, 0..) |idx, i| {
            grid_data[idx] = coeffs[i];
        }
    }

    pub fn gather(self: PwGridMap, grid_data: []const math.Complex, out: []math.Complex) void {
        for (self.indices, 0..) |idx, i| {
            out[i] = grid_data[idx];
        }
    }

    /// Scatter PW coefficients directly to FFT-ordered buffer with scaling.
    pub fn scatter_fft(
        self: PwGridMap,
        coeffs: []const math.Complex,
        fft_data: []math.Complex,
        scale: f64,
    ) void {
        @memset(fft_data, math.complex.init(0.0, 0.0));
        for (self.fft_indices, 0..) |idx, i| {
            fft_data[idx] = math.complex.scale(coeffs[i], scale);
        }
    }

    /// Gather PW coefficients directly from FFT-ordered buffer with scaling.
    pub fn gather_fft(
        self: PwGridMap,
        fft_data: []const math.Complex,
        out: []math.Complex,
        scale: f64,
    ) void {
        for (self.fft_indices, 0..) |idx, i| {
            out[i] = math.complex.scale(fft_data[idx], scale);
        }
    }
};
