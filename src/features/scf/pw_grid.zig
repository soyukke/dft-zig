const std = @import("std");
const config = @import("../config/config.zig");
const fft_sizing = @import("../../lib/fft/sizing.zig");
const math = @import("../math/math.zig");

const next_fft_size = fft_sizing.next_fft_size;

/// FFT grid metadata used in SCF.
pub const Grid = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    cell: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    min_h: i32,
    min_k: i32,
    min_l: i32,

    /// Return total grid points.
    pub fn count(self: Grid) usize {
        return self.nx * self.ny * self.nz;
    }
};

/// Build a Grid from config, using auto_grid for any zero dimensions.
pub fn grid_from_config(cfg: config.Config, recip: math.Mat3, volume: f64) Grid {
    var nx = cfg.scf.grid[0];
    var ny = cfg.scf.grid[1];
    var nz = cfg.scf.grid[2];
    if (nx == 0 or ny == 0 or nz == 0) {
        const auto = auto_grid(cfg.scf.ecut_ry, cfg.scf.grid_scale, recip);
        if (nx == 0) nx = auto[0];
        if (ny == 0) ny = auto[1];
        if (nz == 0) nz = auto[2];
    }
    const min_h = min_index(nx);
    const min_k = min_index(ny);
    const min_l = min_index(nz);
    return Grid{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .cell = cfg.cell.scale(math.units_scale_to_bohr(cfg.units)),
        .recip = recip,
        .volume = volume,
        .min_h = min_h,
        .min_k = min_k,
        .min_l = min_l,
    };
}

/// Compute automatic grid size from cutoff.
pub fn auto_grid(ecut_ry: f64, grid_scale: f64, recip: math.Mat3) [3]usize {
    const scale = if (grid_scale > 0.0) grid_scale else 1.0;
    const density_gmax = @max(2.0, scale) * @sqrt(ecut_ry);
    const b1 = math.Vec3.norm(recip.row(0));
    const b2 = math.Vec3.norm(recip.row(1));
    const b3 = math.Vec3.norm(recip.row(2));
    const n1 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / b1))) * 2 + 1;
    const n2 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / b2))) * 2 + 1;
    const n3 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / b3))) * 2 + 1;
    return .{
        next_fft_size(@max(n1, 3)),
        next_fft_size(@max(n2, 3)),
        next_fft_size(@max(n3, 3)),
    };
}

pub fn min_index(n: usize) i32 {
    return -@as(i32, @intCast(n / 2));
}
