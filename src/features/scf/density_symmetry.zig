const std = @import("std");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const math = @import("../math/math.zig");
const symmetry = @import("../symmetry/symmetry.zig");

const Grid = grid_mod.Grid;

const real_to_reciprocal = fft_grid.real_to_reciprocal;
const reciprocal_to_real = fft_grid.reciprocal_to_real;

pub fn total_charge(rho: []const f64, grid: Grid) f64 {
    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var sum: f64 = 0.0;
    for (rho) |value| {
        sum += value * dv;
    }
    return sum;
}

fn wrap_grid_index(g: i32, min: i32, n: usize) usize {
    const ni = @as(i32, @intCast(n));
    const idx = @mod(g - min, ni);
    return @as(usize, @intCast(idx));
}

pub fn symmetrize_density(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    ops: []const symmetry.SymOp,
    use_rfft: bool,
) !void {
    if (ops.len <= 1) return;

    const rho_g = try real_to_reciprocal(alloc, grid, rho, use_rfft);
    defer alloc.free(rho_g);

    const total = grid.count();
    const rho_sym = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho_sym);

    @memset(rho_sym, math.complex.init(0.0, 0.0));

    const inv_ops = 1.0 / @as(f64, @floatFromInt(ops.len));
    const two_pi = 2.0 * std.math.pi;
    const phase_tol = 1e-12;

    var z: usize = 0;
    while (z < grid.nz) : (z += 1) {
        const gl = grid.min_l + @as(i32, @intCast(z));
        var y: usize = 0;
        while (y < grid.ny) : (y += 1) {
            const gk = grid.min_k + @as(i32, @intCast(y));
            var x: usize = 0;
            while (x < grid.nx) : (x += 1) {
                const gh = grid.min_h + @as(i32, @intCast(x));
                var sum = math.complex.init(0.0, 0.0);

                for (ops) |op| {
                    const m = op.k_rot.m;
                    const mh = m[0][0] * gh + m[0][1] * gk + m[0][2] * gl;
                    const mk = m[1][0] * gh + m[1][1] * gk + m[1][2] * gl;
                    const ml = m[2][0] * gh + m[2][1] * gk + m[2][2] * gl;

                    const ix = wrap_grid_index(mh, grid.min_h, grid.nx);
                    const iy = wrap_grid_index(mk, grid.min_k, grid.ny);
                    const iz = wrap_grid_index(ml, grid.min_l, grid.nz);
                    const idx = ix + grid.nx * (iy + grid.ny * iz);
                    var term = rho_g[idx];

                    const dot = @as(f64, @floatFromInt(gh)) * op.trans.x +
                        @as(f64, @floatFromInt(gk)) * op.trans.y +
                        @as(f64, @floatFromInt(gl)) * op.trans.z;
                    const frac = dot - std.math.floor(dot);
                    if (frac > phase_tol and frac < 1.0 - phase_tol) {
                        const phase = math.complex.expi(-two_pi * frac);
                        term = math.complex.mul(term, phase);
                    }

                    sum = math.complex.add(sum, term);
                }

                const out_idx = x + grid.nx * (y + grid.ny * z);
                rho_sym[out_idx] = math.complex.scale(sum, inv_ops);
            }
        }
    }

    const rho_real = try reciprocal_to_real(alloc, grid, rho_sym);
    defer alloc.free(rho_real);

    std.mem.copyForwards(f64, rho, rho_real);
}
