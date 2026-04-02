const std = @import("std");
const math = @import("../math/math.zig");
const scf = @import("../scf/scf.zig");
const stress_util = @import("stress.zig");

const Stress3x3 = stress_util.Stress3x3;
const Grid = stress_util.Grid;

/// Hartree stress: σ_αβ = -(E_H/Ω) δ_αβ + 8π Σ_{G≠0} |ρ(G)|² G_α G_β / |G|⁴
pub fn hartreeStress(grid: Grid, rho_g: []const math.Complex, e_hartree: f64, inv_volume: f64, ecutrho: f64) Stress3x3 {
    var sigma = stress_util.zeroStress();

    var it = scf.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.gh == 0 and g.gk == 0 and g.gl == 0) {
            continue;
        }
        if (g.g2 < 1e-12 or g.g2 >= ecutrho) {
            continue;
        }
        const rho_val = rho_g[g.idx];
        const rho2 = rho_val.r * rho_val.r + rho_val.i * rho_val.i;
        const factor = 8.0 * std.math.pi * rho2 / (g.g2 * g.g2);
        const gv = [3]f64{ g.gvec.x, g.gvec.y, g.gvec.z };

        for (0..3) |a| {
            for (a..3) |b| {
                sigma[a][b] += factor * gv[a] * gv[b];
            }
        }
    }

    // Add diagonal: -(E_H/Ω) δ_αβ
    for (0..3) |a| {
        sigma[a][a] -= e_hartree * inv_volume;
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| {
            sigma[b][a] = sigma[a][b];
        }
    }
    return sigma;
}
