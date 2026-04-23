const std = @import("std");
const math = @import("../math/math.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const scf = @import("../scf/scf.zig");
const stress_util = @import("stress.zig");

const Stress3x3 = stress_util.Stress3x3;
const Grid = stress_util.Grid;

/// Kinetic stress: σ_αβ = -(2 × spin / Ω) Σ_nk f w Σ_G (k+G)_α (k+G)_β |c(G)|²
pub fn kineticStress(
    alloc: std.mem.Allocator,
    wavefunctions: ?scf.WavefunctionData,
    recip: math.Mat3,
    inv_volume: f64,
    spin_factor: f64,
) !Stress3x3 {
    var sigma = stress_util.zeroStress();
    const wf = wavefunctions orelse return sigma;

    for (wf.kpoints) |kp| {
        var basis = try plane_wave.generate(alloc, recip, wf.ecut_ry, kp.k_cart);
        defer basis.deinit(alloc);

        const gvecs = basis.gvecs;
        const n = gvecs.len;
        if (n != kp.basis_len) continue;

        for (0..kp.nbands) |band| {
            const occ = kp.occupations[band];
            if (occ <= 0.0) continue;
            const c = kp.coefficients[band * n .. (band + 1) * n];
            const prefactor = -2.0 * occ * kp.weight * spin_factor;

            for (0..n) |g| {
                const q = gvecs[g].kpg; // k+G cartesian
                const c2 = c[g].r * c[g].r + c[g].i * c[g].i;
                const qv = [3]f64{ q.x, q.y, q.z };
                for (0..3) |a| {
                    for (a..3) |b| {
                        sigma[a][b] += prefactor * qv[a] * qv[b] * c2;
                    }
                }
            }
        }
    }

    // Symmetrize and scale by inv_volume
    for (0..3) |a| {
        for (a..3) |b| {
            sigma[a][b] *= inv_volume;
            if (a != b) sigma[b][a] = sigma[a][b];
        }
    }
    return sigma;
}
