const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const scf = @import("../scf/scf.zig");
const stress_util = @import("stress.zig");

const Stress3x3 = stress_util.Stress3x3;
const Grid = stress_util.Grid;

/// Local pseudopotential stress:
/// σ_αβ = -(E_loc/Ω) δ_αβ - (1/Ω) Σ_{G≠0} (G_αG_β/|G|) × Σ_I V'_form(|G|) × Re[ρ*(G) S_I(G)]
pub fn localStress(
    grid: Grid,
    rho_g: []const math.Complex,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    inv_volume: f64,
    ecutrho: f64,
) Stress3x3 {
    var sigma = stress_util.zeroStress();

    // Accumulate E_loc = Σ_{G≠0} V_form(|G|) × Re[ρ(G) exp(+iGR)] internally.
    // For PAW, rho_g is the augmented density ρ̃+n̂, giving the correct evloc.
    // This matches QE's stres_loc which uses rho%of_r (augmented density).
    var evloc: f64 = 0.0;

    var it = scf.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.gh == 0 and g.gk == 0 and g.gl == 0) {
            continue;
        }
        const g_norm = math.Vec3.norm(g.gvec);
        if (g_norm < 1e-12 or g.g2 >= ecutrho) {
            continue;
        }
        const rho_val = rho_g[g.idx];
        const gv = [3]f64{ g.gvec.x, g.gvec.y, g.gvec.z };

        // Accumulate: Σ_I V_form(|G|) and V_form'(|G|) contributions
        var vloc_rho_re: f64 = 0.0; // Re[V_form × S*_I(G) × ρ(G)]
        var dvloc_rho_re: f64 = 0.0; // Re[V'_form × S*_I(G) × ρ(G)]

        for (atoms) |atom| {
            const v_loc = if (ff_tables) |tables|
                tables[atom.species_index].eval(g_norm)
            else
                hamiltonian.localFormFactor(&species[atom.species_index], g_norm);

            const dv_loc = if (ff_tables) |tables|
                tables[atom.species_index].evalDeriv(g_norm)
            else blk: {
                // Numerical derivative
                const dq: f64 = 0.01;
                const vp = hamiltonian.localFormFactor(&species[atom.species_index], g_norm + dq);
                const vm = hamiltonian.localFormFactor(&species[atom.species_index], g_norm - dq);
                break :blk (vp - vm) / (2.0 * dq);
            };

            const phase = math.Vec3.dot(g.gvec, atom.position);
            const cos_phase = std.math.cos(phase);
            const sin_phase = std.math.sin(phase);
            // Energy: E_loc = Ω Σ_G Re[ρ̃ conj(V_G)] where V_G = (1/Ω) V_form exp(-iGR)
            // conj(V_G) = (1/Ω) V_form exp(+iGR)
            // Re[ρ̃ exp(+iGR)] = Re[(ρ_r+iρ_i)(cos+isin)] = ρ_r cos - ρ_i sin
            // The stress G-derivative uses the same phase factor.
            const re_rho_si = rho_val.r * cos_phase - rho_val.i * sin_phase;

            vloc_rho_re += v_loc * re_rho_si;
            dvloc_rho_re += dv_loc * re_rho_si;
        }

        evloc += vloc_rho_re;

        // Local stress contribution from this G:
        // From V_form(|G|) dependence: dV_form/dε_αβ = V'_form × d|G|/dε_αβ = V'_form × (-G_αG_β/|G|)
        // From 1/Ω (in ρ): -(1-Tr(ε)) gives -δ_αβ × E_loc at the end
        // So: σ_αβ += -(1/Ω) × dvloc_rho_re × G_αG_β/|G|
        const inv_gnorm = 1.0 / g_norm;
        for (0..3) |a| {
            for (a..3) |b| {
                sigma[a][b] -= dvloc_rho_re * gv[a] * gv[b] * inv_gnorm * inv_volume;
            }
        }
    }

    // Diagonal: -(E_loc/Ω) δ_αβ, using internally-computed evloc from the augmented density.
    for (0..3) |a| {
        sigma[a][a] -= evloc * inv_volume;
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}
