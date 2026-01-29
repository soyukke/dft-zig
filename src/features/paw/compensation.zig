const std = @import("std");
const math = @import("../math/math.zig");
const paw_tab = @import("paw_tab.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const PawTab = paw_tab.PawTab;

/// Build the PAW compensation charge density n_hat(G) in reciprocal space.
///
/// n_hat(G) = Σ_a Σ_{ij} ρ_ij^a × Σ_L Q_ij^L(|G|) × Σ_M Y_LM(Ĝ) × exp(-iG·R_a) / Ω
///
/// For the spherically symmetric case (L=0 only), this simplifies to:
/// n_hat(G) = Σ_a Σ_{ij} ρ_ij^a × Q_ij^0(|G|) × Y_00 × exp(-iG·R_a) / Ω
///
/// This function adds n_hat(G) to the provided density array.
pub fn buildCompensationDensityG(
    rho_g: []math.Complex,
    rhoij_values: []const []const f64,
    tabs: []const *const PawTab,
    atom_species: []const usize,
    atom_positions: []const math.Vec3,
    gvecs: []const [3]i32,
    recip_lat: [3]math.Vec3,
    omega: f64,
    natom: usize,
) void {
    const inv_omega = 1.0 / omega;

    for (gvecs, 0..) |gv, ig| {
        // Compute G vector in Cartesian coordinates
        const gx = @as(f64, @floatFromInt(gv[0])) * recip_lat[0].x +
            @as(f64, @floatFromInt(gv[1])) * recip_lat[1].x +
            @as(f64, @floatFromInt(gv[2])) * recip_lat[2].x;
        const gy = @as(f64, @floatFromInt(gv[0])) * recip_lat[0].y +
            @as(f64, @floatFromInt(gv[1])) * recip_lat[1].y +
            @as(f64, @floatFromInt(gv[2])) * recip_lat[2].y;
        const gz = @as(f64, @floatFromInt(gv[0])) * recip_lat[0].z +
            @as(f64, @floatFromInt(gv[1])) * recip_lat[1].z +
            @as(f64, @floatFromInt(gv[2])) * recip_lat[2].z;
        const g_abs = @sqrt(gx * gx + gy * gy + gz * gz);

        var sum_re: f64 = 0.0;
        var sum_im: f64 = 0.0;

        for (0..natom) |a| {
            const sp = atom_species[a];
            const tab = tabs[sp];
            const nb = tab.nbeta;
            const rhoij = rhoij_values[a];
            const pos = atom_positions[a];

            // Structure factor: exp(-iG·R_a)
            const g_dot_r = gx * pos.x + gy * pos.y + gz * pos.z;
            const sf_re = @cos(g_dot_r);
            const sf_im = -@sin(g_dot_r);

            // Sum over all (i,j) pairs and L values
            for (0..tab.n_qijl_entries) |e| {
                const idx = tab.qijl_indices[e];
                const i = idx.first;
                const j = idx.second;
                const l = idx.l;

                const rij = rhoij[i * nb + j];
                if (@abs(rij) < 1e-30) continue;

                // Q_ij^L(|G|) from table
                const qijl_g = tab.evalQijlForm(e, g_abs);
                if (@abs(qijl_g) < 1e-30) continue;

                // Spherical harmonic sum: Σ_M Y_LM(Ĝ) Y_LM(R̂_a)
                // For augmentation charges, we need Σ_M Y_LM(Ĝ)
                // In the PAW formalism with PSQ augmentation, the spherical
                // harmonic structure factor is:
                // For L=0: Y_00 = 1/sqrt(4π), giving a factor of 1/sqrt(4π)
                // For L>0: requires real spherical harmonics of G-direction
                var ylm_sum: f64 = 0.0;
                if (g_abs > 1e-10) {
                    const l_i32: i32 = @intCast(l);
                    var m: i32 = -l_i32;
                    while (m <= l_i32) : (m += 1) {
                        const ylm = nonlocal.realSphericalHarmonic(l_i32, m, gx, gy, gz);
                        ylm_sum += ylm * ylm;
                    }
                    // The sum Σ_M |Y_LM(Ĝ)|² = (2L+1)/(4π) by the addition theorem
                    // But for the density we need Σ_M Y_LM(Ĝ) which for individual atoms
                    // gives Y_L0 only for the spherical part. For the general case:
                    // n_hat = Σ_a Σ_ij ρ_ij Σ_L Q_ij^L(G) × (4π/(2L+1)) Σ_M Y_LM(Ĝ) Y*_LM(R̂_a)
                    // But since augmentation charges are spherically symmetric around each atom,
                    // and ρ_ij is also spherically summed, for L=0:
                    ylm_sum = nonlocal.realSphericalHarmonic(@intCast(l), 0, gx, gy, gz);
                    // This needs the full angular structure for non-spherical PAW
                    // For now, use the standard formula with Y_L0 only (spherical approx)
                } else {
                    // G=0: only L=0 contributes
                    if (l == 0) {
                        ylm_sum = 1.0 / @sqrt(4.0 * std.math.pi);
                    }
                }

                // Symmetry factor: for i!=j, both (i,j) and (j,i) contribute
                const sym_factor: f64 = if (i != j) 2.0 else 1.0;

                const contrib = rij * qijl_g * ylm_sum * sym_factor * inv_omega;
                sum_re += contrib * sf_re;
                sum_im += contrib * sf_im;
            }
        }

        rho_g[ig].r += sum_re;
        rho_g[ig].i += sum_im;
    }
}
