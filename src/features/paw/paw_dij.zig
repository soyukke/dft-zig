const std = @import("std");
const math = @import("../math/math.zig");
const paw_tab = @import("paw_tab.zig");
const PawTab = paw_tab.PawTab;

/// Compute total D_ij for one PAW atom.
///
/// D_ij = D^0_ij + D^hat_ij [+ D^H_ij + D^xc_ij (Phase 5)]
///
/// - D^0_ij: From UPF file (atomic reference)
/// - D^hat_ij: ∫ V_eff(G) × Q_ij(G) dG (G-space, using form factor tables)
/// - D^H_ij: On-site Hartree correction (Phase 5)
/// - D^xc_ij: On-site XC correction (Phase 5)
pub fn computePawDij(
    dij_out: []f64,
    dij0: []const f64,
    tab: *const PawTab,
    v_eff_g: []const math.Complex,
    gvecs: []const [3]i32,
    recip_lat: [3]math.Vec3,
    atom_pos: math.Vec3,
    omega: f64,
) void {
    const nbeta = tab.nbeta;
    const n_ij = nbeta * nbeta;

    // Start with D^0_ij from UPF
    if (dij0.len >= n_ij) {
        @memcpy(dij_out[0..n_ij], dij0[0..n_ij]);
    } else {
        @memset(dij_out[0..n_ij], 0.0);
    }

    // Add D^hat_ij = Σ_G V_eff(G) × Q_ij(G) × exp(-iG·R_a)
    // Q_ij(G) = Σ_L Q_ij^L(|G|) × Y_L0(Ĝ) (spherical approximation)
    for (gvecs, 0..) |gv, ig| {
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

        // Structure factor: exp(-iG·R_a)
        const g_dot_r = gx * atom_pos.x + gy * atom_pos.y + gz * atom_pos.z;
        const sf_re = @cos(g_dot_r);
        const sf_im = -@sin(g_dot_r);

        // V_eff(G) * exp(-iG·R_a) - take real part
        const veff = v_eff_g[ig];
        const prod_re = veff.r * sf_re - veff.i * sf_im;

        // For each (i,j,L) entry in the QIJL table
        for (0..tab.n_qijl_entries) |e| {
            const idx = tab.qijl_indices[e];
            const i = idx.first;
            const j = idx.second;

            // Q_ij^L(|G|) form factor
            const qijl_g = tab.evalQijlForm(e, g_abs);
            if (@abs(qijl_g) < 1e-30) continue;

            // D^hat_ij contribution (real part only since D_ij is Hermitian real)
            const contrib = prod_re * qijl_g * omega;
            dij_out[i * nbeta + j] += contrib;
            if (i != j) {
                dij_out[j * nbeta + i] += contrib;
            }
        }
    }
}
