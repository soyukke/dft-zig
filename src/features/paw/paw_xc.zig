const std = @import("std");
const paw_data = @import("../pseudopotential/paw_data.zig");
const PawData = paw_data.PawData;
const QijlEntry = paw_data.QijlEntry;
const xc = @import("../xc/xc.zig");
const gaunt_mod = @import("gaunt.zig");
const GauntTable = gaunt_mod.GauntTable;

const ctrapWeight = @import("../math/math.zig").radial.ctrapWeight;
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const lebedev = @import("../grid/lebedev.zig");

/// Find Q_ij^L(r) entry from paw.qijl for given (i,j,L) triplet.
fn findQijL(paw: PawData, i: usize, j: usize, big_l: usize) ?[]const f64 {
    for (paw.qijl) |entry| {
        if (entry.angular_momentum == big_l and
            ((entry.first_index == i and entry.second_index == j) or
                (entry.first_index == j and entry.second_index == i)))
        {
            return entry.values;
        }
    }
    return null;
}

/// Solve L=0 radial Poisson equation on logarithmic grid (Rydberg units).
///
/// Input: rho(r) = spherical density (number density, not multiplied by 4πr²)
/// Output: v_h(r) = Hartree potential
///
/// Formula: V_H(r) = 8π [Q_in(r)/r + Q_out(r)]
///   where Q_in(r) = ∫₀ʳ ρ(r')r'² dr', Q_out(r) = ∫ᵣ^∞ ρ(r')r' dr'
fn radialHartreePotentialL0(
    r: []const f64,
    rab: []const f64,
    rho: []const f64,
    v_h: []f64,
    n_mesh: usize,
) void {
    const eight_pi = 8.0 * std.math.pi;

    // Forward cumulative integral: Q_in(r) = ∫₀ʳ ρ(r') r'² dr'
    // Use trapezoidal rule with rab weights
    var q_in: f64 = 0.0;
    // Store q_in values for each grid point
    var q_in_vals: [4096]f64 = undefined;
    for (0..n_mesh) |k| {
        const integrand = rho[k] * r[k] * r[k] * rab[k] * ctrapWeight(k, n_mesh);
        q_in += integrand;
        q_in_vals[k] = q_in;
    }

    // Backward cumulative integral: Q_out(r) = ∫ᵣ^∞ ρ(r') r' dr'
    var q_out: f64 = 0.0;
    var k_rev: usize = n_mesh;
    while (k_rev > 0) {
        k_rev -= 1;
        const integrand = rho[k_rev] * r[k_rev] * rab[k_rev] * ctrapWeight(k_rev, n_mesh);
        q_out += integrand;
        // V_H(r) = 8π [Q_in(r)/r + Q_out(r)]
        if (r[k_rev] > 1e-10) {
            v_h[k_rev] = eight_pi * (q_in_vals[k_rev] / r[k_rev] + q_out);
        } else {
            v_h[k_rev] = 0.0;
        }
    }
    // Remove self-interaction at origin: V_H(0) = 8π × Q_out(0) = 8π × total charge / r
    // The r=0 point contribution to Q_in is zero, but Q_out includes everything.
    // At r=0, V_H = 8π × Q_out(0) which we set to the limit.
    if (n_mesh > 1 and r[0] < 1e-10) {
        // Extrapolate from next point
        v_h[0] = v_h[1];
    }
}

/// Solve general-L radial Poisson equation on logarithmic grid (Rydberg units).
///
/// Input: rho_L(r) = density multipole (number density, not multiplied by 4πr²)
/// Output: v_h_L(r) = Hartree potential for angular momentum L
///
/// Formula: V_L(r) = (8π/(2L+1)) × [r^{-L-1} I_in(r) + r^L I_out(r)]
///   where I_in(r) = ∫₀ʳ ρ(r') r'^{L+2} dr', I_out(r) = ∫ᵣ^∞ ρ(r') r'^{1-L} dr'
fn radialHartreePotentialL(
    r: []const f64,
    rab: []const f64,
    rho: []const f64,
    v_h: []f64,
    n_mesh: usize,
    big_l: usize,
) void {
    const fl: f64 = @floatFromInt(big_l);
    const prefactor = 8.0 * std.math.pi / (2.0 * fl + 1.0);

    // Forward cumulative integral: I_in(r) = ∫₀ʳ ρ(r') r'^{L+2} dr'
    var i_in: f64 = 0.0;
    var i_in_vals: [4096]f64 = undefined;
    for (0..n_mesh) |k| {
        const rp = r[k];
        const rp_pow = std.math.pow(f64, rp, fl + 2.0);
        i_in += rho[k] * rp_pow * rab[k] * ctrapWeight(k, n_mesh);
        i_in_vals[k] = i_in;
    }

    // Backward cumulative integral: I_out(r) = ∫ᵣ^∞ ρ(r') r'^{1-L} dr'
    var i_out: f64 = 0.0;
    var k_rev: usize = n_mesh;
    while (k_rev > 0) {
        k_rev -= 1;
        const rp = r[k_rev];
        const rp_pow = std.math.pow(f64, rp, 1.0 - fl);
        i_out += rho[k_rev] * rp_pow * rab[k_rev] * ctrapWeight(k_rev, n_mesh);
        if (rp > 1e-10) {
            const r_neg = std.math.pow(f64, rp, -(fl + 1.0));
            const r_pos = std.math.pow(f64, rp, fl);
            v_h[k_rev] = prefactor * (r_neg * i_in_vals[k_rev] + r_pos * i_out);
        } else {
            v_h[k_rev] = 0.0;
        }
    }
    if (n_mesh > 1 and r[0] < 1e-10) {
        v_h[0] = if (big_l == 0) v_h[1] else 0.0;
    }
}

/// Compute multi-L on-site Hartree energy for one PAW atom.
///
/// E_H = Σ_{L,M} ½ ∫ V_H^{LM}(r) × ρ_{LM}(r) × r² dr
/// Uses full multipole expansion with Gaunt coefficients.
/// Returns E_H^AE - E_H^PS (including Q contributions in PS density).
pub fn computePawEhOnsiteMultiL(
    alloc: std.mem.Allocator,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    gaunt_table: *const GauntTable,
) !f64 {
    const nbeta = paw.number_of_proj;
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0) @min(n_mesh_full, paw.cutoff_r_index) else n_mesh_full;
    if (n_mesh > 4096) return error.MeshTooLarge;

    const lmax_aug = gaunt_table.lmax_aug;
    const n_lm_aug = (lmax_aug + 1) * (lmax_aug + 1);

    // Allocate density arrays for each (L,M) channel
    const rho_ae_lm = try alloc.alloc([]f64, n_lm_aug);
    defer {
        for (rho_ae_lm) |s| alloc.free(s);
        alloc.free(rho_ae_lm);
    }
    const rho_ps_lm = try alloc.alloc([]f64, n_lm_aug);
    defer {
        for (rho_ps_lm) |s| alloc.free(s);
        alloc.free(rho_ps_lm);
    }
    for (0..n_lm_aug) |lm| {
        rho_ae_lm[lm] = try alloc.alloc(f64, n_mesh);
        @memset(rho_ae_lm[lm], 0.0);
        rho_ps_lm[lm] = try alloc.alloc(f64, n_mesh);
        @memset(rho_ps_lm[lm], 0.0);
    }

    // Build ρ_{LM}(r) for each (L,M) channel
    for (0..nbeta) |i| {
        const li: usize = @intCast(paw.ae_wfc[i].l);
        const li_i32: i32 = @intCast(paw.ae_wfc[i].l);
        const ae_i = paw.ae_wfc[i].values;
        const ps_i = paw.ps_wfc[i].values;
        for (0..nbeta) |j| {
            const lj: usize = @intCast(paw.ae_wfc[j].l);
            const lj_i32: i32 = @intCast(paw.ae_wfc[j].l);
            const ae_j = paw.ae_wfc[j].values;
            const ps_j = paw.ps_wfc[j].values;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));

            // For each (L,M), contract rhoij with Gaunt
            for (0..lmax_aug + 1) |big_l| {
                const bl_i32: i32 = @intCast(big_l);
                var bm: i32 = -bl_i32;
                while (bm <= bl_i32) : (bm += 1) {
                    const lm_idx = GauntTable.lmIndex(big_l, bm);

                    // ρ_{ij,LM} = Σ_{mi,mj} ρ_{im,jm} × G(li,mi,lj,mj,L,M)
                    var rhoij_lm: f64 = 0.0;
                    var mi: i32 = -li_i32;
                    while (mi <= li_i32) : (mi += 1) {
                        const mi_idx = m_offsets[i] + @as(usize, @intCast(mi + li_i32));
                        var mj: i32 = -lj_i32;
                        while (mj <= lj_i32) : (mj += 1) {
                            const g_coeff = gaunt_table.get(li, mi, lj, mj, big_l, bm);
                            if (g_coeff == 0.0) continue;
                            const mj_idx = m_offsets[j] + @as(usize, @intCast(mj + lj_i32));
                            rhoij_lm += g_coeff * rhoij_m[mi_idx * m_total + mj_idx];
                        }
                    }
                    if (@abs(rhoij_lm) < 1e-30) continue;

                    // Add to AE density
                    for (0..n_r) |k| {
                        if (r[k] < 1e-10) continue;
                        const inv_r2 = 1.0 / (r[k] * r[k]);
                        rho_ae_lm[lm_idx][k] += rhoij_lm * ae_i[k] * ae_j[k] * inv_r2;
                    }

                    // Add to PS density (partial waves)
                    for (0..n_r) |k| {
                        if (r[k] < 1e-10) continue;
                        const inv_r2 = 1.0 / (r[k] * r[k]);
                        rho_ps_lm[lm_idx][k] += rhoij_lm * ps_i[k] * ps_j[k] * inv_r2;
                    }

                    // Add Q_ij^L contribution to PS density
                    if (findQijL(paw, i, j, big_l)) |q_vals| {
                        const n_q = @min(n_r, q_vals.len);
                        for (0..n_q) |k| {
                            if (r[k] < 1e-10) continue;
                            const inv_r2 = 1.0 / (r[k] * r[k]);
                            rho_ps_lm[lm_idx][k] += rhoij_lm * q_vals[k] * inv_r2;
                        }
                    }
                }
            }
        }
    }

    // Solve Poisson and compute energy for each L
    const vh = try alloc.alloc(f64, n_mesh);
    defer alloc.free(vh);

    var eh_ae: f64 = 0.0;
    var eh_ps: f64 = 0.0;

    for (0..lmax_aug + 1) |big_l| {
        const bl_i32: i32 = @intCast(big_l);
        var bm: i32 = -bl_i32;
        while (bm <= bl_i32) : (bm += 1) {
            const lm_idx = GauntTable.lmIndex(big_l, bm);

            // AE Hartree for this (L,M)
            radialHartreePotentialL(r, rab, rho_ae_lm[lm_idx], vh, n_mesh, big_l);
            for (0..n_mesh) |k| {
                const w = rab[k] * ctrapWeight(k, n_mesh);
                eh_ae += 0.5 * vh[k] * rho_ae_lm[lm_idx][k] * r[k] * r[k] * w;
            }

            // PS Hartree for this (L,M)
            radialHartreePotentialL(r, rab, rho_ps_lm[lm_idx], vh, n_mesh, big_l);
            for (0..n_mesh) |k| {
                const w = rab[k] * ctrapWeight(k, n_mesh);
                eh_ps += 0.5 * vh[k] * rho_ps_lm[lm_idx][k] * r[k] * r[k] * w;
            }
        }
    }

    return eh_ae - eh_ps;
}

/// Compute multi-L on-site Hartree D_ij for one PAW atom.
///
/// D^H_{im,jm} = Σ_{LM} G(li,mi,lj,mj,L,M) × ∫ [V^AE_{LM} u^AE_i u^AE_j - V^PS_{LM} (u^PS_i u^PS_j + Q^L)] dr
/// Output is m-resolved: dij_h[im*mt + jm].
pub fn computePawDijHartreeMultiL(
    alloc: std.mem.Allocator,
    dij_h: []f64,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    gaunt_table: *const GauntTable,
) !void {
    const nbeta = paw.number_of_proj;
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0) @min(n_mesh_full, paw.cutoff_r_index) else n_mesh_full;
    if (n_mesh > 4096) return error.MeshTooLarge;

    const lmax_aug = gaunt_table.lmax_aug;
    const n_lm_aug = (lmax_aug + 1) * (lmax_aug + 1);

    // Build density multipoles (same as computePawEhOnsiteMultiL)
    const rho_ae_lm = try alloc.alloc([]f64, n_lm_aug);
    defer {
        for (rho_ae_lm) |s| alloc.free(s);
        alloc.free(rho_ae_lm);
    }
    const rho_ps_lm = try alloc.alloc([]f64, n_lm_aug);
    defer {
        for (rho_ps_lm) |s| alloc.free(s);
        alloc.free(rho_ps_lm);
    }
    for (0..n_lm_aug) |lm| {
        rho_ae_lm[lm] = try alloc.alloc(f64, n_mesh);
        @memset(rho_ae_lm[lm], 0.0);
        rho_ps_lm[lm] = try alloc.alloc(f64, n_mesh);
        @memset(rho_ps_lm[lm], 0.0);
    }

    for (0..nbeta) |i| {
        const li: usize = @intCast(paw.ae_wfc[i].l);
        const li_i32: i32 = @intCast(paw.ae_wfc[i].l);
        const ae_i = paw.ae_wfc[i].values;
        const ps_i = paw.ps_wfc[i].values;
        for (0..nbeta) |j| {
            const lj: usize = @intCast(paw.ae_wfc[j].l);
            const lj_i32: i32 = @intCast(paw.ae_wfc[j].l);
            const ae_j = paw.ae_wfc[j].values;
            const ps_j = paw.ps_wfc[j].values;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));

            for (0..lmax_aug + 1) |big_l| {
                const bl_i32: i32 = @intCast(big_l);
                var bm: i32 = -bl_i32;
                while (bm <= bl_i32) : (bm += 1) {
                    const lm_idx = GauntTable.lmIndex(big_l, bm);
                    var rhoij_lm: f64 = 0.0;
                    var mi: i32 = -li_i32;
                    while (mi <= li_i32) : (mi += 1) {
                        const mi_idx = m_offsets[i] + @as(usize, @intCast(mi + li_i32));
                        var mj: i32 = -lj_i32;
                        while (mj <= lj_i32) : (mj += 1) {
                            const g_coeff = gaunt_table.get(li, mi, lj, mj, big_l, bm);
                            if (g_coeff == 0.0) continue;
                            const mj_idx = m_offsets[j] + @as(usize, @intCast(mj + lj_i32));
                            rhoij_lm += g_coeff * rhoij_m[mi_idx * m_total + mj_idx];
                        }
                    }
                    if (@abs(rhoij_lm) < 1e-30) continue;

                    for (0..n_r) |k| {
                        if (r[k] < 1e-10) continue;
                        const inv_r2 = 1.0 / (r[k] * r[k]);
                        rho_ae_lm[lm_idx][k] += rhoij_lm * ae_i[k] * ae_j[k] * inv_r2;
                        rho_ps_lm[lm_idx][k] += rhoij_lm * ps_i[k] * ps_j[k] * inv_r2;
                    }
                    if (findQijL(paw, i, j, big_l)) |q_vals| {
                        const n_q = @min(n_r, q_vals.len);
                        for (0..n_q) |k| {
                            if (r[k] < 1e-10) continue;
                            rho_ps_lm[lm_idx][k] += rhoij_lm * q_vals[k] / (r[k] * r[k]);
                        }
                    }
                }
            }
        }
    }

    // Solve Poisson for each (L,M) channel
    const vh_ae_lm = try alloc.alloc([]f64, n_lm_aug);
    defer {
        for (vh_ae_lm) |s| alloc.free(s);
        alloc.free(vh_ae_lm);
    }
    const vh_ps_lm = try alloc.alloc([]f64, n_lm_aug);
    defer {
        for (vh_ps_lm) |s| alloc.free(s);
        alloc.free(vh_ps_lm);
    }
    for (0..lmax_aug + 1) |big_l| {
        const bl_i32: i32 = @intCast(big_l);
        var bm: i32 = -bl_i32;
        while (bm <= bl_i32) : (bm += 1) {
            const lm_idx = GauntTable.lmIndex(big_l, bm);
            vh_ae_lm[lm_idx] = try alloc.alloc(f64, n_mesh);
            vh_ps_lm[lm_idx] = try alloc.alloc(f64, n_mesh);
            radialHartreePotentialL(r, rab, rho_ae_lm[lm_idx], vh_ae_lm[lm_idx], n_mesh, big_l);
            radialHartreePotentialL(r, rab, rho_ps_lm[lm_idx], vh_ps_lm[lm_idx], n_mesh, big_l);
        }
    }
    // Initialize unused lm entries to avoid undefined reads
    for (0..n_lm_aug) |lm| {
        // Check if this lm was allocated (belongs to a valid L)
        var valid = false;
        for (0..lmax_aug + 1) |big_l| {
            const bl_i32: i32 = @intCast(big_l);
            var bm: i32 = -bl_i32;
            while (bm <= bl_i32) : (bm += 1) {
                if (GauntTable.lmIndex(big_l, bm) == lm) {
                    valid = true;
                    break;
                }
            }
            if (valid) break;
        }
        if (!valid) {
            vh_ae_lm[lm] = try alloc.alloc(f64, n_mesh);
            @memset(vh_ae_lm[lm], 0.0);
            vh_ps_lm[lm] = try alloc.alloc(f64, n_mesh);
            @memset(vh_ps_lm[lm], 0.0);
        }
    }

    // Compute D^H_{im,jm} = Σ_{LM} G(li,mi,lj,mj,L,M) × ∫ [V^AE × u^AE_i u^AE_j - V^PS × (u^PS_i u^PS_j + Q^L)] dr
    @memset(dij_h, 0.0);
    for (0..nbeta) |i| {
        const li: usize = @intCast(paw.ae_wfc[i].l);
        const li_i32: i32 = @intCast(paw.ae_wfc[i].l);
        const ae_i = paw.ae_wfc[i].values;
        const ps_i = paw.ps_wfc[i].values;
        const mi_count: usize = 2 * li + 1;

        for (0..nbeta) |j| {
            const lj: usize = @intCast(paw.ae_wfc[j].l);
            const lj_i32: i32 = @intCast(paw.ae_wfc[j].l);
            const ae_j = paw.ae_wfc[j].values;
            const ps_j = paw.ps_wfc[j].values;
            const mj_count: usize = 2 * lj + 1;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));

            for (0..lmax_aug + 1) |big_l| {
                const bl_i32: i32 = @intCast(big_l);
                var bm: i32 = -bl_i32;
                while (bm <= bl_i32) : (bm += 1) {
                    const lm_idx = GauntTable.lmIndex(big_l, bm);

                    // Pre-compute radial integrals
                    var int_ae: f64 = 0.0;
                    var int_ps: f64 = 0.0;
                    for (0..n_r) |k| {
                        const w = rab[k] * ctrapWeight(k, n_mesh);
                        int_ae += vh_ae_lm[lm_idx][k] * ae_i[k] * ae_j[k] * w;
                        int_ps += vh_ps_lm[lm_idx][k] * ps_i[k] * ps_j[k] * w;
                    }
                    // Q contribution to PS integral
                    if (findQijL(paw, i, j, big_l)) |q_vals| {
                        const n_q = @min(n_r, q_vals.len);
                        for (0..n_q) |k| {
                            const w = rab[k] * ctrapWeight(k, n_mesh);
                            int_ps += vh_ps_lm[lm_idx][k] * q_vals[k] * w;
                        }
                    }

                    const int_diff = int_ae - int_ps;
                    if (@abs(int_diff) < 1e-30) continue;

                    // Distribute to all (mi,mj) pairs weighted by Gaunt
                    for (0..mi_count) |mi_u| {
                        const mi: i32 = @as(i32, @intCast(mi_u)) - li_i32;
                        const mi_idx = m_offsets[i] + mi_u;
                        for (0..mj_count) |mj_u| {
                            const mj: i32 = @as(i32, @intCast(mj_u)) - lj_i32;
                            const g_coeff = gaunt_table.get(li, mi, lj, mj, big_l, bm);
                            if (g_coeff == 0.0) continue;
                            const mj_idx = m_offsets[j] + mj_u;
                            dij_h[mi_idx * m_total + mj_idx] += g_coeff * int_diff;
                        }
                    }
                }
            }
        }
    }
}

/// Number of Lebedev angular points for on-site XC integration.
/// 110 points is exact for polynomials up to degree 17, sufficient for lmax=2.
const N_ANG: usize = 110;

/// Cartesian gradient of the solid harmonic S_{l,m}(x,y,z) evaluated on the unit sphere.
/// S_lm is the homogeneous polynomial of degree l such that S_lm|_{r=1} = Y_lm.
/// For the surface gradient formula ∇_S Y = ∇S - l×Y×r̂, we need ∇S (not ∇Y).
///
/// Key difference from ∇Y_lm: terms like (3z²-1) in Y_20 become (2z²-x²-y²) in S_20,
/// because S_20 = c0(3z²-r²) is homogeneous while Y_20 = c0(3z²-1) is not.
fn gradSolidHarmonic(l: i32, m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    const pi = std.math.pi;
    switch (l) {
        0 => return .{ 0.0, 0.0, 0.0 },
        1 => {
            // S_1m are already homogeneous: S = c*{x, y, z}
            const c = @sqrt(3.0 / (4.0 * pi));
            return switch (m) {
                -1 => .{ 0.0, c, 0.0 },
                0 => .{ 0.0, 0.0, c },
                1 => .{ c, 0.0, 0.0 },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        2 => {
            const c0 = @sqrt(5.0 / (16.0 * pi));
            const c1 = @sqrt(15.0 / (4.0 * pi));
            const c2 = @sqrt(15.0 / (16.0 * pi));
            return switch (m) {
                // S = c2*2xy (homogeneous)
                -2 => .{ 2.0 * c2 * ny, 2.0 * c2 * nx, 0.0 },
                // S = c1*yz (homogeneous)
                -1 => .{ 0.0, c1 * nz, c1 * ny },
                // S = c0*(2z²-x²-y²) [from 3z²-r² = 2z²-x²-y²]
                0 => .{ -2.0 * c0 * nx, -2.0 * c0 * ny, 4.0 * c0 * nz },
                // S = c1*xz (homogeneous)
                1 => .{ c1 * nz, 0.0, c1 * nx },
                // S = c2*(x²-y²) (homogeneous)
                2 => .{ 2.0 * c2 * nx, -2.0 * c2 * ny, 0.0 },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        3 => {
            return switch (m) {
                // S = c*(3x²y-y³) (homogeneous)
                -3 => blk: {
                    const c = @sqrt(35.0 / (32.0 * pi));
                    break :blk .{
                        c * 6.0 * nx * ny,
                        c * (3.0 * nx * nx - 3.0 * ny * ny),
                        0.0,
                    };
                },
                // S = c*xyz (homogeneous)
                -2 => blk: {
                    const c = @sqrt(105.0 / (4.0 * pi));
                    break :blk .{ c * ny * nz, c * nx * nz, c * nx * ny };
                },
                // S = c*y(4z²-x²-y²) [from y(5z²-r²)]
                -1 => blk: {
                    const c = @sqrt(21.0 / (32.0 * pi));
                    break :blk .{
                        -c * 2.0 * nx * ny,
                        c * (4.0 * nz * nz - nx * nx - 3.0 * ny * ny),
                        c * 8.0 * ny * nz,
                    };
                },
                // S = c*z(2z²-3x²-3y²) [from z(5z²-3r²)]
                0 => blk: {
                    const c = @sqrt(7.0 / (16.0 * pi));
                    break :blk .{
                        -c * 6.0 * nx * nz,
                        -c * 6.0 * ny * nz,
                        c * (6.0 * nz * nz - 3.0 * nx * nx - 3.0 * ny * ny),
                    };
                },
                // S = c*x(4z²-x²-y²) [from x(5z²-r²)]
                1 => blk: {
                    const c = @sqrt(21.0 / (32.0 * pi));
                    break :blk .{
                        c * (4.0 * nz * nz - 3.0 * nx * nx - ny * ny),
                        -c * 2.0 * nx * ny,
                        c * 8.0 * nx * nz,
                    };
                },
                // S = c*(x²-y²)*z (homogeneous)
                2 => blk: {
                    const c = @sqrt(105.0 / (16.0 * pi));
                    break :blk .{
                        2.0 * c * nx * nz,
                        -2.0 * c * ny * nz,
                        c * (nx * nx - ny * ny),
                    };
                },
                // S = c*(x³-3xy²) = c*x(x²-3y²) (homogeneous)
                3 => blk: {
                    const c = @sqrt(35.0 / (32.0 * pi));
                    break :blk .{
                        c * (3.0 * nx * nx - 3.0 * ny * ny),
                        -c * 6.0 * nx * ny,
                        0.0,
                    };
                },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        else => return .{ 0.0, 0.0, 0.0 },
    }
}

/// Surface gradient of real spherical harmonic on unit sphere.
/// ∇_S Y_{lm} = ∇S_{lm} - l × Y_{lm} × r̂
/// where S_lm is the solid harmonic (homogeneous polynomial of degree l).
/// Euler's theorem: r̂ · ∇S = l × S = l × Y on the unit sphere.
fn surfGradYlm(l: i32, m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    const grad = gradSolidHarmonic(l, m, nx, ny, nz);
    const ylm = nonlocal.realSphericalHarmonic(l, m, nx, ny, nz);
    const fl: f64 = @floatFromInt(l);
    return .{
        grad[0] - fl * ylm * nx,
        grad[1] - fl * ylm * ny,
        grad[2] - fl * ylm * nz,
    };
}

/// Surface gradient of real spherical harmonic on unit sphere (general l).
/// Uses analytical gradSolidHarmonic for l <= 3, numerical differentiation for l >= 4.
/// ∇_S Y_{lm} = ∇S_{lm} - l × Y_{lm} × r̂
fn surfGradYlmGeneral(l: i32, m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    if (l <= 3) return surfGradYlm(l, m, nx, ny, nz);

    // Numerical gradient of solid harmonic S_lm(r) = |r|^l × Y_lm(r̂)
    // ∇S is computed via central differences at the unit-sphere point n̂.
    const delta: f64 = 1e-5;
    const fl: f64 = @floatFromInt(l);
    const ylm = nonlocal.realSphericalHarmonic(l, m, nx, ny, nz);
    var grad_s: [3]f64 = undefined;
    const n = [3]f64{ nx, ny, nz };

    for (0..3) |d| {
        var rp = [3]f64{ nx, ny, nz };
        var rm = [3]f64{ nx, ny, nz };
        rp[d] += delta;
        rm[d] -= delta;

        const rp_len = @sqrt(rp[0] * rp[0] + rp[1] * rp[1] + rp[2] * rp[2]);
        const rm_len = @sqrt(rm[0] * rm[0] + rm[1] * rm[1] + rm[2] * rm[2]);

        const s_plus = std.math.pow(f64, rp_len, fl) *
            nonlocal.realSphericalHarmonic(l, m, rp[0] / rp_len, rp[1] / rp_len, rp[2] / rp_len);
        const s_minus = std.math.pow(f64, rm_len, fl) *
            nonlocal.realSphericalHarmonic(l, m, rm[0] / rm_len, rm[1] / rm_len, rm[2] / rm_len);

        grad_s[d] = (s_plus - s_minus) / (2.0 * delta);
    }

    return .{
        grad_s[0] - fl * ylm * n[0],
        grad_s[1] - fl * ylm * n[1],
        grad_s[2] - fl * ylm * n[2],
    };
}

/// Compute explicit V_xc(r) for GGA from density and radial gradient.
/// V_xc = df/dn - (1/r²) d/dr[r² × 2(df/dσ) × dρ/dr]
fn computeVxcRadial(
    rho: []const f64,
    drho: []const f64,
    r_grid: []const f64,
    _: []const f64, // rab_grid (unused, radialDerivative now uses r_grid directly)
    n: usize,
    xc_func: xc.Functional,
    vxc_out: []f64,
    h_work: []f64,
    r2h_work: []f64,
    dr2h_work: []f64,
) void {
    // Step 1: df/dn (LDA part) and h = 2(df/dσ)(dρ/dr)
    for (0..n) |k| {
        const n_val = @max(rho[k], 1e-30);
        const sigma = drho[k] * drho[k];
        const eval_pt = xc.evalPoint(xc_func, n_val, sigma);
        vxc_out[k] = eval_pt.df_dn;
        h_work[k] = 2.0 * eval_pt.df_dg2 * drho[k];
    }
    // Step 2: r² × h
    for (0..n) |k| {
        r2h_work[k] = r_grid[k] * r_grid[k] * h_work[k];
    }
    // Step 3: d(r²h)/dr
    radialDerivative(r2h_work, dr2h_work, r_grid, n);
    // Step 4: V_xc = df/dn - (1/r²) d(r²h)/dr
    for (0..n) |k| {
        if (r_grid[k] > 1e-10) {
            vxc_out[k] -= dr2h_work[k] / (r_grid[k] * r_grid[k]);
        }
    }
}

/// Compute PAW on-site D^xc using angular Lebedev quadrature (for GGA).
///
/// Full gradient D^xc with radial + angular contributions:
///   D^xc_{im,jm'} = Σ_α 4πw_α × {
///     Y_i Y_j × ∫[(df/dn)f_ij + 2(df/dσ)(∂ρ/∂r)f_ij'] r² dr
///     + Σ_d ∫[2(df/dσ)(∇_Sρ)_d × f_ij] dr × (Y_j(∇_SY_i)_d + Y_i(∇_SY_j)_d)
///   }   (AE - PS difference)
///
/// rhoij_m: m-resolved occupation matrix [(β,m_β),(β',m_β')] flattened
/// m_total: total number of m channels (Σ_β 2l_β+1)
/// m_offsets: m offset for each radial projector
pub fn computePawDijXcAngular(
    alloc: std.mem.Allocator,
    dij_xc_m: []f64,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    xc_func: xc.Functional,
    gaunt_table: *const GauntTable,
) !void {
    const nbeta = paw.number_of_proj;
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0) @min(n_mesh_full, paw.cutoff_r_index) else n_mesh_full;
    const grid = lebedev.getLebedevGrid(N_ANG);
    const n_ang = grid.len;
    const n_ij = nbeta * nbeta;
    const four_pi = 4.0 * std.math.pi;

    // Pre-compute Y_{l,m}(Ω_α) and ∇_S Y_{l,m}(Ω_α)
    const ylm_at = try alloc.alloc(f64, n_ang * m_total);
    defer alloc.free(ylm_at);
    const grad_ylm_at = try alloc.alloc([3]f64, n_ang * m_total);
    defer alloc.free(grad_ylm_at);
    for (grid, 0..) |pt, alpha| {
        for (0..nbeta) |b| {
            const l = paw.ae_wfc[b].l;
            const m_count = @as(usize, @intCast(2 * l + 1));
            for (0..m_count) |mi| {
                const m: i32 = @as(i32, @intCast(mi)) - l;
                const idx = alpha * m_total + m_offsets[b] + mi;
                ylm_at[idx] = nonlocal.realSphericalHarmonic(l, m, pt.x, pt.y, pt.z);
                grad_ylm_at[idx] = surfGradYlm(l, m, pt.x, pt.y, pt.z);
            }
        }
    }

    // Pre-compute u_i*u_j/r² and d(u_i*u_j/r²)/dr for AE and PS
    const uiuj_ae = try alloc.alloc([]f64, n_ij);
    defer {
        for (uiuj_ae) |s| alloc.free(s);
        alloc.free(uiuj_ae);
    }
    const duiuj_ae = try alloc.alloc([]f64, n_ij);
    defer {
        for (duiuj_ae) |s| alloc.free(s);
        alloc.free(duiuj_ae);
    }
    const uiuj_ps = try alloc.alloc([]f64, n_ij);
    defer {
        for (uiuj_ps) |s| alloc.free(s);
        alloc.free(uiuj_ps);
    }
    const duiuj_ps = try alloc.alloc([]f64, n_ij);
    defer {
        for (duiuj_ps) |s| alloc.free(s);
        alloc.free(duiuj_ps);
    }
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            const ae_buf = try alloc.alloc(f64, n_mesh);
            const ae_dbuf = try alloc.alloc(f64, n_mesh);
            const ps_buf = try alloc.alloc(f64, n_mesh);
            const ps_dbuf = try alloc.alloc(f64, n_mesh);
            const ae_i = paw.ae_wfc[i].values;
            const ae_j = paw.ae_wfc[j].values;
            const ps_i = paw.ps_wfc[i].values;
            const ps_j = paw.ps_wfc[j].values;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));
            for (0..n_r) |k| {
                if (r[k] < 1e-10) {
                    ae_buf[k] = 0.0;
                    ps_buf[k] = 0.0;
                } else {
                    ae_buf[k] = ae_i[k] * ae_j[k] / (r[k] * r[k]);
                    ps_buf[k] = ps_i[k] * ps_j[k] / (r[k] * r[k]);
                }
            }
            for (n_r..n_mesh) |k| {
                ae_buf[k] = 0.0;
                ps_buf[k] = 0.0;
            }
            radialDerivative(ae_buf, ae_dbuf, r, n_mesh);
            radialDerivative(ps_buf, ps_dbuf, r, n_mesh);
            uiuj_ae[i * nbeta + j] = ae_buf;
            duiuj_ae[i * nbeta + j] = ae_dbuf;
            uiuj_ps[i * nbeta + j] = ps_buf;
            duiuj_ps[i * nbeta + j] = ps_dbuf;
        }
    }

    // Pre-compute augmentation charge Q̂^L_ij/r² and d(Q̂^L_ij/r²)/dr for PS density
    const lmax_aug = paw.lmax_aug;
    const n_l_aug = lmax_aug + 1;
    const n_lm_aug = n_l_aug * n_l_aug;

    const aug_r2 = try alloc.alloc(?[]f64, n_ij * n_l_aug);
    defer {
        for (aug_r2) |maybe_buf| if (maybe_buf) |buf| alloc.free(buf);
        alloc.free(aug_r2);
    }
    const daug_r2 = try alloc.alloc(?[]f64, n_ij * n_l_aug);
    defer {
        for (daug_r2) |maybe_buf| if (maybe_buf) |buf| alloc.free(buf);
        alloc.free(daug_r2);
    }
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            for (0..n_l_aug) |big_l| {
                const flat_idx = i * nbeta * n_l_aug + j * n_l_aug + big_l;
                if (findQijL(paw, i, j, big_l)) |qvals| {
                    const buf = try alloc.alloc(f64, n_mesh);
                    const dbuf = try alloc.alloc(f64, n_mesh);
                    const n_q = @min(n_mesh, qvals.len);
                    for (0..n_q) |k| {
                        if (r[k] < 1e-10) {
                            buf[k] = 0.0;
                        } else {
                            buf[k] = qvals[k] / (r[k] * r[k]);
                        }
                    }
                    for (n_q..n_mesh) |k| buf[k] = 0.0;
                    radialDerivative(buf, dbuf, r, n_mesh);
                    aug_r2[flat_idx] = buf;
                    daug_r2[flat_idx] = dbuf;
                } else {
                    aug_r2[flat_idx] = null;
                    daug_r2[flat_idx] = null;
                }
            }
        }
    }

    // Pre-compute Y_LM and ∇_S Y_LM for augmentation channels
    const ylm_aug_at = try alloc.alloc(f64, n_ang * n_lm_aug);
    defer alloc.free(ylm_aug_at);
    const grad_ylm_aug_at = try alloc.alloc([3]f64, n_ang * n_lm_aug);
    defer alloc.free(grad_ylm_aug_at);
    for (grid, 0..) |leb_pt, alpha_idx| {
        for (0..n_l_aug) |big_l| {
            const bl_i32: i32 = @intCast(big_l);
            for (0..2 * big_l + 1) |bm_idx| {
                const big_m: i32 = @as(i32, @intCast(bm_idx)) - bl_i32;
                const lm_idx = big_l * big_l + bm_idx;
                const flat = alpha_idx * n_lm_aug + lm_idx;
                ylm_aug_at[flat] = nonlocal.realSphericalHarmonic(bl_i32, big_m, leb_pt.x, leb_pt.y, leb_pt.z);
                grad_ylm_aug_at[flat] = surfGradYlmGeneral(bl_i32, big_m, leb_pt.x, leb_pt.y, leb_pt.z);
            }
        }
    }

    // Pre-compute core density radial derivatives
    const dcore_ae = try alloc.alloc(f64, n_mesh);
    defer alloc.free(dcore_ae);
    const dcore_ps = try alloc.alloc(f64, n_mesh);
    defer alloc.free(dcore_ps);
    if (rho_core_ae) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);
        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radialDerivative(buf, dcore_ae, r, n_mesh);
    } else @memset(dcore_ae, 0.0);
    if (rho_core_ps) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);
        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radialDerivative(buf, dcore_ps, r, n_mesh);
    } else @memset(dcore_ps, 0.0);

    // Allocate work arrays
    const rho_ae = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ae);
    const rho_ps = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ps);
    const drho_ae = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ae);
    const drho_ps = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ps);
    const tmp_rho = try alloc.alloc(f64, n_mesh);
    defer alloc.free(tmp_rho);
    // Angular gradient of density (3 components per radial point)
    const grad_s_rho_ae = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(grad_s_rho_ae);
    const grad_s_rho_ps = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(grad_s_rho_ps);

    // Radial integrals I^rad_ij(α) and angular integrals I^ang_ij(α) [3-vector]
    const radial_integrals = try alloc.alloc(f64, n_ang * n_ij);
    defer alloc.free(radial_integrals);
    const angular_integrals = try alloc.alloc([3]f64, n_ang * n_ij);
    defer alloc.free(angular_integrals);

    // Augmentation radial/angular integrals: per (α, i, j, L)
    const aug_rad_integrals = try alloc.alloc(f64, n_ang * n_ij * n_l_aug);
    defer alloc.free(aug_rad_integrals);
    const aug_ang_integrals = try alloc.alloc([3]f64, n_ang * n_ij * n_l_aug);
    defer alloc.free(aug_ang_integrals);

    for (grid, 0..) |pt, alpha| {
        _ = pt;
        const ylm_base = alpha * m_total;

        // Build AE and PS densities and angular gradients at this angular point
        @memset(rho_ae, 0.0);
        @memset(rho_ps, 0.0);
        for (grad_s_rho_ae) |*g| g.* = .{ 0.0, 0.0, 0.0 };
        for (grad_s_rho_ps) |*g| g.* = .{ 0.0, 0.0, 0.0 };

        for (0..nbeta) |i| {
            const li = paw.ae_wfc[i].l;
            const li_u: usize = @intCast(li);
            const mi_count = @as(usize, @intCast(2 * li + 1));
            for (0..nbeta) |j| {
                const lj = paw.ae_wfc[j].l;
                const lj_u: usize = @intCast(lj);
                const mj_count = @as(usize, @intCast(2 * lj + 1));
                const ae_f = uiuj_ae[i * nbeta + j];
                const ps_f = uiuj_ps[i * nbeta + j];
                for (0..mi_count) |mi| {
                    const idx_i = ylm_base + m_offsets[i] + mi;
                    const yi = ylm_at[idx_i];
                    const gyi = grad_ylm_at[idx_i];
                    const mi_val: i32 = @as(i32, @intCast(mi)) - li;
                    for (0..mj_count) |mj| {
                        const idx_j = ylm_base + m_offsets[j] + mj;
                        const rij = rhoij_m[(m_offsets[i] + mi) * m_total + (m_offsets[j] + mj)];
                        if (@abs(rij) < 1e-30) continue;
                        const yj = ylm_at[idx_j];
                        const gyj = grad_ylm_at[idx_j];
                        const coeff = rij * yi * yj;
                        const grad_coeff: [3]f64 = .{
                            rij * (yj * gyi[0] + yi * gyj[0]),
                            rij * (yj * gyi[1] + yi * gyj[1]),
                            rij * (yj * gyi[2] + yi * gyj[2]),
                        };
                        for (0..n_mesh) |k| {
                            rho_ae[k] += coeff * ae_f[k];
                            rho_ps[k] += coeff * ps_f[k];
                            grad_s_rho_ae[k][0] += grad_coeff[0] * ae_f[k];
                            grad_s_rho_ae[k][1] += grad_coeff[1] * ae_f[k];
                            grad_s_rho_ae[k][2] += grad_coeff[2] * ae_f[k];
                            grad_s_rho_ps[k][0] += grad_coeff[0] * ps_f[k];
                            grad_s_rho_ps[k][1] += grad_coeff[1] * ps_f[k];
                            grad_s_rho_ps[k][2] += grad_coeff[2] * ps_f[k];
                        }

                        // Augmentation charge contribution to PS density
                        const mj_val: i32 = @as(i32, @intCast(mj)) - lj;
                        const l_min = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
                        const l_max = @min(li_u + lj_u, lmax_aug);
                        var big_l = l_min;
                        while (big_l <= l_max) : (big_l += 1) {
                            const aug_vals = aug_r2[i * nbeta * n_l_aug + j * n_l_aug + big_l] orelse continue;
                            const bl_i32: i32 = @intCast(big_l);
                            var big_m: i32 = -bl_i32;
                            while (big_m <= bl_i32) : (big_m += 1) {
                                const gaunt_val = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l, big_m);
                                if (@abs(gaunt_val) < 1e-30) continue;
                                const lm_aug = big_l * big_l + @as(usize, @intCast(@as(i64, @intCast(big_l)) + big_m));
                                const ylm_a = ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                const gylm_a = grad_ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                const aug_coeff = rij * gaunt_val * ylm_a;
                                for (0..n_mesh) |k| {
                                    rho_ps[k] += aug_coeff * aug_vals[k];
                                    grad_s_rho_ps[k][0] += rij * gaunt_val * gylm_a[0] * aug_vals[k];
                                    grad_s_rho_ps[k][1] += rij * gaunt_val * gylm_a[1] * aug_vals[k];
                                    grad_s_rho_ps[k][2] += rij * gaunt_val * gylm_a[2] * aug_vals[k];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compute radial derivative of valence density
        @memcpy(tmp_rho, rho_ae);
        radialDerivative(tmp_rho, drho_ae, r, n_mesh);
        @memcpy(tmp_rho, rho_ps);
        radialDerivative(tmp_rho, drho_ps, r, n_mesh);

        // Add core density and gradient (core is spherical: no angular gradient)
        for (0..n_mesh) |k| {
            if (rho_core_ae) |core| {
                if (k < core.len) {
                    rho_ae[k] += core[k];
                    drho_ae[k] += dcore_ae[k];
                }
            }
            if (rho_core_ps) |core| {
                if (k < core.len) {
                    rho_ps[k] += core[k];
                    drho_ps[k] += dcore_ps[k];
                }
            }
        }

        // Compute radial integrals I^rad_ij and angular integrals I^ang_ij for all (i,j)
        for (0..nbeta) |i| {
            for (0..nbeta) |j| {
                const ae_f = uiuj_ae[i * nbeta + j];
                const ae_df = duiuj_ae[i * nbeta + j];
                const ps_f = uiuj_ps[i * nbeta + j];
                const ps_df = duiuj_ps[i * nbeta + j];

                var sum_rad: f64 = 0.0;
                var sum_ang: [3]f64 = .{ 0.0, 0.0, 0.0 };
                for (0..n_mesh) |k| {
                    const r2 = r[k] * r[k];
                    const wk = rab[k] * ctrapWeight(k, n_mesh);

                    // AE contribution with full gradient sigma
                    const n_ae = @max(rho_ae[k], 1e-30);
                    const ang_sq_ae = grad_s_rho_ae[k][0] * grad_s_rho_ae[k][0] +
                        grad_s_rho_ae[k][1] * grad_s_rho_ae[k][1] +
                        grad_s_rho_ae[k][2] * grad_s_rho_ae[k][2];
                    const sigma_ae = drho_ae[k] * drho_ae[k] +
                        if (r2 > 1e-20) ang_sq_ae / r2 else 0.0;
                    const eval_ae = xc.evalPoint(xc_func, n_ae, sigma_ae);

                    // PS contribution with full gradient sigma
                    const n_ps = @max(rho_ps[k], 1e-30);
                    const ang_sq_ps = grad_s_rho_ps[k][0] * grad_s_rho_ps[k][0] +
                        grad_s_rho_ps[k][1] * grad_s_rho_ps[k][1] +
                        grad_s_rho_ps[k][2] * grad_s_rho_ps[k][2];
                    const sigma_ps = drho_ps[k] * drho_ps[k] +
                        if (r2 > 1e-20) ang_sq_ps / r2 else 0.0;
                    const eval_ps = xc.evalPoint(xc_func, n_ps, sigma_ps);

                    // Radial integral: (df/dn)f_ij + 2(df/dσ)(∂ρ/∂r)f_ij'
                    const ae_vxc_rad = eval_ae.df_dn * ae_f[k] + 2.0 * eval_ae.df_dg2 * drho_ae[k] * ae_df[k];
                    const ps_vxc_rad = eval_ps.df_dn * ps_f[k] + 2.0 * eval_ps.df_dg2 * drho_ps[k] * ps_df[k];
                    sum_rad += (ae_vxc_rad - ps_vxc_rad) * r2 * wk;

                    // Angular integral: 2(df/dσ)(∇_Sρ)_d × f_ij × dr
                    // Factor 1/r² from sigma cancels with r² from volume element
                    for (0..3) |d| {
                        const ae_vxc_ang = 2.0 * eval_ae.df_dg2 * grad_s_rho_ae[k][d] * ae_f[k];
                        const ps_vxc_ang = 2.0 * eval_ps.df_dg2 * grad_s_rho_ps[k][d] * ps_f[k];
                        sum_ang[d] += (ae_vxc_ang - ps_vxc_ang) * wk;
                    }
                }
                radial_integrals[alpha * n_ij + i * nbeta + j] = sum_rad;
                angular_integrals[alpha * n_ij + i * nbeta + j] = sum_ang;

                // Augmentation integrals: D^PS_aug uses Q̂^L as additional basis function
                // I^aug_rad_{ij,L} = ∫ [df/dn × Q̂^L/r² + 2(df/dσ)(∂ρ/∂r) d(Q̂^L/r²)/dr] r² dr
                // I^aug_ang_{ij,L}[d] = ∫ [2(df/dσ)(∇_Sρ)_d × Q̂^L/r²] dr
                // Note: these use PS eval (not AE-PS), and are SUBTRACTED in D_ij.
                for (0..n_l_aug) |big_l| {
                    const aug_flat = i * nbeta * n_l_aug + j * n_l_aug + big_l;
                    const aug_f = aug_r2[aug_flat] orelse {
                        aug_rad_integrals[(alpha * n_ij + i * nbeta + j) * n_l_aug + big_l] = 0.0;
                        aug_ang_integrals[(alpha * n_ij + i * nbeta + j) * n_l_aug + big_l] = .{ 0.0, 0.0, 0.0 };
                        continue;
                    };
                    const aug_df = daug_r2[aug_flat].?;
                    var a_rad: f64 = 0.0;
                    var a_ang: [3]f64 = .{ 0.0, 0.0, 0.0 };
                    for (0..n_mesh) |k| {
                        const r2 = r[k] * r[k];
                        const wk = rab[k] * ctrapWeight(k, n_mesh);
                        const n_ps = @max(rho_ps[k], 1e-30);
                        const ang_sq_ps = grad_s_rho_ps[k][0] * grad_s_rho_ps[k][0] +
                            grad_s_rho_ps[k][1] * grad_s_rho_ps[k][1] +
                            grad_s_rho_ps[k][2] * grad_s_rho_ps[k][2];
                        const sigma_ps = drho_ps[k] * drho_ps[k] +
                            if (r2 > 1e-20) ang_sq_ps / r2 else 0.0;
                        const eval_ps = xc.evalPoint(xc_func, n_ps, sigma_ps);

                        a_rad += (eval_ps.df_dn * aug_f[k] + 2.0 * eval_ps.df_dg2 * drho_ps[k] * aug_df[k]) * r2 * wk;
                        for (0..3) |d| {
                            a_ang[d] += 2.0 * eval_ps.df_dg2 * grad_s_rho_ps[k][d] * aug_f[k] * wk;
                        }
                    }
                    aug_rad_integrals[(alpha * n_ij + i * nbeta + j) * n_l_aug + big_l] = a_rad;
                    aug_ang_integrals[(alpha * n_ij + i * nbeta + j) * n_l_aug + big_l] = a_ang;
                }
            }
        }
    }

    // Accumulate m-resolved D^xc from radial + angular integrals
    @memset(dij_xc_m[0 .. m_total * m_total], 0.0);

    for (grid, 0..) |pt, alpha| {
        const w_ang = pt.w * four_pi;
        const ylm_base = alpha * m_total;

        for (0..nbeta) |i| {
            const li = paw.ae_wfc[i].l;
            const li_u: usize = @intCast(li);
            const mi_count = @as(usize, @intCast(2 * li + 1));
            for (0..nbeta) |j| {
                const lj = paw.ae_wfc[j].l;
                const lj_u: usize = @intCast(lj);
                const mj_count = @as(usize, @intCast(2 * lj + 1));
                const I_rad = radial_integrals[alpha * n_ij + i * nbeta + j];
                const I_ang = angular_integrals[alpha * n_ij + i * nbeta + j];

                for (0..mi_count) |mi| {
                    const idx_i = ylm_base + m_offsets[i] + mi;
                    const yi = ylm_at[idx_i];
                    const gyi = grad_ylm_at[idx_i];
                    const im = m_offsets[i] + mi;
                    const mi_val: i32 = @as(i32, @intCast(mi)) - li;
                    for (0..mj_count) |mj| {
                        const idx_j = ylm_base + m_offsets[j] + mj;
                        const yj = ylm_at[idx_j];
                        const gyj = grad_ylm_at[idx_j];
                        const jm = m_offsets[j] + mj;
                        const mj_val: i32 = @as(i32, @intCast(mj)) - lj;
                        // Radial contribution: Y_i Y_j × I^rad
                        var contrib = w_ang * yi * yj * I_rad;
                        // Angular contribution: I^ang · (Y_j ∇_S Y_i + Y_i ∇_S Y_j)
                        for (0..3) |d| {
                            contrib += w_ang * I_ang[d] * (yj * gyi[d] + yi * gyj[d]);
                        }
                        // Augmentation correction (subtracted because part of PS contribution)
                        const l_min_aug = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
                        const l_max_aug = @min(li_u + lj_u, lmax_aug);
                        var big_l = l_min_aug;
                        while (big_l <= l_max_aug) : (big_l += 1) {
                            const aug_base = (alpha * n_ij + i * nbeta + j) * n_l_aug + big_l;
                            const I_aug_rad = aug_rad_integrals[aug_base];
                            const I_aug_ang = aug_ang_integrals[aug_base];
                            if (@abs(I_aug_rad) < 1e-30 and @abs(I_aug_ang[0]) < 1e-30 and
                                @abs(I_aug_ang[1]) < 1e-30 and @abs(I_aug_ang[2]) < 1e-30) continue;
                            const bl_i32: i32 = @intCast(big_l);
                            var big_m: i32 = -bl_i32;
                            while (big_m <= bl_i32) : (big_m += 1) {
                                const gaunt_val = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l, big_m);
                                if (@abs(gaunt_val) < 1e-30) continue;
                                const lm_aug = big_l * big_l + @as(usize, @intCast(@as(i64, @intCast(big_l)) + big_m));
                                const ylm_a = ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                const gylm_a = grad_ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                // Subtract PS augmentation contribution
                                contrib -= w_ang * gaunt_val * ylm_a * I_aug_rad;
                                for (0..3) |d| {
                                    contrib -= w_ang * gaunt_val * gylm_a[d] * I_aug_ang[d];
                                }
                            }
                        }
                        dij_xc_m[im * m_total + jm] += contrib;
                    }
                }
            }
        }
    }
}

/// Compute PAW on-site XC energy with angular Lebedev quadrature (for GGA).
///
/// E_xc = ∫ f_xc(ρ, σ) d³r using Lebedev angular × radial integration.
/// Full gradient: σ = (∂ρ/∂r)² + |∇_S ρ|²/r²
/// where ∇_S ρ is the surface gradient on the unit sphere.
///
/// PS density includes augmentation charges Q̂^L_ij for consistency with
/// augmented PW density (ρ̃ + n̂). This is the QE "newd" convention.
pub fn computePawExcOnsiteAngular(
    alloc: std.mem.Allocator,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    xc_func: xc.Functional,
    gaunt_table: *const GauntTable,
) !f64 {
    const nbeta = paw.number_of_proj;
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0) @min(n_mesh_full, paw.cutoff_r_index) else n_mesh_full;
    const grid = lebedev.getLebedevGrid(N_ANG);
    const n_ang = grid.len;
    const four_pi = 4.0 * std.math.pi;

    // Pre-compute Y_{l,m}(Ω_α) and ∇_S Y_{l,m}(Ω_α) for all projector channels
    const ylm_at = try alloc.alloc(f64, n_ang * m_total);
    defer alloc.free(ylm_at);
    const grad_ylm_at = try alloc.alloc([3]f64, n_ang * m_total);
    defer alloc.free(grad_ylm_at);
    for (grid, 0..) |pt, alpha| {
        for (0..nbeta) |b| {
            const l = paw.ae_wfc[b].l;
            const m_count = @as(usize, @intCast(2 * l + 1));
            for (0..m_count) |mi| {
                const m: i32 = @as(i32, @intCast(mi)) - l;
                const idx = alpha * m_total + m_offsets[b] + mi;
                ylm_at[idx] = nonlocal.realSphericalHarmonic(l, m, pt.x, pt.y, pt.z);
                grad_ylm_at[idx] = surfGradYlm(l, m, pt.x, pt.y, pt.z);
            }
        }
    }

    // Pre-compute u_i*u_j/r² for density
    const n_ij = nbeta * nbeta;
    const ae_uiuj = try alloc.alloc([]f64, n_ij);
    defer {
        for (ae_uiuj) |s| alloc.free(s);
        alloc.free(ae_uiuj);
    }
    const ps_uiuj = try alloc.alloc([]f64, n_ij);
    defer {
        for (ps_uiuj) |s| alloc.free(s);
        alloc.free(ps_uiuj);
    }
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            const ae_buf = try alloc.alloc(f64, n_mesh);
            const ps_buf = try alloc.alloc(f64, n_mesh);
            const ae_i = paw.ae_wfc[i].values;
            const ae_j = paw.ae_wfc[j].values;
            const ps_i = paw.ps_wfc[i].values;
            const ps_j = paw.ps_wfc[j].values;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));
            for (0..n_r) |k| {
                if (r[k] < 1e-10) {
                    ae_buf[k] = 0.0;
                    ps_buf[k] = 0.0;
                } else {
                    ae_buf[k] = ae_i[k] * ae_j[k] / (r[k] * r[k]);
                    ps_buf[k] = ps_i[k] * ps_j[k] / (r[k] * r[k]);
                }
            }
            for (n_r..n_mesh) |k| {
                ae_buf[k] = 0.0;
                ps_buf[k] = 0.0;
            }
            ae_uiuj[i * nbeta + j] = ae_buf;
            ps_uiuj[i * nbeta + j] = ps_buf;
        }
    }

    // Pre-compute augmentation charge Q̂^L_ij/r² for PS density
    // aug_uiuj[i * nbeta + j] is an array of (L, values) pairs
    const lmax_aug = paw.lmax_aug;
    const n_l_aug = lmax_aug + 1;
    const n_lm_aug = n_l_aug * n_l_aug;

    // Flat lookup: aug_r2[i * nbeta * n_l_aug + j * n_l_aug + L] = Q̂^L_ij/r² or null
    const aug_r2 = try alloc.alloc(?[]f64, n_ij * n_l_aug);
    defer {
        for (aug_r2) |maybe_buf| if (maybe_buf) |buf| alloc.free(buf);
        alloc.free(aug_r2);
    }
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            for (0..n_l_aug) |big_l| {
                const flat_idx = i * nbeta * n_l_aug + j * n_l_aug + big_l;
                if (findQijL(paw, i, j, big_l)) |qvals| {
                    const buf = try alloc.alloc(f64, n_mesh);
                    const n_q = @min(n_mesh, qvals.len);
                    for (0..n_q) |k| {
                        if (r[k] < 1e-10) {
                            buf[k] = 0.0;
                        } else {
                            buf[k] = qvals[k] / (r[k] * r[k]);
                        }
                    }
                    for (n_q..n_mesh) |k| buf[k] = 0.0;
                    aug_r2[flat_idx] = buf;
                } else {
                    aug_r2[flat_idx] = null;
                }
            }
        }
    }

    // Pre-compute Y_LM(Ω_α) and ∇_S Y_LM(Ω_α) for augmentation L,M channels
    const ylm_aug_at = try alloc.alloc(f64, n_ang * n_lm_aug);
    defer alloc.free(ylm_aug_at);
    const grad_ylm_aug_at = try alloc.alloc([3]f64, n_ang * n_lm_aug);
    defer alloc.free(grad_ylm_aug_at);
    for (grid, 0..) |leb_pt, alpha_idx| {
        for (0..n_l_aug) |big_l| {
            const bl_i32: i32 = @intCast(big_l);
            for (0..2 * big_l + 1) |bm_idx| {
                const big_m: i32 = @as(i32, @intCast(bm_idx)) - bl_i32;
                const lm_idx = big_l * big_l + bm_idx;
                const flat = alpha_idx * n_lm_aug + lm_idx;
                ylm_aug_at[flat] = nonlocal.realSphericalHarmonic(bl_i32, big_m, leb_pt.x, leb_pt.y, leb_pt.z);
                grad_ylm_aug_at[flat] = surfGradYlmGeneral(bl_i32, big_m, leb_pt.x, leb_pt.y, leb_pt.z);
            }
        }
    }

    // Pre-compute core density derivatives
    const dcore_ae = try alloc.alloc(f64, n_mesh);
    defer alloc.free(dcore_ae);
    const dcore_ps = try alloc.alloc(f64, n_mesh);
    defer alloc.free(dcore_ps);
    if (rho_core_ae) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);
        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radialDerivative(buf, dcore_ae, r, n_mesh);
    } else @memset(dcore_ae, 0.0);
    if (rho_core_ps) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);
        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radialDerivative(buf, dcore_ps, r, n_mesh);
    } else @memset(dcore_ps, 0.0);

    // Allocate work arrays
    const rho_ae = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ae);
    const rho_ps = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ps);
    const drho_ae = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ae);
    const drho_ps = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ps);
    // Angular gradient components (3 Cartesian components per radial point)
    const grad_s_rho_ae = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(grad_s_rho_ae);
    const grad_s_rho_ps = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(grad_s_rho_ps);

    var exc_ae: f64 = 0.0;
    var exc_ps: f64 = 0.0;

    for (grid, 0..) |pt, alpha| {
        _ = pt;
        const w_ang = grid[alpha].w * four_pi;
        const ylm_base = alpha * m_total;

        // Build densities, radial gradients, and angular gradients at this angular point
        @memset(rho_ae, 0.0);
        @memset(rho_ps, 0.0);
        for (grad_s_rho_ae) |*g| g.* = .{ 0.0, 0.0, 0.0 };
        for (grad_s_rho_ps) |*g| g.* = .{ 0.0, 0.0, 0.0 };

        for (0..nbeta) |i| {
            const li = paw.ae_wfc[i].l;
            const li_u: usize = @intCast(li);
            const mi_count = @as(usize, @intCast(2 * li + 1));
            for (0..nbeta) |j| {
                const lj = paw.ae_wfc[j].l;
                const lj_u: usize = @intCast(lj);
                const mj_count = @as(usize, @intCast(2 * lj + 1));
                const ae_f = ae_uiuj[i * nbeta + j];
                const ps_f = ps_uiuj[i * nbeta + j];
                for (0..mi_count) |mi| {
                    const idx_i = ylm_base + m_offsets[i] + mi;
                    const yi = ylm_at[idx_i];
                    const gyi = grad_ylm_at[idx_i];
                    const mi_val: i32 = @as(i32, @intCast(mi)) - li;
                    for (0..mj_count) |mj| {
                        const idx_j = ylm_base + m_offsets[j] + mj;
                        const rij = rhoij_m[(m_offsets[i] + mi) * m_total + (m_offsets[j] + mj)];
                        if (@abs(rij) < 1e-30) continue;
                        const yj = ylm_at[idx_j];
                        const gyj = grad_ylm_at[idx_j];
                        const coeff = rij * yi * yj;
                        // ∇_S(Y_i Y_j) = Y_j ∇_S Y_i + Y_i ∇_S Y_j
                        const grad_coeff: [3]f64 = .{
                            rij * (yj * gyi[0] + yi * gyj[0]),
                            rij * (yj * gyi[1] + yi * gyj[1]),
                            rij * (yj * gyi[2] + yi * gyj[2]),
                        };
                        for (0..n_mesh) |k| {
                            rho_ae[k] += coeff * ae_f[k];
                            rho_ps[k] += coeff * ps_f[k];
                            grad_s_rho_ae[k][0] += grad_coeff[0] * ae_f[k];
                            grad_s_rho_ae[k][1] += grad_coeff[1] * ae_f[k];
                            grad_s_rho_ae[k][2] += grad_coeff[2] * ae_f[k];
                            grad_s_rho_ps[k][0] += grad_coeff[0] * ps_f[k];
                            grad_s_rho_ps[k][1] += grad_coeff[1] * ps_f[k];
                            grad_s_rho_ps[k][2] += grad_coeff[2] * ps_f[k];
                        }

                        // Augmentation charge contribution to PS density:
                        // ρ_aug = Σ_LM G(li,mi,lj,mj,L,M) × Y_LM(Ω) × Q̂^L_ij/r²
                        const mj_val: i32 = @as(i32, @intCast(mj)) - lj;
                        const l_min = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
                        const l_max = @min(li_u + lj_u, lmax_aug);
                        var big_l = l_min;
                        while (big_l <= l_max) : (big_l += 1) {
                            const aug_vals = aug_r2[i * nbeta * n_l_aug + j * n_l_aug + big_l] orelse continue;
                            const bl_i32: i32 = @intCast(big_l);
                            var big_m: i32 = -bl_i32;
                            while (big_m <= bl_i32) : (big_m += 1) {
                                const gaunt_val = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l, big_m);
                                if (@abs(gaunt_val) < 1e-30) continue;
                                const lm_aug = big_l * big_l + @as(usize, @intCast(@as(i64, @intCast(big_l)) + big_m));
                                const ylm_a = ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                const gylm_a = grad_ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                const aug_coeff = rij * gaunt_val * ylm_a;
                                for (0..n_mesh) |k| {
                                    rho_ps[k] += aug_coeff * aug_vals[k];
                                    grad_s_rho_ps[k][0] += rij * gaunt_val * gylm_a[0] * aug_vals[k];
                                    grad_s_rho_ps[k][1] += rij * gaunt_val * gylm_a[1] * aug_vals[k];
                                    grad_s_rho_ps[k][2] += rij * gaunt_val * gylm_a[2] * aug_vals[k];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compute radial gradient of valence density
        {
            const tmp_ae = try alloc.alloc(f64, n_mesh);
            defer alloc.free(tmp_ae);
            const tmp_ps = try alloc.alloc(f64, n_mesh);
            defer alloc.free(tmp_ps);
            @memcpy(tmp_ae, rho_ae);
            @memcpy(tmp_ps, rho_ps);
            radialDerivative(tmp_ae, drho_ae, r, n_mesh);
            radialDerivative(tmp_ps, drho_ps, r, n_mesh);
        }

        // Add core density and gradient (core is spherical: no angular gradient)
        for (0..n_mesh) |k| {
            if (rho_core_ae) |core| {
                if (k < core.len) {
                    rho_ae[k] += core[k];
                    drho_ae[k] += dcore_ae[k];
                }
            }
            if (rho_core_ps) |core| {
                if (k < core.len) {
                    rho_ps[k] += core[k];
                    drho_ps[k] += dcore_ps[k];
                }
            }
        }

        // Integrate energy with full gradient σ = (∂ρ/∂r)² + |∇_S ρ|²/r²
        for (0..n_mesh) |k| {
            const r2 = r[k] * r[k];
            const wk = rab[k] * ctrapWeight(k, n_mesh);

            const n_ae = @max(rho_ae[k], 1e-30);
            const ang_sq_ae = grad_s_rho_ae[k][0] * grad_s_rho_ae[k][0] +
                grad_s_rho_ae[k][1] * grad_s_rho_ae[k][1] +
                grad_s_rho_ae[k][2] * grad_s_rho_ae[k][2];
            const sigma_ae = drho_ae[k] * drho_ae[k] +
                if (r2 > 1e-20) ang_sq_ae / r2 else 0.0;
            const eval_ae = xc.evalPoint(xc_func, n_ae, sigma_ae);
            exc_ae += w_ang * eval_ae.f * r2 * wk;

            const n_ps = @max(rho_ps[k], 1e-30);
            const ang_sq_ps = grad_s_rho_ps[k][0] * grad_s_rho_ps[k][0] +
                grad_s_rho_ps[k][1] * grad_s_rho_ps[k][1] +
                grad_s_rho_ps[k][2] * grad_s_rho_ps[k][2];
            const sigma_ps = drho_ps[k] * drho_ps[k] +
                if (r2 > 1e-20) ang_sq_ps / r2 else 0.0;
            const eval_ps = xc.evalPoint(xc_func, n_ps, sigma_ps);
            exc_ps += w_ang * eval_ps.f * r2 * wk;
        }
    }
    return exc_ae - exc_ps;
}

/// Spin-polarized PAW D^xc: compute D^xc_up and D^xc_down from rhoij_up/down.
/// Core density (spherical) is split equally between up and down.
/// For non-magnetic input (rhoij_up == rhoij_down == total/2), this reduces
/// to the same result as the non-spin version applied identically to both channels.
pub fn computePawDijXcAngularSpin(
    alloc: std.mem.Allocator,
    dij_xc_m_up: []f64,
    dij_xc_m_down: []f64,
    paw: PawData,
    rhoij_m_up: []const f64,
    rhoij_m_down: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    xc_func: xc.Functional,
    gaunt_table: *const GauntTable,
) !void {
    const nbeta = paw.number_of_proj;
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0) @min(n_mesh_full, paw.cutoff_r_index) else n_mesh_full;
    const grid = lebedev.getLebedevGrid(N_ANG);
    const n_ang = grid.len;
    const n_ij = nbeta * nbeta;
    const four_pi = 4.0 * std.math.pi;

    // Pre-compute Y_{l,m}(Ω_α) and ∇_S Y_{l,m}(Ω_α)
    const ylm_at = try alloc.alloc(f64, n_ang * m_total);
    defer alloc.free(ylm_at);
    const grad_ylm_at = try alloc.alloc([3]f64, n_ang * m_total);
    defer alloc.free(grad_ylm_at);
    for (grid, 0..) |pt, alpha| {
        for (0..nbeta) |b| {
            const l = paw.ae_wfc[b].l;
            const m_count = @as(usize, @intCast(2 * l + 1));
            for (0..m_count) |mi| {
                const m: i32 = @as(i32, @intCast(mi)) - l;
                const idx = alpha * m_total + m_offsets[b] + mi;
                ylm_at[idx] = nonlocal.realSphericalHarmonic(l, m, pt.x, pt.y, pt.z);
                grad_ylm_at[idx] = surfGradYlm(l, m, pt.x, pt.y, pt.z);
            }
        }
    }

    // Pre-compute u_i*u_j/r² and derivatives
    const uiuj_ae = try alloc.alloc([]f64, n_ij);
    defer {
        for (uiuj_ae) |s| alloc.free(s);
        alloc.free(uiuj_ae);
    }
    const duiuj_ae = try alloc.alloc([]f64, n_ij);
    defer {
        for (duiuj_ae) |s| alloc.free(s);
        alloc.free(duiuj_ae);
    }
    const uiuj_ps = try alloc.alloc([]f64, n_ij);
    defer {
        for (uiuj_ps) |s| alloc.free(s);
        alloc.free(uiuj_ps);
    }
    const duiuj_ps = try alloc.alloc([]f64, n_ij);
    defer {
        for (duiuj_ps) |s| alloc.free(s);
        alloc.free(duiuj_ps);
    }
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            const ae_buf = try alloc.alloc(f64, n_mesh);
            const ae_dbuf = try alloc.alloc(f64, n_mesh);
            const ps_buf = try alloc.alloc(f64, n_mesh);
            const ps_dbuf = try alloc.alloc(f64, n_mesh);
            const ae_i = paw.ae_wfc[i].values;
            const ae_j = paw.ae_wfc[j].values;
            const ps_i = paw.ps_wfc[i].values;
            const ps_j = paw.ps_wfc[j].values;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));
            for (0..n_r) |k| {
                if (r[k] < 1e-10) {
                    ae_buf[k] = 0.0;
                    ps_buf[k] = 0.0;
                } else {
                    ae_buf[k] = ae_i[k] * ae_j[k] / (r[k] * r[k]);
                    ps_buf[k] = ps_i[k] * ps_j[k] / (r[k] * r[k]);
                }
            }
            for (n_r..n_mesh) |k| {
                ae_buf[k] = 0.0;
                ps_buf[k] = 0.0;
            }
            radialDerivative(ae_buf, ae_dbuf, r, n_mesh);
            radialDerivative(ps_buf, ps_dbuf, r, n_mesh);
            uiuj_ae[i * nbeta + j] = ae_buf;
            duiuj_ae[i * nbeta + j] = ae_dbuf;
            uiuj_ps[i * nbeta + j] = ps_buf;
            duiuj_ps[i * nbeta + j] = ps_dbuf;
        }
    }

    // Pre-compute augmentation charge Q̂^L_ij/r² and derivatives
    const lmax_aug = paw.lmax_aug;
    const n_l_aug = lmax_aug + 1;
    const n_lm_aug = n_l_aug * n_l_aug;

    const aug_r2 = try alloc.alloc(?[]f64, n_ij * n_l_aug);
    defer {
        for (aug_r2) |maybe_buf| if (maybe_buf) |buf| alloc.free(buf);
        alloc.free(aug_r2);
    }
    const daug_r2 = try alloc.alloc(?[]f64, n_ij * n_l_aug);
    defer {
        for (daug_r2) |maybe_buf| if (maybe_buf) |buf| alloc.free(buf);
        alloc.free(daug_r2);
    }
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            for (0..n_l_aug) |big_l| {
                const flat_idx = i * nbeta * n_l_aug + j * n_l_aug + big_l;
                if (findQijL(paw, i, j, big_l)) |qvals| {
                    const buf = try alloc.alloc(f64, n_mesh);
                    const dbuf = try alloc.alloc(f64, n_mesh);
                    const n_q = @min(n_mesh, qvals.len);
                    for (0..n_q) |k| {
                        if (r[k] < 1e-10) {
                            buf[k] = 0.0;
                        } else {
                            buf[k] = qvals[k] / (r[k] * r[k]);
                        }
                    }
                    for (n_q..n_mesh) |k| buf[k] = 0.0;
                    radialDerivative(buf, dbuf, r, n_mesh);
                    aug_r2[flat_idx] = buf;
                    daug_r2[flat_idx] = dbuf;
                } else {
                    aug_r2[flat_idx] = null;
                    daug_r2[flat_idx] = null;
                }
            }
        }
    }

    // Pre-compute Y_LM and ∇_S Y_LM for augmentation channels
    const ylm_aug_at = try alloc.alloc(f64, n_ang * n_lm_aug);
    defer alloc.free(ylm_aug_at);
    const grad_ylm_aug_at = try alloc.alloc([3]f64, n_ang * n_lm_aug);
    defer alloc.free(grad_ylm_aug_at);
    for (grid, 0..) |leb_pt, alpha_idx| {
        for (0..n_l_aug) |big_l| {
            const bl_i32: i32 = @intCast(big_l);
            for (0..2 * big_l + 1) |bm_idx| {
                const big_m: i32 = @as(i32, @intCast(bm_idx)) - bl_i32;
                const lm_idx = big_l * big_l + bm_idx;
                const flat = alpha_idx * n_lm_aug + lm_idx;
                ylm_aug_at[flat] = nonlocal.realSphericalHarmonic(bl_i32, big_m, leb_pt.x, leb_pt.y, leb_pt.z);
                grad_ylm_aug_at[flat] = surfGradYlmGeneral(bl_i32, big_m, leb_pt.x, leb_pt.y, leb_pt.z);
            }
        }
    }

    // Pre-compute core density radial derivatives
    const dcore_ae = try alloc.alloc(f64, n_mesh);
    defer alloc.free(dcore_ae);
    const dcore_ps = try alloc.alloc(f64, n_mesh);
    defer alloc.free(dcore_ps);
    if (rho_core_ae) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);
        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radialDerivative(buf, dcore_ae, r, n_mesh);
    } else @memset(dcore_ae, 0.0);
    if (rho_core_ps) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);
        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radialDerivative(buf, dcore_ps, r, n_mesh);
    } else @memset(dcore_ps, 0.0);

    // Work arrays: per-spin densities and gradients
    const rho_ae_up = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ae_up);
    const rho_ae_down = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ae_down);
    const rho_ps_up = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ps_up);
    const rho_ps_down = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ps_down);
    const drho_ae_up = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ae_up);
    const drho_ae_down = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ae_down);
    const drho_ps_up = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ps_up);
    const drho_ps_down = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ps_down);
    const tmp_rho = try alloc.alloc(f64, n_mesh);
    defer alloc.free(tmp_rho);
    const grad_s_ae_up = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(grad_s_ae_up);
    const grad_s_ae_down = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(grad_s_ae_down);
    const grad_s_ps_up = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(grad_s_ps_up);
    const grad_s_ps_down = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(grad_s_ps_down);

    // Radial integrals per (alpha, i, j) for up and down
    const rad_int_up = try alloc.alloc(f64, n_ang * n_ij);
    defer alloc.free(rad_int_up);
    const rad_int_down = try alloc.alloc(f64, n_ang * n_ij);
    defer alloc.free(rad_int_down);
    const ang_int_up = try alloc.alloc([3]f64, n_ang * n_ij);
    defer alloc.free(ang_int_up);
    const ang_int_down = try alloc.alloc([3]f64, n_ang * n_ij);
    defer alloc.free(ang_int_down);

    // Augmentation integrals per (alpha, i, j, L) for up and down
    const aug_rad_up = try alloc.alloc(f64, n_ang * n_ij * n_l_aug);
    defer alloc.free(aug_rad_up);
    const aug_rad_down = try alloc.alloc(f64, n_ang * n_ij * n_l_aug);
    defer alloc.free(aug_rad_down);
    const aug_ang_up = try alloc.alloc([3]f64, n_ang * n_ij * n_l_aug);
    defer alloc.free(aug_ang_up);
    const aug_ang_down = try alloc.alloc([3]f64, n_ang * n_ij * n_l_aug);
    defer alloc.free(aug_ang_down);

    for (grid, 0..) |_, alpha| {
        const ylm_base = alpha * m_total;

        // Build spin-resolved AE and PS densities at this angular point
        @memset(rho_ae_up, 0.0);
        @memset(rho_ae_down, 0.0);
        @memset(rho_ps_up, 0.0);
        @memset(rho_ps_down, 0.0);
        for (grad_s_ae_up) |*g| g.* = .{ 0.0, 0.0, 0.0 };
        for (grad_s_ae_down) |*g| g.* = .{ 0.0, 0.0, 0.0 };
        for (grad_s_ps_up) |*g| g.* = .{ 0.0, 0.0, 0.0 };
        for (grad_s_ps_down) |*g| g.* = .{ 0.0, 0.0, 0.0 };

        for (0..nbeta) |i| {
            const li = paw.ae_wfc[i].l;
            const li_u: usize = @intCast(li);
            const mi_count = @as(usize, @intCast(2 * li + 1));
            for (0..nbeta) |j| {
                const lj = paw.ae_wfc[j].l;
                const lj_u: usize = @intCast(lj);
                const mj_count = @as(usize, @intCast(2 * lj + 1));
                const ae_f = uiuj_ae[i * nbeta + j];
                const ps_f = uiuj_ps[i * nbeta + j];
                for (0..mi_count) |mi| {
                    const idx_i = ylm_base + m_offsets[i] + mi;
                    const yi = ylm_at[idx_i];
                    const gyi = grad_ylm_at[idx_i];
                    const mi_val: i32 = @as(i32, @intCast(mi)) - li;
                    for (0..mj_count) |mj| {
                        const idx_j = ylm_base + m_offsets[j] + mj;
                        const yj = ylm_at[idx_j];
                        const gyj = grad_ylm_at[idx_j];
                        const im = m_offsets[i] + mi;
                        const jm = m_offsets[j] + mj;
                        const rij_up = rhoij_m_up[im * m_total + jm];
                        const rij_dn = rhoij_m_down[im * m_total + jm];
                        const mj_val: i32 = @as(i32, @intCast(mj)) - lj;

                        // Up spin density
                        if (@abs(rij_up) > 1e-30) {
                            const coeff = rij_up * yi * yj;
                            const gc: [3]f64 = .{
                                rij_up * (yj * gyi[0] + yi * gyj[0]),
                                rij_up * (yj * gyi[1] + yi * gyj[1]),
                                rij_up * (yj * gyi[2] + yi * gyj[2]),
                            };
                            for (0..n_mesh) |k| {
                                rho_ae_up[k] += coeff * ae_f[k];
                                rho_ps_up[k] += coeff * ps_f[k];
                                inline for (0..3) |d| {
                                    grad_s_ae_up[k][d] += gc[d] * ae_f[k];
                                    grad_s_ps_up[k][d] += gc[d] * ps_f[k];
                                }
                            }
                            // Augmentation for PS up
                            const l_min = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
                            const l_max = @min(li_u + lj_u, lmax_aug);
                            var big_l = l_min;
                            while (big_l <= l_max) : (big_l += 1) {
                                const aug_vals = aug_r2[i * nbeta * n_l_aug + j * n_l_aug + big_l] orelse continue;
                                const bl_i32: i32 = @intCast(big_l);
                                var big_m: i32 = -bl_i32;
                                while (big_m <= bl_i32) : (big_m += 1) {
                                    const gv = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l, big_m);
                                    if (@abs(gv) < 1e-30) continue;
                                    const lm_aug = big_l * big_l + @as(usize, @intCast(@as(i64, @intCast(big_l)) + big_m));
                                    const ylm_a = ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                    const gylm_a = grad_ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                    const aug_coeff = rij_up * gv * ylm_a;
                                    for (0..n_mesh) |k| {
                                        rho_ps_up[k] += aug_coeff * aug_vals[k];
                                        inline for (0..3) |d| {
                                            grad_s_ps_up[k][d] += rij_up * gv * gylm_a[d] * aug_vals[k];
                                        }
                                    }
                                }
                            }
                        }

                        // Down spin density
                        if (@abs(rij_dn) > 1e-30) {
                            const coeff = rij_dn * yi * yj;
                            const gc: [3]f64 = .{
                                rij_dn * (yj * gyi[0] + yi * gyj[0]),
                                rij_dn * (yj * gyi[1] + yi * gyj[1]),
                                rij_dn * (yj * gyi[2] + yi * gyj[2]),
                            };
                            for (0..n_mesh) |k| {
                                rho_ae_down[k] += coeff * ae_f[k];
                                rho_ps_down[k] += coeff * ps_f[k];
                                inline for (0..3) |d| {
                                    grad_s_ae_down[k][d] += gc[d] * ae_f[k];
                                    grad_s_ps_down[k][d] += gc[d] * ps_f[k];
                                }
                            }
                            // Augmentation for PS down
                            const l_min2 = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
                            const l_max2 = @min(li_u + lj_u, lmax_aug);
                            var big_l2 = l_min2;
                            while (big_l2 <= l_max2) : (big_l2 += 1) {
                                const aug_vals = aug_r2[i * nbeta * n_l_aug + j * n_l_aug + big_l2] orelse continue;
                                const bl_i32: i32 = @intCast(big_l2);
                                var big_m: i32 = -bl_i32;
                                while (big_m <= bl_i32) : (big_m += 1) {
                                    const gv = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l2, big_m);
                                    if (@abs(gv) < 1e-30) continue;
                                    const lm_aug = big_l2 * big_l2 + @as(usize, @intCast(@as(i64, @intCast(big_l2)) + big_m));
                                    const ylm_a = ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                    const gylm_a = grad_ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                    const aug_coeff = rij_dn * gv * ylm_a;
                                    for (0..n_mesh) |k| {
                                        rho_ps_down[k] += aug_coeff * aug_vals[k];
                                        inline for (0..3) |d| {
                                            grad_s_ps_down[k][d] += rij_dn * gv * gylm_a[d] * aug_vals[k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compute radial derivatives of spin densities
        @memcpy(tmp_rho, rho_ae_up);
        radialDerivative(tmp_rho, drho_ae_up, r, n_mesh);
        @memcpy(tmp_rho, rho_ae_down);
        radialDerivative(tmp_rho, drho_ae_down, r, n_mesh);
        @memcpy(tmp_rho, rho_ps_up);
        radialDerivative(tmp_rho, drho_ps_up, r, n_mesh);
        @memcpy(tmp_rho, rho_ps_down);
        radialDerivative(tmp_rho, drho_ps_down, r, n_mesh);

        // Add core density (split equally between spins)
        for (0..n_mesh) |k| {
            if (rho_core_ae) |core| {
                if (k < core.len) {
                    rho_ae_up[k] += core[k] * 0.5;
                    rho_ae_down[k] += core[k] * 0.5;
                    drho_ae_up[k] += dcore_ae[k] * 0.5;
                    drho_ae_down[k] += dcore_ae[k] * 0.5;
                }
            }
            if (rho_core_ps) |core| {
                if (k < core.len) {
                    rho_ps_up[k] += core[k] * 0.5;
                    rho_ps_down[k] += core[k] * 0.5;
                    drho_ps_up[k] += dcore_ps[k] * 0.5;
                    drho_ps_down[k] += dcore_ps[k] * 0.5;
                }
            }
        }

        // Compute spin XC integrals for each (i,j) pair
        for (0..nbeta) |i| {
            for (0..nbeta) |j| {
                const ae_f = uiuj_ae[i * nbeta + j];
                const ae_df = duiuj_ae[i * nbeta + j];
                const ps_f = uiuj_ps[i * nbeta + j];
                const ps_df = duiuj_ps[i * nbeta + j];

                var sum_rad_up: f64 = 0.0;
                var sum_rad_down: f64 = 0.0;
                var sum_ang_up: [3]f64 = .{ 0.0, 0.0, 0.0 };
                var sum_ang_down: [3]f64 = .{ 0.0, 0.0, 0.0 };
                for (0..n_mesh) |k| {
                    const r2 = r[k] * r[k];
                    const wk = rab[k] * ctrapWeight(k, n_mesh);
                    const inv_r2 = if (r2 > 1e-20) 1.0 / r2 else 0.0;

                    // AE spin XC
                    const n_ae_up = @max(rho_ae_up[k], 1e-30);
                    const n_ae_dn = @max(rho_ae_down[k], 1e-30);
                    const g2_uu_ae = drho_ae_up[k] * drho_ae_up[k] +
                        (grad_s_ae_up[k][0] * grad_s_ae_up[k][0] +
                            grad_s_ae_up[k][1] * grad_s_ae_up[k][1] +
                            grad_s_ae_up[k][2] * grad_s_ae_up[k][2]) * inv_r2;
                    const g2_dd_ae = drho_ae_down[k] * drho_ae_down[k] +
                        (grad_s_ae_down[k][0] * grad_s_ae_down[k][0] +
                            grad_s_ae_down[k][1] * grad_s_ae_down[k][1] +
                            grad_s_ae_down[k][2] * grad_s_ae_down[k][2]) * inv_r2;
                    const g2_ud_ae = drho_ae_up[k] * drho_ae_down[k] +
                        (grad_s_ae_up[k][0] * grad_s_ae_down[k][0] +
                            grad_s_ae_up[k][1] * grad_s_ae_down[k][1] +
                            grad_s_ae_up[k][2] * grad_s_ae_down[k][2]) * inv_r2;
                    const ev_ae = xc.evalPointSpin(xc_func, n_ae_up, n_ae_dn, g2_uu_ae, g2_dd_ae, g2_ud_ae);

                    // PS spin XC
                    const n_ps_up = @max(rho_ps_up[k], 1e-30);
                    const n_ps_dn = @max(rho_ps_down[k], 1e-30);
                    const g2_uu_ps = drho_ps_up[k] * drho_ps_up[k] +
                        (grad_s_ps_up[k][0] * grad_s_ps_up[k][0] +
                            grad_s_ps_up[k][1] * grad_s_ps_up[k][1] +
                            grad_s_ps_up[k][2] * grad_s_ps_up[k][2]) * inv_r2;
                    const g2_dd_ps = drho_ps_down[k] * drho_ps_down[k] +
                        (grad_s_ps_down[k][0] * grad_s_ps_down[k][0] +
                            grad_s_ps_down[k][1] * grad_s_ps_down[k][1] +
                            grad_s_ps_down[k][2] * grad_s_ps_down[k][2]) * inv_r2;
                    const g2_ud_ps = drho_ps_up[k] * drho_ps_down[k] +
                        (grad_s_ps_up[k][0] * grad_s_ps_down[k][0] +
                            grad_s_ps_up[k][1] * grad_s_ps_down[k][1] +
                            grad_s_ps_up[k][2] * grad_s_ps_down[k][2]) * inv_r2;
                    const ev_ps = xc.evalPointSpin(xc_func, n_ps_up, n_ps_dn, g2_uu_ps, g2_dd_ps, g2_ud_ps);

                    // D^xc_up: df/dn_up × f_ij + (2×df/dg2_uu × ∂ρ_up/∂r + df/dg2_ud × ∂ρ_down/∂r) × f'_ij
                    const ae_rad_up = ev_ae.df_dn_up * ae_f[k] + (2.0 * ev_ae.df_dg2_uu * drho_ae_up[k] + ev_ae.df_dg2_ud * drho_ae_down[k]) * ae_df[k];
                    const ps_rad_up = ev_ps.df_dn_up * ps_f[k] + (2.0 * ev_ps.df_dg2_uu * drho_ps_up[k] + ev_ps.df_dg2_ud * drho_ps_down[k]) * ps_df[k];
                    sum_rad_up += (ae_rad_up - ps_rad_up) * r2 * wk;

                    // D^xc_down: df/dn_down × f_ij + (2×df/dg2_dd × ∂ρ_down/∂r + df/dg2_ud × ∂ρ_up/∂r) × f'_ij
                    const ae_rad_dn = ev_ae.df_dn_down * ae_f[k] + (2.0 * ev_ae.df_dg2_dd * drho_ae_down[k] + ev_ae.df_dg2_ud * drho_ae_up[k]) * ae_df[k];
                    const ps_rad_dn = ev_ps.df_dn_down * ps_f[k] + (2.0 * ev_ps.df_dg2_dd * drho_ps_down[k] + ev_ps.df_dg2_ud * drho_ps_up[k]) * ps_df[k];
                    sum_rad_down += (ae_rad_dn - ps_rad_dn) * r2 * wk;

                    // Angular integrals: (2×df/dg2_σσ × ∇_S ρ_σ + df/dg2_ud × ∇_S ρ_σ') × f_ij
                    for (0..3) |d| {
                        const ae_a_up = (2.0 * ev_ae.df_dg2_uu * grad_s_ae_up[k][d] + ev_ae.df_dg2_ud * grad_s_ae_down[k][d]) * ae_f[k];
                        const ps_a_up = (2.0 * ev_ps.df_dg2_uu * grad_s_ps_up[k][d] + ev_ps.df_dg2_ud * grad_s_ps_down[k][d]) * ps_f[k];
                        sum_ang_up[d] += (ae_a_up - ps_a_up) * wk;

                        const ae_a_dn = (2.0 * ev_ae.df_dg2_dd * grad_s_ae_down[k][d] + ev_ae.df_dg2_ud * grad_s_ae_up[k][d]) * ae_f[k];
                        const ps_a_dn = (2.0 * ev_ps.df_dg2_dd * grad_s_ps_down[k][d] + ev_ps.df_dg2_ud * grad_s_ps_up[k][d]) * ps_f[k];
                        sum_ang_down[d] += (ae_a_dn - ps_a_dn) * wk;
                    }
                }
                rad_int_up[alpha * n_ij + i * nbeta + j] = sum_rad_up;
                rad_int_down[alpha * n_ij + i * nbeta + j] = sum_rad_down;
                ang_int_up[alpha * n_ij + i * nbeta + j] = sum_ang_up;
                ang_int_down[alpha * n_ij + i * nbeta + j] = sum_ang_down;

                // Augmentation integrals (PS only, subtracted)
                for (0..n_l_aug) |big_l| {
                    const aug_flat = i * nbeta * n_l_aug + j * n_l_aug + big_l;
                    const base_idx = (alpha * n_ij + i * nbeta + j) * n_l_aug + big_l;
                    const aug_f = aug_r2[aug_flat] orelse {
                        aug_rad_up[base_idx] = 0.0;
                        aug_rad_down[base_idx] = 0.0;
                        aug_ang_up[base_idx] = .{ 0.0, 0.0, 0.0 };
                        aug_ang_down[base_idx] = .{ 0.0, 0.0, 0.0 };
                        continue;
                    };
                    const aug_df = daug_r2[aug_flat].?;
                    var a_rad_up: f64 = 0.0;
                    var a_rad_down: f64 = 0.0;
                    var a_ang_up: [3]f64 = .{ 0.0, 0.0, 0.0 };
                    var a_ang_down: [3]f64 = .{ 0.0, 0.0, 0.0 };
                    for (0..n_mesh) |k| {
                        const r2 = r[k] * r[k];
                        const wk = rab[k] * ctrapWeight(k, n_mesh);
                        const inv_r2 = if (r2 > 1e-20) 1.0 / r2 else 0.0;
                        const n_ps_up = @max(rho_ps_up[k], 1e-30);
                        const n_ps_dn = @max(rho_ps_down[k], 1e-30);
                        const g2_uu_ps = drho_ps_up[k] * drho_ps_up[k] +
                            (grad_s_ps_up[k][0] * grad_s_ps_up[k][0] +
                                grad_s_ps_up[k][1] * grad_s_ps_up[k][1] +
                                grad_s_ps_up[k][2] * grad_s_ps_up[k][2]) * inv_r2;
                        const g2_dd_ps = drho_ps_down[k] * drho_ps_down[k] +
                            (grad_s_ps_down[k][0] * grad_s_ps_down[k][0] +
                                grad_s_ps_down[k][1] * grad_s_ps_down[k][1] +
                                grad_s_ps_down[k][2] * grad_s_ps_down[k][2]) * inv_r2;
                        const g2_ud_ps = drho_ps_up[k] * drho_ps_down[k] +
                            (grad_s_ps_up[k][0] * grad_s_ps_down[k][0] +
                                grad_s_ps_up[k][1] * grad_s_ps_down[k][1] +
                                grad_s_ps_up[k][2] * grad_s_ps_down[k][2]) * inv_r2;
                        const ev_ps = xc.evalPointSpin(xc_func, n_ps_up, n_ps_dn, g2_uu_ps, g2_dd_ps, g2_ud_ps);

                        a_rad_up += (ev_ps.df_dn_up * aug_f[k] + (2.0 * ev_ps.df_dg2_uu * drho_ps_up[k] + ev_ps.df_dg2_ud * drho_ps_down[k]) * aug_df[k]) * r2 * wk;
                        a_rad_down += (ev_ps.df_dn_down * aug_f[k] + (2.0 * ev_ps.df_dg2_dd * drho_ps_down[k] + ev_ps.df_dg2_ud * drho_ps_up[k]) * aug_df[k]) * r2 * wk;
                        for (0..3) |d| {
                            a_ang_up[d] += (2.0 * ev_ps.df_dg2_uu * grad_s_ps_up[k][d] + ev_ps.df_dg2_ud * grad_s_ps_down[k][d]) * aug_f[k] * wk;
                            a_ang_down[d] += (2.0 * ev_ps.df_dg2_dd * grad_s_ps_down[k][d] + ev_ps.df_dg2_ud * grad_s_ps_up[k][d]) * aug_f[k] * wk;
                        }
                    }
                    aug_rad_up[base_idx] = a_rad_up;
                    aug_rad_down[base_idx] = a_rad_down;
                    aug_ang_up[base_idx] = a_ang_up;
                    aug_ang_down[base_idx] = a_ang_down;
                }
            }
        }
    }

    // Accumulate m-resolved D^xc_up and D^xc_down
    @memset(dij_xc_m_up[0 .. m_total * m_total], 0.0);
    @memset(dij_xc_m_down[0 .. m_total * m_total], 0.0);

    for (grid, 0..) |pt, alpha| {
        const w_ang = pt.w * four_pi;
        const ylm_base = alpha * m_total;

        for (0..nbeta) |i| {
            const li = paw.ae_wfc[i].l;
            const li_u: usize = @intCast(li);
            const mi_count = @as(usize, @intCast(2 * li + 1));
            for (0..nbeta) |j| {
                const lj = paw.ae_wfc[j].l;
                const lj_u: usize = @intCast(lj);
                const mj_count = @as(usize, @intCast(2 * lj + 1));
                const I_rad_up = rad_int_up[alpha * n_ij + i * nbeta + j];
                const I_rad_dn = rad_int_down[alpha * n_ij + i * nbeta + j];
                const I_ang_up = ang_int_up[alpha * n_ij + i * nbeta + j];
                const I_ang_dn = ang_int_down[alpha * n_ij + i * nbeta + j];

                for (0..mi_count) |mi| {
                    const idx_i = ylm_base + m_offsets[i] + mi;
                    const yi = ylm_at[idx_i];
                    const gyi = grad_ylm_at[idx_i];
                    const im = m_offsets[i] + mi;
                    const mi_val: i32 = @as(i32, @intCast(mi)) - li;
                    for (0..mj_count) |mj| {
                        const idx_j = ylm_base + m_offsets[j] + mj;
                        const yj = ylm_at[idx_j];
                        const gyj = grad_ylm_at[idx_j];
                        const jm = m_offsets[j] + mj;
                        const mj_val: i32 = @as(i32, @intCast(mj)) - lj;
                        var c_up = w_ang * yi * yj * I_rad_up;
                        var c_dn = w_ang * yi * yj * I_rad_dn;
                        for (0..3) |d| {
                            const grad_sym = yj * gyi[d] + yi * gyj[d];
                            c_up += w_ang * I_ang_up[d] * grad_sym;
                            c_dn += w_ang * I_ang_dn[d] * grad_sym;
                        }
                        // Augmentation correction (subtracted)
                        const l_min_aug = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
                        const l_max_aug = @min(li_u + lj_u, lmax_aug);
                        var big_l = l_min_aug;
                        while (big_l <= l_max_aug) : (big_l += 1) {
                            const aug_base = (alpha * n_ij + i * nbeta + j) * n_l_aug + big_l;
                            const bl_i32: i32 = @intCast(big_l);
                            var big_m: i32 = -bl_i32;
                            while (big_m <= bl_i32) : (big_m += 1) {
                                const gv = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l, big_m);
                                if (@abs(gv) < 1e-30) continue;
                                const lm_aug = big_l * big_l + @as(usize, @intCast(@as(i64, @intCast(big_l)) + big_m));
                                const ylm_a = ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                const gylm_a = grad_ylm_aug_at[alpha * n_lm_aug + lm_aug];
                                c_up -= w_ang * gv * ylm_a * aug_rad_up[aug_base];
                                c_dn -= w_ang * gv * ylm_a * aug_rad_down[aug_base];
                                for (0..3) |d| {
                                    c_up -= w_ang * gv * gylm_a[d] * aug_ang_up[aug_base][d];
                                    c_dn -= w_ang * gv * gylm_a[d] * aug_ang_down[aug_base][d];
                                }
                            }
                        }
                        dij_xc_m_up[im * m_total + jm] += c_up;
                        dij_xc_m_down[im * m_total + jm] += c_dn;
                    }
                }
            }
        }
    }
}

/// Spin-polarized PAW on-site XC energy with angular Lebedev quadrature.
/// Returns E_xc^AE(up,down) - E_xc^PS(up,down).
pub fn computePawExcOnsiteAngularSpin(
    alloc: std.mem.Allocator,
    paw: PawData,
    rhoij_m_up: []const f64,
    rhoij_m_down: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    xc_func: xc.Functional,
    gaunt_table: *const GauntTable,
) !f64 {
    const nbeta = paw.number_of_proj;
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0) @min(n_mesh_full, paw.cutoff_r_index) else n_mesh_full;
    const leb_grid = lebedev.getLebedevGrid(N_ANG);
    const n_ang = leb_grid.len;
    const four_pi = 4.0 * std.math.pi;
    const lmax_aug = paw.lmax_aug;
    const n_l_aug = lmax_aug + 1;
    const n_lm_aug = n_l_aug * n_l_aug;

    // Pre-compute Y_{l,m}(Ω)
    const ylm_at = try alloc.alloc(f64, n_ang * m_total);
    defer alloc.free(ylm_at);
    const grad_ylm_at = try alloc.alloc([3]f64, n_ang * m_total);
    defer alloc.free(grad_ylm_at);
    for (leb_grid, 0..) |pt, alpha| {
        for (0..nbeta) |b| {
            const l = paw.ae_wfc[b].l;
            const mc = @as(usize, @intCast(2 * l + 1));
            for (0..mc) |mi| {
                const m: i32 = @as(i32, @intCast(mi)) - l;
                const idx = alpha * m_total + m_offsets[b] + mi;
                ylm_at[idx] = nonlocal.realSphericalHarmonic(l, m, pt.x, pt.y, pt.z);
                grad_ylm_at[idx] = surfGradYlm(l, m, pt.x, pt.y, pt.z);
            }
        }
    }

    // Pre-compute u_i*u_j/r²
    const n_ij = nbeta * nbeta;
    const ae_uiuj = try alloc.alloc([]f64, n_ij);
    defer {
        for (ae_uiuj) |s| alloc.free(s);
        alloc.free(ae_uiuj);
    }
    const ps_uiuj = try alloc.alloc([]f64, n_ij);
    defer {
        for (ps_uiuj) |s| alloc.free(s);
        alloc.free(ps_uiuj);
    }
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            const ae_buf = try alloc.alloc(f64, n_mesh);
            const ps_buf = try alloc.alloc(f64, n_mesh);
            const ae_i = paw.ae_wfc[i].values;
            const ae_j = paw.ae_wfc[j].values;
            const ps_i = paw.ps_wfc[i].values;
            const ps_j = paw.ps_wfc[j].values;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));
            for (0..n_r) |k| {
                if (r[k] < 1e-10) {
                    ae_buf[k] = 0.0;
                    ps_buf[k] = 0.0;
                } else {
                    ae_buf[k] = ae_i[k] * ae_j[k] / (r[k] * r[k]);
                    ps_buf[k] = ps_i[k] * ps_j[k] / (r[k] * r[k]);
                }
            }
            for (n_r..n_mesh) |k| {
                ae_buf[k] = 0.0;
                ps_buf[k] = 0.0;
            }
            ae_uiuj[i * nbeta + j] = ae_buf;
            ps_uiuj[i * nbeta + j] = ps_buf;
        }
    }

    // Pre-compute augmentation Q̂^L/r²
    const aug_r2 = try alloc.alloc(?[]f64, n_ij * n_l_aug);
    defer {
        for (aug_r2) |mb| if (mb) |buf| alloc.free(buf);
        alloc.free(aug_r2);
    }
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            for (0..n_l_aug) |big_l| {
                const fi = i * nbeta * n_l_aug + j * n_l_aug + big_l;
                if (findQijL(paw, i, j, big_l)) |qv| {
                    const buf = try alloc.alloc(f64, n_mesh);
                    const nq = @min(n_mesh, qv.len);
                    for (0..nq) |k| {
                        buf[k] = if (r[k] < 1e-10) 0.0 else qv[k] / (r[k] * r[k]);
                    }
                    for (nq..n_mesh) |k| buf[k] = 0.0;
                    aug_r2[fi] = buf;
                } else {
                    aug_r2[fi] = null;
                }
            }
        }
    }

    // Y_LM for augmentation
    const ylm_aug_at = try alloc.alloc(f64, n_ang * n_lm_aug);
    defer alloc.free(ylm_aug_at);
    const grad_ylm_aug_at = try alloc.alloc([3]f64, n_ang * n_lm_aug);
    defer alloc.free(grad_ylm_aug_at);
    for (leb_grid, 0..) |lp, ai| {
        for (0..n_l_aug) |bl| {
            const bli: i32 = @intCast(bl);
            for (0..2 * bl + 1) |bmi| {
                const bm: i32 = @as(i32, @intCast(bmi)) - bli;
                const lmi = bl * bl + bmi;
                const flat = ai * n_lm_aug + lmi;
                ylm_aug_at[flat] = nonlocal.realSphericalHarmonic(bli, bm, lp.x, lp.y, lp.z);
                grad_ylm_aug_at[flat] = surfGradYlmGeneral(bli, bm, lp.x, lp.y, lp.z);
            }
        }
    }

    // Core density derivatives
    const dcore_ae = try alloc.alloc(f64, n_mesh);
    defer alloc.free(dcore_ae);
    const dcore_ps = try alloc.alloc(f64, n_mesh);
    defer alloc.free(dcore_ps);
    if (rho_core_ae) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);
        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radialDerivative(buf, dcore_ae, r, n_mesh);
    } else @memset(dcore_ae, 0.0);
    if (rho_core_ps) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);
        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radialDerivative(buf, dcore_ps, r, n_mesh);
    } else @memset(dcore_ps, 0.0);

    // Work arrays
    const rho_ae_up = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ae_up);
    const rho_ae_dn = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ae_dn);
    const rho_ps_up = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ps_up);
    const rho_ps_dn = try alloc.alloc(f64, n_mesh);
    defer alloc.free(rho_ps_dn);
    const drho_ae_up = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ae_up);
    const drho_ae_dn = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ae_dn);
    const drho_ps_up = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ps_up);
    const drho_ps_dn = try alloc.alloc(f64, n_mesh);
    defer alloc.free(drho_ps_dn);
    const gs_ae_up = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(gs_ae_up);
    const gs_ae_dn = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(gs_ae_dn);
    const gs_ps_up = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(gs_ps_up);
    const gs_ps_dn = try alloc.alloc([3]f64, n_mesh);
    defer alloc.free(gs_ps_dn);

    const tmp_rad = try alloc.alloc(f64, n_mesh);
    defer alloc.free(tmp_rad);

    var exc_ae: f64 = 0.0;
    var exc_ps: f64 = 0.0;

    for (leb_grid, 0..) |_, alpha| {
        const w_ang = leb_grid[alpha].w * four_pi;
        const ylm_base = alpha * m_total;

        @memset(rho_ae_up, 0.0);
        @memset(rho_ae_dn, 0.0);
        @memset(rho_ps_up, 0.0);
        @memset(rho_ps_dn, 0.0);
        for (gs_ae_up) |*g| g.* = .{ 0.0, 0.0, 0.0 };
        for (gs_ae_dn) |*g| g.* = .{ 0.0, 0.0, 0.0 };
        for (gs_ps_up) |*g| g.* = .{ 0.0, 0.0, 0.0 };
        for (gs_ps_dn) |*g| g.* = .{ 0.0, 0.0, 0.0 };

        // Build spin-resolved densities
        for (0..nbeta) |i| {
            const li = paw.ae_wfc[i].l;
            const li_u: usize = @intCast(li);
            const mic = @as(usize, @intCast(2 * li + 1));
            for (0..nbeta) |j| {
                const lj = paw.ae_wfc[j].l;
                const lj_u: usize = @intCast(lj);
                const mjc = @as(usize, @intCast(2 * lj + 1));
                const ae_f = ae_uiuj[i * nbeta + j];
                const ps_f = ps_uiuj[i * nbeta + j];
                for (0..mic) |mi| {
                    const idx_i = ylm_base + m_offsets[i] + mi;
                    const yi = ylm_at[idx_i];
                    const gyi = grad_ylm_at[idx_i];
                    const mi_val: i32 = @as(i32, @intCast(mi)) - li;
                    for (0..mjc) |mj| {
                        const idx_j = ylm_base + m_offsets[j] + mj;
                        const yj = ylm_at[idx_j];
                        const gyj = grad_ylm_at[idx_j];
                        const im = m_offsets[i] + mi;
                        const jm = m_offsets[j] + mj;
                        const rij_up = rhoij_m_up[im * m_total + jm];
                        const rij_dn = rhoij_m_down[im * m_total + jm];
                        const mj_val: i32 = @as(i32, @intCast(mj)) - lj;

                        // Build densities for up spin
                        if (@abs(rij_up) > 1e-30) {
                            const coeff = rij_up * yi * yj;
                            const gc: [3]f64 = .{
                                rij_up * (yj * gyi[0] + yi * gyj[0]),
                                rij_up * (yj * gyi[1] + yi * gyj[1]),
                                rij_up * (yj * gyi[2] + yi * gyj[2]),
                            };
                            for (0..n_mesh) |k| {
                                rho_ae_up[k] += coeff * ae_f[k];
                                rho_ps_up[k] += coeff * ps_f[k];
                                inline for (0..3) |d| {
                                    gs_ae_up[k][d] += gc[d] * ae_f[k];
                                    gs_ps_up[k][d] += gc[d] * ps_f[k];
                                }
                            }
                            // Augmentation for up
                            const l_min = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
                            const l_max = @min(li_u + lj_u, lmax_aug);
                            var big_l = l_min;
                            while (big_l <= l_max) : (big_l += 1) {
                                const av = aug_r2[i * nbeta * n_l_aug + j * n_l_aug + big_l] orelse continue;
                                const bli2: i32 = @intCast(big_l);
                                var bm: i32 = -bli2;
                                while (bm <= bli2) : (bm += 1) {
                                    const gv = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l, bm);
                                    if (@abs(gv) < 1e-30) continue;
                                    const lma = big_l * big_l + @as(usize, @intCast(@as(i64, @intCast(big_l)) + bm));
                                    const ylma = ylm_aug_at[alpha * n_lm_aug + lma];
                                    const gylma = grad_ylm_aug_at[alpha * n_lm_aug + lma];
                                    const ac = rij_up * gv * ylma;
                                    for (0..n_mesh) |k| {
                                        rho_ps_up[k] += ac * av[k];
                                        inline for (0..3) |d2| {
                                            gs_ps_up[k][d2] += rij_up * gv * gylma[d2] * av[k];
                                        }
                                    }
                                }
                            }
                        }
                        // Build densities for down spin
                        if (@abs(rij_dn) > 1e-30) {
                            const coeff = rij_dn * yi * yj;
                            const gc: [3]f64 = .{
                                rij_dn * (yj * gyi[0] + yi * gyj[0]),
                                rij_dn * (yj * gyi[1] + yi * gyj[1]),
                                rij_dn * (yj * gyi[2] + yi * gyj[2]),
                            };
                            for (0..n_mesh) |k| {
                                rho_ae_dn[k] += coeff * ae_f[k];
                                rho_ps_dn[k] += coeff * ps_f[k];
                                inline for (0..3) |d| {
                                    gs_ae_dn[k][d] += gc[d] * ae_f[k];
                                    gs_ps_dn[k][d] += gc[d] * ps_f[k];
                                }
                            }
                            // Augmentation for down
                            const l_min2 = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
                            const l_max2 = @min(li_u + lj_u, lmax_aug);
                            var big_l2 = l_min2;
                            while (big_l2 <= l_max2) : (big_l2 += 1) {
                                const av = aug_r2[i * nbeta * n_l_aug + j * n_l_aug + big_l2] orelse continue;
                                const bli2: i32 = @intCast(big_l2);
                                var bm: i32 = -bli2;
                                while (bm <= bli2) : (bm += 1) {
                                    const gv = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l2, bm);
                                    if (@abs(gv) < 1e-30) continue;
                                    const lma = big_l2 * big_l2 + @as(usize, @intCast(@as(i64, @intCast(big_l2)) + bm));
                                    const ylma = ylm_aug_at[alpha * n_lm_aug + lma];
                                    const gylma = grad_ylm_aug_at[alpha * n_lm_aug + lma];
                                    const ac = rij_dn * gv * ylma;
                                    for (0..n_mesh) |k| {
                                        rho_ps_dn[k] += ac * av[k];
                                        inline for (0..3) |d2| {
                                            gs_ps_dn[k][d2] += rij_dn * gv * gylma[d2] * av[k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Radial derivatives (reuse pre-allocated tmp_rad)
        @memcpy(tmp_rad, rho_ae_up);
        radialDerivative(tmp_rad, drho_ae_up, r, n_mesh);
        @memcpy(tmp_rad, rho_ae_dn);
        radialDerivative(tmp_rad, drho_ae_dn, r, n_mesh);
        @memcpy(tmp_rad, rho_ps_up);
        radialDerivative(tmp_rad, drho_ps_up, r, n_mesh);
        @memcpy(tmp_rad, rho_ps_dn);
        radialDerivative(tmp_rad, drho_ps_dn, r, n_mesh);

        // Add core density (split equally)
        for (0..n_mesh) |k| {
            if (rho_core_ae) |core| {
                if (k < core.len) {
                    rho_ae_up[k] += core[k] * 0.5;
                    rho_ae_dn[k] += core[k] * 0.5;
                    drho_ae_up[k] += dcore_ae[k] * 0.5;
                    drho_ae_dn[k] += dcore_ae[k] * 0.5;
                }
            }
            if (rho_core_ps) |core| {
                if (k < core.len) {
                    rho_ps_up[k] += core[k] * 0.5;
                    rho_ps_dn[k] += core[k] * 0.5;
                    drho_ps_up[k] += dcore_ps[k] * 0.5;
                    drho_ps_dn[k] += dcore_ps[k] * 0.5;
                }
            }
        }

        // Integrate spin XC energy
        for (0..n_mesh) |k| {
            const r2 = r[k] * r[k];
            const wk = rab[k] * ctrapWeight(k, n_mesh);
            const inv_r2 = if (r2 > 1e-20) 1.0 / r2 else 0.0;

            const n_ae_u = @max(rho_ae_up[k], 1e-30);
            const n_ae_d = @max(rho_ae_dn[k], 1e-30);
            const g2_uu_ae = drho_ae_up[k] * drho_ae_up[k] +
                (gs_ae_up[k][0] * gs_ae_up[k][0] + gs_ae_up[k][1] * gs_ae_up[k][1] + gs_ae_up[k][2] * gs_ae_up[k][2]) * inv_r2;
            const g2_dd_ae = drho_ae_dn[k] * drho_ae_dn[k] +
                (gs_ae_dn[k][0] * gs_ae_dn[k][0] + gs_ae_dn[k][1] * gs_ae_dn[k][1] + gs_ae_dn[k][2] * gs_ae_dn[k][2]) * inv_r2;
            const g2_ud_ae = drho_ae_up[k] * drho_ae_dn[k] +
                (gs_ae_up[k][0] * gs_ae_dn[k][0] + gs_ae_up[k][1] * gs_ae_dn[k][1] + gs_ae_up[k][2] * gs_ae_dn[k][2]) * inv_r2;
            const ev_ae = xc.evalPointSpin(xc_func, n_ae_u, n_ae_d, g2_uu_ae, g2_dd_ae, g2_ud_ae);
            exc_ae += w_ang * ev_ae.f * r2 * wk;

            const n_ps_u = @max(rho_ps_up[k], 1e-30);
            const n_ps_d = @max(rho_ps_dn[k], 1e-30);
            const g2_uu_ps = drho_ps_up[k] * drho_ps_up[k] +
                (gs_ps_up[k][0] * gs_ps_up[k][0] + gs_ps_up[k][1] * gs_ps_up[k][1] + gs_ps_up[k][2] * gs_ps_up[k][2]) * inv_r2;
            const g2_dd_ps = drho_ps_dn[k] * drho_ps_dn[k] +
                (gs_ps_dn[k][0] * gs_ps_dn[k][0] + gs_ps_dn[k][1] * gs_ps_dn[k][1] + gs_ps_dn[k][2] * gs_ps_dn[k][2]) * inv_r2;
            const g2_ud_ps = drho_ps_up[k] * drho_ps_dn[k] +
                (gs_ps_up[k][0] * gs_ps_dn[k][0] + gs_ps_up[k][1] * gs_ps_dn[k][1] + gs_ps_up[k][2] * gs_ps_dn[k][2]) * inv_r2;
            const ev_ps = xc.evalPointSpin(xc_func, n_ps_u, n_ps_d, g2_uu_ps, g2_dd_ps, g2_ud_ps);
            exc_ps += w_ang * ev_ps.f * r2 * wk;
        }
    }
    return exc_ae - exc_ps;
}

/// Compute radial derivative df/dr using QE-style 3-point Lagrange formula
/// for non-uniform grids. More accurate than simple centered differences
/// on logarithmic grids.
///
/// Interior: df/dr[k] = [h2²(f[k-1]-f[k]) - h1²(f[k+1]-f[k])] / [h1*h2*(h1+h2)]
///   where h1 = r[k]-r[k-1], h2 = r[k+1]-r[k]
/// Last point: df = 0
/// First point: linear extrapolation from interior points
fn radialDerivative(f: []const f64, df: []f64, r_grid: []const f64, n: usize) void {
    if (n < 3) {
        @memset(df[0..n], 0.0);
        return;
    }
    // Interior points: 3-point Lagrange derivative for non-uniform grid
    for (1..n - 1) |k| {
        const h1 = r_grid[k] - r_grid[k - 1]; // r[k] - r[k-1]
        const h2 = r_grid[k + 1] - r_grid[k]; // r[k+1] - r[k]
        const denom = h1 * h2 * (h1 + h2);
        if (@abs(denom) > 1e-30) {
            df[k] = (h2 * h2 * (f[k - 1] - f[k]) - h1 * h1 * (f[k + 1] - f[k])) / denom;
        } else {
            df[k] = 0.0;
        }
    }
    // Last point: zero (QE convention)
    df[n - 1] = 0.0;
    // First point: linear extrapolation from points 2 and 3
    if (n >= 3) {
        const dr = r_grid[2] - r_grid[1];
        if (@abs(dr) > 1e-30) {
            df[0] = df[1] + (df[2] - df[1]) * (r_grid[0] - r_grid[1]) / dr;
        } else {
            df[0] = df[1];
        }
    }
}

test "evalPointSpin nonmagnetic consistency" {
    // For nonmagnetic input: evalPointSpin(n/2, n/2, σ/4, σ/4, σ/4)
    // should give df_dn_up = df_dn_down = evalPoint(n, σ).df_dn
    const n = 0.05;
    const sigma = 0.01;

    const ns = xc.evalPoint(.pbe, n, sigma);
    const sp = xc.evalPointSpin(.pbe, n / 2.0, n / 2.0, sigma / 4.0, sigma / 4.0, sigma / 4.0);

    // f should match
    try std.testing.expectApproxEqAbs(ns.f, sp.f, 1e-10);
    // df/dn_up = df/dn_down = df/dn
    try std.testing.expectApproxEqAbs(ns.df_dn, sp.df_dn_up, 1e-8);
    try std.testing.expectApproxEqAbs(ns.df_dn, sp.df_dn_down, 1e-8);
    // df/dσ consistency: at the nonmagnetic point σ_uu = σ_dd = σ_ud = σ/4,
    // so dσ_xx/dσ = 1/4 for each component, and by chain rule:
    // df/dσ = (df/dσ_uu + df/dσ_dd + df/dσ_ud) / 4
    const dg2_total = (sp.df_dg2_uu + sp.df_dg2_dd + sp.df_dg2_ud) / 4.0;
    try std.testing.expectApproxEqAbs(ns.df_dg2, dg2_total, 1e-8);
}
