const std = @import("std");
const paw_data = @import("../pseudopotential/paw_data.zig");
const PawData = paw_data.PawData;
const QijlEntry = paw_data.QijlEntry;
const xc = @import("../xc/xc.zig");
const gaunt_mod = @import("gaunt.zig");
const GauntTable = gaunt_mod.GauntTable;

const ctrap_weight = @import("../math/math.zig").radial.ctrap_weight;
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const lebedev = @import("../grid/lebedev.zig");

/// Find Q_ij^L(r) entry from paw.qijl for given (i,j,L) triplet.
fn find_qij_l(paw: PawData, i: usize, j: usize, big_l: usize) ?[]const f64 {
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
fn radial_hartree_potential_l0(
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
        const integrand = rho[k] * r[k] * r[k] * rab[k] * ctrap_weight(k, n_mesh);
        q_in += integrand;
        q_in_vals[k] = q_in;
    }

    // Backward cumulative integral: Q_out(r) = ∫ᵣ^∞ ρ(r') r' dr'
    var q_out: f64 = 0.0;
    var k_rev: usize = n_mesh;
    while (k_rev > 0) {
        k_rev -= 1;
        const integrand = rho[k_rev] * r[k_rev] * rab[k_rev] * ctrap_weight(k_rev, n_mesh);
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
fn radial_hartree_potential_l(
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
        i_in += rho[k] * rp_pow * rab[k] * ctrap_weight(k, n_mesh);
        i_in_vals[k] = i_in;
    }

    // Backward cumulative integral: I_out(r) = ∫ᵣ^∞ ρ(r') r'^{1-L} dr'
    var i_out: f64 = 0.0;
    var k_rev: usize = n_mesh;
    while (k_rev > 0) {
        k_rev -= 1;
        const rp = r[k_rev];
        const rp_pow = std.math.pow(f64, rp, 1.0 - fl);
        i_out += rho[k_rev] * rp_pow * rab[k_rev] * ctrap_weight(k_rev, n_mesh);
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

/// Allocate per-(L,M) density array slices initialised to zero.
fn alloc_lm_densities(alloc: std.mem.Allocator, n_lm: usize, n_mesh: usize) ![][]f64 {
    const out = try alloc.alloc([]f64, n_lm);
    for (0..n_lm) |lm| {
        out[lm] = try alloc.alloc(f64, n_mesh);
        @memset(out[lm], 0.0);
    }
    return out;
}

fn free_lm_densities(alloc: std.mem.Allocator, buf: [][]f64) void {
    for (buf) |s| alloc.free(s);
    alloc.free(buf);
}

/// Contract m-resolved rhoij with Gaunt coefficients to get rhoij_{LM}.
fn contract_rhoij_lm(
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    gaunt_table: *const GauntTable,
    i: usize,
    j: usize,
    li_i32: i32,
    lj_i32: i32,
    big_l: usize,
    bm: i32,
) f64 {
    const li_u: usize = @intCast(li_i32);
    const lj_u: usize = @intCast(lj_i32);
    var sum: f64 = 0.0;
    var mi: i32 = -li_i32;
    while (mi <= li_i32) : (mi += 1) {
        const mi_idx = m_offsets[i] + @as(usize, @intCast(mi + li_i32));
        var mj: i32 = -lj_i32;
        while (mj <= lj_i32) : (mj += 1) {
            const g_coeff = gaunt_table.get(li_u, mi, lj_u, mj, big_l, bm);
            if (g_coeff == 0.0) continue;
            const mj_idx = m_offsets[j] + @as(usize, @intCast(mj + lj_i32));
            sum += g_coeff * rhoij_m[mi_idx * m_total + mj_idx];
        }
    }
    return sum;
}

/// Accumulate u_i u_j / r² into density array rho_lm (for one (i,j) and scalar coefficient).
fn add_wfc_squared_to_lm(
    rho_lm: []f64,
    r: []const f64,
    wfc_i: []const f64,
    wfc_j: []const f64,
    n_r: usize,
    coeff: f64,
) void {
    for (0..n_r) |k| {
        if (r[k] < 1e-10) continue;
        const inv_r2 = 1.0 / (r[k] * r[k]);
        rho_lm[k] += coeff * wfc_i[k] * wfc_j[k] * inv_r2;
    }
}

/// Accumulate Q^L / r² into density array rho_lm.
fn add_augmentation_to_lm(
    rho_lm: []f64,
    r: []const f64,
    q_vals: []const f64,
    n_max: usize,
    coeff: f64,
) void {
    const n_q = @min(n_max, q_vals.len);
    for (0..n_q) |k| {
        if (r[k] < 1e-10) continue;
        const inv_r2 = 1.0 / (r[k] * r[k]);
        rho_lm[k] += coeff * q_vals[k] * inv_r2;
    }
}

/// Build multipole-decomposed AE and PS densities (with augmentation).
fn build_lm_densities_ae_ps(
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    n_mesh: usize,
    gaunt_table: *const GauntTable,
    rho_ae_lm: [][]f64,
    rho_ps_lm: [][]f64,
    include_augmentation: bool,
) void {
    const nbeta = paw.number_of_proj;
    const lmax_aug = gaunt_table.lmax_aug;
    for (0..nbeta) |i| {
        const li_i32: i32 = @intCast(paw.ae_wfc[i].l);
        const ae_i = paw.ae_wfc[i].values;
        const ps_i = paw.ps_wfc[i].values;
        for (0..nbeta) |j| {
            const lj_i32: i32 = @intCast(paw.ae_wfc[j].l);
            const ae_j = paw.ae_wfc[j].values;
            const ps_j = paw.ps_wfc[j].values;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));
            for (0..lmax_aug + 1) |big_l| {
                const bl_i32: i32 = @intCast(big_l);
                var bm: i32 = -bl_i32;
                while (bm <= bl_i32) : (bm += 1) {
                    const lm_idx = GauntTable.lm_index(big_l, bm);
                    const rhoij_lm = contract_rhoij_lm(
                        rhoij_m,
                        m_total,
                        m_offsets,
                        gaunt_table,
                        i,
                        j,
                        li_i32,
                        lj_i32,
                        big_l,
                        bm,
                    );
                    if (@abs(rhoij_lm) < 1e-30) continue;
                    add_wfc_squared_to_lm(rho_ae_lm[lm_idx], r, ae_i, ae_j, n_r, rhoij_lm);
                    add_wfc_squared_to_lm(rho_ps_lm[lm_idx], r, ps_i, ps_j, n_r, rhoij_lm);
                    if (include_augmentation) {
                        if (find_qij_l(paw, i, j, big_l)) |q_vals| {
                            add_augmentation_to_lm(rho_ps_lm[lm_idx], r, q_vals, n_r, rhoij_lm);
                        }
                    }
                }
            }
        }
    }
}

/// Compute multi-L on-site Hartree energy for one PAW atom.
///
/// E_H = Σ_{L,M} ½ ∫ V_H^{LM}(r) × ρ_{LM}(r) × r² dr
/// Uses full multipole expansion with Gaunt coefficients.
/// Returns E_H^AE - E_H^PS (including Q contributions in PS density).
/// Accumulate E_H = ½ ∫ V_H ρ r² dr for a single (L,M) density channel.
fn accumulate_hartree_channel_energy(
    r: []const f64,
    rab: []const f64,
    rho_lm: []const f64,
    vh: []f64,
    n_mesh: usize,
    big_l: usize,
) f64 {
    radial_hartree_potential_l(r, rab, rho_lm, vh, n_mesh, big_l);
    var sum: f64 = 0.0;
    for (0..n_mesh) |k| {
        const w = rab[k] * ctrap_weight(k, n_mesh);
        sum += 0.5 * vh[k] * rho_lm[k] * r[k] * r[k] * w;
    }
    return sum;
}

pub fn compute_paw_eh_onsite_multi_l(
    alloc: std.mem.Allocator,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    gaunt_table: *const GauntTable,
) !f64 {
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0)
        @min(n_mesh_full, paw.cutoff_r_index)
    else
        n_mesh_full;
    if (n_mesh > 4096) return error.MeshTooLarge;

    const lmax_aug = gaunt_table.lmax_aug;
    const n_lm_aug = (lmax_aug + 1) * (lmax_aug + 1);

    const rho_ae_lm = try alloc_lm_densities(alloc, n_lm_aug, n_mesh);
    defer free_lm_densities(alloc, rho_ae_lm);

    const rho_ps_lm = try alloc_lm_densities(alloc, n_lm_aug, n_mesh);
    defer free_lm_densities(alloc, rho_ps_lm);

    build_lm_densities_ae_ps(
        paw,
        rhoij_m,
        m_total,
        m_offsets,
        r,
        n_mesh,
        gaunt_table,
        rho_ae_lm,
        rho_ps_lm,
        true,
    );

    const vh = try alloc.alloc(f64, n_mesh);
    defer alloc.free(vh);

    var eh_ae: f64 = 0.0;
    var eh_ps: f64 = 0.0;
    for (0..lmax_aug + 1) |big_l| {
        const bl_i32: i32 = @intCast(big_l);
        var bm: i32 = -bl_i32;
        while (bm <= bl_i32) : (bm += 1) {
            const lm_idx = GauntTable.lm_index(big_l, bm);
            eh_ae += accumulate_hartree_channel_energy(
                r,
                rab,
                rho_ae_lm[lm_idx],
                vh,
                n_mesh,
                big_l,
            );
            eh_ps += accumulate_hartree_channel_energy(
                r,
                rab,
                rho_ps_lm[lm_idx],
                vh,
                n_mesh,
                big_l,
            );
        }
    }
    return eh_ae - eh_ps;
}

/// Compute multi-L on-site Hartree D_ij for one PAW atom.
///
/// D^H_{im,jm} = Σ_{LM} G(li,mi,lj,mj,L,M)
///             × ∫ [V^AE_{LM} u^AE_i u^AE_j - V^PS_{LM} (u^PS_i u^PS_j + Q^L)] dr
/// Output is m-resolved: dij_h[im*mt + jm].
/// Allocate per-(L,M) potential arrays and solve the radial Poisson equation
/// for every (L,M) channel covered by the augmentation expansion. Unused
/// channels are filled with zero so subsequent loops can blindly read them.
fn alloc_and_solve_lm_potentials(
    alloc: std.mem.Allocator,
    r: []const f64,
    rab: []const f64,
    rho_lm: [][]f64,
    n_mesh: usize,
    lmax_aug: usize,
    n_lm_aug: usize,
) ![][]f64 {
    const vh_lm = try alloc.alloc([]f64, n_lm_aug);
    for (0..n_lm_aug) |lm| {
        vh_lm[lm] = try alloc.alloc(f64, n_mesh);
        @memset(vh_lm[lm], 0.0);
    }
    for (0..lmax_aug + 1) |big_l| {
        const bl_i32: i32 = @intCast(big_l);
        var bm: i32 = -bl_i32;
        while (bm <= bl_i32) : (bm += 1) {
            const lm_idx = GauntTable.lm_index(big_l, bm);
            radial_hartree_potential_l(r, rab, rho_lm[lm_idx], vh_lm[lm_idx], n_mesh, big_l);
        }
    }
    return vh_lm;
}

/// Compute ∫ V × u_i u_j dr and its Q augmentation addition (used for Dij^H).
fn compute_hartree_radial_integrals(
    r: []const f64,
    rab: []const f64,
    n_mesh: usize,
    vh_ae_lm: []const f64,
    vh_ps_lm: []const f64,
    paw: PawData,
    i: usize,
    j: usize,
    big_l: usize,
    ae_i: []const f64,
    ae_j: []const f64,
    ps_i: []const f64,
    ps_j: []const f64,
    n_r: usize,
) struct { int_ae: f64, int_ps: f64 } {
    _ = r;
    var int_ae: f64 = 0.0;
    var int_ps: f64 = 0.0;
    for (0..n_r) |k| {
        const w = rab[k] * ctrap_weight(k, n_mesh);
        int_ae += vh_ae_lm[k] * ae_i[k] * ae_j[k] * w;
        int_ps += vh_ps_lm[k] * ps_i[k] * ps_j[k] * w;
    }
    if (find_qij_l(paw, i, j, big_l)) |q_vals| {
        const n_q = @min(n_r, q_vals.len);
        for (0..n_q) |k| {
            const w = rab[k] * ctrap_weight(k, n_mesh);
            int_ps += vh_ps_lm[k] * q_vals[k] * w;
        }
    }
    return .{ .int_ae = int_ae, .int_ps = int_ps };
}

/// Distribute scalar Δ over m-resolved D block using Gaunt(li,mi,lj,mj,L,M).
fn distribute_gaunt_to_dij(
    dij_h: []f64,
    m_total: usize,
    m_offsets: []const usize,
    gaunt_table: *const GauntTable,
    i: usize,
    j: usize,
    li: usize,
    lj: usize,
    li_i32: i32,
    lj_i32: i32,
    big_l: usize,
    bm: i32,
    int_diff: f64,
) void {
    const mi_count: usize = 2 * li + 1;
    const mj_count: usize = 2 * lj + 1;
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

/// Build ρ^AE, ρ^PS multipole densities and solve Poisson for V^AE, V^PS.
/// Returns the allocated vh_ae / vh_ps arrays; caller is responsible for free.
fn build_paw_hartree_potentials(
    alloc: std.mem.Allocator,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    n_mesh: usize,
    lmax_aug: usize,
    n_lm_aug: usize,
    gaunt_table: *const GauntTable,
) !struct { vh_ae: [][]f64, vh_ps: [][]f64, rho_ae: [][]f64, rho_ps: [][]f64 } {
    const rho_ae_lm = try alloc_lm_densities(alloc, n_lm_aug, n_mesh);
    const rho_ps_lm = try alloc_lm_densities(alloc, n_lm_aug, n_mesh);
    build_lm_densities_ae_ps(
        paw,
        rhoij_m,
        m_total,
        m_offsets,
        r,
        n_mesh,
        gaunt_table,
        rho_ae_lm,
        rho_ps_lm,
        true,
    );
    const vh_ae_lm = try alloc_and_solve_lm_potentials(
        alloc,
        r,
        rab,
        rho_ae_lm,
        n_mesh,
        lmax_aug,
        n_lm_aug,
    );
    const vh_ps_lm = try alloc_and_solve_lm_potentials(
        alloc,
        r,
        rab,
        rho_ps_lm,
        n_mesh,
        lmax_aug,
        n_lm_aug,
    );
    return .{ .vh_ae = vh_ae_lm, .vh_ps = vh_ps_lm, .rho_ae = rho_ae_lm, .rho_ps = rho_ps_lm };
}

pub fn compute_paw_dij_hartree_multi_l(
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
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0)
        @min(n_mesh_full, paw.cutoff_r_index)
    else
        n_mesh_full;
    if (n_mesh > 4096) return error.MeshTooLarge;

    const lmax_aug = gaunt_table.lmax_aug;
    const n_lm_aug = (lmax_aug + 1) * (lmax_aug + 1);

    const bundle = try build_paw_hartree_potentials(
        alloc,
        paw,
        rhoij_m,
        m_total,
        m_offsets,
        r,
        rab,
        n_mesh,
        lmax_aug,
        n_lm_aug,
        gaunt_table,
    );
    defer free_lm_densities(alloc, bundle.rho_ae);
    defer free_lm_densities(alloc, bundle.rho_ps);
    defer free_lm_densities(alloc, bundle.vh_ae);
    defer free_lm_densities(alloc, bundle.vh_ps);

    @memset(dij_h, 0.0);
    try accumulate_paw_dij_hartree_pairs(
        dij_h,
        paw,
        m_total,
        m_offsets,
        r,
        rab,
        n_mesh,
        bundle.vh_ae,
        bundle.vh_ps,
        lmax_aug,
        gaunt_table,
    );
}

fn accumulate_paw_dij_hartree_pairs(
    dij_h: []f64,
    paw: PawData,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    n_mesh: usize,
    vh_ae_lm: [][]f64,
    vh_ps_lm: [][]f64,
    lmax_aug: usize,
    gaunt_table: *const GauntTable,
) !void {
    const nbeta = paw.number_of_proj;
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
                    const lm_idx = GauntTable.lm_index(big_l, bm);
                    const ints = compute_hartree_radial_integrals(
                        r,
                        rab,
                        n_mesh,
                        vh_ae_lm[lm_idx],
                        vh_ps_lm[lm_idx],
                        paw,
                        i,
                        j,
                        big_l,
                        ae_i,
                        ae_j,
                        ps_i,
                        ps_j,
                        n_r,
                    );
                    const int_diff = ints.int_ae - ints.int_ps;
                    if (@abs(int_diff) < 1e-30) continue;
                    distribute_gaunt_to_dij(
                        dij_h,
                        m_total,
                        m_offsets,
                        gaunt_table,
                        i,
                        j,
                        li,
                        lj,
                        li_i32,
                        lj_i32,
                        big_l,
                        bm,
                        int_diff,
                    );
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
fn grad_solid_harmonic_l1(m: i32) [3]f64 {
    const pi = std.math.pi;
    const c = @sqrt(3.0 / (4.0 * pi));
    return switch (m) {
        -1 => .{ 0.0, c, 0.0 },
        0 => .{ 0.0, 0.0, c },
        1 => .{ c, 0.0, 0.0 },
        else => .{ 0.0, 0.0, 0.0 },
    };
}

fn grad_solid_harmonic_l2(m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    const pi = std.math.pi;
    const c0 = @sqrt(5.0 / (16.0 * pi));
    const c1 = @sqrt(15.0 / (4.0 * pi));
    const c2 = @sqrt(15.0 / (16.0 * pi));
    return switch (m) {
        -2 => .{ 2.0 * c2 * ny, 2.0 * c2 * nx, 0.0 },
        -1 => .{ 0.0, c1 * nz, c1 * ny },
        0 => .{ -2.0 * c0 * nx, -2.0 * c0 * ny, 4.0 * c0 * nz },
        1 => .{ c1 * nz, 0.0, c1 * nx },
        2 => .{ 2.0 * c2 * nx, -2.0 * c2 * ny, 0.0 },
        else => .{ 0.0, 0.0, 0.0 },
    };
}

fn grad_solid_harmonic_l3(m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    const pi = std.math.pi;
    return switch (m) {
        -3 => blk: {
            const c = @sqrt(35.0 / (32.0 * pi));
            break :blk .{ c * 6.0 * nx * ny, c * (3.0 * nx * nx - 3.0 * ny * ny), 0.0 };
        },
        -2 => blk: {
            const c = @sqrt(105.0 / (4.0 * pi));
            break :blk .{ c * ny * nz, c * nx * nz, c * nx * ny };
        },
        -1 => blk: {
            const c = @sqrt(21.0 / (32.0 * pi));
            break :blk .{
                -c * 2.0 * nx * ny,
                c * (4.0 * nz * nz - nx * nx - 3.0 * ny * ny),
                c * 8.0 * ny * nz,
            };
        },
        0 => blk: {
            const c = @sqrt(7.0 / (16.0 * pi));
            break :blk .{
                -c * 6.0 * nx * nz,
                -c * 6.0 * ny * nz,
                c * (6.0 * nz * nz - 3.0 * nx * nx - 3.0 * ny * ny),
            };
        },
        1 => blk: {
            const c = @sqrt(21.0 / (32.0 * pi));
            break :blk .{
                c * (4.0 * nz * nz - 3.0 * nx * nx - ny * ny),
                -c * 2.0 * nx * ny,
                c * 8.0 * nx * nz,
            };
        },
        2 => blk: {
            const c = @sqrt(105.0 / (16.0 * pi));
            break :blk .{ 2.0 * c * nx * nz, -2.0 * c * ny * nz, c * (nx * nx - ny * ny) };
        },
        3 => blk: {
            const c = @sqrt(35.0 / (32.0 * pi));
            break :blk .{ c * (3.0 * nx * nx - 3.0 * ny * ny), -c * 6.0 * nx * ny, 0.0 };
        },
        else => .{ 0.0, 0.0, 0.0 },
    };
}

fn grad_solid_harmonic(l: i32, m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    return switch (l) {
        0 => .{ 0.0, 0.0, 0.0 },
        1 => grad_solid_harmonic_l1(m),
        2 => grad_solid_harmonic_l2(m, nx, ny, nz),
        3 => grad_solid_harmonic_l3(m, nx, ny, nz),
        else => .{ 0.0, 0.0, 0.0 },
    };
}

/// Surface gradient of real spherical harmonic on unit sphere.
/// ∇_S Y_{lm} = ∇S_{lm} - l × Y_{lm} × r̂
/// where S_lm is the solid harmonic (homogeneous polynomial of degree l).
/// Euler's theorem: r̂ · ∇S = l × S = l × Y on the unit sphere.
fn surf_grad_ylm(l: i32, m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    const grad = grad_solid_harmonic(l, m, nx, ny, nz);
    const ylm = nonlocal.real_spherical_harmonic(l, m, nx, ny, nz);
    const fl: f64 = @floatFromInt(l);
    return .{
        grad[0] - fl * ylm * nx,
        grad[1] - fl * ylm * ny,
        grad[2] - fl * ylm * nz,
    };
}

/// Surface gradient of real spherical harmonic on unit sphere (general l).
/// Uses analytical grad_solid_harmonic for l <= 3, numerical differentiation for l >= 4.
/// ∇_S Y_{lm} = ∇S_{lm} - l × Y_{lm} × r̂
fn surf_grad_ylm_general(l: i32, m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    if (l <= 3) return surf_grad_ylm(l, m, nx, ny, nz);

    // Numerical gradient of solid harmonic S_lm(r) = |r|^l × Y_lm(r̂)
    // ∇S is computed via central differences at the unit-sphere point n̂.
    const delta: f64 = 1e-5;
    const fl: f64 = @floatFromInt(l);
    const ylm = nonlocal.real_spherical_harmonic(l, m, nx, ny, nz);
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
            nonlocal.real_spherical_harmonic(l, m, rp[0] / rp_len, rp[1] / rp_len, rp[2] / rp_len);
        const s_minus = std.math.pow(f64, rm_len, fl) *
            nonlocal.real_spherical_harmonic(l, m, rm[0] / rm_len, rm[1] / rm_len, rm[2] / rm_len);

        grad_s[d] = (s_plus - s_minus) / (2.0 * delta);
    }

    return .{
        grad_s[0] - fl * ylm * n[0],
        grad_s[1] - fl * ylm * n[1],
        grad_s[2] - fl * ylm * n[2],
    };
}

/// Pre-compute Y_{l,m}(Ω_α) and ∇_S Y_{l,m}(Ω_α) for all projector (β,m) channels
/// over the Lebedev angular grid.
fn precompute_projector_ylm_grads(
    grid: anytype,
    paw: PawData,
    m_total: usize,
    m_offsets: []const usize,
    ylm_at: []f64,
    grad_ylm_at: [][3]f64,
) void {
    const nbeta = paw.number_of_proj;
    for (grid, 0..) |pt, alpha| {
        for (0..nbeta) |b| {
            const l = paw.ae_wfc[b].l;
            const m_count = @as(usize, @intCast(2 * l + 1));
            for (0..m_count) |mi| {
                const m: i32 = @as(i32, @intCast(mi)) - l;
                const idx = alpha * m_total + m_offsets[b] + mi;
                ylm_at[idx] = nonlocal.real_spherical_harmonic(l, m, pt.x, pt.y, pt.z);
                grad_ylm_at[idx] = surf_grad_ylm(l, m, pt.x, pt.y, pt.z);
            }
        }
    }
}

/// Pre-compute Y_LM(Ω_α) and ∇_S Y_LM(Ω_α) for augmentation channels.
fn precompute_aug_ylm_grads(
    grid: anytype,
    n_l_aug: usize,
    n_lm_aug: usize,
    ylm_aug_at: []f64,
    grad_ylm_aug_at: [][3]f64,
) void {
    for (grid, 0..) |leb_pt, alpha_idx| {
        for (0..n_l_aug) |big_l| {
            const bl_i32: i32 = @intCast(big_l);
            for (0..2 * big_l + 1) |bm_idx| {
                const big_m: i32 = @as(i32, @intCast(bm_idx)) - bl_i32;
                const lm_idx = big_l * big_l + bm_idx;
                const flat = alpha_idx * n_lm_aug + lm_idx;
                ylm_aug_at[flat] = nonlocal.real_spherical_harmonic(
                    bl_i32,
                    big_m,
                    leb_pt.x,
                    leb_pt.y,
                    leb_pt.z,
                );
                grad_ylm_aug_at[flat] = surf_grad_ylm_general(
                    bl_i32,
                    big_m,
                    leb_pt.x,
                    leb_pt.y,
                    leb_pt.z,
                );
            }
        }
    }
}

/// Fill a buffer with u_i(k) * u_j(k) / r(k)² (0 if r~0 or beyond n_r).
fn fill_wfc_product_over_r2(
    out: []f64,
    wfc_i: []const f64,
    wfc_j: []const f64,
    r: []const f64,
    n_r: usize,
    n_mesh: usize,
) void {
    for (0..n_r) |k| {
        if (r[k] < 1e-10) {
            out[k] = 0.0;
        } else {
            out[k] = wfc_i[k] * wfc_j[k] / (r[k] * r[k]);
        }
    }
    for (n_r..n_mesh) |k| out[k] = 0.0;
}

/// Pre-compute u_i(k) u_j(k) / r² arrays (and optional derivatives) for every (i,j) pair.
const UiUjBuffers = struct {
    ae: [][]f64,
    ps: [][]f64,
    d_ae: ?[][]f64,
    d_ps: ?[][]f64,
};

fn precompute_ui_uj(
    alloc: std.mem.Allocator,
    paw: PawData,
    r: []const f64,
    n_mesh: usize,
    with_derivative: bool,
) !UiUjBuffers {
    const nbeta = paw.number_of_proj;
    const n_ij = nbeta * nbeta;
    const ae = try alloc.alloc([]f64, n_ij);
    const ps = try alloc.alloc([]f64, n_ij);
    const d_ae: ?[][]f64 = if (with_derivative) try alloc.alloc([]f64, n_ij) else null;
    const d_ps: ?[][]f64 = if (with_derivative) try alloc.alloc([]f64, n_ij) else null;
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            const ae_buf = try alloc.alloc(f64, n_mesh);
            const ps_buf = try alloc.alloc(f64, n_mesh);
            const ae_i = paw.ae_wfc[i].values;
            const ae_j = paw.ae_wfc[j].values;
            const ps_i = paw.ps_wfc[i].values;
            const ps_j = paw.ps_wfc[j].values;
            const n_r = @min(n_mesh, @min(@min(ae_i.len, ae_j.len), @min(ps_i.len, ps_j.len)));
            fill_wfc_product_over_r2(ae_buf, ae_i, ae_j, r, n_r, n_mesh);
            fill_wfc_product_over_r2(ps_buf, ps_i, ps_j, r, n_r, n_mesh);
            ae[i * nbeta + j] = ae_buf;
            ps[i * nbeta + j] = ps_buf;
            if (with_derivative) {
                const ae_dbuf = try alloc.alloc(f64, n_mesh);
                const ps_dbuf = try alloc.alloc(f64, n_mesh);
                radial_derivative(ae_buf, ae_dbuf, r, n_mesh);
                radial_derivative(ps_buf, ps_dbuf, r, n_mesh);
                d_ae.?[i * nbeta + j] = ae_dbuf;
                d_ps.?[i * nbeta + j] = ps_dbuf;
            }
        }
    }
    return .{ .ae = ae, .ps = ps, .d_ae = d_ae, .d_ps = d_ps };
}

/// Free buffers allocated by precompute_ui_uj.
fn free_ui_uj(
    alloc: std.mem.Allocator,
    buf: UiUjBuffers,
) void {
    for (buf.ae) |s| alloc.free(s);
    alloc.free(buf.ae);
    for (buf.ps) |s| alloc.free(s);
    alloc.free(buf.ps);
    if (buf.d_ae) |a| {
        for (a) |s| alloc.free(s);
        alloc.free(a);
    }
    if (buf.d_ps) |a| {
        for (a) |s| alloc.free(s);
        alloc.free(a);
    }
}

/// Pre-compute Q̂^L_ij / r² arrays (and optional derivatives) for every (i,j,L).
const AugR2Buffers = struct {
    vals: []?[]f64,
    d_vals: ?[]?[]f64,
};

fn precompute_aug_r2(
    alloc: std.mem.Allocator,
    paw: PawData,
    r: []const f64,
    n_mesh: usize,
    n_l_aug: usize,
    with_derivative: bool,
) !AugR2Buffers {
    const nbeta = paw.number_of_proj;
    const n_ij = nbeta * nbeta;
    const vals = try alloc.alloc(?[]f64, n_ij * n_l_aug);
    const d_vals: ?[]?[]f64 = if (with_derivative)
        try alloc.alloc(?[]f64, n_ij * n_l_aug)
    else
        null;
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            for (0..n_l_aug) |big_l| {
                const flat_idx = i * nbeta * n_l_aug + j * n_l_aug + big_l;
                if (find_qij_l(paw, i, j, big_l)) |qvals| {
                    const buf = try alloc.alloc(f64, n_mesh);
                    const n_q = @min(n_mesh, qvals.len);
                    for (0..n_q) |k| {
                        buf[k] = if (r[k] < 1e-10) 0.0 else qvals[k] / (r[k] * r[k]);
                    }
                    for (n_q..n_mesh) |k| buf[k] = 0.0;
                    vals[flat_idx] = buf;
                    if (with_derivative) {
                        const dbuf = try alloc.alloc(f64, n_mesh);
                        radial_derivative(buf, dbuf, r, n_mesh);
                        d_vals.?[flat_idx] = dbuf;
                    }
                } else {
                    vals[flat_idx] = null;
                    if (with_derivative) d_vals.?[flat_idx] = null;
                }
            }
        }
    }
    return .{ .vals = vals, .d_vals = d_vals };
}

fn free_aug_r2(
    alloc: std.mem.Allocator,
    buf: AugR2Buffers,
) void {
    for (buf.vals) |maybe| if (maybe) |b| alloc.free(b);
    alloc.free(buf.vals);
    if (buf.d_vals) |dv| {
        for (dv) |maybe| if (maybe) |b| alloc.free(b);
        alloc.free(dv);
    }
}

/// Compute radial derivative of an optional core density, extended with zeros
/// where the core array does not cover n_mesh. Writes into dcore_out.
fn precompute_core_derivative(
    alloc: std.mem.Allocator,
    rho_core: ?[]const f64,
    r: []const f64,
    n_mesh: usize,
    dcore_out: []f64,
) !void {
    if (rho_core) |core| {
        const buf = try alloc.alloc(f64, n_mesh);
        defer alloc.free(buf);

        const nc = @min(n_mesh, core.len);
        @memcpy(buf[0..nc], core[0..nc]);
        for (nc..n_mesh) |k| buf[k] = 0.0;
        radial_derivative(buf, dcore_out, r, n_mesh);
    } else {
        @memset(dcore_out, 0.0);
    }
}

/// Compute explicit V_xc(r) for GGA from density and radial gradient.
/// V_xc = df/dn - (1/r²) d/dr[r² × 2(df/dσ) × dρ/dr]
fn compute_vxc_radial(
    rho: []const f64,
    drho: []const f64,
    r_grid: []const f64,
    _: []const f64, // rab_grid (unused, radial_derivative now uses r_grid directly)
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
        const eval_pt = xc.eval_point(xc_func, n_val, sigma);
        vxc_out[k] = eval_pt.df_dn;
        h_work[k] = 2.0 * eval_pt.df_dg2 * drho[k];
    }
    // Step 2: r² × h
    for (0..n) |k| {
        r2h_work[k] = r_grid[k] * r_grid[k] * h_work[k];
    }
    // Step 3: d(r²h)/dr
    radial_derivative(r2h_work, dr2h_work, r_grid, n);
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
/// Build AE and PS densities (with angular gradients) at a single Lebedev point.
/// Writes into rho_ae/rho_ps and grad_s_rho_ae/grad_s_rho_ps in-place.
fn add_wfc_pair_to_density(
    rij: f64,
    yi: f64,
    yj: f64,
    gyi: [3]f64,
    gyj: [3]f64,
    ae_f: []const f64,
    ps_f: []const f64,
    n_mesh: usize,
    rho_ae: []f64,
    rho_ps: []f64,
    grad_s_rho_ae: [][3]f64,
    grad_s_rho_ps: [][3]f64,
) void {
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
}

/// Density-accumulation parameters shared across nested Lebedev loops. Using
/// a single struct keeps helper signatures short.
const DensityAccumCtx = struct {
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    alpha: usize,
    ylm_at: []const f64,
    grad_ylm_at: []const [3]f64,
    ylm_aug_at: []const f64,
    grad_ylm_aug_at: []const [3]f64,
    n_lm_aug: usize,
    n_l_aug: usize,
    lmax_aug: usize,
    aug_r2: []const ?[]f64,
    uiuj_ae: []const []f64,
    uiuj_ps: []const []f64,
    gaunt_table: *const GauntTable,
    n_mesh: usize,
    rho_ae: []f64,
    rho_ps: []f64,
    grad_s_rho_ae: [][3]f64,
    grad_s_rho_ps: [][3]f64,
};

fn add_mj_pair_density_contribution(
    ctx: *const DensityAccumCtx,
    i: usize,
    j: usize,
    mi: usize,
    mj: usize,
    li: i32,
    lj: i32,
    li_u: usize,
    lj_u: usize,
    yi: f64,
    gyi: [3]f64,
    ae_f: []const f64,
    ps_f: []const f64,
    ylm_base: usize,
) void {
    const nbeta = ctx.paw.number_of_proj;
    const mi_val: i32 = @as(i32, @intCast(mi)) - li;
    const rij = ctx.rhoij_m[(ctx.m_offsets[i] + mi) * ctx.m_total + (ctx.m_offsets[j] + mj)];
    if (@abs(rij) < 1e-30) return;
    const idx_j = ylm_base + ctx.m_offsets[j] + mj;
    const yj = ctx.ylm_at[idx_j];
    const gyj = ctx.grad_ylm_at[idx_j];
    add_wfc_pair_to_density(
        rij,
        yi,
        yj,
        gyi,
        gyj,
        ae_f,
        ps_f,
        ctx.n_mesh,
        ctx.rho_ae,
        ctx.rho_ps,
        ctx.grad_s_rho_ae,
        ctx.grad_s_rho_ps,
    );
    const mj_val: i32 = @as(i32, @intCast(mj)) - lj;
    add_augmentation_density_contribution(
        rij,
        mi_val,
        mj_val,
        li_u,
        lj_u,
        ctx.lmax_aug,
        nbeta,
        i,
        j,
        ctx.alpha,
        ctx.n_lm_aug,
        ctx.n_l_aug,
        ctx.aug_r2,
        ctx.ylm_aug_at,
        ctx.grad_ylm_aug_at,
        ctx.gaunt_table,
        ctx.n_mesh,
        ctx.rho_ps,
        ctx.grad_s_rho_ps,
    );
}

fn add_ij_pair_density_contribution(ctx: *const DensityAccumCtx, i: usize, j: usize) void {
    const nbeta = ctx.paw.number_of_proj;
    const li = ctx.paw.ae_wfc[i].l;
    const li_u: usize = @intCast(li);
    const mi_count = @as(usize, @intCast(2 * li + 1));
    const lj = ctx.paw.ae_wfc[j].l;
    const lj_u: usize = @intCast(lj);
    const mj_count = @as(usize, @intCast(2 * lj + 1));
    const ae_f = ctx.uiuj_ae[i * nbeta + j];
    const ps_f = ctx.uiuj_ps[i * nbeta + j];
    const ylm_base = ctx.alpha * ctx.m_total;
    for (0..mi_count) |mi| {
        const idx_i = ylm_base + ctx.m_offsets[i] + mi;
        const yi = ctx.ylm_at[idx_i];
        const gyi = ctx.grad_ylm_at[idx_i];
        for (0..mj_count) |mj| {
            add_mj_pair_density_contribution(
                ctx,
                i,
                j,
                mi,
                mj,
                li,
                lj,
                li_u,
                lj_u,
                yi,
                gyi,
                ae_f,
                ps_f,
                ylm_base,
            );
        }
    }
}

fn build_density_at_angular_point(ctx: *const DensityAccumCtx) void {
    const nbeta = ctx.paw.number_of_proj;
    @memset(ctx.rho_ae, 0.0);
    @memset(ctx.rho_ps, 0.0);
    for (ctx.grad_s_rho_ae) |*g| g.* = .{ 0.0, 0.0, 0.0 };
    for (ctx.grad_s_rho_ps) |*g| g.* = .{ 0.0, 0.0, 0.0 };

    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            add_ij_pair_density_contribution(ctx, i, j);
        }
    }
}

fn add_augmentation_density_contribution(
    rij: f64,
    mi_val: i32,
    mj_val: i32,
    li_u: usize,
    lj_u: usize,
    lmax_aug: usize,
    nbeta: usize,
    i: usize,
    j: usize,
    alpha: usize,
    n_lm_aug: usize,
    n_l_aug: usize,
    aug_r2: []const ?[]f64,
    ylm_aug_at: []const f64,
    grad_ylm_aug_at: []const [3]f64,
    gaunt_table: *const GauntTable,
    n_mesh: usize,
    rho_ps: []f64,
    grad_s_rho_ps: [][3]f64,
) void {
    const l_min = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
    const l_max = @min(li_u + lj_u, lmax_aug);
    var big_l = l_min;
    while (big_l <= l_max) : (big_l += 1) {
        const flat = i * nbeta * n_l_aug + j * n_l_aug + big_l;
        const aug_vals = aug_r2[flat] orelse continue;
        const bl_i32: i32 = @intCast(big_l);
        var big_m: i32 = -bl_i32;
        while (big_m <= bl_i32) : (big_m += 1) {
            const gaunt_val = gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l, big_m);
            if (@abs(gaunt_val) < 1e-30) continue;
            const lm_signed = @as(i64, @intCast(big_l)) + big_m;
            const lm_offset: usize = @intCast(lm_signed);
            const lm_aug = big_l * big_l + lm_offset;
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

/// Add core density (and its radial derivative) to the given ρ/dρ arrays.
/// Core is assumed spherical and so does not contribute to the angular gradient.
fn add_core_density(
    rho: []f64,
    drho: []f64,
    core_opt: ?[]const f64,
    dcore: []const f64,
    n_mesh: usize,
) void {
    const core = core_opt orelse return;
    const n_core = @min(n_mesh, core.len);
    for (0..n_core) |k| {
        rho[k] += core[k];
        drho[k] += dcore[k];
    }
}

const DijXcIntegrals = struct {
    sum_rad: f64,
    sum_ang: [3]f64,
};

fn compute_gga_radial_angular_integrals(
    n_mesh: usize,
    r: []const f64,
    rab: []const f64,
    ae_f: []const f64,
    ae_df: []const f64,
    ps_f: []const f64,
    ps_df: []const f64,
    rho_ae: []const f64,
    rho_ps: []const f64,
    drho_ae: []const f64,
    drho_ps: []const f64,
    grad_s_rho_ae: []const [3]f64,
    grad_s_rho_ps: []const [3]f64,
    xc_func: xc.Functional,
) DijXcIntegrals {
    var sum_rad: f64 = 0.0;
    var sum_ang: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (0..n_mesh) |k| {
        const r2 = r[k] * r[k];
        const wk = rab[k] * ctrap_weight(k, n_mesh);

        const n_ae = @max(rho_ae[k], 1e-30);
        const ang_sq_ae = grad_s_rho_ae[k][0] * grad_s_rho_ae[k][0] +
            grad_s_rho_ae[k][1] * grad_s_rho_ae[k][1] +
            grad_s_rho_ae[k][2] * grad_s_rho_ae[k][2];
        const sigma_ae = drho_ae[k] * drho_ae[k] +
            if (r2 > 1e-20) ang_sq_ae / r2 else 0.0;
        const eval_ae = xc.eval_point(xc_func, n_ae, sigma_ae);

        const n_ps = @max(rho_ps[k], 1e-30);
        const ang_sq_ps = grad_s_rho_ps[k][0] * grad_s_rho_ps[k][0] +
            grad_s_rho_ps[k][1] * grad_s_rho_ps[k][1] +
            grad_s_rho_ps[k][2] * grad_s_rho_ps[k][2];
        const sigma_ps = drho_ps[k] * drho_ps[k] +
            if (r2 > 1e-20) ang_sq_ps / r2 else 0.0;
        const eval_ps = xc.eval_point(xc_func, n_ps, sigma_ps);

        const ae_vxc_rad = eval_ae.df_dn * ae_f[k] +
            2.0 * eval_ae.df_dg2 * drho_ae[k] * ae_df[k];
        const ps_vxc_rad = eval_ps.df_dn * ps_f[k] +
            2.0 * eval_ps.df_dg2 * drho_ps[k] * ps_df[k];
        sum_rad += (ae_vxc_rad - ps_vxc_rad) * r2 * wk;

        for (0..3) |d| {
            const ae_vxc_ang = 2.0 * eval_ae.df_dg2 * grad_s_rho_ae[k][d] * ae_f[k];
            const ps_vxc_ang = 2.0 * eval_ps.df_dg2 * grad_s_rho_ps[k][d] * ps_f[k];
            sum_ang[d] += (ae_vxc_ang - ps_vxc_ang) * wk;
        }
    }
    return .{ .sum_rad = sum_rad, .sum_ang = sum_ang };
}

fn compute_gga_aug_integrals(
    n_mesh: usize,
    r: []const f64,
    rab: []const f64,
    aug_f: []const f64,
    aug_df: []const f64,
    rho_ps: []const f64,
    drho_ps: []const f64,
    grad_s_rho_ps: []const [3]f64,
    xc_func: xc.Functional,
) DijXcIntegrals {
    var a_rad: f64 = 0.0;
    var a_ang: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (0..n_mesh) |k| {
        const r2 = r[k] * r[k];
        const wk = rab[k] * ctrap_weight(k, n_mesh);
        const n_ps = @max(rho_ps[k], 1e-30);
        const ang_sq_ps = grad_s_rho_ps[k][0] * grad_s_rho_ps[k][0] +
            grad_s_rho_ps[k][1] * grad_s_rho_ps[k][1] +
            grad_s_rho_ps[k][2] * grad_s_rho_ps[k][2];
        const sigma_ps = drho_ps[k] * drho_ps[k] +
            if (r2 > 1e-20) ang_sq_ps / r2 else 0.0;
        const eval_ps = xc.eval_point(xc_func, n_ps, sigma_ps);

        a_rad += (eval_ps.df_dn * aug_f[k] +
            2.0 * eval_ps.df_dg2 * drho_ps[k] * aug_df[k]) * r2 * wk;
        for (0..3) |d| {
            a_ang[d] += 2.0 * eval_ps.df_dg2 * grad_s_rho_ps[k][d] * aug_f[k] * wk;
        }
    }
    return .{ .sum_rad = a_rad, .sum_ang = a_ang };
}

/// Context bundle for a single-angular-point GGA integration pass.
const DijXcAngularCtx = struct {
    alpha: usize,
    paw: PawData,
    r: []const f64,
    rab: []const f64,
    n_mesh: usize,
    xc_func: xc.Functional,
    n_lm_aug: usize,
    n_l_aug: usize,
    lmax_aug: usize,
    aug_r2: []const ?[]f64,
    daug_r2: []const ?[]f64,
    uiuj_ae: []const []f64,
    duiuj_ae: []const []f64,
    uiuj_ps: []const []f64,
    duiuj_ps: []const []f64,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    dcore_ae: []const f64,
    dcore_ps: []const f64,
    rho_ae: []f64,
    rho_ps: []f64,
    drho_ae: []f64,
    drho_ps: []f64,
    grad_s_rho_ae: [][3]f64,
    grad_s_rho_ps: [][3]f64,
    tmp_rho: []f64,
    radial_integrals: []f64,
    angular_integrals: [][3]f64,
    aug_rad_integrals: []f64,
    aug_ang_integrals: [][3]f64,
};

fn write_aug_integrals_for_ij(
    ctx: *const DijXcAngularCtx,
    i: usize,
    j: usize,
) void {
    const nbeta = ctx.paw.number_of_proj;
    const n_ij = nbeta * nbeta;
    for (0..ctx.n_l_aug) |big_l| {
        const aug_flat = i * nbeta * ctx.n_l_aug + j * ctx.n_l_aug + big_l;
        const out_idx = (ctx.alpha * n_ij + i * nbeta + j) * ctx.n_l_aug + big_l;
        const aug_f = ctx.aug_r2[aug_flat] orelse {
            ctx.aug_rad_integrals[out_idx] = 0.0;
            ctx.aug_ang_integrals[out_idx] = .{ 0.0, 0.0, 0.0 };
            continue;
        };
        const aug_df = ctx.daug_r2[aug_flat].?;
        const aug_res = compute_gga_aug_integrals(
            ctx.n_mesh,
            ctx.r,
            ctx.rab,
            aug_f,
            aug_df,
            ctx.rho_ps,
            ctx.drho_ps,
            ctx.grad_s_rho_ps,
            ctx.xc_func,
        );
        ctx.aug_rad_integrals[out_idx] = aug_res.sum_rad;
        ctx.aug_ang_integrals[out_idx] = aug_res.sum_ang;
    }
}

fn compute_dij_xc_one_angular(ctx: *const DijXcAngularCtx, dens: *const DensityAccumCtx) void {
    const nbeta = ctx.paw.number_of_proj;
    const n_ij = nbeta * nbeta;

    build_density_at_angular_point(dens);

    @memcpy(ctx.tmp_rho, ctx.rho_ae);
    radial_derivative(ctx.tmp_rho, ctx.drho_ae, ctx.r, ctx.n_mesh);
    @memcpy(ctx.tmp_rho, ctx.rho_ps);
    radial_derivative(ctx.tmp_rho, ctx.drho_ps, ctx.r, ctx.n_mesh);

    add_core_density(ctx.rho_ae, ctx.drho_ae, ctx.rho_core_ae, ctx.dcore_ae, ctx.n_mesh);
    add_core_density(ctx.rho_ps, ctx.drho_ps, ctx.rho_core_ps, ctx.dcore_ps, ctx.n_mesh);

    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            const res = compute_gga_radial_angular_integrals(
                ctx.n_mesh,
                ctx.r,
                ctx.rab,
                ctx.uiuj_ae[i * nbeta + j],
                ctx.duiuj_ae[i * nbeta + j],
                ctx.uiuj_ps[i * nbeta + j],
                ctx.duiuj_ps[i * nbeta + j],
                ctx.rho_ae,
                ctx.rho_ps,
                ctx.drho_ae,
                ctx.drho_ps,
                ctx.grad_s_rho_ae,
                ctx.grad_s_rho_ps,
                ctx.xc_func,
            );
            ctx.radial_integrals[ctx.alpha * n_ij + i * nbeta + j] = res.sum_rad;
            ctx.angular_integrals[ctx.alpha * n_ij + i * nbeta + j] = res.sum_ang;
            write_aug_integrals_for_ij(ctx, i, j);
        }
    }
}

fn augmentation_contrib_dij(
    alpha: usize,
    w_ang: f64,
    i: usize,
    j: usize,
    nbeta: usize,
    li_u: usize,
    lj_u: usize,
    mi_val: i32,
    mj_val: i32,
    lmax_aug: usize,
    n_l_aug: usize,
    n_lm_aug: usize,
    aug_rad_integrals: []const f64,
    aug_ang_integrals: []const [3]f64,
    ylm_aug_at: []const f64,
    grad_ylm_aug_at: []const [3]f64,
    gaunt_table: *const GauntTable,
) f64 {
    const n_ij = nbeta * nbeta;
    const l_min_aug = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
    const l_max_aug = @min(li_u + lj_u, lmax_aug);
    var total: f64 = 0.0;
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
            const lm_signed = @as(i64, @intCast(big_l)) + big_m;
            const lm_offset: usize = @intCast(lm_signed);
            const lm_aug = big_l * big_l + lm_offset;
            const ylm_a = ylm_aug_at[alpha * n_lm_aug + lm_aug];
            const gylm_a = grad_ylm_aug_at[alpha * n_lm_aug + lm_aug];
            total -= w_ang * gaunt_val * ylm_a * I_aug_rad;
            for (0..3) |d| {
                total -= w_ang * gaunt_val * gylm_a[d] * I_aug_ang[d];
            }
        }
    }
    return total;
}

const DijXcAccumCtx = struct {
    alpha: usize,
    w_ang: f64,
    paw: PawData,
    m_total: usize,
    m_offsets: []const usize,
    ylm_at: []const f64,
    grad_ylm_at: []const [3]f64,
    ylm_aug_at: []const f64,
    grad_ylm_aug_at: []const [3]f64,
    n_lm_aug: usize,
    n_l_aug: usize,
    lmax_aug: usize,
    radial_integrals: []const f64,
    angular_integrals: []const [3]f64,
    aug_rad_integrals: []const f64,
    aug_ang_integrals: []const [3]f64,
    gaunt_table: *const GauntTable,
    dij_xc_m: []f64,
};

fn accumulate_dij_xc_for_ij(ctx: *const DijXcAccumCtx, i: usize, j: usize) void {
    const nbeta = ctx.paw.number_of_proj;
    const n_ij = nbeta * nbeta;
    const ylm_base = ctx.alpha * ctx.m_total;
    const li = ctx.paw.ae_wfc[i].l;
    const li_u: usize = @intCast(li);
    const mi_count = @as(usize, @intCast(2 * li + 1));
    const lj = ctx.paw.ae_wfc[j].l;
    const lj_u: usize = @intCast(lj);
    const mj_count = @as(usize, @intCast(2 * lj + 1));
    const I_rad = ctx.radial_integrals[ctx.alpha * n_ij + i * nbeta + j];
    const I_ang = ctx.angular_integrals[ctx.alpha * n_ij + i * nbeta + j];

    for (0..mi_count) |mi| {
        const idx_i = ylm_base + ctx.m_offsets[i] + mi;
        const yi = ctx.ylm_at[idx_i];
        const gyi = ctx.grad_ylm_at[idx_i];
        const im = ctx.m_offsets[i] + mi;
        const mi_val: i32 = @as(i32, @intCast(mi)) - li;
        for (0..mj_count) |mj| {
            const idx_j = ylm_base + ctx.m_offsets[j] + mj;
            const yj = ctx.ylm_at[idx_j];
            const gyj = ctx.grad_ylm_at[idx_j];
            const jm = ctx.m_offsets[j] + mj;
            const mj_val: i32 = @as(i32, @intCast(mj)) - lj;
            var contrib = ctx.w_ang * yi * yj * I_rad;
            for (0..3) |d| {
                contrib += ctx.w_ang * I_ang[d] * (yj * gyi[d] + yi * gyj[d]);
            }
            contrib += augmentation_contrib_dij(
                ctx.alpha,
                ctx.w_ang,
                i,
                j,
                nbeta,
                li_u,
                lj_u,
                mi_val,
                mj_val,
                ctx.lmax_aug,
                ctx.n_l_aug,
                ctx.n_lm_aug,
                ctx.aug_rad_integrals,
                ctx.aug_ang_integrals,
                ctx.ylm_aug_at,
                ctx.grad_ylm_aug_at,
                ctx.gaunt_table,
            );
            ctx.dij_xc_m[im * ctx.m_total + jm] += contrib;
        }
    }
}

fn accumulate_dij_xc_for_angular(ctx: *const DijXcAccumCtx) void {
    const nbeta = ctx.paw.number_of_proj;
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            accumulate_dij_xc_for_ij(ctx, i, j);
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
/// Per-mesh scratch buffers reused for each Lebedev angular point.
const DijXcScratch = struct {
    rho_ae: []f64,
    rho_ps: []f64,
    drho_ae: []f64,
    drho_ps: []f64,
    tmp_rho: []f64,
    grad_s_rho_ae: [][3]f64,
    grad_s_rho_ps: [][3]f64,
    dcore_ae: []f64,
    dcore_ps: []f64,
    radial_integrals: []f64,
    angular_integrals: [][3]f64,
    aug_rad_integrals: []f64,
    aug_ang_integrals: [][3]f64,
};

fn alloc_dij_xc_scratch(
    alloc: std.mem.Allocator,
    n_mesh: usize,
    n_ang: usize,
    n_ij: usize,
    n_l_aug: usize,
) !DijXcScratch {
    return DijXcScratch{
        .rho_ae = try alloc.alloc(f64, n_mesh),
        .rho_ps = try alloc.alloc(f64, n_mesh),
        .drho_ae = try alloc.alloc(f64, n_mesh),
        .drho_ps = try alloc.alloc(f64, n_mesh),
        .tmp_rho = try alloc.alloc(f64, n_mesh),
        .grad_s_rho_ae = try alloc.alloc([3]f64, n_mesh),
        .grad_s_rho_ps = try alloc.alloc([3]f64, n_mesh),
        .dcore_ae = try alloc.alloc(f64, n_mesh),
        .dcore_ps = try alloc.alloc(f64, n_mesh),
        .radial_integrals = try alloc.alloc(f64, n_ang * n_ij),
        .angular_integrals = try alloc.alloc([3]f64, n_ang * n_ij),
        .aug_rad_integrals = try alloc.alloc(f64, n_ang * n_ij * n_l_aug),
        .aug_ang_integrals = try alloc.alloc([3]f64, n_ang * n_ij * n_l_aug),
    };
}

fn free_dij_xc_scratch(alloc: std.mem.Allocator, sc: DijXcScratch) void {
    alloc.free(sc.rho_ae);
    alloc.free(sc.rho_ps);
    alloc.free(sc.drho_ae);
    alloc.free(sc.drho_ps);
    alloc.free(sc.tmp_rho);
    alloc.free(sc.grad_s_rho_ae);
    alloc.free(sc.grad_s_rho_ps);
    alloc.free(sc.dcore_ae);
    alloc.free(sc.dcore_ps);
    alloc.free(sc.radial_integrals);
    alloc.free(sc.angular_integrals);
    alloc.free(sc.aug_rad_integrals);
    alloc.free(sc.aug_ang_integrals);
}

/// Per-call Lebedev buffers shared between Dij and Exc entry points.
const LebedevBuffers = struct {
    ylm_at: []f64,
    grad_ylm_at: [][3]f64,
    ylm_aug_at: []f64,
    grad_ylm_aug_at: [][3]f64,
};

fn alloc_lebedev_buffers(
    alloc: std.mem.Allocator,
    grid: anytype,
    paw: PawData,
    m_total: usize,
    m_offsets: []const usize,
    n_ang: usize,
    n_l_aug: usize,
    n_lm_aug: usize,
) !LebedevBuffers {
    const ylm_at = try alloc.alloc(f64, n_ang * m_total);
    const grad_ylm_at = try alloc.alloc([3]f64, n_ang * m_total);
    precompute_projector_ylm_grads(grid, paw, m_total, m_offsets, ylm_at, grad_ylm_at);
    const ylm_aug_at = try alloc.alloc(f64, n_ang * n_lm_aug);
    const grad_ylm_aug_at = try alloc.alloc([3]f64, n_ang * n_lm_aug);
    precompute_aug_ylm_grads(grid, n_l_aug, n_lm_aug, ylm_aug_at, grad_ylm_aug_at);
    return .{
        .ylm_at = ylm_at,
        .grad_ylm_at = grad_ylm_at,
        .ylm_aug_at = ylm_aug_at,
        .grad_ylm_aug_at = grad_ylm_aug_at,
    };
}

fn free_lebedev_buffers(alloc: std.mem.Allocator, lb: LebedevBuffers) void {
    alloc.free(lb.ylm_at);
    alloc.free(lb.grad_ylm_at);
    alloc.free(lb.ylm_aug_at);
    alloc.free(lb.grad_ylm_aug_at);
}

const MeshAndAug = struct {
    n_mesh: usize,
    lmax_aug: usize,
    n_l_aug: usize,
    n_lm_aug: usize,
};

fn compute_mesh_and_aug(paw: PawData, r: []const f64, rab: []const f64) MeshAndAug {
    const n_mesh_full = @min(r.len, rab.len);
    const n_mesh = if (paw.cutoff_r_index > 0)
        @min(n_mesh_full, paw.cutoff_r_index)
    else
        n_mesh_full;
    const lmax_aug = paw.lmax_aug;
    const n_l_aug = lmax_aug + 1;
    return .{
        .n_mesh = n_mesh,
        .lmax_aug = lmax_aug,
        .n_l_aug = n_l_aug,
        .n_lm_aug = n_l_aug * n_l_aug,
    };
}

pub fn compute_paw_dij_xc_angular(
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
    const dims = compute_mesh_and_aug(paw, r, rab);
    const grid = lebedev.get_lebedev_grid(N_ANG);
    const n_ang = grid.len;
    const n_ij = nbeta * nbeta;
    const four_pi = 4.0 * std.math.pi;

    const lb = try alloc_lebedev_buffers(
        alloc,
        grid,
        paw,
        m_total,
        m_offsets,
        n_ang,
        dims.n_l_aug,
        dims.n_lm_aug,
    );
    defer free_lebedev_buffers(alloc, lb);

    const uiuj = try precompute_ui_uj(alloc, paw, r, dims.n_mesh, true);
    defer free_ui_uj(alloc, uiuj);

    const aug = try precompute_aug_r2(alloc, paw, r, dims.n_mesh, dims.n_l_aug, true);
    defer free_aug_r2(alloc, aug);

    const sc = try alloc_dij_xc_scratch(alloc, dims.n_mesh, n_ang, n_ij, dims.n_l_aug);
    defer free_dij_xc_scratch(alloc, sc);

    try precompute_core_derivative(alloc, rho_core_ae, r, dims.n_mesh, sc.dcore_ae);
    try precompute_core_derivative(alloc, rho_core_ps, r, dims.n_mesh, sc.dcore_ps);

    try finalize_paw_dij_xc(
        grid,
        paw,
        rhoij_m,
        m_total,
        m_offsets,
        r,
        rab,
        dims.n_mesh,
        xc_func,
        gaunt_table,
        &lb,
        dims.n_lm_aug,
        dims.n_l_aug,
        dims.lmax_aug,
        aug,
        uiuj,
        rho_core_ae,
        rho_core_ps,
        &sc,
        dij_xc_m,
        four_pi,
    );
}

fn finalize_paw_dij_xc(
    grid: anytype,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    n_mesh: usize,
    xc_func: xc.Functional,
    gaunt_table: *const GauntTable,
    lb: *const LebedevBuffers,
    n_lm_aug: usize,
    n_l_aug: usize,
    lmax_aug: usize,
    aug: anytype,
    uiuj: anytype,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    sc: *const DijXcScratch,
    dij_xc_m: []f64,
    four_pi: f64,
) !void {
    run_dij_xc_angular_pass(
        grid,
        paw,
        rhoij_m,
        m_total,
        m_offsets,
        r,
        rab,
        n_mesh,
        xc_func,
        gaunt_table,
        lb.ylm_at,
        lb.grad_ylm_at,
        lb.ylm_aug_at,
        lb.grad_ylm_aug_at,
        n_lm_aug,
        n_l_aug,
        lmax_aug,
        aug,
        uiuj,
        rho_core_ae,
        rho_core_ps,
        sc,
    );
    @memset(dij_xc_m[0 .. m_total * m_total], 0.0);
    run_dij_xc_accumulation(
        grid,
        paw,
        m_total,
        m_offsets,
        lb.ylm_at,
        lb.grad_ylm_at,
        lb.ylm_aug_at,
        lb.grad_ylm_aug_at,
        n_lm_aug,
        n_l_aug,
        lmax_aug,
        sc,
        gaunt_table,
        dij_xc_m,
        four_pi,
    );
}

fn run_dij_xc_accumulation(
    grid: anytype,
    paw: PawData,
    m_total: usize,
    m_offsets: []const usize,
    ylm_at: []const f64,
    grad_ylm_at: []const [3]f64,
    ylm_aug_at: []const f64,
    grad_ylm_aug_at: []const [3]f64,
    n_lm_aug: usize,
    n_l_aug: usize,
    lmax_aug: usize,
    sc: *const DijXcScratch,
    gaunt_table: *const GauntTable,
    dij_xc_m: []f64,
    four_pi: f64,
) void {
    for (grid, 0..) |pt, alpha| {
        const ctx = DijXcAccumCtx{
            .alpha = alpha,
            .w_ang = pt.w * four_pi,
            .paw = paw,
            .m_total = m_total,
            .m_offsets = m_offsets,
            .ylm_at = ylm_at,
            .grad_ylm_at = grad_ylm_at,
            .ylm_aug_at = ylm_aug_at,
            .grad_ylm_aug_at = grad_ylm_aug_at,
            .n_lm_aug = n_lm_aug,
            .n_l_aug = n_l_aug,
            .lmax_aug = lmax_aug,
            .radial_integrals = sc.radial_integrals,
            .angular_integrals = sc.angular_integrals,
            .aug_rad_integrals = sc.aug_rad_integrals,
            .aug_ang_integrals = sc.aug_ang_integrals,
            .gaunt_table = gaunt_table,
            .dij_xc_m = dij_xc_m,
        };
        accumulate_dij_xc_for_angular(&ctx);
    }
}

fn build_density_accum_ctx(
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    alpha: usize,
    ylm_at: []const f64,
    grad_ylm_at: []const [3]f64,
    ylm_aug_at: []const f64,
    grad_ylm_aug_at: []const [3]f64,
    n_lm_aug: usize,
    n_l_aug: usize,
    lmax_aug: usize,
    aug_r2: []const ?[]f64,
    uiuj_ae: []const []f64,
    uiuj_ps: []const []f64,
    gaunt_table: *const GauntTable,
    n_mesh: usize,
    sc: *const DijXcScratch,
) DensityAccumCtx {
    return .{
        .paw = paw,
        .rhoij_m = rhoij_m,
        .m_total = m_total,
        .m_offsets = m_offsets,
        .alpha = alpha,
        .ylm_at = ylm_at,
        .grad_ylm_at = grad_ylm_at,
        .ylm_aug_at = ylm_aug_at,
        .grad_ylm_aug_at = grad_ylm_aug_at,
        .n_lm_aug = n_lm_aug,
        .n_l_aug = n_l_aug,
        .lmax_aug = lmax_aug,
        .aug_r2 = aug_r2,
        .uiuj_ae = uiuj_ae,
        .uiuj_ps = uiuj_ps,
        .gaunt_table = gaunt_table,
        .n_mesh = n_mesh,
        .rho_ae = sc.rho_ae,
        .rho_ps = sc.rho_ps,
        .grad_s_rho_ae = sc.grad_s_rho_ae,
        .grad_s_rho_ps = sc.grad_s_rho_ps,
    };
}

fn build_dij_xc_angular_ctx(
    alpha: usize,
    paw: PawData,
    r: []const f64,
    rab: []const f64,
    n_mesh: usize,
    xc_func: xc.Functional,
    n_lm_aug: usize,
    n_l_aug: usize,
    lmax_aug: usize,
    aug: anytype,
    uiuj: anytype,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    sc: *const DijXcScratch,
) DijXcAngularCtx {
    return .{
        .alpha = alpha,
        .paw = paw,
        .r = r,
        .rab = rab,
        .n_mesh = n_mesh,
        .xc_func = xc_func,
        .n_lm_aug = n_lm_aug,
        .n_l_aug = n_l_aug,
        .lmax_aug = lmax_aug,
        .aug_r2 = aug.vals,
        .daug_r2 = aug.d_vals.?,
        .uiuj_ae = uiuj.ae,
        .duiuj_ae = uiuj.d_ae.?,
        .uiuj_ps = uiuj.ps,
        .duiuj_ps = uiuj.d_ps.?,
        .rho_core_ae = rho_core_ae,
        .rho_core_ps = rho_core_ps,
        .dcore_ae = sc.dcore_ae,
        .dcore_ps = sc.dcore_ps,
        .rho_ae = sc.rho_ae,
        .rho_ps = sc.rho_ps,
        .drho_ae = sc.drho_ae,
        .drho_ps = sc.drho_ps,
        .grad_s_rho_ae = sc.grad_s_rho_ae,
        .grad_s_rho_ps = sc.grad_s_rho_ps,
        .tmp_rho = sc.tmp_rho,
        .radial_integrals = sc.radial_integrals,
        .angular_integrals = sc.angular_integrals,
        .aug_rad_integrals = sc.aug_rad_integrals,
        .aug_ang_integrals = sc.aug_ang_integrals,
    };
}

fn run_dij_xc_angular_pass(
    grid: anytype,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    n_mesh: usize,
    xc_func: xc.Functional,
    gaunt_table: *const GauntTable,
    ylm_at: []const f64,
    grad_ylm_at: []const [3]f64,
    ylm_aug_at: []const f64,
    grad_ylm_aug_at: []const [3]f64,
    n_lm_aug: usize,
    n_l_aug: usize,
    lmax_aug: usize,
    aug: anytype,
    uiuj: anytype,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    sc: *const DijXcScratch,
) void {
    for (grid, 0..) |_, alpha| {
        const dens = build_density_accum_ctx(
            paw,
            rhoij_m,
            m_total,
            m_offsets,
            alpha,
            ylm_at,
            grad_ylm_at,
            ylm_aug_at,
            grad_ylm_aug_at,
            n_lm_aug,
            n_l_aug,
            lmax_aug,
            aug.vals,
            uiuj.ae,
            uiuj.ps,
            gaunt_table,
            n_mesh,
            sc,
        );
        const ctx = build_dij_xc_angular_ctx(
            alpha,
            paw,
            r,
            rab,
            n_mesh,
            xc_func,
            n_lm_aug,
            n_l_aug,
            lmax_aug,
            aug,
            uiuj,
            rho_core_ae,
            rho_core_ps,
            sc,
        );
        compute_dij_xc_one_angular(&ctx, &dens);
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
/// Integrate ∫ f(ρ, σ) r² dr using the full-gradient σ formula at a single
/// angular point. Returns (exc_ae_contribution, exc_ps_contribution).
const ExcRadialPair = struct {
    ae: f64,
    ps: f64,
};

fn integrate_exc_radial(
    w_ang: f64,
    n_mesh: usize,
    r: []const f64,
    rab: []const f64,
    rho_ae: []const f64,
    rho_ps: []const f64,
    drho_ae: []const f64,
    drho_ps: []const f64,
    grad_s_rho_ae: []const [3]f64,
    grad_s_rho_ps: []const [3]f64,
    xc_func: xc.Functional,
) ExcRadialPair {
    var exc_ae: f64 = 0.0;
    var exc_ps: f64 = 0.0;
    for (0..n_mesh) |k| {
        const r2 = r[k] * r[k];
        const wk = rab[k] * ctrap_weight(k, n_mesh);
        const n_ae = @max(rho_ae[k], 1e-30);
        const ang_sq_ae = grad_s_rho_ae[k][0] * grad_s_rho_ae[k][0] +
            grad_s_rho_ae[k][1] * grad_s_rho_ae[k][1] +
            grad_s_rho_ae[k][2] * grad_s_rho_ae[k][2];
        const sigma_ae = drho_ae[k] * drho_ae[k] +
            if (r2 > 1e-20) ang_sq_ae / r2 else 0.0;
        const eval_ae = xc.eval_point(xc_func, n_ae, sigma_ae);
        exc_ae += w_ang * eval_ae.f * r2 * wk;

        const n_ps = @max(rho_ps[k], 1e-30);
        const ang_sq_ps = grad_s_rho_ps[k][0] * grad_s_rho_ps[k][0] +
            grad_s_rho_ps[k][1] * grad_s_rho_ps[k][1] +
            grad_s_rho_ps[k][2] * grad_s_rho_ps[k][2];
        const sigma_ps = drho_ps[k] * drho_ps[k] +
            if (r2 > 1e-20) ang_sq_ps / r2 else 0.0;
        const eval_ps = xc.eval_point(xc_func, n_ps, sigma_ps);
        exc_ps += w_ang * eval_ps.f * r2 * wk;
    }
    return .{ .ae = exc_ae, .ps = exc_ps };
}

/// Process one Lebedev angular point for Exc: build density + integrate energy.
/// Appends to total (exc_ae, exc_ps).
fn process_exc_one_angular(
    alpha: usize,
    w_ang: f64,
    dens: *const DensityAccumCtx,
    tmp_rho: []f64,
    drho_ae: []f64,
    drho_ps: []f64,
    dcore_ae: []const f64,
    dcore_ps: []const f64,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    r: []const f64,
    rab: []const f64,
    n_mesh: usize,
    xc_func: xc.Functional,
) ExcRadialPair {
    _ = alpha;
    build_density_at_angular_point(dens);
    @memcpy(tmp_rho, dens.rho_ae);
    radial_derivative(tmp_rho, drho_ae, r, n_mesh);
    @memcpy(tmp_rho, dens.rho_ps);
    radial_derivative(tmp_rho, drho_ps, r, n_mesh);

    add_core_density(dens.rho_ae, drho_ae, rho_core_ae, dcore_ae, n_mesh);
    add_core_density(dens.rho_ps, drho_ps, rho_core_ps, dcore_ps, n_mesh);

    return integrate_exc_radial(
        w_ang,
        n_mesh,
        r,
        rab,
        dens.rho_ae,
        dens.rho_ps,
        drho_ae,
        drho_ps,
        dens.grad_s_rho_ae,
        dens.grad_s_rho_ps,
        xc_func,
    );
}

const ExcScratch = struct {
    rho_ae: []f64,
    rho_ps: []f64,
    drho_ae: []f64,
    drho_ps: []f64,
    tmp_rho: []f64,
    grad_s_rho_ae: [][3]f64,
    grad_s_rho_ps: [][3]f64,
    dcore_ae: []f64,
    dcore_ps: []f64,
};

fn alloc_exc_scratch(alloc: std.mem.Allocator, n_mesh: usize) !ExcScratch {
    return .{
        .rho_ae = try alloc.alloc(f64, n_mesh),
        .rho_ps = try alloc.alloc(f64, n_mesh),
        .drho_ae = try alloc.alloc(f64, n_mesh),
        .drho_ps = try alloc.alloc(f64, n_mesh),
        .tmp_rho = try alloc.alloc(f64, n_mesh),
        .grad_s_rho_ae = try alloc.alloc([3]f64, n_mesh),
        .grad_s_rho_ps = try alloc.alloc([3]f64, n_mesh),
        .dcore_ae = try alloc.alloc(f64, n_mesh),
        .dcore_ps = try alloc.alloc(f64, n_mesh),
    };
}

fn build_exc_density_ctx(
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    alpha: usize,
    lb: *const LebedevBuffers,
    dims: anytype,
    aug: anytype,
    uiuj: anytype,
    gaunt_table: *const GauntTable,
    sc: *const ExcScratch,
) DensityAccumCtx {
    return .{
        .paw = paw,
        .rhoij_m = rhoij_m,
        .m_total = m_total,
        .m_offsets = m_offsets,
        .alpha = alpha,
        .ylm_at = lb.ylm_at,
        .grad_ylm_at = lb.grad_ylm_at,
        .ylm_aug_at = lb.ylm_aug_at,
        .grad_ylm_aug_at = lb.grad_ylm_aug_at,
        .n_lm_aug = dims.n_lm_aug,
        .n_l_aug = dims.n_l_aug,
        .lmax_aug = dims.lmax_aug,
        .aug_r2 = aug.vals,
        .uiuj_ae = uiuj.ae,
        .uiuj_ps = uiuj.ps,
        .gaunt_table = gaunt_table,
        .n_mesh = dims.n_mesh,
        .rho_ae = sc.rho_ae,
        .rho_ps = sc.rho_ps,
        .grad_s_rho_ae = sc.grad_s_rho_ae,
        .grad_s_rho_ps = sc.grad_s_rho_ps,
    };
}

fn free_exc_scratch(alloc: std.mem.Allocator, sc: ExcScratch) void {
    alloc.free(sc.rho_ae);
    alloc.free(sc.rho_ps);
    alloc.free(sc.drho_ae);
    alloc.free(sc.drho_ps);
    alloc.free(sc.tmp_rho);
    alloc.free(sc.grad_s_rho_ae);
    alloc.free(sc.grad_s_rho_ps);
    alloc.free(sc.dcore_ae);
    alloc.free(sc.dcore_ps);
}

pub fn compute_paw_exc_onsite_angular(
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
    const dims = compute_mesh_and_aug(paw, r, rab);
    const grid = lebedev.get_lebedev_grid(N_ANG);
    const n_ang = grid.len;
    const four_pi = 4.0 * std.math.pi;

    const lb = try alloc_lebedev_buffers(
        alloc,
        grid,
        paw,
        m_total,
        m_offsets,
        n_ang,
        dims.n_l_aug,
        dims.n_lm_aug,
    );
    defer free_lebedev_buffers(alloc, lb);

    const uiuj = try precompute_ui_uj(alloc, paw, r, dims.n_mesh, false);
    defer free_ui_uj(alloc, uiuj);

    const aug = try precompute_aug_r2(alloc, paw, r, dims.n_mesh, dims.n_l_aug, false);
    defer free_aug_r2(alloc, aug);

    const sc = try alloc_exc_scratch(alloc, dims.n_mesh);
    defer free_exc_scratch(alloc, sc);

    try precompute_core_derivative(alloc, rho_core_ae, r, dims.n_mesh, sc.dcore_ae);
    try precompute_core_derivative(alloc, rho_core_ps, r, dims.n_mesh, sc.dcore_ps);

    return run_exc_angular_loop(
        grid,
        paw,
        rhoij_m,
        m_total,
        m_offsets,
        &lb,
        &dims,
        aug,
        uiuj,
        gaunt_table,
        &sc,
        rho_core_ae,
        rho_core_ps,
        r,
        rab,
        xc_func,
        four_pi,
    );
}

fn run_exc_angular_loop(
    grid: anytype,
    paw: PawData,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    lb: *const LebedevBuffers,
    dims: anytype,
    aug: anytype,
    uiuj: anytype,
    gaunt_table: *const GauntTable,
    sc: *const ExcScratch,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    r: []const f64,
    rab: []const f64,
    xc_func: xc.Functional,
    four_pi: f64,
) f64 {
    var exc_ae: f64 = 0.0;
    var exc_ps: f64 = 0.0;
    for (grid, 0..) |pt, alpha| {
        const dens = build_exc_density_ctx(
            paw,
            rhoij_m,
            m_total,
            m_offsets,
            alpha,
            lb,
            dims,
            aug,
            uiuj,
            gaunt_table,
            sc,
        );
        const res = process_exc_one_angular(
            alpha,
            pt.w * four_pi,
            &dens,
            sc.tmp_rho,
            sc.drho_ae,
            sc.drho_ps,
            sc.dcore_ae,
            sc.dcore_ps,
            rho_core_ae,
            rho_core_ps,
            r,
            rab,
            dims.n_mesh,
            xc_func,
        );
        exc_ae += res.ae;
        exc_ps += res.ps;
    }
    return exc_ae - exc_ps;
}

const SpinGgaChannels = struct {
    rho_up: []const f64,
    rho_down: []const f64,
    drho_up: []const f64,
    drho_down: []const f64,
    grad_up: []const [3]f64,
    grad_down: []const [3]f64,
};

const SpinDensityChannels = struct {
    ae: SpinGgaChannels,
    ps: SpinGgaChannels,
};

const SpinDijXcIntegrals = struct {
    up: DijXcIntegrals,
    down: DijXcIntegrals,
};

const SpinExcResult = struct {
    ae: f64,
    ps: f64,
};

const SpinExcScratch = struct {
    up: ExcScratch,
    down: ExcScratch,
};

fn alloc_spin_exc_scratch(alloc: std.mem.Allocator, n_mesh: usize) !SpinExcScratch {
    return .{
        .up = try alloc_exc_scratch(alloc, n_mesh),
        .down = try alloc_exc_scratch(alloc, n_mesh),
    };
}

fn free_spin_exc_scratch(alloc: std.mem.Allocator, sc: SpinExcScratch) void {
    free_exc_scratch(alloc, sc.up);
    free_exc_scratch(alloc, sc.down);
}

const SpinDijXcScratch = struct {
    up: DijXcScratch,
    down: DijXcScratch,
};

fn alloc_spin_dij_xc_scratch(
    alloc: std.mem.Allocator,
    n_mesh: usize,
    n_ang: usize,
    n_ij: usize,
    n_l_aug: usize,
) !SpinDijXcScratch {
    return .{
        .up = try alloc_dij_xc_scratch(alloc, n_mesh, n_ang, n_ij, n_l_aug),
        .down = try alloc_dij_xc_scratch(alloc, n_mesh, n_ang, n_ij, n_l_aug),
    };
}

fn free_spin_dij_xc_scratch(alloc: std.mem.Allocator, sc: SpinDijXcScratch) void {
    free_dij_xc_scratch(alloc, sc.up);
    free_dij_xc_scratch(alloc, sc.down);
}

fn build_spin_gga_channels(
    rho_up: []const f64,
    rho_down: []const f64,
    drho_up: []const f64,
    drho_down: []const f64,
    grad_up: []const [3]f64,
    grad_down: []const [3]f64,
) SpinGgaChannels {
    return .{
        .rho_up = rho_up,
        .rho_down = rho_down,
        .drho_up = drho_up,
        .drho_down = drho_down,
        .grad_up = grad_up,
        .grad_down = grad_down,
    };
}

fn build_spin_density_channels(up: anytype, down: anytype) SpinDensityChannels {
    return .{
        .ae = build_spin_gga_channels(
            up.rho_ae,
            down.rho_ae,
            up.drho_ae,
            down.drho_ae,
            up.grad_s_rho_ae,
            down.grad_s_rho_ae,
        ),
        .ps = build_spin_gga_channels(
            up.rho_ps,
            down.rho_ps,
            up.drho_ps,
            down.drho_ps,
            up.grad_s_rho_ps,
            down.grad_s_rho_ps,
        ),
    };
}

fn precompute_spin_core_derivatives(
    alloc: std.mem.Allocator,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    r: []const f64,
    n_mesh: usize,
    dcore_ae: []f64,
    dcore_ps: []f64,
) !void {
    try precompute_core_derivative(alloc, rho_core_ae, r, n_mesh, dcore_ae);
    try precompute_core_derivative(alloc, rho_core_ps, r, n_mesh, dcore_ps);
}

fn radial_derivative_density_channel(
    rho_ae: []const f64,
    rho_ps: []const f64,
    tmp_rho: []f64,
    drho_ae: []f64,
    drho_ps: []f64,
    r: []const f64,
    n_mesh: usize,
) void {
    @memcpy(tmp_rho, rho_ae);
    radial_derivative(tmp_rho, drho_ae, r, n_mesh);
    @memcpy(tmp_rho, rho_ps);
    radial_derivative(tmp_rho, drho_ps, r, n_mesh);
}

fn split_core_density(
    rho_up: []f64,
    rho_down: []f64,
    drho_up: []f64,
    drho_down: []f64,
    core_opt: ?[]const f64,
    dcore: []const f64,
    n_mesh: usize,
) void {
    const core = core_opt orelse return;
    const n_core = @min(n_mesh, core.len);
    for (0..n_core) |k| {
        const half_core = 0.5 * core[k];
        const half_dcore = 0.5 * dcore[k];
        rho_up[k] += half_core;
        rho_down[k] += half_core;
        drho_up[k] += half_dcore;
        drho_down[k] += half_dcore;
    }
}

fn norm_sq3(v: [3]f64) f64 {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

fn dot3(a: [3]f64, b: [3]f64) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn integrate_spin_exc_radial(
    w_ang: f64,
    n_mesh: usize,
    r: []const f64,
    rab: []const f64,
    channels: SpinDensityChannels,
    xc_func: xc.Functional,
) SpinExcResult {
    var exc_ae: f64 = 0.0;
    var exc_ps: f64 = 0.0;
    for (0..n_mesh) |k| {
        const r2 = r[k] * r[k];
        const wk = rab[k] * ctrap_weight(k, n_mesh);
        const inv_r2 = if (r2 > 1e-20) 1.0 / r2 else 0.0;
        const n_ae_u = @max(channels.ae.rho_up[k], 1e-30);
        const n_ae_d = @max(channels.ae.rho_down[k], 1e-30);
        const g2_uu_ae = channels.ae.drho_up[k] * channels.ae.drho_up[k] +
            norm_sq3(channels.ae.grad_up[k]) * inv_r2;
        const g2_dd_ae = channels.ae.drho_down[k] * channels.ae.drho_down[k] +
            norm_sq3(channels.ae.grad_down[k]) * inv_r2;
        const g2_ud_ae = channels.ae.drho_up[k] * channels.ae.drho_down[k] +
            dot3(channels.ae.grad_up[k], channels.ae.grad_down[k]) * inv_r2;
        exc_ae +=
            w_ang * xc.eval_point_spin(xc_func, n_ae_u, n_ae_d, g2_uu_ae, g2_dd_ae, g2_ud_ae).f *
            r2 * wk;

        const n_ps_u = @max(channels.ps.rho_up[k], 1e-30);
        const n_ps_d = @max(channels.ps.rho_down[k], 1e-30);
        const g2_uu_ps = channels.ps.drho_up[k] * channels.ps.drho_up[k] +
            norm_sq3(channels.ps.grad_up[k]) * inv_r2;
        const g2_dd_ps = channels.ps.drho_down[k] * channels.ps.drho_down[k] +
            norm_sq3(channels.ps.grad_down[k]) * inv_r2;
        const g2_ud_ps = channels.ps.drho_up[k] * channels.ps.drho_down[k] +
            dot3(channels.ps.grad_up[k], channels.ps.grad_down[k]) * inv_r2;
        exc_ps +=
            w_ang * xc.eval_point_spin(xc_func, n_ps_u, n_ps_d, g2_uu_ps, g2_dd_ps, g2_ud_ps).f *
            r2 * wk;
    }
    return .{ .ae = exc_ae, .ps = exc_ps };
}

fn compute_gga_spin_radial_angular_integrals(
    n_mesh: usize,
    r: []const f64,
    rab: []const f64,
    ae_f: []const f64,
    ae_df: []const f64,
    ps_f: []const f64,
    ps_df: []const f64,
    channels: SpinDensityChannels,
    xc_func: xc.Functional,
) SpinDijXcIntegrals {
    var sum_rad_up: f64 = 0.0;
    var sum_rad_down: f64 = 0.0;
    var sum_ang_up: [3]f64 = .{ 0.0, 0.0, 0.0 };
    var sum_ang_down: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (0..n_mesh) |k| {
        const r2 = r[k] * r[k];
        const wk = rab[k] * ctrap_weight(k, n_mesh);
        const inv_r2 = if (r2 > 1e-20) 1.0 / r2 else 0.0;
        const ev_ae = xc.eval_point_spin(
            xc_func,
            @max(channels.ae.rho_up[k], 1e-30),
            @max(channels.ae.rho_down[k], 1e-30),
            channels.ae.drho_up[k] * channels.ae.drho_up[k] +
                norm_sq3(channels.ae.grad_up[k]) * inv_r2,
            channels.ae.drho_down[k] * channels.ae.drho_down[k] +
                norm_sq3(channels.ae.grad_down[k]) * inv_r2,
            channels.ae.drho_up[k] * channels.ae.drho_down[k] +
                dot3(channels.ae.grad_up[k], channels.ae.grad_down[k]) * inv_r2,
        );
        const ev_ps = xc.eval_point_spin(
            xc_func,
            @max(channels.ps.rho_up[k], 1e-30),
            @max(channels.ps.rho_down[k], 1e-30),
            channels.ps.drho_up[k] * channels.ps.drho_up[k] +
                norm_sq3(channels.ps.grad_up[k]) * inv_r2,
            channels.ps.drho_down[k] * channels.ps.drho_down[k] +
                norm_sq3(channels.ps.grad_down[k]) * inv_r2,
            channels.ps.drho_up[k] * channels.ps.drho_down[k] +
                dot3(channels.ps.grad_up[k], channels.ps.grad_down[k]) * inv_r2,
        );
        sum_rad_up += ((ev_ae.df_dn_up * ae_f[k] +
            (2.0 * ev_ae.df_dg2_uu * channels.ae.drho_up[k] +
                ev_ae.df_dg2_ud * channels.ae.drho_down[k]) * ae_df[k]) -
            (ev_ps.df_dn_up * ps_f[k] +
                (2.0 * ev_ps.df_dg2_uu * channels.ps.drho_up[k] +
                    ev_ps.df_dg2_ud * channels.ps.drho_down[k]) * ps_df[k])) * r2 * wk;
        sum_rad_down += ((ev_ae.df_dn_down * ae_f[k] +
            (2.0 * ev_ae.df_dg2_dd * channels.ae.drho_down[k] +
                ev_ae.df_dg2_ud * channels.ae.drho_up[k]) * ae_df[k]) -
            (ev_ps.df_dn_down * ps_f[k] +
                (2.0 * ev_ps.df_dg2_dd * channels.ps.drho_down[k] +
                    ev_ps.df_dg2_ud * channels.ps.drho_up[k]) * ps_df[k])) * r2 * wk;
        for (0..3) |d| {
            sum_ang_up[d] += ((2.0 * ev_ae.df_dg2_uu * channels.ae.grad_up[k][d] +
                ev_ae.df_dg2_ud * channels.ae.grad_down[k][d]) * ae_f[k] -
                (2.0 * ev_ps.df_dg2_uu * channels.ps.grad_up[k][d] +
                    ev_ps.df_dg2_ud * channels.ps.grad_down[k][d]) * ps_f[k]) * wk;
            sum_ang_down[d] += ((2.0 * ev_ae.df_dg2_dd * channels.ae.grad_down[k][d] +
                ev_ae.df_dg2_ud * channels.ae.grad_up[k][d]) * ae_f[k] -
                (2.0 * ev_ps.df_dg2_dd * channels.ps.grad_down[k][d] +
                    ev_ps.df_dg2_ud * channels.ps.grad_up[k][d]) * ps_f[k]) * wk;
        }
    }
    return .{
        .up = .{ .sum_rad = sum_rad_up, .sum_ang = sum_ang_up },
        .down = .{ .sum_rad = sum_rad_down, .sum_ang = sum_ang_down },
    };
}

fn compute_gga_spin_aug_integrals(
    n_mesh: usize,
    r: []const f64,
    rab: []const f64,
    aug_f: []const f64,
    aug_df: []const f64,
    ps: SpinGgaChannels,
    xc_func: xc.Functional,
) SpinDijXcIntegrals {
    var a_rad_up: f64 = 0.0;
    var a_rad_down: f64 = 0.0;
    var a_ang_up: [3]f64 = .{ 0.0, 0.0, 0.0 };
    var a_ang_down: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (0..n_mesh) |k| {
        const r2 = r[k] * r[k];
        const wk = rab[k] * ctrap_weight(k, n_mesh);
        const inv_r2 = if (r2 > 1e-20) 1.0 / r2 else 0.0;
        const ev_ps = xc.eval_point_spin(
            xc_func,
            @max(ps.rho_up[k], 1e-30),
            @max(ps.rho_down[k], 1e-30),
            ps.drho_up[k] * ps.drho_up[k] + norm_sq3(ps.grad_up[k]) * inv_r2,
            ps.drho_down[k] * ps.drho_down[k] + norm_sq3(ps.grad_down[k]) * inv_r2,
            ps.drho_up[k] * ps.drho_down[k] + dot3(ps.grad_up[k], ps.grad_down[k]) * inv_r2,
        );
        a_rad_up += (ev_ps.df_dn_up * aug_f[k] +
            (2.0 * ev_ps.df_dg2_uu * ps.drho_up[k] + ev_ps.df_dg2_ud * ps.drho_down[k]) *
                aug_df[k]) * r2 * wk;
        a_rad_down += (ev_ps.df_dn_down * aug_f[k] +
            (2.0 * ev_ps.df_dg2_dd * ps.drho_down[k] + ev_ps.df_dg2_ud * ps.drho_up[k]) *
                aug_df[k]) * r2 * wk;
        for (0..3) |d| {
            a_ang_up[d] +=
                (2.0 * ev_ps.df_dg2_uu * ps.grad_up[k][d] + ev_ps.df_dg2_ud * ps.grad_down[k][d]) *
                aug_f[k] * wk;
            a_ang_down[d] +=
                (2.0 * ev_ps.df_dg2_dd * ps.grad_down[k][d] + ev_ps.df_dg2_ud * ps.grad_up[k][d]) *
                aug_f[k] * wk;
        }
    }
    return .{
        .up = .{ .sum_rad = a_rad_up, .sum_ang = a_ang_up },
        .down = .{ .sum_rad = a_rad_down, .sum_ang = a_ang_down },
    };
}

const SpinExcAngularCtx = struct {
    paw: PawData,
    rhoij_m_up: []const f64,
    rhoij_m_down: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    lb: *const LebedevBuffers,
    dims: MeshAndAug,
    aug: *const AugR2Buffers,
    uiuj: *const UiUjBuffers,
    gaunt_table: *const GauntTable,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    r: []const f64,
    rab: []const f64,
    xc_func: xc.Functional,
};

fn prepare_spin_exc_density_channels(
    ctx: *const SpinExcAngularCtx,
    sc: *const SpinExcScratch,
    dens_up: *DensityAccumCtx,
    dens_down: *DensityAccumCtx,
) void {
    build_density_at_angular_point(dens_up);
    build_density_at_angular_point(dens_down);
    radial_derivative_density_channel(
        sc.up.rho_ae,
        sc.up.rho_ps,
        sc.up.tmp_rho,
        sc.up.drho_ae,
        sc.up.drho_ps,
        ctx.r,
        ctx.dims.n_mesh,
    );
    radial_derivative_density_channel(
        sc.down.rho_ae,
        sc.down.rho_ps,
        sc.down.tmp_rho,
        sc.down.drho_ae,
        sc.down.drho_ps,
        ctx.r,
        ctx.dims.n_mesh,
    );
    split_core_density(
        sc.up.rho_ae,
        sc.down.rho_ae,
        sc.up.drho_ae,
        sc.down.drho_ae,
        ctx.rho_core_ae,
        sc.up.dcore_ae,
        ctx.dims.n_mesh,
    );
    split_core_density(
        sc.up.rho_ps,
        sc.down.rho_ps,
        sc.up.drho_ps,
        sc.down.drho_ps,
        ctx.rho_core_ps,
        sc.up.dcore_ps,
        ctx.dims.n_mesh,
    );
}

fn process_spin_exc_one_angular(
    alpha: usize,
    w_ang: f64,
    ctx: *const SpinExcAngularCtx,
    sc: *const SpinExcScratch,
) SpinExcResult {
    var dens_up = build_exc_density_ctx(
        ctx.paw,
        ctx.rhoij_m_up,
        ctx.m_total,
        ctx.m_offsets,
        alpha,
        ctx.lb,
        ctx.dims,
        ctx.aug.*,
        ctx.uiuj.*,
        ctx.gaunt_table,
        &sc.up,
    );
    var dens_down = build_exc_density_ctx(
        ctx.paw,
        ctx.rhoij_m_down,
        ctx.m_total,
        ctx.m_offsets,
        alpha,
        ctx.lb,
        ctx.dims,
        ctx.aug.*,
        ctx.uiuj.*,
        ctx.gaunt_table,
        &sc.down,
    );
    prepare_spin_exc_density_channels(ctx, sc, &dens_up, &dens_down);
    return integrate_spin_exc_radial(
        w_ang,
        ctx.dims.n_mesh,
        ctx.r,
        ctx.rab,
        build_spin_density_channels(&sc.up, &sc.down),
        ctx.xc_func,
    );
}

fn run_spin_exc_angular_loop(
    grid: anytype,
    ctx: *const SpinExcAngularCtx,
    sc: *const SpinExcScratch,
    four_pi: f64,
) f64 {
    var exc_ae: f64 = 0.0;
    var exc_ps: f64 = 0.0;
    for (grid, 0..) |pt, alpha| {
        const res = process_spin_exc_one_angular(alpha, pt.w * four_pi, ctx, sc);
        exc_ae += res.ae;
        exc_ps += res.ps;
    }
    return exc_ae - exc_ps;
}

const SpinDijXcAngularCtx = struct {
    alpha: usize,
    paw: PawData,
    r: []const f64,
    rab: []const f64,
    dims: MeshAndAug,
    xc_func: xc.Functional,
    aug: *const AugR2Buffers,
    uiuj: *const UiUjBuffers,
    up: *const DijXcScratch,
    down: *const DijXcScratch,
};

fn write_spin_aug_integrals_for_ij(
    ctx: *const SpinDijXcAngularCtx,
    channels: SpinDensityChannels,
    i: usize,
    j: usize,
) void {
    const nbeta = ctx.paw.number_of_proj;
    const n_ij = nbeta * nbeta;
    for (0..ctx.dims.n_l_aug) |big_l| {
        const aug_flat = i * nbeta * ctx.dims.n_l_aug + j * ctx.dims.n_l_aug + big_l;
        const out_idx = (ctx.alpha * n_ij + i * nbeta + j) * ctx.dims.n_l_aug + big_l;
        const aug_f = ctx.aug.vals[aug_flat] orelse {
            ctx.up.aug_rad_integrals[out_idx] = 0.0;
            ctx.down.aug_rad_integrals[out_idx] = 0.0;
            ctx.up.aug_ang_integrals[out_idx] = .{ 0.0, 0.0, 0.0 };
            ctx.down.aug_ang_integrals[out_idx] = .{ 0.0, 0.0, 0.0 };
            continue;
        };
        const res = compute_gga_spin_aug_integrals(
            ctx.dims.n_mesh,
            ctx.r,
            ctx.rab,
            aug_f,
            ctx.aug.d_vals.?[aug_flat].?,
            channels.ps,
            ctx.xc_func,
        );
        ctx.up.aug_rad_integrals[out_idx] = res.up.sum_rad;
        ctx.down.aug_rad_integrals[out_idx] = res.down.sum_rad;
        ctx.up.aug_ang_integrals[out_idx] = res.up.sum_ang;
        ctx.down.aug_ang_integrals[out_idx] = res.down.sum_ang;
    }
}

fn write_spin_dij_integrals_for_ij(
    ctx: *const SpinDijXcAngularCtx,
    channels: SpinDensityChannels,
    i: usize,
    j: usize,
) void {
    const nbeta = ctx.paw.number_of_proj;
    const res = compute_gga_spin_radial_angular_integrals(
        ctx.dims.n_mesh,
        ctx.r,
        ctx.rab,
        ctx.uiuj.ae[i * nbeta + j],
        ctx.uiuj.d_ae.?[i * nbeta + j],
        ctx.uiuj.ps[i * nbeta + j],
        ctx.uiuj.d_ps.?[i * nbeta + j],
        channels,
        ctx.xc_func,
    );
    const out_idx = ctx.alpha * nbeta * nbeta + i * nbeta + j;
    ctx.up.radial_integrals[out_idx] = res.up.sum_rad;
    ctx.down.radial_integrals[out_idx] = res.down.sum_rad;
    ctx.up.angular_integrals[out_idx] = res.up.sum_ang;
    ctx.down.angular_integrals[out_idx] = res.down.sum_ang;
    write_spin_aug_integrals_for_ij(ctx, channels, i, j);
}

fn compute_spin_dij_xc_one_angular(
    ctx: *const SpinDijXcAngularCtx,
    dens_up: *const DensityAccumCtx,
    dens_down: *const DensityAccumCtx,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
) void {
    build_density_at_angular_point(dens_up);
    build_density_at_angular_point(dens_down);
    radial_derivative_density_channel(
        ctx.up.rho_ae,
        ctx.up.rho_ps,
        ctx.up.tmp_rho,
        ctx.up.drho_ae,
        ctx.up.drho_ps,
        ctx.r,
        ctx.dims.n_mesh,
    );
    radial_derivative_density_channel(
        ctx.down.rho_ae,
        ctx.down.rho_ps,
        ctx.down.tmp_rho,
        ctx.down.drho_ae,
        ctx.down.drho_ps,
        ctx.r,
        ctx.dims.n_mesh,
    );
    split_core_density(
        ctx.up.rho_ae,
        ctx.down.rho_ae,
        ctx.up.drho_ae,
        ctx.down.drho_ae,
        rho_core_ae,
        ctx.up.dcore_ae,
        ctx.dims.n_mesh,
    );
    split_core_density(
        ctx.up.rho_ps,
        ctx.down.rho_ps,
        ctx.up.drho_ps,
        ctx.down.drho_ps,
        rho_core_ps,
        ctx.up.dcore_ps,
        ctx.dims.n_mesh,
    );
    const channels = build_spin_density_channels(ctx.up, ctx.down);
    for (0..ctx.paw.number_of_proj) |i| {
        for (0..ctx.paw.number_of_proj) |j| {
            write_spin_dij_integrals_for_ij(ctx, channels, i, j);
        }
    }
}

const SpinDijDensityPair = struct {
    up: DensityAccumCtx,
    down: DensityAccumCtx,
};

fn build_spin_dij_density_pair(
    paw: PawData,
    rhoij_m_up: []const f64,
    rhoij_m_down: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    alpha: usize,
    lb: *const LebedevBuffers,
    dims: MeshAndAug,
    aug: *const AugR2Buffers,
    uiuj: *const UiUjBuffers,
    gaunt_table: *const GauntTable,
    sc: *const SpinDijXcScratch,
) SpinDijDensityPair {
    return .{
        .up = build_density_accum_ctx(
            paw,
            rhoij_m_up,
            m_total,
            m_offsets,
            alpha,
            lb.ylm_at,
            lb.grad_ylm_at,
            lb.ylm_aug_at,
            lb.grad_ylm_aug_at,
            dims.n_lm_aug,
            dims.n_l_aug,
            dims.lmax_aug,
            aug.vals,
            uiuj.ae,
            uiuj.ps,
            gaunt_table,
            dims.n_mesh,
            &sc.up,
        ),
        .down = build_density_accum_ctx(
            paw,
            rhoij_m_down,
            m_total,
            m_offsets,
            alpha,
            lb.ylm_at,
            lb.grad_ylm_at,
            lb.ylm_aug_at,
            lb.grad_ylm_aug_at,
            dims.n_lm_aug,
            dims.n_l_aug,
            dims.lmax_aug,
            aug.vals,
            uiuj.ae,
            uiuj.ps,
            gaunt_table,
            dims.n_mesh,
            &sc.down,
        ),
    };
}

fn run_spin_dij_xc_angular_pass(
    grid: anytype,
    paw: PawData,
    rhoij_m_up: []const f64,
    rhoij_m_down: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    gaunt_table: *const GauntTable,
    lb: *const LebedevBuffers,
    dims: MeshAndAug,
    aug: *const AugR2Buffers,
    uiuj: *const UiUjBuffers,
    sc: *const SpinDijXcScratch,
    r: []const f64,
    rab: []const f64,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    xc_func: xc.Functional,
) void {
    for (grid, 0..) |_, alpha| {
        const dens = build_spin_dij_density_pair(
            paw,
            rhoij_m_up,
            rhoij_m_down,
            m_total,
            m_offsets,
            alpha,
            lb,
            dims,
            aug,
            uiuj,
            gaunt_table,
            sc,
        );
        const ctx = SpinDijXcAngularCtx{
            .alpha = alpha,
            .paw = paw,
            .r = r,
            .rab = rab,
            .dims = dims,
            .xc_func = xc_func,
            .aug = aug,
            .uiuj = uiuj,
            .up = &sc.up,
            .down = &sc.down,
        };
        compute_spin_dij_xc_one_angular(&ctx, &dens.up, &dens.down, rho_core_ae, rho_core_ps);
    }
}

const SpinDijXcAccumCtx = struct {
    alpha: usize,
    w_ang: f64,
    paw: PawData,
    m_total: usize,
    m_offsets: []const usize,
    lb: *const LebedevBuffers,
    dims: MeshAndAug,
    gaunt_table: *const GauntTable,
    sc: *const SpinDijXcScratch,
    dij_xc_m_up: []f64,
    dij_xc_m_down: []f64,
};

fn spin_augmentation_contrib_dij(
    ctx: *const SpinDijXcAccumCtx,
    i: usize,
    j: usize,
    li_u: usize,
    lj_u: usize,
    mi_val: i32,
    mj_val: i32,
) struct { up: f64, down: f64 } {
    const nbeta = ctx.paw.number_of_proj;
    const n_ij = nbeta * nbeta;
    var sum_up: f64 = 0.0;
    var sum_down: f64 = 0.0;
    var big_l = if (li_u > lj_u) li_u - lj_u else lj_u - li_u;
    while (big_l <= @min(li_u + lj_u, ctx.dims.lmax_aug)) : (big_l += 1) {
        const aug_base = (ctx.alpha * n_ij + i * nbeta + j) * ctx.dims.n_l_aug + big_l;
        const bl_i32: i32 = @intCast(big_l);
        var big_m: i32 = -bl_i32;
        while (big_m <= bl_i32) : (big_m += 1) {
            const gv = ctx.gaunt_table.get(li_u, mi_val, lj_u, mj_val, big_l, big_m);
            if (@abs(gv) < 1e-30) continue;
            const lm_offset: usize = @intCast(@as(i64, @intCast(big_l)) + big_m);
            const lm_aug = big_l * big_l + lm_offset;
            const ylm_a = ctx.lb.ylm_aug_at[ctx.alpha * ctx.dims.n_lm_aug + lm_aug];
            const gylm_a = ctx.lb.grad_ylm_aug_at[ctx.alpha * ctx.dims.n_lm_aug + lm_aug];
            sum_up -= ctx.w_ang * gv * ylm_a * ctx.sc.up.aug_rad_integrals[aug_base];
            sum_down -= ctx.w_ang * gv * ylm_a * ctx.sc.down.aug_rad_integrals[aug_base];
            for (0..3) |d| {
                sum_up -= ctx.w_ang * gv * gylm_a[d] * ctx.sc.up.aug_ang_integrals[aug_base][d];
                sum_down -= ctx.w_ang * gv * gylm_a[d] * ctx.sc.down.aug_ang_integrals[aug_base][d];
            }
        }
    }
    return .{ .up = sum_up, .down = sum_down };
}

fn accumulate_spin_dij_xc_for_ij(ctx: *const SpinDijXcAccumCtx, i: usize, j: usize) void {
    const nbeta = ctx.paw.number_of_proj;
    const n_ij = nbeta * nbeta;
    const ylm_base = ctx.alpha * ctx.m_total;
    const li = ctx.paw.ae_wfc[i].l;
    const lj = ctx.paw.ae_wfc[j].l;
    const li_u: usize = @intCast(li);
    const lj_u: usize = @intCast(lj);
    const i_rad_up = ctx.sc.up.radial_integrals[ctx.alpha * n_ij + i * nbeta + j];
    const i_rad_down = ctx.sc.down.radial_integrals[ctx.alpha * n_ij + i * nbeta + j];
    const i_ang_up = ctx.sc.up.angular_integrals[ctx.alpha * n_ij + i * nbeta + j];
    const i_ang_down = ctx.sc.down.angular_integrals[ctx.alpha * n_ij + i * nbeta + j];
    for (0..@as(usize, @intCast(2 * li + 1))) |mi| {
        const idx_i = ylm_base + ctx.m_offsets[i] + mi;
        const yi = ctx.lb.ylm_at[idx_i];
        const gyi = ctx.lb.grad_ylm_at[idx_i];
        const im = ctx.m_offsets[i] + mi;
        const mi_val: i32 = @as(i32, @intCast(mi)) - li;
        for (0..@as(usize, @intCast(2 * lj + 1))) |mj| {
            const idx_j = ylm_base + ctx.m_offsets[j] + mj;
            const yj = ctx.lb.ylm_at[idx_j];
            const gyj = ctx.lb.grad_ylm_at[idx_j];
            const aug = spin_augmentation_contrib_dij(
                ctx,
                i,
                j,
                li_u,
                lj_u,
                mi_val,
                @as(i32, @intCast(mj)) - lj,
            );
            const jm = ctx.m_offsets[j] + mj;
            var c_up = ctx.w_ang * yi * yj * i_rad_up;
            var c_down = ctx.w_ang * yi * yj * i_rad_down;
            for (0..3) |d| {
                const grad_sym = yj * gyi[d] + yi * gyj[d];
                c_up += ctx.w_ang * i_ang_up[d] * grad_sym;
                c_down += ctx.w_ang * i_ang_down[d] * grad_sym;
            }
            ctx.dij_xc_m_up[im * ctx.m_total + jm] += c_up + aug.up;
            ctx.dij_xc_m_down[im * ctx.m_total + jm] += c_down + aug.down;
        }
    }
}

fn run_spin_dij_xc_accumulation(
    grid: anytype,
    paw: PawData,
    m_total: usize,
    m_offsets: []const usize,
    lb: *const LebedevBuffers,
    dims: MeshAndAug,
    sc: *const SpinDijXcScratch,
    gaunt_table: *const GauntTable,
    dij_xc_m_up: []f64,
    dij_xc_m_down: []f64,
    four_pi: f64,
) void {
    for (grid, 0..) |pt, alpha| {
        const ctx = SpinDijXcAccumCtx{
            .alpha = alpha,
            .w_ang = pt.w * four_pi,
            .paw = paw,
            .m_total = m_total,
            .m_offsets = m_offsets,
            .lb = lb,
            .dims = dims,
            .gaunt_table = gaunt_table,
            .sc = sc,
            .dij_xc_m_up = dij_xc_m_up,
            .dij_xc_m_down = dij_xc_m_down,
        };
        for (0..paw.number_of_proj) |i| {
            for (0..paw.number_of_proj) |j| {
                accumulate_spin_dij_xc_for_ij(&ctx, i, j);
            }
        }
    }
}

fn run_spin_dij_xc_with_buffers(
    grid: anytype,
    paw: PawData,
    rhoij_m_up: []const f64,
    rhoij_m_down: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    gaunt_table: *const GauntTable,
    lb: *const LebedevBuffers,
    dims: MeshAndAug,
    aug: *const AugR2Buffers,
    uiuj: *const UiUjBuffers,
    sc: *const SpinDijXcScratch,
    r: []const f64,
    rab: []const f64,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    xc_func: xc.Functional,
    dij_xc_m_up: []f64,
    dij_xc_m_down: []f64,
    four_pi: f64,
) void {
    @memset(dij_xc_m_up[0 .. m_total * m_total], 0.0);
    @memset(dij_xc_m_down[0 .. m_total * m_total], 0.0);
    run_spin_dij_xc_angular_pass(
        grid,
        paw,
        rhoij_m_up,
        rhoij_m_down,
        m_total,
        m_offsets,
        gaunt_table,
        lb,
        dims,
        aug,
        uiuj,
        sc,
        r,
        rab,
        rho_core_ae,
        rho_core_ps,
        xc_func,
    );
    run_spin_dij_xc_accumulation(
        grid,
        paw,
        m_total,
        m_offsets,
        lb,
        dims,
        sc,
        gaunt_table,
        dij_xc_m_up,
        dij_xc_m_down,
        four_pi,
    );
}

fn precompute_spin_dij_core_derivatives(
    alloc: std.mem.Allocator,
    rho_core_ae: ?[]const f64,
    rho_core_ps: ?[]const f64,
    r: []const f64,
    dims: MeshAndAug,
    sc: *const SpinDijXcScratch,
) !void {
    try precompute_spin_core_derivatives(
        alloc,
        rho_core_ae,
        rho_core_ps,
        r,
        dims.n_mesh,
        sc.up.dcore_ae,
        sc.up.dcore_ps,
    );
}

/// Spin-polarized PAW D^xc: compute D^xc_up and D^xc_down from rhoij_up/down.
/// Core density (spherical) is split equally between up and down.
/// For non-magnetic input (rhoij_up == rhoij_down == total/2), this reduces
/// to the same result as the non-spin version applied identically to both channels.
pub fn compute_paw_dij_xc_angular_spin(
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
    const dims = compute_mesh_and_aug(paw, r, rab);
    const grid = lebedev.get_lebedev_grid(N_ANG);
    const n_ij = nbeta * nbeta;
    const four_pi = 4.0 * std.math.pi;
    const lb = try alloc_lebedev_buffers(
        alloc,
        grid,
        paw,
        m_total,
        m_offsets,
        grid.len,
        dims.n_l_aug,
        dims.n_lm_aug,
    );
    defer free_lebedev_buffers(alloc, lb);

    const uiuj = try precompute_ui_uj(alloc, paw, r, dims.n_mesh, true);
    defer free_ui_uj(alloc, uiuj);

    const aug = try precompute_aug_r2(alloc, paw, r, dims.n_mesh, dims.n_l_aug, true);
    defer free_aug_r2(alloc, aug);

    const sc = try alloc_spin_dij_xc_scratch(alloc, dims.n_mesh, grid.len, n_ij, dims.n_l_aug);
    defer free_spin_dij_xc_scratch(alloc, sc);

    try precompute_spin_dij_core_derivatives(alloc, rho_core_ae, rho_core_ps, r, dims, &sc);

    run_spin_dij_xc_with_buffers(
        grid,
        paw,
        rhoij_m_up,
        rhoij_m_down,
        m_total,
        m_offsets,
        gaunt_table,
        &lb,
        dims,
        &aug,
        &uiuj,
        &sc,
        r,
        rab,
        rho_core_ae,
        rho_core_ps,
        xc_func,
        dij_xc_m_up,
        dij_xc_m_down,
        four_pi,
    );
}

/// Spin-polarized PAW on-site XC energy with angular Lebedev quadrature.
/// Returns E_xc^AE(up,down) - E_xc^PS(up,down).
pub fn compute_paw_exc_onsite_angular_spin(
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
    const dims = compute_mesh_and_aug(paw, r, rab);
    const grid = lebedev.get_lebedev_grid(N_ANG);
    const n_ang = grid.len;
    const four_pi = 4.0 * std.math.pi;

    const lb = try alloc_lebedev_buffers(
        alloc,
        grid,
        paw,
        m_total,
        m_offsets,
        n_ang,
        dims.n_l_aug,
        dims.n_lm_aug,
    );
    defer free_lebedev_buffers(alloc, lb);

    const uiuj = try precompute_ui_uj(alloc, paw, r, dims.n_mesh, false);
    defer free_ui_uj(alloc, uiuj);

    const aug = try precompute_aug_r2(alloc, paw, r, dims.n_mesh, dims.n_l_aug, false);
    defer free_aug_r2(alloc, aug);

    const sc = try alloc_spin_exc_scratch(alloc, dims.n_mesh);
    defer free_spin_exc_scratch(alloc, sc);

    try precompute_spin_core_derivatives(
        alloc,
        rho_core_ae,
        rho_core_ps,
        r,
        dims.n_mesh,
        sc.up.dcore_ae,
        sc.up.dcore_ps,
    );

    const ctx = SpinExcAngularCtx{
        .paw = paw,
        .rhoij_m_up = rhoij_m_up,
        .rhoij_m_down = rhoij_m_down,
        .m_total = m_total,
        .m_offsets = m_offsets,
        .lb = &lb,
        .dims = dims,
        .aug = &aug,
        .uiuj = &uiuj,
        .gaunt_table = gaunt_table,
        .rho_core_ae = rho_core_ae,
        .rho_core_ps = rho_core_ps,
        .r = r,
        .rab = rab,
        .xc_func = xc_func,
    };
    return run_spin_exc_angular_loop(grid, &ctx, &sc, four_pi);
}

/// Compute radial derivative df/dr using QE-style 3-point Lagrange formula
/// for non-uniform grids. More accurate than simple centered differences
/// on logarithmic grids.
///
/// Interior: df/dr[k] = [h2²(f[k-1]-f[k]) - h1²(f[k+1]-f[k])] / [h1*h2*(h1+h2)]
///   where h1 = r[k]-r[k-1], h2 = r[k+1]-r[k]
/// Last point: df = 0
/// First point: linear extrapolation from interior points
fn radial_derivative(f: []const f64, df: []f64, r_grid: []const f64, n: usize) void {
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

test "eval_point_spin nonmagnetic consistency" {
    // For nonmagnetic input: eval_point_spin(n/2, n/2, σ/4, σ/4, σ/4)
    // should give df_dn_up = df_dn_down = eval_point(n, σ).df_dn
    const n = 0.05;
    const sigma = 0.01;

    const ns = xc.eval_point(.pbe, n, sigma);
    const sp = xc.eval_point_spin(.pbe, n / 2.0, n / 2.0, sigma / 4.0, sigma / 4.0, sigma / 4.0);

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
