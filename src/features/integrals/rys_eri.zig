//! Rys quadrature electron repulsion integral engine.
//!
//! Implements the Rys quadrature algorithm for computing 2-electron integrals
//! (ab|cd) over Cartesian Gaussian basis functions, following the approach
//! used by libcint/PySCF.
//!
//! The algorithm has four stages:
//!   1. Compute Rys roots and weights for each primitive quartet
//!   2. Build 2D recurrence tables for x, y, z axes (separated)
//!   3. Apply horizontal recurrence to transfer angular momentum
//!   4. Extract and accumulate contracted ERIs
//!
//! References:
//!   - Dupuis, Rys, King, J. Chem. Phys. 65, 111 (1976)
//!   - Lindh, Ryu, Liu, J. Chem. Phys. 95, 5889 (1991)
//!   - libcint: https://github.com/sunqm/libcint

const std = @import("std");
const math = @import("../math/math.zig");
const basis_mod = @import("../basis/basis.zig");
const boys_mod = @import("boys.zig");
const rys_roots_mod = @import("rys_roots.zig");

const ContractedShell = basis_mod.ContractedShell;
const AngularMomentum = basis_mod.AngularMomentum;

// ============================================================================
// Constants
// ============================================================================

/// Maximum angular momentum per center (f=3 for 6-31G(2df,p)).
const MAX_L: usize = 4;

/// Maximum total angular momentum across bra pair (la+lb).
const MAX_IJ: usize = 2 * MAX_L;

/// Maximum total angular momentum across ket pair (lc+ld).
const MAX_KL: usize = 2 * MAX_L;

/// Maximum number of Rys roots.
const MAX_NROOTS: usize = rys_roots_mod.MAX_NROOTS;

/// Maximum dimension in the 2D recurrence for ij direction.
const MAX_DIM_IJ: usize = MAX_IJ + 1;

/// Maximum dimension in the 2D recurrence for kl direction.
const MAX_DIM_KL: usize = MAX_KL + 1;

/// Maximum Cartesian components per shell (f=10, g=15).
const MAX_CART: usize = basis_mod.MAX_CART;

/// Maximum primitives per shell.
const MAX_PRIM: usize = 16;

/// Maximum entries in norm table (prim * cart).
const MAX_NORM_TABLE: usize = MAX_PRIM * MAX_CART;

/// Size of the 2D g-table per axis: nroots * dim_ij * dim_kl.
/// For worst case (ff|ff): 7 * 7 * 7 = 343 per axis.
const MAX_G_SIZE: usize = MAX_NROOTS * MAX_DIM_IJ * MAX_DIM_KL;

/// Primitive screening threshold.
const PRIM_SCREEN_THRESHOLD: f64 = 1e-15;

/// Maximum number of primitive pairs per bra or ket side.
const MAX_PRIM_PAIRS: usize = MAX_PRIM * MAX_PRIM;

// ============================================================================
// Primitive pair precomputation
// ============================================================================

/// Pre-computed data for a bra primitive pair (a, b).
const BraPair = struct {
    p: f64, // alpha + beta
    inv_2p: f64, // 0.5 / p
    px: f64,
    py: f64,
    pz: f64, // Gaussian product center P
    pax: f64,
    pay: f64,
    paz: f64, // PA = P - A
    coeff_ab: f64, // c_a * c_b
    exp_ab: f64, // exp(-mu_ab * r2_ab)
    max_norm_ab: f64, // max_norm_a * max_norm_b
    ipa: usize, // primitive index in shell_a
    ipb: usize, // primitive index in shell_b
};

/// Pre-computed data for a ket primitive pair (c, d).
const KetPair = struct {
    q: f64, // gamma + delta
    inv_2q: f64, // 0.5 / q
    qx: f64,
    qy: f64,
    qz: f64, // Gaussian product center Q
    qcx: f64,
    qcy: f64,
    qcz: f64, // QC = Q - C
    coeff_cd: f64, // c_c * c_d
    exp_cd: f64, // exp(-mu_cd * r2_cd)
    max_norm_cd: f64, // max_norm_c * max_norm_d
    ipc: usize, // primitive index in shell_c
    ipd: usize, // primitive index in shell_d
};

// ============================================================================
// 2D Recurrence (per axis, per Rys root)
// ============================================================================

/// Build the 2D recurrence table for one Cartesian axis for all Rys roots.
///
/// The table g[root * dim_ij * dim_kl + i * dim_kl + k] stores the
/// intermediate integral for a given Rys root, where i = angular momentum
/// index in the ij direction and k = angular momentum index in the kl direction.
///
/// Recurrence relations (for each Rys root r):
///   g(0, 0) = 1 (for x,y) or w_r * prefactor (for z)
///   g(i+1, k) = c00 * g(i, k) + i * b10 * g(i-1, k) + k * b00 * g(i, k-1)
///   g(i, k+1) = c0p * g(i, k) + k * b01 * g(i, k-1) + i * b00 * g(i+1, k-1)  -- not used
///
/// We use the "upward" 2D recurrence from libcint (CINTg0_2e_2d):
///   First build g(i, 0) for i = 0..dim_ij-1 using:
///     g(i+1, 0) = c00 * g(i, 0) + i * b10 * g(i-1, 0)
///   Then build g(i, k) for k = 1..dim_kl-1 using:
///     g(i, k+1) = c0p * g(i, k) + k * b01 * g(i, k-1) + i * b00 * g(i-1, k)
///
/// Parameters:
///   nroots: number of Rys quadrature points
///   dim_ij: number of entries in ij direction (la+lb+1 or la+1 depending on strategy)
///   dim_kl: number of entries in kl direction (lc+ld+1 or lc+1 depending on strategy)
///   c00: array[nroots] of PA_axis + WP_axis * t coefficients
///   c0p: array[nroots] of QC_axis + WQ_axis * t coefficients
///   b10: array[nroots] of 0.5/aij * (1 - akl/(aij+akl)*t)
///   b01: array[nroots] of 0.5/akl * (1 - aij/(aij+akl)*t)
///   b00: array[nroots] of 0.5/(aij+akl) * t
///   g: output table of size nroots * dim_ij * dim_kl
fn buildG2d(
    nroots: usize,
    dim_ij: usize,
    dim_kl: usize,
    c00: []const f64,
    c0p: []const f64,
    b10: []const f64,
    b01: []const f64,
    b00: []const f64,
    g: []f64,
) void {
    const stride_k = dim_ij;

    // For each Rys root
    for (0..nroots) |r| {
        const base = r * dim_ij * dim_kl;

        // Base case: g(0, 0) = 1.0 (we'll multiply by weight*prefactor in z-component)
        g[base + 0 * stride_k + 0] = 1.0;

        // Build g(i, 0) for i = 1..dim_ij-1
        if (dim_ij > 1) {
            g[base + 0 * stride_k + 1] = c00[r];
        }
        {
            var i: usize = 2;
            while (i < dim_ij) : (i += 1) {
                g[base + 0 * stride_k + i] = c00[r] * g[base + 0 * stride_k + i - 1] +
                    @as(f64, @floatFromInt(i - 1)) * b10[r] * g[base + 0 * stride_k + i - 2];
            }
        }

        // Build g(i, k) for k = 1..dim_kl-1
        {
            var k: usize = 1;
            while (k < dim_kl) : (k += 1) {
                // g(0, k) = c0p * g(0, k-1) + (k-1) * b01 * g(0, k-2)
                var val = c0p[r] * g[base + (k - 1) * stride_k + 0];
                if (k >= 2) {
                    val += @as(f64, @floatFromInt(k - 1)) * b01[r] * g[base + (k - 2) * stride_k + 0];
                }
                g[base + k * stride_k + 0] = val;

                // g(i, k) for i = 1..dim_ij-1
                {
                    var i: usize = 1;
                    while (i < dim_ij) : (i += 1) {
                        val = c0p[r] * g[base + (k - 1) * stride_k + i] +
                            @as(f64, @floatFromInt(i)) * b00[r] * g[base + (k - 1) * stride_k + i - 1];
                        if (k >= 2) {
                            val += @as(f64, @floatFromInt(k - 1)) * b01[r] * g[base + (k - 2) * stride_k + i];
                        }
                        g[base + k * stride_k + i] = val;
                    }
                }
            }
        }
    }
}

/// Build the 2D recurrence table for the z-axis, which includes the
/// Rys weight * prefactor in the base case.
fn buildG2dWeighted(
    nroots: usize,
    dim_ij: usize,
    dim_kl: usize,
    c00: []const f64,
    c0p: []const f64,
    b10: []const f64,
    b01: []const f64,
    b00: []const f64,
    w_prefactor: []const f64,
    g: []f64,
) void {
    const stride_k = dim_ij;

    for (0..nroots) |r| {
        const base = r * dim_ij * dim_kl;

        // Base case includes weight * prefactor
        g[base + 0 * stride_k + 0] = w_prefactor[r];

        if (dim_ij > 1) {
            g[base + 0 * stride_k + 1] = c00[r] * w_prefactor[r];
        }
        {
            var i: usize = 2;
            while (i < dim_ij) : (i += 1) {
                g[base + 0 * stride_k + i] = c00[r] * g[base + 0 * stride_k + i - 1] +
                    @as(f64, @floatFromInt(i - 1)) * b10[r] * g[base + 0 * stride_k + i - 2];
            }
        }

        {
            var k: usize = 1;
            while (k < dim_kl) : (k += 1) {
                var val = c0p[r] * g[base + (k - 1) * stride_k + 0];
                if (k >= 2) {
                    val += @as(f64, @floatFromInt(k - 1)) * b01[r] * g[base + (k - 2) * stride_k + 0];
                }
                g[base + k * stride_k + 0] = val;

                {
                    var i: usize = 1;
                    while (i < dim_ij) : (i += 1) {
                        val = c0p[r] * g[base + (k - 1) * stride_k + i] +
                            @as(f64, @floatFromInt(i)) * b00[r] * g[base + (k - 1) * stride_k + i - 1];
                        if (k >= 2) {
                            val += @as(f64, @floatFromInt(k - 1)) * b01[r] * g[base + (k - 2) * stride_k + i];
                        }
                        g[base + k * stride_k + i] = val;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Horizontal Recurrence (2D -> 4D transfer)
// ============================================================================

/// Apply horizontal recurrence to transfer angular momentum from i to j.
///
/// Given g2d[nroots, dim_kl, i+j] (the "2D" table with combined ij index),
/// compute g4d[nroots, dim_kl, i, j] using:
///   g(i, j, k) = g(i+1, j-1, k) + (Ai - Bj)_axis * g(i, j-1, k)
///
/// We work in-place by building from j=0 upward.
/// The j=0 case is just copying from the 2D table.
///
/// This function processes one axis at a time and stores into a 4D-like
/// intermediate buffer.
/// Compute the full (ab|cd) shell quartet using Rys quadrature.
///
/// This is the main entry point, compatible with the Obara-Saika interface.
/// Output layout: output[ia * nb*nc*nd + ib * nc*nd + ic * nd + id]
///
/// Returns the number of ERIs computed (na * nb * nc * nd).
pub fn contractedShellQuartetERI(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    output: []f64,
) usize {
    const la: usize = shell_a.l;
    const lb: usize = shell_b.l;
    const lc: usize = shell_c.l;
    const ld: usize = shell_d.l;

    const na = basis_mod.numCartesian(@as(u32, @intCast(la)));
    const nb = basis_mod.numCartesian(@as(u32, @intCast(lb)));
    const nc = basis_mod.numCartesian(@as(u32, @intCast(lc)));
    const nd = basis_mod.numCartesian(@as(u32, @intCast(ld)));
    const total_out = na * nb * nc * nd;

    std.debug.assert(output.len >= total_out);

    // Zero output
    @memset(output[0..total_out], 0.0);

    const cart_a = basis_mod.cartesianExponents(@as(u32, @intCast(la)));
    const cart_b = basis_mod.cartesianExponents(@as(u32, @intCast(lb)));
    const cart_c = basis_mod.cartesianExponents(@as(u32, @intCast(lc)));
    const cart_d = basis_mod.cartesianExponents(@as(u32, @intCast(ld)));

    // Pre-compute normalization tables
    var norm_a: [MAX_NORM_TABLE]f64 = undefined;
    var norm_b: [MAX_NORM_TABLE]f64 = undefined;
    var norm_c: [MAX_NORM_TABLE]f64 = undefined;
    var norm_d: [MAX_NORM_TABLE]f64 = undefined;

    var max_norm_a: [MAX_PRIM]f64 = undefined;
    var max_norm_b: [MAX_PRIM]f64 = undefined;
    var max_norm_c: [MAX_PRIM]f64 = undefined;
    var max_norm_d: [MAX_PRIM]f64 = undefined;

    for (shell_a.primitives, 0..) |pa, ip| {
        var mx: f64 = 0.0;
        for (0..na) |ic| {
            const n_val = basis_mod.normalization(pa.alpha, cart_a[ic].x, cart_a[ic].y, cart_a[ic].z);
            norm_a[ip * na + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_a[ip] = mx;
    }
    for (shell_b.primitives, 0..) |pb, ip| {
        var mx: f64 = 0.0;
        for (0..nb) |ic| {
            const n_val = basis_mod.normalization(pb.alpha, cart_b[ic].x, cart_b[ic].y, cart_b[ic].z);
            norm_b[ip * nb + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_b[ip] = mx;
    }
    for (shell_c.primitives, 0..) |pc, ip| {
        var mx: f64 = 0.0;
        for (0..nc) |ic| {
            const n_val = basis_mod.normalization(pc.alpha, cart_c[ic].x, cart_c[ic].y, cart_c[ic].z);
            norm_c[ip * nc + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_c[ip] = mx;
    }
    for (shell_d.primitives, 0..) |pd, ip| {
        var mx: f64 = 0.0;
        for (0..nd) |ic| {
            const n_val = basis_mod.normalization(pd.alpha, cart_d[ic].x, cart_d[ic].y, cart_d[ic].z);
            norm_d[ip * nd + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_d[ip] = mx;
    }

    // Dimensions for the 2D recurrence tables
    // In the 2D recurrence, we need indices 0..la+lb for ij and 0..lc+ld for kl.
    const dim_ij = la + lb + 1;
    const dim_kl = lc + ld + 1;

    // Number of Rys roots
    const l_total = la + lb + lc + ld;
    const nroots = l_total / 2 + 1;
    std.debug.assert(nroots <= MAX_NROOTS);

    // AB and CD vectors (for horizontal recurrence)
    const ab = [3]f64{
        shell_a.center.x - shell_b.center.x,
        shell_a.center.y - shell_b.center.y,
        shell_a.center.z - shell_b.center.z,
    };
    const cd = [3]f64{
        shell_c.center.x - shell_d.center.x,
        shell_c.center.y - shell_d.center.y,
        shell_c.center.z - shell_d.center.z,
    };

    // Pre-compute distance-dependent quantities
    const diff_ab = math.Vec3.sub(shell_a.center, shell_b.center);
    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);
    const diff_cd = math.Vec3.sub(shell_c.center, shell_d.center);
    const r2_cd = math.Vec3.dot(diff_cd, diff_cd);

    // g-tables for x, y, z axes: each nroots * dim_ij * dim_kl
    const g_axis_size = nroots * dim_ij * dim_kl;
    var gx: [MAX_G_SIZE]f64 = undefined;
    var gy: [MAX_G_SIZE]f64 = undefined;
    var gz: [MAX_G_SIZE]f64 = undefined;

    // Temporary buffer for primitive ERIs (before contraction)
    var prim_eri: [MAX_CART * MAX_CART * MAX_CART * MAX_CART]f64 = undefined;

    // Rys roots and weights
    var rys_roots: [MAX_NROOTS]f64 = undefined;
    var rys_weights: [MAX_NROOTS]f64 = undefined;

    // Recurrence coefficients (per Rys root)
    var c00x: [MAX_NROOTS]f64 = undefined;
    var c00y: [MAX_NROOTS]f64 = undefined;
    var c00z: [MAX_NROOTS]f64 = undefined;
    var c0px: [MAX_NROOTS]f64 = undefined;
    var c0py: [MAX_NROOTS]f64 = undefined;
    var c0pz: [MAX_NROOTS]f64 = undefined;
    var b10_arr: [MAX_NROOTS]f64 = undefined;
    var b01_arr: [MAX_NROOTS]f64 = undefined;
    var b00_arr: [MAX_NROOTS]f64 = undefined;
    var w_pref: [MAX_NROOTS]f64 = undefined;

    // =========================================================
    // Pre-compute bra and ket primitive pairs
    // =========================================================
    var bra_pairs: [MAX_PRIM_PAIRS]BraPair = undefined;
    var n_bra: usize = 0;

    for (shell_a.primitives, 0..) |prim_a, ipa| {
        const alpha = prim_a.alpha;
        for (shell_b.primitives, 0..) |prim_b, ipb| {
            const beta = prim_b.alpha;
            const p_val = alpha + beta;
            const mu_ab = alpha * beta / p_val;
            bra_pairs[n_bra] = .{
                .p = p_val,
                .inv_2p = 0.5 / p_val,
                .px = (alpha * shell_a.center.x + beta * shell_b.center.x) / p_val,
                .py = (alpha * shell_a.center.y + beta * shell_b.center.y) / p_val,
                .pz = (alpha * shell_a.center.z + beta * shell_b.center.z) / p_val,
                .pax = (alpha * shell_a.center.x + beta * shell_b.center.x) / p_val - shell_a.center.x,
                .pay = (alpha * shell_a.center.y + beta * shell_b.center.y) / p_val - shell_a.center.y,
                .paz = (alpha * shell_a.center.z + beta * shell_b.center.z) / p_val - shell_a.center.z,
                .coeff_ab = prim_a.coeff * prim_b.coeff,
                .exp_ab = @exp(-mu_ab * r2_ab),
                .max_norm_ab = max_norm_a[ipa] * max_norm_b[ipb],
                .ipa = ipa,
                .ipb = ipb,
            };
            n_bra += 1;
        }
    }

    var ket_pairs: [MAX_PRIM_PAIRS]KetPair = undefined;
    var n_ket: usize = 0;

    for (shell_c.primitives, 0..) |prim_c, ipc| {
        const gamma_val = prim_c.alpha;
        for (shell_d.primitives, 0..) |prim_d, ipd| {
            const delta_val = prim_d.alpha;
            const q_val = gamma_val + delta_val;
            const mu_cd = gamma_val * delta_val / q_val;
            ket_pairs[n_ket] = .{
                .q = q_val,
                .inv_2q = 0.5 / q_val,
                .qx = (gamma_val * shell_c.center.x + delta_val * shell_d.center.x) / q_val,
                .qy = (gamma_val * shell_c.center.y + delta_val * shell_d.center.y) / q_val,
                .qz = (gamma_val * shell_c.center.z + delta_val * shell_d.center.z) / q_val,
                .qcx = (gamma_val * shell_c.center.x + delta_val * shell_d.center.x) / q_val - shell_c.center.x,
                .qcy = (gamma_val * shell_c.center.y + delta_val * shell_d.center.y) / q_val - shell_c.center.y,
                .qcz = (gamma_val * shell_c.center.z + delta_val * shell_d.center.z) / q_val - shell_c.center.z,
                .coeff_cd = prim_c.coeff * prim_d.coeff,
                .exp_cd = @exp(-mu_cd * r2_cd),
                .max_norm_cd = max_norm_c[ipc] * max_norm_d[ipd],
                .ipc = ipc,
                .ipd = ipd,
            };
            n_ket += 1;
        }
    }

    // Loop over pre-computed primitive pairs
    for (bra_pairs[0..n_bra]) |bra| {
        for (ket_pairs[0..n_ket]) |ket| {
            const p_val = bra.p;
            const q_val = ket.q;
            const exp_factor = bra.exp_ab * ket.exp_cd;
            const coeff_abcd = bra.coeff_ab * ket.coeff_cd;

            // Primitive screening
            const prefactor_bound = 2.0 * std.math.pow(f64, std.math.pi, 2.5) /
                (p_val * q_val * @sqrt(p_val + q_val)) * exp_factor;
            const max_norm_product = bra.max_norm_ab * ket.max_norm_cd;
            if (@abs(coeff_abcd) * max_norm_product * prefactor_bound < PRIM_SCREEN_THRESHOLD)
                continue;

            // Weighted center W = (p*P + q*Q) / (p+q)
            const pq = p_val + q_val;
            const wx = (p_val * bra.px + q_val * ket.qx) / pq;
            const wy = (p_val * bra.py + q_val * ket.qy) / pq;
            const wz = (p_val * bra.pz + q_val * ket.qz) / pq;

            // PQ distance squared
            const dpqx = bra.px - ket.qx;
            const dpqy = bra.py - ket.qy;
            const dpqz = bra.pz - ket.qz;
            const r2_pq = dpqx * dpqx + dpqy * dpqy + dpqz * dpqz;

            // Boys function argument
            const rho = p_val * q_val / pq;
            const arg = rho * r2_pq;

            // Compute Rys roots and weights
            rys_roots_mod.rysRoots(nroots, arg, &rys_roots, &rys_weights);

            // Prefactor
            const prefactor = prefactor_bound;

            // WP and WQ vectors
            const wpx = wx - bra.px;
            const wpy = wy - bra.py;
            const wpz = wz - bra.pz;
            const wqx = wx - ket.qx;
            const wqy = wy - ket.qy;
            const wqz = wz - ket.qz;

            for (0..nroots) |r| {
                const t2 = rys_roots[r]; // t² value, in [0, 1]

                b00_arr[r] = 0.5 / pq * t2;
                b10_arr[r] = bra.inv_2p * (1.0 - q_val / pq * t2);
                b01_arr[r] = ket.inv_2q * (1.0 - p_val / pq * t2);

                c00x[r] = bra.pax + wpx * t2;
                c00y[r] = bra.pay + wpy * t2;
                c00z[r] = bra.paz + wpz * t2;
                c0px[r] = ket.qcx + wqx * t2;
                c0py[r] = ket.qcy + wqy * t2;
                c0pz[r] = ket.qcz + wqz * t2;

                // Weight * prefactor goes into z-component base case
                w_pref[r] = rys_weights[r] * prefactor;
            }

            // Build 2D recurrence tables for x, y, z
            buildG2d(nroots, dim_ij, dim_kl, &c00x, &c0px, &b10_arr, &b01_arr, &b00_arr, gx[0..g_axis_size]);
            buildG2d(nroots, dim_ij, dim_kl, &c00y, &c0py, &b10_arr, &b01_arr, &b00_arr, gy[0..g_axis_size]);
            buildG2dWeighted(nroots, dim_ij, dim_kl, &c00z, &c0pz, &b10_arr, &b01_arr, &b00_arr, &w_pref, gz[0..g_axis_size]);

            // Extract ERIs using horizontal recurrence and component mapping
            // For each Cartesian component combination (ia, ib, ic, id),
            // the ERI is:
            //   (ab|cd) = Σ_r gx[r, ix, kx] * gy[r, iy, ky] * gz[r, iz, kz]
            //
            // Where the horizontal recurrence gives:
            //   g(a, b, c, d) uses indices:
            //     ix = combined index for ax+bx in the ij direction
            //     kx = combined index for cx+dx in the kl direction
            //
            // Horizontal recurrence (transfer from j to i, and from l to k):
            //   g(a,b) = g(a+1, b-1) + AB * g(a, b-1)
            //   g(c,d) = g(c+1, d-1) + CD * g(c, d-1)

            @memset(prim_eri[0..total_out], 0.0);

            // =========================================================
            // Iterative Horizontal Recurrence (HR)
            // =========================================================
            //
            // For each Rys root and each axis, build a 4-index table
            // from the 2D recurrence table g2d[i][k].
            //
            // Step 1 (Ket HR): For each i in 0..la+lb, build ket_hr[i][c][d]:
            //   Base: ket_hr[i][c][0] = g2d[i][c]  for c = 0..lc+ld
            //   Recurrence: ket_hr[i][c][d] = ket_hr[i][c+1][d-1] + CD * ket_hr[i][c][d-1]
            //   At step d, c ranges over 0..lc+ld-d (shrinks by 1 each step).
            //   We only need the final values at c=0..lc, d=0..ld.
            //
            // Step 2 (Bra HR): For each c=0..lc, d=0..ld, build bra_hr[a][b]:
            //   Base: bra_hr[a][0] = ket_hr[a][c][d]  for a = 0..la+lb
            //   Recurrence: bra_hr[a][b] = bra_hr[a+1][b-1] + AB * bra_hr[a][b-1]
            //   At step b, a ranges over 0..la+lb-b (shrinks by 1 each step).
            //   We only need the final value at a=0..la, b=0..lb.
            //
            // Ket HR table: dim_ij * dim_kl * (ld+1) entries
            //   Max (ff|ff): 7 * 7 * 4 = 196
            // Bra HR table: dim_ij * (lb+1) entries
            //   Max (ff|ff): 7 * 4 = 28

            const ld_1 = ld + 1;
            const lb_1 = lb + 1;
            const la_1 = la + 1;
            const lc_1 = lc + 1;

            // Ket HR table: ket[i * dim_kl * ld_1 + c * ld_1 + d]
            // At step d, valid c range is 0..dim_kl-1-d
            const ket_stride_c = ld_1;
            const ket_stride_i = dim_kl * ld_1;
            const ket_size = dim_ij * ket_stride_i;

            // Full 4D HR table per axis: hr4d[a * lb_1*lc_1*ld_1 + b*lc_1*ld_1 + c*ld_1 + d]
            const hr4d_d_stride: usize = 1;
            _ = hr4d_d_stride;
            const hr4d_c_stride = ld_1;
            const hr4d_b_stride = lc_1 * ld_1;
            const hr4d_a_stride = lb_1 * lc_1 * ld_1;
            const hr4d_size = la_1 * hr4d_a_stride;

            const MAX_KET_HR: usize = MAX_DIM_IJ * MAX_DIM_KL * (MAX_L + 1);
            const MAX_HR4D: usize = (MAX_L + 1) * (MAX_L + 1) * (MAX_L + 1) * (MAX_L + 1);

            var ket_hr_x: [MAX_KET_HR]f64 = undefined;
            var ket_hr_y: [MAX_KET_HR]f64 = undefined;
            var ket_hr_z: [MAX_KET_HR]f64 = undefined;
            var hr4d_x: [MAX_HR4D]f64 = undefined;
            var hr4d_y: [MAX_HR4D]f64 = undefined;
            var hr4d_z: [MAX_HR4D]f64 = undefined;

            for (0..nroots) |r| {
                const g2d_base = r * dim_ij * dim_kl;

                // Process each axis: build ket HR then bra HR into hr4d table
                inline for (0..3) |axis| {
                    const g_axis = switch (axis) {
                        0 => gx[0..g_axis_size],
                        1 => gy[0..g_axis_size],
                        2 => gz[0..g_axis_size],
                        else => unreachable,
                    };
                    const ab_val: f64 = ab[axis];
                    const cd_val: f64 = cd[axis];
                    const ket_hr = switch (axis) {
                        0 => ket_hr_x[0..ket_size],
                        1 => ket_hr_y[0..ket_size],
                        2 => ket_hr_z[0..ket_size],
                        else => unreachable,
                    };
                    const hr4d = switch (axis) {
                        0 => hr4d_x[0..hr4d_size],
                        1 => hr4d_y[0..hr4d_size],
                        2 => hr4d_z[0..hr4d_size],
                        else => unreachable,
                    };

                    // === Step 1: Ket HR ===
                    // Base case: d=0, ket[i][c][0] = g2d[i][c]
                    for (0..dim_ij) |i| {
                        for (0..dim_kl) |c| {
                            ket_hr[i * ket_stride_i + c * ket_stride_c + 0] = g_axis[g2d_base + c * dim_ij + i];
                        }
                    }
                    // Recurrence: d = 1..ld
                    {
                        var d: usize = 1;
                        while (d < ld_1) : (d += 1) {
                            const c_max = dim_kl - d;
                            for (0..dim_ij) |i| {
                                for (0..c_max) |c| {
                                    ket_hr[i * ket_stride_i + c * ket_stride_c + d] =
                                        ket_hr[i * ket_stride_i + (c + 1) * ket_stride_c + (d - 1)] +
                                        cd_val * ket_hr[i * ket_stride_i + c * ket_stride_c + (d - 1)];
                                }
                            }
                        }
                    }

                    // === Step 2: Bra HR → full 4D table ===
                    // For each (c, d) pair we need (c=0..lc, d=0..ld):
                    //   Base: hr4d[a][0][c][d] = ket_hr[a][c][d]  for a = 0..la+lb
                    //   Recurrence: hr4d[a][b][c][d] = hr4d[a+1][b-1][c][d] + AB * hr4d[a][b-1][c][d]
                    //
                    // We use a small work array per (c,d) pair to avoid storing
                    // the full intermediate dim_ij × lb_1 table.
                    for (0..lc_1) |c| {
                        for (0..ld_1) |d| {
                            // Work array: work[a] for the current b-step
                            // We iterate b from 0 upward, updating in-place
                            var work: [MAX_DIM_IJ]f64 = undefined;

                            // Base: b=0, work[a] = ket_hr[a][c][d]
                            for (0..dim_ij) |a| {
                                work[a] = ket_hr[a * ket_stride_i + c * ket_stride_c + d];
                            }

                            // Store b=0 values
                            for (0..la_1) |a| {
                                hr4d[a * hr4d_a_stride + 0 * hr4d_b_stride + c * hr4d_c_stride + d] = work[a];
                            }

                            // Recurrence: b = 1..lb
                            {
                                var b: usize = 1;
                                while (b < lb_1) : (b += 1) {
                                    // At step b, we need work[a] for a = 0..dim_ij-1-b
                                    const a_count = dim_ij - b;
                                    for (0..a_count) |a| {
                                        work[a] = work[a + 1] + ab_val * work[a];
                                    }
                                    // Store needed (a = 0..la) values
                                    for (0..la_1) |a| {
                                        hr4d[a * hr4d_a_stride + b * hr4d_b_stride + c * hr4d_c_stride + d] = work[a];
                                    }
                                }
                            }
                        }
                    }
                }

                // Extract ERIs: simple table lookups, no computation
                for (0..na) |ia| {
                    const ax = cart_a[ia].x;
                    const ay = cart_a[ia].y;
                    const az = cart_a[ia].z;
                    for (0..nb) |ib| {
                        const bx = cart_b[ib].x;
                        const by = cart_b[ib].y;
                        const bz = cart_b[ib].z;
                        for (0..nc) |ic| {
                            const cx = cart_c[ic].x;
                            const cy = cart_c[ic].y;
                            const cz = cart_c[ic].z;
                            for (0..nd) |id| {
                                const dx = cart_d[id].x;
                                const dy = cart_d[id].y;
                                const dz = cart_d[id].z;

                                const gx_val = hr4d_x[ax * hr4d_a_stride + bx * hr4d_b_stride + cx * hr4d_c_stride + dx];
                                const gy_val = hr4d_y[ay * hr4d_a_stride + by * hr4d_b_stride + cy * hr4d_c_stride + dy];
                                const gz_val = hr4d_z[az * hr4d_a_stride + bz * hr4d_b_stride + cz * hr4d_c_stride + dz];

                                prim_eri[ia * nb * nc * nd + ib * nc * nd + ic * nd + id] += gx_val * gy_val * gz_val;
                            }
                        }
                    }
                }
            }

            // Accumulate into output with contraction coefficients and normalization
            for (0..na) |ia| {
                const na_val = norm_a[bra.ipa * na + ia];
                for (0..nb) |ib| {
                    const nb_val = norm_b[bra.ipb * nb + ib];
                    for (0..nc) |ic| {
                        const nc_val = norm_c[ket.ipc * nc + ic];
                        for (0..nd) |id| {
                            const nd_val = norm_d[ket.ipd * nd + id];
                            const idx = ia * nb * nc * nd + ib * nc * nd + ic * nd + id;
                            output[idx] += coeff_abcd * na_val * nb_val * nc_val * nd_val * prim_eri[idx];
                        }
                    }
                }
            }
        }
    }

    return total_out;
}

// ============================================================================
// Shell-quartet ERI derivative w.r.t. center A
// ============================================================================

/// Compute derivative of (ab|cd) shell quartet with respect to center A.
///
/// d(ab|cd)/dA_x = 2*alpha_a * (a+1_x b|cd) - a_x * (a-1_x b|cd)
///
/// This is the "first center" derivative; combined with translational invariance
/// and the factor-of-2 strategy used in the gradient code, this gives the
/// complete 2-electron gradient contribution.
///
/// Output layout:
///   deriv_x[ia * nb*nc*nd + ib * nc*nd + ic * nd + id]
///   deriv_y[...], deriv_z[...]
///
/// The function stores alpha_a values per primitive pair so that the
/// differentiation formula can be applied inside the primitive loop.
///
/// Returns the number of derivative elements per component (na * nb * nc * nd).
pub fn contractedShellQuartetEriDeriv(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    deriv_x: []f64,
    deriv_y: []f64,
    deriv_z: []f64,
) usize {
    const la: usize = shell_a.l;
    const lb: usize = shell_b.l;
    const lc: usize = shell_c.l;
    const ld: usize = shell_d.l;

    const na = basis_mod.numCartesian(@as(u32, @intCast(la)));
    const nb = basis_mod.numCartesian(@as(u32, @intCast(lb)));
    const nc = basis_mod.numCartesian(@as(u32, @intCast(lc)));
    const nd = basis_mod.numCartesian(@as(u32, @intCast(ld)));
    const total_out = na * nb * nc * nd;

    std.debug.assert(deriv_x.len >= total_out);
    std.debug.assert(deriv_y.len >= total_out);
    std.debug.assert(deriv_z.len >= total_out);

    // Zero output
    @memset(deriv_x[0..total_out], 0.0);
    @memset(deriv_y[0..total_out], 0.0);
    @memset(deriv_z[0..total_out], 0.0);

    const cart_a = basis_mod.cartesianExponents(@as(u32, @intCast(la)));
    const cart_b = basis_mod.cartesianExponents(@as(u32, @intCast(lb)));
    const cart_c = basis_mod.cartesianExponents(@as(u32, @intCast(lc)));
    const cart_d = basis_mod.cartesianExponents(@as(u32, @intCast(ld)));

    // Pre-compute normalization tables
    var norm_a: [MAX_NORM_TABLE]f64 = undefined;
    var norm_b: [MAX_NORM_TABLE]f64 = undefined;
    var norm_c: [MAX_NORM_TABLE]f64 = undefined;
    var norm_d: [MAX_NORM_TABLE]f64 = undefined;

    var max_norm_a: [MAX_PRIM]f64 = undefined;
    var max_norm_b: [MAX_PRIM]f64 = undefined;
    var max_norm_c: [MAX_PRIM]f64 = undefined;
    var max_norm_d: [MAX_PRIM]f64 = undefined;

    for (shell_a.primitives, 0..) |pa, ip| {
        var mx: f64 = 0.0;
        for (0..na) |ic| {
            const n_val = basis_mod.normalization(pa.alpha, cart_a[ic].x, cart_a[ic].y, cart_a[ic].z);
            norm_a[ip * na + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_a[ip] = mx;
    }
    for (shell_b.primitives, 0..) |pb, ip| {
        var mx: f64 = 0.0;
        for (0..nb) |ic| {
            const n_val = basis_mod.normalization(pb.alpha, cart_b[ic].x, cart_b[ic].y, cart_b[ic].z);
            norm_b[ip * nb + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_b[ip] = mx;
    }
    for (shell_c.primitives, 0..) |pc, ip| {
        var mx: f64 = 0.0;
        for (0..nc) |ic| {
            const n_val = basis_mod.normalization(pc.alpha, cart_c[ic].x, cart_c[ic].y, cart_c[ic].z);
            norm_c[ip * nc + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_c[ip] = mx;
    }
    for (shell_d.primitives, 0..) |pd, ip| {
        var mx: f64 = 0.0;
        for (0..nd) |ic| {
            const n_val = basis_mod.normalization(pd.alpha, cart_d[ic].x, cart_d[ic].y, cart_d[ic].z);
            norm_d[ip * nd + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_d[ip] = mx;
    }

    // Dimensions for the 2D recurrence tables
    // For derivatives, we need la+1 index in the ij direction, so:
    //   dim_ij = (la+1) + lb + 1 = la + lb + 2
    const dim_ij = la + lb + 2;
    const dim_kl = lc + ld + 1;

    // Number of Rys roots: must accommodate the augmented l_total
    //   (la+1) + lb + lc + ld = l_total + 1
    const l_total = la + lb + lc + ld;
    const nroots = (l_total + 1) / 2 + 1;
    std.debug.assert(nroots <= MAX_NROOTS);

    // AB and CD vectors (for horizontal recurrence)
    const ab = [3]f64{
        shell_a.center.x - shell_b.center.x,
        shell_a.center.y - shell_b.center.y,
        shell_a.center.z - shell_b.center.z,
    };
    const cd = [3]f64{
        shell_c.center.x - shell_d.center.x,
        shell_c.center.y - shell_d.center.y,
        shell_c.center.z - shell_d.center.z,
    };

    // Pre-compute distance-dependent quantities
    const diff_ab = math.Vec3.sub(shell_a.center, shell_b.center);
    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);
    const diff_cd = math.Vec3.sub(shell_c.center, shell_d.center);
    const r2_cd = math.Vec3.dot(diff_cd, diff_cd);

    // g-tables for x, y, z axes: each nroots * dim_ij * dim_kl
    const g_axis_size = nroots * dim_ij * dim_kl;
    var gx: [MAX_NROOTS * (2 * MAX_L + 2) * (2 * MAX_L + 1)]f64 = undefined;
    var gy: [MAX_NROOTS * (2 * MAX_L + 2) * (2 * MAX_L + 1)]f64 = undefined;
    var gz: [MAX_NROOTS * (2 * MAX_L + 2) * (2 * MAX_L + 1)]f64 = undefined;

    // Temporary buffer for primitive derivative ERIs (before contraction)
    var prim_deriv_x: [MAX_CART * MAX_CART * MAX_CART * MAX_CART]f64 = undefined;
    var prim_deriv_y: [MAX_CART * MAX_CART * MAX_CART * MAX_CART]f64 = undefined;
    var prim_deriv_z: [MAX_CART * MAX_CART * MAX_CART * MAX_CART]f64 = undefined;

    // Rys roots and weights
    var rys_roots: [MAX_NROOTS]f64 = undefined;
    var rys_weights: [MAX_NROOTS]f64 = undefined;

    // Recurrence coefficients (per Rys root)
    var c00x: [MAX_NROOTS]f64 = undefined;
    var c00y: [MAX_NROOTS]f64 = undefined;
    var c00z: [MAX_NROOTS]f64 = undefined;
    var c0px: [MAX_NROOTS]f64 = undefined;
    var c0py: [MAX_NROOTS]f64 = undefined;
    var c0pz: [MAX_NROOTS]f64 = undefined;
    var b10_arr: [MAX_NROOTS]f64 = undefined;
    var b01_arr: [MAX_NROOTS]f64 = undefined;
    var b00_arr: [MAX_NROOTS]f64 = undefined;
    var w_pref: [MAX_NROOTS]f64 = undefined;

    // =========================================================
    // Pre-compute bra and ket primitive pairs
    // BraPair now also stores alpha_a for differentiation
    // =========================================================
    var bra_pairs: [MAX_PRIM_PAIRS]BraPair = undefined;
    var bra_alpha_a: [MAX_PRIM_PAIRS]f64 = undefined;
    var n_bra: usize = 0;

    for (shell_a.primitives, 0..) |prim_a, ipa| {
        const alpha = prim_a.alpha;
        for (shell_b.primitives, 0..) |prim_b, ipb| {
            const beta = prim_b.alpha;
            const p_val = alpha + beta;
            const mu_ab = alpha * beta / p_val;
            bra_alpha_a[n_bra] = alpha;
            bra_pairs[n_bra] = .{
                .p = p_val,
                .inv_2p = 0.5 / p_val,
                .px = (alpha * shell_a.center.x + beta * shell_b.center.x) / p_val,
                .py = (alpha * shell_a.center.y + beta * shell_b.center.y) / p_val,
                .pz = (alpha * shell_a.center.z + beta * shell_b.center.z) / p_val,
                .pax = (alpha * shell_a.center.x + beta * shell_b.center.x) / p_val - shell_a.center.x,
                .pay = (alpha * shell_a.center.y + beta * shell_b.center.y) / p_val - shell_a.center.y,
                .paz = (alpha * shell_a.center.z + beta * shell_b.center.z) / p_val - shell_a.center.z,
                .coeff_ab = prim_a.coeff * prim_b.coeff,
                .exp_ab = @exp(-mu_ab * r2_ab),
                .max_norm_ab = max_norm_a[ipa] * max_norm_b[ipb],
                .ipa = ipa,
                .ipb = ipb,
            };
            n_bra += 1;
        }
    }

    var ket_pairs: [MAX_PRIM_PAIRS]KetPair = undefined;
    var n_ket: usize = 0;

    for (shell_c.primitives, 0..) |prim_c, ipc| {
        const gamma_val = prim_c.alpha;
        for (shell_d.primitives, 0..) |prim_d, ipd| {
            const delta_val = prim_d.alpha;
            const q_val = gamma_val + delta_val;
            const mu_cd = gamma_val * delta_val / q_val;
            ket_pairs[n_ket] = .{
                .q = q_val,
                .inv_2q = 0.5 / q_val,
                .qx = (gamma_val * shell_c.center.x + delta_val * shell_d.center.x) / q_val,
                .qy = (gamma_val * shell_c.center.y + delta_val * shell_d.center.y) / q_val,
                .qz = (gamma_val * shell_c.center.z + delta_val * shell_d.center.z) / q_val,
                .qcx = (gamma_val * shell_c.center.x + delta_val * shell_d.center.x) / q_val - shell_c.center.x,
                .qcy = (gamma_val * shell_c.center.y + delta_val * shell_d.center.y) / q_val - shell_c.center.y,
                .qcz = (gamma_val * shell_c.center.z + delta_val * shell_d.center.z) / q_val - shell_c.center.z,
                .coeff_cd = prim_c.coeff * prim_d.coeff,
                .exp_cd = @exp(-mu_cd * r2_cd),
                .max_norm_cd = max_norm_c[ipc] * max_norm_d[ipd],
                .ipc = ipc,
                .ipd = ipd,
            };
            n_ket += 1;
        }
    }

    // HR table constants for the derivative case
    // la_1 for derivative: need la+1 indices (0..la+1), so la_1 = la + 2
    const ld_1 = ld + 1;
    const lb_1 = lb + 1;
    const la_1_deriv = la + 2; // Extra +1 for the derivative
    const lc_1 = lc + 1;

    // Ket HR table
    const ket_stride_c = ld_1;
    const ket_stride_i = dim_kl * ld_1;
    const ket_size = dim_ij * ket_stride_i;

    // Full 4D HR table per axis: hr4d[a * lb_1*lc_1*ld_1 + b*lc_1*ld_1 + c*ld_1 + d]
    const hr4d_c_stride = ld_1;
    const hr4d_b_stride = lc_1 * ld_1;
    const hr4d_a_stride = lb_1 * lc_1 * ld_1;
    const hr4d_size = la_1_deriv * hr4d_a_stride;

    const MAX_KET_HR_DERIV: usize = (2 * MAX_L + 2) * (2 * MAX_L + 1) * (MAX_L + 1);
    const MAX_HR4D_DERIV: usize = (MAX_L + 2) * (MAX_L + 1) * (MAX_L + 1) * (MAX_L + 1);

    var ket_hr_x: [MAX_KET_HR_DERIV]f64 = undefined;
    var ket_hr_y: [MAX_KET_HR_DERIV]f64 = undefined;
    var ket_hr_z: [MAX_KET_HR_DERIV]f64 = undefined;
    var hr4d_x: [MAX_HR4D_DERIV]f64 = undefined;
    var hr4d_y: [MAX_HR4D_DERIV]f64 = undefined;
    var hr4d_z: [MAX_HR4D_DERIV]f64 = undefined;

    // Loop over pre-computed primitive pairs
    for (bra_pairs[0..n_bra], 0..) |bra, ibra| {
        const alpha_a_val = bra_alpha_a[ibra];

        for (ket_pairs[0..n_ket]) |ket| {
            const p_val = bra.p;
            const q_val = ket.q;
            const exp_factor = bra.exp_ab * ket.exp_cd;
            const coeff_abcd = bra.coeff_ab * ket.coeff_cd;

            // Primitive screening
            const prefactor_bound = 2.0 * std.math.pow(f64, std.math.pi, 2.5) /
                (p_val * q_val * @sqrt(p_val + q_val)) * exp_factor;
            const max_norm_product = bra.max_norm_ab * ket.max_norm_cd;
            if (@abs(coeff_abcd) * max_norm_product * prefactor_bound < PRIM_SCREEN_THRESHOLD)
                continue;

            // Weighted center W = (p*P + q*Q) / (p+q)
            const pq = p_val + q_val;
            const wx = (p_val * bra.px + q_val * ket.qx) / pq;
            const wy = (p_val * bra.py + q_val * ket.qy) / pq;
            const wz = (p_val * bra.pz + q_val * ket.qz) / pq;

            // PQ distance squared
            const dpqx = bra.px - ket.qx;
            const dpqy = bra.py - ket.qy;
            const dpqz = bra.pz - ket.qz;
            const r2_pq = dpqx * dpqx + dpqy * dpqy + dpqz * dpqz;

            // Boys function argument
            const rho = p_val * q_val / pq;
            const arg = rho * r2_pq;

            // Compute Rys roots and weights
            rys_roots_mod.rysRoots(nroots, arg, &rys_roots, &rys_weights);

            // Prefactor
            const prefactor = prefactor_bound;

            // WP and WQ vectors
            const wpx = wx - bra.px;
            const wpy = wy - bra.py;
            const wpz = wz - bra.pz;
            const wqx = wx - ket.qx;
            const wqy = wy - ket.qy;
            const wqz = wz - ket.qz;

            for (0..nroots) |r| {
                const t2 = rys_roots[r];

                b00_arr[r] = 0.5 / pq * t2;
                b10_arr[r] = bra.inv_2p * (1.0 - q_val / pq * t2);
                b01_arr[r] = ket.inv_2q * (1.0 - p_val / pq * t2);

                c00x[r] = bra.pax + wpx * t2;
                c00y[r] = bra.pay + wpy * t2;
                c00z[r] = bra.paz + wpz * t2;
                c0px[r] = ket.qcx + wqx * t2;
                c0py[r] = ket.qcy + wqy * t2;
                c0pz[r] = ket.qcz + wqz * t2;

                w_pref[r] = rys_weights[r] * prefactor;
            }

            // Build 2D recurrence tables with expanded dim_ij for derivative
            buildG2d(nroots, dim_ij, dim_kl, &c00x, &c0px, &b10_arr, &b01_arr, &b00_arr, gx[0..g_axis_size]);
            buildG2d(nroots, dim_ij, dim_kl, &c00y, &c0py, &b10_arr, &b01_arr, &b00_arr, gy[0..g_axis_size]);
            buildG2dWeighted(nroots, dim_ij, dim_kl, &c00z, &c0pz, &b10_arr, &b01_arr, &b00_arr, &w_pref, gz[0..g_axis_size]);

            // Zero primitive derivative buffers
            @memset(prim_deriv_x[0..total_out], 0.0);
            @memset(prim_deriv_y[0..total_out], 0.0);
            @memset(prim_deriv_z[0..total_out], 0.0);

            // =========================================================
            // Iterative Horizontal Recurrence (HR) — same as energy but
            // with expanded la range (0..la+1 instead of 0..la)
            // =========================================================

            for (0..nroots) |r| {
                const g2d_base = r * dim_ij * dim_kl;

                // Process each axis
                inline for (0..3) |axis| {
                    const g_axis = switch (axis) {
                        0 => gx[0..g_axis_size],
                        1 => gy[0..g_axis_size],
                        2 => gz[0..g_axis_size],
                        else => unreachable,
                    };
                    const ab_val: f64 = ab[axis];
                    const cd_val: f64 = cd[axis];
                    const ket_hr = switch (axis) {
                        0 => ket_hr_x[0..ket_size],
                        1 => ket_hr_y[0..ket_size],
                        2 => ket_hr_z[0..ket_size],
                        else => unreachable,
                    };
                    const hr4d = switch (axis) {
                        0 => hr4d_x[0..hr4d_size],
                        1 => hr4d_y[0..hr4d_size],
                        2 => hr4d_z[0..hr4d_size],
                        else => unreachable,
                    };

                    // === Step 1: Ket HR ===
                    for (0..dim_ij) |i| {
                        for (0..dim_kl) |c| {
                            ket_hr[i * ket_stride_i + c * ket_stride_c + 0] = g_axis[g2d_base + c * dim_ij + i];
                        }
                    }
                    {
                        var d: usize = 1;
                        while (d < ld_1) : (d += 1) {
                            const c_max = dim_kl - d;
                            for (0..dim_ij) |i| {
                                for (0..c_max) |c| {
                                    ket_hr[i * ket_stride_i + c * ket_stride_c + d] =
                                        ket_hr[i * ket_stride_i + (c + 1) * ket_stride_c + (d - 1)] +
                                        cd_val * ket_hr[i * ket_stride_i + c * ket_stride_c + (d - 1)];
                                }
                            }
                        }
                    }

                    // === Step 2: Bra HR → full 4D table with la_1_deriv ===
                    for (0..lc_1) |c| {
                        for (0..ld_1) |d| {
                            var work: [2 * MAX_L + 2]f64 = undefined;

                            for (0..dim_ij) |a| {
                                work[a] = ket_hr[a * ket_stride_i + c * ket_stride_c + d];
                            }

                            // Store b=0 values (need 0..la+1 for derivative)
                            for (0..la_1_deriv) |a| {
                                hr4d[a * hr4d_a_stride + 0 * hr4d_b_stride + c * hr4d_c_stride + d] = work[a];
                            }

                            // Recurrence: b = 1..lb
                            {
                                var b: usize = 1;
                                while (b < lb_1) : (b += 1) {
                                    const a_count = dim_ij - b;
                                    for (0..a_count) |a| {
                                        work[a] = work[a + 1] + ab_val * work[a];
                                    }
                                    for (0..la_1_deriv) |a| {
                                        hr4d[a * hr4d_a_stride + b * hr4d_b_stride + c * hr4d_c_stride + d] = work[a];
                                    }
                                }
                            }
                        }
                    }
                }

                // =========================================================
                // Extract derivative ERIs using the differentiation formula:
                //   d(ab|cd)/dA_x = 2*alpha_a * hr4d[ax+1][bx][cx][dx] * ...
                //                 - ax * hr4d[ax-1][bx][cx][dx] * ...
                // (product over y and z uses undifferentiated hr4d values)
                // =========================================================

                for (0..na) |ia| {
                    const ax = cart_a[ia].x;
                    const ay = cart_a[ia].y;
                    const az = cart_a[ia].z;
                    for (0..nb) |ib| {
                        const bx = cart_b[ib].x;
                        const by = cart_b[ib].y;
                        const bz = cart_b[ib].z;
                        for (0..nc) |ic| {
                            const cx = cart_c[ic].x;
                            const cy = cart_c[ic].y;
                            const cz = cart_c[ic].z;
                            for (0..nd) |id| {
                                const dx = cart_d[id].x;
                                const dy = cart_d[id].y;
                                const dz = cart_d[id].z;

                                const idx = ia * nb * nc * nd + ib * nc * nd + ic * nd + id;

                                // Undifferentiated y,z products
                                const gy_val = hr4d_y[ay * hr4d_a_stride + by * hr4d_b_stride + cy * hr4d_c_stride + dy];
                                const gz_val = hr4d_z[az * hr4d_a_stride + bz * hr4d_b_stride + cz * hr4d_c_stride + dz];

                                // x-derivative: d/dAx
                                {
                                    // 2*alpha * (a+1_x, b, c, d)
                                    const gx_plus = hr4d_x[(ax + 1) * hr4d_a_stride + bx * hr4d_b_stride + cx * hr4d_c_stride + dx];
                                    var dval = 2.0 * alpha_a_val * gx_plus * gy_val * gz_val;
                                    // -ax * (a-1_x, b, c, d)
                                    if (ax > 0) {
                                        const gx_minus = hr4d_x[(ax - 1) * hr4d_a_stride + bx * hr4d_b_stride + cx * hr4d_c_stride + dx];
                                        dval -= @as(f64, @floatFromInt(ax)) * gx_minus * gy_val * gz_val;
                                    }
                                    prim_deriv_x[idx] += dval;
                                }

                                // Undifferentiated x,z products
                                const gx_val = hr4d_x[ax * hr4d_a_stride + bx * hr4d_b_stride + cx * hr4d_c_stride + dx];

                                // y-derivative: d/dAy
                                {
                                    const gy_plus = hr4d_y[(ay + 1) * hr4d_a_stride + by * hr4d_b_stride + cy * hr4d_c_stride + dy];
                                    var dval = 2.0 * alpha_a_val * gy_plus * gx_val * gz_val;
                                    if (ay > 0) {
                                        const gy_minus = hr4d_y[(ay - 1) * hr4d_a_stride + by * hr4d_b_stride + cy * hr4d_c_stride + dy];
                                        dval -= @as(f64, @floatFromInt(ay)) * gy_minus * gx_val * gz_val;
                                    }
                                    prim_deriv_y[idx] += dval;
                                }

                                // z-derivative: d/dAz
                                {
                                    const gz_plus = hr4d_z[(az + 1) * hr4d_a_stride + bz * hr4d_b_stride + cz * hr4d_c_stride + dz];
                                    var dval = 2.0 * alpha_a_val * gz_plus * gx_val * gy_val;
                                    if (az > 0) {
                                        const gz_minus = hr4d_z[(az - 1) * hr4d_a_stride + bz * hr4d_b_stride + cz * hr4d_c_stride + dz];
                                        dval -= @as(f64, @floatFromInt(az)) * gz_minus * gx_val * gy_val;
                                    }
                                    prim_deriv_z[idx] += dval;
                                }
                            }
                        }
                    }
                }
            }

            // Accumulate into output with contraction coefficients and normalization
            for (0..na) |ia| {
                const na_val = norm_a[bra.ipa * na + ia];
                for (0..nb) |ib| {
                    const nb_val = norm_b[bra.ipb * nb + ib];
                    for (0..nc) |ic| {
                        const nc_val = norm_c[ket.ipc * nc + ic];
                        for (0..nd) |id| {
                            const nd_val = norm_d[ket.ipd * nd + id];
                            const idx = ia * nb * nc * nd + ib * nc * nd + ic * nd + id;
                            const c_norm = coeff_abcd * na_val * nb_val * nc_val * nd_val;
                            deriv_x[idx] += c_norm * prim_deriv_x[idx];
                            deriv_y[idx] += c_norm * prim_deriv_y[idx];
                            deriv_z[idx] += c_norm * prim_deriv_z[idx];
                        }
                    }
                }
            }
        }
    }

    return total_out;
}

// ============================================================================
// Tests
// ============================================================================

const obara_saika = @import("obara_saika.zig");

test "rys ERI vs OS: (ss|ss) H2 case" {
    const testing = std.testing;
    const tol: f64 = 1e-10;

    // Two s-type shells (H atoms) with STO-3G-like single primitive
    const shell_a = ContractedShell{
        .l = 0,
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.42525, .coeff = 1.0 },
        },
    };
    const shell_b = ContractedShell{
        .l = 0,
        .center = .{ .x = 0.0, .y = 0.0, .z = 1.4 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.42525, .coeff = 1.0 },
        },
    };

    var output_rys: [1]f64 = undefined;
    var output_os: [1]f64 = undefined;

    _ = contractedShellQuartetERI(shell_a, shell_b, shell_a, shell_b, &output_rys);
    _ = obara_saika.contractedShellQuartetERI(shell_a, shell_b, shell_a, shell_b, &output_os);

    try testing.expectApproxEqAbs(output_os[0], output_rys[0], tol);
}

test "rys ERI vs OS: (sp|sp) case" {
    const testing = std.testing;
    const tol: f64 = 1e-10;

    const shell_s = ContractedShell{
        .l = 0,
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 5.0, .coeff = 0.8 },
            .{ .alpha = 1.5, .coeff = 0.3 },
        },
    };
    const shell_p = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.0, .y = 0.0, .z = 1.2 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.0, .coeff = 0.7 },
            .{ .alpha = 0.8, .coeff = 0.4 },
        },
    };

    const na = basis_mod.numCartesian(0); // 1
    const nb = basis_mod.numCartesian(1); // 3
    const total = na * nb * na * nb; // 9
    var output_rys: [9]f64 = undefined;
    var output_os: [9]f64 = undefined;

    _ = contractedShellQuartetERI(shell_s, shell_p, shell_s, shell_p, &output_rys);
    _ = obara_saika.contractedShellQuartetERI(shell_s, shell_p, shell_s, shell_p, &output_os);

    for (0..total) |i| {
        try testing.expectApproxEqAbs(output_os[i], output_rys[i], tol);
    }
}

test "rys ERI vs OS: (pp|pp) case" {
    const testing = std.testing;
    const tol: f64 = 1e-10;

    const shell_p1 = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 4.0, .coeff = 0.6 },
            .{ .alpha = 1.0, .coeff = 0.4 },
        },
    };
    const shell_p2 = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.5, .y = 0.3, .z = 1.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.5, .coeff = 0.5 },
            .{ .alpha = 0.9, .coeff = 0.5 },
        },
    };

    const np = basis_mod.numCartesian(1); // 3
    const total = np * np * np * np; // 81
    var output_rys: [81]f64 = undefined;
    var output_os: [81]f64 = undefined;

    _ = contractedShellQuartetERI(shell_p1, shell_p2, shell_p1, shell_p2, &output_rys);
    _ = obara_saika.contractedShellQuartetERI(shell_p1, shell_p2, shell_p1, shell_p2, &output_os);

    for (0..total) |i| {
        try testing.expectApproxEqAbs(output_os[i], output_rys[i], tol);
    }
}

test "rys ERI vs OS: (dd|dd) case" {
    const testing = std.testing;
    const tol: f64 = 1e-9;

    const shell_d1 = ContractedShell{
        .l = 2,
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 2.5, .coeff = 1.0 },
        },
    };
    const shell_d2 = ContractedShell{
        .l = 2,
        .center = .{ .x = 0.3, .y = -0.2, .z = 0.8 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 1.8, .coeff = 1.0 },
        },
    };

    const nd = basis_mod.numCartesian(2); // 6
    const total = nd * nd * nd * nd; // 1296
    var output_rys: [1296]f64 = undefined;
    var output_os: [1296]f64 = undefined;

    _ = contractedShellQuartetERI(shell_d1, shell_d2, shell_d1, shell_d2, &output_rys);
    _ = obara_saika.contractedShellQuartetERI(shell_d1, shell_d2, shell_d1, shell_d2, &output_os);

    for (0..total) |i| {
        try testing.expectApproxEqAbs(output_os[i], output_rys[i], tol);
    }
}

test "rys ERI vs OS: (sd|ps) mixed angular momentum" {
    const testing = std.testing;
    const tol: f64 = 1e-10;

    const shell_s = ContractedShell{
        .l = 0,
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 5.0, .coeff = 1.0 },
        },
    };
    const shell_p = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.0, .y = 0.0, .z = 1.5 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.0, .coeff = 1.0 },
        },
    };
    const shell_d = ContractedShell{
        .l = 2,
        .center = .{ .x = 0.5, .y = 0.3, .z = 0.8 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 2.0, .coeff = 1.0 },
        },
    };

    const ns = basis_mod.numCartesian(0); // 1
    const np = basis_mod.numCartesian(1); // 3
    const nd = basis_mod.numCartesian(2); // 6
    const total = ns * nd * np * ns; // 1*6*3*1 = 18
    var output_rys: [18]f64 = undefined;
    var output_os: [18]f64 = undefined;

    _ = contractedShellQuartetERI(shell_s, shell_d, shell_p, shell_s, &output_rys);
    _ = obara_saika.contractedShellQuartetERI(shell_s, shell_d, shell_p, shell_s, &output_os);

    for (0..total) |i| {
        try testing.expectApproxEqAbs(output_os[i], output_rys[i], tol);
    }
}

test "rys ERI deriv vs FD: (ss|ss) case" {
    const testing = std.testing;
    const tol: f64 = 1e-7;
    const delta: f64 = 1e-5;

    const shell_a = ContractedShell{
        .l = 0,
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.42525, .coeff = 1.0 },
        },
    };
    const shell_b = ContractedShell{
        .l = 0,
        .center = .{ .x = 0.0, .y = 0.0, .z = 1.4 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.42525, .coeff = 1.0 },
        },
    };

    var dx_buf: [1]f64 = undefined;
    var dy_buf: [1]f64 = undefined;
    var dz_buf: [1]f64 = undefined;
    _ = contractedShellQuartetEriDeriv(shell_a, shell_b, shell_a, shell_b, &dx_buf, &dy_buf, &dz_buf);

    // Finite difference for each direction
    const dirs = [3][3]f64{
        .{ delta, 0.0, 0.0 },
        .{ 0.0, delta, 0.0 },
        .{ 0.0, 0.0, delta },
    };
    const analytical = [3]f64{ dx_buf[0], dy_buf[0], dz_buf[0] };

    for (dirs, 0..) |d, i| {
        const shell_a_p = ContractedShell{
            .l = 0,
            .center = .{ .x = d[0], .y = d[1], .z = d[2] },
            .primitives = shell_a.primitives,
        };
        const shell_a_m = ContractedShell{
            .l = 0,
            .center = .{ .x = -d[0], .y = -d[1], .z = -d[2] },
            .primitives = shell_a.primitives,
        };
        var eri_p: [1]f64 = undefined;
        var eri_m: [1]f64 = undefined;
        _ = contractedShellQuartetERI(shell_a_p, shell_b, shell_a, shell_b, &eri_p);
        _ = contractedShellQuartetERI(shell_a_m, shell_b, shell_a, shell_b, &eri_m);
        const fd = (eri_p[0] - eri_m[0]) / (2.0 * delta);
        try testing.expectApproxEqAbs(fd, analytical[i], tol);
    }
}

test "rys ERI deriv vs FD: (sp|sp) case" {
    const testing = std.testing;
    const tol: f64 = 1e-7;
    const delta: f64 = 1e-5;

    const shell_s = ContractedShell{
        .l = 0,
        .center = .{ .x = 0.1, .y = -0.2, .z = 0.3 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 5.0, .coeff = 0.8 },
            .{ .alpha = 1.5, .coeff = 0.3 },
        },
    };
    const shell_p = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.0, .y = 0.0, .z = 1.2 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.0, .coeff = 0.7 },
            .{ .alpha = 0.8, .coeff = 0.4 },
        },
    };

    const ns = basis_mod.numCartesian(0);
    const np_val = basis_mod.numCartesian(1);
    const total = ns * np_val * ns * np_val;

    var dx_arr: [9]f64 = undefined;
    var dy_arr: [9]f64 = undefined;
    var dz_arr: [9]f64 = undefined;
    _ = contractedShellQuartetEriDeriv(shell_s, shell_p, shell_s, shell_p, &dx_arr, &dy_arr, &dz_arr);

    // FD in z-direction for center A (shell_s center)
    const shell_s_p = ContractedShell{
        .l = 0,
        .center = .{ .x = shell_s.center.x, .y = shell_s.center.y, .z = shell_s.center.z + delta },
        .primitives = shell_s.primitives,
    };
    const shell_s_m = ContractedShell{
        .l = 0,
        .center = .{ .x = shell_s.center.x, .y = shell_s.center.y, .z = shell_s.center.z - delta },
        .primitives = shell_s.primitives,
    };

    var eri_p: [9]f64 = undefined;
    var eri_m: [9]f64 = undefined;
    _ = contractedShellQuartetERI(shell_s_p, shell_p, shell_s, shell_p, &eri_p);
    _ = contractedShellQuartetERI(shell_s_m, shell_p, shell_s, shell_p, &eri_m);

    for (0..total) |i| {
        const fd = (eri_p[i] - eri_m[i]) / (2.0 * delta);
        try testing.expectApproxEqAbs(fd, dz_arr[i], tol);
    }
}

test "rys ERI deriv vs FD: (pp|pp) case" {
    const testing = std.testing;
    const tol: f64 = 1e-7;
    const delta: f64 = 1e-5;

    const shell_p1 = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 4.0, .coeff = 0.6 },
            .{ .alpha = 1.0, .coeff = 0.4 },
        },
    };
    const shell_p2 = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.5, .y = 0.3, .z = 1.0 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.5, .coeff = 0.5 },
            .{ .alpha = 0.9, .coeff = 0.5 },
        },
    };

    const np_v = basis_mod.numCartesian(1);
    const total = np_v * np_v * np_v * np_v;

    var dx_arr: [81]f64 = undefined;
    var dy_arr: [81]f64 = undefined;
    var dz_arr: [81]f64 = undefined;
    _ = contractedShellQuartetEriDeriv(shell_p1, shell_p2, shell_p1, shell_p2, &dx_arr, &dy_arr, &dz_arr);

    // FD in y-direction for center A (shell_p1 center)
    const shell_p1_p = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.0, .y = delta, .z = 0.0 },
        .primitives = shell_p1.primitives,
    };
    const shell_p1_m = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.0, .y = -delta, .z = 0.0 },
        .primitives = shell_p1.primitives,
    };

    var eri_p: [81]f64 = undefined;
    var eri_m: [81]f64 = undefined;
    _ = contractedShellQuartetERI(shell_p1_p, shell_p2, shell_p1, shell_p2, &eri_p);
    _ = contractedShellQuartetERI(shell_p1_m, shell_p2, shell_p1, shell_p2, &eri_m);

    for (0..total) |i| {
        const fd = (eri_p[i] - eri_m[i]) / (2.0 * delta);
        try testing.expectApproxEqAbs(fd, dy_arr[i], tol);
    }
}

test "rys ERI deriv vs FD: (sd|ps) mixed case" {
    const testing = std.testing;
    const tol: f64 = 1e-7;
    const delta: f64 = 1e-5;

    const shell_s = ContractedShell{
        .l = 0,
        .center = .{ .x = 0.1, .y = 0.2, .z = -0.1 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 5.0, .coeff = 1.0 },
        },
    };
    const shell_p = ContractedShell{
        .l = 1,
        .center = .{ .x = 0.0, .y = 0.0, .z = 1.5 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 3.0, .coeff = 1.0 },
        },
    };
    const shell_d = ContractedShell{
        .l = 2,
        .center = .{ .x = 0.5, .y = 0.3, .z = 0.8 },
        .primitives = &[_]basis_mod.PrimitiveGaussian{
            .{ .alpha = 2.0, .coeff = 1.0 },
        },
    };

    const ns = basis_mod.numCartesian(0);
    const np_val = basis_mod.numCartesian(1);
    const nd = basis_mod.numCartesian(2);
    const total = ns * nd * np_val * ns;

    var dx_arr: [18]f64 = undefined;
    var dy_arr: [18]f64 = undefined;
    var dz_arr: [18]f64 = undefined;
    _ = contractedShellQuartetEriDeriv(shell_s, shell_d, shell_p, shell_s, &dx_arr, &dy_arr, &dz_arr);

    // FD in x-direction for center A (shell_s center)
    const shell_s_p = ContractedShell{
        .l = 0,
        .center = .{ .x = shell_s.center.x + delta, .y = shell_s.center.y, .z = shell_s.center.z },
        .primitives = shell_s.primitives,
    };
    const shell_s_m = ContractedShell{
        .l = 0,
        .center = .{ .x = shell_s.center.x - delta, .y = shell_s.center.y, .z = shell_s.center.z },
        .primitives = shell_s.primitives,
    };

    var eri_p: [18]f64 = undefined;
    var eri_m: [18]f64 = undefined;
    _ = contractedShellQuartetERI(shell_s_p, shell_d, shell_p, shell_s, &eri_p);
    _ = contractedShellQuartetERI(shell_s_m, shell_d, shell_p, shell_s, &eri_m);

    for (0..total) |i| {
        const fd = (eri_p[i] - eri_m[i]) / (2.0 * delta);
        try testing.expectApproxEqAbs(fd, dx_arr[i], tol);
    }
}
