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
const PrimitiveGaussian = basis_mod.PrimitiveGaussian;

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

/// Maximum dimension in the 2D recurrence for the derivative ij direction.
const MAX_DIM_IJ_DERIV: usize = MAX_IJ + 2;

/// Maximum dimension in the 2D recurrence for kl direction.
const MAX_DIM_KL: usize = MAX_KL + 1;

/// Maximum Cartesian components per shell (f=10, g=15).
const MAX_CART: usize = basis_mod.MAX_CART;

/// Maximum primitives per shell.
const MAX_PRIM: usize = 16;

/// Maximum entries in norm table (prim * cart).
const MAX_NORM_TABLE: usize = MAX_PRIM * MAX_CART;

/// Size of the 2D g-table per axis, including the derivative case.
const MAX_G_SIZE: usize = MAX_NROOTS * MAX_DIM_IJ_DERIV * MAX_DIM_KL;

/// Primitive screening threshold.
const PRIM_SCREEN_THRESHOLD: f64 = 1e-15;

/// Maximum number of primitive pairs per bra or ket side.
const MAX_PRIM_PAIRS: usize = MAX_PRIM * MAX_PRIM;

/// Maximum ket HR table size.
const MAX_KET_HR: usize = MAX_DIM_IJ_DERIV * MAX_DIM_KL * (MAX_L + 1);

/// Maximum 4D HR table size, including the derivative case.
const MAX_HR4D: usize = (MAX_L + 2) * (MAX_L + 1) * (MAX_L + 1) * (MAX_L + 1);

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
    alpha_a: f64, // alpha for the differentiated center
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

const QuartetSetup = struct {
    na: usize,
    nb: usize,
    nc: usize,
    nd: usize,
    total_out: usize,
    cart_a: [MAX_CART]AngularMomentum,
    cart_b: [MAX_CART]AngularMomentum,
    cart_c: [MAX_CART]AngularMomentum,
    cart_d: [MAX_CART]AngularMomentum,
    dim_ij: usize,
    dim_kl: usize,
    nroots: usize,
    ab: [3]f64,
    cd: [3]f64,
    r2_ab: f64,
    r2_cd: f64,
    a_hr_count: usize,
    lb_1: usize,
    lc_1: usize,
    ld_1: usize,
    ket_stride_c: usize,
    ket_stride_i: usize,
    ket_size: usize,
    hr4d_c_stride: usize,
    hr4d_b_stride: usize,
    hr4d_a_stride: usize,
    hr4d_size: usize,
};

const RecurrenceWorkspace = struct {
    gx: [MAX_G_SIZE]f64 = undefined,
    gy: [MAX_G_SIZE]f64 = undefined,
    gz: [MAX_G_SIZE]f64 = undefined,
    rys_roots: [MAX_NROOTS]f64 = undefined,
    rys_weights: [MAX_NROOTS]f64 = undefined,
    c00x: [MAX_NROOTS]f64 = undefined,
    c00y: [MAX_NROOTS]f64 = undefined,
    c00z: [MAX_NROOTS]f64 = undefined,
    c0px: [MAX_NROOTS]f64 = undefined,
    c0py: [MAX_NROOTS]f64 = undefined,
    c0pz: [MAX_NROOTS]f64 = undefined,
    b10_arr: [MAX_NROOTS]f64 = undefined,
    b01_arr: [MAX_NROOTS]f64 = undefined,
    b00_arr: [MAX_NROOTS]f64 = undefined,
    w_pref: [MAX_NROOTS]f64 = undefined,
};

const EnergyWorkspace = struct {
    recurrence: RecurrenceWorkspace = undefined,
    ket_hr_x: [MAX_KET_HR]f64 = undefined,
    ket_hr_y: [MAX_KET_HR]f64 = undefined,
    ket_hr_z: [MAX_KET_HR]f64 = undefined,
    hr4d_x: [MAX_HR4D]f64 = undefined,
    hr4d_y: [MAX_HR4D]f64 = undefined,
    hr4d_z: [MAX_HR4D]f64 = undefined,
    prim_eri: [MAX_CART * MAX_CART * MAX_CART * MAX_CART]f64 = undefined,
};

const DerivativeWorkspace = struct {
    recurrence: RecurrenceWorkspace = undefined,
    ket_hr_x: [MAX_KET_HR]f64 = undefined,
    ket_hr_y: [MAX_KET_HR]f64 = undefined,
    ket_hr_z: [MAX_KET_HR]f64 = undefined,
    hr4d_x: [MAX_HR4D]f64 = undefined,
    hr4d_y: [MAX_HR4D]f64 = undefined,
    hr4d_z: [MAX_HR4D]f64 = undefined,
    prim_deriv_x: [MAX_CART * MAX_CART * MAX_CART * MAX_CART]f64 = undefined,
    prim_deriv_y: [MAX_CART * MAX_CART * MAX_CART * MAX_CART]f64 = undefined,
    prim_deriv_z: [MAX_CART * MAX_CART * MAX_CART * MAX_CART]f64 = undefined,
};

const QuartetNormTables = struct {
    norm_a: []const f64,
    norm_b: []const f64,
    norm_c: []const f64,
    norm_d: []const f64,
};

const PreparedQuartetPairs = struct {
    n_bra: usize,
    n_ket: usize,
};

fn fill_quartet_norm_tables(
    setup: QuartetSetup,
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    norm_a: []f64,
    norm_b: []f64,
    norm_c: []f64,
    norm_d: []f64,
    max_norm_a: []f64,
    max_norm_b: []f64,
    max_norm_c: []f64,
    max_norm_d: []f64,
) void {
    fill_normalization_table_and_max(shell_a, setup.na, &setup.cart_a, norm_a, max_norm_a);
    fill_normalization_table_and_max(shell_b, setup.nb, &setup.cart_b, norm_b, max_norm_b);
    fill_normalization_table_and_max(shell_c, setup.nc, &setup.cart_c, norm_c, max_norm_c);
    fill_normalization_table_and_max(shell_d, setup.nd, &setup.cart_d, norm_d, max_norm_d);
}

fn prepare_quartet_pairs(
    setup: QuartetSetup,
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    norm_a: []f64,
    norm_b: []f64,
    norm_c: []f64,
    norm_d: []f64,
    max_norm_a: []f64,
    max_norm_b: []f64,
    max_norm_c: []f64,
    max_norm_d: []f64,
    bra_pairs: []BraPair,
    ket_pairs: []KetPair,
) PreparedQuartetPairs {
    fill_quartet_norm_tables(
        setup,
        shell_a,
        shell_b,
        shell_c,
        shell_d,
        norm_a,
        norm_b,
        norm_c,
        norm_d,
        max_norm_a,
        max_norm_b,
        max_norm_c,
        max_norm_d,
    );
    return .{
        .n_bra = prepare_bra_pairs(
            shell_a,
            shell_b,
            max_norm_a[0..shell_a.primitives.len],
            max_norm_b[0..shell_b.primitives.len],
            setup.r2_ab,
            bra_pairs,
        ),
        .n_ket = prepare_ket_pairs(
            shell_c,
            shell_d,
            max_norm_c[0..shell_c.primitives.len],
            max_norm_d[0..shell_d.primitives.len],
            setup.r2_cd,
            ket_pairs,
        ),
    };
}

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
fn build_g2d(
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
                    const g_km2_0 = g[base + (k - 2) * stride_k + 0];
                    val += @as(f64, @floatFromInt(k - 1)) * b01[r] * g_km2_0;
                }
                g[base + k * stride_k + 0] = val;

                // g(i, k) for i = 1..dim_ij-1
                {
                    var i: usize = 1;
                    while (i < dim_ij) : (i += 1) {
                        const g_km1_im1 = g[base + (k - 1) * stride_k + i - 1];
                        val = c0p[r] * g[base + (k - 1) * stride_k + i] +
                            @as(f64, @floatFromInt(i)) * b00[r] * g_km1_im1;
                        if (k >= 2) {
                            const g_km2_i = g[base + (k - 2) * stride_k + i];
                            val += @as(f64, @floatFromInt(k - 1)) * b01[r] * g_km2_i;
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
fn build_g2d_weighted(
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
                    const g_km2_0 = g[base + (k - 2) * stride_k + 0];
                    val += @as(f64, @floatFromInt(k - 1)) * b01[r] * g_km2_0;
                }
                g[base + k * stride_k + 0] = val;

                {
                    var i: usize = 1;
                    while (i < dim_ij) : (i += 1) {
                        const g_km1_im1 = g[base + (k - 1) * stride_k + i - 1];
                        val = c0p[r] * g[base + (k - 1) * stride_k + i] +
                            @as(f64, @floatFromInt(i)) * b00[r] * g_km1_im1;
                        if (k >= 2) {
                            const g_km2_i = g[base + (k - 2) * stride_k + i];
                            val += @as(f64, @floatFromInt(k - 1)) * b01[r] * g_km2_i;
                        }
                        g[base + k * stride_k + i] = val;
                    }
                }
            }
        }
    }
}

fn init_quartet_setup(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    output: []f64,
    a_order_extra: usize,
) QuartetSetup {
    const la: usize = @intCast(shell_a.l);
    const lb: usize = @intCast(shell_b.l);
    const lc: usize = @intCast(shell_c.l);
    const ld: usize = @intCast(shell_d.l);

    const na = basis_mod.num_cartesian(shell_a.l);
    const nb = basis_mod.num_cartesian(shell_b.l);
    const nc = basis_mod.num_cartesian(shell_c.l);
    const nd = basis_mod.num_cartesian(shell_d.l);
    const total_out = na * nb * nc * nd;
    std.debug.assert(output.len >= total_out);
    @memset(output[0..total_out], 0.0);

    const diff_ab = math.Vec3.sub(shell_a.center, shell_b.center);
    const diff_cd = math.Vec3.sub(shell_c.center, shell_d.center);
    const dim_ij = la + lb + 1 + a_order_extra;
    const dim_kl = lc + ld + 1;
    const nroots = (la + lb + lc + ld + a_order_extra) / 2 + 1;
    const lb_1 = lb + 1;
    const lc_1 = lc + 1;
    const ld_1 = ld + 1;
    const ket_stride_c = ld_1;
    const ket_stride_i = dim_kl * ld_1;
    const hr4d_c_stride = ld_1;
    const hr4d_b_stride = lc_1 * ld_1;
    const hr4d_a_stride = lb_1 * lc_1 * ld_1;
    return .{
        .na = na,
        .nb = nb,
        .nc = nc,
        .nd = nd,
        .total_out = total_out,
        .cart_a = basis_mod.cartesian_exponents(shell_a.l),
        .cart_b = basis_mod.cartesian_exponents(shell_b.l),
        .cart_c = basis_mod.cartesian_exponents(shell_c.l),
        .cart_d = basis_mod.cartesian_exponents(shell_d.l),
        .dim_ij = dim_ij,
        .dim_kl = dim_kl,
        .nroots = nroots,
        .ab = .{ diff_ab.x, diff_ab.y, diff_ab.z },
        .cd = .{ diff_cd.x, diff_cd.y, diff_cd.z },
        .r2_ab = math.Vec3.dot(diff_ab, diff_ab),
        .r2_cd = math.Vec3.dot(diff_cd, diff_cd),
        .a_hr_count = la + 1 + a_order_extra,
        .lb_1 = lb_1,
        .lc_1 = lc_1,
        .ld_1 = ld_1,
        .ket_stride_c = ket_stride_c,
        .ket_stride_i = ket_stride_i,
        .ket_size = dim_ij * ket_stride_i,
        .hr4d_c_stride = hr4d_c_stride,
        .hr4d_b_stride = hr4d_b_stride,
        .hr4d_a_stride = hr4d_a_stride,
        .hr4d_size = (la + 1 + a_order_extra) * hr4d_a_stride,
    };
}

fn fill_normalization_table(
    shell: ContractedShell,
    n_cart: usize,
    cart: *const [MAX_CART]AngularMomentum,
    norm: []f64,
) void {
    for (shell.primitives, 0..) |prim, ip| {
        for (0..n_cart) |ic| {
            const c = cart[ic];
            norm[ip * n_cart + ic] = basis_mod.normalization(prim.alpha, c.x, c.y, c.z);
        }
    }
}

fn fill_normalization_table_and_max(
    shell: ContractedShell,
    n_cart: usize,
    cart: *const [MAX_CART]AngularMomentum,
    norm: []f64,
    max_norm: []f64,
) void {
    for (shell.primitives, 0..) |prim, ip| {
        var max_value: f64 = 0.0;
        for (0..n_cart) |ic| {
            const c = cart[ic];
            const n_val = basis_mod.normalization(prim.alpha, c.x, c.y, c.z);
            norm[ip * n_cart + ic] = n_val;
            if (n_val > max_value) max_value = n_val;
        }
        max_norm[ip] = max_value;
    }
}

fn prepare_bra_pairs(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    max_norm_a: []const f64,
    max_norm_b: []const f64,
    r2_ab: f64,
    bra_pairs: []BraPair,
) usize {
    var n_bra: usize = 0;
    for (shell_a.primitives, 0..) |prim_a, ipa| {
        const alpha = prim_a.alpha;
        for (shell_b.primitives, 0..) |prim_b, ipb| {
            const beta = prim_b.alpha;
            const p_val = alpha + beta;
            const mu_ab = alpha * beta / p_val;
            const px = (alpha * shell_a.center.x + beta * shell_b.center.x) / p_val;
            const py = (alpha * shell_a.center.y + beta * shell_b.center.y) / p_val;
            const pz = (alpha * shell_a.center.z + beta * shell_b.center.z) / p_val;
            bra_pairs[n_bra] = .{
                .p = p_val,
                .inv_2p = 0.5 / p_val,
                .px = px,
                .py = py,
                .pz = pz,
                .pax = px - shell_a.center.x,
                .pay = py - shell_a.center.y,
                .paz = pz - shell_a.center.z,
                .coeff_ab = prim_a.coeff * prim_b.coeff,
                .exp_ab = @exp(-mu_ab * r2_ab),
                .max_norm_ab = max_norm_a[ipa] * max_norm_b[ipb],
                .alpha_a = alpha,
                .ipa = ipa,
                .ipb = ipb,
            };
            n_bra += 1;
        }
    }
    return n_bra;
}

fn prepare_ket_pairs(
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    max_norm_c: []const f64,
    max_norm_d: []const f64,
    r2_cd: f64,
    ket_pairs: []KetPair,
) usize {
    var n_ket: usize = 0;
    for (shell_c.primitives, 0..) |prim_c, ipc| {
        const gamma_val = prim_c.alpha;
        for (shell_d.primitives, 0..) |prim_d, ipd| {
            const delta_val = prim_d.alpha;
            const q_val = gamma_val + delta_val;
            const mu_cd = gamma_val * delta_val / q_val;
            const qx = (gamma_val * shell_c.center.x + delta_val * shell_d.center.x) / q_val;
            const qy = (gamma_val * shell_c.center.y + delta_val * shell_d.center.y) / q_val;
            const qz = (gamma_val * shell_c.center.z + delta_val * shell_d.center.z) / q_val;
            ket_pairs[n_ket] = .{
                .q = q_val,
                .inv_2q = 0.5 / q_val,
                .qx = qx,
                .qy = qy,
                .qz = qz,
                .qcx = qx - shell_c.center.x,
                .qcy = qy - shell_c.center.y,
                .qcz = qz - shell_c.center.z,
                .coeff_cd = prim_c.coeff * prim_d.coeff,
                .exp_cd = @exp(-mu_cd * r2_cd),
                .max_norm_cd = max_norm_c[ipc] * max_norm_d[ipd],
                .ipc = ipc,
                .ipd = ipd,
            };
            n_ket += 1;
        }
    }
    return n_ket;
}

fn fill_primitive_quartet_root_data(
    workspace: *RecurrenceWorkspace,
    nroots: usize,
    pq: f64,
    bra: BraPair,
    ket: KetPair,
    wx: f64,
    wy: f64,
    wz: f64,
    prefactor: f64,
) void {
    for (0..nroots) |r| {
        const t2 = workspace.rys_roots[r];
        workspace.b00_arr[r] = 0.5 / pq * t2;
        workspace.b10_arr[r] = bra.inv_2p * (1.0 - ket.q / pq * t2);
        workspace.b01_arr[r] = ket.inv_2q * (1.0 - bra.p / pq * t2);
        workspace.c00x[r] = bra.pax + (wx - bra.px) * t2;
        workspace.c00y[r] = bra.pay + (wy - bra.py) * t2;
        workspace.c00z[r] = bra.paz + (wz - bra.pz) * t2;
        workspace.c0px[r] = ket.qcx + (wx - ket.qx) * t2;
        workspace.c0py[r] = ket.qcy + (wy - ket.qy) * t2;
        workspace.c0pz[r] = ket.qcz + (wz - ket.qz) * t2;
        workspace.w_pref[r] = workspace.rys_weights[r] * prefactor;
    }
}

fn prepare_primitive_quartet(
    workspace: *RecurrenceWorkspace,
    setup: QuartetSetup,
    bra: BraPair,
    ket: KetPair,
) ?f64 {
    const exp_factor = bra.exp_ab * ket.exp_cd;
    const coeff_abcd = bra.coeff_ab * ket.coeff_cd;
    const prefactor = 2.0 * std.math.pow(f64, std.math.pi, 2.5) /
        (bra.p * ket.q * @sqrt(bra.p + ket.q)) * exp_factor;
    if (@abs(coeff_abcd) * bra.max_norm_ab * ket.max_norm_cd * prefactor <
        PRIM_SCREEN_THRESHOLD) return null;

    const pq = bra.p + ket.q;
    const wx = (bra.p * bra.px + ket.q * ket.qx) / pq;
    const wy = (bra.p * bra.py + ket.q * ket.qy) / pq;
    const wz = (bra.p * bra.pz + ket.q * ket.qz) / pq;
    const dpqx = bra.px - ket.qx;
    const dpqy = bra.py - ket.qy;
    const dpqz = bra.pz - ket.qz;
    const rho = bra.p * ket.q / pq;

    rys_roots_mod.rys_roots(
        setup.nroots,
        rho * (dpqx * dpqx + dpqy * dpqy + dpqz * dpqz),
        &workspace.rys_roots,
        &workspace.rys_weights,
    );
    fill_primitive_quartet_root_data(workspace, setup.nroots, pq, bra, ket, wx, wy, wz, prefactor);

    const g_axis_size = setup.nroots * setup.dim_ij * setup.dim_kl;
    build_g2d(
        setup.nroots,
        setup.dim_ij,
        setup.dim_kl,
        &workspace.c00x,
        &workspace.c0px,
        &workspace.b10_arr,
        &workspace.b01_arr,
        &workspace.b00_arr,
        workspace.gx[0..g_axis_size],
    );
    build_g2d(
        setup.nroots,
        setup.dim_ij,
        setup.dim_kl,
        &workspace.c00y,
        &workspace.c0py,
        &workspace.b10_arr,
        &workspace.b01_arr,
        &workspace.b00_arr,
        workspace.gy[0..g_axis_size],
    );
    build_g2d_weighted(
        setup.nroots,
        setup.dim_ij,
        setup.dim_kl,
        &workspace.c00z,
        &workspace.c0pz,
        &workspace.b10_arr,
        &workspace.b01_arr,
        &workspace.b00_arr,
        &workspace.w_pref,
        workspace.gz[0..g_axis_size],
    );
    return coeff_abcd;
}

fn build_hr_axis(
    setup: QuartetSetup,
    g_axis: []const f64,
    ket_hr: []f64,
    hr4d: []f64,
    g2d_base: usize,
    ab_val: f64,
    cd_val: f64,
) void {
    for (0..setup.dim_ij) |i| {
        for (0..setup.dim_kl) |c| {
            ket_hr[i * setup.ket_stride_i + c * setup.ket_stride_c] =
                g_axis[g2d_base + c * setup.dim_ij + i];
        }
    }
    var d: usize = 1;
    while (d < setup.ld_1) : (d += 1) {
        const c_max = setup.dim_kl - d;
        for (0..setup.dim_ij) |i| {
            const i_base = i * setup.ket_stride_i;
            for (0..c_max) |c| {
                const up = ket_hr[i_base + (c + 1) * setup.ket_stride_c + (d - 1)];
                const same = ket_hr[i_base + c * setup.ket_stride_c + (d - 1)];
                ket_hr[i_base + c * setup.ket_stride_c + d] = up + cd_val * same;
            }
        }
    }

    for (0..setup.lc_1) |c| {
        for (0..setup.ld_1) |d_idx| {
            var work: [MAX_DIM_IJ_DERIV]f64 = undefined;
            for (0..setup.dim_ij) |a| {
                work[a] = ket_hr[a * setup.ket_stride_i + c * setup.ket_stride_c + d_idx];
            }
            for (0..setup.a_hr_count) |a| {
                hr4d[a * setup.hr4d_a_stride + c * setup.hr4d_c_stride + d_idx] = work[a];
            }
            var b: usize = 1;
            while (b < setup.lb_1) : (b += 1) {
                const a_count = setup.dim_ij - b;
                for (0..a_count) |a| {
                    work[a] = work[a + 1] + ab_val * work[a];
                }
                for (0..setup.a_hr_count) |a| {
                    hr4d[
                        a * setup.hr4d_a_stride +
                            b * setup.hr4d_b_stride +
                            c * setup.hr4d_c_stride + d_idx
                    ] = work[a];
                }
            }
        }
    }
}

fn build_quartet_hr_tables(
    setup: QuartetSetup,
    workspace: *RecurrenceWorkspace,
    ket_hr_x: []f64,
    ket_hr_y: []f64,
    ket_hr_z: []f64,
    hr4d_x: []f64,
    hr4d_y: []f64,
    hr4d_z: []f64,
    root: usize,
) void {
    const g_axis_size = setup.nroots * setup.dim_ij * setup.dim_kl;
    const g2d_base = root * setup.dim_ij * setup.dim_kl;
    build_hr_axis(
        setup,
        workspace.gx[0..g_axis_size],
        ket_hr_x,
        hr4d_x,
        g2d_base,
        setup.ab[0],
        setup.cd[0],
    );
    build_hr_axis(
        setup,
        workspace.gy[0..g_axis_size],
        ket_hr_y,
        hr4d_y,
        g2d_base,
        setup.ab[1],
        setup.cd[1],
    );
    build_hr_axis(
        setup,
        workspace.gz[0..g_axis_size],
        ket_hr_z,
        hr4d_z,
        g2d_base,
        setup.ab[2],
        setup.cd[2],
    );
}

fn accumulate_energy_root_contribution(
    setup: QuartetSetup,
    prim_eri: []f64,
    hr4d_x: []const f64,
    hr4d_y: []const f64,
    hr4d_z: []const f64,
) void {
    for (0..setup.na) |ia| {
        const a_cart = setup.cart_a[ia];
        const ax: usize = @intCast(a_cart.x);
        const ay: usize = @intCast(a_cart.y);
        const az: usize = @intCast(a_cart.z);
        for (0..setup.nb) |ib| {
            const b_cart = setup.cart_b[ib];
            const bx: usize = @intCast(b_cart.x);
            const by: usize = @intCast(b_cart.y);
            const bz: usize = @intCast(b_cart.z);
            for (0..setup.nc) |ic| {
                const c_cart = setup.cart_c[ic];
                const cx: usize = @intCast(c_cart.x);
                const cy: usize = @intCast(c_cart.y);
                const cz: usize = @intCast(c_cart.z);
                for (0..setup.nd) |id| {
                    const d_cart = setup.cart_d[id];
                    const dx: usize = @intCast(d_cart.x);
                    const dy: usize = @intCast(d_cart.y);
                    const dz: usize = @intCast(d_cart.z);
                    const vx = hr4d_x[
                        ax * setup.hr4d_a_stride +
                            bx * setup.hr4d_b_stride +
                            cx * setup.hr4d_c_stride + dx
                    ];
                    const vy = hr4d_y[
                        ay * setup.hr4d_a_stride +
                            by * setup.hr4d_b_stride +
                            cy * setup.hr4d_c_stride + dy
                    ];
                    const vz = hr4d_z[
                        az * setup.hr4d_a_stride +
                            bz * setup.hr4d_b_stride +
                            cz * setup.hr4d_c_stride + dz
                    ];
                    prim_eri[
                        ia * setup.nb * setup.nc * setup.nd +
                            ib * setup.nc * setup.nd +
                            ic * setup.nd + id
                    ] += vx * vy * vz;
                }
            }
        }
    }
}

fn accumulate_derivative_root_contribution(
    setup: QuartetSetup,
    alpha_a: f64,
    prim_deriv_x: []f64,
    prim_deriv_y: []f64,
    prim_deriv_z: []f64,
    hr4d_x: []const f64,
    hr4d_y: []const f64,
    hr4d_z: []const f64,
) void {
    for (0..setup.na) |ia| {
        const a_cart = setup.cart_a[ia];
        const ax: usize = @intCast(a_cart.x);
        const ay: usize = @intCast(a_cart.y);
        const az: usize = @intCast(a_cart.z);
        for (0..setup.nb) |ib| {
            const b_cart = setup.cart_b[ib];
            const bx: usize = @intCast(b_cart.x);
            const by: usize = @intCast(b_cart.y);
            const bz: usize = @intCast(b_cart.z);
            for (0..setup.nc) |ic| {
                const c_cart = setup.cart_c[ic];
                const cx: usize = @intCast(c_cart.x);
                const cy: usize = @intCast(c_cart.y);
                const cz: usize = @intCast(c_cart.z);
                for (0..setup.nd) |id| {
                    const d_cart = setup.cart_d[id];
                    const dx: usize = @intCast(d_cart.x);
                    const dy: usize = @intCast(d_cart.y);
                    const dz: usize = @intCast(d_cart.z);
                    const idx = ia * setup.nb * setup.nc * setup.nd +
                        ib * setup.nc * setup.nd +
                        ic * setup.nd + id;
                    const gx_idx = ax * setup.hr4d_a_stride +
                        bx * setup.hr4d_b_stride +
                        cx * setup.hr4d_c_stride + dx;
                    const gy_idx = ay * setup.hr4d_a_stride +
                        by * setup.hr4d_b_stride +
                        cy * setup.hr4d_c_stride + dy;
                    const gz_idx = az * setup.hr4d_a_stride +
                        bz * setup.hr4d_b_stride +
                        cz * setup.hr4d_c_stride + dz;
                    const gx_val = hr4d_x[gx_idx];
                    const gy_val = hr4d_y[gy_idx];
                    const gz_val = hr4d_z[gz_idx];
                    prim_deriv_x[idx] += derivative_axis_contribution(
                        alpha_a,
                        ax,
                        bx,
                        cx,
                        dx,
                        gy_val,
                        gz_val,
                        hr4d_x,
                        setup,
                    );
                    prim_deriv_y[idx] += derivative_axis_contribution(
                        alpha_a,
                        ay,
                        by,
                        cy,
                        dy,
                        gx_val,
                        gz_val,
                        hr4d_y,
                        setup,
                    );
                    prim_deriv_z[idx] += derivative_axis_contribution(
                        alpha_a,
                        az,
                        bz,
                        cz,
                        dz,
                        gx_val,
                        gy_val,
                        hr4d_z,
                        setup,
                    );
                }
            }
        }
    }
}

fn derivative_axis_contribution(
    alpha_a: f64,
    a_exp: usize,
    b_exp: usize,
    c_exp: usize,
    d_exp: usize,
    other_axis_1: f64,
    other_axis_2: f64,
    hr4d: []const f64,
    setup: QuartetSetup,
) f64 {
    const plus_idx = (a_exp + 1) * setup.hr4d_a_stride +
        b_exp * setup.hr4d_b_stride +
        c_exp * setup.hr4d_c_stride + d_exp;
    var value = 2.0 * alpha_a * hr4d[plus_idx] * other_axis_1 * other_axis_2;
    if (a_exp > 0) {
        const minus_idx = (a_exp - 1) * setup.hr4d_a_stride +
            b_exp * setup.hr4d_b_stride +
            c_exp * setup.hr4d_c_stride + d_exp;
        value -= @as(f64, @floatFromInt(a_exp)) *
            hr4d[minus_idx] * other_axis_1 * other_axis_2;
    }
    return value;
}

fn contract_energy_primitive(
    setup: QuartetSetup,
    coeff_abcd: f64,
    bra: BraPair,
    ket: KetPair,
    norms: QuartetNormTables,
    prim_eri: []const f64,
    output: []f64,
) void {
    for (0..setup.na) |ia| {
        const na_val = norms.norm_a[bra.ipa * setup.na + ia];
        for (0..setup.nb) |ib| {
            const nb_val = norms.norm_b[bra.ipb * setup.nb + ib];
            for (0..setup.nc) |ic| {
                const nc_val = norms.norm_c[ket.ipc * setup.nc + ic];
                for (0..setup.nd) |id| {
                    const nd_val = norms.norm_d[ket.ipd * setup.nd + id];
                    const idx = ia * setup.nb * setup.nc * setup.nd +
                        ib * setup.nc * setup.nd +
                        ic * setup.nd + id;
                    output[idx] += coeff_abcd * na_val * nb_val * nc_val * nd_val * prim_eri[idx];
                }
            }
        }
    }
}

fn contract_derivative_primitive(
    setup: QuartetSetup,
    coeff_abcd: f64,
    bra: BraPair,
    ket: KetPair,
    norms: QuartetNormTables,
    prim_deriv_x: []const f64,
    prim_deriv_y: []const f64,
    prim_deriv_z: []const f64,
    deriv_x: []f64,
    deriv_y: []f64,
    deriv_z: []f64,
) void {
    for (0..setup.na) |ia| {
        const na_val = norms.norm_a[bra.ipa * setup.na + ia];
        for (0..setup.nb) |ib| {
            const nb_val = norms.norm_b[bra.ipb * setup.nb + ib];
            for (0..setup.nc) |ic| {
                const nc_val = norms.norm_c[ket.ipc * setup.nc + ic];
                for (0..setup.nd) |id| {
                    const nd_val = norms.norm_d[ket.ipd * setup.nd + id];
                    const idx = ia * setup.nb * setup.nc * setup.nd +
                        ib * setup.nc * setup.nd +
                        ic * setup.nd + id;
                    const norm = coeff_abcd * na_val * nb_val * nc_val * nd_val;
                    deriv_x[idx] += norm * prim_deriv_x[idx];
                    deriv_y[idx] += norm * prim_deriv_y[idx];
                    deriv_z[idx] += norm * prim_deriv_z[idx];
                }
            }
        }
    }
}

fn accumulate_energy_primitive_pair(
    workspace: *EnergyWorkspace,
    setup: QuartetSetup,
    bra: BraPair,
    ket: KetPair,
    norms: QuartetNormTables,
    output: []f64,
) void {
    const coeff_abcd =
        prepare_primitive_quartet(&workspace.recurrence, setup, bra, ket) orelse return;
    @memset(workspace.prim_eri[0..setup.total_out], 0.0);
    for (0..setup.nroots) |root| {
        build_quartet_hr_tables(
            setup,
            &workspace.recurrence,
            workspace.ket_hr_x[0..setup.ket_size],
            workspace.ket_hr_y[0..setup.ket_size],
            workspace.ket_hr_z[0..setup.ket_size],
            workspace.hr4d_x[0..setup.hr4d_size],
            workspace.hr4d_y[0..setup.hr4d_size],
            workspace.hr4d_z[0..setup.hr4d_size],
            root,
        );
        accumulate_energy_root_contribution(
            setup,
            workspace.prim_eri[0..setup.total_out],
            workspace.hr4d_x[0..setup.hr4d_size],
            workspace.hr4d_y[0..setup.hr4d_size],
            workspace.hr4d_z[0..setup.hr4d_size],
        );
    }
    contract_energy_primitive(
        setup,
        coeff_abcd,
        bra,
        ket,
        norms,
        workspace.prim_eri[0..setup.total_out],
        output,
    );
}

fn accumulate_derivative_primitive_pair(
    workspace: *DerivativeWorkspace,
    setup: QuartetSetup,
    bra: BraPair,
    ket: KetPair,
    norms: QuartetNormTables,
    deriv_x: []f64,
    deriv_y: []f64,
    deriv_z: []f64,
) void {
    const coeff_abcd =
        prepare_primitive_quartet(&workspace.recurrence, setup, bra, ket) orelse return;
    @memset(workspace.prim_deriv_x[0..setup.total_out], 0.0);
    @memset(workspace.prim_deriv_y[0..setup.total_out], 0.0);
    @memset(workspace.prim_deriv_z[0..setup.total_out], 0.0);
    for (0..setup.nroots) |root| {
        build_quartet_hr_tables(
            setup,
            &workspace.recurrence,
            workspace.ket_hr_x[0..setup.ket_size],
            workspace.ket_hr_y[0..setup.ket_size],
            workspace.ket_hr_z[0..setup.ket_size],
            workspace.hr4d_x[0..setup.hr4d_size],
            workspace.hr4d_y[0..setup.hr4d_size],
            workspace.hr4d_z[0..setup.hr4d_size],
            root,
        );
        accumulate_derivative_root_contribution(
            setup,
            bra.alpha_a,
            workspace.prim_deriv_x[0..setup.total_out],
            workspace.prim_deriv_y[0..setup.total_out],
            workspace.prim_deriv_z[0..setup.total_out],
            workspace.hr4d_x[0..setup.hr4d_size],
            workspace.hr4d_y[0..setup.hr4d_size],
            workspace.hr4d_z[0..setup.hr4d_size],
        );
    }
    contract_derivative_primitive(
        setup,
        coeff_abcd,
        bra,
        ket,
        norms,
        workspace.prim_deriv_x[0..setup.total_out],
        workspace.prim_deriv_y[0..setup.total_out],
        workspace.prim_deriv_z[0..setup.total_out],
        deriv_x,
        deriv_y,
        deriv_z,
    );
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
pub fn contracted_shell_quartet_eri(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    output: []f64,
) usize {
    const setup = init_quartet_setup(shell_a, shell_b, shell_c, shell_d, output, 0);
    var norm_a: [MAX_NORM_TABLE]f64 = undefined;
    var norm_b: [MAX_NORM_TABLE]f64 = undefined;
    var norm_c: [MAX_NORM_TABLE]f64 = undefined;
    var norm_d: [MAX_NORM_TABLE]f64 = undefined;
    var max_norm_a: [MAX_PRIM]f64 = undefined;
    var max_norm_b: [MAX_PRIM]f64 = undefined;
    var max_norm_c: [MAX_PRIM]f64 = undefined;
    var max_norm_d: [MAX_PRIM]f64 = undefined;
    var bra_pairs: [MAX_PRIM_PAIRS]BraPair = undefined;
    var ket_pairs: [MAX_PRIM_PAIRS]KetPair = undefined;
    var workspace: EnergyWorkspace = undefined;

    const pairs = prepare_quartet_pairs(
        setup,
        shell_a,
        shell_b,
        shell_c,
        shell_d,
        norm_a[0..],
        norm_b[0..],
        norm_c[0..],
        norm_d[0..],
        max_norm_a[0..],
        max_norm_b[0..],
        max_norm_c[0..],
        max_norm_d[0..],
        bra_pairs[0..],
        ket_pairs[0..],
    );
    const norms = QuartetNormTables{
        .norm_a = norm_a[0..],
        .norm_b = norm_b[0..],
        .norm_c = norm_c[0..],
        .norm_d = norm_d[0..],
    };
    for (bra_pairs[0..pairs.n_bra]) |bra| {
        for (ket_pairs[0..pairs.n_ket]) |ket| {
            accumulate_energy_primitive_pair(&workspace, setup, bra, ket, norms, output);
        }
    }
    return setup.total_out;
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
pub fn contracted_shell_quartet_eri_deriv(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    deriv_x: []f64,
    deriv_y: []f64,
    deriv_z: []f64,
) usize {
    const setup = init_quartet_setup(shell_a, shell_b, shell_c, shell_d, deriv_x, 1);
    std.debug.assert(deriv_y.len >= setup.total_out);
    std.debug.assert(deriv_z.len >= setup.total_out);
    @memset(deriv_y[0..setup.total_out], 0.0);
    @memset(deriv_z[0..setup.total_out], 0.0);

    var norm_a: [MAX_NORM_TABLE]f64 = undefined;
    var norm_b: [MAX_NORM_TABLE]f64 = undefined;
    var norm_c: [MAX_NORM_TABLE]f64 = undefined;
    var norm_d: [MAX_NORM_TABLE]f64 = undefined;
    var max_norm_a: [MAX_PRIM]f64 = undefined;
    var max_norm_b: [MAX_PRIM]f64 = undefined;
    var max_norm_c: [MAX_PRIM]f64 = undefined;
    var max_norm_d: [MAX_PRIM]f64 = undefined;
    var bra_pairs: [MAX_PRIM_PAIRS]BraPair = undefined;
    var ket_pairs: [MAX_PRIM_PAIRS]KetPair = undefined;
    var workspace: DerivativeWorkspace = undefined;

    const pairs = prepare_quartet_pairs(
        setup,
        shell_a,
        shell_b,
        shell_c,
        shell_d,
        norm_a[0..],
        norm_b[0..],
        norm_c[0..],
        norm_d[0..],
        max_norm_a[0..],
        max_norm_b[0..],
        max_norm_c[0..],
        max_norm_d[0..],
        bra_pairs[0..],
        ket_pairs[0..],
    );
    const norms = QuartetNormTables{
        .norm_a = norm_a[0..],
        .norm_b = norm_b[0..],
        .norm_c = norm_c[0..],
        .norm_d = norm_d[0..],
    };
    for (bra_pairs[0..pairs.n_bra]) |bra| {
        for (ket_pairs[0..pairs.n_ket]) |ket| {
            accumulate_derivative_primitive_pair(
                &workspace,
                setup,
                bra,
                ket,
                norms,
                deriv_x,
                deriv_y,
                deriv_z,
            );
        }
    }
    return setup.total_out;
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

    _ = contracted_shell_quartet_eri(shell_a, shell_b, shell_a, shell_b, &output_rys);
    _ = obara_saika.contracted_shell_quartet_eri(shell_a, shell_b, shell_a, shell_b, &output_os);

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

    const na = basis_mod.num_cartesian(0); // 1
    const nb = basis_mod.num_cartesian(1); // 3
    const total = na * nb * na * nb; // 9
    var output_rys: [9]f64 = undefined;
    var output_os: [9]f64 = undefined;

    _ = contracted_shell_quartet_eri(shell_s, shell_p, shell_s, shell_p, &output_rys);
    _ = obara_saika.contracted_shell_quartet_eri(shell_s, shell_p, shell_s, shell_p, &output_os);

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

    const np = basis_mod.num_cartesian(1); // 3
    const total = np * np * np * np; // 81
    var output_rys: [81]f64 = undefined;
    var output_os: [81]f64 = undefined;

    _ = contracted_shell_quartet_eri(shell_p1, shell_p2, shell_p1, shell_p2, &output_rys);
    _ = obara_saika.contracted_shell_quartet_eri(
        shell_p1,
        shell_p2,
        shell_p1,
        shell_p2,
        &output_os,
    );

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

    const nd = basis_mod.num_cartesian(2); // 6
    const total = nd * nd * nd * nd; // 1296
    var output_rys: [1296]f64 = undefined;
    var output_os: [1296]f64 = undefined;

    _ = contracted_shell_quartet_eri(shell_d1, shell_d2, shell_d1, shell_d2, &output_rys);
    _ = obara_saika.contracted_shell_quartet_eri(
        shell_d1,
        shell_d2,
        shell_d1,
        shell_d2,
        &output_os,
    );

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

    const ns = basis_mod.num_cartesian(0); // 1
    const np = basis_mod.num_cartesian(1); // 3
    const nd = basis_mod.num_cartesian(2); // 6
    const total = ns * nd * np * ns; // 1*6*3*1 = 18
    var output_rys: [18]f64 = undefined;
    var output_os: [18]f64 = undefined;

    _ = contracted_shell_quartet_eri(shell_s, shell_d, shell_p, shell_s, &output_rys);
    _ = obara_saika.contracted_shell_quartet_eri(shell_s, shell_d, shell_p, shell_s, &output_os);

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
    _ = contracted_shell_quartet_eri_deriv(
        shell_a,
        shell_b,
        shell_a,
        shell_b,
        &dx_buf,
        &dy_buf,
        &dz_buf,
    );

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
        _ = contracted_shell_quartet_eri(shell_a_p, shell_b, shell_a, shell_b, &eri_p);
        _ = contracted_shell_quartet_eri(shell_a_m, shell_b, shell_a, shell_b, &eri_m);
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

    const ns = basis_mod.num_cartesian(0);
    const np_val = basis_mod.num_cartesian(1);
    const total = ns * np_val * ns * np_val;

    var dx_arr: [9]f64 = undefined;
    var dy_arr: [9]f64 = undefined;
    var dz_arr: [9]f64 = undefined;
    _ = contracted_shell_quartet_eri_deriv(
        shell_s,
        shell_p,
        shell_s,
        shell_p,
        &dx_arr,
        &dy_arr,
        &dz_arr,
    );

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
    _ = contracted_shell_quartet_eri(shell_s_p, shell_p, shell_s, shell_p, &eri_p);
    _ = contracted_shell_quartet_eri(shell_s_m, shell_p, shell_s, shell_p, &eri_m);

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

    const np_v = basis_mod.num_cartesian(1);
    const total = np_v * np_v * np_v * np_v;

    var dx_arr: [81]f64 = undefined;
    var dy_arr: [81]f64 = undefined;
    var dz_arr: [81]f64 = undefined;
    _ = contracted_shell_quartet_eri_deriv(
        shell_p1,
        shell_p2,
        shell_p1,
        shell_p2,
        &dx_arr,
        &dy_arr,
        &dz_arr,
    );

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
    _ = contracted_shell_quartet_eri(shell_p1_p, shell_p2, shell_p1, shell_p2, &eri_p);
    _ = contracted_shell_quartet_eri(shell_p1_m, shell_p2, shell_p1, shell_p2, &eri_m);

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

    const ns = basis_mod.num_cartesian(0);
    const np_val = basis_mod.num_cartesian(1);
    const nd = basis_mod.num_cartesian(2);
    const total = ns * nd * np_val * ns;

    var dx_arr: [18]f64 = undefined;
    var dy_arr: [18]f64 = undefined;
    var dz_arr: [18]f64 = undefined;
    _ = contracted_shell_quartet_eri_deriv(
        shell_s,
        shell_d,
        shell_p,
        shell_s,
        &dx_arr,
        &dy_arr,
        &dz_arr,
    );

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
    _ = contracted_shell_quartet_eri(shell_s_p, shell_d, shell_p, shell_s, &eri_p);
    _ = contracted_shell_quartet_eri(shell_s_m, shell_d, shell_p, shell_s, &eri_m);

    for (0..total) |i| {
        const fd = (eri_p[i] - eri_m[i]) / (2.0 * delta);
        try testing.expectApproxEqAbs(fd, dx_arr[i], tol);
    }
}
