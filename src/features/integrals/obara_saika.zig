//! Obara-Saika recurrence relations for Gaussian-type orbital integrals.
//!
//! Implements overlap (S), kinetic energy (T), nuclear attraction (V),
//! and electron repulsion integrals (ERI) for arbitrary angular momentum
//! using the Obara-Saika recurrence scheme.
//!
//! References:
//!   - Obara & Saika, J. Chem. Phys. 84, 3963 (1986)
//!   - Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure Theory", Ch. 9

const std = @import("std");
const math = @import("../math/math.zig");
const basis_mod = @import("../basis/basis.zig");
const boys_mod = @import("boys.zig");
const rys_eri = @import("rys_eri.zig");

const ContractedShell = basis_mod.ContractedShell;
const AngularMomentum = basis_mod.AngularMomentum;
const PrimitiveGaussian = basis_mod.PrimitiveGaussian;

// ============================================================================
// 1D Overlap Recurrence
// ============================================================================

/// Maximum angular momentum supported (l_a + l_b per axis).
/// For d-type (l=2) on both sides, max per axis = 4. We support up to l=4 (g).
const MAX_AM: usize = 10;

/// Compute 1D overlap integrals S_ij^(1d) for axis direction using OS recurrence.
///
/// The 1D primitive overlap between x^a exp(-α(x-Ax)²) and x^b exp(-β(x-Bx)²) is:
///   S(a,b) = (Px-Ax) S(a-1,b) + 1/(2p) [a S(a-2,b) + b S(a-1,b-1)]
///
/// where p = α + β, Px = (α·Ax + β·Bx) / p.
///
/// Parameters:
///   la_max: maximum angular momentum on center A (for this axis)
///   lb_max: maximum angular momentum on center B (for this axis)
///   pa: Px - Ax (component of Gaussian product center minus center A)
///   pb: Px - Bx (component of Gaussian product center minus center B)
///   inv_2p: 1 / (2*(α+β))
///   s00: the (0,0) overlap = sqrt(π/p) × exp(-μ × (Ax-Bx)²)
///         where μ = α×β/p. NOTE: For 3D factorized case, we pass in 1D factor.
///
/// Returns: (la_max+1) × (lb_max+1) array of 1D overlaps indexed as [a][b].
fn overlap1_d(
    la_max: usize,
    lb_max: usize,
    pa: f64,
    pb: f64,
    inv_2p: f64,
    s00: f64,
) [MAX_AM + 1][MAX_AM + 1]f64 {
    var s: [MAX_AM + 1][MAX_AM + 1]f64 = undefined;
    // Initialize all to zero
    for (0..MAX_AM + 1) |i| {
        for (0..MAX_AM + 1) |j| {
            s[i][j] = 0.0;
        }
    }

    s[0][0] = s00;

    // Build s[a][0] by incrementing a (vertical recurrence)
    for (1..la_max + 1) |a| {
        s[a][0] = pa * s[a - 1][0];
        if (a >= 2) {
            s[a][0] += @as(f64, @floatFromInt(a - 1)) * inv_2p * s[a - 2][0];
        }
    }

    // Build s[a][b] by incrementing b (horizontal recurrence)
    for (1..lb_max + 1) |b| {
        for (0..la_max + 1) |a| {
            s[a][b] = pb * s[a][b - 1];
            if (b >= 2) {
                s[a][b] += @as(f64, @floatFromInt(b - 1)) * inv_2p * s[a][b - 2];
            }
            if (a >= 1) {
                s[a][b] += @as(f64, @floatFromInt(a)) * inv_2p * s[a - 1][b - 1];
            }
        }
    }

    return s;
}

// ============================================================================
// 3D Primitive Overlap
// ============================================================================

/// Compute the overlap integral between two primitive Cartesian Gaussians.
///
/// g_a(r) = (x-Ax)^ax (y-Ay)^ay (z-Az)^az exp(-α|r-A|²)
/// g_b(r) = (x-Bx)^bx (y-By)^by (z-Bz)^bz exp(-β|r-B|²)
///
/// S = S_x(ax,bx) × S_y(ay,by) × S_z(az,bz) × exp(-μ|A-B|²)
///     × (π/p)^(3/2) absorbed via the factored 1D S00 values.
pub fn primitive_overlap(
    alpha: f64,
    center_a: math.Vec3,
    a: AngularMomentum,
    beta: f64,
    center_b: math.Vec3,
    b: AngularMomentum,
) f64 {
    const p = alpha + beta;
    const mu = alpha * beta / p;
    const inv_2p = 0.5 / p;

    // Gaussian product center P = (α·A + β·B) / p
    const px = (alpha * center_a.x + beta * center_b.x) / p;
    const py = (alpha * center_a.y + beta * center_b.y) / p;
    const pz = (alpha * center_a.z + beta * center_b.z) / p;

    const dx = center_a.x - center_b.x;
    const dy = center_a.y - center_b.y;
    const dz = center_a.z - center_b.z;
    const r2 = dx * dx + dy * dy + dz * dz;

    // Exponential prefactor
    const exp_factor = @exp(-mu * r2);

    // 1D base overlaps: sqrt(π/p) for each dimension
    const sqrt_pi_p = @sqrt(std.math.pi / p);

    // PA and PB components
    const pa_x = px - center_a.x;
    const pa_y = py - center_a.y;
    const pa_z = pz - center_a.z;
    const pb_x = px - center_b.x;
    const pb_y = py - center_b.y;
    const pb_z = pz - center_b.z;

    // 1D overlap recurrences (each starts with S_00 = sqrt(π/p))
    // The exponential factor is applied once at the end.
    const sx = overlap1_d(a.x, b.x, pa_x, pb_x, inv_2p, sqrt_pi_p);
    const sy = overlap1_d(a.y, b.y, pa_y, pb_y, inv_2p, sqrt_pi_p);
    const sz = overlap1_d(a.z, b.z, pa_z, pb_z, inv_2p, sqrt_pi_p);

    return exp_factor * sx[a.x][b.x] * sy[a.y][b.y] * sz[a.z][b.z];
}

/// Compute the overlap integral between two contracted shells for specific
/// Cartesian components (a_cart, b_cart).
///
/// S = Σ_i Σ_j c_i c_j N_i N_j × primitive_overlap(...)
pub fn contracted_overlap(
    shell_a: ContractedShell,
    a_cart: AngularMomentum,
    shell_b: ContractedShell,
    b_cart: AngularMomentum,
) f64 {
    var result: f64 = 0.0;

    for (shell_a.primitives) |prim_a| {
        const na = basis_mod.normalization(prim_a.alpha, a_cart.x, a_cart.y, a_cart.z);
        for (shell_b.primitives) |prim_b| {
            const nb = basis_mod.normalization(prim_b.alpha, b_cart.x, b_cart.y, b_cart.z);
            const prim_s = primitive_overlap(
                prim_a.alpha,
                shell_a.center,
                a_cart,
                prim_b.alpha,
                shell_b.center,
                b_cart,
            );
            result += prim_a.coeff * prim_b.coeff * na * nb * prim_s;
        }
    }

    return result;
}

// ============================================================================
// 1D Kinetic Energy Recurrence
// ============================================================================

/// Compute 1D kinetic energy integrals using Obara-Saika recurrence.
///
/// The kinetic energy integral can be decomposed as:
///   T = Tx × Sy × Sz + Sx × Ty × Sz + Sx × Sy × Tz
///
/// where Tx(a,b) is the 1D kinetic energy integral:
///   Tx(a,b) = β(2b+1) Sx(a,b) - 2β² Sx(a,b+1) ... (not direct OS form)
///
/// Actually it's simpler to compute T from the overlap relations:
///   T(a,b) = -½ [a(a-1) S(a-2,b) - 2α(2a+1) S(a,b) + 4α² S(a+2,b)]  (differentiation)
///
/// But the cleanest approach is:
///   T_ij = ½ β (2bx+1) S(ax,bx)·Sy·Sz
///        - 2β² S(ax,bx+2)·Sy·Sz + ... (for each dimension)
///
/// We use the simpler formula based on differentiation of overlap:
///   <a|∇²|b> = <a|∂²/∂x²|b> + <a|∂²/∂y²|b> + <a|∂²/∂z²|b>
///
/// For 1D:
///   <a|∂²/∂x²|b> = β(2b+1) S(a,b) - 2β² S(a,b+2) - ½b(b-1) S(a,b-2)
///
/// Then T = -½ (<a|∇²|b>) using -½∇² convention.
/// So T_x = -½ [β(2b+1) S(a,b) - 2β² S(a,b+2) - ½b(b-1) S(a,b-2)]
///
/// Wait, let me use the standard Obara-Saika formulation directly:
///   T_x(a,b) = ½ × ξ × [4αβ × S(a,b) - 2α × b × S(a,b-1) - 2β × a × S(a-1,b)]
///     ... no, this is incorrect.
///
/// The correct formulation from Helgaker eq. 9.3.41 is:
///   T(a,b) = ξ[(a|b)_00 + ...] where (a|b) is the overlap.
///
/// Simplest correct formula for primitive kinetic energy (per dimension):
///   K_1d(a,b) = β(2b+1)/(2p) × δ_1d(a,b)
///             - 2β²/(2p) × δ_1d(a,b+2)
///             - b(b-1)/(2×2p) × δ_1d(a,b-2)
///     ... no.
///
/// Let me use the well-known result for Cartesian Gaussians:
/// For the kinetic energy integral T_{ab} = <a|-½∇²|b>:
///
///   T = Tx_ab × Sy_ab × Sz_ab + Sx_ab × Ty_ab × Sz_ab + Sx_ab × Sy_ab × Tz_ab
///
/// where Sx_ab, Sy_ab, Sz_ab are 1D overlaps and:
///   Tx_ab = ½ β (2 bx + 1) Sx(ax, bx) - 2β² Sx(ax, bx+2) - ½ bx(bx-1) Sx(ax, bx-2)
///         ... all multiplied by exp(-μ r²)
///
/// Actually the standard formula is T(a,b) = <a|-½∇²|b> with:
///   ∂²/∂x² [x^b exp(-β x²)] = [b(b-1)x^{b-2} - 2β(2b+1)x^b + 4β² x^{b+2}] exp(-β x²)
///
/// So <a|∂²/∂x²|b> = b(b-1)·S(a,b-2) - 2β(2b+1)·S(a,b) + 4β²·S(a,b+2)
///
/// And T_x = -½ × [b(b-1)·S(a,b-2) - 2β(2b+1)·S(a,b) + 4β²·S(a,b+2)]
///         = β(2b+1)·S(a,b)/1 - 2β²·S(a,b+2) - ½b(b-1)·S(a,b-2)
///         ... (signs rearranged with the -½)
///
/// Actually just: T_x(ax,bx) = -½ × second_deriv_x
/// T_x = β(2bx+1)S(ax,bx) - 2β² S(ax,bx+2) - ½bx(bx-1)S(ax,bx-2)
///
/// This uses the 1D overlap array which we already have from overlap1_d.
/// Compute 1D kinetic energy contribution for one dimension.
/// T_x(a,b) = β(2b+1)·S(a,b) - 2β²·S(a,b+2) - ½·b(b-1)·S(a,b-2)
/// where S is the 1D overlap array and β is the exponent of center B.
fn kinetic1_d(
    a: usize,
    b: usize,
    beta: f64,
    s: [MAX_AM + 1][MAX_AM + 1]f64,
) f64 {
    const bf = @as(f64, @floatFromInt(b));
    var t: f64 = beta * (2.0 * bf + 1.0) * s[a][b];
    t -= 2.0 * beta * beta * s[a][b + 2];
    if (b >= 2) {
        t -= 0.5 * bf * (bf - 1.0) * s[a][b - 2];
    }
    return t;
}

/// Compute the kinetic energy integral between two primitive Cartesian Gaussians.
/// T = -½ <a|∇²|b> = T_x·S_y·S_z + S_x·T_y·S_z + S_x·S_y·T_z
pub fn primitive_kinetic(
    alpha: f64,
    center_a: math.Vec3,
    a: AngularMomentum,
    beta: f64,
    center_b: math.Vec3,
    b: AngularMomentum,
) f64 {
    const p = alpha + beta;
    const mu = alpha * beta / p;
    const inv_2p = 0.5 / p;

    const px = (alpha * center_a.x + beta * center_b.x) / p;
    const py = (alpha * center_a.y + beta * center_b.y) / p;
    const pz = (alpha * center_a.z + beta * center_b.z) / p;

    const dx = center_a.x - center_b.x;
    const dy = center_a.y - center_b.y;
    const dz = center_a.z - center_b.z;
    const r2 = dx * dx + dy * dy + dz * dz;

    const exp_factor = @exp(-mu * r2);
    const sqrt_pi_p = @sqrt(std.math.pi / p);

    const pa_x = px - center_a.x;
    const pa_y = py - center_a.y;
    const pa_z = pz - center_a.z;
    const pb_x = px - center_b.x;
    const pb_y = py - center_b.y;
    const pb_z = pz - center_b.z;

    // Need overlaps up to la+2 and lb+2 for kinetic energy
    const sx = overlap1_d(a.x + 2, b.x + 2, pa_x, pb_x, inv_2p, sqrt_pi_p);
    const sy = overlap1_d(a.y + 2, b.y + 2, pa_y, pb_y, inv_2p, sqrt_pi_p);
    const sz = overlap1_d(a.z + 2, b.z + 2, pa_z, pb_z, inv_2p, sqrt_pi_p);

    // T = T_x · S_y · S_z + S_x · T_y · S_z + S_x · S_y · T_z
    const tx = kinetic1_d(a.x, b.x, beta, sx);
    const ty = kinetic1_d(a.y, b.y, beta, sy);
    const tz = kinetic1_d(a.z, b.z, beta, sz);

    const sov_x = sx[a.x][b.x];
    const sov_y = sy[a.y][b.y];
    const sov_z = sz[a.z][b.z];

    return exp_factor * (tx * sov_y * sov_z + sov_x * ty * sov_z + sov_x * sov_y * tz);
}

/// Compute the kinetic energy integral between two contracted shells for
/// specific Cartesian components.
pub fn contracted_kinetic(
    shell_a: ContractedShell,
    a_cart: AngularMomentum,
    shell_b: ContractedShell,
    b_cart: AngularMomentum,
) f64 {
    var result: f64 = 0.0;

    for (shell_a.primitives) |prim_a| {
        const na = basis_mod.normalization(prim_a.alpha, a_cart.x, a_cart.y, a_cart.z);
        for (shell_b.primitives) |prim_b| {
            const nb = basis_mod.normalization(prim_b.alpha, b_cart.x, b_cart.y, b_cart.z);
            const prim_t = primitive_kinetic(
                prim_a.alpha,
                shell_a.center,
                a_cart,
                prim_b.alpha,
                shell_b.center,
                b_cart,
            );
            result += prim_a.coeff * prim_b.coeff * na * nb * prim_t;
        }
    }

    return result;
}

// ============================================================================
// Nuclear Attraction Integral (Obara-Saika with Boys function)
// ============================================================================

/// Compute the nuclear attraction integral between two primitive Cartesian Gaussians
/// for a single nucleus at position C with charge Z.
///
/// V = -Z × Σ over auxiliary integrals using Obara-Saika recurrence with Boys function.
///
/// The key recurrence (Helgaker eq. 9.9.14):
///   Θ(a+1_i, b; m) = (Pi - Ai) Θ(a, b; m) + (Wi - Pi) Θ(a, b; m+1)
///                   + a_i/(2p) [Θ(a-1_i, b; m) - ρ/p Θ(a-1_i, b; m+1)]
///                   + b_i/(2p) [Θ(a, b-1_i; m) - ρ/p Θ(a, b-1_i; m+1)]
///
/// where W = (p·P + ζ·C) / (p + ζ), ρ = p·ζ/(p+ζ) for nuclear integral ζ→∞ so
/// W → C is not right... Actually for nuclear attraction, the auxiliary is:
///   R^m(0,0) = (-2p)^m × F_m(p × |P-C|²) × K_AB
///
/// Following standard OS for nuclear attraction:
///   A(a+1_i, b; m) = PA_i × A(a,b;m) - PC_i × ρ/p × ...
///
/// The standard Obara-Saika recurrence for nuclear attraction is:
///   V(a+1_i, b; m) = PA_i V(a,b;m) + WP_i V(a,b;m+1)
///     + a_i/(2p) [V(a-1_i,b;m) + ρ/p V(a-1_i,b;m+1)]   ... no, sign wrong
///
/// Correct (Helgaker, 9.9.18-20):
///   Θ^(m)_{a+1_i,b} = XPA_i × Θ^(m)_{a,b} + XWP_i × Θ^(m+1)_{a,b}
///     + (a_i)/(2p) × [Θ^(m)_{a-1_i,b} - (p_c/p) × Θ^(m+1)_{a-1_i,b}]
///     + (b_i)/(2p) × [Θ^(m)_{a,b-1_i} - (p_c/p) × Θ^(m+1)_{a,b-1_i}]
///
/// where p_c = p·ζ/(p+ζ), but for nuclear attraction (point charge), ζ → ∞,
/// so p_c → p, and W → C, XWP = W - P = C - P.
///
/// Actually the standard result for nuclear attraction (point charge nucleus):
///   Θ^(m)_{a+1_i,b} = PA_i × Θ^(m)_{a,b} + CP_i × Θ^(m+1)_{a,b}
///     + a_i/(2p) × [Θ^(m)_{a-1_i,b} - Θ^(m+1)_{a-1_i,b}]
///     + b_i/(2p) × [Θ^(m)_{a,b-1_i} - Θ^(m+1)_{a,b-1_i}]
///
/// where CP_i = C_i - P_i (nucleus position minus product center).
///
/// Base case: Θ^(m)_{0,0} = (-2π/p) × exp(-μ|A-B|²) × F_m(p|P-C|²)
///
/// But we want V = -Z × Θ^(0)_{a,b}
/// Maximum total angular momentum for nuclear attraction auxiliary functions.
/// For la + lb up to 4 (e.g., d+d), we need m up to la+lb = 4.
const MAX_AM_AUX: usize = 12;

/// Compute nuclear attraction integral between two primitive Cartesian Gaussians
/// and a point charge nucleus.
///
/// Returns: -Z × 2π/p × exp(-μ|A-B|²) × Θ(a,b)
pub fn primitive_nuclear_attraction(
    alpha: f64,
    center_a: math.Vec3,
    a: AngularMomentum,
    beta: f64,
    center_b: math.Vec3,
    b: AngularMomentum,
    nuc_pos: math.Vec3,
    z_nuc: f64,
) f64 {
    const p = alpha + beta;
    const mu = alpha * beta / p;
    const inv_2p = 0.5 / p;

    // Gaussian product center
    const pc = math.Vec3{
        .x = (alpha * center_a.x + beta * center_b.x) / p,
        .y = (alpha * center_a.y + beta * center_b.y) / p,
        .z = (alpha * center_a.z + beta * center_b.z) / p,
    };

    const diff_ab = math.Vec3.sub(center_a, center_b);
    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);

    const diff_pc = math.Vec3.sub(pc, nuc_pos);
    const r2_pc = math.Vec3.dot(diff_pc, diff_pc);

    const exp_factor = @exp(-mu * r2_ab);
    const prefactor = -z_nuc * 2.0 * std.math.pi / p * exp_factor;

    // PA and CP components
    const pa_x = pc.x - center_a.x;
    const pa_y = pc.y - center_a.y;
    const pa_z = pc.z - center_a.z;

    const pb_x = pc.x - center_b.x;
    const pb_y = pc.y - center_b.y;
    const pb_z = pc.z - center_b.z;

    const cp_x = nuc_pos.x - pc.x;
    const cp_y = nuc_pos.y - pc.y;
    const cp_z = nuc_pos.z - pc.z;

    const total_am = a.x + a.y + a.z + b.x + b.y + b.z;

    // Compute Boys function values F_m(p × |P-C|²) for m = 0 to total_am
    var boys: [MAX_AM_AUX + 1]f64 = undefined;
    const arg = p * r2_pc;
    for (0..total_am + 1) |m| {
        boys[m] = boys_mod.boys_n(@as(u32, @intCast(m)), arg);
    }

    // Now use 3D Obara-Saika recurrence for the nuclear attraction auxiliary integral.
    // We need to build Θ^(m)[ax][bx][ay][by][az][bz] but this is too much memory.
    // Instead, we use a recursive approach or flat indexing.
    //
    // The factored approach: nuclear attraction does NOT factor into 1D components
    // (unlike overlap and kinetic), so we use a direct 3D recurrence.
    //
    // We'll implement using a compact array indexed by (ax, bx, ay, by, az, bz, m).
    // For practical efficiency, we use a flat buffer.

    // Use a simpler recursive implementation for now (sufficient for s, p, d).
    const aux_result = nuclear_aux3_d(
        a.x,
        a.y,
        a.z,
        b.x,
        b.y,
        b.z,
        0,
        pa_x,
        pa_y,
        pa_z,
        pb_x,
        pb_y,
        pb_z,
        cp_x,
        cp_y,
        cp_z,
        inv_2p,
        &boys,
    );

    return prefactor * aux_result;
}

fn first_non_zero_axis(comptime T: type, values: [3]T) ?usize {
    for (0..3) |axis| {
        if (values[axis] != 0) return axis;
    }
    return null;
}

fn nuclear_aux3_d_arr(
    a_arr: [3]u32,
    b_arr: [3]u32,
    m: u32,
    pa: [3]f64,
    pb: [3]f64,
    cp: [3]f64,
    inv_2p: f64,
    boys: []const f64,
) f64 {
    if (first_non_zero_axis(u32, b_arr)) |axis| {
        var b_dec = b_arr;
        b_dec[axis] -= 1;
        var a_inc = a_arr;
        a_inc[axis] += 1;
        const t1 = nuclear_aux3_d_arr(a_inc, b_dec, m, pa, pb, cp, inv_2p, boys);
        const t2 = nuclear_aux3_d_arr(a_arr, b_dec, m, pa, pb, cp, inv_2p, boys);
        return t1 + (pb[axis] - pa[axis]) * t2;
    }

    return nuclear_aux_vertical_arr(a_arr, m, pa, cp, inv_2p, boys);
}

fn nuclear_aux_vertical_arr(
    a_arr: [3]u32,
    m: u32,
    pa: [3]f64,
    cp: [3]f64,
    inv_2p: f64,
    boys: []const f64,
) f64 {
    if (first_non_zero_axis(u32, a_arr)) |axis| {
        var a_dec = a_arr;
        a_dec[axis] -= 1;
        const t_m0 = nuclear_aux_vertical_arr(a_dec, m, pa, cp, inv_2p, boys);
        const t_m1 = nuclear_aux_vertical_arr(a_dec, m + 1, pa, cp, inv_2p, boys);
        var result = pa[axis] * t_m0 + cp[axis] * t_m1;
        if (a_dec[axis] > 0) {
            var a_dec2 = a_dec;
            a_dec2[axis] -= 1;
            const ai = @as(f64, @floatFromInt(a_dec[axis]));
            result += ai * inv_2p * (nuclear_aux_vertical_arr(a_dec2, m, pa, cp, inv_2p, boys) -
                nuclear_aux_vertical_arr(a_dec2, m + 1, pa, cp, inv_2p, boys));
        }
        return result;
    }

    return boys[m];
}

/// Recursive 3D Obara-Saika auxiliary nuclear attraction integral.
/// Θ^(m)_{a,b} with a = (ax,ay,az), b = (bx,by,bz).
///
/// Uses horizontal transfer to reduce b → 0 first, then vertical recurrence on a.
fn nuclear_aux3_d(
    ax: u32,
    ay: u32,
    az: u32,
    bx: u32,
    by: u32,
    bz: u32,
    m: u32,
    pa_x: f64,
    pa_y: f64,
    pa_z: f64,
    pb_x: f64,
    pb_y: f64,
    pb_z: f64,
    cp_x: f64,
    cp_y: f64,
    cp_z: f64,
    inv_2p: f64,
    boys: []const f64,
) f64 {
    return nuclear_aux3_d_arr(
        .{ ax, ay, az },
        .{ bx, by, bz },
        m,
        .{ pa_x, pa_y, pa_z },
        .{ pb_x, pb_y, pb_z },
        .{ cp_x, cp_y, cp_z },
        inv_2p,
        boys,
    );
}

/// Vertical recurrence for nuclear attraction with b = (0,0,0).
/// Θ^(m)_{a,0}
fn nuclear_aux_vertical(
    ax: u32,
    ay: u32,
    az: u32,
    m: u32,
    pa_x: f64,
    pa_y: f64,
    pa_z: f64,
    cp_x: f64,
    cp_y: f64,
    cp_z: f64,
    inv_2p: f64,
    boys: []const f64,
) f64 {
    return nuclear_aux_vertical_arr(
        .{ ax, ay, az },
        m,
        .{ pa_x, pa_y, pa_z },
        .{ cp_x, cp_y, cp_z },
        inv_2p,
        boys,
    );
}

/// Compute nuclear attraction integral between two contracted shells for
/// specific Cartesian components and a single nucleus.
pub fn contracted_nuclear_attraction(
    shell_a: ContractedShell,
    a_cart: AngularMomentum,
    shell_b: ContractedShell,
    b_cart: AngularMomentum,
    nuc_pos: math.Vec3,
    z_nuc: f64,
) f64 {
    var result: f64 = 0.0;

    for (shell_a.primitives) |prim_a| {
        const na = basis_mod.normalization(prim_a.alpha, a_cart.x, a_cart.y, a_cart.z);
        for (shell_b.primitives) |prim_b| {
            const nb = basis_mod.normalization(prim_b.alpha, b_cart.x, b_cart.y, b_cart.z);
            const prim_v = primitive_nuclear_attraction(
                prim_a.alpha,
                shell_a.center,
                a_cart,
                prim_b.alpha,
                shell_b.center,
                b_cart,
                nuc_pos,
                z_nuc,
            );
            result += prim_a.coeff * prim_b.coeff * na * nb * prim_v;
        }
    }

    return result;
}

/// Compute total nuclear attraction integral for all nuclei.
pub fn contracted_total_nuclear_attraction(
    shell_a: ContractedShell,
    a_cart: AngularMomentum,
    shell_b: ContractedShell,
    b_cart: AngularMomentum,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
) f64 {
    var result: f64 = 0.0;
    for (nuc_positions, 0..) |pos, i| {
        result += contracted_nuclear_attraction(
            shell_a,
            a_cart,
            shell_b,
            b_cart,
            pos,
            nuc_charges[i],
        );
    }
    return result;
}

// ============================================================================
// Electron Repulsion Integral (Obara-Saika) — Table-based implementation
// ============================================================================

/// Maximum total angular momentum on one side of an ERI after horizontal transfer.
/// For f+f (l=3+3=6). Increase if g-type orbitals are needed.
const ERI_MAX_AM: usize = 7; // 0..6 inclusive

/// Maximum auxiliary index m = La + Lc, where La = la+lb, Lc = lc+ld.
/// For f+f|f+f: m_max = 6+6 = 12.
const ERI_MAX_M: usize = 13; // 0..12 inclusive

/// Number of Cartesian monomials with total angular momentum <= L.
/// This is (L+1)(L+2)(L+3)/6.
fn num_cartesian_up_to(comptime l: usize) usize {
    return (l + 1) * (l + 2) * (l + 3) / 6;
}

/// Maximum number of Cartesian indices for one side of the vertical table.
const ERI_MAX_CART: usize = num_cartesian_up_to(ERI_MAX_AM - 1); // 84 for L=6

/// Map (ax,ay,az) to a linear index. Requires ax+ay+az <= ERI_MAX_AM-1.
/// Uses a simple 3D layout: idx = az + (ERI_MAX_AM)*(ay + ERI_MAX_AM*ax).
fn eri_cart_index(ax: usize, ay: usize, az: usize) usize {
    return az + ERI_MAX_AM * (ay + ERI_MAX_AM * ax);
}

/// Total size of the 3D Cartesian index space.
const ERI_CART_STRIDE: usize = ERI_MAX_AM * ERI_MAX_AM * ERI_MAX_AM; // 343

/// Vertical recurrence table type.
/// theta[idx_a][idx_c][m] = [a,0|c,0]^(m)
/// where idx_a = eri_cart_index(ax,ay,az), idx_c = eri_cart_index(cx,cy,cz).
///
/// We use a flat array: theta[idx_a * ERI_CART_STRIDE * ERI_MAX_M + idx_c * ERI_MAX_M + m]
const THETA_SIZE: usize = ERI_CART_STRIDE * ERI_CART_STRIDE * ERI_MAX_M;

const ThetaView = struct {
    theta: []const f64,
    a_stride_x: usize,
    a_stride_y: usize,
    c_size: usize,
    c_stride_x: usize,
    c_stride_y: usize,
    m_stride: usize,
};

fn theta_lookup(view: ThetaView, a_arr: [3]u32, c_arr: [3]u32, m: usize) f64 {
    const a_idx = @as(usize, a_arr[0]) * view.a_stride_x +
        @as(usize, a_arr[1]) * view.a_stride_y +
        @as(usize, a_arr[2]);
    const c_idx = @as(usize, c_arr[0]) * view.c_stride_x +
        @as(usize, c_arr[1]) * view.c_stride_y +
        @as(usize, c_arr[2]);
    return view.theta[a_idx * view.c_size * view.m_stride + c_idx * view.m_stride + m];
}

fn eri_horizontal_arr(
    a_arr: [3]u32,
    b_arr: [3]u32,
    c_arr: [3]u32,
    d_arr: [3]u32,
    view: ThetaView,
    ab: [3]f64,
    cd_vec: [3]f64,
) f64 {
    if (first_non_zero_axis(u32, d_arr)) |axis| {
        var d_dec = d_arr;
        d_dec[axis] -= 1;
        var c_inc = c_arr;
        c_inc[axis] += 1;
        const t1 = eri_horizontal_arr(a_arr, b_arr, c_inc, d_dec, view, ab, cd_vec);
        const t2 = eri_horizontal_arr(a_arr, b_arr, c_arr, d_dec, view, ab, cd_vec);
        return t1 + cd_vec[axis] * t2;
    }
    if (first_non_zero_axis(u32, b_arr)) |axis| {
        var b_dec = b_arr;
        b_dec[axis] -= 1;
        var a_inc = a_arr;
        a_inc[axis] += 1;
        const t1 = eri_horizontal_arr(a_inc, b_dec, c_arr, d_arr, view, ab, cd_vec);
        const t2 = eri_horizontal_arr(a_arr, b_dec, c_arr, d_arr, view, ab, cd_vec);
        return t1 + ab[axis] * t2;
    }
    return theta_lookup(view, a_arr, c_arr, 0);
}

const MAX_STACK_THETA_TABLE: usize = 256 * 1024;

fn coord_index3(coords: [3]usize, stride_x: usize, stride_y: usize) usize {
    return coords[0] * stride_x + coords[1] * stride_y + coords[2];
}

const ThetaLayout = struct {
    La: usize,
    Lc: usize,
    m_max: usize,
    a_stride_x: usize,
    a_stride_y: usize,
    c_size: usize,
    c_stride_x: usize,
    c_stride_y: usize,
    m_stride: usize,
    theta_size: usize,

    fn init(La: usize, Lc: usize) ThetaLayout {
        const a_stride_y = La + 1;
        const a_stride_x = (La + 1) * a_stride_y;
        const c_stride_y = Lc + 1;
        const c_stride_x = (Lc + 1) * c_stride_y;
        const c_size = (Lc + 1) * c_stride_x;
        const m_stride = La + Lc + 1;
        return .{
            .La = La,
            .Lc = Lc,
            .m_max = La + Lc,
            .a_stride_x = a_stride_x,
            .a_stride_y = a_stride_y,
            .c_size = c_size,
            .c_stride_x = c_stride_x,
            .c_stride_y = c_stride_y,
            .m_stride = m_stride,
            .theta_size = ((La + 1) * a_stride_x) * c_size * m_stride,
        };
    }

    fn view(self: ThetaLayout, theta: []const f64) ThetaView {
        return .{
            .theta = theta,
            .a_stride_x = self.a_stride_x,
            .a_stride_y = self.a_stride_y,
            .c_size = self.c_size,
            .c_stride_x = self.c_stride_x,
            .c_stride_y = self.c_stride_y,
            .m_stride = self.m_stride,
        };
    }
};

const ThetaBuildInputs = struct {
    boys: []const f64,
    pa: [3]f64,
    qc: [3]f64,
    wp: [3]f64,
    wq: [3]f64,
    inv_2p: f64,
    inv_2q: f64,
    inv_2pq: f64,
    rho_over_p: f64,
    rho_over_q: f64,
};

const PrimitiveEriSetup = struct {
    prefactor: f64,
    layout: ThetaLayout,
    boys: [ERI_MAX_M]f64,
    pa: [3]f64,
    qc: [3]f64,
    wp: [3]f64,
    wq: [3]f64,
    ab: [3]f64,
    cd: [3]f64,
    inv_2p: f64,
    inv_2q: f64,
    inv_2pq: f64,
    rho_over_p: f64,
    rho_over_q: f64,
};

fn init_primitive_eri_setup(
    alpha: f64,
    center_a: math.Vec3,
    a: AngularMomentum,
    beta: f64,
    center_b: math.Vec3,
    b: AngularMomentum,
    gamma: f64,
    center_c: math.Vec3,
    c_am: AngularMomentum,
    delta: f64,
    center_d: math.Vec3,
    d_am: AngularMomentum,
) PrimitiveEriSetup {
    const p = alpha + beta;
    const q = gamma + delta;
    const rho = p * q / (p + q);
    const p_center = math.Vec3{
        .x = (alpha * center_a.x + beta * center_b.x) / p,
        .y = (alpha * center_a.y + beta * center_b.y) / p,
        .z = (alpha * center_a.z + beta * center_b.z) / p,
    };
    const q_center = math.Vec3{
        .x = (gamma * center_c.x + delta * center_d.x) / q,
        .y = (gamma * center_c.y + delta * center_d.y) / q,
        .z = (gamma * center_c.z + delta * center_d.z) / q,
    };
    const w_center = math.Vec3{
        .x = (p * p_center.x + q * q_center.x) / (p + q),
        .y = (p * p_center.y + q * q_center.y) / (p + q),
        .z = (p * p_center.z + q * q_center.z) / (p + q),
    };
    const mu_ab = alpha * beta / p;
    const mu_cd = gamma * delta / q;
    const diff_ab = math.Vec3.sub(center_a, center_b);
    const diff_cd = math.Vec3.sub(center_c, center_d);
    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);
    const r2_cd = math.Vec3.dot(diff_cd, diff_cd);
    const diff_pq = math.Vec3.sub(p_center, q_center);
    const arg = rho * math.Vec3.dot(diff_pq, diff_pq);
    const la = a.x + a.y + a.z;
    const lb = b.x + b.y + b.z;
    const lc = c_am.x + c_am.y + c_am.z;
    const ld = d_am.x + d_am.y + d_am.z;
    var boys: [ERI_MAX_M]f64 = undefined;
    boys_mod.boys_batch(@as(u32, @intCast(la + lb + lc + ld)), arg, &boys);
    return .{
        .prefactor = 2.0 * std.math.pow(f64, std.math.pi, 2.5) /
            (p * q * @sqrt(p + q)) * @exp(-mu_ab * r2_ab - mu_cd * r2_cd),
        .layout = ThetaLayout.init(la + lb, lc + ld),
        .boys = boys,
        .pa = .{ p_center.x - center_a.x, p_center.y - center_a.y, p_center.z - center_a.z },
        .qc = .{ q_center.x - center_c.x, q_center.y - center_c.y, q_center.z - center_c.z },
        .wp = .{ w_center.x - p_center.x, w_center.y - p_center.y, w_center.z - p_center.z },
        .wq = .{ w_center.x - q_center.x, w_center.y - q_center.y, w_center.z - q_center.z },
        .ab = .{ center_a.x - center_b.x, center_a.y - center_b.y, center_a.z - center_b.z },
        .cd = .{ center_c.x - center_d.x, center_c.y - center_d.y, center_c.z - center_d.z },
        .inv_2p = 0.5 / p,
        .inv_2q = 0.5 / q,
        .inv_2pq = 0.5 / (p + q),
        .rho_over_p = rho / p,
        .rho_over_q = rho / q,
    };
}

fn theta_build_inputs(setup: *const PrimitiveEriSetup) ThetaBuildInputs {
    return .{
        .boys = &setup.boys,
        .pa = setup.pa,
        .qc = setup.qc,
        .wp = setup.wp,
        .wq = setup.wq,
        .inv_2p = setup.inv_2p,
        .inv_2q = setup.inv_2q,
        .inv_2pq = setup.inv_2pq,
        .rho_over_p = setup.rho_over_p,
        .rho_over_q = setup.rho_over_q,
    };
}

fn init_theta_base(
    theta: []f64,
    layout: ThetaLayout,
    boys: []const f64,
    zero_used: bool,
) void {
    if (zero_used) @memset(theta[0..layout.theta_size], 0.0);
    for (0..layout.m_max + 1) |m| {
        theta[m] = boys[m];
    }
}

fn fill_theta_c_entry(
    theta: []f64,
    layout: ThetaLayout,
    inputs: ThetaBuildInputs,
    c_coords: [3]usize,
    lc_total: usize,
) void {
    const axis = first_non_zero_axis(usize, c_coords).?;
    var c_dec = c_coords;
    c_dec[axis] -= 1;
    const c_idx = coord_index3(c_coords, layout.c_stride_x, layout.c_stride_y);
    const c_dec_idx = coord_index3(c_dec, layout.c_stride_x, layout.c_stride_y);
    const base_dec = c_dec_idx * layout.m_stride;
    const base_out = c_idx * layout.m_stride;
    const ci_after = @as(f64, @floatFromInt(c_dec[axis]));
    for (0..layout.m_max + 1 - lc_total) |m| {
        var val = inputs.qc[axis] * theta[base_dec + m] + inputs.wq[axis] * theta[base_dec + m + 1];
        if (c_dec[axis] > 0) {
            var c_dec2 = c_dec;
            c_dec2[axis] -= 1;
            const base_dec2 =
                coord_index3(c_dec2, layout.c_stride_x, layout.c_stride_y) * layout.m_stride;
            val += ci_after * inputs.inv_2q *
                (theta[base_dec2 + m] - inputs.rho_over_q * theta[base_dec2 + m + 1]);
        }
        theta[base_out + m] = val;
    }
}

fn build_theta_c_direction(theta: []f64, layout: ThetaLayout, inputs: ThetaBuildInputs) void {
    for (1..layout.Lc + 1) |lc_total| {
        var cx: usize = lc_total;
        while (true) {
            var cy: usize = lc_total - cx;
            while (true) {
                fill_theta_c_entry(
                    theta,
                    layout,
                    inputs,
                    .{ cx, cy, lc_total - cx - cy },
                    lc_total,
                );
                if (cy == 0) break;
                cy -= 1;
            }
            if (cx == 0) break;
            cx -= 1;
        }
    }
}

fn fill_theta_a_entry(
    theta: []f64,
    layout: ThetaLayout,
    inputs: ThetaBuildInputs,
    a_coords: [3]usize,
    a_dec: [3]usize,
    axis: usize,
    c_coords: [3]usize,
    la_total: usize,
    lc_total: usize,
) void {
    const a_idx = coord_index3(a_coords, layout.a_stride_x, layout.a_stride_y);
    const a_dec_idx = coord_index3(a_dec, layout.a_stride_x, layout.a_stride_y);
    const c_idx = coord_index3(c_coords, layout.c_stride_x, layout.c_stride_y);
    const base_dec = a_dec_idx * layout.c_size * layout.m_stride + c_idx * layout.m_stride;
    const base_out = a_idx * layout.c_size * layout.m_stride + c_idx * layout.m_stride;
    const ai_after = @as(f64, @floatFromInt(a_dec[axis]));
    for (0..layout.m_max + 1 - la_total - lc_total) |m| {
        var val = inputs.pa[axis] * theta[base_dec + m] + inputs.wp[axis] * theta[base_dec + m + 1];
        if (a_dec[axis] > 0) {
            var a_dec2 = a_dec;
            a_dec2[axis] -= 1;
            const base_dec2 =
                coord_index3(a_dec2, layout.a_stride_x, layout.a_stride_y) *
                layout.c_size *
                layout.m_stride +
                c_idx * layout.m_stride;
            val += ai_after * inputs.inv_2p *
                (theta[base_dec2 + m] - inputs.rho_over_p * theta[base_dec2 + m + 1]);
        }
        if (c_coords[axis] > 0) {
            var c_dec = c_coords;
            c_dec[axis] -= 1;
            const coupling_base =
                a_dec_idx * layout.c_size * layout.m_stride +
                coord_index3(c_dec, layout.c_stride_x, layout.c_stride_y) * layout.m_stride;
            val += @as(f64, @floatFromInt(c_coords[axis])) *
                inputs.inv_2pq *
                theta[coupling_base + m + 1];
        }
        theta[base_out + m] = val;
    }
}

fn build_theta_a_direction(theta: []f64, layout: ThetaLayout, inputs: ThetaBuildInputs) void {
    for (1..layout.La + 1) |la_total| {
        var ax: usize = la_total;
        while (true) {
            var ay: usize = la_total - ax;
            while (true) {
                const a_coords = [3]usize{ ax, ay, la_total - ax - ay };
                const axis = first_non_zero_axis(usize, a_coords).?;
                var a_dec = a_coords;
                a_dec[axis] -= 1;
                for (0..layout.Lc + 1) |lc_total| {
                    var cx: usize = lc_total;
                    while (true) {
                        var cy: usize = lc_total - cx;
                        while (true) {
                            fill_theta_a_entry(
                                theta,
                                layout,
                                inputs,
                                a_coords,
                                a_dec,
                                axis,
                                .{ cx, cy, lc_total - cx - cy },
                                la_total,
                                lc_total,
                            );
                            if (cy == 0) break;
                            cy -= 1;
                        }
                        if (cx == 0) break;
                        cx -= 1;
                    }
                }
                if (ay == 0) break;
                ay -= 1;
            }
            if (ax == 0) break;
            ax -= 1;
        }
    }
}

fn build_theta_table(
    theta: []f64,
    layout: ThetaLayout,
    inputs: ThetaBuildInputs,
    zero_used: bool,
) void {
    init_theta_base(theta, layout, inputs.boys, zero_used);
    build_theta_c_direction(theta, layout, inputs);
    build_theta_a_direction(theta, layout, inputs);
}

fn eri_horizontal_with_layout(
    a_arr: [3]u32,
    b_arr: [3]u32,
    c_arr: [3]u32,
    d_arr: [3]u32,
    theta: []const f64,
    layout: ThetaLayout,
    ab: [3]f64,
    cd_vec: [3]f64,
) f64 {
    return eri_horizontal_arr(a_arr, b_arr, c_arr, d_arr, layout.view(theta), ab, cd_vec);
}

/// Compute ERI between four primitive Cartesian Gaussians using table-based Obara-Saika.
///
/// (ab|cd) = <ab|1/r₁₂|cd>
///
/// Algorithm:
///   1. Compute intermediate quantities (P, Q, W, Boys values, etc.)
///   2. Build vertical recurrence table theta[a][c][m] = [a,0|c,0]^(m) bottom-up
///   3. Apply horizontal recurrence to recover [a,b|c,d]
pub fn primitive_eri(
    alpha: f64,
    center_a: math.Vec3,
    a: AngularMomentum,
    beta: f64,
    center_b: math.Vec3,
    b: AngularMomentum,
    gamma: f64,
    center_c: math.Vec3,
    c_am: AngularMomentum,
    delta: f64,
    center_d: math.Vec3,
    d_am: AngularMomentum,
) f64 {
    const setup = init_primitive_eri_setup(
        alpha,
        center_a,
        a,
        beta,
        center_b,
        b,
        gamma,
        center_c,
        c_am,
        delta,
        center_d,
        d_am,
    );
    if (setup.layout.theta_size > MAX_STACK_THETA_TABLE) {
        return setup.prefactor * primitive_eri_recursive(
            a,
            b,
            c_am,
            d_am,
            &setup.boys,
            setup.pa,
            setup.qc,
            setup.wp,
            setup.wq,
            setup.ab,
            setup.cd,
            setup.inv_2p,
            setup.inv_2q,
            setup.inv_2pq,
            setup.rho_over_p,
            setup.rho_over_q,
        );
    }

    var theta: [MAX_STACK_THETA_TABLE]f64 = undefined;
    build_theta_table(
        theta[0..setup.layout.theta_size],
        setup.layout,
        theta_build_inputs(&setup),
        true,
    );
    return setup.prefactor * eri_horizontal_with_layout(
        .{ a.x, a.y, a.z },
        .{ b.x, b.y, b.z },
        .{ c_am.x, c_am.y, c_am.z },
        .{ d_am.x, d_am.y, d_am.z },
        theta[0..setup.layout.theta_size],
        setup.layout,
        setup.ab,
        setup.cd,
    );
}

/// Horizontal recurrence using the pre-built vertical table.
/// This is still recursive but only in b and d (which are small: l <= 3),
/// and each call does O(1) work (table lookup), so the total cost is
/// O(2^(lb+ld)) which is at most 2^6 = 64 for f+f.
fn eri_horizontal(
    a_arr: [3]u32,
    b_arr: [3]u32,
    c_arr: [3]u32,
    d_arr: [3]u32,
    theta: []const f64,
    a_stride_x: usize,
    a_stride_y: usize,
    c_size: usize,
    c_stride_x: usize,
    c_stride_y: usize,
    m_stride: usize,
    ab: [3]f64,
    cd_vec: [3]f64,
) f64 {
    return eri_horizontal_arr(a_arr, b_arr, c_arr, d_arr, .{
        .theta = theta,
        .a_stride_x = a_stride_x,
        .a_stride_y = a_stride_y,
        .c_size = c_size,
        .c_stride_x = c_stride_x,
        .c_stride_y = c_stride_y,
        .m_stride = m_stride,
    }, ab, cd_vec);
}

/// Fallback recursive ERI for very large angular momentum (exceeds stack table size).
fn primitive_eri_recursive(
    a: AngularMomentum,
    b: AngularMomentum,
    c_am: AngularMomentum,
    d_am: AngularMomentum,
    boys_vals: []const f64,
    pa_v: [3]f64,
    qc_v: [3]f64,
    wp_v: [3]f64,
    wq_v: [3]f64,
    ab_v: [3]f64,
    cd_v: [3]f64,
    inv_2p: f64,
    inv_2q: f64,
    inv_2pq: f64,
    rho_over_p: f64,
    rho_over_q: f64,
) f64 {
    const params = EriParams{
        .pa = pa_v,
        .qc = qc_v,
        .wp = wp_v,
        .wq = wq_v,
        .ab = ab_v,
        .cd = cd_v,
        .inv_2p = inv_2p,
        .inv_2q = inv_2q,
        .inv_2pq = inv_2pq,
        .rho_over_p = rho_over_p,
        .rho_over_q = rho_over_q,
        .boys = boys_vals,
    };
    return eri_recursive(
        .{ a.x, a.y, a.z },
        .{ b.x, b.y, b.z },
        .{ c_am.x, c_am.y, c_am.z },
        .{ d_am.x, d_am.y, d_am.z },
        0,
        params,
    );
}

const EriParams = struct {
    pa: [3]f64,
    qc: [3]f64,
    wp: [3]f64,
    wq: [3]f64,
    ab: [3]f64,
    cd: [3]f64,
    inv_2p: f64,
    inv_2q: f64,
    inv_2pq: f64,
    rho_over_p: f64,
    rho_over_q: f64,
    boys: []const f64,
};

/// Recursive ERI evaluation (fallback for large angular momentum).
fn eri_recursive(
    a: [3]u32,
    b: [3]u32,
    c: [3]u32,
    d: [3]u32,
    m: u32,
    params: EriParams,
) f64 {
    // Transfer d → c
    for (0..3) |i| {
        if (d[i] > 0) {
            var d_dec = d;
            d_dec[i] -= 1;
            var c_inc = c;
            c_inc[i] += 1;
            const cd_i = params.cd[i];
            return eri_recursive(a, b, c_inc, d_dec, m, params) +
                cd_i * eri_recursive(a, b, c, d_dec, m, params);
        }
    }

    // Transfer b → a
    for (0..3) |i| {
        if (b[i] > 0) {
            var b_dec = b;
            b_dec[i] -= 1;
            var a_inc = a;
            a_inc[i] += 1;
            const ab_i = params.ab[i];
            return eri_recursive(a_inc, b_dec, c, d, m, params) +
                ab_i * eri_recursive(a, b_dec, c, d, m, params);
        }
    }

    // b = d = 0: vertical recurrence
    return eri_vertical(a, c, m, params);
}

/// Vertical recurrence for [a,0|c,0]^(m) (fallback recursive version).
fn eri_vertical(
    a: [3]u32,
    c: [3]u32,
    m: u32,
    params: EriParams,
) f64 {
    if (a[0] == 0 and a[1] == 0 and a[2] == 0 and c[0] == 0 and c[1] == 0 and c[2] == 0) {
        return params.boys[m];
    }

    var axis: usize = 0;
    var from_a = true;

    if (a[0] > 0 or a[1] > 0 or a[2] > 0) {
        if (a[0] > 0) {
            axis = 0;
        } else if (a[1] > 0) {
            axis = 1;
        } else {
            axis = 2;
        }
        from_a = true;
    } else {
        if (c[0] > 0) {
            axis = 0;
        } else if (c[1] > 0) {
            axis = 1;
        } else {
            axis = 2;
        }
        from_a = false;
    }

    if (from_a) {
        var a_dec = a;
        a_dec[axis] -= 1;

        var result = params.pa[axis] * eri_vertical(a_dec, c, m, params) +
            params.wp[axis] * eri_vertical(a_dec, c, m + 1, params);

        if (a_dec[axis] >= 1) {
            var a_dec2 = a_dec;
            a_dec2[axis] -= 1;
            const ai = @as(f64, @floatFromInt(a_dec[axis]));
            result += ai * params.inv_2p * (eri_vertical(a_dec2, c, m, params) -
                params.rho_over_p * eri_vertical(a_dec2, c, m + 1, params));
        }

        if (c[axis] >= 1) {
            var c_dec = c;
            c_dec[axis] -= 1;
            const ci = @as(f64, @floatFromInt(c[axis]));
            result += ci * params.inv_2pq * eri_vertical(a_dec, c_dec, m + 1, params);
        }

        return result;
    } else {
        var c_dec = c;
        c_dec[axis] -= 1;

        var result = params.qc[axis] * eri_vertical(a, c_dec, m, params) +
            params.wq[axis] * eri_vertical(a, c_dec, m + 1, params);

        if (c_dec[axis] >= 1) {
            var c_dec2 = c_dec;
            c_dec2[axis] -= 1;
            const ci = @as(f64, @floatFromInt(c_dec[axis]));
            result += ci * params.inv_2q * (eri_vertical(a, c_dec2, m, params) -
                params.rho_over_q * eri_vertical(a, c_dec2, m + 1, params));
        }

        return result;
    }
}

/// Compute contracted ERI between four shells for specific Cartesian components.
pub fn contracted_eri(
    shell_a: ContractedShell,
    a_cart: AngularMomentum,
    shell_b: ContractedShell,
    b_cart: AngularMomentum,
    shell_c: ContractedShell,
    c_cart: AngularMomentum,
    shell_d: ContractedShell,
    d_cart: AngularMomentum,
) f64 {
    var result: f64 = 0.0;

    // Pre-compute distances for screening
    const diff_ab = math.Vec3.sub(shell_a.center, shell_b.center);
    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);
    const diff_cd = math.Vec3.sub(shell_c.center, shell_d.center);
    const r2_cd = math.Vec3.dot(diff_cd, diff_cd);
    const prim_screen_threshold: f64 = 1e-15;

    for (shell_a.primitives) |pa| {
        const na = basis_mod.normalization(pa.alpha, a_cart.x, a_cart.y, a_cart.z);
        for (shell_b.primitives) |pb| {
            const nb = basis_mod.normalization(pb.alpha, b_cart.x, b_cart.y, b_cart.z);
            const p_val = pa.alpha + pb.alpha;
            const mu_ab = pa.alpha * pb.alpha / p_val;
            const exp_ab = @exp(-mu_ab * r2_ab);

            for (shell_c.primitives) |pc| {
                const nc = basis_mod.normalization(pc.alpha, c_cart.x, c_cart.y, c_cart.z);
                for (shell_d.primitives) |pd| {
                    const nd = basis_mod.normalization(pd.alpha, d_cart.x, d_cart.y, d_cart.z);

                    // Primitive screening
                    const q_val = pc.alpha + pd.alpha;
                    const mu_cd = pc.alpha * pd.alpha / q_val;
                    const exp_factor = exp_ab * @exp(-mu_cd * r2_cd);
                    const two_pi_2p5 = 2.0 * std.math.pow(f64, std.math.pi, 2.5);
                    const prefactor_bound =
                        two_pi_2p5 / (p_val * q_val * @sqrt(p_val + q_val)) * exp_factor;
                    const coeff_product = pa.coeff * pb.coeff * pc.coeff * pd.coeff;
                    const norm_product = na * nb * nc * nd;
                    if (@abs(coeff_product) * norm_product * prefactor_bound <
                        prim_screen_threshold) continue;

                    const prim = primitive_eri(
                        pa.alpha,
                        shell_a.center,
                        a_cart,
                        pb.alpha,
                        shell_b.center,
                        b_cart,
                        pc.alpha,
                        shell_c.center,
                        c_cart,
                        pd.alpha,
                        shell_d.center,
                        d_cart,
                    );
                    result += coeff_product * na * nb * nc * nd * prim;
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Shell-Quartet Batch ERI — computes ALL Cartesian ERIs for a shell quartet at once
// ============================================================================

/// Pre-computed normalization constants for all primitives × all Cartesian components in a shell.
/// norm_table[ip * num_cart + ic] =
///   normalization(prim[ip].alpha, cart[ic].x, cart[ic].y, cart[ic].z)
const MAX_PRIM: usize = 16; // max primitives per shell
const MAX_CART_BATCH: usize = 15; // max Cartesian components (f-type = 10, d-type = 6)
const MAX_NORM_TABLE: usize = MAX_PRIM * MAX_CART_BATCH; // 240
const MAX_SHELL_BATCH: usize = 15 * 15 * 15 * 15;
const PRIM_SCREEN_THRESHOLD: f64 = 1e-15;
const SCHWARZ_THRESHOLD: f64 = 1e-12;

const NormTable = struct {
    values: [MAX_NORM_TABLE]f64,
    max: [MAX_PRIM]f64,
};

const ShellQuartetBatchSetup = struct {
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    cart_a: [MAX_CART_BATCH]AngularMomentum,
    cart_b: [MAX_CART_BATCH]AngularMomentum,
    cart_c: [MAX_CART_BATCH]AngularMomentum,
    cart_d: [MAX_CART_BATCH]AngularMomentum,
    na: usize,
    nb: usize,
    nc: usize,
    nd: usize,
    total_out: usize,
    layout: ThetaLayout,
    ab: [3]f64,
    cd: [3]f64,
    r2_ab: f64,
    r2_cd: f64,
};

const BatchNormTables = struct {
    a: NormTable,
    b: NormTable,
    c: NormTable,
    d: NormTable,
};

const PrimitiveAbSetup = struct {
    ipa: usize,
    ipb: usize,
    coeff_ab: f64,
    exp_ab: f64,
    p_val: f64,
    p_center: math.Vec3,
    pa: [3]f64,
    inv_2p: f64,
};

fn init_norm_table(
    primitives: []const PrimitiveGaussian,
    cart: []const AngularMomentum,
    num_cart: usize,
) NormTable {
    var table: NormTable = undefined;
    for (primitives, 0..) |prim, ip| {
        var mx: f64 = 0.0;
        for (0..num_cart) |ic| {
            const am = cart[ic];
            const norm = basis_mod.normalization(prim.alpha, am.x, am.y, am.z);
            table.values[ip * num_cart + ic] = norm;
            if (norm > mx) mx = norm;
        }
        table.max[ip] = mx;
    }
    return table;
}

fn init_shell_quartet_batch_setup(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
) ShellQuartetBatchSetup {
    const na = basis_mod.num_cartesian(shell_a.l);
    const nb = basis_mod.num_cartesian(shell_b.l);
    const nc = basis_mod.num_cartesian(shell_c.l);
    const nd = basis_mod.num_cartesian(shell_d.l);
    const diff_ab = math.Vec3.sub(shell_a.center, shell_b.center);
    const diff_cd = math.Vec3.sub(shell_c.center, shell_d.center);
    return .{
        .shell_a = shell_a,
        .shell_b = shell_b,
        .shell_c = shell_c,
        .shell_d = shell_d,
        .cart_a = basis_mod.cartesian_exponents(shell_a.l),
        .cart_b = basis_mod.cartesian_exponents(shell_b.l),
        .cart_c = basis_mod.cartesian_exponents(shell_c.l),
        .cart_d = basis_mod.cartesian_exponents(shell_d.l),
        .na = na,
        .nb = nb,
        .nc = nc,
        .nd = nd,
        .total_out = na * nb * nc * nd,
        .layout = ThetaLayout.init(shell_a.l + shell_b.l, shell_c.l + shell_d.l),
        .ab = .{
            shell_a.center.x - shell_b.center.x,
            shell_a.center.y - shell_b.center.y,
            shell_a.center.z - shell_b.center.z,
        },
        .cd = .{
            shell_c.center.x - shell_d.center.x,
            shell_c.center.y - shell_d.center.y,
            shell_c.center.z - shell_d.center.z,
        },
        .r2_ab = math.Vec3.dot(diff_ab, diff_ab),
        .r2_cd = math.Vec3.dot(diff_cd, diff_cd),
    };
}

fn init_batch_norm_tables(setup: ShellQuartetBatchSetup) BatchNormTables {
    return .{
        .a = init_norm_table(setup.shell_a.primitives, setup.cart_a[0..], setup.na),
        .b = init_norm_table(setup.shell_b.primitives, setup.cart_b[0..], setup.nb),
        .c = init_norm_table(setup.shell_c.primitives, setup.cart_c[0..], setup.nc),
        .d = init_norm_table(setup.shell_d.primitives, setup.cart_d[0..], setup.nd),
    };
}

fn fill_contracted_shell_quartet_fallback(
    setup: ShellQuartetBatchSetup,
    output: []f64,
) void {
    for (0..setup.na) |ia| {
        for (0..setup.nb) |ib| {
            for (0..setup.nc) |ic| {
                for (0..setup.nd) |id| {
                    const idx =
                        ia * setup.nb * setup.nc * setup.nd +
                        ib * setup.nc * setup.nd +
                        ic * setup.nd +
                        id;
                    output[idx] = contracted_eri(
                        setup.shell_a,
                        setup.cart_a[ia],
                        setup.shell_b,
                        setup.cart_b[ib],
                        setup.shell_c,
                        setup.cart_c[ic],
                        setup.shell_d,
                        setup.cart_d[id],
                    );
                }
            }
        }
    }
}

fn init_primitive_ab_setup(
    setup: ShellQuartetBatchSetup,
    prim_a: PrimitiveGaussian,
    ipa: usize,
    prim_b: PrimitiveGaussian,
    ipb: usize,
) PrimitiveAbSetup {
    const p_val = prim_a.alpha + prim_b.alpha;
    const p_center = math.Vec3{
        .x = (prim_a.alpha * setup.shell_a.center.x +
            prim_b.alpha * setup.shell_b.center.x) / p_val,
        .y = (prim_a.alpha * setup.shell_a.center.y +
            prim_b.alpha * setup.shell_b.center.y) / p_val,
        .z = (prim_a.alpha * setup.shell_a.center.z +
            prim_b.alpha * setup.shell_b.center.z) / p_val,
    };
    return .{
        .ipa = ipa,
        .ipb = ipb,
        .coeff_ab = prim_a.coeff * prim_b.coeff,
        .exp_ab = @exp(-(prim_a.alpha * prim_b.alpha / p_val) * setup.r2_ab),
        .p_val = p_val,
        .p_center = p_center,
        .pa = .{
            p_center.x - setup.shell_a.center.x,
            p_center.y - setup.shell_a.center.y,
            p_center.z - setup.shell_a.center.z,
        },
        .inv_2p = 0.5 / p_val,
    };
}

fn maybe_build_primitive_quartet_theta(
    theta: []f64,
    setup: ShellQuartetBatchSetup,
    norms: BatchNormTables,
    ab_setup: PrimitiveAbSetup,
    prim_c: PrimitiveGaussian,
    ipc: usize,
    prim_d: PrimitiveGaussian,
    ipd: usize,
) ?f64 {
    const q_val = prim_c.alpha + prim_d.alpha;
    const mu_cd = prim_c.alpha * prim_d.alpha / q_val;
    const coeff_abcd = ab_setup.coeff_ab * prim_c.coeff * prim_d.coeff;
    const prefactor = 2.0 * std.math.pow(f64, std.math.pi, 2.5) /
        (ab_setup.p_val * q_val * @sqrt(ab_setup.p_val + q_val)) *
        ab_setup.exp_ab *
        @exp(-mu_cd * setup.r2_cd);
    const max_norm = norms.a.max[ab_setup.ipa] * norms.b.max[ab_setup.ipb] *
        norms.c.max[ipc] * norms.d.max[ipd];
    if (@abs(coeff_abcd) * max_norm * prefactor < PRIM_SCREEN_THRESHOLD) return null;

    const rho = ab_setup.p_val * q_val / (ab_setup.p_val + q_val);
    const q_center = math.Vec3{
        .x = (prim_c.alpha * setup.shell_c.center.x +
            prim_d.alpha * setup.shell_d.center.x) / q_val,
        .y = (prim_c.alpha * setup.shell_c.center.y +
            prim_d.alpha * setup.shell_d.center.y) / q_val,
        .z = (prim_c.alpha * setup.shell_c.center.z +
            prim_d.alpha * setup.shell_d.center.z) / q_val,
    };
    const w_center = math.Vec3{
        .x = (ab_setup.p_val * ab_setup.p_center.x + q_val * q_center.x) / (ab_setup.p_val + q_val),
        .y = (ab_setup.p_val * ab_setup.p_center.y + q_val * q_center.y) / (ab_setup.p_val + q_val),
        .z = (ab_setup.p_val * ab_setup.p_center.z + q_val * q_center.z) / (ab_setup.p_val + q_val),
    };
    const diff_pq = math.Vec3.sub(ab_setup.p_center, q_center);
    var boys: [ERI_MAX_M]f64 = undefined;
    boys_mod.boys_batch(
        @as(u32, @intCast(setup.layout.m_max)),
        rho * math.Vec3.dot(diff_pq, diff_pq),
        &boys,
    );
    build_theta_table(theta, setup.layout, .{
        .boys = &boys,
        .pa = ab_setup.pa,
        .qc = .{
            q_center.x - setup.shell_c.center.x,
            q_center.y - setup.shell_c.center.y,
            q_center.z - setup.shell_c.center.z,
        },
        .wp = .{
            w_center.x - ab_setup.p_center.x,
            w_center.y - ab_setup.p_center.y,
            w_center.z - ab_setup.p_center.z,
        },
        .wq = .{
            w_center.x - q_center.x,
            w_center.y - q_center.y,
            w_center.z - q_center.z,
        },
        .inv_2p = ab_setup.inv_2p,
        .inv_2q = 0.5 / q_val,
        .inv_2pq = 0.5 / (ab_setup.p_val + q_val),
        .rho_over_p = rho / ab_setup.p_val,
        .rho_over_q = rho / q_val,
    }, false);
    return prefactor * coeff_abcd;
}

fn accumulate_primitive_quartet_output(
    theta: []const f64,
    setup: ShellQuartetBatchSetup,
    norms: BatchNormTables,
    output: []f64,
    ipa: usize,
    ipb: usize,
    ipc: usize,
    ipd: usize,
    prim_prefactor: f64,
) void {
    for (0..setup.na) |ia| {
        const na_val = norms.a.values[ipa * setup.na + ia];
        for (0..setup.nb) |ib| {
            const nab = na_val * norms.b.values[ipb * setup.nb + ib];
            for (0..setup.nc) |ic| {
                const nabc = nab * norms.c.values[ipc * setup.nc + ic];
                for (0..setup.nd) |id| {
                    const idx =
                        ia * setup.nb * setup.nc * setup.nd +
                        ib * setup.nc * setup.nd +
                        ic * setup.nd +
                        id;
                    output[idx] += prim_prefactor *
                        nabc *
                        norms.d.values[ipd * setup.nd + id] *
                        eri_horizontal_with_layout(
                            .{ setup.cart_a[ia].x, setup.cart_a[ia].y, setup.cart_a[ia].z },
                            .{ setup.cart_b[ib].x, setup.cart_b[ib].y, setup.cart_b[ib].z },
                            .{ setup.cart_c[ic].x, setup.cart_c[ic].y, setup.cart_c[ic].z },
                            .{ setup.cart_d[id].x, setup.cart_d[id].y, setup.cart_d[id].z },
                            theta,
                            setup.layout,
                            setup.ab,
                            setup.cd,
                        );
                }
            }
        }
    }
}

fn accumulate_contracted_shell_quartet(
    theta: []f64,
    setup: ShellQuartetBatchSetup,
    norms: BatchNormTables,
    output: []f64,
) void {
    for (setup.shell_a.primitives, 0..) |prim_a, ipa| {
        for (setup.shell_b.primitives, 0..) |prim_b, ipb| {
            const ab_setup = init_primitive_ab_setup(setup, prim_a, ipa, prim_b, ipb);
            for (setup.shell_c.primitives, 0..) |prim_c, ipc| {
                for (setup.shell_d.primitives, 0..) |prim_d, ipd| {
                    const prim_prefactor = maybe_build_primitive_quartet_theta(
                        theta,
                        setup,
                        norms,
                        ab_setup,
                        prim_c,
                        ipc,
                        prim_d,
                        ipd,
                    ) orelse continue;
                    accumulate_primitive_quartet_output(
                        theta,
                        setup,
                        norms,
                        output,
                        ipa,
                        ipb,
                        ipc,
                        ipd,
                        prim_prefactor,
                    );
                }
            }
        }
    }
}

const ShellIndexMap = struct {
    offsets: [128]usize,
    sizes: [128]usize,
    n_shells: usize,
};

fn init_shell_index_map(shells: []const ContractedShell) ShellIndexMap {
    var map: ShellIndexMap = undefined;
    map.n_shells = shells.len;
    var offset: usize = 0;
    for (shells, 0..) |shell, si| {
        map.offsets[si] = offset;
        map.sizes[si] = shell.num_cartesian_functions();
        offset += map.sizes[si];
    }
    return map;
}

fn build_schwarz_table(shells: []const ContractedShell, map: ShellIndexMap) [128 * 128]f64 {
    var schwarz_q: [128 * 128]f64 = undefined;
    var schwarz_buf: [MAX_SHELL_BATCH]f64 = undefined;
    for (0..map.n_shells) |si| {
        for (si..map.n_shells) |sj| {
            _ = rys_eri.contracted_shell_quartet_eri(
                shells[si],
                shells[sj],
                shells[si],
                shells[sj],
                &schwarz_buf,
            );
            var max_val: f64 = 0.0;
            for (0..map.sizes[si]) |ia| {
                for (0..map.sizes[sj]) |ib| {
                    const idx = ia * map.sizes[sj] * map.sizes[si] * map.sizes[sj] +
                        ib * map.sizes[si] * map.sizes[sj] +
                        ia * map.sizes[sj] +
                        ib;
                    max_val = @max(max_val, @abs(schwarz_buf[idx]));
                }
            }
            const q_val = @sqrt(max_val);
            schwarz_q[si * map.n_shells + sj] = q_val;
            schwarz_q[sj * map.n_shells + si] = q_val;
        }
    }
    return schwarz_q;
}

fn distribute_shell_quartet(
    values: []f64,
    eri_buf: []const f64,
    map: ShellIndexMap,
    si: usize,
    sj: usize,
    sk: usize,
    sl: usize,
) void {
    for (0..map.sizes[si]) |ia| {
        const i = map.offsets[si] + ia;
        for (0..map.sizes[sj]) |ib| {
            const j = map.offsets[sj] + ib;
            if (i < j) continue;
            const ij = triangular_index(i, j);
            for (0..map.sizes[sk]) |ic| {
                const k = map.offsets[sk] + ic;
                for (0..map.sizes[sl]) |id| {
                    const l = map.offsets[sl] + id;
                    if (k < l) continue;
                    const kl = triangular_index(k, l);
                    const idx = ia * map.sizes[sj] * map.sizes[sk] * map.sizes[sl] +
                        ib * map.sizes[sk] * map.sizes[sl] +
                        ic * map.sizes[sl] +
                        id;
                    values[triangular_index(@max(ij, kl), @min(ij, kl))] = eri_buf[idx];
                }
            }
        }
    }
}

fn fill_eri_table_values(
    values: []f64,
    shells: []const ContractedShell,
    map: ShellIndexMap,
    schwarz_q: [128 * 128]f64,
) void {
    var eri_buf: [MAX_SHELL_BATCH]f64 = undefined;
    for (0..map.n_shells) |si| {
        for (0..si + 1) |sj| {
            const ab_pair = shell_pair_index(si, sj);
            const q_ab = schwarz_q[si * map.n_shells + sj];
            for (0..map.n_shells) |sk| {
                for (0..sk + 1) |sl| {
                    if (ab_pair < shell_pair_index(sk, sl)) continue;
                    if (q_ab * schwarz_q[sk * map.n_shells + sl] < SCHWARZ_THRESHOLD) continue;
                    _ = rys_eri.contracted_shell_quartet_eri(
                        shells[si],
                        shells[sj],
                        shells[sk],
                        shells[sl],
                        &eri_buf,
                    );
                    distribute_shell_quartet(values, &eri_buf, map, si, sj, sk, sl);
                }
            }
        }
    }
}

/// Compute ALL contracted ERIs for a shell quartet (A,B|C,D) at once.
///
/// This is the key optimization: for each primitive quartet (pa,pb,pc,pd),
/// the theta table (vertical recurrence) and Boys function values are computed ONCE,
/// then the horizontal recurrence is applied for ALL Cartesian component combinations.
///
/// Output layout: output[ia * nb*nc*nd + ib * nc*nd + ic * nd + id]
/// where ia,ib,ic,id index Cartesian components within each shell.
///
/// Returns the number of ERIs computed (na * nb * nc * nd), or 0 if the shell quartet
/// could not be handled (falls back to per-integral computation).
pub fn contracted_shell_quartet_eri(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    output: []f64,
) usize {
    const setup = init_shell_quartet_batch_setup(shell_a, shell_b, shell_c, shell_d);
    std.debug.assert(output.len >= setup.total_out);
    @memset(output[0..setup.total_out], 0.0);

    if (setup.layout.theta_size > MAX_STACK_THETA_TABLE) {
        fill_contracted_shell_quartet_fallback(setup, output[0..setup.total_out]);
        return setup.total_out;
    }

    var theta: [MAX_STACK_THETA_TABLE]f64 = undefined;
    accumulate_contracted_shell_quartet(
        theta[0..setup.layout.theta_size],
        setup,
        init_batch_norm_tables(setup),
        output[0..setup.total_out],
    );
    return setup.total_out;
}

// ============================================================================
// General Matrix Builders (working with shells of any angular momentum)
// ============================================================================

/// Build the full overlap matrix for a set of shells with arbitrary angular momentum.
/// Each shell contributes num_cartesian(l) basis functions.
/// Returns a flat row-major N×N matrix where N = Σ num_cartesian(shell.l).
pub fn build_overlap_matrix(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
) ![]f64 {
    const n = total_basis_functions(shells);
    const mat = try alloc.alloc(f64, n * n);
    @memset(mat, 0.0);

    var mu: usize = 0;
    for (shells) |shell_a| {
        const cart_a = basis_mod.cartesian_exponents(shell_a.l);
        const na = shell_a.num_cartesian_functions();
        var nu: usize = 0;
        for (shells) |shell_b| {
            const cart_b = basis_mod.cartesian_exponents(shell_b.l);
            const nb = shell_b.num_cartesian_functions();

            for (0..na) |ia| {
                for (0..nb) |ib| {
                    const val = contracted_overlap(shell_a, cart_a[ia], shell_b, cart_b[ib]);
                    mat[(mu + ia) * n + (nu + ib)] = val;
                }
            }

            nu += nb;
        }
        mu += na;
    }

    return mat;
}

/// Build the full kinetic energy matrix for a set of shells.
pub fn build_kinetic_matrix(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
) ![]f64 {
    const n = total_basis_functions(shells);
    const mat = try alloc.alloc(f64, n * n);
    @memset(mat, 0.0);

    var mu: usize = 0;
    for (shells) |shell_a| {
        const cart_a = basis_mod.cartesian_exponents(shell_a.l);
        const na = shell_a.num_cartesian_functions();
        var nu: usize = 0;
        for (shells) |shell_b| {
            const cart_b = basis_mod.cartesian_exponents(shell_b.l);
            const nb = shell_b.num_cartesian_functions();

            for (0..na) |ia| {
                for (0..nb) |ib| {
                    const val = contracted_kinetic(shell_a, cart_a[ia], shell_b, cart_b[ib]);
                    mat[(mu + ia) * n + (nu + ib)] = val;
                }
            }

            nu += nb;
        }
        mu += na;
    }

    return mat;
}

/// Build the full nuclear attraction matrix for a set of shells.
pub fn build_nuclear_matrix(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
) ![]f64 {
    const n = total_basis_functions(shells);
    const mat = try alloc.alloc(f64, n * n);
    @memset(mat, 0.0);

    var mu: usize = 0;
    for (shells) |shell_a| {
        const cart_a = basis_mod.cartesian_exponents(shell_a.l);
        const na = shell_a.num_cartesian_functions();
        var nu: usize = 0;
        for (shells) |shell_b| {
            const cart_b = basis_mod.cartesian_exponents(shell_b.l);
            const nb = shell_b.num_cartesian_functions();

            for (0..na) |ia| {
                for (0..nb) |ib| {
                    const val = contracted_total_nuclear_attraction(
                        shell_a,
                        cart_a[ia],
                        shell_b,
                        cart_b[ib],
                        nuc_positions,
                        nuc_charges,
                    );
                    mat[(mu + ia) * n + (nu + ib)] = val;
                }
            }

            nu += nb;
        }
        mu += na;
    }

    return mat;
}

/// Build the ERI table for a set of shells with arbitrary angular momentum.
/// Uses Schwarz screening at the shell-quartet level, then batch ERI computation
/// (contracted_shell_quartet_eri) to share theta tables across Cartesian components.
/// Falls back to per-integral computation for shell quartets that exceed stack limits.
/// Returns a flat array indexed by compound index.
pub fn build_eri_table(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
) !GeneralEriTable {
    const n = total_basis_functions(shells);
    const nn = n * (n + 1) / 2;
    const size = nn * (nn + 1) / 2;
    const values = try alloc.alloc(f64, size);
    @memset(values, 0.0);
    const map = init_shell_index_map(shells);
    fill_eri_table_values(values, shells, map, build_schwarz_table(shells, map));
    return GeneralEriTable{ .values = values, .n = n };
}

/// Shell pair index: maps (a, b) with a >= b to a*(a+1)/2 + b.
fn shell_pair_index(a: usize, b: usize) usize {
    if (a >= b) return a * (a + 1) / 2 + b;
    return b * (b + 1) / 2 + a;
}

/// Triangular index: maps (i,j) with i >= j to i*(i+1)/2 + j.
fn triangular_index(i: usize, j: usize) usize {
    if (i >= j) {
        return i * (i + 1) / 2 + j;
    } else {
        return j * (j + 1) / 2 + i;
    }
}

/// ERI table for general angular momentum.
pub const GeneralEriTable = struct {
    values: []f64,
    n: usize,

    pub fn deinit(self: *GeneralEriTable, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
    }

    /// Get the ERI (ij|kl) using symmetry.
    pub fn get(self: GeneralEriTable, i: usize, j: usize, k: usize, l: usize) f64 {
        const ij = triangular_index(i, j);
        const kl = triangular_index(k, l);
        const idx = triangular_index(ij, kl);
        return self.values[idx];
    }
};

/// Compute total number of basis functions from a set of shells.
pub fn total_basis_functions(shells: []const ContractedShell) usize {
    var n: usize = 0;
    for (shells) |shell| {
        n += shell.num_cartesian_functions();
    }
    return n;
}

// ============================================================================
// Tests
// ============================================================================

test "OS overlap ss matches old implementation" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");

    const center_a = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = math.Vec3{ .x = 1.4, .y = 0.0, .z = 0.0 };

    const shell_a = ContractedShell{
        .center = center_a,
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const shell_b = ContractedShell{
        .center = center_b,
        .l = 0,
        .primitives = &sto3g.H_1s,
    };

    const s_cart = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

    // OS overlap for (ss)
    const s_os = contracted_overlap(shell_a, s_cart, shell_b, s_cart);

    // Old implementation
    const overlap_old = @import("overlap.zig");
    const s_old = overlap_old.overlap_ss(shell_a, shell_b);

    try testing.expectApproxEqAbs(s_old, s_os, 1e-12);
}

test "OS overlap self-overlap s" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const shell = ContractedShell{
        .center = center,
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const s_cart = AngularMomentum{ .x = 0, .y = 0, .z = 0 };
    const s = contracted_overlap(shell, s_cart, shell, s_cart);
    try testing.expectApproxEqAbs(1.0, s, 1e-4);
}

test "OS kinetic ss matches old implementation" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");

    const center_a = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = math.Vec3{ .x = 1.4, .y = 0.0, .z = 0.0 };

    const shell_a = ContractedShell{ .center = center_a, .l = 0, .primitives = &sto3g.H_1s };
    const shell_b = ContractedShell{ .center = center_b, .l = 0, .primitives = &sto3g.H_1s };

    const s_cart = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

    const t_os = contracted_kinetic(shell_a, s_cart, shell_b, s_cart);
    const kinetic_old = @import("kinetic.zig");
    const t_old = kinetic_old.kinetic_ss(shell_a, shell_b);

    try testing.expectApproxEqAbs(t_old, t_os, 1e-10);
}

test "OS nuclear attraction primitive p-type displaced nucleus" {
    const testing = std.testing;
    // Single primitive py at origin, nucleus at (0, 1.43, 1.11)
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const nuc = math.Vec3{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 };

    const alpha: f64 = 5.0331513;

    // <py|V|py> with a = b = (0,1,0), same exponent
    const v_yy = primitive_nuclear_attraction(
        alpha,
        center,
        .{ .x = 0, .y = 1, .z = 0 },
        alpha,
        center,
        .{ .x = 0, .y = 1, .z = 0 },
        nuc,
        1.0,
    );
    const v_xx = primitive_nuclear_attraction(
        alpha,
        center,
        .{ .x = 1, .y = 0, .z = 0 },
        alpha,
        center,
        .{ .x = 1, .y = 0, .z = 0 },
        nuc,
        1.0,
    );

    // Now test with alpha != beta
    const beta: f64 = 1.1695961;
    const v_yy_ab = primitive_nuclear_attraction(
        alpha,
        center,
        .{ .x = 0, .y = 1, .z = 0 },
        beta,
        center,
        .{ .x = 0, .y = 1, .z = 0 },
        nuc,
        1.0,
    );
    const v_xx_ab = primitive_nuclear_attraction(
        alpha,
        center,
        .{ .x = 1, .y = 0, .z = 0 },
        beta,
        center,
        .{ .x = 1, .y = 0, .z = 0 },
        nuc,
        1.0,
    );

    try testing.expectApproxEqAbs(-0.004711472014, v_xx, 1e-6);
    try testing.expectApproxEqAbs(-0.004847328847, v_yy, 1e-6);
    try testing.expectApproxEqAbs(-0.015656249025, v_xx_ab, 1e-6);
    try testing.expectApproxEqAbs(-0.016395994025, v_yy_ab, 1e-6);
}

test "OS nuclear attraction ss matches old implementation" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");

    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const shell = ContractedShell{ .center = center, .l = 0, .primitives = &sto3g.H_1s };

    const s_cart = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

    const v_os = contracted_nuclear_attraction(shell, s_cart, shell, s_cart, center, 1.0);
    const nuclear_old = @import("nuclear.zig");
    const v_old = nuclear_old.nuclear_attraction_ss(shell, shell, center, 1.0);

    try testing.expectApproxEqAbs(v_old, v_os, 1e-10);
}

test "OS ERI ssss matches old implementation" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");

    const center_a = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = math.Vec3{ .x = 1.4, .y = 0.0, .z = 0.0 };

    const shell_a = ContractedShell{ .center = center_a, .l = 0, .primitives = &sto3g.H_1s };
    const shell_b = ContractedShell{ .center = center_b, .l = 0, .primitives = &sto3g.H_1s };

    const s_cart = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

    // (aa|aa)
    const eri_os = contracted_eri(
        shell_a,
        s_cart,
        shell_a,
        s_cart,
        shell_a,
        s_cart,
        shell_a,
        s_cart,
    );
    const eri_old_mod = @import("eri.zig");
    const eri_old = eri_old_mod.eri_ssss(shell_a, shell_a, shell_a, shell_a);
    try testing.expectApproxEqAbs(eri_old, eri_os, 1e-10);

    // (ab|ab)
    const eri_os2 = contracted_eri(
        shell_a,
        s_cart,
        shell_b,
        s_cart,
        shell_a,
        s_cart,
        shell_b,
        s_cart,
    );
    const eri_old2 = eri_old_mod.eri_ssss(shell_a, shell_b, shell_a, shell_b);
    try testing.expectApproxEqAbs(eri_old2, eri_os2, 1e-10);
}

test "OS overlap pp self is identity block" {
    const testing = std.testing;
    // A single p-shell at origin with a single primitive alpha=1.0, coeff=1.0
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const prims = [_]PrimitiveGaussian{
        .{ .alpha = 1.0, .coeff = 1.0 },
    };
    const shell = ContractedShell{
        .center = center,
        .l = 1,
        .primitives = &prims,
    };

    const px = AngularMomentum{ .x = 1, .y = 0, .z = 0 };
    const py = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
    const pz = AngularMomentum{ .x = 0, .y = 0, .z = 1 };

    // Self-overlaps should be 1.0
    const sxx = contracted_overlap(shell, px, shell, px);
    const syy = contracted_overlap(shell, py, shell, py);
    const szz = contracted_overlap(shell, pz, shell, pz);
    try testing.expectApproxEqAbs(1.0, sxx, 1e-10);
    try testing.expectApproxEqAbs(1.0, syy, 1e-10);
    try testing.expectApproxEqAbs(1.0, szz, 1e-10);

    // Cross-overlaps should be 0.0
    const sxy = contracted_overlap(shell, px, shell, py);
    const sxz = contracted_overlap(shell, px, shell, pz);
    const syz = contracted_overlap(shell, py, shell, pz);
    try testing.expectApproxEqAbs(0.0, sxy, 1e-10);
    try testing.expectApproxEqAbs(0.0, sxz, 1e-10);
    try testing.expectApproxEqAbs(0.0, syz, 1e-10);
}

test "OS general matrix builder matches old for H2" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    const shells = [_]ContractedShell{
        .{ .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = .{ .x = 1.4, .y = 0.0, .z = 0.0 }, .l = 0, .primitives = &sto3g.H_1s },
    };

    // Build with general builder
    const s_gen = try build_overlap_matrix(alloc, &shells);
    defer alloc.free(s_gen);

    // Build with old builder
    const overlap_old = @import("overlap.zig");
    const s_old = try overlap_old.build_overlap_matrix(alloc, &shells);
    defer alloc.free(s_old);

    for (0..4) |i| {
        try testing.expectApproxEqAbs(s_old[i], s_gen[i], 1e-12);
    }
}
