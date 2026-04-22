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
fn overlap1D(
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
pub fn primitiveOverlap(
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
    const sx = overlap1D(a.x, b.x, pa_x, pb_x, inv_2p, sqrt_pi_p);
    const sy = overlap1D(a.y, b.y, pa_y, pb_y, inv_2p, sqrt_pi_p);
    const sz = overlap1D(a.z, b.z, pa_z, pb_z, inv_2p, sqrt_pi_p);

    return exp_factor * sx[a.x][b.x] * sy[a.y][b.y] * sz[a.z][b.z];
}

/// Compute the overlap integral between two contracted shells for specific
/// Cartesian components (a_cart, b_cart).
///
/// S = Σ_i Σ_j c_i c_j N_i N_j × primitiveOverlap(...)
pub fn contractedOverlap(
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
            const prim_s = primitiveOverlap(
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
/// This uses the 1D overlap array which we already have from overlap1D.
/// Compute 1D kinetic energy contribution for one dimension.
/// T_x(a,b) = β(2b+1)·S(a,b) - 2β²·S(a,b+2) - ½·b(b-1)·S(a,b-2)
/// where S is the 1D overlap array and β is the exponent of center B.
fn kinetic1D(
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
pub fn primitiveKinetic(
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
    const sx = overlap1D(a.x + 2, b.x + 2, pa_x, pb_x, inv_2p, sqrt_pi_p);
    const sy = overlap1D(a.y + 2, b.y + 2, pa_y, pb_y, inv_2p, sqrt_pi_p);
    const sz = overlap1D(a.z + 2, b.z + 2, pa_z, pb_z, inv_2p, sqrt_pi_p);

    // T = T_x · S_y · S_z + S_x · T_y · S_z + S_x · S_y · T_z
    const tx = kinetic1D(a.x, b.x, beta, sx);
    const ty = kinetic1D(a.y, b.y, beta, sy);
    const tz = kinetic1D(a.z, b.z, beta, sz);

    const sov_x = sx[a.x][b.x];
    const sov_y = sy[a.y][b.y];
    const sov_z = sz[a.z][b.z];

    return exp_factor * (tx * sov_y * sov_z + sov_x * ty * sov_z + sov_x * sov_y * tz);
}

/// Compute the kinetic energy integral between two contracted shells for
/// specific Cartesian components.
pub fn contractedKinetic(
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
            const prim_t = primitiveKinetic(
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
pub fn primitiveNuclearAttraction(
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
        boys[m] = boys_mod.boysN(@as(u32, @intCast(m)), arg);
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
    const aux_result = nuclearAux3D(
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

/// Recursive 3D Obara-Saika auxiliary nuclear attraction integral.
/// Θ^(m)_{a,b} with a = (ax,ay,az), b = (bx,by,bz).
///
/// Uses horizontal transfer to reduce b → 0 first, then vertical recurrence on a.
fn nuclearAux3D(
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
    // Step 1: Transfer from b to a using horizontal recurrence
    // Θ^(m)_{a,b+1_i} = Θ^(m)_{a+1_i,b} + (A_i - B_i) × Θ^(m)_{a,b}
    // So: Θ^(m)_{a,b} = Θ^(m)_{a+1_i,b-1_i} - (A_i - B_i) × Θ^(m)_{a,b-1_i}
    //
    // Actually the standard horizontal transfer relation is:
    //   Θ_{a,b+1_i}^(m) = Θ_{a+1_i,b}^(m) + (Ai-Bi) × Θ_{a,b}^(m)
    //
    // So to reduce b, we use: b_i > 0 =>
    //   Θ_{a,b}^(m) = Θ_{a+1_i,b-1_i}^(m) + (Bi-Ai) × Θ_{a,b-1_i}^(m)
    //                                         ^-- note sign!
    // Wait, from Θ_{a,b+1_i} = Θ_{a+1_i,b} + (Ai-Bi) Θ_{a,b}:
    // Set b' = b-1_i: Θ_{a,b} = Θ_{a+1_i,b-1_i} + (Ai-Bi) Θ_{a,b-1_i}
    //
    // So if bx > 0:
    if (bx > 0) {
        // Horizontal transfer: Θ_{a,b} = Θ_{a+1_i,b-1_i} + (Ai-Bi) Θ_{a,b-1_i}
        // PA = P - A, PB = P - B, so A - B = PB - PA.
        const t1 = nuclearAux3D(
            ax + 1,
            ay,
            az,
            bx - 1,
            by,
            bz,
            m,
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
            boys,
        );
        const t2 = nuclearAux3D(
            ax,
            ay,
            az,
            bx - 1,
            by,
            bz,
            m,
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
            boys,
        );
        return t1 + (pb_x - pa_x) * t2;
    }
    if (by > 0) {
        const t1 = nuclearAux3D(
            ax,
            ay + 1,
            az,
            bx,
            by - 1,
            bz,
            m,
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
            boys,
        );
        const t2 = nuclearAux3D(
            ax,
            ay,
            az,
            bx,
            by - 1,
            bz,
            m,
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
            boys,
        );
        return t1 + (pb_y - pa_y) * t2;
    }
    if (bz > 0) {
        const t1 = nuclearAux3D(
            ax,
            ay,
            az + 1,
            bx,
            by,
            bz - 1,
            m,
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
            boys,
        );
        const t2 = nuclearAux3D(
            ax,
            ay,
            az,
            bx,
            by,
            bz - 1,
            m,
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
            boys,
        );
        return t1 + (pb_z - pa_z) * t2;
    }

    // Step 2: b = (0,0,0), use vertical recurrence on a
    // Θ^(m)_{a+1_i,0} = PA_i × Θ^(m)_{a,0} + CP_i × Θ^(m+1)_{a,0}
    //     + a_i/(2p) × [Θ^(m)_{a-1_i,0} - Θ^(m+1)_{a-1_i,0}]
    return nuclearAuxVertical(ax, ay, az, m, pa_x, pa_y, pa_z, cp_x, cp_y, cp_z, inv_2p, boys);
}

/// Vertical recurrence for nuclear attraction with b = (0,0,0).
/// Θ^(m)_{a,0}
fn nuclearAuxVertical(
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
    // Base case: a = (0,0,0)
    if (ax == 0 and ay == 0 and az == 0) {
        return boys[m];
    }

    // Choose axis to decrement: pick one with nonzero angular momentum
    if (ax > 0) {
        // Θ^(m)_{ax,ay,az,0} via x-axis recurrence from (ax-1)
        const t_m0 = nuclearAuxVertical(
            ax - 1,
            ay,
            az,
            m,
            pa_x,
            pa_y,
            pa_z,
            cp_x,
            cp_y,
            cp_z,
            inv_2p,
            boys,
        );
        const t_m1 = nuclearAuxVertical(
            ax - 1,
            ay,
            az,
            m + 1,
            pa_x,
            pa_y,
            pa_z,
            cp_x,
            cp_y,
            cp_z,
            inv_2p,
            boys,
        );
        var result = pa_x * t_m0;
        result += cp_x * t_m1;
        if (ax >= 2) {
            const ai = @as(f64, @floatFromInt(ax - 1));
            const t2_m0 = nuclearAuxVertical(
                ax - 2,
                ay,
                az,
                m,
                pa_x,
                pa_y,
                pa_z,
                cp_x,
                cp_y,
                cp_z,
                inv_2p,
                boys,
            );
            const t2_m1 = nuclearAuxVertical(
                ax - 2,
                ay,
                az,
                m + 1,
                pa_x,
                pa_y,
                pa_z,
                cp_x,
                cp_y,
                cp_z,
                inv_2p,
                boys,
            );
            result += ai * inv_2p * (t2_m0 - t2_m1);
        }
        return result;
    }

    if (ay > 0) {
        const t_m0 = nuclearAuxVertical(
            ax,
            ay - 1,
            az,
            m,
            pa_x,
            pa_y,
            pa_z,
            cp_x,
            cp_y,
            cp_z,
            inv_2p,
            boys,
        );
        const t_m1 = nuclearAuxVertical(
            ax,
            ay - 1,
            az,
            m + 1,
            pa_x,
            pa_y,
            pa_z,
            cp_x,
            cp_y,
            cp_z,
            inv_2p,
            boys,
        );
        var result = pa_y * t_m0;
        result += cp_y * t_m1;
        if (ay >= 2) {
            const ai = @as(f64, @floatFromInt(ay - 1));
            const t2_m0 = nuclearAuxVertical(
                ax,
                ay - 2,
                az,
                m,
                pa_x,
                pa_y,
                pa_z,
                cp_x,
                cp_y,
                cp_z,
                inv_2p,
                boys,
            );
            const t2_m1 = nuclearAuxVertical(
                ax,
                ay - 2,
                az,
                m + 1,
                pa_x,
                pa_y,
                pa_z,
                cp_x,
                cp_y,
                cp_z,
                inv_2p,
                boys,
            );
            result += ai * inv_2p * (t2_m0 - t2_m1);
        }
        return result;
    }

    // az > 0
    const t_m0 = nuclearAuxVertical(
        ax,
        ay,
        az - 1,
        m,
        pa_x,
        pa_y,
        pa_z,
        cp_x,
        cp_y,
        cp_z,
        inv_2p,
        boys,
    );
    const t_m1 = nuclearAuxVertical(
        ax,
        ay,
        az - 1,
        m + 1,
        pa_x,
        pa_y,
        pa_z,
        cp_x,
        cp_y,
        cp_z,
        inv_2p,
        boys,
    );
    var result = pa_z * t_m0;
    result += cp_z * t_m1;
    if (az >= 2) {
        const ai = @as(f64, @floatFromInt(az - 1));
        const t2_m0 = nuclearAuxVertical(
            ax,
            ay,
            az - 2,
            m,
            pa_x,
            pa_y,
            pa_z,
            cp_x,
            cp_y,
            cp_z,
            inv_2p,
            boys,
        );
        const t2_m1 = nuclearAuxVertical(
            ax,
            ay,
            az - 2,
            m + 1,
            pa_x,
            pa_y,
            pa_z,
            cp_x,
            cp_y,
            cp_z,
            inv_2p,
            boys,
        );
        result += ai * inv_2p * (t2_m0 - t2_m1);
    }
    return result;
}

/// Compute nuclear attraction integral between two contracted shells for
/// specific Cartesian components and a single nucleus.
pub fn contractedNuclearAttraction(
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
            const prim_v = primitiveNuclearAttraction(
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
pub fn contractedTotalNuclearAttraction(
    shell_a: ContractedShell,
    a_cart: AngularMomentum,
    shell_b: ContractedShell,
    b_cart: AngularMomentum,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
) f64 {
    var result: f64 = 0.0;
    for (nuc_positions, 0..) |pos, i| {
        result += contractedNuclearAttraction(
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
fn numCartesianUpTo(comptime l: usize) usize {
    return (l + 1) * (l + 2) * (l + 3) / 6;
}

/// Maximum number of Cartesian indices for one side of the vertical table.
const ERI_MAX_CART: usize = numCartesianUpTo(ERI_MAX_AM - 1); // 84 for L=6

/// Map (ax,ay,az) to a linear index. Requires ax+ay+az <= ERI_MAX_AM-1.
/// Uses a simple 3D layout: idx = az + (ERI_MAX_AM)*(ay + ERI_MAX_AM*ax).
fn eriCartIndex(ax: usize, ay: usize, az: usize) usize {
    return az + ERI_MAX_AM * (ay + ERI_MAX_AM * ax);
}

/// Total size of the 3D Cartesian index space.
const ERI_CART_STRIDE: usize = ERI_MAX_AM * ERI_MAX_AM * ERI_MAX_AM; // 343

/// Vertical recurrence table type.
/// theta[idx_a][idx_c][m] = [a,0|c,0]^(m)
/// where idx_a = eriCartIndex(ax,ay,az), idx_c = eriCartIndex(cx,cy,cz).
///
/// We use a flat array: theta[idx_a * ERI_CART_STRIDE * ERI_MAX_M + idx_c * ERI_MAX_M + m]
const THETA_SIZE: usize = ERI_CART_STRIDE * ERI_CART_STRIDE * ERI_MAX_M;

/// Compute ERI between four primitive Cartesian Gaussians using table-based Obara-Saika.
///
/// (ab|cd) = <ab|1/r₁₂|cd>
///
/// Algorithm:
///   1. Compute intermediate quantities (P, Q, W, Boys values, etc.)
///   2. Build vertical recurrence table theta[a][c][m] = [a,0|c,0]^(m) bottom-up
///   3. Apply horizontal recurrence to recover [a,b|c,d]
pub fn primitiveERI(
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
    const p = alpha + beta;
    const q = gamma + delta;
    const rho = p * q / (p + q);

    // Gaussian product centers
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

    const diff_ab = math.Vec3.sub(center_a, center_b);
    const diff_cd = math.Vec3.sub(center_c, center_d);
    const diff_pq = math.Vec3.sub(p_center, q_center);

    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);
    const r2_cd = math.Vec3.dot(diff_cd, diff_cd);
    const r2_pq = math.Vec3.dot(diff_pq, diff_pq);

    const mu_ab = alpha * beta / p;
    const mu_cd = gamma * delta / q;

    const exp_factor = @exp(-mu_ab * r2_ab - mu_cd * r2_cd);
    const two_pi_2p5 = 2.0 * std.math.pow(f64, std.math.pi, 2.5);
    const prefactor = two_pi_2p5 / (p * q * @sqrt(p + q)) * exp_factor;

    // Boys function argument
    const arg = rho * r2_pq;

    const la = a.x + a.y + a.z;
    const lb = b.x + b.y + b.z;
    const lc = c_am.x + c_am.y + c_am.z;
    const ld = d_am.x + d_am.y + d_am.z;
    const total_am = la + lb + lc + ld;

    // Pre-compute Boys function values F_m(T) for m = 0..total_am
    var boys: [ERI_MAX_M]f64 = undefined;
    boys_mod.boysBatch(@as(u32, @intCast(total_am)), arg, &boys);

    // Intermediate vectors
    const pa = [3]f64{ p_center.x - center_a.x, p_center.y - center_a.y, p_center.z - center_a.z };
    const qc = [3]f64{ q_center.x - center_c.x, q_center.y - center_c.y, q_center.z - center_c.z };
    const wp = [3]f64{ w_center.x - p_center.x, w_center.y - p_center.y, w_center.z - p_center.z };
    const wq = [3]f64{ w_center.x - q_center.x, w_center.y - q_center.y, w_center.z - q_center.z };
    const ab = [3]f64{ center_a.x - center_b.x, center_a.y - center_b.y, center_a.z - center_b.z };
    const cd = [3]f64{ center_c.x - center_d.x, center_c.y - center_d.y, center_c.z - center_d.z };

    const inv_2p = 0.5 / p;
    const inv_2q = 0.5 / q;
    const inv_2pq = 0.5 / (p + q);
    const rho_over_p = rho / p;
    const rho_over_q = rho / q;

    // La = la + lb, Lc = lc + ld: maximum angular momentum on each side
    // after horizontal transfer moves b→a and d→c.
    const La: usize = la + lb;
    const Lc: usize = lc + ld;
    const m_max: usize = La + Lc;

    // ---------------------------------------------------------------
    // Step 1: Build vertical recurrence table [a,0|c,0]^(m) bottom-up
    // ---------------------------------------------------------------
    // We allocate on the stack. The table is indexed as:
    // theta[eriCartIndex(ax,ay,az)][eriCartIndex(cx,cy,cz)][m]
    // but stored flat as theta_flat[(idx_a * cart_c_stride + idx_c) * m_stride + m].
    //
    // For compactness, we only allocate entries needed: ax+ay+az <= La, cx+cy+cz <= Lc, m <= m_max.

    // Use a flat heap-allocated table would be cleaner, but for performance we use
    // a comptime-sized stack buffer. The maximum size is ERI_CART_STRIDE^2 * ERI_MAX_M.
    // For f+f|f+f: 343^2 * 13 ≈ 1.5M entries × 8 bytes = 12MB — too large for stack.
    //
    // Instead, we use a more compact indexing where each axis is bounded by
    // its actual range, not the global maximum. But that makes indexing complex.
    //
    // Pragmatic approach: use the simple 3D indexing but note that most entries
    // are never touched. With (La+1)^3 * (Lc+1)^3 * (m_max+1) actual entries:
    // f+f|f+f: 7^3 * 7^3 * 13 = 343 * 343 * 13 = 1,529,437 → 12MB (too big for stack)
    //
    // For stack-friendly: limit each axis to La+1 and Lc+1 by using strides La+1 and Lc+1.
    const a_stride_z: usize = 1;
    const a_stride_y: usize = (La + 1) * a_stride_z;
    const a_stride_x: usize = (La + 1) * a_stride_y;
    const a_size: usize = (La + 1) * a_stride_x;

    const c_stride_z: usize = 1;
    const c_stride_y: usize = (Lc + 1) * c_stride_z;
    const c_stride_x: usize = (Lc + 1) * c_stride_y;
    const c_size: usize = (Lc + 1) * c_stride_x;

    const m_stride: usize = m_max + 1;
    const theta_size = a_size * c_size * m_stride;

    // Stack allocation: worst case f+f|f+f = 343 * 343 * 13 ≈ 1.5M × 8 = 12MB.
    // This may exceed default stack. Use a static thread-local buffer instead.
    // Actually, for practical cases (up to d+p or so), this is much smaller.
    // For safety, we cap at a reasonable stack size and fall back to a simpler approach.
    const MAX_STACK_THETA: usize = 256 * 1024; // 256K entries = 2MB
    if (theta_size > MAX_STACK_THETA) {
        // Fall back to recursive implementation for very large angular momentum.
        const fallback = primitiveERIRecursive(
            a,
            b,
            c_am,
            d_am,
            &boys,
            pa,
            qc,
            wp,
            wq,
            ab,
            cd,
            inv_2p,
            inv_2q,
            inv_2pq,
            rho_over_p,
            rho_over_q,
        );
        return fallback * prefactor;
    }

    var theta: [MAX_STACK_THETA]f64 = undefined;
    // Zero the used portion
    @memset(theta[0..theta_size], 0.0);

    // Base case: [0,0,0 | 0,0,0]^(m) = boys[m]
    for (0..m_max + 1) |m| {
        theta[0 * c_size * m_stride + 0 * m_stride + m] = boys[m];
    }

    // Build c-direction first (increment c while a=0):
    // [0,0|c+1_i,0]^(m) = QC_i [0,0|c,0]^(m) + WQ_i [0,0|c,0]^(m+1)
    //     + c_i/(2q) {[0,0|c-1_i,0]^(m) - ρ/q [0,0|c-1_i,0]^(m+1)}
    for (1..Lc + 1) |lc_total| {
        // Iterate over all (cx,cy,cz) with cx+cy+cz = lc_total
        var cx: usize = lc_total;
        while (true) {
            var cy: usize = lc_total - cx;
            while (true) {
                const cz: usize = lc_total - cx - cy;
                const c_idx = cx * c_stride_x + cy * c_stride_y + cz * c_stride_z;
                const a_idx: usize = 0; // a = (0,0,0)

                // Find which axis to decrement: pick first non-zero
                var axis: usize = undefined;
                if (cx > 0) {
                    axis = 0;
                } else if (cy > 0) {
                    axis = 1;
                } else {
                    axis = 2;
                }

                // c_dec = c - e_{axis}
                var cx_d = cx;
                var cy_d = cy;
                var cz_d = cz;
                if (axis == 0) cx_d -= 1 else if (axis == 1) cy_d -= 1 else cz_d -= 1;
                const c_dec_idx = cx_d * c_stride_x + cy_d * c_stride_y + cz_d * c_stride_z;

                // c_dec[axis] value (after decrement)
                const ci_after: f64 = @floatFromInt(
                    if (axis == 0) cx_d else if (axis == 1) cy_d else cz_d,
                );

                const base_dec = a_idx * c_size * m_stride + c_dec_idx * m_stride;
                const base_out = a_idx * c_size * m_stride + c_idx * m_stride;

                for (0..m_max + 1 - lc_total) |m| {
                    var val = qc[axis] * theta[base_dec + m] +
                        wq[axis] * theta[base_dec + m + 1];

                    if (ci_after >= 1.0) {
                        // c_dec2 = c_dec - e_{axis}
                        var cx_d2 = cx_d;
                        var cy_d2 = cy_d;
                        var cz_d2 = cz_d;
                        if (axis == 0) cx_d2 -= 1 else if (axis == 1) cy_d2 -= 1 else cz_d2 -= 1;
                        const c_dec2_idx =
                            cx_d2 * c_stride_x + cy_d2 * c_stride_y + cz_d2 * c_stride_z;
                        const base_dec2 = a_idx * c_size * m_stride + c_dec2_idx * m_stride;

                        val += ci_after * inv_2q * (theta[base_dec2 + m] -
                            rho_over_q * theta[base_dec2 + m + 1]);
                    }

                    theta[base_out + m] = val;
                }

                if (cy == 0) break;
                cy -= 1;
            }
            if (cx == 0) break;
            cx -= 1;
        }
    }

    // Build a-direction (increment a for all c values):
    // [a+1_i,0|c,0]^(m) = PA_i [a,0|c,0]^(m) + WP_i [a,0|c,0]^(m+1)
    //     + a_i/(2p) {[a-1_i,0|c,0]^(m) - ρ/p [a-1_i,0|c,0]^(m+1)}
    //     + c_i/(2(p+q)) [a,0|c-1_i,0]^(m+1)
    for (1..La + 1) |la_total| {
        var ax: usize = la_total;
        while (true) {
            var ay: usize = la_total - ax;
            while (true) {
                const az: usize = la_total - ax - ay;
                const a_idx = ax * a_stride_x + ay * a_stride_y + az * a_stride_z;

                // Find which axis to decrement
                var axis: usize = undefined;
                if (ax > 0) {
                    axis = 0;
                } else if (ay > 0) {
                    axis = 1;
                } else {
                    axis = 2;
                }

                // a_dec = a - e_{axis}
                var ax_d = ax;
                var ay_d = ay;
                var az_d = az;
                if (axis == 0) ax_d -= 1 else if (axis == 1) ay_d -= 1 else az_d -= 1;
                const a_dec_idx = ax_d * a_stride_x + ay_d * a_stride_y + az_d * a_stride_z;

                const ai_after: f64 = @floatFromInt(
                    if (axis == 0) ax_d else if (axis == 1) ay_d else az_d,
                );

                // For all c indices with cx+cy+cz <= Lc
                for (0..Lc + 1) |lc_total| {
                    var cx2: usize = lc_total;
                    while (true) {
                        var cy2: usize = lc_total - cx2;
                        while (true) {
                            const cz2: usize = lc_total - cx2 - cy2;
                            const c_idx =
                                cx2 * c_stride_x + cy2 * c_stride_y + cz2 * c_stride_z;

                            const base_dec = a_dec_idx * c_size * m_stride + c_idx * m_stride;
                            const base_out = a_idx * c_size * m_stride + c_idx * m_stride;

                            const m_limit = m_max + 1 - la_total - lc_total;
                            for (0..m_limit) |m| {
                                var val = pa[axis] * theta[base_dec + m] +
                                    wp[axis] * theta[base_dec + m + 1];

                                if (ai_after >= 1.0) {
                                    var ax_d2 = ax_d;
                                    var ay_d2 = ay_d;
                                    var az_d2 = az_d;
                                    if (axis == 0)
                                        ax_d2 -= 1
                                    else if (axis == 1) ay_d2 -= 1 else az_d2 -= 1;
                                    const a_dec2_idx = ax_d2 * a_stride_x +
                                        ay_d2 * a_stride_y +
                                        az_d2 * a_stride_z;
                                    const base_dec2 =
                                        a_dec2_idx * c_size * m_stride + c_idx * m_stride;

                                    val += ai_after * inv_2p * (theta[base_dec2 + m] -
                                        rho_over_p * theta[base_dec2 + m + 1]);
                                }

                                // Coupling term: c_i/(2(p+q)) [a_dec,0|c-1_i,0]^(m+1)
                                const ci_val: f64 = @floatFromInt(
                                    if (axis == 0) cx2 else if (axis == 1) cy2 else cz2,
                                );
                                if (ci_val >= 1.0) {
                                    var cx2_d = cx2;
                                    var cy2_d = cy2;
                                    var cz2_d = cz2;
                                    if (axis == 0)
                                        cx2_d -= 1
                                    else if (axis == 1) cy2_d -= 1 else cz2_d -= 1;
                                    const c_dec_idx = cx2_d * c_stride_x +
                                        cy2_d * c_stride_y +
                                        cz2_d * c_stride_z;
                                    const coupling_base =
                                        a_dec_idx * c_size * m_stride + c_dec_idx * m_stride;
                                    const coupling_idx = coupling_base + m + 1;

                                    val += ci_val * inv_2pq * theta[coupling_idx];
                                }

                                theta[base_out + m] = val;
                            }

                            if (cy2 == 0) break;
                            cy2 -= 1;
                        }
                        if (cx2 == 0) break;
                        cx2 -= 1;
                    }
                }

                if (ay == 0) break;
                ay -= 1;
            }
            if (ax == 0) break;
            ax -= 1;
        }
    }

    // ---------------------------------------------------------------
    // Step 2: Horizontal recurrence [a,b|c,d] from theta = [a',0|c',0]^(0)
    // ---------------------------------------------------------------
    // [a, b+1_i | c, d] = [a+1_i, b | c, d] + (A_i - B_i) [a, b | c, d]
    // [a, b | c, d+1_i] = [a, b | c+1_i, d] + (C_i - D_i) [a, b | c, d]
    //
    // We apply horizontal in two stages:
    // Stage A: Build [a',0|c',d]^(0) for target d from theta[a'][c'][0].
    //   [a',0|c'+1_i,d-1_i]^(0) + (C_i-D_i) [a',0|c',d-1_i]^(0) = [a',0|c',d]^(0)
    //   Start from d=0: [a',0|c',0] = theta[a'][c'][0].
    //   Then increment d: for each component of d.
    // Stage B: Build [a,b|c,d] from [a',0|c,d].
    //   [a'+1_i,b-1_i|c,d] + (A_i-B_i) [a',b-1_i|c,d] = [a',b|c,d]
    //
    // For a single (a,b,c_am,d_am) we can use recursion with the theta table.
    // The recursion is now cheap because theta lookups are O(1).

    const result = eriHorizontal(
        .{ a.x, a.y, a.z },
        .{ b.x, b.y, b.z },
        .{ c_am.x, c_am.y, c_am.z },
        .{ d_am.x, d_am.y, d_am.z },
        &theta,
        a_stride_x,
        a_stride_y,
        c_size,
        c_stride_x,
        c_stride_y,
        m_stride,
        ab,
        cd,
    );

    return prefactor * result;
}

/// Horizontal recurrence using the pre-built vertical table.
/// This is still recursive but only in b and d (which are small: l <= 3),
/// and each call does O(1) work (table lookup), so the total cost is
/// O(2^(lb+ld)) which is at most 2^6 = 64 for f+f.
fn eriHorizontal(
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
    // Transfer d → c first
    for (0..3) |i| {
        if (d_arr[i] > 0) {
            var d_dec = d_arr;
            d_dec[i] -= 1;
            var c_inc = c_arr;
            c_inc[i] += 1;
            const t1 = eriHorizontal(
                a_arr,
                b_arr,
                c_inc,
                d_dec,
                theta,
                a_stride_x,
                a_stride_y,
                c_size,
                c_stride_x,
                c_stride_y,
                m_stride,
                ab,
                cd_vec,
            );
            const t2 = eriHorizontal(
                a_arr,
                b_arr,
                c_arr,
                d_dec,
                theta,
                a_stride_x,
                a_stride_y,
                c_size,
                c_stride_x,
                c_stride_y,
                m_stride,
                ab,
                cd_vec,
            );
            return t1 + cd_vec[i] * t2;
        }
    }

    // Transfer b → a
    for (0..3) |i| {
        if (b_arr[i] > 0) {
            var b_dec = b_arr;
            b_dec[i] -= 1;
            var a_inc = a_arr;
            a_inc[i] += 1;
            const t1 = eriHorizontal(
                a_inc,
                b_dec,
                c_arr,
                d_arr,
                theta,
                a_stride_x,
                a_stride_y,
                c_size,
                c_stride_x,
                c_stride_y,
                m_stride,
                ab,
                cd_vec,
            );
            const t2 = eriHorizontal(
                a_arr,
                b_dec,
                c_arr,
                d_arr,
                theta,
                a_stride_x,
                a_stride_y,
                c_size,
                c_stride_x,
                c_stride_y,
                m_stride,
                ab,
                cd_vec,
            );
            return t1 + ab[i] * t2;
        }
    }

    // b = d = (0,0,0): look up from theta table at m=0
    const a_idx = @as(usize, a_arr[0]) * a_stride_x +
        @as(usize, a_arr[1]) * a_stride_y +
        @as(usize, a_arr[2]);
    const c_idx = @as(usize, c_arr[0]) * c_stride_x +
        @as(usize, c_arr[1]) * c_stride_y +
        @as(usize, c_arr[2]);
    return theta[a_idx * c_size * m_stride + c_idx * m_stride + 0];
}

/// Fallback recursive ERI for very large angular momentum (exceeds stack table size).
fn primitiveERIRecursive(
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
    return eriRecursive(
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
fn eriRecursive(
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
            return eriRecursive(a, b, c_inc, d_dec, m, params) +
                cd_i * eriRecursive(a, b, c, d_dec, m, params);
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
            return eriRecursive(a_inc, b_dec, c, d, m, params) +
                ab_i * eriRecursive(a, b_dec, c, d, m, params);
        }
    }

    // b = d = 0: vertical recurrence
    return eriVertical(a, c, m, params);
}

/// Vertical recurrence for [a,0|c,0]^(m) (fallback recursive version).
fn eriVertical(
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

        var result = params.pa[axis] * eriVertical(a_dec, c, m, params) +
            params.wp[axis] * eriVertical(a_dec, c, m + 1, params);

        if (a_dec[axis] >= 1) {
            var a_dec2 = a_dec;
            a_dec2[axis] -= 1;
            const ai = @as(f64, @floatFromInt(a_dec[axis]));
            result += ai * params.inv_2p * (eriVertical(a_dec2, c, m, params) -
                params.rho_over_p * eriVertical(a_dec2, c, m + 1, params));
        }

        if (c[axis] >= 1) {
            var c_dec = c;
            c_dec[axis] -= 1;
            const ci = @as(f64, @floatFromInt(c[axis]));
            result += ci * params.inv_2pq * eriVertical(a_dec, c_dec, m + 1, params);
        }

        return result;
    } else {
        var c_dec = c;
        c_dec[axis] -= 1;

        var result = params.qc[axis] * eriVertical(a, c_dec, m, params) +
            params.wq[axis] * eriVertical(a, c_dec, m + 1, params);

        if (c_dec[axis] >= 1) {
            var c_dec2 = c_dec;
            c_dec2[axis] -= 1;
            const ci = @as(f64, @floatFromInt(c_dec[axis]));
            result += ci * params.inv_2q * (eriVertical(a, c_dec2, m, params) -
                params.rho_over_q * eriVertical(a, c_dec2, m + 1, params));
        }

        return result;
    }
}

/// Compute contracted ERI between four shells for specific Cartesian components.
pub fn contractedERI(
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

                    const prim = primitiveERI(
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
pub fn contractedShellQuartetERI(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    output: []f64,
) usize {
    const la = shell_a.l;
    const lb = shell_b.l;
    const lc = shell_c.l;
    const ld = shell_d.l;

    const na = basis_mod.numCartesian(la);
    const nb = basis_mod.numCartesian(lb);
    const nc = basis_mod.numCartesian(lc);
    const nd = basis_mod.numCartesian(ld);
    const total_out = na * nb * nc * nd;

    std.debug.assert(output.len >= total_out);

    // Zero output buffer
    @memset(output[0..total_out], 0.0);

    const cart_a = basis_mod.cartesianExponents(la);
    const cart_b = basis_mod.cartesianExponents(lb);
    const cart_c = basis_mod.cartesianExponents(lc);
    const cart_d = basis_mod.cartesianExponents(ld);

    // Pre-compute normalization tables for all primitives × Cartesian components
    // This avoids recomputing normalization inside the innermost loops.
    var norm_a: [MAX_NORM_TABLE]f64 = undefined;
    var norm_b: [MAX_NORM_TABLE]f64 = undefined;
    var norm_c: [MAX_NORM_TABLE]f64 = undefined;
    var norm_d: [MAX_NORM_TABLE]f64 = undefined;

    // Track max normalization per primitive for screening
    var max_norm_a: [MAX_PRIM]f64 = undefined;
    var max_norm_b: [MAX_PRIM]f64 = undefined;
    var max_norm_c: [MAX_PRIM]f64 = undefined;
    var max_norm_d: [MAX_PRIM]f64 = undefined;

    for (shell_a.primitives, 0..) |pa, ip| {
        var mx: f64 = 0.0;
        for (0..na) |ic| {
            const ca = cart_a[ic];
            const n_val = basis_mod.normalization(pa.alpha, ca.x, ca.y, ca.z);
            norm_a[ip * na + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_a[ip] = mx;
    }
    for (shell_b.primitives, 0..) |pb, ip| {
        var mx: f64 = 0.0;
        for (0..nb) |ic| {
            const cb = cart_b[ic];
            const n_val = basis_mod.normalization(pb.alpha, cb.x, cb.y, cb.z);
            norm_b[ip * nb + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_b[ip] = mx;
    }
    for (shell_c.primitives, 0..) |pc, ip| {
        var mx: f64 = 0.0;
        for (0..nc) |ic| {
            const cc = cart_c[ic];
            const n_val = basis_mod.normalization(pc.alpha, cc.x, cc.y, cc.z);
            norm_c[ip * nc + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_c[ip] = mx;
    }
    for (shell_d.primitives, 0..) |pd, ip| {
        var mx: f64 = 0.0;
        for (0..nd) |ic| {
            const cd_el = cart_d[ic];
            const n_val = basis_mod.normalization(pd.alpha, cd_el.x, cd_el.y, cd_el.z);
            norm_d[ip * nd + ic] = n_val;
            if (n_val > mx) mx = n_val;
        }
        max_norm_d[ip] = mx;
    }

    // Maximum total angular momentum for this shell quartet
    const La: usize = la + lb;
    const Lc: usize = lc + ld;
    const m_max: usize = La + Lc;

    // Compute theta table strides (same as in primitiveERI)
    const a_stride_z: usize = 1;
    const a_stride_y: usize = (La + 1) * a_stride_z;
    const a_stride_x: usize = (La + 1) * a_stride_y;
    const a_size: usize = (La + 1) * a_stride_x;

    const c_stride_z: usize = 1;
    const c_stride_y: usize = (Lc + 1) * c_stride_z;
    const c_stride_x: usize = (Lc + 1) * c_stride_y;
    const c_size: usize = (Lc + 1) * c_stride_x;

    const m_stride: usize = m_max + 1;
    const theta_size = a_size * c_size * m_stride;

    // Check stack size limit
    const MAX_STACK_THETA_BATCH: usize = 256 * 1024;
    if (theta_size > MAX_STACK_THETA_BATCH) {
        // Fall back to per-integral computation for very large angular momentum
        for (0..na) |ia| {
            for (0..nb) |ib| {
                for (0..nc) |ic| {
                    for (0..nd) |id| {
                        output[ia * nb * nc * nd + ib * nc * nd + ic * nd + id] = contractedERI(
                            shell_a,
                            cart_a[ia],
                            shell_b,
                            cart_b[ib],
                            shell_c,
                            cart_c[ic],
                            shell_d,
                            cart_d[id],
                        );
                    }
                }
            }
        }
        return total_out;
    }

    // AB and CD vectors for horizontal recurrence
    const ab = [3]f64{
        shell_a.center.x - shell_b.center.x,
        shell_a.center.y - shell_b.center.y,
        shell_a.center.z - shell_b.center.z,
    };
    const cd_vec = [3]f64{
        shell_c.center.x - shell_d.center.x,
        shell_c.center.y - shell_d.center.y,
        shell_c.center.z - shell_d.center.z,
    };

    var theta: [MAX_STACK_THETA_BATCH]f64 = undefined;

    // Pre-compute AB and CD distances (constant for all primitives in this shell quartet)
    const diff_ab = math.Vec3.sub(shell_a.center, shell_b.center);
    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);
    const diff_cd = math.Vec3.sub(shell_c.center, shell_d.center);
    const r2_cd = math.Vec3.dot(diff_cd, diff_cd);

    // Primitive screening threshold: skip quartets with negligible contribution.
    // The contribution scales as
    //   prefactor * coeff * norm
    //     ~ exp(-mu_ab*r2_ab - mu_cd*r2_cd) * pi^2.5 / (p*q*sqrt(p+q))
    // We use a conservative threshold on the exponential decay factor.
    const prim_screen_threshold: f64 = 1e-15;

    // Loop over all primitive quartets
    for (shell_a.primitives, 0..) |prim_a, ipa| {
        const alpha = prim_a.alpha;
        for (shell_b.primitives, 0..) |prim_b, ipb| {
            const beta = prim_b.alpha;
            const p_val = alpha + beta;
            const mu_ab = alpha * beta / p_val;

            const exp_ab = @exp(-mu_ab * r2_ab);

            const coeff_ab = prim_a.coeff * prim_b.coeff;

            // Gaussian product center P
            const p_center = math.Vec3{
                .x = (alpha * shell_a.center.x + beta * shell_b.center.x) / p_val,
                .y = (alpha * shell_a.center.y + beta * shell_b.center.y) / p_val,
                .z = (alpha * shell_a.center.z + beta * shell_b.center.z) / p_val,
            };

            // PA vector
            const pa_vec = [3]f64{
                p_center.x - shell_a.center.x,
                p_center.y - shell_a.center.y,
                p_center.z - shell_a.center.z,
            };

            const inv_2p = 0.5 / p_val;

            for (shell_c.primitives, 0..) |prim_c, ipc| {
                const gamma_val = prim_c.alpha;
                for (shell_d.primitives, 0..) |prim_d, ipd| {
                    const delta_val = prim_d.alpha;
                    const q_val = gamma_val + delta_val;
                    const mu_cd = gamma_val * delta_val / q_val;

                    // --- Primitive screening ---
                    const exp_cd = @exp(-mu_cd * r2_cd);
                    const exp_factor = exp_ab * exp_cd;
                    const coeff_abcd = coeff_ab * prim_c.coeff * prim_d.coeff;

                    // Upper bound: prefactor ~ 2 * pi^2.5 / (p*q*sqrt(p+q)) * exp_factor
                    // Include max normalization constants for accurate screening
                    const two_pi_2p5 = 2.0 * std.math.pow(f64, std.math.pi, 2.5);
                    const prefactor_bound =
                        two_pi_2p5 / (p_val * q_val * @sqrt(p_val + q_val)) * exp_factor;
                    const max_norm_product = max_norm_a[ipa] * max_norm_b[ipb] *
                        max_norm_c[ipc] * max_norm_d[ipd];
                    if (@abs(coeff_abcd) * max_norm_product * prefactor_bound <
                        prim_screen_threshold) continue;

                    const rho = p_val * q_val / (p_val + q_val);
                    const prefactor = prefactor_bound; // Same expression, already computed

                    // Gaussian product center Q
                    const q_center = math.Vec3{
                        .x = (gamma_val * shell_c.center.x + delta_val * shell_d.center.x) / q_val,
                        .y = (gamma_val * shell_c.center.y + delta_val * shell_d.center.y) / q_val,
                        .z = (gamma_val * shell_c.center.z + delta_val * shell_d.center.z) / q_val,
                    };
                    const w_center = math.Vec3{
                        .x = (p_val * p_center.x + q_val * q_center.x) / (p_val + q_val),
                        .y = (p_val * p_center.y + q_val * q_center.y) / (p_val + q_val),
                        .z = (p_val * p_center.z + q_val * q_center.z) / (p_val + q_val),
                    };

                    const diff_pq = math.Vec3.sub(p_center, q_center);
                    const r2_pq = math.Vec3.dot(diff_pq, diff_pq);

                    // Boys function argument
                    const arg = rho * r2_pq;

                    // Pre-compute Boys function values using batch computation
                    var boys: [ERI_MAX_M]f64 = undefined;
                    boys_mod.boysBatch(@as(u32, @intCast(m_max)), arg, &boys);

                    // Intermediate vectors
                    const qc = [3]f64{
                        q_center.x - shell_c.center.x,
                        q_center.y - shell_c.center.y,
                        q_center.z - shell_c.center.z,
                    };
                    const wp = [3]f64{
                        w_center.x - p_center.x,
                        w_center.y - p_center.y,
                        w_center.z - p_center.z,
                    };
                    const wq = [3]f64{
                        w_center.x - q_center.x,
                        w_center.y - q_center.y,
                        w_center.z - q_center.z,
                    };

                    const inv_2q = 0.5 / q_val;
                    const inv_2pq = 0.5 / (p_val + q_val);
                    const rho_over_p = rho / p_val;
                    const rho_over_q = rho / q_val;

                    // ---- Build theta table (vertical recurrence) ----
                    // Note: No @memset needed. The vertical recurrence writes all entries via
                    // direct assignment (=), not accumulation (+=). Each theta entry is written
                    // before it is read. This avoids a costly memset on large theta buffers.

                    // Base case: [0,0,0 | 0,0,0]^(m) = boys[m]
                    for (0..m_max + 1) |m| {
                        theta[0 * c_size * m_stride + 0 * m_stride + m] = boys[m];
                    }

                    // Build c-direction (increment c while a=0)
                    for (1..Lc + 1) |lc_total| {
                        var cx: usize = lc_total;
                        while (true) {
                            var cy: usize = lc_total - cx;
                            while (true) {
                                const cz: usize = lc_total - cx - cy;
                                const c_idx = cx * c_stride_x + cy * c_stride_y + cz * c_stride_z;
                                const a_idx: usize = 0;

                                var axis: usize = undefined;
                                if (cx > 0) {
                                    axis = 0;
                                } else if (cy > 0) {
                                    axis = 1;
                                } else {
                                    axis = 2;
                                }

                                var cx_d = cx;
                                var cy_d = cy;
                                var cz_d = cz;
                                if (axis == 0)
                                    cx_d -= 1
                                else if (axis == 1) cy_d -= 1 else cz_d -= 1;
                                const c_dec_idx = cx_d * c_stride_x +
                                    cy_d * c_stride_y +
                                    cz_d * c_stride_z;
                                const ci_after: f64 = @floatFromInt(
                                    if (axis == 0) cx_d else if (axis == 1) cy_d else cz_d,
                                );

                                const base_dec =
                                    a_idx * c_size * m_stride + c_dec_idx * m_stride;
                                const base_out =
                                    a_idx * c_size * m_stride + c_idx * m_stride;

                                for (0..m_max + 1 - lc_total) |m| {
                                    var val = qc[axis] * theta[base_dec + m] +
                                        wq[axis] * theta[base_dec + m + 1];

                                    if (ci_after >= 1.0) {
                                        var cx_d2 = cx_d;
                                        var cy_d2 = cy_d;
                                        var cz_d2 = cz_d;
                                        if (axis == 0)
                                            cx_d2 -= 1
                                        else if (axis == 1) cy_d2 -= 1 else cz_d2 -= 1;
                                        const c_dec2_idx = cx_d2 * c_stride_x +
                                            cy_d2 * c_stride_y +
                                            cz_d2 * c_stride_z;
                                        const base_dec2 = a_idx * c_size * m_stride +
                                            c_dec2_idx * m_stride;

                                        val += ci_after * inv_2q * (theta[base_dec2 + m] -
                                            rho_over_q * theta[base_dec2 + m + 1]);
                                    }

                                    theta[base_out + m] = val;
                                }

                                if (cy == 0) break;
                                cy -= 1;
                            }
                            if (cx == 0) break;
                            cx -= 1;
                        }
                    }

                    // Build a-direction (increment a for all c values)
                    for (1..La + 1) |la_total| {
                        var ax: usize = la_total;
                        while (true) {
                            var ay: usize = la_total - ax;
                            while (true) {
                                const az: usize = la_total - ax - ay;
                                const a_idx = ax * a_stride_x + ay * a_stride_y + az * a_stride_z;

                                var axis: usize = undefined;
                                if (ax > 0) {
                                    axis = 0;
                                } else if (ay > 0) {
                                    axis = 1;
                                } else {
                                    axis = 2;
                                }

                                var ax_d = ax;
                                var ay_d = ay;
                                var az_d = az;
                                if (axis == 0)
                                    ax_d -= 1
                                else if (axis == 1) ay_d -= 1 else az_d -= 1;
                                const a_dec_idx = ax_d * a_stride_x +
                                    ay_d * a_stride_y +
                                    az_d * a_stride_z;
                                const ai_after: f64 = @floatFromInt(
                                    if (axis == 0) ax_d else if (axis == 1) ay_d else az_d,
                                );

                                for (0..Lc + 1) |lc_total| {
                                    var cx2: usize = lc_total;
                                    while (true) {
                                        var cy2: usize = lc_total - cx2;
                                        while (true) {
                                            const cz2: usize = lc_total - cx2 - cy2;
                                            const c_idx = cx2 * c_stride_x +
                                                cy2 * c_stride_y +
                                                cz2 * c_stride_z;

                                            const base_dec = a_dec_idx * c_size * m_stride +
                                                c_idx * m_stride;
                                            const base_out = a_idx * c_size * m_stride +
                                                c_idx * m_stride;

                                            const m_limit = m_max + 1 - la_total - lc_total;
                                            for (0..m_limit) |m| {
                                                var val = pa_vec[axis] * theta[base_dec + m] +
                                                    wp[axis] * theta[base_dec + m + 1];

                                                if (ai_after >= 1.0) {
                                                    var ax_d2 = ax_d;
                                                    var ay_d2 = ay_d;
                                                    var az_d2 = az_d;
                                                    if (axis == 0)
                                                        ax_d2 -= 1
                                                    else if (axis == 1) ay_d2 -= 1 else az_d2 -= 1;
                                                    const a_dec2_idx = ax_d2 * a_stride_x +
                                                        ay_d2 * a_stride_y +
                                                        az_d2 * a_stride_z;
                                                    const base_dec2 =
                                                        a_dec2_idx * c_size * m_stride +
                                                        c_idx * m_stride;

                                                    val += ai_after * inv_2p *
                                                        (theta[base_dec2 + m] -
                                                            rho_over_p * theta[base_dec2 + m + 1]);
                                                }

                                                const ci_val: f64 = @floatFromInt(
                                                    if (axis == 0)
                                                        cx2
                                                    else if (axis == 1) cy2 else cz2,
                                                );
                                                if (ci_val >= 1.0) {
                                                    var cx2_d = cx2;
                                                    var cy2_d = cy2;
                                                    var cz2_d = cz2;
                                                    if (axis == 0)
                                                        cx2_d -= 1
                                                    else if (axis == 1) cy2_d -= 1 else cz2_d -= 1;
                                                    const c_dec_idx = cx2_d * c_stride_x +
                                                        cy2_d * c_stride_y +
                                                        cz2_d * c_stride_z;
                                                    const coupling_base =
                                                        a_dec_idx * c_size * m_stride +
                                                        c_dec_idx * m_stride;

                                                    val += ci_val * inv_2pq *
                                                        theta[coupling_base + m + 1];
                                                }

                                                theta[base_out + m] = val;
                                            }

                                            if (cy2 == 0) break;
                                            cy2 -= 1;
                                        }
                                        if (cx2 == 0) break;
                                        cx2 -= 1;
                                    }
                                }

                                if (ay == 0) break;
                                ay -= 1;
                            }
                            if (ax == 0) break;
                            ax -= 1;
                        }
                    }

                    // ---- Horizontal recurrence for ALL Cartesian component combinations ----
                    // The theta table is built; now extract (ia,ib|ic,id) for all combinations.
                    const prim_prefactor = prefactor * coeff_abcd;

                    for (0..na) |ia| {
                        const na_val = norm_a[ipa * na + ia];
                        for (0..nb) |ib| {
                            const nb_val = norm_b[ipb * nb + ib];
                            const nab = na_val * nb_val;
                            for (0..nc) |ic| {
                                const nc_val = norm_c[ipc * nc + ic];
                                const nabc = nab * nc_val;
                                for (0..nd) |id| {
                                    const nd_val = norm_d[ipd * nd + id];

                                    const eri_val = eriHorizontal(
                                        .{ cart_a[ia].x, cart_a[ia].y, cart_a[ia].z },
                                        .{ cart_b[ib].x, cart_b[ib].y, cart_b[ib].z },
                                        .{ cart_c[ic].x, cart_c[ic].y, cart_c[ic].z },
                                        .{ cart_d[id].x, cart_d[id].y, cart_d[id].z },
                                        &theta,
                                        a_stride_x,
                                        a_stride_y,
                                        c_size,
                                        c_stride_x,
                                        c_stride_y,
                                        m_stride,
                                        ab,
                                        cd_vec,
                                    );

                                    output[ia * nb * nc * nd + ib * nc * nd + ic * nd + id] +=
                                        prim_prefactor * nabc * nd_val * eri_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return total_out;
}

// ============================================================================
// General Matrix Builders (working with shells of any angular momentum)
// ============================================================================

/// Build the full overlap matrix for a set of shells with arbitrary angular momentum.
/// Each shell contributes numCartesian(l) basis functions.
/// Returns a flat row-major N×N matrix where N = Σ numCartesian(shell.l).
pub fn buildOverlapMatrix(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
) ![]f64 {
    const n = totalBasisFunctions(shells);
    const mat = try alloc.alloc(f64, n * n);
    @memset(mat, 0.0);

    var mu: usize = 0;
    for (shells) |shell_a| {
        const cart_a = basis_mod.cartesianExponents(shell_a.l);
        const na = shell_a.numCartesianFunctions();
        var nu: usize = 0;
        for (shells) |shell_b| {
            const cart_b = basis_mod.cartesianExponents(shell_b.l);
            const nb = shell_b.numCartesianFunctions();

            for (0..na) |ia| {
                for (0..nb) |ib| {
                    const val = contractedOverlap(shell_a, cart_a[ia], shell_b, cart_b[ib]);
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
pub fn buildKineticMatrix(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
) ![]f64 {
    const n = totalBasisFunctions(shells);
    const mat = try alloc.alloc(f64, n * n);
    @memset(mat, 0.0);

    var mu: usize = 0;
    for (shells) |shell_a| {
        const cart_a = basis_mod.cartesianExponents(shell_a.l);
        const na = shell_a.numCartesianFunctions();
        var nu: usize = 0;
        for (shells) |shell_b| {
            const cart_b = basis_mod.cartesianExponents(shell_b.l);
            const nb = shell_b.numCartesianFunctions();

            for (0..na) |ia| {
                for (0..nb) |ib| {
                    const val = contractedKinetic(shell_a, cart_a[ia], shell_b, cart_b[ib]);
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
pub fn buildNuclearMatrix(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
) ![]f64 {
    const n = totalBasisFunctions(shells);
    const mat = try alloc.alloc(f64, n * n);
    @memset(mat, 0.0);

    var mu: usize = 0;
    for (shells) |shell_a| {
        const cart_a = basis_mod.cartesianExponents(shell_a.l);
        const na = shell_a.numCartesianFunctions();
        var nu: usize = 0;
        for (shells) |shell_b| {
            const cart_b = basis_mod.cartesianExponents(shell_b.l);
            const nb = shell_b.numCartesianFunctions();

            for (0..na) |ia| {
                for (0..nb) |ib| {
                    const val = contractedTotalNuclearAttraction(
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
/// (contractedShellQuartetERI) to share theta tables across Cartesian components.
/// Falls back to per-integral computation for shell quartets that exceed stack limits.
/// Returns a flat array indexed by compound index.
pub fn buildEriTable(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
) !GeneralEriTable {
    const n = totalBasisFunctions(shells);
    const nn = n * (n + 1) / 2;
    const size = nn * (nn + 1) / 2;
    const values = try alloc.alloc(f64, size);
    @memset(values, 0.0);

    const n_shells = shells.len;

    // Build shell → basis function index mapping
    var shell_offsets: [128]usize = undefined;
    var shell_sizes: [128]usize = undefined;
    var offset: usize = 0;
    for (shells, 0..) |shell, si| {
        shell_offsets[si] = offset;
        shell_sizes[si] = shell.numCartesianFunctions();
        offset += shell_sizes[si];
    }

    // --- Step 1: Build Schwarz screening table ---
    // Q_AB = max_{a in A, b in B} sqrt(|(ab|ab)|)
    // Stored as a flat n_shells × n_shells matrix (symmetric).
    // Uses Rys quadrature shell-quartet ERI for efficiency.
    const MAX_SCHWARZ_BATCH: usize = 15 * 15 * 15 * 15;
    var schwarz_buf: [MAX_SCHWARZ_BATCH]f64 = undefined;
    var schwarz_q: [128 * 128]f64 = undefined;
    for (0..n_shells) |si| {
        const na = shell_sizes[si];
        for (si..n_shells) |sj| {
            const nb = shell_sizes[sj];

            // Compute (AB|AB) shell quartet using Rys ERI
            _ = rys_eri.contractedShellQuartetERI(
                shells[si],
                shells[sj],
                shells[si],
                shells[sj],
                &schwarz_buf,
            );

            // Find max |ERI| over all Cartesian components
            var max_val: f64 = 0.0;
            for (0..na) |ia| {
                for (0..nb) |ib| {
                    const val = schwarz_buf[ia * nb * na * nb + ib * na * nb + ia * nb + ib];
                    const abs_val = @abs(val);
                    if (abs_val > max_val) max_val = abs_val;
                }
            }
            const q_val = @sqrt(max_val);
            schwarz_q[si * n_shells + sj] = q_val;
            schwarz_q[sj * n_shells + si] = q_val;
        }
    }

    // --- Step 2: Iterate shell quartets with Schwarz screening + batch ERI ---
    const schwarz_threshold: f64 = 1e-12;

    // Stack buffer for batch ERI output.
    // Maximum size: MAX_CART^4 = 15^4 = 50625 (f|f|f|f case).
    const MAX_BATCH: usize = 15 * 15 * 15 * 15;
    var eri_buf: [MAX_BATCH]f64 = undefined;

    // Shell-quartet loop with 8-fold shell-level symmetry:
    //   si >= sj, sk >= sl, shellPairIndex(si,sj) >= shellPairIndex(sk,sl)
    // Since ERI(ab|cd) = ERI(cd|ab), when distributing to the basis-function
    // level table we must handle both ij>=kl and ij<kl cases (swap if needed).
    for (0..n_shells) |si| {
        const na = shell_sizes[si];
        const off_a = shell_offsets[si];

        for (0..si + 1) |sj| {
            const nb = shell_sizes[sj];
            const off_b = shell_offsets[sj];
            const q_ab = schwarz_q[si * n_shells + sj];

            const ab_pair = shellPairIndex(si, sj);

            for (0..n_shells) |sk| {
                const nc = shell_sizes[sk];
                const off_c = shell_offsets[sk];

                for (0..sk + 1) |sl| {
                    const cd_pair = shellPairIndex(sk, sl);
                    if (ab_pair < cd_pair) continue; // bra-ket symmetry at shell level

                    const q_cd = schwarz_q[sk * n_shells + sl];
                    if (q_ab * q_cd < schwarz_threshold) {
                        continue; // Schwarz screening
                    }

                    const nd = shell_sizes[sl];
                    const off_d = shell_offsets[sl];

                    // Compute ALL ERIs for this shell quartet at once using Rys quadrature.
                    // This computes roots/weights once per primitive quartet and extracts
                    // all Cartesian component ERIs via 2D recurrence + horizontal recurrence.
                    _ = rys_eri.contractedShellQuartetERI(
                        shells[si],
                        shells[sj],
                        shells[sk],
                        shells[sl],
                        &eri_buf,
                    );

                    // Distribute batch ERIs to the triangular ERI table
                    for (0..na) |ia| {
                        const i = off_a + ia;
                        for (0..nb) |ib| {
                            const j = off_b + ib;
                            if (i < j) continue; // enforce i >= j

                            const ij = triangularIndex(i, j);

                            for (0..nc) |ic| {
                                const k = off_c + ic;
                                for (0..nd) |id| {
                                    const l = off_d + id;
                                    if (k < l) continue; // enforce k >= l

                                    const kl = triangularIndex(k, l);

                                    const buf_idx =
                                        ia * nb * nc * nd + ib * nc * nd + ic * nd + id;
                                    const val = eri_buf[buf_idx];
                                    // Use max/min to store at the canonical index (ij>=kl)
                                    const big = @max(ij, kl);
                                    const small = @min(ij, kl);
                                    values[triangularIndex(big, small)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return GeneralEriTable{ .values = values, .n = n };
}

/// Shell pair index: maps (a, b) with a >= b to a*(a+1)/2 + b.
fn shellPairIndex(a: usize, b: usize) usize {
    if (a >= b) return a * (a + 1) / 2 + b;
    return b * (b + 1) / 2 + a;
}

/// Triangular index: maps (i,j) with i >= j to i*(i+1)/2 + j.
fn triangularIndex(i: usize, j: usize) usize {
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
        const ij = triangularIndex(i, j);
        const kl = triangularIndex(k, l);
        const idx = triangularIndex(ij, kl);
        return self.values[idx];
    }
};

/// Compute total number of basis functions from a set of shells.
pub fn totalBasisFunctions(shells: []const ContractedShell) usize {
    var n: usize = 0;
    for (shells) |shell| {
        n += shell.numCartesianFunctions();
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
    const s_os = contractedOverlap(shell_a, s_cart, shell_b, s_cart);

    // Old implementation
    const overlap_old = @import("overlap.zig");
    const s_old = overlap_old.overlapSS(shell_a, shell_b);

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
    const s = contractedOverlap(shell, s_cart, shell, s_cart);
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

    const t_os = contractedKinetic(shell_a, s_cart, shell_b, s_cart);
    const kinetic_old = @import("kinetic.zig");
    const t_old = kinetic_old.kineticSS(shell_a, shell_b);

    try testing.expectApproxEqAbs(t_old, t_os, 1e-10);
}

test "OS nuclear attraction primitive p-type displaced nucleus" {
    const testing = std.testing;
    // Single primitive py at origin, nucleus at (0, 1.43, 1.11)
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const nuc = math.Vec3{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 };

    const alpha: f64 = 5.0331513;

    // <py|V|py> with a = b = (0,1,0), same exponent
    const v_yy = primitiveNuclearAttraction(
        alpha,
        center,
        .{ .x = 0, .y = 1, .z = 0 },
        alpha,
        center,
        .{ .x = 0, .y = 1, .z = 0 },
        nuc,
        1.0,
    );
    const v_xx = primitiveNuclearAttraction(
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
    const v_yy_ab = primitiveNuclearAttraction(
        alpha,
        center,
        .{ .x = 0, .y = 1, .z = 0 },
        beta,
        center,
        .{ .x = 0, .y = 1, .z = 0 },
        nuc,
        1.0,
    );
    const v_xx_ab = primitiveNuclearAttraction(
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

    const v_os = contractedNuclearAttraction(shell, s_cart, shell, s_cart, center, 1.0);
    const nuclear_old = @import("nuclear.zig");
    const v_old = nuclear_old.nuclearAttractionSS(shell, shell, center, 1.0);

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
    const eri_os = contractedERI(
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
    const eri_old = eri_old_mod.eriSSSS(shell_a, shell_a, shell_a, shell_a);
    try testing.expectApproxEqAbs(eri_old, eri_os, 1e-10);

    // (ab|ab)
    const eri_os2 = contractedERI(
        shell_a,
        s_cart,
        shell_b,
        s_cart,
        shell_a,
        s_cart,
        shell_b,
        s_cart,
    );
    const eri_old2 = eri_old_mod.eriSSSS(shell_a, shell_b, shell_a, shell_b);
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
    const sxx = contractedOverlap(shell, px, shell, px);
    const syy = contractedOverlap(shell, py, shell, py);
    const szz = contractedOverlap(shell, pz, shell, pz);
    try testing.expectApproxEqAbs(1.0, sxx, 1e-10);
    try testing.expectApproxEqAbs(1.0, syy, 1e-10);
    try testing.expectApproxEqAbs(1.0, szz, 1e-10);

    // Cross-overlaps should be 0.0
    const sxy = contractedOverlap(shell, px, shell, py);
    const sxz = contractedOverlap(shell, px, shell, pz);
    const syz = contractedOverlap(shell, py, shell, pz);
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
    const s_gen = try buildOverlapMatrix(alloc, &shells);
    defer alloc.free(s_gen);

    // Build with old builder
    const overlap_old = @import("overlap.zig");
    const s_old = try overlap_old.buildOverlapMatrix(alloc, &shells);
    defer alloc.free(s_old);

    for (0..4) |i| {
        try testing.expectApproxEqAbs(s_old[i], s_gen[i], 1e-12);
    }
}
