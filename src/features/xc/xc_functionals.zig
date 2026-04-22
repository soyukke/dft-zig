//! Exchange-correlation functionals for GTO-based DFT.
//!
//! All quantities are in Hartree atomic units.
//! Provides LDA (Slater exchange + VWN5 correlation), GGA (Becke88 + LYP),
//! and the B3LYP hybrid functional.
//!
//! Closed-shell (spin-unpolarized) only.

const std = @import("std");
const math = std.math;

// ==========================================================================
// Compile-time constants (avoid recomputing math.pow at every grid point)
// ==========================================================================

/// Slater exchange constant: C_x = -(3/4)*(3/pi)^(1/3)
const slater_c_x: f64 = -0.75 * math.pow(f64, 3.0 / math.pi, 1.0 / 3.0);

/// 2^(1/3) and 2^(-1/3) for Becke88
const two_1_3: f64 = math.pow(f64, 2.0, 1.0 / 3.0);
const two_m1_3: f64 = 1.0 / math.pow(f64, 2.0, 1.0 / 3.0);

/// 2^(2/3) for LYP
const two_2_3: f64 = math.pow(f64, 2.0, 2.0 / 3.0);

/// LYP auxiliary constants
const lyp_aux6: f64 = 1.0 / math.pow(f64, 2.0, 8.0 / 3.0);
const lyp_aux4: f64 = lyp_aux6 / 4.0;
const lyp_aux5: f64 = lyp_aux4 / (9.0 * 2.0);

/// Thomas-Fermi kinetic energy density prefactor: c_f = 0.3 * (3*pi^2)^(2/3)
const lyp_c_f: f64 = 0.3 * math.pow(f64, 3.0 * math.pi * math.pi, 2.0 / 3.0);

/// Result of evaluating an XC functional at a single grid point.
pub const XcResult = struct {
    /// Exchange-correlation energy density eps_xc(r) [Ha].
    /// E_xc = integral(eps_xc * rho * d^3r).
    eps_xc: f64,
    /// XC potential v_xc = d(rho * eps_xc)/d(rho) [Ha].
    v_xc: f64,
    /// For GGA: d(rho * eps_xc)/d(sigma) where sigma = |grad rho|^2.
    /// Zero for LDA.
    v_sigma: f64,
};

// ==========================================================================
// Slater (Dirac) Exchange (LDA)
// ==========================================================================

/// Slater exchange: eps_x = C_x * rho^(1/3), v_x = (4/3)*eps_x.
/// C_x = -(3/4)*(3/pi)^(1/3).
pub fn slaterExchange(rho: f64) XcResult {
    if (rho < 1e-30) return .{ .eps_xc = 0.0, .v_xc = 0.0, .v_sigma = 0.0 };

    const rho_13 = math.pow(f64, rho, 1.0 / 3.0);
    const eps_x = slater_c_x * rho_13;
    const v_x = (4.0 / 3.0) * slater_c_x * rho_13;

    return .{ .eps_xc = eps_x, .v_xc = v_x, .v_sigma = 0.0 };
}

// ==========================================================================
// VWN Correlation (LDA)
// ==========================================================================

/// VWN paramagnetic parameters (parameterized for VWN5 and VWN_RPA).
const VwnParams = struct {
    A: f64,
    x0: f64,
    b: f64,
    c: f64,
};

/// VWN5 (form V) paramagnetic parameters.
const vwn5_params = VwnParams{
    .A = 0.0621814,
    .x0 = -0.10498,
    .b = 3.72744,
    .c = 12.9352,
};

/// VWN_RPA (form III) paramagnetic parameters.
/// Used by B3LYP (Gaussian convention).
const vwn_rpa_params = VwnParams{
    .A = 0.0621814,
    .x0 = -0.409286,
    .b = 13.0720,
    .c = 42.7198,
};

/// Generic VWN correlation with given parameters.
fn vwnCorrelationGeneric(rho: f64, p: VwnParams) XcResult {
    if (rho < 1e-30) return .{ .eps_xc = 0.0, .v_xc = 0.0, .v_sigma = 0.0 };

    const rs = math.pow(f64, 3.0 / (4.0 * math.pi * rho), 1.0 / 3.0);
    const x = @sqrt(rs);
    const q = @sqrt(4.0 * p.c - p.b * p.b);
    const x_x = x * x + p.b * x + p.c;
    const x_x0 = p.x0 * p.x0 + p.b * p.x0 + p.c;
    const a = p.A / 2.0;

    const term1 = @log(x * x / x_x);
    const term2 = 2.0 * p.b / q * math.atan2(q, 2.0 * x + p.b);
    const term3_a = @log((x - p.x0) * (x - p.x0) / x_x);
    const term3_b = 2.0 * (p.b + 2.0 * p.x0) / q * math.atan2(q, 2.0 * x + p.b);
    const term3 = -p.x0 * p.b / x_x0 * (term3_a + term3_b);

    const eps_c = a * (term1 + term2 + term3);

    // v_c = eps_c - (rs/3) * deps_c/drs, with deps_c/drs = deps_c/dx / (2*x)
    const dx_x_val = 2.0 * x + p.b;
    const q2_plus_dx2 = q * q + dx_x_val * dx_x_val;
    const dterm1 = 2.0 / x - dx_x_val / x_x;
    // d/dx arctan(q/(2x+b)) = -2q / (q^2 + (2x+b)^2)
    const dterm2 = 2.0 * p.b / q * (-2.0 * q / q2_plus_dx2);
    const dterm3_a_d = 2.0 / (x - p.x0) - dx_x_val / x_x;
    const dterm3_b_d = 2.0 * (p.b + 2.0 * p.x0) / q * (-2.0 * q / q2_plus_dx2);
    const dterm3_d = -p.x0 * p.b / x_x0 * (dterm3_a_d + dterm3_b_d);

    const deps_dx = a * (dterm1 + dterm2 + dterm3_d);
    const deps_drs = deps_dx / (2.0 * x);
    const v_c = eps_c - (rs / 3.0) * deps_drs;

    return .{ .eps_xc = eps_c, .v_xc = v_c, .v_sigma = 0.0 };
}

/// VWN5 correlation energy and potential.
/// Reference: Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980), Table 4.4, form V.
pub fn vwnCorrelation(rho: f64) XcResult {
    return vwnCorrelationGeneric(rho, vwn5_params);
}

/// VWN_RPA correlation (form III).
/// Used as the LDA correlation component in B3LYP (Gaussian convention).
pub fn vwnRpaCorrelation(rho: f64) XcResult {
    return vwnCorrelationGeneric(rho, vwn_rpa_params);
}

/// LDA (SVWN5): Slater exchange + VWN5 correlation.
pub fn ldaSvwn(rho: f64) XcResult {
    const ex = slaterExchange(rho);
    const ec = vwnCorrelation(rho);
    return .{
        .eps_xc = ex.eps_xc + ec.eps_xc,
        .v_xc = ex.v_xc + ec.v_xc,
        .v_sigma = 0.0,
    };
}

// ==========================================================================
// Becke88 Exchange (GGA)
// ==========================================================================

/// Becke88 gradient-corrected exchange.
/// Reference: Becke, PRA 38, 3098 (1988).
///
/// For closed-shell, using spin-resolved formula:
///   x_s = |grad rho_a| / rho_a^(4/3) = sqrt(sigma) * 2^(1/3) / rho^(4/3)
///   F(x_s) = -beta * x_s^2 / (1 + 6*beta*x_s*asinh(x_s))
///   eps_x = eps_slater + 2^(-1/3) * rho^(1/3) * F(x_s)
pub fn becke88Exchange(rho: f64, sigma: f64) XcResult {
    if (rho < 1e-30) return .{ .eps_xc = 0.0, .v_xc = 0.0, .v_sigma = 0.0 };

    const slater = slaterExchange(rho);
    const beta: f64 = 0.0042;
    const rho_13 = math.pow(f64, rho, 1.0 / 3.0);
    const rho_43 = rho * rho_13;
    const grad_rho = @sqrt(@max(sigma, 0.0));

    // Spin-channel reduced gradient: x_s = sqrt(sigma) * 2^(1/3) / rho^(4/3)
    const x_s = grad_rho * two_1_3 / @max(rho_43, 1e-300);
    const x_s2 = x_s * x_s;
    const asinh_xs = @log(x_s + @sqrt(x_s2 + 1.0));
    const denom = 1.0 + 6.0 * beta * x_s * asinh_xs;

    const f_b88 = -beta * x_s2 / denom;
    const eps_xc = slater.eps_xc + two_m1_3 * rho_13 * f_b88;

    // dF/dx_s
    const sqrt_xs2p1 = @sqrt(x_s2 + 1.0);
    const ddenom_dxs = 6.0 * beta * (asinh_xs + x_s / sqrt_xs2p1);
    const df_dxs = -beta * (2.0 * x_s * denom - x_s2 * ddenom_dxs) / (denom * denom);

    // v_xc = d(rho * eps_xc)/d(rho)
    // The B88 correction energy per volume is: 2^{-1/3} * rho^{4/3} * F(x_s)
    // where x_s = sqrt(sigma)*2^{1/3}/rho^{4/3}, so dx_s/drho = -(4/3)*x_s/rho.
    // d(2^{-1/3}*rho^{4/3}*F)/drho
    //   = 2^{-1/3}*(4/3)*rho^{1/3}*F + 2^{-1/3}*rho^{4/3}*dF/dxs*(-4/3*x_s/rho)
    // = 2^{-1/3}*(4/3)*rho^{1/3}*(F - x_s*dF/dxs)
    const v_b88 = two_m1_3 * (4.0 / 3.0) * rho_13 * (f_b88 - x_s * df_dxs);
    const v_xc = slater.v_xc + v_b88;

    // v_sigma = d(rho*eps_xc)/d(sigma)
    // dx_s/dsigma = x_s/(2*sigma)
    // d(2^{-1/3}*rho^{4/3}*F)/dsigma = 2^{-1/3}*rho^{4/3}*dF/dxs*x_s/(2*sigma)
    var v_sigma: f64 = 0.0;
    if (sigma > 1e-30) {
        v_sigma = two_m1_3 * rho_43 * df_dxs * x_s / (2.0 * sigma);
    }

    return .{ .eps_xc = eps_xc, .v_xc = v_xc, .v_sigma = v_sigma };
}

// ==========================================================================
// LYP Correlation (GGA) - Closed Shell
// ==========================================================================

/// LYP parameters.
const lyp_a: f64 = 0.04918;
const lyp_b: f64 = 0.132;
const lyp_c: f64 = 0.2533;
const lyp_d: f64 = 0.349;

/// Evaluate closed-shell LYP energy density (eps_c = E_c/N) at (rho, sigma).
/// Follows the libxc/Miehlich parametrization exactly.
///
/// Variables:
///   rr = rho^(-1/3)
///   xt^2 = sigma / rho^(8/3)       (total reduced gradient squared)
///   xs^2 = sigma * 2^(2/3) / rho^(8/3) (spin-channel reduced gradient squared)
///   omega = b * exp(-c*rr) / (1 + d*rr)
///   delta = (c + d/(1+d*rr)) * rr
///
/// For closed shell (z=0):
///   eps_c = a * (t1 + omega * (t2 + t3 + t4 + t5 + t6))
///
/// Reference: Miehlich et al., CPL 157, 200 (1989); libxc gga_c_lyp.mpl
fn lypEpsilon(rho_val: f64, sigma_val: f64) f64 {
    if (rho_val < 1e-30) return 0.0;

    const rho_83 = math.pow(f64, rho_val, 8.0 / 3.0);
    const rr = 1.0 / math.pow(f64, rho_val, 1.0 / 3.0); // rho^(-1/3)
    const xt2 = sigma_val / rho_83;
    const xs2 = sigma_val * two_2_3 / rho_83;

    const den = 1.0 + lyp_d * rr;
    const omega = lyp_b * @exp(-lyp_c * rr) / den; // NOTE: no 'a' factor
    const delta = (lyp_c + lyp_d / den) * rr;

    const t1 = -1.0 / den;
    const t2 = -xt2 * ((47.0 - 7.0 * delta) / 72.0 - 2.0 / 3.0);
    const t3 = -lyp_c_f;
    const t4 = lyp_aux4 * (5.0 / 2.0 - delta / 18.0) * (2.0 * xs2);
    const t5 = lyp_aux5 * (delta - 11.0) * (2.0 * xs2);
    const t6 = -lyp_aux6 * xs2 * 5.0 / 6.0;

    return lyp_a * (t1 + omega * (t2 + t3 + t4 + t5 + t6));
}

/// Lee-Yang-Parr correlation for closed-shell systems.
/// Reference: Lee, Yang, Parr, PRB 37, 785 (1988).
/// Miehlich parametrization: CPL 157, 200 (1989).
/// Implementation follows libxc gga_c_lyp.mpl exactly.
/// Uses analytical derivatives for both v_xc and v_sigma.
pub fn lypCorrelation(rho: f64, sigma: f64) XcResult {
    if (rho < 1e-30) return .{ .eps_xc = 0.0, .v_xc = 0.0, .v_sigma = 0.0 };

    // Intermediate variables
    const rho_13 = math.pow(f64, rho, 1.0 / 3.0);
    const rho_83 = math.pow(f64, rho, 8.0 / 3.0);
    const rr = 1.0 / rho_13; // rho^(-1/3)
    const xt2 = sigma / rho_83;
    const xs2 = sigma * two_2_3 / rho_83;

    const den = 1.0 + lyp_d * rr;
    const exp_crr = @exp(-lyp_c * rr);
    const omega = lyp_b * exp_crr / den;
    const delta = (lyp_c + lyp_d / den) * rr;

    // Energy density terms
    const t1 = -1.0 / den;
    const t2 = -xt2 * ((47.0 - 7.0 * delta) / 72.0 - 2.0 / 3.0);
    const t3 = -lyp_c_f;
    const t4 = lyp_aux4 * (5.0 / 2.0 - delta / 18.0) * (2.0 * xs2);
    const t5 = lyp_aux5 * (delta - 11.0) * (2.0 * xs2);
    const t6 = -lyp_aux6 * xs2 * 5.0 / 6.0;

    const sigma_terms = t2 + t3 + t4 + t5 + t6;
    const eps_c = lyp_a * (t1 + omega * sigma_terms);

    // --- Analytical v_xc = d(rho * eps_c) / d(rho) ---
    // Chain rule through intermediate variables:
    //   drr/drho = -1/(3*rho^(4/3)) = -rr/(3*rho)
    const drr_drho = -rr / (3.0 * rho);

    //   dden/drho = d * drr/drho
    const dden_drho = lyp_d * drr_drho;

    //   dt1/drho = d(−1/den)/drho = dden/(den^2)
    const dt1_drho = dden_drho / (den * den);

    //   d(omega)/drho = omega * (−c*drr/drho − dden/drho/den)
    const domega_drho = omega * (-lyp_c * drr_drho - dden_drho / den);

    //   d(delta)/drho = d((c + d/den)*rr)/drho
    //     = (−d*dden/drho/(den^2))*rr + (c + d/den)*drr/drho
    const ddelta_drho = (-lyp_d * dden_drho / (den * den)) * rr + (lyp_c + lyp_d / den) * drr_drho;

    //   d(xt2)/drho = −(8/3) * xt2 / rho
    const dxt2_drho = -(8.0 / 3.0) * xt2 / rho;
    //   d(xs2)/drho = −(8/3) * xs2 / rho
    const dxs2_drho = -(8.0 / 3.0) * xs2 / rho;

    // Derivatives of t2..t6 w.r.t. rho
    // t2 = -xt2 * ((47 - 7*delta)/72 - 2/3)
    const dt2_drho = -dxt2_drho * ((47.0 - 7.0 * delta) / 72.0 - 2.0 / 3.0) -
        xt2 * (-7.0 * ddelta_drho / 72.0);

    // t3 = -c_f (constant w.r.t. rho)
    // dt3_drho = 0

    // t4 = aux4 * (5/2 - delta/18) * 2*xs2
    const dt4_drho = lyp_aux4 *
        ((-ddelta_drho / 18.0) * (2.0 * xs2) +
            (5.0 / 2.0 - delta / 18.0) * (2.0 * dxs2_drho));

    // t5 = aux5 * (delta - 11) * 2*xs2
    const dt5_drho = lyp_aux5 * (ddelta_drho * (2.0 * xs2) + (delta - 11.0) * (2.0 * dxs2_drho));

    // t6 = -aux6 * xs2 * 5/6
    const dt6_drho = -lyp_aux6 * dxs2_drho * 5.0 / 6.0;

    const dsigma_terms_drho = dt2_drho + dt4_drho + dt5_drho + dt6_drho;

    // d(eps_c)/drho = a * (dt1/drho + domega/drho * sigma_terms + omega * dsigma_terms/drho)
    const deps_drho = lyp_a * (dt1_drho + domega_drho * sigma_terms + omega * dsigma_terms_drho);

    const v_xc = eps_c + rho * deps_drho;

    // --- Analytical v_sigma = d(rho * eps_c) / d(sigma) ---
    // Only t2, t4, t5, t6 depend on sigma (via xt2 and xs2).
    // d(xt2)/d(sigma) = 1/rho^(8/3), d(xs2)/d(sigma) = 2^(2/3)/rho^(8/3)
    const dxt2_ds = 1.0 / rho_83;
    const dxs2_ds = two_2_3 / rho_83;

    const dt2_ds = -dxt2_ds * ((47.0 - 7.0 * delta) / 72.0 - 2.0 / 3.0);
    const dt4_ds = lyp_aux4 * (5.0 / 2.0 - delta / 18.0) * 2.0 * dxs2_ds;
    const dt5_ds = lyp_aux5 * (delta - 11.0) * 2.0 * dxs2_ds;
    const dt6_ds = -lyp_aux6 * dxs2_ds * 5.0 / 6.0;

    const v_s = rho * lyp_a * omega * (dt2_ds + dt4_ds + dt5_ds + dt6_ds);

    return .{ .eps_xc = eps_c, .v_xc = v_xc, .v_sigma = v_s };
}

// ==========================================================================
// B3LYP Hybrid Functional
// ==========================================================================

/// B3LYP parameters: a0=0.20, ax=0.72, ac=0.81.
/// E_xc = (1-a0)*E_x^Slater + a0*E_x^HF + ax*dE_x^B88 + (1-ac)*E_c^VWN + ac*E_c^LYP
const b3lyp_a0: f64 = 0.20;
const b3lyp_ax: f64 = 0.72;
const b3lyp_ac: f64 = 0.81;

/// B3LYP result (DFT part only; HF exchange added separately).
pub const B3lypResult = struct {
    eps_xc: f64,
    v_xc: f64,
    v_sigma: f64,
    hf_exchange_fraction: f64,
};

/// Evaluate B3LYP DFT components at a grid point.
/// The caller must add a0 * E_x^HF separately.
/// Optimized: computes Slater exchange only once; B88 gradient correction inline.
pub fn b3lyp(rho: f64, sigma: f64) B3lypResult {
    if (rho < 1e-30) return .{
        .eps_xc = 0.0,
        .v_xc = 0.0,
        .v_sigma = 0.0,
        .hf_exchange_fraction = b3lyp_a0,
    };

    // --- Slater exchange (computed once) ---
    const rho_13 = math.pow(f64, rho, 1.0 / 3.0);
    const slater_eps = slater_c_x * rho_13;
    const slater_v = (4.0 / 3.0) * slater_c_x * rho_13;

    // --- Becke88 GGA exchange correction (B88 - Slater part) ---
    const beta: f64 = 0.0042;
    const rho_43 = rho * rho_13;
    const grad_rho = @sqrt(@max(sigma, 0.0));

    const x_s = grad_rho * two_1_3 / @max(rho_43, 1e-300);
    const x_s2 = x_s * x_s;
    const asinh_xs = @log(x_s + @sqrt(x_s2 + 1.0));
    const denom = 1.0 + 6.0 * beta * x_s * asinh_xs;

    const f_b88 = -beta * x_s2 / denom;
    // B88 correction to eps (the part beyond Slater)
    const db88_eps = two_m1_3 * rho_13 * f_b88;

    // dF/dx_s for B88 correction v_xc and v_sigma
    const sqrt_xs2p1 = @sqrt(x_s2 + 1.0);
    const ddenom_dxs = 6.0 * beta * (asinh_xs + x_s / sqrt_xs2p1);
    const df_dxs = -beta * (2.0 * x_s * denom - x_s2 * ddenom_dxs) / (denom * denom);

    const db88_v = two_m1_3 * (4.0 / 3.0) * rho_13 * (f_b88 - x_s * df_dxs);

    var b88_v_sigma: f64 = 0.0;
    if (sigma > 1e-30) {
        b88_v_sigma = two_m1_3 * rho_43 * df_dxs * x_s / (2.0 * sigma);
    }

    // --- VWN_RPA correlation ---
    const vwn = vwnRpaCorrelation(rho);

    // --- LYP correlation ---
    const lyp = lypCorrelation(rho, sigma);

    // --- B3LYP combination ---
    // E_xc = (1-a0)*Slater + ax*dB88 + (1-ac)*VWN + ac*LYP
    // Note: (1-a0)*S + ax*(B88-S) = (1-a0-ax)*S + ax*B88 = (1-a0)*S + ax*dB88
    return .{
        .eps_xc = (1.0 - b3lyp_a0) * slater_eps + b3lyp_ax * db88_eps +
            (1.0 - b3lyp_ac) * vwn.eps_xc + b3lyp_ac * lyp.eps_xc,
        .v_xc = (1.0 - b3lyp_a0) * slater_v + b3lyp_ax * db88_v +
            (1.0 - b3lyp_ac) * vwn.v_xc + b3lyp_ac * lyp.v_xc,
        .v_sigma = b3lyp_ax * b88_v_sigma + b3lyp_ac * lyp.v_sigma,
        .hf_exchange_fraction = b3lyp_a0,
    };
}

// ==========================================================================
// Tests against PySCF/libxc reference values
// ==========================================================================

test "slater exchange vs PySCF" {
    // PySCF: LDA_X at rho=0.1 => eps_x = -0.342808612300562, v_x = -0.457078149734083
    const result = slaterExchange(0.1);
    try std.testing.expectApproxEqAbs(-0.342808612300562, result.eps_xc, 1e-12);
    try std.testing.expectApproxEqAbs(-0.457078149734083, result.v_xc, 1e-12);
}

test "vwn5 correlation vs PySCF" {
    // PySCF: LDA_C_VWN at rho=0.1 => eps_c = -0.053397289185950, v_c = -0.060812030331262
    const result = vwnCorrelation(0.1);
    try std.testing.expectApproxEqAbs(-0.053397289185950, result.eps_xc, 1e-10);
    try std.testing.expectApproxEqAbs(-0.060812030331262, result.v_xc, 1e-10);
}

test "vwn_rpa correlation vs PySCF" {
    // PySCF: LDA_C_VWN_RPA at rho=0.1 => eps_c = -0.072059367828481, v_c = -0.080233973560799
    const result = vwnRpaCorrelation(0.1);
    try std.testing.expectApproxEqAbs(-0.072059367828481, result.eps_xc, 1e-10);
    try std.testing.expectApproxEqAbs(-0.080233973560799, result.v_xc, 1e-10);
}

test "lda svwn vs PySCF" {
    // PySCF: LDA_X + LDA_C_VWN at rho=0.1 => eps = -0.396205901486512, v = -0.517890180065345
    const result = ldaSvwn(0.1);
    try std.testing.expectApproxEqAbs(-0.396205901486512, result.eps_xc, 1e-10);
    try std.testing.expectApproxEqAbs(-0.517890180065345, result.v_xc, 1e-10);
}

test "becke88 exchange vs PySCF" {
    // PySCF: GGA_X_B88 at rho=0.1, sigma=0.01
    // eps = -0.353006520959622, vrho = -0.445695995143018, vsigma = -0.093672623011794
    const result = becke88Exchange(0.1, 0.01);
    try std.testing.expectApproxEqAbs(-0.353006520959622, result.eps_xc, 1e-10);
    try std.testing.expectApproxEqAbs(-0.445695995143018, result.v_xc, 1e-8);
    try std.testing.expectApproxEqAbs(-0.093672623011794, result.v_sigma, 1e-8);
}

test "becke88 reduces to slater at zero gradient" {
    const rho = 0.1;
    const b88 = becke88Exchange(rho, 0.0);
    const slater = slaterExchange(rho);
    try std.testing.expectApproxEqAbs(slater.eps_xc, b88.eps_xc, 1e-12);
}

test "lyp correlation vs PySCF" {
    // PySCF: GGA_C_LYP at rho=0.1, sigma=0.01
    // eps = -0.032877381431883, vrho = -0.042336616391100, vsigma = 0.013598460610504
    const result = lypCorrelation(0.1, 0.01);
    try std.testing.expectApproxEqAbs(-0.032877381431883, result.eps_xc, 1e-12);
    try std.testing.expectApproxEqAbs(-0.042336616391100, result.v_xc, 1e-8);
    try std.testing.expectApproxEqAbs(0.013598460610504, result.v_sigma, 1e-12);
}

test "lyp correlation at rho=0.5, sigma=0.1 vs PySCF" {
    // PySCF: eps = -0.043355808865476, vrho = -0.049351917016856, vsigma = 0.001065244936754
    const result = lypCorrelation(0.5, 0.1);
    try std.testing.expectApproxEqAbs(-0.043355808865476, result.eps_xc, 1e-12);
    try std.testing.expectApproxEqAbs(-0.049351917016856, result.v_xc, 1e-8);
    try std.testing.expectApproxEqAbs(0.001065244936754, result.v_sigma, 1e-12);
}

test "b3lyp vs PySCF" {
    // PySCF: B3LYP at rho=0.1, sigma=0.01
    // eps = -0.321911342922210, vrho = -0.407004482735042, vsigma = -0.056429535473984
    const result = b3lyp(0.1, 0.01);
    try std.testing.expectApproxEqAbs(-0.321911342922210, result.eps_xc, 1e-10);
    try std.testing.expectApproxEqAbs(-0.407004482735042, result.v_xc, 1e-6);
    try std.testing.expectApproxEqAbs(-0.056429535473984, result.v_sigma, 1e-8);
    try std.testing.expectApproxEqAbs(0.20, result.hf_exchange_fraction, 1e-14);
}
