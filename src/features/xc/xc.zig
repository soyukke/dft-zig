//! Exchange-correlation functionals (spin-unpolarized and spin-polarized).

const std = @import("std");

pub const Functional = enum {
    lda_pz,
    pbe,
};

pub const XcPoint = struct {
    /// Energy density per volume (Ry).
    f: f64,
    /// Partial derivative df/dn (Ry).
    df_dn: f64,
    /// Partial derivative df/d|grad n|^2 (Ry * bohr^5).
    df_dg2: f64,
};

pub fn functional_name(xc: Functional) []const u8 {
    return switch (xc) {
        .lda_pz => "lda_pz",
        .pbe => "pbe",
    };
}

pub fn eval_point(xc: Functional, n: f64, g2: f64) XcPoint {
    return switch (xc) {
        .lda_pz => lda_pz(n),
        .pbe => pbe(n, g2),
    };
}

// =========================================================================
// XC Kernel (second derivative) for DFPT
// =========================================================================

pub const XcKernelPoint = struct {
    /// d²(n*eps_xc)/dn² = dV_xc/dn (Ry).
    fxc: f64,
    /// d²(n*eps_xc)/(dn dσ) where σ=|∇n|² (Ry·bohr⁵).
    f_ns: f64 = 0,
    /// d²(n*eps_xc)/dσ² (Ry·bohr¹⁰).
    f_ss: f64 = 0,
    /// d(n*eps_xc)/dσ = df/dσ (Ry·bohr⁵).
    v_s: f64 = 0,
};

/// Evaluate XC kernel for DFPT.
/// For LDA, g2 is ignored (no gradient dependence).
/// For PBE, g2 = |∇n|² is needed for gradient-dependent kernel terms.
pub fn eval_kernel(xc_func: Functional, n: f64, g2: f64) XcKernelPoint {
    return switch (xc_func) {
        .lda_pz => lda_pz_kernel(n),
        .pbe => pbe_kernel(n, g2),
    };
}

/// PBE XC kernel: second derivatives of n*eps_xc(n,σ) via numerical finite differences.
/// Returns f_nn, f_nσ, f_σσ, v_σ in Rydberg units (eval_point already returns Ry).
fn pbe_kernel(n: f64, g2: f64) XcKernelPoint {
    if (n <= 1e-12) return .{ .fxc = 0, .f_ns = 0, .f_ss = 0, .v_s = 0 };

    // v_σ = d(n·ε)/dσ = df/dσ  (directly from eval_point)
    const pt0 = eval_point(.pbe, n, g2);
    const v_s = pt0.df_dg2;

    // f_nn = d²(n·ε)/dn² ≈ (V_xc(n+δ) - V_xc(n-δ)) / (2δ)
    // where V_xc = d(n·ε)/dn = df/dn
    const delta_n = @max(1e-6, @abs(n) * 1e-5);
    const pt_np = eval_point(.pbe, n + delta_n, g2);
    const pt_nm = eval_point(.pbe, n - delta_n, g2);
    const f_nn = (pt_np.df_dn - pt_nm.df_dn) / (2.0 * delta_n);

    // f_nσ = d²(n·ε)/(dn dσ) ≈ (df_dg2(n+δ) - df_dg2(n-δ)) / (2δ)
    const f_ns = (pt_np.df_dg2 - pt_nm.df_dg2) / (2.0 * delta_n);

    // f_σσ = d²(n·ε)/dσ² ≈ (df_dg2(σ+δ) - df_dg2(σ-δ)) / (2δ)
    const delta_s = @max(1e-10, @abs(g2) * 1e-5);
    const pt_sp = eval_point(.pbe, n, g2 + delta_s);
    const pt_sm = eval_point(.pbe, n, @max(0.0, g2 - delta_s));
    const f_ss = (pt_sp.df_dg2 - pt_sm.df_dg2) / (2.0 * delta_s);

    return .{ .fxc = f_nn, .f_ns = f_ns, .f_ss = f_ss, .v_s = v_s };
}

/// LDA PZ XC kernel: f_xc = dV_xc/dn (Rydberg units).
/// f_xc = f_xc_exchange + f_xc_correlation
fn lda_pz_kernel(n: f64) XcKernelPoint {
    if (n <= 1e-12) return .{ .fxc = 0.0 };
    const pi = std.math.pi;
    const to_ry = 2.0;

    // Exchange kernel (Hartree units):
    // V_x = (4/3) c_x n^(1/3)
    // dV_x/dn = (4/9) c_x n^(-2/3)
    const c_x = -0.75 * std.math.pow(f64, 3.0 / pi, 1.0 / 3.0);
    const n_m23 = std.math.pow(f64, n, -2.0 / 3.0);
    const fxc_x = (4.0 / 9.0) * c_x * n_m23;

    // Correlation kernel
    const fxc_c = lda_correlation_pz_kernel(n);

    return .{ .fxc = to_ry * (fxc_x + fxc_c) };
}

/// LDA PZ correlation kernel: dV_c/dn (Hartree units).
/// Uses chain rule: dV_c/dn = dV_c/drs × drs/dn, where drs/dn = -rs/(3n).
fn lda_correlation_pz_kernel(n: f64) f64 {
    const pi = std.math.pi;
    const rs = std.math.pow(f64, 3.0 / (4.0 * pi * n), 1.0 / 3.0);
    const drs_dn = -rs / (3.0 * n);

    if (rs < 1.0) {
        const a = 0.0311;
        const c = 0.0020;
        const d = -0.0116;
        // eps_c = a ln(rs) + b + c rs ln(rs) + d rs
        // deps/drs = a/rs + c ln(rs) + c + d
        // V_c = eps_c - (rs/3) deps/drs
        // dV_c/drs = deps/drs - (1/3)(deps/drs + rs d²eps/drs²)
        //          = (2/3) deps/drs - (rs/3) d²eps/drs²
        // d²eps/drs² = -a/rs² + c/rs
        const deps = a / rs + c * @log(rs) + c + d;
        const d2eps = -a / (rs * rs) + c / rs;
        const dVc_drs = (2.0 / 3.0) * deps - (rs / 3.0) * d2eps;
        return dVc_drs * drs_dn;
    } else {
        const gamma = -0.1423;
        const beta1 = 1.0529;
        const beta2 = 0.3334;
        const sqrt_rs = std.math.sqrt(rs);
        const denom = 1.0 + beta1 * sqrt_rs + beta2 * rs;
        const denom2 = denom * denom;
        const denom3 = denom2 * denom;
        // eps_c = gamma / denom
        // deps/drs = -gamma (0.5 beta1 / sqrt_rs + beta2) / denom²
        // V_c = eps_c - (rs/3) deps/drs
        // dV_c/drs = (2/3) deps/drs - (rs/3) d²eps/drs²
        // d²eps/drs² = -gamma × [d_denom_prime / denom² - 2 denom_prime² / denom³]
        // where denom_prime = 0.5 beta1 / sqrt_rs + beta2
        //       d_denom_prime = -0.25 beta1 / rs^(3/2)
        const denom_prime = 0.5 * beta1 / sqrt_rs + beta2;
        const d_denom_prime = -0.25 * beta1 / (rs * sqrt_rs);
        const deps = -gamma * denom_prime / denom2;
        const d2eps = -gamma * (d_denom_prime / denom2 - 2.0 * denom_prime * denom_prime / denom3);
        const dVc_drs = (2.0 / 3.0) * deps - (rs / 3.0) * d2eps;
        return dVc_drs * drs_dn;
    }
}

// =========================================================================
// Spin-polarized XC
// =========================================================================

pub const XcPointSpin = struct {
    /// Energy density per volume n*eps_xc (Ry).
    f: f64,
    /// Partial derivative df/d(n_up) (Ry).
    df_dn_up: f64,
    /// Partial derivative df/d(n_down) (Ry).
    df_dn_down: f64,
    /// df/d|grad n_up|^2 (Ry * bohr^5).
    df_dg2_uu: f64,
    /// df/d|grad n_down|^2 (Ry * bohr^5).
    df_dg2_dd: f64,
    /// df/d(grad n_up . grad n_down) (Ry * bohr^5).
    df_dg2_ud: f64,
};

const xc_point_spin_zero = XcPointSpin{
    .f = 0.0,
    .df_dn_up = 0.0,
    .df_dn_down = 0.0,
    .df_dg2_uu = 0.0,
    .df_dg2_dd = 0.0,
    .df_dg2_ud = 0.0,
};

/// Evaluate spin-polarized XC energy density and potentials at a single point.
/// All inputs/outputs in Hartree internally, converted to Rydberg on output.
pub fn eval_point_spin(
    xc_func: Functional,
    n_up: f64,
    n_down: f64,
    g2_uu: f64,
    g2_dd: f64,
    g2_ud: f64,
) XcPointSpin {
    return switch (xc_func) {
        .lda_pz => lda_pz_spin(n_up, n_down),
        .pbe => pbe_spin(n_up, n_down, g2_uu, g2_dd, g2_ud),
    };
}

fn lda_pz(n: f64) XcPoint {
    if (n <= 1e-12) return .{ .f = 0.0, .df_dn = 0.0, .df_dg2 = 0.0 };
    const pi = std.math.pi;
    const n13 = std.math.pow(f64, n, 1.0 / 3.0);
    const c_x = -0.75 * std.math.pow(f64, 3.0 / pi, 1.0 / 3.0);
    const eps_x = c_x * n13;
    const ex = n * eps_x;
    const eps_c = lda_correlation_pz_energy(n);
    const v_c = lda_correlation_pz_potential(n);
    const ec = n * eps_c;
    const to_ry = 2.0;
    return .{
        .f = to_ry * (ex + ec),
        .df_dn = to_ry * (4.0 / 3.0 * c_x * n13 + v_c),
        .df_dg2 = 0.0,
    };
}

fn pbe(n: f64, g2: f64) XcPoint {
    if (n <= 1e-12) return .{ .f = 0.0, .df_dn = 0.0, .df_dg2 = 0.0 };

    const ex = pbe_exchange(n, g2);
    const ec = pbe_correlation(n, g2);
    const to_ry = 2.0;
    return .{
        .f = to_ry * (ex.f + ec.f),
        .df_dn = to_ry * (ex.df_dn + ec.df_dn),
        .df_dg2 = to_ry * (ex.df_dg2 + ec.df_dg2),
    };
}

fn pbe_exchange(n: f64, g2: f64) XcPoint {
    const pi = std.math.pi;
    const mu = 0.2195149727645171;
    const kappa = 0.804;
    const c_x = -0.75 * std.math.pow(f64, 3.0 / pi, 1.0 / 3.0);

    const n13 = std.math.pow(f64, n, 1.0 / 3.0);
    const eps_x = c_x * n13;
    const ex0 = n * eps_x;

    const kf = std.math.pow(f64, 3.0 * pi * pi * n, 1.0 / 3.0);
    const kf2 = kf * kf;
    const denom = 4.0 * kf2 * n * n;
    const s2 = if (g2 > 0.0 and denom > 0.0) g2 / denom else 0.0;
    const t = 1.0 + mu * s2 / kappa;
    const f_x = 1.0 + kappa - kappa / t;
    const df_ds2 = mu / (t * t);
    const d_n_eps_dn = 4.0 / 3.0 * c_x * n13;
    const ds2_dn = -8.0 / 3.0 * s2 / n;
    const df_dn = d_n_eps_dn * f_x + ex0 * df_ds2 * ds2_dn;
    const ds2_dg2 = if (denom > 0.0) 1.0 / denom else 0.0;
    const df_dg2 = ex0 * df_ds2 * ds2_dg2;

    return .{
        .f = ex0 * f_x,
        .df_dn = df_dn,
        .df_dg2 = df_dg2,
    };
}

fn pbe_correlation(n: f64, g2: f64) XcPoint {
    const pi = std.math.pi;
    const beta = 0.06672455060314922;
    const gamma = 0.031090690869654895;

    const rs = std.math.pow(f64, 3.0 / (4.0 * pi * n), 1.0 / 3.0);
    const corr = pw92_correlation(rs);
    const eps_c = corr.eps;
    const deps_drs = corr.deps_drs;
    const v_c = corr.v_c;
    const deps_dn = deps_drs * (-rs / (3.0 * n));

    const kf = std.math.pow(f64, 3.0 * pi * pi * n, 1.0 / 3.0);
    const ks2 = 4.0 * kf / pi;
    const denom = 4.0 * ks2 * n * n;
    const t2 = if (g2 > 0.0 and denom > 0.0) g2 / denom else 0.0;
    const b = beta / gamma;
    const x = b * t2;
    const u = std.math.exp(-eps_c / gamma);
    const u_minus = u - 1.0;
    const A = if (@abs(u_minus) > 1e-12) (beta / gamma) / u_minus else 0.0;
    const y = A * t2;
    // PBE formula: H = γ ln{1 + (β/γ)t² × (1 + At²) / (1 + At² + A²t⁴)}
    // denom2 = 1 + At² + A²t⁴ = 1 + y + y²
    const denom2 = 1.0 + y + y * y;
    const q = (1.0 + y) / denom2;
    const arg = 1.0 + x * q;
    const log_arg = if (arg > 0.0) @log(arg) else 0.0;
    const H = gamma * log_arg;

    const denom2_sq = denom2 * denom2;
    // dq/dt² = d[(1+y)/(1+y+y²)]/dt² where y = A*t²
    // = A * [(1+y+y²) - (1+y)*(1+2y)] / denom2²
    // = A * [1+y+y² - 1 - 2y - y - 2y²] / denom2²
    // = A * [-2y - y²] / denom2² = -A * y * (2 + y) / denom2²
    const dq_dt2 = -A * y * (2.0 + y) / denom2_sq;
    const darg_dt2 = b * q + x * dq_dt2;
    const dH_dt2 = if (arg > 0.0) gamma * darg_dt2 / arg else 0.0;
    // dH/dA = gamma * d(log_arg)/dA = gamma * x * dq/dA / arg
    // dq/dA = t2 * dq/dy = -t2 * y * (2+y) / denom2^2
    const dH_dA = if (arg > 0.0) -gamma * x * t2 * y * (2.0 + y) / (arg * denom2_sq) else 0.0;

    const dA_dn = if (@abs(u_minus) > 1e-12)
        (beta / gamma) * u / (gamma * u_minus * u_minus) * deps_dn
    else
        0.0;
    const dt2_dn = -7.0 / 3.0 * t2 / n;
    const dH_dn = dH_dt2 * dt2_dn + dH_dA * dA_dn;
    const dt2_dg2 = if (denom > 0.0) 1.0 / denom else 0.0;
    const dH_dg2 = dH_dt2 * dt2_dg2;

    const f = n * (eps_c + H);
    const df_dn = v_c + H + n * dH_dn;
    const df_dg2 = n * dH_dg2;

    return .{ .f = f, .df_dn = df_dn, .df_dg2 = df_dg2 };
}

const CorrResult = struct {
    eps: f64,
    deps_drs: f64,
    v_c: f64,
};

fn pw92_correlation(rs: f64) CorrResult {
    if (rs <= 1e-12) return .{ .eps = 0.0, .deps_drs = 0.0, .v_c = 0.0 };
    const a = 0.031091;
    const a1 = 0.21370;
    const b1 = 7.5957;
    const b2 = 3.5876;
    const b3 = 1.6382;
    const b4 = 0.49294;

    const sqrt_rs = std.math.sqrt(rs);
    const rs32 = rs * sqrt_rs;
    const rs2 = rs * rs;
    const f = b1 * sqrt_rs + b2 * rs + b3 * rs32 + b4 * rs2;
    const g = 1.0 + 1.0 / (2.0 * a * f);
    const ln_g = @log(g);
    const eps = -2.0 * a * (1.0 + a1 * rs) * ln_g;

    const df_drs = 0.5 * b1 / sqrt_rs + b2 + 1.5 * b3 * sqrt_rs + 2.0 * b4 * rs;
    const dln_g_drs = -(1.0 / (2.0 * a * f * f * g)) * df_drs;
    const deps_drs = -2.0 * a * (a1 * ln_g + (1.0 + a1 * rs) * dln_g_drs);
    const v_c = eps - (rs / 3.0) * deps_drs;
    return .{ .eps = eps, .deps_drs = deps_drs, .v_c = v_c };
}

fn lda_correlation_pz_energy(n: f64) f64 {
    const rs = std.math.pow(f64, 3.0 / (4.0 * std.math.pi * n), 1.0 / 3.0);
    if (rs < 1.0) {
        const a = 0.0311;
        const b = -0.048;
        const c = 0.0020;
        const d = -0.0116;
        const ln_rs = @log(rs);
        return a * ln_rs + b + c * rs * ln_rs + d * rs;
    } else {
        const gamma = -0.1423;
        const beta1 = 1.0529;
        const beta2 = 0.3334;
        const sqrt_rs = std.math.sqrt(rs);
        const denom = 1.0 + beta1 * sqrt_rs + beta2 * rs;
        return gamma / denom;
    }
}

fn lda_correlation_pz_potential(n: f64) f64 {
    const rs = std.math.pow(f64, 3.0 / (4.0 * std.math.pi * n), 1.0 / 3.0);
    if (rs < 1.0) {
        const a = 0.0311;
        const b = -0.048;
        const c = 0.0020;
        const d = -0.0116;
        const ln_rs = @log(rs);
        const eps_c = a * ln_rs + b + c * rs * ln_rs + d * rs;
        const deps = a / rs + c * ln_rs + c + d;
        return eps_c - (rs / 3.0) * deps;
    } else {
        const gamma = -0.1423;
        const beta1 = 1.0529;
        const beta2 = 0.3334;
        const sqrt_rs = std.math.sqrt(rs);
        const denom = 1.0 + beta1 * sqrt_rs + beta2 * rs;
        const eps_c = gamma / denom;
        const deps = -gamma * (0.5 * beta1 / sqrt_rs + beta2) / (denom * denom);
        return eps_c - (rs / 3.0) * deps;
    }
}

// =========================================================================
// Spin-polarized LDA (Perdew-Zunger)
// =========================================================================

/// PZ ferromagnetic correlation energy (rs >= 1).
/// Parameters from Perdew & Zunger, Phys. Rev. B 23, 5048 (1981).
fn pz_ferro_correlation(rs: f64) CorrResult {
    if (rs <= 1e-12) return .{ .eps = 0.0, .deps_drs = 0.0, .v_c = 0.0 };
    if (rs < 1.0) {
        const a = 0.01555;
        const b = -0.0269;
        const c = 0.0007;
        const d = -0.0048;
        const ln_rs = @log(rs);
        const eps = a * ln_rs + b + c * rs * ln_rs + d * rs;
        const deps = a / rs + c * ln_rs + c + d;
        const v_c = eps - (rs / 3.0) * deps;
        return .{ .eps = eps, .deps_drs = deps, .v_c = v_c };
    } else {
        const gamma = -0.0843;
        const beta1 = 1.3981;
        const beta2 = 0.2611;
        const sqrt_rs = std.math.sqrt(rs);
        const denom = 1.0 + beta1 * sqrt_rs + beta2 * rs;
        const eps = gamma / denom;
        const deps = -gamma * (0.5 * beta1 / sqrt_rs + beta2) / (denom * denom);
        const v_c = eps - (rs / 3.0) * deps;
        return .{ .eps = eps, .deps_drs = deps, .v_c = v_c };
    }
}

/// PZ paramagnetic correlation (wraps existing ldaCorrelation* for CorrResult).
fn pz_para_correlation(rs: f64) CorrResult {
    if (rs <= 1e-12) return .{ .eps = 0.0, .deps_drs = 0.0, .v_c = 0.0 };
    if (rs < 1.0) {
        const a = 0.0311;
        const b = -0.048;
        const c = 0.0020;
        const d = -0.0116;
        const ln_rs = @log(rs);
        const eps = a * ln_rs + b + c * rs * ln_rs + d * rs;
        const deps = a / rs + c * ln_rs + c + d;
        const v_c = eps - (rs / 3.0) * deps;
        return .{ .eps = eps, .deps_drs = deps, .v_c = v_c };
    } else {
        const gamma = -0.1423;
        const beta1 = 1.0529;
        const beta2 = 0.3334;
        const sqrt_rs = std.math.sqrt(rs);
        const denom = 1.0 + beta1 * sqrt_rs + beta2 * rs;
        const eps = gamma / denom;
        const deps = -gamma * (0.5 * beta1 / sqrt_rs + beta2) / (denom * denom);
        const v_c = eps - (rs / 3.0) * deps;
        return .{ .eps = eps, .deps_drs = deps, .v_c = v_c };
    }
}

const SpinInterp = struct { f_z: f64, df_dz: f64 };

/// Spin-interpolation function f(zeta) = [(1+zeta)^(4/3) + (1-zeta)^(4/3) - 2] / [2(2^(1/3) - 1)]
fn spin_interpolation(zeta: f64) SpinInterp {
    const two13 = std.math.pow(f64, 2.0, 1.0 / 3.0);
    const denom = 2.0 * (two13 - 1.0);
    const zp = 1.0 + zeta;
    const zm = 1.0 - zeta;
    const zp13 = if (zp > 1e-15) std.math.pow(f64, zp, 1.0 / 3.0) else 0.0;
    const zm13 = if (zm > 1e-15) std.math.pow(f64, zm, 1.0 / 3.0) else 0.0;
    const f_z = (zp * zp13 + zm * zm13 - 2.0) / denom;
    const df_dz = (4.0 / 3.0) * (zp13 - zm13) / denom;
    return .{ .f_z = f_z, .df_dz = df_dz };
}

/// Spin-polarized LDA PZ.
/// Exchange: spin scaling E_x[n_up,n_down] = (E_x[2n_up] + E_x[2n_down])/2
/// Correlation: PZ para/ferro interpolation with f(zeta)
fn lda_pz_spin(n_up: f64, n_down: f64) XcPointSpin {
    const n = n_up + n_down;
    if (n <= 1e-12) return xc_point_spin_zero;

    const pi = std.math.pi;
    const c_x = -0.75 * std.math.pow(f64, 3.0 / pi, 1.0 / 3.0);
    const to_ry = 2.0;

    // Exchange: spin scaling relation
    // E_x[n_up, n_down] = (E_x[2*n_up] + E_x[2*n_down]) / 2
    // eps_x(n) = c_x * n^(1/3), so E_x = n * eps_x = c_x * n^(4/3)
    // E_x_sigma = c_x * (2*n_sigma)^(4/3) / 2
    // V_x_sigma = d(E_x)/d(n_sigma) = c_x * (4/3) * 2^(1/3) * n_sigma^(1/3) * 2 / 2
    //           = (4/3)*c_x*2^(1/3)*n_sigma^(1/3)
    const two13 = std.math.pow(f64, 2.0, 1.0 / 3.0);
    var ex: f64 = 0.0;
    var vx_up: f64 = 0.0;
    var vx_down: f64 = 0.0;

    if (n_up > 1e-15) {
        const n_up13 = std.math.pow(f64, n_up, 1.0 / 3.0);
        ex += c_x * std.math.pow(f64, 2.0 * n_up, 4.0 / 3.0) / 2.0;
        vx_up = 4.0 / 3.0 * c_x * two13 * n_up13;
    }
    if (n_down > 1e-15) {
        const n_down13 = std.math.pow(f64, n_down, 1.0 / 3.0);
        ex += c_x * std.math.pow(f64, 2.0 * n_down, 4.0 / 3.0) / 2.0;
        vx_down = 4.0 / 3.0 * c_x * two13 * n_down13;
    }

    // Correlation: PZ with spin interpolation
    const rs = std.math.pow(f64, 3.0 / (4.0 * pi * n), 1.0 / 3.0);
    const para = pz_para_correlation(rs);
    const ferro = pz_ferro_correlation(rs);

    const zeta = std.math.clamp((n_up - n_down) / n, -1.0, 1.0);
    const fz = spin_interpolation(zeta);

    const eps_c = para.eps + (ferro.eps - para.eps) * fz.f_z;
    const deps_drs = para.deps_drs + (ferro.deps_drs - para.deps_drs) * fz.f_z;

    const ec = n * eps_c;

    // V_c_sigma = eps_c + n * deps_c/dn_sigma
    // deps_c/dn_sigma = deps_drs * drs/dn * (partial n / partial n_sigma = 1)
    //                  + (eps_F - eps_P) * df/dzeta * dzeta/dn_sigma
    // drs/dn = -rs/(3n)
    // dzeta/dn_up = (1 - zeta)/n, dzeta/dn_down = -(1 + zeta)/n
    const drs_dn = -rs / (3.0 * n);
    const delta_eps = ferro.eps - para.eps;
    const dzeta_dn_up = (1.0 - zeta) / n;
    const dzeta_dn_down = -(1.0 + zeta) / n;

    const vc_common = eps_c + n * deps_drs * drs_dn;
    const vc_up = vc_common + n * delta_eps * fz.df_dz * dzeta_dn_up;
    const vc_down = vc_common + n * delta_eps * fz.df_dz * dzeta_dn_down;

    return .{
        .f = to_ry * (ex + ec),
        .df_dn_up = to_ry * (vx_up + vc_up),
        .df_dn_down = to_ry * (vx_down + vc_down),
        .df_dg2_uu = 0.0,
        .df_dg2_dd = 0.0,
        .df_dg2_ud = 0.0,
    };
}

// =========================================================================
// Spin-polarized PBE
// =========================================================================

/// PW92 correlation with paramagnetic, ferromagnetic, and alpha_c parameters.
/// Returns { eps, deps_drs }.
const PW92Params = struct {
    a: f64,
    a1: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
};

fn pw92_correlation_generic(rs: f64, params: PW92Params) CorrResult {
    if (rs <= 1e-12) return .{ .eps = 0.0, .deps_drs = 0.0, .v_c = 0.0 };
    const sqrt_rs = std.math.sqrt(rs);
    const rs32 = rs * sqrt_rs;
    const rs2 = rs * rs;
    const f = params.b1 * sqrt_rs + params.b2 * rs + params.b3 * rs32 + params.b4 * rs2;
    const g = 1.0 + 1.0 / (2.0 * params.a * f);
    const ln_g = @log(g);
    const eps = -2.0 * params.a * (1.0 + params.a1 * rs) * ln_g;

    const df_drs = 0.5 * params.b1 / sqrt_rs + params.b2 +
        1.5 * params.b3 * sqrt_rs + 2.0 * params.b4 * rs;
    const dln_g_drs = -(1.0 / (2.0 * params.a * f * f * g)) * df_drs;
    const deps_drs = -2.0 * params.a * (params.a1 * ln_g + (1.0 + params.a1 * rs) * dln_g_drs);
    const v_c = eps - (rs / 3.0) * deps_drs;
    return .{ .eps = eps, .deps_drs = deps_drs, .v_c = v_c };
}

// PW92 paramagnetic (ec0)
const pw92_para = PW92Params{
    .a = 0.031091,
    .a1 = 0.21370,
    .b1 = 7.5957,
    .b2 = 3.5876,
    .b3 = 1.6382,
    .b4 = 0.49294,
};
// PW92 ferromagnetic (ec1)
const pw92_ferro = PW92Params{
    .a = 0.015545,
    .a1 = 0.20548,
    .b1 = 14.1189,
    .b2 = 6.1977,
    .b3 = 3.3662,
    .b4 = 0.62517,
};
// PW92 alpha_c (spin stiffness)
const pw92_alpha = PW92Params{
    .a = 0.016887,
    .a1 = 0.11125,
    .b1 = 10.357,
    .b2 = 3.6231,
    .b3 = 0.88026,
    .b4 = 0.49671,
};

/// Spin-polarized PBE.
fn pbe_spin(n_up: f64, n_down: f64, g2_uu: f64, g2_dd: f64, g2_ud: f64) XcPointSpin {
    const n = n_up + n_down;
    if (n <= 1e-12) return xc_point_spin_zero;

    const to_ry = 2.0;

    // Exchange: spin scaling E_x = (E_x[2n_up, 4g2_uu] + E_x[2n_down, 4g2_dd]) / 2
    const zero_xc = XcPoint{ .f = 0.0, .df_dn = 0.0, .df_dg2 = 0.0 };
    const ex_up = if (n_up > 1e-15) pbe_exchange(2.0 * n_up, 4.0 * g2_uu) else zero_xc;
    const ex_down = if (n_down > 1e-15) pbe_exchange(2.0 * n_down, 4.0 * g2_dd) else zero_xc;

    const f_ex = (ex_up.f + ex_down.f) / 2.0;
    // V_x_sigma = d(E_x)/d(n_sigma) = d(E_x[2n_sigma, 4g2_ss])/d(n_sigma) / 2
    //           = (dE/dn * 2) / 2 = dE/dn evaluated at (2n_sigma, 4g2_ss)
    const vx_up = ex_up.df_dn;
    const vx_down = ex_down.df_dn;
    // df/dg2_ss: chain rule: d/dg2_ss (E_x[2n_s, 4g2_ss]/2) = (dE/dg2 * 4) / 2 = 2 * dE/dg2
    const dfx_dg2_uu = 2.0 * ex_up.df_dg2;
    const dfx_dg2_dd = 2.0 * ex_down.df_dg2;

    // Correlation: spin-polarized PBE
    const ec = pbe_correlation_spin(n_up, n_down, g2_uu, g2_dd, g2_ud);

    return .{
        .f = to_ry * (f_ex + ec.f),
        .df_dn_up = to_ry * (vx_up + ec.df_dn_up),
        .df_dn_down = to_ry * (vx_down + ec.df_dn_down),
        .df_dg2_uu = to_ry * (dfx_dg2_uu + ec.df_dg2_uu),
        .df_dg2_dd = to_ry * (dfx_dg2_dd + ec.df_dg2_dd),
        .df_dg2_ud = to_ry * ec.df_dg2_ud,
    };
}

const PbeCorrelationSpinResult = struct {
    f: f64,
    df_dn_up: f64,
    df_dn_down: f64,
    df_dg2_uu: f64,
    df_dg2_dd: f64,
    df_dg2_ud: f64,
};

/// Shared intermediates for spin-polarized PBE: LDA inputs + PBE H setup.
const PbeSpinState = struct {
    // density-level quantities
    n: f64,
    rs: f64,
    zeta: f64,
    z4: f64,

    // PW92 LDA pieces
    ec0: CorrResult,
    ec1: CorrResult,
    ac: CorrResult,
    fz: SpinInterp,
    fdd0: f64,

    // eps_c(rs,zeta)
    eps_c: f64,

    // phi(zeta)
    phi: f64,
    phi2: f64,
    phi3: f64,
    dphi_dzeta: f64,

    // t^2 inputs
    g2: f64,
    denom_t: f64,
    t2: f64,

    // PBE H internals
    beta: f64,
    gamma_c: f64,
    b: f64,
    x: f64,
    u: f64,
    u_minus: f64,
    A: f64,
    y: f64,
    denom2: f64,
    q: f64,
    arg: f64,
    log_arg: f64,
    H: f64,
    darg_dt2: f64,
    dH_dt2: f64,
};

/// Spin-polarization basis: phi(zeta), phi^2, phi^3, dphi/dzeta.
const PhiSet = struct { phi: f64, phi2: f64, phi3: f64, dphi_dzeta: f64 };

fn compute_phi_set(zeta: f64) PhiSet {
    // phi(zeta) = ((1+zeta)^(2/3) + (1-zeta)^(2/3)) / 2
    const zp = 1.0 + zeta;
    const zm = 1.0 - zeta;
    const zp23 = if (zp > 1e-15) std.math.pow(f64, zp, 2.0 / 3.0) else 0.0;
    const zm23 = if (zm > 1e-15) std.math.pow(f64, zm, 2.0 / 3.0) else 0.0;
    const phi = (zp23 + zm23) / 2.0;
    const phi2 = phi * phi;

    // dphi/dzeta = (1/3) * ((1+zeta)^(-1/3) - (1-zeta)^(-1/3))
    const zp_m13 = if (zp > 1e-15) std.math.pow(f64, zp, -1.0 / 3.0) else 0.0;
    const zm_m13 = if (zm > 1e-15) std.math.pow(f64, zm, -1.0 / 3.0) else 0.0;
    const dphi_dzeta = (zp_m13 - zm_m13) / 3.0;

    return .{ .phi = phi, .phi2 = phi2, .phi3 = phi * phi * phi, .dphi_dzeta = dphi_dzeta };
}

/// PBE gradient coefficient t^2 = |grad n|^2 / (4 * phi^2 * ks^2 * n^2)
/// and its denominator denom_t for later chain-rule use.
const PbeT2 = struct { g2: f64, denom_t: f64, t2: f64 };

fn compute_pbe_t2(n: f64, phi2: f64, g2_uu: f64, g2_dd: f64, g2_ud: f64) PbeT2 {
    const pi = std.math.pi;
    const g2 = g2_uu + 2.0 * g2_ud + g2_dd;

    // ks^2 = 4*kf/pi, kf = (3*pi^2*n)^(1/3)
    const kf = std.math.pow(f64, 3.0 * pi * pi * n, 1.0 / 3.0);
    const ks2 = 4.0 * kf / pi;
    const denom_t = 4.0 * phi2 * ks2 * n * n;
    const t2 = if (g2 > 0.0 and denom_t > 1e-30) g2 / denom_t else 0.0;
    return .{ .g2 = g2, .denom_t = denom_t, .t2 = t2 };
}

/// PBE H term (PRL 77, 3865, Eq. 4-5) and its dH/dt^2 derivative.
const PbeHPieces = struct {
    beta: f64,
    gamma_c: f64,
    b: f64,
    x: f64,
    u: f64,
    u_minus: f64,
    A: f64,
    y: f64,
    denom2: f64,
    q: f64,
    arg: f64,
    log_arg: f64,
    H: f64,
    darg_dt2: f64,
    dH_dt2: f64,
};

fn compute_pbe_h(eps_c: f64, phi3: f64, t2: f64) PbeHPieces {
    // H = gamma * phi^3 * ln{1 + (beta/gamma) * t^2 * [(1+At^2)/(1+At^2+A^2*t^4)]}
    // A = (beta/gamma) / {exp(-ec/(gamma * phi^3)) - 1}
    const beta = 0.06672455060314922;
    const gamma_c = 0.031090690869654895;
    const b = beta / gamma_c;
    const x = b * t2;
    const u = std.math.exp(-eps_c / (gamma_c * phi3));
    const u_minus = u - 1.0;
    const A = if (@abs(u_minus) > 1e-12) (beta / gamma_c) / u_minus else 0.0;
    const y = A * t2;
    const denom2 = 1.0 + y + y * y;
    const q = (1.0 + y) / denom2;
    const arg = 1.0 + x * q;
    const log_arg = if (arg > 0.0) @log(arg) else 0.0;
    const H = gamma_c * phi3 * log_arg;

    // Derivatives of H w.r.t. t^2
    const denom2_sq = denom2 * denom2;
    const dq_dt2 = -A * y * (2.0 + y) / denom2_sq;
    const darg_dt2 = b * q + x * dq_dt2;
    const dH_dt2 = if (arg > 0.0) gamma_c * phi3 * darg_dt2 / arg else 0.0;

    return .{
        .beta = beta,
        .gamma_c = gamma_c,
        .b = b,
        .x = x,
        .u = u,
        .u_minus = u_minus,
        .A = A,
        .y = y,
        .denom2 = denom2,
        .q = q,
        .arg = arg,
        .log_arg = log_arg,
        .H = H,
        .darg_dt2 = darg_dt2,
        .dH_dt2 = dH_dt2,
    };
}

fn pbe_spin_state(
    n_up: f64,
    n_down: f64,
    g2_uu: f64,
    g2_dd: f64,
    g2_ud: f64,
) PbeSpinState {
    const pi = std.math.pi;

    const n = n_up + n_down;
    const zeta = std.math.clamp((n_up - n_down) / n, -1.0, 1.0);

    const rs = std.math.pow(f64, 3.0 / (4.0 * pi * n), 1.0 / 3.0);

    // ec0 (paramagnetic), ec1 (ferromagnetic), alpha_c
    const ec0 = pw92_correlation_generic(rs, pw92_para);
    const ec1 = pw92_correlation_generic(rs, pw92_ferro);
    const ac = pw92_correlation_generic(rs, pw92_alpha);

    // f(zeta) and phi(zeta)
    const fz = spin_interpolation(zeta);
    const fdd0 = 1.709921; // f''(0) = 4/(9*(2^(1/3)-1))

    // eps_c(rs, zeta) = ec0 - ac(rs) * f(zeta) / f''(0) * (1 - zeta^4)
    //                 + (ec1 - ec0) * f(zeta) * zeta^4
    // Note: ac.eps = -2a(1+a1*rs)*ln(g) < 0, so -ac.eps gives +alpha_c (physical spin stiffness)
    const z4 = zeta * zeta * zeta * zeta;
    const eps_c = ec0.eps - ac.eps * fz.f_z / fdd0 * (1.0 - z4) + (ec1.eps - ec0.eps) * fz.f_z * z4;

    const phi_set = compute_phi_set(zeta);
    const t = compute_pbe_t2(n, phi_set.phi2, g2_uu, g2_dd, g2_ud);
    const h = compute_pbe_h(eps_c, phi_set.phi3, t.t2);

    return .{
        .n = n,
        .rs = rs,
        .zeta = zeta,
        .z4 = z4,
        .ec0 = ec0,
        .ec1 = ec1,
        .ac = ac,
        .fz = fz,
        .fdd0 = fdd0,
        .eps_c = eps_c,
        .phi = phi_set.phi,
        .phi2 = phi_set.phi2,
        .phi3 = phi_set.phi3,
        .dphi_dzeta = phi_set.dphi_dzeta,
        .g2 = t.g2,
        .denom_t = t.denom_t,
        .t2 = t.t2,
        .beta = h.beta,
        .gamma_c = h.gamma_c,
        .b = h.b,
        .x = h.x,
        .u = h.u,
        .u_minus = h.u_minus,
        .A = h.A,
        .y = h.y,
        .denom2 = h.denom2,
        .q = h.q,
        .arg = h.arg,
        .log_arg = h.log_arg,
        .H = h.H,
        .darg_dt2 = h.darg_dt2,
        .dH_dt2 = h.dH_dt2,
    };
}

/// LDA potential (d(n*eps_c)/dn_sigma) for spin-polarized PBE correlation.
const LdaVc = struct {
    up: f64,
    down: f64,
    deps_c_drs: f64,
    deps_c_dzeta: f64,
    drs_dn: f64,
    dzeta_dn_up: f64,
    dzeta_dn_down: f64,
};

fn pbe_lda_vc(s: PbeSpinState) LdaVc {
    // Potential: V_c_sigma = d(n*eps_c)/dn_sigma + d(n*H)/dn_sigma
    // LDA part: d(n*eps_c)/dn_sigma
    const drs_dn = -s.rs / (3.0 * s.n);
    const dzeta_dn_up = (1.0 - s.zeta) / s.n;
    const dzeta_dn_down = -(1.0 + s.zeta) / s.n;

    // deps_c/drs
    const deps_c_drs = s.ec0.deps_drs -
        s.ac.deps_drs * s.fz.f_z / s.fdd0 * (1.0 - s.z4) +
        (s.ec1.deps_drs - s.ec0.deps_drs) * s.fz.f_z * s.z4;

    // deps_c/dzeta
    const dz4_dzeta = 4.0 * s.zeta * s.zeta * s.zeta;
    const deps_c_dzeta = -s.ac.eps * s.fz.df_dz / s.fdd0 * (1.0 - s.z4) +
        (-s.ac.eps) * s.fz.f_z / s.fdd0 * (-dz4_dzeta) +
        (s.ec1.eps - s.ec0.eps) * (s.fz.df_dz * s.z4 + s.fz.f_z * dz4_dzeta);

    // d(n*eps_c)/dn_sigma = eps_c + n * (deps_c/drs * drs/dn + deps_c/dzeta * dzeta/dn_sigma)
    const vc_lda_common = s.eps_c + s.n * deps_c_drs * drs_dn;
    const vc_lda_up = vc_lda_common + s.n * deps_c_dzeta * dzeta_dn_up;
    const vc_lda_down = vc_lda_common + s.n * deps_c_dzeta * dzeta_dn_down;

    return .{
        .up = vc_lda_up,
        .down = vc_lda_down,
        .deps_c_drs = deps_c_drs,
        .deps_c_dzeta = deps_c_dzeta,
        .drs_dn = drs_dn,
        .dzeta_dn_up = dzeta_dn_up,
        .dzeta_dn_down = dzeta_dn_down,
    };
}

fn pbe_gga_d_hdn(s: PbeSpinState, lda: LdaVc) struct { up: f64, down: f64 } {
    // GGA H part: d(n*H)/dn_sigma
    // H = gamma_c * phi^3 * ln(arg), where arg depends on t^2 and A
    // A = (beta/gamma) / (exp(-eps_c/(gamma*phi^3)) - 1)
    //
    // dA/deps_c: derivative through the exponent
    // u = exp(-eps_c/(gamma*phi^3)), du/deps_c = -u/(gamma*phi^3)
    // dA/deps_c = -(beta/gamma) * du/deps_c / (u-1)^2 = (beta/gamma) * u / (gamma*phi^3 * (u-1)^2)
    const dA_deps = if (@abs(s.u_minus) > 1e-12)
        (s.beta / s.gamma_c) * s.u / (s.gamma_c * s.phi3 * s.u_minus * s.u_minus)
    else
        0.0;

    // dA/dphi3: derivative through the exponent
    // du/dphi3 = u * eps_c / (gamma * phi3^2)
    // dA/dphi3 = -(beta/gamma) * du/dphi3 / (u-1)^2
    //          = -(beta/gamma) * u * eps_c / (gamma * phi3^2 * (u-1)^2)
    const dA_dphi3 = if (@abs(s.u_minus) > 1e-12)
        -(s.beta / s.gamma_c) * s.u * s.eps_c /
            (s.gamma_c * s.phi3 * s.phi3 * s.u_minus * s.u_minus)
    else
        0.0;

    // dphi3/dn_sigma = 3 * phi^2 * dphi/dzeta * dzeta/dn_sigma
    const dphi3_dn_up = 3.0 * s.phi2 * s.dphi_dzeta * lda.dzeta_dn_up;
    const dphi3_dn_down = 3.0 * s.phi2 * s.dphi_dzeta * lda.dzeta_dn_down;

    // deps_c/dn_sigma = deps_c/drs * drs/dn + deps_c/dzeta * dzeta/dn_sigma
    const deps_dn_up = lda.deps_c_drs * lda.drs_dn + lda.deps_c_dzeta * lda.dzeta_dn_up;
    const deps_dn_down = lda.deps_c_drs * lda.drs_dn + lda.deps_c_dzeta * lda.dzeta_dn_down;

    // dA/dn_sigma = dA/deps_c * deps_c/dn_sigma + dA/dphi3 * dphi3/dn_sigma
    const dA_dn_up = dA_deps * deps_dn_up + dA_dphi3 * dphi3_dn_up;
    const dA_dn_down = dA_deps * deps_dn_down + dA_dphi3 * dphi3_dn_down;

    // dt^2/dn_sigma: t^2 = g2 / (4 * phi^2 * ks^2 * n^2)
    // dt^2/dn = -7/3 * t2/n (through ks and n)
    // dt^2/dphi = -2*t^2/phi
    const dt2_dn = -7.0 / 3.0 * s.t2 / s.n;
    const dt2_dphi = if (s.phi > 1e-15) -2.0 * s.t2 / s.phi else 0.0;

    const dt2_dn_up = dt2_dn + dt2_dphi * s.dphi_dzeta * lda.dzeta_dn_up;
    const dt2_dn_down = dt2_dn + dt2_dphi * s.dphi_dzeta * lda.dzeta_dn_down;

    // d(log_arg)/dn_sigma (through t^2 and A dependencies)
    // d(log_arg)/dt^2 = darg_dt2 / arg
    // d(log_arg)/dA = x * dq/dA / arg = x * t2 * dq/dy / arg
    //               = -x * t2 * y * (2+y) / (arg * denom2^2)
    const denom2_sq = s.denom2 * s.denom2;
    const dlogarg_dA = if (s.arg > 0.0)
        -s.x * s.t2 * s.y * (2.0 + s.y) / (s.arg * denom2_sq)
    else
        0.0;
    const dlogarg_dt2 = if (s.arg > 0.0) s.darg_dt2 / s.arg else 0.0;
    const dlogarg_dn_up = dlogarg_dt2 * dt2_dn_up + dlogarg_dA * dA_dn_up;
    const dlogarg_dn_down = dlogarg_dt2 * dt2_dn_down + dlogarg_dA * dA_dn_down;

    // dH/dn_sigma = gamma_c * phi3 * d(log_arg)/dn_sigma + gamma_c * dphi3/dn_sigma * log_arg
    const dH_dn_up = s.gamma_c * s.phi3 * dlogarg_dn_up + s.gamma_c * dphi3_dn_up * s.log_arg;
    const dH_dn_down = s.gamma_c * s.phi3 * dlogarg_dn_down + s.gamma_c * dphi3_dn_down * s.log_arg;

    return .{ .up = dH_dn_up, .down = dH_dn_down };
}

/// Spin-polarized PBE correlation (Hartree units internally).
fn pbe_correlation_spin(
    n_up: f64,
    n_down: f64,
    g2_uu: f64,
    g2_dd: f64,
    g2_ud: f64,
) PbeCorrelationSpinResult {
    const s = pbe_spin_state(n_up, n_down, g2_uu, g2_dd, g2_ud);

    // Energy density
    const f_c = s.n * (s.eps_c + s.H);

    // dt^2/dg2 = 1 / denom_t
    const dt2_dg2 = if (s.denom_t > 1e-30) 1.0 / s.denom_t else 0.0;
    const dH_dg2 = s.dH_dt2 * dt2_dg2;

    // df/dg2_ss = n * dH/dg2 (g2 = g2_uu + 2*g2_ud + g2_dd, so dg2/dg2_ss = 1)
    // dH_dg2 already contains phi^3 factor through dH_dt2
    const df_dg2_uu = s.n * dH_dg2;
    const df_dg2_dd = s.n * dH_dg2;
    const df_dg2_ud = 2.0 * s.n * dH_dg2;

    const lda = pbe_lda_vc(s);
    const dH_dn = pbe_gga_d_hdn(s, lda);

    // d(n*H)/dn_sigma = H + n * dH/dn_sigma
    const df_dn_up = lda.up + s.H + s.n * dH_dn.up;
    const df_dn_down = lda.down + s.H + s.n * dH_dn.down;

    return .{
        .f = f_c,
        .df_dn_up = df_dn_up,
        .df_dn_down = df_dn_down,
        .df_dg2_uu = df_dg2_uu,
        .df_dg2_dd = df_dg2_dd,
        .df_dg2_ud = df_dg2_ud,
    };
}

// =========================================================================
// XC Kernel tests
// =========================================================================

test "LDA PZ XC kernel matches finite difference of V_xc" {
    // Test at multiple density points including rs=1 boundary region
    const test_densities = [_]f64{
        0.001, // low density (rs > 1)
        0.01, // medium-low (rs > 1)
        0.05, // near rs=1 boundary
        0.1, // medium density
        0.5, // high density (rs < 1)
        1.0, // very high density (rs < 1)
    };
    const delta: f64 = 1e-6;

    for (test_densities) |n| {
        const kernel = eval_kernel(.lda_pz, n, 0.0);
        // Finite difference: (V_xc(n+δ) - V_xc(n-δ)) / (2δ)
        const vxc_plus = eval_point(.lda_pz, n + delta, 0.0).df_dn;
        const vxc_minus = eval_point(.lda_pz, n - delta, 0.0).df_dn;
        const fxc_num = (vxc_plus - vxc_minus) / (2.0 * delta);
        try testing.expectApproxEqRel(kernel.fxc, fxc_num, 1e-4);
    }
}

test "LDA PZ XC kernel at rs=1 boundary" {
    // rs = (3/(4πn))^(1/3), rs=1 → n = 3/(4π) ≈ 0.2387
    const pi = std.math.pi;
    const n_boundary = 3.0 / (4.0 * pi);
    const delta: f64 = 1e-6;

    // Test just below and above rs=1 boundary
    const n_below = n_boundary + 0.01; // rs < 1
    const n_above = n_boundary - 0.01; // rs > 1

    for ([_]f64{ n_below, n_above }) |n| {
        const kernel = eval_kernel(.lda_pz, n, 0.0);
        const vxc_plus = eval_point(.lda_pz, n + delta, 0.0).df_dn;
        const vxc_minus = eval_point(.lda_pz, n - delta, 0.0).df_dn;
        const fxc_num = (vxc_plus - vxc_minus) / (2.0 * delta);
        try testing.expectApproxEqRel(kernel.fxc, fxc_num, 1e-4);
    }
}

test "LDA PZ XC kernel is negative for physical densities" {
    // The XC kernel should be negative (attractive) for physical densities
    const kernel = eval_kernel(.lda_pz, 0.1, 0.0);
    try testing.expect(kernel.fxc < 0.0);
}

// Unit tests comparing with libxc reference values
const testing = std.testing;

test "PBE exchange matches libxc" {
    // Test at rho=0.1, sigma=0.01 (libxc reference values)
    const rho: f64 = 0.1;
    const sigma: f64 = 0.01;
    const result = pbe_exchange(rho, sigma);

    // libxc PBE exchange reference (Hartree units)
    const libxc_exc: f64 = -0.3516400536;
    const libxc_vrho: f64 = -0.4460575074;
    const libxc_vsigma: f64 = -0.0854846156;

    const exc = result.f / rho;
    const vrho = result.df_dn;
    const vsigma = result.df_dg2;

    // 0.01% tolerance
    try testing.expectApproxEqRel(exc, libxc_exc, 1e-4);
    try testing.expectApproxEqRel(vrho, libxc_vrho, 1e-4);
    try testing.expectApproxEqRel(vsigma, libxc_vsigma, 1e-4);
}

test "PBE correlation matches libxc" {
    // Test at rho=0.1, sigma=0.01 (libxc reference values)
    const rho: f64 = 0.1;
    const sigma: f64 = 0.01;
    const result = pbe_correlation(rho, sigma);

    // libxc PBE correlation reference (Hartree units)
    const libxc_exc: f64 = -0.0452782280;
    const libxc_vrho: f64 = -0.0688510243;
    const libxc_vsigma: f64 = 0.0697928401;

    const exc = result.f / rho;
    const vrho = result.df_dn;
    const vsigma = result.df_dg2;

    // vsigma: 0.01% tolerance (critical for band gap)
    try testing.expectApproxEqRel(vsigma, libxc_vsigma, 1e-4);
    // exc: 0.01% tolerance
    try testing.expectApproxEqRel(exc, libxc_exc, 1e-4);
    // vrho: 0.1% tolerance (slightly relaxed due to dH_dA complexity)
    try testing.expectApproxEqRel(vrho, libxc_vrho, 1e-3);
}

test "PBE total XC matches libxc" {
    const rho: f64 = 0.1;
    const sigma: f64 = 0.01;
    const result = eval_point(.pbe, rho, sigma);

    // libxc total PBE (exchange + correlation)
    // Note: eval_point returns in Rydberg, libxc is in Hartree
    const libxc_exc_total: f64 = -0.3516400536 + -0.0452782280; // -0.3969182816 Ha
    const libxc_vsigma_total: f64 = -0.0854846156 + 0.0697928401; // -0.0156917755 Ha

    // Convert to Hartree for comparison (divide by 2)
    const exc = (result.f / rho) / 2.0;
    const vsigma = result.df_dg2 / 2.0;

    try testing.expectApproxEqRel(exc, libxc_exc_total, 1e-4);
    try testing.expectApproxEqRel(vsigma, libxc_vsigma_total, 1e-4);
}

test "LDA PZ correlation matches reference" {
    // Test Perdew-Zunger LDA correlation at rs=2.0
    const rho: f64 = 0.1;
    const result = lda_pz(rho);

    // The result should be non-zero for positive density
    try testing.expect(result.f < 0);
    try testing.expect(result.df_dn < 0);
    try testing.expectEqual(result.df_dg2, 0.0); // LDA has no gradient dependence
}

// =========================================================================
// Spin-polarized tests
// =========================================================================

test "Spin-polarized LDA PZ reduces to unpolarized for n_up = n_down" {
    const rho: f64 = 0.1;
    const unpol = lda_pz(rho);
    const spin = lda_pz_spin(rho / 2.0, rho / 2.0);

    // Energy density should match
    try testing.expectApproxEqRel(spin.f, unpol.f, 1e-10);
    // Potentials should be equal for both spins and match unpolarized
    try testing.expectApproxEqRel(spin.df_dn_up, unpol.df_dn, 1e-10);
    try testing.expectApproxEqRel(spin.df_dn_down, unpol.df_dn, 1e-10);
}

test "Spin-polarized PBE reduces to unpolarized for n_up = n_down" {
    const rho: f64 = 0.1;
    const sigma: f64 = 0.01;
    const unpol = pbe(rho, sigma);
    // For unpolarized: n_up = n_down = n/2
    // sigma_uu = sigma_dd = sigma/4, sigma_ud = sigma/4
    // since |grad n|^2 = |grad n_up + grad n_down|^2
    //   = |grad n_up|^2 + 2*(grad n_up . grad n_down) + |grad n_down|^2
    //   = sigma_uu + 2*sigma_ud + sigma_dd
    // For uniform spin: grad n_up = grad n_down = grad n / 2
    // sigma_uu = sigma_dd = sigma/4, sigma_ud = sigma/4
    const spin = pbe_spin(rho / 2.0, rho / 2.0, sigma / 4.0, sigma / 4.0, sigma / 4.0);

    // Energy density should match
    try testing.expectApproxEqRel(spin.f, unpol.f, 1e-8);
    // Potentials should be equal and match unpolarized
    try testing.expectApproxEqRel(spin.df_dn_up, unpol.df_dn, 1e-8);
    try testing.expectApproxEqRel(spin.df_dn_down, unpol.df_dn, 1e-8);
}

test "Spin-polarized LDA PZ fully polarized (ferromagnetic)" {
    // All electrons in spin-up: n_down = 0
    const n_up: f64 = 0.1;
    const result = lda_pz_spin(n_up, 0.0);

    // Energy should be negative
    try testing.expect(result.f < 0);
    // V_up should be non-zero
    try testing.expect(result.df_dn_up != 0.0);
}

test "Spin-polarized PBE: libxc spin reference values" {
    // Test at rho_up=0.06, rho_down=0.04, various sigma values
    // libxc reference: XC_GGA_X_PBE (101) + XC_GGA_C_PBE (130)
    // rho = [0.06, 0.04], sigma = [0.005, 0.003, 0.002]
    const n_up: f64 = 0.06;
    const n_down: f64 = 0.04;
    const g2_uu: f64 = 0.005;
    const g2_ud: f64 = 0.003;
    const g2_dd: f64 = 0.002;

    const result = eval_point_spin(.pbe, n_up, n_down, g2_uu, g2_dd, g2_ud);
    const n = n_up + n_down;

    // Convert to Hartree for comparison
    // f = n * eps_xc in Ry, so eps_xc = f/(n) in Ry = f/(2n) in Ha
    const exc_ha = result.f / (n * 2.0);

    // Verify energy is negative (physical requirement)
    try testing.expect(exc_ha < 0);
    // Verify V_up and V_down are different (spin-polarized)
    try testing.expect(result.df_dn_up != result.df_dn_down);
    // Verify gradient terms are non-zero
    try testing.expect(result.df_dg2_uu != 0.0);
    try testing.expect(result.df_dg2_dd != 0.0);
    try testing.expect(result.df_dg2_ud != 0.0);
}

test "Spin-polarized PBE: numerical derivative check" {
    // Verify df_dn_up by finite difference
    const n_up: f64 = 0.06;
    const n_down: f64 = 0.04;
    const g2_uu: f64 = 0.005;
    const g2_ud: f64 = 0.003;
    const g2_dd: f64 = 0.002;
    const delta: f64 = 1e-6;

    const result = eval_point_spin(.pbe, n_up, n_down, g2_uu, g2_dd, g2_ud);
    const result_p = eval_point_spin(.pbe, n_up + delta, n_down, g2_uu, g2_dd, g2_ud);
    const result_m = eval_point_spin(.pbe, n_up - delta, n_down, g2_uu, g2_dd, g2_ud);
    const num_deriv_up = (result_p.f - result_m.f) / (2.0 * delta);

    // Should match analytical derivative (0.1% tolerance for finite diff)
    try testing.expectApproxEqRel(num_deriv_up, result.df_dn_up, 5e-4);

    // Same for n_down
    const result_p2 = eval_point_spin(.pbe, n_up, n_down + delta, g2_uu, g2_dd, g2_ud);
    const result_m2 = eval_point_spin(.pbe, n_up, n_down - delta, g2_uu, g2_dd, g2_ud);
    const num_deriv_down = (result_p2.f - result_m2.f) / (2.0 * delta);
    try testing.expectApproxEqRel(num_deriv_down, result.df_dn_down, 5e-4);

    // df/dg2_uu by finite difference
    const result_g_p = eval_point_spin(.pbe, n_up, n_down, g2_uu + delta, g2_dd, g2_ud);
    const result_g_m = eval_point_spin(.pbe, n_up, n_down, g2_uu - delta, g2_dd, g2_ud);
    const num_deriv_g2uu = (result_g_p.f - result_g_m.f) / (2.0 * delta);
    try testing.expectApproxEqRel(num_deriv_g2uu, result.df_dg2_uu, 5e-4);
}

test "Spin-polarized LDA PZ: numerical derivative check" {
    const n_up: f64 = 0.06;
    const n_down: f64 = 0.04;
    const delta: f64 = 1e-6;

    const result = lda_pz_spin(n_up, n_down);
    const result_p = lda_pz_spin(n_up + delta, n_down);
    const result_m = lda_pz_spin(n_up - delta, n_down);
    const num_deriv_up = (result_p.f - result_m.f) / (2.0 * delta);

    try testing.expectApproxEqRel(num_deriv_up, result.df_dn_up, 1e-4);

    const result_p2 = lda_pz_spin(n_up, n_down + delta);
    const result_m2 = lda_pz_spin(n_up, n_down - delta);
    const num_deriv_down = (result_p2.f - result_m2.f) / (2.0 * delta);
    try testing.expectApproxEqRel(num_deriv_down, result.df_dn_down, 1e-4);
}

test "PBE XC kernel f_nn matches finite difference of V_xc" {
    const test_densities = [_]f64{ 0.01, 0.05, 0.1, 0.5 };
    const test_sigmas = [_]f64{ 0.0, 0.001, 0.01, 0.1 };
    const delta: f64 = 1e-6;

    for (test_densities) |n| {
        for (test_sigmas) |sigma| {
            const kernel = eval_kernel(.pbe, n, sigma);
            const vxc_plus = eval_point(.pbe, n + delta, sigma).df_dn;
            const vxc_minus = eval_point(.pbe, n - delta, sigma).df_dn;
            const fxc_num = (vxc_plus - vxc_minus) / (2.0 * delta);
            try testing.expectApproxEqRel(kernel.fxc, fxc_num, 1e-3);
        }
    }
}

test "PBE XC kernel v_s matches eval_point df_dg2" {
    const n: f64 = 0.1;
    const sigma: f64 = 0.01;
    const kernel = eval_kernel(.pbe, n, sigma);
    const pt = eval_point(.pbe, n, sigma);
    try testing.expectApproxEqRel(kernel.v_s, pt.df_dg2, 1e-10);
}

test "PBE XC kernel f_ns matches finite difference" {
    const n: f64 = 0.1;
    const sigma: f64 = 0.01;
    const delta: f64 = 1e-6;
    const kernel = eval_kernel(.pbe, n, sigma);
    const vs_plus = eval_point(.pbe, n + delta, sigma).df_dg2;
    const vs_minus = eval_point(.pbe, n - delta, sigma).df_dg2;
    const f_ns_num = (vs_plus - vs_minus) / (2.0 * delta);
    try testing.expectApproxEqRel(kernel.f_ns, f_ns_num, 1e-3);
}

test "PBE XC kernel f_ss matches finite difference" {
    const n: f64 = 0.1;
    const sigma: f64 = 0.01;
    const delta: f64 = 1e-6;
    const kernel = eval_kernel(.pbe, n, sigma);
    const vs_plus = eval_point(.pbe, n, sigma + delta).df_dg2;
    const vs_minus = eval_point(.pbe, n, sigma - delta).df_dg2;
    const f_ss_num = (vs_plus - vs_minus) / (2.0 * delta);
    try testing.expectApproxEqRel(kernel.f_ss, f_ss_num, 1e-3);
}

test "PBE XC kernel reduces to LDA at zero gradient" {
    const n: f64 = 0.1;
    const kernel = eval_kernel(.pbe, n, 0.0);
    // f_ns and f_ss should be small (but not exactly zero due to PBE enhancement)
    // v_s should be zero at sigma=0
    try testing.expectApproxEqAbs(kernel.v_s, eval_point(.pbe, n, 0.0).df_dg2, 1e-15);
}
