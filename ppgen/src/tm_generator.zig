//! Troullier-Martins norm-conserving pseudopotential generator.
//!
//! Inside rc: φ̃(r) = r^{l+1} exp(p(r)) where p(r) = c0 + c2*r² + c4*r⁴ + c6*r⁶ + ... + c12*r¹²
//! Outside rc: φ̃(r) = φ_AE(r)
//!
//! Constraints:
//!   1. Norm conservation: ∫₀^rc |φ̃|² r² dr = ∫₀^rc |φ_AE|² r² dr
//!   2. Continuity at rc: φ̃, φ̃', φ̃'', φ̃''', φ̃'''' all match φ_AE
//!   3. V_ps(r=0) is finite (from the Schrödinger equation inversion)
//!
//! Reference: Troullier & Martins, PRB 43, 1993 (1991)

const std = @import("std");
const RadialGrid = @import("radial_grid.zig").RadialGrid;

pub const PseudoWavefunction = struct {
    /// Pseudo wavefunction u_ps(r) = r * R_ps(r)
    u: []f64,
    /// Screened pseudopotential V_ps,l(r) (Ry) — includes Hartree + XC
    v_ps: []f64,
    /// Cutoff radius (Bohr)
    rc: f64,
    /// Angular momentum
    l: u32,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *PseudoWavefunction) void {
        self.allocator.free(self.u);
        self.allocator.free(self.v_ps);
    }
};

/// Generate a Troullier-Martins pseudo wavefunction for a single channel.
///
/// u_ae: all-electron reduced wavefunction u(r) = r*R(r)
/// v_eff: effective potential V_eff(r) used to generate u_ae (Ry)
/// energy: eigenvalue of the all-electron state (Ry)
/// l: angular momentum quantum number
/// rc: cutoff radius (Bohr)
pub fn generate(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    u_ae: []const f64,
    v_eff: []const f64,
    energy: f64,
    l: u32,
    rc: f64,
) !PseudoWavefunction {
    const i_rc = find_grid_index(grid, rc);
    const rc_actual = grid.r[i_rc];
    const norm_ae = integrate_norm(grid, u_ae, i_rc);
    const ae_at_rc = wavefunction_derivatives(grid, u_ae, i_rc);
    const coeffs = try solve_coefficients(ae_at_rc, rc_actual, l, norm_ae);

    const u_ps = try build_pseudo_wavefunction(allocator, grid, u_ae, coeffs, l, i_rc);
    errdefer allocator.free(u_ps);
    const v_ps = try build_pseudo_potential(allocator, grid, v_eff, coeffs, energy, l, i_rc);

    return .{
        .u = u_ps,
        .v_ps = v_ps,
        .rc = rc_actual,
        .l = l,
        .allocator = allocator,
    };
}

fn build_pseudo_wavefunction(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    u_ae: []const f64,
    coeffs: TmCoefficients,
    l: u32,
    i_rc: usize,
) ![]f64 {
    const u_ps = try allocator.alloc(f64, grid.n);
    const fl: f64 = @floatFromInt(l);
    for (0..grid.n) |i| {
        if (i <= i_rc) {
            const r = grid.r[i];
            u_ps[i] = std.math.pow(f64, r, fl + 1.0) * @exp(tm_polynomial(coeffs, r));
        } else {
            u_ps[i] = u_ae[i];
        }
    }
    return u_ps;
}

fn build_pseudo_potential(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    v_eff: []const f64,
    coeffs: TmCoefficients,
    energy: f64,
    l: u32,
    i_rc: usize,
) ![]f64 {
    const v_ps = try allocator.alloc(f64, grid.n);
    const fl: f64 = @floatFromInt(l);
    for (0..grid.n) |i| {
        if (i <= i_rc) {
            v_ps[i] = tm_potential_value(coeffs, grid.r[i], fl, energy);
        } else {
            v_ps[i] = v_eff[i];
        }
    }
    return v_ps;
}

fn tm_polynomial(coeffs: TmCoefficients, r: f64) f64 {
    const r2 = r * r;
    const r4 = r2 * r2;
    const r6 = r4 * r2;
    return coeffs.c0 + coeffs.c2 * r2 + coeffs.c4 * r4 + coeffs.c6 * r6 +
        coeffs.c8 * r4 * r4 + coeffs.c10 * r6 * r4 + coeffs.c12 * r6 * r6;
}

fn tm_potential_value(coeffs: TmCoefficients, r: f64, fl: f64, energy: f64) f64 {
    const r2 = r * r;
    const r4 = r2 * r2;
    const r6 = r4 * r2;
    const r8 = r4 * r4;
    const r10 = r8 * r2;
    const p_prime_over_r = 2.0 * coeffs.c2 + 4.0 * coeffs.c4 * r2 +
        6.0 * coeffs.c6 * r4 + 8.0 * coeffs.c8 * r6 + 10.0 * coeffs.c10 * r8 +
        12.0 * coeffs.c12 * r10;
    const p_prime = r * p_prime_over_r;
    const p_double_prime = 2.0 * coeffs.c2 + 12.0 * coeffs.c4 * r2 +
        30.0 * coeffs.c6 * r4 + 56.0 * coeffs.c8 * r6 + 90.0 * coeffs.c10 * r8 +
        132.0 * coeffs.c12 * r10;
    return energy + p_double_prime + 2.0 * (fl + 1.0) * p_prime_over_r + p_prime * p_prime;
}

const TmCoefficients = struct {
    c0: f64,
    c2: f64,
    c4: f64,
    c6: f64,
    c8: f64,
    c10: f64,
    c12: f64,
};

/// Values and derivatives of u(r) at a grid point, extracted via finite differences.
const WfDerivatives = struct {
    u: f64, // u(rc)
    du: f64, // du/dr at rc
    d2u: f64, // d²u/dr² at rc
    d3u: f64, // d³u/dr³ at rc
    d4u: f64, // d⁴u/dr⁴ at rc
};

fn wavefunction_derivatives(grid: *const RadialGrid, u: []const f64, i: usize) WfDerivatives {
    // 5-point stencil for derivatives
    const n = grid.n;
    std.debug.assert(i >= 2 and i + 2 < n);

    const r = grid.r;
    const h1 = r[i] - r[i - 1];
    const h2 = r[i + 1] - r[i];

    // Use polynomial interpolation for non-uniform grid
    // Central finite differences with variable spacing
    const u_m2 = u[i - 2];
    const u_m1 = u[i - 1];
    const u_0 = u[i];
    const u_p1 = u[i + 1];
    const u_p2 = u[i + 2];

    // Average spacing for higher derivatives
    const h = 0.5 * (h1 + h2);
    _ = u_m2;
    _ = u_p2;

    const du = (u_p1 - u_m1) / (2.0 * h);
    const d2u = (u_p1 - 2.0 * u_0 + u_m1) / (h * h);

    // 3rd and 4th derivatives using wider stencil
    const h_wide = r[i + 1] - r[i - 1];
    const h_avg = h_wide / 2.0;
    const d3u =
        (u[i + 2] - 2.0 * u[i + 1] + 2.0 * u[i - 1] - u[i - 2]) / (2.0 * h_avg * h_avg * h_avg);
    const d4u =
        (u[i + 2] - 4.0 * u[i + 1] + 6.0 * u[i] - 4.0 * u[i - 1] + u[i - 2]) /
        (h_avg * h_avg * h_avg * h_avg);

    return .{
        .u = u_0,
        .du = du,
        .d2u = d2u,
        .d3u = d3u,
        .d4u = d4u,
    };
}

const TmPowers = struct {
    rc: f64,
    rc2: f64,
    rc4: f64,
    rc6: f64,
    rc8: f64,
    rc10: f64,

    fn init(rc: f64) TmPowers {
        const rc2 = rc * rc;
        const rc4 = rc2 * rc2;
        const rc6 = rc4 * rc2;
        return .{
            .rc = rc,
            .rc2 = rc2,
            .rc4 = rc4,
            .rc6 = rc6,
            .rc8 = rc4 * rc4,
            .rc10 = rc6 * rc4,
        };
    }
};

const TmMatch = struct {
    f0: f64,
    f1: f64,
    f2: f64,
    f3: f64,
    f4: f64,

    fn is_finite(self: TmMatch) bool {
        return std.math.isFinite(self.f0) and std.math.isFinite(self.f1) and
            std.math.isFinite(self.f2) and std.math.isFinite(self.f3) and
            std.math.isFinite(self.f4);
    }
};

fn tm_matching_values(ae: WfDerivatives, powers: TmPowers, l: u32) !TmMatch {
    if (!std.math.isFinite(ae.u) or @abs(ae.u) <= 1e-30) {
        return error.InvalidTmCoefficients;
    }
    const fl: f64 = @floatFromInt(l);
    const u_rc = ae.u;
    const du_rc = ae.du / u_rc;
    const d2u_rc = ae.d2u / u_rc;
    const d3u_rc = ae.d3u / u_rc;
    const d4u_rc = ae.d4u / u_rc;

    const result = TmMatch{
        .f0 = @log(@abs(u_rc)) - (fl + 1.0) * @log(powers.rc),
        .f1 = du_rc - (fl + 1.0) / powers.rc,
        .f2 = d2u_rc - du_rc * du_rc + (fl + 1.0) / powers.rc2,
        .f3 = d3u_rc - 3.0 * du_rc * d2u_rc + 2.0 * du_rc * du_rc * du_rc -
            2.0 * (fl + 1.0) / (powers.rc2 * powers.rc),
        .f4 = d4u_rc - 4.0 * du_rc * d3u_rc - 3.0 * d2u_rc * d2u_rc +
            12.0 * du_rc * du_rc * d2u_rc - 6.0 * du_rc * du_rc * du_rc * du_rc +
            6.0 * (fl + 1.0) / powers.rc4,
    };
    if (!result.is_finite()) return error.InvalidTmCoefficients;
    return result;
}

/// Solve for TM polynomial coefficients.
fn solve_coefficients(
    ae: WfDerivatives,
    rc: f64,
    l: u32,
    norm_ae: f64,
) !TmCoefficients {
    if (!std.math.isFinite(rc) or rc <= 0.0) return error.InvalidTmCoefficients;
    if (!std.math.isFinite(norm_ae) or norm_ae <= 0.0) return error.InvalidTmCoefficients;
    const powers = TmPowers.init(rc);
    const m = try tm_matching_values(ae, powers, l);

    const c2_guesses = [_]f64{ 0.0, -0.25, 0.25, -1.0, 1.0, -4.0, 4.0, -12.0, 12.0 };
    for (c2_guesses) |c2| {
        if (solve_coefficients_from_state(
            .{ .c0 = m.f0 - m.f1 * rc / 2.0, .c2 = c2 },
            m,
            powers,
            l,
            norm_ae,
        )) |coeffs| return coeffs else |_| continue;
    }

    return error.InvalidTmCoefficients;
}

fn solve_coefficients_from_state(
    initial_state: TmNonlinearState,
    m: TmMatch,
    powers: TmPowers,
    l: u32,
    norm_ae: f64,
) !TmCoefficients {
    var state = initial_state;
    var residual = try evaluate_tm_residual(state, m, powers, l, norm_ae);
    for (0..50) |_| {
        if (residual.converged()) return residual.coeffs;
        const step = try nonlinear_step(state, residual, m, powers, l, norm_ae);
        state = try damped_tm_step(state, step, residual, m, powers, l, norm_ae);
        residual = try evaluate_tm_residual(state, m, powers, l, norm_ae);
    }

    return error.InvalidTmCoefficients;
}

const TmNonlinearState = struct {
    c0: f64,
    c2: f64,
};

const TmResidual = struct {
    coeffs: TmCoefficients,
    norm: f64,
    curvature: f64,

    fn converged(self: TmResidual) bool {
        return @abs(self.norm) < 1e-11 and @abs(self.curvature) < 1e-11;
    }
};

fn evaluate_tm_residual(
    state: TmNonlinearState,
    m: TmMatch,
    powers: TmPowers,
    l: u32,
    norm_ae: f64,
) !TmResidual {
    const coeffs = try coefficients_from_c0_c2(state.c0, state.c2, m, powers);
    const norm_ps = compute_pseudo_norm(coeffs, powers.rc, l, 4000);
    if (!std.math.isFinite(norm_ps)) return error.InvalidTmCoefficients;
    return .{
        .coeffs = coeffs,
        .norm = norm_ps - norm_ae,
        .curvature = tm_zero_curvature_residual(coeffs, l),
    };
}

fn nonlinear_step(
    state: TmNonlinearState,
    residual: TmResidual,
    m: TmMatch,
    powers: TmPowers,
    l: u32,
    norm_ae: f64,
) !TmNonlinearState {
    const dc0 = 1e-6 * @max(1.0, @abs(state.c0));
    const dc2 = 1e-6 * @max(1.0, @abs(state.c2));
    const res_c0 = try evaluate_tm_residual(
        .{ .c0 = state.c0 + dc0, .c2 = state.c2 },
        m,
        powers,
        l,
        norm_ae,
    );
    const res_c2 = try evaluate_tm_residual(
        .{ .c0 = state.c0, .c2 = state.c2 + dc2 },
        m,
        powers,
        l,
        norm_ae,
    );

    const a = (res_c0.norm - residual.norm) / dc0;
    const b = (res_c2.norm - residual.norm) / dc2;
    const c = (res_c0.curvature - residual.curvature) / dc0;
    const d = (res_c2.curvature - residual.curvature) / dc2;
    const det = a * d - b * c;
    if (!std.math.isFinite(det) or @abs(det) < 1e-20) return error.InvalidTmCoefficients;
    return .{
        .c0 = (-residual.norm * d + b * residual.curvature) / det,
        .c2 = (c * residual.norm - a * residual.curvature) / det,
    };
}

fn damped_tm_step(
    state: TmNonlinearState,
    step: TmNonlinearState,
    residual: TmResidual,
    m: TmMatch,
    powers: TmPowers,
    l: u32,
    norm_ae: f64,
) !TmNonlinearState {
    const current_norm = residual_norm(residual);
    var damping: f64 = 1.0;
    var last_valid = state;
    while (damping >= 1.0 / 1024.0) : (damping *= 0.5) {
        const candidate = TmNonlinearState{
            .c0 = state.c0 + damping * step.c0,
            .c2 = state.c2 + damping * step.c2,
        };
        const candidate_residual =
            evaluate_tm_residual(candidate, m, powers, l, norm_ae) catch continue;
        last_valid = candidate;
        if (residual_norm(candidate_residual) < current_norm) return candidate;
    }
    return last_valid;
}

fn residual_norm(residual: TmResidual) f64 {
    return @abs(residual.norm) + @abs(residual.curvature);
}

fn tm_zero_curvature_residual(coeffs: TmCoefficients, l: u32) f64 {
    const fl: f64 = @floatFromInt(l);
    return coeffs.c4 + coeffs.c2 * coeffs.c2 / (2.0 * fl + 5.0);
}

/// Given c0 and c2, compute the remaining coefficients from rc matching.
fn coefficients_from_c0_c2(c0: f64, c2: f64, m: TmMatch, p: TmPowers) !TmCoefficients {
    const p12 = p.rc6 * p.rc6;
    const p11 = p.rc10 * p.rc;
    const p9 = p.rc8 * p.rc;
    const p7 = p.rc6 * p.rc;
    const p5 = p.rc4 * p.rc;
    const p3 = p.rc2 * p.rc;
    const rhs = [5]f64{
        m.f0 - c0 - c2 * p.rc2,
        m.f1 - 2.0 * c2 * p.rc,
        m.f2 - 2.0 * c2,
        m.f3,
        m.f4,
    };

    var a = [5][5]f64{
        .{ p.rc4, p.rc6, p.rc8, p.rc10, p12 },
        .{
            4.0 * p3,
            6.0 * p5,
            8.0 * p7,
            10.0 * p9,
            12.0 * p11,
        },
        .{ 12.0 * p.rc2, 30.0 * p.rc4, 56.0 * p.rc6, 90.0 * p.rc8, 132.0 * p.rc10 },
        .{ 24.0 * p.rc, 120.0 * p3, 336.0 * p5, 720.0 * p7, 1320.0 * p9 },
        .{ 24.0, 360.0 * p.rc2, 1680.0 * p.rc4, 5040.0 * p.rc6, 11880.0 * p.rc8 },
    };

    var b = rhs;
    const x = try solve_5x5(&a, &b);
    return .{
        .c0 = c0,
        .c2 = c2,
        .c4 = x[0],
        .c6 = x[1],
        .c8 = x[2],
        .c10 = x[3],
        .c12 = x[4],
    };
}

fn solve_5x5(a: *[5][5]f64, b: *[5]f64) ![5]f64 {
    for (0..5) |col| {
        var max_val: f64 = 0;
        var max_row: usize = col;
        for (col..5) |row| {
            if (@abs(a.*[row][col]) > max_val) {
                max_val = @abs(a.*[row][col]);
                max_row = row;
            }
        }

        if (max_row != col) {
            const tmp_a = a.*[col];
            a.*[col] = a.*[max_row];
            a.*[max_row] = tmp_a;
            const tmp_b = b.*[col];
            b.*[col] = b.*[max_row];
            b.*[max_row] = tmp_b;
        }

        if (!std.math.isFinite(a.*[col][col]) or @abs(a.*[col][col]) < 1e-30) {
            return error.InvalidTmCoefficients;
        }
        for (col + 1..5) |row| {
            const factor = a.*[row][col] / a.*[col][col];
            for (col..5) |j| {
                a.*[row][j] -= factor * a.*[col][j];
            }
            b.*[row] -= factor * b.*[col];
        }
    }

    var x = [5]f64{ 0, 0, 0, 0, 0 };
    var i: usize = 5;
    while (i > 0) {
        i -= 1;
        var sum: f64 = b.*[i];
        for (i + 1..5) |j| {
            sum -= a.*[i][j] * x[j];
        }
        if (!std.math.isFinite(a.*[i][i]) or @abs(a.*[i][i]) < 1e-30) {
            return error.InvalidTmCoefficients;
        }
        x[i] = sum / a.*[i][i];
        if (!std.math.isFinite(x[i])) return error.InvalidTmCoefficients;
    }
    return x;
}

/// Compute ∫₀^rc |u_ps|² dr for the TM pseudo wavefunction.
/// Uses Gauss-Legendre quadrature on [0, rc].
fn compute_pseudo_norm(coeffs: TmCoefficients, rc: f64, l: u32, n_quad: usize) f64 {
    const fl: f64 = @floatFromInt(l);
    var sum: f64 = 0;
    // Simple trapezoidal on uniform sub-grid
    const dr = rc / @as(f64, @floatFromInt(n_quad));
    for (0..n_quad + 1) |i| {
        const r = @as(f64, @floatFromInt(i)) * dr;
        const u = std.math.pow(f64, r, fl + 1.0) * @exp(tm_polynomial(coeffs, r));
        const w: f64 = if (i == 0 or i == n_quad) 0.5 else 1.0;
        sum += w * u * u * dr;
    }
    return sum;
}

/// Compute ∫₀^{i_rc} |u|² dr using trapezoidal rule on the radial grid.
fn integrate_norm(grid: *const RadialGrid, u: []const f64, i_rc: usize) f64 {
    var sum: f64 = 0;
    for (1..i_rc + 1) |i| {
        const f_prev = u[i - 1] * u[i - 1] * grid.rab[i - 1];
        const f_curr = u[i] * u[i] * grid.rab[i];
        sum += 0.5 * (f_prev + f_curr);
    }
    return sum;
}

/// Find grid index closest to target radius.
pub fn find_grid_index(grid: *const RadialGrid, r_target: f64) usize {
    var best: usize = 0;
    var best_diff: f64 = @abs(grid.r[0] - r_target);
    for (1..grid.n) |i| {
        const diff = @abs(grid.r[i] - r_target);
        if (diff < best_diff) {
            best_diff = diff;
            best = i;
        }
    }
    return best;
}

// ============================================================================
// Tests
// ============================================================================

const schrodinger = @import("schrodinger.zig");
const RadialGridMod = @import("radial_grid.zig");

test "TM pseudo wavefunction: continuity at rc" {
    // Generate H 1s all-electron, then pseudize with TM.
    // Verify: u_ps(rc) = u_ae(rc) and smooth behavior.
    const allocator = std.testing.allocator;
    var grid = try RadialGridMod.RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    // Bare Coulomb potential for H
    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    // Solve for 1s
    var sol = try schrodinger.solve(allocator, &grid, v, .{ .n = 1, .l = 0 }, -0.5);
    defer sol.deinit();

    // Generate TM pseudo wavefunction with rc = 1.5 Bohr
    var pw = try generate(allocator, &grid, sol.u, v, sol.energy, 0, 1.5);
    defer pw.deinit();

    // Check continuity at rc
    const i_rc = find_grid_index(&grid, pw.rc);
    try std.testing.expectApproxEqAbs(sol.u[i_rc], pw.u[i_rc], 1e-10);

    // u_ps must equal u_ae outside rc
    try std.testing.expectApproxEqAbs(sol.u[i_rc + 5], pw.u[i_rc + 5], 1e-15);

    // u_ps should be smooth and nodeless inside rc
    for (1..i_rc) |i| {
        try std.testing.expect(pw.u[i] > 0); // no nodes for s-wave
    }
}

test "TM pseudo wavefunction: norm conservation" {
    const allocator = std.testing.allocator;
    var grid = try RadialGridMod.RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    var sol = try schrodinger.solve(allocator, &grid, v, .{ .n = 1, .l = 0 }, -0.5);
    defer sol.deinit();

    var pw = try generate(allocator, &grid, sol.u, v, sol.energy, 0, 1.5);
    defer pw.deinit();

    const i_rc = find_grid_index(&grid, pw.rc);

    // Norm inside rc must match
    const norm_ae = integrate_norm(&grid, sol.u, i_rc);
    const norm_ps = integrate_norm(&grid, pw.u, i_rc);

    try std.testing.expectApproxEqAbs(norm_ae, norm_ps, 1e-6);
}

test "TM coefficient solve enforces norm and zero curvature" {
    const allocator = std.testing.allocator;
    var grid = try RadialGridMod.RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    var sol = try schrodinger.solve(allocator, &grid, v, .{ .n = 1, .l = 0 }, -0.5);
    defer sol.deinit();

    const i_rc = find_grid_index(&grid, 1.5);
    const rc = grid.r[i_rc];
    const norm_ae = integrate_norm(&grid, sol.u, i_rc);
    const ae_at_rc = wavefunction_derivatives(&grid, sol.u, i_rc);
    const coeffs = try solve_coefficients(ae_at_rc, rc, 0, norm_ae);

    const norm_ps = compute_pseudo_norm(coeffs, rc, 0, 4000);
    try std.testing.expectApproxEqAbs(norm_ae, norm_ps, 1e-9);
    try std.testing.expectApproxEqAbs(0.0, tm_zero_curvature_residual(coeffs, 0), 1e-9);
    try std.testing.expect(std.math.isFinite(coeffs.c12));
    try std.testing.expect(@abs(coeffs.c12) > 1e-14);
}

test "TM coefficient solve rejects singular matching system" {
    const ae = WfDerivatives{
        .u = 1.0,
        .du = 0.0,
        .d2u = 0.0,
        .d3u = 0.0,
        .d4u = 0.0,
    };
    try std.testing.expectError(error.InvalidTmCoefficients, solve_coefficients(ae, 0.0, 0, 1.0));
    try std.testing.expectError(error.InvalidTmCoefficients, solve_coefficients(ae, 1.0, 0, 0.0));
}

test "TM screened pseudopotential: finite at origin" {
    const allocator = std.testing.allocator;
    var grid = try RadialGridMod.RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    var sol = try schrodinger.solve(allocator, &grid, v, .{ .n = 1, .l = 0 }, -0.5);
    defer sol.deinit();

    var pw = try generate(allocator, &grid, sol.u, v, sol.energy, 0, 1.5);
    defer pw.deinit();

    // V_ps should be finite everywhere inside rc (no 1/r divergence)
    const i_rc = find_grid_index(&grid, pw.rc);
    for (0..i_rc) |i| {
        try std.testing.expect(std.math.isFinite(pw.v_ps[i]));
        try std.testing.expect(@abs(pw.v_ps[i]) < 100.0); // reasonable bound
    }
}
