//! Radial Schrödinger equation solver using Numerov's method on a logarithmic grid.
//!
//! Solves: -u''(r) + [l(l+1)/r² + V(r)] u(r) = E u(r)
//! where u(r) = r * R(r) is the reduced radial wavefunction.
//!
//! For the logarithmic grid r_i = a(exp(b*i) - 1), we transform to the
//! index variable t=i (uniform, step h=1) via φ(i) = u(r_i) * exp(-b*i/2).
//! This gives: φ'' = g(i) φ, where g(i) = rab_i²(V_eff - E) + b²/4.
//!
//! Strategy: inward/outward shooting with matching at classical turning point,
//! bisection on energy to find eigenvalue with correct node count.

const std = @import("std");
const RadialGrid = @import("radial_grid.zig").RadialGrid;

pub const OrbitalQuantumNumbers = struct {
    n: u32, // principal
    l: u32, // angular momentum
};

pub const Solution = struct {
    /// Eigenvalue (Ry)
    energy: f64,
    /// Reduced wavefunction u(r) = r*R(r)
    u: []f64,
    /// Radial wavefunction R(r) = u(r)/r
    R: []f64,
    /// Quantum numbers
    qn: OrbitalQuantumNumbers,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *Solution) void {
        self.allocator.free(self.u);
        self.allocator.free(self.R);
    }
};

const EnergyBracket = struct {
    lo: f64,
    hi: f64,
};

/// Solve the radial Schrödinger equation for a single orbital.
///
/// V_eff(r) should be the total effective potential in Rydberg:
///   V_eff(r) = V_coulomb(r) + V_hartree(r) + V_xc(r)
/// The centrifugal term l(l+1)/r² is added internally.
pub fn solve(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    v_eff: []const f64,
    qn: OrbitalQuantumNumbers,
    energy_guess: f64,
) !Solution {
    const n = grid.n;
    std.debug.assert(v_eff.len == n);

    const required_nodes = qn.n - qn.l - 1;

    const v_full = try build_full_potential(allocator, grid, v_eff, qn.l);
    defer allocator.free(v_full);

    const g = try allocator.alloc(f64, n);
    defer allocator.free(g);

    var bracket = find_energy_bracket(grid, v_full, g, required_nodes, energy_guess);
    bisect_energy_bracket(grid, v_full, g, required_nodes, &bracket);

    const phi_out = try allocator.alloc(f64, n);
    defer allocator.free(phi_out);

    const phi_in = try allocator.alloc(f64, n);
    defer allocator.free(phi_in);

    const u = try allocator.alloc(f64, n);
    errdefer allocator.free(u);

    const energy = run_shooting_refinement(grid, v_full, g, qn.l, &bracket, phi_out, phi_in, u);
    finalize_wavefunction(grid, v_full, g, qn.l, energy, phi_out, phi_in, u);
    normalize_wavefunction(grid, u);

    const R = try build_radial_wavefunction(allocator, grid, u);

    return .{
        .energy = energy,
        .u = u,
        .R = R,
        .qn = qn,
        .allocator = allocator,
    };
}

fn build_full_potential(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    v_eff: []const f64,
    l: u32,
) ![]f64 {
    const v_full = try allocator.alloc(f64, grid.n);
    const ll: f64 = @floatFromInt(l);
    const centrifugal = ll * (ll + 1.0);
    for (0..grid.n) |i| {
        const r = grid.r[i];
        const r2 = if (r > 1e-30) r * r else 1e-30;
        v_full[i] = v_eff[i] + centrifugal / r2;
    }
    return v_full;
}

fn find_energy_bracket(
    grid: *const RadialGrid,
    v_full: []const f64,
    g: []f64,
    required_nodes: u32,
    energy_guess: f64,
) EnergyBracket {
    var bracket = EnergyBracket{
        .lo = energy_guess - 10.0,
        .hi = energy_guess + 10.0,
    };
    for (0..100) |_| {
        build_g(grid, v_full, bracket.lo, g);
        const nodes_lo = count_nodes_outward(g, grid.n);
        build_g(grid, v_full, bracket.hi, g);
        const nodes_hi = count_nodes_outward(g, grid.n);
        if (nodes_lo <= required_nodes and nodes_hi > required_nodes) break;
        if (nodes_lo > required_nodes) bracket.lo -= 10.0;
        if (nodes_hi <= required_nodes) bracket.hi += 10.0;
    }
    return bracket;
}

fn bisect_energy_bracket(
    grid: *const RadialGrid,
    v_full: []const f64,
    g: []f64,
    required_nodes: u32,
    bracket: *EnergyBracket,
) void {
    for (0..100) |_| {
        const e_mid = 0.5 * (bracket.lo + bracket.hi);
        build_g(grid, v_full, e_mid, g);
        const nodes = count_nodes_outward(g, grid.n);
        if (nodes > required_nodes) {
            bracket.hi = e_mid;
        } else {
            bracket.lo = e_mid;
        }
        if (bracket.hi - bracket.lo < 1e-14) break;
    }
}

fn run_shooting_refinement(
    grid: *const RadialGrid,
    v_full: []const f64,
    g: []f64,
    l: u32,
    bracket: *const EnergyBracket,
    phi_out: []f64,
    phi_in: []f64,
    u: []f64,
) f64 {
    var energy = 0.5 * (bracket.lo + bracket.hi);
    for (0..300) |_| {
        build_g(grid, v_full, energy, g);
        const i_match = find_turning_point(v_full, energy, grid.n);
        numerov_outward(g, l, grid, phi_out, grid.n);
        numerov_inward(g, energy, grid, phi_in, grid.n);
        if (!match_wavefunctions(phi_out, phi_in, i_match)) break;

        build_reduced_wavefunction(grid, phi_out, phi_in, i_match, u);
        const norm = compute_wavefunction_norm(grid, u);
        if (norm < 1e-30) break;

        const de = energy_correction(grid, phi_out, phi_in, i_match, norm);
        energy = @max(bracket.lo, @min(bracket.hi, energy + de));
        if (@abs(de) < 1e-12) break;
    }
    return energy;
}

fn finalize_wavefunction(
    grid: *const RadialGrid,
    v_full: []const f64,
    g: []f64,
    l: u32,
    energy: f64,
    phi_out: []f64,
    phi_in: []f64,
    u: []f64,
) void {
    build_g(grid, v_full, energy, g);
    const i_match = find_turning_point(v_full, energy, grid.n);
    numerov_outward(g, l, grid, phi_out, grid.n);
    numerov_inward(g, energy, grid, phi_in, grid.n);
    _ = match_wavefunctions(phi_out, phi_in, i_match);
    build_reduced_wavefunction(grid, phi_out, phi_in, i_match, u);
}

fn match_wavefunctions(phi_out: []const f64, phi_in: []f64, i_match: usize) bool {
    if (@abs(phi_in[i_match]) < 1e-30) return false;
    const scale = phi_out[i_match] / phi_in[i_match];
    for (phi_in) |*phi| {
        phi.* *= scale;
    }
    return true;
}

fn build_reduced_wavefunction(
    grid: *const RadialGrid,
    phi_out: []const f64,
    phi_in: []const f64,
    i_match: usize,
    u: []f64,
) void {
    for (0..grid.n) |i| {
        const phi = if (i <= i_match) phi_out[i] else phi_in[i];
        u[i] = phi * @exp(grid.b * @as(f64, @floatFromInt(i)) * 0.5);
    }
}

fn compute_wavefunction_norm(grid: *const RadialGrid, u: []const f64) f64 {
    var norm: f64 = 0;
    for (0..grid.n) |i| {
        norm += ctrap_weight(i, grid.n) * u[i] * u[i] * grid.rab[i];
    }
    return norm;
}

fn energy_correction(
    grid: *const RadialGrid,
    phi_out: []const f64,
    phi_in: []const f64,
    i_match: usize,
    norm: f64,
) f64 {
    const dphi_out = phi_out[i_match + 1] - phi_out[i_match - 1];
    const dphi_in = phi_in[i_match + 1] - phi_in[i_match - 1];
    const mismatch = dphi_out - dphi_in;
    const match_scale = @exp(grid.b * @as(f64, @floatFromInt(i_match)));
    return -mismatch * phi_out[i_match] * match_scale / norm;
}

fn normalize_wavefunction(grid: *const RadialGrid, u: []f64) void {
    const norm = compute_wavefunction_norm(grid, u);
    const inv_sqrt_norm = 1.0 / @sqrt(norm);
    for (u) |*u_i| {
        u_i.* *= inv_sqrt_norm;
    }
}

fn build_radial_wavefunction(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    u: []const f64,
) ![]f64 {
    const R = try allocator.alloc(f64, grid.n);
    for (0..grid.n) |i| {
        if (grid.r[i] > 1e-30) {
            R[i] = u[i] / grid.r[i];
        } else {
            R[i] = if (grid.n > 1) u[1] / grid.r[1] else 0;
        }
    }
    return R;
}

/// Build Numerov g-function for the transformed equation φ'' = g(i)φ.
/// g(i) = rab_i² × (V_eff(r_i) - E) + b²/4
fn build_g(grid: *const RadialGrid, v_full: []const f64, energy: f64, g: []f64) void {
    const b2_4 = grid.b * grid.b * 0.25;
    for (0..grid.n) |i| {
        g[i] = grid.rab[i] * grid.rab[i] * (v_full[i] - energy) + b2_4;
    }
}

/// Outward Numerov integration in φ-space (h=1).
/// Boundary: φ(0) = 0, φ(1) = r_1^{l+1} * exp(-b/2).
fn numerov_outward(g: []const f64, l: u32, grid: *const RadialGrid, phi: []f64, n: usize) void {
    phi[0] = 0;
    const r1 = grid.r[1];
    const fl: f64 = @floatFromInt(l);
    phi[1] = std.math.pow(f64, r1, fl + 1.0) * @exp(-grid.b * 0.5);

    for (2..n) |i| {
        const denom = 1.0 - g[i] / 12.0;
        if (@abs(denom) < 1e-30) {
            phi[i] = phi[i - 1];
            continue;
        }
        phi[i] = (2.0 * phi[i - 1] * (1.0 + 5.0 * g[i - 1] / 12.0) -
            phi[i - 2] * (1.0 - g[i - 2] / 12.0)) / denom;
    }
}

/// Inward Numerov integration in φ-space (h=1).
/// Boundary: φ(n-1) ~ exp(-κr) * exp(-b(n-1)/2).
fn numerov_inward(g: []const f64, energy: f64, grid: *const RadialGrid, phi: []f64, n: usize) void {
    const kappa = if (energy < 0) @sqrt(-energy) else 1.0;

    const fi_last: f64 = @floatFromInt(n - 1);
    const fi_prev: f64 = @floatFromInt(n - 2);
    phi[n - 1] = @exp(-kappa * grid.r[n - 1] - grid.b * fi_last * 0.5);
    phi[n - 2] = @exp(-kappa * grid.r[n - 2] - grid.b * fi_prev * 0.5);

    if (n < 3) return;
    var i: usize = n - 3;
    while (true) {
        const denom = 1.0 - g[i] / 12.0;
        if (@abs(denom) < 1e-30) {
            phi[i] = phi[i + 1];
        } else {
            phi[i] = (2.0 * phi[i + 1] * (1.0 + 5.0 * g[i + 1] / 12.0) -
                phi[i + 2] * (1.0 - g[i + 2] / 12.0)) / denom;
        }
        if (i == 0) break;
        i -= 1;
    }
}

/// Count nodes of outward-integrated φ.
fn count_nodes_outward(g: []const f64, n: usize) u32 {
    var phi_prev: f64 = 0;
    var phi_curr: f64 = 1e-10;
    var nodes: u32 = 0;

    for (2..n) |i| {
        const denom = 1.0 - g[i] / 12.0;
        if (@abs(denom) < 1e-30) break;
        const phi_next = (2.0 * phi_curr * (1.0 + 5.0 * g[i - 1] / 12.0) -
            phi_prev * (1.0 - g[i - 2] / 12.0)) / denom;

        if (phi_next * phi_curr < 0) nodes += 1;
        phi_prev = phi_curr;
        phi_curr = phi_next;

        // Stop well into classically forbidden region
        if (g[i] > 10.0 and i > n / 4) break;
    }
    return nodes;
}

/// Find classical turning point: outermost r where V_eff(r) < E.
fn find_turning_point(v_full: []const f64, energy: f64, n: usize) usize {
    var i_turn: usize = n * 3 / 4;
    var i: usize = n - 2;
    while (i > 1) : (i -= 1) {
        if (v_full[i] < energy) {
            i_turn = i;
            break;
        }
    }
    if (i_turn < 5) i_turn = 5;
    if (i_turn > n - 5) i_turn = n - 5;
    return i_turn;
}

/// Newton-Cotes endpoint correction.
fn ctrap_weight(i: usize, n: usize) f64 {
    const endpoint_weights = [5]f64{
        23.75 / 72.0,
        95.10 / 72.0,
        55.20 / 72.0,
        79.30 / 72.0,
        70.65 / 72.0,
    };
    if (n < 10) {
        if (i == 0 or i == n - 1) return 0.5;
        return 1.0;
    }
    if (i < 5) return endpoint_weights[i];
    if (i >= n - 5) return endpoint_weights[n - 1 - i];
    return 1.0;
}

// ============================================================================
// Tests
// ============================================================================

test "hydrogen 1s eigenvalue" {
    // H atom: V(r) = -2/r (Rydberg), E_1s = -1.0 Ry
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 2000, 1e-7, 80.0);
    defer grid.deinit();

    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    var sol = try solve(allocator, &grid, v, .{ .n = 1, .l = 0 }, -0.5);
    defer sol.deinit();

    // E_1s = -1.0 Ry
    try std.testing.expectApproxEqAbs(-1.0, sol.energy, 1e-6);
}

test "hydrogen 2s eigenvalue" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 2000, 1e-7, 120.0);
    defer grid.deinit();

    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    var sol = try solve(allocator, &grid, v, .{ .n = 2, .l = 0 }, -0.2);
    defer sol.deinit();

    // E_2s = -0.25 Ry
    try std.testing.expectApproxEqAbs(-0.25, sol.energy, 1e-5);
}

test "hydrogen 2p eigenvalue" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 2000, 1e-7, 120.0);
    defer grid.deinit();

    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    var sol = try solve(allocator, &grid, v, .{ .n = 2, .l = 1 }, -0.2);
    defer sol.deinit();

    // E_2p = -0.25 Ry
    try std.testing.expectApproxEqAbs(-0.25, sol.energy, 1e-5);
}
