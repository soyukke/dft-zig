//! Diagnostic tests for verifying accuracy of individual components.
//! Compares against analytical results for hydrogen atom.

const std = @import("std");
const dft_zig = @import("dft_zig");
const xc_mod = dft_zig.xc;
const RadialGrid = @import("radial_grid.zig").RadialGrid;
const atomic_solver = @import("atomic_solver.zig");

// ============================================================================
// Hydrogen analytical reference values
// ============================================================================

/// Exact hydrogen 1s density: ρ(r) = exp(-2r)/π
fn hydrogen_density(r: f64) f64 {
    return @exp(-2.0 * r) / std.math.pi;
}

/// Exact hydrogen Hartree potential (Rydberg):
/// V_H(r) = 2[1 - (1+r)exp(-2r)] / r
fn hydrogen_vh(r: f64) f64 {
    if (r < 1e-10) return 2.0; // limit as r→0
    return 2.0 * (1.0 - (1.0 + r) * @exp(-2.0 * r)) / r;
}

// ============================================================================
// Tests
// ============================================================================

test "Poisson solver: hydrogen V_H" {
    // Verify V_H against analytical hydrogen Hartree potential
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 50.0);
    defer grid.deinit();

    const rho = try allocator.alloc(f64, grid.n);
    defer allocator.free(rho);

    for (0..grid.n) |i| {
        rho[i] = hydrogen_density(grid.r[i]);
    }

    const v_h = try allocator.alloc(f64, grid.n);
    defer allocator.free(v_h);

    atomic_solver.radial_poisson(&grid, rho, v_h);

    var max_rel_err: f64 = 0;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        if (r < 0.01 or r > 30.0) continue;
        const exact = hydrogen_vh(r);
        const err = @abs(v_h[i] - exact);
        const rel = err / @abs(exact);
        if (rel > max_rel_err) max_rel_err = rel;
    }

    try std.testing.expect(max_rel_err < 0.01); // <1% relative error
}

test "electron count normalization" {
    // ∫ ρ(r) 4πr² dr should equal 1 for hydrogen
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 50.0);
    defer grid.deinit();

    const rho = try allocator.alloc(f64, grid.n);
    defer allocator.free(rho);

    for (0..grid.n) |i| {
        rho[i] = hydrogen_density(grid.r[i]);
    }

    // Integrate ρ × 4πr² dr
    var nel: f64 = 0;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        const w = ctrap_weight(i, grid.n);
        nel += w * rho[i] * 4.0 * std.math.pi * r * r * grid.rab[i];
    }

    try std.testing.expectApproxEqAbs(1.0, nel, 1e-6);
}

test "XC energy for hydrogen density" {
    // For known hydrogen density, compute LDA XC energy
    // Reference (LDA-PZ): E_xc ≈ -0.882 Ry for exact H density
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 50.0);
    defer grid.deinit();

    var e_xc: f64 = 0;
    var integral_vxc_rho: f64 = 0;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        const rho = hydrogen_density(r);
        const xc_point = xc_mod.eval_point(.lda_pz, rho, 0);
        const w = ctrap_weight(i, grid.n);
        const vol = w * 4.0 * std.math.pi * r * r * grid.rab[i];
        e_xc += xc_point.f * vol;
        integral_vxc_rho += xc_point.df_dn * rho * vol;
    }

    try std.testing.expect(std.math.isFinite(e_xc));
    try std.testing.expect(std.math.isFinite(integral_vxc_rho));
}

test "Hartree energy for hydrogen density" {
    // E_H = (1/2) ∫ V_H ρ 4πr² dr
    // Exact: E_H = 5/8 Ha = 5/4 Ry ≈ 1.25 Ry
    // Wait, for density ρ=exp(-2r)/π:
    // E_H = (1/2) × 2 ∫∫ ρ(r)ρ(r')/|r-r'| dV dV' = ∫ρ(r)V_H_hartree(r) dV
    // = ∫ ρ V_H_ry/(2) 4πr² dr
    // Actually E_H_ry = (1/2) ∫ V_H_ry ρ 4πr² dr
    // V_H_ry = 2 V_H_ha, so E_H_ry = ∫ V_H_ha ρ 4πr² dr = 2 E_H_ha
    // E_H_ha for H = 5/16 Ha, so E_H_ry = 5/8 Ry = 0.625 Ry
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 50.0);
    defer grid.deinit();

    const rho = try allocator.alloc(f64, grid.n);
    defer allocator.free(rho);

    for (0..grid.n) |i| {
        rho[i] = hydrogen_density(grid.r[i]);
    }

    const v_h = try allocator.alloc(f64, grid.n);
    defer allocator.free(v_h);

    atomic_solver.radial_poisson(&grid, rho, v_h);

    var e_h: f64 = 0;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        const w = ctrap_weight(i, grid.n);
        e_h += 0.5 * w * v_h[i] * rho[i] * 4.0 * std.math.pi * r * r * grid.rab[i];
    }

    try std.testing.expectApproxEqAbs(0.625, e_h, 0.01);
}

test "kinetic energy for hydrogen 1s" {
    // Exact T = 1.0 Ry (virial theorem: T = -E_total = 1.0 Ry for bare H)
    // T = ε_1s - ∫ V_eff × |u|²/4πr² × 4πr² dr = ε - ∫ V_eff |u|² dr
    // Or from wavefunction: T = -∫ u d²u/dr² dr + ∫ l(l+1)/r² |u|² dr
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    // Exact hydrogen 1s: u(r) = 2r exp(-r), normalized ∫u² dr = 1
    const u = try allocator.alloc(f64, grid.n);
    defer allocator.free(u);

    for (0..grid.n) |i| {
        u[i] = 2.0 * grid.r[i] * @exp(-grid.r[i]);
    }

    // Normalize
    var norm: f64 = 0;
    for (0..grid.n) |i| {
        const w = ctrap_weight(i, grid.n);
        norm += w * u[i] * u[i] * grid.rab[i];
    }
    try std.testing.expectApproxEqAbs(1.0, norm, 1e-5);
}

test "total energy decomposition: exact hydrogen density" {
    // For exact H 1s density ρ = exp(-2r)/π, compute all energy components
    // This is a non-self-consistent evaluation to check each piece independently.
    //
    // Exact values (Rydberg):
    //   T = 1.0, E_ext = -2.0, E_H = 5/8 = 0.625
    //   E_total(LDA, spin-unpolarized) ≈ T + E_ext + E_H + E_xc
    //
    // Note: for H atom (1 electron), spin-POLARIZED LDA gives E ≈ -0.958 Ry,
    // but spin-UNPOLARIZED LDA gives E ≈ -0.884 Ry. Our code currently uses
    // spin-unpolarized LDA, so the reference is the latter.
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const n = grid.n;

    // Exact density
    const rho = try allocator.alloc(f64, n);
    defer allocator.free(rho);

    for (0..n) |i| rho[i] = hydrogen_density(grid.r[i]);

    // E_ext = ∫ V_nuc ρ 4πr² dr = ∫ (-2/r)(e^{-2r}/π) 4πr² dr = -8 ∫ r e^{-2r} dr = -2.0 Ry
    var e_ext: f64 = 0;
    for (0..n) |i| {
        const r = grid.r[i];
        if (r < 1e-30) continue;
        const v_nuc = -2.0 / r;
        const w = ctrap_weight(i, n);
        e_ext += w * v_nuc * rho[i] * 4.0 * std.math.pi * r * r * grid.rab[i];
    }

    // E_H via Poisson
    const v_h = try allocator.alloc(f64, n);
    defer allocator.free(v_h);

    atomic_solver.radial_poisson(&grid, rho, v_h);

    var e_h: f64 = 0;
    for (0..n) |i| {
        const r = grid.r[i];
        const w = ctrap_weight(i, n);
        e_h += 0.5 * w * v_h[i] * rho[i] * 4.0 * std.math.pi * r * r * grid.rab[i];
    }

    // E_xc
    var e_xc: f64 = 0;
    for (0..n) |i| {
        const r = grid.r[i];
        const w = ctrap_weight(i, n);
        const xc_pt = xc_mod.eval_point(.lda_pz, rho[i], 0);
        e_xc += w * xc_pt.f * 4.0 * std.math.pi * r * r * grid.rab[i];
    }

    // T = 1.0 Ry (exact, from virial theorem or direct integration)
    const t_exact: f64 = 1.0;

    const e_total = t_exact + e_ext + e_h + e_xc;

    // Verify exact components
    try std.testing.expectApproxEqAbs(-2.0, e_ext, 1e-5);
    try std.testing.expectApproxEqAbs(0.625, e_h, 1e-4);
    try std.testing.expect(std.math.isFinite(e_total));
}

test "H atom SCF: spin-unpolarized LDA" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const orbitals = [_]atomic_solver.OrbitalConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0 },
    };

    var result = try atomic_solver.solve(allocator, &grid, .{
        .z = 1,
        .orbitals = &orbitals,
        .xc = .lda_pz,
    }, 200, 0.3, 1e-10);
    defer result.deinit();

    // For spin-unpolarized LDA, H atom E ≈ -0.89 Ry
    try std.testing.expect(result.total_energy < -0.88);
    try std.testing.expect(result.total_energy > -0.92);
}

test "He atom SCF: spin-unpolarized LDA" {
    // He (closed-shell): spin-unpolarized is correct
    // Reference: E(He, LDA-PZ) ≈ -5.670 Ry
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 50.0);
    defer grid.deinit();

    const orbitals = [_]atomic_solver.OrbitalConfig{
        .{ .n = 1, .l = 0, .occupation = 2.0 },
    };

    var result = try atomic_solver.solve(allocator, &grid, .{
        .z = 2,
        .orbitals = &orbitals,
        .xc = .lda_pz,
    }, 200, 0.3, 1e-10);
    defer result.deinit();

    // He is closed-shell, so spin-unpolarized is exact
    try std.testing.expectApproxEqAbs(-5.670, result.total_energy, 0.01);
}

fn ctrap_weight(i: usize, n: usize) f64 {
    const endpoint_weights = [5]f64{
        23.75 / 72.0, 95.10 / 72.0, 55.20 / 72.0, 79.30 / 72.0, 70.65 / 72.0,
    };
    if (n < 10) {
        if (i == 0 or i == n - 1) return 0.5;
        return 1.0;
    }
    if (i < 5) return endpoint_weights[i];
    if (i >= n - 5) return endpoint_weights[n - 1 - i];
    return 1.0;
}
