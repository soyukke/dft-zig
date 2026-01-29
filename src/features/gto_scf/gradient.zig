//! Analytical nuclear gradients for RHF (and KS-DFT) energy.
//!
//! The RHF energy gradient with respect to nuclear coordinate R_A is:
//!
//!   dE/dR_A = sum_{mu,nu} P_{mu,nu} * dH_{mu,nu}/dR_A
//!           + sum_{mu,nu,lam,sig} Gamma_{mu,nu,lam,sig} * d(mu nu|lam sig)/dR_A
//!           - sum_{mu,nu} W_{mu,nu} * dS_{mu,nu}/dR_A
//!           + dV_nn/dR_A
//!
//! where Gamma is the two-particle density matrix for RHF:
//!   Gamma_{mu,nu,lam,sig} = P_{mu,nu} * P_{lam,sig} - 0.5 * P_{mu,lam} * P_{nu,sig}
//!
//! and W is the energy-weighted density matrix:
//!   W_{mu,nu} = 2 * sum_i^{occ} eps_i * C_{mu,i} * C_{nu,i}
//!
//! Derivative of a primitive Gaussian centered at A with respect to A_x:
//!   d/dA_x [x^a * exp(-alpha*r_A^2)] = 2*alpha * x^{a+1} * exp(...) - a * x^{a-1} * exp(...)
//!
//! This means:
//!   d<mu|O|nu>/dA_x = 2*alpha_mu * <mu+1_x|O|nu> - a_x * <mu-1_x|O|nu>
//!
//! Units: Hartree atomic units throughout.

const std = @import("std");
const math_mod = @import("../math/math.zig");
const basis_mod = @import("../basis/basis.zig");
const integrals = @import("../integrals/integrals.zig");
const obara_saika = integrals.obara_saika;
const linalg = @import("../linalg/linalg.zig");
const rys_eri = integrals.rys_eri;
const fock_mod = @import("fock.zig");
const energy_mod = @import("energy.zig");
const kohn_sham = @import("kohn_sham.zig");
const grid_mod = @import("../grid/grid.zig");
const becke = grid_mod.becke;
const xc_functionals = grid_mod.xc_functionals;

const ContractedShell = basis_mod.ContractedShell;
const AngularMomentum = basis_mod.AngularMomentum;
const PrimitiveGaussian = basis_mod.PrimitiveGaussian;
const Vec3 = math_mod.Vec3;
const GridPoint = becke.GridPoint;
const XcFunctional = kohn_sham.XcFunctional;
const BasisOnGrid = kohn_sham.BasisOnGrid;

// ============================================================================
// Nuclear repulsion gradient
// ============================================================================

/// Compute the gradient of nuclear repulsion energy.
///
///   dV_nn/dR_A = -Z_A * sum_{B != A} Z_B * (R_A - R_B) / |R_A - R_B|^3
///
/// Returns an array of Vec3 gradients, one per atom.
pub fn nuclearRepulsionGradient(
    alloc: std.mem.Allocator,
    nuc_positions: []const Vec3,
    nuc_charges: []const f64,
) ![]Vec3 {
    const n_atoms = nuc_positions.len;
    const grad = try alloc.alloc(Vec3, n_atoms);
    for (grad) |*g| g.* = .{ .x = 0.0, .y = 0.0, .z = 0.0 };

    for (0..n_atoms) |a| {
        for (0..n_atoms) |b| {
            if (a == b) continue;
            const dx = nuc_positions[a].x - nuc_positions[b].x;
            const dy = nuc_positions[a].y - nuc_positions[b].y;
            const dz = nuc_positions[a].z - nuc_positions[b].z;
            const r2 = dx * dx + dy * dy + dz * dz;
            const r = @sqrt(r2);
            const r3 = r2 * r;
            const factor = -nuc_charges[a] * nuc_charges[b] / r3;
            grad[a].x += factor * dx;
            grad[a].y += factor * dy;
            grad[a].z += factor * dz;
        }
    }

    return grad;
}

test "contracted overlap derivative vs FD (p-type)" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const delta: f64 = 1e-5;
    const tol: f64 = 1e-7;

    // O 2p shell at origin, H 1s shell at displaced position
    const center_a = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = Vec3{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 };
    const shell_a = ContractedShell{ .center = center_a, .l = 1, .primitives = &sto3g.O_2p };
    const shell_b = ContractedShell{ .center = center_b, .l = 0, .primitives = &sto3g.H_1s };

    // Test p_y (ang = 0,1,0) vs s (0,0,0)
    const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
    const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

    const analytical = contractedOverlapDeriv(shell_a, ang_a, shell_b, ang_b);

    const dirs = [3]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = delta, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = delta },
    };
    const labels = [3][]const u8{ "x", "y", "z" };
    for (dirs, 0..) |d, i| {
        const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
        const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
        const shell_a_p = ContractedShell{ .center = ca_p, .l = 1, .primitives = &sto3g.O_2p };
        const shell_a_m = ContractedShell{ .center = ca_m, .l = 1, .primitives = &sto3g.O_2p };
        const s_p = obara_saika.contractedOverlap(shell_a_p, ang_a, shell_b, ang_b);
        const s_m = obara_saika.contractedOverlap(shell_a_m, ang_a, shell_b, ang_b);
        const fd = (s_p - s_m) / (2.0 * delta);
        std.debug.print("  contracted dS/d{s}(p_y, s): analytical={d:20.12} fd={d:20.12} diff={e:12.4}\n", .{ labels[i], analytical[i], fd, @abs(analytical[i] - fd) });
        try testing.expectApproxEqAbs(fd, analytical[i], tol);
    }
}

test "contracted kinetic derivative vs FD (p-type)" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const delta: f64 = 1e-5;
    const tol: f64 = 1e-7;

    const center_a = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = Vec3{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 };
    const shell_a = ContractedShell{ .center = center_a, .l = 1, .primitives = &sto3g.O_2p };
    const shell_b = ContractedShell{ .center = center_b, .l = 0, .primitives = &sto3g.H_1s };

    const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
    const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

    const analytical = contractedKineticDeriv(shell_a, ang_a, shell_b, ang_b);

    const dirs = [3]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = delta, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = delta },
    };
    const labels = [3][]const u8{ "x", "y", "z" };
    for (dirs, 0..) |d, i| {
        const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
        const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
        const shell_a_p = ContractedShell{ .center = ca_p, .l = 1, .primitives = &sto3g.O_2p };
        const shell_a_m = ContractedShell{ .center = ca_m, .l = 1, .primitives = &sto3g.O_2p };
        const t_p = obara_saika.contractedKinetic(shell_a_p, ang_a, shell_b, ang_b);
        const t_m = obara_saika.contractedKinetic(shell_a_m, ang_a, shell_b, ang_b);
        const fd = (t_p - t_m) / (2.0 * delta);
        std.debug.print("  contracted dT/d{s}(p_y, s): analytical={d:20.12} fd={d:20.12} diff={e:12.4}\n", .{ labels[i], analytical[i], fd, @abs(analytical[i] - fd) });
        try testing.expectApproxEqAbs(fd, analytical[i], tol);
    }
}

test "contracted nuclear derivative vs FD (p-type)" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const delta: f64 = 1e-5;
    const tol: f64 = 1e-7;

    const center_a = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = Vec3{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 };
    const nuc_pos = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const nuc_charge: f64 = 8.0;
    const shell_a = ContractedShell{ .center = center_a, .l = 1, .primitives = &sto3g.O_2p };
    const shell_b = ContractedShell{ .center = center_b, .l = 0, .primitives = &sto3g.H_1s };

    const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
    const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

    const analytical = contractedNuclearDeriv(shell_a, ang_a, shell_b, ang_b, nuc_pos, nuc_charge);

    const dirs = [3]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = delta, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = delta },
    };
    const labels = [3][]const u8{ "x", "y", "z" };
    for (dirs, 0..) |d, i| {
        const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
        const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
        const shell_a_p = ContractedShell{ .center = ca_p, .l = 1, .primitives = &sto3g.O_2p };
        const shell_a_m = ContractedShell{ .center = ca_m, .l = 1, .primitives = &sto3g.O_2p };
        const v_p = obara_saika.contractedTotalNuclearAttraction(shell_a_p, ang_a, shell_b, ang_b, &[_]Vec3{nuc_pos}, &[_]f64{nuc_charge});
        const v_m = obara_saika.contractedTotalNuclearAttraction(shell_a_m, ang_a, shell_b, ang_b, &[_]Vec3{nuc_pos}, &[_]f64{nuc_charge});
        const fd = (v_p - v_m) / (2.0 * delta);
        std.debug.print("  contracted dV/d{s}(p_y, s): analytical={d:20.12} fd={d:20.12} diff={e:12.4}\n", .{ labels[i], analytical[i], fd, @abs(analytical[i] - fd) });
        try testing.expectApproxEqAbs(fd, analytical[i], tol);
    }
}

// ============================================================================
// Energy-weighted density matrix
// ============================================================================

/// Build the energy-weighted density matrix:
///   W_{mu,nu} = 2 * sum_{i}^{occ} eps_i * C_{mu,i} * C_{nu,i}
///
/// C is in column-major order: C[j*n + i] = C_{i,j}
pub fn buildEnergyWeightedDensity(
    alloc: std.mem.Allocator,
    n: usize,
    n_occ: usize,
    orbital_energies: []const f64,
    mo_coefficients: []const f64,
) ![]f64 {
    const w = try alloc.alloc(f64, n * n);
    @memset(w, 0.0);

    for (0..n) |mu| {
        for (0..n) |nu| {
            var sum: f64 = 0.0;
            for (0..n_occ) |i| {
                // Column-major: C_{mu,i} = mo_coefficients[i*n + mu]
                sum += orbital_energies[i] * mo_coefficients[i * n + mu] * mo_coefficients[i * n + nu];
            }
            w[mu * n + nu] = 2.0 * sum;
        }
    }

    return w;
}

// ============================================================================
// Derivative integrals using Obara-Saika
// ============================================================================

// The key identity for derivatives:
//   d/dA_x <a|O|b> = 2*alpha * <a+1_x|O|b> - a_x * <a-1_x|O|b>
//
// For contracted shells, we loop over primitives and apply this identity.
// The existing obara_saika module computes integrals for arbitrary angular
// momentum, so we can evaluate the augmented integrals by constructing
// temporary shells with shifted angular momentum.

/// Compute the 3x3 matrix of overlap integral derivatives d<mu|nu>/dA_x, dA_y, dA_z
/// for contracted shells, returning contributions per basis function pair.
///
/// For derivative with respect to center A (where mu is centered):
///   dS_{mu,nu}/dA_x = sum_prim c_i * c_j * N_i * N_j *
///       [2*alpha_i * S_prim(a+1_x, b) - a_x * S_prim(a-1_x, b)]
///
/// The derivative matrix dS/dA for all mu on atom A, nu on any atom:
///   is stored as dS_x[mu*n + nu], dS_y[mu*n + nu], dS_z[mu*n + nu]
///
/// We use the fact that dS/dR_A = sum over mu on A of dS/dA contributions,
/// and dS/dR_B = -dS/dR_A (translational invariance) for the two-center case.
/// Compute dS_{mu,nu}/dA for a pair of *primitives*.
/// Returns (dS/dAx, dS/dAy, dS/dAz).
///
/// Uses: d/dA_x S(a,b) = 2*alpha * S(a+1_x, b) - a_x * S(a-1_x, b)
fn primOverlapDeriv(
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
) [3]f64 {
    var result: [3]f64 = .{ 0.0, 0.0, 0.0 };

    // x-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x + 1, .y = ang_a.y, .z = ang_a.z };
        const s_plus = obara_saika.primitiveOverlap(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b);
        result[0] = 2.0 * alpha_a * s_plus;
        if (ang_a.x > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x - 1, .y = ang_a.y, .z = ang_a.z };
            const s_minus = obara_saika.primitiveOverlap(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b);
            result[0] -= @as(f64, @floatFromInt(ang_a.x)) * s_minus;
        }
    }

    // y-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y + 1, .z = ang_a.z };
        const s_plus = obara_saika.primitiveOverlap(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b);
        result[1] = 2.0 * alpha_a * s_plus;
        if (ang_a.y > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y - 1, .z = ang_a.z };
            const s_minus = obara_saika.primitiveOverlap(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b);
            result[1] -= @as(f64, @floatFromInt(ang_a.y)) * s_minus;
        }
    }

    // z-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y, .z = ang_a.z + 1 };
        const s_plus = obara_saika.primitiveOverlap(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b);
        result[2] = 2.0 * alpha_a * s_plus;
        if (ang_a.z > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y, .z = ang_a.z - 1 };
            const s_minus = obara_saika.primitiveOverlap(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b);
            result[2] -= @as(f64, @floatFromInt(ang_a.z)) * s_minus;
        }
    }

    return result;
}

/// Compute dT_{mu,nu}/dA for a pair of *primitives*.
/// Returns (dT/dAx, dT/dAy, dT/dAz).
fn primKineticDeriv(
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
) [3]f64 {
    var result: [3]f64 = .{ 0.0, 0.0, 0.0 };

    // x-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x + 1, .y = ang_a.y, .z = ang_a.z };
        const t_plus = obara_saika.primitiveKinetic(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b);
        result[0] = 2.0 * alpha_a * t_plus;
        if (ang_a.x > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x - 1, .y = ang_a.y, .z = ang_a.z };
            const t_minus = obara_saika.primitiveKinetic(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b);
            result[0] -= @as(f64, @floatFromInt(ang_a.x)) * t_minus;
        }
    }

    // y-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y + 1, .z = ang_a.z };
        const t_plus = obara_saika.primitiveKinetic(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b);
        result[1] = 2.0 * alpha_a * t_plus;
        if (ang_a.y > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y - 1, .z = ang_a.z };
            const t_minus = obara_saika.primitiveKinetic(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b);
            result[1] -= @as(f64, @floatFromInt(ang_a.y)) * t_minus;
        }
    }

    // z-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y, .z = ang_a.z + 1 };
        const t_plus = obara_saika.primitiveKinetic(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b);
        result[2] = 2.0 * alpha_a * t_plus;
        if (ang_a.z > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y, .z = ang_a.z - 1 };
            const t_minus = obara_saika.primitiveKinetic(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b);
            result[2] -= @as(f64, @floatFromInt(ang_a.z)) * t_minus;
        }
    }

    return result;
}

/// Compute dV_{mu,nu}/dA for a pair of *primitives* and a single nucleus.
/// Returns (dV/dAx, dV/dAy, dV/dAz).
fn primNuclearDeriv(
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
    nuc_pos: Vec3,
    nuc_charge: f64,
) [3]f64 {
    var result: [3]f64 = .{ 0.0, 0.0, 0.0 };

    // x-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x + 1, .y = ang_a.y, .z = ang_a.z };
        const v_plus = obara_saika.primitiveNuclearAttraction(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
        result[0] = 2.0 * alpha_a * v_plus;
        if (ang_a.x > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x - 1, .y = ang_a.y, .z = ang_a.z };
            const v_minus = obara_saika.primitiveNuclearAttraction(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
            result[0] -= @as(f64, @floatFromInt(ang_a.x)) * v_minus;
        }
    }

    // y-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y + 1, .z = ang_a.z };
        const v_plus = obara_saika.primitiveNuclearAttraction(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
        result[1] = 2.0 * alpha_a * v_plus;
        if (ang_a.y > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y - 1, .z = ang_a.z };
            const v_minus = obara_saika.primitiveNuclearAttraction(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
            result[1] -= @as(f64, @floatFromInt(ang_a.y)) * v_minus;
        }
    }

    // z-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y, .z = ang_a.z + 1 };
        const v_plus = obara_saika.primitiveNuclearAttraction(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
        result[2] = 2.0 * alpha_a * v_plus;
        if (ang_a.z > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y, .z = ang_a.z - 1 };
            const v_minus = obara_saika.primitiveNuclearAttraction(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
            result[2] -= @as(f64, @floatFromInt(ang_a.z)) * v_minus;
        }
    }

    return result;
}

/// Compute d(ab|cd)/dA for a pair of primitives a (centered on A) and b, c, d.
/// Returns (d/dAx, d/dAy, d/dAz).
///
/// d/dA_x (a b | c d) = 2*alpha_a * (a+1_x b | c d) - a_x * (a-1_x b | c d)
fn primEriDeriv(
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
    alpha_c: f64,
    center_c: Vec3,
    ang_c: AngularMomentum,
    alpha_d: f64,
    center_d: Vec3,
    ang_d: AngularMomentum,
) [3]f64 {
    var result: [3]f64 = .{ 0.0, 0.0, 0.0 };

    // x-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x + 1, .y = ang_a.y, .z = ang_a.z };
        const eri_plus = obara_saika.primitiveERI(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b, alpha_c, center_c, ang_c, alpha_d, center_d, ang_d);
        result[0] = 2.0 * alpha_a * eri_plus;
        if (ang_a.x > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x - 1, .y = ang_a.y, .z = ang_a.z };
            const eri_minus = obara_saika.primitiveERI(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b, alpha_c, center_c, ang_c, alpha_d, center_d, ang_d);
            result[0] -= @as(f64, @floatFromInt(ang_a.x)) * eri_minus;
        }
    }

    // y-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y + 1, .z = ang_a.z };
        const eri_plus = obara_saika.primitiveERI(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b, alpha_c, center_c, ang_c, alpha_d, center_d, ang_d);
        result[1] = 2.0 * alpha_a * eri_plus;
        if (ang_a.y > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y - 1, .z = ang_a.z };
            const eri_minus = obara_saika.primitiveERI(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b, alpha_c, center_c, ang_c, alpha_d, center_d, ang_d);
            result[1] -= @as(f64, @floatFromInt(ang_a.y)) * eri_minus;
        }
    }

    // z-derivative
    {
        const a_plus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y, .z = ang_a.z + 1 };
        const eri_plus = obara_saika.primitiveERI(alpha_a, center_a, a_plus, alpha_b, center_b, ang_b, alpha_c, center_c, ang_c, alpha_d, center_d, ang_d);
        result[2] = 2.0 * alpha_a * eri_plus;
        if (ang_a.z > 0) {
            const a_minus = AngularMomentum{ .x = ang_a.x, .y = ang_a.y, .z = ang_a.z - 1 };
            const eri_minus = obara_saika.primitiveERI(alpha_a, center_a, a_minus, alpha_b, center_b, ang_b, alpha_c, center_c, ang_c, alpha_d, center_d, ang_d);
            result[2] -= @as(f64, @floatFromInt(ang_a.z)) * eri_minus;
        }
    }

    return result;
}

// ============================================================================
// Contracted derivative integrals
// ============================================================================

/// Contracted overlap derivative: d<shell_a, ang_a | shell_b, ang_b>/dA_{x,y,z}
fn contractedOverlapDeriv(
    shell_a: ContractedShell,
    ang_a: AngularMomentum,
    shell_b: ContractedShell,
    ang_b: AngularMomentum,
) [3]f64 {
    var result: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (shell_a.primitives) |prim_a| {
        const norm_a = basis_mod.normalization(prim_a.alpha, ang_a.x, ang_a.y, ang_a.z);
        for (shell_b.primitives) |prim_b| {
            const norm_b = basis_mod.normalization(prim_b.alpha, ang_b.x, ang_b.y, ang_b.z);
            const d = primOverlapDeriv(prim_a.alpha, shell_a.center, ang_a, prim_b.alpha, shell_b.center, ang_b);
            const coeff = prim_a.coeff * prim_b.coeff * norm_a * norm_b;
            result[0] += coeff * d[0];
            result[1] += coeff * d[1];
            result[2] += coeff * d[2];
        }
    }
    return result;
}

/// Contracted kinetic derivative: d<shell_a, ang_a | T | shell_b, ang_b>/dA_{x,y,z}
fn contractedKineticDeriv(
    shell_a: ContractedShell,
    ang_a: AngularMomentum,
    shell_b: ContractedShell,
    ang_b: AngularMomentum,
) [3]f64 {
    var result: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (shell_a.primitives) |prim_a| {
        const norm_a = basis_mod.normalization(prim_a.alpha, ang_a.x, ang_a.y, ang_a.z);
        for (shell_b.primitives) |prim_b| {
            const norm_b = basis_mod.normalization(prim_b.alpha, ang_b.x, ang_b.y, ang_b.z);
            const d = primKineticDeriv(prim_a.alpha, shell_a.center, ang_a, prim_b.alpha, shell_b.center, ang_b);
            const coeff = prim_a.coeff * prim_b.coeff * norm_a * norm_b;
            result[0] += coeff * d[0];
            result[1] += coeff * d[1];
            result[2] += coeff * d[2];
        }
    }
    return result;
}

/// Contracted nuclear attraction derivative for a single nucleus.
fn contractedNuclearDeriv(
    shell_a: ContractedShell,
    ang_a: AngularMomentum,
    shell_b: ContractedShell,
    ang_b: AngularMomentum,
    nuc_pos: Vec3,
    nuc_charge: f64,
) [3]f64 {
    var result: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (shell_a.primitives) |prim_a| {
        const norm_a = basis_mod.normalization(prim_a.alpha, ang_a.x, ang_a.y, ang_a.z);
        for (shell_b.primitives) |prim_b| {
            const norm_b = basis_mod.normalization(prim_b.alpha, ang_b.x, ang_b.y, ang_b.z);
            const d = primNuclearDeriv(prim_a.alpha, shell_a.center, ang_a, prim_b.alpha, shell_b.center, ang_b, nuc_pos, nuc_charge);
            const coeff = prim_a.coeff * prim_b.coeff * norm_a * norm_b;
            result[0] += coeff * d[0];
            result[1] += coeff * d[1];
            result[2] += coeff * d[2];
        }
    }
    return result;
}

/// Contracted ERI derivative: d(mu nu | lam sig)/dA for mu centered on shell_a.
fn contractedEriDeriv(
    shell_a: ContractedShell,
    ang_a: AngularMomentum,
    shell_b: ContractedShell,
    ang_b: AngularMomentum,
    shell_c: ContractedShell,
    ang_c: AngularMomentum,
    shell_d: ContractedShell,
    ang_d: AngularMomentum,
) [3]f64 {
    var result: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (shell_a.primitives) |prim_a| {
        const norm_a = basis_mod.normalization(prim_a.alpha, ang_a.x, ang_a.y, ang_a.z);
        for (shell_b.primitives) |prim_b| {
            const norm_b = basis_mod.normalization(prim_b.alpha, ang_b.x, ang_b.y, ang_b.z);
            for (shell_c.primitives) |prim_c| {
                const norm_c = basis_mod.normalization(prim_c.alpha, ang_c.x, ang_c.y, ang_c.z);
                for (shell_d.primitives) |prim_d| {
                    const norm_d = basis_mod.normalization(prim_d.alpha, ang_d.x, ang_d.y, ang_d.z);
                    const d = primEriDeriv(
                        prim_a.alpha,
                        shell_a.center,
                        ang_a,
                        prim_b.alpha,
                        shell_b.center,
                        ang_b,
                        prim_c.alpha,
                        shell_c.center,
                        ang_c,
                        prim_d.alpha,
                        shell_d.center,
                        ang_d,
                    );
                    const coeff = prim_a.coeff * prim_b.coeff * prim_c.coeff * prim_d.coeff * norm_a * norm_b * norm_c * norm_d;
                    result[0] += coeff * d[0];
                    result[1] += coeff * d[1];
                    result[2] += coeff * d[2];
                }
            }
        }
    }
    return result;
}

// ============================================================================
// Helper: map basis function index to shell and angular momentum
// ============================================================================

const BasisInfo = struct {
    shell_idx: usize,
    ang: AngularMomentum,
};

/// Build a mapping from basis function index to (shell_index, angular_momentum).
fn buildBasisMap(alloc: std.mem.Allocator, shells: []const ContractedShell) ![]BasisInfo {
    const n = obara_saika.totalBasisFunctions(shells);
    const map = try alloc.alloc(BasisInfo, n);
    var idx: usize = 0;
    for (shells, 0..) |shell, si| {
        const cart = basis_mod.cartesianExponents(shell.l);
        const n_cart = basis_mod.numCartesian(shell.l);
        for (0..n_cart) |ic| {
            map[idx] = .{ .shell_idx = si, .ang = cart[ic] };
            idx += 1;
        }
    }
    return map;
}

/// Map a basis function to its atom index.
/// atom_of_shell[shell_idx] = atom_index
fn buildShellToAtomMap(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const Vec3,
) ![]usize {
    const map = try alloc.alloc(usize, shells.len);
    for (shells, 0..) |shell, si| {
        // Find which atom this shell belongs to by matching center
        var best_atom: usize = 0;
        var best_dist: f64 = std.math.inf(f64);
        for (nuc_positions, 0..) |pos, ai| {
            const dx = shell.center.x - pos.x;
            const dy = shell.center.y - pos.y;
            const dz = shell.center.z - pos.z;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best_dist) {
                best_dist = d2;
                best_atom = ai;
            }
        }
        map[si] = best_atom;
    }
    return map;
}

// ============================================================================
// RHF Gradient
// ============================================================================

/// Gradient result.
pub const GradientResult = struct {
    /// Gradient vectors per atom (Hartree/Bohr).
    gradients: []Vec3,

    pub fn deinit(self: *GradientResult, alloc: std.mem.Allocator) void {
        if (self.gradients.len > 0) alloc.free(self.gradients);
    }
};

/// Compute the RHF analytical gradient.
///
/// Requires a converged RHF calculation (density matrix, orbital energies, MO coefficients).
///
///   dE/dR_A = sum_{mu,nu} P * dH_core/dR_A
///           + sum_{mu,nu,lam,sig} Gamma * dERI/dR_A
///           - sum_{mu,nu} W * dS/dR_A
///           + dV_nn/dR_A
///
/// The nuclear attraction integral V also has an explicit derivative with respect
/// to the nuclear position R_C (the nucleus, not the basis center):
///   dV_C/dR_C = sum_{mu,nu} P * d<mu|V_C|nu>/dR_C
///
/// This extra "Hellmann-Feynman" term must be added for each nucleus.
pub fn computeRhfGradient(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const Vec3,
    nuc_charges: []const f64,
    p_mat: []const f64,
    orbital_energies: []const f64,
    mo_coefficients: []const f64,
    n_occ: usize,
) !GradientResult {
    const n = obara_saika.totalBasisFunctions(shells);
    const n_atoms = nuc_positions.len;

    // Build maps
    const basis_map = try buildBasisMap(alloc, shells);
    defer alloc.free(basis_map);

    const shell_atom_map = try buildShellToAtomMap(alloc, shells, nuc_positions);
    defer alloc.free(shell_atom_map);

    // Build energy-weighted density matrix
    const w_mat = try buildEnergyWeightedDensity(alloc, n, n_occ, orbital_energies, mo_coefficients);
    defer alloc.free(w_mat);

    // Nuclear repulsion gradient
    const grad_vnn = try nuclearRepulsionGradient(alloc, nuc_positions, nuc_charges);
    defer alloc.free(grad_vnn);

    // Initialize total gradient with nuclear repulsion
    const grad = try alloc.alloc(Vec3, n_atoms);

    for (0..n_atoms) |a| {
        grad[a] = grad_vnn[a];
    }

    // ========================================================================
    // One-electron terms: overlap, kinetic, nuclear attraction (basis + HF)
    // ========================================================================
    //
    // Strategy: "first-center × 2" for all basis-center derivatives.
    //
    // For a 2-center integral I(mu,nu), translational invariance gives
    //   dI/d(center_mu) + dI/d(center_nu) = 0
    // so first-center × 2 correctly accounts for both centers.
    //
    // For the 3-center nuclear attraction integral V_C(mu,nu), translational
    // invariance is d/d(center_mu) + d/d(center_nu) + d/dR_C = 0.
    // Therefore d/d(center_nu) ≠ -d/d(center_mu) — we CANNOT use 2-center TI!
    // Instead we use first-center × 2 for the basis derivative part, and a
    // separate Hellmann-Feynman (HF) term for the nuclear-position derivative:
    //   dV_C/dR_C = -(dV_C/d(center_mu) + dV_C/d(center_nu))
    //
    // For each atom A, the one-electron gradient is:
    //   grad_1e[A] = sum_{mu on A, all nu} 2 * P * dT/d(center_mu)       [kinetic]
    //              - sum_{mu on A, all nu} 2 * W * dS/d(center_mu)       [overlap/Pulay]
    //              + sum_{mu on A, all nu, C} 2 * P * dV_C/d(center_mu)  [nuc. att. basis]
    //              + sum_{mu, nu} P * dV_A/dR_A                          [Hellmann-Feynman]
    //
    // The HF term uses: dV_A/dR_A = -(dV_A/d(center_mu) + dV_A/d(center_nu))
    // where dV_A/d(center_nu) is obtained by swapping mu↔nu in the derivative.
    // ========================================================================

    for (0..n) |mu| {
        const mu_info = basis_map[mu];
        const mu_shell = shells[mu_info.shell_idx];
        const atom_a = shell_atom_map[mu_info.shell_idx];

        for (0..n) |nu| {
            const nu_info = basis_map[nu];
            const nu_shell = shells[nu_info.shell_idx];

            const p_val = p_mat[mu * n + nu];
            const w_val = w_mat[mu * n + nu];

            // --- Overlap derivative: -2 * W * dS/d(center_mu) to atom_a ---
            const ds_dA = contractedOverlapDeriv(mu_shell, mu_info.ang, nu_shell, nu_info.ang);
            grad[atom_a].x -= 2.0 * w_val * ds_dA[0];
            grad[atom_a].y -= 2.0 * w_val * ds_dA[1];
            grad[atom_a].z -= 2.0 * w_val * ds_dA[2];

            // --- Kinetic energy derivative: 2 * P * dT/d(center_mu) to atom_a ---
            const dt_dA = contractedKineticDeriv(mu_shell, mu_info.ang, nu_shell, nu_info.ang);
            grad[atom_a].x += 2.0 * p_val * dt_dA[0];
            grad[atom_a].y += 2.0 * p_val * dt_dA[1];
            grad[atom_a].z += 2.0 * p_val * dt_dA[2];

            // --- Nuclear attraction derivative ---
            for (0..n_atoms) |c| {
                // Basis derivative: 2 * P * dV_C/d(center_mu) to atom_a
                const dv_dA = contractedNuclearDeriv(mu_shell, mu_info.ang, nu_shell, nu_info.ang, nuc_positions[c], nuc_charges[c]);
                grad[atom_a].x += 2.0 * p_val * dv_dA[0];
                grad[atom_a].y += 2.0 * p_val * dv_dA[1];
                grad[atom_a].z += 2.0 * p_val * dv_dA[2];

                // Hellmann-Feynman: dV_C/dR_C = -(dV_C/d(center_mu) + dV_C/d(center_nu))
                // dV_C/d(center_nu) is obtained by swapping mu and nu:
                const dv_dB = contractedNuclearDeriv(nu_shell, nu_info.ang, mu_shell, mu_info.ang, nuc_positions[c], nuc_charges[c]);
                grad[c].x -= p_val * (dv_dA[0] + dv_dB[0]);
                grad[c].y -= p_val * (dv_dA[1] + dv_dB[1]);
                grad[c].z -= p_val * (dv_dA[2] + dv_dB[2]);
            }
        }
    }

    // ========================================================================
    // Two-electron terms: first-center × 2 (shell-based batch Rys derivatives)
    // ========================================================================
    //
    // The ERI is a 4-center integral. In a full n^4 loop, computing only the
    // first-center derivative d(mu nu|lam sig)/d(center_mu) covers 1 of 4 centers.
    // Multiplying by 2 gives the correct result because:
    //   (J - 0.5*K) contracted with P, first-center × 2
    // is equivalent to the full 4-center derivative × 0.5 prefactor.

    // Build Schwarz table for screening
    var schwarz_rhf = try fock_mod.buildSchwarzTable(alloc, shells);
    defer schwarz_rhf.deinit(alloc);

    const n_shells_rhf = shells.len;
    const schwarz_threshold_rhf: f64 = 1e-12;

    // Pre-compute max |P| per shell pair for additional screening
    const max_p_shell_rhf = try alloc.alloc(f64, n_shells_rhf * n_shells_rhf);
    defer alloc.free(max_p_shell_rhf);
    for (0..n_shells_rhf) |si| {
        const ni = schwarz_rhf.shell_sizes[si];
        const off_i = schwarz_rhf.shell_offsets[si];
        for (0..n_shells_rhf) |sj| {
            const nj = schwarz_rhf.shell_sizes[sj];
            const off_j = schwarz_rhf.shell_offsets[sj];
            var mx: f64 = 0.0;
            for (0..ni) |ii| {
                for (0..nj) |jj| {
                    const v = @abs(p_mat[(off_i + ii) * n + (off_j + jj)]);
                    if (v > mx) mx = v;
                }
            }
            max_p_shell_rhf[si * n_shells_rhf + sj] = mx;
        }
    }

    const MAX_CART_RHF = basis_mod.MAX_CART;
    const MAX_BATCH_RHF = MAX_CART_RHF * MAX_CART_RHF * MAX_CART_RHF * MAX_CART_RHF;
    var batch_dx_rhf: [MAX_BATCH_RHF]f64 = undefined;
    var batch_dy_rhf: [MAX_BATCH_RHF]f64 = undefined;
    var batch_dz_rhf: [MAX_BATCH_RHF]f64 = undefined;

    for (0..n_shells_rhf) |sa| {
        const na_s = schwarz_rhf.shell_sizes[sa];
        const off_a = schwarz_rhf.shell_offsets[sa];
        const atom_sa = shell_atom_map[sa];

        for (0..n_shells_rhf) |sb| {
            const nb_s = schwarz_rhf.shell_sizes[sb];
            const off_b = schwarz_rhf.shell_offsets[sb];
            const q_ab = schwarz_rhf.get(sa, sb);

            for (0..n_shells_rhf) |sc| {
                const nc_s = schwarz_rhf.shell_sizes[sc];
                const off_c = schwarz_rhf.shell_offsets[sc];

                for (0..n_shells_rhf) |sd| {
                    const nd_s = schwarz_rhf.shell_sizes[sd];
                    const off_d = schwarz_rhf.shell_offsets[sd];
                    const q_cd = schwarz_rhf.get(sc, sd);

                    // Schwarz screening
                    if (q_ab * q_cd < schwarz_threshold_rhf) continue;

                    // Density screening
                    const max_p_ab_r = max_p_shell_rhf[sa * n_shells_rhf + sb];
                    const max_p_cd_r = max_p_shell_rhf[sc * n_shells_rhf + sd];
                    const max_p_ac_r = max_p_shell_rhf[sa * n_shells_rhf + sc];
                    const max_p_bd_r = max_p_shell_rhf[sb * n_shells_rhf + sd];
                    const max_gamma_est_r = max_p_ab_r * max_p_cd_r + 0.5 * max_p_ac_r * max_p_bd_r;
                    if (max_gamma_est_r * q_ab * q_cd < schwarz_threshold_rhf) continue;

                    // Batch compute all derivative ERIs for this shell quartet
                    _ = rys_eri.contractedShellQuartetEriDeriv(
                        shells[sa],
                        shells[sb],
                        shells[sc],
                        shells[sd],
                        &batch_dx_rhf,
                        &batch_dy_rhf,
                        &batch_dz_rhf,
                    );

                    // Accumulate gradient from batch results
                    for (0..na_s) |ia| {
                        const mu = off_a + ia;
                        for (0..nb_s) |ib| {
                            const nu = off_b + ib;
                            const p_mu_nu = p_mat[mu * n + nu];
                            for (0..nc_s) |ic| {
                                const lam = off_c + ic;
                                const p_mu_lam = p_mat[mu * n + lam];
                                const p_lam_sig_base = lam * n;
                                for (0..nd_s) |id_d| {
                                    const sig = off_d + id_d;

                                    // RHF Gamma
                                    const gamma_r = p_mu_nu * p_mat[p_lam_sig_base + sig] - 0.5 * p_mu_lam * p_mat[nu * n + sig];
                                    if (@abs(gamma_r) < 1e-15) continue;

                                    const idx = ia * nb_s * nc_s * nd_s + ib * nc_s * nd_s + ic * nd_s + id_d;
                                    // First-center × 2 factor
                                    grad[atom_sa].x += 2.0 * gamma_r * batch_dx_rhf[idx];
                                    grad[atom_sa].y += 2.0 * gamma_r * batch_dy_rhf[idx];
                                    grad[atom_sa].z += 2.0 * gamma_r * batch_dz_rhf[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return GradientResult{ .gradients = grad };
}

// ============================================================================
// KS-DFT analytical gradient
// ============================================================================

/// Compute the analytical gradient for a KS-DFT calculation.
///
/// Compared to RHF, the differences are:
///   1. The two-electron Gamma tensor uses a configurable HF exchange fraction:
///      Gamma = P*P - hf_frac * 0.5 * P*P  (hf_frac=0 for LDA, 0.20 for B3LYP)
///   2. An additional XC grid-based gradient term is added.
///
/// For LDA (v_sigma = 0):
///   dE_xc/dR_A = -2 * sum_{mu on A, nu} sum_g w_g * P_{mu,nu} * v_xc(r_g) * grad(phi_mu(r_g)) * phi_nu(r_g)
///
/// For GGA (v_sigma != 0, e.g. B3LYP):
///   Additional terms involving second derivatives of basis functions and grad(rho) dot products.
///
/// The "first-center x 2" strategy is used for all derivative terms, same as RHF.
pub fn computeKsDftGradient(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const Vec3,
    nuc_charges: []const f64,
    p_mat: []const f64,
    orbital_energies: []const f64,
    mo_coefficients: []const f64,
    n_occ: usize,
    grid_points: []const GridPoint,
    xc_func: XcFunctional,
) !GradientResult {
    const n = obara_saika.totalBasisFunctions(shells);
    const n_atoms = nuc_positions.len;

    // Determine HF exchange fraction from the XC functional type
    const hf_frac: f64 = switch (xc_func) {
        .lda_svwn => 0.0,
        .b3lyp => 0.20,
    };

    // Build maps
    const basis_map = try buildBasisMap(alloc, shells);
    defer alloc.free(basis_map);

    const shell_atom_map = try buildShellToAtomMap(alloc, shells, nuc_positions);
    defer alloc.free(shell_atom_map);

    // Build energy-weighted density matrix
    const w_mat = try buildEnergyWeightedDensity(alloc, n, n_occ, orbital_energies, mo_coefficients);
    defer alloc.free(w_mat);

    // Nuclear repulsion gradient
    const grad_vnn = try nuclearRepulsionGradient(alloc, nuc_positions, nuc_charges);
    defer alloc.free(grad_vnn);

    // Initialize total gradient with nuclear repulsion
    const grad = try alloc.alloc(Vec3, n_atoms);

    for (0..n_atoms) |a| {
        grad[a] = grad_vnn[a];
    }

    // ========================================================================
    // One-electron terms: overlap, kinetic, nuclear attraction
    // (identical to RHF)
    // ========================================================================

    for (0..n) |mu| {
        const mu_info = basis_map[mu];
        const mu_shell = shells[mu_info.shell_idx];
        const atom_a = shell_atom_map[mu_info.shell_idx];

        for (0..n) |nu| {
            const nu_info = basis_map[nu];
            const nu_shell = shells[nu_info.shell_idx];

            const p_val = p_mat[mu * n + nu];
            const w_val = w_mat[mu * n + nu];

            // --- Overlap derivative: -2 * W * dS/d(center_mu) to atom_a ---
            const ds_dA = contractedOverlapDeriv(mu_shell, mu_info.ang, nu_shell, nu_info.ang);
            grad[atom_a].x -= 2.0 * w_val * ds_dA[0];
            grad[atom_a].y -= 2.0 * w_val * ds_dA[1];
            grad[atom_a].z -= 2.0 * w_val * ds_dA[2];

            // --- Kinetic energy derivative: 2 * P * dT/d(center_mu) to atom_a ---
            const dt_dA = contractedKineticDeriv(mu_shell, mu_info.ang, nu_shell, nu_info.ang);
            grad[atom_a].x += 2.0 * p_val * dt_dA[0];
            grad[atom_a].y += 2.0 * p_val * dt_dA[1];
            grad[atom_a].z += 2.0 * p_val * dt_dA[2];

            // --- Nuclear attraction derivative ---
            for (0..n_atoms) |c| {
                // Basis derivative: 2 * P * dV_C/d(center_mu) to atom_a
                const dv_dA = contractedNuclearDeriv(mu_shell, mu_info.ang, nu_shell, nu_info.ang, nuc_positions[c], nuc_charges[c]);
                grad[atom_a].x += 2.0 * p_val * dv_dA[0];
                grad[atom_a].y += 2.0 * p_val * dv_dA[1];
                grad[atom_a].z += 2.0 * p_val * dv_dA[2];

                // Hellmann-Feynman: dV_C/dR_C
                const dv_dB = contractedNuclearDeriv(nu_shell, nu_info.ang, mu_shell, mu_info.ang, nuc_positions[c], nuc_charges[c]);
                grad[c].x -= p_val * (dv_dA[0] + dv_dB[0]);
                grad[c].y -= p_val * (dv_dA[1] + dv_dB[1]);
                grad[c].z -= p_val * (dv_dA[2] + dv_dB[2]);
            }
        }
    }

    // ========================================================================
    // Two-electron terms: shell-based loop with Schwarz screening
    // first-center x 2, with configurable HF exchange
    // Gamma = P_{mu,nu} * P_{lam,sig} - hf_frac * 0.5 * P_{mu,lam} * P_{nu,sig}
    // ========================================================================

    // Build Schwarz table for screening
    var schwarz = try fock_mod.buildSchwarzTable(alloc, shells);
    defer schwarz.deinit(alloc);

    const n_shells = shells.len;
    const schwarz_threshold: f64 = 1e-12;

    // Pre-compute max |P| per shell pair for additional screening
    const max_p_shell = try alloc.alloc(f64, n_shells * n_shells);
    defer alloc.free(max_p_shell);
    for (0..n_shells) |si| {
        const ni = schwarz.shell_sizes[si];
        const off_i = schwarz.shell_offsets[si];
        for (0..n_shells) |sj| {
            const nj = schwarz.shell_sizes[sj];
            const off_j = schwarz.shell_offsets[sj];
            var mx: f64 = 0.0;
            for (0..ni) |ii| {
                for (0..nj) |jj| {
                    const v = @abs(p_mat[(off_i + ii) * n + (off_j + jj)]);
                    if (v > mx) mx = v;
                }
            }
            max_p_shell[si * n_shells + sj] = mx;
        }
    }

    // Shell-quartet loop: sa over all shells (first center)
    // Uses batch Rys-based contractedShellQuartetEriDeriv for performance
    const MAX_CART = basis_mod.MAX_CART;
    const MAX_BATCH = MAX_CART * MAX_CART * MAX_CART * MAX_CART;
    var batch_dx: [MAX_BATCH]f64 = undefined;
    var batch_dy: [MAX_BATCH]f64 = undefined;
    var batch_dz: [MAX_BATCH]f64 = undefined;

    for (0..n_shells) |sa| {
        const na_s = schwarz.shell_sizes[sa];
        const off_a = schwarz.shell_offsets[sa];
        const atom_sa = shell_atom_map[sa];

        for (0..n_shells) |sb| {
            const nb_s = schwarz.shell_sizes[sb];
            const off_b = schwarz.shell_offsets[sb];
            const q_ab = schwarz.get(sa, sb);

            for (0..n_shells) |sc| {
                const nc_s = schwarz.shell_sizes[sc];
                const off_c = schwarz.shell_offsets[sc];

                for (0..n_shells) |sd| {
                    const nd_s = schwarz.shell_sizes[sd];
                    const off_d = schwarz.shell_offsets[sd];
                    const q_cd = schwarz.get(sc, sd);

                    // Schwarz screening
                    if (q_ab * q_cd < schwarz_threshold) continue;

                    // Density screening: estimate max |Gamma|
                    const max_p_ab = max_p_shell[sa * n_shells + sb];
                    const max_p_cd = max_p_shell[sc * n_shells + sd];
                    const max_p_ac = max_p_shell[sa * n_shells + sc];
                    const max_p_bd = max_p_shell[sb * n_shells + sd];
                    const max_gamma_est = max_p_ab * max_p_cd + hf_frac * 0.5 * max_p_ac * max_p_bd;
                    if (max_gamma_est * q_ab * q_cd < schwarz_threshold) continue;

                    // Batch compute all derivative ERIs for this shell quartet
                    _ = rys_eri.contractedShellQuartetEriDeriv(
                        shells[sa],
                        shells[sb],
                        shells[sc],
                        shells[sd],
                        &batch_dx,
                        &batch_dy,
                        &batch_dz,
                    );

                    // Accumulate gradient from batch results
                    for (0..na_s) |ia| {
                        const mu = off_a + ia;
                        for (0..nb_s) |ib| {
                            const nu = off_b + ib;
                            const p_mu_nu = p_mat[mu * n + nu];
                            for (0..nc_s) |ic| {
                                const lam = off_c + ic;
                                const p_mu_lam = p_mat[mu * n + lam];
                                const p_lam_sig_base = lam * n;
                                for (0..nd_s) |id_d| {
                                    const sig = off_d + id_d;

                                    // KS-DFT Gamma
                                    const gamma = p_mu_nu * p_mat[p_lam_sig_base + sig] - hf_frac * 0.5 * p_mu_lam * p_mat[nu * n + sig];
                                    if (@abs(gamma) < 1e-15) continue;

                                    const idx = ia * nb_s * nc_s * nd_s + ib * nc_s * nd_s + ic * nd_s + id_d;
                                    // First-center x 2 factor
                                    grad[atom_sa].x += 2.0 * gamma * batch_dx[idx];
                                    grad[atom_sa].y += 2.0 * gamma * batch_dy[idx];
                                    grad[atom_sa].z += 2.0 * gamma * batch_dz[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // XC gradient: grid-based exchange-correlation gradient
    // ========================================================================
    //
    // For LDA:
    //   dE_xc/dR_A = sum_g w_g * v_xc(r_g) * d(rho)/dR_A(r_g)
    //
    // where d(rho)/dR_A = sum_{mu,nu} P_{mu,nu} * [d(phi_mu)/dR_A * phi_nu + phi_mu * d(phi_nu)/dR_A]
    //
    // Since d(phi_mu)/dR_A = -d(phi_mu)/dr when mu is centered on atom A (else 0),
    // we get (using "first-center" approach):
    //   dE_xc/dR_A = -2 * sum_{mu on A} sum_nu P_{mu,nu}
    //                    * sum_g w_g * v_xc(r_g) * grad(phi_mu(r_g)) . e_xyz * phi_nu(r_g)
    //
    // For GGA (B3LYP), there is an additional v_sigma term involving grad(rho):
    //   Additional contribution: -4 * sum_{mu on A} sum_nu P_{mu,nu}
    //       * sum_g w_g * v_sigma * [grad_rho . grad(phi_nu)] * grad(phi_mu)
    //       + sum_g w_g * v_sigma * [grad_rho . grad(phi_mu)] * grad(phi_nu)]
    //   using first-center x 2 to handle the phi_mu center derivative.
    //
    // The full GGA gradient also requires second derivatives of basis functions,
    // which we avoid for now by using a simpler formulation that only needs first derivatives.

    const n_grid = grid_points.len;

    if (n_grid > 0) {
        // Evaluate basis functions on the grid
        var bog = try kohn_sham.evaluateBasisOnGrid(alloc, shells, grid_points);
        defer bog.deinit(alloc);

        // Compute density and density gradient on the grid
        const density_data = try kohn_sham.computeDensityOnGrid(alloc, n, n_grid, p_mat, bog);
        defer alloc.free(density_data.rho);
        defer alloc.free(density_data.grad_x);
        defer alloc.free(density_data.grad_y);
        defer alloc.free(density_data.grad_z);

        // Pre-compute v_xc (and v_sigma for GGA) at each grid point
        const v_xc_arr = try alloc.alloc(f64, n_grid);
        defer alloc.free(v_xc_arr);
        const v_sigma_arr = try alloc.alloc(f64, n_grid);
        defer alloc.free(v_sigma_arr);

        for (0..n_grid) |ig| {
            const rho_g = density_data.rho[ig];
            if (rho_g < 1e-20) {
                v_xc_arr[ig] = 0.0;
                v_sigma_arr[ig] = 0.0;
                continue;
            }
            switch (xc_func) {
                .lda_svwn => {
                    const xc = xc_functionals.ldaSvwn(rho_g);
                    v_xc_arr[ig] = xc.v_xc;
                    v_sigma_arr[ig] = 0.0;
                },
                .b3lyp => {
                    const grx = density_data.grad_x[ig];
                    const gry = density_data.grad_y[ig];
                    const grz = density_data.grad_z[ig];
                    const sigma = grx * grx + gry * gry + grz * grz;
                    const xc = xc_functionals.b3lyp(rho_g, sigma);
                    v_xc_arr[ig] = xc.v_xc;
                    v_sigma_arr[ig] = xc.v_sigma;
                },
            }
        }

        // Pre-compute P * phi for each grid point: p_phi[g * n + mu] = sum_nu P[mu,nu] * phi[g*n+nu]
        const p_phi = try alloc.alloc(f64, n_grid * n);
        defer alloc.free(p_phi);

        for (0..n_grid) |ig| {
            const g_off = ig * n;
            for (0..n) |mu| {
                var sum: f64 = 0.0;
                for (0..n) |nu| {
                    sum += p_mat[mu * n + nu] * bog.phi[g_off + nu];
                }
                p_phi[g_off + mu] = sum;
            }
        }

        // Accumulate XC gradient using "first-center x 2" strategy.
        //
        // LDA term (exact):
        //   dE_xc/dR_Ax = -2 * sum_{mu on A} sum_g w_g * v_xc(g) * (P*phi)_mu(g) * dphi_mu_x(g)
        //
        // GGA sigma term (complete):
        //   d(sigma)/dR_Ax has two parts:
        //     Part 1: from d(dphi_nu)/dR_A in grad_rho -- involves dphi_mu * (P*dphi_i)_mu
        //     Part 2: from d²phi_mu/(dx dx_i) -- involves (P*phi)_mu * d²phi_mu/(dx dx_i)
        //
        //   Combined GGA contribution per mu on A, per grid point:
        //     Part 1: -4 * w * v_sigma * [sum_i grad_rho_i * (P*dphi_i)_mu] * dphi_mu/dx
        //     Part 2: -4 * w * v_sigma * [sum_i grad_rho_i * (P*phi)_mu] * d²phi_mu/(dx dx_i)

        // Pre-compute P * dphi for GGA: p_dphi_{x,y,z}[g * n + mu] = sum_nu P[mu,nu] * dphi_{x,y,z}[g*n+nu]
        var p_dphi_x: ?[]f64 = null;
        var p_dphi_y: ?[]f64 = null;
        var p_dphi_z: ?[]f64 = null;
        defer if (p_dphi_x) |arr| alloc.free(arr);
        defer if (p_dphi_y) |arr| alloc.free(arr);
        defer if (p_dphi_z) |arr| alloc.free(arr);

        const has_gga = (xc_func == .b3lyp);
        if (has_gga) {
            p_dphi_x = try alloc.alloc(f64, n_grid * n);
            p_dphi_y = try alloc.alloc(f64, n_grid * n);
            p_dphi_z = try alloc.alloc(f64, n_grid * n);

            for (0..n_grid) |ig| {
                const g_off = ig * n;
                for (0..n) |mu| {
                    var sx: f64 = 0.0;
                    var sy: f64 = 0.0;
                    var sz: f64 = 0.0;
                    for (0..n) |nu| {
                        const p_mn = p_mat[mu * n + nu];
                        sx += p_mn * bog.dphi_x[g_off + nu];
                        sy += p_mn * bog.dphi_y[g_off + nu];
                        sz += p_mn * bog.dphi_z[g_off + nu];
                    }
                    p_dphi_x.?[g_off + mu] = sx;
                    p_dphi_y.?[g_off + mu] = sy;
                    p_dphi_z.?[g_off + mu] = sz;
                }
            }
        }

        // Accumulate XC gradient per atom
        for (0..n) |mu| {
            const mu_info = basis_map[mu];
            const atom_a = shell_atom_map[mu_info.shell_idx];

            var xc_grad_x: f64 = 0.0;
            var xc_grad_y: f64 = 0.0;
            var xc_grad_z: f64 = 0.0;

            for (0..n_grid) |ig| {
                const rho_g = density_data.rho[ig];
                if (rho_g < 1e-20) continue;

                const g_off = ig * n;
                const w = grid_points[ig].w;
                const pp_mu = p_phi[g_off + mu]; // (P*phi)_mu at grid point g
                const dphi_mu_x = bog.dphi_x[g_off + mu];
                const dphi_mu_y = bog.dphi_y[g_off + mu];
                const dphi_mu_z = bog.dphi_z[g_off + mu];

                // LDA contribution: v_xc * (P*phi)_mu * dphi_mu/dr
                const v_xc = v_xc_arr[ig];
                const lda_factor = w * v_xc * pp_mu;
                xc_grad_x += lda_factor * dphi_mu_x;
                xc_grad_y += lda_factor * dphi_mu_y;
                xc_grad_z += lda_factor * dphi_mu_z;

                // GGA contribution (Part 1 + Part 2):
                // Part 1: F_mu(g) = grad_rho . (P*dphi)_mu
                //         gga1_factor = -4 * w * v_sigma * F_mu(g)
                //         xc_grad += gga1_factor * dphi_mu
                //
                // Part 2: G_mu_x(g) = sum_i grad_rho_i * d²phi_mu/(dx dx_i)
                //         gga2_factor = -4 * w * v_sigma * (P*phi)_mu
                //         xc_grad_x += gga2_factor * G_mu_x

                if (has_gga) {
                    const v_sig = v_sigma_arr[ig];
                    if (@abs(v_sig) > 1e-30) {
                        const grx = density_data.grad_x[ig];
                        const gry = density_data.grad_y[ig];
                        const grz = density_data.grad_z[ig];

                        // Part 1: F_mu(g) = grad_rho . (P*dphi)_mu
                        const f_mu = grx * p_dphi_x.?[g_off + mu] +
                            gry * p_dphi_y.?[g_off + mu] +
                            grz * p_dphi_z.?[g_off + mu];

                        const gga1_factor = 2.0 * w * v_sig * f_mu;
                        xc_grad_x += gga1_factor * dphi_mu_x;
                        xc_grad_y += gga1_factor * dphi_mu_y;
                        xc_grad_z += gga1_factor * dphi_mu_z;

                        // Part 2: basis function Hessian contribution
                        // Evaluate d²phi_mu at this grid point on-the-fly
                        const shell_mu = shells[mu_info.shell_idx];
                        const hess = kohn_sham.evalBasisFunctionWithHessian(
                            shell_mu,
                            mu_info.ang,
                            grid_points[ig].x,
                            grid_points[ig].y,
                            grid_points[ig].z,
                        );
                        // G_mu_x = sum_i grad_rho_i * d²phi_mu/(dx dx_i)
                        const g_mu_x = grx * hess.dxx + gry * hess.dxy + grz * hess.dxz;
                        const g_mu_y = grx * hess.dxy + gry * hess.dyy + grz * hess.dyz;
                        const g_mu_z = grx * hess.dxz + gry * hess.dyz + grz * hess.dzz;

                        const gga2_factor = 2.0 * w * v_sig * pp_mu;
                        xc_grad_x += gga2_factor * g_mu_x;
                        xc_grad_y += gga2_factor * g_mu_y;
                        xc_grad_z += gga2_factor * g_mu_z;
                    }
                }
            }

            // Apply the -2 factor (first-center x 2) and add to total gradient
            grad[atom_a].x -= 2.0 * xc_grad_x;
            grad[atom_a].y -= 2.0 * xc_grad_y;
            grad[atom_a].z -= 2.0 * xc_grad_z;
        }
    }

    return GradientResult{ .gradients = grad };
}

// ============================================================================
// Tests
// ============================================================================

test "KS-DFT LDA gradient H2 STO-3G vs finite difference" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    // H2 geometry: R = 1.4 bohr along x-axis
    const nuc_positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const nuc_charges = [_]f64{ 1.0, 1.0 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Build molecular grid for DFT
    const atoms = [_]becke.Atom{
        .{ .x = nuc_positions[0].x, .y = nuc_positions[0].y, .z = nuc_positions[0].z, .z_number = 1 },
        .{ .x = nuc_positions[1].x, .y = nuc_positions[1].y, .z = nuc_positions[1].z, .z_number = 1 },
    };
    const grid_config = becke.GridConfig{
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
    };
    const grid_points = try becke.buildMolecularGrid(alloc, &atoms, grid_config);
    defer alloc.free(grid_points);

    // Run KS-DFT LDA SCF at equilibrium geometry
    const ks_params = kohn_sham.KsParams{
        .xc_functional = .lda_svwn,
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
        .energy_threshold = 1e-10,
        .density_threshold = 1e-8,
    };
    var ks_result = try kohn_sham.runKohnShamScf(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        2, // 2 electrons
        ks_params,
    );
    defer ks_result.deinit(alloc);
    try testing.expect(ks_result.converged);

    // Compute analytical gradient
    var grad_result = try computeKsDftGradient(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        ks_result.density_matrix_result,
        ks_result.orbital_energies,
        ks_result.mo_coefficients,
        1, // n_occ = 2/2
        grid_points,
        .lda_svwn,
    );
    defer grad_result.deinit(alloc);

    const grad = grad_result.gradients;

    std.debug.print("\nKS-DFT LDA Gradient H2 STO-3G (Ha/Bohr):\n", .{});
    std.debug.print("  H1: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[0].x, grad[0].y, grad[0].z });
    std.debug.print("  H2: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[1].x, grad[1].y, grad[1].z });

    // Translational invariance check
    const sum_x = grad[0].x + grad[1].x;
    const sum_y = grad[0].y + grad[1].y;
    const sum_z = grad[0].z + grad[1].z;
    std.debug.print("  Sum: {d:20.12} {d:20.12} {d:20.12}\n", .{ sum_x, sum_y, sum_z });

    // Numerical gradient by finite difference
    const delta: f64 = 1e-5;
    const pos_p = [_]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const pos_m = [_]Vec3{
        .{ .x = -delta, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const shells_p = [_]ContractedShell{
        .{ .center = pos_p[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = pos_p[1], .l = 0, .primitives = &sto3g.H_1s },
    };
    const shells_m = [_]ContractedShell{
        .{ .center = pos_m[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = pos_m[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    var ks_p = try kohn_sham.runKohnShamScf(alloc, &shells_p, &pos_p, &nuc_charges, 2, ks_params);
    defer ks_p.deinit(alloc);
    var ks_m = try kohn_sham.runKohnShamScf(alloc, &shells_m, &pos_m, &nuc_charges, 2, ks_params);
    defer ks_m.deinit(alloc);

    const num_grad_x = (ks_p.total_energy - ks_m.total_energy) / (2.0 * delta);
    std.debug.print("  Numerical dE/dx(H1): {d:20.12}\n", .{num_grad_x});
    std.debug.print("  Analytical dE/dx(H1): {d:20.12}\n", .{grad[0].x});
    std.debug.print("  Diff: {e:12.4}\n", .{@abs(grad[0].x - num_grad_x)});

    // Analytical vs FD should match to reasonable accuracy
    // Note: FD accuracy limited by grid changes with geometry; use generous tolerance
    const fd_tol: f64 = 5e-3;
    try testing.expectApproxEqAbs(num_grad_x, grad[0].x, fd_tol);

    // Translational invariance should hold approximately
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_x, 1e-3);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_y, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_z, 1e-6);
}

test "KS-DFT LDA gradient H2O STO-3G vs PySCF" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    // H2O geometry in bohr (same as RHF H2O test)
    const nuc_positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 }, // O
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 }, // H
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 }, // H
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    // Build STO-3G shells for H2O: O(1s,2s,2p) + H(1s) x 2
    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Build molecular grid for DFT
    const atoms = [_]becke.Atom{
        .{ .x = nuc_positions[0].x, .y = nuc_positions[0].y, .z = nuc_positions[0].z, .z_number = 8 },
        .{ .x = nuc_positions[1].x, .y = nuc_positions[1].y, .z = nuc_positions[1].z, .z_number = 1 },
        .{ .x = nuc_positions[2].x, .y = nuc_positions[2].y, .z = nuc_positions[2].z, .z_number = 1 },
    };
    const grid_config = becke.GridConfig{
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
    };
    const grid_points = try becke.buildMolecularGrid(alloc, &atoms, grid_config);
    defer alloc.free(grid_points);

    // Run KS-DFT LDA SCF
    const ks_params = kohn_sham.KsParams{
        .xc_functional = .lda_svwn,
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
        .energy_threshold = 1e-10,
        .density_threshold = 1e-8,
    };
    var ks_result = try kohn_sham.runKohnShamScf(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        10, // 10 electrons
        ks_params,
    );
    defer ks_result.deinit(alloc);
    try testing.expect(ks_result.converged);

    std.debug.print("\nKS-DFT LDA H2O energy: {d:20.12}\n", .{ks_result.total_energy});
    std.debug.print("PySCF reference:       {d:20.12}\n", .{-74.732105188947});

    // Compute analytical gradient
    var grad_result = try computeKsDftGradient(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        ks_result.density_matrix_result,
        ks_result.orbital_energies,
        ks_result.mo_coefficients,
        5, // n_occ = 10/2
        grid_points,
        .lda_svwn,
    );
    defer grad_result.deinit(alloc);

    const grad = grad_result.gradients;

    std.debug.print("\nKS-DFT LDA Gradient H2O STO-3G (Ha/Bohr):\n", .{});
    std.debug.print("  O:  {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[0].x, grad[0].y, grad[0].z });
    std.debug.print("  H1: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[1].x, grad[1].y, grad[1].z });
    std.debug.print("  H2: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[2].x, grad[2].y, grad[2].z });
    std.debug.print("PySCF reference:\n", .{});
    std.debug.print("  O:  +0.000000000000 +0.000000000000 +0.109695457484\n", .{});
    std.debug.print("  H1: +0.000000000000 -0.049549298854 -0.054856132039\n", .{});
    std.debug.print("  H2: -0.000000000000 +0.049549298854 -0.054856132039\n", .{});

    // PySCF LDA SVWN gradient reference (STO-3G, cart=True):
    //   O:  -0.000000000000  +0.000000000000  +0.109695457484
    //   H1: +0.000000000000  -0.049549298854  -0.054856132039
    //   H2: -0.000000000000  +0.049549298854  -0.054856132039
    //
    // Tolerance: expect ~1e-3 or better for LDA gradient vs PySCF
    const pyscf_tol: f64 = 2e-3;

    // Oxygen
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[0].x, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[0].y, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.109695457484), grad[0].z, pyscf_tol);

    // Hydrogen 1
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[1].x, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, -0.049549298854), grad[1].y, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, -0.054856132039), grad[1].z, pyscf_tol);

    // Hydrogen 2 (mirror of H1)
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[2].x, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.049549298854), grad[2].y, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, -0.054856132039), grad[2].z, pyscf_tol);

    // Translational invariance: sum of all gradients should be approximately zero.
    // Note: DFT grid gradient omits Becke weight derivatives, so translational
    // invariance is approximate (~1e-5 for moderate grids).
    const sum_x = grad[0].x + grad[1].x + grad[2].x;
    const sum_y = grad[0].y + grad[1].y + grad[2].y;
    const sum_z = grad[0].z + grad[1].z + grad[2].z;
    std.debug.print("  Sum: {d:20.12} {d:20.12} {d:20.12}\n", .{ sum_x, sum_y, sum_z });
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_x, 1e-5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_y, 1e-5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_z, 1e-5);
}

test "KS-DFT B3LYP gradient H2 STO-3G vs finite difference" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    // H2 geometry: R = 1.4 bohr along x-axis
    const nuc_positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const nuc_charges = [_]f64{ 1.0, 1.0 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Build molecular grid for DFT
    const atoms = [_]becke.Atom{
        .{ .x = nuc_positions[0].x, .y = nuc_positions[0].y, .z = nuc_positions[0].z, .z_number = 1 },
        .{ .x = nuc_positions[1].x, .y = nuc_positions[1].y, .z = nuc_positions[1].z, .z_number = 1 },
    };
    const grid_config = becke.GridConfig{
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
    };
    const grid_points = try becke.buildMolecularGrid(alloc, &atoms, grid_config);
    defer alloc.free(grid_points);

    // Run KS-DFT B3LYP SCF
    const ks_params = kohn_sham.KsParams{
        .xc_functional = .b3lyp,
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
        .energy_threshold = 1e-10,
        .density_threshold = 1e-8,
    };
    var ks_result = try kohn_sham.runKohnShamScf(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        2,
        ks_params,
    );
    defer ks_result.deinit(alloc);
    try testing.expect(ks_result.converged);

    std.debug.print("\nKS-DFT B3LYP H2 energy: {d:20.12}\n", .{ks_result.total_energy});

    // Compute analytical gradient
    var grad_result = try computeKsDftGradient(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        ks_result.density_matrix_result,
        ks_result.orbital_energies,
        ks_result.mo_coefficients,
        1,
        grid_points,
        .b3lyp,
    );
    defer grad_result.deinit(alloc);

    const grad = grad_result.gradients;

    std.debug.print("KS-DFT B3LYP Gradient H2 STO-3G (Ha/Bohr):\n", .{});
    std.debug.print("  H1: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[0].x, grad[0].y, grad[0].z });
    std.debug.print("  H2: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[1].x, grad[1].y, grad[1].z });

    // PySCF reference: H1: -0.011277589721, H2: +0.011277589721
    std.debug.print("  PySCF ref dE/dx(H1): -0.011277589721\n", .{});
    std.debug.print("  Diff vs PySCF: {e:12.4}\n", .{@abs(grad[0].x - (-0.011277589721))});

    // Numerical gradient by finite difference
    const delta: f64 = 1e-5;
    const pos_p = [_]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const pos_m = [_]Vec3{
        .{ .x = -delta, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const shells_p = [_]ContractedShell{
        .{ .center = pos_p[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = pos_p[1], .l = 0, .primitives = &sto3g.H_1s },
    };
    const shells_m = [_]ContractedShell{
        .{ .center = pos_m[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = pos_m[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    var ks_p = try kohn_sham.runKohnShamScf(alloc, &shells_p, &pos_p, &nuc_charges, 2, ks_params);
    defer ks_p.deinit(alloc);
    var ks_m = try kohn_sham.runKohnShamScf(alloc, &shells_m, &pos_m, &nuc_charges, 2, ks_params);
    defer ks_m.deinit(alloc);

    const num_grad_x = (ks_p.total_energy - ks_m.total_energy) / (2.0 * delta);
    std.debug.print("  Numerical dE/dx(H1): {d:20.12}\n", .{num_grad_x});
    std.debug.print("  Analytical dE/dx(H1): {d:20.12}\n", .{grad[0].x});
    std.debug.print("  Diff (anal vs FD): {e:12.4}\n", .{@abs(grad[0].x - num_grad_x)});

    // Analytical vs FD should match reasonably well
    const fd_tol: f64 = 5e-3;
    try testing.expectApproxEqAbs(num_grad_x, grad[0].x, fd_tol);

    // Translational invariance
    const sum_x = grad[0].x + grad[1].x;
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_x, 1e-3);

    // Check against PySCF reference (generous tolerance due to grid/energy accuracy)
    const pyscf_tol: f64 = 5e-3;
    try testing.expectApproxEqAbs(@as(f64, -0.011277589721), grad[0].x, pyscf_tol);
}

test "KS-DFT B3LYP gradient H2O STO-3G vs PySCF" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    // H2O geometry in bohr (same as other H2O gradient tests)
    const nuc_positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 }, // O
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 }, // H1
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 }, // H2
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };
    const atomic_numbers = [_]u8{ 8, 1, 1 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Build Becke grid
    const becke_atoms = [_]becke.Atom{
        .{ .x = nuc_positions[0].x, .y = nuc_positions[0].y, .z = nuc_positions[0].z, .z_number = atomic_numbers[0] },
        .{ .x = nuc_positions[1].x, .y = nuc_positions[1].y, .z = nuc_positions[1].z, .z_number = atomic_numbers[1] },
        .{ .x = nuc_positions[2].x, .y = nuc_positions[2].y, .z = nuc_positions[2].z, .z_number = atomic_numbers[2] },
    };
    const grid_config = becke.GridConfig{
        .n_radial = 80,
        .n_angular = 302,
    };
    const grid_points = try becke.buildMolecularGrid(alloc, &becke_atoms, grid_config);
    defer alloc.free(grid_points);

    // Run B3LYP SCF
    const ks_params = kohn_sham.KsParams{
        .xc_functional = .b3lyp,
        .n_radial = 80,
        .n_angular = 302,
        .prune = false,
        .energy_threshold = 1e-10,
        .density_threshold = 1e-8,
    };

    var ks_result = try kohn_sham.runKohnShamScf(alloc, &shells, &nuc_positions, &nuc_charges, 10, ks_params);
    defer ks_result.deinit(alloc);
    try testing.expect(ks_result.converged);

    std.debug.print("\nKS-DFT B3LYP H2O energy: {d:20.12}\n", .{ks_result.total_energy});

    // Compute analytical gradient
    var grad_result = try computeKsDftGradient(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        ks_result.density_matrix_result,
        ks_result.orbital_energies,
        ks_result.mo_coefficients,
        5, // n_occ = 10 electrons / 2
        grid_points,
        .b3lyp,
    );
    defer grad_result.deinit(alloc);

    const grad = grad_result.gradients;

    std.debug.print("KS-DFT B3LYP Gradient H2O STO-3G (Ha/Bohr):\n", .{});
    std.debug.print("  O:  {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[0].x, grad[0].y, grad[0].z });
    std.debug.print("  H1: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[1].x, grad[1].y, grad[1].z });
    std.debug.print("  H2: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[2].x, grad[2].y, grad[2].z });

    // PySCF reference (B3LYP/STO-3G):
    //   O:  0.000000000000  0.000000000000  0.107150408881
    //   H1: 0.000000000000 -0.047576883737 -0.053581380690
    //   H2:-0.000000000000  0.047576883737 -0.053581380690
    std.debug.print("PySCF reference:\n", .{});
    std.debug.print("  O:   0.000000000000  0.000000000000  0.107150408881\n", .{});
    std.debug.print("  H1:  0.000000000000 -0.047576883737 -0.053581380690\n", .{});
    std.debug.print("  H2: -0.000000000000  0.047576883737 -0.053581380690\n", .{});

    // Translational invariance
    const sum_x = grad[0].x + grad[1].x + grad[2].x;
    const sum_y = grad[0].y + grad[1].y + grad[2].y;
    const sum_z = grad[0].z + grad[1].z + grad[2].z;
    std.debug.print("  Sum: {d:20.12} {d:20.12} {d:20.12}\n", .{ sum_x, sum_y, sum_z });

    const trans_tol: f64 = 5e-3;
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_x, trans_tol);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_y, trans_tol);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_z, trans_tol);

    // Check vs PySCF reference (tolerance accounts for energy/grid accuracy)
    const pyscf_tol: f64 = 2e-3;

    // O gradient
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[0].x, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[0].y, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, 0.107150408881), grad[0].z, pyscf_tol);

    // H1 gradient
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[1].x, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, -0.047576883737), grad[1].y, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, -0.053581380690), grad[1].z, pyscf_tol);

    // H2 gradient (mirror of H1)
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[2].x, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, 0.047576883737), grad[2].y, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, -0.053581380690), grad[2].z, pyscf_tol);
}

test "nuclear repulsion gradient H2O" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // H2O geometry in bohr
    const nuc_positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    const grad = try nuclearRepulsionGradient(alloc, &nuc_positions, &nuc_charges);
    defer alloc.free(grad);

    // PySCF reference:
    //   O: +0.0000000000 +0.0000000000 +2.9920404717
    //   H: +0.0000000000 -2.0514459748 -1.4960202359
    //   H: +0.0000000000 +2.0514459748 -1.4960202359
    try testing.expectApproxEqAbs(0.0, grad[0].x, 1e-8);
    try testing.expectApproxEqAbs(0.0, grad[0].y, 1e-8);
    try testing.expectApproxEqAbs(2.9920404717, grad[0].z, 1e-6);

    try testing.expectApproxEqAbs(0.0, grad[1].x, 1e-8);
    try testing.expectApproxEqAbs(-2.0514459748, grad[1].y, 1e-6);
    try testing.expectApproxEqAbs(-1.4960202359, grad[1].z, 1e-6);

    try testing.expectApproxEqAbs(0.0, grad[2].x, 1e-8);
    try testing.expectApproxEqAbs(2.0514459748, grad[2].y, 1e-6);
    try testing.expectApproxEqAbs(-1.4960202359, grad[2].z, 1e-6);

    // Sum of forces should be zero (Newton's third law)
    const sum_x = grad[0].x + grad[1].x + grad[2].x;
    const sum_y = grad[0].y + grad[1].y + grad[2].y;
    const sum_z = grad[0].z + grad[1].z + grad[2].z;
    try testing.expectApproxEqAbs(0.0, sum_x, 1e-10);
    try testing.expectApproxEqAbs(0.0, sum_y, 1e-10);
    try testing.expectApproxEqAbs(0.0, sum_z, 1e-10);
}

test "RHF gradient H2O STO-3G" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");
    const gto_scf = @import("gto_scf.zig");

    // H2O geometry in bohr
    const nuc_positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 }, // O
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 }, // H
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 }, // H
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    // Build STO-3G shells for H2O
    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Run RHF SCF first
    var scf_result = try gto_scf.runGeneralRhfScf(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        10,
        .{ .energy_threshold = 1e-12, .density_threshold = 1e-10 },
    );
    defer scf_result.deinit(alloc);

    try testing.expect(scf_result.converged);

    // Compute analytical gradient
    var grad_result = try computeRhfGradient(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        scf_result.density_matrix,
        scf_result.orbital_energies,
        scf_result.mo_coefficients,
        5, // n_occ = 10/2
    );
    defer grad_result.deinit(alloc);

    const grad = grad_result.gradients;

    std.debug.print("\nRHF Gradient H2O STO-3G (Ha/Bohr):\n", .{});
    std.debug.print("  O: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[0].x, grad[0].y, grad[0].z });
    std.debug.print("  H: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[1].x, grad[1].y, grad[1].z });
    std.debug.print("  H: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[2].x, grad[2].y, grad[2].z });

    // ---- PySCF analytical gradient reference (same geometry, STO-3G, cart=True) ----
    // O:  0.000000000000   0.000000000000   0.061008878575
    // H: -0.000000000000  -0.023592244890  -0.030504439287
    // H: -0.000000000000   0.023592244890  -0.030504439287
    //
    // Our analytical gradient vs PySCF (expected diffs ~3.5e-4 due to 8e-6 Ha energy accuracy):
    const pyscf_tol: f64 = 5e-4;

    // Oxygen
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[0].x, 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[0].y, 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 0.061008878575), grad[0].z, pyscf_tol);

    // Hydrogen 1
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[1].x, 1e-8);
    try testing.expectApproxEqAbs(@as(f64, -0.023592244890), grad[1].y, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, -0.030504439287), grad[1].z, pyscf_tol);

    // Hydrogen 2 (mirror of H1)
    try testing.expectApproxEqAbs(@as(f64, 0.0), grad[2].x, 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 0.023592244890), grad[2].y, pyscf_tol);
    try testing.expectApproxEqAbs(@as(f64, -0.030504439287), grad[2].z, pyscf_tol);

    // Translational invariance: sum of all gradients should be zero
    const sum_x = grad[0].x + grad[1].x + grad[2].x;
    const sum_y = grad[0].y + grad[1].y + grad[2].y;
    const sum_z = grad[0].z + grad[1].z + grad[2].z;
    std.debug.print("  Sum: {d:20.12} {d:20.12} {d:20.12}\n", .{ sum_x, sum_y, sum_z });
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_x, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_y, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_z, 1e-6);
}

test "RHF gradient H2 STO-3G" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");
    const gto_scf = @import("gto_scf.zig");

    // H2 geometry: R = 1.4 bohr along x-axis
    const nuc_positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const nuc_charges = [_]f64{ 1.0, 1.0 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Run RHF SCF
    var scf_result = try gto_scf.runGeneralRhfScf(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        2,
        .{},
    );
    defer scf_result.deinit(alloc);

    try testing.expect(scf_result.converged);

    // Compute analytical gradient
    var grad_result = try computeRhfGradient(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        scf_result.density_matrix,
        scf_result.orbital_energies,
        scf_result.mo_coefficients,
        1, // n_occ = 2/2
    );
    defer grad_result.deinit(alloc);

    const grad = grad_result.gradients;

    std.debug.print("\nRHF Gradient H2 STO-3G (Ha/Bohr):\n", .{});
    std.debug.print("  H1: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[0].x, grad[0].y, grad[0].z });
    std.debug.print("  H2: {d:20.12} {d:20.12} {d:20.12}\n", .{ grad[1].x, grad[1].y, grad[1].z });

    // Numerical gradient by finite difference
    const delta: f64 = 1e-5;
    // Perturb H1 in x-direction
    const pos_p = [_]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const pos_m = [_]Vec3{
        .{ .x = -delta, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const shells_p = [_]ContractedShell{
        .{ .center = pos_p[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = pos_p[1], .l = 0, .primitives = &sto3g.H_1s },
    };
    const shells_m = [_]ContractedShell{
        .{ .center = pos_m[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = pos_m[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    var scf_p = try gto_scf.runGeneralRhfScf(alloc, &shells_p, &pos_p, &nuc_charges, 2, .{});
    defer scf_p.deinit(alloc);
    var scf_m = try gto_scf.runGeneralRhfScf(alloc, &shells_m, &pos_m, &nuc_charges, 2, .{});
    defer scf_m.deinit(alloc);

    const num_grad_x = (scf_p.total_energy - scf_m.total_energy) / (2.0 * delta);
    std.debug.print("  Numerical dE/dx(H1): {d:20.12}\n", .{num_grad_x});
    std.debug.print("  Analytical dE/dx(H1): {d:20.12}\n", .{grad[0].x});
    std.debug.print("  Diff: {e:12.4}\n", .{@abs(grad[0].x - num_grad_x)});

    // Analytical vs numerical gradient should match closely
    const num_tol: f64 = 1e-5;
    try testing.expectApproxEqAbs(num_grad_x, grad[0].x, num_tol);

    // Translational invariance
    const sum_x = grad[0].x + grad[1].x;
    const sum_y = grad[0].y + grad[1].y;
    const sum_z = grad[0].z + grad[1].z;
    std.debug.print("  Sum: {d:20.12} {d:20.12} {d:20.12}\n", .{ sum_x, sum_y, sum_z });
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_x, 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_y, 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_z, 1e-8);
}

test "primitive overlap derivative vs FD" {
    const testing = std.testing;
    const delta: f64 = 1e-5;
    const tol: f64 = 1e-5;

    // Common parameters
    const alpha_a: f64 = 3.42525091;
    const alpha_b: f64 = 0.16885540;
    const center_a = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = Vec3{ .x = 0.0, .y = 1.43, .z = 1.11 };

    // ---- Test case 1: (s, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 0, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = primOverlapDeriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
            const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
            const s_p = obara_saika.primitiveOverlap(alpha_a, ca_p, ang_a, alpha_b, center_b, ang_b);
            const s_m = obara_saika.primitiveOverlap(alpha_a, ca_m, ang_a, alpha_b, center_b, ang_b);
            const fd = (s_p - s_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 2: (p_y, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = primOverlapDeriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
            const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
            const s_p = obara_saika.primitiveOverlap(alpha_a, ca_p, ang_a, alpha_b, center_b, ang_b);
            const s_m = obara_saika.primitiveOverlap(alpha_a, ca_m, ang_a, alpha_b, center_b, ang_b);
            const fd = (s_p - s_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 3: (p_z, p_y) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 0, .z = 1 };
        const ang_b = AngularMomentum{ .x = 0, .y = 1, .z = 0 };

        const analytical = primOverlapDeriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
            const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
            const s_p = obara_saika.primitiveOverlap(alpha_a, ca_p, ang_a, alpha_b, center_b, ang_b);
            const s_m = obara_saika.primitiveOverlap(alpha_a, ca_m, ang_a, alpha_b, center_b, ang_b);
            const fd = (s_p - s_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }
}

test "primitive kinetic derivative vs FD" {
    const testing = std.testing;
    const delta: f64 = 1e-5;
    const tol: f64 = 1e-5;

    const alpha_a: f64 = 3.42525091;
    const alpha_b: f64 = 0.16885540;
    const center_a = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = Vec3{ .x = 0.0, .y = 1.43, .z = 1.11 };

    // ---- Test case 1: (s, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 0, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = primKineticDeriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
            const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
            const t_p = obara_saika.primitiveKinetic(alpha_a, ca_p, ang_a, alpha_b, center_b, ang_b);
            const t_m = obara_saika.primitiveKinetic(alpha_a, ca_m, ang_a, alpha_b, center_b, ang_b);
            const fd = (t_p - t_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 2: (p_y, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = primKineticDeriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
            const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
            const t_p = obara_saika.primitiveKinetic(alpha_a, ca_p, ang_a, alpha_b, center_b, ang_b);
            const t_m = obara_saika.primitiveKinetic(alpha_a, ca_m, ang_a, alpha_b, center_b, ang_b);
            const fd = (t_p - t_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 3: (p_z, p_y) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 0, .z = 1 };
        const ang_b = AngularMomentum{ .x = 0, .y = 1, .z = 0 };

        const analytical = primKineticDeriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
            const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
            const t_p = obara_saika.primitiveKinetic(alpha_a, ca_p, ang_a, alpha_b, center_b, ang_b);
            const t_m = obara_saika.primitiveKinetic(alpha_a, ca_m, ang_a, alpha_b, center_b, ang_b);
            const fd = (t_p - t_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }
}

test "primitive nuclear derivative vs FD" {
    const testing = std.testing;
    const delta: f64 = 1e-5;
    const tol: f64 = 1e-5;

    const alpha_a: f64 = 3.42525091;
    const alpha_b: f64 = 0.16885540;
    const center_a = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const center_b = Vec3{ .x = 0.0, .y = 1.43, .z = 1.11 };
    const nuc_pos = Vec3{ .x = 0.5, .y = -0.3, .z = 0.8 };
    const nuc_charge: f64 = 6.0;

    // ---- Test case 1: (s, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 0, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = primNuclearDeriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
            const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
            const v_p = obara_saika.primitiveNuclearAttraction(alpha_a, ca_p, ang_a, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
            const v_m = obara_saika.primitiveNuclearAttraction(alpha_a, ca_m, ang_a, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
            const fd = (v_p - v_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 2: (p_y, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = primNuclearDeriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
            const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
            const v_p = obara_saika.primitiveNuclearAttraction(alpha_a, ca_p, ang_a, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
            const v_m = obara_saika.primitiveNuclearAttraction(alpha_a, ca_m, ang_a, alpha_b, center_b, ang_b, nuc_pos, nuc_charge);
            const fd = (v_p - v_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }
}
