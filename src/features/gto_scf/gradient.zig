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
pub fn nuclear_repulsion_gradient(
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

    const analytical = contracted_overlap_deriv(shell_a, ang_a, shell_b, ang_b);

    const dirs = [3]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = delta, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = delta },
    };
    for (dirs, 0..) |d, i| {
        const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
        const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
        const shell_a_p = ContractedShell{ .center = ca_p, .l = 1, .primitives = &sto3g.O_2p };
        const shell_a_m = ContractedShell{ .center = ca_m, .l = 1, .primitives = &sto3g.O_2p };
        const s_p = obara_saika.contracted_overlap(shell_a_p, ang_a, shell_b, ang_b);
        const s_m = obara_saika.contracted_overlap(shell_a_m, ang_a, shell_b, ang_b);
        const fd = (s_p - s_m) / (2.0 * delta);
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

    const analytical = contracted_kinetic_deriv(shell_a, ang_a, shell_b, ang_b);

    const dirs = [3]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = delta, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = delta },
    };
    for (dirs, 0..) |d, i| {
        const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
        const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
        const shell_a_p = ContractedShell{ .center = ca_p, .l = 1, .primitives = &sto3g.O_2p };
        const shell_a_m = ContractedShell{ .center = ca_m, .l = 1, .primitives = &sto3g.O_2p };
        const t_p = obara_saika.contracted_kinetic(shell_a_p, ang_a, shell_b, ang_b);
        const t_m = obara_saika.contracted_kinetic(shell_a_m, ang_a, shell_b, ang_b);
        const fd = (t_p - t_m) / (2.0 * delta);
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

    const analytical = contracted_nuclear_deriv(
        shell_a,
        ang_a,
        shell_b,
        ang_b,
        nuc_pos,
        nuc_charge,
    );

    const dirs = [3]Vec3{
        .{ .x = delta, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = delta, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = delta },
    };
    for (dirs, 0..) |d, i| {
        const ca_p = Vec3{ .x = center_a.x + d.x, .y = center_a.y + d.y, .z = center_a.z + d.z };
        const ca_m = Vec3{ .x = center_a.x - d.x, .y = center_a.y - d.y, .z = center_a.z - d.z };
        const shell_a_p = ContractedShell{ .center = ca_p, .l = 1, .primitives = &sto3g.O_2p };
        const shell_a_m = ContractedShell{ .center = ca_m, .l = 1, .primitives = &sto3g.O_2p };
        const v_p = obara_saika.contracted_total_nuclear_attraction(
            shell_a_p,
            ang_a,
            shell_b,
            ang_b,
            &[_]Vec3{nuc_pos},
            &[_]f64{nuc_charge},
        );
        const v_m = obara_saika.contracted_total_nuclear_attraction(
            shell_a_m,
            ang_a,
            shell_b,
            ang_b,
            &[_]Vec3{nuc_pos},
            &[_]f64{nuc_charge},
        );
        const fd = (v_p - v_m) / (2.0 * delta);
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
pub fn build_energy_weighted_density(
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
                const c_mu = mo_coefficients[i * n + mu];
                const c_nu = mo_coefficients[i * n + nu];
                sum += orbital_energies[i] * c_mu * c_nu;
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

const Axis = enum { x, y, z };

fn increment_angular(ang: AngularMomentum, axis: Axis) AngularMomentum {
    var next = ang;
    switch (axis) {
        .x => next.x += 1,
        .y => next.y += 1,
        .z => next.z += 1,
    }
    return next;
}

fn angular_component(ang: AngularMomentum, axis: Axis) u32 {
    return switch (axis) {
        .x => ang.x,
        .y => ang.y,
        .z => ang.z,
    };
}

fn decrement_angular(ang: AngularMomentum, axis: Axis) AngularMomentum {
    var prev = ang;
    switch (axis) {
        .x => prev.x -= 1,
        .y => prev.y -= 1,
        .z => prev.z -= 1,
    }
    return prev;
}

fn derivative_two_center_axis(
    comptime integral_fn: anytype,
    axis: Axis,
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
) f64 {
    const plus = integral_fn(
        alpha_a,
        center_a,
        increment_angular(ang_a, axis),
        alpha_b,
        center_b,
        ang_b,
    );
    var deriv = 2.0 * alpha_a * plus;
    const component = angular_component(ang_a, axis);
    if (component > 0) {
        const minus = integral_fn(
            alpha_a,
            center_a,
            decrement_angular(ang_a, axis),
            alpha_b,
            center_b,
            ang_b,
        );
        deriv -= @as(f64, @floatFromInt(component)) * minus;
    }
    return deriv;
}

fn primitive_two_center_derivative(
    comptime integral_fn: anytype,
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
) [3]f64 {
    return .{
        derivative_two_center_axis(
            integral_fn,
            .x,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
        ),
        derivative_two_center_axis(
            integral_fn,
            .y,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
        ),
        derivative_two_center_axis(
            integral_fn,
            .z,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
        ),
    };
}

fn derivative_nuclear_axis(
    axis: Axis,
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
    nuc_pos: Vec3,
    nuc_charge: f64,
) f64 {
    const plus = obara_saika.primitive_nuclear_attraction(
        alpha_a,
        center_a,
        increment_angular(ang_a, axis),
        alpha_b,
        center_b,
        ang_b,
        nuc_pos,
        nuc_charge,
    );
    var deriv = 2.0 * alpha_a * plus;
    const component = angular_component(ang_a, axis);
    if (component > 0) {
        const minus = obara_saika.primitive_nuclear_attraction(
            alpha_a,
            center_a,
            decrement_angular(ang_a, axis),
            alpha_b,
            center_b,
            ang_b,
            nuc_pos,
            nuc_charge,
        );
        deriv -= @as(f64, @floatFromInt(component)) * minus;
    }
    return deriv;
}

fn primitive_nuclear_derivative(
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
    nuc_pos: Vec3,
    nuc_charge: f64,
) [3]f64 {
    return .{
        derivative_nuclear_axis(
            .x,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
            nuc_pos,
            nuc_charge,
        ),
        derivative_nuclear_axis(
            .y,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
            nuc_pos,
            nuc_charge,
        ),
        derivative_nuclear_axis(
            .z,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
            nuc_pos,
            nuc_charge,
        ),
    };
}

fn derivative_eri_axis(
    axis: Axis,
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
) f64 {
    const plus = obara_saika.primitive_eri(
        alpha_a,
        center_a,
        increment_angular(ang_a, axis),
        alpha_b,
        center_b,
        ang_b,
        alpha_c,
        center_c,
        ang_c,
        alpha_d,
        center_d,
        ang_d,
    );
    var deriv = 2.0 * alpha_a * plus;
    const component = angular_component(ang_a, axis);
    if (component > 0) {
        const minus = obara_saika.primitive_eri(
            alpha_a,
            center_a,
            decrement_angular(ang_a, axis),
            alpha_b,
            center_b,
            ang_b,
            alpha_c,
            center_c,
            ang_c,
            alpha_d,
            center_d,
            ang_d,
        );
        deriv -= @as(f64, @floatFromInt(component)) * minus;
    }
    return deriv;
}

fn primitive_eri_derivative(
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
    return .{
        derivative_eri_axis(
            .x,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
            alpha_c,
            center_c,
            ang_c,
            alpha_d,
            center_d,
            ang_d,
        ),
        derivative_eri_axis(
            .y,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
            alpha_c,
            center_c,
            ang_c,
            alpha_d,
            center_d,
            ang_d,
        ),
        derivative_eri_axis(
            .z,
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
            alpha_c,
            center_c,
            ang_c,
            alpha_d,
            center_d,
            ang_d,
        ),
    };
}

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
fn prim_overlap_deriv(
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
) [3]f64 {
    return primitive_two_center_derivative(
        obara_saika.primitive_overlap,
        alpha_a,
        center_a,
        ang_a,
        alpha_b,
        center_b,
        ang_b,
    );
}

/// Compute dT_{mu,nu}/dA for a pair of *primitives*.
/// Returns (dT/dAx, dT/dAy, dT/dAz).
fn prim_kinetic_deriv(
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
) [3]f64 {
    return primitive_two_center_derivative(
        obara_saika.primitive_kinetic,
        alpha_a,
        center_a,
        ang_a,
        alpha_b,
        center_b,
        ang_b,
    );
}

/// Compute dV_{mu,nu}/dA for a pair of *primitives* and a single nucleus.
/// Returns (dV/dAx, dV/dAy, dV/dAz).
fn prim_nuclear_deriv(
    alpha_a: f64,
    center_a: Vec3,
    ang_a: AngularMomentum,
    alpha_b: f64,
    center_b: Vec3,
    ang_b: AngularMomentum,
    nuc_pos: Vec3,
    nuc_charge: f64,
) [3]f64 {
    return primitive_nuclear_derivative(
        alpha_a,
        center_a,
        ang_a,
        alpha_b,
        center_b,
        ang_b,
        nuc_pos,
        nuc_charge,
    );
}

/// Compute d(ab|cd)/dA for a pair of primitives a (centered on A) and b, c, d.
/// Returns (d/dAx, d/dAy, d/dAz).
///
/// d/dA_x (a b | c d) = 2*alpha_a * (a+1_x b | c d) - a_x * (a-1_x b | c d)
fn prim_eri_deriv(
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
    return primitive_eri_derivative(
        alpha_a,
        center_a,
        ang_a,
        alpha_b,
        center_b,
        ang_b,
        alpha_c,
        center_c,
        ang_c,
        alpha_d,
        center_d,
        ang_d,
    );
}

// ============================================================================
// Contracted derivative integrals
// ============================================================================

/// Contracted overlap derivative: d<shell_a, ang_a | shell_b, ang_b>/dA_{x,y,z}
fn contracted_overlap_deriv(
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
            const d = prim_overlap_deriv(
                prim_a.alpha,
                shell_a.center,
                ang_a,
                prim_b.alpha,
                shell_b.center,
                ang_b,
            );
            const coeff = prim_a.coeff * prim_b.coeff * norm_a * norm_b;
            result[0] += coeff * d[0];
            result[1] += coeff * d[1];
            result[2] += coeff * d[2];
        }
    }
    return result;
}

/// Contracted kinetic derivative: d<shell_a, ang_a | T | shell_b, ang_b>/dA_{x,y,z}
fn contracted_kinetic_deriv(
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
            const d = prim_kinetic_deriv(
                prim_a.alpha,
                shell_a.center,
                ang_a,
                prim_b.alpha,
                shell_b.center,
                ang_b,
            );
            const coeff = prim_a.coeff * prim_b.coeff * norm_a * norm_b;
            result[0] += coeff * d[0];
            result[1] += coeff * d[1];
            result[2] += coeff * d[2];
        }
    }
    return result;
}

/// Contracted nuclear attraction derivative for a single nucleus.
fn contracted_nuclear_deriv(
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
            const d = prim_nuclear_deriv(
                prim_a.alpha,
                shell_a.center,
                ang_a,
                prim_b.alpha,
                shell_b.center,
                ang_b,
                nuc_pos,
                nuc_charge,
            );
            const coeff = prim_a.coeff * prim_b.coeff * norm_a * norm_b;
            result[0] += coeff * d[0];
            result[1] += coeff * d[1];
            result[2] += coeff * d[2];
        }
    }
    return result;
}

/// Contracted ERI derivative: d(mu nu | lam sig)/dA for mu centered on shell_a.
fn contracted_eri_deriv(
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
                    const d = prim_eri_deriv(
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
                    const coeff_prim = prim_a.coeff * prim_b.coeff * prim_c.coeff * prim_d.coeff;
                    const coeff_norm = norm_a * norm_b * norm_c * norm_d;
                    const coeff = coeff_prim * coeff_norm;
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
fn build_basis_map(alloc: std.mem.Allocator, shells: []const ContractedShell) ![]BasisInfo {
    const n = obara_saika.total_basis_functions(shells);
    const map = try alloc.alloc(BasisInfo, n);
    var idx: usize = 0;
    for (shells, 0..) |shell, si| {
        const cart = basis_mod.cartesian_exponents(shell.l);
        const n_cart = basis_mod.num_cartesian(shell.l);
        for (0..n_cart) |ic| {
            map[idx] = .{ .shell_idx = si, .ang = cart[ic] };
            idx += 1;
        }
    }
    return map;
}

/// Map a basis function to its atom index.
/// atom_of_shell[shell_idx] = atom_index
fn build_shell_to_atom_map(
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

const GradientWorkspace = struct {
    basis_map: []BasisInfo,
    shell_atom_map: []usize,
    w_mat: []f64,

    pub fn deinit(self: GradientWorkspace, alloc: std.mem.Allocator) void {
        alloc.free(self.basis_map);
        alloc.free(self.shell_atom_map);
        alloc.free(self.w_mat);
    }
};

const OneElectronGradientCtx = struct {
    grad: []Vec3,
    shells: []const ContractedShell,
    basis_map: []const BasisInfo,
    shell_atom_map: []const usize,
    nuc_positions: []const Vec3,
    nuc_charges: []const f64,
    p_mat: []const f64,
    w_mat: []const f64,
    n: usize,
};

const TwoElectronGradientCtx = struct {
    grad: []Vec3,
    shells: []const ContractedShell,
    shell_atom_map: []const usize,
    p_mat: []const f64,
    n: usize,
    hf_frac: f64,
};

const ShellQuartetInfo = struct {
    atom_sa: usize,
    off_a: usize,
    off_b: usize,
    off_c: usize,
    off_d: usize,
    na_s: usize,
    nb_s: usize,
    nc_s: usize,
    nd_s: usize,
};

const XcPotentialData = struct {
    v_xc: []f64,
    v_sigma: []f64,

    pub fn deinit(self: XcPotentialData, alloc: std.mem.Allocator) void {
        alloc.free(self.v_xc);
        alloc.free(self.v_sigma);
    }
};

const ProjectedDerivativeData = struct {
    x: ?[]f64 = null,
    y: ?[]f64 = null,
    z: ?[]f64 = null,

    pub fn deinit(self: ProjectedDerivativeData, alloc: std.mem.Allocator) void {
        if (self.x) |arr| alloc.free(arr);
        if (self.y) |arr| alloc.free(arr);
        if (self.z) |arr| alloc.free(arr);
    }
};

fn init_gradient_workspace(
    alloc: std.mem.Allocator,
    n: usize,
    shells: []const ContractedShell,
    nuc_positions: []const Vec3,
    orbital_energies: []const f64,
    mo_coefficients: []const f64,
    n_occ: usize,
) !GradientWorkspace {
    const basis_map = try build_basis_map(alloc, shells);
    errdefer alloc.free(basis_map);
    const shell_atom_map = try build_shell_to_atom_map(alloc, shells, nuc_positions);
    errdefer alloc.free(shell_atom_map);
    const w_mat = try build_energy_weighted_density(
        alloc,
        n,
        n_occ,
        orbital_energies,
        mo_coefficients,
    );
    errdefer alloc.free(w_mat);
    return .{
        .basis_map = basis_map,
        .shell_atom_map = shell_atom_map,
        .w_mat = w_mat,
    };
}

fn init_gradient_vector(
    alloc: std.mem.Allocator,
    nuc_positions: []const Vec3,
    nuc_charges: []const f64,
) ![]Vec3 {
    const grad_vnn = try nuclear_repulsion_gradient(alloc, nuc_positions, nuc_charges);
    defer alloc.free(grad_vnn);

    const grad = try alloc.alloc(Vec3, nuc_positions.len);
    @memcpy(grad, grad_vnn);
    return grad;
}

fn add_scaled_derivative(target: *Vec3, scale: f64, deriv: [3]f64) void {
    target.x += scale * deriv[0];
    target.y += scale * deriv[1];
    target.z += scale * deriv[2];
}

fn subtract_derivative_pair(target: *Vec3, scale: f64, lhs: [3]f64, rhs: [3]f64) void {
    target.x -= scale * (lhs[0] + rhs[0]);
    target.y -= scale * (lhs[1] + rhs[1]);
    target.z -= scale * (lhs[2] + rhs[2]);
}

fn accumulate_one_electron_pair(ctx: *const OneElectronGradientCtx, mu: usize, nu: usize) void {
    const mu_info = ctx.basis_map[mu];
    const nu_info = ctx.basis_map[nu];
    const mu_shell = ctx.shells[mu_info.shell_idx];
    const nu_shell = ctx.shells[nu_info.shell_idx];
    const atom_a = ctx.shell_atom_map[mu_info.shell_idx];
    const p_val = ctx.p_mat[mu * ctx.n + nu];
    const w_val = ctx.w_mat[mu * ctx.n + nu];

    add_scaled_derivative(
        &ctx.grad[atom_a],
        -2.0 * w_val,
        contracted_overlap_deriv(mu_shell, mu_info.ang, nu_shell, nu_info.ang),
    );
    add_scaled_derivative(
        &ctx.grad[atom_a],
        2.0 * p_val,
        contracted_kinetic_deriv(mu_shell, mu_info.ang, nu_shell, nu_info.ang),
    );

    for (ctx.nuc_positions, ctx.nuc_charges, 0..) |nuc_pos, nuc_charge, c| {
        const dv_da = contracted_nuclear_deriv(
            mu_shell,
            mu_info.ang,
            nu_shell,
            nu_info.ang,
            nuc_pos,
            nuc_charge,
        );
        const dv_db = contracted_nuclear_deriv(
            nu_shell,
            nu_info.ang,
            mu_shell,
            mu_info.ang,
            nuc_pos,
            nuc_charge,
        );
        add_scaled_derivative(&ctx.grad[atom_a], 2.0 * p_val, dv_da);
        subtract_derivative_pair(&ctx.grad[c], p_val, dv_da, dv_db);
    }
}

fn accumulate_one_electron_gradient(ctx: *const OneElectronGradientCtx) void {
    for (0..ctx.n) |mu| {
        for (0..ctx.n) |nu| {
            accumulate_one_electron_pair(ctx, mu, nu);
        }
    }
}

fn build_max_shell_density(
    alloc: std.mem.Allocator,
    p_mat: []const f64,
    n: usize,
    schwarz: anytype,
) ![]f64 {
    const n_shells = schwarz.shell_sizes.len;
    const max_p_shell = try alloc.alloc(f64, n_shells * n_shells);
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
    return max_p_shell;
}

fn shell_quartet_passes_density_screen(
    ctx: *const TwoElectronGradientCtx,
    max_p_shell: []const f64,
    n_shells: usize,
    sa: usize,
    sb: usize,
    sc: usize,
    sd: usize,
    q_ab: f64,
    q_cd: f64,
    threshold: f64,
) bool {
    if (q_ab * q_cd < threshold) return false;
    const max_p_ab = max_p_shell[sa * n_shells + sb];
    const max_p_cd = max_p_shell[sc * n_shells + sd];
    const max_p_ac = max_p_shell[sa * n_shells + sc];
    const max_p_bd = max_p_shell[sb * n_shells + sd];
    const max_gamma_est = max_p_ab * max_p_cd + ctx.hf_frac * 0.5 * max_p_ac * max_p_bd;
    return max_gamma_est * q_ab * q_cd >= threshold;
}

fn accumulate_shell_quartet_gradient(
    ctx: *const TwoElectronGradientCtx,
    info: ShellQuartetInfo,
    batch_dx: []const f64,
    batch_dy: []const f64,
    batch_dz: []const f64,
) void {
    for (0..info.na_s) |ia| {
        const mu = info.off_a + ia;
        for (0..info.nb_s) |ib| {
            const nu = info.off_b + ib;
            const p_mu_nu = ctx.p_mat[mu * ctx.n + nu];
            for (0..info.nc_s) |ic| {
                const lam = info.off_c + ic;
                const p_mu_lam = ctx.p_mat[mu * ctx.n + lam];
                const p_lam_sig_base = lam * ctx.n;
                for (0..info.nd_s) |id_d| {
                    const sig = info.off_d + id_d;
                    const coulomb_term = p_mu_nu * ctx.p_mat[p_lam_sig_base + sig];
                    const exchange_term =
                        ctx.hf_frac * 0.5 * p_mu_lam * ctx.p_mat[nu * ctx.n + sig];
                    const gamma = coulomb_term - exchange_term;
                    if (@abs(gamma) < 1e-15) continue;

                    const idx = ia * info.nb_s * info.nc_s * info.nd_s +
                        ib * info.nc_s * info.nd_s +
                        ic * info.nd_s +
                        id_d;
                    add_scaled_derivative(&ctx.grad[info.atom_sa], 2.0 * gamma, .{
                        batch_dx[idx],
                        batch_dy[idx],
                        batch_dz[idx],
                    });
                }
            }
        }
    }
}

fn accumulate_two_electron_gradient(
    alloc: std.mem.Allocator,
    ctx: *const TwoElectronGradientCtx,
) !void {
    const threshold: f64 = 1e-12;
    var schwarz = try fock_mod.build_schwarz_table(alloc, ctx.shells);
    defer schwarz.deinit(alloc);

    const max_p_shell = try build_max_shell_density(alloc, ctx.p_mat, ctx.n, schwarz);
    defer alloc.free(max_p_shell);

    const n_shells = ctx.shells.len;
    const max_cart = basis_mod.MAX_CART;
    const max_batch = max_cart * max_cart * max_cart * max_cart;
    var batch_dx: [max_batch]f64 = undefined;
    var batch_dy: [max_batch]f64 = undefined;
    var batch_dz: [max_batch]f64 = undefined;

    for (0..n_shells) |sa| {
        for (0..n_shells) |sb| {
            const q_ab = schwarz.get(sa, sb);
            for (0..n_shells) |sc| {
                for (0..n_shells) |sd| {
                    const q_cd = schwarz.get(sc, sd);
                    if (!shell_quartet_passes_density_screen(
                        ctx,
                        max_p_shell,
                        n_shells,
                        sa,
                        sb,
                        sc,
                        sd,
                        q_ab,
                        q_cd,
                        threshold,
                    )) continue;

                    _ = rys_eri.contracted_shell_quartet_eri_deriv(
                        ctx.shells[sa],
                        ctx.shells[sb],
                        ctx.shells[sc],
                        ctx.shells[sd],
                        &batch_dx,
                        &batch_dy,
                        &batch_dz,
                    );
                    accumulate_shell_quartet_gradient(ctx, .{
                        .atom_sa = ctx.shell_atom_map[sa],
                        .off_a = schwarz.shell_offsets[sa],
                        .off_b = schwarz.shell_offsets[sb],
                        .off_c = schwarz.shell_offsets[sc],
                        .off_d = schwarz.shell_offsets[sd],
                        .na_s = schwarz.shell_sizes[sa],
                        .nb_s = schwarz.shell_sizes[sb],
                        .nc_s = schwarz.shell_sizes[sc],
                        .nd_s = schwarz.shell_sizes[sd],
                    }, batch_dx[0..], batch_dy[0..], batch_dz[0..]);
                }
            }
        }
    }
}

fn free_density_grid_data(alloc: std.mem.Allocator, density_data: anytype) void {
    alloc.free(density_data.rho);
    alloc.free(density_data.grad_x);
    alloc.free(density_data.grad_y);
    alloc.free(density_data.grad_z);
}

fn build_xc_potential_data(
    alloc: std.mem.Allocator,
    density_data: anytype,
    xc_func: XcFunctional,
) !XcPotentialData {
    const n_grid = density_data.rho.len;
    const v_xc = try alloc.alloc(f64, n_grid);
    errdefer alloc.free(v_xc);
    const v_sigma = try alloc.alloc(f64, n_grid);
    errdefer alloc.free(v_sigma);
    for (0..n_grid) |ig| {
        const rho_g = density_data.rho[ig];
        if (rho_g < 1e-20) {
            v_xc[ig] = 0.0;
            v_sigma[ig] = 0.0;
            continue;
        }
        switch (xc_func) {
            .lda_svwn => {
                const xc_eval = xc_functionals.lda_svwn(rho_g);
                v_xc[ig] = xc_eval.v_xc;
                v_sigma[ig] = 0.0;
            },
            .b3lyp => {
                const sigma = density_data.grad_x[ig] * density_data.grad_x[ig] +
                    density_data.grad_y[ig] * density_data.grad_y[ig] +
                    density_data.grad_z[ig] * density_data.grad_z[ig];
                const xc_eval = xc_functionals.b3lyp(rho_g, sigma);
                v_xc[ig] = xc_eval.v_xc;
                v_sigma[ig] = xc_eval.v_sigma;
            },
        }
    }
    return .{ .v_xc = v_xc, .v_sigma = v_sigma };
}

fn build_projected_phi(
    alloc: std.mem.Allocator,
    p_mat: []const f64,
    bog: BasisOnGrid,
    n_grid: usize,
    n: usize,
) ![]f64 {
    const p_phi = try alloc.alloc(f64, n_grid * n);
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
    return p_phi;
}

fn build_projected_derivatives(
    alloc: std.mem.Allocator,
    p_mat: []const f64,
    bog: BasisOnGrid,
    n_grid: usize,
    n: usize,
    has_gga: bool,
) !ProjectedDerivativeData {
    if (!has_gga) return .{};

    const p_dphi_x = try alloc.alloc(f64, n_grid * n);
    errdefer alloc.free(p_dphi_x);
    const p_dphi_y = try alloc.alloc(f64, n_grid * n);
    errdefer alloc.free(p_dphi_y);
    const p_dphi_z = try alloc.alloc(f64, n_grid * n);
    errdefer alloc.free(p_dphi_z);
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
            p_dphi_x[g_off + mu] = sx;
            p_dphi_y[g_off + mu] = sy;
            p_dphi_z[g_off + mu] = sz;
        }
    }
    return .{
        .x = p_dphi_x,
        .y = p_dphi_y,
        .z = p_dphi_z,
    };
}

fn accumulate_gga_basis_point_contribution(
    shells: []const ContractedShell,
    mu_info: BasisInfo,
    grid_point: GridPoint,
    density_data: anytype,
    p_dphi: *const ProjectedDerivativeData,
    pp_mu: f64,
    dphi_mu: [3]f64,
    g_off: usize,
    mu: usize,
    ig: usize,
    w: f64,
    v_sig: f64,
) [3]f64 {
    const grx = density_data.grad_x[ig];
    const gry = density_data.grad_y[ig];
    const grz = density_data.grad_z[ig];
    const f_mu = grx * p_dphi.x.?[g_off + mu] +
        gry * p_dphi.y.?[g_off + mu] +
        grz * p_dphi.z.?[g_off + mu];
    const gga1_factor = 2.0 * w * v_sig * f_mu;
    const hess = kohn_sham.eval_basis_function_with_hessian(
        shells[mu_info.shell_idx],
        mu_info.ang,
        grid_point.x,
        grid_point.y,
        grid_point.z,
    );
    const gga2_factor = 2.0 * w * v_sig * pp_mu;
    return .{
        gga1_factor * dphi_mu[0] + gga2_factor * (grx * hess.dxx + gry * hess.dxy + grz * hess.dxz),
        gga1_factor * dphi_mu[1] + gga2_factor * (grx * hess.dxy + gry * hess.dyy + grz * hess.dyz),
        gga1_factor * dphi_mu[2] + gga2_factor * (grx * hess.dxz + gry * hess.dyz + grz * hess.dzz),
    };
}

fn accumulate_single_basis_xc_gradient(
    shells: []const ContractedShell,
    basis_map: []const BasisInfo,
    bog: BasisOnGrid,
    density_data: anytype,
    grid_points: []const GridPoint,
    xc_data: *const XcPotentialData,
    p_phi: []const f64,
    p_dphi: *const ProjectedDerivativeData,
    mu: usize,
    n: usize,
) [3]f64 {
    var xc_grad: [3]f64 = .{ 0.0, 0.0, 0.0 };
    const mu_info = basis_map[mu];
    const has_gga = p_dphi.x != null;
    for (0..grid_points.len) |ig| {
        if (density_data.rho[ig] < 1e-20) continue;
        const g_off = ig * n;
        const w = grid_points[ig].w;
        const pp_mu = p_phi[g_off + mu];
        const dphi_mu = [3]f64{
            bog.dphi_x[g_off + mu],
            bog.dphi_y[g_off + mu],
            bog.dphi_z[g_off + mu],
        };
        const lda_factor = w * xc_data.v_xc[ig] * pp_mu;
        xc_grad[0] += lda_factor * dphi_mu[0];
        xc_grad[1] += lda_factor * dphi_mu[1];
        xc_grad[2] += lda_factor * dphi_mu[2];
        if (has_gga and @abs(xc_data.v_sigma[ig]) > 1e-30) {
            const gga = accumulate_gga_basis_point_contribution(
                shells,
                mu_info,
                grid_points[ig],
                density_data,
                p_dphi,
                pp_mu,
                dphi_mu,
                g_off,
                mu,
                ig,
                w,
                xc_data.v_sigma[ig],
            );
            xc_grad[0] += gga[0];
            xc_grad[1] += gga[1];
            xc_grad[2] += gga[2];
        }
    }
    return xc_grad;
}

fn accumulate_xc_gradient(
    alloc: std.mem.Allocator,
    grad: []Vec3,
    shells: []const ContractedShell,
    basis_map: []const BasisInfo,
    shell_atom_map: []const usize,
    p_mat: []const f64,
    grid_points: []const GridPoint,
    xc_func: XcFunctional,
    n: usize,
) !void {
    if (grid_points.len == 0) return;
    var bog = try kohn_sham.evaluate_basis_on_grid(alloc, shells, grid_points);
    defer bog.deinit(alloc);

    const density_data = try kohn_sham.compute_density_on_grid(
        alloc,
        n,
        grid_points.len,
        p_mat,
        bog,
    );
    defer free_density_grid_data(alloc, density_data);

    const xc_data = try build_xc_potential_data(alloc, density_data, xc_func);
    defer xc_data.deinit(alloc);

    const p_phi = try build_projected_phi(alloc, p_mat, bog, grid_points.len, n);
    defer alloc.free(p_phi);

    const p_dphi = try build_projected_derivatives(
        alloc,
        p_mat,
        bog,
        grid_points.len,
        n,
        xc_func == .b3lyp,
    );
    defer p_dphi.deinit(alloc);

    for (0..n) |mu| {
        const xc_grad = accumulate_single_basis_xc_gradient(
            shells,
            basis_map,
            bog,
            density_data,
            grid_points,
            &xc_data,
            p_phi,
            &p_dphi,
            mu,
            n,
        );
        add_scaled_derivative(&grad[shell_atom_map[basis_map[mu].shell_idx]], -2.0, xc_grad);
    }
}

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
pub fn compute_rhf_gradient(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const Vec3,
    nuc_charges: []const f64,
    p_mat: []const f64,
    orbital_energies: []const f64,
    mo_coefficients: []const f64,
    n_occ: usize,
) !GradientResult {
    const n = obara_saika.total_basis_functions(shells);
    const workspace = try init_gradient_workspace(
        alloc,
        n,
        shells,
        nuc_positions,
        orbital_energies,
        mo_coefficients,
        n_occ,
    );
    defer workspace.deinit(alloc);

    const grad = try init_gradient_vector(alloc, nuc_positions, nuc_charges);
    errdefer alloc.free(grad);

    const one_electron = OneElectronGradientCtx{
        .grad = grad,
        .shells = shells,
        .basis_map = workspace.basis_map,
        .shell_atom_map = workspace.shell_atom_map,
        .nuc_positions = nuc_positions,
        .nuc_charges = nuc_charges,
        .p_mat = p_mat,
        .w_mat = workspace.w_mat,
        .n = n,
    };
    accumulate_one_electron_gradient(&one_electron);

    const two_electron = TwoElectronGradientCtx{
        .grad = grad,
        .shells = shells,
        .shell_atom_map = workspace.shell_atom_map,
        .p_mat = p_mat,
        .n = n,
        .hf_frac = 1.0,
    };
    try accumulate_two_electron_gradient(alloc, &two_electron);

    return .{ .gradients = grad };
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
///   dE_xc/dR_A = -2 * sum_{mu on A, nu} sum_g w_g * P_{mu,nu}
///                  * v_xc(r_g) * grad(phi_mu(r_g)) * phi_nu(r_g)
///
/// For GGA (v_sigma != 0, e.g. B3LYP):
///   Additional terms involving second derivatives of basis functions and grad(rho) dot products.
///
/// The "first-center x 2" strategy is used for all derivative terms, same as RHF.
pub fn compute_ks_dft_gradient(
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
    const n = obara_saika.total_basis_functions(shells);
    const hf_frac: f64 = switch (xc_func) {
        .lda_svwn => 0.0,
        .b3lyp => 0.20,
    };
    const workspace = try init_gradient_workspace(
        alloc,
        n,
        shells,
        nuc_positions,
        orbital_energies,
        mo_coefficients,
        n_occ,
    );
    defer workspace.deinit(alloc);

    const grad = try init_gradient_vector(alloc, nuc_positions, nuc_charges);
    errdefer alloc.free(grad);

    const one_electron = OneElectronGradientCtx{
        .grad = grad,
        .shells = shells,
        .basis_map = workspace.basis_map,
        .shell_atom_map = workspace.shell_atom_map,
        .nuc_positions = nuc_positions,
        .nuc_charges = nuc_charges,
        .p_mat = p_mat,
        .w_mat = workspace.w_mat,
        .n = n,
    };
    accumulate_one_electron_gradient(&one_electron);

    const two_electron = TwoElectronGradientCtx{
        .grad = grad,
        .shells = shells,
        .shell_atom_map = workspace.shell_atom_map,
        .p_mat = p_mat,
        .n = n,
        .hf_frac = hf_frac,
    };
    try accumulate_two_electron_gradient(alloc, &two_electron);
    try accumulate_xc_gradient(
        alloc,
        grad,
        shells,
        workspace.basis_map,
        workspace.shell_atom_map,
        p_mat,
        grid_points,
        xc_func,
        n,
    );

    return .{ .gradients = grad };
}

// ============================================================================
// Tests
// ============================================================================

test "KS-DFT LDA gradient H2 STO-3G vs finite difference" {
    const io = std.testing.io;
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
        .{
            .x = nuc_positions[0].x,
            .y = nuc_positions[0].y,
            .z = nuc_positions[0].z,
            .z_number = 1,
        },
        .{
            .x = nuc_positions[1].x,
            .y = nuc_positions[1].y,
            .z = nuc_positions[1].z,
            .z_number = 1,
        },
    };
    const grid_config = becke.GridConfig{
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
    };
    const grid_points = try becke.build_molecular_grid(alloc, &atoms, grid_config);
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
    var ks_result = try kohn_sham.run_kohn_sham_scf(
        alloc,
        io,
        &shells,
        &nuc_positions,
        &nuc_charges,
        2, // 2 electrons
        ks_params,
    );
    defer ks_result.deinit(alloc);

    try testing.expect(ks_result.converged);

    // Compute analytical gradient
    var grad_result = try compute_ks_dft_gradient(
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

    // Translational invariance check
    const sum_x = grad[0].x + grad[1].x;
    const sum_y = grad[0].y + grad[1].y;
    const sum_z = grad[0].z + grad[1].z;

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

    var ks_p = try kohn_sham.run_kohn_sham_scf(
        alloc,
        io,
        &shells_p,
        &pos_p,
        &nuc_charges,
        2,
        ks_params,
    );
    defer ks_p.deinit(alloc);

    var ks_m = try kohn_sham.run_kohn_sham_scf(
        alloc,
        io,
        &shells_m,
        &pos_m,
        &nuc_charges,
        2,
        ks_params,
    );
    defer ks_m.deinit(alloc);

    const num_grad_x = (ks_p.total_energy - ks_m.total_energy) / (2.0 * delta);

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
    const io = std.testing.io;
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
        .{
            .x = nuc_positions[0].x,
            .y = nuc_positions[0].y,
            .z = nuc_positions[0].z,
            .z_number = 8,
        },
        .{
            .x = nuc_positions[1].x,
            .y = nuc_positions[1].y,
            .z = nuc_positions[1].z,
            .z_number = 1,
        },
        .{
            .x = nuc_positions[2].x,
            .y = nuc_positions[2].y,
            .z = nuc_positions[2].z,
            .z_number = 1,
        },
    };
    const grid_config = becke.GridConfig{
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
    };
    const grid_points = try becke.build_molecular_grid(alloc, &atoms, grid_config);
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
    var ks_result = try kohn_sham.run_kohn_sham_scf(
        alloc,
        io,
        &shells,
        &nuc_positions,
        &nuc_charges,
        10, // 10 electrons
        ks_params,
    );
    defer ks_result.deinit(alloc);

    try testing.expect(ks_result.converged);

    // Compute analytical gradient
    var grad_result = try compute_ks_dft_gradient(
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
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_x, 1e-5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_y, 1e-5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum_z, 1e-5);
}

test "KS-DFT B3LYP gradient H2 STO-3G vs finite difference" {
    const io = std.testing.io;
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
        .{
            .x = nuc_positions[0].x,
            .y = nuc_positions[0].y,
            .z = nuc_positions[0].z,
            .z_number = 1,
        },
        .{
            .x = nuc_positions[1].x,
            .y = nuc_positions[1].y,
            .z = nuc_positions[1].z,
            .z_number = 1,
        },
    };
    const grid_config = becke.GridConfig{
        .n_radial = 50,
        .n_angular = 194,
        .prune = false,
    };
    const grid_points = try becke.build_molecular_grid(alloc, &atoms, grid_config);
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
    var ks_result = try kohn_sham.run_kohn_sham_scf(
        alloc,
        io,
        &shells,
        &nuc_positions,
        &nuc_charges,
        2,
        ks_params,
    );
    defer ks_result.deinit(alloc);

    try testing.expect(ks_result.converged);

    // Compute analytical gradient
    var grad_result = try compute_ks_dft_gradient(
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

    // PySCF reference: H1: -0.011277589721, H2: +0.011277589721

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

    var ks_p = try kohn_sham.run_kohn_sham_scf(
        alloc,
        io,
        &shells_p,
        &pos_p,
        &nuc_charges,
        2,
        ks_params,
    );
    defer ks_p.deinit(alloc);

    var ks_m = try kohn_sham.run_kohn_sham_scf(
        alloc,
        io,
        &shells_m,
        &pos_m,
        &nuc_charges,
        2,
        ks_params,
    );
    defer ks_m.deinit(alloc);

    const num_grad_x = (ks_p.total_energy - ks_m.total_energy) / (2.0 * delta);

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
    const io = std.testing.io;
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
        .{
            .x = nuc_positions[0].x,
            .y = nuc_positions[0].y,
            .z = nuc_positions[0].z,
            .z_number = atomic_numbers[0],
        },
        .{
            .x = nuc_positions[1].x,
            .y = nuc_positions[1].y,
            .z = nuc_positions[1].z,
            .z_number = atomic_numbers[1],
        },
        .{
            .x = nuc_positions[2].x,
            .y = nuc_positions[2].y,
            .z = nuc_positions[2].z,
            .z_number = atomic_numbers[2],
        },
    };
    const grid_config = becke.GridConfig{
        .n_radial = 80,
        .n_angular = 302,
    };
    const grid_points = try becke.build_molecular_grid(alloc, &becke_atoms, grid_config);
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

    var ks_result = try kohn_sham.run_kohn_sham_scf(
        alloc,
        io,
        &shells,
        &nuc_positions,
        &nuc_charges,
        10,
        ks_params,
    );
    defer ks_result.deinit(alloc);

    try testing.expect(ks_result.converged);

    // Compute analytical gradient
    var grad_result = try compute_ks_dft_gradient(
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

    // PySCF reference (B3LYP/STO-3G):
    //   O:  0.000000000000  0.000000000000  0.107150408881
    //   H1: 0.000000000000 -0.047576883737 -0.053581380690
    //   H2:-0.000000000000  0.047576883737 -0.053581380690

    // Translational invariance
    const sum_x = grad[0].x + grad[1].x + grad[2].x;
    const sum_y = grad[0].y + grad[1].y + grad[2].y;
    const sum_z = grad[0].z + grad[1].z + grad[2].z;

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

    const grad = try nuclear_repulsion_gradient(alloc, &nuc_positions, &nuc_charges);
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
    var scf_result = try gto_scf.run_general_rhf_scf(
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
    var grad_result = try compute_rhf_gradient(
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
    var scf_result = try gto_scf.run_general_rhf_scf(
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
    var grad_result = try compute_rhf_gradient(
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

    var scf_p = try gto_scf.run_general_rhf_scf(alloc, &shells_p, &pos_p, &nuc_charges, 2, .{});
    defer scf_p.deinit(alloc);

    var scf_m = try gto_scf.run_general_rhf_scf(alloc, &shells_m, &pos_m, &nuc_charges, 2, .{});
    defer scf_m.deinit(alloc);

    const num_grad_x = (scf_p.total_energy - scf_m.total_energy) / (2.0 * delta);

    // Analytical vs numerical gradient should match closely
    const num_tol: f64 = 1e-5;
    try testing.expectApproxEqAbs(num_grad_x, grad[0].x, num_tol);

    // Translational invariance
    const sum_x = grad[0].x + grad[1].x;
    const sum_y = grad[0].y + grad[1].y;
    const sum_z = grad[0].z + grad[1].z;
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

        const analytical = prim_overlap_deriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{
                .x = center_a.x + d.x,
                .y = center_a.y + d.y,
                .z = center_a.z + d.z,
            };
            const ca_m = Vec3{
                .x = center_a.x - d.x,
                .y = center_a.y - d.y,
                .z = center_a.z - d.z,
            };
            const s_p = obara_saika.primitive_overlap(
                alpha_a,
                ca_p,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const s_m = obara_saika.primitive_overlap(
                alpha_a,
                ca_m,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const fd = (s_p - s_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 2: (p_y, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = prim_overlap_deriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{
                .x = center_a.x + d.x,
                .y = center_a.y + d.y,
                .z = center_a.z + d.z,
            };
            const ca_m = Vec3{
                .x = center_a.x - d.x,
                .y = center_a.y - d.y,
                .z = center_a.z - d.z,
            };
            const s_p = obara_saika.primitive_overlap(
                alpha_a,
                ca_p,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const s_m = obara_saika.primitive_overlap(
                alpha_a,
                ca_m,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const fd = (s_p - s_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 3: (p_z, p_y) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 0, .z = 1 };
        const ang_b = AngularMomentum{ .x = 0, .y = 1, .z = 0 };

        const analytical = prim_overlap_deriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{
                .x = center_a.x + d.x,
                .y = center_a.y + d.y,
                .z = center_a.z + d.z,
            };
            const ca_m = Vec3{
                .x = center_a.x - d.x,
                .y = center_a.y - d.y,
                .z = center_a.z - d.z,
            };
            const s_p = obara_saika.primitive_overlap(
                alpha_a,
                ca_p,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const s_m = obara_saika.primitive_overlap(
                alpha_a,
                ca_m,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
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

        const analytical = prim_kinetic_deriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{
                .x = center_a.x + d.x,
                .y = center_a.y + d.y,
                .z = center_a.z + d.z,
            };
            const ca_m = Vec3{
                .x = center_a.x - d.x,
                .y = center_a.y - d.y,
                .z = center_a.z - d.z,
            };
            const t_p = obara_saika.primitive_kinetic(
                alpha_a,
                ca_p,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const t_m = obara_saika.primitive_kinetic(
                alpha_a,
                ca_m,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const fd = (t_p - t_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 2: (p_y, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = prim_kinetic_deriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{
                .x = center_a.x + d.x,
                .y = center_a.y + d.y,
                .z = center_a.z + d.z,
            };
            const ca_m = Vec3{
                .x = center_a.x - d.x,
                .y = center_a.y - d.y,
                .z = center_a.z - d.z,
            };
            const t_p = obara_saika.primitive_kinetic(
                alpha_a,
                ca_p,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const t_m = obara_saika.primitive_kinetic(
                alpha_a,
                ca_m,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const fd = (t_p - t_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 3: (p_z, p_y) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 0, .z = 1 };
        const ang_b = AngularMomentum{ .x = 0, .y = 1, .z = 0 };

        const analytical = prim_kinetic_deriv(alpha_a, center_a, ang_a, alpha_b, center_b, ang_b);

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{
                .x = center_a.x + d.x,
                .y = center_a.y + d.y,
                .z = center_a.z + d.z,
            };
            const ca_m = Vec3{
                .x = center_a.x - d.x,
                .y = center_a.y - d.y,
                .z = center_a.z - d.z,
            };
            const t_p = obara_saika.primitive_kinetic(
                alpha_a,
                ca_p,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
            const t_m = obara_saika.primitive_kinetic(
                alpha_a,
                ca_m,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
            );
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

        const analytical = prim_nuclear_deriv(
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
            nuc_pos,
            nuc_charge,
        );

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{
                .x = center_a.x + d.x,
                .y = center_a.y + d.y,
                .z = center_a.z + d.z,
            };
            const ca_m = Vec3{
                .x = center_a.x - d.x,
                .y = center_a.y - d.y,
                .z = center_a.z - d.z,
            };
            const v_p = obara_saika.primitive_nuclear_attraction(
                alpha_a,
                ca_p,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
                nuc_pos,
                nuc_charge,
            );
            const v_m = obara_saika.primitive_nuclear_attraction(
                alpha_a,
                ca_m,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
                nuc_pos,
                nuc_charge,
            );
            const fd = (v_p - v_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }

    // ---- Test case 2: (p_y, s) ----
    {
        const ang_a = AngularMomentum{ .x = 0, .y = 1, .z = 0 };
        const ang_b = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

        const analytical = prim_nuclear_deriv(
            alpha_a,
            center_a,
            ang_a,
            alpha_b,
            center_b,
            ang_b,
            nuc_pos,
            nuc_charge,
        );

        const dirs = [3]Vec3{
            .{ .x = delta, .y = 0.0, .z = 0.0 },
            .{ .x = 0.0, .y = delta, .z = 0.0 },
            .{ .x = 0.0, .y = 0.0, .z = delta },
        };
        for (dirs, 0..) |d, i| {
            const ca_p = Vec3{
                .x = center_a.x + d.x,
                .y = center_a.y + d.y,
                .z = center_a.z + d.z,
            };
            const ca_m = Vec3{
                .x = center_a.x - d.x,
                .y = center_a.y - d.y,
                .z = center_a.z - d.z,
            };
            const v_p = obara_saika.primitive_nuclear_attraction(
                alpha_a,
                ca_p,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
                nuc_pos,
                nuc_charge,
            );
            const v_m = obara_saika.primitive_nuclear_attraction(
                alpha_a,
                ca_m,
                ang_a,
                alpha_b,
                center_b,
                ang_b,
                nuc_pos,
                nuc_charge,
            );
            const fd = (v_p - v_m) / (2.0 * delta);
            try testing.expectApproxEqAbs(fd, analytical[i], tol);
        }
    }
}
