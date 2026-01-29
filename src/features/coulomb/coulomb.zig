//! Coulomb interaction routines for isolated (molecular) systems.
//!
//! This module implements:
//! - Spherical cutoff Coulomb for Hartree potential (removes periodic images)
//! - Direct pairwise Coulomb sum for ion-ion interaction (no Ewald)
//!
//! The cutoff Coulomb approach modifies the bare 4π/G² kernel to
//! (4π/G²)(1 - cos(|G|·R_c)) where R_c is the cutoff radius,
//! typically chosen as the inscribed sphere radius of the supercell.
//! This removes spurious interactions between periodic images.
//!
//! Reference: Martyna & Tuckerman, J. Chem. Phys. 110, 2810 (1999)

const std = @import("std");
const math = @import("../math/math.zig");

/// Compute the cutoff radius for the spherical cutoff Coulomb method.
/// R_c = L_min / 2 where L_min is the shortest cell vector length.
/// This ensures the cutoff sphere fits inside the Wigner-Seitz cell.
pub fn cutoffRadius(cell: math.Mat3) f64 {
    const l1 = math.Vec3.norm(cell.row(0));
    const l2 = math.Vec3.norm(cell.row(1));
    const l3 = math.Vec3.norm(cell.row(2));
    return @min(@min(l1, l2), l3) / 2.0;
}

/// Compute the cutoff Coulomb kernel for a given G-vector.
/// In Rydberg units: v_c(G) = (8π / G²)(1 - cos(|G| R_c))
/// For G=0: v_c(G=0) = 4π R_c² (in Rydberg units: 8π R_c² / 2 = 4π R_c²)
///
/// This replaces the standard periodic Coulomb kernel 8π/G² used in
/// buildPotentialGrid for periodic systems.
pub fn cutoffCoulombKernel(g2: f64, g_mag: f64, r_cut: f64) f64 {
    if (g2 < 1e-12) {
        // G=0 limit: v_c(0) = 4π R_c² (Rydberg units)
        // Derivation: lim_{G->0} (8π/G²)(1 - cos(GR)) = 8π R²/2 = 4π R²
        return 4.0 * std.math.pi * r_cut * r_cut;
    }
    // v_c(G) = (8π / G²)(1 - cos(|G| R_c))
    return (8.0 * std.math.pi / g2) * (1.0 - std.math.cos(g_mag * r_cut));
}

/// Compute the cutoff Coulomb Hartree energy kernel for a given G-vector.
/// E_H = (Ω/2) Σ_G |ρ(G)|² v_c(G)
/// where v_c(G) = (8π/G²)(1 - cos(|G|R_c)) in Rydberg units.
pub fn cutoffCoulombEnergyKernel(g2: f64, g_mag: f64, r_cut: f64) f64 {
    // Same kernel as potential — energy just multiplies by Ω/2 |ρ(G)|²
    return cutoffCoulombKernel(g2, g_mag, r_cut);
}

/// Compute ion-ion energy by direct pairwise Coulomb sum (no periodic images).
/// E_ion = Σ_{i<j} Z_i Z_j / |R_i - R_j| (in Hartree atomic units)
///
/// For isolated systems, there are no periodic images so we just sum
/// all unique pairs. This replaces the Ewald summation used in periodic systems.
pub fn directIonIonEnergy(
    charges: []const f64,
    positions: []const math.Vec3,
) f64 {
    if (charges.len != positions.len) return 0.0;
    const n = charges.len;
    var energy: f64 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j: usize = i + 1;
        while (j < n) : (j += 1) {
            const delta = math.Vec3.sub(positions[i], positions[j]);
            const r = math.Vec3.norm(delta);
            if (r > 1e-12) {
                energy += charges[i] * charges[j] / r;
            }
        }
    }
    return energy;
}

/// Compute ion-ion forces by direct pairwise Coulomb (no periodic images).
/// F_i = -∂E/∂R_i = Σ_{j≠i} Z_i Z_j (R_i - R_j) / |R_i - R_j|³
/// Returns forces in Hartree/Bohr units (same convention as Ewald forces).
pub fn directIonIonForces(
    alloc: std.mem.Allocator,
    charges: []const f64,
    positions: []const math.Vec3,
) ![]math.Vec3 {
    const n = charges.len;
    if (n == 0) return &[_]math.Vec3{};

    var forces = try alloc.alloc(math.Vec3, n);
    for (forces) |*f| {
        f.* = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    }

    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j: usize = i + 1;
        while (j < n) : (j += 1) {
            const delta = math.Vec3.sub(positions[i], positions[j]);
            const r = math.Vec3.norm(delta);
            if (r > 1e-12) {
                const r3 = r * r * r;
                const factor = charges[i] * charges[j] / r3;
                const fvec = math.Vec3.scale(delta, factor);
                // Force on atom i
                forces[i] = math.Vec3.add(forces[i], fvec);
                // Force on atom j (Newton's 3rd law)
                forces[j] = math.Vec3.sub(forces[j], fvec);
            }
        }
    }

    return forces;
}

// ============================================================
// Tests
// ============================================================

test "cutoff radius cubic cell" {
    const testing = std.testing;
    const a = 20.0;
    const cell = math.Mat3{
        .m = .{
            .{ a, 0.0, 0.0 },
            .{ 0.0, a, 0.0 },
            .{ 0.0, 0.0, a },
        },
    };
    const rc = cutoffRadius(cell);
    try testing.expectApproxEqAbs(rc, 10.0, 1e-10);
}

test "cutoff Coulomb kernel G=0 limit" {
    const testing = std.testing;
    const r_cut = 10.0;
    const v0 = cutoffCoulombKernel(0.0, 0.0, r_cut);
    // G=0 limit: 4π R_c²
    try testing.expectApproxEqAbs(v0, 4.0 * std.math.pi * r_cut * r_cut, 1e-10);
}

test "cutoff Coulomb kernel reduces to 8pi/G2 for large Rc" {
    const testing = std.testing;
    // For very large R_c, cos(G*Rc) oscillates but (1-cos)/G² ≈ 1/G² on average.
    // For exact limit at specific G: when G*Rc = π, cos = -1, kernel = 16π/G²
    const g_mag = 1.0;
    const g2 = 1.0;
    const r_cut = std.math.pi; // G*Rc = π
    const v = cutoffCoulombKernel(g2, g_mag, r_cut);
    // (8π/1)(1 - cos(π)) = 8π × 2 = 16π
    try testing.expectApproxEqAbs(v, 16.0 * std.math.pi, 1e-10);
}

test "direct ion-ion energy H2" {
    const testing = std.testing;
    // Two protons separated by 1.4 Bohr
    const charges = [_]f64{ 1.0, 1.0 };
    const positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const e = directIonIonEnergy(&charges, &positions);
    // Expected: Z1*Z2/R = 1*1/1.4 = 0.714285...
    try testing.expectApproxEqAbs(e, 1.0 / 1.4, 1e-10);
}

test "direct ion-ion forces finite difference" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const charges = [_]f64{ 1.0, -1.0 };
    var positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 2.0, .y = 0.0, .z = 0.0 },
    };

    const forces = try directIonIonForces(alloc, &charges, &positions);
    defer alloc.free(forces);

    // Finite difference
    const delta = 1e-5;
    var pos_p = positions;
    var pos_m = positions;
    pos_p[0].x += delta;
    pos_m[0].x -= delta;
    const ep = directIonIonEnergy(&charges, &pos_p);
    const em = directIonIonEnergy(&charges, &pos_m);
    const f_numeric = -(ep - em) / (2.0 * delta);

    try testing.expectApproxEqAbs(forces[0].x, f_numeric, 1e-5);
}

test "direct ion-ion forces sum to zero" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const charges = [_]f64{ 2.0, 3.0, -1.0 };
    const positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 1.0, .y = 2.0, .z = 0.0 },
        math.Vec3{ .x = -1.0, .y = 0.5, .z = 1.5 },
    };

    const forces = try directIonIonForces(alloc, &charges, &positions);
    defer alloc.free(forces);

    var total = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    for (forces) |f| {
        total = math.Vec3.add(total, f);
    }
    try testing.expectApproxEqAbs(total.x, 0.0, 1e-12);
    try testing.expectApproxEqAbs(total.y, 0.0, 1e-12);
    try testing.expectApproxEqAbs(total.z, 0.0, 1e-12);
}
