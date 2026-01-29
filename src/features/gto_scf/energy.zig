//! Energy evaluation for Restricted Hartree-Fock.
//!
//! Total RHF energy:
//!   E_total = E_elec + V_nn
//!
//! Electronic energy:
//!   E_elec = ½ × Σ_μν P_μν × (H_core_μν + F_μν)
//!          = ½ × Tr[P × (H_core + F)]
//!
//! Nuclear repulsion energy (for point charges):
//!   V_nn = Σ_{A<B} Z_A × Z_B / |R_A - R_B|

const std = @import("std");
const math = @import("../math/math.zig");

/// Compute the electronic energy: E_elec = ½ Tr[P(H_core + F)].
///
/// All matrices are row-major n×n.
pub fn electronicEnergy(
    n: usize,
    p: []const f64,
    h_core: []const f64,
    f: []const f64,
) f64 {
    std.debug.assert(p.len == n * n);
    std.debug.assert(h_core.len == n * n);
    std.debug.assert(f.len == n * n);

    var e_elec: f64 = 0.0;
    for (0..n) |mu| {
        for (0..n) |nu| {
            const idx = mu * n + nu;
            e_elec += p[idx] * (h_core[idx] + f[idx]);
        }
    }
    return 0.5 * e_elec;
}

/// Compute the nuclear repulsion energy V_nn = Σ_{A<B} Z_A Z_B / R_AB.
///
/// Uses Hartree atomic units.
pub fn nuclearRepulsionEnergy(
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
) f64 {
    std.debug.assert(nuc_positions.len == nuc_charges.len);
    const n_atoms = nuc_positions.len;

    var v_nn: f64 = 0.0;
    for (0..n_atoms) |a| {
        for (a + 1..n_atoms) |b| {
            const diff = math.Vec3.sub(nuc_positions[a], nuc_positions[b]);
            const r = @sqrt(math.Vec3.dot(diff, diff));
            v_nn += nuc_charges[a] * nuc_charges[b] / r;
        }
    }
    return v_nn;
}

/// Compute total RHF energy: E_total = E_elec + V_nn.
pub fn totalEnergy(
    n: usize,
    p: []const f64,
    h_core: []const f64,
    f: []const f64,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
) f64 {
    const e_elec = electronicEnergy(n, p, h_core, f);
    const v_nn = nuclearRepulsionEnergy(nuc_positions, nuc_charges);
    return e_elec + v_nn;
}

test "nuclear repulsion H2" {
    const testing = std.testing;
    // H2 at R = 1.4 bohr: V_nn = 1/1.4 = 0.71428... Hartree
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const charges = [_]f64{ 1.0, 1.0 };
    const v_nn = nuclearRepulsionEnergy(&positions, &charges);
    try testing.expectApproxEqAbs(v_nn, 1.0 / 1.4, 1e-12);
}

test "electronic energy with identity-like matrices" {
    // Simple check: P=I, H_core=aI, F=bI
    // E_elec = 0.5 * Tr[I * (a+b)I] = 0.5 * n * (a+b)
    const n: usize = 2;
    const p = [_]f64{ 1.0, 0.0, 0.0, 1.0 };
    const h = [_]f64{ -1.5, 0.0, 0.0, -1.5 };
    const f = [_]f64{ -0.5, 0.0, 0.0, -0.5 };
    const e = electronicEnergy(n, &p, &h, &f);
    try std.testing.expectApproxEqAbs(e, 0.5 * 2.0 * (-1.5 + -0.5), 1e-12);
}
