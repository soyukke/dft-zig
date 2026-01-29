//! Kinetic energy integrals T_μν = <μ|-½∇²|ν> for s-type Gaussians.
//!
//! For two primitive s-type Gaussians with exponents a, b centered at A, B:
//!   T_ij = a×b/(a+b) × (3 - 2×a×b/(a+b)×|A-B|²) × S_ij
//! where S_ij is the primitive overlap integral.
//!
//! Note: This is the kinetic energy in Hartree units (-½∇²).

const std = @import("std");
const math = @import("../math/math.zig");
const basis = @import("../basis/basis.zig");

const PrimitiveGaussian = basis.PrimitiveGaussian;
const ContractedShell = basis.ContractedShell;
const normalizationS = basis.normalizationS;

/// Compute kinetic energy integral between two contracted s-type shells.
/// Returns the integral in Hartree atomic units.
pub fn kineticSS(shell_a: ContractedShell, shell_b: ContractedShell) f64 {
    std.debug.assert(shell_a.l == 0 and shell_b.l == 0);

    var result: f64 = 0.0;
    const diff = math.Vec3.sub(shell_a.center, shell_b.center);
    const r2 = math.Vec3.dot(diff, diff);

    for (shell_a.primitives) |prim_a| {
        const na = normalizationS(prim_a.alpha);
        for (shell_b.primitives) |prim_b| {
            const nb = normalizationS(prim_b.alpha);
            const a = prim_a.alpha;
            const b = prim_b.alpha;
            const p = a + b;
            const mu = a * b / p; // reduced exponent

            // Primitive overlap
            const prefactor = std.math.pow(f64, std.math.pi / p, 1.5);
            const exponential = @exp(-mu * r2);
            const s_prim = prefactor * exponential;

            // Kinetic energy: T = mu × (3 - 2×mu×r²) × S
            const t_prim = mu * (3.0 - 2.0 * mu * r2) * s_prim;

            result += prim_a.coeff * prim_b.coeff * na * nb * t_prim;
        }
    }
    return result;
}

/// Build the full kinetic energy matrix T for a set of s-type shells.
/// Returns a flat row-major n×n matrix.
pub fn buildKineticMatrix(alloc: std.mem.Allocator, shells: []const ContractedShell) ![]f64 {
    const n = shells.len;
    const mat = try alloc.alloc(f64, n * n);
    for (0..n) |i| {
        for (0..n) |j| {
            if (j >= i) {
                mat[i * n + j] = kineticSS(shells[i], shells[j]);
                mat[j * n + i] = mat[i * n + j];
            }
        }
    }
    return mat;
}

test "kinetic energy positive" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const shell = ContractedShell{
        .center = center,
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const t = kineticSS(shell, shell);
    // Kinetic energy of hydrogen 1s should be positive
    try testing.expect(t > 0.0);
    // STO-3G H 1s kinetic energy ≈ 0.7600 Hartree
    try testing.expectApproxEqAbs(t, 0.7600, 0.01);
}
