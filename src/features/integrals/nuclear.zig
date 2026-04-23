//! Nuclear attraction integrals V_μν = <μ| -Z/|r-R_C| |ν> for s-type Gaussians.
//!
//! For two primitive s-type Gaussians with exponents a, b centered at A, B,
//! and a nucleus with charge Z at position C:
//!
//!   V_ij = -Z × (2π/(a+b)) × exp(-a×b/(a+b) × |A-B|²) × F_0(p × |P-C|²)
//!
//! where p = a+b, P = (a×A + b×B)/(a+b) is the Gaussian product center,
//! and F_0 is the Boys function.

const std = @import("std");
const math = @import("../math/math.zig");
const basis = @import("../basis/basis.zig");
const boys = @import("boys.zig");

const PrimitiveGaussian = basis.PrimitiveGaussian;
const ContractedShell = basis.ContractedShell;
const normalization_s = basis.normalization_s;

/// Compute nuclear attraction integral between two contracted s-type shells
/// for a single nucleus at position `nuc_pos` with charge `z`.
/// Returns the integral in Hartree atomic units (negative for attraction).
pub fn nuclear_attraction_ss(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    nuc_pos: math.Vec3,
    z: f64,
) f64 {
    std.debug.assert(shell_a.l == 0 and shell_b.l == 0);

    var result: f64 = 0.0;
    const diff_ab = math.Vec3.sub(shell_a.center, shell_b.center);
    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);

    for (shell_a.primitives) |prim_a| {
        const na = normalization_s(prim_a.alpha);
        for (shell_b.primitives) |prim_b| {
            const nb = normalization_s(prim_b.alpha);
            const a = prim_a.alpha;
            const b = prim_b.alpha;
            const p = a + b;

            // Gaussian product center P = (a*A + b*B) / p
            const p_center = math.Vec3{
                .x = (a * shell_a.center.x + b * shell_b.center.x) / p,
                .y = (a * shell_a.center.y + b * shell_b.center.y) / p,
                .z = (a * shell_a.center.z + b * shell_b.center.z) / p,
            };

            // |P - C|²
            const diff_pc = math.Vec3.sub(p_center, nuc_pos);
            const r2_pc = math.Vec3.dot(diff_pc, diff_pc);

            const exponential = @exp(-a * b / p * r2_ab);
            const f0 = boys.boys0(p * r2_pc);

            // V = -Z × 2π/p × exp(...) × F_0(...)
            const v_prim = -z * 2.0 * std.math.pi / p * exponential * f0;

            result += prim_a.coeff * prim_b.coeff * na * nb * v_prim;
        }
    }
    return result;
}

/// Compute total nuclear attraction integral for all nuclei.
/// V_μν = Σ_C V_μν^C
pub fn total_nuclear_attraction(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
) f64 {
    var result: f64 = 0.0;
    for (nuc_positions, 0..) |pos, i| {
        result += nuclear_attraction_ss(shell_a, shell_b, pos, nuc_charges[i]);
    }
    return result;
}

/// Build the full nuclear attraction matrix V for a set of s-type shells.
/// Returns a flat row-major n×n matrix.
pub fn build_nuclear_matrix(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
) ![]f64 {
    const n = shells.len;
    const mat = try alloc.alloc(f64, n * n);
    for (0..n) |i| {
        for (0..n) |j| {
            if (j >= i) {
                mat[i * n + j] = total_nuclear_attraction(
                    shells[i],
                    shells[j],
                    nuc_positions,
                    nuc_charges,
                );
                mat[j * n + i] = mat[i * n + j];
            }
        }
    }
    return mat;
}

test "nuclear attraction negative for H at origin" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const shell = ContractedShell{
        .center = center,
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const v = nuclear_attraction_ss(shell, shell, center, 1.0);
    // Nuclear attraction should be negative (attractive)
    try testing.expect(v < 0.0);
    // STO-3G H 1s nuclear attraction ≈ -1.2266 Hartree
    try testing.expectApproxEqAbs(v, -1.2266, 0.01);
}
