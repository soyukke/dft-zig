//! Overlap integrals S_μν = <μ|ν> for s-type Gaussian basis functions.
//!
//! For two contracted s-type shells A and B:
//!   S_AB = Σ_i Σ_j c_i c_j N_i N_j × s_ij
//! where s_ij is the primitive overlap:
//!   s_ij = (π/(α_i + α_j))^(3/2) × exp(-α_i α_j/(α_i+α_j) × |A-B|²)

const std = @import("std");
const math = @import("../math/math.zig");
const basis = @import("../basis/basis.zig");

const PrimitiveGaussian = basis.PrimitiveGaussian;
const ContractedShell = basis.ContractedShell;
const normalizationS = basis.normalizationS;

/// Compute overlap integral between two contracted s-type shells.
pub fn overlapSS(shell_a: ContractedShell, shell_b: ContractedShell) f64 {
    std.debug.assert(shell_a.l == 0 and shell_b.l == 0);

    var result: f64 = 0.0;
    const diff = math.Vec3.sub(shell_a.center, shell_b.center);
    const r2 = math.Vec3.dot(diff, diff);

    for (shell_a.primitives) |prim_a| {
        const na = normalizationS(prim_a.alpha);
        for (shell_b.primitives) |prim_b| {
            const nb = normalizationS(prim_b.alpha);
            const p = prim_a.alpha + prim_b.alpha;
            const prefactor = std.math.pow(f64, std.math.pi / p, 1.5);
            const exponential = @exp(-prim_a.alpha * prim_b.alpha / p * r2);
            result += prim_a.coeff * prim_b.coeff * na * nb * prefactor * exponential;
        }
    }
    return result;
}

/// Build the full overlap matrix S for a set of s-type shells.
/// Returns a flat row-major n×n matrix.
pub fn buildOverlapMatrix(alloc: std.mem.Allocator, shells: []const ContractedShell) ![]f64 {
    const n = shells.len;
    const mat = try alloc.alloc(f64, n * n);
    for (0..n) |i| {
        for (0..n) |j| {
            if (j >= i) {
                mat[i * n + j] = overlapSS(shells[i], shells[j]);
                mat[j * n + i] = mat[i * n + j]; // symmetric
            }
        }
    }
    return mat;
}

test "overlap self-overlap is 1.0 for normalized STO-3G H" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const shell = ContractedShell{
        .center = center,
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const s = overlapSS(shell, shell);
    // Self-overlap of a properly normalized contracted shell should be ~1.0
    try testing.expectApproxEqAbs(s, 1.0, 1e-4);
}

test "overlap decreases with distance" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const a = ContractedShell{
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const b_near = ContractedShell{
        .center = .{ .x = 1.0, .y = 0.0, .z = 0.0 },
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const b_far = ContractedShell{
        .center = .{ .x = 5.0, .y = 0.0, .z = 0.0 },
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const s_near = overlapSS(a, b_near);
    const s_far = overlapSS(a, b_far);
    try testing.expect(s_near > s_far);
    try testing.expect(s_far >= 0.0);
}
