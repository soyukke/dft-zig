//! Electron repulsion integrals (ERI) for s-type Gaussians.
//!
//! Two-electron integral (μν|λσ) = ∫∫ μ(r₁)ν(r₁) × 1/|r₁-r₂| × λ(r₂)σ(r₂) dr₁ dr₂
//!
//! For four primitive s-type Gaussians with exponents a,b,c,d
//! centered at A,B,C,D:
//!
//!   (ab|cd) = 2π^(5/2) / (p·q·√(p+q)) × exp(-a·b/p×|A-B|² - c·d/q×|C-D|²)
//!             × F_0((p·q/(p+q)) × |P-Q|²)
//!
//! where p = a+b, q = c+d, P = (aA+bB)/p, Q = (cC+dD)/q.

const std = @import("std");
const math = @import("../math/math.zig");
const basis = @import("../basis/basis.zig");
const boys = @import("boys.zig");

const PrimitiveGaussian = basis.PrimitiveGaussian;
const ContractedShell = basis.ContractedShell;
const normalization_s = basis.normalization_s;

/// Compute a primitive (ss|ss) ERI.
fn primitive_eri(
    a: f64,
    center_a: math.Vec3,
    b: f64,
    center_b: math.Vec3,
    c: f64,
    center_c: math.Vec3,
    d: f64,
    center_d: math.Vec3,
) f64 {
    const p = a + b;
    const q = c + d;
    const alpha = p * q / (p + q);

    // Gaussian product centers
    const p_center = math.Vec3{
        .x = (a * center_a.x + b * center_b.x) / p,
        .y = (a * center_a.y + b * center_b.y) / p,
        .z = (a * center_a.z + b * center_b.z) / p,
    };
    const q_center = math.Vec3{
        .x = (c * center_c.x + d * center_d.x) / q,
        .y = (c * center_c.y + d * center_d.y) / q,
        .z = (c * center_c.z + d * center_d.z) / q,
    };

    const diff_ab = math.Vec3.sub(center_a, center_b);
    const diff_cd = math.Vec3.sub(center_c, center_d);
    const diff_pq = math.Vec3.sub(p_center, q_center);

    const r2_ab = math.Vec3.dot(diff_ab, diff_ab);
    const r2_cd = math.Vec3.dot(diff_cd, diff_cd);
    const r2_pq = math.Vec3.dot(diff_pq, diff_pq);

    const prefactor = 2.0 * std.math.pow(f64, std.math.pi, 2.5) / (p * q * @sqrt(p + q));
    const exponential = @exp(-a * b / p * r2_ab - c * d / q * r2_cd);
    const f0 = boys.boys0(alpha * r2_pq);

    return prefactor * exponential * f0;
}

/// Compute contracted (ss|ss) ERI for four s-type shells.
pub fn eri_ssss(
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
) f64 {
    std.debug.assert(shell_a.l == 0 and shell_b.l == 0);
    std.debug.assert(shell_c.l == 0 and shell_d.l == 0);

    var result: f64 = 0.0;

    for (shell_a.primitives) |pa| {
        const na = normalization_s(pa.alpha);
        for (shell_b.primitives) |pb| {
            const nb = normalization_s(pb.alpha);
            for (shell_c.primitives) |pc| {
                const nc = normalization_s(pc.alpha);
                for (shell_d.primitives) |pd| {
                    const nd = normalization_s(pd.alpha);

                    const prim = primitive_eri(
                        pa.alpha,
                        shell_a.center,
                        pb.alpha,
                        shell_b.center,
                        pc.alpha,
                        shell_c.center,
                        pd.alpha,
                        shell_d.center,
                    );

                    result += pa.coeff * pb.coeff * pc.coeff * pd.coeff * na * nb * nc * nd * prim;
                }
            }
        }
    }
    return result;
}

/// Compute all unique ERIs for n s-type basis functions.
/// Stores them in a flat array indexed by compound index.
/// Uses 8-fold symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij) etc.
/// Returns a flat array of size n*(n+1)/2 × (n*(n+1)/2 + 1) / 2
/// indexed via compoundIndex.
pub fn build_eri_table(alloc: std.mem.Allocator, shells: []const ContractedShell) !EriTable {
    const n = shells.len;
    const nn = n * (n + 1) / 2;
    const size = nn * (nn + 1) / 2;
    const values = try alloc.alloc(f64, size);
    @memset(values, 0.0);

    for (0..n) |i| {
        for (0..i + 1) |j| {
            const ij = triangular_index(i, j);
            for (0..n) |k| {
                for (0..k + 1) |l| {
                    const kl = triangular_index(k, l);
                    if (kl > ij) continue; // use symmetry (ij|kl) = (kl|ij)
                    const idx = triangular_index(ij, kl);
                    values[idx] = eri_ssss(shells[i], shells[j], shells[k], shells[l]);
                }
            }
        }
    }

    return EriTable{ .values = values, .n = n };
}

/// Triangular index: maps (i,j) with i >= j to i*(i+1)/2 + j.
fn triangular_index(i: usize, j: usize) usize {
    if (i >= j) {
        return i * (i + 1) / 2 + j;
    } else {
        return j * (j + 1) / 2 + i;
    }
}

/// Table of precomputed ERIs with 8-fold symmetry.
pub const EriTable = struct {
    values: []f64,
    n: usize,

    pub fn deinit(self: *EriTable, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
    }

    /// Get the ERI (ij|kl) using symmetry.
    pub fn get(self: EriTable, i: usize, j: usize, k: usize, l: usize) f64 {
        const ij = triangular_index(i, j);
        const kl = triangular_index(k, l);
        const idx = triangular_index(ij, kl);
        return self.values[idx];
    }
};

test "ERI (ss|ss) positive for identical shells" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const shell = ContractedShell{
        .center = center,
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const eri = eri_ssss(shell, shell, shell, shell);
    // (1s1s|1s1s) for H STO-3G should be positive (electron repulsion)
    try testing.expect(eri > 0.0);
    // Known value ≈ 0.7746 Hartree
    try testing.expectApproxEqAbs(eri, 0.7746, 0.01);
}

test "ERI symmetry" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");
    const a = ContractedShell{
        .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const b = ContractedShell{
        .center = .{ .x = 1.4, .y = 0.0, .z = 0.0 },
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    // (ab|ab) == (ba|ab) == (ab|ba) == (ba|ba)
    const eri1 = eri_ssss(a, b, a, b);
    const eri2 = eri_ssss(b, a, a, b);
    const eri3 = eri_ssss(a, b, b, a);
    const eri4 = eri_ssss(b, a, b, a);
    try testing.expectApproxEqAbs(eri1, eri2, 1e-12);
    try testing.expectApproxEqAbs(eri1, eri3, 1e-12);
    try testing.expectApproxEqAbs(eri1, eri4, 1e-12);
}
