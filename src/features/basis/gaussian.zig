//! Gaussian-type orbital (GTO) basis function definitions.
//!
//! A primitive Gaussian is: g(r) = N × r^l × exp(-α|r-R|²)
//! where α is the exponent, R is the center, and l is the angular momentum.
//!
//! A contracted Gaussian shell is a fixed linear combination of primitives:
//!   φ(r) = Σ_i c_i × g_i(r)
//! sharing the same center and angular momentum.

const std = @import("std");
const math = @import("../math/math.zig");

/// A single primitive Gaussian function.
/// g(r) = coeff × (normalization) × exp(-alpha × |r - center|²)
pub const PrimitiveGaussian = struct {
    /// Orbital exponent (in bohr⁻²).
    alpha: f64,
    /// Contraction coefficient (unnormalized — normalization is applied separately).
    coeff: f64,
};

/// A contracted Gaussian shell: a linear combination of primitives
/// centered at the same point with the same angular momentum.
pub const ContractedShell = struct {
    /// Center of the shell in Cartesian coordinates (bohr).
    center: math.Vec3,
    /// Angular momentum quantum number (0=s, 1=p, 2=d, ...).
    l: u32,
    /// Primitive Gaussians in this contraction.
    primitives: []const PrimitiveGaussian,

    /// Number of primitives in the contraction.
    pub fn num_primitives(self: ContractedShell) usize {
        return self.primitives.len;
    }

    /// Number of Cartesian functions in this shell: (l+1)(l+2)/2.
    pub fn num_cartesian_functions(self: ContractedShell) usize {
        return num_cartesian(self.l);
    }
};

/// Cartesian angular momentum indices (a_x, a_y, a_z) where a_x + a_y + a_z = l.
/// For example: s → (0,0,0), p → (1,0,0),(0,1,0),(0,0,1), etc.
pub const AngularMomentum = struct {
    x: u32,
    y: u32,
    z: u32,

    pub fn total(self: AngularMomentum) u32 {
        return self.x + self.y + self.z;
    }
};

/// Number of Cartesian components for angular momentum l: (l+1)(l+2)/2.
pub fn num_cartesian(l: u32) usize {
    return @as(usize, (l + 1) * (l + 2) / 2);
}

/// Get all Cartesian angular momentum tuples for a given l.
/// Ordered as: iterate z descending for each x ascending
/// s: (0,0,0)
/// p: (1,0,0), (0,1,0), (0,0,1)
/// d: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
pub fn cartesian_exponents(l: u32) [MAX_CART]AngularMomentum {
    var result: [MAX_CART]AngularMomentum = undefined;
    var idx: usize = 0;
    var ax: u32 = l;
    while (true) {
        var ay: u32 = l - ax;
        while (true) {
            const az: u32 = l - ax - ay;
            result[idx] = .{ .x = ax, .y = ay, .z = az };
            idx += 1;
            if (ay == 0) break;
            ay -= 1;
        }
        if (ax == 0) break;
        ax -= 1;
    }
    // Fill remaining with zeros
    while (idx < MAX_CART) {
        result[idx] = .{ .x = 0, .y = 0, .z = 0 };
        idx += 1;
    }
    return result;
}

/// Maximum number of Cartesian components we support (up to l=4, f-type: 15).
pub const MAX_CART = 15;

/// A basis function (individual orbital) within a shell.
/// For s-type (l=0): 1 function per shell.
/// For p-type (l=1): 3 functions (px, py, pz) per shell.
/// For d-type (l=2): 5 or 6 functions per shell.
pub const BasisFunction = struct {
    /// Index of the shell this function belongs to.
    shell_index: usize,
    /// Angular momentum quantum number.
    l: u32,
    /// Magnetic quantum number index (for Cartesian: encodes x^a y^b z^c).
    m: i32,
    /// Center of the function (bohr).
    center: math.Vec3,
    /// Primitives (shared with the shell).
    primitives: []const PrimitiveGaussian,
};

/// Normalization constant for a primitive s-type Gaussian.
/// N = (2α/π)^(3/4)
pub fn normalization_s(alpha: f64) f64 {
    return std.math.pow(f64, 2.0 * alpha / std.math.pi, 0.75);
}

/// Normalization constant for a Cartesian Gaussian primitive:
///   g_{a,b,c}(r) = N × x^a × y^b × z^c × exp(-α|r-R|²)
///
/// N(α; a,b,c) = (2α/π)^(3/4) × (4α)^((a+b+c)/2) / sqrt((2a-1)!! × (2b-1)!! × (2c-1)!!)
///
/// where (2n-1)!! = 1×3×5×...×(2n-1), and (-1)!! = 1 by convention.
pub fn normalization(alpha: f64, ax: u32, ay: u32, az: u32) f64 {
    const l_total = ax + ay + az;
    const s_norm = std.math.pow(f64, 2.0 * alpha / std.math.pi, 0.75);
    const angular_factor = std.math.pow(f64, 4.0 * alpha, @as(f64, @floatFromInt(l_total)) / 2.0);
    const df = double_factorial(ax) * double_factorial(ay) * double_factorial(az);
    return s_norm * angular_factor / @sqrt(df);
}

/// Compute (2n-1)!! = 1 × 3 × 5 × ... × (2n-1).
/// Convention: (-1)!! = 1, (0-1)!! = 1, so double_factorial(0) = 1.
pub fn double_factorial(n: u32) f64 {
    if (n == 0) return 1.0;
    var result: f64 = 1.0;
    var k: u32 = 1;
    while (k <= 2 * n - 1) : (k += 2) {
        result *= @as(f64, @floatFromInt(k));
    }
    return result;
}

/// Compute the overlap between two primitive s-type Gaussians
/// centered at A and B with exponents a and b.
/// S = (π / (a+b))^(3/2) × exp(-a×b/(a+b) × |A-B|²)
pub fn primitive_overlap_ss(a: f64, center_a: math.Vec3, b: f64, center_b: math.Vec3) f64 {
    const p = a + b;
    const diff = math.Vec3.sub(center_a, center_b);
    const r2 = math.Vec3.dot(diff, diff);
    const prefactor = std.math.pow(f64, std.math.pi / p, 1.5);
    return prefactor * @exp(-a * b / p * r2);
}

test "normalization s-type" {
    const testing = std.testing;
    // For alpha=1.0: N = (2/π)^(3/4) ≈ 0.71271
    const n = normalization_s(1.0);
    try testing.expectApproxEqAbs(0.71270547036, n, 1e-6);
}

test "normalization general s equals normalization_s" {
    const testing = std.testing;
    // normalization(alpha, 0,0,0) should equal normalization_s(alpha)
    const alpha = 1.5;
    const ns = normalization_s(alpha);
    const ng = normalization(alpha, 0, 0, 0);
    try testing.expectApproxEqAbs(ns, ng, 1e-12);
}

test "normalization p-type" {
    const testing = std.testing;
    // For a px primitive with alpha=1.0:
    // N(1.0; 1,0,0) = (2/π)^(3/4) × (4)^(1/2) / sqrt(1) = (2/π)^(3/4) × 2
    const n = normalization(1.0, 1, 0, 0);
    const expected = std.math.pow(f64, 2.0 / std.math.pi, 0.75) * 2.0;
    try testing.expectApproxEqAbs(expected, n, 1e-10);
}

test "cartesian_exponents s/p/d" {
    const testing = std.testing;
    // s: 1 component
    try testing.expectEqual(@as(usize, 1), num_cartesian(0));
    // p: 3 components
    try testing.expectEqual(@as(usize, 3), num_cartesian(1));
    // d: 6 components (Cartesian)
    try testing.expectEqual(@as(usize, 6), num_cartesian(2));

    // Check s exponents
    const s_exp = cartesian_exponents(0);
    try testing.expectEqual(@as(u32, 0), s_exp[0].x);
    try testing.expectEqual(@as(u32, 0), s_exp[0].y);
    try testing.expectEqual(@as(u32, 0), s_exp[0].z);

    // Check p exponents: should be (1,0,0), (0,1,0), (0,0,1)
    const p_exp = cartesian_exponents(1);
    try testing.expectEqual(@as(u32, 1), p_exp[0].x);
    try testing.expectEqual(@as(u32, 0), p_exp[1].x);
    try testing.expectEqual(@as(u32, 0), p_exp[2].x);
    try testing.expectEqual(@as(u32, 0), p_exp[0].z);
    try testing.expectEqual(@as(u32, 0), p_exp[1].z);
    try testing.expectEqual(@as(u32, 1), p_exp[2].z);
}

test "double_factorial" {
    const testing = std.testing;
    // (2*0-1)!! = (-1)!! = 1 by convention
    try testing.expectApproxEqAbs(@as(f64, 1.0), double_factorial(0), 1e-12);
    // (2*1-1)!! = 1!! = 1
    try testing.expectApproxEqAbs(@as(f64, 1.0), double_factorial(1), 1e-12);
    // (2*2-1)!! = 3!! = 1*3 = 3
    try testing.expectApproxEqAbs(@as(f64, 3.0), double_factorial(2), 1e-12);
    // (2*3-1)!! = 5!! = 1*3*5 = 15
    try testing.expectApproxEqAbs(@as(f64, 15.0), double_factorial(3), 1e-12);
}

test "primitive overlap identical" {
    const testing = std.testing;
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    // Two identical s primitives with alpha=1.0 at origin
    // S = (π/2)^(3/2) ≈ 2.4674
    const s = primitive_overlap_ss(1.0, center, 1.0, center);
    try testing.expectApproxEqAbs(std.math.pow(f64, std.math.pi / 2.0, 1.5), s, 1e-10);
}
