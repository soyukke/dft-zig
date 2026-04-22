//! 6-31G(2df,p) polarized basis set data.
//!
//! The 6-31G(2df,p) basis extends 6-31G with polarization functions:
//!   - Heavy atoms (C, N, O, F): 2 sets of d functions + 1 set of f functions
//!   - Hydrogen: 1 set of p functions
//!
//! The sp part (core + valence) is identical to 6-31G and is reused from
//! basis631g.zig.
//!
//! Polarization exponents from:
//!   M. J. Frisch, J. A. Pople, J. S. Binkley,
//!   J. Chem. Phys. 80, 3265 (1984).
//!
//! Data validated against PySCF's built-in 6-31G(2df,p) basis.

const gaussian = @import("gaussian.zig");
const PrimitiveGaussian = gaussian.PrimitiveGaussian;
const ContractedShell = gaussian.ContractedShell;
const math = @import("../math/math.zig");
const basis631g = @import("basis631g.zig");

// ============================================================================
// Polarization function exponents
// ============================================================================

// --- Hydrogen: 1p polarization ---

/// Hydrogen p polarization (1 primitive).
pub const H_p_pol = [_]PrimitiveGaussian{
    .{ .alpha = 1.1000000, .coeff = 1.0000000 },
};

// --- Carbon: 2d + 1f polarization ---

/// Carbon d polarization, set 1 (1 primitive).
pub const C_d_pol1 = [_]PrimitiveGaussian{
    .{ .alpha = 1.2520000, .coeff = 1.0000000 },
};

/// Carbon d polarization, set 2 (1 primitive).
pub const C_d_pol2 = [_]PrimitiveGaussian{
    .{ .alpha = 0.3130000, .coeff = 1.0000000 },
};

/// Carbon f polarization (1 primitive).
pub const C_f_pol = [_]PrimitiveGaussian{
    .{ .alpha = 0.8000000, .coeff = 1.0000000 },
};

// --- Nitrogen: 2d + 1f polarization ---

/// Nitrogen d polarization, set 1 (1 primitive).
pub const N_d_pol1 = [_]PrimitiveGaussian{
    .{ .alpha = 1.8260000, .coeff = 1.0000000 },
};

/// Nitrogen d polarization, set 2 (1 primitive).
pub const N_d_pol2 = [_]PrimitiveGaussian{
    .{ .alpha = 0.4565000, .coeff = 1.0000000 },
};

/// Nitrogen f polarization (1 primitive).
pub const N_f_pol = [_]PrimitiveGaussian{
    .{ .alpha = 1.0000000, .coeff = 1.0000000 },
};

// --- Oxygen: 2d + 1f polarization ---

/// Oxygen d polarization, set 1 (1 primitive).
pub const O_d_pol1 = [_]PrimitiveGaussian{
    .{ .alpha = 2.5840000, .coeff = 1.0000000 },
};

/// Oxygen d polarization, set 2 (1 primitive).
pub const O_d_pol2 = [_]PrimitiveGaussian{
    .{ .alpha = 0.6460000, .coeff = 1.0000000 },
};

/// Oxygen f polarization (1 primitive).
pub const O_f_pol = [_]PrimitiveGaussian{
    .{ .alpha = 1.4000000, .coeff = 1.0000000 },
};

// --- Fluorine: 2d + 1f polarization ---

/// Fluorine d polarization, set 1 (1 primitive).
pub const F_d_pol1 = [_]PrimitiveGaussian{
    .{ .alpha = 3.5000000, .coeff = 1.0000000 },
};

/// Fluorine d polarization, set 2 (1 primitive).
pub const F_d_pol2 = [_]PrimitiveGaussian{
    .{ .alpha = 0.8750000, .coeff = 1.0000000 },
};

/// Fluorine f polarization (1 primitive).
pub const F_f_pol = [_]PrimitiveGaussian{
    .{ .alpha = 1.8500000, .coeff = 1.0000000 },
};

// ============================================================================
// Atom shell builders
// ============================================================================

/// Maximum number of shells per atom in 6-31G(2df,p) basis.
/// H: 3 shells (2 from 6-31G + 1 p polarization)
/// C-F: 8 shells (5 from 6-31G + 2 d polarization + 1 f polarization)
pub const MAX_SHELLS_PER_ATOM = 8;

/// Build contracted shells for an atom at a given center using 6-31G(2df,p) basis.
///
/// Returns the shells and the count of valid shells, or null if the element
/// is not supported.
pub fn buildAtomShells(
    z: u32,
    center: math.Vec3,
) ?struct { shells: [MAX_SHELLS_PER_ATOM]ContractedShell, count: usize } {
    const empty_prims = &[_]PrimitiveGaussian{};
    const dummy = ContractedShell{ .center = center, .l = 0, .primitives = empty_prims };
    var result: [MAX_SHELLS_PER_ATOM]ContractedShell = .{
        dummy, dummy, dummy, dummy,
        dummy, dummy, dummy, dummy,
    };

    switch (z) {
        1 => {
            // H: 6-31G sp (2 shells) + 1 p polarization
            result[0] = .{ .center = center, .l = 0, .primitives = &basis631g.H_1s_inner };
            result[1] = .{ .center = center, .l = 0, .primitives = &basis631g.H_1s_outer };
            result[2] = .{ .center = center, .l = 1, .primitives = &H_p_pol };
            return .{ .shells = result, .count = 3 };
        },
        6 => {
            // Carbon: 6-31G sp (5 shells) + 2d + 1f
            // PySCF ordering: 1s, 2s_inner, 2s_outer, 2p_inner, 2p_outer, d1, d2, f
            result[0] = .{ .center = center, .l = 0, .primitives = &basis631g.C_1s };
            result[1] = .{ .center = center, .l = 0, .primitives = &basis631g.C_2s_inner };
            result[2] = .{ .center = center, .l = 0, .primitives = &basis631g.C_2s_outer };
            result[3] = .{ .center = center, .l = 1, .primitives = &basis631g.C_2p_inner };
            result[4] = .{ .center = center, .l = 1, .primitives = &basis631g.C_2p_outer };
            result[5] = .{ .center = center, .l = 2, .primitives = &C_d_pol1 };
            result[6] = .{ .center = center, .l = 2, .primitives = &C_d_pol2 };
            result[7] = .{ .center = center, .l = 3, .primitives = &C_f_pol };
            return .{ .shells = result, .count = 8 };
        },
        7 => {
            // Nitrogen
            result[0] = .{ .center = center, .l = 0, .primitives = &basis631g.N_1s };
            result[1] = .{ .center = center, .l = 0, .primitives = &basis631g.N_2s_inner };
            result[2] = .{ .center = center, .l = 0, .primitives = &basis631g.N_2s_outer };
            result[3] = .{ .center = center, .l = 1, .primitives = &basis631g.N_2p_inner };
            result[4] = .{ .center = center, .l = 1, .primitives = &basis631g.N_2p_outer };
            result[5] = .{ .center = center, .l = 2, .primitives = &N_d_pol1 };
            result[6] = .{ .center = center, .l = 2, .primitives = &N_d_pol2 };
            result[7] = .{ .center = center, .l = 3, .primitives = &N_f_pol };
            return .{ .shells = result, .count = 8 };
        },
        8 => {
            // Oxygen
            result[0] = .{ .center = center, .l = 0, .primitives = &basis631g.O_1s };
            result[1] = .{ .center = center, .l = 0, .primitives = &basis631g.O_2s_inner };
            result[2] = .{ .center = center, .l = 0, .primitives = &basis631g.O_2s_outer };
            result[3] = .{ .center = center, .l = 1, .primitives = &basis631g.O_2p_inner };
            result[4] = .{ .center = center, .l = 1, .primitives = &basis631g.O_2p_outer };
            result[5] = .{ .center = center, .l = 2, .primitives = &O_d_pol1 };
            result[6] = .{ .center = center, .l = 2, .primitives = &O_d_pol2 };
            result[7] = .{ .center = center, .l = 3, .primitives = &O_f_pol };
            return .{ .shells = result, .count = 8 };
        },
        9 => {
            // Fluorine
            result[0] = .{ .center = center, .l = 0, .primitives = &basis631g.F_1s };
            result[1] = .{ .center = center, .l = 0, .primitives = &basis631g.F_2s_inner };
            result[2] = .{ .center = center, .l = 0, .primitives = &basis631g.F_2s_outer };
            result[3] = .{ .center = center, .l = 1, .primitives = &basis631g.F_2p_inner };
            result[4] = .{ .center = center, .l = 1, .primitives = &basis631g.F_2p_outer };
            result[5] = .{ .center = center, .l = 2, .primitives = &F_d_pol1 };
            result[6] = .{ .center = center, .l = 2, .primitives = &F_d_pol2 };
            result[7] = .{ .center = center, .l = 3, .primitives = &F_f_pol };
            return .{ .shells = result, .count = 8 };
        },
        else => return null,
    }
}

/// Return number of shells for a given atomic number in 6-31G(2df,p).
pub fn numShellsForAtom(z: u32) ?usize {
    return switch (z) {
        1 => 3,
        6, 7, 8, 9 => 8,
        else => null,
    };
}

/// Return number of basis functions for a given atomic number in 6-31G(2df,p).
/// H: 5  (2s + 3p = 5)
/// C-F: 26 (3s + 6p + 12d + 10f = 3 + 6 + 6 + 6 + 10 = ... wait)
/// Actually: 3s + 6p + 6d + 6d + 10f = 3 + 6 + 6 + 6 + 10 = ... no
/// Cartesian: s=1, p=3, d=6, f=10
/// H: 1+1+3 = 5
/// C-F: 1+1+1+3+3+6+6+10 = 31? No...
/// PySCF says 26 for C. Let's check: 3s(1+1+1) + 2p(3+3) + 2d(6+6) + 1f(10) = 3+6+12+10 = 31
/// But PySCF says 26! That means PySCF uses spherical harmonics for d and f:
/// spherical d = 5 (not 6), spherical f = 7 (not 10)
/// 3 + 6 + 5+5 + 7 = 26
/// Our code uses Cartesian throughout, so:
/// H: 1+1+3 = 5 (same, no d/f)
/// C-F: 1+1+1+3+3+6+6+10 = 31 (Cartesian)
///
/// NOTE: PySCF uses spherical harmonics (5d, 7f) giving 26.
/// Our Cartesian basis gives 31. This is correct for Cartesian GTOs.
pub fn numBasisForAtom(z: u32) ?usize {
    return switch (z) {
        1 => 5, // 1+1+3
        6, 7, 8, 9 => 31, // 1+1+1+3+3+6+6+10
        else => null,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "6-31G(2df,p) H shell count" {
    const testing = @import("std").testing;
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const data = buildAtomShells(1, center).?;
    try testing.expectEqual(@as(usize, 3), data.count);
    // s, s, p
    try testing.expectEqual(@as(u32, 0), data.shells[0].l);
    try testing.expectEqual(@as(u32, 0), data.shells[1].l);
    try testing.expectEqual(@as(u32, 1), data.shells[2].l);
}

test "6-31G(2df,p) O shell count and basis functions" {
    const testing = @import("std").testing;
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const data = buildAtomShells(8, center).?;
    try testing.expectEqual(@as(usize, 8), data.count);

    // Shell types: s, s, s, p, p, d, d, f
    try testing.expectEqual(@as(u32, 0), data.shells[0].l);
    try testing.expectEqual(@as(u32, 0), data.shells[1].l);
    try testing.expectEqual(@as(u32, 0), data.shells[2].l);
    try testing.expectEqual(@as(u32, 1), data.shells[3].l);
    try testing.expectEqual(@as(u32, 1), data.shells[4].l);
    try testing.expectEqual(@as(u32, 2), data.shells[5].l);
    try testing.expectEqual(@as(u32, 2), data.shells[6].l);
    try testing.expectEqual(@as(u32, 3), data.shells[7].l);

    // Count total basis functions (Cartesian):
    // s(1) + s(1) + s(1) + p(3) + p(3) + d(6) + d(6) + f(10) = 31
    const obara_saika = @import("../integrals/obara_saika.zig");
    const n = obara_saika.totalBasisFunctions(data.shells[0..data.count]);
    try testing.expectEqual(@as(usize, 31), n);
}

test "6-31G(2df,p) H2O basis count (Cartesian)" {
    const testing = @import("std").testing;
    const obara_saika = @import("../integrals/obara_saika.zig");

    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };

    const o_data = buildAtomShells(8, nuc_positions[0]).?;
    const h1_data = buildAtomShells(1, nuc_positions[1]).?;
    const h2_data = buildAtomShells(1, nuc_positions[2]).?;

    // Combine all shells
    var all_shells: [MAX_SHELLS_PER_ATOM * 3]ContractedShell = undefined;
    var count: usize = 0;
    for (o_data.shells[0..o_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h1_data.shells[0..h1_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h2_data.shells[0..h2_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }

    const n = obara_saika.totalBasisFunctions(all_shells[0..count]);
    // O: 31, H: 5, H: 5 = 41 (Cartesian)
    try testing.expectEqual(@as(usize, 41), n);
}
