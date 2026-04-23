//! 6-31G split-valence basis set data.
//!
//! The 6-31G basis uses a double-zeta description for the valence shell:
//!   - Core: 6 primitives contracted to 1 function
//!   - Valence inner: 3 primitives contracted to 1 function
//!   - Valence outer: 1 primitive (uncontracted)
//!
//! For second-row atoms, the valence sp shells share the same exponents
//! between the s and p components but have different contraction coefficients.
//!
//! Data from W. J. Hehre, R. Ditchfield, J. A. Pople,
//! J. Chem. Phys. 56, 2257 (1972).
//!
//! Exponents and coefficients are for unnormalized primitives.
//! Normalization is applied during integral evaluation.

const gaussian = @import("gaussian.zig");
const PrimitiveGaussian = gaussian.PrimitiveGaussian;
const ContractedShell = gaussian.ContractedShell;
const math = @import("../math/math.zig");

// ============================================================================
// Hydrogen (Z=1)
// ============================================================================

/// Hydrogen 1s inner: 6-31G (3 primitives)
pub const H_1s_inner = [_]PrimitiveGaussian{
    .{ .alpha = 18.7311370, .coeff = 0.0334946 },
    .{ .alpha = 2.8253937, .coeff = 0.2347270 },
    .{ .alpha = 0.6401217, .coeff = 0.8137573 },
};

/// Hydrogen 1s outer: 6-31G (1 primitive)
pub const H_1s_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.1612778, .coeff = 1.0000000 },
};

// ============================================================================
// Carbon (Z=6)
// ============================================================================

/// Carbon 1s core: 6-31G (6 primitives)
pub const C_1s = [_]PrimitiveGaussian{
    .{ .alpha = 3047.5249000, .coeff = 0.0018347 },
    .{ .alpha = 457.3695100, .coeff = 0.0140373 },
    .{ .alpha = 103.9486900, .coeff = 0.0688426 },
    .{ .alpha = 29.2101550, .coeff = 0.2321844 },
    .{ .alpha = 9.2866630, .coeff = 0.4679413 },
    .{ .alpha = 3.1639270, .coeff = 0.3623120 },
};

/// Carbon 2s inner (valence): 6-31G (3 primitives, shared exponents with 2p)
pub const C_2s_inner = [_]PrimitiveGaussian{
    .{ .alpha = 7.8682724, .coeff = -0.1193324 },
    .{ .alpha = 1.8812885, .coeff = -0.1608542 },
    .{ .alpha = 0.5442493, .coeff = 1.1434564 },
};

/// Carbon 2p inner (valence): 6-31G (3 primitives, shared exponents with 2s)
pub const C_2p_inner = [_]PrimitiveGaussian{
    .{ .alpha = 7.8682724, .coeff = 0.0689991 },
    .{ .alpha = 1.8812885, .coeff = 0.3164240 },
    .{ .alpha = 0.5442493, .coeff = 0.7443083 },
};

/// Carbon 2s outer (valence): 6-31G (1 primitive)
pub const C_2s_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.1687144, .coeff = 1.0000000 },
};

/// Carbon 2p outer (valence): 6-31G (1 primitive)
pub const C_2p_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.1687144, .coeff = 1.0000000 },
};

// ============================================================================
// Nitrogen (Z=7)
// ============================================================================

/// Nitrogen 1s core: 6-31G (6 primitives)
pub const N_1s = [_]PrimitiveGaussian{
    .{ .alpha = 4173.5110000, .coeff = 0.0018348 },
    .{ .alpha = 627.4579000, .coeff = 0.0139950 },
    .{ .alpha = 142.9021000, .coeff = 0.0685870 },
    .{ .alpha = 40.2343300, .coeff = 0.2322410 },
    .{ .alpha = 12.8202100, .coeff = 0.4690700 },
    .{ .alpha = 4.3904370, .coeff = 0.3604550 },
};

/// Nitrogen 2s inner: 6-31G (3 primitives, shared exponents with 2p)
pub const N_2s_inner = [_]PrimitiveGaussian{
    .{ .alpha = 11.6263580, .coeff = -0.1149610 },
    .{ .alpha = 2.7162800, .coeff = -0.1691180 },
    .{ .alpha = 0.7722180, .coeff = 1.1458520 },
};

/// Nitrogen 2p inner: 6-31G (3 primitives, shared exponents with 2s)
pub const N_2p_inner = [_]PrimitiveGaussian{
    .{ .alpha = 11.6263580, .coeff = 0.0675800 },
    .{ .alpha = 2.7162800, .coeff = 0.3239070 },
    .{ .alpha = 0.7722180, .coeff = 0.7408950 },
};

/// Nitrogen 2s outer: 6-31G (1 primitive)
pub const N_2s_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.2120313, .coeff = 1.0000000 },
};

/// Nitrogen 2p outer: 6-31G (1 primitive)
pub const N_2p_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.2120313, .coeff = 1.0000000 },
};

// ============================================================================
// Oxygen (Z=8)
// ============================================================================

/// Oxygen 1s core: 6-31G (6 primitives)
pub const O_1s = [_]PrimitiveGaussian{
    .{ .alpha = 5484.6717000, .coeff = 0.0018311 },
    .{ .alpha = 825.2349500, .coeff = 0.0139501 },
    .{ .alpha = 188.0469600, .coeff = 0.0684451 },
    .{ .alpha = 52.9645000, .coeff = 0.2327143 },
    .{ .alpha = 16.8975700, .coeff = 0.4701930 },
    .{ .alpha = 5.7996353, .coeff = 0.3585209 },
};

/// Oxygen 2s inner: 6-31G (3 primitives, shared exponents with 2p)
pub const O_2s_inner = [_]PrimitiveGaussian{
    .{ .alpha = 15.5396160, .coeff = -0.1107775 },
    .{ .alpha = 3.5999336, .coeff = -0.1480263 },
    .{ .alpha = 1.0137618, .coeff = 1.1307670 },
};

/// Oxygen 2p inner: 6-31G (3 primitives, shared exponents with 2s)
pub const O_2p_inner = [_]PrimitiveGaussian{
    .{ .alpha = 15.5396160, .coeff = 0.0708743 },
    .{ .alpha = 3.5999336, .coeff = 0.3397528 },
    .{ .alpha = 1.0137618, .coeff = 0.7271586 },
};

/// Oxygen 2s outer: 6-31G (1 primitive)
pub const O_2s_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.2700058, .coeff = 1.0000000 },
};

/// Oxygen 2p outer: 6-31G (1 primitive)
pub const O_2p_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.2700058, .coeff = 1.0000000 },
};

// ============================================================================
// Fluorine (Z=9)
// ============================================================================

/// Fluorine 1s core: 6-31G (6 primitives)
pub const F_1s = [_]PrimitiveGaussian{
    .{ .alpha = 7001.7130900, .coeff = 0.0018196 },
    .{ .alpha = 1051.3660900, .coeff = 0.0139161 },
    .{ .alpha = 239.2856900, .coeff = 0.0684053 },
    .{ .alpha = 67.3974453, .coeff = 0.2331858 },
    .{ .alpha = 21.5199573, .coeff = 0.4712674 },
    .{ .alpha = 7.4031013, .coeff = 0.3566185 },
};

/// Fluorine 2s inner: 6-31G (3 primitives, shared exponents with 2p)
pub const F_2s_inner = [_]PrimitiveGaussian{
    .{ .alpha = 20.8479528, .coeff = -0.1085070 },
    .{ .alpha = 4.8083083, .coeff = -0.1464517 },
    .{ .alpha = 1.3440699, .coeff = 1.1286886 },
};

/// Fluorine 2p inner: 6-31G (3 primitives, shared exponents with 2s)
pub const F_2p_inner = [_]PrimitiveGaussian{
    .{ .alpha = 20.8479528, .coeff = 0.0716287 },
    .{ .alpha = 4.8083083, .coeff = 0.3459121 },
    .{ .alpha = 1.3440699, .coeff = 0.7224700 },
};

/// Fluorine 2s outer: 6-31G (1 primitive)
pub const F_2s_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.3581514, .coeff = 1.0000000 },
};

/// Fluorine 2p outer: 6-31G (1 primitive)
pub const F_2p_outer = [_]PrimitiveGaussian{
    .{ .alpha = 0.3581514, .coeff = 1.0000000 },
};

// ============================================================================
// Atom shell builders
// ============================================================================

/// Maximum number of shells per atom in 6-31G basis.
/// H: 2 shells (1s_inner, 1s_outer)
/// C-F: 5 shells (1s, 2s_inner, 2p_inner, 2s_outer, 2p_outer)
pub const MAX_SHELLS_PER_ATOM = 5;

/// Build contracted shells for an atom at a given center using 6-31G basis.
///
/// Returns the shells and the count of valid shells, or null if the element
/// is not supported.
pub fn build_atom_shells(
    z: u32,
    center: math.Vec3,
) ?struct { shells: [MAX_SHELLS_PER_ATOM]ContractedShell, count: usize } {
    const empty_prims = &[_]PrimitiveGaussian{};
    const dummy = ContractedShell{ .center = center, .l = 0, .primitives = empty_prims };
    var result: [MAX_SHELLS_PER_ATOM]ContractedShell = .{ dummy, dummy, dummy, dummy, dummy };

    switch (z) {
        1 => {
            // H: 2 s-shells
            result[0] = .{ .center = center, .l = 0, .primitives = &H_1s_inner };
            result[1] = .{ .center = center, .l = 0, .primitives = &H_1s_outer };
            return .{ .shells = result, .count = 2 };
        },
        6 => {
            // Carbon: 1s + 2s_inner + 2p_inner + 2s_outer + 2p_outer
            result[0] = .{ .center = center, .l = 0, .primitives = &C_1s };
            result[1] = .{ .center = center, .l = 0, .primitives = &C_2s_inner };
            result[2] = .{ .center = center, .l = 1, .primitives = &C_2p_inner };
            result[3] = .{ .center = center, .l = 0, .primitives = &C_2s_outer };
            result[4] = .{ .center = center, .l = 1, .primitives = &C_2p_outer };
            return .{ .shells = result, .count = 5 };
        },
        7 => {
            // Nitrogen
            result[0] = .{ .center = center, .l = 0, .primitives = &N_1s };
            result[1] = .{ .center = center, .l = 0, .primitives = &N_2s_inner };
            result[2] = .{ .center = center, .l = 1, .primitives = &N_2p_inner };
            result[3] = .{ .center = center, .l = 0, .primitives = &N_2s_outer };
            result[4] = .{ .center = center, .l = 1, .primitives = &N_2p_outer };
            return .{ .shells = result, .count = 5 };
        },
        8 => {
            // Oxygen
            result[0] = .{ .center = center, .l = 0, .primitives = &O_1s };
            result[1] = .{ .center = center, .l = 0, .primitives = &O_2s_inner };
            result[2] = .{ .center = center, .l = 1, .primitives = &O_2p_inner };
            result[3] = .{ .center = center, .l = 0, .primitives = &O_2s_outer };
            result[4] = .{ .center = center, .l = 1, .primitives = &O_2p_outer };
            return .{ .shells = result, .count = 5 };
        },
        9 => {
            // Fluorine
            result[0] = .{ .center = center, .l = 0, .primitives = &F_1s };
            result[1] = .{ .center = center, .l = 0, .primitives = &F_2s_inner };
            result[2] = .{ .center = center, .l = 1, .primitives = &F_2p_inner };
            result[3] = .{ .center = center, .l = 0, .primitives = &F_2s_outer };
            result[4] = .{ .center = center, .l = 1, .primitives = &F_2p_outer };
            return .{ .shells = result, .count = 5 };
        },
        else => return null,
    }
}

/// Return number of shells for a given atomic number in 6-31G.
pub fn num_shells_for_atom(z: u32) ?usize {
    return switch (z) {
        1 => 2,
        6, 7, 8, 9 => 5,
        else => null,
    };
}

/// Return number of basis functions for a given atomic number in 6-31G.
/// H: 2 (two s functions)
/// C-F: 9 (3s + 6p = 1+1+1 + 3+3 = 9)
pub fn num_basis_for_atom(z: u32) ?usize {
    return switch (z) {
        1 => 2,
        6, 7, 8, 9 => 9,
        else => null,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "6-31G H shell count" {
    const testing = @import("std").testing;
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const data = build_atom_shells(1, center).?;
    try testing.expectEqual(@as(usize, 2), data.count);
    try testing.expectEqual(@as(u32, 0), data.shells[0].l);
    try testing.expectEqual(@as(u32, 0), data.shells[1].l);
    try testing.expectEqual(@as(usize, 3), data.shells[0].primitives.len);
    try testing.expectEqual(@as(usize, 1), data.shells[1].primitives.len);
}

test "6-31G O shell count and basis functions" {
    const testing = @import("std").testing;
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const data = build_atom_shells(8, center).?;
    try testing.expectEqual(@as(usize, 5), data.count);

    // Count total basis functions: 1s(1) + 2s(1) + 2p(3) + 2s'(1) + 2p'(3) = 9
    const obara_saika = @import("../integrals/obara_saika.zig");
    const n = obara_saika.total_basis_functions(data.shells[0..data.count]);
    try testing.expectEqual(@as(usize, 9), n);
}

test "6-31G H2O basis count is 13" {
    const testing = @import("std").testing;
    const obara_saika = @import("../integrals/obara_saika.zig");

    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };

    const o_data = build_atom_shells(8, nuc_positions[0]).?;
    const h1_data = build_atom_shells(1, nuc_positions[1]).?;
    const h2_data = build_atom_shells(1, nuc_positions[2]).?;

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

    const n = obara_saika.total_basis_functions(all_shells[0..count]);
    // O: 9, H: 2, H: 2 = 13
    try testing.expectEqual(@as(usize, 13), n);
}
