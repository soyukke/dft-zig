//! STO-3G minimal basis set data.
//!
//! STO-3G approximates each Slater-type orbital (STO) as a contraction
//! of 3 primitive Gaussians. The exponents and coefficients are from
//! W. J. Hehre, R. F. Stewart, J. A. Pople, J. Chem. Phys. 51, 2657 (1969).
//!
//! Exponents and coefficients are for unnormalized primitives.
//! Normalization is applied during integral evaluation.
//!
//! For second-row atoms (Li-Ne), the basis uses:
//!   - Inner shell (1s): separate exponents and coefficients
//!   - Outer s (2s): shared exponents with 2p, separate coefficients
//!   - Outer p (2p): shared exponents with 2s, separate coefficients

const gaussian = @import("gaussian.zig");
const PrimitiveGaussian = gaussian.PrimitiveGaussian;
const ContractedShell = gaussian.ContractedShell;
const math = @import("../math/math.zig");

/// Hydrogen 1s: STO-3G (ζ = 1.24)
pub const H_1s = [_]PrimitiveGaussian{
    .{ .alpha = 3.42525091, .coeff = 0.15432897 },
    .{ .alpha = 0.62391373, .coeff = 0.53532814 },
    .{ .alpha = 0.16885540, .coeff = 0.44463454 },
};

/// Helium 1s: STO-3G (ζ = 1.6875)
pub const He_1s = [_]PrimitiveGaussian{
    .{ .alpha = 6.36242139, .coeff = 0.15432897 },
    .{ .alpha = 1.15892300, .coeff = 0.53532814 },
    .{ .alpha = 0.31364979, .coeff = 0.44463454 },
};

// ============================================================================
// Carbon (Z=6)
// ============================================================================

/// Carbon 1s (inner): STO-3G
pub const C_1s = [_]PrimitiveGaussian{
    .{ .alpha = 71.6168370, .coeff = 0.15432897 },
    .{ .alpha = 13.0450960, .coeff = 0.53532814 },
    .{ .alpha = 3.5305122, .coeff = 0.44463454 },
};

/// Carbon 2s (outer s): STO-3G — shared exponents with 2p
pub const C_2s = [_]PrimitiveGaussian{
    .{ .alpha = 2.9412494, .coeff = -0.09996723 },
    .{ .alpha = 0.6834831, .coeff = 0.39951283 },
    .{ .alpha = 0.2222899, .coeff = 0.70011547 },
};

/// Carbon 2p (outer p): STO-3G — shared exponents with 2s
pub const C_2p = [_]PrimitiveGaussian{
    .{ .alpha = 2.9412494, .coeff = 0.15591627 },
    .{ .alpha = 0.6834831, .coeff = 0.60768372 },
    .{ .alpha = 0.2222899, .coeff = 0.39195739 },
};

// ============================================================================
// Nitrogen (Z=7)
// ============================================================================

/// Nitrogen 1s (inner): STO-3G
pub const N_1s = [_]PrimitiveGaussian{
    .{ .alpha = 99.1061690, .coeff = 0.15432897 },
    .{ .alpha = 18.0523120, .coeff = 0.53532814 },
    .{ .alpha = 4.8856602, .coeff = 0.44463454 },
};

/// Nitrogen 2s (outer s): STO-3G
pub const N_2s = [_]PrimitiveGaussian{
    .{ .alpha = 3.7804559, .coeff = -0.09996723 },
    .{ .alpha = 0.8784966, .coeff = 0.39951283 },
    .{ .alpha = 0.2857144, .coeff = 0.70011547 },
};

/// Nitrogen 2p (outer p): STO-3G
pub const N_2p = [_]PrimitiveGaussian{
    .{ .alpha = 3.7804559, .coeff = 0.15591627 },
    .{ .alpha = 0.8784966, .coeff = 0.60768372 },
    .{ .alpha = 0.2857144, .coeff = 0.39195739 },
};

// ============================================================================
// Oxygen (Z=8)
// ============================================================================

/// Oxygen 1s (inner): STO-3G
pub const O_1s = [_]PrimitiveGaussian{
    .{ .alpha = 130.7093200, .coeff = 0.15432897 },
    .{ .alpha = 23.8088610, .coeff = 0.53532814 },
    .{ .alpha = 6.4436083, .coeff = 0.44463454 },
};

/// Oxygen 2s (outer s): STO-3G
pub const O_2s = [_]PrimitiveGaussian{
    .{ .alpha = 5.0331513, .coeff = -0.09996723 },
    .{ .alpha = 1.1695961, .coeff = 0.39951283 },
    .{ .alpha = 0.3803890, .coeff = 0.70011547 },
};

/// Oxygen 2p (outer p): STO-3G
pub const O_2p = [_]PrimitiveGaussian{
    .{ .alpha = 5.0331513, .coeff = 0.15591627 },
    .{ .alpha = 1.1695961, .coeff = 0.60768372 },
    .{ .alpha = 0.3803890, .coeff = 0.39195739 },
};

// ============================================================================
// Fluorine (Z=9)
// ============================================================================

/// Fluorine 1s (inner): STO-3G
pub const F_1s = [_]PrimitiveGaussian{
    .{ .alpha = 166.6791300, .coeff = 0.15432897 },
    .{ .alpha = 30.3608120, .coeff = 0.53532814 },
    .{ .alpha = 8.2168207, .coeff = 0.44463454 },
};

/// Fluorine 2s (outer s): STO-3G
pub const F_2s = [_]PrimitiveGaussian{
    .{ .alpha = 6.4648032, .coeff = -0.09996723 },
    .{ .alpha = 1.5022812, .coeff = 0.39951283 },
    .{ .alpha = 0.4885885, .coeff = 0.70011547 },
};

/// Fluorine 2p (outer p): STO-3G
pub const F_2p = [_]PrimitiveGaussian{
    .{ .alpha = 6.4648032, .coeff = 0.15591627 },
    .{ .alpha = 1.5022812, .coeff = 0.60768372 },
    .{ .alpha = 0.4885885, .coeff = 0.39195739 },
};

/// Shell data for a second-row atom: inner 1s + outer 2s + outer 2p.
pub const SecondRowBasis = struct {
    inner_1s: []const PrimitiveGaussian,
    outer_2s: []const PrimitiveGaussian,
    outer_2p: []const PrimitiveGaussian,
};

/// Look up the 1s basis for a given atomic number (H, He only).
/// Returns null if the element is not in the STO-3G database.
pub fn getBasis1s(z: u32) ?[]const PrimitiveGaussian {
    return switch (z) {
        1 => &H_1s,
        2 => &He_1s,
        else => null,
    };
}

/// Look up the full basis (all shells) for a given atomic number.
/// Returns null if the element is not in the STO-3G database.
pub fn getSecondRowBasis(z: u32) ?SecondRowBasis {
    return switch (z) {
        6 => SecondRowBasis{ .inner_1s = &C_1s, .outer_2s = &C_2s, .outer_2p = &C_2p },
        7 => SecondRowBasis{ .inner_1s = &N_1s, .outer_2s = &N_2s, .outer_2p = &N_2p },
        8 => SecondRowBasis{ .inner_1s = &O_1s, .outer_2s = &O_2s, .outer_2p = &O_2p },
        9 => SecondRowBasis{ .inner_1s = &F_1s, .outer_2s = &F_2s, .outer_2p = &F_2p },
        else => null,
    };
}

/// Build contracted shells for an atom at a given center.
/// For H/He: returns 1 shell (1s).
/// For C/N/O/F: returns 3 shells (1s, 2s, 2p).
/// Returns the shells via a static buffer (max 3 shells per atom).
pub const MAX_SHELLS_PER_ATOM = 3;

pub fn buildAtomShells(z: u32, center: math.Vec3) ?[MAX_SHELLS_PER_ATOM]ContractedShell {
    // First-row atoms: just 1s
    if (getBasis1s(z)) |prims_1s| {
        var result: [MAX_SHELLS_PER_ATOM]ContractedShell = undefined;
        result[0] = .{ .center = center, .l = 0, .primitives = prims_1s };
        // Fill unused with dummy
        result[1] = .{ .center = center, .l = 0, .primitives = &[_]PrimitiveGaussian{} };
        result[2] = .{ .center = center, .l = 0, .primitives = &[_]PrimitiveGaussian{} };
        return result;
    }

    // Second-row atoms
    if (getSecondRowBasis(z)) |basis| {
        var result: [MAX_SHELLS_PER_ATOM]ContractedShell = undefined;
        result[0] = .{ .center = center, .l = 0, .primitives = basis.inner_1s };
        result[1] = .{ .center = center, .l = 0, .primitives = basis.outer_2s };
        result[2] = .{ .center = center, .l = 1, .primitives = basis.outer_2p };
        return result;
    }

    return null;
}

/// Return number of shells for a given atomic number.
pub fn numShellsForAtom(z: u32) ?usize {
    return switch (z) {
        1, 2 => 1,
        6, 7, 8, 9 => 3,
        else => null,
    };
}
