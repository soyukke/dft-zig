//! Molecular input module for GTO calculations.
//!
//! Provides:
//!   - XYZ format parsing (string or file)
//!   - Element symbol ↔ atomic number conversions
//!   - Basis set selection and shell construction
//!   - Molecular data (positions, charges, shells) assembly
//!
//! The XYZ format uses Angstroms; all internal coordinates are in Bohr.
//! Units: Hartree atomic units.

const std = @import("std");
const math = @import("../math/math.zig");
const Vec3 = math.Vec3;
const basis_mod = @import("../basis/basis.zig");
const ContractedShell = basis_mod.ContractedShell;
const sto3g = basis_mod.sto3g;
const basis631g = basis_mod.basis631g;
const basis631g_2dfp = basis_mod.basis631g_2dfp;

// ============================================================================
// Constants
// ============================================================================

/// Angstrom to Bohr conversion factor.
const angstrom_to_bohr: f64 = 1.8897259886;

// ============================================================================
// Element data
// ============================================================================

/// Supported elements with their properties.
const ElementData = struct {
    symbol: []const u8,
    z: u32,
    mass: f64, // atomic mass in AMU (most abundant isotope)
};

/// Element database (Z=1..9 for now, matching basis set coverage).
const elements = [_]ElementData{
    .{ .symbol = "H", .z = 1, .mass = 1.00794 },
    .{ .symbol = "He", .z = 2, .mass = 4.00260 },
    .{ .symbol = "Li", .z = 3, .mass = 6.941 },
    .{ .symbol = "Be", .z = 4, .mass = 9.01218 },
    .{ .symbol = "B", .z = 5, .mass = 10.811 },
    .{ .symbol = "C", .z = 6, .mass = 12.011 },
    .{ .symbol = "N", .z = 7, .mass = 14.007 },
    .{ .symbol = "O", .z = 8, .mass = 15.999 },
    .{ .symbol = "F", .z = 9, .mass = 18.998 },
    .{ .symbol = "Ne", .z = 10, .mass = 20.180 },
};

/// Convert element symbol to atomic number.
/// Case-insensitive: "H", "h", "He", "he", "HE" all work.
pub fn symbolToZ(symbol: []const u8) ?u32 {
    if (symbol.len == 0 or symbol.len > 2) return null;
    for (elements) |e| {
        if (eqlIgnoreCase(e.symbol, symbol)) return e.z;
    }
    return null;
}

/// Convert atomic number to element symbol.
pub fn zToSymbol(z: u32) ?[]const u8 {
    if (z == 0 or z > elements.len) return null;
    return elements[z - 1].symbol;
}

/// Get atomic mass in AMU for a given atomic number.
pub fn atomicMass(z: u32) ?f64 {
    if (z == 0 or z > elements.len) return null;
    return elements[z - 1].mass;
}

/// Case-insensitive string comparison.
fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        if (toLower(ca) != toLower(cb)) return false;
    }
    return true;
}

fn toLower(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}

// ============================================================================
// Basis set enumeration
// ============================================================================

/// Available basis sets.
pub const BasisSet = enum {
    sto_3g,
    @"6-31g",
    @"6-31g_2dfp",
};

// ============================================================================
// Molecule struct
// ============================================================================

/// Parsed molecular data ready for SCF calculations.
pub const Molecule = struct {
    /// Atomic positions in Bohr.
    positions: []Vec3,
    /// Nuclear charges (atomic numbers as f64).
    charges: []f64,
    /// Atomic numbers.
    atomic_numbers: []u32,
    /// Total number of electrons (sum of atomic numbers - charge).
    n_electrons: usize,
    /// Number of atoms.
    n_atoms: usize,
    /// Contracted Gaussian shells for all atoms.
    shells: []ContractedShell,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *Molecule) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.charges);
        self.allocator.free(self.atomic_numbers);
        self.allocator.free(self.shells);
    }
};

// ============================================================================
// XYZ parsing
// ============================================================================

/// Parse error types.
pub const ParseError = error{
    InvalidFormat,
    UnknownElement,
    UnsupportedElement,
    InvalidAtomCount,
    InvalidCoordinate,
    OutOfMemory,
};

/// Parse XYZ-format string and build a Molecule with the specified basis set.
///
/// XYZ format:
/// ```
/// N                     (number of atoms)
/// comment line          (ignored)
/// Symbol  x  y  z       (Angstrom)
/// Symbol  x  y  z
/// ...
/// ```
///
/// The `charge` parameter specifies the molecular charge (0 for neutral).
pub fn parseXyzString(
    alloc: std.mem.Allocator,
    input: []const u8,
    basis_set: BasisSet,
    charge: i32,
) !Molecule {
    var lines_iter = std.mem.splitScalar(u8, input, '\n');

    // Line 1: atom count
    const count_line = skipEmpty(&lines_iter) orelse return ParseError.InvalidFormat;
    const trimmed_count = std.mem.trim(u8, count_line, &std.ascii.whitespace);
    const n_atoms = std.fmt.parseInt(usize, trimmed_count, 10) catch
        return ParseError.InvalidAtomCount;

    if (n_atoms == 0) return ParseError.InvalidAtomCount;

    // Line 2: comment (skip)
    _ = lines_iter.next() orelse return ParseError.InvalidFormat;

    // Allocate arrays
    const positions = try alloc.alloc(Vec3, n_atoms);
    errdefer alloc.free(positions);
    const charges = try alloc.alloc(f64, n_atoms);
    errdefer alloc.free(charges);
    const atomic_numbers = try alloc.alloc(u32, n_atoms);
    errdefer alloc.free(atomic_numbers);

    var total_z: u32 = 0;

    // Parse atom lines
    for (0..n_atoms) |i| {
        const line = lines_iter.next() orelse return ParseError.InvalidFormat;
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (trimmed.len == 0) return ParseError.InvalidFormat;

        var tokens = std.mem.tokenizeAny(u8, trimmed, &std.ascii.whitespace);

        const symbol = tokens.next() orelse return ParseError.InvalidFormat;
        const z = symbolToZ(symbol) orelse return ParseError.UnknownElement;

        const x_str = tokens.next() orelse return ParseError.InvalidCoordinate;
        const y_str = tokens.next() orelse return ParseError.InvalidCoordinate;
        const z_str = tokens.next() orelse return ParseError.InvalidCoordinate;

        const x = std.fmt.parseFloat(f64, x_str) catch return ParseError.InvalidCoordinate;
        const y = std.fmt.parseFloat(f64, y_str) catch return ParseError.InvalidCoordinate;
        const z_coord = std.fmt.parseFloat(f64, z_str) catch return ParseError.InvalidCoordinate;

        // Convert Angstrom to Bohr
        positions[i] = .{
            .x = x * angstrom_to_bohr,
            .y = y * angstrom_to_bohr,
            .z = z_coord * angstrom_to_bohr,
        };
        charges[i] = @floatFromInt(z);
        atomic_numbers[i] = z;
        total_z += z;
    }

    // Compute electron count
    const total_z_signed: i32 = @intCast(total_z);
    const n_electrons_signed = total_z_signed - charge;
    if (n_electrons_signed <= 0) return ParseError.InvalidFormat;
    const n_electrons: usize = @intCast(n_electrons_signed);

    // Build shells
    const shells = try buildShells(alloc, atomic_numbers, positions, basis_set);
    errdefer alloc.free(shells);

    return .{
        .positions = positions,
        .charges = charges,
        .atomic_numbers = atomic_numbers,
        .n_electrons = n_electrons,
        .n_atoms = n_atoms,
        .shells = shells,
        .allocator = alloc,
    };
}

/// Skip empty lines and return the first non-empty line.
fn skipEmpty(iter: *std.mem.SplitIterator(u8, .scalar)) ?[]const u8 {
    while (iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (trimmed.len > 0) return trimmed;
    }
    return null;
}

/// Load an XYZ file from disk and build a Molecule with the specified basis set.
///
/// The file should be in standard XYZ format (Angstrom coordinates).
/// The `charge` parameter specifies the molecular charge (0 for neutral).
pub fn loadXyzFile(
    alloc: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    basis_set: BasisSet,
    charge: i32,
) !Molecule {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1024 * 1024));
    defer alloc.free(content);
    return parseXyzString(alloc, content, basis_set, charge);
}

// ============================================================================
// Shell construction
// ============================================================================

/// Maximum shells per atom across all basis sets.
const MAX_SHELLS_PER_ATOM = 8; // 6-31G(2df,p) has up to 8

/// Build contracted shells for all atoms with the specified basis set.
pub fn buildShells(
    alloc: std.mem.Allocator,
    atomic_numbers: []const u32,
    positions: []const Vec3,
    basis_set: BasisSet,
) ![]ContractedShell {
    std.debug.assert(atomic_numbers.len == positions.len);

    // First pass: count total shells
    var total_shells: usize = 0;
    for (atomic_numbers) |z| {
        const count = shellCountForAtom(z, basis_set) orelse
            return ParseError.UnsupportedElement;
        total_shells += count;
    }

    const shells = try alloc.alloc(ContractedShell, total_shells);
    errdefer alloc.free(shells);

    // Second pass: populate shells
    var idx: usize = 0;
    for (atomic_numbers, positions) |z, pos| {
        const count = appendAtomShells(z, pos, basis_set, shells[idx..]) orelse
            return ParseError.UnsupportedElement;
        idx += count;
    }

    std.debug.assert(idx == total_shells);
    return shells;
}

/// Get number of shells for an atom in a given basis set.
fn shellCountForAtom(z: u32, basis_set: BasisSet) ?usize {
    return switch (basis_set) {
        .sto_3g => sto3g.numShellsForAtom(z),
        .@"6-31g" => blk: {
            const data = basis631g.buildAtomShells(z, .{ .x = 0, .y = 0, .z = 0 }) orelse
                break :blk null;
            break :blk data.count;
        },
        .@"6-31g_2dfp" => blk: {
            const data = basis631g_2dfp.buildAtomShells(z, .{ .x = 0, .y = 0, .z = 0 }) orelse
                break :blk null;
            break :blk data.count;
        },
    };
}

/// Append shells for one atom into the buffer. Returns number of shells written.
fn appendAtomShells(z: u32, center: Vec3, basis_set: BasisSet, buf: []ContractedShell) ?usize {
    switch (basis_set) {
        .sto_3g => {
            const atom_shells = sto3g.buildAtomShells(z, center) orelse return null;
            const count = sto3g.numShellsForAtom(z) orelse return null;
            for (0..count) |i| {
                buf[i] = atom_shells[i];
            }
            return count;
        },
        .@"6-31g" => {
            const data = basis631g.buildAtomShells(z, center) orelse return null;
            for (0..data.count) |i| {
                buf[i] = data.shells[i];
            }
            return data.count;
        },
        .@"6-31g_2dfp" => {
            const data = basis631g_2dfp.buildAtomShells(z, center) orelse return null;
            for (0..data.count) |i| {
                buf[i] = data.shells[i];
            }
            return data.count;
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

test "symbolToZ basic elements" {
    const testing = std.testing;
    try testing.expectEqual(@as(?u32, 1), symbolToZ("H"));
    try testing.expectEqual(@as(?u32, 6), symbolToZ("C"));
    try testing.expectEqual(@as(?u32, 8), symbolToZ("O"));
    try testing.expectEqual(@as(?u32, 7), symbolToZ("N"));
    try testing.expectEqual(@as(?u32, 9), symbolToZ("F"));
    try testing.expectEqual(@as(?u32, 2), symbolToZ("He"));
}

test "symbolToZ case insensitive" {
    const testing = std.testing;
    try testing.expectEqual(@as(?u32, 1), symbolToZ("h"));
    try testing.expectEqual(@as(?u32, 2), symbolToZ("he"));
    try testing.expectEqual(@as(?u32, 2), symbolToZ("HE"));
    try testing.expectEqual(@as(?u32, 6), symbolToZ("c"));
}

test "symbolToZ unknown element" {
    const testing = std.testing;
    try testing.expectEqual(@as(?u32, null), symbolToZ("Xx"));
    try testing.expectEqual(@as(?u32, null), symbolToZ(""));
    try testing.expectEqual(@as(?u32, null), symbolToZ("Abc"));
}

test "zToSymbol" {
    const testing = std.testing;
    try testing.expectEqualStrings("H", zToSymbol(1).?);
    try testing.expectEqualStrings("C", zToSymbol(6).?);
    try testing.expectEqualStrings("O", zToSymbol(8).?);
    try testing.expectEqual(@as(?[]const u8, null), zToSymbol(0));
    try testing.expectEqual(@as(?[]const u8, null), zToSymbol(100));
}

test "atomicMass" {
    const testing = std.testing;
    try testing.expectApproxEqAbs(1.00794, atomicMass(1).?, 1e-4);
    try testing.expectApproxEqAbs(15.999, atomicMass(8).?, 1e-3);
    try testing.expectEqual(@as(?f64, null), atomicMass(0));
}

test "parseXyzString H2O STO-3G" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const xyz =
        \\3
        \\water molecule
        \\O  0.000000  0.000000  0.117370
        \\H  0.000000  0.757160 -0.469483
        \\H  0.000000 -0.757160 -0.469483
    ;

    var mol = try parseXyzString(alloc, xyz, .sto_3g, 0);
    defer mol.deinit();

    try testing.expectEqual(@as(usize, 3), mol.n_atoms);
    try testing.expectEqual(@as(usize, 10), mol.n_electrons);
    try testing.expectEqual(@as(u32, 8), mol.atomic_numbers[0]);
    try testing.expectEqual(@as(u32, 1), mol.atomic_numbers[1]);
    try testing.expectEqual(@as(u32, 1), mol.atomic_numbers[2]);

    // STO-3G: O has 3 shells, each H has 1 shell = 5 total
    try testing.expectEqual(@as(usize, 5), mol.shells.len);

    // Check coordinates are in Bohr (original x=0, y=0, z=0.117370 Ang)
    try testing.expectApproxEqAbs(0.0, mol.positions[0].x, 1e-10);
    try testing.expectApproxEqAbs(0.0, mol.positions[0].y, 1e-10);
    try testing.expectApproxEqAbs(0.117370 * angstrom_to_bohr, mol.positions[0].z, 1e-6);
}

test "parseXyzString H2 6-31G" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const xyz =
        \\2
        \\H2 molecule
        \\H  0.0  0.0  0.0
        \\H  0.0  0.0  0.74
    ;

    var mol = try parseXyzString(alloc, xyz, .@"6-31g", 0);
    defer mol.deinit();

    try testing.expectEqual(@as(usize, 2), mol.n_atoms);
    try testing.expectEqual(@as(usize, 2), mol.n_electrons);
    // 6-31G: each H has 2 shells (inner + outer) = 4 total
    try testing.expectEqual(@as(usize, 4), mol.shells.len);
}

test "parseXyzString charged molecule" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const xyz =
        \\1
        \\H+ cation
        \\H  0.0  0.0  0.0
    ;

    // H+ has 0 electrons with charge +1
    const result = parseXyzString(alloc, xyz, .sto_3g, 1);
    // Should fail because 0 electrons is invalid
    try testing.expectError(ParseError.InvalidFormat, result);
}

test "buildShells STO-3G H2O" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.43, .z = 1.11 },
        .{ .x = 0.0, .y = -1.43, .z = 1.11 },
    };
    const atomic_numbers = [_]u32{ 8, 1, 1 };

    const shells = try buildShells(alloc, &atomic_numbers, &positions, .sto_3g);
    defer alloc.free(shells);

    // O: 3 shells (1s, 2s, 2p), H: 1 shell each = 5
    try testing.expectEqual(@as(usize, 5), shells.len);
    // First shell: O 1s (l=0)
    try testing.expectEqual(@as(u32, 0), shells[0].l);
    // Third shell: O 2p (l=1)
    try testing.expectEqual(@as(u32, 1), shells[2].l);
}
