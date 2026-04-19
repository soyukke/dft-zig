//! Atomic data tables for DFPT calculations.
//!
//! Provides atomic masses (AMU) looked up by element symbol.

const builtin = @import("builtin");
const std = @import("std");
const runtime_logging = @import("../runtime/logging.zig");

/// Look up atomic mass in AMU from element symbol.
pub fn atomicMass(symbol: []const u8) f64 {
    const table = [_]struct { sym: []const u8, mass: f64 }{
        .{ .sym = "H", .mass = 1.008 },
        .{ .sym = "He", .mass = 4.003 },
        .{ .sym = "Li", .mass = 6.941 },
        .{ .sym = "Be", .mass = 9.012 },
        .{ .sym = "B", .mass = 10.81 },
        .{ .sym = "C", .mass = 12.011 },
        .{ .sym = "N", .mass = 14.007 },
        .{ .sym = "O", .mass = 15.999 },
        .{ .sym = "F", .mass = 18.998 },
        .{ .sym = "Ne", .mass = 20.180 },
        .{ .sym = "Na", .mass = 22.990 },
        .{ .sym = "Mg", .mass = 24.305 },
        .{ .sym = "Al", .mass = 26.982 },
        .{ .sym = "Si", .mass = 28.085 },
        .{ .sym = "P", .mass = 30.974 },
        .{ .sym = "S", .mass = 32.06 },
        .{ .sym = "Cl", .mass = 35.45 },
        .{ .sym = "Ar", .mass = 39.948 },
        .{ .sym = "K", .mass = 39.098 },
        .{ .sym = "Ca", .mass = 40.078 },
        .{ .sym = "Sc", .mass = 44.956 },
        .{ .sym = "Ti", .mass = 47.867 },
        .{ .sym = "V", .mass = 50.942 },
        .{ .sym = "Cr", .mass = 51.996 },
        .{ .sym = "Mn", .mass = 54.938 },
        .{ .sym = "Fe", .mass = 55.845 },
        .{ .sym = "Co", .mass = 58.933 },
        .{ .sym = "Ni", .mass = 58.693 },
        .{ .sym = "Cu", .mass = 63.546 },
        .{ .sym = "Zn", .mass = 65.38 },
        .{ .sym = "Ga", .mass = 69.723 },
        .{ .sym = "Ge", .mass = 72.63 },
        .{ .sym = "As", .mass = 74.922 },
        .{ .sym = "Se", .mass = 78.971 },
        .{ .sym = "Br", .mass = 79.904 },
        .{ .sym = "Kr", .mass = 83.798 },
        .{ .sym = "Rb", .mass = 85.468 },
        .{ .sym = "Sr", .mass = 87.62 },
        .{ .sym = "Y", .mass = 88.906 },
        .{ .sym = "Zr", .mass = 91.224 },
        .{ .sym = "Nb", .mass = 92.906 },
        .{ .sym = "Mo", .mass = 95.95 },
        .{ .sym = "Ru", .mass = 101.07 },
        .{ .sym = "Rh", .mass = 102.91 },
        .{ .sym = "Pd", .mass = 106.42 },
        .{ .sym = "Ag", .mass = 107.87 },
        .{ .sym = "Cd", .mass = 112.41 },
        .{ .sym = "In", .mass = 114.82 },
        .{ .sym = "Sn", .mass = 118.71 },
        .{ .sym = "Sb", .mass = 121.76 },
        .{ .sym = "Te", .mass = 127.60 },
        .{ .sym = "I", .mass = 126.90 },
        .{ .sym = "Xe", .mass = 131.29 },
        .{ .sym = "Cs", .mass = 132.91 },
        .{ .sym = "Ba", .mass = 137.33 },
        .{ .sym = "La", .mass = 138.91 },
        .{ .sym = "Hf", .mass = 178.49 },
        .{ .sym = "Ta", .mass = 180.95 },
        .{ .sym = "W", .mass = 183.84 },
        .{ .sym = "Re", .mass = 186.21 },
        .{ .sym = "Os", .mass = 190.23 },
        .{ .sym = "Ir", .mass = 192.22 },
        .{ .sym = "Pt", .mass = 195.08 },
        .{ .sym = "Au", .mass = 196.97 },
        .{ .sym = "Pb", .mass = 207.2 },
        .{ .sym = "Bi", .mass = 208.98 },
    };

    for (table) |entry| {
        if (std.mem.eql(u8, symbol, entry.sym)) {
            return entry.mass;
        }
    }
    // Fallback: return a reasonable default and log warning
    if (!builtin.is_test) {
        runtime_logging.debugPrint(.warn, .warn, "dfpt: WARNING: unknown element '{s}', using mass=1.0 AMU\n", .{symbol});
    }
    return 1.0;
}

test "atomicMass lookups" {
    try std.testing.expectApproxEqRel(atomicMass("Si"), 28.085, 1e-6);
    try std.testing.expectApproxEqRel(atomicMass("C"), 12.011, 1e-6);
    try std.testing.expectApproxEqRel(atomicMass("Fe"), 55.845, 1e-6);
    // Unknown element returns 1.0
    try std.testing.expectApproxEqRel(atomicMass("Xx"), 1.0, 1e-6);
}
