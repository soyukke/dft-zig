const std = @import("std");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");

pub fn nextPow2(value: usize) usize {
    if (value <= 1) return 1;
    var v = value - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if (@sizeOf(usize) == 8) v |= v >> 32;
    return v + 1;
}

/// Compute RMS density difference.
pub fn densityDiff(rho: []f64, rho_new: []f64) f64 {
    var sum: f64 = 0.0;
    for (rho, 0..) |value, i| {
        const diff = rho_new[i] - value;
        sum += diff * diff;
    }
    return std.math.sqrt(sum / @as(f64, @floatFromInt(rho.len)));
}

/// Check if any species has nonlocal coefficients.
pub fn hasNonlocal(species: []const hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.dij.len > 0) return true;
    }
    return false;
}

/// Check if any species has QIJ coefficients.
pub fn hasQij(species: []const hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.qij.len > 0) return true;
    }
    return false;
}

/// Check if any species uses PAW.
pub fn hasPaw(species: []const hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.paw != null) return true;
    }
    return false;
}

/// Compute total valence electrons in the cell.
pub fn totalElectrons(species: []const hamiltonian.SpeciesEntry, atoms: []const hamiltonian.AtomData) f64 {
    var total: f64 = 0.0;
    for (atoms) |atom| {
        total += species[atom.species_index].z_valence;
    }
    return total;
}
