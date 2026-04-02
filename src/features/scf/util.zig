const plane_wave = @import("../plane_wave/basis.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");

pub const GridRequirement = struct {
    nx: usize,
    ny: usize,
    nz: usize,
};

pub fn gridRequirement(gvecs: []plane_wave.GVector) GridRequirement {
    var max_h: usize = 0;
    var max_k: usize = 0;
    var max_l: usize = 0;
    for (gvecs) |g| {
        const ah = @abs(g.h);
        const ak = @abs(g.k);
        const al = @abs(g.l);
        if (@as(usize, @intCast(ah)) > max_h) max_h = @intCast(ah);
        if (@as(usize, @intCast(ak)) > max_k) max_k = @intCast(ak);
        if (@as(usize, @intCast(al)) > max_l) max_l = @intCast(al);
    }
    return .{
        .nx = 2 * max_h + 1,
        .ny = 2 * max_k + 1,
        .nz = 2 * max_l + 1,
    };
}

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

pub fn nextFftSize(value: usize) usize {
    if (value <= 1) return 1;
    var n = value;
    while (!isFftSize(n)) : (n += 1) {}
    return n;
}

fn isFftSize(value: usize) bool {
    var n = value;
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
}

/// Check if any species has nonlocal coefficients.
pub fn hasNonlocal(species: []hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.dij.len > 0) return true;
    }
    return false;
}

/// Check if any species has QIJ coefficients.
pub fn hasQij(species: []hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.qij.len > 0) return true;
    }
    return false;
}

/// Check if any species uses PAW.
pub fn hasPaw(species: []hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.paw != null) return true;
    }
    return false;
}

/// Compute total valence electrons in the cell.
pub fn totalElectrons(species: []hamiltonian.SpeciesEntry, atoms: []hamiltonian.AtomData) f64 {
    var total: f64 = 0.0;
    for (atoms) |atom| {
        total += species[atom.species_index].z_valence;
    }
    return total;
}
