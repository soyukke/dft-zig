//! Molecular integral module for Gaussian-type orbitals.
//!
//! Re-exports integral evaluation routines for overlap, kinetic energy,
//! nuclear attraction, and electron repulsion integrals.

pub const boys = @import("boys.zig");
pub const overlap = @import("overlap.zig");
pub const kinetic = @import("kinetic.zig");
pub const nuclear = @import("nuclear.zig");
pub const eri = @import("eri.zig");
pub const obara_saika = @import("obara_saika.zig");
pub const rys_roots = @import("rys_roots.zig");
pub const rys_eri = @import("rys_eri.zig");
pub const libcint = @import("libcint.zig");
pub const eri_df = @import("eri_df.zig");

// Re-export key functions (s-only, legacy)
pub const overlapSS = overlap.overlapSS;
pub const buildOverlapMatrix = overlap.buildOverlapMatrix;
pub const kineticSS = kinetic.kineticSS;
pub const buildKineticMatrix = kinetic.buildKineticMatrix;
pub const nuclearAttractionSS = nuclear.nuclearAttractionSS;
pub const totalNuclearAttraction = nuclear.totalNuclearAttraction;
pub const buildNuclearMatrix = nuclear.buildNuclearMatrix;
pub const eriSSSS = eri.eriSSSS;
pub const buildEriTable = eri.buildEriTable;
pub const EriTable = eri.EriTable;

// Re-export general (Obara-Saika) functions
pub const GeneralEriTable = obara_saika.GeneralEriTable;
pub const totalBasisFunctions = obara_saika.totalBasisFunctions;

test {
    _ = boys;
    _ = overlap;
    _ = kinetic;
    _ = nuclear;
    _ = eri;
    _ = obara_saika;
    _ = rys_roots;
    _ = rys_eri;
    _ = libcint;
    _ = eri_df;
}
