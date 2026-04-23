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
pub const overlap_ss = overlap.overlap_ss;
pub const build_overlap_matrix = overlap.build_overlap_matrix;
pub const kinetic_ss = kinetic.kinetic_ss;
pub const build_kinetic_matrix = kinetic.build_kinetic_matrix;
pub const nuclear_attraction_ss = nuclear.nuclear_attraction_ss;
pub const total_nuclear_attraction = nuclear.total_nuclear_attraction;
pub const build_nuclear_matrix = nuclear.build_nuclear_matrix;
pub const eri_ssss = eri.eri_ssss;
pub const build_eri_table = eri.build_eri_table;
pub const EriTable = eri.EriTable;

// Re-export general (Obara-Saika) functions
pub const GeneralEriTable = obara_saika.GeneralEriTable;
pub const total_basis_functions = obara_saika.total_basis_functions;

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
