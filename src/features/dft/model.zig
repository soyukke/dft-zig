//! Physical model of the system: geometry, species, and pseudopotentials.
//!
//! Separates "what we are simulating" from numerical parameters (ecut,
//! kmesh, fft_size) which belong in the Basis. Following DFTK.jl:
//! Model = physics, Basis = discretization, Hamiltonian = Model + Basis + ρ.
//!
//! A Model borrows all slices; the caller keeps the underlying storage alive.

const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const term_mod = @import("term.zig");

pub const Term = term_mod.Term;

pub const Model = struct {
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
};
