//! Physical terms of the DFT Hamiltonian.
//!
//! Each Term contributes:
//!   (1) an energy value (possibly depending on the current density)
//!   (2) a per-k-point operator that sums into H|ψ⟩
//!
//! This is the dft-zig analogue of DFTK.jl's Term abstraction. The long-term
//! goal is to replace the monolithic buildHamiltonian / buildPotentialGrid
//! flow with an iteration over physics terms, each self-contained.
//!
//! This module (Step 2a) defines the types. Wiring to the SCF loop happens
//! in subsequent commits.

const std = @import("std");
const config_mod = @import("../config/config.zig");
const ewald_mod = @import("../ewald/ewald.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const xc_mod = @import("../xc/xc.zig");

/// Kinetic energy: T|ψ⟩ = |G+k|²/2 |ψ⟩ (Rydberg).
/// Density-independent; operator-only (does not contribute to V_eff(r)).
pub const TermKinetic = struct {};

/// Local component of the ionic pseudopotential: V_loc(r) = Σ_a v^a_loc(r-R_a).
/// Depends on atomic positions (via Model), not on ρ. Contributes to V_eff(G).
pub const TermAtomicLocal = struct {};

/// Non-local pseudopotential: Σ_a Σ_ij |β^a_i⟩ D_ij ⟨β^a_j|.
/// Density-independent. Operator-only.
pub const TermAtomicNonlocal = struct {};

/// Hartree: V_H(G) = 8π/G² ρ(G) (Rydberg).
/// Uses a real-space Coulomb cutoff for isolated boundary conditions.
pub const TermHartree = struct {
    /// true for isolated boundary conditions (Coulomb cutoff applied at eval time).
    isolated: bool = false,
};

/// Exchange-correlation: V_xc[ρ](r), E_xc[ρ].
pub const TermXc = struct {
    functional: xc_mod.Functional,
};

/// Ion-ion Ewald sum. Energy-only (no contribution to H|ψ⟩).
pub const TermEwald = struct {
    alpha: f64,
    rcut: f64 = 0.0,
    gcut: f64 = 0.0,
    tol: f64 = 0.0,
    quiet: bool = false,
};

pub const Term = union(enum) {
    kinetic: TermKinetic,
    atomic_local: TermAtomicLocal,
    atomic_nonlocal: TermAtomicNonlocal,
    hartree: TermHartree,
    xc: TermXc,
    ewald: TermEwald,
};

/// Inputs passed to term energy evaluators.
///
/// Mandatory fields describe the physical system (geometry + species + config).
/// Optional fields carry state that only certain terms need (e.g. ρ for
/// Hartree/XC). Callers populate what they have; each term either uses it
/// or ignores it.
pub const EvalInput = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
    rho: ?[]const f64 = null,
};

/// Return the scalar energy contribution of a term.
///
/// Terms that do not yet route through this contract return 0 and are
/// still accumulated via the existing SCF code path. As each term is
/// wired, the corresponding legacy path is retired.
pub fn termEnergy(term: Term, input: EvalInput) !f64 {
    return switch (term) {
        .kinetic, .atomic_local, .atomic_nonlocal, .hartree, .xc => 0.0,
        .ewald => |t| try ewaldEnergy(t, input),
    };
}

fn ewaldEnergy(term: TermEwald, input: EvalInput) !f64 {
    const count = input.atoms.len;
    if (count == 0) return 0.0;
    const charges = try input.alloc.alloc(f64, count);
    defer input.alloc.free(charges);
    const positions = try input.alloc.alloc(math.Vec3, count);
    defer input.alloc.free(positions);
    for (input.atoms, 0..) |atom, i| {
        charges[i] = input.species[atom.species_index].z_valence;
        positions[i] = atom.position;
    }
    const params = ewald_mod.Params{
        .alpha = term.alpha,
        .rcut = term.rcut,
        .gcut = term.gcut,
        .tol = term.tol,
        .quiet = term.quiet,
    };
    return try ewald_mod.ionIonEnergy(input.io, input.cell_bohr, input.recip, charges, positions, params);
}

/// Build the canonical term list for a given configuration.
/// Caller owns the returned slice.
pub fn termsFromConfig(alloc: std.mem.Allocator, cfg: config_mod.Config) ![]Term {
    var list: std.ArrayList(Term) = .empty;
    errdefer list.deinit(alloc);

    try list.append(alloc, .{ .kinetic = .{} });
    try list.append(alloc, .{ .atomic_local = .{} });
    if (cfg.scf.enable_nonlocal) {
        try list.append(alloc, .{ .atomic_nonlocal = .{} });
    }
    try list.append(alloc, .{ .hartree = .{ .isolated = (cfg.boundary == .isolated) } });
    try list.append(alloc, .{ .xc = .{ .functional = cfg.scf.xc } });
    try list.append(alloc, .{ .ewald = .{
        .alpha = cfg.ewald.alpha,
        .rcut = cfg.ewald.rcut,
        .gcut = cfg.ewald.gcut,
        .tol = cfg.ewald.tol,
        .quiet = cfg.scf.quiet,
    } });

    return list.toOwnedSlice(alloc);
}

test "termEnergy(.ewald) matches direct ionIonEnergy" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    // Graphene cell (Bohr) — reuse the established benchmark.
    const a = 4.6487262675;
    const c = 37.7945225;
    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * std.math.sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );
    const recip = math.reciprocal(cell);
    const volume = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));

    var species_arr = [_]hamiltonian.SpeciesEntry{
        .{ .symbol = "C", .upf = undefined, .z_valence = 4.0, .epsatm_ry = 0.0 },
    };
    const atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .species_index = 0 },
        .{ .position = .{ .x = 3.0991508450, .y = 2.6839433578, .z = 0.0 }, .species_index = 0 },
    };

    const charges = [_]f64{ 4.0, 4.0 };
    const positions = [_]math.Vec3{ atoms[0].position, atoms[1].position };
    const direct_params = ewald_mod.Params{ .alpha = 0.0, .rcut = 0.0, .gcut = 0.0, .tol = 0.0, .quiet = true };
    const e_direct = try ewald_mod.ionIonEnergy(io, cell, recip, &charges, &positions, direct_params);

    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .species = &species_arr,
        .atoms = &atoms,
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
    };
    const e_term = try termEnergy(.{ .ewald = .{ .alpha = 0.0, .quiet = true } }, input);

    try testing.expectApproxEqRel(e_direct, e_term, 1e-12);
}
