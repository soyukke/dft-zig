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
};

pub const Term = union(enum) {
    kinetic: TermKinetic,
    atomic_local: TermAtomicLocal,
    atomic_nonlocal: TermAtomicNonlocal,
    hartree: TermHartree,
    xc: TermXc,
    ewald: TermEwald,
};

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
    try list.append(alloc, .{ .ewald = .{ .alpha = cfg.ewald.alpha } });

    return list.toOwnedSlice(alloc);
}
