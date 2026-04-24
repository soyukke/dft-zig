//! Finite-q DFPT phonon band structure — facade for the split submodules.
//!
//! The finite-q DFPT code used to live in a single ~3800-line file. It is
//! now split by concern under `phonon_q/`, and this file only wires the
//! public surface that downstream callers (`dfpt.zig`, `electric.zig`)
//! depend on.
//!
//! Submodule layout:
//!   - qpath           — Q-path generation (FCC and from config).
//!   - cross_basis     — V^(1)|ψ⟩ and ρ^(1) across k-basis → k+q-basis.
//!   - dynmat_elem_q   — Complex dynmat element computations for q≠0.
//!   - kpt_gs          — K-point ground-state data (q-independent).
//!   - kpt_dfpt        — Multi-k DFPT data types and k+q builder.
//!   - solver_single   — Single-k DFPT SCF solver (reference implementation).
//!   - solver_multik   — Multi-k DFPT SCF solver (production path).
//!   - dynmat_build    — build_q_dynmat / build_q_dynmat_multi_k + multi-k helpers.
//!   - band_direct     — run_phonon_band (direct q-path) and per-q parallelism.
//!   - band_ifc        — run_phonon_band_ifc (coarse grid → IFC → interpolation).

// Submodules
const qpath_mod = @import("phonon_q/qpath.zig");
const cross_basis = @import("phonon_q/cross_basis.zig");
const dynmat_elem_q = @import("phonon_q/dynmat_elem_q.zig");
const kpt_gs_mod = @import("phonon_q/kpt_gs.zig");
const kpt_dfpt_mod = @import("phonon_q/kpt_dfpt.zig");
const solver_single_mod = @import("phonon_q/solver_single.zig");
const solver_multik_mod = @import("phonon_q/solver_multik.zig");
const dynmat_build_mod = @import("phonon_q/dynmat_build.zig");
const band_direct_mod = @import("phonon_q/band_direct.zig");
const band_ifc_mod = @import("phonon_q/band_ifc.zig");

// Q-path generation
pub const generate_fcc_q_path = qpath_mod.generate_fcc_q_path;
pub const generate_q_path_from_config = qpath_mod.generate_q_path_from_config;

// Cross-basis ops (electric.zig consumes the Cached variants)
pub const apply_v1_psi_q_cached = cross_basis.apply_v1_psi_q_cached;
pub const compute_rho1_q_cached = cross_basis.compute_rho1_q_cached;

// Complex dynmat element computations
pub const compute_elec_dynmat_element_q = dynmat_elem_q.compute_elec_dynmat_element_q;
pub const compute_nonlocal_response_dynmat_q = dynmat_elem_q.compute_nonlocal_response_dynmat_q;
pub const compute_nlcc_cross_dynmat_q = dynmat_elem_q.compute_nlcc_cross_dynmat_q;

// K-point ground-state data (electric.zig uses these directly)
pub const KPointGsData = kpt_gs_mod.KPointGsData;
pub const prepare_full_bz_kpoints = kpt_gs_mod.prepare_full_bz_kpoints;
pub const prepare_full_bz_kpoints_from_ibz = kpt_gs_mod.prepare_full_bz_kpoints_from_ibz;

// Multi-k DFPT data
pub const KPointDfptData = kpt_dfpt_mod.KPointDfptData;
pub const MultiKPertResult = kpt_dfpt_mod.MultiKPertResult;

// DFPT SCF solvers
pub const solve_perturbation_q = solver_single_mod.solve_perturbation_q;
pub const solve_perturbation_q_multi_k = solver_multik_mod.solve_perturbation_q_multi_k;

// Dynamical matrix construction
pub const build_q_dynmat = dynmat_build_mod.build_q_dynmat;
pub const compute_nonlocal_response_dynmat_q_multi_k =
    dynmat_build_mod.compute_nonlocal_response_dynmat_q_multi_k;
pub const compute_nonlocal_self_dynmat_multi_k =
    dynmat_build_mod.compute_nonlocal_self_dynmat_multi_k;

// Phonon band structure entry points
pub const PhononBandResult = band_direct_mod.PhononBandResult;
pub const run_phonon_band = band_direct_mod.run_phonon_band;
pub const run_phonon_band_ifc = band_ifc_mod.run_phonon_band_ifc;

test {
    _ = qpath_mod;
    _ = cross_basis;
    _ = dynmat_elem_q;
    _ = kpt_gs_mod;
    _ = kpt_dfpt_mod;
    _ = solver_single_mod;
    _ = solver_multik_mod;
    _ = dynmat_build_mod;
    _ = band_direct_mod;
    _ = band_ifc_mod;
}
