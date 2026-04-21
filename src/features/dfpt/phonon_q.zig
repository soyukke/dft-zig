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
//!   - dynmat_build    — buildQDynmat / buildQDynmatMultiK + multi-k helpers.
//!   - band_direct     — runPhononBand (direct q-path) and per-q parallelism.
//!   - band_ifc        — runPhononBandIFC (coarse grid → IFC → interpolation).

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
pub const generateFccQPath = qpath_mod.generateFccQPath;
pub const generateQPathFromConfig = qpath_mod.generateQPathFromConfig;

// Cross-basis ops (electric.zig consumes the Cached variants)
pub const applyV1PsiQCached = cross_basis.applyV1PsiQCached;
pub const computeRho1QCached = cross_basis.computeRho1QCached;

// Complex dynmat element computations
pub const computeElecDynmatElementQ = dynmat_elem_q.computeElecDynmatElementQ;
pub const computeNonlocalResponseDynmatQ = dynmat_elem_q.computeNonlocalResponseDynmatQ;
pub const computeNlccCrossDynmatQ = dynmat_elem_q.computeNlccCrossDynmatQ;

// K-point ground-state data (electric.zig uses these directly)
pub const KPointGsData = kpt_gs_mod.KPointGsData;
pub const prepareFullBZKpoints = kpt_gs_mod.prepareFullBZKpoints;
pub const prepareFullBZKpointsFromIBZ = kpt_gs_mod.prepareFullBZKpointsFromIBZ;

// Multi-k DFPT data
pub const KPointDfptData = kpt_dfpt_mod.KPointDfptData;
pub const MultiKPertResult = kpt_dfpt_mod.MultiKPertResult;

// DFPT SCF solvers
pub const solvePerturbationQ = solver_single_mod.solvePerturbationQ;
pub const solvePerturbationQMultiK = solver_multik_mod.solvePerturbationQMultiK;

// Dynamical matrix construction
pub const buildQDynmat = dynmat_build_mod.buildQDynmat;
pub const computeNonlocalResponseDynmatQMultiK = dynmat_build_mod.computeNonlocalResponseDynmatQMultiK;
pub const computeNonlocalSelfDynmatMultiK = dynmat_build_mod.computeNonlocalSelfDynmatMultiK;

// Phonon band structure entry points
pub const PhononBandResult = band_direct_mod.PhononBandResult;
pub const runPhononBand = band_direct_mod.runPhononBand;
pub const runPhononBandIFC = band_ifc_mod.runPhononBandIFC;

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
