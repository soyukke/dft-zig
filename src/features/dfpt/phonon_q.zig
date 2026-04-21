//! Finite-q DFPT phonon band structure calculation.
//!
//! Contains the q≠0 perturbation solver, cross-basis operations,
//! complex dynamical matrix construction, q-path generation,
//! and the top-level `runPhononBand` entry point.

const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../scf/scf.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const d3 = @import("../vdw/d3.zig");
const d3_params = @import("../vdw/d3_params.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const config_mod = @import("../config/config.zig");
const model_mod = @import("../dft/model.zig");
const iterative = @import("../linalg/iterative.zig");
const symmetry_mod = @import("../symmetry/symmetry.zig");

const dfpt = @import("dfpt.zig");
const perturbation = dfpt.perturbation;
const sternheimer = dfpt.sternheimer;
const ewald2 = dfpt.ewald2;
const dynmat_mod = dfpt.dynmat;
const dynmat_contrib = dfpt.dynmat_contrib;
const gamma = dfpt.gamma;

const GroundState = dfpt.GroundState;
const PreparedGroundState = dfpt.PreparedGroundState;
const DfptConfig = dfpt.DfptConfig;
const IonicData = dfpt.IonicData;
const PerturbationResult = dfpt.PerturbationResult;
const logDfpt = dfpt.logDfpt;
const logDfptInfo = dfpt.logDfptInfo;

const kpoints_mod = @import("../kpoints/kpoints.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const mesh_mod = @import("../kpoints/mesh.zig");
const reduction = @import("../kpoints/reduction.zig");
const wfn_rot = @import("../symmetry/wavefunction_rotation.zig");

// Submodules (split out of phonon_q.zig)
const qpath_mod = @import("phonon_q/qpath.zig");
const cross_basis = @import("phonon_q/cross_basis.zig");
const dynmat_elem_q = @import("phonon_q/dynmat_elem_q.zig");
const kpt_gs_mod = @import("phonon_q/kpt_gs.zig");
const kpt_dfpt_mod = @import("phonon_q/kpt_dfpt.zig");
const solver_single_mod = @import("phonon_q/solver_single.zig");
const solver_multik_mod = @import("phonon_q/solver_multik.zig");

pub const generateFccQPath = qpath_mod.generateFccQPath;
pub const generateQPathFromConfig = qpath_mod.generateQPathFromConfig;

const applyV1PsiQ = cross_basis.applyV1PsiQ;
pub const applyV1PsiQCached = cross_basis.applyV1PsiQCached;
const computeRho1Q = cross_basis.computeRho1Q;
pub const computeRho1QCached = cross_basis.computeRho1QCached;
const complexRealToReciprocal = cross_basis.complexRealToReciprocal;

pub const computeElecDynmatElementQ = dynmat_elem_q.computeElecDynmatElementQ;
pub const computeNonlocalResponseDynmatQ = dynmat_elem_q.computeNonlocalResponseDynmatQ;
pub const computeNlccCrossDynmatQ = dynmat_elem_q.computeNlccCrossDynmatQ;

pub const KPointGsData = kpt_gs_mod.KPointGsData;
pub const prepareFullBZKpoints = kpt_gs_mod.prepareFullBZKpoints;
pub const prepareFullBZKpointsFromIBZ = kpt_gs_mod.prepareFullBZKpointsFromIBZ;

pub const KPointDfptData = kpt_dfpt_mod.KPointDfptData;
pub const MultiKPertResult = kpt_dfpt_mod.MultiKPertResult;
const buildKPointDfptDataFromGS = kpt_dfpt_mod.buildKPointDfptDataFromGS;

pub const solvePerturbationQ = solver_single_mod.solvePerturbationQ;
pub const solvePerturbationQMultiK = solver_multik_mod.solvePerturbationQMultiK;

const Grid = scf_mod.Grid;


// =====================================================================
// K-point ground-state data (§1) — moved to phonon_q/kpt_gs.zig
// Multi-k-point DFPT data    (§2) — moved to phonon_q/kpt_dfpt.zig
// =====================================================================

// =====================================================================
// Cross-basis operations — moved to phonon_q/cross_basis.zig
// =====================================================================

// =====================================================================
// Complex dynmat element computations — moved to phonon_q/dynmat_elem_q.zig
// =====================================================================

// =====================================================================
// DFPT SCF solver for q≠0 (§5) — moved to phonon_q/solver_single.zig
// =====================================================================

// =====================================================================
// Multi-k-point DFPT perturbation solver (§6) — moved to phonon_q/solver_multik.zig
// =====================================================================
// =====================================================================
// Complex dynamical matrix construction
// =====================================================================

/// Build the full complex dynamical matrix for a finite q-point.
/// Combines electronic, nonlocal, NLCC, Ewald, and self-energy contributions.
/// All (I,J) pairs are computed explicitly; no Hermitianization needed.
fn buildQDynmat(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    vloc1_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []math.Complex,
    charges: []const f64,
    positions: []const math.Vec3,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    q_cart: math.Vec3,
    gvecs_kq: []const plane_wave.GVector,
    apply_ctx_kq: *scf_mod.ApplyContext,
    vxc_g: ?[]const math.Complex,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    const n_atoms = gs.atoms.len;
    const dim = 3 * n_atoms;
    const grid = gs.grid;

    var dyn_q = try alloc.alloc(math.Complex, dim * dim);
    errdefer alloc.free(dyn_q);
    @memset(dyn_q, math.complex.init(0.0, 0.0));

    // ================================================================
    // Step 1: Accumulate monochromatic (+q) terms into dyn_q
    // These need Hermitianization D = D_raw + conj(D_raw^T)
    // ================================================================

    // Electronic contribution (monochromatic)
    for (0..dim) |i| {
        for (0..dim) |j| {
            dyn_q[i * dim + j] = computeElecDynmatElementQ(
                vloc1_gs[i],
                pert_results[j].rho1_g,
                volume,
            );
        }
    }
    logDfpt("dfptQ_dyn: D_elec(0x,0x)=({e:.6},{e:.6}) D_elec(0x,1x)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i, dyn_q[3].r, dyn_q[3].i });

    // Nonlocal response contribution (monochromatic)
    const nl_resp_q = try computeNonlocalResponseDynmatQ(alloc, gs, pert_results, gvecs_kq, apply_ctx_kq, n_atoms);
    defer alloc.free(nl_resp_q);
    logDfpt("dfptQ_dyn: D_nl_resp(0x,0x)=({e:.6},{e:.6}) D_nl_resp(0x,1x)=({e:.6},{e:.6})\n", .{ nl_resp_q[0].r, nl_resp_q[0].i, nl_resp_q[3].r, nl_resp_q[3].i });
    logDfpt("dfptQ_dyn: D_nl_resp(1x,0x)=({e:.6},{e:.6}) D_nl_resp(1x,1x)=({e:.6},{e:.6})\n", .{ nl_resp_q[3 * dim].r, nl_resp_q[3 * dim].i, nl_resp_q[3 * dim + 3].r, nl_resp_q[3 * dim + 3].i });
    // Print D_elec for atom1 block too
    logDfpt("dfptQ_dyn: D_elec(1x,0x)=({e:.6},{e:.6}) D_elec(1x,1x)=({e:.6},{e:.6})\n", .{ dyn_q[3 * dim].r, dyn_q[3 * dim].i, dyn_q[3 * dim + 3].r, dyn_q[3 * dim + 3].i });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], nl_resp_q[i]);
    }

    // NLCC cross contribution (monochromatic)
    if (gs.rho_core != null) {
        const rho1_val_gs = try alloc.alloc([]math.Complex, dim);
        defer alloc.free(rho1_val_gs);
        for (0..dim) |i| {
            rho1_val_gs[i] = pert_results[i].rho1_g;
        }

        const nlcc_cross = try computeNlccCrossDynmatQ(alloc, grid, gs, rho1_val_gs, rho1_core_gs, n_atoms, irr_info);
        defer alloc.free(nlcc_cross);
        logDfpt("dfptQ_dyn: D_nlcc_cross(0x,0x)=({e:.6},{e:.6})\n", .{ nlcc_cross[0].r, nlcc_cross[0].i });
        logDfpt("dfptQ_dyn: D_nlcc_cross(1x,1x)=({e:.6},{e:.6}) D_nlcc_cross(1x,0x)=({e:.6},{e:.6})\n", .{ nlcc_cross[3 * dim + 3].r, nlcc_cross[3 * dim + 3].i, nlcc_cross[3 * dim].r, nlcc_cross[3 * dim].i });
        for (0..dim * dim) |i| {
            dyn_q[i] = math.complex.add(dyn_q[i], nlcc_cross[i]);
        }
    }

    // ================================================================
    // Step 2: No Hermitianization needed.
    // All (I,J) pairs are computed explicitly, and rho1 includes
    // the full factor 4/Ω (matching ABINIT's dfpt_mkrho convention).
    // D_elec uses rho1 (factor 4/Ω included), D_nl_resp uses factor=4
    // directly (matching ABINIT's d2nl: wtk×occ×two = 4).
    // ABINIT's d2sym3 only does one-sided copy for missing elements;
    // since we compute all pairs, no symmetrization is needed.
    // ================================================================

    // ================================================================
    // Step 3: Add Hermitian (q-independent) terms directly
    // These are already the full contribution.
    // ================================================================

    // Ewald contribution (Ha → Ry: ×2)
    const ewald_dyn_q = try ewald2.ewaldDynmatQ(alloc, cell_bohr, recip, charges, positions, q_cart);
    defer alloc.free(ewald_dyn_q);
    logDfpt("dfptQ_dyn: D_ewald(0x,0x)=({e:.6},{e:.6}) D_ewald(0x,1x)=({e:.6},{e:.6}) [Ha]\n", .{ ewald_dyn_q[0].r, ewald_dyn_q[0].i, ewald_dyn_q[3].r, ewald_dyn_q[3].i });
    logDfpt("dfptQ_dyn: D_ewald(0x,0y)=({e:.6},{e:.6}) D_ewald(0y,0y)=({e:.6},{e:.6}) [Ha]\n", .{ ewald_dyn_q[1].r, ewald_dyn_q[1].i, ewald_dyn_q[dim + 1].r, ewald_dyn_q[dim + 1].i });
    logDfpt("dfptQ_dyn: D_ewald full [Ha]:\n", .{});
    for (0..dim) |row| {
        for (0..dim) |col| {
            logDfpt("  ({e:.6},{e:.6})", .{ ewald_dyn_q[row * dim + col].r, ewald_dyn_q[row * dim + col].i });
        }
        logDfpt("\n", .{});
    }
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.scale(ewald_dyn_q[i], 2.0));
    }

    // Self-energy contribution (local V_loc^(2)) — q-independent
    const self_dyn_real = try dynmat_contrib.computeSelfEnergyDynmat(alloc, grid, gs.species, gs.atoms, rho0_g, gs.local_cfg, gs.ff_tables);
    defer alloc.free(self_dyn_real);
    logDfpt("dfptQ_dyn: D_self(0x,0x)={e:.6} D_self(0x,1x)={e:.6}\n", .{ self_dyn_real[0], self_dyn_real[3] });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(self_dyn_real[i], 0.0));
    }

    // Nonlocal self-energy contribution: V_nl^(2) (q-independent)
    const nl_self_real = try dynmat_contrib.computeNonlocalSelfEnergyDynmat(alloc, gs, n_atoms);
    defer alloc.free(nl_self_real);
    logDfpt("dfptQ_dyn: D_nl_self(0x,0x)={e:.6} D_nl_self(0x,1x)={e:.6}\n", .{ nl_self_real[0], nl_self_real[3] });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nl_self_real[i], 0.0));
    }

    // NLCC self contribution (q-independent)
    if (gs.rho_core != null) {
        if (vxc_g) |vg| {
            const nlcc_self_real = try dynmat_contrib.computeNlccSelfDynmat(alloc, grid, gs.species, gs.atoms, vg, gs.rho_core_tables);
            defer alloc.free(nlcc_self_real);
            logDfpt("dfptQ_dyn: D_nlcc_self(0x,0x)={e:.6}\n", .{nlcc_self_real[0]});
            for (0..dim * dim) |i| {
                dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nlcc_self_real[i], 0.0));
            }
        }
    }
    logDfpt("dfptQ_dyn: total(0x,0x)=({e:.6},{e:.6}) total(0x,1x)=({e:.6},{e:.6}) total(0x,1y)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i, dyn_q[3].r, dyn_q[3].i, dyn_q[4].r, dyn_q[4].i });

    return dyn_q;
}

// =====================================================================
// Multi-k-point dynamical matrix construction
// =====================================================================

/// Compute nonlocal response dynmat D_nl_resp summed over all k-points.
/// D(Iα,Jβ) = Σ_k wtk × 4 × Σ_n ⟨dV_nl_{Iα,q} ψ^(0)_{n,k} | δψ_{n,k,Jβ}⟩
pub fn computeNonlocalResponseDynmatQMultiK(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    pert_results: []MultiKPertResult,
    atoms: []const hamiltonian.AtomData,
    n_atoms: usize,
    volume: f64,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    const dim = 3 * n_atoms;
    const dyn = try alloc.alloc(math.Complex, dim * dim);
    @memset(dyn, math.complex.init(0.0, 0.0));

    for (kpts, 0..) |*kd, ik| {
        const n_pw_kq = kd.n_pw_kq;
        const nl_ctx_k = kd.apply_ctx_k.nonlocal_ctx orelse continue;
        const nl_ctx_kq = kd.apply_ctx_kq.nonlocal_ctx orelse continue;

        const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
        defer alloc.free(nl_out);

        for (0..dim) |i| {
            const ia = i / 3;
            const dir_a = i % 3;
            for (0..kd.n_occ) |n| {
                try perturbation.applyNonlocalPerturbationQ(
                    alloc,
                    kd.basis_k.gvecs,
                    kd.basis_kq.gvecs,
                    atoms,
                    nl_ctx_k,
                    nl_ctx_kq,
                    ia,
                    dir_a,
                    1.0 / volume,
                    kd.wavefunctions_k_const[n],
                    nl_out,
                );

                for (0..dim) |j| {
                    if (!irr_info.is_irreducible[j / 3]) continue;
                    var ip = math.complex.init(0.0, 0.0);
                    for (0..n_pw_kq) |g| {
                        ip = math.complex.add(ip, math.complex.mul(
                            math.complex.conj(nl_out[g]),
                            pert_results[j].psi1_per_k[ik][n][g],
                        ));
                    }
                    // Factor 4 × wtk
                    dyn[i * dim + j] = math.complex.add(dyn[i * dim + j], math.complex.scale(ip, 4.0 * kd.weight));
                }
            }
        }
    }

    return dyn;
}

/// Compute nonlocal self-energy dynmat D_nl_self summed over all k-points.
/// This term only contributes to diagonal (I=J) blocks.
pub fn computeNonlocalSelfDynmatMultiK(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    atoms: []const hamiltonian.AtomData,
    n_atoms: usize,
    volume: f64,
) ![]f64 {
    const dim = 3 * n_atoms;
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    const inv_volume = 1.0 / volume;

    for (kpts) |*kd| {
        const n_pw = kd.n_pw_k;
        const nl_ctx = kd.apply_ctx_k.nonlocal_ctx orelse continue;

        const phase = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(phase);

        for (nl_ctx.species) |entry| {
            const g_count = entry.g_count;
            if (g_count != n_pw) continue;
            if (entry.m_total == 0) continue;

            const proj_std = try alloc.alloc(math.Complex, entry.m_total);
            defer alloc.free(proj_std);
            const proj_alpha = try alloc.alloc([3]math.Complex, entry.m_total);
            defer alloc.free(proj_alpha);
            const proj_alpha_beta = try alloc.alloc([3][3]math.Complex, entry.m_total);
            defer alloc.free(proj_alpha_beta);

            for (atoms, 0..) |atom, atom_idx| {
                if (atom.species_index != entry.species_index) continue;

                for (0..kd.n_occ) |n| {
                    const psi_n = kd.wavefunctions_k_const[n];

                    for (0..n_pw) |g| {
                        phase[g] = math.complex.expi(math.Vec3.dot(kd.basis_k.gvecs[g].cart, atom.position));
                    }

                    var b: usize = 0;
                    while (b < entry.beta_count) : (b += 1) {
                        const offset = entry.m_offsets[b];
                        const m_count = entry.m_counts[b];
                        var m_idx: usize = 0;
                        while (m_idx < m_count) : (m_idx += 1) {
                            const phi = entry.phi[(offset + m_idx) * g_count .. (offset + m_idx + 1) * g_count];
                            var p_std = math.complex.init(0.0, 0.0);
                            var p_a: [3]math.Complex = .{ math.complex.init(0.0, 0.0), math.complex.init(0.0, 0.0), math.complex.init(0.0, 0.0) };
                            var p_ab: [3][3]math.Complex = undefined;
                            for (0..3) |a| {
                                for (0..3) |bb| {
                                    p_ab[a][bb] = math.complex.init(0.0, 0.0);
                                }
                            }

                            for (0..n_pw) |g| {
                                const gvec = kd.basis_k.gvecs[g].cart;
                                const gc = [3]f64{ gvec.x, gvec.y, gvec.z };
                                const base = math.complex.scale(math.complex.mul(phase[g], psi_n[g]), phi[g]);
                                p_std = math.complex.add(p_std, base);
                                for (0..3) |a| {
                                    const weighted = math.complex.scale(base, gc[a]);
                                    p_a[a] = math.complex.add(p_a[a], math.complex.init(-weighted.i, weighted.r));
                                }
                                for (0..3) |a| {
                                    for (0..3) |bb| {
                                        p_ab[a][bb] = math.complex.add(p_ab[a][bb], math.complex.scale(base, -gc[a] * gc[bb]));
                                    }
                                }
                            }

                            proj_std[offset + m_idx] = p_std;
                            proj_alpha[offset + m_idx] = p_a;
                            proj_alpha_beta[offset + m_idx] = p_ab;
                        }
                    }

                    // Accumulate with wtk
                    b = 0;
                    while (b < entry.beta_count) : (b += 1) {
                        const l_b = entry.l_list[b];
                        const off_b = entry.m_offsets[b];
                        const mc_b = entry.m_counts[b];

                        var bp: usize = 0;
                        while (bp < entry.beta_count) : (bp += 1) {
                            if (entry.l_list[bp] != l_b) continue;
                            const dij = entry.coeffs[b * entry.beta_count + bp];
                            if (dij == 0.0) continue;
                            const off_bp = entry.m_offsets[bp];

                            var m_idx: usize = 0;
                            while (m_idx < mc_b) : (m_idx += 1) {
                                const p_std_bp = proj_std[off_bp + m_idx];

                                for (0..3) |alpha| {
                                    for (0..3) |beta| {
                                        const t1 = math.complex.mul(
                                            math.complex.conj(proj_alpha_beta[off_b + m_idx][alpha][beta]),
                                            p_std_bp,
                                        );
                                        const t2 = math.complex.mul(
                                            math.complex.conj(proj_alpha[off_b + m_idx][alpha]),
                                            proj_alpha[off_bp + m_idx][beta],
                                        );

                                        // 4/Ω × wtk × dij × Re(t1+t2)
                                        const val = 4.0 * inv_volume * kd.weight * dij * (t1.r + t2.r);
                                        const i_idx = 3 * atom_idx + alpha;
                                        const j_idx = 3 * atom_idx + beta;
                                        dyn[i_idx * dim + j_idx] += val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dyn;
}

/// Build the full complex dynamical matrix for a finite q-point with multiple k-points.
/// Combines electronic, nonlocal, NLCC, Ewald, and self-energy contributions.
fn buildQDynmatMultiK(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    pert_results: []MultiKPertResult,
    vloc1_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []math.Complex,
    gs: GroundState,
    charges: []const f64,
    positions: []const math.Vec3,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    q_cart: math.Vec3,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    rho_core: ?[]const f64,
    vxc_g: ?[]const math.Complex,
    vdw_cfg: config_mod.VdwConfig,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;

    var dyn_q = try alloc.alloc(math.Complex, dim * dim);
    errdefer alloc.free(dyn_q);
    @memset(dyn_q, math.complex.init(0.0, 0.0));

    // ================================================================
    // Step 1: Electronic contribution (k-independent: uses total ρ^(1))
    // Only irreducible columns j are computed (others restored by symmetry)
    // ================================================================
    for (0..dim) |i| {
        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            dyn_q[i * dim + j] = computeElecDynmatElementQ(
                vloc1_gs[i],
                pert_results[j].rho1_g,
                volume,
            );
        }
    }
    logDfpt("dfptQ_mk_dyn: D_elec(0x,0x)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i });

    // ================================================================
    // Step 2: Nonlocal response (k-dependent, summed over k-points)
    // Only irreducible columns j are computed
    // ================================================================
    const nl_resp_q = try computeNonlocalResponseDynmatQMultiK(alloc, kpts, pert_results, atoms, n_atoms, volume, irr_info);
    defer alloc.free(nl_resp_q);
    logDfpt("dfptQ_mk_dyn: D_nl_resp(0x,0x)=({e:.6},{e:.6})\n", .{ nl_resp_q[0].r, nl_resp_q[0].i });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], nl_resp_q[i]);
    }

    // ================================================================
    // Step 3: NLCC cross contribution (k-independent: uses total ρ^(1))
    // ================================================================
    if (rho_core != null) {
        const rho1_val_gs = try alloc.alloc([]math.Complex, dim);
        defer alloc.free(rho1_val_gs);
        for (0..dim) |i| {
            rho1_val_gs[i] = pert_results[i].rho1_g;
        }

        const nlcc_cross = try computeNlccCrossDynmatQ(alloc, grid, gs, rho1_val_gs, rho1_core_gs, n_atoms, irr_info);
        defer alloc.free(nlcc_cross);
        logDfpt("dfptQ_mk_dyn: D_nlcc_cross(0x,0x)=({e:.6},{e:.6})\n", .{ nlcc_cross[0].r, nlcc_cross[0].i });
        for (0..dim * dim) |i| {
            dyn_q[i] = math.complex.add(dyn_q[i], nlcc_cross[i]);
        }
    }

    // ================================================================
    // Step 4: k-independent terms
    // ================================================================

    // Ewald (Ha → Ry: ×2)
    const ewald_dyn_q = try ewald2.ewaldDynmatQ(alloc, cell_bohr, recip, charges, positions, q_cart);
    defer alloc.free(ewald_dyn_q);
    logDfpt("dfptQ_mk_dyn: D_ewald(0x,0x)=({e:.6},{e:.6}) [Ha]\n", .{ ewald_dyn_q[0].r, ewald_dyn_q[0].i });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.scale(ewald_dyn_q[i], 2.0));
    }

    // Self-energy (local V_loc^(2))
    const self_dyn_real = try dynmat_contrib.computeSelfEnergyDynmat(alloc, grid, species, atoms, rho0_g, gs.local_cfg, ff_tables);
    defer alloc.free(self_dyn_real);
    logDfpt("dfptQ_mk_dyn: D_self(0x,0x)={e:.6}\n", .{self_dyn_real[0]});
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(self_dyn_real[i], 0.0));
    }

    // Nonlocal self-energy: V_nl^(2) (k-dependent, summed over k-points)
    const nl_self_real = try computeNonlocalSelfDynmatMultiK(alloc, kpts, atoms, n_atoms, volume);
    defer alloc.free(nl_self_real);
    logDfpt("dfptQ_mk_dyn: D_nl_self(0x,0x)={e:.6}\n", .{nl_self_real[0]});
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nl_self_real[i], 0.0));
    }

    // NLCC self contribution
    if (rho_core != null) {
        if (vxc_g) |vg| {
            const nlcc_self_real = try dynmat_contrib.computeNlccSelfDynmat(alloc, grid, species, atoms, vg, rho_core_tables);
            defer alloc.free(nlcc_self_real);
            logDfpt("dfptQ_mk_dyn: D_nlcc_self(0x,0x)={e:.6}\n", .{nlcc_self_real[0]});
            for (0..dim * dim) |i| {
                dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nlcc_self_real[i], 0.0));
            }
        }
    }

    // ================================================================
    // Step 5: D3 dispersion contribution
    // ================================================================
    if (vdw_cfg.enabled) {
        const atomic_numbers = try alloc.alloc(usize, n_atoms);
        defer alloc.free(atomic_numbers);
        for (atoms, 0..) |atom, i| {
            atomic_numbers[i] = d3_params.atomicNumber(species[atom.species_index].symbol) orelse 0;
        }
        var damping_params = d3_params.pbe_d3bj;
        if (vdw_cfg.s6) |v| damping_params.s6 = v;
        if (vdw_cfg.s8) |v| damping_params.s8 = v;
        if (vdw_cfg.a1) |v| damping_params.a1 = v;
        if (vdw_cfg.a2) |v| damping_params.a2 = v;
        const d3_dyn_q = try d3.computeDynmatQ(
            alloc,
            atomic_numbers,
            positions,
            cell_bohr,
            damping_params,
            vdw_cfg.cutoff_radius,
            vdw_cfg.cn_cutoff,
            q_cart,
        );
        defer alloc.free(d3_dyn_q);
        logDfpt("dfptQ_mk_dyn: D_d3(0x,0x)=({e:.6},{e:.6})\n", .{ d3_dyn_q[0].r, d3_dyn_q[0].i });
        for (0..dim * dim) |i| {
            dyn_q[i] = math.complex.add(dyn_q[i], d3_dyn_q[i]);
        }
    }

    logDfpt("dfptQ_mk_dyn: total(0x,0x)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i });

    return dyn_q;
}

// =====================================================================
// Q-path generation — moved to phonon_q/qpath.zig (see re-exports above)
// =====================================================================

// =====================================================================
// Phonon band structure entry point
// =====================================================================

/// Result of phonon band structure calculation.
pub const PhononBandResult = struct {
    /// q-path distances for plotting
    distances: []f64,
    /// Frequencies [n_q][n_modes] in cm⁻¹
    frequencies: [][]f64,
    /// Number of modes (3 × n_atoms)
    n_modes: usize,
    /// Number of q-points
    n_q: usize,
    /// Labels for high-symmetry points
    labels: [][]const u8,
    /// Label positions (indices into distances)
    label_positions: []usize,

    pub fn deinit(self: *PhononBandResult, alloc: std.mem.Allocator) void {
        for (self.frequencies) |f| alloc.free(f);
        alloc.free(self.frequencies);
        alloc.free(self.distances);
        alloc.free(self.labels);
        alloc.free(self.label_positions);
    }
};

/// Run DFPT phonon band structure calculation.
/// Computes phonon frequencies along a q-path for the FCC lattice.
pub fn runPhononBand(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    model: *const model_mod.Model,
    npoints_per_seg: usize,
) !PhononBandResult {
    const species = model.species;
    const atoms = model.atoms;
    const cell_bohr = model.cell_bohr;
    const recip = model.recip;
    const volume = model.volume_bohr;
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const grid = scf_result.grid;

    logDfptInfo("dfpt_band: starting phonon band calculation ({d} atoms)\n", .{n_atoms});

    // Prepare ground state (PW basis, eigenvalues, wavefunctions, NLCC, etc.)
    var prepared = try dfpt.prepareGroundState(alloc, io, cfg, scf_result, species, atoms, volume, recip);
    defer prepared.deinit();
    const gs = prepared.gs;

    // Ionic data for dynmat construction
    const ionic = try IonicData.init(alloc, species, atoms);
    defer ionic.deinit(alloc);

    // Ground-state density in G-space
    const rho0_g = try scf_mod.realToReciprocal(alloc, grid, scf_result.density, false);
    defer alloc.free(rho0_g);

    // V_xc(G) for NLCC self-energy (need mutable copy since realToReciprocal requires mutable input)
    var vxc_g: ?[]math.Complex = null;
    if (prepared.vxc_r) |v| {
        vxc_g = try scf_mod.realToReciprocal(alloc, grid, v, false);
    }
    defer if (vxc_g) |v| alloc.free(v);

    // Generate q-path
    const qpath = try generateFccQPath(alloc, recip, npoints_per_seg);
    defer alloc.free(qpath.q_points_frac);
    defer alloc.free(qpath.q_points_cart);

    const n_q = qpath.q_points_cart.len;
    logDfptInfo("dfpt_band: {d} q-points along path\n", .{n_q});

    // Build symmetry operations and atom mapping table for dynmat symmetrization
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    defer alloc.free(symops);
    logDfptInfo("dfpt_band: {d} symmetry operations found\n", .{symops.len});

    const sym_data = try dynmat_mod.buildIndsym(alloc, symops, atoms, recip, 1e-5);
    defer {
        for (sym_data.indsym) |row| alloc.free(row);
        alloc.free(sym_data.indsym);
        for (sym_data.tnons_shift) |row| alloc.free(row);
        alloc.free(sym_data.tnons_shift);
    }

    // Allocate result arrays
    var frequencies = try alloc.alloc([]f64, n_q);
    var freq_count: usize = 0;
    errdefer {
        for (0..freq_count) |i| alloc.free(frequencies[i]);
        alloc.free(frequencies);
    }

    const dfpt_cfg = DfptConfig.fromConfig(cfg);

    // ---------------------------------------------------------------
    // Prepare full-BZ k-point ground-state data (q-independent).
    // DFPT requires the full BZ sum over k-points. Even if SCF used
    // IBZ k-points, DFPT needs all k-points in the full BZ because
    // for q≠0, the function f(k, k+q) is not invariant under the
    // symmetry operations that map k to its star.
    // ---------------------------------------------------------------
    const kgs_data = try prepareFullBZKpoints(
        alloc,
        io,
        cfg,
        &gs,
        prepared.local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
    );
    defer {
        for (kgs_data) |*kg| kg.deinit(alloc);
        alloc.free(kgs_data);
    }
    const n_kpts = kgs_data.len;

    // q-point loop
    for (0..n_q) |iq| {
        const q_cart = qpath.q_points_cart[iq];
        const qf = qpath.q_points_frac[iq];
        const q_norm = math.Vec3.norm(q_cart);

        logDfpt("dfpt_band: q[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n", .{ iq, qf.x, qf.y, qf.z, q_norm });
        logDfpt("dfpt_band: q[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });

        // Find irreducible atoms for this q-point
        var irr_info = try dynmat_mod.findIrreducibleAtoms(alloc, symops, sym_data.indsym, n_atoms, qf);
        defer irr_info.deinit(alloc);
        logDfpt("dfpt_band: q[{d}] {d}/{d} irreducible atoms\n", .{ iq, irr_info.n_irr_atoms, n_atoms });

        // Find irreducible perturbations (atom+direction) for this q-point
        // Note: direction-level perturbation reduction (findIrreduciblePerturbations) is available
        // in dynmat_mod but not yet used for SCF solving — buildQDynmatMultiK requires all 3
        // directions per irreducible atom. Log atom-level reduction only.

        // Build KPointDfptData from precomputed ground-state data + q-dependent k+q data
        const pert_thread_count = dfpt.perturbationThreadCount(dim, dfpt_cfg.perturbation_threads);
        const kpts = try buildKPointDfptDataFromGS(
            alloc,
            io,
            kgs_data,
            q_cart,
            q_norm,
            cfg,
            prepared.local_r,
            species,
            atoms,
            recip,
            volume,
            grid,
            pert_thread_count,
        );
        defer {
            for (kpts) |*kd| kd.deinitQOnly(alloc);
            alloc.free(kpts);
        }

        // Solve perturbations for each atom and direction using multi-k solver
        var pert_results_mk = try alloc.alloc(MultiKPertResult, dim);
        var vloc1_gs = try alloc.alloc([]math.Complex, dim);
        var rho1_core_gs = try alloc.alloc([]math.Complex, dim);
        var pert_count_local: usize = 0;
        var vloc1_count_local: usize = 0;
        var rho1_core_count_local: usize = 0;
        defer {
            for (0..pert_count_local) |i| pert_results_mk[i].deinit(alloc);
            alloc.free(pert_results_mk);
            for (0..vloc1_count_local) |i| alloc.free(vloc1_gs[i]);
            alloc.free(vloc1_gs);
            for (0..rho1_core_count_local) |i| alloc.free(rho1_core_gs[i]);
            alloc.free(rho1_core_gs);
        }

        if (pert_thread_count <= 1) {
            // Phase 1: Build vloc1, rho1_core for ALL perturbations (cheap, analytic)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.local_cfg,
                        gs.ff_tables,
                    );
                    vloc1_count_local = pidx + 1;

                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                    rho1_core_count_local = pidx + 1;
                }
            }

            // Phase 2: Solve perturbation SCF for irreducible atoms only (expensive)
            // Direction-level reduction is used for dynmat reconstruction, not for SCF solving,
            // because buildQDynmatMultiK requires all 3 directions per irreducible atom.
            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
            }
            pert_count_local = dim;

            for (irr_info.irr_atom_indices) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    pert_results_mk[pidx] = try solvePerturbationQMultiK(
                        alloc,
                        io,
                        kpts,
                        ia,
                        dir,
                        dfpt_cfg,
                        q_cart,
                        grid,
                        gs,
                        species,
                        atoms,
                        gs.ff_tables,
                        gs.rho_core_tables,
                    );

                    {
                        var rho1_norm: f64 = 0.0;
                        for (pert_results_mk[pidx].rho1_g) |c| {
                            rho1_norm += c.r * c.r + c.i * c.i;
                        }
                        logDfpt("dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n", .{ ia, dir, @sqrt(rho1_norm) });
                    }
                }
            }
        } else {
            // Parallel path — solve only irreducible atoms' perturbations concurrently
            const n_irr_perts = irr_info.n_irr_atoms * 3;
            var pert_dfpt_cfg = dfpt_cfg;
            pert_dfpt_cfg.kpoint_threads = dfpt.kpointThreadsForPertParallel(pert_thread_count, dfpt_cfg.kpoint_threads);

            logDfptInfo("dfpt_band: using {d} pert threads × {d} kpt threads for {d} perturbations ({d} irreducible)\n", .{ pert_thread_count, pert_dfpt_cfg.kpoint_threads, dim, n_irr_perts });

            // Initialize output arrays to safe defaults for cleanup
            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
                vloc1_gs[i] = &.{};
                rho1_core_gs[i] = &.{};
            }
            pert_count_local = dim;
            vloc1_count_local = dim;
            rho1_core_count_local = dim;

            // Build vloc1 and rho1_core for ALL perturbations (cheap, serial)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;
                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.local_cfg,
                        gs.ff_tables,
                    );
                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                }
            }

            var next_index = std.atomic.Value(usize).init(0);
            var stop_flag = std.atomic.Value(u8).init(0);
            var worker_err: ?anyerror = null;
            var err_mutex = std.Io.Mutex.init;
            var log_mutex = std.Io.Mutex.init;

            var qshared = QPointPertShared{
                .alloc = alloc,
                .io = io,
                .kpts = kpts,
                .dfpt_cfg = &pert_dfpt_cfg,
                .q_cart = q_cart,
                .grid = grid,
                .gs = gs,
                .species = species,
                .atoms = atoms,
                .ff_tables = gs.ff_tables,
                .rho_core_tables = gs.rho_core_tables,
                .pert_results_mk = pert_results_mk,
                .vloc1_gs = vloc1_gs,
                .rho1_core_gs = rho1_core_gs,
                .dim = n_irr_perts,
                .irr_pert_indices = null,
                .next_index = &next_index,
                .stop = &stop_flag,
                .err = &worker_err,
                .err_mutex = &err_mutex,
                .log_mutex = &log_mutex,
            };

            // Build irr_pert_indices for atom-level reduction
            const irr_pert_indices = try alloc.alloc(usize, n_irr_perts);
            defer alloc.free(irr_pert_indices);
            {
                var pi: usize = 0;
                for (irr_info.irr_atom_indices) |ia| {
                    for (0..3) |dir_idx| {
                        irr_pert_indices[pi] = 3 * ia + dir_idx;
                        pi += 1;
                    }
                }
            }
            qshared.irr_pert_indices = irr_pert_indices;

            var workers = try alloc.alloc(QPointPertWorker, pert_thread_count);
            defer alloc.free(workers);
            var threads_arr = try alloc.alloc(std.Thread, pert_thread_count - 1);
            defer alloc.free(threads_arr);

            for (0..pert_thread_count) |ti| {
                workers[ti] = .{ .shared = &qshared, .thread_index = ti };
            }

            for (0..pert_thread_count - 1) |ti| {
                threads_arr[ti] = try std.Thread.spawn(.{}, qpointPertWorkerFn, .{&workers[ti + 1]});
            }

            qpointPertWorkerFn(&workers[0]);

            for (threads_arr) |t| {
                t.join();
            }

            if (worker_err) |e| return e;
        }

        // Build complex dynamical matrix from all contributions (multi-k version)
        const dyn_q = try buildQDynmatMultiK(
            alloc,
            kpts,
            pert_results_mk,
            vloc1_gs,
            rho1_core_gs,
            rho0_g,
            gs,
            ionic.charges,
            ionic.positions,
            cell_bohr,
            recip,
            volume,
            q_cart,
            grid,
            species,
            atoms,
            gs.ff_tables,
            gs.rho_core_tables,
            gs.rho_core,
            vxc_g,
            cfg.vdw,
            irr_info,
        );
        defer alloc.free(dyn_q);

        // Reconstruct non-irreducible columns from symmetry
        if (irr_info.n_irr_atoms < n_atoms) {
            dynmat_mod.reconstructDynmatColumnsComplex(dyn_q, n_atoms, irr_info, symops, sym_data.indsym, sym_data.tnons_shift, cell_bohr, qf);
        }

        // Apply acoustic sum rule at Γ-point (q=0)
        if (q_norm < 1e-10) {
            dynmat_mod.applyASRComplex(dyn_q, n_atoms);
        }

        // Mass-weight
        dynmat_mod.massWeightComplex(dyn_q, n_atoms, ionic.masses);

        // Diagonalize complex Hermitian matrix
        var result_q = try dynmat_mod.diagonalizeComplex(alloc, dyn_q, dim);
        defer result_q.deinit(alloc);

        frequencies[iq] = try alloc.alloc(f64, dim);
        @memcpy(frequencies[iq], result_q.frequencies_cm1);
        freq_count = iq + 1;

        logDfptInfo("dfpt_band: q[{d}] freqs:", .{iq});
        for (result_q.frequencies_cm1) |f| {
            logDfptInfo(" {d:.1}", .{f});
        }
        logDfptInfo("\n", .{});
    }

    return PhononBandResult{
        .distances = qpath.distances,
        .frequencies = frequencies,
        .n_modes = dim,
        .n_q = n_q,
        .labels = qpath.labels,
        .label_positions = qpath.label_positions,
    };
}

// =====================================================================
// Perturbation parallelism for q≠0
// =====================================================================

const QPointPertShared = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    dfpt_cfg: *const DfptConfig,
    q_cart: math.Vec3,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    pert_results_mk: []MultiKPertResult,
    vloc1_gs: [][]math.Complex,
    rho1_core_gs: [][]math.Complex,
    dim: usize,
    irr_pert_indices: ?[]const usize = null,

    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

const QPointPertWorker = struct {
    shared: *QPointPertShared,
    thread_index: usize,
};

fn setQPointPertError(shared: *QPointPertShared, e: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);
    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

fn qpointPertWorkerFn(worker: *QPointPertWorker) void {
    const shared = worker.shared;
    const alloc = shared.alloc;

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const work_idx = shared.next_index.fetchAdd(1, .acq_rel);
        if (work_idx >= shared.dim) break;

        // Map work index to actual perturbation index
        const idx = if (shared.irr_pert_indices) |indices| indices[work_idx] else work_idx;
        const ia = idx / 3;
        const dir = idx % 3;

        {
            shared.log_mutex.lockUncancelable(shared.io);
            defer shared.log_mutex.unlock(shared.io);
            const dir_names = [_][]const u8{ "x", "y", "z" };
            logDfpt("dfpt_band: [thread {d}] solving perturbation atom={d} dir={s} ({d}/{d})\n", .{ worker.thread_index, ia, dir_names[dir], work_idx + 1, shared.dim });
        }

        // Solve perturbation SCF with all k-points (vloc1/rho1_core already built by caller)
        shared.pert_results_mk[idx] = solvePerturbationQMultiK(
            alloc,
            shared.io,
            shared.kpts,
            ia,
            dir,
            shared.dfpt_cfg.*,
            shared.q_cart,
            shared.grid,
            shared.gs,
            shared.species,
            shared.atoms,
            shared.ff_tables,
            shared.rho_core_tables,
        ) catch |e| {
            setQPointPertError(shared, e);
            shared.stop.store(1, .release);
            break;
        };

        // Debug: print rho1 norm
        {
            var rho1_norm: f64 = 0.0;
            for (shared.pert_results_mk[idx].rho1_g) |c| {
                rho1_norm += c.r * c.r + c.i * c.i;
            }
            shared.log_mutex.lockUncancelable(shared.io);
            defer shared.log_mutex.unlock(shared.io);
            logDfpt("dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n", .{ ia, dir, @sqrt(rho1_norm) });
        }
    }
}

// =====================================================================
// IFC-interpolated phonon band structure
// =====================================================================

const ifc_mod = @import("ifc.zig");

/// Run DFPT phonon band structure using IFC interpolation.
/// 1. Compute D(q) on a coarse q-grid via DFPT
/// 2. Fourier transform to IFC: C(R)
/// 3. Interpolate D(q') at arbitrary q-path points
pub fn runPhononBandIFC(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    model: *const model_mod.Model,
) !PhononBandResult {
    const species = model.species;
    const atoms = model.atoms;
    const cell_bohr = model.cell_bohr;
    const recip = model.recip;
    const volume = model.volume_bohr;
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const grid = scf_result.grid;
    const qgrid = cfg.dfpt.qgrid orelse return error.MissingQgrid;

    logDfptInfo("dfpt_ifc: starting IFC phonon band ({d} atoms, qgrid={d}x{d}x{d})\n", .{ n_atoms, qgrid[0], qgrid[1], qgrid[2] });

    // Prepare ground state
    var prepared = try dfpt.prepareGroundState(alloc, io, cfg, scf_result, species, atoms, volume, recip);
    defer prepared.deinit();
    const gs = prepared.gs;

    // Ionic data for dynmat construction
    const ionic = try IonicData.init(alloc, species, atoms);
    defer ionic.deinit(alloc);

    // Ground-state density in G-space
    const rho0_g = try scf_mod.realToReciprocal(alloc, grid, scf_result.density, false);
    defer alloc.free(rho0_g);

    // V_xc(G) for NLCC self-energy
    var vxc_g: ?[]math.Complex = null;
    if (prepared.vxc_r) |v| {
        vxc_g = try scf_mod.realToReciprocal(alloc, grid, v, false);
    }
    defer if (vxc_g) |v| alloc.free(v);

    // Build symmetry operations for dynmat symmetrization
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    defer alloc.free(symops);
    logDfptInfo("dfpt_ifc: {d} symmetry operations found\n", .{symops.len});

    const sym_data = try dynmat_mod.buildIndsym(alloc, symops, atoms, recip, 1e-5);
    defer {
        for (sym_data.indsym) |row| alloc.free(row);
        alloc.free(sym_data.indsym);
        for (sym_data.tnons_shift) |row| alloc.free(row);
        alloc.free(sym_data.tnons_shift);
    }

    const dfpt_cfg = DfptConfig.fromConfig(cfg);

    // Prepare full-BZ k-point ground-state data
    const kgs_data = try prepareFullBZKpoints(
        alloc,
        io,
        cfg,
        &gs,
        prepared.local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
    );
    defer {
        for (kgs_data) |*kg| kg.deinit(alloc);
        alloc.free(kgs_data);
    }
    const n_kpts = kgs_data.len;

    // =============================================================
    // Phase 1: Compute D(q) on the coarse q-grid
    // =============================================================
    const shift_zero = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const qgrid_points = try mesh_mod.generateKmesh(alloc, qgrid, recip, shift_zero);
    defer alloc.free(qgrid_points);
    const n_qgrid = qgrid_points.len;
    logDfptInfo("dfpt_ifc: {d} q-grid points for DFPT\n", .{n_qgrid});

    // Store fractional q-points and dynamical matrices
    var q_frac_grid = try alloc.alloc(math.Vec3, n_qgrid);
    defer alloc.free(q_frac_grid);
    var dynmat_grid = try alloc.alloc([]math.Complex, n_qgrid);
    var dynmat_count: usize = 0;
    defer {
        for (0..dynmat_count) |i| alloc.free(dynmat_grid[i]);
        alloc.free(dynmat_grid);
    }

    for (0..n_qgrid) |iq| {
        const q_cart = qgrid_points[iq].k_cart;
        const qf = qgrid_points[iq].k_frac;
        const q_norm = math.Vec3.norm(q_cart);

        q_frac_grid[iq] = qf;

        logDfpt("dfpt_ifc: q_grid[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n", .{ iq, qf.x, qf.y, qf.z, q_norm });
        logDfpt("dfpt_ifc: q_grid[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });

        // Find irreducible atoms for this q-point
        var irr_info = try dynmat_mod.findIrreducibleAtoms(alloc, symops, sym_data.indsym, n_atoms, qf);
        defer irr_info.deinit(alloc);
        logDfpt("dfpt_ifc: q_grid[{d}] {d}/{d} irreducible atoms\n", .{ iq, irr_info.n_irr_atoms, n_atoms });

        // Find irreducible perturbations (atom+direction) for this q-point

        // Build k+q data for this q-point
        const pert_thread_count = dfpt.perturbationThreadCount(dim, dfpt_cfg.perturbation_threads);
        const kpts = try buildKPointDfptDataFromGS(
            alloc,
            io,
            kgs_data,
            q_cart,
            q_norm,
            cfg,
            prepared.local_r,
            species,
            atoms,
            recip,
            volume,
            grid,
            pert_thread_count,
        );
        defer {
            for (kpts) |*kd| kd.deinitQOnly(alloc);
            alloc.free(kpts);
        }

        // Solve perturbations
        var pert_results_mk = try alloc.alloc(MultiKPertResult, dim);
        var vloc1_gs = try alloc.alloc([]math.Complex, dim);
        var rho1_core_gs = try alloc.alloc([]math.Complex, dim);
        var pert_count_local: usize = 0;
        var vloc1_count_local: usize = 0;
        var rho1_core_count_local: usize = 0;
        defer {
            for (0..pert_count_local) |i| pert_results_mk[i].deinit(alloc);
            alloc.free(pert_results_mk);
            for (0..vloc1_count_local) |i| alloc.free(vloc1_gs[i]);
            alloc.free(vloc1_gs);
            for (0..rho1_core_count_local) |i| alloc.free(rho1_core_gs[i]);
            alloc.free(rho1_core_gs);
        }

        if (pert_thread_count <= 1) {
            // Phase 1: Build vloc1, rho1_core for ALL perturbations (cheap)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.local_cfg,
                        gs.ff_tables,
                    );
                    vloc1_count_local = pidx + 1;

                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                    rho1_core_count_local = pidx + 1;
                }
            }

            // Phase 2: Solve perturbation SCF for irreducible atoms only
            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
            }
            pert_count_local = dim;

            for (irr_info.irr_atom_indices) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    pert_results_mk[pidx] = try solvePerturbationQMultiK(
                        alloc,
                        io,
                        kpts,
                        ia,
                        dir,
                        dfpt_cfg,
                        q_cart,
                        grid,
                        gs,
                        species,
                        atoms,
                        gs.ff_tables,
                        gs.rho_core_tables,
                    );
                }
            }
        } else {
            const n_irr_perts = irr_info.n_irr_atoms * 3;
            var pert_dfpt_cfg = dfpt_cfg;
            pert_dfpt_cfg.kpoint_threads = dfpt.kpointThreadsForPertParallel(pert_thread_count, dfpt_cfg.kpoint_threads);

            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
                vloc1_gs[i] = &.{};
                rho1_core_gs[i] = &.{};
            }
            pert_count_local = dim;
            vloc1_count_local = dim;
            rho1_core_count_local = dim;

            // Build vloc1 and rho1_core for ALL perturbations (cheap, serial)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;
                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.local_cfg,
                        gs.ff_tables,
                    );
                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                }
            }

            var next_index = std.atomic.Value(usize).init(0);
            var stop_flag = std.atomic.Value(u8).init(0);
            var worker_err: ?anyerror = null;
            var err_mutex = std.Io.Mutex.init;
            var log_mutex = std.Io.Mutex.init;

            var qshared = QPointPertShared{
                .alloc = alloc,
                .io = io,
                .kpts = kpts,
                .dfpt_cfg = &pert_dfpt_cfg,
                .q_cart = q_cart,
                .grid = grid,
                .gs = gs,
                .species = species,
                .atoms = atoms,
                .ff_tables = gs.ff_tables,
                .rho_core_tables = gs.rho_core_tables,
                .pert_results_mk = pert_results_mk,
                .vloc1_gs = vloc1_gs,
                .rho1_core_gs = rho1_core_gs,
                .dim = n_irr_perts,
                .irr_pert_indices = null,
                .next_index = &next_index,
                .stop = &stop_flag,
                .err = &worker_err,
                .err_mutex = &err_mutex,
                .log_mutex = &log_mutex,
            };

            // Build irr_pert_indices for atom-level reduction
            const irr_pert_indices = try alloc.alloc(usize, n_irr_perts);
            defer alloc.free(irr_pert_indices);
            {
                var pi: usize = 0;
                for (irr_info.irr_atom_indices) |ia| {
                    for (0..3) |dir_idx| {
                        irr_pert_indices[pi] = 3 * ia + dir_idx;
                        pi += 1;
                    }
                }
            }
            qshared.irr_pert_indices = irr_pert_indices;

            var workers = try alloc.alloc(QPointPertWorker, pert_thread_count);
            defer alloc.free(workers);
            var threads_arr = try alloc.alloc(std.Thread, pert_thread_count - 1);
            defer alloc.free(threads_arr);

            for (0..pert_thread_count) |ti| {
                workers[ti] = .{ .shared = &qshared, .thread_index = ti };
            }

            for (0..pert_thread_count - 1) |ti| {
                threads_arr[ti] = try std.Thread.spawn(.{}, qpointPertWorkerFn, .{&workers[ti + 1]});
            }

            qpointPertWorkerFn(&workers[0]);

            for (threads_arr) |t| {
                t.join();
            }

            if (worker_err) |e| return e;
        }

        // Build dynamical matrix D(q) for this q-grid point
        const dyn_q = try buildQDynmatMultiK(
            alloc,
            kpts,
            pert_results_mk,
            vloc1_gs,
            rho1_core_gs,
            rho0_g,
            gs,
            ionic.charges,
            ionic.positions,
            cell_bohr,
            recip,
            volume,
            q_cart,
            grid,
            species,
            atoms,
            gs.ff_tables,
            gs.rho_core_tables,
            gs.rho_core,
            vxc_g,
            cfg.vdw,
            irr_info,
        );

        // Reconstruct non-irreducible columns from symmetry
        if (irr_info.n_irr_atoms < n_atoms) {
            dynmat_mod.reconstructDynmatColumnsComplex(dyn_q, n_atoms, irr_info, symops, sym_data.indsym, sym_data.tnons_shift, cell_bohr, qf);
        }

        dynmat_grid[iq] = dyn_q;
        dynmat_count = iq + 1;

        logDfpt("dfpt_ifc: q_grid[{d}] D(q) computed\n", .{iq});
    }

    // =============================================================
    // Phase 2: Compute IFC: C(R) = FT[D(q)]
    // =============================================================
    logDfptInfo("dfpt_ifc: computing IFC from {d} q-grid points\n", .{n_qgrid});

    // Cast dynmat_grid to const slices for computeIFC
    var dynmat_const = try alloc.alloc([]const math.Complex, n_qgrid);
    defer alloc.free(dynmat_const);
    for (0..n_qgrid) |i| {
        dynmat_const[i] = dynmat_grid[i];
    }

    var ifc_data = try ifc_mod.computeIFC(alloc, dynmat_const, q_frac_grid, qgrid, n_atoms);
    defer ifc_data.deinit(alloc);

    // Apply ASR in IFC space
    ifc_mod.applyASR(&ifc_data);
    logDfptInfo("dfpt_ifc: IFC ASR applied\n", .{});

    // =============================================================
    // Phase 3: Generate q-path and interpolate
    // =============================================================
    const npoints_per_seg = cfg.dfpt.qpath_npoints;

    // Use custom q-path if specified, otherwise FCC default
    var q_points_frac: []math.Vec3 = undefined;
    var q_points_cart: []math.Vec3 = undefined;
    var qpath_distances: []f64 = undefined;
    var qpath_labels: [][]const u8 = undefined;
    var qpath_label_positions: []usize = undefined;

    if (cfg.dfpt.qpath.len >= 2) {
        const qpath = try generateQPathFromConfig(alloc, cfg.dfpt.qpath, npoints_per_seg, recip);
        q_points_frac = qpath.q_points_frac;
        q_points_cart = qpath.q_points_cart;
        qpath_distances = qpath.distances;
        qpath_labels = qpath.labels;
        qpath_label_positions = qpath.label_positions;
    } else {
        const qpath = try generateFccQPath(alloc, recip, npoints_per_seg);
        q_points_frac = qpath.q_points_frac;
        q_points_cart = qpath.q_points_cart;
        qpath_distances = qpath.distances;
        qpath_labels = qpath.labels;
        qpath_label_positions = qpath.label_positions;
    }
    defer alloc.free(q_points_frac);
    defer alloc.free(q_points_cart);

    const n_q = q_points_cart.len;
    logDfptInfo("dfpt_ifc: interpolating {d} q-path points\n", .{n_q});

    // Allocate result arrays
    var frequencies = try alloc.alloc([]f64, n_q);
    var freq_count: usize = 0;
    errdefer {
        for (0..freq_count) |i| alloc.free(frequencies[i]);
        alloc.free(frequencies);
    }

    for (0..n_q) |iq| {
        const qf = q_points_frac[iq];
        const q_norm = math.Vec3.norm(q_points_cart[iq]);

        // Interpolate D(q') from IFC
        const dyn_interp = try ifc_mod.interpolate(alloc, &ifc_data, qf);
        defer alloc.free(dyn_interp);

        // Apply ASR at Gamma
        if (q_norm < 1e-10) {
            dynmat_mod.applyASRComplex(dyn_interp, n_atoms);
        }

        // Mass-weight
        dynmat_mod.massWeightComplex(dyn_interp, n_atoms, ionic.masses);

        // Diagonalize
        var result_q = try dynmat_mod.diagonalizeComplex(alloc, dyn_interp, dim);
        defer result_q.deinit(alloc);

        frequencies[iq] = try alloc.alloc(f64, dim);
        @memcpy(frequencies[iq], result_q.frequencies_cm1);
        freq_count = iq + 1;

        if (iq % 10 == 0 or iq == n_q - 1) {
            logDfptInfo("dfpt_ifc: q[{d}] freqs:", .{iq});
            for (result_q.frequencies_cm1) |f| {
                logDfptInfo(" {d:.1}", .{f});
            }
            logDfptInfo("\n", .{});
        }
    }

    // =============================================================
    // Phase 4 (optional): Phonon DOS from IFC interpolation
    // =============================================================
    if (cfg.dfpt.dos_qmesh) |dos_qmesh| {
        const phonon_dos_mod = @import("phonon_dos.zig");
        logDfptInfo("dfpt_ifc: computing phonon DOS on {d}x{d}x{d} mesh\n", .{ dos_qmesh[0], dos_qmesh[1], dos_qmesh[2] });

        var pdos = try phonon_dos_mod.computePhononDos(
            alloc,
            &ifc_data,
            ionic.masses,
            n_atoms,
            dos_qmesh,
            cfg.dfpt.dos_sigma,
            cfg.dfpt.dos_nbin,
        );
        defer pdos.deinit(alloc);

        // Write to out_dir
        var out_dir = try std.Io.Dir.cwd().openDir(io, cfg.out_dir, .{});
        defer out_dir.close(io);
        try phonon_dos_mod.writePhononDosCsv(io, out_dir, pdos);
        logDfptInfo("dfpt_ifc: phonon DOS written to phonon_dos.csv\n", .{});
    }

    return PhononBandResult{
        .distances = qpath_distances,
        .frequencies = frequencies,
        .n_modes = dim,
        .n_q = n_q,
        .labels = qpath_labels,
        .label_positions = qpath_label_positions,
    };
}
