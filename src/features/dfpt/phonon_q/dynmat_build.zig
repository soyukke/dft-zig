//! Dynamical-matrix construction for a single q-point.
//!
//! Two top-level builders live here:
//!   - buildQDynmat        — reference single-k builder (currently dead code,
//!                           retained alongside its single-k solver).
//!   - buildQDynmatMultiK  — multi-k production path used by runPhononBand
//!                           and runPhononBandIFC.
//!
//! Each builder combines the q≠0 contributions (electronic, nonlocal
//! response, NLCC cross) with the k-/q-independent self-energy + Ewald
//! + D3 terms. Multi-k nonlocal helpers (computeNonlocal*MultiK) are also
//! here since they are logically part of the dynmat assembly step.

const std = @import("std");
const math = @import("../../math/math.zig");
const scf_mod = @import("../../scf/scf.zig");
const plane_wave = @import("../../plane_wave/basis.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const form_factor = @import("../../pseudopotential/form_factor.zig");
const config_mod = @import("../../config/config.zig");
const d3 = @import("../../vdw/d3.zig");
const d3_params = @import("../../vdw/d3_params.zig");

const dfpt = @import("../dfpt.zig");
const perturbation = dfpt.perturbation;
const ewald2 = dfpt.ewald2;
const dynmat_mod = dfpt.dynmat;
const dynmat_contrib = dfpt.dynmat_contrib;
const GroundState = dfpt.GroundState;
const PerturbationResult = dfpt.PerturbationResult;
const logDfpt = dfpt.logDfpt;

const kpt_dfpt = @import("kpt_dfpt.zig");
const KPointDfptData = kpt_dfpt.KPointDfptData;
const MultiKPertResult = kpt_dfpt.MultiKPertResult;

const dynmat_elem_q = @import("dynmat_elem_q.zig");
const computeElecDynmatElementQ = dynmat_elem_q.computeElecDynmatElementQ;
const computeNonlocalResponseDynmatQ = dynmat_elem_q.computeNonlocalResponseDynmatQ;
const computeNlccCrossDynmatQ = dynmat_elem_q.computeNlccCrossDynmatQ;

const Grid = scf_mod.Grid;

/// Build the full complex dynamical matrix for a finite q-point.
/// Combines electronic, nonlocal, NLCC, Ewald, and self-energy contributions.
/// All (I,J) pairs are computed explicitly; no Hermitianization needed.
pub fn buildQDynmat(
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
    logDfpt(
        "dfptQ_dyn: D_elec(0x,0x)=({e:.6},{e:.6}) D_elec(0x,1x)=({e:.6},{e:.6})\n",
        .{ dyn_q[0].r, dyn_q[0].i, dyn_q[3].r, dyn_q[3].i },
    );

    // Nonlocal response contribution (monochromatic)
    const nl_resp_q = try computeNonlocalResponseDynmatQ(
        alloc,
        gs,
        pert_results,
        gvecs_kq,
        apply_ctx_kq,
        n_atoms,
    );
    defer alloc.free(nl_resp_q);
    logDfpt(
        "dfptQ_dyn: D_nl_resp(0x,0x)=({e:.6},{e:.6}) D_nl_resp(0x,1x)=({e:.6},{e:.6})\n",
        .{ nl_resp_q[0].r, nl_resp_q[0].i, nl_resp_q[3].r, nl_resp_q[3].i },
    );
    logDfpt(
        "dfptQ_dyn: D_nl_resp(1x,0x)=({e:.6},{e:.6}) D_nl_resp(1x,1x)=({e:.6},{e:.6})\n",
        .{
            nl_resp_q[3 * dim].r,
            nl_resp_q[3 * dim].i,
            nl_resp_q[3 * dim + 3].r,
            nl_resp_q[3 * dim + 3].i,
        },
    );
    // Print D_elec for atom1 block too
    logDfpt(
        "dfptQ_dyn: D_elec(1x,0x)=({e:.6},{e:.6}) D_elec(1x,1x)=({e:.6},{e:.6})\n",
        .{ dyn_q[3 * dim].r, dyn_q[3 * dim].i, dyn_q[3 * dim + 3].r, dyn_q[3 * dim + 3].i },
    );
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

        const nlcc_cross = try computeNlccCrossDynmatQ(
            alloc,
            grid,
            gs,
            rho1_val_gs,
            rho1_core_gs,
            n_atoms,
            irr_info,
        );
        defer alloc.free(nlcc_cross);
        logDfpt(
            "dfptQ_dyn: D_nlcc_cross(0x,0x)=({e:.6},{e:.6})\n",
            .{ nlcc_cross[0].r, nlcc_cross[0].i },
        );
        logDfpt(
            "dfptQ_dyn: D_nlcc_cross(1x,1x)=({e:.6},{e:.6})" ++
                " D_nlcc_cross(1x,0x)=({e:.6},{e:.6})\n",
            .{
                nlcc_cross[3 * dim + 3].r,
                nlcc_cross[3 * dim + 3].i,
                nlcc_cross[3 * dim].r,
                nlcc_cross[3 * dim].i,
            },
        );
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
    const ewald_dyn_q = try ewald2.ewaldDynmatQ(
        alloc,
        cell_bohr,
        recip,
        charges,
        positions,
        q_cart,
    );
    defer alloc.free(ewald_dyn_q);
    logDfpt(
        "dfptQ_dyn: D_ewald(0x,0x)=({e:.6},{e:.6}) D_ewald(0x,1x)=({e:.6},{e:.6}) [Ha]\n",
        .{ ewald_dyn_q[0].r, ewald_dyn_q[0].i, ewald_dyn_q[3].r, ewald_dyn_q[3].i },
    );
    logDfpt(
        "dfptQ_dyn: D_ewald(0x,0y)=({e:.6},{e:.6}) D_ewald(0y,0y)=({e:.6},{e:.6}) [Ha]\n",
        .{
            ewald_dyn_q[1].r,
            ewald_dyn_q[1].i,
            ewald_dyn_q[dim + 1].r,
            ewald_dyn_q[dim + 1].i,
        },
    );
    logDfpt("dfptQ_dyn: D_ewald full [Ha]:\n", .{});
    for (0..dim) |row| {
        for (0..dim) |col| {
            logDfpt(
                "  ({e:.6},{e:.6})",
                .{ ewald_dyn_q[row * dim + col].r, ewald_dyn_q[row * dim + col].i },
            );
        }
        logDfpt("\n", .{});
    }
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.scale(ewald_dyn_q[i], 2.0));
    }

    // Self-energy contribution (local V_loc^(2)) — q-independent
    const self_dyn_real = try dynmat_contrib.computeSelfEnergyDynmat(
        alloc,
        grid,
        gs.species,
        gs.atoms,
        rho0_g,
        gs.local_cfg,
        gs.ff_tables,
    );
    defer alloc.free(self_dyn_real);
    logDfpt(
        "dfptQ_dyn: D_self(0x,0x)={e:.6} D_self(0x,1x)={e:.6}\n",
        .{ self_dyn_real[0], self_dyn_real[3] },
    );
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(self_dyn_real[i], 0.0));
    }

    // Nonlocal self-energy contribution: V_nl^(2) (q-independent)
    const nl_self_real = try dynmat_contrib.computeNonlocalSelfEnergyDynmat(alloc, gs, n_atoms);
    defer alloc.free(nl_self_real);
    logDfpt(
        "dfptQ_dyn: D_nl_self(0x,0x)={e:.6} D_nl_self(0x,1x)={e:.6}\n",
        .{ nl_self_real[0], nl_self_real[3] },
    );
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nl_self_real[i], 0.0));
    }

    // NLCC self contribution (q-independent)
    if (gs.rho_core != null) {
        if (vxc_g) |vg| {
            const nlcc_self_real = try dynmat_contrib.computeNlccSelfDynmat(
                alloc,
                grid,
                gs.species,
                gs.atoms,
                vg,
                gs.rho_core_tables,
            );
            defer alloc.free(nlcc_self_real);
            logDfpt("dfptQ_dyn: D_nlcc_self(0x,0x)={e:.6}\n", .{nlcc_self_real[0]});
            for (0..dim * dim) |i| {
                dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nlcc_self_real[i], 0.0));
            }
        }
    }
    logDfpt(
        "dfptQ_dyn: total(0x,0x)=({e:.6},{e:.6}) total(0x,1x)=({e:.6},{e:.6})" ++
            " total(0x,1y)=({e:.6},{e:.6})\n",
        .{ dyn_q[0].r, dyn_q[0].i, dyn_q[3].r, dyn_q[3].i, dyn_q[4].r, dyn_q[4].i },
    );

    return dyn_q;
}

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
                    dyn[i * dim + j] = math.complex.add(
                        dyn[i * dim + j],
                        math.complex.scale(ip, 4.0 * kd.weight),
                    );
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
                        phase[g] = math.complex.expi(
                            math.Vec3.dot(kd.basis_k.gvecs[g].cart, atom.position),
                        );
                    }

                    var b: usize = 0;
                    while (b < entry.beta_count) : (b += 1) {
                        const offset = entry.m_offsets[b];
                        const m_count = entry.m_counts[b];
                        var m_idx: usize = 0;
                        while (m_idx < m_count) : (m_idx += 1) {
                            const phi_start = (offset + m_idx) * g_count;
                            const phi_end = (offset + m_idx + 1) * g_count;
                            const phi = entry.phi[phi_start..phi_end];
                            var p_std = math.complex.init(0.0, 0.0);
                            var p_a: [3]math.Complex = .{
                                math.complex.init(0.0, 0.0),
                                math.complex.init(0.0, 0.0),
                                math.complex.init(0.0, 0.0),
                            };
                            var p_ab: [3][3]math.Complex = undefined;
                            for (0..3) |a| {
                                for (0..3) |bb| {
                                    p_ab[a][bb] = math.complex.init(0.0, 0.0);
                                }
                            }

                            for (0..n_pw) |g| {
                                const gvec = kd.basis_k.gvecs[g].cart;
                                const gc = [3]f64{ gvec.x, gvec.y, gvec.z };
                                const base = math.complex.scale(
                                    math.complex.mul(phase[g], psi_n[g]),
                                    phi[g],
                                );
                                p_std = math.complex.add(p_std, base);
                                for (0..3) |a| {
                                    const weighted = math.complex.scale(base, gc[a]);
                                    p_a[a] = math.complex.add(
                                        p_a[a],
                                        math.complex.init(-weighted.i, weighted.r),
                                    );
                                }
                                for (0..3) |a| {
                                    for (0..3) |bb| {
                                        p_ab[a][bb] = math.complex.add(
                                            p_ab[a][bb],
                                            math.complex.scale(base, -gc[a] * gc[bb]),
                                        );
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
                                        const pab_conj = math.complex.conj(
                                            proj_alpha_beta[off_b + m_idx][alpha][beta],
                                        );
                                        const t1 = math.complex.mul(pab_conj, p_std_bp);
                                        const t2 = math.complex.mul(
                                            math.complex.conj(proj_alpha[off_b + m_idx][alpha]),
                                            proj_alpha[off_bp + m_idx][beta],
                                        );

                                        // 4/Ω × wtk × dij × Re(t1+t2)
                                        const val = 4.0 * inv_volume * kd.weight * dij *
                                            (t1.r + t2.r);
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
pub fn buildQDynmatMultiK(
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
    const nl_resp_q = try computeNonlocalResponseDynmatQMultiK(
        alloc,
        kpts,
        pert_results,
        atoms,
        n_atoms,
        volume,
        irr_info,
    );
    defer alloc.free(nl_resp_q);
    logDfpt(
        "dfptQ_mk_dyn: D_nl_resp(0x,0x)=({e:.6},{e:.6})\n",
        .{ nl_resp_q[0].r, nl_resp_q[0].i },
    );
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

        const nlcc_cross = try computeNlccCrossDynmatQ(
            alloc,
            grid,
            gs,
            rho1_val_gs,
            rho1_core_gs,
            n_atoms,
            irr_info,
        );
        defer alloc.free(nlcc_cross);
        logDfpt(
            "dfptQ_mk_dyn: D_nlcc_cross(0x,0x)=({e:.6},{e:.6})\n",
            .{ nlcc_cross[0].r, nlcc_cross[0].i },
        );
        for (0..dim * dim) |i| {
            dyn_q[i] = math.complex.add(dyn_q[i], nlcc_cross[i]);
        }
    }

    // ================================================================
    // Step 4: k-independent terms
    // ================================================================

    // Ewald (Ha → Ry: ×2)
    const ewald_dyn_q = try ewald2.ewaldDynmatQ(
        alloc,
        cell_bohr,
        recip,
        charges,
        positions,
        q_cart,
    );
    defer alloc.free(ewald_dyn_q);
    logDfpt(
        "dfptQ_mk_dyn: D_ewald(0x,0x)=({e:.6},{e:.6}) [Ha]\n",
        .{ ewald_dyn_q[0].r, ewald_dyn_q[0].i },
    );
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.scale(ewald_dyn_q[i], 2.0));
    }

    // Self-energy (local V_loc^(2))
    const self_dyn_real = try dynmat_contrib.computeSelfEnergyDynmat(
        alloc,
        grid,
        species,
        atoms,
        rho0_g,
        gs.local_cfg,
        ff_tables,
    );
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
            const nlcc_self_real = try dynmat_contrib.computeNlccSelfDynmat(
                alloc,
                grid,
                species,
                atoms,
                vg,
                rho_core_tables,
            );
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
