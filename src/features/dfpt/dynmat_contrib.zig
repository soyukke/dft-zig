//! Shared dynamical matrix contribution functions.
//!
//! These are q-independent (diagonal-block) terms used by both
//! the Γ-point (`gamma.zig`) and finite-q (`phonon_q.zig`) phonon routines:
//!   - Local self-energy:    V_loc^(2) × ρ^(0)
//!   - Nonlocal self-energy: V_nl^(2)
//!   - NLCC self-energy:     V_xc^(2) × ρ_core

const std = @import("std");
const math = @import("../math/math.zig");
const scf_mod = @import("../scf/scf.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");

const dfpt = @import("dfpt.zig");
const GroundState = dfpt.GroundState;
const logDfpt = dfpt.logDfpt;

const Grid = scf_mod.Grid;

/// Compute the self-energy (non-variational) contribution to the dynamical matrix.
/// D^{self}_{Iα,Iβ} = Ω × Σ_G conj(∂²V_loc/∂u_{Iα}∂u_{Iβ}(G)) × ρ(G)
///                   = -Σ_G G_α G_β × V_form(|G|) × exp(+iG·τ_I) × ρ(G)
///
/// This term only contributes to the diagonal (I=J) blocks.
/// It represents the interaction of the ground-state density with the
/// second derivative of the ionic potential.
pub fn computeSelfEnergyDynmat(
    alloc: std.mem.Allocator,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho0_g: []const math.Complex,
    local_cfg: local_potential.LocalPotentialConfig,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
) ![]f64 {
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    for (0..n_atoms) |ia| {
        const atom = atoms[ia];
        const sp = &species[atom.species_index];

        var it = scf_mod.GVecIterator.init(grid);
        while (it.next()) |g| {
            if (g.gh == 0 and g.gk == 0 and g.gl == 0) continue;

            const g_norm = math.Vec3.norm(g.gvec);

            // V_form(|G|)
            const v_loc = if (ff_tables) |tables|
                tables[atom.species_index].eval(g_norm)
            else
                hamiltonian.localFormFactor(sp, g_norm, local_cfg);

            // exp(+iG·τ_I) × ρ(G) × V_form
            const phase = math.complex.expi(math.Vec3.dot(g.gvec, atom.position));
            const rho_g = rho0_g[g.idx];
            // product = exp(+iG·τ) × ρ(G) × V_form
            const prod = math.complex.scale(math.complex.mul(phase, rho_g), v_loc);

            // Accumulate -G_α G_β × Re(product) for each pair (α, β)
            const g_comp = [3]f64{ g.gvec.x, g.gvec.y, g.gvec.z };
            for (0..3) |a| {
                for (0..3) |b| {
                    const i_idx = 3 * ia + a;
                    const j_idx = 3 * ia + b;
                    dyn[i_idx * dim + j_idx] += -g_comp[a] * g_comp[b] * prod.r;
                }
            }
        }
    }

    return dyn;
}

/// Compute the nonlocal self-energy (V_nl^(2)) contribution to the dynamical matrix.
/// C_nl2_{Iα,Iβ} = (4/Ω) × Σ_n Re[Σ_{ββ'} D_{ββ'} Σ_m
///     {conj(P^{αβ}_{βm}) P_{β'm} + conj(P^α_{βm}) P^β_{β'm}}]
///
/// where P_{βm} = Σ_G φ_β(G) exp(+iG·τ_I) ψ_n(G), etc.
/// This contributes only to diagonal blocks (I=J).
pub fn computeNonlocalSelfEnergyDynmat(
    alloc: std.mem.Allocator,
    gs: GroundState,
    n_atoms: usize,
) ![]f64 {
    const dim = 3 * n_atoms;
    const n_pw = gs.gvecs.len;
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    const nl_ctx = gs.apply_ctx.nonlocal_ctx orelse return dyn;
    const inv_volume = 1.0 / gs.grid.volume;

    // Work buffers for projections
    const phase = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(phase);

    for (nl_ctx.species) |entry| {
        const g_count = entry.g_count;
        if (g_count != n_pw) continue;
        if (entry.m_total == 0) continue;

        // Allocate projection arrays
        const proj_std = try alloc.alloc(math.Complex, entry.m_total);
        defer alloc.free(proj_std);
        const proj_alpha = try alloc.alloc([3]math.Complex, entry.m_total);
        defer alloc.free(proj_alpha);
        const proj_alpha_beta = try alloc.alloc([3][3]math.Complex, entry.m_total);
        defer alloc.free(proj_alpha_beta);

        for (gs.atoms, 0..) |atom, atom_idx| {
            if (atom.species_index != entry.species_index) continue;

            for (0..gs.n_occ) |n| {
                const psi_n = gs.wavefunctions[n];

                // Compute phases: exp(+iG·τ)
                for (0..n_pw) |g| {
                    phase[g] = math.complex.expi(math.Vec3.dot(gs.gvecs[g].cart, atom.position));
                }

                // Compute all projections for this band and atom
                var b: usize = 0;
                while (b < entry.beta_count) : (b += 1) {
                    const offset = entry.m_offsets[b];
                    const m_count = entry.m_counts[b];
                    var m_idx: usize = 0;
                    while (m_idx < m_count) : (m_idx += 1) {
                        const phi_start = (offset + m_idx) * g_count;
                        const phi_end = (offset + m_idx + 1) * g_count;
                        const phi = entry.phi[phi_start..phi_end];
                        const zero_c = math.complex.init(0.0, 0.0);
                        var p_std = zero_c;
                        var p_a: [3]math.Complex = .{ zero_c, zero_c, zero_c };
                        var p_ab: [3][3]math.Complex = undefined;
                        for (0..3) |a| {
                            for (0..3) |bb| {
                                p_ab[a][bb] = math.complex.init(0.0, 0.0);
                            }
                        }

                        for (0..n_pw) |g| {
                            const gvec = gs.gvecs[g].cart;
                            const gc = [3]f64{ gvec.x, gvec.y, gvec.z };
                            const phase_psi = math.complex.mul(phase[g], psi_n[g]);
                            const base = math.complex.scale(phase_psi, phi[g]);
                            // P = Σ_G φ(G) e^{+iGτ} ψ(G)
                            p_std = math.complex.add(p_std, base);
                            // P^α = Σ_G (+iG_α) φ(G) e^{+iGτ} ψ(G)
                            for (0..3) |a| {
                                // (+iG_α) × base = i × G_α × base
                                const weighted = math.complex.scale(base, gc[a]);
                                // multiply by +i: i×(a+bi) = (-b + ai)
                                const i_weighted = math.complex.init(-weighted.i, weighted.r);
                                p_a[a] = math.complex.add(p_a[a], i_weighted);
                            }
                            // P^{αβ} = Σ_G (-G_α G_β) φ(G) e^{+iGτ} ψ(G)
                            for (0..3) |a| {
                                for (0..3) |bb| {
                                    const term = math.complex.scale(base, -gc[a] * gc[bb]);
                                    p_ab[a][bb] = math.complex.add(p_ab[a][bb], term);
                                }
                            }
                        }

                        proj_std[offset + m_idx] = p_std;
                        proj_alpha[offset + m_idx] = p_a;
                        proj_alpha_beta[offset + m_idx] = p_ab;
                    }
                }

                // Accumulate dynmat: (4/Ω) × Re[Σ D Σ_m
                //     {conj(P^{αβ}_β) P_{β'} + conj(P^α_β) P^β_{β'}}]
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
                                    // Term 1: conj(P^{αβ}_b) × P_{b'}
                                    const p_ab_b = proj_alpha_beta[off_b + m_idx][alpha][beta];
                                    const t1 = math.complex.mul(
                                        math.complex.conj(p_ab_b),
                                        p_std_bp,
                                    );
                                    // Term 2: conj(P^α_b) × P^β_{b'}
                                    const t2 = math.complex.mul(
                                        math.complex.conj(proj_alpha[off_b + m_idx][alpha]),
                                        proj_alpha[off_bp + m_idx][beta],
                                    );

                                    const val = 4.0 * inv_volume * dij * (t1.r + t2.r);
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

    return dyn;
}

/// Compute the NLCC self-energy contribution to the dynamical matrix.
///
/// D_NLCC_self(I,αβ) = Σ_G Re[V_xc*(G) × (-G_α G_β) × ρ_core_form(|G|) × exp(-iG·τ_I)]
///
/// This is the second-order term from the rigid shift of the core charge.
/// Only contributes to diagonal (I=J) blocks.
pub fn computeNlccSelfDynmat(
    alloc: std.mem.Allocator,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    vxc_g: []const math.Complex,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) ![]f64 {
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    for (0..n_atoms) |ia| {
        const atom = atoms[ia];
        const sp = &species[atom.species_index];

        if (sp.upf.nlcc.len == 0) continue;

        var it = scf_mod.GVecIterator.init(grid);
        while (it.next()) |g| {
            if (g.gh == 0 and g.gk == 0 and g.gl == 0) continue;

            const g_norm = math.Vec3.norm(g.gvec);

            // ρ_core_form(|G|)
            const rho_core_form = if (rho_core_tables) |tables|
                tables[atom.species_index].eval(g_norm)
            else
                form_factor.rhoCoreG(sp.upf.*, g_norm);

            // exp(-iG·τ_I) — note: using -iGτ consistent with convention
            const phase_val = math.complex.expi(-math.Vec3.dot(g.gvec, atom.position));

            // V_xc*(G) × exp(-iG·τ) × ρ_core_form
            const vxc_conj = math.complex.conj(vxc_g[g.idx]);
            const prod = math.complex.scale(math.complex.mul(vxc_conj, phase_val), rho_core_form);

            // Accumulate -G_α G_β × Re(product) for each pair (α, β)
            const g_comp = [3]f64{ g.gvec.x, g.gvec.y, g.gvec.z };
            for (0..3) |a| {
                for (0..3) |b| {
                    const i_idx = 3 * ia + a;
                    const j_idx = 3 * ia + b;
                    dyn[i_idx * dim + j_idx] += -g_comp[a] * g_comp[b] * prod.r;
                }
            }
        }
    }

    return dyn;
}
