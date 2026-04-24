//! Dynamical-matrix construction for a single q-point.
//!
//! Two top-level builders live here:
//!   - build_q_dynmat        — reference single-k builder (currently dead code,
//!                           retained alongside its single-k solver).
//!   - build_q_dynmat_multi_k  — multi-k production path used by run_phonon_band
//!                           and run_phonon_band_ifc.
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
const log_dfpt = dfpt.log_dfpt;

const kpt_dfpt = @import("kpt_dfpt.zig");
const KPointDfptData = kpt_dfpt.KPointDfptData;
const MultiKPertResult = kpt_dfpt.MultiKPertResult;

const dynmat_elem_q = @import("dynmat_elem_q.zig");
const compute_elec_dynmat_element_q = dynmat_elem_q.compute_elec_dynmat_element_q;
const compute_nonlocal_response_dynmat_q = dynmat_elem_q.compute_nonlocal_response_dynmat_q;
const compute_nlcc_cross_dynmat_q = dynmat_elem_q.compute_nlcc_cross_dynmat_q;

const Grid = scf_mod.Grid;

fn alloc_dynmat(
    alloc: std.mem.Allocator,
    dim: usize,
) ![]math.Complex {
    const dyn = try alloc.alloc(math.Complex, dim * dim);
    @memset(dyn, math.complex.init(0.0, 0.0));
    return dyn;
}

fn add_complex_dynmat(
    dst: []math.Complex,
    src: []const math.Complex,
) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*value, addend| {
        value.* = math.complex.add(value.*, addend);
    }
}

fn add_real_dynmat(
    dst: []math.Complex,
    src: []const f64,
) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*value, addend| {
        value.* = math.complex.add(value.*, math.complex.init(addend, 0.0));
    }
}

fn collect_rho1_grids(
    alloc: std.mem.Allocator,
    pert_results: anytype,
    dim: usize,
) ![][]math.Complex {
    const rho1_val_gs = try alloc.alloc([]math.Complex, dim);
    errdefer alloc.free(rho1_val_gs);

    for (0..dim) |i| {
        rho1_val_gs[i] = pert_results[i].rho1_g;
    }
    return rho1_val_gs;
}

fn fill_electronic_dynmat(
    dyn_q: []math.Complex,
    dim: usize,
    vloc1_gs: []const []math.Complex,
    pert_results: []PerturbationResult,
    volume: f64,
) void {
    for (0..dim) |i| {
        for (0..dim) |j| {
            dyn_q[i * dim + j] = compute_elec_dynmat_element_q(
                vloc1_gs[i],
                pert_results[j].rho1_g,
                volume,
            );
        }
    }
}

fn fill_irreducible_electronic_dynmat(
    dyn_q: []math.Complex,
    dim: usize,
    vloc1_gs: []const []math.Complex,
    pert_results: []MultiKPertResult,
    volume: f64,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) void {
    for (0..dim) |i| {
        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            dyn_q[i * dim + j] = compute_elec_dynmat_element_q(
                vloc1_gs[i],
                pert_results[j].rho1_g,
                volume,
            );
        }
    }
}

fn add_single_k_nlcc_cross_contribution(
    alloc: std.mem.Allocator,
    dyn_q: []math.Complex,
    dim: usize,
    grid: Grid,
    gs: GroundState,
    pert_results: []PerturbationResult,
    rho1_core_gs: []const []math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) !void {
    if (gs.rho_core == null) return;

    const rho1_val_gs = try collect_rho1_grids(alloc, pert_results, dim);
    defer alloc.free(rho1_val_gs);

    const nlcc_cross = try compute_nlcc_cross_dynmat_q(
        alloc,
        grid,
        gs,
        rho1_val_gs,
        rho1_core_gs,
        n_atoms,
        irr_info,
    );
    defer alloc.free(nlcc_cross);

    log_dfpt(
        "dfptQ_dyn: D_nlcc_cross(0x,0x)=({e:.6},{e:.6})\n",
        .{ nlcc_cross[0].r, nlcc_cross[0].i },
    );
    log_dfpt(
        "dfptQ_dyn: D_nlcc_cross(1x,1x)=({e:.6},{e:.6})" ++
            " D_nlcc_cross(1x,0x)=({e:.6},{e:.6})\n",
        .{
            nlcc_cross[3 * dim + 3].r,
            nlcc_cross[3 * dim + 3].i,
            nlcc_cross[3 * dim].r,
            nlcc_cross[3 * dim].i,
        },
    );
    add_complex_dynmat(dyn_q, nlcc_cross);
}

fn add_multi_k_nlcc_cross_contribution(
    alloc: std.mem.Allocator,
    dyn_q: []math.Complex,
    dim: usize,
    grid: Grid,
    gs: GroundState,
    pert_results: []MultiKPertResult,
    rho1_core_gs: []const []math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    rho_core: ?[]const f64,
) !void {
    if (rho_core == null) return;

    const rho1_val_gs = try collect_rho1_grids(alloc, pert_results, dim);
    defer alloc.free(rho1_val_gs);

    const nlcc_cross = try compute_nlcc_cross_dynmat_q(
        alloc,
        grid,
        gs,
        rho1_val_gs,
        rho1_core_gs,
        n_atoms,
        irr_info,
    );
    defer alloc.free(nlcc_cross);

    log_dfpt(
        "dfptQ_mk_dyn: D_nlcc_cross(0x,0x)=({e:.6},{e:.6})\n",
        .{ nlcc_cross[0].r, nlcc_cross[0].i },
    );
    add_complex_dynmat(dyn_q, nlcc_cross);
}

fn add_single_kq_independent_terms(
    alloc: std.mem.Allocator,
    dyn_q: []math.Complex,
    dim: usize,
    grid: Grid,
    gs: GroundState,
    rho0_g: []const math.Complex,
    charges: []const f64,
    positions: []const math.Vec3,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    q_cart: math.Vec3,
    vxc_g: ?[]const math.Complex,
    n_atoms: usize,
) !void {
    const ewald_dyn_q = try ewald2.ewald_dynmat_q(
        alloc,
        cell_bohr,
        recip,
        charges,
        positions,
        q_cart,
    );
    defer alloc.free(ewald_dyn_q);

    log_dfpt(
        "dfptQ_dyn: D_ewald(0x,0x)=({e:.6},{e:.6}) D_ewald(0x,1x)=({e:.6},{e:.6}) [Ha]\n",
        .{ ewald_dyn_q[0].r, ewald_dyn_q[0].i, ewald_dyn_q[3].r, ewald_dyn_q[3].i },
    );
    log_dfpt(
        "dfptQ_dyn: D_ewald(0x,0y)=({e:.6},{e:.6}) D_ewald(0y,0y)=({e:.6},{e:.6}) [Ha]\n",
        .{
            ewald_dyn_q[1].r,
            ewald_dyn_q[1].i,
            ewald_dyn_q[dim + 1].r,
            ewald_dyn_q[dim + 1].i,
        },
    );
    log_dfpt("dfptQ_dyn: D_ewald full [Ha]:\n", .{});
    for (0..dim) |row| {
        for (0..dim) |col| {
            log_dfpt(
                "  ({e:.6},{e:.6})",
                .{ ewald_dyn_q[row * dim + col].r, ewald_dyn_q[row * dim + col].i },
            );
        }
        log_dfpt("\n", .{});
    }
    for (dyn_q, ewald_dyn_q) |*value, addend| {
        value.* = math.complex.add(value.*, math.complex.scale(addend, 2.0));
    }

    const self_dyn_real = try dynmat_contrib.compute_self_energy_dynmat(
        alloc,
        grid,
        gs.species,
        gs.atoms,
        rho0_g,
        gs.local_cfg,
        gs.ff_tables,
    );
    defer alloc.free(self_dyn_real);

    log_dfpt(
        "dfptQ_dyn: D_self(0x,0x)={e:.6} D_self(0x,1x)={e:.6}\n",
        .{ self_dyn_real[0], self_dyn_real[3] },
    );
    add_real_dynmat(dyn_q, self_dyn_real);

    const nl_self_real = try dynmat_contrib.compute_nonlocal_self_energy_dynmat(alloc, gs, n_atoms);
    defer alloc.free(nl_self_real);

    log_dfpt(
        "dfptQ_dyn: D_nl_self(0x,0x)={e:.6} D_nl_self(0x,1x)={e:.6}\n",
        .{ nl_self_real[0], nl_self_real[3] },
    );
    add_real_dynmat(dyn_q, nl_self_real);

    if (gs.rho_core == null) return;
    if (vxc_g) |vg| {
        const nlcc_self_real = try dynmat_contrib.compute_nlcc_self_dynmat(
            alloc,
            grid,
            gs.species,
            gs.atoms,
            vg,
            gs.rho_core_tables,
        );
        defer alloc.free(nlcc_self_real);

        log_dfpt("dfptQ_dyn: D_nlcc_self(0x,0x)={e:.6}\n", .{nlcc_self_real[0]});
        add_real_dynmat(dyn_q, nlcc_self_real);
    }
}

fn add_multi_kq_independent_terms(
    alloc: std.mem.Allocator,
    dyn_q: []math.Complex,
    kpts: []KPointDfptData,
    rho0_g: []const math.Complex,
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
    n_atoms: usize,
) !void {
    const dim = 3 * n_atoms;

    const ewald_dyn_q = try ewald2.ewald_dynmat_q(
        alloc,
        cell_bohr,
        recip,
        charges,
        positions,
        q_cart,
    );
    defer alloc.free(ewald_dyn_q);

    log_dfpt(
        "dfptQ_mk_dyn: D_ewald(0x,0x)=({e:.6},{e:.6}) [Ha]\n",
        .{ ewald_dyn_q[0].r, ewald_dyn_q[0].i },
    );
    for (dyn_q, ewald_dyn_q) |*value, addend| {
        value.* = math.complex.add(value.*, math.complex.scale(addend, 2.0));
    }

    const self_dyn_real = try dynmat_contrib.compute_self_energy_dynmat(
        alloc,
        grid,
        species,
        atoms,
        rho0_g,
        gs.local_cfg,
        ff_tables,
    );
    defer alloc.free(self_dyn_real);

    log_dfpt("dfptQ_mk_dyn: D_self(0x,0x)={e:.6}\n", .{self_dyn_real[0]});
    add_real_dynmat(dyn_q, self_dyn_real);

    const nl_self_real = try compute_nonlocal_self_dynmat_multi_k(
        alloc,
        kpts,
        atoms,
        n_atoms,
        volume,
    );
    defer alloc.free(nl_self_real);

    log_dfpt("dfptQ_mk_dyn: D_nl_self(0x,0x)={e:.6}\n", .{nl_self_real[0]});
    add_real_dynmat(dyn_q, nl_self_real);

    if (rho_core != null) {
        if (vxc_g) |vg| {
            const nlcc_self_real = try dynmat_contrib.compute_nlcc_self_dynmat(
                alloc,
                grid,
                species,
                atoms,
                vg,
                rho_core_tables,
            );
            defer alloc.free(nlcc_self_real);

            log_dfpt("dfptQ_mk_dyn: D_nlcc_self(0x,0x)={e:.6}\n", .{nlcc_self_real[0]});
            add_real_dynmat(dyn_q, nlcc_self_real);
        }
    }

    if (!vdw_cfg.enabled) return;

    const atomic_numbers = try alloc.alloc(usize, n_atoms);
    defer alloc.free(atomic_numbers);

    for (atoms, 0..) |atom, i| {
        atomic_numbers[i] = d3_params.atomic_number(species[atom.species_index].symbol) orelse 0;
    }
    var damping_params = d3_params.pbe_d3bj;
    if (vdw_cfg.s6) |v| damping_params.s6 = v;
    if (vdw_cfg.s8) |v| damping_params.s8 = v;
    if (vdw_cfg.a1) |v| damping_params.a1 = v;
    if (vdw_cfg.a2) |v| damping_params.a2 = v;

    const d3_dyn_q = try d3.compute_dynmat_q(
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

    log_dfpt("dfptQ_mk_dyn: D_d3(0x,0x)=({e:.6},{e:.6})\n", .{ d3_dyn_q[0].r, d3_dyn_q[0].i });
    add_complex_dynmat(dyn_q, d3_dyn_q);
    _ = dim;
}

fn fill_phase_factors(
    phase: []math.Complex,
    gvecs: []const plane_wave.GVector,
    atom_position: math.Vec3,
) void {
    for (0..phase.len) |g| {
        phase[g] = math.complex.expi(math.Vec3.dot(gvecs[g].cart, atom_position));
    }
}

fn project_nonlocal_self_entry(
    entry: anytype,
    phase: []const math.Complex,
    gvecs: []const plane_wave.GVector,
    psi_n: []const math.Complex,
    proj_std: []math.Complex,
    proj_alpha: [][3]math.Complex,
    proj_alpha_beta: [][3][3]math.Complex,
) void {
    var b: usize = 0;
    while (b < entry.beta_count) : (b += 1) {
        const offset = entry.m_offsets[b];
        const m_count = entry.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const phi_start = (offset + m_idx) * entry.g_count;
            const phi_end = (offset + m_idx + 1) * entry.g_count;
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

            for (0..phase.len) |g| {
                const gvec = gvecs[g].cart;
                const gc = [3]f64{ gvec.x, gvec.y, gvec.z };
                const base = math.complex.scale(math.complex.mul(phase[g], psi_n[g]), phi[g]);
                p_std = math.complex.add(p_std, base);
                for (0..3) |a| {
                    const weighted = math.complex.scale(base, gc[a]);
                    p_a[a] = math.complex.add(p_a[a], math.complex.init(-weighted.i, weighted.r));
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
}

fn accumulate_nonlocal_self_entry(
    dyn: []f64,
    dim: usize,
    atom_idx: usize,
    entry: anytype,
    proj_std: []const math.Complex,
    proj_alpha: []const [3]math.Complex,
    proj_alpha_beta: []const [3][3]math.Complex,
    weight_scale: f64,
) void {
    var b: usize = 0;
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
                        const i_idx = 3 * atom_idx + alpha;
                        const j_idx = 3 * atom_idx + beta;
                        dyn[i_idx * dim + j_idx] += weight_scale * dij * (t1.r + t2.r);
                    }
                }
            }
        }
    }
}

/// Build the full complex dynamical matrix for a finite q-point.
/// Combines electronic, nonlocal, NLCC, Ewald, and self-energy contributions.
/// All (I,J) pairs are computed explicitly; no Hermitianization needed.
pub fn build_q_dynmat(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    vloc1_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []const math.Complex,
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
    const dyn_q = try alloc_dynmat(alloc, dim);
    errdefer alloc.free(dyn_q);
    fill_electronic_dynmat(dyn_q, dim, vloc1_gs, pert_results, volume);
    log_single_k_dynmat_electronic_samples(dyn_q, dim);
    const nl_resp_q = try compute_nonlocal_response_dynmat_q(
        alloc,
        gs,
        pert_results,
        gvecs_kq,
        apply_ctx_kq,
        n_atoms,
    );
    defer alloc.free(nl_resp_q);

    log_single_k_dynmat_nonlocal_samples(dyn_q, nl_resp_q, dim);
    add_complex_dynmat(dyn_q, nl_resp_q);
    try add_single_k_remaining_dynmat_terms(
        alloc,
        dyn_q,
        dim,
        gs,
        pert_results,
        rho1_core_gs,
        rho0_g,
        charges,
        positions,
        cell_bohr,
        recip,
        q_cart,
        vxc_g,
        n_atoms,
        irr_info,
    );
    log_single_k_dynmat_total_samples(dyn_q);

    return dyn_q;
}

fn log_single_k_dynmat_electronic_samples(dyn_q: []const math.Complex, dim: usize) void {
    _ = dim;
    log_dfpt(
        "dfptQ_dyn: D_elec(0x,0x)=({e:.6},{e:.6}) D_elec(0x,1x)=({e:.6},{e:.6})\n",
        .{ dyn_q[0].r, dyn_q[0].i, dyn_q[3].r, dyn_q[3].i },
    );
}

fn log_single_k_dynmat_nonlocal_samples(
    dyn_q: []const math.Complex,
    nl_resp_q: []const math.Complex,
    dim: usize,
) void {
    log_dfpt(
        "dfptQ_dyn: D_nl_resp(0x,0x)=({e:.6},{e:.6}) D_nl_resp(0x,1x)=({e:.6},{e:.6})\n",
        .{ nl_resp_q[0].r, nl_resp_q[0].i, nl_resp_q[3].r, nl_resp_q[3].i },
    );
    log_dfpt(
        "dfptQ_dyn: D_nl_resp(1x,0x)=({e:.6},{e:.6}) D_nl_resp(1x,1x)=({e:.6},{e:.6})\n",
        .{
            nl_resp_q[3 * dim].r,
            nl_resp_q[3 * dim].i,
            nl_resp_q[3 * dim + 3].r,
            nl_resp_q[3 * dim + 3].i,
        },
    );
    log_dfpt(
        "dfptQ_dyn: D_elec(1x,0x)=({e:.6},{e:.6}) D_elec(1x,1x)=({e:.6},{e:.6})\n",
        .{ dyn_q[3 * dim].r, dyn_q[3 * dim].i, dyn_q[3 * dim + 3].r, dyn_q[3 * dim + 3].i },
    );
}

fn add_single_k_remaining_dynmat_terms(
    alloc: std.mem.Allocator,
    dyn_q: []math.Complex,
    dim: usize,
    gs: GroundState,
    pert_results: []PerturbationResult,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []const math.Complex,
    charges: []const f64,
    positions: []const math.Vec3,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    q_cart: math.Vec3,
    vxc_g: ?[]const math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) !void {
    try add_single_k_nlcc_cross_contribution(
        alloc,
        dyn_q,
        dim,
        gs.grid,
        gs,
        pert_results,
        rho1_core_gs,
        n_atoms,
        irr_info,
    );
    try add_single_kq_independent_terms(
        alloc,
        dyn_q,
        dim,
        gs.grid,
        gs,
        rho0_g,
        charges,
        positions,
        cell_bohr,
        recip,
        q_cart,
        vxc_g,
        n_atoms,
    );
}

fn log_single_k_dynmat_total_samples(dyn_q: []const math.Complex) void {
    log_dfpt(
        "dfptQ_dyn: total(0x,0x)=({e:.6},{e:.6}) total(0x,1x)=({e:.6},{e:.6})" ++
            " total(0x,1y)=({e:.6},{e:.6})\n",
        .{ dyn_q[0].r, dyn_q[0].i, dyn_q[3].r, dyn_q[3].i, dyn_q[4].r, dyn_q[4].i },
    );
}

/// Compute nonlocal response dynmat D_nl_resp summed over all k-points.
/// D(Iα,Jβ) = Σ_k wtk × 4 × Σ_n ⟨dV_nl_{Iα,q} ψ^(0)_{n,k} | δψ_{n,k,Jβ}⟩
pub fn compute_nonlocal_response_dynmat_q_multi_k(
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
                try perturbation.apply_nonlocal_perturbation_q(
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
pub fn compute_nonlocal_self_dynmat_multi_k(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    atoms: []const hamiltonian.AtomData,
    n_atoms: usize,
    volume: f64,
) ![]f64 {
    const dim = 3 * n_atoms;
    const dyn = try alloc.alloc(f64, dim * dim);
    errdefer alloc.free(dyn);
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
                    fill_phase_factors(phase, kd.basis_k.gvecs, atom.position);
                    project_nonlocal_self_entry(
                        entry,
                        phase,
                        kd.basis_k.gvecs,
                        kd.wavefunctions_k_const[n],
                        proj_std,
                        proj_alpha,
                        proj_alpha_beta,
                    );
                    accumulate_nonlocal_self_entry(
                        dyn,
                        dim,
                        atom_idx,
                        entry,
                        proj_std,
                        proj_alpha,
                        proj_alpha_beta,
                        4.0 * inv_volume * kd.weight,
                    );
                }
            }
        }
    }

    return dyn;
}

/// Build the full complex dynamical matrix for a finite q-point with multiple k-points.
/// Combines electronic, nonlocal, NLCC, Ewald, and self-energy contributions.
pub fn build_q_dynmat_multi_k(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    pert_results: []MultiKPertResult,
    vloc1_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []const math.Complex,
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
    const dim = 3 * atoms.len;
    const dyn_q = try alloc_dynmat(alloc, dim);
    errdefer alloc.free(dyn_q);
    fill_irreducible_electronic_dynmat(dyn_q, dim, vloc1_gs, pert_results, volume, irr_info);
    log_dfpt("dfptQ_mk_dyn: D_elec(0x,0x)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i });
    const nl_resp_q = try compute_multi_k_nl_resp(
        alloc,
        kpts,
        pert_results,
        atoms,
        volume,
        irr_info,
    );
    defer alloc.free(nl_resp_q);

    log_multi_k_dynmat_nl_sample(nl_resp_q);
    add_complex_dynmat(dyn_q, nl_resp_q);
    try add_multi_k_remaining_dynmat_terms(
        alloc,
        dyn_q,
        dim,
        kpts,
        rho0_g,
        gs,
        pert_results,
        rho1_core_gs,
        charges,
        positions,
        cell_bohr,
        recip,
        volume,
        q_cart,
        grid,
        species,
        atoms,
        ff_tables,
        rho_core_tables,
        rho_core,
        vxc_g,
        vdw_cfg,
        atoms.len,
        irr_info,
    );
    log_dfpt("dfptQ_mk_dyn: total(0x,0x)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i });
    return dyn_q;
}

fn compute_multi_k_nl_resp(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    pert_results: []MultiKPertResult,
    atoms: []const hamiltonian.AtomData,
    volume: f64,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    return try compute_nonlocal_response_dynmat_q_multi_k(
        alloc,
        kpts,
        pert_results,
        atoms,
        atoms.len,
        volume,
        irr_info,
    );
}

fn log_multi_k_dynmat_nl_sample(nl_resp_q: []const math.Complex) void {
    log_dfpt(
        "dfptQ_mk_dyn: D_nl_resp(0x,0x)=({e:.6},{e:.6})\n",
        .{ nl_resp_q[0].r, nl_resp_q[0].i },
    );
}

fn add_multi_k_remaining_dynmat_terms(
    alloc: std.mem.Allocator,
    dyn_q: []math.Complex,
    dim: usize,
    kpts: []KPointDfptData,
    rho0_g: []const math.Complex,
    gs: GroundState,
    pert_results: []MultiKPertResult,
    rho1_core_gs: []const []math.Complex,
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
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) !void {
    try add_multi_k_nlcc_cross_contribution(
        alloc,
        dyn_q,
        dim,
        grid,
        gs,
        pert_results,
        rho1_core_gs,
        n_atoms,
        irr_info,
        rho_core,
    );
    try add_multi_kq_independent_terms(
        alloc,
        dyn_q,
        kpts,
        rho0_g,
        gs,
        charges,
        positions,
        cell_bohr,
        recip,
        volume,
        q_cart,
        grid,
        species,
        atoms,
        ff_tables,
        rho_core_tables,
        rho_core,
        vxc_g,
        vdw_cfg,
        n_atoms,
    );
}
