const std = @import("std");
const apply = @import("apply.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const gvec_iter = @import("gvec_iter.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_data = @import("../pseudopotential/paw_data.zig");
const paw_mod = @import("../paw/paw.zig");
const xc = @import("../xc/xc.zig");

const Grid = grid_mod.Grid;

const reciprocal_to_real = fft_grid.reciprocal_to_real;
const MAX_YLM_AUG: usize = 25;
const MAX_PAW_BETA: usize = 32;

const SpeciesPawSetup = struct {
    si: usize,
    tab: *const paw_mod.PawTab,
    paw: paw_data.PawData,
    dij0: []const f64,
    r: []const f64,
    rab: []const f64,
    rho_core_ps: ?[]const f64,
    nb: usize,
    n_ij: usize,
    mt: usize,
    n_m: usize,
    natom: usize,
    sp_m_offsets: [MAX_PAW_BETA]usize,
};

/// Symmetrize PAW rhoij by averaging over equivalent atoms of the same species.
/// In a bulk crystal, all atoms of the same species share the same Wyckoff position
/// and their on-site occupation matrices must be identical by symmetry.
pub fn symmetrize_rho_ij(
    alloc: std.mem.Allocator,
    rhoij: *paw_mod.RhoIJ,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) !void {
    for (species, 0..) |_, si| {
        // Find atoms of this species
        var count: usize = 0;
        var n_ij: usize = 0;
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index == si) {
                n_ij = rhoij.values[ai].len;
                count += 1;
            }
        }
        if (count <= 1 or n_ij == 0) continue;

        // Average rhoij over all atoms of this species
        const avg = try alloc.alloc(f64, n_ij);
        defer alloc.free(avg);

        @memset(avg, 0.0);
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index != si) continue;
            for (0..n_ij) |idx| {
                avg[idx] += rhoij.values[ai][idx];
            }
        }
        const inv_count = 1.0 / @as(f64, @floatFromInt(count));
        for (0..n_ij) |idx| {
            avg[idx] *= inv_count;
        }
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index != si) continue;
            @memcpy(rhoij.values[ai], avg);
        }
    }
}

fn init_species_paw_setup(
    entry_s: hamiltonian.SpeciesEntry,
    si: usize,
    tab: *const paw_mod.PawTab,
    atoms: []const hamiltonian.AtomData,
) ?SpeciesPawSetup {
    const paw = entry_s.upf.paw orelse return null;
    var sp_m_offsets: [MAX_PAW_BETA]usize = undefined;
    var mt: usize = 0;
    var offset: usize = 0;
    for (0..tab.nbeta) |b| {
        sp_m_offsets[b] = offset;
        const m_count = @as(usize, @intCast(2 * tab.l_list[b] + 1));
        mt += m_count;
        offset += m_count;
    }

    var natom: usize = 0;
    for (atoms) |atom| {
        if (atom.species_index == si) natom += 1;
    }
    if (natom == 0) return null;

    return .{
        .si = si,
        .tab = tab,
        .paw = paw,
        .dij0 = entry_s.upf.dij,
        .r = entry_s.upf.r,
        .rab = entry_s.upf.rab,
        .rho_core_ps = if (entry_s.upf.nlcc.len > 0) entry_s.upf.nlcc else null,
        .nb = tab.nbeta,
        .n_ij = tab.nbeta * tab.nbeta,
        .mt = mt,
        .n_m = mt * mt,
        .natom = natom,
        .sp_m_offsets = sp_m_offsets,
    };
}

fn ensure_species_dij_storage(
    apply_caches: []apply.KpointApplyCache,
    setup: SpeciesPawSetup,
) !void {
    for (apply_caches) |*ac| {
        if (ac.nonlocal_ctx) |*nl| {
            try nl.ensure_dij_per_atom(ac.cache_alloc, setup.si, setup.natom);
            try nl.ensure_dij_m_per_atom(ac.cache_alloc, setup.si, setup.natom);
        }
    }
}

fn fill_aug_ylm(
    gaunt_table: *const paw_mod.GauntTable,
    gvec: math.Vec3,
    g_abs: f64,
    ylm_g: *[MAX_YLM_AUG]f64,
) void {
    if (g_abs <= 1e-10) {
        @memset(ylm_g, 0.0);
        ylm_g[0] = 1.0 / @sqrt(4.0 * std.math.pi);
        return;
    }

    for (0..gaunt_table.lmax_aug + 1) |big_l| {
        const bl_i32: i32 = @intCast(big_l);
        var bm: i32 = -bl_i32;
        while (bm <= bl_i32) : (bm += 1) {
            ylm_g[paw_mod.GauntTable.lm_index(big_l, bm)] =
                nonlocal_mod.real_spherical_harmonic(bl_i32, bm, gvec.x, gvec.y, gvec.z);
        }
    }
}

fn accumulate_gaunt_projected_rhoij(
    gaunt_table: *const paw_mod.GauntTable,
    rij_m: []const f64,
    mt: usize,
    m_offsets: []const usize,
    l_i: usize,
    l_j: usize,
    i_beta: usize,
    j_beta: usize,
    big_l: usize,
    bm: i32,
) f64 {
    var value: f64 = 0.0;
    const li_i32: i32 = @intCast(l_i);
    const lj_i32: i32 = @intCast(l_j);
    var mi: i32 = -li_i32;
    while (mi <= li_i32) : (mi += 1) {
        const mi_idx = m_offsets[i_beta] + @as(usize, @intCast(mi + li_i32));
        var mj: i32 = -lj_i32;
        while (mj <= lj_i32) : (mj += 1) {
            const g_coeff = gaunt_table.get(l_i, mi, l_j, mj, big_l, bm);
            if (g_coeff == 0.0) continue;
            const mj_idx = m_offsets[j_beta] + @as(usize, @intCast(mj + lj_i32));
            value += g_coeff * rij_m[mi_idx * mt + mj_idx];
        }
    }
    return value;
}

fn compensation_charge_for_atom(
    rhoij: *const paw_mod.RhoIJ,
    tab: *const paw_mod.PawTab,
    atom: hamiltonian.AtomData,
    ai: usize,
    gaunt_table: *const paw_mod.GauntTable,
    gvec: math.Vec3,
    g_abs: f64,
    ylm_g: *const [MAX_YLM_AUG]f64,
    inv_omega: f64,
) math.Complex {
    const mt = rhoij.m_total_per_atom[ai];
    const rij_m = rhoij.values[ai];
    const l_list_r = rhoij.l_per_beta[ai];
    const m_offsets_r = rhoij.m_offsets[ai];
    const g_dot_r = math.Vec3.dot(gvec, atom.position);
    const sf_re = @cos(g_dot_r);
    const sf_im = -@sin(g_dot_r);
    var sum_re: f64 = 0.0;
    var sum_im: f64 = 0.0;

    for (0..tab.n_qijl_entries) |e| {
        const qidx = tab.qijl_indices[e];
        const qijl_g = tab.eval_qijl_form(e, g_abs);
        if (@abs(qijl_g) < 1e-30) continue;

        const l_i = @as(usize, @intCast(l_list_r[qidx.first]));
        const l_j = @as(usize, @intCast(l_list_r[qidx.second]));
        const bl_i32: i32 = @intCast(qidx.l);
        var bm: i32 = -bl_i32;
        while (bm <= bl_i32) : (bm += 1) {
            const ylm_val = ylm_g[paw_mod.GauntTable.lm_index(qidx.l, bm)];
            if (@abs(ylm_val) < 1e-30) continue;
            const gaunt_rhoij = accumulate_gaunt_projected_rhoij(
                gaunt_table,
                rij_m,
                mt,
                m_offsets_r,
                l_i,
                l_j,
                qidx.first,
                qidx.second,
                qidx.l,
                bm,
            );
            if (@abs(gaunt_rhoij) < 1e-30) continue;
            const sym_factor: f64 = if (qidx.first != qidx.second) 2.0 else 1.0;
            const contrib = gaunt_rhoij * qijl_g * ylm_val * sym_factor * inv_omega;
            sum_re += contrib * sf_re;
            sum_im += contrib * sf_im;
        }
    }
    return math.complex.init(sum_re, sum_im);
}

fn initialize_paw_dij_buffers(
    setup: SpeciesPawSetup,
    dij: []f64,
    dij_m: []f64,
) void {
    if (setup.dij0.len >= setup.n_ij) {
        @memcpy(dij, setup.dij0[0..setup.n_ij]);
    } else {
        @memset(dij, 0.0);
    }
    @memset(dij_m, 0.0);

    for (0..setup.nb) |i| {
        for (0..setup.nb) |j| {
            if (setup.tab.l_list[i] != setup.tab.l_list[j]) continue;
            const d0 = if (setup.dij0.len >= setup.n_ij) setup.dij0[i * setup.nb + j] else 0.0;
            const m_count = @as(usize, @intCast(2 * setup.tab.l_list[i] + 1));
            for (0..m_count) |m| {
                const im = setup.sp_m_offsets[i] + m;
                const jm = setup.sp_m_offsets[j] + m;
                dij_m[im * setup.mt + jm] = d0;
            }
        }
    }
}

fn accumulate_dhat_m_resolved_entry(
    setup: SpeciesPawSetup,
    gaunt_table: *const paw_mod.GauntTable,
    ylm_g: *const [MAX_YLM_AUG]f64,
    big_l: usize,
    i_beta: usize,
    j_beta: usize,
    qijl_g: f64,
    prod_re: f64,
    dij_m: []f64,
) void {
    const l_i = @as(usize, @intCast(setup.tab.l_list[i_beta]));
    const l_j = @as(usize, @intCast(setup.tab.l_list[j_beta]));
    const bl_i32: i32 = @intCast(big_l);
    var bm: i32 = -bl_i32;
    while (bm <= bl_i32) : (bm += 1) {
        const ylm_val = ylm_g[paw_mod.GauntTable.lm_index(big_l, bm)];
        if (@abs(ylm_val) < 1e-30) continue;
        const li_i32: i32 = @intCast(l_i);
        const lj_i32: i32 = @intCast(l_j);
        var mi: i32 = -li_i32;
        while (mi <= li_i32) : (mi += 1) {
            const im = setup.sp_m_offsets[i_beta] + @as(usize, @intCast(mi + li_i32));
            var mj: i32 = -lj_i32;
            while (mj <= lj_i32) : (mj += 1) {
                const g_coeff = gaunt_table.get(l_i, mi, l_j, mj, big_l, bm);
                if (g_coeff == 0.0) continue;
                const jm = setup.sp_m_offsets[j_beta] + @as(usize, @intCast(mj + lj_i32));
                const contrib = prod_re * qijl_g * ylm_val * g_coeff;
                dij_m[im * setup.mt + jm] += contrib;
                if (i_beta != j_beta) dij_m[jm * setup.mt + im] += contrib;
            }
        }
    }
}

fn accumulate_dhat_for_g(
    setup: SpeciesPawSetup,
    gaunt_table: *const paw_mod.GauntTable,
    ylm_g: *const [MAX_YLM_AUG]f64,
    g_abs: f64,
    prod_re: f64,
    dij: []f64,
    dij_m: []f64,
) void {
    for (0..setup.tab.n_qijl_entries) |e| {
        const qidx = setup.tab.qijl_indices[e];
        const qijl_g = setup.tab.eval_qijl_form(e, g_abs);
        if (@abs(qijl_g) < 1e-30) continue;

        if (qidx.l == 0) {
            const ylm_00 = 1.0 / @sqrt(4.0 * std.math.pi);
            const gaunt_00 = 1.0 / @sqrt(4.0 * std.math.pi);
            const contrib = prod_re * qijl_g * ylm_00 * gaunt_00;
            dij[qidx.first * setup.nb + qidx.second] += contrib;
            if (qidx.first != qidx.second) {
                dij[qidx.second * setup.nb + qidx.first] += contrib;
            }
        }
        accumulate_dhat_m_resolved_entry(
            setup,
            gaunt_table,
            ylm_g,
            qidx.l,
            qidx.first,
            qidx.second,
            qijl_g,
            prod_re,
            dij_m,
        );
    }
}

fn add_paw_dhat_for_atom(
    grid: Grid,
    ionic: hamiltonian.PotentialGrid,
    potential: hamiltonian.PotentialGrid,
    setup: SpeciesPawSetup,
    pos: math.Vec3,
    ecutrho: f64,
    gaunt_table: *const paw_mod.GauntTable,
    dij: []f64,
    dij_m: []f64,
) void {
    const total = grid.count();
    var git = gvec_iter.GVecIterator.init(grid);
    while (git.next()) |g| {
        if (g.g2 >= ecutrho) continue;
        const v_hxc = if (g.idx < total) potential.values[g.idx] else math.complex.init(0.0, 0.0);
        const v_loc = if (g.idx < total) ionic.values[g.idx] else math.complex.init(0.0, 0.0);
        const v_eff = math.complex.add(v_hxc, v_loc);
        const g_dot_r = math.Vec3.dot(g.gvec, pos);
        const prod_re = v_eff.r * @cos(g_dot_r) - v_eff.i * @sin(g_dot_r);
        var ylm_g: [MAX_YLM_AUG]f64 = undefined;
        fill_aug_ylm(gaunt_table, g.gvec, @sqrt(g.g2), &ylm_g);
        accumulate_dhat_for_g(setup, gaunt_table, &ylm_g, @sqrt(g.g2), prod_re, dij, dij_m);
    }
}

fn contract_m_resolved_diagonal_to_radial(
    setup: SpeciesPawSetup,
    correction_m: []const f64,
    dij: []f64,
) void {
    for (0..setup.nb) |i| {
        for (0..setup.nb) |j| {
            if (setup.tab.l_list[i] != setup.tab.l_list[j]) continue;
            const m_count = @as(usize, @intCast(2 * setup.tab.l_list[i] + 1));
            var sum: f64 = 0.0;
            for (0..m_count) |m| {
                const im = setup.sp_m_offsets[i] + m;
                const jm = setup.sp_m_offsets[j] + m;
                sum += correction_m[im * setup.mt + jm];
            }
            dij[i * setup.nb + j] += sum / @as(f64, @floatFromInt(m_count));
        }
    }
}

fn add_m_resolved_correction(
    setup: SpeciesPawSetup,
    correction_m: []const f64,
    dij: []f64,
    dij_m: []f64,
) void {
    for (0..setup.n_m) |idx| {
        dij_m[idx] += correction_m[idx];
    }
    contract_m_resolved_diagonal_to_radial(setup, correction_m, dij);
}

fn fill_paw_dij_xc_matrix(
    alloc: std.mem.Allocator,
    setup: SpeciesPawSetup,
    rhoij: *const paw_mod.RhoIJ,
    ai: usize,
    rhoij_spin: ?*const paw_mod.RhoIJ,
    xc_func: xc.Functional,
    gaunt_table: *const paw_mod.GauntTable,
    dij_xc_m: []f64,
) !void {
    if (rhoij_spin) |rij_s| {
        const rij_other = try alloc.alloc(f64, setup.n_m);
        defer alloc.free(rij_other);

        for (0..setup.n_m) |idx| {
            rij_other[idx] = rhoij.values[ai][idx] - rij_s.values[ai][idx];
        }
        const dij_xc_other = try alloc.alloc(f64, setup.n_m);
        defer alloc.free(dij_xc_other);

        try paw_mod.paw_xc.compute_paw_dij_xc_angular_spin(
            alloc,
            dij_xc_m,
            dij_xc_other,
            setup.paw,
            rij_s.values[ai],
            rij_other,
            rhoij.m_total_per_atom[ai],
            rhoij.m_offsets[ai],
            setup.r,
            setup.rab,
            setup.paw.ae_core_density,
            setup.rho_core_ps,
            xc_func,
            gaunt_table,
        );
        return;
    }

    try paw_mod.paw_xc.compute_paw_dij_xc_angular(
        alloc,
        dij_xc_m,
        setup.paw,
        rhoij.values[ai],
        rhoij.m_total_per_atom[ai],
        rhoij.m_offsets[ai],
        setup.r,
        setup.rab,
        setup.paw.ae_core_density,
        setup.rho_core_ps,
        xc_func,
        gaunt_table,
    );
}

fn fill_paw_dij_hartree_matrix(
    alloc: std.mem.Allocator,
    setup: SpeciesPawSetup,
    rhoij: *const paw_mod.RhoIJ,
    ai: usize,
    gaunt_table: *const paw_mod.GauntTable,
    dij_h_m: []f64,
) !void {
    try paw_mod.paw_xc.compute_paw_dij_hartree_multi_l(
        alloc,
        dij_h_m,
        setup.paw,
        rhoij.values[ai],
        rhoij.m_total_per_atom[ai],
        rhoij.m_offsets[ai],
        setup.r,
        setup.rab,
        gaunt_table,
    );
}

fn mix_and_write_paw_dij_atom(
    apply_caches: []apply.KpointApplyCache,
    si: usize,
    atom_counter: usize,
    dij_mix_beta: f64,
    dij: []f64,
    dij_m: []f64,
) void {
    if (dij_mix_beta < 1.0 - 1e-10 and apply_caches.len > 0) {
        if (apply_caches[0].nonlocal_ctx) |*nl| {
            if (nl.species[si].dij_per_atom) |dpa| {
                if (atom_counter < dpa.len) {
                    for (0..@min(dij.len, dpa[atom_counter].len)) |ii| {
                        dij[ii] = (1.0 - dij_mix_beta) * dpa[atom_counter][ii] +
                            dij_mix_beta * dij[ii];
                    }
                }
            }
            if (nl.species[si].dij_m_per_atom) |dpa| {
                if (atom_counter < dpa.len) {
                    for (0..@min(dij_m.len, dpa[atom_counter].len)) |ii| {
                        dij_m[ii] = (1.0 - dij_mix_beta) * dpa[atom_counter][ii] +
                            dij_mix_beta * dij_m[ii];
                    }
                }
            }
        }
    }

    for (apply_caches) |*ac| {
        if (ac.nonlocal_ctx) |*nl| {
            nl.update_dij_atom(si, atom_counter, dij);
            nl.update_dij_m_atom(si, atom_counter, dij_m);
        }
    }
}

fn symmetrize_species_radial_dij(
    alloc: std.mem.Allocator,
    apply_caches: []apply.KpointApplyCache,
    si: usize,
    natom: usize,
    n_ij: usize,
) !void {
    const avg_dij = try alloc.alloc(f64, n_ij);
    defer alloc.free(avg_dij);

    @memset(avg_dij, 0.0);

    if (apply_caches.len > 0) {
        if (apply_caches[0].nonlocal_ctx) |*nl| {
            if (nl.species[si].dij_per_atom) |dpa| {
                for (0..natom) |a| {
                    for (0..n_ij) |idx| avg_dij[idx] += dpa[a][idx];
                }
            }
        }
    }
    const inv_natom = 1.0 / @as(f64, @floatFromInt(natom));
    for (0..n_ij) |idx| avg_dij[idx] *= inv_natom;
    for (apply_caches) |*ac| {
        if (ac.nonlocal_ctx) |*nl| {
            if (nl.species[si].dij_per_atom) |dpa| {
                for (0..natom) |a| @memcpy(dpa[a], avg_dij);
            }
        }
    }
}

fn symmetrize_species_m_resolved_dij(
    alloc: std.mem.Allocator,
    apply_caches: []apply.KpointApplyCache,
    si: usize,
    natom: usize,
    n_m: usize,
) !void {
    const avg_dij_m = try alloc.alloc(f64, n_m);
    defer alloc.free(avg_dij_m);

    @memset(avg_dij_m, 0.0);

    if (apply_caches.len > 0) {
        if (apply_caches[0].nonlocal_ctx) |*nl| {
            if (nl.species[si].dij_m_per_atom) |dpa| {
                for (0..natom) |a| {
                    for (0..n_m) |idx| avg_dij_m[idx] += dpa[a][idx];
                }
            }
        }
    }
    const inv_natom = 1.0 / @as(f64, @floatFromInt(natom));
    for (0..n_m) |idx| avg_dij_m[idx] *= inv_natom;
    for (apply_caches) |*ac| {
        if (ac.nonlocal_ctx) |*nl| {
            if (nl.species[si].dij_m_per_atom) |dpa| {
                for (0..natom) |a| @memcpy(dpa[a], avg_dij_m);
            }
        }
    }
}

/// Add PAW compensation charge n_hat(r) to density array (multi-L with Gaunt coefficients).
///
/// n̂(G) = Σ_a Σ_{i,m_i,j,m_j} ρ_{(i,m_i),(j,m_j)}^a × Σ_{L,M} G(l_i,m_i,l_j,m_j,L,M)
///         × Q^L_{ij}(|G|) × Y_{L,M}(Ĝ) × exp(-iGR_a) / Ω
pub fn add_paw_compensation_charge(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    rhoij: *const paw_mod.RhoIJ,
    paw_tabs: []const paw_mod.PawTab,
    atoms: []const hamiltonian.AtomData,
    ecutrho: f64,
    gaunt_table: *const paw_mod.GauntTable,
) !void {
    const total = grid.count();
    const n_hat_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(n_hat_g);

    @memset(n_hat_g, math.complex.init(0.0, 0.0));

    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);
    const inv_omega = 1.0 / grid.volume;

    var idx: usize = 0;
    var il: usize = 0;
    while (il < grid.nz) : (il += 1) {
        var ik: usize = 0;
        while (ik < grid.ny) : (ik += 1) {
            var ih: usize = 0;
            while (ih < grid.nx) : (ih += 1) {
                const gh = grid.min_h + @as(i32, @intCast(ih));
                const gk = grid.min_k + @as(i32, @intCast(ik));
                const gl = grid.min_l + @as(i32, @intCast(il));
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_abs = math.Vec3.norm(gvec);
                const g2 = math.Vec3.dot(gvec, gvec);
                if (g2 >= ecutrho) {
                    idx += 1;
                    continue;
                }

                var ylm_g: [MAX_YLM_AUG]f64 = undefined;
                fill_aug_ylm(gaunt_table, gvec, g_abs, &ylm_g);
                for (atoms, 0..) |atom, ai| {
                    const sp = atom.species_index;
                    if (sp >= paw_tabs.len or paw_tabs[sp].nbeta == 0) continue;
                    const contrib = compensation_charge_for_atom(
                        rhoij,
                        &paw_tabs[sp],
                        atom,
                        ai,
                        gaunt_table,
                        gvec,
                        g_abs,
                        &ylm_g,
                        inv_omega,
                    );
                    n_hat_g[idx] = math.complex.add(n_hat_g[idx], contrib);
                }
                idx += 1;
            }
        }
    }

    // IFFT n_hat(G) → n_hat(r) and add to density
    const n_hat_r = try reciprocal_to_real(alloc, grid, n_hat_g);
    defer alloc.free(n_hat_r);

    for (0..@min(rho.len, n_hat_r.len)) |i| {
        rho[i] += n_hat_r[i];
    }
}

/// Update PAW D_ij per-atom from the total effective potential.
/// D_ij(atom) = D^0_ij + D^hat_ij(atom) + D^xc_ij(atom) + D^H_ij(atom)
/// D^hat depends on atom position via structure factor exp(-iG·R_a).
/// D^xc, D^H depend on atom-specific rhoij.
pub fn update_paw_dij(
    alloc: std.mem.Allocator,
    grid: Grid,
    ionic: hamiltonian.PotentialGrid,
    potential: hamiltonian.PotentialGrid,
    paw_tabs: []const paw_mod.PawTab,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    apply_caches: []apply.KpointApplyCache,
    ecutrho: f64,
    rhoij: *const paw_mod.RhoIJ,
    xc_func: xc.Functional,
    symmetrize: bool,
    gaunt_table: *const paw_mod.GauntTable,
    skip_dxc: bool,
    rhoij_spin: ?*const paw_mod.RhoIJ,
    dij_mix_beta: f64,
) !void {
    for (species, 0..) |entry_s, si| {
        if (si >= paw_tabs.len) continue;
        const setup = init_species_paw_setup(entry_s, si, &paw_tabs[si], atoms) orelse continue;
        try ensure_species_dij_storage(apply_caches, setup);

        var atom_counter: usize = 0;
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index != si) continue;
            const dij = try alloc.alloc(f64, setup.n_ij);
            defer alloc.free(dij);

            const dij_m = try alloc.alloc(f64, setup.n_m);
            defer alloc.free(dij_m);

            initialize_paw_dij_buffers(setup, dij, dij_m);
            add_paw_dhat_for_atom(
                grid,
                ionic,
                potential,
                setup,
                atom.position,
                ecutrho,
                gaunt_table,
                dij,
                dij_m,
            );

            const dij_xc_m = try alloc.alloc(f64, setup.n_m);
            defer alloc.free(dij_xc_m);

            if (skip_dxc) {
                @memset(dij_xc_m, 0.0);
            } else {
                try fill_paw_dij_xc_matrix(
                    alloc,
                    setup,
                    rhoij,
                    ai,
                    rhoij_spin,
                    xc_func,
                    gaunt_table,
                    dij_xc_m,
                );
            }
            add_m_resolved_correction(setup, dij_xc_m, dij, dij_m);

            const dij_h_m = try alloc.alloc(f64, setup.n_m);
            defer alloc.free(dij_h_m);

            try fill_paw_dij_hartree_matrix(alloc, setup, rhoij, ai, gaunt_table, dij_h_m);
            add_m_resolved_correction(setup, dij_h_m, dij, dij_m);

            mix_and_write_paw_dij_atom(
                apply_caches,
                setup.si,
                atom_counter,
                dij_mix_beta,
                dij,
                dij_m,
            );

            atom_counter += 1;
        }

        if (symmetrize and setup.natom > 1) {
            try symmetrize_species_radial_dij(
                alloc,
                apply_caches,
                setup.si,
                setup.natom,
                setup.n_ij,
            );
            try symmetrize_species_m_resolved_dij(
                alloc,
                apply_caches,
                setup.si,
                setup.natom,
                setup.n_m,
            );
        }
    }
}

/// Compute PAW on-site energy correction for all atoms.
/// When rhoij_up/rhoij_down are provided, uses spin-resolved E_xc.
pub fn compute_paw_onsite_energy_total(
    alloc: std.mem.Allocator,
    rhoij: *const paw_mod.RhoIJ,
    paw_tabs: []const paw_mod.PawTab,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    xc_func: xc.Functional,
    gaunt_table: *const paw_mod.GauntTable,
    rhoij_up: ?*const paw_mod.RhoIJ,
    rhoij_down: ?*const paw_mod.RhoIJ,
) !f64 {
    var e_paw: f64 = 0.0;
    for (0..atoms.len) |a| {
        const sp = atoms[a].species_index;
        if (sp >= paw_tabs.len or paw_tabs[sp].nbeta == 0) continue;
        const paw = species[sp].upf.paw orelse continue;
        const tab = &paw_tabs[sp];
        const upf = species[sp].upf.*;
        const rho_core_ps: ?[]const f64 = if (upf.nlcc.len > 0) upf.nlcc else null;

        if (rhoij_up != null and rhoij_down != null) {
            // Spin-resolved E_xc on-site
            const mt = rhoij.m_total_per_atom[a];
            const m_off = rhoij.m_offsets[a];
            const e_xc = try paw_mod.paw_xc.compute_paw_exc_onsite_angular_spin(
                alloc,
                paw,
                rhoij_up.?.values[a],
                rhoij_down.?.values[a],
                mt,
                m_off,
                upf.r,
                upf.rab,
                paw.ae_core_density,
                rho_core_ps,
                xc_func,
                gaunt_table,
            );
            const e_h = try paw_mod.paw_xc.compute_paw_eh_onsite_multi_l(
                alloc,
                paw,
                rhoij.values[a],
                mt,
                m_off,
                upf.r,
                upf.rab,
                gaunt_table,
            );
            e_paw += e_xc + e_h + paw.core_energy;
        } else {
            const e_atom = try paw_mod.paw_energy.compute_paw_onsite_energy(
                alloc,
                paw,
                tab,
                rhoij.values[a],
                rhoij.m_total_per_atom[a],
                rhoij.m_offsets[a],
                upf.r,
                upf.rab,
                rho_core_ps,
                xc_func,
                gaunt_table,
            );
            e_paw += e_atom;
        }
    }
    return e_paw;
}
