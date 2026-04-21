const std = @import("std");
const apply = @import("apply.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const gvec_iter = @import("gvec_iter.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const xc = @import("../xc/xc.zig");

const Grid = grid_mod.Grid;

const reciprocalToReal = fft_grid.reciprocalToReal;

/// Symmetrize PAW rhoij by averaging over equivalent atoms of the same species.
/// In a bulk crystal, all atoms of the same species share the same Wyckoff position
/// and their on-site occupation matrices must be identical by symmetry.
pub fn symmetrizeRhoIJ(
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

/// Add PAW compensation charge n_hat(r) to density array (multi-L with Gaunt coefficients).
///
/// n̂(G) = Σ_a Σ_{i,m_i,j,m_j} ρ_{(i,m_i),(j,m_j)}^a × Σ_{L,M} G(l_i,m_i,l_j,m_j,L,M)
///         × Q^L_{ij}(|G|) × Y_{L,M}(Ĝ) × exp(-iGR_a) / Ω
pub fn addPawCompensationCharge(
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

                // Pre-compute Y_{L,M}(Ĝ) for all (L,M) up to lmax_aug
                var ylm_g: [25]f64 = undefined; // (4+1)^2 = 25 max
                const lmax_aug = gaunt_table.lmax_aug;
                if (g_abs > 1e-10) {
                    for (0..lmax_aug + 1) |big_l| {
                        const bl_i32: i32 = @intCast(big_l);
                        var bm: i32 = -bl_i32;
                        while (bm <= bl_i32) : (bm += 1) {
                            ylm_g[paw_mod.GauntTable.lmIndex(big_l, bm)] =
                                nonlocal_mod.realSphericalHarmonic(bl_i32, bm, gvec.x, gvec.y, gvec.z);
                        }
                    }
                } else {
                    @memset(&ylm_g, 0.0);
                    ylm_g[0] = 1.0 / @sqrt(4.0 * std.math.pi); // Y_00 at G=0
                }

                var sum_re: f64 = 0.0;
                var sum_im: f64 = 0.0;

                for (0..atoms.len) |a| {
                    const sp = atoms[a].species_index;
                    if (sp >= paw_tabs.len) continue;
                    const tab = &paw_tabs[sp];
                    if (tab.nbeta == 0) continue;
                    const mt = rhoij.m_total_per_atom[a];
                    const rij_m = rhoij.values[a];
                    const l_list_r = rhoij.l_per_beta[a];
                    const m_offsets_r = rhoij.m_offsets[a];
                    const pos = atoms[a].position;

                    const g_dot_r = math.Vec3.dot(gvec, pos);
                    const sf_re = @cos(g_dot_r);
                    const sf_im = -@sin(g_dot_r);

                    // For each Q^L entry, sum over m-resolved rhoij with Gaunt coefficients
                    for (0..tab.n_qijl_entries) |e| {
                        const qidx = tab.qijl_indices[e];
                        const big_l = qidx.l;
                        const i_beta = qidx.first;
                        const j_beta = qidx.second;

                        const qijl_g = tab.evalQijlForm(e, g_abs);
                        if (@abs(qijl_g) < 1e-30) continue;

                        const l_i = @as(usize, @intCast(l_list_r[i_beta]));
                        const l_j = @as(usize, @intCast(l_list_r[j_beta]));

                        // Sum over M: Σ_M Y_{L,M}(Ĝ) × [Σ_{m_i,m_j} G(l_i,m_i,l_j,m_j,L,M) × ρ_{(i,m_i),(j,m_j)}]
                        const bl_i32: i32 = @intCast(big_l);
                        var bm: i32 = -bl_i32;
                        while (bm <= bl_i32) : (bm += 1) {
                            const ylm_val = ylm_g[paw_mod.GauntTable.lmIndex(big_l, bm)];
                            if (@abs(ylm_val) < 1e-30) continue;

                            // Sum over m_i, m_j: Σ G(l_i,m_i,l_j,m_j,L,M) × ρ_{(i,m_i),(j,m_j)}
                            var gaunt_rhoij: f64 = 0.0;
                            const li_i32: i32 = @intCast(l_i);
                            const lj_i32: i32 = @intCast(l_j);
                            var mi: i32 = -li_i32;
                            while (mi <= li_i32) : (mi += 1) {
                                const mi_idx = m_offsets_r[i_beta] + @as(usize, @intCast(mi + li_i32));
                                var mj: i32 = -lj_i32;
                                while (mj <= lj_i32) : (mj += 1) {
                                    const g_coeff = gaunt_table.get(l_i, mi, l_j, mj, big_l, bm);
                                    if (g_coeff == 0.0) continue;
                                    const mj_idx = m_offsets_r[j_beta] + @as(usize, @intCast(mj + lj_i32));
                                    gaunt_rhoij += g_coeff * rij_m[mi_idx * mt + mj_idx];
                                }
                            }
                            if (@abs(gaunt_rhoij) < 1e-30) continue;

                            // For i!=j, both (i,j) and (j,i) Q entries should be in qijl.
                            // If only upper triangle is stored, we need sym_factor.
                            const sym_factor: f64 = if (i_beta != j_beta) 2.0 else 1.0;
                            const contrib = gaunt_rhoij * qijl_g * ylm_val * sym_factor * inv_omega;
                            sum_re += contrib * sf_re;
                            sum_im += contrib * sf_im;
                        }
                    }
                }

                n_hat_g[idx].r += sum_re;
                n_hat_g[idx].i += sum_im;
                idx += 1;
            }
        }
    }

    // IFFT n_hat(G) → n_hat(r) and add to density
    const n_hat_r = try reciprocalToReal(alloc, grid, n_hat_g);
    defer alloc.free(n_hat_r);
    for (0..@min(rho.len, n_hat_r.len)) |i| {
        rho[i] += n_hat_r[i];
    }
}

/// Update PAW D_ij per-atom from the total effective potential.
/// D_ij(atom) = D^0_ij + D^hat_ij(atom) + D^xc_ij(atom) + D^H_ij(atom)
/// D^hat depends on atom position via structure factor exp(-iG·R_a).
/// D^xc, D^H depend on atom-specific rhoij.
pub fn updatePawDij(
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
    const total = grid.count();

    for (species, 0..) |entry_s, si| {
        if (si >= paw_tabs.len or paw_tabs[si].nbeta == 0) continue;
        const tab = &paw_tabs[si];
        const nb = tab.nbeta;
        const n_ij = nb * nb;
        const upf = entry_s.upf;
        const paw = upf.paw orelse continue;

        // Compute m_total for this species
        var mt: usize = 0;
        for (0..nb) |b| {
            mt += @as(usize, @intCast(2 * tab.l_list[b] + 1));
        }
        const n_m = mt * mt;

        // Compute m_offsets for this species
        var sp_m_offsets: [32]usize = undefined;
        var off: usize = 0;
        for (0..nb) |b| {
            sp_m_offsets[b] = off;
            off += @as(usize, @intCast(2 * tab.l_list[b] + 1));
        }

        // Count atoms of this species
        var natom: usize = 0;
        for (atoms) |atom| {
            if (atom.species_index == si) natom += 1;
        }
        if (natom == 0) continue;

        // Ensure per-atom D_ij arrays are allocated in each cache
        for (apply_caches) |*ac| {
            if (ac.nonlocal_ctx) |*nl| {
                try nl.ensureDijPerAtom(ac.cache_alloc, si, natom);
                try nl.ensureDijMPerAtom(ac.cache_alloc, si, natom);
            }
        }

        // Compute D_ij for each atom of this species
        var atom_counter: usize = 0;
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index != si) continue;
            const pos = atom.position;

            // Radial D_ij (nb×nb) for forces/stress
            const dij = try alloc.alloc(f64, n_ij);
            defer alloc.free(dij);
            if (upf.dij.len >= n_ij) {
                @memcpy(dij, upf.dij[0..n_ij]);
            } else {
                @memset(dij, 0.0);
            }

            // m-resolved D_ij (mt×mt) for Hamiltonian
            const dij_m = try alloc.alloc(f64, n_m);
            defer alloc.free(dij_m);
            @memset(dij_m, 0.0);

            // Expand D^0 from radial to m-resolved: D^0_{(i,m),(j,m')} = D^0_ij × δ_{mm'} × δ_{li,lj}
            for (0..nb) |i| {
                for (0..nb) |j| {
                    if (tab.l_list[i] != tab.l_list[j]) continue;
                    const d0 = if (upf.dij.len >= n_ij) upf.dij[i * nb + j] else 0.0;
                    const m_count = @as(usize, @intCast(2 * tab.l_list[i] + 1));
                    for (0..m_count) |m| {
                        const im = sp_m_offsets[i] + m;
                        const jm = sp_m_offsets[j] + m;
                        dij_m[im * mt + jm] = d0;
                    }
                }
            }

            // Add D^hat: radial (L=0 only) and m-resolved (all L with Gaunt)
            var git = gvec_iter.GVecIterator.init(grid);
            while (git.next()) |g| {
                if (g.g2 >= ecutrho) continue;
                const g_abs = @sqrt(g.g2);

                const v_hxc = if (g.idx < total) potential.values[g.idx] else math.complex.init(0.0, 0.0);
                const v_loc = if (g.idx < total) ionic.values[g.idx] else math.complex.init(0.0, 0.0);
                const v_eff = math.complex.add(v_hxc, v_loc);

                const g_dot_r = math.Vec3.dot(g.gvec, pos);
                const sf_re = @cos(g_dot_r);
                const sf_im = @sin(g_dot_r);
                const prod_re = v_eff.r * sf_re - v_eff.i * sf_im;

                // Pre-compute Y_{L,M}(Ĝ) for m-resolved D^hat
                var ylm_g: [25]f64 = undefined;
                const lmax_aug = gaunt_table.lmax_aug;
                if (g_abs > 1e-10) {
                    for (0..lmax_aug + 1) |big_l| {
                        const bl_i32: i32 = @intCast(big_l);
                        var bm: i32 = -bl_i32;
                        while (bm <= bl_i32) : (bm += 1) {
                            ylm_g[paw_mod.GauntTable.lmIndex(big_l, bm)] =
                                nonlocal_mod.realSphericalHarmonic(bl_i32, bm, g.gvec.x, g.gvec.y, g.gvec.z);
                        }
                    }
                } else {
                    @memset(&ylm_g, 0.0);
                    ylm_g[0] = 1.0 / @sqrt(4.0 * std.math.pi);
                }

                for (0..tab.n_qijl_entries) |e| {
                    const qidx_e = tab.qijl_indices[e];
                    const big_l = qidx_e.l;
                    const i_beta = qidx_e.first;
                    const j_beta = qidx_e.second;

                    const qijl_g = tab.evalQijlForm(e, g_abs);
                    if (@abs(qijl_g) < 1e-30) continue;

                    // Radial D^hat (L=0 only, for forces)
                    if (big_l == 0) {
                        const ylm_00 = 1.0 / @sqrt(4.0 * std.math.pi);
                        const gaunt_00 = 1.0 / @sqrt(4.0 * std.math.pi);
                        const contrib = prod_re * qijl_g * ylm_00 * gaunt_00;
                        dij[i_beta * nb + j_beta] += contrib;
                        if (i_beta != j_beta) {
                            dij[j_beta * nb + i_beta] += contrib;
                        }
                    }

                    // m-resolved D^hat (all L, for Hamiltonian)
                    const l_i = @as(usize, @intCast(tab.l_list[i_beta]));
                    const l_j = @as(usize, @intCast(tab.l_list[j_beta]));
                    const bl_i32: i32 = @intCast(big_l);
                    var bm: i32 = -bl_i32;
                    while (bm <= bl_i32) : (bm += 1) {
                        const ylm_val = ylm_g[paw_mod.GauntTable.lmIndex(big_l, bm)];
                        if (@abs(ylm_val) < 1e-30) continue;

                        const li_i32: i32 = @intCast(l_i);
                        const lj_i32: i32 = @intCast(l_j);
                        var mi: i32 = -li_i32;
                        while (mi <= li_i32) : (mi += 1) {
                            const im = sp_m_offsets[i_beta] + @as(usize, @intCast(mi + li_i32));
                            var mj: i32 = -lj_i32;
                            while (mj <= lj_i32) : (mj += 1) {
                                const g_coeff = gaunt_table.get(l_i, mi, l_j, mj, big_l, bm);
                                if (g_coeff == 0.0) continue;
                                const jm = sp_m_offsets[j_beta] + @as(usize, @intCast(mj + lj_i32));
                                const contrib_m = prod_re * qijl_g * ylm_val * g_coeff;
                                dij_m[im * mt + jm] += contrib_m;
                                if (i_beta != j_beta) {
                                    dij_m[jm * mt + im] += contrib_m;
                                }
                            }
                        }
                    }
                }
            }

            // D^xc is m-resolved: dij_xc_m[mt × mt]
            const dij_xc_m = try alloc.alloc(f64, mt * mt);
            defer alloc.free(dij_xc_m);
            if (skip_dxc) {
                @memset(dij_xc_m, 0.0);
            } else if (rhoij_spin) |rij_s| {
                // Spin-resolved D^xc: compute from (this_channel, other_channel)
                // other_channel = total - this_channel
                const rij_other = try alloc.alloc(f64, mt * mt);
                defer alloc.free(rij_other);
                for (0..mt * mt) |idx2| {
                    rij_other[idx2] = rhoij.values[ai][idx2] - rij_s.values[ai][idx2];
                }
                const dij_xc_other = try alloc.alloc(f64, mt * mt);
                defer alloc.free(dij_xc_other);
                try paw_mod.paw_xc.computePawDijXcAngularSpin(
                    alloc,
                    dij_xc_m,
                    dij_xc_other,
                    paw,
                    rij_s.values[ai],
                    rij_other,
                    rhoij.m_total_per_atom[ai],
                    rhoij.m_offsets[ai],
                    upf.r,
                    upf.rab,
                    paw.ae_core_density,
                    if (upf.nlcc.len > 0) upf.nlcc else null,
                    xc_func,
                    gaunt_table,
                );
            } else {
                try paw_mod.paw_xc.computePawDijXcAngular(
                    alloc,
                    dij_xc_m,
                    paw,
                    rhoij.values[ai],
                    rhoij.m_total_per_atom[ai],
                    rhoij.m_offsets[ai],
                    upf.r,
                    upf.rab,
                    paw.ae_core_density,
                    if (upf.nlcc.len > 0) upf.nlcc else null,
                    xc_func,
                    gaunt_table,
                );
            }
            // Add m-resolved D^xc directly to dij_m
            for (0..mt * mt) |idx2| {
                dij_m[idx2] += dij_xc_m[idx2];
            }
            // Contract D^xc to radial for dij (used in stress/forces).
            // Convention: dij[nb×nb] stores the per-m value (same as D^0, D^hat).
            // Average over m gives the per-m representative value.
            for (0..nb) |i| {
                for (0..nb) |j| {
                    if (tab.l_list[i] != tab.l_list[j]) continue;
                    const m_count = @as(usize, @intCast(2 * tab.l_list[i] + 1));
                    var sum_dxc: f64 = 0.0;
                    for (0..m_count) |m| {
                        const im = sp_m_offsets[i] + m;
                        const jm = sp_m_offsets[j] + m;
                        sum_dxc += dij_xc_m[im * mt + jm];
                    }
                    // Average over m: radial D convention is per-m value
                    dij[i * nb + j] += sum_dxc / @as(f64, @floatFromInt(m_count));
                }
            }

            // Add D^H (on-site Hartree, multi-L with Gaunt) to D_full.
            // E_paw_onsite includes E_H_onsite (computePawEhOnsiteMultiL), so D^H must
            // also be in D_full and double-counting for Hellmann-Feynman consistency.
            {
                const dij_h_m = try alloc.alloc(f64, n_m);
                defer alloc.free(dij_h_m);
                try paw_mod.paw_xc.computePawDijHartreeMultiL(
                    alloc,
                    dij_h_m,
                    paw,
                    rhoij.values[ai],
                    rhoij.m_total_per_atom[ai],
                    rhoij.m_offsets[ai],
                    upf.r,
                    upf.rab,
                    gaunt_table,
                );
                // Add m-resolved D^H directly to dij_m
                for (0..n_m) |idx2| {
                    dij_m[idx2] += dij_h_m[idx2];
                }
                // Contract D^H to radial for dij (used in stress/forces).
                for (0..nb) |i| {
                    for (0..nb) |j| {
                        if (tab.l_list[i] != tab.l_list[j]) continue;
                        const m_count = @as(usize, @intCast(2 * tab.l_list[i] + 1));
                        var sum_dh: f64 = 0.0;
                        for (0..m_count) |m| {
                            const im = sp_m_offsets[i] + m;
                            const jm = sp_m_offsets[j] + m;
                            sum_dh += dij_h_m[im * mt + jm];
                        }
                        dij[i * nb + j] += sum_dh / @as(f64, @floatFromInt(m_count));
                    }
                }
            }

            // Mix D_ij with old value from first cache, then write to all caches
            if (dij_mix_beta < 1.0 - 1e-10) {
                if (apply_caches.len > 0) {
                    if (apply_caches[0].nonlocal_ctx) |*nl| {
                        if (nl.species[si].dij_per_atom) |dpa| {
                            if (atom_counter < dpa.len) {
                                for (0..@min(dij.len, dpa[atom_counter].len)) |ii| {
                                    dij[ii] = (1.0 - dij_mix_beta) * dpa[atom_counter][ii] + dij_mix_beta * dij[ii];
                                }
                            }
                        }
                        if (nl.species[si].dij_m_per_atom) |dpa| {
                            if (atom_counter < dpa.len) {
                                for (0..@min(dij_m.len, dpa[atom_counter].len)) |ii| {
                                    dij_m[ii] = (1.0 - dij_mix_beta) * dpa[atom_counter][ii] + dij_mix_beta * dij_m[ii];
                                }
                            }
                        }
                    }
                }
            }
            for (apply_caches) |*ac| {
                if (ac.nonlocal_ctx) |*nl| {
                    nl.updateDijAtom(si, atom_counter, dij);
                    nl.updateDijMAtom(si, atom_counter, dij_m);
                }
            }

            atom_counter += 1;
        }

        // Symmetrize D_ij across equivalent atoms of this species
        if (symmetrize and natom > 1) {
            // Symmetrize radial D_ij
            const avg_dij = try alloc.alloc(f64, n_ij);
            defer alloc.free(avg_dij);
            @memset(avg_dij, 0.0);
            if (apply_caches.len > 0) {
                if (apply_caches[0].nonlocal_ctx) |*nl| {
                    if (nl.species[si].dij_per_atom) |dpa| {
                        for (0..natom) |a| {
                            for (0..n_ij) |idx2| {
                                avg_dij[idx2] += dpa[a][idx2];
                            }
                        }
                    }
                }
            }
            const inv_natom = 1.0 / @as(f64, @floatFromInt(natom));
            for (0..n_ij) |idx2| {
                avg_dij[idx2] *= inv_natom;
            }
            for (apply_caches) |*ac| {
                if (ac.nonlocal_ctx) |*nl| {
                    if (nl.species[si].dij_per_atom) |dpa| {
                        for (0..natom) |a| {
                            @memcpy(dpa[a], avg_dij);
                        }
                    }
                }
            }

            // Symmetrize m-resolved D_ij
            const avg_dij_m = try alloc.alloc(f64, n_m);
            defer alloc.free(avg_dij_m);
            @memset(avg_dij_m, 0.0);
            if (apply_caches.len > 0) {
                if (apply_caches[0].nonlocal_ctx) |*nl| {
                    if (nl.species[si].dij_m_per_atom) |dpa| {
                        for (0..natom) |a| {
                            for (0..n_m) |idx2| {
                                avg_dij_m[idx2] += dpa[a][idx2];
                            }
                        }
                    }
                }
            }
            for (0..n_m) |idx2| {
                avg_dij_m[idx2] *= inv_natom;
            }
            for (apply_caches) |*ac| {
                if (ac.nonlocal_ctx) |*nl| {
                    if (nl.species[si].dij_m_per_atom) |dpa| {
                        for (0..natom) |a| {
                            @memcpy(dpa[a], avg_dij_m);
                        }
                    }
                }
            }
        }
    }
}

/// Compute PAW on-site energy correction for all atoms.
/// When rhoij_up/rhoij_down are provided, uses spin-resolved E_xc.
pub fn computePawOnsiteEnergyTotal(
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
            const e_xc = try paw_mod.paw_xc.computePawExcOnsiteAngularSpin(
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
            const e_h = try paw_mod.paw_xc.computePawEhOnsiteMultiL(
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
            const e_atom = try paw_mod.paw_energy.computePawOnsiteEnergy(
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
