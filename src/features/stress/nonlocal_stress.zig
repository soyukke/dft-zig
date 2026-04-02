const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const scf = @import("../scf/scf.zig");
const paw_mod = @import("../paw/paw_tab.zig");
const stress_util = @import("stress.zig");

const Stress3x3 = stress_util.Stress3x3;
const dYlm_dq = stress_util.dYlm_dq;

/// Nonlocal pseudopotential stress.
/// σ_αβ = -(E_nl/Ω) δ_αβ - (2 spin/Ω²) Σ_nk f w Re[Σ D conj(dp_αβ) p]
/// where dp_αβ = Σ_G (∂φ/∂q_α × q_β) × S(G) × c(G)
pub fn nonlocalStress(
    alloc: std.mem.Allocator,
    wavefunctions: ?scf.WavefunctionData,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    radial_tables_list: ?[]nonlocal.RadialTableSet,
    paw_dij_per_atom: ?[]const []const f64,
    paw_dij_m_per_atom: ?[]const []const f64,
    paw_tabs: ?[]const paw_mod.PawTab,
    spin_factor: f64,
) !Stress3x3 {
    var sigma = stress_util.zeroStress();
    const wf = wavefunctions orelse return sigma;
    const inv_volume = 1.0 / volume;

    for (wf.kpoints) |kp| {
        var basis = try plane_wave.generate(alloc, recip, wf.ecut_ry, kp.k_cart);
        defer basis.deinit(alloc);
        const gvecs = basis.gvecs;
        const n = gvecs.len;
        if (n != kp.basis_len) continue;

        for (atoms, 0..) |atom, atom_idx| {
            const sp = &species[atom.species_index];
            const upf = sp.upf;
            if (upf.beta.len == 0 or upf.dij.len == 0) continue;
            const nb = upf.beta.len;
            const dij_data: []const f64 = if (paw_dij_per_atom) |pda| pda[atom_idx] else upf.dij;
            const dij_m_data: ?[]const f64 = if (paw_dij_m_per_atom) |pda| pda[atom_idx] else null;
            const qij_data: ?[]const f64 = if (paw_tabs) |tabs| if (atom.species_index < tabs.len) tabs[atom.species_index].sij else null else null;
            const dij_eff_buf: ?[]f64 = if (qij_data != null) try alloc.alloc(f64, nb * nb) else null;
            defer if (dij_eff_buf) |buf| alloc.free(buf);

            const tables = if (radial_tables_list) |rtl| if (atom.species_index < rtl.len) rtl[atom.species_index] else null else null;

            const radial_vals = try alloc.alloc(f64, nb * n);
            defer alloc.free(radial_vals);
            const radial_derivs = try alloc.alloc(f64, nb * n);
            defer alloc.free(radial_derivs);

            for (0..nb) |b_idx| {
                const l_val = upf.beta[b_idx].l orelse 0;
                _ = l_val;
                for (0..n) |g| {
                    const gmag = math.Vec3.norm(gvecs[g].kpg);
                    if (tables) |t| {
                        radial_vals[b_idx * n + g] = t.tables[b_idx].eval(gmag);
                        radial_derivs[b_idx * n + g] = t.tables[b_idx].evalDeriv(gmag);
                    } else {
                        const beta = upf.beta[b_idx];
                        const l = beta.l orelse 0;
                        radial_vals[b_idx * n + g] = nonlocal.radialProjector(beta.values, upf.r, upf.rab, l, gmag);
                        const dg: f64 = 0.001;
                        const rp = nonlocal.radialProjector(beta.values, upf.r, upf.rab, l, gmag + dg);
                        const rm = nonlocal.radialProjector(beta.values, upf.r, upf.rab, l, if (gmag > dg) gmag - dg else 0.0);
                        radial_derivs[b_idx * n + g] = (rp - rm) / (2.0 * dg);
                    }
                }
            }

            var m_total: usize = 0;
            for (0..nb) |b_idx| {
                const l_val = upf.beta[b_idx].l orelse 0;
                m_total += @as(usize, @intCast(2 * l_val + 1));
            }

            const phase_buf = try alloc.alloc(math.Complex, n);
            defer alloc.free(phase_buf);
            for (gvecs, 0..) |gv, g| {
                phase_buf[g] = math.complex.expi(math.Vec3.dot(gv.cart, atom.position));
            }

            const m_offsets = try alloc.alloc(usize, nb);
            defer alloc.free(m_offsets);
            const m_counts = try alloc.alloc(usize, nb);
            defer alloc.free(m_counts);
            {
                var off: usize = 0;
                for (0..nb) |b_idx| {
                    const l_val = upf.beta[b_idx].l orelse 0;
                    m_offsets[b_idx] = off;
                    m_counts[b_idx] = @as(usize, @intCast(2 * l_val + 1));
                    off += m_counts[b_idx];
                }
            }

            const p_buf = try alloc.alloc(math.Complex, m_total);
            defer alloc.free(p_buf);
            const dp_buf = try alloc.alloc(math.Complex, 3 * m_total);
            defer alloc.free(dp_buf);
            const dij_m_eff_buf: ?[]f64 = if (dij_m_data != null and qij_data != null) try alloc.alloc(f64, m_total * m_total) else null;
            defer if (dij_m_eff_buf) |buf| alloc.free(buf);

            for (0..kp.nbands) |band| {
                const occ = kp.occupations[band];
                if (occ <= 0.0) continue;
                const c = kp.coefficients[band * n .. (band + 1) * n];

                if (dij_m_eff_buf) |dm_buf| {
                    const eigenval = kp.eigenvalues[band];
                    const dm = dij_m_data.?;
                    const qij = qij_data.?;
                    @memcpy(dm_buf, dm);
                    for (0..nb) |bi| {
                        for (0..nb) |bj| {
                            if ((upf.beta[bi].l orelse 0) != (upf.beta[bj].l orelse 0)) continue;
                            const q_ij = qij[bi * nb + bj] - (if (bi == bj) @as(f64, 1.0) else @as(f64, 0.0));
                            if (q_ij == 0.0) continue;
                            const mc = m_counts[bi];
                            for (0..mc) |mi| {
                                const bm = m_offsets[bi] + mi;
                                const jm = m_offsets[bj] + mi;
                                dm_buf[bm * m_total + jm] -= eigenval * q_ij;
                            }
                        }
                    }
                }
                const dij_for_band: []const f64 = if (dij_eff_buf) |buf| blk: {
                    const eigenval = kp.eigenvalues[band];
                    const qij = qij_data.?;
                    for (0..nb) |bi| {
                        for (0..nb) |bj| {
                            const idx_ij = bi * nb + bj;
                            const q_ij = qij[idx_ij] - (if (bi == bj) @as(f64, 1.0) else @as(f64, 0.0));
                            buf[idx_ij] = dij_data[idx_ij] - eigenval * q_ij;
                        }
                    }
                    break :blk buf;
                } else dij_data;
                const use_dij_m = dij_m_eff_buf != null;

                @memset(p_buf, math.complex.init(0, 0));
                @memset(dp_buf, math.complex.init(0, 0));

                for (0..nb) |b_idx| {
                    const l_val = upf.beta[b_idx].l orelse 0;
                    const m_count = m_counts[b_idx];
                    const m_off = m_offsets[b_idx];

                    for (0..m_count) |m_idx| {
                        const m = @as(i32, @intCast(m_idx)) - l_val;

                        for (0..n) |g| {
                            const q = gvecs[g].kpg;
                            const q_mag = math.Vec3.norm(q);
                            const radial = radial_vals[b_idx * n + g];
                            const ylm = nonlocal.realSphericalHarmonic(l_val, m, q.x, q.y, q.z);
                            const phi = 4.0 * std.math.pi * radial * ylm;

                            const pc = math.complex.mul(phase_buf[g], c[g]);
                            p_buf[m_off + m_idx] = math.complex.add(p_buf[m_off + m_idx], math.complex.scale(pc, phi));

                            if (q_mag < 1e-12) continue;
                            const dradial = radial_derivs[b_idx * n + g];
                            const inv_qmag = 1.0 / q_mag;
                            const nhat = [3]f64{ q.x * inv_qmag, q.y * inv_qmag, q.z * inv_qmag };

                            const dy = dYlm_dq(l_val, m, q.x, q.y, q.z, q_mag);

                            for (0..3) |dir| {
                                const dphi = 4.0 * std.math.pi * (dradial * nhat[dir] * ylm + radial * dy[dir]);
                                dp_buf[dir * m_total + m_off + m_idx] = math.complex.add(
                                    dp_buf[dir * m_total + m_off + m_idx],
                                    math.complex.scale(pc, dphi),
                                );
                            }
                        }
                    }
                }

                const prefactor = 2.0 * occ * kp.weight * spin_factor * inv_volume * inv_volume;

                for (0..n) |g| {
                    const q = gvecs[g].kpg;
                    const q_mag = math.Vec3.norm(q);
                    const pc = math.complex.mul(phase_buf[g], c[g]);
                    if (q_mag < 1e-12) continue;
                    const inv_qmag = 1.0 / q_mag;
                    const nhat = [3]f64{ q.x * inv_qmag, q.y * inv_qmag, q.z * inv_qmag };
                    const qv = [3]f64{ q.x, q.y, q.z };

                    for (0..nb) |b_idx| {
                        const l_b = upf.beta[b_idx].l orelse 0;
                        const m_count_b = m_counts[b_idx];

                        for (0..m_count_b) |m_idx| {
                            const m = @as(i32, @intCast(m_idx)) - l_b;
                            const bm = m_offsets[b_idx] + m_idx;

                            var dp_bm = math.complex.init(0, 0);
                            if (use_dij_m) {
                                const dm_eff = dij_m_eff_buf.?;
                                for (0..nb) |j_idx| {
                                    if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                                    const mj_count = m_counts[j_idx];
                                    for (0..mj_count) |mj| {
                                        const jm = m_offsets[j_idx] + mj;
                                        const d_val = dm_eff[bm * m_total + jm];
                                        if (d_val == 0) continue;
                                        dp_bm = math.complex.add(dp_bm, math.complex.scale(p_buf[jm], d_val));
                                    }
                                }
                            } else {
                                for (0..nb) |j_idx| {
                                    if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                                    const d_val = dij_for_band[b_idx * nb + j_idx];
                                    if (d_val == 0) continue;
                                    dp_bm = math.complex.add(dp_bm, math.complex.scale(p_buf[m_offsets[j_idx] + m_idx], d_val));
                                }
                            }
                            if (dp_bm.r == 0 and dp_bm.i == 0) continue;

                            const radial = radial_vals[b_idx * n + g];
                            const dradial_val = radial_derivs[b_idx * n + g];
                            const ylm = nonlocal.realSphericalHarmonic(l_b, m, q.x, q.y, q.z);
                            const dy = dYlm_dq(l_b, m, q.x, q.y, q.z, q_mag);

                            const z = math.complex.mul(math.complex.conj(dp_bm), pc);

                            for (0..3) |a| {
                                const dphi_a = 4.0 * std.math.pi * (dradial_val * nhat[a] * ylm + radial * dy[a]);
                                for (a..3) |b| {
                                    sigma[a][b] -= prefactor * dphi_a * qv[b] * z.r;
                                }
                            }
                        }
                    }
                }

                var e_nl_nk: f64 = 0.0;
                for (0..nb) |b_idx| {
                    const l_b = upf.beta[b_idx].l orelse 0;
                    const m_count_b = m_counts[b_idx];

                    for (0..m_count_b) |m_idx| {
                        const bm = m_offsets[b_idx] + m_idx;
                        var dp_bm_e = math.complex.init(0, 0);
                        if (use_dij_m) {
                            const dm_eff = dij_m_eff_buf.?;
                            for (0..nb) |j_idx| {
                                if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                                const mj_count = m_counts[j_idx];
                                for (0..mj_count) |mj| {
                                    const jm = m_offsets[j_idx] + mj;
                                    const d_val = dm_eff[bm * m_total + jm];
                                    if (d_val == 0) continue;
                                    dp_bm_e = math.complex.add(dp_bm_e, math.complex.scale(p_buf[jm], d_val));
                                }
                            }
                        } else {
                            for (0..nb) |j_idx| {
                                if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                                dp_bm_e = math.complex.add(dp_bm_e, math.complex.scale(
                                    p_buf[m_offsets[j_idx] + m_idx],
                                    dij_for_band[b_idx * nb + j_idx],
                                ));
                            }
                        }
                        const p_bm = p_buf[bm];
                        e_nl_nk += math.complex.mul(math.complex.conj(p_bm), dp_bm_e).r;
                    }
                }
                e_nl_nk *= inv_volume;
                const diag_contrib = -occ * kp.weight * spin_factor * e_nl_nk * inv_volume;
                for (0..3) |a| sigma[a][a] += diag_contrib;
            }
        }
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}
