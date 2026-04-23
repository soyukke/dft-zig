const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const scf = @import("../scf/scf.zig");
const paw_mod = @import("../paw/paw_tab.zig");
const stress_util = @import("stress.zig");

const Stress3x3 = stress_util.Stress3x3;
const dYlm_dq = stress_util.dYlm_dq;

/// Fill radial_vals and radial_derivs arrays for all beta channels and G-vectors.
fn fillRadialValuesAndDerivs(
    upf: pseudo.UpfData,
    nb: usize,
    n: usize,
    gvecs: []const plane_wave.GVector,
    tables: ?nonlocal.RadialTableSet,
    radial_vals: []f64,
    radial_derivs: []f64,
) void {
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
                const bv = beta.values;
                radial_vals[b_idx * n + g] =
                    nonlocal.radialProjector(bv, upf.r, upf.rab, l, gmag);
                const dg: f64 = 0.001;
                const rp = nonlocal.radialProjector(bv, upf.r, upf.rab, l, gmag + dg);
                const gminus = if (gmag > dg) gmag - dg else 0.0;
                const rm = nonlocal.radialProjector(bv, upf.r, upf.rab, l, gminus);
                radial_derivs[b_idx * n + g] = (rp - rm) / (2.0 * dg);
            }
        }
    }
}

/// Populate m_offsets and m_counts, and return m_total for a species' beta channels.
fn buildMLayout(upf: pseudo.UpfData, nb: usize, m_offsets: []usize, m_counts: []usize) usize {
    var off: usize = 0;
    for (0..nb) |b_idx| {
        const l_val = upf.beta[b_idx].l orelse 0;
        m_offsets[b_idx] = off;
        m_counts[b_idx] = @as(usize, @intCast(2 * l_val + 1));
        off += m_counts[b_idx];
    }
    return off;
}

/// Build the effective per-band D_ij matrix (D^0 - ε_nk × q_ij) when PAW Q is active.
fn fillDijEffective(
    nb: usize,
    dij_data: []const f64,
    qij_data: []const f64,
    eigenval: f64,
    buf: []f64,
) void {
    for (0..nb) |bi| {
        for (0..nb) |bj| {
            const idx_ij = bi * nb + bj;
            const delta_bi_bj: f64 = if (bi == bj) 1.0 else 0.0;
            const q_ij = qij_data[idx_ij] - delta_bi_bj;
            buf[idx_ij] = dij_data[idx_ij] - eigenval * q_ij;
        }
    }
}

/// Apply the -ε_nk × q_ij correction to the expanded (b,m)-indexed D_m matrix in-place.
fn fillDijMEffective(
    upf: pseudo.UpfData,
    nb: usize,
    dm: []const f64,
    qij: []const f64,
    eigenval: f64,
    m_offsets: []const usize,
    m_counts: []const usize,
    m_total: usize,
    dm_buf: []f64,
) void {
    @memcpy(dm_buf, dm);
    for (0..nb) |bi| {
        for (0..nb) |bj| {
            if ((upf.beta[bi].l orelse 0) != (upf.beta[bj].l orelse 0)) continue;
            const delta_bi_bj: f64 = if (bi == bj) 1.0 else 0.0;
            const q_ij = qij[bi * nb + bj] - delta_bi_bj;
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

/// Compute p_bm = Σ_G φ × phase × c and dp_bm/dq_dir = Σ_G (∂φ/∂q) × phase × c.
fn projectPAndDp(
    upf: pseudo.UpfData,
    nb: usize,
    n: usize,
    gvecs: []const plane_wave.GVector,
    phase_buf: []const math.Complex,
    c: []const math.Complex,
    radial_vals: []const f64,
    radial_derivs: []const f64,
    m_offsets: []const usize,
    m_counts: []const usize,
    m_total: usize,
    p_buf: []math.Complex,
    dp_buf: []math.Complex,
) void {
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
                const scaled_phi = math.complex.scale(pc, phi);
                p_buf[m_off + m_idx] =
                    math.complex.add(p_buf[m_off + m_idx], scaled_phi);

                if (q_mag < 1e-12) continue;
                const dradial = radial_derivs[b_idx * n + g];
                const inv_qmag = 1.0 / q_mag;
                const nhat = [3]f64{ q.x * inv_qmag, q.y * inv_qmag, q.z * inv_qmag };

                const dy = dYlm_dq(l_val, m, q.x, q.y, q.z, q_mag);

                for (0..3) |dir| {
                    const dphi = 4.0 * std.math.pi *
                        (dradial * nhat[dir] * ylm + radial * dy[dir]);
                    dp_buf[dir * m_total + m_off + m_idx] = math.complex.add(
                        dp_buf[dir * m_total + m_off + m_idx],
                        math.complex.scale(pc, dphi),
                    );
                }
            }
        }
    }
}

/// Compute Dp_bm = Σ_j D_{bm,jm'} × p_jm for a single (b, m) channel.
fn computeDpBm(
    upf: pseudo.UpfData,
    nb: usize,
    b_idx: usize,
    l_b: i32,
    m_idx: usize,
    bm: usize,
    p_buf: []const math.Complex,
    dij_for_band: []const f64,
    dij_m_eff: ?[]const f64,
    m_offsets: []const usize,
    m_counts: []const usize,
    m_total: usize,
) math.Complex {
    var dp_bm = math.complex.init(0, 0);
    if (dij_m_eff) |dm_eff| {
        for (0..nb) |j_idx| {
            if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
            const mj_count = m_counts[j_idx];
            for (0..mj_count) |mj| {
                const jm = m_offsets[j_idx] + mj;
                const d_val = dm_eff[bm * m_total + jm];
                if (d_val == 0) continue;
                dp_bm = math.complex.add(
                    dp_bm,
                    math.complex.scale(p_buf[jm], d_val),
                );
            }
        }
    } else {
        for (0..nb) |j_idx| {
            if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
            const d_val = dij_for_band[b_idx * nb + j_idx];
            if (d_val == 0) continue;
            const pj = p_buf[m_offsets[j_idx] + m_idx];
            dp_bm = math.complex.add(
                dp_bm,
                math.complex.scale(pj, d_val),
            );
        }
    }
    return dp_bm;
}

/// Accumulate σ_αβ over all (G, b, m) at a single band.
fn accumulateNonlocalSigmaForBand(
    upf: pseudo.UpfData,
    nb: usize,
    n: usize,
    gvecs: []const plane_wave.GVector,
    phase_buf: []const math.Complex,
    c: []const math.Complex,
    radial_vals: []const f64,
    radial_derivs: []const f64,
    p_buf: []const math.Complex,
    dij_for_band: []const f64,
    dij_m_eff: ?[]const f64,
    m_offsets: []const usize,
    m_counts: []const usize,
    m_total: usize,
    prefactor: f64,
    sigma: *Stress3x3,
) void {
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

                const dp_bm = computeDpBm(
                    upf,
                    nb,
                    b_idx,
                    l_b,
                    m_idx,
                    bm,
                    p_buf,
                    dij_for_band,
                    dij_m_eff,
                    m_offsets,
                    m_counts,
                    m_total,
                );
                if (dp_bm.r == 0 and dp_bm.i == 0) continue;

                const radial = radial_vals[b_idx * n + g];
                const dradial_val = radial_derivs[b_idx * n + g];
                const ylm = nonlocal.realSphericalHarmonic(l_b, m, q.x, q.y, q.z);
                const dy = dYlm_dq(l_b, m, q.x, q.y, q.z, q_mag);

                const z = math.complex.mul(math.complex.conj(dp_bm), pc);

                for (0..3) |a| {
                    const dphi_a = 4.0 * std.math.pi *
                        (dradial_val * nhat[a] * ylm + radial * dy[a]);
                    for (a..3) |b| {
                        sigma[a][b] -= prefactor * dphi_a * qv[b] * z.r;
                    }
                }
            }
        }
    }
}

/// Compute E_nl per (k, band) as Σ_{b,m} Re[conj(p_bm) × Σ_j D_bj p_jm] / Ω.
fn computeEnlForBand(
    upf: pseudo.UpfData,
    nb: usize,
    p_buf: []const math.Complex,
    dij_for_band: []const f64,
    dij_m_eff: ?[]const f64,
    m_offsets: []const usize,
    m_counts: []const usize,
    m_total: usize,
) f64 {
    var e_nl_nk: f64 = 0.0;
    for (0..nb) |b_idx| {
        const l_b = upf.beta[b_idx].l orelse 0;
        const m_count_b = m_counts[b_idx];

        for (0..m_count_b) |m_idx| {
            const bm = m_offsets[b_idx] + m_idx;
            var dp_bm_e = math.complex.init(0, 0);
            if (dij_m_eff) |dm_eff| {
                for (0..nb) |j_idx| {
                    if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                    const mj_count = m_counts[j_idx];
                    for (0..mj_count) |mj| {
                        const jm = m_offsets[j_idx] + mj;
                        const d_val = dm_eff[bm * m_total + jm];
                        if (d_val == 0) continue;
                        dp_bm_e = math.complex.add(
                            dp_bm_e,
                            math.complex.scale(p_buf[jm], d_val),
                        );
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
    return e_nl_nk;
}

/// Shared inputs for nonlocal stress atom/band helpers.
const NonlocalStressInputs = struct {
    kp: scf.KpointWavefunction,
    species: []const hamiltonian.SpeciesEntry,
    gvecs: []const plane_wave.GVector,
    n: usize,
    radial_tables_list: ?[]nonlocal.RadialTableSet,
    paw_dij_per_atom: ?[]const []const f64,
    paw_dij_m_per_atom: ?[]const []const f64,
    paw_tabs: ?[]const paw_mod.PawTab,
    spin_factor: f64,
    inv_volume: f64,
};

/// Per-atom work buffers and effective D pointers used inside the band loop.
const NonlocalAtomWork = struct {
    upf: pseudo.UpfData,
    nb: usize,
    phase_buf: []math.Complex,
    radial_vals: []f64,
    radial_derivs: []f64,
    m_offsets: []usize,
    m_counts: []usize,
    m_total: usize,
    p_buf: []math.Complex,
    dp_buf: []math.Complex,
    dij_data: []const f64,
    dij_m_data: ?[]const f64,
    qij_data: ?[]const f64,
    dij_eff_buf: ?[]f64,
    dij_m_eff_buf: ?[]f64,

    fn deinit(self: *NonlocalAtomWork, alloc: std.mem.Allocator) void {
        alloc.free(self.phase_buf);
        alloc.free(self.radial_vals);
        alloc.free(self.radial_derivs);
        alloc.free(self.m_offsets);
        alloc.free(self.m_counts);
        alloc.free(self.p_buf);
        alloc.free(self.dp_buf);
        if (self.dij_eff_buf) |buf| alloc.free(buf);
        if (self.dij_m_eff_buf) |buf| alloc.free(buf);
    }
};

/// Allocate all per-atom work buffers and initialize pointers. Caller must call work.deinit.
fn allocNonlocalAtomWork(
    alloc: std.mem.Allocator,
    inputs: NonlocalStressInputs,
    atom: hamiltonian.AtomData,
    atom_idx: usize,
) !?NonlocalAtomWork {
    const sp = &inputs.species[atom.species_index];
    const upf = sp.upf;
    if (upf.beta.len == 0 or upf.dij.len == 0) return null;
    const nb = upf.beta.len;
    const n = inputs.n;
    const dij_data: []const f64 = if (inputs.paw_dij_per_atom) |pda| pda[atom_idx] else upf.dij;
    const dij_m_data: ?[]const f64 = if (inputs.paw_dij_m_per_atom) |pda| pda[atom_idx] else null;
    const qij_data: ?[]const f64 = if (inputs.paw_tabs) |tabs|
        if (atom.species_index < tabs.len) tabs[atom.species_index].sij else null
    else
        null;
    const dij_eff_buf: ?[]f64 = if (qij_data != null)
        try alloc.alloc(f64, nb * nb)
    else
        null;
    const tables = if (inputs.radial_tables_list) |rtl|
        if (atom.species_index < rtl.len) rtl[atom.species_index] else null
    else
        null;

    const radial_vals = try alloc.alloc(f64, nb * n);
    const radial_derivs = try alloc.alloc(f64, nb * n);
    fillRadialValuesAndDerivs(upf.*, nb, n, inputs.gvecs, tables, radial_vals, radial_derivs);

    const phase_buf = try alloc.alloc(math.Complex, n);
    for (inputs.gvecs, 0..) |gv, g| {
        phase_buf[g] = math.complex.expi(math.Vec3.dot(gv.cart, atom.position));
    }

    const m_offsets = try alloc.alloc(usize, nb);
    const m_counts = try alloc.alloc(usize, nb);
    const m_total = buildMLayout(upf.*, nb, m_offsets, m_counts);

    const p_buf = try alloc.alloc(math.Complex, m_total);
    const dp_buf = try alloc.alloc(math.Complex, 3 * m_total);
    const dij_m_eff_buf: ?[]f64 = if (dij_m_data != null and qij_data != null)
        try alloc.alloc(f64, m_total * m_total)
    else
        null;

    return NonlocalAtomWork{
        .upf = upf.*,
        .nb = nb,
        .phase_buf = phase_buf,
        .radial_vals = radial_vals,
        .radial_derivs = radial_derivs,
        .m_offsets = m_offsets,
        .m_counts = m_counts,
        .m_total = m_total,
        .p_buf = p_buf,
        .dp_buf = dp_buf,
        .dij_data = dij_data,
        .dij_m_data = dij_m_data,
        .qij_data = qij_data,
        .dij_eff_buf = dij_eff_buf,
        .dij_m_eff_buf = dij_m_eff_buf,
    };
}

/// Accumulate nonlocal stress contributions from all bands of one atom at a single k-point.
fn accumulateNonlocalStressAtom(
    alloc: std.mem.Allocator,
    inputs: NonlocalStressInputs,
    atom: hamiltonian.AtomData,
    atom_idx: usize,
    sigma: *Stress3x3,
) !void {
    var work_opt = try allocNonlocalAtomWork(alloc, inputs, atom, atom_idx);
    if (work_opt == null) return;
    defer work_opt.?.deinit(alloc);

    for (0..inputs.kp.nbands) |band| {
        try processNonlocalStressBand(inputs, work_opt.?, band, sigma);
    }
}

/// Process a single band: build effective D matrices, project p/dp, accumulate σ and E_nl.
/// Build effective D matrices (both Dij and Dij_m) for the current band.
fn prepareBandEffectiveD(
    work: NonlocalAtomWork,
    kp: scf.KpointWavefunction,
    band: usize,
) struct { dij_for_band: []const f64, dij_m_eff: ?[]const f64 } {
    if (work.dij_m_eff_buf) |dm_buf| {
        fillDijMEffective(
            work.upf,
            work.nb,
            work.dij_m_data.?,
            work.qij_data.?,
            kp.eigenvalues[band],
            work.m_offsets,
            work.m_counts,
            work.m_total,
            dm_buf,
        );
    }
    const dij_for_band: []const f64 = if (work.dij_eff_buf) |buf| blk: {
        fillDijEffective(work.nb, work.dij_data, work.qij_data.?, kp.eigenvalues[band], buf);
        break :blk buf;
    } else work.dij_data;
    const dij_m_eff: ?[]const f64 = if (work.dij_m_eff_buf) |b| b else null;
    return .{ .dij_for_band = dij_for_band, .dij_m_eff = dij_m_eff };
}

fn processNonlocalStressBand(
    inputs: NonlocalStressInputs,
    work: NonlocalAtomWork,
    band: usize,
    sigma: *Stress3x3,
) !void {
    const kp = inputs.kp;
    const occ = kp.occupations[band];
    if (occ <= 0.0) return;
    const n = inputs.n;
    const c = kp.coefficients[band * n .. (band + 1) * n];

    const dij_bundle = prepareBandEffectiveD(work, kp, band);
    const dij_for_band = dij_bundle.dij_for_band;
    const dij_m_eff = dij_bundle.dij_m_eff;

    projectPAndDp(
        work.upf,
        work.nb,
        n,
        inputs.gvecs,
        work.phase_buf,
        c,
        work.radial_vals,
        work.radial_derivs,
        work.m_offsets,
        work.m_counts,
        work.m_total,
        work.p_buf,
        work.dp_buf,
    );

    const prefactor = 2.0 * occ * kp.weight *
        inputs.spin_factor * inputs.inv_volume * inputs.inv_volume;
    accumulateNonlocalSigmaForBand(
        work.upf,
        work.nb,
        n,
        inputs.gvecs,
        work.phase_buf,
        c,
        work.radial_vals,
        work.radial_derivs,
        work.p_buf,
        dij_for_band,
        dij_m_eff,
        work.m_offsets,
        work.m_counts,
        work.m_total,
        prefactor,
        sigma,
    );

    const e_nl_nk_raw = computeEnlForBand(
        work.upf,
        work.nb,
        work.p_buf,
        dij_for_band,
        dij_m_eff,
        work.m_offsets,
        work.m_counts,
        work.m_total,
    );
    const e_nl_nk = e_nl_nk_raw * inputs.inv_volume;
    const diag_contrib = -occ * kp.weight * inputs.spin_factor * e_nl_nk * inputs.inv_volume;
    for (0..3) |a| sigma[a][a] += diag_contrib;
}

/// Nonlocal pseudopotential stress.
/// σ_αβ = -(E_nl/Ω) δ_αβ - (2 spin/Ω²) Σ_nk f w Re[Σ D conj(dp_αβ) p]
/// where dp_αβ = Σ_G (∂φ/∂q_α × q_β) × S(G) × c(G)
pub fn nonlocalStress(
    alloc: std.mem.Allocator,
    wavefunctions: ?scf.WavefunctionData,
    species: []const hamiltonian.SpeciesEntry,
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

        const inputs = NonlocalStressInputs{
            .kp = kp,
            .species = species,
            .gvecs = gvecs,
            .n = n,
            .radial_tables_list = radial_tables_list,
            .paw_dij_per_atom = paw_dij_per_atom,
            .paw_dij_m_per_atom = paw_dij_m_per_atom,
            .paw_tabs = paw_tabs,
            .spin_factor = spin_factor,
            .inv_volume = inv_volume,
        };

        for (atoms, 0..) |atom, atom_idx| {
            try accumulateNonlocalStressAtom(alloc, inputs, atom, atom_idx, &sigma);
        }
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}
