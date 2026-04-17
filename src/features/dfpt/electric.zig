//! Electric field response: ddk perturbation for ε∞.
//!
//! Solves ddk Sternheimer at ALL k-points in the BZ.
//!
//! Self-consistent dielectric tensor (ABINIT rfelfd=3 equivalent):
//!   1) du/dk = -(H-ε)^{-1} P_c dH/dk |u⟩  (ddk Sternheimer)
//!   2) For each direction β, SCF loop:
//!      a) ψ¹_E = -(H-ε)^{-1} P_c (+i du/dk_β + V^(1)_Hxc |ψ⟩)
//!      b) ρ^(1) = Σ_k w_k × 2(spin) × 2 Re[ψ_k*(r) × ψ¹_E_k(r)]
//!      c) V^(1)_H = 8π ρ^(1)(G)/|G|²,  V^(1)_xc = f_xc × ρ^(1)(r)
//!      d) Mix V^(1)_Hxc, check convergence
//!   3) ε_αβ = δ_αβ - (16π/Ω) × occ × Σ w_k Re[<+i du/dk_α | ψ¹_{E,β}>]

const std = @import("std");
const math = @import("../math/math.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../scf/scf.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const config_mod = @import("../config/config.zig");

const dfpt = @import("dfpt.zig");
const sternheimer = dfpt.sternheimer;
const perturbation = dfpt.perturbation;
const phonon_q = dfpt.phonon_q;

const GroundState = dfpt.GroundState;
const DfptConfig = dfpt.DfptConfig;
const KPointGsData = phonon_q.KPointGsData;

const logDfpt = dfpt.logDfpt;

pub const DielectricResult = struct {
    epsilon: [3][3]f64,
};


/// Compute ε∞ by solving ddk Sternheimer at all k-points,
/// then self-consistently solving the efield Sternheimer with V^(1)_Hxc.
pub fn computeDielectricAllK(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    gs: *const GroundState,
    local_r: []const f64,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
) !DielectricResult {
    const grid = gs.grid;
    const total = grid.count();
    const dfpt_cfg = DfptConfig.fromConfig(cfg);

    const kgs = try phonon_q.prepareFullBZKpointsFromIBZ(alloc, io, cfg, gs, local_r, species, atoms, cell_bohr, recip, volume, grid);
    defer {
        for (@constCast(kgs)) |*k| k.deinit(alloc);
        alloc.free(kgs);
    }
    const n_kpts = kgs.len;
    logDfpt("ddk: {d} k-points in full BZ\n", .{n_kpts});

    // Build RadialTableSets once
    const n_species = species.len;
    const radial_tables = try alloc.alloc(nonlocal.RadialTableSet, n_species);
    defer {
        for (radial_tables) |*t| t.deinit(alloc);
        alloc.free(radial_tables);
    }
    {
        var g_max: f64 = 0.0;
        for (kgs) |kg| {
            for (kg.basis_k.gvecs) |gv| {
                const gnorm = math.Vec3.norm(gv.kpg);
                if (gnorm > g_max) g_max = gnorm;
            }
        }
        g_max += 1.0;
        for (0..n_species) |si| {
            radial_tables[si] = try nonlocal.RadialTableSet.init(
                alloc,
                species[si].upf.beta,
                species[si].upf.r,
                species[si].upf.rab,
                g_max,
            );
        }
    }

    // ================================================================
    // Phase 1: Solve ddk Sternheimer for ALL k-points, 3 directions.
    // Store psi1_ddk[k][dir][band][G].
    // ================================================================
    // Allocate psi1_ddk: [n_kpts][3][n_occ][n_pw]
    const psi1_ddk = try alloc.alloc([3][][]math.Complex, n_kpts);
    var kpts_built: usize = 0;
    defer {
        for (0..kpts_built) |ik| {
            for (0..3) |d| {
                for (psi1_ddk[ik][d]) |p| alloc.free(p);
                alloc.free(psi1_ddk[ik][d]);
            }
        }
        alloc.free(psi1_ddk);
    }

    for (kgs, 0..) |*kg, ik| {
        const n_occ = kg.n_occ;
        const n_pw = kg.n_pw_k;
        const gvecs = kg.basis_k.gvecs;

        const nl_ctx_opt = kg.apply_ctx_k.nonlocal_ctx;
        const max_m = if (nl_ctx_opt) |nc| nc.max_m_total else 0;

        const nl_out = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(nl_out);
        const nl_phase = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(nl_phase);
        const nl_c1 = try alloc.alloc(math.Complex, @max(max_m, 1));
        defer alloc.free(nl_c1);
        const nl_c2 = try alloc.alloc(math.Complex, @max(max_m, 1));
        defer alloc.free(nl_c2);
        const nl_dc1 = try alloc.alloc(math.Complex, @max(max_m, 1));
        defer alloc.free(nl_dc1);
        const nl_dc2 = try alloc.alloc(math.Complex, @max(max_m, 1));
        defer alloc.free(nl_dc2);

        for (0..3) |dir| {
            const psi1 = try alloc.alloc([]math.Complex, n_occ);
            var bands_built: usize = 0;
            errdefer {
                for (psi1[0..bands_built]) |p| alloc.free(p);
                alloc.free(psi1);
            }

            for (0..n_occ) |n| {
                const h1psi = try alloc.alloc(math.Complex, n_pw);
                defer alloc.free(h1psi);

                for (0..n_pw) |g| {
                    h1psi[g] = math.complex.scale(kg.wavefunctions_k[n][g], 2.0 * perturbation.gComponent(gvecs[g].kpg, dir));
                }
                if (nl_ctx_opt) |nl_ctx| {
                    applyDdkNonlocal(gvecs, atoms, nl_ctx, dir, 1.0 / volume, kg.wavefunctions_k[n], radial_tables, nl_out, nl_phase, nl_c1, nl_c2, nl_dc1, nl_dc2);
                    for (0..n_pw) |g| {
                        h1psi[g] = math.complex.add(h1psi[g], nl_out[g]);
                    }
                }

                const rhs = try alloc.alloc(math.Complex, n_pw);
                defer alloc.free(rhs);
                for (0..n_pw) |g| {
                    rhs[g] = math.complex.scale(h1psi[g], -1.0);
                }
                sternheimer.projectConduction(rhs, kg.wavefunctions_k_const, n_occ);

                const result = try sternheimer.solve(
                    alloc,
                    kg.apply_ctx_k,
                    rhs,
                    kg.eigenvalues_k[n],
                    kg.wavefunctions_k_const,
                    n_occ,
                    gvecs,
                    .{ .tol = dfpt_cfg.sternheimer_tol, .max_iter = dfpt_cfg.sternheimer_max_iter, .alpha_shift = dfpt_cfg.alpha_shift },
                );
                psi1[n] = result.psi1;
                bands_built = n + 1;
            }
            psi1_ddk[ik][dir] = psi1;
        }

        if (ik == 0 or (ik + 1) % 8 == 0 or ik + 1 == n_kpts) {
            logDfpt("ddk: k-point {d}/{d} done\n", .{ ik + 1, n_kpts });
        }
        kpts_built = ik + 1;
    }

    // ================================================================
    // Pre-cache ψ^(0)(r) for all k-points (IFFT once)
    // ================================================================
    const psi0_r_cache = try alloc.alloc([][]math.Complex, n_kpts);
    var psi0_cache_built: usize = 0;
    defer {
        for (0..psi0_cache_built) |ik| {
            for (psi0_r_cache[ik]) |p| alloc.free(p);
            alloc.free(psi0_r_cache[ik]);
        }
        alloc.free(psi0_r_cache);
    }

    for (kgs, 0..) |*kg, ik| {
        psi0_r_cache[ik] = try alloc.alloc([]math.Complex, kg.n_occ);
        var bands_built: usize = 0;
        errdefer {
            for (psi0_r_cache[ik][0..bands_built]) |p| alloc.free(p);
            alloc.free(psi0_r_cache[ik]);
        }
        for (0..kg.n_occ) |n| {
            psi0_r_cache[ik][n] = try alloc.alloc(math.Complex, total);
            const work = try alloc.alloc(math.Complex, total);
            defer alloc.free(work);
            @memset(work, math.complex.init(0.0, 0.0));
            kg.map_k.scatter(kg.wavefunctions_k_const[n], work);
            try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work, psi0_r_cache[ik][n], null);
            bands_built = n + 1;
        }
        psi0_cache_built = ik + 1;
    }

    // ================================================================
    // Phase 2: Self-consistent efield Sternheimer for each direction β.
    // ================================================================
    var epsilon: [3][3]f64 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };

    for (0..3) |beta| {
        logDfpt("efield SCF: direction β={d}\n", .{beta});

        // V^(1)_Hxc in reciprocal space (real potential → Hermitian in G)
        // For q=0 efield, V^(1) is real, so we work in real G-space representation.
        const v1_hxc_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(v1_hxc_g);
        @memset(v1_hxc_g, math.complex.init(0.0, 0.0));

        // Allocate ψ¹_E storage per k-point
        const psi1_ef = try alloc.alloc([][]math.Complex, n_kpts);
        var ef_kpts_built: usize = 0;
        defer {
            for (0..ef_kpts_built) |ik| {
                for (psi1_ef[ik]) |p| alloc.free(p);
                alloc.free(psi1_ef[ik]);
            }
            alloc.free(psi1_ef);
        }
        for (kgs, 0..) |*kg, ik| {
            psi1_ef[ik] = try alloc.alloc([]math.Complex, kg.n_occ);
            var bands_built: usize = 0;
            errdefer {
                for (psi1_ef[ik][0..bands_built]) |p| alloc.free(p);
                alloc.free(psi1_ef[ik]);
            }
            for (0..kg.n_occ) |n| {
                psi1_ef[ik][n] = try alloc.alloc(math.Complex, kg.n_pw_k);
                @memset(psi1_ef[ik][n], math.complex.init(0.0, 0.0));
                bands_built = n + 1;
            }
            ef_kpts_built = ik + 1;
        }

        // Pulay mixer for V^(1)_Hxc potential mixing
        var pulay = scf_mod.ComplexPulayMixer.init(alloc, dfpt_cfg.pulay_history);
        defer pulay.deinit();

        var best_vresid: f64 = std.math.inf(f64);
        var best_v1: ?[]math.Complex = null;
        defer if (best_v1) |v| alloc.free(v);
        var pulay_active_since: usize = dfpt_cfg.pulay_start;
        const restart_factor: f64 = 5.0;
        var force_converge: bool = false;

        // SCF loop
        var iter: usize = 0;
        while (iter < dfpt_cfg.scf_max_iter) : (iter += 1) {
            // IFFT V^(1)_Hxc(G) → V^(1)_Hxc(r) (real, but stored as complex for mul)
            const v1_g_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(v1_g_copy);
            @memcpy(v1_g_copy, v1_hxc_g);
            const v1_hxc_r = try alloc.alloc(math.Complex, total);
            defer alloc.free(v1_hxc_r);
            try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, v1_g_copy, v1_hxc_r, null);

            // Accumulate ρ^(1) over all k-points
            const rho1_r = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_r);
            @memset(rho1_r, math.complex.init(0.0, 0.0));

            for (kgs, 0..) |*kg, ik| {
                const n_occ = kg.n_occ;
                const n_pw = kg.n_pw_k;
                const gvecs = kg.basis_k.gvecs;
                const map_k_ptr: *const scf_mod.PwGridMap = &kg.map_k;

                for (0..n_occ) |n| {
                    // RHS = -P_c[+i × du/dk_β + V^(1)_Hxc |ψ_k⟩]
                    const rhs_ef = try alloc.alloc(math.Complex, n_pw);
                    defer alloc.free(rhs_ef);

                    // Term 1: +i × du/dk_β
                    // +i × (a+bi) = (-b+ai)
                    for (0..n_pw) |g| {
                        rhs_ef[g] = math.complex.init(
                            -psi1_ddk[ik][beta][n][g].i,
                            psi1_ddk[ik][beta][n][g].r,
                        );
                    }

                    // Term 2: V^(1)_Hxc |ψ_k⟩ (apply real-space potential)
                    // Use cached ψ^(0)(r), multiply by V^(1)(r), FFT back, gather
                    const v1psi = try phonon_q.applyV1PsiQCached(alloc, grid, map_k_ptr, v1_hxc_r, psi0_r_cache[ik][n], n_pw);
                    defer alloc.free(v1psi);

                    // Combine: rhs = -(term1 + term2)
                    for (0..n_pw) |g| {
                        rhs_ef[g] = math.complex.scale(
                            math.complex.add(rhs_ef[g], v1psi[g]),
                            -1.0,
                        );
                    }

                    // Project onto conduction bands
                    sternheimer.projectConduction(rhs_ef, kg.wavefunctions_k_const, n_occ);

                    // Solve Sternheimer
                    const ef_result = try sternheimer.solve(
                        alloc,
                        kg.apply_ctx_k,
                        rhs_ef,
                        kg.eigenvalues_k[n],
                        kg.wavefunctions_k_const,
                        n_occ,
                        gvecs,
                        .{ .tol = dfpt_cfg.sternheimer_tol, .max_iter = dfpt_cfg.sternheimer_max_iter, .alpha_shift = dfpt_cfg.alpha_shift },
                    );
                    // Store result
                    @memcpy(psi1_ef[ik][n], ef_result.psi1);
                    alloc.free(ef_result.psi1);
                }

                // Compute ρ^(1) for this k-point: 2(spin)×2(cc)×wtk/Ω × Σ_n ψ*(r)×ψ¹(r)
                const psi1_const = try alloc.alloc([]const math.Complex, n_occ);
                defer alloc.free(psi1_const);
                for (0..n_occ) |nn| psi1_const[nn] = psi1_ef[ik][nn];

                const psi0_r_const = try alloc.alloc([]const math.Complex, n_occ);
                defer alloc.free(psi0_r_const);
                for (0..n_occ) |nn| psi0_r_const[nn] = psi0_r_cache[ik][nn];

                const rho1_k_r = try phonon_q.computeRho1QCached(
                    alloc,
                    grid,
                    map_k_ptr,
                    psi0_r_const,
                    psi1_const,
                    n_occ,
                    kg.weight,
                );
                defer alloc.free(rho1_k_r);

                for (0..total) |i| {
                    rho1_r[i] = math.complex.add(rho1_r[i], rho1_k_r[i]);
                }
            }

            // FFT ρ^(1)(r) → ρ^(1)(G)
            const rho1_r_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_r_copy);
            @memcpy(rho1_r_copy, rho1_r);
            const rho1_g = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_g);
            try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, rho1_r_copy, rho1_g, null);

            // Build V^(1)_H(G) = 8π ρ^(1)(G) / |G|²  (G=0 → 0)
            const vh1_g = try perturbation.buildHartreePerturbation(alloc, grid, rho1_g);
            defer alloc.free(vh1_g);

            // Build V^(1)_xc(r) = f_xc × ρ^(1)(r) (no NLCC for efield: ρ^(1)_core=0)
            // ρ^(1) for efield at q=0 is real; extract real part for XC kernel
            const rho1_real = try alloc.alloc(f64, total);
            defer alloc.free(rho1_real);
            for (0..total) |i| {
                rho1_real[i] = rho1_r[i].r;
            }
            const vxc1_r = try perturbation.buildXcPerturbationFull(alloc, gs.*, rho1_real);
            defer alloc.free(vxc1_r);

            // FFT V^(1)_xc(r) → V^(1)_xc(G)
            const vxc1_g = try scf_mod.realToReciprocal(alloc, grid, vxc1_r, false);
            defer alloc.free(vxc1_g);

            // V^(1)_out(G) = V_H^(1) + V_xc^(1)
            const v_out_g = try alloc.alloc(math.Complex, total);
            defer alloc.free(v_out_g);
            for (0..total) |i| {
                v_out_g[i] = math.complex.add(vh1_g[i], vxc1_g[i]);
            }

            // Compute residual
            var residual_norm: f64 = 0.0;
            const residual = try alloc.alloc(math.Complex, total);
            for (0..total) |i| {
                residual[i] = math.complex.sub(v_out_g[i], v1_hxc_g[i]);
                residual_norm += residual[i].r * residual[i].r + residual[i].i * residual[i].i;
            }
            residual_norm = @sqrt(residual_norm);

            logDfpt("efield SCF β={d}: iter={d} vresid={e:.6}\n", .{ beta, iter, residual_norm });

            if (residual_norm < dfpt_cfg.scf_tol or (force_converge and residual_norm < 10.0 * dfpt_cfg.scf_tol)) {
                alloc.free(residual);
                logDfpt("efield SCF β={d}: converged at iter={d} vresid={e:.6}\n", .{ beta, iter, residual_norm });
                break;
            }

            // Track best residual
            if (residual_norm < best_vresid) {
                best_vresid = residual_norm;
                if (best_v1 == null) best_v1 = try alloc.alloc(math.Complex, total);
                @memcpy(best_v1.?, v1_hxc_g);
            }

            // Pulay restart check
            if (iter >= pulay_active_since and residual_norm > restart_factor * best_vresid and best_vresid < 1.0) {
                if (best_v1) |v| @memcpy(v1_hxc_g, v);
                if (best_vresid < 10.0 * dfpt_cfg.scf_tol) {
                    force_converge = true;
                    logDfpt("efield SCF β={d}: Pulay restart (near-converged) iter={d}\n", .{ beta, iter });
                    alloc.free(residual);
                    continue;
                }
                pulay.reset();
                pulay_active_since = iter + 1 + dfpt_cfg.pulay_start;
                logDfpt("efield SCF β={d}: Pulay restart iter={d} vresid={e:.6} best={e:.6}\n", .{ beta, iter, residual_norm, best_vresid });
                alloc.free(residual);
                continue;
            }

            // Mix V^(1)
            if (dfpt_cfg.pulay_history > 0 and iter >= pulay_active_since) {
                try pulay.mixWithResidual(v1_hxc_g, residual, dfpt_cfg.mixing_beta);
            } else {
                const mix_beta = dfpt_cfg.mixing_beta;
                for (0..total) |i| {
                    v1_hxc_g[i] = math.complex.add(v1_hxc_g[i], math.complex.scale(residual[i], mix_beta));
                }
                alloc.free(residual);
            }
        }

        // Reset Pulay state for next direction
        best_vresid = std.math.inf(f64);
        if (best_v1) |v| {
            alloc.free(v);
            best_v1 = null;
        }
        pulay.reset();
        pulay_active_since = dfpt_cfg.pulay_start;
        force_converge = false;

        // ================================================================
        // Accumulate ε_αβ from converged ψ¹_E for this β
        // ε_αβ -= (16π/Ω) × occ × Σ_k w_k × Re[<+i du/dk_α | ψ¹_E_β>]
        // ================================================================
        const occ: f64 = 2.0;
        for (kgs, 0..) |*kg, ik| {
            const n_occ = kg.n_occ;
            const n_pw = kg.n_pw_k;
            for (0..n_occ) |n| {
                for (0..3) |alpha| {
                    var overlap: f64 = 0.0;
                    for (0..n_pw) |g| {
                        // +i × du/dk_α = (+i)(a+bi) = (-b+ai)
                        const i_psi1_r = -psi1_ddk[ik][alpha][n][g].i;
                        const i_psi1_i = psi1_ddk[ik][alpha][n][g].r;
                        // Re[conj(+i psi1_α) × psi1_ef_β]
                        overlap += i_psi1_r * psi1_ef[ik][n][g].r + i_psi1_i * psi1_ef[ik][n][g].i;
                    }
                    epsilon[alpha][beta] -= (16.0 * std.math.pi / volume) * occ * kg.weight * overlap;
                }
            }
        }
    }

    for (0..3) |i| epsilon[i][i] += 1.0;

    // Symmetrize ε tensor using crystal point group.
    // Even with IBZ-expanded wavefunctions, the Sternheimer ddk perturbation
    // can break point-group invariance due to finite-difference derivatives
    // of the nonlocal potential. Symmetrize: ε_sym = (1/N) Σ_R R^T ε R.
    {
        const symmetry_mod = @import("../symmetry/symmetry.zig");
        const symops_eps = symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5) catch null;

        if (symops_eps) |ops| {
            defer alloc.free(ops);
            var eps_sym: [3][3]f64 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
            const inv2pi = 1.0 / (2.0 * std.math.pi);
            for (ops) |op| {
                var rc: [3][3]f64 = undefined;
                for (0..3) |i| {
                    for (0..3) |j| {
                        var s: f64 = 0.0;
                        for (0..3) |k2| {
                            for (0..3) |l2| {
                                s += cell_bohr.m[k2][i] * @as(f64, @floatFromInt(op.rot.m[k2][l2])) * recip.m[l2][j] * inv2pi;
                            }
                        }
                        rc[i][j] = s;
                    }
                }
                for (0..3) |i| {
                    for (0..3) |j| {
                        var val: f64 = 0.0;
                        for (0..3) |a2| {
                            for (0..3) |b2| {
                                val += rc[a2][i] * epsilon[a2][b2] * rc[b2][j];
                            }
                        }
                        eps_sym[i][j] += val;
                    }
                }
            }
            const nsym_f: f64 = @floatFromInt(ops.len);
            for (0..3) |i| {
                for (0..3) |j| {
                    eps_sym[i][j] /= nsym_f;
                }
            }
            epsilon = eps_sym;
        }
    }

    return .{ .epsilon = epsilon };
}

/// Compute nonlocal projector value: φ(q) = 4π f(|q|) Y_lm(q̂)
/// Uses nonlocal.zig's Y_lm to be consistent with computeDphiBeta.
fn computePhiBeta(kpg: math.Vec3, l: i32, m: i32, table: *const nonlocal.RadialTable) f64 {
    const r = math.Vec3.norm(kpg);
    return 4.0 * std.math.pi * table.eval(r) * nonlocal.realSphericalHarmonic(l, m, kpg.x, kpg.y, kpg.z);
}

/// Analytical derivative of the nonlocal projector:
///   φ(q) = 4π f(|q|) Y_lm(q̂)
///   dφ/dq_α = 4π [f'(|q|) (q_α/|q|) Y_lm(q̂) + f(|q|) dY_lm(q̂)/dq_α]
///
/// where dY_lm(q̂)/dq_α = Σ_β (∂Y_lm/∂n_β)(δ_αβ - n_α n_β)/|q|
fn computeDphiBeta(kpg: math.Vec3, direction: usize, l: i32, m: i32, table: *const nonlocal.RadialTable) f64 {
    const four_pi = 4.0 * std.math.pi;
    const r2 = math.Vec3.dot(kpg, kpg);

    if (r2 < 1e-30) {
        // At q=0: only l=1 has non-zero derivative
        if (l == 1) {
            // At q=0 for l=1: analytical limit is singular, use FD fallback
            const delta: f64 = 1e-5;
            var qp = kpg;
            var qm = kpg;
            switch (direction) {
                0 => {
                    qp.x += delta;
                    qm.x -= delta;
                },
                1 => {
                    qp.y += delta;
                    qm.y -= delta;
                },
                2 => {
                    qp.z += delta;
                    qm.z -= delta;
                },
                else => {},
            }
            return four_pi * (table.eval(math.Vec3.norm(qp)) * nonlocal.realSphericalHarmonic(l, m, qp.x, qp.y, qp.z) -
                table.eval(math.Vec3.norm(qm)) * nonlocal.realSphericalHarmonic(l, m, qm.x, qm.y, qm.z)) / (2.0 * delta);
        }
        return 0.0;
    }

    const r = @sqrt(r2);
    const inv_r = 1.0 / r;
    const nx = kpg.x * inv_r;
    const ny = kpg.y * inv_r;
    const nz = kpg.z * inv_r;
    const n_alpha = switch (direction) {
        0 => nx,
        1 => ny,
        2 => nz,
        else => 0.0,
    };

    const f_val = table.eval(r);
    const fp_val = table.evalDeriv(r);
    const ylm = nonlocal.realSphericalHarmonic(l, m, kpg.x, kpg.y, kpg.z);

    // Term 1: f'(|q|) * (q_α/|q|) * Y_lm(q̂)
    const term1 = fp_val * n_alpha * ylm;

    // Term 2: f(|q|) * dY_lm(q̂)/dq_α
    // dY_lm/dq_α = (1/|q|) * Σ_β (∂Y_lm/∂n_β) * (δ_αβ - n_α * n_β)
    //            = (1/|q|) * [∂Y_lm/∂n_α - n_α * Σ_β n_β * ∂Y_lm/∂n_β]
    const grad = dYlm_dn(l, m, nx, ny, nz);
    const dot_n_grad = nx * grad[0] + ny * grad[1] + nz * grad[2];
    const dylm_dq = inv_r * (grad[direction] - n_alpha * dot_n_grad);
    const term2 = f_val * dylm_dq;

    return four_pi * (term1 + term2);
}

/// Partial derivatives of real spherical harmonics with respect to n_x, n_y, n_z.
/// Returns [∂Y_lm/∂n_x, ∂Y_lm/∂n_y, ∂Y_lm/∂n_z].
fn dYlm_dn(l: i32, m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    const pi = std.math.pi;
    switch (l) {
        0 => return .{ 0.0, 0.0, 0.0 },
        1 => {
            // Y_1^{-1} = c*ny, Y_1^0 = c*nz, Y_1^1 = c*nx
            const c = @sqrt(3.0 / (4.0 * pi));
            return switch (m) {
                -1 => .{ 0.0, c, 0.0 },
                0 => .{ 0.0, 0.0, c },
                1 => .{ c, 0.0, 0.0 },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        2 => {
            const c0 = @sqrt(5.0 / (16.0 * pi));
            const c1 = @sqrt(15.0 / (4.0 * pi));
            const c2 = @sqrt(15.0 / (16.0 * pi));
            return switch (m) {
                // Y_2^{-2} = c2 * 2*nx*ny
                -2 => .{ c2 * 2.0 * ny, c2 * 2.0 * nx, 0.0 },
                // Y_2^{-1} = c1 * ny*nz
                -1 => .{ 0.0, c1 * nz, c1 * ny },
                // Y_2^0 = c0 * (3*nz²-1)
                0 => .{ 0.0, 0.0, c0 * 6.0 * nz },
                // Y_2^1 = c1 * nx*nz
                1 => .{ c1 * nz, 0.0, c1 * nx },
                // Y_2^2 = c2 * (nx²-ny²)
                2 => .{ c2 * 2.0 * nx, c2 * (-2.0) * ny, 0.0 },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        3 => {
            const c3m = @sqrt(35.0 / (32.0 * pi));
            const c2m = @sqrt(105.0 / (4.0 * pi));
            const c1m = @sqrt(21.0 / (32.0 * pi));
            const c0v = @sqrt(7.0 / (16.0 * pi));
            const c2p = @sqrt(105.0 / (16.0 * pi));
            const c3p = @sqrt(35.0 / (32.0 * pi));
            return switch (m) {
                // Y_3^{-3} = c3m * (3nx²-ny²)*ny
                -3 => .{
                    c3m * 6.0 * nx * ny,
                    c3m * (3.0 * nx * nx - 3.0 * ny * ny),
                    0.0,
                },
                // Y_3^{-2} = c2m * nx*ny*nz
                -2 => .{ c2m * ny * nz, c2m * nx * nz, c2m * nx * ny },
                // Y_3^{-1} = c1m * ny*(5nz²-1)
                -1 => .{ 0.0, c1m * (5.0 * nz * nz - 1.0), c1m * 10.0 * ny * nz },
                // Y_3^0 = c0v * (5nz³-3nz)
                0 => .{ 0.0, 0.0, c0v * (15.0 * nz * nz - 3.0) },
                // Y_3^1 = c1m * nx*(5nz²-1)  (same c as m=-1)
                1 => .{ c1m * (5.0 * nz * nz - 1.0), 0.0, c1m * 10.0 * nx * nz },
                // Y_3^2 = c2p * (nx²-ny²)*nz
                2 => .{ c2p * 2.0 * nx * nz, c2p * (-2.0) * ny * nz, c2p * (nx * nx - ny * ny) },
                // Y_3^3 = c3p * (nx²-3ny²)*nx
                3 => .{
                    c3p * (3.0 * nx * nx - 3.0 * ny * ny),
                    c3p * (-6.0) * nx * ny,
                    0.0,
                },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        else => return .{ 0.0, 0.0, 0.0 },
    }
}

fn applyDdkNonlocal(
    gvecs: []const plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    nl_ctx: scf_mod.NonlocalContext,
    direction: usize,
    inv_volume: f64,
    psi: []const math.Complex,
    radial_tables: []const nonlocal.RadialTableSet,
    out: []math.Complex,
    work_phase: []math.Complex,
    work_coeff: []math.Complex,
    work_coeff2: []math.Complex,
    work_dcoeff: []math.Complex,
    work_dcoeff2: []math.Complex,
) void {
    const n_pw = gvecs.len;
    @memset(out, math.complex.init(0.0, 0.0));

    for (nl_ctx.species) |sp| {
        const g_count = sp.g_count;
        if (g_count != n_pw) continue;
        if (sp.m_total == 0) continue;

        for (atoms) |atom| {
            if (atom.species_index != sp.species_index) continue;
            for (0..n_pw) |g| {
                work_phase[g] = math.complex.expi(math.Vec3.dot(gvecs[g].cart, atom.position));
            }

            var b: usize = 0;
            while (b < sp.beta_count) : (b += 1) {
                const l_val = sp.l_list[b];
                const offset = sp.m_offsets[b];
                const m_count = sp.m_counts[b];
                const table = &radial_tables[atom.species_index].tables[b];
                var m_idx: usize = 0;
                while (m_idx < m_count) : (m_idx += 1) {
                    const m_val = @as(i32, @intCast(m_idx)) - l_val;
                    var coeff = math.complex.init(0.0, 0.0);
                    var dcoeff = math.complex.init(0.0, 0.0);
                    for (0..n_pw) |g| {
                        const xphase = math.complex.mul(psi[g], work_phase[g]);
                        // Use fresh phi consistent with computeDphiBeta's Y_lm
                        const phi_val = computePhiBeta(gvecs[g].kpg, l_val, m_val, table);
                        coeff = math.complex.add(coeff, math.complex.scale(xphase, phi_val));
                        dcoeff = math.complex.add(dcoeff, math.complex.scale(xphase, computeDphiBeta(gvecs[g].kpg, direction, l_val, m_val, table)));
                    }
                    work_coeff[offset + m_idx] = math.complex.scale(coeff, inv_volume);
                    work_dcoeff[offset + m_idx] = math.complex.scale(dcoeff, inv_volume);
                }
            }
            applyDij(sp, work_coeff, work_coeff2);
            applyDij(sp, work_dcoeff, work_dcoeff2);

            b = 0;
            while (b < sp.beta_count) : (b += 1) {
                const l_val = sp.l_list[b];
                const offset = sp.m_offsets[b];
                const m_count = sp.m_counts[b];
                const table = &radial_tables[atom.species_index].tables[b];
                var m_idx: usize = 0;
                while (m_idx < m_count) : (m_idx += 1) {
                    const m_val = @as(i32, @intCast(m_idx)) - l_val;
                    const dc = work_coeff2[offset + m_idx];
                    const ddc = work_dcoeff2[offset + m_idx];
                    if (@abs(dc.r) + @abs(dc.i) + @abs(ddc.r) + @abs(ddc.i) < 1e-30) continue;
                    for (0..n_pw) |g| {
                        const phase_conj = math.complex.conj(work_phase[g]);
                        const phi_val = computePhiBeta(gvecs[g].kpg, l_val, m_val, table);
                        const dphi = computeDphiBeta(gvecs[g].kpg, direction, l_val, m_val, table);
                        out[g] = math.complex.add(out[g], math.complex.add(
                            math.complex.mul(math.complex.scale(phase_conj, dphi), dc),
                            math.complex.mul(math.complex.scale(phase_conj, phi_val), ddc),
                        ));
                    }
                }
            }
        }
    }
}

fn applyDij(sp: scf_mod.NonlocalSpecies, input: []const math.Complex, output: []math.Complex) void {
    var b: usize = 0;
    while (b < sp.beta_count) : (b += 1) {
        const l_val = sp.l_list[b];
        const offset = sp.m_offsets[b];
        const m_count = sp.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            var sum = math.complex.init(0.0, 0.0);
            var j: usize = 0;
            while (j < sp.beta_count) : (j += 1) {
                if (sp.l_list[j] != l_val) continue;
                const dij = sp.coeffs[b * sp.beta_count + j];
                if (dij == 0.0) continue;
                sum = math.complex.add(sum, math.complex.scale(input[sp.m_offsets[j] + m_idx], dij));
            }
            output[offset + m_idx] = sum;
        }
    }
}

pub fn writeElectricResults(
    io: std.Io,
    dir: std.Io.Dir,
    dielectric: DielectricResult,
) !void {
    const file = try dir.createFile(io, "electric.dat", .{});
    defer file.close(io);
    var buf: [1024]u8 = undefined;
    var writer = file.writer(io, &buf);
    const out = &writer.interface;

    try out.print("# Dielectric tensor epsilon_inf\n", .{});
    for (0..3) |i| {
        try out.print("{d:12.6} {d:12.6} {d:12.6}\n", .{
            dielectric.epsilon[i][0], dielectric.epsilon[i][1], dielectric.epsilon[i][2],
        });
    }

    try out.flush();
}
