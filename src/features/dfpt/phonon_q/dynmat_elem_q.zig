//! Complex dynamical-matrix element computations for q≠0 DFPT.
//!
//! Three pieces:
//!   - Electronic local contribution  D^elec_{Iα,Jβ} = Σ_G V^(1)*·ρ^(1) Ω
//!   - Nonlocal response contribution D^nl_resp
//!   - NLCC cross term                D^nlcc_cross
//!
//! All routines return complex dynamical matrix pieces that the caller
//! sums / Hermitianizes.

const std = @import("std");
const math = @import("../../math/math.zig");
const plane_wave = @import("../../plane_wave/basis.zig");
const scf_mod = @import("../../scf/scf.zig");

const dfpt = @import("../dfpt.zig");
const perturbation = dfpt.perturbation;
const dynmat_mod = dfpt.dynmat;
const GroundState = dfpt.GroundState;
const PerturbationResult = dfpt.PerturbationResult;

const Grid = scf_mod.Grid;

/// Compute the electronic contribution to the dynamical matrix element (complex, q≠0).
/// D^elec_{Iα,Jβ} = Σ_G conj(V^(1)_{Iα}(G)) × ρ^(1)_{Jβ}(G) × Ω
pub fn computeElecDynmatElementQ(
    vloc1_g: []const math.Complex,
    rho1_g: []const math.Complex,
    volume: f64,
) math.Complex {
    var sum = math.complex.init(0.0, 0.0);
    for (0..vloc1_g.len) |i| {
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(vloc1_g[i]), rho1_g[i]));
    }
    return math.complex.scale(sum, volume);
}

/// Compute the nonlocal response contribution to the dynamical matrix for q≠0.
/// D(Iα,Jβ) = 2 × Σ_n ⟨dV_nl_{Iα,q} ψ^(0)_n | δψ_{n,Jβ}⟩
/// Monochromatic convention: +q only. Hermitianization at total level.
pub fn computeNonlocalResponseDynmatQ(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    gvecs_kq: []const plane_wave.GVector,
    apply_ctx_kq: *scf_mod.ApplyContext,
    n_atoms: usize,
) ![]math.Complex {
    const dim = 3 * n_atoms;
    const n_pw_kq = gvecs_kq.len;
    const dyn = try alloc.alloc(math.Complex, dim * dim);
    @memset(dyn, math.complex.init(0.0, 0.0));

    const nl_ctx_k = gs.apply_ctx.nonlocal_ctx orelse return dyn;
    const nl_ctx_kq = apply_ctx_kq.nonlocal_ctx orelse return dyn;

    const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
    defer alloc.free(nl_out);

    // Compute D(Iα,Jβ) = 4 × Σ_n ⟨dV_nl_{Iα,q} ψ^(0)_n | δψ_{n,Jβ}⟩
    // Factor 4 = 2(spin) × 2(d²E/dτ₁dτ₂), matching q=0 convention.
    // All (I,J) pairs computed explicitly — no Hermitianization needed.
    for (0..dim) |i| {
        const ia = i / 3;
        const dir_a = i % 3;
        for (0..gs.n_occ) |n| {
            // Apply dV_nl_{Iα,q} to ψ^(0)_{n,k}: k-basis → k+q-basis
            try perturbation.applyNonlocalPerturbationQ(
                alloc,
                gs.gvecs,
                gvecs_kq,
                gs.atoms,
                nl_ctx_k,
                nl_ctx_kq,
                ia,
                dir_a,
                1.0 / gs.grid.volume,
                gs.wavefunctions[n],
                nl_out,
            );

            // Inner product with each δψ_{Jβ} in k+q space
            for (0..dim) |j| {
                var ip = math.complex.init(0.0, 0.0);
                for (0..n_pw_kq) |g| {
                    ip = math.complex.add(ip, math.complex.mul(
                        math.complex.conj(nl_out[g]),
                        pert_results[j].psi1[n][g],
                    ));
                }
                // Factor 4 = 2(spin) × 2(d²E/dτ₁dτ₂ = 2·E^(τ₁τ₂)),
                // matching ABINIT's d2nl convention: wtk × occ × two = 1 × 2 × 2 = 4.
                // All (I,J) pairs are computed explicitly, no Hermitianization needed.
                dyn[i * dim + j] = math.complex.add(dyn[i * dim + j], math.complex.scale(ip, 4.0));
            }
        }
    }

    return dyn;
}

/// Compute NLCC cross-term for q≠0.
/// D_NLCC_cross(Iα,Jβ) = ∫ V_xc^(1)[ρ^(1)_core,I](r) × conj(ρ^(1)_total,J(r)) dr
///
/// For LDA this reduces to ∫ f_xc × ρ^(1)_core,I × ρ^(1)_total,J dr.
/// For GGA, V_xc^(1) includes gradient-dependent terms.
/// For q≠0, ρ^(1)(r) is complex, so result is complex.
pub fn computeNlccCrossDynmatQ(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: GroundState,
    rho1_val_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    const dim = 3 * n_atoms;
    const total = grid.count();
    const dyn = try alloc.alloc(math.Complex, dim * dim);
    @memset(dyn, math.complex.init(0.0, 0.0));

    const dv = grid.volume / @as(f64, @floatFromInt(total));

    for (0..dim) |i| {
        // Get ρ^(1)_core,Iα(r) in complex form
        const rho1_core_i_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_core_i_copy);
        @memcpy(rho1_core_i_copy, rho1_core_gs[i]);
        const rho1_core_i_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_core_i_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_i_copy, rho1_core_i_r, null);

        // Build V_xc^(1)[ρ^(1)_core,I] using GGA-aware kernel
        const vxc1_core_i = try perturbation.buildXcPerturbationFullComplex(alloc, gs, rho1_core_i_r);
        defer alloc.free(vxc1_core_i);

        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            // Get ρ^(1)_val,Jβ(r)
            const rho1_val_g_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_val_g_copy);
            @memcpy(rho1_val_g_copy, rho1_val_gs[j]);
            const work_r_j = try alloc.alloc(math.Complex, total);
            defer alloc.free(work_r_j);
            try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_val_g_copy, work_r_j, null);

            // Get ρ^(1)_core,Jβ(r)
            const rho1_core_j_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_core_j_copy);
            @memcpy(rho1_core_j_copy, rho1_core_gs[j]);
            const rho1_core_j_r = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_core_j_r);
            try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_j_copy, rho1_core_j_r, null);

            // D(I,J) = ∫ conj(V_xc^(1)[ρ^(1)_core,I]) × ρ^(1)_total,J dr
            var sum = math.complex.init(0.0, 0.0);
            for (0..total) |r| {
                const rho1_total_j = math.complex.add(work_r_j[r], rho1_core_j_r[r]);
                const prod = math.complex.mul(math.complex.conj(vxc1_core_i[r]), rho1_total_j);
                sum = math.complex.add(sum, prod);
            }
            dyn[i * dim + j] = math.complex.scale(sum, dv);
        }
    }

    return dyn;
}
