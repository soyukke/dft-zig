//! Cross-basis operations for q≠0 DFPT.
//!
//! Applies perturbation potentials between k-basis and k+q-basis wavefunctions
//! and computes the first-order density response ρ^(1) from those cross-basis
//! products. Every routine here is an FFT-bounded building block used by the
//! q≠0 Sternheimer / SCF solver.

const std = @import("std");
const math = @import("../../math/math.zig");
const scf_mod = @import("../../scf/scf.zig");

const Grid = scf_mod.Grid;

/// Apply V^(1)(r)|ψ⟩ with complex V^(1) and cross-basis (k → k+q).
/// Scatters k-basis coefficients to grid, IFFTs, multiplies by V^(1)(r),
/// FFTs back, and gathers to k+q-basis.
pub fn apply_v1_psi_q(
    alloc: std.mem.Allocator,
    grid: Grid,
    map_k: *const scf_mod.PwGridMap,
    map_kq: *const scf_mod.PwGridMap,
    v1_r_complex: []const math.Complex,
    psi_k: []const math.Complex,
    n_pw_k: usize,
    n_pw_kq: usize,
) ![]math.Complex {
    const total = grid.count();

    // Scatter k-basis PW coefficients to full grid
    const work_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g);

    @memset(work_g, math.complex.init(0.0, 0.0));
    map_k.scatter(psi_k, work_g);

    // IFFT to real space
    const work_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r);

    try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, work_g, work_r, null);

    // Multiply by V^(1)(r) (complex × complex)
    for (0..total) |i| {
        work_r[i] = math.complex.mul(work_r[i], v1_r_complex[i]);
    }

    // FFT back to reciprocal space
    const work_g_out = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g_out);

    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, grid, work_r, work_g_out, null);

    // Gather to k+q-basis
    const result = try alloc.alloc(math.Complex, n_pw_kq);
    map_kq.gather(work_g_out, result);
    _ = n_pw_k;

    return result;
}

/// Cached variant of apply_v1_psi_q that uses pre-computed ψ^(0)(r) in real space.
/// Skips the scatter + IFFT step for ψ^(0), saving one FFT per band per SCF iteration.
pub fn apply_v1_psi_q_cached(
    alloc: std.mem.Allocator,
    grid: Grid,
    map_kq: *const scf_mod.PwGridMap,
    v1_r_complex: []const math.Complex,
    psi0_r: []const math.Complex,
    n_pw_kq: usize,
) ![]math.Complex {
    const total = grid.count();

    // Multiply ψ^(0)(r) by V^(1)(r) (complex × complex)
    const work_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r);

    for (0..total) |i| {
        work_r[i] = math.complex.mul(psi0_r[i], v1_r_complex[i]);
    }

    // FFT back to reciprocal space
    const work_g_out = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g_out);

    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, grid, work_r, work_g_out, null);

    // Gather to k+q-basis
    const result = try alloc.alloc(math.Complex, n_pw_kq);
    map_kq.gather(work_g_out, result);

    return result;
}

/// Compute first-order density response for q≠0 from cross-basis wavefunctions.
/// ρ^(1)(r) = (4 × wtk / Ω) × Σ_n ψ^(0)*_{n,k}(r) × ψ^(1)_{n,k,q}(r)
/// Weight = 4×wtk/Ω: factor 2 from spin (occ=2), factor 2 from c.c. (2n+1 theorem),
/// wtk from k-point weight, 1/Ω from normalization.
/// Matches ABINIT's dfpt_mkrho convention:
/// weight = two * occ * wtk / ucvol.
/// For single k-point (Γ only), wtk=1.0 gives the original 4/Ω.
pub fn compute_rho1_q(
    alloc: std.mem.Allocator,
    grid: Grid,
    map_k: *const scf_mod.PwGridMap,
    map_kq: *const scf_mod.PwGridMap,
    psi0_k: []const []const math.Complex,
    psi1_kq: []const []const math.Complex,
    n_occ: usize,
    n_pw_k: usize,
    n_pw_kq: usize,
    wtk: f64,
) ![]math.Complex {
    const total = grid.count();
    // ρ^(1)(r) is complex for q≠0
    const rho1_r = try alloc.alloc(math.Complex, total);
    @memset(rho1_r, math.complex.init(0.0, 0.0));

    const work_g0 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g0);

    const work_r0 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r0);

    const work_g1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g1);

    const work_r1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r1);

    // Weight: 4×wtk/Ω = 2(spin/occ) × 2(c.c./2n+1) × wtk(k-point) × (1/Ω)(normalization)
    // Matches ABINIT dfpt_mkrho: weight = two * occ_k * wtk_k / ucvol
    const weight = 4.0 * wtk / grid.volume;

    for (0..n_occ) |n| {
        // ψ^(0)(r) via IFFT using k-basis map
        @memset(work_g0, math.complex.init(0.0, 0.0));
        map_k.scatter(psi0_k[n], work_g0);
        try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, work_g0, work_r0, null);

        // ψ^(1)(r) via IFFT using k+q-basis map
        @memset(work_g1, math.complex.init(0.0, 0.0));
        map_kq.scatter(psi1_kq[n], work_g1);
        try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, work_g1, work_r1, null);

        // ρ^(1)(r) += (4×wtk/Ω) × ψ^(0)*(r) × ψ^(1)(r)  [complex]
        for (0..total) |i| {
            const conj0 = math.complex.conj(work_r0[i]);
            const prod = math.complex.mul(conj0, work_r1[i]);
            rho1_r[i] = math.complex.add(rho1_r[i], math.complex.scale(prod, weight));
        }
    }
    _ = n_pw_k;
    _ = n_pw_kq;

    return rho1_r;
}

/// Cached variant of compute_rho1_q that uses pre-computed ψ^(0)(r) in real space.
/// Skips the scatter + IFFT for ψ^(0) each band, saving n_occ FFTs per call.
pub fn compute_rho1_q_cached(
    alloc: std.mem.Allocator,
    grid: Grid,
    map_kq: *const scf_mod.PwGridMap,
    psi0_r_cache: []const []const math.Complex,
    psi1_kq: []const []const math.Complex,
    n_occ: usize,
    wtk: f64,
) ![]math.Complex {
    const total = grid.count();
    const rho1_r = try alloc.alloc(math.Complex, total);
    @memset(rho1_r, math.complex.init(0.0, 0.0));

    const work_g1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g1);

    const work_r1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r1);

    const weight = 4.0 * wtk / grid.volume;

    for (0..n_occ) |n| {
        // ψ^(0)(r) — use cached version (no IFFT needed)
        const work_r0 = psi0_r_cache[n];

        // ψ^(1)(r) via IFFT using k+q-basis map
        @memset(work_g1, math.complex.init(0.0, 0.0));
        map_kq.scatter(psi1_kq[n], work_g1);
        try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, work_g1, work_r1, null);

        // ρ^(1)(r) += (4×wtk/Ω) × ψ^(0)*(r) × ψ^(1)(r)  [complex]
        for (0..total) |i| {
            const conj0 = math.complex.conj(work_r0[i]);
            const prod = math.complex.mul(conj0, work_r1[i]);
            rho1_r[i] = math.complex.add(rho1_r[i], math.complex.scale(prod, weight));
        }
    }

    return rho1_r;
}

/// Compute ρ^(1)(G) from ρ^(1)(r) complex via FFT.
pub fn complex_real_to_reciprocal(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho1_r: []math.Complex,
) ![]math.Complex {
    const total = grid.count();
    const out = try alloc.alloc(math.Complex, total);
    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, grid, rho1_r, out, null);
    return out;
}
