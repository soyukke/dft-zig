//! First-order perturbation potentials for DFPT.
//!
//! Provides V_loc^(1), V_H^(1), V_xc^(1), and V_nl^(1) for atomic displacement perturbations.

const std = @import("std");
const math = @import("../math/math.zig");
const scf_mod = @import("../scf/scf.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const xc = @import("../xc/xc.zig");
const dfpt_mod = @import("dfpt.zig");
const test_support = @import("../../test_support.zig");

const Grid = scf_mod.Grid;
const GroundState = dfpt_mod.GroundState;

/// Perturbation specification: atom index and Cartesian direction.
pub const Perturbation = struct {
    atom_index: usize,
    direction: usize, // 0=x, 1=y, 2=z
};

/// Build V_loc^(1)(G) for a single-atom displacement perturbation.
///
/// V_loc^(1)(G) = -i × G_α × V_form(|G|) × exp(-iG·τ_I) / Ω
///
/// Returns a complex array over the full FFT grid.
pub fn build_local_perturbation(
    alloc: std.mem.Allocator,
    grid: Grid,
    atom: hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
    direction: usize,
    local_cfg: local_potential.LocalPotentialConfig,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
) ![]math.Complex {
    const total = grid.count();
    const result = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(result);

    const inv_volume = 1.0 / grid.volume;

    var it = scf_mod.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.gh == 0 and g.gk == 0 and g.gl == 0) {
            result[g.idx] = math.complex.init(0.0, 0.0);
            continue;
        }

        const g_norm = math.Vec3.norm(g.gvec);
        const g_alpha = g_component(g.gvec, direction);

        // V_form(|G|)
        const v_loc = if (ff_tables) |tables|
            tables[atom.species_index].eval(g_norm)
        else
            hamiltonian.local_form_factor(&species[atom.species_index], g_norm, local_cfg);

        // exp(-iG·τ_I)
        const phase = math.complex.expi(-math.Vec3.dot(g.gvec, atom.position));

        // V_loc^(1)(G) = -i × G_α × V_form × exp(-iG·τ) / Ω
        // -i × z = -i × (a + bi) = (b - ai)
        const temp = math.complex.scale(phase, g_alpha * v_loc * inv_volume);
        result[g.idx] = math.complex.init(temp.i, -temp.r); // multiply by -i
    }

    return result;
}

/// Build ρ^(1)_core(G) for a single-atom displacement perturbation (NLCC).
///
/// ρ^(1)_core(G) = -i × G_α × ρ_core_form(|G|) × exp(-iG·τ_I) / Ω
///
/// Same structure as V_loc^(1) but uses the core charge form factor.
/// Returns a complex array over the full FFT grid.
pub fn build_core_perturbation(
    alloc: std.mem.Allocator,
    grid: Grid,
    atom: hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
    direction: usize,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) ![]math.Complex {
    const total = grid.count();
    const result = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(result);

    // Check if this species has NLCC
    const sp = &species[atom.species_index];
    if (sp.upf.nlcc.len == 0) {
        @memset(result, math.complex.init(0.0, 0.0));
        return result;
    }

    const inv_volume = 1.0 / grid.volume;

    var it = scf_mod.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.gh == 0 and g.gk == 0 and g.gl == 0) {
            result[g.idx] = math.complex.init(0.0, 0.0);
            continue;
        }

        const g_norm = math.Vec3.norm(g.gvec);
        const g_alpha = g_component(g.gvec, direction);

        // ρ_core_form(|G|)
        const rho_core_g = if (rho_core_tables) |tables|
            tables[atom.species_index].eval(g_norm)
        else
            form_factor.rho_core_g(sp.upf.*, g_norm);

        // exp(-iG·τ_I)
        const phase = math.complex.expi(-math.Vec3.dot(g.gvec, atom.position));

        // ρ^(1)_core(G) = -i × G_α × ρ_core_form × exp(-iG·τ) / Ω
        const temp = math.complex.scale(phase, g_alpha * rho_core_g * inv_volume);
        result[g.idx] = math.complex.init(temp.i, -temp.r); // multiply by -i
    }

    return result;
}

/// Build V_H^(1)(G) = 8π × ρ^(1)(G) / G²  (Rydberg, G≠0).
pub fn build_hartree_perturbation(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho1_g: []const math.Complex,
) ![]math.Complex {
    const total = grid.count();
    const result = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(result);

    var it = scf_mod.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.g2 > 1e-12) {
            result[g.idx] = math.complex.scale(rho1_g[g.idx], 8.0 * std.math.pi / g.g2);
        } else {
            result[g.idx] = math.complex.init(0.0, 0.0);
        }
    }

    return result;
}

/// Build V_xc^(1)(r) = f_xc(r) × ρ^(1)(r) in real space.
pub fn build_xc_perturbation(
    alloc: std.mem.Allocator,
    fxc_r: []const f64,
    rho1_r: []const f64,
) ![]f64 {
    const n = fxc_r.len;
    const result = try alloc.alloc(f64, n);
    for (result, 0..) |*v, i| {
        v.* = fxc_r[i] * rho1_r[i];
    }
    return result;
}

/// Build V_xc^(1)(r) for both LDA and GGA functionals (real, γ-point version).
/// For LDA: V_xc^(1) = f_xc × n¹
/// For GGA/PBE: V_xc^(1) = f_nn·n¹ + f_nσ·σ¹
///                         - 2·∇·[f_nσ·n¹·∇n₀ + f_σσ·σ¹·∇n₀ + v_σ·∇n¹]
///   where σ¹ = 2·(∇n₀·∇n¹)
pub fn build_xc_perturbation_full(
    alloc: std.mem.Allocator,
    gs: GroundState,
    rho1_r: []const f64,
) ![]f64 {
    if (gs.xc_func == .lda_pz) {
        return build_xc_perturbation(alloc, gs.fxc_r, rho1_r);
    }

    // GGA (PBE) path
    const grid = gs.grid;
    const total = grid.count();
    const f_nn = gs.fxc_r;
    const f_ns = gs.fxc_ns_r.?;
    const f_ss = gs.fxc_ss_r.?;
    const v_s = gs.v_sigma_r.?;
    const gn0_x = gs.grad_n0_x.?;
    const gn0_y = gs.grad_n0_y.?;
    const gn0_z = gs.grad_n0_z.?;

    // 1. ∇n¹ = gradient_from_real(n¹)
    var grad1 = try scf_mod.gradient_from_real(alloc, grid, rho1_r, false);
    defer grad1.deinit(alloc);

    // 2. σ¹(r) = 2·Σ_α (∂n₀/∂x_α)(∂n¹/∂x_α)
    // 3. direct = f_nn·n¹ + f_nσ·σ¹
    // 4. A_α = f_nσ·n¹·(∂n₀/∂x_α) + f_σσ·σ¹·(∂n₀/∂x_α)
    //          + v_σ·(∂n¹/∂x_α)
    const ax = try alloc.alloc(f64, total);
    defer alloc.free(ax);

    const ay = try alloc.alloc(f64, total);
    defer alloc.free(ay);

    const az = try alloc.alloc(f64, total);
    defer alloc.free(az);

    const direct = try alloc.alloc(f64, total);
    errdefer alloc.free(direct);

    for (0..total) |i| {
        const sigma1 = 2.0 *
            (gn0_x[i] * grad1.x[i] + gn0_y[i] * grad1.y[i] + gn0_z[i] * grad1.z[i]);
        direct[i] = f_nn[i] * rho1_r[i] + f_ns[i] * sigma1;

        const fns_n1 = f_ns[i] * rho1_r[i];
        const fss_s1 = f_ss[i] * sigma1;
        ax[i] = (fns_n1 + fss_s1) * gn0_x[i] + v_s[i] * grad1.x[i];
        ay[i] = (fns_n1 + fss_s1) * gn0_y[i] + v_s[i] * grad1.y[i];
        az[i] = (fns_n1 + fss_s1) * gn0_z[i] + v_s[i] * grad1.z[i];
    }

    // 5. div_A = divergence_from_real(A_x, A_y, A_z)
    const div_a = try scf_mod.divergence_from_real(alloc, grid, ax, ay, az, false);
    defer alloc.free(div_a);

    // 6. result = direct - 2·div_A
    for (0..total) |i| {
        direct[i] -= 2.0 * div_a[i];
    }

    return direct;
}

/// Build V_xc^(1)(r) for both LDA and GGA functionals (complex, q≠0 version).
/// For LDA: V_xc^(1) = f_xc × n¹
/// For GGA/PBE: same formula as real version, but n¹(r) is complex.
pub fn build_xc_perturbation_full_complex(
    alloc: std.mem.Allocator,
    gs: GroundState,
    rho1_r: []const math.Complex,
) ![]math.Complex {
    if (gs.xc_func == .lda_pz) {
        return build_xc_perturbation_complex(alloc, gs.fxc_r, rho1_r);
    }

    // GGA (PBE) path — complex version
    const grid = gs.grid;
    const total = grid.count();
    const f_nn = gs.fxc_r;
    const f_ns = gs.fxc_ns_r.?;
    const f_ss = gs.fxc_ss_r.?;
    const v_s = gs.v_sigma_r.?;
    const gn0_x = gs.grad_n0_x.?;
    const gn0_y = gs.grad_n0_y.?;
    const gn0_z = gs.grad_n0_z.?;

    // 1. ∇n¹ = gradient_from_complex(n¹)
    var grad1 = try gradient_from_complex(alloc, grid, rho1_r);
    defer grad1.deinit(alloc);

    // Compute all terms
    const ax = try alloc.alloc(math.Complex, total);
    defer alloc.free(ax);

    const ay = try alloc.alloc(math.Complex, total);
    defer alloc.free(ay);

    const az = try alloc.alloc(math.Complex, total);
    defer alloc.free(az);

    const result = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(result);

    for (0..total) |i| {
        // σ¹ = 2·(∇n₀·∇n¹) — ∇n₀ is real, ∇n¹ is complex
        const sigma1 = math.complex.scale(
            math.complex.add(
                math.complex.add(
                    math.complex.scale(grad1.x[i], gn0_x[i]),
                    math.complex.scale(grad1.y[i], gn0_y[i]),
                ),
                math.complex.scale(grad1.z[i], gn0_z[i]),
            ),
            2.0,
        );

        // direct = f_nn·n¹ + f_nσ·σ¹
        result[i] = math.complex.add(
            math.complex.scale(rho1_r[i], f_nn[i]),
            math.complex.scale(sigma1, f_ns[i]),
        );

        // A_α = (f_nσ·n¹ + f_σσ·σ¹)·∂n₀/∂x_α + v_σ·∂n¹/∂x_α
        const fns_n1 = math.complex.scale(rho1_r[i], f_ns[i]);
        const fss_s1 = math.complex.scale(sigma1, f_ss[i]);
        const coeff = math.complex.add(fns_n1, fss_s1);
        ax[i] = math.complex.add(
            math.complex.scale(coeff, gn0_x[i]),
            math.complex.scale(grad1.x[i], v_s[i]),
        );
        ay[i] = math.complex.add(
            math.complex.scale(coeff, gn0_y[i]),
            math.complex.scale(grad1.y[i], v_s[i]),
        );
        az[i] = math.complex.add(
            math.complex.scale(coeff, gn0_z[i]),
            math.complex.scale(grad1.z[i], v_s[i]),
        );
    }

    // 5. div_A = divergence_from_complex(A_x, A_y, A_z)
    const div_a = try divergence_from_complex(alloc, grid, ax, ay, az);
    defer alloc.free(div_a);

    // 6. result -= 2·div_A
    for (0..total) |i| {
        result[i] = math.complex.sub(result[i], math.complex.scale(div_a[i], 2.0));
    }

    return result;
}

/// Complex gradient: ∇f(r) from complex f(r) on FFT grid.
const ComplexGradient = struct {
    x: []math.Complex,
    y: []math.Complex,
    z: []math.Complex,

    fn deinit(self: *ComplexGradient, alloc: std.mem.Allocator) void {
        if (self.x.len > 0) alloc.free(self.x);
        if (self.y.len > 0) alloc.free(self.y);
        if (self.z.len > 0) alloc.free(self.z);
    }
};

fn gradient_from_complex(
    alloc: std.mem.Allocator,
    grid: Grid,
    values_r: []const math.Complex,
) !ComplexGradient {
    const total = grid.count();

    // FFT complex real-space → reciprocal
    const values_r_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(values_r_copy);

    @memcpy(values_r_copy, values_r);
    const values_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(values_g);

    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, grid, values_r_copy, values_g, null);

    const gx_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gx_g);
    const gy_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gy_g);
    const gz_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gz_g);

    const i_unit = math.complex.init(0.0, 1.0);

    var it = scf_mod.GVecIterator.init(grid);
    while (it.next()) |g| {
        const i_rho = math.complex.mul(values_g[g.idx], i_unit);
        gx_g[g.idx] = math.complex.scale(i_rho, g.gvec.x);
        gy_g[g.idx] = math.complex.scale(i_rho, g.gvec.y);
        gz_g[g.idx] = math.complex.scale(i_rho, g.gvec.z);
    }

    // IFFT each component back to real space
    const gx_r = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gx_r);
    try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, gx_g, gx_r, null);
    alloc.free(gx_g);

    const gy_r = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gy_r);
    try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, gy_g, gy_r, null);
    alloc.free(gy_g);

    const gz_r = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(gz_r);
    try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, gz_g, gz_r, null);
    alloc.free(gz_g);

    return .{ .x = gx_r, .y = gy_r, .z = gz_r };
}

fn divergence_from_complex(
    alloc: std.mem.Allocator,
    grid: Grid,
    bx: []const math.Complex,
    by: []const math.Complex,
    bz: []const math.Complex,
) ![]math.Complex {
    const total = grid.count();

    // FFT each component to reciprocal space
    const bx_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(bx_copy);

    @memcpy(bx_copy, bx);
    const bx_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(bx_g);

    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, grid, bx_copy, bx_g, null);

    const by_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(by_copy);

    @memcpy(by_copy, by);
    const by_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(by_g);

    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, grid, by_copy, by_g, null);

    const bz_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(bz_copy);

    @memcpy(bz_copy, bz);
    const bz_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(bz_g);

    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, grid, bz_copy, bz_g, null);

    const div_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(div_g);

    const i_unit = math.complex.init(0.0, 1.0);

    var it = scf_mod.GVecIterator.init(grid);
    while (it.next()) |g| {
        const sum = math.complex.add(
            math.complex.add(
                math.complex.scale(bx_g[g.idx], g.gvec.x),
                math.complex.scale(by_g[g.idx], g.gvec.y),
            ),
            math.complex.scale(bz_g[g.idx], g.gvec.z),
        );
        div_g[g.idx] = math.complex.mul(sum, i_unit);
    }

    // IFFT back to real space
    const div_r = try alloc.alloc(math.Complex, total);
    try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, div_g, div_r, null);
    alloc.free(div_g);
    return div_r;
}

fn apply_nonlocal_dij_coefficients(
    entry: anytype,
    coeffs: []const f64,
    coeff: []const math.Complex,
    coeff2: []math.Complex,
) void {
    var b: usize = 0;
    while (b < entry.beta_count) : (b += 1) {
        const l_val = entry.l_list[b];
        const offset = entry.m_offsets[b];
        const m_count = entry.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            var sum = math.complex.init(0.0, 0.0);
            var j: usize = 0;
            while (j < entry.beta_count) : (j += 1) {
                if (entry.l_list[j] != l_val) continue;
                const dij = coeffs[b * entry.beta_count + j];
                if (dij == 0.0) continue;
                const c = coeff[entry.m_offsets[j] + m_idx];
                sum = math.complex.add(sum, math.complex.scale(c, dij));
            }
            coeff2[offset + m_idx] = sum;
        }
    }
}

fn fill_gamma_phase_products(
    gvecs: []const plane_wave.GVector,
    atom_position: math.Vec3,
    x: []const math.Complex,
    work_phase: []math.Complex,
    work_xphase: []math.Complex,
) void {
    for (0..gvecs.len) |g| {
        const phase = math.complex.expi(math.Vec3.dot(gvecs[g].cart, atom_position));
        work_phase[g] = phase;
        work_xphase[g] = math.complex.mul(x[g], phase);
    }
}

fn project_gamma_derivative_bra(
    entry: anytype,
    gvecs: []const plane_wave.GVector,
    direction: usize,
    work_xphase: []const math.Complex,
    coeff: []math.Complex,
) void {
    var b: usize = 0;
    while (b < entry.beta_count) : (b += 1) {
        const offset = entry.m_offsets[b];
        const m_count = entry.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const phi_start = (offset + m_idx) * entry.g_count;
            const phi = entry.phi[phi_start .. phi_start + entry.g_count];
            var sum = math.complex.init(0.0, 0.0);
            for (0..gvecs.len) |g| {
                const g_alpha = g_component(gvecs[g].kpg, direction);
                const term = math.complex.scale(work_xphase[g], phi[g] * g_alpha);
                sum = math.complex.add(sum, term);
            }
            coeff[offset + m_idx] = math.complex.init(-sum.i, sum.r);
        }
    }
}

fn project_gamma_bra(
    entry: anytype,
    work_xphase: []const math.Complex,
    coeff: []math.Complex,
) void {
    var b: usize = 0;
    while (b < entry.beta_count) : (b += 1) {
        const offset = entry.m_offsets[b];
        const m_count = entry.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const phi_start = (offset + m_idx) * entry.g_count;
            const phi = entry.phi[phi_start .. phi_start + entry.g_count];
            var sum = math.complex.init(0.0, 0.0);
            for (0..work_xphase.len) |g| {
                sum = math.complex.add(sum, math.complex.scale(work_xphase[g], phi[g]));
            }
            coeff[offset + m_idx] = sum;
        }
    }
}

fn reconstruct_gamma_bra_derivative(
    entry: anytype,
    work_phase: []const math.Complex,
    coeff2: []const math.Complex,
    inv_volume: f64,
    out: []math.Complex,
) void {
    for (0..out.len) |g| {
        var accum = math.complex.init(0.0, 0.0);
        var b: usize = 0;
        while (b < entry.beta_count) : (b += 1) {
            const offset = entry.m_offsets[b];
            const m_count = entry.m_counts[b];
            var m_idx: usize = 0;
            while (m_idx < m_count) : (m_idx += 1) {
                const phi_val = entry.phi[(offset + m_idx) * entry.g_count + g];
                const c = coeff2[offset + m_idx];
                accum = math.complex.add(accum, math.complex.scale(c, phi_val));
            }
        }
        const phase_conj = math.complex.conj(work_phase[g]);
        const add = math.complex.mul(phase_conj, accum);
        out[g] = math.complex.add(out[g], math.complex.scale(add, inv_volume));
    }
}

fn reconstruct_gamma_ket_derivative(
    entry: anytype,
    gvecs: []const plane_wave.GVector,
    direction: usize,
    work_phase: []const math.Complex,
    coeff2: []const math.Complex,
    inv_volume: f64,
    out: []math.Complex,
) void {
    for (0..out.len) |g| {
        var accum = math.complex.init(0.0, 0.0);
        var b: usize = 0;
        while (b < entry.beta_count) : (b += 1) {
            const offset = entry.m_offsets[b];
            const m_count = entry.m_counts[b];
            var m_idx: usize = 0;
            while (m_idx < m_count) : (m_idx += 1) {
                const phi_val = entry.phi[(offset + m_idx) * entry.g_count + g];
                const c = coeff2[offset + m_idx];
                const g_alpha = g_component(gvecs[g].kpg, direction);
                const weighted = math.complex.scale(c, phi_val * g_alpha);
                accum = math.complex.add(accum, math.complex.init(weighted.i, -weighted.r));
            }
        }
        const phase_conj = math.complex.conj(work_phase[g]);
        const add = math.complex.mul(phase_conj, accum);
        out[g] = math.complex.add(out[g], math.complex.scale(add, inv_volume));
    }
}

fn project_q_derivative_bra(
    entry: anytype,
    gvecs: []const plane_wave.GVector,
    atom_position: math.Vec3,
    direction: usize,
    x: []const math.Complex,
    coeff: []math.Complex,
) void {
    var b: usize = 0;
    while (b < entry.beta_count) : (b += 1) {
        const offset = entry.m_offsets[b];
        const m_count = entry.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const phi_start = (offset + m_idx) * gvecs.len;
            const phi = entry.phi[phi_start .. phi_start + gvecs.len];
            var sum = math.complex.init(0.0, 0.0);
            for (0..gvecs.len) |g| {
                const kpg = gvecs[g].kpg;
                const phase = math.complex.expi(math.Vec3.dot(kpg, atom_position));
                const g_alpha = g_component(kpg, direction);
                const term = math.complex.scale(
                    math.complex.mul(
                        math.complex.mul(x[g], phase),
                        math.complex.init(phi[g], 0.0),
                    ),
                    g_alpha,
                );
                sum = math.complex.add(sum, term);
            }
            coeff[offset + m_idx] = math.complex.init(-sum.i, sum.r);
        }
    }
}

fn project_q_bra(
    entry: anytype,
    gvecs: []const plane_wave.GVector,
    atom_position: math.Vec3,
    x: []const math.Complex,
    coeff: []math.Complex,
) void {
    var b: usize = 0;
    while (b < entry.beta_count) : (b += 1) {
        const offset = entry.m_offsets[b];
        const m_count = entry.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const phi_start = (offset + m_idx) * gvecs.len;
            const phi = entry.phi[phi_start .. phi_start + gvecs.len];
            var sum = math.complex.init(0.0, 0.0);
            for (0..gvecs.len) |g| {
                const phase = math.complex.expi(math.Vec3.dot(gvecs[g].kpg, atom_position));
                sum = math.complex.add(sum, math.complex.scale(
                    math.complex.mul(x[g], phase),
                    phi[g],
                ));
            }
            coeff[offset + m_idx] = sum;
        }
    }
}

fn reconstruct_q_bra_derivative(
    entry: anytype,
    gvecs: []const plane_wave.GVector,
    atom_position: math.Vec3,
    coeff2: []const math.Complex,
    inv_volume: f64,
    out: []math.Complex,
) void {
    for (0..gvecs.len) |g| {
        var accum = math.complex.init(0.0, 0.0);
        var b: usize = 0;
        while (b < entry.beta_count) : (b += 1) {
            const offset = entry.m_offsets[b];
            const m_count = entry.m_counts[b];
            var m_idx: usize = 0;
            while (m_idx < m_count) : (m_idx += 1) {
                const phi_val = entry.phi[(offset + m_idx) * gvecs.len + g];
                const c = coeff2[offset + m_idx];
                accum = math.complex.add(accum, math.complex.scale(c, phi_val));
            }
        }
        const phase_conj = math.complex.expi(-math.Vec3.dot(gvecs[g].kpg, atom_position));
        const term = math.complex.scale(math.complex.mul(phase_conj, accum), inv_volume);
        out[g] = math.complex.add(out[g], term);
    }
}

fn reconstruct_q_ket_derivative(
    entry: anytype,
    gvecs: []const plane_wave.GVector,
    atom_position: math.Vec3,
    direction: usize,
    coeff2: []const math.Complex,
    inv_volume: f64,
    out: []math.Complex,
) void {
    for (0..gvecs.len) |g| {
        var accum = math.complex.init(0.0, 0.0);
        var b: usize = 0;
        while (b < entry.beta_count) : (b += 1) {
            const offset = entry.m_offsets[b];
            const m_count = entry.m_counts[b];
            var m_idx: usize = 0;
            while (m_idx < m_count) : (m_idx += 1) {
                const phi_val = entry.phi[(offset + m_idx) * gvecs.len + g];
                const c = coeff2[offset + m_idx];
                accum = math.complex.add(accum, math.complex.scale(c, phi_val));
            }
        }
        const gpq_alpha = g_component(gvecs[g].kpg, direction);
        const weighted = math.complex.scale(accum, gpq_alpha);
        const neg_i_weighted = math.complex.init(weighted.i, -weighted.r);
        const phase_conj = math.complex.expi(-math.Vec3.dot(gvecs[g].kpg, atom_position));
        const term = math.complex.scale(
            math.complex.mul(phase_conj, neg_i_weighted),
            inv_volume,
        );
        out[g] = math.complex.add(out[g], term);
    }
}

/// Apply V_nl^(1)|ψ⟩ for atom displacement perturbation.
///
/// The nonlocal pseudopotential contribution to the first-order Hamiltonian
/// for an atomic displacement perturbation comes from the derivative of the
/// structure factor exp(-i(k+G)·τ_I):
///
/// V_nl^(1)|ψ⟩ = Σ_β Σ_m D_ββ' × [
///                 -i(k+G)_α × φ_βm(k+G) e^{-i(k+G)·τ}
///                   × ⟨φ_β'm(k+G) e^{-i(k+G)·τ}|ψ⟩
///              + φ_βm(k+G) e^{-i(k+G)·τ} ⟨-i(k+G)_α × φ_β'm(k+G) e^{-i(k+G)·τ}|ψ⟩]
///
/// This is symmetric: both the ket and bra projectors get the -iG_α derivative.
pub fn apply_nonlocal_perturbation(
    gvecs: []const plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    nonlocal_ctx: anytype, // NonlocalContext from apply.zig
    perturbed_atom: usize,
    direction: usize,
    inv_volume: f64,
    x: []const math.Complex,
    out: []math.Complex,
    work_phase: []math.Complex,
    work_xphase: []math.Complex,
    work_coeff: []math.Complex,
    work_coeff2: []math.Complex,
) void {
    const n = gvecs.len;
    @memset(out, math.complex.init(0.0, 0.0));

    for (nonlocal_ctx.species) |entry| {
        const coeffs = entry.coeffs;
        const g_count = entry.g_count;
        if (g_count != n) return;
        if (entry.m_total == 0) continue;
        const coeff = work_coeff[0..entry.m_total];
        const coeff2 = work_coeff2[0..entry.m_total];

        for (atoms, 0..) |atom, atom_idx| {
            if (atom.species_index != entry.species_index) continue;
            if (atom_idx != perturbed_atom) continue;
            fill_gamma_phase_products(gvecs, atom.position, x, work_phase, work_xphase);
            project_gamma_derivative_bra(entry, gvecs, direction, work_xphase, coeff);
            apply_nonlocal_dij_coefficients(entry, coeffs, coeff, coeff2);
            reconstruct_gamma_bra_derivative(entry, work_phase, coeff2, inv_volume, out);

            project_gamma_bra(entry, work_xphase, coeff);
            apply_nonlocal_dij_coefficients(entry, coeffs, coeff, coeff2);
            reconstruct_gamma_ket_derivative(
                entry,
                gvecs,
                direction,
                work_phase,
                coeff2,
                inv_volume,
                out,
            );
        }
    }
}

/// Apply V_nl^(1) perturbation for q≠0: cross-basis (k→k+q).
///
/// V_nl^(1)|ψ_k⟩ has two terms:
/// Term 1 (bra derivative): Σ_{ββ'} D_{ββ'} |φ_{β,k+q}⟩ × ⟨(+iG_α)φ_{β',k}|ψ_k⟩
/// Term 2 (ket derivative): Σ_{ββ'} D_{ββ'} |(-iG'_α)φ_{β,k+q}⟩ × ⟨φ_{β',k}|ψ_k⟩
///
/// Input ψ_k is in k-basis (n_pw_k), output is in k+q-basis (n_pw_kq).
/// Phase factors use G (cart) for both bra and ket (NOT G+k or G'+q).
pub fn apply_nonlocal_perturbation_q(
    alloc: std.mem.Allocator,
    gvecs_k: []const plane_wave.GVector,
    gvecs_kq: []const plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    nl_ctx_k: anytype,
    nl_ctx_kq: anytype,
    perturbed_atom: usize,
    direction: usize,
    inv_volume: f64,
    x: []const math.Complex,
    out: []math.Complex,
) !void {
    @memset(out, math.complex.init(0.0, 0.0));

    for (nl_ctx_k.species, 0..) |entry_k, sp_idx| {
        if (entry_k.g_count != gvecs_k.len) continue;
        if (entry_k.m_total == 0) continue;

        const entry_kq = nl_ctx_kq.species[sp_idx];
        if (entry_kq.g_count != gvecs_kq.len) continue;

        const coeffs = entry_k.coeffs;
        const m_total = entry_k.m_total;

        // Work buffers
        const coeff = try alloc.alloc(math.Complex, m_total);
        defer alloc.free(coeff);

        const coeff2 = try alloc.alloc(math.Complex, m_total);
        defer alloc.free(coeff2);

        for (atoms, 0..) |atom, atom_idx| {
            if (atom.species_index != entry_k.species_index) continue;
            if (atom_idx != perturbed_atom) continue;
            project_q_derivative_bra(entry_k, gvecs_k, atom.position, direction, x, coeff);
            apply_nonlocal_dij_coefficients(entry_k, coeffs, coeff, coeff2);
            reconstruct_q_bra_derivative(
                entry_kq,
                gvecs_kq,
                atom.position,
                coeff2,
                inv_volume,
                out,
            );

            project_q_bra(entry_k, gvecs_k, atom.position, x, coeff);
            apply_nonlocal_dij_coefficients(entry_k, coeffs, coeff, coeff2);
            reconstruct_q_ket_derivative(
                entry_kq,
                gvecs_kq,
                atom.position,
                direction,
                coeff2,
                inv_volume,
                out,
            );
        }
    }
}

/// Build V_loc^(1)(G) for q≠0.
///
/// V_loc^(1)_q(G) = -i × (G+q)_α × V_form(|G+q|) × exp(-i(G+q)·τ_I) / Ω
///
/// Key difference from q=0: G → G+q everywhere.
pub fn build_local_perturbation_q(
    alloc: std.mem.Allocator,
    grid: Grid,
    atom: hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
    direction: usize,
    q_cart: math.Vec3,
    local_cfg: local_potential.LocalPotentialConfig,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
) ![]math.Complex {
    const total = grid.count();
    const result = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(result);

    const inv_volume = 1.0 / grid.volume;

    var it = scf_mod.GVecIterator.init(grid);
    while (it.next()) |g| {
        // G+q vector
        const gpq = math.Vec3.add(g.gvec, q_cart);
        const gpq_norm = math.Vec3.norm(gpq);
        const gpq_alpha = g_component(gpq, direction);

        // Skip only when |G+q| ≈ 0 (i.e. G=0 at q=0).
        // For q≠0, G=0 gives |G+q| = |q| > 0, which is a valid contribution.
        // Skipping G=0 unconditionally at finite q was a bug that caused
        // the electronic response to be ~5× too small.
        if (gpq_norm < 1e-12) {
            result[g.idx] = math.complex.init(0.0, 0.0);
            continue;
        }

        // V_form(|G+q|)
        const v_loc = if (ff_tables) |tables|
            tables[atom.species_index].eval(gpq_norm)
        else
            hamiltonian.local_form_factor(&species[atom.species_index], gpq_norm, local_cfg);

        // exp(-i(G+q)·τ_I)
        const phase = math.complex.expi(-math.Vec3.dot(gpq, atom.position));

        // V_loc^(1)_q(G) = -i × (G+q)_α × V_form × exp(-i(G+q)·τ) / Ω
        const temp = math.complex.scale(phase, gpq_alpha * v_loc * inv_volume);
        result[g.idx] = math.complex.init(temp.i, -temp.r); // multiply by -i
    }

    return result;
}

/// Build ρ^(1)_core(G) for q≠0 perturbation.
///
/// ρ^(1)_core,q(G) = -i × (G+q)_α × ρ_core_form(|G+q|) × exp(-i(G+q)·τ_I) / Ω
pub fn build_core_perturbation_q(
    alloc: std.mem.Allocator,
    grid: Grid,
    atom: hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
    direction: usize,
    q_cart: math.Vec3,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) ![]math.Complex {
    const total = grid.count();
    const result = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(result);

    const sp = &species[atom.species_index];
    if (sp.upf.nlcc.len == 0) {
        @memset(result, math.complex.init(0.0, 0.0));
        return result;
    }

    const inv_volume = 1.0 / grid.volume;

    var it = scf_mod.GVecIterator.init(grid);
    while (it.next()) |g| {
        const gpq = math.Vec3.add(g.gvec, q_cart);
        const gpq_norm = math.Vec3.norm(gpq);
        const gpq_alpha = g_component(gpq, direction);

        // Skip only when |G+q| ≈ 0 (i.e. G=0 at q=0).
        // For q≠0, G=0 gives |G+q| = |q| > 0 → valid contribution.
        if (gpq_norm < 1e-12) {
            result[g.idx] = math.complex.init(0.0, 0.0);
            continue;
        }

        const rho_core_g = if (rho_core_tables) |tables|
            tables[atom.species_index].eval(gpq_norm)
        else
            form_factor.rho_core_g(sp.upf.*, gpq_norm);

        const phase = math.complex.expi(-math.Vec3.dot(gpq, atom.position));
        const temp = math.complex.scale(phase, gpq_alpha * rho_core_g * inv_volume);
        result[g.idx] = math.complex.init(temp.i, -temp.r); // multiply by -i
    }

    return result;
}

/// Build V_H^(1)(G) = 8π × ρ^(1)(G) / |G+q|²  for q≠0.
pub fn build_hartree_perturbation_q(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho1_g: []const math.Complex,
    q_cart: math.Vec3,
) ![]math.Complex {
    const total = grid.count();
    const result = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(result);

    var it = scf_mod.GVecIterator.init(grid);
    while (it.next()) |g| {
        const gpq = math.Vec3.add(g.gvec, q_cart);
        const gpq2 = math.Vec3.dot(gpq, gpq);

        // Skip only when |G+q| ≈ 0 (i.e. G=0 at q=0).
        // For q≠0, G=0 gives |G+q| = |q| > 0 → finite V_H^(1).
        if (gpq2 > 1e-12) {
            result[g.idx] = math.complex.scale(rho1_g[g.idx], 8.0 * std.math.pi / gpq2);
        } else {
            result[g.idx] = math.complex.init(0.0, 0.0);
        }
    }

    return result;
}

/// Build V_xc^(1)(r) = f_xc(r) × ρ^(1)(r) in real space (complex version for q≠0).
/// For q≠0, ρ^(1)(r) is complex.
pub fn build_xc_perturbation_complex(
    alloc: std.mem.Allocator,
    fxc_r: []const f64,
    rho1_r: []const math.Complex,
) ![]math.Complex {
    const n = fxc_r.len;
    const result = try alloc.alloc(math.Complex, n);
    for (result, 0..) |*v, i| {
        v.* = math.complex.scale(rho1_r[i], fxc_r[i]);
    }
    return result;
}

/// Extract the alpha-component of a vector.
pub fn g_component(v: math.Vec3, direction: usize) f64 {
    return switch (direction) {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        else => 0.0,
    };
}

// =========================================================================
// Tests
// =========================================================================

test "V_loc perturbation finite difference" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;
    try test_support.require_file(io, "pseudo/Si.upf");

    // Simple cubic grid
    const a = 10.0;
    const pi = std.math.pi;
    const grid = Grid{
        .nx = 5,
        .ny = 5,
        .nz = 5,
        .min_h = -2,
        .min_k = -2,
        .min_l = -2,
        .cell = math.Mat3{ .m = .{
            .{ a, 0.0, 0.0 },
            .{ 0.0, a, 0.0 },
            .{ 0.0, 0.0, a },
        } },
        .recip = math.Mat3{ .m = .{
            .{ 2.0 * pi / a, 0.0, 0.0 },
            .{ 0.0, 2.0 * pi / a, 0.0 },
            .{ 0.0, 0.0, 2.0 * pi / a },
        } },
        .volume = a * a * a,
    };

    // Load Si pseudopotential
    var element_buf: [2]u8 = .{ 'S', 'i' };
    var path_buf: [24]u8 = undefined;
    const path_slice = "pseudo/Si.upf";
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..2],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(alloc, io, spec);
    defer parsed.deinit(alloc);

    var parsed_items = [_]pseudo.Parsed{parsed};
    const species = try hamiltonian.build_species_entries(alloc, parsed_items[0..]);
    defer {
        for (species) |*entry| {
            entry.deinit();
        }
        alloc.free(species);
    }

    const atom_pos = math.Vec3{ .x = 1.0, .y = 2.0, .z = 0.5 };
    const atom = hamiltonian.AtomData{ .position = atom_pos, .species_index = 0 };

    // Build perturbation for x-direction
    const local_cfg = local_potential.LocalPotentialConfig.init(.short_range, 0.0);
    const vloc1 = try build_local_perturbation(alloc, grid, atom, species, 0, local_cfg, null);
    defer alloc.free(vloc1);

    // Finite difference: (V_loc(τ+δ) - V_loc(τ-δ)) / (2δ)
    const delta: f64 = 1e-5;
    const inv_vol = 1.0 / grid.volume;
    var atoms_plus = [_]hamiltonian.AtomData{.{
        .position = math.Vec3{ .x = atom_pos.x + delta, .y = atom_pos.y, .z = atom_pos.z },
        .species_index = 0,
    }};
    var atoms_minus = [_]hamiltonian.AtomData{.{
        .position = math.Vec3{ .x = atom_pos.x - delta, .y = atom_pos.y, .z = atom_pos.z },
        .species_index = 0,
    }};

    // Compute V_loc at displaced positions for each G point
    var max_rel_err: f64 = 0.0;
    var tested: usize = 0;
    var it2 = scf_mod.GVecIterator.init(grid);
    while (it2.next()) |g| {
        if (g.gh == 0 and g.gk == 0 and g.gl == 0) continue;

        const vp = try hamiltonian.ionic_local_potential(
            g.gvec,
            species,
            atoms_plus[0..],
            inv_vol,
            local_cfg,
        );
        const vm = try hamiltonian.ionic_local_potential(
            g.gvec,
            species,
            atoms_minus[0..],
            inv_vol,
            local_cfg,
        );

        const fd_r = (vp.r - vm.r) / (2.0 * delta);
        const fd_i = (vp.i - vm.i) / (2.0 * delta);

        const abs_val = @sqrt(vloc1[g.idx].r * vloc1[g.idx].r + vloc1[g.idx].i * vloc1[g.idx].i);
        if (abs_val > 1e-8) {
            const err_r = @abs(vloc1[g.idx].r - fd_r);
            const err_i = @abs(vloc1[g.idx].i - fd_i);
            const err = @sqrt(err_r * err_r + err_i * err_i);
            const rel = err / abs_val;
            if (rel > max_rel_err) max_rel_err = rel;
            tested += 1;
        }
    }

    try std.testing.expect(tested > 10);
    try std.testing.expect(max_rel_err < 1e-4);
}

test "V_H perturbation basic properties" {
    const alloc = std.testing.allocator;
    const pi = std.math.pi;
    const a = 10.0;

    const grid = Grid{
        .nx = 3,
        .ny = 3,
        .nz = 3,
        .min_h = -1,
        .min_k = -1,
        .min_l = -1,
        .cell = math.Mat3{ .m = .{
            .{ a, 0.0, 0.0 },
            .{ 0.0, a, 0.0 },
            .{ 0.0, 0.0, a },
        } },
        .recip = math.Mat3{ .m = .{
            .{ 2.0 * pi / a, 0.0, 0.0 },
            .{ 0.0, 2.0 * pi / a, 0.0 },
            .{ 0.0, 0.0, 2.0 * pi / a },
        } },
        .volume = a * a * a,
    };

    const total = grid.count();
    const rho1_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_g);

    // Set a simple test density response
    for (rho1_g) |*r| {
        r.* = math.complex.init(0.01, 0.0);
    }
    // G=0 component
    const g0_idx = 1 + 3 * (1 + 3 * 1); // (0,0,0) maps to index (1,1,1)
    rho1_g[g0_idx] = math.complex.init(0.0, 0.0);

    const vh1 = try build_hartree_perturbation(alloc, grid, rho1_g);
    defer alloc.free(vh1);

    // V_H^(1)(G=0) should be 0
    try std.testing.expectApproxEqAbs(vh1[g0_idx].r, 0.0, 1e-15);
    try std.testing.expectApproxEqAbs(vh1[g0_idx].i, 0.0, 1e-15);

    // V_H^(1)(G≠0) should be nonzero for nonzero ρ^(1)
    var has_nonzero = false;
    for (0..total) |i| {
        if (i == g0_idx) continue;
        if (@abs(vh1[i].r) > 1e-10 or @abs(vh1[i].i) > 1e-10) {
            has_nonzero = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero);
}

test "V_xc perturbation" {
    const alloc = std.testing.allocator;
    const n = 8;
    const fxc_r = try alloc.alloc(f64, n);
    defer alloc.free(fxc_r);

    const rho1_r = try alloc.alloc(f64, n);
    defer alloc.free(rho1_r);

    for (0..n) |i| {
        fxc_r[i] = -0.5 * @as(f64, @floatFromInt(i + 1));
        rho1_r[i] = 0.01 * @as(f64, @floatFromInt(i + 1));
    }

    const vxc1 = try build_xc_perturbation(alloc, fxc_r, rho1_r);
    defer alloc.free(vxc1);

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(vxc1[i], fxc_r[i] * rho1_r[i], 1e-15);
    }
}
