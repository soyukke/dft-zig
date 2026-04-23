const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf = @import("../scf/scf.zig");
const paw_mod = @import("../paw/paw_tab.zig");
const stress_util = @import("stress.zig");

const Stress3x3 = stress_util.Stress3x3;
const Grid = stress_util.Grid;

/// Accumulate augmented-density contributions at a single G-vector.
/// Returns (sum_re, sum_im) from all atoms' QIJL L=0 channels.
fn accumulateAugmentedDensityG(
    g_vec: math.Vec3,
    g_abs: f64,
    paw_rhoij: []const []const f64,
    paw_tabs: []const paw_mod.PawTab,
    atoms: []const hamiltonian.AtomData,
    inv_omega: f64,
) struct { f64, f64 } {
    var sum_re: f64 = 0.0;
    var sum_im: f64 = 0.0;

    for (0..atoms.len) |a| {
        const sp = atoms[a].species_index;
        if (sp >= paw_tabs.len) continue;
        const tab = &paw_tabs[sp];
        if (tab.nbeta == 0) continue;
        const nb = tab.nbeta;
        const rij = paw_rhoij[a];
        const pos = atoms[a].position;

        const g_dot_r = math.Vec3.dot(g_vec, pos);
        const sf_re = @cos(g_dot_r);
        const sf_im = -@sin(g_dot_r);

        for (0..tab.n_qijl_entries) |e| {
            const qidx = tab.qijl_indices[e];
            if (qidx.l != 0) continue;
            const i = qidx.first;
            const j = qidx.second;
            const rij_val = rij[i * nb + j];
            if (@abs(rij_val) < 1e-30) continue;

            const qijl_g = tab.evalQijlForm(e, g_abs);
            if (@abs(qijl_g) < 1e-30) continue;

            const ylm = 1.0 / @sqrt(4.0 * std.math.pi);
            const gaunt = 1.0 / @sqrt(4.0 * std.math.pi);
            const sym_factor: f64 = if (i != j) 2.0 else 1.0;
            const contrib = rij_val * qijl_g * ylm * gaunt * sym_factor * inv_omega;
            sum_re += contrib * sf_re;
            sum_im += contrib * sf_im;
        }
    }
    return .{ sum_re, sum_im };
}

/// Build augmented density ρ̃ + n̂ for PAW stress calculation.
/// Same logic as addPawCompensationCharge in scf.zig.
pub fn buildAugmentedDensity(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    paw_rhoij: []const []const f64,
    paw_tabs: []const paw_mod.PawTab,
    atoms: []const hamiltonian.AtomData,
    ecutrho: f64,
) !void {
    const total = grid.nx * grid.ny * grid.nz;
    const n_hat_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(n_hat_g);

    @memset(n_hat_g, math.complex.init(0.0, 0.0));

    const inv_omega = 1.0 / grid.volume;

    var it = scf.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.g2 >= ecutrho) {
            continue;
        }

        const g_abs = math.Vec3.norm(g.gvec);
        const sums = accumulateAugmentedDensityG(
            g.gvec,
            g_abs,
            paw_rhoij,
            paw_tabs,
            atoms,
            inv_omega,
        );
        n_hat_g[g.idx].r += sums[0];
        n_hat_g[g.idx].i += sums[1];
    }

    // IFFT n_hat(G) → n_hat(r) and add to density
    const fft_obj = scf.Grid{
        .nx = grid.nx,
        .ny = grid.ny,
        .nz = grid.nz,
        .min_h = grid.min_h,
        .min_k = grid.min_k,
        .min_l = grid.min_l,
        .cell = grid.cell,
        .recip = grid.recip,
        .volume = grid.volume,
    };
    const n_hat_r = try scf.reciprocalToReal(alloc, fft_obj, n_hat_g);
    defer alloc.free(n_hat_r);

    for (0..@min(rho.len, n_hat_r.len)) |i| {
        rho[i] += n_hat_r[i];
    }
}

/// Accumulate augmentation stress contributions at a single G-vector for a single atom.
fn accumulateAugmentationStressAtom(
    g_vec: math.Vec3,
    g_norm: f64,
    v_eff: math.Complex,
    tab: *const paw_mod.PawTab,
    rij: []const f64,
    inv_volume: f64,
    ylm_gaunt: f64,
    atom_pos: math.Vec3,
    sigma: *Stress3x3,
) void {
    const nb = tab.nbeta;
    const g_dot_r = math.Vec3.dot(g_vec, atom_pos);
    // S*_a(G) = exp(+iG·R)
    const sf_re = @cos(g_dot_r);
    const sf_im = @sin(g_dot_r);
    // Re[V_eff(G) × exp(+iG·R)] = V_re cos(GR) - V_im sin(GR)
    const re_vs = v_eff.r * sf_re - v_eff.i * sf_im;

    for (0..tab.n_qijl_entries) |e| {
        const qidx = tab.qijl_indices[e];
        if (qidx.l != 0) continue;
        const i = qidx.first;
        const j = qidx.second;
        const rij_val = rij[i * nb + j];
        if (@abs(rij_val) < 1e-30) continue;

        const sym_factor: f64 = if (i != j) 2.0 else 1.0;
        const rij_sym = rij_val * sym_factor;

        // Off-diagonal: dQ/dG derivative term (G≠0 only)
        if (g_norm < 1e-12) continue;
        const dqijl_g = tab.evalQijlFormDeriv(e, g_norm);
        const factor = -rij_sym * dqijl_g * re_vs * ylm_gaunt * inv_volume / g_norm;
        const gv = [3]f64{ g_vec.x, g_vec.y, g_vec.z };
        for (0..3) |a2| {
            for (a2..3) |b| {
                sigma[a2][b] += factor * gv[a2] * gv[b];
            }
        }
    }
}

/// PAW augmentation charge stress (off-diagonal only, matching QE's addusstress).
/// σ^aug_αβ = (1/Ω) Σ_{G≠0} Σ_{a,ij} ρ_ij × dQ_ij(|G|)/d|G|
///             × V_eff(G) × S*_a(G) × (-G_αG_β/|G|) / Ω
pub fn augmentationStress(
    alloc: std.mem.Allocator,
    grid: Grid,
    potential_values: ?[]const math.Complex,
    paw_rhoij: ?[]const []const f64,
    paw_tabs: ?[]const paw_mod.PawTab,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    ecutrho: f64,
) !Stress3x3 {
    var sigma = stress_util.zeroStress();
    const tabs = paw_tabs orelse return sigma;
    const rhoij = paw_rhoij orelse return sigma;
    const pot_vals = potential_values orelse return sigma;
    _ = alloc;

    const ylm_gaunt = 1.0 / (4.0 * std.math.pi); // Y_00 × Gaunt for L=0

    var it = scf.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.g2 >= ecutrho) {
            continue;
        }

        const g_norm = math.Vec3.norm(g.gvec);
        const v_eff = pot_vals[g.idx];

        for (atoms, 0..) |atom, a| {
            const si = atom.species_index;
            if (si >= tabs.len) continue;
            const tab = &tabs[si];
            if (tab.nbeta == 0) continue;
            const rij = rhoij[a];
            accumulateAugmentationStressAtom(
                g.gvec,
                g_norm,
                v_eff,
                tab,
                rij,
                inv_volume,
                ylm_gaunt,
                atom.position,
                &sigma,
            );
        }
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}
