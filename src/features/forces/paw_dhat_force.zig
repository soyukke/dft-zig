const std = @import("std");
const math = @import("../math/math.zig");
const scf = @import("../scf/scf.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const paw_mod = @import("../paw/paw.zig");
const local_force = @import("local_force.zig");

const Grid = local_force.Grid;
const Vec3 = math.Vec3;
const Complex = math.Complex;

/// Accumulate the PAW D^hat force contribution from a single G-vector for one atom.
fn accumulatePawDhatForceG(
    g_vec: Vec3,
    g_abs: f64,
    prod_im: f64,
    tab: *const paw_mod.PawTab,
    atom_rij: []const f64,
    nb: usize,
    fx: *f64,
    fy: *f64,
    fz: *f64,
) void {
    // Sum over L=0 QIJL entries
    for (0..tab.n_qijl_entries) |e| {
        const qidx = tab.qijl_indices[e];
        if (qidx.l != 0) continue;
        const i = qidx.first;
        const j = qidx.second;

        const qijl_g = tab.evalQijlForm(e, g_abs);
        if (@abs(qijl_g) < 1e-30) continue;

        // Y_00 = 1/√(4π), Gaunt = 1/√(4π) for m-summed L=0
        const ylm = 1.0 / @sqrt(4.0 * std.math.pi);
        const gaunt = 1.0 / @sqrt(4.0 * std.math.pi);

        // ρ_ij contribution (symmetric: count off-diagonal twice)
        const rho_ij = atom_rij[i * nb + j];
        const rho_factor = if (i != j) 2.0 * rho_ij else rho_ij;

        const coeff = rho_factor * prod_im * qijl_g * ylm * gaunt;
        fx.* += g_vec.x * coeff;
        fy.* += g_vec.y * coeff;
        fz.* += g_vec.z * coeff;
    }
}

/// Compute PAW D^hat forces from the position-dependence of the compensation charge.
///
/// D^hat_ij = Σ_G Re[V_eff(G) × exp(+iG·R_a)] × Q_form(|G|) × Y_00 × Gaunt
///
/// The force is the position derivative of the D^hat contribution to the energy:
///   E_dhat = Σ_ij ρ_ij × D^hat_ij
///   F_{a,α} = -∂E_dhat/∂R_{a,α}
///           = -Σ_ij ρ_ij × ∂D^hat_ij/∂R_{a,α}
///
/// Since only exp(+iG·R_a) depends on position:
///   ∂/∂R_α exp(+iG·R) = +iG_α × exp(+iG·R)
///   ∂D^hat_ij/∂R_α = Σ_G Im[V_eff(G) × exp(+iG·R)] × (-G_α) × Q × Y × Gaunt
///
/// Therefore:
///   F_{a,α} = +Σ_ij ρ_ij × Σ_G G_α × Im[V_eff(G) × exp(+iG·R_a)] × Q × Y × Gaunt
///
/// Only L=0 contributes (m-summed rhoij).
///
/// Returns forces in Rydberg/Bohr units.
pub fn pawDhatForces(
    alloc: std.mem.Allocator,
    grid: Grid,
    potential: []const Complex,
    paw_tabs: []const paw_mod.PawTab,
    paw_rhoij: []const []const f64,
    _: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) ![]Vec3 {
    const n_atoms = atoms.len;
    if (n_atoms == 0) return &[_]Vec3{};

    const forces = try alloc.alloc(Vec3, n_atoms);
    errdefer alloc.free(forces);

    for (forces) |*f| {
        f.* = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    }

    const total = grid.nx * grid.ny * grid.nz;

    for (atoms, 0..) |atom, ai| {
        if (ai >= paw_rhoij.len) continue;
        const si = atom.species_index;
        if (si >= paw_tabs.len or paw_tabs[si].nbeta == 0) continue;
        const tab = &paw_tabs[si];
        const nb = tab.nbeta;
        const atom_rij = paw_rhoij[ai];
        const pos = atom.position;

        var fx: f64 = 0.0;
        var fy: f64 = 0.0;
        var fz: f64 = 0.0;

        var it = scf.GVecIterator.init(grid);
        while (it.next()) |g| {
            const g_abs = Vec3.norm(g.gvec);

            // V_eff(G) = V_H(G) + V_xc(G) — no V_ionic
            // D^hat only involves V_Hxc acting on augmentation charges.
            const v_eff = if (g.idx < total) potential[g.idx] else math.complex.init(0.0, 0.0);

            // Structure factor: exp(+iG·R_a)
            const g_dot_r = Vec3.dot(g.gvec, pos);
            const sf_re = @cos(g_dot_r);
            const sf_im = @sin(g_dot_r);
            // Im[V_eff × exp(+iGR)] = V_eff.r × sf_im + V_eff.i × sf_re
            const prod_im = v_eff.r * sf_im + v_eff.i * sf_re;

            accumulatePawDhatForceG(g.gvec, g_abs, prod_im, tab, atom_rij, nb, &fx, &fy, &fz);
        }

        forces[ai] = Vec3{ .x = fx, .y = fy, .z = fz };
    }

    return forces;
}
