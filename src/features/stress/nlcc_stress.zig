const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const scf = @import("../scf/scf.zig");
const xc_mod = @import("../xc/xc.zig");
const stress_util = @import("stress.zig");

const Stress3x3 = stress_util.Stress3x3;
const Grid = stress_util.Grid;

/// Return true if any species has non-empty NLCC core data.
fn any_species_has_nlcc(species: []const hamiltonian.SpeciesEntry) bool {
    for (species) |sp| {
        if (sp.upf.nlcc.len > 0) return true;
    }
    return false;
}

/// Convert this stress module's Grid into the SCF module's Grid representation.
fn to_scf_grid(grid: Grid) scf.Grid {
    return scf.Grid{
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
}

/// Accumulate NLCC stress contributions at a single G-vector.
fn accumulate_nlcc_stress_g(
    g_vec: math.Vec3,
    g_norm: f64,
    vxc_val: math.Complex,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    core_tables: []const form_factor.RadialFormFactorTable,
    inv_volume: f64,
    sigma: *Stress3x3,
) void {
    const gv = [3]f64{ g_vec.x, g_vec.y, g_vec.z };

    for (atoms) |atom| {
        const si = atom.species_index;
        if (species[si].upf.nlcc.len == 0) continue;

        const drhoc = core_tables[si].eval_deriv(g_norm);
        const phase = math.Vec3.dot(g_vec, atom.position);
        const cos_phase = std.math.cos(phase);
        const sin_phase = std.math.sin(phase);
        // Re[V_xc(G) × conj(exp(-iG·R))] = Re[V_xc × exp(+iGR)] = Vxc_r cos - Vxc_i sin
        const re_vxc_si = vxc_val.r * cos_phase - vxc_val.i * sin_phase;

        const factor = -drhoc * re_vxc_si / g_norm * inv_volume;
        for (0..3) |a| {
            for (a..3) |b| {
                sigma[a][b] += factor * gv[a] * gv[b];
            }
        }
    }
}

/// NLCC stress: contribution from core charge density dependence on strain.
/// σ_αβ = -(1/Ω) Σ_{G≠0} V_xc(G) × Σ_I ρ_core_form'(|G|) × G_αG_β/|G| × Re[S*_I(G)]
///        -(E_nlcc/Ω) δ_αβ  (volume scaling)
pub fn nlcc_stress(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_r: []const f64,
    rho_core: ?[]const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    xc_func: xc_mod.Functional,
    ecutrho: f64,
) !Stress3x3 {
    var sigma = stress_util.zero_stress();
    const core_tables = rho_core_tables orelse return sigma;

    if (!any_species_has_nlcc(species)) return sigma;
    if (rho_core == null) return sigma;

    const inv_volume = 1.0 / grid.volume;

    // Compute V_xc(G) from V_xc(r)
    const n_grid = grid.nx * grid.ny * grid.nz;
    const rho_total = try alloc.alloc(f64, n_grid);
    defer alloc.free(rho_total);

    for (0..n_grid) |i| {
        rho_total[i] = rho_r[i];
        if (rho_core) |rc| rho_total[i] += rc[i];
    }

    const scf_grid = to_scf_grid(grid);
    const xc_fields = try scf.compute_xc_fields(alloc, scf_grid, rho_total, null, false, xc_func);
    defer {
        alloc.free(xc_fields.vxc);
        alloc.free(xc_fields.exc);
    }

    const vxc_g = try scf.real_to_reciprocal(alloc, scf_grid, xc_fields.vxc, false);
    defer alloc.free(vxc_g);

    var it = scf.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.gh == 0 and g.gk == 0 and g.gl == 0) {
            continue;
        }
        const g_norm = math.Vec3.norm(g.gvec);
        if (g_norm < 1e-12 or g.g2 >= ecutrho) {
            continue;
        }

        accumulate_nlcc_stress_g(
            g.gvec,
            g_norm,
            vxc_g[g.idx],
            species,
            atoms,
            core_tables,
            inv_volume,
            &sigma,
        );
    }

    // NLCC diagonal: -(∫V_xc × ρ_core dr)/Ω δ_αβ
    {
        const dv = grid.volume / @as(f64, @floatFromInt(n_grid));
        var vxc_core: f64 = 0.0;
        for (0..n_grid) |i| {
            if (rho_core) |rc| {
                vxc_core += xc_fields.vxc[i] * rc[i] * dv;
            }
        }
        const nlcc_diag = -vxc_core * inv_volume;
        for (0..3) |a| sigma[a][a] += nlcc_diag;
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}
