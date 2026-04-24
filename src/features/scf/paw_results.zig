const std = @import("std");
const apply = @import("apply.zig");
const common_mod = @import("common.zig");
const config = @import("../config/config.zig");
const density_symmetry = @import("density_symmetry.zig");
const energy_mod = @import("energy.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const paw_mod = @import("../paw/paw.zig");
const paw_scf = @import("paw_scf.zig");
const potential_mod = @import("potential.zig");

const Grid = grid_mod.Grid;
const ScfCommon = common_mod.ScfCommon;

pub const FinalPawResults = struct {
    paw_tabs: ?[]paw_mod.PawTab = null,
    paw_dij: ?[][]f64 = null,
    paw_dij_m: ?[][]f64 = null,
    paw_dxc: ?[][]f64 = null,
    paw_rhoij: ?[][]f64 = null,
};

/// Duplicate per-atom []f64 slices from an iterator to an owned slice of owned slices.
fn dup_per_atom_dij(
    alloc: std.mem.Allocator,
    nc_species: anytype,
    select: enum { radial, m_resolved },
) !?[][]f64 {
    var list: std.ArrayList([]f64) = .empty;
    errdefer {
        for (list.items) |d| alloc.free(d);
        list.deinit(alloc);
    }
    for (nc_species) |entry| {
        const source = switch (select) {
            .radial => entry.dij_per_atom,
            .m_resolved => entry.dij_m_per_atom,
        };
        if (source) |dpa| {
            for (dpa) |atom_dij| {
                const copy = try alloc.alloc(f64, atom_dij.len);
                @memcpy(copy, atom_dij);
                try list.append(alloc, copy);
            }
        }
    }
    if (list.items.len > 0) return try list.toOwnedSlice(alloc);
    list.deinit(alloc);
    return null;
}

fn append_empty_dxc(
    alloc: std.mem.Allocator,
    dxc_list: *std.ArrayList([]f64),
) !void {
    try dxc_list.append(alloc, try alloc.alloc(f64, 0));
}

fn accumulate_matrix_rhoij_trace(
    sum_dxc_rhoij: *f64,
    matrix: []const f64,
    rhoij_m: []const f64,
    mt: usize,
) void {
    for (0..mt) |im| {
        for (0..mt) |jm| {
            sum_dxc_rhoij.* += matrix[im * mt + jm] * rhoij_m[im * mt + jm];
        }
    }
}

/// Compute per-atom D^xc (angular) and accumulate the D^H/D^xc
/// double-counting correction into sum_dxc_rhoij.
fn compute_per_atom_dxc_and_dc(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    prij: *const paw_mod.RhoIJ,
    tabs: []const paw_mod.PawTab,
    gaunt: *paw_mod.GauntTable,
) !struct { dxc: ?[][]f64, sum_dxc_rhoij: f64 } {
    var dxc_list: std.ArrayList([]f64) = .empty;
    errdefer {
        for (dxc_list.items) |d| alloc.free(d);
        dxc_list.deinit(alloc);
    }
    var sum_dxc_rhoij: f64 = 0.0;
    for (atoms, 0..) |atom, ai| {
        const si = atom.species_index;
        const upf = species[si].upf;
        const paw = upf.paw orelse {
            try append_empty_dxc(alloc, &dxc_list);
            continue;
        };
        if (si >= tabs.len or tabs[si].nbeta == 0) {
            try append_empty_dxc(alloc, &dxc_list);
            continue;
        }
        const mt = prij.m_total_per_atom[ai];
        const sp_m_offsets = prij.m_offsets[ai];

        const dxc_m = try alloc.alloc(f64, mt * mt);
        try paw_mod.paw_xc.compute_paw_dij_xc_angular(
            alloc,
            dxc_m,
            paw,
            prij.values[ai],
            mt,
            sp_m_offsets,
            upf.r,
            upf.rab,
            paw.ae_core_density,
            if (upf.nlcc.len > 0) upf.nlcc else null,
            cfg.scf.xc,
            gaunt,
        );
        const rhoij_m = prij.values[ai];
        accumulate_matrix_rhoij_trace(&sum_dxc_rhoij, dxc_m, rhoij_m, mt);
        try dxc_list.append(alloc, dxc_m);

        const dij_h_dc = try alloc.alloc(f64, mt * mt);
        defer alloc.free(dij_h_dc);

        try paw_mod.paw_xc.compute_paw_dij_hartree_multi_l(
            alloc,
            dij_h_dc,
            paw,
            rhoij_m,
            mt,
            sp_m_offsets,
            upf.r,
            upf.rab,
            gaunt,
        );
        accumulate_matrix_rhoij_trace(&sum_dxc_rhoij, dij_h_dc, rhoij_m, mt);
    }
    const dxc = if (dxc_list.items.len > 0) try dxc_list.toOwnedSlice(alloc) else null_blk: {
        dxc_list.deinit(alloc);
        break :null_blk @as(?[][]f64, null);
    };
    return .{ .dxc = dxc, .sum_dxc_rhoij = sum_dxc_rhoij };
}

/// Copy contracted per-atom rhoij for force calculation.
fn duplicate_contracted_rho_ij(
    alloc: std.mem.Allocator,
    prij: *paw_mod.RhoIJ,
) !?[][]f64 {
    var rij_list: std.ArrayList([]f64) = .empty;
    errdefer {
        for (rij_list.items) |r| alloc.free(r);
        rij_list.deinit(alloc);
    }
    for (0..prij.natom) |a| {
        const nb = prij.nbeta_per_atom[a];
        const copy = try alloc.alloc(f64, nb * nb);
        prij.contract_to_radial(a, copy);
        try rij_list.append(alloc, copy);
    }
    if (rij_list.items.len > 0) return try rij_list.toOwnedSlice(alloc);
    rij_list.deinit(alloc);
    return null;
}

/// Top-level PAW result extraction (transfers ownership + builds result arrays).
pub fn extract_paw_results(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    apply_caches: []apply.KpointApplyCache,
    common: *ScfCommon,
    energy_terms: *energy_mod.EnergyTerms,
    out_tabs: *?[]paw_mod.PawTab,
    out_dij: *?[][]f64,
    out_dij_m: *?[][]f64,
    out_dxc: *?[][]f64,
    out_rhoij: *?[][]f64,
) !void {
    if (common.paw_tabs) |tabs| {
        out_tabs.* = tabs;
        common.paw_tabs = null;
    }
    if (apply_caches.len > 0) {
        if (apply_caches[0].nonlocal_ctx) |nc| {
            out_dij.* = try dup_per_atom_dij(alloc, nc.species, .radial);
            out_dij_m.* = try dup_per_atom_dij(alloc, nc.species, .m_resolved);
        }
    }
    if (common.paw_rhoij) |*prij| {
        if (out_tabs.*) |tabs| {
            const dxc_result = try compute_per_atom_dxc_and_dc(
                alloc,
                cfg,
                species,
                atoms,
                prij,
                tabs,
                &common.paw_gaunt.?,
            );
            out_dxc.* = dxc_result.dxc;
            energy_terms.paw_dxc_rhoij = -dxc_result.sum_dxc_rhoij;
            energy_terms.total += energy_terms.paw_dxc_rhoij;
        }
        out_rhoij.* = try duplicate_contracted_rho_ij(alloc, prij);
    }
}

/// Compute the ecutrho cutoff used for in-loop density filtering.
pub fn paw_ecutrho_compute(cfg: *const config.Config) f64 {
    const gs_comp = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_comp * gs_comp;
}

/// Symmetrize density over crystal operations, and PAW rhoij when applicable.
pub fn symmetrize_density_and_rho_ij(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    common: *ScfCommon,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho: []f64,
) !void {
    if (common.sym_ops) |ops| {
        if (ops.len > 1) {
            try density_symmetry.symmetrize_density(alloc, grid, rho, ops, cfg.scf.use_rfft);
        }
    }
    if (common.paw_rhoij) |*prij| {
        if (cfg.scf.symmetry) {
            try paw_scf.symmetrize_rho_ij(alloc, prij, species, atoms);
        }
    }
}

/// Filter density to |G|² < ecutrho sphere, writing the result back into rho.
pub fn filter_augmented_density_in_place(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    ecutrho: f64,
    use_rfft: bool,
) !void {
    const filtered = try potential_mod.filter_density_to_ecutrho(
        alloc,
        grid,
        rho,
        ecutrho,
        use_rfft,
    );
    defer alloc.free(filtered);

    @memcpy(rho, filtered);
}

/// Allocate an augmented density ρ̃ + n̂_hat for potential construction (PAW).
pub fn build_augmented_density(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []const f64,
    common: *ScfCommon,
    atoms: []const hamiltonian.AtomData,
    ecutrho: f64,
    grid_count: usize,
) ![]f64 {
    const aug = try alloc.alloc(f64, grid_count);
    @memcpy(aug, rho);
    if (common.paw_rhoij) |*prij| {
        try paw_scf.add_paw_compensation_charge(
            alloc,
            grid,
            aug,
            prij,
            common.paw_tabs.?,
            atoms,
            ecutrho,
            &common.paw_gaunt.?,
        );
    }
    return aug;
}

/// Build rho_aug for final energy evaluation (PAW only); returns null for non-PAW.
pub fn maybe_build_rho_aug_for_energy(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    common: *ScfCommon,
    atoms: []const hamiltonian.AtomData,
    rho: []const f64,
) !?[]f64 {
    if (!common.is_paw) return null;
    const prij = (common.paw_rhoij orelse return null);
    _ = prij;
    const aug = try alloc.alloc(f64, grid.count());
    @memcpy(aug, rho);
    const gs_en = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    const ecutrho_scf = cfg.scf.ecut_ry * gs_en * gs_en;
    try paw_scf.add_paw_compensation_charge(
        alloc,
        grid,
        aug,
        &common.paw_rhoij.?,
        common.paw_tabs.?,
        atoms,
        ecutrho_scf,
        &common.paw_gaunt.?,
    );
    const filtered = try potential_mod.filter_density_to_ecutrho(
        alloc,
        grid,
        aug,
        ecutrho_scf,
        cfg.scf.use_rfft,
    );
    alloc.free(aug);
    return filtered;
}
