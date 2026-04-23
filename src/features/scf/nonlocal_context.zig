const std = @import("std");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");

pub const NonlocalSpecies = struct {
    species_index: usize,
    beta_count: usize,
    g_count: usize,
    l_list: []i32,
    coeffs: []const f64,
    m_offsets: []usize,
    m_counts: []usize,
    m_total: usize,
    phi: []f64,
    // PAW fields: overlap coefficients q_ij and mutable D_ij buffer
    overlap_coeffs: ?[]const f64 = null, // q_ij = S_ij - delta_ij (PAW overlap correction)
    dij_buf: ?[]f64 = null, // Owned mutable D_ij buffer (coeffs points here for PAW)
    dij_per_atom: ?[][]f64 = null, // Per-atom D_ij: [atom_of_species][nbeta*nbeta]
    dij_m_per_atom: ?[][]f64 = null, // Per-atom m-resolved D: [atom_of_species][m_total*m_total]

    pub fn deinit(self: *NonlocalSpecies, alloc: std.mem.Allocator) void {
        if (self.l_list.len > 0) alloc.free(self.l_list);
        if (self.m_offsets.len > 0) alloc.free(self.m_offsets);
        if (self.m_counts.len > 0) alloc.free(self.m_counts);
        if (self.phi.len > 0) alloc.free(self.phi);
        if (self.dij_m_per_atom) |dpa| {
            for (dpa) |buf| alloc.free(buf);
            alloc.free(dpa);
        }
        if (self.dij_per_atom) |dpa| {
            for (dpa) |buf| alloc.free(buf);
            alloc.free(dpa);
        }
        if (self.dij_buf) |buf| alloc.free(buf);
        if (self.overlap_coeffs) |oc| alloc.free(@constCast(oc));
    }
};

pub const NonlocalContext = struct {
    species: []NonlocalSpecies,
    max_m_total: usize,
    has_paw: bool = false, // true if any species has PAW overlap

    pub fn deinit(self: *NonlocalContext, alloc: std.mem.Allocator) void {
        for (self.species) |*entry| {
            entry.deinit(alloc);
        }
        if (self.species.len > 0) alloc.free(self.species);
    }

    /// Check if any species has PAW overlap coefficients.
    pub fn has_paw_overlap(self: *const NonlocalContext) bool {
        return self.has_paw;
    }

    /// Update D_ij coefficients for a PAW species (by species_index).
    /// new_dij must have length beta_count * beta_count.
    pub fn update_dij(self: *NonlocalContext, species_index: usize, new_dij: []const f64) void {
        for (self.species) |*entry| {
            if (entry.species_index == species_index) {
                if (entry.dij_buf) |buf| {
                    const n = @min(buf.len, new_dij.len);
                    @memcpy(buf[0..n], new_dij[0..n]);
                }
                return;
            }
        }
    }

    /// Ensure per-atom D_ij arrays are allocated for a PAW species.
    /// Lazy init: does nothing if already allocated with matching natom.
    pub fn ensure_dij_per_atom(
        self: *NonlocalContext,
        alloc: std.mem.Allocator,
        species_index: usize,
        natom: usize,
    ) !void {
        for (self.species) |*entry| {
            if (entry.species_index == species_index) {
                if (entry.dij_per_atom != null) return; // already allocated
                const n_ij = entry.beta_count * entry.beta_count;
                const dpa = try alloc.alloc([]f64, natom);
                for (0..natom) |a| {
                    dpa[a] = try alloc.alloc(f64, n_ij);
                    @memcpy(dpa[a], entry.coeffs[0..n_ij]);
                }
                entry.dij_per_atom = dpa;
                return;
            }
        }
    }

    /// Update D_ij for a specific atom of a PAW species.
    pub fn update_dij_atom(
        self: *NonlocalContext,
        species_index: usize,
        atom_of_species: usize,
        new_dij: []const f64,
    ) void {
        for (self.species) |*entry| {
            if (entry.species_index == species_index) {
                if (entry.dij_per_atom) |dpa| {
                    const n = @min(dpa[atom_of_species].len, new_dij.len);
                    @memcpy(dpa[atom_of_species][0..n], new_dij[0..n]);
                }
                return;
            }
        }
    }

    /// Ensure per-atom m-resolved D_ij arrays are allocated for a PAW species.
    pub fn ensure_dij_m_per_atom(
        self: *NonlocalContext,
        alloc: std.mem.Allocator,
        species_index: usize,
        natom: usize,
    ) !void {
        for (self.species) |*entry| {
            if (entry.species_index == species_index) {
                if (entry.dij_m_per_atom != null) return;
                const mt = entry.m_total;
                const n_m = mt * mt;
                const dpa = try alloc.alloc([]f64, natom);
                for (0..natom) |a| {
                    dpa[a] = try alloc.alloc(f64, n_m);
                    @memset(dpa[a], 0.0);
                }
                entry.dij_m_per_atom = dpa;
                return;
            }
        }
    }

    /// Update m-resolved D_ij for a specific atom of a PAW species.
    pub fn update_dij_m_atom(
        self: *NonlocalContext,
        species_index: usize,
        atom_of_species: usize,
        new_dij_m: []const f64,
    ) void {
        for (self.species) |*entry| {
            if (entry.species_index == species_index) {
                if (entry.dij_m_per_atom) |dpa| {
                    const n = @min(dpa[atom_of_species].len, new_dij_m.len);
                    @memcpy(dpa[atom_of_species][0..n], new_dij_m[0..n]);
                }
                return;
            }
        }
    }
};

/// Public wrapper for build_nonlocal_context (used by kpoints.zig for caching).
pub fn build_nonlocal_context_pub(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    gvecs: []const plane_wave.GVector,
) !?NonlocalContext {
    return build_nonlocal_context(alloc, species, gvecs);
}

/// Build NonlocalContext using pre-computed radial tables for fast evaluation.
pub fn build_nonlocal_context_with_tables(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    gvecs: []const plane_wave.GVector,
    radial_tables: []const nonlocal.RadialTableSet,
) !?NonlocalContext {
    var list: std.ArrayList(NonlocalSpecies) = .empty;
    errdefer {
        for (list.items) |*entry| {
            entry.deinit(alloc);
        }
        list.deinit(alloc);
    }

    var max_total: usize = 0;
    var table_idx: usize = 0;
    for (species, 0..) |entry, s| {
        const upf = entry.upf.*;
        if (upf.beta.len == 0 or upf.dij.len == 0) continue;
        const tables = if (table_idx < radial_tables.len) &radial_tables[table_idx] else null;
        const nl = try build_nonlocal_species_with_tables(alloc, s, upf, gvecs, tables);
        if (nl.m_total > max_total) max_total = nl.m_total;
        try list.append(alloc, nl);
        table_idx += 1;
    }

    if (list.items.len == 0) return null;
    const slice = try list.toOwnedSlice(alloc);
    return NonlocalContext{ .species = slice, .max_m_total = max_total };
}

/// Build NonlocalContext with PAW support (mutable D_ij and overlap coefficients).
/// paw_tabs should have one entry per species. Species with nbeta=0 are treated as non-PAW.
pub fn build_nonlocal_context_paw(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    gvecs: []const plane_wave.GVector,
    radial_tables: ?[]const nonlocal.RadialTableSet,
    paw_tabs: []const paw_mod.PawTab,
) !?NonlocalContext {
    var list: std.ArrayList(NonlocalSpecies) = .empty;
    errdefer {
        for (list.items) |*entry| {
            entry.deinit(alloc);
        }
        list.deinit(alloc);
    }

    var max_total: usize = 0;
    var table_idx: usize = 0;
    var has_paw = false;

    for (species, 0..) |entry, s| {
        const upf = entry.upf.*;
        if (upf.beta.len == 0 or upf.dij.len == 0) continue;
        const tables = if (radial_tables) |rt|
            (if (table_idx < rt.len) &rt[table_idx] else null)
        else
            null;
        var nl = try build_nonlocal_species_with_tables(alloc, s, upf, gvecs, tables);

        // If this species has PAW data, set up mutable D_ij buffer and overlap coefficients
        if (s < paw_tabs.len and paw_tabs[s].nbeta > 0) {
            const tab = &paw_tabs[s];
            const n_ij = tab.nbeta * tab.nbeta;
            // Allocate mutable D_ij buffer, initialized with UPF dij
            const dij_buf = try alloc.alloc(f64, n_ij);
            @memcpy(dij_buf, upf.dij[0..n_ij]);
            nl.dij_buf = dij_buf;
            nl.coeffs = dij_buf; // Point coeffs to mutable buffer

            // Allocate and compute q_ij = S_ij - delta_ij (overlap correction)
            const qij = try alloc.alloc(f64, n_ij);
            for (0..tab.nbeta) |i| {
                for (0..tab.nbeta) |j| {
                    const delta: f64 = if (i == j) 1.0 else 0.0;
                    qij[i * tab.nbeta + j] = tab.sij[i * tab.nbeta + j] - delta;
                }
            }
            nl.overlap_coeffs = qij;
            has_paw = true;
        }

        if (nl.m_total > max_total) max_total = nl.m_total;
        try list.append(alloc, nl);
        table_idx += 1;
    }

    if (list.items.len == 0) return null;
    const slice = try list.toOwnedSlice(alloc);
    return NonlocalContext{ .species = slice, .max_m_total = max_total, .has_paw = has_paw };
}

fn build_nonlocal_context(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    gvecs: []const plane_wave.GVector,
) !?NonlocalContext {
    var list: std.ArrayList(NonlocalSpecies) = .empty;
    errdefer {
        for (list.items) |*entry| {
            entry.deinit(alloc);
        }
        list.deinit(alloc);
    }

    var max_total: usize = 0;
    for (species, 0..) |entry, s| {
        const upf = entry.upf.*;
        if (upf.beta.len == 0 or upf.dij.len == 0) continue;
        const nl = try build_nonlocal_species(alloc, s, upf, gvecs);
        if (nl.m_total > max_total) max_total = nl.m_total;
        try list.append(alloc, nl);
    }

    if (list.items.len == 0) return null;
    const slice = try list.toOwnedSlice(alloc);
    return NonlocalContext{ .species = slice, .max_m_total = max_total };
}

fn build_nonlocal_species(
    alloc: std.mem.Allocator,
    species_index: usize,
    upf: pseudo.UpfData,
    gvecs: []const plane_wave.GVector,
) !NonlocalSpecies {
    return build_nonlocal_species_with_tables(alloc, species_index, upf, gvecs, null);
}

fn build_nonlocal_species_with_tables(
    alloc: std.mem.Allocator,
    species_index: usize,
    upf: pseudo.UpfData,
    gvecs: []const plane_wave.GVector,
    radial_tables: ?*const nonlocal.RadialTableSet,
) !NonlocalSpecies {
    const beta_count = upf.beta.len;
    const g_count = gvecs.len;
    if (beta_count == 0 or upf.dij.len == 0) return error.InvalidPseudopotential;
    if (upf.dij.len != beta_count * beta_count) return error.InvalidPseudopotential;

    const l_list = try alloc.alloc(i32, beta_count);
    errdefer alloc.free(l_list);
    const m_offsets = try alloc.alloc(usize, beta_count);
    errdefer alloc.free(m_offsets);
    const m_counts = try alloc.alloc(usize, beta_count);
    errdefer alloc.free(m_counts);

    const radial = try alloc.alloc(f64, beta_count * g_count);
    defer alloc.free(radial);

    const total_m = fill_nonlocal_radial_terms(
        upf,
        gvecs,
        radial_tables,
        l_list,
        m_offsets,
        m_counts,
        radial,
    );

    const phi = try alloc.alloc(f64, total_m * g_count);
    errdefer alloc.free(phi);

    fill_nonlocal_phi(gvecs, l_list, m_offsets, m_counts, radial, phi);

    return NonlocalSpecies{
        .species_index = species_index,
        .beta_count = beta_count,
        .g_count = g_count,
        .l_list = l_list,
        .coeffs = upf.dij,
        .m_offsets = m_offsets,
        .m_counts = m_counts,
        .m_total = total_m,
        .phi = phi,
    };
}

fn fill_nonlocal_radial_terms(
    upf: pseudo.UpfData,
    gvecs: []const plane_wave.GVector,
    radial_tables: ?*const nonlocal.RadialTableSet,
    l_list: []i32,
    m_offsets: []usize,
    m_counts: []usize,
    radial: []f64,
) usize {
    const beta_count = upf.beta.len;
    const g_count = gvecs.len;
    var total_m: usize = 0;
    var b: usize = 0;
    while (b < beta_count) : (b += 1) {
        const l_val = upf.beta[b].l orelse 0;
        l_list[b] = l_val;
        const m_count = @as(usize, @intCast(2 * l_val + 1));
        m_offsets[b] = total_m;
        m_counts[b] = m_count;
        total_m += m_count;
        if (radial_tables) |tables| {
            for (0..g_count) |g| {
                const gmag = math.Vec3.norm(gvecs[g].kpg);
                radial[b * g_count + g] = tables.tables[b].eval(gmag);
            }
        } else {
            var g: usize = 0;
            while (g < g_count) : (g += 1) {
                const gmag = math.Vec3.norm(gvecs[g].kpg);
                radial[b * g_count + g] = nonlocal.radial_projector(
                    upf.beta[b].values,
                    upf.r,
                    upf.rab,
                    l_val,
                    gmag,
                );
            }
        }
    }
    return total_m;
}

fn fill_nonlocal_phi(
    gvecs: []const plane_wave.GVector,
    l_list: []const i32,
    m_offsets: []const usize,
    m_counts: []const usize,
    radial: []const f64,
    phi: []f64,
) void {
    const g_count = gvecs.len;
    var b: usize = 0;
    while (b < l_list.len) : (b += 1) {
        const l_val = l_list[b];
        const m_count = m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const m = @as(i32, @intCast(m_idx)) - l_val;
            const offset = m_offsets[b] + m_idx;
            var g: usize = 0;
            while (g < g_count) : (g += 1) {
                const kpg = gvecs[g].kpg;
                const ylm = nonlocal.real_spherical_harmonic(l_val, m, kpg.x, kpg.y, kpg.z);
                phi[offset * g_count + g] = 4.0 * std.math.pi * radial[b * g_count + g] * ylm;
            }
        }
    }
}
