const std = @import("std");
const math = @import("../math/math.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const xyz = @import("../structure/xyz.zig");

pub const AtomData = struct {
    position: math.Vec3,
    species_index: usize,
};

pub const SpeciesEntry = struct {
    symbol: []const u8,
    upf: *pseudo.UpfData,
    z_valence: f64,
    epsatm_ry: f64,
    is_paw: bool = false,

    pub fn deinit(self: *SpeciesEntry) void {
        _ = self;
    }
};

pub const PotentialGrid = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    min_h: i32,
    min_k: i32,
    min_l: i32,
    values: []math.Complex,

    /// Free potential grid.
    pub fn deinit(self: *PotentialGrid, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) {
            alloc.free(self.values);
        }
    }

    /// Return value for (h,k,l) or zero if out of range.
    /// Return value for (h,k,l). Returns zero if out of grid range.
    pub fn valueAt(self: PotentialGrid, h: i32, k: i32, l: i32) math.Complex {
        const idx = self.indexOf(h, k, l) orelse return math.complex.init(0.0, 0.0);
        return self.values[idx];
    }

    /// Alias for valueAt. Returns zero if out of grid range.
    pub fn valueAtWrapped(self: PotentialGrid, h: i32, k: i32, l: i32) math.Complex {
        return self.valueAt(h, k, l);
    }

    /// Convert integer indices to flat index.
    pub fn indexOf(self: PotentialGrid, h: i32, k: i32, l: i32) ?usize {
        const hi = index1D(h, self.nx, self.min_h) orelse return null;
        const ki = index1D(k, self.ny, self.min_k) orelse return null;
        const li = index1D(l, self.nz, self.min_l) orelse return null;
        return hi + self.nx * (ki + self.ny * li);
    }
};

const SpeciesProjectors = struct {
    radial: []f64,
    l: []i32,
    beta_count: usize,
    g_count: usize,

    /// Free projector cache arrays.
    pub fn deinit(self: *SpeciesProjectors, alloc: std.mem.Allocator) void {
        if (self.radial.len > 0) alloc.free(self.radial);
        if (self.l.len > 0) alloc.free(self.l);
    }
};

const NonlocalMode = enum {
    dij,
    qij,
};

/// Build species table from parsed pseudopotentials.
pub fn buildSpeciesEntries(alloc: std.mem.Allocator, items: []pseudo.Parsed) ![]SpeciesEntry {
    var list: std.ArrayList(SpeciesEntry) = .empty;
    errdefer {
        for (list.items) |*entry| {
            entry.deinit();
        }
        list.deinit(alloc);
    }

    for (items) |*item| {
        if (item.upf) |*upf| {
            const z = item.header.z_valence orelse return error.MissingZValence;
            try list.append(alloc, .{
                .symbol = item.spec.element,
                .upf = upf,
                .z_valence = z,
                .epsatm_ry = form_factor.computeEpsatm(upf.*, z),
                .is_paw = upf.paw != null,
            });
        } else {
            return error.MissingUpfData;
        }
    }
    return try list.toOwnedSlice(alloc);
}

/// Release species table resources.
pub fn deinitSpeciesEntries(alloc: std.mem.Allocator, entries: []SpeciesEntry) void {
    if (entries.len == 0) return;
    for (entries) |*entry| {
        entry.deinit();
    }
    alloc.free(entries);
}

/// Map atoms to species indices and convert to bohr.
pub fn buildAtomData(
    alloc: std.mem.Allocator,
    atoms: []xyz.Atom,
    unit_scale_bohr: f64,
    species: []const SpeciesEntry,
) ![]AtomData {
    const data = try alloc.alloc(AtomData, atoms.len);
    errdefer alloc.free(data);
    for (atoms, 0..) |atom, i| {
        const idx = findSpeciesIndex(species, atom.symbol) orelse return error.MissingPseudopotential;
        data[i] = .{
            .position = math.Vec3.scale(atom.position, unit_scale_bohr),
            .species_index = idx,
        };
    }
    return data;
}

/// Locate species entry by symbol.
pub fn findSpeciesIndex(species: []const SpeciesEntry, symbol: []const u8) ?usize {
    for (species, 0..) |entry, i| {
        if (std.mem.eql(u8, entry.symbol, symbol)) return i;
    }
    return null;
}

/// Build Hamiltonian with local and nonlocal terms.
pub fn buildHamiltonian(
    alloc: std.mem.Allocator,
    gvecs: []plane_wave.GVector,
    species: []const SpeciesEntry,
    atoms: []const AtomData,
    inv_volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    extra: ?PotentialGrid,
) ![]math.Complex {
    const n = gvecs.len;
    if (n == 0) return error.NoPlaneWaves;
    const h = try alloc.alloc(math.Complex, n * n);
    errdefer alloc.free(h);

    var j: usize = 0;
    while (j < n) : (j += 1) {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const q = math.Vec3.sub(gvecs[i].cart, gvecs[j].cart);
            var value = try localPotential(q, gvecs[i], gvecs[j], species, atoms, inv_volume, local_cfg, extra);
            if (i == j) {
                value.r += gvecs[i].kinetic;
            }
            h[i + j * n] = value;
        }
    }

    try addNonlocalContribution(alloc, h, gvecs, species, atoms, inv_volume, .dij);
    return h;
}

/// Build nonlocal matrix for projector terms only.
pub fn buildNonlocalMatrix(
    alloc: std.mem.Allocator,
    gvecs: []plane_wave.GVector,
    species: []const SpeciesEntry,
    atoms: []const AtomData,
    inv_volume: f64,
) ![]math.Complex {
    const n = gvecs.len;
    const h = try alloc.alloc(math.Complex, n * n);
    errdefer alloc.free(h);
    @memset(h, math.complex.init(0.0, 0.0));
    try addNonlocalContribution(alloc, h, gvecs, species, atoms, inv_volume, .dij);
    return h;
}

/// Build overlap matrix for generalized eigenproblem.
pub fn buildOverlapMatrix(
    alloc: std.mem.Allocator,
    gvecs: []plane_wave.GVector,
    species: []const SpeciesEntry,
    atoms: []const AtomData,
    inv_volume: f64,
) ![]math.Complex {
    const n = gvecs.len;
    const s = try alloc.alloc(math.Complex, n * n);
    errdefer alloc.free(s);
    initIdentity(s, n);
    try addNonlocalContribution(alloc, s, gvecs, species, atoms, inv_volume, .qij);
    return s;
}

/// Initialize matrix to identity.
pub fn initIdentity(m: []math.Complex, n: usize) void {
    var j: usize = 0;
    while (j < n) : (j += 1) {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const idx = i + j * n;
            m[idx] = if (i == j) math.complex.init(1.0, 0.0) else math.complex.init(0.0, 0.0);
        }
    }
}

/// Add nonlocal pseudopotential to Hamiltonian.
fn addNonlocalContribution(
    alloc: std.mem.Allocator,
    h: []math.Complex,
    gvecs: []plane_wave.GVector,
    species: []const SpeciesEntry,
    atoms: []const AtomData,
    inv_volume: f64,
    mode: NonlocalMode,
) !void {
    const n = gvecs.len;
    if (n == 0) return;

    var s: usize = 0;
    while (s < species.len) : (s += 1) {
        const entry = &species[s];
        const upf = entry.upf.*;
        const beta_count = upf.beta.len;
        const coeffs = switch (mode) {
            .dij => upf.dij,
            .qij => upf.qij,
        };
        if (beta_count == 0 or coeffs.len == 0) continue;
        if (coeffs.len != beta_count * beta_count) return error.InvalidPseudopotential;

        var proj = try buildProjectors(alloc, upf, gvecs);
        defer proj.deinit(alloc);

        // Process each atom individually to avoid incorrect cross terms.
        // The nonlocal structure factor exp(-i*(G_i-G_j)·r_a) must be computed
        // per-atom: Σ_a exp(-i*G_i·r_a)*exp(+i*G_j·r_a), NOT as
        // [Σ_a exp(-i*G_i·r_a)] * [Σ_b exp(+i*G_j·r_b)].
        const atom_phase = try alloc.alloc(math.Complex, n);
        defer alloc.free(atom_phase);

        for (atoms) |atom| {
            if (atom.species_index != s) continue;

            for (gvecs, 0..) |g, idx| {
                atom_phase[idx] = math.complex.expi(-math.Vec3.dot(g.cart, atom.position));
            }

            var j: usize = 0;
            while (j < n) : (j += 1) {
                const phase_j_conj = math.complex.conj(atom_phase[j]);
                var i: usize = 0;
                while (i < n) : (i += 1) {
                    const phase = math.complex.mul(atom_phase[i], phase_j_conj);
                    if (phase.r == 0.0 and phase.i == 0.0) continue;

                    const nonlocal_value = computeNonlocalValue(
                        proj,
                        i,
                        j,
                        gvecs[i].kpg,
                        gvecs[j].kpg,
                        coeffs,
                    );
                    if (nonlocal_value == 0.0) continue;
                    const add = math.complex.scale(phase, inv_volume * nonlocal_value);
                    h[i + j * n] = math.complex.add(h[i + j * n], add);
                }
            }
        }
    }
}

/// Build projector cache for a species and basis.
fn buildProjectors(
    alloc: std.mem.Allocator,
    upf: pseudo.UpfData,
    gvecs: []plane_wave.GVector,
) !SpeciesProjectors {
    const beta_count = upf.beta.len;
    const g_count = gvecs.len;
    var radial = try alloc.alloc(f64, beta_count * g_count);
    errdefer alloc.free(radial);
    var l_list = try alloc.alloc(i32, beta_count);
    errdefer alloc.free(l_list);

    var b: usize = 0;
    while (b < beta_count) : (b += 1) {
        const l_val = upf.beta[b].l orelse 0;
        l_list[b] = l_val;
        var g: usize = 0;
        while (g < g_count) : (g += 1) {
            const gmag = math.Vec3.norm(gvecs[g].kpg);
            const value = nonlocal.radialProjector(upf.beta[b].values, upf.r, upf.rab, l_val, gmag);
            radial[b * g_count + g] = value;
        }
    }

    return SpeciesProjectors{
        .radial = radial,
        .l = l_list,
        .beta_count = beta_count,
        .g_count = g_count,
    };
}

/// Compute nonlocal term contribution for a G pair.
fn computeNonlocalValue(
    proj: SpeciesProjectors,
    gi: usize,
    gj: usize,
    kpg_i: math.Vec3,
    kpg_j: math.Vec3,
    coeffs: []const f64,
) f64 {
    const gmag_i = math.Vec3.norm(kpg_i);
    const gmag_j = math.Vec3.norm(kpg_j);
    var cos_gamma = if (gmag_i == 0.0 or gmag_j == 0.0)
        1.0
    else
        math.Vec3.dot(kpg_i, kpg_j) / (gmag_i * gmag_j);

    if (cos_gamma > 1.0) cos_gamma = 1.0;
    if (cos_gamma < -1.0) cos_gamma = -1.0;

    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < proj.beta_count) : (i += 1) {
        const li = proj.l[i];
        const angular = nonlocal.angularFactor(li, cos_gamma);
        var j: usize = 0;
        while (j < proj.beta_count) : (j += 1) {
            if (proj.l[j] != li) continue;
            const coeff = coeffs[i * proj.beta_count + j];
            if (coeff == 0.0) continue;
            const radial_i = proj.radial[i * proj.g_count + gi];
            const radial_j = proj.radial[j * proj.g_count + gj];
            sum += coeff * angular * radial_i * radial_j;
        }
    }
    return sum;
}

/// Compute phase sum for atoms of one species.
fn phaseSum(q: math.Vec3, atoms: []const AtomData, species_index: usize) math.Complex {
    var sum = math.complex.init(0.0, 0.0);
    for (atoms) |atom| {
        if (atom.species_index != species_index) continue;
        const phase = math.complex.expi(-math.Vec3.dot(q, atom.position));
        sum = math.complex.add(sum, phase);
    }
    return sum;
}

/// Compute local potential in reciprocal space for q.
fn localPotential(
    q: math.Vec3,
    gi: plane_wave.GVector,
    gj: plane_wave.GVector,
    species: []const SpeciesEntry,
    atoms: []const AtomData,
    inv_volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    extra: ?PotentialGrid,
) !math.Complex {
    var value = try ionicLocalPotential(q, species, atoms, inv_volume, local_cfg);
    if (extra) |pot| {
        const dh = gi.h - gj.h;
        const dk = gi.k - gj.k;
        const dl = gi.l - gj.l;
        const add = pot.valueAtWrapped(dh, dk, dl);
        value = math.complex.add(value, add);
    }
    return value;
}

/// Compute ionic local potential in reciprocal space.
/// Note: G=0 component is skipped as it cancels with the Hartree G=0 divergence
/// in neutral systems. The G=0 ionic potential is handled separately in the
/// energy calculation via Ewald summation.
pub fn ionicLocalPotential(
    q: math.Vec3,
    species: []const SpeciesEntry,
    atoms: []const AtomData,
    inv_volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
) !math.Complex {
    return ionicLocalPotentialWithTable(q, species, atoms, inv_volume, local_cfg, null);
}

/// Fast version using pre-computed form factor tables.
pub fn ionicLocalPotentialWithTable(
    q: math.Vec3,
    species: []const SpeciesEntry,
    atoms: []const AtomData,
    inv_volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
) !math.Complex {
    const q_mag = math.Vec3.norm(q);
    // Skip G=0 component - it cancels with Hartree G=0 in neutral systems
    if (q_mag < 1e-10) {
        return math.complex.init(0.0, 0.0);
    }
    var sum = math.complex.init(0.0, 0.0);
    for (atoms) |atom| {
        const vq = if (ff_tables) |tables|
            tables[atom.species_index].eval(q_mag)
        else blk: {
            const entry = &species[atom.species_index];
            break :blk localVqCached(entry, q_mag, local_cfg);
        };
        const phase = math.complex.expi(-math.Vec3.dot(q, atom.position));
        sum = math.complex.add(sum, math.complex.scale(phase, vq));
    }
    return math.complex.scale(sum, inv_volume);
}

pub fn localFormFactor(
    entry: *const SpeciesEntry,
    q: f64,
    local_cfg: local_potential.LocalPotentialConfig,
) f64 {
    return switch (local_cfg.mode) {
        .tail => form_factor.localVqWithTail(entry.upf.*, entry.z_valence, q),
        .ewald => form_factor.localVqEwald(entry.upf.*, entry.z_valence, q, local_cfg.alpha),
        .short_range => form_factor.localVqShortRange(entry.upf.*, entry.z_valence, q),
    };
}

/// Get local form factor for a species with Coulomb tail correction.
fn localVqCached(
    entry: *const SpeciesEntry,
    q: f64,
    local_cfg: local_potential.LocalPotentialConfig,
) f64 {
    return localFormFactor(entry, q, local_cfg);
}

/// Convert integer coordinate into flat index component.
fn index1D(value: i32, n: usize, min: i32) ?usize {
    const max = min + @as(i32, @intCast(n)) - 1;
    if (value < min or value > max) return null;
    return @intCast(value - min);
}
