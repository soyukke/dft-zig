const std = @import("std");
const config = @import("../config/config.zig");
const core_density = @import("core_density.zig");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kmesh_mod = @import("../kpoints/kpoints.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const mixing = @import("mixing.zig");
const model_mod = @import("../dft/model.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const potential_mod = @import("potential.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const util = @import("util.zig");

const KPoint = symmetry.KPoint;
const Grid = grid_mod.Grid;
const PulayMixer = mixing.PulayMixer;
const ScfLog = logging.ScfLog;
const has_nonlocal = util.has_nonlocal;
const has_paw = util.has_paw;
const total_electrons = util.total_electrons;

pub const InitParams = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    model: *const model_mod.Model,
    ff_tables: ?[]const form_factor.LocalFormFactorTable = null,
};

/// Common state shared by both spin-unpolarized and spin-polarized SCF.
pub const ScfCommon = struct {
    alloc: std.mem.Allocator,
    cfg: config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    grid: Grid,
    total_electrons: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    ionic: hamiltonian.PotentialGrid,
    log: ScfLog,
    kpoints: []KPoint,
    sym_ops: ?[]const symmetry.SymOp,
    rho_core: ?[]f64,
    radial_tables_buf: ?[]nonlocal_mod.RadialTableSet,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    pulay_mixer: ?PulayMixer,
    coulomb_r_cut: ?f64, // Cutoff radius for isolated systems (null = periodic)
    // PAW fields
    paw_tabs: ?[]paw_mod.PawTab = null, // One per species (only for PAW species)
    paw_rhoij: ?paw_mod.RhoIJ = null, // Occupation matrix (one per atom, m-resolved)
    paw_gaunt: ?paw_mod.GauntTable = null, // Gaunt coefficient table for multi-L PAW
    is_paw: bool = false,

    pub fn deinit(self: *ScfCommon) void {
        self.ionic.deinit(self.alloc);
        self.log.deinit();
        self.alloc.free(self.kpoints);
        if (self.sym_ops) |ops| self.alloc.free(ops);
        if (self.rho_core) |values| self.alloc.free(values);
        if (self.radial_tables_buf) |buf| {
            for (buf) |*t| {
                if (t.tables.len > 0) t.deinit(self.alloc);
            }
            self.alloc.free(buf);
        }
        if (self.pulay_mixer) |*mixer| mixer.deinit();
        if (self.paw_tabs) |tabs| {
            for (tabs) |*t| t.deinit(self.alloc);
            self.alloc.free(tabs);
        }
        if (self.paw_rhoij) |*rij| rij.deinit(self.alloc);
        if (self.paw_gaunt) |*gt| gt.deinit(self.alloc);
    }
};

/// Initialize common SCF state shared by spin-unpolarized and spin-polarized paths.
/// Generate the k-point mesh respecting boundary=isolated (Gamma-only) and symmetry options.
fn generate_kpoints_for_config(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    recip: math.Mat3,
    cell: math.Mat3,
    atoms: []const hamiltonian.AtomData,
) ![]KPoint {
    if (cfg.boundary == .isolated) {
        const gamma_kpoints = try alloc.alloc(KPoint, 1);
        gamma_kpoints[0] = KPoint{
            .k_frac = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .k_cart = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .weight = 1.0,
        };
        return gamma_kpoints;
    }
    const shift: math.Vec3 = .{
        .x = cfg.scf.kmesh_shift[0],
        .y = cfg.scf.kmesh_shift[1],
        .z = cfg.scf.kmesh_shift[2],
    };
    if (cfg.scf.symmetry) {
        return try kmesh_mod.generate_kmesh_symmetry(
            alloc,
            io,
            cfg.scf.kmesh,
            shift,
            recip,
            cell,
            atoms,
            cfg.scf.time_reversal,
        );
    }
    return try kmesh_mod.generate_kmesh(alloc, cfg.scf.kmesh, recip, shift);
}

/// Build per-species radial projector tables used by the nonlocal context.
fn build_radial_tables_for_species(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    ecut_ry: f64,
) !?[]nonlocal_mod.RadialTableSet {
    const g_max = @sqrt(2.0 * ecut_ry) * 1.5;
    var buf = try alloc.alloc(nonlocal_mod.RadialTableSet, species.len);
    for (buf) |*t| {
        t.* = .{ .tables = &[_]nonlocal_mod.RadialTable{} };
    }
    errdefer {
        for (buf) |*t| {
            if (t.tables.len > 0) t.deinit(alloc);
        }
        alloc.free(buf);
    }
    for (species, 0..) |entry, si| {
        const upf = entry.upf.*;
        if (upf.beta.len == 0 or upf.dij.len == 0) {
            buf[si] = .{ .tables = &[_]nonlocal_mod.RadialTable{} };
            continue;
        }
        buf[si] = try nonlocal_mod.RadialTableSet.init(alloc, upf.beta, upf.r, upf.rab, g_max);
    }
    return buf;
}

/// Build PAW per-species tables and the m-resolved RhoIJ matrix.
fn build_paw_tables_and_rho_ij(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ecut_ry: f64,
) !struct { tabs: []paw_mod.PawTab, rhoij: paw_mod.RhoIJ } {
    const q_max = @sqrt(2.0 * ecut_ry) * 1.5;
    var tabs = try alloc.alloc(paw_mod.PawTab, species.len);
    for (tabs) |*tab| {
        tab.* = empty_paw_tab();
    }
    errdefer {
        for (tabs) |*t| t.deinit(alloc);
        alloc.free(tabs);
    }
    for (species, 0..) |entry, si| {
        if (entry.upf.paw) |paw| {
            tabs[si] = try paw_mod.PawTab.init(alloc, paw, entry.upf.r, entry.upf.rab, q_max);
        } else {
            tabs[si] = empty_paw_tab();
        }
    }
    const natom = atoms.len;
    const nbeta_list = try alloc.alloc(usize, natom);
    defer alloc.free(nbeta_list);

    const l_lists = try alloc.alloc([]const i32, natom);
    defer alloc.free(l_lists);

    for (0..natom) |a| {
        const sp = atoms[a].species_index;
        nbeta_list[a] = tabs[sp].nbeta;
        l_lists[a] = tabs[sp].l_list;
    }
    const rhoij = try paw_mod.RhoIJ.init(alloc, natom, nbeta_list, l_lists);
    return .{ .tabs = tabs, .rhoij = rhoij };
}

fn empty_paw_tab() paw_mod.PawTab {
    return .{
        .sij = &[_]f64{},
        .kij = &[_]f64{},
        .qijl_form = &[_]f64{},
        .n_qijl_entries = 0,
        .qijl_indices = &[_]paw_mod.PawTab.QijlIndex{},
        .n_qpoints = 0,
        .dq = 0.0,
        .nbeta = 0,
        .l_list = &[_]i32{},
    };
}

/// Determine lmax_proj, lmax_aug from PAW tabs and build the Gaunt coefficient table.
fn build_paw_gaunt_table(
    alloc: std.mem.Allocator,
    tabs: []const paw_mod.PawTab,
) !paw_mod.GauntTable {
    var lmax_proj: usize = 0;
    var lmax_aug: usize = 0;
    for (tabs) |tab| {
        if (tab.nbeta == 0) continue;
        for (tab.l_list) |l| {
            const lu = @as(usize, @intCast(l));
            if (lu > lmax_proj) lmax_proj = lu;
        }
        for (0..tab.n_qijl_entries) |e| {
            if (tab.qijl_indices[e].l > lmax_aug) lmax_aug = tab.qijl_indices[e].l;
        }
    }
    return try paw_mod.GauntTable.init(alloc, lmax_proj, lmax_aug);
}

/// Compute ecutrho (spherical cutoff for density) for PAW systems; otherwise null.
pub fn ecutrho_for_paw(cfg: *const config.Config, is_paw: bool) ?f64 {
    if (!is_paw) return null;
    const gs_val = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_val * gs_val;
}

/// Build the optional radial tables (for nonlocal context warm-start).
fn maybe_build_radial_tables(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
) !?[]nonlocal_mod.RadialTableSet {
    const nonlocal_enabled_run = cfg.scf.enable_nonlocal and has_nonlocal(species);
    if (!nonlocal_enabled_run) return null;
    return try build_radial_tables_for_species(alloc, species, cfg.scf.ecut_ry);
}

/// Free radial tables (helper for errdefer blocks).
fn free_radial_tables(alloc: std.mem.Allocator, buf: []nonlocal_mod.RadialTableSet) void {
    for (buf) |*t| {
        if (t.tables.len > 0) t.deinit(alloc);
    }
    alloc.free(buf);
}

/// Free PAW bundle contents (helper for errdefer blocks).
fn free_paw_bundle(alloc: std.mem.Allocator, bundle: *PawBundle) void {
    if (bundle.tabs) |tabs| {
        for (tabs) |*t| t.deinit(alloc);
        alloc.free(tabs);
    }
    if (bundle.rhoij) |*rij| rij.deinit(alloc);
    if (bundle.gaunt) |*gt| gt.deinit(alloc);
}

/// Build optional NLCC core density.
fn maybe_build_rho_core(
    alloc: std.mem.Allocator,
    grid: grid_mod.Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) !?[]f64 {
    if (!core_density.has_nlcc(species)) return null;
    return try core_density.build_core_density(alloc, grid, species, atoms);
}

/// Combined PAW table + Gaunt build for init_scf_common.
const PawBundle = struct {
    tabs: ?[]paw_mod.PawTab,
    rhoij: ?paw_mod.RhoIJ,
    gaunt: ?paw_mod.GauntTable,
};

fn maybe_build_paw_bundle(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    is_paw: bool,
) !PawBundle {
    if (!is_paw) return .{ .tabs = null, .rhoij = null, .gaunt = null };
    const paw_init = try build_paw_tables_and_rho_ij(alloc, species, atoms, cfg.scf.ecut_ry);
    errdefer {
        for (paw_init.tabs) |*tab| tab.deinit(alloc);
        alloc.free(paw_init.tabs);
        var rhoij = paw_init.rhoij;
        rhoij.deinit(alloc);
    }
    const gaunt = try build_paw_gaunt_table(alloc, paw_init.tabs);
    return .{ .tabs = paw_init.tabs, .rhoij = paw_init.rhoij, .gaunt = gaunt };
}

const ScfInitDerived = struct {
    grid: Grid,
    coulomb_r_cut: ?f64,
    local_cfg: local_potential.LocalPotentialConfig,
    is_paw: bool,
    ecutrho: ?f64,
};

fn derive_scf_common_data(
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    recip: math.Mat3,
    volume_bohr: f64,
) ScfInitDerived {
    const grid = grid_mod.grid_from_config(cfg.*, recip, volume_bohr);
    const is_paw = has_paw(species);
    return .{
        .grid = grid,
        .coulomb_r_cut = if (cfg.boundary == .isolated)
            coulomb_mod.cutoff_radius(grid.cell)
        else
            null,
        .local_cfg = local_potential.resolve(
            cfg.scf.local_potential,
            cfg.ewald.alpha,
            grid.cell,
        ),
        .is_paw = is_paw,
        .ecutrho = ecutrho_for_paw(cfg, is_paw),
    };
}

pub fn init_scf_common(params: InitParams) !ScfCommon {
    const alloc = params.alloc;
    const io = params.io;
    const cfg = &params.cfg;
    const species = params.model.species;
    const atoms = params.model.atoms;
    const recip = params.model.recip;
    const volume_bohr = params.model.volume_bohr;
    const derived = derive_scf_common_data(cfg, species, recip, volume_bohr);
    var ionic = try potential_mod.build_ionic_potential_grid(
        alloc,
        derived.grid,
        species,
        atoms,
        derived.local_cfg,
        params.ff_tables,
        derived.ecutrho,
    );
    errdefer ionic.deinit(alloc);

    var log = try init_scf_log(alloc, io, cfg.out_dir);
    errdefer log.deinit();

    const kpoints = try generate_kpoints_for_config(
        alloc,
        io,
        cfg,
        recip,
        derived.grid.cell,
        atoms,
    );
    errdefer alloc.free(kpoints);

    const sym_ops = if (cfg.scf.symmetry)
        try symmetry.get_symmetry_ops(alloc, derived.grid.cell, atoms, 1e-6)
    else
        null;
    errdefer if (sym_ops) |ops| alloc.free(ops);

    const rho_core = try maybe_build_rho_core(alloc, derived.grid, species, atoms);
    errdefer if (rho_core) |values| alloc.free(values);

    const radial_tables_buf = try maybe_build_radial_tables(alloc, cfg, species);
    errdefer if (radial_tables_buf) |buf| free_radial_tables(alloc, buf);

    const radial_tables: ?[]const nonlocal_mod.RadialTableSet = radial_tables_buf;

    const pulay_mixer: ?PulayMixer = if (cfg.scf.pulay_history > 0)
        PulayMixer.init(alloc, cfg.scf.pulay_history)
    else
        null;

    var paw = try maybe_build_paw_bundle(alloc, cfg, species, atoms, derived.is_paw);
    errdefer free_paw_bundle(alloc, &paw);

    return finish_scf_common_init(scf_common_init_result(
        params,
        derived,
        total_electrons(species, atoms),
        ionic,
        log,
        kpoints,
        sym_ops,
        rho_core,
        radial_tables_buf,
        radial_tables,
        pulay_mixer,
        paw,
    ));
}

fn init_scf_log(alloc: std.mem.Allocator, io: std.Io, out_dir: []const u8) !ScfLog {
    var log = try ScfLog.init(alloc, io, out_dir);
    errdefer log.deinit();

    try log.write_header();
    return log;
}

const ScfCommonInitResult = struct {
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    derived: ScfInitDerived,
    total_electrons: f64,
    ionic: hamiltonian.PotentialGrid,
    log: ScfLog,
    kpoints: []KPoint,
    sym_ops: ?[]const symmetry.SymOp,
    rho_core: ?[]f64,
    radial_tables_buf: ?[]nonlocal_mod.RadialTableSet,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    pulay_mixer: ?PulayMixer,
    paw: PawBundle,
};

fn scf_common_init_result(
    params: InitParams,
    derived: ScfInitDerived,
    total_electrons_value: f64,
    ionic: hamiltonian.PotentialGrid,
    log: ScfLog,
    kpoints: []KPoint,
    sym_ops: ?[]const symmetry.SymOp,
    rho_core: ?[]f64,
    radial_tables_buf: ?[]nonlocal_mod.RadialTableSet,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    pulay_mixer: ?PulayMixer,
    paw: PawBundle,
) ScfCommonInitResult {
    return .{
        .alloc = params.alloc,
        .cfg = &params.cfg,
        .species = params.model.species,
        .atoms = params.model.atoms,
        .recip = params.model.recip,
        .volume_bohr = params.model.volume_bohr,
        .derived = derived,
        .total_electrons = total_electrons_value,
        .ionic = ionic,
        .log = log,
        .kpoints = kpoints,
        .sym_ops = sym_ops,
        .rho_core = rho_core,
        .radial_tables_buf = radial_tables_buf,
        .radial_tables = radial_tables,
        .pulay_mixer = pulay_mixer,
        .paw = paw,
    };
}

fn finish_scf_common_init(result: ScfCommonInitResult) ScfCommon {
    return build_scf_common_struct(.{
        .alloc = result.alloc,
        .cfg = result.cfg,
        .species = result.species,
        .atoms = result.atoms,
        .recip = result.recip,
        .volume_bohr = result.volume_bohr,
        .grid = result.derived.grid,
        .total_electrons = result.total_electrons,
        .local_cfg = result.derived.local_cfg,
        .ionic = result.ionic,
        .log = result.log,
        .kpoints = result.kpoints,
        .sym_ops = result.sym_ops,
        .rho_core = result.rho_core,
        .radial_tables_buf = result.radial_tables_buf,
        .radial_tables = result.radial_tables,
        .pulay_mixer = result.pulay_mixer,
        .coulomb_r_cut = result.derived.coulomb_r_cut,
        .paw = result.paw,
        .is_paw = result.derived.is_paw,
    });
}

/// Params bag for ScfCommon assembly — all fields correspond to ScfCommon plus the PawBundle.
const ScfCommonSetup = struct {
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    grid: grid_mod.Grid,
    total_electrons: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    ionic: hamiltonian.PotentialGrid,
    log: ScfLog,
    kpoints: []KPoint,
    sym_ops: ?[]const symmetry.SymOp,
    rho_core: ?[]f64,
    radial_tables_buf: ?[]nonlocal_mod.RadialTableSet,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    pulay_mixer: ?PulayMixer,
    coulomb_r_cut: ?f64,
    paw: PawBundle,
    is_paw: bool,
};

fn build_scf_common_struct(s: ScfCommonSetup) ScfCommon {
    return ScfCommon{
        .alloc = s.alloc,
        .cfg = s.cfg.*,
        .species = s.species,
        .atoms = s.atoms,
        .recip = s.recip,
        .volume_bohr = s.volume_bohr,
        .grid = s.grid,
        .total_electrons = s.total_electrons,
        .local_cfg = s.local_cfg,
        .ionic = s.ionic,
        .log = s.log,
        .kpoints = s.kpoints,
        .sym_ops = s.sym_ops,
        .rho_core = s.rho_core,
        .radial_tables_buf = s.radial_tables_buf,
        .radial_tables = s.radial_tables,
        .pulay_mixer = s.pulay_mixer,
        .coulomb_r_cut = s.coulomb_r_cut,
        .paw_tabs = s.paw.tabs,
        .paw_rhoij = s.paw.rhoij,
        .paw_gaunt = s.paw.gaunt,
        .is_paw = s.is_paw,
    };
}
