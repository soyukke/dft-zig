const std = @import("std");
const apply = @import("apply.zig");
const core_density = @import("core_density.zig");
const config = @import("../config/config.zig");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const energy_mod = @import("energy.zig");
const fft = @import("../fft/fft.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoints_mod = @import("kpoint_parallel.zig");
const iterative = @import("../linalg/iterative.zig");
const linalg = @import("../linalg/linalg.zig");
const kmesh_mod = @import("../kpoints/kpoints.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const mixing = @import("mixing.zig");
const model_mod = @import("../dft/model.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const paw_scf = @import("paw_scf.zig");
const potential_mod = @import("potential.zig");
const xc_fields_mod = @import("xc_fields.zig");
const pw_grid_map = @import("pw_grid_map.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const smearing_mod = @import("smearing.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const thread_pool = @import("../thread_pool.zig");
const util = @import("util.zig");
const band_solver = @import("band_solver.zig");
const gvec_iter = @import("gvec_iter.zig");
const scf_spin = @import("scf_spin.zig");

pub const ThreadPool = thread_pool.ThreadPool;

const KPoint = symmetry.KPoint;

pub const Grid = grid_mod.Grid;

pub const ApplyContext = apply.ApplyContext;
pub const apply_hamiltonian = apply.apply_hamiltonian;
pub const apply_hamiltonian_batched = apply.apply_hamiltonian_batched;
const check_hamiltonian_apply = apply.check_hamiltonian_apply;
pub const KpointApplyCache = apply.KpointApplyCache;
pub const NonlocalContext = apply.NonlocalContext;
pub const NonlocalSpecies = apply.NonlocalSpecies;
pub const apply_nonlocal_potential = apply.apply_nonlocal_potential;

pub const KpointCache = kpoints_mod.KpointCache;
const KpointShared = kpoints_mod.KpointShared;
const KpointWorker = kpoints_mod.KpointWorker;
const KpointEigenData = kpoints_mod.KpointEigenData;
const compute_kpoint_contribution = kpoints_mod.compute_kpoint_contribution;
const compute_kpoint_eigen_data = kpoints_mod.compute_kpoint_eigen_data;
pub const kpoint_thread_count = kpoints_mod.kpoint_thread_count;
const kpoint_worker = kpoints_mod.kpoint_worker;
const find_fermi_level_spin = kpoints_mod.find_fermi_level_spin;
const accumulate_kpoint_density_smearing_spin = kpoints_mod.accumulate_kpoint_density_smearing_spin;
const SmearingShared = kpoints_mod.SmearingShared;

const build_fft_index_map = fft_grid.build_fft_index_map;
pub const real_to_reciprocal = fft_grid.real_to_reciprocal;
pub const reciprocal_to_real = fft_grid.reciprocal_to_real;
pub const fft_reciprocal_to_complex_in_place = fft_grid.fft_reciprocal_to_complex_in_place;
const fft_reciprocal_to_complex_in_place_mapped =
    fft_grid.fft_reciprocal_to_complex_in_place_mapped;
pub const fft_complex_to_reciprocal_in_place = fft_grid.fft_complex_to_reciprocal_in_place;
const fft_complex_to_reciprocal_in_place_mapped =
    fft_grid.fft_complex_to_reciprocal_in_place_mapped;
const index_to_freq = fft_grid.index_to_freq;

const mix_density = mixing.mix_density;
const mix_density_kerker = mixing.mix_density_kerker;
const PulayMixer = mixing.PulayMixer;
pub const ComplexPulayMixer = mixing.ComplexPulayMixer;

const ScfLog = logging.ScfLog;
const ScfProfile = logging.ScfProfile;
const log_progress = logging.log_progress;
const log_iter_start = logging.log_iter_start;
const log_kpoint = logging.log_kpoint;
const log_profile = logging.log_profile;
const log_loop_profile = logging.log_loop_profile;
const log_nonlocal_diagnostics = logging.log_nonlocal_diagnostics;
const log_local_diagnostics = logging.log_local_diagnostics;
const log_energy_summary = logging.log_energy_summary;
const log_local_potential_mean = logging.log_local_potential_mean;
const log_iterative_solver_disabled = logging.log_iterative_solver_disabled;
const profile_start = logging.profile_start;
const profile_add = logging.profile_add;
const merge_profile = logging.merge_profile;

pub const PwGridMap = pw_grid_map.PwGridMap;

// gvec_iter re-exports
pub const GVecIterator = gvec_iter.GVecIterator;
const GVecItem = gvec_iter.GVecItem;

// xc_fields re-exports
const XcFields = xc_fields_mod.XcFields;
const XcFieldsSpin = xc_fields_mod.XcFieldsSpin;
const Gradient = xc_fields_mod.Gradient;
pub const compute_xc_fields = xc_fields_mod.compute_xc_fields;
const compute_xc_fields_spin = xc_fields_mod.compute_xc_fields_spin;
pub const gradient_from_real = xc_fields_mod.gradient_from_real;
pub const divergence_from_real = xc_fields_mod.divergence_from_real;

// potential re-exports
const build_potential_grid = potential_mod.build_potential_grid;
const build_potential_grid_spin = potential_mod.build_potential_grid_spin;
pub const build_ionic_potential_grid = potential_mod.build_ionic_potential_grid;
pub const build_local_potential_real = potential_mod.build_local_potential_real;
const filter_density_to_ecutrho = potential_mod.filter_density_to_ecutrho;
const SpinPotentialGrids = potential_mod.SpinPotentialGrids;

// core_density re-exports
pub const has_nlcc = core_density.has_nlcc;
pub const build_core_density = core_density.build_core_density;

/// Wavefunction data for a single k-point.
pub const KpointWavefunction = struct {
    k_frac: math.Vec3,
    k_cart: math.Vec3,
    weight: f64,
    basis_len: usize,
    nbands: usize,
    eigenvalues: []f64,
    coefficients: []math.Complex,
    occupations: []f64,

    pub fn deinit(self: *KpointWavefunction, alloc: std.mem.Allocator) void {
        if (self.eigenvalues.len > 0) alloc.free(self.eigenvalues);
        if (self.coefficients.len > 0) alloc.free(self.coefficients);
        if (self.occupations.len > 0) alloc.free(self.occupations);
    }
};

/// Wavefunction data for all k-points (for force calculation).
pub const WavefunctionData = struct {
    kpoints: []KpointWavefunction,
    ecut_ry: f64,
    fermi_level: f64,

    pub fn deinit(self: *WavefunctionData, alloc: std.mem.Allocator) void {
        for (self.kpoints) |*kp| {
            kp.deinit(alloc);
        }
        if (self.kpoints.len > 0) alloc.free(self.kpoints);
    }
};

pub const ScfResult = struct {
    potential: hamiltonian.PotentialGrid,
    density: []f64,
    iterations: usize,
    converged: bool,
    energy: EnergyTerms,
    fermi_level: f64,
    potential_residual: f64,
    wavefunctions: ?WavefunctionData,
    vresid: ?hamiltonian.PotentialGrid,
    grid: Grid,
    kpoint_cache: ?[]KpointCache = null,
    apply_caches: ?[]apply.KpointApplyCache = null,
    vxc_r: ?[]f64 = null,
    // Spin-polarized fields (nspin=2 only)
    density_up: ?[]f64 = null,
    density_down: ?[]f64 = null,
    potential_down: ?hamiltonian.PotentialGrid = null,
    magnetization: f64 = 0.0,
    wavefunctions_down: ?WavefunctionData = null,
    vxc_r_up: ?[]f64 = null,
    vxc_r_down: ?[]f64 = null,
    fermi_level_down: f64 = 0.0,
    // PAW fields for band calculation
    paw_tabs: ?[]paw_mod.PawTab = null, // Owned PAW tables (one per species)
    paw_dij: ?[][]f64 = null, // Per-atom converged D_ij (radial): [natom][nbeta*nbeta]
    paw_dij_m: ?[][]f64 = null, // Per-atom converged D_ij (m-resolved): [natom][mt*mt]
    paw_dxc: ?[][]f64 = null, // Per-atom D^xc_ij (m-resolved): [natom][mt*mt]
    paw_rhoij: ?[][]f64 = null, // Per-atom rhoij: [natom][nbeta*nbeta]
    ionic_g: ?[]math.Complex = null, // Ionic potential in G-space (for PAW D^hat force)
    rho_core: ?[]f64 = null, // NLCC core density in real space (for stress)

    /// Free allocated SCF results.
    pub fn deinit(self: *ScfResult, alloc: std.mem.Allocator) void {
        self.potential.deinit(alloc);
        if (self.density.len > 0) alloc.free(self.density);
        if (self.wavefunctions) |*wf| wf.deinit(alloc);
        if (self.vresid) |*vresid| vresid.deinit(alloc);
        if (self.kpoint_cache) |cache| {
            for (cache) |*c| c.deinit();
            alloc.free(cache);
        }
        if (self.apply_caches) |caches| {
            for (caches) |*ac| ac.deinit(alloc);
            alloc.free(caches);
        }
        if (self.vxc_r) |v| alloc.free(v);
        if (self.density_up) |d| alloc.free(d);
        if (self.density_down) |d| alloc.free(d);
        if (self.potential_down) |*p| p.deinit(alloc);
        if (self.wavefunctions_down) |*wf| wf.deinit(alloc);
        if (self.vxc_r_up) |v| alloc.free(v);
        if (self.vxc_r_down) |v| alloc.free(v);
        if (self.paw_dij) |dij| {
            for (dij) |d| alloc.free(d);
            alloc.free(dij);
        }
        if (self.paw_dij_m) |dij| {
            for (dij) |d| alloc.free(d);
            alloc.free(dij);
        }
        if (self.paw_dxc) |dxc| {
            for (dxc) |d| alloc.free(d);
            alloc.free(dxc);
        }
        if (self.paw_rhoij) |rij| {
            for (rij) |r| alloc.free(r);
            alloc.free(rij);
        }
        if (self.paw_tabs) |tabs| {
            for (@constCast(tabs)) |*t| t.deinit(alloc);
            alloc.free(tabs);
        }
        if (self.ionic_g) |ig| alloc.free(ig);
        if (self.rho_core) |rc| alloc.free(rc);
    }
};

pub const EnergyTerms = energy_mod.EnergyTerms;

pub const DensityResult = smearing_mod.DensityResult;

// Band solver re-exports
pub const BandIterativeContext = band_solver.BandIterativeContext;
pub const BandVectorCache = band_solver.BandVectorCache;
pub const BandEigenOptions = band_solver.BandEigenOptions;
pub const init_band_iterative_context = band_solver.init_band_iterative_context;
pub const band_eigenvalues_iterative = band_solver.band_eigenvalues_iterative;
pub const band_eigenvalues_iterative_ext = band_solver.band_eigenvalues_iterative_ext;

fn total_charge(rho: []f64, grid: Grid) f64 {
    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var sum: f64 = 0.0;
    for (rho) |value| {
        sum += value * dv;
    }
    return sum;
}

const smearing_active = smearing_mod.smearing_active;

fn wrap_grid_index(g: i32, min: i32, n: usize) usize {
    const ni = @as(i32, @intCast(n));
    const idx = @mod(g - min, ni);
    return @as(usize, @intCast(idx));
}

pub fn symmetrize_density(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    ops: []const symmetry.SymOp,
    use_rfft: bool,
) !void {
    if (ops.len <= 1) return;

    const rho_g = try real_to_reciprocal(alloc, grid, rho, use_rfft);
    defer alloc.free(rho_g);

    const total = grid.count();
    const rho_sym = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho_sym);

    @memset(rho_sym, math.complex.init(0.0, 0.0));

    const inv_ops = 1.0 / @as(f64, @floatFromInt(ops.len));
    const two_pi = 2.0 * std.math.pi;
    const phase_tol = 1e-12;

    var z: usize = 0;
    while (z < grid.nz) : (z += 1) {
        const gl = grid.min_l + @as(i32, @intCast(z));
        var y: usize = 0;
        while (y < grid.ny) : (y += 1) {
            const gk = grid.min_k + @as(i32, @intCast(y));
            var x: usize = 0;
            while (x < grid.nx) : (x += 1) {
                const gh = grid.min_h + @as(i32, @intCast(x));
                var sum = math.complex.init(0.0, 0.0);

                for (ops) |op| {
                    const m = op.k_rot.m;
                    const mh = m[0][0] * gh + m[0][1] * gk + m[0][2] * gl;
                    const mk = m[1][0] * gh + m[1][1] * gk + m[1][2] * gl;
                    const ml = m[2][0] * gh + m[2][1] * gk + m[2][2] * gl;

                    const ix = wrap_grid_index(mh, grid.min_h, grid.nx);
                    const iy = wrap_grid_index(mk, grid.min_k, grid.ny);
                    const iz = wrap_grid_index(ml, grid.min_l, grid.nz);
                    const idx = ix + grid.nx * (iy + grid.ny * iz);
                    var term = rho_g[idx];

                    const dot = @as(f64, @floatFromInt(gh)) * op.trans.x +
                        @as(f64, @floatFromInt(gk)) * op.trans.y +
                        @as(f64, @floatFromInt(gl)) * op.trans.z;
                    const frac = dot - std.math.floor(dot);
                    if (frac > phase_tol and frac < 1.0 - phase_tol) {
                        const phase = math.complex.expi(-two_pi * frac);
                        term = math.complex.mul(term, phase);
                    }

                    sum = math.complex.add(sum, term);
                }

                const out_idx = x + grid.nx * (y + grid.ny * z);
                rho_sym[out_idx] = math.complex.scale(sum, inv_ops);
            }
        }
    }

    const rho_real = try reciprocal_to_real(alloc, grid, rho_sym);
    defer alloc.free(rho_real);

    std.mem.copyForwards(f64, rho, rho_real);
}

/// Parameters for SCF calculation.
pub const ScfParams = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    model: *const model_mod.Model,
    initial_density: ?[]const f64 = null,
    initial_kpoint_cache: ?[]KpointCache = null,
    initial_apply_caches: ?[]apply.KpointApplyCache = null,
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

    fn deinit(self: *ScfCommon) void {
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
    errdefer {
        for (tabs) |*t| t.deinit(alloc);
        alloc.free(tabs);
    }
    for (species, 0..) |entry, si| {
        if (entry.upf.paw) |paw| {
            tabs[si] = try paw_mod.PawTab.init(alloc, paw, entry.upf.r, entry.upf.rab, q_max);
        } else {
            tabs[si] = .{
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
fn ecutrho_for_paw(cfg: *const config.Config, is_paw: bool) ?f64 {
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

fn init_scf_common(params: ScfParams) !ScfCommon {
    const alloc = params.alloc;
    const io = params.io;
    const cfg = &params.cfg;
    const species = params.model.species;
    const atoms = params.model.atoms;
    const recip = params.model.recip;
    const volume_bohr = params.model.volume_bohr;
    const ff_tables = params.ff_tables;
    const derived = derive_scf_common_data(cfg, species, recip, volume_bohr);
    const electron_count = total_electrons(species, atoms);

    var ionic = try potential_mod.build_ionic_potential_grid(
        alloc,
        derived.grid,
        species,
        atoms,
        derived.local_cfg,
        ff_tables,
        derived.ecutrho,
    );
    errdefer ionic.deinit(alloc);

    var log = try ScfLog.init(alloc, io, cfg.out_dir);
    errdefer log.deinit();

    try log.write_header();

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

    return build_scf_common_struct(.{
        .alloc = alloc,
        .cfg = cfg,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume_bohr = volume_bohr,
        .grid = derived.grid,
        .total_electrons = electron_count,
        .local_cfg = derived.local_cfg,
        .ionic = ionic,
        .log = log,
        .kpoints = kpoints,
        .sym_ops = sym_ops,
        .rho_core = rho_core,
        .radial_tables_buf = radial_tables_buf,
        .radial_tables = radial_tables,
        .pulay_mixer = pulay_mixer,
        .coulomb_r_cut = derived.coulomb_r_cut,
        .paw = paw,
        .is_paw = derived.is_paw,
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
    sym_ops: ?[]symmetry.SymOp,
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

/// Run SCF loop to build Hartree+XC potential.
/// Initialise rho with user-provided density (if present and fits) or flat total_electrons/volume.
fn init_rho_from_user_or_flat(
    rho: []f64,
    initial_density: ?[]const f64,
    electron_count: f64,
    grid_volume: f64,
) void {
    if (initial_density) |init_rho| {
        if (init_rho.len == rho.len) {
            @memcpy(rho, init_rho);
            return;
        }
    }
    const rho0 = electron_count / grid_volume;
    @memset(rho, rho0);
}

/// Warm-start the per-kpoint cache from a previously converged SCF run, if provided.
fn warm_start_kpoint_cache(
    kpoint_cache: []KpointCache,
    initial_kpoint_cache: ?[]const KpointCache,
) !void {
    if (initial_kpoint_cache) |init_cache| {
        const copy_len = @min(kpoint_cache.len, init_cache.len);
        for (0..copy_len) |k| {
            if (init_cache[k].vectors.len > 0) {
                try kpoint_cache[k].store(
                    init_cache[k].n,
                    init_cache[k].nbands,
                    init_cache[k].vectors,
                );
            }
        }
    }
}

/// Return paw ecutrho used throughout the SCF loop, or null for non-PAW systems.
fn resolve_paw_ecutrho(cfg: *const config.Config, is_paw: bool) ?f64 {
    if (!is_paw) return null;
    const gs_val = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_val * gs_val;
}

/// Copy NLCC core density to a newly-allocated buffer for result, or null.
fn copy_rho_core_for_result(alloc: std.mem.Allocator, rho_core: ?[]const f64) !?[]f64 {
    if (rho_core) |rc| {
        const copy = try alloc.alloc(f64, rc.len);
        @memcpy(copy, rc);
        return copy;
    }
    return null;
}

/// Bag of SCF loop profiling accumulators.
const ScfLoopProf = struct {
    compute_density_ns: u64 = 0,
    build_potential_ns: u64 = 0,
    residual_ns: u64 = 0,
    mixing_ns: u64 = 0,
    build_local_r_ns: u64 = 0,
    build_fft_map_ns: u64 = 0,
};

/// Either return the caller-supplied apply caches or allocate + default-init a fresh set.
fn get_or_alloc_apply_caches(
    alloc: std.mem.Allocator,
    initial: ?[]apply.KpointApplyCache,
    n: usize,
) ![]apply.KpointApplyCache {
    if (initial) |caches| return caches;
    const caches = try alloc.alloc(apply.KpointApplyCache, n);
    for (caches) |*ac| ac.* = .{};
    return caches;
}

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
fn extract_paw_results(
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
fn paw_ecutrho_compute(cfg: *const config.Config) f64 {
    const gs_comp = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_comp * gs_comp;
}

/// Symmetrize density over crystal operations, and PAW rhoij when applicable.
fn symmetrize_density_and_rho_ij(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: grid_mod.Grid,
    common: *ScfCommon,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho: []f64,
) !void {
    if (common.sym_ops) |ops| {
        if (ops.len > 1) {
            try symmetrize_density(alloc, grid, rho, ops, cfg.scf.use_rfft);
        }
    }
    if (common.paw_rhoij) |*prij| {
        if (cfg.scf.symmetry) {
            try paw_scf.symmetrize_rho_ij(alloc, prij, species, atoms);
        }
    }
}

/// Filter density to |G|² < ecutrho sphere, writing the result back into rho.
fn filter_augmented_density_in_place(
    alloc: std.mem.Allocator,
    grid: grid_mod.Grid,
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
fn build_augmented_density(
    alloc: std.mem.Allocator,
    grid: grid_mod.Grid,
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

/// Compute and store potential residual; updates vresid_last to a freshly-allocated grid.
fn record_potential_residual(
    alloc: std.mem.Allocator,
    grid: grid_mod.Grid,
    potential: hamiltonian.PotentialGrid,
    potential_out: hamiltonian.PotentialGrid,
    vresid_last: *?hamiltonian.PotentialGrid,
) !f64 {
    const nvals = potential.values.len;
    var residual_values = try alloc.alloc(math.Complex, nvals);
    errdefer alloc.free(residual_values);

    var sum_sq: f64 = 0.0;
    for (0..nvals) |idx| {
        const diff = math.complex.sub(potential_out.values[idx], potential.values[idx]);
        residual_values[idx] = diff;
        sum_sq += diff.r * diff.r + diff.i * diff.i;
    }
    const residual_rms = if (nvals > 0)
        std.math.sqrt(sum_sq / @as(f64, @floatFromInt(nvals)))
    else
        0.0;
    if (vresid_last.*) |*old| old.deinit(alloc);
    vresid_last.* = hamiltonian.PotentialGrid{
        .nx = grid.nx,
        .ny = grid.ny,
        .nz = grid.nz,
        .min_h = grid.min_h,
        .min_k = grid.min_k,
        .min_l = grid.min_l,
        .values = residual_values,
    };
    return residual_rms;
}

/// Build rho_aug for final energy evaluation (PAW only); returns null for non-PAW.
fn maybe_build_rho_aug_for_energy(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: grid_mod.Grid,
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

/// Params bag for building the final ScfResult.
const ScfResultSetup = struct {
    potential: hamiltonian.PotentialGrid,
    density: []f64,
    iterations: usize,
    converged: bool,
    energy: energy_mod.EnergyTerms,
    fermi_level: f64,
    potential_residual: f64,
    wavefunctions: ?WavefunctionData,
    vresid: ?hamiltonian.PotentialGrid,
    grid: grid_mod.Grid,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    vxc_r: ?[]f64,
    paw_tabs: ?[]paw_mod.PawTab,
    paw_dij: ?[][]f64,
    paw_dij_m: ?[][]f64,
    paw_dxc: ?[][]f64,
    paw_rhoij: ?[][]f64,
    ionic_g: ?[]math.Complex,
    rho_core_copy: ?[]f64,
};

fn build_scf_result(s: ScfResultSetup) ScfResult {
    return ScfResult{
        .potential = s.potential,
        .density = s.density,
        .iterations = s.iterations,
        .converged = s.converged,
        .energy = s.energy,
        .fermi_level = s.fermi_level,
        .potential_residual = s.potential_residual,
        .wavefunctions = s.wavefunctions,
        .vresid = s.vresid,
        .grid = s.grid,
        .kpoint_cache = s.kpoint_cache,
        .apply_caches = s.apply_caches,
        .vxc_r = s.vxc_r,
        .paw_tabs = s.paw_tabs,
        .paw_dij = s.paw_dij,
        .paw_dij_m = s.paw_dij_m,
        .paw_dxc = s.paw_dxc,
        .paw_rhoij = s.paw_rhoij,
        .ionic_g = s.ionic_g,
        .rho_core = s.rho_core_copy,
    };
}

/// Potential-mixing branch: mixes V_in and V_out, frees potential_out.
fn mix_potential_mode(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: grid_mod.Grid,
    common: *ScfCommon,
    iterations: usize,
    rho: []f64,
    new_rho: []const f64,
    potential: *hamiltonian.PotentialGrid,
    potential_out: *hamiltonian.PotentialGrid,
    keep_potential_out: *bool,
) !void {
    const n_complex = potential.values.len;
    const n_f64 = n_complex * 2;
    const v_in: []f64 = @as([*]f64, @ptrCast(potential.values.ptr))[0..n_f64];

    if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
        const residual_g = try alloc.alloc(math.Complex, n_complex);
        for (0..n_complex) |idx| {
            residual_g[idx] = math.complex.sub(potential_out.values[idx], potential.values[idx]);
        }
        if (cfg.scf.diemac > 1.0) {
            mixing.apply_model_dielectric_preconditioner(
                grid,
                residual_g,
                cfg.scf.diemac,
                cfg.scf.dielng,
            );
        }
        const precond_f64: []f64 = @as([*]f64, @ptrCast(residual_g.ptr))[0..n_f64];
        try common.pulay_mixer.?.mix_with_residual(v_in, precond_f64, cfg.scf.mixing_beta);
    } else {
        const v_out_ptr = @as([*]const f64, @ptrCast(potential_out.values.ptr));
        const v_out: []const f64 = v_out_ptr[0..n_f64];
        mix_density(v_in, v_out, cfg.scf.mixing_beta);
    }

    @memcpy(rho, new_rho);
    potential_out.deinit(alloc);
    keep_potential_out.* = true;
}

/// Density-mixing branch: mixes rho and rebuilds potential.
fn mix_density_mode(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: grid_mod.Grid,
    common: *ScfCommon,
    iterations: usize,
    rho: []f64,
    new_rho: []const f64,
    potential: *hamiltonian.PotentialGrid,
    paw_ecutrho: ?f64,
) !void {
    if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
        if (cfg.scf.kerker_q0 > 0.0) {
            try common.pulay_mixer.?.mix_kerker_pulay(
                rho,
                new_rho,
                cfg.scf.mixing_beta,
                grid,
                cfg.scf.kerker_q0,
                cfg.scf.use_rfft,
            );
        } else {
            try common.pulay_mixer.?.mix(rho, new_rho, cfg.scf.mixing_beta);
        }
    } else if (cfg.scf.kerker_q0 > 0.0) {
        try mix_density_kerker(
            alloc,
            grid,
            rho,
            new_rho,
            cfg.scf.mixing_beta,
            cfg.scf.kerker_q0,
            cfg.scf.use_rfft,
        );
    } else {
        mix_density(rho, new_rho, cfg.scf.mixing_beta);
    }

    potential.deinit(alloc);
    potential.* = try potential_mod.build_potential_grid(
        alloc,
        grid,
        rho,
        common.rho_core,
        cfg.scf.use_rfft,
        cfg.scf.xc,
        null,
        common.coulomb_r_cut,
        paw_ecutrho,
    );
}

/// Update PAW D_ij matrix from the current mixed potential (no-op for non-PAW).
fn update_paw_dij_if_needed(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: grid_mod.Grid,
    common: *ScfCommon,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    potential: hamiltonian.PotentialGrid,
    apply_caches: []apply.KpointApplyCache,
) !void {
    if (!common.is_paw) return;
    const tabs = common.paw_tabs orelse return;
    const gs_paw = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    const ecutrho_paw = cfg.scf.ecut_ry * gs_paw * gs_paw;
    try paw_scf.update_paw_dij(
        alloc,
        grid,
        common.ionic,
        potential,
        tabs,
        species,
        atoms,
        apply_caches,
        ecutrho_paw,
        &common.paw_rhoij.?,
        cfg.scf.xc,
        cfg.scf.symmetry,
        &common.paw_gaunt.?,
        false,
        null,
        1.0,
    );
}

const ScfRunCaches = struct {
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,

    fn deinit(self: *const ScfRunCaches, alloc: std.mem.Allocator) void {
        for (self.kpoint_cache) |*cache| cache.deinit();
        alloc.free(self.kpoint_cache);
        for (self.apply_caches) |*ac| ac.deinit(alloc);
        alloc.free(self.apply_caches);
    }
};

const ScfRunState = struct {
    rho: []f64,
    potential: hamiltonian.PotentialGrid,
    vxc_r: ?[]f64 = null,
    vresid_last: ?hamiltonian.PotentialGrid = null,
    last_band_energy: f64 = 0.0,
    last_nonlocal_energy: f64 = 0.0,
    last_entropy_energy: f64 = 0.0,
    last_fermi_level: f64 = std.math.nan(f64),
    last_potential_residual: f64 = 0.0,
    iterations: usize = 0,
    converged: bool = false,
    wavefunctions: ?WavefunctionData = null,

    fn deinit(self: *ScfRunState, alloc: std.mem.Allocator) void {
        self.potential.deinit(alloc);
        alloc.free(self.rho);
        if (self.vxc_r) |v| alloc.free(v);
        if (self.vresid_last) |*vresid| vresid.deinit(alloc);
        if (self.wavefunctions) |*wf| wf.deinit(alloc);
    }
};

const ScfRunContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    common: *ScfCommon,
    paw_ecutrho: ?f64,
};

const ScfIterationDensity = struct {
    density_result: DensityResult,
    rho_for_potential: []const f64,

    fn deinit(self: *const ScfIterationDensity, alloc: std.mem.Allocator, is_paw: bool) void {
        alloc.free(self.density_result.rho);
        if (is_paw) alloc.free(self.rho_for_potential);
    }
};

const ScfIterationPotential = struct {
    potential_out: hamiltonian.PotentialGrid,
    keep: bool = false,

    fn deinit(self: *const ScfIterationPotential, alloc: std.mem.Allocator) void {
        if (!self.keep) {
            var potential_out = self.potential_out;
            potential_out.deinit(alloc);
        }
    }
};

const FinalPawResults = struct {
    paw_tabs: ?[]paw_mod.PawTab = null,
    paw_dij: ?[][]f64 = null,
    paw_dij_m: ?[][]f64 = null,
    paw_dxc: ?[][]f64 = null,
    paw_rhoij: ?[][]f64 = null,
};

fn init_scf_run_caches(
    alloc: std.mem.Allocator,
    kpoint_count: usize,
    initial_kpoint_cache: ?[]const KpointCache,
    initial_apply_caches: ?[]apply.KpointApplyCache,
) !ScfRunCaches {
    const kpoint_cache = try alloc.alloc(KpointCache, kpoint_count);
    errdefer alloc.free(kpoint_cache);
    for (kpoint_cache) |*cache| cache.* = .{};
    try warm_start_kpoint_cache(kpoint_cache, initial_kpoint_cache);
    return .{
        .kpoint_cache = kpoint_cache,
        .apply_caches = try get_or_alloc_apply_caches(alloc, initial_apply_caches, kpoint_count),
    };
}

fn init_scf_run_state(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    common: *const ScfCommon,
    initial_density: ?[]const f64,
    paw_ecutrho: ?f64,
) !ScfRunState {
    const rho = try alloc.alloc(f64, common.grid.count());
    errdefer alloc.free(rho);
    init_rho_from_user_or_flat(rho, initial_density, common.total_electrons, common.grid.volume);

    var potential = try potential_mod.build_potential_grid(
        alloc,
        common.grid,
        rho,
        common.rho_core,
        cfg.scf.use_rfft,
        cfg.scf.xc,
        null,
        common.coulomb_r_cut,
        paw_ecutrho,
    );
    errdefer potential.deinit(alloc);
    return .{ .rho = rho, .potential = potential };
}

fn prepare_scf_iteration_density(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
    prof: *ScfLoopProf,
) !ScfIterationDensity {
    const t_density_start = if (ctx.cfg.scf.profile) profile_start(ctx.io) else null;
    const density_result = try compute_density(
        ctx.alloc,
        ctx.io,
        ctx.cfg.*,
        ctx.common.grid,
        ctx.common.kpoints,
        ctx.common.ionic,
        ctx.species,
        ctx.atoms,
        ctx.recip,
        ctx.volume_bohr,
        state.potential,
        state.iterations,
        caches.kpoint_cache,
        caches.apply_caches,
        ctx.common.radial_tables,
        if (ctx.cfg.scf.profile) ScfLoopProfile{
            .build_local_r_ns = &prof.build_local_r_ns,
            .build_fft_map_ns = &prof.build_fft_map_ns,
        } else null,
        ctx.common.paw_tabs,
        if (ctx.common.paw_rhoij) |*rij| rij else null,
    );
    if (ctx.cfg.scf.profile) profile_add(ctx.io, &prof.compute_density_ns, t_density_start);

    state.last_band_energy = density_result.band_energy;
    state.last_nonlocal_energy = density_result.nonlocal_energy;
    state.last_entropy_energy = density_result.entropy_energy;
    state.last_fermi_level = density_result.fermi_level;

    try symmetrize_density_and_rho_ij(
        ctx.alloc,
        ctx.cfg,
        ctx.common.grid,
        ctx.common,
        ctx.species,
        ctx.atoms,
        density_result.rho,
    );

    return .{
        .density_result = density_result,
        .rho_for_potential = try prepare_density_for_potential(ctx, density_result.rho),
    };
}

fn build_scf_iteration_potential(
    ctx: *const ScfRunContext,
    state: *ScfRunState,
    rho_for_potential: []const f64,
    prof: *ScfLoopProf,
) !ScfIterationPotential {
    if (state.vxc_r) |old| ctx.alloc.free(old);
    state.vxc_r = null;
    const vxc_r_ptr: ?*?[]f64 = if (ctx.cfg.relax.enabled) &state.vxc_r else null;
    const t_build_pot_start = if (ctx.cfg.scf.profile) profile_start(ctx.io) else null;
    const potential_out = try potential_mod.build_potential_grid(
        ctx.alloc,
        ctx.common.grid,
        rho_for_potential,
        ctx.common.rho_core,
        ctx.cfg.scf.use_rfft,
        ctx.cfg.scf.xc,
        vxc_r_ptr,
        ctx.common.coulomb_r_cut,
        ctx.paw_ecutrho,
    );
    if (ctx.cfg.scf.profile) profile_add(ctx.io, &prof.build_potential_ns, t_build_pot_start);

    const t_resid_start = if (ctx.cfg.scf.profile) profile_start(ctx.io) else null;
    state.last_potential_residual = try record_potential_residual(
        ctx.alloc,
        ctx.common.grid,
        state.potential,
        potential_out,
        &state.vresid_last,
    );
    if (ctx.cfg.scf.profile) profile_add(ctx.io, &prof.residual_ns, t_resid_start);
    return .{ .potential_out = potential_out };
}

fn prepare_density_for_potential(
    ctx: *const ScfRunContext,
    rho: []f64,
) ![]const f64 {
    const ecutrho_comp = paw_ecutrho_compute(ctx.cfg);
    if (ctx.common.is_paw) {
        try filter_augmented_density_in_place(
            ctx.alloc,
            ctx.common.grid,
            rho,
            ecutrho_comp,
            ctx.cfg.scf.use_rfft,
        );
        return build_augmented_density(
            ctx.alloc,
            ctx.common.grid,
            rho,
            ctx.common,
            ctx.atoms,
            ecutrho_comp,
            ctx.common.grid.count(),
        );
    }
    return rho;
}

fn finish_scf_iteration(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
    density: *const ScfIterationDensity,
    potential_step: *ScfIterationPotential,
    prof: *ScfLoopProf,
) !bool {
    const diff = density_diff(state.rho, density.density_result.rho);
    const conv_value = switch (ctx.cfg.scf.convergence_metric) {
        .density => diff,
        .potential => state.last_potential_residual,
    };
    try log_scf_iteration_progress(ctx, state, diff);
    if (conv_value < ctx.cfg.scf.convergence) {
        return finish_converged_scf_iteration(ctx, state, density, potential_step);
    }

    const t_mix_start = if (ctx.cfg.scf.profile) profile_start(ctx.io) else null;
    if (ctx.cfg.scf.mixing_mode == .potential) {
        try mix_potential_mode(
            ctx.alloc,
            ctx.cfg,
            ctx.common.grid,
            ctx.common,
            state.iterations,
            state.rho,
            density.density_result.rho,
            &state.potential,
            &potential_step.potential_out,
            &potential_step.keep,
        );
    } else {
        try mix_density_mode(
            ctx.alloc,
            ctx.cfg,
            ctx.common.grid,
            ctx.common,
            state.iterations,
            state.rho,
            density.density_result.rho,
            &state.potential,
            ctx.paw_ecutrho,
        );
    }
    if (ctx.cfg.scf.profile) profile_add(ctx.io, &prof.mixing_ns, t_mix_start);
    try update_paw_dij_if_needed(
        ctx.alloc,
        ctx.cfg,
        ctx.common.grid,
        ctx.common,
        ctx.species,
        ctx.atoms,
        state.potential,
        caches.apply_caches,
    );
    return false;
}

fn log_scf_iteration_progress(
    ctx: *const ScfRunContext,
    state: *const ScfRunState,
    diff: f64,
) !void {
    try ctx.common.log.write_iter(
        state.iterations,
        diff,
        state.last_potential_residual,
        state.last_band_energy,
        state.last_nonlocal_energy,
    );
    if (!ctx.cfg.scf.quiet) {
        try log_progress(
            ctx.io,
            state.iterations,
            diff,
            state.last_potential_residual,
            state.last_band_energy,
            state.last_nonlocal_energy,
        );
    }
}

fn finish_converged_scf_iteration(
    ctx: *const ScfRunContext,
    state: *ScfRunState,
    density: *const ScfIterationDensity,
    potential_step: *ScfIterationPotential,
) bool {
    state.converged = true;
    @memcpy(state.rho, density.density_result.rho);
    state.potential.deinit(ctx.alloc);
    state.potential = potential_step.potential_out;
    potential_step.keep = true;
    return true;
}

fn run_scf_iterations(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
) !void {
    var prof = ScfLoopProf{};
    while (state.iterations < ctx.cfg.scf.max_iter) : (state.iterations += 1) {
        if (!ctx.cfg.scf.quiet) try log_iter_start(ctx.io, state.iterations);
        const density = try prepare_scf_iteration_density(ctx, caches, state, &prof);
        defer density.deinit(ctx.alloc, ctx.common.is_paw);

        var potential_step = try build_scf_iteration_potential(
            ctx,
            state,
            density.rho_for_potential,
            &prof,
        );
        defer potential_step.deinit(ctx.alloc);

        if (try finish_scf_iteration(ctx, caches, state, &density, &potential_step, &prof)) break;
    }
    if (ctx.cfg.scf.profile and !ctx.cfg.scf.quiet) {
        try log_loop_profile(
            ctx.io,
            prof.compute_density_ns,
            prof.build_potential_ns,
            prof.residual_ns,
            prof.mixing_ns,
            prof.build_local_r_ns,
            prof.build_fft_map_ns,
        );
    }
}

fn add_paw_onsite_energy_if_needed(
    ctx: *const ScfRunContext,
    energy_terms: *energy_mod.EnergyTerms,
) !void {
    if (!ctx.common.is_paw) return;
    const prij = ctx.common.paw_rhoij orelse return;
    const tabs = ctx.common.paw_tabs orelse return;
    energy_terms.paw_onsite = try paw_scf.compute_paw_onsite_energy_total(
        ctx.alloc,
        &prij,
        tabs,
        ctx.species,
        ctx.atoms,
        ctx.cfg.scf.xc,
        &ctx.common.paw_gaunt.?,
        null,
        null,
    );
    energy_terms.total += energy_terms.paw_onsite;
}

fn compute_final_scf_energy_terms(
    ctx: *const ScfRunContext,
    state: *ScfRunState,
) !energy_mod.EnergyTerms {
    const rho_aug_for_energy = try maybe_build_rho_aug_for_energy(
        ctx.alloc,
        ctx.cfg,
        ctx.common.grid,
        ctx.common,
        ctx.atoms,
        state.rho,
    );
    defer if (rho_aug_for_energy) |a| ctx.alloc.free(a);

    var energy_terms = try energy_mod.compute_energy_terms(.{
        .alloc = ctx.alloc,
        .io = ctx.io,
        .grid = ctx.common.grid,
        .species = ctx.species,
        .atoms = ctx.atoms,
        .rho = state.rho,
        .rho_core = ctx.common.rho_core,
        .rho_aug = rho_aug_for_energy,
        .band_energy = state.last_band_energy,
        .nonlocal_energy = state.last_nonlocal_energy,
        .entropy_energy = state.last_entropy_energy,
        .local_cfg = ctx.common.local_cfg,
        .ewald_cfg = ctx.cfg.ewald,
        .vdw_cfg = ctx.cfg.vdw,
        .xc_func = ctx.cfg.scf.xc,
        .use_rfft = ctx.cfg.scf.use_rfft,
        .quiet = ctx.cfg.scf.quiet,
        .coulomb_r_cut = ctx.common.coulomb_r_cut,
        .ecutrho = ctx.paw_ecutrho,
    });
    try add_paw_onsite_energy_if_needed(ctx, &energy_terms);
    return energy_terms;
}

fn maybe_compute_final_scf_wavefunctions(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
) !void {
    if (!(ctx.cfg.relax.enabled or ctx.cfg.dfpt.enabled or
        ctx.cfg.scf.compute_stress or ctx.cfg.dos.enabled))
    {
        return;
    }
    const wfn_result = try compute_final_wavefunctions_with_spin_factor(
        ctx.alloc,
        ctx.io,
        ctx.cfg.*,
        ctx.common.grid,
        ctx.common.kpoints,
        ctx.common.ionic,
        ctx.species,
        ctx.atoms,
        ctx.recip,
        ctx.volume_bohr,
        state.potential,
        caches.kpoint_cache,
        caches.apply_caches,
        ctx.common.radial_tables,
        ctx.common.paw_tabs,
        2.0,
    );
    state.wavefunctions = wfn_result.wavefunctions;
    state.last_band_energy = wfn_result.band_energy;
    state.last_nonlocal_energy = wfn_result.nonlocal_energy;
}

fn extract_final_paw_results(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    energy_terms: *energy_mod.EnergyTerms,
) !FinalPawResults {
    var result: FinalPawResults = .{};
    if (!ctx.common.is_paw) return result;
    try extract_paw_results(
        ctx.alloc,
        ctx.cfg,
        ctx.species,
        ctx.atoms,
        caches.apply_caches,
        ctx.common,
        energy_terms,
        &result.paw_tabs,
        &result.paw_dij,
        &result.paw_dij_m,
        &result.paw_dxc,
        &result.paw_rhoij,
    );
    return result;
}

fn maybe_copy_ionic_g(
    alloc: std.mem.Allocator,
    common: *const ScfCommon,
) !?[]math.Complex {
    if (!common.is_paw) return null;
    const result_ionic_g = try alloc.alloc(math.Complex, common.ionic.values.len);
    @memcpy(result_ionic_g, common.ionic.values);
    return result_ionic_g;
}

fn log_final_scf_summary(
    ctx: *const ScfRunContext,
    state: *const ScfRunState,
    energy_terms: energy_mod.EnergyTerms,
) !void {
    if (!ctx.cfg.scf.quiet) {
        try log_energy_summary(
            ctx.io,
            total_charge(state.rho, ctx.common.grid),
            ctx.common.ionic.value_at(0, 0, 0),
            state.potential.value_at(0, 0, 0),
            energy_terms,
        );
    }
    try ctx.common.log.write_result(
        state.converged,
        state.iterations,
        energy_terms.total,
        energy_terms.band,
        energy_terms.hartree,
        energy_terms.xc,
        energy_terms.ion_ion,
        energy_terms.psp_core,
    );
}

fn finalize_scf_run(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
) !ScfResult {
    var energy_terms = try compute_final_scf_energy_terms(ctx, state);
    try maybe_compute_final_scf_wavefunctions(ctx, caches, state);
    const paw_results = try extract_final_paw_results(ctx, caches, &energy_terms);
    try log_final_scf_summary(ctx, state, energy_terms);
    return build_scf_result(.{
        .potential = state.potential,
        .density = state.rho,
        .iterations = state.iterations,
        .converged = state.converged,
        .energy = energy_terms,
        .fermi_level = state.last_fermi_level,
        .potential_residual = state.last_potential_residual,
        .wavefunctions = state.wavefunctions,
        .vresid = state.vresid_last,
        .grid = ctx.common.grid,
        .kpoint_cache = caches.kpoint_cache,
        .apply_caches = caches.apply_caches,
        .vxc_r = state.vxc_r,
        .paw_tabs = paw_results.paw_tabs,
        .paw_dij = paw_results.paw_dij,
        .paw_dij_m = paw_results.paw_dij_m,
        .paw_dxc = paw_results.paw_dxc,
        .paw_rhoij = paw_results.paw_rhoij,
        .ionic_g = try maybe_copy_ionic_g(ctx.alloc, ctx.common),
        .rho_core_copy = try copy_rho_core_for_result(ctx.alloc, ctx.common.rho_core),
    });
}

pub fn run(params: ScfParams) !ScfResult {
    const alloc = params.alloc;
    const io = params.io;
    const cfg = params.cfg;
    const species = params.model.species;
    const atoms = params.model.atoms;
    const recip = params.model.recip;
    const volume_bohr = params.model.volume_bohr;
    const initial_density = params.initial_density;
    const initial_kpoint_cache = params.initial_kpoint_cache;
    const initial_apply_caches = params.initial_apply_caches;
    if (!cfg.scf.enabled) return error.ScfDisabled;

    var common = try init_scf_common(params);
    defer common.deinit();

    // Dispatch to spin-polarized SCF loop if nspin=2
    if (cfg.scf.nspin == 2) {
        return scf_spin.run_spin_polarized_loop(
            alloc,
            io,
            cfg,
            species,
            atoms,
            volume_bohr,
            &common,
        );
    }

    const paw_ecutrho = resolve_paw_ecutrho(&cfg, common.is_paw);
    const ctx = ScfRunContext{
        .alloc = alloc,
        .io = io,
        .cfg = &cfg,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume_bohr = volume_bohr,
        .common = &common,
        .paw_ecutrho = paw_ecutrho,
    };
    const caches = try init_scf_run_caches(
        alloc,
        common.kpoints.len,
        initial_kpoint_cache,
        initial_apply_caches,
    );
    errdefer caches.deinit(alloc);

    var state = try init_scf_run_state(alloc, &cfg, &common, initial_density, paw_ecutrho);
    errdefer state.deinit(alloc);

    try run_scf_iterations(&ctx, &caches, &state);
    return try finalize_scf_run(&ctx, &caches, &state);
}

pub const build_atomic_density = core_density.build_atomic_density;

const WavefunctionResult = struct {
    wavefunctions: WavefunctionData,
    band_energy: f64,
    nonlocal_energy: f64,
};

const FinalWavefunctionSetup = struct {
    nocc: usize,
    local_cfg: local_potential.LocalPotentialConfig,
    use_iterative_config: bool,
    nonlocal_enabled: bool,
    fft_index_map: []usize,
    local_r: ?[]f64,
    iter_max_iter: usize,
    iter_tol: f64,

    fn deinit(self: *const FinalWavefunctionSetup, alloc: std.mem.Allocator) void {
        alloc.free(self.fft_index_map);
        if (self.local_r) |values| alloc.free(values);
    }
};

fn init_final_wavefunction_setup(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    ionic: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    potential: hamiltonian.PotentialGrid,
) !FinalWavefunctionSetup {
    const is_paw_wf = has_paw(species);
    const qij_enabled = has_qij(species) and !is_paw_wf;
    const use_iterative_config = (cfg.scf.solver == .iterative or
        cfg.scf.solver == .cg or
        cfg.scf.solver == .auto) and !qij_enabled;
    const fft_index_map = try build_fft_index_map(alloc, grid);
    var local_r: ?[]f64 = null;
    if (use_iterative_config) {
        local_r = try potential_mod.build_local_potential_real(alloc, grid, ionic, potential);
    }
    return .{
        .nocc = @as(usize, @intFromFloat(std.math.ceil(total_electrons(species, atoms) / 2.0))),
        .local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, grid.cell),
        .use_iterative_config = use_iterative_config,
        .nonlocal_enabled = cfg.scf.enable_nonlocal and has_nonlocal(species),
        .fft_index_map = fft_index_map,
        .local_r = local_r,
        .iter_max_iter = cfg.scf.iterative_max_iter,
        .iter_tol = cfg.scf.iterative_tol,
    };
}

fn maybe_log_final_wavefunction_potential_mean(
    io: std.Io,
    cfg: *const config.Config,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]const f64,
) !void {
    if (!cfg.scf.debug_fermi) return;
    const values = local_r orelse return;
    var sum: f64 = 0.0;
    for (values) |v| sum += v;
    const mean_local = sum / @as(f64, @floatFromInt(values.len));
    const pot_g0 = potential.value_at(0, 0, 0);
    try log_local_potential_mean(io, "scf", mean_local, "pot_g0", pot_g0.r);
}

fn fill_occupied_bands(
    occupations: []f64,
    eigen_data: KpointEigenData,
    kp: KPoint,
    nocc: usize,
    spin_factor: f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
) void {
    @memset(occupations, 0.0);
    var band: usize = 0;
    while (band < @min(nocc, eigen_data.nbands)) : (band += 1) {
        occupations[band] = 1.0;
        band_energy.* += kp.weight * spin_factor * eigen_data.values[band];
        if (eigen_data.nonlocal) |nl| {
            nonlocal_energy.* += kp.weight * spin_factor * nl[band];
        }
    }
}

fn compute_final_kpoint_eigen_data(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    setup: *const FinalWavefunctionSetup,
    kpoint_cache: *KpointCache,
    apply_cache: *apply.KpointApplyCache,
    wf_fft_plan: fft.Fft3dPlan,
) !KpointEigenData {
    return compute_kpoint_eigen_data(
        alloc,
        io,
        cfg,
        grid,
        kp,
        species,
        atoms,
        recip,
        volume,
        setup.local_cfg,
        potential,
        setup.local_r,
        setup.nocc,
        setup.use_iterative_config,
        has_qij(species) and !has_paw(species),
        setup.nonlocal_enabled,
        setup.fft_index_map,
        setup.iter_max_iter,
        setup.iter_tol,
        cfg.scf.iterative_reuse_vectors,
        kpoint_cache,
        null,
        wf_fft_plan,
        apply_cache,
        radial_tables,
        paw_tabs,
    );
}

fn compute_final_kpoint_wavefunction(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    setup: *const FinalWavefunctionSetup,
    kpoint_cache: *KpointCache,
    apply_cache: *apply.KpointApplyCache,
    wf_fft_plan: fft.Fft3dPlan,
    spin_factor: f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
) !KpointWavefunction {
    const eigen_data = try compute_final_kpoint_eigen_data(
        alloc,
        io,
        cfg,
        grid,
        kp,
        species,
        atoms,
        recip,
        volume,
        potential,
        radial_tables,
        paw_tabs,
        setup,
        kpoint_cache,
        apply_cache,
        wf_fft_plan,
    );
    errdefer {
        var ed = eigen_data;
        ed.deinit(alloc);
    }

    const occupations = try alloc.alloc(f64, eigen_data.nbands);
    errdefer alloc.free(occupations);

    fill_occupied_bands(
        occupations,
        eigen_data,
        kp,
        setup.nocc,
        spin_factor,
        band_energy,
        nonlocal_energy,
    );
    const wavefunction = KpointWavefunction{
        .k_frac = kp.k_frac,
        .k_cart = kp.k_cart,
        .weight = kp.weight,
        .basis_len = eigen_data.basis_len,
        .nbands = eigen_data.nbands,
        .eigenvalues = eigen_data.values,
        .coefficients = eigen_data.vectors,
        .occupations = occupations,
    };
    if (eigen_data.nonlocal) |nl| alloc.free(nl);
    return wavefunction;
}

fn find_wavefunction_fermi_level(kp_wavefunctions: []const KpointWavefunction) f64 {
    var fermi_level: f64 = -std.math.inf(f64);
    for (kp_wavefunctions) |kw| {
        for (kw.eigenvalues, 0..) |e, band| {
            if (kw.occupations[band] > 0.0) {
                fermi_level = @max(fermi_level, e);
            }
        }
    }
    return fermi_level;
}

/// Compute final wavefunctions for force calculation.
pub fn compute_final_wavefunctions_with_spin_factor(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    grid: Grid,
    kpoints: []KPoint,
    ionic: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    spin_factor: f64,
) !WavefunctionResult {
    var setup = try init_final_wavefunction_setup(
        alloc,
        &cfg,
        grid,
        ionic,
        species,
        atoms,
        potential,
    );
    defer setup.deinit(alloc);

    try maybe_log_final_wavefunction_potential_mean(io, &cfg, potential, setup.local_r);

    var kp_wavefunctions = try alloc.alloc(KpointWavefunction, kpoints.len);
    var filled: usize = 0;
    errdefer {
        for (kp_wavefunctions[0..filled]) |*kw| {
            kw.deinit(alloc);
        }
        alloc.free(kp_wavefunctions);
    }

    var wf_fft_plan = try fft.Fft3dPlan.init_with_backend(
        alloc,
        io,
        grid.nx,
        grid.ny,
        grid.nz,
        cfg.scf.fft_backend,
    );
    defer wf_fft_plan.deinit(alloc);

    var band_energy: f64 = 0.0;
    var nonlocal_energy: f64 = 0.0;

    for (kpoints, 0..) |kp, kidx| {
        kp_wavefunctions[kidx] = try compute_final_kpoint_wavefunction(
            alloc,
            io,
            &cfg,
            grid,
            kp,
            species,
            atoms,
            recip,
            volume,
            potential,
            radial_tables,
            paw_tabs,
            &setup,
            &kpoint_cache[kidx],
            &apply_caches[kidx],
            wf_fft_plan,
            spin_factor,
            &band_energy,
            &nonlocal_energy,
        );
        filled += 1;
    }

    return WavefunctionResult{
        .wavefunctions = WavefunctionData{
            .kpoints = kp_wavefunctions,
            .ecut_ry = cfg.scf.ecut_ry,
            .fermi_level = find_wavefunction_fermi_level(kp_wavefunctions),
        },
        .band_energy = band_energy,
        .nonlocal_energy = nonlocal_energy,
    };
}

/// Compute density from Kohn-Sham eigenvectors.
const ScfLoopProfile = logging.ScfLoopProfile;

const DensitySetup = struct {
    nelec: f64,
    nocc: usize,
    has_qij: bool,
    use_iterative_config: bool,
    nonlocal_enabled: bool,
    fft_index_map: []usize,
    local_r: ?[]f64,
    iter_max_iter: usize,
    iter_tol: f64,
    local_cfg: local_potential.LocalPotentialConfig,

    fn deinit(self: *const DensitySetup, alloc: std.mem.Allocator) void {
        alloc.free(self.fft_index_map);
        if (self.local_r) |values| alloc.free(values);
    }
};

const DensityContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    ionic: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    kpoint_cache: []KpointCache,
    apply_caches: ?[]apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_rhoij: ?*paw_mod.RhoIJ,
};

fn build_density_context(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    ionic: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    kpoint_cache: []KpointCache,
    apply_caches: ?[]apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_rhoij: ?*paw_mod.RhoIJ,
) DensityContext {
    return .{
        .alloc = alloc,
        .io = io,
        .cfg = cfg,
        .grid = grid,
        .kpoints = kpoints,
        .ionic = ionic,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume = volume,
        .potential = potential,
        .kpoint_cache = kpoint_cache,
        .apply_caches = apply_caches,
        .radial_tables = radial_tables,
        .paw_tabs = paw_tabs,
        .paw_rhoij = paw_rhoij,
    };
}

fn init_density_setup(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    ionic: hamiltonian.PotentialGrid,
    potential: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    scf_iter: usize,
    loop_profile: ?ScfLoopProfile,
    paw_rhoij: ?*paw_mod.RhoIJ,
) !DensitySetup {
    if (paw_rhoij) |rij| rij.reset();
    const nelec = total_electrons(species, atoms);
    const qij_enabled = has_qij(species) and !has_paw(species);
    const use_iterative_config = (cfg.scf.solver == .iterative or
        cfg.scf.solver == .cg or
        cfg.scf.solver == .auto) and !qij_enabled;
    if ((cfg.scf.solver == .iterative or cfg.scf.solver == .cg or cfg.scf.solver == .auto) and
        !use_iterative_config and qij_enabled)
    {
        try log_iterative_solver_disabled(io, "QIJ present");
    }

    const t_lr = if (loop_profile != null) profile_start(io) else null;
    const local_r = if (use_iterative_config)
        try potential_mod.build_local_potential_real(alloc, grid, ionic, potential)
    else
        null;
    if (loop_profile) |lp| profile_add(io, lp.build_local_r_ns, t_lr);

    const t_fm = if (loop_profile != null) profile_start(io) else null;
    const fft_index_map = try build_fft_index_map(alloc, grid);
    if (loop_profile) |lp| profile_add(io, lp.build_fft_map_ns, t_fm);

    return .{
        .nelec = nelec,
        .nocc = @as(usize, @intFromFloat(std.math.ceil(nelec / 2.0))),
        .has_qij = qij_enabled,
        .use_iterative_config = use_iterative_config,
        .nonlocal_enabled = cfg.scf.enable_nonlocal and has_nonlocal(species),
        .fft_index_map = fft_index_map,
        .local_r = local_r,
        .iter_max_iter = if (cfg.scf.iterative_warmup_steps > 0 and
            scf_iter < cfg.scf.iterative_warmup_steps)
            cfg.scf.iterative_warmup_max_iter
        else
            cfg.scf.iterative_max_iter,
        .iter_tol = if (cfg.scf.iterative_warmup_steps > 0 and
            scf_iter < cfg.scf.iterative_warmup_steps)
            cfg.scf.iterative_warmup_tol
        else
            cfg.scf.iterative_tol,
        .local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, grid.cell),
    };
}

fn maybe_check_density_hamiltonian_apply(
    ctx: *const DensityContext,
    setup: *const DensitySetup,
    scf_iter: usize,
) !void {
    if (!(ctx.cfg.scf.profile and scf_iter == 0 and ctx.kpoints.len > 0)) return;
    var check_local = setup.local_r;
    var check_allocated = false;
    if (check_local == null) {
        check_local = try potential_mod.build_local_potential_real(
            ctx.alloc,
            ctx.grid,
            ctx.ionic,
            ctx.potential,
        );
        check_allocated = true;
    }
    defer if (check_allocated) if (check_local) |values| ctx.alloc.free(values);

    var basis = try plane_wave.generate(
        ctx.alloc,
        ctx.recip,
        ctx.cfg.scf.ecut_ry,
        ctx.kpoints[0].k_cart,
    );
    defer basis.deinit(ctx.alloc);

    try check_hamiltonian_apply(
        ctx.alloc,
        ctx.io,
        ctx.grid,
        basis.gvecs,
        ctx.species,
        ctx.atoms,
        1.0 / ctx.volume,
        ctx.potential,
        check_local.?,
        setup.nonlocal_enabled,
        setup.fft_index_map,
    );
}

fn maybe_log_density_debug_diagnostics(
    ctx: *const DensityContext,
    setup: *const DensitySetup,
    scf_iter: usize,
) !void {
    if (!((ctx.cfg.scf.debug_nonlocal or ctx.cfg.scf.debug_local) and
        scf_iter == 0 and ctx.kpoints.len > 0)) return;
    var basis = try plane_wave.generate(
        ctx.alloc,
        ctx.recip,
        ctx.cfg.scf.ecut_ry,
        ctx.kpoints[0].k_cart,
    );
    defer basis.deinit(ctx.alloc);

    if (ctx.cfg.scf.debug_local) {
        try log_local_diagnostics(ctx.io, basis.gvecs, ctx.species, ctx.atoms, setup.local_cfg);
    }
    if (ctx.cfg.scf.debug_nonlocal) {
        try log_nonlocal_diagnostics(
            ctx.alloc,
            ctx.io,
            basis.gvecs,
            ctx.species,
            ctx.atoms,
            1.0 / ctx.volume,
        );
    }
}

fn compute_density_sequential(
    ctx: *const DensityContext,
    setup: *const DensitySetup,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
) !ScfProfile {
    var profile_total = ScfProfile{};
    var shared_fft_plan = try fft.Fft3dPlan.init_with_backend(
        ctx.alloc,
        ctx.io,
        ctx.grid.nx,
        ctx.grid.ny,
        ctx.grid.nz,
        ctx.cfg.scf.fft_backend,
    );
    defer shared_fft_plan.deinit(ctx.alloc);

    const profile_ptr: ?*ScfProfile = if (ctx.cfg.scf.profile) &profile_total else null;
    for (ctx.kpoints, 0..) |kp, kidx| {
        if (!ctx.cfg.scf.quiet) try log_kpoint(ctx.io, kidx, ctx.kpoints.len);
        const ac_ptr: ?*apply.KpointApplyCache = if (ctx.apply_caches) |acs|
            (if (kidx < acs.len) &acs[kidx] else null)
        else
            null;
        try compute_kpoint_contribution(
            ctx.alloc,
            ctx.io,
            ctx.cfg,
            ctx.grid,
            kp,
            ctx.species,
            ctx.atoms,
            ctx.recip,
            ctx.volume,
            setup.local_cfg,
            ctx.potential,
            setup.local_r,
            setup.nocc,
            setup.nelec,
            setup.use_iterative_config,
            setup.has_qij,
            setup.nonlocal_enabled,
            setup.fft_index_map,
            setup.iter_max_iter,
            setup.iter_tol,
            ctx.cfg.scf.iterative_reuse_vectors,
            &ctx.kpoint_cache[kidx],
            rho,
            band_energy,
            nonlocal_energy,
            profile_ptr,
            shared_fft_plan,
            ac_ptr,
            ctx.radial_tables,
            ctx.paw_tabs,
            ctx.paw_rhoij,
        );
    }
    return profile_total;
}

const DensityParallelState = struct {
    rho_locals: []f64,
    band_energies: []f64,
    nonlocal_energies: []f64,
    profiles: ?[]ScfProfile,
    fft_plans: []fft.Fft3dPlan,

    fn deinit(self: *DensityParallelState, alloc: std.mem.Allocator) void {
        for (self.fft_plans) |*plan| plan.deinit(alloc);
        alloc.free(self.fft_plans);
        if (self.profiles) |profiles| alloc.free(profiles);
        alloc.free(self.nonlocal_energies);
        alloc.free(self.band_energies);
        alloc.free(self.rho_locals);
    }
};

fn alloc_density_profiles(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    thread_count: usize,
) !?[]ScfProfile {
    if (!cfg.scf.profile) return null;
    const profiles = try alloc.alloc(ScfProfile, thread_count);
    for (profiles) |*p| p.* = .{};
    return profiles;
}

fn init_density_fft_plans(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    cfg: *const config.Config,
    thread_count: usize,
) ![]fft.Fft3dPlan {
    const fft_plans = try alloc.alloc(fft.Fft3dPlan, thread_count);
    errdefer alloc.free(fft_plans);
    for (fft_plans) |*plan| {
        plan.* = try fft.Fft3dPlan.init_with_backend(
            alloc,
            io,
            grid.nx,
            grid.ny,
            grid.nz,
            cfg.scf.fft_backend,
        );
    }
    return fft_plans;
}

fn init_density_parallel_state(
    ctx: *const DensityContext,
    thread_count: usize,
) !DensityParallelState {
    const ngrid = ctx.grid.count();
    const rho_locals = try ctx.alloc.alloc(f64, ngrid * thread_count);
    errdefer ctx.alloc.free(rho_locals);
    @memset(rho_locals, 0.0);

    const band_energies = try ctx.alloc.alloc(f64, thread_count);
    errdefer ctx.alloc.free(band_energies);
    @memset(band_energies, 0.0);

    const nonlocal_energies = try ctx.alloc.alloc(f64, thread_count);
    errdefer ctx.alloc.free(nonlocal_energies);
    @memset(nonlocal_energies, 0.0);

    return .{
        .rho_locals = rho_locals,
        .band_energies = band_energies,
        .nonlocal_energies = nonlocal_energies,
        .profiles = try alloc_density_profiles(ctx.alloc, ctx.cfg, thread_count),
        .fft_plans = try init_density_fft_plans(ctx.alloc, ctx.io, ctx.grid, ctx.cfg, thread_count),
    };
}

fn run_density_worker_threads(
    alloc: std.mem.Allocator,
    shared: *KpointShared,
    thread_count: usize,
) !void {
    const workers = try alloc.alloc(KpointWorker, thread_count);
    defer alloc.free(workers);

    const threads = try alloc.alloc(std.Thread, thread_count);
    defer alloc.free(threads);

    for (0..thread_count) |t| {
        workers[t] = .{ .shared = shared, .thread_index = t };
        threads[t] = try std.Thread.spawn(.{}, kpoint_worker, .{&workers[t]});
    }
    for (threads) |thread| thread.join();
}

fn reduce_density_parallel_results(
    rho: []f64,
    state: *const DensityParallelState,
    thread_count: usize,
    ngrid: usize,
    band_energy: *f64,
    nonlocal_energy: *f64,
    profile_total: *ScfProfile,
) void {
    for (0..thread_count) |t| {
        band_energy.* += state.band_energies[t];
        nonlocal_energy.* += state.nonlocal_energies[t];
        const start = t * ngrid;
        for (state.rho_locals[start .. start + ngrid], 0..) |value, i| {
            rho[i] += value;
        }
        if (state.profiles) |profiles| merge_profile(profile_total, profiles[t]);
    }
}

fn build_density_kpoint_shared(
    ctx: *const DensityContext,
    setup: *const DensitySetup,
    state: *const DensityParallelState,
    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    worker_error: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
) KpointShared {
    return .{
        .io = ctx.io,
        .cfg = ctx.cfg,
        .grid = ctx.grid,
        .kpoints = ctx.kpoints,
        .ionic = ctx.ionic,
        .species = ctx.species,
        .atoms = ctx.atoms,
        .recip = ctx.recip,
        .volume = ctx.volume,
        .local_cfg = setup.local_cfg,
        .potential = ctx.potential,
        .local_r = setup.local_r,
        .nocc = setup.nocc,
        .nelec = setup.nelec,
        .use_iterative_config = setup.use_iterative_config,
        .has_qij = setup.has_qij,
        .nonlocal_enabled = setup.nonlocal_enabled,
        .fft_index_map = setup.fft_index_map,
        .iter_max_iter = setup.iter_max_iter,
        .iter_tol = setup.iter_tol,
        .reuse_vectors = ctx.cfg.scf.iterative_reuse_vectors,
        .rho_locals = state.rho_locals,
        .band_energies = state.band_energies,
        .nonlocal_energies = state.nonlocal_energies,
        .profiles = state.profiles,
        .ngrid = ctx.grid.count(),
        .kpoint_cache = ctx.kpoint_cache,
        .apply_caches = ctx.apply_caches,
        .fft_plans = state.fft_plans,
        .radial_tables = ctx.radial_tables,
        .paw_tabs = ctx.paw_tabs,
        .next_index = next_index,
        .stop = stop,
        .err = worker_error,
        .err_mutex = err_mutex,
        .log_mutex = log_mutex,
    };
}

fn compute_density_parallel(
    ctx: *const DensityContext,
    setup: *const DensitySetup,
    rho: []f64,
    thread_count: usize,
) !DensityResult {
    var state = try init_density_parallel_state(ctx, thread_count);
    defer state.deinit(ctx.alloc);

    var next_index = std.atomic.Value(usize).init(0);
    var stop = std.atomic.Value(u8).init(0);
    var worker_error: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;
    var shared = build_density_kpoint_shared(
        ctx,
        setup,
        &state,
        &next_index,
        &stop,
        &worker_error,
        &err_mutex,
        &log_mutex,
    );

    try run_density_worker_threads(ctx.alloc, &shared, thread_count);
    if (worker_error) |err| return err;

    var band_energy: f64 = 0.0;
    var nonlocal_energy: f64 = 0.0;
    var profile_total = ScfProfile{};
    reduce_density_parallel_results(
        rho,
        &state,
        thread_count,
        ctx.grid.count(),
        &band_energy,
        &nonlocal_energy,
        &profile_total,
    );
    if (ctx.cfg.scf.profile and !ctx.cfg.scf.quiet) {
        try log_profile(ctx.io, profile_total, ctx.kpoints.len);
    }
    return .{
        .rho = rho,
        .band_energy = band_energy,
        .nonlocal_energy = nonlocal_energy,
        .fermi_level = std.math.nan(f64),
    };
}

fn compute_density_no_smearing(
    ctx: *const DensityContext,
    setup: *const DensitySetup,
) !DensityResult {
    const rho = try ctx.alloc.alloc(f64, ctx.grid.count());
    errdefer ctx.alloc.free(rho);
    @memset(rho, 0.0);

    const thread_count = if (ctx.paw_rhoij != null)
        @as(usize, 1)
    else
        kpoint_thread_count(ctx.kpoints.len, ctx.cfg.scf.kpoint_threads);
    if (thread_count <= 1) {
        var band_energy: f64 = 0.0;
        var nonlocal_energy: f64 = 0.0;
        const profile_total = try compute_density_sequential(
            ctx,
            setup,
            rho,
            &band_energy,
            &nonlocal_energy,
        );
        if (ctx.cfg.scf.profile and !ctx.cfg.scf.quiet) {
            try log_profile(ctx.io, profile_total, ctx.kpoints.len);
        }
        return .{
            .rho = rho,
            .band_energy = band_energy,
            .nonlocal_energy = nonlocal_energy,
            .fermi_level = std.math.nan(f64),
        };
    }
    return try compute_density_parallel(ctx, setup, rho, thread_count);
}

fn compute_density_with_smearing(
    ctx: *const DensityContext,
    setup: *const DensitySetup,
) !DensityResult {
    return try compute_density_smearing(
        ctx.alloc,
        ctx.io,
        ctx.cfg,
        ctx.grid,
        ctx.kpoints,
        ctx.species,
        ctx.atoms,
        ctx.recip,
        ctx.volume,
        ctx.potential,
        setup.local_r,
        setup.nocc,
        setup.nelec,
        setup.use_iterative_config,
        setup.has_qij,
        setup.nonlocal_enabled,
        setup.fft_index_map,
        setup.iter_max_iter,
        setup.iter_tol,
        ctx.kpoint_cache,
        ctx.apply_caches,
        ctx.radial_tables,
        ctx.paw_tabs,
        ctx.paw_rhoij,
    );
}

fn compute_density(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    grid: Grid,
    kpoints: []KPoint,
    ionic: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    scf_iter: usize,
    kpoint_cache: []KpointCache,
    apply_caches: ?[]apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    loop_profile: ?ScfLoopProfile,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_rhoij: ?*paw_mod.RhoIJ,
) !DensityResult {
    const ctx = build_density_context(
        alloc,
        io,
        &cfg,
        grid,
        kpoints,
        ionic,
        species,
        atoms,
        recip,
        volume,
        potential,
        kpoint_cache,
        apply_caches,
        radial_tables,
        paw_tabs,
        paw_rhoij,
    );
    var setup = try init_density_setup(
        alloc,
        io,
        &cfg,
        grid,
        ionic,
        potential,
        species,
        atoms,
        scf_iter,
        loop_profile,
        paw_rhoij,
    );
    defer setup.deinit(alloc);

    try maybe_check_density_hamiltonian_apply(&ctx, &setup, scf_iter);
    try maybe_log_density_debug_diagnostics(&ctx, &setup, scf_iter);
    if (smearing_active(&cfg)) return try compute_density_with_smearing(&ctx, &setup);
    return try compute_density_no_smearing(&ctx, &setup);
}

const compute_density_smearing = smearing_mod.compute_density_smearing;

pub const density_diff = util.density_diff;

pub const has_nonlocal = util.has_nonlocal;
pub const has_qij = util.has_qij;
pub const has_paw = util.has_paw;
pub const total_electrons = util.total_electrons;

test {
    _ = @import("band_solver.zig");
    _ = @import("gvec_iter.zig");
    _ = @import("paw_scf.zig");
    _ = @import("smearing.zig");
}

test "auto grid chooses fft-friendly size for aluminum" {
    const cell_ang = math.Mat3.from_rows(
        .{ .x = 0.0, .y = 2.025, .z = 2.025 },
        .{ .x = 2.025, .y = 0.0, .z = 2.025 },
        .{ .x = 2.025, .y = 2.025, .z = 0.0 },
    );
    const cell_bohr = cell_ang.scale(math.units_scale_to_bohr(.angstrom));
    const recip = math.reciprocal(cell_bohr);
    const grid = grid_mod.auto_grid(15.0, 1.0, recip);
    try std.testing.expect(grid[0] >= 3);
    try std.testing.expect(grid[1] >= 3);
    try std.testing.expect(grid[2] >= 3);
}
