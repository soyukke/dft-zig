// style-ignore: function_too_long

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
pub const applyHamiltonian = apply.applyHamiltonian;
pub const applyHamiltonianBatched = apply.applyHamiltonianBatched;
const checkHamiltonianApply = apply.checkHamiltonianApply;
pub const KpointApplyCache = apply.KpointApplyCache;
pub const NonlocalContext = apply.NonlocalContext;
pub const NonlocalSpecies = apply.NonlocalSpecies;
pub const applyNonlocalPotential = apply.applyNonlocalPotential;

pub const KpointCache = kpoints_mod.KpointCache;
const KpointShared = kpoints_mod.KpointShared;
const KpointWorker = kpoints_mod.KpointWorker;
const KpointEigenData = kpoints_mod.KpointEigenData;
const computeKpointContribution = kpoints_mod.computeKpointContribution;
const computeKpointEigenData = kpoints_mod.computeKpointEigenData;
pub const kpointThreadCount = kpoints_mod.kpointThreadCount;
const kpointWorker = kpoints_mod.kpointWorker;
const findFermiLevelSpin = kpoints_mod.findFermiLevelSpin;
const accumulateKpointDensitySmearingSpin = kpoints_mod.accumulateKpointDensitySmearingSpin;
const SmearingShared = kpoints_mod.SmearingShared;

const buildFftIndexMap = fft_grid.buildFftIndexMap;
pub const realToReciprocal = fft_grid.realToReciprocal;
pub const reciprocalToReal = fft_grid.reciprocalToReal;
pub const fftReciprocalToComplexInPlace = fft_grid.fftReciprocalToComplexInPlace;
const fftReciprocalToComplexInPlaceMapped = fft_grid.fftReciprocalToComplexInPlaceMapped;
pub const fftComplexToReciprocalInPlace = fft_grid.fftComplexToReciprocalInPlace;
const fftComplexToReciprocalInPlaceMapped = fft_grid.fftComplexToReciprocalInPlaceMapped;
const indexToFreq = fft_grid.indexToFreq;

const mixDensity = mixing.mixDensity;
const mixDensityKerker = mixing.mixDensityKerker;
const PulayMixer = mixing.PulayMixer;
pub const ComplexPulayMixer = mixing.ComplexPulayMixer;

const ScfLog = logging.ScfLog;
const ScfProfile = logging.ScfProfile;
const logProgress = logging.logProgress;
const logIterStart = logging.logIterStart;
const logKpoint = logging.logKpoint;
const logProfile = logging.logProfile;
const logLoopProfile = logging.logLoopProfile;
const logNonlocalDiagnostics = logging.logNonlocalDiagnostics;
const logLocalDiagnostics = logging.logLocalDiagnostics;
const logEnergySummary = logging.logEnergySummary;
const logLocalPotentialMean = logging.logLocalPotentialMean;
const logIterativeSolverDisabled = logging.logIterativeSolverDisabled;
const profileStart = logging.profileStart;
const profileAdd = logging.profileAdd;
const mergeProfile = logging.mergeProfile;

pub const PwGridMap = pw_grid_map.PwGridMap;

// gvec_iter re-exports
pub const GVecIterator = gvec_iter.GVecIterator;
const GVecItem = gvec_iter.GVecItem;

// xc_fields re-exports
const XcFields = xc_fields_mod.XcFields;
const XcFieldsSpin = xc_fields_mod.XcFieldsSpin;
const Gradient = xc_fields_mod.Gradient;
pub const computeXcFields = xc_fields_mod.computeXcFields;
const computeXcFieldsSpin = xc_fields_mod.computeXcFieldsSpin;
pub const gradientFromReal = xc_fields_mod.gradientFromReal;
pub const divergenceFromReal = xc_fields_mod.divergenceFromReal;

// potential re-exports
const buildPotentialGrid = potential_mod.buildPotentialGrid;
const buildPotentialGridSpin = potential_mod.buildPotentialGridSpin;
pub const buildIonicPotentialGrid = potential_mod.buildIonicPotentialGrid;
pub const buildLocalPotentialReal = potential_mod.buildLocalPotentialReal;
const filterDensityToEcutrho = potential_mod.filterDensityToEcutrho;
const SpinPotentialGrids = potential_mod.SpinPotentialGrids;

// core_density re-exports
pub const hasNlcc = core_density.hasNlcc;
pub const buildCoreDensity = core_density.buildCoreDensity;

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
pub const initBandIterativeContext = band_solver.initBandIterativeContext;
pub const bandEigenvaluesIterative = band_solver.bandEigenvaluesIterative;
pub const bandEigenvaluesIterativeExt = band_solver.bandEigenvaluesIterativeExt;

fn totalCharge(rho: []f64, grid: Grid) f64 {
    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var sum: f64 = 0.0;
    for (rho) |value| {
        sum += value * dv;
    }
    return sum;
}

const smearingActive = smearing_mod.smearingActive;

fn wrapGridIndex(g: i32, min: i32, n: usize) usize {
    const ni = @as(i32, @intCast(n));
    const idx = @mod(g - min, ni);
    return @as(usize, @intCast(idx));
}

pub fn symmetrizeDensity(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    ops: []const symmetry.SymOp,
    use_rfft: bool,
) !void {
    if (ops.len <= 1) return;

    const rho_g = try realToReciprocal(alloc, grid, rho, use_rfft);
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

                    const ix = wrapGridIndex(mh, grid.min_h, grid.nx);
                    const iy = wrapGridIndex(mk, grid.min_k, grid.ny);
                    const iz = wrapGridIndex(ml, grid.min_l, grid.nz);
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

    const rho_real = try reciprocalToReal(alloc, grid, rho_sym);
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
fn generateKpointsForConfig(
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
        return try kmesh_mod.generateKmeshSymmetry(
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
    return try kmesh_mod.generateKmesh(alloc, cfg.scf.kmesh, recip, shift);
}

/// Build per-species radial projector tables used by the nonlocal context.
fn buildRadialTablesForSpecies(
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
fn buildPawTablesAndRhoIJ(
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
fn buildPawGauntTable(
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
fn ecutrhoForPaw(cfg: *const config.Config, is_paw: bool) ?f64 {
    if (!is_paw) return null;
    const gs_val = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_val * gs_val;
}

/// Build the optional radial tables (for nonlocal context warm-start).
fn maybeBuildRadialTables(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
) !?[]nonlocal_mod.RadialTableSet {
    const nonlocal_enabled_run = cfg.scf.enable_nonlocal and hasNonlocal(species);
    if (!nonlocal_enabled_run) return null;
    return try buildRadialTablesForSpecies(alloc, species, cfg.scf.ecut_ry);
}

/// Free radial tables (helper for errdefer blocks).
fn freeRadialTables(alloc: std.mem.Allocator, buf: []nonlocal_mod.RadialTableSet) void {
    for (buf) |*t| {
        if (t.tables.len > 0) t.deinit(alloc);
    }
    alloc.free(buf);
}

/// Free PAW bundle contents (helper for errdefer blocks).
fn freePawBundle(alloc: std.mem.Allocator, bundle: *PawBundle) void {
    if (bundle.tabs) |tabs| {
        for (tabs) |*t| t.deinit(alloc);
        alloc.free(tabs);
    }
    if (bundle.rhoij) |*rij| rij.deinit(alloc);
    if (bundle.gaunt) |*gt| gt.deinit(alloc);
}

/// Build optional NLCC core density.
fn maybeBuildRhoCore(
    alloc: std.mem.Allocator,
    grid: grid_mod.Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) !?[]f64 {
    if (!core_density.hasNlcc(species)) return null;
    return try core_density.buildCoreDensity(alloc, grid, species, atoms);
}

/// Combined PAW table + Gaunt build for initScfCommon.
const PawBundle = struct {
    tabs: ?[]paw_mod.PawTab,
    rhoij: ?paw_mod.RhoIJ,
    gaunt: ?paw_mod.GauntTable,
};

fn maybeBuildPawBundle(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    is_paw: bool,
) !PawBundle {
    if (!is_paw) return .{ .tabs = null, .rhoij = null, .gaunt = null };
    const paw_init = try buildPawTablesAndRhoIJ(alloc, species, atoms, cfg.scf.ecut_ry);
    const gaunt = try buildPawGauntTable(alloc, paw_init.tabs);
    return .{ .tabs = paw_init.tabs, .rhoij = paw_init.rhoij, .gaunt = gaunt };
}

fn initScfCommon(params: ScfParams) !ScfCommon {
    const alloc = params.alloc;
    const io = params.io;
    const cfg = params.cfg;
    const species = params.model.species;
    const atoms = params.model.atoms;
    const recip = params.model.recip;
    const volume_bohr = params.model.volume_bohr;
    const ff_tables = params.ff_tables;

    const grid = grid_mod.gridFromConfig(cfg, recip, volume_bohr);
    const total_electrons = totalElectrons(species, atoms);

    // Compute cutoff radius for isolated systems
    const coulomb_r_cut: ?f64 = if (cfg.boundary == .isolated)
        coulomb_mod.cutoffRadius(grid.cell)
    else
        null;

    const local_cfg = local_potential.resolve(
        cfg.scf.local_potential,
        cfg.ewald.alpha,
        grid.cell,
    );

    const is_paw = hasPaw(species);
    const ecutrho = ecutrhoForPaw(cfg, is_paw);

    var ionic = try potential_mod.buildIonicPotentialGrid(
        alloc,
        grid,
        species,
        atoms,
        local_cfg,
        ff_tables,
        ecutrho,
    );
    errdefer ionic.deinit(alloc);

    var log = try ScfLog.init(alloc, io, cfg.out_dir);
    errdefer log.deinit();

    try log.writeHeader();

    const kpoints = try generateKpointsForConfig(
        alloc,
        io,
        cfg,
        recip,
        grid.cell,
        atoms,
    );
    errdefer alloc.free(kpoints);

    const sym_ops = if (cfg.scf.symmetry)
        try symmetry.getSymmetryOps(alloc, grid.cell, atoms, 1e-6)
    else
        null;
    errdefer if (sym_ops) |ops| alloc.free(ops);

    const rho_core = try maybeBuildRhoCore(alloc, grid, species, atoms);
    errdefer if (rho_core) |values| alloc.free(values);

    const radial_tables_buf = try maybeBuildRadialTables(alloc, cfg, species);
    errdefer if (radial_tables_buf) |buf| freeRadialTables(alloc, buf);

    const radial_tables: ?[]const nonlocal_mod.RadialTableSet = radial_tables_buf;

    const pulay_mixer: ?PulayMixer = if (cfg.scf.pulay_history > 0)
        PulayMixer.init(alloc, cfg.scf.pulay_history)
    else
        null;

    var paw = try maybeBuildPawBundle(alloc, cfg, species, atoms, is_paw);
    errdefer freePawBundle(alloc, &paw);

    return buildScfCommonStruct(.{
        .alloc = alloc,
        .cfg = cfg,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume_bohr = volume_bohr,
        .grid = grid,
        .total_electrons = total_electrons,
        .local_cfg = local_cfg,
        .ionic = ionic,
        .log = log,
        .kpoints = kpoints,
        .sym_ops = sym_ops,
        .rho_core = rho_core,
        .radial_tables_buf = radial_tables_buf,
        .radial_tables = radial_tables,
        .pulay_mixer = pulay_mixer,
        .coulomb_r_cut = coulomb_r_cut,
        .paw = paw,
        .is_paw = is_paw,
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
    ionic: potential_mod.IonicPotentialGrid,
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

fn buildScfCommonStruct(s: ScfCommonSetup) ScfCommon {
    return ScfCommon{
        .alloc = s.alloc,
        .cfg = s.cfg,
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
fn initRhoFromUserOrFlat(
    rho: []f64,
    initial_density: ?[]const f64,
    total_electrons: f64,
    grid_volume: f64,
) void {
    if (initial_density) |init_rho| {
        if (init_rho.len == rho.len) {
            @memcpy(rho, init_rho);
            return;
        }
    }
    const rho0 = total_electrons / grid_volume;
    @memset(rho, rho0);
}

/// Warm-start the per-kpoint cache from a previously converged SCF run, if provided.
fn warmStartKpointCache(
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
fn pawEcutrho(cfg: *const config.Config, is_paw: bool) ?f64 {
    if (!is_paw) return null;
    const gs_val = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_val * gs_val;
}

/// Copy NLCC core density to a newly-allocated buffer for result, or null.
fn copyRhoCoreForResult(alloc: std.mem.Allocator, rho_core: ?[]const f64) !?[]f64 {
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
fn getOrAllocApplyCaches(
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
fn dupPerAtomDij(
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

fn appendEmptyDxc(
    alloc: std.mem.Allocator,
    dxc_list: *std.ArrayList([]f64),
) !void {
    try dxc_list.append(alloc, try alloc.alloc(f64, 0));
}

fn accumulateMatrixRhoijTrace(
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
fn computePerAtomDxcAndDC(
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
            try appendEmptyDxc(alloc, &dxc_list);
            continue;
        };
        if (si >= tabs.len or tabs[si].nbeta == 0) {
            try appendEmptyDxc(alloc, &dxc_list);
            continue;
        }
        const mt = prij.m_total_per_atom[ai];
        const sp_m_offsets = prij.m_offsets[ai];

        const dxc_m = try alloc.alloc(f64, mt * mt);
        try paw_mod.paw_xc.computePawDijXcAngular(
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
        accumulateMatrixRhoijTrace(&sum_dxc_rhoij, dxc_m, rhoij_m, mt);
        try dxc_list.append(alloc, dxc_m);

        const dij_h_dc = try alloc.alloc(f64, mt * mt);
        defer alloc.free(dij_h_dc);

        try paw_mod.paw_xc.computePawDijHartreeMultiL(
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
        accumulateMatrixRhoijTrace(&sum_dxc_rhoij, dij_h_dc, rhoij_m, mt);
    }
    const dxc = if (dxc_list.items.len > 0) try dxc_list.toOwnedSlice(alloc) else null_blk: {
        dxc_list.deinit(alloc);
        break :null_blk @as(?[][]f64, null);
    };
    return .{ .dxc = dxc, .sum_dxc_rhoij = sum_dxc_rhoij };
}

/// Copy contracted per-atom rhoij for force calculation.
fn duplicateContractedRhoIJ(
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
        prij.contractToRadial(a, copy);
        try rij_list.append(alloc, copy);
    }
    if (rij_list.items.len > 0) return try rij_list.toOwnedSlice(alloc);
    rij_list.deinit(alloc);
    return null;
}

/// Top-level PAW result extraction (transfers ownership + builds result arrays).
fn extractPawResults(
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
            out_dij.* = try dupPerAtomDij(alloc, nc.species, .radial);
            out_dij_m.* = try dupPerAtomDij(alloc, nc.species, .m_resolved);
        }
    }
    if (common.paw_rhoij) |*prij| {
        if (out_tabs.*) |tabs| {
            const dxc_result = try computePerAtomDxcAndDC(
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
        out_rhoij.* = try duplicateContractedRhoIJ(alloc, prij);
    }
}

/// Compute the ecutrho cutoff used for in-loop density filtering.
fn pawEcutrhoCompute(cfg: *const config.Config) f64 {
    const gs_comp = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_comp * gs_comp;
}

/// Symmetrize density over crystal operations, and PAW rhoij when applicable.
fn symmetrizeDensityAndRhoIJ(
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
            try symmetrizeDensity(alloc, grid, rho, ops, cfg.scf.use_rfft);
        }
    }
    if (common.paw_rhoij) |*prij| {
        if (cfg.scf.symmetry) {
            try paw_scf.symmetrizeRhoIJ(alloc, prij, species, atoms);
        }
    }
}

/// Filter density to |G|² < ecutrho sphere, writing the result back into rho.
fn filterAugmentedDensityInPlace(
    alloc: std.mem.Allocator,
    grid: grid_mod.Grid,
    rho: []f64,
    ecutrho: f64,
    use_rfft: bool,
) !void {
    const filtered = try potential_mod.filterDensityToEcutrho(alloc, grid, rho, ecutrho, use_rfft);
    defer alloc.free(filtered);

    @memcpy(rho, filtered);
}

/// Allocate an augmented density ρ̃ + n̂_hat for potential construction (PAW).
fn buildAugmentedDensity(
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
        try paw_scf.addPawCompensationCharge(
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
fn recordPotentialResidual(
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
fn maybeBuildRhoAugForEnergy(
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
    try paw_scf.addPawCompensationCharge(
        alloc,
        grid,
        aug,
        &common.paw_rhoij.?,
        common.paw_tabs.?,
        atoms,
        ecutrho_scf,
        &common.paw_gaunt.?,
    );
    const filtered = try potential_mod.filterDensityToEcutrho(
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

fn buildScfResult(s: ScfResultSetup) ScfResult {
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
fn mixPotentialMode(
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
            mixing.applyModelDielectricPreconditioner(
                grid,
                residual_g,
                cfg.scf.diemac,
                cfg.scf.dielng,
            );
        }
        const precond_f64: []f64 = @as([*]f64, @ptrCast(residual_g.ptr))[0..n_f64];
        try common.pulay_mixer.?.mixWithResidual(v_in, precond_f64, cfg.scf.mixing_beta);
    } else {
        const v_out_ptr = @as([*]const f64, @ptrCast(potential_out.values.ptr));
        const v_out: []const f64 = v_out_ptr[0..n_f64];
        mixDensity(v_in, v_out, cfg.scf.mixing_beta);
    }

    @memcpy(rho, new_rho);
    potential_out.deinit(alloc);
    keep_potential_out.* = true;
}

/// Density-mixing branch: mixes rho and rebuilds potential.
fn mixDensityMode(
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
            try common.pulay_mixer.?.mixKerkerPulay(
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
        try mixDensityKerker(
            alloc,
            grid,
            rho,
            new_rho,
            cfg.scf.mixing_beta,
            cfg.scf.kerker_q0,
            cfg.scf.use_rfft,
        );
    } else {
        mixDensity(rho, new_rho, cfg.scf.mixing_beta);
    }

    potential.deinit(alloc);
    potential.* = try potential_mod.buildPotentialGrid(
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
fn updatePawDijIfNeeded(
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
    try paw_scf.updatePawDij(
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

    var common = try initScfCommon(params);
    defer common.deinit();

    const grid = common.grid;
    const kpoints = common.kpoints;
    const radial_tables = common.radial_tables;

    // Dispatch to spin-polarized SCF loop if nspin=2
    if (cfg.scf.nspin == 2) {
        return scf_spin.runSpinPolarizedLoop(alloc, io, cfg, species, atoms, volume_bohr, &common);
    }

    const kpoint_cache = try alloc.alloc(KpointCache, kpoints.len);
    var kpoint_cache_owned = true; // track ownership for conditional defer
    defer {
        if (kpoint_cache_owned) {
            for (kpoint_cache) |*cache| {
                cache.deinit();
            }
            alloc.free(kpoint_cache);
        }
    }

    for (kpoint_cache) |*cache| {
        cache.* = .{};
    }
    try warmStartKpointCache(kpoint_cache, initial_kpoint_cache);

    const grid_count = grid.count();
    const rho = try alloc.alloc(f64, grid_count);
    errdefer alloc.free(rho);

    initRhoFromUserOrFlat(rho, initial_density, common.total_electrons, grid.volume);

    var iterations: usize = 0;
    var converged = false;
    // ecutrho spherical cutoff for PAW: limit G-space sums to |G|² < ecutrho
    const paw_ecutrho = pawEcutrho(cfg, common.is_paw);
    var potential = try potential_mod.buildPotentialGrid(
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
    errdefer potential.deinit(alloc);

    var vxc_r: ?[]f64 = null;
    errdefer if (vxc_r) |v| alloc.free(v);

    var vresid_last: ?hamiltonian.PotentialGrid = null;
    errdefer if (vresid_last) |*vresid| vresid.deinit(alloc);

    var last_band_energy: f64 = 0.0;
    var last_nonlocal_energy: f64 = 0.0;
    var last_entropy_energy: f64 = 0.0;
    var last_fermi_level: f64 = std.math.nan(f64);
    var last_potential_residual: f64 = 0.0;

    // Per-kpoint cache for NonlocalContext and PwGridMap (reused across SCF iterations)
    const apply_caches = try getOrAllocApplyCaches(alloc, initial_apply_caches, kpoints.len);
    var apply_caches_owned = true;
    defer {
        if (apply_caches_owned) {
            for (apply_caches) |*ac| ac.deinit(alloc);
            alloc.free(apply_caches);
        }
    }

    // SCF loop profiling accumulators (for unaccounted time analysis)
    var prof = ScfLoopProf{};

    while (iterations < cfg.scf.max_iter) : (iterations += 1) {
        if (!cfg.scf.quiet) {
            try logIterStart(io, iterations);
        }
        const t_density_start = if (cfg.scf.profile) profileStart(io) else null;
        const density_result = try computeDensity(
            alloc,
            io,
            cfg,
            grid,
            kpoints,
            common.ionic,
            species,
            atoms,
            recip,
            volume_bohr,
            potential,
            iterations,
            kpoint_cache,
            apply_caches,
            radial_tables,
            if (cfg.scf.profile) ScfLoopProfile{
                .build_local_r_ns = &prof.build_local_r_ns,
                .build_fft_map_ns = &prof.build_fft_map_ns,
            } else null,
            common.paw_tabs,
            if (common.paw_rhoij) |*rij| rij else null,
        );
        if (cfg.scf.profile) profileAdd(io, &prof.compute_density_ns, t_density_start);
        defer alloc.free(density_result.rho);

        last_band_energy = density_result.band_energy;
        last_nonlocal_energy = density_result.nonlocal_energy;
        last_entropy_energy = density_result.entropy_energy;
        last_fermi_level = density_result.fermi_level;

        try symmetrizeDensityAndRhoIJ(
            alloc,
            cfg,
            grid,
            &common,
            species,
            atoms,
            density_result.rho,
        );

        const ecutrho_comp = pawEcutrhoCompute(cfg);
        if (common.is_paw) {
            try filterAugmentedDensityInPlace(
                alloc,
                grid,
                density_result.rho,
                ecutrho_comp,
                cfg.scf.use_rfft,
            );
        }

        // PAW: build augmented density (ρ̃ + n_hat) for potential construction
        const rho_for_potential = if (common.is_paw)
            try buildAugmentedDensity(
                alloc,
                grid,
                density_result.rho,
                &common,
                atoms,
                ecutrho_comp,
                grid_count,
            )
        else
            density_result.rho;
        defer if (common.is_paw) alloc.free(rho_for_potential);

        // Capture V_xc(r) for NLCC force calculation when relax is enabled
        if (vxc_r) |old| alloc.free(old);
        vxc_r = null;
        const vxc_r_ptr: ?*?[]f64 = if (cfg.relax.enabled) &vxc_r else null;
        const t_build_pot_start = if (cfg.scf.profile) profileStart(io) else null;
        var potential_out = try potential_mod.buildPotentialGrid(
            alloc,
            grid,
            rho_for_potential,
            common.rho_core,
            cfg.scf.use_rfft,
            cfg.scf.xc,
            vxc_r_ptr,
            common.coulomb_r_cut,
            paw_ecutrho,
        );
        if (cfg.scf.profile) profileAdd(io, &prof.build_potential_ns, t_build_pot_start);
        var keep_potential_out = false;
        defer if (!keep_potential_out) potential_out.deinit(alloc);

        {
            const t_resid_start = if (cfg.scf.profile) profileStart(io) else null;
            last_potential_residual = try recordPotentialResidual(
                alloc,
                grid,
                potential,
                potential_out,
                &vresid_last,
            );
            if (cfg.scf.profile) profileAdd(io, &prof.residual_ns, t_resid_start);
        }

        const diff = densityDiff(rho, density_result.rho);
        const conv_value = switch (cfg.scf.convergence_metric) {
            .density => diff,
            .potential => last_potential_residual,
        };
        try common.log.writeIter(
            iterations,
            diff,
            last_potential_residual,
            last_band_energy,
            last_nonlocal_energy,
        );
        if (!cfg.scf.quiet) {
            try logProgress(
                io,
                iterations,
                diff,
                last_potential_residual,
                last_band_energy,
                last_nonlocal_energy,
            );
        }

        if (conv_value < cfg.scf.convergence) {
            converged = true;
            @memcpy(rho, density_result.rho);
            potential.deinit(alloc);
            potential = potential_out;
            keep_potential_out = true;
            break;
        }

        const t_mix_start = if (cfg.scf.profile) profileStart(io) else null;
        if (cfg.scf.mixing_mode == .potential) {
            try mixPotentialMode(
                alloc,
                cfg,
                grid,
                &common,
                iterations,
                rho,
                density_result.rho,
                &potential,
                &potential_out,
                &keep_potential_out,
            );
        } else {
            try mixDensityMode(
                alloc,
                cfg,
                grid,
                &common,
                iterations,
                rho,
                density_result.rho,
                &potential,
                paw_ecutrho,
            );
        }
        if (cfg.scf.profile) profileAdd(io, &prof.mixing_ns, t_mix_start);

        try updatePawDijIfNeeded(
            alloc,
            cfg,
            grid,
            &common,
            species,
            atoms,
            potential,
            apply_caches,
        );
    }

    // Print SCF loop profile
    if (cfg.scf.profile and !cfg.scf.quiet) {
        try logLoopProfile(
            io,
            prof.compute_density_ns,
            prof.build_potential_ns,
            prof.residual_ns,
            prof.mixing_ns,
            prof.build_local_r_ns,
            prof.build_fft_map_ns,
        );
    }

    // For PAW: build augmented density ρ̃ + n̂hat for energy computation.
    const rho_aug_for_energy = try maybeBuildRhoAugForEnergy(alloc, cfg, grid, &common, atoms, rho);
    defer if (rho_aug_for_energy) |a| alloc.free(a);

    var energy_terms = try energy_mod.computeEnergyTerms(.{
        .alloc = alloc,
        .io = io,
        .grid = grid,
        .species = species,
        .atoms = atoms,
        .rho = rho,
        .rho_core = common.rho_core,
        .rho_aug = rho_aug_for_energy,
        .band_energy = last_band_energy,
        .nonlocal_energy = last_nonlocal_energy,
        .entropy_energy = last_entropy_energy,
        .local_cfg = common.local_cfg,
        .ewald_cfg = cfg.ewald,
        .vdw_cfg = cfg.vdw,
        .xc_func = cfg.scf.xc,
        .use_rfft = cfg.scf.use_rfft,
        .quiet = cfg.scf.quiet,
        .coulomb_r_cut = common.coulomb_r_cut,
        .ecutrho = paw_ecutrho,
    });

    // Add PAW on-site energy correction
    if (common.is_paw) {
        if (common.paw_rhoij) |*prij| {
            if (common.paw_tabs) |tabs| {
                energy_terms.paw_onsite = try paw_scf.computePawOnsiteEnergyTotal(
                    alloc,
                    prij,
                    tabs,
                    species,
                    atoms,
                    cfg.scf.xc,
                    &common.paw_gaunt.?,
                    null,
                    null,
                );
                energy_terms.total += energy_terms.paw_onsite;
            }
        }
    }

    // Compute final wavefunctions for force calculation
    var wavefunctions: ?WavefunctionData = null;
    if (cfg.relax.enabled or cfg.dfpt.enabled or cfg.scf.compute_stress or cfg.dos.enabled) {
        const wfn_result = try computeFinalWavefunctionsWithSpinFactor(
            alloc,
            io,
            cfg,
            grid,
            kpoints,
            common.ionic,
            species,
            atoms,
            recip,
            volume_bohr,
            potential,
            kpoint_cache,
            apply_caches,
            radial_tables,
            common.paw_tabs,
            2.0,
        );
        wavefunctions = wfn_result.wavefunctions;
        last_band_energy = wfn_result.band_energy;
        last_nonlocal_energy = wfn_result.nonlocal_energy;
    }
    errdefer if (wavefunctions) |*wf| wf.deinit(alloc);

    // Transfer ownership to result (disable defer cleanup)
    kpoint_cache_owned = false;
    apply_caches_owned = false;

    // Extract PAW data for band calculation and forces
    var result_paw_tabs: ?[]paw_mod.PawTab = null;
    var result_paw_dij: ?[][]f64 = null;
    var result_paw_dij_m: ?[][]f64 = null;
    var result_paw_dxc: ?[][]f64 = null;
    var result_paw_rhoij: ?[][]f64 = null;
    if (common.is_paw) {
        try extractPawResults(
            alloc,
            cfg,
            species,
            atoms,
            apply_caches,
            &common,
            &energy_terms,
            &result_paw_tabs,
            &result_paw_dij,
            &result_paw_dij_m,
            &result_paw_dxc,
            &result_paw_rhoij,
        );
    }

    if (!cfg.scf.quiet) {
        const ionic_g0 = common.ionic.valueAt(0, 0, 0);
        const pot_g0 = potential.valueAt(0, 0, 0);
        try logEnergySummary(io, totalCharge(rho, grid), ionic_g0, pot_g0, energy_terms);
    }

    // Write SCF log after all energy corrections (including PAW D^xc double-counting)
    try common.log.writeResult(
        converged,
        iterations,
        energy_terms.total,
        energy_terms.band,
        energy_terms.hartree,
        energy_terms.xc,
        energy_terms.ion_ion,
        energy_terms.psp_core,
    );

    // Copy ionic potential G-space data for PAW D^hat force
    var result_ionic_g: ?[]math.Complex = null;
    if (common.is_paw) {
        const ionic_vals = common.ionic.values;
        result_ionic_g = try alloc.alloc(math.Complex, ionic_vals.len);
        @memcpy(result_ionic_g.?, ionic_vals);
    }

    return buildScfResult(.{
        .potential = potential,
        .density = rho,
        .iterations = iterations,
        .converged = converged,
        .energy = energy_terms,
        .fermi_level = last_fermi_level,
        .potential_residual = last_potential_residual,
        .wavefunctions = wavefunctions,
        .vresid = vresid_last,
        .grid = grid,
        .kpoint_cache = kpoint_cache,
        .apply_caches = apply_caches,
        .vxc_r = vxc_r,
        .paw_tabs = result_paw_tabs,
        .paw_dij = result_paw_dij,
        .paw_dij_m = result_paw_dij_m,
        .paw_dxc = result_paw_dxc,
        .paw_rhoij = result_paw_rhoij,
        .ionic_g = result_ionic_g,
        .rho_core_copy = try copyRhoCoreForResult(alloc, common.rho_core),
    });
}

pub const buildAtomicDensity = core_density.buildAtomicDensity;

const WavefunctionResult = struct {
    wavefunctions: WavefunctionData,
    band_energy: f64,
    nonlocal_energy: f64,
};

/// Compute final wavefunctions for force calculation.
pub fn computeFinalWavefunctionsWithSpinFactor(
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
    const nelec = totalElectrons(species, atoms);
    const nocc = @as(usize, @intFromFloat(std.math.ceil(nelec / 2.0)));
    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, grid.cell);
    const is_paw_wf = hasPaw(species);
    const has_qij = hasQij(species) and !is_paw_wf;
    // auto: let kpoints.zig decide based on basis size (iterative for large, dense for small)
    const use_iterative_config = (cfg.scf.solver == .iterative or
        cfg.scf.solver == .cg or
        cfg.scf.solver == .auto) and !has_qij;
    const nonlocal_enabled = cfg.scf.enable_nonlocal and hasNonlocal(species);
    const fft_index_map = try buildFftIndexMap(alloc, grid);
    defer alloc.free(fft_index_map);

    var local_r: ?[]f64 = null;
    if (use_iterative_config) {
        local_r = try potential_mod.buildLocalPotentialReal(alloc, grid, ionic, potential);
    }
    defer if (local_r) |values| alloc.free(values);

    if (cfg.scf.debug_fermi) {
        if (local_r) |values| {
            var sum: f64 = 0.0;
            for (values) |v| {
                sum += v;
            }
            const mean_local = sum / @as(f64, @floatFromInt(values.len));
            const pot_g0 = potential.valueAt(0, 0, 0);
            try logLocalPotentialMean(io, "scf", mean_local, "pot_g0", pot_g0.r);
        }
    }

    const iter_max_iter = cfg.scf.iterative_max_iter;
    const iter_tol = cfg.scf.iterative_tol;

    var kp_wavefunctions = try alloc.alloc(KpointWavefunction, kpoints.len);
    errdefer {
        for (kp_wavefunctions) |*kw| {
            kw.deinit(alloc);
        }
        alloc.free(kp_wavefunctions);
    }

    // Pre-create shared FFT plan for final wavefunction computation
    var wf_fft_plan = try fft.Fft3dPlan.initWithBackend(
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

    var filled: usize = 0;
    for (kpoints, 0..) |kp, kidx| {
        const eigen_data = try computeKpointEigenData(
            alloc,
            io,
            &cfg,
            grid,
            kp,
            species,
            atoms,
            recip,
            volume,
            local_cfg,
            potential,
            local_r,
            nocc,
            use_iterative_config,
            has_qij,
            nonlocal_enabled,
            fft_index_map,
            iter_max_iter,
            iter_tol,
            cfg.scf.iterative_reuse_vectors,
            &kpoint_cache[kidx],
            null,
            wf_fft_plan,
            &apply_caches[kidx],
            radial_tables,
            paw_tabs,
        );
        errdefer {
            var ed = eigen_data;
            ed.deinit(alloc);
        }

        // Allocate occupations array (per-spin occupation, scf uses spin_factor later)
        const occupations = try alloc.alloc(f64, eigen_data.nbands);
        errdefer alloc.free(occupations);

        @memset(occupations, 0.0);

        // Set occupations for occupied bands and accumulate energies
        var band: usize = 0;
        while (band < @min(nocc, eigen_data.nbands)) : (band += 1) {
            occupations[band] = 1.0;
            band_energy += kp.weight * spin_factor * eigen_data.values[band];
            if (eigen_data.nonlocal) |nl| {
                nonlocal_energy += kp.weight * spin_factor * nl[band];
            }
        }

        kp_wavefunctions[kidx] = .{
            .k_frac = kp.k_frac,
            .k_cart = kp.k_cart,
            .weight = kp.weight,
            .basis_len = eigen_data.basis_len,
            .nbands = eigen_data.nbands,
            .eigenvalues = eigen_data.values,
            .coefficients = eigen_data.vectors,
            .occupations = occupations,
        };
        // Don't free eigen_data.values and vectors as they're now owned by kp_wavefunctions
        if (eigen_data.nonlocal) |nl| alloc.free(nl);
        filled += 1;
    }

    // Find Fermi level (simple estimate for metallic case)
    var fermi_level: f64 = -std.math.inf(f64);
    for (kp_wavefunctions) |kw| {
        for (kw.eigenvalues, 0..) |e, band| {
            if (kw.occupations[band] > 0.0) {
                fermi_level = @max(fermi_level, e);
            }
        }
    }

    return WavefunctionResult{
        .wavefunctions = WavefunctionData{
            .kpoints = kp_wavefunctions,
            .ecut_ry = cfg.scf.ecut_ry,
            .fermi_level = fermi_level,
        },
        .band_energy = band_energy,
        .nonlocal_energy = nonlocal_energy,
    };
}

/// Compute density from Kohn-Sham eigenvectors.
const ScfLoopProfile = logging.ScfLoopProfile;

fn computeDensity(
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
    const nelec = totalElectrons(species, atoms);
    const nocc = @as(usize, @intFromFloat(std.math.ceil(nelec / 2.0)));
    const is_paw = hasPaw(species);

    // Reset rhoij before accumulation
    if (paw_rhoij) |rij| rij.reset();
    const has_qij = hasQij(species) and !is_paw; // PAW handles overlap via apply_s
    // auto: let kpoints.zig decide based on basis size (iterative for large, dense for small)
    const use_iterative_config = (cfg.scf.solver == .iterative or
        cfg.scf.solver == .cg or
        cfg.scf.solver == .auto) and !has_qij;

    if ((cfg.scf.solver == .iterative or
        cfg.scf.solver == .cg or
        cfg.scf.solver == .auto) and !use_iterative_config)
    {
        if (has_qij) {
            try logIterativeSolverDisabled(io, "QIJ present");
        }
    }

    var local_r: ?[]f64 = null;
    if (use_iterative_config) {
        const t_lr = if (loop_profile != null) profileStart(io) else null;
        local_r = try potential_mod.buildLocalPotentialReal(alloc, grid, ionic, potential);
        if (loop_profile) |lp| profileAdd(io, lp.build_local_r_ns, t_lr);
    }
    defer if (local_r) |values| alloc.free(values);

    const nonlocal_enabled = cfg.scf.enable_nonlocal and hasNonlocal(species);
    // FFT index map is always available now (Bluestein supports arbitrary sizes)
    const t_fm = if (loop_profile != null) profileStart(io) else null;
    const fft_index_map = try buildFftIndexMap(alloc, grid);
    if (loop_profile) |lp| profileAdd(io, lp.build_fft_map_ns, t_fm);
    defer alloc.free(fft_index_map);

    var iter_max_iter = cfg.scf.iterative_max_iter;
    var iter_tol = cfg.scf.iterative_tol;
    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, grid.cell);
    if (cfg.scf.iterative_warmup_steps > 0 and scf_iter < cfg.scf.iterative_warmup_steps) {
        iter_max_iter = cfg.scf.iterative_warmup_max_iter;
        iter_tol = cfg.scf.iterative_warmup_tol;
    }

    if (cfg.scf.profile and scf_iter == 0 and kpoints.len > 0) {
        {
            var check_local = local_r;
            var check_allocated = false;
            if (check_local == null) {
                check_local = try potential_mod.buildLocalPotentialReal(
                    alloc,
                    grid,
                    ionic,
                    potential,
                );
                check_allocated = true;
            }
            defer if (check_allocated) {
                if (check_local) |values| alloc.free(values);
            };

            var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kpoints[0].k_cart);
            defer basis.deinit(alloc);

            const inv_volume = 1.0 / volume;
            try checkHamiltonianApply(
                alloc,
                io,
                grid,
                basis.gvecs,
                species,
                atoms,
                inv_volume,
                potential,
                check_local.?,
                nonlocal_enabled,
                fft_index_map,
            );
        }
    }

    if ((cfg.scf.debug_nonlocal or cfg.scf.debug_local) and scf_iter == 0 and kpoints.len > 0) {
        var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kpoints[0].k_cart);
        defer basis.deinit(alloc);

        const inv_volume = 1.0 / volume;
        if (cfg.scf.debug_local) {
            try logLocalDiagnostics(io, basis.gvecs, species, atoms, local_cfg);
        }
        if (cfg.scf.debug_nonlocal) {
            try logNonlocalDiagnostics(alloc, io, basis.gvecs, species, atoms, inv_volume);
        }
    }

    const cfg_ptr = &cfg;
    if (smearingActive(cfg_ptr)) {
        return try computeDensitySmearing(
            alloc,
            io,
            cfg_ptr,
            grid,
            kpoints,
            species,
            atoms,
            recip,
            volume,
            potential,
            local_r,
            nocc,
            nelec,
            use_iterative_config,
            has_qij,
            nonlocal_enabled,
            fft_index_map,
            iter_max_iter,
            iter_tol,
            kpoint_cache,
            apply_caches,
            radial_tables,
            paw_tabs,
            paw_rhoij,
        );
    }

    const ngrid = grid.count();
    const rho = try alloc.alloc(f64, ngrid);
    errdefer alloc.free(rho);

    @memset(rho, 0.0);

    var band_energy: f64 = 0.0;
    var nonlocal_energy: f64 = 0.0;

    var profile_total = ScfProfile{};
    // PAW rhoij accumulation is not thread-safe, force single-threaded when PAW
    const thread_count = if (paw_rhoij != null)
        @as(usize, 1)
    else
        kpointThreadCount(kpoints.len, cfg.scf.kpoint_threads);

    if (thread_count <= 1) {
        // Pre-create shared FFT plan for single-threaded mode to avoid
        // expensive FFTW plan creation for each kpoint
        var shared_fft_plan = try fft.Fft3dPlan.initWithBackend(
            alloc,
            io,
            grid.nx,
            grid.ny,
            grid.nz,
            cfg.scf.fft_backend,
        );
        defer shared_fft_plan.deinit(alloc);

        const profile_ptr: ?*ScfProfile = if (cfg.scf.profile) &profile_total else null;
        for (kpoints, 0..) |kp, kidx| {
            if (!cfg.scf.quiet) {
                try logKpoint(io, kidx, kpoints.len);
            }
            const ac_ptr: ?*apply.KpointApplyCache = if (apply_caches) |acs|
                (if (kidx < acs.len) &acs[kidx] else null)
            else
                null;
            try computeKpointContribution(
                alloc,
                io,
                cfg_ptr,
                grid,
                kp,
                species,
                atoms,
                recip,
                volume,
                local_cfg,
                potential,
                local_r,
                nocc,
                nelec,
                use_iterative_config,
                has_qij,
                nonlocal_enabled,
                fft_index_map,
                iter_max_iter,
                iter_tol,
                cfg.scf.iterative_reuse_vectors,
                &kpoint_cache[kidx],
                rho,
                &band_energy,
                &nonlocal_energy,
                profile_ptr,
                shared_fft_plan,
                ac_ptr,
                radial_tables,
                paw_tabs,
                paw_rhoij,
            );
        }
        if (cfg.scf.profile and !cfg.scf.quiet) {
            try logProfile(io, profile_total, kpoints.len);
        }
        return DensityResult{
            .rho = rho,
            .band_energy = band_energy,
            .nonlocal_energy = nonlocal_energy,
            .fermi_level = std.math.nan(f64),
        };
    }

    const rho_locals = try alloc.alloc(f64, ngrid * thread_count);
    defer alloc.free(rho_locals);

    @memset(rho_locals, 0.0);

    const band_energies = try alloc.alloc(f64, thread_count);
    defer alloc.free(band_energies);

    @memset(band_energies, 0.0);

    const nonlocal_energies = try alloc.alloc(f64, thread_count);
    defer alloc.free(nonlocal_energies);

    @memset(nonlocal_energies, 0.0);

    var profiles: ?[]ScfProfile = null;
    if (cfg.scf.profile) {
        profiles = try alloc.alloc(ScfProfile, thread_count);
        for (profiles.?) |*p| {
            p.* = ScfProfile{};
        }
    }
    defer if (profiles) |p| alloc.free(p);

    // Pre-create FFT plans for each thread to avoid mutex contention
    // This is the key fix for k-point parallelization performance
    const fft_plans = try alloc.alloc(fft.Fft3dPlan, thread_count);
    defer {
        for (fft_plans) |*plan| {
            plan.deinit(alloc);
        }
        alloc.free(fft_plans);
    }

    for (fft_plans) |*plan| {
        plan.* = try fft.Fft3dPlan.initWithBackend(
            alloc,
            io,
            grid.nx,
            grid.ny,
            grid.nz,
            cfg.scf.fft_backend,
        );
    }

    var next_index = std.atomic.Value(usize).init(0);
    var stop = std.atomic.Value(u8).init(0);
    var worker_error: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;

    var shared = KpointShared{
        .io = io,
        .cfg = cfg_ptr,
        .grid = grid,
        .kpoints = kpoints,
        .ionic = ionic,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume = volume,
        .local_cfg = local_cfg,
        .potential = potential,
        .local_r = local_r,
        .nocc = nocc,
        .nelec = nelec,
        .use_iterative_config = use_iterative_config,
        .has_qij = has_qij,
        .nonlocal_enabled = nonlocal_enabled,
        .fft_index_map = fft_index_map,
        .iter_max_iter = iter_max_iter,
        .iter_tol = iter_tol,
        .reuse_vectors = cfg.scf.iterative_reuse_vectors,
        .rho_locals = rho_locals,
        .band_energies = band_energies,
        .nonlocal_energies = nonlocal_energies,
        .profiles = profiles,
        .ngrid = ngrid,
        .kpoint_cache = kpoint_cache,
        .apply_caches = apply_caches,
        .fft_plans = fft_plans,
        .radial_tables = radial_tables,
        .paw_tabs = paw_tabs,
        .next_index = &next_index,
        .stop = &stop,
        .err = &worker_error,
        .err_mutex = &err_mutex,
        .log_mutex = &log_mutex,
    };

    const workers = try alloc.alloc(KpointWorker, thread_count);
    defer alloc.free(workers);

    const threads = try alloc.alloc(std.Thread, thread_count);
    defer alloc.free(threads);

    var t: usize = 0;
    while (t < thread_count) : (t += 1) {
        workers[t] = .{ .shared = &shared, .thread_index = t };
        threads[t] = try std.Thread.spawn(.{}, kpointWorker, .{&workers[t]});
    }
    for (threads) |thread| {
        thread.join();
    }

    if (worker_error) |err| return err;

    t = 0;
    while (t < thread_count) : (t += 1) {
        band_energy += band_energies[t];
        nonlocal_energy += nonlocal_energies[t];
        const start = t * ngrid;
        const end = start + ngrid;
        const local_rho = rho_locals[start..end];
        for (local_rho, 0..) |value, i| {
            rho[i] += value;
        }
        if (profiles) |p| {
            mergeProfile(&profile_total, p[t]);
        }
    }

    if (cfg.scf.profile) {
        if (!cfg.scf.quiet) {
            try logProfile(io, profile_total, kpoints.len);
        }
    }
    return DensityResult{
        .rho = rho,
        .band_energy = band_energy,
        .nonlocal_energy = nonlocal_energy,
        .fermi_level = std.math.nan(f64),
    };
}

const computeDensitySmearing = smearing_mod.computeDensitySmearing;

pub const densityDiff = util.densityDiff;

pub const hasNonlocal = util.hasNonlocal;
pub const hasQij = util.hasQij;
pub const hasPaw = util.hasPaw;
pub const totalElectrons = util.totalElectrons;

test {
    _ = @import("band_solver.zig");
    _ = @import("gvec_iter.zig");
    _ = @import("paw_scf.zig");
    _ = @import("smearing.zig");
}

test "auto grid chooses fft-friendly size for aluminum" {
    const cell_ang = math.Mat3.fromRows(
        .{ .x = 0.0, .y = 2.025, .z = 2.025 },
        .{ .x = 2.025, .y = 0.0, .z = 2.025 },
        .{ .x = 2.025, .y = 2.025, .z = 0.0 },
    );
    const cell_bohr = cell_ang.scale(math.unitsScaleToBohr(.angstrom));
    const recip = math.reciprocal(cell_bohr);
    const grid = grid_mod.autoGrid(15.0, 1.0, recip);
    try std.testing.expect(grid[0] >= 3);
    try std.testing.expect(grid[1] >= 3);
    try std.testing.expect(grid[2] >= 3);
}
