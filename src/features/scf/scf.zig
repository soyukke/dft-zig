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
const form_factor = @import("../pseudopotential/form_factor.zig");
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
const logNonlocalDiagnostics = logging.logNonlocalDiagnostics;
const logLocalDiagnostics = logging.logLocalDiagnostics;
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
                    const mh = op.k_rot.m[0][0] * gh + op.k_rot.m[0][1] * gk + op.k_rot.m[0][2] * gl;
                    const mk = op.k_rot.m[1][0] * gh + op.k_rot.m[1][1] * gk + op.k_rot.m[1][2] * gl;
                    const ml = op.k_rot.m[2][0] * gh + op.k_rot.m[2][1] * gk + op.k_rot.m[2][2] * gl;

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
    cfg: config.Config,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    initial_density: ?[]const f64 = null,
    initial_kpoint_cache: ?[]KpointCache = null,
    initial_apply_caches: ?[]apply.KpointApplyCache = null,
    ff_tables: ?[]const form_factor.LocalFormFactorTable = null,
};

/// Common state shared by both spin-unpolarized and spin-polarized SCF.
pub const ScfCommon = struct {
    alloc: std.mem.Allocator,
    cfg: config.Config,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    grid: Grid,
    total_electrons: f64,
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
fn initScfCommon(params: ScfParams) !ScfCommon {
    const alloc = params.alloc;
    const cfg = params.cfg;
    const species = params.species;
    const atoms = params.atoms;
    const recip = params.recip;
    const volume_bohr = params.volume_bohr;
    const ff_tables = params.ff_tables;

    const grid = grid_mod.gridFromConfig(cfg, recip, volume_bohr);
    const total_electrons = totalElectrons(species, atoms);

    // Compute cutoff radius for isolated systems
    const coulomb_r_cut: ?f64 = if (cfg.boundary == .isolated) coulomb_mod.cutoffRadius(grid.cell) else null;

    const local_alpha = localPotentialAlpha(cfg);
    hamiltonian.configureLocalPotential(species, cfg.scf.local_potential, local_alpha);

    // Compute ecutrho spherical cutoff early so it can be applied to V_local(G)
    const is_paw = hasPaw(species);
    const ecutrho: ?f64 = if (is_paw) blk: {
        const gs_val = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
        break :blk cfg.scf.ecut_ry * gs_val * gs_val;
    } else null;

    var ionic = try potential_mod.buildIonicPotentialGrid(alloc, grid, species, atoms, ff_tables, ecutrho);
    errdefer ionic.deinit(alloc);

    var log = try ScfLog.init(alloc, cfg.out_dir);
    errdefer log.deinit();
    try log.writeHeader();

    // For isolated systems, force Gamma-only k-point sampling
    const kpoints = if (cfg.boundary == .isolated) blk: {
        const gamma_kpoints = try alloc.alloc(KPoint, 1);
        gamma_kpoints[0] = KPoint{
            .k_frac = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .k_cart = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .weight = 1.0,
        };
        break :blk gamma_kpoints;
    } else if (cfg.scf.symmetry)
        try kmesh_mod.generateKmeshSymmetry(
            alloc,
            cfg.scf.kmesh,
            .{ .x = cfg.scf.kmesh_shift[0], .y = cfg.scf.kmesh_shift[1], .z = cfg.scf.kmesh_shift[2] },
            recip,
            grid.cell,
            atoms,
            cfg.scf.time_reversal,
        )
    else
        try kmesh_mod.generateKmesh(
            alloc,
            cfg.scf.kmesh,
            recip,
            .{ .x = cfg.scf.kmesh_shift[0], .y = cfg.scf.kmesh_shift[1], .z = cfg.scf.kmesh_shift[2] },
        );
    errdefer alloc.free(kpoints);

    const sym_ops = if (cfg.scf.symmetry)
        try symmetry.getSymmetryOps(alloc, grid.cell, atoms, 1e-6)
    else
        null;
    errdefer if (sym_ops) |ops| alloc.free(ops);

    var rho_core: ?[]f64 = null;
    if (core_density.hasNlcc(species)) {
        rho_core = try core_density.buildCoreDensity(alloc, grid, species, atoms);
    }
    errdefer if (rho_core) |values| alloc.free(values);

    // Build radial lookup tables for fast NonlocalContext construction
    const nonlocal_enabled_run = cfg.scf.enable_nonlocal and hasNonlocal(species);
    var radial_tables_buf: ?[]nonlocal_mod.RadialTableSet = null;
    if (nonlocal_enabled_run) {
        const g_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 1.5;
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
        radial_tables_buf = buf;
    }
    errdefer {
        if (radial_tables_buf) |buf| {
            for (buf) |*t| {
                if (t.tables.len > 0) t.deinit(alloc);
            }
            alloc.free(buf);
        }
    }
    const radial_tables: ?[]const nonlocal_mod.RadialTableSet = radial_tables_buf;

    // Initialize Pulay mixer if configured
    const pulay_mixer: ?PulayMixer = if (cfg.scf.pulay_history > 0)
        PulayMixer.init(alloc, cfg.scf.pulay_history)
    else
        null;

    // Initialize PAW tables if any species uses PAW
    // (is_paw already determined above for ecutrho)
    var paw_tabs: ?[]paw_mod.PawTab = null;
    var paw_rhoij: ?paw_mod.RhoIJ = null;
    if (is_paw) {
        const q_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 1.5;
        var tabs = try alloc.alloc(paw_mod.PawTab, species.len);
        errdefer {
            for (tabs) |*t| t.deinit(alloc);
            alloc.free(tabs);
        }
        for (species, 0..) |entry, si| {
            if (entry.upf.paw) |paw| {
                tabs[si] = try paw_mod.PawTab.init(alloc, paw, entry.upf.r, entry.upf.rab, q_max);
            } else {
                // Non-PAW species: zero-initialize
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
        paw_tabs = tabs;

        // Initialize m-resolved RhoIJ for all atoms
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
        paw_rhoij = try paw_mod.RhoIJ.init(alloc, natom, nbeta_list, l_lists);
    }

    // Initialize Gaunt table for multi-L PAW
    var paw_gaunt: ?paw_mod.GauntTable = null;
    if (paw_tabs) |tabs| {
        // Determine lmax_proj and lmax_aug from PAW tabs
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
        paw_gaunt = try paw_mod.GauntTable.init(alloc, lmax_proj, lmax_aug);
    }

    errdefer {
        if (paw_tabs) |tabs| {
            for (tabs) |*t| t.deinit(alloc);
            alloc.free(tabs);
        }
        if (paw_rhoij) |*rij| rij.deinit(alloc);
        if (paw_gaunt) |*gt| gt.deinit(alloc);
    }

    return ScfCommon{
        .alloc = alloc,
        .cfg = cfg,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume_bohr = volume_bohr,
        .grid = grid,
        .total_electrons = total_electrons,
        .ionic = ionic,
        .log = log,
        .kpoints = kpoints,
        .sym_ops = sym_ops,
        .rho_core = rho_core,
        .radial_tables_buf = radial_tables_buf,
        .radial_tables = radial_tables,
        .pulay_mixer = pulay_mixer,
        .coulomb_r_cut = coulomb_r_cut,
        .paw_tabs = paw_tabs,
        .paw_rhoij = paw_rhoij,
        .paw_gaunt = paw_gaunt,
        .is_paw = is_paw,
    };
}

/// Run SCF loop to build Hartree+XC potential.
pub fn run(params: ScfParams) !ScfResult {
    const alloc = params.alloc;
    const cfg = params.cfg;
    const species = params.species;
    const atoms = params.atoms;
    const recip = params.recip;
    const volume_bohr = params.volume_bohr;
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
        return scf_spin.runSpinPolarizedLoop(alloc, cfg, species, atoms, volume_bohr, &common);
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
    // Warmstart: copy initial eigenvectors from previous SCF run
    if (initial_kpoint_cache) |init_cache| {
        const copy_len = @min(kpoint_cache.len, init_cache.len);
        for (0..copy_len) |k| {
            if (init_cache[k].vectors.len > 0) {
                try kpoint_cache[k].store(init_cache[k].n, init_cache[k].nbands, init_cache[k].vectors);
            }
        }
    }

    const grid_count = grid.count();
    const rho = try alloc.alloc(f64, grid_count);
    errdefer alloc.free(rho);

    if (initial_density) |init_rho| {
        if (init_rho.len == grid_count) {
            @memcpy(rho, init_rho);
        } else {
            const rho0 = common.total_electrons / grid.volume;
            @memset(rho, rho0);
        }
    } else {
        const rho0 = common.total_electrons / grid.volume;
        @memset(rho, rho0);
    }

    var iterations: usize = 0;
    var converged = false;
    // ecutrho spherical cutoff for PAW: limit G-space sums to |G|² < ecutrho
    const paw_ecutrho: ?f64 = if (common.is_paw) blk: {
        const gs_val = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
        break :blk cfg.scf.ecut_ry * gs_val * gs_val;
    } else null;
    var potential = try potential_mod.buildPotentialGrid(alloc, grid, rho, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, common.coulomb_r_cut, paw_ecutrho);
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
    const apply_caches = if (initial_apply_caches) |init_caches| init_caches else try alloc.alloc(apply.KpointApplyCache, kpoints.len);
    var apply_caches_owned = true;
    if (initial_apply_caches == null) {
        for (apply_caches) |*ac| ac.* = .{};
    }
    defer {
        if (apply_caches_owned) {
            for (apply_caches) |*ac| ac.deinit(alloc);
            alloc.free(apply_caches);
        }
    }

    // SCF loop profiling accumulators (for unaccounted time analysis)
    var prof_compute_density_ns: u64 = 0;
    var prof_build_potential_ns: u64 = 0;
    var prof_residual_ns: u64 = 0;
    var prof_mixing_ns: u64 = 0;
    var prof_build_local_r_ns: u64 = 0;
    var prof_build_fft_map_ns: u64 = 0;

    while (iterations < cfg.scf.max_iter) : (iterations += 1) {
        if (!cfg.scf.quiet) {
            try logIterStart(iterations);
        }
        const t_density_start = if (cfg.scf.profile) profileStart() else null;
        const density_result = try computeDensity(
            alloc,
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
                .build_local_r_ns = &prof_build_local_r_ns,
                .build_fft_map_ns = &prof_build_fft_map_ns,
            } else null,
            common.paw_tabs,
            if (common.paw_rhoij) |*rij| rij else null,
        );
        if (cfg.scf.profile) profileAdd(&prof_compute_density_ns, t_density_start);
        defer alloc.free(density_result.rho);
        last_band_energy = density_result.band_energy;
        last_nonlocal_energy = density_result.nonlocal_energy;
        last_entropy_energy = density_result.entropy_energy;
        last_fermi_level = density_result.fermi_level;

        if (common.sym_ops) |ops| {
            if (ops.len > 1) {
                try symmetrizeDensity(alloc, grid, density_result.rho, ops, cfg.scf.use_rfft);
            }
        }

        // PAW: symmetrize rhoij between equivalent atoms of the same species.
        // LOBPCG eigenvectors for degenerate states may break crystal symmetry,
        // leading to different rhoij for symmetry-equivalent atoms.  Averaging
        // restores the correct symmetry and prevents spurious band splitting.
        // Skip when symmetry is disabled (e.g. during relaxation) to preserve
        // per-atom rhoij differences needed for accurate force calculation.
        if (common.paw_rhoij) |*prij| {
            if (cfg.scf.symmetry) {
                try paw_scf.symmetrizeRhoIJ(alloc, prij, species, atoms);
            }
        }

        // PAW: filter density to ecutrho sphere — matches QE's convention.
        // QE stores density as G-vectors inside |G|²<ecutrho sphere (10777 for Si 32³).
        // Without filtering, high-G components from the cubic grid enter V_xc(r),
        // causing a different SCF solution.
        const gs_comp = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
        const ecutrho_comp = cfg.scf.ecut_ry * gs_comp * gs_comp;
        if (common.is_paw) {
            const filtered = try potential_mod.filterDensityToEcutrho(alloc, grid, density_result.rho, ecutrho_comp, cfg.scf.use_rfft);
            defer alloc.free(filtered);
            @memcpy(density_result.rho, filtered);
        }

        // PAW: build augmented density (ρ̃ + n_hat) for potential construction
        const rho_for_potential = if (common.is_paw) blk: {
            const aug = try alloc.alloc(f64, grid_count);
            @memcpy(aug, density_result.rho);
            if (common.paw_rhoij) |*prij| {
                try paw_scf.addPawCompensationCharge(alloc, grid, aug, prij, common.paw_tabs.?, atoms, ecutrho_comp, &common.paw_gaunt.?);
            }
            break :blk aug;
        } else density_result.rho;
        defer if (common.is_paw) alloc.free(rho_for_potential);

        // Capture V_xc(r) for NLCC force calculation when relax is enabled
        if (vxc_r) |old| alloc.free(old);
        vxc_r = null;
        const vxc_r_ptr: ?*?[]f64 = if (cfg.relax.enabled) &vxc_r else null;
        const t_build_pot_start = if (cfg.scf.profile) profileStart() else null;
        var potential_out = try potential_mod.buildPotentialGrid(alloc, grid, rho_for_potential, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, vxc_r_ptr, common.coulomb_r_cut, paw_ecutrho);
        if (cfg.scf.profile) profileAdd(&prof_build_potential_ns, t_build_pot_start);
        var keep_potential_out = false;
        defer if (!keep_potential_out) potential_out.deinit(alloc);

        {
            const t_resid_start = if (cfg.scf.profile) profileStart() else null;
            const nvals = potential.values.len;
            var residual_values = try alloc.alloc(math.Complex, nvals);
            errdefer alloc.free(residual_values);
            var sum_sq: f64 = 0.0;
            for (0..nvals) |idx| {
                const diff = math.complex.sub(potential_out.values[idx], potential.values[idx]);
                residual_values[idx] = diff;
                sum_sq += diff.r * diff.r + diff.i * diff.i;
            }
            last_potential_residual = if (nvals > 0)
                std.math.sqrt(sum_sq / @as(f64, @floatFromInt(nvals)))
            else
                0.0;
            if (vresid_last) |*old| {
                old.deinit(alloc);
            }
            vresid_last = hamiltonian.PotentialGrid{
                .nx = grid.nx,
                .ny = grid.ny,
                .nz = grid.nz,
                .min_h = grid.min_h,
                .min_k = grid.min_k,
                .min_l = grid.min_l,
                .values = residual_values,
            };
            if (cfg.scf.profile) profileAdd(&prof_residual_ns, t_resid_start);
        }

        const diff = densityDiff(rho, density_result.rho);
        const conv_value = switch (cfg.scf.convergence_metric) {
            .density => diff,
            .potential => last_potential_residual,
        };
        try common.log.writeIter(iterations, diff, last_potential_residual, last_band_energy, last_nonlocal_energy);
        if (!cfg.scf.quiet) {
            try logProgress(iterations, diff, last_potential_residual, last_band_energy, last_nonlocal_energy);
        }

        if (conv_value < cfg.scf.convergence) {
            converged = true;
            @memcpy(rho, density_result.rho);
            potential.deinit(alloc);
            potential = potential_out;
            keep_potential_out = true;
            break;
        }

        const t_mix_start = if (cfg.scf.profile) profileStart() else null;
        if (cfg.scf.mixing_mode == .potential) {
            // Potential mixing: mix V_in and V_out directly (like ABINIT iscf=7)
            const n_complex = potential.values.len;
            const n_f64 = n_complex * 2;
            const v_in: []f64 = @as([*]f64, @ptrCast(potential.values.ptr))[0..n_f64];

            if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
                // Compute complex residual R(G) = V_out(G) - V_in(G)
                const residual_g = try alloc.alloc(math.Complex, n_complex);
                // Ownership transfers to mixer via mixWithResidual — no defer free.
                // Safety: Complex{r:f64,i:f64} has same size/align as [2]f64,
                // so alloc(Complex, n) can be freed as alloc.free([]f64) of len n*2.
                for (0..n_complex) |idx| {
                    residual_g[idx] = math.complex.sub(potential_out.values[idx], potential.values[idx]);
                }

                // Apply model dielectric preconditioner if enabled
                if (cfg.scf.diemac > 1.0) {
                    mixing.applyModelDielectricPreconditioner(grid, residual_g, cfg.scf.diemac, cfg.scf.dielng);
                }

                const precond_f64: []f64 = @as([*]f64, @ptrCast(residual_g.ptr))[0..n_f64];
                try common.pulay_mixer.?.mixWithResidual(v_in, precond_f64, cfg.scf.mixing_beta);
            } else {
                const v_out: []const f64 = @as([*]const f64, @ptrCast(potential_out.values.ptr))[0..n_f64];
                mixDensity(v_in, v_out, cfg.scf.mixing_beta);
            }

            // Update density from output (for convergence tracking and energy computation)
            @memcpy(rho, density_result.rho);
            // potential.values now contains the mixed potential in-place
            potential_out.deinit(alloc);
            keep_potential_out = true; // prevent double-free of potential_out
        } else {
            // Density mixing: mix rho_in and rho_out, then rebuild potential
            if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
                if (cfg.scf.kerker_q0 > 0.0) {
                    try common.pulay_mixer.?.mixKerkerPulay(rho, density_result.rho, cfg.scf.mixing_beta, grid, cfg.scf.kerker_q0, cfg.scf.use_rfft);
                } else {
                    try common.pulay_mixer.?.mix(rho, density_result.rho, cfg.scf.mixing_beta);
                }
            } else if (cfg.scf.kerker_q0 > 0.0) {
                try mixDensityKerker(alloc, grid, rho, density_result.rho, cfg.scf.mixing_beta, cfg.scf.kerker_q0, cfg.scf.use_rfft);
            } else {
                mixDensity(rho, density_result.rho, cfg.scf.mixing_beta);
            }

            potential.deinit(alloc);
            potential = try potential_mod.buildPotentialGrid(alloc, grid, rho, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, common.coulomb_r_cut, paw_ecutrho);
        }
        if (cfg.scf.profile) profileAdd(&prof_mixing_ns, t_mix_start);

        // PAW: update D_ij from the current mixed potential
        if (common.is_paw) {
            if (common.paw_tabs) |tabs| {
                const gs_paw = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
                const ecutrho_paw = cfg.scf.ecut_ry * gs_paw * gs_paw;
                try paw_scf.updatePawDij(alloc, grid, common.ionic, potential, tabs, species, atoms, apply_caches, ecutrho_paw, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, null, 1.0);
            }
        }
    }

    // Print SCF loop profile
    if (cfg.scf.profile and !cfg.scf.quiet) {
        const to_ms = 1.0 / @as(f64, @floatFromInt(std.time.ns_per_ms));
        var buffer2: [512]u8 = undefined;
        var writer2 = std.Io.File.stderr().writer(&buffer2);
        const out2 = &writer2.interface;
        try out2.print(
            "scf_loop_profile compute_density_ms={d:.3} build_potential_ms={d:.3} residual_ms={d:.3} mixing_ms={d:.3} build_local_r_ms={d:.3} build_fft_map_ms={d:.3}\n",
            .{
                @as(f64, @floatFromInt(prof_compute_density_ns)) * to_ms,
                @as(f64, @floatFromInt(prof_build_potential_ns)) * to_ms,
                @as(f64, @floatFromInt(prof_residual_ns)) * to_ms,
                @as(f64, @floatFromInt(prof_mixing_ns)) * to_ms,
                @as(f64, @floatFromInt(prof_build_local_r_ns)) * to_ms,
                @as(f64, @floatFromInt(prof_build_fft_map_ns)) * to_ms,
            },
        );
        try out2.flush();
    }

    // For PAW: build augmented density ρ̃ + n̂hat for energy computation.
    // E_H and E_xc must use the augmented density to be variationally consistent
    // with the potential used during SCF (which was built from augmented density).
    // Filter to ecutrho sphere to match QE convention (cube corners excluded).
    var rho_aug_for_energy: ?[]f64 = null;
    if (common.is_paw) {
        if (common.paw_rhoij) |*prij| {
            const aug = try alloc.alloc(f64, grid.count());
            @memcpy(aug, rho);
            const gs_en = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
            const ecutrho_scf = cfg.scf.ecut_ry * gs_en * gs_en;
            try paw_scf.addPawCompensationCharge(alloc, grid, aug, prij, common.paw_tabs.?, atoms, ecutrho_scf, &common.paw_gaunt.?);
            // Filter augmented density to ecutrho sphere for E_xc consistency
            const filtered = try potential_mod.filterDensityToEcutrho(alloc, grid, aug, ecutrho_scf, cfg.scf.use_rfft);
            alloc.free(aug);
            rho_aug_for_energy = filtered;
        }
    }
    defer if (rho_aug_for_energy) |a| alloc.free(a);

    var energy_terms = try energy_mod.computeEnergyTerms(
        alloc,
        grid,
        rho,
        common.rho_core,
        last_band_energy,
        last_nonlocal_energy,
        last_entropy_energy,
        species,
        atoms,
        cfg.ewald,
        cfg.scf.use_rfft,
        cfg.scf.xc,
        cfg.scf.quiet,
        common.coulomb_r_cut,
        cfg.vdw,
        rho_aug_for_energy,
        paw_ecutrho,
    );

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
        // Transfer paw_tabs ownership from common to result
        if (common.paw_tabs) |tabs| {
            result_paw_tabs = tabs;
            common.paw_tabs = null; // prevent common.deinit from freeing
        }
        // Extract per-atom D_ij from first apply cache
        if (apply_caches.len > 0) {
            if (apply_caches[0].nonlocal_ctx) |nc| {
                var dij_list: std.ArrayList([]f64) = .empty;
                errdefer {
                    for (dij_list.items) |d| alloc.free(d);
                    dij_list.deinit(alloc);
                }
                for (nc.species) |entry| {
                    if (entry.dij_per_atom) |dpa| {
                        for (dpa) |atom_dij| {
                            const copy = try alloc.alloc(f64, atom_dij.len);
                            @memcpy(copy, atom_dij);
                            try dij_list.append(alloc, copy);
                        }
                    }
                }
                if (dij_list.items.len > 0) {
                    result_paw_dij = try dij_list.toOwnedSlice(alloc);
                } else {
                    dij_list.deinit(alloc);
                }
                // Extract m-resolved D_ij for stress
                var dij_m_list: std.ArrayList([]f64) = .empty;
                errdefer {
                    for (dij_m_list.items) |d| alloc.free(d);
                    dij_m_list.deinit(alloc);
                }
                for (nc.species) |entry| {
                    if (entry.dij_m_per_atom) |dpa| {
                        for (dpa) |atom_dij_m| {
                            const copy = try alloc.alloc(f64, atom_dij_m.len);
                            @memcpy(copy, atom_dij_m);
                            try dij_m_list.append(alloc, copy);
                        }
                    }
                }
                if (dij_m_list.items.len > 0) {
                    result_paw_dij_m = try dij_m_list.toOwnedSlice(alloc);
                } else {
                    dij_m_list.deinit(alloc);
                }
            }
        }
        // Compute per-atom D^xc for on-site stress
        if (common.paw_rhoij) |prij| {
            if (result_paw_tabs) |tabs| {
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
                        const zero = try alloc.alloc(f64, 0);
                        try dxc_list.append(alloc, zero);
                        continue;
                    };
                    if (si >= tabs.len or tabs[si].nbeta == 0) {
                        const zero = try alloc.alloc(f64, 0);
                        try dxc_list.append(alloc, zero);
                        continue;
                    }
                    const mt = prij.m_total_per_atom[ai];
                    const sp_m_offsets = prij.m_offsets[ai];

                    // D^xc (m-resolved, angular Lebedev quadrature with GGA gradients)
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
                        &common.paw_gaunt.?,
                    );
                    // m-resolved double-counting: DC_xc = -Σ_{im,jm} D^xc_m × ρ_m
                    const rhoij_m = prij.values[ai];
                    for (0..mt) |im| {
                        for (0..mt) |jm| {
                            sum_dxc_rhoij += dxc_m[im * mt + jm] * rhoij_m[im * mt + jm];
                        }
                    }
                    try dxc_list.append(alloc, dxc_m);

                    // D^H double-counting (m-resolved, multi-L with Gaunt)
                    {
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
                            &common.paw_gaunt.?,
                        );
                        for (0..mt) |im| {
                            for (0..mt) |jm| {
                                sum_dxc_rhoij += dij_h_dc[im * mt + jm] * rhoij_m[im * mt + jm];
                            }
                        }
                    }
                }
                if (dxc_list.items.len > 0) {
                    result_paw_dxc = try dxc_list.toOwnedSlice(alloc);
                } else {
                    dxc_list.deinit(alloc);
                }
                // PAW double-counting correction: subtract Σ (D^xc + D^H) × ρ_ij from total energy.
                // D^xc and D^H both use m-resolved sums for Hellmann-Feynman consistency.
                energy_terms.paw_dxc_rhoij = -sum_dxc_rhoij;
                energy_terms.total += energy_terms.paw_dxc_rhoij;
            }
        }
        // Copy per-atom rhoij (contracted to radial basis) for force calculation
        if (common.paw_rhoij) |*prij| {
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
            if (rij_list.items.len > 0) {
                result_paw_rhoij = try rij_list.toOwnedSlice(alloc);
            } else {
                rij_list.deinit(alloc);
            }
        }
    }

    if (!cfg.scf.quiet) {
        std.debug.print("scf: electron_count {d:.6}\n", .{totalCharge(rho, grid)});
        const ionic_g0 = common.ionic.valueAt(0, 0, 0);
        const pot_g0 = potential.valueAt(0, 0, 0);
        std.debug.print("scf: ionic_g0 {d:.6} {d:.6}\n", .{ ionic_g0.r, ionic_g0.i });
        std.debug.print("scf: hartree_xc_g0 {d:.6} {d:.6}\n", .{ pot_g0.r, pot_g0.i });
        std.debug.print("scf: E_band={d:.8} E_H={d:.8} E_xc={d:.8} E_ion={d:.8}\n", .{ energy_terms.band, energy_terms.hartree, energy_terms.xc, energy_terms.ion_ion });
        std.debug.print("scf: E_psp={d:.8} E_dc={d:.8} E_local={d:.8} E_nl={d:.8}\n", .{ energy_terms.psp_core, energy_terms.double_counting, energy_terms.local_pseudo, energy_terms.nonlocal_pseudo });
        std.debug.print("scf: E_paw_onsite={d:.8} E_paw_dxc={d:.8} E_total={d:.8}\n", .{ energy_terms.paw_onsite, energy_terms.paw_dxc_rhoij, energy_terms.total });
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

    return ScfResult{
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
        .rho_core = if (common.rho_core) |rc| blk: {
            const copy = try alloc.alloc(f64, rc.len);
            @memcpy(copy, rc);
            break :blk copy;
        } else null,
    };
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
    cfg: config.Config,
    grid: Grid,
    kpoints: []KPoint,
    ionic: hamiltonian.PotentialGrid,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
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
    const is_paw_wf = hasPaw(species);
    const has_qij = hasQij(species) and !is_paw_wf;
    // auto: let kpoints.zig decide based on basis size (iterative for large, dense for small)
    const use_iterative_config = (cfg.scf.solver == .iterative or cfg.scf.solver == .cg or cfg.scf.solver == .auto) and !has_qij;
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
            var buffer: [256]u8 = undefined;
            var writer = std.Io.File.stderr().writer(&buffer);
            const out = &writer.interface;
            try out.print(
                "scf: local_r mean={d:.6} pot_g0={d:.6}\n",
                .{ mean_local, pot_g0.r },
            );
            try out.flush();
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
    var wf_fft_plan = try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend);
    defer wf_fft_plan.deinit(alloc);

    var band_energy: f64 = 0.0;
    var nonlocal_energy: f64 = 0.0;

    var filled: usize = 0;
    for (kpoints, 0..) |kp, kidx| {
        const eigen_data = try computeKpointEigenData(
            alloc,
            &cfg,
            grid,
            kp,
            species,
            atoms,
            recip,
            volume,
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
    cfg: config.Config,
    grid: Grid,
    kpoints: []KPoint,
    ionic: hamiltonian.PotentialGrid,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
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
    const use_iterative_config = (cfg.scf.solver == .iterative or cfg.scf.solver == .cg or cfg.scf.solver == .auto) and !has_qij;

    if ((cfg.scf.solver == .iterative or cfg.scf.solver == .cg or cfg.scf.solver == .auto) and !use_iterative_config) {
        var buffer: [256]u8 = undefined;
        var writer = std.Io.File.stderr().writer(&buffer);
        const out = &writer.interface;
        if (has_qij) {
            try out.writeAll("scf: iterative solver disabled (QIJ present)\n");
        }
        try out.flush();
    }

    var local_r: ?[]f64 = null;
    if (use_iterative_config) {
        const t_lr = if (loop_profile != null) profileStart() else null;
        local_r = try potential_mod.buildLocalPotentialReal(alloc, grid, ionic, potential);
        if (loop_profile) |lp| profileAdd(lp.build_local_r_ns, t_lr);
    }
    defer if (local_r) |values| alloc.free(values);

    const nonlocal_enabled = cfg.scf.enable_nonlocal and hasNonlocal(species);
    // FFT index map is always available now (Bluestein supports arbitrary sizes)
    const t_fm = if (loop_profile != null) profileStart() else null;
    const fft_index_map = try buildFftIndexMap(alloc, grid);
    if (loop_profile) |lp| profileAdd(lp.build_fft_map_ns, t_fm);
    defer alloc.free(fft_index_map);

    var iter_max_iter = cfg.scf.iterative_max_iter;
    var iter_tol = cfg.scf.iterative_tol;
    if (cfg.scf.iterative_warmup_steps > 0 and scf_iter < cfg.scf.iterative_warmup_steps) {
        iter_max_iter = cfg.scf.iterative_warmup_max_iter;
        iter_tol = cfg.scf.iterative_warmup_tol;
    }

    if (cfg.scf.profile and scf_iter == 0 and kpoints.len > 0) {
        {
            var check_local = local_r;
            var check_allocated = false;
            if (check_local == null) {
                check_local = try potential_mod.buildLocalPotentialReal(alloc, grid, ionic, potential);
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
            try logLocalDiagnostics(basis.gvecs, species, atoms);
        }
        if (cfg.scf.debug_nonlocal) {
            try logNonlocalDiagnostics(alloc, basis.gvecs, species, atoms, inv_volume);
        }
    }

    const cfg_ptr = &cfg;
    if (smearingActive(cfg_ptr)) {
        return try computeDensitySmearing(
            alloc,
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
    const thread_count = if (paw_rhoij != null) @as(usize, 1) else kpointThreadCount(kpoints.len, cfg.scf.kpoint_threads);

    if (thread_count <= 1) {
        // Pre-create shared FFT plan for single-threaded mode to avoid
        // expensive FFTW plan creation for each kpoint
        var shared_fft_plan = try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend);
        defer shared_fft_plan.deinit(alloc);

        const profile_ptr: ?*ScfProfile = if (cfg.scf.profile) &profile_total else null;
        for (kpoints, 0..) |kp, kidx| {
            if (!cfg.scf.quiet) {
                try logKpoint(kidx, kpoints.len);
            }
            const ac_ptr: ?*apply.KpointApplyCache = if (apply_caches) |acs|
                (if (kidx < acs.len) &acs[kidx] else null)
            else
                null;
            try computeKpointContribution(
                alloc,
                cfg_ptr,
                grid,
                kp,
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
            try logProfile(profile_total, kpoints.len);
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
        plan.* = try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend);
    }

    var next_index = std.atomic.Value(usize).init(0);
    var stop = std.atomic.Value(u8).init(0);
    var worker_error: ?anyerror = null;
    var err_mutex = std.Thread.Mutex{};
    var log_mutex = std.Thread.Mutex{};

    var shared = KpointShared{
        .cfg = cfg_ptr,
        .grid = grid,
        .kpoints = kpoints,
        .ionic = ionic,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume = volume,
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
            try logProfile(profile_total, kpoints.len);
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

fn localPotentialAlpha(cfg: config.Config) f64 {
    if (cfg.scf.local_potential != .ewald) return 0.0;
    if (cfg.ewald.alpha > 0.0) return cfg.ewald.alpha;
    const cell_bohr = cfg.cell.scale(math.unitsScaleToBohr(cfg.units));
    const lmin = @min(
        @min(math.Vec3.norm(cell_bohr.row(0)), math.Vec3.norm(cell_bohr.row(1))),
        math.Vec3.norm(cell_bohr.row(2)),
    );
    return 5.0 / lmin;
}

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
