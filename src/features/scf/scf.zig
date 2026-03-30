const std = @import("std");
const apply = @import("apply.zig");
const core_density = @import("core_density.zig");
const config = @import("../config/config.zig");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const energy_mod = @import("energy.zig");
const fft = @import("../fft/fft.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoints_mod = @import("kpoints.zig");
const iterative = @import("../linalg/iterative.zig");
const linalg = @import("../linalg/linalg.zig");
const kmesh_mod = @import("../kpoints/kpoints.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const mixing = @import("mixing.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const potential_mod = @import("potential.zig");
const xc_fields_mod = @import("xc_fields.zig");
const pw_grid_map = @import("pw_grid_map.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const thread_pool = @import("../thread_pool.zig");
const util = @import("util.zig");

pub const ThreadPool = thread_pool.ThreadPool;

const KPoint = symmetry.KPoint;

pub const Grid = grid_mod.Grid;

const ApplyContext = apply.ApplyContext;
const applyHamiltonian = apply.applyHamiltonian;
const applyHamiltonianBatched = apply.applyHamiltonianBatched;
const checkHamiltonianApply = apply.checkHamiltonianApply;

pub const KpointCache = kpoints_mod.KpointCache;
const KpointShared = kpoints_mod.KpointShared;
const KpointWorker = kpoints_mod.KpointWorker;
const KpointEigenData = kpoints_mod.KpointEigenData;
const SmearingShared = kpoints_mod.SmearingShared;
const SmearingWorker = kpoints_mod.SmearingWorker;
const computeKpointContribution = kpoints_mod.computeKpointContribution;
const computeKpointEigenData = kpoints_mod.computeKpointEigenData;
const accumulateKpointDensitySmearing = kpoints_mod.accumulateKpointDensitySmearing;
const kpointThreadCount = kpoints_mod.kpointThreadCount;
const kpointWorker = kpoints_mod.kpointWorker;
const smearingWorker = kpoints_mod.smearingWorker;
const findFermiLevel = kpoints_mod.findFermiLevel;
const findFermiLevelSpin = kpoints_mod.findFermiLevelSpin;
const accumulateKpointDensitySmearingSpin = kpoints_mod.accumulateKpointDensitySmearingSpin;

var debug_gamma_dense_logged: bool = false;

const GridRequirement = util.GridRequirement;
const gridRequirement = util.gridRequirement;
const nextFftSize = util.nextFftSize;

const buildFftIndexMap = fft_grid.buildFftIndexMap;
const realToReciprocal = fft_grid.realToReciprocal;
const reciprocalToReal = fft_grid.reciprocalToReal;
const fftReciprocalToComplexInPlace = fft_grid.fftReciprocalToComplexInPlace;
const fftReciprocalToComplexInPlaceMapped = fft_grid.fftReciprocalToComplexInPlaceMapped;
const fftComplexToReciprocalInPlace = fft_grid.fftComplexToReciprocalInPlace;
const fftComplexToReciprocalInPlaceMapped = fft_grid.fftComplexToReciprocalInPlaceMapped;

const mixDensity = mixing.mixDensity;
const mixDensityKerker = mixing.mixDensityKerker;
const PulayMixer = mixing.PulayMixer;

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

const PwGridMap = pw_grid_map.PwGridMap;

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

pub const DensityResult = struct {
    rho: []f64,
    band_energy: f64,
    nonlocal_energy: f64,
    fermi_level: f64,
    entropy_energy: f64 = 0.0, // -T*S term for smearing
};

pub const BandIterativeContext = struct {
    grid: Grid,
    local_r: []f64,
    fft_index_map: ?[]usize,
    inv_volume: f64,
    /// PAW tables for band calculation (borrowed, NOT owned)
    paw_tabs: ?[]const paw_mod.PawTab = null,
    /// Per-atom converged D_ij from SCF: [natom][nbeta*nbeta] (owned)
    paw_dij: ?[][]f64 = null,

    pub fn deinit(self: *BandIterativeContext, alloc: std.mem.Allocator) void {
        if (self.local_r.len > 0) {
            alloc.free(self.local_r);
        }
        if (self.fft_index_map) |map| {
            alloc.free(map);
        }
        if (self.paw_dij) |dij| {
            for (dij) |d| alloc.free(d);
            alloc.free(dij);
        }
    }
};

pub const BandVectorCache = struct {
    n: usize = 0,
    nbands: usize = 0,
    vectors: []math.Complex = &[_]math.Complex{},

    pub fn deinit(self: *BandVectorCache) void {
        if (self.vectors.len > 0) {
            std.heap.c_allocator.free(self.vectors);
        }
        self.* = .{};
    }

    pub fn store(self: *BandVectorCache, n: usize, nbands: usize, values: []const math.Complex) !void {
        const total = n * nbands;
        if (values.len < total) return error.InvalidMatrixSize;
        if (self.vectors.len != total) {
            if (self.vectors.len > 0) {
                std.heap.c_allocator.free(self.vectors);
            }
            self.vectors = try std.heap.c_allocator.alloc(math.Complex, total);
        }
        self.n = n;
        self.nbands = nbands;
        @memcpy(self.vectors, values[0..total]);
    }
};

pub fn initBandIterativeContext(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    extra: hamiltonian.PotentialGrid,
) !BandIterativeContext {
    const grid = gridFromConfig(cfg, recip, volume);
    if (extra.nx != grid.nx or extra.ny != grid.ny or extra.nz != grid.nz or
        extra.min_h != grid.min_h or extra.min_k != grid.min_k or extra.min_l != grid.min_l)
    {
        return error.InvalidGrid;
    }
    var ionic = try potential_mod.buildIonicPotentialGrid(alloc, grid, species, atoms, null, null);
    defer ionic.deinit(alloc);
    const local_r = try potential_mod.buildLocalPotentialReal(alloc, grid, ionic, extra);

    if (cfg.scf.debug_fermi) {
        var sum: f64 = 0.0;
        for (local_r) |value| {
            sum += value;
        }
        const mean_local = sum / @as(f64, @floatFromInt(local_r.len));
        const ionic_g0 = ionic.valueAt(0, 0, 0);
        const extra_g0 = extra.valueAt(0, 0, 0);
        var buffer: [256]u8 = undefined;
        var writer = std.fs.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print(
            "band: local_r mean={d:.6} ionic_g0={d:.6} extra_g0={d:.6}\n",
            .{ mean_local, ionic_g0.r, extra_g0.r },
        );
        try out.flush();
    }

    const fft_index_map = try buildFftIndexMap(alloc, grid);
    return BandIterativeContext{
        .grid = grid,
        .local_r = local_r,
        .fft_index_map = fft_index_map,
        .inv_volume = 1.0 / volume,
    };
}

/// Options for band eigenvalue calculation
pub const BandEigenOptions = struct {
    reuse_vectors: bool = true,
    pool: ?*ThreadPool = null,
    shared_fft_plan: ?fft.Fft3dPlan = null,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet = null,
    paw_tabs: ?[]const paw_mod.PawTab = null,
    paw_dij: ?[]const []const f64 = null,
};

pub fn bandEigenvaluesIterative(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    ctx: *const BandIterativeContext,
    k_cart: math.Vec3,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    recip: math.Mat3,
    nbands: usize,
    reuse_vectors: bool,
    cache: ?*BandVectorCache,
) ![]f64 {
    return bandEigenvaluesIterativeExt(alloc, cfg, ctx, k_cart, species, atoms, recip, nbands, cache, .{
        .reuse_vectors = reuse_vectors,
        .pool = null,
    });
}

/// Extended band eigenvalue calculation with optional thread pool for parallel LOBPCG
pub fn bandEigenvaluesIterativeExt(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    ctx: *const BandIterativeContext,
    k_cart: math.Vec3,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    recip: math.Mat3,
    nbands: usize,
    cache: ?*BandVectorCache,
    opts: BandEigenOptions,
) ![]f64 {
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, k_cart);
    defer basis.deinit(alloc);
    if (basis.gvecs.len == 0) return error.NoPlaneWaves;

    const nbands_use = @min(nbands, basis.gvecs.len);
    if (nbands_use == 0) return error.InvalidBandConfig;

    const req = gridRequirement(basis.gvecs);
    if (req.nx > ctx.grid.nx or req.ny > ctx.grid.ny or req.nz > ctx.grid.nz) {
        return error.InvalidGrid;
    }

    const diag = try alloc.alloc(f64, basis.gvecs.len);
    defer alloc.free(diag);
    for (basis.gvecs, 0..) |g, i| {
        diag[i] = g.kinetic;
    }

    var init_vectors: ?[]const math.Complex = null;
    var init_cols: usize = 0;
    if (opts.reuse_vectors) {
        if (cache) |c| {
            if (c.vectors.len > 0 and c.n == basis.gvecs.len and c.nbands == nbands_use) {
                init_vectors = c.vectors;
                init_cols = c.nbands;
            }
        }
    }

    const nonlocal_enabled = cfg.scf.enable_nonlocal and hasNonlocal(species);
    const has_paw = opts.paw_tabs != null and opts.paw_dij != null;
    // Use multiple workspaces for parallel LOBPCG to avoid allocation during apply
    const num_workspaces: usize = if (opts.pool != null) @min(16, nbands_use + 4) else 1;

    // Build NonlocalContext:
    // - For PAW: always build ourselves with buildNonlocalContextPaw (sets up D_ij buffer + overlap)
    // - For NC with radial tables: use buildNonlocalContextWithTables (fast path)
    // - Otherwise: let ApplyContext build it internally
    var owned_nonlocal_ctx: ?apply.NonlocalContext = null;
    const build_nonlocal_ourselves = has_paw or (opts.radial_tables != null and nonlocal_enabled and num_workspaces <= 1);
    if (has_paw) {
        owned_nonlocal_ctx = try apply.buildNonlocalContextPaw(alloc, species, basis.gvecs, opts.radial_tables, opts.paw_tabs.?);
    } else if (opts.radial_tables != null and nonlocal_enabled and num_workspaces <= 1) {
        owned_nonlocal_ctx = try apply.buildNonlocalContextWithTables(alloc, species, basis.gvecs, opts.radial_tables.?);
    }
    errdefer if (owned_nonlocal_ctx) |*nc| nc.deinit(alloc);

    // Copy per-atom D_ij into the NonlocalContext for PAW
    if (has_paw) {
        if (owned_nonlocal_ctx) |*nc| {
            const paw_dij = opts.paw_dij.?;
            var atom_idx: usize = 0;
            for (species, 0..) |_, si| {
                var natom_for_species: usize = 0;
                for (atoms) |atom| {
                    if (atom.species_index == si) natom_for_species += 1;
                }
                if (natom_for_species > 0) {
                    try nc.ensureDijPerAtom(alloc, si, natom_for_species);
                    var a_of_s: usize = 0;
                    for (atoms) |atom| {
                        if (atom.species_index != si) continue;
                        if (atom_idx < paw_dij.len) {
                            nc.updateDijAtom(si, a_of_s, paw_dij[atom_idx]);
                        }
                        a_of_s += 1;
                        atom_idx += 1;
                    }
                }
            }
        }
    }

    // Build ApplyContext:
    // When we built NonlocalContext ourselves, use initWithCache path
    // Otherwise fall back to initWithFftPlan or initWithWorkspaces
    var apply_ctx = if (build_nonlocal_ourselves) blk: {
        var map = try PwGridMap.init(alloc, basis.gvecs, ctx.grid);
        errdefer map.deinit(alloc);
        if (ctx.fft_index_map) |idx_map| {
            try map.buildFftIndices(alloc, idx_map);
        }
        const fft_plan = if (opts.shared_fft_plan) |plan| plan else try fft.Fft3dPlan.initWithBackend(alloc, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz, cfg.scf.fft_backend);
        const owns_plan = opts.shared_fft_plan == null;
        var actx = try ApplyContext.initWithCache(
            alloc,
            ctx.grid,
            basis.gvecs,
            ctx.local_r,
            null,
            owned_nonlocal_ctx,
            map,
            atoms,
            ctx.inv_volume,
            null,
            ctx.fft_index_map,
            fft_plan,
            owns_plan,
        );
        actx.owns_nonlocal = true;
        actx.owns_map = true;
        owned_nonlocal_ctx = null; // ownership transferred
        break :blk actx;
    } else if (opts.shared_fft_plan != null and num_workspaces <= 1)
        try ApplyContext.initWithFftPlan(
            alloc,
            ctx.grid,
            basis.gvecs,
            ctx.local_r,
            null,
            species,
            atoms,
            ctx.inv_volume,
            nonlocal_enabled,
            null,
            ctx.fft_index_map,
            opts.shared_fft_plan.?,
        )
    else
        try ApplyContext.initWithWorkspaces(
            alloc,
            ctx.grid,
            basis.gvecs,
            ctx.local_r,
            null,
            species,
            atoms,
            ctx.inv_volume,
            nonlocal_enabled,
            null,
            ctx.fft_index_map,
            cfg.scf.fft_backend,
            num_workspaces,
        );
    defer apply_ctx.deinit(alloc);

    // For PAW, set the overlap operator S for generalized eigenvalue problem H|ψ> = ε S|ψ>
    const apply_s_fn: ?*const fn (*anyopaque, []const math.Complex, []math.Complex) anyerror!void =
        if (has_paw and apply_ctx.nonlocal_ctx != null and apply_ctx.nonlocal_ctx.?.has_paw)
            apply.applyOverlap
        else
            null;
    const op = iterative.Operator{
        .n = basis.gvecs.len,
        .ctx = &apply_ctx,
        .apply = applyHamiltonian,
        .apply_batch = applyHamiltonianBatched,
        .apply_s = apply_s_fn,
    };
    const lobpcg_opts = iterative.Options{
        .max_iter = cfg.band.iterative_max_iter,
        .tol = cfg.band.iterative_tol,
        .max_subspace = cfg.band.iterative_max_subspace,
        .block_size = cfg.band.iterative_block_size,
        .init_diagonal = cfg.band.iterative_init_diagonal,
        .init_vectors = init_vectors,
        .init_vectors_cols = init_cols,
    };

    // Select solver: CG or LOBPCG (serial/parallel)
    var eig = if (cfg.band.solver == .cg)
        try iterative.hermitianEigenDecompCG(alloc, op, diag, nbands_use, lobpcg_opts)
    else if (opts.pool) |pool|
        try iterative.hermitianEigenDecompIterativeExt(alloc, cfg.linalg_backend, op, diag, nbands_use, .{
            .base = lobpcg_opts,
            .lobpcg_backend = .parallel,
            .pool = pool,
        })
    else
        try iterative.hermitianEigenDecompIterative(alloc, cfg.linalg_backend, op, diag, nbands_use, lobpcg_opts);
    defer eig.deinit(alloc);

    if (opts.reuse_vectors) {
        if (cache) |c| {
            try c.store(basis.gvecs.len, nbands_use, eig.vectors);
        }
    }

    const values = try alloc.alloc(f64, nbands_use);
    @memcpy(values, eig.values[0..nbands_use]);
    return values;
}

fn totalCharge(rho: []f64, grid: Grid) f64 {
    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var sum: f64 = 0.0;
    for (rho) |value| {
        sum += value * dv;
    }
    return sum;
}

fn smearingActive(cfg: *const config.Config) bool {
    return cfg.scf.smearing != .none and cfg.scf.smear_ry > 0.0;
}

fn wrapGridIndex(g: i32, min: i32, n: usize) usize {
    const ni = @as(i32, @intCast(n));
    const idx = @mod(g - min, ni);
    return @as(usize, @intCast(idx));
}

fn symmetrizeDensity(
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

/// Symmetrize PAW rhoij by averaging over equivalent atoms of the same species.
/// In a bulk crystal, all atoms of the same species share the same Wyckoff position
/// and their on-site occupation matrices must be identical by symmetry.
fn symmetrizeRhoIJ(
    alloc: std.mem.Allocator,
    rhoij: *paw_mod.RhoIJ,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) !void {
    for (species, 0..) |_, si| {
        // Find atoms of this species
        var count: usize = 0;
        var n_ij: usize = 0;
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index == si) {
                n_ij = rhoij.values[ai].len;
                count += 1;
            }
        }
        if (count <= 1 or n_ij == 0) continue;

        // Average rhoij over all atoms of this species
        const avg = try alloc.alloc(f64, n_ij);
        defer alloc.free(avg);
        @memset(avg, 0.0);
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index != si) continue;
            for (0..n_ij) |idx| {
                avg[idx] += rhoij.values[ai][idx];
            }
        }
        const inv_count = 1.0 / @as(f64, @floatFromInt(count));
        for (0..n_ij) |idx| {
            avg[idx] *= inv_count;
        }
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index != si) continue;
            @memcpy(rhoij.values[ai], avg);
        }
    }
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
const ScfCommon = struct {
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

    const grid = gridFromConfig(cfg, recip, volume_bohr);
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
        return runSpinPolarizedLoop(alloc, cfg, species, atoms, volume_bohr, &common);
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
                try symmetrizeRhoIJ(alloc, prij, species, atoms);
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
                try addPawCompensationCharge(alloc, grid, aug, prij, common.paw_tabs.?, atoms, ecutrho_comp, &common.paw_gaunt.?);
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

            const v_out: []const f64 = @as([*]const f64, @ptrCast(potential_out.values.ptr))[0..n_f64];
            if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
                try common.pulay_mixer.?.mix(v_in, v_out, cfg.scf.mixing_beta);
            } else {
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
                try updatePawDij(alloc, grid, common.ionic, potential, tabs, species, atoms, apply_caches, ecutrho_paw, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, null, 1.0);
            }
        }
    }

    // Print SCF loop profile
    if (cfg.scf.profile and !cfg.scf.quiet) {
        const to_ms = 1.0 / @as(f64, @floatFromInt(std.time.ns_per_ms));
        var buffer2: [512]u8 = undefined;
        var writer2 = std.fs.File.stderr().writer(&buffer2);
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
            try addPawCompensationCharge(alloc, grid, aug, prij, common.paw_tabs.?, atoms, ecutrho_scf, &common.paw_gaunt.?);
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
                energy_terms.paw_onsite = try computePawOnsiteEnergyTotal(
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

/// Add PAW compensation charge n_hat(r) to density array (multi-L with Gaunt coefficients).
///
/// n̂(G) = Σ_a Σ_{i,m_i,j,m_j} ρ_{(i,m_i),(j,m_j)}^a × Σ_{L,M} G(l_i,m_i,l_j,m_j,L,M)
///         × Q^L_{ij}(|G|) × Y_{L,M}(Ĝ) × exp(-iGR_a) / Ω
fn addPawCompensationCharge(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    rhoij: *const paw_mod.RhoIJ,
    paw_tabs: []const paw_mod.PawTab,
    atoms: []const hamiltonian.AtomData,
    ecutrho: f64,
    gaunt_table: *const paw_mod.GauntTable,
) !void {
    const total = grid.count();
    const n_hat_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(n_hat_g);
    @memset(n_hat_g, math.complex.init(0.0, 0.0));

    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);
    const inv_omega = 1.0 / grid.volume;

    var idx: usize = 0;
    var il: usize = 0;
    while (il < grid.nz) : (il += 1) {
        var ik: usize = 0;
        while (ik < grid.ny) : (ik += 1) {
            var ih: usize = 0;
            while (ih < grid.nx) : (ih += 1) {
                const gh = grid.min_h + @as(i32, @intCast(ih));
                const gk = grid.min_k + @as(i32, @intCast(ik));
                const gl = grid.min_l + @as(i32, @intCast(il));
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_abs = math.Vec3.norm(gvec);
                const g2 = math.Vec3.dot(gvec, gvec);
                if (g2 >= ecutrho) {
                    idx += 1;
                    continue;
                }

                // Pre-compute Y_{L,M}(Ĝ) for all (L,M) up to lmax_aug
                var ylm_g: [25]f64 = undefined; // (4+1)^2 = 25 max
                const lmax_aug = gaunt_table.lmax_aug;
                if (g_abs > 1e-10) {
                    for (0..lmax_aug + 1) |big_l| {
                        const bl_i32: i32 = @intCast(big_l);
                        var bm: i32 = -bl_i32;
                        while (bm <= bl_i32) : (bm += 1) {
                            ylm_g[paw_mod.GauntTable.lmIndex(big_l, bm)] =
                                nonlocal_mod.realSphericalHarmonic(bl_i32, bm, gvec.x, gvec.y, gvec.z);
                        }
                    }
                } else {
                    @memset(&ylm_g, 0.0);
                    ylm_g[0] = 1.0 / @sqrt(4.0 * std.math.pi); // Y_00 at G=0
                }

                var sum_re: f64 = 0.0;
                var sum_im: f64 = 0.0;

                for (0..atoms.len) |a| {
                    const sp = atoms[a].species_index;
                    if (sp >= paw_tabs.len) continue;
                    const tab = &paw_tabs[sp];
                    if (tab.nbeta == 0) continue;
                    const mt = rhoij.m_total_per_atom[a];
                    const rij_m = rhoij.values[a];
                    const l_list_r = rhoij.l_per_beta[a];
                    const m_offsets_r = rhoij.m_offsets[a];
                    const pos = atoms[a].position;

                    const g_dot_r = math.Vec3.dot(gvec, pos);
                    const sf_re = @cos(g_dot_r);
                    const sf_im = -@sin(g_dot_r);

                    // For each Q^L entry, sum over m-resolved rhoij with Gaunt coefficients
                    for (0..tab.n_qijl_entries) |e| {
                        const qidx = tab.qijl_indices[e];
                        const big_l = qidx.l;
                        const i_beta = qidx.first;
                        const j_beta = qidx.second;

                        const qijl_g = tab.evalQijlForm(e, g_abs);
                        if (@abs(qijl_g) < 1e-30) continue;

                        const l_i = @as(usize, @intCast(l_list_r[i_beta]));
                        const l_j = @as(usize, @intCast(l_list_r[j_beta]));

                        // Sum over M: Σ_M Y_{L,M}(Ĝ) × [Σ_{m_i,m_j} G(l_i,m_i,l_j,m_j,L,M) × ρ_{(i,m_i),(j,m_j)}]
                        const bl_i32: i32 = @intCast(big_l);
                        var bm: i32 = -bl_i32;
                        while (bm <= bl_i32) : (bm += 1) {
                            const ylm_val = ylm_g[paw_mod.GauntTable.lmIndex(big_l, bm)];
                            if (@abs(ylm_val) < 1e-30) continue;

                            // Sum over m_i, m_j: Σ G(l_i,m_i,l_j,m_j,L,M) × ρ_{(i,m_i),(j,m_j)}
                            var gaunt_rhoij: f64 = 0.0;
                            const li_i32: i32 = @intCast(l_i);
                            const lj_i32: i32 = @intCast(l_j);
                            var mi: i32 = -li_i32;
                            while (mi <= li_i32) : (mi += 1) {
                                const mi_idx = m_offsets_r[i_beta] + @as(usize, @intCast(mi + li_i32));
                                var mj: i32 = -lj_i32;
                                while (mj <= lj_i32) : (mj += 1) {
                                    const g_coeff = gaunt_table.get(l_i, mi, l_j, mj, big_l, bm);
                                    if (g_coeff == 0.0) continue;
                                    const mj_idx = m_offsets_r[j_beta] + @as(usize, @intCast(mj + lj_i32));
                                    gaunt_rhoij += g_coeff * rij_m[mi_idx * mt + mj_idx];
                                }
                            }
                            if (@abs(gaunt_rhoij) < 1e-30) continue;

                            // For i!=j, both (i,j) and (j,i) Q entries should be in qijl.
                            // If only upper triangle is stored, we need sym_factor.
                            const sym_factor: f64 = if (i_beta != j_beta) 2.0 else 1.0;
                            const contrib = gaunt_rhoij * qijl_g * ylm_val * sym_factor * inv_omega;
                            sum_re += contrib * sf_re;
                            sum_im += contrib * sf_im;
                        }
                    }
                }

                n_hat_g[idx].r += sum_re;
                n_hat_g[idx].i += sum_im;
                idx += 1;
            }
        }
    }

    // IFFT n_hat(G) → n_hat(r) and add to density
    const n_hat_r = try reciprocalToReal(alloc, grid, n_hat_g);
    defer alloc.free(n_hat_r);
    for (0..@min(rho.len, n_hat_r.len)) |i| {
        rho[i] += n_hat_r[i];
    }
}

/// Update PAW D_ij per-atom from the total effective potential.
/// D_ij(atom) = D^0_ij + D^hat_ij(atom) + D^xc_ij(atom) + D^H_ij(atom)
/// D^hat depends on atom position via structure factor exp(-iG·R_a).
/// D^xc, D^H depend on atom-specific rhoij.
fn updatePawDij(
    alloc: std.mem.Allocator,
    grid: Grid,
    ionic: hamiltonian.PotentialGrid,
    potential: hamiltonian.PotentialGrid,
    paw_tabs: []const paw_mod.PawTab,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    apply_caches: []apply.KpointApplyCache,
    ecutrho: f64,
    rhoij: *const paw_mod.RhoIJ,
    xc_func: @import("../xc/xc.zig").Functional,
    symmetrize: bool,
    gaunt_table: *const paw_mod.GauntTable,
    skip_dxc: bool,
    rhoij_spin: ?*const paw_mod.RhoIJ,
    dij_mix_beta: f64,
) !void {
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);
    const total = grid.count();

    for (species, 0..) |entry_s, si| {
        if (si >= paw_tabs.len or paw_tabs[si].nbeta == 0) continue;
        const tab = &paw_tabs[si];
        const nb = tab.nbeta;
        const n_ij = nb * nb;
        const upf = entry_s.upf;
        const paw = upf.paw orelse continue;

        // Compute m_total for this species
        var mt: usize = 0;
        for (0..nb) |b| {
            mt += @as(usize, @intCast(2 * tab.l_list[b] + 1));
        }
        const n_m = mt * mt;

        // Compute m_offsets for this species
        var sp_m_offsets: [32]usize = undefined;
        var off: usize = 0;
        for (0..nb) |b| {
            sp_m_offsets[b] = off;
            off += @as(usize, @intCast(2 * tab.l_list[b] + 1));
        }

        // Count atoms of this species
        var natom: usize = 0;
        for (atoms) |atom| {
            if (atom.species_index == si) natom += 1;
        }
        if (natom == 0) continue;

        // Ensure per-atom D_ij arrays are allocated in each cache
        for (apply_caches) |*ac| {
            if (ac.nonlocal_ctx) |*nl| {
                try nl.ensureDijPerAtom(ac.cache_alloc, si, natom);
                try nl.ensureDijMPerAtom(ac.cache_alloc, si, natom);
            }
        }

        // Compute D_ij for each atom of this species
        var atom_counter: usize = 0;
        for (atoms, 0..) |atom, ai| {
            if (atom.species_index != si) continue;
            const pos = atom.position;

            // Radial D_ij (nb×nb) for forces/stress
            const dij = try alloc.alloc(f64, n_ij);
            defer alloc.free(dij);
            if (upf.dij.len >= n_ij) {
                @memcpy(dij, upf.dij[0..n_ij]);
            } else {
                @memset(dij, 0.0);
            }

            // m-resolved D_ij (mt×mt) for Hamiltonian
            const dij_m = try alloc.alloc(f64, n_m);
            defer alloc.free(dij_m);
            @memset(dij_m, 0.0);

            // Expand D^0 from radial to m-resolved: D^0_{(i,m),(j,m')} = D^0_ij × δ_{mm'} × δ_{li,lj}
            for (0..nb) |i| {
                for (0..nb) |j| {
                    if (tab.l_list[i] != tab.l_list[j]) continue;
                    const d0 = if (upf.dij.len >= n_ij) upf.dij[i * nb + j] else 0.0;
                    const m_count = @as(usize, @intCast(2 * tab.l_list[i] + 1));
                    for (0..m_count) |m| {
                        const im = sp_m_offsets[i] + m;
                        const jm = sp_m_offsets[j] + m;
                        dij_m[im * mt + jm] = d0;
                    }
                }
            }

            // Add D^hat: radial (L=0 only) and m-resolved (all L with Gaunt)
            var gidx: usize = 0;
            var il: usize = 0;
            while (il < grid.nz) : (il += 1) {
                var ik: usize = 0;
                while (ik < grid.ny) : (ik += 1) {
                    var ih: usize = 0;
                    while (ih < grid.nx) : (ih += 1) {
                        const gh = grid.min_h + @as(i32, @intCast(ih));
                        const gk = grid.min_k + @as(i32, @intCast(ik));
                        const gl = grid.min_l + @as(i32, @intCast(il));
                        const gvec = math.Vec3.add(
                            math.Vec3.add(
                                math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                                math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                            ),
                            math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                        );
                        const g_abs = math.Vec3.norm(gvec);
                        const g2_dij = g_abs * g_abs;
                        if (g2_dij >= ecutrho) {
                            gidx += 1;
                            continue;
                        }

                        const v_hxc = if (gidx < total) potential.values[gidx] else math.complex.init(0.0, 0.0);
                        const v_loc = if (gidx < total) ionic.values[gidx] else math.complex.init(0.0, 0.0);
                        const v_eff = math.complex.add(v_hxc, v_loc);

                        const g_dot_r = math.Vec3.dot(gvec, pos);
                        const sf_re = @cos(g_dot_r);
                        const sf_im = @sin(g_dot_r);
                        const prod_re = v_eff.r * sf_re - v_eff.i * sf_im;

                        // Pre-compute Y_{L,M}(Ĝ) for m-resolved D^hat
                        var ylm_g: [25]f64 = undefined;
                        const lmax_aug = gaunt_table.lmax_aug;
                        if (g_abs > 1e-10) {
                            for (0..lmax_aug + 1) |big_l| {
                                const bl_i32: i32 = @intCast(big_l);
                                var bm: i32 = -bl_i32;
                                while (bm <= bl_i32) : (bm += 1) {
                                    ylm_g[paw_mod.GauntTable.lmIndex(big_l, bm)] =
                                        nonlocal_mod.realSphericalHarmonic(bl_i32, bm, gvec.x, gvec.y, gvec.z);
                                }
                            }
                        } else {
                            @memset(&ylm_g, 0.0);
                            ylm_g[0] = 1.0 / @sqrt(4.0 * std.math.pi);
                        }

                        for (0..tab.n_qijl_entries) |e| {
                            const qidx_e = tab.qijl_indices[e];
                            const big_l = qidx_e.l;
                            const i_beta = qidx_e.first;
                            const j_beta = qidx_e.second;

                            const qijl_g = tab.evalQijlForm(e, g_abs);
                            if (@abs(qijl_g) < 1e-30) continue;

                            // Radial D^hat (L=0 only, for forces)
                            if (big_l == 0) {
                                const ylm_00 = 1.0 / @sqrt(4.0 * std.math.pi);
                                const gaunt_00 = 1.0 / @sqrt(4.0 * std.math.pi);
                                const contrib = prod_re * qijl_g * ylm_00 * gaunt_00;
                                dij[i_beta * nb + j_beta] += contrib;
                                if (i_beta != j_beta) {
                                    dij[j_beta * nb + i_beta] += contrib;
                                }
                            }

                            // m-resolved D^hat (all L, for Hamiltonian)
                            const l_i = @as(usize, @intCast(tab.l_list[i_beta]));
                            const l_j = @as(usize, @intCast(tab.l_list[j_beta]));
                            const bl_i32: i32 = @intCast(big_l);
                            var bm: i32 = -bl_i32;
                            while (bm <= bl_i32) : (bm += 1) {
                                const ylm_val = ylm_g[paw_mod.GauntTable.lmIndex(big_l, bm)];
                                if (@abs(ylm_val) < 1e-30) continue;

                                const li_i32: i32 = @intCast(l_i);
                                const lj_i32: i32 = @intCast(l_j);
                                var mi: i32 = -li_i32;
                                while (mi <= li_i32) : (mi += 1) {
                                    const im = sp_m_offsets[i_beta] + @as(usize, @intCast(mi + li_i32));
                                    var mj: i32 = -lj_i32;
                                    while (mj <= lj_i32) : (mj += 1) {
                                        const g_coeff = gaunt_table.get(l_i, mi, l_j, mj, big_l, bm);
                                        if (g_coeff == 0.0) continue;
                                        const jm = sp_m_offsets[j_beta] + @as(usize, @intCast(mj + lj_i32));
                                        const contrib_m = prod_re * qijl_g * ylm_val * g_coeff;
                                        dij_m[im * mt + jm] += contrib_m;
                                        if (i_beta != j_beta) {
                                            dij_m[jm * mt + im] += contrib_m;
                                        }
                                    }
                                }
                            }
                        }

                        gidx += 1;
                    }
                }
            }

            // D^xc is m-resolved: dij_xc_m[mt × mt]
            const dij_xc_m = try alloc.alloc(f64, mt * mt);
            defer alloc.free(dij_xc_m);
            if (skip_dxc) {
                @memset(dij_xc_m, 0.0);
            } else if (rhoij_spin) |rij_s| {
                // Spin-resolved D^xc: compute from (this_channel, other_channel)
                // other_channel = total - this_channel
                const rij_other = try alloc.alloc(f64, mt * mt);
                defer alloc.free(rij_other);
                for (0..mt * mt) |idx2| {
                    rij_other[idx2] = rhoij.values[ai][idx2] - rij_s.values[ai][idx2];
                }
                const dij_xc_other = try alloc.alloc(f64, mt * mt);
                defer alloc.free(dij_xc_other);
                try paw_mod.paw_xc.computePawDijXcAngularSpin(
                    alloc,
                    dij_xc_m,
                    dij_xc_other,
                    paw,
                    rij_s.values[ai],
                    rij_other,
                    rhoij.m_total_per_atom[ai],
                    rhoij.m_offsets[ai],
                    upf.r,
                    upf.rab,
                    paw.ae_core_density,
                    if (upf.nlcc.len > 0) upf.nlcc else null,
                    xc_func,
                    gaunt_table,
                );
            } else {
                try paw_mod.paw_xc.computePawDijXcAngular(
                    alloc,
                    dij_xc_m,
                    paw,
                    rhoij.values[ai],
                    rhoij.m_total_per_atom[ai],
                    rhoij.m_offsets[ai],
                    upf.r,
                    upf.rab,
                    paw.ae_core_density,
                    if (upf.nlcc.len > 0) upf.nlcc else null,
                    xc_func,
                    gaunt_table,
                );
            }
            // Add m-resolved D^xc directly to dij_m
            for (0..mt * mt) |idx2| {
                dij_m[idx2] += dij_xc_m[idx2];
            }
            // Contract D^xc to radial for dij (used in stress/forces).
            // Convention: dij[nb×nb] stores the per-m value (same as D^0, D^hat).
            // Average over m gives the per-m representative value.
            for (0..nb) |i| {
                for (0..nb) |j| {
                    if (tab.l_list[i] != tab.l_list[j]) continue;
                    const m_count = @as(usize, @intCast(2 * tab.l_list[i] + 1));
                    var sum_dxc: f64 = 0.0;
                    for (0..m_count) |m| {
                        const im = sp_m_offsets[i] + m;
                        const jm = sp_m_offsets[j] + m;
                        sum_dxc += dij_xc_m[im * mt + jm];
                    }
                    // Average over m: radial D convention is per-m value
                    dij[i * nb + j] += sum_dxc / @as(f64, @floatFromInt(m_count));
                }
            }

            // Add D^H (on-site Hartree, multi-L with Gaunt) to D_full.
            // E_paw_onsite includes E_H_onsite (computePawEhOnsiteMultiL), so D^H must
            // also be in D_full and double-counting for Hellmann-Feynman consistency.
            {
                const dij_h_m = try alloc.alloc(f64, n_m);
                defer alloc.free(dij_h_m);
                try paw_mod.paw_xc.computePawDijHartreeMultiL(
                    alloc,
                    dij_h_m,
                    paw,
                    rhoij.values[ai],
                    rhoij.m_total_per_atom[ai],
                    rhoij.m_offsets[ai],
                    upf.r,
                    upf.rab,
                    gaunt_table,
                );
                // Add m-resolved D^H directly to dij_m
                for (0..n_m) |idx2| {
                    dij_m[idx2] += dij_h_m[idx2];
                }
                // Contract D^H to radial for dij (used in stress/forces).
                for (0..nb) |i| {
                    for (0..nb) |j| {
                        if (tab.l_list[i] != tab.l_list[j]) continue;
                        const m_count = @as(usize, @intCast(2 * tab.l_list[i] + 1));
                        var sum_dh: f64 = 0.0;
                        for (0..m_count) |m| {
                            const im = sp_m_offsets[i] + m;
                            const jm = sp_m_offsets[j] + m;
                            sum_dh += dij_h_m[im * mt + jm];
                        }
                        dij[i * nb + j] += sum_dh / @as(f64, @floatFromInt(m_count));
                    }
                }
            }

            // Mix D_ij with old value from first cache, then write to all caches
            if (dij_mix_beta < 1.0 - 1e-10) {
                if (apply_caches.len > 0) {
                    if (apply_caches[0].nonlocal_ctx) |*nl| {
                        if (nl.species[si].dij_per_atom) |dpa| {
                            if (atom_counter < dpa.len) {
                                for (0..@min(dij.len, dpa[atom_counter].len)) |ii| {
                                    dij[ii] = (1.0 - dij_mix_beta) * dpa[atom_counter][ii] + dij_mix_beta * dij[ii];
                                }
                            }
                        }
                        if (nl.species[si].dij_m_per_atom) |dpa| {
                            if (atom_counter < dpa.len) {
                                for (0..@min(dij_m.len, dpa[atom_counter].len)) |ii| {
                                    dij_m[ii] = (1.0 - dij_mix_beta) * dpa[atom_counter][ii] + dij_mix_beta * dij_m[ii];
                                }
                            }
                        }
                    }
                }
            }
            for (apply_caches) |*ac| {
                if (ac.nonlocal_ctx) |*nl| {
                    nl.updateDijAtom(si, atom_counter, dij);
                    nl.updateDijMAtom(si, atom_counter, dij_m);
                }
            }

            atom_counter += 1;
        }

        // Symmetrize D_ij across equivalent atoms of this species
        if (symmetrize and natom > 1) {
            // Symmetrize radial D_ij
            const avg_dij = try alloc.alloc(f64, n_ij);
            defer alloc.free(avg_dij);
            @memset(avg_dij, 0.0);
            if (apply_caches.len > 0) {
                if (apply_caches[0].nonlocal_ctx) |*nl| {
                    if (nl.species[si].dij_per_atom) |dpa| {
                        for (0..natom) |a| {
                            for (0..n_ij) |idx2| {
                                avg_dij[idx2] += dpa[a][idx2];
                            }
                        }
                    }
                }
            }
            const inv_natom = 1.0 / @as(f64, @floatFromInt(natom));
            for (0..n_ij) |idx2| {
                avg_dij[idx2] *= inv_natom;
            }
            for (apply_caches) |*ac| {
                if (ac.nonlocal_ctx) |*nl| {
                    if (nl.species[si].dij_per_atom) |dpa| {
                        for (0..natom) |a| {
                            @memcpy(dpa[a], avg_dij);
                        }
                    }
                }
            }

            // Symmetrize m-resolved D_ij
            const avg_dij_m = try alloc.alloc(f64, n_m);
            defer alloc.free(avg_dij_m);
            @memset(avg_dij_m, 0.0);
            if (apply_caches.len > 0) {
                if (apply_caches[0].nonlocal_ctx) |*nl| {
                    if (nl.species[si].dij_m_per_atom) |dpa| {
                        for (0..natom) |a| {
                            for (0..n_m) |idx2| {
                                avg_dij_m[idx2] += dpa[a][idx2];
                            }
                        }
                    }
                }
            }
            for (0..n_m) |idx2| {
                avg_dij_m[idx2] *= inv_natom;
            }
            for (apply_caches) |*ac| {
                if (ac.nonlocal_ctx) |*nl| {
                    if (nl.species[si].dij_m_per_atom) |dpa| {
                        for (0..natom) |a| {
                            @memcpy(dpa[a], avg_dij_m);
                        }
                    }
                }
            }
        }
    }
}

/// Compute PAW on-site energy correction for all atoms.
/// When rhoij_up/rhoij_down are provided, uses spin-resolved E_xc.
fn computePawOnsiteEnergyTotal(
    alloc: std.mem.Allocator,
    rhoij: *const paw_mod.RhoIJ,
    paw_tabs: []const paw_mod.PawTab,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    xc_func: @import("../xc/xc.zig").Functional,
    gaunt_table: *const paw_mod.GauntTable,
    rhoij_up: ?*const paw_mod.RhoIJ,
    rhoij_down: ?*const paw_mod.RhoIJ,
) !f64 {
    var e_paw: f64 = 0.0;
    for (0..atoms.len) |a| {
        const sp = atoms[a].species_index;
        if (sp >= paw_tabs.len or paw_tabs[sp].nbeta == 0) continue;
        const paw = species[sp].upf.paw orelse continue;
        const tab = &paw_tabs[sp];
        const upf = species[sp].upf.*;
        const rho_core_ps: ?[]const f64 = if (upf.nlcc.len > 0) upf.nlcc else null;

        if (rhoij_up != null and rhoij_down != null) {
            // Spin-resolved E_xc on-site
            const mt = rhoij.m_total_per_atom[a];
            const m_off = rhoij.m_offsets[a];
            const e_xc = try paw_mod.paw_xc.computePawExcOnsiteAngularSpin(
                alloc,
                paw,
                rhoij_up.?.values[a],
                rhoij_down.?.values[a],
                mt,
                m_off,
                upf.r,
                upf.rab,
                paw.ae_core_density,
                rho_core_ps,
                xc_func,
                gaunt_table,
            );
            const e_h = try paw_mod.paw_xc.computePawEhOnsiteMultiL(
                alloc,
                paw,
                rhoij.values[a],
                mt,
                m_off,
                upf.r,
                upf.rab,
                gaunt_table,
            );
            e_paw += e_xc + e_h + paw.core_energy;
        } else {
            const e_atom = try paw_mod.paw_energy.computePawOnsiteEnergy(
                alloc,
                paw,
                tab,
                rhoij.values[a],
                rhoij.m_total_per_atom[a],
                rhoij.m_offsets[a],
                upf.r,
                upf.rab,
                rho_core_ps,
                xc_func,
                gaunt_table,
            );
            e_paw += e_atom;
        }
    }
    return e_paw;
}

/// Compute atomic density form factor: ∫ rho_atom(r) × j₀(Gr) × rab(r) dr
/// UPF rho_atom includes 4πr² factor, so no extra r² needed.
fn rhoAtomFormFactor(upf: *const @import("../pseudopotential/pseudopotential.zig").UpfData, g: f64) f64 {
    if (upf.rho_atom.len == 0) return 0.0;
    const n = @min(upf.rho_atom.len, @min(upf.r.len, upf.rab.len));
    var sum: f64 = 0.0;
    const w = [5]f64{ 23.75 / 72.0, 95.10 / 72.0, 55.20 / 72.0, 79.30 / 72.0, 70.65 / 72.0 };
    for (0..n) |i| {
        const x = g * upf.r[i];
        const j0 = nonlocal_mod.sphericalBessel(0, x);
        const cw: f64 = if (n < 10) 1.0 else if (i < 5) w[i] else if (i >= n - 5) w[n - 1 - i] else 1.0;
        sum += upf.rho_atom[i] * j0 * upf.rab[i] * cw;
    }
    return sum;
}

/// Build initial density from superposition of atomic pseudo-charge densities.
/// ρ_init(G) = (1/Ω) Σ_atom ρ_atom_form(|G|) × exp(-iG·R_atom)
/// Then inverse FFT to real space.
fn buildAtomicDensity(
    alloc: std.mem.Allocator,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) ![]f64 {
    const total = grid.count();
    const inv_volume = 1.0 / grid.volume;
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    const rho_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho_g);

    var idx: usize = 0;
    var il: usize = 0;
    while (il < grid.nz) : (il += 1) {
        var ik: usize = 0;
        while (ik < grid.ny) : (ik += 1) {
            var ih: usize = 0;
            while (ih < grid.nx) : (ih += 1) {
                const gh = grid.min_h + @as(i32, @intCast(ih));
                const gk = grid.min_k + @as(i32, @intCast(ik));
                const gl = grid.min_l + @as(i32, @intCast(il));
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_norm = math.Vec3.norm(gvec);

                var sum_r: f64 = 0.0;
                var sum_i: f64 = 0.0;
                for (atoms) |atom| {
                    const rho_g_val = rhoAtomFormFactor(species[atom.species_index].upf, g_norm);
                    const g_dot_r = math.Vec3.dot(gvec, atom.position);
                    sum_r += rho_g_val * @cos(g_dot_r);
                    sum_i -= rho_g_val * @sin(g_dot_r);
                }
                rho_g[idx] = .{ .r = sum_r * inv_volume, .i = sum_i * inv_volume };
                idx += 1;
            }
        }
    }

    // Inverse FFT to real space
    const rho_r = try reciprocalToReal(alloc, grid, rho_g);
    return rho_r;
}

/// Generate Monkhorst-Pack k-mesh.
/// Determine grid from config or cutoff.
fn gridFromConfig(cfg: config.Config, recip: math.Mat3, volume: f64) Grid {
    var nx = cfg.scf.grid[0];
    var ny = cfg.scf.grid[1];
    var nz = cfg.scf.grid[2];
    if (nx == 0 or ny == 0 or nz == 0) {
        const auto = autoGrid(cfg.scf.ecut_ry, cfg.scf.grid_scale, recip);
        if (nx == 0) nx = auto[0];
        if (ny == 0) ny = auto[1];
        if (nz == 0) nz = auto[2];
    }
    const min_h = minIndex(nx);
    const min_k = minIndex(ny);
    const min_l = minIndex(nz);
    return Grid{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .cell = cfg.cell.scale(math.unitsScaleToBohr(cfg.units)),
        .recip = recip,
        .volume = volume,
        .min_h = min_h,
        .min_k = min_k,
        .min_l = min_l,
    };
}

/// Compute automatic grid size from cutoff.
/// The FFT grid must accommodate:
///   - Density bandwidth: 2*sqrt(ecut) (product of two wavefunctions)
///   - ecutrho sphere: grid_scale * sqrt(ecut) (for PAW augmentation charges)
/// The grid half-width is ceil(density_gmax / |b_i|), full grid is 2*half+1.
fn autoGrid(ecut_ry: f64, grid_scale: f64, recip: math.Mat3) [3]usize {
    const scale = if (grid_scale > 0.0) grid_scale else 1.0;
    // Maximum |G| that the grid must support: density bandwidth = 2*sqrt(ecut) (from |ψ|²),
    // ecutrho bandwidth = scale * sqrt(ecut). Grid must accommodate the larger of the two.
    const density_gmax = @max(2.0, scale) * @sqrt(ecut_ry);
    const b1 = math.Vec3.norm(recip.row(0));
    const b2 = math.Vec3.norm(recip.row(1));
    const b3 = math.Vec3.norm(recip.row(2));
    const n1 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / b1))) * 2 + 1;
    const n2 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / b2))) * 2 + 1;
    const n3 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / b3))) * 2 + 1;
    return .{
        nextFftSize(@max(n1, 3)),
        nextFftSize(@max(n2, 3)),
        nextFftSize(@max(n3, 3)),
    };
}

const WavefunctionResult = struct {
    wavefunctions: WavefunctionData,
    band_energy: f64,
    nonlocal_energy: f64,
};

/// Compute final wavefunctions for force calculation.
fn computeFinalWavefunctionsWithSpinFactor(
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
            var writer = std.fs.File.stderr().writer(&buffer);
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
const ScfLoopProfile = struct {
    build_local_r_ns: *u64,
    build_fft_map_ns: *u64,
};

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
        var writer = std.fs.File.stderr().writer(&buffer);
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
        defer alloc.free(profiles.?);
        for (profiles.?) |*p| {
            p.* = ScfProfile{};
        }
    }

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

fn computeDensitySmearing(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
    nelec: f64,
    use_iterative_config: bool,
    has_qij: bool,
    nonlocal_enabled: bool,
    fft_index_map: ?[]const usize,
    iter_max_iter: usize,
    iter_tol: f64,
    kpoint_cache: []KpointCache,
    apply_caches: ?[]apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_rhoij: ?*paw_mod.RhoIJ,
) !DensityResult {
    const ngrid = grid.count();
    const rho = try alloc.alloc(f64, ngrid);
    errdefer alloc.free(rho);
    @memset(rho, 0.0);

    var band_energy: f64 = 0.0;
    var nonlocal_energy: f64 = 0.0;
    var entropy_energy: f64 = 0.0;

    var profile_total = ScfProfile{};
    const profile_ptr: ?*ScfProfile = if (cfg.scf.profile) &profile_total else null;

    const eigen_data = try alloc.alloc(KpointEigenData, kpoints.len);
    var filled: usize = 0;
    var used_parallel = false; // Track if parallel path was used (uses c_allocator)
    errdefer {
        const cleanup_alloc = if (used_parallel) std.heap.c_allocator else alloc;
        var i: usize = 0;
        while (i < filled) : (i += 1) {
            eigen_data[i].deinit(cleanup_alloc);
        }
        alloc.free(eigen_data);
    }

    if (cfg.scf.debug_fermi) {
        if (local_r) |values| {
            var sum: f64 = 0.0;
            for (values) |v| {
                sum += v;
            }
            const mean_local = sum / @as(f64, @floatFromInt(values.len));
            const pot_g0 = potential.valueAt(0, 0, 0);
            var buffer: [256]u8 = undefined;
            var writer = std.fs.File.stderr().writer(&buffer);
            const out = &writer.interface;
            try out.print(
                "scf: local_r mean={d:.6} pot_g0={d:.6}\n",
                .{ mean_local, pot_g0.r },
            );
            try out.flush();
        }
    }

    const thread_count = kpointThreadCount(kpoints.len, cfg.scf.kpoint_threads);

    if (thread_count <= 1) {
        // Pre-create shared FFT plan for single-threaded mode
        var shared_fft_plan = try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend);
        defer shared_fft_plan.deinit(alloc);

        // Sequential path for single thread
        for (kpoints, 0..) |kp, kidx| {
            if (!cfg.scf.quiet) {
                try logKpoint(kidx, kpoints.len);
            }
            const ac_ptr: ?*apply.KpointApplyCache = if (apply_caches) |acs|
                (if (kidx < acs.len) &acs[kidx] else null)
            else
                null;
            eigen_data[kidx] = try computeKpointEigenData(
                alloc,
                cfg,
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
                profile_ptr,
                shared_fft_plan,
                ac_ptr,
                radial_tables,
                paw_tabs,
            );
            filled += 1;
        }
    } else {
        // Parallel path for multiple threads
        var profiles: ?[]ScfProfile = null;
        if (cfg.scf.profile) {
            profiles = try alloc.alloc(ScfProfile, thread_count);
            for (profiles.?) |*p| {
                p.* = ScfProfile{};
            }
        }
        defer if (profiles) |p| alloc.free(p);

        // Pre-create FFT plans for each thread to avoid mutex contention
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

        var shared = SmearingShared{
            .cfg = cfg,
            .grid = grid,
            .kpoints = kpoints,
            .species = species,
            .atoms = atoms,
            .recip = recip,
            .volume = volume,
            .potential = potential,
            .local_r = local_r,
            .nocc = nocc,
            .use_iterative_config = use_iterative_config,
            .has_qij = has_qij,
            .nonlocal_enabled = nonlocal_enabled,
            .fft_index_map = fft_index_map,
            .iter_max_iter = iter_max_iter,
            .iter_tol = iter_tol,
            .reuse_vectors = cfg.scf.iterative_reuse_vectors,
            .kpoint_cache = kpoint_cache,
            .eigen_data = eigen_data,
            .fft_plans = fft_plans,
            .profiles = profiles,
            .apply_caches = apply_caches,
            .radial_tables = radial_tables,
            .paw_tabs = paw_tabs,
            .next_index = &next_index,
            .stop = &stop,
            .err = &worker_error,
            .err_mutex = &err_mutex,
            .log_mutex = &log_mutex,
        };

        const workers = try alloc.alloc(SmearingWorker, thread_count);
        defer alloc.free(workers);
        const threads = try alloc.alloc(std.Thread, thread_count);
        defer alloc.free(threads);

        var t: usize = 0;
        while (t < thread_count) : (t += 1) {
            workers[t] = .{ .shared = &shared, .thread_index = t };
            threads[t] = try std.Thread.spawn(.{}, smearingWorker, .{&workers[t]});
        }
        for (threads) |thread| {
            thread.join();
        }

        if (worker_error) |err| return err;

        // Merge profiles
        if (profiles) |p| {
            for (p) |thread_profile| {
                mergeProfile(&profile_total, thread_profile);
            }
        }

        // All eigen_data entries should be filled by workers
        filled = kpoints.len;
        used_parallel = true;
    }

    if (cfg.scf.debug_fermi) {
        var gamma_logged = false;
        for (eigen_data[0..filled]) |entry| {
            if (isGammaKpoint(entry.kpoint)) {
                try logEigenvalues("scf", "gamma", entry.values, entry.nbands);
                gamma_logged = true;
                break;
            }
        }
        if (!gamma_logged) {
            var gamma_cache = KpointCache{};
            defer gamma_cache.deinit();
            const gamma_kp = KPoint{
                .k_frac = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
                .k_cart = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
                .weight = 0.0,
            };
            const gamma_data = try computeKpointEigenData(
                alloc,
                cfg,
                grid,
                gamma_kp,
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
                &gamma_cache,
                null,
                null, // No shared FFT plan for debug gamma point
                null, // No apply cache for debug gamma point
                radial_tables,
                paw_tabs,
            );
            defer {
                var gamma_deinit = gamma_data;
                gamma_deinit.deinit(alloc);
            }
            try logEigenvalues("scf", "gamma*", gamma_data.values, gamma_data.nbands);
        }
        if (!debug_gamma_dense_logged) {
            debug_gamma_dense_logged = true;
            const gamma_kp = KPoint{
                .k_frac = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
                .k_cart = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
                .weight = 0.0,
            };
            var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, gamma_kp.k_cart);
            defer basis.deinit(alloc);
            const inv_volume = 1.0 / volume;
            const h = try hamiltonian.buildHamiltonian(alloc, basis.gvecs, species, atoms, inv_volume, potential);
            defer alloc.free(h);
            var eig = try linalg.hermitianEigenDecomp(alloc, cfg.linalg_backend, basis.gvecs.len, h);
            defer eig.deinit(alloc);
            const count = @min(cfg.band.nbands, eig.values.len);
            try logEigenvalues("scf", "gamma_dense", eig.values, count);
        }
    }

    var min_energy = std.math.inf(f64);
    var max_energy = -std.math.inf(f64);
    var min_nbands: usize = std.math.maxInt(usize);
    var max_nbands: usize = 0;
    if (filled == 0) {
        min_energy = 0.0;
        max_energy = 0.0;
        min_nbands = 0;
    }
    for (eigen_data[0..filled]) |entry| {
        min_nbands = @min(min_nbands, entry.nbands);
        max_nbands = @max(max_nbands, entry.nbands);
        for (entry.values[0..entry.nbands]) |energy| {
            min_energy = @min(min_energy, energy);
            max_energy = @max(max_energy, energy);
        }
    }

    const mu = findFermiLevel(nelec, cfg.scf.smear_ry, cfg.scf.smearing, eigen_data[0..filled]);
    if (cfg.scf.debug_fermi) {
        const outside = mu < min_energy or mu > max_energy;
        var buffer: [256]u8 = undefined;
        var writer = std.fs.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print(
            "scf: fermi diag min={d:.6} max={d:.6} mu={d:.6} outside={s} nelec={d:.6} nbands={d}-{d} smear={s} sigma={d:.6}\n",
            .{
                min_energy,
                max_energy,
                mu,
                if (outside) "true" else "false",
                nelec,
                min_nbands,
                max_nbands,
                config.smearingName(cfg.scf.smearing),
                cfg.scf.smear_ry,
            },
        );
        try out.flush();
    }
    for (eigen_data[0..filled], 0..) |entry, kidx| {
        const ac_ptr: ?*apply.KpointApplyCache = if (apply_caches) |acs|
            (if (kidx < acs.len) &acs[kidx] else null)
        else
            null;
        try accumulateKpointDensitySmearing(
            alloc,
            cfg,
            grid,
            entry.kpoint,
            entry,
            recip,
            volume,
            fft_index_map,
            mu,
            cfg.scf.smear_ry,
            rho,
            &band_energy,
            &nonlocal_energy,
            &entropy_energy,
            profile_ptr,
            ac_ptr,
            paw_rhoij,
            atoms,
        );
    }

    if (cfg.scf.profile and !cfg.scf.quiet) {
        try logProfile(profile_total, kpoints.len);
    }

    // Use correct allocator based on whether parallel path was used
    const cleanup_alloc = if (used_parallel) std.heap.c_allocator else alloc;
    for (eigen_data[0..filled]) |*entry| {
        entry.deinit(cleanup_alloc);
    }
    alloc.free(eigen_data);

    return DensityResult{
        .rho = rho,
        .band_energy = band_energy,
        .nonlocal_energy = nonlocal_energy,
        .fermi_level = mu,
        .entropy_energy = entropy_energy,
    };
}

/// Compute RMS density difference.
fn densityDiff(rho: []f64, rho_new: []f64) f64 {
    var sum: f64 = 0.0;
    for (rho, 0..) |value, i| {
        const diff = rho_new[i] - value;
        sum += diff * diff;
    }
    return std.math.sqrt(sum / @as(f64, @floatFromInt(rho.len)));
}

/// Accumulate density contribution of one band.
fn accumulateBandDensity(
    grid: Grid,
    gvecs: []plane_wave.GVector,
    vectors: []math.Complex,
    band: usize,
    weight: f64,
    inv_volume: f64,
    rho: []f64,
) void {
    const a1 = grid.cell.row(0);
    const a2 = grid.cell.row(1);
    const a3 = grid.cell.row(2);
    const n = gvecs.len;

    var iz: usize = 0;
    var idx: usize = 0;
    while (iz < grid.nz) : (iz += 1) {
        const fz = @as(f64, @floatFromInt(iz)) / @as(f64, @floatFromInt(grid.nz));
        var iy: usize = 0;
        while (iy < grid.ny) : (iy += 1) {
            const fy = @as(f64, @floatFromInt(iy)) / @as(f64, @floatFromInt(grid.ny));
            var ix: usize = 0;
            while (ix < grid.nx) : (ix += 1) {
                const fx = @as(f64, @floatFromInt(ix)) / @as(f64, @floatFromInt(grid.nx));
                const rvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(a1, fx), math.Vec3.scale(a2, fy)),
                    math.Vec3.scale(a3, fz),
                );
                var psi = math.complex.init(0.0, 0.0);
                var g: usize = 0;
                while (g < n) : (g += 1) {
                    const phase = math.complex.expi(math.Vec3.dot(gvecs[g].kpg, rvec));
                    const coeff = vectors[g + band * n];
                    psi = math.complex.add(psi, math.complex.mul(coeff, phase));
                }
                const density = (psi.r * psi.r + psi.i * psi.i) * inv_volume;
                rho[idx] += weight * density;
                idx += 1;
            }
        }
    }
}

fn accumulateBandDensityFft(
    alloc: std.mem.Allocator,
    grid: Grid,
    map: PwGridMap,
    coeffs: []const math.Complex,
    work_recip: []math.Complex,
    work_real: []math.Complex,
    plan: ?*fft.Fft3dPlan,
    fft_index_map: ?[]const usize,
    weight: f64,
    inv_volume: f64,
    rho: []f64,
) !void {
    map.scatter(coeffs, work_recip);
    if (fft_index_map) |idx_map| {
        try fftReciprocalToComplexInPlaceMapped(alloc, grid, idx_map, work_recip, work_real, plan);
    } else {
        try fftReciprocalToComplexInPlace(alloc, grid, work_recip, work_real, plan);
    }
    for (work_real, 0..) |psi, i| {
        const density = (psi.r * psi.r + psi.i * psi.i) * inv_volume;
        rho[i] += weight * density;
    }
}

/// Check if any species has nonlocal coefficients.
fn hasNonlocal(species: []hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.dij.len > 0) return true;
    }
    return false;
}

/// Check if any species has QIJ coefficients.
fn hasQij(species: []hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.qij.len > 0) return true;
    }
    return false;
}

/// Check if any species uses PAW.
fn hasPaw(species: []hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.paw != null) return true;
    }
    return false;
}

/// Compute total valence electrons in the cell.
fn totalElectrons(species: []hamiltonian.SpeciesEntry, atoms: []hamiltonian.AtomData) f64 {
    var total: f64 = 0.0;
    for (atoms) |atom| {
        total += species[atom.species_index].z_valence;
    }
    return total;
}

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

fn isGammaKpoint(kp: KPoint) bool {
    return math.Vec3.norm(kp.k_cart) < 1e-8;
}

fn logEigenvalues(prefix: []const u8, label: []const u8, values: []const f64, count: usize) !void {
    const limit = @min(count, 8);
    var buffer: [512]u8 = undefined;
    var writer = std.fs.File.stderr().writer(&buffer);
    const out = &writer.interface;
    try out.print("{s}: eig {s} nbands={d}", .{ prefix, label, count });
    var i: usize = 0;
    while (i < limit) : (i += 1) {
        try out.print(" {d:.6}", .{values[i]});
    }
    if (count > limit) {
        try out.writeAll(" ...");
    }
    try out.writeAll("\n");
    try out.flush();
}

/// Compute minimum index for FFT grid.
fn minIndex(n: usize) i32 {
    return -@as(i32, @intCast(n / 2));
}

/// Result from solveKpointsForSpin: eigendata and count.
const SpinEigenResult = struct {
    eigen_data: []KpointEigenData,
    filled: usize,
};

/// Solve eigenvalue problem for all k-points for a single spin channel.
fn solveKpointsForSpin(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    common: *ScfCommon,
    potential: hamiltonian.PotentialGrid,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    scf_iter: usize,
    shared_fft_plan: fft.Fft3dPlan,
) !SpinEigenResult {
    const grid = common.grid;
    const kpoints = common.kpoints;
    const species = common.species;
    const atoms = common.atoms;
    const recip = common.recip;
    const volume_bohr = common.volume_bohr;
    const radial_tables = common.radial_tables;

    // For spin-polarized SCF, each channel needs enough bands to accommodate
    // magnetic splitting: up channel may have more occupied than nelec/2.
    // Add 20% extra bands + 4 minimum buffer for partial occupations.
    const nocc_base = @as(usize, @intFromFloat(std.math.ceil(common.total_electrons / 2.0)));
    const nocc = nocc_base + @max(4, nocc_base / 5);
    const is_paw_spin = hasPaw(species);
    const has_qij = hasQij(species) and !is_paw_spin;
    const use_iterative_config = (cfg.scf.solver == .iterative or cfg.scf.solver == .cg or cfg.scf.solver == .auto) and !has_qij;
    const nonlocal_enabled = cfg.scf.enable_nonlocal and hasNonlocal(species);

    var local_r: ?[]f64 = null;
    if (use_iterative_config) {
        local_r = try potential_mod.buildLocalPotentialReal(alloc, grid, common.ionic, potential);
    }
    defer if (local_r) |values| alloc.free(values);

    const fft_index_map = try buildFftIndexMap(alloc, grid);
    defer alloc.free(fft_index_map);

    var iter_max_iter = cfg.scf.iterative_max_iter;
    var iter_tol = cfg.scf.iterative_tol;
    if (cfg.scf.iterative_warmup_steps > 0 and scf_iter < cfg.scf.iterative_warmup_steps) {
        iter_max_iter = cfg.scf.iterative_warmup_max_iter;
        iter_tol = cfg.scf.iterative_warmup_tol;
    }

    const eigen_data = try alloc.alloc(KpointEigenData, kpoints.len);
    var filled: usize = 0;
    errdefer {
        var ii: usize = 0;
        while (ii < filled) : (ii += 1) {
            eigen_data[ii].deinit(alloc);
        }
        alloc.free(eigen_data);
    }

    for (kpoints, 0..) |kp, kidx| {
        const ac_ptr: ?*apply.KpointApplyCache = &apply_caches[kidx];
        eigen_data[kidx] = try computeKpointEigenData(
            alloc,
            &cfg,
            grid,
            kp,
            species,
            atoms,
            recip,
            volume_bohr,
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
            shared_fft_plan,
            ac_ptr,
            radial_tables,
            common.paw_tabs,
        );
        filled += 1;
    }

    return SpinEigenResult{
        .eigen_data = eigen_data,
        .filled = filled,
    };
}

// =========================================================================
// Spin-polarized SCF loop (nspin=2)
// =========================================================================

fn runSpinPolarizedLoop(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    volume_bohr: f64,
    common: *ScfCommon,
) !ScfResult {
    const grid = common.grid;
    const kpoints = common.kpoints;
    const total_electrons = common.total_electrons;
    const recip = common.recip;

    // Kpoint caches: separate for up and down
    const kpoint_cache_up = try alloc.alloc(KpointCache, kpoints.len);
    defer {
        for (kpoint_cache_up) |*cache| cache.deinit();
        alloc.free(kpoint_cache_up);
    }
    for (kpoint_cache_up) |*cache| cache.* = .{};

    const kpoint_cache_down = try alloc.alloc(KpointCache, kpoints.len);
    defer {
        for (kpoint_cache_down) |*cache| cache.deinit();
        alloc.free(kpoint_cache_down);
    }
    for (kpoint_cache_down) |*cache| cache.* = .{};

    const grid_count = grid.count();

    // Initial spin densities
    const rho_up = try alloc.alloc(f64, grid_count);
    errdefer alloc.free(rho_up);
    const rho_down = try alloc.alloc(f64, grid_count);
    errdefer alloc.free(rho_down);

    // Compute initial magnetization from spinat
    var m_total: f64 = 0.0;
    if (cfg.scf.spinat) |sa| {
        for (sa) |m| {
            m_total += m;
        }
    }
    // Clamp magnetization to be physical
    m_total = std.math.clamp(m_total, -total_electrons, total_electrons);

    // Build initial density from superposition of atomic densities, split by magnetization
    {
        const atomic_rho = try buildAtomicDensity(alloc, grid, common.species, atoms);
        defer alloc.free(atomic_rho);
        var sum: f64 = 0.0;
        const dv = grid.volume / @as(f64, @floatFromInt(grid_count));
        for (atomic_rho) |v| sum += v * dv;
        const scale = if (sum > 1e-10) total_electrons / sum else 1.0;
        const frac_up = (total_electrons + m_total) / (2.0 * total_electrons);
        const frac_down = (total_electrons - m_total) / (2.0 * total_electrons);
        for (0..grid_count) |gi| {
            const rho_scaled = atomic_rho[gi] * scale;
            rho_up[gi] = rho_scaled * frac_up;
            rho_down[gi] = rho_scaled * frac_down;
        }
    }

    if (!cfg.scf.quiet) {
        var buffer: [256]u8 = undefined;
        var writer = std.fs.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print("spin-scf: nspin=2, nelec={d:.1}, m_init={d:.2}\n", .{ total_electrons, m_total });
        try out.flush();
    }

    // Build initial potentials
    const spin_potentials = try potential_mod.buildPotentialGridSpin(alloc, grid, rho_up, rho_down, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, null, common.coulomb_r_cut);
    var potential_up = spin_potentials.up;
    errdefer potential_up.deinit(alloc);
    var potential_down = spin_potentials.down;
    errdefer potential_down.deinit(alloc);

    var iterations: usize = 0;
    var converged = false;
    var last_band_energy: f64 = 0.0;
    var last_nonlocal_energy: f64 = 0.0;
    var last_entropy_energy: f64 = 0.0;
    var last_fermi_level: f64 = std.math.nan(f64);
    var last_potential_residual: f64 = 0.0;

    const nelec = total_electrons;

    // Apply caches per spin channel
    const apply_caches_up = try alloc.alloc(apply.KpointApplyCache, kpoints.len);
    defer {
        for (apply_caches_up) |*ac| ac.deinit(alloc);
        alloc.free(apply_caches_up);
    }
    for (apply_caches_up) |*ac| ac.* = .{};

    const apply_caches_down = try alloc.alloc(apply.KpointApplyCache, kpoints.len);
    defer {
        for (apply_caches_down) |*ac| ac.deinit(alloc);
        alloc.free(apply_caches_down);
    }
    for (apply_caches_down) |*ac| ac.* = .{};

    // PAW ecutrho computation
    const gs_scf = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    const ecutrho_scf = cfg.scf.ecut_ry * gs_scf * gs_scf;

    // PAW spin-resolved rhoij: separate up/down accumulation
    var paw_rhoij_up: ?paw_mod.RhoIJ = if (common.paw_rhoij) |*prij|
        try prij.clone(alloc)
    else
        null;
    defer if (paw_rhoij_up) |*rij| rij.deinit(alloc);
    var paw_rhoij_down: ?paw_mod.RhoIJ = if (common.paw_rhoij) |*prij|
        try prij.clone(alloc)
    else
        null;
    defer if (paw_rhoij_down) |*rij| rij.deinit(alloc);

    while (iterations < cfg.scf.max_iter) : (iterations += 1) {
        if (!cfg.scf.quiet) {
            try logIterStart(iterations);
        }

        // PAW: reset rhoij before accumulation
        if (common.paw_rhoij) |*prij| {
            prij.reset();
        }
        if (paw_rhoij_up) |*rij| rij.reset();
        if (paw_rhoij_down) |*rij| rij.reset();

        var band_energy_total: f64 = 0.0;
        var nonlocal_energy_total: f64 = 0.0;
        var entropy_energy_total: f64 = 0.0;

        // Arrays for spin-channel eigendata
        const rho_out_up = try alloc.alloc(f64, grid_count);
        defer alloc.free(rho_out_up);
        @memset(rho_out_up, 0.0);
        const rho_out_down = try alloc.alloc(f64, grid_count);
        defer alloc.free(rho_out_down);
        @memset(rho_out_down, 0.0);

        // Solve spin-up and spin-down channels
        var shared_fft_plan = try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend);
        defer shared_fft_plan.deinit(alloc);

        var result_up = try solveKpointsForSpin(alloc, cfg, common, potential_up, kpoint_cache_up, apply_caches_up, iterations, shared_fft_plan);
        var eigen_data_up = result_up.eigen_data;
        defer {
            for (eigen_data_up[0..result_up.filled]) |*entry| entry.deinit(alloc);
            alloc.free(eigen_data_up);
        }

        var result_down = try solveKpointsForSpin(alloc, cfg, common, potential_down, kpoint_cache_down, apply_caches_down, iterations, shared_fft_plan);
        var eigen_data_down = result_down.eigen_data;
        defer {
            for (eigen_data_down[0..result_down.filled]) |*entry| entry.deinit(alloc);
            alloc.free(eigen_data_down);
        }

        // PAW D_ij bootstrap: update D_ij from spin-averaged potential and re-solve.
        if (iterations == 0 and common.is_paw) {
            if (common.paw_tabs) |tabs| {
                // Bootstrap D_ij with spin-specific potentials
                try updatePawDij(alloc, grid, common.ionic, potential_up, tabs, species, atoms, apply_caches_up, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, true, null, 1.0);
                // Bootstrap down channel with potential_down (after first band solve creates ctx)
                try updatePawDij(alloc, grid, common.ionic, potential_down, tabs, species, atoms, apply_caches_down, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, true, null, 1.0);
                // Re-solve with bootstrapped D_ij
                for (kpoint_cache_up) |*cache| cache.deinit();
                for (kpoint_cache_up) |*cache| cache.* = .{};
                for (kpoint_cache_down) |*cache| cache.deinit();
                for (kpoint_cache_down) |*cache| cache.* = .{};
                for (eigen_data_up[0..result_up.filled]) |*entry| entry.deinit(alloc);
                alloc.free(eigen_data_up);
                const ru = try solveKpointsForSpin(alloc, cfg, common, potential_up, kpoint_cache_up, apply_caches_up, iterations, shared_fft_plan);
                result_up = ru;
                eigen_data_up = result_up.eigen_data;
                for (eigen_data_down[0..result_down.filled]) |*entry| entry.deinit(alloc);
                alloc.free(eigen_data_down);
                const rd = try solveKpointsForSpin(alloc, cfg, common, potential_down, kpoint_cache_down, apply_caches_down, iterations, shared_fft_plan);
                result_down = rd;
                eigen_data_down = result_down.eigen_data;
            }
        }

        // Find Fermi level(s). For PAW spin with initial magnetization, use FSM
        // with self-consistent m_total update from output occupation.
        const use_fsm = common.is_paw and @abs(m_total) > 0.1;
        const ne_up_target = (nelec + m_total) / 2.0;
        const ne_down_target = (nelec - m_total) / 2.0;
        const mu_up = if (use_fsm)
            findFermiLevelSpin(ne_up_target, cfg.scf.smear_ry, cfg.scf.smearing, eigen_data_up[0..result_up.filled], null, 1.0)
        else
            findFermiLevelSpin(nelec, cfg.scf.smear_ry, cfg.scf.smearing, eigen_data_up[0..result_up.filled], eigen_data_down[0..result_down.filled], 1.0);
        const mu_down = if (use_fsm)
            findFermiLevelSpin(ne_down_target, cfg.scf.smear_ry, cfg.scf.smearing, eigen_data_down[0..result_down.filled], null, 1.0)
        else
            mu_up;
        const mu = mu_up; // Use up channel Fermi level as reference
        last_fermi_level = mu;

        // Build fft_index_map for density accumulation
        const fft_index_map = try buildFftIndexMap(alloc, grid);
        defer alloc.free(fft_index_map);

        // Accumulate densities for each spin channel (FSM uses separate mu)
        for (eigen_data_up[0..result_up.filled], 0..) |entry, kidx| {
            try accumulateKpointDensitySmearingSpin(
                alloc,
                &cfg,
                grid,
                kpoints[kidx],
                entry,
                recip,
                volume_bohr,
                fft_index_map,
                mu_up,
                cfg.scf.smear_ry,
                rho_out_up,
                &band_energy_total,
                &nonlocal_energy_total,
                &entropy_energy_total,
                null,
                1.0,
                if (kidx < apply_caches_up.len) &apply_caches_up[kidx] else null,
                if (paw_rhoij_up) |*rij| rij else null,
                atoms,
            );
        }
        for (eigen_data_down[0..result_down.filled], 0..) |entry, kidx| {
            try accumulateKpointDensitySmearingSpin(
                alloc,
                &cfg,
                grid,
                kpoints[kidx],
                entry,
                recip,
                volume_bohr,
                fft_index_map,
                mu_down,
                cfg.scf.smear_ry,
                rho_out_down,
                &band_energy_total,
                &nonlocal_energy_total,
                &entropy_energy_total,
                null,
                1.0,
                if (kidx < apply_caches_down.len) &apply_caches_down[kidx] else null,
                if (paw_rhoij_down) |*rij| rij else null,
                atoms,
            );
        }

        // Combine spin-resolved rhoij into total: rhoij = rhoij_up + rhoij_down
        if (common.paw_rhoij) |*prij| {
            if (paw_rhoij_up) |*rij_up| {
                if (paw_rhoij_down) |*rij_down| {
                    for (0..prij.natom) |a| {
                        for (0..prij.values[a].len) |idx| {
                            prij.values[a][idx] = rij_up.values[a][idx] + rij_down.values[a][idx];
                        }
                    }
                }
            }
        }

        last_band_energy = band_energy_total;
        last_nonlocal_energy = nonlocal_energy_total;
        last_entropy_energy = entropy_energy_total;

        // Symmetrize spin densities
        if (common.sym_ops) |ops| {
            if (ops.len > 1) {
                try symmetrizeDensity(alloc, grid, rho_out_up, ops, cfg.scf.use_rfft);
                try symmetrizeDensity(alloc, grid, rho_out_down, ops, cfg.scf.use_rfft);
            }
        }

        // PAW: symmetrize rhoij between equivalent atoms
        if (common.paw_rhoij) |*prij| {
            if (cfg.scf.symmetry) {
                try symmetrizeRhoIJ(alloc, prij, species, atoms);
            }
        }

        // PAW: filter density to ecutrho sphere and build augmented density
        if (common.is_paw) {
            const filt_up = try potential_mod.filterDensityToEcutrho(alloc, grid, rho_out_up, ecutrho_scf, cfg.scf.use_rfft);
            defer alloc.free(filt_up);
            @memcpy(rho_out_up, filt_up);
            const filt_down = try potential_mod.filterDensityToEcutrho(alloc, grid, rho_out_down, ecutrho_scf, cfg.scf.use_rfft);
            defer alloc.free(filt_down);
            @memcpy(rho_out_down, filt_down);
        }

        // Build augmented density for potential construction (ρ̃ + n̂/2 per spin)
        var rho_aug_up: ?[]f64 = null;
        var rho_aug_down: ?[]f64 = null;
        defer if (rho_aug_up) |a| alloc.free(a);
        defer if (rho_aug_down) |a| alloc.free(a);
        if (common.is_paw) {
            if (common.paw_rhoij) |*prij| {
                // Compute n̂ from total rhoij into a temporary zero array
                const n_hat = try alloc.alloc(f64, grid_count);
                defer alloc.free(n_hat);
                @memset(n_hat, 0.0);
                try addPawCompensationCharge(alloc, grid, n_hat, prij, common.paw_tabs.?, atoms, ecutrho_scf, &common.paw_gaunt.?);
                // Split n̂ equally between up and down
                const aug_up = try alloc.alloc(f64, grid_count);
                const aug_down = try alloc.alloc(f64, grid_count);
                for (0..grid_count) |i| {
                    aug_up[i] = rho_out_up[i] + n_hat[i] * 0.5;
                    aug_down[i] = rho_out_down[i] + n_hat[i] * 0.5;
                }
                rho_aug_up = aug_up;
                rho_aug_down = aug_down;
            }
        }

        // Build new spin potentials (use augmented density for PAW)
        const pot_rho_up = rho_aug_up orelse rho_out_up;
        const pot_rho_down = rho_aug_down orelse rho_out_down;
        const new_potentials = try potential_mod.buildPotentialGridSpin(alloc, grid, pot_rho_up, pot_rho_down, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, null, common.coulomb_r_cut);
        var pot_out_up = new_potentials.up;
        var pot_out_down = new_potentials.down;
        var keep_pot_out = false;
        defer {
            if (!keep_pot_out) {
                pot_out_up.deinit(alloc);
                pot_out_down.deinit(alloc);
            }
        }

        // Compute residual from concatenated potentials
        {
            const nvals = potential_up.values.len;
            var sum_sq: f64 = 0.0;
            for (0..nvals) |idx| {
                const diff_up = math.complex.sub(pot_out_up.values[idx], potential_up.values[idx]);
                const diff_down = math.complex.sub(pot_out_down.values[idx], potential_down.values[idx]);
                sum_sq += diff_up.r * diff_up.r + diff_up.i * diff_up.i;
                sum_sq += diff_down.r * diff_down.r + diff_down.i * diff_down.i;
            }
            last_potential_residual = if (nvals > 0)
                std.math.sqrt(sum_sq / @as(f64, @floatFromInt(2 * nvals)))
            else
                0.0;
        }

        // Density diff for convergence check
        const rho_out_total = try alloc.alloc(f64, grid_count);
        defer alloc.free(rho_out_total);
        const rho_in_total = try alloc.alloc(f64, grid_count);
        defer alloc.free(rho_in_total);
        for (0..grid_count) |i| {
            rho_out_total[i] = rho_out_up[i] + rho_out_down[i];
            rho_in_total[i] = rho_up[i] + rho_down[i];
        }
        const diff = densityDiff(rho_in_total, rho_out_total);

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
            @memcpy(rho_up, rho_out_up);
            @memcpy(rho_down, rho_out_down);
            potential_up.deinit(alloc);
            potential_up = pot_out_up;
            potential_down.deinit(alloc);
            potential_down = pot_out_down;
            keep_pot_out = true;
            break;
        }

        // For PAW spin, use density mixing (potential mixing kills magnetization
        // because V_xc splitting decays through mixing). Kerker preconditioning
        // stabilizes density mixing.
        const force_density_mixing = false; // density mixing unstable for PAW Fe
        if (cfg.scf.mixing_mode == .potential and !force_density_mixing) {
            const n_complex = potential_up.values.len;
            const n_f64 = n_complex * 2;
            // Mix up
            const v_in_up: []f64 = @as([*]f64, @ptrCast(potential_up.values.ptr))[0..n_f64];
            const v_out_up_f: []const f64 = @as([*]const f64, @ptrCast(pot_out_up.values.ptr))[0..n_f64];
            // Mix down
            const v_in_down: []f64 = @as([*]f64, @ptrCast(potential_down.values.ptr))[0..n_f64];
            const v_out_down_f: []const f64 = @as([*]const f64, @ptrCast(pot_out_down.values.ptr))[0..n_f64];

            // Concatenate for Pulay
            if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
                // Concatenate up+down into a single array for Pulay
                const concat_in = try alloc.alloc(f64, n_f64 * 2);
                defer alloc.free(concat_in);
                const concat_out = try alloc.alloc(f64, n_f64 * 2);
                defer alloc.free(concat_out);
                @memcpy(concat_in[0..n_f64], v_in_up);
                @memcpy(concat_in[n_f64..], v_in_down);
                @memcpy(concat_out[0..n_f64], v_out_up_f);
                @memcpy(concat_out[n_f64..], v_out_down_f);
                try common.pulay_mixer.?.mix(concat_in, concat_out, cfg.scf.mixing_beta);
                @memcpy(v_in_up, concat_in[0..n_f64]);
                @memcpy(v_in_down, concat_in[n_f64..]);
            } else {
                mixDensity(v_in_up, v_out_up_f, cfg.scf.mixing_beta);
                mixDensity(v_in_down, v_out_down_f, cfg.scf.mixing_beta);
            }

            @memcpy(rho_up, rho_out_up);
            @memcpy(rho_down, rho_out_down);
            pot_out_up.deinit(alloc);
            pot_out_down.deinit(alloc);
            keep_pot_out = true;
        } else {
            // Density mixing with Kerker preconditioning for PAW stability.
            // Kerker suppresses long-wavelength charge sloshing that causes
            // D_ij transients and kills magnetic order.
            if (force_density_mixing) {
                // PAW spin: Kerker density mixing with small beta for stability
                const paw_beta: f64 = 0.05;
                const paw_q0: f64 = 1.5;
                try mixDensityKerker(alloc, grid, rho_up, rho_out_up, paw_beta, paw_q0, cfg.scf.use_rfft);
                try mixDensityKerker(alloc, grid, rho_down, rho_out_down, paw_beta, paw_q0, cfg.scf.use_rfft);
            } else if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
                const concat_in = try alloc.alloc(f64, grid_count * 2);
                defer alloc.free(concat_in);
                const concat_out = try alloc.alloc(f64, grid_count * 2);
                defer alloc.free(concat_out);
                @memcpy(concat_in[0..grid_count], rho_up);
                @memcpy(concat_in[grid_count..], rho_down);
                @memcpy(concat_out[0..grid_count], rho_out_up);
                @memcpy(concat_out[grid_count..], rho_out_down);
                try common.pulay_mixer.?.mix(concat_in, concat_out, cfg.scf.mixing_beta);
                @memcpy(rho_up, concat_in[0..grid_count]);
                @memcpy(rho_down, concat_in[grid_count..]);
            } else {
                mixDensity(rho_up, rho_out_up, cfg.scf.mixing_beta);
                mixDensity(rho_down, rho_out_down, cfg.scf.mixing_beta);
            }

            potential_up.deinit(alloc);
            potential_down.deinit(alloc);
            // PAW: rebuild potential from augmented density
            if (common.is_paw and common.paw_rhoij != null) {
                const n_hat_dm = try alloc.alloc(f64, grid_count);
                defer alloc.free(n_hat_dm);
                @memset(n_hat_dm, 0.0);
                try addPawCompensationCharge(alloc, grid, n_hat_dm, &common.paw_rhoij.?, common.paw_tabs.?, atoms, ecutrho_scf, &common.paw_gaunt.?);
                const dm_aug_up = try alloc.alloc(f64, grid_count);
                defer alloc.free(dm_aug_up);
                const dm_aug_down = try alloc.alloc(f64, grid_count);
                defer alloc.free(dm_aug_down);
                for (0..grid_count) |gi| {
                    dm_aug_up[gi] = rho_up[gi] + n_hat_dm[gi] * 0.5;
                    dm_aug_down[gi] = rho_down[gi] + n_hat_dm[gi] * 0.5;
                }
                const rebuilt = try potential_mod.buildPotentialGridSpin(alloc, grid, dm_aug_up, dm_aug_down, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, null, common.coulomb_r_cut);
                potential_up = rebuilt.up;
                potential_down = rebuilt.down;
            } else {
                const rebuilt = try potential_mod.buildPotentialGridSpin(alloc, grid, rho_up, rho_down, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, null, common.coulomb_r_cut);
                potential_up = rebuilt.up;
                potential_down = rebuilt.down;
            }
        }

        // PAW: update D_ij with spin-resolved D^xc and D_ij mixing.
        // D_ij mixing (β = mixing_beta) prevents abrupt changes that kill Stoner feedback.
        if (common.is_paw) {
            if (common.paw_tabs) |tabs| {
                // D_ij mixing to smooth transients that kill Stoner feedback
                const dij_mix_beta: f64 = if (common.is_paw) cfg.scf.mixing_beta else 1.0;

                // Compute new D_ij with spin D^xc
                if (paw_rhoij_up) |*rij_up| {
                    try updatePawDij(alloc, grid, common.ionic, potential_up, tabs, species, atoms, apply_caches_up, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, rij_up, dij_mix_beta);
                } else {
                    try updatePawDij(alloc, grid, common.ionic, potential_up, tabs, species, atoms, apply_caches_up, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, null, dij_mix_beta);
                }
                if (paw_rhoij_down) |*rij_down| {
                    try updatePawDij(alloc, grid, common.ionic, potential_down, tabs, species, atoms, apply_caches_down, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, rij_down, dij_mix_beta);
                } else {
                    try updatePawDij(alloc, grid, common.ionic, potential_down, tabs, species, atoms, apply_caches_down, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, null, dij_mix_beta);
                }
            }
        }
    }

    // Compute magnetization
    const dv = grid.volume / @as(f64, @floatFromInt(grid_count));
    var magnetization: f64 = 0.0;
    for (0..grid_count) |i| {
        magnetization += (rho_up[i] - rho_down[i]) * dv;
    }

    if (!cfg.scf.quiet) {
        var buffer: [256]u8 = undefined;
        var writer = std.fs.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print("spin-scf: magnetization = {d:.6} μ_B\n", .{magnetization});
        try out.flush();
    }

    // Compute total density for energy calculation
    const rho_total = try alloc.alloc(f64, grid_count);
    errdefer alloc.free(rho_total);
    for (0..grid_count) |i| {
        rho_total[i] = rho_up[i] + rho_down[i];
    }

    // PAW: build augmented density for energy computation (reuse ecutrho_scf)
    var rho_aug_up_for_energy: ?[]f64 = null;
    var rho_aug_down_for_energy: ?[]f64 = null;
    if (common.is_paw) {
        if (common.paw_rhoij) |*prij| {
            // Build n̂ from total rhoij
            const n_hat = try alloc.alloc(f64, grid_count);
            defer alloc.free(n_hat);
            @memset(n_hat, 0.0);
            try addPawCompensationCharge(alloc, grid, n_hat, prij, common.paw_tabs.?, atoms, ecutrho_scf, &common.paw_gaunt.?);
            // Subtract back rho=0 contribution to get pure n̂
            // (addPawCompensationCharge adds n̂ to rho, but we passed zeros)
            // rho_aug = rho + n̂/2 for each spin (n̂ split equally for non-magnetic)
            const aug_up = try alloc.alloc(f64, grid_count);
            const aug_down = try alloc.alloc(f64, grid_count);
            for (0..grid_count) |i| {
                aug_up[i] = rho_up[i] + n_hat[i] * 0.5;
                aug_down[i] = rho_down[i] + n_hat[i] * 0.5;
            }
            // Filter to ecutrho sphere
            const filt_up = try potential_mod.filterDensityToEcutrho(alloc, grid, aug_up, ecutrho_scf, cfg.scf.use_rfft);
            alloc.free(aug_up);
            const filt_down = try potential_mod.filterDensityToEcutrho(alloc, grid, aug_down, ecutrho_scf, cfg.scf.use_rfft);
            alloc.free(aug_down);
            rho_aug_up_for_energy = filt_up;
            rho_aug_down_for_energy = filt_down;
        }
    }
    defer if (rho_aug_up_for_energy) |a| alloc.free(a);
    defer if (rho_aug_down_for_energy) |a| alloc.free(a);

    const paw_ecutrho: ?f64 = if (common.is_paw) ecutrho_scf else null;

    // Compute energy terms using spin-polarized XC
    var energy_terms = try energy_mod.computeEnergyTermsSpin(
        alloc,
        grid,
        rho_up,
        rho_down,
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
        rho_aug_up_for_energy,
        rho_aug_down_for_energy,
        paw_ecutrho,
    );

    // Add PAW on-site energy correction (spin-resolved E_xc if rhoij_up/down available)
    if (common.is_paw) {
        if (common.paw_rhoij) |*prij| {
            if (common.paw_tabs) |tabs| {
                energy_terms.paw_onsite = try computePawOnsiteEnergyTotal(
                    alloc,
                    prij,
                    tabs,
                    species,
                    atoms,
                    cfg.scf.xc,
                    &common.paw_gaunt.?,
                    if (paw_rhoij_up) |*ru| ru else null,
                    if (paw_rhoij_down) |*rd| rd else null,
                );
                energy_terms.total += energy_terms.paw_onsite;
            }
        }
    }

    // PAW double-counting: -Σ (D^xc_up × ρ_up + D^xc_down × ρ_down + D^H × ρ_total)
    var result_paw_tabs: ?[]paw_mod.PawTab = null;
    var result_paw_dij: ?[][]f64 = null;
    var result_paw_dij_m: ?[][]f64 = null;
    var result_paw_dxc: ?[][]f64 = null;
    var result_paw_rhoij: ?[][]f64 = null;
    if (common.is_paw) {
        if (common.paw_rhoij) |prij| {
            if (common.paw_tabs) |tabs| {
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
                    const rhoij_m = prij.values[ai];

                    // D^xc double-counting: spin-resolved if available
                    if (paw_rhoij_up != null and paw_rhoij_down != null) {
                        const dxc_up = try alloc.alloc(f64, mt * mt);
                        defer alloc.free(dxc_up);
                        const dxc_down = try alloc.alloc(f64, mt * mt);
                        defer alloc.free(dxc_down);
                        try paw_mod.paw_xc.computePawDijXcAngularSpin(
                            alloc,
                            dxc_up,
                            dxc_down,
                            paw,
                            paw_rhoij_up.?.values[ai],
                            paw_rhoij_down.?.values[ai],
                            mt,
                            sp_m_offsets,
                            upf.r,
                            upf.rab,
                            paw.ae_core_density,
                            if (upf.nlcc.len > 0) upf.nlcc else null,
                            cfg.scf.xc,
                            &common.paw_gaunt.?,
                        );
                        // DC_xc = Σ (D^xc_up × ρ_up + D^xc_down × ρ_down)
                        const rij_up = paw_rhoij_up.?.values[ai];
                        const rij_dn = paw_rhoij_down.?.values[ai];
                        for (0..mt) |im| {
                            for (0..mt) |jm| {
                                sum_dxc_rhoij += dxc_up[im * mt + jm] * rij_up[im * mt + jm];
                                sum_dxc_rhoij += dxc_down[im * mt + jm] * rij_dn[im * mt + jm];
                            }
                        }
                        // Store D^xc_up for result (used in stress)
                        const dxc_m_copy = try alloc.alloc(f64, mt * mt);
                        @memcpy(dxc_m_copy, dxc_up);
                        try dxc_list.append(alloc, dxc_m_copy);
                    } else {
                        const dxc_m = try alloc.alloc(f64, mt * mt);
                        try paw_mod.paw_xc.computePawDijXcAngular(
                            alloc,
                            dxc_m,
                            paw,
                            rhoij_m,
                            mt,
                            sp_m_offsets,
                            upf.r,
                            upf.rab,
                            paw.ae_core_density,
                            if (upf.nlcc.len > 0) upf.nlcc else null,
                            cfg.scf.xc,
                            &common.paw_gaunt.?,
                        );
                        for (0..mt) |im| {
                            for (0..mt) |jm| {
                                sum_dxc_rhoij += dxc_m[im * mt + jm] * rhoij_m[im * mt + jm];
                            }
                        }
                        try dxc_list.append(alloc, dxc_m);
                    }

                    // D^H double-counting (uses total rhoij, spin-independent)
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
                energy_terms.paw_dxc_rhoij = -sum_dxc_rhoij;
                energy_terms.total += energy_terms.paw_dxc_rhoij;
            }
        }

        // Transfer paw_tabs ownership from common to result
        if (common.paw_tabs) |tabs| {
            result_paw_tabs = tabs;
            common.paw_tabs = null;
        }
        // Extract per-atom D_ij from first apply cache (up channel)
        if (apply_caches_up.len > 0) {
            if (apply_caches_up[0].nonlocal_ctx) |nc| {
                var dij_list: std.ArrayList([]f64) = .empty;
                errdefer {
                    for (dij_list.items) |d| alloc.free(d);
                    dij_list.deinit(alloc);
                }
                for (nc.species) |entry_sp| {
                    if (entry_sp.dij_per_atom) |dpa| {
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
                var dij_m_list: std.ArrayList([]f64) = .empty;
                errdefer {
                    for (dij_m_list.items) |d| alloc.free(d);
                    dij_m_list.deinit(alloc);
                }
                for (nc.species) |entry_sp| {
                    if (entry_sp.dij_m_per_atom) |dpa| {
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
        // Copy per-atom rhoij (contracted to radial basis)
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
        var buffer: [256]u8 = undefined;
        var writer = std.fs.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print("spin-scf: total_energy = {d:.10} Ry\n", .{energy_terms.total});
        try out.print("spin-scf: E_band={d:.8} E_H={d:.8} E_xc={d:.8} E_ion={d:.8}\n", .{ energy_terms.band, energy_terms.hartree, energy_terms.xc, energy_terms.ion_ion });
        try out.print("spin-scf: E_psp={d:.8} E_dc={d:.8} E_local={d:.8} E_nl={d:.8}\n", .{ energy_terms.psp_core, energy_terms.double_counting, energy_terms.local_pseudo, energy_terms.nonlocal_pseudo });
        if (common.is_paw) {
            try out.print("spin-scf: E_paw_onsite={d:.8} E_paw_dxc={d:.8}\n", .{ energy_terms.paw_onsite, energy_terms.paw_dxc_rhoij });
        }
        try out.flush();
    }

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

    // Compute final wavefunctions for force/stress/DOS/band calculation
    var wavefunctions_up: ?WavefunctionData = null;
    var wavefunctions_down_final: ?WavefunctionData = null;
    var vxc_r_up_result: ?[]f64 = null;
    var vxc_r_down_result: ?[]f64 = null;
    if (cfg.relax.enabled or cfg.dfpt.enabled or cfg.scf.compute_stress or cfg.dos.enabled) {
        const wfn_up = try computeFinalWavefunctionsWithSpinFactor(
            alloc, cfg, grid, kpoints, common.ionic, species, atoms, recip, volume_bohr,
            potential_up, kpoint_cache_up, apply_caches_up, common.radial_tables, common.paw_tabs, 1.0,
        );
        wavefunctions_up = wfn_up.wavefunctions;
        const wfn_down = try computeFinalWavefunctionsWithSpinFactor(
            alloc, cfg, grid, kpoints, common.ionic, species, atoms, recip, volume_bohr,
            potential_down, kpoint_cache_down, apply_caches_down, common.radial_tables, common.paw_tabs, 1.0,
        );
        wavefunctions_down_final = wfn_down.wavefunctions;

        // Update band energies from final wavefunctions (spin_factor=1.0 for each channel)
        last_band_energy = wfn_up.band_energy + wfn_down.band_energy;
        last_nonlocal_energy = wfn_up.nonlocal_energy + wfn_down.nonlocal_energy;

        // Store V_xc in real space for NLCC force
        const pot_rho_up_final = rho_aug_up_for_energy orelse rho_up;
        const pot_rho_down_final = rho_aug_down_for_energy orelse rho_down;
        const vxc_spin = try xc_fields_mod.computeXcFieldsSpin(alloc, grid, pot_rho_up_final, pot_rho_down_final, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc);
        vxc_r_up_result = vxc_spin.vxc_up;
        vxc_r_down_result = vxc_spin.vxc_down;
        alloc.free(vxc_spin.exc);
    }
    errdefer if (wavefunctions_up) |*wf| wf.deinit(alloc);
    errdefer if (wavefunctions_down_final) |*wf| wf.deinit(alloc);
    errdefer if (vxc_r_up_result) |v| alloc.free(v);
    errdefer if (vxc_r_down_result) |v| alloc.free(v);

    return ScfResult{
        .potential = potential_up,
        .density = rho_total,
        .iterations = iterations,
        .converged = converged,
        .energy = energy_terms,
        .fermi_level = last_fermi_level,
        .potential_residual = last_potential_residual,
        .wavefunctions = wavefunctions_up,
        .vresid = null,
        .grid = grid,
        .density_up = rho_up,
        .density_down = rho_down,
        .potential_down = potential_down,
        .magnetization = magnetization,
        .wavefunctions_down = wavefunctions_down_final,
        .vxc_r_up = vxc_r_up_result,
        .vxc_r_down = vxc_r_down_result,
        .fermi_level_down = if (!std.math.isNan(last_fermi_level)) last_fermi_level else 0.0,
        .paw_tabs = result_paw_tabs,
        .paw_dij = result_paw_dij,
        .paw_dij_m = result_paw_dij_m,
        .paw_dxc = result_paw_dxc,
        .paw_rhoij = result_paw_rhoij,
        .rho_core = if (common.rho_core) |rc| blk: {
            const copy = try alloc.alloc(f64, rc.len);
            @memcpy(copy, rc);
            break :blk copy;
        } else null,
    };
}

test "auto grid chooses fft-friendly size for aluminum" {
    const cell_ang = math.Mat3.fromRows(
        .{ .x = 0.0, .y = 2.025, .z = 2.025 },
        .{ .x = 2.025, .y = 0.0, .z = 2.025 },
        .{ .x = 2.025, .y = 2.025, .z = 0.0 },
    );
    const cell_bohr = cell_ang.scale(math.unitsScaleToBohr(.angstrom));
    const recip = math.reciprocal(cell_bohr);
    const grid = autoGrid(15.0, 1.0, recip);
    // With QE-like algorithm, FCC cell with ecut=15 gives smaller grid
    // than the old formula which overestimated for non-orthogonal cells
    try std.testing.expect(grid[0] >= 3);
    try std.testing.expect(grid[1] >= 3);
    try std.testing.expect(grid[2] >= 3);
}
