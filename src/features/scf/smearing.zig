const std = @import("std");
const apply = @import("apply.zig");
const config = @import("../config/config.zig");
const fft = @import("../fft/fft.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoints_mod = @import("kpoint_parallel.zig");
const linalg = @import("../linalg/linalg.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const symmetry = @import("../symmetry/symmetry.zig");

const Grid = grid_mod.Grid;
const KPoint = symmetry.KPoint;
const KpointCache = kpoints_mod.KpointCache;
const KpointEigenData = kpoints_mod.KpointEigenData;
const SmearingShared = kpoints_mod.SmearingShared;
const SmearingWorker = kpoints_mod.SmearingWorker;
const ScfProfile = logging.ScfProfile;

const computeKpointEigenData = kpoints_mod.computeKpointEigenData;
const findFermiLevel = kpoints_mod.findFermiLevel;
const accumulateKpointDensitySmearing = kpoints_mod.accumulateKpointDensitySmearing;
const smearingWorker = kpoints_mod.smearingWorker;
const kpointThreadCount = kpoints_mod.kpointThreadCount;

const logKpoint = logging.logKpoint;
const logProfile = logging.logProfile;
const logEigenvalues = logging.logEigenvalues;
const logLocalPotentialMean = logging.logLocalPotentialMean;
const logFermiDiag = logging.logFermiDiag;
const mergeProfile = logging.mergeProfile;

pub const DensityResult = struct {
    rho: []f64,
    band_energy: f64,
    nonlocal_energy: f64,
    fermi_level: f64,
    entropy_energy: f64 = 0.0, // -T*S term for smearing
};

const SmearingCollectResult = struct {
    filled: usize,
    used_parallel: bool,
};

const SmearingEnergyRange = struct {
    min_energy: f64,
    max_energy: f64,
    min_nbands: usize,
    max_nbands: usize,
};

const SmearingInput = struct {
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
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
    local_cfg: local_potential.LocalPotentialConfig,
};

const SmearingRun = struct {
    input: SmearingInput,
    rho: []f64,
    eigen_data: []KpointEigenData,
    band_energy: f64 = 0.0,
    nonlocal_energy: f64 = 0.0,
    entropy_energy: f64 = 0.0,
    profile_total: ScfProfile = .{},
    filled: usize = 0,
    used_parallel: bool = false,
};

pub fn smearingActive(cfg: *const config.Config) bool {
    return cfg.scf.smearing != .none and cfg.scf.smear_ry > 0.0;
}

var debug_gamma_dense_logged: bool = false;

fn isGammaKpoint(kp: KPoint) bool {
    return math.Vec3.norm(kp.k_cart) < 1e-8;
}

pub fn computeDensitySmearing(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
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
    var run = try initSmearingRun(
        alloc,
        io,
        cfg,
        grid,
        kpoints,
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
        kpoint_cache,
        apply_caches,
        radial_tables,
        paw_tabs,
        paw_rhoij,
    );
    errdefer alloc.free(run.rho);
    defer deinitSmearingEigenData(alloc, run.eigen_data, run.filled, run.used_parallel);
    defer alloc.free(run.eigen_data);

    try maybeLogSmearingLocalPotential(
        run.input.io,
        run.input.cfg,
        run.input.potential,
        run.input.local_r,
    );
    const collected = try collectSmearingEigenDataFromInput(
        alloc,
        run.input,
        &run.profile_total,
        run.eigen_data,
    );
    run.filled = collected.filled;
    run.used_parallel = collected.used_parallel;

    try maybeLogSmearingGammaDiagnosticsFromInput(alloc, run.input, run.eigen_data[0..run.filled]);

    const energy_range = computeSmearingEnergyRange(run.eigen_data[0..run.filled]);
    const mu = findFermiLevel(
        nelec,
        cfg.scf.smear_ry,
        cfg.scf.smearing,
        run.eigen_data[0..run.filled],
    );
    try maybeLogSmearingFermiDiag(io, cfg, nelec, mu, energy_range);
    try accumulateSmearingDensityResultsFromInput(
        alloc,
        run.input,
        mu,
        if (cfg.scf.profile) &run.profile_total else null,
        run.eigen_data[0..run.filled],
        run.rho,
        &run.band_energy,
        &run.nonlocal_energy,
        &run.entropy_energy,
    );

    if (cfg.scf.profile and !cfg.scf.quiet) {
        try logProfile(io, run.profile_total, kpoints.len);
    }

    return finishSmearingResult(
        run.rho,
        run.band_energy,
        run.nonlocal_energy,
        mu,
        run.entropy_energy,
    );
}

fn deinitSmearingEigenData(
    alloc: std.mem.Allocator,
    eigen_data: []KpointEigenData,
    filled: usize,
    used_parallel: bool,
) void {
    const cleanup_alloc = if (used_parallel) std.heap.c_allocator else alloc;
    for (eigen_data[0..filled]) |*entry| {
        entry.deinit(cleanup_alloc);
    }
}

fn initSmearingRun(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
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
) !SmearingRun {
    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, grid.cell);
    const rho = try alloc.alloc(f64, grid.count());
    errdefer alloc.free(rho);
    @memset(rho, 0.0);

    return .{
        .input = buildSmearingInput(
            io,
            cfg,
            grid,
            kpoints,
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
            kpoint_cache,
            apply_caches,
            radial_tables,
            paw_tabs,
            paw_rhoij,
            local_cfg,
        ),
        .rho = rho,
        .eigen_data = try alloc.alloc(KpointEigenData, kpoints.len),
    };
}

fn buildSmearingInput(
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
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
    local_cfg: local_potential.LocalPotentialConfig,
) SmearingInput {
    return .{
        .io = io,
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
        .kpoint_cache = kpoint_cache,
        .apply_caches = apply_caches,
        .radial_tables = radial_tables,
        .paw_tabs = paw_tabs,
        .paw_rhoij = paw_rhoij,
        .local_cfg = local_cfg,
    };
}

fn maybeLogSmearingFermiDiag(
    io: std.Io,
    cfg: *const config.Config,
    nelec: f64,
    mu: f64,
    energy_range: SmearingEnergyRange,
) !void {
    if (!cfg.scf.debug_fermi) return;
    try logFermiDiag(
        io,
        energy_range.min_energy,
        energy_range.max_energy,
        mu,
        nelec,
        energy_range.min_nbands,
        energy_range.max_nbands,
        config.smearingName(cfg.scf.smearing),
        cfg.scf.smear_ry,
    );
}

fn finishSmearingResult(
    rho: []f64,
    band_energy: f64,
    nonlocal_energy: f64,
    fermi_level: f64,
    entropy_energy: f64,
) DensityResult {
    return .{
        .rho = rho,
        .band_energy = band_energy,
        .nonlocal_energy = nonlocal_energy,
        .fermi_level = fermi_level,
        .entropy_energy = entropy_energy,
    };
}

fn collectSmearingEigenDataFromInput(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    profile_total: *ScfProfile,
    eigen_data: []KpointEigenData,
) !SmearingCollectResult {
    return try collectSmearingEigenData(
        alloc,
        input.io,
        input.cfg,
        input.grid,
        input.kpoints,
        input.species,
        input.atoms,
        input.recip,
        input.volume,
        input.local_cfg,
        input.potential,
        input.local_r,
        input.nocc,
        input.use_iterative_config,
        input.has_qij,
        input.nonlocal_enabled,
        input.fft_index_map,
        input.iter_max_iter,
        input.iter_tol,
        input.kpoint_cache,
        input.apply_caches,
        input.radial_tables,
        input.paw_tabs,
        profile_total,
        eigen_data,
    );
}

fn maybeLogSmearingGammaDiagnosticsFromInput(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    eigen_data: []const KpointEigenData,
) !void {
    try maybeLogSmearingGammaDiagnostics(
        alloc,
        input.io,
        input.cfg,
        input.grid,
        input.species,
        input.atoms,
        input.recip,
        input.volume,
        input.local_cfg,
        input.potential,
        input.local_r,
        input.nocc,
        input.use_iterative_config,
        input.has_qij,
        input.nonlocal_enabled,
        input.fft_index_map,
        input.iter_max_iter,
        input.iter_tol,
        input.radial_tables,
        input.paw_tabs,
        eigen_data,
    );
}

fn accumulateSmearingDensityResultsFromInput(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    mu: f64,
    profile_ptr: ?*ScfProfile,
    eigen_data: []const KpointEigenData,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    entropy_energy: *f64,
) !void {
    try accumulateSmearingDensityResults(
        alloc,
        input.io,
        input.cfg,
        input.grid,
        input.recip,
        input.volume,
        input.fft_index_map,
        mu,
        input.apply_caches,
        input.paw_rhoij,
        input.atoms,
        profile_ptr,
        eigen_data,
        rho,
        band_energy,
        nonlocal_energy,
        entropy_energy,
    );
}

fn maybeLogSmearingLocalPotential(
    io: std.Io,
    cfg: *const config.Config,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
) !void {
    if (!cfg.scf.debug_fermi) return;
    if (local_r) |values| {
        var sum: f64 = 0.0;
        for (values) |value| {
            sum += value;
        }
        const mean_local = sum / @as(f64, @floatFromInt(values.len));
        const pot_g0 = potential.valueAt(0, 0, 0);
        try logLocalPotentialMean(io, "scf", mean_local, "pot_g0", pot_g0.r);
    }
}

fn collectSmearingEigenData(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
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
    profile_total: *ScfProfile,
    eigen_data: []KpointEigenData,
) !SmearingCollectResult {
    const thread_count = kpointThreadCount(kpoints.len, cfg.scf.kpoint_threads);
    if (thread_count <= 1) {
        return .{
            .filled = try collectSmearingEigenDataSequential(
                alloc,
                io,
                cfg,
                grid,
                kpoints,
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
                kpoint_cache,
                apply_caches,
                radial_tables,
                paw_tabs,
                if (cfg.scf.profile) profile_total else null,
                eigen_data,
            ),
            .used_parallel = false,
        };
    }
    try collectSmearingEigenDataParallel(
        alloc,
        io,
        cfg,
        grid,
        kpoints,
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
        kpoint_cache,
        apply_caches,
        radial_tables,
        paw_tabs,
        thread_count,
        profile_total,
        eigen_data,
    );
    return .{ .filled = kpoints.len, .used_parallel = true };
}

fn collectSmearingEigenDataSequential(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
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
    profile_ptr: ?*ScfProfile,
    eigen_data: []KpointEigenData,
) !usize {
    var shared_fft_plan = try fft.Fft3dPlan.initWithBackend(
        alloc,
        io,
        grid.nx,
        grid.ny,
        grid.nz,
        cfg.scf.fft_backend,
    );
    defer shared_fft_plan.deinit(alloc);

    var filled: usize = 0;
    for (kpoints, 0..) |kp, kidx| {
        if (!cfg.scf.quiet) {
            try logKpoint(io, kidx, kpoints.len);
        }
        eigen_data[kidx] = try computeKpointEigenData(
            alloc,
            io,
            cfg,
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
            profile_ptr,
            shared_fft_plan,
            selectApplyCache(apply_caches, kidx),
            radial_tables,
            paw_tabs,
        );
        filled += 1;
    }
    return filled;
}

fn collectSmearingEigenDataParallel(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
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
    thread_count: usize,
    profile_total: *ScfProfile,
    eigen_data: []KpointEigenData,
) !void {
    const profiles = try initSmearingProfiles(alloc, thread_count, cfg.scf.profile);
    defer if (profiles) |values| alloc.free(values);

    const fft_plans = try initSmearingFftPlans(alloc, io, grid, cfg.scf.fft_backend, thread_count);
    defer deinitSmearingFftPlans(alloc, fft_plans);

    var next_index = std.atomic.Value(usize).init(0);
    var stop = std.atomic.Value(u8).init(0);
    var worker_error: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;
    var shared = SmearingShared{
        .io = io,
        .cfg = cfg,
        .grid = grid,
        .kpoints = kpoints,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume = volume,
        .local_cfg = local_cfg,
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
    try runSmearingWorkers(alloc, thread_count, &shared);
    if (worker_error) |err| return err;
    if (profiles) |values| {
        for (values) |thread_profile| {
            mergeProfile(profile_total, thread_profile);
        }
    }
}

fn initSmearingProfiles(
    alloc: std.mem.Allocator,
    thread_count: usize,
    enabled: bool,
) !?[]ScfProfile {
    if (!enabled) return null;
    const profiles = try alloc.alloc(ScfProfile, thread_count);
    for (profiles) |*profile| {
        profile.* = ScfProfile{};
    }
    return profiles;
}

fn initSmearingFftPlans(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    backend: config.FftBackend,
    thread_count: usize,
) ![]fft.Fft3dPlan {
    const fft_plans = try alloc.alloc(fft.Fft3dPlan, thread_count);
    errdefer alloc.free(fft_plans);
    for (fft_plans) |*plan| {
        plan.* = try fft.Fft3dPlan.initWithBackend(
            alloc,
            io,
            grid.nx,
            grid.ny,
            grid.nz,
            backend,
        );
    }
    return fft_plans;
}

fn deinitSmearingFftPlans(alloc: std.mem.Allocator, fft_plans: []fft.Fft3dPlan) void {
    for (fft_plans) |*plan| {
        plan.deinit(alloc);
    }
    alloc.free(fft_plans);
}

fn runSmearingWorkers(
    alloc: std.mem.Allocator,
    thread_count: usize,
    shared: *SmearingShared,
) !void {
    const workers = try alloc.alloc(SmearingWorker, thread_count);
    defer alloc.free(workers);

    const threads = try alloc.alloc(std.Thread, thread_count);
    defer alloc.free(threads);

    var t: usize = 0;
    while (t < thread_count) : (t += 1) {
        workers[t] = .{ .shared = shared, .thread_index = t };
        threads[t] = try std.Thread.spawn(.{}, smearingWorker, .{&workers[t]});
    }
    for (threads) |thread| {
        thread.join();
    }
}

fn selectApplyCache(
    apply_caches: ?[]apply.KpointApplyCache,
    kidx: usize,
) ?*apply.KpointApplyCache {
    if (apply_caches) |caches| {
        if (kidx < caches.len) return &caches[kidx];
    }
    return null;
}

fn maybeLogSmearingGammaDiagnostics(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
    use_iterative_config: bool,
    has_qij: bool,
    nonlocal_enabled: bool,
    fft_index_map: ?[]const usize,
    iter_max_iter: usize,
    iter_tol: f64,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    eigen_data: []const KpointEigenData,
) !void {
    if (!cfg.scf.debug_fermi) return;
    if (!(try logGammaEigenvaluesFromEntries(io, eigen_data))) {
        try logComputedGammaEigenvalues(
            alloc,
            io,
            cfg,
            grid,
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
            radial_tables,
            paw_tabs,
        );
    }
    if (!debug_gamma_dense_logged) {
        debug_gamma_dense_logged = true;
        try logDenseGammaEigenvalues(
            alloc,
            io,
            cfg,
            recip,
            volume,
            local_cfg,
            potential,
            species,
            atoms,
        );
    }
}

fn logGammaEigenvaluesFromEntries(io: std.Io, eigen_data: []const KpointEigenData) !bool {
    for (eigen_data) |entry| {
        if (isGammaKpoint(entry.kpoint)) {
            try logEigenvalues(io, "scf", "gamma", entry.values, entry.nbands);
            return true;
        }
    }
    return false;
}

fn buildGammaKpoint() KPoint {
    return .{
        .k_frac = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .k_cart = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .weight = 0.0,
    };
}

fn logComputedGammaEigenvalues(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
    nocc: usize,
    use_iterative_config: bool,
    has_qij: bool,
    nonlocal_enabled: bool,
    fft_index_map: ?[]const usize,
    iter_max_iter: usize,
    iter_tol: f64,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
) !void {
    var gamma_cache = KpointCache{};
    defer gamma_cache.deinit();

    const gamma_data = try computeKpointEigenData(
        alloc,
        io,
        cfg,
        grid,
        buildGammaKpoint(),
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
        &gamma_cache,
        null,
        null,
        null,
        radial_tables,
        paw_tabs,
    );
    defer {
        var gamma_deinit = gamma_data;
        gamma_deinit.deinit(alloc);
    }

    try logEigenvalues(io, "scf", "gamma*", gamma_data.values, gamma_data.nbands);
}

fn logDenseGammaEigenvalues(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    recip: math.Mat3,
    volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    potential: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) !void {
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, buildGammaKpoint().k_cart);
    defer basis.deinit(alloc);

    const h = try hamiltonian.buildHamiltonian(
        alloc,
        basis.gvecs,
        species,
        atoms,
        1.0 / volume,
        local_cfg,
        potential,
    );
    defer alloc.free(h);

    var eig = try linalg.hermitianEigenDecomp(
        alloc,
        cfg.linalg_backend,
        basis.gvecs.len,
        h,
    );
    defer eig.deinit(alloc);

    try logEigenvalues(io, "scf", "gamma_dense", eig.values, @min(cfg.band.nbands, eig.values.len));
}

fn computeSmearingEnergyRange(eigen_data: []const KpointEigenData) SmearingEnergyRange {
    if (eigen_data.len == 0) {
        return .{
            .min_energy = 0.0,
            .max_energy = 0.0,
            .min_nbands = 0,
            .max_nbands = 0,
        };
    }
    var range = SmearingEnergyRange{
        .min_energy = std.math.inf(f64),
        .max_energy = -std.math.inf(f64),
        .min_nbands = std.math.maxInt(usize),
        .max_nbands = 0,
    };
    for (eigen_data) |entry| {
        range.min_nbands = @min(range.min_nbands, entry.nbands);
        range.max_nbands = @max(range.max_nbands, entry.nbands);
        for (entry.values[0..entry.nbands]) |energy| {
            range.min_energy = @min(range.min_energy, energy);
            range.max_energy = @max(range.max_energy, energy);
        }
    }
    return range;
}

fn accumulateSmearingDensityResults(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    recip: math.Mat3,
    volume: f64,
    fft_index_map: ?[]const usize,
    mu: f64,
    apply_caches: ?[]apply.KpointApplyCache,
    paw_rhoij: ?*paw_mod.RhoIJ,
    atoms: []const hamiltonian.AtomData,
    profile_ptr: ?*ScfProfile,
    eigen_data: []const KpointEigenData,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    entropy_energy: *f64,
) !void {
    for (eigen_data, 0..) |entry, kidx| {
        try accumulateKpointDensitySmearing(
            alloc,
            io,
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
            band_energy,
            nonlocal_energy,
            entropy_energy,
            profile_ptr,
            selectApplyCache(apply_caches, kidx),
            paw_rhoij,
            atoms,
        );
    }
}
