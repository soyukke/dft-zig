const std = @import("std");
const apply = @import("apply.zig");
const config = @import("../config/config.zig");
const fermi_mod = @import("fermi_level.zig");
const fft = @import("../fft/fft.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoint = @import("kpoint.zig");
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
const KpointCache = kpoint.KpointCache;
const KpointEigenData = kpoint.KpointEigenData;
const SmearingShared = kpoint.workers.SmearingShared;
const SmearingWorker = kpoint.workers.SmearingWorker;
const ScfProfile = logging.ScfProfile;

const compute_kpoint_eigen_data = kpoint.solve.compute_kpoint_eigen_data;
const find_fermi_level = fermi_mod.find_fermi_level;
const accumulate_kpoint_density_smearing = kpoint.density.accumulate_kpoint_density_smearing;
const smearing_worker = kpoint.workers.smearing_worker;
const kpoint_thread_count = kpoint.workers.kpoint_thread_count;

const log_kpoint = logging.log_kpoint;
const log_profile = logging.log_profile;
const log_eigenvalues = logging.log_eigenvalues;
const log_local_potential_mean = logging.log_local_potential_mean;
const log_fermi_diag = logging.log_fermi_diag;
const merge_profile = logging.merge_profile;

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

pub const SmearingInput = struct {
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

pub fn smearing_active(cfg: *const config.Config) bool {
    return cfg.scf.smearing != .none and cfg.scf.smear_ry > 0.0;
}

var debug_gamma_dense_logged: bool = false;

fn is_gamma_kpoint(kp: KPoint) bool {
    return math.Vec3.norm(kp.k_cart) < 1e-8;
}

pub fn compute_density_smearing(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    nelec: f64,
) !DensityResult {
    var run = try init_smearing_run(alloc, input);
    errdefer alloc.free(run.rho);
    defer deinit_smearing_eigen_data(alloc, run.eigen_data, run.filled, run.used_parallel);
    defer alloc.free(run.eigen_data);

    try maybe_log_smearing_local_potential(
        run.input.io,
        run.input.cfg,
        run.input.potential,
        run.input.local_r,
    );
    const collected = try collect_smearing_eigen_data_from_input(
        alloc,
        run.input,
        &run.profile_total,
        run.eigen_data,
    );
    run.filled = collected.filled;
    run.used_parallel = collected.used_parallel;

    try maybe_log_smearing_gamma_diagnostics_from_input(
        alloc,
        run.input,
        run.eigen_data[0..run.filled],
    );

    const energy_range = compute_smearing_energy_range(run.eigen_data[0..run.filled]);
    const mu = find_fermi_level(
        nelec,
        input.cfg.scf.smear_ry,
        input.cfg.scf.smearing,
        run.eigen_data[0..run.filled],
    );
    try maybe_log_smearing_fermi_diag(input.io, input.cfg, nelec, mu, energy_range);
    try accumulate_smearing_density_results_from_input(
        alloc,
        run.input,
        mu,
        if (input.cfg.scf.profile) &run.profile_total else null,
        run.eigen_data[0..run.filled],
        run.rho,
        &run.band_energy,
        &run.nonlocal_energy,
        &run.entropy_energy,
    );

    if (input.cfg.scf.profile and !input.cfg.scf.quiet) {
        try log_profile(input.io, run.profile_total, input.kpoints.len);
    }

    return finish_smearing_result(
        run.rho,
        run.band_energy,
        run.nonlocal_energy,
        mu,
        run.entropy_energy,
    );
}

fn deinit_smearing_eigen_data(
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

fn init_smearing_run(
    alloc: std.mem.Allocator,
    input: SmearingInput,
) !SmearingRun {
    const rho = try alloc.alloc(f64, input.grid.count());
    errdefer alloc.free(rho);
    @memset(rho, 0.0);

    return .{
        .input = input,
        .rho = rho,
        .eigen_data = try alloc.alloc(KpointEigenData, input.kpoints.len),
    };
}

fn maybe_log_smearing_fermi_diag(
    io: std.Io,
    cfg: *const config.Config,
    nelec: f64,
    mu: f64,
    energy_range: SmearingEnergyRange,
) !void {
    if (!cfg.scf.debug_fermi) return;
    try log_fermi_diag(
        io,
        energy_range.min_energy,
        energy_range.max_energy,
        mu,
        nelec,
        energy_range.min_nbands,
        energy_range.max_nbands,
        config.smearing_name(cfg.scf.smearing),
        cfg.scf.smear_ry,
    );
}

fn finish_smearing_result(
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

fn collect_smearing_eigen_data_from_input(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    profile_total: *ScfProfile,
    eigen_data: []KpointEigenData,
) !SmearingCollectResult {
    return try collect_smearing_eigen_data(
        alloc,
        input,
        profile_total,
        eigen_data,
    );
}

fn maybe_log_smearing_gamma_diagnostics_from_input(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    eigen_data: []const KpointEigenData,
) !void {
    try maybe_log_smearing_gamma_diagnostics(
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

fn accumulate_smearing_density_results_from_input(
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
    try accumulate_smearing_density_results(
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

fn maybe_log_smearing_local_potential(
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
        const pot_g0 = potential.value_at(0, 0, 0);
        try log_local_potential_mean(io, "scf", mean_local, "pot_g0", pot_g0.r);
    }
}

fn collect_smearing_eigen_data(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    profile_total: *ScfProfile,
    eigen_data: []KpointEigenData,
) !SmearingCollectResult {
    const thread_count = kpoint_thread_count(input.kpoints.len, input.cfg.scf.kpoint_threads);
    if (thread_count <= 1) return try collect_smearing_eigen_data_single_thread(
        alloc,
        input,
        profile_total,
        eigen_data,
    );
    try collect_smearing_eigen_data_parallel(
        alloc,
        input,
        thread_count,
        profile_total,
        eigen_data,
    );
    return .{ .filled = input.kpoints.len, .used_parallel = true };
}

fn collect_smearing_eigen_data_single_thread(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    profile_total: *ScfProfile,
    eigen_data: []KpointEigenData,
) !SmearingCollectResult {
    return .{
        .filled = try collect_smearing_eigen_data_sequential(
            alloc,
            input,
            if (input.cfg.scf.profile) profile_total else null,
            eigen_data,
        ),
        .used_parallel = false,
    };
}

fn init_shared_smearing_fft_plan(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    cfg: *const config.Config,
) !fft.Fft3dPlan {
    return try fft.Fft3dPlan.init_with_backend(
        alloc,
        io,
        grid.nx,
        grid.ny,
        grid.nz,
        cfg.scf.fft_backend,
    );
}

fn collect_smearing_eigen_data_sequential(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    profile_ptr: ?*ScfProfile,
    eigen_data: []KpointEigenData,
) !usize {
    var shared_fft_plan = try init_shared_smearing_fft_plan(alloc, input.io, input.grid, input.cfg);
    defer shared_fft_plan.deinit(alloc);

    var filled: usize = 0;
    for (input.kpoints, 0..) |kp, kidx| {
        if (!input.cfg.scf.quiet) {
            try log_kpoint(input.io, kidx, input.kpoints.len);
        }
        eigen_data[kidx] = try compute_kpoint_eigen_data(
            alloc,
            input.io,
            input.cfg,
            input.grid,
            kp,
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
            input.cfg.scf.iterative_reuse_vectors,
            &input.kpoint_cache[kidx],
            profile_ptr,
            shared_fft_plan,
            select_apply_cache(input.apply_caches, kidx),
            input.radial_tables,
            input.paw_tabs,
        );
        filled += 1;
    }
    return filled;
}

fn collect_smearing_eigen_data_parallel(
    alloc: std.mem.Allocator,
    input: SmearingInput,
    thread_count: usize,
    profile_total: *ScfProfile,
    eigen_data: []KpointEigenData,
) !void {
    const profiles = try init_smearing_profiles(alloc, thread_count, input.cfg.scf.profile);
    defer if (profiles) |values| alloc.free(values);

    const fft_plans = try init_smearing_fft_plans(
        alloc,
        input.io,
        input.grid,
        input.cfg.scf.fft_backend,
        thread_count,
    );
    defer deinit_smearing_fft_plans(alloc, fft_plans);

    var next_index = std.atomic.Value(usize).init(0);
    var stop = std.atomic.Value(u8).init(0);
    var worker_error: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;
    var shared = SmearingShared{
        .io = input.io,
        .cfg = input.cfg,
        .grid = input.grid,
        .kpoints = input.kpoints,
        .species = input.species,
        .atoms = input.atoms,
        .recip = input.recip,
        .volume = input.volume,
        .local_cfg = input.local_cfg,
        .potential = input.potential,
        .local_r = input.local_r,
        .nocc = input.nocc,
        .use_iterative_config = input.use_iterative_config,
        .has_qij = input.has_qij,
        .nonlocal_enabled = input.nonlocal_enabled,
        .fft_index_map = input.fft_index_map,
        .iter_max_iter = input.iter_max_iter,
        .iter_tol = input.iter_tol,
        .reuse_vectors = input.cfg.scf.iterative_reuse_vectors,
        .kpoint_cache = input.kpoint_cache,
        .eigen_data = eigen_data,
        .fft_plans = fft_plans,
        .profiles = profiles,
        .apply_caches = input.apply_caches,
        .radial_tables = input.radial_tables,
        .paw_tabs = input.paw_tabs,
        .next_index = &next_index,
        .stop = &stop,
        .err = &worker_error,
        .err_mutex = &err_mutex,
        .log_mutex = &log_mutex,
    };
    try run_smearing_workers(alloc, thread_count, &shared);
    if (worker_error) |err| return err;
    if (profiles) |values| {
        for (values) |thread_profile| {
            merge_profile(profile_total, thread_profile);
        }
    }
}

fn init_smearing_profiles(
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

fn init_smearing_fft_plans(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    backend: config.FftBackend,
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
            backend,
        );
    }
    return fft_plans;
}

fn deinit_smearing_fft_plans(alloc: std.mem.Allocator, fft_plans: []fft.Fft3dPlan) void {
    for (fft_plans) |*plan| {
        plan.deinit(alloc);
    }
    alloc.free(fft_plans);
}

fn run_smearing_workers(
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
        threads[t] = try std.Thread.spawn(.{}, smearing_worker, .{&workers[t]});
    }
    for (threads) |thread| {
        thread.join();
    }
}

fn select_apply_cache(
    apply_caches: ?[]apply.KpointApplyCache,
    kidx: usize,
) ?*apply.KpointApplyCache {
    if (apply_caches) |caches| {
        if (kidx < caches.len) return &caches[kidx];
    }
    return null;
}

fn maybe_log_smearing_gamma_diagnostics(
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
    if (!(try log_gamma_eigenvalues_from_entries(io, eigen_data))) {
        try log_computed_gamma_eigenvalues(
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
        try log_dense_gamma_eigenvalues(
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

fn log_gamma_eigenvalues_from_entries(io: std.Io, eigen_data: []const KpointEigenData) !bool {
    for (eigen_data) |entry| {
        if (is_gamma_kpoint(entry.kpoint)) {
            try log_eigenvalues(io, "scf", "gamma", entry.values, entry.nbands);
            return true;
        }
    }
    return false;
}

fn build_gamma_kpoint() KPoint {
    return .{
        .k_frac = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .k_cart = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .weight = 0.0,
    };
}

fn log_computed_gamma_eigenvalues(
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

    const gamma_data = try compute_kpoint_eigen_data(
        alloc,
        io,
        cfg,
        grid,
        build_gamma_kpoint(),
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

    try log_eigenvalues(io, "scf", "gamma*", gamma_data.values, gamma_data.nbands);
}

fn log_dense_gamma_eigenvalues(
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
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, build_gamma_kpoint().k_cart);
    defer basis.deinit(alloc);

    const h = try hamiltonian.build_hamiltonian(
        alloc,
        basis.gvecs,
        species,
        atoms,
        1.0 / volume,
        local_cfg,
        potential,
    );
    defer alloc.free(h);

    var eig = try linalg.hermitian_eigen_decomp(
        alloc,
        cfg.linalg_backend,
        basis.gvecs.len,
        h,
    );
    defer eig.deinit(alloc);

    try log_eigenvalues(
        io,
        "scf",
        "gamma_dense",
        eig.values,
        @min(cfg.band.nbands, eig.values.len),
    );
}

fn compute_smearing_energy_range(eigen_data: []const KpointEigenData) SmearingEnergyRange {
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

fn accumulate_smearing_density_results(
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
        try accumulate_kpoint_density_smearing(
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
            select_apply_cache(apply_caches, kidx),
            paw_rhoij,
            atoms,
        );
    }
}
