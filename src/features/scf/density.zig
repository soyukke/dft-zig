const std = @import("std");
const apply = @import("apply.zig");
const config = @import("../config/config.zig");
const fft = @import("../fft/fft.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoint = @import("kpoint.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const potential_mod = @import("potential.zig");
const smearing_mod = @import("smearing.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const util = @import("util.zig");

const Grid = grid_mod.Grid;
const KPoint = symmetry.KPoint;
const KpointCache = kpoint.KpointCache;
const KpointShared = kpoint.workers.KpointShared;
const KpointWorker = kpoint.workers.KpointWorker;

const check_hamiltonian_apply = apply.check_hamiltonian_apply;
const compute_kpoint_contribution = kpoint.solve.compute_kpoint_contribution;
const kpoint_thread_count = kpoint.workers.kpoint_thread_count;
const kpoint_worker = kpoint.workers.kpoint_worker;
const build_fft_index_map = fft_grid.build_fft_index_map;
const log_iterative_solver_disabled = logging.log_iterative_solver_disabled;
const log_kpoint = logging.log_kpoint;
const log_profile = logging.log_profile;
const log_nonlocal_diagnostics = logging.log_nonlocal_diagnostics;
const log_local_diagnostics = logging.log_local_diagnostics;
const merge_profile = logging.merge_profile;
const profile_start = logging.profile_start;
const profile_add = logging.profile_add;
const compute_density_smearing = smearing_mod.compute_density_smearing;
const smearing_active = smearing_mod.smearing_active;
const has_nonlocal = util.has_nonlocal;
const has_qij = util.has_qij;
const has_paw = util.has_paw;
const total_electrons = util.total_electrons;

pub const DensityResult = smearing_mod.DensityResult;
pub const ScfLoopProfile = logging.ScfLoopProfile;

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

pub const ComputeDensityParams = struct {
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
};

fn build_density_context(params: *const ComputeDensityParams) DensityContext {
    return .{
        .alloc = params.alloc,
        .io = params.io,
        .cfg = &params.cfg,
        .grid = params.grid,
        .kpoints = params.kpoints,
        .ionic = params.ionic,
        .species = params.species,
        .atoms = params.atoms,
        .recip = params.recip,
        .volume = params.volume,
        .potential = params.potential,
        .kpoint_cache = params.kpoint_cache,
        .apply_caches = params.apply_caches,
        .radial_tables = params.radial_tables,
        .paw_tabs = params.paw_tabs,
        .paw_rhoij = params.paw_rhoij,
    };
}

fn init_density_setup(params: *const ComputeDensityParams) !DensitySetup {
    if (params.paw_rhoij) |rij| rij.reset();
    const nelec = total_electrons(params.species, params.atoms);
    const qij_enabled = has_qij(params.species) and !has_paw(params.species);
    const use_iterative_config = (params.cfg.scf.solver == .iterative or
        params.cfg.scf.solver == .cg or
        params.cfg.scf.solver == .auto) and !qij_enabled;
    if ((params.cfg.scf.solver == .iterative or
        params.cfg.scf.solver == .cg or
        params.cfg.scf.solver == .auto) and
        !use_iterative_config and qij_enabled)
    {
        try log_iterative_solver_disabled(params.io, "QIJ present");
    }

    const t_lr = if (params.loop_profile != null) profile_start(params.io) else null;
    const local_r = if (use_iterative_config)
        try potential_mod.build_local_potential_real(
            params.alloc,
            params.grid,
            params.ionic,
            params.potential,
        )
    else
        null;
    if (params.loop_profile) |lp| profile_add(params.io, lp.build_local_r_ns, t_lr);

    const t_fm = if (params.loop_profile != null) profile_start(params.io) else null;
    const fft_index_map = try build_fft_index_map(params.alloc, params.grid);
    if (params.loop_profile) |lp| profile_add(params.io, lp.build_fft_map_ns, t_fm);

    return .{
        .nelec = nelec,
        .nocc = @as(usize, @intFromFloat(std.math.ceil(nelec / 2.0))),
        .has_qij = qij_enabled,
        .use_iterative_config = use_iterative_config,
        .nonlocal_enabled = params.cfg.scf.enable_nonlocal and has_nonlocal(params.species),
        .fft_index_map = fft_index_map,
        .local_r = local_r,
        .iter_max_iter = if (params.cfg.scf.iterative_warmup_steps > 0 and
            params.scf_iter < params.cfg.scf.iterative_warmup_steps)
            params.cfg.scf.iterative_warmup_max_iter
        else
            params.cfg.scf.iterative_max_iter,
        .iter_tol = if (params.cfg.scf.iterative_warmup_steps > 0 and
            params.scf_iter < params.cfg.scf.iterative_warmup_steps)
            params.cfg.scf.iterative_warmup_tol
        else
            params.cfg.scf.iterative_tol,
        .local_cfg = local_potential.resolve(
            params.cfg.scf.local_potential,
            params.cfg.ewald.alpha,
            params.grid.cell,
        ),
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
) !logging.ScfProfile {
    var profile_total = logging.ScfProfile{};
    var shared_fft_plan = try fft.Fft3dPlan.init_with_backend(
        ctx.alloc,
        ctx.io,
        ctx.grid.nx,
        ctx.grid.ny,
        ctx.grid.nz,
        ctx.cfg.scf.fft_backend,
    );
    defer shared_fft_plan.deinit(ctx.alloc);

    const profile_ptr: ?*logging.ScfProfile = if (ctx.cfg.scf.profile) &profile_total else null;
    for (ctx.kpoints, 0..) |kp, kidx| {
        if (!ctx.cfg.scf.quiet) try log_kpoint(ctx.io, kidx, ctx.kpoints.len);
        const ac_ptr: ?*apply.KpointApplyCache = if (ctx.apply_caches) |acs|
            (if (kidx < acs.len) &acs[kidx] else null)
        else
            null;
        try compute_kpoint_contribution(
            ctx.alloc,
            .{
                .io = ctx.io,
                .cfg = ctx.cfg,
                .grid = ctx.grid,
                .kp = kp,
                .species = ctx.species,
                .atoms = ctx.atoms,
                .recip = ctx.recip,
                .volume = ctx.volume,
                .local_cfg = setup.local_cfg,
                .potential = ctx.potential,
                .local_r = setup.local_r,
                .nocc = setup.nocc,
                .use_iterative_config = setup.use_iterative_config,
                .has_qij = setup.has_qij,
                .nonlocal_enabled = setup.nonlocal_enabled,
                .fft_index_map = setup.fft_index_map,
                .iter_max_iter = setup.iter_max_iter,
                .iter_tol = setup.iter_tol,
                .reuse_vectors = ctx.cfg.scf.iterative_reuse_vectors,
                .cache = &ctx.kpoint_cache[kidx],
                .profile_ptr = profile_ptr,
                .shared_fft_plan = shared_fft_plan,
                .apply_cache = ac_ptr,
                .radial_tables = ctx.radial_tables,
                .paw_tabs = ctx.paw_tabs,
                .loose_init = false,
            },
            setup.nelec,
            rho,
            band_energy,
            nonlocal_energy,
            ctx.paw_rhoij,
        );
    }
    return profile_total;
}

const DensityParallelState = struct {
    rho_locals: []f64,
    band_energies: []f64,
    nonlocal_energies: []f64,
    profiles: ?[]logging.ScfProfile,
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
) !?[]logging.ScfProfile {
    if (!cfg.scf.profile) return null;
    const profiles = try alloc.alloc(logging.ScfProfile, thread_count);
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
    var initialized: usize = 0;
    errdefer {
        for (fft_plans[0..initialized]) |*plan| plan.deinit(alloc);
        alloc.free(fft_plans);
    }
    for (fft_plans) |*plan| {
        plan.* = try fft.Fft3dPlan.init_with_backend(
            alloc,
            io,
            grid.nx,
            grid.ny,
            grid.nz,
            cfg.scf.fft_backend,
        );
        initialized += 1;
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
    profile_total: *logging.ScfProfile,
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
    var profile_total = logging.ScfProfile{};
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
        .{
            .io = ctx.io,
            .cfg = ctx.cfg,
            .grid = ctx.grid,
            .kpoints = ctx.kpoints,
            .species = ctx.species,
            .atoms = ctx.atoms,
            .recip = ctx.recip,
            .volume = ctx.volume,
            .potential = ctx.potential,
            .local_r = setup.local_r,
            .nocc = setup.nocc,
            .use_iterative_config = setup.use_iterative_config,
            .has_qij = setup.has_qij,
            .nonlocal_enabled = setup.nonlocal_enabled,
            .fft_index_map = setup.fft_index_map,
            .iter_max_iter = setup.iter_max_iter,
            .iter_tol = setup.iter_tol,
            .kpoint_cache = ctx.kpoint_cache,
            .apply_caches = ctx.apply_caches,
            .radial_tables = ctx.radial_tables,
            .paw_tabs = ctx.paw_tabs,
            .paw_rhoij = ctx.paw_rhoij,
            .local_cfg = setup.local_cfg,
        },
        setup.nelec,
    );
}

pub fn compute_density(params: ComputeDensityParams) !DensityResult {
    const ctx = build_density_context(&params);
    var setup = try init_density_setup(&params);
    defer setup.deinit(params.alloc);

    try maybe_check_density_hamiltonian_apply(&ctx, &setup, params.scf_iter);
    try maybe_log_density_debug_diagnostics(&ctx, &setup, params.scf_iter);
    if (smearing_active(&params.cfg)) return try compute_density_with_smearing(&ctx, &setup);
    return try compute_density_no_smearing(&ctx, &setup);
}
