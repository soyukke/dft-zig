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
const mergeProfile = logging.mergeProfile;

pub const DensityResult = struct {
    rho: []f64,
    band_energy: f64,
    nonlocal_energy: f64,
    fermi_level: f64,
    entropy_energy: f64 = 0.0, // -T*S term for smearing
};

pub fn smearingActive(cfg: *const config.Config) bool {
    return cfg.scf.smearing != .none and cfg.scf.smear_ry > 0.0;
}

var debug_gamma_dense_logged: bool = false;

fn isGammaKpoint(kp: KPoint) bool {
    return math.Vec3.norm(kp.k_cart) < 1e-8;
}

fn logEigenvalues(prefix: []const u8, label: []const u8, values: []const f64, count: usize) !void {
    const limit = @min(count, 8);
    var buffer: [512]u8 = undefined;
    var writer = std.Io.File.stderr().writer(&buffer);
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

pub fn computeDensitySmearing(
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
            var writer = std.Io.File.stderr().writer(&buffer);
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
        var writer = std.Io.File.stderr().writer(&buffer);
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
        try logProfile(params.io, profile_total, kpoints.len);
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
