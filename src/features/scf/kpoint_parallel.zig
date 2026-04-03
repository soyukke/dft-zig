const std = @import("std");
const config = @import("../config/config.zig");
const fft = @import("../fft/fft.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const iterative = @import("../linalg/iterative.zig");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const apply = @import("apply.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const logging = @import("logging.zig");
const pw_grid_map = @import("pw_grid_map.zig");
const util = @import("util.zig");

/// Threshold for auto solver selection: basis size <= threshold uses dense, > threshold uses iterative.
/// Based on benchmarks, dense solver (LAPACK zheev) is faster for small matrices.
pub const auto_solver_threshold: usize = 400;

pub const Grid = grid_mod.Grid;
pub const KPoint = symmetry.KPoint;

const ApplyContext = apply.ApplyContext;
const applyHamiltonian = apply.applyHamiltonian;
const applyHamiltonianBatched = apply.applyHamiltonianBatched;
const applyNonlocalPotential = apply.applyNonlocalPotential;
const ScfProfile = logging.ScfProfile;
const logKpoint = logging.logKpoint;
const profileStart = logging.profileStart;
const profileAdd = logging.profileAdd;
const PwGridMap = pw_grid_map.PwGridMap;
const gridRequirement = util.gridRequirement;
const nextFftSize = util.nextFftSize;
const fftReciprocalToComplexInPlace = fft_grid.fftReciprocalToComplexInPlace;
const fftReciprocalToComplexInPlaceMapped = fft_grid.fftReciprocalToComplexInPlaceMapped;
const fftComplexToReciprocalInPlace = fft_grid.fftComplexToReciprocalInPlace;
const fftComplexToReciprocalInPlaceMapped = fft_grid.fftComplexToReciprocalInPlaceMapped;

pub const KpointCache = struct {
    n: usize = 0,
    nbands: usize = 0,
    vectors: []math.Complex = &[_]math.Complex{},
    eigenvalues: []f64 = &[_]f64{},

    pub fn deinit(self: *KpointCache) void {
        if (self.vectors.len > 0) {
            std.heap.c_allocator.free(self.vectors);
        }
        if (self.eigenvalues.len > 0) {
            std.heap.c_allocator.free(self.eigenvalues);
        }
        self.* = .{};
    }

    pub fn store(self: *KpointCache, n: usize, nbands: usize, values: []const math.Complex) !void {
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

    pub fn storeEigenvalues(self: *KpointCache, values: []const f64) !void {
        if (self.eigenvalues.len != values.len) {
            if (self.eigenvalues.len > 0) {
                std.heap.c_allocator.free(self.eigenvalues);
            }
            self.eigenvalues = try std.heap.c_allocator.alloc(f64, values.len);
        }
        @memcpy(self.eigenvalues, values);
    }
};

pub const KpointShared = struct {
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []const KPoint,
    ionic: hamiltonian.PotentialGrid,
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
    reuse_vectors: bool,
    rho_locals: []f64,
    band_energies: []f64,
    nonlocal_energies: []f64,
    profiles: ?[]ScfProfile,
    ngrid: usize,
    kpoint_cache: []KpointCache,
    apply_caches: ?[]apply.KpointApplyCache,
    fft_plans: []fft.Fft3dPlan, // Pre-created FFT plans per thread
    radial_tables: ?[]const nonlocal.RadialTableSet = null,
    paw_tabs: ?[]const paw_mod.PawTab = null,
    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Thread.Mutex,
    log_mutex: *std.Thread.Mutex,
};

pub const KpointWorker = struct {
    shared: *KpointShared,
    thread_index: usize,
};

fn setWorkerError(shared: *KpointShared, err: anyerror) void {
    shared.err_mutex.lock();
    defer shared.err_mutex.unlock();
    if (shared.err.* == null) {
        shared.err.* = err;
    }
}

pub fn kpointWorker(worker: *KpointWorker) void {
    const shared = worker.shared;
    const thread_index = worker.thread_index;
    const start = thread_index * shared.ngrid;
    const end = start + shared.ngrid;
    const rho_local = shared.rho_locals[start..end];

    var local_band: f64 = 0.0;
    var local_nonlocal: f64 = 0.0;
    var local_profile = ScfProfile{};
    const profile_ptr: ?*ScfProfile = if (shared.profiles != null) &local_profile else null;

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const idx = shared.next_index.fetchAdd(1, .acq_rel);
        if (idx >= shared.kpoints.len) break;

        if (!shared.cfg.scf.quiet) {
            shared.log_mutex.lock();
            logKpoint(idx, shared.kpoints.len) catch {};
            shared.log_mutex.unlock();
        }
        _ = arena.reset(.retain_capacity);
        const kalloc = arena.allocator();
        // Get thread's pre-created FFT plan to avoid mutex contention
        const thread_fft_plan: ?fft.Fft3dPlan = if (shared.fft_plans.len > thread_index)
            shared.fft_plans[thread_index]
        else
            null;
        computeKpointContribution(
            kalloc,
            shared.cfg,
            shared.grid,
            shared.kpoints[idx],
            shared.species,
            shared.atoms,
            shared.recip,
            shared.volume,
            shared.potential,
            shared.local_r,
            shared.nocc,
            shared.nelec,
            shared.use_iterative_config,
            shared.has_qij,
            shared.nonlocal_enabled,
            shared.fft_index_map,
            shared.iter_max_iter,
            shared.iter_tol,
            shared.reuse_vectors,
            &shared.kpoint_cache[idx],
            rho_local,
            &local_band,
            &local_nonlocal,
            profile_ptr,
            thread_fft_plan,
            if (shared.apply_caches) |acs| (if (idx < acs.len) &acs[idx] else null) else null,
            shared.radial_tables,
            shared.paw_tabs,
            null, // paw_rhoij: not thread-safe, handled in smearing path
        ) catch |err| {
            setWorkerError(shared, err);
            shared.stop.store(1, .release);
            break;
        };
    }

    shared.band_energies[thread_index] = local_band;
    shared.nonlocal_energies[thread_index] = local_nonlocal;
    if (shared.profiles) |profiles| {
        profiles[thread_index] = local_profile;
    }
}

pub fn kpointThreadCount(total: usize, cfg_threads: usize) usize {
    if (total <= 1) return 1;
    if (cfg_threads > 0) return @min(total, cfg_threads);
    const cpu_count = std.Thread.getCpuCount() catch 1;
    if (cpu_count == 0) return 1;
    return @min(total, cpu_count);
}

/// Shared data for parallel smearing eigendata computation
pub const SmearingShared = struct {
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []const KPoint,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
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
    reuse_vectors: bool,
    kpoint_cache: []KpointCache,
    eigen_data: []KpointEigenData,
    fft_plans: []fft.Fft3dPlan,
    profiles: ?[]ScfProfile,
    apply_caches: ?[]apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal.RadialTableSet = null,
    paw_tabs: ?[]const paw_mod.PawTab = null,
    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Thread.Mutex,
    log_mutex: *std.Thread.Mutex,
};

pub const SmearingWorker = struct {
    shared: *SmearingShared,
    thread_index: usize,
};

fn setSmearingWorkerError(shared: *SmearingShared, err: anyerror) void {
    shared.err_mutex.lock();
    defer shared.err_mutex.unlock();
    if (shared.err.* == null) {
        shared.err.* = err;
    }
}

pub fn smearingWorker(worker: *SmearingWorker) void {
    const shared = worker.shared;
    const thread_index = worker.thread_index;

    var local_profile = ScfProfile{};
    const profile_ptr: ?*ScfProfile = if (shared.profiles != null) &local_profile else null;

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const idx = shared.next_index.fetchAdd(1, .acq_rel);
        if (idx >= shared.kpoints.len) break;

        if (!shared.cfg.scf.quiet) {
            shared.log_mutex.lock();
            logKpoint(idx, shared.kpoints.len) catch {};
            shared.log_mutex.unlock();
        }
        _ = arena.reset(.retain_capacity);
        const kalloc = arena.allocator();

        // Get thread's pre-created FFT plan to avoid mutex contention
        const thread_fft_plan: ?fft.Fft3dPlan = if (shared.fft_plans.len > thread_index)
            shared.fft_plans[thread_index]
        else
            null;

        const eigen_result = computeKpointEigenData(
            kalloc,
            shared.cfg,
            shared.grid,
            shared.kpoints[idx],
            shared.species,
            shared.atoms,
            shared.recip,
            shared.volume,
            shared.potential,
            shared.local_r,
            shared.nocc,
            shared.use_iterative_config,
            shared.has_qij,
            shared.nonlocal_enabled,
            shared.fft_index_map,
            shared.iter_max_iter,
            shared.iter_tol,
            shared.reuse_vectors,
            &shared.kpoint_cache[idx],
            profile_ptr,
            thread_fft_plan,
            if (shared.apply_caches) |acs| (if (idx < acs.len) &acs[idx] else null) else null,
            shared.radial_tables,
            shared.paw_tabs,
        ) catch |err| {
            setSmearingWorkerError(shared, err);
            shared.stop.store(1, .release);
            break;
        };

        // Copy eigendata to persistent storage (arena will be reset)
        const values = std.heap.c_allocator.alloc(f64, eigen_result.nbands) catch {
            setSmearingWorkerError(shared, error.OutOfMemory);
            shared.stop.store(1, .release);
            break;
        };
        @memcpy(values, eigen_result.values);

        const vectors = std.heap.c_allocator.alloc(math.Complex, eigen_result.vectors.len) catch {
            std.heap.c_allocator.free(values);
            setSmearingWorkerError(shared, error.OutOfMemory);
            shared.stop.store(1, .release);
            break;
        };
        @memcpy(vectors, eigen_result.vectors);

        var nonlocal_vals: ?[]f64 = null;
        if (eigen_result.nonlocal) |nl| {
            nonlocal_vals = std.heap.c_allocator.alloc(f64, nl.len) catch {
                std.heap.c_allocator.free(values);
                std.heap.c_allocator.free(vectors);
                setSmearingWorkerError(shared, error.OutOfMemory);
                shared.stop.store(1, .release);
                break;
            };
            @memcpy(nonlocal_vals.?, nl);
        }

        shared.eigen_data[idx] = .{
            .kpoint = shared.kpoints[idx],
            .basis_len = eigen_result.basis_len,
            .nbands = eigen_result.nbands,
            .values = values,
            .vectors = vectors,
            .nonlocal = nonlocal_vals,
        };
    }

    if (shared.profiles) |profiles| {
        profiles[thread_index] = local_profile;
    }
}

pub fn computeKpointContribution(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
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
    reuse_vectors: bool,
    cache: *KpointCache,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    profile_ptr: ?*ScfProfile,
    shared_fft_plan: ?fft.Fft3dPlan,
    apply_cache: ?*apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_rhoij: ?*paw_mod.RhoIJ,
) !void {
    const basis_start = if (profile_ptr != null) profileStart() else null;
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kp.k_cart);
    if (profile_ptr) |p| profileAdd(&p.basis_ns, basis_start);
    defer basis.deinit(alloc);

    const nbands_full = @min(@max(cfg.band.nbands, nocc), basis.gvecs.len);
    if (nocc > basis.gvecs.len) return error.InsufficientBands;
    const inv_volume = 1.0 / volume;

    var apply_ctx: ?ApplyContext = null;
    defer if (apply_ctx) |*ctx| ctx.deinit(alloc);

    var use_iterative = use_iterative_config;
    // Auto solver: use dense for small basis, iterative for large
    if (cfg.scf.solver == .auto and basis.gvecs.len <= auto_solver_threshold) {
        use_iterative = false;
    }
    if (use_iterative) {
        const req = gridRequirement(basis.gvecs);
        if (req.nx > grid.nx or req.ny > grid.ny or req.nz > grid.nz) {
            var buffer: [256]u8 = undefined;
            var writer = std.fs.File.stderr().writer(&buffer);
            const out = &writer.interface;
            try out.print(
                "scf: iterative grid too small (need >= {d},{d},{d}, suggest {d},{d},{d})\n",
                .{ req.nx, req.ny, req.nz, nextFftSize(req.nx), nextFftSize(req.ny), nextFftSize(req.nz) },
            );
            try out.flush();
            use_iterative = false;
        }
    }

    const use_cg = (cfg.scf.solver == .cg);
    // Always use nbands_full for iterative solver to capture partially occupied
    // bands in metals. Using only nocc misses states at the Fermi surface when
    // bands cross the Fermi level at some k-points (e.g. Al with smearing).
    const nbands = nbands_full;

    const vnl = if (!use_iterative and nonlocal_enabled) blk: {
        const vnl_start = if (profile_ptr != null) profileStart() else null;
        const mat = try hamiltonian.buildNonlocalMatrix(alloc, basis.gvecs, species, atoms, inv_volume);
        if (profile_ptr) |p| profileAdd(&p.vnl_build_ns, vnl_start);
        break :blk mat;
    } else null;
    defer if (vnl) |mat| alloc.free(mat);

    const eig_start = if (profile_ptr != null) profileStart() else null;
    var init_vectors: ?[]const math.Complex = null;
    var init_cols: usize = 0;
    if (use_iterative and reuse_vectors and cache.vectors.len > 0 and cache.n == basis.gvecs.len and cache.nbands == nbands) {
        init_vectors = cache.vectors;
        init_cols = cache.nbands;
    }

    var eig = if (use_iterative) blk: {
        const diag = try alloc.alloc(f64, basis.gvecs.len);
        defer alloc.free(diag);
        for (basis.gvecs, 0..) |g, i| {
            diag[i] = g.kinetic;
        }
        const local_values = local_r orelse return error.InvalidGrid;
        // Try to use cached NonlocalContext and PwGridMap (avoids expensive recomputation)
        const ctx = if (apply_cache) |ac| blk2: {
            if (!ac.isValid(basis.gvecs.len)) {
                // First call for this kpoint: build and cache using persistent allocator
                ac.nonlocal_ctx = if (nonlocal_enabled)
                    (if (paw_tabs) |tabs|
                        try apply.buildNonlocalContextPaw(ac.cache_alloc, species, basis.gvecs, radial_tables, tabs)
                    else if (radial_tables) |tables|
                        try apply.buildNonlocalContextWithTables(ac.cache_alloc, species, basis.gvecs, tables)
                    else
                        try apply.buildNonlocalContextPub(ac.cache_alloc, species, basis.gvecs))
                else
                    null;
                ac.map = try PwGridMap.init(ac.cache_alloc, basis.gvecs, grid);
                if (fft_index_map) |idx_map| {
                    try ac.map.?.buildFftIndices(ac.cache_alloc, idx_map);
                }
                ac.basis_len = basis.gvecs.len;
            }
            if (shared_fft_plan) |plan| {
                break :blk2 try ApplyContext.initWithCache(
                    alloc,
                    grid,
                    basis.gvecs,
                    local_values,
                    vnl,
                    ac.nonlocal_ctx,
                    ac.map.?,
                    atoms,
                    inv_volume,
                    profile_ptr,
                    fft_index_map,
                    plan,
                    false, // shared plan, not owned
                );
            } else {
                break :blk2 try ApplyContext.initWithCache(
                    alloc,
                    grid,
                    basis.gvecs,
                    local_values,
                    vnl,
                    ac.nonlocal_ctx,
                    ac.map.?,
                    atoms,
                    inv_volume,
                    profile_ptr,
                    fft_index_map,
                    try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend),
                    true, // new plan, we own it
                );
            }
        } else if (shared_fft_plan) |plan|
            try ApplyContext.initWithFftPlan(
                alloc,
                grid,
                basis.gvecs,
                local_values,
                vnl,
                species,
                atoms,
                inv_volume,
                nonlocal_enabled,
                profile_ptr,
                fft_index_map,
                plan,
            )
        else
            try ApplyContext.init(
                alloc,
                grid,
                basis.gvecs,
                local_values,
                vnl,
                species,
                atoms,
                inv_volume,
                nonlocal_enabled,
                profile_ptr,
                fft_index_map,
                cfg.scf.fft_backend,
            );
        apply_ctx = ctx;
        // For PAW, provide S operator for generalized eigenvalue problem
        const has_paw_overlap = if (ctx.nonlocal_ctx) |nc| nc.hasPawOverlap() else false;
        const op = iterative.Operator{
            .n = basis.gvecs.len,
            .ctx = &apply_ctx.?,
            .apply = applyHamiltonian,
            .apply_batch = applyHamiltonianBatched,
            .apply_s = if (has_paw_overlap) apply.applyOverlap else null,
        };
        const opts = iterative.Options{
            .max_iter = iter_max_iter,
            .tol = iter_tol,
            .max_subspace = cfg.scf.iterative_max_subspace,
            .block_size = cfg.scf.iterative_block_size,
            .init_diagonal = cfg.scf.iterative_init_diagonal,
            .init_vectors = init_vectors,
            .init_vectors_cols = init_cols,
        };
        break :blk if (use_cg)
            try iterative.hermitianEigenDecompCG(alloc, op, diag, nbands, opts)
        else
            try iterative.hermitianEigenDecompIterative(alloc, cfg.linalg_backend, op, diag, nbands, opts);
    } else if (has_qij) blk: {
        const h_start = if (profile_ptr != null) profileStart() else null;
        const h = try hamiltonian.buildHamiltonian(alloc, basis.gvecs, species, atoms, inv_volume, potential);
        if (profile_ptr) |p| profileAdd(&p.h_build_ns, h_start);
        defer alloc.free(h);
        const s_start = if (profile_ptr != null) profileStart() else null;
        const s = try hamiltonian.buildOverlapMatrix(alloc, basis.gvecs, species, atoms, inv_volume);
        if (profile_ptr) |p| profileAdd(&p.s_build_ns, s_start);
        defer alloc.free(s);
        break :blk try linalg.hermitianGenEigenDecomp(alloc, cfg.linalg_backend, basis.gvecs.len, h, s);
    } else blk: {
        const h_start = if (profile_ptr != null) profileStart() else null;
        const h = try hamiltonian.buildHamiltonian(alloc, basis.gvecs, species, atoms, inv_volume, potential);
        if (profile_ptr) |p| profileAdd(&p.h_build_ns, h_start);
        defer alloc.free(h);
        break :blk try linalg.hermitianEigenDecomp(alloc, cfg.linalg_backend, basis.gvecs.len, h);
    };
    defer eig.deinit(alloc);
    if (profile_ptr) |p| profileAdd(&p.eig_ns, eig_start);

    // FFT now supports arbitrary sizes via Bluestein's algorithm
    const use_fft_density = true;
    var density_map: ?PwGridMap = try PwGridMap.init(alloc, basis.gvecs, grid);
    const total = grid.count();
    const density_recip: ?[]math.Complex = try alloc.alloc(math.Complex, total);
    const density_real: ?[]math.Complex = try alloc.alloc(math.Complex, total);
    var density_plan: ?fft.Fft3dPlan = try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend);
    defer if (density_plan) |*plan| plan.deinit(alloc);
    defer if (density_map) |*map| map.deinit(alloc);
    defer if (density_recip) |buf| alloc.free(buf);
    defer if (density_real) |buf| alloc.free(buf);

    const occ_last = if (nocc == 0)
        0.0
    else
        nelec / 2.0 - @as(f64, @floatFromInt(nocc - 1));
    const spin_factor = 2.0;
    var band: usize = 0;
    while (band < nocc and band < nbands) : (band += 1) {
        const occ = if (band + 1 == nocc) occ_last else 1.0;
        if (occ <= 0.0) continue;
        band_energy.* += kp.weight * occ * spin_factor * eig.values[band];
        if (vnl) |mat| {
            const e_nl = bandNonlocalEnergy(basis.gvecs.len, mat, eig.vectors, band);
            nonlocal_energy.* += kp.weight * occ * spin_factor * e_nl;
        } else if (apply_ctx) |*ctx| {
            if (ctx.nonlocal_ctx != null) {
                const psi = eig.vectors[band * basis.gvecs.len .. (band + 1) * basis.gvecs.len];
                try applyNonlocalPotential(ctx, psi, ctx.work_vec);
                const e_nl = innerProduct(basis.gvecs.len, psi, ctx.work_vec).r;
                nonlocal_energy.* += kp.weight * occ * spin_factor * e_nl;
            }
        }
        const density_start = if (profile_ptr != null) profileStart() else null;
        if (use_fft_density) {
            const coeffs = eig.vectors[band * basis.gvecs.len .. (band + 1) * basis.gvecs.len];
            try accumulateBandDensityFft(
                alloc,
                grid,
                density_map.?,
                coeffs,
                density_recip.?,
                density_real.?,
                if (density_plan) |*plan| plan else null,
                fft_index_map,
                kp.weight * occ * spin_factor,
                inv_volume,
                rho,
            );
        } else {
            accumulateBandDensity(
                grid,
                basis.gvecs,
                eig.vectors,
                band,
                kp.weight * occ * spin_factor,
                inv_volume,
                rho,
            );
        }
        if (profile_ptr) |p| profileAdd(&p.density_ns, density_start);

        // PAW: accumulate rhoij for this band
        if (paw_rhoij) |rij| {
            if (apply_cache) |ac| {
                if (ac.nonlocal_ctx) |*nl| {
                    if (nl.has_paw) {
                        const coeffs_paw = eig.vectors[band * basis.gvecs.len .. (band + 1) * basis.gvecs.len];
                        try apply.accumulatePawRhoIJ(alloc, nl, basis.gvecs, atoms, coeffs_paw, kp.weight * occ * spin_factor, inv_volume, rij);
                    }
                }
            } else if (apply_ctx) |*ctx| {
                if (ctx.nonlocal_ctx) |*nl| {
                    if (nl.has_paw) {
                        const coeffs_paw = eig.vectors[band * basis.gvecs.len .. (band + 1) * basis.gvecs.len];
                        try apply.accumulatePawRhoIJ(alloc, nl, basis.gvecs, atoms, coeffs_paw, kp.weight * occ * spin_factor, inv_volume, rij);
                    }
                }
            }
        }
    }

    if (use_iterative and reuse_vectors) {
        try cache.store(basis.gvecs.len, nbands, eig.vectors);
    }
    // Always save eigenvalues for potential use in constructing WavefunctionData
    try cache.storeEigenvalues(eig.values[0..nbands]);
}

pub const KpointEigenData = struct {
    kpoint: KPoint,
    basis_len: usize,
    nbands: usize,
    values: []f64,
    vectors: []math.Complex,
    nonlocal: ?[]f64,

    pub fn deinit(self: *KpointEigenData, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
        if (self.vectors.len > 0) alloc.free(self.vectors);
        if (self.nonlocal) |values| alloc.free(values);
        self.* = undefined;
    }
};

pub fn computeKpointEigenData(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
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
    reuse_vectors: bool,
    cache: *KpointCache,
    profile_ptr: ?*ScfProfile,
    shared_fft_plan: ?fft.Fft3dPlan,
    apply_cache: ?*apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
) !KpointEigenData {
    const basis_start = if (profile_ptr != null) profileStart() else null;
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kp.k_cart);
    if (profile_ptr) |p| profileAdd(&p.basis_ns, basis_start);
    defer basis.deinit(alloc);

    const nbands_full = @min(@max(cfg.band.nbands, nocc), basis.gvecs.len);
    if (nocc > basis.gvecs.len) return error.InsufficientBands;
    const inv_volume = 1.0 / volume;

    var apply_ctx: ?ApplyContext = null;
    defer if (apply_ctx) |*ctx| ctx.deinit(alloc);

    var use_iterative = use_iterative_config;
    // Auto solver: use dense for small basis, iterative for large
    if (cfg.scf.solver == .auto and basis.gvecs.len <= auto_solver_threshold) {
        use_iterative = false;
    }
    if (use_iterative) {
        const req = gridRequirement(basis.gvecs);
        if (req.nx > grid.nx or req.ny > grid.ny or req.nz > grid.nz) {
            var buffer: [256]u8 = undefined;
            var writer = std.fs.File.stderr().writer(&buffer);
            const out = &writer.interface;
            try out.print(
                "scf: iterative grid too small (need >= {d},{d},{d}, suggest {d},{d},{d})\n",
                .{ req.nx, req.ny, req.nz, nextFftSize(req.nx), nextFftSize(req.ny), nextFftSize(req.nz) },
            );
            try out.flush();
            use_iterative = false;
        }
    }

    const use_cg = (cfg.scf.solver == .cg);
    // Always use nbands_full for both solvers to ensure partially occupied
    // bands at the Fermi surface are captured (important for metals).
    const nbands = nbands_full;

    const vnl = if (!use_iterative and nonlocal_enabled) blk: {
        const vnl_start = if (profile_ptr != null) profileStart() else null;
        const mat = try hamiltonian.buildNonlocalMatrix(alloc, basis.gvecs, species, atoms, inv_volume);
        if (profile_ptr) |p| profileAdd(&p.vnl_build_ns, vnl_start);
        break :blk mat;
    } else null;
    defer if (vnl) |mat| alloc.free(mat);

    const eig_start = if (profile_ptr != null) profileStart() else null;
    var init_vectors: ?[]const math.Complex = null;
    var init_cols: usize = 0;
    if (use_iterative and reuse_vectors and cache.vectors.len > 0 and cache.n == basis.gvecs.len and cache.nbands >= nbands) {
        init_vectors = cache.vectors;
        init_cols = nbands;
    }

    var eig = if (use_iterative) blk: {
        const diag = try alloc.alloc(f64, basis.gvecs.len);
        defer alloc.free(diag);
        for (basis.gvecs, 0..) |g, i| {
            diag[i] = g.kinetic;
        }
        const local_values = local_r orelse return error.InvalidGrid;
        // Try to use cached NonlocalContext and PwGridMap
        const ctx = if (apply_cache) |ac| blk2: {
            if (!ac.isValid(basis.gvecs.len)) {
                ac.nonlocal_ctx = if (nonlocal_enabled)
                    (if (paw_tabs) |tabs|
                        try apply.buildNonlocalContextPaw(ac.cache_alloc, species, basis.gvecs, radial_tables, tabs)
                    else if (radial_tables) |tables|
                        try apply.buildNonlocalContextWithTables(ac.cache_alloc, species, basis.gvecs, tables)
                    else
                        try apply.buildNonlocalContextPub(ac.cache_alloc, species, basis.gvecs))
                else
                    null;
                ac.map = try PwGridMap.init(ac.cache_alloc, basis.gvecs, grid);
                if (fft_index_map) |idx_map| {
                    try ac.map.?.buildFftIndices(ac.cache_alloc, idx_map);
                }
                ac.basis_len = basis.gvecs.len;
            }
            if (shared_fft_plan) |plan| {
                break :blk2 try ApplyContext.initWithCache(
                    alloc,
                    grid,
                    basis.gvecs,
                    local_values,
                    vnl,
                    ac.nonlocal_ctx,
                    ac.map.?,
                    atoms,
                    inv_volume,
                    profile_ptr,
                    fft_index_map,
                    plan,
                    false,
                );
            } else {
                break :blk2 try ApplyContext.initWithCache(
                    alloc,
                    grid,
                    basis.gvecs,
                    local_values,
                    vnl,
                    ac.nonlocal_ctx,
                    ac.map.?,
                    atoms,
                    inv_volume,
                    profile_ptr,
                    fft_index_map,
                    try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend),
                    true,
                );
            }
        } else if (shared_fft_plan) |plan|
            try ApplyContext.initWithFftPlan(
                alloc,
                grid,
                basis.gvecs,
                local_values,
                vnl,
                species,
                atoms,
                inv_volume,
                nonlocal_enabled,
                profile_ptr,
                fft_index_map,
                plan,
            )
        else
            try ApplyContext.init(
                alloc,
                grid,
                basis.gvecs,
                local_values,
                vnl,
                species,
                atoms,
                inv_volume,
                nonlocal_enabled,
                profile_ptr,
                fft_index_map,
                cfg.scf.fft_backend,
            );
        apply_ctx = ctx;
        // For PAW, provide S operator for generalized eigenvalue problem
        const has_paw_overlap = if (ctx.nonlocal_ctx) |nc| nc.hasPawOverlap() else false;
        const op = iterative.Operator{
            .n = basis.gvecs.len,
            .ctx = &apply_ctx.?,
            .apply = applyHamiltonian,
            .apply_batch = applyHamiltonianBatched,
            .apply_s = if (has_paw_overlap) apply.applyOverlap else null,
        };
        const opts = iterative.Options{
            .max_iter = iter_max_iter,
            .tol = iter_tol,
            .max_subspace = cfg.scf.iterative_max_subspace,
            .block_size = cfg.scf.iterative_block_size,
            .init_diagonal = cfg.scf.iterative_init_diagonal,
            .init_vectors = init_vectors,
            .init_vectors_cols = init_cols,
        };
        break :blk if (use_cg)
            try iterative.hermitianEigenDecompCG(alloc, op, diag, nbands, opts)
        else
            try iterative.hermitianEigenDecompIterative(alloc, cfg.linalg_backend, op, diag, nbands, opts);
    } else if (has_qij) blk: {
        const h_start = if (profile_ptr != null) profileStart() else null;
        const h = try hamiltonian.buildHamiltonian(alloc, basis.gvecs, species, atoms, inv_volume, potential);
        if (profile_ptr) |p| profileAdd(&p.h_build_ns, h_start);
        defer alloc.free(h);
        const s_start = if (profile_ptr != null) profileStart() else null;
        const s = try hamiltonian.buildOverlapMatrix(alloc, basis.gvecs, species, atoms, inv_volume);
        if (profile_ptr) |p| profileAdd(&p.s_build_ns, s_start);
        defer alloc.free(s);
        break :blk try linalg.hermitianGenEigenDecomp(alloc, cfg.linalg_backend, basis.gvecs.len, h, s);
    } else blk: {
        const h_start = if (profile_ptr != null) profileStart() else null;
        const h = try hamiltonian.buildHamiltonian(alloc, basis.gvecs, species, atoms, inv_volume, potential);
        if (profile_ptr) |p| profileAdd(&p.h_build_ns, h_start);
        defer alloc.free(h);
        break :blk try linalg.hermitianEigenDecomp(alloc, cfg.linalg_backend, basis.gvecs.len, h);
    };
    defer eig.deinit(alloc);
    if (profile_ptr) |p| profileAdd(&p.eig_ns, eig_start);

    const values = try alloc.alloc(f64, nbands);
    errdefer alloc.free(values);
    @memcpy(values, eig.values[0..nbands]);
    const vector_count = basis.gvecs.len * nbands;
    const vectors = try alloc.alloc(math.Complex, vector_count);
    errdefer alloc.free(vectors);
    @memcpy(vectors, eig.vectors[0..vector_count]);

    var nonlocal_band: ?[]f64 = null;
    if (nonlocal_enabled) {
        const entries = try alloc.alloc(f64, nbands);
        errdefer alloc.free(entries);
        @memset(entries, 0.0);
        if (vnl) |mat| {
            for (entries, 0..) |*value, band| {
                value.* = bandNonlocalEnergy(basis.gvecs.len, mat, vectors, band);
            }
        } else if (apply_ctx) |*ctx| {
            if (ctx.nonlocal_ctx != null) {
                var band: usize = 0;
                while (band < nbands) : (band += 1) {
                    const psi = vectors[band * basis.gvecs.len .. (band + 1) * basis.gvecs.len];
                    try applyNonlocalPotential(ctx, psi, ctx.work_vec);
                    entries[band] = innerProduct(basis.gvecs.len, psi, ctx.work_vec).r;
                }
            }
        }
        nonlocal_band = entries;
    }

    if (use_iterative and reuse_vectors and nbands >= cache.nbands) {
        try cache.store(basis.gvecs.len, nbands, vectors);
    }
    if (nbands >= cache.eigenvalues.len) {
        try cache.storeEigenvalues(values[0..nbands]);
    }

    return .{
        .kpoint = kp,
        .basis_len = basis.gvecs.len,
        .nbands = nbands,
        .values = values,
        .vectors = vectors,
        .nonlocal = nonlocal_band,
    };
}

pub fn accumulateKpointDensitySmearing(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    data: KpointEigenData,
    recip: math.Mat3,
    volume: f64,
    fft_index_map: ?[]const usize,
    mu: f64,
    sigma: f64,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    entropy_energy: *f64,
    profile_ptr: ?*ScfProfile,
    apply_cache: ?*apply.KpointApplyCache,
    paw_rhoij: ?*paw_mod.RhoIJ,
    atoms: ?[]const hamiltonian.AtomData,
) !void {
    return accumulateKpointDensitySmearingSpin(alloc, cfg, grid, kp, data, recip, volume, fft_index_map, mu, sigma, rho, band_energy, nonlocal_energy, entropy_energy, profile_ptr, 2.0, apply_cache, paw_rhoij, atoms);
}

/// Spin-factor parameterized version.
pub fn accumulateKpointDensitySmearingSpin(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    data: KpointEigenData,
    recip: math.Mat3,
    volume: f64,
    fft_index_map: ?[]const usize,
    mu: f64,
    sigma: f64,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    entropy_energy: *f64,
    profile_ptr: ?*ScfProfile,
    spin_factor: f64,
    apply_cache: ?*apply.KpointApplyCache,
    paw_rhoij: ?*paw_mod.RhoIJ,
    atoms: ?[]const hamiltonian.AtomData,
) !void {
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kp.k_cart);
    defer basis.deinit(alloc);
    if (basis.gvecs.len != data.basis_len) return error.InvalidBasis;

    const inv_volume = 1.0 / volume;

    // FFT now supports arbitrary sizes via Bluestein's algorithm
    const use_fft_density = true;
    var density_map: ?PwGridMap = try PwGridMap.init(alloc, basis.gvecs, grid);
    const total = grid.count();
    const density_recip: ?[]math.Complex = try alloc.alloc(math.Complex, total);
    const density_real: ?[]math.Complex = try alloc.alloc(math.Complex, total);
    var density_plan: ?fft.Fft3dPlan = try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend);
    defer if (density_plan) |*plan| plan.deinit(alloc);
    defer if (density_map) |*map| map.deinit(alloc);
    defer if (density_recip) |buf| alloc.free(buf);
    defer if (density_real) |buf| alloc.free(buf);

    var band: usize = 0;
    while (band < data.nbands) : (band += 1) {
        const occ = smearingOcc(cfg.scf.smearing, data.values[band], mu, sigma);
        // Accumulate entropy term: -T*S = sigma * sum[f*ln(f) + (1-f)*ln(1-f)]
        // Note: smearingEntropy returns -[f*ln(f) + (1-f)*ln(1-f)] >= 0
        // So we add sigma * entropy * weight to get the -TS contribution
        const entropy_contrib = kp.weight * spin_factor * sigma * smearingEntropy(occ);
        entropy_energy.* += entropy_contrib;
        if (occ <= 0.0) continue;
        const weight = kp.weight * occ * spin_factor;
        band_energy.* += weight * data.values[band];
        if (data.nonlocal) |entries| {
            nonlocal_energy.* += weight * entries[band];
        }
        const density_start = if (profile_ptr != null) profileStart() else null;
        if (use_fft_density) {
            const coeffs = data.vectors[band * basis.gvecs.len .. (band + 1) * basis.gvecs.len];
            try accumulateBandDensityFft(
                alloc,
                grid,
                density_map.?,
                coeffs,
                density_recip.?,
                density_real.?,
                if (density_plan) |*plan| plan else null,
                fft_index_map,
                weight,
                inv_volume,
                rho,
            );
        } else {
            accumulateBandDensity(
                grid,
                basis.gvecs,
                data.vectors,
                band,
                weight,
                inv_volume,
                rho,
            );
        }
        if (profile_ptr) |p| profileAdd(&p.density_ns, density_start);

        // PAW: accumulate rhoij for this band
        if (paw_rhoij) |rij| {
            if (apply_cache) |ac| {
                if (ac.nonlocal_ctx) |*nl| {
                    if (nl.has_paw) {
                        const coeffs_paw = data.vectors[band * basis.gvecs.len .. (band + 1) * basis.gvecs.len];
                        try apply.accumulatePawRhoIJ(alloc, nl, basis.gvecs, atoms.?, coeffs_paw, weight, inv_volume, rij);
                    }
                }
            }
        }
    }
}

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

fn bandNonlocalEnergy(
    n: usize,
    vnl: []const math.Complex,
    vectors: []const math.Complex,
    band: usize,
) f64 {
    var sum = math.complex.init(0.0, 0.0);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const ci = vectors[i + band * n];
        var tmp = math.complex.init(0.0, 0.0);
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const cj = vectors[j + band * n];
            const hij = vnl[i + j * n];
            tmp = math.complex.add(tmp, math.complex.mul(hij, cj));
        }
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(ci), tmp));
    }
    return sum.r;
}

fn innerProduct(n: usize, a: []const math.Complex, b: []const math.Complex) math.Complex {
    var sum = math.complex.init(0.0, 0.0);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(a[i]), b[i]));
    }
    return sum;
}

fn smearingOcc(method: config.SmearingMethod, energy: f64, mu: f64, sigma: f64) f64 {
    if (sigma <= 0.0) return if (energy <= mu) 1.0 else 0.0;
    if (method == .none) return if (energy <= mu) 1.0 else 0.0;
    return 1.0 / (1.0 + std.math.exp((energy - mu) / sigma));
}

/// Compute entropy contribution for a given occupation number.
/// Returns -[f*ln(f) + (1-f)*ln(1-f)] which is always >= 0.
fn smearingEntropy(occ: f64) f64 {
    const eps = 1e-12;
    if (occ <= eps or occ >= 1.0 - eps) return 0.0;
    return -(occ * @log(occ) + (1.0 - occ) * @log(1.0 - occ));
}

fn electronCountForMu(mu: f64, sigma: f64, method: config.SmearingMethod, data: []const KpointEigenData) f64 {
    return electronCountForMuSpin(mu, sigma, method, data, 2.0);
}

/// Electron count with configurable spin factor (1.0 for spin-polarized per-channel, 2.0 for unpolarized).
pub fn electronCountForMuSpin(mu: f64, sigma: f64, method: config.SmearingMethod, data: []const KpointEigenData, spin_factor: f64) f64 {
    var count: f64 = 0.0;
    for (data) |entry| {
        const weight = spin_factor * entry.kpoint.weight;
        for (entry.values[0..entry.nbands]) |energy| {
            count += weight * smearingOcc(method, energy, mu, sigma);
        }
    }
    return count;
}

pub fn findFermiLevel(
    nelec: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data: []const KpointEigenData,
) f64 {
    return findFermiLevelSpin(nelec, sigma, method, data, null, 2.0);
}

/// Find Fermi level for spin-polarized calculation.
/// Both spin channels' eigendata are passed; spin_factor should be 1.0.
/// If data_down is null, only data_up is used (equivalent to unpolarized with given spin_factor).
pub fn findFermiLevelSpin(
    nelec: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data_up: []const KpointEigenData,
    data_down: ?[]const KpointEigenData,
    spin_factor: f64,
) f64 {
    var min_energy = std.math.inf(f64);
    var max_energy = -std.math.inf(f64);
    for (data_up) |entry| {
        for (entry.values[0..entry.nbands]) |energy| {
            min_energy = @min(min_energy, energy);
            max_energy = @max(max_energy, energy);
        }
    }
    if (data_down) |dd| {
        for (dd) |entry| {
            for (entry.values[0..entry.nbands]) |energy| {
                min_energy = @min(min_energy, energy);
                max_energy = @max(max_energy, energy);
            }
        }
    }
    var padding = @max(10.0 * sigma, 1e-3);
    var low = min_energy - padding;
    var high = max_energy + padding;
    var count_low = electronCountForMuSpin(low, sigma, method, data_up, spin_factor);
    var count_high = electronCountForMuSpin(high, sigma, method, data_up, spin_factor);
    if (data_down) |dd| {
        count_low += electronCountForMuSpin(low, sigma, method, dd, spin_factor);
        count_high += electronCountForMuSpin(high, sigma, method, dd, spin_factor);
    }
    var expand: usize = 0;
    while ((count_low > nelec or count_high < nelec) and expand < 12) : (expand += 1) {
        padding *= 2.0;
        low = min_energy - padding;
        high = max_energy + padding;
        count_low = electronCountForMuSpin(low, sigma, method, data_up, spin_factor);
        count_high = electronCountForMuSpin(high, sigma, method, data_up, spin_factor);
        if (data_down) |dd| {
            count_low += electronCountForMuSpin(low, sigma, method, dd, spin_factor);
            count_high += electronCountForMuSpin(high, sigma, method, dd, spin_factor);
        }
    }
    var left = low;
    var right = high;
    var iter: usize = 0;
    while (iter < 80) : (iter += 1) {
        const mid = 0.5 * (left + right);
        var count = electronCountForMuSpin(mid, sigma, method, data_up, spin_factor);
        if (data_down) |dd| {
            count += electronCountForMuSpin(mid, sigma, method, dd, spin_factor);
        }
        if (@abs(count - nelec) < 1e-8) return mid;
        if (count > nelec) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return 0.5 * (left + right);
}
