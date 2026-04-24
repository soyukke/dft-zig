const std = @import("std");
const config = @import("../config/config.zig");
const fft = @import("../fft/fft.zig");
const fft_sizing = @import("../../lib/fft/sizing.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const iterative = @import("../linalg/iterative.zig");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const grid_requirements = @import("../plane_wave/grid_requirements.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const apply = @import("apply.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const logging = @import("logging.zig");
const pw_grid_map = @import("pw_grid_map.zig");
const util = @import("util.zig");

/// Threshold for auto solver selection: basis size <= threshold uses dense,
/// > threshold uses iterative.
/// Based on benchmarks, dense solver (LAPACK zheev) is faster for small matrices.
pub const auto_solver_threshold: usize = 400;

var iterative_grid_warning_logged = std.atomic.Value(u8).init(0);

pub const Grid = grid_mod.Grid;
pub const KPoint = symmetry.KPoint;

const ApplyContext = apply.ApplyContext;
const apply_hamiltonian = apply.apply_hamiltonian;
const apply_hamiltonian_batched = apply.apply_hamiltonian_batched;
const apply_nonlocal_potential = apply.apply_nonlocal_potential;
const ScfProfile = logging.ScfProfile;
const log_kpoint = logging.log_kpoint;
const log_iterative_grid_too_small = logging.log_iterative_grid_too_small;
const profile_start = logging.profile_start;
const profile_add = logging.profile_add;
const PwGridMap = pw_grid_map.PwGridMap;
const grid_requirement = grid_requirements.grid_requirement;
const next_fft_size = fft_sizing.next_fft_size;
const fft_reciprocal_to_complex_in_place = fft_grid.fft_reciprocal_to_complex_in_place;
const fft_reciprocal_to_complex_in_place_mapped =
    fft_grid.fft_reciprocal_to_complex_in_place_mapped;
const fft_complex_to_reciprocal_in_place = fft_grid.fft_complex_to_reciprocal_in_place;
const fft_complex_to_reciprocal_in_place_mapped =
    fft_grid.fft_complex_to_reciprocal_in_place_mapped;

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

    pub fn store_eigenvalues(self: *KpointCache, values: []const f64) !void {
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
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []const KPoint,
    ionic: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
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
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

pub const KpointWorker = struct {
    shared: *KpointShared,
    thread_index: usize,
};

fn set_worker_error(shared: *KpointShared, err: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);

    if (shared.err.* == null) {
        shared.err.* = err;
    }
}

/// Dispatch compute_kpoint_contribution for the given k-point index.
fn dispatch_kpoint_contribution(
    shared: anytype,
    thread_index: usize,
    idx: usize,
    kalloc: std.mem.Allocator,
    rho_local: []f64,
    local_band: *f64,
    local_nonlocal: *f64,
    profile_ptr: ?*ScfProfile,
) !void {
    const thread_fft_plan: ?fft.Fft3dPlan = if (shared.fft_plans.len > thread_index)
        shared.fft_plans[thread_index]
    else
        null;
    try compute_kpoint_contribution(
        kalloc,
        shared.io,
        shared.cfg,
        shared.grid,
        shared.kpoints[idx],
        shared.species,
        shared.atoms,
        shared.recip,
        shared.volume,
        shared.local_cfg,
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
        local_band,
        local_nonlocal,
        profile_ptr,
        thread_fft_plan,
        if (shared.apply_caches) |acs| (if (idx < acs.len) &acs[idx] else null) else null,
        shared.radial_tables,
        shared.paw_tabs,
        null, // paw_rhoij: not thread-safe, handled in smearing path
    );
}

pub fn kpoint_worker(worker: *KpointWorker) void {
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
            shared.log_mutex.lockUncancelable(shared.io);
            log_kpoint(shared.io, idx, shared.kpoints.len) catch {};
            shared.log_mutex.unlock(shared.io);
        }
        _ = arena.reset(.retain_capacity);
        const kalloc = arena.allocator();
        dispatch_kpoint_contribution(
            shared,
            thread_index,
            idx,
            kalloc,
            rho_local,
            &local_band,
            &local_nonlocal,
            profile_ptr,
        ) catch |err| {
            set_worker_error(shared, err);
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

pub fn kpoint_thread_count(total: usize, cfg_threads: usize) usize {
    if (total <= 1) return 1;
    if (cfg_threads > 0) return @min(total, cfg_threads);
    const cpu_count = std.Thread.getCpuCount() catch 1;
    if (cpu_count == 0) return 1;
    return @min(total, cpu_count);
}

/// Shared data for parallel smearing eigendata computation
pub const SmearingShared = struct {
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []const KPoint,
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
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

pub const SmearingWorker = struct {
    shared: *SmearingShared,
    thread_index: usize,
};

fn set_smearing_worker_error(shared: *SmearingShared, err: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);

    if (shared.err.* == null) {
        shared.err.* = err;
    }
}

/// Dispatch compute_kpoint_eigen_data for the given k-point index.
fn dispatch_kpoint_eigen_data(
    shared: anytype,
    thread_index: usize,
    idx: usize,
    kalloc: std.mem.Allocator,
    profile_ptr: ?*ScfProfile,
) !KpointEigenData {
    const thread_fft_plan: ?fft.Fft3dPlan = if (shared.fft_plans.len > thread_index)
        shared.fft_plans[thread_index]
    else
        null;
    return try compute_kpoint_eigen_data(
        kalloc,
        shared.io,
        shared.cfg,
        shared.grid,
        shared.kpoints[idx],
        shared.species,
        shared.atoms,
        shared.recip,
        shared.volume,
        shared.local_cfg,
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
    );
}

/// Copy eigendata from arena to persistent c_allocator buffers.
fn persist_eigen_result(
    eigen_result: anytype,
) !struct { values: []f64, vectors: []math.Complex, nonlocal: ?[]f64 } {
    const values = try std.heap.c_allocator.alloc(f64, eigen_result.nbands);
    errdefer std.heap.c_allocator.free(values);

    @memcpy(values, eigen_result.values);

    const vectors = try std.heap.c_allocator.alloc(math.Complex, eigen_result.vectors.len);
    errdefer std.heap.c_allocator.free(vectors);

    @memcpy(vectors, eigen_result.vectors);

    var nonlocal_vals: ?[]f64 = null;
    if (eigen_result.nonlocal) |nl| {
        nonlocal_vals = try std.heap.c_allocator.alloc(f64, nl.len);
        @memcpy(nonlocal_vals.?, nl);
    }
    return .{ .values = values, .vectors = vectors, .nonlocal = nonlocal_vals };
}

pub fn smearing_worker(worker: *SmearingWorker) void {
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
            shared.log_mutex.lockUncancelable(shared.io);
            log_kpoint(shared.io, idx, shared.kpoints.len) catch {};
            shared.log_mutex.unlock(shared.io);
        }
        _ = arena.reset(.retain_capacity);
        const kalloc = arena.allocator();

        const eigen_result = dispatch_kpoint_eigen_data(
            shared,
            thread_index,
            idx,
            kalloc,
            profile_ptr,
        ) catch |err| {
            set_smearing_worker_error(shared, err);
            shared.stop.store(1, .release);
            break;
        };

        const persisted = persist_eigen_result(eigen_result) catch |err| {
            set_smearing_worker_error(shared, err);
            shared.stop.store(1, .release);
            break;
        };

        shared.eigen_data[idx] = .{
            .kpoint = shared.kpoints[idx],
            .basis_len = eigen_result.basis_len,
            .nbands = eigen_result.nbands,
            .values = persisted.values,
            .vectors = persisted.vectors,
            .nonlocal = persisted.nonlocal,
        };
    }

    if (shared.profiles) |profiles| {
        profiles[thread_index] = local_profile;
    }
}

/// Build or refresh the apply-cache entry for this k-point and wrap it in an ApplyContext.
fn init_apply_context_without_cache(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
    local_values: []const f64,
    vnl: ?[]math.Complex,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    nonlocal_enabled: bool,
    profile_ptr: ?*ScfProfile,
    fft_index_map: ?[]const usize,
    shared_fft_plan: ?fft.Fft3dPlan,
) !ApplyContext {
    if (shared_fft_plan) |plan| {
        return try ApplyContext.init_with_fft_plan(
            alloc,
            io,
            grid,
            basis_gvecs,
            local_values,
            vnl,
            species,
            atoms,
            inv_volume,
            nonlocal_enabled,
            profile_ptr,
            fft_index_map,
            plan,
        );
    }
    return try ApplyContext.init(
        alloc,
        io,
        grid,
        basis_gvecs,
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
}

fn apply_context_from_cache_or_build(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    local_values: []const f64,
    vnl: ?[]math.Complex,
    nonlocal_enabled: bool,
    profile_ptr: ?*ScfProfile,
    fft_index_map: ?[]const usize,
    shared_fft_plan: ?fft.Fft3dPlan,
    apply_cache: ?*apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
) !ApplyContext {
    if (apply_cache) |ac| {
        try refresh_apply_cache(
            ac,
            grid,
            basis_gvecs,
            species,
            nonlocal_enabled,
            fft_index_map,
            radial_tables,
            paw_tabs,
        );
        return try init_apply_context_with_cache(
            alloc,
            io,
            cfg,
            grid,
            basis_gvecs,
            local_values,
            vnl,
            atoms,
            inv_volume,
            profile_ptr,
            fft_index_map,
            shared_fft_plan,
            ac,
        );
    }
    return try init_apply_context_without_cache(
        alloc,
        io,
        cfg,
        grid,
        basis_gvecs,
        local_values,
        vnl,
        species,
        atoms,
        inv_volume,
        nonlocal_enabled,
        profile_ptr,
        fft_index_map,
        shared_fft_plan,
    );
}

fn refresh_apply_cache(
    ac: *apply.KpointApplyCache,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    nonlocal_enabled: bool,
    fft_index_map: ?[]const usize,
    radial_tables: ?[]const nonlocal.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
) !void {
    if (ac.is_valid(basis_gvecs.len)) return;

    ac.nonlocal_ctx = if (nonlocal_enabled)
        (if (paw_tabs) |tabs|
            try apply.build_nonlocal_context_paw(
                ac.cache_alloc,
                species,
                basis_gvecs,
                radial_tables,
                tabs,
            )
        else if (radial_tables) |tables|
            try apply.build_nonlocal_context_with_tables(
                ac.cache_alloc,
                species,
                basis_gvecs,
                tables,
            )
        else
            try apply.build_nonlocal_context_pub(ac.cache_alloc, species, basis_gvecs))
    else
        null;
    ac.map = try PwGridMap.init(ac.cache_alloc, basis_gvecs, grid);
    if (fft_index_map) |idx_map| {
        try ac.map.?.build_fft_indices(ac.cache_alloc, idx_map);
    }
    ac.basis_len = basis_gvecs.len;
}

fn init_apply_context_with_cache(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
    local_values: []const f64,
    vnl: ?[]math.Complex,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    profile_ptr: ?*ScfProfile,
    fft_index_map: ?[]const usize,
    shared_fft_plan: ?fft.Fft3dPlan,
    ac: *apply.KpointApplyCache,
) !ApplyContext {
    if (shared_fft_plan) |plan| {
        return try ApplyContext.init_with_cache(
            alloc,
            io,
            grid,
            basis_gvecs,
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
    }
    const new_plan = try fft.Fft3dPlan.init_with_backend(
        alloc,
        io,
        grid.nx,
        grid.ny,
        grid.nz,
        cfg.scf.fft_backend,
    );
    return try ApplyContext.init_with_cache(
        alloc,
        io,
        grid,
        basis_gvecs,
        local_values,
        vnl,
        ac.nonlocal_ctx,
        ac.map.?,
        atoms,
        inv_volume,
        profile_ptr,
        fft_index_map,
        new_plan,
        true,
    );
}

/// Run iterative (LOBPCG/CG) eigendecomposition using a fresh ApplyContext stored in apply_ctx_out.
fn solve_iterative_eigenproblem(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    basis_len: usize,
    diag: []const f64,
    apply_ctx_out: *?ApplyContext,
    iter_max_iter: usize,
    iter_tol: f64,
    init_vectors: ?[]const math.Complex,
    init_cols: usize,
    nbands: usize,
    use_cg: bool,
) !linalg.EigenDecomp {
    const ctx = &apply_ctx_out.*.?;
    const has_paw_overlap = if (ctx.nonlocal_ctx) |nc| nc.has_paw_overlap() else false;
    const op = iterative.Operator{
        .n = basis_len,
        .ctx = ctx,
        .apply = apply_hamiltonian,
        .apply_batch = apply_hamiltonian_batched,
        .apply_s = if (has_paw_overlap) apply.apply_overlap else null,
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
    if (use_cg) return try iterative.hermitian_eigen_decomp_cg(alloc, op, diag, nbands, opts);
    return try iterative.hermitian_eigen_decomp_iterative(
        alloc,
        cfg.linalg_backend,
        op,
        diag,
        nbands,
        opts,
    );
}

fn run_iterative_eigen_decomp(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    local_r: ?[]f64,
    vnl: ?[]math.Complex,
    nonlocal_enabled: bool,
    profile_ptr: ?*ScfProfile,
    fft_index_map: ?[]const usize,
    shared_fft_plan: ?fft.Fft3dPlan,
    apply_cache: ?*apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    iter_max_iter: usize,
    iter_tol: f64,
    init_vectors: ?[]const math.Complex,
    init_cols: usize,
    nbands: usize,
    use_cg: bool,
    apply_ctx_out: *?ApplyContext,
) !linalg.EigenDecomp {
    const diag = try alloc.alloc(f64, basis_gvecs.len);
    defer alloc.free(diag);

    for (basis_gvecs, 0..) |g, i| {
        diag[i] = g.kinetic;
    }
    const local_values = local_r orelse return error.InvalidGrid;
    const ctx = try apply_context_from_cache_or_build(
        alloc,
        io,
        cfg,
        grid,
        basis_gvecs,
        species,
        atoms,
        inv_volume,
        local_values,
        vnl,
        nonlocal_enabled,
        profile_ptr,
        fft_index_map,
        shared_fft_plan,
        apply_cache,
        radial_tables,
        paw_tabs,
    );
    apply_ctx_out.* = ctx;
    return try solve_iterative_eigenproblem(
        alloc,
        cfg,
        basis_gvecs.len,
        diag,
        apply_ctx_out,
        iter_max_iter,
        iter_tol,
        init_vectors,
        init_cols,
        nbands,
        use_cg,
    );
}

/// Dense-solver eigendecomposition; uses build_hamiltonian + optional build_overlap_matrix for PAW.
fn run_dense_eigen_decomp(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    basis_gvecs: []plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    potential: hamiltonian.PotentialGrid,
    has_qij: bool,
    profile_ptr: ?*ScfProfile,
) !linalg.EigenDecomp {
    const h_start = if (profile_ptr != null) profile_start(io) else null;
    const h = try hamiltonian.build_hamiltonian(
        alloc,
        basis_gvecs,
        species,
        atoms,
        inv_volume,
        local_cfg,
        potential,
    );
    if (profile_ptr) |p| profile_add(io, &p.h_build_ns, h_start);
    defer alloc.free(h);

    if (has_qij) {
        const s_start = if (profile_ptr != null) profile_start(io) else null;
        const s = try hamiltonian.build_overlap_matrix(
            alloc,
            basis_gvecs,
            species,
            atoms,
            inv_volume,
        );
        if (profile_ptr) |p| profile_add(io, &p.s_build_ns, s_start);
        defer alloc.free(s);

        return try linalg.hermitian_gen_eigen_decomp(
            alloc,
            cfg.linalg_backend,
            basis_gvecs.len,
            h,
            s,
        );
    }
    return try linalg.hermitian_eigen_decomp(
        alloc,
        cfg.linalg_backend,
        basis_gvecs.len,
        h,
    );
}

/// Generate plane-wave basis with optional timing profile.
fn generate_basis_profiled(
    alloc: std.mem.Allocator,
    io: std.Io,
    recip: math.Mat3,
    ecut_ry: f64,
    k_cart: math.Vec3,
    profile_ptr: ?*ScfProfile,
) !plane_wave.Basis {
    const basis_start = if (profile_ptr != null) profile_start(io) else null;
    const basis = try plane_wave.generate(alloc, recip, ecut_ry, k_cart);
    if (profile_ptr) |p| profile_add(io, &p.basis_ns, basis_start);
    return basis;
}

/// Decide whether to use the iterative solver (respecting auto mode and grid fit).
fn choose_iterative_solver(
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
    use_iterative_config: bool,
) !bool {
    var use_iterative = use_iterative_config;
    if (cfg.scf.solver == .auto and basis_gvecs.len <= auto_solver_threshold) {
        use_iterative = false;
    }
    if (use_iterative) {
        const req = grid_requirement(basis_gvecs);
        if (req.nx > grid.nx or req.ny > grid.ny or req.nz > grid.nz) {
            if (iterative_grid_warning_logged.cmpxchgStrong(
                0,
                1,
                .acquire,
                .acquire,
            ) == null) {
                try log_iterative_grid_too_small(
                    io,
                    req.nx,
                    req.ny,
                    req.nz,
                    next_fft_size(req.nx),
                    next_fft_size(req.ny),
                    next_fft_size(req.nz),
                );
            }
            use_iterative = false;
        }
    }
    return use_iterative;
}

/// Build the nonlocal matrix for dense solver paths; returns null for iterative or no nonlocal.
fn build_vnl_if_needed(
    alloc: std.mem.Allocator,
    io: std.Io,
    basis_gvecs: []plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    use_iterative: bool,
    nonlocal_enabled: bool,
    profile_ptr: ?*ScfProfile,
) !?[]math.Complex {
    if (use_iterative or !nonlocal_enabled) return null;
    const vnl_start = if (profile_ptr != null) profile_start(io) else null;
    const mat = try hamiltonian.build_nonlocal_matrix(
        alloc,
        basis_gvecs,
        species,
        atoms,
        inv_volume,
    );
    if (profile_ptr) |p| profile_add(io, &p.vnl_build_ns, vnl_start);
    return mat;
}

/// Pick the cached initial vectors for LOBPCG warm-start when they match the current basis.
const InitVectors = struct {
    init_vectors: ?[]const math.Complex,
    init_cols: usize,
};

fn pick_init_vectors(
    cache: *const KpointCache,
    basis_len: usize,
    nbands: usize,
    use_iterative: bool,
    reuse_vectors: bool,
) InitVectors {
    if (use_iterative and reuse_vectors and
        cache.vectors.len > 0 and
        cache.n == basis_len and
        cache.nbands == nbands)
    {
        return .{ .init_vectors = cache.vectors, .init_cols = cache.nbands };
    }
    return .{ .init_vectors = null, .init_cols = 0 };
}

/// Allocate PwGridMap, recip/real complex buffers and FFT plan used during density accumulation.
const DensityBuffers = struct {
    map: ?PwGridMap,
    recip: ?[]math.Complex,
    real: ?[]math.Complex,
    plan: ?fft.Fft3dPlan,
};

fn allocate_density_buffers(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
) !DensityBuffers {
    const total = grid.count();
    const map = try PwGridMap.init(alloc, basis_gvecs, grid);
    const recip = try alloc.alloc(math.Complex, total);
    const real = try alloc.alloc(math.Complex, total);
    const plan = try fft.Fft3dPlan.init_with_backend(
        alloc,
        io,
        grid.nx,
        grid.ny,
        grid.nz,
        cfg.scf.fft_backend,
    );
    return .{ .map = map, .recip = recip, .real = real, .plan = plan };
}

fn deinit_density_buffers(alloc: std.mem.Allocator, bufs: *DensityBuffers) void {
    if (bufs.plan) |*plan| plan.deinit(alloc);
    if (bufs.map) |*m| m.deinit(alloc);
    if (bufs.recip) |buf| alloc.free(buf);
    if (bufs.real) |buf| alloc.free(buf);
}

const SolvedKpoint = struct {
    basis: plane_wave.Basis,
    eig: linalg.EigenDecomp,
    vnl: ?[]math.Complex,
    apply_ctx: ?ApplyContext,
    inv_volume: f64,
    nbands: usize,
    store_vectors: bool,

    fn deinit(self: *SolvedKpoint, alloc: std.mem.Allocator) void {
        if (self.apply_ctx) |*ctx| ctx.deinit(alloc);
        if (self.vnl) |mat| alloc.free(mat);
        self.eig.deinit(alloc);
        self.basis.deinit(alloc);
    }
};

fn run_kpoint_solve(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]f64,
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
    nbands: usize,
    use_iterative: bool,
    loose_init: bool,
    vnl: ?[]math.Complex,
    apply_ctx: *?ApplyContext,
) !linalg.EigenDecomp {
    const init = if (loose_init)
        pick_init_vectors_loose(cache, basis_gvecs.len, nbands, use_iterative, reuse_vectors)
    else
        pick_init_vectors(cache, basis_gvecs.len, nbands, use_iterative, reuse_vectors);
    const use_cg = (cfg.scf.solver == .cg);
    const eig_start = if (profile_ptr != null) profile_start(io) else null;
    const eig = if (use_iterative)
        try run_iterative_eigen_decomp(
            alloc,
            io,
            cfg,
            grid,
            basis_gvecs,
            species,
            atoms,
            inv_volume,
            local_r,
            vnl,
            nonlocal_enabled,
            profile_ptr,
            fft_index_map,
            shared_fft_plan,
            apply_cache,
            radial_tables,
            paw_tabs,
            iter_max_iter,
            iter_tol,
            init.init_vectors,
            init.init_cols,
            nbands,
            use_cg,
            apply_ctx,
        )
    else
        try run_dense_eigen_decomp(
            alloc,
            io,
            cfg,
            basis_gvecs,
            species,
            atoms,
            inv_volume,
            local_cfg,
            potential,
            has_qij,
            profile_ptr,
        );
    if (profile_ptr) |p| profile_add(io, &p.eig_ns, eig_start);
    return eig;
}

fn solve_kpoint(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
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
    reuse_vectors: bool,
    cache: *KpointCache,
    profile_ptr: ?*ScfProfile,
    shared_fft_plan: ?fft.Fft3dPlan,
    apply_cache: ?*apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    loose_init: bool,
) !SolvedKpoint {
    var basis = try generate_basis_profiled(
        alloc,
        io,
        recip,
        cfg.scf.ecut_ry,
        kp.k_cart,
        profile_ptr,
    );
    errdefer basis.deinit(alloc);

    const nbands = @min(@max(cfg.band.nbands, nocc), basis.gvecs.len);
    if (nocc > basis.gvecs.len) return error.InsufficientBands;
    const inv_volume = 1.0 / volume;
    var apply_ctx: ?ApplyContext = null;
    errdefer if (apply_ctx) |*ctx| ctx.deinit(alloc);

    const use_iterative = try choose_iterative_solver(
        io,
        cfg,
        grid,
        basis.gvecs,
        use_iterative_config,
    );
    const vnl = try build_vnl_if_needed(
        alloc,
        io,
        basis.gvecs,
        species,
        atoms,
        inv_volume,
        use_iterative,
        nonlocal_enabled,
        profile_ptr,
    );
    errdefer if (vnl) |mat| alloc.free(mat);

    var eig = try run_kpoint_solve(
        alloc,
        io,
        cfg,
        grid,
        basis.gvecs,
        species,
        atoms,
        inv_volume,
        local_cfg,
        potential,
        local_r,
        has_qij,
        nonlocal_enabled,
        fft_index_map,
        iter_max_iter,
        iter_tol,
        reuse_vectors,
        cache,
        profile_ptr,
        shared_fft_plan,
        apply_cache,
        radial_tables,
        paw_tabs,
        nbands,
        use_iterative,
        loose_init,
        vnl,
        &apply_ctx,
    );
    errdefer eig.deinit(alloc);

    return .{
        .basis = basis,
        .eig = eig,
        .vnl = vnl,
        .apply_ctx = apply_ctx,
        .inv_volume = inv_volume,
        .nbands = nbands,
        .store_vectors = use_iterative and reuse_vectors,
    };
}

pub fn compute_kpoint_contribution(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    local_cfg: local_potential.LocalPotentialConfig,
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
    var solved = try solve_kpoint(
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
        reuse_vectors,
        cache,
        profile_ptr,
        shared_fft_plan,
        apply_cache,
        radial_tables,
        paw_tabs,
        false,
    );
    defer solved.deinit(alloc);

    // FFT now supports arbitrary sizes via Bluestein's algorithm
    var dbufs = try allocate_density_buffers(alloc, io, cfg, grid, solved.basis.gvecs);
    defer deinit_density_buffers(alloc, &dbufs);

    try accumulate_band_contributions(
        alloc,
        io,
        cfg,
        grid,
        kp,
        solved.basis.gvecs,
        atoms,
        solved.inv_volume,
        nocc,
        nelec,
        solved.nbands,
        solved.eig,
        solved.vnl,
        &solved.apply_ctx,
        apply_cache,
        fft_index_map,
        dbufs.map,
        dbufs.recip,
        dbufs.real,
        &dbufs.plan,
        profile_ptr,
        rho,
        band_energy,
        nonlocal_energy,
        paw_rhoij,
    );

    try store_eig_result(
        cache,
        solved.basis.gvecs.len,
        solved.nbands,
        solved.eig,
        solved.store_vectors,
    );
}

fn store_eig_result(
    cache: *KpointCache,
    basis_len: usize,
    nbands: usize,
    eig: linalg.EigenDecomp,
    store_vectors: bool,
) !void {
    if (store_vectors) {
        try cache.store(basis_len, nbands, eig.vectors);
    }
    // Always save eigenvalues for potential use in constructing WavefunctionData
    try cache.store_eigenvalues(eig.values[0..nbands]);
}

/// Accumulate rhoij contribution for a single band from either cached or fresh nonlocal context.
fn accumulate_band_paw_rho_ij(
    alloc: std.mem.Allocator,
    apply_cache: ?*apply.KpointApplyCache,
    apply_ctx: *?ApplyContext,
    basis_gvecs: []plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    coeffs_paw: []const math.Complex,
    weight_factor: f64,
    inv_volume: f64,
    rij: *paw_mod.RhoIJ,
) !void {
    if (apply_cache) |ac| {
        if (ac.nonlocal_ctx) |*nl| {
            if (nl.has_paw) {
                try apply.accumulate_paw_rho_ij(
                    alloc,
                    nl,
                    basis_gvecs,
                    atoms,
                    coeffs_paw,
                    weight_factor,
                    inv_volume,
                    rij,
                );
            }
        }
        return;
    }
    if (apply_ctx.*) |*ctx| {
        if (ctx.nonlocal_ctx) |*nl| {
            if (nl.has_paw) {
                try apply.accumulate_paw_rho_ij(
                    alloc,
                    nl,
                    basis_gvecs,
                    atoms,
                    coeffs_paw,
                    weight_factor,
                    inv_volume,
                    rij,
                );
            }
        }
    }
}

/// Per-band band-energy, nonlocal-energy, density and (optionally) PAW rhoij accumulation.
fn accumulate_band_contributions(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    basis_gvecs: []plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    nocc: usize,
    nelec: f64,
    nbands: usize,
    eig: linalg.EigenDecomp,
    vnl: ?[]math.Complex,
    apply_ctx: *?ApplyContext,
    apply_cache: ?*apply.KpointApplyCache,
    fft_index_map: ?[]const usize,
    density_map: ?PwGridMap,
    density_recip: ?[]math.Complex,
    density_real: ?[]math.Complex,
    density_plan: *?fft.Fft3dPlan,
    profile_ptr: ?*ScfProfile,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    paw_rhoij: ?*paw_mod.RhoIJ,
) !void {
    _ = cfg;
    const occ_last = if (nocc == 0)
        0.0
    else
        nelec / 2.0 - @as(f64, @floatFromInt(nocc - 1));
    const spin_factor = 2.0;
    var band: usize = 0;
    while (band < nocc and band < nbands) : (band += 1) {
        const occ = if (band + 1 == nocc) occ_last else 1.0;
        if (occ <= 0.0) continue;
        const w = kp.weight * occ * spin_factor;
        band_energy.* += w * eig.values[band];
        if (vnl) |mat| {
            const e_nl = band_nonlocal_energy(basis_gvecs.len, mat, eig.vectors, band);
            nonlocal_energy.* += w * e_nl;
        } else if (apply_ctx.*) |*ctx| {
            if (ctx.nonlocal_ctx != null) {
                const psi = eig.vectors[band * basis_gvecs.len .. (band + 1) * basis_gvecs.len];
                try apply_nonlocal_potential(ctx, psi, ctx.work_vec);
                const e_nl = inner_product(basis_gvecs.len, psi, ctx.work_vec).r;
                nonlocal_energy.* += w * e_nl;
            }
        }
        const density_start = if (profile_ptr != null) profile_start(io) else null;
        const coeffs = eig.vectors[band * basis_gvecs.len .. (band + 1) * basis_gvecs.len];
        try accumulate_band_density_fft(
            alloc,
            grid,
            density_map.?,
            coeffs,
            density_recip.?,
            density_real.?,
            if (density_plan.*) |*plan| plan else null,
            fft_index_map,
            w,
            inv_volume,
            rho,
        );
        if (profile_ptr) |p| profile_add(io, &p.density_ns, density_start);

        if (paw_rhoij) |rij| {
            try accumulate_band_paw_rho_ij(
                alloc,
                apply_cache,
                apply_ctx,
                basis_gvecs,
                atoms,
                coeffs,
                w,
                inv_volume,
                rij,
            );
        }
    }
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

/// Pick warm-start init vectors for smearing path where cache.nbands can exceed nbands.
fn pick_init_vectors_loose(
    cache: *const KpointCache,
    basis_len: usize,
    nbands: usize,
    use_iterative: bool,
    reuse_vectors: bool,
) InitVectors {
    if (use_iterative and reuse_vectors and
        cache.vectors.len > 0 and
        cache.n == basis_len and
        cache.nbands >= nbands)
    {
        return .{ .init_vectors = cache.vectors, .init_cols = nbands };
    }
    return .{ .init_vectors = null, .init_cols = 0 };
}

/// Compute per-band nonlocal expectation values for the smearing path.
fn compute_nonlocal_band_entries(
    alloc: std.mem.Allocator,
    nbands: usize,
    basis_len: usize,
    vectors: []const math.Complex,
    vnl: ?[]math.Complex,
    apply_ctx: *?ApplyContext,
) !?[]f64 {
    const entries = try alloc.alloc(f64, nbands);
    errdefer alloc.free(entries);

    @memset(entries, 0.0);
    if (vnl) |mat| {
        for (entries, 0..) |*value, band| {
            value.* = band_nonlocal_energy(basis_len, mat, vectors, band);
        }
    } else if (apply_ctx.*) |*ctx| {
        if (ctx.nonlocal_ctx != null) {
            var band: usize = 0;
            while (band < nbands) : (band += 1) {
                const psi = vectors[band * basis_len .. (band + 1) * basis_len];
                try apply_nonlocal_potential(ctx, psi, ctx.work_vec);
                entries[band] = inner_product(basis_len, psi, ctx.work_vec).r;
            }
        }
    }
    return entries;
}

fn finalize_solved_eigen_data(
    alloc: std.mem.Allocator,
    kp: KPoint,
    nonlocal_enabled: bool,
    cache: *KpointCache,
    solved: *SolvedKpoint,
) !KpointEigenData {
    return try finalize_eigen_data(
        alloc,
        kp,
        solved.eig,
        solved.basis.gvecs.len,
        solved.nbands,
        nonlocal_enabled,
        solved.vnl,
        &solved.apply_ctx,
        cache,
        solved.store_vectors,
    );
}

pub fn compute_kpoint_eigen_data(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
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
    reuse_vectors: bool,
    cache: *KpointCache,
    profile_ptr: ?*ScfProfile,
    shared_fft_plan: ?fft.Fft3dPlan,
    apply_cache: ?*apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
) !KpointEigenData {
    var solved = try solve_kpoint(
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
        reuse_vectors,
        cache,
        profile_ptr,
        shared_fft_plan,
        apply_cache,
        radial_tables,
        paw_tabs,
        true,
    );
    defer solved.deinit(alloc);

    return try finalize_solved_eigen_data(alloc, kp, nonlocal_enabled, cache, &solved);
}

/// Copy eig result to persistent storage, compute nonlocal-band entries,
/// update cache, return result.
fn finalize_eigen_data(
    alloc: std.mem.Allocator,
    kp: KPoint,
    eig: linalg.EigenDecomp,
    basis_len: usize,
    nbands: usize,
    nonlocal_enabled: bool,
    vnl: ?[]math.Complex,
    apply_ctx: *?ApplyContext,
    cache: *KpointCache,
    store_vectors: bool,
) !KpointEigenData {
    const dup = try duplicate_eig_result(alloc, eig, basis_len, nbands);
    errdefer {
        alloc.free(dup.values);
        alloc.free(dup.vectors);
    }
    var nonlocal_band: ?[]f64 = null;
    if (nonlocal_enabled) {
        nonlocal_band = try compute_nonlocal_band_entries(
            alloc,
            nbands,
            basis_len,
            dup.vectors,
            vnl,
            apply_ctx,
        );
    }
    try maybe_store_eigen_in_cache(
        cache,
        basis_len,
        nbands,
        dup.vectors,
        dup.values,
        store_vectors,
    );
    return .{
        .kpoint = kp,
        .basis_len = basis_len,
        .nbands = nbands,
        .values = dup.values,
        .vectors = dup.vectors,
        .nonlocal = nonlocal_band,
    };
}

/// Save eigenvectors (if allowed) and eigenvalues into the per-kpoint cache.
fn maybe_store_eigen_in_cache(
    cache: *KpointCache,
    basis_len: usize,
    nbands: usize,
    vectors: []const math.Complex,
    values: []const f64,
    store_vectors: bool,
) !void {
    if (store_vectors and nbands >= cache.nbands) {
        try cache.store(basis_len, nbands, vectors);
    }
    if (nbands >= cache.eigenvalues.len) {
        try cache.store_eigenvalues(values[0..nbands]);
    }
}

/// Allocate and copy eig.values[0..nbands] and eig.vectors[0..basis_len*nbands] to new buffers.
fn duplicate_eig_result(
    alloc: std.mem.Allocator,
    eig: linalg.EigenDecomp,
    basis_len: usize,
    nbands: usize,
) !struct { values: []f64, vectors: []math.Complex } {
    const values = try alloc.alloc(f64, nbands);
    errdefer alloc.free(values);

    @memcpy(values, eig.values[0..nbands]);
    const vector_count = basis_len * nbands;
    const vectors = try alloc.alloc(math.Complex, vector_count);
    errdefer alloc.free(vectors);

    @memcpy(vectors, eig.vectors[0..vector_count]);
    return .{ .values = values, .vectors = vectors };
}

pub fn accumulate_kpoint_density_smearing(
    alloc: std.mem.Allocator,
    io: std.Io,
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
    return accumulate_kpoint_density_smearing_spin(
        alloc,
        io,
        cfg,
        grid,
        kp,
        data,
        recip,
        volume,
        fft_index_map,
        mu,
        sigma,
        rho,
        band_energy,
        nonlocal_energy,
        entropy_energy,
        profile_ptr,
        2.0,
        apply_cache,
        paw_rhoij,
        atoms,
    );
}

/// Spin-factor parameterized version.
/// Process one band for the smearing-spin density accumulation.
fn accumulate_smearing_band(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    data: KpointEigenData,
    basis_gvecs: []plane_wave.GVector,
    band: usize,
    mu: f64,
    sigma: f64,
    spin_factor: f64,
    inv_volume: f64,
    dbufs: *DensityBuffers,
    fft_index_map: ?[]const usize,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    entropy_energy: *f64,
    profile_ptr: ?*ScfProfile,
    apply_cache: ?*apply.KpointApplyCache,
    paw_rhoij: ?*paw_mod.RhoIJ,
    atoms: ?[]const hamiltonian.AtomData,
) !void {
    const occ = smearing_occ(cfg.scf.smearing, data.values[band], mu, sigma);
    // Accumulate entropy term: -T*S = sigma * sum[f*ln(f) + (1-f)*ln(1-f)]
    const entropy_contrib = kp.weight * spin_factor * sigma * smearing_entropy(occ);
    entropy_energy.* += entropy_contrib;
    if (occ <= 0.0) return;
    const weight = kp.weight * occ * spin_factor;
    band_energy.* += weight * data.values[band];
    if (data.nonlocal) |entries| {
        nonlocal_energy.* += weight * entries[band];
    }
    const density_start = if (profile_ptr != null) profile_start(io) else null;
    const coeffs = data.vectors[band * basis_gvecs.len .. (band + 1) * basis_gvecs.len];
    try accumulate_band_density_fft(
        alloc,
        grid,
        dbufs.map.?,
        coeffs,
        dbufs.recip.?,
        dbufs.real.?,
        if (dbufs.plan) |*plan| plan else null,
        fft_index_map,
        weight,
        inv_volume,
        rho,
    );
    if (profile_ptr) |p| profile_add(io, &p.density_ns, density_start);

    if (paw_rhoij) |rij| {
        if (apply_cache) |ac| {
            if (ac.nonlocal_ctx) |*nl| {
                if (nl.has_paw) {
                    try apply.accumulate_paw_rho_ij(
                        alloc,
                        nl,
                        basis_gvecs,
                        atoms.?,
                        coeffs,
                        weight,
                        inv_volume,
                        rij,
                    );
                }
            }
        }
    }
}

pub fn accumulate_kpoint_density_smearing_spin(
    alloc: std.mem.Allocator,
    io: std.Io,
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
    var dbufs = try allocate_density_buffers(alloc, io, cfg, grid, basis.gvecs);
    defer deinit_density_buffers(alloc, &dbufs);

    var band: usize = 0;
    while (band < data.nbands) : (band += 1) {
        try accumulate_smearing_band(
            alloc,
            io,
            cfg,
            grid,
            kp,
            data,
            basis.gvecs,
            band,
            mu,
            sigma,
            spin_factor,
            inv_volume,
            &dbufs,
            fft_index_map,
            rho,
            band_energy,
            nonlocal_energy,
            entropy_energy,
            profile_ptr,
            apply_cache,
            paw_rhoij,
            atoms,
        );
    }
}

fn accumulate_band_density(
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

fn accumulate_band_density_fft(
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
        try fft_reciprocal_to_complex_in_place_mapped(
            alloc,
            grid,
            idx_map,
            work_recip,
            work_real,
            plan,
        );
    } else {
        try fft_reciprocal_to_complex_in_place(alloc, grid, work_recip, work_real, plan);
    }
    for (work_real, 0..) |psi, i| {
        const density = (psi.r * psi.r + psi.i * psi.i) * inv_volume;
        rho[i] += weight * density;
    }
}

fn band_nonlocal_energy(
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

fn inner_product(n: usize, a: []const math.Complex, b: []const math.Complex) math.Complex {
    var sum = math.complex.init(0.0, 0.0);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(a[i]), b[i]));
    }
    return sum;
}

fn smearing_occ(method: config.SmearingMethod, energy: f64, mu: f64, sigma: f64) f64 {
    if (sigma <= 0.0) return if (energy <= mu) 1.0 else 0.0;
    if (method == .none) return if (energy <= mu) 1.0 else 0.0;
    return 1.0 / (1.0 + std.math.exp((energy - mu) / sigma));
}

/// Compute entropy contribution for a given occupation number.
/// Returns -[f*ln(f) + (1-f)*ln(1-f)] which is always >= 0.
fn smearing_entropy(occ: f64) f64 {
    const eps = 1e-12;
    if (occ <= eps or occ >= 1.0 - eps) return 0.0;
    return -(occ * @log(occ) + (1.0 - occ) * @log(1.0 - occ));
}

fn electron_count_for_mu(
    mu: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data: []const KpointEigenData,
) f64 {
    return electron_count_for_mu_spin(mu, sigma, method, data, 2.0);
}

/// Electron count with configurable spin factor
/// (1.0 for spin-polarized per-channel, 2.0 for unpolarized).
pub fn electron_count_for_mu_spin(
    mu: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data: []const KpointEigenData,
    spin_factor: f64,
) f64 {
    var count: f64 = 0.0;
    for (data) |entry| {
        const weight = spin_factor * entry.kpoint.weight;
        for (entry.values[0..entry.nbands]) |energy| {
            count += weight * smearing_occ(method, energy, mu, sigma);
        }
    }
    return count;
}

pub fn find_fermi_level(
    nelec: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data: []const KpointEigenData,
) f64 {
    return find_fermi_level_spin(nelec, sigma, method, data, null, 2.0);
}

/// Find Fermi level for spin-polarized calculation.
/// Both spin channels' eigendata are passed; spin_factor should be 1.0.
/// If data_down is null, only data_up is used (equivalent to unpolarized with given spin_factor).
pub fn find_fermi_level_spin(
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
    var count_low = electron_count_for_mu_spin(low, sigma, method, data_up, spin_factor);
    var count_high = electron_count_for_mu_spin(high, sigma, method, data_up, spin_factor);
    if (data_down) |dd| {
        count_low += electron_count_for_mu_spin(low, sigma, method, dd, spin_factor);
        count_high += electron_count_for_mu_spin(high, sigma, method, dd, spin_factor);
    }
    var expand: usize = 0;
    while ((count_low > nelec or count_high < nelec) and expand < 12) : (expand += 1) {
        padding *= 2.0;
        low = min_energy - padding;
        high = max_energy + padding;
        count_low = electron_count_for_mu_spin(low, sigma, method, data_up, spin_factor);
        count_high = electron_count_for_mu_spin(high, sigma, method, data_up, spin_factor);
        if (data_down) |dd| {
            count_low += electron_count_for_mu_spin(low, sigma, method, dd, spin_factor);
            count_high += electron_count_for_mu_spin(high, sigma, method, dd, spin_factor);
        }
    }
    var left = low;
    var right = high;
    var iter: usize = 0;
    while (iter < 80) : (iter += 1) {
        const mid = 0.5 * (left + right);
        var count = electron_count_for_mu_spin(mid, sigma, method, data_up, spin_factor);
        if (data_down) |dd| {
            count += electron_count_for_mu_spin(mid, sigma, method, dd, spin_factor);
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
