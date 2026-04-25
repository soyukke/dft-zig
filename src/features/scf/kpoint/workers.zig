const std = @import("std");
const apply = @import("../apply.zig");
const config = @import("../../config/config.zig");
const fft = @import("../../fft/fft.zig");
const grid_mod = @import("../pw_grid.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const kpoint_data = @import("data.zig");
const kpoint_solver = @import("solve.zig");
const logging = @import("../logging.zig");
const math = @import("../../math/math.zig");
const local_potential = @import("../../pseudopotential/local_potential.zig");
const nonlocal = @import("../../pseudopotential/nonlocal.zig");
const paw_mod = @import("../../paw/paw.zig");
const symmetry = @import("../../symmetry/symmetry.zig");

pub const Grid = grid_mod.Grid;
pub const KPoint = symmetry.KPoint;

const KpointCache = kpoint_data.KpointCache;
const KpointEigenData = kpoint_data.KpointEigenData;
const ScfProfile = logging.ScfProfile;
const log_kpoint = logging.log_kpoint;

pub const KpointShared = struct {
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
    fft_plans: []fft.Fft3dPlan,
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

fn select_thread_fft_plan(plans: []fft.Fft3dPlan, thread_index: usize) ?fft.Fft3dPlan {
    if (plans.len <= thread_index) return null;
    return plans[thread_index];
}

/// Dispatch compute_kpoint_contribution for the given k-point index.
fn dispatch_kpoint_contribution(
    shared: *KpointShared,
    thread_index: usize,
    idx: usize,
    kalloc: std.mem.Allocator,
    rho_local: []f64,
    local_band: *f64,
    local_nonlocal: *f64,
    profile_ptr: ?*ScfProfile,
) !void {
    try kpoint_solver.compute_kpoint_contribution(
        kalloc,
        .{
            .io = shared.io,
            .cfg = shared.cfg,
            .grid = shared.grid,
            .kp = shared.kpoints[idx],
            .species = shared.species,
            .atoms = shared.atoms,
            .recip = shared.recip,
            .volume = shared.volume,
            .local_cfg = shared.local_cfg,
            .potential = shared.potential,
            .local_r = shared.local_r,
            .nocc = shared.nocc,
            .use_iterative_config = shared.use_iterative_config,
            .has_qij = shared.has_qij,
            .nonlocal_enabled = shared.nonlocal_enabled,
            .fft_index_map = shared.fft_index_map,
            .iter_max_iter = shared.iter_max_iter,
            .iter_tol = shared.iter_tol,
            .reuse_vectors = shared.reuse_vectors,
            .cache = &shared.kpoint_cache[idx],
            .profile_ptr = profile_ptr,
            .shared_fft_plan = select_thread_fft_plan(shared.fft_plans, thread_index),
            .apply_cache = if (shared.apply_caches) |caches|
                (if (idx < caches.len) &caches[idx] else null)
            else
                null,
            .radial_tables = shared.radial_tables,
            .paw_tabs = shared.paw_tabs,
            .loose_init = false,
        },
        shared.nelec,
        rho_local,
        local_band,
        local_nonlocal,
        null,
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
        dispatch_kpoint_contribution(
            shared,
            thread_index,
            idx,
            arena.allocator(),
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

/// Shared data for parallel smearing eigendata computation.
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
    shared: *SmearingShared,
    thread_index: usize,
    idx: usize,
    kalloc: std.mem.Allocator,
    profile_ptr: ?*ScfProfile,
) !KpointEigenData {
    return try kpoint_solver.compute_kpoint_eigen_data(
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
        select_thread_fft_plan(shared.fft_plans, thread_index),
        if (shared.apply_caches) |caches| (if (idx < caches.len) &caches[idx] else null) else null,
        shared.radial_tables,
        shared.paw_tabs,
    );
}

/// Copy eigendata from arena to persistent c_allocator buffers.
fn persist_eigen_result(
    eigen_result: KpointEigenData,
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

        const eigen_result = dispatch_kpoint_eigen_data(
            shared,
            thread_index,
            idx,
            arena.allocator(),
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
