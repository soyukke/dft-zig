const std = @import("std");
const blas = @import("../../lib/linalg/blas.zig");
const fft = @import("../fft/fft.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal_context = @import("nonlocal_context.zig");
const paw_mod = @import("../paw/paw.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const pw_grid_map = @import("pw_grid_map.zig");
const runtime_logging = @import("../runtime/logging.zig");

pub const Grid = grid_mod.Grid;
pub const ScfProfile = logging.ScfProfile;

// Re-export from nonlocal_context.zig
pub const NonlocalSpecies = nonlocal_context.NonlocalSpecies;
pub const NonlocalContext = nonlocal_context.NonlocalContext;
pub const build_nonlocal_context_pub = nonlocal_context.build_nonlocal_context_pub;
pub const build_nonlocal_context_with_tables = nonlocal_context.build_nonlocal_context_with_tables;
pub const build_nonlocal_context_paw = nonlocal_context.build_nonlocal_context_paw;

const PwGridMap = pw_grid_map.PwGridMap;

const fft_reciprocal_to_complex_in_place = fft_grid.fft_reciprocal_to_complex_in_place;
const fft_reciprocal_to_complex_in_place_mapped =
    fft_grid.fft_reciprocal_to_complex_in_place_mapped;
const fft_complex_to_reciprocal_in_place = fft_grid.fft_complex_to_reciprocal_in_place;
const fft_complex_to_reciprocal_in_place_mapped =
    fft_grid.fft_complex_to_reciprocal_in_place_mapped;

const profile_start = logging.profile_start;
const profile_add = logging.profile_add;

const SharedWorkspaceState = struct {
    count: usize,
    workspaces: []ApplyWorkspace,
    ws_in_use: []std.atomic.Value(u8),
};

/// Thread-local workspace for apply_hamiltonian
pub const ApplyWorkspace = struct {
    work_recip: []math.Complex,
    work_real: []math.Complex,
    work_recip_out: []math.Complex,
    work_vec: []math.Complex,
    work_phase: []math.Complex,
    work_xphase: []math.Complex,
    work_coeff: []math.Complex,
    work_coeff2: []math.Complex,
    fft_plan: ?fft.Fft3dPlan,
    alloc: std.mem.Allocator,

    pub fn init(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: Grid,
        n_gvecs: usize,
        max_m_total: usize,
        fft_backend: fft.FftBackend,
    ) !ApplyWorkspace {
        // Create new FFT plan
        const fft_plan: ?fft.Fft3dPlan = try fft.Fft3dPlan.init_with_backend(
            alloc,
            io,
            grid.nx,
            grid.ny,
            grid.nz,
            fft_backend,
        );
        return init_with_plan(alloc, grid, n_gvecs, max_m_total, fft_plan);
    }

    /// Initialize with an existing FFT plan (avoids mutex contention in parallel execution)
    pub fn init_with_plan(
        alloc: std.mem.Allocator,
        grid: Grid,
        n_gvecs: usize,
        max_m_total: usize,
        fft_plan: ?fft.Fft3dPlan,
    ) !ApplyWorkspace {
        const grid_total = grid.count();
        const work_recip = try alloc.alloc(math.Complex, grid_total);
        errdefer alloc.free(work_recip);

        const work_real = try alloc.alloc(math.Complex, grid_total);
        errdefer alloc.free(work_real);

        const work_recip_out = try alloc.alloc(math.Complex, grid_total);
        errdefer alloc.free(work_recip_out);

        const work_vec = try alloc.alloc(math.Complex, n_gvecs);
        errdefer alloc.free(work_vec);

        const work_phase = try alloc.alloc(math.Complex, n_gvecs);
        errdefer alloc.free(work_phase);

        const work_xphase = try alloc.alloc(math.Complex, n_gvecs);
        errdefer alloc.free(work_xphase);

        const work_coeff = if (max_m_total > 0)
            try alloc.alloc(math.Complex, max_m_total)
        else
            &[_]math.Complex{};
        errdefer if (max_m_total > 0) alloc.free(work_coeff);

        const work_coeff2 = if (max_m_total > 0)
            try alloc.alloc(math.Complex, max_m_total)
        else
            &[_]math.Complex{};
        errdefer if (max_m_total > 0) alloc.free(work_coeff2);

        return .{
            .work_recip = work_recip,
            .work_real = work_real,
            .work_recip_out = work_recip_out,
            .work_vec = work_vec,
            .work_phase = work_phase,
            .work_xphase = work_xphase,
            .work_coeff = @constCast(work_coeff),
            .work_coeff2 = @constCast(work_coeff2),
            .fft_plan = fft_plan,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *ApplyWorkspace, alloc: std.mem.Allocator) void {
        self.deinit_without_plan(alloc);
        if (self.fft_plan) |*plan| {
            plan.deinit(alloc);
        }
    }

    /// Deinit without freeing the FFT plan (for shared plans)
    pub fn deinit_without_plan(self: *ApplyWorkspace, alloc: std.mem.Allocator) void {
        if (self.work_recip.len > 0) alloc.free(self.work_recip);
        if (self.work_real.len > 0) alloc.free(self.work_real);
        if (self.work_recip_out.len > 0) alloc.free(self.work_recip_out);
        if (self.work_vec.len > 0) alloc.free(self.work_vec);
        if (self.work_phase.len > 0) alloc.free(self.work_phase);
        if (self.work_xphase.len > 0) alloc.free(self.work_xphase);
        if (self.work_coeff.len > 0) alloc.free(self.work_coeff);
        if (self.work_coeff2.len > 0) alloc.free(self.work_coeff2);
    }
};

/// Cache for per-kpoint ApplyContext components that don't change across SCF iterations.
/// NonlocalContext and PwGridMap only depend on gvecs (fixed per kpoint) and species data.
/// Cache contents are always allocated with a persistent allocator (not arena) so they
/// survive arena resets in parallel workers.
pub const KpointApplyCache = struct {
    nonlocal_ctx: ?NonlocalContext = null,
    map: ?PwGridMap = null,
    basis_len: usize = 0,
    cache_alloc: std.mem.Allocator = std.heap.c_allocator,

    pub fn is_valid(self: *const KpointApplyCache, expected_basis_len: usize) bool {
        return self.basis_len > 0 and self.basis_len == expected_basis_len and self.map != null;
    }

    pub fn deinit(self: *KpointApplyCache, _: std.mem.Allocator) void {
        if (self.nonlocal_ctx) |*ctx| ctx.deinit(self.cache_alloc);
        if (self.map) |*m| m.deinit(self.cache_alloc);
        self.* = .{};
    }
};

/// Shared allocation for ws_in_use atomic flag array.
fn alloc_ws_in_use(alloc: std.mem.Allocator, ws_count: usize) ![]std.atomic.Value(u8) {
    const ws_in_use = try alloc.alloc(std.atomic.Value(u8), ws_count);
    for (ws_in_use) |*entry| {
        entry.store(0, .release);
    }
    return ws_in_use;
}

/// Allocate and initialize `ws_count` workspaces sharing a single FFT plan.
fn alloc_shared_plan_workspaces(
    alloc: std.mem.Allocator,
    grid: Grid,
    n_gvecs: usize,
    max_m_total: usize,
    plan: fft.Fft3dPlan,
    ws_count: usize,
) ![]ApplyWorkspace {
    const workspaces = try alloc.alloc(ApplyWorkspace, ws_count);
    errdefer {
        for (workspaces) |*ws| {
            ws.deinit_without_plan(alloc);
        }
        alloc.free(workspaces);
    }
    var i: usize = 0;
    while (i < ws_count) : (i += 1) {
        workspaces[i] = try ApplyWorkspace.init_with_plan(alloc, grid, n_gvecs, max_m_total, plan);
    }
    return workspaces;
}

/// Parameters shared by all ApplyContext init paths; ownership flags encode who frees what.
const ApplyContextSetup = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    map: PwGridMap,
    gvecs: []const plane_wave.GVector,
    local_r: []const f64,
    vnl: ?[]math.Complex,
    nonlocal_ctx: ?NonlocalContext,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    profile: ?*ScfProfile,
    fft_plan: ?fft.Fft3dPlan,
    fft_index_map: ?[]const usize,
    owns_fft_plan: bool,
    owns_nonlocal: bool,
    owns_map: bool,
    workspaces: []ApplyWorkspace,
    num_workspaces: usize,
    ws_in_use: []std.atomic.Value(u8),
};

/// Build a fully-populated ApplyContext from setup params and the first workspace.
fn build_apply_context(s: ApplyContextSetup) ApplyContext {
    return .{
        .alloc = s.alloc,
        .io = s.io,
        .grid = s.grid,
        .map = s.map,
        .gvecs = s.gvecs,
        .local_r = s.local_r,
        .vnl = s.vnl,
        .nonlocal_ctx = s.nonlocal_ctx,
        .atoms = s.atoms,
        .inv_volume = s.inv_volume,
        .profile = s.profile,
        .fft_plan = s.fft_plan,
        .fft_index_map = s.fft_index_map,
        .owns_fft_plan = s.owns_fft_plan,
        .owns_nonlocal = s.owns_nonlocal,
        .owns_map = s.owns_map,
        .workspaces = s.workspaces,
        .num_workspaces = s.num_workspaces,
        .ws_in_use = s.ws_in_use,
        .ws_mutex = .init,
        .work_recip = s.workspaces[0].work_recip,
        .work_real = s.workspaces[0].work_real,
        .work_recip_out = s.workspaces[0].work_recip_out,
        .work_vec = s.workspaces[0].work_vec,
        .work_phase = s.workspaces[0].work_phase,
        .work_xphase = s.workspaces[0].work_xphase,
        .work_coeff = s.workspaces[0].work_coeff,
        .work_coeff2 = s.workspaces[0].work_coeff2,
    };
}

fn init_shared_workspace_state(
    alloc: std.mem.Allocator,
    grid: Grid,
    gvec_count: usize,
    max_m_total: usize,
    plan: fft.Fft3dPlan,
    num_workspaces: usize,
) !SharedWorkspaceState {
    const ws_count = @max(num_workspaces, 1);
    const workspaces = try alloc_shared_plan_workspaces(
        alloc,
        grid,
        gvec_count,
        max_m_total,
        plan,
        ws_count,
    );
    errdefer {
        for (workspaces) |*ws| {
            ws.deinit_without_plan(alloc);
        }
        alloc.free(workspaces);
    }

    const ws_in_use = try alloc_ws_in_use(alloc, ws_count);
    errdefer alloc.free(ws_in_use);
    return .{
        .count = ws_count,
        .workspaces = workspaces,
        .ws_in_use = ws_in_use,
    };
}

fn maybe_build_fft_indices(
    alloc: std.mem.Allocator,
    map: *PwGridMap,
    fft_index_map: ?[]const usize,
) !void {
    if (fft_index_map) |idx_map| {
        try map.build_fft_indices(alloc, idx_map);
    }
}

pub const ApplyContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    map: PwGridMap,
    gvecs: []const plane_wave.GVector,
    local_r: []const f64,
    vnl: ?[]math.Complex,
    nonlocal_ctx: ?NonlocalContext,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    profile: ?*ScfProfile,
    fft_plan: ?fft.Fft3dPlan,
    fft_index_map: ?[]const usize,
    owns_fft_plan: bool, // If true, deinit will free the FFT plan
    owns_nonlocal: bool, // If true, deinit will free nonlocal_ctx
    owns_map: bool, // If true, deinit will free map
    // Multiple workspaces for parallel execution
    workspaces: []ApplyWorkspace,
    num_workspaces: usize,
    // Workspace allocation tracking for parallel use
    ws_in_use: []std.atomic.Value(u8),
    ws_mutex: std.Io.Mutex,
    // First workspace mirrors for single-workspace helper paths.
    work_recip: []math.Complex,
    work_real: []math.Complex,
    work_recip_out: []math.Complex,
    work_vec: []math.Complex,
    work_phase: []math.Complex,
    work_xphase: []math.Complex,
    work_coeff: []math.Complex,
    work_coeff2: []math.Complex,

    /// Initialize with pre-created FFT plan (avoids mutex contention in parallel execution)
    pub fn init_with_fft_plan(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: Grid,
        gvecs: []const plane_wave.GVector,
        local_r: []const f64,
        vnl: ?[]math.Complex,
        species: []const hamiltonian.SpeciesEntry,
        atoms: []const hamiltonian.AtomData,
        inv_volume: f64,
        enable_nonlocal: bool,
        profile: ?*ScfProfile,
        fft_index_map: ?[]const usize,
        existing_plan: fft.Fft3dPlan,
    ) !ApplyContext {
        var nonlocal_ctx: ?NonlocalContext = null;
        if (enable_nonlocal) {
            nonlocal_ctx = try build_nonlocal_context_pub(alloc, species, gvecs);
        }
        errdefer if (nonlocal_ctx) |*ctx| ctx.deinit(alloc);

        var map = try PwGridMap.init(alloc, gvecs, grid);
        errdefer map.deinit(alloc);

        if (fft_index_map) |idx_map| {
            try map.build_fft_indices(alloc, idx_map);
        }

        const max_m_total = if (nonlocal_ctx) |ctx| ctx.max_m_total else 0;
        const workspaces = try alloc.alloc(ApplyWorkspace, 1);
        errdefer {
            for (workspaces) |*ws| {
                ws.deinit(alloc);
            }
            alloc.free(workspaces);
        }
        // Use init_with_plan to avoid creating another FFT plan
        workspaces[0] = try ApplyWorkspace.init_with_plan(
            alloc,
            grid,
            gvecs.len,
            max_m_total,
            existing_plan,
        );

        const ws_in_use = try alloc_ws_in_use(alloc, 1);
        errdefer alloc.free(ws_in_use);

        return build_apply_context(.{
            .alloc = alloc,
            .io = io,
            .grid = grid,
            .map = map,
            .gvecs = gvecs,
            .local_r = local_r,
            .vnl = vnl,
            .nonlocal_ctx = nonlocal_ctx,
            .atoms = atoms,
            .inv_volume = inv_volume,
            .profile = profile,
            .fft_plan = existing_plan,
            .fft_index_map = fft_index_map,
            .owns_fft_plan = false,
            .owns_nonlocal = true,
            .owns_map = true,
            .workspaces = workspaces,
            .num_workspaces = 1,
            .ws_in_use = ws_in_use,
        });
    }

    /// Initialize using cached NonlocalContext and PwGridMap (avoids expensive recomputation).
    /// The caller retains ownership of the cached components (nonlocal_ctx and map).
    pub fn init_with_cache(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: Grid,
        gvecs: []const plane_wave.GVector,
        local_r: []const f64,
        vnl: ?[]math.Complex,
        cached_nonlocal: ?NonlocalContext,
        cached_map: PwGridMap,
        atoms: []const hamiltonian.AtomData,
        inv_volume: f64,
        profile: ?*ScfProfile,
        fft_index_map: ?[]const usize,
        fft_plan: fft.Fft3dPlan,
        owns_plan: bool,
    ) !ApplyContext {
        const max_m_total = if (cached_nonlocal) |ctx| ctx.max_m_total else 0;
        const workspaces = try alloc.alloc(ApplyWorkspace, 1);
        errdefer alloc.free(workspaces);

        workspaces[0] = try ApplyWorkspace.init_with_plan(
            alloc,
            grid,
            gvecs.len,
            max_m_total,
            fft_plan,
        );

        const ws_in_use = try alloc_ws_in_use(alloc, 1);
        errdefer alloc.free(ws_in_use);

        return build_apply_context(.{
            .alloc = alloc,
            .io = io,
            .grid = grid,
            .map = cached_map, // Borrowed from cache
            .gvecs = gvecs,
            .local_r = local_r,
            .vnl = vnl,
            .nonlocal_ctx = cached_nonlocal, // Borrowed from cache
            .atoms = atoms,
            .inv_volume = inv_volume,
            .profile = profile,
            .fft_plan = fft_plan,
            .fft_index_map = fft_index_map,
            .owns_fft_plan = owns_plan,
            .owns_nonlocal = false, // Cache owns these
            .owns_map = false, // Cache owns these
            .workspaces = workspaces,
            .num_workspaces = 1,
            .ws_in_use = ws_in_use,
        });
    }

    /// Initialize with multiple workspaces for parallel execution
    pub fn init_with_workspaces(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: Grid,
        gvecs: []const plane_wave.GVector,
        local_r: []const f64,
        vnl: ?[]math.Complex,
        species: []const hamiltonian.SpeciesEntry,
        atoms: []const hamiltonian.AtomData,
        inv_volume: f64,
        enable_nonlocal: bool,
        profile: ?*ScfProfile,
        fft_index_map: ?[]const usize,
        fft_backend: fft.FftBackend,
        num_workspaces: usize,
    ) !ApplyContext {
        var nonlocal_ctx: ?NonlocalContext = null;
        if (enable_nonlocal) {
            nonlocal_ctx = try build_nonlocal_context_pub(alloc, species, gvecs);
        }
        errdefer if (nonlocal_ctx) |*ctx| ctx.deinit(alloc);

        var map = try PwGridMap.init(alloc, gvecs, grid);
        errdefer map.deinit(alloc);
        try maybe_build_fft_indices(alloc, &map, fft_index_map);

        var plan = try fft.Fft3dPlan.init_with_backend(
            alloc,
            io,
            grid.nx,
            grid.ny,
            grid.nz,
            fft_backend,
        );
        errdefer plan.deinit(alloc);

        const max_m_total = if (nonlocal_ctx) |ctx| ctx.max_m_total else 0;
        const workspace_state = try init_shared_workspace_state(
            alloc,
            grid,
            gvecs.len,
            max_m_total,
            plan,
            num_workspaces,
        );

        return build_apply_context(.{
            .alloc = alloc,
            .io = io,
            .grid = grid,
            .map = map,
            .gvecs = gvecs,
            .local_r = local_r,
            .vnl = vnl,
            .nonlocal_ctx = nonlocal_ctx,
            .atoms = atoms,
            .inv_volume = inv_volume,
            .profile = profile,
            .fft_plan = plan,
            .fft_index_map = fft_index_map,
            .owns_fft_plan = true,
            .owns_nonlocal = true,
            .owns_map = true,
            .workspaces = workspace_state.workspaces,
            .num_workspaces = workspace_state.count,
            .ws_in_use = workspace_state.ws_in_use,
        });
    }

    pub fn deinit(self: *ApplyContext, alloc: std.mem.Allocator) void {
        if (self.owns_nonlocal) {
            if (self.nonlocal_ctx) |*ctx| ctx.deinit(alloc);
        }
        if (self.owns_map) {
            self.map.deinit(alloc);
        }
        // Only free the FFT plan if we own it
        if (self.owns_fft_plan) {
            if (self.fft_plan) |*plan| plan.deinit(alloc);
        }
        for (self.workspaces) |*ws| {
            // Don't free workspace's plan (shared with context)
            ws.deinit_without_plan(alloc);
        }
        if (self.workspaces.len > 0) alloc.free(self.workspaces);
        if (self.ws_in_use.len > 0) alloc.free(self.ws_in_use);
    }

    /// Acquire an available workspace index for parallel use.
    pub fn acquire_workspace(self: *ApplyContext) ?usize {
        for (self.ws_in_use, 0..) |*entry, idx| {
            if (entry.cmpxchgStrong(0, 1, .acquire, .acquire) == null) {
                return idx;
            }
        }
        return null;
    }

    /// Release a workspace index back to the pool.
    pub fn release_workspace(self: *ApplyContext, idx: usize) void {
        if (idx < self.ws_in_use.len) {
            self.ws_in_use[idx].store(0, .release);
        }
    }

    /// Get workspace by index (wraps around if index >= num_workspaces)
    pub fn get_workspace(self: *ApplyContext, idx: usize) *ApplyWorkspace {
        return &self.workspaces[idx % self.num_workspaces];
    }
};

/// Apply Hamiltonian using pre-allocated workspace (no allocation during call)
fn apply_hamiltonian_with_workspace(
    ctx: *ApplyContext,
    ws: *ApplyWorkspace,
    x: []const math.Complex,
    y: []math.Complex,
) !void {
    const n = ctx.gvecs.len;
    if (x.len != n or y.len != n) return error.InvalidMatrixSize;

    const start = if (ctx.profile != null) profile_start(ctx.io) else null;

    // Kinetic energy: y = T*x
    var i: usize = 0;
    while (i < n) : (i += 1) {
        y[i] = math.complex.scale(x[i], ctx.gvecs[i].kinetic);
    }

    // Local potential: y += V_local * x
    const local_start = if (ctx.profile != null) profile_start(ctx.io) else null;
    const plan_ptr: ?*fft.Fft3dPlan = if (ws.fft_plan != null) @constCast(&ws.fft_plan.?) else null;
    try apply_local_potential_safe(
        ctx,
        x,
        ws.work_vec,
        ws.work_recip,
        ws.work_real,
        ws.work_recip_out,
        plan_ptr,
    );
    if (ctx.profile) |p| profile_add(ctx.io, &p.apply_local_ns, local_start);
    i = 0;
    while (i < n) : (i += 1) {
        y[i] = math.complex.add(y[i], ws.work_vec[i]);
    }

    // Nonlocal potential: y += V_nl * x
    if (ctx.nonlocal_ctx != null) {
        const nonlocal_start = if (ctx.profile != null) profile_start(ctx.io) else null;
        try apply_nonlocal_potential_safe(
            ctx,
            x,
            ws.work_vec,
            ws.work_phase,
            ws.work_xphase,
            ws.work_coeff,
            ws.work_coeff2,
        );
        if (ctx.profile) |p| profile_add(ctx.io, &p.apply_nonlocal_ns, nonlocal_start);
        i = 0;
        while (i < n) : (i += 1) {
            y[i] = math.complex.add(y[i], ws.work_vec[i]);
        }
    } else if (ctx.vnl) |mat| {
        apply_dense_matrix(n, mat, x, ws.work_vec);
        i = 0;
        while (i < n) : (i += 1) {
            y[i] = math.complex.add(y[i], ws.work_vec[i]);
        }
    }

    if (ctx.profile) |p| {
        profile_add(ctx.io, &p.apply_h_ns, start);
        p.apply_h_calls += 1;
    }
}

pub fn apply_hamiltonian(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
    const ctx: *ApplyContext = @ptrCast(@alignCast(ctx_ptr));

    // Try to acquire a workspace from the pool (thread-safe)
    if (ctx.acquire_workspace()) |ws_idx| {
        defer ctx.release_workspace(ws_idx);

        try apply_hamiltonian_with_workspace(ctx, &ctx.workspaces[ws_idx], x, y);
    } else {
        // All workspaces in use - use first one with lock
        ctx.ws_mutex.lockUncancelable(ctx.io);
        defer ctx.ws_mutex.unlock(ctx.io);

        try apply_hamiltonian_with_workspace(ctx, &ctx.workspaces[0], x, y);
    }
}

pub fn check_hamiltonian_apply(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    gvecs: []const plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    potential: hamiltonian.PotentialGrid,
    local_r: []const f64,
    nonlocal_enabled: bool,
    fft_index_map: ?[]const usize,
) !void {
    var ctx = try ApplyContext.init_with_workspaces(
        alloc,
        io,
        grid,
        gvecs,
        local_r,
        null,
        species,
        atoms,
        inv_volume,
        nonlocal_enabled,
        null,
        fft_index_map,
        .zig, // Use default Zig FFT for testing
        1,
    );
    defer ctx.deinit(alloc);

    const n = gvecs.len;
    const x = try alloc.alloc(math.Complex, n);
    const y_apply = try alloc.alloc(math.Complex, n);
    const y_dense = try alloc.alloc(math.Complex, n);
    defer alloc.free(x);
    defer alloc.free(y_apply);
    defer alloc.free(y_dense);

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const fi = @as(f64, @floatFromInt(i + 1));
        x[i] = math.complex.init(std.math.sin(fi), std.math.cos(fi));
    }
    try apply_hamiltonian(&ctx, x, y_apply);
    const cfg = local_potential.LocalPotentialConfig.init(.short_range, 0.0);
    const h = try hamiltonian
        .build_hamiltonian(alloc, gvecs, species, atoms, inv_volume, cfg, potential);
    defer alloc.free(h);

    apply_dense_matrix(n, h, x, y_dense);
    var max_abs: f64 = 0.0;
    var max_diff: f64 = 0.0;
    i = 0;
    while (i < n) : (i += 1) {
        const dr = y_apply[i].r - y_dense[i].r;
        const di = y_apply[i].i - y_dense[i].i;
        const diff = std.math.sqrt(dr * dr + di * di);
        const abs_val = std.math.sqrt(y_dense[i].r * y_dense[i].r + y_dense[i].i * y_dense[i].i);
        if (diff > max_diff) max_diff = diff;
        if (abs_val > max_abs) max_abs = abs_val;
    }
    const rel = if (max_abs > 0.0) max_diff / max_abs else 0.0;
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(
        .info,
        "scf: apply_check max_abs={d:.6} max_diff={d:.6} rel={d:.6}\n",
        .{ max_abs, max_diff, rel },
    );
}

fn apply_dense_matrix(
    n: usize,
    mat: []const math.Complex,
    x: []const math.Complex,
    out: []math.Complex,
) void {
    @memset(out, math.complex.init(0.0, 0.0));
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const hij = mat[i + j * n];
            out[i] = math.complex.add(out[i], math.complex.mul(hij, x[j]));
        }
    }
}

fn apply_local_potential(ctx: *ApplyContext, x: []const math.Complex, out: []math.Complex) !void {
    ctx.map.scatter(x, ctx.work_recip);
    const plan_ptr = if (ctx.fft_plan) |*p| p else null;
    if (ctx.fft_index_map) |map| {
        try fft_reciprocal_to_complex_in_place_mapped(
            ctx.alloc,
            ctx.grid,
            map,
            ctx.work_recip,
            ctx.work_real,
            plan_ptr,
        );
    } else {
        try fft_reciprocal_to_complex_in_place(
            ctx.alloc,
            ctx.grid,
            ctx.work_recip,
            ctx.work_real,
            plan_ptr,
        );
    }
    for (ctx.work_real, 0..) |*v, i| {
        v.* = math.complex.scale(v.*, ctx.local_r[i]);
    }
    if (ctx.fft_index_map) |map| {
        try fft_complex_to_reciprocal_in_place_mapped(
            ctx.alloc,
            ctx.grid,
            map,
            ctx.work_real,
            ctx.work_recip_out,
            plan_ptr,
        );
    } else {
        try fft_complex_to_reciprocal_in_place(
            ctx.alloc,
            ctx.grid,
            ctx.work_real,
            ctx.work_recip_out,
            plan_ptr,
        );
    }
    ctx.map.gather(ctx.work_recip_out, out);
}

pub fn apply_nonlocal_potential(
    ctx: *ApplyContext,
    x: []const math.Complex,
    out: []math.Complex,
) !void {
    try apply_nonlocal_potential_safe(
        ctx,
        x,
        out,
        ctx.work_phase,
        ctx.work_xphase,
        ctx.work_coeff,
        ctx.work_coeff2,
    );
}

/// Thread-safe version of apply_local_potential with explicit work buffers
/// Uses provided FFT plan (from workspace) for efficiency
fn apply_local_potential_safe(
    ctx: *ApplyContext,
    x: []const math.Complex,
    out: []math.Complex,
    work_recip: []math.Complex,
    work_real: []math.Complex,
    work_recip_out: []math.Complex,
    plan: ?*fft.Fft3dPlan,
) !void {
    if (ctx.map.fft_indices.len > 0) {
        // Fused path: scatter/gather directly in FFT order, skip full-grid remap
        const total: f64 = @floatFromInt(ctx.grid.count());
        const inv_scale = 1.0 / total;
        const t_scatter = if (ctx.profile != null) profile_start(ctx.io) else null;
        ctx.map.scatter_fft(x, work_real, total);
        if (ctx.profile) |p| profile_add(ctx.io, &p.local_scatter_ns, t_scatter);
        const t_ifft = if (ctx.profile != null) profile_start(ctx.io) else null;
        if (plan) |p| {
            try fft.fft3d_inverse_in_place_plan(p, work_real);
        } else {
            const g = ctx.grid;
            try fft.fft3d_inverse_in_place(ctx.alloc, work_real, g.nx, g.ny, g.nz);
        }
        if (ctx.profile) |p| profile_add(ctx.io, &p.local_ifft_ns, t_ifft);
        const t_vmul = if (ctx.profile != null) profile_start(ctx.io) else null;
        for (work_real, 0..) |*v, i| {
            v.* = math.complex.scale(v.*, ctx.local_r[i]);
        }
        if (ctx.profile) |p| profile_add(ctx.io, &p.local_vmul_ns, t_vmul);
        // FFT in-place
        const t_fft = if (ctx.profile != null) profile_start(ctx.io) else null;
        if (plan) |p| {
            try fft.fft3d_forward_in_place_plan(p, work_real);
        } else {
            const g = ctx.grid;
            try fft.fft3d_forward_in_place(ctx.alloc, work_real, g.nx, g.ny, g.nz);
        }
        if (ctx.profile) |p| profile_add(ctx.io, &p.local_fft_ns, t_fft);
        // Gather FFT order -> PW with FFT scaling
        const t_gather = if (ctx.profile != null) profile_start(ctx.io) else null;
        ctx.map.gather_fft(work_real, out, inv_scale);
        if (ctx.profile) |p| profile_add(ctx.io, &p.local_gather_ns, t_gather);
    } else {
        // Original path with full-grid remap
        const a = ctx.alloc;
        const g = ctx.grid;
        ctx.map.scatter(x, work_recip);
        if (ctx.fft_index_map) |map| {
            try fft_reciprocal_to_complex_in_place_mapped(a, g, map, work_recip, work_real, plan);
        } else {
            try fft_reciprocal_to_complex_in_place(a, g, work_recip, work_real, plan);
        }
        for (work_real, 0..) |*v, i| {
            v.* = math.complex.scale(v.*, ctx.local_r[i]);
        }
        if (ctx.fft_index_map) |map| {
            try fft_complex_to_reciprocal_in_place_mapped(
                a,
                g,
                map,
                work_real,
                work_recip_out,
                plan,
            );
        } else {
            try fft_complex_to_reciprocal_in_place(a, g, work_real, work_recip_out, plan);
        }
        ctx.map.gather(work_recip_out, out);
    }
}

/// Compute atom phase factors and x*phase for subsequent projection.
fn compute_atom_phase(
    gvecs: []const plane_wave.GVector,
    position: math.Vec3,
    x: []const math.Complex,
    work_phase: []math.Complex,
    work_xphase: []math.Complex,
    n: usize,
) void {
    var g: usize = 0;
    while (g < n) : (g += 1) {
        const phase = math.complex.expi(math.Vec3.dot(gvecs[g].cart, position));
        work_phase[g] = phase;
        work_xphase[g] = math.complex.mul(x[g], phase);
    }
}

/// Step 1: coeff[β,m] = Σ_g phi[β,m,g] * work_xphase[g]
fn project_nonlocal_coeff(
    entry: *const nonlocal_context.NonlocalSpecies,
    work_xphase: []const math.Complex,
    coeff: []math.Complex,
    g_count: usize,
    n: usize,
) void {
    var b: usize = 0;
    while (b < entry.beta_count) : (b += 1) {
        const offset = entry.m_offsets[b];
        const m_count = entry.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const phi_start = (offset + m_idx) * g_count;
            const phi = entry.phi[phi_start .. phi_start + g_count];
            var sum = math.complex.init(0.0, 0.0);
            var g: usize = 0;
            while (g < n) : (g += 1) {
                sum = math.complex.add(sum, math.complex.scale(work_xphase[g], phi[g]));
            }
            coeff[offset + m_idx] = sum;
        }
    }
}

/// Step 2: coeff2 = D_ij * coeff (m-resolved or radial form).
fn apply_nonlocal_dij(
    entry: *const nonlocal_context.NonlocalSpecies,
    coeffs: []const f64,
    dij_m: ?[]const f64,
    coeff: []const math.Complex,
    coeff2: []math.Complex,
) void {
    if (dij_m) |dm| {
        const mt = entry.m_total;
        var im: usize = 0;
        while (im < mt) : (im += 1) {
            var sum = math.complex.init(0.0, 0.0);
            var jm: usize = 0;
            while (jm < mt) : (jm += 1) {
                const d_val = dm[im * mt + jm];
                if (d_val == 0.0) continue;
                sum = math.complex.add(sum, math.complex.scale(coeff[jm], d_val));
            }
            coeff2[im] = sum;
        }
    } else {
        var b: usize = 0;
        while (b < entry.beta_count) : (b += 1) {
            const l_val = entry.l_list[b];
            const offset = entry.m_offsets[b];
            const m_count = entry.m_counts[b];
            var m_idx: usize = 0;
            while (m_idx < m_count) : (m_idx += 1) {
                var sum = math.complex.init(0.0, 0.0);
                var j: usize = 0;
                while (j < entry.beta_count) : (j += 1) {
                    if (entry.l_list[j] != l_val) continue;
                    const dij = coeffs[b * entry.beta_count + j];
                    if (dij == 0.0) continue;
                    const c = coeff[entry.m_offsets[j] + m_idx];
                    sum = math.complex.add(sum, math.complex.scale(c, dij));
                }
                coeff2[offset + m_idx] = sum;
            }
        }
    }
}

/// Step 3: out[g] += (1/Ω) * conj(phase[g]) * Σ_{β,m} phi[β,m,g] * coeff2[β,m]
fn backproject_nonlocal_accum(
    entry: *const nonlocal_context.NonlocalSpecies,
    coeff2: []const math.Complex,
    work_phase: []const math.Complex,
    out: []math.Complex,
    inv_volume: f64,
    g_count: usize,
    n: usize,
) void {
    var g: usize = 0;
    while (g < n) : (g += 1) {
        var accum = math.complex.init(0.0, 0.0);
        var b: usize = 0;
        while (b < entry.beta_count) : (b += 1) {
            const offset = entry.m_offsets[b];
            const m_count = entry.m_counts[b];
            var m_idx: usize = 0;
            while (m_idx < m_count) : (m_idx += 1) {
                const phi_val = entry.phi[(offset + m_idx) * g_count + g];
                const c = coeff2[offset + m_idx];
                accum = math.complex.add(accum, math.complex.scale(c, phi_val));
            }
        }
        const phase_conj = math.complex.conj(work_phase[g]);
        const add = math.complex.mul(phase_conj, accum);
        out[g] = math.complex.add(out[g], math.complex.scale(add, inv_volume));
    }
}

/// Thread-safe version of apply_nonlocal_potential with explicit work buffers
fn apply_nonlocal_potential_safe(
    ctx: *ApplyContext,
    x: []const math.Complex,
    out: []math.Complex,
    work_phase: []math.Complex,
    work_xphase: []math.Complex,
    work_coeff: []math.Complex,
    work_coeff2: []math.Complex,
) !void {
    const nl = ctx.nonlocal_ctx orelse {
        @memset(out, math.complex.init(0.0, 0.0));
        return;
    };
    const n = ctx.gvecs.len;
    @memset(out, math.complex.init(0.0, 0.0));

    for (nl.species) |entry| {
        const g_count = entry.g_count;
        if (g_count != n) return error.InvalidMatrixSize;
        if (entry.m_total == 0) continue;
        const coeff = work_coeff[0..entry.m_total];
        const coeff2 = work_coeff2[0..entry.m_total];

        var atom_of_species: usize = 0;
        for (ctx.atoms) |atom| {
            if (atom.species_index != entry.species_index) continue;
            const coeffs = if (entry.dij_per_atom) |dpa| dpa[atom_of_species] else entry.coeffs;
            const dij_m = if (entry.dij_m_per_atom) |dpa| dpa[atom_of_species] else null;
            atom_of_species += 1;

            compute_atom_phase(ctx.gvecs, atom.position, x, work_phase, work_xphase, n);
            project_nonlocal_coeff(&entry, work_xphase, coeff, g_count, n);
            apply_nonlocal_dij(&entry, coeffs, dij_m, coeff, coeff2);
            backproject_nonlocal_accum(&entry, coeff2, work_phase, out, ctx.inv_volume, g_count, n);
        }
    }
}

/// Apply PAW overlap operator: S|ψ> = |ψ> + Σ_a Σ_{ij} q_ij |p_i^a><p_j^a|ψ>
/// This is the S operator for the generalized eigenvalue problem H|ψ> = ε S|ψ>.
/// Uses the same projector infrastructure as the nonlocal potential but with q_ij coefficients.
pub fn apply_overlap(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
    const ctx: *ApplyContext = @ptrCast(@alignCast(ctx_ptr));

    if (ctx.acquire_workspace()) |ws_idx| {
        defer ctx.release_workspace(ws_idx);

        const ws = &ctx.workspaces[ws_idx];
        try apply_overlap_safe(
            ctx,
            x,
            y,
            ws.work_phase,
            ws.work_xphase,
            ws.work_coeff,
            ws.work_coeff2,
        );
    } else {
        ctx.ws_mutex.lockUncancelable(ctx.io);
        defer ctx.ws_mutex.unlock(ctx.io);

        const ws = &ctx.workspaces[0];
        try apply_overlap_safe(
            ctx,
            x,
            y,
            ws.work_phase,
            ws.work_xphase,
            ws.work_coeff,
            ws.work_coeff2,
        );
    }
}

/// Thread-safe overlap operator application with explicit work buffers.
/// y = S·x = x + Σ_a Σ_{ij} q_ij <p_j|x> |p_i>
fn apply_overlap_safe(
    ctx: *ApplyContext,
    x: []const math.Complex,
    out: []math.Complex,
    work_phase: []math.Complex,
    work_xphase: []math.Complex,
    work_coeff: []math.Complex,
    work_coeff2: []math.Complex,
) !void {
    const nl = ctx.nonlocal_ctx orelse {
        // No nonlocal context: S = I
        @memcpy(out, x);
        return;
    };
    const n = ctx.gvecs.len;
    // Start with identity: y = x
    @memcpy(out[0..n], x[0..n]);

    for (nl.species) |entry| {
        const qij = entry.overlap_coeffs orelse continue; // Skip non-PAW species
        const g_count = entry.g_count;
        if (g_count != n) return error.InvalidMatrixSize;
        if (entry.m_total == 0) continue;
        const coeff = work_coeff[0..entry.m_total];
        const coeff2 = work_coeff2[0..entry.m_total];

        for (ctx.atoms) |atom| {
            if (atom.species_index != entry.species_index) continue;

            compute_atom_phase(ctx.gvecs, atom.position, x, work_phase, work_xphase, n);
            project_nonlocal_coeff(&entry, work_xphase, coeff, g_count, n);
            apply_nonlocal_dij(&entry, qij, null, coeff, coeff2);
            backproject_nonlocal_accum(&entry, coeff2, work_phase, out, ctx.inv_volume, g_count, n);
        }
    }
}

/// Batched nonlocal potential application using BLAS-3 (dgemm).
/// Processes ncols input vectors simultaneously for better cache efficiency.
/// x_batch and out_batch are column-major: x_batch[col * n_pw + g].
/// out_batch is zeroed and filled (not accumulated).
/// Step 1+2: Fill work_phase[g] = exp(i G·τ) and xphase[g,b] = x_batch[b,g] * phase[g].
fn batch_phase_and_xphase(
    gvecs: []const plane_wave.GVector,
    position: math.Vec3,
    x_batch: []const math.Complex,
    work_phase: []math.Complex,
    xphase_re: []f64,
    xphase_im: []f64,
    n_pw: usize,
    ncols: usize,
) void {
    for (0..n_pw) |g| {
        work_phase[g] = math.complex.expi(math.Vec3.dot(gvecs[g].cart, position));
    }
    for (0..n_pw) |g| {
        const ph = work_phase[g];
        const row = g * ncols;
        for (0..ncols) |col| {
            const xv = x_batch[col * n_pw + g];
            xphase_re[row + col] = xv.r * ph.r - xv.i * ph.i;
            xphase_im[row + col] = xv.r * ph.i + xv.i * ph.r;
        }
    }
}

/// Step 3: coeffs = phi * xphase (real and imaginary parts).
fn batch_project_coeffs(
    phi: []const f64,
    xphase_re: []const f64,
    xphase_im: []const f64,
    coeffs_re: []f64,
    coeffs_im: []f64,
    m_total: usize,
    g_count: usize,
    ncols: usize,
) void {
    blas.dgemm(
        .no_trans,
        .no_trans,
        m_total,
        ncols,
        g_count,
        1.0,
        phi,
        g_count,
        xphase_re,
        ncols,
        0.0,
        coeffs_re[0 .. m_total * ncols],
        ncols,
    );
    blas.dgemm(
        .no_trans,
        .no_trans,
        m_total,
        ncols,
        g_count,
        1.0,
        phi,
        g_count,
        xphase_im,
        ncols,
        0.0,
        coeffs_im[0 .. m_total * ncols],
        ncols,
    );
}

/// Step 4: coeffs2 = D * coeffs (m-resolved via BLAS or radial sparse).
fn batch_apply_dij_radial(
    entry: *const nonlocal_context.NonlocalSpecies,
    dij_coeffs: []const f64,
    coeffs_re: []const f64,
    coeffs_im: []const f64,
    coeffs2_re: []f64,
    coeffs2_im: []f64,
    m_total: usize,
    ncols: usize,
) void {
    @memset(coeffs2_re[0 .. m_total * ncols], 0.0);
    @memset(coeffs2_im[0 .. m_total * ncols], 0.0);

    var b_idx: usize = 0;
    while (b_idx < entry.beta_count) : (b_idx += 1) {
        const l_val = entry.l_list[b_idx];
        const offset = entry.m_offsets[b_idx];
        const m_count = entry.m_counts[b_idx];

        var j: usize = 0;
        while (j < entry.beta_count) : (j += 1) {
            if (entry.l_list[j] != l_val) continue;
            const dij = dij_coeffs[b_idx * entry.beta_count + j];
            if (dij == 0.0) continue;
            const j_offset = entry.m_offsets[j];

            var m_idx: usize = 0;
            while (m_idx < m_count) : (m_idx += 1) {
                const src_row = (j_offset + m_idx) * ncols;
                const dst_row = (offset + m_idx) * ncols;
                for (0..ncols) |col| {
                    coeffs2_re[dst_row + col] += dij * coeffs_re[src_row + col];
                    coeffs2_im[dst_row + col] += dij * coeffs_im[src_row + col];
                }
            }
        }
    }
}

fn batch_apply_dij_dense(
    dm: []const f64,
    coeffs_src: []const f64,
    coeffs_dst: []f64,
    m_total: usize,
    ncols: usize,
) void {
    blas.dgemm(
        .no_trans,
        .no_trans,
        m_total,
        ncols,
        m_total,
        1.0,
        dm,
        m_total,
        coeffs_src[0 .. m_total * ncols],
        ncols,
        0.0,
        coeffs_dst[0 .. m_total * ncols],
        ncols,
    );
}

fn batch_apply_dij(
    entry: *const nonlocal_context.NonlocalSpecies,
    dij_coeffs: []const f64,
    dij_m: ?[]const f64,
    coeffs_re: []const f64,
    coeffs_im: []const f64,
    coeffs2_re: []f64,
    coeffs2_im: []f64,
    m_total: usize,
    ncols: usize,
) void {
    if (dij_m) |dm| {
        batch_apply_dij_dense(dm, coeffs_re, coeffs2_re, m_total, ncols);
        batch_apply_dij_dense(dm, coeffs_im, coeffs2_im, m_total, ncols);
        return;
    }
    batch_apply_dij_radial(
        entry,
        dij_coeffs,
        coeffs_re,
        coeffs_im,
        coeffs2_re,
        coeffs2_im,
        m_total,
        ncols,
    );
}

/// Step 5+6: result = phi^T * coeffs2; out_batch += (1/Ω) * conj(phase) * result.
fn batch_backproject_and_accumulate(
    phi: []const f64,
    coeffs2_re: []const f64,
    coeffs2_im: []const f64,
    xphase_re: []f64,
    xphase_im: []f64,
    work_phase: []const math.Complex,
    out_batch: []math.Complex,
    inv_volume: f64,
    m_total: usize,
    g_count: usize,
    n_pw: usize,
    ncols: usize,
) void {
    blas.dgemm(
        .trans,
        .no_trans,
        g_count,
        ncols,
        m_total,
        1.0,
        phi,
        g_count,
        coeffs2_re[0 .. m_total * ncols],
        ncols,
        0.0,
        xphase_re,
        ncols,
    );
    blas.dgemm(
        .trans,
        .no_trans,
        g_count,
        ncols,
        m_total,
        1.0,
        phi,
        g_count,
        coeffs2_im[0 .. m_total * ncols],
        ncols,
        0.0,
        xphase_im,
        ncols,
    );

    for (0..n_pw) |g| {
        const ph_conj = math.complex.conj(work_phase[g]);
        const row = g * ncols;
        for (0..ncols) |col| {
            const re = xphase_re[row + col];
            const im = xphase_im[row + col];
            const out_r = (ph_conj.r * re - ph_conj.i * im) * inv_volume;
            const out_i = (ph_conj.r * im + ph_conj.i * re) * inv_volume;
            const idx = col * n_pw + g;
            const add_val = math.complex.init(out_r, out_i);
            out_batch[idx] = math.complex.add(out_batch[idx], add_val);
        }
    }
}

const NonlocalBatchBuffers = struct {
    gn: []f64,
    mn: []f64,
    xphase_re: []f64,
    xphase_im: []f64,
    coeffs_re: []f64,
    coeffs_im: []f64,
    coeffs2_re: []f64,
    coeffs2_im: []f64,

    fn init(
        alloc: std.mem.Allocator,
        n_pw: usize,
        ncols: usize,
        max_m: usize,
    ) !NonlocalBatchBuffers {
        const gn = try alloc.alloc(f64, n_pw * ncols * 2);
        errdefer alloc.free(gn);

        const mn = try alloc.alloc(f64, max_m * ncols * 4);
        errdefer alloc.free(mn);

        return .{
            .gn = gn,
            .mn = mn,
            .xphase_re = gn[0 .. n_pw * ncols],
            .xphase_im = gn[n_pw * ncols .. 2 * n_pw * ncols],
            .coeffs_re = mn[0 .. max_m * ncols],
            .coeffs_im = mn[max_m * ncols .. 2 * max_m * ncols],
            .coeffs2_re = mn[2 * max_m * ncols .. 3 * max_m * ncols],
            .coeffs2_im = mn[3 * max_m * ncols .. 4 * max_m * ncols],
        };
    }

    fn deinit(self: *const NonlocalBatchBuffers, alloc: std.mem.Allocator) void {
        alloc.free(self.mn);
        alloc.free(self.gn);
    }
};

fn accumulate_species_nonlocal_batch(
    ctx: *ApplyContext,
    entry: *const NonlocalSpecies,
    x_batch: []const math.Complex,
    out_batch: []math.Complex,
    n_pw: usize,
    ncols: usize,
    work_phase: []math.Complex,
    bufs: *const NonlocalBatchBuffers,
) !void {
    const g_count = entry.g_count;
    if (g_count != n_pw) return error.InvalidMatrixSize;
    if (entry.m_total == 0) return;

    const m_total = entry.m_total;
    const phi = entry.phi[0 .. m_total * g_count];
    var atom_of_species: usize = 0;
    for (ctx.atoms) |atom| {
        if (atom.species_index != entry.species_index) continue;
        const dij_coeffs = if (entry.dij_per_atom) |dpa| dpa[atom_of_species] else entry.coeffs;
        const dij_m = if (entry.dij_m_per_atom) |dpa| dpa[atom_of_species] else null;
        atom_of_species += 1;

        batch_phase_and_xphase(
            ctx.gvecs,
            atom.position,
            x_batch,
            work_phase,
            bufs.xphase_re,
            bufs.xphase_im,
            n_pw,
            ncols,
        );
        batch_project_coeffs(
            phi,
            bufs.xphase_re,
            bufs.xphase_im,
            bufs.coeffs_re,
            bufs.coeffs_im,
            m_total,
            g_count,
            ncols,
        );
        batch_apply_dij(
            entry,
            dij_coeffs,
            dij_m,
            bufs.coeffs_re,
            bufs.coeffs_im,
            bufs.coeffs2_re,
            bufs.coeffs2_im,
            m_total,
            ncols,
        );
        batch_backproject_and_accumulate(
            phi,
            bufs.coeffs2_re,
            bufs.coeffs2_im,
            bufs.xphase_re,
            bufs.xphase_im,
            work_phase,
            out_batch,
            ctx.inv_volume,
            m_total,
            g_count,
            n_pw,
            ncols,
        );
    }
}

fn apply_nonlocal_batched(
    ctx: *ApplyContext,
    alloc: std.mem.Allocator,
    x_batch: []const math.Complex,
    out_batch: []math.Complex,
    n_pw: usize,
    ncols: usize,
    work_phase: []math.Complex,
) !void {
    const nl = ctx.nonlocal_ctx orelse {
        @memset(out_batch[0 .. n_pw * ncols], math.complex.init(0.0, 0.0));
        return;
    };
    @memset(out_batch[0 .. n_pw * ncols], math.complex.init(0.0, 0.0));

    const max_m = nl.max_m_total;
    if (max_m == 0) return;

    const bufs = try NonlocalBatchBuffers.init(alloc, n_pw, ncols, max_m);
    defer bufs.deinit(alloc);

    for (nl.species) |entry| {
        try accumulate_species_nonlocal_batch(
            ctx,
            &entry,
            x_batch,
            out_batch,
            n_pw,
            ncols,
            work_phase,
            &bufs,
        );
    }
}

/// Batched Hamiltonian application: y_batch = H * x_batch for ncols vectors.
/// x_batch and y_batch are column-major: [n_pw × ncols].
/// Uses BLAS-3 batched nonlocal for better cache efficiency.
/// Per-column kinetic + local potential accumulation into y.
fn apply_kinetic_and_local_for_column(
    ctx: *ApplyContext,
    x: []const math.Complex,
    y: []math.Complex,
    n_pw: usize,
) !void {
    for (0..n_pw) |g| {
        y[g] = math.complex.scale(x[g], ctx.gvecs[g].kinetic);
    }
    if (ctx.acquire_workspace()) |ws_idx| {
        defer ctx.release_workspace(ws_idx);

        const ws = &ctx.workspaces[ws_idx];
        const plan_ptr: ?*fft.Fft3dPlan = if (ws.fft_plan != null)
            @constCast(&ws.fft_plan.?)
        else
            null;
        try apply_local_potential_safe(
            ctx,
            x,
            ws.work_vec,
            ws.work_recip,
            ws.work_real,
            ws.work_recip_out,
            plan_ptr,
        );
        for (0..n_pw) |g| {
            y[g] = math.complex.add(y[g], ws.work_vec[g]);
        }
    } else {
        ctx.ws_mutex.lockUncancelable(ctx.io);
        defer ctx.ws_mutex.unlock(ctx.io);

        const ws = &ctx.workspaces[0];
        const plan_ptr: ?*fft.Fft3dPlan = if (ws.fft_plan != null)
            @constCast(&ws.fft_plan.?)
        else
            null;
        try apply_local_potential_safe(
            ctx,
            x,
            ws.work_vec,
            ws.work_recip,
            ws.work_real,
            ws.work_recip_out,
            plan_ptr,
        );
        for (0..n_pw) |g| {
            y[g] = math.complex.add(y[g], ws.work_vec[g]);
        }
    }
}

/// Accumulate batched nonlocal potential contribution into y_batch.
fn accumulate_batched_nonlocal(
    ctx: *ApplyContext,
    x_batch: []const math.Complex,
    y_batch: []math.Complex,
    n_pw: usize,
    ncols: usize,
) !void {
    const nl_out = try ctx.alloc.alloc(math.Complex, n_pw * ncols);
    defer ctx.alloc.free(nl_out);

    const work_phase = try ctx.alloc.alloc(math.Complex, n_pw);
    defer ctx.alloc.free(work_phase);

    try apply_nonlocal_batched(ctx, ctx.alloc, x_batch, nl_out, n_pw, ncols, work_phase);

    for (0..n_pw * ncols) |i| {
        y_batch[i] = math.complex.add(y_batch[i], nl_out[i]);
    }
}

/// Accumulate dense-matrix nonlocal contribution into y_batch (no NonlocalContext path).
fn accumulate_batched_dense_vnl(
    ctx: *ApplyContext,
    mat: []const math.Complex,
    x_batch: []const math.Complex,
    y_batch: []math.Complex,
    n_pw: usize,
    ncols: usize,
) !void {
    for (0..ncols) |col| {
        const x = x_batch[col * n_pw .. (col + 1) * n_pw];
        const ws_vec = try ctx.alloc.alloc(math.Complex, n_pw);
        defer ctx.alloc.free(ws_vec);

        apply_dense_matrix(n_pw, mat, x, ws_vec);
        const y = y_batch[col * n_pw .. (col + 1) * n_pw];
        for (0..n_pw) |g| {
            y[g] = math.complex.add(y[g], ws_vec[g]);
        }
    }
}

pub fn apply_hamiltonian_batched(
    ctx_ptr: *anyopaque,
    x_batch: []const math.Complex,
    y_batch: []math.Complex,
    n_pw: usize,
    ncols: usize,
) !void {
    const ctx: *ApplyContext = @ptrCast(@alignCast(ctx_ptr));
    if (n_pw != ctx.gvecs.len) return error.InvalidMatrixSize;

    const h_start = if (ctx.profile != null) profile_start(ctx.io) else null;

    // Kinetic + local: per-vector (FFT can't easily be batched)
    const local_start = if (ctx.profile != null) profile_start(ctx.io) else null;
    for (0..ncols) |col| {
        const x = x_batch[col * n_pw .. (col + 1) * n_pw];
        const y = y_batch[col * n_pw .. (col + 1) * n_pw];
        try apply_kinetic_and_local_for_column(ctx, x, y, n_pw);
    }
    if (ctx.profile) |p| profile_add(ctx.io, &p.apply_local_ns, local_start);

    // Nonlocal potential: batched BLAS-3
    if (ctx.nonlocal_ctx != null) {
        const nonlocal_start = if (ctx.profile != null) profile_start(ctx.io) else null;
        try accumulate_batched_nonlocal(ctx, x_batch, y_batch, n_pw, ncols);
        if (ctx.profile) |p| profile_add(ctx.io, &p.apply_nonlocal_ns, nonlocal_start);
    } else if (ctx.vnl) |mat| {
        try accumulate_batched_dense_vnl(ctx, mat, x_batch, y_batch, n_pw, ncols);
    }

    if (ctx.profile) |p| {
        profile_add(ctx.io, &p.apply_h_ns, h_start);
        p.apply_h_calls += 1;
    }
}

/// Compute projector overlaps and accumulate m-resolved PAW occupation matrix rhoij.
/// Must be called for each occupied band at each k-point.
/// Uses the NonlocalContext's phi arrays to compute <p_{β,m}|ψ> projections.
///
/// The rhoij matrix is m-resolved: ρ_{(i,m_i),(j,m_j)} stores individual
/// projector-m products (no m-summation). This enables multi-L on-site density.
/// Accumulate Re[<p_i|ψ> conj(<p_j|ψ>)] * effective_weight into rhoij for one atom.
fn accumulate_rho_ij_for_atom(
    work_coeff: []const math.Complex,
    rhoij: *paw_mod.RhoIJ,
    atom_idx: usize,
    m_total: usize,
    effective_weight: f64,
) void {
    const mt = rhoij.m_total_per_atom[atom_idx];
    var im: usize = 0;
    while (im < m_total) : (im += 1) {
        const ci = work_coeff[im];
        var jm: usize = 0;
        while (jm < m_total) : (jm += 1) {
            const cj = work_coeff[jm];
            const rho_val = ci.r * cj.r + ci.i * cj.i;
            rhoij.values[atom_idx][im * mt + jm] += effective_weight * rho_val;
        }
    }
}

pub fn accumulate_paw_rho_ij(
    alloc: std.mem.Allocator,
    nl: *const NonlocalContext,
    gvecs: []const plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    psi: []const math.Complex,
    weight: f64,
    inv_volume: f64,
    rhoij: *paw_mod.RhoIJ,
) !void {
    // The projector form factors phi include no 1/Ω factor, but the nonlocal
    // Hamiltonian multiplies by inv_volume. For consistency, ρ_ij must also
    // include the inv_volume factor: ρ_ij = (1/Ω) Σ f_n <ψ|p_i><p_j|ψ>.
    const effective_weight = weight * inv_volume;
    const n = gvecs.len;
    const work_phase = try alloc.alloc(math.Complex, n);
    defer alloc.free(work_phase);

    const work_xphase = try alloc.alloc(math.Complex, n);
    defer alloc.free(work_xphase);

    const work_coeff = try alloc.alloc(math.Complex, nl.max_m_total);
    defer alloc.free(work_coeff);

    for (nl.species) |entry| {
        if (entry.overlap_coeffs == null) continue; // Skip non-PAW species
        const g_count = entry.g_count;
        if (g_count != n) continue;
        if (entry.m_total == 0) continue;

        for (atoms, 0..) |atom, atom_idx| {
            if (atom.species_index != entry.species_index) continue;

            compute_atom_phase(gvecs, atom.position, psi, work_phase, work_xphase, n);
            project_nonlocal_coeff(&entry, work_xphase, work_coeff[0..entry.m_total], g_count, n);
            accumulate_rho_ij_for_atom(
                work_coeff,
                rhoij,
                atom_idx,
                entry.m_total,
                effective_weight,
            );
        }
    }
}
