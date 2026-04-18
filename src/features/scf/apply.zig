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

pub const Grid = grid_mod.Grid;
pub const ScfProfile = logging.ScfProfile;

// Re-export from nonlocal_context.zig
pub const NonlocalSpecies = nonlocal_context.NonlocalSpecies;
pub const NonlocalContext = nonlocal_context.NonlocalContext;
pub const buildNonlocalContextPub = nonlocal_context.buildNonlocalContextPub;
pub const buildNonlocalContextWithTables = nonlocal_context.buildNonlocalContextWithTables;
pub const buildNonlocalContextPaw = nonlocal_context.buildNonlocalContextPaw;

const PwGridMap = pw_grid_map.PwGridMap;

const fftReciprocalToComplexInPlace = fft_grid.fftReciprocalToComplexInPlace;
const fftReciprocalToComplexInPlaceMapped = fft_grid.fftReciprocalToComplexInPlaceMapped;
const fftComplexToReciprocalInPlace = fft_grid.fftComplexToReciprocalInPlace;
const fftComplexToReciprocalInPlaceMapped = fft_grid.fftComplexToReciprocalInPlaceMapped;

const profileStart = logging.profileStart;
const profileAdd = logging.profileAdd;


/// Thread-local workspace for applyHamiltonian
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

    pub fn init(alloc: std.mem.Allocator, io: std.Io, grid: Grid, n_gvecs: usize, max_m_total: usize, fft_backend: fft.FftBackend) !ApplyWorkspace {
        // Create new FFT plan
        const fft_plan: ?fft.Fft3dPlan = try fft.Fft3dPlan.initWithBackend(alloc, io, grid.nx, grid.ny, grid.nz, fft_backend);
        return initWithPlan(alloc, grid, n_gvecs, max_m_total, fft_plan);
    }

    /// Initialize with an existing FFT plan (avoids mutex contention in parallel execution)
    pub fn initWithPlan(alloc: std.mem.Allocator, grid: Grid, n_gvecs: usize, max_m_total: usize, fft_plan: ?fft.Fft3dPlan) !ApplyWorkspace {
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
        const work_coeff = if (max_m_total > 0) try alloc.alloc(math.Complex, max_m_total) else &[_]math.Complex{};
        errdefer if (max_m_total > 0) alloc.free(work_coeff);
        const work_coeff2 = if (max_m_total > 0) try alloc.alloc(math.Complex, max_m_total) else &[_]math.Complex{};
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
        self.deinitWithoutPlan(alloc);
        if (self.fft_plan) |*plan| {
            plan.deinit(alloc);
        }
    }

    /// Deinit without freeing the FFT plan (for shared plans)
    pub fn deinitWithoutPlan(self: *ApplyWorkspace, alloc: std.mem.Allocator) void {
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

    pub fn isValid(self: *const KpointApplyCache, expected_basis_len: usize) bool {
        return self.basis_len > 0 and self.basis_len == expected_basis_len and self.map != null;
    }

    pub fn deinit(self: *KpointApplyCache, _: std.mem.Allocator) void {
        if (self.nonlocal_ctx) |*ctx| ctx.deinit(self.cache_alloc);
        if (self.map) |*m| m.deinit(self.cache_alloc);
        self.* = .{};
    }
};

pub const ApplyContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    map: PwGridMap,
    gvecs: []plane_wave.GVector,
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
    // Backward compatibility: first workspace fields
    work_recip: []math.Complex,
    work_real: []math.Complex,
    work_recip_out: []math.Complex,
    work_vec: []math.Complex,
    work_phase: []math.Complex,
    work_xphase: []math.Complex,
    work_coeff: []math.Complex,
    work_coeff2: []math.Complex,

    /// Initialize with single workspace (backward compatible)
    pub fn init(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: Grid,
        gvecs: []plane_wave.GVector,
        local_r: []const f64,
        vnl: ?[]math.Complex,
        species: []hamiltonian.SpeciesEntry,
        atoms: []const hamiltonian.AtomData,
        inv_volume: f64,
        enable_nonlocal: bool,
        profile: ?*ScfProfile,
        fft_index_map: ?[]const usize,
        fft_backend: fft.FftBackend,
    ) !ApplyContext {
        return initWithWorkspaces(alloc, io, grid, gvecs, local_r, vnl, species, atoms, inv_volume, enable_nonlocal, profile, fft_index_map, fft_backend, 1);
    }

    /// Initialize with pre-created FFT plan (avoids mutex contention in parallel execution)
    pub fn initWithFftPlan(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: Grid,
        gvecs: []plane_wave.GVector,
        local_r: []const f64,
        vnl: ?[]math.Complex,
        species: []hamiltonian.SpeciesEntry,
        atoms: []const hamiltonian.AtomData,
        inv_volume: f64,
        enable_nonlocal: bool,
        profile: ?*ScfProfile,
        fft_index_map: ?[]const usize,
        existing_plan: fft.Fft3dPlan,
    ) !ApplyContext {
        var nonlocal_ctx: ?NonlocalContext = null;
        if (enable_nonlocal) {
            nonlocal_ctx = try buildNonlocalContextPub(alloc, species, gvecs);
        }
        errdefer if (nonlocal_ctx) |*ctx| ctx.deinit(alloc);

        var map = try PwGridMap.init(alloc, gvecs, grid);
        errdefer map.deinit(alloc);
        if (fft_index_map) |idx_map| {
            try map.buildFftIndices(alloc, idx_map);
        }

        const max_m_total = if (nonlocal_ctx) |ctx| ctx.max_m_total else 0;
        const workspaces = try alloc.alloc(ApplyWorkspace, 1);
        errdefer {
            for (workspaces) |*ws| {
                ws.deinit(alloc);
            }
            alloc.free(workspaces);
        }
        // Use initWithPlan to avoid creating another FFT plan
        workspaces[0] = try ApplyWorkspace.initWithPlan(alloc, grid, gvecs.len, max_m_total, existing_plan);

        const ws_in_use = try alloc.alloc(std.atomic.Value(u8), 1);
        errdefer alloc.free(ws_in_use);
        ws_in_use[0].store(0, .release);

        return .{
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
            .fft_plan = existing_plan, // Use the provided plan
            .fft_index_map = fft_index_map,
            .owns_fft_plan = false, // We don't own this plan, caller will free it
            .owns_nonlocal = true,
            .owns_map = true,
            .workspaces = workspaces,
            .num_workspaces = 1,
            .ws_in_use = ws_in_use,
            .ws_mutex = .init,
            .work_recip = workspaces[0].work_recip,
            .work_real = workspaces[0].work_real,
            .work_recip_out = workspaces[0].work_recip_out,
            .work_vec = workspaces[0].work_vec,
            .work_phase = workspaces[0].work_phase,
            .work_xphase = workspaces[0].work_xphase,
            .work_coeff = workspaces[0].work_coeff,
            .work_coeff2 = workspaces[0].work_coeff2,
        };
    }

    /// Initialize using cached NonlocalContext and PwGridMap (avoids expensive recomputation).
    /// The caller retains ownership of the cached components (nonlocal_ctx and map).
    pub fn initWithCache(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: Grid,
        gvecs: []plane_wave.GVector,
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
        workspaces[0] = try ApplyWorkspace.initWithPlan(alloc, grid, gvecs.len, max_m_total, fft_plan);

        const ws_in_use = try alloc.alloc(std.atomic.Value(u8), 1);
        errdefer alloc.free(ws_in_use);
        ws_in_use[0].store(0, .release);

        return .{
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
            .ws_mutex = .init,
            .work_recip = workspaces[0].work_recip,
            .work_real = workspaces[0].work_real,
            .work_recip_out = workspaces[0].work_recip_out,
            .work_vec = workspaces[0].work_vec,
            .work_phase = workspaces[0].work_phase,
            .work_xphase = workspaces[0].work_xphase,
            .work_coeff = workspaces[0].work_coeff,
            .work_coeff2 = workspaces[0].work_coeff2,
        };
    }

    /// Initialize with multiple workspaces for parallel execution
    pub fn initWithWorkspaces(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: Grid,
        gvecs: []plane_wave.GVector,
        local_r: []const f64,
        vnl: ?[]math.Complex,
        species: []hamiltonian.SpeciesEntry,
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
            nonlocal_ctx = try buildNonlocalContextPub(alloc, species, gvecs);
        }
        errdefer if (nonlocal_ctx) |*ctx| ctx.deinit(alloc);

        var map = try PwGridMap.init(alloc, gvecs, grid);
        errdefer map.deinit(alloc);
        if (fft_index_map) |idx_map| {
            try map.buildFftIndices(alloc, idx_map);
        }

        var plan = try fft.Fft3dPlan.initWithBackend(alloc, io, grid.nx, grid.ny, grid.nz, fft_backend);
        errdefer plan.deinit(alloc);

        const max_m_total = if (nonlocal_ctx) |ctx| ctx.max_m_total else 0;
        const ws_count = @max(num_workspaces, 1);
        const workspaces = try alloc.alloc(ApplyWorkspace, ws_count);
        errdefer {
            for (workspaces) |*ws| {
                ws.deinitWithoutPlan(alloc); // Don't free shared FFT plan (freed by errdefer on plan)
            }
            alloc.free(workspaces);
        }
        var i: usize = 0;
        while (i < ws_count) : (i += 1) {
            // Use shared FFT plan to avoid leaking per-workspace plans
            workspaces[i] = try ApplyWorkspace.initWithPlan(alloc, grid, gvecs.len, max_m_total, plan);
        }

        const ws_in_use = try alloc.alloc(std.atomic.Value(u8), ws_count);
        errdefer alloc.free(ws_in_use);
        for (ws_in_use) |*entry| {
            entry.store(0, .release);
        }

        return .{
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
            .owns_fft_plan = true, // We created this plan, so we own it
            .owns_nonlocal = true,
            .owns_map = true,
            .workspaces = workspaces,
            .num_workspaces = ws_count,
            .ws_in_use = ws_in_use,
            .ws_mutex = .init,
            .work_recip = workspaces[0].work_recip,
            .work_real = workspaces[0].work_real,
            .work_recip_out = workspaces[0].work_recip_out,
            .work_vec = workspaces[0].work_vec,
            .work_phase = workspaces[0].work_phase,
            .work_xphase = workspaces[0].work_xphase,
            .work_coeff = workspaces[0].work_coeff,
            .work_coeff2 = workspaces[0].work_coeff2,
        };
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
            ws.deinitWithoutPlan(alloc); // Don't free workspace's plan (shared with context)
        }
        if (self.workspaces.len > 0) alloc.free(self.workspaces);
        if (self.ws_in_use.len > 0) alloc.free(self.ws_in_use);
    }

    /// Acquire an available workspace index for parallel use.
    pub fn acquireWorkspace(self: *ApplyContext) ?usize {
        for (self.ws_in_use, 0..) |*entry, idx| {
            if (entry.cmpxchgStrong(0, 1, .acquire, .acquire) == null) {
                return idx;
            }
        }
        return null;
    }

    /// Release a workspace index back to the pool.
    pub fn releaseWorkspace(self: *ApplyContext, idx: usize) void {
        if (idx < self.ws_in_use.len) {
            self.ws_in_use[idx].store(0, .release);
        }
    }

    /// Get workspace by index (wraps around if index >= num_workspaces)
    pub fn getWorkspace(self: *ApplyContext, idx: usize) *ApplyWorkspace {
        return &self.workspaces[idx % self.num_workspaces];
    }
};

/// Apply Hamiltonian using pre-allocated workspace (no allocation during call)
fn applyHamiltonianWithWorkspace(ctx: *ApplyContext, ws: *ApplyWorkspace, x: []const math.Complex, y: []math.Complex) !void {
    const n = ctx.gvecs.len;
    if (x.len != n or y.len != n) return error.InvalidMatrixSize;

    const start = if (ctx.profile != null) profileStart(ctx.io) else null;

    // Kinetic energy: y = T*x
    var i: usize = 0;
    while (i < n) : (i += 1) {
        y[i] = math.complex.scale(x[i], ctx.gvecs[i].kinetic);
    }

    // Local potential: y += V_local * x
    const local_start = if (ctx.profile != null) profileStart(ctx.io) else null;
    const plan_ptr: ?*fft.Fft3dPlan = if (ws.fft_plan != null) @constCast(&ws.fft_plan.?) else null;
    try applyLocalPotentialSafe(ctx, x, ws.work_vec, ws.work_recip, ws.work_real, ws.work_recip_out, plan_ptr);
    if (ctx.profile) |p| profileAdd(ctx.io, &p.apply_local_ns, local_start);
    i = 0;
    while (i < n) : (i += 1) {
        y[i] = math.complex.add(y[i], ws.work_vec[i]);
    }

    // Nonlocal potential: y += V_nl * x
    if (ctx.nonlocal_ctx != null) {
        const nonlocal_start = if (ctx.profile != null) profileStart(ctx.io) else null;
        try applyNonlocalPotentialSafe(ctx, x, ws.work_vec, ws.work_phase, ws.work_xphase, ws.work_coeff, ws.work_coeff2);
        if (ctx.profile) |p| profileAdd(ctx.io, &p.apply_nonlocal_ns, nonlocal_start);
        i = 0;
        while (i < n) : (i += 1) {
            y[i] = math.complex.add(y[i], ws.work_vec[i]);
        }
    } else if (ctx.vnl) |mat| {
        applyDenseMatrix(n, mat, x, ws.work_vec);
        i = 0;
        while (i < n) : (i += 1) {
            y[i] = math.complex.add(y[i], ws.work_vec[i]);
        }
    }

    if (ctx.profile) |p| {
        profileAdd(ctx.io, &p.apply_h_ns, start);
        p.apply_h_calls += 1;
    }
}

pub fn applyHamiltonian(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
    const ctx: *ApplyContext = @ptrCast(@alignCast(ctx_ptr));

    // Try to acquire a workspace from the pool (thread-safe)
    if (ctx.acquireWorkspace()) |ws_idx| {
        defer ctx.releaseWorkspace(ws_idx);
        try applyHamiltonianWithWorkspace(ctx, &ctx.workspaces[ws_idx], x, y);
    } else {
        // All workspaces in use - use first one with lock
        ctx.ws_mutex.lockUncancelable(ctx.io);
        defer ctx.ws_mutex.unlock(ctx.io);
        try applyHamiltonianWithWorkspace(ctx, &ctx.workspaces[0], x, y);
    }
}

pub fn checkHamiltonianApply(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    gvecs: []plane_wave.GVector,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    potential: hamiltonian.PotentialGrid,
    local_r: []const f64,
    nonlocal_enabled: bool,
    fft_index_map: ?[]const usize,
) !void {
    var ctx = try ApplyContext.init(
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

    try applyHamiltonian(&ctx, x, y_apply);

    const local_cfg = local_potential.LocalPotentialConfig.init(.short_range, 0.0);
    const h = try hamiltonian.buildHamiltonian(alloc, gvecs, species, atoms, inv_volume, local_cfg, potential);
    defer alloc.free(h);
    applyDenseMatrix(n, h, x, y_dense);

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

    var buffer: [160]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    const rel = if (max_abs > 0.0) max_diff / max_abs else 0.0;
    try out.print("scf: apply_check max_abs={d:.6} max_diff={d:.6} rel={d:.6}\n", .{ max_abs, max_diff, rel });
    try out.flush();
}

fn applyDenseMatrix(n: usize, mat: []const math.Complex, x: []const math.Complex, out: []math.Complex) void {
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

fn applyLocalPotential(ctx: *ApplyContext, x: []const math.Complex, out: []math.Complex) !void {
    ctx.map.scatter(x, ctx.work_recip);
    const plan_ptr = if (ctx.fft_plan) |*p| p else null;
    if (ctx.fft_index_map) |map| {
        try fftReciprocalToComplexInPlaceMapped(ctx.alloc, ctx.grid, map, ctx.work_recip, ctx.work_real, plan_ptr);
    } else {
        try fftReciprocalToComplexInPlace(ctx.alloc, ctx.grid, ctx.work_recip, ctx.work_real, plan_ptr);
    }
    for (ctx.work_real, 0..) |*v, i| {
        v.* = math.complex.scale(v.*, ctx.local_r[i]);
    }
    if (ctx.fft_index_map) |map| {
        try fftComplexToReciprocalInPlaceMapped(ctx.alloc, ctx.grid, map, ctx.work_real, ctx.work_recip_out, plan_ptr);
    } else {
        try fftComplexToReciprocalInPlace(ctx.alloc, ctx.grid, ctx.work_real, ctx.work_recip_out, plan_ptr);
    }
    ctx.map.gather(ctx.work_recip_out, out);
}

pub fn applyNonlocalPotential(ctx: *ApplyContext, x: []const math.Complex, out: []math.Complex) !void {
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
        const coeff = ctx.work_coeff[0..entry.m_total];
        const coeff2 = ctx.work_coeff2[0..entry.m_total];

        var atom_of_species: usize = 0;
        for (ctx.atoms) |atom| {
            if (atom.species_index != entry.species_index) continue;
            const coeffs = if (entry.dij_per_atom) |dpa| dpa[atom_of_species] else entry.coeffs;
            const dij_m = if (entry.dij_m_per_atom) |dpa| dpa[atom_of_species] else null;
            atom_of_species += 1;

            var g: usize = 0;
            while (g < n) : (g += 1) {
                const phase = math.complex.expi(math.Vec3.dot(ctx.gvecs[g].cart, atom.position));
                ctx.work_phase[g] = phase;
                ctx.work_xphase[g] = math.complex.mul(x[g], phase);
            }

            // Step 1: Compute projector overlaps <p_{β,m}|ψ>
            var b: usize = 0;
            while (b < entry.beta_count) : (b += 1) {
                const offset = entry.m_offsets[b];
                const m_count = entry.m_counts[b];
                var m_idx: usize = 0;
                while (m_idx < m_count) : (m_idx += 1) {
                    const phi = entry.phi[(offset + m_idx) * g_count .. (offset + m_idx + 1) * g_count];
                    var sum = math.complex.init(0.0, 0.0);
                    g = 0;
                    while (g < n) : (g += 1) {
                        sum = math.complex.add(sum, math.complex.scale(ctx.work_xphase[g], phi[g]));
                    }
                    coeff[offset + m_idx] = sum;
                }
            }

            // Step 2: Apply D_ij
            if (dij_m) |dm| {
                // m-resolved D: coeff2[im] = Σ_jm D[im,jm] × coeff[jm]
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
                // Radial D_ij: same-m coupling only
                b = 0;
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

            g = 0;
            while (g < n) : (g += 1) {
                var accum = math.complex.init(0.0, 0.0);
                b = 0;
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
                const phase_conj = math.complex.conj(ctx.work_phase[g]);
                const add = math.complex.mul(phase_conj, accum);
                out[g] = math.complex.add(out[g], math.complex.scale(add, ctx.inv_volume));
            }
        }
    }
}

/// Thread-safe version of applyLocalPotential with explicit work buffers
/// Uses provided FFT plan (from workspace) for efficiency
fn applyLocalPotentialSafe(
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

        // Scatter PW -> FFT order with iFFT scaling
        const t_scatter = if (ctx.profile != null) profileStart(ctx.io) else null;
        ctx.map.scatterFft(x, work_real, total);
        if (ctx.profile) |p| profileAdd(ctx.io, &p.local_scatter_ns, t_scatter);

        // iFFT in-place
        const t_ifft = if (ctx.profile != null) profileStart(ctx.io) else null;
        if (plan) |p| {
            try fft.fft3dInverseInPlacePlan(p, work_real);
        } else {
            try fft.fft3dInverseInPlace(ctx.alloc, work_real, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz);
        }
        if (ctx.profile) |p| profileAdd(ctx.io, &p.local_ifft_ns, t_ifft);

        // V(r) * psi(r)
        const t_vmul = if (ctx.profile != null) profileStart(ctx.io) else null;
        for (work_real, 0..) |*v, i| {
            v.* = math.complex.scale(v.*, ctx.local_r[i]);
        }
        if (ctx.profile) |p| profileAdd(ctx.io, &p.local_vmul_ns, t_vmul);

        // FFT in-place
        const t_fft = if (ctx.profile != null) profileStart(ctx.io) else null;
        if (plan) |p| {
            try fft.fft3dForwardInPlacePlan(p, work_real);
        } else {
            try fft.fft3dForwardInPlace(ctx.alloc, work_real, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz);
        }
        if (ctx.profile) |p| profileAdd(ctx.io, &p.local_fft_ns, t_fft);

        // Gather FFT order -> PW with FFT scaling
        const t_gather = if (ctx.profile != null) profileStart(ctx.io) else null;
        ctx.map.gatherFft(work_real, out, inv_scale);
        if (ctx.profile) |p| profileAdd(ctx.io, &p.local_gather_ns, t_gather);
    } else {
        // Original path with full-grid remap
        ctx.map.scatter(x, work_recip);
        if (ctx.fft_index_map) |map| {
            try fftReciprocalToComplexInPlaceMapped(ctx.alloc, ctx.grid, map, work_recip, work_real, plan);
        } else {
            try fftReciprocalToComplexInPlace(ctx.alloc, ctx.grid, work_recip, work_real, plan);
        }
        for (work_real, 0..) |*v, i| {
            v.* = math.complex.scale(v.*, ctx.local_r[i]);
        }
        if (ctx.fft_index_map) |map| {
            try fftComplexToReciprocalInPlaceMapped(ctx.alloc, ctx.grid, map, work_real, work_recip_out, plan);
        } else {
            try fftComplexToReciprocalInPlace(ctx.alloc, ctx.grid, work_real, work_recip_out, plan);
        }
        ctx.map.gather(work_recip_out, out);
    }
}

/// Thread-safe version of applyNonlocalPotential with explicit work buffers
fn applyNonlocalPotentialSafe(
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

            var g: usize = 0;
            while (g < n) : (g += 1) {
                const phase = math.complex.expi(math.Vec3.dot(ctx.gvecs[g].cart, atom.position));
                work_phase[g] = phase;
                work_xphase[g] = math.complex.mul(x[g], phase);
            }

            var b: usize = 0;
            while (b < entry.beta_count) : (b += 1) {
                const offset = entry.m_offsets[b];
                const m_count = entry.m_counts[b];
                var m_idx: usize = 0;
                while (m_idx < m_count) : (m_idx += 1) {
                    const phi = entry.phi[(offset + m_idx) * g_count .. (offset + m_idx + 1) * g_count];
                    var sum = math.complex.init(0.0, 0.0);
                    g = 0;
                    while (g < n) : (g += 1) {
                        sum = math.complex.add(sum, math.complex.scale(work_xphase[g], phi[g]));
                    }
                    coeff[offset + m_idx] = sum;
                }
            }

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
                b = 0;
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

            g = 0;
            while (g < n) : (g += 1) {
                var accum = math.complex.init(0.0, 0.0);
                b = 0;
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
                out[g] = math.complex.add(out[g], math.complex.scale(add, ctx.inv_volume));
            }
        }
    }
}

/// Apply PAW overlap operator: S|ψ> = |ψ> + Σ_a Σ_{ij} q_ij |p_i^a><p_j^a|ψ>
/// This is the S operator for the generalized eigenvalue problem H|ψ> = ε S|ψ>.
/// Uses the same projector infrastructure as the nonlocal potential but with q_ij coefficients.
pub fn applyOverlap(ctx_ptr: *anyopaque, x: []const math.Complex, y: []math.Complex) !void {
    const ctx: *ApplyContext = @ptrCast(@alignCast(ctx_ptr));

    if (ctx.acquireWorkspace()) |ws_idx| {
        defer ctx.releaseWorkspace(ws_idx);
        const ws = &ctx.workspaces[ws_idx];
        try applyOverlapSafe(ctx, x, y, ws.work_phase, ws.work_xphase, ws.work_coeff, ws.work_coeff2);
    } else {
        ctx.ws_mutex.lockUncancelable(ctx.io);
        defer ctx.ws_mutex.unlock(ctx.io);
        const ws = &ctx.workspaces[0];
        try applyOverlapSafe(ctx, x, y, ws.work_phase, ws.work_xphase, ws.work_coeff, ws.work_coeff2);
    }
}

/// Thread-safe overlap operator application with explicit work buffers.
/// y = S·x = x + Σ_a Σ_{ij} q_ij <p_j|x> |p_i>
fn applyOverlapSafe(
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

            // Compute phase factors and x*phase
            var g: usize = 0;
            while (g < n) : (g += 1) {
                const phase = math.complex.expi(math.Vec3.dot(ctx.gvecs[g].cart, atom.position));
                work_phase[g] = phase;
                work_xphase[g] = math.complex.mul(x[g], phase);
            }

            // Step 1: Project onto beta functions: c_i = <p_i|ψ>
            var b: usize = 0;
            while (b < entry.beta_count) : (b += 1) {
                const offset = entry.m_offsets[b];
                const m_count = entry.m_counts[b];
                var m_idx: usize = 0;
                while (m_idx < m_count) : (m_idx += 1) {
                    const phi = entry.phi[(offset + m_idx) * g_count .. (offset + m_idx + 1) * g_count];
                    var sum = math.complex.init(0.0, 0.0);
                    g = 0;
                    while (g < n) : (g += 1) {
                        sum = math.complex.add(sum, math.complex.scale(work_xphase[g], phi[g]));
                    }
                    coeff[offset + m_idx] = sum;
                }
            }

            // Step 2: Apply q_ij: c'_i = Σ_j q_ij c_j
            b = 0;
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
                        const q = qij[b * entry.beta_count + j];
                        if (q == 0.0) continue;
                        const c = coeff[entry.m_offsets[j] + m_idx];
                        sum = math.complex.add(sum, math.complex.scale(c, q));
                    }
                    coeff2[offset + m_idx] = sum;
                }
            }

            // Step 3: Back-project: out += Σ_i |p_i> c'_i
            g = 0;
            while (g < n) : (g += 1) {
                var accum = math.complex.init(0.0, 0.0);
                b = 0;
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
                out[g] = math.complex.add(out[g], math.complex.scale(add, ctx.inv_volume));
            }
        }
    }
}

/// Batched nonlocal potential application using BLAS-3 (dgemm).
/// Processes ncols input vectors simultaneously for better cache efficiency.
/// x_batch and out_batch are column-major: x_batch[col * n_pw + g].
/// out_batch is zeroed and filled (not accumulated).
fn applyNonlocalBatched(
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

    // Allocate work buffers: xphase_re/im (reused for result), coeffs, coeffs2
    const buf_gn = try alloc.alloc(f64, n_pw * ncols * 2);
    defer alloc.free(buf_gn);
    const xphase_re = buf_gn[0 .. n_pw * ncols];
    const xphase_im = buf_gn[n_pw * ncols .. 2 * n_pw * ncols];

    const buf_mn = try alloc.alloc(f64, max_m * ncols * 4);
    defer alloc.free(buf_mn);
    const coeffs_re = buf_mn[0 .. max_m * ncols];
    const coeffs_im = buf_mn[max_m * ncols .. 2 * max_m * ncols];
    const coeffs2_re = buf_mn[2 * max_m * ncols .. 3 * max_m * ncols];
    const coeffs2_im = buf_mn[3 * max_m * ncols .. 4 * max_m * ncols];

    for (nl.species) |entry| {
        const g_count = entry.g_count;
        if (g_count != n_pw) return error.InvalidMatrixSize;
        if (entry.m_total == 0) continue;
        const m_total = entry.m_total;
        const phi = entry.phi[0 .. m_total * g_count];

        var atom_of_species: usize = 0;
        for (ctx.atoms) |atom| {
            if (atom.species_index != entry.species_index) continue;
            const dij_coeffs = if (entry.dij_per_atom) |dpa| dpa[atom_of_species] else entry.coeffs;
            const dij_m = if (entry.dij_m_per_atom) |dpa| dpa[atom_of_species] else null;
            atom_of_species += 1;

            // Step 1: Phase factors exp(i G·τ)
            for (0..n_pw) |g| {
                work_phase[g] = math.complex.expi(math.Vec3.dot(ctx.gvecs[g].cart, atom.position));
            }

            // Step 2: xphase[g,b] = x_batch[b,g] * phase[g]
            // Layout: row-major [g_count × ncols]
            for (0..n_pw) |g| {
                const ph = work_phase[g];
                const row = g * ncols;
                for (0..ncols) |col| {
                    const xv = x_batch[col * n_pw + g];
                    xphase_re[row + col] = xv.r * ph.r - xv.i * ph.i;
                    xphase_im[row + col] = xv.r * ph.i + xv.i * ph.r;
                }
            }

            // Step 3: coeffs = phi × xphase (projection)
            // phi: [m_total × g_count] row-major, xphase: [g_count × ncols] row-major
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

            // Step 4: Apply D_ij
            if (dij_m) |dm| {
                // m-resolved D: coeffs2 = D_m × coeffs via BLAS
                blas.dgemm(
                    .no_trans,
                    .no_trans,
                    m_total,
                    ncols,
                    m_total,
                    1.0,
                    dm,
                    m_total,
                    coeffs_re[0 .. m_total * ncols],
                    ncols,
                    0.0,
                    coeffs2_re[0 .. m_total * ncols],
                    ncols,
                );
                blas.dgemm(
                    .no_trans,
                    .no_trans,
                    m_total,
                    ncols,
                    m_total,
                    1.0,
                    dm,
                    m_total,
                    coeffs_im[0 .. m_total * ncols],
                    ncols,
                    0.0,
                    coeffs2_im[0 .. m_total * ncols],
                    ncols,
                );
            } else {
                // Radial D_ij: same-m coupling only
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

            // Step 5: Back-project: result = phi^T × coeffs2
            // Reuse xphase buffers for result
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

            // Step 6: out += (1/Ω) * conj(phase) * result
            const inv_vol = ctx.inv_volume;
            for (0..n_pw) |g| {
                const ph_conj = math.complex.conj(work_phase[g]);
                const row = g * ncols;
                for (0..ncols) |col| {
                    const re = xphase_re[row + col];
                    const im = xphase_im[row + col];
                    const out_r = (ph_conj.r * re - ph_conj.i * im) * inv_vol;
                    const out_i = (ph_conj.r * im + ph_conj.i * re) * inv_vol;
                    const idx = col * n_pw + g;
                    out_batch[idx] = math.complex.add(out_batch[idx], math.complex.init(out_r, out_i));
                }
            }
        }
    }
}

/// Batched Hamiltonian application: y_batch = H * x_batch for ncols vectors.
/// x_batch and y_batch are column-major: [n_pw × ncols].
/// Uses BLAS-3 batched nonlocal for better cache efficiency.
pub fn applyHamiltonianBatched(
    ctx_ptr: *anyopaque,
    x_batch: []const math.Complex,
    y_batch: []math.Complex,
    n_pw: usize,
    ncols: usize,
) !void {
    const ctx: *ApplyContext = @ptrCast(@alignCast(ctx_ptr));
    if (n_pw != ctx.gvecs.len) return error.InvalidMatrixSize;

    const h_start = if (ctx.profile != null) profileStart(ctx.io) else null;

    // Kinetic + local: per-vector (FFT can't easily be batched)
    const local_start = if (ctx.profile != null) profileStart(ctx.io) else null;
    for (0..ncols) |col| {
        const x = x_batch[col * n_pw .. (col + 1) * n_pw];
        const y = y_batch[col * n_pw .. (col + 1) * n_pw];

        // Kinetic energy: y = T*x
        for (0..n_pw) |g| {
            y[g] = math.complex.scale(x[g], ctx.gvecs[g].kinetic);
        }

        // Local potential: y += V_local * x
        if (ctx.acquireWorkspace()) |ws_idx| {
            defer ctx.releaseWorkspace(ws_idx);
            const ws = &ctx.workspaces[ws_idx];
            const plan_ptr: ?*fft.Fft3dPlan = if (ws.fft_plan != null) @constCast(&ws.fft_plan.?) else null;
            try applyLocalPotentialSafe(ctx, x, ws.work_vec, ws.work_recip, ws.work_real, ws.work_recip_out, plan_ptr);
            for (0..n_pw) |g| {
                y[g] = math.complex.add(y[g], ws.work_vec[g]);
            }
        } else {
            ctx.ws_mutex.lockUncancelable(ctx.io);
            defer ctx.ws_mutex.unlock(ctx.io);
            const ws = &ctx.workspaces[0];
            const plan_ptr: ?*fft.Fft3dPlan = if (ws.fft_plan != null) @constCast(&ws.fft_plan.?) else null;
            try applyLocalPotentialSafe(ctx, x, ws.work_vec, ws.work_recip, ws.work_real, ws.work_recip_out, plan_ptr);
            for (0..n_pw) |g| {
                y[g] = math.complex.add(y[g], ws.work_vec[g]);
            }
        }
    }
    if (ctx.profile) |p| profileAdd(ctx.io, &p.apply_local_ns, local_start);

    // Nonlocal potential: batched BLAS-3
    if (ctx.nonlocal_ctx != null) {
        const nonlocal_start = if (ctx.profile != null) profileStart(ctx.io) else null;
        const nl_out = try ctx.alloc.alloc(math.Complex, n_pw * ncols);
        defer ctx.alloc.free(nl_out);
        const work_phase = try ctx.alloc.alloc(math.Complex, n_pw);
        defer ctx.alloc.free(work_phase);

        try applyNonlocalBatched(ctx, ctx.alloc, x_batch, nl_out, n_pw, ncols, work_phase);

        for (0..n_pw * ncols) |i| {
            y_batch[i] = math.complex.add(y_batch[i], nl_out[i]);
        }
        if (ctx.profile) |p| profileAdd(ctx.io, &p.apply_nonlocal_ns, nonlocal_start);
    } else if (ctx.vnl) |mat| {
        for (0..ncols) |col| {
            const x = x_batch[col * n_pw .. (col + 1) * n_pw];
            const ws_vec = try ctx.alloc.alloc(math.Complex, n_pw);
            defer ctx.alloc.free(ws_vec);
            applyDenseMatrix(n_pw, mat, x, ws_vec);
            const y = y_batch[col * n_pw .. (col + 1) * n_pw];
            for (0..n_pw) |g| {
                y[g] = math.complex.add(y[g], ws_vec[g]);
            }
        }
    }

    if (ctx.profile) |p| {
        profileAdd(ctx.io, &p.apply_h_ns, h_start);
        p.apply_h_calls += 1;
    }
}

/// Compute projector overlaps and accumulate m-resolved PAW occupation matrix rhoij.
/// Must be called for each occupied band at each k-point.
/// Uses the NonlocalContext's phi arrays to compute <p_{β,m}|ψ> projections.
///
/// The rhoij matrix is m-resolved: ρ_{(i,m_i),(j,m_j)} stores individual
/// projector-m products (no m-summation). This enables multi-L on-site density.
pub fn accumulatePawRhoIJ(
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

            // Compute phase factors
            var g: usize = 0;
            while (g < n) : (g += 1) {
                const phase = math.complex.expi(math.Vec3.dot(gvecs[g].cart, atom.position));
                work_phase[g] = phase;
                work_xphase[g] = math.complex.mul(psi[g], phase);
            }

            // Step 1: Compute <p_{beta,m}|ψ> for each (beta, m)
            var b: usize = 0;
            while (b < entry.beta_count) : (b += 1) {
                const offset = entry.m_offsets[b];
                const m_count = entry.m_counts[b];
                var m_idx: usize = 0;
                while (m_idx < m_count) : (m_idx += 1) {
                    const phi = entry.phi[(offset + m_idx) * g_count .. (offset + m_idx + 1) * g_count];
                    var sum = math.complex.init(0.0, 0.0);
                    g = 0;
                    while (g < n) : (g += 1) {
                        sum = math.complex.add(sum, math.complex.scale(work_xphase[g], phi[g]));
                    }
                    work_coeff[offset + m_idx] = sum;
                }
            }

            // Step 2: Accumulate m-resolved ρ_{(i,m_i),(j,m_j)}
            // Store Re[<p_{i,m_i}|ψ> conj(<p_{j,m_j}|ψ>)] for ALL (m_i, m_j) pairs
            const mt = rhoij.m_total_per_atom[atom_idx];
            var im: usize = 0;
            while (im < entry.m_total) : (im += 1) {
                const ci = work_coeff[im];
                var jm: usize = 0;
                while (jm < entry.m_total) : (jm += 1) {
                    const cj = work_coeff[jm];
                    rhoij.values[atom_idx][im * mt + jm] += effective_weight * (ci.r * cj.r + ci.i * cj.i);
                }
            }
        }
    }
}
