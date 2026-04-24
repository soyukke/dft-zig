const std = @import("std");
const apply = @import("apply.zig");
const common_mod = @import("common.zig");
const config = @import("../config/config.zig");
const density_mod = @import("density.zig");
const final_wavefunction = @import("final_wavefunction.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoints_mod = @import("kpoint_parallel.zig");
const math = @import("../math/math.zig");
const model_mod = @import("../dft/model.zig");
const potential_mod = @import("potential.zig");

const KpointCache = kpoints_mod.KpointCache;
const ScfCommon = common_mod.ScfCommon;
const WavefunctionData = final_wavefunction.WavefunctionData;

/// Bag of SCF loop profiling accumulators.
pub const ScfLoopProf = struct {
    compute_density_ns: u64 = 0,
    build_potential_ns: u64 = 0,
    residual_ns: u64 = 0,
    mixing_ns: u64 = 0,
    build_local_r_ns: u64 = 0,
    build_fft_map_ns: u64 = 0,
};

pub const ScfRunCaches = struct {
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,

    pub fn deinit(self: *const ScfRunCaches, alloc: std.mem.Allocator) void {
        for (self.kpoint_cache) |*cache| cache.deinit();
        alloc.free(self.kpoint_cache);
        for (self.apply_caches) |*ac| ac.deinit(alloc);
        alloc.free(self.apply_caches);
    }
};

pub const ScfRunState = struct {
    rho: []f64,
    potential: hamiltonian.PotentialGrid,
    vxc_r: ?[]f64 = null,
    vresid_last: ?hamiltonian.PotentialGrid = null,
    last_band_energy: f64 = 0.0,
    last_nonlocal_energy: f64 = 0.0,
    last_entropy_energy: f64 = 0.0,
    last_fermi_level: f64 = std.math.nan(f64),
    last_potential_residual: f64 = 0.0,
    iterations: usize = 0,
    converged: bool = false,
    wavefunctions: ?WavefunctionData = null,

    pub fn deinit(self: *ScfRunState, alloc: std.mem.Allocator) void {
        self.potential.deinit(alloc);
        alloc.free(self.rho);
        if (self.vxc_r) |v| alloc.free(v);
        if (self.vresid_last) |*vresid| vresid.deinit(alloc);
        if (self.wavefunctions) |*wf| wf.deinit(alloc);
    }
};

pub const ScfRunContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    common: *ScfCommon,
    paw_ecutrho: ?f64,
};

pub const ScfIterationDensity = struct {
    density_result: density_mod.DensityResult,
    rho_for_potential: []const f64,

    pub fn deinit(self: *const ScfIterationDensity, alloc: std.mem.Allocator, is_paw: bool) void {
        alloc.free(self.density_result.rho);
        if (is_paw) alloc.free(self.rho_for_potential);
    }
};

pub const ScfIterationPotential = struct {
    potential_out: hamiltonian.PotentialGrid,
    keep: bool = false,

    pub fn deinit(self: *const ScfIterationPotential, alloc: std.mem.Allocator) void {
        if (!self.keep) {
            var potential_out = self.potential_out;
            potential_out.deinit(alloc);
        }
    }
};

/// Initialise rho with user-provided density (if present and fits) or flat total_electrons/volume.
pub fn init_rho_from_user_or_flat(
    rho: []f64,
    initial_density: ?[]const f64,
    electron_count: f64,
    grid_volume: f64,
) !void {
    std.debug.assert(rho.len > 0);
    std.debug.assert(std.math.isFinite(electron_count));
    std.debug.assert(std.math.isFinite(grid_volume));
    std.debug.assert(grid_volume > 0.0);
    if (initial_density) |init_rho| {
        if (init_rho.len != rho.len) return error.InvalidInitialDensitySize;
        @memcpy(rho, init_rho);
        return;
    }
    const rho0 = electron_count / grid_volume;
    @memset(rho, rho0);
}

/// Warm-start the per-kpoint cache from a previously converged SCF run, if provided.
pub fn warm_start_kpoint_cache(
    kpoint_cache: []KpointCache,
    initial_kpoint_cache: ?[]const KpointCache,
) !void {
    if (initial_kpoint_cache) |init_cache| {
        if (init_cache.len != kpoint_cache.len) return error.InvalidInitialKpointCacheSize;
        for (0..kpoint_cache.len) |k| {
            if (init_cache[k].vectors.len > 0) {
                try kpoint_cache[k].store(
                    init_cache[k].n,
                    init_cache[k].nbands,
                    init_cache[k].vectors,
                );
            }
        }
    }
}

/// Either return the caller-supplied apply caches or allocate + default-init a fresh set.
fn get_or_alloc_apply_caches(
    alloc: std.mem.Allocator,
    initial: ?[]apply.KpointApplyCache,
    n: usize,
) ![]apply.KpointApplyCache {
    if (initial) |caches| return caches;
    const caches = try alloc.alloc(apply.KpointApplyCache, n);
    for (caches) |*ac| ac.* = .{};
    return caches;
}

pub fn init_scf_run_caches(
    alloc: std.mem.Allocator,
    kpoint_count: usize,
    initial_kpoint_cache: ?[]const KpointCache,
    initial_apply_caches: ?[]apply.KpointApplyCache,
) !ScfRunCaches {
    const kpoint_cache = try alloc.alloc(KpointCache, kpoint_count);
    errdefer alloc.free(kpoint_cache);
    for (kpoint_cache) |*cache| cache.* = .{};
    try warm_start_kpoint_cache(kpoint_cache, initial_kpoint_cache);
    return .{
        .kpoint_cache = kpoint_cache,
        .apply_caches = try get_or_alloc_apply_caches(alloc, initial_apply_caches, kpoint_count),
    };
}

pub fn init_scf_run_state(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    common: *const ScfCommon,
    initial_density: ?[]const f64,
    paw_ecutrho: ?f64,
) !ScfRunState {
    const rho = try alloc.alloc(f64, common.grid.count());
    errdefer alloc.free(rho);
    try init_rho_from_user_or_flat(rho, initial_density, common.total_electrons, common.grid.volume);

    var potential = try potential_mod.build_potential_grid(
        alloc,
        common.grid,
        rho,
        common.rho_core,
        cfg.scf.use_rfft,
        cfg.scf.xc,
        null,
        common.coulomb_r_cut,
        paw_ecutrho,
    );
    errdefer potential.deinit(alloc);
    return .{ .rho = rho, .potential = potential };
}

/// Compute and store potential residual; updates vresid_last to a freshly-allocated grid.
pub fn record_potential_residual(
    alloc: std.mem.Allocator,
    grid: grid_mod.Grid,
    potential: hamiltonian.PotentialGrid,
    potential_out: hamiltonian.PotentialGrid,
    vresid_last: *?hamiltonian.PotentialGrid,
) !f64 {
    const nvals = potential.values.len;
    var residual_values = try alloc.alloc(math.Complex, nvals);
    errdefer alloc.free(residual_values);

    var sum_sq: f64 = 0.0;
    for (0..nvals) |idx| {
        const diff = math.complex.sub(potential_out.values[idx], potential.values[idx]);
        residual_values[idx] = diff;
        sum_sq += diff.r * diff.r + diff.i * diff.i;
    }
    const residual_rms = if (nvals > 0)
        std.math.sqrt(sum_sq / @as(f64, @floatFromInt(nvals)))
    else
        0.0;
    if (vresid_last.*) |*old| old.deinit(alloc);
    vresid_last.* = hamiltonian.PotentialGrid{
        .nx = grid.nx,
        .ny = grid.ny,
        .nz = grid.nz,
        .min_h = grid.min_h,
        .min_k = grid.min_k,
        .min_l = grid.min_l,
        .values = residual_values,
    };
    return residual_rms;
}

pub fn make_run_context(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    model: *const model_mod.Model,
    common: *ScfCommon,
    paw_ecutrho: ?f64,
) ScfRunContext {
    return .{
        .alloc = alloc,
        .io = io,
        .cfg = cfg,
        .species = model.species,
        .atoms = model.atoms,
        .recip = model.recip,
        .volume_bohr = model.volume_bohr,
        .common = common,
        .paw_ecutrho = paw_ecutrho,
    };
}

test "initial density length mismatch is rejected" {
    var rho = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    const init = [_]f64{ 1.0, 2.0 };

    try std.testing.expectError(
        error.InvalidInitialDensitySize,
        init_rho_from_user_or_flat(&rho, &init, 4.0, 2.0),
    );
}

test "initial kpoint cache count mismatch is rejected" {
    var caches = [_]KpointCache{ .{}, .{} };
    const initial = [_]KpointCache{.{}};

    try std.testing.expectError(
        error.InvalidInitialKpointCacheSize,
        warm_start_kpoint_cache(&caches, &initial),
    );
}
