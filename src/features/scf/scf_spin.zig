const std = @import("std");
const apply = @import("apply.zig");
const config = @import("../config/config.zig");
const energy_mod = @import("energy.zig");
const fft = @import("../fft/fft.zig");
const fft_grid = @import("fft_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoints_mod = @import("kpoint_parallel.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const mixing = @import("mixing.zig");
const paw_mod = @import("../paw/paw.zig");
const paw_scf = @import("paw_scf.zig");
const potential_mod = @import("potential.zig");
const xc_fields_mod = @import("xc_fields.zig");
const scf_mod = @import("scf.zig");
const symmetry = @import("../symmetry/symmetry.zig");

const Grid = scf_mod.Grid;
const KpointCache = kpoints_mod.KpointCache;
const KpointEigenData = kpoints_mod.KpointEigenData;
const ScfResult = scf_mod.ScfResult;
const WavefunctionData = scf_mod.WavefunctionData;
const EnergyTerms = energy_mod.EnergyTerms;

const compute_kpoint_eigen_data = kpoints_mod.compute_kpoint_eigen_data;
const find_fermi_level_spin = kpoints_mod.find_fermi_level_spin;
const accumulate_kpoint_density_smearing_spin = kpoints_mod.accumulate_kpoint_density_smearing_spin;
const build_fft_index_map = fft_grid.build_fft_index_map;
const mix_density = mixing.mix_density;
const mix_density_kerker = mixing.mix_density_kerker;
const log_progress = logging.log_progress;
const log_iter_start = logging.log_iter_start;
const log_spin_init = logging.log_spin_init;
const log_spin_magnetization = logging.log_spin_magnetization;
const log_spin_energy_summary = logging.log_spin_energy_summary;

const KPoint = symmetry.KPoint;

/// Result from solve_kpoints_for_spin: eigendata and count.
const SpinEigenResult = struct {
    eigen_data: []KpointEigenData,
    filled: usize,
};

const SpinKpointSolveConfig = struct {
    nocc: usize,
    has_qij: bool,
    use_iterative: bool,
    nonlocal_enabled: bool,
    local_r: ?[]f64,
    fft_index_map: []usize,
    iter_max_iter: usize,
    iter_tol: f64,

    fn deinit(self: *const SpinKpointSolveConfig, alloc: std.mem.Allocator) void {
        if (self.local_r) |values| alloc.free(values);
        alloc.free(self.fft_index_map);
    }
};

const SpinContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    volume_bohr: f64,
    common: *scf_mod.ScfCommon,
};

fn init_spin_kpoint_solve_config(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    common: *scf_mod.ScfCommon,
    potential: hamiltonian.PotentialGrid,
    scf_iter: usize,
) !SpinKpointSolveConfig {
    const nocc_base = @as(usize, @intFromFloat(std.math.ceil(common.total_electrons / 2.0)));
    const has_qij = scf_mod.has_qij(common.species) and !scf_mod.has_paw(common.species);
    const use_iterative = (cfg.scf.solver == .iterative or
        cfg.scf.solver == .cg or
        cfg.scf.solver == .auto) and !has_qij;

    var local_r: ?[]f64 = null;
    if (use_iterative) {
        local_r = try potential_mod.build_local_potential_real(
            alloc,
            common.grid,
            common.ionic,
            potential,
        );
    }
    errdefer if (local_r) |values| alloc.free(values);

    const fft_index_map = try build_fft_index_map(alloc, common.grid);
    errdefer alloc.free(fft_index_map);

    var iter_max_iter = cfg.scf.iterative_max_iter;
    var iter_tol = cfg.scf.iterative_tol;
    if (cfg.scf.iterative_warmup_steps > 0 and scf_iter < cfg.scf.iterative_warmup_steps) {
        iter_max_iter = cfg.scf.iterative_warmup_max_iter;
        iter_tol = cfg.scf.iterative_warmup_tol;
    }

    return .{
        .nocc = nocc_base + @max(4, nocc_base / 5),
        .has_qij = has_qij,
        .use_iterative = use_iterative,
        .nonlocal_enabled = cfg.scf.enable_nonlocal and scf_mod.has_nonlocal(common.species),
        .local_r = local_r,
        .fft_index_map = fft_index_map,
        .iter_max_iter = iter_max_iter,
        .iter_tol = iter_tol,
    };
}

fn compute_spin_channel_eigen_data(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    common: *scf_mod.ScfCommon,
    potential: hamiltonian.PotentialGrid,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    shared_fft_plan: fft.Fft3dPlan,
    solve_cfg: *const SpinKpointSolveConfig,
) !SpinEigenResult {
    const eigen_data = try alloc.alloc(KpointEigenData, common.kpoints.len);
    var filled: usize = 0;
    errdefer {
        var ii: usize = 0;
        while (ii < filled) : (ii += 1) {
            eigen_data[ii].deinit(alloc);
        }
        alloc.free(eigen_data);
    }

    for (common.kpoints, 0..) |kp, kidx| {
        const ac_ptr: ?*apply.KpointApplyCache = &apply_caches[kidx];
        eigen_data[kidx] = try compute_kpoint_eigen_data(
            alloc,
            io,
            cfg,
            common.grid,
            kp,
            common.species,
            common.atoms,
            common.recip,
            common.volume_bohr,
            common.local_cfg,
            potential,
            solve_cfg.local_r,
            solve_cfg.nocc,
            solve_cfg.use_iterative,
            solve_cfg.has_qij,
            solve_cfg.nonlocal_enabled,
            solve_cfg.fft_index_map,
            solve_cfg.iter_max_iter,
            solve_cfg.iter_tol,
            cfg.scf.iterative_reuse_vectors,
            &kpoint_cache[kidx],
            null,
            shared_fft_plan,
            ac_ptr,
            common.radial_tables,
            common.paw_tabs,
        );
        filled += 1;
    }
    return .{ .eigen_data = eigen_data, .filled = filled };
}

const SpinLoopMetrics = struct {
    iterations: usize = 0,
    converged: bool = false,
    target_magnetization: f64 = 0.0,
    last_band_energy: f64 = 0.0,
    last_nonlocal_energy: f64 = 0.0,
    last_entropy_energy: f64 = 0.0,
    last_fermi_level: f64 = std.math.nan(f64),
    last_potential_residual: f64 = 0.0,
};

const SpinFermiLevels = struct {
    up: f64,
    down: f64,
    reference: f64,
};

const SpinDensityPair = struct {
    up: []f64,
    down: []f64,

    fn init_zero(alloc: std.mem.Allocator, count: usize) !SpinDensityPair {
        const up = try alloc.alloc(f64, count);
        errdefer alloc.free(up);

        const down = try alloc.alloc(f64, count);
        errdefer alloc.free(down);

        @memset(up, 0.0);
        @memset(down, 0.0);
        return .{ .up = up, .down = down };
    }

    fn deinit(self: *SpinDensityPair, alloc: std.mem.Allocator) void {
        alloc.free(self.up);
        alloc.free(self.down);
    }
};

const SpinOptionalDensityPair = struct {
    up: ?[]f64 = null,
    down: ?[]f64 = null,

    fn deinit(self: *SpinOptionalDensityPair, alloc: std.mem.Allocator) void {
        if (self.up) |values| alloc.free(values);
        if (self.down) |values| alloc.free(values);
    }
};

const SpinPotentialPair = struct {
    up: hamiltonian.PotentialGrid,
    down: hamiltonian.PotentialGrid,

    fn deinit(self: *SpinPotentialPair, alloc: std.mem.Allocator) void {
        self.up.deinit(alloc);
        self.down.deinit(alloc);
    }

    fn take_up(self: *SpinPotentialPair) hamiltonian.PotentialGrid {
        const values = self.up;
        self.up.values = &.{};
        return values;
    }

    fn take_down(self: *SpinPotentialPair) hamiltonian.PotentialGrid {
        const values = self.down;
        self.down.values = &.{};
        return values;
    }
};

const SpinChannelResults = struct {
    up: SpinEigenResult,
    down: SpinEigenResult,

    fn deinit(self: *SpinChannelResults, alloc: std.mem.Allocator) void {
        for (self.up.eigen_data[0..self.up.filled]) |*entry| entry.deinit(alloc);
        alloc.free(self.up.eigen_data);
        for (self.down.eigen_data[0..self.down.filled]) |*entry| entry.deinit(alloc);
        alloc.free(self.down.eigen_data);
    }
};

const SpinIterationEnergies = struct {
    band: f64 = 0.0,
    nonlocal: f64 = 0.0,
    entropy: f64 = 0.0,
};

const SpinIterationConvergence = struct {
    diff: f64,
    residual: f64,
    conv_value: f64,
};

const SpinPawResults = struct {
    tabs: ?[]paw_mod.PawTab = null,
    dij: ?[][]f64 = null,
    dij_m: ?[][]f64 = null,
    dxc: ?[][]f64 = null,
    rhoij: ?[][]f64 = null,

    fn deinit(self: *SpinPawResults, alloc: std.mem.Allocator) void {
        if (self.dij) |values| {
            for (values) |entry| alloc.free(entry);
            alloc.free(values);
        }
        if (self.dij_m) |values| {
            for (values) |entry| alloc.free(entry);
            alloc.free(values);
        }
        if (self.dxc) |values| {
            for (values) |entry| alloc.free(entry);
            alloc.free(values);
        }
        if (self.rhoij) |values| {
            for (values) |entry| alloc.free(entry);
            alloc.free(values);
        }
        if (self.tabs) |tabs| {
            for (@constCast(tabs)) |*tab| tab.deinit(alloc);
            alloc.free(tabs);
        }
    }
};

const SpinFinalWavefunctionData = struct {
    wavefunctions_up: ?WavefunctionData = null,
    wavefunctions_down: ?WavefunctionData = null,
    vxc_r_up: ?[]f64 = null,
    vxc_r_down: ?[]f64 = null,
    band_energy: f64 = 0.0,
    nonlocal_energy: f64 = 0.0,

    fn deinit(self: *SpinFinalWavefunctionData, alloc: std.mem.Allocator) void {
        if (self.wavefunctions_up) |*wf| wf.deinit(alloc);
        if (self.wavefunctions_down) |*wf| wf.deinit(alloc);
        if (self.vxc_r_up) |values| alloc.free(values);
        if (self.vxc_r_down) |values| alloc.free(values);
    }
};

const SpinLoopResources = struct {
    kpoint_cache_up: []KpointCache,
    kpoint_cache_down: []KpointCache,
    apply_caches_up: []apply.KpointApplyCache,
    apply_caches_down: []apply.KpointApplyCache,
    rho_up: []f64,
    rho_down: []f64,
    potential_up: hamiltonian.PotentialGrid,
    potential_down: hamiltonian.PotentialGrid,
    paw_rhoij_up: ?paw_mod.RhoIJ,
    paw_rhoij_down: ?paw_mod.RhoIJ,
    ecutrho_scf: f64,

    fn deinit(self: *SpinLoopResources, alloc: std.mem.Allocator) void {
        deinit_kpoint_caches(alloc, self.kpoint_cache_up);
        deinit_kpoint_caches(alloc, self.kpoint_cache_down);
        deinit_apply_caches(alloc, self.apply_caches_up);
        deinit_apply_caches(alloc, self.apply_caches_down);
        if (self.rho_up.len > 0) alloc.free(self.rho_up);
        if (self.rho_down.len > 0) alloc.free(self.rho_down);
        self.potential_up.deinit(alloc);
        self.potential_down.deinit(alloc);
        if (self.paw_rhoij_up) |*rij| rij.deinit(alloc);
        if (self.paw_rhoij_down) |*rij| rij.deinit(alloc);
    }

    fn reset_kpoint_caches(self: *SpinLoopResources) void {
        for (self.kpoint_cache_up) |*cache| cache.deinit();
        for (self.kpoint_cache_up) |*cache| cache.* = .{};
        for (self.kpoint_cache_down) |*cache| cache.deinit();
        for (self.kpoint_cache_down) |*cache| cache.* = .{};
    }

    fn take_potential_up(self: *SpinLoopResources) hamiltonian.PotentialGrid {
        const result = self.potential_up;
        self.potential_up.values = &.{};
        return result;
    }

    fn take_potential_down(self: *SpinLoopResources) hamiltonian.PotentialGrid {
        const result = self.potential_down;
        self.potential_down.values = &.{};
        return result;
    }

    fn take_rho_up(self: *SpinLoopResources) []f64 {
        const result = self.rho_up;
        self.rho_up = &.{};
        return result;
    }

    fn take_rho_down(self: *SpinLoopResources) []f64 {
        const result = self.rho_down;
        self.rho_down = &.{};
        return result;
    }
};

fn init_kpoint_caches(alloc: std.mem.Allocator, count: usize) ![]KpointCache {
    const caches = try alloc.alloc(KpointCache, count);
    for (caches) |*cache| cache.* = .{};
    return caches;
}

fn deinit_kpoint_caches(alloc: std.mem.Allocator, caches: []KpointCache) void {
    for (caches) |*cache| cache.deinit();
    alloc.free(caches);
}

fn init_apply_caches(alloc: std.mem.Allocator, count: usize) ![]apply.KpointApplyCache {
    const caches = try alloc.alloc(apply.KpointApplyCache, count);
    for (caches) |*cache| cache.* = .{};
    return caches;
}

fn deinit_apply_caches(
    alloc: std.mem.Allocator,
    caches: []apply.KpointApplyCache,
) void {
    for (caches) |*cache| cache.deinit(alloc);
    alloc.free(caches);
}

fn spin_ecutrho(cfg: *const config.Config) f64 {
    const gs_scf = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_scf * gs_scf;
}

fn compute_initial_magnetization(cfg: *const config.Config, total_electrons: f64) f64 {
    var magnetization: f64 = 0.0;
    if (cfg.scf.spinat) |values| {
        for (values) |value| magnetization += value;
    }
    return std.math.clamp(magnetization, -total_electrons, total_electrons);
}

fn init_spin_densities(
    ctx: *const SpinContext,
    rho_up: []f64,
    rho_down: []f64,
) !f64 {
    const total_electrons = ctx.common.total_electrons;
    const magnetization = compute_initial_magnetization(ctx.cfg, total_electrons);
    const atomic_rho = try scf_mod.build_atomic_density(
        ctx.alloc,
        ctx.common.grid,
        ctx.common.species,
        ctx.atoms,
    );
    defer ctx.alloc.free(atomic_rho);

    var sum: f64 = 0.0;
    const dv = ctx.common.grid.volume / @as(f64, @floatFromInt(ctx.common.grid.count()));
    for (atomic_rho) |value| sum += value * dv;

    const scale = if (sum > 1e-10) total_electrons / sum else 1.0;
    const frac_up = (total_electrons + magnetization) / (2.0 * total_electrons);
    const frac_down = (total_electrons - magnetization) / (2.0 * total_electrons);
    for (atomic_rho, 0..) |value, index| {
        const rho_scaled = value * scale;
        rho_up[index] = rho_scaled * frac_up;
        rho_down[index] = rho_scaled * frac_down;
    }
    return magnetization;
}

fn build_initial_spin_potentials(
    ctx: *const SpinContext,
    rho_up: []const f64,
    rho_down: []const f64,
) !SpinPotentialPair {
    const spin_potentials = try potential_mod.build_potential_grid_spin(
        ctx.alloc,
        ctx.common.grid,
        rho_up,
        rho_down,
        ctx.common.rho_core,
        ctx.cfg.scf.use_rfft,
        ctx.cfg.scf.xc,
        null,
        null,
        ctx.common.coulomb_r_cut,
    );
    return .{
        .up = spin_potentials.up,
        .down = spin_potentials.down,
    };
}

fn clone_spin_rhoij(
    alloc: std.mem.Allocator,
    common: *scf_mod.ScfCommon,
) !struct { up: ?paw_mod.RhoIJ, down: ?paw_mod.RhoIJ } {
    var up = if (common.paw_rhoij) |*prij| try prij.clone(alloc) else null;
    errdefer if (up) |*rij| rij.deinit(alloc);

    var down = if (common.paw_rhoij) |*prij| try prij.clone(alloc) else null;
    errdefer if (down) |*rij| rij.deinit(alloc);

    return .{ .up = up, .down = down };
}

fn init_spin_loop_resources(ctx: *const SpinContext) !struct {
    resources: SpinLoopResources,
    magnetization: f64,
} {
    const count = ctx.common.kpoints.len;
    const grid_count = ctx.common.grid.count();
    const kpoint_cache_up = try init_kpoint_caches(ctx.alloc, count);
    errdefer deinit_kpoint_caches(ctx.alloc, kpoint_cache_up);

    const kpoint_cache_down = try init_kpoint_caches(ctx.alloc, count);
    errdefer deinit_kpoint_caches(ctx.alloc, kpoint_cache_down);

    const apply_caches_up = try init_apply_caches(ctx.alloc, count);
    errdefer deinit_apply_caches(ctx.alloc, apply_caches_up);

    const apply_caches_down = try init_apply_caches(ctx.alloc, count);
    errdefer deinit_apply_caches(ctx.alloc, apply_caches_down);

    const rho_up = try ctx.alloc.alloc(f64, grid_count);
    errdefer ctx.alloc.free(rho_up);

    const rho_down = try ctx.alloc.alloc(f64, grid_count);
    errdefer ctx.alloc.free(rho_down);

    const magnetization = try init_spin_densities(ctx, rho_up, rho_down);
    const potentials = try build_initial_spin_potentials(ctx, rho_up, rho_down);
    errdefer {
        var pair = potentials;
        pair.deinit(ctx.alloc);
    }

    const paw_rhoij = try clone_spin_rhoij(ctx.alloc, ctx.common);
    errdefer {
        if (paw_rhoij.up) |*rij| rij.deinit(ctx.alloc);
        if (paw_rhoij.down) |*rij| rij.deinit(ctx.alloc);
    }

    return .{
        .resources = .{
            .kpoint_cache_up = kpoint_cache_up,
            .kpoint_cache_down = kpoint_cache_down,
            .apply_caches_up = apply_caches_up,
            .apply_caches_down = apply_caches_down,
            .rho_up = rho_up,
            .rho_down = rho_down,
            .potential_up = potentials.up,
            .potential_down = potentials.down,
            .paw_rhoij_up = paw_rhoij.up,
            .paw_rhoij_down = paw_rhoij.down,
            .ecutrho_scf = spin_ecutrho(ctx.cfg),
        },
        .magnetization = magnetization,
    };
}

fn reset_spin_iteration_rhoij(
    common: *scf_mod.ScfCommon,
    resources: *SpinLoopResources,
) void {
    if (common.paw_rhoij) |*prij| prij.reset();
    if (resources.paw_rhoij_up) |*rij| rij.reset();
    if (resources.paw_rhoij_down) |*rij| rij.reset();
}

fn solve_spin_channels_once(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    iteration: usize,
    shared_fft_plan: fft.Fft3dPlan,
) !SpinChannelResults {
    const up = try solve_kpoints_for_spin(
        ctx.alloc,
        ctx.io,
        ctx.cfg.*,
        ctx.common,
        resources.potential_up,
        resources.kpoint_cache_up,
        resources.apply_caches_up,
        iteration,
        shared_fft_plan,
    );
    errdefer {
        for (up.eigen_data[0..up.filled]) |*entry| entry.deinit(ctx.alloc);
        ctx.alloc.free(up.eigen_data);
    }

    const down = try solve_kpoints_for_spin(
        ctx.alloc,
        ctx.io,
        ctx.cfg.*,
        ctx.common,
        resources.potential_down,
        resources.kpoint_cache_down,
        resources.apply_caches_down,
        iteration,
        shared_fft_plan,
    );
    errdefer {
        for (down.eigen_data[0..down.filled]) |*entry| entry.deinit(ctx.alloc);
        ctx.alloc.free(down.eigen_data);
    }

    return .{ .up = up, .down = down };
}

fn bootstrap_spin_paw_dij_if_needed(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    iteration: usize,
    channels: *SpinChannelResults,
    shared_fft_plan: fft.Fft3dPlan,
) !void {
    if (iteration != 0 or !ctx.common.is_paw) return;
    const tabs = ctx.common.paw_tabs orelse return;

    try paw_scf.update_paw_dij(
        ctx.alloc,
        ctx.common.grid,
        ctx.common.ionic,
        resources.potential_up,
        tabs,
        ctx.species,
        ctx.atoms,
        resources.apply_caches_up,
        resources.ecutrho_scf,
        &ctx.common.paw_rhoij.?,
        ctx.cfg.scf.xc,
        ctx.cfg.scf.symmetry,
        &ctx.common.paw_gaunt.?,
        true,
        null,
        1.0,
    );
    try paw_scf.update_paw_dij(
        ctx.alloc,
        ctx.common.grid,
        ctx.common.ionic,
        resources.potential_down,
        tabs,
        ctx.species,
        ctx.atoms,
        resources.apply_caches_down,
        resources.ecutrho_scf,
        &ctx.common.paw_rhoij.?,
        ctx.cfg.scf.xc,
        ctx.cfg.scf.symmetry,
        &ctx.common.paw_gaunt.?,
        true,
        null,
        1.0,
    );

    resources.reset_kpoint_caches();
    const replacement = try solve_spin_channels_once(
        ctx,
        resources,
        iteration,
        shared_fft_plan,
    );
    channels.deinit(ctx.alloc);
    channels.* = replacement;
}

fn solve_spin_channels(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    iteration: usize,
) !SpinChannelResults {
    var shared_fft_plan = try fft.Fft3dPlan.init_with_backend(
        ctx.alloc,
        ctx.io,
        ctx.common.grid.nx,
        ctx.common.grid.ny,
        ctx.common.grid.nz,
        ctx.cfg.scf.fft_backend,
    );
    defer shared_fft_plan.deinit(ctx.alloc);

    var channels = try solve_spin_channels_once(
        ctx,
        resources,
        iteration,
        shared_fft_plan,
    );
    errdefer channels.deinit(ctx.alloc);

    try bootstrap_spin_paw_dij_if_needed(
        ctx,
        resources,
        iteration,
        &channels,
        shared_fft_plan,
    );
    return channels;
}

fn compute_spin_fermi_levels(
    ctx: *const SpinContext,
    metrics: *SpinLoopMetrics,
    channels: *const SpinChannelResults,
) SpinFermiLevels {
    const use_fsm = ctx.common.is_paw and @abs(metrics.target_magnetization) > 0.1;
    const nelec = ctx.common.total_electrons;
    const ne_up_target = (nelec + metrics.target_magnetization) / 2.0;
    const ne_down_target = (nelec - metrics.target_magnetization) / 2.0;
    const up = if (use_fsm)
        find_fermi_level_spin(
            ne_up_target,
            ctx.cfg.scf.smear_ry,
            ctx.cfg.scf.smearing,
            channels.up.eigen_data[0..channels.up.filled],
            null,
            1.0,
        )
    else
        find_fermi_level_spin(
            nelec,
            ctx.cfg.scf.smear_ry,
            ctx.cfg.scf.smearing,
            channels.up.eigen_data[0..channels.up.filled],
            channels.down.eigen_data[0..channels.down.filled],
            1.0,
        );
    const down = if (use_fsm)
        find_fermi_level_spin(
            ne_down_target,
            ctx.cfg.scf.smear_ry,
            ctx.cfg.scf.smearing,
            channels.down.eigen_data[0..channels.down.filled],
            null,
            1.0,
        )
    else
        up;
    return .{ .up = up, .down = down, .reference = up };
}

fn accumulate_spin_channel_density(
    ctx: *const SpinContext,
    entries: []const KpointEigenData,
    mu: f64,
    rho_out: []f64,
    apply_caches: []apply.KpointApplyCache,
    paw_rhoij: ?*paw_mod.RhoIJ,
    energies: *SpinIterationEnergies,
    fft_index_map: []const usize,
) !void {
    for (entries, 0..) |entry, kidx| {
        try accumulate_kpoint_density_smearing_spin(
            ctx.alloc,
            ctx.io,
            ctx.cfg,
            ctx.common.grid,
            ctx.common.kpoints[kidx],
            entry,
            ctx.common.recip,
            ctx.volume_bohr,
            fft_index_map,
            mu,
            ctx.cfg.scf.smear_ry,
            rho_out,
            &energies.band,
            &energies.nonlocal,
            &energies.entropy,
            null,
            1.0,
            if (kidx < apply_caches.len) &apply_caches[kidx] else null,
            paw_rhoij,
            ctx.atoms,
        );
    }
}

fn combine_spin_paw_rhoij(
    common: *scf_mod.ScfCommon,
    resources: *SpinLoopResources,
) void {
    if (common.paw_rhoij) |*prij| {
        const rij_up = resources.paw_rhoij_up orelse return;
        const rij_down = resources.paw_rhoij_down orelse return;
        for (0..prij.natom) |atom_index| {
            for (0..prij.values[atom_index].len) |value_index| {
                prij.values[atom_index][value_index] =
                    rij_up.values[atom_index][value_index] +
                    rij_down.values[atom_index][value_index];
            }
        }
    }
}

fn accumulate_spin_iteration_outputs(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    channels: *const SpinChannelResults,
    fermi_levels: SpinFermiLevels,
    rho_out: *SpinDensityPair,
    energies: *SpinIterationEnergies,
) !void {
    const fft_index_map = try build_fft_index_map(ctx.alloc, ctx.common.grid);
    defer ctx.alloc.free(fft_index_map);

    try accumulate_spin_channel_density(
        ctx,
        channels.up.eigen_data[0..channels.up.filled],
        fermi_levels.up,
        rho_out.up,
        resources.apply_caches_up,
        if (resources.paw_rhoij_up) |*rij| rij else null,
        energies,
        fft_index_map,
    );
    try accumulate_spin_channel_density(
        ctx,
        channels.down.eigen_data[0..channels.down.filled],
        fermi_levels.down,
        rho_out.down,
        resources.apply_caches_down,
        if (resources.paw_rhoij_down) |*rij| rij else null,
        energies,
        fft_index_map,
    );
    combine_spin_paw_rhoij(ctx.common, resources);
}

fn symmetrize_spin_outputs(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *SpinDensityPair,
) !void {
    if (ctx.common.sym_ops) |ops| {
        if (ops.len > 1) {
            try scf_mod.symmetrize_density(
                ctx.alloc,
                ctx.common.grid,
                rho_out.up,
                ops,
                ctx.cfg.scf.use_rfft,
            );
            try scf_mod.symmetrize_density(
                ctx.alloc,
                ctx.common.grid,
                rho_out.down,
                ops,
                ctx.cfg.scf.use_rfft,
            );
        }
    }
    if (ctx.common.paw_rhoij) |*prij| {
        if (ctx.cfg.scf.symmetry) {
            try paw_scf.symmetrize_rho_ij(ctx.alloc, prij, ctx.species, ctx.atoms);
        }
    }
    _ = resources;
}

fn filter_spin_density_in_place(
    ctx: *const SpinContext,
    density: []f64,
    ecutrho_scf: f64,
) !void {
    const filtered = try potential_mod.filter_density_to_ecutrho(
        ctx.alloc,
        ctx.common.grid,
        density,
        ecutrho_scf,
        ctx.cfg.scf.use_rfft,
    );
    defer ctx.alloc.free(filtered);

    @memcpy(density, filtered);
}

fn build_spin_compensated_density_pair(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_up: []const f64,
    rho_down: []const f64,
) !SpinDensityPair {
    const grid_count = ctx.common.grid.count();
    const n_hat = try ctx.alloc.alloc(f64, grid_count);
    defer ctx.alloc.free(n_hat);

    @memset(n_hat, 0.0);
    try paw_scf.add_paw_compensation_charge(
        ctx.alloc,
        ctx.common.grid,
        n_hat,
        &ctx.common.paw_rhoij.?,
        ctx.common.paw_tabs.?,
        ctx.atoms,
        resources.ecutrho_scf,
        &ctx.common.paw_gaunt.?,
    );

    var augmented = try SpinDensityPair.init_zero(ctx.alloc, grid_count);
    for (0..grid_count) |index| {
        augmented.up[index] = rho_up[index] + n_hat[index] * 0.5;
        augmented.down[index] = rho_down[index] + n_hat[index] * 0.5;
    }
    return augmented;
}

fn build_spin_potential_inputs(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *SpinDensityPair,
) !SpinOptionalDensityPair {
    try symmetrize_spin_outputs(ctx, resources, rho_out);
    if (!ctx.common.is_paw) return .{};

    try filter_spin_density_in_place(ctx, rho_out.up, resources.ecutrho_scf);
    try filter_spin_density_in_place(ctx, rho_out.down, resources.ecutrho_scf);
    if (ctx.common.paw_rhoij == null) return .{};

    const augmented = try build_spin_compensated_density_pair(
        ctx,
        resources,
        rho_out.up,
        rho_out.down,
    );
    return .{
        .up = augmented.up,
        .down = augmented.down,
    };
}

fn build_spin_output_potentials(
    ctx: *const SpinContext,
    rho_out: *const SpinDensityPair,
    rho_aug: *const SpinOptionalDensityPair,
) !SpinPotentialPair {
    const rho_up = rho_aug.up orelse rho_out.up;
    const rho_down = rho_aug.down orelse rho_out.down;
    return build_initial_spin_potentials(ctx, rho_up, rho_down);
}

fn compute_spin_potential_residual(
    resources: *const SpinLoopResources,
    pot_out: *const SpinPotentialPair,
) f64 {
    const nvals = resources.potential_up.values.len;
    var sum_sq: f64 = 0.0;
    for (0..nvals) |index| {
        const diff_up = math.complex.sub(
            pot_out.up.values[index],
            resources.potential_up.values[index],
        );
        const diff_down = math.complex.sub(
            pot_out.down.values[index],
            resources.potential_down.values[index],
        );
        sum_sq += diff_up.r * diff_up.r + diff_up.i * diff_up.i;
        sum_sq += diff_down.r * diff_down.r + diff_down.i * diff_down.i;
    }
    if (nvals == 0) return 0.0;
    return std.math.sqrt(sum_sq / @as(f64, @floatFromInt(2 * nvals)));
}

fn compute_spin_iteration_convergence(
    ctx: *const SpinContext,
    resources: *const SpinLoopResources,
    rho_out: *const SpinDensityPair,
    pot_out: *const SpinPotentialPair,
) SpinIterationConvergence {
    const residual = compute_spin_potential_residual(resources, pot_out);
    const grid_count = ctx.common.grid.count();
    var sum_diff_sq: f64 = 0.0;
    for (0..grid_count) |index| {
        const rho_total = rho_out.up[index] + rho_out.down[index];
        const rho_in = resources.rho_up[index] + resources.rho_down[index];
        const diff = rho_total - rho_in;
        sum_diff_sq += diff * diff;
    }
    const diff = if (grid_count > 0)
        std.math.sqrt(sum_diff_sq / @as(f64, @floatFromInt(grid_count)))
    else
        0.0;
    const conv_value = switch (ctx.cfg.scf.convergence_metric) {
        .density => diff,
        .potential => residual,
    };
    return .{ .diff = diff, .residual = residual, .conv_value = conv_value };
}

fn log_spin_iteration_progress(
    ctx: *const SpinContext,
    metrics: *const SpinLoopMetrics,
    convergence: SpinIterationConvergence,
) !void {
    try ctx.common.log.write_iter(
        metrics.iterations,
        convergence.diff,
        convergence.residual,
        metrics.last_band_energy,
        metrics.last_nonlocal_energy,
    );
    if (!ctx.cfg.scf.quiet) {
        try log_progress(
            ctx.io,
            metrics.iterations,
            convergence.diff,
            convergence.residual,
            metrics.last_band_energy,
            metrics.last_nonlocal_energy,
        );
    }
}

fn accept_spin_converged_step(
    alloc: std.mem.Allocator,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    pot_out: *SpinPotentialPair,
) void {
    @memcpy(resources.rho_up, rho_out.up);
    @memcpy(resources.rho_down, rho_out.down);
    resources.potential_up.deinit(alloc);
    resources.potential_up = pot_out.take_up();
    resources.potential_down.deinit(alloc);
    resources.potential_down = pot_out.take_down();
}

fn apply_spin_potential_pulay(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    pot_out: *const SpinPotentialPair,
    n_complex: usize,
    n_f64: usize,
    v_in_up: []f64,
    v_in_down: []f64,
) !void {
    const residual_concat = try ctx.alloc.alloc(f64, n_f64 * 2);
    const pot_out_up_f: [*]const f64 = @ptrCast(pot_out.up.values.ptr);
    const pot_out_down_f: [*]const f64 = @ptrCast(pot_out.down.values.ptr);
    for (0..n_f64) |index| {
        residual_concat[index] = pot_out_up_f[index] - v_in_up[index];
    }
    for (0..n_f64) |index| {
        residual_concat[n_f64 + index] = pot_out_down_f[index] - v_in_down[index];
    }
    if (ctx.cfg.scf.diemac > 1.0) {
        const res_up_c: []math.Complex = @as(
            [*]math.Complex,
            @ptrCast(@alignCast(residual_concat.ptr)),
        )[0..n_complex];
        const res_down_c: []math.Complex = @as(
            [*]math.Complex,
            @ptrCast(@alignCast(residual_concat[n_f64..].ptr)),
        )[0..n_complex];
        mixing.apply_model_dielectric_preconditioner(
            ctx.common.grid,
            res_up_c,
            ctx.cfg.scf.diemac,
            ctx.cfg.scf.dielng,
        );
        mixing.apply_model_dielectric_preconditioner(
            ctx.common.grid,
            res_down_c,
            ctx.cfg.scf.diemac,
            ctx.cfg.scf.dielng,
        );
    }

    const concat_in = try ctx.alloc.alloc(f64, n_f64 * 2);
    defer ctx.alloc.free(concat_in);

    @memcpy(concat_in[0..n_f64], v_in_up);
    @memcpy(concat_in[n_f64..], v_in_down);
    try ctx.common.pulay_mixer.?.mix_with_residual(
        concat_in,
        residual_concat,
        ctx.cfg.scf.mixing_beta,
    );
    @memcpy(v_in_up, concat_in[0..n_f64]);
    @memcpy(v_in_down, concat_in[n_f64..]);
    _ = resources;
}

fn apply_spin_potential_mixing(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    iteration: usize,
    pot_out: *SpinPotentialPair,
) !void {
    const n_complex = resources.potential_up.values.len;
    const n_f64 = n_complex * 2;
    const v_in_up: []f64 = @as([*]f64, @ptrCast(resources.potential_up.values.ptr))[0..n_f64];
    const v_out_up_f: []const f64 =
        @as([*]const f64, @ptrCast(pot_out.up.values.ptr))[0..n_f64];
    const v_in_down: []f64 = @as([*]f64, @ptrCast(resources.potential_down.values.ptr))[0..n_f64];
    const v_out_down_f: []const f64 =
        @as([*]const f64, @ptrCast(pot_out.down.values.ptr))[0..n_f64];

    if (ctx.common.pulay_mixer != null and iteration >= ctx.cfg.scf.pulay_start) {
        try apply_spin_potential_pulay(
            ctx,
            resources,
            pot_out,
            n_complex,
            n_f64,
            v_in_up,
            v_in_down,
        );
    } else {
        mix_density(v_in_up, v_out_up_f, ctx.cfg.scf.mixing_beta);
        mix_density(v_in_down, v_out_down_f, ctx.cfg.scf.mixing_beta);
    }
    @memcpy(resources.rho_up, rho_out.up);
    @memcpy(resources.rho_down, rho_out.down);
    pot_out.deinit(ctx.alloc);
}

fn mix_spin_density_inputs(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    iteration: usize,
) !void {
    const force_density_mixing = false;
    if (force_density_mixing) {
        const paw_beta: f64 = 0.05;
        const paw_q0: f64 = 1.5;
        try mix_density_kerker(
            ctx.alloc,
            ctx.common.grid,
            resources.rho_up,
            rho_out.up,
            paw_beta,
            paw_q0,
            ctx.cfg.scf.use_rfft,
        );
        try mix_density_kerker(
            ctx.alloc,
            ctx.common.grid,
            resources.rho_down,
            rho_out.down,
            paw_beta,
            paw_q0,
            ctx.cfg.scf.use_rfft,
        );
        return;
    }
    if (ctx.common.pulay_mixer != null and iteration >= ctx.cfg.scf.pulay_start) {
        const grid_count = ctx.common.grid.count();
        const concat_in = try ctx.alloc.alloc(f64, grid_count * 2);
        defer ctx.alloc.free(concat_in);

        const concat_out = try ctx.alloc.alloc(f64, grid_count * 2);
        defer ctx.alloc.free(concat_out);

        @memcpy(concat_in[0..grid_count], resources.rho_up);
        @memcpy(concat_in[grid_count..], resources.rho_down);
        @memcpy(concat_out[0..grid_count], rho_out.up);
        @memcpy(concat_out[grid_count..], rho_out.down);
        try ctx.common.pulay_mixer.?.mix(
            concat_in,
            concat_out,
            ctx.cfg.scf.mixing_beta,
        );
        @memcpy(resources.rho_up, concat_in[0..grid_count]);
        @memcpy(resources.rho_down, concat_in[grid_count..]);
        return;
    }
    mix_density(resources.rho_up, rho_out.up, ctx.cfg.scf.mixing_beta);
    mix_density(resources.rho_down, rho_out.down, ctx.cfg.scf.mixing_beta);
}

fn build_spin_mixed_density_augmentation(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
) !SpinOptionalDensityPair {
    if (!ctx.common.is_paw or ctx.common.paw_rhoij == null) return .{};
    const augmented = try build_spin_compensated_density_pair(
        ctx,
        resources,
        resources.rho_up,
        resources.rho_down,
    );
    return .{
        .up = augmented.up,
        .down = augmented.down,
    };
}

fn rebuild_spin_potentials_from_mixed_density(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
) !void {
    var augmented = try build_spin_mixed_density_augmentation(ctx, resources);
    defer augmented.deinit(ctx.alloc);

    const rho_up = augmented.up orelse resources.rho_up;
    const rho_down = augmented.down orelse resources.rho_down;
    const rebuilt = try build_initial_spin_potentials(ctx, rho_up, rho_down);
    resources.potential_up = rebuilt.up;
    resources.potential_down = rebuilt.down;
}

fn apply_spin_density_mixing(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    iteration: usize,
    pot_out: *SpinPotentialPair,
) !void {
    pot_out.deinit(ctx.alloc);
    resources.potential_up.deinit(ctx.alloc);
    resources.potential_down.deinit(ctx.alloc);
    try mix_spin_density_inputs(ctx, resources, rho_out, iteration);
    try rebuild_spin_potentials_from_mixed_density(ctx, resources);
}

fn mix_spin_iteration_state(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    iteration: usize,
    pot_out: *SpinPotentialPair,
) !void {
    const force_density_mixing = false;
    if (ctx.cfg.scf.mixing_mode == .potential and !force_density_mixing) {
        try apply_spin_potential_mixing(
            ctx,
            resources,
            rho_out,
            iteration,
            pot_out,
        );
        return;
    }
    try apply_spin_density_mixing(ctx, resources, rho_out, iteration, pot_out);
}

fn update_one_spin_paw_dij(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    potential: hamiltonian.PotentialGrid,
    apply_caches: []apply.KpointApplyCache,
    rhoij: ?*paw_mod.RhoIJ,
    tabs: []paw_mod.PawTab,
    mix_beta: f64,
) !void {
    try paw_scf.update_paw_dij(
        ctx.alloc,
        ctx.common.grid,
        ctx.common.ionic,
        potential,
        tabs,
        ctx.species,
        ctx.atoms,
        apply_caches,
        resources.ecutrho_scf,
        &ctx.common.paw_rhoij.?,
        ctx.cfg.scf.xc,
        ctx.cfg.scf.symmetry,
        &ctx.common.paw_gaunt.?,
        false,
        rhoij,
        mix_beta,
    );
}

fn update_spin_paw_dij_if_needed(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
) !void {
    if (!ctx.common.is_paw) return;
    const tabs = ctx.common.paw_tabs orelse return;
    const mix_beta = ctx.cfg.scf.mixing_beta;
    try update_one_spin_paw_dij(
        ctx,
        resources,
        resources.potential_up,
        resources.apply_caches_up,
        if (resources.paw_rhoij_up) |*rij| rij else null,
        tabs,
        mix_beta,
    );
    try update_one_spin_paw_dij(
        ctx,
        resources,
        resources.potential_down,
        resources.apply_caches_down,
        if (resources.paw_rhoij_down) |*rij| rij else null,
        tabs,
        mix_beta,
    );
}

fn run_spin_iteration(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    metrics: *SpinLoopMetrics,
) !bool {
    if (!ctx.cfg.scf.quiet) {
        try log_iter_start(ctx.io, metrics.iterations);
    }
    reset_spin_iteration_rhoij(ctx.common, resources);

    var rho_out = try SpinDensityPair.init_zero(ctx.alloc, ctx.common.grid.count());
    defer rho_out.deinit(ctx.alloc);

    var channels = try solve_spin_channels(ctx, resources, metrics.iterations);
    defer channels.deinit(ctx.alloc);

    const fermi_levels = compute_spin_fermi_levels(ctx, metrics, &channels);
    var energies: SpinIterationEnergies = .{};
    try accumulate_spin_iteration_outputs(
        ctx,
        resources,
        &channels,
        fermi_levels,
        &rho_out,
        &energies,
    );
    metrics.last_band_energy = energies.band;
    metrics.last_nonlocal_energy = energies.nonlocal;
    metrics.last_entropy_energy = energies.entropy;
    metrics.last_fermi_level = fermi_levels.reference;

    var rho_aug = try build_spin_potential_inputs(ctx, resources, &rho_out);
    defer rho_aug.deinit(ctx.alloc);

    var pot_out = try build_spin_output_potentials(ctx, &rho_out, &rho_aug);
    var pot_out_owned = true;
    defer if (pot_out_owned) pot_out.deinit(ctx.alloc);

    const convergence = compute_spin_iteration_convergence(
        ctx,
        resources,
        &rho_out,
        &pot_out,
    );
    metrics.last_potential_residual = convergence.residual;
    try log_spin_iteration_progress(ctx, metrics, convergence);

    if (convergence.conv_value < ctx.cfg.scf.convergence) {
        accept_spin_converged_step(ctx.alloc, resources, &rho_out, &pot_out);
        pot_out_owned = false;
        return true;
    }

    try mix_spin_iteration_state(
        ctx,
        resources,
        &rho_out,
        metrics.iterations,
        &pot_out,
    );
    pot_out_owned = false;
    try update_spin_paw_dij_if_needed(ctx, resources);
    return false;
}

fn run_spin_scf_loop(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    target_magnetization: f64,
) !SpinLoopMetrics {
    var metrics = SpinLoopMetrics{
        .target_magnetization = target_magnetization,
    };
    while (metrics.iterations < ctx.cfg.scf.max_iter) : (metrics.iterations += 1) {
        if (try run_spin_iteration(ctx, resources, &metrics)) {
            metrics.converged = true;
            break;
        }
    }
    return metrics;
}

fn compute_spin_magnetization(
    grid: Grid,
    rho_up: []const f64,
    rho_down: []const f64,
) f64 {
    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var magnetization: f64 = 0.0;
    for (0..grid.count()) |index| {
        magnetization += (rho_up[index] - rho_down[index]) * dv;
    }
    return magnetization;
}

fn build_spin_total_density(
    alloc: std.mem.Allocator,
    rho_up: []const f64,
    rho_down: []const f64,
) ![]f64 {
    const density = try alloc.alloc(f64, rho_up.len);
    for (0..rho_up.len) |index| {
        density[index] = rho_up[index] + rho_down[index];
    }
    return density;
}

fn build_spin_energy_augmented_densities(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
) !SpinOptionalDensityPair {
    if (!ctx.common.is_paw or ctx.common.paw_rhoij == null) return .{};

    var augmented = try build_spin_compensated_density_pair(
        ctx,
        resources,
        resources.rho_up,
        resources.rho_down,
    );
    errdefer augmented.deinit(ctx.alloc);

    try filter_spin_density_in_place(ctx, augmented.up, resources.ecutrho_scf);
    try filter_spin_density_in_place(ctx, augmented.down, resources.ecutrho_scf);
    return .{
        .up = augmented.up,
        .down = augmented.down,
    };
}

fn build_spin_energy_terms(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    metrics: *const SpinLoopMetrics,
    rho_aug_energy: *const SpinOptionalDensityPair,
) !EnergyTerms {
    const paw_ecutrho: ?f64 = if (ctx.common.is_paw) resources.ecutrho_scf else null;
    return energy_mod.compute_energy_terms_spin(.{
        .alloc = ctx.alloc,
        .io = ctx.io,
        .grid = ctx.common.grid,
        .species = ctx.species,
        .atoms = ctx.atoms,
        .rho_up = resources.rho_up,
        .rho_down = resources.rho_down,
        .rho_core = ctx.common.rho_core,
        .rho_aug_up = rho_aug_energy.up,
        .rho_aug_down = rho_aug_energy.down,
        .band_energy = metrics.last_band_energy,
        .nonlocal_energy = metrics.last_nonlocal_energy,
        .entropy_energy = metrics.last_entropy_energy,
        .local_cfg = ctx.common.local_cfg,
        .ewald_cfg = ctx.cfg.ewald,
        .vdw_cfg = ctx.cfg.vdw,
        .xc_func = ctx.cfg.scf.xc,
        .use_rfft = ctx.cfg.scf.use_rfft,
        .quiet = ctx.cfg.scf.quiet,
        .coulomb_r_cut = ctx.common.coulomb_r_cut,
        .ecutrho = paw_ecutrho,
    });
}

fn apply_spin_paw_onsite_energy(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    energy_terms: *EnergyTerms,
) !void {
    if (!ctx.common.is_paw) return;
    const prij = &ctx.common.paw_rhoij.?;
    const tabs = ctx.common.paw_tabs orelse return;
    energy_terms.paw_onsite = try paw_scf.compute_paw_onsite_energy_total(
        ctx.alloc,
        prij,
        tabs,
        ctx.species,
        ctx.atoms,
        ctx.cfg.scf.xc,
        &ctx.common.paw_gaunt.?,
        if (resources.paw_rhoij_up) |*rij| rij else null,
        if (resources.paw_rhoij_down) |*rij| rij else null,
    );
    energy_terms.total += energy_terms.paw_onsite;
}

fn copy_spin_rho_core(
    alloc: std.mem.Allocator,
    rho_core: ?[]const f64,
) !?[]f64 {
    const values = rho_core orelse return null;
    const copy = try alloc.alloc(f64, values.len);
    @memcpy(copy, values);
    return copy;
}

fn dup_spin_per_atom_dij(
    alloc: std.mem.Allocator,
    nc_species: anytype,
    select: enum { radial, m_resolved },
) !?[][]f64 {
    var list: std.ArrayList([]f64) = .empty;
    errdefer {
        for (list.items) |entry| alloc.free(entry);
        list.deinit(alloc);
    }
    for (nc_species) |entry| {
        const source = switch (select) {
            .radial => entry.dij_per_atom,
            .m_resolved => entry.dij_m_per_atom,
        };
        if (source) |per_atom| {
            for (per_atom) |atom_values| {
                const copy = try alloc.alloc(f64, atom_values.len);
                @memcpy(copy, atom_values);
                try list.append(alloc, copy);
            }
        }
    }
    if (list.items.len > 0) return try list.toOwnedSlice(alloc);
    list.deinit(alloc);
    return null;
}

fn append_empty_spin_dxc(
    alloc: std.mem.Allocator,
    dxc_list: *std.ArrayList([]f64),
) !void {
    try dxc_list.append(alloc, try alloc.alloc(f64, 0));
}

fn accumulate_spin_matrix_trace(
    sum_dxc_rhoij: *f64,
    matrix: []const f64,
    rhoij: []const f64,
    mt: usize,
) void {
    for (0..mt) |im| {
        for (0..mt) |jm| {
            sum_dxc_rhoij.* += matrix[im * mt + jm] * rhoij[im * mt + jm];
        }
    }
}

fn compute_spin_atom_hartree_dc(
    ctx: *const SpinContext,
    paw: anytype,
    rhoij_m: []const f64,
    mt: usize,
    sp_m_offsets: anytype,
    upf: anytype,
) !f64 {
    const dij_h_dc = try ctx.alloc.alloc(f64, mt * mt);
    defer ctx.alloc.free(dij_h_dc);

    try paw_mod.paw_xc.compute_paw_dij_hartree_multi_l(
        ctx.alloc,
        dij_h_dc,
        paw,
        rhoij_m,
        mt,
        sp_m_offsets,
        upf.r,
        upf.rab,
        &ctx.common.paw_gaunt.?,
    );
    var sum_dxc_rhoij: f64 = 0.0;
    accumulate_spin_matrix_trace(&sum_dxc_rhoij, dij_h_dc, rhoij_m, mt);
    return sum_dxc_rhoij;
}

const SpinAtomDxcAndDc = struct {
    dxc: []f64,
    sum: f64,
};

fn compute_polarized_atom_dxc_and_dc(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    atom_index: usize,
    paw: anytype,
    rhoij_m: []const f64,
    mt: usize,
    sp_m_offsets: anytype,
    upf: anytype,
) !SpinAtomDxcAndDc {
    const dxc_up = try ctx.alloc.alloc(f64, mt * mt);
    errdefer ctx.alloc.free(dxc_up);

    const dxc_down = try ctx.alloc.alloc(f64, mt * mt);
    defer ctx.alloc.free(dxc_down);

    try paw_mod.paw_xc.compute_paw_dij_xc_angular_spin(
        ctx.alloc,
        dxc_up,
        dxc_down,
        paw,
        resources.paw_rhoij_up.?.values[atom_index],
        resources.paw_rhoij_down.?.values[atom_index],
        mt,
        sp_m_offsets,
        upf.r,
        upf.rab,
        paw.ae_core_density,
        if (upf.nlcc.len > 0) upf.nlcc else null,
        ctx.cfg.scf.xc,
        &ctx.common.paw_gaunt.?,
    );
    var sum_dxc_rhoij: f64 = 0.0;
    accumulate_spin_matrix_trace(
        &sum_dxc_rhoij,
        dxc_up,
        resources.paw_rhoij_up.?.values[atom_index],
        mt,
    );
    accumulate_spin_matrix_trace(
        &sum_dxc_rhoij,
        dxc_down,
        resources.paw_rhoij_down.?.values[atom_index],
        mt,
    );
    sum_dxc_rhoij += try compute_spin_atom_hartree_dc(
        ctx,
        paw,
        rhoij_m,
        mt,
        sp_m_offsets,
        upf,
    );
    return .{ .dxc = dxc_up, .sum = sum_dxc_rhoij };
}

fn compute_unpolarized_atom_dxc_and_dc(
    ctx: *const SpinContext,
    paw: anytype,
    rhoij_m: []const f64,
    mt: usize,
    sp_m_offsets: anytype,
    upf: anytype,
) !SpinAtomDxcAndDc {
    const dxc_m = try ctx.alloc.alloc(f64, mt * mt);
    errdefer ctx.alloc.free(dxc_m);

    try paw_mod.paw_xc.compute_paw_dij_xc_angular(
        ctx.alloc,
        dxc_m,
        paw,
        rhoij_m,
        mt,
        sp_m_offsets,
        upf.r,
        upf.rab,
        paw.ae_core_density,
        if (upf.nlcc.len > 0) upf.nlcc else null,
        ctx.cfg.scf.xc,
        &ctx.common.paw_gaunt.?,
    );
    var sum_dxc_rhoij: f64 = 0.0;
    accumulate_spin_matrix_trace(&sum_dxc_rhoij, dxc_m, rhoij_m, mt);
    sum_dxc_rhoij += try compute_spin_atom_hartree_dc(
        ctx,
        paw,
        rhoij_m,
        mt,
        sp_m_offsets,
        upf,
    );
    return .{ .dxc = dxc_m, .sum = sum_dxc_rhoij };
}

fn compute_spin_atom_dxc_and_dc(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    prij: *const paw_mod.RhoIJ,
    tabs: []const paw_mod.PawTab,
    atom_index: usize,
) !SpinAtomDxcAndDc {
    const atom = ctx.atoms[atom_index];
    const species_index = atom.species_index;
    const upf = ctx.species[species_index].upf;
    const paw = upf.paw orelse return .{ .dxc = try ctx.alloc.alloc(f64, 0), .sum = 0.0 };
    if (species_index >= tabs.len or tabs[species_index].nbeta == 0) {
        return .{ .dxc = try ctx.alloc.alloc(f64, 0), .sum = 0.0 };
    }

    const mt = prij.m_total_per_atom[atom_index];
    const sp_m_offsets = prij.m_offsets[atom_index];
    const rhoij_m = prij.values[atom_index];
    if (resources.paw_rhoij_up != null and resources.paw_rhoij_down != null) {
        return compute_polarized_atom_dxc_and_dc(
            ctx,
            resources,
            atom_index,
            paw,
            rhoij_m,
            mt,
            sp_m_offsets,
            upf,
        );
    }
    return compute_unpolarized_atom_dxc_and_dc(
        ctx,
        paw,
        rhoij_m,
        mt,
        sp_m_offsets,
        upf,
    );
}

fn build_spin_paw_dxc_results(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    prij: *const paw_mod.RhoIJ,
    tabs: []const paw_mod.PawTab,
) !struct { dxc: ?[][]f64, sum: f64 } {
    var dxc_list: std.ArrayList([]f64) = .empty;
    errdefer {
        for (dxc_list.items) |entry| ctx.alloc.free(entry);
        dxc_list.deinit(ctx.alloc);
    }
    var sum_dxc_rhoij: f64 = 0.0;
    for (ctx.atoms, 0..) |_, atom_index| {
        const dxc_atom = try compute_spin_atom_dxc_and_dc(
            ctx,
            resources,
            prij,
            tabs,
            atom_index,
        );
        sum_dxc_rhoij += dxc_atom.sum;
        if (dxc_atom.dxc.len == 0) {
            ctx.alloc.free(dxc_atom.dxc);
            try append_empty_spin_dxc(ctx.alloc, &dxc_list);
        } else {
            try dxc_list.append(ctx.alloc, dxc_atom.dxc);
        }
    }
    const dxc = if (dxc_list.items.len > 0) try dxc_list.toOwnedSlice(ctx.alloc) else null_blk: {
        dxc_list.deinit(ctx.alloc);
        break :null_blk @as(?[][]f64, null);
    };
    return .{ .dxc = dxc, .sum = sum_dxc_rhoij };
}

fn duplicate_spin_contracted_rhoij(
    alloc: std.mem.Allocator,
    prij: *paw_mod.RhoIJ,
) !?[][]f64 {
    var rhoij_list: std.ArrayList([]f64) = .empty;
    errdefer {
        for (rhoij_list.items) |entry| alloc.free(entry);
        rhoij_list.deinit(alloc);
    }
    for (0..prij.natom) |atom_index| {
        const nbeta = prij.nbeta_per_atom[atom_index];
        const copy = try alloc.alloc(f64, nbeta * nbeta);
        prij.contract_to_radial(atom_index, copy);
        try rhoij_list.append(alloc, copy);
    }
    if (rhoij_list.items.len > 0) return try rhoij_list.toOwnedSlice(alloc);
    rhoij_list.deinit(alloc);
    return null;
}

fn extract_spin_paw_results(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    energy_terms: *EnergyTerms,
) !SpinPawResults {
    if (!ctx.common.is_paw) return .{};

    var results: SpinPawResults = .{};
    errdefer results.deinit(ctx.alloc);
    if (ctx.common.paw_rhoij) |*prij| {
        if (ctx.common.paw_tabs) |tabs| {
            const dxc_result = try build_spin_paw_dxc_results(
                ctx,
                resources,
                prij,
                tabs,
            );
            results.dxc = dxc_result.dxc;
            energy_terms.paw_dxc_rhoij = -dxc_result.sum;
            energy_terms.total += energy_terms.paw_dxc_rhoij;
        }
        results.rhoij = try duplicate_spin_contracted_rhoij(ctx.alloc, prij);
    }
    if (ctx.common.paw_tabs) |tabs| {
        results.tabs = tabs;
        ctx.common.paw_tabs = null;
    }
    if (resources.apply_caches_up.len > 0) {
        if (resources.apply_caches_up[0].nonlocal_ctx) |nc| {
            results.dij = try dup_spin_per_atom_dij(ctx.alloc, nc.species, .radial);
            results.dij_m = try dup_spin_per_atom_dij(
                ctx.alloc,
                nc.species,
                .m_resolved,
            );
        }
    }
    return results;
}

fn should_compute_spin_final_wavefunctions(cfg: *const config.Config) bool {
    return cfg.relax.enabled or cfg.dfpt.enabled or cfg.scf.compute_stress or cfg.dos.enabled;
}

fn compute_spin_final_wavefunctions(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_aug_energy: *const SpinOptionalDensityPair,
) !SpinFinalWavefunctionData {
    if (!should_compute_spin_final_wavefunctions(ctx.cfg)) return .{};

    var result: SpinFinalWavefunctionData = .{};
    errdefer result.deinit(ctx.alloc);

    const wfn_up = try scf_mod.compute_final_wavefunctions_with_spin_factor(
        ctx.alloc,
        ctx.io,
        ctx.cfg.*,
        ctx.common.grid,
        ctx.common.kpoints,
        ctx.common.ionic,
        ctx.species,
        ctx.atoms,
        ctx.common.recip,
        ctx.volume_bohr,
        resources.potential_up,
        resources.kpoint_cache_up,
        resources.apply_caches_up,
        ctx.common.radial_tables,
        ctx.common.paw_tabs,
        1.0,
    );
    result.wavefunctions_up = wfn_up.wavefunctions;
    result.band_energy += wfn_up.band_energy;
    result.nonlocal_energy += wfn_up.nonlocal_energy;

    const wfn_down = try scf_mod.compute_final_wavefunctions_with_spin_factor(
        ctx.alloc,
        ctx.io,
        ctx.cfg.*,
        ctx.common.grid,
        ctx.common.kpoints,
        ctx.common.ionic,
        ctx.species,
        ctx.atoms,
        ctx.common.recip,
        ctx.volume_bohr,
        resources.potential_down,
        resources.kpoint_cache_down,
        resources.apply_caches_down,
        ctx.common.radial_tables,
        ctx.common.paw_tabs,
        1.0,
    );
    result.wavefunctions_down = wfn_down.wavefunctions;
    result.band_energy += wfn_down.band_energy;
    result.nonlocal_energy += wfn_down.nonlocal_energy;

    const pot_rho_up = rho_aug_energy.up orelse resources.rho_up;
    const pot_rho_down = rho_aug_energy.down orelse resources.rho_down;
    const vxc_spin = try xc_fields_mod.compute_xc_fields_spin(
        ctx.alloc,
        ctx.common.grid,
        pot_rho_up,
        pot_rho_down,
        ctx.common.rho_core,
        ctx.cfg.scf.use_rfft,
        ctx.cfg.scf.xc,
    );
    result.vxc_r_up = vxc_spin.vxc_up;
    result.vxc_r_down = vxc_spin.vxc_down;
    ctx.alloc.free(vxc_spin.exc);
    return result;
}

fn build_spin_scf_result(
    alloc: std.mem.Allocator,
    resources: *SpinLoopResources,
    metrics: *const SpinLoopMetrics,
    grid: Grid,
    magnetization: f64,
    rho_total: []f64,
    energy_terms: EnergyTerms,
    paw_results: SpinPawResults,
    final_data: SpinFinalWavefunctionData,
    rho_core_copy: ?[]f64,
) ScfResult {
    _ = alloc;
    return .{
        .potential = resources.take_potential_up(),
        .density = rho_total,
        .iterations = metrics.iterations,
        .converged = metrics.converged,
        .energy = energy_terms,
        .fermi_level = metrics.last_fermi_level,
        .potential_residual = metrics.last_potential_residual,
        .wavefunctions = final_data.wavefunctions_up,
        .vresid = null,
        .grid = grid,
        .density_up = resources.take_rho_up(),
        .density_down = resources.take_rho_down(),
        .potential_down = resources.take_potential_down(),
        .magnetization = magnetization,
        .wavefunctions_down = final_data.wavefunctions_down,
        .vxc_r_up = final_data.vxc_r_up,
        .vxc_r_down = final_data.vxc_r_down,
        .fermi_level_down = if (!std.math.isNan(metrics.last_fermi_level))
            metrics.last_fermi_level
        else
            0.0,
        .paw_tabs = paw_results.tabs,
        .paw_dij = paw_results.dij,
        .paw_dij_m = paw_results.dij_m,
        .paw_dxc = paw_results.dxc,
        .paw_rhoij = paw_results.rhoij,
        .rho_core = rho_core_copy,
    };
}

fn log_spin_final_result(
    ctx: *const SpinContext,
    metrics: *const SpinLoopMetrics,
    energy_terms: EnergyTerms,
) !void {
    if (!ctx.cfg.scf.quiet) {
        try log_spin_energy_summary(ctx.io, energy_terms, ctx.common.is_paw);
    }
    try ctx.common.log.write_result(
        metrics.converged,
        metrics.iterations,
        energy_terms.total,
        energy_terms.band,
        energy_terms.hartree,
        energy_terms.xc,
        energy_terms.ion_ion,
        energy_terms.psp_core,
    );
}

fn finalize_spin_scf_result(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    metrics: *const SpinLoopMetrics,
) !ScfResult {
    const magnetization = compute_spin_magnetization(
        ctx.common.grid,
        resources.rho_up,
        resources.rho_down,
    );
    if (!ctx.cfg.scf.quiet) {
        try log_spin_magnetization(ctx.io, magnetization);
    }

    const rho_total = try build_spin_total_density(
        ctx.alloc,
        resources.rho_up,
        resources.rho_down,
    );
    errdefer ctx.alloc.free(rho_total);

    var rho_aug_energy = try build_spin_energy_augmented_densities(ctx, resources);
    defer rho_aug_energy.deinit(ctx.alloc);

    var energy_terms = try build_spin_energy_terms(
        ctx,
        resources,
        metrics,
        &rho_aug_energy,
    );
    try apply_spin_paw_onsite_energy(ctx, resources, &energy_terms);

    var paw_results = try extract_spin_paw_results(ctx, resources, &energy_terms);
    errdefer paw_results.deinit(ctx.alloc);

    try log_spin_final_result(ctx, metrics, energy_terms);

    var final_data = try compute_spin_final_wavefunctions(
        ctx,
        resources,
        &rho_aug_energy,
    );
    errdefer final_data.deinit(ctx.alloc);

    const rho_core_copy = try copy_spin_rho_core(ctx.alloc, ctx.common.rho_core);
    errdefer if (rho_core_copy) |values| ctx.alloc.free(values);

    return build_spin_scf_result(
        ctx.alloc,
        resources,
        metrics,
        ctx.common.grid,
        magnetization,
        rho_total,
        energy_terms,
        paw_results,
        final_data,
        rho_core_copy,
    );
}

/// Solve eigenvalue problem for all k-points for a single spin channel.
fn solve_kpoints_for_spin(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    common: *scf_mod.ScfCommon,
    potential: hamiltonian.PotentialGrid,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    scf_iter: usize,
    shared_fft_plan: fft.Fft3dPlan,
) !SpinEigenResult {
    // For spin-polarized SCF, each channel needs enough bands to accommodate
    // magnetic splitting: up channel may have more occupied than nelec/2.
    // Add 20% extra bands + 4 minimum buffer for partial occupations.
    const solve_cfg = try init_spin_kpoint_solve_config(alloc, cfg, common, potential, scf_iter);
    defer solve_cfg.deinit(alloc);

    return compute_spin_channel_eigen_data(
        alloc,
        io,
        &cfg,
        common,
        potential,
        kpoint_cache,
        apply_caches,
        shared_fft_plan,
        &solve_cfg,
    );
}

// =========================================================================
// Spin-polarized SCF loop (nspin=2)
// =========================================================================

pub fn run_spin_polarized_loop(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    volume_bohr: f64,
    common: *scf_mod.ScfCommon,
) !ScfResult {
    const ctx = SpinContext{
        .alloc = alloc,
        .io = io,
        .cfg = &cfg,
        .species = species,
        .atoms = atoms,
        .volume_bohr = volume_bohr,
        .common = common,
    };
    const setup = try init_spin_loop_resources(&ctx);
    var resources = setup.resources;
    defer resources.deinit(alloc);

    if (!cfg.scf.quiet) {
        try log_spin_init(io, common.total_electrons, setup.magnetization);
    }

    var metrics = try run_spin_scf_loop(
        &ctx,
        &resources,
        setup.magnetization,
    );
    return finalize_spin_scf_result(&ctx, &resources, &metrics);
}
