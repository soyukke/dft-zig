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

const computeKpointEigenData = kpoints_mod.computeKpointEigenData;
const findFermiLevelSpin = kpoints_mod.findFermiLevelSpin;
const accumulateKpointDensitySmearingSpin = kpoints_mod.accumulateKpointDensitySmearingSpin;
const buildFftIndexMap = fft_grid.buildFftIndexMap;
const mixDensity = mixing.mixDensity;
const mixDensityKerker = mixing.mixDensityKerker;
const logProgress = logging.logProgress;
const logIterStart = logging.logIterStart;
const logSpinInit = logging.logSpinInit;
const logSpinMagnetization = logging.logSpinMagnetization;
const logSpinEnergySummary = logging.logSpinEnergySummary;

const KPoint = symmetry.KPoint;

/// Result from solveKpointsForSpin: eigendata and count.
const SpinEigenResult = struct {
    eigen_data: []KpointEigenData,
    filled: usize,
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

    fn initZero(alloc: std.mem.Allocator, count: usize) !SpinDensityPair {
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

    fn takeUp(self: *SpinPotentialPair) hamiltonian.PotentialGrid {
        const values = self.up;
        self.up.values = &.{};
        return values;
    }

    fn takeDown(self: *SpinPotentialPair) hamiltonian.PotentialGrid {
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
        deinitKpointCaches(alloc, self.kpoint_cache_up);
        deinitKpointCaches(alloc, self.kpoint_cache_down);
        deinitApplyCaches(alloc, self.apply_caches_up);
        deinitApplyCaches(alloc, self.apply_caches_down);
        if (self.rho_up.len > 0) alloc.free(self.rho_up);
        if (self.rho_down.len > 0) alloc.free(self.rho_down);
        self.potential_up.deinit(alloc);
        self.potential_down.deinit(alloc);
        if (self.paw_rhoij_up) |*rij| rij.deinit(alloc);
        if (self.paw_rhoij_down) |*rij| rij.deinit(alloc);
    }

    fn resetKpointCaches(self: *SpinLoopResources) void {
        for (self.kpoint_cache_up) |*cache| cache.deinit();
        for (self.kpoint_cache_up) |*cache| cache.* = .{};
        for (self.kpoint_cache_down) |*cache| cache.deinit();
        for (self.kpoint_cache_down) |*cache| cache.* = .{};
    }

    fn takePotentialUp(self: *SpinLoopResources) hamiltonian.PotentialGrid {
        const result = self.potential_up;
        self.potential_up.values = &.{};
        return result;
    }

    fn takePotentialDown(self: *SpinLoopResources) hamiltonian.PotentialGrid {
        const result = self.potential_down;
        self.potential_down.values = &.{};
        return result;
    }

    fn takeRhoUp(self: *SpinLoopResources) []f64 {
        const result = self.rho_up;
        self.rho_up = &.{};
        return result;
    }

    fn takeRhoDown(self: *SpinLoopResources) []f64 {
        const result = self.rho_down;
        self.rho_down = &.{};
        return result;
    }
};

fn initKpointCaches(alloc: std.mem.Allocator, count: usize) ![]KpointCache {
    const caches = try alloc.alloc(KpointCache, count);
    for (caches) |*cache| cache.* = .{};
    return caches;
}

fn deinitKpointCaches(alloc: std.mem.Allocator, caches: []KpointCache) void {
    for (caches) |*cache| cache.deinit();
    alloc.free(caches);
}

fn initApplyCaches(alloc: std.mem.Allocator, count: usize) ![]apply.KpointApplyCache {
    const caches = try alloc.alloc(apply.KpointApplyCache, count);
    for (caches) |*cache| cache.* = .{};
    return caches;
}

fn deinitApplyCaches(
    alloc: std.mem.Allocator,
    caches: []apply.KpointApplyCache,
) void {
    for (caches) |*cache| cache.deinit(alloc);
    alloc.free(caches);
}

fn spinEcutrho(cfg: *const config.Config) f64 {
    const gs_scf = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    return cfg.scf.ecut_ry * gs_scf * gs_scf;
}

fn computeInitialMagnetization(cfg: *const config.Config, total_electrons: f64) f64 {
    var magnetization: f64 = 0.0;
    if (cfg.scf.spinat) |values| {
        for (values) |value| magnetization += value;
    }
    return std.math.clamp(magnetization, -total_electrons, total_electrons);
}

fn initSpinDensities(
    ctx: *const SpinContext,
    rho_up: []f64,
    rho_down: []f64,
) !f64 {
    const total_electrons = ctx.common.total_electrons;
    const magnetization = computeInitialMagnetization(ctx.cfg, total_electrons);
    const atomic_rho = try scf_mod.buildAtomicDensity(
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

fn buildInitialSpinPotentials(
    ctx: *const SpinContext,
    rho_up: []const f64,
    rho_down: []const f64,
) !SpinPotentialPair {
    const spin_potentials = try potential_mod.buildPotentialGridSpin(
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

fn cloneSpinRhoij(
    alloc: std.mem.Allocator,
    common: *scf_mod.ScfCommon,
) !struct { up: ?paw_mod.RhoIJ, down: ?paw_mod.RhoIJ } {
    const up = if (common.paw_rhoij) |*prij| try prij.clone(alloc) else null;
    errdefer if (up) |*rij| rij.deinit(alloc);

    const down = if (common.paw_rhoij) |*prij| try prij.clone(alloc) else null;
    errdefer if (down) |*rij| rij.deinit(alloc);

    return .{ .up = up, .down = down };
}

fn initSpinLoopResources(ctx: *const SpinContext) !struct {
    resources: SpinLoopResources,
    magnetization: f64,
} {
    const count = ctx.common.kpoints.len;
    const grid_count = ctx.common.grid.count();
    const kpoint_cache_up = try initKpointCaches(ctx.alloc, count);
    errdefer deinitKpointCaches(ctx.alloc, kpoint_cache_up);

    const kpoint_cache_down = try initKpointCaches(ctx.alloc, count);
    errdefer deinitKpointCaches(ctx.alloc, kpoint_cache_down);

    const apply_caches_up = try initApplyCaches(ctx.alloc, count);
    errdefer deinitApplyCaches(ctx.alloc, apply_caches_up);

    const apply_caches_down = try initApplyCaches(ctx.alloc, count);
    errdefer deinitApplyCaches(ctx.alloc, apply_caches_down);

    const rho_up = try ctx.alloc.alloc(f64, grid_count);
    errdefer ctx.alloc.free(rho_up);

    const rho_down = try ctx.alloc.alloc(f64, grid_count);
    errdefer ctx.alloc.free(rho_down);

    const magnetization = try initSpinDensities(ctx, rho_up, rho_down);
    const potentials = try buildInitialSpinPotentials(ctx, rho_up, rho_down);
    errdefer {
        var pair = potentials;
        pair.deinit(ctx.alloc);
    }

    const paw_rhoij = try cloneSpinRhoij(ctx.alloc, ctx.common);
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
            .ecutrho_scf = spinEcutrho(ctx.cfg),
        },
        .magnetization = magnetization,
    };
}

fn resetSpinIterationRhoij(
    common: *scf_mod.ScfCommon,
    resources: *SpinLoopResources,
) void {
    if (common.paw_rhoij) |*prij| prij.reset();
    if (resources.paw_rhoij_up) |*rij| rij.reset();
    if (resources.paw_rhoij_down) |*rij| rij.reset();
}

fn solveSpinChannelsOnce(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    iteration: usize,
    shared_fft_plan: fft.Fft3dPlan,
) !SpinChannelResults {
    const up = try solveKpointsForSpin(
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

    const down = try solveKpointsForSpin(
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

fn bootstrapSpinPawDijIfNeeded(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    iteration: usize,
    channels: *SpinChannelResults,
    shared_fft_plan: fft.Fft3dPlan,
) !void {
    if (iteration != 0 or !ctx.common.is_paw) return;
    const tabs = ctx.common.paw_tabs orelse return;

    try paw_scf.updatePawDij(
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
    try paw_scf.updatePawDij(
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

    resources.resetKpointCaches();
    const replacement = try solveSpinChannelsOnce(
        ctx,
        resources,
        iteration,
        shared_fft_plan,
    );
    channels.deinit(ctx.alloc);
    channels.* = replacement;
}

fn solveSpinChannels(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    iteration: usize,
) !SpinChannelResults {
    var shared_fft_plan = try fft.Fft3dPlan.initWithBackend(
        ctx.alloc,
        ctx.io,
        ctx.common.grid.nx,
        ctx.common.grid.ny,
        ctx.common.grid.nz,
        ctx.cfg.scf.fft_backend,
    );
    defer shared_fft_plan.deinit(ctx.alloc);

    var channels = try solveSpinChannelsOnce(
        ctx,
        resources,
        iteration,
        shared_fft_plan,
    );
    errdefer channels.deinit(ctx.alloc);

    try bootstrapSpinPawDijIfNeeded(
        ctx,
        resources,
        iteration,
        &channels,
        shared_fft_plan,
    );
    return channels;
}

fn computeSpinFermiLevels(
    ctx: *const SpinContext,
    metrics: *SpinLoopMetrics,
    channels: *const SpinChannelResults,
) SpinFermiLevels {
    const use_fsm = ctx.common.is_paw and @abs(metrics.target_magnetization) > 0.1;
    const nelec = ctx.common.total_electrons;
    const ne_up_target = (nelec + metrics.target_magnetization) / 2.0;
    const ne_down_target = (nelec - metrics.target_magnetization) / 2.0;
    const up = if (use_fsm)
        findFermiLevelSpin(
            ne_up_target,
            ctx.cfg.scf.smear_ry,
            ctx.cfg.scf.smearing,
            channels.up.eigen_data[0..channels.up.filled],
            null,
            1.0,
        )
    else
        findFermiLevelSpin(
            nelec,
            ctx.cfg.scf.smear_ry,
            ctx.cfg.scf.smearing,
            channels.up.eigen_data[0..channels.up.filled],
            channels.down.eigen_data[0..channels.down.filled],
            1.0,
        );
    const down = if (use_fsm)
        findFermiLevelSpin(
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

fn accumulateSpinChannelDensity(
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
        try accumulateKpointDensitySmearingSpin(
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

fn combineSpinPawRhoij(
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

fn accumulateSpinIterationOutputs(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    channels: *const SpinChannelResults,
    fermi_levels: SpinFermiLevels,
    rho_out: *SpinDensityPair,
    energies: *SpinIterationEnergies,
) !void {
    const fft_index_map = try buildFftIndexMap(ctx.alloc, ctx.common.grid);
    defer ctx.alloc.free(fft_index_map);

    try accumulateSpinChannelDensity(
        ctx,
        channels.up.eigen_data[0..channels.up.filled],
        fermi_levels.up,
        rho_out.up,
        resources.apply_caches_up,
        if (resources.paw_rhoij_up) |*rij| rij else null,
        energies,
        fft_index_map,
    );
    try accumulateSpinChannelDensity(
        ctx,
        channels.down.eigen_data[0..channels.down.filled],
        fermi_levels.down,
        rho_out.down,
        resources.apply_caches_down,
        if (resources.paw_rhoij_down) |*rij| rij else null,
        energies,
        fft_index_map,
    );
    combineSpinPawRhoij(ctx.common, resources);
}

fn symmetrizeSpinOutputs(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *SpinDensityPair,
) !void {
    if (ctx.common.sym_ops) |ops| {
        if (ops.len > 1) {
            try scf_mod.symmetrizeDensity(
                ctx.alloc,
                ctx.common.grid,
                rho_out.up,
                ops,
                ctx.cfg.scf.use_rfft,
            );
            try scf_mod.symmetrizeDensity(
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
            try paw_scf.symmetrizeRhoIJ(ctx.alloc, prij, ctx.species, ctx.atoms);
        }
    }
    _ = resources;
}

fn filterSpinDensityInPlace(
    ctx: *const SpinContext,
    density: []f64,
    ecutrho_scf: f64,
) !void {
    const filtered = try potential_mod.filterDensityToEcutrho(
        ctx.alloc,
        ctx.common.grid,
        density,
        ecutrho_scf,
        ctx.cfg.scf.use_rfft,
    );
    defer ctx.alloc.free(filtered);

    @memcpy(density, filtered);
}

fn buildSpinCompensatedDensityPair(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_up: []const f64,
    rho_down: []const f64,
) !SpinDensityPair {
    const grid_count = ctx.common.grid.count();
    const n_hat = try ctx.alloc.alloc(f64, grid_count);
    defer ctx.alloc.free(n_hat);

    @memset(n_hat, 0.0);
    try paw_scf.addPawCompensationCharge(
        ctx.alloc,
        ctx.common.grid,
        n_hat,
        &ctx.common.paw_rhoij.?,
        ctx.common.paw_tabs.?,
        ctx.atoms,
        resources.ecutrho_scf,
        &ctx.common.paw_gaunt.?,
    );

    var augmented = try SpinDensityPair.initZero(ctx.alloc, grid_count);
    for (0..grid_count) |index| {
        augmented.up[index] = rho_up[index] + n_hat[index] * 0.5;
        augmented.down[index] = rho_down[index] + n_hat[index] * 0.5;
    }
    return augmented;
}

fn buildSpinPotentialInputs(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *SpinDensityPair,
) !SpinOptionalDensityPair {
    try symmetrizeSpinOutputs(ctx, resources, rho_out);
    if (!ctx.common.is_paw) return .{};

    try filterSpinDensityInPlace(ctx, rho_out.up, resources.ecutrho_scf);
    try filterSpinDensityInPlace(ctx, rho_out.down, resources.ecutrho_scf);
    if (ctx.common.paw_rhoij == null) return .{};

    const augmented = try buildSpinCompensatedDensityPair(
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

fn buildSpinOutputPotentials(
    ctx: *const SpinContext,
    rho_out: *const SpinDensityPair,
    rho_aug: *const SpinOptionalDensityPair,
) !SpinPotentialPair {
    const rho_up = rho_aug.up orelse rho_out.up;
    const rho_down = rho_aug.down orelse rho_out.down;
    return buildInitialSpinPotentials(ctx, rho_up, rho_down);
}

fn computeSpinPotentialResidual(
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

fn computeSpinIterationConvergence(
    ctx: *const SpinContext,
    resources: *const SpinLoopResources,
    rho_out: *const SpinDensityPair,
    pot_out: *const SpinPotentialPair,
) SpinIterationConvergence {
    const residual = computeSpinPotentialResidual(resources, pot_out);
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

fn logSpinIterationProgress(
    ctx: *const SpinContext,
    metrics: *const SpinLoopMetrics,
    convergence: SpinIterationConvergence,
) !void {
    try ctx.common.log.writeIter(
        metrics.iterations,
        convergence.diff,
        convergence.residual,
        metrics.last_band_energy,
        metrics.last_nonlocal_energy,
    );
    if (!ctx.cfg.scf.quiet) {
        try logProgress(
            ctx.io,
            metrics.iterations,
            convergence.diff,
            convergence.residual,
            metrics.last_band_energy,
            metrics.last_nonlocal_energy,
        );
    }
}

fn acceptSpinConvergedStep(
    alloc: std.mem.Allocator,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    pot_out: *SpinPotentialPair,
) void {
    @memcpy(resources.rho_up, rho_out.up);
    @memcpy(resources.rho_down, rho_out.down);
    resources.potential_up.deinit(alloc);
    resources.potential_up = pot_out.takeUp();
    resources.potential_down.deinit(alloc);
    resources.potential_down = pot_out.takeDown();
}

fn applySpinPotentialPulay(
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
        mixing.applyModelDielectricPreconditioner(
            ctx.common.grid,
            res_up_c,
            ctx.cfg.scf.diemac,
            ctx.cfg.scf.dielng,
        );
        mixing.applyModelDielectricPreconditioner(
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
    try ctx.common.pulay_mixer.?.mixWithResidual(
        concat_in,
        residual_concat,
        ctx.cfg.scf.mixing_beta,
    );
    @memcpy(v_in_up, concat_in[0..n_f64]);
    @memcpy(v_in_down, concat_in[n_f64..]);
    _ = resources;
}

fn applySpinPotentialMixing(
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
        try applySpinPotentialPulay(
            ctx,
            resources,
            pot_out,
            n_complex,
            n_f64,
            v_in_up,
            v_in_down,
        );
    } else {
        mixDensity(v_in_up, v_out_up_f, ctx.cfg.scf.mixing_beta);
        mixDensity(v_in_down, v_out_down_f, ctx.cfg.scf.mixing_beta);
    }
    @memcpy(resources.rho_up, rho_out.up);
    @memcpy(resources.rho_down, rho_out.down);
    pot_out.deinit(ctx.alloc);
}

fn mixSpinDensityInputs(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    iteration: usize,
) !void {
    const force_density_mixing = false;
    if (force_density_mixing) {
        const paw_beta: f64 = 0.05;
        const paw_q0: f64 = 1.5;
        try mixDensityKerker(
            ctx.alloc,
            ctx.common.grid,
            resources.rho_up,
            rho_out.up,
            paw_beta,
            paw_q0,
            ctx.cfg.scf.use_rfft,
        );
        try mixDensityKerker(
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
    mixDensity(resources.rho_up, rho_out.up, ctx.cfg.scf.mixing_beta);
    mixDensity(resources.rho_down, rho_out.down, ctx.cfg.scf.mixing_beta);
}

fn buildSpinMixedDensityAugmentation(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
) !SpinOptionalDensityPair {
    if (!ctx.common.is_paw or ctx.common.paw_rhoij == null) return .{};
    const augmented = try buildSpinCompensatedDensityPair(
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

fn rebuildSpinPotentialsFromMixedDensity(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
) !void {
    var augmented = try buildSpinMixedDensityAugmentation(ctx, resources);
    defer augmented.deinit(ctx.alloc);

    const rho_up = augmented.up orelse resources.rho_up;
    const rho_down = augmented.down orelse resources.rho_down;
    const rebuilt = try buildInitialSpinPotentials(ctx, rho_up, rho_down);
    resources.potential_up = rebuilt.up;
    resources.potential_down = rebuilt.down;
}

fn applySpinDensityMixing(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    iteration: usize,
    pot_out: *SpinPotentialPair,
) !void {
    pot_out.deinit(ctx.alloc);
    resources.potential_up.deinit(ctx.alloc);
    resources.potential_down.deinit(ctx.alloc);
    try mixSpinDensityInputs(ctx, resources, rho_out, iteration);
    try rebuildSpinPotentialsFromMixedDensity(ctx, resources);
}

fn mixSpinIterationState(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_out: *const SpinDensityPair,
    iteration: usize,
    pot_out: *SpinPotentialPair,
) !void {
    const force_density_mixing = false;
    if (ctx.cfg.scf.mixing_mode == .potential and !force_density_mixing) {
        try applySpinPotentialMixing(
            ctx,
            resources,
            rho_out,
            iteration,
            pot_out,
        );
        return;
    }
    try applySpinDensityMixing(ctx, resources, rho_out, iteration, pot_out);
}

fn updateOneSpinPawDij(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    potential: hamiltonian.PotentialGrid,
    apply_caches: []apply.KpointApplyCache,
    rhoij: ?*paw_mod.RhoIJ,
    tabs: []paw_mod.PawTab,
    mix_beta: f64,
) !void {
    try paw_scf.updatePawDij(
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

fn updateSpinPawDijIfNeeded(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
) !void {
    if (!ctx.common.is_paw) return;
    const tabs = ctx.common.paw_tabs orelse return;
    const mix_beta = ctx.cfg.scf.mixing_beta;
    try updateOneSpinPawDij(
        ctx,
        resources,
        resources.potential_up,
        resources.apply_caches_up,
        if (resources.paw_rhoij_up) |*rij| rij else null,
        tabs,
        mix_beta,
    );
    try updateOneSpinPawDij(
        ctx,
        resources,
        resources.potential_down,
        resources.apply_caches_down,
        if (resources.paw_rhoij_down) |*rij| rij else null,
        tabs,
        mix_beta,
    );
}

fn runSpinIteration(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    metrics: *SpinLoopMetrics,
) !bool {
    if (!ctx.cfg.scf.quiet) {
        try logIterStart(ctx.io, metrics.iterations);
    }
    resetSpinIterationRhoij(ctx.common, resources);

    var rho_out = try SpinDensityPair.initZero(ctx.alloc, ctx.common.grid.count());
    defer rho_out.deinit(ctx.alloc);

    var channels = try solveSpinChannels(ctx, resources, metrics.iterations);
    defer channels.deinit(ctx.alloc);

    const fermi_levels = computeSpinFermiLevels(ctx, metrics, &channels);
    var energies: SpinIterationEnergies = .{};
    try accumulateSpinIterationOutputs(
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

    var rho_aug = try buildSpinPotentialInputs(ctx, resources, &rho_out);
    defer rho_aug.deinit(ctx.alloc);

    var pot_out = try buildSpinOutputPotentials(ctx, &rho_out, &rho_aug);
    var pot_out_owned = true;
    defer if (pot_out_owned) pot_out.deinit(ctx.alloc);

    const convergence = computeSpinIterationConvergence(
        ctx,
        resources,
        &rho_out,
        &pot_out,
    );
    metrics.last_potential_residual = convergence.residual;
    try logSpinIterationProgress(ctx, metrics, convergence);

    if (convergence.conv_value < ctx.cfg.scf.convergence) {
        acceptSpinConvergedStep(ctx.alloc, resources, &rho_out, &pot_out);
        pot_out_owned = false;
        return true;
    }

    try mixSpinIterationState(
        ctx,
        resources,
        &rho_out,
        metrics.iterations,
        &pot_out,
    );
    pot_out_owned = false;
    try updateSpinPawDijIfNeeded(ctx, resources);
    return false;
}

fn runSpinScfLoop(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    target_magnetization: f64,
) !SpinLoopMetrics {
    var metrics = SpinLoopMetrics{
        .target_magnetization = target_magnetization,
    };
    while (metrics.iterations < ctx.cfg.scf.max_iter) : (metrics.iterations += 1) {
        if (try runSpinIteration(ctx, resources, &metrics)) {
            metrics.converged = true;
            break;
        }
    }
    return metrics;
}

fn computeSpinMagnetization(
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

fn buildSpinTotalDensity(
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

fn buildSpinEnergyAugmentedDensities(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
) !SpinOptionalDensityPair {
    if (!ctx.common.is_paw or ctx.common.paw_rhoij == null) return .{};

    var augmented = try buildSpinCompensatedDensityPair(
        ctx,
        resources,
        resources.rho_up,
        resources.rho_down,
    );
    errdefer augmented.deinit(ctx.alloc);

    try filterSpinDensityInPlace(ctx, augmented.up, resources.ecutrho_scf);
    try filterSpinDensityInPlace(ctx, augmented.down, resources.ecutrho_scf);
    return .{
        .up = augmented.up,
        .down = augmented.down,
    };
}

fn buildSpinEnergyTerms(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    metrics: *const SpinLoopMetrics,
    rho_aug_energy: *const SpinOptionalDensityPair,
) !EnergyTerms {
    const paw_ecutrho: ?f64 = if (ctx.common.is_paw) resources.ecutrho_scf else null;
    return energy_mod.computeEnergyTermsSpin(.{
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

fn applySpinPawOnsiteEnergy(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    energy_terms: *EnergyTerms,
) !void {
    if (!ctx.common.is_paw) return;
    const prij = &ctx.common.paw_rhoij.?;
    const tabs = ctx.common.paw_tabs orelse return;
    energy_terms.paw_onsite = try paw_scf.computePawOnsiteEnergyTotal(
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

fn copySpinRhoCore(
    alloc: std.mem.Allocator,
    rho_core: ?[]const f64,
) !?[]f64 {
    const values = rho_core orelse return null;
    const copy = try alloc.alloc(f64, values.len);
    @memcpy(copy, values);
    return copy;
}

fn dupSpinPerAtomDij(
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

fn appendEmptySpinDxc(
    alloc: std.mem.Allocator,
    dxc_list: *std.ArrayList([]f64),
) !void {
    try dxc_list.append(alloc, try alloc.alloc(f64, 0));
}

fn accumulateSpinMatrixTrace(
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

fn computeSpinAtomHartreeDc(
    ctx: *const SpinContext,
    paw: anytype,
    rhoij_m: []const f64,
    mt: usize,
    sp_m_offsets: anytype,
    upf: anytype,
) !f64 {
    const dij_h_dc = try ctx.alloc.alloc(f64, mt * mt);
    defer ctx.alloc.free(dij_h_dc);

    try paw_mod.paw_xc.computePawDijHartreeMultiL(
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
    accumulateSpinMatrixTrace(&sum_dxc_rhoij, dij_h_dc, rhoij_m, mt);
    return sum_dxc_rhoij;
}

fn computePolarizedAtomDxcAndDc(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    atom_index: usize,
    paw: anytype,
    rhoij_m: []const f64,
    mt: usize,
    sp_m_offsets: anytype,
    upf: anytype,
) !struct { dxc: []f64, sum: f64 } {
    const dxc_up = try ctx.alloc.alloc(f64, mt * mt);
    errdefer ctx.alloc.free(dxc_up);

    const dxc_down = try ctx.alloc.alloc(f64, mt * mt);
    defer ctx.alloc.free(dxc_down);

    try paw_mod.paw_xc.computePawDijXcAngularSpin(
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
    accumulateSpinMatrixTrace(
        &sum_dxc_rhoij,
        dxc_up,
        resources.paw_rhoij_up.?.values[atom_index],
        mt,
    );
    accumulateSpinMatrixTrace(
        &sum_dxc_rhoij,
        dxc_down,
        resources.paw_rhoij_down.?.values[atom_index],
        mt,
    );
    sum_dxc_rhoij += try computeSpinAtomHartreeDc(
        ctx,
        paw,
        rhoij_m,
        mt,
        sp_m_offsets,
        upf,
    );
    return .{ .dxc = dxc_up, .sum = sum_dxc_rhoij };
}

fn computeUnpolarizedAtomDxcAndDc(
    ctx: *const SpinContext,
    paw: anytype,
    rhoij_m: []const f64,
    mt: usize,
    sp_m_offsets: anytype,
    upf: anytype,
) !struct { dxc: []f64, sum: f64 } {
    const dxc_m = try ctx.alloc.alloc(f64, mt * mt);
    errdefer ctx.alloc.free(dxc_m);

    try paw_mod.paw_xc.computePawDijXcAngular(
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
    accumulateSpinMatrixTrace(&sum_dxc_rhoij, dxc_m, rhoij_m, mt);
    sum_dxc_rhoij += try computeSpinAtomHartreeDc(
        ctx,
        paw,
        rhoij_m,
        mt,
        sp_m_offsets,
        upf,
    );
    return .{ .dxc = dxc_m, .sum = sum_dxc_rhoij };
}

fn computeSpinAtomDxcAndDc(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    prij: *const paw_mod.RhoIJ,
    tabs: []const paw_mod.PawTab,
    atom_index: usize,
) !struct { dxc: []f64, sum: f64 } {
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
        return computePolarizedAtomDxcAndDc(
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
    return computeUnpolarizedAtomDxcAndDc(
        ctx,
        paw,
        rhoij_m,
        mt,
        sp_m_offsets,
        upf,
    );
}

fn buildSpinPawDxcResults(
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
        const dxc_atom = try computeSpinAtomDxcAndDc(
            ctx,
            resources,
            prij,
            tabs,
            atom_index,
        );
        sum_dxc_rhoij += dxc_atom.sum;
        if (dxc_atom.dxc.len == 0) {
            ctx.alloc.free(dxc_atom.dxc);
            try appendEmptySpinDxc(ctx.alloc, &dxc_list);
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

fn duplicateSpinContractedRhoij(
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
        prij.contractToRadial(atom_index, copy);
        try rhoij_list.append(alloc, copy);
    }
    if (rhoij_list.items.len > 0) return try rhoij_list.toOwnedSlice(alloc);
    rhoij_list.deinit(alloc);
    return null;
}

fn extractSpinPawResults(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    energy_terms: *EnergyTerms,
) !SpinPawResults {
    if (!ctx.common.is_paw) return .{};

    var results: SpinPawResults = .{};
    errdefer results.deinit(ctx.alloc);
    if (ctx.common.paw_rhoij) |*prij| {
        if (ctx.common.paw_tabs) |tabs| {
            const dxc_result = try buildSpinPawDxcResults(
                ctx,
                resources,
                prij,
                tabs,
            );
            results.dxc = dxc_result.dxc;
            energy_terms.paw_dxc_rhoij = -dxc_result.sum;
            energy_terms.total += energy_terms.paw_dxc_rhoij;
        }
        results.rhoij = try duplicateSpinContractedRhoij(ctx.alloc, prij);
    }
    if (ctx.common.paw_tabs) |tabs| {
        results.tabs = tabs;
        ctx.common.paw_tabs = null;
    }
    if (resources.apply_caches_up.len > 0) {
        if (resources.apply_caches_up[0].nonlocal_ctx) |nc| {
            results.dij = try dupSpinPerAtomDij(ctx.alloc, nc.species, .radial);
            results.dij_m = try dupSpinPerAtomDij(
                ctx.alloc,
                nc.species,
                .m_resolved,
            );
        }
    }
    return results;
}

fn shouldComputeSpinFinalWavefunctions(cfg: *const config.Config) bool {
    return cfg.relax.enabled or cfg.dfpt.enabled or cfg.scf.compute_stress or cfg.dos.enabled;
}

fn computeSpinFinalWavefunctions(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    rho_aug_energy: *const SpinOptionalDensityPair,
) !SpinFinalWavefunctionData {
    if (!shouldComputeSpinFinalWavefunctions(ctx.cfg)) return .{};

    var result: SpinFinalWavefunctionData = .{};
    errdefer result.deinit(ctx.alloc);

    const wfn_up = try scf_mod.computeFinalWavefunctionsWithSpinFactor(
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

    const wfn_down = try scf_mod.computeFinalWavefunctionsWithSpinFactor(
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
    const vxc_spin = try xc_fields_mod.computeXcFieldsSpin(
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

fn buildSpinScfResult(
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
        .potential = resources.takePotentialUp(),
        .density = rho_total,
        .iterations = metrics.iterations,
        .converged = metrics.converged,
        .energy = energy_terms,
        .fermi_level = metrics.last_fermi_level,
        .potential_residual = metrics.last_potential_residual,
        .wavefunctions = final_data.wavefunctions_up,
        .vresid = null,
        .grid = grid,
        .density_up = resources.takeRhoUp(),
        .density_down = resources.takeRhoDown(),
        .potential_down = resources.takePotentialDown(),
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

fn finalizeSpinScfResult(
    ctx: *const SpinContext,
    resources: *SpinLoopResources,
    metrics: *const SpinLoopMetrics,
) !ScfResult {
    const magnetization = computeSpinMagnetization(
        ctx.common.grid,
        resources.rho_up,
        resources.rho_down,
    );
    if (!ctx.cfg.scf.quiet) {
        try logSpinMagnetization(ctx.io, magnetization);
    }

    const rho_total = try buildSpinTotalDensity(
        ctx.alloc,
        resources.rho_up,
        resources.rho_down,
    );
    errdefer ctx.alloc.free(rho_total);

    var rho_aug_energy = try buildSpinEnergyAugmentedDensities(ctx, resources);
    defer rho_aug_energy.deinit(ctx.alloc);

    var energy_terms = try buildSpinEnergyTerms(
        ctx,
        resources,
        metrics,
        &rho_aug_energy,
    );
    try applySpinPawOnsiteEnergy(ctx, resources, &energy_terms);

    var paw_results = try extractSpinPawResults(ctx, resources, &energy_terms);
    errdefer paw_results.deinit(ctx.alloc);

    if (!ctx.cfg.scf.quiet) {
        try logSpinEnergySummary(ctx.io, energy_terms, ctx.common.is_paw);
    }
    try ctx.common.log.writeResult(
        metrics.converged,
        metrics.iterations,
        energy_terms.total,
        energy_terms.band,
        energy_terms.hartree,
        energy_terms.xc,
        energy_terms.ion_ion,
        energy_terms.psp_core,
    );

    var final_data = try computeSpinFinalWavefunctions(
        ctx,
        resources,
        &rho_aug_energy,
    );
    errdefer final_data.deinit(ctx.alloc);

    const rho_core_copy = try copySpinRhoCore(ctx.alloc, ctx.common.rho_core);
    errdefer if (rho_core_copy) |values| ctx.alloc.free(values);

    return buildSpinScfResult(
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
fn solveKpointsForSpin(
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
    const grid = common.grid;
    const kpoints = common.kpoints;
    const species = common.species;
    const atoms = common.atoms;
    const recip = common.recip;
    const volume_bohr = common.volume_bohr;
    const radial_tables = common.radial_tables;

    // For spin-polarized SCF, each channel needs enough bands to accommodate
    // magnetic splitting: up channel may have more occupied than nelec/2.
    // Add 20% extra bands + 4 minimum buffer for partial occupations.
    const nocc_base = @as(usize, @intFromFloat(std.math.ceil(common.total_electrons / 2.0)));
    const nocc = nocc_base + @max(4, nocc_base / 5);
    const is_paw_spin = scf_mod.hasPaw(species);
    const has_qij = scf_mod.hasQij(species) and !is_paw_spin;
    const use_iterative_config = (cfg.scf.solver == .iterative or
        cfg.scf.solver == .cg or
        cfg.scf.solver == .auto) and !has_qij;
    const nonlocal_enabled = cfg.scf.enable_nonlocal and scf_mod.hasNonlocal(species);

    var local_r: ?[]f64 = null;
    if (use_iterative_config) {
        local_r = try potential_mod.buildLocalPotentialReal(alloc, grid, common.ionic, potential);
    }
    defer if (local_r) |values| alloc.free(values);

    const fft_index_map = try buildFftIndexMap(alloc, grid);
    defer alloc.free(fft_index_map);

    var iter_max_iter = cfg.scf.iterative_max_iter;
    var iter_tol = cfg.scf.iterative_tol;
    if (cfg.scf.iterative_warmup_steps > 0 and scf_iter < cfg.scf.iterative_warmup_steps) {
        iter_max_iter = cfg.scf.iterative_warmup_max_iter;
        iter_tol = cfg.scf.iterative_warmup_tol;
    }

    const eigen_data = try alloc.alloc(KpointEigenData, kpoints.len);
    var filled: usize = 0;
    errdefer {
        var ii: usize = 0;
        while (ii < filled) : (ii += 1) {
            eigen_data[ii].deinit(alloc);
        }
        alloc.free(eigen_data);
    }

    for (kpoints, 0..) |kp, kidx| {
        const ac_ptr: ?*apply.KpointApplyCache = &apply_caches[kidx];
        eigen_data[kidx] = try computeKpointEigenData(
            alloc,
            io,
            &cfg,
            grid,
            kp,
            species,
            atoms,
            recip,
            volume_bohr,
            common.local_cfg,
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
            null,
            shared_fft_plan,
            ac_ptr,
            radial_tables,
            common.paw_tabs,
        );
        filled += 1;
    }

    return SpinEigenResult{
        .eigen_data = eigen_data,
        .filled = filled,
    };
}

// =========================================================================
// Spin-polarized SCF loop (nspin=2)
// =========================================================================

pub fn runSpinPolarizedLoop(
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
    const setup = try initSpinLoopResources(&ctx);
    var resources = setup.resources;
    defer resources.deinit(alloc);

    if (!cfg.scf.quiet) {
        try logSpinInit(io, common.total_electrons, setup.magnetization);
    }

    var metrics = try runSpinScfLoop(
        &ctx,
        &resources,
        setup.magnetization,
    );
    return finalizeSpinScfResult(&ctx, &resources, &metrics);
}
