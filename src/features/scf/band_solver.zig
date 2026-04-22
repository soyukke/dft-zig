const std = @import("std");
const apply = @import("apply.zig");
const config = @import("../config/config.zig");
const fft = @import("../fft/fft.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const iterative = @import("../linalg/iterative.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const grid_requirements = @import("../plane_wave/grid_requirements.zig");
const logging = @import("logging.zig");
const potential_mod = @import("potential.zig");
const pw_grid_map = @import("pw_grid_map.zig");
const thread_pool = @import("../thread_pool.zig");
const util = @import("util.zig");

const Grid = grid_mod.Grid;
const ApplyContext = apply.ApplyContext;
const PwGridMap = pw_grid_map.PwGridMap;
const ThreadPool = thread_pool.ThreadPool;

const gridFromConfig = grid_mod.gridFromConfig;
const gridRequirement = grid_requirements.gridRequirement;
const hasNonlocal = util.hasNonlocal;
const buildFftIndexMap = fft_grid.buildFftIndexMap;
const applyHamiltonian = apply.applyHamiltonian;
const applyHamiltonianBatched = apply.applyHamiltonianBatched;
const logBandLocalPotentialMean = logging.logBandLocalPotentialMean;

pub const BandIterativeContext = struct {
    grid: Grid,
    local_r: []f64,
    fft_index_map: ?[]usize,
    inv_volume: f64,
    /// PAW tables for band calculation (borrowed, NOT owned)
    paw_tabs: ?[]const paw_mod.PawTab = null,
    /// Per-atom converged D_ij from SCF: [natom][nbeta*nbeta] (owned)
    paw_dij: ?[][]f64 = null,

    pub fn deinit(self: *BandIterativeContext, alloc: std.mem.Allocator) void {
        if (self.local_r.len > 0) {
            alloc.free(self.local_r);
        }
        if (self.fft_index_map) |map| {
            alloc.free(map);
        }
        if (self.paw_dij) |dij| {
            for (dij) |d| alloc.free(d);
            alloc.free(dij);
        }
    }
};

pub const BandVectorCache = struct {
    n: usize = 0,
    nbands: usize = 0,
    vectors: []math.Complex = &[_]math.Complex{},

    pub fn deinit(self: *BandVectorCache) void {
        if (self.vectors.len > 0) {
            std.heap.c_allocator.free(self.vectors);
        }
        self.* = .{};
    }

    pub fn store(
        self: *BandVectorCache,
        n: usize,
        nbands: usize,
        values: []const math.Complex,
    ) !void {
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
};

pub fn initBandIterativeContext(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    extra: hamiltonian.PotentialGrid,
) !BandIterativeContext {
    const grid = gridFromConfig(cfg, recip, volume);
    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, grid.cell);
    if (extra.nx != grid.nx or extra.ny != grid.ny or extra.nz != grid.nz or
        extra.min_h != grid.min_h or extra.min_k != grid.min_k or extra.min_l != grid.min_l)
    {
        return error.InvalidGrid;
    }
    var ionic = try potential_mod.buildIonicPotentialGrid(
        alloc,
        grid,
        species,
        atoms,
        local_cfg,
        null,
        null,
    );
    defer ionic.deinit(alloc);
    const local_r = try potential_mod.buildLocalPotentialReal(alloc, grid, ionic, extra);

    if (cfg.scf.debug_fermi) {
        var sum: f64 = 0.0;
        for (local_r) |value| {
            sum += value;
        }
        const mean_local = sum / @as(f64, @floatFromInt(local_r.len));
        const ionic_g0 = ionic.valueAt(0, 0, 0);
        const extra_g0 = extra.valueAt(0, 0, 0);
        try logBandLocalPotentialMean(io, mean_local, ionic_g0.r, extra_g0.r);
    }

    const fft_index_map = try buildFftIndexMap(alloc, grid);
    return BandIterativeContext{
        .grid = grid,
        .local_r = local_r,
        .fft_index_map = fft_index_map,
        .inv_volume = 1.0 / volume,
    };
}

/// Options for band eigenvalue calculation
pub const BandEigenOptions = struct {
    reuse_vectors: bool = true,
    pool: ?*ThreadPool = null,
    shared_fft_plan: ?fft.Fft3dPlan = null,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet = null,
    paw_tabs: ?[]const paw_mod.PawTab = null,
    paw_dij: ?[]const []const f64 = null,
};

pub fn bandEigenvaluesIterative(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    ctx: *const BandIterativeContext,
    k_cart: math.Vec3,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    nbands: usize,
    reuse_vectors: bool,
    cache: ?*BandVectorCache,
) ![]f64 {
    return bandEigenvaluesIterativeExt(
        alloc,
        io,
        cfg,
        ctx,
        k_cart,
        species,
        atoms,
        recip,
        nbands,
        cache,
        .{
            .reuse_vectors = reuse_vectors,
            .pool = null,
        },
    );
}

/// Extended band eigenvalue calculation with optional thread pool for parallel LOBPCG
pub fn bandEigenvaluesIterativeExt(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    ctx: *const BandIterativeContext,
    k_cart: math.Vec3,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    nbands: usize,
    cache: ?*BandVectorCache,
    opts: BandEigenOptions,
) ![]f64 {
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, k_cart);
    defer basis.deinit(alloc);
    if (basis.gvecs.len == 0) return error.NoPlaneWaves;

    const nbands_use = @min(nbands, basis.gvecs.len);
    if (nbands_use == 0) return error.InvalidBandConfig;

    const req = gridRequirement(basis.gvecs);
    if (req.nx > ctx.grid.nx or req.ny > ctx.grid.ny or req.nz > ctx.grid.nz) {
        return error.InvalidGrid;
    }

    const diag = try alloc.alloc(f64, basis.gvecs.len);
    defer alloc.free(diag);
    for (basis.gvecs, 0..) |g, i| {
        diag[i] = g.kinetic;
    }

    var init_vectors: ?[]const math.Complex = null;
    var init_cols: usize = 0;
    if (opts.reuse_vectors) {
        if (cache) |c| {
            if (c.vectors.len > 0 and c.n == basis.gvecs.len and c.nbands == nbands_use) {
                init_vectors = c.vectors;
                init_cols = c.nbands;
            }
        }
    }

    const nonlocal_enabled = cfg.scf.enable_nonlocal and hasNonlocal(species);
    const has_paw = opts.paw_tabs != null and opts.paw_dij != null;
    // Use multiple workspaces for parallel LOBPCG to avoid allocation during apply
    const num_workspaces: usize = if (opts.pool != null) @min(16, nbands_use + 4) else 1;

    // Build NonlocalContext:
    // - For PAW: always build ourselves with buildNonlocalContextPaw
    //   (sets up D_ij buffer + overlap)
    // - For NC with radial tables: use buildNonlocalContextWithTables (fast path)
    // - Otherwise: let ApplyContext build it internally
    var owned_nonlocal_ctx: ?apply.NonlocalContext = null;
    const build_nonlocal_ourselves = has_paw or
        (opts.radial_tables != null and nonlocal_enabled and num_workspaces <= 1);
    if (has_paw) {
        owned_nonlocal_ctx = try apply.buildNonlocalContextPaw(
            alloc,
            species,
            basis.gvecs,
            opts.radial_tables,
            opts.paw_tabs.?,
        );
    } else if (opts.radial_tables != null and nonlocal_enabled and num_workspaces <= 1) {
        owned_nonlocal_ctx = try apply.buildNonlocalContextWithTables(
            alloc,
            species,
            basis.gvecs,
            opts.radial_tables.?,
        );
    }
    errdefer if (owned_nonlocal_ctx) |*nc| nc.deinit(alloc);

    // Copy per-atom D_ij into the NonlocalContext for PAW
    if (has_paw) {
        if (owned_nonlocal_ctx) |*nc| {
            const paw_dij = opts.paw_dij.?;
            var atom_idx: usize = 0;
            for (species, 0..) |_, si| {
                var natom_for_species: usize = 0;
                for (atoms) |atom| {
                    if (atom.species_index == si) natom_for_species += 1;
                }
                if (natom_for_species > 0) {
                    try nc.ensureDijPerAtom(alloc, si, natom_for_species);
                    var a_of_s: usize = 0;
                    for (atoms) |atom| {
                        if (atom.species_index != si) continue;
                        if (atom_idx < paw_dij.len) {
                            nc.updateDijAtom(si, a_of_s, paw_dij[atom_idx]);
                        }
                        a_of_s += 1;
                        atom_idx += 1;
                    }
                }
            }
        }
    }

    // Build ApplyContext:
    // When we built NonlocalContext ourselves, use initWithCache path
    // Otherwise fall back to initWithFftPlan or initWithWorkspaces
    var apply_ctx = if (build_nonlocal_ourselves) blk: {
        var map = try PwGridMap.init(alloc, basis.gvecs, ctx.grid);
        errdefer map.deinit(alloc);
        if (ctx.fft_index_map) |idx_map| {
            try map.buildFftIndices(alloc, idx_map);
        }
        const fft_plan = if (opts.shared_fft_plan) |plan|
            plan
        else
            try fft.Fft3dPlan.initWithBackend(
                alloc,
                io,
                ctx.grid.nx,
                ctx.grid.ny,
                ctx.grid.nz,
                cfg.scf.fft_backend,
            );
        const owns_plan = opts.shared_fft_plan == null;
        var actx = try ApplyContext.initWithCache(
            alloc,
            io,
            ctx.grid,
            basis.gvecs,
            ctx.local_r,
            null,
            owned_nonlocal_ctx,
            map,
            atoms,
            ctx.inv_volume,
            null,
            ctx.fft_index_map,
            fft_plan,
            owns_plan,
        );
        actx.owns_nonlocal = true;
        actx.owns_map = true;
        owned_nonlocal_ctx = null; // ownership transferred
        break :blk actx;
    } else if (opts.shared_fft_plan != null and num_workspaces <= 1)
        try ApplyContext.initWithFftPlan(
            alloc,
            io,
            ctx.grid,
            basis.gvecs,
            ctx.local_r,
            null,
            species,
            atoms,
            ctx.inv_volume,
            nonlocal_enabled,
            null,
            ctx.fft_index_map,
            opts.shared_fft_plan.?,
        )
    else
        try ApplyContext.initWithWorkspaces(
            alloc,
            io,
            ctx.grid,
            basis.gvecs,
            ctx.local_r,
            null,
            species,
            atoms,
            ctx.inv_volume,
            nonlocal_enabled,
            null,
            ctx.fft_index_map,
            cfg.scf.fft_backend,
            num_workspaces,
        );
    defer apply_ctx.deinit(alloc);

    // For PAW, set the overlap operator S for generalized eigenvalue problem H|ψ> = ε S|ψ>
    const apply_s_fn: ?*const fn (*anyopaque, []const math.Complex, []math.Complex) anyerror!void =
        if (has_paw and apply_ctx.nonlocal_ctx != null and apply_ctx.nonlocal_ctx.?.has_paw)
            apply.applyOverlap
        else
            null;
    const op = iterative.Operator{
        .n = basis.gvecs.len,
        .ctx = &apply_ctx,
        .apply = applyHamiltonian,
        .apply_batch = applyHamiltonianBatched,
        .apply_s = apply_s_fn,
    };
    const lobpcg_opts = iterative.Options{
        .max_iter = cfg.band.iterative_max_iter,
        .tol = cfg.band.iterative_tol,
        .max_subspace = cfg.band.iterative_max_subspace,
        .block_size = cfg.band.iterative_block_size,
        .init_diagonal = cfg.band.iterative_init_diagonal,
        .init_vectors = init_vectors,
        .init_vectors_cols = init_cols,
    };

    // Select solver: CG or LOBPCG (serial/parallel)
    var eig = if (cfg.band.solver == .cg)
        try iterative.hermitianEigenDecompCG(alloc, op, diag, nbands_use, lobpcg_opts)
    else if (opts.pool) |pool|
        try iterative.hermitianEigenDecompIterativeExt(
            alloc,
            cfg.linalg_backend,
            op,
            diag,
            nbands_use,
            .{
                .base = lobpcg_opts,
                .lobpcg_backend = .parallel,
                .pool = pool,
            },
        )
    else
        try iterative.hermitianEigenDecompIterative(
            alloc,
            cfg.linalg_backend,
            op,
            diag,
            nbands_use,
            lobpcg_opts,
        );
    defer eig.deinit(alloc);

    if (opts.reuse_vectors) {
        if (cache) |c| {
            try c.store(basis.gvecs.len, nbands_use, eig.vectors);
        }
    }

    const values = try alloc.alloc(f64, nbands_use);
    @memcpy(values, eig.values[0..nbands_use]);
    return values;
}
