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

const BandInitVectors = struct {
    values: ?[]const math.Complex = null,
    cols: usize = 0,
};

const BandNonlocalSetup = struct {
    nonlocal_enabled: bool,
    has_paw: bool,
    num_workspaces: usize,
    build_nonlocal_ourselves: bool,
    owned_nonlocal_ctx: ?apply.NonlocalContext = null,

    fn deinit(self: *BandNonlocalSetup, alloc: std.mem.Allocator) void {
        if (self.owned_nonlocal_ctx) |*nc| nc.deinit(alloc);
    }
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

    fillBandDiagonal(diag, basis.gvecs);

    const init_vectors = loadBandInitVectors(
        cache,
        opts.reuse_vectors,
        basis.gvecs.len,
        nbands_use,
    );
    var nonlocal_setup = try initBandNonlocalSetup(
        alloc,
        species,
        atoms,
        basis.gvecs,
        nbands_use,
        opts,
        cfg.scf.enable_nonlocal and hasNonlocal(species),
    );
    defer nonlocal_setup.deinit(alloc);

    var apply_ctx = try initBandApplyContext(
        alloc,
        io,
        cfg,
        ctx,
        basis.gvecs,
        species,
        atoms,
        opts,
        &nonlocal_setup,
    );
    defer apply_ctx.deinit(alloc);

    return try solveBandEigenproblem(
        alloc,
        cfg,
        &apply_ctx,
        basis.gvecs.len,
        nbands_use,
        diag,
        init_vectors,
        cache,
        opts,
        nonlocal_setup.has_paw,
    );
}

fn fillBandDiagonal(diag: []f64, gvecs: []const plane_wave.GVector) void {
    for (gvecs, 0..) |g, i| {
        diag[i] = g.kinetic;
    }
}

fn loadBandInitVectors(
    cache: ?*BandVectorCache,
    reuse_vectors: bool,
    basis_len: usize,
    nbands_use: usize,
) BandInitVectors {
    if (!reuse_vectors) return .{};
    if (cache) |c| {
        if (c.vectors.len > 0 and c.n == basis_len and c.nbands == nbands_use) {
            return .{ .values = c.vectors, .cols = c.nbands };
        }
    }
    return .{};
}

fn initBandNonlocalSetup(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    gvecs: []const plane_wave.GVector,
    nbands_use: usize,
    opts: BandEigenOptions,
    nonlocal_enabled: bool,
) !BandNonlocalSetup {
    const has_paw = opts.paw_tabs != null and opts.paw_dij != null;
    const num_workspaces: usize = if (opts.pool != null) @min(16, nbands_use + 4) else 1;
    var setup: BandNonlocalSetup = .{
        .nonlocal_enabled = nonlocal_enabled,
        .has_paw = has_paw,
        .num_workspaces = num_workspaces,
        .build_nonlocal_ourselves = has_paw or
            (opts.radial_tables != null and nonlocal_enabled and num_workspaces <= 1),
    };
    errdefer setup.deinit(alloc);

    if (has_paw) {
        setup.owned_nonlocal_ctx = try apply.buildNonlocalContextPaw(
            alloc,
            species,
            gvecs,
            opts.radial_tables,
            opts.paw_tabs.?,
        );
        try copyBandPawDij(alloc, &setup.owned_nonlocal_ctx.?, species, atoms, opts.paw_dij.?);
    } else if (opts.radial_tables != null and nonlocal_enabled and num_workspaces <= 1) {
        setup.owned_nonlocal_ctx = try apply.buildNonlocalContextWithTables(
            alloc,
            species,
            gvecs,
            opts.radial_tables.?,
        );
    }
    return setup;
}

fn copyBandPawDij(
    alloc: std.mem.Allocator,
    nonlocal_ctx: *apply.NonlocalContext,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    paw_dij: []const []const f64,
) !void {
    var atom_idx: usize = 0;
    for (species, 0..) |_, si| {
        const atom_count = countSpeciesAtoms(atoms, si);
        if (atom_count == 0) continue;
        try nonlocal_ctx.ensureDijPerAtom(alloc, si, atom_count);
        var species_atom_index: usize = 0;
        for (atoms) |atom| {
            if (atom.species_index != si) continue;
            if (atom_idx < paw_dij.len) {
                nonlocal_ctx.updateDijAtom(si, species_atom_index, paw_dij[atom_idx]);
            }
            species_atom_index += 1;
            atom_idx += 1;
        }
    }
}

fn countSpeciesAtoms(atoms: []const hamiltonian.AtomData, species_index: usize) usize {
    var count: usize = 0;
    for (atoms) |atom| {
        if (atom.species_index == species_index) count += 1;
    }
    return count;
}

fn initBandApplyContext(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    ctx: *const BandIterativeContext,
    gvecs: []const plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    opts: BandEigenOptions,
    nonlocal_setup: *BandNonlocalSetup,
) !ApplyContext {
    if (nonlocal_setup.build_nonlocal_ourselves) {
        var map = try PwGridMap.init(alloc, gvecs, ctx.grid);
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
        var apply_ctx = try ApplyContext.initWithCache(
            alloc,
            io,
            ctx.grid,
            gvecs,
            ctx.local_r,
            null,
            nonlocal_setup.owned_nonlocal_ctx,
            map,
            atoms,
            ctx.inv_volume,
            null,
            ctx.fft_index_map,
            fft_plan,
            owns_plan,
        );
        apply_ctx.owns_nonlocal = true;
        apply_ctx.owns_map = true;
        nonlocal_setup.owned_nonlocal_ctx = null;
        return apply_ctx;
    }
    if (opts.shared_fft_plan != null and nonlocal_setup.num_workspaces <= 1) {
        return try ApplyContext.initWithFftPlan(
            alloc,
            io,
            ctx.grid,
            gvecs,
            ctx.local_r,
            null,
            species,
            atoms,
            ctx.inv_volume,
            nonlocal_setup.nonlocal_enabled,
            null,
            ctx.fft_index_map,
            opts.shared_fft_plan.?,
        );
    }
    return try ApplyContext.initWithWorkspaces(
        alloc,
        io,
        ctx.grid,
        gvecs,
        ctx.local_r,
        null,
        species,
        atoms,
        ctx.inv_volume,
        nonlocal_setup.nonlocal_enabled,
        null,
        ctx.fft_index_map,
        cfg.scf.fft_backend,
        nonlocal_setup.num_workspaces,
    );
}

fn bandOverlapApplyFn(
    has_paw: bool,
    apply_ctx: *const ApplyContext,
) ?*const fn (*anyopaque, []const math.Complex, []math.Complex) anyerror!void {
    if (has_paw and apply_ctx.nonlocal_ctx != null and apply_ctx.nonlocal_ctx.?.has_paw) {
        return apply.applyOverlap;
    }
    return null;
}

fn solveBandEigenproblem(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    apply_ctx: *ApplyContext,
    basis_len: usize,
    nbands_use: usize,
    diag: []const f64,
    init_vectors: BandInitVectors,
    cache: ?*BandVectorCache,
    opts: BandEigenOptions,
    has_paw: bool,
) ![]f64 {
    const op = iterative.Operator{
        .n = basis_len,
        .ctx = apply_ctx,
        .apply = applyHamiltonian,
        .apply_batch = applyHamiltonianBatched,
        .apply_s = bandOverlapApplyFn(has_paw, apply_ctx),
    };
    const lobpcg_opts = iterative.Options{
        .max_iter = cfg.band.iterative_max_iter,
        .tol = cfg.band.iterative_tol,
        .max_subspace = cfg.band.iterative_max_subspace,
        .block_size = cfg.band.iterative_block_size,
        .init_diagonal = cfg.band.iterative_init_diagonal,
        .init_vectors = init_vectors.values,
        .init_vectors_cols = init_vectors.cols,
    };
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
            try c.store(basis_len, nbands_use, eig.vectors);
        }
    }

    const values = try alloc.alloc(f64, nbands_use);
    @memcpy(values, eig.values[0..nbands_use]);
    return values;
}
