const std = @import("std");
const config = @import("../../config/config.zig");
const fft = @import("../../fft/fft.zig");
const fft_sizing = @import("../../../lib/fft/sizing.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const iterative = @import("../../linalg/iterative.zig");
const linalg = @import("../../linalg/linalg.zig");
const math = @import("../../math/math.zig");
const local_potential = @import("../../pseudopotential/local_potential.zig");
const plane_wave = @import("../../plane_wave/basis.zig");
const grid_requirements = @import("../../plane_wave/grid_requirements.zig");
const symmetry = @import("../../symmetry/symmetry.zig");
const nonlocal = @import("../../pseudopotential/nonlocal.zig");
const paw_mod = @import("../../paw/paw.zig");
const apply = @import("../apply.zig");
const fft_grid = @import("../fft_grid.zig");
const grid_mod = @import("../pw_grid.zig");
const kpoint_data = @import("data.zig");
const kpoint_density = @import("density.zig");
const logging = @import("../logging.zig");
const pw_grid_map = @import("../pw_grid_map.zig");
const util = @import("../util.zig");

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
const ScfProfile = logging.ScfProfile;
const log_iterative_grid_too_small = logging.log_iterative_grid_too_small;
const profile_start = logging.profile_start;
const profile_add = logging.profile_add;
const grid_requirement = grid_requirements.grid_requirement;
const PwGridMap = pw_grid_map.PwGridMap;
const next_fft_size = fft_sizing.next_fft_size;
const fft_complex_to_reciprocal_in_place = fft_grid.fft_complex_to_reciprocal_in_place;
const fft_complex_to_reciprocal_in_place_mapped =
    fft_grid.fft_complex_to_reciprocal_in_place_mapped;
const KpointCache = kpoint_data.KpointCache;
const KpointEigenData = kpoint_data.KpointEigenData;

pub const KpointSolveInput = struct {
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
};

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
    return try ApplyContext.init_with_workspaces(
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
        1,
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
    input: KpointSolveInput,
    basis_gvecs: []plane_wave.GVector,
    inv_volume: f64,
    nbands: usize,
    use_iterative: bool,
    vnl: ?[]math.Complex,
    apply_ctx: *?ApplyContext,
) !linalg.EigenDecomp {
    const init = if (input.loose_init)
        pick_init_vectors_loose(
            input.cache,
            basis_gvecs.len,
            nbands,
            use_iterative,
            input.reuse_vectors,
        )
    else
        pick_init_vectors(input.cache, basis_gvecs.len, nbands, use_iterative, input.reuse_vectors);
    const use_cg = (input.cfg.scf.solver == .cg);
    const eig_start = if (input.profile_ptr != null) profile_start(input.io) else null;
    const eig = if (use_iterative)
        try run_iterative_eigen_decomp(
            alloc,
            input.io,
            input.cfg,
            input.grid,
            basis_gvecs,
            input.species,
            input.atoms,
            inv_volume,
            input.local_r,
            vnl,
            input.nonlocal_enabled,
            input.profile_ptr,
            input.fft_index_map,
            input.shared_fft_plan,
            input.apply_cache,
            input.radial_tables,
            input.paw_tabs,
            input.iter_max_iter,
            input.iter_tol,
            init.init_vectors,
            init.init_cols,
            nbands,
            use_cg,
            apply_ctx,
        )
    else
        try run_dense_eigen_decomp(
            alloc,
            input.io,
            input.cfg,
            basis_gvecs,
            input.species,
            input.atoms,
            inv_volume,
            input.local_cfg,
            input.potential,
            input.has_qij,
            input.profile_ptr,
        );
    if (input.profile_ptr) |p| profile_add(input.io, &p.eig_ns, eig_start);
    return eig;
}

fn solve_kpoint(
    alloc: std.mem.Allocator,
    input: KpointSolveInput,
) !SolvedKpoint {
    var basis = try generate_basis_profiled(
        alloc,
        input.io,
        input.recip,
        input.cfg.scf.ecut_ry,
        input.kp.k_cart,
        input.profile_ptr,
    );
    errdefer basis.deinit(alloc);

    const nbands = @min(@max(input.cfg.band.nbands, input.nocc), basis.gvecs.len);
    if (input.nocc > basis.gvecs.len) return error.InsufficientBands;
    const inv_volume = 1.0 / input.volume;
    var apply_ctx: ?ApplyContext = null;
    errdefer if (apply_ctx) |*ctx| ctx.deinit(alloc);

    const use_iterative = try choose_iterative_solver(
        input.io,
        input.cfg,
        input.grid,
        basis.gvecs,
        input.use_iterative_config,
    );
    const vnl = try build_vnl_if_needed(
        alloc,
        input.io,
        basis.gvecs,
        input.species,
        input.atoms,
        inv_volume,
        use_iterative,
        input.nonlocal_enabled,
        input.profile_ptr,
    );
    errdefer if (vnl) |mat| alloc.free(mat);

    var eig = try run_kpoint_solve(
        alloc,
        input,
        basis.gvecs,
        inv_volume,
        nbands,
        use_iterative,
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
        .store_vectors = use_iterative and input.reuse_vectors,
    };
}

pub fn compute_kpoint_contribution(
    alloc: std.mem.Allocator,
    input: KpointSolveInput,
    nelec: f64,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    paw_rhoij: ?*paw_mod.RhoIJ,
) !void {
    var solved = try solve_kpoint(alloc, input);
    defer solved.deinit(alloc);

    // FFT now supports arbitrary sizes via Bluestein's algorithm
    var dbufs = try kpoint_density.allocate_density_buffers(
        alloc,
        input.io,
        input.cfg,
        input.grid,
        solved.basis.gvecs,
    );
    defer kpoint_density.deinit_density_buffers(alloc, &dbufs);

    try kpoint_density.accumulate_band_contributions(
        alloc,
        .{
            .io = input.io,
            .grid = input.grid,
            .kp = input.kp,
            .basis_gvecs = solved.basis.gvecs,
            .atoms = input.atoms,
            .inv_volume = solved.inv_volume,
            .nocc = input.nocc,
            .nelec = nelec,
            .nbands = solved.nbands,
            .eig = solved.eig,
            .vnl = solved.vnl,
            .apply_ctx = &solved.apply_ctx,
            .apply_cache = input.apply_cache,
            .fft_index_map = input.fft_index_map,
            .density = &dbufs,
            .profile_ptr = input.profile_ptr,
        },
        rho,
        band_energy,
        nonlocal_energy,
        paw_rhoij,
    );

    try store_eig_result(
        input.cache,
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
        .{
            .io = io,
            .cfg = cfg,
            .grid = grid,
            .kp = kp,
            .species = species,
            .atoms = atoms,
            .recip = recip,
            .volume = volume,
            .local_cfg = local_cfg,
            .potential = potential,
            .local_r = local_r,
            .nocc = nocc,
            .use_iterative_config = use_iterative_config,
            .has_qij = has_qij,
            .nonlocal_enabled = nonlocal_enabled,
            .fft_index_map = fft_index_map,
            .iter_max_iter = iter_max_iter,
            .iter_tol = iter_tol,
            .reuse_vectors = reuse_vectors,
            .cache = cache,
            .profile_ptr = profile_ptr,
            .shared_fft_plan = shared_fft_plan,
            .apply_cache = apply_cache,
            .radial_tables = radial_tables,
            .paw_tabs = paw_tabs,
            .loose_init = true,
        },
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
        nonlocal_band = try kpoint_density.compute_nonlocal_band_entries(
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
