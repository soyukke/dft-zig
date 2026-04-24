const std = @import("std");
const config = @import("../config/config.zig");
const fft = @import("../fft/fft.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpath = @import("../kpath/kpath.zig");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const model_mod = @import("../dft/model.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const runtime_logging = @import("../runtime/logging.zig");
const scf = @import("../scf/scf.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const thread_pool = @import("../thread_pool.zig");

const ThreadPool = thread_pool.ThreadPool;

fn log_step(io: std.Io, msg: []const u8) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "{s}\n", .{msg});
}

fn band_thread_count(total: usize, cfg_threads: usize) usize {
    if (total <= 1) return 1;
    if (cfg_threads > 0) return @min(total, cfg_threads);
    const cpu_count = std.Thread.getCpuCount() catch 1;
    if (cpu_count == 0) return 1;
    return @min(total, cpu_count);
}

fn log_band_kpoint(io: std.Io, idx: usize, total: usize) void {
    const logger = runtime_logging.stderr(io, .info);
    logger.print(.info, "band kpoint {d}/{d}\n", .{ idx + 1, total }) catch {};
}

fn log_band_timing(io: std.Io, idx: usize, total: usize, ns: u64) void {
    const ms = @as(f64, @floatFromInt(ns)) / 1e6;
    const logger = runtime_logging.stderr(io, .info);
    logger.print(
        .info,
        "band_profile kpoint={d}/{d} ms={d:.1}\n",
        .{ idx + 1, total, ms },
    ) catch {};
}

fn log_band_debug(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    const logger = runtime_logging.stderr(io, .debug);
    try logger.print(.debug, fmt, args);
}

const BandRadialTables = struct {
    storage: ?[]nonlocal.RadialTableSet = null,
    view: ?[]nonlocal.RadialTableSet = null,

    fn init(
        alloc: std.mem.Allocator,
        cfg: config.Config,
        species: []const hamiltonian.SpeciesEntry,
        enabled: bool,
    ) !BandRadialTables {
        if (!enabled) return .{};

        const g_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 1.5;
        const storage = try alloc.alloc(nonlocal.RadialTableSet, species.len);
        var n_tables: usize = 0;
        errdefer {
            for (storage[0..n_tables]) |*table| table.deinit(alloc);
            alloc.free(storage);
        }

        for (species) |entry| {
            const upf = entry.upf.*;
            if (upf.beta.len == 0 or upf.dij.len == 0) continue;
            storage[n_tables] = try nonlocal.RadialTableSet.init(
                alloc,
                upf.beta,
                upf.r,
                upf.rab,
                g_max,
            );
            n_tables += 1;
        }

        return .{
            .storage = storage,
            .view = storage[0..n_tables],
        };
    }

    fn deinit(self: *BandRadialTables, alloc: std.mem.Allocator) void {
        if (self.view) |tables| {
            for (tables) |*table| table.deinit(alloc);
        }
        if (self.storage) |storage| {
            alloc.free(storage);
        }
    }
};

fn copy_band_paw_dij(
    alloc: std.mem.Allocator,
    paw_dij: []const []const f64,
) ![][]f64 {
    const dij_copy = try alloc.alloc([]f64, paw_dij.len);
    errdefer {
        for (dij_copy) |values| {
            if (values.len > 0) alloc.free(values);
        }
        alloc.free(dij_copy);
    }

    for (dij_copy) |*values| values.* = &.{};
    for (paw_dij, 0..) |atom_dij, ai| {
        dij_copy[ai] = try alloc.alloc(f64, atom_dij.len);
        @memcpy(dij_copy[ai], atom_dij);
    }
    return dij_copy;
}

fn attach_band_paw_data(
    alloc: std.mem.Allocator,
    ctx: *scf.BandIterativeContext,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_dij: ?[]const []const f64,
) !void {
    if (paw_tabs) |tabs| {
        ctx.paw_tabs = tabs;
    }
    if (paw_dij) |dij| {
        ctx.paw_dij = try copy_band_paw_dij(alloc, dij);
    }
}

fn init_optional_band_context(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    extra: ?*hamiltonian.PotentialGrid,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_dij: ?[]const []const f64,
) !?scf.BandIterativeContext {
    if (extra == null) return null;

    const ctx_result = scf.init_band_iterative_context(
        alloc,
        io,
        cfg,
        species,
        atoms,
        recip,
        volume_bohr,
        extra.?.*,
    );
    if (ctx_result) |ctx| {
        var band_ctx = ctx;
        errdefer band_ctx.deinit(alloc);
        try attach_band_paw_data(alloc, &band_ctx, paw_tabs, paw_dij);
        return band_ctx;
    } else |err| {
        if (err == error.InvalidGrid) return null;
        return err;
    }
}

fn init_band_fft_plan(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    band_ctx: ?*const scf.BandIterativeContext,
) !?fft.Fft3dPlan {
    if (band_ctx) |ctx| {
        return try fft.Fft3dPlan.init_with_backend(
            alloc,
            io,
            ctx.grid.nx,
            ctx.grid.ny,
            ctx.grid.nz,
            cfg.scf.fft_backend,
        );
    }
    return null;
}

fn compute_spin_band_point_eigenvalues(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    kp: kpath.KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    extra: ?*hamiltonian.PotentialGrid,
    nbands: usize,
    band_ctx: ?*scf.BandIterativeContext,
    cache: *scf.BandVectorCache,
    pool: ?*ThreadPool,
    band_fft_plan: ?fft.Fft3dPlan,
    radial_tables: ?[]const nonlocal.RadialTableSet,
) ![]f64 {
    if (band_ctx) |ctx| {
        const result = scf.band_eigenvalues_iterative_ext(
            alloc,
            io,
            cfg,
            ctx,
            kp.k_cart,
            species,
            atoms,
            recip,
            nbands,
            cache,
            .{
                .reuse_vectors = cfg.band.iterative_reuse_vectors,
                .pool = pool,
                .shared_fft_plan = band_fft_plan,
                .radial_tables = radial_tables,
                .paw_tabs = ctx.paw_tabs,
                .paw_dij = ctx.paw_dij,
            },
        );
        if (result) |values| return values else |_| {}
    }

    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kp.k_cart);
    defer basis.deinit(alloc);

    const inv_volume = 1.0 / volume_bohr;
    const extra_grid = if (extra) |ptr| ptr.* else null;
    const h = try hamiltonian.build_hamiltonian(
        alloc,
        basis.gvecs,
        species,
        atoms,
        inv_volume,
        local_cfg,
        extra_grid,
    );
    defer alloc.free(h);

    return linalg.hermitian_eigenvalues(
        alloc,
        cfg.linalg_backend,
        basis.gvecs.len,
        h,
    );
}

fn compute_spin_band_results(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    path: kpath.KPath,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume_bohr: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    extra: ?*hamiltonian.PotentialGrid,
    nbands: usize,
    band_ctx: ?*scf.BandIterativeContext,
    radial_tables: ?[]const nonlocal.RadialTableSet,
) ![]f64 {
    var results = try alloc.alloc(f64, path.points.len * nbands);
    errdefer alloc.free(results);

    var band_fft_plan = try init_band_fft_plan(alloc, io, cfg, band_ctx);
    defer if (band_fft_plan) |*plan| plan.deinit(alloc);

    var cache = scf.BandVectorCache{};
    defer cache.deinit();

    var pool = try ThreadPool.init(alloc, io, 0);
    defer pool.deinit();

    for (path.points, 0..) |kp, idx| {
        const eigvals = try compute_spin_band_point_eigenvalues(
            alloc,
            io,
            cfg,
            kp,
            species,
            atoms,
            recip,
            volume_bohr,
            local_cfg,
            extra,
            nbands,
            band_ctx,
            &cache,
            if (cfg.band.lobpcg_parallel) &pool else null,
            band_fft_plan,
            radial_tables,
        );
        defer alloc.free(eigvals);

        const offset = idx * nbands;
        @memcpy(results[offset .. offset + nbands], eigvals[0..nbands]);
    }
    return results;
}

fn write_band_csv(
    io: std.Io,
    dir: std.Io.Dir,
    filename: []const u8,
    path: kpath.KPath,
    nbands: usize,
    results: []const f64,
) !void {
    var file = try dir.createFile(io, filename, .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    try out.writeAll("index,dist");
    var band_index: usize = 0;
    while (band_index < nbands) : (band_index += 1) {
        try out.print(",band{d}", .{band_index});
    }
    try out.writeAll("\n");

    for (path.points, 0..) |kp, idx| {
        const offset = idx * nbands;
        try out.print("{d},{d:.10}", .{ idx, kp.distance });
        var eig_index: usize = 0;
        while (eig_index < nbands) : (eig_index += 1) {
            try out.print(",{d:.10}", .{results[offset + eig_index]});
        }
        try out.writeAll("\n");
    }
    try out.flush();
}

fn set_parallel_band_error(shared: anytype, err: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);

    if (shared.err.* == null) {
        shared.err.* = err;
    }
}

fn compute_parallel_iterative_band_values(
    kalloc: std.mem.Allocator,
    shared: anytype,
    kp: kpath.KPoint,
    thread_index: usize,
) !?[]f64 {
    const use_iter =
        shared.use_iterative and
        shared.fallback_dense.load(.acquire) == 0 and
        shared.ctx != null;
    if (!use_iter) return null;

    const cache = if (shared.reuse_vectors)
        &shared.caches[thread_index]
    else
        null;
    const result = scf.band_eigenvalues_iterative_ext(
        kalloc,
        shared.io,
        shared.cfg.*,
        shared.ctx.?,
        kp.k_cart,
        shared.species,
        shared.atoms,
        shared.recip,
        shared.nbands,
        cache,
        .{
            .reuse_vectors = shared.reuse_vectors,
            .pool = null,
            .paw_tabs = if (shared.ctx) |ctx| ctx.paw_tabs else null,
            .paw_dij = if (shared.ctx) |ctx| ctx.paw_dij else null,
        },
    );
    if (result) |values| return values else |err| {
        if (err == error.InvalidGrid) {
            shared.fallback_dense.store(1, .release);
            return null;
        }
        return err;
    }
}

fn compute_parallel_dense_band_values(
    kalloc: std.mem.Allocator,
    shared: anytype,
    kp: kpath.KPoint,
) ![]f64 {
    var basis = try plane_wave.generate(
        kalloc,
        shared.recip,
        shared.cfg.scf.ecut_ry,
        kp.k_cart,
    );
    defer basis.deinit(kalloc);

    const inv_volume = 1.0 / shared.volume;
    const h = try hamiltonian.build_hamiltonian(
        kalloc,
        basis.gvecs,
        shared.species,
        shared.atoms,
        inv_volume,
        shared.local_cfg,
        shared.extra,
    );

    if (has_qij(shared.species)) {
        return build_and_solve_generalized(
            kalloc,
            shared.cfg.linalg_backend,
            basis.gvecs,
            shared.species,
            shared.atoms,
            inv_volume,
            h,
        );
    }
    if (shared.cfg.band.use_symmetry and shared.sym_ops.len > 0) {
        return symmetry.symmetry_basis.compute_band_eigenvalues(
            kalloc,
            shared.cfg.linalg_backend,
            basis.gvecs,
            h,
            kp.k_frac,
            kp.k_cart,
            shared.sym_ops,
        );
    }
    return linalg.hermitian_eigenvalues(
        kalloc,
        shared.cfg.linalg_backend,
        basis.gvecs.len,
        h,
    );
}

fn process_parallel_band_work_item(
    kalloc: std.mem.Allocator,
    shared: anytype,
    thread_index: usize,
    idx: usize,
) !void {
    const kp = shared.points[idx];
    const eigvals =
        (try compute_parallel_iterative_band_values(kalloc, shared, kp, thread_index)) orelse
        try compute_parallel_dense_band_values(kalloc, shared, kp);
    if (eigvals.len < shared.nbands) return error.InvalidBandConfig;

    const offset = idx * shared.nbands;
    const dst = shared.results[offset .. offset + shared.nbands];
    @memcpy(dst, eigvals[0..shared.nbands]);
}

/// Write band energies for k-path.
pub fn write_band_energies(
    alloc: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    cfg: config.Config,
    path: kpath.KPath,
    model: *const model_mod.Model,
    extra: ?*hamiltonian.PotentialGrid,
    extra_down: ?*hamiltonian.PotentialGrid,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_dij: ?[]const []const f64,
) !void {
    const species = model.species;
    const atoms = model.atoms;
    const cell_bohr = model.cell_bohr;
    const recip = model.recip;
    const volume_bohr = model.volume_bohr;
    // For nspin=2, compute bands for each spin channel separately
    if (cfg.scf.nspin == 2 and extra != null and extra_down != null) {
        try write_band_energies_for_spin(
            alloc,
            io,
            dir,
            cfg,
            path,
            model,
            extra,
            "band_energies_up.csv",
            paw_tabs,
            paw_dij,
        );
        try write_band_energies_for_spin(
            alloc,
            io,
            dir,
            cfg,
            path,
            model,
            extra_down,
            "band_energies_down.csv",
            paw_tabs,
            paw_dij,
        );
        return;
    }
    if (cfg.band.nbands == 0) return error.InvalidBandConfig;
    if (species.len == 0) return error.MissingPseudopotential;

    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, cell_bohr);

    const min_npw = try min_plane_waves(alloc, cfg.scf.ecut_ry, path, recip);
    if (min_npw == 0) return error.NoPlaneWaves;
    const nbands = @min(cfg.band.nbands, min_npw);

    const sym_ops = if (cfg.band.use_symmetry)
        try symmetry.get_symmetry_ops(alloc, cell_bohr, atoms, 1e-6)
    else
        try alloc.alloc(symmetry.SymOp, 0);
    defer alloc.free(sym_ops);

    // For auto solver, prefer iterative if SCF potential exists
    // PAW species (with QIJ) are now supported via generalized eigenvalue problem
    var use_iterative =
        cfg.band.solver == .iterative or
        cfg.band.solver == .cg or
        cfg.band.solver == .auto;
    var band_ctx: ?scf.BandIterativeContext = null;
    if (use_iterative) {
        if (extra == null) {
            try log_step(io, "band: iterative solver disabled (no SCF potential)");
            use_iterative = false;
        } else {
            const ctx_result = scf.init_band_iterative_context(
                alloc,
                io,
                cfg,
                species,
                atoms,
                recip,
                volume_bohr,
                extra.?.*,
            );
            if (ctx_result) |ctx| {
                band_ctx = ctx;
                // Set PAW data if available
                if (paw_tabs) |tabs| {
                    band_ctx.?.paw_tabs = tabs;
                }
                if (paw_dij) |dij| {
                    // Make owned copies of per-atom D_ij for BandIterativeContext
                    const dij_copy = try alloc.alloc([]f64, dij.len);
                    errdefer {
                        for (dij_copy) |d| alloc.free(d);
                        alloc.free(dij_copy);
                    }
                    for (dij, 0..) |atom_dij, ai| {
                        dij_copy[ai] = try alloc.alloc(f64, atom_dij.len);
                        @memcpy(dij_copy[ai], atom_dij);
                    }
                    band_ctx.?.paw_dij = dij_copy;
                }
            } else |err| {
                if (err == error.InvalidGrid) {
                    try log_step(io, "band: iterative solver disabled (grid mismatch)");
                    use_iterative = false;
                } else {
                    return err;
                }
            }
        }
    }
    defer if (band_ctx) |*ctx| ctx.deinit(alloc);

    const total_points = path.points.len;
    const thread_count = band_thread_count(total_points, cfg.band.kpoint_threads);

    var results = try alloc.alloc(f64, total_points * nbands);
    defer alloc.free(results);

    var pool = try ThreadPool.init(alloc, io, 0);
    defer pool.deinit();

    // Build radial projector lookup tables for fast NonlocalContext construction.
    // The tables are shared across all band k-points and eliminate expensive
    // numerical integration (O(N_r) per G-vector -> O(1) per G-vector).
    var radial_tables: ?[]nonlocal.RadialTableSet = null;
    if (use_iterative and cfg.scf.enable_nonlocal) {
        // Estimate g_max from ecut: |G+k|^2/2 <= ecut_ry, so |G+k|_max = sqrt(2*ecut_ry)
        // Add margin for k-point offset
        const g_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 1.5;
        var tables = try alloc.alloc(nonlocal.RadialTableSet, species.len);
        var n_tables: usize = 0;
        errdefer {
            for (tables[0..n_tables]) |*t| t.deinit(alloc);
            alloc.free(tables);
        }
        for (species) |entry| {
            const upf = entry.upf.*;
            if (upf.beta.len == 0 or upf.dij.len == 0) continue;
            tables[n_tables] = try nonlocal.RadialTableSet.init(
                alloc,
                upf.beta,
                upf.r,
                upf.rab,
                g_max,
            );
            n_tables += 1;
        }
        radial_tables = tables[0..n_tables];
    }
    defer if (radial_tables) |tables| {
        for (tables) |*t| t.deinit(alloc);
        // Free the original allocation (species.len elements)
        const full_ptr = @as([*]nonlocal.RadialTableSet, @ptrCast(tables.ptr));
        const full_slice = full_ptr[0..species.len];
        alloc.free(full_slice);
    };

    if (thread_count <= 1) {
        // Pre-create shared FFT plan for band calculation to avoid
        // expensive FFTW plan creation for each k-point
        var band_fft_plan: ?fft.Fft3dPlan = null;
        if (band_ctx) |ctx| {
            band_fft_plan = try fft.Fft3dPlan.init_with_backend(
                alloc,
                io,
                ctx.grid.nx,
                ctx.grid.ny,
                ctx.grid.nz,
                cfg.scf.fft_backend,
            );
        }
        defer if (band_fft_plan) |*plan| plan.deinit(alloc);

        var cache = scf.BandVectorCache{};
        defer cache.deinit();

        for (path.points, 0..) |kp, idx| {
            const offset = idx * nbands;
            var eigvals_opt: ?[]f64 = null;
            const band_start_ts = std.Io.Clock.Timestamp.now(io, .awake);
            if (use_iterative) {
                const result = scf.band_eigenvalues_iterative_ext(
                    alloc,
                    io,
                    cfg,
                    &band_ctx.?,
                    kp.k_cart,
                    species,
                    atoms,
                    recip,
                    nbands,
                    &cache,
                    .{
                        .reuse_vectors = cfg.band.iterative_reuse_vectors,
                        .pool = if (cfg.band.lobpcg_parallel) &pool else null,
                        .shared_fft_plan = band_fft_plan,
                        .radial_tables = radial_tables,
                        .paw_tabs = if (band_ctx) |ctx| ctx.paw_tabs else null,
                        .paw_dij = if (band_ctx) |ctx| ctx.paw_dij else null,
                    },
                );
                const band_ns: u64 = @intCast(band_start_ts.untilNow(io).raw.nanoseconds);
                if (idx < 5 or idx % 20 == 0) {
                    log_band_timing(io, idx, total_points, band_ns);
                }
                if (result) |values| {
                    eigvals_opt = values;
                } else |err| {
                    if (err == error.InvalidGrid) {
                        try log_step(
                            io,
                            "band: iterative solver disabled (grid too small)," ++
                                " falling back to dense",
                        );
                        use_iterative = false;
                    } else {
                        return err;
                    }
                }
            }

            if (eigvals_opt == null) {
                var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kp.k_cart);
                defer basis.deinit(alloc);

                const inv_volume = 1.0 / volume_bohr;
                const extra_grid = if (extra) |ptr| ptr.* else null;
                const h = try hamiltonian.build_hamiltonian(
                    alloc,
                    basis.gvecs,
                    species,
                    atoms,
                    inv_volume,
                    local_cfg,
                    extra_grid,
                );
                defer alloc.free(h);

                if (has_qij(species)) {
                    eigvals_opt = try build_and_solve_generalized(
                        alloc,
                        cfg.linalg_backend,
                        basis.gvecs,
                        species,
                        atoms,
                        inv_volume,
                        h,
                    );
                } else if (cfg.band.use_symmetry and sym_ops.len > 0) {
                    eigvals_opt = try symmetry.symmetry_basis.compute_band_eigenvalues(
                        alloc,
                        cfg.linalg_backend,
                        basis.gvecs,
                        h,
                        kp.k_frac,
                        kp.k_cart,
                        sym_ops,
                    );
                } else {
                    eigvals_opt = try linalg.hermitian_eigenvalues(
                        alloc,
                        cfg.linalg_backend,
                        basis.gvecs.len,
                        h,
                    );
                }
            }
            const eigvals = eigvals_opt.?;
            defer alloc.free(eigvals);

            @memcpy(results[offset .. offset + nbands], eigvals[0..nbands]);
        }
    } else {
        const BandWork = struct {
            io: std.Io,
            cfg: *const config.Config,
            points: []const kpath.KPoint,
            species: []const hamiltonian.SpeciesEntry,
            atoms: []const hamiltonian.AtomData,
            recip: math.Mat3,
            volume: f64,
            local_cfg: local_potential.LocalPotentialConfig,
            extra: ?hamiltonian.PotentialGrid,
            nbands: usize,
            use_iterative: bool,
            ctx: ?*scf.BandIterativeContext,
            reuse_vectors: bool,
            caches: []scf.BandVectorCache,
            sym_ops: []const symmetry.SymOp,
            results: []f64,
            next_index: *std.atomic.Value(usize),
            stop: *std.atomic.Value(u8),
            fallback_dense: *std.atomic.Value(u8),
            err: *?anyerror,
            err_mutex: *std.Io.Mutex,
            log_mutex: *std.Io.Mutex,
        };

        const BandWorker = struct {
            shared: *BandWork,
            thread_index: usize,
        };

        const setBandError = struct {
            fn run(shared: *BandWork, err: anyerror) void {
                set_parallel_band_error(shared, err);
            }
        }.run;

        const workerFn = struct {
            fn run(worker: *BandWorker) void {
                const shared = worker.shared;
                var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
                defer arena.deinit();

                while (true) {
                    if (shared.stop.load(.acquire) != 0) break;
                    const idx = shared.next_index.fetchAdd(1, .acq_rel);
                    if (idx >= shared.points.len) break;

                    shared.log_mutex.lockUncancelable(shared.io);
                    if (idx % 10 == 0 or idx + 1 == shared.points.len) {
                        log_band_kpoint(shared.io, idx, shared.points.len);
                    }
                    shared.log_mutex.unlock(shared.io);

                    _ = arena.reset(.retain_capacity);
                    const kalloc = arena.allocator();
                    process_parallel_band_work_item(
                        kalloc,
                        shared,
                        worker.thread_index,
                        idx,
                    ) catch |err| {
                        setBandError(shared, err);
                        shared.stop.store(1, .release);
                        break;
                    };
                }
            }
        }.run;

        var next_index = std.atomic.Value(usize).init(0);
        var stop = std.atomic.Value(u8).init(0);
        var fallback_dense = std.atomic.Value(u8).init(0);
        var worker_error: ?anyerror = null;
        var err_mutex = std.Io.Mutex.init;
        var log_mutex = std.Io.Mutex.init;

        const caches = try alloc.alloc(scf.BandVectorCache, thread_count);
        defer {
            for (caches) |*cache| {
                cache.deinit();
            }
            alloc.free(caches);
        }

        for (caches) |*cache| {
            cache.* = .{};
        }

        var shared = BandWork{
            .io = io,
            .cfg = &cfg,
            .points = path.points,
            .species = species,
            .atoms = atoms,
            .recip = recip,
            .volume = volume_bohr,
            .local_cfg = local_cfg,
            .extra = if (extra) |ptr| ptr.* else null,
            .nbands = nbands,
            .use_iterative = use_iterative,
            .ctx = if (band_ctx) |*ctx| ctx else null,
            .reuse_vectors = cfg.band.iterative_reuse_vectors,
            .caches = caches,
            .sym_ops = sym_ops,
            .results = results,
            .next_index = &next_index,
            .stop = &stop,
            .fallback_dense = &fallback_dense,
            .err = &worker_error,
            .err_mutex = &err_mutex,
            .log_mutex = &log_mutex,
        };

        const workers = try alloc.alloc(BandWorker, thread_count);
        defer alloc.free(workers);

        const threads = try alloc.alloc(std.Thread, thread_count);
        defer alloc.free(threads);

        var t: usize = 0;
        while (t < thread_count) : (t += 1) {
            workers[t] = .{ .shared = &shared, .thread_index = t };
            threads[t] = try std.Thread.spawn(.{}, workerFn, .{&workers[t]});
        }
        for (threads) |thread| {
            thread.join();
        }

        if (worker_error) |err| return err;
    }

    if (cfg.scf.debug_fermi) {
        var gamma_index: ?usize = null;
        for (path.points, 0..) |kp, idx| {
            if (math.Vec3.norm(kp.k_cart) < 1e-8) {
                gamma_index = idx;
                break;
            }
        }
        if (gamma_index) |idx| {
            const offset = idx * nbands;
            const eigvals = results[offset .. offset + nbands];
            try log_eigenvalues(io, "band", "gamma", eigvals, nbands);

            var basis = try plane_wave.generate(
                alloc,
                recip,
                cfg.scf.ecut_ry,
                path.points[idx].k_cart,
            );
            defer basis.deinit(alloc);

            const inv_volume = 1.0 / volume_bohr;
            const extra_grid = if (extra) |ptr| ptr.* else null;
            const h = try hamiltonian.build_hamiltonian(
                alloc,
                basis.gvecs,
                species,
                atoms,
                inv_volume,
                local_cfg,
                extra_grid,
            );
            defer alloc.free(h);

            var eig_dense = try linalg.hermitian_eigen_decomp(
                alloc,
                cfg.linalg_backend,
                basis.gvecs.len,
                h,
            );
            defer eig_dense.deinit(alloc);

            const count = @min(nbands, eig_dense.values.len);
            try log_eigenvalues(io, "band", "gamma_dense", eig_dense.values, count);
        } else {
            try log_band_debug(io, "band: eig gamma not found\n", .{});
        }
        var min_energy = std.math.inf(f64);
        var max_energy = -std.math.inf(f64);
        for (results) |value| {
            min_energy = @min(min_energy, value);
            max_energy = @max(max_energy, value);
        }
        try log_band_debug(
            io,
            "band: eig min={d:.6} max={d:.6} nbands={d} points={d}\n",
            .{ min_energy, max_energy, nbands, total_points },
        );
    }

    var file = try dir.createFile(io, "band_energies.csv", .{ .truncate = true });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    try out.writeAll("index,dist");
    var b: usize = 0;
    while (b < nbands) : (b += 1) {
        try out.print(",band{d}", .{b});
    }
    try out.writeAll("\n");
    for (path.points, 0..) |kp, idx| {
        const offset = idx * nbands;
        const eigvals = results[offset .. offset + nbands];
        try out.print("{d},{d:.10}", .{ idx, kp.distance });
        var j: usize = 0;
        while (j < nbands) : (j += 1) {
            try out.print(",{d:.10}", .{eigvals[j]});
        }
        try out.writeAll("\n");
    }
    try out.flush();
}

/// Write band energies for a single spin channel.
fn write_band_energies_for_spin(
    alloc: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    cfg: config.Config,
    path: kpath.KPath,
    model: *const model_mod.Model,
    extra: ?*hamiltonian.PotentialGrid,
    filename: []const u8,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_dij: ?[]const []const f64,
) !void {
    const species = model.species;
    const atoms = model.atoms;
    const cell_bohr = model.cell_bohr;
    const recip = model.recip;
    const volume_bohr = model.volume_bohr;

    if (cfg.band.nbands == 0) return error.InvalidBandConfig;
    if (species.len == 0) return error.MissingPseudopotential;

    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, cell_bohr);
    const min_npw = try min_plane_waves(alloc, cfg.scf.ecut_ry, path, recip);
    if (min_npw == 0) return error.NoPlaneWaves;
    const nbands = @min(cfg.band.nbands, min_npw);

    var band_ctx = try init_optional_band_context(
        alloc,
        io,
        cfg,
        species,
        atoms,
        recip,
        volume_bohr,
        extra,
        paw_tabs,
        paw_dij,
    );
    defer if (band_ctx) |*ctx| ctx.deinit(alloc);

    var radial_tables = try BandRadialTables.init(
        alloc,
        cfg,
        species,
        band_ctx != null and cfg.scf.enable_nonlocal,
    );
    defer radial_tables.deinit(alloc);

    const results = try compute_spin_band_results(
        alloc,
        io,
        cfg,
        path,
        species,
        atoms,
        recip,
        volume_bohr,
        local_cfg,
        extra,
        nbands,
        if (band_ctx) |*ctx| ctx else null,
        radial_tables.view,
    );
    defer alloc.free(results);

    try write_band_csv(io, dir, filename, path, nbands, results);
}

fn log_eigenvalues(
    io: std.Io,
    prefix: []const u8,
    label: []const u8,
    values: []const f64,
    count: usize,
) !void {
    const limit = @min(count, 8);
    try log_band_debug(io, "{s}: eig {s} nbands={d}", .{ prefix, label, count });
    var i: usize = 0;
    while (i < limit) : (i += 1) {
        try log_band_debug(io, " {d:.6}", .{values[i]});
    }
    if (count > limit) {
        const logger = runtime_logging.stderr(io, .debug);
        try logger.write_all(.debug, " ...");
    }
    const logger = runtime_logging.stderr(io, .debug);
    try logger.write_all(.debug, "\n");
}

/// Check if any species has QIJ coefficients.
fn has_qij(species: []const hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.qij.len > 0) return true;
    }
    return false;
}

/// Count minimal plane waves across k-path.
fn min_plane_waves(
    alloc: std.mem.Allocator,
    ecut_ry: f64,
    path: kpath.KPath,
    recip: math.Mat3,
) !usize {
    var min_npw: usize = std.math.maxInt(usize);
    for (path.points) |kp| {
        var basis = try plane_wave.generate(alloc, recip, ecut_ry, kp.k_cart);
        const count = basis.gvecs.len;
        basis.deinit(alloc);
        min_npw = @min(min_npw, count);
    }
    return min_npw;
}

/// Build overlap matrix and solve generalized eigenvalues.
fn build_and_solve_generalized(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    gvecs: []plane_wave.GVector,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    h: []math.Complex,
) ![]f64 {
    const s = try hamiltonian.build_overlap_matrix(alloc, gvecs, species, atoms, inv_volume);
    errdefer alloc.free(s);
    const eigvals = try linalg.hermitian_gen_eigenvalues(alloc, backend, gvecs.len, h, s);
    alloc.free(s);
    return eigvals;
}
