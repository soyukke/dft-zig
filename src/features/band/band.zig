const std = @import("std");
const config = @import("../config/config.zig");
const fft = @import("../fft/fft.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpath = @import("../kpath/kpath.zig");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const scf = @import("../scf/scf.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const thread_pool = @import("../thread_pool.zig");

const ThreadPool = thread_pool.ThreadPool;

fn logStep(io: std.Io, msg: []const u8) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    try out.print("{s}\n", .{msg});
    try out.flush();
}

fn bandThreadCount(total: usize, cfg_threads: usize) usize {
    if (total <= 1) return 1;
    if (cfg_threads > 0) return @min(total, cfg_threads);
    const cpu_count = std.Thread.getCpuCount() catch 1;
    if (cpu_count == 0) return 1;
    return @min(total, cpu_count);
}

fn logBandKpoint(io: std.Io, idx: usize, total: usize) void {
    var buffer: [128]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    out.print("band kpoint {d}/{d}\n", .{ idx + 1, total }) catch {};
    out.flush() catch {};
}

fn logBandTiming(io: std.Io, idx: usize, total: usize, ns: u64) void {
    var buffer: [128]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    const ms = @as(f64, @floatFromInt(ns)) / 1e6;
    out.print("band_profile kpoint={d}/{d} ms={d:.1}\n", .{ idx + 1, total, ms }) catch {};
    out.flush() catch {};
}

/// Write band energies for k-path.
pub fn writeBandEnergies(
    alloc: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    cfg: config.Config,
    path: kpath.KPath,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
    extra: ?*hamiltonian.PotentialGrid,
    extra_down: ?*hamiltonian.PotentialGrid,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_dij: ?[]const []const f64,
) !void {
    // For nspin=2, compute bands for each spin channel separately
    if (cfg.scf.nspin == 2 and extra != null and extra_down != null) {
        try writeBandEnergiesForSpin(alloc, io, dir, cfg, path, species, atoms, cell_bohr, recip, volume_bohr, extra, "band_energies_up.csv", paw_tabs, paw_dij);
        try writeBandEnergiesForSpin(alloc, io, dir, cfg, path, species, atoms, cell_bohr, recip, volume_bohr, extra_down, "band_energies_down.csv", paw_tabs, paw_dij);
        return;
    }
    if (cfg.band.nbands == 0) return error.InvalidBandConfig;
    if (species.len == 0) return error.MissingPseudopotential;

    const local_alpha = if (cfg.scf.local_potential == .ewald) blk: {
        if (cfg.ewald.alpha > 0.0) break :blk cfg.ewald.alpha;
        const lmin = @min(
            @min(math.Vec3.norm(cell_bohr.row(0)), math.Vec3.norm(cell_bohr.row(1))),
            math.Vec3.norm(cell_bohr.row(2)),
        );
        break :blk 5.0 / lmin;
    } else 0.0;
    hamiltonian.configureLocalPotential(species, cfg.scf.local_potential, local_alpha);

    const min_npw = try minPlaneWaves(alloc, cfg.scf.ecut_ry, path, recip);
    if (min_npw == 0) return error.NoPlaneWaves;
    const nbands = @min(cfg.band.nbands, min_npw);

    const sym_ops = if (cfg.band.use_symmetry)
        try symmetry.getSymmetryOps(alloc, cell_bohr, atoms, 1e-6)
    else
        try alloc.alloc(symmetry.SymOp, 0);
    defer alloc.free(sym_ops);

    // For auto solver, prefer iterative if SCF potential exists
    // PAW species (with QIJ) are now supported via generalized eigenvalue problem
    var use_iterative = (cfg.band.solver == .iterative or cfg.band.solver == .cg or cfg.band.solver == .auto);
    var band_ctx: ?scf.BandIterativeContext = null;
    if (use_iterative) {
        if (extra == null) {
            try logStep(io, "band: iterative solver disabled (no SCF potential)");
            use_iterative = false;
        } else {
            const ctx_result = scf.initBandIterativeContext(
                alloc, io,
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
                    try logStep(io, "band: iterative solver disabled (grid mismatch)");
                    use_iterative = false;
                } else {
                    return err;
                }
            }
        }
    }
    defer if (band_ctx) |*ctx| ctx.deinit(alloc);

    const total_points = path.points.len;
    const thread_count = bandThreadCount(total_points, cfg.band.kpoint_threads);

    var results = try alloc.alloc(f64, total_points * nbands);
    defer alloc.free(results);

    var pool = try ThreadPool.init(alloc, 0);
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
            tables[n_tables] = try nonlocal.RadialTableSet.init(alloc, upf.beta, upf.r, upf.rab, g_max);
            n_tables += 1;
        }
        radial_tables = tables[0..n_tables];
    }
    defer if (radial_tables) |tables| {
        for (tables) |*t| t.deinit(alloc);
        // Free the original allocation (species.len elements)
        const full_slice = @as([*]nonlocal.RadialTableSet, @ptrCast(tables.ptr))[0..species.len];
        alloc.free(full_slice);
    };

    if (thread_count <= 1) {
        // Pre-create shared FFT plan for band calculation to avoid
        // expensive FFTW plan creation for each k-point
        var band_fft_plan: ?fft.Fft3dPlan = null;
        if (band_ctx) |ctx| {
            band_fft_plan = try fft.Fft3dPlan.initWithBackend(alloc, io, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz, cfg.scf.fft_backend);
        }
        defer if (band_fft_plan) |*plan| plan.deinit(alloc);

        var cache = scf.BandVectorCache{};
        defer cache.deinit();
        var band_timer = std.time.Timer.start() catch null;
        for (path.points, 0..) |kp, idx| {
            const offset = idx * nbands;
            var eigvals_opt: ?[]f64 = null;
            if (use_iterative) {
                if (band_timer) |*t| t.reset();
                const result = scf.bandEigenvaluesIterativeExt(
                    alloc, io,
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
                if (band_timer) |*t| {
                    const band_ns = t.read();
                    if (idx < 5 or idx % 20 == 0) {
                        logBandTiming(idx, total_points, band_ns);
                    }
                }
                if (result) |values| {
                    eigvals_opt = values;
                } else |err| {
                    if (err == error.InvalidGrid) {
                        try logStep(io, "band: iterative solver disabled (grid too small), falling back to dense");
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
                const h = try hamiltonian.buildHamiltonian(alloc, basis.gvecs, species, atoms, inv_volume, extra_grid);
                defer alloc.free(h);

                if (hasQij(species)) {
                    eigvals_opt = try buildAndSolveGeneralized(alloc, cfg.linalg_backend, basis.gvecs, species, atoms, inv_volume, h);
                } else if (cfg.band.use_symmetry and sym_ops.len > 0) {
                    eigvals_opt = try symmetry.symmetry_basis.computeBandEigenvalues(
                        alloc,
                        cfg.linalg_backend,
                        basis.gvecs,
                        h,
                        kp.k_frac,
                        kp.k_cart,
                        sym_ops,
                    );
                } else {
                    eigvals_opt = try linalg.hermitianEigenvalues(alloc, cfg.linalg_backend, basis.gvecs.len, h);
                }
            }
            const eigvals = eigvals_opt.?;
            defer alloc.free(eigvals);
            @memcpy(results[offset .. offset + nbands], eigvals[0..nbands]);
        }
    } else {
        const BandWork = struct {
            cfg: *const config.Config,
            points: []const kpath.KPoint,
            species: []hamiltonian.SpeciesEntry,
            atoms: []hamiltonian.AtomData,
            recip: math.Mat3,
            volume: f64,
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
                shared.err_mutex.lock();
                defer shared.err_mutex.unlock();
                if (shared.err.* == null) {
                    shared.err.* = err;
                }
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
                    const offset = idx * shared.nbands;

                    shared.log_mutex.lock();
                    if (idx % 10 == 0 or idx + 1 == shared.points.len) {
                        logBandKpoint(idx, shared.points.len);
                    }
                    shared.log_mutex.unlock();

                    _ = arena.reset(.retain_capacity);
                    const kalloc = arena.allocator();
                    const kp = shared.points[idx];
                    var eigvals: []f64 = &[_]f64{};
                    var have_vals = false;

                    const use_iter = shared.use_iterative and shared.fallback_dense.load(.acquire) == 0 and shared.ctx != null;
                    const cache = if (shared.reuse_vectors) &shared.caches[worker.thread_index] else null;
                    if (use_iter) {
                        const result = scf.bandEigenvaluesIterativeExt(
                            kalloc,
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
                                .paw_tabs = if (shared.ctx) |c| c.paw_tabs else null,
                                .paw_dij = if (shared.ctx) |c| c.paw_dij else null,
                            },
                        );
                        if (result) |values| {
                            eigvals = values;
                            have_vals = true;
                        } else |err| {
                            if (err == error.InvalidGrid) {
                                shared.fallback_dense.store(1, .release);
                            } else {
                                setBandError(shared, err);
                                shared.stop.store(1, .release);
                                break;
                            }
                        }
                    }

                    if (!have_vals) {
                        var basis = plane_wave.generate(kalloc, shared.recip, shared.cfg.scf.ecut_ry, kp.k_cart) catch |err| {
                            setBandError(shared, err);
                            shared.stop.store(1, .release);
                            break;
                        };
                        const inv_volume = 1.0 / shared.volume;
                        const h = hamiltonian.buildHamiltonian(kalloc, basis.gvecs, shared.species, shared.atoms, inv_volume, shared.extra) catch |err| {
                            basis.deinit(kalloc);
                            setBandError(shared, err);
                            shared.stop.store(1, .release);
                            break;
                        };
                        if (hasQij(shared.species)) {
                            eigvals = buildAndSolveGeneralized(kalloc, shared.cfg.linalg_backend, basis.gvecs, shared.species, shared.atoms, inv_volume, h) catch |err| {
                                basis.deinit(kalloc);
                                setBandError(shared, err);
                                shared.stop.store(1, .release);
                                break;
                            };
                        } else if (shared.cfg.band.use_symmetry and shared.sym_ops.len > 0) {
                            eigvals = symmetry.symmetry_basis.computeBandEigenvalues(
                                kalloc,
                                shared.cfg.linalg_backend,
                                basis.gvecs,
                                h,
                                kp.k_frac,
                                kp.k_cart,
                                shared.sym_ops,
                            ) catch |err| {
                                basis.deinit(kalloc);
                                setBandError(shared, err);
                                shared.stop.store(1, .release);
                                break;
                            };
                        } else {
                            eigvals = linalg.hermitianEigenvalues(kalloc, shared.cfg.linalg_backend, basis.gvecs.len, h) catch |err| {
                                basis.deinit(kalloc);
                                setBandError(shared, err);
                                shared.stop.store(1, .release);
                                break;
                            };
                        }
                        basis.deinit(kalloc);
                    }

                    if (eigvals.len < shared.nbands) {
                        setBandError(shared, error.InvalidBandConfig);
                        shared.stop.store(1, .release);
                        break;
                    }
                    @memcpy(shared.results[offset .. offset + shared.nbands], eigvals[0..shared.nbands]);
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
            .cfg = &cfg,
            .points = path.points,
            .species = species,
            .atoms = atoms,
            .recip = recip,
            .volume = volume_bohr,
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
            try logEigenvalues("band", "gamma", eigvals, nbands);

            var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, path.points[idx].k_cart);
            defer basis.deinit(alloc);
            const inv_volume = 1.0 / volume_bohr;
            const extra_grid = if (extra) |ptr| ptr.* else null;
            const h = try hamiltonian.buildHamiltonian(alloc, basis.gvecs, species, atoms, inv_volume, extra_grid);
            defer alloc.free(h);
            var eig_dense = try linalg.hermitianEigenDecomp(alloc, cfg.linalg_backend, basis.gvecs.len, h);
            defer eig_dense.deinit(alloc);
            const count = @min(nbands, eig_dense.values.len);
            try logEigenvalues("band", "gamma_dense", eig_dense.values, count);
        } else {
            var buffer: [128]u8 = undefined;
            var writer = std.Io.File.stderr().writer(io, &buffer);
            const out = &writer.interface;
            try out.writeAll("band: eig gamma not found\n");
            try out.flush();
        }
        var min_energy = std.math.inf(f64);
        var max_energy = -std.math.inf(f64);
        for (results) |value| {
            min_energy = @min(min_energy, value);
            max_energy = @max(max_energy, value);
        }
        var buffer: [256]u8 = undefined;
        var writer = std.Io.File.stderr().writer(io, &buffer);
        const out = &writer.interface;
        try out.print(
            "band: eig min={d:.6} max={d:.6} nbands={d} points={d}\n",
            .{ min_energy, max_energy, nbands, total_points },
        );
        try out.flush();
    }

    var file = try dir.createFile("band_energies.csv", .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(&buffer);
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
fn writeBandEnergiesForSpin(
    alloc: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    cfg: config.Config,
    path: kpath.KPath,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
    extra: ?*hamiltonian.PotentialGrid,
    filename: []const u8,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_dij: ?[]const []const f64,
) !void {
    if (cfg.band.nbands == 0) return error.InvalidBandConfig;
    if (species.len == 0) return error.MissingPseudopotential;

    const local_alpha = if (cfg.scf.local_potential == .ewald) blk: {
        if (cfg.ewald.alpha > 0.0) break :blk cfg.ewald.alpha;
        const lmin = @min(
            @min(math.Vec3.norm(cell_bohr.row(0)), math.Vec3.norm(cell_bohr.row(1))),
            math.Vec3.norm(cell_bohr.row(2)),
        );
        break :blk 5.0 / lmin;
    } else 0.0;
    hamiltonian.configureLocalPotential(species, cfg.scf.local_potential, local_alpha);

    const min_npw = try minPlaneWaves(alloc, cfg.scf.ecut_ry, path, recip);
    if (min_npw == 0) return error.NoPlaneWaves;
    const nbands = @min(cfg.band.nbands, min_npw);

    var band_ctx: ?scf.BandIterativeContext = null;
    if (extra != null) {
        const ctx_result = scf.initBandIterativeContext(
            alloc, io,
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
            if (err != error.InvalidGrid) return err;
        }
    }
    defer if (band_ctx) |*ctx| ctx.deinit(alloc);

    const total_points = path.points.len;
    var results = try alloc.alloc(f64, total_points * nbands);
    defer alloc.free(results);

    // Build radial tables
    var radial_tables: ?[]nonlocal.RadialTableSet = null;
    if (band_ctx != null and cfg.scf.enable_nonlocal) {
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
            tables[n_tables] = try nonlocal.RadialTableSet.init(alloc, upf.beta, upf.r, upf.rab, g_max);
            n_tables += 1;
        }
        radial_tables = tables[0..n_tables];
    }
    defer if (radial_tables) |tables| {
        for (tables) |*t| t.deinit(alloc);
        const full_slice = @as([*]nonlocal.RadialTableSet, @ptrCast(tables.ptr))[0..species.len];
        alloc.free(full_slice);
    };

    // Serial band calculation
    var band_fft_plan: ?fft.Fft3dPlan = null;
    if (band_ctx) |ctx| {
        band_fft_plan = try fft.Fft3dPlan.initWithBackend(alloc, io, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz, cfg.scf.fft_backend);
    }
    defer if (band_fft_plan) |*plan| plan.deinit(alloc);

    var cache = scf.BandVectorCache{};
    defer cache.deinit();
    var pool = try ThreadPool.init(alloc, 0);
    defer pool.deinit();

    for (path.points, 0..) |kp, idx| {
        const offset = idx * nbands;
        var eigvals_opt: ?[]f64 = null;
        if (band_ctx != null) {
            const result = scf.bandEigenvaluesIterativeExt(
                alloc, io,
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
                    .paw_tabs = if (band_ctx) |c| c.paw_tabs else null,
                    .paw_dij = if (band_ctx) |c| c.paw_dij else null,
                },
            );
            if (result) |values| {
                eigvals_opt = values;
            } else |_| {}
        }
        if (eigvals_opt == null) {
            var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kp.k_cart);
            defer basis.deinit(alloc);
            const inv_volume = 1.0 / volume_bohr;
            const extra_grid = if (extra) |ptr| ptr.* else null;
            const h = try hamiltonian.buildHamiltonian(alloc, basis.gvecs, species, atoms, inv_volume, extra_grid);
            defer alloc.free(h);
            eigvals_opt = try linalg.hermitianEigenvalues(alloc, cfg.linalg_backend, basis.gvecs.len, h);
        }
        const eigvals = eigvals_opt.?;
        defer alloc.free(eigvals);
        @memcpy(results[offset .. offset + nbands], eigvals[0..nbands]);
    }

    // Write CSV
    var file = try dir.createFile(filename, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var writer = file.writer(&buffer);
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

fn logEigenvalues(io: std.Io, prefix: []const u8, label: []const u8, values: []const f64, count: usize) !void {
    const limit = @min(count, 8);
    var buffer: [512]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    try out.print("{s}: eig {s} nbands={d}", .{ prefix, label, count });
    var i: usize = 0;
    while (i < limit) : (i += 1) {
        try out.print(" {d:.6}", .{values[i]});
    }
    if (count > limit) {
        try out.writeAll(" ...");
    }
    try out.writeAll("\n");
    try out.flush();
}

/// Check if any species has QIJ coefficients.
fn hasQij(species: []hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.qij.len > 0) return true;
    }
    return false;
}

/// Count minimal plane waves across k-path.
fn minPlaneWaves(alloc: std.mem.Allocator, ecut_ry: f64, path: kpath.KPath, recip: math.Mat3) !usize {
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
fn buildAndSolveGeneralized(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    gvecs: []plane_wave.GVector,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    inv_volume: f64,
    h: []math.Complex,
) ![]f64 {
    const s = try hamiltonian.buildOverlapMatrix(alloc, gvecs, species, atoms, inv_volume);
    errdefer alloc.free(s);
    const eigvals = try linalg.hermitianGenEigenvalues(alloc, backend, gvecs.len, h, s);
    alloc.free(s);
    return eigvals;
}
