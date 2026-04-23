//! Multi-k-point q≠0 DFPT SCF solver.
//!
//! For each k-point in the full BZ, solves Sternheimer and accumulates
//! ρ^(1) with the k-point weight, then mixes V^(1)_SCF until self-consistency.
//!
//! Supports k-point threading via a worker pool driven by an atomic counter;
//! each worker owns a scratch arena and a thread-local ρ^(1) buffer, which
//! are summed into the shared ρ^(1) after the per-iteration barrier.

const std = @import("std");
const math = @import("../../math/math.zig");
const scf_mod = @import("../../scf/scf.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const form_factor = @import("../../pseudopotential/form_factor.zig");

const dfpt = @import("../dfpt.zig");
const perturbation = dfpt.perturbation;
const sternheimer = dfpt.sternheimer;
const GroundState = dfpt.GroundState;
const DfptConfig = dfpt.DfptConfig;
const logDfpt = dfpt.logDfpt;
const logDfptInfo = dfpt.logDfptInfo;

const kpt_dfpt = @import("kpt_dfpt.zig");
const KPointDfptData = kpt_dfpt.KPointDfptData;
const MultiKPertResult = kpt_dfpt.MultiKPertResult;

const cross_basis = @import("cross_basis.zig");
const applyV1PsiQCached = cross_basis.applyV1PsiQCached;
const computeRho1Q = cross_basis.computeRho1Q;
const computeRho1QCached = cross_basis.computeRho1QCached;
const complexRealToReciprocal = cross_basis.complexRealToReciprocal;

const dynmat_elem_q = @import("dynmat_elem_q.zig");
const computeElecDynmatElementQ = dynmat_elem_q.computeElecDynmatElementQ;

const Grid = scf_mod.Grid;

/// Shared data for parallel DFPT k-point processing within one SCF iteration.
const DfptKpointShared = struct {
    io: std.Io,
    kpts: []KPointDfptData,
    v_scf_r: []const math.Complex,
    cfg: DfptConfig,
    atom_index: usize,
    direction: usize,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    /// Per-k-point ψ^(1) storage [n_kpts][n_occ][n_pw_kq] — each worker writes to its own ik
    psi1_per_k: [][][]math.Complex,
    /// Thread-local ρ^(1) buffers: rho1_locals[thread_index * total .. (thread_index+1) * total]
    rho1_locals: []math.Complex,
    total: usize,
    /// Pre-computed ψ^(0)_k(r) cache: [n_kpts][n_occ][total]
    psi0_r_cache: []const []const []const math.Complex,

    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

const DfptKpointWorker = struct {
    shared: *DfptKpointShared,
    thread_index: usize,
};

fn setDfptWorkerError(shared: *DfptKpointShared, e: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);

    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

fn dfptKpointWorkerFn(worker: *DfptKpointWorker) void {
    const shared = worker.shared;
    const thread_index = worker.thread_index;
    const total = shared.total;
    const start = thread_index * total;
    const rho1_local = shared.rho1_locals[start .. start + total];

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const ik = shared.next_index.fetchAdd(1, .acq_rel);
        if (ik >= shared.kpts.len) break;

        _ = arena.reset(.retain_capacity);
        const kalloc = arena.allocator();

        processOneKpointDfpt(
            kalloc,
            shared,
            ik,
            rho1_local,
        ) catch |e| {
            setDfptWorkerError(shared, e);
            shared.stop.store(1, .release);
            break;
        };
    }
}

/// Process a single k-point within one DFPT SCF iteration.
/// Solves Sternheimer for each occupied band, updates psi1_per_k[ik],
/// and accumulates ρ^(1) into the provided rho1_local buffer.
fn processOneKpointDfpt(
    alloc: std.mem.Allocator,
    shared: *DfptKpointShared,
    ik: usize,
    rho1_local: []math.Complex,
) !void {
    const kd = &shared.kpts[ik];
    const n_occ = kd.n_occ;
    const n_pw_kq = kd.n_pw_kq;
    const map_kq_ptr: *const scf_mod.PwGridMap = &kd.map_kq;
    const total = shared.total;

    // Nonlocal contexts
    const nl_ctx_k_opt = kd.apply_ctx_k.nonlocal_ctx;
    const nl_ctx_kq_opt = kd.apply_ctx_kq.nonlocal_ctx;

    // ψ^(0)(r) cache for this k-point
    const psi0_r_k = shared.psi0_r_cache[ik];

    // Solve Sternheimer for each occupied band at this k-point
    for (0..n_occ) |n| {
        // RHS: -P_c^{k+q} × H^(1)|ψ^(0)_{n,k}⟩
        const rhs = try applyV1PsiQCached(
            alloc,
            shared.grid,
            map_kq_ptr,
            shared.v_scf_r,
            psi0_r_k[n],
            n_pw_kq,
        );
        defer alloc.free(rhs);

        // Add nonlocal perturbation
        if (nl_ctx_k_opt != null and nl_ctx_kq_opt != null) {
            const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
            defer alloc.free(nl_out);

            try perturbation.applyNonlocalPerturbationQ(
                alloc,
                kd.basis_k.gvecs,
                kd.basis_kq.gvecs,
                shared.atoms,
                nl_ctx_k_opt.?,
                nl_ctx_kq_opt.?,
                shared.atom_index,
                shared.direction,
                1.0 / shared.grid.volume,
                kd.wavefunctions_k_const[n],
                nl_out,
            );
            for (0..n_pw_kq) |g| {
                rhs[g] = math.complex.add(rhs[g], nl_out[g]);
            }
        }

        // Negate
        for (0..n_pw_kq) |g| {
            rhs[g] = math.complex.scale(rhs[g], -1.0);
        }

        // Project onto conduction band in k+q space
        sternheimer.projectConduction(rhs, kd.occ_kq_const, kd.n_occ_kq);

        // Solve Sternheimer
        const result = try sternheimer.solve(
            alloc,
            kd.apply_ctx_kq,
            rhs,
            kd.eigenvalues_k[n],
            kd.occ_kq_const,
            kd.n_occ_kq,
            kd.basis_kq.gvecs,
            .{
                .tol = shared.cfg.sternheimer_tol,
                .max_iter = shared.cfg.sternheimer_max_iter,
                .alpha_shift = shared.cfg.alpha_shift,
            },
        );

        // Copy result to psi1_per_k (fixed-size buffer, no reallocation needed)
        @memcpy(shared.psi1_per_k[ik][n], result.psi1);
        alloc.free(result.psi1);
    }

    // Compute ρ^(1) for this k-point (weighted by wtk)
    const psi1_const = try alloc.alloc([]const math.Complex, n_occ);
    defer alloc.free(psi1_const);

    for (0..n_occ) |n| psi1_const[n] = shared.psi1_per_k[ik][n];

    const rho1_k_r = try computeRho1QCached(
        alloc,
        shared.grid,
        map_kq_ptr,
        psi0_r_k,
        psi1_const,
        n_occ,
        kd.weight,
    );
    defer alloc.free(rho1_k_r);

    // Accumulate into thread-local buffer
    for (0..total) |i| {
        rho1_local[i] = math.complex.add(rho1_local[i], rho1_k_r[i]);
    }
}

const Psi1Buffers = struct {
    values: [][][]math.Complex,
    consumed: bool = false,

    fn init(
        alloc: std.mem.Allocator,
        kpts: []const KPointDfptData,
    ) !Psi1Buffers {
        var self = Psi1Buffers{ .values = try alloc.alloc([][]math.Complex, kpts.len) };
        for (self.values) |*psi1_k| psi1_k.* = &.{};
        errdefer self.deinit(alloc);

        for (kpts, 0..) |kd, ik| {
            self.values[ik] = try initPsi1ForKpoint(alloc, kd.n_occ, kd.n_pw_kq);
        }
        return self;
    }

    fn deinit(self: *Psi1Buffers, alloc: std.mem.Allocator) void {
        if (self.consumed) return;
        for (self.values) |psi1_k| {
            for (psi1_k) |psi1_n| {
                if (psi1_n.len > 0) alloc.free(psi1_n);
            }
            if (psi1_k.len > 0) alloc.free(psi1_k);
        }
        alloc.free(self.values);
    }

    fn intoResult(
        self: *Psi1Buffers,
        alloc: std.mem.Allocator,
    ) ![][]const []math.Complex {
        const psi1_result = try alloc.alloc([]const []math.Complex, self.values.len);
        errdefer alloc.free(psi1_result);

        var built: usize = 0;
        errdefer {
            for (psi1_result[0..built]) |psi1_k| {
                alloc.free(@constCast(psi1_k));
            }
        }

        for (self.values, 0..) |psi1_k, ik| {
            const psi1_const = try alloc.alloc([]math.Complex, psi1_k.len);
            for (psi1_k, 0..) |psi1_n, n| psi1_const[n] = psi1_n;
            psi1_result[ik] = psi1_const;
            built += 1;
            alloc.free(psi1_k);
        }
        alloc.free(self.values);
        self.consumed = true;
        return psi1_result;
    }
};

const Psi0RCache = struct {
    values: [][][]math.Complex,

    fn init(
        alloc: std.mem.Allocator,
        grid: Grid,
        kpts: []const KPointDfptData,
    ) !Psi0RCache {
        var self = Psi0RCache{ .values = try alloc.alloc([][]math.Complex, kpts.len) };
        for (self.values) |*psi0_k| psi0_k.* = &.{};
        errdefer self.deinit(alloc);

        for (kpts, 0..) |*kd, ik| {
            self.values[ik] = try buildPsi0RForKpoint(
                alloc,
                grid,
                kd,
                grid.count(),
            );
        }
        return self;
    }

    fn deinit(self: *Psi0RCache, alloc: std.mem.Allocator) void {
        for (self.values) |psi0_k| {
            for (psi0_k) |psi0_n| {
                if (psi0_n.len > 0) alloc.free(psi0_n);
            }
            if (psi0_k.len > 0) alloc.free(psi0_k);
        }
        alloc.free(self.values);
    }

    fn constView(
        self: *const Psi0RCache,
        alloc: std.mem.Allocator,
    ) ![]const []const []const math.Complex {
        const view = try alloc.alloc([]const []const math.Complex, self.values.len);
        for (self.values, 0..) |psi0_k, ik| {
            view[ik] = @ptrCast(psi0_k);
        }
        return view;
    }
};

const ResidualInfo = struct {
    values: []math.Complex,
    norm: f64,

    fn deinit(self: *ResidualInfo, alloc: std.mem.Allocator) void {
        alloc.free(self.values);
    }
};

const MixingAction = enum {
    mixed,
    restarted,
};

const PulayState = struct {
    mixer: scf_mod.ComplexPulayMixer,
    best_vresid: f64,
    best_v_scf: ?[]math.Complex,
    pulay_active_since: usize,
    force_converge: bool,

    fn init(
        alloc: std.mem.Allocator,
        cfg: DfptConfig,
    ) PulayState {
        return .{
            .mixer = scf_mod.ComplexPulayMixer.init(alloc, cfg.pulay_history),
            .best_vresid = std.math.inf(f64),
            .best_v_scf = null,
            .pulay_active_since = cfg.pulay_start,
            .force_converge = false,
        };
    }

    fn deinit(self: *PulayState, alloc: std.mem.Allocator) void {
        if (self.best_v_scf) |values| {
            alloc.free(values);
        }
        self.mixer.deinit();
    }
};

fn initPsi1ForKpoint(
    alloc: std.mem.Allocator,
    n_occ: usize,
    n_pw_kq: usize,
) ![][]math.Complex {
    const psi1_k = try alloc.alloc([]math.Complex, n_occ);
    for (psi1_k) |*psi1_n| psi1_n.* = &.{};
    errdefer {
        for (psi1_k) |psi1_n| {
            if (psi1_n.len > 0) alloc.free(psi1_n);
        }
        alloc.free(psi1_k);
    }

    for (0..n_occ) |n| {
        psi1_k[n] = try alloc.alloc(math.Complex, n_pw_kq);
        @memset(psi1_k[n], math.complex.init(0.0, 0.0));
    }
    return psi1_k;
}

fn buildPsi0RForKpoint(
    alloc: std.mem.Allocator,
    grid: Grid,
    kd: *const KPointDfptData,
    total: usize,
) ![][]math.Complex {
    const psi0_k = try alloc.alloc([]math.Complex, kd.n_occ);
    for (psi0_k) |*psi0_n| psi0_n.* = &.{};
    errdefer {
        for (psi0_k) |psi0_n| {
            if (psi0_n.len > 0) alloc.free(psi0_n);
        }
        alloc.free(psi0_k);
    }

    for (0..kd.n_occ) |n| {
        psi0_k[n] = try alloc.alloc(math.Complex, total);
        const work = try alloc.alloc(math.Complex, total);
        defer alloc.free(work);

        @memset(work, math.complex.init(0.0, 0.0));
        kd.map_k.scatter(kd.wavefunctions_k_const[n], work);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work, psi0_k[n], null);
    }
    return psi0_k;
}

fn initDfptKpointShared(
    io: std.Io,
    kpts: []KPointDfptData,
    v_scf_r: []const math.Complex,
    cfg: DfptConfig,
    atom_index: usize,
    direction: usize,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    psi1_per_k: [][][]math.Complex,
    rho1_locals: []math.Complex,
    total: usize,
    psi0_r_cache: []const []const []const math.Complex,
    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    worker_err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
) DfptKpointShared {
    return .{
        .io = io,
        .kpts = kpts,
        .v_scf_r = v_scf_r,
        .cfg = cfg,
        .atom_index = atom_index,
        .direction = direction,
        .grid = grid,
        .species = species,
        .atoms = atoms,
        .psi1_per_k = psi1_per_k,
        .rho1_locals = rho1_locals,
        .total = total,
        .psi0_r_cache = psi0_r_cache,
        .next_index = next_index,
        .stop = stop,
        .err = worker_err,
        .err_mutex = err_mutex,
        .log_mutex = log_mutex,
    };
}

fn buildCorePerturbationReal(
    alloc: std.mem.Allocator,
    grid: Grid,
    atom: hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
    direction: usize,
    q_cart: math.Vec3,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) ![]math.Complex {
    const rho1_core_g = try perturbation.buildCorePerturbationQ(
        alloc,
        grid,
        atom,
        species,
        direction,
        q_cart,
        rho_core_tables,
    );
    defer alloc.free(rho1_core_g);

    const rho1_core_fft = try alloc.alloc(math.Complex, rho1_core_g.len);
    defer alloc.free(rho1_core_fft);

    @memcpy(rho1_core_fft, rho1_core_g);
    const rho1_core_r = try alloc.alloc(math.Complex, rho1_core_g.len);
    errdefer alloc.free(rho1_core_r);

    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_fft, rho1_core_r, null);
    return rho1_core_r;
}

fn reciprocalToComplexField(
    alloc: std.mem.Allocator,
    grid: Grid,
    values_g: []const math.Complex,
) ![]math.Complex {
    const values_fft = try alloc.alloc(math.Complex, values_g.len);
    defer alloc.free(values_fft);

    @memcpy(values_fft, values_g);
    const values_r = try alloc.alloc(math.Complex, values_g.len);
    errdefer alloc.free(values_r);

    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, values_fft, values_r, null);
    return values_r;
}

fn complexToReciprocalField(
    alloc: std.mem.Allocator,
    grid: Grid,
    values_r: []const math.Complex,
) ![]math.Complex {
    const values_fft = try alloc.alloc(math.Complex, values_r.len);
    defer alloc.free(values_fft);

    @memcpy(values_fft, values_r);
    const values_g = try alloc.alloc(math.Complex, values_r.len);
    errdefer alloc.free(values_g);

    try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, values_fft, values_g, null);
    return values_g;
}

fn accumulateSequentialRho1(
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    v_scf_r: []const math.Complex,
    cfg: DfptConfig,
    atom_index: usize,
    direction: usize,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    psi1_per_k: [][][]math.Complex,
    psi0_r_cache: []const []const []const math.Complex,
    rho1_r: []math.Complex,
) !void {
    var next_index = std.atomic.Value(usize).init(0);
    var stop = std.atomic.Value(u8).init(0);
    var worker_err: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;
    var shared = initDfptKpointShared(
        io,
        kpts,
        v_scf_r,
        cfg,
        atom_index,
        direction,
        grid,
        species,
        atoms,
        psi1_per_k,
        &.{},
        grid.count(),
        psi0_r_cache,
        &next_index,
        &stop,
        &worker_err,
        &err_mutex,
        &log_mutex,
    );

    for (0..kpts.len) |ik| {
        try processOneKpointDfpt(alloc, &shared, ik, rho1_r);
    }
}

fn sumThreadLocalRho1(
    total: usize,
    thread_count: usize,
    rho1_locals: []const math.Complex,
    rho1_r: []math.Complex,
) void {
    for (0..thread_count) |ti| {
        const local_start = ti * total;
        const local = rho1_locals[local_start .. local_start + total];
        for (0..total) |i| {
            rho1_r[i] = math.complex.add(rho1_r[i], local[i]);
        }
    }
}

fn accumulateParallelRho1(
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    v_scf_r: []const math.Complex,
    cfg: DfptConfig,
    atom_index: usize,
    direction: usize,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    psi1_per_k: [][][]math.Complex,
    psi0_r_cache: []const []const []const math.Complex,
    thread_count: usize,
    rho1_r: []math.Complex,
) !void {
    const total = grid.count();
    const rho1_locals = try alloc.alloc(math.Complex, total * thread_count);
    defer alloc.free(rho1_locals);

    @memset(rho1_locals, math.complex.init(0.0, 0.0));

    var next_index = std.atomic.Value(usize).init(0);
    var stop = std.atomic.Value(u8).init(0);
    var worker_err: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;
    var shared = initDfptKpointShared(
        io,
        kpts,
        v_scf_r,
        cfg,
        atom_index,
        direction,
        grid,
        species,
        atoms,
        psi1_per_k,
        rho1_locals,
        total,
        psi0_r_cache,
        &next_index,
        &stop,
        &worker_err,
        &err_mutex,
        &log_mutex,
    );

    var workers = try alloc.alloc(DfptKpointWorker, thread_count);
    defer alloc.free(workers);

    var threads = try alloc.alloc(std.Thread, thread_count - 1);
    defer alloc.free(threads);

    for (0..thread_count) |ti| {
        workers[ti] = .{ .shared = &shared, .thread_index = ti };
    }
    for (0..thread_count - 1) |ti| {
        threads[ti] = try std.Thread.spawn(.{}, dfptKpointWorkerFn, .{&workers[ti + 1]});
    }
    dfptKpointWorkerFn(&workers[0]);
    for (threads) |t| {
        t.join();
    }
    if (worker_err) |e| return e;

    sumThreadLocalRho1(total, thread_count, rho1_locals, rho1_r);
}

fn accumulateRho1Response(
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    cfg: DfptConfig,
    atom_index: usize,
    direction: usize,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    v_scf_r: []const math.Complex,
    psi1_buffers: *Psi1Buffers,
    psi0_r_cache: *const Psi0RCache,
    thread_count: usize,
) ![]math.Complex {
    const rho1_r = try alloc.alloc(math.Complex, grid.count());
    errdefer alloc.free(rho1_r);
    @memset(rho1_r, math.complex.init(0.0, 0.0));

    const psi0_view = try psi0_r_cache.constView(alloc);
    defer alloc.free(psi0_view);

    if (thread_count <= 1) {
        try accumulateSequentialRho1(
            alloc,
            io,
            kpts,
            v_scf_r,
            cfg,
            atom_index,
            direction,
            grid,
            species,
            atoms,
            psi1_buffers.values,
            psi0_view,
            rho1_r,
        );
        return rho1_r;
    }

    try accumulateParallelRho1(
        alloc,
        io,
        kpts,
        v_scf_r,
        cfg,
        atom_index,
        direction,
        grid,
        species,
        atoms,
        psi1_buffers.values,
        psi0_view,
        thread_count,
        rho1_r,
    );
    return rho1_r;
}

fn buildIterationRho1G(
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    cfg: DfptConfig,
    atom_index: usize,
    direction: usize,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    v_scf_g: []const math.Complex,
    psi1_buffers: *Psi1Buffers,
    psi0_r_cache: *const Psi0RCache,
    thread_count: usize,
) ![]math.Complex {
    const v_scf_r = try reciprocalToComplexField(alloc, grid, v_scf_g);
    defer alloc.free(v_scf_r);

    const rho1_r = try accumulateRho1Response(
        alloc,
        io,
        kpts,
        cfg,
        atom_index,
        direction,
        grid,
        species,
        atoms,
        v_scf_r,
        psi1_buffers,
        psi0_r_cache,
        thread_count,
    );
    defer alloc.free(rho1_r);

    return complexRealToReciprocal(alloc, grid, rho1_r);
}

fn logIterationDensity(
    iter: usize,
    rho1_g: []const math.Complex,
    vloc1_g: []const math.Complex,
    volume: f64,
    n_kpts: usize,
) void {
    var rho_norm: f64 = 0.0;
    for (rho1_g) |value| {
        rho_norm += value.r * value.r + value.i * value.i;
    }
    const d_elec_diag = computeElecDynmatElementQ(vloc1_g, rho1_g, volume);
    logDfpt(
        "dfptQ_mk: iter={d} |rho1|={e:.6} D_elec_bare=({e:.6},{e:.6}) nk={d}\n",
        .{ iter, @sqrt(rho_norm), d_elec_diag.r, d_elec_diag.i, n_kpts },
    );
}

fn buildOutputPotential(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: GroundState,
    q_cart: math.Vec3,
    vloc1_g: []const math.Complex,
    rho1_g: []const math.Complex,
    rho1_core_r: []const math.Complex,
) ![]math.Complex {
    const vh1_g = try perturbation.buildHartreePerturbationQ(alloc, grid, rho1_g, q_cart);
    defer alloc.free(vh1_g);

    const rho1_val_r = try reciprocalToComplexField(alloc, grid, rho1_g);
    defer alloc.free(rho1_val_r);

    const rho1_total_r = try alloc.alloc(math.Complex, rho1_g.len);
    defer alloc.free(rho1_total_r);

    for (0..rho1_total_r.len) |i| {
        rho1_total_r[i] = math.complex.add(rho1_val_r[i], rho1_core_r[i]);
    }

    const vxc1_r = try perturbation.buildXcPerturbationFullComplex(alloc, gs, rho1_total_r);
    defer alloc.free(vxc1_r);

    const vxc1_g = try complexToReciprocalField(alloc, grid, vxc1_r);
    defer alloc.free(vxc1_g);

    const v_out_g = try alloc.alloc(math.Complex, rho1_g.len);
    errdefer alloc.free(v_out_g);

    for (0..v_out_g.len) |i| {
        v_out_g[i] = math.complex.add(
            math.complex.add(vloc1_g[i], vh1_g[i]),
            vxc1_g[i],
        );
    }
    return v_out_g;
}

fn computeResidual(
    alloc: std.mem.Allocator,
    v_out_g: []const math.Complex,
    v_scf_g: []const math.Complex,
) !ResidualInfo {
    const residual = try alloc.alloc(math.Complex, v_out_g.len);
    errdefer alloc.free(residual);

    var residual_norm: f64 = 0.0;
    for (0..residual.len) |i| {
        residual[i] = math.complex.sub(v_out_g[i], v_scf_g[i]);
        residual_norm += residual[i].r * residual[i].r + residual[i].i * residual[i].i;
    }
    return .{
        .values = residual,
        .norm = @sqrt(residual_norm),
    };
}

fn updateBestPotential(
    alloc: std.mem.Allocator,
    residual_norm: f64,
    v_scf_g: []const math.Complex,
    state: *PulayState,
) !void {
    if (residual_norm >= state.best_vresid) return;

    state.best_vresid = residual_norm;
    if (state.best_v_scf == null) {
        state.best_v_scf = try alloc.alloc(math.Complex, v_scf_g.len);
    }
    @memcpy(state.best_v_scf.?, v_scf_g);
}

fn applyPotentialMixing(
    alloc: std.mem.Allocator,
    iter: usize,
    cfg: DfptConfig,
    v_scf_g: []math.Complex,
    residual: ResidualInfo,
    state: *PulayState,
) !MixingAction {
    const restart_triggered = iter >= state.pulay_active_since and
        residual.norm > 5.0 * state.best_vresid and
        state.best_vresid < 1.0;
    if (restart_triggered) {
        if (state.best_v_scf) |values| @memcpy(v_scf_g, values);
        if (state.best_vresid < 10.0 * cfg.scf_tol) {
            state.force_converge = true;
            logDfpt(
                "dfptQ_mk: Pulay restart (near-converged) at iter={d}" ++
                    " vresid={e:.6} best={e:.6}\n",
                .{ iter, residual.norm, state.best_vresid },
            );
            alloc.free(residual.values);
            return .restarted;
        }
        state.mixer.reset();
        state.pulay_active_since = iter + 1 + cfg.pulay_start;
        logDfpt(
            "dfptQ_mk: Pulay restart at iter={d} vresid={e:.6} best={e:.6}\n",
            .{ iter, residual.norm, state.best_vresid },
        );
        alloc.free(residual.values);
        return .restarted;
    }

    if (cfg.pulay_history > 0 and iter >= state.pulay_active_since) {
        try state.mixer.mixWithResidual(v_scf_g, residual.values, cfg.mixing_beta);
    } else {
        for (0..v_scf_g.len) |i| {
            v_scf_g[i] = math.complex.add(
                v_scf_g[i],
                math.complex.scale(residual.values[i], cfg.mixing_beta),
            );
        }
        alloc.free(residual.values);
    }
    return .mixed;
}

fn runMultiKScfLoop(
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
    q_cart: math.Vec3,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    vloc1_g: []const math.Complex,
    rho1_core_r: []const math.Complex,
    psi1_buffers: *Psi1Buffers,
    psi0_r_cache: *const Psi0RCache,
    v_scf_g: []math.Complex,
) !void {
    const n_kpts = kpts.len;
    const thread_count = scf_mod.kpointThreadCount(n_kpts, cfg.kpoint_threads);
    var state = PulayState.init(alloc, cfg);
    defer state.deinit(alloc);

    var iter: usize = 0;
    while (iter < cfg.scf_max_iter) : (iter += 1) {
        if (iter == 0 and thread_count > 1) {
            logDfptInfo(
                "dfptQ_mk: using {d} threads for {d} k-points\n",
                .{ thread_count, n_kpts },
            );
        }

        const rho1_g = try buildIterationRho1G(
            alloc,
            io,
            kpts,
            cfg,
            atom_index,
            direction,
            grid,
            species,
            atoms,
            v_scf_g,
            psi1_buffers,
            psi0_r_cache,
            thread_count,
        );
        defer alloc.free(rho1_g);

        logIterationDensity(iter, rho1_g, vloc1_g, grid.volume, n_kpts);

        const v_out_g = try buildOutputPotential(
            alloc,
            grid,
            gs,
            q_cart,
            vloc1_g,
            rho1_g,
            rho1_core_r,
        );
        defer alloc.free(v_out_g);

        const residual = try computeResidual(alloc, v_out_g, v_scf_g);
        logDfpt("dfptQ_mk: iter={d} vresid={e:.6}\n", .{ iter, residual.norm });

        const converged_now = residual.norm < cfg.scf_tol or
            (state.force_converge and residual.norm < 10.0 * cfg.scf_tol);
        if (converged_now) {
            var converged_residual = residual;
            converged_residual.deinit(alloc);
            logDfpt(
                "dfptQ_mk: converged at iter={d} vresid={e:.6}\n",
                .{ iter, residual.norm },
            );
            break;
        }

        try updateBestPotential(alloc, residual.norm, v_scf_g, &state);
        const action = try applyPotentialMixing(alloc, iter, cfg, v_scf_g, residual, &state);
        if (action == .restarted) continue;
    }
}

fn computeFinalRho1G(
    alloc: std.mem.Allocator,
    grid: Grid,
    kpts: []KPointDfptData,
    psi1_buffers: *const Psi1Buffers,
) ![]math.Complex {
    const total = grid.count();
    const final_rho1_r = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(final_rho1_r);
    @memset(final_rho1_r, math.complex.init(0.0, 0.0));

    for (kpts, 0..) |*kd, ik| {
        const psi1_const = try alloc.alloc([]const math.Complex, kd.n_occ);
        defer alloc.free(psi1_const);

        for (0..kd.n_occ) |n| psi1_const[n] = psi1_buffers.values[ik][n];
        const rho1_k_r = try computeRho1Q(
            alloc,
            grid,
            &kd.map_k,
            &kd.map_kq,
            kd.wavefunctions_k_const,
            psi1_const,
            kd.n_occ,
            kd.n_pw_k,
            kd.n_pw_kq,
            kd.weight,
        );
        defer alloc.free(rho1_k_r);

        for (0..total) |i| {
            final_rho1_r[i] = math.complex.add(final_rho1_r[i], rho1_k_r[i]);
        }
    }

    const final_rho1_g = try complexRealToReciprocal(alloc, grid, final_rho1_r);
    alloc.free(final_rho1_r);
    return final_rho1_g;
}

/// Solve DFPT perturbation at q≠0 with multiple k-points.
/// For each k-point, solves Sternheimer equations and accumulates ρ^(1)
/// with the k-point weight. Uses potential mixing for stable convergence.
///
/// Returns the converged ρ^(1)(G) (summed over all k-points) and
/// per-k-point ψ^(1) wavefunctions.
pub fn solvePerturbationQMultiK(
    alloc: std.mem.Allocator,
    io: std.Io,
    kpts: []KPointDfptData,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
    q_cart: math.Vec3,
    grid: Grid,
    gs: GroundState,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) !MultiKPertResult {
    const vloc1_g = try perturbation.buildLocalPerturbationQ(
        alloc,
        grid,
        atoms[atom_index],
        species,
        direction,
        q_cart,
        gs.local_cfg,
        ff_tables,
    );
    defer alloc.free(vloc1_g);

    const rho1_core_r = try buildCorePerturbationReal(
        alloc,
        grid,
        atoms[atom_index],
        species,
        direction,
        q_cart,
        rho_core_tables,
    );
    defer alloc.free(rho1_core_r);

    var psi1_buffers = try Psi1Buffers.init(alloc, kpts);
    errdefer psi1_buffers.deinit(alloc);

    const v_scf_g = try alloc.alloc(math.Complex, grid.count());
    defer alloc.free(v_scf_g);

    @memcpy(v_scf_g, vloc1_g);

    var psi0_r_cache = try Psi0RCache.init(alloc, grid, kpts);
    defer psi0_r_cache.deinit(alloc);

    try runMultiKScfLoop(
        alloc,
        io,
        kpts,
        atom_index,
        direction,
        cfg,
        q_cart,
        grid,
        gs,
        species,
        atoms,
        vloc1_g,
        rho1_core_r,
        &psi1_buffers,
        &psi0_r_cache,
        v_scf_g,
    );

    const final_rho1_g = try computeFinalRho1G(alloc, grid, kpts, &psi1_buffers);
    errdefer alloc.free(final_rho1_g);
    const psi1_result = try psi1_buffers.intoResult(alloc);

    return .{
        .rho1_g = final_rho1_g,
        .psi1_per_k = psi1_result,
    };
}
