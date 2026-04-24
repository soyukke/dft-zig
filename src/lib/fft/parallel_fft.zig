//! Parallel 3D FFT implementation with thread pool
//!
//! Uses a persistent thread pool to parallelize 1D FFTs along each axis.
//! Threads are created once at plan initialization and reused for all FFT calls.
//! Each thread has its own FFT plan and scratch buffer for thread safety.

const std = @import("std");
pub const Complex = @import("complex.zig").Complex;
const Plan1d = @import("fft.zig").Plan1d;

/// Thread-local workspace for parallel FFT
const ThreadWorkspace = struct {
    plan_x: Plan1d,
    plan_y: Plan1d,
    plan_z: Plan1d,
    scratch: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !ThreadWorkspace {
        var plan_x = try Plan1d.init(allocator, nx);
        errdefer plan_x.deinit();

        var plan_y = try Plan1d.init(allocator, ny);
        errdefer plan_y.deinit();

        var plan_z = try Plan1d.init(allocator, nz);
        errdefer plan_z.deinit();

        const max_len = @max(nx, @max(ny, nz));
        const scratch = try allocator.alloc(Complex, max_len);
        errdefer allocator.free(scratch);

        return .{
            .plan_x = plan_x,
            .plan_y = plan_y,
            .plan_z = plan_z,
            .scratch = scratch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadWorkspace) void {
        self.plan_x.deinit();
        self.plan_y.deinit();
        self.plan_z.deinit();
        if (self.scratch.len > 0) self.allocator.free(self.scratch);
    }
};

/// Task type for thread pool
const TaskType = enum {
    none,
    fft_x,
    fft_y,
    fft_z,
    shutdown,
};

/// Shared state for thread pool synchronization
const ThreadPoolState = struct {
    // Task description
    task: TaskType,
    data: ?[]Complex,
    inverse: bool,

    // Grid dimensions (cached from plan)
    nx: usize,
    ny: usize,
    nz: usize,

    // Work distribution
    total_work: usize,
    next_work_item: std.atomic.Value(usize),

    // Task generation for synchronization (incremented each task)
    task_generation: usize,

    // Barrier for synchronization
    barrier_count: std.atomic.Value(usize),
    num_threads: usize,

    io: std.Io,
    // Mutex and condition for waking workers
    mutex: std.Io.Mutex,
    work_available: std.Io.Condition,
    work_done: std.Io.Condition,

    fn init(nx: usize, ny: usize, nz: usize, num_threads: usize, io: std.Io) ThreadPoolState {
        return .{
            .io = io,
            .task = .none,
            .data = null,
            .inverse = false,
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .total_work = 0,
            .next_work_item = std.atomic.Value(usize).init(0),
            .task_generation = 0,
            .barrier_count = std.atomic.Value(usize).init(0),
            .num_threads = num_threads,
            .mutex = .init,
            .work_available = .init,
            .work_done = .init,
        };
    }
};

/// Parallel 3D FFT Plan with thread pool
pub const ParallelPlan3d = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    num_threads: usize,
    workspaces: []ThreadWorkspace,
    threads: []std.Thread,
    state: *ThreadPoolState,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        nx: usize,
        ny: usize,
        nz: usize,
    ) !ParallelPlan3d {
        return init_with_threads(allocator, io, nx, ny, nz, 0);
    }

    pub fn init_with_threads(
        allocator: std.mem.Allocator,
        io: std.Io,
        nx: usize,
        ny: usize,
        nz: usize,
        num_threads_hint: usize,
    ) !ParallelPlan3d {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;

        const cpu_count = std.Thread.getCpuCount() catch 4;
        const num_threads = if (num_threads_hint == 0) @min(cpu_count, 16) else num_threads_hint;

        // Allocate workspaces
        const workspaces = try allocator.alloc(ThreadWorkspace, num_threads);
        errdefer allocator.free(workspaces);

        var init_count: usize = 0;
        errdefer {
            for (0..init_count) |i| {
                workspaces[i].deinit();
            }
        }

        for (0..num_threads) |i| {
            workspaces[i] = try ThreadWorkspace.init(allocator, nx, ny, nz);
            init_count += 1;
        }

        // Allocate shared state
        const state = try allocator.create(ThreadPoolState);
        errdefer allocator.destroy(state);
        state.* = ThreadPoolState.init(nx, ny, nz, num_threads, io);

        // Allocate thread handles
        const threads = try allocator.alloc(std.Thread, num_threads);
        errdefer allocator.free(threads);

        var spawned: usize = 0;
        errdefer {
            // Signal shutdown to spawned threads
            state.mutex.lockUncancelable(state.io);
            state.task_generation += 1;
            state.task = .shutdown;
            state.work_available.broadcast(state.io);
            state.mutex.unlock(state.io);
            for (0..spawned) |i| {
                threads[i].join();
            }
        }

        // Spawn worker threads
        for (0..num_threads) |i| {
            threads[i] = try std.Thread.spawn(.{}, worker_thread, .{ state, &workspaces[i], i });
            spawned += 1;
        }

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .num_threads = num_threads,
            .workspaces = workspaces,
            .threads = threads,
            .state = state,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ParallelPlan3d) void {
        // Signal shutdown (increment generation so workers wake up)
        self.state.mutex.lockUncancelable(self.state.io);
        self.state.task_generation += 1;
        self.state.task = .shutdown;
        self.state.work_available.broadcast(self.state.io);
        self.state.mutex.unlock(self.state.io);

        // Wait for all threads to finish
        for (self.threads) |t| {
            t.join();
        }

        // Clean up
        for (self.workspaces) |*ws| {
            ws.deinit();
        }
        self.allocator.free(self.workspaces);
        self.allocator.free(self.threads);
        self.allocator.destroy(self.state);
    }

    pub fn forward(self: *ParallelPlan3d, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *ParallelPlan3d, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *ParallelPlan3d, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;

        if (data.len != nx * ny * nz) return;

        // FFT along x-axis (parallel over y*z)
        self.dispatch_task(.fft_x, data, inv, ny * nz);

        // FFT along y-axis (parallel over x*z)
        self.dispatch_task(.fft_y, data, inv, nx * nz);

        // FFT along z-axis (parallel over x*y)
        self.dispatch_task(.fft_z, data, inv, nx * ny);
    }

    fn dispatch_task(
        self: *ParallelPlan3d,
        task: TaskType,
        data: []Complex,
        inv: bool,
        total_work: usize,
    ) void {
        self.state.mutex.lockUncancelable(self.state.io);

        // Increment generation and set up task
        self.state.task_generation += 1;
        self.state.task = task;
        self.state.data = data;
        self.state.inverse = inv;
        self.state.total_work = total_work;
        self.state.next_work_item.store(0, .seq_cst);
        self.state.barrier_count.store(0, .seq_cst);

        // Wake all workers
        self.state.work_available.broadcast(self.state.io);
        self.state.mutex.unlock(self.state.io);

        // Wait for all workers to complete
        self.state.mutex.lockUncancelable(self.state.io);
        while (self.state.barrier_count.load(.seq_cst) < self.num_threads) {
            self.state.work_done.waitUncancelable(self.state.io, &self.state.mutex);
        }

        // Reset task (workers will wait for next generation)
        self.state.task = .none;
        self.state.mutex.unlock(self.state.io);
    }

    fn worker_thread(state: *ThreadPoolState, ws: *ThreadWorkspace, thread_id: usize) void {
        _ = thread_id;
        var last_generation: usize = 0;

        while (true) {
            // Wait for new work (generation must change)
            state.mutex.lockUncancelable(state.io);
            while (state.task == .none or state.task_generation == last_generation) {
                if (state.task == .shutdown) {
                    state.mutex.unlock(state.io);
                    return;
                }
                state.work_available.waitUncancelable(state.io, &state.mutex);
            }

            const task = state.task;
            const data = state.data;
            const inv = state.inverse;
            last_generation = state.task_generation;
            state.mutex.unlock(state.io);

            if (task == .shutdown) {
                return;
            }

            // Process work items using work-stealing pattern
            if (data) |d| {
                while (true) {
                    const item = state.next_work_item.fetchAdd(1, .seq_cst);
                    if (item >= state.total_work) break;

                    switch (task) {
                        .fft_x => process_x_axis(state, ws, d, inv, item),
                        .fft_y => process_y_axis(state, ws, d, inv, item),
                        .fft_z => process_z_axis(state, ws, d, inv, item),
                        else => {},
                    }
                }
            }

            // Signal completion via barrier
            const count = state.barrier_count.fetchAdd(1, .seq_cst) + 1;
            if (count == state.num_threads) {
                state.mutex.lockUncancelable(state.io);
                state.work_done.signal(state.io);
                state.mutex.unlock(state.io);
            }
        }
    }

    fn process_x_axis(
        state: *ThreadPoolState,
        ws: *ThreadWorkspace,
        data: []Complex,
        inv: bool,
        item: usize,
    ) void {
        const nx = state.nx;
        const ny = state.ny;
        const y = item % ny;
        const z = item / ny;
        const offset = nx * (y + ny * z);

        if (inv) {
            ws.plan_x.inverse(data[offset .. offset + nx]);
        } else {
            ws.plan_x.forward(data[offset .. offset + nx]);
        }
    }

    fn process_y_axis(
        state: *ThreadPoolState,
        ws: *ThreadWorkspace,
        data: []Complex,
        inv: bool,
        item: usize,
    ) void {
        const nx = state.nx;
        const ny = state.ny;
        const x = item % nx;
        const z = item / nx;

        // Gather
        for (0..ny) |j| {
            ws.scratch[j] = data[x + nx * (j + ny * z)];
        }

        if (inv) {
            ws.plan_y.inverse(ws.scratch[0..ny]);
        } else {
            ws.plan_y.forward(ws.scratch[0..ny]);
        }

        // Scatter
        for (0..ny) |j| {
            data[x + nx * (j + ny * z)] = ws.scratch[j];
        }
    }

    fn process_z_axis(
        state: *ThreadPoolState,
        ws: *ThreadWorkspace,
        data: []Complex,
        inv: bool,
        item: usize,
    ) void {
        const nx = state.nx;
        const ny = state.ny;
        const nz = state.nz;
        const x = item % nx;
        const y = item / nx;

        // Gather
        for (0..nz) |k| {
            ws.scratch[k] = data[x + nx * (y + ny * k)];
        }

        if (inv) {
            ws.plan_z.inverse(ws.scratch[0..nz]);
        } else {
            ws.plan_z.forward(ws.scratch[0..nz]);
        }

        // Scatter
        for (0..nz) |k| {
            data[x + nx * (y + ny * k)] = ws.scratch[k];
        }
    }
};

// ============== Tests ==============

test "ParallelPlan3d roundtrip" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var plan = try ParallelPlan3d.init_with_threads(allocator, io, 8, 8, 8, 4);
    defer plan.deinit();

    var data: [512]Complex = undefined;
    var original: [512]Complex = undefined;
    for (0..512) |i| {
        data[i] = Complex.init(@floatFromInt(i), 0);
        original[i] = data[i];
    }

    plan.forward(&data);
    plan.inverse(&data);

    for (0..512) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "ParallelPlan3d matches sequential" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const Plan3d = @import("fft.zig").Plan3d;

    var par_plan = try ParallelPlan3d.init_with_threads(allocator, io, 8, 8, 8, 4);
    defer par_plan.deinit();

    var seq_plan = try Plan3d.init(allocator, 8, 8, 8);
    defer seq_plan.deinit();

    var par_data: [512]Complex = undefined;
    var seq_data: [512]Complex = undefined;
    for (0..512) |i| {
        const val = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        par_data[i] = val;
        seq_data[i] = val;
    }

    par_plan.forward(&par_data);
    seq_plan.forward(&seq_data);

    for (0..512) |i| {
        try std.testing.expectApproxEqAbs(par_data[i].re, seq_data[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(par_data[i].im, seq_data[i].im, 1e-9);
    }
}

test "ParallelPlan3d multiple calls" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var plan = try ParallelPlan3d.init_with_threads(allocator, io, 8, 8, 8, 4);
    defer plan.deinit();

    var data: [512]Complex = undefined;

    // Run multiple FFTs to test thread pool reuse
    for (0..10) |iter| {
        for (0..512) |i| {
            data[i] = Complex.init(@floatFromInt((i + iter) % 17), 0);
        }
        const original = data;

        plan.forward(&data);
        plan.inverse(&data);

        for (0..512) |i| {
            try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
            try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
        }
    }
}
