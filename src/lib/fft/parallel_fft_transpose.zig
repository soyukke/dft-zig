//! Transpose-based Parallel 3D FFT implementation
//!
//! Uses transpose operations to ensure all FFTs operate on contiguous memory.
//! Pattern:
//!   FFT(x) -> Transpose(xyz->yxz) -> FFT(y)
//!          -> Transpose(yxz->zxy) -> FFT(z) -> Transpose(zxy->xyz)
//!
//! This maximizes cache efficiency by avoiding strided memory access.

const std = @import("std");
pub const Complex = @import("complex.zig").Complex;
const Plan1d = @import("fft.zig").Plan1d;

/// Thread-local workspace for parallel FFT
const ThreadWorkspace = struct {
    plan: Plan1d, // Single plan for the current axis size
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !ThreadWorkspace {
        const plan = try Plan1d.init(allocator, n);
        return .{
            .plan = plan,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadWorkspace) void {
        self.plan.deinit();
    }
};

/// Task type for thread pool
const TaskType = enum {
    none,
    fft_x, // FFT along x-axis
    fft_y, // FFT along y-axis
    fft_z, // FFT along z-axis
    transpose_xyz_yxz,
    transpose_yxz_zxy,
    transpose_zxy_xyz,
    shutdown,
};

/// Shared state for thread pool synchronization
const ThreadPoolState = struct {
    task: TaskType,
    src: ?[]Complex,
    dst: ?[]Complex,
    inverse: bool,
    nx: usize,
    ny: usize,
    nz: usize,
    axis_size: usize, // Current FFT axis size
    total_work: usize,
    next_work_item: std.atomic.Value(usize),
    task_generation: usize,
    barrier_count: std.atomic.Value(usize),
    num_threads: usize,
    io: std.Io,
    mutex: std.Io.Mutex,
    work_available: std.Io.Condition,
    work_done: std.Io.Condition,

    fn init(nx: usize, ny: usize, nz: usize, num_threads: usize, io: std.Io) ThreadPoolState {
        return .{
            .io = io,
            .task = .none,
            .src = null,
            .dst = null,
            .inverse = false,
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .axis_size = 0,
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

/// Transpose-based Parallel 3D FFT Plan
pub const TransposePlan3d = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    num_threads: usize,
    workspaces_x: []ThreadWorkspace,
    workspaces_y: []ThreadWorkspace,
    workspaces_z: []ThreadWorkspace,
    buffer: []Complex, // Temporary buffer for transpose
    threads: []std.Thread,
    state: *ThreadPoolState,
    allocator: std.mem.Allocator,

    /// Per-axis ThreadWorkspace buffers, plus incremental init state so we can
    /// release the partially initialized workspaces on failure.
    const WorkspacePools = struct {
        x: []ThreadWorkspace,
        y: []ThreadWorkspace,
        z: []ThreadWorkspace,
        init_x: usize = 0,
        init_y: usize = 0,
        init_z: usize = 0,

        fn deinit(self: *WorkspacePools, allocator: std.mem.Allocator) void {
            for (0..self.init_x) |i| self.x[i].deinit();
            for (0..self.init_y) |i| self.y[i].deinit();
            for (0..self.init_z) |i| self.z[i].deinit();
            allocator.free(self.x);
            allocator.free(self.y);
            allocator.free(self.z);
        }

        fn init_all(
            self: *WorkspacePools,
            allocator: std.mem.Allocator,
            nx: usize,
            ny: usize,
            nz: usize,
        ) !void {
            for (0..self.x.len) |i| {
                self.x[i] = try ThreadWorkspace.init(allocator, nx);
                self.init_x += 1;
            }
            for (0..self.y.len) |i| {
                self.y[i] = try ThreadWorkspace.init(allocator, ny);
                self.init_y += 1;
            }
            for (0..self.z.len) |i| {
                self.z[i] = try ThreadWorkspace.init(allocator, nz);
                self.init_z += 1;
            }
        }
    };

    fn alloc_workspace_pools(allocator: std.mem.Allocator, num_threads: usize) !WorkspacePools {
        const x = try allocator.alloc(ThreadWorkspace, num_threads);
        errdefer allocator.free(x);

        const y = try allocator.alloc(ThreadWorkspace, num_threads);
        errdefer allocator.free(y);

        const z = try allocator.alloc(ThreadWorkspace, num_threads);
        errdefer allocator.free(z);

        return .{ .x = x, .y = y, .z = z };
    }

    /// Spawn `threads.len` worker threads, joining any successfully spawned
    /// threads on partial-failure after signalling shutdown.
    fn spawn_worker_threads(
        state: *ThreadPoolState,
        threads: []std.Thread,
        pools: WorkspacePools,
    ) !void {
        var spawned: usize = 0;
        errdefer {
            state.mutex.lockUncancelable(state.io);
            state.task_generation += 1;
            state.task = .shutdown;
            state.work_available.broadcast(state.io);
            state.mutex.unlock(state.io);
            for (0..spawned) |i| {
                threads[i].join();
            }
        }
        for (0..threads.len) |i| {
            threads[i] = try std.Thread.spawn(.{}, worker_thread, .{
                state,
                &pools.x[i],
                &pools.y[i],
                &pools.z[i],
            });
            spawned += 1;
        }
    }

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        nx: usize,
        ny: usize,
        nz: usize,
    ) !TransposePlan3d {
        return init_with_threads(allocator, io, nx, ny, nz, 0);
    }

    pub fn init_with_threads(
        allocator: std.mem.Allocator,
        io: std.Io,
        nx: usize,
        ny: usize,
        nz: usize,
        num_threads_hint: usize,
    ) !TransposePlan3d {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;

        const cpu_count = std.Thread.getCpuCount() catch 4;
        const num_threads = if (num_threads_hint == 0) @min(cpu_count, 16) else num_threads_hint;

        var pools = try alloc_workspace_pools(allocator, num_threads);
        errdefer pools.deinit(allocator);

        try pools.init_all(allocator, nx, ny, nz);

        // Allocate transpose buffer
        const buffer = try allocator.alloc(Complex, nx * ny * nz);
        errdefer allocator.free(buffer);

        // Allocate shared state
        const state = try allocator.create(ThreadPoolState);
        errdefer allocator.destroy(state);

        state.* = ThreadPoolState.init(nx, ny, nz, num_threads, io);

        // Allocate thread handles
        const threads = try allocator.alloc(std.Thread, num_threads);
        errdefer allocator.free(threads);

        try spawn_worker_threads(state, threads, pools);

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .num_threads = num_threads,
            .workspaces_x = pools.x,
            .workspaces_y = pools.y,
            .workspaces_z = pools.z,
            .buffer = buffer,
            .threads = threads,
            .state = state,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TransposePlan3d) void {
        // Signal shutdown
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
        for (self.workspaces_x) |*ws| ws.deinit();
        for (self.workspaces_y) |*ws| ws.deinit();
        for (self.workspaces_z) |*ws| ws.deinit();
        self.allocator.free(self.workspaces_x);
        self.allocator.free(self.workspaces_y);
        self.allocator.free(self.workspaces_z);
        self.allocator.free(self.buffer);
        self.allocator.free(self.threads);
        self.allocator.destroy(self.state);
    }

    pub fn forward(self: *TransposePlan3d, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *TransposePlan3d, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *TransposePlan3d, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;

        if (data.len != nx * ny * nz) return;

        // Step 1: FFT along x-axis (data is xyz, x is contiguous)
        self.dispatch_fft(.fft_x, data, nx, ny * nz, inv);

        // Step 2: Transpose xyz -> yxz
        self.dispatch_transpose(.transpose_xyz_yxz, data, self.buffer);

        // Step 3: FFT along y-axis (buffer is yxz, y is contiguous)
        self.dispatch_fft(.fft_y, self.buffer, ny, nx * nz, inv);

        // Step 4: Transpose yxz -> zxy
        self.dispatch_transpose(.transpose_yxz_zxy, self.buffer, data);

        // Step 5: FFT along z-axis (data is zxy, z is contiguous)
        self.dispatch_fft(.fft_z, data, nz, nx * ny, inv);

        // Step 6: Transpose zxy -> xyz
        self.dispatch_transpose(.transpose_zxy_xyz, data, self.buffer);

        // Copy result back to data
        @memcpy(data, self.buffer);
    }

    fn dispatch_fft(
        self: *TransposePlan3d,
        task: TaskType,
        data: []Complex,
        axis_size: usize,
        num_ffts: usize,
        inv: bool,
    ) void {
        self.state.mutex.lockUncancelable(self.state.io);

        self.state.task_generation += 1;
        self.state.task = task;
        self.state.src = data;
        self.state.dst = null;
        self.state.inverse = inv;
        self.state.axis_size = axis_size;
        self.state.total_work = num_ffts;
        self.state.next_work_item.store(0, .seq_cst);
        self.state.barrier_count.store(0, .seq_cst);

        self.state.work_available.broadcast(self.state.io);
        self.state.mutex.unlock(self.state.io);

        // Wait for completion
        self.state.mutex.lockUncancelable(self.state.io);
        while (self.state.barrier_count.load(.seq_cst) < self.num_threads) {
            self.state.work_done.waitUncancelable(self.state.io, &self.state.mutex);
        }
        self.state.task = .none;
        self.state.mutex.unlock(self.state.io);
    }

    fn dispatch_transpose(
        self: *TransposePlan3d,
        task: TaskType,
        src: []Complex,
        dst: []Complex,
    ) void {
        self.state.mutex.lockUncancelable(self.state.io);

        self.state.task_generation += 1;
        self.state.task = task;
        self.state.src = src;
        self.state.dst = dst;
        self.state.total_work = self.compute_transpose_work(task);
        self.state.next_work_item.store(0, .seq_cst);
        self.state.barrier_count.store(0, .seq_cst);

        self.state.work_available.broadcast(self.state.io);
        self.state.mutex.unlock(self.state.io);

        // Wait for completion
        self.state.mutex.lockUncancelable(self.state.io);
        while (self.state.barrier_count.load(.seq_cst) < self.num_threads) {
            self.state.work_done.waitUncancelable(self.state.io, &self.state.mutex);
        }
        self.state.task = .none;
        self.state.mutex.unlock(self.state.io);
    }

    fn compute_transpose_work(self: *TransposePlan3d, task: TaskType) usize {
        return switch (task) {
            .transpose_xyz_yxz => self.ny * self.nz, // parallelize over y*z
            .transpose_yxz_zxy => self.nz * self.nx, // parallelize over z*x
            .transpose_zxy_xyz => self.nx * self.ny, // parallelize over x*y
            else => 0,
        };
    }

    /// Execute a single work item for the current task on behalf of one worker.
    fn run_work_item(
        state: *ThreadPoolState,
        task: @TypeOf(state.task),
        src: @TypeOf(state.src),
        dst: @TypeOf(state.dst),
        inv: bool,
        axis_size: usize,
        item: usize,
        ws_x: *ThreadWorkspace,
        ws_y: *ThreadWorkspace,
        ws_z: *ThreadWorkspace,
    ) void {
        switch (task) {
            .fft_x => if (src) |s| process_fft_axis(s, axis_size, item, inv, ws_x),
            .fft_y => if (src) |s| process_fft_axis(s, axis_size, item, inv, ws_y),
            .fft_z => if (src) |s| process_fft_axis(s, axis_size, item, inv, ws_z),
            .transpose_xyz_yxz => if (src != null and dst != null)
                transpose_xyz_to_yxz(state, src.?, dst.?, item),
            .transpose_yxz_zxy => if (src != null and dst != null)
                transpose_yxz_to_zxy(state, src.?, dst.?, item),
            .transpose_zxy_xyz => if (src != null and dst != null)
                transpose_zxy_to_xyz(state, src.?, dst.?, item),
            else => {},
        }
    }

    fn worker_thread(
        state: *ThreadPoolState,
        ws_x: *ThreadWorkspace,
        ws_y: *ThreadWorkspace,
        ws_z: *ThreadWorkspace,
    ) void {
        var last_generation: usize = 0;

        while (true) {
            state.mutex.lockUncancelable(state.io);
            while (state.task == .none or state.task_generation == last_generation) {
                if (state.task == .shutdown) {
                    state.mutex.unlock(state.io);
                    return;
                }
                state.work_available.waitUncancelable(state.io, &state.mutex);
            }

            const task = state.task;
            const src = state.src;
            const dst = state.dst;
            const inv = state.inverse;
            const axis_size = state.axis_size;
            last_generation = state.task_generation;
            state.mutex.unlock(state.io);

            if (task == .shutdown) return;

            // Process work items
            while (true) {
                const item = state.next_work_item.fetchAdd(1, .seq_cst);
                if (item >= state.total_work) break;
                run_work_item(state, task, src, dst, inv, axis_size, item, ws_x, ws_y, ws_z);
            }

            // Barrier
            const count = state.barrier_count.fetchAdd(1, .seq_cst) + 1;
            if (count == state.num_threads) {
                state.mutex.lockUncancelable(state.io);
                state.work_done.signal(state.io);
                state.mutex.unlock(state.io);
            }
        }
    }

    fn process_fft_axis(
        data: []Complex,
        axis_size: usize,
        item: usize,
        inv: bool,
        ws: *ThreadWorkspace,
    ) void {
        const offset = item * axis_size;
        if (inv) {
            ws.plan.inverse(data[offset .. offset + axis_size]);
        } else {
            ws.plan.forward(data[offset .. offset + axis_size]);
        }
    }

    // xyz[x + nx*(y + ny*z)] -> yxz[y + ny*(x + nx*z)]
    fn transpose_xyz_to_yxz(
        state: *ThreadPoolState,
        src: []Complex,
        dst: []Complex,
        item: usize,
    ) void {
        const nx = state.nx;
        const ny = state.ny;
        const y = item % ny;
        const z = item / ny;

        for (0..nx) |x| {
            const src_idx = x + nx * (y + ny * z);
            const dst_idx = y + ny * (x + nx * z);
            dst[dst_idx] = src[src_idx];
        }
    }

    // yxz[y + ny*(x + nx*z)] -> zxy[z + nz*(x + nx*y)]
    fn transpose_yxz_to_zxy(
        state: *ThreadPoolState,
        src: []Complex,
        dst: []Complex,
        item: usize,
    ) void {
        const nx = state.nx;
        const ny = state.ny;
        const nz = state.nz;
        const z = item % nz;
        const x = item / nz;

        for (0..ny) |y| {
            const src_idx = y + ny * (x + nx * z);
            const dst_idx = z + nz * (x + nx * y);
            dst[dst_idx] = src[src_idx];
        }
    }

    // zxy[z + nz*(x + nx*y)] -> xyz[x + nx*(y + ny*z)]
    fn transpose_zxy_to_xyz(
        state: *ThreadPoolState,
        src: []Complex,
        dst: []Complex,
        item: usize,
    ) void {
        const nx = state.nx;
        const ny = state.ny;
        const nz = state.nz;
        const x = item % nx;
        const y = item / nx;

        for (0..nz) |z| {
            const src_idx = z + nz * (x + nx * y);
            const dst_idx = x + nx * (y + ny * z);
            dst[dst_idx] = src[src_idx];
        }
    }
};

// ============== Tests ==============

test "TransposePlan3d roundtrip" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var plan = try TransposePlan3d.init_with_threads(allocator, io, 8, 8, 8, 4);
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

test "TransposePlan3d matches sequential" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const Plan3d = @import("fft.zig").Plan3d;

    var trans_plan = try TransposePlan3d.init_with_threads(allocator, io, 8, 8, 8, 4);
    defer trans_plan.deinit();

    var seq_plan = try Plan3d.init(allocator, 8, 8, 8);
    defer seq_plan.deinit();

    var trans_data: [512]Complex = undefined;
    var seq_data: [512]Complex = undefined;
    for (0..512) |i| {
        const val = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        trans_data[i] = val;
        seq_data[i] = val;
    }

    trans_plan.forward(&trans_data);
    seq_plan.forward(&seq_data);

    for (0..512) |i| {
        try std.testing.expectApproxEqAbs(trans_data[i].re, seq_data[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(trans_data[i].im, seq_data[i].im, 1e-9);
    }
}

test "TransposePlan3d non-cubic" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const Plan3d = @import("fft.zig").Plan3d;

    // Test with 24x24x24 (non-power-of-2)
    var trans_plan = try TransposePlan3d.init_with_threads(allocator, io, 24, 24, 24, 4);
    defer trans_plan.deinit();

    var seq_plan = try Plan3d.init(allocator, 24, 24, 24);
    defer seq_plan.deinit();

    const size = 24 * 24 * 24;
    var trans_data = try allocator.alloc(Complex, size);
    defer allocator.free(trans_data);

    var seq_data = try allocator.alloc(Complex, size);
    defer allocator.free(seq_data);

    for (0..size) |i| {
        const val = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        trans_data[i] = val;
        seq_data[i] = val;
    }

    trans_plan.forward(trans_data);
    seq_plan.forward(seq_data);

    for (0..size) |i| {
        try std.testing.expectApproxEqAbs(trans_data[i].re, seq_data[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(trans_data[i].im, seq_data[i].im, 1e-9);
    }
}
