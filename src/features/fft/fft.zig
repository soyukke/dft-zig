//! FFT module for DFT-Zig
//!
//! Wrapper around the standalone FFT library (src/lib/fft/)
//! to provide compatibility with the existing math.Complex type.
//!
//! Now supports arbitrary grid sizes via Bluestein's algorithm.
//! Supports vDSP backend for macOS Accelerate framework.

const std = @import("std");
const builtin = @import("builtin");
const math = @import("../math/math.zig");
const config_mod = @import("../config/config.zig");

// Import the standalone FFT library
const fft_lib = @import("../../lib/fft/fft.zig");
const rfft3d_lib = @import("../../lib/fft/rfft3d.zig");
const vdsp_fft = fft_lib.vdsp_fft;
const parallel_fft = fft_lib.parallel_fft;
const parallel_fft_transpose = fft_lib.parallel_fft_transpose;
const parallel_fft24 = fft_lib.parallel_fft24;
const fftw_fft = fft_lib.fftw_fft;
const metal_fft_lib = fft_lib.metal_fft;

/// Complex type from the FFT library (public for SCF use)
pub const LibComplex = fft_lib.Complex;

/// Check if value is power of two.
pub fn isPowerOfTwo(n: usize) bool {
    return fft_lib.isPowerOfTwo(n);
}

/// Convert math.Complex slice to LibComplex slice (zero-copy, same memory layout).
/// Note: This relies on both types having the same memory layout (two f64 values).
fn toLibComplex(data: []math.Complex) []LibComplex {
    const ptr: [*]LibComplex = @ptrCast(data.ptr);
    return ptr[0..data.len];
}

/// FFT Backend type
pub const FftBackend = config_mod.FftBackend;

/// 3D FFT Plan - supports arbitrary sizes.
/// Can use either pure Zig FFT or vDSP (macOS Accelerate).
pub const Fft3dPlan = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    backend: FftBackend,
    plan: PlanUnion,
    allocator: std.mem.Allocator,

    const PlanUnion = union(FftBackend) {
        zig: fft_lib.Plan3d,
        zig_parallel: parallel_fft.ParallelPlan3d,
        zig_transpose: parallel_fft_transpose.TransposePlan3d,
        zig_comptime24: parallel_fft24.ParallelPlan3d24,
        vdsp: if (builtin.os.tag == .macos) vdsp_fft.VdspPlan3d else void,
        fftw: fftw_fft.FftwPlan3d,
        metal: if (builtin.os.tag == .macos) metal_fft_lib.MetalPlan3d else void,
    };

    /// Initialize with default Zig backend
    pub fn init(alloc: std.mem.Allocator, io: std.Io, nx: usize, ny: usize, nz: usize) !Fft3dPlan {
        return initWithBackend(alloc, io, nx, ny, nz, .zig);
    }

    /// Initialize with specified backend
    pub fn initWithBackend(
        alloc: std.mem.Allocator,
        io: std.Io,
        nx: usize,
        ny: usize,
        nz: usize,
        backend: FftBackend,
    ) !Fft3dPlan {
        return switch (backend) {
            .zig => initZigPlan(alloc, nx, ny, nz),
            .zig_parallel => initZigParallelPlan(alloc, io, nx, ny, nz),
            .zig_transpose => initZigTransposePlan(alloc, io, nx, ny, nz),
            .zig_comptime24 => initZigComptime24Plan(alloc, io, nx, ny, nz),
            .fftw => initFftwPlan(alloc, nx, ny, nz),
            .vdsp => initVdspPlan(alloc, nx, ny, nz),
            .metal => initMetalPlan(alloc, nx, ny, nz),
        };
    }

    fn initZigPlan(alloc: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !Fft3dPlan {
        const plan = try fft_lib.Plan3d.init(alloc, nx, ny, nz);
        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .backend = .zig,
            .plan = .{ .zig = plan },
            .allocator = alloc,
        };
    }

    fn initZigParallelPlan(
        alloc: std.mem.Allocator,
        io: std.Io,
        nx: usize,
        ny: usize,
        nz: usize,
    ) !Fft3dPlan {
        const plan = try parallel_fft.ParallelPlan3d.init(alloc, io, nx, ny, nz);
        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .backend = .zig_parallel,
            .plan = .{ .zig_parallel = plan },
            .allocator = alloc,
        };
    }

    fn initZigTransposePlan(
        alloc: std.mem.Allocator,
        io: std.Io,
        nx: usize,
        ny: usize,
        nz: usize,
    ) !Fft3dPlan {
        const plan = try parallel_fft_transpose.TransposePlan3d.init(alloc, io, nx, ny, nz);
        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .backend = .zig_transpose,
            .plan = .{ .zig_transpose = plan },
            .allocator = alloc,
        };
    }

    fn initZigComptime24Plan(
        alloc: std.mem.Allocator,
        io: std.Io,
        nx: usize,
        ny: usize,
        nz: usize,
    ) !Fft3dPlan {
        // Only supports 24×24×24, fall back to zig_parallel for other sizes
        if (nx != 24 or ny != 24 or nz != 24) {
            return initZigParallelPlan(alloc, io, nx, ny, nz);
        }
        const plan = try parallel_fft24.ParallelPlan3d24.init(alloc, io, nx, ny, nz);
        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .backend = .zig_comptime24,
            .plan = .{ .zig_comptime24 = plan },
            .allocator = alloc,
        };
    }

    fn initFftwPlan(alloc: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !Fft3dPlan {
        const plan = try fftw_fft.FftwPlan3d.init(alloc, nx, ny, nz);
        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .backend = .fftw,
            .plan = .{ .fftw = plan },
            .allocator = alloc,
        };
    }

    fn initVdspPlan(alloc: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !Fft3dPlan {
        if (comptime builtin.os.tag != .macos) {
            return error.VdspNotAvailable;
        }
        // vDSP only supports power-of-2 sizes — fall back to the pure-Zig FFT.
        if (!isPowerOfTwo(nx) or !isPowerOfTwo(ny) or !isPowerOfTwo(nz)) {
            return initZigPlan(alloc, nx, ny, nz);
        }
        const plan = try vdsp_fft.VdspPlan3d.init(alloc, nx, ny, nz);
        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .backend = .vdsp,
            .plan = .{ .vdsp = plan },
            .allocator = alloc,
        };
    }

    fn initMetalPlan(alloc: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !Fft3dPlan {
        if (comptime builtin.os.tag != .macos) {
            return error.MetalNotAvailable;
        }
        const plan = try metal_fft_lib.MetalPlan3d.init(alloc, nx, ny, nz);
        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .backend = .metal,
            .plan = .{ .metal = plan },
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *Fft3dPlan, alloc: std.mem.Allocator) void {
        _ = alloc;
        switch (self.plan) {
            .zig => |*p| p.deinit(),
            .zig_parallel => |*p| p.deinit(),
            .zig_transpose => |*p| p.deinit(),
            .zig_comptime24 => |*p| p.deinit(),
            .fftw => |*p| p.deinit(),
            .vdsp => |*p| {
                if (comptime builtin.os.tag == .macos) {
                    p.deinit();
                }
            },
            .metal => |*p| {
                if (comptime builtin.os.tag == .macos) {
                    p.deinit();
                }
            },
        }
    }

    pub fn forward(self: *Fft3dPlan, data: []math.Complex) void {
        switch (self.plan) {
            .zig => |*p| p.forward(toLibComplex(data)),
            .zig_parallel => |*p| p.forward(toLibComplex(data)),
            .zig_transpose => |*p| p.forward(toLibComplex(data)),
            .zig_comptime24 => |*p| p.forward(toLibComplex(data)),
            .fftw => |*p| p.forward(toLibComplex(data)),
            .vdsp => |*p| {
                if (comptime builtin.os.tag == .macos) {
                    p.forward(toLibComplex(data));
                }
            },
            .metal => |*p| {
                if (comptime builtin.os.tag == .macos) {
                    p.forward(toLibComplex(data));
                }
            },
        }
    }

    pub fn inverse(self: *Fft3dPlan, data: []math.Complex) void {
        switch (self.plan) {
            .zig => |*p| p.inverse(toLibComplex(data)),
            .zig_parallel => |*p| p.inverse(toLibComplex(data)),
            .zig_transpose => |*p| p.inverse(toLibComplex(data)),
            .zig_comptime24 => |*p| p.inverse(toLibComplex(data)),
            .fftw => |*p| p.inverse(toLibComplex(data)),
            .vdsp => |*p| {
                if (comptime builtin.os.tag == .macos) {
                    p.inverse(toLibComplex(data));
                }
            },
            .metal => |*p| {
                if (comptime builtin.os.tag == .macos) {
                    p.inverse(toLibComplex(data));
                }
            },
        }
    }
};

/// Forward 3D FFT in-place using plan.
pub fn fft3dForwardInPlacePlan(plan: *Fft3dPlan, data: []math.Complex) !void {
    plan.forward(data);
}

/// Inverse 3D FFT in-place using plan.
pub fn fft3dInverseInPlacePlan(plan: *Fft3dPlan, data: []math.Complex) !void {
    plan.inverse(data);
}

/// Forward 3D FFT in-place (creates temporary plan).
pub fn fft3dForwardInPlace(
    alloc: std.mem.Allocator,
    data: []math.Complex,
    nx: usize,
    ny: usize,
    nz: usize,
) !void {
    if (data.len != nx * ny * nz) return error.InvalidGrid;

    var plan = try fft_lib.Plan3d.init(alloc, nx, ny, nz);
    defer plan.deinit();

    plan.forward(toLibComplex(data));
}

/// Inverse 3D FFT in-place (creates temporary plan).
pub fn fft3dInverseInPlace(
    alloc: std.mem.Allocator,
    data: []math.Complex,
    nx: usize,
    ny: usize,
    nz: usize,
) !void {
    if (data.len != nx * ny * nz) return error.InvalidGrid;

    var plan = try fft_lib.Plan3d.init(alloc, nx, ny, nz);
    defer plan.deinit();

    plan.inverse(toLibComplex(data));
}

// ============== 1D FFT functions (for compatibility) ==============

/// 3D Real FFT Plan - for real-valued data like electron density
/// Output size is (nx/2+1) * ny * nz complex values
pub const RealFft3dPlan = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    nx_complex: usize, // = nx/2 + 1
    plan: rfft3d_lib.RealPlan3d,
    allocator: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !RealFft3dPlan {
        if (nx % 2 != 0) return error.InvalidSize; // nx must be even for RFFT
        const plan = try rfft3d_lib.RealPlan3d.init(alloc, nx, ny, nz);
        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .nx_complex = nx / 2 + 1,
            .plan = plan,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *RealFft3dPlan, alloc: std.mem.Allocator) void {
        _ = alloc;
        self.plan.deinit();
    }

    /// Forward RFFT: real[nx*ny*nz] -> complex[(nx/2+1)*ny*nz]
    pub fn forward(
        self: *RealFft3dPlan,
        real_input: []const f64,
        complex_output: []LibComplex,
    ) void {
        self.plan.forward(real_input, complex_output);
    }

    /// Inverse RFFT: complex[(nx/2+1)*ny*nz] -> real[nx*ny*nz]
    /// Note: complex_input is modified during computation
    pub fn inverse(self: *RealFft3dPlan, complex_input: []LibComplex, real_output: []f64) void {
        self.plan.inverse(complex_input, real_output);
    }

    /// Get the complex output size
    pub fn complexSize(self: *const RealFft3dPlan) usize {
        return self.nx_complex * self.ny * self.nz;
    }

    /// Get the real input size
    pub fn realSize(self: *const RealFft3dPlan) usize {
        return self.nx * self.ny * self.nz;
    }
};

/// Convert LibComplex slice to math.Complex slice (zero-copy)
fn toMathComplex(data: []LibComplex) []math.Complex {
    const ptr: [*]math.Complex = @ptrCast(data.ptr);
    return ptr[0..data.len];
}

/// 1D FFT Plan
pub const Fft1dPlan = struct {
    plan: fft_lib.Plan1d,

    pub fn init(alloc: std.mem.Allocator, n: usize) !Fft1dPlan {
        const plan = try fft_lib.Plan1d.init(alloc, n);
        return .{ .plan = plan };
    }

    pub fn deinit(self: *Fft1dPlan, alloc: std.mem.Allocator) void {
        _ = alloc;
        self.plan.deinit();
    }
};

/// In-place 1D FFT.
pub fn fft1d(data: []math.Complex, inverse: bool) void {
    // For simple 1D FFT without plan, we need to check if power of 2
    // and use the appropriate algorithm
    const lib_data = toLibComplex(data);
    const n = data.len;

    if (isPowerOfTwo(n)) {
        // Use radix-2 directly
        const radix2 = @import("../../lib/fft/radix2.zig");
        if (inverse) {
            radix2.fft1d(lib_data, true);
        } else {
            radix2.fft1d(lib_data, false);
        }
    } else {
        // For non-power-of-2, we'd need to create a plan
        // This is a limitation - callers should use Plan1d for non-power-of-2
        @panic("fft1d requires power-of-2 size. Use Fft1dPlan for arbitrary sizes.");
    }
}

/// 1D FFT using pre-computed plan.
pub fn fft1dPlanned(plan: Fft1dPlan, data: []math.Complex, inverse: bool) void {
    const lib_data = toLibComplex(data);
    if (inverse) {
        plan.plan.inverse(lib_data);
    } else {
        plan.plan.forward(lib_data);
    }
}

// ============== Tests ==============

test "Fft3dPlan arbitrary size" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Test with non-power-of-2 size (24x24x24)
    var plan = try Fft3dPlan.init(allocator, io, 6, 6, 6);
    defer plan.deinit(allocator);

    var data: [216]math.Complex = undefined;
    for (0..216) |i| {
        data[i] = math.complex.init(@floatFromInt(i), 0);
    }

    const original = data;

    plan.forward(&data);
    plan.inverse(&data);

    // Should recover original
    for (0..216) |i| {
        try std.testing.expectApproxEqAbs(data[i].r, original[i].r, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].i, original[i].i, 1e-9);
    }
}

test "fft3dForwardInPlace arbitrary size" {
    const allocator = std.testing.allocator;

    var data: [24]math.Complex = undefined;
    for (0..24) |i| {
        data[i] = math.complex.init(@floatFromInt(i), 0);
    }

    const original = data;

    // 2x3x4 grid (all non-power-of-2 except 2 and 4)
    try fft3dForwardInPlace(allocator, &data, 2, 3, 4);
    try fft3dInverseInPlace(allocator, &data, 2, 3, 4);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].r, original[i].r, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].i, original[i].i, 1e-9);
    }
}

test "RealFft3dPlan roundtrip" {
    const allocator = std.testing.allocator;

    const nx = 8;
    const ny = 8;
    const nz = 8;
    const real_size = nx * ny * nz;

    var plan = try RealFft3dPlan.init(allocator, nx, ny, nz);
    defer plan.deinit(allocator);

    // Original real data
    var original: [real_size]f64 = undefined;
    for (0..real_size) |i| {
        original[i] = @floatFromInt(i % 17);
    }

    // Forward
    const complex_size = plan.complexSize();
    const spectrum = try allocator.alloc(LibComplex, complex_size);
    defer allocator.free(spectrum);

    plan.forward(&original, spectrum);

    // Inverse
    var recovered: [real_size]f64 = undefined;
    plan.inverse(spectrum, &recovered);

    // Check
    for (0..real_size) |i| {
        try std.testing.expectApproxEqAbs(recovered[i], original[i], 1e-9);
    }
}

test "RealFft3dPlan vs complex Fft3dPlan" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const nx = 8;
    const ny = 8;
    const nz = 8;
    const real_size = nx * ny * nz;
    const nx_c = nx / 2 + 1;

    var rfft_plan = try RealFft3dPlan.init(allocator, nx, ny, nz);
    defer rfft_plan.deinit(allocator);

    var cfft_plan = try Fft3dPlan.init(allocator, io, nx, ny, nz);
    defer cfft_plan.deinit(allocator);

    // Test input
    var real_input: [real_size]f64 = undefined;
    for (0..real_size) |i| {
        real_input[i] = @floatFromInt((i * 7) % 23);
    }

    // RFFT
    const complex_size = rfft_plan.complexSize();
    const rfft_output = try allocator.alloc(LibComplex, complex_size);
    defer allocator.free(rfft_output);

    rfft_plan.forward(&real_input, rfft_output);

    // Complex FFT
    var cfft_data: [real_size]math.Complex = undefined;
    for (0..real_size) |i| {
        cfft_data[i] = math.complex.init(real_input[i], 0);
    }
    cfft_plan.forward(&cfft_data);

    // Compare the non-redundant part
    for (0..nz) |iz| {
        for (0..ny) |iy| {
            for (0..nx_c) |ix| {
                const rfft_idx = ix + nx_c * (iy + ny * iz);
                const cfft_idx = ix + nx * (iy + ny * iz);
                const rfft_val = toMathComplex(rfft_output)[rfft_idx];
                try std.testing.expectApproxEqAbs(rfft_val.r, cfft_data[cfft_idx].r, 1e-9);
                try std.testing.expectApproxEqAbs(rfft_val.i, cfft_data[cfft_idx].i, 1e-9);
            }
        }
    }
}
