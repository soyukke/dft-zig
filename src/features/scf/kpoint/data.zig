const std = @import("std");
const math = @import("../../math/math.zig");
const symmetry = @import("../../symmetry/symmetry.zig");

pub const KPoint = symmetry.KPoint;

pub const KpointCache = struct {
    n: usize = 0,
    nbands: usize = 0,
    vectors: []math.Complex = &[_]math.Complex{},
    eigenvalues: []f64 = &[_]f64{},

    pub fn deinit(self: *KpointCache) void {
        if (self.vectors.len > 0) {
            std.heap.c_allocator.free(self.vectors);
        }
        if (self.eigenvalues.len > 0) {
            std.heap.c_allocator.free(self.eigenvalues);
        }
        self.* = .{};
    }

    pub fn store(self: *KpointCache, n: usize, nbands: usize, values: []const math.Complex) !void {
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

    pub fn store_eigenvalues(self: *KpointCache, values: []const f64) !void {
        if (self.eigenvalues.len != values.len) {
            if (self.eigenvalues.len > 0) {
                std.heap.c_allocator.free(self.eigenvalues);
            }
            self.eigenvalues = try std.heap.c_allocator.alloc(f64, values.len);
        }
        @memcpy(self.eigenvalues, values);
    }
};

pub const KpointEigenData = struct {
    kpoint: KPoint,
    basis_len: usize,
    nbands: usize,
    values: []f64,
    vectors: []math.Complex,
    nonlocal: ?[]f64,

    pub fn deinit(self: *KpointEigenData, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
        if (self.vectors.len > 0) alloc.free(self.vectors);
        if (self.nonlocal) |values| alloc.free(values);
        self.* = undefined;
    }
};
