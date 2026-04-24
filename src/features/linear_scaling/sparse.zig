const std = @import("std");

pub const Triplet = struct {
    row: usize,
    col: usize,
    value: f64,
};

fn empty_csr(alloc: std.mem.Allocator, nrows: usize, ncols: usize) !CsrMatrix {
    const row_ptr = try alloc.alloc(usize, nrows + 1);
    @memset(row_ptr, 0);
    return .{
        .nrows = nrows,
        .ncols = ncols,
        .row_ptr = row_ptr,
        .col_idx = &.{},
        .values = &.{},
    };
}

pub const CsrMatrix = struct {
    nrows: usize,
    ncols: usize,
    row_ptr: []usize,
    col_idx: []usize,
    values: []f64,

    pub fn init_from_triplets(
        alloc: std.mem.Allocator,
        nrows: usize,
        ncols: usize,
        triplets: []const Triplet,
    ) !CsrMatrix {
        if (nrows == 0 or ncols == 0) return error.InvalidShape;
        if (triplets.len == 0) return empty_csr(alloc, nrows, ncols);

        const sorted = try alloc.alloc(Triplet, triplets.len);
        defer alloc.free(sorted);

        @memcpy(sorted, triplets);
        std.sort.block(Triplet, sorted, {}, triplet_less);

        var merged: std.ArrayList(Triplet) = .empty;
        defer merged.deinit(alloc);

        for (sorted) |t| {
            if (t.row >= nrows or t.col >= ncols) return error.IndexOutOfBounds;
            if (merged.items.len > 0) {
                const last = &merged.items[merged.items.len - 1];
                if (last.row == t.row and last.col == t.col) {
                    last.value += t.value;
                    continue;
                }
            }
            try merged.append(alloc, t);
        }

        const row_ptr = try alloc.alloc(usize, nrows + 1);
        errdefer alloc.free(row_ptr);
        @memset(row_ptr, 0);
        for (merged.items) |t| {
            row_ptr[t.row + 1] += 1;
        }
        var i: usize = 1;
        while (i < row_ptr.len) : (i += 1) {
            row_ptr[i] += row_ptr[i - 1];
        }

        const nnz = merged.items.len;
        const col_idx = try alloc.alloc(usize, nnz);
        errdefer alloc.free(col_idx);
        const values = try alloc.alloc(f64, nnz);
        errdefer alloc.free(values);

        var next = try alloc.alloc(usize, nrows);
        defer alloc.free(next);

        @memcpy(next, row_ptr[0..nrows]);

        for (merged.items) |t| {
            const offset = next[t.row];
            col_idx[offset] = t.col;
            values[offset] = t.value;
            next[t.row] += 1;
        }

        return .{
            .nrows = nrows,
            .ncols = ncols,
            .row_ptr = row_ptr,
            .col_idx = col_idx,
            .values = values,
        };
    }

    pub fn deinit(self: *CsrMatrix, alloc: std.mem.Allocator) void {
        if (self.row_ptr.len > 0) alloc.free(self.row_ptr);
        if (self.col_idx.len > 0) alloc.free(self.col_idx);
        if (self.values.len > 0) alloc.free(self.values);
        self.* = undefined;
    }

    pub fn value_at(self: *const CsrMatrix, row: usize, col: usize) f64 {
        if (row >= self.nrows or col >= self.ncols) return 0.0;
        const start = self.row_ptr[row];
        const end = self.row_ptr[row + 1];
        var idx: usize = start;
        while (idx < end) : (idx += 1) {
            if (self.col_idx[idx] == col) return self.values[idx];
        }
        return 0.0;
    }

    pub fn mul_vec(self: *const CsrMatrix, x: []const f64, out: []f64) !void {
        if (x.len != self.ncols or out.len != self.nrows) return error.InvalidShape;
        @memset(out, 0.0);
        var row: usize = 0;
        while (row < self.nrows) : (row += 1) {
            const start = self.row_ptr[row];
            const end = self.row_ptr[row + 1];
            var sum: f64 = 0.0;
            var idx: usize = start;
            while (idx < end) : (idx += 1) {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            out[row] = sum;
        }
    }
};

pub fn diagonal_values(alloc: std.mem.Allocator, matrix: CsrMatrix) ![]f64 {
    if (matrix.nrows == 0 or matrix.ncols == 0) return error.InvalidShape;
    const n = @min(matrix.nrows, matrix.ncols);
    const out = try alloc.alloc(f64, n);
    @memset(out, 0.0);
    var row: usize = 0;
    while (row < n) : (row += 1) {
        const start = matrix.row_ptr[row];
        const end = matrix.row_ptr[row + 1];
        var idx: usize = start;
        while (idx < end) : (idx += 1) {
            if (matrix.col_idx[idx] == row) {
                out[row] = matrix.values[idx];
                break;
            }
        }
    }
    return out;
}

pub fn scale_in_place(matrix: *CsrMatrix, factor: f64) void {
    var idx: usize = 0;
    while (idx < matrix.values.len) : (idx += 1) {
        matrix.values[idx] *= factor;
    }
}

pub fn trace(matrix: CsrMatrix) f64 {
    if (matrix.nrows == 0 or matrix.ncols == 0) return 0.0;
    const n = @min(matrix.nrows, matrix.ncols);
    var sum: f64 = 0.0;
    var row: usize = 0;
    while (row < n) : (row += 1) {
        const start = matrix.row_ptr[row];
        const end = matrix.row_ptr[row + 1];
        var idx: usize = start;
        while (idx < end) : (idx += 1) {
            if (matrix.col_idx[idx] == row) {
                sum += matrix.values[idx];
                break;
            }
        }
    }
    return sum;
}

pub fn trace_product(a: CsrMatrix, b: CsrMatrix) !f64 {
    if (a.ncols != b.nrows or a.nrows != b.ncols) return error.InvalidShape;
    var sum: f64 = 0.0;
    var row: usize = 0;
    while (row < a.nrows) : (row += 1) {
        const start = a.row_ptr[row];
        const end = a.row_ptr[row + 1];
        var idx: usize = start;
        while (idx < end) : (idx += 1) {
            const col = a.col_idx[idx];
            const b_value = b.value_at(col, row);
            if (b_value != 0.0) {
                sum += a.values[idx] * b_value;
            }
        }
    }
    return sum;
}

pub fn diagonal(alloc: std.mem.Allocator, n: usize, value: f64) !CsrMatrix {
    if (n == 0) return error.InvalidShape;
    const row_ptr = try alloc.alloc(usize, n + 1);
    errdefer alloc.free(row_ptr);
    const col_idx = try alloc.alloc(usize, n);
    errdefer alloc.free(col_idx);
    const values = try alloc.alloc(f64, n);
    errdefer alloc.free(values);
    row_ptr[0] = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        row_ptr[i + 1] = i + 1;
        col_idx[i] = i;
        values[i] = value;
    }
    return .{ .nrows = n, .ncols = n, .row_ptr = row_ptr, .col_idx = col_idx, .values = values };
}

pub fn clone(alloc: std.mem.Allocator, matrix: CsrMatrix) !CsrMatrix {
    const row_ptr = try alloc.alloc(usize, matrix.row_ptr.len);
    errdefer alloc.free(row_ptr);
    @memcpy(row_ptr, matrix.row_ptr);
    const col_idx = try alloc.alloc(usize, matrix.col_idx.len);
    errdefer alloc.free(col_idx);
    @memcpy(col_idx, matrix.col_idx);
    const values = try alloc.alloc(f64, matrix.values.len);
    errdefer alloc.free(values);
    @memcpy(values, matrix.values);
    return .{
        .nrows = matrix.nrows,
        .ncols = matrix.ncols,
        .row_ptr = row_ptr,
        .col_idx = col_idx,
        .values = values,
    };
}

pub fn add_scaled(
    alloc: std.mem.Allocator,
    a: CsrMatrix,
    alpha: f64,
    b: CsrMatrix,
    beta: f64,
    tol: f64,
) !CsrMatrix {
    if (a.nrows != b.nrows or a.ncols != b.ncols) return error.InvalidShape;
    if (a.nrows == 0 or a.ncols == 0) return error.InvalidShape;
    var triplets: std.ArrayList(Triplet) = .empty;
    defer triplets.deinit(alloc);

    var row_accum = std.AutoHashMap(usize, f64).init(alloc);
    defer row_accum.deinit();

    var row: usize = 0;
    while (row < a.nrows) : (row += 1) {
        row_accum.clearRetainingCapacity();
        if (alpha != 0.0) {
            const start = a.row_ptr[row];
            const end = a.row_ptr[row + 1];
            var idx: usize = start;
            while (idx < end) : (idx += 1) {
                const value = alpha * a.values[idx];
                try add_to_map(&row_accum, a.col_idx[idx], value);
            }
        }
        if (beta != 0.0) {
            const start = b.row_ptr[row];
            const end = b.row_ptr[row + 1];
            var idx: usize = start;
            while (idx < end) : (idx += 1) {
                const value = beta * b.values[idx];
                try add_to_map(&row_accum, b.col_idx[idx], value);
            }
        }
        var it = row_accum.iterator();
        while (it.next()) |entry| {
            const value = entry.value_ptr.*;
            if (@abs(value) <= tol) continue;
            try triplets.append(alloc, .{ .row = row, .col = entry.key_ptr.*, .value = value });
        }
    }
    return CsrMatrix.init_from_triplets(alloc, a.nrows, a.ncols, triplets.items);
}

pub fn mul(
    alloc: std.mem.Allocator,
    a: CsrMatrix,
    b: CsrMatrix,
    tol: f64,
) !CsrMatrix {
    if (a.ncols != b.nrows) return error.InvalidShape;
    if (a.nrows == 0 or b.ncols == 0) return error.InvalidShape;
    var triplets: std.ArrayList(Triplet) = .empty;
    defer triplets.deinit(alloc);

    var row_accum = std.AutoHashMap(usize, f64).init(alloc);
    defer row_accum.deinit();

    var row: usize = 0;
    while (row < a.nrows) : (row += 1) {
        row_accum.clearRetainingCapacity();
        const start = a.row_ptr[row];
        const end = a.row_ptr[row + 1];
        var idx: usize = start;
        while (idx < end) : (idx += 1) {
            const k = a.col_idx[idx];
            const a_val = a.values[idx];
            const b_start = b.row_ptr[k];
            const b_end = b.row_ptr[k + 1];
            var j: usize = b_start;
            while (j < b_end) : (j += 1) {
                const value = a_val * b.values[j];
                try add_to_map(&row_accum, b.col_idx[j], value);
            }
        }
        var it = row_accum.iterator();
        while (it.next()) |entry| {
            const value = entry.value_ptr.*;
            if (@abs(value) <= tol) continue;
            try triplets.append(alloc, .{ .row = row, .col = entry.key_ptr.*, .value = value });
        }
    }
    return CsrMatrix.init_from_triplets(alloc, a.nrows, b.ncols, triplets.items);
}

fn add_to_map(map: *std.AutoHashMap(usize, f64), col: usize, value: f64) !void {
    if (value == 0.0) return;
    const entry = try map.getOrPut(col);
    if (entry.found_existing) {
        entry.value_ptr.* += value;
    } else {
        entry.value_ptr.* = value;
    }
}

fn triplet_less(_: void, a: Triplet, b: Triplet) bool {
    if (a.row < b.row) return true;
    if (a.row > b.row) return false;
    return a.col < b.col;
}

test "csr merges duplicate triplets" {
    const alloc = std.testing.allocator;
    const triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 1.0 },
        .{ .row = 0, .col = 0, .value = 2.0 },
        .{ .row = 0, .col = 2, .value = 4.0 },
        .{ .row = 1, .col = 1, .value = 3.0 },
    };
    var csr = try CsrMatrix.init_from_triplets(alloc, 2, 3, triplets[0..]);
    defer csr.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 0), csr.row_ptr[0]);
    try std.testing.expectEqual(@as(usize, 2), csr.row_ptr[1]);
    try std.testing.expectEqual(@as(usize, 3), csr.row_ptr[2]);
    try std.testing.expectEqual(@as(usize, 0), csr.col_idx[0]);
    try std.testing.expectEqual(@as(usize, 2), csr.col_idx[1]);
    try std.testing.expectEqual(@as(f64, 3.0), csr.values[0]);
    try std.testing.expectEqual(@as(f64, 4.0), csr.values[1]);
}

test "csr mul vec" {
    const alloc = std.testing.allocator;
    const triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 3.0 },
        .{ .row = 0, .col = 2, .value = 4.0 },
        .{ .row = 1, .col = 1, .value = 3.0 },
    };
    var csr = try CsrMatrix.init_from_triplets(alloc, 2, 3, triplets[0..]);
    defer csr.deinit(alloc);

    const x = [_]f64{ 1.0, 2.0, 3.0 };
    var out = [_]f64{ 0.0, 0.0 };
    try csr.mul_vec(x[0..], out[0..]);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), out[1], 1e-12);
}

test "csr trace and diagonal" {
    const alloc = std.testing.allocator;
    var diag = try diagonal(alloc, 3, 2.0);
    defer diag.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f64, 6.0), trace(diag), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), diag.value_at(1, 1), 1e-12);
}

test "csr scale_in_place" {
    const alloc = std.testing.allocator;
    var diag = try diagonal(alloc, 2, 4.0);
    defer diag.deinit(alloc);

    scale_in_place(&diag, 0.5);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), trace(diag), 1e-12);
}

test "csr trace_product" {
    const alloc = std.testing.allocator;
    const a_triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 1.0 },
        .{ .row = 0, .col = 1, .value = 2.0 },
        .{ .row = 1, .col = 1, .value = 3.0 },
    };
    const b_triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 4.0 },
        .{ .row = 0, .col = 1, .value = 5.0 },
        .{ .row = 1, .col = 0, .value = 6.0 },
        .{ .row = 1, .col = 1, .value = 7.0 },
    };
    var a = try CsrMatrix.init_from_triplets(alloc, 2, 2, a_triplets[0..]);
    defer a.deinit(alloc);

    var b = try CsrMatrix.init_from_triplets(alloc, 2, 2, b_triplets[0..]);
    defer b.deinit(alloc);

    const result = try trace_product(a, b);
    try std.testing.expectApproxEqAbs(@as(f64, 37.0), result, 1e-12);
}

test "csr diagonal_values" {
    const alloc = std.testing.allocator;
    const triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 1.0 },
        .{ .row = 1, .col = 1, .value = 3.0 },
        .{ .row = 2, .col = 1, .value = 2.0 },
    };
    var csr = try CsrMatrix.init_from_triplets(alloc, 3, 3, triplets[0..]);
    defer csr.deinit(alloc);

    const diag = try diagonal_values(alloc, csr);
    defer alloc.free(diag);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), diag[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), diag[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), diag[2], 1e-12);
}

test "csr value_at and clone" {
    const alloc = std.testing.allocator;
    const triplets = [_]Triplet{
        .{ .row = 0, .col = 1, .value = 2.0 },
        .{ .row = 1, .col = 0, .value = 3.0 },
    };
    var csr = try CsrMatrix.init_from_triplets(alloc, 2, 2, triplets[0..]);
    defer csr.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), csr.value_at(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), csr.value_at(0, 0), 1e-12);

    var copy = try clone(alloc, csr);
    defer copy.deinit(alloc);

    copy.values[0] = 4.0;
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), csr.value_at(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), copy.value_at(0, 1), 1e-12);
}

test "csr add_scaled" {
    const alloc = std.testing.allocator;
    const a_triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 1.0 },
        .{ .row = 1, .col = 1, .value = 2.0 },
    };
    const b_triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 3.0 },
        .{ .row = 1, .col = 1, .value = -1.0 },
    };
    var a = try CsrMatrix.init_from_triplets(alloc, 2, 2, a_triplets[0..]);
    defer a.deinit(alloc);

    var b = try CsrMatrix.init_from_triplets(alloc, 2, 2, b_triplets[0..]);
    defer b.deinit(alloc);

    var out = try add_scaled(alloc, a, 1.0, b, 1.0, 0.0);
    defer out.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f64, 4.0), out.value_at(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), out.value_at(1, 1), 1e-12);
}

test "csr mul" {
    const alloc = std.testing.allocator;
    const a_triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 1.0 },
        .{ .row = 0, .col = 1, .value = 2.0 },
        .{ .row = 1, .col = 1, .value = 3.0 },
    };
    const b_triplets = [_]Triplet{
        .{ .row = 0, .col = 0, .value = 4.0 },
        .{ .row = 1, .col = 0, .value = 5.0 },
        .{ .row = 1, .col = 1, .value = 6.0 },
    };
    var a = try CsrMatrix.init_from_triplets(alloc, 2, 2, a_triplets[0..]);
    defer a.deinit(alloc);

    var b = try CsrMatrix.init_from_triplets(alloc, 2, 2, b_triplets[0..]);
    defer b.deinit(alloc);

    var out = try mul(alloc, a, b, 0.0);
    defer out.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f64, 14.0), out.value_at(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), out.value_at(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), out.value_at(1, 0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.0), out.value_at(1, 1), 1e-12);
}
