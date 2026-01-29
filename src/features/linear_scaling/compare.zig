const std = @import("std");

const math = @import("../math/math.zig");

pub const ComparisonTolerance = struct {
    abs: f64,
    rel: f64,
};

pub const ComparisonResult = struct {
    max_abs: f64,
    mean_abs: f64,
    rms: f64,
    rel_max: f64,
    rel_rms: f64,
};

pub const ScalarComparison = struct {
    abs: f64,
    rel: f64,
};

pub fn compareScalar(reference: f64, candidate: f64) ScalarComparison {
    const diff = candidate - reference;
    const abs_diff = @abs(diff);
    const denom = @max(@abs(reference), 1e-12);
    return .{ .abs = abs_diff, .rel = abs_diff / denom };
}

pub fn compareSlices(reference: []const f64, candidate: []const f64) !ComparisonResult {
    if (reference.len != candidate.len) return error.MismatchedLength;
    if (reference.len == 0) {
        return .{ .max_abs = 0.0, .mean_abs = 0.0, .rms = 0.0, .rel_max = 0.0, .rel_rms = 0.0 };
    }
    var max_abs: f64 = 0.0;
    var sum_abs: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    var sum_ref_sq: f64 = 0.0;
    var max_ref_abs: f64 = 0.0;

    for (reference, 0..) |ref_value, i| {
        const diff = candidate[i] - ref_value;
        const abs_diff = @abs(diff);
        sum_abs += abs_diff;
        sum_sq += diff * diff;
        max_abs = @max(max_abs, abs_diff);
        sum_ref_sq += ref_value * ref_value;
        max_ref_abs = @max(max_ref_abs, @abs(ref_value));
    }

    const count = @as(f64, @floatFromInt(reference.len));
    const rms = std.math.sqrt(sum_sq / count);
    const mean_abs = sum_abs / count;
    const rel_max = if (max_ref_abs > 0.0) max_abs / max_ref_abs else max_abs;
    const rel_rms = if (sum_ref_sq > 0.0) rms / std.math.sqrt(sum_ref_sq / count) else rms;
    return .{ .max_abs = max_abs, .mean_abs = mean_abs, .rms = rms, .rel_max = rel_max, .rel_rms = rel_rms };
}

pub fn compareVec3Slices(reference: []const math.Vec3, candidate: []const math.Vec3) !ComparisonResult {
    if (reference.len != candidate.len) return error.MismatchedLength;
    if (reference.len == 0) {
        return .{ .max_abs = 0.0, .mean_abs = 0.0, .rms = 0.0, .rel_max = 0.0, .rel_rms = 0.0 };
    }
    var max_abs: f64 = 0.0;
    var sum_abs: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    var sum_ref_sq: f64 = 0.0;
    var max_ref_abs: f64 = 0.0;

    for (reference, 0..) |ref_value, i| {
        const diff = math.Vec3.sub(candidate[i], ref_value);
        const abs_diff = math.Vec3.norm(diff);
        sum_abs += abs_diff;
        sum_sq += abs_diff * abs_diff;
        max_abs = @max(max_abs, abs_diff);
        const ref_abs = math.Vec3.norm(ref_value);
        sum_ref_sq += ref_abs * ref_abs;
        max_ref_abs = @max(max_ref_abs, ref_abs);
    }

    const count = @as(f64, @floatFromInt(reference.len));
    const rms = std.math.sqrt(sum_sq / count);
    const mean_abs = sum_abs / count;
    const rel_max = if (max_ref_abs > 0.0) max_abs / max_ref_abs else max_abs;
    const rel_rms = if (sum_ref_sq > 0.0) rms / std.math.sqrt(sum_ref_sq / count) else rms;
    return .{ .max_abs = max_abs, .mean_abs = mean_abs, .rms = rms, .rel_max = rel_max, .rel_rms = rel_rms };
}

pub fn withinTolerance(result: ComparisonResult, tol: ComparisonTolerance) bool {
    return result.max_abs <= tol.abs or result.rel_max <= tol.rel;
}

pub fn withinScalarTolerance(result: ScalarComparison, tol: ComparisonTolerance) bool {
    return result.abs <= tol.abs or result.rel <= tol.rel;
}

test "compareScalar computes absolute and relative diff" {
    const result = compareScalar(2.0, 2.1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), result.abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.05), result.rel, 1e-12);
}

test "compareSlices identical arrays" {
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    const result = try compareSlices(a[0..], a[0..]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.max_abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.rms, 1e-12);
}

test "compareSlices simple diff" {
    const ref = [_]f64{ 1.0, 2.0 };
    const cand = [_]f64{ 1.1, 1.9 };
    const result = try compareSlices(ref[0..], cand[0..]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), result.max_abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), result.rms, 1e-12);
    try std.testing.expect(result.rel_max > 0.0);
}

test "compareVec3Slices norms" {
    const ref = [_]math.Vec3{ .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .{ .x = 1.0, .y = 0.0, .z = 0.0 } };
    const cand = [_]math.Vec3{ .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .{ .x = 1.5, .y = 0.0, .z = 0.0 } };
    const result = try compareVec3Slices(ref[0..], cand[0..]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), result.max_abs, 1e-12);
}

test "withinTolerance uses abs or rel" {
    const ref = [_]f64{ 10.0, 10.0 };
    const cand = [_]f64{ 10.05, 9.95 };
    const result = try compareSlices(ref[0..], cand[0..]);
    const tol = ComparisonTolerance{ .abs = 0.01, .rel = 0.01 };
    try std.testing.expect(withinTolerance(result, tol));
}

test "withinScalarTolerance uses abs or rel" {
    const tol = ComparisonTolerance{ .abs = 0.05, .rel = 0.1 };
    const pass_abs = ScalarComparison{ .abs = 0.04, .rel = 0.2 };
    const pass_rel = ScalarComparison{ .abs = 0.2, .rel = 0.05 };
    const fail = ScalarComparison{ .abs = 0.2, .rel = 0.2 };
    try std.testing.expect(withinScalarTolerance(pass_abs, tol));
    try std.testing.expect(withinScalarTolerance(pass_rel, tol));
    try std.testing.expect(!withinScalarTolerance(fail, tol));
}
