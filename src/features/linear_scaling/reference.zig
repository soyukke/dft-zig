const std = @import("std");

const compare = @import("compare.zig");

pub const ReferenceData = struct {
    energy: f64,
    density: []const f64,
};

pub const ComparisonReport = struct {
    energy: compare.ScalarComparison,
    density: compare.ComparisonResult,
};

pub fn compare_reference(reference: ReferenceData, candidate: ReferenceData) !ComparisonReport {
    const energy = compare.compare_scalar(reference.energy, candidate.energy);
    const density = try compare.compare_slices(reference.density, candidate.density);
    return .{ .energy = energy, .density = density };
}

test "compare_reference computes energy and density metrics" {
    const ref_density = [_]f64{ 1.0, 2.0, 3.0 };
    const cand_density = [_]f64{ 1.1, 1.9, 3.0 };
    const ref = ReferenceData{ .energy = -10.0, .density = ref_density[0..] };
    const cand = ReferenceData{ .energy = -9.9, .density = cand_density[0..] };
    const report = try compare_reference(ref, cand);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), report.energy.abs, 1e-12);
    try std.testing.expect(report.density.max_abs > 0.0);
}

test "compare_reference detects mismatched density length" {
    const ref_density = [_]f64{ 1.0, 2.0 };
    const cand_density = [_]f64{ 1.0, 2.0, 3.0 };
    const ref = ReferenceData{ .energy = 0.0, .density = ref_density[0..] };
    const cand = ReferenceData{ .energy = 0.0, .density = cand_density[0..] };
    try std.testing.expectError(error.MismatchedLength, compare_reference(ref, cand));
}
