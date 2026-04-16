const std = @import("std");

const energy_compare = @import("energy_compare.zig");
const reference = @import("reference.zig");
const scf_harness = @import("scf_harness.zig");
const scf = @import("../scf/scf.zig");

pub const ScfReferenceJson = struct {
    schema_version: u32,
    energy_terms: scf.EnergyTerms,
    density: []const f64,
};

pub const ScfReferenceOwned = struct {
    schema_version: u32,
    energy_terms: scf.EnergyTerms,
    density: []f64,

    pub fn deinit(self: *ScfReferenceOwned, alloc: std.mem.Allocator) void {
        if (self.density.len > 0) {
            alloc.free(self.density);
        }
        self.* = undefined;
    }
};

pub const ScfComparisonJson = struct {
    schema_version: u32,
    total: reference.ComparisonReport,
    energy_terms: energy_compare.EnergyComparison,
};

pub fn writeReferenceJson(
    dir: std.Io.Dir,
    path: []const u8,
    snapshot: scf_harness.ScfSnapshot,
) !void {
    var file = try dir.createFile(path, .{ .truncate = true });
    defer file.close();

    const payload = ScfReferenceJson{
        .schema_version = 1,
        .energy_terms = snapshot.energy_terms,
        .density = snapshot.density,
    };
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(&buffer);
    const out = &writer.interface;
    try std.json.Stringify.value(payload, .{ .whitespace = .indent_2 }, out);
    try out.writeAll("\n");
    try out.flush();
}

pub fn writeReferenceFromScfResult(
    dir: std.Io.Dir,
    path: []const u8,
    result: *const scf.ScfResult,
) !void {
    const snapshot = scf_harness.snapshotFromScfResult(result);
    try writeReferenceJson(dir, path, snapshot);
}

pub fn readReferenceJson(
    alloc: std.mem.Allocator,
    dir: std.Io.Dir,
    path: []const u8,
) !ScfReferenceOwned {
    const content = try dir.readFileAlloc(alloc, path, 64 * 1024 * 1024);
    defer alloc.free(content);
    var parsed = try std.json.parseFromSlice(ScfReferenceJson, alloc, content, .{});
    defer parsed.deinit();
    if (parsed.value.schema_version != 1) return error.UnsupportedSchemaVersion;

    const density = try alloc.alloc(f64, parsed.value.density.len);
    @memcpy(density, parsed.value.density);
    return .{
        .schema_version = parsed.value.schema_version,
        .energy_terms = parsed.value.energy_terms,
        .density = density,
    };
}

pub fn compareReferenceToScfResult(
    reference_data: *const ScfReferenceOwned,
    result: *const scf.ScfResult,
) !scf_harness.ScfComparisonReport {
    const ref_snapshot = scf_harness.ScfSnapshot{
        .energy_terms = reference_data.energy_terms,
        .density = reference_data.density,
    };
    const cand_snapshot = scf_harness.snapshotFromScfResult(result);
    return scf_harness.compareSnapshots(ref_snapshot, cand_snapshot);
}

pub fn writeComparisonJson(
    dir: std.Io.Dir,
    path: []const u8,
    report: scf_harness.ScfComparisonReport,
) !void {
    var file = try dir.createFile(path, .{ .truncate = true });
    defer file.close();

    const payload = ScfComparisonJson{
        .schema_version = 1,
        .total = report.total,
        .energy_terms = report.energy_terms,
    };
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(&buffer);
    const out = &writer.interface;
    try std.json.Stringify.value(payload, .{ .whitespace = .indent_2 }, out);
    try out.writeAll("\n");
    try out.flush();
}

pub fn writeComparisonFromScfResults(
    dir: std.Io.Dir,
    path: []const u8,
    reference_result: *const scf.ScfResult,
    candidate_result: *const scf.ScfResult,
) !scf_harness.ScfComparisonReport {
    const report = try scf_harness.compareScfResults(reference_result, candidate_result);
    try writeComparisonJson(dir, path, report);
    return report;
}

test "scf reference json round trip" {
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const density = [_]f64{ 1.0, 2.0, 3.0 };
    const terms = scf.EnergyTerms{
        .total = -10.0,
        .band = 1.0,
        .hartree = 2.0,
        .vxc_rho = 3.0,
        .xc = 4.0,
        .ion_ion = 5.0,
        .psp_core = 6.0,
        .double_counting = 7.0,
        .local_pseudo = 8.0,
        .nonlocal_pseudo = 9.0,
    };
    const snapshot = scf_harness.ScfSnapshot{ .energy_terms = terms, .density = density[0..] };
    try writeReferenceJson(tmp.dir, "ref.json", snapshot);

    var loaded = try readReferenceJson(alloc, tmp.dir, "ref.json");
    defer loaded.deinit(alloc);
    try std.testing.expectEqual(@as(usize, density.len), loaded.density.len);
    try std.testing.expectApproxEqAbs(@as(f64, -10.0), loaded.energy_terms.total, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), loaded.density[1], 1e-12);
}

test "scf comparison json parses" {
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const report = scf_harness.ScfComparisonReport{
        .total = .{
            .energy = .{ .abs = 0.1, .rel = 0.01 },
            .density = .{ .max_abs = 0.2, .mean_abs = 0.1, .rms = 0.12, .rel_max = 0.02, .rel_rms = 0.03 },
        },
        .energy_terms = .{
            .total = .{ .abs = 0.1, .rel = 0.01 },
            .band = .{ .abs = 0.2, .rel = 0.02 },
            .hartree = .{ .abs = 0.3, .rel = 0.03 },
            .vxc_rho = .{ .abs = 0.4, .rel = 0.04 },
            .xc = .{ .abs = 0.5, .rel = 0.05 },
            .ion_ion = .{ .abs = 0.6, .rel = 0.06 },
            .psp_core = .{ .abs = 0.7, .rel = 0.07 },
            .double_counting = .{ .abs = 0.8, .rel = 0.08 },
            .local_pseudo = .{ .abs = 0.9, .rel = 0.09 },
            .nonlocal_pseudo = .{ .abs = 1.0, .rel = 0.10 },
        },
    };
    try writeComparisonJson(tmp.dir, "compare.json", report);

    const content = try tmp.dir.readFileAlloc(alloc, "compare.json", 1024 * 1024);
    defer alloc.free(content);
    var parsed = try std.json.parseFromSlice(ScfComparisonJson, alloc, content, .{});
    defer parsed.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), parsed.value.total.energy.abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), parsed.value.energy_terms.xc.abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.12), parsed.value.total.density.rms, 1e-12);
}
