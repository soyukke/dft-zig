const std = @import("std");

const local_orbital_scf = @import("local_orbital_scf.zig");
const sparse = @import("sparse.zig");

/// Energy terms for O(N) SCF calculation
pub const EnergyTerms = struct {
    total: f64,
    hartree: f64,
    xc: f64,
    vxc_rho: f64,
    nonlocal: f64 = 0.0,
};

/// Reference data for O(N) SCF results
pub const ScfReferenceData = struct {
    schema_version: u32 = 1,
    energy: EnergyTerms,
    density_diagonal: []const f64,
    iterations: usize,
    converged: bool,
};

/// Write SCF reference data to JSON file
pub fn write_reference_json(
    io: std.Io,
    alloc: std.mem.Allocator,
    dir: std.Io.Dir,
    path: []const u8,
    result: *const local_orbital_scf.ScfGridResult,
) !void {
    const diag = try sparse.diagonal_values(alloc, result.density);
    defer alloc.free(diag);

    var file = try dir.createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    const payload = ScfReferenceData{
        .energy = .{
            .total = result.energy,
            .hartree = result.energy_hartree,
            .xc = result.energy_xc,
            .vxc_rho = result.energy_vxc_rho,
            .nonlocal = result.energy_nonlocal,
        },
        .density_diagonal = diag,
        .iterations = result.iterations,
        .converged = result.converged,
    };

    var buffer: [8192]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;
    try std.json.Stringify.value(payload, .{ .whitespace = .indent_2 }, out);
    try out.writeAll("\n");
    try out.flush();
}

/// Read SCF reference data from JSON file
pub fn read_reference_json(
    io: std.Io,
    alloc: std.mem.Allocator,
    dir: std.Io.Dir,
    path: []const u8,
) !ScfReferenceOwned {
    const content = try dir.readFileAlloc(io, path, alloc, .limited(64 * 1024 * 1024));
    defer alloc.free(content);

    var parsed = try std.json.parseFromSlice(ScfReferenceData, alloc, content, .{});
    defer parsed.deinit();

    if (parsed.value.schema_version != 1) return error.UnsupportedSchemaVersion;

    const density = try alloc.alloc(f64, parsed.value.density_diagonal.len);
    @memcpy(density, parsed.value.density_diagonal);

    return .{
        .energy = parsed.value.energy,
        .density_diagonal = density,
        .iterations = parsed.value.iterations,
        .converged = parsed.value.converged,
    };
}

pub const ScfReferenceOwned = struct {
    energy: EnergyTerms,
    density_diagonal: []f64,
    iterations: usize,
    converged: bool,

    pub fn deinit(self: *ScfReferenceOwned, alloc: std.mem.Allocator) void {
        if (self.density_diagonal.len > 0) {
            alloc.free(self.density_diagonal);
        }
        self.* = undefined;
    }
};

/// Compare two SCF results
pub const ComparisonResult = struct {
    energy_diff: f64,
    energy_rel_diff: f64,
    density_max_diff: f64,
    density_rms_diff: f64,
};

pub fn compare_results(
    alloc: std.mem.Allocator,
    result: *const local_orbital_scf.ScfGridResult,
    reference: *const ScfReferenceOwned,
) !ComparisonResult {
    const diag = try sparse.diagonal_values(alloc, result.density);
    defer alloc.free(diag);

    if (diag.len != reference.density_diagonal.len) return error.MismatchedSize;

    var max_diff: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    for (diag, 0..) |val, i| {
        const diff = @abs(val - reference.density_diagonal[i]);
        max_diff = @max(max_diff, diff);
        sum_sq += diff * diff;
    }
    const rms = @sqrt(sum_sq / @as(f64, @floatFromInt(diag.len)));

    const energy_diff = result.energy - reference.energy.total;
    const energy_rel = if (@abs(reference.energy.total) > 1e-10)
        energy_diff / @abs(reference.energy.total)
    else
        energy_diff;

    return .{
        .energy_diff = energy_diff,
        .energy_rel_diff = energy_rel,
        .density_max_diff = max_diff,
        .density_rms_diff = rms,
    };
}

test "round-trip JSON" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    const triplets = [_]sparse.Triplet{
        .{ .row = 0, .col = 0, .value = 1.0 },
        .{ .row = 1, .col = 1, .value = 2.0 },
    };
    var density = try sparse.CsrMatrix.init_from_triplets(alloc, 2, 2, triplets[0..]);
    defer density.deinit(alloc);

    var overlap = try sparse.CsrMatrix.init_from_triplets(alloc, 2, 2, triplets[0..]);
    defer overlap.deinit(alloc);

    var hamiltonian = try sparse.CsrMatrix.init_from_triplets(alloc, 2, 2, triplets[0..]);
    defer hamiltonian.deinit(alloc);

    const result = local_orbital_scf.ScfGridResult{
        .overlap = overlap,
        .hamiltonian = hamiltonian,
        .density = density,
        .energy = -10.5,
        .energy_hartree = 5.0,
        .energy_xc = -3.0,
        .energy_vxc_rho = -4.0,
        .energy_nonlocal = 1.5,
        .iterations = 5,
        .converged = true,
    };

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try write_reference_json(io, alloc, tmp_dir.dir, "test.json", &result);

    var ref = try read_reference_json(io, alloc, tmp_dir.dir, "test.json");
    defer ref.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f64, -10.5), ref.energy.total, 1e-10);
    try std.testing.expectEqual(@as(usize, 5), ref.iterations);
    try std.testing.expect(ref.converged);
    try std.testing.expectEqual(@as(usize, 2), ref.density_diagonal.len);
}
