const std = @import("std");

const energy_compare = @import("energy_compare.zig");
const reference = @import("reference.zig");
const structures = @import("structures.zig");
const scf = @import("../scf/scf.zig");
const math = @import("../math/math.zig");
const xyz = @import("../structure/xyz.zig");

pub const ScfSnapshot = struct {
    energy_terms: scf.EnergyTerms,
    density: []const f64,
};

pub const ScfComparisonReport = struct {
    total: reference.ComparisonReport,
    energy_terms: energy_compare.EnergyComparison,
};

pub const silicon_supercell_reps = [3]usize{ 2, 2, 2 };

pub fn snapshotFromScfResult(result: *const scf.ScfResult) ScfSnapshot {
    return .{ .energy_terms = result.energy, .density = result.density };
}

pub fn asReference(snapshot: ScfSnapshot) reference.ReferenceData {
    return .{ .energy = snapshot.energy_terms.total, .density = snapshot.density };
}

pub fn compareSnapshots(
    reference_snapshot: ScfSnapshot,
    candidate_snapshot: ScfSnapshot,
) !ScfComparisonReport {
    const total = try reference.compareReference(
        asReference(reference_snapshot),
        asReference(candidate_snapshot),
    );
    const energy_terms = energy_compare.compareEnergyTerms(
        reference_snapshot.energy_terms,
        candidate_snapshot.energy_terms,
    );
    return .{ .total = total, .energy_terms = energy_terms };
}

pub fn compareScfResults(
    reference_result: *const scf.ScfResult,
    candidate_result: *const scf.ScfResult,
) !ScfComparisonReport {
    const ref_snapshot = snapshotFromScfResult(reference_result);
    const cand_snapshot = snapshotFromScfResult(candidate_result);
    return compareSnapshots(ref_snapshot, cand_snapshot);
}

pub fn siliconConventionalSupercellCell(a: f64, reps: [3]usize) math.Mat3 {
    return math.Mat3.fromRows(
        .{ .x = a * @as(f64, @floatFromInt(reps[0])), .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = a * @as(f64, @floatFromInt(reps[1])), .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = a * @as(f64, @floatFromInt(reps[2])) },
    );
}

pub fn siliconConventional2x2x2Cell(a: f64) math.Mat3 {
    return siliconConventionalSupercellCell(a, silicon_supercell_reps);
}

pub fn buildSiliconSupercellAtoms(
    alloc: std.mem.Allocator,
    a: f64,
    reps: [3]usize,
    symbol: []const u8,
) ![]xyz.Atom {
    const positions = try structures.diamondConventionalSupercell(alloc, a, reps);
    defer alloc.free(positions);
    const atoms = try alloc.alloc(xyz.Atom, positions.len);
    errdefer {
        for (atoms) |atom| {
            alloc.free(atom.symbol);
        }
        alloc.free(atoms);
    }
    for (positions, 0..) |pos, i| {
        atoms[i] = .{ .symbol = try alloc.dupe(u8, symbol), .position = pos };
    }
    return atoms;
}

pub fn buildSiliconConventional2x2x2Atoms(alloc: std.mem.Allocator, a: f64) ![]xyz.Atom {
    return buildSiliconSupercellAtoms(alloc, a, silicon_supercell_reps, "Si");
}

pub fn deinitAtoms(alloc: std.mem.Allocator, atoms: []xyz.Atom) void {
    for (atoms) |atom| {
        alloc.free(atom.symbol);
    }
    if (atoms.len > 0) {
        alloc.free(atoms);
    }
}

test "silicon supercell atoms and cell" {
    const alloc = std.testing.allocator;
    const a = 5.0;
    const cell = siliconConventional2x2x2Cell(a);
    const atoms = try buildSiliconConventional2x2x2Atoms(alloc, a);
    defer deinitAtoms(alloc, atoms);
    try std.testing.expectEqual(@as(usize, 64), atoms.len);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), cell.row(0).x, 1e-12);
    try xyz.validateInCell(atoms, cell);
}

test "compareSnapshots compares total and energy terms" {
    const ref_density = [_]f64{ 1.0, 2.0, 3.0 };
    const cand_density = [_]f64{ 1.1, 1.9, 3.0 };
    const ref_terms = scf.EnergyTerms{
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
    const cand_terms = scf.EnergyTerms{
        .total = -9.9,
        .band = 1.2,
        .hartree = 2.0,
        .vxc_rho = 3.0,
        .xc = 4.1,
        .ion_ion = 5.0,
        .psp_core = 6.0,
        .double_counting = 7.0,
        .local_pseudo = 8.0,
        .nonlocal_pseudo = 9.0,
    };
    const ref_snapshot = ScfSnapshot{ .energy_terms = ref_terms, .density = ref_density[0..] };
    const cand_snapshot = ScfSnapshot{ .energy_terms = cand_terms, .density = cand_density[0..] };
    const report = try compareSnapshots(ref_snapshot, cand_snapshot);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), report.total.energy.abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), report.energy_terms.band.abs, 1e-12);
}

test "compareSnapshots fails on mismatched density" {
    const ref_density = [_]f64{ 1.0, 2.0 };
    const cand_density = [_]f64{ 1.0, 2.0, 3.0 };
    const terms = scf.EnergyTerms{
        .total = 0.0,
        .band = 0.0,
        .hartree = 0.0,
        .vxc_rho = 0.0,
        .xc = 0.0,
        .ion_ion = 0.0,
        .psp_core = 0.0,
        .double_counting = 0.0,
        .local_pseudo = 0.0,
        .nonlocal_pseudo = 0.0,
    };
    const ref_snapshot = ScfSnapshot{ .energy_terms = terms, .density = ref_density[0..] };
    const cand_snapshot = ScfSnapshot{ .energy_terms = terms, .density = cand_density[0..] };
    try std.testing.expectError(
        error.MismatchedLength,
        compareSnapshots(ref_snapshot, cand_snapshot),
    );
}
