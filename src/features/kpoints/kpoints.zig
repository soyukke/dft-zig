const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const mesh = @import("mesh.zig");
const reduction = @import("reduction.zig");

pub const KPoint = symmetry.KPoint;

pub fn generateKmesh(
    alloc: std.mem.Allocator,
    kmesh: [3]usize,
    recip: math.Mat3,
    shift: math.Vec3,
) ![]KPoint {
    return mesh.generateKmesh(alloc, kmesh, recip, shift);
}

pub fn generateKmeshSymmetry(
    alloc: std.mem.Allocator,
    kmesh: [3]usize,
    shift: math.Vec3,
    recip: math.Mat3,
    cell: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    time_reversal: bool,
) ![]KPoint {
    const full = try mesh.generateKmesh(alloc, kmesh, recip, shift);
    errdefer alloc.free(full);

    const ops = try symmetry.getSymmetryOps(alloc, cell, atoms, 1e-6);
    defer alloc.free(ops);
    if (ops.len == 0) return full;

    const filtered_ops = try reduction.filterSymOpsForKmesh(alloc, ops, kmesh, shift, 1e-8);
    defer alloc.free(filtered_ops);
    if (filtered_ops.len == 0) return full;

    if (filtered_ops.len != ops.len) {
        var buffer: [128]u8 = undefined;
        var writer = std.fs.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print("scf: kmesh-compatible symmetry ops {d}/{d}\n", .{ filtered_ops.len, ops.len });
        try out.flush();
    }

    var ops_buffer: [128]u8 = undefined;
    var ops_writer = std.fs.File.stderr().writer(&ops_buffer);
    const ops_out = &ops_writer.interface;
    try ops_out.print("scf: symmetry ops {d}\n", .{ops.len});
    try ops_out.flush();

    const reduced = try reduction.reduceKmesh(alloc, kmesh, shift, filtered_ops, recip, time_reversal);
    const verified = try reduction.verifyKmeshReduction(
        alloc,
        full,
        reduced,
        kmesh,
        shift,
        filtered_ops,
        time_reversal,
        1e-6,
    );
    if (!verified) {
        var buffer: [192]u8 = undefined;
        var writer = std.fs.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.writeAll("scf: kpoint reduction failed verification; using full mesh\n");
        try out.flush();
        alloc.free(reduced);
        return full;
    }

    if (reduced.len < full.len) {
        var buffer: [128]u8 = undefined;
        var writer = std.fs.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print("scf: kpoints reduced {d} -> {d}\n", .{ full.len, reduced.len });
        try out.flush();
    }

    alloc.free(full);
    return reduced;
}

test "abinit kmesh reduction for aluminum fcc" {
    const alloc = std.testing.allocator;
    const cell_ang = math.Mat3.fromRows(
        .{ .x = 0.0, .y = 2.025, .z = 2.025 },
        .{ .x = 2.025, .y = 0.0, .z = 2.025 },
        .{ .x = 2.025, .y = 2.025, .z = 0.0 },
    );
    const cell_bohr = cell_ang.scale(math.unitsScaleToBohr(.angstrom));
    const recip = math.reciprocal(cell_bohr);
    const atoms = [_]hamiltonian.AtomData{.{ .position = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .species_index = 0 }};
    const shift = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const kpoints = try generateKmeshSymmetry(alloc, .{ 6, 6, 6 }, shift, recip, cell_bohr, atoms[0..], true);
    defer alloc.free(kpoints);
    try std.testing.expectEqual(@as(usize, 16), kpoints.len);
    var weight_sum: f64 = 0.0;
    for (kpoints) |kp| {
        weight_sum += kp.weight;
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), weight_sum, 1e-12);
}
