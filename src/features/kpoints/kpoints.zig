const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const runtime_logging = @import("../runtime/logging.zig");
const mesh = @import("mesh.zig");
const reduction = @import("reduction.zig");

pub const KPoint = symmetry.KPoint;

fn logKpointInfo(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, fmt, args);
}

fn logKpointWarn(io: std.Io, msg: []const u8) !void {
    const logger = runtime_logging.stderr(io, .warn);
    try logger.print(.warn, "{s}\n", .{msg});
}

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
    io: std.Io,
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
        try logKpointInfo(io, "scf: kmesh-compatible symmetry ops {d}/{d}\n", .{ filtered_ops.len, ops.len });
    }

    try logKpointInfo(io, "scf: symmetry ops {d}\n", .{ops.len});

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
        try logKpointWarn(io, "scf: kpoint reduction failed verification; using full mesh");
        alloc.free(reduced);
        return full;
    }

    if (reduced.len < full.len) {
        try logKpointInfo(io, "scf: kpoints reduced {d} -> {d}\n", .{ full.len, reduced.len });
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
    const kpoints = try generateKmeshSymmetry(alloc, std.testing.io, .{ 6, 6, 6 }, shift, recip, cell_bohr, atoms[0..], true);
    defer alloc.free(kpoints);
    try std.testing.expectEqual(@as(usize, 16), kpoints.len);
    var weight_sum: f64 = 0.0;
    for (kpoints) |kp| {
        weight_sum += kp.weight;
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), weight_sum, 1e-12);
}
