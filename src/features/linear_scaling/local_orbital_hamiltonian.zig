const std = @import("std");

const local_orbital = @import("local_orbital.zig");
const local_orbital_potential = @import("local_orbital_potential.zig");
const neighbor_list = @import("neighbor_list.zig");
const sparse = @import("sparse.zig");
const math = @import("../math/math.zig");

pub const HamiltonianOptions = struct {
    sigma: f64,
    cutoff: f64,
    local_potential: f64,
    kinetic_scale: f64 = 1.0,
    threshold: f64 = 0.0,
};

pub const HamiltonianResult = struct {
    overlap: sparse.CsrMatrix,
    hamiltonian: sparse.CsrMatrix,

    pub fn deinit(self: *HamiltonianResult, alloc: std.mem.Allocator) void {
        self.overlap.deinit(alloc);
        self.hamiltonian.deinit(alloc);
        self.* = undefined;
    }
};

pub const HamiltonianGridOptions = struct {
    sigma: f64,
    cutoff: f64,
    kinetic_scale: f64 = 1.0,
    threshold: f64 = 0.0,
};

pub fn buildHamiltonianFromCenters(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: HamiltonianOptions,
) !HamiltonianResult {
    if (centers.len == 0) return error.InvalidShape;
    var overlap = try local_orbital.buildOverlapCsrFromCenters(
        alloc,
        centers,
        opts.sigma,
        opts.cutoff,
        pbc,
        cell,
    );
    errdefer overlap.deinit(alloc);

    var kinetic = try local_orbital.buildKineticCsrFromCenters(
        alloc,
        centers,
        opts.sigma,
        opts.cutoff,
        pbc,
        cell,
    );
    defer kinetic.deinit(alloc);

    if (opts.kinetic_scale != 1.0) {
        sparse.scaleInPlace(&kinetic, opts.kinetic_scale);
    }
    const hamiltonian = try sparse.addScaled(
        alloc,
        kinetic,
        1.0,
        overlap,
        opts.local_potential,
        opts.threshold,
    );
    return .{ .overlap = overlap, .hamiltonian = hamiltonian };
}

pub fn buildHamiltonianFromCentersWithGrid(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    pbc: neighbor_list.Pbc,
    grid: local_orbital_potential.PotentialGrid,
    opts: HamiltonianGridOptions,
) !HamiltonianResult {
    if (centers.len == 0) return error.InvalidShape;
    var overlap = try local_orbital.buildOverlapCsrFromCenters(
        alloc,
        centers,
        opts.sigma,
        opts.cutoff,
        pbc,
        grid.cell,
    );
    errdefer overlap.deinit(alloc);

    var kinetic = try local_orbital.buildKineticCsrFromCenters(
        alloc,
        centers,
        opts.sigma,
        opts.cutoff,
        pbc,
        grid.cell,
    );
    defer kinetic.deinit(alloc);

    if (opts.kinetic_scale != 1.0) {
        sparse.scaleInPlace(&kinetic, opts.kinetic_scale);
    }
    var local = try local_orbital_potential.buildLocalPotentialCsrFromCenters(
        alloc,
        centers,
        opts.sigma,
        opts.cutoff,
        pbc,
        grid,
    );
    defer local.deinit(alloc);

    const hamiltonian = try sparse.addScaled(alloc, kinetic, 1.0, local, 1.0, opts.threshold);
    return .{ .overlap = overlap, .hamiltonian = hamiltonian };
}

test "buildHamiltonianFromCenters combines kinetic and local potential" {
    const alloc = std.testing.allocator;
    const centers = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.5, .y = 0.0, .z = 0.0 },
    };
    const cell = math.Mat3.fromRows(
        .{ .x = 5.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 5.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 5.0 },
    );
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const opts = HamiltonianOptions{
        .sigma = 0.5,
        .cutoff = 2.0,
        .local_potential = 0.2,
        .kinetic_scale = 1.0,
        .threshold = 0.0,
    };
    var result = try buildHamiltonianFromCenters(alloc, centers[0..], cell, pbc, opts);
    defer result.deinit(alloc);

    const alpha = 1.0 / (opts.sigma * opts.sigma);
    const orbitals = [_]local_orbital.Orbital{
        .{ .center = centers[0], .alpha = alpha, .cutoff = opts.cutoff },
        .{ .center = centers[1], .alpha = alpha, .cutoff = opts.cutoff },
    };
    const expected_diag =
        local_orbital.kineticIntegral(orbitals[0], orbitals[0]) +
        opts.local_potential * local_orbital.overlapIntegral(orbitals[0], orbitals[0]);
    const expected_off =
        local_orbital.kineticIntegral(orbitals[0], orbitals[1]) +
        opts.local_potential * local_orbital.overlapIntegral(orbitals[0], orbitals[1]);
    try std.testing.expectApproxEqAbs(expected_diag, result.hamiltonian.valueAt(0, 0), 1e-10);
    try std.testing.expectApproxEqAbs(expected_off, result.hamiltonian.valueAt(0, 1), 1e-10);
}

test "buildHamiltonianFromCentersWithGrid matches constant potential" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    );
    const dims = [3]usize{ 16, 16, 16 };
    const count = dims[0] * dims[1] * dims[2];
    const values = try alloc.alloc(f64, count);
    defer alloc.free(values);

    @memset(values, 0.4);
    const grid = local_orbital_potential.PotentialGrid{
        .cell = cell,
        .dims = dims,
        .values = values,
    };
    const centers = [_]math.Vec3{
        .{ .x = 2.0, .y = 2.0, .z = 2.0 },
        .{ .x = 3.0, .y = 2.0, .z = 2.0 },
    };
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const opts = HamiltonianGridOptions{
        .sigma = 0.5,
        .cutoff = 3.0,
        .kinetic_scale = 1.0,
        .threshold = 0.0,
    };
    var result = try buildHamiltonianFromCentersWithGrid(alloc, centers[0..], pbc, grid, opts);
    defer result.deinit(alloc);

    const alpha = 1.0 / (opts.sigma * opts.sigma);
    const orbitals = [_]local_orbital.Orbital{
        .{ .center = centers[0], .alpha = alpha, .cutoff = opts.cutoff },
        .{ .center = centers[1], .alpha = alpha, .cutoff = opts.cutoff },
    };
    const expected_diag =
        local_orbital.kineticIntegral(orbitals[0], orbitals[0]) +
        0.4 * local_orbital.overlapIntegral(orbitals[0], orbitals[0]);
    const expected_off =
        local_orbital.kineticIntegral(orbitals[0], orbitals[1]) +
        0.4 * local_orbital.overlapIntegral(orbitals[0], orbitals[1]);
    try std.testing.expectApproxEqAbs(expected_diag, result.hamiltonian.valueAt(0, 0), 2e-2);
    try std.testing.expectApproxEqAbs(expected_off, result.hamiltonian.valueAt(0, 1), 2e-2);
}
