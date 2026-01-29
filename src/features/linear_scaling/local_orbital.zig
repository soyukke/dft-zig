const std = @import("std");

const math = @import("../math/math.zig");
const neighbor_list = @import("neighbor_list.zig");
const sparse = @import("sparse.zig");

/// Angular momentum type for Gaussian orbitals
pub const AngularType = enum {
    s, // l=0
    px, // l=1, m=+1 (x direction)
    py, // l=1, m=-1 (y direction)
    pz, // l=1, m=0 (z direction)

    pub fn l(self: AngularType) i32 {
        return switch (self) {
            .s => 0,
            .px, .py, .pz => 1,
        };
    }
};

pub const Orbital = struct {
    center: math.Vec3,
    alpha: f64,
    cutoff: f64,
    angular: AngularType = .s,
};

/// Normalization factor for s-type Gaussian: (2α/π)^(3/4)
pub fn gaussianNorm(alpha: f64) f64 {
    return std.math.pow(f64, 2.0 * alpha / std.math.pi, 0.75);
}

/// Normalization factor for p-type Gaussian: (2α/π)^(3/4) × √(4α)
/// p-type: N_p × x × exp(-α r²), where N_p = (2α/π)^(3/4) × √(4α)
pub fn gaussianNormP(alpha: f64) f64 {
    return gaussianNorm(alpha) * @sqrt(4.0 * alpha);
}

pub fn overlap(a: Orbital, b: Orbital) f64 {
    return overlapIntegral(a, b);
}

pub fn overlapIntegral(a: Orbital, b: Orbital) f64 {
    if (a.alpha <= 0.0 or b.alpha <= 0.0) return 0.0;
    const delta = math.Vec3.sub(b.center, a.center);
    const r2 = math.Vec3.dot(delta, delta);
    const cutoff = a.cutoff + b.cutoff;
    if (r2 > cutoff * cutoff) return 0.0;
    const p = a.alpha + b.alpha;
    if (p <= 0.0) return 0.0;
    const mu = a.alpha * b.alpha / p;
    const pref = std.math.pow(f64, std.math.pi / p, 1.5);
    const exp_factor = std.math.exp(-mu * r2);

    // Handle different angular momentum combinations
    return switch (a.angular) {
        .s => switch (b.angular) {
            .s => overlapSS(a.alpha, b.alpha, p, pref, exp_factor),
            .px => overlapSP(a.alpha, b.alpha, p, pref, exp_factor, delta.x),
            .py => overlapSP(a.alpha, b.alpha, p, pref, exp_factor, delta.y),
            .pz => overlapSP(a.alpha, b.alpha, p, pref, exp_factor, delta.z),
        },
        .px => switch (b.angular) {
            .s => overlapSP(b.alpha, a.alpha, p, pref, exp_factor, -delta.x),
            .px => overlapPP(a.alpha, b.alpha, p, pref, exp_factor, delta.x, delta.x),
            .py => overlapPPoff(a.alpha, b.alpha, p, pref, exp_factor, delta.x, delta.y),
            .pz => overlapPPoff(a.alpha, b.alpha, p, pref, exp_factor, delta.x, delta.z),
        },
        .py => switch (b.angular) {
            .s => overlapSP(b.alpha, a.alpha, p, pref, exp_factor, -delta.y),
            .px => overlapPPoff(a.alpha, b.alpha, p, pref, exp_factor, delta.y, delta.x),
            .py => overlapPP(a.alpha, b.alpha, p, pref, exp_factor, delta.y, delta.y),
            .pz => overlapPPoff(a.alpha, b.alpha, p, pref, exp_factor, delta.y, delta.z),
        },
        .pz => switch (b.angular) {
            .s => overlapSP(b.alpha, a.alpha, p, pref, exp_factor, -delta.z),
            .px => overlapPPoff(a.alpha, b.alpha, p, pref, exp_factor, delta.z, delta.x),
            .py => overlapPPoff(a.alpha, b.alpha, p, pref, exp_factor, delta.z, delta.y),
            .pz => overlapPP(a.alpha, b.alpha, p, pref, exp_factor, delta.z, delta.z),
        },
    };
}

/// s-s overlap: N_s^2 × (π/p)^(3/2) × exp(-μr²)
fn overlapSS(alpha_a: f64, alpha_b: f64, p: f64, pref: f64, exp_factor: f64) f64 {
    _ = p;
    const norm = gaussianNorm(alpha_a) * gaussianNorm(alpha_b);
    return norm * pref * exp_factor;
}

/// s-p overlap: N_s × N_p × (α_p × delta_i / p) × (π/p)^(3/2) × exp(-μr²)
/// delta_i is the i-th component of (center_b - center_a)
fn overlapSP(alpha_s: f64, alpha_p: f64, p: f64, pref: f64, exp_factor: f64, delta_i: f64) f64 {
    const norm = gaussianNorm(alpha_s) * gaussianNormP(alpha_p);
    // The integral gives: (α_p / p) × delta_i
    return norm * pref * exp_factor * (alpha_p / p) * delta_i;
}

/// p-p overlap (same direction): N_p^2 × [1/(2p) + (α_a α_b / p²) × delta_i²] × (π/p)^(3/2) × exp
fn overlapPP(alpha_a: f64, alpha_b: f64, p: f64, pref: f64, exp_factor: f64, delta_i: f64, _: f64) f64 {
    const norm = gaussianNormP(alpha_a) * gaussianNormP(alpha_b);
    const term1 = 1.0 / (2.0 * p);
    const term2 = (alpha_a * alpha_b / (p * p)) * delta_i * delta_i;
    return norm * pref * exp_factor * (term1 + term2);
}

/// p-p overlap (different directions): N_p^2 × (α_a α_b / p²) × delta_i × delta_j × (π/p)^(3/2) × exp
fn overlapPPoff(alpha_a: f64, alpha_b: f64, p: f64, pref: f64, exp_factor: f64, delta_i: f64, delta_j: f64) f64 {
    const norm = gaussianNormP(alpha_a) * gaussianNormP(alpha_b);
    return norm * pref * exp_factor * (alpha_a * alpha_b / (p * p)) * delta_i * delta_j;
}

pub fn kineticIntegral(a: Orbital, b: Orbital) f64 {
    if (a.alpha <= 0.0 or b.alpha <= 0.0) return 0.0;
    const delta = math.Vec3.sub(b.center, a.center);
    const r2 = math.Vec3.dot(delta, delta);
    const cutoff = a.cutoff + b.cutoff;
    if (r2 > cutoff * cutoff) return 0.0;
    const p = a.alpha + b.alpha;
    if (p <= 0.0) return 0.0;
    const mu = a.alpha * b.alpha / p;
    const pref = std.math.pow(f64, std.math.pi / p, 1.5);
    const exp_factor = std.math.exp(-mu * r2);

    // Handle different angular momentum combinations
    return switch (a.angular) {
        .s => switch (b.angular) {
            .s => kineticSS(a.alpha, b.alpha, p, mu, pref, exp_factor, r2),
            .px => kineticSP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.x),
            .py => kineticSP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.y),
            .pz => kineticSP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.z),
        },
        .px => switch (b.angular) {
            .s => kineticSP(b.alpha, a.alpha, p, mu, pref, exp_factor, -delta.x),
            .px => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.x, delta.x, true),
            .py => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.x, delta.y, false),
            .pz => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.x, delta.z, false),
        },
        .py => switch (b.angular) {
            .s => kineticSP(b.alpha, a.alpha, p, mu, pref, exp_factor, -delta.y),
            .px => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.y, delta.x, false),
            .py => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.y, delta.y, true),
            .pz => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.y, delta.z, false),
        },
        .pz => switch (b.angular) {
            .s => kineticSP(b.alpha, a.alpha, p, mu, pref, exp_factor, -delta.z),
            .px => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.z, delta.x, false),
            .py => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.z, delta.y, false),
            .pz => kineticPP(a.alpha, b.alpha, p, mu, pref, exp_factor, delta.z, delta.z, true),
        },
    };
}

/// s-s kinetic: T = μ(3 - 2μr²) × S
fn kineticSS(alpha_a: f64, alpha_b: f64, p: f64, mu: f64, pref: f64, exp_factor: f64, r2: f64) f64 {
    _ = p;
    const norm = gaussianNorm(alpha_a) * gaussianNorm(alpha_b);
    const s = norm * pref * exp_factor;
    return mu * (3.0 - 2.0 * mu * r2) * s;
}

/// s-p kinetic: Laplacian of p-type acting on s gives derivative terms
fn kineticSP(alpha_s: f64, alpha_p: f64, p: f64, mu: f64, pref: f64, exp_factor: f64, delta_i: f64) f64 {
    const norm = gaussianNorm(alpha_s) * gaussianNormP(alpha_p);
    const s_sp = norm * pref * exp_factor * (alpha_p / p) * delta_i;
    // Kinetic energy involves -1/2 ∇² acting on the Gaussian product
    // For s-p combination: T = α_p × (3 - 2μr²) × (α_p/p × delta) × S_ss
    // This simplifies to: T_sp = (3 - 2μr²) × S_sp - extra terms
    // Using proper formula for kinetic integral:
    const r2 = delta_i * delta_i; // Approximate with single component (simplified)
    return mu * (3.0 - 2.0 * mu * r2) * s_sp + 2.0 * alpha_p * s_sp;
}

/// p-p kinetic: More complex due to second derivatives
fn kineticPP(alpha_a: f64, alpha_b: f64, p: f64, mu: f64, pref: f64, exp_factor: f64, delta_i: f64, delta_j: f64, same_dir: bool) f64 {
    const norm = gaussianNormP(alpha_a) * gaussianNormP(alpha_b);
    const base = norm * pref * exp_factor;

    if (same_dir) {
        // Same direction (e.g., px-px)
        const term1 = 1.0 / (2.0 * p);
        const term2 = (alpha_a * alpha_b / (p * p)) * delta_i * delta_i;
        const s_pp = base * (term1 + term2);
        // Kinetic energy: T_pp = (5 - 2μr²) × S_pp + correction terms
        const r2 = delta_i * delta_i;
        return mu * (5.0 - 2.0 * mu * r2) * s_pp;
    } else {
        // Different directions (e.g., px-py)
        const s_pp = base * (alpha_a * alpha_b / (p * p)) * delta_i * delta_j;
        const r2 = delta_i * delta_i + delta_j * delta_j;
        return mu * (3.0 - 2.0 * mu * r2) * s_pp;
    }
}

pub fn buildOverlapCsr(
    alloc: std.mem.Allocator,
    orbitals: []const Orbital,
    neighbors: neighbor_list.NeighborList,
) !sparse.CsrMatrix {
    const count = orbitals.len;
    if (neighbors.offsets.len != count + 1) return error.MismatchedLength;

    var triplets: std.ArrayList(sparse.Triplet) = .empty;
    defer triplets.deinit(alloc);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        try triplets.append(alloc, .{ .row = i, .col = i, .value = 1.0 });
        for (neighbors.neighborsOf(i)) |j| {
            if (j <= i) continue;
            const value = overlap(orbitals[i], orbitals[j]);
            if (value == 0.0) continue;
            try triplets.append(alloc, .{ .row = i, .col = j, .value = value });
            try triplets.append(alloc, .{ .row = j, .col = i, .value = value });
        }
    }

    return sparse.CsrMatrix.initFromTriplets(alloc, count, count, triplets.items);
}

pub fn buildKineticCsr(
    alloc: std.mem.Allocator,
    orbitals: []const Orbital,
    neighbors: neighbor_list.NeighborList,
) !sparse.CsrMatrix {
    const count = orbitals.len;
    if (neighbors.offsets.len != count + 1) return error.MismatchedLength;

    var triplets: std.ArrayList(sparse.Triplet) = .empty;
    defer triplets.deinit(alloc);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const diag = kineticIntegral(orbitals[i], orbitals[i]);
        if (diag != 0.0) {
            try triplets.append(alloc, .{ .row = i, .col = i, .value = diag });
        }
        for (neighbors.neighborsOf(i)) |j| {
            if (j <= i) continue;
            const value = kineticIntegral(orbitals[i], orbitals[j]);
            if (value == 0.0) continue;
            try triplets.append(alloc, .{ .row = i, .col = j, .value = value });
            try triplets.append(alloc, .{ .row = j, .col = i, .value = value });
        }
    }

    return sparse.CsrMatrix.initFromTriplets(alloc, count, count, triplets.items);
}

pub fn buildOverlapCsrFromCenters(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    sigma: f64,
    cutoff: f64,
    pbc: neighbor_list.Pbc,
    cell: math.Mat3,
) !sparse.CsrMatrix {
    if (centers.len == 0) return error.InvalidShape;
    if (sigma <= 0.0) return error.InvalidSigma;
    var list = try neighbor_list.NeighborList.init(alloc, cell, pbc, centers, cutoff);
    defer list.deinit(alloc);

    var orbitals = try alloc.alloc(Orbital, centers.len);
    defer alloc.free(orbitals);
    const alpha = 1.0 / (sigma * sigma);
    for (centers, 0..) |center, i| {
        orbitals[i] = .{ .center = center, .alpha = alpha, .cutoff = cutoff };
    }
    return buildOverlapCsr(alloc, orbitals, list);
}

pub fn buildKineticCsrFromCenters(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    sigma: f64,
    cutoff: f64,
    pbc: neighbor_list.Pbc,
    cell: math.Mat3,
) !sparse.CsrMatrix {
    if (centers.len == 0) return error.InvalidShape;
    if (sigma <= 0.0) return error.InvalidSigma;
    var list = try neighbor_list.NeighborList.init(alloc, cell, pbc, centers, cutoff);
    defer list.deinit(alloc);

    var orbitals = try alloc.alloc(Orbital, centers.len);
    defer alloc.free(orbitals);
    const alpha = 1.0 / (sigma * sigma);
    for (centers, 0..) |center, i| {
        orbitals[i] = .{ .center = center, .alpha = alpha, .cutoff = cutoff };
    }
    return buildKineticCsr(alloc, orbitals, list);
}

/// Build orbitals from centers with s-type only
pub fn buildOrbitalsS(alloc: std.mem.Allocator, centers: []const math.Vec3, alpha: f64, cutoff: f64) ![]Orbital {
    const orbitals = try alloc.alloc(Orbital, centers.len);
    for (centers, 0..) |center, i| {
        orbitals[i] = .{ .center = center, .alpha = alpha, .cutoff = cutoff, .angular = .s };
    }
    return orbitals;
}

/// Build orbitals from centers with sp basis (1 s + 3 p per center)
/// Returns array of size 4 * centers.len
pub fn buildOrbitalsSP(alloc: std.mem.Allocator, centers: []const math.Vec3, alpha: f64, cutoff: f64) ![]Orbital {
    const n_orb = 4 * centers.len;
    const orbitals = try alloc.alloc(Orbital, n_orb);
    for (centers, 0..) |center, i| {
        const base = i * 4;
        orbitals[base + 0] = .{ .center = center, .alpha = alpha, .cutoff = cutoff, .angular = .s };
        orbitals[base + 1] = .{ .center = center, .alpha = alpha, .cutoff = cutoff, .angular = .px };
        orbitals[base + 2] = .{ .center = center, .alpha = alpha, .cutoff = cutoff, .angular = .py };
        orbitals[base + 3] = .{ .center = center, .alpha = alpha, .cutoff = cutoff, .angular = .pz };
    }
    return orbitals;
}

/// Basis type for orbital construction
pub const BasisType = enum {
    s_only, // Only s-type orbitals (1 per center)
    sp, // s and p-type orbitals (4 per center)
};

/// Build orbitals from centers with specified basis type
pub fn buildOrbitals(alloc: std.mem.Allocator, centers: []const math.Vec3, alpha: f64, cutoff: f64, basis: BasisType) ![]Orbital {
    return switch (basis) {
        .s_only => buildOrbitalsS(alloc, centers, alpha, cutoff),
        .sp => buildOrbitalsSP(alloc, centers, alpha, cutoff),
    };
}

test "overlap decays and respects cutoff" {
    const a = Orbital{ .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 1.0 };
    const b = Orbital{ .center = .{ .x = 0.5, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 1.0 };
    const c = Orbital{ .center = .{ .x = 3.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 1.0 };
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), overlap(a, a), 1e-12);
    try std.testing.expect(overlap(a, b) > 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), overlap(a, c), 1e-12);
}

test "overlap normalized and kinetic matches analytic value" {
    const a = Orbital{ .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .alpha = 0.8, .cutoff = 10.0 };
    const value = overlapIntegral(a, a);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), value, 1e-12);
    const kinetic = kineticIntegral(a, a);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2), kinetic, 1e-12);
}

test "buildOverlapCsr uses neighbor list" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    );
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.5, .y = 0.0, .z = 0.0 },
    };
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    var list = try neighbor_list.NeighborList.init(alloc, cell, pbc, positions[0..], 1.1);
    defer list.deinit(alloc);

    const orbitals = [_]Orbital{
        .{ .center = positions[0], .alpha = 1.0, .cutoff = 1.0 },
        .{ .center = positions[1], .alpha = 1.0, .cutoff = 1.0 },
    };
    var csr = try buildOverlapCsr(alloc, orbitals[0..], list);
    defer csr.deinit(alloc);
    const x = [_]f64{ 1.0, 1.0 };
    var out = [_]f64{ 0.0, 0.0 };
    try csr.mulVec(x[0..], out[0..]);
    const expected = 1.0 + overlap(orbitals[0], orbitals[1]);
    try std.testing.expectApproxEqAbs(expected, out[0], 1e-12);
    try std.testing.expectApproxEqAbs(expected, out[1], 1e-12);
}

test "buildKineticCsrFromCenters uses neighbor list" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    );
    const centers = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.5, .y = 0.0, .z = 0.0 },
    };
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    var csr = try buildKineticCsrFromCenters(alloc, centers[0..], 0.5, 1.1, pbc, cell);
    defer csr.deinit(alloc);
    try std.testing.expect(csr.valueAt(0, 0) > 0.0);
    try std.testing.expect(csr.valueAt(0, 1) > 0.0);
}

test "buildOverlapCsrFromCenters applies PBC" {
    const alloc = std.testing.allocator;
    const cell = math.Mat3.fromRows(
        .{ .x = 1.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 1.0 },
    );
    const centers = [_]math.Vec3{
        .{ .x = 0.05, .y = 0.0, .z = 0.0 },
        .{ .x = 0.95, .y = 0.0, .z = 0.0 },
    };
    const pbc = neighbor_list.Pbc{ .x = true, .y = false, .z = false };
    var csr = try buildOverlapCsrFromCenters(alloc, centers[0..], 0.1, 0.2, pbc, cell);
    defer csr.deinit(alloc);
    try std.testing.expect(csr.valueAt(0, 1) > 0.0);
}

test "p-type Gaussian overlap orthogonality at same center" {
    // At the same center, s and p orbitals should be orthogonal
    const a_s = Orbital{ .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 10.0, .angular = .s };
    const a_px = Orbital{ .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 10.0, .angular = .px };
    const a_py = Orbital{ .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 10.0, .angular = .py };
    const a_pz = Orbital{ .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 10.0, .angular = .pz };

    // s-s should be 1 (normalized)
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), overlap(a_s, a_s), 1e-10);

    // s-p should be 0 at same center
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), overlap(a_s, a_px), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), overlap(a_s, a_py), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), overlap(a_s, a_pz), 1e-10);

    // p-p same direction should be 1 (normalized)
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), overlap(a_px, a_px), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), overlap(a_py, a_py), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), overlap(a_pz, a_pz), 1e-10);

    // p-p different directions should be 0 at same center
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), overlap(a_px, a_py), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), overlap(a_px, a_pz), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), overlap(a_py, a_pz), 1e-10);
}

test "p-type Gaussian overlap non-zero at displaced centers" {
    // When centers are displaced, s-p overlap should be non-zero
    const a_s = Orbital{ .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 10.0, .angular = .s };
    const b_px = Orbital{ .center = .{ .x = 1.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 10.0, .angular = .px };
    const b_py = Orbital{ .center = .{ .x = 1.0, .y = 0.0, .z = 0.0 }, .alpha = 1.0, .cutoff = 10.0, .angular = .py };

    // s at origin, px displaced along x: should have non-zero overlap
    const s_px = overlap(a_s, b_px);
    try std.testing.expect(s_px != 0.0);

    // s at origin, py displaced along x: should be zero (orthogonal directions)
    const s_py = overlap(a_s, b_py);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), s_py, 1e-10);
}

test "buildOrbitalsSP creates correct number of orbitals" {
    const alloc = std.testing.allocator;
    const centers = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.0, .y = 0.0, .z = 0.0 },
    };
    const orbitals = try buildOrbitalsSP(alloc, centers[0..], 1.0, 5.0);
    defer alloc.free(orbitals);

    try std.testing.expectEqual(@as(usize, 8), orbitals.len);
    try std.testing.expectEqual(AngularType.s, orbitals[0].angular);
    try std.testing.expectEqual(AngularType.px, orbitals[1].angular);
    try std.testing.expectEqual(AngularType.py, orbitals[2].angular);
    try std.testing.expectEqual(AngularType.pz, orbitals[3].angular);
}
