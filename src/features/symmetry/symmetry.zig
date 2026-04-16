const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoints_mod = @import("../kpoints/kpoints.zig");

// Re-export related modules for convenience
pub const point_group = @import("point_group.zig");
pub const little_group = @import("little_group.zig");
pub const symmetry_basis = @import("symmetry_basis.zig");

pub const KPoint = struct {
    k_frac: math.Vec3,
    k_cart: math.Vec3,
    weight: f64,
};

pub const SymOp = struct {
    rot: Mat3i,
    k_rot: Mat3i,
    trans: math.Vec3,
};

pub const Mat3i = struct {
    m: [3][3]i32,

    pub fn det(self: Mat3i) i32 {
        const a = self.m;
        return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
            a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
            a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    }

    pub fn transpose(self: Mat3i) Mat3i {
        return .{ .m = .{
            .{ self.m[0][0], self.m[1][0], self.m[2][0] },
            .{ self.m[0][1], self.m[1][1], self.m[2][1] },
            .{ self.m[0][2], self.m[1][2], self.m[2][2] },
        } };
    }

    pub fn inverse(self: Mat3i) ?Mat3i {
        const det_val = self.det();
        if (det_val != 1 and det_val != -1) return null;
        const a = self.m;
        const adj = [3][3]i32{
            .{
                a[1][1] * a[2][2] - a[1][2] * a[2][1],
                -(a[0][1] * a[2][2] - a[0][2] * a[2][1]),
                a[0][1] * a[1][2] - a[0][2] * a[1][1],
            },
            .{
                -(a[1][0] * a[2][2] - a[1][2] * a[2][0]),
                a[0][0] * a[2][2] - a[0][2] * a[2][0],
                -(a[0][0] * a[1][2] - a[0][2] * a[1][0]),
            },
            .{
                a[1][0] * a[2][1] - a[1][1] * a[2][0],
                -(a[0][0] * a[2][1] - a[0][1] * a[2][0]),
                a[0][0] * a[1][1] - a[0][1] * a[1][0],
            },
        };
        var out = adj;
        if (det_val == -1) {
            var i: usize = 0;
            while (i < 3) : (i += 1) {
                var j: usize = 0;
                while (j < 3) : (j += 1) {
                    out[i][j] = -out[i][j];
                }
            }
        }
        return Mat3i{ .m = out };
    }

    pub fn mulVec(self: Mat3i, v: math.Vec3) math.Vec3 {
        return .{
            .x = @as(f64, @floatFromInt(self.m[0][0])) * v.x +
                @as(f64, @floatFromInt(self.m[0][1])) * v.y +
                @as(f64, @floatFromInt(self.m[0][2])) * v.z,
            .y = @as(f64, @floatFromInt(self.m[1][0])) * v.x +
                @as(f64, @floatFromInt(self.m[1][1])) * v.y +
                @as(f64, @floatFromInt(self.m[1][2])) * v.z,
            .z = @as(f64, @floatFromInt(self.m[2][0])) * v.x +
                @as(f64, @floatFromInt(self.m[2][1])) * v.y +
                @as(f64, @floatFromInt(self.m[2][2])) * v.z,
        };
    }

    pub fn trace(self: Mat3i) i32 {
        return self.m[0][0] + self.m[1][1] + self.m[2][2];
    }
};

/// Generate Monkhorst-Pack k-mesh.
pub fn generateKmesh(
    alloc: std.mem.Allocator,
    kmesh: [3]usize,
    recip: math.Mat3,
) ![]KPoint {
    return kpoints_mod.generateKmesh(
        alloc,
        kmesh,
        recip,
        .{ .x = 0.5, .y = 0.5, .z = 0.5 },
    );
}

/// Generate Monkhorst-Pack k-mesh with symmetry reduction.
pub fn generateKmeshSymmetry(
    alloc: std.mem.Allocator,
    io: std.Io,
    kmesh: [3]usize,
    recip: math.Mat3,
    cell: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    time_reversal: bool,
) ![]KPoint {
    return kpoints_mod.generateKmeshSymmetry(
        alloc,
        io,
        kmesh,
        .{ .x = 0.5, .y = 0.5, .z = 0.5 },
        recip,
        cell,
        atoms,
        time_reversal,
    );
}

/// Enumerate symmetry operations (rotation + translation) for the given cell.
pub fn getSymmetryOps(
    alloc: std.mem.Allocator,
    cell: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    tol: f64,
) ![]SymOp {
    return try buildSymmetryOps(alloc, cell, atoms, tol);
}

const AtomFrac = struct {
    frac: math.Vec3,
    species_index: usize,
};

fn buildSymmetryOps(
    alloc: std.mem.Allocator,
    cell: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    tol: f64,
) ![]SymOp {
    if (atoms.len == 0) {
        return try alloc.alloc(SymOp, 0);
    }

    // Enumerate integer rotation matrices with entries in {-1,0,1},
    // keep those that preserve the lattice metric, then check atom mapping.
    const metric = latticeMetric(cell);
    const recip = math.reciprocal(cell);
    const atom_fracs = try buildAtomFracs(alloc, recip, atoms);
    defer alloc.free(atom_fracs);

    var ops_list: std.ArrayList(SymOp) = .empty;
    errdefer ops_list.deinit(alloc);
    const matched = try alloc.alloc(bool, atom_fracs.len);
    defer alloc.free(matched);

    var m0: i32 = -1;
    while (m0 <= 1) : (m0 += 1) {
        var m1: i32 = -1;
        while (m1 <= 1) : (m1 += 1) {
            var m2: i32 = -1;
            while (m2 <= 1) : (m2 += 1) {
                var m3: i32 = -1;
                while (m3 <= 1) : (m3 += 1) {
                    var m4: i32 = -1;
                    while (m4 <= 1) : (m4 += 1) {
                        var m5: i32 = -1;
                        while (m5 <= 1) : (m5 += 1) {
                            var m6: i32 = -1;
                            while (m6 <= 1) : (m6 += 1) {
                                var m7: i32 = -1;
                                while (m7 <= 1) : (m7 += 1) {
                                    var m8: i32 = -1;
                                    while (m8 <= 1) : (m8 += 1) {
                                        const rot = Mat3i{ .m = .{
                                            .{ m0, m1, m2 },
                                            .{ m3, m4, m5 },
                                            .{ m6, m7, m8 },
                                        } };
                                        const det_val = rot.det();
                                        if (det_val != 1 and det_val != -1) continue;
                                        if (!latticeSymmetric(rot, metric, tol)) continue;
                                        const translations = try findTranslations(alloc, rot, atom_fracs, matched, tol);
                                        defer alloc.free(translations);
                                        if (translations.len == 0) continue;
                                        const inv = rot.inverse() orelse continue;
                                        const k_rot = inv.transpose();
                                        for (translations) |t| {
                                            if (hasSymOp(ops_list.items, rot, t, tol)) continue;
                                            try ops_list.append(alloc, .{ .rot = rot, .k_rot = k_rot, .trans = t });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return try ops_list.toOwnedSlice(alloc);
}

fn latticeMetric(cell: math.Mat3) [3][3]f64 {
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    return .{
        .{ math.Vec3.dot(a1, a1), math.Vec3.dot(a1, a2), math.Vec3.dot(a1, a3) },
        .{ math.Vec3.dot(a2, a1), math.Vec3.dot(a2, a2), math.Vec3.dot(a2, a3) },
        .{ math.Vec3.dot(a3, a1), math.Vec3.dot(a3, a2), math.Vec3.dot(a3, a3) },
    };
}

fn latticeSymmetric(rot: Mat3i, metric: [3][3]f64, tol: f64) bool {
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 3) : (j += 1) {
            var sum: f64 = 0.0;
            var a: usize = 0;
            while (a < 3) : (a += 1) {
                var b: usize = 0;
                while (b < 3) : (b += 1) {
                    sum += @as(f64, @floatFromInt(rot.m[a][i])) * metric[a][b] * @as(f64, @floatFromInt(rot.m[b][j]));
                }
            }
            if (@abs(sum - metric[i][j]) > tol) return false;
        }
    }
    return true;
}

fn buildAtomFracs(alloc: std.mem.Allocator, recip: math.Mat3, atoms: []const hamiltonian.AtomData) ![]AtomFrac {
    const two_pi = 2.0 * std.math.pi;
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);
    const list = try alloc.alloc(AtomFrac, atoms.len);
    var i: usize = 0;
    while (i < atoms.len) : (i += 1) {
        const pos = atoms[i].position;
        const fx = math.Vec3.dot(b1, pos) / two_pi;
        const fy = math.Vec3.dot(b2, pos) / two_pi;
        const fz = math.Vec3.dot(b3, pos) / two_pi;
        list[i] = .{
            .frac = wrap01(.{ .x = fx, .y = fy, .z = fz }),
            .species_index = atoms[i].species_index,
        };
    }
    return list;
}

fn findTranslations(
    alloc: std.mem.Allocator,
    rot: Mat3i,
    atoms: []AtomFrac,
    matched: []bool,
    tol: f64,
) ![]math.Vec3 {
    var list: std.ArrayList(math.Vec3) = .empty;
    errdefer list.deinit(alloc);

    const ref = atoms[0];
    const rot_ref = rot.mulVec(ref.frac);
    for (atoms) |target| {
        if (target.species_index != ref.species_index) continue;
        const t = wrap01(math.Vec3.sub(target.frac, rot_ref));
        if (translationExists(list.items, t, tol)) continue;
        if (translationValid(rot, t, atoms, matched, tol)) {
            try list.append(alloc, t);
        }
    }

    return try list.toOwnedSlice(alloc);
}

fn translationValid(rot: Mat3i, t: math.Vec3, atoms: []AtomFrac, matched: []bool, tol: f64) bool {
    @memset(matched, false);
    var i: usize = 0;
    while (i < atoms.len) : (i += 1) {
        const target = wrap01(math.Vec3.add(rot.mulVec(atoms[i].frac), t));
        var found = false;
        var j: usize = 0;
        while (j < atoms.len) : (j += 1) {
            if (matched[j]) continue;
            if (atoms[j].species_index != atoms[i].species_index) continue;
            if (fracClose(target, atoms[j].frac, tol)) {
                matched[j] = true;
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

fn translationExists(list: []math.Vec3, t: math.Vec3, tol: f64) bool {
    for (list) |existing| {
        if (fracClose(existing, t, tol)) return true;
    }
    return false;
}

fn hasSymOp(list: []SymOp, rot: Mat3i, t: math.Vec3, tol: f64) bool {
    for (list) |op| {
        if (!mat3iEqual(op.rot, rot)) continue;
        if (fracClose(op.trans, t, tol)) return true;
    }
    return false;
}

fn mat3iEqual(a: Mat3i, b: Mat3i) bool {
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 3) : (j += 1) {
            if (a.m[i][j] != b.m[i][j]) return false;
        }
    }
    return true;
}

fn wrap01(v: math.Vec3) math.Vec3 {
    return .{
        .x = v.x - std.math.floor(v.x),
        .y = v.y - std.math.floor(v.y),
        .z = v.z - std.math.floor(v.z),
    };
}

fn wrapCentered(v: math.Vec3) math.Vec3 {
    return .{
        .x = v.x - std.math.round(v.x),
        .y = v.y - std.math.round(v.y),
        .z = v.z - std.math.round(v.z),
    };
}

fn fracClose(a: math.Vec3, b: math.Vec3, tol: f64) bool {
    const d = wrapCentered(math.Vec3.sub(a, b));
    return @abs(d.x) < tol and @abs(d.y) < tol and @abs(d.z) < tol;
}

/// Irreducible representation index for symmetry block-diagonalization.
pub const Irrep = enum(u8) {
    even, // χ = +1 under mirror
    odd, // χ = -1 under mirror
    none, // no special symmetry
};

/// A pair of G-vector indices related by mirror symmetry.
pub const MirrorPair = struct {
    i: usize, // Index of G with k > 0
    j: usize, // Index of G' with k < 0 (σ_y G = G')
};

/// Result of classifying G-vectors by irreducible representation.
pub const IrrepClassification = struct {
    invariant_indices: []usize, // G-vectors with k=0 (invariant under σ_y)
    pairs: []MirrorPair, // Pairs (G, σ_y G) for k ≠ 0
    has_mirror: bool,

    pub fn deinit(self: *IrrepClassification, alloc: std.mem.Allocator) void {
        if (self.invariant_indices.len > 0) alloc.free(self.invariant_indices);
        if (self.pairs.len > 0) alloc.free(self.pairs);
    }
};

/// Check if k-point lies on mirror plane (ky = 0 in fractional coords).
/// Returns true if k is invariant under σ_y: (kx, ky, kz) → (kx, -ky, kz).
pub fn hasYMirrorSymmetry(k_frac: math.Vec3, tol: f64) bool {
    // k is on mirror plane if ky = 0 (or ky = 0.5 which maps to itself)
    const ky = k_frac.y;
    return @abs(ky) < tol or @abs(ky - 0.5) < tol or @abs(ky + 0.5) < tol;
}

/// Classify G-vectors by σ_y mirror symmetry for block-diagonalization.
/// For a k-point on the M-Γ path (ky=0), G-vectors split into:
/// - Invariant: G with k_index=0 (transform as χ=+1 under σ_y)
/// - Pairs: (h,+k,l) and (h,-k,l) form even (+) and odd (-) combinations
pub fn classifyGVectorsByMirror(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    k_frac: math.Vec3,
    tol: f64,
) !IrrepClassification {
    const has_mirror = hasYMirrorSymmetry(k_frac, tol);

    if (!has_mirror) {
        // No mirror symmetry - no block-diagonalization possible
        return IrrepClassification{
            .invariant_indices = try alloc.alloc(usize, 0),
            .pairs = try alloc.alloc(MirrorPair, 0),
            .has_mirror = false,
        };
    }

    var invariant_list: std.ArrayList(usize) = .empty;
    errdefer invariant_list.deinit(alloc);
    var pair_list: std.ArrayList(MirrorPair) = .empty;
    errdefer pair_list.deinit(alloc);

    const processed = try alloc.alloc(bool, gvecs.len);
    defer alloc.free(processed);
    @memset(processed, false);

    var i: usize = 0;
    while (i < gvecs.len) : (i += 1) {
        if (processed[i]) continue;

        const g = gvecs[i];
        if (g.k == 0) {
            // G with k=0 is invariant under σ_y
            try invariant_list.append(alloc, i);
            processed[i] = true;
        } else {
            // Find partner G' = (h, -k, l)
            var partner_idx: ?usize = null;
            var j: usize = i + 1;
            while (j < gvecs.len) : (j += 1) {
                if (processed[j]) continue;
                const gp = gvecs[j];
                if (gp.h == g.h and gp.k == -g.k and gp.l == g.l) {
                    partner_idx = j;
                    break;
                }
            }

            if (partner_idx) |pi| {
                // Pair found: record indices with g.k > 0 as first
                if (g.k > 0) {
                    try pair_list.append(alloc, .{ .i = i, .j = pi });
                } else {
                    try pair_list.append(alloc, .{ .i = pi, .j = i });
                }
                processed[i] = true;
                processed[pi] = true;
            } else {
                // No partner found (shouldn't happen for complete basis)
                // Treat as invariant
                try invariant_list.append(alloc, i);
                processed[i] = true;
            }
        }
    }

    return IrrepClassification{
        .invariant_indices = try invariant_list.toOwnedSlice(alloc),
        .pairs = try pair_list.toOwnedSlice(alloc),
        .has_mirror = true,
    };
}

const GVector = @import("../plane_wave/basis.zig").GVector;
const linalg = @import("../linalg/linalg.zig");

/// Result of symmetry block-diagonal eigenvalue computation.
pub const BlockEigenResult = struct {
    eigenvalues: []f64,
    even_count: usize,
    odd_count: usize,

    pub fn deinit(self: *BlockEigenResult, alloc: std.mem.Allocator) void {
        if (self.eigenvalues.len > 0) alloc.free(self.eigenvalues);
    }
};

/// Build symmetry-adapted Hamiltonian blocks and solve eigenvalue problems separately.
/// This properly handles band crossings at high-symmetry points by avoiding
/// spurious mixing between states of different symmetry.
///
/// For σ_y mirror symmetry:
/// - Invariant G-vectors (k=0) belong to even irrep
/// - Pairs (G, G') where G'=σ_y G transform as:
///   |+⟩ = (|G⟩ + |G'⟩) / √2  (even, χ=+1)
///   |-⟩ = (|G⟩ - |G'⟩) / √2  (odd, χ=-1)
///
/// The transformed Hamiltonian elements for pairs are:
///   H_++ = (H_ii + H_jj + H_ij + H_ji) / 2
///   H_-- = (H_ii + H_jj - H_ij - H_ji) / 2
pub fn blockDiagonalEigenvalues(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    full_h: []math.Complex,
    n: usize,
    classification: IrrepClassification,
) !BlockEigenResult {
    if (!classification.has_mirror) {
        // No symmetry - solve full matrix
        const values = try linalg.hermitianEigenvalues(alloc, backend, n, full_h);
        return BlockEigenResult{
            .eigenvalues = values,
            .even_count = n,
            .odd_count = 0,
        };
    }

    const n_inv = classification.invariant_indices.len;
    const n_pairs = classification.pairs.len;
    const n_even = n_inv + n_pairs;
    const n_odd = n_pairs;

    // Build even block: invariant + even combinations of pairs
    var even_values: []f64 = &[_]f64{};
    if (n_even > 0) {
        const h_even = try alloc.alloc(math.Complex, n_even * n_even);
        defer alloc.free(h_even);
        @memset(h_even, math.complex.init(0.0, 0.0));

        // Invariant-invariant block
        for (classification.invariant_indices, 0..) |idx_b, b| {
            for (classification.invariant_indices, 0..) |idx_a, a| {
                h_even[a + b * n_even] = full_h[idx_a + idx_b * n];
            }
        }

        // Invariant-pair (even) coupling
        for (classification.pairs, 0..) |pair, p| {
            const col = n_inv + p;
            for (classification.invariant_indices, 0..) |idx_inv, row| {
                // ⟨inv|H|+⟩ = (⟨inv|H|G⟩ + ⟨inv|H|G'⟩) / √2
                const h_i = full_h[idx_inv + pair.i * n];
                const h_j = full_h[idx_inv + pair.j * n];
                const val = math.complex.scale(math.complex.add(h_i, h_j), 1.0 / std.math.sqrt(2.0));
                h_even[row + col * n_even] = val;
                h_even[col + row * n_even] = math.complex.conj(val);
            }
        }

        // Pair-pair (even-even) coupling
        for (classification.pairs, 0..) |pair_b, b| {
            const col = n_inv + b;
            for (classification.pairs, 0..) |pair_a, a| {
                const row = n_inv + a;
                // ⟨+_a|H|+_b⟩ = (H_ii + H_jj + H_ij + H_ji) / 2
                // where i,j are indices of pair_a and pair_b
                const h_ii = full_h[pair_a.i + pair_b.i * n];
                const h_jj = full_h[pair_a.j + pair_b.j * n];
                const h_ij = full_h[pair_a.i + pair_b.j * n];
                const h_ji = full_h[pair_a.j + pair_b.i * n];
                const sum = math.complex.add(math.complex.add(h_ii, h_jj), math.complex.add(h_ij, h_ji));
                h_even[row + col * n_even] = math.complex.scale(sum, 0.5);
            }
        }

        even_values = try linalg.hermitianEigenvalues(alloc, backend, n_even, h_even);
    }
    errdefer if (even_values.len > 0) alloc.free(even_values);

    // Build odd block: odd combinations of pairs only
    var odd_values: []f64 = &[_]f64{};
    if (n_odd > 0) {
        const h_odd = try alloc.alloc(math.Complex, n_odd * n_odd);
        defer alloc.free(h_odd);
        @memset(h_odd, math.complex.init(0.0, 0.0));

        // Pair-pair (odd-odd) coupling
        for (classification.pairs, 0..) |pair_b, b| {
            for (classification.pairs, 0..) |pair_a, a| {
                // ⟨-_a|H|-_b⟩ = (H_ii + H_jj - H_ij - H_ji) / 2
                const h_ii = full_h[pair_a.i + pair_b.i * n];
                const h_jj = full_h[pair_a.j + pair_b.j * n];
                const h_ij = full_h[pair_a.i + pair_b.j * n];
                const h_ji = full_h[pair_a.j + pair_b.i * n];
                const sum = math.complex.sub(math.complex.add(h_ii, h_jj), math.complex.add(h_ij, h_ji));
                h_odd[a + b * n_odd] = math.complex.scale(sum, 0.5);
            }
        }

        odd_values = try linalg.hermitianEigenvalues(alloc, backend, n_odd, h_odd);
    }
    errdefer if (odd_values.len > 0) alloc.free(odd_values);

    // Merge and sort all eigenvalues
    const total = n_even + n_odd;
    const merged = try alloc.alloc(f64, total);
    errdefer alloc.free(merged);

    if (even_values.len > 0) {
        @memcpy(merged[0..n_even], even_values);
        alloc.free(even_values);
    }
    if (odd_values.len > 0) {
        @memcpy(merged[n_even..total], odd_values);
        alloc.free(odd_values);
    }

    // Sort eigenvalues in ascending order
    std.sort.block(f64, merged, {}, std.sort.asc(f64));

    return BlockEigenResult{
        .eigenvalues = merged,
        .even_count = n_even,
        .odd_count = n_odd,
    };
}
