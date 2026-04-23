const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("symmetry.zig");
const point_group = @import("point_group.zig");

/// The little group (stabilizer) of a k-point.
/// Contains all symmetry operations that leave k invariant (modulo reciprocal lattice vectors).
pub const LittleGroup = struct {
    k_frac: math.Vec3,
    ops: []symmetry.SymOp,
    point_group_type: point_group.PointGroupType,

    /// Free allocated operations.
    pub fn deinit(self: *LittleGroup, alloc: std.mem.Allocator) void {
        if (self.ops.len > 0) {
            alloc.free(self.ops);
        }
    }

    /// Check if this little group has a specific symmetry operation (by rotation part).
    pub fn has_operation(self: LittleGroup, rot: symmetry.Mat3i) bool {
        for (self.ops) |op| {
            if (mat3i_equal(op.rot, rot)) return true;
        }
        return false;
    }

    /// Check if there's a mirror operation (det = -1, trace = 1).
    pub fn has_mirror(self: LittleGroup) bool {
        for (self.ops) |op| {
            if (op.rot.det() == -1 and op.rot.trace() == 1) return true;
        }
        return false;
    }

    /// Get the order (number of operations) of the little group.
    pub fn order(self: LittleGroup) usize {
        return self.ops.len;
    }
};

/// Compute the little group of a k-point.
/// The little group consists of all space group operations S such that:
/// S * k = k + G (where G is a reciprocal lattice vector)
///
/// In fractional coordinates: S_k * k_frac ≡ k_frac (mod 1)
/// where S_k = (R^-1)^T is the reciprocal space rotation.
pub fn compute_little_group(
    alloc: std.mem.Allocator,
    k_frac: math.Vec3,
    space_group_ops: []const symmetry.SymOp,
    tol: f64,
) !LittleGroup {
    var little_ops: std.ArrayList(symmetry.SymOp) = .empty;
    errdefer little_ops.deinit(alloc);

    for (space_group_ops) |op| {
        // Apply k-space rotation: k' = S_k * k
        const k_prime = op.k_rot.mul_vec(k_frac);

        // Check if k' ≡ k (mod 1), i.e., k' - k is a lattice vector
        if (is_equivalent_k_point(k_prime, k_frac, tol)) {
            try little_ops.append(alloc, op);
        }
    }

    // If no operations found, at least include identity
    if (little_ops.items.len == 0) {
        const identity = symmetry.SymOp{
            .rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .k_rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .trans = math.Vec3{ .x = 0, .y = 0, .z = 0 },
        };
        try little_ops.append(alloc, identity);
    }

    const ops = try little_ops.toOwnedSlice(alloc);
    const pg_type = point_group.identify_point_group(ops);

    return LittleGroup{
        .k_frac = k_frac,
        .ops = ops,
        .point_group_type = pg_type,
    };
}

/// Check if two k-points are equivalent (differ by a reciprocal lattice vector).
fn is_equivalent_k_point(k1: math.Vec3, k2: math.Vec3, tol: f64) bool {
    const diff = math.Vec3.sub(k1, k2);
    // Each component should be close to an integer
    const dx = diff.x - std.math.round(diff.x);
    const dy = diff.y - std.math.round(diff.y);
    const dz = diff.z - std.math.round(diff.z);
    return @abs(dx) < tol and @abs(dy) < tol and @abs(dz) < tol;
}

/// Check if two Mat3i are equal.
fn mat3i_equal(a: symmetry.Mat3i, b: symmetry.Mat3i) bool {
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 3) : (j += 1) {
            if (a.m[i][j] != b.m[i][j]) return false;
        }
    }
    return true;
}

/// Classification of a G-vector under the little group.
pub const GVectorClass = struct {
    /// Index of the G-vector in the basis.
    index: usize,
    /// Indices of all G-vectors in the same orbit (including this one).
    orbit: []usize,
    /// Irreducible representation this G-vector belongs to.
    irrep: point_group.IrrepLabel,
    /// Phase factor for symmetry-adapted combination (for multi-dimensional irreps).
    phase: math.Complex,
};

/// Result of symmetry analysis for a plane-wave basis.
pub const BasisSymmetryInfo = struct {
    little_group: LittleGroup,
    /// For each irrep, list of basis function indices in that irrep.
    irrep_blocks: []IrrepBlock,

    pub fn deinit(self: *BasisSymmetryInfo, alloc: std.mem.Allocator) void {
        for (self.irrep_blocks) |*block| {
            block.deinit(alloc);
        }
        if (self.irrep_blocks.len > 0) {
            alloc.free(self.irrep_blocks);
        }
        self.little_group.deinit(alloc);
    }
};

/// Information about a G-vector pair under mirror symmetry.
pub const GVectorPair = struct {
    /// Index of G in the original basis
    g_index: usize,
    /// Index of σG (the mirror partner) in the original basis
    partner_index: usize,
};

/// Information about a G-vector triplet under C3 rotation.
/// These form 3-member orbits: G, C3*G, C3^2*G
pub const GVectorTriplet = struct {
    /// Index of G in the original basis
    g1_index: usize,
    /// Index of C3*G in the original basis
    g2_index: usize,
    /// Index of C3^2*G in the original basis
    g3_index: usize,
};

/// A block of basis functions belonging to the same irrep.
/// For Cs symmetry:
/// - A' block contains: invariant G-vectors + |+⟩ = (|G⟩+|σG⟩)/√2 states
/// - A'' block contains: |-⟩ = (|G⟩-|σG⟩)/√2 states
/// For C3v symmetry:
/// - A1 block contains: invariant G-vectors + symmetric combination from triplets
/// - E block contains: 2D representation states from triplets (degenerate pairs)
pub const IrrepBlock = struct {
    irrep: point_group.IrrepLabel,
    /// Indices of invariant G-vectors (transform trivially).
    invariant_indices: []usize,
    /// G-vector pairs (for Cs symmetry).
    pairs: []GVectorPair,
    /// G-vector triplets (for C3v symmetry).
    /// For A1: contributes |A1⟩ = (|G1⟩ + |G2⟩ + |G3⟩)/√3
    /// For E: contributes 2 states per triplet
    triplets: []GVectorTriplet,

    /// Total number of basis functions in this block.
    pub fn size(self: IrrepBlock) usize {
        // For E representation, each triplet contributes 2 states
        const triplet_contrib = if (self.irrep == .e or self.irrep == .e1 or self.irrep == .e2)
            self.triplets.len * 2
        else
            self.triplets.len; // A1/A2: 1 state per triplet
        return self.invariant_indices.len + self.pairs.len + triplet_contrib;
    }

    pub fn deinit(self: *IrrepBlock, alloc: std.mem.Allocator) void {
        if (self.invariant_indices.len > 0) alloc.free(self.invariant_indices);
        if (self.pairs.len > 0) alloc.free(self.pairs);
        if (self.triplets.len > 0) alloc.free(self.triplets);
    }
};

/// Find the mirror operation (det=-1) in the little group, if any.
fn find_mirror_operation(little_grp: LittleGroup) ?symmetry.SymOp {
    for (little_grp.ops) |op| {
        if (op.rot.det() == -1) return op;
    }
    return null;
}

/// Classify G-vectors under a mirror into invariant indices and (G, σG) pairs.
fn classify_g_vectors_by_mirror_frac(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    mirror: symmetry.SymOp,
    invariant_list: *std.ArrayList(usize),
    pair_list: *std.ArrayList(GVectorPair),
) !void {
    const processed = try alloc.alloc(bool, gvecs.len);
    defer alloc.free(processed);

    @memset(processed, false);

    var i: usize = 0;
    while (i < gvecs.len) : (i += 1) {
        if (processed[i]) continue;

        const g = gvecs[i];
        const g_int = [3]i32{ g.h, g.k, g.l };

        // Apply mirror: G' = mirror.k_rot * G
        const r = mirror.k_rot.m;
        const g_prime_int = [3]i32{
            r[0][0] * g_int[0] + r[0][1] * g_int[1] + r[0][2] * g_int[2],
            r[1][0] * g_int[0] + r[1][1] * g_int[1] + r[1][2] * g_int[2],
            r[2][0] * g_int[0] + r[2][1] * g_int[1] + r[2][2] * g_int[2],
        };

        // Check if G is invariant (G' = G)
        const is_inv = g_prime_int[0] == g_int[0] and
            g_prime_int[1] == g_int[1] and
            g_prime_int[2] == g_int[2];
        if (is_inv) {
            // Invariant under mirror -> belongs only to A'
            try invariant_list.append(alloc, i);
            processed[i] = true;
        } else {
            // Find the partner G'
            var partner: ?usize = null;
            var j: usize = 0;
            while (j < gvecs.len) : (j += 1) {
                if (j == i or processed[j]) continue;
                const gj = gvecs[j];
                if (gj.h == g_prime_int[0] and gj.k == g_prime_int[1] and gj.l == g_prime_int[2]) {
                    partner = j;
                    break;
                }
            }

            if (partner) |p| {
                // Found a pair (G, σG)
                // |+⟩ = (|G⟩ + |σG⟩)/√2 -> A'
                // |-⟩ = (|G⟩ - |σG⟩)/√2 -> A''
                try pair_list.append(alloc, .{
                    .g_index = i,
                    .partner_index = p,
                });
                processed[i] = true;
                processed[p] = true;
            } else {
                // Partner not in basis - treat as invariant
                // This shouldn't happen if the basis is complete, but handle it gracefully
                try invariant_list.append(alloc, i);
                processed[i] = true;
            }
        }
    }
}

/// Build the A' and A'' irrep blocks from invariant G-vectors and mirror pairs.
fn build_mirror_irrep_blocks(
    alloc: std.mem.Allocator,
    little_grp: LittleGroup,
    invariant_indices: []usize,
    pairs: []GVectorPair,
) !BasisSymmetryInfo {
    var blocks: std.ArrayList(IrrepBlock) = .empty;
    errdefer {
        for (blocks.items) |*b| b.deinit(alloc);
        blocks.deinit(alloc);
    }

    // A' block: contains invariant G-vectors and |+⟩ states from pairs
    // Block size = num_invariant + num_pairs
    try blocks.append(alloc, .{
        .irrep = .a_prime,
        .invariant_indices = invariant_indices,
        .pairs = pairs,
        .triplets = try alloc.alloc(GVectorTriplet, 0),
    });

    // A'' block: contains only |-⟩ states from pairs
    // Block size = num_pairs
    // We need to duplicate the pairs array for A'' block
    if (pairs.len > 0) {
        const pairs_copy = try alloc.alloc(GVectorPair, pairs.len);
        @memcpy(pairs_copy, pairs);
        try blocks.append(alloc, .{
            .irrep = .a_double_prime,
            .invariant_indices = try alloc.alloc(usize, 0), // No invariants in A''
            .pairs = pairs_copy,
            .triplets = try alloc.alloc(GVectorTriplet, 0),
        });
    }

    return BasisSymmetryInfo{
        .little_group = little_grp,
        .irrep_blocks = try blocks.toOwnedSlice(alloc),
    };
}

/// Analyze basis symmetry for a Cs little group (single mirror).
/// This is the relevant case for the M-Γ path in graphene.
///
/// For Cs symmetry:
/// - Invariant G-vectors (σG = G): belong only to A'
/// - Paired G-vectors (σG ≠ G): form |+⟩ = (|G⟩+|σG⟩)/√2 in A'
///                               and |-⟩ = (|G⟩-|σG⟩)/√2 in A''
pub fn analyze_cs(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    little_grp: LittleGroup,
) !BasisSymmetryInfo {
    const mirror_op = find_mirror_operation(little_grp);
    if (mirror_op == null) {
        // No mirror found - return trivial classification
        return trivial_classification(alloc, gvecs, little_grp);
    }
    const mirror = mirror_op.?;

    // Lists to collect invariant G-vectors and pairs
    var invariant_list: std.ArrayList(usize) = .empty;
    errdefer invariant_list.deinit(alloc);

    var pair_list: std.ArrayList(GVectorPair) = .empty;
    errdefer pair_list.deinit(alloc);

    try classify_g_vectors_by_mirror_frac(alloc, gvecs, mirror, &invariant_list, &pair_list);

    const invariant_indices = try invariant_list.toOwnedSlice(alloc);
    const pairs = try pair_list.toOwnedSlice(alloc);

    return try build_mirror_irrep_blocks(alloc, little_grp, invariant_indices, pairs);
}

/// Trivial classification when no useful symmetry is found.
/// All basis functions are treated as invariant in a single A block.
fn trivial_classification(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    little_grp: LittleGroup,
) !BasisSymmetryInfo {
    // All basis functions in one block as invariant
    const indices = try alloc.alloc(usize, gvecs.len);
    for (indices, 0..) |*idx, i| idx.* = i;

    const blocks = try alloc.alloc(IrrepBlock, 1);
    blocks[0] = .{
        .irrep = .a,
        .invariant_indices = indices,
        .pairs = try alloc.alloc(GVectorPair, 0),
        .triplets = try alloc.alloc(GVectorTriplet, 0),
    };

    return BasisSymmetryInfo{
        .little_group = little_grp,
        .irrep_blocks = blocks,
    };
}

/// Classify G-vectors under the Cartesian σ_y mirror (Gy → -Gy).
fn classify_g_vectors_by_cartesian_y(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    tol: f64,
    invariant_list: *std.ArrayList(usize),
    pair_list: *std.ArrayList(GVectorPair),
) !void {
    const processed = try alloc.alloc(bool, gvecs.len);
    defer alloc.free(processed);

    @memset(processed, false);

    var i: usize = 0;
    while (i < gvecs.len) : (i += 1) {
        if (processed[i]) continue;

        const g = gvecs[i];
        // σ_y in Cartesian: (Gx, Gy, Gz) -> (Gx, -Gy, Gz)
        const gy_mirror = -g.cart.y;

        // Check if G is invariant (Gy ≈ 0)
        if (@abs(g.cart.y) < tol) {
            // Invariant under mirror -> belongs only to A'
            try invariant_list.append(alloc, i);
            processed[i] = true;
        } else {
            // Find the partner with (Gx, -Gy, Gz)
            var partner: ?usize = null;
            var j: usize = 0;
            while (j < gvecs.len) : (j += 1) {
                if (j == i or processed[j]) continue;
                const gj = gvecs[j];
                if (@abs(gj.cart.x - g.cart.x) < tol and
                    @abs(gj.cart.y - gy_mirror) < tol and
                    @abs(gj.cart.z - g.cart.z) < tol)
                {
                    partner = j;
                    break;
                }
            }

            if (partner) |p| {
                // Found a pair (G, σG)
                try pair_list.append(alloc, .{
                    .g_index = i,
                    .partner_index = p,
                });
                processed[i] = true;
                processed[p] = true;
            } else {
                // Partner not in basis - treat as invariant
                try invariant_list.append(alloc, i);
                processed[i] = true;
            }
        }
    }
}

/// Analyze basis symmetry using Cartesian y-mirror for 2D systems on M-Γ path.
/// This is useful when the fractional-coordinate symmetry analysis fails to
/// detect the relevant mirror for band crossings.
///
/// For 2D graphene on M-Γ path (Cartesian k_y = 0), the relevant mirror is σ_y:
/// G_cart = (Gx, Gy, Gz) -> (Gx, -Gy, Gz)
pub fn analyze_cartesian_mirror_y(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    little_grp: LittleGroup,
    tol: f64,
) !BasisSymmetryInfo {
    // Lists to collect invariant G-vectors and pairs
    var invariant_list: std.ArrayList(usize) = .empty;
    errdefer invariant_list.deinit(alloc);

    var pair_list: std.ArrayList(GVectorPair) = .empty;
    errdefer pair_list.deinit(alloc);

    try classify_g_vectors_by_cartesian_y(alloc, gvecs, tol, &invariant_list, &pair_list);

    const invariant_indices = try invariant_list.toOwnedSlice(alloc);
    const pairs = try pair_list.toOwnedSlice(alloc);

    return try build_mirror_irrep_blocks(alloc, little_grp, invariant_indices, pairs);
}

/// Find the C3 rotation operation (det=1, trace=0) in the little group.
fn find_c3_operation(little_grp: LittleGroup) ?symmetry.SymOp {
    for (little_grp.ops) |op| {
        const det = op.rot.det();
        const trace = op.rot.trace();
        if (det == 1 and trace == 0) return op;
    }
    return null;
}

/// Classify G-vectors under a C3 rotation into invariants and orbit-3 triplets.
fn classify_g_vectors_by_c3(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    c3: symmetry.SymOp,
    invariant_list: *std.ArrayList(usize),
    triplet_list: *std.ArrayList(GVectorTriplet),
) !void {
    const processed = try alloc.alloc(bool, gvecs.len);
    defer alloc.free(processed);

    @memset(processed, false);

    var i: usize = 0;
    while (i < gvecs.len) : (i += 1) {
        if (processed[i]) continue;

        const g = gvecs[i];
        const g_int = [3]i32{ g.h, g.k, g.l };

        // Apply C3: G' = C3.k_rot * G
        const g2_int = apply_rotation(c3.k_rot, g_int);

        // Check if G is invariant (C3*G = G)
        if (g2_int[0] == g_int[0] and g2_int[1] == g_int[1] and g2_int[2] == g_int[2]) {
            // Invariant under C3 -> belongs to A1
            try invariant_list.append(alloc, i);
            processed[i] = true;
        } else {
            // Apply C3 again: G'' = C3^2*G
            const g3_int = apply_rotation(c3.k_rot, g2_int);

            // Find G' and G'' in the basis
            var g2_idx: ?usize = null;
            var g3_idx: ?usize = null;

            var j: usize = 0;
            while (j < gvecs.len) : (j += 1) {
                if (processed[j]) continue;
                const gj = gvecs[j];
                if (gj.h == g2_int[0] and gj.k == g2_int[1] and gj.l == g2_int[2]) {
                    g2_idx = j;
                }
                if (gj.h == g3_int[0] and gj.k == g3_int[1] and gj.l == g3_int[2]) {
                    g3_idx = j;
                }
            }

            if (g2_idx != null and g3_idx != null) {
                // Found a complete triplet (G, C3*G, C3^2*G)
                try triplet_list.append(alloc, .{
                    .g1_index = i,
                    .g2_index = g2_idx.?,
                    .g3_index = g3_idx.?,
                });
                processed[i] = true;
                processed[g2_idx.?] = true;
                processed[g3_idx.?] = true;
            } else {
                // Incomplete orbit - treat as invariant
                try invariant_list.append(alloc, i);
                processed[i] = true;
            }
        }
    }
}

/// Build the A1 and E irrep blocks from invariant G-vectors and C3 triplets.
fn build_c3v_irrep_blocks(
    alloc: std.mem.Allocator,
    little_grp: LittleGroup,
    invariant_indices: []usize,
    triplets: []GVectorTriplet,
) !BasisSymmetryInfo {
    var blocks: std.ArrayList(IrrepBlock) = .empty;
    errdefer {
        for (blocks.items) |*b| b.deinit(alloc);
        blocks.deinit(alloc);
    }

    // A1 block: invariant G-vectors + symmetric combination from triplets
    // |A1_i⟩ = (|G1_i⟩ + |G2_i⟩ + |G3_i⟩)/√3
    try blocks.append(alloc, .{
        .irrep = .a1,
        .invariant_indices = invariant_indices,
        .pairs = try alloc.alloc(GVectorPair, 0),
        .triplets = triplets,
    });

    // E block: 2D representation from triplets only
    // Each triplet contributes 2 states (degenerate pair)
    // |E_a⟩ = (2|G1⟩ - |G2⟩ - |G3⟩)/√6
    // |E_b⟩ = (|G2⟩ - |G3⟩)/√2
    if (triplets.len > 0) {
        const triplets_copy = try alloc.alloc(GVectorTriplet, triplets.len);
        @memcpy(triplets_copy, triplets);
        try blocks.append(alloc, .{
            .irrep = .e,
            .invariant_indices = try alloc.alloc(usize, 0),
            .pairs = try alloc.alloc(GVectorPair, 0),
            .triplets = triplets_copy,
        });
    }

    return BasisSymmetryInfo{
        .little_group = little_grp,
        .irrep_blocks = try blocks.toOwnedSlice(alloc),
    };
}

/// Analyze basis symmetry for a C3v little group (3-fold rotation + 3 mirrors).
/// This is the relevant case for the K point in graphene.
///
/// For C3v symmetry:
/// - Invariant G-vectors (C3*G = G): belong to A1
/// - G-vector triplets (orbit size 3): contribute 1 state to A1 and 2 states to E
///
/// The E representation is 2-dimensional, causing degeneracy at K point (Dirac cone).
pub fn analyze_c3v(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    little_grp: LittleGroup,
) !BasisSymmetryInfo {
    const c3_op = find_c3_operation(little_grp);
    if (c3_op == null) {
        // No C3 found - return trivial classification
        return trivial_classification(alloc, gvecs, little_grp);
    }
    const c3 = c3_op.?;

    // Lists to collect invariant G-vectors and triplets
    var invariant_list: std.ArrayList(usize) = .empty;
    errdefer invariant_list.deinit(alloc);

    var triplet_list: std.ArrayList(GVectorTriplet) = .empty;
    errdefer triplet_list.deinit(alloc);

    try classify_g_vectors_by_c3(alloc, gvecs, c3, &invariant_list, &triplet_list);

    const invariant_indices = try invariant_list.toOwnedSlice(alloc);
    const triplets = try triplet_list.toOwnedSlice(alloc);

    return try build_c3v_irrep_blocks(alloc, little_grp, invariant_indices, triplets);
}

/// Apply rotation matrix to integer G-vector.
fn apply_rotation(rot: symmetry.Mat3i, g: [3]i32) [3]i32 {
    return [3]i32{
        rot.m[0][0] * g[0] + rot.m[0][1] * g[1] + rot.m[0][2] * g[2],
        rot.m[1][0] * g[0] + rot.m[1][1] * g[1] + rot.m[1][2] * g[2],
        rot.m[2][0] * g[0] + rot.m[2][1] * g[1] + rot.m[2][2] * g[2],
    };
}

/// Analyze basis functions using the little group symmetry.
pub fn analyze_basis_symmetry(
    alloc: std.mem.Allocator,
    gvecs: []const GVector,
    k_frac: math.Vec3,
    k_cart: math.Vec3,
    space_group_ops: []const symmetry.SymOp,
    tol: f64,
) !BasisSymmetryInfo {
    // Compute little group
    var little_grp = try compute_little_group(alloc, k_frac, space_group_ops, tol);
    errdefer little_grp.deinit(alloc);

    // Check if k is on a Cartesian mirror plane (k_y ≈ 0)
    // This is the case for M-Γ path in graphene
    const on_y_mirror_plane = @abs(k_cart.y) < tol;

    if (on_y_mirror_plane and gvecs.len > 0) {
        // Use Cartesian σ_y mirror for M-Γ path analysis
        return try analyze_cartesian_mirror_y(alloc, gvecs, little_grp, tol);
    }

    // Dispatch based on point group type
    switch (little_grp.point_group_type) {
        .cs => {
            // Cs symmetry: single mirror
            return try analyze_cs(alloc, gvecs, little_grp);
        },
        .c3v => {
            // C3v symmetry: 3-fold rotation + 3 mirrors (K point in graphene)
            return try analyze_c3v(alloc, gvecs, little_grp);
        },
        .c1 => {
            // No symmetry - trivial classification
            return try trivial_classification(alloc, gvecs, little_grp);
        },
        else => {
            // For other point groups, use trivial classification for now
            // TODO: implement C2v, C6v, D6h analysis
            return try trivial_classification(alloc, gvecs, little_grp);
        },
    }
}

const GVector = @import("../plane_wave/basis.zig").GVector;

test "little group at Gamma" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // Simple cubic with identity only
    const ops = [_]symmetry.SymOp{
        .{
            .rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .k_rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .trans = math.Vec3{ .x = 0, .y = 0, .z = 0 },
        },
    };

    const k_gamma = math.Vec3{ .x = 0, .y = 0, .z = 0 };
    var lg = try compute_little_group(alloc, k_gamma, &ops, 1e-6);
    defer lg.deinit(alloc);

    try testing.expectEqual(@as(usize, 1), lg.order());
    try testing.expectEqual(point_group.PointGroupType.c1, lg.point_group_type);
}

test "little group with mirror" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // Identity + σ_y mirror
    const ops = [_]symmetry.SymOp{
        .{
            .rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .k_rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .trans = math.Vec3{ .x = 0, .y = 0, .z = 0 },
        },
        .{
            .rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, -1, 0 }, .{ 0, 0, 1 } } },
            .k_rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, -1, 0 }, .{ 0, 0, 1 } } },
            .trans = math.Vec3{ .x = 0, .y = 0, .z = 0 },
        },
    };

    // k on mirror plane (ky = 0)
    const k_on_mirror = math.Vec3{ .x = 0.25, .y = 0, .z = 0 };
    var lg = try compute_little_group(alloc, k_on_mirror, &ops, 1e-6);
    defer lg.deinit(alloc);

    try testing.expectEqual(@as(usize, 2), lg.order());
    try testing.expectEqual(point_group.PointGroupType.cs, lg.point_group_type);
    try testing.expect(lg.has_mirror());

    // k off mirror plane
    const k_off_mirror = math.Vec3{ .x = 0.25, .y = 0.1, .z = 0 };
    var lg2 = try compute_little_group(alloc, k_off_mirror, &ops, 1e-6);
    defer lg2.deinit(alloc);

    try testing.expectEqual(@as(usize, 1), lg2.order());
    try testing.expect(!lg2.has_mirror());
}
