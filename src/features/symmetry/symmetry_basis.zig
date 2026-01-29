const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("symmetry.zig");
const little_group = @import("little_group.zig");
const point_group = @import("point_group.zig");
const linalg = @import("../linalg/linalg.zig");
const plane_wave = @import("../plane_wave/basis.zig");

/// Result of symmetry block-diagonalized eigenvalue computation.
pub const BlockDiagResult = struct {
    /// All eigenvalues sorted in ascending order.
    eigenvalues: []f64,
    /// Number of eigenvalues from each irrep block.
    irrep_counts: []IrrepCount,

    pub const IrrepCount = struct {
        irrep: point_group.IrrepLabel,
        count: usize,
    };

    pub fn deinit(self: *BlockDiagResult, alloc: std.mem.Allocator) void {
        if (self.eigenvalues.len > 0) alloc.free(self.eigenvalues);
        if (self.irrep_counts.len > 0) alloc.free(self.irrep_counts);
    }
};

/// Compute eigenvalues using symmetry block-diagonalization.
///
/// This function:
/// 1. Analyzes the symmetry of the plane-wave basis at k-point
/// 2. Partitions basis functions by irreducible representation
/// 3. Builds sub-Hamiltonians for each irrep block
/// 4. Diagonalizes each block separately
/// 5. Merges and sorts the eigenvalues
///
/// This approach properly handles band crossings at high-symmetry points
/// by preventing spurious mixing between states of different symmetry.
pub fn symmetryBlockDiagEigenvalues(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    gvecs: []const plane_wave.GVector,
    full_h: []math.Complex,
    n: usize,
    k_frac: math.Vec3,
    k_cart: math.Vec3,
    space_group_ops: []const symmetry.SymOp,
    tol: f64,
) !BlockDiagResult {
    // Analyze basis symmetry
    var sym_info = try little_group.analyzeBasisSymmetry(alloc, gvecs, k_frac, k_cart, space_group_ops, tol);
    defer sym_info.deinit(alloc);

    // If only one block with no pairs, just solve the full matrix
    if (sym_info.irrep_blocks.len <= 1 and sym_info.irrep_blocks[0].pairs.len == 0) {
        const values = try linalg.hermitianEigenvalues(alloc, backend, n, full_h);
        const counts = try alloc.alloc(BlockDiagResult.IrrepCount, 1);
        counts[0] = .{ .irrep = .a, .count = n };
        return BlockDiagResult{
            .eigenvalues = values,
            .irrep_counts = counts,
        };
    }

    // Process each irrep block
    var all_values: std.ArrayList(f64) = .empty;
    errdefer all_values.deinit(alloc);

    var counts: std.ArrayList(BlockDiagResult.IrrepCount) = .empty;
    errdefer counts.deinit(alloc);

    for (sym_info.irrep_blocks) |block| {
        const block_size = block.size();
        if (block_size == 0) continue;

        // Build sub-Hamiltonian for this irrep block
        const h_block = try buildBlockHamiltonian(alloc, full_h, n, block);
        defer alloc.free(h_block);

        // Diagonalize
        const block_values = try linalg.hermitianEigenvalues(alloc, backend, block_size, h_block);
        defer alloc.free(block_values);

        // Add to results
        for (block_values) |val| {
            try all_values.append(alloc, val);
        }

        try counts.append(alloc, .{
            .irrep = block.irrep,
            .count = block_size,
        });
    }

    // Sort all eigenvalues
    const merged = try all_values.toOwnedSlice(alloc);
    std.sort.block(f64, merged, {}, std.sort.asc(f64));

    return BlockDiagResult{
        .eigenvalues = merged,
        .irrep_counts = try counts.toOwnedSlice(alloc),
    };
}

/// Build the Hamiltonian matrix for a specific irrep block in the symmetry-adapted basis.
///
/// For Cs symmetry (A' block):
/// - Basis ordering: [invariant_0, invariant_1, ..., |+_0⟩, |+_1⟩, ...]
/// - |+_i⟩ = (|G_i⟩ + |σG_i⟩)/√2
///
/// For Cs symmetry (A'' block):
/// - Basis ordering: [|-_0⟩, |-_1⟩, ...]
/// - |-_i⟩ = (|G_i⟩ - |σG_i⟩)/√2
///
/// For C3v symmetry (A1 block):
/// - Basis ordering: [invariant_0, ..., |A1_0⟩, |A1_1⟩, ...]
/// - |A1_i⟩ = (|G1_i⟩ + |G2_i⟩ + |G3_i⟩)/√3
///
/// For C3v symmetry (E block):
/// - Basis ordering: [|Ea_0⟩, |Eb_0⟩, |Ea_1⟩, |Eb_1⟩, ...]
/// - |Ea_i⟩ = (2|G1_i⟩ - |G2_i⟩ - |G3_i⟩)/√6
/// - |Eb_i⟩ = (|G2_i⟩ - |G3_i⟩)/√2
fn buildBlockHamiltonian(
    alloc: std.mem.Allocator,
    full_h: []const math.Complex,
    n: usize,
    block: little_group.IrrepBlock,
) ![]math.Complex {
    // Check if this is a C3v block with triplets
    if (block.triplets.len > 0) {
        return buildC3vBlockHamiltonian(alloc, full_h, n, block);
    }

    // Original Cs/trivial implementation
    const num_inv = block.invariant_indices.len;
    const num_pairs = block.pairs.len;
    const block_size = num_inv + num_pairs;

    const h_block = try alloc.alloc(math.Complex, block_size * block_size);
    errdefer alloc.free(h_block);

    const is_even = (block.irrep == .a_prime or block.irrep == .a);
    const sqrt2 = @sqrt(2.0);
    const inv_sqrt2 = 1.0 / sqrt2;

    // Fill the block Hamiltonian
    // Row/column indices: 0..num_inv-1 are invariant, num_inv..block_size-1 are paired states

    var row: usize = 0;
    while (row < block_size) : (row += 1) {
        var col: usize = 0;
        while (col < block_size) : (col += 1) {
            const val = computeBlockElement(full_h, n, block, row, col, num_inv, is_even, inv_sqrt2);
            h_block[row + col * block_size] = val;
        }
    }

    return h_block;
}

/// Build Hamiltonian block for C3v symmetry with triplets.
fn buildC3vBlockHamiltonian(
    alloc: std.mem.Allocator,
    full_h: []const math.Complex,
    n: usize,
    block: little_group.IrrepBlock,
) ![]math.Complex {
    const num_inv = block.invariant_indices.len;
    const num_triplets = block.triplets.len;
    const is_E_block = (block.irrep == .e or block.irrep == .e1 or block.irrep == .e2);

    // E block: 2 states per triplet
    // A1 block: invariants + 1 state per triplet
    const block_size = if (is_E_block) num_triplets * 2 else num_inv + num_triplets;

    const h_block = try alloc.alloc(math.Complex, block_size * block_size);
    errdefer alloc.free(h_block);

    // Normalization constants
    const inv_sqrt3 = 1.0 / @sqrt(3.0);
    const inv_sqrt6 = 1.0 / @sqrt(6.0);
    const inv_sqrt2 = 1.0 / @sqrt(2.0);

    if (is_E_block) {
        // E block: |Ea⟩ = (2|G1⟩ - |G2⟩ - |G3⟩)/√6, |Eb⟩ = (|G2⟩ - |G3⟩)/√2
        // Basis ordering: |Ea_0⟩, |Eb_0⟩, |Ea_1⟩, |Eb_1⟩, ...
        var row: usize = 0;
        while (row < block_size) : (row += 1) {
            var col: usize = 0;
            while (col < block_size) : (col += 1) {
                const val = computeEBlockElement(full_h, n, block.triplets, row, col, inv_sqrt6, inv_sqrt2);
                h_block[row + col * block_size] = val;
            }
        }
    } else {
        // A1 block: invariants + |A1⟩ = (|G1⟩ + |G2⟩ + |G3⟩)/√3
        var row: usize = 0;
        while (row < block_size) : (row += 1) {
            var col: usize = 0;
            while (col < block_size) : (col += 1) {
                const val = computeA1BlockElement(full_h, n, block, row, col, num_inv, inv_sqrt3);
                h_block[row + col * block_size] = val;
            }
        }
    }

    return h_block;
}

/// Compute element for E block (2D irrep from C3v triplets).
fn computeEBlockElement(
    full_h: []const math.Complex,
    n: usize,
    triplets: []const little_group.GVectorTriplet,
    row: usize,
    col: usize,
    inv_sqrt6: f64,
    inv_sqrt2: f64,
) math.Complex {
    // Row/col index: 0,1 -> triplet 0; 2,3 -> triplet 1; etc.
    const row_triplet = row / 2;
    const row_type = row % 2; // 0 = Ea, 1 = Eb
    const col_triplet = col / 2;
    const col_type = col % 2;

    const tr = triplets[row_triplet];
    const tc = triplets[col_triplet];

    // H matrix elements between original G-vectors
    const h11 = full_h[tr.g1_index + tc.g1_index * n];
    const h12 = full_h[tr.g1_index + tc.g2_index * n];
    const h13 = full_h[tr.g1_index + tc.g3_index * n];
    const h21 = full_h[tr.g2_index + tc.g1_index * n];
    const h22 = full_h[tr.g2_index + tc.g2_index * n];
    const h23 = full_h[tr.g2_index + tc.g3_index * n];
    const h31 = full_h[tr.g3_index + tc.g1_index * n];
    const h32 = full_h[tr.g3_index + tc.g2_index * n];
    const h33 = full_h[tr.g3_index + tc.g3_index * n];

    if (row_type == 0 and col_type == 0) {
        // ⟨Ea|H|Ea⟩ = (4h11 - 2h12 - 2h13 - 2h21 + h22 + h23 - 2h31 + h32 + h33) / 6
        var sum = math.complex.scale(h11, 4.0);
        sum = math.complex.add(sum, math.complex.scale(h12, -2.0));
        sum = math.complex.add(sum, math.complex.scale(h13, -2.0));
        sum = math.complex.add(sum, math.complex.scale(h21, -2.0));
        sum = math.complex.add(sum, h22);
        sum = math.complex.add(sum, h23);
        sum = math.complex.add(sum, math.complex.scale(h31, -2.0));
        sum = math.complex.add(sum, h32);
        sum = math.complex.add(sum, h33);
        return math.complex.scale(sum, 1.0 / 6.0);
    } else if (row_type == 0 and col_type == 1) {
        // ⟨Ea|H|Eb⟩ = (2h12 - 2h13 - h22 + h23 - h32 + h33) / (√6 * √2)
        var sum = math.complex.scale(h12, 2.0);
        sum = math.complex.add(sum, math.complex.scale(h13, -2.0));
        sum = math.complex.add(sum, math.complex.scale(h22, -1.0));
        sum = math.complex.add(sum, h23);
        sum = math.complex.add(sum, math.complex.scale(h32, -1.0));
        sum = math.complex.add(sum, h33);
        return math.complex.scale(sum, inv_sqrt6 * inv_sqrt2);
    } else if (row_type == 1 and col_type == 0) {
        // ⟨Eb|H|Ea⟩ = (2h21 - h22 - h23 - 2h31 + h32 + h33) / (√2 * √6)
        var sum = math.complex.scale(h21, 2.0);
        sum = math.complex.add(sum, math.complex.scale(h22, -1.0));
        sum = math.complex.add(sum, math.complex.scale(h23, -1.0));
        sum = math.complex.add(sum, math.complex.scale(h31, -2.0));
        sum = math.complex.add(sum, h32);
        sum = math.complex.add(sum, h33);
        return math.complex.scale(sum, inv_sqrt2 * inv_sqrt6);
    } else {
        // ⟨Eb|H|Eb⟩ = (h22 - h23 - h32 + h33) / 2
        var sum = h22;
        sum = math.complex.add(sum, math.complex.scale(h23, -1.0));
        sum = math.complex.add(sum, math.complex.scale(h32, -1.0));
        sum = math.complex.add(sum, h33);
        return math.complex.scale(sum, 0.5);
    }
}

/// Compute element for A1 block (invariants + symmetric triplet combination).
fn computeA1BlockElement(
    full_h: []const math.Complex,
    n: usize,
    block: little_group.IrrepBlock,
    row: usize,
    col: usize,
    num_inv: usize,
    inv_sqrt3: f64,
) math.Complex {
    const row_is_inv = row < num_inv;
    const col_is_inv = col < num_inv;

    if (row_is_inv and col_is_inv) {
        // Both invariant: H(I_a, I_b)
        const i_a = block.invariant_indices[row];
        const i_b = block.invariant_indices[col];
        return full_h[i_a + i_b * n];
    } else if (row_is_inv and !col_is_inv) {
        // Row invariant, column is A1 from triplet
        // ⟨I|H|A1⟩ = (H(I,G1) + H(I,G2) + H(I,G3)) / √3
        const i_a = block.invariant_indices[row];
        const triplet_idx = col - num_inv;
        const t = block.triplets[triplet_idx];

        const h1 = full_h[i_a + t.g1_index * n];
        const h2 = full_h[i_a + t.g2_index * n];
        const h3 = full_h[i_a + t.g3_index * n];

        return math.complex.scale(math.complex.add(math.complex.add(h1, h2), h3), inv_sqrt3);
    } else if (!row_is_inv and col_is_inv) {
        // Row is A1 from triplet, column invariant
        const triplet_idx = row - num_inv;
        const t = block.triplets[triplet_idx];
        const i_b = block.invariant_indices[col];

        const h1 = full_h[t.g1_index + i_b * n];
        const h2 = full_h[t.g2_index + i_b * n];
        const h3 = full_h[t.g3_index + i_b * n];

        return math.complex.scale(math.complex.add(math.complex.add(h1, h2), h3), inv_sqrt3);
    } else {
        // Both A1 from triplets
        // ⟨A1_a|H|A1_b⟩ = (sum of all 9 H elements) / 3
        const triplet_a = row - num_inv;
        const triplet_b = col - num_inv;
        const ta = block.triplets[triplet_a];
        const tb = block.triplets[triplet_b];

        var sum = math.complex.init(0, 0);
        const g_a = [_]usize{ ta.g1_index, ta.g2_index, ta.g3_index };
        const g_b = [_]usize{ tb.g1_index, tb.g2_index, tb.g3_index };

        for (g_a) |ga| {
            for (g_b) |gb| {
                sum = math.complex.add(sum, full_h[ga + gb * n]);
            }
        }

        return math.complex.scale(sum, 1.0 / 3.0);
    }
}

/// Compute a single element of the block Hamiltonian.
fn computeBlockElement(
    full_h: []const math.Complex,
    n: usize,
    block: little_group.IrrepBlock,
    row: usize,
    col: usize,
    num_inv: usize,
    is_even: bool,
    inv_sqrt2: f64,
) math.Complex {
    const row_is_inv = row < num_inv;
    const col_is_inv = col < num_inv;

    if (row_is_inv and col_is_inv) {
        // Both are invariant: H'(I_a, I_b) = H(I_a, I_b)
        const i_a = block.invariant_indices[row];
        const i_b = block.invariant_indices[col];
        return full_h[i_a + i_b * n];
    } else if (row_is_inv and !col_is_inv) {
        // Row is invariant, column is paired
        // H'(I_a, ±_b) = (H(I_a, G_b) ± H(I_a, σG_b)) / √2
        const i_a = block.invariant_indices[row];
        const pair_idx = col - num_inv;
        const g_b = block.pairs[pair_idx].g_index;
        const sg_b = block.pairs[pair_idx].partner_index;

        const h_ag = full_h[i_a + g_b * n];
        const h_asg = full_h[i_a + sg_b * n];

        if (is_even) {
            // (H(I_a, G_b) + H(I_a, σG_b)) / √2
            return math.complex.scale(math.complex.add(h_ag, h_asg), inv_sqrt2);
        } else {
            // (H(I_a, G_b) - H(I_a, σG_b)) / √2
            return math.complex.scale(math.complex.sub(h_ag, h_asg), inv_sqrt2);
        }
    } else if (!row_is_inv and col_is_inv) {
        // Row is paired, column is invariant
        // H'(±_a, I_b) = (H(G_a, I_b) ± H(σG_a, I_b)) / √2
        const pair_idx = row - num_inv;
        const g_a = block.pairs[pair_idx].g_index;
        const sg_a = block.pairs[pair_idx].partner_index;
        const i_b = block.invariant_indices[col];

        const h_gi = full_h[g_a + i_b * n];
        const h_sgi = full_h[sg_a + i_b * n];

        if (is_even) {
            return math.complex.scale(math.complex.add(h_gi, h_sgi), inv_sqrt2);
        } else {
            return math.complex.scale(math.complex.sub(h_gi, h_sgi), inv_sqrt2);
        }
    } else {
        // Both are paired
        // H'(±_a, ±_b) = (H(G_a,G_b) ± H(G_a,σG_b) ± H(σG_a,G_b) + H(σG_a,σG_b)) / 2
        const pair_a = row - num_inv;
        const pair_b = col - num_inv;
        const g_a = block.pairs[pair_a].g_index;
        const sg_a = block.pairs[pair_a].partner_index;
        const g_b = block.pairs[pair_b].g_index;
        const sg_b = block.pairs[pair_b].partner_index;

        const h_gg = full_h[g_a + g_b * n]; // H(G_a, G_b)
        const h_gsg = full_h[g_a + sg_b * n]; // H(G_a, σG_b)
        const h_sgg = full_h[sg_a + g_b * n]; // H(σG_a, G_b)
        const h_sgsg = full_h[sg_a + sg_b * n]; // H(σG_a, σG_b)

        if (is_even) {
            // ⟨+_a|H|+_b⟩ = (H_gg + H_gsg + H_sgg + H_sgsg) / 2
            const sum1 = math.complex.add(h_gg, h_sgsg);
            const sum2 = math.complex.add(h_gsg, h_sgg);
            return math.complex.scale(math.complex.add(sum1, sum2), 0.5);
        } else {
            // ⟨-_a|H|-_b⟩ = (H_gg - H_gsg - H_sgg + H_sgsg) / 2
            const sum1 = math.complex.add(h_gg, h_sgsg);
            const sum2 = math.complex.add(h_gsg, h_sgg);
            return math.complex.scale(math.complex.sub(sum1, sum2), 0.5);
        }
    }
}

/// Convenience function for band calculations.
/// Returns just the eigenvalues sorted in ascending order.
pub fn computeBandEigenvalues(
    alloc: std.mem.Allocator,
    backend: linalg.Backend,
    gvecs: []const plane_wave.GVector,
    full_h: []math.Complex,
    k_frac: math.Vec3,
    k_cart: math.Vec3,
    space_group_ops: []const symmetry.SymOp,
) ![]f64 {
    const n = gvecs.len;
    if (n == 0) return error.NoPlaneWaves;

    const result = try symmetryBlockDiagEigenvalues(
        alloc,
        backend,
        gvecs,
        full_h,
        n,
        k_frac,
        k_cart,
        space_group_ops,
        1e-6,
    );

    // Free the irrep counts but keep eigenvalues
    if (result.irrep_counts.len > 0) {
        alloc.free(result.irrep_counts);
    }

    return result.eigenvalues;
}

test "symmetry block diag - trivial case" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // Create a simple 2x2 Hamiltonian
    var h = [_]math.Complex{
        math.complex.init(1.0, 0.0), math.complex.init(0.1, 0.0),
        math.complex.init(0.1, 0.0), math.complex.init(2.0, 0.0),
    };

    const gvecs = [_]plane_wave.GVector{
        .{ .h = 0, .k = 0, .l = 0, .cart = .{ .x = 0, .y = 0, .z = 0 }, .kpg = .{ .x = 0, .y = 0, .z = 0 }, .kinetic = 0 },
        .{ .h = 1, .k = 0, .l = 0, .cart = .{ .x = 1, .y = 0, .z = 0 }, .kpg = .{ .x = 1, .y = 0, .z = 0 }, .kinetic = 1 },
    };

    // No symmetry operations except identity
    const ops = [_]symmetry.SymOp{
        .{
            .rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .k_rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .trans = math.Vec3{ .x = 0, .y = 0, .z = 0 },
        },
    };

    const k = math.Vec3{ .x = 0, .y = 0, .z = 0 };

    var result = try symmetryBlockDiagEigenvalues(
        alloc,
        .zig,
        &gvecs,
        &h,
        2,
        k,
        &ops,
        1e-6,
    );
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 2), result.eigenvalues.len);
    // Eigenvalues should be approximately 0.99 and 2.01
    try testing.expect(result.eigenvalues[0] < result.eigenvalues[1]);
}

test "Cs symmetry block diag - paired G-vectors" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // Create a 4x4 Hamiltonian with Cs symmetry
    // G-vectors: (0,0,0), (1,0,0), (-1,0,0), (0,1,0)
    // Mirror σ_y: (h,k,l) -> (h,-k,l)
    // (0,0,0) is invariant
    // (1,0,0) is invariant
    // (-1,0,0) is invariant
    // (0,1,0) and (0,-1,0) would be a pair, but (0,-1,0) is not in basis

    // Simpler test: all invariant G-vectors
    var h = [_]math.Complex{
        math.complex.init(1.0, 0.0), math.complex.init(0.1, 0.0), math.complex.init(0.1, 0.0), math.complex.init(0.1, 0.0),
        math.complex.init(0.1, 0.0), math.complex.init(2.0, 0.0), math.complex.init(0.1, 0.0), math.complex.init(0.1, 0.0),
        math.complex.init(0.1, 0.0), math.complex.init(0.1, 0.0), math.complex.init(3.0, 0.0), math.complex.init(0.1, 0.0),
        math.complex.init(0.1, 0.0), math.complex.init(0.1, 0.0), math.complex.init(0.1, 0.0), math.complex.init(4.0, 0.0),
    };

    const gvecs = [_]plane_wave.GVector{
        .{ .h = 0, .k = 0, .l = 0, .cart = .{ .x = 0, .y = 0, .z = 0 }, .kpg = .{ .x = 0, .y = 0, .z = 0 }, .kinetic = 0 },
        .{ .h = 1, .k = 0, .l = 0, .cart = .{ .x = 1, .y = 0, .z = 0 }, .kpg = .{ .x = 1, .y = 0, .z = 0 }, .kinetic = 1 },
        .{ .h = -1, .k = 0, .l = 0, .cart = .{ .x = -1, .y = 0, .z = 0 }, .kpg = .{ .x = -1, .y = 0, .z = 0 }, .kinetic = 1 },
        .{ .h = 0, .k = 1, .l = 0, .cart = .{ .x = 0, .y = 1, .z = 0 }, .kpg = .{ .x = 0, .y = 1, .z = 0 }, .kinetic = 1 },
    };

    // σ_y mirror: (h,k,l) -> (h,-k,l)
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

    const k = math.Vec3{ .x = 0.25, .y = 0, .z = 0 }; // On mirror plane

    var result = try symmetryBlockDiagEigenvalues(
        alloc,
        .zig,
        &gvecs,
        &h,
        4,
        k,
        &ops,
        1e-6,
    );
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 4), result.eigenvalues.len);
    // All eigenvalues should be present and sorted
    try testing.expect(result.eigenvalues[0] <= result.eigenvalues[1]);
    try testing.expect(result.eigenvalues[1] <= result.eigenvalues[2]);
    try testing.expect(result.eigenvalues[2] <= result.eigenvalues[3]);
}
