const std = @import("std");
const math = @import("../math/math.zig");
const symmetry = @import("symmetry.zig");

/// Point group types commonly encountered in crystals.
/// Named using Schoenflies notation.
pub const PointGroupType = enum {
    c1, // Identity only (order 1)
    ci, // Inversion (order 2)
    cs, // Single mirror plane (order 2)
    c2, // 2-fold rotation (order 2)
    c3, // 3-fold rotation (order 3)
    c4, // 4-fold rotation (order 4)
    c6, // 6-fold rotation (order 6)
    c2v, // C2 + 2 mirrors (order 4)
    c3v, // C3 + 3 mirrors (order 6)
    c4v, // C4 + 4 mirrors (order 8)
    c6v, // C6 + 6 mirrors (order 12)
    c2h, // C2 + horizontal mirror (order 4)
    c3h, // C3 + horizontal mirror (order 6)
    c4h, // C4 + horizontal mirror (order 8)
    c6h, // C6 + horizontal mirror (order 12)
    d2, // 3 perpendicular C2 (order 4)
    d3, // C3 + 3 C2 (order 6)
    d4, // C4 + 4 C2 (order 8)
    d6, // C6 + 6 C2 (order 12)
    d2h, // D2 + inversion (order 8)
    d3h, // D3 + horizontal mirror (order 12)
    d4h, // D4 + horizontal mirror (order 16)
    d6h, // D6 + horizontal mirror (order 24) - graphene
    d2d, // D2 + diagonal mirrors (order 8)
    d3d, // D3 + diagonal mirrors (order 12)
    s4, // S4 axis (order 4)
    s6, // S6 axis (order 6)
    t, // Tetrahedral (order 12)
    th, // T + inversion (order 24)
    td, // T + diagonal mirrors (order 24)
    o, // Octahedral (order 24)
    oh, // O + inversion (order 48)
    unknown,
};

/// Irreducible representation labels.
/// Uses Mulliken symbols (A, B, E, T for 1D, 2D, 3D representations).
pub const IrrepLabel = enum {
    // 1D representations
    a, // Symmetric under principal rotation
    a1, // A with additional symmetry
    a2, // A with different symmetry
    a1g, // A1 gerade (even under inversion)
    a1u, // A1 ungerade (odd under inversion)
    a2g, // A2 gerade
    a2u, // A2 ungerade
    b, // Antisymmetric under C2 perpendicular to principal axis
    b1, // B with additional symmetry
    b2, // B with different symmetry
    b1g, // B1 gerade
    b1u, // B1 ungerade
    b2g, // B2 gerade
    b2u, // B2 ungerade
    b3g, // B3 gerade (D2h)
    b3u, // B3 ungerade (D2h)
    a_prime, // A' (Cs: symmetric under mirror)
    a_double_prime, // A'' (Cs: antisymmetric under mirror)

    // 2D representations
    e, // Doubly degenerate
    e1, // E with subscript 1
    e2, // E with subscript 2
    eg, // E gerade
    eu, // E ungerade
    e1g, // E1 gerade
    e1u, // E1 ungerade
    e2g, // E2 gerade
    e2u, // E2 ungerade

    // 3D representations
    t, // Triply degenerate
    t1, // T with subscript 1
    t2, // T with subscript 2
    t1g, // T1 gerade
    t1u, // T1 ungerade
    t2g, // T2 gerade
    t2u, // T2 ungerade
};

/// Information about an irreducible representation.
pub const IrrepInfo = struct {
    label: IrrepLabel,
    dimension: u8, // 1, 2, or 3
    characters: []const f64, // Character for each class (for 1D irreps)
};

/// Character table for a point group.
pub const CharacterTable = struct {
    point_group: PointGroupType,
    order: usize, // Number of operations
    num_classes: usize, // Number of conjugacy classes
    irreps: []const IrrepInfo,

    /// Get character of an operation for a given irrep.
    /// For 1D irreps, this returns the character directly.
    /// For higher-dimensional irreps, returns the trace of the representation matrix.
    pub fn get_character(self: CharacterTable, irrep_idx: usize, class_idx: usize) f64 {
        if (irrep_idx >= self.irreps.len) return 0.0;
        const irrep = self.irreps[irrep_idx];
        if (class_idx >= irrep.characters.len) return 0.0;
        return irrep.characters[class_idx];
    }
};

// Character tables for common point groups.
// Classes are ordered: E, C_n, C_n^2, ..., σ, i, ...

/// Cs point group (mirror symmetry only)
/// Classes: E, σ
pub const cs_table = CharacterTable{
    .point_group = .cs,
    .order = 2,
    .num_classes = 2,
    .irreps = &[_]IrrepInfo{
        .{ .label = .a_prime, .dimension = 1, .characters = &[_]f64{ 1.0, 1.0 } },
        .{ .label = .a_double_prime, .dimension = 1, .characters = &[_]f64{ 1.0, -1.0 } },
    },
};

/// C2 point group (2-fold rotation)
/// Classes: E, C2
pub const c2_table = CharacterTable{
    .point_group = .c2,
    .order = 2,
    .num_classes = 2,
    .irreps = &[_]IrrepInfo{
        .{ .label = .a, .dimension = 1, .characters = &[_]f64{ 1.0, 1.0 } },
        .{ .label = .b, .dimension = 1, .characters = &[_]f64{ 1.0, -1.0 } },
    },
};

/// C2v point group
/// Classes: E, C2, σv, σv'
pub const c2v_table = CharacterTable{
    .point_group = .c2v,
    .order = 4,
    .num_classes = 4,
    .irreps = &[_]IrrepInfo{
        .{ .label = .a1, .dimension = 1, .characters = &[_]f64{ 1.0, 1.0, 1.0, 1.0 } },
        .{ .label = .a2, .dimension = 1, .characters = &[_]f64{ 1.0, 1.0, -1.0, -1.0 } },
        .{ .label = .b1, .dimension = 1, .characters = &[_]f64{ 1.0, -1.0, 1.0, -1.0 } },
        .{ .label = .b2, .dimension = 1, .characters = &[_]f64{ 1.0, -1.0, -1.0, 1.0 } },
    },
};

/// C3v point group
/// Classes: E, 2C3, 3σv
pub const c3v_table = CharacterTable{
    .point_group = .c3v,
    .order = 6,
    .num_classes = 3,
    .irreps = &[_]IrrepInfo{
        .{ .label = .a1, .dimension = 1, .characters = &[_]f64{ 1.0, 1.0, 1.0 } },
        .{ .label = .a2, .dimension = 1, .characters = &[_]f64{ 1.0, 1.0, -1.0 } },
        .{ .label = .e, .dimension = 2, .characters = &[_]f64{ 2.0, -1.0, 0.0 } },
    },
};

/// C6v point group
/// Classes: E, 2C6, 2C3, C2, 3σv, 3σd
pub const c6v_table = CharacterTable{
    .point_group = .c6v,
    .order = 12,
    .num_classes = 6,
    .irreps = &[_]IrrepInfo{
        .{ .label = .a1, .dimension = 1, .characters = &[_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 } },
        .{ .label = .a2, .dimension = 1, .characters = &[_]f64{ 1.0, 1.0, 1.0, 1.0, -1.0, -1.0 } },
        .{ .label = .b1, .dimension = 1, .characters = &[_]f64{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 } },
        .{ .label = .b2, .dimension = 1, .characters = &[_]f64{ 1.0, -1.0, 1.0, -1.0, -1.0, 1.0 } },
        .{ .label = .e1, .dimension = 2, .characters = &[_]f64{ 2.0, 1.0, -1.0, -2.0, 0.0, 0.0 } },
        .{ .label = .e2, .dimension = 2, .characters = &[_]f64{ 2.0, -1.0, -1.0, 2.0, 0.0, 0.0 } },
    },
};

/// D6h point group (graphene)
/// Classes: E, 2C6, 2C3, C2, 3C2', 3C2'', i, 2S3, 2S6, σh, 3σd, 3σv
pub const d6h_table = CharacterTable{
    .point_group = .d6h,
    .order = 24,
    .num_classes = 12,
    .irreps = &[_]IrrepInfo{
        .{
            .label = .a1g,
            .dimension = 1,
            .characters = &[_]f64{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        },
        .{
            .label = .a2g,
            .dimension = 1,
            .characters = &[_]f64{ 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1 },
        },
        .{
            .label = .b1g,
            .dimension = 1,
            .characters = &[_]f64{ 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 },
        },
        .{
            .label = .b2g,
            .dimension = 1,
            .characters = &[_]f64{ 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1 },
        },
        .{
            .label = .e1g,
            .dimension = 2,
            .characters = &[_]f64{ 2, 1, -1, -2, 0, 0, 2, 1, -1, -2, 0, 0 },
        },
        .{
            .label = .e2g,
            .dimension = 2,
            .characters = &[_]f64{ 2, -1, -1, 2, 0, 0, 2, -1, -1, 2, 0, 0 },
        },
        .{
            .label = .a1u,
            .dimension = 1,
            .characters = &[_]f64{ 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1 },
        },
        .{
            .label = .a2u,
            .dimension = 1,
            .characters = &[_]f64{ 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1 },
        },
        .{
            .label = .b1u,
            .dimension = 1,
            .characters = &[_]f64{ 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1 },
        },
        .{
            .label = .b2u,
            .dimension = 1,
            .characters = &[_]f64{ 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1 },
        },
        .{
            .label = .e1u,
            .dimension = 2,
            .characters = &[_]f64{ 2, 1, -1, -2, 0, 0, -2, -1, 1, 2, 0, 0 },
        },
        .{
            .label = .e2u,
            .dimension = 2,
            .characters = &[_]f64{ 2, -1, -1, 2, 0, 0, -2, 1, 1, -2, 0, 0 },
        },
    },
};

/// Get character table for a point group type.
pub fn get_character_table(pg: PointGroupType) ?CharacterTable {
    return switch (pg) {
        .cs => cs_table,
        .c2 => c2_table,
        .c2v => c2v_table,
        .c3v => c3v_table,
        .c6v => c6v_table,
        .d6h => d6h_table,
        else => null,
    };
}

/// Determine point group type from symmetry operations.
/// This examines the rotation parts of the operations.
pub fn identify_point_group(ops: []const symmetry.SymOp) PointGroupType {
    const n = ops.len;

    // Count operation types
    var has_inversion = false;
    var max_rotation_order: i32 = 1;
    var num_c2_axes: usize = 0;
    var num_mirrors: usize = 0;

    for (ops) |op| {
        const det = op.rot.det();
        const trace = op.rot.trace();

        if (det == 1) {
            // Proper rotation
            const order = rotation_order(trace);
            if (order > max_rotation_order) {
                max_rotation_order = order;
            }
            if (order == 2) {
                num_c2_axes += 1;
            }
        } else {
            // Improper rotation
            if (trace == -3) {
                has_inversion = true;
            } else {
                num_mirrors += 1;
            }
        }
    }

    // Identify based on counts
    if (n == 1) return .c1;
    if (n == 2) {
        if (has_inversion) return .ci;
        if (num_mirrors == 1) return .cs;
        if (num_c2_axes == 1) return .c2;
    }
    if (n == 24 and max_rotation_order == 6 and has_inversion) return .d6h;
    if (n == 12 and max_rotation_order == 6) return .c6v;
    if (n == 6 and max_rotation_order == 3) return .c3v;
    if (n == 4 and max_rotation_order == 2 and num_mirrors == 2) return .c2v;
    if (n == 4 and max_rotation_order == 2 and num_c2_axes == 3) return .d2;

    return .unknown;
}

/// Get rotation order from trace of rotation matrix.
fn rotation_order(trace: i32) i32 {
    return switch (trace) {
        3 => 1, // Identity
        2 => 6, // C6
        1 => 4, // C4
        0 => 3, // C3
        -1 => 2, // C2
        else => 1,
    };
}

/// Classify symmetry operation into conjugacy class.
/// Returns class index for the character table lookup.
pub fn classify_operation(op: symmetry.SymOp, pg: PointGroupType) usize {
    const det = op.rot.det();
    const trace = op.rot.trace();

    // For Cs: class 0 = E, class 1 = σ
    if (pg == .cs) {
        if (det == 1 and trace == 3) return 0; // Identity
        return 1; // Mirror
    }

    // For C2: class 0 = E, class 1 = C2
    if (pg == .c2) {
        if (det == 1 and trace == 3) return 0; // Identity
        return 1; // C2
    }

    // For C2v: class 0 = E, 1 = C2, 2 = σv, 3 = σv'
    if (pg == .c2v) {
        if (det == 1 and trace == 3) return 0; // Identity
        if (det == 1 and trace == -1) return 1; // C2
        // Distinguish mirrors by axis
        return 2; // Simplified - would need axis info for proper distinction
    }

    // Default: identity class
    if (det == 1 and trace == 3) return 0;
    return 0;
}

test "point group identification" {
    const testing = std.testing;

    // Identity only -> C1
    const c1_ops = [_]symmetry.SymOp{
        .{
            .rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .k_rot = symmetry.Mat3i{ .m = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
            .trans = math.Vec3{ .x = 0, .y = 0, .z = 0 },
        },
    };
    try testing.expectEqual(PointGroupType.c1, identify_point_group(&c1_ops));

    // Identity + mirror -> Cs
    const cs_ops = [_]symmetry.SymOp{
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
    try testing.expectEqual(PointGroupType.cs, identify_point_group(&cs_ops));
}
