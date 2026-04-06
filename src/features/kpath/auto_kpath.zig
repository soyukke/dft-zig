const std = @import("std");
const math = @import("../math/math.zig");
const config = @import("../config/config.zig");

/// Bravais lattice types (Setyawan-Curtarolo convention).
pub const BravaisLattice = enum {
    cub, // Simple cubic
    fcc, // Face-centered cubic
    bcc, // Body-centered cubic
    hex, // Hexagonal
    tet, // Simple tetragonal
    // Future: orc, rhl, mcl, tri, etc.
};

/// A high-symmetry point with label and fractional coordinates.
pub const HighSymPoint = struct {
    label: []const u8,
    k: math.Vec3,
};

/// Result of auto k-path generation.
pub const AutoKPathResult = struct {
    points: []config.BandPathPoint,

    pub fn deinit(self: *AutoKPathResult, alloc: std.mem.Allocator) void {
        for (self.points) |p| {
            alloc.free(p.label);
        }
        alloc.free(self.points);
    }
};

/// Detect Bravais lattice type from cell vectors (in Bohr).
/// Uses metric tensor eigenvalues and angles.
pub fn detectBravaisLattice(cell: math.Mat3) !BravaisLattice {
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);

    const a = math.Vec3.norm(a1);
    const b = math.Vec3.norm(a2);
    const c = math.Vec3.norm(a3);

    const cos_alpha = math.Vec3.dot(a2, a3) / (b * c); // angle(a2, a3)
    const cos_beta = math.Vec3.dot(a1, a3) / (a * c); // angle(a1, a3)
    const cos_gamma = math.Vec3.dot(a1, a2) / (a * b); // angle(a1, a2)

    const tol = 1e-4;

    const alpha_90 = @abs(cos_alpha) < tol;
    const beta_90 = @abs(cos_beta) < tol;
    const gamma_90 = @abs(cos_gamma) < tol;
    const gamma_120 = @abs(cos_gamma + 0.5) < tol; // 120 degrees

    const ab_eq = @abs(a - b) / a < tol;
    const ac_eq = @abs(a - c) / a < tol;
    const all_eq = ab_eq and ac_eq;

    // Hexagonal: a == b != c, alpha=beta=90, gamma=120
    if (ab_eq and alpha_90 and beta_90 and gamma_120) {
        return .hex;
    }

    // Cubic family: a == b == c, all angles 90
    if (all_eq and alpha_90 and beta_90 and gamma_90) {
        return .cub;
    }

    // FCC conventional cell is cubic, but primitive cell has
    // a == b == c, angles all 60 degrees
    if (all_eq) {
        const cos_60 = 0.5;
        const all_60 = @abs(cos_alpha - cos_60) < tol and
            @abs(cos_beta - cos_60) < tol and
            @abs(cos_gamma - cos_60) < tol;
        if (all_60) return .fcc;

        // BCC primitive: a == b == c, angles ~109.47 (cos = -1/3)
        const cos_bcc = -1.0 / 3.0;
        const all_bcc = @abs(cos_alpha - cos_bcc) < tol and
            @abs(cos_beta - cos_bcc) < tol and
            @abs(cos_gamma - cos_bcc) < tol;
        if (all_bcc) return .bcc;
    }

    // Tetragonal: a == b != c, all angles 90
    if (ab_eq and !ac_eq and alpha_90 and beta_90 and gamma_90) {
        return .tet;
    }

    return error.UnsupportedLattice;
}

/// Standard high-symmetry k-path for a given Bravais lattice.
/// Returns path points in fractional coordinates (Setyawan-Curtarolo convention).
pub fn getStandardKPath(alloc: std.mem.Allocator, lattice: BravaisLattice) !AutoKPathResult {
    const default_path: []const u8 = switch (lattice) {
        .cub => "G-X-M-G-R",
        .fcc => "G-X-W-K-G-L",
        .bcc => "G-H-N-G-P",
        .hex => "G-M-K-G-A",
        .tet => "G-X-M-G-Z",
    };
    return parsePathString(alloc, default_path, lattice);
}

/// Parse a path string like "G-X-W-K-G-L" into BandPathPoint array.
/// Looks up labels in the standard table for the given lattice.
/// Supports "|" as segment separator (non-continuous path marker, ignored for now).
pub fn parsePathString(
    alloc: std.mem.Allocator,
    path_str: []const u8,
    lattice: BravaisLattice,
) !AutoKPathResult {
    const table: []const HighSymPoint = getHighSymTable(lattice);

    var points_list: std.ArrayList(config.BandPathPoint) = .empty;
    errdefer {
        for (points_list.items) |p| alloc.free(p.label);
        points_list.deinit(alloc);
    }

    // Split by "-" or "|"
    var it = std.mem.tokenizeAny(u8, path_str, "-|");
    while (it.next()) |label_raw| {
        const label = std.mem.trim(u8, label_raw, " ");
        if (label.len == 0) continue;

        const k = findHighSymPoint(table, label) orelse return error.UnknownHighSymmetryPoint;
        try points_list.append(alloc, .{
            .label = try alloc.dupe(u8, label),
            .k = k,
        });
    }

    if (points_list.items.len < 2) return error.InsufficientPathPoints;

    return .{ .points = try points_list.toOwnedSlice(alloc) };
}

fn getHighSymTable(lattice: BravaisLattice) []const HighSymPoint {
    return switch (lattice) {
        .cub => &.{
            .{ .label = "G", .k = .{ .x = 0.0, .y = 0.0, .z = 0.0 } },
            .{ .label = "X", .k = .{ .x = 0.5, .y = 0.0, .z = 0.0 } },
            .{ .label = "M", .k = .{ .x = 0.5, .y = 0.5, .z = 0.0 } },
            .{ .label = "R", .k = .{ .x = 0.5, .y = 0.5, .z = 0.5 } },
        },
        .fcc => &.{
            .{ .label = "G", .k = .{ .x = 0.0, .y = 0.0, .z = 0.0 } },
            .{ .label = "X", .k = .{ .x = 0.0, .y = 0.5, .z = 0.5 } },
            .{ .label = "W", .k = .{ .x = 0.25, .y = 0.5, .z = 0.75 } },
            .{ .label = "K", .k = .{ .x = 0.375, .y = 0.375, .z = 0.75 } },
            .{ .label = "L", .k = .{ .x = 0.5, .y = 0.5, .z = 0.5 } },
            .{ .label = "U", .k = .{ .x = 0.25, .y = 0.625, .z = 0.625 } },
        },
        .bcc => &.{
            .{ .label = "G", .k = .{ .x = 0.0, .y = 0.0, .z = 0.0 } },
            .{ .label = "H", .k = .{ .x = -0.5, .y = 0.5, .z = 0.5 } },
            .{ .label = "N", .k = .{ .x = 0.0, .y = 0.0, .z = 0.5 } },
            .{ .label = "P", .k = .{ .x = 0.25, .y = 0.25, .z = 0.25 } },
        },
        .hex => &.{
            .{ .label = "G", .k = .{ .x = 0.0, .y = 0.0, .z = 0.0 } },
            .{ .label = "M", .k = .{ .x = 0.5, .y = 0.0, .z = 0.0 } },
            .{ .label = "K", .k = .{ .x = 1.0 / 3.0, .y = 1.0 / 3.0, .z = 0.0 } },
            .{ .label = "A", .k = .{ .x = 0.0, .y = 0.0, .z = 0.5 } },
            .{ .label = "L", .k = .{ .x = 0.5, .y = 0.0, .z = 0.5 } },
            .{ .label = "H", .k = .{ .x = 1.0 / 3.0, .y = 1.0 / 3.0, .z = 0.5 } },
        },
        .tet => &.{
            .{ .label = "G", .k = .{ .x = 0.0, .y = 0.0, .z = 0.0 } },
            .{ .label = "X", .k = .{ .x = 0.5, .y = 0.0, .z = 0.0 } },
            .{ .label = "M", .k = .{ .x = 0.5, .y = 0.5, .z = 0.0 } },
            .{ .label = "Z", .k = .{ .x = 0.0, .y = 0.0, .z = 0.5 } },
            .{ .label = "R", .k = .{ .x = 0.0, .y = 0.5, .z = 0.5 } },
            .{ .label = "A", .k = .{ .x = 0.5, .y = 0.5, .z = 0.5 } },
        },
    };
}

fn findHighSymPoint(table: []const HighSymPoint, label: []const u8) ?math.Vec3 {
    for (table) |pt| {
        if (std.mem.eql(u8, pt.label, label)) return pt.k;
    }
    return null;
}

/// Resolve a path string to BandPathPoint array.
/// - "auto": detect Bravais lattice from cell and use standard path
/// - "G-X-W-K-G-L": parse label string and look up coordinates
pub fn resolvePathString(
    alloc: std.mem.Allocator,
    path_str: []const u8,
    cell_bohr: math.Mat3,
) !AutoKPathResult {
    const lattice = try detectBravaisLattice(cell_bohr);

    if (std.mem.eql(u8, path_str, "auto")) {
        return getStandardKPath(alloc, lattice);
    }

    return parsePathString(alloc, path_str, lattice);
}

// ============================================================
// Tests
// ============================================================

const testing = std.testing;

// --- Task 1: Bravais lattice detection ---

test "detectBravaisLattice: simple cubic" {
    // a=b=c=10 Bohr, all 90 degrees
    const cell = math.Mat3.fromRows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    );
    try testing.expectEqual(BravaisLattice.cub, try detectBravaisLattice(cell));
}

test "detectBravaisLattice: FCC primitive" {
    // FCC primitive vectors: a/2 * (0,1,1), (1,0,1), (1,1,0)
    const a = 10.26; // Si lattice constant in Bohr
    const h = a / 2.0;
    const cell = math.Mat3.fromRows(
        .{ .x = 0.0, .y = h, .z = h },
        .{ .x = h, .y = 0.0, .z = h },
        .{ .x = h, .y = h, .z = 0.0 },
    );
    try testing.expectEqual(BravaisLattice.fcc, try detectBravaisLattice(cell));
}

test "detectBravaisLattice: BCC primitive" {
    // BCC primitive vectors: a/2 * (-1,1,1), (1,-1,1), (1,1,-1)
    const a = 5.0;
    const h = a / 2.0;
    const cell = math.Mat3.fromRows(
        .{ .x = -h, .y = h, .z = h },
        .{ .x = h, .y = -h, .z = h },
        .{ .x = h, .y = h, .z = -h },
    );
    try testing.expectEqual(BravaisLattice.bcc, try detectBravaisLattice(cell));
}

test "detectBravaisLattice: hexagonal" {
    // Hex: a1=(a,0,0), a2=(-a/2, a*sqrt3/2, 0), a3=(0,0,c)
    const a = 4.65; // Graphene
    const c = 20.0;
    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = -a / 2.0, .y = a * @sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );
    try testing.expectEqual(BravaisLattice.hex, try detectBravaisLattice(cell));
}

test "detectBravaisLattice: tetragonal" {
    // a == b != c, all 90 degrees
    const cell = math.Mat3.fromRows(
        .{ .x = 5.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 5.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 8.0 },
    );
    try testing.expectEqual(BravaisLattice.tet, try detectBravaisLattice(cell));
}

// --- Task 2: Standard k-path generation ---

test "getStandardKPath: FCC has 6 points G-X-W-K-G-L" {
    const alloc = testing.allocator;
    var result = try getStandardKPath(alloc, .fcc);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 6), result.points.len);
    try testing.expectEqualStrings("G", result.points[0].label);
    try testing.expectEqualStrings("X", result.points[1].label);
    try testing.expectEqualStrings("W", result.points[2].label);
    try testing.expectEqualStrings("K", result.points[3].label);
    try testing.expectEqualStrings("G", result.points[4].label);
    try testing.expectEqualStrings("L", result.points[5].label);

    // Check Gamma = origin
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.points[0].k.x, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.points[0].k.y, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.points[0].k.z, 1e-10);

    // Check L = (0.5, 0.5, 0.5)
    try testing.expectApproxEqAbs(@as(f64, 0.5), result.points[5].k.x, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.5), result.points[5].k.y, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.5), result.points[5].k.z, 1e-10);
}

test "getStandardKPath: HEX has 5 points G-M-K-G-A" {
    const alloc = testing.allocator;
    var result = try getStandardKPath(alloc, .hex);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 5), result.points.len);
    try testing.expectEqualStrings("G", result.points[0].label);
    try testing.expectEqualStrings("M", result.points[1].label);
    try testing.expectEqualStrings("K", result.points[2].label);
    try testing.expectEqualStrings("G", result.points[3].label);
    try testing.expectEqualStrings("A", result.points[4].label);

    // K point = (1/3, 1/3, 0) for hex
    try testing.expectApproxEqAbs(1.0 / 3.0, result.points[2].k.x, 1e-10);
    try testing.expectApproxEqAbs(1.0 / 3.0, result.points[2].k.y, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.points[2].k.z, 1e-10);
}

test "getStandardKPath: BCC has 5 points G-H-N-G-P" {
    const alloc = testing.allocator;
    var result = try getStandardKPath(alloc, .bcc);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 5), result.points.len);
    try testing.expectEqualStrings("G", result.points[0].label);
    try testing.expectEqualStrings("H", result.points[1].label);
    try testing.expectEqualStrings("N", result.points[2].label);
    try testing.expectEqualStrings("G", result.points[3].label);
    try testing.expectEqualStrings("P", result.points[4].label);
}

// --- Task 3: Path string parser ---

test "parsePathString: G-X-W-K-G-L for FCC" {
    const alloc = testing.allocator;
    var result = try parsePathString(alloc, "G-X-W-K-G-L", .fcc);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 6), result.points.len);
    try testing.expectEqualStrings("G", result.points[0].label);
    try testing.expectEqualStrings("X", result.points[1].label);
    try testing.expectEqualStrings("L", result.points[5].label);

    // X for FCC = (0, 0.5, 0.5)
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.points[1].k.x, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.5), result.points[1].k.y, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.5), result.points[1].k.z, 1e-10);
}

test "parsePathString: with | separator" {
    const alloc = testing.allocator;
    var result = try parsePathString(alloc, "G-X-W|K-G-L", .fcc);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 6), result.points.len);
    try testing.expectEqualStrings("W", result.points[2].label);
    try testing.expectEqualStrings("K", result.points[3].label);
}

test "parsePathString: unknown label returns error" {
    const alloc = testing.allocator;
    const result = parsePathString(alloc, "G-X-INVALID", .fcc);
    try testing.expectError(error.UnknownHighSymmetryPoint, result);
}

test "parsePathString: single point returns error" {
    const alloc = testing.allocator;
    const result = parsePathString(alloc, "G", .fcc);
    try testing.expectError(error.InsufficientPathPoints, result);
}

test "parsePathString: hex G-M-K-G-A" {
    const alloc = testing.allocator;
    var result = try parsePathString(alloc, "G-M-K-G-A", .hex);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 5), result.points.len);
    // M for hex = (0.5, 0, 0)
    try testing.expectApproxEqAbs(@as(f64, 0.5), result.points[1].k.x, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.points[1].k.y, 1e-10);
}

// --- Task 4: resolvePathString integration ---

test "resolvePathString: auto for FCC Si cell" {
    const alloc = testing.allocator;
    const a = 10.26;
    const h = a / 2.0;
    const cell = math.Mat3.fromRows(
        .{ .x = 0.0, .y = h, .z = h },
        .{ .x = h, .y = 0.0, .z = h },
        .{ .x = h, .y = h, .z = 0.0 },
    );
    var result = try resolvePathString(alloc, "auto", cell);
    defer result.deinit(alloc);

    // FCC auto path: G-X-W-K-G-L (6 points)
    try testing.expectEqual(@as(usize, 6), result.points.len);
    try testing.expectEqualStrings("G", result.points[0].label);
    try testing.expectEqualStrings("L", result.points[5].label);
}

test "resolvePathString: explicit path string for FCC" {
    const alloc = testing.allocator;
    const a = 10.26;
    const h = a / 2.0;
    const cell = math.Mat3.fromRows(
        .{ .x = 0.0, .y = h, .z = h },
        .{ .x = h, .y = 0.0, .z = h },
        .{ .x = h, .y = h, .z = 0.0 },
    );
    var result = try resolvePathString(alloc, "G-L-W-G", cell);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 4), result.points.len);
    try testing.expectEqualStrings("G", result.points[0].label);
    try testing.expectEqualStrings("L", result.points[1].label);
    try testing.expectEqualStrings("W", result.points[2].label);
    try testing.expectEqualStrings("G", result.points[3].label);
}

test "resolvePathString: auto for hexagonal graphene cell" {
    const alloc = testing.allocator;
    const a = 4.65;
    const c = 20.0;
    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = -a / 2.0, .y = a * @sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );
    var result = try resolvePathString(alloc, "auto", cell);
    defer result.deinit(alloc);

    // HEX auto path: G-M-K-G-A (5 points)
    try testing.expectEqual(@as(usize, 5), result.points.len);
    try testing.expectEqualStrings("G", result.points[0].label);
    try testing.expectEqualStrings("K", result.points[2].label);
}
