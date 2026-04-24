const std = @import("std");
const math = @import("../math/math.zig");

pub const Atom = struct {
    symbol: []u8,
    position: math.Vec3,
};

pub const AtomList = struct {
    items: []Atom,

    /// Free owned atom symbols and list.
    pub fn deinit(self: *AtomList, alloc: std.mem.Allocator) void {
        for (self.items) |atom| {
            alloc.free(atom.symbol);
        }
        alloc.free(self.items);
    }
};

/// Load atoms from XYZ file.
pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !AtomList {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1024 * 1024));
    defer alloc.free(content);

    var it = std.mem.splitScalar(u8, content, '\n');
    const count_line = next_non_empty_line(&it) orelse return error.InvalidXyz;
    const atom_count = try parse_count(count_line);

    _ = next_non_empty_line(&it) orelse return error.InvalidXyz;

    var list: std.ArrayList(Atom) = .empty;
    errdefer {
        for (list.items) |atom| {
            alloc.free(atom.symbol);
        }
        list.deinit(alloc);
    }

    var read_atoms: usize = 0;
    while (read_atoms < atom_count) {
        const line = next_non_empty_line(&it) orelse return error.InvalidXyz;
        var tokens = std.mem.tokenizeAny(u8, line, " \t\r");
        const symbol = tokens.next() orelse return error.InvalidXyz;
        const x_str = tokens.next() orelse return error.InvalidXyz;
        const y_str = tokens.next() orelse return error.InvalidXyz;
        const z_str = tokens.next() orelse return error.InvalidXyz;

        const pos = math.Vec3{
            .x = try std.fmt.parseFloat(f64, x_str),
            .y = try std.fmt.parseFloat(f64, y_str),
            .z = try std.fmt.parseFloat(f64, z_str),
        };

        const symbol_copy = try alloc.dupe(u8, symbol);
        try list.append(alloc, .{ .symbol = symbol_copy, .position = pos });
        read_atoms += 1;
    }

    const atoms = try list.toOwnedSlice(alloc);
    return AtomList{ .items = atoms };
}

/// Validate that atoms are inside the given cell.
pub fn validate_in_cell(atoms: []Atom, cell: math.Mat3) !void {
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    const volume = math.Vec3.dot(a1, math.Vec3.cross(a2, a3));
    if (@abs(volume) <= 1e-12) return error.InvalidCell;

    const inv_volume = 1.0 / volume;
    const b1 = math.Vec3.scale(math.Vec3.cross(a2, a3), inv_volume);
    const b2 = math.Vec3.scale(math.Vec3.cross(a3, a1), inv_volume);
    const b3 = math.Vec3.scale(math.Vec3.cross(a1, a2), inv_volume);

    const tol = 1e-6;
    for (atoms) |atom| {
        const fx = math.Vec3.dot(b1, atom.position);
        const fy = math.Vec3.dot(b2, atom.position);
        const fz = math.Vec3.dot(b3, atom.position);
        if (fx < -tol or fx > 1.0 + tol) return error.InvalidXyzCell;
        if (fy < -tol or fy > 1.0 + tol) return error.InvalidXyzCell;
        if (fz < -tol or fz > 1.0 + tol) return error.InvalidXyzCell;
    }
}

/// Read next non-empty line.
fn next_non_empty_line(it: anytype) ?[]const u8 {
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        return line;
    }
    return null;
}

/// Parse atom count from line.
fn parse_count(line: []const u8) !usize {
    return try std.fmt.parseInt(usize, std.mem.trim(u8, line, " \t\r"), 10);
}
