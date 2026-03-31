const std = @import("std");
const complex_mod = @import("complex.zig");

pub const Complex = complex_mod.Complex;
pub const complex = complex_mod;
pub const radial = @import("radial.zig");

test {
    _ = radial;
}

pub const Units = enum {
    angstrom,
    bohr,
};

/// Return scale factor to angstrom.
pub fn unitsScaleToAngstrom(units: Units) f64 {
    return switch (units) {
        .angstrom => 1.0,
        .bohr => 0.52917721092,
    };
}

/// Return scale factor to bohr.
pub fn unitsScaleToBohr(units: Units) f64 {
    return switch (units) {
        .angstrom => 1.0 / 0.52917721092,
        .bohr => 1.0,
    };
}

pub const Vec3 = struct {
    x: f64,
    y: f64,
    z: f64,

    /// Add two vectors.
    pub fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }

    /// Subtract two vectors.
    pub fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
    }

    /// Scale a vector by s.
    pub fn scale(a: Vec3, s: f64) Vec3 {
        return .{ .x = a.x * s, .y = a.y * s, .z = a.z * s };
    }

    /// Dot product of two vectors.
    pub fn dot(a: Vec3, b: Vec3) f64 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    /// Cross product of two vectors.
    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    /// Euclidean norm of a vector.
    pub fn norm(a: Vec3) f64 {
        return std.math.sqrt(Vec3.dot(a, a));
    }
};

pub const Mat3 = struct {
    m: [3][3]f64,

    /// Construct a matrix from row vectors.
    pub fn fromRows(a: Vec3, b: Vec3, c: Vec3) Mat3 {
        return .{
            .m = .{
                .{ a.x, a.y, a.z },
                .{ b.x, b.y, b.z },
                .{ c.x, c.y, c.z },
            },
        };
    }

    /// Get a row vector.
    pub fn row(self: Mat3, index: usize) Vec3 {
        return .{
            .x = self.m[index][0],
            .y = self.m[index][1],
            .z = self.m[index][2],
        };
    }

    /// Get a column vector.
    pub fn col(self: Mat3, index: usize) Vec3 {
        return .{
            .x = self.m[0][index],
            .y = self.m[1][index],
            .z = self.m[2][index],
        };
    }

    /// Multiply matrix by vector.
    pub fn mulVec(self: Mat3, v: Vec3) Vec3 {
        return .{
            .x = self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            .y = self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            .z = self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        };
    }

    /// Scale a matrix by s.
    pub fn scale(self: Mat3, s: f64) Mat3 {
        var out = self;
        out.m[0][0] *= s;
        out.m[0][1] *= s;
        out.m[0][2] *= s;
        out.m[1][0] *= s;
        out.m[1][1] *= s;
        out.m[1][2] *= s;
        out.m[2][0] *= s;
        out.m[2][1] *= s;
        out.m[2][2] *= s;
        return out;
    }
};

/// Compute reciprocal lattice (2pi convention).
pub fn reciprocal(cell: Mat3) Mat3 {
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);

    const volume = Vec3.dot(a1, Vec3.cross(a2, a3));
    const scale = 2.0 * std.math.pi / volume;
    const b1 = Vec3.scale(Vec3.cross(a2, a3), scale);
    const b2 = Vec3.scale(Vec3.cross(a3, a1), scale);
    const b3 = Vec3.scale(Vec3.cross(a1, a2), scale);
    return Mat3.fromRows(b1, b2, b3);
}

/// Convert fractional coordinates to Cartesian using lattice matrix.
/// result = frac.x * row0 + frac.y * row1 + frac.z * row2
/// Use this for k-point conversion: k_cart = fracToCart(k_frac, recip)
/// Or for real-space: r_cart = fracToCart(r_frac, cell)
pub fn fracToCart(frac: Vec3, lattice: Mat3) Vec3 {
    const v0 = lattice.row(0);
    const v1 = lattice.row(1);
    const v2 = lattice.row(2);
    return Vec3.add(
        Vec3.add(
            Vec3.scale(v0, frac.x),
            Vec3.scale(v1, frac.y),
        ),
        Vec3.scale(v2, frac.z),
    );
}
