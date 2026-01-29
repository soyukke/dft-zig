const math = @import("../math/math.zig");

/// FFT grid metadata used in SCF.
pub const Grid = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    cell: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    min_h: i32,
    min_k: i32,
    min_l: i32,

    /// Return total grid points.
    pub fn count(self: Grid) usize {
        return self.nx * self.ny * self.nz;
    }
};
