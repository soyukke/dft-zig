//! Numerical integration grids for DFT exchange-correlation.
//!
//! Provides Becke molecular grids combining:
//! - Lebedev angular quadrature
//! - Treutler-Ahlrichs or Mura-Knowles radial quadrature
//! - Becke space partitioning for multi-atom systems

pub const lebedev = @import("lebedev.zig");
pub const radial = @import("radial.zig");
pub const becke = @import("becke.zig");
pub const xc_functionals = @import("../xc/xc_functionals.zig");

// Re-export commonly used types
pub const GridPoint = becke.GridPoint;
pub const GridConfig = becke.GridConfig;
pub const AtomGrid = becke.Atom;
pub const LebedevPoint = lebedev.LebedevPoint;
pub const RadialPoint = radial.RadialPoint;

pub const buildMolecularGrid = becke.buildMolecularGrid;

test {
    _ = lebedev;
    _ = radial;
    _ = becke;
    _ = xc_functionals;
}
