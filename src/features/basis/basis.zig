//! Gaussian-type orbital (GTO) basis set module.
//!
//! Re-exports types and data for constructing molecular basis sets.

pub const gaussian = @import("gaussian.zig");
pub const sto3g = @import("sto3g.zig");
pub const basis631g = @import("basis631g.zig");
pub const basis631g_2dfp = @import("basis631g_2dfp.zig");
pub const aux_basis = @import("aux_basis.zig");

pub const PrimitiveGaussian = gaussian.PrimitiveGaussian;
pub const ContractedShell = gaussian.ContractedShell;
pub const BasisFunction = gaussian.BasisFunction;
pub const AngularMomentum = gaussian.AngularMomentum;
pub const normalizationS = gaussian.normalizationS;
pub const normalization = gaussian.normalization;
pub const primitiveOverlapSS = gaussian.primitiveOverlapSS;
pub const numCartesian = gaussian.numCartesian;
pub const cartesianExponents = gaussian.cartesianExponents;
pub const doubleFactorial = gaussian.doubleFactorial;
pub const MAX_CART = gaussian.MAX_CART;

test {
    _ = gaussian;
    _ = sto3g;
    _ = basis631g;
    _ = basis631g_2dfp;
    _ = aux_basis;
}
