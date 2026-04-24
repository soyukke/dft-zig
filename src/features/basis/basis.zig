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
pub const normalization_s = gaussian.normalization_s;
pub const normalization = gaussian.normalization;
pub const primitive_overlap_ss = gaussian.primitive_overlap_ss;
pub const num_cartesian = gaussian.num_cartesian;
pub const cartesian_exponents = gaussian.cartesian_exponents;
pub const double_factorial = gaussian.double_factorial;
pub const MAX_CART = gaussian.MAX_CART;

test {
    _ = gaussian;
    _ = sto3g;
    _ = basis631g;
    _ = basis631g_2dfp;
    _ = aux_basis;
}
