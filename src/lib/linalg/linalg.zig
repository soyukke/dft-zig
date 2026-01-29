//! SIMD-optimized linear algebra library
//!
//! Provides high-performance implementations for DFT calculations.

pub const complex_vec = @import("complex_vec.zig");
pub const blas = @import("blas.zig");

pub const Complex = complex_vec.Complex;

// Re-export commonly used functions
pub const innerProduct = complex_vec.innerProduct;
pub const vectorNorm = complex_vec.vectorNorm;
pub const axpy = complex_vec.axpy;
pub const axpyComplex = complex_vec.axpyComplex;
pub const scale = complex_vec.scale;
pub const scaleInPlace = complex_vec.scaleInPlace;
pub const zero = complex_vec.zero;
pub const copy = complex_vec.copy;

test {
    _ = complex_vec;
}
