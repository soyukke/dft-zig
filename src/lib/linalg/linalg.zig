//! SIMD-optimized linear algebra library
//!
//! Provides high-performance implementations for DFT calculations.

pub const complex_vec = @import("complex_vec.zig");
pub const blas = @import("blas.zig");

pub const Complex = complex_vec.Complex;

// Re-export commonly used functions
pub const inner_product = complex_vec.inner_product;
pub const vector_norm = complex_vec.vector_norm;
pub const axpy = complex_vec.axpy;
pub const axpy_complex = complex_vec.axpy_complex;
pub const scale = complex_vec.scale;
pub const scale_in_place = complex_vec.scale_in_place;
pub const zero = complex_vec.zero;
pub const copy = complex_vec.copy;

test {
    _ = complex_vec;
}
