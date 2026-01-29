//! Metal GPU bridge for DFT-Zig
//!
//! C-callable interface to Apple Metal for GPU-accelerated FFT and compute.
//! This header is consumed by both the Objective-C implementation and Zig via @cImport.

#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle for a Metal GPU context (device + command queue).
typedef struct MetalContext MetalContext;

/// Opaque handle for a Metal FFT plan.
typedef struct MetalFftPlan MetalFftPlan;

// ============== Device Management ==============

/// Create a Metal context using the system default GPU device.
/// Returns NULL on failure (e.g., no Metal-capable GPU).
MetalContext* metal_create_context(void);

/// Destroy a Metal context and release all associated resources.
void metal_destroy_context(MetalContext* ctx);

/// Get the GPU device name (for diagnostics). Caller must NOT free the string.
const char* metal_device_name(const MetalContext* ctx);

/// Check if Metal is available on this system.
bool metal_is_available(void);

// ============== FFT ==============

/// Create a 3D FFT plan for complex-to-complex transforms.
/// data_re and data_im are staging buffers on the GPU side (split complex format).
/// nx, ny, nz: grid dimensions.
/// Returns NULL on failure.
MetalFftPlan* metal_fft_create_plan(MetalContext* ctx,
                                    uint64_t nx, uint64_t ny, uint64_t nz);

/// Destroy a Metal FFT plan.
void metal_fft_destroy_plan(MetalFftPlan* plan);

/// Execute forward FFT (complex-to-complex, in-place on interleaved data).
/// data: pointer to interleaved complex doubles (re0, im0, re1, im1, ...).
/// count: number of complex elements (must equal nx*ny*nz).
/// The result is written back to `data`.
void metal_fft_forward(MetalFftPlan* plan, double* data, uint64_t count);

/// Execute inverse FFT (complex-to-complex, in-place on interleaved data).
/// Result is normalized by 1/(nx*ny*nz).
void metal_fft_inverse(MetalFftPlan* plan, double* data, uint64_t count);

// ============== GPU Buffer Management ==============

/// Allocate a GPU-visible buffer of the given size in bytes.
/// Returns an opaque handle, or NULL on failure.
void* metal_alloc_buffer(MetalContext* ctx, uint64_t size_bytes);

/// Free a GPU buffer.
void metal_free_buffer(void* buffer);

/// Copy data from CPU to GPU buffer.
void metal_upload(void* gpu_buffer, const void* cpu_data, uint64_t size_bytes);

/// Copy data from GPU buffer to CPU.
void metal_download(const void* gpu_buffer, void* cpu_data, uint64_t size_bytes);

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H
