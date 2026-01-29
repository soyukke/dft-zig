//! Metal GPU bridge implementation for DFT-Zig
//!
//! Provides GPU-accelerated FFT via Metal Performance Shaders (MPS)
//! and basic GPU buffer management.
//!
//! This is an Objective-C file that wraps the Metal API for use from Zig.
//! Compiled with -fobjc-arc (Automatic Reference Counting).
//! Uses CFBridgingRetain/Release to store ObjC objects in C structs.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include "metal_bridge.h"
#include <string.h>

// ============== Internal structures ==============
// Store ObjC objects as void* since C structs cannot hold __strong references.
// Ownership is managed via CFBridgingRetain (take ownership) and
// CFBridgingRelease (release ownership).

struct MetalContext {
    void* device;        // id<MTLDevice>, retained via CFBridgingRetain
    void* commandQueue;  // id<MTLCommandQueue>, retained via CFBridgingRetain
    char deviceName[256];
};

struct MetalFftPlan {
    MetalContext* ctx;
    uint64_t nx;
    uint64_t ny;
    uint64_t nz;
    uint64_t total; // nx * ny * nz
};

// ============== Device Management ==============

MetalContext* metal_create_context(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return NULL;
        }

        MetalContext* ctx = (MetalContext*)calloc(1, sizeof(MetalContext));
        if (!ctx) {
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            free(ctx);
            return NULL;
        }

        // Store device name before transferring ownership
        const char* name = [[device name] UTF8String];
        if (name) {
            strncpy(ctx->deviceName, name, sizeof(ctx->deviceName) - 1);
            ctx->deviceName[sizeof(ctx->deviceName) - 1] = '\0';
        }

        // Transfer ownership to C struct via CFBridgingRetain (+1 retain)
        ctx->device = (void*)CFBridgingRetain(device);
        ctx->commandQueue = (void*)CFBridgingRetain(queue);

        return ctx;
    }
}

void metal_destroy_context(MetalContext* ctx) {
    if (!ctx) return;
    @autoreleasepool {
        // Release ownership via CFBridgingRelease (-1 retain)
        if (ctx->commandQueue) {
            (void)CFBridgingRelease(ctx->commandQueue);
            ctx->commandQueue = NULL;
        }
        if (ctx->device) {
            (void)CFBridgingRelease(ctx->device);
            ctx->device = NULL;
        }
        free(ctx);
    }
}

const char* metal_device_name(const MetalContext* ctx) {
    if (!ctx) return "unknown";
    return ctx->deviceName;
}

bool metal_is_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return (device != nil);
    }
}

// ============== FFT ==============

MetalFftPlan* metal_fft_create_plan(MetalContext* ctx,
                                    uint64_t nx, uint64_t ny, uint64_t nz) {
    if (!ctx || nx == 0 || ny == 0 || nz == 0) return NULL;

    MetalFftPlan* plan = (MetalFftPlan*)calloc(1, sizeof(MetalFftPlan));
    if (!plan) return NULL;

    plan->ctx = ctx;
    plan->nx = nx;
    plan->ny = ny;
    plan->nz = nz;
    plan->total = nx * ny * nz;

    return plan;
}

void metal_fft_destroy_plan(MetalFftPlan* plan) {
    if (!plan) return;
    free(plan);
}

/// Perform 3D FFT using Metal compute pipeline.
///
/// Strategy: Use MTLBuffer for GPU-side storage, encode compute commands
/// to perform element-wise operations. For the actual FFT, we use a
/// row-by-row approach with 1D FFT compute shaders.
///
/// NOTE: This initial implementation copies data to/from GPU and exercises
/// the full GPU data path (upload -> GPU roundtrip -> download).
/// The actual FFT compute kernel will be added in a subsequent phase.
/// Currently the data passes through GPU unmodified (identity transform),
/// so forward() is a no-op and inverse() only applies normalization.
static void metal_fft_execute(MetalFftPlan* plan, double* data, uint64_t count, bool inverse) {
    if (!plan || !data || count != plan->total) return;

    @autoreleasepool {
        MetalContext* ctx = plan->ctx;
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->commandQueue;
        uint64_t buffer_size = count * 2 * sizeof(double); // interleaved complex

        // Create GPU buffer with shared storage (CPU+GPU visible on Apple Silicon)
        id<MTLBuffer> buffer = [device newBufferWithBytes:data
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
        if (!buffer) return;

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) return;

        // TODO: Encode actual FFT compute commands here.
        // For this foundation step, we set up the GPU pipeline infrastructure
        // and perform a GPU roundtrip to verify the data path works.
        //
        // The actual FFT kernel will be added in Phase 2:
        // - Metal compute shaders for radix-2 butterfly operations
        // - Or integration with vDSP on GPU via shared memory

        // Commit and wait for GPU work to complete
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back to CPU
        memcpy(data, [buffer contents], buffer_size);

        // Apply normalization for inverse
        if (inverse) {
            double scale = 1.0 / (double)(plan->total);
            for (uint64_t i = 0; i < count * 2; i++) {
                data[i] *= scale;
            }
        }
    }
}

void metal_fft_forward(MetalFftPlan* plan, double* data, uint64_t count) {
    metal_fft_execute(plan, data, count, false);
}

void metal_fft_inverse(MetalFftPlan* plan, double* data, uint64_t count) {
    metal_fft_execute(plan, data, count, true);
}

// ============== GPU Buffer Management ==============

void* metal_alloc_buffer(MetalContext* ctx, uint64_t size_bytes) {
    if (!ctx || size_bytes == 0) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
        id<MTLBuffer> buffer = [device newBufferWithLength:size_bytes
                                                  options:MTLResourceStorageModeShared];
        if (!buffer) return NULL;
        // Transfer ownership to caller via CFBridgingRetain
        return (void*)CFBridgingRetain(buffer);
    }
}

void metal_free_buffer(void* buffer) {
    if (!buffer) return;
    @autoreleasepool {
        // Release ownership
        (void)CFBridgingRelease(buffer);
    }
}

void metal_upload(void* gpu_buffer, const void* cpu_data, uint64_t size_bytes) {
    if (!gpu_buffer || !cpu_data || size_bytes == 0) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)gpu_buffer;
    memcpy([buffer contents], cpu_data, size_bytes);
}

void metal_download(const void* gpu_buffer, void* cpu_data, uint64_t size_bytes) {
    if (!gpu_buffer || !cpu_data || size_bytes == 0) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)gpu_buffer;
    memcpy(cpu_data, [buffer contents], size_bytes);
}
