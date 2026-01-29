//! Zig wrapper for the libcint C library.
//!
//! libcint is a high-performance library for computing Gaussian-type orbital
//! (GTO) integrals in quantum chemistry. This module provides:
//!
//!   - Conversion from DFT-Zig's `ContractedShell` format to libcint's
//!     `atm/bas/env` data layout.
//!   - Wrappers for computing 1-electron integrals (overlap, kinetic, nuclear).
//!   - Wrappers for computing 2-electron integrals (ERI).
//!   - Wrappers for gradient (first derivative) integrals.
//!
//! Usage:
//!   const cint_data = try LibcintData.init(alloc, shells, nuc_positions, nuc_charges);
//!   defer cint_data.deinit(alloc);
//!
//!   // Build overlap matrix
//!   const S = try buildOverlapMatrix(alloc, cint_data);
//!
//! When libcint is not available (not linked), all functions return
//! `error.LibcintNotAvailable`.

const std = @import("std");
const basis_mod = @import("../basis/basis.zig");
const ContractedShell = basis_mod.ContractedShell;

const libcint_options = @import("libcint_options");
const enable_libcint = libcint_options.enable_libcint;

/// Import libcint C API when available.
/// We only import cint.h for data structures (CINTOpt, etc.)
/// and declare the integral functions manually because cint_funcs.h
/// uses typedef'd function types that Zig's @cImport cannot resolve.
const c = if (enable_libcint) @cImport({
    @cInclude("cint.h");
}) else struct {};

/// Integral function type: the common signature for all libcint integral routines.
/// FINT = c_int (when HAVE_DEFINED_FINT_TYPE is not set, which is the default).
const CINTIntegralFn = if (enable_libcint) *const fn (
    out: [*]f64,
    dims: ?[*]c_int,
    shls: [*]c_int,
    atm: [*]c_int,
    natm: c_int,
    bas: [*]c_int,
    nbas: c_int,
    env: [*]f64,
    opt: ?*c.CINTOpt,
    cache: ?[*]f64,
) callconv(.c) c_int else void;

/// Extern declarations for the libcint integral functions we use.
const cint_fns = if (enable_libcint) struct {
    extern fn int1e_ovlp_cart(out: [*]f64, dims: ?[*]c_int, shls: [*]c_int, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64, opt: ?*c.CINTOpt, cache: ?[*]f64) callconv(.c) c_int;
    extern fn int1e_kin_cart(out: [*]f64, dims: ?[*]c_int, shls: [*]c_int, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64, opt: ?*c.CINTOpt, cache: ?[*]f64) callconv(.c) c_int;
    extern fn int1e_nuc_cart(out: [*]f64, dims: ?[*]c_int, shls: [*]c_int, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64, opt: ?*c.CINTOpt, cache: ?[*]f64) callconv(.c) c_int;
    extern fn int2e_cart(out: [*]f64, dims: ?[*]c_int, shls: [*]c_int, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64, opt: ?*c.CINTOpt, cache: ?[*]f64) callconv(.c) c_int;
    extern fn int1e_ipovlp_cart(out: [*]f64, dims: ?[*]c_int, shls: [*]c_int, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64, opt: ?*c.CINTOpt, cache: ?[*]f64) callconv(.c) c_int;
    extern fn int1e_ipkin_cart(out: [*]f64, dims: ?[*]c_int, shls: [*]c_int, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64, opt: ?*c.CINTOpt, cache: ?[*]f64) callconv(.c) c_int;
    extern fn int1e_ipnuc_cart(out: [*]f64, dims: ?[*]c_int, shls: [*]c_int, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64, opt: ?*c.CINTOpt, cache: ?[*]f64) callconv(.c) c_int;
    extern fn int2e_ip1_cart(out: [*]f64, dims: ?[*]c_int, shls: [*]c_int, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64, opt: ?*c.CINTOpt, cache: ?[*]f64) callconv(.c) c_int;

    // Optimizer functions
    extern fn cint2e_cart_optimizer(opt: *?*c.CINTOpt, atm: [*]c_int, natm: c_int, bas: [*]c_int, nbas: c_int, env: [*]f64) callconv(.c) void;
    extern fn CINTdel_optimizer(opt: *?*c.CINTOpt) callconv(.c) void;
} else struct {};

// ============================================================================
// Constants (matching cint.h)
// ============================================================================

const ATM_SLOTS: usize = 6;
const BAS_SLOTS: usize = 8;
const PTR_ENV_START: usize = 20;

// atm slots
const CHARGE_OF: usize = 0;
const PTR_COORD: usize = 1;
const NUC_MOD_OF: usize = 2;
const PTR_ZETA: usize = 3;

// bas slots
const ATOM_OF: usize = 0;
const ANG_OF: usize = 1;
const NPRIM_OF: usize = 2;
const NCTR_OF: usize = 3;
const KAPPA_OF: usize = 4;
const PTR_EXP: usize = 5;
const PTR_COEFF: usize = 6;

const POINT_NUC: c_int = 1;

/// Data layout for libcint: atm, bas, env arrays.
///
/// Follows libcint convention:
///   - atm: ATM_SLOTS ints per atom (charge, coord pointer, nuc model, ...)
///   - bas: BAS_SLOTS ints per shell (atom_of, ang, nprim, nctr, kappa, ptr_exp, ptr_coeff, ...)
///   - env: double array with coordinates, exponents, and normalized coefficients
pub const LibcintData = struct {
    atm: []c_int,
    bas: []c_int,
    env: []f64,
    natm: c_int,
    nbas: c_int,

    /// Shell offset table: shell_offsets[i] = first AO index of shell i.
    /// shell_offsets[nbas] = total number of AOs.
    shell_offsets: []usize,

    /// Initialize libcint data from DFT-Zig shell and atom data.
    ///
    /// Args:
    ///   shells: Array of contracted shells.
    ///   nuc_positions: Nuclear positions in Bohr, flat [natm][3].
    ///   nuc_charges: Nuclear charges (atomic numbers as f64).
    pub fn init(
        alloc: std.mem.Allocator,
        shells: []const ContractedShell,
        nuc_positions: []const [3]f64,
        nuc_charges: []const f64,
    ) !LibcintData {
        if (!enable_libcint) return error.LibcintNotAvailable;

        const natm = nuc_charges.len;
        const nbas = shells.len;

        // Calculate total env size needed
        // env layout: [PTR_ENV_START][atom coords: natm*3][exps+coeffs per shell]
        var env_size: usize = PTR_ENV_START + natm * 3;
        for (shells) |shell| {
            env_size += shell.primitives.len; // exponents
            env_size += shell.primitives.len; // normalized coefficients
        }

        const atm = try alloc.alloc(c_int, natm * ATM_SLOTS);
        const bas = try alloc.alloc(c_int, nbas * BAS_SLOTS);
        const env = try alloc.alloc(f64, env_size);
        const shell_offsets = try alloc.alloc(usize, nbas + 1);

        // Zero-initialize
        @memset(atm, 0);
        @memset(bas, 0);
        @memset(env, 0.0);

        // Fill atom data
        var env_ptr: usize = PTR_ENV_START;
        for (0..natm) |i| {
            const off = i * ATM_SLOTS;
            atm[off + CHARGE_OF] = @intFromFloat(nuc_charges[i]);
            atm[off + PTR_COORD] = @intCast(env_ptr);
            atm[off + NUC_MOD_OF] = POINT_NUC;
            env[env_ptr] = nuc_positions[i][0];
            env[env_ptr + 1] = nuc_positions[i][1];
            env[env_ptr + 2] = nuc_positions[i][2];
            env_ptr += 3;
        }

        // Fill basis data
        // We need to figure out which atom each shell belongs to.
        // Match by comparing shell center to atom positions.
        var ao_offset: usize = 0;
        for (0..nbas) |i| {
            const shell = shells[i];
            const nprim = shell.primitives.len;

            // Find the atom this shell belongs to
            const atom_idx = findAtom(nuc_positions, shell.center);

            const off = i * BAS_SLOTS;
            bas[off + ATOM_OF] = @intCast(atom_idx);
            bas[off + ANG_OF] = @intCast(shell.l);
            bas[off + NPRIM_OF] = @intCast(nprim);
            bas[off + NCTR_OF] = 1; // always 1 contraction in our representation
            bas[off + KAPPA_OF] = 0;
            bas[off + PTR_EXP] = @intCast(env_ptr);

            // Store exponents
            for (0..nprim) |p| {
                env[env_ptr + p] = shell.primitives[p].alpha;
            }
            env_ptr += nprim;

            bas[off + PTR_COEFF] = @intCast(env_ptr);

            // Store normalized coefficients: coeff * gto_norm(l, alpha)
            // This matches PySCF/libcint convention where env stores
            // raw_coeff * gto_norm(l, alpha).
            for (0..nprim) |p| {
                const alpha = shell.primitives[p].alpha;
                const raw_coeff = shell.primitives[p].coeff;
                env[env_ptr + p] = raw_coeff * gtoNorm(shell.l, alpha);
            }
            env_ptr += nprim;

            // Shell AO offsets
            shell_offsets[i] = ao_offset;
            ao_offset += basis_mod.numCartesian(shell.l);
        }
        shell_offsets[nbas] = ao_offset;

        return LibcintData{
            .atm = atm,
            .bas = bas,
            .env = env,
            .natm = @intCast(natm),
            .nbas = @intCast(nbas),
            .shell_offsets = shell_offsets,
        };
    }

    pub fn deinit(self: *const LibcintData, alloc: std.mem.Allocator) void {
        alloc.free(self.atm);
        alloc.free(self.bas);
        alloc.free(self.env);
        alloc.free(self.shell_offsets);
    }

    /// Total number of AOs (Cartesian).
    pub fn nao(self: *const LibcintData) usize {
        return self.shell_offsets[@intCast(self.nbas)];
    }
};

// ============================================================================
// Normalization
// ============================================================================

/// Compute libcint's GTO normalization factor for a primitive.
///
/// gto_norm(l, alpha) = 1 / sqrt( Gamma(l + 1.5) / (2 * (2*alpha)^(l+1.5)) )
///
/// This is the normalization for the radial part g(r) = r^l * exp(-alpha*r^2),
/// consistent with PySCF's gto.gto_norm and libcint's CINTgto_norm.
fn gtoNorm(l: u32, alpha: f64) f64 {
    // gaussian_int(s, a) = Gamma((s+1)/2) / (2 * a^((s+1)/2))
    // For s = 2*l + 2:
    //   gaussian_int(2l+2, 2a) = Gamma(l + 1.5) / (2 * (2a)^(l+1.5))
    const l_f: f64 = @floatFromInt(l);
    const two_alpha = 2.0 * alpha;

    // Gamma(l + 1.5) via recursion from Gamma(0.5) = sqrt(pi)
    // Gamma(n + 0.5) = (2n-1)!! / 2^n * sqrt(pi)  for n >= 0
    // Here n = l + 1, so Gamma(l + 1.5) = (2l+1)!! / 2^(l+1) * sqrt(pi)
    const gamma_val = gammaHalfInt(l + 1);

    // (2*alpha)^(l + 1.5)
    const pow_val = std.math.pow(f64, two_alpha, l_f + 1.5);

    const gaussian_int_val = gamma_val / (2.0 * pow_val);
    return 1.0 / @sqrt(gaussian_int_val);
}

/// Compute Gamma(n + 0.5) for integer n >= 0.
///
/// Gamma(0.5) = sqrt(pi)
/// Gamma(n + 0.5) = (n - 0.5) * (n - 1.5) * ... * 0.5 * sqrt(pi)
///                = (2n-1)!! / 2^n * sqrt(pi)
fn gammaHalfInt(n: u32) f64 {
    var result: f64 = std.math.sqrt(std.math.pi);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        result *= (@as(f64, @floatFromInt(i)) + 0.5);
    }
    return result;
}

// ============================================================================
// Cartesian normalization correction
// ============================================================================

/// Compute the Cartesian normalization correction factor for AO component (ax,ay,az).
///
/// With gtoNorm coefficients in env, libcint's Cartesian overlap diagonal is:
///   - S_diag = 1.0 for s-type (l=0) and p-type (l=1)
///   - S_diag = (2ax-1)!! * (2ay-1)!! * (2az-1)!! * 4π / (2l+1)!! for l >= 2
///
/// Our native code normalizes all Cartesian components to S_diag = 1.0.
/// The correction factor converts libcint integrals to our convention:
///   native_integral[i,j] = cint_integral[i,j] * factor[i] * factor[j]
///
/// factor = 1/sqrt(S_diag_cint) = sqrt((2l+1)!! / (4π * df_cart))  for l >= 2
/// factor = 1.0  for l <= 1
///
/// Values:
///   s (0,0,0): 1.0
///   p (1,0,0): 1.0
///   d (2,0,0): sqrt(5/(4π)) ≈ 0.6308,  d (1,1,0): sqrt(15/(4π)) ≈ 1.0925
///   f (3,0,0): sqrt(105/(4π*15)) ≈ 0.7464, f (1,1,1): sqrt(105/(4π)) ≈ 2.8906
fn cartNormFactor(ax: u32, ay: u32, az: u32) f64 {
    const l = ax + ay + az;
    if (l <= 1) return 1.0;
    // (2l+1)!! = 1 * 3 * 5 * ... * (2l+1)
    var df_2l1: f64 = 1.0;
    {
        var k: u32 = 1;
        while (k <= 2 * l + 1) : (k += 2) {
            df_2l1 *= @as(f64, @floatFromInt(k));
        }
    }
    const df_cart = basis_mod.doubleFactorial(ax) * basis_mod.doubleFactorial(ay) * basis_mod.doubleFactorial(az);
    return @sqrt(df_2l1 / (4.0 * std.math.pi * df_cart));
}

/// Build a table of Cartesian normalization correction factors for all AOs.
///
/// Returns an array of length n = total number of AOs. Each element is the
/// correction factor for the corresponding AO, such that:
///   native_integral[i,j] = libcint_integral[i,j] * factors[i] * factors[j]
fn buildCartNormFactors(alloc: std.mem.Allocator, data: LibcintData) ![]f64 {
    const n = data.nao();
    const factors = try alloc.alloc(f64, n);

    var shell_idx: usize = 0;
    while (shell_idx < @as(usize, @intCast(data.nbas))) : (shell_idx += 1) {
        const l: u32 = @intCast(data.bas[shell_idx * BAS_SLOTS + ANG_OF]);
        const ncart = basis_mod.numCartesian(l);
        const offset = data.shell_offsets[shell_idx];
        const cart_exps = basis_mod.cartesianExponents(l);

        for (0..ncart) |k| {
            factors[offset + k] = cartNormFactor(cart_exps[k].x, cart_exps[k].y, cart_exps[k].z);
        }
    }

    return factors;
}

// ============================================================================
// Atom matching
// ============================================================================

/// Find the atom index whose position matches the given shell center.
fn findAtom(nuc_positions: []const [3]f64, center: @import("../math/math.zig").Vec3) usize {
    const tol = 1e-10;
    for (nuc_positions, 0..) |pos, i| {
        const dx = pos[0] - center.x;
        const dy = pos[1] - center.y;
        const dz = pos[2] - center.z;
        if (dx * dx + dy * dy + dz * dz < tol) {
            return i;
        }
    }
    // Fallback: return atom 0 (should not happen with valid data)
    return 0;
}

// ============================================================================
// 1-electron integrals
// ============================================================================

/// Build the overlap matrix using libcint.
///
/// Returns a flat n×n array in row-major order.
pub fn buildOverlapMatrix(alloc: std.mem.Allocator, data: LibcintData) ![]f64 {
    if (!enable_libcint) return error.LibcintNotAvailable;
    return buildOneElectronMatrix(alloc, data, cint_fns.int1e_ovlp_cart);
}

/// Build the kinetic energy matrix using libcint.
pub fn buildKineticMatrix(alloc: std.mem.Allocator, data: LibcintData) ![]f64 {
    if (!enable_libcint) return error.LibcintNotAvailable;
    return buildOneElectronMatrix(alloc, data, cint_fns.int1e_kin_cart);
}

/// Build the nuclear attraction matrix using libcint.
pub fn buildNuclearMatrix(alloc: std.mem.Allocator, data: LibcintData) ![]f64 {
    if (!enable_libcint) return error.LibcintNotAvailable;
    return buildOneElectronMatrix(alloc, data, cint_fns.int1e_nuc_cart);
}

/// Generic 1-electron integral matrix builder.
fn buildOneElectronMatrix(
    alloc: std.mem.Allocator,
    data: LibcintData,
    comptime integral_fn: anytype,
) ![]f64 {
    const n = data.nao();
    const mat = try alloc.alloc(f64, n * n);
    @memset(mat, 0.0);

    // Build Cartesian normalization correction factors
    const cart_norms = try buildCartNormFactors(alloc, data);
    defer alloc.free(cart_norms);

    // Buffer for shell-pair integrals (max: 15 × 15 = 225 for g-type)
    var buf: [225]f64 = undefined;
    var shls: [2]c_int = undefined;

    var i: c_int = 0;
    while (i < data.nbas) : (i += 1) {
        var j: c_int = 0;
        while (j < data.nbas) : (j += 1) {
            shls[0] = i;
            shls[1] = j;

            const ni = basis_mod.numCartesian(
                @intCast(data.bas[@as(usize, @intCast(i)) * BAS_SLOTS + ANG_OF]),
            );
            const nj = basis_mod.numCartesian(
                @intCast(data.bas[@as(usize, @intCast(j)) * BAS_SLOTS + ANG_OF]),
            );

            _ = integral_fn(
                &buf,
                null, // dims (null = use default)
                &shls,
                data.atm.ptr,
                data.natm,
                data.bas.ptr,
                data.nbas,
                data.env.ptr,
                null, // CINTOpt
                null, // cache
            );

            // Copy into matrix (libcint uses column-major for shell block)
            // Apply Cartesian normalization correction: native = cint * norm_i * norm_j
            const oi = data.shell_offsets[@intCast(i)];
            const oj = data.shell_offsets[@intCast(j)];
            for (0..ni) |ii| {
                for (0..nj) |jj| {
                    // libcint output: column-major within shell block
                    mat[(oi + ii) * n + (oj + jj)] = buf[jj * ni + ii] * cart_norms[oi + ii] * cart_norms[oj + jj];
                }
            }
        }
    }

    return mat;
}

// ============================================================================
// 2-electron integrals (ERI)
// ============================================================================

/// Compute a single shell-quartet ERI block (ij|kl) using libcint.
///
/// Returns the buffer size (number of doubles written).
/// The output is in the caller-provided buffer, column-major within the block:
///   buf[l_idx * nk * nj * ni + k_idx * nj * ni + j_idx * ni + i_idx]
pub fn shellQuartetEri(
    data: LibcintData,
    shell_i: usize,
    shell_j: usize,
    shell_k: usize,
    shell_l: usize,
    buf: []f64,
) !usize {
    if (!enable_libcint) return error.LibcintNotAvailable;

    var shls: [4]c_int = .{
        @intCast(shell_i),
        @intCast(shell_j),
        @intCast(shell_k),
        @intCast(shell_l),
    };

    const ni = basis_mod.numCartesian(@intCast(data.bas[shell_i * BAS_SLOTS + ANG_OF]));
    const nj = basis_mod.numCartesian(@intCast(data.bas[shell_j * BAS_SLOTS + ANG_OF]));
    const nk = basis_mod.numCartesian(@intCast(data.bas[shell_k * BAS_SLOTS + ANG_OF]));
    const nl = basis_mod.numCartesian(@intCast(data.bas[shell_l * BAS_SLOTS + ANG_OF]));

    const size = ni * nj * nk * nl;
    std.debug.assert(buf.len >= size);

    _ = cint_fns.int2e_cart(
        buf.ptr,
        null, // dims
        &shls,
        data.atm.ptr,
        data.natm,
        data.bas.ptr,
        data.nbas,
        data.env.ptr,
        null, // CINTOpt
        null, // cache
    );

    return size;
}

/// Build the full ERI tensor as a flat array indexed as (ij|kl).
///
/// This is primarily for validation — for production use, the direct SCF
/// approach in fock.zig is preferred.
pub fn buildEriTensor(alloc: std.mem.Allocator, data: LibcintData) ![]f64 {
    if (!enable_libcint) return error.LibcintNotAvailable;

    const n = data.nao();
    const n4 = n * n * n * n;
    const eri = try alloc.alloc(f64, n4);
    @memset(eri, 0.0);

    // Build Cartesian normalization correction factors
    const cart_norms = try buildCartNormFactors(alloc, data);
    defer alloc.free(cart_norms);

    // Buffer for max shell quartet: (15*15*15*15 = 50625 for g-type)
    // For f-type (l=3): 10^4 = 10000
    const max_buf_size = 50625;
    var buf: [max_buf_size]f64 = undefined;
    var shls: [4]c_int = undefined;

    var si: c_int = 0;
    while (si < data.nbas) : (si += 1) {
        var sj: c_int = 0;
        while (sj < data.nbas) : (sj += 1) {
            var sk: c_int = 0;
            while (sk < data.nbas) : (sk += 1) {
                var sl: c_int = 0;
                while (sl < data.nbas) : (sl += 1) {
                    shls[0] = si;
                    shls[1] = sj;
                    shls[2] = sk;
                    shls[3] = sl;

                    const ni = basis_mod.numCartesian(@intCast(data.bas[@as(usize, @intCast(si)) * BAS_SLOTS + ANG_OF]));
                    const nj = basis_mod.numCartesian(@intCast(data.bas[@as(usize, @intCast(sj)) * BAS_SLOTS + ANG_OF]));
                    const nk = basis_mod.numCartesian(@intCast(data.bas[@as(usize, @intCast(sk)) * BAS_SLOTS + ANG_OF]));
                    const nl = basis_mod.numCartesian(@intCast(data.bas[@as(usize, @intCast(sl)) * BAS_SLOTS + ANG_OF]));

                    _ = cint_fns.int2e_cart(
                        &buf,
                        null,
                        &shls,
                        data.atm.ptr,
                        data.natm,
                        data.bas.ptr,
                        data.nbas,
                        data.env.ptr,
                        null,
                        null,
                    );

                    // Copy: libcint column-major (i fastest) → our row-major (ij|kl)
                    // Apply Cartesian normalization correction
                    const oi = data.shell_offsets[@intCast(si)];
                    const oj = data.shell_offsets[@intCast(sj)];
                    const ok = data.shell_offsets[@intCast(sk)];
                    const ol = data.shell_offsets[@intCast(sl)];

                    for (0..ni) |ii| {
                        for (0..nj) |jj| {
                            for (0..nk) |kk| {
                                for (0..nl) |ll| {
                                    // libcint index: col-major: buf[i + j*ni + k*ni*nj + l*ni*nj*nk]
                                    const cidx = ii + jj * ni + kk * ni * nj + ll * ni * nj * nk;
                                    const mu = oi + ii;
                                    const nu = oj + jj;
                                    const lam = ok + kk;
                                    const sig = ol + ll;
                                    const norm = cart_norms[mu] * cart_norms[nu] * cart_norms[lam] * cart_norms[sig];
                                    eri[mu * n * n * n + nu * n * n + lam * n + sig] = buf[cidx] * norm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return eri;
}

// ============================================================================
// Gradient integrals (1e derivatives)
// ============================================================================

/// Compute overlap gradient integrals <NABLA i|j> using libcint.
///
/// Returns a [3][]f64 array: one n×n matrix per Cartesian direction (x, y, z).
/// Each matrix gives d/dR_i <i|j> for shell i.
pub fn buildOverlapGradient(alloc: std.mem.Allocator, data: LibcintData) ![3][]f64 {
    if (!enable_libcint) return error.LibcintNotAvailable;
    return buildOneElectronGradient(alloc, data, cint_fns.int1e_ipovlp_cart);
}

/// Compute kinetic gradient integrals <NABLA i|T|j> using libcint.
pub fn buildKineticGradient(alloc: std.mem.Allocator, data: LibcintData) ![3][]f64 {
    if (!enable_libcint) return error.LibcintNotAvailable;
    return buildOneElectronGradient(alloc, data, cint_fns.int1e_ipkin_cart);
}

/// Compute nuclear gradient integrals <NABLA i|V|j> using libcint.
pub fn buildNuclearGradient(alloc: std.mem.Allocator, data: LibcintData) ![3][]f64 {
    if (!enable_libcint) return error.LibcintNotAvailable;
    return buildOneElectronGradient(alloc, data, cint_fns.int1e_ipnuc_cart);
}

/// Generic 1-electron gradient integral builder.
/// The ip (NABLA) integrals have 3 components (x, y, z).
fn buildOneElectronGradient(
    alloc: std.mem.Allocator,
    data: LibcintData,
    comptime integral_fn: anytype,
) ![3][]f64 {
    const n = data.nao();
    var mats: [3][]f64 = undefined;
    for (0..3) |d| {
        mats[d] = try alloc.alloc(f64, n * n);
        @memset(mats[d], 0.0);
    }

    // Build Cartesian normalization correction factors
    const cart_norms = try buildCartNormFactors(alloc, data);
    defer alloc.free(cart_norms);

    // Buffer for 3 components × shell pair (max: 3 × 15 × 15 = 675)
    var buf: [675]f64 = undefined;
    var shls: [2]c_int = undefined;

    var i: c_int = 0;
    while (i < data.nbas) : (i += 1) {
        var j: c_int = 0;
        while (j < data.nbas) : (j += 1) {
            shls[0] = i;
            shls[1] = j;

            const ni = basis_mod.numCartesian(
                @intCast(data.bas[@as(usize, @intCast(i)) * BAS_SLOTS + ANG_OF]),
            );
            const nj = basis_mod.numCartesian(
                @intCast(data.bas[@as(usize, @intCast(j)) * BAS_SLOTS + ANG_OF]),
            );

            _ = integral_fn(
                &buf,
                null,
                &shls,
                data.atm.ptr,
                data.natm,
                data.bas.ptr,
                data.nbas,
                data.env.ptr,
                null,
                null,
            );

            // Output: 3 components, each ni×nj, column-major
            // Apply Cartesian normalization correction
            const oi = data.shell_offsets[@intCast(i)];
            const oj = data.shell_offsets[@intCast(j)];
            const block_size = ni * nj;

            for (0..3) |d| {
                for (0..ni) |ii| {
                    for (0..nj) |jj| {
                        const cidx = d * block_size + jj * ni + ii;
                        mats[d][(oi + ii) * n + (oj + jj)] = buf[cidx] * cart_norms[oi + ii] * cart_norms[oj + jj];
                    }
                }
            }
        }
    }

    return mats;
}

// ============================================================================
// 2-electron gradient integrals
// ============================================================================

/// Compute ERI first derivatives (NABLA i j|kl) for a shell quartet.
///
/// Returns 3 × ni × nj × nk × nl values in the buffer.
/// Layout: buf[comp * ni*nj*nk*nl + i + j*ni + k*ni*nj + l*ni*nj*nk]
/// comp = 0(x), 1(y), 2(z)
pub fn shellQuartetEriGradient(
    data: LibcintData,
    shell_i: usize,
    shell_j: usize,
    shell_k: usize,
    shell_l: usize,
    buf: []f64,
) !usize {
    if (!enable_libcint) return error.LibcintNotAvailable;

    var shls: [4]c_int = .{
        @intCast(shell_i),
        @intCast(shell_j),
        @intCast(shell_k),
        @intCast(shell_l),
    };

    const ni = basis_mod.numCartesian(@intCast(data.bas[shell_i * BAS_SLOTS + ANG_OF]));
    const nj = basis_mod.numCartesian(@intCast(data.bas[shell_j * BAS_SLOTS + ANG_OF]));
    const nk = basis_mod.numCartesian(@intCast(data.bas[shell_k * BAS_SLOTS + ANG_OF]));
    const nl = basis_mod.numCartesian(@intCast(data.bas[shell_l * BAS_SLOTS + ANG_OF]));

    const size = 3 * ni * nj * nk * nl;
    std.debug.assert(buf.len >= size);

    _ = cint_fns.int2e_ip1_cart(
        buf.ptr,
        null,
        &shls,
        data.atm.ptr,
        data.natm,
        data.bas.ptr,
        data.nbas,
        data.env.ptr,
        null,
        null,
    );

    return size;
}

// ============================================================================
// LibcintJKBuilder: Reusable J/K builder with pre-computed Schwarz + CINTOpt
// ============================================================================

/// Result of J/K matrix construction.
pub const JKResult = struct {
    j_matrix: []f64,
    k_matrix: []f64,
};

/// Pre-computed data for fast J/K matrix construction using libcint.
///
/// Create once before the SCF loop. The Schwarz table and CINTOpt are
/// density-independent and can be reused across iterations. Only the density
/// screening table needs to be rebuilt each iteration (done inside buildJK).
pub const LibcintJKBuilder = struct {
    data: LibcintData,
    cart_norms: []f64,
    schwarz: []f64, // nbas * nbas
    opt: ?*c.CINTOpt,
    nbas: usize,
    n: usize,
    schwarz_threshold: f64,

    pub fn init(alloc: std.mem.Allocator, data: LibcintData, schwarz_threshold: f64) !LibcintJKBuilder {
        if (!enable_libcint) return error.LibcintNotAvailable;

        const nbas: usize = @intCast(data.nbas);
        const n = data.nao();

        // Build cart norms
        const cart_norms = try buildCartNormFactors(alloc, data);

        // CINTOpt
        var opt: ?*c.CINTOpt = null;
        cint_fns.cint2e_cart_optimizer(
            &opt,
            data.atm.ptr,
            data.natm,
            data.bas.ptr,
            data.nbas,
            data.env.ptr,
        );

        // Schwarz table
        const schwarz = try alloc.alloc(f64, nbas * nbas);
        {
            const max_buf_size = 50625;
            var buf: [max_buf_size]f64 = undefined;
            var shls: [4]c_int = undefined;

            for (0..nbas) |si| {
                var sj: usize = 0;
                while (sj <= si) : (sj += 1) {
                    shls[0] = @intCast(si);
                    shls[1] = @intCast(sj);
                    shls[2] = @intCast(si);
                    shls[3] = @intCast(sj);

                    const ni = numCartesianFromBas(data, si);
                    const nj = numCartesianFromBas(data, sj);
                    const size = ni * nj * ni * nj;

                    _ = cint_fns.int2e_cart(
                        &buf,
                        null,
                        &shls,
                        data.atm.ptr,
                        data.natm,
                        data.bas.ptr,
                        data.nbas,
                        data.env.ptr,
                        opt,
                        null,
                    );

                    var max_val: f64 = 0.0;
                    for (0..size) |idx| {
                        const abs_val = @abs(buf[idx]);
                        if (abs_val > max_val) max_val = abs_val;
                    }
                    const q = @sqrt(max_val);
                    schwarz[si * nbas + sj] = q;
                    schwarz[sj * nbas + si] = q;
                }
            }
        }

        return .{
            .data = data,
            .cart_norms = cart_norms,
            .schwarz = schwarz,
            .opt = opt,
            .nbas = nbas,
            .n = n,
            .schwarz_threshold = schwarz_threshold,
        };
    }

    pub fn deinit(self: *LibcintJKBuilder, alloc: std.mem.Allocator) void {
        alloc.free(self.cart_norms);
        alloc.free(self.schwarz);
        if (self.opt != null) {
            cint_fns.CINTdel_optimizer(&self.opt);
        }
    }

    /// Build J and K matrices for the given density matrix.
    pub fn buildJK(
        self: *const LibcintJKBuilder,
        alloc: std.mem.Allocator,
        density: []const f64,
    ) !JKResult {
        const n = self.n;
        const nbas = self.nbas;
        const data = self.data;
        const cart_norms = self.cart_norms;
        const schwarz = self.schwarz;
        const schwarz_threshold = self.schwarz_threshold;

        const j_matrix = try alloc.alloc(f64, n * n);
        const k_matrix = try alloc.alloc(f64, n * n);
        @memset(j_matrix, 0.0);
        @memset(k_matrix, 0.0);

        // Density screening: D_max(i,j) = max|P[mu,nu]| over shell pair
        const d_max = try alloc.alloc(f64, nbas * nbas);
        defer alloc.free(d_max);
        for (0..nbas) |si| {
            const oi = data.shell_offsets[si];
            const ni = numCartesianFromBas(data, si);
            for (0..nbas) |sj| {
                const oj = data.shell_offsets[sj];
                const nj = numCartesianFromBas(data, sj);
                var max_p: f64 = 0.0;
                for (0..ni) |ii| {
                    for (0..nj) |jj| {
                        const abs_p = @abs(density[(oi + ii) * n + (oj + jj)]);
                        if (abs_p > max_p) max_p = abs_p;
                    }
                }
                d_max[si * nbas + sj] = max_p;
            }
        }

        // Main shell quartet loop with screening
        const max_buf_size = 50625;
        var buf: [max_buf_size]f64 = undefined;
        var shls: [4]c_int = undefined;

        for (0..nbas) |sa| {
            for (0..sa + 1) |sb| {
                const q_ab = schwarz[sa * nbas + sb];
                if (q_ab == 0.0) continue;

                const ab_pair = pairIndex(sa, sb);

                for (0..nbas) |sc| {
                    for (0..sc + 1) |sd| {
                        const cd_pair = pairIndex(sc, sd);
                        if (cd_pair > ab_pair) continue;

                        // Schwarz screening
                        const q_cd = schwarz[sc * nbas + sd];
                        const q_bound = q_ab * q_cd;
                        if (q_bound < schwarz_threshold) continue;

                        // Density-based screening
                        const d_max_abcd = @max(
                            @max(d_max[sa * nbas + sb], d_max[sc * nbas + sd]),
                            @max(
                                @max(d_max[sa * nbas + sc], d_max[sa * nbas + sd]),
                                @max(d_max[sb * nbas + sc], d_max[sb * nbas + sd]),
                            ),
                        );
                        if (d_max_abcd * q_bound < schwarz_threshold) continue;

                        shls[0] = @intCast(sa);
                        shls[1] = @intCast(sb);
                        shls[2] = @intCast(sc);
                        shls[3] = @intCast(sd);

                        const na = numCartesianFromBas(data, sa);
                        const nb = numCartesianFromBas(data, sb);
                        const nc = numCartesianFromBas(data, sc);
                        const nd = numCartesianFromBas(data, sd);

                        _ = cint_fns.int2e_cart(
                            &buf,
                            null,
                            &shls,
                            data.atm.ptr,
                            data.natm,
                            data.bas.ptr,
                            data.nbas,
                            data.env.ptr,
                            self.opt,
                            null,
                        );

                        const off_a = data.shell_offsets[sa];
                        const off_b = data.shell_offsets[sb];
                        const off_c = data.shell_offsets[sc];
                        const off_d = data.shell_offsets[sd];

                        const ab_same = (sa == sb);
                        const cd_same = (sc == sd);
                        const abcd_same = (ab_pair == cd_pair);

                        // Distribute ERIs with 8-fold symmetry
                        for (0..na) |ia| {
                            const mu = off_a + ia;
                            for (0..nb) |ib| {
                                const nu = off_b + ib;
                                for (0..nc) |ic| {
                                    const lam = off_c + ic;
                                    for (0..nd) |id_d| {
                                        const sig = off_d + id_d;

                                        const cidx = ia + ib * na + ic * na * nb + id_d * na * nb * nc;
                                        const eri = buf[cidx] * cart_norms[mu] * cart_norms[nu] * cart_norms[lam] * cart_norms[sig];

                                        // 1. (mu,nu | lam,sig)
                                        j_matrix[mu * n + nu] += density[lam * n + sig] * eri;
                                        k_matrix[mu * n + lam] += density[nu * n + sig] * eri;

                                        // 2. (nu,mu | lam,sig)
                                        if (!ab_same) {
                                            j_matrix[nu * n + mu] += density[lam * n + sig] * eri;
                                            k_matrix[nu * n + lam] += density[mu * n + sig] * eri;
                                        }

                                        // 3. (mu,nu | sig,lam)
                                        if (!cd_same) {
                                            j_matrix[mu * n + nu] += density[sig * n + lam] * eri;
                                            k_matrix[mu * n + sig] += density[nu * n + lam] * eri;
                                        }

                                        // 4. (nu,mu | sig,lam)
                                        if (!ab_same and !cd_same) {
                                            j_matrix[nu * n + mu] += density[sig * n + lam] * eri;
                                            k_matrix[nu * n + sig] += density[mu * n + lam] * eri;
                                        }

                                        // 5-8. bra-ket exchange
                                        if (!abcd_same) {
                                            j_matrix[lam * n + sig] += density[mu * n + nu] * eri;
                                            k_matrix[lam * n + mu] += density[sig * n + nu] * eri;

                                            if (!cd_same) {
                                                j_matrix[sig * n + lam] += density[mu * n + nu] * eri;
                                                k_matrix[sig * n + mu] += density[lam * n + nu] * eri;
                                            }

                                            if (!ab_same) {
                                                j_matrix[lam * n + sig] += density[nu * n + mu] * eri;
                                                k_matrix[lam * n + nu] += density[sig * n + mu] * eri;
                                            }

                                            if (!ab_same and !cd_same) {
                                                j_matrix[sig * n + lam] += density[nu * n + mu] * eri;
                                                k_matrix[sig * n + nu] += density[lam * n + mu] * eri;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return .{ .j_matrix = j_matrix, .k_matrix = k_matrix };
    }
};

/// Build J (Coulomb) and K (exchange) matrices using direct libcint ERIs.
///
/// This is the simple one-shot version (no pre-computed screening).
/// For SCF loops, prefer LibcintJKBuilder which reuses CINTOpt + Schwarz table.
pub fn buildJKDirect(
    alloc: std.mem.Allocator,
    data: LibcintData,
    density: []const f64,
) !JKResult {
    if (!enable_libcint) return error.LibcintNotAvailable;

    var builder = try LibcintJKBuilder.init(alloc, data, 1e-14);
    defer builder.deinit(alloc);
    return builder.buildJK(alloc, density);
}

/// Pair index for shell pairs: maps (a, b) with a >= b to a*(a+1)/2 + b.
fn pairIndex(a: usize, b: usize) usize {
    if (a >= b) return a * (a + 1) / 2 + b;
    return b * (b + 1) / 2 + a;
}

/// Get number of Cartesian functions for a shell in the libcint data.
fn numCartesianFromBas(data: LibcintData, shell_idx: usize) usize {
    return basis_mod.numCartesian(@intCast(data.bas[shell_idx * BAS_SLOTS + ANG_OF]));
}

// ============================================================================
// Tests
// ============================================================================

test "gtoNorm matches PySCF" {
    const testing = std.testing;

    // PySCF reference: gto.gto_norm(0, 3.42525091) = 6.36113067148303
    try testing.expectApproxEqAbs(6.36113067148303, gtoNorm(0, 3.42525091), 1e-8);

    // gto.gto_norm(0, 0.62391373) = 1.77361124
    try testing.expectApproxEqAbs(1.77361124, gtoNorm(0, 0.62391373), 1e-5);

    // gto.gto_norm(1, 1.0) = 2.917322170855303
    try testing.expectApproxEqAbs(2.917322170855303, gtoNorm(1, 1.0), 1e-8);

    // gto.gto_norm(2, 1.0) — for d-type
    // Gamma(3.5) = 15*sqrt(pi)/8
    // gaussian_int(6, 2) = Gamma(3.5)/(2 * 2^3.5) = 15*sqrt(pi)/(8*2*2^3.5)
    // = 15*1.7724539/(8*2*11.3137) = 15*1.7724539/181.019 = 0.14689
    // gto_norm(2,1) = 1/sqrt(0.14689) = 2.6087
    // Let's just validate l=0 and l=1 more precisely
}

test "gammaHalfInt" {
    const testing = std.testing;

    // Gamma(0.5) = sqrt(pi)
    try testing.expectApproxEqAbs(std.math.sqrt(std.math.pi), gammaHalfInt(0), 1e-12);

    // Gamma(1.5) = 0.5 * sqrt(pi)
    try testing.expectApproxEqAbs(0.5 * std.math.sqrt(std.math.pi), gammaHalfInt(1), 1e-12);

    // Gamma(2.5) = 1.5 * 0.5 * sqrt(pi) = 0.75 * sqrt(pi)
    try testing.expectApproxEqAbs(0.75 * std.math.sqrt(std.math.pi), gammaHalfInt(2), 1e-12);
}
