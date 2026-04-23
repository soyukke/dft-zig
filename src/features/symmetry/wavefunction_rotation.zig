//! Wavefunction rotation under symmetry operations.
//!
//! Given wavefunctions at IBZ k-point, generate wavefunctions at S*k
//! by rotating PW coefficients.
//!
//! Without time reversal ({S|τ} only):
//!   c'[i] = exp(-i 2π (k_rot*G_i + Sk_unwrapped)·τ) × c[i]
//!
//! With time reversal ({S|τ} + TR):
//!   c'[i] = exp(+i 2π (k_rot*G_i + Sk_unwrapped)·τ) × conj(c[i])
//!
//! The target basis G-vectors are constructed externally by rotating the IBZ
//! G-vectors (with BZ folding delta), so the mapping is identity (same order).

const std = @import("std");
const math = @import("../math/math.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const symmetry = @import("symmetry.zig");

/// Compute per-G-vector phase factors for the translation-rotation symmetry operation.
fn compute_rotation_phases(
    basis_k: plane_wave.Basis,
    symop: symmetry.SymOp,
    sk_unwrapped: math.Vec3,
    time_reversal: bool,
    phases: []math.Complex,
) void {
    const n_pw = basis_k.gvecs.len;
    const tau = symop.trans;
    const has_translation = (@abs(tau.x) > 1e-12 or @abs(tau.y) > 1e-12 or @abs(tau.z) > 1e-12);

    if (!has_translation) {
        for (0..n_pw) |i| {
            phases[i] = math.complex.init(1.0, 0.0);
        }
        return;
    }

    const rot = symop.k_rot;
    const phase_sign: f64 = if (time_reversal) 1.0 else -1.0;

    for (basis_k.gvecs, 0..) |gv, i| {
        const h0 = gv.h;
        const k0 = gv.k;
        const l0 = gv.l;
        // G_rot = k_rot * G (fractional hkl, unnegated)
        const rm = rot.m;
        const gr0 = @as(f64, @floatFromInt(rm[0][0] * h0 + rm[0][1] * k0 + rm[0][2] * l0));
        const gr1 = @as(f64, @floatFromInt(rm[1][0] * h0 + rm[1][1] * k0 + rm[1][2] * l0));
        const gr2 = @as(f64, @floatFromInt(rm[2][0] * h0 + rm[2][1] * k0 + rm[2][2] * l0));

        // (k_rot*G + sk_unwrapped) · τ
        const dot = (gr0 + sk_unwrapped.x) * tau.x +
            (gr1 + sk_unwrapped.y) * tau.y +
            (gr2 + sk_unwrapped.z) * tau.z;
        const angle = phase_sign * 2.0 * std.math.pi * dot;
        phases[i] = math.complex.expi(angle);
    }
}

/// Rotate wavefunctions in-place (identity mapping: same G-vector order).
///
/// The target basis is constructed by rotating the IBZ basis G-vectors,
/// so no explicit mapping is needed — the coefficient at index i in the
/// IBZ basis maps to index i in the target basis.
///
/// sk_unwrapped: k_rot * k_ibz_frac (BEFORE BZ wrapping, used for phase).
pub fn rotate_wavefunctions_in_place(
    alloc: std.mem.Allocator,
    psi_k: []const []const math.Complex,
    basis_k: plane_wave.Basis,
    symop: symmetry.SymOp,
    sk_unwrapped: math.Vec3,
    time_reversal: bool,
) !struct { wfn: [][]math.Complex, wfn_const: [][]const math.Complex } {
    const n_occ = psi_k.len;
    const n_pw = basis_k.gvecs.len;

    // Phase = exp(phase_sign * i 2π (k_rot*G + sk_unwrapped)·τ)
    // Non-TR: phase_sign = -1
    // TR:     phase_sign = +1, and conjugate the input coefficient
    const phases = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(phases);

    compute_rotation_phases(basis_k, symop, sk_unwrapped, time_reversal, phases);

    const wfn = try alloc.alloc([]math.Complex, n_occ);
    const wfn_const = try alloc.alloc([]const math.Complex, n_occ);
    var bands_built: usize = 0;
    errdefer {
        for (0..bands_built) |b| alloc.free(wfn[b]);
        alloc.free(wfn);
        alloc.free(wfn_const);
    }

    for (0..n_occ) |n| {
        const psi_out = try alloc.alloc(math.Complex, n_pw);

        for (0..n_pw) |i| {
            if (time_reversal) {
                psi_out[i] = math.complex.mul(phases[i], math.complex.conj(psi_k[n][i]));
            } else {
                psi_out[i] = math.complex.mul(phases[i], psi_k[n][i]);
            }
        }

        wfn[n] = psi_out;
        wfn_const[n] = psi_out;
        bands_built = n + 1;
    }

    return .{ .wfn = wfn, .wfn_const = wfn_const };
}
