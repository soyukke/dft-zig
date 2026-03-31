const std = @import("std");
const nonlocal = @import("nonlocal.zig");
const pseudo = @import("pseudopotential.zig");
const config = @import("../config/config.zig");

const ctrapWeight = @import("../math/math.zig").radial.ctrapWeight;

/// Pre-computed lookup table for local form factor V(q).
/// Uses uniform grid + linear interpolation for O(1) evaluation.
pub const LocalFormFactorTable = struct {
    values: []f64,
    dq: f64,
    n_points: usize,

    const N_POINTS: usize = 4096;

    pub fn init(
        alloc: std.mem.Allocator,
        upf: pseudo.UpfData,
        z_valence: f64,
        mode: config.LocalPotentialMode,
        alpha: f64,
        q_max: f64,
    ) !LocalFormFactorTable {
        const n = N_POINTS;
        const dq = q_max / @as(f64, @floatFromInt(n - 1));
        const values = try alloc.alloc(f64, n);
        for (0..n) |i| {
            const q = @as(f64, @floatFromInt(i)) * dq;
            values[i] = switch (mode) {
                .tail => localVqWithTail(upf, z_valence, q),
                .ewald => localVqEwald(upf, z_valence, q, alpha),
                .short_range => localVqShortRange(upf, z_valence, q),
            };
        }
        return .{ .values = values, .dq = dq, .n_points = n };
    }

    pub fn eval(self: LocalFormFactorTable, q: f64) f64 {
        const idx_f = q / self.dq;
        const idx: usize = @intFromFloat(idx_f);
        if (idx >= self.n_points - 1) return self.values[self.n_points - 1];
        const t = idx_f - @as(f64, @floatFromInt(idx));
        return self.values[idx] * (1.0 - t) + self.values[idx + 1] * t;
    }

    /// Numerical derivative dV/dq using the table values.
    pub fn evalDeriv(self: LocalFormFactorTable, q: f64) f64 {
        if (q < self.dq) {
            return (self.eval(self.dq) - self.eval(0.0)) / self.dq;
        }
        return (self.eval(q + self.dq) - self.eval(q - self.dq)) / (2.0 * self.dq);
    }

    pub fn deinit(self: *LocalFormFactorTable, alloc: std.mem.Allocator) void {
        alloc.free(self.values);
    }
};

/// Pre-computed lookup table for a radial form factor (generic).
/// Uses uniform grid + linear interpolation for O(1) evaluation.
pub const RadialFormFactorTable = struct {
    values: []f64,
    dq: f64,
    n_points: usize,

    const N_POINTS: usize = 4096;

    pub fn initRhoAtom(alloc: std.mem.Allocator, upf: pseudo.UpfData, q_max: f64) !RadialFormFactorTable {
        if (upf.rho_atom.len == 0) return .{ .values = &[_]f64{}, .dq = 1.0, .n_points = 0 };
        return buildTable(alloc, upf, upf.rho_atom, q_max);
    }

    pub fn initRhoCore(alloc: std.mem.Allocator, upf: pseudo.UpfData, q_max: f64) !RadialFormFactorTable {
        if (upf.nlcc.len == 0) return .{ .values = &[_]f64{}, .dq = 1.0, .n_points = 0 };
        return buildTable(alloc, upf, upf.nlcc, q_max);
    }

    fn buildTable(alloc: std.mem.Allocator, upf: pseudo.UpfData, data: []const f64, q_max: f64) !RadialFormFactorTable {
        const n = N_POINTS;
        const dq = q_max / @as(f64, @floatFromInt(n - 1));
        const values = try alloc.alloc(f64, n);
        const nr = @min(data.len, @min(upf.r.len, upf.rab.len));
        for (0..n) |i| {
            const q = @as(f64, @floatFromInt(i)) * dq;
            var sum: f64 = 0.0;
            var j: usize = 0;
            while (j < nr) : (j += 1) {
                const x = q * upf.r[j];
                const j0 = nonlocal.sphericalBessel(0, x);
                sum += upf.r[j] * upf.r[j] * data[j] * j0 * upf.rab[j] * ctrapWeight(j, nr);
            }
            values[i] = 4.0 * std.math.pi * sum;
        }
        return .{ .values = values, .dq = dq, .n_points = n };
    }

    pub fn eval(self: RadialFormFactorTable, q: f64) f64 {
        if (self.n_points == 0) return 0.0;
        const idx_f = q / self.dq;
        const idx: usize = @intFromFloat(idx_f);
        if (idx >= self.n_points - 1) return self.values[self.n_points - 1];
        const t = idx_f - @as(f64, @floatFromInt(idx));
        return self.values[idx] * (1.0 - t) + self.values[idx + 1] * t;
    }

    /// Numerical derivative dF/dq using the table values.
    pub fn evalDeriv(self: RadialFormFactorTable, q: f64) f64 {
        if (self.n_points == 0) return 0.0;
        if (q < self.dq) {
            return (self.eval(self.dq) - self.eval(0.0)) / self.dq;
        }
        return (self.eval(q + self.dq) - self.eval(q - self.dq)) / (2.0 * self.dq);
    }

    pub fn deinit(self: *RadialFormFactorTable, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
    }
};

/// Compute the Ewald-compensated local pseudopotential form factor.
/// This uses erf(αr)/r screening to remove the Coulomb divergence.
///
/// V_comp(r) = V_local(r) + Z×erf(αr)/r
///
/// The Fourier transform is:
/// V_comp(q) = 4π ∫₀^∞ r² V_comp(r) sin(qr)/(qr) dr
///
/// The full form factor is:
/// V(q) = V_comp(q) - 4πZ/q² × exp(-q²/4α²)
///
/// The second term handles the long-range Coulomb consistently with Ewald.
/// This approach ensures numerical stability for all q values.
pub fn localVqEwald(upf: pseudo.UpfData, z_valence: f64, q: f64, alpha: f64) f64 {
    if (upf.r.len == 0) return 0.0;
    if (alpha <= 0.0) return localVq(upf, q);

    const n = upf.r.len;
    // Integrate V_comp(r) = V_local(r) + Z×erf(αr)/r
    var sum: f64 = 0.0;
    for (upf.r, 0..) |r, i| {
        const v_local = upf.v_local[i];
        const rab = upf.rab[i];

        // erf(αr)/r - handle r→0 limit: erf(αr)/r → 2α/√π
        const erf_term = if (r < 1e-10)
            z_valence * 2.0 * alpha / std.math.sqrt(std.math.pi)
        else
            z_valence * erfApprox(alpha * r) / r;

        const v_comp = v_local + erf_term;
        const x = q * r;
        const sinc = if (@abs(x) < 1e-12) 1.0 else std.math.sin(x) / x;
        sum += r * r * v_comp * sinc * rab * ctrapWeight(i, n);
    }
    const v_comp_q = 4.0 * std.math.pi * sum;

    // For q=0, the analytical Coulomb term diverges, return only the compensated part
    if (q < 1e-10) {
        return v_comp_q;
    }

    // Subtract the analytical long-range contribution: 4πZ/q² × exp(-q²/4α²)
    const exp_factor = std.math.exp(-q * q / (4.0 * alpha * alpha));
    const v_coulomb_analytical = 4.0 * std.math.pi * z_valence / (q * q) * exp_factor;
    return v_comp_q - v_coulomb_analytical;
}

/// Approximate error function using polynomial approximation.
fn erfApprox(x: f64) f64 {
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    const result = 1.0 - poly * std.math.exp(-ax * ax);
    return if (x >= 0.0) result else -result;
}

/// Compute local pseudopotential form factor using the short-range decomposition.
/// This is the ABINIT-style method that separates the short-range part from
/// the analytical Coulomb contribution.
///
/// In Rydberg units (UPF convention), the Coulomb potential is -2Z/r.
/// We decompose: V_loc(r) = V_SR(r) + (-2Z/r)
/// where V_SR(r) = V_loc(r) + 2Z/r is smooth and short-ranged.
///
/// The Fourier transform gives:
/// V_loc(q) = V_SR(q) - 8πZ/q²
///
/// where V_SR(q) = 4π ∫₀^rmax r² [V_loc(r) + 2Z/r] sinc(qr) dr
///
/// This method is more numerically stable than the tail correction because
/// the integrand V_SR(r) decays to zero at large r, unlike V_loc(r) which
/// contains the long-range Coulomb tail.
pub fn localVqShortRange(upf: pseudo.UpfData, z_valence: f64, q: f64) f64 {
    if (upf.r.len == 0) return 0.0;

    const n = upf.r.len;
    // Integrate the short-range part: V_loc(r) + 2Z/r (Rydberg units)
    var sum: f64 = 0.0;
    for (upf.r, 0..) |r, i| {
        const v_local = upf.v_local[i];
        const rab = upf.rab[i];
        // V_SR(r) = V_loc_Ry(r) + 2Z/r (cancels Rydberg Coulomb -2Z/r)
        // At r=0, both terms diverge but cancel; skip r≈0 point.
        const v_coulomb_r = if (r < 1e-10) 0.0 else 2.0 * z_valence / r;
        const v_sr = v_local + v_coulomb_r;
        const x = q * r;
        const sinc = if (@abs(x) < 1e-12) 1.0 else std.math.sin(x) / x;
        sum += r * r * v_sr * sinc * rab * ctrapWeight(i, n);
    }
    const v_sr_q = 4.0 * std.math.pi * sum;

    // For q≈0, the Coulomb term diverges; return only the SR part.
    // (G=0 is handled separately in ionicLocalPotential)
    if (q < 1e-10) {
        return v_sr_q;
    }

    // Subtract Coulomb FT: -8πZ/q² (Rydberg)
    const v_coulomb_q = -8.0 * std.math.pi * z_valence / (q * q);
    return v_sr_q + v_coulomb_q;
}

/// Compute epsatm: the q=0 limit of the short-range local form factor.
/// epsatm = 4π ∫₀^∞ r² [V_loc(r) + 2Z/r] dr  (Rydberg units)
/// This is the pseudopotential core energy per atom used in the total energy.
pub fn computeEpsatm(upf: pseudo.UpfData, z_valence: f64) f64 {
    return localVqShortRange(upf, z_valence, 0.0);
}

/// Compute local pseudopotential form factor V(q) with Coulomb subtraction.
/// The result is the "short-range" part: V_ps(q) - V_Coulomb(q)
/// where V_Coulomb(q) = -4πZ/q² for q>0.
/// This ensures the long-range Coulomb divergence cancels with Hartree.
/// NOTE: This uses a naive subtraction which has numerical issues for small q.
/// Prefer localVqShortRange for better numerical stability.
pub fn localVqWithZ(upf: pseudo.UpfData, z_valence: f64, q: f64) f64 {
    if (upf.r.len == 0) return 0.0;

    var sum: f64 = 0.0;
    for (upf.r, 0..) |r, i| {
        const v = upf.v_local[i];
        const rab = upf.rab[i];
        const x = q * r;
        const sinc = if (@abs(x) < 1e-12) 1.0 else std.math.sin(x) / x;
        sum += r * r * v * sinc * rab;
    }
    const v_ps = 4.0 * std.math.pi * sum;

    // For small q, the Coulomb term diverges but V_ps also diverges,
    // and their difference should be finite
    if (q < 1e-6) {
        // At q=0, return 0 (G=0 handled separately in ionicLocalPotential)
        return 0.0;
    }

    // Subtract long-range Coulomb: V_ps(q) - (-4πZ/q²) = V_ps(q) + 4πZ/q²
    const v_coulomb = -4.0 * std.math.pi * z_valence / (q * q);
    return v_ps - v_coulomb;
}

/// Compute local pseudopotential form factor with Coulomb tail correction.
/// The numerical integral is truncated at r_max, so we add the analytical
/// Coulomb tail contribution from r_max to infinity.
///
/// V(q) = V_numeric(q) + V_tail(q)
///
/// In Rydberg units (as used in UPF files), the Coulomb potential is -2Z/r.
/// The 3D Fourier transform is:
/// V_Coul(q) = -8πZ/q² (Rydberg)
///
/// For the truncated integral from r_max to infinity:
/// ∫_{r_max}^∞ sin(qr) dr = cos(qr_max)/q
///
/// So the tail contribution is:
/// V_tail(q) = -8πZ×cos(qr_max)/q² (Rydberg)
pub fn localVqWithTail(upf: pseudo.UpfData, z_valence: f64, q: f64) f64 {
    if (upf.r.len == 0) return 0.0;

    const n = upf.r.len;
    // Numerical integral from 0 to r_max
    var sum: f64 = 0.0;
    for (upf.r, 0..) |r, i| {
        const v = upf.v_local[i];
        const rab = upf.rab[i];
        const x = q * r;
        const sinc = if (@abs(x) < 1e-12) 1.0 else std.math.sin(x) / x;
        sum += r * r * v * sinc * rab * ctrapWeight(i, n);
    }
    const v_numeric = 4.0 * std.math.pi * sum;

    // For q ≈ 0, the tail integral diverges but is handled separately
    if (q < 1e-10) {
        return v_numeric;
    }

    // Add analytical Coulomb tail from r_max to infinity (Rydberg units)
    // V_tail = -8πZ × cos(qr_max) / q² (factor 2 for Ry vs Ha)
    const r_max = upf.r[upf.r.len - 1];
    const qr = q * r_max;
    const cos_qr = std.math.cos(qr);
    const v_tail = -8.0 * std.math.pi * z_valence * cos_qr / (q * q);

    return v_numeric + v_tail;
}

/// Legacy function for compatibility - returns raw V(q) without tail correction
/// NOTE: This gives incorrect results for small q due to missing Coulomb tail.
/// Use localVqWithTail for accurate form factors.
pub fn localVq(upf: pseudo.UpfData, q: f64) f64 {
    if (upf.r.len == 0) return 0.0;
    var sum: f64 = 0.0;
    for (upf.r, 0..) |r, i| {
        const v = upf.v_local[i];
        const rab = upf.rab[i];
        const x = q * r;
        const sinc = if (@abs(x) < 1e-12) 1.0 else std.math.sin(x) / x;
        sum += r * r * v * sinc * rab;
    }
    return 4.0 * std.math.pi * sum;
}

/// Compute atomic valence density form factor ρ_atom(G).
/// Uses ρ_atom(G) = 4π ∫ r² ρ_atom(r) j0(Gr) dr.
pub fn rhoAtomG(upf: pseudo.UpfData, g: f64) f64 {
    if (upf.rho_atom.len == 0) return 0.0;
    const n = @min(upf.rho_atom.len, @min(upf.r.len, upf.rab.len));
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const x = g * upf.r[i];
        const j0 = nonlocal.sphericalBessel(0, x);
        sum += upf.r[i] * upf.r[i] * upf.rho_atom[i] * j0 * upf.rab[i] * ctrapWeight(i, n);
    }
    return 4.0 * std.math.pi * sum;
}

/// Compute core density form factor ρ_core(G) from NLCC.
pub fn rhoCoreG(upf: pseudo.UpfData, g: f64) f64 {
    if (upf.nlcc.len == 0) return 0.0;
    const n = @min(upf.nlcc.len, @min(upf.r.len, upf.rab.len));
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const x = g * upf.r[i];
        const j0 = nonlocal.sphericalBessel(0, x);
        sum += upf.r[i] * upf.r[i] * upf.nlcc[i] * j0 * upf.rab[i] * ctrapWeight(i, n);
    }
    return 4.0 * std.math.pi * sum;
}
