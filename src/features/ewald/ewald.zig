const std = @import("std");
const math = @import("../math/math.zig");
const runtime_logging = @import("../runtime/logging.zig");
const c = @cImport({
    @cInclude("math.h");
});

pub const Params = struct {
    alpha: f64,
    rcut: f64,
    gcut: f64,
    tol: f64,
    quiet: bool,
};

/// Resolved Ewald setup (alpha, cutoffs, periodic image ranges) shared by
/// the energy/force/stress entry points.
const Setup = struct {
    alpha: f64,
    rcut: f64,
    gcut: f64,
    n: [3]i32,
    g: [3]i32,
};

fn resolveSetup(cell: math.Mat3, recip: math.Mat3, params: ?Params) Setup {
    const defaults = defaultParams(cell);
    const tol = if (params) |p| if (p.tol > 0.0) p.tol else defaults.tol else defaults.tol;
    const alpha = if (params) |p|
        if (p.alpha > 0.0) p.alpha else defaults.alpha
    else
        defaults.alpha;
    const rcut = if (params) |p| if (p.rcut > 0.0) p.rcut else defaults.rcut else defaults.rcut;
    const gcut = if (params) |p| if (p.gcut > 0.0) p.gcut else defaults.gcut else defaults.gcut;

    const auto = autoCuts(alpha, tol);
    const rcut_final = if (rcut > 0.0) rcut else auto.rcut;
    const gcut_final = if (gcut > 0.0) gcut else auto.gcut;

    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);

    return .{
        .alpha = alpha,
        .rcut = rcut_final,
        .gcut = gcut_final,
        .n = .{
            @as(i32, @intFromFloat(std.math.ceil(rcut_final / math.Vec3.norm(a1)))),
            @as(i32, @intFromFloat(std.math.ceil(rcut_final / math.Vec3.norm(a2)))),
            @as(i32, @intFromFloat(std.math.ceil(rcut_final / math.Vec3.norm(a3)))),
        },
        .g = .{
            @as(i32, @intFromFloat(std.math.ceil(gcut_final / math.Vec3.norm(b1)))),
            @as(i32, @intFromFloat(std.math.ceil(gcut_final / math.Vec3.norm(b2)))),
            @as(i32, @intFromFloat(std.math.ceil(gcut_final / math.Vec3.norm(b3)))),
        },
    };
}

fn maybeWarnNonNeutral(io: std.Io, params: ?Params, qsum: f64) !void {
    const quiet = if (params) |p| p.quiet else false;
    if (!quiet and @abs(qsum) > 1e-6) {
        const logger = runtime_logging.stderr(io, .warn);
        try logger.print(
            .warn,
            "ewald: non-neutral ionic charge {d:.6}," ++
                " applying neutralizing background\n",
            .{qsum},
        );
    }
}

/// Real-space lattice sum for the Ewald energy.
fn ewaldEnergyRealSum(
    cell: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    setup: Setup,
) f64 {
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    var real_sum: f64 = 0.0;
    var i: usize = 0;
    while (i < positions.len) : (i += 1) {
        var j: usize = 0;
        while (j < positions.len) : (j += 1) {
            var n1i: i32 = -setup.n[0];
            while (n1i <= setup.n[0]) : (n1i += 1) {
                var n2i: i32 = -setup.n[1];
                while (n2i <= setup.n[1]) : (n2i += 1) {
                    var n3i: i32 = -setup.n[2];
                    while (n3i <= setup.n[2]) : (n3i += 1) {
                        if (i == j and n1i == 0 and n2i == 0 and n3i == 0) continue;
                        const a1_scaled = math.Vec3.scale(a1, @as(f64, @floatFromInt(n1i)));
                        const a2_scaled = math.Vec3.scale(a2, @as(f64, @floatFromInt(n2i)));
                        const a3_scaled = math.Vec3.scale(a3, @as(f64, @floatFromInt(n3i)));
                        const rvec = math.Vec3.add(
                            math.Vec3.add(a1_scaled, a2_scaled),
                            a3_scaled,
                        );
                        const delta = math.Vec3.add(
                            math.Vec3.sub(positions[i], positions[j]),
                            rvec,
                        );
                        const r = math.Vec3.norm(delta);
                        if (r > setup.rcut or r <= 1e-12) continue;
                        real_sum += charges[i] * charges[j] * erfcValue(setup.alpha * r) / r;
                    }
                }
            }
        }
    }
    return real_sum;
}

/// Reciprocal-space lattice sum for the Ewald energy.
fn ewaldEnergyRecipSum(
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    setup: Setup,
) f64 {
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);
    var recip_sum: f64 = 0.0;
    var h: i32 = -setup.g[0];
    while (h <= setup.g[0]) : (h += 1) {
        var k: i32 = -setup.g[1];
        while (k <= setup.g[1]) : (k += 1) {
            var l: i32 = -setup.g[2];
            while (l <= setup.g[2]) : (l += 1) {
                if (h == 0 and k == 0 and l == 0) continue;
                const b1_scaled = math.Vec3.scale(b1, @as(f64, @floatFromInt(h)));
                const b2_scaled = math.Vec3.scale(b2, @as(f64, @floatFromInt(k)));
                const b3_scaled = math.Vec3.scale(b3, @as(f64, @floatFromInt(l)));
                const gvec = math.Vec3.add(
                    math.Vec3.add(b1_scaled, b2_scaled),
                    b3_scaled,
                );
                const g2val = math.Vec3.dot(gvec, gvec);
                if (g2val > setup.gcut * setup.gcut) continue;
                const factor = std.math.exp(-g2val / (4.0 * setup.alpha * setup.alpha)) / g2val;
                var sr: f64 = 0.0;
                var si: f64 = 0.0;
                var aidx: usize = 0;
                while (aidx < positions.len) : (aidx += 1) {
                    const phase = math.Vec3.dot(gvec, positions[aidx]);
                    sr += charges[aidx] * std.math.cos(phase);
                    si += charges[aidx] * std.math.sin(phase);
                }
                recip_sum += factor * (sr * sr + si * si);
            }
        }
    }
    return recip_sum;
}

/// Compute ion-ion Ewald energy for a periodic cell.
pub fn ionIonEnergy(
    io: std.Io,
    cell: math.Mat3,
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    params: ?Params,
) !f64 {
    if (charges.len != positions.len) return error.InvalidInput;
    if (charges.len == 0) return 0.0;

    const volume = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));
    if (volume <= 1e-12) return error.InvalidCell;

    var qsum: f64 = 0.0;
    for (charges) |q| {
        qsum += q;
    }

    try maybeWarnNonNeutral(io, params, qsum);

    const setup = resolveSetup(cell, recip, params);
    const alpha = setup.alpha;

    const real_sum = ewaldEnergyRealSum(cell, charges, positions, setup);
    const e_real = 0.5 * real_sum;

    const recip_sum = ewaldEnergyRecipSum(recip, charges, positions, setup);
    const e_recip = (2.0 * std.math.pi / volume) * recip_sum;

    var self_sum: f64 = 0.0;
    for (charges) |q| {
        self_sum += q * q;
    }
    const e_self = -alpha / std.math.sqrt(std.math.pi) * self_sum;

    const e_background = if (@abs(qsum) > 1e-6)
        -(std.math.pi / (2.0 * alpha * alpha * volume)) * qsum * qsum
    else
        0.0;

    return e_real + e_recip + e_self + e_background;
}

/// Build default Ewald parameters.
fn defaultParams(cell: math.Mat3) Params {
    const n0 = math.Vec3.norm(cell.row(0));
    const n1 = math.Vec3.norm(cell.row(1));
    const n2 = math.Vec3.norm(cell.row(2));
    const lmin = @min(@min(n0, n1), n2);
    const alpha = 5.0 / lmin;
    const tol = 1e-8;
    const cuts = autoCuts(alpha, tol);
    return Params{
        .alpha = alpha,
        .rcut = cuts.rcut,
        .gcut = cuts.gcut,
        .tol = tol,
        .quiet = false,
    };
}

const Cuts = struct {
    rcut: f64,
    gcut: f64,
};

/// Compute rcut/gcut from alpha and tolerance.
fn autoCuts(alpha: f64, tol: f64) Cuts {
    const rcut = std.math.sqrt(-@log(tol)) / alpha;
    const gcut = 2.0 * alpha * std.math.sqrt(-@log(tol));
    return Cuts{ .rcut = rcut, .gcut = gcut };
}

/// Complementary error function using std.math.erf.
fn erfcValue(x: f64) f64 {
    return c.erfc(x);
}

/// Accumulate real-space contribution to Ewald forces into `forces`.
/// Real-space forces
/// F_real(i) = -d/dR_i [Σ_{j,n} Z_i Z_j erfc(α|r|)/(2|r|)]
///           = Σ_{j,n} Z_i Z_j × [erfc(αr)/r² + 2α/√π × exp(-α²r²)/r] × (r/r)
/// where r = R_i - R_j + n
fn accumulateRealForces(
    cell: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    setup: Setup,
    forces: []math.Vec3,
) void {
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    const alpha = setup.alpha;
    const alpha_sq = alpha * alpha;
    const two_alpha_sqrtpi = 2.0 * alpha / std.math.sqrt(std.math.pi);
    const n_atoms = charges.len;

    var i: usize = 0;
    while (i < n_atoms) : (i += 1) {
        var j: usize = 0;
        while (j < n_atoms) : (j += 1) {
            var n1i: i32 = -setup.n[0];
            while (n1i <= setup.n[0]) : (n1i += 1) {
                var n2i: i32 = -setup.n[1];
                while (n2i <= setup.n[1]) : (n2i += 1) {
                    var n3i: i32 = -setup.n[2];
                    while (n3i <= setup.n[2]) : (n3i += 1) {
                        if (i == j and n1i == 0 and n2i == 0 and n3i == 0) continue;
                        const a1_scaled = math.Vec3.scale(a1, @as(f64, @floatFromInt(n1i)));
                        const a2_scaled = math.Vec3.scale(a2, @as(f64, @floatFromInt(n2i)));
                        const a3_scaled = math.Vec3.scale(a3, @as(f64, @floatFromInt(n3i)));
                        const rvec = math.Vec3.add(
                            math.Vec3.add(a1_scaled, a2_scaled),
                            a3_scaled,
                        );
                        // r = R_i - R_j + n (vector from j to i, including lattice translation)
                        const delta = math.Vec3.add(
                            math.Vec3.sub(positions[i], positions[j]),
                            rvec,
                        );
                        const r = math.Vec3.norm(delta);
                        if (r > setup.rcut or r <= 1e-12) continue;

                        const r_inv = 1.0 / r;
                        const ar = alpha * r;
                        const erfc_ar = erfcValue(ar);
                        const exp_ar2 = std.math.exp(-alpha_sq * r * r);

                        // Force magnitude:
                        //   Z_i Z_j × [erfc(αr)/r² + 2α/√π × exp(-α²r²)/r]
                        const bracket = erfc_ar * r_inv * r_inv +
                            two_alpha_sqrtpi * exp_ar2 * r_inv;
                        const force_mag = charges[i] * charges[j] * bracket;

                        // Force direction: r/|r| (pointing from j to i)
                        const force_vec = math.Vec3.scale(delta, force_mag * r_inv);

                        // Force on atom i (negative gradient of energy)
                        forces[i] = math.Vec3.add(forces[i], force_vec);
                    }
                }
            }
        }
    }
}

/// Accumulate reciprocal-space contribution to Ewald forces into `forces`.
/// E_recip = (2π/V) Σ_{G≠0} exp(-G²/4α²)/G² × |S(G)|²
/// where S(G) = Σ_j Z_j exp(i G·R_j)
/// F_recip(i) = -d/dR_i E_recip
///   = -(4π/V) × Z_i × Σ_{G≠0} [exp(-G²/4α²)/G²] × G
///       × Im[S(G)* exp(i G·R_i)]
///   = -(4π/V) × Z_i × Σ_{G≠0} [exp(-G²/4α²)/G²] × G
///       × [S_r sin(G·R_i) - S_i cos(G·R_i)]
fn accumulateRecipForces(
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    setup: Setup,
    volume: f64,
    forces: []math.Vec3,
) void {
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);
    const alpha = setup.alpha;
    const alpha_sq = alpha * alpha;
    const four_pi_over_v = 4.0 * std.math.pi / volume;
    const inv_4alpha2 = 1.0 / (4.0 * alpha_sq);
    const n_atoms = charges.len;

    var h: i32 = -setup.g[0];
    while (h <= setup.g[0]) : (h += 1) {
        var k: i32 = -setup.g[1];
        while (k <= setup.g[1]) : (k += 1) {
            var l: i32 = -setup.g[2];
            while (l <= setup.g[2]) : (l += 1) {
                if (h == 0 and k == 0 and l == 0) continue;
                const b1_scaled = math.Vec3.scale(b1, @as(f64, @floatFromInt(h)));
                const b2_scaled = math.Vec3.scale(b2, @as(f64, @floatFromInt(k)));
                const b3_scaled = math.Vec3.scale(b3, @as(f64, @floatFromInt(l)));
                const gvec = math.Vec3.add(
                    math.Vec3.add(b1_scaled, b2_scaled),
                    b3_scaled,
                );
                const g2val = math.Vec3.dot(gvec, gvec);
                if (g2val > setup.gcut * setup.gcut) continue;

                const factor = std.math.exp(-g2val * inv_4alpha2) / g2val;

                // Compute structure factor S(G) = Σ_j Z_j exp(i G·R_j)
                var sr: f64 = 0.0; // Real part
                var si: f64 = 0.0; // Imaginary part
                var aidx: usize = 0;
                while (aidx < n_atoms) : (aidx += 1) {
                    const phase = math.Vec3.dot(gvec, positions[aidx]);
                    sr += charges[aidx] * std.math.cos(phase);
                    si += charges[aidx] * std.math.sin(phase);
                }

                // Force on each atom
                var atom_idx: usize = 0;
                while (atom_idx < n_atoms) : (atom_idx += 1) {
                    const phase_i = math.Vec3.dot(gvec, positions[atom_idx]);
                    const cos_i = std.math.cos(phase_i);
                    const sin_i = std.math.sin(phase_i);

                    // Im[S(G)* exp(i G·R_i)] = S_r sin(G·R_i) - S_i cos(G·R_i)
                    const imag_part = sr * sin_i - si * cos_i;

                    // F = (4π/V) × Z_i × factor × G × imag_part
                    const force_factor = four_pi_over_v * charges[atom_idx] * factor * imag_part;
                    const force_vec = math.Vec3.scale(gvec, force_factor);

                    forces[atom_idx] = math.Vec3.add(forces[atom_idx], force_vec);
                }
            }
        }
    }
}

/// Compute ion-ion Ewald forces for a periodic cell.
/// Returns forces on each atom in Hartree/Bohr units.
/// (Multiply by 2 to convert to Rydberg/Bohr)
pub fn ionIonForces(
    alloc: std.mem.Allocator,
    cell: math.Mat3,
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    params: ?Params,
) ![]math.Vec3 {
    if (charges.len != positions.len) return error.InvalidInput;
    if (charges.len == 0) return &[_]math.Vec3{};

    const n_atoms = charges.len;
    const volume = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));
    if (volume <= 1e-12) return error.InvalidCell;

    const setup = resolveSetup(cell, recip, params);

    // Allocate force array
    const forces = try alloc.alloc(math.Vec3, n_atoms);
    for (forces) |*f| {
        f.* = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    }

    accumulateRealForces(cell, charges, positions, setup, forces);
    accumulateRecipForces(recip, charges, positions, setup, volume, forces);

    // Self-energy has no position dependence, so no force contribution
    // Background term also has no position dependence for uniform background

    return forces;
}

/// Real-space stress contribution:
/// σ^real_αβ = (1/2Ω) Σ_{i,j,n}' Z_i Z_j [erfc(αr)/r³ + (2α/√π)exp(-α²r²)/r²
///             + (4α³/√π)exp(-α²r²)] × r_α r_β
/// (the extra 4α³ term comes from the second derivative of erfc)
fn accumulateRealStress(
    cell: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    setup: Setup,
    sigma: *[3][3]f64,
) void {
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    const alpha = setup.alpha;
    const alpha_sq = alpha * alpha;
    const two_alpha_sqrtpi = 2.0 * alpha / std.math.sqrt(std.math.pi);

    var i: usize = 0;
    while (i < positions.len) : (i += 1) {
        var j: usize = 0;
        while (j < positions.len) : (j += 1) {
            var n1i: i32 = -setup.n[0];
            while (n1i <= setup.n[0]) : (n1i += 1) {
                var n2i: i32 = -setup.n[1];
                while (n2i <= setup.n[1]) : (n2i += 1) {
                    var n3i: i32 = -setup.n[2];
                    while (n3i <= setup.n[2]) : (n3i += 1) {
                        if (i == j and n1i == 0 and n2i == 0 and n3i == 0) continue;
                        const a1_scaled = math.Vec3.scale(a1, @as(f64, @floatFromInt(n1i)));
                        const a2_scaled = math.Vec3.scale(a2, @as(f64, @floatFromInt(n2i)));
                        const a3_scaled = math.Vec3.scale(a3, @as(f64, @floatFromInt(n3i)));
                        const rvec = math.Vec3.add(
                            math.Vec3.add(a1_scaled, a2_scaled),
                            a3_scaled,
                        );
                        const delta = math.Vec3.add(
                            math.Vec3.sub(positions[i], positions[j]),
                            rvec,
                        );
                        const r2 = math.Vec3.dot(delta, delta);
                        const r = @sqrt(r2);
                        if (r > setup.rcut or r <= 1e-12) continue;

                        const ar = alpha * r;
                        const erfc_ar = erfcValue(ar);
                        const exp_ar2 = std.math.exp(-alpha_sq * r2);
                        // d/dr [erfc(αr)/r] = -erfc(αr)/r² - (2α/√π)exp(-α²r²)/r
                        // σ_αβ += Z_iZ_j × [erfc(αr)/r³ + (2α/√π)exp(-α²r²)/r²
                        //                   + (4α³/√π)exp(-α²r²)] × r_αr_β
                        // Simplified: coeff × r_αr_β / r²
                        const bracket = erfc_ar / (r2 * r) +
                            two_alpha_sqrtpi * exp_ar2 / r2;
                        const coeff = charges[i] * charges[j] * bracket;

                        const rv = [3]f64{ delta.x, delta.y, delta.z };
                        for (0..3) |a| {
                            for (a..3) |b| {
                                sigma[a][b] -= 0.5 * coeff * rv[a] * rv[b];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Reciprocal-space stress contribution:
/// σ^recip_αβ = (2π/Ω²) Σ_{G≠0} |S(G)|² exp(-G²/4α²)/G² ×
///              [-δ_αβ + G_αG_β(1/(2α²) + 2/G²)]
fn accumulateRecipStress(
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    setup: Setup,
    inv_volume: f64,
    sigma: *[3][3]f64,
) void {
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);
    const alpha = setup.alpha;
    const alpha_sq = alpha * alpha;
    const two_pi = 2.0 * std.math.pi;
    const inv_4alpha2 = 1.0 / (4.0 * alpha_sq);

    var h: i32 = -setup.g[0];
    while (h <= setup.g[0]) : (h += 1) {
        var k: i32 = -setup.g[1];
        while (k <= setup.g[1]) : (k += 1) {
            var l: i32 = -setup.g[2];
            while (l <= setup.g[2]) : (l += 1) {
                if (h == 0 and k == 0 and l == 0) continue;
                const b1_scaled = math.Vec3.scale(b1, @as(f64, @floatFromInt(h)));
                const b2_scaled = math.Vec3.scale(b2, @as(f64, @floatFromInt(k)));
                const b3_scaled = math.Vec3.scale(b3, @as(f64, @floatFromInt(l)));
                const gvec = math.Vec3.add(
                    math.Vec3.add(b1_scaled, b2_scaled),
                    b3_scaled,
                );
                const g2val = math.Vec3.dot(gvec, gvec);
                if (g2val > setup.gcut * setup.gcut) continue;

                const factor = std.math.exp(-g2val * inv_4alpha2) / g2val;

                // Structure factor |S(G)|²
                var sr: f64 = 0.0;
                var si: f64 = 0.0;
                var aidx: usize = 0;
                while (aidx < positions.len) : (aidx += 1) {
                    const phase = math.Vec3.dot(gvec, positions[aidx]);
                    sr += charges[aidx] * std.math.cos(phase);
                    si += charges[aidx] * std.math.sin(phase);
                }
                const s2 = sr * sr + si * si;

                const prefactor = two_pi * inv_volume * factor * s2;
                const gv = [3]f64{ gvec.x, gvec.y, gvec.z };
                const gg_factor = inv_4alpha2 + 1.0 / g2val; // 1/(4α²) + 1/G²

                for (0..3) |a| {
                    for (a..3) |b| {
                        const diag: f64 = if (a == b) 1.0 else 0.0;
                        const tensor_term = 2.0 * gv[a] * gv[b] * gg_factor;
                        sigma[a][b] += prefactor * (-diag + tensor_term);
                    }
                }
            }
        }
    }
}

/// Compute Ewald stress tensor for a periodic cell.
/// Returns stress in Hartree/Bohr³ units (multiply by 2 for Ry/Bohr³).
/// σ_αβ = (1/Ω) ∂E_ewald/∂ε_αβ
pub fn ionIonStress(
    cell: math.Mat3,
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    params: ?Params,
) ![3][3]f64 {
    if (charges.len != positions.len) return error.InvalidInput;

    var sigma = [3][3]f64{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    if (charges.len == 0) return sigma;

    const volume = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));
    if (volume <= 1e-12) return error.InvalidCell;
    const inv_volume = 1.0 / volume;

    var qsum: f64 = 0.0;
    for (charges) |q| qsum += q;

    const setup = resolveSetup(cell, recip, params);
    const alpha_sq = setup.alpha * setup.alpha;

    accumulateRealStress(cell, charges, positions, setup, &sigma);
    accumulateRecipStress(recip, charges, positions, setup, inv_volume, &sigma);

    // Background stress (for non-neutral systems):
    // dE_bg/dε_αβ = δ_αβ × π Q² / (2α² Ω)
    if (@abs(qsum) > 1e-6) {
        const bg = std.math.pi * qsum * qsum / (2.0 * alpha_sq * volume);
        for (0..3) |a| sigma[a][a] += bg;
    }

    // Convert from dE/dε to stress σ = (1/Ω) dE/dε
    for (0..3) |a| {
        for (0..3) |b| {
            sigma[a][b] *= inv_volume;
        }
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| {
            sigma[b][a] = sigma[a][b];
        }
    }

    return sigma;
}

test "ewald forces finite difference" {
    const io = std.testing.io;
    const testing = std.testing;
    const alloc = testing.allocator;

    // Simple 2-atom system: NaCl-like
    const a = 5.0; // Bohr
    const cell = math.Mat3{
        .m = .{
            .{ a, 0.0, 0.0 },
            .{ 0.0, a, 0.0 },
            .{ 0.0, 0.0, a },
        },
    };
    const recip = math.Mat3{
        .m = .{
            .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
            .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
            .{ 0.0, 0.0, 2.0 * std.math.pi / a },
        },
    };

    const charges = [_]f64{ 1.0, -1.0 };
    var positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = a / 2.0, .y = a / 2.0, .z = a / 2.0 },
    };

    // Compute analytic forces
    const forces = try ionIonForces(alloc, cell, recip, &charges, &positions, null);
    defer alloc.free(forces);

    // Verify with finite difference
    const delta = 1e-5;
    for (0..2) |atom_idx| {
        // Test x component
        var pos_plus = positions;
        var pos_minus = positions;
        pos_plus[atom_idx].x += delta;
        pos_minus[atom_idx].x -= delta;

        const e_plus = try ionIonEnergy(io, cell, recip, &charges, &pos_plus, null);
        const e_minus = try ionIonEnergy(io, cell, recip, &charges, &pos_minus, null);
        const f_numeric_x = -(e_plus - e_minus) / (2.0 * delta);

        try testing.expectApproxEqAbs(forces[atom_idx].x, f_numeric_x, 1e-5);

        // Test y component
        pos_plus = positions;
        pos_minus = positions;
        pos_plus[atom_idx].y += delta;
        pos_minus[atom_idx].y -= delta;

        const e_plus_y = try ionIonEnergy(io, cell, recip, &charges, &pos_plus, null);
        const e_minus_y = try ionIonEnergy(io, cell, recip, &charges, &pos_minus, null);
        const f_numeric_y = -(e_plus_y - e_minus_y) / (2.0 * delta);

        try testing.expectApproxEqAbs(forces[atom_idx].y, f_numeric_y, 1e-5);

        // Test z component
        pos_plus = positions;
        pos_minus = positions;
        pos_plus[atom_idx].z += delta;
        pos_minus[atom_idx].z -= delta;

        const e_plus_z = try ionIonEnergy(io, cell, recip, &charges, &pos_plus, null);
        const e_minus_z = try ionIonEnergy(io, cell, recip, &charges, &pos_minus, null);
        const f_numeric_z = -(e_plus_z - e_minus_z) / (2.0 * delta);

        try testing.expectApproxEqAbs(forces[atom_idx].z, f_numeric_z, 1e-5);
    }
}

test "ewald forces sum to zero" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // For a periodic system, total force should be zero (Newton's 3rd law)
    const a = 5.431; // Silicon lattice constant
    const cell = math.Mat3{
        .m = .{
            .{ a, 0.0, 0.0 },
            .{ 0.0, a, 0.0 },
            .{ 0.0, 0.0, a },
        },
    };
    const recip = math.Mat3{
        .m = .{
            .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
            .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
            .{ 0.0, 0.0, 2.0 * std.math.pi / a },
        },
    };

    // Diamond structure Si (8 atoms)
    const charges = [_]f64{ 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0 };
    const positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.0, .y = a / 2.0, .z = a / 2.0 },
        math.Vec3{ .x = a / 2.0, .y = 0.0, .z = a / 2.0 },
        math.Vec3{ .x = a / 2.0, .y = a / 2.0, .z = 0.0 },
        math.Vec3{ .x = a / 4.0, .y = a / 4.0, .z = a / 4.0 },
        math.Vec3{ .x = a / 4.0, .y = 3.0 * a / 4.0, .z = 3.0 * a / 4.0 },
        math.Vec3{ .x = 3.0 * a / 4.0, .y = a / 4.0, .z = 3.0 * a / 4.0 },
        math.Vec3{ .x = 3.0 * a / 4.0, .y = 3.0 * a / 4.0, .z = a / 4.0 },
    };

    const forces = try ionIonForces(alloc, cell, recip, &charges, &positions, null);
    defer alloc.free(forces);

    // Sum of all forces should be zero
    var total_force = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    for (forces) |f| {
        total_force = math.Vec3.add(total_force, f);
    }

    try testing.expectApproxEqAbs(total_force.x, 0.0, 1e-10);
    try testing.expectApproxEqAbs(total_force.y, 0.0, 1e-10);
    try testing.expectApproxEqAbs(total_force.z, 0.0, 1e-10);
}
