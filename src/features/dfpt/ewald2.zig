//! Second-order Ewald derivatives for the ion-ion contribution to the dynamical matrix.
//!
//! D^ewald_{Iα,Jβ}(q=0) = ∂²E_ion/∂u_{Iα}∂u_{Jβ}
//!
//! For q=0, this is the standard second derivative of the Ewald energy.

const std = @import("std");
const math = @import("../math/math.zig");
const c = @cImport({
    @cInclude("math.h");
});

/// Compute Ewald second derivatives (dynamical matrix contribution).
/// Returns a flat array of size (3*n_atoms)^2 in Hartree/bohr² units.
/// D[3*I+α, 3*J+β] is the second derivative with respect to u_{Iα} and u_{Jβ}.
///
/// Strategy: compute off-diagonal blocks (I≠J) directly, then set diagonal
/// blocks via the acoustic sum rule D_{Iα,Iβ} = -Σ_{J≠I} D_{Iα,Jβ}.
/// This is exact for the Ewald sum (translational invariance is built in).
pub fn ewaldDynmat(
    alloc: std.mem.Allocator,
    cell: math.Mat3,
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
) ![]f64 {
    const n = charges.len;
    const dim = 3 * n;
    const dynmat = try alloc.alloc(f64, dim * dim);
    @memset(dynmat, 0.0);

    const volume = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));

    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);

    // Ewald parameters
    const lmin = @min(@min(math.Vec3.norm(a1), math.Vec3.norm(a2)), math.Vec3.norm(a3));
    const alpha = 5.0 / lmin;
    const alpha2 = alpha * alpha;
    const tol = 1e-8;
    const rcut = std.math.sqrt(-@log(tol)) / alpha;
    const gcut = 2.0 * alpha * std.math.sqrt(-@log(tol));

    const nr1 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a1))));
    const nr2 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a2))));
    const nr3 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a3))));
    const ng1 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b1))));
    const ng2 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b2))));
    const ng3 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b3))));

    const pi = std.math.pi;
    const two_alpha_sqrtpi = 2.0 * alpha / std.math.sqrt(pi);

    // =========================================================================
    // Real-space contribution (off-diagonal only, I ≠ J)
    // =========================================================================
    // d²E_real/dR_{Iα}dR_{Jβ} = Z_I Z_J Σ_L d²[erfc(α|r|)/r]/dr_α dr_β
    // where r = R_I - R_J + L, and we use d²/dR_Iα dR_Jβ = -d²/dr_α dr_β
    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) continue;
            var n1: i32 = -nr1;
            while (n1 <= nr1) : (n1 += 1) {
                var n2: i32 = -nr2;
                while (n2 <= nr2) : (n2 += 1) {
                    var n3: i32 = -nr3;
                    while (n3 <= nr3) : (n3 += 1) {
                        const lvec = math.Vec3.add(
                            math.Vec3.add(
                                math.Vec3.scale(a1, @as(f64, @floatFromInt(n1))),
                                math.Vec3.scale(a2, @as(f64, @floatFromInt(n2))),
                            ),
                            math.Vec3.scale(a3, @as(f64, @floatFromInt(n3))),
                        );
                        const delta = math.Vec3.add(math.Vec3.sub(positions[i], positions[j]), lvec);
                        const r = math.Vec3.norm(delta);
                        if (r > rcut or r < 1e-12) continue;

                        const r2 = r * r;
                        const ar = alpha * r;
                        const erfc_ar = c.erfc(ar);
                        const exp_ar2 = std.math.exp(-alpha2 * r2);

                        // Second derivative of erfc(αr)/r:
                        // d²/dr_α dr_β [erfc(αr)/r]
                        // = δ_{αβ} × A + r_α r_β × B
                        // A = -[erfc(αr) + 2αr/√π exp(-α²r²)] / r³
                        // B = [3 erfc(αr) + 2αr/√π (3 + 2α²r²) exp(-α²r²)] / r⁵
                        const r_inv = 1.0 / r;
                        const r3_inv = r_inv * r_inv * r_inv;
                        const r5_inv = r3_inv * r_inv * r_inv;

                        const A = -(erfc_ar + two_alpha_sqrtpi * r * exp_ar2) * r3_inv;
                        const B = (3.0 * erfc_ar + two_alpha_sqrtpi * r * (3.0 + 2.0 * alpha2 * r2) * exp_ar2) * r5_inv;

                        const d = [3]f64{ delta.x, delta.y, delta.z };
                        const zz = charges[i] * charges[j];

                        for (0..3) |a| {
                            for (0..3) |b| {
                                const kronecker: f64 = if (a == b) 1.0 else 0.0;
                                const val = zz * (A * kronecker + B * d[a] * d[b]);
                                // D_{Iα,Jβ} = d²E/dR_Iα dR_Jβ = -d²f/dr_α dr_β (chain rule)
                                dynmat[(3 * i + a) * dim + (3 * j + b)] -= val;
                            }
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // Reciprocal-space contribution (off-diagonal only, I ≠ J)
    // =========================================================================
    // d²E_recip/dR_{Iα}dR_{Jβ} = (4π/V) Z_I Z_J Σ_{G≠0} exp(-G²/4α²)/G²
    //                              × G_α G_β × cos(G·(R_I-R_J))
    const four_pi_over_v = 4.0 * pi / volume;
    const inv_4alpha2 = 1.0 / (4.0 * alpha2);

    var h: i32 = -ng1;
    while (h <= ng1) : (h += 1) {
        var k: i32 = -ng2;
        while (k <= ng2) : (k += 1) {
            var l: i32 = -ng3;
            while (l <= ng3) : (l += 1) {
                if (h == 0 and k == 0 and l == 0) continue;
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(h))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(k))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(l))),
                );
                const g2val = math.Vec3.dot(gvec, gvec);
                if (g2val > gcut * gcut) continue;

                const factor = four_pi_over_v * std.math.exp(-g2val * inv_4alpha2) / g2val;
                const gcomp = [3]f64{ gvec.x, gvec.y, gvec.z };

                for (0..n) |i| {
                    for (0..n) |j| {
                        if (i == j) continue;
                        const dr = math.Vec3.sub(positions[i], positions[j]);
                        const gdotr = math.Vec3.dot(gvec, dr);
                        const cos_gr = std.math.cos(gdotr);
                        const zz = charges[i] * charges[j];

                        for (0..3) |a| {
                            for (0..3) |b| {
                                const val = zz * factor * gcomp[a] * gcomp[b] * cos_gr;
                                // d²E_recip/dR_Iα dR_Jβ for I≠J:
                                // = +(4π/V) Z_I Z_J Σ_G exp/G² × G_α G_β × cos(G·(R_I-R_J))
                                // (from differentiating S* w.r.t R_J and S w.r.t R_I)
                                dynmat[(3 * i + a) * dim + (3 * j + b)] += val;
                            }
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // Set diagonal blocks via acoustic sum rule:
    // D_{Iα,Iβ} = -Σ_{J≠I} D_{Iα,Jβ}
    // =========================================================================
    for (0..n) |i| {
        for (0..3) |a| {
            for (0..3) |b| {
                var sum: f64 = 0.0;
                for (0..n) |j| {
                    if (j == i) continue;
                    sum += dynmat[(3 * i + a) * dim + (3 * j + b)];
                }
                dynmat[(3 * i + a) * dim + (3 * i + b)] = -sum;
            }
        }
    }

    return dynmat;
}

/// Compute Ewald second derivatives at arbitrary q-point.
/// Returns a complex Hermitian matrix of size (3*n_atoms)^2 in Hartree/bohr² units.
///
/// Strategy:
/// - Off-diagonal (I≠J): real-space with exp(iq·L) + reciprocal G≠0 with G→G+q
/// - Diagonal (I=I): use q=0 ASR values (q-independent to machine precision
///   because L≠0 real-space contributions are exponentially suppressed by erfc)
/// - G=0 is excluded from both electronic perturbation and Ewald (macroscopic
///   field is handled separately via the non-analytic correction if needed)
pub fn ewaldDynmatQ(
    alloc: std.mem.Allocator,
    cell: math.Mat3,
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
    q_cart: math.Vec3,
) ![]math.Complex {
    const n = charges.len;
    const dim = 3 * n;
    const dynmat = try alloc.alloc(math.Complex, dim * dim);
    @memset(dynmat, math.complex.init(0.0, 0.0));

    const volume = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));

    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);

    // Ewald parameters
    const lmin = @min(@min(math.Vec3.norm(a1), math.Vec3.norm(a2)), math.Vec3.norm(a3));
    const alpha = 5.0 / lmin;
    const alpha2 = alpha * alpha;
    const tol = 1e-8;
    const rcut = std.math.sqrt(-@log(tol)) / alpha;
    const gcut = 2.0 * alpha * std.math.sqrt(-@log(tol));

    const nr1 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a1))));
    const nr2 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a2))));
    const nr3 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a3))));
    const ng1 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b1))));
    const ng2 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b2))));
    const ng3 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b3))));

    const pi = std.math.pi;
    const two_alpha_sqrtpi = 2.0 * alpha / std.math.sqrt(pi);

    // =========================================================================
    // Real-space contribution (off-diagonal I≠J only)
    // =========================================================================
    // D_{Iα,Jβ}(q) = Z_I Z_J Σ_L exp(iq·L) × d²[erfc(α|r|)/r]/dr_α dr_β
    // where r = R_I - R_J + L
    // I=J real-space terms (L≠0) are exponentially suppressed: erfc(α|L_min|) < 1e-12
    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) continue;
            var n1: i32 = -nr1;
            while (n1 <= nr1) : (n1 += 1) {
                var n2: i32 = -nr2;
                while (n2 <= nr2) : (n2 += 1) {
                    var n3: i32 = -nr3;
                    while (n3 <= nr3) : (n3 += 1) {
                        const lvec = math.Vec3.add(
                            math.Vec3.add(
                                math.Vec3.scale(a1, @as(f64, @floatFromInt(n1))),
                                math.Vec3.scale(a2, @as(f64, @floatFromInt(n2))),
                            ),
                            math.Vec3.scale(a3, @as(f64, @floatFromInt(n3))),
                        );
                        const delta = math.Vec3.add(math.Vec3.sub(positions[i], positions[j]), lvec);
                        const r = math.Vec3.norm(delta);
                        if (r > rcut or r < 1e-12) continue;

                        const r2 = r * r;
                        const ar = alpha * r;
                        const erfc_ar = c.erfc(ar);
                        const exp_ar2 = std.math.exp(-alpha2 * r2);

                        const r_inv = 1.0 / r;
                        const r3_inv = r_inv * r_inv * r_inv;
                        const r5_inv = r3_inv * r_inv * r_inv;

                        const A = -(erfc_ar + two_alpha_sqrtpi * r * exp_ar2) * r3_inv;
                        const B = (3.0 * erfc_ar + two_alpha_sqrtpi * r * (3.0 + 2.0 * alpha2 * r2) * exp_ar2) * r5_inv;

                        const d = [3]f64{ delta.x, delta.y, delta.z };
                        const zz = charges[i] * charges[j];

                        // Phase factor exp(iq·L)
                        const q_dot_l = math.Vec3.dot(q_cart, lvec);
                        const phase = math.complex.expi(q_dot_l);

                        for (0..3) |a| {
                            for (0..3) |b_idx| {
                                const kronecker: f64 = if (a == b_idx) 1.0 else 0.0;
                                const val = zz * (A * kronecker + B * d[a] * d[b_idx]);
                                dynmat[(3 * i + a) * dim + (3 * j + b_idx)] = math.complex.sub(
                                    dynmat[(3 * i + a) * dim + (3 * j + b_idx)],
                                    math.complex.scale(phase, val),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // Reciprocal-space contribution (all atom pairs including I=J)
    // =========================================================================
    // D_{Iα,Jβ}(q) = (4π/V) Z_I Z_J Σ_G exp(-|G+q|²/4α²)/|G+q|²
    //                × (G+q)_α(G+q)_β × exp(i(G+q)·(R_I-R_J))
    // Skip only |G+q|² ≈ 0.
    const four_pi_over_v = 4.0 * pi / volume;
    const inv_4alpha2 = 1.0 / (4.0 * alpha2);

    var h: i32 = -ng1;
    while (h <= ng1) : (h += 1) {
        var k: i32 = -ng2;
        while (k <= ng2) : (k += 1) {
            var l: i32 = -ng3;
            while (l <= ng3) : (l += 1) {
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(h))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(k))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(l))),
                );
                const gpq = math.Vec3.add(gvec, q_cart);
                const gpq2 = math.Vec3.dot(gpq, gpq);
                // Skip only if |G+q|²≈0 (not if G=0), since q≠0 means G=0 contributes
                if (gpq2 < 1e-12) continue;
                if (gpq2 > gcut * gcut) continue;

                const factor = four_pi_over_v * std.math.exp(-gpq2 * inv_4alpha2) / gpq2;
                const gpq_comp = [3]f64{ gpq.x, gpq.y, gpq.z };

                for (0..n) |i| {
                    for (0..n) |j| {
                        const dr = math.Vec3.sub(positions[i], positions[j]);
                        const gpq_dot_dr = math.Vec3.dot(gpq, dr);
                        const phase = math.complex.expi(gpq_dot_dr);
                        const zz = charges[i] * charges[j];

                        for (0..3) |a| {
                            for (0..3) |b_idx| {
                                const val = zz * factor * gpq_comp[a] * gpq_comp[b_idx];
                                dynmat[(3 * i + a) * dim + (3 * j + b_idx)] = math.complex.add(
                                    dynmat[(3 * i + a) * dim + (3 * j + b_idx)],
                                    math.complex.scale(phase, val),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // C-bar(q=0) diagonal correction: dyewq0
    // =========================================================================
    // ABINIT subtracts the q=0 row-sum from diagonal blocks:
    //   dyewq0(μ,ν,ia) = Σ_ib C-bar(q=0)(μ,ia; ν,ib)
    //   D_ewald(μ,ia; ν,ia) -= dyewq0(μ,ν,ia)
    //
    // C-bar(q=0) has two parts:
    //   (a) Reciprocal: (4π/V) Σ_{G≠0} exp(-G²/4α²)/G² × G_μ G_ν × cos(G·(R_ia-R_ib)) × Z_ia Z_ib
    //   (b) Real-space off-diagonal: same erfc terms as above but with q=0 (no phase)

    // dyewq0[ia][3*μ+ν] = Σ_ib contribution
    var dyewq0: [16][9]f64 = undefined; // up to 16 atoms
    for (0..n) |ia| {
        for (0..9) |idx| {
            dyewq0[ia][idx] = 0.0;
        }
    }

    // (a) Reciprocal-space q=0 contribution to dyewq0
    {
        var rh: i32 = -ng1;
        while (rh <= ng1) : (rh += 1) {
            var rk: i32 = -ng2;
            while (rk <= ng2) : (rk += 1) {
                var rl: i32 = -ng3;
                while (rl <= ng3) : (rl += 1) {
                    if (rh == 0 and rk == 0 and rl == 0) continue; // G=0 excluded at q=0
                    const gvec = math.Vec3.add(
                        math.Vec3.add(
                            math.Vec3.scale(b1, @as(f64, @floatFromInt(rh))),
                            math.Vec3.scale(b2, @as(f64, @floatFromInt(rk))),
                        ),
                        math.Vec3.scale(b3, @as(f64, @floatFromInt(rl))),
                    );
                    const g2val = math.Vec3.dot(gvec, gvec);
                    if (g2val > gcut * gcut) continue;

                    const factor_q0 = four_pi_over_v * std.math.exp(-g2val * inv_4alpha2) / g2val;
                    const gc = [3]f64{ gvec.x, gvec.y, gvec.z };

                    for (0..n) |ia| {
                        for (0..n) |ib| {
                            const dr = math.Vec3.sub(positions[ia], positions[ib]);
                            const gdotr = math.Vec3.dot(gvec, dr);
                            const cos_gr = std.math.cos(gdotr);
                            const zz = charges[ia] * charges[ib];
                            const w = zz * factor_q0 * cos_gr;

                            for (0..3) |mu| {
                                for (0..3) |nu| {
                                    dyewq0[ia][3 * mu + nu] += w * gc[mu] * gc[nu];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // (b) Real-space off-diagonal q=0 contribution to dyewq0
    for (0..n) |ia| {
        for (0..n) |ib| {
            if (ia == ib) continue;
            var rn1: i32 = -nr1;
            while (rn1 <= nr1) : (rn1 += 1) {
                var rn2: i32 = -nr2;
                while (rn2 <= nr2) : (rn2 += 1) {
                    var rn3: i32 = -nr3;
                    while (rn3 <= nr3) : (rn3 += 1) {
                        const lvec = math.Vec3.add(
                            math.Vec3.add(
                                math.Vec3.scale(a1, @as(f64, @floatFromInt(rn1))),
                                math.Vec3.scale(a2, @as(f64, @floatFromInt(rn2))),
                            ),
                            math.Vec3.scale(a3, @as(f64, @floatFromInt(rn3))),
                        );
                        const delta = math.Vec3.add(math.Vec3.sub(positions[ia], positions[ib]), lvec);
                        const r = math.Vec3.norm(delta);
                        if (r > rcut or r < 1e-12) continue;

                        const r2 = r * r;
                        const ar = alpha * r;
                        const erfc_ar = c.erfc(ar);
                        const exp_ar2 = std.math.exp(-alpha2 * r2);

                        const r_inv = 1.0 / r;
                        const r3_inv = r_inv * r_inv * r_inv;
                        const r5_inv = r3_inv * r_inv * r_inv;

                        const Aq0 = -(erfc_ar + two_alpha_sqrtpi * r * exp_ar2) * r3_inv;
                        const Bq0 = (3.0 * erfc_ar + two_alpha_sqrtpi * r * (3.0 + 2.0 * alpha2 * r2) * exp_ar2) * r5_inv;

                        const dd = [3]f64{ delta.x, delta.y, delta.z };
                        const zz = charges[ia] * charges[ib];

                        for (0..3) |mu| {
                            for (0..3) |nu| {
                                const kronecker: f64 = if (mu == nu) 1.0 else 0.0;
                                // Real-space D(ia,ib) = -Z_ia Z_ib (A δ + B r_μ r_ν)
                                const val = -zz * (Aq0 * kronecker + Bq0 * dd[mu] * dd[nu]);
                                dyewq0[ia][3 * mu + nu] += val;
                            }
                        }
                    }
                }
            }
        }
    }

    // Subtract dyewq0 from diagonal blocks of dynmat
    for (0..n) |ia| {
        for (0..3) |mu| {
            for (0..3) |nu| {
                dynmat[(3 * ia + mu) * dim + (3 * ia + nu)] = math.complex.sub(
                    dynmat[(3 * ia + mu) * dim + (3 * ia + nu)],
                    math.complex.init(dyewq0[ia][3 * mu + nu], 0.0),
                );
            }
        }
    }

    return dynmat;
}

// =========================================================================
// Tests
// =========================================================================

test "ewald dynmat finite difference" {
    const alloc = std.testing.allocator;
    const ewald = @import("../ewald/ewald.zig");

    const a = 5.431; // Bohr
    const cell = math.Mat3{ .m = .{
        .{ a, 0.0, 0.0 },
        .{ 0.0, a, 0.0 },
        .{ 0.0, 0.0, a },
    } };
    const recip_lat = math.Mat3{ .m = .{
        .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
        .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
        .{ 0.0, 0.0, 2.0 * std.math.pi / a },
    } };

    const charges = [_]f64{ 4.0, 4.0 };
    var positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = a / 4.0, .y = a / 4.0, .z = a / 4.0 },
    };

    const dynmat = try ewaldDynmat(alloc, cell, recip_lat, &charges, &positions);
    defer alloc.free(dynmat);

    // Finite difference of forces
    const delta: f64 = 1e-5;
    const n_atoms = charges.len;
    const dim = 3 * n_atoms;

    var max_rel_err: f64 = 0.0;
    for (0..n_atoms) |j| {
        for (0..3) |beta| {
            // Displace atom j in direction beta
            var pos_plus = positions;
            var pos_minus = positions;
            switch (beta) {
                0 => {
                    pos_plus[j].x += delta;
                    pos_minus[j].x -= delta;
                },
                1 => {
                    pos_plus[j].y += delta;
                    pos_minus[j].y -= delta;
                },
                2 => {
                    pos_plus[j].z += delta;
                    pos_minus[j].z -= delta;
                },
                else => {},
            }

            const forces_plus = try ewald.ionIonForces(alloc, cell, recip_lat, &charges, &pos_plus, null);
            defer alloc.free(forces_plus);
            const forces_minus = try ewald.ionIonForces(alloc, cell, recip_lat, &charges, &pos_minus, null);
            defer alloc.free(forces_minus);

            for (0..n_atoms) |i| {
                for (0..3) |alpha_idx| {
                    const f_plus = switch (alpha_idx) {
                        0 => forces_plus[i].x,
                        1 => forces_plus[i].y,
                        2 => forces_plus[i].z,
                        else => 0.0,
                    };
                    const f_minus = switch (alpha_idx) {
                        0 => forces_minus[i].x,
                        1 => forces_minus[i].y,
                        2 => forces_minus[i].z,
                        else => 0.0,
                    };
                    // D_{Iα,Jβ} = -dF_{Iα}/du_{Jβ} (force = -dE/dR)
                    const fd_val = -(f_plus - f_minus) / (2.0 * delta);
                    const analytical = dynmat[(3 * i + alpha_idx) * dim + (3 * j + beta)];
                    const abs_val = @abs(analytical);
                    if (abs_val > 1e-6) {
                        const err = @abs(analytical - fd_val) / abs_val;
                        if (err > max_rel_err) max_rel_err = err;
                    }
                }
            }
        }
    }

    try std.testing.expect(max_rel_err < 1e-3);
}

test "ewald dynmat symmetry" {
    const alloc = std.testing.allocator;

    const a = 5.431;
    const cell = math.Mat3{ .m = .{
        .{ a, 0.0, 0.0 },
        .{ 0.0, a, 0.0 },
        .{ 0.0, 0.0, a },
    } };
    const recip_lat = math.Mat3{ .m = .{
        .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
        .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
        .{ 0.0, 0.0, 2.0 * std.math.pi / a },
    } };

    const charges = [_]f64{ 4.0, 4.0 };
    const positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = a / 4.0, .y = a / 4.0, .z = a / 4.0 },
    };

    const dynmat_r = try ewaldDynmat(alloc, cell, recip_lat, &charges, &positions);
    defer alloc.free(dynmat_r);

    const dim = 6;
    // Check symmetry: D[i,j] = D[j,i]
    var max_asym: f64 = 0.0;
    for (0..dim) |i| {
        for (i + 1..dim) |j| {
            const diff = @abs(dynmat_r[i * dim + j] - dynmat_r[j * dim + i]);
            if (diff > max_asym) max_asym = diff;
        }
    }
    try std.testing.expect(max_asym < 1e-10);
}

test "ewald dynmatQ at L-point matches Python reference" {
    // FCC primitive cell for Si (a_bohr = 5.1315, same as verify_ewald_L.py)
    const alloc = std.testing.allocator;

    const a_bohr = 5.1315;
    // FCC primitive lattice vectors
    const cell = math.Mat3{ .m = .{
        .{ 0.0, a_bohr, a_bohr },
        .{ a_bohr, 0.0, a_bohr },
        .{ a_bohr, a_bohr, 0.0 },
    } };

    // Reciprocal lattice (2π convention)
    const pi = std.math.pi;
    const bp = pi / a_bohr;
    const recip_lat = math.Mat3{ .m = .{
        .{ -bp, bp, bp },
        .{ bp, -bp, bp },
        .{ bp, bp, -bp },
    } };

    const charges = [_]f64{ 4.0, 4.0 };
    const positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = a_bohr / 2.0, .y = a_bohr / 2.0, .z = a_bohr / 2.0 },
    };

    // L-point: q = 0.5*(b1+b2+b3) in Cartesian
    const q_L = math.Vec3{ .x = bp * 0.5, .y = bp * 0.5, .z = bp * 0.5 };

    const dynmat_q = try ewaldDynmatQ(alloc, cell, recip_lat, &charges, &positions, q_L);
    defer alloc.free(dynmat_q);

    const dim = 6;

    // Python reference values (Ha/bohr^2), after C-bar(q=0) diagonal correction:
    // D(0x,0x) = 3.936550 - 3.440557 = 0.495993
    // D(0x,0y) = 0.214034 (unchanged, dyewq0 off-diagonal is 0 for cubic)
    // D(0x,1x) ≈ 0, D(0x,1y) = -0.414259 (cross-atom blocks unaffected)
    const ref_d00_diag: f64 = 0.495993; // Ha/bohr^2
    const ref_d00_offdiag: f64 = 0.214034; // Ha/bohr^2
    const ref_d01_offdiag: f64 = -0.414259; // Ha/bohr^2

    // Check diagonal same-atom block: D(0x,0x), D(0y,0y), D(0z,0z) all equal
    const tol = 1e-4; // Ha/bohr^2 tolerance
    try std.testing.expectApproxEqAbs(dynmat_q[0 * dim + 0].r, ref_d00_diag, tol);
    try std.testing.expectApproxEqAbs(dynmat_q[1 * dim + 1].r, ref_d00_diag, tol);
    try std.testing.expectApproxEqAbs(dynmat_q[2 * dim + 2].r, ref_d00_diag, tol);

    // Check off-diagonal same-atom block: D(0x,0y) etc.
    try std.testing.expectApproxEqAbs(dynmat_q[0 * dim + 1].r, ref_d00_offdiag, tol);
    try std.testing.expectApproxEqAbs(dynmat_q[0 * dim + 2].r, ref_d00_offdiag, tol);
    try std.testing.expectApproxEqAbs(dynmat_q[1 * dim + 2].r, ref_d00_offdiag, tol);

    // Check cross-atom block: D(0x,1x) ≈ 0
    try std.testing.expectApproxEqAbs(dynmat_q[0 * dim + 3].r, 0.0, tol);

    // Check cross-atom off-diagonal: D(0x,1y) = -0.414259
    try std.testing.expectApproxEqAbs(dynmat_q[0 * dim + 4].r, ref_d01_offdiag, tol);
    try std.testing.expectApproxEqAbs(dynmat_q[0 * dim + 5].r, ref_d01_offdiag, tol);

    // Imaginary parts should be ~0 at L-point for this symmetric structure
    var max_imag: f64 = 0.0;
    for (0..dim) |ii| {
        for (0..dim) |jj| {
            const imag = @abs(dynmat_q[ii * dim + jj].i);
            if (imag > max_imag) max_imag = imag;
        }
    }
    try std.testing.expect(max_imag < 1e-6);
}

test "ewald dynmatQ Hermiticity" {
    const alloc = std.testing.allocator;

    const a = 5.431;
    const cell = math.Mat3{ .m = .{
        .{ a, 0.0, 0.0 },
        .{ 0.0, a, 0.0 },
        .{ 0.0, 0.0, a },
    } };
    const recip_lat = math.Mat3{ .m = .{
        .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
        .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
        .{ 0.0, 0.0, 2.0 * std.math.pi / a },
    } };

    const charges = [_]f64{ 4.0, 4.0 };
    const positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = a / 4.0, .y = a / 4.0, .z = a / 4.0 },
    };

    // Test at X-point: q = (π/a, 0, 0) = (0.5, 0, 0) in fractional
    const q_cart = math.Vec3{ .x = std.math.pi / a, .y = 0.0, .z = 0.0 };
    const dynmat_q = try ewaldDynmatQ(alloc, cell, recip_lat, &charges, &positions, q_cart);
    defer alloc.free(dynmat_q);

    const dim = 6;
    // D(q)_{iα,jβ} = conj(D(q)_{jβ,iα}) (Hermiticity)
    var max_asym: f64 = 0.0;
    for (0..dim) |i| {
        for (i..dim) |j| {
            const d_ij = dynmat_q[i * dim + j];
            const d_ji = dynmat_q[j * dim + i];
            const diff_r = @abs(d_ij.r - d_ji.r);
            const diff_i = @abs(d_ij.i + d_ji.i); // conj: imaginary parts should be opposite
            const diff = @max(diff_r, diff_i);
            if (diff > max_asym) max_asym = diff;
        }
    }
    try std.testing.expect(max_asym < 1e-10);
}
