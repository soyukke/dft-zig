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

const RealSpaceTensor = [3][3]f64;
const Dyewq0Block = [9]f64;

const EwaldContext = struct {
    n: usize,
    dim: usize,
    volume: f64,
    a1: math.Vec3,
    a2: math.Vec3,
    a3: math.Vec3,
    b1: math.Vec3,
    b2: math.Vec3,
    b3: math.Vec3,
    alpha: f64,
    alpha2: f64,
    rcut: f64,
    gcut: f64,
    nr1: i32,
    nr2: i32,
    nr3: i32,
    ng1: i32,
    ng2: i32,
    ng3: i32,
    two_alpha_sqrtpi: f64,
    four_pi_over_v: f64,
    inv_4alpha2: f64,
};

fn initEwaldContext(
    cell: math.Mat3,
    recip: math.Mat3,
    n: usize,
) EwaldContext {
    const volume = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);

    const lmin = @min(@min(math.Vec3.norm(a1), math.Vec3.norm(a2)), math.Vec3.norm(a3));
    const alpha = 5.0 / lmin;
    const alpha2 = alpha * alpha;
    const tol = 1e-8;
    const rcut = std.math.sqrt(-@log(tol)) / alpha;
    const gcut = 2.0 * alpha * std.math.sqrt(-@log(tol));

    return .{
        .n = n,
        .dim = 3 * n,
        .volume = volume,
        .a1 = a1,
        .a2 = a2,
        .a3 = a3,
        .b1 = b1,
        .b2 = b2,
        .b3 = b3,
        .alpha = alpha,
        .alpha2 = alpha2,
        .rcut = rcut,
        .gcut = gcut,
        .nr1 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a1)))),
        .nr2 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a2)))),
        .nr3 = @as(i32, @intFromFloat(std.math.ceil(rcut / math.Vec3.norm(a3)))),
        .ng1 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b1)))),
        .ng2 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b2)))),
        .ng3 = @as(i32, @intFromFloat(std.math.ceil(gcut / math.Vec3.norm(b3)))),
        .two_alpha_sqrtpi = 2.0 * alpha / std.math.sqrt(std.math.pi),
        .four_pi_over_v = 4.0 * std.math.pi / volume,
        .inv_4alpha2 = 1.0 / (4.0 * alpha2),
    };
}

fn latticeVector(
    ctx: *const EwaldContext,
    n1: i32,
    n2: i32,
    n3: i32,
) math.Vec3 {
    return math.Vec3.add(
        math.Vec3.add(
            math.Vec3.scale(ctx.a1, @as(f64, @floatFromInt(n1))),
            math.Vec3.scale(ctx.a2, @as(f64, @floatFromInt(n2))),
        ),
        math.Vec3.scale(ctx.a3, @as(f64, @floatFromInt(n3))),
    );
}

fn reciprocalVector(
    ctx: *const EwaldContext,
    h: i32,
    k: i32,
    l: i32,
) math.Vec3 {
    return math.Vec3.add(
        math.Vec3.add(
            math.Vec3.scale(ctx.b1, @as(f64, @floatFromInt(h))),
            math.Vec3.scale(ctx.b2, @as(f64, @floatFromInt(k))),
        ),
        math.Vec3.scale(ctx.b3, @as(f64, @floatFromInt(l))),
    );
}

fn realSpaceSecondDerivativeTensor(
    ctx: *const EwaldContext,
    delta: math.Vec3,
) ?RealSpaceTensor {
    const r = math.Vec3.norm(delta);
    if (r > ctx.rcut or r < 1e-12) return null;

    const r2 = r * r;
    const ar = ctx.alpha * r;
    const erfc_ar = c.erfc(ar);
    const exp_ar2 = std.math.exp(-ctx.alpha2 * r2);
    const r_inv = 1.0 / r;
    const r3_inv = r_inv * r_inv * r_inv;
    const r5_inv = r3_inv * r_inv * r_inv;
    const a_term = -(erfc_ar + ctx.two_alpha_sqrtpi * r * exp_ar2) * r3_inv;
    const b_poly = 3.0 + 2.0 * ctx.alpha2 * r2;
    const b_num = 3.0 * erfc_ar + ctx.two_alpha_sqrtpi * r * b_poly * exp_ar2;
    const b_term = b_num * r5_inv;

    const d = [3]f64{ delta.x, delta.y, delta.z };
    var tensor: RealSpaceTensor = undefined;
    for (0..3) |a| {
        for (0..3) |b| {
            const kronecker: f64 = if (a == b) 1.0 else 0.0;
            tensor[a][b] = a_term * kronecker + b_term * d[a] * d[b];
        }
    }
    return tensor;
}

fn dynmatIndex(
    ctx: *const EwaldContext,
    i: usize,
    a: usize,
    j: usize,
    b: usize,
) usize {
    return (3 * i + a) * ctx.dim + (3 * j + b);
}

fn dyewq0Index(mu: usize, nu: usize) usize {
    return 3 * mu + nu;
}

fn addRealSpaceOffDiagonal(
    ctx: *const EwaldContext,
    dynmat: []f64,
    charges: []const f64,
    positions: []const math.Vec3,
) void {
    for (0..ctx.n) |i| {
        for (0..ctx.n) |j| {
            if (i == j) continue;
            var n1: i32 = -ctx.nr1;
            while (n1 <= ctx.nr1) : (n1 += 1) {
                var n2: i32 = -ctx.nr2;
                while (n2 <= ctx.nr2) : (n2 += 1) {
                    var n3: i32 = -ctx.nr3;
                    while (n3 <= ctx.nr3) : (n3 += 1) {
                        const delta = math.Vec3.add(
                            math.Vec3.sub(positions[i], positions[j]),
                            latticeVector(ctx, n1, n2, n3),
                        );
                        const tensor = realSpaceSecondDerivativeTensor(ctx, delta) orelse continue;
                        const zz = charges[i] * charges[j];
                        for (0..3) |a| {
                            for (0..3) |b| {
                                dynmat[dynmatIndex(ctx, i, a, j, b)] -= zz * tensor[a][b];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn addReciprocalOffDiagonal(
    ctx: *const EwaldContext,
    dynmat: []f64,
    charges: []const f64,
    positions: []const math.Vec3,
) void {
    var h: i32 = -ctx.ng1;
    while (h <= ctx.ng1) : (h += 1) {
        var k: i32 = -ctx.ng2;
        while (k <= ctx.ng2) : (k += 1) {
            var l: i32 = -ctx.ng3;
            while (l <= ctx.ng3) : (l += 1) {
                if (h == 0 and k == 0 and l == 0) continue;
                const gvec = reciprocalVector(ctx, h, k, l);
                const g2val = math.Vec3.dot(gvec, gvec);
                if (g2val > ctx.gcut * ctx.gcut) continue;

                const factor = ctx.four_pi_over_v * std.math.exp(-g2val * ctx.inv_4alpha2) / g2val;
                const gcomp = [3]f64{ gvec.x, gvec.y, gvec.z };
                for (0..ctx.n) |i| {
                    for (0..ctx.n) |j| {
                        if (i == j) continue;
                        const cos_gr = std.math.cos(
                            math.Vec3.dot(gvec, math.Vec3.sub(positions[i], positions[j])),
                        );
                        const zz = charges[i] * charges[j];
                        for (0..3) |a| {
                            for (0..3) |b| {
                                dynmat[dynmatIndex(ctx, i, a, j, b)] +=
                                    zz * factor * gcomp[a] * gcomp[b] * cos_gr;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn applyAcousticSumRule(
    ctx: *const EwaldContext,
    dynmat: []f64,
) void {
    for (0..ctx.n) |i| {
        for (0..3) |a| {
            for (0..3) |b| {
                var sum: f64 = 0.0;
                for (0..ctx.n) |j| {
                    if (j == i) continue;
                    sum += dynmat[dynmatIndex(ctx, i, a, j, b)];
                }
                dynmat[dynmatIndex(ctx, i, a, i, b)] = -sum;
            }
        }
    }
}

fn addRealSpaceOffDiagonalQ(
    ctx: *const EwaldContext,
    dynmat: []math.Complex,
    charges: []const f64,
    positions: []const math.Vec3,
    q_cart: math.Vec3,
) void {
    for (0..ctx.n) |i| {
        for (0..ctx.n) |j| {
            if (i == j) continue;
            var n1: i32 = -ctx.nr1;
            while (n1 <= ctx.nr1) : (n1 += 1) {
                var n2: i32 = -ctx.nr2;
                while (n2 <= ctx.nr2) : (n2 += 1) {
                    var n3: i32 = -ctx.nr3;
                    while (n3 <= ctx.nr3) : (n3 += 1) {
                        const lvec = latticeVector(ctx, n1, n2, n3);
                        const delta = math.Vec3.add(
                            math.Vec3.sub(positions[i], positions[j]),
                            lvec,
                        );
                        const tensor = realSpaceSecondDerivativeTensor(ctx, delta) orelse continue;
                        const phase = math.complex.expi(math.Vec3.dot(q_cart, lvec));
                        const zz = charges[i] * charges[j];
                        for (0..3) |a| {
                            for (0..3) |b| {
                                const idx = dynmatIndex(ctx, i, a, j, b);
                                dynmat[idx] = math.complex.sub(
                                    dynmat[idx],
                                    math.complex.scale(phase, zz * tensor[a][b]),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn addReciprocalContributionQ(
    ctx: *const EwaldContext,
    dynmat: []math.Complex,
    charges: []const f64,
    positions: []const math.Vec3,
    q_cart: math.Vec3,
) void {
    var h: i32 = -ctx.ng1;
    while (h <= ctx.ng1) : (h += 1) {
        var k: i32 = -ctx.ng2;
        while (k <= ctx.ng2) : (k += 1) {
            var l: i32 = -ctx.ng3;
            while (l <= ctx.ng3) : (l += 1) {
                const gvec = reciprocalVector(ctx, h, k, l);
                const gpq = math.Vec3.add(gvec, q_cart);
                const gpq2 = math.Vec3.dot(gpq, gpq);
                if (gpq2 < 1e-12 or gpq2 > ctx.gcut * ctx.gcut) continue;

                const factor = ctx.four_pi_over_v * std.math.exp(-gpq2 * ctx.inv_4alpha2) / gpq2;
                const gpq_comp = [3]f64{ gpq.x, gpq.y, gpq.z };
                for (0..ctx.n) |i| {
                    for (0..ctx.n) |j| {
                        const phase = math.complex.expi(
                            math.Vec3.dot(gpq, math.Vec3.sub(positions[i], positions[j])),
                        );
                        const zz = charges[i] * charges[j];
                        for (0..3) |a| {
                            for (0..3) |b| {
                                const idx = dynmatIndex(ctx, i, a, j, b);
                                dynmat[idx] = math.complex.add(
                                    dynmat[idx],
                                    math.complex.scale(
                                        phase,
                                        zz * factor * gpq_comp[a] * gpq_comp[b],
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn accumulateDyewq0Reciprocal(
    ctx: *const EwaldContext,
    dyewq0: []Dyewq0Block,
    charges: []const f64,
    positions: []const math.Vec3,
) void {
    var h: i32 = -ctx.ng1;
    while (h <= ctx.ng1) : (h += 1) {
        var k: i32 = -ctx.ng2;
        while (k <= ctx.ng2) : (k += 1) {
            var l: i32 = -ctx.ng3;
            while (l <= ctx.ng3) : (l += 1) {
                if (h == 0 and k == 0 and l == 0) continue;
                const gvec = reciprocalVector(ctx, h, k, l);
                const g2val = math.Vec3.dot(gvec, gvec);
                if (g2val > ctx.gcut * ctx.gcut) continue;

                const factor = ctx.four_pi_over_v * std.math.exp(-g2val * ctx.inv_4alpha2) / g2val;
                const gc = [3]f64{ gvec.x, gvec.y, gvec.z };
                for (0..ctx.n) |ia| {
                    for (0..ctx.n) |ib| {
                        const cos_gr = std.math.cos(
                            math.Vec3.dot(gvec, math.Vec3.sub(positions[ia], positions[ib])),
                        );
                        const w = charges[ia] * charges[ib] * factor * cos_gr;
                        for (0..3) |mu| {
                            for (0..3) |nu| {
                                dyewq0[ia][dyewq0Index(mu, nu)] += w * gc[mu] * gc[nu];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn accumulateDyewq0Real(
    ctx: *const EwaldContext,
    dyewq0: []Dyewq0Block,
    charges: []const f64,
    positions: []const math.Vec3,
) void {
    for (0..ctx.n) |ia| {
        for (0..ctx.n) |ib| {
            if (ia == ib) continue;
            var n1: i32 = -ctx.nr1;
            while (n1 <= ctx.nr1) : (n1 += 1) {
                var n2: i32 = -ctx.nr2;
                while (n2 <= ctx.nr2) : (n2 += 1) {
                    var n3: i32 = -ctx.nr3;
                    while (n3 <= ctx.nr3) : (n3 += 1) {
                        const delta = math.Vec3.add(
                            math.Vec3.sub(positions[ia], positions[ib]),
                            latticeVector(ctx, n1, n2, n3),
                        );
                        const tensor = realSpaceSecondDerivativeTensor(ctx, delta) orelse continue;
                        const zz = charges[ia] * charges[ib];
                        for (0..3) |mu| {
                            for (0..3) |nu| {
                                dyewq0[ia][dyewq0Index(mu, nu)] -= zz * tensor[mu][nu];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn buildDyewq0(
    alloc: std.mem.Allocator,
    ctx: *const EwaldContext,
    charges: []const f64,
    positions: []const math.Vec3,
) ![]Dyewq0Block {
    const dyewq0 = try alloc.alloc(Dyewq0Block, ctx.n);
    @memset(dyewq0, [_]f64{0.0} ** 9);
    accumulateDyewq0Reciprocal(ctx, dyewq0, charges, positions);
    accumulateDyewq0Real(ctx, dyewq0, charges, positions);
    return dyewq0;
}

fn applyDyewq0DiagonalCorrection(
    ctx: *const EwaldContext,
    dynmat: []math.Complex,
    dyewq0: []const Dyewq0Block,
) void {
    for (0..ctx.n) |ia| {
        for (0..3) |mu| {
            for (0..3) |nu| {
                const idx = dynmatIndex(ctx, ia, mu, ia, nu);
                dynmat[idx] = math.complex.sub(
                    dynmat[idx],
                    math.complex.init(dyewq0[ia][dyewq0Index(mu, nu)], 0.0),
                );
            }
        }
    }
}

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
    const ctx = initEwaldContext(cell, recip, charges.len);
    const dynmat = try alloc.alloc(f64, ctx.dim * ctx.dim);
    @memset(dynmat, 0.0);
    addRealSpaceOffDiagonal(&ctx, dynmat, charges, positions);
    addReciprocalOffDiagonal(&ctx, dynmat, charges, positions);
    applyAcousticSumRule(&ctx, dynmat);
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
    const ctx = initEwaldContext(cell, recip, charges.len);
    const dynmat = try alloc.alloc(math.Complex, ctx.dim * ctx.dim);
    errdefer alloc.free(dynmat);
    @memset(dynmat, math.complex.init(0.0, 0.0));
    addRealSpaceOffDiagonalQ(&ctx, dynmat, charges, positions, q_cart);
    addReciprocalContributionQ(&ctx, dynmat, charges, positions, q_cart);
    const dyewq0 = try buildDyewq0(alloc, &ctx, charges, positions);
    defer alloc.free(dyewq0);

    applyDyewq0DiagonalCorrection(&ctx, dynmat, dyewq0);
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

            const forces_plus = try ewald.ionIonForces(
                alloc,
                cell,
                recip_lat,
                &charges,
                &pos_plus,
                null,
            );
            defer alloc.free(forces_plus);

            const forces_minus = try ewald.ionIonForces(
                alloc,
                cell,
                recip_lat,
                &charges,
                &pos_minus,
                null,
            );
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
