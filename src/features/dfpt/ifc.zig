//! Interatomic Force Constants (IFC) module.
//!
//! Provides Fourier interpolation of dynamical matrices:
//!   D(q) → C(R) via forward FT, then C(R) → D(q') via inverse FT.
//! This enables computing phonon frequencies at arbitrary q-points
//! from a coarse q-grid DFPT calculation.

const std = @import("std");
const math = @import("../math/math.zig");
const complex_mod = @import("../math/complex.zig");

const Complex = math.Complex;

/// Interatomic Force Constants in real space.
pub const IFC = struct {
    /// IFC matrices C(R): [n_rvec][dim*dim]
    c_r: [][]Complex,
    /// R-vector lattice coefficients (n1, n2, n3)
    r_vecs: [][3]i32,
    n_rvec: usize,
    n_atoms: usize,
    dim: usize,
    qgrid: [3]usize,

    pub fn deinit(self: *IFC, alloc: std.mem.Allocator) void {
        for (self.c_r) |row| alloc.free(row);
        alloc.free(self.c_r);
        alloc.free(self.r_vecs);
    }
};

/// Generate R-vectors for the given q-grid.
/// R = n1*a1 + n2*a2 + n3*a3, where n_i ∈ {0, ..., N_i-1}.
pub fn generateRvectors(alloc: std.mem.Allocator, qgrid: [3]usize) ![][3]i32 {
    const n_total = qgrid[0] * qgrid[1] * qgrid[2];
    var rvecs = try alloc.alloc([3]i32, n_total);
    var idx: usize = 0;
    for (0..qgrid[0]) |i| {
        for (0..qgrid[1]) |j| {
            for (0..qgrid[2]) |k| {
                rvecs[idx] = .{
                    @as(i32, @intCast(i)),
                    @as(i32, @intCast(j)),
                    @as(i32, @intCast(k)),
                };
                idx += 1;
            }
        }
    }
    return rvecs;
}

/// Compute IFC from dynamical matrices on a q-grid.
/// C(R) = (1/Nq) Σ_q D(q) × exp(-i 2π q_frac · n)
pub fn computeIFC(
    alloc: std.mem.Allocator,
    dynmat_q: []const []const Complex,
    q_points_frac: []const math.Vec3,
    qgrid: [3]usize,
    n_atoms: usize,
) !IFC {
    const dim = 3 * n_atoms;
    const dim2 = dim * dim;
    const n_q = q_points_frac.len;
    const nq_f = @as(f64, @floatFromInt(n_q));

    const r_vecs = try generateRvectors(alloc, qgrid);
    errdefer alloc.free(r_vecs);
    const n_rvec = r_vecs.len;

    var c_r = try alloc.alloc([]Complex, n_rvec);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| alloc.free(c_r[i]);
        alloc.free(c_r);
    }

    for (0..n_rvec) |ir| {
        c_r[ir] = try alloc.alloc(Complex, dim2);
        built = ir + 1;
        @memset(c_r[ir], Complex{ .r = 0.0, .i = 0.0 });

        const n = r_vecs[ir];
        const n1 = @as(f64, @floatFromInt(n[0]));
        const n2 = @as(f64, @floatFromInt(n[1]));
        const n3 = @as(f64, @floatFromInt(n[2]));

        for (0..n_q) |iq| {
            const q = q_points_frac[iq];
            // Phase: exp(-i 2π q · n)
            const phase_arg = -2.0 * std.math.pi * (q.x * n1 + q.y * n2 + q.z * n3);
            const phase = complex_mod.expi(phase_arg);

            for (0..dim2) |idx| {
                const prod = complex_mod.mul(dynmat_q[iq][idx], phase);
                c_r[ir][idx] = complex_mod.add(c_r[ir][idx], prod);
            }
        }

        // Normalize by 1/Nq
        for (0..dim2) |idx| {
            c_r[ir][idx] = complex_mod.scale(c_r[ir][idx], 1.0 / nq_f);
        }
    }

    return IFC{
        .c_r = c_r,
        .r_vecs = r_vecs,
        .n_rvec = n_rvec,
        .n_atoms = n_atoms,
        .dim = dim,
        .qgrid = qgrid,
    };
}

/// Apply acoustic sum rule in IFC space.
/// For each atom I and directions α,β:
///   C_{Iα,Iβ}(R=0) = -Σ_{(J,R)≠(I,0)} C_{Iα,Jβ}(R)
pub fn applyASR(ifc: *IFC) void {
    const dim = ifc.dim;
    const n_atoms = ifc.n_atoms;

    // Find R=0 index
    var r0_idx: ?usize = null;
    for (ifc.r_vecs, 0..) |rv, ir| {
        if (rv[0] == 0 and rv[1] == 0 and rv[2] == 0) {
            r0_idx = ir;
            break;
        }
    }
    const ir0 = r0_idx orelse return;

    // For each atom I and directions α, β
    for (0..n_atoms) |ia| {
        for (0..3) |alpha| {
            for (0..3) |beta| {
                var sum = Complex{ .r = 0.0, .i = 0.0 };

                // Sum over all (J, R) excluding (I, R=0)
                for (0..ifc.n_rvec) |ir| {
                    for (0..n_atoms) |ja| {
                        if (ir == ir0 and ja == ia) continue;
                        const idx = (3 * ia + alpha) * dim + (3 * ja + beta);
                        sum = complex_mod.add(sum, ifc.c_r[ir][idx]);
                    }
                }

                // Set diagonal: C_{Iα,Iβ}(R=0) = -sum
                const diag_idx = (3 * ia + alpha) * dim + (3 * ia + beta);
                ifc.c_r[ir0][diag_idx] = Complex{ .r = -sum.r, .i = -sum.i };
            }
        }
    }
}

/// Interpolate dynamical matrix at arbitrary q-point using IFC.
/// D(q') = Σ_R C(R) × exp(+i 2π q'_frac · n)
pub fn interpolate(
    alloc: std.mem.Allocator,
    ifc: *const IFC,
    q_frac: math.Vec3,
) ![]Complex {
    const dim2 = ifc.dim * ifc.dim;
    var dyn = try alloc.alloc(Complex, dim2);
    @memset(dyn, Complex{ .r = 0.0, .i = 0.0 });

    for (0..ifc.n_rvec) |ir| {
        const n = ifc.r_vecs[ir];
        const n1 = @as(f64, @floatFromInt(n[0]));
        const n2 = @as(f64, @floatFromInt(n[1]));
        const n3 = @as(f64, @floatFromInt(n[2]));

        // Phase: exp(+i 2π q' · n)
        const phase_arg = 2.0 * std.math.pi * (q_frac.x * n1 + q_frac.y * n2 + q_frac.z * n3);
        const phase = complex_mod.expi(phase_arg);

        for (0..dim2) |idx| {
            const prod = complex_mod.mul(ifc.c_r[ir][idx], phase);
            dyn[idx] = complex_mod.add(dyn[idx], prod);
        }
    }

    return dyn;
}

/// Non-analytic correction (NAC) data for LO-TO splitting.
pub const NacData = struct {
    /// Born effective charges Z*[atom][α][β]
    z_star: [][3][3]f64,
    /// Dielectric tensor ε∞[3][3]
    epsilon: [3][3]f64,
    /// Cell volume (bohr³)
    volume: f64,
    n_atoms: usize,
};

/// Interpolate dynamical matrix at q-point with non-analytic correction.
/// D(q) = D^ana(q) + D^NA(q→0)
///
/// D^NA_{κα,κ'β}(q̂) = (8π/Ω) × (q̂·Z*_κ)_α × (q̂·Z*_κ')_β / (q̂·ε∞·q̂)
///
/// The NAC is only applied for q ≠ 0 (q_norm > threshold).
pub fn interpolateWithNAC(
    alloc: std.mem.Allocator,
    ifc_data: *const IFC,
    q_frac: math.Vec3,
    q_cart: math.Vec3,
    nac: NacData,
) ![]Complex {
    // Start with analytic (short-range) interpolation
    var dyn = try interpolate(alloc, ifc_data, q_frac);

    // Only add NAC for finite q
    const q_norm = math.Vec3.norm(q_cart);
    if (q_norm < 1e-8) return dyn;

    // q̂ = q / |q| as array for indexed access
    const inv_qnorm = 1.0 / q_norm;
    const qh = [3]f64{ q_cart.x * inv_qnorm, q_cart.y * inv_qnorm, q_cart.z * inv_qnorm };

    // Compute q̂·ε∞·q̂
    var qeq: f64 = 0.0;
    for (0..3) |a| {
        for (0..3) |b| {
            qeq += qh[a] * nac.epsilon[a][b] * qh[b];
        }
    }

    if (@abs(qeq) < 1e-12) return dyn;

    const dim = ifc_data.dim;
    const prefactor = 8.0 * std.math.pi / nac.volume;

    for (0..nac.n_atoms) |ka| {
        for (0..3) |alpha| {
            var qz_ka: f64 = 0.0;
            for (0..3) |beta| {
                qz_ka += qh[beta] * nac.z_star[ka][alpha][beta];
            }

            for (0..nac.n_atoms) |kb| {
                for (0..3) |beta2| {
                    var qz_kb: f64 = 0.0;
                    for (0..3) |gamma| {
                        qz_kb += qh[gamma] * nac.z_star[kb][beta2][gamma];
                    }

                    const nac_val = prefactor * qz_ka * qz_kb / qeq;
                    const idx = (3 * ka + alpha) * dim + (3 * kb + beta2);
                    dyn[idx].r += nac_val;
                }
            }
        }
    }

    return dyn;
}

// =====================================================================
// Tests
// =====================================================================

test "IFC round-trip: D(q) -> C(R) -> D(q) recovers original" {
    const alloc = std.testing.allocator;

    // Simple 1-atom system (dim=3), 1×1×2 q-grid
    const n_atoms: usize = 1;
    const dim: usize = 3;
    const dim2 = dim * dim;
    const qgrid = [3]usize{ 1, 1, 2 };

    // Create two q-points: q=(0,0,0) and q=(0,0,0.5)
    const q_points = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.5 },
    };

    // Create simple dynamical matrices
    var dyn0 = try alloc.alloc(Complex, dim2);
    defer alloc.free(dyn0);

    var dyn1 = try alloc.alloc(Complex, dim2);
    defer alloc.free(dyn1);

    // D(q=0): diagonal 1.0
    for (0..dim2) |i| {
        const row = i / dim;
        const col = i % dim;
        dyn0[i] = if (row == col) Complex{ .r = 1.0, .i = 0.0 } else Complex{ .r = 0.0, .i = 0.0 };
    }
    // D(q=0.5): diagonal 3.0
    for (0..dim2) |i| {
        const row = i / dim;
        const col = i % dim;
        dyn1[i] = if (row == col) Complex{ .r = 3.0, .i = 0.0 } else Complex{ .r = 0.0, .i = 0.0 };
    }

    const dynmat_q = [_][]const Complex{ dyn0, dyn1 };

    // Forward: D(q) -> C(R)
    var ifc_data = try computeIFC(alloc, &dynmat_q, &q_points, qgrid, n_atoms);
    defer ifc_data.deinit(alloc);

    // Inverse: C(R) -> D(q) should recover original
    for (0..2) |iq| {
        const dyn_interp = try interpolate(alloc, &ifc_data, q_points[iq]);
        defer alloc.free(dyn_interp);

        for (0..dim2) |i| {
            const orig = dynmat_q[iq][i];
            const interp = dyn_interp[i];
            try std.testing.expectApproxEqAbs(orig.r, interp.r, 1e-12);
            try std.testing.expectApproxEqAbs(orig.i, interp.i, 1e-12);
        }
    }
}

test "IFC ASR: acoustic modes have zero frequency at Gamma" {
    const alloc = std.testing.allocator;

    // 2-atom system, 1×1×2 q-grid
    const n_atoms: usize = 2;
    const dim: usize = 6;
    const dim2 = dim * dim;
    const qgrid = [3]usize{ 1, 1, 2 };

    const q_points = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.5 },
    };

    // Create D(q) with known structure
    var dyn0 = try alloc.alloc(Complex, dim2);
    defer alloc.free(dyn0);

    var dyn1 = try alloc.alloc(Complex, dim2);
    defer alloc.free(dyn1);

    // Simple spring model: atom1-atom2 coupled along z
    @memset(dyn0, Complex{ .r = 0.0, .i = 0.0 });
    @memset(dyn1, Complex{ .r = 0.0, .i = 0.0 });

    // D(q=0): spring constant k=2 for z-z coupling
    // Atom 1 self: k, Atom 2 self: k, cross: -k
    const k_spring: f64 = 2.0;
    dyn0[2 * dim + 2] = Complex{ .r = k_spring, .i = 0.0 }; // (1z,1z)
    dyn0[2 * dim + 5] = Complex{ .r = -k_spring, .i = 0.0 }; // (1z,2z)
    dyn0[5 * dim + 2] = Complex{ .r = -k_spring, .i = 0.0 }; // (2z,1z)
    dyn0[5 * dim + 5] = Complex{ .r = k_spring, .i = 0.0 }; // (2z,2z)

    // D(q=0.5): for simplicity same structure
    dyn1[2 * dim + 2] = Complex{ .r = k_spring, .i = 0.0 };
    dyn1[2 * dim + 5] = Complex{ .r = k_spring, .i = 0.0 }; // sign flip at BZ boundary
    dyn1[5 * dim + 2] = Complex{ .r = k_spring, .i = 0.0 };
    dyn1[5 * dim + 5] = Complex{ .r = k_spring, .i = 0.0 };

    const dynmat_q = [_][]const Complex{ dyn0, dyn1 };

    var ifc_data = try computeIFC(alloc, &dynmat_q, &q_points, qgrid, n_atoms);
    defer ifc_data.deinit(alloc);

    // Apply ASR
    applyASR(&ifc_data);

    // Interpolate at Γ and check ASR: row sums should be zero
    const dyn_gamma = try interpolate(alloc, &ifc_data, q_points[0]);
    defer alloc.free(dyn_gamma);

    for (0..dim) |row| {
        var row_sum_r: f64 = 0.0;
        var row_sum_i: f64 = 0.0;
        for (0..dim) |col| {
            row_sum_r += dyn_gamma[row * dim + col].r;
            row_sum_i += dyn_gamma[row * dim + col].i;
        }
        try std.testing.expectApproxEqAbs(0.0, row_sum_r, 1e-12);
        try std.testing.expectApproxEqAbs(0.0, row_sum_i, 1e-12);
    }
}
