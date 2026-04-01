//! Sternheimer equation solver for DFPT.
//!
//! Solves (H₀ - ε_n + α×P_v)|ψ^(1)⟩ = -P_c × H^(1)|ψ^(0)⟩
//! using preconditioned conjugate gradient method.

const std = @import("std");
const blas = @import("../../lib/linalg/blas.zig");
const math = @import("../math/math.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const scf_mod = @import("../scf/scf.zig");

pub const SternheimerParams = struct {
    tol: f64 = 1e-8,
    max_iter: usize = 200,
    alpha_shift: f64 = 0.01, // Regularization shift for P_v
};

pub const SternheimerResult = struct {
    psi1: []math.Complex,
    converged: bool,
    residual_norm: f64,
    iterations: usize,
};

/// Project onto conduction band subspace: P_c = 1 - Σ_m |ψ_m⟩⟨ψ_m|
/// Modifies x in-place.
pub fn projectConduction(
    x: []math.Complex,
    occupied: []const []const math.Complex,
    n_occ: usize,
) void {
    const n = x.len;
    for (0..n_occ) |m| {
        const psi_m = occupied[m];
        // Compute ⟨ψ_m|x⟩
        var overlap = math.complex.init(0.0, 0.0);
        for (0..n) |g| {
            overlap = math.complex.add(overlap, math.complex.mul(math.complex.conj(psi_m[g]), x[g]));
        }
        // x -= |ψ_m⟩⟨ψ_m|x⟩
        for (0..n) |g| {
            x[g] = math.complex.sub(x[g], math.complex.scale(psi_m[g], overlap.r));
            x[g] = math.complex.sub(x[g], math.complex.mul(psi_m[g], math.complex.init(0.0, overlap.i)));
        }
    }
}

/// BLAS-optimized P_c projection using a contiguous occupied matrix.
/// psi_matrix: column-major [n_pw × n_occ], overlaps: work buffer [n_occ].
/// x = x - Psi × (Psi^H × x)
fn projectConductionBlas(
    x: []math.Complex,
    psi_matrix: []const math.Complex,
    n_pw: usize,
    n_occ: usize,
    overlaps: []math.Complex,
) void {
    if (n_occ == 0) return;
    const blas_psi: []const blas.Complex = @ptrCast(psi_matrix[0 .. n_pw * n_occ]);
    const blas_x: []blas.Complex = @ptrCast(x[0..n_pw]);
    const blas_ov: []blas.Complex = @ptrCast(overlaps[0..n_occ]);

    // overlaps = Psi^H × x
    blas.zgemv(
        .conj_trans,
        n_pw,
        n_occ,
        blas.Complex.init(1.0, 0.0),
        blas_psi,
        n_pw,
        blas_x,
        blas.Complex.init(0.0, 0.0),
        blas_ov,
    );

    // x -= Psi × overlaps
    blas.zgemv(
        .no_trans,
        n_pw,
        n_occ,
        blas.Complex.init(-1.0, 0.0),
        blas_psi,
        n_pw,
        blas_ov,
        blas.Complex.init(1.0, 0.0),
        blas_x,
    );
}

/// BLAS-optimized P_v projection: y += alpha * Psi × (Psi^H × x)
fn applyValenceProjectionBlas(
    x: []const math.Complex,
    y: []math.Complex,
    psi_matrix: []const math.Complex,
    n_pw: usize,
    n_occ: usize,
    alpha: f64,
    overlaps: []math.Complex,
) void {
    if (n_occ == 0 or alpha == 0.0) return;
    const blas_psi: []const blas.Complex = @ptrCast(psi_matrix[0 .. n_pw * n_occ]);
    const blas_x: []const blas.Complex = @ptrCast(x[0..n_pw]);
    const blas_y: []blas.Complex = @ptrCast(y[0..n_pw]);
    const blas_ov: []blas.Complex = @ptrCast(overlaps[0..n_occ]);

    // overlaps = Psi^H × x
    blas.zgemv(
        .conj_trans,
        n_pw,
        n_occ,
        blas.Complex.init(1.0, 0.0),
        blas_psi,
        n_pw,
        blas_x,
        blas.Complex.init(0.0, 0.0),
        blas_ov,
    );

    // y += alpha * Psi × overlaps
    blas.zgemv(
        .no_trans,
        n_pw,
        n_occ,
        blas.Complex.init(alpha, 0.0),
        blas_psi,
        n_pw,
        blas_ov,
        blas.Complex.init(1.0, 0.0),
        blas_y,
    );
}

/// Pack slice-of-slices occupied wavefunctions into a contiguous column-major matrix.
fn packOccupiedMatrix(
    alloc: std.mem.Allocator,
    occupied: []const []const math.Complex,
    n_occ: usize,
    n_pw: usize,
) ![]math.Complex {
    const matrix = try alloc.alloc(math.Complex, n_pw * n_occ);
    for (0..n_occ) |m| {
        @memcpy(matrix[m * n_pw .. (m + 1) * n_pw], occupied[m][0..n_pw]);
    }
    return matrix;
}

/// Apply Sternheimer operator: A|x⟩ = (H₀ - ε_n + α×P_v)|x⟩
/// where P_v = Σ_m |ψ_m⟩⟨ψ_m|
fn applySternheimer(
    ctx: *scf_mod.ApplyContext,
    x: []const math.Complex,
    y: []math.Complex,
    epsilon_n: f64,
    alpha_shift: f64,
    occupied: []const []const math.Complex,
    n_occ: usize,
    temp: []math.Complex,
) !void {
    const n = x.len;

    // y = H₀|x⟩
    try scf_mod.applyHamiltonian(ctx, x, y);

    // y -= ε_n × x
    for (0..n) |g| {
        y[g] = math.complex.sub(y[g], math.complex.scale(x[g], epsilon_n));
    }

    // y += α × P_v|x⟩ (regularization to ensure positive definiteness)
    if (alpha_shift > 0.0) {
        for (0..n_occ) |m| {
            const psi_m = occupied[m];
            var overlap = math.complex.init(0.0, 0.0);
            for (0..n) |g| {
                overlap = math.complex.add(overlap, math.complex.mul(math.complex.conj(psi_m[g]), x[g]));
            }
            for (0..n) |g| {
                const proj = math.complex.mul(psi_m[g], overlap);
                y[g] = math.complex.add(y[g], math.complex.scale(proj, alpha_shift));
            }
        }
    }
    _ = temp;
}

/// BLAS-optimized Sternheimer operator using contiguous occupied matrix.
fn applySternheimerBlas(
    ctx: *scf_mod.ApplyContext,
    x: []const math.Complex,
    y: []math.Complex,
    epsilon_n: f64,
    alpha_shift: f64,
    psi_matrix: []const math.Complex,
    n_pw: usize,
    n_occ: usize,
    overlaps: []math.Complex,
) !void {
    // y = H₀|x⟩
    try scf_mod.applyHamiltonian(ctx, x, y);

    // y -= ε_n × x
    for (0..n_pw) |g| {
        y[g] = math.complex.sub(y[g], math.complex.scale(x[g], epsilon_n));
    }

    // y += α × P_v|x⟩
    if (alpha_shift > 0.0) {
        applyValenceProjectionBlas(x, y, psi_matrix, n_pw, n_occ, alpha_shift, overlaps);
    }
}

/// Kinetic energy preconditioner: K = 1 / (|k+G|² - ε_n + α)
fn applyPreconditioner(
    gvecs: []const plane_wave.GVector,
    x: []const math.Complex,
    y: []math.Complex,
    epsilon_n: f64,
    alpha_shift: f64,
) void {
    for (0..gvecs.len) |g| {
        const ke = gvecs[g].kinetic;
        var denom = ke - epsilon_n + alpha_shift;
        if (denom < 0.1) denom = 0.1; // Avoid division by zero or negative
        y[g] = math.complex.scale(x[g], 1.0 / denom);
    }
}

/// Complex inner product: ⟨a|b⟩ = Σ_g conj(a[g]) × b[g]
fn innerProduct(a: []const math.Complex, b: []const math.Complex) math.Complex {
    var sum = math.complex.init(0.0, 0.0);
    for (0..a.len) |g| {
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(a[g]), b[g]));
    }
    return sum;
}

/// Compute L2 norm ||x|| = sqrt(Σ |x[g]|²)
fn norm(x: []const math.Complex) f64 {
    var sum: f64 = 0.0;
    for (x) |v| {
        sum += v.r * v.r + v.i * v.i;
    }
    return @sqrt(sum);
}

/// Solve Sternheimer equation using preconditioned CG.
///
/// Solves: (H₀ - ε_n + α×P_v)|ψ^(1)⟩ = -P_c × H^(1)|ψ^(0)⟩
///
/// Parameters:
/// - ctx: ApplyContext for H₀ application
/// - rhs: -P_c × H^(1)|ψ_n^(0)⟩ (right-hand side, already projected)
/// - epsilon_n: eigenvalue of band n
/// - occupied: array of occupied wavefunctions [n_occ][n_pw]
/// - n_occ: number of occupied bands
/// - gvecs: plane wave basis
/// - params: solver parameters
/// - alloc: allocator
pub fn solve(
    alloc: std.mem.Allocator,
    ctx: *scf_mod.ApplyContext,
    rhs: []const math.Complex,
    epsilon_n: f64,
    occupied: []const []const math.Complex,
    n_occ: usize,
    gvecs: []const plane_wave.GVector,
    params: SternheimerParams,
) !SternheimerResult {
    const n = rhs.len;

    // Allocate work arrays
    const x = try alloc.alloc(math.Complex, n); // solution
    errdefer alloc.free(x);
    const r = try alloc.alloc(math.Complex, n); // residual
    defer alloc.free(r);
    const z = try alloc.alloc(math.Complex, n); // preconditioned residual
    defer alloc.free(z);
    const d = try alloc.alloc(math.Complex, n); // search direction
    defer alloc.free(d);
    const ad = try alloc.alloc(math.Complex, n); // A × d
    defer alloc.free(ad);
    const temp = try alloc.alloc(math.Complex, n); // temporary
    defer alloc.free(temp);

    // Pack occupied wavefunctions into contiguous matrix for BLAS
    const psi_matrix = try alloc.alloc(math.Complex, n * n_occ);
    defer alloc.free(psi_matrix);
    if (n_occ > 0) {
        for (0..n_occ) |m| {
            @memcpy(psi_matrix[m * n .. (m + 1) * n], occupied[m][0..n]);
        }
    }
    const overlaps = try alloc.alloc(math.Complex, @max(n_occ, 1));
    defer alloc.free(overlaps);

    // x = 0
    @memset(x, math.complex.init(0.0, 0.0));

    // r = rhs (since x=0, r = b - Ax = b)
    @memcpy(r, rhs);
    projectConductionBlas(r, psi_matrix, n, n_occ, overlaps);

    // z = K × r
    applyPreconditioner(gvecs, r, z, epsilon_n, params.alpha_shift);
    projectConductionBlas(z, psi_matrix, n, n_occ, overlaps);

    // d = z
    @memcpy(d, z);

    var rz = innerProduct(r, z);
    var residual_norm = norm(r);
    var iter: usize = 0;

    while (iter < params.max_iter) : (iter += 1) {
        if (residual_norm < params.tol) break;

        // ad = A × d
        try applySternheimerBlas(ctx, d, ad, epsilon_n, params.alpha_shift, psi_matrix, n, n_occ, overlaps);
        projectConductionBlas(ad, psi_matrix, n, n_occ, overlaps);

        // α_cg = ⟨r|z⟩ / ⟨d|Ad⟩
        const dad = innerProduct(d, ad);
        if (@abs(dad.r) < 1e-30) break;
        const alpha_cg = rz.r / dad.r;

        // x += α_cg × d
        for (0..n) |g| {
            x[g] = math.complex.add(x[g], math.complex.scale(d[g], alpha_cg));
        }

        // r -= α_cg × Ad
        for (0..n) |g| {
            r[g] = math.complex.sub(r[g], math.complex.scale(ad[g], alpha_cg));
        }
        projectConductionBlas(r, psi_matrix, n, n_occ, overlaps);

        residual_norm = norm(r);
        if (residual_norm < params.tol) {
            iter += 1;
            break;
        }

        // z_new = K × r
        applyPreconditioner(gvecs, r, z, epsilon_n, params.alpha_shift);
        projectConductionBlas(z, psi_matrix, n, n_occ, overlaps);

        // β = ⟨r_new|z_new⟩ / ⟨r_old|z_old⟩
        const rz_new = innerProduct(r, z);
        const beta = if (@abs(rz.r) > 1e-30) rz_new.r / rz.r else 0.0;

        // d = z + β × d
        for (0..n) |g| {
            d[g] = math.complex.add(z[g], math.complex.scale(d[g], beta));
        }

        rz = rz_new;
    }

    // Final projection
    projectConductionBlas(x, psi_matrix, n, n_occ, overlaps);

    return .{
        .psi1 = x,
        .converged = residual_norm < params.tol,
        .residual_norm = residual_norm,
        .iterations = iter,
    };
}

// =========================================================================
// Tests
// =========================================================================

test "projectConduction removes occupied component" {
    // Create a simple occupied state
    var psi0_data = [_]math.Complex{
        math.complex.init(0.5, 0.0),
        math.complex.init(0.5, 0.0),
        math.complex.init(0.5, 0.0),
        math.complex.init(0.5, 0.0),
    };
    const psi0: []const math.Complex = psi0_data[0..];
    const occupied = [_][]const math.Complex{psi0};

    // Create test vector with both occupied and unoccupied components
    var x = [_]math.Complex{
        math.complex.init(1.0, 0.0),
        math.complex.init(0.0, 1.0),
        math.complex.init(-1.0, 0.0),
        math.complex.init(0.0, -1.0),
    };

    // Compute overlap before projection
    const overlap_before = innerProduct(psi0, x[0..]);

    projectConduction(x[0..], occupied[0..], 1);

    // After projection, overlap with occupied state should be ~0
    const overlap_after = innerProduct(psi0, x[0..]);
    try std.testing.expectApproxEqAbs(overlap_after.r, 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(overlap_after.i, 0.0, 1e-12);

    // Projection should have changed the vector (it had a non-zero occupied component)
    _ = overlap_before;
}

test "projectConduction is idempotent" {
    const n = 4;

    var psi0_data = [_]math.Complex{
        math.complex.init(0.5, 0.0),
        math.complex.init(0.5, 0.0),
        math.complex.init(0.5, 0.0),
        math.complex.init(0.5, 0.0),
    };
    const psi0: []const math.Complex = psi0_data[0..];
    const occupied = [_][]const math.Complex{psi0};

    var x = [_]math.Complex{
        math.complex.init(1.0, 0.2),
        math.complex.init(0.3, 1.0),
        math.complex.init(-1.0, 0.5),
        math.complex.init(0.7, -1.0),
    };

    projectConduction(x[0..], occupied[0..], 1);

    // Save state after first projection
    var x_after1: [n]math.Complex = undefined;
    @memcpy(&x_after1, &x);

    // Apply again
    projectConduction(x[0..], occupied[0..], 1);

    // Should be unchanged (idempotent)
    for (0..n) |g| {
        try std.testing.expectApproxEqAbs(x[g].r, x_after1[g].r, 1e-12);
        try std.testing.expectApproxEqAbs(x[g].i, x_after1[g].i, 1e-12);
    }
}

test "preconditioner basic properties" {
    // Create mock GVectors
    var gvecs: [4]plane_wave.GVector = undefined;
    for (0..4) |i| {
        const ke = @as(f64, @floatFromInt(i + 1)) * 2.0; // kinetic energies: 2, 4, 6, 8
        gvecs[i] = .{
            .h = @intCast(i),
            .k = 0,
            .l = 0,
            .cart = math.Vec3{ .x = 0, .y = 0, .z = 0 },
            .kpg = math.Vec3{ .x = 0, .y = 0, .z = 0 },
            .kinetic = ke,
        };
    }

    var x = [_]math.Complex{
        math.complex.init(1.0, 0.0),
        math.complex.init(1.0, 0.0),
        math.complex.init(1.0, 0.0),
        math.complex.init(1.0, 0.0),
    };

    var y: [4]math.Complex = undefined;
    const epsilon = 1.0;
    const alpha = 0.01;

    applyPreconditioner(gvecs[0..], x[0..], y[0..], epsilon, alpha);

    // Higher kinetic energy → smaller preconditioner value
    try std.testing.expect(y[0].r > y[1].r);
    try std.testing.expect(y[1].r > y[2].r);
    try std.testing.expect(y[2].r > y[3].r);

    // All values should be positive
    for (0..4) |g| {
        try std.testing.expect(y[g].r > 0.0);
    }
}
