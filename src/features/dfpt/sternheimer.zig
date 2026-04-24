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

const SolveWorkspace = struct {
    x: []math.Complex,
    r: []math.Complex,
    z: []math.Complex,
    d: []math.Complex,
    ad: []math.Complex,
    psi_matrix: []math.Complex,
    overlaps: []math.Complex,

    fn deinit(self: *SolveWorkspace, alloc: std.mem.Allocator) void {
        alloc.free(self.r);
        alloc.free(self.z);
        alloc.free(self.d);
        alloc.free(self.ad);
        alloc.free(self.psi_matrix);
        alloc.free(self.overlaps);
    }
};

const IterationUpdate = struct {
    rz: math.Complex,
    residual_norm: f64,
    finished: bool,
};

/// Project onto conduction band subspace: P_c = 1 - Σ_m |ψ_m⟩⟨ψ_m|
/// Modifies x in-place.
pub fn project_conduction(
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
            overlap = math.complex.add(
                overlap,
                math.complex.mul(math.complex.conj(psi_m[g]), x[g]),
            );
        }
        // x -= |ψ_m⟩⟨ψ_m|x⟩
        for (0..n) |g| {
            x[g] = math.complex.sub(x[g], math.complex.scale(psi_m[g], overlap.r));
            x[g] = math.complex.sub(
                x[g],
                math.complex.mul(psi_m[g], math.complex.init(0.0, overlap.i)),
            );
        }
    }
}

/// BLAS-optimized P_c projection using a contiguous occupied matrix.
/// psi_matrix: column-major [n_pw × n_occ], overlaps: work buffer [n_occ].
/// x = x - Psi × (Psi^H × x)
fn project_conduction_blas(
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
fn apply_valence_projection_blas(
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
fn pack_occupied_matrix(
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
fn apply_sternheimer(
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
    try scf_mod.apply_hamiltonian(ctx, x, y);

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
                overlap = math.complex.add(
                    overlap,
                    math.complex.mul(math.complex.conj(psi_m[g]), x[g]),
                );
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
fn apply_sternheimer_blas(
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
    try scf_mod.apply_hamiltonian(ctx, x, y);

    // y -= ε_n × x
    for (0..n_pw) |g| {
        y[g] = math.complex.sub(y[g], math.complex.scale(x[g], epsilon_n));
    }

    // y += α × P_v|x⟩
    if (alpha_shift > 0.0) {
        apply_valence_projection_blas(x, y, psi_matrix, n_pw, n_occ, alpha_shift, overlaps);
    }
}

/// Kinetic energy preconditioner: K = 1 / (|k+G|² - ε_n + α)
fn apply_preconditioner(
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
fn inner_product(a: []const math.Complex, b: []const math.Complex) math.Complex {
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
    var ws = try init_solve_workspace(
        alloc,
        rhs,
        occupied,
        n_occ,
        gvecs,
        epsilon_n,
        params.alpha_shift,
    );
    errdefer alloc.free(ws.x);
    defer ws.deinit(alloc);

    var rz = inner_product(ws.r, ws.z);
    var residual_norm = norm(ws.r);
    var iter: usize = 0;

    while (iter < params.max_iter) : (iter += 1) {
        if (residual_norm < params.tol) break;
        const update = try sternheimer_iteration(
            ctx,
            gvecs,
            epsilon_n,
            n_occ,
            params,
            &ws,
            rz,
        );
        rz = update.rz;
        residual_norm = update.residual_norm;
        if (update.finished) {
            iter += 1;
            break;
        }
    }

    // Final projection
    project_conduction_blas(ws.x, ws.psi_matrix, n, n_occ, ws.overlaps);

    return .{
        .psi1 = ws.x,
        .converged = residual_norm < params.tol,
        .residual_norm = residual_norm,
        .iterations = iter,
    };
}

fn init_solve_workspace(
    alloc: std.mem.Allocator,
    rhs: []const math.Complex,
    occupied: []const []const math.Complex,
    n_occ: usize,
    gvecs: []const plane_wave.GVector,
    epsilon_n: f64,
    alpha_shift: f64,
) !SolveWorkspace {
    const n = rhs.len;
    const x = try alloc.alloc(math.Complex, n);
    errdefer alloc.free(x);
    const r = try alloc.alloc(math.Complex, n);
    errdefer alloc.free(r);
    const z = try alloc.alloc(math.Complex, n);
    errdefer alloc.free(z);
    const d = try alloc.alloc(math.Complex, n);
    errdefer alloc.free(d);
    const ad = try alloc.alloc(math.Complex, n);
    errdefer alloc.free(ad);
    const psi_matrix = try alloc.alloc(math.Complex, n * n_occ);
    errdefer alloc.free(psi_matrix);
    const overlaps = try alloc.alloc(math.Complex, @max(n_occ, 1));
    errdefer alloc.free(overlaps);

    if (n_occ > 0) {
        for (0..n_occ) |m| {
            @memcpy(psi_matrix[m * n .. (m + 1) * n], occupied[m][0..n]);
        }
    }

    @memset(x, math.complex.init(0.0, 0.0));
    @memcpy(r, rhs);
    project_conduction_blas(r, psi_matrix, n, n_occ, overlaps);
    apply_preconditioner(gvecs, r, z, epsilon_n, alpha_shift);
    project_conduction_blas(z, psi_matrix, n, n_occ, overlaps);
    @memcpy(d, z);

    return .{
        .x = x,
        .r = r,
        .z = z,
        .d = d,
        .ad = ad,
        .psi_matrix = psi_matrix,
        .overlaps = overlaps,
    };
}

fn sternheimer_iteration(
    ctx: *scf_mod.ApplyContext,
    gvecs: []const plane_wave.GVector,
    epsilon_n: f64,
    n_occ: usize,
    params: SternheimerParams,
    ws: *SolveWorkspace,
    rz: math.Complex,
) !IterationUpdate {
    const n = ws.r.len;
    try apply_sternheimer_blas(
        ctx,
        ws.d,
        ws.ad,
        epsilon_n,
        params.alpha_shift,
        ws.psi_matrix,
        n,
        n_occ,
        ws.overlaps,
    );
    project_conduction_blas(ws.ad, ws.psi_matrix, n, n_occ, ws.overlaps);

    const dad = inner_product(ws.d, ws.ad);
    if (@abs(dad.r) < 1e-30) {
        return .{ .rz = rz, .residual_norm = norm(ws.r), .finished = true };
    }
    const alpha_cg = rz.r / dad.r;
    for (0..n) |g| {
        ws.x[g] = math.complex.add(ws.x[g], math.complex.scale(ws.d[g], alpha_cg));
        ws.r[g] = math.complex.sub(ws.r[g], math.complex.scale(ws.ad[g], alpha_cg));
    }
    project_conduction_blas(ws.r, ws.psi_matrix, n, n_occ, ws.overlaps);

    const residual_norm = norm(ws.r);
    if (residual_norm < params.tol) {
        return .{ .rz = rz, .residual_norm = residual_norm, .finished = true };
    }

    apply_preconditioner(gvecs, ws.r, ws.z, epsilon_n, params.alpha_shift);
    project_conduction_blas(ws.z, ws.psi_matrix, n, n_occ, ws.overlaps);

    const rz_new = inner_product(ws.r, ws.z);
    const beta = if (@abs(rz.r) > 1e-30) rz_new.r / rz.r else 0.0;
    for (0..n) |g| {
        ws.d[g] = math.complex.add(ws.z[g], math.complex.scale(ws.d[g], beta));
    }
    return .{ .rz = rz_new, .residual_norm = residual_norm, .finished = false };
}

// =========================================================================
// Tests
// =========================================================================

test "project_conduction removes occupied component" {
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
    const overlap_before = inner_product(psi0, x[0..]);

    project_conduction(x[0..], occupied[0..], 1);

    // After projection, overlap with occupied state should be ~0
    const overlap_after = inner_product(psi0, x[0..]);
    try std.testing.expectApproxEqAbs(overlap_after.r, 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(overlap_after.i, 0.0, 1e-12);

    // Projection should have changed the vector (it had a non-zero occupied component)
    _ = overlap_before;
}

test "project_conduction is idempotent" {
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

    project_conduction(x[0..], occupied[0..], 1);

    // Save state after first projection
    var x_after1: [n]math.Complex = undefined;
    @memcpy(&x_after1, &x);

    // Apply again
    project_conduction(x[0..], occupied[0..], 1);

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

    apply_preconditioner(gvecs[0..], x[0..], y[0..], epsilon, alpha);

    // Higher kinetic energy → smaller preconditioner value
    try std.testing.expect(y[0].r > y[1].r);
    try std.testing.expect(y[1].r > y[2].r);
    try std.testing.expect(y[2].r > y[3].r);

    // All values should be positive
    for (0..4) |g| {
        try std.testing.expect(y[g].r > 0.0);
    }
}
