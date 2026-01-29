const std = @import("std");
const math = @import("../math/math.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("grid.zig");

pub const Grid = grid_mod.Grid;

/// Mix densities by linear mixing.
pub fn mixDensity(rho: []f64, rho_new: []const f64, beta: f64) void {
    for (rho, 0..) |value, i| {
        rho[i] = (1.0 - beta) * value + beta * rho_new[i];
    }
}

/// Mix densities with Kerker preconditioning.
/// Kerker kernel: K(G) = G² / (G² + q0²)
/// This suppresses long-wavelength (small G) charge oscillations.
pub fn mixDensityKerker(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    rho_new: []const f64,
    beta: f64,
    q0: f64,
    use_rfft: bool,
) !void {
    const q0_sq = q0 * q0;

    // Compute density difference
    const delta_rho = try alloc.alloc(f64, rho.len);
    defer alloc.free(delta_rho);
    for (rho, 0..) |value, i| {
        delta_rho[i] = rho_new[i] - value;
    }

    // FFT to reciprocal space
    const delta_rho_g = try fft_grid.realToReciprocal(alloc, grid, delta_rho, use_rfft);
    defer alloc.free(delta_rho_g);

    // Apply Kerker kernel in reciprocal space
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    var idx: usize = 0;
    var h: i32 = grid.min_h;
    while (h < grid.min_h + @as(i32, @intCast(grid.nx))) : (h += 1) {
        var k: i32 = grid.min_k;
        while (k < grid.min_k + @as(i32, @intCast(grid.ny))) : (k += 1) {
            var l: i32 = grid.min_l;
            while (l < grid.min_l + @as(i32, @intCast(grid.nz))) : (l += 1) {
                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(h))), math.Vec3.scale(b2, @as(f64, @floatFromInt(k)))),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(l))),
                );
                const g2 = math.Vec3.dot(gvec, gvec);

                // Kerker kernel: K(G) = G² / (G² + q0²)
                // For G=0, K=0 (no update to average density)
                const kerker = if (g2 > 1e-12) g2 / (g2 + q0_sq) else 0.0;
                delta_rho_g[idx] = math.complex.scale(delta_rho_g[idx], kerker);
                idx += 1;
            }
        }
    }

    // IFFT back to real space
    const delta_rho_precond = try fft_grid.reciprocalToReal(alloc, grid, delta_rho_g);
    defer alloc.free(delta_rho_precond);

    // Mix: ρ = ρ_old + β × K(Δρ)
    for (rho, 0..) |*value, i| {
        value.* += beta * delta_rho_precond[i];
    }
}

/// Pulay/DIIS mixer for accelerated SCF convergence.
/// Stores history of densities and residuals, computes optimal linear combination.
pub const PulayMixer = struct {
    history: usize,
    rho_history: std.ArrayList([]f64),
    residual_history: std.ArrayList([]f64),
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, history: usize) PulayMixer {
        return PulayMixer{
            .history = history,
            .rho_history = .empty,
            .residual_history = .empty,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *PulayMixer) void {
        for (self.rho_history.items) |arr| {
            self.alloc.free(arr);
        }
        self.rho_history.deinit(self.alloc);
        for (self.residual_history.items) |arr| {
            self.alloc.free(arr);
        }
        self.residual_history.deinit(self.alloc);
    }

    /// Mix density using Pulay/DIIS method.
    /// Returns the optimal mixed density.
    pub fn mix(self: *PulayMixer, rho: []f64, rho_new: []const f64, beta: f64) !void {
        const n = rho.len;

        // Compute residual R = rho_new - rho
        const residual = try self.alloc.alloc(f64, n);
        errdefer self.alloc.free(residual);
        for (0..n) |i| {
            residual[i] = rho_new[i] - rho[i];
        }

        // Store current density and residual
        const rho_copy = try self.alloc.alloc(f64, n);
        errdefer self.alloc.free(rho_copy);
        @memcpy(rho_copy, rho);

        try self.rho_history.append(self.alloc, rho_copy);
        try self.residual_history.append(self.alloc, residual);

        // Remove oldest entries if history is full
        while (self.rho_history.items.len > self.history) {
            self.alloc.free(self.rho_history.orderedRemove(0));
            self.alloc.free(self.residual_history.orderedRemove(0));
        }

        const m = self.rho_history.items.len;
        if (m < 2) {
            // Not enough history, use simple linear mixing
            for (0..n) |i| {
                rho[i] = rho[i] + beta * (rho_new[i] - rho[i]);
            }
            return;
        }

        // Build overlap matrix B_ij = <R_i | R_j>
        const matrix_size = m + 1;
        const B = try self.alloc.alloc(f64, matrix_size * matrix_size);
        defer self.alloc.free(B);
        const rhs = try self.alloc.alloc(f64, matrix_size);
        defer self.alloc.free(rhs);

        for (0..m) |i| {
            for (0..m) |j| {
                var dot: f64 = 0.0;
                for (0..n) |k| {
                    dot += self.residual_history.items[i][k] * self.residual_history.items[j][k];
                }
                B[i * matrix_size + j] = dot;
            }
            B[i * matrix_size + m] = -1.0;
            B[m * matrix_size + i] = -1.0;
            rhs[i] = 0.0;
        }
        B[m * matrix_size + m] = 0.0;
        rhs[m] = -1.0;

        // Solve linear system B * c = rhs
        const coeffs = try solvePulaySystem(self.alloc, B, rhs, matrix_size);
        defer self.alloc.free(coeffs);

        // Compute optimal density: rho = sum_i c_i * (rho_i + beta * R_i)
        @memset(rho, 0.0);
        for (0..m) |i| {
            const c = coeffs[i];
            for (0..n) |k| {
                rho[k] += c * (self.rho_history.items[i][k] + beta * self.residual_history.items[i][k]);
            }
        }
    }

    /// Mix using Pulay/DIIS with a pre-computed (preconditioned) residual.
    /// The caller is responsible for computing and preconditioning the residual.
    /// Ownership of precond_residual is transferred to the mixer.
    pub fn mixWithResidual(self: *PulayMixer, rho: []f64, precond_residual: []f64, beta: f64) !void {
        const n = rho.len;

        // Store current input and preconditioned residual
        const rho_copy = try self.alloc.alloc(f64, n);
        errdefer self.alloc.free(rho_copy);
        @memcpy(rho_copy, rho);

        try self.rho_history.append(self.alloc, rho_copy);
        try self.residual_history.append(self.alloc, precond_residual);

        // Remove oldest entries if history is full
        while (self.rho_history.items.len > self.history) {
            self.alloc.free(self.rho_history.orderedRemove(0));
            self.alloc.free(self.residual_history.orderedRemove(0));
        }

        const m = self.rho_history.items.len;
        if (m < 2) {
            // Not enough history, use simple linear mixing
            for (0..n) |i| {
                rho[i] = rho[i] + beta * precond_residual[i];
            }
            return;
        }

        // Build overlap matrix B_ij = <PR_i | PR_j>
        const matrix_size = m + 1;
        const B = try self.alloc.alloc(f64, matrix_size * matrix_size);
        defer self.alloc.free(B);
        const rhs = try self.alloc.alloc(f64, matrix_size);
        defer self.alloc.free(rhs);

        for (0..m) |i| {
            for (0..m) |j| {
                var dot: f64 = 0.0;
                for (0..n) |k| {
                    dot += self.residual_history.items[i][k] * self.residual_history.items[j][k];
                }
                B[i * matrix_size + j] = dot;
            }
            B[i * matrix_size + m] = -1.0;
            B[m * matrix_size + i] = -1.0;
            rhs[i] = 0.0;
        }
        B[m * matrix_size + m] = 0.0;
        rhs[m] = -1.0;

        const coeffs = try solvePulaySystem(self.alloc, B, rhs, matrix_size);
        defer self.alloc.free(coeffs);

        // Compute optimal: rho = sum_i c_i * (rho_i + beta * PR_i)
        @memset(rho, 0.0);
        for (0..m) |i| {
            const c = coeffs[i];
            for (0..n) |k| {
                rho[k] += c * (self.rho_history.items[i][k] + beta * self.residual_history.items[i][k]);
            }
        }
    }

    /// Mix density using Pulay/DIIS with Kerker preconditioning.
    /// Kerker suppresses long-wavelength charge oscillations for faster convergence.
    pub fn mixKerkerPulay(self: *PulayMixer, rho: []f64, rho_new: []const f64, beta: f64, grid: Grid, q0: f64, use_rfft: bool) !void {
        const n = rho.len;

        // Compute raw residual R = rho_new - rho
        const residual = try self.alloc.alloc(f64, n);
        errdefer self.alloc.free(residual);
        for (0..n) |i| {
            residual[i] = rho_new[i] - rho[i];
        }

        // Apply Kerker preconditioning to residual
        const precond_residual = try applyKerkerPreconditioner(self.alloc, grid, residual, q0, use_rfft);
        errdefer self.alloc.free(precond_residual);
        self.alloc.free(residual);

        // Store current density and preconditioned residual
        const rho_copy = try self.alloc.alloc(f64, n);
        errdefer self.alloc.free(rho_copy);
        @memcpy(rho_copy, rho);

        try self.rho_history.append(self.alloc, rho_copy);
        try self.residual_history.append(self.alloc, precond_residual);

        // Remove oldest entries if history is full
        while (self.rho_history.items.len > self.history) {
            self.alloc.free(self.rho_history.orderedRemove(0));
            self.alloc.free(self.residual_history.orderedRemove(0));
        }

        const m = self.rho_history.items.len;
        if (m < 2) {
            // Not enough history, use Kerker-preconditioned linear mixing
            for (0..n) |i| {
                rho[i] = rho[i] + beta * self.residual_history.items[m - 1][i];
            }
            return;
        }

        // Build overlap matrix using preconditioned residuals
        const matrix_size = m + 1;
        const B = try self.alloc.alloc(f64, matrix_size * matrix_size);
        defer self.alloc.free(B);
        const rhs_vec = try self.alloc.alloc(f64, matrix_size);
        defer self.alloc.free(rhs_vec);

        for (0..m) |i| {
            for (0..m) |j| {
                var dot: f64 = 0.0;
                for (0..n) |k| {
                    dot += self.residual_history.items[i][k] * self.residual_history.items[j][k];
                }
                B[i * matrix_size + j] = dot;
            }
            B[i * matrix_size + m] = -1.0;
            B[m * matrix_size + i] = -1.0;
            rhs_vec[i] = 0.0;
        }
        B[m * matrix_size + m] = 0.0;
        rhs_vec[m] = -1.0;

        const coeffs = try solvePulaySystem(self.alloc, B, rhs_vec, matrix_size);
        defer self.alloc.free(coeffs);

        // Compute optimal density: rho = sum_i c_i * (rho_i + beta * PR_i)
        @memset(rho, 0.0);
        for (0..m) |i| {
            const c = coeffs[i];
            for (0..n) |k| {
                rho[k] += c * (self.rho_history.items[i][k] + beta * self.residual_history.items[i][k]);
            }
        }
    }
};

/// Apply Kerker preconditioner to a residual vector.
/// Returns a new allocated array with the preconditioned residual.
fn applyKerkerPreconditioner(alloc: std.mem.Allocator, grid: Grid, residual: []const f64, q0: f64, use_rfft: bool) ![]f64 {
    const q0_sq = q0 * q0;

    // FFT to reciprocal space
    const res_g = try fft_grid.realToReciprocal(alloc, grid, residual, use_rfft);
    defer alloc.free(res_g);

    // Apply Kerker kernel in reciprocal space
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    var idx: usize = 0;
    var h: i32 = grid.min_h;
    while (h < grid.min_h + @as(i32, @intCast(grid.nx))) : (h += 1) {
        var k: i32 = grid.min_k;
        while (k < grid.min_k + @as(i32, @intCast(grid.ny))) : (k += 1) {
            var l: i32 = grid.min_l;
            while (l < grid.min_l + @as(i32, @intCast(grid.nz))) : (l += 1) {
                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(h))), math.Vec3.scale(b2, @as(f64, @floatFromInt(k)))),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(l))),
                );
                const g2 = math.Vec3.dot(gvec, gvec);
                const kerker = if (g2 > 1e-12) g2 / (g2 + q0_sq) else 0.0;
                res_g[idx] = math.complex.scale(res_g[idx], kerker);
                idx += 1;
            }
        }
    }

    // IFFT back to real space
    return fft_grid.reciprocalToReal(alloc, grid, res_g);
}

/// Solve the Pulay linear system using simple Gaussian elimination.
fn solvePulaySystem(alloc: std.mem.Allocator, B: []f64, rhs: []f64, size: usize) ![]f64 {
    // Copy matrix and rhs for in-place solving
    const A = try alloc.alloc(f64, size * size);
    defer alloc.free(A);
    @memcpy(A, B);

    const b = try alloc.alloc(f64, size);
    defer alloc.free(b);
    @memcpy(b, rhs);

    // Gaussian elimination with partial pivoting
    for (0..size) |col| {
        // Find pivot
        var max_row = col;
        var max_val = @abs(A[col * size + col]);
        for (col + 1..size) |row| {
            const val = @abs(A[row * size + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if (max_row != col) {
            for (0..size) |j| {
                const tmp = A[col * size + j];
                A[col * size + j] = A[max_row * size + j];
                A[max_row * size + j] = tmp;
            }
            const tmp = b[col];
            b[col] = b[max_row];
            b[max_row] = tmp;
        }

        // Eliminate below
        const pivot = A[col * size + col];
        if (@abs(pivot) < 1e-12) {
            // Singular matrix, fall back to equal weights
            const result = try alloc.alloc(f64, size);
            for (0..size - 1) |i| {
                result[i] = 1.0 / @as(f64, @floatFromInt(size - 1));
            }
            result[size - 1] = 0.0;
            return result;
        }

        for (col + 1..size) |row| {
            const factor = A[row * size + col] / pivot;
            for (col..size) |j| {
                A[row * size + j] -= factor * A[col * size + j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    const result = try alloc.alloc(f64, size);
    var i: usize = size;
    while (i > 0) {
        i -= 1;
        var sum: f64 = b[i];
        for (i + 1..size) |j| {
            sum -= A[i * size + j] * result[j];
        }
        result[i] = sum / A[i * size + i];
    }

    return result;
}

/// Pulay/DIIS mixer for complex arrays (DFPT q≠0 density response).
/// Stores history of complex densities and preconditioned residuals.
pub const ComplexPulayMixer = struct {
    history: usize,
    rho_history: std.ArrayList([]math.Complex),
    residual_history: std.ArrayList([]math.Complex),
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, history: usize) ComplexPulayMixer {
        return ComplexPulayMixer{
            .history = history,
            .rho_history = .empty,
            .residual_history = .empty,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *ComplexPulayMixer) void {
        for (self.rho_history.items) |arr| self.alloc.free(arr);
        self.rho_history.deinit(self.alloc);
        for (self.residual_history.items) |arr| self.alloc.free(arr);
        self.residual_history.deinit(self.alloc);
    }

    pub fn reset(self: *ComplexPulayMixer) void {
        for (self.rho_history.items) |arr| self.alloc.free(arr);
        self.rho_history.clearRetainingCapacity();
        for (self.residual_history.items) |arr| self.alloc.free(arr);
        self.residual_history.clearRetainingCapacity();
    }

    /// Mix complex arrays using Pulay/DIIS with pre-computed (preconditioned) residual.
    /// Ownership of precond_residual is transferred to the mixer.
    pub fn mixWithResidual(self: *ComplexPulayMixer, rho: []math.Complex, precond_residual: []math.Complex, beta: f64) !void {
        const n = rho.len;

        // Store current input and preconditioned residual
        const rho_copy = try self.alloc.alloc(math.Complex, n);
        errdefer self.alloc.free(rho_copy);
        @memcpy(rho_copy, rho);

        try self.rho_history.append(self.alloc, rho_copy);
        try self.residual_history.append(self.alloc, precond_residual);

        // Remove oldest entries if history is full
        while (self.rho_history.items.len > self.history) {
            self.alloc.free(self.rho_history.orderedRemove(0));
            self.alloc.free(self.residual_history.orderedRemove(0));
        }

        const m = self.rho_history.items.len;
        if (m < 2) {
            // Not enough history, use simple preconditioned linear mixing
            for (0..n) |i| {
                rho[i] = math.complex.add(rho[i], math.complex.scale(precond_residual[i], beta));
            }
            return;
        }

        // Build overlap matrix B_ij = Re⟨PR_i | PR_j⟩
        const matrix_size = m + 1;
        const B = try self.alloc.alloc(f64, matrix_size * matrix_size);
        defer self.alloc.free(B);
        const rhs = try self.alloc.alloc(f64, matrix_size);
        defer self.alloc.free(rhs);

        for (0..m) |i| {
            for (0..m) |j| {
                var dot: f64 = 0.0;
                for (0..n) |k| {
                    // Re(conj(a) * b) = a.r*b.r + a.i*b.i
                    dot += self.residual_history.items[i][k].r * self.residual_history.items[j][k].r +
                        self.residual_history.items[i][k].i * self.residual_history.items[j][k].i;
                }
                B[i * matrix_size + j] = dot;
            }
            B[i * matrix_size + m] = -1.0;
            B[m * matrix_size + i] = -1.0;
            rhs[i] = 0.0;
        }
        B[m * matrix_size + m] = 0.0;
        rhs[m] = -1.0;

        const coeffs = try solvePulaySystem(self.alloc, B, rhs, matrix_size);
        defer self.alloc.free(coeffs);

        // Compute optimal: rho = sum_i c_i * (rho_i + beta * PR_i)
        for (0..n) |k| {
            rho[k] = math.complex.init(0.0, 0.0);
        }
        for (0..m) |i| {
            const c = coeffs[i];
            for (0..n) |k| {
                rho[k] = math.complex.add(rho[k], math.complex.scale(
                    math.complex.add(
                        self.rho_history.items[i][k],
                        math.complex.scale(self.residual_history.items[i][k], beta),
                    ),
                    c,
                ));
            }
        }
    }
};
