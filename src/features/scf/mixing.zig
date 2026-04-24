const std = @import("std");
const math = @import("../math/math.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const gvec_iter = @import("gvec_iter.zig");

pub const Grid = grid_mod.Grid;

/// Mix densities by linear mixing.
pub fn mix_density(rho: []f64, rho_new: []const f64, beta: f64) void {
    for (rho, 0..) |value, i| {
        rho[i] = (1.0 - beta) * value + beta * rho_new[i];
    }
}

/// Mix densities with Kerker preconditioning.
/// Kerker kernel: K(G) = G² / (G² + q0²)
/// This suppresses long-wavelength (small G) charge oscillations.
pub fn mix_density_kerker(
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
    const delta_rho_g = try fft_grid.real_to_reciprocal(alloc, grid, delta_rho, use_rfft);
    defer alloc.free(delta_rho_g);

    // Apply Kerker kernel in reciprocal space
    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        // Kerker kernel: K(G) = G² / (G² + q0²)
        // For G=0, K=0 (no update to average density)
        const kerker = if (g.g2 > 1e-12) g.g2 / (g.g2 + q0_sq) else 0.0;
        delta_rho_g[g.idx] = math.complex.scale(delta_rho_g[g.idx], kerker);
    }

    // IFFT back to real space
    const delta_rho_precond = try fft_grid.reciprocal_to_real(alloc, grid, delta_rho_g);
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

    fn append_preconditioned_history(
        self: *PulayMixer,
        rho: []const f64,
        precond_residual: []f64,
    ) !void {
        const rho_copy = try self.alloc.alloc(f64, rho.len);
        @memcpy(rho_copy, rho);

        self.rho_history.append(self.alloc, rho_copy) catch |err| {
            self.alloc.free(rho_copy);
            return err;
        };
        self.residual_history.append(self.alloc, precond_residual) catch |err| {
            self.alloc.free(self.rho_history.pop().?);
            return err;
        };
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
        const coeffs = try solve_pulay_system(self.alloc, B, rhs, matrix_size);
        defer self.alloc.free(coeffs);

        // Compute optimal density: rho = sum_i c_i * (rho_i + beta * R_i)
        @memset(rho, 0.0);
        for (0..m) |i| {
            const c = coeffs[i];
            const rho_i = self.rho_history.items[i];
            const res_i = self.residual_history.items[i];
            for (0..n) |k| {
                rho[k] += c * (rho_i[k] + beta * res_i[k]);
            }
        }
    }

    /// Mix using Pulay/DIIS with a pre-computed (preconditioned) residual.
    /// The caller is responsible for computing and preconditioning the residual.
    /// Ownership of precond_residual is transferred to the mixer.
    pub fn mix_with_residual(
        self: *PulayMixer,
        rho: []f64,
        precond_residual: []f64,
        beta: f64,
    ) !void {
        const n = rho.len;

        // Store current input and preconditioned residual
        try self.append_preconditioned_history(rho, precond_residual);

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
            const res_i = self.residual_history.items[i];
            for (0..m) |j| {
                const res_j = self.residual_history.items[j];
                var dot: f64 = 0.0;
                for (0..n) |k| {
                    dot += res_i[k] * res_j[k];
                }
                B[i * matrix_size + j] = dot;
            }
            B[i * matrix_size + m] = -1.0;
            B[m * matrix_size + i] = -1.0;
            rhs[i] = 0.0;
        }
        B[m * matrix_size + m] = 0.0;
        rhs[m] = -1.0;

        const coeffs = try solve_pulay_system(self.alloc, B, rhs, matrix_size);
        defer self.alloc.free(coeffs);

        // Compute optimal: rho = sum_i c_i * (rho_i + beta * PR_i)
        @memset(rho, 0.0);
        for (0..m) |i| {
            const c = coeffs[i];
            const rho_i = self.rho_history.items[i];
            const res_i = self.residual_history.items[i];
            for (0..n) |k| {
                rho[k] += c * (rho_i[k] + beta * res_i[k]);
            }
        }
    }

    /// Mix density using Pulay/DIIS with Kerker preconditioning.
    /// Kerker suppresses long-wavelength charge oscillations for faster convergence.
    pub fn mix_kerker_pulay(
        self: *PulayMixer,
        rho: []f64,
        rho_new: []const f64,
        beta: f64,
        grid: Grid,
        q0: f64,
        use_rfft: bool,
    ) !void {
        const n = rho.len;

        // Compute raw residual R = rho_new - rho
        const residual = try self.alloc.alloc(f64, n);
        errdefer self.alloc.free(residual);
        for (0..n) |i| {
            residual[i] = rho_new[i] - rho[i];
        }

        // Apply Kerker preconditioning to residual
        const precond_residual = try apply_kerker_preconditioner(
            self.alloc,
            grid,
            residual,
            q0,
            use_rfft,
        );
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
            const res_i = self.residual_history.items[i];
            for (0..m) |j| {
                const res_j = self.residual_history.items[j];
                var dot: f64 = 0.0;
                for (0..n) |k| {
                    dot += res_i[k] * res_j[k];
                }
                B[i * matrix_size + j] = dot;
            }
            B[i * matrix_size + m] = -1.0;
            B[m * matrix_size + i] = -1.0;
            rhs_vec[i] = 0.0;
        }
        B[m * matrix_size + m] = 0.0;
        rhs_vec[m] = -1.0;

        const coeffs = try solve_pulay_system(self.alloc, B, rhs_vec, matrix_size);
        defer self.alloc.free(coeffs);

        // Compute optimal density: rho = sum_i c_i * (rho_i + beta * PR_i)
        @memset(rho, 0.0);
        for (0..m) |i| {
            const c = coeffs[i];
            const rho_i = self.rho_history.items[i];
            const res_i = self.residual_history.items[i];
            for (0..n) |k| {
                rho[k] += c * (rho_i[k] + beta * res_i[k]);
            }
        }
    }
};

/// Apply Kerker preconditioner to a residual vector.
/// Returns a new allocated array with the preconditioned residual.
fn apply_kerker_preconditioner(
    alloc: std.mem.Allocator,
    grid: Grid,
    residual: []const f64,
    q0: f64,
    use_rfft: bool,
) ![]f64 {
    const q0_sq = q0 * q0;

    // FFT to reciprocal space
    const res_g = try fft_grid.real_to_reciprocal(alloc, grid, residual, use_rfft);
    defer alloc.free(res_g);

    // Apply Kerker kernel in reciprocal space
    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        const kerker = if (g.g2 > 1e-12) g.g2 / (g.g2 + q0_sq) else 0.0;
        res_g[g.idx] = math.complex.scale(res_g[g.idx], kerker);
    }

    // IFFT back to real space
    return fft_grid.reciprocal_to_real(alloc, grid, res_g);
}

/// Apply ABINIT-style model dielectric preconditioner to a G-space potential residual.
/// P(G) = (dielng^2 * |G|^2 + 1/diemac) / (dielng^2 * |G|^2 + 1)
/// G=0: P = 1/diemac.  G→∞: P → 1.
/// Modifies residual_g in place. No FFTs needed since input is already in G-space.
///
/// Corresponds to ABINIT's moddiel subroutine (m_prcref.F90) with diemix=1.
/// The mixing parameter (beta/diemix) is applied separately in PulayMixer.mix_with_residual().
///
/// Note: For metals with very large diemac (>1e4), G=0 is nearly zeroed (P≈0),
/// which may cause slow convergence of the macroscopic potential component.
pub fn apply_model_dielectric_preconditioner(
    grid: Grid,
    residual_g: []math.Complex,
    diemac: f64,
    dielng: f64,
) void {
    const dielng_sq = dielng * dielng;
    const inv_diemac = 1.0 / diemac;

    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.g2 < 1e-12) {
            residual_g[g.idx] = math.complex.scale(residual_g[g.idx], inv_diemac);
        } else {
            const x = dielng_sq * g.g2;
            const precond = (x + inv_diemac) / (x + 1.0);
            residual_g[g.idx] = math.complex.scale(residual_g[g.idx], precond);
        }
    }
}

/// Solve the Pulay linear system using simple Gaussian elimination.
fn solve_pulay_system(alloc: std.mem.Allocator, B: []f64, rhs: []f64, size: usize) ![]f64 {
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
    pub fn mix_with_residual(
        self: *ComplexPulayMixer,
        rho: []math.Complex,
        precond_residual: []math.Complex,
        beta: f64,
    ) !void {
        const n = rho.len;

        // Store current input and preconditioned residual
        const rho_copy = try self.alloc.alloc(math.Complex, n);
        @memcpy(rho_copy, rho);

        self.rho_history.append(self.alloc, rho_copy) catch |err| {
            self.alloc.free(rho_copy);
            return err;
        };
        self.residual_history.append(self.alloc, precond_residual) catch |err| {
            self.alloc.free(self.rho_history.pop().?);
            return err;
        };

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
            const res_i = self.residual_history.items[i];
            for (0..m) |j| {
                const res_j = self.residual_history.items[j];
                var dot: f64 = 0.0;
                for (0..n) |k| {
                    // Re(conj(a) * b) = a.r*b.r + a.i*b.i
                    dot += res_i[k].r * res_j[k].r + res_i[k].i * res_j[k].i;
                }
                B[i * matrix_size + j] = dot;
            }
            B[i * matrix_size + m] = -1.0;
            B[m * matrix_size + i] = -1.0;
            rhs[i] = 0.0;
        }
        B[m * matrix_size + m] = 0.0;
        rhs[m] = -1.0;

        const coeffs = try solve_pulay_system(self.alloc, B, rhs, matrix_size);
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
