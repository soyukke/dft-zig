//! All-electron atomic SCF solver.
//!
//! Self-consistent solution of the Kohn-Sham equations for a spherical atom:
//!   1. Solve radial Schrödinger eq for each orbital
//!   2. Build electron density ρ(r)
//!   3. Compute V_H(r) from radial Poisson equation
//!   4. Compute V_xc(r) from XC functional
//!   5. Mix potentials, iterate to convergence

const std = @import("std");
const dft_zig = @import("dft_zig");
const xc_mod = dft_zig.xc;
const RadialGrid = @import("radial_grid.zig").RadialGrid;
const schrodinger = @import("schrodinger.zig");

pub const AtomicOrbital = struct {
    n: u32,
    l: u32,
    occupation: f64, // total occupation (sum over m and spin)
    energy: f64, // eigenvalue (Ry)
};

pub const AtomConfig = struct {
    z: f64, // nuclear charge
    orbitals: []const OrbitalConfig,
    xc: xc_mod.Functional,
};

pub const OrbitalConfig = struct {
    n: u32,
    l: u32,
    occupation: f64,
};

const OrbitalState = struct {
    eigenvalues: []f64,
    wavefunctions: [][]f64,
    allocator: std.mem.Allocator,

    fn init(
        allocator: std.mem.Allocator,
        grid: *const RadialGrid,
        orbitals: []const OrbitalConfig,
        z: f64,
    ) !OrbitalState {
        const n = grid.n;
        const n_orb = orbitals.len;
        const eigenvalues = try allocator.alloc(f64, n_orb);
        errdefer allocator.free(eigenvalues);

        const wavefunctions = try allocator.alloc([]f64, n_orb);
        errdefer allocator.free(wavefunctions);

        var initialized_wavefunctions: usize = 0;
        errdefer {
            for (wavefunctions[0..initialized_wavefunctions]) |wf| {
                allocator.free(wf);
            }
        }

        for (orbitals, 0..) |orb, iorb| {
            wavefunctions[iorb] = try allocator.alloc(f64, n);
            initialized_wavefunctions += 1;
            @memset(wavefunctions[iorb], 0);

            const n_f: f64 = @floatFromInt(orb.n);
            eigenvalues[iorb] = -z * z / (n_f * n_f);
        }

        return .{
            .eigenvalues = eigenvalues,
            .wavefunctions = wavefunctions,
            .allocator = allocator,
        };
    }

    fn deinit(self: *OrbitalState) void {
        self.allocator.free(self.eigenvalues);
        for (self.wavefunctions) |wf| {
            self.allocator.free(wf);
        }
        self.allocator.free(self.wavefunctions);
    }
};

const ScfStep = struct {
    total_energy: f64,
    max_drho: f64,
};

const ScfWork = struct {
    rho: []f64,
    rho_new: []f64,
    v_eff: []f64,
    v_h: []f64,
    v_xc: []f64,
    v_coul: []f64,
    grad_rho: []f64,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, grid: *const RadialGrid, z: f64) !ScfWork {
        const n = grid.n;
        const rho = try allocator.alloc(f64, n);
        errdefer allocator.free(rho);
        const rho_new = try allocator.alloc(f64, n);
        errdefer allocator.free(rho_new);
        const v_eff = try allocator.alloc(f64, n);
        errdefer allocator.free(v_eff);
        const v_h = try allocator.alloc(f64, n);
        errdefer allocator.free(v_h);
        const v_xc = try allocator.alloc(f64, n);
        errdefer allocator.free(v_xc);
        const v_coul = try allocator.alloc(f64, n);
        errdefer allocator.free(v_coul);
        const grad_rho = try allocator.alloc(f64, n);
        errdefer allocator.free(grad_rho);

        init_coulomb_potential(grid, z, v_coul);
        @memcpy(v_eff, v_coul);
        init_density_guess(grid, z, rho);

        return .{
            .rho = rho,
            .rho_new = rho_new,
            .v_eff = v_eff,
            .v_h = v_h,
            .v_xc = v_xc,
            .v_coul = v_coul,
            .grad_rho = grad_rho,
            .allocator = allocator,
        };
    }

    fn deinit_all(self: *ScfWork) void {
        self.allocator.free(self.rho);
        self.allocator.free(self.v_eff);
        self.deinit_temps();
    }

    fn deinit_temps(self: *ScfWork) void {
        self.allocator.free(self.rho_new);
        self.allocator.free(self.v_h);
        self.allocator.free(self.v_xc);
        self.allocator.free(self.v_coul);
        self.allocator.free(self.grad_rho);
    }
};

pub const AtomResult = struct {
    /// Total energy (Ry)
    total_energy: f64,
    /// Orbital eigenvalues (Ry)
    eigenvalues: []f64,
    /// Electron density ρ(r)
    rho: []f64,
    /// Total effective potential V_eff(r)
    v_eff: []f64,
    /// Orbital wavefunctions u_nl(r) = r*R_nl(r)
    wavefunctions: [][]f64,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *AtomResult) void {
        self.allocator.free(self.eigenvalues);
        self.allocator.free(self.rho);
        self.allocator.free(self.v_eff);
        for (self.wavefunctions) |wf| {
            self.allocator.free(wf);
        }
        self.allocator.free(self.wavefunctions);
    }
};

/// Run self-consistent atomic DFT calculation.
pub fn solve(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    config: AtomConfig,
    max_iter: u32,
    mix_beta: f64,
    tol: f64,
) !AtomResult {
    var work = try ScfWork.init(allocator, grid, config.z);
    errdefer work.deinit_all();
    defer work.deinit_temps();

    var orbitals = try OrbitalState.init(allocator, grid, config.orbitals, config.z);
    errdefer orbitals.deinit();

    var total_energy: f64 = 0;
    var converged = false;

    for (0..max_iter) |iter| {
        const step = try run_scf_step(
            allocator,
            grid,
            config,
            mix_beta,
            work.rho,
            work.rho_new,
            work.v_eff,
            work.v_coul,
            work.v_h,
            work.v_xc,
            work.grad_rho,
            &orbitals,
        );
        total_energy = step.total_energy;

        if (iter % 10 == 0 or step.max_drho < tol) {
            std.log.info(
                "SCF iter {d:4}: E_total = {d:16.8} Ry, max|d_rho| = {e:10.3}",
                .{ iter, total_energy, step.max_drho },
            );
        }

        if (step.max_drho < tol) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        std.log.warn("atomic SCF did not converge", .{});
    }

    return .{
        .total_energy = total_energy,
        .eigenvalues = orbitals.eigenvalues,
        .rho = work.rho,
        .v_eff = work.v_eff,
        .wavefunctions = orbitals.wavefunctions,
        .allocator = allocator,
    };
}

fn init_coulomb_potential(grid: *const RadialGrid, z: f64, v_coul: []f64) void {
    for (0..grid.n) |i| {
        const r = grid.r[i];
        v_coul[i] = if (r > 1e-30) -2.0 * z / r else -2.0 * z / 1e-30;
    }
}

fn init_density_guess(grid: *const RadialGrid, z: f64, rho: []f64) void {
    for (0..grid.n) |i| {
        const r = grid.r[i];
        if (r > 1e-30) {
            rho[i] = z * @exp(-2.0 * z * r) * z * z / std.math.pi;
        } else {
            rho[i] = z * z * z / std.math.pi;
        }
    }
}

fn run_scf_step(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    config: AtomConfig,
    mix_beta: f64,
    rho: []f64,
    rho_new: []f64,
    v_eff: []f64,
    v_coul: []const f64,
    v_h: []f64,
    v_xc: []f64,
    grad_rho: []f64,
    orbitals: *OrbitalState,
) !ScfStep {
    try solve_orbitals(allocator, grid, config.orbitals, v_eff, orbitals);
    build_density(grid, config.orbitals, orbitals.wavefunctions, rho_new);
    const max_drho = mix_density(rho, rho_new, mix_beta);

    radial_poisson(grid, rho, v_h);
    compute_xc_potential(grid, config.xc, rho, grad_rho, v_xc);
    update_effective_potential(v_eff, v_coul, v_h, v_xc);

    return .{
        .total_energy = compute_total_energy(
            grid,
            config,
            rho,
            v_h,
            v_xc,
            orbitals.eigenvalues,
            grad_rho,
        ),
        .max_drho = max_drho,
    };
}

fn solve_orbitals(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    orbitals: []const OrbitalConfig,
    v_eff: []const f64,
    state: *OrbitalState,
) !void {
    for (orbitals, 0..) |orb, iorb| {
        const sol = try schrodinger.solve(
            allocator,
            grid,
            v_eff,
            .{ .n = orb.n, .l = orb.l },
            state.eigenvalues[iorb],
        );
        state.eigenvalues[iorb] = sol.energy;
        state.allocator.free(state.wavefunctions[iorb]);
        state.wavefunctions[iorb] = sol.u;
        state.allocator.free(sol.R);
    }
}

fn build_density(
    grid: *const RadialGrid,
    orbitals: []const OrbitalConfig,
    wavefunctions: [][]f64,
    rho_new: []f64,
) void {
    @memset(rho_new, 0);
    for (orbitals, wavefunctions) |orb, u| {
        const occ = orb.occupation;
        for (0..grid.n) |i| {
            const r = grid.r[i];
            const r2 = if (r > 1e-30) r * r else 1e-30;
            rho_new[i] += occ * u[i] * u[i] / (4.0 * std.math.pi * r2);
        }
    }
}

fn mix_density(rho: []f64, rho_new: []const f64, mix_beta: f64) f64 {
    var max_drho: f64 = 0;
    for (rho, rho_new) |*current, next| {
        const diff = next - current.*;
        max_drho = @max(max_drho, @abs(diff));
        current.* += mix_beta * diff;
    }
    return max_drho;
}

fn compute_xc_potential(
    grid: *const RadialGrid,
    xc: xc_mod.Functional,
    rho: []const f64,
    grad_rho: []f64,
    v_xc: []f64,
) void {
    compute_radial_gradient(grid, rho, grad_rho);
    for (rho, grad_rho, 0..) |rho_i, grad_i, i| {
        const g2 = grad_i * grad_i;
        const xc_point = xc_mod.eval_point(xc, rho_i, g2);
        v_xc[i] = xc_point.df_dn;
    }
}

fn update_effective_potential(
    v_eff: []f64,
    v_coul: []const f64,
    v_h: []const f64,
    v_xc: []const f64,
) void {
    for (v_eff, v_coul, v_h, v_xc) |*v_eff_i, v_coul_i, v_h_i, v_xc_i| {
        v_eff_i.* = v_coul_i + v_h_i + v_xc_i;
    }
}

/// Solve radial Poisson equation to get V_H(r) from ρ(r).
///
/// V_H(r) = 2 * [4π/r ∫₀ʳ ρ(r')r'² dr' + 4π ∫ᵣ^∞ ρ(r')r' dr']
///
/// Factor of 2 for Rydberg units.
/// Uses trapezoidal rule for both forward and backward integrals.
pub fn radial_poisson(grid: *const RadialGrid, rho: []const f64, v_h: []f64) void {
    const n = grid.n;

    // Forward integral: I_in(r_i) = ∫₀^{r_i} ρ(r') r'² dr'
    // Trapezoidal: I_in(i) = I_in(i-1) + 0.5*(f_{i-1} + f_i)
    // where f_i = ρ_i r_i² rab_i
    v_h[0] = 0;
    for (1..n) |i| {
        const f_prev = rho[i - 1] * grid.r[i - 1] * grid.r[i - 1] * grid.rab[i - 1];
        const f_curr = rho[i] * grid.r[i] * grid.r[i] * grid.rab[i];
        v_h[i] = v_h[i - 1] + 0.5 * (f_prev + f_curr);
    }

    // Backward integral: I_out(r_i) = ∫_{r_i}^{∞} ρ(r') r' dr'
    // Trapezoidal from right: I_out(i) = I_out(i+1) + 0.5*(f_{i+1} + f_i)
    var i_out: f64 = 0;
    var i: usize = n - 1;
    while (true) {
        const r_safe = if (grid.r[i] > 1e-30) grid.r[i] else 1e-30;
        // V_H(r) = 2 * 4π [I_in(r)/r + I_out(r)]
        v_h[i] = 2.0 * 4.0 * std.math.pi * (v_h[i] / r_safe + i_out);

        if (i == 0) break;
        // Trapezoidal accumulation for I_out
        const f_curr = rho[i] * grid.r[i] * grid.rab[i];
        const f_prev = rho[i - 1] * grid.r[i - 1] * grid.rab[i - 1];
        i_out += 0.5 * (f_curr + f_prev);
        i -= 1;
    }
}

/// Compute radial gradient dρ/dr using finite differences.
fn compute_radial_gradient(grid: *const RadialGrid, f: []const f64, grad: []f64) void {
    const n = grid.n;
    if (n < 3) return;

    // Forward difference at r=0
    grad[0] = (f[1] - f[0]) / grid.rab[0];

    // Central differences
    for (1..n - 1) |i| {
        const dr = 0.5 * (grid.rab[i - 1] + grid.rab[i + 1]);
        if (dr > 1e-30) {
            grad[i] = (f[i + 1] - f[i - 1]) / (2.0 * dr);
        } else {
            grad[i] = 0;
        }
    }

    // Backward difference at r_max
    grad[n - 1] = (f[n - 1] - f[n - 2]) / grid.rab[n - 1];
}

/// Compute total energy.
///
/// E_total = Σ_nl f_nl ε_nl - E_H + E_xc - ∫ V_xc ρ dr
fn compute_total_energy(
    grid: *const RadialGrid,
    config: AtomConfig,
    rho: []const f64,
    v_h: []const f64,
    v_xc: []const f64,
    eigenvalues: []const f64,
    grad_rho: []const f64,
) f64 {
    const n = grid.n;
    const n_orb = config.orbitals.len;

    // Band energy: Σ f ε
    var e_band: f64 = 0;
    for (0..n_orb) |iorb| {
        e_band += config.orbitals[iorb].occupation * eigenvalues[iorb];
    }

    // Hartree energy: E_H = (1/2) ∫ V_H ρ 4π r² dr
    var e_h: f64 = 0;
    for (0..n) |i| {
        const r = grid.r[i];
        const w = ctrap_weight(i, n);
        e_h += 0.5 * w * v_h[i] * rho[i] * 4.0 * std.math.pi * r * r * grid.rab[i];
    }

    // XC energy and ∫ V_xc ρ dr
    var e_xc: f64 = 0;
    var vxc_rho: f64 = 0;
    for (0..n) |i| {
        const r = grid.r[i];
        const w = ctrap_weight(i, n);
        const g2 = grad_rho[i] * grad_rho[i];
        const xc_point = xc_mod.eval_point(config.xc, rho[i], g2);
        const vol_elem = w * 4.0 * std.math.pi * r * r * grid.rab[i];
        e_xc += xc_point.f * vol_elem;
        vxc_rho += v_xc[i] * rho[i] * vol_elem;
    }

    return e_band - e_h + e_xc - vxc_rho;
}

/// Newton-Cotes endpoint correction.
fn ctrap_weight(i: usize, n: usize) f64 {
    const endpoint_weights = [5]f64{
        23.75 / 72.0,
        95.10 / 72.0,
        55.20 / 72.0,
        79.30 / 72.0,
        70.65 / 72.0,
    };
    if (n < 10) {
        if (i == 0 or i == n - 1) return 0.5;
        return 1.0;
    }
    if (i < 5) return endpoint_weights[i];
    if (i >= n - 5) return endpoint_weights[n - 1 - i];
    return 1.0;
}

// ============================================================================
// Tests
// ============================================================================

test "hydrogen atom total energy" {
    // Spin-unpolarized LDA-PZ for H atom.
    // Note: reference -0.958 Ry is LSDA (spin-polarized). Our spin-unpolarized
    // result is ~-0.89 Ry, which is correct for this approximation.
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const orbitals = [_]OrbitalConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0 },
    };

    var result = try solve(allocator, &grid, .{
        .z = 1,
        .orbitals = &orbitals,
        .xc = .lda_pz,
    }, 200, 0.3, 1e-10);
    defer result.deinit();

    try std.testing.expect(result.total_energy < -0.88);
    try std.testing.expect(result.total_energy > -0.92);
    try std.testing.expect(result.eigenvalues[0] < 0);
}

test "helium atom total energy" {
    // He (closed-shell): spin-unpolarized LDA is exact.
    // Reference: E(He, LDA-PZ) ≈ -5.670 Ry
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 50.0);
    defer grid.deinit();

    const orbitals = [_]OrbitalConfig{
        .{ .n = 1, .l = 0, .occupation = 2.0 },
    };

    var result = try solve(allocator, &grid, .{
        .z = 2,
        .orbitals = &orbitals,
        .xc = .lda_pz,
    }, 200, 0.3, 1e-10);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(-5.670, result.total_energy, 0.01);
}
