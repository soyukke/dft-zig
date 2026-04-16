//! Kohn-Sham DFT SCF for Gaussian-type orbital basis.
//!
//! Implements the Kohn-Sham equations using numerical integration
//! on a Becke molecular grid. Supports LDA (SVWN) and hybrid (B3LYP)
//! exchange-correlation functionals.
//!
//! Algorithm:
//!   1. Build one-electron integrals: S, T, V → H_core = T + V
//!   2. Build two-electron integrals: ERI table (for J and optional K)
//!   3. Build molecular grid (Becke partitioning + Lebedev angular + radial)
//!   4. Pre-evaluate basis functions on grid points
//!   5. Initial guess: diagonalize H_core
//!   6. SCF loop:
//!      a. Compute rho(r) and grad_rho(r) on grid from density matrix
//!      b. Evaluate XC functional on grid → eps_xc, v_xc, v_sigma
//!      c. Build Vxc matrix via numerical integration
//!      d. Build Fock matrix: F = H_core + J + alpha*K + Vxc
//!      e. Solve generalized eigenvalue problem: FC = SCε
//!      f. Build density matrix and check convergence
//!   7. Return converged energy
//!
//! Units: Hartree atomic units throughout.

const std = @import("std");
const Timer = @import("../../lib/timer.zig").Timer;
const math_mod = @import("../math/math.zig");
const basis_mod = @import("../basis/basis.zig");
const integrals = @import("../integrals/integrals.zig");
const obara_saika = integrals.obara_saika;
const libcint = integrals.libcint;
const linalg = @import("../linalg/linalg.zig");
const density_matrix = @import("density_matrix.zig");
const fock = @import("fock.zig");
const energy_mod = @import("energy.zig");
const diis_mod = @import("diis.zig");
const GtoDiis = diis_mod.GtoDiis;
const grid_mod = @import("../grid/grid.zig");
const becke = grid_mod.becke;
const xc_functionals = grid_mod.xc_functionals;
const blas = @import("../../lib/linalg/blas.zig");
const density_fitting_mod = @import("density_fitting.zig");
const DensityFittingContext = density_fitting_mod.DensityFittingContext;
const aux_basis_mod = basis_mod.aux_basis;

const ContractedShell = basis_mod.ContractedShell;
const AngularMomentum = basis_mod.AngularMomentum;
const GridPoint = becke.GridPoint;

/// Exchange-correlation functional type.
pub const XcFunctional = enum {
    /// LDA: Slater exchange + VWN5 correlation
    lda_svwn,
    /// B3LYP hybrid: 20% HF exchange
    b3lyp,
};

/// KS-DFT SCF parameters.
pub const KsParams = struct {
    /// Maximum number of SCF iterations.
    max_iter: usize = 100,
    /// Energy convergence threshold (Hartree).
    energy_threshold: f64 = 1e-8,
    /// Density matrix RMS convergence threshold.
    density_threshold: f64 = 1e-6,
    /// Enable DIIS acceleration.
    use_diis: bool = true,
    /// Maximum number of DIIS vectors.
    diis_max_vectors: usize = 6,
    /// Iteration at which DIIS starts.
    diis_start_iter: usize = 1,
    /// Exchange-correlation functional.
    xc_functional: XcFunctional = .lda_svwn,
    /// Number of radial grid points per atom.
    n_radial: usize = 99,
    /// Number of angular (Lebedev) grid points.
    n_angular: usize = 590,
    /// Whether to prune the angular grid.
    prune: bool = false,
    /// Use direct SCF with Schwarz screening instead of ERI table.
    /// This avoids O(N^4) memory and is essential for large basis sets.
    use_direct_scf: bool = false,
    /// Schwarz screening threshold for direct SCF.
    schwarz_threshold: f64 = 1e-12,
    /// Print SCF iteration details to stderr.
    verbose: bool = false,
    /// Use libcint for integral evaluation (1e and 2e).
    /// Requires libcint to be linked at build time.
    use_libcint: bool = false,
    /// Use density fitting (RI-J/K) for Coulomb and exchange matrices.
    /// Reduces scaling from O(N^4) to O(N^2 × N_aux).
    /// Requires auxiliary basis set (def2-universal-JKFIT).
    use_density_fitting: bool = false,
    /// Optional pre-built auxiliary shells for density fitting.
    /// If null and use_density_fitting is true, def2-universal-JKFIT is used.
    aux_shells: ?[]const ContractedShell = null,
};

/// Result of a converged KS-DFT calculation.
pub const KsResult = struct {
    /// Total energy (Hartree).
    total_energy: f64,
    /// Electronic energy (Hartree).
    electronic_energy: f64,
    /// Nuclear repulsion energy (Hartree).
    nuclear_repulsion: f64,
    /// One-electron energy: Tr[P * H_core].
    one_electron_energy: f64,
    /// Coulomb energy: 0.5 * Tr[P * J].
    coulomb_energy: f64,
    /// XC energy from grid integration.
    xc_energy: f64,
    /// HF exchange energy (only for hybrid functionals).
    hf_exchange_energy: f64,
    /// Orbital energies (ascending).
    orbital_energies: []f64,
    /// MO coefficients (column-major n×n).
    mo_coefficients: []f64,
    /// Density matrix (row-major n×n).
    density_matrix_result: []f64,
    /// Number of SCF iterations.
    iterations: usize,
    /// Whether converged.
    converged: bool,

    pub fn deinit(self: *KsResult, alloc: std.mem.Allocator) void {
        if (self.orbital_energies.len > 0) alloc.free(self.orbital_energies);
        if (self.mo_coefficients.len > 0) alloc.free(self.mo_coefficients);
        if (self.density_matrix_result.len > 0) alloc.free(self.density_matrix_result);
    }
};

/// Pre-computed basis function values on grid points.
/// For each grid point g and basis function mu:
///   phi[g * n_basis + mu] = phi_mu(r_g)
///   dphi_x[g * n_basis + mu] = d(phi_mu)/dx at r_g
///   (similarly for dphi_y, dphi_z)
pub const BasisOnGrid = struct {
    /// phi[g * n_basis + mu]: value of basis function mu at grid point g.
    phi: []f64,
    /// dphi_x[g * n_basis + mu]: x-derivative of basis function mu at grid point g.
    dphi_x: []f64,
    /// dphi_y[g * n_basis + mu].
    dphi_y: []f64,
    /// dphi_z[g * n_basis + mu].
    dphi_z: []f64,
    /// Number of grid points.
    n_grid: usize,
    /// Number of basis functions.
    n_basis: usize,

    pub fn deinit(self: *BasisOnGrid, alloc: std.mem.Allocator) void {
        alloc.free(self.phi);
        alloc.free(self.dphi_x);
        alloc.free(self.dphi_y);
        alloc.free(self.dphi_z);
    }
};

/// Evaluate a single contracted Cartesian Gaussian basis function at point r.
///
///   phi(r) = sum_i c_i * N(alpha_i, ax, ay, az) * (x-Cx)^ax * (y-Cy)^ay * (z-Cz)^az * exp(-alpha_i * |r-C|^2)
///
/// Also returns the gradient (dphi/dx, dphi/dy, dphi/dz).
pub fn evalBasisFunction(
    shell: ContractedShell,
    ang: AngularMomentum,
    rx: f64,
    ry: f64,
    rz: f64,
) struct { val: f64, dx: f64, dy: f64, dz: f64 } {
    const dx_c = rx - shell.center.x;
    const dy_c = ry - shell.center.y;
    const dz_c = rz - shell.center.z;
    const r2 = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c;

    // Angular part: x^ax * y^ay * z^az
    const x_pow = intPow(dx_c, ang.x);
    const y_pow = intPow(dy_c, ang.y);
    const z_pow = intPow(dz_c, ang.z);
    const angular = x_pow * y_pow * z_pow;

    // d(angular)/dx = ax * x^(ax-1) * y^ay * z^az, etc.
    const dang_dx = if (ang.x > 0) @as(f64, @floatFromInt(ang.x)) * intPow(dx_c, ang.x - 1) * y_pow * z_pow else 0.0;
    const dang_dy = if (ang.y > 0) x_pow * @as(f64, @floatFromInt(ang.y)) * intPow(dy_c, ang.y - 1) * z_pow else 0.0;
    const dang_dz = if (ang.z > 0) x_pow * y_pow * @as(f64, @floatFromInt(ang.z)) * intPow(dz_c, ang.z - 1) else 0.0;

    var val: f64 = 0.0;
    var grad_x: f64 = 0.0;
    var grad_y: f64 = 0.0;
    var grad_z: f64 = 0.0;

    for (shell.primitives) |prim| {
        const norm = basis_mod.normalization(prim.alpha, ang.x, ang.y, ang.z);
        const g = @exp(-prim.alpha * r2);
        const c_n_g = prim.coeff * norm * g;

        val += c_n_g * angular;

        // d(phi)/dx = c*N * [dang_dx * g + angular * (-2*alpha*dx) * g]
        grad_x += c_n_g * (dang_dx + angular * (-2.0 * prim.alpha * dx_c));
        grad_y += c_n_g * (dang_dy + angular * (-2.0 * prim.alpha * dy_c));
        grad_z += c_n_g * (dang_dz + angular * (-2.0 * prim.alpha * dz_c));
    }

    return .{ .val = val, .dx = grad_x, .dy = grad_y, .dz = grad_z };
}

/// Result of evaluating a basis function with second derivatives.
pub const BasisEvalHessian = struct {
    val: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    dxx: f64,
    dxy: f64,
    dxz: f64,
    dyy: f64,
    dyz: f64,
    dzz: f64,
};

/// Evaluate a single contracted Cartesian Gaussian basis function at point r,
/// returning the value, gradient, and Hessian (all 6 unique second derivatives).
///
/// phi(r) = sum_i c_i * N_i * A(r) * G_i(r)
/// where A = x^ax * y^ay * z^az, G = exp(-alpha * r2)
///
/// d²phi/dx_i dx_j = sum_i c_i * N_i * [
///   d²A/dx_i dx_j * G
///   + dA/dx_i * dG/dx_j
///   + dA/dx_j * dG/dx_i
///   + A * d²G/dx_i dx_j
/// ]
///
/// where dG/dx_i = -2*alpha*x_i*G
///       d²G/dx_i dx_j = (-2*alpha*delta_ij + 4*alpha²*x_i*x_j) * G
pub fn evalBasisFunctionWithHessian(
    shell: ContractedShell,
    ang: AngularMomentum,
    rx: f64,
    ry: f64,
    rz: f64,
) BasisEvalHessian {
    const dx_c = rx - shell.center.x;
    const dy_c = ry - shell.center.y;
    const dz_c = rz - shell.center.z;
    const r2 = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c;
    const coords = [3]f64{ dx_c, dy_c, dz_c };
    const angs = [3]u32{ ang.x, ang.y, ang.z };

    // Angular part: x^ax * y^ay * z^az
    var pows: [3]f64 = undefined;
    for (0..3) |i| {
        pows[i] = intPow(coords[i], angs[i]);
    }
    const angular = pows[0] * pows[1] * pows[2];

    // First derivatives of angular part: dA/dx_i
    var dang: [3]f64 = undefined;
    for (0..3) |i| {
        if (angs[i] > 0) {
            var prod: f64 = @floatFromInt(angs[i]);
            prod *= intPow(coords[i], angs[i] - 1);
            for (0..3) |j| {
                if (j != i) prod *= pows[j];
            }
            dang[i] = prod;
        } else {
            dang[i] = 0.0;
        }
    }

    // Second derivatives of angular part: d²A/dx_i dx_j
    // For i == j: d²A/dx_i² = ax*(ax-1)*x^(ax-2) * y^ay * z^az (if ax >= 2)
    // For i != j: d²A/dx_i dx_j = ax*x^(ax-1) * ay*y^(ay-1) * z^az (for x,y), etc.
    var d2ang: [3][3]f64 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            if (i == j) {
                // d²A/dx_i²
                if (angs[i] >= 2) {
                    var prod: f64 = @as(f64, @floatFromInt(angs[i])) * @as(f64, @floatFromInt(angs[i] - 1));
                    prod *= intPow(coords[i], angs[i] - 2);
                    for (0..3) |k| {
                        if (k != i) prod *= pows[k];
                    }
                    d2ang[i][j] = prod;
                } else {
                    d2ang[i][j] = 0.0;
                }
            } else {
                // d²A/dx_i dx_j  (i != j)
                // = (a_i * x_i^(a_i-1)) * (a_j * x_j^(a_j-1)) * product_{k != i,j} x_k^a_k
                if (angs[i] > 0 and angs[j] > 0) {
                    var prod: f64 = @as(f64, @floatFromInt(angs[i])) * intPow(coords[i], angs[i] - 1);
                    prod *= @as(f64, @floatFromInt(angs[j])) * intPow(coords[j], angs[j] - 1);
                    for (0..3) |k| {
                        if (k != i and k != j) prod *= pows[k];
                    }
                    d2ang[i][j] = prod;
                } else {
                    d2ang[i][j] = 0.0;
                }
            }
        }
    }

    var val: f64 = 0.0;
    var grad: [3]f64 = .{ 0.0, 0.0, 0.0 };
    var hess: [3][3]f64 = .{ .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 } };

    for (shell.primitives) |prim| {
        const norm = basis_mod.normalization(prim.alpha, ang.x, ang.y, ang.z);
        const g = @exp(-prim.alpha * r2);
        const c_n_g = prim.coeff * norm * g;
        const a2 = -2.0 * prim.alpha;

        val += c_n_g * angular;

        // Gradient: d(phi)/dx_i = c*N * [dA/dx_i * G + A * dG/dx_i]
        //                        = c*N*G * [dA/dx_i + A * (-2*alpha*x_i)]
        for (0..3) |i| {
            grad[i] += c_n_g * (dang[i] + angular * a2 * coords[i]);
        }

        // Hessian: d²phi/dx_i dx_j = c*N*G * [
        //   d²A/dx_i dx_j
        //   + dA/dx_i * (-2*alpha*x_j)
        //   + dA/dx_j * (-2*alpha*x_i)
        //   + A * (-2*alpha*delta_ij + 4*alpha²*x_i*x_j)
        // ]
        for (0..3) |i| {
            for (i..3) |j| {
                const delta_ij: f64 = if (i == j) 1.0 else 0.0;
                // d²G/dxi dxj / G = -2α δ_ij + 4α² xi xj
                const d2_gaussian = a2 * delta_ij + 4.0 * prim.alpha * prim.alpha * coords[i] * coords[j];
                const h_val = d2ang[i][j] + dang[i] * a2 * coords[j] + dang[j] * a2 * coords[i] + angular * d2_gaussian;
                hess[i][j] += c_n_g * h_val;
            }
        }
    }

    return .{
        .val = val,
        .dx = grad[0],
        .dy = grad[1],
        .dz = grad[2],
        .dxx = hess[0][0],
        .dxy = hess[0][1],
        .dxz = hess[0][2],
        .dyy = hess[1][1],
        .dyz = hess[1][2],
        .dzz = hess[2][2],
    };
}

/// Integer power: x^n for small non-negative n.
pub fn intPow(x: f64, n: u32) f64 {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    if (n == 2) return x * x;
    if (n == 3) return x * x * x;
    return std.math.pow(f64, x, @floatFromInt(n));
}

/// Per-basis-function precomputed data for fast grid evaluation.
/// Avoids recomputing normalization constants per grid point.
const BasisFuncInfo = struct {
    /// Shell center coordinates.
    cx: f64,
    cy: f64,
    cz: f64,
    /// Angular momentum exponents.
    ax: u32,
    ay: u32,
    az: u32,
    /// Precomputed coeff*norm and alpha for each primitive.
    /// These slices point into the flat precomputed arrays.
    coeff_norm: []const f64,
    alphas: []const f64,
    /// Screening radius squared: r² beyond which exp(-alpha_min*r²) < threshold.
    /// Used for grid-point sparsity screening.
    screen_r2: f64,
};

/// Pre-evaluate all basis functions (and their gradients) on all grid points.
/// Uses precomputed normalization constants and grid-point screening to skip
/// negligible basis functions (exp(-alpha_min*r²) < threshold).
pub fn evaluateBasisOnGrid(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    grid_points: []const GridPoint,
) !BasisOnGrid {
    const n_basis = obara_saika.totalBasisFunctions(shells);
    const n_grid = grid_points.len;
    const total = n_grid * n_basis;

    // Grid screening threshold: skip when exp(-alpha_min*r²) < this value.
    // -ln(1e-14) ≈ 32.24
    const screen_threshold_log: f64 = 32.24; // -ln(1e-14)

    const phi = try alloc.alloc(f64, total);
    const dphi_x = try alloc.alloc(f64, total);
    const dphi_y = try alloc.alloc(f64, total);
    const dphi_z = try alloc.alloc(f64, total);

    // Initialize all to zero (screened-out values default to 0)
    @memset(phi, 0.0);
    @memset(dphi_x, 0.0);
    @memset(dphi_y, 0.0);
    @memset(dphi_z, 0.0);

    // --- Precompute coeff*norm and alpha for each (basis function, primitive) pair ---
    // Count total number of (basis function, primitive) pairs.
    var total_pairs: usize = 0;
    for (shells) |shell| {
        const n_cart = basis_mod.numCartesian(shell.l);
        total_pairs += n_cart * shell.primitives.len;
    }

    const cn_flat = try alloc.alloc(f64, total_pairs);
    defer alloc.free(cn_flat);
    const alpha_flat = try alloc.alloc(f64, total_pairs);
    defer alloc.free(alpha_flat);

    const basis_info = try alloc.alloc(BasisFuncInfo, n_basis);
    defer alloc.free(basis_info);

    {
        var mu: usize = 0;
        var pair_off: usize = 0;
        for (shells) |shell| {
            const cart = basis_mod.cartesianExponents(shell.l);
            const n_cart = basis_mod.numCartesian(shell.l);

            // Find minimum alpha for this shell (most diffuse primitive)
            var alpha_min: f64 = shell.primitives[0].alpha;
            for (shell.primitives[1..]) |prim| {
                if (prim.alpha < alpha_min) alpha_min = prim.alpha;
            }
            // screen_r2: beyond this r², exp(-alpha_min*r²) < threshold
            const sr2 = screen_threshold_log / alpha_min;

            for (0..n_cart) |ic| {
                const ang = cart[ic];
                const n_prim = shell.primitives.len;
                for (shell.primitives, 0..) |prim, ip| {
                    cn_flat[pair_off + ip] = prim.coeff * basis_mod.normalization(prim.alpha, ang.x, ang.y, ang.z);
                    alpha_flat[pair_off + ip] = prim.alpha;
                }
                basis_info[mu] = .{
                    .cx = shell.center.x,
                    .cy = shell.center.y,
                    .cz = shell.center.z,
                    .ax = ang.x,
                    .ay = ang.y,
                    .az = ang.z,
                    .coeff_norm = cn_flat[pair_off .. pair_off + n_prim],
                    .alphas = alpha_flat[pair_off .. pair_off + n_prim],
                    .screen_r2 = sr2,
                };
                pair_off += n_prim;
                mu += 1;
            }
        }
    }

    // --- Evaluate on grid using precomputed tables with screening ---
    for (grid_points, 0..) |gp, ig| {
        const row_off = ig * n_basis;
        for (basis_info, 0..) |bi, mu| {
            const dx_c = gp.x - bi.cx;
            const dy_c = gp.y - bi.cy;
            const dz_c = gp.z - bi.cz;
            const r2 = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c;

            // Grid-point screening: skip if all primitives are negligible
            if (r2 > bi.screen_r2) {
                // phi/dphi already initialized to 0
                continue;
            }

            // Angular part
            const x_pow = intPow(dx_c, bi.ax);
            const y_pow = intPow(dy_c, bi.ay);
            const z_pow = intPow(dz_c, bi.az);
            const angular = x_pow * y_pow * z_pow;

            // Angular derivatives
            const dang_dx = if (bi.ax > 0) @as(f64, @floatFromInt(bi.ax)) * intPow(dx_c, bi.ax - 1) * y_pow * z_pow else 0.0;
            const dang_dy = if (bi.ay > 0) x_pow * @as(f64, @floatFromInt(bi.ay)) * intPow(dy_c, bi.ay - 1) * z_pow else 0.0;
            const dang_dz = if (bi.az > 0) x_pow * y_pow * @as(f64, @floatFromInt(bi.az)) * intPow(dz_c, bi.az - 1) else 0.0;

            // Contract primitives using precomputed coeff*norm
            var val: f64 = 0.0;
            var gx: f64 = 0.0;
            var gy: f64 = 0.0;
            var gz: f64 = 0.0;

            for (bi.coeff_norm, bi.alphas) |cn, alpha| {
                const g = @exp(-alpha * r2);
                const cng = cn * g;
                val += cng * angular;
                gx += cng * (dang_dx + angular * (-2.0 * alpha * dx_c));
                gy += cng * (dang_dy + angular * (-2.0 * alpha * dy_c));
                gz += cng * (dang_dz + angular * (-2.0 * alpha * dz_c));
            }

            const idx = row_off + mu;
            phi[idx] = val;
            dphi_x[idx] = gx;
            dphi_y[idx] = gy;
            dphi_z[idx] = gz;
        }
    }

    return .{
        .phi = phi,
        .dphi_x = dphi_x,
        .dphi_y = dphi_y,
        .dphi_z = dphi_z,
        .n_grid = n_grid,
        .n_basis = n_basis,
    };
}

/// Compute electron density and density gradient on the grid.
///
///   rho(r) = sum_{mu,nu} P_{mu,nu} * phi_mu(r) * phi_nu(r)
///   grad_rho(r) = 2 * sum_{mu,nu} P_{mu,nu} * grad(phi_mu(r)) * phi_nu(r)
///
/// Returns (rho, grad_rho_x, grad_rho_y, grad_rho_z) arrays of length n_grid.
pub fn computeDensityOnGrid(
    alloc: std.mem.Allocator,
    n_basis: usize,
    n_grid: usize,
    p_mat: []const f64,
    bog: BasisOnGrid,
) !struct { rho: []f64, grad_x: []f64, grad_y: []f64, grad_z: []f64 } {
    const rho = try alloc.alloc(f64, n_grid);
    const grad_x = try alloc.alloc(f64, n_grid);
    const grad_y = try alloc.alloc(f64, n_grid);
    const grad_z = try alloc.alloc(f64, n_grid);

    // Precompute P * phi using BLAS dgemm:
    // P_phi(n_grid × n_basis) = Phi(n_grid × n_basis) × P^T(n_basis × n_basis)
    // Since P is symmetric, P^T = P, so: P_phi = Phi * P
    //
    // In row-major: Phi[g,mu] = bog.phi[g * n_basis + mu]
    //               P[mu,nu] = p_mat[mu * n_basis + nu]
    //               P_phi[g,mu] = p_phi[g * n_basis + mu]
    const p_phi = try alloc.alloc(f64, n_grid * n_basis);
    defer alloc.free(p_phi);

    // dgemm: C = alpha * A * B + beta * C
    // A = Phi (n_grid × n_basis), B = P (n_basis × n_basis), C = P_phi (n_grid × n_basis)
    blas.dgemm(
        .no_trans, // A = Phi
        .no_trans, // B = P (symmetric, so no_trans = trans)
        n_grid, // m = rows of result
        n_basis, // n = cols of result
        n_basis, // k = inner dimension
        1.0, // alpha
        bog.phi, // A
        n_basis, // lda
        p_mat, // B
        n_basis, // ldb
        0.0, // beta
        p_phi, // C
        n_basis, // ldc
    );

    // Compute rho and grad_rho from P_phi and phi/dphi
    for (0..n_grid) |ig| {
        const g_off = ig * n_basis;
        var r: f64 = 0.0;
        var gx: f64 = 0.0;
        var gy: f64 = 0.0;
        var gz: f64 = 0.0;

        for (0..n_basis) |mu| {
            const pp = p_phi[g_off + mu];
            r += pp * bog.phi[g_off + mu];
            gx += pp * bog.dphi_x[g_off + mu];
            gy += pp * bog.dphi_y[g_off + mu];
            gz += pp * bog.dphi_z[g_off + mu];
        }

        rho[ig] = r;
        grad_x[ig] = 2.0 * gx;
        grad_y[ig] = 2.0 * gy;
        grad_z[ig] = 2.0 * gz;
    }

    return .{ .rho = rho, .grad_x = grad_x, .grad_y = grad_y, .grad_z = grad_z };
}

/// Build the XC contribution to the Fock matrix and compute the XC energy.
///
/// For LDA:
///   Vxc_{mu,nu} = sum_g w_g * v_xc(r_g) * phi_mu(r_g) * phi_nu(r_g)
///   E_xc = sum_g w_g * eps_xc(r_g) * rho(r_g)
///
/// For GGA:
///   Vxc_{mu,nu} = sum_g w_g * [v_xc * phi_mu * phi_nu
///                 + 2 * v_sigma * (grad_rho . (grad(phi_mu)*phi_nu + phi_mu*grad(phi_nu)))]
///   E_xc = sum_g w_g * eps_xc(r_g) * rho(r_g)
fn buildXcContribution(
    n_basis: usize,
    n_grid: usize,
    grid_points: []const GridPoint,
    bog: BasisOnGrid,
    rho_vals: []const f64,
    grad_rho_x: []const f64,
    grad_rho_y: []const f64,
    grad_rho_z: []const f64,
    xc_func: XcFunctional,
    vxc_mat: []f64,
    work_buf: []f64,
) f64 {
    // Zero out Vxc
    @memset(vxc_mat, 0.0);

    var e_xc: f64 = 0.0;

    // Step 1: Evaluate XC functional at all grid points and build
    // weighted basis function array H[g, mu]:
    //   H[g,mu] = 0.5*w*v_xc*phi[g,mu] + w*v_sigma*(grx*dphi_x + gry*dphi_y + grz*dphi_z)[g,mu]
    //
    // Then: Vxc = H^T * Phi + Phi^T * H = Tmp + Tmp^T where Tmp = H^T * Phi
    //
    // This replaces the O(n_grid * n_basis^2) scalar triple loop with
    // O(n_grid * n_basis) for building H + one dgemm call for the matrix multiply.

    // H buffer: n_grid × n_basis, stored in work_buf
    const h_buf = work_buf[0 .. n_grid * n_basis];
    @memset(h_buf, 0.0);

    for (0..n_grid) |ig| {
        const rho_g = rho_vals[ig];
        if (rho_g < 1e-20) continue;

        const w = grid_points[ig].w;
        const grx = grad_rho_x[ig];
        const gry = grad_rho_y[ig];
        const grz = grad_rho_z[ig];
        const sigma = grx * grx + gry * gry + grz * grz;

        // Evaluate XC functional
        var v_xc: f64 = undefined;
        var v_sigma: f64 = undefined;
        var eps_xc: f64 = undefined;

        switch (xc_func) {
            .lda_svwn => {
                const xc = xc_functionals.ldaSvwn(rho_g);
                eps_xc = xc.eps_xc;
                v_xc = xc.v_xc;
                v_sigma = 0.0;
            },
            .b3lyp => {
                const xc = xc_functionals.b3lyp(rho_g, sigma);
                eps_xc = xc.eps_xc;
                v_xc = xc.v_xc;
                v_sigma = xc.v_sigma;
            },
        }

        // Accumulate XC energy
        e_xc += w * eps_xc * rho_g;

        // Build H row for this grid point
        const g_off = ig * n_basis;
        const half_w_vxc = 0.5 * w * v_xc;
        const w_vsigma = w * v_sigma;

        if (w_vsigma != 0.0) {
            // GGA: H[g,mu] = 0.5*w*v_xc*phi + 2*w*v_sigma*(grx*dphi_x + gry*dphi_y + grz*dphi_z)
            // The factor 2 comes from the chain rule: d(sigma)/d(∇ρ) = 2*∇ρ
            // where sigma = |∇ρ|², so d(E_xc)/d(∇ρ) = v_sigma * 2*∇ρ
            const wvs_grx = 2.0 * w_vsigma * grx;
            const wvs_gry = 2.0 * w_vsigma * gry;
            const wvs_grz = 2.0 * w_vsigma * grz;
            for (0..n_basis) |mu| {
                h_buf[g_off + mu] = half_w_vxc * bog.phi[g_off + mu] +
                    wvs_grx * bog.dphi_x[g_off + mu] +
                    wvs_gry * bog.dphi_y[g_off + mu] +
                    wvs_grz * bog.dphi_z[g_off + mu];
            }
        } else {
            // LDA only: H[g,mu] = 0.5*w*v_xc*phi
            for (0..n_basis) |mu| {
                h_buf[g_off + mu] = half_w_vxc * bog.phi[g_off + mu];
            }
        }
    }

    // Step 2: Vxc = H^T * Phi (dgemm), then symmetrize: Vxc += Vxc^T
    // H is n_grid × n_basis (row-major), Phi is n_grid × n_basis (row-major)
    // We want Tmp = H^T * Phi: (n_basis × n_grid) * (n_grid × n_basis) = n_basis × n_basis
    // In row-major dgemm: C(m×n) = alpha * A^T(m×k) * B(k×n) + beta * C
    //   m = n_basis, n = n_basis, k = n_grid
    //   A = H (k×m in memory = n_grid × n_basis), transA = trans
    //   B = Phi (k×n in memory = n_grid × n_basis), transB = no_trans
    blas.dgemm(
        .trans, // H^T
        .no_trans, // Phi
        n_basis, // m = rows of result
        n_basis, // n = cols of result
        n_grid, // k = inner dimension
        1.0, // alpha
        h_buf, // A = H
        n_basis, // lda = n_basis (row stride of H)
        bog.phi, // B = Phi
        n_basis, // ldb = n_basis (row stride of Phi)
        0.0, // beta
        vxc_mat, // C = Vxc
        n_basis, // ldc
    );

    // Step 3: Symmetrize: Vxc = Tmp + Tmp^T
    for (0..n_basis) |mu| {
        for (mu + 1..n_basis) |nu| {
            const val = vxc_mat[mu * n_basis + nu] + vxc_mat[nu * n_basis + mu];
            vxc_mat[mu * n_basis + nu] = val;
            vxc_mat[nu * n_basis + mu] = val;
        }
        // Diagonal: Vxc[mu,mu] = 2 * Tmp[mu,mu] (from H^T*Phi, symmetric part)
        vxc_mat[mu * n_basis + mu] *= 2.0;
    }

    return e_xc;
}

/// Build the Coulomb (J) matrix from the density matrix and ERI table.
///   J_{mu,nu} = sum_{lam,sig} P_{lam,sig} * (mu nu | lam sig)
fn buildCoulombMatrix(
    n: usize,
    p_mat: []const f64,
    eri_table: obara_saika.GeneralEriTable,
    j_mat: []f64,
) void {
    for (0..n) |mu| {
        for (0..n) |nu| {
            var j_val: f64 = 0.0;
            for (0..n) |lam| {
                for (0..n) |sig| {
                    j_val += p_mat[lam * n + sig] * eri_table.get(mu, nu, lam, sig);
                }
            }
            j_mat[mu * n + nu] = j_val;
        }
    }
}

/// Build the exchange (K) matrix from the density matrix and ERI table.
///   K_{mu,nu} = sum_{lam,sig} P_{lam,sig} * (mu lam | nu sig)
fn buildExchangeMatrix(
    n: usize,
    p_mat: []const f64,
    eri_table: obara_saika.GeneralEriTable,
    k_mat: []f64,
) void {
    for (0..n) |mu| {
        for (0..n) |nu| {
            var k_val: f64 = 0.0;
            for (0..n) |lam| {
                for (0..n) |sig| {
                    k_val += p_mat[lam * n + sig] * eri_table.get(mu, lam, nu, sig);
                }
            }
            k_mat[mu * n + nu] = k_val;
        }
    }
}

/// Run a Kohn-Sham DFT SCF calculation.
pub fn runKohnShamScf(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    params: KsParams,
) !KsResult {
    std.debug.assert(n_electrons % 2 == 0);
    const n = obara_saika.totalBasisFunctions(shells);
    const n_occ = n_electrons / 2;

    // HF exchange fraction
    const hf_frac: f64 = switch (params.xc_functional) {
        .lda_svwn => 0.0,
        .b3lyp => 0.20,
    };

    // Step 1: Build one-electron integrals
    var timer = try Timer.start();
    if (params.verbose) std.debug.print("  [KS] Step 1: Building one-electron integrals (n={d}, libcint={})...\n", .{ n, params.use_libcint });

    // Initialize libcint data if enabled
    // Convert Vec3 positions to [3]f64 arrays for libcint
    const nuc_pos_flat = try alloc.alloc([3]f64, nuc_positions.len);
    defer alloc.free(nuc_pos_flat);
    for (nuc_positions, 0..) |pos, i| {
        nuc_pos_flat[i] = .{ pos.x, pos.y, pos.z };
    }

    var cint_data_storage: ?libcint.LibcintData = null;
    if (params.use_libcint) {
        cint_data_storage = libcint.LibcintData.init(alloc, shells, nuc_pos_flat, nuc_charges) catch null;
    }
    defer {
        if (cint_data_storage) |*cd| cd.deinit(alloc);
    }
    const use_libcint_actual = cint_data_storage != null;

    const s_mat = if (use_libcint_actual)
        try libcint.buildOverlapMatrix(alloc, cint_data_storage.?)
    else
        try obara_saika.buildOverlapMatrix(alloc, shells);
    defer alloc.free(s_mat);

    const t_mat = if (use_libcint_actual)
        try libcint.buildKineticMatrix(alloc, cint_data_storage.?)
    else
        try obara_saika.buildKineticMatrix(alloc, shells);
    defer alloc.free(t_mat);

    const v_mat = if (use_libcint_actual)
        try libcint.buildNuclearMatrix(alloc, cint_data_storage.?)
    else
        try obara_saika.buildNuclearMatrix(alloc, shells, nuc_positions, nuc_charges);
    defer alloc.free(v_mat);
    if (params.verbose) std.debug.print("  [KS] Step 1: Done. ({d:.2}s)\n", .{@as(f64, @floatFromInt(timer.read())) / 1e9});

    const h_core = try alloc.alloc(f64, n * n);
    defer alloc.free(h_core);
    for (0..n * n) |i| {
        h_core[i] = t_mat[i] + v_mat[i];
    }

    // Step 2: Build ERI table or Schwarz table
    timer.reset();
    if (params.verbose) std.debug.print("  [KS] Step 2: Building Schwarz/ERI table (direct={}, libcint={})...\n", .{ params.use_direct_scf, use_libcint_actual });
    var eri_table: ?obara_saika.GeneralEriTable = null;
    var schwarz_table: ?fock.SchwarzTable = null;
    var jk_builder: ?libcint.LibcintJKBuilder = null;
    if (use_libcint_actual) {
        // libcint path: build LibcintJKBuilder once (CINTOpt + Schwarz table).
        // Reused across all SCF iterations.
        jk_builder = try libcint.LibcintJKBuilder.init(alloc, cint_data_storage.?, params.schwarz_threshold);
    } else if (params.use_direct_scf) {
        schwarz_table = try fock.buildSchwarzTable(alloc, shells);
    } else {
        eri_table = try obara_saika.buildEriTable(alloc, shells);
    }
    if (params.verbose) std.debug.print("  [KS] Step 2: Done. ({d:.2}s)\n", .{@as(f64, @floatFromInt(timer.read())) / 1e9});
    defer {
        if (eri_table) |*et| et.deinit(alloc);
        if (schwarz_table) |*st| st.deinit(alloc);
        if (jk_builder) |*jkb| jkb.deinit(alloc);
    }

    // Step 2b: Build density fitting context if requested
    var df_context: ?DensityFittingContext = null;
    if (params.use_density_fitting) {
        timer.reset();
        if (params.verbose) std.debug.print("  [KS] Step 2b: Building density fitting context...\n", .{});

        var aux_buf_to_free: ?[]ContractedShell = null;
        const aux_shells = if (params.aux_shells) |as| as else blk: {
            // Build def2-universal-JKFIT auxiliary basis automatically
            const aux_buf = try alloc.alloc(ContractedShell, n * 10); // generous upper bound
            errdefer alloc.free(aux_buf);
            var aux_count: usize = 0;

            for (0..nuc_positions.len) |i| {
                const z: u32 = @intFromFloat(nuc_charges[i]);
                if (aux_basis_mod.buildDef2UniversalJkfit(z, nuc_positions[i])) |aux_result| {
                    for (aux_result.shells[0..aux_result.count]) |s| {
                        aux_buf[aux_count] = s;
                        aux_count += 1;
                    }
                }
            }
            aux_buf_to_free = aux_buf;
            break :blk aux_buf[0..aux_count];
        };

        df_context = try DensityFittingContext.init(alloc, shells, aux_shells);
        if (aux_buf_to_free) |buf| {
            alloc.free(buf);
        }

        if (params.verbose) std.debug.print("  [KS] Step 2b: Done (n_aux={d}). ({d:.2}s)\n", .{ df_context.?.n_aux, @as(f64, @floatFromInt(timer.read())) / 1e9 });
    }
    defer if (df_context) |*dfc| dfc.deinit();

    // Nuclear repulsion
    const v_nn = energy_mod.nuclearRepulsionEnergy(nuc_positions, nuc_charges);

    // Step 3: Build molecular grid
    const n_atoms = nuc_positions.len;
    const atoms = try alloc.alloc(becke.Atom, n_atoms);
    defer alloc.free(atoms);
    for (0..n_atoms) |i| {
        atoms[i] = .{
            .x = nuc_positions[i].x,
            .y = nuc_positions[i].y,
            .z = nuc_positions[i].z,
            .z_number = @intFromFloat(nuc_charges[i]),
        };
    }

    const grid_config = becke.GridConfig{
        .n_radial = params.n_radial,
        .n_angular = params.n_angular,
        .prune = params.prune,
        .use_atomic_radii = true,
        .becke_hardness = 3,
    };

    timer.reset();
    if (params.verbose) std.debug.print("  [KS] Step 3: Building molecular grid ({d} radial, {d} angular)...\n", .{ params.n_radial, params.n_angular });
    const grid_points = try becke.buildMolecularGrid(alloc, atoms, grid_config);
    defer alloc.free(grid_points);
    if (params.verbose) std.debug.print("  [KS] Step 3: Done ({d} grid points). ({d:.2}s)\n", .{ grid_points.len, @as(f64, @floatFromInt(timer.read())) / 1e9 });

    // Step 4: Pre-evaluate basis functions on grid
    timer.reset();
    if (params.verbose) std.debug.print("  [KS] Step 4: Pre-evaluating basis functions on grid...\n", .{});
    var bog = try evaluateBasisOnGrid(alloc, shells, grid_points);
    defer bog.deinit(alloc);
    if (params.verbose) std.debug.print("  [KS] Step 4: Done. ({d:.2}s)\n", .{@as(f64, @floatFromInt(timer.read())) / 1e9});

    // Step 5: Initial guess — diagonalize H_core
    timer.reset();
    if (params.verbose) std.debug.print("  [KS] Step 5: Initial guess (diagonalize H_core)...\n", .{});
    var eigen = try solveRoothaanHall(alloc, n, h_core, s_mat);
    if (params.verbose) {
        std.debug.print("  [KS] Step 5: Done. ({d:.2}s)\n", .{@as(f64, @floatFromInt(timer.read())) / 1e9});
        // Print initial orbital eigenvalues for comparison with PySCF
        std.debug.print("  [KS] Initial orbital eigenvalues (ALL {d}):\n", .{n});
        for (0..n) |i| {
            std.debug.print("    [{d:2}] {e:20.12}\n", .{ i, eigen.values[i] });
        }
        // Print S matrix eigenvalues for condition number check
        // Compute S eigenvalues using dsyev_ (standard eigenvalue problem)
        const s_copy = try alloc.alloc(f64, n * n);
        defer alloc.free(s_copy);
        @memcpy(s_copy, s_mat);
        const s_eigen = try solveStandardEigen(alloc, n, s_copy);
        defer {
            alloc.free(s_eigen.values);
            alloc.free(s_eigen.vectors);
        }
        std.debug.print("  [KS] S matrix eigenvalues:\n", .{});
        std.debug.print("    min = {e:20.12}\n", .{s_eigen.values[0]});
        std.debug.print("    max = {e:20.12}\n", .{s_eigen.values[n - 1]});
        std.debug.print("    condition number = {e:10.3}\n", .{s_eigen.values[n - 1] / s_eigen.values[0]});
        // Print first few and last few
        for (0..@min(n, 5)) |i| {
            std.debug.print("    [{d:2}] {e:20.12}\n", .{ i, s_eigen.values[i] });
        }
        if (n > 10) {
            std.debug.print("    ...\n", .{});
            for (n - 5..n) |i| {
                std.debug.print("    [{d:2}] {e:20.12}\n", .{ i, s_eigen.values[i] });
            }
        }
    }

    const p_mat = try density_matrix.buildDensityMatrix(alloc, n, n_occ, eigen.vectors);
    const p_old = try alloc.alloc(f64, n * n);
    defer alloc.free(p_old);
    @memset(p_old, 0.0);

    // Allocate work matrices
    const f_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(f_mat);

    const j_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(j_mat);

    const k_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(k_mat);

    const vxc_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(vxc_mat);

    // Work buffer for XC integration (H matrix: n_grid × n_basis)
    const xc_work_buf = try alloc.alloc(f64, grid_points.len * n);
    defer alloc.free(xc_work_buf);

    // DIIS
    var diis: ?GtoDiis = if (params.use_diis) GtoDiis.init(alloc, n, params.diis_max_vectors) else null;
    defer if (diis) |*d| d.deinit();

    const f_diis = if (params.use_diis) try alloc.alloc(f64, n * n) else null;
    defer if (f_diis) |buf| alloc.free(buf);

    // Step 6: SCF loop
    if (params.verbose) std.debug.print("  [KS] Step 6: Starting SCF loop (max_iter={d})...\n", .{params.max_iter});
    var e_total: f64 = 0.0;
    var e_old: f64 = 0.0;
    var converged = false;
    var iter: usize = 0;
    var final_e_xc: f64 = 0.0;
    var final_e_1e: f64 = 0.0;
    var final_e_j: f64 = 0.0;
    var final_e_k: f64 = 0.0;
    var scf_jk_ns: u64 = 0;
    var scf_xc_ns: u64 = 0;
    var scf_diag_ns: u64 = 0;

    while (iter < params.max_iter) : (iter += 1) {
        // Build J and K matrices
        timer.reset();
        if (df_context != null) {
            try df_context.?.buildJ(alloc, p_mat, j_mat);
            if (hf_frac > 0.0) {
                try df_context.?.buildK(alloc, p_mat, k_mat);
            } else {
                @memset(k_mat, 0.0);
            }
        } else if (use_libcint_actual) {
            const jk = try jk_builder.?.buildJK(alloc, p_mat);
            @memcpy(j_mat, jk.j_matrix);
            @memcpy(k_mat, jk.k_matrix);
            alloc.free(jk.j_matrix);
            alloc.free(jk.k_matrix);
            if (hf_frac == 0.0) {
                @memset(k_mat, 0.0);
            }
        } else if (params.use_direct_scf) {
            fock.buildJKDirect(n, p_mat, shells, &schwarz_table.?, params.schwarz_threshold, j_mat, k_mat);
            if (hf_frac == 0.0) {
                @memset(k_mat, 0.0);
            }
        } else {
            buildCoulombMatrix(n, p_mat, eri_table.?, j_mat);
            if (hf_frac > 0.0) {
                buildExchangeMatrix(n, p_mat, eri_table.?, k_mat);
            }
        }
        scf_jk_ns += timer.read();

        // Compute density on grid
        timer.reset();
        const dens = try computeDensityOnGrid(alloc, n, grid_points.len, p_mat, bog);
        defer {
            alloc.free(dens.rho);
            alloc.free(dens.grad_x);
            alloc.free(dens.grad_y);
            alloc.free(dens.grad_z);
        }

        // Build Vxc matrix and compute E_xc
        const e_xc = buildXcContribution(
            n,
            grid_points.len,
            grid_points,
            bog,
            dens.rho,
            dens.grad_x,
            dens.grad_y,
            dens.grad_z,
            params.xc_functional,
            vxc_mat,
            xc_work_buf,
        );
        scf_xc_ns += timer.read();

        // Build Fock matrix: F = H_core + J - 0.5 * hf_frac * K + Vxc
        // The factor of 0.5 on K comes from the closed-shell RHF antisymmetry:
        // in closed-shell, the two-electron part is G = J - 0.5*K, so for
        // hybrid DFT we replace the full exchange with a fraction c_x:
        // G = J - c_x * 0.5 * K + Vxc
        for (0..n * n) |i| {
            f_mat[i] = h_core[i] + j_mat[i] + vxc_mat[i];
            if (hf_frac > 0.0) {
                f_mat[i] -= 0.5 * hf_frac * k_mat[i];
            }
        }

        // Compute energy components
        // E_1e = Tr[P * H_core]
        var e_1e: f64 = 0.0;
        for (0..n) |mu| {
            for (0..n) |nu| {
                e_1e += p_mat[mu * n + nu] * h_core[mu * n + nu];
            }
        }

        // E_J = 0.5 * Tr[P * J]
        var e_j: f64 = 0.0;
        for (0..n) |mu| {
            for (0..n) |nu| {
                e_j += p_mat[mu * n + nu] * j_mat[mu * n + nu];
            }
        }
        e_j *= 0.5;

        // E_K = -0.25 * hf_frac * Tr[P * K]
        // The factor 0.25 = 0.5 (from closed-shell G matrix) * 0.5 (from E = 0.5*Tr[P*(H+F)])
        var e_k: f64 = 0.0;
        if (hf_frac > 0.0) {
            for (0..n) |mu| {
                for (0..n) |nu| {
                    e_k += p_mat[mu * n + nu] * k_mat[mu * n + nu];
                }
            }
            e_k *= -0.25 * hf_frac;
        }

        e_total = e_1e + e_j + e_k + e_xc + v_nn;

        // Store for final output
        final_e_xc = e_xc;
        final_e_1e = e_1e;
        final_e_j = e_j;
        final_e_k = e_k;

        // Check convergence
        const delta_e = @abs(e_total - e_old);
        const rms_p = density_matrix.densityRmsDiff(n, p_mat, p_old);

        if (params.verbose) {
            std.debug.print("  SCF iter {d:3}: E = {d:20.12}  dE = {e:10.3}  dP = {e:10.3}\n", .{ iter, e_total, delta_e, rms_p });
            if (iter == 0) {
                std.debug.print("    E_1e = {d:20.12}  E_J = {d:20.12}  E_K = {d:20.12}  E_XC = {d:20.12}  V_nn = {d:20.12}\n", .{ e_1e, e_j, e_k, e_xc, v_nn });
                // Print Tr(P), Tr(P*S)
                var tr_p: f64 = 0.0;
                var tr_ps: f64 = 0.0;
                for (0..n) |imu| {
                    tr_p += p_mat[imu * n + imu];
                    for (0..n) |inu| {
                        tr_ps += p_mat[imu * n + inu] * s_mat[imu * n + inu];
                    }
                }
                std.debug.print("    Tr(P) = {d:20.12}  Tr(P*S) = {d:20.12}\n", .{ tr_p, tr_ps });
            }
        }

        if (iter > 0 and delta_e < params.energy_threshold and rms_p < params.density_threshold) {
            converged = true;
            break;
        }

        e_old = e_total;
        @memcpy(p_old, p_mat);

        // Apply DIIS
        const f_to_diag = if (diis != null and iter >= params.diis_start_iter) blk: {
            try diis.?.extrapolate(f_mat, p_mat, s_mat, f_diis.?);
            break :blk f_diis.?;
        } else f_mat;

        alloc.free(eigen.vectors);
        alloc.free(eigen.values);

        timer.reset();
        eigen = try solveRoothaanHall(alloc, n, f_to_diag, s_mat);
        scf_diag_ns += timer.read();

        density_matrix.updateDensityMatrix(n, n_occ, eigen.vectors, p_mat);
    }

    if (params.verbose) {
        std.debug.print("  [KS] SCF timing: J/K={d:.2}s  XC={d:.2}s  diag={d:.2}s\n", .{
            @as(f64, @floatFromInt(scf_jk_ns)) / 1e9,
            @as(f64, @floatFromInt(scf_xc_ns)) / 1e9,
            @as(f64, @floatFromInt(scf_diag_ns)) / 1e9,
        });
    }

    // If not converged, do final energy evaluation
    if (!converged) {
        if (df_context != null) {
            try df_context.?.buildJ(alloc, p_mat, j_mat);
            if (hf_frac > 0.0) {
                try df_context.?.buildK(alloc, p_mat, k_mat);
            } else {
                @memset(k_mat, 0.0);
            }
        } else if (use_libcint_actual) {
            const jk = try jk_builder.?.buildJK(alloc, p_mat);
            @memcpy(j_mat, jk.j_matrix);
            @memcpy(k_mat, jk.k_matrix);
            alloc.free(jk.j_matrix);
            alloc.free(jk.k_matrix);
            if (hf_frac == 0.0) {
                @memset(k_mat, 0.0);
            }
        } else if (params.use_direct_scf) {
            fock.buildJKDirect(n, p_mat, shells, &schwarz_table.?, params.schwarz_threshold, j_mat, k_mat);
            if (hf_frac == 0.0) {
                @memset(k_mat, 0.0);
            }
        } else {
            buildCoulombMatrix(n, p_mat, eri_table.?, j_mat);
            if (hf_frac > 0.0) {
                buildExchangeMatrix(n, p_mat, eri_table.?, k_mat);
            }
        }

        const dens = try computeDensityOnGrid(alloc, n, grid_points.len, p_mat, bog);
        defer {
            alloc.free(dens.rho);
            alloc.free(dens.grad_x);
            alloc.free(dens.grad_y);
            alloc.free(dens.grad_z);
        }

        final_e_xc = buildXcContribution(
            n,
            grid_points.len,
            grid_points,
            bog,
            dens.rho,
            dens.grad_x,
            dens.grad_y,
            dens.grad_z,
            params.xc_functional,
            vxc_mat,
            xc_work_buf,
        );

        final_e_1e = 0.0;
        for (0..n) |mu| {
            for (0..n) |nu| {
                final_e_1e += p_mat[mu * n + nu] * h_core[mu * n + nu];
            }
        }

        final_e_j = 0.0;
        for (0..n) |mu| {
            for (0..n) |nu| {
                final_e_j += p_mat[mu * n + nu] * j_mat[mu * n + nu];
            }
        }
        final_e_j *= 0.5;

        final_e_k = 0.0;
        if (hf_frac > 0.0) {
            for (0..n) |mu| {
                for (0..n) |nu| {
                    final_e_k += p_mat[mu * n + nu] * k_mat[mu * n + nu];
                }
            }
            final_e_k *= -0.25 * hf_frac;
        }

        e_total = final_e_1e + final_e_j + final_e_k + final_e_xc + v_nn;
    }

    return KsResult{
        .total_energy = e_total,
        .electronic_energy = e_total - v_nn,
        .nuclear_repulsion = v_nn,
        .one_electron_energy = final_e_1e,
        .coulomb_energy = final_e_j,
        .xc_energy = final_e_xc,
        .hf_exchange_energy = final_e_k,
        .orbital_energies = eigen.values,
        .mo_coefficients = eigen.vectors,
        .density_matrix_result = p_mat,
        .iterations = iter,
        .converged = converged,
    };
}

/// Solve the Roothaan-Hall equation FC = SCε using canonical orthogonalization.
///
/// Instead of dsygv_ (which uses Cholesky decomposition of S), we use:
///   1. Diagonalize S: S = U * diag(s) * U^T
///   2. Form S^{-1/2} = U * diag(1/sqrt(s)) * U^T
///   3. Transform: F' = S^{-1/2} * F * S^{-1/2}
///   4. Solve standard eigenproblem: F' * C' = C' * diag(ε)
///   5. Back-transform: C = S^{-1/2} * C'
///
/// This is more robust than dsygv_ for near-linearly-dependent basis sets.
fn solveRoothaanHall(
    alloc: std.mem.Allocator,
    n: usize,
    f_mat: []const f64,
    s_mat: []const f64,
) !linalg.RealEigenDecomp {
    // Step 1: Diagonalize S matrix
    const s_copy = try alloc.alloc(f64, n * n);
    defer alloc.free(s_copy);
    @memcpy(s_copy, s_mat);

    const s_eigen = try linalg.realSymmetricEigenDecomp(alloc, .accelerate, n, s_copy);
    defer alloc.free(s_eigen.values);
    // s_eigen.vectors is column-major: eigenvector j is at s_eigen.vectors[j*n .. (j+1)*n]
    defer alloc.free(s_eigen.vectors);

    // Step 2: Build S^{-1/2} = U * diag(1/sqrt(s)) * U^T
    // Threshold for linear dependence
    const threshold: f64 = 1e-8;
    var n_indep: usize = 0;
    for (0..n) |i| {
        if (s_eigen.values[i] > threshold) n_indep += 1;
    }

    // Build X = U * diag(1/sqrt(s)) for independent eigenvectors only
    // X is n x n_indep, stored column-major
    const x_mat = try alloc.alloc(f64, n * n_indep);
    defer alloc.free(x_mat);

    var col: usize = 0;
    for (0..n) |i| {
        if (s_eigen.values[i] > threshold) {
            const inv_sqrt_s = 1.0 / @sqrt(s_eigen.values[i]);
            // Column i of U (eigenvector i) is at s_eigen.vectors[i*n .. (i+1)*n]
            for (0..n) |mu| {
                x_mat[col * n + mu] = s_eigen.vectors[i * n + mu] * inv_sqrt_s;
            }
            col += 1;
        }
    }

    // Step 3: F' = X^T * F * X  (n_indep x n_indep)
    // First compute temp = F * X (n x n_indep)
    const temp = try alloc.alloc(f64, n * n_indep);
    defer alloc.free(temp);

    for (0..n_indep) |j| {
        for (0..n) |mu| {
            var sum: f64 = 0.0;
            for (0..n) |nu| {
                sum += f_mat[mu * n + nu] * x_mat[j * n + nu];
            }
            temp[j * n + mu] = sum;
        }
    }

    // F' = X^T * temp  (n_indep x n_indep), stored as flat array
    const f_prime = try alloc.alloc(f64, n_indep * n_indep);
    defer alloc.free(f_prime);

    for (0..n_indep) |i| {
        for (0..n_indep) |j| {
            var sum: f64 = 0.0;
            for (0..n) |mu| {
                sum += x_mat[i * n + mu] * temp[j * n + mu];
            }
            // Store in both row-major and column-major (symmetric)
            f_prime[i * n_indep + j] = sum;
        }
    }

    // Step 4: Solve standard eigenvalue problem for F'
    var eigen_prime = try linalg.realSymmetricEigenDecomp(alloc, .accelerate, n_indep, f_prime);
    // eigen_prime.vectors is column-major: eigenvector j at [j*n_indep .. (j+1)*n_indep]
    defer alloc.free(eigen_prime.vectors);

    // Step 5: Back-transform C = X * C'
    // Result: n x n matrix in column-major order
    // We need to produce n x n output (padded if n_indep < n)
    const result_values = try alloc.alloc(f64, n);
    errdefer alloc.free(result_values);
    const result_vectors = try alloc.alloc(f64, n * n);
    errdefer alloc.free(result_vectors);

    // Fill eigenvalues: first n_indep from the eigensolver, rest = very large
    for (0..n_indep) |i| {
        result_values[i] = eigen_prime.values[i];
    }
    for (n_indep..n) |i| {
        result_values[i] = 1e10; // mark dependent vectors with large energy
    }

    // Back-transform eigenvectors: C_mu,j = sum_k X_mu,k * C'_k,j
    @memset(result_vectors, 0.0);
    for (0..n_indep) |j| {
        for (0..n) |mu| {
            var sum: f64 = 0.0;
            for (0..n_indep) |k| {
                sum += x_mat[k * n + mu] * eigen_prime.vectors[j * n_indep + k];
            }
            result_vectors[j * n + mu] = sum;
        }
    }

    alloc.free(eigen_prime.values);
    eigen_prime.values = &.{}; // prevent double free in defer

    return linalg.RealEigenDecomp{
        .values = result_values,
        .vectors = result_vectors,
        .n = n,
    };
}

/// Solve the standard eigenvalue problem A·x = λ·x for a real symmetric matrix.
/// Used for computing S matrix eigenvalues (condition number check).
fn solveStandardEigen(
    alloc: std.mem.Allocator,
    n: usize,
    a: []f64,
) !linalg.RealEigenDecomp {
    return try linalg.realSymmetricEigenDecomp(alloc, .accelerate, n, a);
}

// ==========================================================================
// Tests
// ==========================================================================

test "basis function evaluation s-type at center" {
    const sto3g = @import("../basis/sto3g.zig");
    const center = math_mod.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const shell = ContractedShell{
        .center = center,
        .l = 0,
        .primitives = &sto3g.H_1s,
    };
    const ang = AngularMomentum{ .x = 0, .y = 0, .z = 0 };

    // At center, exp(-alpha*0) = 1, so phi = sum(c_i * N_i)
    const result = evalBasisFunction(shell, ang, 0.0, 0.0, 0.0);
    // Value should be positive and nonzero
    try std.testing.expect(result.val > 0.0);
    // Gradient at center of s-type should be zero by symmetry
    try std.testing.expectApproxEqAbs(0.0, result.dx, 1e-14);
    try std.testing.expectApproxEqAbs(0.0, result.dy, 1e-14);
    try std.testing.expectApproxEqAbs(0.0, result.dz, 1e-14);
}

test "basis function evaluation p-type" {
    const sto3g = @import("../basis/sto3g.zig");
    const center = math_mod.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const shell = ContractedShell{
        .center = center,
        .l = 1,
        .primitives = &sto3g.O_2p,
    };

    // px at center: phi = x * sum(c_i * N_i * exp(-alpha*0)) = 0 (since x=0)
    const ang_px = AngularMomentum{ .x = 1, .y = 0, .z = 0 };
    const result = evalBasisFunction(shell, ang_px, 0.0, 0.0, 0.0);
    try std.testing.expectApproxEqAbs(0.0, result.val, 1e-14);
    // dphi_px/dx at center = sum(c_i * N_i) (nonzero)
    try std.testing.expect(@abs(result.dx) > 0.0);
    // dphi_px/dy and dphi_px/dz at center = 0
    try std.testing.expectApproxEqAbs(0.0, result.dy, 1e-14);
    try std.testing.expectApproxEqAbs(0.0, result.dz, 1e-14);
}

test "density integration equals n_electrons (H2O STO-3G RHF)" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");
    const gto_scf = @import("gto_scf.zig");

    // H2O geometry in bohr
    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    // First get converged RHF density
    var rhf_result = try gto_scf.runGeneralRhfScf(alloc, &shells, &nuc_positions, &nuc_charges, 10, .{});
    defer rhf_result.deinit(alloc);

    const n = obara_saika.totalBasisFunctions(&shells);

    // Build a grid
    const atoms = [_]becke.Atom{
        .{ .x = 0.0, .y = 0.0, .z = 0.0, .z_number = 8 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351, .z_number = 1 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351, .z_number = 1 },
    };

    const grid_config = becke.GridConfig{
        .n_radial = 75,
        .n_angular = 302,
        .prune = false,
    };

    const grid_points = try becke.buildMolecularGrid(alloc, &atoms, grid_config);
    defer alloc.free(grid_points);

    // Evaluate basis on grid
    var bog = try evaluateBasisOnGrid(alloc, &shells, grid_points);
    defer bog.deinit(alloc);

    // Compute density on grid
    const dens = try computeDensityOnGrid(alloc, n, grid_points.len, rhf_result.density_matrix, bog);
    defer {
        alloc.free(dens.rho);
        alloc.free(dens.grad_x);
        alloc.free(dens.grad_y);
        alloc.free(dens.grad_z);
    }

    // Integrate density: should give n_electrons = 10
    var n_elec: f64 = 0.0;
    for (0..grid_points.len) |ig| {
        n_elec += grid_points[ig].w * dens.rho[ig];
    }

    std.debug.print("\nDensity integration test:\n", .{});
    std.debug.print("  Grid points: {d}\n", .{grid_points.len});
    std.debug.print("  Integrated electrons: {d:.6}\n", .{n_elec});
    std.debug.print("  Expected: 10.0\n", .{});

    try testing.expectApproxEqAbs(10.0, n_elec, 0.01);
}

test "KS-DFT H2O STO-3G LDA (SVWN)" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    // H2O geometry in bohr
    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    var result = try runKohnShamScf(alloc, &shells, &nuc_positions, &nuc_charges, 10, .{
        .xc_functional = .lda_svwn,
        .n_radial = 99,
        .n_angular = 590,
        .prune = false,
    });
    defer result.deinit(alloc);

    std.debug.print("\nH2O STO-3G KS-DFT LDA (SVWN):\n", .{});
    std.debug.print("  Total energy:     {d:.10} Ha\n", .{result.total_energy});
    std.debug.print("  1e energy:        {d:.10} Ha\n", .{result.one_electron_energy});
    std.debug.print("  Coulomb energy:   {d:.10} Ha\n", .{result.coulomb_energy});
    std.debug.print("  XC energy:        {d:.10} Ha\n", .{result.xc_energy});
    std.debug.print("  Nuclear repulsion:{d:.10} Ha\n", .{result.nuclear_repulsion});
    std.debug.print("  Iterations:       {d}\n", .{result.iterations});
    std.debug.print("  Converged:        {}\n", .{result.converged});
    std.debug.print("  PySCF reference:  -74.7321048790 Ha\n", .{});

    try testing.expect(result.converged);
    // PySCF grid-converged LDA: -74.7321048790
    try testing.expectApproxEqAbs(-74.7321048790, result.total_energy, 1e-3);
}

test "KS-DFT H2O STO-3G B3LYP" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    var result = try runKohnShamScf(alloc, &shells, &nuc_positions, &nuc_charges, 10, .{
        .xc_functional = .b3lyp,
        .n_radial = 99,
        .n_angular = 590,
        .prune = false,
    });
    defer result.deinit(alloc);

    std.debug.print("\nH2O STO-3G KS-DFT B3LYP:\n", .{});
    std.debug.print("  Total energy:     {d:.10} Ha\n", .{result.total_energy});
    std.debug.print("  1e energy:        {d:.10} Ha\n", .{result.one_electron_energy});
    std.debug.print("  Coulomb energy:   {d:.10} Ha\n", .{result.coulomb_energy});
    std.debug.print("  XC energy:        {d:.10} Ha\n", .{result.xc_energy});
    std.debug.print("  HF exchange:      {d:.10} Ha\n", .{result.hf_exchange_energy});
    std.debug.print("  Nuclear repulsion:{d:.10} Ha\n", .{result.nuclear_repulsion});
    std.debug.print("  Iterations:       {d}\n", .{result.iterations});
    std.debug.print("  Converged:        {}\n", .{result.converged});
    std.debug.print("  PySCF reference:  -75.3125872072 Ha\n", .{});

    try testing.expect(result.converged);
    // PySCF grid-converged B3LYP: -75.3125872072
    try testing.expectApproxEqAbs(-75.3125872072, result.total_energy, 1e-3);
}

// ============================================================================
// QM9 validation tests: B3LYP/6-31G(2df,p) vs PySCF
// ============================================================================
// These tests use Direct SCF (Schwarz screening) because the 6-31G(2df,p)
// basis is too large for in-memory ERI tables. They are slower than STO-3G
// tests but validate the full QM9-level calculation pipeline.

test "QM9 validation: H2O B3LYP/6-31G(2df,p) vs PySCF" {
    const alloc = std.testing.allocator;
    const b = @import("../basis/basis631g_2dfp.zig");

    // H2O geometry in Bohr (from PySCF mol.atom_coords())
    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.2217971552 }, // O
        .{ .x = 0.0, .y = 1.4308250325, .z = -0.8871886210 }, // H
        .{ .x = 0.0, .y = -1.4308250325, .z = -0.8871886210 }, // H
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    // Build 6-31G(2df,p) shells
    const o_data = b.buildAtomShells(8, nuc_positions[0]).?;
    const h1_data = b.buildAtomShells(1, nuc_positions[1]).?;
    const h2_data = b.buildAtomShells(1, nuc_positions[2]).?;

    var all_shells: [b.MAX_SHELLS_PER_ATOM * 3]ContractedShell = undefined;
    var count: usize = 0;
    for (o_data.shells[0..o_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h1_data.shells[0..h1_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h2_data.shells[0..h2_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    const shells = all_shells[0..count];

    std.debug.print("\n=== QM9 validation: H2O B3LYP/6-31G(2df,p) ===\n", .{});
    std.debug.print("  Shells: {d}, Basis functions: {d}\n", .{ count, obara_saika.totalBasisFunctions(shells) });

    var result = try runKohnShamScf(alloc, shells, &nuc_positions, &nuc_charges, 10, .{
        .xc_functional = .b3lyp,
        .n_radial = 50,
        .n_angular = 302,
        .prune = false,
        .use_direct_scf = true,
        .schwarz_threshold = 1e-10,
    });
    defer result.deinit(alloc);

    std.debug.print("  Total energy:     {d:.12} Ha\n", .{result.total_energy});
    std.debug.print("  Iterations:       {d}\n", .{result.iterations});
    std.debug.print("  Converged:        {}\n", .{result.converged});
    std.debug.print("  PySCF reference:  -76.425747169883 Ha\n", .{});
    std.debug.print("  Diff:             {e:.6}\n", .{@abs(result.total_energy - (-76.425747169883))});

    try std.testing.expect(result.converged);
    // PySCF B3LYP/6-31G(2df,p) cart=True: -76.425747169883 Ha
    // Using 50/302 grid: expect ~1e-3 accuracy
    try std.testing.expectApproxEqAbs(@as(f64, -76.425747169883), result.total_energy, 5e-3);
}

test "QM9 validation: CH4 B3LYP/6-31G(2df,p) vs PySCF" {
    const alloc = std.testing.allocator;
    const b = @import("../basis/basis631g_2dfp.zig");

    // CH4 geometry in Bohr (from PySCF mol.atom_coords())
    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 }, // C
        .{ .x = 1.1888607200, .y = 1.1888607200, .z = 1.1888607200 }, // H
        .{ .x = -1.1888607200, .y = -1.1888607200, .z = 1.1888607200 }, // H
        .{ .x = -1.1888607200, .y = 1.1888607200, .z = -1.1888607200 }, // H
        .{ .x = 1.1888607200, .y = -1.1888607200, .z = -1.1888607200 }, // H
    };
    const nuc_charges = [_]f64{ 6.0, 1.0, 1.0, 1.0, 1.0 };

    // Build 6-31G(2df,p) shells
    const c_data = b.buildAtomShells(6, nuc_positions[0]).?;
    const h1_data = b.buildAtomShells(1, nuc_positions[1]).?;
    const h2_data = b.buildAtomShells(1, nuc_positions[2]).?;
    const h3_data = b.buildAtomShells(1, nuc_positions[3]).?;
    const h4_data = b.buildAtomShells(1, nuc_positions[4]).?;

    var all_shells: [b.MAX_SHELLS_PER_ATOM * 5]ContractedShell = undefined;
    var count: usize = 0;
    inline for (.{ c_data, h1_data, h2_data, h3_data, h4_data }) |atom_data| {
        for (atom_data.shells[0..atom_data.count]) |s| {
            all_shells[count] = s;
            count += 1;
        }
    }
    const shells = all_shells[0..count];

    std.debug.print("\n=== QM9 validation: CH4 B3LYP/6-31G(2df,p) ===\n", .{});
    std.debug.print("  Shells: {d}, Basis functions: {d}\n", .{ count, obara_saika.totalBasisFunctions(shells) });

    var result = try runKohnShamScf(alloc, shells, &nuc_positions, &nuc_charges, 10, .{
        .xc_functional = .b3lyp,
        .n_radial = 50,
        .n_angular = 302,
        .prune = false,
        .use_direct_scf = true,
        .schwarz_threshold = 1e-10,
    });
    defer result.deinit(alloc);

    std.debug.print("  Total energy:     {d:.12} Ha\n", .{result.total_energy});
    std.debug.print("  Iterations:       {d}\n", .{result.iterations});
    std.debug.print("  Converged:        {}\n", .{result.converged});
    std.debug.print("  PySCF reference:  -40.525553681791 Ha\n", .{});
    std.debug.print("  Diff:             {e:.6}\n", .{@abs(result.total_energy - (-40.525553681791))});

    try std.testing.expect(result.converged);
    // PySCF B3LYP/6-31G(2df,p) cart=True: -40.525553681791 Ha
    // Using 50/302 grid: expect ~1e-3 accuracy
    try std.testing.expectApproxEqAbs(@as(f64, -40.525553681791), result.total_energy, 5e-3);
}

test "QM9 validation: CH2O B3LYP/6-31G(2df,p) vs PySCF" {
    const alloc = std.testing.allocator;
    const b = @import("../basis/basis631g_2dfp.zig");

    // CH2O (formaldehyde) geometry in Bohr (from PySCF mol.atom_coords())
    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = -1.0000430651 }, // C
        .{ .x = 0.0, .y = 0.0, .z = 1.2757541067 }, // O
        .{ .x = 0.0, .y = 1.7680277621, .z = -2.0970290804 }, // H
        .{ .x = 0.0, .y = -1.7680277621, .z = -2.0970290804 }, // H
    };
    const nuc_charges = [_]f64{ 6.0, 8.0, 1.0, 1.0 };

    // Build 6-31G(2df,p) shells
    const c_data = b.buildAtomShells(6, nuc_positions[0]).?;
    const o_data = b.buildAtomShells(8, nuc_positions[1]).?;
    const h1_data = b.buildAtomShells(1, nuc_positions[2]).?;
    const h2_data = b.buildAtomShells(1, nuc_positions[3]).?;

    var all_shells: [b.MAX_SHELLS_PER_ATOM * 4]ContractedShell = undefined;
    var count: usize = 0;
    inline for (.{ c_data, o_data, h1_data, h2_data }) |atom_data| {
        for (atom_data.shells[0..atom_data.count]) |s| {
            all_shells[count] = s;
            count += 1;
        }
    }
    const shells = all_shells[0..count];

    std.debug.print("\n=== QM9 validation: CH2O B3LYP/6-31G(2df,p) ===\n", .{});
    std.debug.print("  Shells: {d}, Basis functions: {d}\n", .{ count, obara_saika.totalBasisFunctions(shells) });

    var result = try runKohnShamScf(alloc, shells, &nuc_positions, &nuc_charges, 16, .{
        .xc_functional = .b3lyp,
        .n_radial = 50,
        .n_angular = 302,
        .prune = false,
        .use_direct_scf = true,
        .schwarz_threshold = 1e-10,
    });
    defer result.deinit(alloc);

    std.debug.print("  Total energy:     {d:.12} Ha\n", .{result.total_energy});
    std.debug.print("  Iterations:       {d}\n", .{result.iterations});
    std.debug.print("  Converged:        {}\n", .{result.converged});
    std.debug.print("  PySCF reference:  -114.511830940393 Ha\n", .{});
    std.debug.print("  Diff:             {e:.6}\n", .{@abs(result.total_energy - (-114.511830940393))});

    try std.testing.expect(result.converged);
    // PySCF B3LYP/6-31G(2df,p) cart=True: -114.511830940393 Ha
    // Using 50/302 grid: expect ~1e-3 accuracy
    try std.testing.expectApproxEqAbs(@as(f64, -114.511830940393), result.total_energy, 5e-3);
}
