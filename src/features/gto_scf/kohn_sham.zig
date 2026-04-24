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
const logging = @import("logging.zig");

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
///   phi(r) = sum_i c_i * N(alpha_i, ax, ay, az)
///            * (x-Cx)^ax * (y-Cy)^ay * (z-Cz)^az * exp(-alpha_i * |r-C|^2)
///
/// Also returns the gradient (dphi/dx, dphi/dy, dphi/dz).
pub fn eval_basis_function(
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
    const x_pow = int_pow(dx_c, ang.x);
    const y_pow = int_pow(dy_c, ang.y);
    const z_pow = int_pow(dz_c, ang.z);
    const angular = x_pow * y_pow * z_pow;

    // d(angular)/dx = ax * x^(ax-1) * y^ay * z^az, etc.
    const dang_dx = if (ang.x > 0)
        @as(f64, @floatFromInt(ang.x)) * int_pow(dx_c, ang.x - 1) * y_pow * z_pow
    else
        0.0;
    const dang_dy = if (ang.y > 0)
        x_pow * @as(f64, @floatFromInt(ang.y)) * int_pow(dy_c, ang.y - 1) * z_pow
    else
        0.0;
    const dang_dz = if (ang.z > 0)
        x_pow * y_pow * @as(f64, @floatFromInt(ang.z)) * int_pow(dz_c, ang.z - 1)
    else
        0.0;

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
fn angular_first_deriv(coords: [3]f64, angs: [3]u32, pows: [3]f64) [3]f64 {
    var dang: [3]f64 = undefined;
    for (0..3) |i| {
        if (angs[i] > 0) {
            var prod: f64 = @floatFromInt(angs[i]);
            prod *= int_pow(coords[i], angs[i] - 1);
            for (0..3) |j| {
                if (j != i) prod *= pows[j];
            }
            dang[i] = prod;
        } else {
            dang[i] = 0.0;
        }
    }
    return dang;
}

fn angular_second_deriv(coords: [3]f64, angs: [3]u32, pows: [3]f64) [3][3]f64 {
    var d2ang: [3][3]f64 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            if (i == j) {
                // d²A/dx_i²
                if (angs[i] >= 2) {
                    const ai_f = @as(f64, @floatFromInt(angs[i]));
                    const aim1_f = @as(f64, @floatFromInt(angs[i] - 1));
                    var prod: f64 = ai_f * aim1_f;
                    prod *= int_pow(coords[i], angs[i] - 2);
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
                    const ai_f = @as(f64, @floatFromInt(angs[i]));
                    const aj_f = @as(f64, @floatFromInt(angs[j]));
                    var prod: f64 = ai_f * int_pow(coords[i], angs[i] - 1);
                    prod *= aj_f * int_pow(coords[j], angs[j] - 1);
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
    return d2ang;
}

fn accumulate_primitive_hessian(
    prim_alpha: f64,
    c_n_g: f64,
    coords: [3]f64,
    angular: f64,
    dang: [3]f64,
    d2ang: [3][3]f64,
    val: *f64,
    grad: *[3]f64,
    hess: *[3][3]f64,
) void {
    const a2 = -2.0 * prim_alpha;
    val.* += c_n_g * angular;

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
    const alpha2 = prim_alpha * prim_alpha;
    for (0..3) |i| {
        for (i..3) |j| {
            const delta_ij: f64 = if (i == j) 1.0 else 0.0;
            // d²G/dxi dxj / G = -2α δ_ij + 4α² xi xj
            const d2_gaussian = a2 * delta_ij + 4.0 * alpha2 * coords[i] * coords[j];
            const h_val = d2ang[i][j] + dang[i] * a2 * coords[j] +
                dang[j] * a2 * coords[i] + angular * d2_gaussian;
            hess[i][j] += c_n_g * h_val;
        }
    }
}

pub fn eval_basis_function_with_hessian(
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
        pows[i] = int_pow(coords[i], angs[i]);
    }
    const angular = pows[0] * pows[1] * pows[2];

    // First derivatives of angular part: dA/dx_i
    const dang = angular_first_deriv(coords, angs, pows);

    // Second derivatives of angular part: d²A/dx_i dx_j
    const d2ang = angular_second_deriv(coords, angs, pows);

    var val: f64 = 0.0;
    var grad: [3]f64 = .{ 0.0, 0.0, 0.0 };
    var hess: [3][3]f64 = .{ .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 } };

    for (shell.primitives) |prim| {
        const norm = basis_mod.normalization(prim.alpha, ang.x, ang.y, ang.z);
        const g = @exp(-prim.alpha * r2);
        const c_n_g = prim.coeff * norm * g;
        accumulate_primitive_hessian(
            prim.alpha,
            c_n_g,
            coords,
            angular,
            dang,
            d2ang,
            &val,
            &grad,
            &hess,
        );
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
pub fn int_pow(x: f64, n: u32) f64 {
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
fn count_basis_primitive_pairs(shells: []const ContractedShell) usize {
    var total_pairs: usize = 0;
    for (shells) |shell| {
        const n_cart = basis_mod.num_cartesian(shell.l);
        total_pairs += n_cart * shell.primitives.len;
    }
    return total_pairs;
}

fn build_basis_info_table(
    shells: []const ContractedShell,
    cn_flat: []f64,
    alpha_flat: []f64,
    basis_info: []BasisFuncInfo,
    screen_threshold_log: f64,
) void {
    var mu: usize = 0;
    var pair_off: usize = 0;
    for (shells) |shell| {
        const cart = basis_mod.cartesian_exponents(shell.l);
        const n_cart = basis_mod.num_cartesian(shell.l);

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
                const norm_val = basis_mod.normalization(prim.alpha, ang.x, ang.y, ang.z);
                cn_flat[pair_off + ip] = prim.coeff * norm_val;
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

fn eval_basis_at_grid_point(
    bi: BasisFuncInfo,
    gp: GridPoint,
    idx: usize,
    phi: []f64,
    dphi_x: []f64,
    dphi_y: []f64,
    dphi_z: []f64,
) void {
    const dx_c = gp.x - bi.cx;
    const dy_c = gp.y - bi.cy;
    const dz_c = gp.z - bi.cz;
    const r2 = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c;

    // Grid-point screening: skip if all primitives are negligible
    if (r2 > bi.screen_r2) {
        // phi/dphi already initialized to 0
        return;
    }

    // Angular part
    const x_pow = int_pow(dx_c, bi.ax);
    const y_pow = int_pow(dy_c, bi.ay);
    const z_pow = int_pow(dz_c, bi.az);
    const angular = x_pow * y_pow * z_pow;

    // Angular derivatives
    const dang_dx = if (bi.ax > 0)
        @as(f64, @floatFromInt(bi.ax)) * int_pow(dx_c, bi.ax - 1) * y_pow * z_pow
    else
        0.0;
    const dang_dy = if (bi.ay > 0)
        x_pow * @as(f64, @floatFromInt(bi.ay)) * int_pow(dy_c, bi.ay - 1) * z_pow
    else
        0.0;
    const dang_dz = if (bi.az > 0)
        x_pow * y_pow * @as(f64, @floatFromInt(bi.az)) * int_pow(dz_c, bi.az - 1)
    else
        0.0;

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

    phi[idx] = val;
    dphi_x[idx] = gx;
    dphi_y[idx] = gy;
    dphi_z[idx] = gz;
}

pub fn evaluate_basis_on_grid(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    grid_points: []const GridPoint,
) !BasisOnGrid {
    const n_basis = obara_saika.total_basis_functions(shells);
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
    const total_pairs = count_basis_primitive_pairs(shells);

    const cn_flat = try alloc.alloc(f64, total_pairs);
    defer alloc.free(cn_flat);

    const alpha_flat = try alloc.alloc(f64, total_pairs);
    defer alloc.free(alpha_flat);

    const basis_info = try alloc.alloc(BasisFuncInfo, n_basis);
    defer alloc.free(basis_info);

    build_basis_info_table(shells, cn_flat, alpha_flat, basis_info, screen_threshold_log);

    // --- Evaluate on grid using precomputed tables with screening ---
    for (grid_points, 0..) |gp, ig| {
        const row_off = ig * n_basis;
        for (basis_info, 0..) |bi, mu| {
            eval_basis_at_grid_point(bi, gp, row_off + mu, phi, dphi_x, dphi_y, dphi_z);
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
pub fn compute_density_on_grid(
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
const XcPointValues = struct { eps_xc: f64, v_xc: f64, v_sigma: f64 };

fn evaluate_xc_at_point(xc_func: XcFunctional, rho_g: f64, sigma: f64) XcPointValues {
    switch (xc_func) {
        .lda_svwn => {
            const xc = xc_functionals.lda_svwn(rho_g);
            return .{ .eps_xc = xc.eps_xc, .v_xc = xc.v_xc, .v_sigma = 0.0 };
        },
        .b3lyp => {
            const xc = xc_functionals.b3lyp(rho_g, sigma);
            return .{ .eps_xc = xc.eps_xc, .v_xc = xc.v_xc, .v_sigma = xc.v_sigma };
        },
    }
}

fn write_h_row_for_grid_point(
    n_basis: usize,
    g_off: usize,
    bog: BasisOnGrid,
    h_buf: []f64,
    w: f64,
    v_xc: f64,
    v_sigma: f64,
    grx: f64,
    gry: f64,
    grz: f64,
) void {
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

fn build_h_buffer_and_xc_energy(
    n_basis: usize,
    n_grid: usize,
    grid_points: []const GridPoint,
    bog: BasisOnGrid,
    rho_vals: []const f64,
    grad_rho_x: []const f64,
    grad_rho_y: []const f64,
    grad_rho_z: []const f64,
    xc_func: XcFunctional,
    h_buf: []f64,
) f64 {
    @memset(h_buf, 0.0);
    var e_xc: f64 = 0.0;
    for (0..n_grid) |ig| {
        const rho_g = rho_vals[ig];
        if (rho_g < 1e-20) continue;

        const w = grid_points[ig].w;
        const grx = grad_rho_x[ig];
        const gry = grad_rho_y[ig];
        const grz = grad_rho_z[ig];
        const sigma = grx * grx + gry * gry + grz * grz;

        const xc_vals = evaluate_xc_at_point(xc_func, rho_g, sigma);

        // Accumulate XC energy
        e_xc += w * xc_vals.eps_xc * rho_g;

        // Build H row for this grid point
        write_h_row_for_grid_point(
            n_basis,
            ig * n_basis,
            bog,
            h_buf,
            w,
            xc_vals.v_xc,
            xc_vals.v_sigma,
            grx,
            gry,
            grz,
        );
    }
    return e_xc;
}

fn symmetrize_vxc_mat(n_basis: usize, vxc_mat: []f64) void {
    for (0..n_basis) |mu| {
        for (mu + 1..n_basis) |nu| {
            const val = vxc_mat[mu * n_basis + nu] + vxc_mat[nu * n_basis + mu];
            vxc_mat[mu * n_basis + nu] = val;
            vxc_mat[nu * n_basis + mu] = val;
        }
        // Diagonal: Vxc[mu,mu] = 2 * Tmp[mu,mu] (from H^T*Phi, symmetric part)
        vxc_mat[mu * n_basis + mu] *= 2.0;
    }
}

fn build_xc_contribution(
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
    const e_xc = build_h_buffer_and_xc_energy(
        n_basis,
        n_grid,
        grid_points,
        bog,
        rho_vals,
        grad_rho_x,
        grad_rho_y,
        grad_rho_z,
        xc_func,
        h_buf,
    );

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
    symmetrize_vxc_mat(n_basis, vxc_mat);

    return e_xc;
}

/// Build the Coulomb (J) matrix from the density matrix and ERI table.
///   J_{mu,nu} = sum_{lam,sig} P_{lam,sig} * (mu nu | lam sig)
fn build_coulomb_matrix(
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
fn build_exchange_matrix(
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

const KsJkCtx = struct {
    n: usize,
    hf_frac: f64,
    df_context: ?*DensityFittingContext,
    use_libcint_actual: bool,
    jk_builder: ?*libcint.LibcintJKBuilder,
    use_direct_scf: bool,
    shells: []const ContractedShell,
    schwarz_table: ?*fock.SchwarzTable,
    schwarz_threshold: f64,
    eri_table: ?*obara_saika.GeneralEriTable,
};

fn build_jk_for_ks(
    alloc: std.mem.Allocator,
    ctx: KsJkCtx,
    p_mat: []const f64,
    j_mat: []f64,
    k_mat: []f64,
) !void {
    if (ctx.df_context != null) {
        try ctx.df_context.?.build_j(alloc, p_mat, j_mat);
        if (ctx.hf_frac > 0.0) {
            try ctx.df_context.?.build_k(alloc, p_mat, k_mat);
        } else {
            @memset(k_mat, 0.0);
        }
    } else if (ctx.use_libcint_actual) {
        const jk = try ctx.jk_builder.?.build_jk(alloc, p_mat);
        @memcpy(j_mat, jk.j_matrix);
        @memcpy(k_mat, jk.k_matrix);
        alloc.free(jk.j_matrix);
        alloc.free(jk.k_matrix);
        if (ctx.hf_frac == 0.0) {
            @memset(k_mat, 0.0);
        }
    } else if (ctx.use_direct_scf) {
        fock.build_jk_direct(
            ctx.n,
            p_mat,
            ctx.shells,
            ctx.schwarz_table.?,
            ctx.schwarz_threshold,
            j_mat,
            k_mat,
        );
        if (ctx.hf_frac == 0.0) {
            @memset(k_mat, 0.0);
        }
    } else {
        build_coulomb_matrix(ctx.n, p_mat, ctx.eri_table.?.*, j_mat);
        if (ctx.hf_frac > 0.0) {
            build_exchange_matrix(ctx.n, p_mat, ctx.eri_table.?.*, k_mat);
        }
    }
}

const KsEnergyTerms = struct {
    e_1e: f64,
    e_j: f64,
    e_k: f64,
};

fn compute_ks_energy_terms(
    n: usize,
    p_mat: []const f64,
    h_core: []const f64,
    j_mat: []const f64,
    k_mat: []const f64,
    hf_frac: f64,
) KsEnergyTerms {
    var e_1e: f64 = 0.0;
    for (0..n) |mu| {
        for (0..n) |nu| {
            e_1e += p_mat[mu * n + nu] * h_core[mu * n + nu];
        }
    }

    var e_j: f64 = 0.0;
    for (0..n) |mu| {
        for (0..n) |nu| {
            e_j += p_mat[mu * n + nu] * j_mat[mu * n + nu];
        }
    }
    e_j *= 0.5;

    var e_k: f64 = 0.0;
    if (hf_frac > 0.0) {
        for (0..n) |mu| {
            for (0..n) |nu| {
                e_k += p_mat[mu * n + nu] * k_mat[mu * n + nu];
            }
        }
        e_k *= -0.25 * hf_frac;
    }
    return .{ .e_1e = e_1e, .e_j = e_j, .e_k = e_k };
}

fn assemble_fock_matrix(
    n: usize,
    h_core: []const f64,
    j_mat: []const f64,
    vxc_mat: []const f64,
    k_mat: []const f64,
    hf_frac: f64,
    f_mat: []f64,
) void {
    // Build Fock matrix: F = H_core + J - 0.5 * hf_frac * K + Vxc
    for (0..n * n) |i| {
        f_mat[i] = h_core[i] + j_mat[i] + vxc_mat[i];
        if (hf_frac > 0.0) {
            f_mat[i] -= 0.5 * hf_frac * k_mat[i];
        }
    }
}

fn build_density_and_xc(
    alloc: std.mem.Allocator,
    n: usize,
    grid_points: []const GridPoint,
    p_mat: []const f64,
    bog: BasisOnGrid,
    xc_func: XcFunctional,
    vxc_mat: []f64,
    xc_work_buf: []f64,
) !f64 {
    const dens = try compute_density_on_grid(alloc, n, grid_points.len, p_mat, bog);
    defer {
        alloc.free(dens.rho);
        alloc.free(dens.grad_x);
        alloc.free(dens.grad_y);
        alloc.free(dens.grad_z);
    }

    return build_xc_contribution(
        n,
        grid_points.len,
        grid_points,
        bog,
        dens.rho,
        dens.grad_x,
        dens.grad_y,
        dens.grad_z,
        xc_func,
        vxc_mat,
        xc_work_buf,
    );
}

fn log_ks_iter_verbose(
    verbose: bool,
    iter: usize,
    n: usize,
    e_total: f64,
    delta_e: f64,
    rms_p: f64,
    terms: KsEnergyTerms,
    e_xc: f64,
    v_nn: f64,
    p_mat: []const f64,
    s_mat: []const f64,
) void {
    if (!verbose) return;
    logging.verbose(
        true,
        "  SCF iter {d:3}: E = {d:20.12}  dE = {e:10.3}  dP = {e:10.3}\n",
        .{ iter, e_total, delta_e, rms_p },
    );
    if (iter == 0) {
        logging.verbose(
            true,
            "    E_1e = {d:20.12}  E_J = {d:20.12}  E_K = {d:20.12}" ++
                "  E_XC = {d:20.12}  V_nn = {d:20.12}\n",
            .{ terms.e_1e, terms.e_j, terms.e_k, e_xc, v_nn },
        );
        var tr_p: f64 = 0.0;
        var tr_ps: f64 = 0.0;
        for (0..n) |imu| {
            tr_p += p_mat[imu * n + imu];
            for (0..n) |inu| {
                tr_ps += p_mat[imu * n + inu] * s_mat[imu * n + inu];
            }
        }
        logging.verbose(
            true,
            "    Tr(P) = {d:20.12}  Tr(P*S) = {d:20.12}\n",
            .{ tr_p, tr_ps },
        );
    }
}

fn log_initial_eigen_and_s_matrix(
    alloc: std.mem.Allocator,
    n: usize,
    eigen_values: []const f64,
    s_mat: []const f64,
    step5_secs: f64,
) !void {
    logging.verbose(
        true,
        "  [KS] Step 5: Done. ({d:.2}s)\n",
        .{step5_secs},
    );
    // Print initial orbital eigenvalues for comparison with PySCF
    logging.verbose(true, "  [KS] Initial orbital eigenvalues (ALL {d}):\n", .{n});
    for (0..n) |i| {
        logging.verbose(true, "    [{d:2}] {e:20.12}\n", .{ i, eigen_values[i] });
    }
    // Print S matrix eigenvalues for condition number check
    const s_copy = try alloc.alloc(f64, n * n);
    defer alloc.free(s_copy);

    @memcpy(s_copy, s_mat);
    const s_eigen = try solve_standard_eigen(alloc, n, s_copy);
    defer {
        alloc.free(s_eigen.values);
        alloc.free(s_eigen.vectors);
    }

    logging.verbose(true, "  [KS] S matrix eigenvalues:\n", .{});
    logging.verbose(true, "    min = {e:20.12}\n", .{s_eigen.values[0]});
    logging.verbose(true, "    max = {e:20.12}\n", .{s_eigen.values[n - 1]});
    const cond_num = s_eigen.values[n - 1] / s_eigen.values[0];
    logging.verbose(true, "    condition number = {e:10.3}\n", .{cond_num});
    for (0..@min(n, 5)) |i| {
        logging.verbose(true, "    [{d:2}] {e:20.12}\n", .{ i, s_eigen.values[i] });
    }
    if (n > 10) {
        logging.verbose(true, "    ...\n", .{});
        for (n - 5..n) |i| {
            logging.verbose(true, "    [{d:2}] {e:20.12}\n", .{ i, s_eigen.values[i] });
        }
    }
}

const OneElectronMats = struct {
    s_mat: []f64,
    t_mat: []f64,
    v_mat: []f64,
    h_core: []f64,
};

fn build_ks_one_electron_mats(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    n: usize,
    cint_data_storage: ?libcint.LibcintData,
    use_libcint_actual: bool,
) !OneElectronMats {
    const s_mat = if (use_libcint_actual)
        try libcint.build_overlap_matrix(alloc, cint_data_storage.?)
    else
        try obara_saika.build_overlap_matrix(alloc, shells);

    const t_mat = if (use_libcint_actual)
        try libcint.build_kinetic_matrix(alloc, cint_data_storage.?)
    else
        try obara_saika.build_kinetic_matrix(alloc, shells);

    const v_mat = if (use_libcint_actual)
        try libcint.build_nuclear_matrix(alloc, cint_data_storage.?)
    else
        try obara_saika.build_nuclear_matrix(alloc, shells, nuc_positions, nuc_charges);

    const h_core = try alloc.alloc(f64, n * n);
    for (0..n * n) |i| {
        h_core[i] = t_mat[i] + v_mat[i];
    }
    return .{ .s_mat = s_mat, .t_mat = t_mat, .v_mat = v_mat, .h_core = h_core };
}

fn build_auto_aux_shells(
    alloc: std.mem.Allocator,
    n: usize,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    out_buf_to_free: *?[]ContractedShell,
) ![]const ContractedShell {
    // Build def2-universal-JKFIT auxiliary basis automatically
    const aux_buf = try alloc.alloc(ContractedShell, n * 10); // generous upper bound
    errdefer alloc.free(aux_buf);

    var aux_count: usize = 0;

    for (0..nuc_positions.len) |i| {
        const z: u32 = @intFromFloat(nuc_charges[i]);
        if (aux_basis_mod.build_def2_universal_jkfit(z, nuc_positions[i])) |aux_result| {
            for (aux_result.shells[0..aux_result.count]) |s| {
                aux_buf[aux_count] = s;
                aux_count += 1;
            }
        }
    }
    out_buf_to_free.* = aux_buf;
    return aux_buf[0..aux_count];
}

fn build_molecular_grid_from_nuclei(
    alloc: std.mem.Allocator,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    params: KsParams,
) ![]becke.GridPoint {
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
    return try becke.build_molecular_grid(alloc, atoms, grid_config);
}

const KsScfState = struct {
    e_total: f64,
    e_old: f64,
    converged: bool,
    final_e_xc: f64,
    final_e_1e: f64,
    final_e_j: f64,
    final_e_k: f64,
    scf_jk_ns: u64,
    scf_xc_ns: u64,
    scf_diag_ns: u64,
};

const KsScfBuffers = struct {
    p_old: []f64,
    f_mat: []f64,
    j_mat: []f64,
    k_mat: []f64,
    vxc_mat: []f64,
    xc_work_buf: []f64,
    diis: ?GtoDiis,
    f_diis: ?[]f64,

    fn deinit(self: *KsScfBuffers, alloc: std.mem.Allocator) void {
        alloc.free(self.p_old);
        alloc.free(self.f_mat);
        alloc.free(self.j_mat);
        alloc.free(self.k_mat);
        alloc.free(self.vxc_mat);
        alloc.free(self.xc_work_buf);
        if (self.diis) |*d| d.deinit();
        if (self.f_diis) |buf| alloc.free(buf);
    }
};

fn alloc_ks_scf_buffers(
    alloc: std.mem.Allocator,
    n: usize,
    grid_point_count: usize,
    params: KsParams,
) !KsScfBuffers {
    const p_old = try alloc.alloc(f64, n * n);
    errdefer alloc.free(p_old);
    @memset(p_old, 0.0);

    const f_mat = try alloc.alloc(f64, n * n);
    errdefer alloc.free(f_mat);
    const j_mat = try alloc.alloc(f64, n * n);
    errdefer alloc.free(j_mat);
    const k_mat = try alloc.alloc(f64, n * n);
    errdefer alloc.free(k_mat);
    const vxc_mat = try alloc.alloc(f64, n * n);
    errdefer alloc.free(vxc_mat);
    const xc_work_buf = try alloc.alloc(f64, grid_point_count * n);
    errdefer alloc.free(xc_work_buf);

    var diis: ?GtoDiis = if (params.use_diis)
        GtoDiis.init(alloc, n, params.diis_max_vectors)
    else
        null;
    errdefer if (diis) |*d| d.deinit();

    const f_diis = if (params.use_diis) try alloc.alloc(f64, n * n) else null;
    errdefer if (f_diis) |buf| alloc.free(buf);

    return .{
        .p_old = p_old,
        .f_mat = f_mat,
        .j_mat = j_mat,
        .k_mat = k_mat,
        .vxc_mat = vxc_mat,
        .xc_work_buf = xc_work_buf,
        .diis = diis,
        .f_diis = f_diis,
    };
}

fn init_ks_scf_state() KsScfState {
    return .{
        .e_total = 0.0,
        .e_old = 0.0,
        .converged = false,
        .final_e_xc = 0.0,
        .final_e_1e = 0.0,
        .final_e_j = 0.0,
        .final_e_k = 0.0,
        .scf_jk_ns = 0,
        .scf_xc_ns = 0,
        .scf_diag_ns = 0,
    };
}

fn log_ks_scf_timing(verbose: bool, state: KsScfState) void {
    if (!verbose) return;
    logging.verbose(true, "  [KS] SCF timing: J/K={d:.2}s  XC={d:.2}s  diag={d:.2}s\n", .{
        @as(f64, @floatFromInt(state.scf_jk_ns)) / 1e9,
        @as(f64, @floatFromInt(state.scf_xc_ns)) / 1e9,
        @as(f64, @floatFromInt(state.scf_diag_ns)) / 1e9,
    });
}

fn finalize_unconverged_ks_state(
    alloc: std.mem.Allocator,
    n: usize,
    hf_frac: f64,
    v_nn: f64,
    h_core: []const f64,
    jk_ctx: KsJkCtx,
    grid_points: []const becke.GridPoint,
    p_mat: []f64,
    bog: BasisOnGrid,
    xc_functional: XcFunctional,
    bufs: *const KsScfBuffers,
    state: *KsScfState,
) !void {
    if (state.converged) return;
    try build_jk_for_ks(alloc, jk_ctx, p_mat, bufs.j_mat, bufs.k_mat);
    state.final_e_xc = try build_density_and_xc(
        alloc,
        n,
        grid_points,
        p_mat,
        bog,
        xc_functional,
        bufs.vxc_mat,
        bufs.xc_work_buf,
    );
    const terms = compute_ks_energy_terms(n, p_mat, h_core, bufs.j_mat, bufs.k_mat, hf_frac);
    state.final_e_1e = terms.e_1e;
    state.final_e_j = terms.e_j;
    state.final_e_k = terms.e_k;
    state.e_total = terms.e_1e + terms.e_j + terms.e_k + state.final_e_xc + v_nn;
}

fn compute_ks_iter_energy_and_fock(
    n: usize,
    hf_frac: f64,
    v_nn: f64,
    h_core: []const f64,
    j_mat: []const f64,
    k_mat: []const f64,
    vxc_mat: []const f64,
    e_xc: f64,
    p_mat: []const f64,
    f_mat: []f64,
    state: *KsScfState,
) KsEnergyTerms {
    assemble_fock_matrix(n, h_core, j_mat, vxc_mat, k_mat, hf_frac, f_mat);
    const terms = compute_ks_energy_terms(n, p_mat, h_core, j_mat, k_mat, hf_frac);
    state.e_total = terms.e_1e + terms.e_j + terms.e_k + e_xc + v_nn;
    state.final_e_xc = e_xc;
    state.final_e_1e = terms.e_1e;
    state.final_e_j = terms.e_j;
    state.final_e_k = terms.e_k;
    return terms;
}

fn diagonalize_and_update_density(
    alloc: std.mem.Allocator,
    io: std.Io,
    iter: usize,
    n: usize,
    n_occ: usize,
    s_mat: []const f64,
    f_mat: []f64,
    p_mat: []f64,
    f_diis: ?[]f64,
    diis: *?GtoDiis,
    eigen: *linalg.RealEigenDecomp,
    params: KsParams,
    state: *KsScfState,
) !void {
    const f_to_diag = if (diis.* != null and iter >= params.diis_start_iter) blk: {
        try diis.*.?.extrapolate(f_mat, p_mat, s_mat, f_diis.?);
        break :blk f_diis.?;
    } else f_mat;

    alloc.free(eigen.vectors);
    alloc.free(eigen.values);

    const timer = std.Io.Clock.Timestamp.now(io, .awake);
    eigen.* = try solve_roothaan_hall(alloc, n, f_to_diag, s_mat);
    state.scf_diag_ns += @as(u64, @intCast(timer.untilNow(io).raw.nanoseconds));

    density_matrix.update_density_matrix(n, n_occ, eigen.vectors, p_mat);
}

fn run_ks_scf_iteration(
    alloc: std.mem.Allocator,
    io: std.Io,
    iter: usize,
    n: usize,
    n_occ: usize,
    hf_frac: f64,
    v_nn: f64,
    h_core: []const f64,
    s_mat: []const f64,
    jk_ctx: KsJkCtx,
    grid_points: []const becke.GridPoint,
    bog: BasisOnGrid,
    xc_functional: XcFunctional,
    params: KsParams,
    p_mat: []f64,
    p_old: []f64,
    f_mat: []f64,
    j_mat: []f64,
    k_mat: []f64,
    vxc_mat: []f64,
    xc_work_buf: []f64,
    f_diis: ?[]f64,
    diis: *?GtoDiis,
    eigen: *linalg.RealEigenDecomp,
    state: *KsScfState,
) !bool {
    var timer = std.Io.Clock.Timestamp.now(io, .awake);
    try build_jk_for_ks(alloc, jk_ctx, p_mat, j_mat, k_mat);
    state.scf_jk_ns += @as(u64, @intCast(timer.untilNow(io).raw.nanoseconds));

    timer = std.Io.Clock.Timestamp.now(io, .awake);
    const e_xc = try build_density_and_xc(
        alloc,
        n,
        grid_points,
        p_mat,
        bog,
        xc_functional,
        vxc_mat,
        xc_work_buf,
    );
    state.scf_xc_ns += @as(u64, @intCast(timer.untilNow(io).raw.nanoseconds));

    const terms = compute_ks_iter_energy_and_fock(
        n,
        hf_frac,
        v_nn,
        h_core,
        j_mat,
        k_mat,
        vxc_mat,
        e_xc,
        p_mat,
        f_mat,
        state,
    );

    const delta_e = @abs(state.e_total - state.e_old);
    const rms_p = density_matrix.density_rms_diff(n, p_mat, p_old);

    log_ks_iter_verbose(
        params.verbose,
        iter,
        n,
        state.e_total,
        delta_e,
        rms_p,
        terms,
        e_xc,
        v_nn,
        p_mat,
        s_mat,
    );

    if (iter > 0 and delta_e < params.energy_threshold and rms_p < params.density_threshold) {
        state.converged = true;
        return true;
    }

    state.e_old = state.e_total;
    @memcpy(p_old, p_mat);

    try diagonalize_and_update_density(
        alloc,
        io,
        iter,
        n,
        n_occ,
        s_mat,
        f_mat,
        p_mat,
        f_diis,
        diis,
        eigen,
        params,
        state,
    );
    return false;
}

fn elapsed_seconds(io: std.Io, timer: std.Io.Clock.Timestamp) f64 {
    return @as(
        f64,
        @floatFromInt(@as(u64, @intCast(timer.untilNow(io).raw.nanoseconds))),
    ) / 1e9;
}

fn log_ks_step_done(
    verbose: bool,
    step_label: []const u8,
    io: std.Io,
    timer: std.Io.Clock.Timestamp,
) void {
    const secs = elapsed_seconds(io, timer);
    logging.verbose(verbose, "  [KS] {s}: Done. ({d:.2}s)\n", .{ step_label, secs });
}

fn log_ks_step3_done(
    verbose: bool,
    grid_count: usize,
    io: std.Io,
    timer: std.Io.Clock.Timestamp,
) void {
    const secs = elapsed_seconds(io, timer);
    logging.verbose(
        verbose,
        "  [KS] Step 3: Done ({d} grid points). ({d:.2}s)\n",
        .{ grid_count, secs },
    );
}

fn log_ks_step5_verbose(
    alloc: std.mem.Allocator,
    io: std.Io,
    timer: std.Io.Clock.Timestamp,
    n: usize,
    eigen_values: []const f64,
    s_mat: []const f64,
) !void {
    const step5_secs = elapsed_seconds(io, timer);
    try log_initial_eigen_and_s_matrix(alloc, n, eigen_values, s_mat, step5_secs);
}

fn prepare_libcint_if_enabled(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    use_libcint: bool,
    out_nuc_pos_flat: *[]const [3]f64,
) !?libcint.LibcintData {
    const nuc_pos_flat = try alloc.alloc([3]f64, nuc_positions.len);
    for (nuc_positions, 0..) |pos, i| {
        nuc_pos_flat[i] = .{ pos.x, pos.y, pos.z };
    }
    out_nuc_pos_flat.* = nuc_pos_flat;

    if (!use_libcint) return null;
    return libcint.LibcintData.init(alloc, shells, nuc_pos_flat, nuc_charges) catch null;
}

fn build_eri_or_schwarz_tables(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    cint_data_storage: ?libcint.LibcintData,
    use_libcint_actual: bool,
    params: KsParams,
    eri_table: *?obara_saika.GeneralEriTable,
    schwarz_table: *?fock.SchwarzTable,
    jk_builder: *?libcint.LibcintJKBuilder,
) !void {
    if (use_libcint_actual) {
        jk_builder.* = try libcint.LibcintJKBuilder.init(
            alloc,
            cint_data_storage.?,
            params.schwarz_threshold,
        );
    } else if (params.use_direct_scf) {
        schwarz_table.* = try fock.build_schwarz_table(alloc, shells);
    } else {
        eri_table.* = try obara_saika.build_eri_table(alloc, shells);
    }
}

fn build_density_fitting_if_enabled(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []const ContractedShell,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    n: usize,
    params: KsParams,
) !?DensityFittingContext {
    if (!params.use_density_fitting) return null;
    const timer = std.Io.Clock.Timestamp.now(io, .awake);
    logging.verbose(params.verbose, "  [KS] Step 2b: Building density fitting context...\n", .{});

    var aux_buf_to_free: ?[]ContractedShell = null;
    const aux_shells = if (params.aux_shells) |as|
        as
    else
        try build_auto_aux_shells(alloc, n, nuc_positions, nuc_charges, &aux_buf_to_free);

    const df = try DensityFittingContext.init(alloc, shells, aux_shells);
    if (aux_buf_to_free) |buf| alloc.free(buf);

    const step2b_ns = @as(u64, @intCast(timer.untilNow(io).raw.nanoseconds));
    logging.verbose(
        params.verbose,
        "  [KS] Step 2b: Done (n_aux={d}). ({d:.2}s)\n",
        .{ df.n_aux, @as(f64, @floatFromInt(step2b_ns)) / 1e9 },
    );
    return df;
}

fn package_ks_result(
    loop_out: anytype,
    v_nn: f64,
    eigen: linalg.RealEigenDecomp,
    p_mat: []f64,
) KsResult {
    return .{
        .total_energy = loop_out.state.e_total,
        .electronic_energy = loop_out.state.e_total - v_nn,
        .nuclear_repulsion = v_nn,
        .one_electron_energy = loop_out.state.final_e_1e,
        .coulomb_energy = loop_out.state.final_e_j,
        .xc_energy = loop_out.state.final_e_xc,
        .hf_exchange_energy = loop_out.state.final_e_k,
        .orbital_energies = eigen.values,
        .mo_coefficients = eigen.vectors,
        .density_matrix_result = p_mat,
        .iterations = loop_out.iter,
        .converged = loop_out.state.converged,
    };
}

const KsScfPreparation = struct {
    v_nn: f64,
    grid_points: []becke.GridPoint,
    bog: BasisOnGrid,
    eigen: linalg.RealEigenDecomp,

    fn deinit(self: *KsScfPreparation, alloc: std.mem.Allocator) void {
        alloc.free(self.grid_points);
        self.bog.deinit(alloc);
    }
};

fn hf_exchange_fraction(xc_functional: XcFunctional) f64 {
    return switch (xc_functional) {
        .lda_svwn => 0.0,
        .b3lyp => 0.20,
    };
}

fn prepare_ks_scf_loop(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []const ContractedShell,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    n: usize,
    h_core: []const f64,
    s_mat: []const f64,
    params: KsParams,
) !KsScfPreparation {
    const v_nn = energy_mod.nuclear_repulsion_energy(nuc_positions, nuc_charges);
    const grid_points = try run_ks_step_three(alloc, io, nuc_positions, nuc_charges, params);
    errdefer alloc.free(grid_points);

    var bog = try run_ks_step_four(alloc, io, shells, grid_points, params);
    errdefer bog.deinit(alloc);

    const eigen = try run_ks_step_five(alloc, io, n, h_core, s_mat, params);
    return .{
        .v_nn = v_nn,
        .grid_points = grid_points,
        .bog = bog,
        .eigen = eigen,
    };
}

fn run_ks_scf_loop(
    alloc: std.mem.Allocator,
    io: std.Io,
    n: usize,
    n_occ: usize,
    hf_frac: f64,
    shells: []const ContractedShell,
    resources: *KsIntegralResources,
    h_core: []const f64,
    s_mat: []const f64,
    prep: *KsScfPreparation,
    params: KsParams,
    p_mat: []f64,
) !struct { state: KsScfState, iter: usize } {
    logging.verbose(
        params.verbose,
        "  [KS] Step 6: Starting SCF loop (max_iter={d})...\n",
        .{params.max_iter},
    );
    const jk_ctx = make_ks_jk_ctx(
        n,
        hf_frac,
        shells,
        resources.use_libcint_actual,
        params,
        &resources.df_context,
        &resources.jk_builder,
        &resources.schwarz_table,
        &resources.eri_table,
    );
    return run_ks_main_scc_loop(
        alloc,
        io,
        n,
        n_occ,
        hf_frac,
        prep.v_nn,
        h_core,
        s_mat,
        jk_ctx,
        prep.grid_points,
        prep.bog,
        params,
        p_mat,
        &prep.eigen,
    );
}

fn make_ks_jk_ctx(
    n: usize,
    hf_frac: f64,
    shells: []const ContractedShell,
    use_libcint_actual: bool,
    params: KsParams,
    df_context: *?DensityFittingContext,
    jk_builder: *?libcint.LibcintJKBuilder,
    schwarz_table: *?fock.SchwarzTable,
    eri_table: *?obara_saika.GeneralEriTable,
) KsJkCtx {
    return .{
        .n = n,
        .hf_frac = hf_frac,
        .df_context = if (df_context.*) |*dfc| dfc else null,
        .use_libcint_actual = use_libcint_actual,
        .jk_builder = if (jk_builder.*) |*jkb| jkb else null,
        .use_direct_scf = params.use_direct_scf,
        .shells = shells,
        .schwarz_table = if (schwarz_table.*) |*st| st else null,
        .schwarz_threshold = params.schwarz_threshold,
        .eri_table = if (eri_table.*) |*et| et else null,
    };
}

const KsIntegralResources = struct {
    nuc_pos_flat: []const [3]f64,
    cint_data_storage: ?libcint.LibcintData,
    one_mats: OneElectronMats,
    eri_table: ?obara_saika.GeneralEriTable,
    schwarz_table: ?fock.SchwarzTable,
    jk_builder: ?libcint.LibcintJKBuilder,
    df_context: ?DensityFittingContext,
    use_libcint_actual: bool,

    fn deinit(self: *KsIntegralResources, alloc: std.mem.Allocator) void {
        alloc.free(self.nuc_pos_flat);
        if (self.cint_data_storage) |*cd| cd.deinit(alloc);
        alloc.free(self.one_mats.s_mat);
        alloc.free(self.one_mats.t_mat);
        alloc.free(self.one_mats.v_mat);
        alloc.free(self.one_mats.h_core);
        if (self.eri_table) |*et| et.deinit(alloc);
        if (self.schwarz_table) |*st| st.deinit(alloc);
        if (self.jk_builder) |*jkb| jkb.deinit(alloc);
        if (self.df_context) |*dfc| dfc.deinit();
    }
};

const KsOneElectronStage = struct {
    nuc_pos_flat: []const [3]f64,
    cint_data_storage: ?libcint.LibcintData,
    one_mats: OneElectronMats,
    use_libcint_actual: bool,
};

fn run_ks_step_one(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []const ContractedShell,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    n: usize,
    params: KsParams,
) !KsOneElectronStage {
    const timer = std.Io.Clock.Timestamp.now(io, .awake);
    logging.verbose(
        params.verbose,
        "  [KS] Step 1: Building one-electron integrals (n={d}, libcint={})...\n",
        .{ n, params.use_libcint },
    );

    var nuc_pos_flat: []const [3]f64 = &.{};
    const cint_data_storage = try prepare_libcint_if_enabled(
        alloc,
        shells,
        nuc_positions,
        nuc_charges,
        params.use_libcint,
        &nuc_pos_flat,
    );
    const use_libcint_actual = cint_data_storage != null;

    const one_mats = try build_ks_one_electron_mats(
        alloc,
        shells,
        nuc_positions,
        nuc_charges,
        n,
        cint_data_storage,
        use_libcint_actual,
    );
    log_ks_step_done(params.verbose, "Step 1", io, timer);

    return .{
        .nuc_pos_flat = nuc_pos_flat,
        .cint_data_storage = cint_data_storage,
        .one_mats = one_mats,
        .use_libcint_actual = use_libcint_actual,
    };
}

const KsTwoElectronStage = struct {
    eri_table: ?obara_saika.GeneralEriTable,
    schwarz_table: ?fock.SchwarzTable,
    jk_builder: ?libcint.LibcintJKBuilder,
};

fn run_ks_step_two(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []const ContractedShell,
    cint_data_storage: ?libcint.LibcintData,
    use_libcint_actual: bool,
    params: KsParams,
) !KsTwoElectronStage {
    const timer = std.Io.Clock.Timestamp.now(io, .awake);
    logging.verbose(
        params.verbose,
        "  [KS] Step 2: Building Schwarz/ERI table (direct={}, libcint={})...\n",
        .{ params.use_direct_scf, use_libcint_actual },
    );
    var eri_table: ?obara_saika.GeneralEriTable = null;
    var schwarz_table: ?fock.SchwarzTable = null;
    var jk_builder: ?libcint.LibcintJKBuilder = null;
    try build_eri_or_schwarz_tables(
        alloc,
        shells,
        cint_data_storage,
        use_libcint_actual,
        params,
        &eri_table,
        &schwarz_table,
        &jk_builder,
    );
    log_ks_step_done(params.verbose, "Step 2", io, timer);
    return .{ .eri_table = eri_table, .schwarz_table = schwarz_table, .jk_builder = jk_builder };
}

fn prepare_ks_integral_resources(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []const ContractedShell,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    n: usize,
    params: KsParams,
) !KsIntegralResources {
    const stage1 = try run_ks_step_one(alloc, io, shells, nuc_positions, nuc_charges, n, params);
    const stage2 = try run_ks_step_two(
        alloc,
        io,
        shells,
        stage1.cint_data_storage,
        stage1.use_libcint_actual,
        params,
    );
    const df_context = try build_density_fitting_if_enabled(
        alloc,
        io,
        shells,
        nuc_positions,
        nuc_charges,
        n,
        params,
    );

    return .{
        .nuc_pos_flat = stage1.nuc_pos_flat,
        .cint_data_storage = stage1.cint_data_storage,
        .one_mats = stage1.one_mats,
        .eri_table = stage2.eri_table,
        .schwarz_table = stage2.schwarz_table,
        .jk_builder = stage2.jk_builder,
        .df_context = df_context,
        .use_libcint_actual = stage1.use_libcint_actual,
    };
}

fn run_ks_main_scc_loop(
    alloc: std.mem.Allocator,
    io: std.Io,
    n: usize,
    n_occ: usize,
    hf_frac: f64,
    v_nn: f64,
    h_core: []const f64,
    s_mat: []const f64,
    jk_ctx: KsJkCtx,
    grid_points: []const becke.GridPoint,
    bog: BasisOnGrid,
    params: KsParams,
    p_mat: []f64,
    eigen: *linalg.RealEigenDecomp,
) !struct { state: KsScfState, iter: usize } {
    var bufs = try alloc_ks_scf_buffers(alloc, n, grid_points.len, params);
    defer bufs.deinit(alloc);

    var state = init_ks_scf_state();

    var iter: usize = 0;
    while (iter < params.max_iter) : (iter += 1) {
        if (try run_ks_scf_iteration(
            alloc,
            io,
            iter,
            n,
            n_occ,
            hf_frac,
            v_nn,
            h_core,
            s_mat,
            jk_ctx,
            grid_points,
            bog,
            params.xc_functional,
            params,
            p_mat,
            bufs.p_old,
            bufs.f_mat,
            bufs.j_mat,
            bufs.k_mat,
            bufs.vxc_mat,
            bufs.xc_work_buf,
            bufs.f_diis,
            &bufs.diis,
            eigen,
            &state,
        )) break;
    }

    log_ks_scf_timing(params.verbose, state);
    try finalize_unconverged_ks_state(
        alloc,
        n,
        hf_frac,
        v_nn,
        h_core,
        jk_ctx,
        grid_points,
        p_mat,
        bog,
        params.xc_functional,
        &bufs,
        &state,
    );

    return .{ .state = state, .iter = iter };
}

fn run_ks_step_three(
    alloc: std.mem.Allocator,
    io: std.Io,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    params: KsParams,
) ![]becke.GridPoint {
    const timer = std.Io.Clock.Timestamp.now(io, .awake);
    logging.verbose(
        params.verbose,
        "  [KS] Step 3: Building molecular grid ({d} radial, {d} angular)...\n",
        .{ params.n_radial, params.n_angular },
    );
    const grid_points = try build_molecular_grid_from_nuclei(
        alloc,
        nuc_positions,
        nuc_charges,
        params,
    );
    log_ks_step3_done(params.verbose, grid_points.len, io, timer);
    return grid_points;
}

fn run_ks_step_four(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []const ContractedShell,
    grid_points: []const becke.GridPoint,
    params: KsParams,
) !BasisOnGrid {
    const timer = std.Io.Clock.Timestamp.now(io, .awake);
    logging.verbose(
        params.verbose,
        "  [KS] Step 4: Pre-evaluating basis functions on grid...\n",
        .{},
    );
    const bog = try evaluate_basis_on_grid(alloc, shells, grid_points);
    log_ks_step_done(params.verbose, "Step 4", io, timer);
    return bog;
}

fn run_ks_step_five(
    alloc: std.mem.Allocator,
    io: std.Io,
    n: usize,
    h_core: []const f64,
    s_mat: []const f64,
    params: KsParams,
) !linalg.RealEigenDecomp {
    const timer = std.Io.Clock.Timestamp.now(io, .awake);
    logging.verbose(params.verbose, "  [KS] Step 5: Initial guess (diagonalize H_core)...\n", .{});
    const eigen = try solve_roothaan_hall(alloc, n, h_core, s_mat);
    if (params.verbose) try log_ks_step5_verbose(alloc, io, timer, n, eigen.values, s_mat);
    return eigen;
}

/// Run a Kohn-Sham DFT SCF calculation.
pub fn run_kohn_sham_scf(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []const ContractedShell,
    nuc_positions: []const math_mod.Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    params: KsParams,
) !KsResult {
    std.debug.assert(n_electrons % 2 == 0);
    const n = obara_saika.total_basis_functions(shells);
    const n_occ = n_electrons / 2;
    const hf_frac = hf_exchange_fraction(params.xc_functional);

    // Steps 1, 2, 2b: one-electron integrals, ERI/Schwarz tables, DF context
    var resources = try prepare_ks_integral_resources(
        alloc,
        io,
        shells,
        nuc_positions,
        nuc_charges,
        n,
        params,
    );
    defer resources.deinit(alloc);

    const s_mat = resources.one_mats.s_mat;
    const h_core = resources.one_mats.h_core;

    var prep = try prepare_ks_scf_loop(
        alloc,
        io,
        shells,
        nuc_positions,
        nuc_charges,
        n,
        h_core,
        s_mat,
        params,
    );
    defer prep.deinit(alloc);

    const p_mat = try density_matrix.build_density_matrix(alloc, n, n_occ, prep.eigen.vectors);

    const loop_out = try run_ks_scf_loop(
        alloc,
        io,
        n,
        n_occ,
        hf_frac,
        shells,
        &resources,
        h_core,
        s_mat,
        &prep,
        params,
        p_mat,
    );

    return package_ks_result(loop_out, prep.v_nn, prep.eigen, p_mat);
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
fn build_s_inv_half_transform(
    alloc: std.mem.Allocator,
    n: usize,
    s_values: []const f64,
    s_vectors: []const f64,
    threshold: f64,
) !struct { x_mat: []f64, n_indep: usize } {
    var n_indep: usize = 0;
    for (0..n) |i| {
        if (s_values[i] > threshold) n_indep += 1;
    }
    const x_mat = try alloc.alloc(f64, n * n_indep);

    var col: usize = 0;
    for (0..n) |i| {
        if (s_values[i] > threshold) {
            const inv_sqrt_s = 1.0 / @sqrt(s_values[i]);
            for (0..n) |mu| {
                x_mat[col * n + mu] = s_vectors[i * n + mu] * inv_sqrt_s;
            }
            col += 1;
        }
    }
    return .{ .x_mat = x_mat, .n_indep = n_indep };
}

fn transform_fock_to_orth_basis(
    alloc: std.mem.Allocator,
    n: usize,
    n_indep: usize,
    f_mat: []const f64,
    x_mat: []const f64,
) ![]f64 {
    // temp = F * X (n x n_indep, column-major X)
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

    // F' = X^T * temp  (n_indep x n_indep)
    const f_prime = try alloc.alloc(f64, n_indep * n_indep);
    errdefer alloc.free(f_prime);

    for (0..n_indep) |i| {
        for (0..n_indep) |j| {
            var sum: f64 = 0.0;
            for (0..n) |mu| {
                sum += x_mat[i * n + mu] * temp[j * n + mu];
            }
            f_prime[i * n_indep + j] = sum;
        }
    }
    return f_prime;
}

fn back_transform_eigenvectors(
    alloc: std.mem.Allocator,
    n: usize,
    n_indep: usize,
    x_mat: []const f64,
    eigen_prime_values: []const f64,
    eigen_prime_vectors: []const f64,
) !linalg.RealEigenDecomp {
    const result_values = try alloc.alloc(f64, n);
    errdefer alloc.free(result_values);

    const result_vectors = try alloc.alloc(f64, n * n);
    errdefer alloc.free(result_vectors);

    // Fill eigenvalues: first n_indep from the eigensolver, rest = very large
    for (0..n_indep) |i| {
        result_values[i] = eigen_prime_values[i];
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
                sum += x_mat[k * n + mu] * eigen_prime_vectors[j * n_indep + k];
            }
            result_vectors[j * n + mu] = sum;
        }
    }

    return linalg.RealEigenDecomp{
        .values = result_values,
        .vectors = result_vectors,
        .n = n,
    };
}

fn solve_roothaan_hall(
    alloc: std.mem.Allocator,
    n: usize,
    f_mat: []const f64,
    s_mat: []const f64,
) !linalg.RealEigenDecomp {
    // Step 1: Diagonalize S matrix
    const s_copy = try alloc.alloc(f64, n * n);
    defer alloc.free(s_copy);

    @memcpy(s_copy, s_mat);

    const s_eigen = try linalg.real_symmetric_eigen_decomp(alloc, .accelerate, n, s_copy);
    defer alloc.free(s_eigen.values);
    // s_eigen.vectors is column-major: eigenvector j is at s_eigen.vectors[j*n .. (j+1)*n]
    defer alloc.free(s_eigen.vectors);

    // Step 2: Build S^{-1/2} = U * diag(1/sqrt(s)) * U^T
    const threshold: f64 = 1e-8;
    const xt = try build_s_inv_half_transform(alloc, n, s_eigen.values, s_eigen.vectors, threshold);
    const x_mat = xt.x_mat;
    const n_indep = xt.n_indep;
    defer alloc.free(x_mat);

    // Step 3: F' = X^T * F * X  (n_indep x n_indep)
    const f_prime = try transform_fock_to_orth_basis(alloc, n, n_indep, f_mat, x_mat);
    defer alloc.free(f_prime);

    // Step 4: Solve standard eigenvalue problem for F'
    var eigen_prime = try linalg.real_symmetric_eigen_decomp(alloc, .accelerate, n_indep, f_prime);
    defer alloc.free(eigen_prime.vectors);

    // Step 5: Back-transform C = X * C'
    const result = try back_transform_eigenvectors(
        alloc,
        n,
        n_indep,
        x_mat,
        eigen_prime.values,
        eigen_prime.vectors,
    );

    alloc.free(eigen_prime.values);
    eigen_prime.values = &.{}; // prevent double free in defer

    return result;
}

/// Solve the standard eigenvalue problem A·x = λ·x for a real symmetric matrix.
/// Used for computing S matrix eigenvalues (condition number check).
fn solve_standard_eigen(
    alloc: std.mem.Allocator,
    n: usize,
    a: []f64,
) !linalg.RealEigenDecomp {
    return try linalg.real_symmetric_eigen_decomp(alloc, .accelerate, n, a);
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
    const result = eval_basis_function(shell, ang, 0.0, 0.0, 0.0);
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
    const result = eval_basis_function(shell, ang_px, 0.0, 0.0, 0.0);
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
    var rhf_result = try gto_scf.run_general_rhf_scf(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        10,
        .{},
    );
    defer rhf_result.deinit(alloc);

    const n = obara_saika.total_basis_functions(&shells);

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

    const grid_points = try becke.build_molecular_grid(alloc, &atoms, grid_config);
    defer alloc.free(grid_points);

    // Evaluate basis on grid
    var bog = try evaluate_basis_on_grid(alloc, &shells, grid_points);
    defer bog.deinit(alloc);

    // Compute density on grid
    const dens = try compute_density_on_grid(
        alloc,
        n,
        grid_points.len,
        rhf_result.density_matrix,
        bog,
    );
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

    var result = try run_kohn_sham_scf(
        alloc,
        std.testing.io,
        &shells,
        &nuc_positions,
        &nuc_charges,
        10,
        .{
            .xc_functional = .lda_svwn,
            .n_radial = 99,
            .n_angular = 590,
            .prune = false,
        },
    );
    defer result.deinit(alloc);

    try testing.expect(result.converged);
    // PySCF grid-converged LDA: -74.7321048790
    try testing.expectApproxEqAbs(-74.7321048790, result.total_energy, 1e-3);
}

// Slow B3LYP/QM9 regression coverage lives in `regression_tests.zig`.
