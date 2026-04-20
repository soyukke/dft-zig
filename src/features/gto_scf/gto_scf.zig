//! Restricted Hartree-Fock SCF loop for Gaussian-type orbital basis.
//!
//! Algorithm:
//!   1. Build one-electron integrals: S, T, V → H_core = T + V
//!   2. Build two-electron integrals: ERI table
//!   3. Initial guess: diagonalize H_core (P=0 → F=H_core)
//!   4. SCF loop:
//!      a. Build Fock matrix: F = H_core + G(P)
//!      b. Solve generalized eigenvalue problem: FC = SCε
//!      c. Build density matrix: P = 2 × C_occ × C_occ†
//!      d. Compute energy: E = ½Tr[P(H_core+F)] + V_nn
//!      e. Check convergence: ΔE and ΔRMS(P)
//!   5. Return converged energy and MO coefficients
//!
//! Units: Hartree atomic units throughout.

const std = @import("std");
const math = @import("../math/math.zig");
const basis_mod = @import("../basis/basis.zig");
const integrals = @import("../integrals/integrals.zig");
const obara_saika = integrals.obara_saika;
const linalg = @import("../linalg/linalg.zig");
const density_matrix = @import("density_matrix.zig");
pub const fock = @import("fock.zig");
const energy_mod = @import("energy.zig");

const diis_mod = @import("diis.zig");
const GtoDiis = diis_mod.GtoDiis;
const logging = @import("logging.zig");
pub const kohn_sham = @import("kohn_sham.zig");
pub const gradient = @import("gradient.zig");
pub const optimizer = @import("optimizer.zig");
pub const vibrational = @import("vibrational.zig");
pub const molecule = @import("molecule.zig");
pub const density_fitting = @import("density_fitting.zig");

const ContractedShell = basis_mod.ContractedShell;

/// SCF convergence parameters.
pub const ScfParams = struct {
    /// Maximum number of SCF iterations.
    max_iter: usize = 100,
    /// Energy convergence threshold (Hartree).
    energy_threshold: f64 = 1e-8,
    /// Density matrix RMS convergence threshold.
    density_threshold: f64 = 1e-6,
    /// Enable DIIS acceleration.
    use_diis: bool = true,
    /// Maximum number of DIIS vectors to store.
    diis_max_vectors: usize = 6,
    /// Iteration at which DIIS starts (0 = from first iteration).
    diis_start_iter: usize = 1,
    /// Optional initial density matrix (row-major n*n).
    /// When provided, skip the core Hamiltonian diagonalization guess
    /// and use this density matrix as the starting point.
    /// Useful for finite-difference gradient validation where perturbed
    /// SCFs should start from the base SCF density to avoid converging
    /// to different solutions.
    initial_density: ?[]const f64 = null,
    /// Use direct SCF with Schwarz screening instead of ERI table.
    /// Essential for large basis sets (e.g. 6-31G(2df,p)) where the
    /// ERI table would be too expensive to build.
    use_direct_scf: bool = false,
    /// Schwarz screening threshold for direct SCF.
    schwarz_threshold: f64 = 1e-12,
};

/// Result of a converged RHF SCF calculation.
pub const ScfResult = struct {
    /// Total energy (Hartree): E_elec + V_nn.
    total_energy: f64,
    /// Electronic energy (Hartree).
    electronic_energy: f64,
    /// Nuclear repulsion energy (Hartree).
    nuclear_repulsion: f64,
    /// Orbital energies (eigenvalues), ascending order.
    orbital_energies: []f64,
    /// MO coefficient matrix (column-major n×n).
    mo_coefficients: []f64,
    /// Density matrix (row-major n×n).
    density_matrix: []f64,
    /// Number of SCF iterations performed.
    iterations: usize,
    /// Whether SCF converged.
    converged: bool,

    pub fn deinit(self: *ScfResult, alloc: std.mem.Allocator) void {
        if (self.orbital_energies.len > 0) alloc.free(self.orbital_energies);
        if (self.mo_coefficients.len > 0) alloc.free(self.mo_coefficients);
        if (self.density_matrix.len > 0) alloc.free(self.density_matrix);
    }
};

/// Run a Restricted Hartree-Fock SCF calculation (s-type shells only, legacy).
///
/// For general angular momentum, use runGeneralRhfScf.
pub fn runRhfScf(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    params: ScfParams,
) !ScfResult {
    std.debug.assert(n_electrons % 2 == 0); // RHF requires even electrons
    const n = shells.len; // number of basis functions (s-only: 1 per shell)
    const n_occ = n_electrons / 2; // number of doubly-occupied orbitals

    // Step 1: Build one-electron integrals
    const s_mat = try integrals.buildOverlapMatrix(alloc, shells);
    defer alloc.free(s_mat);

    const t_mat = try integrals.buildKineticMatrix(alloc, shells);
    defer alloc.free(t_mat);

    const v_mat = try integrals.buildNuclearMatrix(alloc, shells, nuc_positions, nuc_charges);
    defer alloc.free(v_mat);

    // H_core = T + V
    const h_core = try alloc.alloc(f64, n * n);
    defer alloc.free(h_core);
    for (0..n * n) |i| {
        h_core[i] = t_mat[i] + v_mat[i];
    }

    // Step 2: Build ERI table
    var eri_table = try integrals.buildEriTable(alloc, shells);
    defer eri_table.deinit(alloc);

    // Nuclear repulsion energy
    const v_nn = energy_mod.nuclearRepulsionEnergy(nuc_positions, nuc_charges);

    // Step 3: Initial guess — diagonalize H_core (F = H_core when P = 0)
    var eigen = try solveRoothaanHall(alloc, n, h_core, s_mat);

    // Build initial density matrix
    const p_mat = try density_matrix.buildDensityMatrix(alloc, n, n_occ, eigen.vectors);
    const p_old = try alloc.alloc(f64, n * n);
    defer alloc.free(p_old);
    @memset(p_old, 0.0);

    // Step 4: SCF loop
    var e_total: f64 = 0.0;
    var e_old: f64 = 0.0;
    var converged = false;
    var iter: usize = 0;

    // Allocate Fock matrix buffer
    const f_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(f_mat);

    while (iter < params.max_iter) : (iter += 1) {
        // Build Fock matrix
        fock.updateFockMatrix(n, h_core, p_mat, eri_table, f_mat);

        // Compute energy
        const e_elec = energy_mod.electronicEnergy(n, p_mat, h_core, f_mat);
        e_total = e_elec + v_nn;

        // Check convergence
        const delta_e = @abs(e_total - e_old);
        const rms_p = density_matrix.densityRmsDiff(n, p_mat, p_old);

        if (iter > 0 and delta_e < params.energy_threshold and rms_p < params.density_threshold) {
            converged = true;
            break;
        }

        e_old = e_total;
        @memcpy(p_old, p_mat);

        // Solve Roothaan-Hall: FC = SCε
        // Free previous eigen decomposition
        alloc.free(eigen.vectors);
        alloc.free(eigen.values);

        eigen = try solveRoothaanHall(alloc, n, f_mat, s_mat);

        // Update density matrix
        density_matrix.updateDensityMatrix(n, n_occ, eigen.vectors, p_mat);
    }

    // If not converged, do a final energy evaluation
    if (!converged) {
        fock.updateFockMatrix(n, h_core, p_mat, eri_table, f_mat);
        const e_elec = energy_mod.electronicEnergy(n, p_mat, h_core, f_mat);
        e_total = e_elec + v_nn;
    }

    // Package result — transfer ownership of eigen data and p_mat
    return ScfResult{
        .total_energy = e_total,
        .electronic_energy = e_total - v_nn,
        .nuclear_repulsion = v_nn,
        .orbital_energies = eigen.values,
        .mo_coefficients = eigen.vectors,
        .density_matrix = p_mat,
        .iterations = iter,
        .converged = converged,
    };
}

/// Run a Restricted Hartree-Fock SCF calculation for shells with any angular momentum.
///
/// Uses Obara-Saika integrals for overlap, kinetic, nuclear, and ERI.
/// Supports s, p, d, ... shells.
pub fn runGeneralRhfScf(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const math.Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    params: ScfParams,
) !ScfResult {
    std.debug.assert(n_electrons % 2 == 0);
    const n = obara_saika.totalBasisFunctions(shells);
    const n_occ = n_electrons / 2;

    // Step 1: Build one-electron integrals using Obara-Saika
    const s_mat = try obara_saika.buildOverlapMatrix(alloc, shells);
    defer alloc.free(s_mat);

    const t_mat = try obara_saika.buildKineticMatrix(alloc, shells);
    defer alloc.free(t_mat);

    const v_mat = try obara_saika.buildNuclearMatrix(alloc, shells, nuc_positions, nuc_charges);
    defer alloc.free(v_mat);

    // H_core = T + V
    const h_core = try alloc.alloc(f64, n * n);
    defer alloc.free(h_core);
    for (0..n * n) |i| {
        h_core[i] = t_mat[i] + v_mat[i];
    }

    // Step 2: Build ERI table or Schwarz table for direct SCF
    var eri_table: ?obara_saika.GeneralEriTable = null;
    var schwarz_table: ?fock.SchwarzTable = null;
    if (params.use_direct_scf) {
        schwarz_table = try fock.buildSchwarzTable(alloc, shells);
    } else {
        eri_table = try obara_saika.buildEriTable(alloc, shells);
    }
    defer {
        if (eri_table) |*et| et.deinit(alloc);
        if (schwarz_table) |*st| st.deinit(alloc);
    }

    // Nuclear repulsion energy
    const v_nn = energy_mod.nuclearRepulsionEnergy(nuc_positions, nuc_charges);

    // Step 3: Initial guess
    var eigen: linalg.RealEigenDecomp = undefined;
    var eigen_initialized = false;
    const p_mat: []f64 = if (params.initial_density) |init_p| blk: {
        // Use provided initial density matrix directly as starting point.
        // Skip core Hamiltonian diagonalization. The eigen decomposition will
        // be computed in the first SCF iteration when solving Roothaan-Hall.
        const p_init = try alloc.alloc(f64, n * n);
        @memcpy(p_init, init_p);
        break :blk p_init;
    } else blk: {
        // Default: diagonalize H_core (P=0 → F=H_core)
        eigen = try solveRoothaanHall(alloc, n, h_core, s_mat);
        eigen_initialized = true;
        break :blk try density_matrix.buildDensityMatrix(alloc, n, n_occ, eigen.vectors);
    };
    const p_old = try alloc.alloc(f64, n * n);
    defer alloc.free(p_old);
    @memset(p_old, 0.0);

    // Step 4: SCF loop
    var e_total: f64 = 0.0;
    var e_old: f64 = 0.0;
    var converged = false;
    var iter: usize = 0;

    const f_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(f_mat);

    // DIIS accelerator (Fock-matrix extrapolation)
    var diis: ?GtoDiis = if (params.use_diis) GtoDiis.init(alloc, n, params.diis_max_vectors) else null;
    defer if (diis) |*d| d.deinit();

    // Buffer for DIIS-extrapolated Fock matrix
    const f_diis = if (params.use_diis) try alloc.alloc(f64, n * n) else null;
    defer if (f_diis) |buf| alloc.free(buf);

    while (iter < params.max_iter) : (iter += 1) {
        // Build Fock matrix
        if (params.use_direct_scf) {
            fock.buildFockDirect(n, h_core, p_mat, shells, &schwarz_table.?, params.schwarz_threshold, f_mat);
        } else {
            fock.updateFockMatrixGeneral(n, h_core, p_mat, eri_table.?, f_mat);
        }

        // Compute energy using the un-extrapolated Fock matrix
        const e_elec = energy_mod.electronicEnergy(n, p_mat, h_core, f_mat);
        e_total = e_elec + v_nn;

        // Check convergence
        const delta_e = @abs(e_total - e_old);
        const rms_p = density_matrix.densityRmsDiff(n, p_mat, p_old);

        if (iter > 0 and delta_e < params.energy_threshold and rms_p < params.density_threshold) {
            converged = true;
            break;
        }

        e_old = e_total;
        @memcpy(p_old, p_mat);

        // Apply DIIS extrapolation to the Fock matrix
        const f_to_diag = if (diis != null and iter >= params.diis_start_iter) blk: {
            try diis.?.extrapolate(f_mat, p_mat, s_mat, f_diis.?);
            break :blk f_diis.?;
        } else f_mat;

        if (eigen_initialized) {
            alloc.free(eigen.vectors);
            alloc.free(eigen.values);
        }

        eigen = try solveRoothaanHall(alloc, n, f_to_diag, s_mat);
        eigen_initialized = true;

        density_matrix.updateDensityMatrix(n, n_occ, eigen.vectors, p_mat);
    }

    if (!converged) {
        if (params.use_direct_scf) {
            fock.buildFockDirect(n, h_core, p_mat, shells, &schwarz_table.?, params.schwarz_threshold, f_mat);
        } else {
            fock.updateFockMatrixGeneral(n, h_core, p_mat, eri_table.?, f_mat);
        }
        const e_elec = energy_mod.electronicEnergy(n, p_mat, h_core, f_mat);
        e_total = e_elec + v_nn;
    }

    return ScfResult{
        .total_energy = e_total,
        .electronic_energy = e_total - v_nn,
        .nuclear_repulsion = v_nn,
        .orbital_energies = eigen.values,
        .mo_coefficients = eigen.vectors,
        .density_matrix = p_mat,
        .iterations = iter,
        .converged = converged,
    };
}

/// Solve the Roothaan-Hall equation FC = SCε using the generalized
/// eigenvalue solver. Uses LAPACK dsygv (via Accelerate).
fn solveRoothaanHall(
    alloc: std.mem.Allocator,
    n: usize,
    f_mat: []const f64,
    s_mat: []const f64,
) !linalg.RealEigenDecomp {
    const a = try alloc.alloc(f64, n * n);
    defer alloc.free(a);
    @memcpy(a, f_mat);

    const b = try alloc.alloc(f64, n * n);
    defer alloc.free(b);
    @memcpy(b, s_mat);

    return try linalg.realSymmetricGenEigenDecomp(alloc, .accelerate, n, a, b);
}

/// Print SCF iteration info to stderr for debugging.
pub fn printIterInfo(
    iter: usize,
    e_total: f64,
    delta_e: f64,
    rms_p: f64,
) void {
    logging.verbose(true, "SCF iter {d:3}: E = {d:16.10} Ha, dE = {e:10.3}, dP = {e:10.3}\n", .{
        iter, e_total, delta_e, rms_p,
    });
}

test "SCF module imports" {
    _ = density_matrix;
    _ = fock;
    _ = energy_mod;
    _ = kohn_sham;
    _ = gradient;
    _ = optimizer;
    _ = vibrational;
    _ = molecule;
    _ = density_fitting;
}

test "RHF H2 STO-3G at R=1.4 bohr" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    const r = 1.4;
    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = r, .y = 0.0, .z = 0.0 },
    };
    const nuc_charges = [_]f64{ 1.0, 1.0 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    var result = try runRhfScf(
        alloc,
        &shells,
        &nuc_positions,
        &nuc_charges,
        2,
        .{},
    );
    defer result.deinit(alloc);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(-1.1167, result.total_energy, 0.001);
    try testing.expectApproxEqAbs(1.0 / 1.4, result.nuclear_repulsion, 1e-10);
    try testing.expect(result.iterations < 20);
    try testing.expect(result.orbital_energies[0] < 0.0);

    std.debug.print("\nH2 STO-3G RHF Results:\n", .{});
    std.debug.print("  Total energy:     {d:.10} Ha\n", .{result.total_energy});
    std.debug.print("  Electronic energy: {d:.10} Ha\n", .{result.electronic_energy});
    std.debug.print("  Nuclear repulsion: {d:.10} Ha\n", .{result.nuclear_repulsion});
    std.debug.print("  Iterations:        {d}\n", .{result.iterations});
    std.debug.print("  Orbital energies:  {d:.6}, {d:.6}\n", .{ result.orbital_energies[0], result.orbital_energies[1] });
}

test "General RHF H2 STO-3G matches legacy" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    const r = 1.4;
    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = r, .y = 0.0, .z = 0.0 },
    };
    const nuc_charges = [_]f64{ 1.0, 1.0 };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Legacy s-only SCF
    var result_legacy = try runRhfScf(alloc, &shells, &nuc_positions, &nuc_charges, 2, .{});
    defer result_legacy.deinit(alloc);

    // General SCF (should give same result for s-only shells)
    var result_general = try runGeneralRhfScf(alloc, &shells, &nuc_positions, &nuc_charges, 2, .{});
    defer result_general.deinit(alloc);

    try testing.expectApproxEqAbs(result_legacy.total_energy, result_general.total_energy, 1e-10);
    try testing.expect(result_general.converged);
}

test "Nuclear attraction p-orbital symmetry at origin" {
    const testing = std.testing;
    const sto3g = @import("../basis/sto3g.zig");

    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const nuc_pos = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const shell_2p = ContractedShell{
        .center = center,
        .l = 1,
        .primitives = &sto3g.O_2p,
    };

    const px = basis_mod.AngularMomentum{ .x = 1, .y = 0, .z = 0 };
    const py = basis_mod.AngularMomentum{ .x = 0, .y = 1, .z = 0 };
    const pz = basis_mod.AngularMomentum{ .x = 0, .y = 0, .z = 1 };

    // Nuclear attraction for a single nucleus at origin with Z=8
    const v_xx = obara_saika.contractedNuclearAttraction(shell_2p, px, shell_2p, px, nuc_pos, 8.0);
    const v_yy = obara_saika.contractedNuclearAttraction(shell_2p, py, shell_2p, py, nuc_pos, 8.0);
    const v_zz = obara_saika.contractedNuclearAttraction(shell_2p, pz, shell_2p, pz, nuc_pos, 8.0);

    std.debug.print("\nNuclear attraction <p|V|p> at origin (Z=8):\n", .{});
    std.debug.print("  V_xx = {d:.10}\n", .{v_xx});
    std.debug.print("  V_yy = {d:.10}\n", .{v_yy});
    std.debug.print("  V_zz = {d:.10}\n", .{v_zz});
    std.debug.print("  PySCF = -8.9794547266\n", .{});

    // These should all be equal by spherical symmetry
    try testing.expectApproxEqAbs(v_xx, v_yy, 1e-10);
    try testing.expectApproxEqAbs(v_xx, v_zz, 1e-10);

    // Check against PySCF reference
    try testing.expectApproxEqAbs(-8.9794547266, v_xx, 1e-4);

    // Now test with a displaced nucleus (H at y=1.43, z=1.11)
    const h_pos = math.Vec3{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 };
    const v_xx_h = obara_saika.contractedNuclearAttraction(shell_2p, px, shell_2p, px, h_pos, 1.0);
    const v_yy_h = obara_saika.contractedNuclearAttraction(shell_2p, py, shell_2p, py, h_pos, 1.0);
    const v_zz_h = obara_saika.contractedNuclearAttraction(shell_2p, pz, shell_2p, pz, h_pos, 1.0);
    const v_yz_h = obara_saika.contractedNuclearAttraction(shell_2p, py, shell_2p, pz, h_pos, 1.0);

    std.debug.print("\nNuclear attraction <p|V_H|p> with H at (0, 1.43, 1.11) (Z=1):\n", .{});
    std.debug.print("  V_xx = {d:.10}  (PySCF: -0.5031599190)\n", .{v_xx_h});
    std.debug.print("  V_yy = {d:.10}  (PySCF: -0.5810854958)\n", .{v_yy_h});
    std.debug.print("  V_zz = {d:.10}  (PySCF: -0.5500158390)\n", .{v_zz_h});
    std.debug.print("  V_yz = {d:.10}  (PySCF: -0.0604257775)\n", .{v_yz_h});

    // PySCF reference for H nucleus at (0, 1.43, 1.11) with Z=1
    try testing.expectApproxEqAbs(-0.5031599190, v_xx_h, 1e-4);
    try testing.expectApproxEqAbs(-0.5810854958, v_yy_h, 1e-4);
    try testing.expectApproxEqAbs(-0.5500158390, v_zz_h, 1e-4);
    try testing.expectApproxEqAbs(-0.0604257775, v_yz_h, 1e-4);
}

test "General RHF H2O STO-3G integrals" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    // H2O geometry in bohr (from PySCF)
    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 }, // O
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 }, // H
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 }, // H
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    // Build shells: O has 1s, 2s, 2p; H has 1s
    const shells = [_]ContractedShell{
        // Oxygen 1s
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        // Oxygen 2s
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        // Oxygen 2p (one shell, 3 Cartesian functions: px, py, pz)
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        // Hydrogen 1s (atom 1)
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        // Hydrogen 1s (atom 2)
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    const n = obara_saika.totalBasisFunctions(&shells);
    try testing.expectEqual(@as(usize, 7), n);

    // Build S, T, V matrices
    const s_mat = try obara_saika.buildOverlapMatrix(alloc, &shells);
    defer alloc.free(s_mat);
    const t_mat = try obara_saika.buildKineticMatrix(alloc, &shells);
    defer alloc.free(t_mat);
    const v_mat = try obara_saika.buildNuclearMatrix(alloc, &shells, &nuc_positions, &nuc_charges);
    defer alloc.free(v_mat);

    std.debug.print("\n=== H2O STO-3G Integral Diagnostics ===\n", .{});
    std.debug.print("Overlap matrix S:\n", .{});
    for (0..n) |i| {
        for (0..n) |j| {
            std.debug.print("{d:12.8} ", .{s_mat[i * n + j]});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("\nKinetic matrix T:\n", .{});
    for (0..n) |i| {
        for (0..n) |j| {
            std.debug.print("{d:12.8} ", .{t_mat[i * n + j]});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("\nNuclear attraction matrix V:\n", .{});
    for (0..n) |i| {
        for (0..n) |j| {
            std.debug.print("{d:12.8} ", .{v_mat[i * n + j]});
        }
        std.debug.print("\n", .{});
    }

    // Check S matrix against PySCF
    // S[0][0] should be 1.0 (O 1s self-overlap)
    try testing.expectApproxEqAbs(@as(f64, 1.0), s_mat[0 * n + 0], 1e-4);
    // S[0][1] = 0.23670394
    try testing.expectApproxEqAbs(@as(f64, 0.23670394), s_mat[0 * n + 1], 1e-4);
    // S[5][5] = 1.0 (H1 1s self-overlap)
    try testing.expectApproxEqAbs(@as(f64, 1.0), s_mat[5 * n + 5], 1e-4);

    // Check T matrix diagonal elements
    // T[0][0] = 29.0031999
    try testing.expectApproxEqAbs(@as(f64, 29.0031999), t_mat[0 * n + 0], 0.01);
    // T[2][2] = T[3][3] = T[4][4] = 2.52873120 (O 2p kinetic)
    try testing.expectApproxEqAbs(@as(f64, 2.52873120), t_mat[2 * n + 2], 0.01);

    // Check V matrix diagonal
    // V[0][0] = -61.7232969
    try testing.expectApproxEqAbs(@as(f64, -61.7232969), v_mat[0 * n + 0], 0.01);
}

test "General RHF H2O STO-3G" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    // H2O geometry in bohr (from PySCF)
    const nuc_positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 }, // O
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 }, // H
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 }, // H
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };

    // Build shells: O has 1s, 2s, 2p; H has 1s
    const shells = [_]ContractedShell{
        // Oxygen 1s
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        // Oxygen 2s
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        // Oxygen 2p (one shell, 3 Cartesian functions: px, py, pz)
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        // Hydrogen 1s (atom 1)
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        // Hydrogen 1s (atom 2)
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Total basis functions: 1 + 1 + 3 + 1 + 1 = 7
    const n_basis = obara_saika.totalBasisFunctions(&shells);
    try testing.expectEqual(@as(usize, 7), n_basis);

    // 10 electrons (O=8, H=1, H=1)
    var result = try runGeneralRhfScf(alloc, &shells, &nuc_positions, &nuc_charges, 10, .{});
    defer result.deinit(alloc);

    std.debug.print("\nH2O STO-3G RHF Results:\n", .{});
    std.debug.print("  Total energy:     {d:.10} Ha\n", .{result.total_energy});
    std.debug.print("  Electronic energy: {d:.10} Ha\n", .{result.electronic_energy});
    std.debug.print("  Nuclear repulsion: {d:.10} Ha\n", .{result.nuclear_repulsion});
    std.debug.print("  Iterations:        {d}\n", .{result.iterations});
    std.debug.print("  Converged:         {}\n", .{result.converged});
    std.debug.print("  Orbital energies:", .{});
    for (result.orbital_energies) |e| {
        std.debug.print(" {d:.6}", .{e});
    }
    std.debug.print("\n", .{});

    // Reference: PySCF RHF/STO-3G H2O = -74.9630631297 Ha
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(-74.9630631297, result.total_energy, 1e-4);

    // Nuclear repulsion: 9.1882584177 Ha
    try testing.expectApproxEqAbs(9.1882584177, result.nuclear_repulsion, 1e-6);
}

test "DIIS accelerates H2O convergence" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");

    // H2O geometry in bohr
    const nuc_positions = [_]math.Vec3{
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

    // Run WITHOUT DIIS
    var result_no_diis = try runGeneralRhfScf(alloc, &shells, &nuc_positions, &nuc_charges, 10, .{
        .use_diis = false,
    });
    defer result_no_diis.deinit(alloc);

    // Run WITH DIIS (default)
    var result_diis = try runGeneralRhfScf(alloc, &shells, &nuc_positions, &nuc_charges, 10, .{
        .use_diis = true,
    });
    defer result_diis.deinit(alloc);

    std.debug.print("\nDIIS comparison for H2O STO-3G:\n", .{});
    std.debug.print("  Without DIIS: {d} iterations, E = {d:.10} Ha\n", .{ result_no_diis.iterations, result_no_diis.total_energy });
    std.debug.print("  With DIIS:    {d} iterations, E = {d:.10} Ha\n", .{ result_diis.iterations, result_diis.total_energy });

    // Both should converge to the same energy
    try testing.expect(result_no_diis.converged);
    try testing.expect(result_diis.converged);
    try testing.expectApproxEqAbs(result_no_diis.total_energy, result_diis.total_energy, 1e-7);

    // DIIS should require fewer iterations (or at most the same)
    try testing.expect(result_diis.iterations <= result_no_diis.iterations);
}

// Slow larger-basis regression coverage lives in `regression_tests.zig`.
