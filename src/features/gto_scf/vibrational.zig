//! Vibrational analysis for GTO-based Hartree-Fock calculations.
//!
//! Computes vibrational frequencies, zero-point vibrational energy (ZPVE),
//! and thermodynamic properties from a numerical Hessian (finite differences
//! of analytical gradients).
//!
//! Algorithm:
//!   1. At the equilibrium geometry, compute numerical Hessian by displacing
//!      each nuclear coordinate by ±δ and computing the gradient at each
//!      displaced geometry (central differences).
//!   2. Symmetrize the Hessian: H_sym = (H + H^T) / 2.
//!   3. Mass-weight the Hessian: H_mw[i,j] = H[i,j] / sqrt(m_i * m_j).
//!   4. Diagonalize H_mw to get eigenvalues (ω²) and eigenvectors (normal modes).
//!   5. Convert eigenvalues to frequencies in cm⁻¹.
//!   6. Compute ZPVE and thermodynamic corrections.
//!
//! Units: Hartree atomic units throughout.
//!   - Hessian: Ha/Bohr²
//!   - Masses: AMU
//!   - Frequencies: cm⁻¹

const std = @import("std");
const math = @import("../math/math.zig");
const Vec3 = math.Vec3;
const basis_mod = @import("../basis/basis.zig");
const ContractedShell = basis_mod.ContractedShell;
const gto_scf = @import("gto_scf.zig");
const ScfParams = gto_scf.ScfParams;
const gradient_mod = gto_scf.gradient;
const logging = @import("logging.zig");

// LAPACK dsyev extern
extern fn dsyev_(
    jobz: [*]u8,
    uplo: [*]u8,
    n: *c_int,
    a: [*]f64,
    lda: *c_int,
    w: [*]f64,
    work: [*]f64,
    lwork: *c_int,
    info: *c_int,
) callconv(.c) void;

var lapack_mutex: @import("../../lib/spinlock.zig").SpinLock = .{};

// ============================================================================
// Physical constants
// ============================================================================

/// Hartree to Joule
const ha_to_j: f64 = 4.3597447222071e-18;
/// Bohr radius in meters
const bohr_to_m: f64 = 5.29177210903e-11;
/// Atomic mass unit in kg
const amu_to_kg: f64 = 1.66053906660e-27;
/// Speed of light in cm/s
const c_cm_s: f64 = 2.99792458e10;
/// Boltzmann constant in J/K
const k_b: f64 = 1.380649e-23;
/// Planck constant in J·s
const h_planck: f64 = 6.62607015e-34;
/// Boltzmann constant in Ha/K
const k_b_ha: f64 = k_b / ha_to_j;

/// Convert Hartree to kcal/mol.
const ha_to_kcal: f64 = 627.509;

/// Pi constant.
const pi: f64 = std.math.pi;

/// Atomic masses (most abundant isotope) in AMU, indexed by atomic number.
/// Using integer masses to match PySCF default behavior.
pub fn atomicMass(z: u8) f64 {
    return switch (z) {
        1 => 1.008, // H
        2 => 4.003, // He
        6 => 12.011, // C
        7 => 14.007, // N
        8 => 15.999, // O
        9 => 18.998, // F
        else => 1.0,
    };
}

/// Atomic masses matching PySCF defaults (integer values).
pub fn atomicMassInt(z: u8) f64 {
    return switch (z) {
        1 => 1.0, // H
        2 => 4.0, // He
        6 => 12.0, // C
        7 => 14.0, // N
        8 => 16.0, // O
        9 => 19.0, // F
        else => 1.0,
    };
}

// ============================================================================
// Parameters and result types
// ============================================================================

/// Parameters for vibrational analysis.
pub const VibrationalParams = struct {
    /// Displacement for finite differences (Bohr).
    displacement: f64 = 0.001,
    /// SCF parameters for gradient calculations.
    scf_params: ScfParams = .{},
    /// Temperature for thermodynamic properties (K).
    temperature: f64 = 298.15,
    /// Use integer masses (PySCF compatible) vs IUPAC masses.
    use_integer_masses: bool = true,
    /// Print progress.
    print_progress: bool = true,
    /// Threshold below which a frequency is considered translational/rotational (cm⁻¹).
    rot_trans_threshold: f64 = 100.0,
};

/// Result of vibrational analysis.
pub const VibrationalResult = struct {
    /// All eigenvalues of mass-weighted Hessian (Ha/(Bohr²·AMU)), length 3*n_atoms.
    eigenvalues: []f64,
    /// All frequencies in cm⁻¹ (negative for imaginary), length 3*n_atoms.
    frequencies_cm1: []f64,
    /// Vibrational frequencies only (excluding translations/rotations), in cm⁻¹.
    vib_frequencies_cm1: []f64,
    /// Normal mode eigenvectors (column-major, 3*n_atoms × 3*n_atoms).
    normal_modes: []f64,
    /// Raw Hessian matrix (Ha/Bohr², 3*n_atoms × 3*n_atoms).
    hessian: []f64,
    /// Zero-point vibrational energy (Ha).
    zpve: f64,
    /// Number of atoms.
    n_atoms: usize,

    pub fn deinit(self: *VibrationalResult, alloc: std.mem.Allocator) void {
        if (self.eigenvalues.len > 0) alloc.free(self.eigenvalues);
        if (self.frequencies_cm1.len > 0) alloc.free(self.frequencies_cm1);
        if (self.vib_frequencies_cm1.len > 0) alloc.free(self.vib_frequencies_cm1);
        if (self.normal_modes.len > 0) alloc.free(self.normal_modes);
        if (self.hessian.len > 0) alloc.free(self.hessian);
    }
};

// ============================================================================
// Unit conversion
// ============================================================================

/// Convert eigenvalue of mass-weighted Hessian (Ha/(Bohr²·AMU)) to cm⁻¹.
/// ω = sqrt(|λ| * Ha_to_J / (Bohr_to_m² * AMU_to_kg))  [rad/s]
/// ν = ω / (2π c)  [cm⁻¹]
pub fn eigenvalueToCm1(eigenvalue: f64) f64 {
    const conv = ha_to_j / (bohr_to_m * bohr_to_m * amu_to_kg);
    if (eigenvalue >= 0.0) {
        const omega_si = @sqrt(eigenvalue * conv);
        return omega_si / (2.0 * std.math.pi * c_cm_s);
    } else {
        // Imaginary frequency (saddle point / translation / rotation)
        const omega_si = @sqrt(-eigenvalue * conv);
        return -omega_si / (2.0 * std.math.pi * c_cm_s);
    }
}

/// Convert frequency in cm⁻¹ to Hartree.
/// E = h * c * ν_cm⁻¹  (in Joules), then convert to Ha.
/// c is in cm/s, so h * c_cm_s * ν gives energy directly.
pub fn cm1ToHartree(freq_cm1: f64) f64 {
    return h_planck * c_cm_s * freq_cm1 / ha_to_j;
}

// ============================================================================
// Thermodynamic properties
// ============================================================================

/// Result of thermodynamic analysis.
pub const ThermoResult = struct {
    /// Temperature (K).
    temperature: f64,
    /// Pressure (Pa).
    pressure: f64,

    // --- Electronic contribution ---
    /// Electronic energy E0 from SCF (Ha).
    e_elec: f64,

    // --- Translational contribution ---
    /// Translational entropy (Ha/K).
    s_trans: f64,
    /// Translational internal energy (Ha).
    e_trans: f64,
    /// Translational Cv (Ha/K).
    cv_trans: f64,

    // --- Rotational contribution ---
    /// Rotational constants (GHz), sorted descending. For linear: [inf, B, B].
    rot_const_ghz: [3]f64,
    /// Symmetry number.
    sym_number: u32,
    /// Whether molecule is linear.
    is_linear: bool,
    /// Rotational entropy (Ha/K).
    s_rot: f64,
    /// Rotational internal energy (Ha).
    e_rot: f64,
    /// Rotational Cv (Ha/K).
    cv_rot: f64,

    // --- Vibrational contribution ---
    /// Zero-point vibrational energy (Ha).
    zpve: f64,
    /// Vibrational entropy (Ha/K).
    s_vib: f64,
    /// Vibrational internal energy including ZPE (Ha).
    e_vib: f64,
    /// Vibrational Cv (Ha/K).
    cv_vib: f64,

    // --- Totals ---
    /// E at 0K = E_elec + ZPE.
    e_0k: f64,
    /// Total internal energy (Ha).
    e_tot: f64,
    /// Total enthalpy H = E_tot + kT (Ha).
    h_tot: f64,
    /// Total entropy (Ha/K).
    s_tot: f64,
    /// Total Gibbs free energy G = H - TS (Ha).
    g_tot: f64,
};

/// Compute the inertia tensor for a set of atoms.
/// Returns the 3x3 tensor in a flat array (row-major).
fn computeInertiaTensor(
    positions: []const Vec3,
    atomic_numbers: []const u8,
    use_integer_masses: bool,
) [9]f64 {
    const n_atoms = positions.len;

    // Compute center of mass
    var com = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    var total_mass: f64 = 0.0;
    for (0..n_atoms) |i| {
        const m = if (use_integer_masses)
            atomicMassInt(atomic_numbers[i])
        else
            atomicMass(atomic_numbers[i]);
        com.x += m * positions[i].x;
        com.y += m * positions[i].y;
        com.z += m * positions[i].z;
        total_mass += m;
    }
    com.x /= total_mass;
    com.y /= total_mass;
    com.z /= total_mass;

    // Build inertia tensor relative to COM
    var tensor = [_]f64{0.0} ** 9;
    for (0..n_atoms) |i| {
        const m = if (use_integer_masses)
            atomicMassInt(atomic_numbers[i])
        else
            atomicMass(atomic_numbers[i]);
        const rx = positions[i].x - com.x;
        const ry = positions[i].y - com.y;
        const rz = positions[i].z - com.z;
        const r2 = rx * rx + ry * ry + rz * rz;

        // Diagonal
        tensor[0] += m * (r2 - rx * rx); // Ixx
        tensor[4] += m * (r2 - ry * ry); // Iyy
        tensor[8] += m * (r2 - rz * rz); // Izz
        // Off-diagonal
        tensor[1] -= m * rx * ry; // Ixy
        tensor[3] -= m * ry * rx; // Iyx
        tensor[2] -= m * rx * rz; // Ixz
        tensor[6] -= m * rz * rx; // Izx
        tensor[5] -= m * ry * rz; // Iyz
        tensor[7] -= m * rz * ry; // Izy
    }

    return tensor;
}

/// Diagonalize a 3x3 real symmetric matrix using the analytical method.
/// Returns eigenvalues in descending order.
fn diag3x3(tensor: [9]f64) [3]f64 {
    // Use Cardano's method for 3x3 symmetric matrix eigenvalues.
    const a = tensor[0];
    const b = tensor[4];
    const c = tensor[8];
    const d = tensor[1]; // off-diag xy
    const e = tensor[2]; // off-diag xz
    const f = tensor[5]; // off-diag yz

    const p1 = d * d + e * e + f * f;
    if (p1 < 1e-30) {
        // Already diagonal
        var evals = [3]f64{ a, b, c };
        // Sort descending
        if (evals[0] < evals[1]) std.mem.swap(f64, &evals[0], &evals[1]);
        if (evals[1] < evals[2]) std.mem.swap(f64, &evals[1], &evals[2]);
        if (evals[0] < evals[1]) std.mem.swap(f64, &evals[0], &evals[1]);
        return evals;
    }

    const q = (a + b + c) / 3.0;
    const p2 = (a - q) * (a - q) + (b - q) * (b - q) + (c - q) * (c - q) + 2.0 * p1;
    const p = @sqrt(p2 / 6.0);

    // B = (1/p) * (A - qI)
    const inv_p = 1.0 / p;
    const b11 = (a - q) * inv_p;
    const b22 = (b - q) * inv_p;
    const b33 = (c - q) * inv_p;
    const b12 = d * inv_p;
    const b13 = e * inv_p;
    const b23 = f * inv_p;

    // det(B)
    const det_b =
        b11 * (b22 * b33 - b23 * b23) -
        b12 * (b12 * b33 - b23 * b13) +
        b13 * (b12 * b23 - b22 * b13);

    // r = det(B) / 2, clamp to [-1, 1]
    var r = det_b / 2.0;
    if (r <= -1.0) r = -1.0;
    if (r >= 1.0) r = 1.0;

    const phi = std.math.acos(r) / 3.0;

    var evals: [3]f64 = undefined;
    evals[0] = q + 2.0 * p * @cos(phi);
    evals[2] = q + 2.0 * p * @cos(phi + (2.0 * pi / 3.0));
    evals[1] = 3.0 * q - evals[0] - evals[2]; // trace = sum of eigenvalues

    // Sort descending
    if (evals[0] < evals[1]) std.mem.swap(f64, &evals[0], &evals[1]);
    if (evals[1] < evals[2]) std.mem.swap(f64, &evals[1], &evals[2]);
    if (evals[0] < evals[1]) std.mem.swap(f64, &evals[0], &evals[1]);

    return evals;
}

/// Compute principal moments of inertia (AMU·Bohr²) sorted descending.
/// Also returns whether molecule is linear.
fn principalMoments(
    positions: []const Vec3,
    atomic_numbers: []const u8,
    use_integer_masses: bool,
) struct { moments: [3]f64, is_linear: bool } {
    const tensor = computeInertiaTensor(positions, atomic_numbers, use_integer_masses);
    const evals = diag3x3(tensor);

    // Linear if smallest moment is near zero
    const is_linear = (evals[2] < 1e-6);

    return .{ .moments = evals, .is_linear = is_linear };
}

/// Convert principal moment of inertia (AMU·Bohr²) to rotational constant (GHz).
/// B = h / (8π² I)  where I is in kg·m².
fn momentToRotConst(moment_amu_bohr2: f64) f64 {
    if (moment_amu_bohr2 < 1e-10) return std.math.inf(f64);
    // Convert to SI: I_si = moment * amu_to_kg * bohr_to_m^2
    const i_si = moment_amu_bohr2 * amu_to_kg * bohr_to_m * bohr_to_m;
    // B = h / (8π²I) in Hz, then convert to GHz
    return h_planck / (8.0 * pi * pi * i_si) * 1e-9;
}

/// Compute thermodynamic properties from vibrational analysis results.
///
/// Uses the ideal gas / rigid rotor / harmonic oscillator (IGRRHO) model:
/// - Translational: ideal gas (Sackur-Tetrode equation)
/// - Rotational: rigid rotor (classical, high-T limit)
/// - Vibrational: quantum harmonic oscillator
/// - Electronic: ground state only (S_elec = 0 for closed-shell singlet)
pub fn computeThermo(
    vib_result: *const VibrationalResult,
    positions: []const Vec3,
    atomic_numbers: []const u8,
    e_elec: f64,
    temperature: f64,
    pressure: f64,
    sym_number: u32,
    use_integer_masses: bool,
) ThermoResult {
    const kT = k_b_ha * temperature;
    const n_atoms = positions.len;

    // ---- Total molecular mass ----
    var total_mass_amu: f64 = 0.0;
    for (0..n_atoms) |i| {
        total_mass_amu += if (use_integer_masses)
            atomicMassInt(atomic_numbers[i])
        else
            atomicMass(atomic_numbers[i]);
    }

    // ==== Translational contribution (ideal gas) ====
    // E_trans = (3/2) kT
    const e_trans = 1.5 * kT;
    // Cv_trans = (3/2) k
    const cv_trans = 1.5 * k_b_ha;
    // S_trans: Sackur-Tetrode equation
    // S = k * [5/2 + ln( (2πmkT/h²)^(3/2) * kT/P )]
    // where m is in kg, kT and P in SI
    const m_kg = total_mass_amu * amu_to_kg;
    const kT_si = k_b * temperature;
    const thermal_wavelength_factor = (2.0 * pi * m_kg * kT_si) / (h_planck * h_planck);
    const vol_per_particle = kT_si / pressure; // V/N = kT/P
    const q_trans = std.math.pow(f64, thermal_wavelength_factor, 1.5) * vol_per_particle;
    const s_trans = k_b_ha * (2.5 + @log(q_trans));

    // ==== Rotational contribution (rigid rotor) ====
    const pm = principalMoments(positions, atomic_numbers, use_integer_masses);
    const rot_const = [3]f64{
        momentToRotConst(pm.moments[2]), // Largest B from smallest I
        momentToRotConst(pm.moments[1]),
        momentToRotConst(pm.moments[0]), // Smallest B from largest I
    };

    var e_rot: f64 = undefined;
    var cv_rot: f64 = undefined;
    var s_rot: f64 = undefined;

    if (pm.is_linear) {
        // Linear molecule: 2 rotational DOF
        // E_rot = kT, Cv_rot = k
        e_rot = kT;
        cv_rot = k_b_ha;
        // S_rot = k * (1 + ln(kT / (σ * h * B)))
        // where B is rot const in Hz
        // q_rot = kT / (σ * h * B_Hz)
        const b_hz = rot_const[1] * 1e9; // GHz -> Hz
        const sigma_f: f64 = @floatFromInt(sym_number);
        const q_rot = kT_si / (sigma_f * h_planck * b_hz);
        s_rot = k_b_ha * (1.0 + @log(q_rot));
    } else {
        // Nonlinear molecule: 3 rotational DOF
        // E_rot = (3/2) kT, Cv_rot = (3/2) k
        e_rot = 1.5 * kT;
        cv_rot = 1.5 * k_b_ha;
        // q_rot = sqrt(pi) / σ * sqrt(kT³ / (A*B*C)) where A,B,C are rot const in Hz
        // Or equivalently: q_rot = sqrt(π·I_a·I_b·I_c) / σ * (8π²kT/h²)^(3/2)
        const sigma_f: f64 = @floatFromInt(sym_number);
        const a_hz = rot_const[0] * 1e9;
        const b_hz = rot_const[1] * 1e9;
        const c_hz = rot_const[2] * 1e9;
        const q_rot = @sqrt(pi) / sigma_f * @sqrt(
            std.math.pow(f64, kT_si / h_planck, 3.0) / (a_hz * b_hz * c_hz),
        );
        s_rot = k_b_ha * (1.5 + @log(q_rot));
    }

    // ==== Vibrational contribution (quantum harmonic oscillator) ====
    var zpve: f64 = 0.0;
    var e_vib_thermal: f64 = 0.0;
    var s_vib: f64 = 0.0;
    var cv_vib: f64 = 0.0;

    for (vib_result.vib_frequencies_cm1) |freq_cm1| {
        if (freq_cm1 <= 0.0) continue;

        // Convert to Hartree energy quantum
        const e_quantum = cm1ToHartree(freq_cm1);
        zpve += 0.5 * e_quantum;

        // u = hν / kT = e_quantum / kT
        const u = e_quantum / kT;

        // Vibrational partition function contribution
        // q_vib_i = exp(-u/2) / (1 - exp(-u))
        // E_vib = sum[ hν * (1/2 + 1/(exp(u)-1)) ]
        // S_vib = k * sum[ u/(exp(u)-1) - ln(1-exp(-u)) ]
        // Cv_vib = k * sum[ u² * exp(u) / (exp(u)-1)² ]

        if (u > 500.0) {
            // Very high frequency: only ZPE contributes, thermal part negligible
            e_vib_thermal += 0.0;
            continue;
        }

        const exp_u = @exp(u);
        const exp_neg_u = @exp(-u);
        const denom = exp_u - 1.0;

        // Thermal vibrational energy (above ZPE)
        e_vib_thermal += e_quantum / denom;

        // Entropy
        s_vib += k_b_ha * (u / denom - @log(1.0 - exp_neg_u));

        // Heat capacity
        cv_vib += k_b_ha * u * u * exp_u / (denom * denom);
    }

    const e_vib = zpve + e_vib_thermal;

    // ==== Totals ====
    const e_0k = e_elec + zpve;
    const e_tot = e_elec + e_trans + e_rot + e_vib;
    const s_tot = s_trans + s_rot + s_vib; // S_elec = 0 for closed-shell singlet
    const h_tot = e_tot + kT; // H = E + PV = E + nRT = E + kT (per molecule)
    const g_tot = h_tot - temperature * s_tot;

    return .{
        .temperature = temperature,
        .pressure = pressure,
        .e_elec = e_elec,
        .s_trans = s_trans,
        .e_trans = e_trans,
        .cv_trans = cv_trans,
        .rot_const_ghz = rot_const,
        .sym_number = sym_number,
        .is_linear = pm.is_linear,
        .s_rot = s_rot,
        .e_rot = e_rot,
        .cv_rot = cv_rot,
        .zpve = zpve,
        .s_vib = s_vib,
        .e_vib = e_vib,
        .cv_vib = cv_vib,
        .e_0k = e_0k,
        .e_tot = e_tot,
        .h_tot = h_tot,
        .s_tot = s_tot,
        .g_tot = g_tot,
    };
}

/// Print thermodynamic results.
pub fn printThermo(r: *const ThermoResult) void {
    logging.progress(
        true,
        "\n=== Thermodynamic Properties ({d:.2} K, {d:.0} Pa) ===\n",
        .{ r.temperature, r.pressure },
    );
    logging.progress(true, "\nElectronic:\n", .{});
    logging.progress(true, "  E_elec          = {d:.10} Ha\n", .{r.e_elec});
    logging.progress(true, "\nTranslational:\n", .{});
    logging.progress(true, "  E_trans          = {d:.10} Ha\n", .{r.e_trans});
    logging.progress(true, "  S_trans          = {e:15.6} Ha/K\n", .{r.s_trans});
    logging.progress(true, "  Cv_trans         = {e:15.6} Ha/K\n", .{r.cv_trans});
    logging.progress(true, "\nRotational:\n", .{});
    logging.progress(
        true,
        "  Rot constants    = {d:.3}, {d:.3}, {d:.3} GHz\n",
        .{ r.rot_const_ghz[0], r.rot_const_ghz[1], r.rot_const_ghz[2] },
    );
    logging.progress(true, "  Symmetry number  = {d}\n", .{r.sym_number});
    logging.progress(true, "  Linear           = {}\n", .{r.is_linear});
    logging.progress(true, "  E_rot            = {d:.10} Ha\n", .{r.e_rot});
    logging.progress(true, "  S_rot            = {e:15.6} Ha/K\n", .{r.s_rot});
    logging.progress(true, "  Cv_rot           = {e:15.6} Ha/K\n", .{r.cv_rot});
    logging.progress(true, "\nVibrational:\n", .{});
    logging.progress(
        true,
        "  ZPVE             = {d:.10} Ha ({d:.4} kcal/mol)\n",
        .{ r.zpve, r.zpve * ha_to_kcal },
    );
    logging.progress(true, "  E_vib            = {d:.10} Ha\n", .{r.e_vib});
    logging.progress(true, "  S_vib            = {e:15.6} Ha/K\n", .{r.s_vib});
    logging.progress(true, "  Cv_vib           = {e:15.6} Ha/K\n", .{r.cv_vib});
    logging.progress(true, "\nTotals:\n", .{});
    logging.progress(true, "  E(0K)            = {d:.10} Ha\n", .{r.e_0k});
    logging.progress(true, "  E_tot            = {d:.10} Ha\n", .{r.e_tot});
    logging.progress(true, "  H_tot            = {d:.10} Ha\n", .{r.h_tot});
    logging.progress(true, "  S_tot            = {e:15.6} Ha/K\n", .{r.s_tot});
    logging.progress(true, "  G_tot            = {d:.10} Ha\n", .{r.g_tot});
}

// ============================================================================
// Shell-to-atom map (reused from optimizer.zig)
// ============================================================================

fn buildShellToAtomMap(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const Vec3,
) ![]usize {
    const map = try alloc.alloc(usize, shells.len);
    for (shells, 0..) |shell, i| {
        var best_atom: usize = 0;
        var best_dist: f64 = std.math.inf(f64);
        for (nuc_positions, 0..) |pos, a| {
            const dx = shell.center.x - pos.x;
            const dy = shell.center.y - pos.y;
            const dz = shell.center.z - pos.z;
            const dist = dx * dx + dy * dy + dz * dz;
            if (dist < best_dist) {
                best_dist = dist;
                best_atom = a;
            }
        }
        map[i] = best_atom;
    }
    return map;
}

fn updateShellCenters(
    shells: []ContractedShell,
    shell_to_atom: []const usize,
    positions: []const Vec3,
) void {
    for (shells, 0..) |*shell, i| {
        shell.center = positions[shell_to_atom[i]];
    }
}

// ============================================================================
// Numerical Hessian
// ============================================================================

/// Compute the numerical Hessian matrix by central finite differences of
/// analytical gradients.
///
/// H[i,j] = (g_j(x+δe_i) - g_j(x-δe_i)) / (2δ)
///
/// where g_j is the j-th component of the gradient and e_i is the i-th
/// unit vector in 3N-dimensional coordinate space.
pub fn computeNumericalHessian(
    alloc: std.mem.Allocator,
    shells: []ContractedShell,
    nuc_positions: []Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    params: VibrationalParams,
) ![]f64 {
    const n_atoms = nuc_positions.len;
    const n3 = n_atoms * 3;
    const n_occ = n_electrons / 2;
    const delta = params.displacement;

    // Build shell-to-atom map from initial geometry
    const shell_to_atom = try buildShellToAtomMap(alloc, shells, nuc_positions);
    defer alloc.free(shell_to_atom);

    // Save original positions
    const orig_positions = try alloc.alloc(Vec3, n_atoms);
    defer alloc.free(orig_positions);
    @memcpy(orig_positions, nuc_positions);

    // Allocate Hessian
    const hessian = try alloc.alloc(f64, n3 * n3);
    @memset(hessian, 0.0);

    // For each coordinate i, displace +δ and -δ, compute gradient
    for (0..n3) |i| {
        const atom_i = i / 3;
        const coord_i = i % 3; // 0=x, 1=y, 2=z

        logging.progress(
            params.print_progress,
            "  Hessian: displacing coord {d}/{d} (atom {d}, {c})\n",
            .{
                i + 1, n3, atom_i,
                @as(u8, switch (coord_i) {
                    0 => 'x',
                    1 => 'y',
                    2 => 'z',
                    else => '?',
                }),
            },
        );

        // +δ displacement
        @memcpy(nuc_positions, orig_positions);
        switch (coord_i) {
            0 => nuc_positions[atom_i].x += delta,
            1 => nuc_positions[atom_i].y += delta,
            2 => nuc_positions[atom_i].z += delta,
            else => {},
        }
        updateShellCenters(shells, shell_to_atom, nuc_positions);

        const grad_plus = try computeGradientFlat(
            alloc,
            shells,
            nuc_positions,
            nuc_charges,
            n_electrons,
            n_occ,
            params.scf_params,
        );
        defer alloc.free(grad_plus);

        // -δ displacement
        @memcpy(nuc_positions, orig_positions);
        switch (coord_i) {
            0 => nuc_positions[atom_i].x -= delta,
            1 => nuc_positions[atom_i].y -= delta,
            2 => nuc_positions[atom_i].z -= delta,
            else => {},
        }
        updateShellCenters(shells, shell_to_atom, nuc_positions);

        const grad_minus = try computeGradientFlat(
            alloc,
            shells,
            nuc_positions,
            nuc_charges,
            n_electrons,
            n_occ,
            params.scf_params,
        );
        defer alloc.free(grad_minus);

        // H[i, j] = (g_j(+) - g_j(-)) / (2δ)
        for (0..n3) |j| {
            hessian[i * n3 + j] = (grad_plus[j] - grad_minus[j]) / (2.0 * delta);
        }
    }

    // Restore original positions
    @memcpy(nuc_positions, orig_positions);
    updateShellCenters(shells, shell_to_atom, nuc_positions);

    // Symmetrize: H = (H + H^T) / 2
    for (0..n3) |i| {
        for (i + 1..n3) |j| {
            const avg = (hessian[i * n3 + j] + hessian[j * n3 + i]) / 2.0;
            hessian[i * n3 + j] = avg;
            hessian[j * n3 + i] = avg;
        }
    }

    return hessian;
}

/// Helper: run SCF + gradient and return flattened gradient array.
fn computeGradientFlat(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    n_occ: usize,
    scf_params: ScfParams,
) ![]f64 {
    const n_atoms = nuc_positions.len;
    const n3 = n_atoms * 3;

    var scf_result = try gto_scf.runGeneralRhfScf(
        alloc,
        shells,
        nuc_positions,
        nuc_charges,
        n_electrons,
        scf_params,
    );
    defer scf_result.deinit(alloc);

    var grad_result = try gradient_mod.computeRhfGradient(
        alloc,
        shells,
        nuc_positions,
        nuc_charges,
        scf_result.density_matrix,
        scf_result.orbital_energies,
        scf_result.mo_coefficients,
        n_occ,
    );
    defer grad_result.deinit(alloc);

    const flat = try alloc.alloc(f64, n3);
    for (0..n_atoms) |i| {
        flat[i * 3 + 0] = grad_result.gradients[i].x;
        flat[i * 3 + 1] = grad_result.gradients[i].y;
        flat[i * 3 + 2] = grad_result.gradients[i].z;
    }
    return flat;
}

// ============================================================================
// Diagonalization
// ============================================================================

/// Diagonalize a real symmetric matrix using LAPACK dsyev.
/// Returns eigenvalues (ascending) and eigenvectors (columns of the output matrix).
/// The input matrix is overwritten with eigenvectors.
fn diagonalizeSymmetric(alloc: std.mem.Allocator, matrix: []f64, n: usize) ![]f64 {
    lapack_mutex.lock();
    defer lapack_mutex.unlock();

    const eigenvalues = try alloc.alloc(f64, n);
    errdefer alloc.free(eigenvalues);

    var nn: c_int = @intCast(n);
    var lda: c_int = @intCast(n);
    var jobz: [1]u8 = .{'V'};
    var uplo: [1]u8 = .{'U'};
    var info: c_int = 0;

    // Workspace query
    var lwork: c_int = -1;
    var work_query: f64 = 0.0;
    dsyev_(
        jobz[0..].ptr,
        uplo[0..].ptr,
        &nn,
        matrix.ptr,
        &lda,
        eigenvalues.ptr,
        @ptrCast(&work_query),
        &lwork,
        &info,
    );
    if (info != 0) return error.LapackFailure;

    lwork = @intFromFloat(work_query);
    if (lwork < 1) lwork = 1;
    const work = try alloc.alloc(f64, @intCast(lwork));
    defer alloc.free(work);

    info = 0;
    dsyev_(
        jobz[0..].ptr,
        uplo[0..].ptr,
        &nn,
        matrix.ptr,
        &lda,
        eigenvalues.ptr,
        work.ptr,
        &lwork,
        &info,
    );
    if (info != 0) return error.LapackFailure;

    return eigenvalues;
}

// ============================================================================
// Full vibrational analysis
// ============================================================================

/// Perform a full vibrational analysis at the given geometry.
///
/// Steps:
///   1. Compute numerical Hessian (central finite differences of analytical gradients).
///   2. Mass-weight the Hessian.
///   3. Diagonalize to get eigenvalues and normal modes.
///   4. Convert eigenvalues to frequencies in cm⁻¹.
///   5. Identify and separate translational/rotational modes.
///   6. Compute ZPVE.
pub fn vibrationalAnalysis(
    alloc: std.mem.Allocator,
    shells: []ContractedShell,
    nuc_positions: []Vec3,
    nuc_charges: []const f64,
    atomic_numbers: []const u8,
    n_electrons: usize,
    params: VibrationalParams,
) !VibrationalResult {
    const n_atoms = nuc_positions.len;
    const n3 = n_atoms * 3;

    logging.progress(
        params.print_progress,
        "\nVibrational Analysis ({d} atoms, {d} coordinates)\n",
        .{ n_atoms, n3 },
    );

    // Step 1: Compute numerical Hessian
    const hessian = try computeNumericalHessian(
        alloc,
        shells,
        nuc_positions,
        nuc_charges,
        n_electrons,
        params,
    );

    // Step 2: Mass-weight the Hessian
    // H_mw[3i+a, 3j+b] = H[3i+a, 3j+b] / sqrt(m_i * m_j)
    const mw_hessian = try alloc.alloc(f64, n3 * n3);
    defer alloc.free(mw_hessian);
    @memcpy(mw_hessian, hessian);

    for (0..n_atoms) |i| {
        const mi = if (params.use_integer_masses)
            atomicMassInt(atomic_numbers[i])
        else
            atomicMass(atomic_numbers[i]);
        for (0..n_atoms) |j| {
            const mj = if (params.use_integer_masses)
                atomicMassInt(atomic_numbers[j])
            else
                atomicMass(atomic_numbers[j]);
            const inv_sqrt_mm = 1.0 / @sqrt(mi * mj);
            for (0..3) |a| {
                for (0..3) |b| {
                    mw_hessian[(3 * i + a) * n3 + (3 * j + b)] *= inv_sqrt_mm;
                }
            }
        }
    }

    // Step 3: Diagonalize mass-weighted Hessian
    // dsyev expects column-major but our matrix is symmetric, so row-major = column-major
    const eigenvalues = try diagonalizeSymmetric(alloc, mw_hessian, n3);
    // After diagonalization, mw_hessian contains eigenvectors (column-major)
    const normal_modes = try alloc.alloc(f64, n3 * n3);
    @memcpy(normal_modes, mw_hessian);

    // Step 4: Convert eigenvalues to frequencies
    const frequencies = try alloc.alloc(f64, n3);
    for (0..n3) |i| {
        frequencies[i] = eigenvalueToCm1(eigenvalues[i]);
    }

    if (params.print_progress) {
        logging.progress(true, "\nAll frequencies (cm^-1):\n", .{});
        for (0..n3) |i| {
            logging.progress(true, "  mode {d}: {d:.2} cm^-1\n", .{ i + 1, frequencies[i] });
        }
    }

    // Step 5: Separate vibrational modes from translations/rotations
    // Translations and rotations have near-zero frequencies
    var n_vib: usize = 0;
    for (0..n3) |i| {
        if (@abs(frequencies[i]) > params.rot_trans_threshold) {
            n_vib += 1;
        }
    }

    const vib_freqs = try alloc.alloc(f64, n_vib);
    var vi: usize = 0;
    for (0..n3) |i| {
        if (@abs(frequencies[i]) > params.rot_trans_threshold) {
            vib_freqs[vi] = frequencies[i];
            vi += 1;
        }
    }

    // Step 6: Compute ZPVE = (1/2) * sum of positive vibrational frequencies (in Ha)
    var zpve: f64 = 0.0;
    for (vib_freqs) |freq| {
        if (freq > 0.0) {
            zpve += cm1ToHartree(freq);
        }
    }
    zpve *= 0.5;

    if (params.print_progress) {
        logging.progress(true, "\nVibrational frequencies (cm^-1):\n", .{});
        for (0..n_vib) |i| {
            logging.progress(true, "  {d:.2}\n", .{vib_freqs[i]});
        }
        logging.progress(true, "\nZPVE: {d:.10} Ha ({d:.4} kcal/mol)\n", .{ zpve, zpve * 627.509 });
    }

    return .{
        .eigenvalues = eigenvalues,
        .frequencies_cm1 = frequencies,
        .vib_frequencies_cm1 = vib_freqs,
        .normal_modes = normal_modes,
        .hessian = hessian,
        .zpve = zpve,
        .n_atoms = n_atoms,
    };
}

// Slow vibrational and thermodynamic regression coverage lives in `regression_tests.zig`.
