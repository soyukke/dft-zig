//! Dynamical matrix construction, acoustic sum rule, and diagonalization.
//!
//! Constructs the full dynamical matrix from electronic and ionic contributions,
//! applies the acoustic sum rule for Γ-point, diagonalizes, and converts to cm⁻¹.

const std = @import("std");
const math = @import("../math/math.zig");
const symmetry_mod = @import("../symmetry/symmetry.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const linalg = @import("../linalg/linalg.zig");

pub const ComplexPhononResult = struct {
    /// Eigenvalues ω² in Ry/(bohr²·amu)
    omega2: []f64,
    /// Frequencies in cm⁻¹
    frequencies_cm1: []f64,
    /// Eigenvectors (column-major, dim×dim, complex)
    eigenvectors: []math.Complex,
    dim: usize,

    pub fn deinit(self: *ComplexPhononResult, alloc: std.mem.Allocator) void {
        if (self.omega2.len > 0) alloc.free(self.omega2);
        if (self.frequencies_cm1.len > 0) alloc.free(self.frequencies_cm1);
        if (self.eigenvectors.len > 0) alloc.free(self.eigenvectors);
    }
};

pub const PhononResult = struct {
    /// Eigenvalues ω² in Ry/(bohr²·amu)
    omega2: []f64,
    /// Frequencies in cm⁻¹
    frequencies_cm1: []f64,
    /// Eigenvectors (column-major, dim×dim)
    eigenvectors: []f64,
    dim: usize,

    pub fn deinit(self: *PhononResult, alloc: std.mem.Allocator) void {
        if (self.omega2.len > 0) alloc.free(self.omega2);
        if (self.frequencies_cm1.len > 0) alloc.free(self.frequencies_cm1);
        if (self.eigenvectors.len > 0) alloc.free(self.eigenvectors);
    }
};

/// Apply acoustic sum rule at Γ-point.
/// For each atom I and directions α,β:
///   D_{Iα,Iβ} = -Σ_{J≠I} D_{Iα,Jβ}
/// This ensures translational invariance: Σ_J D_{Iα,Jβ} = 0.
pub fn applyASR(dynmat: []f64, n_atoms: usize) void {
    const dim = 3 * n_atoms;
    for (0..n_atoms) |i| {
        for (0..3) |alpha| {
            for (0..3) |beta| {
                var sum: f64 = 0.0;
                for (0..n_atoms) |j| {
                    if (j == i) continue;
                    sum += dynmat[(3 * i + alpha) * dim + (3 * j + beta)];
                }
                dynmat[(3 * i + alpha) * dim + (3 * i + beta)] = -sum;
            }
        }
    }
}

/// Divide by masses to get mass-weighted dynamical matrix.
/// D̃_{Iα,Jβ} = D_{Iα,Jβ} / √(M_I × M_J)
/// Masses in AMU.
pub fn massWeight(dynmat: []f64, n_atoms: usize, masses: []const f64) void {
    const dim = 3 * n_atoms;
    for (0..n_atoms) |i| {
        for (0..n_atoms) |j| {
            const inv_sqrt_mm = 1.0 / std.math.sqrt(masses[i] * masses[j]);
            for (0..3) |alpha| {
                for (0..3) |beta| {
                    dynmat[(3 * i + alpha) * dim + (3 * j + beta)] *= inv_sqrt_mm;
                }
            }
        }
    }
}

/// Convert eigenvalue ω² (Ry/(bohr²·amu)) to frequency in cm⁻¹.
/// ω [rad/s] = √(ω² × Ry_to_J / (bohr_to_m² × amu_to_kg))
/// ν [cm⁻¹] = ω / (2π × c)
///
/// Units: 1 Ry = 2.1798723611e-18 J, 1 bohr = 5.29177249e-11 m, 1 amu = 1.6605402e-27 kg
/// c = 2.99792458e10 cm/s
pub fn omega2ToCm1(omega2: f64) f64 {
    const ry_to_j = 2.1798723611e-18;
    const bohr_to_m = 5.29177249e-11;
    const amu_to_kg = 1.6605402e-27;
    const c_cm = 2.99792458e10; // cm/s

    // conversion factor: ω² [Ry/(bohr²·amu)] → ω² [1/s²]
    const conv = ry_to_j / (bohr_to_m * bohr_to_m * amu_to_kg);

    if (omega2 >= 0.0) {
        const omega_si = std.math.sqrt(omega2 * conv);
        return omega_si / (2.0 * std.math.pi * c_cm);
    } else {
        // Imaginary frequency (instability)
        const omega_si = std.math.sqrt(-omega2 * conv);
        return -omega_si / (2.0 * std.math.pi * c_cm);
    }
}

/// Diagonalize a real symmetric dynamical matrix using LAPACK dsyev.
pub fn diagonalize(alloc: std.mem.Allocator, dynmat: []const f64, dim: usize) !PhononResult {
    if (dynmat.len != dim * dim) return error.InvalidMatrixSize;
    var eig = try linalg.realSymmetricEigenDecomp(alloc, .accelerate, dim, @constCast(dynmat));
    errdefer eig.deinit(alloc);

    // Convert to cm⁻¹
    const freq = try alloc.alloc(f64, dim);
    errdefer alloc.free(freq);
    for (0..dim) |i| {
        freq[i] = omega2ToCm1(eig.values[i]);
    }

    return .{
        .omega2 = eig.values,
        .frequencies_cm1 = freq,
        .eigenvectors = eig.vectors,
        .dim = dim,
    };
}

/// Divide by masses to get mass-weighted dynamical matrix (complex version).
/// D̃_{Iα,Jβ} = D_{Iα,Jβ} / √(M_I × M_J)
pub fn massWeightComplex(dynmat_c: []math.Complex, n_atoms: usize, masses: []const f64) void {
    const dim = 3 * n_atoms;
    for (0..n_atoms) |i| {
        for (0..n_atoms) |j| {
            const inv_sqrt_mm = 1.0 / std.math.sqrt(masses[i] * masses[j]);
            for (0..3) |alpha| {
                for (0..3) |beta| {
                    dynmat_c[(3 * i + alpha) * dim + (3 * j + beta)] = math.complex.scale(
                        dynmat_c[(3 * i + alpha) * dim + (3 * j + beta)],
                        inv_sqrt_mm,
                    );
                }
            }
        }
    }
}

/// Diagonalize a complex Hermitian dynamical matrix using LAPACK zheev.
pub fn diagonalizeComplex(alloc: std.mem.Allocator, dynmat_c: []const math.Complex, dim: usize) !ComplexPhononResult {
    if (dynmat_c.len != dim * dim) return error.InvalidMatrixSize;
    const dynmat_copy = try alloc.alloc(math.Complex, dynmat_c.len);
    defer alloc.free(dynmat_copy);
    @memcpy(dynmat_copy, dynmat_c);
    var eig = try linalg.hermitianEigenDecomp(alloc, .accelerate, dim, dynmat_copy);
    errdefer eig.deinit(alloc);

    // Convert to cm⁻¹
    const freq = try alloc.alloc(f64, dim);
    errdefer alloc.free(freq);
    for (0..dim) |i| {
        freq[i] = omega2ToCm1(eig.values[i]);
    }

    return .{
        .omega2 = eig.values,
        .frequencies_cm1 = freq,
        .eigenvectors = eig.vectors,
        .dim = dim,
    };
}

/// Compute only eigenvalues (frequencies) of a complex Hermitian dynamical matrix.
/// Uses zheev with jobz='N' (no eigenvectors), ~2-3x faster than full diagonalization.
pub fn eigenvaluesComplex(alloc: std.mem.Allocator, dynmat_c: []math.Complex, dim: usize) ![]f64 {
    if (dynmat_c.len != dim * dim) return error.InvalidMatrixSize;
    const dynmat_copy = try alloc.alloc(math.Complex, dynmat_c.len);
    defer alloc.free(dynmat_copy);
    @memcpy(dynmat_copy, dynmat_c);
    const omega2 = try linalg.hermitianEigenvalues(alloc, .accelerate, dim, dynmat_copy);
    errdefer alloc.free(omega2);

    // Convert to cm⁻¹ in-place (reuse eigenvalue array)
    for (0..dim) |i| {
        omega2[i] = omega2ToCm1(omega2[i]);
    }

    return omega2;
}

// =========================================================================
// Dynamical matrix symmetrization
// =========================================================================

/// Build atom mapping table for symmetry operations.
/// For each symmetry operation S and atom i, indsym[isym][iatom] = j
/// where atom j is the image of atom i under S (rot + trans).
/// Also returns the translation vectors needed to bring the image back to
/// the fundamental unit cell (for phase factors at q≠0).
pub fn buildIndsym(
    alloc: std.mem.Allocator,
    symops: []const symmetry_mod.SymOp,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    tol: f64,
) !struct {
    indsym: [][]usize,
    /// tnons_shift[isym][iatom] = fractional translation to bring S(atom_i) back to unit cell
    tnons_shift: [][]math.Vec3,
} {
    const nsym = symops.len;
    const natom = atoms.len;

    // Convert atom positions to fractional coordinates
    const two_pi = 2.0 * std.math.pi;
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);

    const atom_fracs = try alloc.alloc(math.Vec3, natom);
    defer alloc.free(atom_fracs);
    for (atoms, 0..) |atom, i| {
        const pos = atom.position;
        atom_fracs[i] = .{
            .x = math.Vec3.dot(b1, pos) / two_pi,
            .y = math.Vec3.dot(b2, pos) / two_pi,
            .z = math.Vec3.dot(b3, pos) / two_pi,
        };
    }

    const indsym = try alloc.alloc([]usize, nsym);
    errdefer {
        for (indsym[0..nsym]) |row| alloc.free(row);
        alloc.free(indsym);
    }
    const tnons_shift = try alloc.alloc([]math.Vec3, nsym);
    errdefer {
        for (tnons_shift[0..nsym]) |row| alloc.free(row);
        alloc.free(tnons_shift);
    }

    for (0..nsym) |isym| {
        indsym[isym] = try alloc.alloc(usize, natom);
        tnons_shift[isym] = try alloc.alloc(math.Vec3, natom);
        const rot = symops[isym].rot;
        const trans = symops[isym].trans;

        for (0..natom) |iatom| {
            // Apply rotation + translation in fractional coords
            const rotated = rot.mulVec(atom_fracs[iatom]);
            const image = math.Vec3{
                .x = rotated.x + trans.x,
                .y = rotated.y + trans.y,
                .z = rotated.z + trans.z,
            };

            // Find matching atom in the unit cell
            var found = false;
            for (0..natom) |jatom| {
                if (atoms[jatom].species_index != atoms[iatom].species_index) continue;
                // Check if image ≡ atom_j (mod lattice)
                const diff = math.Vec3{
                    .x = image.x - atom_fracs[jatom].x,
                    .y = image.y - atom_fracs[jatom].y,
                    .z = image.z - atom_fracs[jatom].z,
                };
                const shift = math.Vec3{
                    .x = std.math.round(diff.x),
                    .y = std.math.round(diff.y),
                    .z = std.math.round(diff.z),
                };
                const residual = math.Vec3{
                    .x = diff.x - shift.x,
                    .y = diff.y - shift.y,
                    .z = diff.z - shift.z,
                };
                if (@abs(residual.x) < tol and @abs(residual.y) < tol and @abs(residual.z) < tol) {
                    indsym[isym][iatom] = jatom;
                    // Store the lattice shift (image = atom_j + shift)
                    tnons_shift[isym][iatom] = shift;
                    found = true;
                    break;
                }
            }
            if (!found) {
                // This should not happen if symmetry operations are valid
                @panic("buildIndsym: atom mapping failed");
            }
        }
    }

    return .{ .indsym = indsym, .tnons_shift = tnons_shift };
}

/// Convert a fractional-coordinate rotation matrix to Cartesian coordinates.
/// R_cart = L R_frac L^{-1} where L is the lattice matrix (columns = lattice vectors).
/// In our convention, cell.m[i][j] = a_i component j, so L = cell^T.
pub fn fracRotToCart(rot: symmetry_mod.Mat3i, cell: math.Mat3) [3][3]f64 {
    // L = cell^T (columns are lattice vectors a1, a2, a3)
    // L^{-1} can be computed via recip: L^{-1} = (cell^T)^{-1}
    // But it's easier to use: recip^T / (2π) = L^{-T}^{-1}... let's just do the multiplication directly.

    // Method: R_cart[i][j] = sum_{a,b} L[i][a] * R_frac[a][b] * L_inv[b][j]
    // where L[i][a] = cell.m[a][i] (L = cell^T)

    // First compute L * R_frac
    var lr: [3][3]f64 = undefined;
    for (0..3) |i| {
        for (0..3) |b| {
            var sum: f64 = 0.0;
            for (0..3) |a| {
                sum += cell.m[a][i] * @as(f64, @floatFromInt(rot.m[a][b]));
            }
            lr[i][b] = sum;
        }
    }

    // Compute L_inv = (cell^T)^{-1}
    // Use the formula: (cell^T)^{-1} = recip^T / (2π)
    // But we need recip here. Instead, compute L_inv from cell directly.
    // For a 3x3 matrix, L_inv = adj(L)^T / det(L)
    // Actually, it's simpler: cell^{-1} exists because det ≠ 0.
    // L = cell^T, so L^{-1} = (cell^{-1})^T

    // Compute cell inverse using cofactor method
    const c = cell.m;
    const det_val = c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) -
        c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0]) +
        c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
    const inv_det = 1.0 / det_val;

    var cell_inv: [3][3]f64 = undefined;
    cell_inv[0][0] = (c[1][1] * c[2][2] - c[1][2] * c[2][1]) * inv_det;
    cell_inv[0][1] = (c[0][2] * c[2][1] - c[0][1] * c[2][2]) * inv_det;
    cell_inv[0][2] = (c[0][1] * c[1][2] - c[0][2] * c[1][1]) * inv_det;
    cell_inv[1][0] = (c[1][2] * c[2][0] - c[1][0] * c[2][2]) * inv_det;
    cell_inv[1][1] = (c[0][0] * c[2][2] - c[0][2] * c[2][0]) * inv_det;
    cell_inv[1][2] = (c[0][2] * c[1][0] - c[0][0] * c[1][2]) * inv_det;
    cell_inv[2][0] = (c[1][0] * c[2][1] - c[1][1] * c[2][0]) * inv_det;
    cell_inv[2][1] = (c[0][1] * c[2][0] - c[0][0] * c[2][1]) * inv_det;
    cell_inv[2][2] = (c[0][0] * c[1][1] - c[0][1] * c[1][0]) * inv_det;

    // L_inv = (cell_inv)^T
    // R_cart = (L * R_frac) * L_inv = lr * (cell_inv)^T
    var r_cart: [3][3]f64 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            var sum: f64 = 0.0;
            for (0..3) |b| {
                sum += lr[i][b] * cell_inv[j][b]; // (cell_inv)^T[b][j] = cell_inv[j][b]
            }
            r_cart[i][j] = sum;
        }
    }

    return r_cart;
}

/// Symmetrize a complex dynamical matrix using space group symmetry operations.
///
/// Formula (Cartesian coordinates):
///   D_sym(α,I; β,J) = (1/N_sym) Σ_S R_cart(α,γ) R_cart(β,δ) D(γ,S(I); δ,S(J))
///                       × exp(-i q·(shift_I - shift_J))
///
/// where:
///   - R_cart is the Cartesian rotation matrix for symmetry operation S
///   - S(I) is the atom index that atom I maps to under S
///   - shift_I is the lattice translation to bring S(I) back to the unit cell
///   - q is in Cartesian coordinates
///
/// The symmetrized matrix preserves the crystal symmetry and ensures
/// D(α,I; β,J) = D*(β,J; α,I) (Hermiticity).
pub fn symmetrizeDynmatComplex(
    alloc: std.mem.Allocator,
    dynmat: []math.Complex,
    n_atoms: usize,
    symops: []const symmetry_mod.SymOp,
    indsym: []const []const usize,
    tnons_shift: []const []const math.Vec3,
    cell_bohr: math.Mat3,
    q_frac: math.Vec3,
) !void {
    const dim = 3 * n_atoms;
    const nsym = symops.len;
    if (nsym <= 1) return; // Nothing to symmetrize

    const two_pi = 2.0 * std.math.pi;

    // Accumulate symmetrized matrix (dynamically allocated for large systems)
    const dyn_sym = try alloc.alloc(math.Complex, dim * dim);
    defer alloc.free(dyn_sym);
    @memset(dyn_sym, math.complex.init(0.0, 0.0));

    var nsym_used: usize = 0;

    for (0..nsym) |isym| {
        // Check if this symmetry operation belongs to the little group of q:
        // S^T q ≡ q (mod G), i.e., (S^T q - q) must be integer
        const rot = symops[isym].rot;
        const sq = symmetry_mod.Mat3i.transpose(rot).mulVec(q_frac);
        const dq_x = sq.x - q_frac.x;
        const dq_y = sq.y - q_frac.y;
        const dq_z = sq.z - q_frac.z;
        const tol_q: f64 = 1e-8;
        if (@abs(dq_x - std.math.round(dq_x)) > tol_q or
            @abs(dq_y - std.math.round(dq_y)) > tol_q or
            @abs(dq_z - std.math.round(dq_z)) > tol_q)
        {
            continue; // Not in little group of q, skip
        }

        // Compute Cartesian rotation matrix for this symmetry operation
        const r_cart = fracRotToCart(symops[isym].rot, cell_bohr);

        for (0..n_atoms) |iatom| {
            for (0..n_atoms) |jatom| {
                // Find images: S(iatom) and S(jatom)
                const si = indsym[isym][iatom];
                const sj = indsym[isym][jatom];

                // Phase factor: exp(-i 2π q·(shift_i - shift_j))
                const shift_i = tnons_shift[isym][iatom];
                const shift_j = tnons_shift[isym][jatom];
                const dshift = math.Vec3{
                    .x = shift_i.x - shift_j.x,
                    .y = shift_i.y - shift_j.y,
                    .z = shift_i.z - shift_j.z,
                };
                // q is in fractional coordinates, shift is in fractional coordinates
                // q·shift = q_frac · shift (dot product in fractional space)
                const arg = -two_pi * (q_frac.x * dshift.x + q_frac.y * dshift.y + q_frac.z * dshift.z);
                const phase = math.complex.init(@cos(arg), @sin(arg));

                // D_sym(α,I; β,J) += R_cart[α][γ] × R_cart[β][δ] × D(γ,S(I); δ,S(J)) × phase
                for (0..3) |alpha| {
                    for (0..3) |beta| {
                        var acc = math.complex.init(0.0, 0.0);
                        for (0..3) |gamma_idx| {
                            for (0..3) |delta| {
                                const d_elem = dynmat[(3 * si + gamma_idx) * dim + (3 * sj + delta)];
                                const coeff = r_cart[alpha][gamma_idx] * r_cart[beta][delta];
                                acc = math.complex.add(acc, math.complex.scale(d_elem, coeff));
                            }
                        }
                        // Multiply by phase
                        acc = math.complex.mul(acc, phase);
                        const idx = (3 * iatom + alpha) * dim + (3 * jatom + beta);
                        dyn_sym[idx] = math.complex.add(dyn_sym[idx], acc);
                    }
                }
            }
        }
        nsym_used += 1;
    }

    // Log little group size for debugging
    std.debug.print("symmetrizeDynmat: nsym_total={d} nsym_used(little_group)={d} q_frac=({d:.4},{d:.4},{d:.4})\n", .{ nsym, nsym_used, q_frac.x, q_frac.y, q_frac.z });

    // Average over symmetry operations
    if (nsym_used > 0) {
        const inv_nsym = 1.0 / @as(f64, @floatFromInt(nsym_used));
        for (0..dim * dim) |i| {
            dynmat[i] = math.complex.scale(dyn_sym[i], inv_nsym);
        }
    }
}

/// Apply acoustic sum rule to a complex dynamical matrix at Γ-point.
/// For each atom I and directions α,β:
///   D_{Iα,Iβ} = -Σ_{J≠I} D_{Iα,Jβ}
pub fn applyASRComplex(dynmat: []math.Complex, n_atoms: usize) void {
    const dim = 3 * n_atoms;
    for (0..n_atoms) |i| {
        for (0..3) |alpha| {
            for (0..3) |beta| {
                var sum = math.complex.init(0.0, 0.0);
                for (0..n_atoms) |j| {
                    if (j == i) continue;
                    sum = math.complex.add(sum, dynmat[(3 * i + alpha) * dim + (3 * j + beta)]);
                }
                dynmat[(3 * i + alpha) * dim + (3 * i + beta)] = math.complex.scale(sum, -1.0);
            }
        }
    }
}

// =========================================================================
// Irreducible atom reduction for DFPT symmetry
// =========================================================================

pub const IrreducibleAtomInfo = struct {
    irr_atom_indices: []usize,
    n_irr_atoms: usize,
    atom_to_irr: []usize,
    atom_sym_idx: []usize,
    is_irreducible: []bool,

    pub fn deinit(self: *IrreducibleAtomInfo, alloc: std.mem.Allocator) void {
        alloc.free(self.irr_atom_indices);
        alloc.free(self.atom_to_irr);
        alloc.free(self.atom_sym_idx);
        alloc.free(self.is_irreducible);
    }
};

/// Find indices of symmetry operations in the little group of q.
/// The little group consists of operations S where S^T q ≡ q (mod G).
fn findLittleGroupIndices(
    alloc: std.mem.Allocator,
    symops: []const symmetry_mod.SymOp,
    q_frac: math.Vec3,
) !std.ArrayList(usize) {
    const tol_q: f64 = 1e-8;
    var little_group: std.ArrayList(usize) = .empty;
    errdefer little_group.deinit(alloc);

    for (symops, 0..) |sym, isym| {
        const rot = sym.rot;
        const sq = symmetry_mod.Mat3i.transpose(rot).mulVec(q_frac);
        const dq_x = sq.x - q_frac.x;
        const dq_y = sq.y - q_frac.y;
        const dq_z = sq.z - q_frac.z;
        if (@abs(dq_x - std.math.round(dq_x)) <= tol_q and
            @abs(dq_y - std.math.round(dq_y)) <= tol_q and
            @abs(dq_z - std.math.round(dq_z)) <= tol_q)
        {
            try little_group.append(alloc, isym);
        }
    }
    return little_group;
}

/// Find irreducible atoms under the little group of q.
/// The little group consists of symmetry operations S where S^T q ≡ q (mod G).
/// Atoms related by little group operations share an orbit; the smallest index
/// in each orbit is the representative (irreducible) atom.
pub fn findIrreducibleAtoms(
    alloc: std.mem.Allocator,
    symops: []const symmetry_mod.SymOp,
    indsym: []const []const usize,
    n_atoms: usize,
    q_frac: math.Vec3,
) !IrreducibleAtomInfo {
    var little_group = try findLittleGroupIndices(alloc, symops, q_frac);
    defer little_group.deinit(alloc);

    // For each atom, find the representative (smallest index in orbit under little group)
    const atom_to_irr = try alloc.alloc(usize, n_atoms);
    errdefer alloc.free(atom_to_irr);
    const atom_sym_idx = try alloc.alloc(usize, n_atoms);
    errdefer alloc.free(atom_sym_idx);
    const is_irreducible = try alloc.alloc(bool, n_atoms);
    errdefer alloc.free(is_irreducible);

    for (0..n_atoms) |ia| {
        var rep = ia;
        var rep_sym: usize = 0; // identity (first in little group if present)
        for (little_group.items) |isym| {
            const mapped = indsym[isym][ia];
            if (mapped < rep) {
                rep = mapped;
                rep_sym = isym;
            }
        }
        atom_to_irr[ia] = rep;
        atom_sym_idx[ia] = rep_sym;
        is_irreducible[ia] = (rep == ia);
    }

    // Collect irreducible atom indices
    var irr_list: std.ArrayList(usize) = .empty;
    errdefer irr_list.deinit(alloc);
    for (0..n_atoms) |ia| {
        if (is_irreducible[ia]) {
            try irr_list.append(alloc, ia);
        }
    }
    const n_irr = irr_list.items.len;
    const irr_indices = try irr_list.toOwnedSlice(alloc);

    return .{
        .irr_atom_indices = irr_indices,
        .n_irr_atoms = n_irr,
        .atom_to_irr = atom_to_irr,
        .atom_sym_idx = atom_sym_idx,
        .is_irreducible = is_irreducible,
    };
}

/// Information about irreducible perturbations (atom, direction) under the little group of q.
/// Extends IrreducibleAtomInfo to also reduce directions using Cartesian rotation matrices.
pub const IrreduciblePertInfo = struct {
    /// Indices of irreducible perturbations: pidx = 3*atom + dir
    irr_pert_indices: []usize,
    n_irr_perts: usize,
    /// Maps each perturbation (3*natom) to its irreducible representative
    pert_to_irr: []usize,
    /// Symmetry operation index that maps perturbation to representative
    pert_sym_idx: []usize,
    /// Direction mapping: the direction component in the representative frame
    /// pert_dir_coeffs[pidx] = R_cart row that maps representative direction to this one
    pert_dir_coeffs: [][3]f64,
    is_irreducible: []bool,

    pub fn deinit(self: *IrreduciblePertInfo, alloc: std.mem.Allocator) void {
        alloc.free(self.irr_pert_indices);
        alloc.free(self.pert_to_irr);
        alloc.free(self.pert_sym_idx);
        alloc.free(self.pert_dir_coeffs);
        alloc.free(self.is_irreducible);
    }
};

/// Find irreducible perturbations (atom, direction) under the little group of q.
/// A perturbation (I, α) is mapped by symmetry S to (S(I), Σ_γ R_cart[α][γ] × e_γ).
/// Two perturbations are equivalent if they lie in the same orbit under these mappings.
pub fn findIrreduciblePerturbations(
    alloc: std.mem.Allocator,
    symops: []const symmetry_mod.SymOp,
    indsym: []const []const usize,
    n_atoms: usize,
    q_frac: math.Vec3,
    cell_bohr: math.Mat3,
) !IrreduciblePertInfo {
    const dim = 3 * n_atoms;

    var little_group = try findLittleGroupIndices(alloc, symops, q_frac);
    defer little_group.deinit(alloc);

    const pert_to_irr = try alloc.alloc(usize, dim);
    errdefer alloc.free(pert_to_irr);
    const pert_sym_idx = try alloc.alloc(usize, dim);
    errdefer alloc.free(pert_sym_idx);
    const pert_dir_coeffs = try alloc.alloc([3]f64, dim);
    errdefer alloc.free(pert_dir_coeffs);
    const is_irreducible = try alloc.alloc(bool, dim);
    errdefer alloc.free(is_irreducible);

    // Initialize: each perturbation maps to itself
    for (0..dim) |p| {
        pert_to_irr[p] = p;
        pert_sym_idx[p] = 0; // identity
        pert_dir_coeffs[p] = .{ 0.0, 0.0, 0.0 };
        pert_dir_coeffs[p][p % 3] = 1.0; // identity mapping
        is_irreducible[p] = true;
    }

    // For each perturbation, check if a smaller-index one is equivalent
    for (0..dim) |pidx| {
        const ia = pidx / 3;
        const dir = pidx % 3;

        for (little_group.items) |isym| {
            const mapped_atom = indsym[isym][ia];
            const r_cart = fracRotToCart(symops[isym].rot, cell_bohr);

            // S maps (ia, dir) to (mapped_atom, Σ_γ R_cart[·][dir] × e_γ)
            // The mapped direction is a linear combination: row = R_cart[·][dir]
            // But for orbit finding, the key insight is:
            // (ia, dir) -> (mapped_atom, R_cart × e_dir) = Σ_γ R_cart[γ][dir] on mapped_atom
            //
            // Check: does this map to a perturbation with smaller index?
            // The mapped perturbation lives on mapped_atom, and its direction
            // is the column R_cart[·][dir]. For cubic symmetry, this column
            // is often a pure unit vector (mapping x->y etc.)

            // Find if R_cart maps e_dir to a pure direction e_mapped_dir
            // i.e., R_cart[mapped_dir][dir] = ±1 and all others 0
            var pure_dir: ?usize = null;
            for (0..3) |d| {
                if (@abs(@abs(r_cart[d][dir]) - 1.0) < 1e-10) {
                    // Check other components are zero
                    var others_zero = true;
                    for (0..3) |d2| {
                        if (d2 != d and @abs(r_cart[d2][dir]) > 1e-10) {
                            others_zero = false;
                            break;
                        }
                    }
                    if (others_zero) {
                        pure_dir = d;
                        break;
                    }
                }
            }

            if (pure_dir) |md| {
                const mapped_pidx = 3 * mapped_atom + md;
                if (mapped_pidx < pert_to_irr[pidx]) {
                    // This perturbation can be obtained from mapped_pidx via inverse of S
                    pert_to_irr[pidx] = mapped_pidx;
                    pert_sym_idx[pidx] = isym;
                    // Store the direction coefficients: R_cart row for dir
                    pert_dir_coeffs[pidx] = .{
                        r_cart[0][dir],
                        r_cart[1][dir],
                        r_cart[2][dir],
                    };
                    is_irreducible[pidx] = false;
                }
            }
        }
    }

    // Resolve transitive mappings: follow chains to final representative
    var changed = true;
    while (changed) {
        changed = false;
        for (0..dim) |p| {
            const rep = pert_to_irr[p];
            if (pert_to_irr[rep] < rep) {
                pert_to_irr[p] = pert_to_irr[rep];
                changed = true;
            }
        }
    }

    // Mark final irreducible set
    for (0..dim) |p| {
        is_irreducible[p] = (pert_to_irr[p] == p);
    }

    // Collect irreducible perturbation indices
    var irr_list: std.ArrayList(usize) = .empty;
    errdefer irr_list.deinit(alloc);
    for (0..dim) |p| {
        if (is_irreducible[p]) {
            try irr_list.append(alloc, p);
        }
    }
    const n_irr = irr_list.items.len;
    const irr_indices = try irr_list.toOwnedSlice(alloc);

    return .{
        .irr_pert_indices = irr_indices,
        .n_irr_perts = n_irr,
        .pert_to_irr = pert_to_irr,
        .pert_sym_idx = pert_sym_idx,
        .pert_dir_coeffs = pert_dir_coeffs,
        .is_irreducible = is_irreducible,
    };
}

/// Reconstruct dynamical matrix columns for non-irreducible atoms (Γ-point, real).
/// For non-irreducible atom J with S(J) = I_irr:
///   D_{block}(I, J) = R × D_{block}(S(I), I_irr) × R^T
/// At Γ-point, phase factor = 1.
pub fn reconstructDynmatColumnsReal(
    dynmat: []f64,
    n_atoms: usize,
    irr_info: IrreducibleAtomInfo,
    symops: []const symmetry_mod.SymOp,
    indsym: []const []const usize,
    cell_bohr: math.Mat3,
) void {
    const dim = 3 * n_atoms;

    for (0..n_atoms) |jatom| {
        if (irr_info.is_irreducible[jatom]) continue;

        // S maps jatom -> irr_atom: indsym[isym][jatom] == irr_atom
        const irr_atom = irr_info.atom_to_irr[jatom];
        const isym = irr_info.atom_sym_idx[jatom];
        const r_cart = fracRotToCart(symops[isym].rot, cell_bohr);

        for (0..n_atoms) |iatom| {
            // S(iatom) under the same symmetry operation
            const si = indsym[isym][iatom];

            // Source block: D(S(iatom), irr_atom) — a 3x3 sub-matrix
            // Target block: D(iatom, jatom) = R × D(S(iatom), irr_atom) × R^T
            for (0..3) |alpha| {
                for (0..3) |beta| {
                    var val: f64 = 0.0;
                    for (0..3) |gamma_idx| {
                        for (0..3) |delta| {
                            val += r_cart[alpha][gamma_idx] * r_cart[beta][delta] *
                                dynmat[(3 * si + gamma_idx) * dim + (3 * irr_atom + delta)];
                        }
                    }
                    dynmat[(3 * iatom + alpha) * dim + (3 * jatom + beta)] = val;
                }
            }
        }
    }
}

/// Reconstruct dynamical matrix columns for non-irreducible atoms (q≠0, complex).
/// Includes phase factor exp(-i 2π q·(shift_I - shift_J)).
pub fn reconstructDynmatColumnsComplex(
    dynmat: []math.Complex,
    n_atoms: usize,
    irr_info: IrreducibleAtomInfo,
    symops: []const symmetry_mod.SymOp,
    indsym: []const []const usize,
    tnons_shift: []const []const math.Vec3,
    cell_bohr: math.Mat3,
    q_frac: math.Vec3,
) void {
    const dim = 3 * n_atoms;
    const two_pi = 2.0 * std.math.pi;

    for (0..n_atoms) |jatom| {
        if (irr_info.is_irreducible[jatom]) continue;

        const irr_atom = irr_info.atom_to_irr[jatom];
        const isym = irr_info.atom_sym_idx[jatom];
        const r_cart = fracRotToCart(symops[isym].rot, cell_bohr);

        for (0..n_atoms) |iatom| {
            const si = indsym[isym][iatom];

            // Phase factor: exp(-i 2π q·(shift_I - shift_J))
            const shift_i = tnons_shift[isym][iatom];
            const shift_j = tnons_shift[isym][jatom];
            const dshift = math.Vec3{
                .x = shift_i.x - shift_j.x,
                .y = shift_i.y - shift_j.y,
                .z = shift_i.z - shift_j.z,
            };
            const arg = -two_pi * (q_frac.x * dshift.x + q_frac.y * dshift.y + q_frac.z * dshift.z);
            const phase = math.complex.init(@cos(arg), @sin(arg));

            for (0..3) |alpha| {
                for (0..3) |beta| {
                    var val = math.complex.init(0.0, 0.0);
                    for (0..3) |gamma_idx| {
                        for (0..3) |delta| {
                            const coeff = r_cart[alpha][gamma_idx] * r_cart[beta][delta];
                            val = math.complex.add(val, math.complex.scale(
                                dynmat[(3 * si + gamma_idx) * dim + (3 * irr_atom + delta)],
                                coeff,
                            ));
                        }
                    }
                    dynmat[(3 * iatom + alpha) * dim + (3 * jatom + beta)] = math.complex.mul(val, phase);
                }
            }
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

test "ASR enforces zero acoustic sum" {
    // 2 atoms, 6×6 matrix
    var dynmat = [_]f64{
        1.0,  0.2,  0.0,  -0.8, -0.1, 0.0,
        0.2,  1.5,  0.1,  -0.1, -1.2, -0.1,
        0.0,  0.1,  0.8,  0.0,  -0.1, -0.7,
        -0.8, -0.1, 0.0,  1.2,  0.3,  0.0,
        -0.1, -1.2, -0.1, 0.3,  1.8,  0.2,
        0.0,  -0.1, -0.7, 0.0,  0.2,  0.9,
    };

    applyASR(&dynmat, 2);

    // Check: Σ_J D_{Iα,Jβ} = 0 for all I, α, β
    const dim = 6;
    for (0..2) |i| {
        for (0..3) |alpha| {
            for (0..3) |beta| {
                var sum: f64 = 0.0;
                for (0..2) |j| {
                    sum += dynmat[(3 * i + alpha) * dim + (3 * j + beta)];
                }
                try std.testing.expectApproxEqAbs(sum, 0.0, 1e-12);
            }
        }
    }
}

test "omega2 to cm-1 conversion" {
    // Test with a known value: Si optical mode ~521 cm⁻¹
    // ω² ≈ 3.5e-3 Ry/(bohr²·amu) for Si optical mode (approximate)
    const freq = omega2ToCm1(3.5e-3);
    // Should be in the right ballpark (hundreds of cm⁻¹)
    try std.testing.expect(freq > 100.0);
    try std.testing.expect(freq < 1000.0);

    // Zero frequency for acoustic
    try std.testing.expectApproxEqAbs(omega2ToCm1(0.0), 0.0, 1e-15);

    // Imaginary frequency returns negative
    try std.testing.expect(omega2ToCm1(-1e-4) < 0.0);
}

test "diagonalize simple matrix" {
    const alloc = std.testing.allocator;

    // Simple 2×2 symmetric matrix
    var mat = [_]f64{
        2.0, 1.0,
        1.0, 3.0,
    };

    var result = try diagonalize(alloc, &mat, 2);
    defer result.deinit(alloc);

    // Eigenvalues of [2,1;1,3] are (5±√5)/2 ≈ 1.382, 3.618
    const expected_1 = (5.0 - std.math.sqrt(5.0)) / 2.0;
    const expected_2 = (5.0 + std.math.sqrt(5.0)) / 2.0;

    try std.testing.expectApproxEqRel(result.omega2[0], expected_1, 1e-10);
    try std.testing.expectApproxEqRel(result.omega2[1], expected_2, 1e-10);
}

test "diagonalizeComplex Hermitian matrix" {
    const alloc = std.testing.allocator;

    // 2×2 Hermitian matrix: [[2, 1-i], [1+i, 3]]
    var mat = [_]math.Complex{
        math.complex.init(2.0, 0.0), math.complex.init(1.0, -1.0),
        math.complex.init(1.0, 1.0), math.complex.init(3.0, 0.0),
    };

    var result = try diagonalizeComplex(alloc, &mat, 2);
    defer result.deinit(alloc);

    // Eigenvalues of [[2, 1-i], [1+i, 3]]:
    // trace = 5, det = 6 - 2 = 4
    // λ = (5 ± √(25-16))/2 = (5 ± 3)/2 → 1, 4
    try std.testing.expectApproxEqRel(result.omega2[0], 1.0, 1e-10);
    try std.testing.expectApproxEqRel(result.omega2[1], 4.0, 1e-10);
}

test "diagonalizeComplex real Hermitian matches dsyev" {
    const alloc = std.testing.allocator;

    // Same matrix as the real test, but as complex with zero imaginary part
    var mat_real = [_]f64{
        2.0, 1.0,
        1.0, 3.0,
    };
    var mat_complex = [_]math.Complex{
        math.complex.init(2.0, 0.0), math.complex.init(1.0, 0.0),
        math.complex.init(1.0, 0.0), math.complex.init(3.0, 0.0),
    };

    var result_real = try diagonalize(alloc, &mat_real, 2);
    defer result_real.deinit(alloc);
    var result_complex = try diagonalizeComplex(alloc, &mat_complex, 2);
    defer result_complex.deinit(alloc);

    for (0..2) |i| {
        try std.testing.expectApproxEqRel(result_real.omega2[i], result_complex.omega2[i], 1e-10);
    }
}
