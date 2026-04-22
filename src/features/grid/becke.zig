//! Becke partitioning and molecular grid construction.
//!
//! Implements the Becke space-partitioning scheme [JCP 88, 2547 (1988)]
//! for multi-atom numerical integration. Each point in space is assigned
//! a weight for each atom, such that the weights sum to 1 everywhere.
//!
//! The molecular grid is constructed by combining:
//! - Radial quadrature (Treutler-Ahlrichs or Mura-Knowles)
//! - Angular quadrature (Lebedev)
//! - Becke partitioning weights
//!
//! All quantities are in Hartree atomic units (Bohr).

const std = @import("std");
const math = std.math;
const lebedev = @import("lebedev.zig");
const radial = @import("radial.zig");

/// A 3D grid point with its integration weight.
pub const GridPoint = struct {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
};

/// Atom specification for grid generation.
pub const Atom = struct {
    x: f64,
    y: f64,
    z: f64,
    z_number: usize,
};

/// Configuration for grid generation.
pub const GridConfig = struct {
    /// Number of radial points per atom.
    n_radial: usize = 75,
    /// Number of angular (Lebedev) points per shell.
    n_angular: usize = 302,
    /// Becke partitioning hardness parameter (number of iterations of
    /// the smoothing function). PySCF uses 3.
    becke_hardness: usize = 3,
    /// Whether to use Bragg-Slater atomic size adjustment.
    use_atomic_radii: bool = true,
    /// Pruning: reduce angular points near nucleus and far out.
    prune: bool = true,
};

/// Computes the distance between two 3D points.
fn distance(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) f64 {
    const dx = x1 - x2;
    const dy = y1 - y2;
    const dz = z1 - z2;
    return @sqrt(dx * dx + dy * dy + dz * dz);
}

/// Becke's smoothing function: f(x) = 0.5 * (1 - p(x))
/// where p(x) = 1.5*x - 0.5*x^3 (the step function polynomial).
/// Applied `k` times for sharper partitioning.
fn beckeSmooth(mu: f64, k: usize) f64 {
    var s = mu;
    for (0..k) |_| {
        s = 1.5 * s - 0.5 * s * s * s;
    }
    return 0.5 * (1.0 - s);
}

/// Computes the Becke partitioning weights for a single point (x, y, z)
/// among all atoms. Returns the weight for atom `atom_idx`.
///
/// The Becke scheme uses confocal ellipsoidal coordinates:
///   mu_ij = (r_i - r_j) / R_ij
/// where r_i = |r - R_i| is the distance from the point to atom i,
/// and R_ij = |R_i - R_j| is the inter-atomic distance.
///
/// With atomic size adjustment (Becke 1988, Eq. 21):
///   mu_ij -> mu_ij + a_ij * (1 - mu_ij^2)
/// where a_ij depends on the ratio of Bragg-Slater radii.
pub fn beckeWeight(
    x: f64,
    y: f64,
    z: f64,
    atoms: []const Atom,
    atom_idx: usize,
    config: GridConfig,
    /// Pre-computed inter-atomic distances (n_atoms x n_atoms, row-major).
    inter_distances: []const f64,
    /// Scratch buffer for per-atom P values (length n_atoms).
    p_buf: []f64,
) f64 {
    const n_atoms = atoms.len;

    if (n_atoms == 1) return 1.0;

    // Compute distances from the point to each atom
    // We'll reuse p_buf temporarily for distances, then overwrite with P values
    var dist_buf: [64]f64 = undefined;
    if (n_atoms > 64) @panic("Too many atoms for stack buffer");

    for (0..n_atoms) |i| {
        dist_buf[i] = distance(x, y, z, atoms[i].x, atoms[i].y, atoms[i].z);
    }

    // Initialize P_i = 1 for all atoms
    for (0..n_atoms) |i| {
        p_buf[i] = 1.0;
    }

    // For each pair (i, j), compute the cell function s(mu_ij)
    for (0..n_atoms) |i| {
        for ((i + 1)..n_atoms) |j| {
            const r_ij = inter_distances[i * n_atoms + j];

            if (r_ij < 1e-14) continue;

            var mu = (dist_buf[i] - dist_buf[j]) / r_ij;

            // Atomic size adjustment
            if (config.use_atomic_radii) {
                const chi = radial.braggSlaterRadius(atoms[i].z_number) /
                    radial.braggSlaterRadius(atoms[j].z_number);
                // u_ij = (chi - 1) / (chi + 1)
                const u = (chi - 1.0) / (chi + 1.0);
                // a_ij = u / (u^2 - 1), clamped to [-0.5, 0.5]
                var a = u / (u * u - 1.0);
                a = @max(-0.5, @min(0.5, a));

                mu += a * (1.0 - mu * mu);
            }

            // Apply smoothing function k times
            const s_ij = beckeSmooth(mu, config.becke_hardness);
            const s_ji = 1.0 - s_ij;

            p_buf[i] *= s_ij;
            p_buf[j] *= s_ji;
        }
    }

    // Normalize: w_i = P_i / sum(P_j)
    var p_sum: f64 = 0.0;
    for (0..n_atoms) |i| {
        p_sum += p_buf[i];
    }

    if (p_sum < 1e-300) return 0.0;

    return p_buf[atom_idx] / p_sum;
}

/// Determines the number of angular points for a given radial shell,
/// using a simple pruning scheme similar to PySCF's nwchem_prune.
fn prunedAngularPoints(i_radial: usize, n_radial: usize, n_angular_max: usize) usize {
    const fi: f64 = @floatFromInt(i_radial);
    const fn_: f64 = @floatFromInt(n_radial);
    const frac = fi / fn_;

    // Near nucleus and far out: use fewer angular points
    if (frac < 0.1) {
        return @min(n_angular_max, 14);
    } else if (frac < 0.2) {
        return @min(n_angular_max, 50);
    } else if (frac < 0.8) {
        return n_angular_max;
    } else if (frac < 0.9) {
        return @min(n_angular_max, 50);
    } else {
        return @min(n_angular_max, 14);
    }
}

/// Builds a molecular integration grid for the given set of atoms.
///
/// The grid combines:
/// 1. Atom-centered radial grids (Treutler-Ahlrichs)
/// 2. Lebedev angular grids
/// 3. Becke space partitioning
///
/// Returns an array of GridPoints with combined weights (including
/// the r^2 volume element, the 4*pi angular normalization, and
/// the Becke partitioning weight).
pub fn buildMolecularGrid(
    allocator: std.mem.Allocator,
    atoms: []const Atom,
    config: GridConfig,
) ![]GridPoint {
    const n_atoms = atoms.len;

    // Pre-compute inter-atomic distances
    const inter_distances = try allocator.alloc(f64, n_atoms * n_atoms);
    defer allocator.free(inter_distances);

    for (0..n_atoms) |i| {
        for (0..n_atoms) |j| {
            inter_distances[i * n_atoms + j] = distance(
                atoms[i].x,
                atoms[i].y,
                atoms[i].z,
                atoms[j].x,
                atoms[j].y,
                atoms[j].z,
            );
        }
    }

    // Becke weight scratch buffer
    const p_buf = try allocator.alloc(f64, n_atoms);
    defer allocator.free(p_buf);

    // Collect all grid points
    var grid_points = std.ArrayList(GridPoint).empty;
    defer grid_points.deinit(allocator);

    // For each atom, generate atom-centered grid
    for (0..n_atoms) |iatom| {
        // Generate radial grid
        const rad_grid = try radial.defaultRadialGrid(
            allocator,
            atoms[iatom].z_number,
            config.n_radial,
        );
        defer allocator.free(rad_grid);

        // For each radial shell
        for (0..config.n_radial) |ir| {
            const r = rad_grid[ir].r;
            const w_r = rad_grid[ir].w;

            if (r < 1e-15 or w_r < 1e-300) continue;

            // Determine angular grid size (with optional pruning)
            const n_ang = if (config.prune)
                prunedAngularPoints(ir, config.n_radial, config.n_angular)
            else
                config.n_angular;

            const ang_grid = lebedev.getLebedevGrid(n_ang);

            // For each angular point
            for (ang_grid) |apt| {
                // Convert to Cartesian coordinates centered on atom
                const gx = atoms[iatom].x + r * apt.x;
                const gy = atoms[iatom].y + r * apt.y;
                const gz = atoms[iatom].z + r * apt.z;

                // Compute Becke partitioning weight for this atom
                const bw = beckeWeight(
                    gx,
                    gy,
                    gz,
                    atoms,
                    iatom,
                    config,
                    inter_distances,
                    p_buf,
                );

                // Combined weight:
                // w = w_radial * r^2 * w_angular * 4*pi * w_becke
                const w = w_r * r * r * apt.w * 4.0 * math.pi * bw;

                if (w > 1e-300) {
                    try grid_points.append(allocator, .{
                        .x = gx,
                        .y = gy,
                        .z = gz,
                        .w = w,
                    });
                }
            }
        }
    }

    return grid_points.toOwnedSlice(allocator);
}

// --- Tests ---

test "becke weight single atom is 1" {
    const atoms = [_]Atom{
        .{ .x = 0.0, .y = 0.0, .z = 0.0, .z_number = 8 },
    };
    var p_buf = [_]f64{0.0};
    const inter_d = [_]f64{0.0};

    const w = beckeWeight(1.0, 0.0, 0.0, &atoms, 0, .{}, &inter_d, &p_buf);
    try std.testing.expectApproxEqAbs(1.0, w, 1e-14);
}

test "becke weights sum to 1" {
    // Two atoms: O at origin, H at (0, 0, 1.5)
    const atoms = [_]Atom{
        .{ .x = 0.0, .y = 0.0, .z = 0.0, .z_number = 8 },
        .{ .x = 0.0, .y = 0.0, .z = 1.5, .z_number = 1 },
    };

    const inter_d = [_]f64{
        0.0, 1.5,
        1.5, 0.0,
    };

    var p_buf = [_]f64{ 0.0, 0.0 };

    // Test at several points
    const test_points = [_][3]f64{
        .{ 0.5, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.75 }, // midpoint
        .{ 0.0, 0.0, 1.0 },
        .{ 1.0, 1.0, 1.0 },
    };

    for (test_points) |pt| {
        const w0 = beckeWeight(pt[0], pt[1], pt[2], &atoms, 0, .{}, &inter_d, &p_buf);
        const w1 = beckeWeight(pt[0], pt[1], pt[2], &atoms, 1, .{}, &inter_d, &p_buf);
        try std.testing.expectApproxEqAbs(1.0, w0 + w1, 1e-14);
    }
}

test "molecular grid integrates constant function" {
    // For a single atom, integral of f(r)=1 over the grid should give
    // approximately 4*pi * integral(r^2 dr) for the radial extent.
    // Instead, let's integrate a normalized Gaussian: (alpha/pi)^(3/2) * exp(-alpha*r^2)
    // which should integrate to 1.0.
    const allocator = std.testing.allocator;

    const atoms = [_]Atom{
        .{ .x = 0.0, .y = 0.0, .z = 0.0, .z_number = 8 },
    };

    const config = GridConfig{
        .n_radial = 75,
        .n_angular = 302,
        .prune = false,
    };

    const grid = try buildMolecularGrid(allocator, &atoms, config);
    defer allocator.free(grid);

    // Integrate a normalized Gaussian: (alpha/pi)^(3/2) * exp(-alpha*r^2)
    const alpha = 1.0;
    const prefactor = @sqrt(alpha / math.pi) * (alpha / math.pi);
    var integral: f64 = 0.0;
    for (grid) |pt| {
        const r2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
        integral += prefactor * @exp(-alpha * r2) * pt.w;
    }

    try std.testing.expectApproxEqAbs(1.0, integral, 1e-6);
}

test "molecular grid H2O integrates Gaussian" {
    const allocator = std.testing.allocator;

    // H2O geometry in Bohr
    const atoms = [_]Atom{
        .{ .x = 0.0, .y = 0.0, .z = 0.0, .z_number = 8 },
        .{ .x = 0.0, .y = -1.43047, .z = 1.10877, .z_number = 1 },
        .{ .x = 0.0, .y = 1.43047, .z = 1.10877, .z_number = 1 },
    };

    const config = GridConfig{
        .n_radial = 75,
        .n_angular = 302,
        .prune = false,
    };

    const grid = try buildMolecularGrid(allocator, &atoms, config);
    defer allocator.free(grid);

    // Integrate a normalized s-type Gaussian centered at oxygen
    const alpha = 0.5;
    const prefactor = @sqrt(alpha / math.pi) * (alpha / math.pi);
    var integral: f64 = 0.0;
    for (grid) |pt| {
        const r2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
        integral += prefactor * @exp(-alpha * r2) * pt.w;
    }

    // Should integrate to 1.0 regardless of Becke partition
    try std.testing.expectApproxEqAbs(1.0, integral, 1e-5);
}
