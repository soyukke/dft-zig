const std = @import("std");

const ionic_potential = @import("ionic_potential.zig");
const local_orbital = @import("local_orbital.zig");
const local_orbital_potential = @import("local_orbital_potential.zig");
const neighbor_list = @import("neighbor_list.zig");
const sparse = @import("sparse.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const pw_nonlocal = @import("../pseudopotential/nonlocal.zig");
const math = @import("../math/math.zig");

/// Ion site with pseudopotential data for nonlocal calculations
/// Reuses the same structure as ionic_potential.IonSite
pub const IonSite = ionic_potential.IonSite;

/// Options for building nonlocal potential matrix
pub const NonlocalOptions = struct {
    sigma: f64,
    cutoff: f64,
    /// Number of radial integration points
    n_radial: usize = 100,
    /// Maximum radius for integration (in Bohr)
    r_max: f64 = 10.0,
    /// Threshold for dropping small matrix elements
    threshold: f64 = 0.0,
    /// Basis type for nonlocal calculation:
    /// - s_only: Only s-type Gaussian overlap (captures s-wave projector)
    /// - sp: Include effective p-character contribution (captures p-wave projector)
    ///
    /// Note: Matrix size is always N×N (one orbital per center).
    /// For sp mode, the p-wave contribution is computed using virtual p-orbitals
    /// at each center and weighted by sp³ hybridization factor (75% p-character).
    basis: local_orbital.BasisType = .s_only,
};

/// Build nonlocal potential matrix in local orbital basis
/// V_nl(i,j) = Σ_I Σ_lm D_l <φ_i|β_lm^I> <β_lm^I|φ_j>
///
/// For sp basis mode, the matrix size is still N×N (one per center),
/// but both s-wave and p-wave projector contributions are included
/// with sp³ hybridization weighting (25% s, 75% p).
pub fn buildNonlocalCsr(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    ions: []const IonSite,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: NonlocalOptions,
) !sparse.CsrMatrix {
    if (centers.len == 0) return error.InvalidShape;

    const n_centers = centers.len;
    const alpha = 1.0 / (opts.sigma * opts.sigma);

    // Build s-type orbitals (always needed)
    const s_orbitals = try local_orbital.buildOrbitalsS(alloc, centers, alpha, opts.cutoff);
    defer alloc.free(s_orbitals);

    // Optionally build p-type orbitals for sp mode
    const sp_orbitals: ?[]local_orbital.Orbital = if (opts.basis == .sp)
        try local_orbital.buildOrbitalsSP(alloc, centers, alpha, opts.cutoff)
    else
        null;
    defer if (sp_orbitals) |orbs| alloc.free(orbs);

    // Matrix size is always N (one orbital per center)
    const n_orb = n_centers;

    // Compute projector overlaps for each ion and each orbital
    // <φ_i|β_b^I> where I is ion index, b is beta index
    var triplets: std.ArrayList(sparse.Triplet) = .empty;
    defer triplets.deinit(alloc);

    const inv_cell = try invertCell(cell);

    for (ions) |ion| {
        const upf = ion.upf;
        if (upf.beta.len == 0) continue;

        // Get D_ij matrix size
        const n_beta = upf.beta.len;
        if (upf.dij.len != n_beta * n_beta) continue;

        // Precompute plane-wave projector values at G=0 for reference
        const pw_projectors = try alloc.alloc(f64, n_beta);
        defer alloc.free(pw_projectors);
        for (upf.beta, 0..) |beta, b| {
            const l = beta.l orelse 0;
            pw_projectors[b] = pw_nonlocal.radialProjector(beta.values, upf.r, upf.rab, l, 0.0);
        }

        // Compute overlaps for s-orbitals
        const s_overlaps = try alloc.alloc(f64, n_orb * n_beta);
        defer alloc.free(s_overlaps);
        @memset(s_overlaps, 0.0);

        for (s_orbitals, 0..) |orb, i| {
            for (upf.beta, 0..) |beta, b| {
                const l = beta.l orelse 0;
                s_overlaps[i * n_beta + b] = computeProjectorOverlap(
                    orb,
                    ion.position,
                    upf.r,
                    upf.rab,
                    beta.values,
                    l,
                    cell,
                    inv_cell,
                    pbc,
                    opts.n_radial,
                    opts.r_max,
                    pw_projectors[b],
                );
            }
        }

        // For sp mode, also compute p-orbital overlaps and combine with sp³ weighting
        // sp³ hybridization: 25% s-character, 75% p-character
        const s_weight: f64 = if (opts.basis == .sp) 0.25 else 1.0;
        const p_weight: f64 = if (opts.basis == .sp) 0.75 else 0.0;

        var p_overlaps: ?[]f64 = null;
        if (sp_orbitals) |sp_orbs| {
            p_overlaps = try alloc.alloc(f64, n_orb * n_beta);
            @memset(p_overlaps.?, 0.0);

            // For each center, average over px, py, pz overlaps
            for (0..n_orb) |center_idx| {
                for (upf.beta, 0..) |beta, b| {
                    const l = beta.l orelse 0;
                    var p_sum: f64 = 0.0;

                    // Compute overlaps for px, py, pz at this center
                    for (1..4) |p_idx| { // indices 1,2,3 are px,py,pz in sp basis
                        const orb_idx = center_idx * 4 + p_idx;
                        const overlap_val = computeProjectorOverlap(
                            sp_orbs[orb_idx],
                            ion.position,
                            upf.r,
                            upf.rab,
                            beta.values,
                            l,
                            cell,
                            inv_cell,
                            pbc,
                            opts.n_radial,
                            opts.r_max,
                            pw_projectors[b],
                        );
                        p_sum += overlap_val * overlap_val; // sum of squares
                    }
                    // RMS average over 3 p-orbitals
                    p_overlaps.?[center_idx * n_beta + b] = @sqrt(p_sum / 3.0);
                }
            }
        }
        defer if (p_overlaps) |p| alloc.free(p);

        // Build V_nl(i,j) = Σ_b,b' <φ_i|β_b> D_{bb'} <β_b'|φ_j>
        // For sp mode: weight = s_weight * s_overlap + p_weight * p_overlap
        for (0..n_orb) |i| {
            for (0..n_orb) |j| {
                var value: f64 = 0.0;
                for (0..n_beta) |b| {
                    for (0..n_beta) |bp| {
                        const d_val = upf.dij[b * n_beta + bp];

                        // s-contribution
                        const s_proj_i = s_overlaps[i * n_beta + b];
                        const s_proj_j = s_overlaps[j * n_beta + bp];
                        value += s_weight * s_proj_i * d_val * s_proj_j;

                        // p-contribution (if sp mode)
                        if (p_overlaps) |p| {
                            const p_proj_i = p[i * n_beta + b];
                            const p_proj_j = p[j * n_beta + bp];
                            value += p_weight * p_proj_i * d_val * p_proj_j;
                        }
                    }
                }
                if (@abs(value) > opts.threshold) {
                    try triplets.append(alloc, .{ .row = i, .col = j, .value = value });
                }
            }
        }
    }

    if (triplets.items.len == 0) {
        // Return empty matrix with correct dimensions
        const row_ptr = try alloc.alloc(usize, n_orb + 1);
        @memset(row_ptr, 0);
        return .{
            .nrows = n_orb,
            .ncols = n_orb,
            .row_ptr = row_ptr,
            .col_idx = &[_]usize{},
            .values = &[_]f64{},
        };
    }

    return sparse.CsrMatrix.initFromTriplets(alloc, n_orb, n_orb, triplets.items);
}

/// Compute overlap integral <φ|β_lm> via numerical integration
/// φ(r) = N f(r) Y_lm_orb is a Gaussian orbital (s or p type)
/// β_lm(r) = f_l(r) Y_lm(θ,φ) is a KB projector centered at R_I
fn computeProjectorOverlap(
    orb: local_orbital.Orbital,
    ion_pos: math.Vec3,
    r_grid: []const f64,
    rab_grid: []const f64,
    beta_data: []const f64,
    l_proj: i32,
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    n_radial: usize,
    r_max: f64,
    pw_projector: f64,
) f64 {
    _ = n_radial;

    // Vector from ion to orbital center
    var delta = math.Vec3.sub(orb.center, ion_pos);
    delta = minimumImageDelta(cell, inv_cell, pbc, delta);
    const dist = @sqrt(math.Vec3.dot(delta, delta));

    // Cutoff: if orbital is too far from ion, overlap is negligible
    // Use effective range = projector r_max + Gaussian sigma
    const sigma = 1.0 / @sqrt(orb.alpha);
    const effective_range = r_max + 3.0 * sigma;
    if (dist > effective_range) {
        return 0.0;
    }

    // Match orbital angular momentum with projector angular momentum.
    // Selection rules determine which combinations give significant overlap.
    const l_orb = orb.angular.l();
    const overlap_fn = selectOverlapFn(l_orb, l_proj) orelse return 0.0;
    return overlap_fn(orb, dist, delta, r_grid, rab_grid, beta_data, pw_projector);
}

const ProjectorOverlapFn = *const fn (
    orb: local_orbital.Orbital,
    dist: f64,
    delta: math.Vec3,
    r_grid: []const f64,
    rab_grid: []const f64,
    beta_data: []const f64,
    pw_projector: f64,
) f64;

fn selectOverlapFn(l_orb: i32, l_proj: i32) ?ProjectorOverlapFn {
    if (l_orb == 0 and l_proj == 0) return computeProjectorOverlapSS;
    if (l_orb == 0 and l_proj == 1) return computeProjectorOverlapSP;
    if (l_orb == 1 and l_proj == 0) return computeProjectorOverlapPS;
    if (l_orb == 1 and l_proj == 1) return computeProjectorOverlapPP;
    return null;
}

/// s-orbital with s-projector overlap
/// <φ_s|β_s> where both have l=0
fn computeProjectorOverlapSS(
    orb: local_orbital.Orbital,
    dist: f64,
    delta: math.Vec3,
    r_grid: []const f64,
    rab_grid: []const f64,
    beta_data: []const f64,
    pw_projector: f64,
) f64 {
    _ = delta;
    var sum: f64 = 0.0;
    var pw_sum: f64 = 0.0;
    const n = @min(beta_data.len, @min(r_grid.len, rab_grid.len));

    for (0..n) |i| {
        const r = r_grid[i];
        if (r < 1e-10) continue;

        const rab = rab_grid[i];
        const r_beta = beta_data[i];
        const gauss_radial = gaussianRadialIntegrand(orb.alpha, r, dist);

        sum += gauss_radial * r_beta * r * rab;
        pw_sum += r_beta * r * rab;
    }

    if (@abs(pw_sum) < 1e-15) return 0.0;
    return pw_projector * (sum / pw_sum);
}

/// s-orbital with p-projector overlap
/// <φ_s|β_p> - small due to angular mismatch
fn computeProjectorOverlapSP(
    orb: local_orbital.Orbital,
    dist: f64,
    delta: math.Vec3,
    r_grid: []const f64,
    rab_grid: []const f64,
    beta_data: []const f64,
    pw_projector: f64,
) f64 {
    _ = pw_projector;

    // For s-orbital with p-projector, the overlap depends on distance
    // At dist=0, the overlap is zero due to angular orthogonality
    // At finite dist, there's a small overlap proportional to dist

    var sum: f64 = 0.0;
    var pw_sum: f64 = 0.0;
    const n = @min(beta_data.len, @min(r_grid.len, rab_grid.len));

    const sigma = 1.0 / @sqrt(orb.alpha);
    const k_eff = 1.0 / sigma;

    for (0..n) |i| {
        const r = r_grid[i];
        if (r < 1e-10) continue;

        const rab = rab_grid[i];
        const r_beta = beta_data[i];

        // Use j_1 for p-projector
        const j1 = pw_nonlocal.sphericalBessel(1, k_eff * r);
        const gauss_radial = gaussianRadialIntegrand(orb.alpha, r, dist);

        sum += gauss_radial * r_beta * r * j1 * rab;
        pw_sum += r_beta * r * j1 * rab;
    }

    // Angular factor: depends on direction of displacement
    // For s-p overlap, proportional to cos(theta) where theta is angle to p-orbital axis
    const delta_norm = @sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    const angular_factor = if (dist < 1e-10) 0.0 else delta_norm / dist * 0.5;

    if (@abs(pw_sum) < 1e-15) return 0.0;
    return sum * angular_factor;
}

/// p-orbital with s-projector overlap
/// <φ_p|β_s> - small due to angular mismatch
fn computeProjectorOverlapPS(
    orb: local_orbital.Orbital,
    dist: f64,
    delta: math.Vec3,
    r_grid: []const f64,
    rab_grid: []const f64,
    beta_data: []const f64,
    pw_projector: f64,
) f64 {
    // Similar to SP but with p-orbital character
    var sum: f64 = 0.0;
    var pw_sum: f64 = 0.0;
    const n = @min(beta_data.len, @min(r_grid.len, rab_grid.len));

    for (0..n) |i| {
        const r = r_grid[i];
        if (r < 1e-10) continue;

        const rab = rab_grid[i];
        const r_beta = beta_data[i];
        const gauss_radial = gaussianRadialIntegrand(orb.alpha, r, dist);

        sum += gauss_radial * r_beta * r * rab;
        pw_sum += r_beta * r * rab;
    }

    // Angular factor for p-orbital: depends on which p component and direction
    const dir_component = switch (orb.angular) {
        .px => delta.x,
        .py => delta.y,
        .pz => delta.z,
        .s => 0.0,
    };
    const angular_factor = if (dist < 1e-10) 0.0 else dir_component / dist * 0.5;

    if (@abs(pw_sum) < 1e-15) return 0.0;
    return pw_projector * (sum / pw_sum) * angular_factor;
}

/// p-orbital with p-projector overlap
/// <φ_p|β_p> - good overlap when directions match
fn computeProjectorOverlapPP(
    orb: local_orbital.Orbital,
    dist: f64,
    delta: math.Vec3,
    r_grid: []const f64,
    rab_grid: []const f64,
    beta_data: []const f64,
    pw_projector: f64,
) f64 {
    _ = pw_projector;

    var sum: f64 = 0.0;
    var pw_sum: f64 = 0.0;
    const n = @min(beta_data.len, @min(r_grid.len, rab_grid.len));

    const sigma = 1.0 / @sqrt(orb.alpha);
    const k_eff = 1.0 / sigma;

    for (0..n) |i| {
        const r = r_grid[i];
        if (r < 1e-10) continue;

        const rab = rab_grid[i];
        const r_beta = beta_data[i];

        // Use j_1 for p-projector
        const j1 = pw_nonlocal.sphericalBessel(1, k_eff * r);
        const gauss_radial = gaussianRadialIntegrand(orb.alpha, r, dist);

        sum += gauss_radial * r_beta * r * j1 * rab;
        pw_sum += r_beta * r * j1 * rab;
    }

    // Angular factor for p-p overlap depends on the orbital direction
    // For px orbital: <px|p_m> is maximized when m=+1 (x-direction)
    // We use an averaged factor that gives reasonable values
    const dir_component = switch (orb.angular) {
        .px => delta.x,
        .py => delta.y,
        .pz => delta.z,
        .s => 0.0,
    };

    // For same-site (dist~0), the overlap is dominated by the radial part
    // For displaced centers, there's an angular modulation
    const angular_factor = if (dist < 1e-10)
        1.0 / 3.0 // 1/3 average over m=-1,0,+1
    else
        (1.0 / 3.0) * (1.0 + 2.0 * dir_component * dir_component / (dist * dist));

    if (@abs(pw_sum) < 1e-15) return 0.0;
    return sum * angular_factor;
}

/// Gaussian radial integrand approximation
fn gaussianRadialIntegrand(alpha: f64, r: f64, dist: f64) f64 {
    // Simplified: use expansion in terms of modified spherical Bessel functions
    // For a Gaussian centered at distance 'dist', integrate over sphere at radius r
    // Result involves I_0(2*alpha*r*dist) * exp(-alpha*(r²+dist²))
    if (dist < 1e-10) {
        return @exp(-alpha * r * r);
    }

    const x = 2.0 * alpha * r * dist;
    const exp_factor = @exp(-alpha * (r * r + dist * dist));

    // sinh(x)/x is the 0th spherical Bessel function of imaginary argument
    // I_0(x) = sinh(x)/x for spherical case
    const bessel_i0 = if (x < 1e-10) 1.0 else std.math.sinh(x) / x;

    return bessel_i0 * exp_factor;
}

fn invertCell(cell: math.Mat3) !math.Mat3 {
    return local_orbital_potential.invertCell(cell);
}

fn minimumImageDelta(
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    delta: math.Vec3,
) math.Vec3 {
    var frac = inv_cell.mulVec(delta);
    if (pbc.x) frac.x -= @round(frac.x);
    if (pbc.y) frac.y -= @round(frac.y);
    if (pbc.z) frac.z -= @round(frac.z);
    return cell.mulVec(frac);
}

test "nonlocal matrix is symmetric" {
    const alloc = std.testing.allocator;

    // Simple test with 2 orbitals
    const centers = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 2.0, .y = 0.0, .z = 0.0 },
    };
    const cell = math.Mat3.fromRows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    );
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };

    // Create minimal UPF data
    var r = [_]f64{ 0.0, 0.5, 1.0, 1.5, 2.0 };
    var rab = [_]f64{ 0.5, 0.5, 0.5, 0.5, 0.5 };
    var beta_data = [_]f64{ 0.0, 0.1, 0.2, 0.1, 0.0 }; // r*β(r)
    var dij = [_]f64{1.0}; // 1x1 D matrix

    const beta = pseudo.Beta{
        .l = 0,
        .values = beta_data[0..],
    };
    const betas = [_]pseudo.Beta{beta};

    const upf = pseudo.UpfData{
        .r = r[0..],
        .rab = rab[0..],
        .v_local = &[_]f64{},
        .beta = betas[0..],
        .dij = dij[0..],
        .qij = &[_]f64{},
        .nlcc = &[_]f64{},
    };

    const ions = [_]IonSite{
        .{ .position = .{ .x = 1.0, .y = 0.0, .z = 0.0 }, .upf = &upf },
    };

    const opts = NonlocalOptions{
        .sigma = 0.5,
        .cutoff = 5.0,
        .n_radial = 50,
        .r_max = 5.0,
        .threshold = 0.0,
    };

    var result = try buildNonlocalCsr(alloc, centers[0..], ions[0..], cell, pbc, opts);
    defer result.deinit(alloc);

    // Check symmetry: V(0,1) should equal V(1,0)
    const v01 = result.valueAt(0, 1);
    const v10 = result.valueAt(1, 0);
    try std.testing.expectApproxEqAbs(v01, v10, 1e-10);
}
