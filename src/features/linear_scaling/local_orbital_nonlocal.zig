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

const OrbitalSet = struct {
    s_orbitals: []local_orbital.Orbital,
    sp_orbitals: ?[]local_orbital.Orbital,

    fn deinit(self: OrbitalSet, alloc: std.mem.Allocator) void {
        alloc.free(self.s_orbitals);
        if (self.sp_orbitals) |sp_orbitals| alloc.free(sp_orbitals);
    }
};

const IonOverlapData = struct {
    n_beta: usize,
    s_overlaps: []f64,
    p_overlaps: ?[]f64,

    fn deinit(self: IonOverlapData, alloc: std.mem.Allocator) void {
        alloc.free(self.s_overlaps);
        if (self.p_overlaps) |p_overlaps| alloc.free(p_overlaps);
    }
};

const SpWeights = struct {
    s: f64,
    p: f64,
};

fn init_orbital_set(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    alpha: f64,
    cutoff: f64,
    basis: local_orbital.BasisType,
) !OrbitalSet {
    return .{
        .s_orbitals = try local_orbital.build_orbitals_s(alloc, centers, alpha, cutoff),
        .sp_orbitals = if (basis == .sp)
            try local_orbital.build_orbitals_sp(alloc, centers, alpha, cutoff)
        else
            null,
    };
}

fn init_projector_reference_values(
    alloc: std.mem.Allocator,
    ion: IonSite,
) ![]f64 {
    const upf = ion.upf;
    var pw_projectors = try alloc.alloc(f64, upf.beta.len);
    for (upf.beta, 0..) |beta, b| {
        const l = beta.l orelse 0;
        pw_projectors[b] = pw_nonlocal.radial_projector(
            beta.values,
            upf.r,
            upf.rab,
            l,
            0.0,
        );
    }
    return pw_projectors;
}

fn compute_orbital_projector_overlaps(
    alloc: std.mem.Allocator,
    orbitals: []const local_orbital.Orbital,
    ion: IonSite,
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: NonlocalOptions,
    pw_projectors: []const f64,
) ![]f64 {
    const upf = ion.upf;
    var overlaps = try alloc.alloc(f64, orbitals.len * upf.beta.len);
    @memset(overlaps, 0.0);

    for (orbitals, 0..) |orb, i| {
        for (upf.beta, 0..) |beta, b| {
            const l = beta.l orelse 0;
            overlaps[i * upf.beta.len + b] = compute_projector_overlap(
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
    return overlaps;
}

fn compute_p_projector_overlaps(
    alloc: std.mem.Allocator,
    sp_orbitals: ?[]const local_orbital.Orbital,
    n_orb: usize,
    ion: IonSite,
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: NonlocalOptions,
    pw_projectors: []const f64,
) !?[]f64 {
    const orbitals = sp_orbitals orelse return null;
    const upf = ion.upf;
    var overlaps = try alloc.alloc(f64, n_orb * upf.beta.len);
    @memset(overlaps, 0.0);

    for (0..n_orb) |center_idx| {
        for (upf.beta, 0..) |beta, b| {
            const l = beta.l orelse 0;
            var p_sum: f64 = 0.0;
            for (1..4) |p_idx| {
                const orb_idx = center_idx * 4 + p_idx;
                const overlap_val = compute_projector_overlap(
                    orbitals[orb_idx],
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
                p_sum += overlap_val * overlap_val;
            }
            overlaps[center_idx * upf.beta.len + b] = @sqrt(p_sum / 3.0);
        }
    }
    return overlaps;
}

fn init_ion_overlap_data(
    alloc: std.mem.Allocator,
    orbitals: OrbitalSet,
    ion: IonSite,
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: NonlocalOptions,
) !?IonOverlapData {
    const upf = ion.upf;
    const n_beta = upf.beta.len;
    if (n_beta == 0 or upf.dij.len != n_beta * n_beta) return null;

    const pw_projectors = try init_projector_reference_values(alloc, ion);
    defer alloc.free(pw_projectors);

    const s_overlaps = try compute_orbital_projector_overlaps(
        alloc,
        orbitals.s_orbitals,
        ion,
        cell,
        inv_cell,
        pbc,
        opts,
        pw_projectors,
    );
    errdefer alloc.free(s_overlaps);

    return .{
        .n_beta = n_beta,
        .s_overlaps = s_overlaps,
        .p_overlaps = try compute_p_projector_overlaps(
            alloc,
            if (orbitals.sp_orbitals) |sp_orbitals| sp_orbitals else null,
            orbitals.s_orbitals.len,
            ion,
            cell,
            inv_cell,
            pbc,
            opts,
            pw_projectors,
        ),
    };
}

fn basis_overlap_weights(basis: local_orbital.BasisType) SpWeights {
    return if (basis == .sp)
        .{ .s = 0.25, .p = 0.75 }
    else
        .{ .s = 1.0, .p = 0.0 };
}

fn p_overlap_at(
    p_overlaps: ?[]const f64,
    n_beta: usize,
    center_idx: usize,
    beta_idx: usize,
) ?f64 {
    const overlaps = p_overlaps orelse return null;
    return overlaps[center_idx * n_beta + beta_idx];
}

fn projector_pair_contribution(
    d_val: f64,
    s_proj_i: f64,
    s_proj_j: f64,
    p_proj_i: ?f64,
    p_proj_j: ?f64,
    weights: SpWeights,
) f64 {
    var value = weights.s * s_proj_i * d_val * s_proj_j;
    if (p_proj_i) |p_i| {
        value += weights.p * p_i * d_val * p_proj_j.?;
    }
    return value;
}

fn append_ion_nonlocal_triplets(
    alloc: std.mem.Allocator,
    triplets: *std.ArrayList(sparse.Triplet),
    n_orb: usize,
    ion: IonSite,
    overlap_data: IonOverlapData,
    basis: local_orbital.BasisType,
    threshold: f64,
) !void {
    const upf = ion.upf;
    const weights = basis_overlap_weights(basis);

    for (0..n_orb) |i| {
        for (0..n_orb) |j| {
            var value: f64 = 0.0;
            for (0..overlap_data.n_beta) |b| {
                for (0..overlap_data.n_beta) |bp| {
                    const d_val = upf.dij[b * overlap_data.n_beta + bp];
                    const s_proj_i = overlap_data.s_overlaps[i * overlap_data.n_beta + b];
                    const s_proj_j = overlap_data.s_overlaps[j * overlap_data.n_beta + bp];
                    value += projector_pair_contribution(
                        d_val,
                        s_proj_i,
                        s_proj_j,
                        p_overlap_at(overlap_data.p_overlaps, overlap_data.n_beta, i, b),
                        p_overlap_at(overlap_data.p_overlaps, overlap_data.n_beta, j, bp),
                        weights,
                    );
                }
            }
            if (@abs(value) > threshold) {
                try triplets.append(alloc, .{ .row = i, .col = j, .value = value });
            }
        }
    }
}

fn empty_nonlocal_csr(
    alloc: std.mem.Allocator,
    n_orb: usize,
) !sparse.CsrMatrix {
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

/// Build nonlocal potential matrix in local orbital basis
/// V_nl(i,j) = Σ_I Σ_lm D_l <φ_i|β_lm^I> <β_lm^I|φ_j>
///
/// For sp basis mode, the matrix size is still N×N (one per center),
/// but both s-wave and p-wave projector contributions are included
/// with sp³ hybridization weighting (25% s, 75% p).
pub fn build_nonlocal_csr(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    ions: []const IonSite,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: NonlocalOptions,
) !sparse.CsrMatrix {
    if (centers.len == 0) return error.InvalidShape;

    const n_orb = centers.len;
    const alpha = 1.0 / (opts.sigma * opts.sigma);
    var orbitals = try init_orbital_set(alloc, centers, alpha, opts.cutoff, opts.basis);
    defer orbitals.deinit(alloc);

    var triplets: std.ArrayList(sparse.Triplet) = .empty;
    defer triplets.deinit(alloc);

    const inv_cell = try invert_cell(cell);

    for (ions) |ion| {
        const overlap_data = try init_ion_overlap_data(
            alloc,
            orbitals,
            ion,
            cell,
            inv_cell,
            pbc,
            opts,
        ) orelse continue;
        defer overlap_data.deinit(alloc);

        try append_ion_nonlocal_triplets(
            alloc,
            &triplets,
            n_orb,
            ion,
            overlap_data,
            opts.basis,
            opts.threshold,
        );
    }

    if (triplets.items.len == 0) {
        return empty_nonlocal_csr(alloc, n_orb);
    }

    return sparse.CsrMatrix.init_from_triplets(alloc, n_orb, n_orb, triplets.items);
}

/// Compute overlap integral <φ|β_lm> via numerical integration
/// φ(r) = N f(r) Y_lm_orb is a Gaussian orbital (s or p type)
/// β_lm(r) = f_l(r) Y_lm(θ,φ) is a KB projector centered at R_I
fn compute_projector_overlap(
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
    delta = minimum_image_delta(cell, inv_cell, pbc, delta);
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
    const overlap_fn = select_overlap_fn(l_orb, l_proj) orelse return 0.0;
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

fn select_overlap_fn(l_orb: i32, l_proj: i32) ?ProjectorOverlapFn {
    if (l_orb == 0 and l_proj == 0) return compute_projector_overlap_ss;
    if (l_orb == 0 and l_proj == 1) return compute_projector_overlap_sp;
    if (l_orb == 1 and l_proj == 0) return compute_projector_overlap_ps;
    if (l_orb == 1 and l_proj == 1) return compute_projector_overlap_pp;
    return null;
}

/// s-orbital with s-projector overlap
/// <φ_s|β_s> where both have l=0
fn compute_projector_overlap_ss(
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
        const gauss_radial = gaussian_radial_integrand(orb.alpha, r, dist);

        sum += gauss_radial * r_beta * r * rab;
        pw_sum += r_beta * r * rab;
    }

    if (@abs(pw_sum) < 1e-15) return 0.0;
    return pw_projector * (sum / pw_sum);
}

/// s-orbital with p-projector overlap
/// <φ_s|β_p> - small due to angular mismatch
fn compute_projector_overlap_sp(
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
        const j1 = pw_nonlocal.spherical_bessel(1, k_eff * r);
        const gauss_radial = gaussian_radial_integrand(orb.alpha, r, dist);

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
fn compute_projector_overlap_ps(
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
        const gauss_radial = gaussian_radial_integrand(orb.alpha, r, dist);

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
fn compute_projector_overlap_pp(
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
        const j1 = pw_nonlocal.spherical_bessel(1, k_eff * r);
        const gauss_radial = gaussian_radial_integrand(orb.alpha, r, dist);

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
fn gaussian_radial_integrand(alpha: f64, r: f64, dist: f64) f64 {
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

fn invert_cell(cell: math.Mat3) !math.Mat3 {
    return local_orbital_potential.invert_cell(cell);
}

fn minimum_image_delta(
    cell: math.Mat3,
    inv_cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    delta: math.Vec3,
) math.Vec3 {
    var frac = inv_cell.mul_vec(delta);
    if (pbc.x) frac.x -= @round(frac.x);
    if (pbc.y) frac.y -= @round(frac.y);
    if (pbc.z) frac.z -= @round(frac.z);
    return cell.mul_vec(frac);
}

test "nonlocal matrix is symmetric" {
    const alloc = std.testing.allocator;

    // Simple test with 2 orbitals
    const centers = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 2.0, .y = 0.0, .z = 0.0 },
    };
    const cell = math.Mat3.from_rows(
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

    var result = try build_nonlocal_csr(alloc, centers[0..], ions[0..], cell, pbc, opts);
    defer result.deinit(alloc);

    // Check symmetry: V(0,1) should equal V(1,0)
    const v01 = result.value_at(0, 1);
    const v10 = result.value_at(1, 0);
    try std.testing.expectApproxEqAbs(v01, v10, 1e-10);
}
