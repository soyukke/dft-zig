const std = @import("std");
const apply = @import("../apply.zig");
const config = @import("../../config/config.zig");
const fermi_level = @import("../fermi_level.zig");
const fft = @import("../../fft/fft.zig");
const fft_grid = @import("../fft_grid.zig");
const grid_mod = @import("../pw_grid.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const kpoint_data = @import("data.zig");
const linalg = @import("../../linalg/linalg.zig");
const logging = @import("../logging.zig");
const math = @import("../../math/math.zig");
const paw_mod = @import("../../paw/paw.zig");
const plane_wave = @import("../../plane_wave/basis.zig");
const pw_grid_map = @import("../pw_grid_map.zig");

pub const Grid = grid_mod.Grid;
pub const KPoint = kpoint_data.KPoint;

const ApplyContext = apply.ApplyContext;
const KpointEigenData = kpoint_data.KpointEigenData;
const PwGridMap = pw_grid_map.PwGridMap;
const ScfProfile = logging.ScfProfile;
const profile_start = logging.profile_start;
const profile_add = logging.profile_add;
const fft_reciprocal_to_complex_in_place = fft_grid.fft_reciprocal_to_complex_in_place;
const fft_reciprocal_to_complex_in_place_mapped =
    fft_grid.fft_reciprocal_to_complex_in_place_mapped;

pub const DensityBuffers = struct {
    map: PwGridMap,
    recip: []math.Complex,
    real: []math.Complex,
    plan: fft.Fft3dPlan,
};

pub fn allocate_density_buffers(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    basis_gvecs: []plane_wave.GVector,
) !DensityBuffers {
    const total = grid.count();
    var map = try PwGridMap.init(alloc, basis_gvecs, grid);
    errdefer map.deinit(alloc);
    const recip = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(recip);
    const real = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(real);
    const plan = try fft.Fft3dPlan.init_with_backend(
        alloc,
        io,
        grid.nx,
        grid.ny,
        grid.nz,
        cfg.scf.fft_backend,
    );
    return .{ .map = map, .recip = recip, .real = real, .plan = plan };
}

pub fn deinit_density_buffers(alloc: std.mem.Allocator, bufs: *DensityBuffers) void {
    bufs.plan.deinit(alloc);
    bufs.map.deinit(alloc);
    alloc.free(bufs.recip);
    alloc.free(bufs.real);
    bufs.* = undefined;
}

/// Per-band band-energy, nonlocal-energy, density and (optionally) PAW rhoij accumulation.
pub const BandAccumulationInput = struct {
    io: std.Io,
    grid: Grid,
    kp: KPoint,
    basis_gvecs: []plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    nocc: usize,
    nelec: f64,
    nbands: usize,
    eig: linalg.EigenDecomp,
    vnl: ?[]math.Complex,
    apply_ctx: *?ApplyContext,
    apply_cache: ?*apply.KpointApplyCache,
    fft_index_map: ?[]const usize,
    density: *DensityBuffers,
    profile_ptr: ?*ScfProfile,
};

pub fn accumulate_band_contributions(
    alloc: std.mem.Allocator,
    input: BandAccumulationInput,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    paw_rhoij: ?*paw_mod.RhoIJ,
) !void {
    const occ_last = if (input.nocc == 0)
        0.0
    else
        input.nelec / 2.0 - @as(f64, @floatFromInt(input.nocc - 1));
    const spin_factor = 2.0;
    var band: usize = 0;
    while (band < input.nocc and band < input.nbands) : (band += 1) {
        const occ = if (band + 1 == input.nocc) occ_last else 1.0;
        if (occ <= 0.0) continue;
        const weight = input.kp.weight * occ * spin_factor;
        band_energy.* += weight * input.eig.values[band];
        try accumulate_nonlocal_energy(
            input.basis_gvecs.len,
            input.vnl,
            input.apply_ctx,
            input.eig.vectors,
            band,
            weight,
            nonlocal_energy,
        );
        const coeff_start = band * input.basis_gvecs.len;
        const coeff_end = (band + 1) * input.basis_gvecs.len;
        const coeffs = input.eig.vectors[coeff_start..coeff_end];
        try accumulate_profiled_density(
            alloc,
            input,
            coeffs,
            weight,
            rho,
        );

        if (paw_rhoij) |rij| {
            try accumulate_band_paw_rho_ij(
                alloc,
                input.apply_cache,
                input.apply_ctx,
                input.basis_gvecs,
                input.atoms,
                coeffs,
                weight,
                input.inv_volume,
                rij,
            );
        }
    }
}

fn accumulate_nonlocal_energy(
    basis_len: usize,
    vnl: ?[]math.Complex,
    apply_ctx: *?ApplyContext,
    vectors: []const math.Complex,
    band: usize,
    weight: f64,
    nonlocal_energy: *f64,
) !void {
    if (vnl) |mat| {
        const e_nl = band_nonlocal_energy(basis_len, mat, vectors, band);
        nonlocal_energy.* += weight * e_nl;
    } else if (apply_ctx.*) |*ctx| {
        if (ctx.nonlocal_ctx != null) {
            const psi = vectors[band * basis_len .. (band + 1) * basis_len];
            try apply.apply_nonlocal_potential(ctx, psi, ctx.work_vec);
            const e_nl = inner_product(basis_len, psi, ctx.work_vec).r;
            nonlocal_energy.* += weight * e_nl;
        }
    }
}

fn accumulate_profiled_density(
    alloc: std.mem.Allocator,
    input: BandAccumulationInput,
    coeffs: []const math.Complex,
    weight: f64,
    rho: []f64,
) !void {
    const density_start = if (input.profile_ptr != null) profile_start(input.io) else null;
    try accumulate_band_density_fft(
        alloc,
        input.grid,
        input.density.map,
        coeffs,
        input.density.recip,
        input.density.real,
        &input.density.plan,
        input.fft_index_map,
        weight,
        input.inv_volume,
        rho,
    );
    if (input.profile_ptr) |p| profile_add(input.io, &p.density_ns, density_start);
}

/// Accumulate rhoij contribution for a single band from either cached or fresh nonlocal context.
fn accumulate_band_paw_rho_ij(
    alloc: std.mem.Allocator,
    apply_cache: ?*apply.KpointApplyCache,
    apply_ctx: *?ApplyContext,
    basis_gvecs: []plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    coeffs_paw: []const math.Complex,
    weight_factor: f64,
    inv_volume: f64,
    rij: *paw_mod.RhoIJ,
) !void {
    if (apply_cache) |cache| {
        if (cache.nonlocal_ctx) |*nl| {
            if (nl.has_paw) {
                try apply.accumulate_paw_rho_ij(
                    alloc,
                    nl,
                    basis_gvecs,
                    atoms,
                    coeffs_paw,
                    weight_factor,
                    inv_volume,
                    rij,
                );
            }
        }
        return;
    }
    if (apply_ctx.*) |*ctx| {
        if (ctx.nonlocal_ctx) |*nl| {
            if (nl.has_paw) {
                try apply.accumulate_paw_rho_ij(
                    alloc,
                    nl,
                    basis_gvecs,
                    atoms,
                    coeffs_paw,
                    weight_factor,
                    inv_volume,
                    rij,
                );
            }
        }
    }
}

/// Compute per-band nonlocal expectation values for the smearing path.
pub fn compute_nonlocal_band_entries(
    alloc: std.mem.Allocator,
    nbands: usize,
    basis_len: usize,
    vectors: []const math.Complex,
    vnl: ?[]math.Complex,
    apply_ctx: *?ApplyContext,
) !?[]f64 {
    const entries = try alloc.alloc(f64, nbands);
    errdefer alloc.free(entries);

    @memset(entries, 0.0);
    if (vnl) |mat| {
        for (entries, 0..) |*value, band| {
            value.* = band_nonlocal_energy(basis_len, mat, vectors, band);
        }
    } else if (apply_ctx.*) |*ctx| {
        if (ctx.nonlocal_ctx != null) {
            var band: usize = 0;
            while (band < nbands) : (band += 1) {
                const psi = vectors[band * basis_len .. (band + 1) * basis_len];
                try apply.apply_nonlocal_potential(ctx, psi, ctx.work_vec);
                entries[band] = inner_product(basis_len, psi, ctx.work_vec).r;
            }
        }
    }
    return entries;
}

pub fn accumulate_kpoint_density_smearing(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    data: KpointEigenData,
    recip: math.Mat3,
    volume: f64,
    fft_index_map: ?[]const usize,
    mu: f64,
    sigma: f64,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    entropy_energy: *f64,
    profile_ptr: ?*ScfProfile,
    apply_cache: ?*apply.KpointApplyCache,
    paw_rhoij: ?*paw_mod.RhoIJ,
    atoms: ?[]const hamiltonian.AtomData,
) !void {
    return accumulate_kpoint_density_smearing_spin(
        alloc,
        io,
        cfg,
        grid,
        kp,
        data,
        recip,
        volume,
        fft_index_map,
        mu,
        sigma,
        rho,
        band_energy,
        nonlocal_energy,
        entropy_energy,
        profile_ptr,
        2.0,
        apply_cache,
        paw_rhoij,
        atoms,
    );
}

pub fn accumulate_kpoint_density_smearing_spin(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    data: KpointEigenData,
    recip: math.Mat3,
    volume: f64,
    fft_index_map: ?[]const usize,
    mu: f64,
    sigma: f64,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    entropy_energy: *f64,
    profile_ptr: ?*ScfProfile,
    spin_factor: f64,
    apply_cache: ?*apply.KpointApplyCache,
    paw_rhoij: ?*paw_mod.RhoIJ,
    atoms: ?[]const hamiltonian.AtomData,
) !void {
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kp.k_cart);
    defer basis.deinit(alloc);

    if (basis.gvecs.len != data.basis_len) return error.InvalidBasis;

    const inv_volume = 1.0 / volume;
    var buffers = try allocate_density_buffers(alloc, io, cfg, grid, basis.gvecs);
    defer deinit_density_buffers(alloc, &buffers);

    var band: usize = 0;
    while (band < data.nbands) : (band += 1) {
        try accumulate_smearing_band(
            alloc,
            io,
            cfg,
            grid,
            kp,
            data,
            basis.gvecs,
            band,
            mu,
            sigma,
            spin_factor,
            inv_volume,
            &buffers,
            fft_index_map,
            rho,
            band_energy,
            nonlocal_energy,
            entropy_energy,
            profile_ptr,
            apply_cache,
            paw_rhoij,
            atoms,
        );
    }
}

/// Process one band for the smearing-spin density accumulation.
fn accumulate_smearing_band(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    data: KpointEigenData,
    basis_gvecs: []plane_wave.GVector,
    band: usize,
    mu: f64,
    sigma: f64,
    spin_factor: f64,
    inv_volume: f64,
    buffers: *DensityBuffers,
    fft_index_map: ?[]const usize,
    rho: []f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
    entropy_energy: *f64,
    profile_ptr: ?*ScfProfile,
    apply_cache: ?*apply.KpointApplyCache,
    paw_rhoij: ?*paw_mod.RhoIJ,
    atoms: ?[]const hamiltonian.AtomData,
) !void {
    const occ = fermi_level.smearing_occ(cfg.scf.smearing, data.values[band], mu, sigma);
    const entropy_contrib = kp.weight * spin_factor * sigma * fermi_level.smearing_entropy(occ);
    entropy_energy.* += entropy_contrib;
    if (occ <= 0.0) return;
    const weight = kp.weight * occ * spin_factor;
    band_energy.* += weight * data.values[band];
    if (data.nonlocal) |entries| {
        nonlocal_energy.* += weight * entries[band];
    }
    const density_start = if (profile_ptr != null) profile_start(io) else null;
    const coeffs = data.vectors[band * basis_gvecs.len .. (band + 1) * basis_gvecs.len];
    try accumulate_band_density_fft(
        alloc,
        grid,
        buffers.map,
        coeffs,
        buffers.recip,
        buffers.real,
        &buffers.plan,
        fft_index_map,
        weight,
        inv_volume,
        rho,
    );
    if (profile_ptr) |p| profile_add(io, &p.density_ns, density_start);

    if (paw_rhoij) |rij| {
        if (apply_cache) |cache| {
            if (cache.nonlocal_ctx) |*nl| {
                if (nl.has_paw) {
                    try apply.accumulate_paw_rho_ij(
                        alloc,
                        nl,
                        basis_gvecs,
                        atoms.?,
                        coeffs,
                        weight,
                        inv_volume,
                        rij,
                    );
                }
            }
        }
    }
}

fn accumulate_band_density_fft(
    alloc: std.mem.Allocator,
    grid: Grid,
    map: PwGridMap,
    coeffs: []const math.Complex,
    work_recip: []math.Complex,
    work_real: []math.Complex,
    plan: ?*fft.Fft3dPlan,
    fft_index_map: ?[]const usize,
    weight: f64,
    inv_volume: f64,
    rho: []f64,
) !void {
    map.scatter(coeffs, work_recip);
    if (fft_index_map) |idx_map| {
        try fft_reciprocal_to_complex_in_place_mapped(
            alloc,
            grid,
            idx_map,
            work_recip,
            work_real,
            plan,
        );
    } else {
        try fft_reciprocal_to_complex_in_place(alloc, grid, work_recip, work_real, plan);
    }
    for (work_real, 0..) |psi, i| {
        const density = (psi.r * psi.r + psi.i * psi.i) * inv_volume;
        rho[i] += weight * density;
    }
}

fn band_nonlocal_energy(
    n: usize,
    vnl: []const math.Complex,
    vectors: []const math.Complex,
    band: usize,
) f64 {
    var sum = math.complex.init(0.0, 0.0);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const ci = vectors[i + band * n];
        var tmp = math.complex.init(0.0, 0.0);
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const cj = vectors[j + band * n];
            const hij = vnl[i + j * n];
            tmp = math.complex.add(tmp, math.complex.mul(hij, cj));
        }
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(ci), tmp));
    }
    return sum.r;
}

fn inner_product(n: usize, a: []const math.Complex, b: []const math.Complex) math.Complex {
    var sum = math.complex.init(0.0, 0.0);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(a[i]), b[i]));
    }
    return sum;
}
