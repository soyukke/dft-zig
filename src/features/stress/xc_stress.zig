const std = @import("std");
const math = @import("../math/math.zig");
const scf = @import("../scf/scf.zig");
const xc_mod = @import("../xc/xc.zig");
const stress_util = @import("stress.zig");

const Stress3x3 = stress_util.Stress3x3;
const Grid = stress_util.Grid;

/// Compute ∂ρ/∂x_α in real space by Cartesian FFT(iG_α ρ(G)).
fn compute_density_gradients_real_space(
    alloc: std.mem.Allocator,
    grid: Grid,
    fft_obj: scf.Grid,
    rho_total_g: []const math.Complex,
    n_grid: usize,
    grad_r: *[3][]f64,
) !void {
    for (0..3) |dir| {
        const grad_g = try alloc.alloc(math.Complex, n_grid);
        defer alloc.free(grad_g);

        var it_g = scf.GVecIterator.init(grid);
        while (it_g.next()) |g| {
            const gdir: f64 = switch (dir) {
                0 => g.gvec.x,
                1 => g.gvec.y,
                2 => g.gvec.z,
                else => unreachable,
            };
            // i * G_dir * ρ(G)
            const rho_val = rho_total_g[g.idx];
            grad_g[g.idx] = math.complex.init(-gdir * rho_val.i, gdir * rho_val.r);
        }

        grad_r[dir] = try scf.reciprocal_to_real(alloc, fft_obj, grad_g);
    }
}

/// Accumulate the GGA gradient-dependent stress over the real-space grid.
fn accumulate_gga_stress(
    grid: Grid,
    rho_total: []const f64,
    grad_r: [3][]f64,
    xc_func: xc_mod.Functional,
    n_grid: usize,
    inv_volume: f64,
    sigma: *Stress3x3,
) void {
    const dv = grid.volume / @as(f64, @floatFromInt(n_grid));
    for (0..n_grid) |i| {
        const rho_val = @max(rho_total[i], 1e-30);
        var g2_val: f64 = 0;
        for (0..3) |d| g2_val += grad_r[d][i] * grad_r[d][i];

        const xc_pt = xc_mod.eval_point(xc_func, rho_val, g2_val);
        const df_ds = xc_pt.df_dg2;
        if (@abs(df_ds) < 1e-30) continue;

        for (0..3) |a| {
            for (a..3) |b| {
                sigma[a][b] -= 2.0 * df_ds * grad_r[a][i] * grad_r[b][i] * dv * inv_volume;
            }
        }
    }
}

/// XC stress.
/// LDA: σ_αβ = δ_αβ (E_xc - ∫V_xc ρ dV) / Ω
/// GGA: + (2/Ω) ∫ (∂f/∂σ) (∂ρ/∂x_α)(∂ρ/∂x_β) dV  (σ here means |∇ρ|²)
pub fn xc_stress(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_r: []const f64,
    rho_core: ?[]const f64,
    e_xc: f64,
    vxc_rho: f64,
    xc_func: xc_mod.Functional,
) !Stress3x3 {
    var sigma = stress_util.zero_stress();
    const inv_volume = 1.0 / grid.volume;

    // LDA part: diagonal only
    const lda_diag = (e_xc - vxc_rho) * inv_volume;
    for (0..3) |a| sigma[a][a] = lda_diag;

    // GGA correction
    if (xc_func == .pbe) {
        // Compute total density (rho + rho_core for NLCC)
        const n_grid = grid.nx * grid.ny * grid.nz;
        const rho_total = try alloc.alloc(f64, n_grid);
        defer alloc.free(rho_total);

        for (0..n_grid) |i| {
            rho_total[i] = rho_r[i];
            if (rho_core) |rc| rho_total[i] += rc[i];
        }

        // Compute density gradients in G-space
        const fft_obj = scf.Grid{
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .cell = grid.cell,
            .recip = grid.recip,
            .volume = grid.volume,
        };
        const rho_total_g = try scf.real_to_reciprocal(alloc, fft_obj, rho_total, false);
        defer alloc.free(rho_total_g);

        // Compute gradient components: ∂ρ/∂x_α in real space
        var grad_r = [3][]f64{ undefined, undefined, undefined };
        try compute_density_gradients_real_space(
            alloc,
            grid,
            fft_obj,
            rho_total_g,
            n_grid,
            &grad_r,
        );
        defer for (0..3) |dir| alloc.free(grad_r[dir]);

        // Evaluate df/dσ at each grid point and accumulate GGA stress
        accumulate_gga_stress(grid, rho_total, grad_r, xc_func, n_grid, inv_volume, &sigma);

        // Symmetrize
        for (0..3) |a| {
            for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
        }
    }

    return sigma;
}
