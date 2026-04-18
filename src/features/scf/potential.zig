const std = @import("std");
const coulomb = @import("../coulomb/coulomb.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const gvec_iter = @import("gvec_iter.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const math = @import("../math/math.zig");
const xc_fields_mod = @import("xc_fields.zig");
const xc = @import("../xc/xc.zig");

const Grid = grid_mod.Grid;

const realToReciprocal = fft_grid.realToReciprocal;
const reciprocalToReal = fft_grid.reciprocalToReal;

const computeXcFields = xc_fields_mod.computeXcFields;
const computeXcFieldsSpin = xc_fields_mod.computeXcFieldsSpin;

/// Build Hartree+XC potential grid.
/// If vxc_r_out is non-null, the real-space V_xc(r) is transferred to the caller
/// instead of being freed (useful for NLCC force calculation).
/// If coulomb_r_cut is non-null, the spherical cutoff Coulomb kernel is used
/// instead of the standard periodic 8π/G² kernel (for isolated/molecular systems).
pub fn buildPotentialGrid(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    rho_core: ?[]const f64,
    use_rfft: bool,
    xc_func: xc.Functional,
    vxc_r_out: ?*?[]f64,
    coulomb_r_cut: ?f64,
    ecutrho: ?f64,
) !hamiltonian.PotentialGrid {
    // When ecutrho is set (PAW), filter the density to the ecutrho sphere
    // before computing V_xc. This matches QE's convention where all G-space
    // operations are limited to |G|² < ecutrho.
    var rho_filtered: ?[]f64 = null;
    defer if (rho_filtered) |rf| alloc.free(rf);
    if (ecutrho) |ecut| {
        rho_filtered = try filterDensityToEcutrho(alloc, grid, rho, ecut, use_rfft);
    }
    const rho_for_xc = rho_filtered orelse rho;

    const rho_g = try realToReciprocal(alloc, grid, rho_for_xc, use_rfft);
    defer alloc.free(rho_g);

    const xc_fields = try computeXcFields(alloc, grid, rho_for_xc, rho_core, use_rfft, xc_func);
    defer {
        if (vxc_r_out) |out| {
            out.* = xc_fields.vxc; // Transfer ownership to caller
        } else {
            alloc.free(xc_fields.vxc);
        }
        alloc.free(xc_fields.exc);
    }
    const vxc_g = try realToReciprocal(alloc, grid, xc_fields.vxc, use_rfft);
    defer alloc.free(vxc_g);

    const total = grid.count();
    const values = try alloc.alloc(math.Complex, total);
    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        // ecutrho spherical cutoff: zero V_H(G) and V_xc(G) for |G|² >= ecutrho
        // to match QE's G-sphere convention (all G-space ops limited to sphere)
        const beyond_ecutrho = if (ecutrho) |ecut| g.g2 >= ecut else false;
        if (beyond_ecutrho) {
            values[g.idx] = math.complex.init(0.0, 0.0);
            continue;
        }
        var vh = math.complex.init(0.0, 0.0);
        if (coulomb_r_cut) |r_cut| {
            // Isolated system: cutoff Coulomb kernel
            const g_mag = @sqrt(g.g2);
            const kernel = coulomb.cutoffCoulombKernel(g.g2, g_mag, r_cut);
            vh = math.complex.scale(rho_g[g.idx], kernel);
        } else {
            if (g.g2 > 1e-12) {
                // Periodic: Hartree potential in Rydberg units: V_H(G) = 8πρ(G)/G²
                // (factor 2 compared to Hartree units for consistency with
                // kinetic energy |k+G|² and UPF pseudopotentials in Rydberg)
                vh = math.complex.scale(rho_g[g.idx], 8.0 * std.math.pi / g.g2);
            }
        }
        values[g.idx] = math.complex.add(vh, vxc_g[g.idx]);
    }

    return hamiltonian.PotentialGrid{
        .nx = grid.nx,
        .ny = grid.ny,
        .nz = grid.nz,
        .min_h = grid.min_h,
        .min_k = grid.min_k,
        .min_l = grid.min_l,
        .values = values,
    };
}

pub const SpinPotentialGrids = struct {
    up: hamiltonian.PotentialGrid,
    down: hamiltonian.PotentialGrid,
};

/// Build spin-polarized Hartree+XC potential grids.
/// Hartree is computed from rho_total = rho_up + rho_down (same for both channels).
/// XC potentials differ: V_H + V_xc_up, V_H + V_xc_down.
/// If coulomb_r_cut is non-null, the spherical cutoff Coulomb kernel is used.
pub fn buildPotentialGridSpin(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_core: ?[]const f64,
    use_rfft: bool,
    xc_func: xc.Functional,
    vxc_r_out_up: ?*?[]f64,
    vxc_r_out_down: ?*?[]f64,
    coulomb_r_cut: ?f64,
) !SpinPotentialGrids {
    const total = grid.count();

    // Compute rho_total for Hartree
    const rho_total = try alloc.alloc(f64, total);
    defer alloc.free(rho_total);
    for (0..total) |i| {
        rho_total[i] = rho_up[i] + rho_down[i];
    }
    const rho_g = try realToReciprocal(alloc, grid, rho_total, use_rfft);
    defer alloc.free(rho_g);

    // Compute spin XC fields
    const xc_fields = try computeXcFieldsSpin(alloc, grid, rho_up, rho_down, rho_core, use_rfft, xc_func);
    defer {
        if (vxc_r_out_up) |out| {
            out.* = xc_fields.vxc_up;
        } else {
            alloc.free(xc_fields.vxc_up);
        }
        if (vxc_r_out_down) |out| {
            out.* = xc_fields.vxc_down;
        } else {
            alloc.free(xc_fields.vxc_down);
        }
        alloc.free(xc_fields.exc);
    }

    const vxc_up_g = try realToReciprocal(alloc, grid, xc_fields.vxc_up, use_rfft);
    defer alloc.free(vxc_up_g);
    const vxc_down_g = try realToReciprocal(alloc, grid, xc_fields.vxc_down, use_rfft);
    defer alloc.free(vxc_down_g);

    // Build V_H(G) + V_xc_up(G) and V_H(G) + V_xc_down(G)
    const values_up = try alloc.alloc(math.Complex, total);
    const values_down = try alloc.alloc(math.Complex, total);
    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        var vh = math.complex.init(0.0, 0.0);
        if (coulomb_r_cut) |r_cut| {
            const g_mag = @sqrt(g.g2);
            const kernel = coulomb.cutoffCoulombKernel(g.g2, g_mag, r_cut);
            vh = math.complex.scale(rho_g[g.idx], kernel);
        } else {
            if (g.g2 > 1e-12) {
                vh = math.complex.scale(rho_g[g.idx], 8.0 * std.math.pi / g.g2);
            }
        }
        values_up[g.idx] = math.complex.add(vh, vxc_up_g[g.idx]);
        values_down[g.idx] = math.complex.add(vh, vxc_down_g[g.idx]);
    }

    return .{
        .up = .{
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .values = values_up,
        },
        .down = .{
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .values = values_down,
        },
    };
}

/// Build ionic local potential grid in reciprocal space.
/// When ecutrho is specified, V_loc(G) is zeroed for |G|² >= ecutrho to match
/// QE's spherical G-vector convention (cube corners excluded).
pub fn buildIonicPotentialGrid(
    alloc: std.mem.Allocator,
    grid: Grid,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    local_cfg: local_potential.LocalPotentialConfig,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    ecutrho: ?f64,
) !hamiltonian.PotentialGrid {
    const total = grid.count();
    const values = try alloc.alloc(math.Complex, total);
    const inv_volume = 1.0 / grid.volume;
    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (ecutrho) |ecut| {
            if (g.g2 >= ecut) {
                values[g.idx] = math.complex.init(0.0, 0.0);
                continue;
            }
        }
        values[g.idx] = try hamiltonian.ionicLocalPotentialWithTable(g.gvec, species, atoms, inv_volume, local_cfg, ff_tables);
    }
    return hamiltonian.PotentialGrid{
        .nx = grid.nx,
        .ny = grid.ny,
        .nz = grid.nz,
        .min_h = grid.min_h,
        .min_k = grid.min_k,
        .min_l = grid.min_l,
        .values = values,
    };
}

/// Build total local potential in real space.
pub fn buildLocalPotentialReal(
    alloc: std.mem.Allocator,
    grid: Grid,
    ionic: hamiltonian.PotentialGrid,
    extra: hamiltonian.PotentialGrid,
) ![]f64 {
    const total = grid.count();
    const combined = try alloc.alloc(math.Complex, total);
    defer alloc.free(combined);
    for (combined, 0..) |*v, i| {
        v.* = math.complex.add(ionic.values[i], extra.values[i]);
    }
    return try reciprocalToReal(alloc, grid, combined);
}

/// Filter real-space density to ecutrho sphere: FFT → zero |G|² >= ecutrho → IFFT.
/// Returns a newly allocated filtered density array.
pub fn filterDensityToEcutrho(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []const f64,
    ecutrho: f64,
    use_rfft: bool,
) ![]f64 {
    const rho_g = try realToReciprocal(alloc, grid, rho, use_rfft);
    defer alloc.free(rho_g);

    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.g2 >= ecutrho) {
            rho_g[g.idx] = math.complex.init(0.0, 0.0);
        }
    }
    return try reciprocalToReal(alloc, grid, rho_g);
}
