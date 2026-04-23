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

const real_to_reciprocal = fft_grid.real_to_reciprocal;
const reciprocal_to_real = fft_grid.reciprocal_to_real;

const compute_xc_fields = xc_fields_mod.compute_xc_fields;
const compute_xc_fields_spin = xc_fields_mod.compute_xc_fields_spin;

fn prepare_density_for_xc(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []const f64,
    ecutrho: ?f64,
    use_rfft: bool,
) !?[]f64 {
    if (ecutrho) |ecut| {
        return try filter_density_to_ecutrho(alloc, grid, rho, ecut, use_rfft);
    }
    return null;
}

fn hartree_potential_at_g(rho_g: math.Complex, g2: f64, coulomb_r_cut: ?f64) math.Complex {
    if (coulomb_r_cut) |r_cut| {
        const g_mag = @sqrt(g2);
        const kernel = coulomb.cutoff_coulomb_kernel(g2, g_mag, r_cut);
        return math.complex.scale(rho_g, kernel);
    }
    if (g2 > 1e-12) {
        return math.complex.scale(rho_g, 8.0 * std.math.pi / g2);
    }
    return math.complex.init(0.0, 0.0);
}

fn sum_spin_density(
    alloc: std.mem.Allocator,
    rho_up: []const f64,
    rho_down: []const f64,
) ![]f64 {
    std.debug.assert(rho_up.len == rho_down.len);
    const rho_total = try alloc.alloc(f64, rho_up.len);
    for (0..rho_up.len) |i| {
        rho_total[i] = rho_up[i] + rho_down[i];
    }
    return rho_total;
}

const SpinPotentialValues = struct {
    up: []math.Complex,
    down: []math.Complex,
};

fn build_spin_potential_values(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_g: []const math.Complex,
    vxc_up_g: []const math.Complex,
    vxc_down_g: []const math.Complex,
    coulomb_r_cut: ?f64,
) !SpinPotentialValues {
    const total = grid.count();
    const values_up = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(values_up);

    const values_down = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(values_down);

    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        const vh = hartree_potential_at_g(rho_g[g.idx], g.g2, coulomb_r_cut);
        values_up[g.idx] = math.complex.add(vh, vxc_up_g[g.idx]);
        values_down[g.idx] = math.complex.add(vh, vxc_down_g[g.idx]);
    }
    return .{ .up = values_up, .down = values_down };
}

const SpinReciprocalXc = struct {
    vxc_up_g: []math.Complex,
    vxc_down_g: []math.Complex,

    fn deinit(self: SpinReciprocalXc, alloc: std.mem.Allocator) void {
        alloc.free(self.vxc_up_g);
        alloc.free(self.vxc_down_g);
    }
};

fn build_spin_reciprocal_xc(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_core: ?[]const f64,
    use_rfft: bool,
    xc_func: xc.Functional,
    vxc_r_out_up: ?*?[]f64,
    vxc_r_out_down: ?*?[]f64,
) !SpinReciprocalXc {
    const xc_fields = try compute_xc_fields_spin(
        alloc,
        grid,
        rho_up,
        rho_down,
        rho_core,
        use_rfft,
        xc_func,
    );
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

    const vxc_up_g = try real_to_reciprocal(alloc, grid, xc_fields.vxc_up, use_rfft);
    errdefer alloc.free(vxc_up_g);
    const vxc_down_g = try real_to_reciprocal(alloc, grid, xc_fields.vxc_down, use_rfft);
    errdefer alloc.free(vxc_down_g);
    return .{ .vxc_up_g = vxc_up_g, .vxc_down_g = vxc_down_g };
}

/// Build Hartree+XC potential grid.
/// If vxc_r_out is non-null, the real-space V_xc(r) is transferred to the caller
/// instead of being freed (useful for NLCC force calculation).
/// If coulomb_r_cut is non-null, the spherical cutoff Coulomb kernel is used
/// instead of the standard periodic 8π/G² kernel (for isolated/molecular systems).
pub fn build_potential_grid(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []const f64,
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
    const rho_filtered = try prepare_density_for_xc(alloc, grid, rho, ecutrho, use_rfft);
    defer if (rho_filtered) |rf| alloc.free(rf);

    const rho_for_xc = rho_filtered orelse rho;

    const rho_g = try real_to_reciprocal(alloc, grid, rho_for_xc, use_rfft);
    defer alloc.free(rho_g);

    const xc_fields = try compute_xc_fields(alloc, grid, rho_for_xc, rho_core, use_rfft, xc_func);
    defer {
        if (vxc_r_out) |out| {
            out.* = xc_fields.vxc; // Transfer ownership to caller
        } else {
            alloc.free(xc_fields.vxc);
        }
        alloc.free(xc_fields.exc);
    }

    const vxc_g = try real_to_reciprocal(alloc, grid, xc_fields.vxc, use_rfft);
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
        const vh = hartree_potential_at_g(rho_g[g.idx], g.g2, coulomb_r_cut);
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
pub fn build_potential_grid_spin(
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
    const rho_total = try sum_spin_density(alloc, rho_up, rho_down);
    defer alloc.free(rho_total);

    const rho_g = try real_to_reciprocal(alloc, grid, rho_total, use_rfft);
    defer alloc.free(rho_g);

    const reciprocal_xc = try build_spin_reciprocal_xc(
        alloc,
        grid,
        rho_up,
        rho_down,
        rho_core,
        use_rfft,
        xc_func,
        vxc_r_out_up,
        vxc_r_out_down,
    );
    defer reciprocal_xc.deinit(alloc);

    const values = try build_spin_potential_values(
        alloc,
        grid,
        rho_g,
        reciprocal_xc.vxc_up_g,
        reciprocal_xc.vxc_down_g,
        coulomb_r_cut,
    );

    return .{
        .up = .{
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .values = values.up,
        },
        .down = .{
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .values = values.down,
        },
    };
}

/// Build ionic local potential grid in reciprocal space.
/// When ecutrho is specified, V_loc(G) is zeroed for |G|² >= ecutrho to match
/// QE's spherical G-vector convention (cube corners excluded).
pub fn build_ionic_potential_grid(
    alloc: std.mem.Allocator,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
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
        values[g.idx] = try hamiltonian.ionic_local_potential_with_table(
            g.gvec,
            species,
            atoms,
            inv_volume,
            local_cfg,
            ff_tables,
        );
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
pub fn build_local_potential_real(
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
    return try reciprocal_to_real(alloc, grid, combined);
}

/// Filter real-space density to ecutrho sphere: FFT → zero |G|² >= ecutrho → IFFT.
/// Returns a newly allocated filtered density array.
pub fn filter_density_to_ecutrho(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []const f64,
    ecutrho: f64,
    use_rfft: bool,
) ![]f64 {
    const rho_g = try real_to_reciprocal(alloc, grid, rho, use_rfft);
    defer alloc.free(rho_g);

    var it = gvec_iter.GVecIterator.init(grid);
    while (it.next()) |g| {
        if (g.g2 >= ecutrho) {
            rho_g[g.idx] = math.complex.init(0.0, 0.0);
        }
    }
    return try reciprocal_to_real(alloc, grid, rho_g);
}
