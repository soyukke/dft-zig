const std = @import("std");
const math = @import("../math/math.zig");
const config = @import("../config/config.zig");
const ewald = @import("../ewald/ewald.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const runtime_logging = @import("../runtime/logging.zig");
const scf = @import("../scf/scf.zig");
const xc_mod = @import("../xc/xc.zig");
const paw_mod = @import("../paw/paw_tab.zig");
const model_mod = @import("../dft/model.zig");

// Sub-module imports
const kinetic_stress = @import("kinetic_stress.zig");
const hartree_stress = @import("hartree_stress.zig");
const xc_stress = @import("xc_stress.zig");
const local_stress = @import("local_stress.zig");
const nonlocal_stress = @import("nonlocal_stress.zig");
const nlcc_stress = @import("nlcc_stress.zig");
const augmentation_stress = @import("augmentation_stress.zig");

pub const Stress3x3 = [3][3]f64;

pub const StressTerms = struct {
    total: Stress3x3,
    kinetic: Stress3x3,
    hartree: Stress3x3,
    xc: Stress3x3,
    local: Stress3x3,
    nonlocal: Stress3x3,
    ewald: Stress3x3,
    psp_core: Stress3x3,
    nlcc: Stress3x3,
    augmentation: Stress3x3 = zeroStress(),
    onsite: Stress3x3 = zeroStress(),
};

pub const Grid = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    min_h: i32,
    min_k: i32,
    min_l: i32,
    cell: math.Mat3,
    recip: math.Mat3,
    volume: f64,
};

pub fn zeroStress() Stress3x3 {
    return .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
}

pub fn addStress(a: Stress3x3, b: Stress3x3) Stress3x3 {
    var r: Stress3x3 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            r[i][j] = a[i][j] + b[i][j];
        }
    }
    return r;
}

fn scaleStress(s: Stress3x3, f: f64) Stress3x3 {
    var r: Stress3x3 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            r[i][j] = s[i][j] * f;
        }
    }
    return r;
}

/// Compute ∂Y_lm(q̂)/∂q_α for real spherical harmonics.
/// Returns [3]f64 for (x, y, z) components.
pub fn dYlm_dq(l: i32, m: i32, qx: f64, qy: f64, qz: f64, q_mag: f64) [3]f64 {
    if (q_mag < 1e-15) return .{ 0, 0, 0 };
    const inv_q = 1.0 / q_mag;
    const inv_q2 = inv_q * inv_q;
    const nx = qx * inv_q;
    const ny = qy * inv_q;
    const nz = qz * inv_q;

    if (l == 0) {
        return .{ 0, 0, 0 };
    }

    if (l == 1) {
        const c = @sqrt(3.0 / (4.0 * std.math.pi));
        var dY_dn = [3]f64{ 0, 0, 0 };
        switch (m) {
            -1 => dY_dn = .{ 0, c, 0 },
            0 => dY_dn = .{ 0, 0, c },
            1 => dY_dn = .{ c, 0, 0 },
            else => {},
        }
        const n_dot_grad = nx * dY_dn[0] + ny * dY_dn[1] + nz * dY_dn[2];
        const nn = [3]f64{ nx, ny, nz };
        return .{
            (dY_dn[0] - nn[0] * n_dot_grad) * inv_q,
            (dY_dn[1] - nn[1] * n_dot_grad) * inv_q,
            (dY_dn[2] - nn[2] * n_dot_grad) * inv_q,
        };
    }

    if (l == 2) {
        const c0 = @sqrt(5.0 / (16.0 * std.math.pi));
        const c1 = @sqrt(15.0 / (4.0 * std.math.pi));
        const c2 = @sqrt(15.0 / (16.0 * std.math.pi));

        var dY_dn = [3]f64{ 0, 0, 0 };
        switch (m) {
            -2 => dY_dn = .{ 2 * c2 * ny, 2 * c2 * nx, 0 },
            -1 => dY_dn = .{ 0, c1 * nz, c1 * ny },
            0 => dY_dn = .{ 0, 0, 6 * c0 * nz },
            1 => dY_dn = .{ c1 * nz, 0, c1 * nx },
            2 => dY_dn = .{ 2 * c2 * nx, -2 * c2 * ny, 0 },
            else => {},
        }
        const n_dot_grad = nx * dY_dn[0] + ny * dY_dn[1] + nz * dY_dn[2];
        const nn = [3]f64{ nx, ny, nz };
        return .{
            (dY_dn[0] - nn[0] * n_dot_grad) * inv_q,
            (dY_dn[1] - nn[1] * n_dot_grad) * inv_q,
            (dY_dn[2] - nn[2] * n_dot_grad) * inv_q,
        };
    }

    if (l == 3) {
        return dYlm_dq_numerical(l, m, qx, qy, qz, q_mag, inv_q2);
    }

    return .{ 0, 0, 0 };
}

fn dYlm_dq_numerical(l: i32, m: i32, qx: f64, qy: f64, qz: f64, q_mag: f64, inv_q2: f64) [3]f64 {
    _ = inv_q2;
    const delta = q_mag * 1e-5;
    var result = [3]f64{ 0, 0, 0 };
    const q = [3]f64{ qx, qy, qz };
    for (0..3) |dir| {
        var qp = q;
        var qm = q;
        qp[dir] += delta;
        qm[dir] -= delta;
        const yp = nonlocal.realSphericalHarmonic(l, m, qp[0], qp[1], qp[2]);
        const ym = nonlocal.realSphericalHarmonic(l, m, qm[0], qm[1], qm[2]);
        result[dir] = (yp - ym) / (2.0 * delta);
    }
    return result;
}

fn printStress(name: []const u8, sigma: Stress3x3, _: f64) void {
    const ry_to_gpa = 14710.507; // 1 Ry/Bohr³ = 14710.507 GPa
    const pressure = -(sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0 * ry_to_gpa;
    runtime_logging.debugPrint(.info, .info, "Stress {s:12} (GPa): {d:10.4} {d:10.4} {d:10.4} / {d:10.4} {d:10.4} {d:10.4} / {d:10.4} {d:10.4} {d:10.4}  P={d:.2}\n", .{
        name,
        sigma[0][0] * ry_to_gpa,
        sigma[0][1] * ry_to_gpa,
        sigma[0][2] * ry_to_gpa,
        sigma[1][0] * ry_to_gpa,
        sigma[1][1] * ry_to_gpa,
        sigma[1][2] * ry_to_gpa,
        sigma[2][0] * ry_to_gpa,
        sigma[2][1] * ry_to_gpa,
        sigma[2][2] * ry_to_gpa,
        pressure,
    });
}

/// Convert stress tensor to pressure in GPa.
pub fn pressureGPa(sigma: Stress3x3) f64 {
    const ry_to_gpa = 14710.507;
    return -(sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0 * ry_to_gpa;
}

pub fn symmetrizeStress(sigma: Stress3x3, sym_ops: []const @import("../symmetry/symmetry.zig").SymOp, cell: math.Mat3) Stress3x3 {
    if (sym_ops.len == 0) return sigma;

    const c = cell.m;
    const det_c = c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) -
        c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0]) +
        c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
    const inv_det = 1.0 / det_c;

    var c_inv: [3][3]f64 = undefined;
    c_inv[0][0] = (c[1][1] * c[2][2] - c[1][2] * c[2][1]) * inv_det;
    c_inv[0][1] = (c[0][2] * c[2][1] - c[0][1] * c[2][2]) * inv_det;
    c_inv[0][2] = (c[0][1] * c[1][2] - c[0][2] * c[1][1]) * inv_det;
    c_inv[1][0] = (c[1][2] * c[2][0] - c[1][0] * c[2][2]) * inv_det;
    c_inv[1][1] = (c[0][0] * c[2][2] - c[0][2] * c[2][0]) * inv_det;
    c_inv[1][2] = (c[0][2] * c[1][0] - c[0][0] * c[1][2]) * inv_det;
    c_inv[2][0] = (c[1][0] * c[2][1] - c[1][1] * c[2][0]) * inv_det;
    c_inv[2][1] = (c[0][1] * c[2][0] - c[0][0] * c[2][1]) * inv_det;
    c_inv[2][2] = (c[0][0] * c[1][1] - c[0][1] * c[1][0]) * inv_det;

    var result = zeroStress();
    for (sym_ops) |op| {
        var ct_rot: [3][3]f64 = undefined;
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0;
                for (0..3) |k| {
                    s += c[k][i] * @as(f64, @floatFromInt(op.rot.m[k][j]));
                }
                ct_rot[i][j] = s;
            }
        }

        var r_cart: [3][3]f64 = undefined;
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0;
                for (0..3) |k| {
                    s += ct_rot[i][k] * c_inv[j][k];
                }
                r_cart[i][j] = s;
            }
        }

        var rs: [3][3]f64 = undefined;
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0;
                for (0..3) |k| {
                    s += r_cart[i][k] * sigma[k][j];
                }
                rs[i][j] = s;
            }
        }
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0;
                for (0..3) |k| {
                    s += rs[i][k] * r_cart[j][k];
                }
                result[i][j] += s;
            }
        }
    }

    const inv_n = 1.0 / @as(f64, @floatFromInt(sym_ops.len));
    for (0..3) |i| {
        for (0..3) |j| {
            result[i][j] *= inv_n;
        }
    }
    return result;
}

/// Convert FFT index to frequency.
fn indexToFreq(i: usize, n: usize) i32 {
    const half = (n - 1) / 2;
    return if (i <= half) @as(i32, @intCast(i)) else @as(i32, @intCast(i)) - @as(i32, @intCast(n));
}

/// FFT real-space density to G-space (reordered to grid layout).
fn densityToReciprocal(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    density: []const f64,
    fft_backend: config.FftBackend,
) ![]math.Complex {
    const fft = @import("../fft/fft.zig");
    const nx = grid.nx;
    const ny = grid.ny;
    const nz = grid.nz;
    const total = nx * ny * nz;

    if (density.len != total) return error.InvalidDensitySize;

    var data = try alloc.alloc(math.Complex, total);
    defer alloc.free(data);
    for (density, 0..) |d, i| {
        data[i] = math.complex.init(d, 0.0);
    }

    var plan = try fft.Fft3dPlan.initWithBackend(alloc, io, nx, ny, nz, fft_backend);
    defer plan.deinit(alloc);
    plan.forward(data);

    const scale = 1.0 / @as(f64, @floatFromInt(total));
    var out = try alloc.alloc(math.Complex, total);

    var idx: usize = 0;
    var z: usize = 0;
    while (z < nz) : (z += 1) {
        var y: usize = 0;
        while (y < ny) : (y += 1) {
            var x: usize = 0;
            while (x < nx) : (x += 1) {
                const fh = indexToFreq(x, nx);
                const fk = indexToFreq(y, ny);
                const fl = indexToFreq(z, nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                const out_idx = th + nx * (tk + ny * tl);
                out[out_idx] = math.complex.init(data[idx].r * scale, data[idx].i * scale);
                idx += 1;
            }
        }
    }

    return out;
}

pub fn computeStress(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_g: []const math.Complex,
    rho_r: []const f64,
    rho_core: ?[]const f64,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ewald_cfg: config.EwaldConfig,
    wavefunctions: ?scf.WavefunctionData,
    energy: anytype,
    xc_func: xc_mod.Functional,
    local_cfg: local_potential.LocalPotentialConfig,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    radial_tables: ?[]nonlocal.RadialTableSet,
    rho_aug: ?[]const f64,
    quiet: bool,
    paw_dij: ?[]const []const f64,
    paw_dij_m: ?[]const []const f64,
    paw_rhoij: ?[]const []const f64,
    paw_tabs: ?[]const paw_mod.PawTab,
    potential_values: ?[]const math.Complex,
    ecutrho: f64,
    wavefunctions_down: ?scf.WavefunctionData,
) !StressTerms {
    const volume = grid.volume;
    const inv_volume = 1.0 / volume;

    // Ewald stress (Hartree → Ry: ×2)
    var charges = try alloc.alloc(f64, atoms.len);
    defer alloc.free(charges);
    var positions = try alloc.alloc(math.Vec3, atoms.len);
    defer alloc.free(positions);
    for (atoms, 0..) |atom, idx| {
        charges[idx] = species[atom.species_index].z_valence;
        positions[idx] = atom.position;
    }
    const ew_params = ewald.Params{
        .alpha = ewald_cfg.alpha,
        .rcut = ewald_cfg.rcut,
        .gcut = ewald_cfg.gcut,
        .tol = ewald_cfg.tol,
        .quiet = true,
    };
    const sigma_ewald = scaleStress(try ewald.ionIonStress(grid.cell, grid.recip, charges, positions, ew_params), 2.0);

    // Kinetic stress (spin-polarized: sum contributions from both channels)
    const is_spin = wavefunctions_down != null;
    const sf: f64 = if (is_spin) 1.0 else 2.0;
    var sigma_kin = try kinetic_stress.kineticStress(alloc, wavefunctions, grid.recip, inv_volume, sf);
    if (wavefunctions_down) |wf_down| {
        const sigma_kin_down = try kinetic_stress.kineticStress(alloc, wf_down, grid.recip, inv_volume, 1.0);
        sigma_kin = addStress(sigma_kin, sigma_kin_down);
    }

    // Hartree stress: use augmented density ρ̃+n̂ for PAW
    const rho_g_for_eh = if (rho_aug) |aug| blk: {
        const fft_obj = scf.Grid{
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .cell = grid.cell,
            .recip = grid.recip,
            .volume = volume,
        };
        break :blk try scf.realToReciprocal(alloc, fft_obj, aug, false);
    } else null;
    defer if (rho_g_for_eh) |g| alloc.free(g);
    const sigma_hartree = hartree_stress.hartreeStress(grid, rho_g_for_eh orelse rho_g, energy.hartree, inv_volume, ecutrho);

    // XC stress
    const rho_for_xc = rho_aug orelse rho_r;
    const sigma_xc = try xc_stress.xcStress(alloc, grid, rho_for_xc, rho_core, energy.xc, energy.vxc_rho, xc_func);

    // Local pseudopotential stress
    const rho_g_for_loc = if (rho_aug) |aug| blk: {
        const fft_obj = scf.Grid{
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .cell = grid.cell,
            .recip = grid.recip,
            .volume = volume,
        };
        break :blk try scf.realToReciprocal(alloc, fft_obj, aug, false);
    } else null;
    defer if (rho_g_for_loc) |g| alloc.free(g);
    const sigma_local = local_stress.localStress(grid, rho_g_for_loc orelse rho_g, species, atoms, local_cfg, ff_tables, inv_volume, ecutrho);

    // Nonlocal stress
    var sigma_nonlocal = try nonlocal_stress.nonlocalStress(alloc, wavefunctions, species, atoms, grid.recip, volume, radial_tables, paw_dij, paw_dij_m, paw_tabs, sf);
    if (wavefunctions_down) |wf_down| {
        const sigma_nl_down = try nonlocal_stress.nonlocalStress(alloc, wf_down, species, atoms, grid.recip, volume, radial_tables, paw_dij, paw_dij_m, paw_tabs, 1.0);
        sigma_nonlocal = addStress(sigma_nonlocal, sigma_nl_down);
    }

    // PSP core stress: σ = -(E_psp/Ω) δ_αβ
    var sigma_psp = zeroStress();
    const psp_core_energy = energy.psp_core;
    for (0..3) |a| sigma_psp[a][a] = -psp_core_energy * inv_volume;

    // NLCC stress
    const sigma_nlcc = try nlcc_stress.nlccStress(alloc, grid, rho_for_xc, rho_core, species, atoms, rho_core_tables, xc_func, ecutrho);

    // PAW augmentation stress
    const sigma_aug = try augmentation_stress.augmentationStress(alloc, grid, potential_values, paw_rhoij, paw_tabs, atoms, inv_volume, ecutrho);

    const sigma_onsite = zeroStress();

    // Total
    var sigma_total = zeroStress();
    const terms = [_]Stress3x3{ sigma_ewald, sigma_kin, sigma_hartree, sigma_xc, sigma_local, sigma_nonlocal, sigma_psp, sigma_nlcc, sigma_aug, sigma_onsite };
    for (terms) |t| {
        sigma_total = addStress(sigma_total, t);
    }

    if (!quiet) {
        printStress("Ewald", sigma_ewald, volume);
        printStress("Kinetic", sigma_kin, volume);
        printStress("Hartree", sigma_hartree, volume);
        printStress("XC", sigma_xc, volume);
        printStress("Local", sigma_local, volume);
        printStress("Nonlocal", sigma_nonlocal, volume);
        printStress("PSP core", sigma_psp, volume);
        printStress("NLCC", sigma_nlcc, volume);
        if (paw_tabs != null) {
            printStress("Augment", sigma_aug, volume);
            printStress("On-site", sigma_onsite, volume);
        }
        printStress("TOTAL", sigma_total, volume);
    }

    return StressTerms{
        .total = sigma_total,
        .kinetic = sigma_kin,
        .hartree = sigma_hartree,
        .xc = sigma_xc,
        .local = sigma_local,
        .nonlocal = sigma_nonlocal,
        .ewald = sigma_ewald,
        .psp_core = sigma_psp,
        .nlcc = sigma_nlcc,
        .augmentation = sigma_aug,
        .onsite = sigma_onsite,
    };
}

/// High-level stress computation from SCF result.
/// Builds required form factor and radial tables internally.
pub fn computeStressFromScf(
    alloc: std.mem.Allocator,
    io: std.Io,
    scf_result: *const scf.ScfResult,
    cfg: config.Config,
    model: *const model_mod.Model,
) !StressTerms {
    const species = model.species;
    const atoms = model.atoms;
    const grid = Grid{
        .nx = scf_result.grid.nx,
        .ny = scf_result.grid.ny,
        .nz = scf_result.grid.nz,
        .min_h = scf_result.grid.min_h,
        .min_k = scf_result.grid.min_k,
        .min_l = scf_result.grid.min_l,
        .cell = scf_result.grid.cell,
        .recip = scf_result.grid.recip,
        .volume = scf_result.grid.volume,
    };

    // FFT density to G-space
    const rho_g = try densityToReciprocal(alloc, io, grid, scf_result.density, cfg.scf.fft_backend);
    defer alloc.free(rho_g);

    // Build form factor tables
    const ff_q_max = 2.0 * @sqrt(cfg.scf.ecut_ry) + 1.0;
    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, scf_result.grid.cell);
    var ff_tables_buf = try alloc.alloc(form_factor.LocalFormFactorTable, species.len);
    for (species, 0..) |entry, si| {
        ff_tables_buf[si] = try form_factor.LocalFormFactorTable.init(alloc, entry.upf.*, entry.z_valence, local_cfg, ff_q_max);
    }
    defer {
        for (ff_tables_buf) |*t| t.deinit(alloc);
        alloc.free(ff_tables_buf);
    }

    // Build rho_core tables
    var rho_core_tables_buf = try alloc.alloc(form_factor.RadialFormFactorTable, species.len);
    for (species, 0..) |entry, si| {
        rho_core_tables_buf[si] = try form_factor.RadialFormFactorTable.initRhoCore(alloc, entry.upf.*, ff_q_max);
    }
    defer {
        for (rho_core_tables_buf) |*t| t.deinit(alloc);
        alloc.free(rho_core_tables_buf);
    }

    // Build radial tables for nonlocal projectors
    var radial_tables_buf = try alloc.alloc(nonlocal.RadialTableSet, species.len);
    for (species, 0..) |entry, si| {
        radial_tables_buf[si] = try nonlocal.RadialTableSet.init(alloc, entry.upf.beta, entry.upf.r, entry.upf.rab, ff_q_max);
    }
    defer {
        for (radial_tables_buf) |*t| t.deinit(alloc);
        alloc.free(radial_tables_buf);
    }

    const gs = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    const ecutrho = cfg.scf.ecut_ry * gs * gs;

    // Build augmented density (ρ̃ + n̂) for PAW
    const is_paw = scf_result.paw_tabs != null and scf_result.paw_rhoij != null;
    var rho_aug_buf: ?[]f64 = null;
    if (is_paw) {
        rho_aug_buf = try alloc.alloc(f64, scf_result.density.len);
        @memcpy(rho_aug_buf.?, scf_result.density);
        try augmentation_stress.buildAugmentedDensity(alloc, grid, rho_aug_buf.?, scf_result.paw_rhoij.?, scf_result.paw_tabs.?, atoms, ecutrho);
    }
    defer if (rho_aug_buf) |buf| alloc.free(buf);

    // Get V_eff(G) for PAW augmentation stress
    var pot_vals_buf: ?[]math.Complex = null;
    if (is_paw) {
        const n = scf_result.potential.values.len;
        pot_vals_buf = try alloc.alloc(math.Complex, n);
        for (0..n) |gi| {
            const v_hxc = scf_result.potential.values[gi];
            const v_loc = if (scf_result.ionic_g) |ig| ig[gi] else math.complex.init(0.0, 0.0);
            pot_vals_buf.?[gi] = math.complex.add(v_hxc, v_loc);
        }
    }
    defer if (pot_vals_buf) |buf| alloc.free(buf);
    const pot_vals: ?[]const math.Complex = if (pot_vals_buf) |buf| buf else null;

    var stress_terms = try computeStress(
        alloc,
        grid,
        rho_g,
        scf_result.density,
        scf_result.rho_core,
        species,
        atoms,
        cfg.ewald,
        scf_result.wavefunctions,
        scf_result.energy,
        cfg.scf.xc,
        local_cfg,
        ff_tables_buf,
        rho_core_tables_buf,
        radial_tables_buf,
        if (rho_aug_buf) |buf| buf else null,
        cfg.scf.quiet,
        scf_result.paw_dij,
        scf_result.paw_dij_m,
        scf_result.paw_rhoij,
        scf_result.paw_tabs,
        pot_vals,
        ecutrho,
        scf_result.wavefunctions_down,
    );

    // Symmetrize stress using crystal symmetry operations
    if (cfg.scf.symmetry) {
        const symmetry_import = @import("../symmetry/symmetry.zig");
        const sym_ops = try symmetry_import.getSymmetryOps(alloc, scf_result.grid.cell, atoms, 1e-5);
        defer alloc.free(sym_ops);
        if (sym_ops.len > 1) {
            const cell_mat = scf_result.grid.cell;
            stress_terms.kinetic = symmetrizeStress(stress_terms.kinetic, sym_ops, cell_mat);
            stress_terms.hartree = symmetrizeStress(stress_terms.hartree, sym_ops, cell_mat);
            stress_terms.xc = symmetrizeStress(stress_terms.xc, sym_ops, cell_mat);
            stress_terms.local = symmetrizeStress(stress_terms.local, sym_ops, cell_mat);
            stress_terms.nonlocal = symmetrizeStress(stress_terms.nonlocal, sym_ops, cell_mat);
            stress_terms.ewald = symmetrizeStress(stress_terms.ewald, sym_ops, cell_mat);
            stress_terms.psp_core = symmetrizeStress(stress_terms.psp_core, sym_ops, cell_mat);
            stress_terms.nlcc = symmetrizeStress(stress_terms.nlcc, sym_ops, cell_mat);
            stress_terms.augmentation = symmetrizeStress(stress_terms.augmentation, sym_ops, cell_mat);
            stress_terms.onsite = symmetrizeStress(stress_terms.onsite, sym_ops, cell_mat);
            stress_terms.total = zeroStress();
            const all = [_]Stress3x3{ stress_terms.ewald, stress_terms.kinetic, stress_terms.hartree, stress_terms.xc, stress_terms.local, stress_terms.nonlocal, stress_terms.psp_core, stress_terms.nlcc, stress_terms.augmentation, stress_terms.onsite };
            for (all) |t| {
                stress_terms.total = addStress(stress_terms.total, t);
            }
            if (!cfg.scf.quiet) {
                printStress("TOTAL (sym)", stress_terms.total, scf_result.grid.volume);
            }
        }
    }

    return stress_terms;
}

test "ewald stress finite difference" {
    const io = std.testing.io;
    const testing = std.testing;

    const a = 10.2;
    const cell = math.Mat3{ .m = .{
        .{ 0.0, a / 2.0, a / 2.0 },
        .{ a / 2.0, 0.0, a / 2.0 },
        .{ a / 2.0, a / 2.0, 0.0 },
    } };
    const recip_lat = math.reciprocal(cell);

    const charges = [_]f64{ 4.0, 4.0 };
    const positions = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = a / 4.0, .y = a / 4.0, .z = a / 4.0 },
    };

    const params = ewald.Params{ .alpha = 0.0, .rcut = 0.0, .gcut = 0.0, .tol = 1e-10, .quiet = true };

    const e0 = try ewald.ionIonEnergy(io, cell, recip_lat, &charges, &positions, params);
    const sigma = try ewald.ionIonStress(cell, recip_lat, &charges, &positions, params);

    const frac = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.25, .y = 0.25, .z = 0.25 },
    };

    const vol0 = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));

    const delta = 1e-5;
    for (0..3) |al| {
        for (al..3) |be| {
            var cell_p = cell;
            var cell_m = cell;
            for (0..3) |i| {
                cell_p.m[i][al] += delta * cell.m[i][be];
                cell_m.m[i][al] -= delta * cell.m[i][be];
                if (al != be) {
                    cell_p.m[i][be] += delta * cell.m[i][al];
                    cell_m.m[i][be] -= delta * cell.m[i][al];
                }
            }
            const recip_p = math.reciprocal(cell_p);
            const recip_m = math.reciprocal(cell_m);

            var pos_p: [2]math.Vec3 = undefined;
            var pos_m: [2]math.Vec3 = undefined;
            for (0..2) |idx| {
                pos_p[idx] = math.Vec3{
                    .x = frac[idx].x * cell_p.m[0][0] + frac[idx].y * cell_p.m[1][0] + frac[idx].z * cell_p.m[2][0],
                    .y = frac[idx].x * cell_p.m[0][1] + frac[idx].y * cell_p.m[1][1] + frac[idx].z * cell_p.m[2][1],
                    .z = frac[idx].x * cell_p.m[0][2] + frac[idx].y * cell_p.m[1][2] + frac[idx].z * cell_p.m[2][2],
                };
                pos_m[idx] = math.Vec3{
                    .x = frac[idx].x * cell_m.m[0][0] + frac[idx].y * cell_m.m[1][0] + frac[idx].z * cell_m.m[2][0],
                    .y = frac[idx].x * cell_m.m[0][1] + frac[idx].y * cell_m.m[1][1] + frac[idx].z * cell_m.m[2][1],
                    .z = frac[idx].x * cell_m.m[0][2] + frac[idx].y * cell_m.m[1][2] + frac[idx].z * cell_m.m[2][2],
                };
            }

            const e_p = try ewald.ionIonEnergy(io, cell_p, recip_p, &charges, &pos_p, params);
            const e_m = try ewald.ionIonEnergy(io, cell_m, recip_m, &charges, &pos_m, params);

            const fd_factor: f64 = if (al == be) 1.0 else 2.0;
            const sigma_fd = (e_p - e_m) / (2.0 * delta * fd_factor) / vol0;

            try testing.expectApproxEqAbs(sigma[al][be], sigma_fd, @max(@abs(sigma_fd) * 1e-3, 1e-6));
        }
    }
    _ = e0;
}
