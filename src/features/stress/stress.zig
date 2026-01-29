const std = @import("std");
const math = @import("../math/math.zig");
const config = @import("../config/config.zig");
const ewald = @import("../ewald/ewald.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const scf = @import("../scf/scf.zig");
const xc_mod = @import("../xc/xc.zig");
const xc_fields_mod = @import("../scf/xc_fields.zig");
const fft_grid = @import("../scf/fft_grid.zig");
const grid_mod = @import("../scf/grid.zig");
const paw_mod = @import("../paw/paw_tab.zig");

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

const Grid = struct {
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

fn zeroStress() Stress3x3 {
    return .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
}

fn addStress(a: Stress3x3, b: Stress3x3) Stress3x3 {
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

/// Compute the full stress tensor (Ry/Bohr³).
/// σ_αβ = (1/Ω) ∂E_total/∂ε_αβ
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
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    radial_tables: ?[]nonlocal.RadialTableSet,
    rho_aug: ?[]const f64,
    quiet: bool,
    // PAW-specific parameters
    paw_dij: ?[]const []const f64,
    paw_dij_m: ?[]const []const f64,
    paw_rhoij: ?[]const []const f64,
    paw_tabs: ?[]const paw_mod.PawTab,
    potential_values: ?[]const math.Complex,
    // Spherical G-vector cutoff (ecutrho in Ry = Bohr⁻²)
    ecutrho: f64,
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

    // Kinetic stress
    const sigma_kin = try kineticStress(alloc, wavefunctions, grid.recip, inv_volume);

    // Hartree stress: use augmented density ρ̃+n̂ for PAW (consistent with E_H).
    // The n̂ response to strain cancels between E_H and double-counting terms;
    // the net augmentation contribution is captured by augmentationStress.
    const rho_g_for_eh = if (rho_aug) |aug| blk: {
        const fft_obj = grid_mod.Grid{
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
        break :blk try fft_grid.realToReciprocal(alloc, fft_obj, aug, false);
    } else null;
    defer if (rho_g_for_eh) |g| alloc.free(g);
    const sigma_hartree = hartreeStress(grid, rho_g_for_eh orelse rho_g, energy.hartree, inv_volume, ecutrho);

    // XC stress: use augmented density for PAW (consistent with E_xc).
    const rho_for_xc = rho_aug orelse rho_r;
    const sigma_xc = try xcStress(alloc, grid, rho_for_xc, rho_core, energy.xc, energy.vxc_rho, xc_func);

    // Local pseudopotential stress: for PAW, use augmented density (ρ̃+n̂) since V_loc acts on full density
    // (matching QE's stres_loc which uses rho%of_r that includes augmentation charges).
    const rho_g_for_loc = if (rho_aug) |aug| blk: {
        const fft_obj = grid_mod.Grid{
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
        break :blk try fft_grid.realToReciprocal(alloc, fft_obj, aug, false);
    } else null;
    defer if (rho_g_for_loc) |g| alloc.free(g);
    const sigma_local = localStress(grid, rho_g_for_loc orelse rho_g, species, atoms, ff_tables, inv_volume, ecutrho);

    // PAW nonlocal stress: uses D_full with overlap correction.
    // For PAW, the generalized eigenvalue problem H|ψ⟩ = ε S|ψ⟩ gives stress:
    //   σ_NL = Σ f (D_full - ε_nk q_ij) × ∂<β|ψ>/∂ε
    // No separate on-site stress is needed: the correct PAW energy has
    // -Σ D^xc ρ + E_paw, and d(-Σ D^xc ρ + E_paw)/dε = 0 because
    // ∂E_paw/∂ρ_ij = D^xc. The D^xc contribution in D_full is sufficient.
    const sigma_nonlocal = try nonlocalStress(alloc, wavefunctions, species, atoms, grid.recip, volume, radial_tables, paw_dij, paw_dij_m, paw_tabs);

    // PSP core stress: σ = -(E_psp/Ω) δ_αβ
    var sigma_psp = zeroStress();
    const psp_core_energy = energy.psp_core;
    for (0..3) |a| sigma_psp[a][a] = -psp_core_energy * inv_volume;

    // NLCC stress
    const sigma_nlcc = try nlccStress(alloc, grid, rho_for_xc, rho_core, species, atoms, rho_core_tables, xc_func, ecutrho);

    // PAW augmentation stress: off-diagonal from dQ/d|G| only.
    // No volume diagonal — matching QE's addusstress which has no diagonal term.
    // The volume dependence of n̂ is captured by computing evloc with ρ_aug in localStress.
    const sigma_aug = try augmentationStress(alloc, grid, potential_values, paw_rhoij, paw_tabs, atoms, inv_volume, ecutrho);

    // No PAW on-site stress needed: E_paw includes E_H_onsite + E_xc_onsite,
    // and D_full includes D^H + D^xc, with double-counting -Σ(D^xc+D^H)ρ.
    // d(-Σ(D^xc+D^H)ρ + E_paw)/dε = 0 because ∂E_paw/∂ρ_ij = D^xc + D^H.
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
    scf_result: *const scf.ScfResult,
    cfg: config.Config,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) !StressTerms {
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
    const rho_g = try densityToReciprocal(alloc, grid, scf_result.density, cfg.scf.fft_backend);
    defer alloc.free(rho_g);

    // Build form factor tables
    const ff_q_max = 2.0 * @sqrt(cfg.scf.ecut_ry) + 1.0;
    // Get Ewald alpha for form factor
    const ew_alpha = if (cfg.ewald.alpha > 0.0) cfg.ewald.alpha else blk: {
        const cell_mat = scf_result.grid.cell;
        const lmin = @min(
            @min(math.Vec3.norm(cell_mat.row(0)), math.Vec3.norm(cell_mat.row(1))),
            math.Vec3.norm(cell_mat.row(2)),
        );
        break :blk 5.0 / lmin;
    };
    var ff_tables_buf = try alloc.alloc(form_factor.LocalFormFactorTable, species.len);
    for (species, 0..) |entry, si| {
        ff_tables_buf[si] = try form_factor.LocalFormFactorTable.init(alloc, entry.upf.*, entry.z_valence, cfg.scf.local_potential, ew_alpha, ff_q_max);
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

    // Spherical ecutrho cutoff for G-space stress sums.
    // Must match SCF's ecutrho to maintain energy-stress consistency.
    const gs = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    const ecutrho = cfg.scf.ecut_ry * gs * gs;

    // Build augmented density (ρ̃ + n̂) for PAW
    const is_paw = scf_result.paw_tabs != null and scf_result.paw_rhoij != null;
    var rho_aug_buf: ?[]f64 = null;
    if (is_paw) {
        rho_aug_buf = try alloc.alloc(f64, scf_result.density.len);
        @memcpy(rho_aug_buf.?, scf_result.density);
        try buildAugmentedDensity(alloc, grid, rho_aug_buf.?, scf_result.paw_rhoij.?, scf_result.paw_tabs.?, atoms, ecutrho);
    }
    defer if (rho_aug_buf) |buf| alloc.free(buf);

    // Get V_eff(G) = V_loc(G) + V_H(G) + V_xc(G) for PAW augmentation stress (matching QE's newd)
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
    );

    // Symmetrize stress using crystal symmetry operations
    if (cfg.scf.symmetry) {
        const symmetry = @import("../symmetry/symmetry.zig");
        const sym_ops = try symmetry.getSymmetryOps(alloc, scf_result.grid.cell, atoms, 1e-5);
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
            // Recompute total (include all terms)
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

/// Convert FFT index to frequency.
fn indexToFreq(i: usize, n: usize) i32 {
    const half = (n - 1) / 2;
    return if (i <= half) @as(i32, @intCast(i)) else @as(i32, @intCast(i)) - @as(i32, @intCast(n));
}

/// FFT real-space density to G-space (reordered to grid layout).
fn densityToReciprocal(
    alloc: std.mem.Allocator,
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

    var plan = try fft.Fft3dPlan.initWithBackend(alloc, nx, ny, nz, fft_backend);
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

/// Kinetic stress: σ_αβ = -(2 × spin / Ω) Σ_nk f w Σ_G (k+G)_α (k+G)_β |c(G)|²
fn kineticStress(alloc: std.mem.Allocator, wavefunctions: ?scf.WavefunctionData, recip: math.Mat3, inv_volume: f64) !Stress3x3 {
    var sigma = zeroStress();
    const wf = wavefunctions orelse return sigma;
    const spin_factor: f64 = 2.0;

    for (wf.kpoints) |kp| {
        var basis = try plane_wave.generate(alloc, recip, wf.ecut_ry, kp.k_cart);
        defer basis.deinit(alloc);
        const gvecs = basis.gvecs;
        const n = gvecs.len;
        if (n != kp.basis_len) continue;

        for (0..kp.nbands) |band| {
            const occ = kp.occupations[band];
            if (occ <= 0.0) continue;
            const c = kp.coefficients[band * n .. (band + 1) * n];
            const prefactor = -2.0 * occ * kp.weight * spin_factor;

            for (0..n) |g| {
                const q = gvecs[g].kpg; // k+G cartesian
                const c2 = c[g].r * c[g].r + c[g].i * c[g].i;
                const qv = [3]f64{ q.x, q.y, q.z };
                for (0..3) |a| {
                    for (a..3) |b| {
                        sigma[a][b] += prefactor * qv[a] * qv[b] * c2;
                    }
                }
            }
        }
    }

    // Symmetrize and scale by inv_volume
    for (0..3) |a| {
        for (a..3) |b| {
            sigma[a][b] *= inv_volume;
            if (a != b) sigma[b][a] = sigma[a][b];
        }
    }
    return sigma;
}

/// Hartree stress: σ_αβ = -(E_H/Ω) δ_αβ + 8π Σ_{G≠0} |ρ(G)|² G_α G_β / |G|⁴
fn hartreeStress(grid: Grid, rho_g: []const math.Complex, e_hartree: f64, inv_volume: f64, ecutrho: f64) Stress3x3 {
    var sigma = zeroStress();
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    var idx: usize = 0;
    var l: usize = 0;
    while (l < grid.nz) : (l += 1) {
        var k: usize = 0;
        while (k < grid.ny) : (k += 1) {
            var h: usize = 0;
            while (h < grid.nx) : (h += 1) {
                const gh = grid.min_h + @as(i32, @intCast(h));
                const gk = grid.min_k + @as(i32, @intCast(k));
                const gl = grid.min_l + @as(i32, @intCast(l));
                if (gh == 0 and gk == 0 and gl == 0) {
                    idx += 1;
                    continue;
                }
                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g2 = math.Vec3.dot(gvec, gvec);
                if (g2 < 1e-12 or g2 >= ecutrho) {
                    idx += 1;
                    continue;
                }
                const rho_val = rho_g[idx];
                const rho2 = rho_val.r * rho_val.r + rho_val.i * rho_val.i;
                const factor = 8.0 * std.math.pi * rho2 / (g2 * g2);
                const gv = [3]f64{ gvec.x, gvec.y, gvec.z };

                for (0..3) |a| {
                    for (a..3) |b| {
                        sigma[a][b] += factor * gv[a] * gv[b];
                    }
                }
                idx += 1;
            }
        }
    }

    // Add diagonal: -(E_H/Ω) δ_αβ
    for (0..3) |a| {
        sigma[a][a] -= e_hartree * inv_volume;
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| {
            sigma[b][a] = sigma[a][b];
        }
    }
    return sigma;
}

/// XC stress.
/// LDA: σ_αβ = δ_αβ (E_xc - ∫V_xc ρ dV) / Ω
/// GGA: + (2/Ω) ∫ (∂f/∂σ) (∂ρ/∂x_α)(∂ρ/∂x_β) dV  (σ here means |∇ρ|²)
fn xcStress(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_r: []const f64,
    rho_core: ?[]const f64,
    e_xc: f64,
    vxc_rho: f64,
    xc_func: xc_mod.Functional,
) !Stress3x3 {
    var sigma = zeroStress();
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
        const fft_obj = grid_mod.Grid{
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
        const rho_total_g = try fft_grid.realToReciprocal(alloc, fft_obj, rho_total, false);
        defer alloc.free(rho_total_g);

        // Compute gradient components: ∂ρ/∂x_α in real space
        var grad_r = [3][]f64{ undefined, undefined, undefined };
        for (0..3) |dir| {
            const grad_g = try alloc.alloc(math.Complex, n_grid);
            defer alloc.free(grad_g);

            const b1 = grid.recip.row(0);
            const b2 = grid.recip.row(1);
            const b3 = grid.recip.row(2);

            var idx_g: usize = 0;
            var lz: usize = 0;
            while (lz < grid.nz) : (lz += 1) {
                var ky: usize = 0;
                while (ky < grid.ny) : (ky += 1) {
                    var hx: usize = 0;
                    while (hx < grid.nx) : (hx += 1) {
                        const gh = grid.min_h + @as(i32, @intCast(hx));
                        const gk = grid.min_k + @as(i32, @intCast(ky));
                        const gl = grid.min_l + @as(i32, @intCast(lz));
                        const gvec = math.Vec3.add(
                            math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                            math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                        );
                        const gdir: f64 = switch (dir) {
                            0 => gvec.x,
                            1 => gvec.y,
                            2 => gvec.z,
                            else => unreachable,
                        };
                        // i * G_dir * ρ(G)
                        const rho_val = rho_total_g[idx_g];
                        grad_g[idx_g] = math.complex.init(-gdir * rho_val.i, gdir * rho_val.r);
                        idx_g += 1;
                    }
                }
            }

            grad_r[dir] = try fft_grid.reciprocalToReal(alloc, fft_obj, grad_g);
        }
        defer for (0..3) |dir| alloc.free(grad_r[dir]);

        // Evaluate df/dσ at each grid point and accumulate GGA stress
        const dv = grid.volume / @as(f64, @floatFromInt(n_grid));
        for (0..n_grid) |i| {
            const rho_val = @max(rho_total[i], 1e-30);
            var g2_val: f64 = 0;
            for (0..3) |d| g2_val += grad_r[d][i] * grad_r[d][i];

            const xc_pt = xc_mod.evalPoint(xc_func, rho_val, g2_val);
            const df_ds = xc_pt.df_dg2;
            if (@abs(df_ds) < 1e-30) continue;

            for (0..3) |a| {
                for (a..3) |b| {
                    sigma[a][b] -= 2.0 * df_ds * grad_r[a][i] * grad_r[b][i] * dv * inv_volume;
                }
            }
        }

        // Symmetrize
        for (0..3) |a| {
            for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
        }
    }

    return sigma;
}

/// Local pseudopotential stress:
/// σ_αβ = -(E_loc/Ω) δ_αβ - (1/Ω) Σ_{G≠0} (G_αG_β/|G|) × Σ_I V'_form(|G|) × Re[ρ*(G) S_I(G)]
fn localStress(
    grid: Grid,
    rho_g: []const math.Complex,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    inv_volume: f64,
    ecutrho: f64,
) Stress3x3 {
    var sigma = zeroStress();
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    // Accumulate E_loc = Σ_{G≠0} V_form(|G|) × Re[ρ(G) exp(+iGR)] internally.
    // For PAW, rho_g is the augmented density ρ̃+n̂, giving the correct evloc.
    // This matches QE's stres_loc which uses rho%of_r (augmented density).
    var evloc: f64 = 0.0;

    var idx: usize = 0;
    var l: usize = 0;
    while (l < grid.nz) : (l += 1) {
        var k: usize = 0;
        while (k < grid.ny) : (k += 1) {
            var h: usize = 0;
            while (h < grid.nx) : (h += 1) {
                const gh = grid.min_h + @as(i32, @intCast(h));
                const gk = grid.min_k + @as(i32, @intCast(k));
                const gl = grid.min_l + @as(i32, @intCast(l));
                if (gh == 0 and gk == 0 and gl == 0) {
                    idx += 1;
                    continue;
                }
                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_norm = math.Vec3.norm(gvec);
                const g2_loc = gvec.x * gvec.x + gvec.y * gvec.y + gvec.z * gvec.z;
                if (g_norm < 1e-12 or g2_loc >= ecutrho) {
                    idx += 1;
                    continue;
                }
                const rho_val = rho_g[idx];
                const gv = [3]f64{ gvec.x, gvec.y, gvec.z };

                // Accumulate: Σ_I V_form(|G|) and V_form'(|G|) contributions
                var vloc_rho_re: f64 = 0.0; // Re[V_form × S*_I(G) × ρ(G)]
                var dvloc_rho_re: f64 = 0.0; // Re[V'_form × S*_I(G) × ρ(G)]

                for (atoms) |atom| {
                    const v_loc = if (ff_tables) |tables|
                        tables[atom.species_index].eval(g_norm)
                    else
                        hamiltonian.localFormFactor(&species[atom.species_index], g_norm);

                    const dv_loc = if (ff_tables) |tables|
                        tables[atom.species_index].evalDeriv(g_norm)
                    else blk: {
                        // Numerical derivative
                        const dq: f64 = 0.01;
                        const vp = hamiltonian.localFormFactor(&species[atom.species_index], g_norm + dq);
                        const vm = hamiltonian.localFormFactor(&species[atom.species_index], g_norm - dq);
                        break :blk (vp - vm) / (2.0 * dq);
                    };

                    const phase = math.Vec3.dot(gvec, atom.position);
                    const cos_phase = std.math.cos(phase);
                    const sin_phase = std.math.sin(phase);
                    // Energy: E_loc = Ω Σ_G Re[ρ̃ conj(V_G)] where V_G = (1/Ω) V_form exp(-iGR)
                    // conj(V_G) = (1/Ω) V_form exp(+iGR)
                    // Re[ρ̃ exp(+iGR)] = Re[(ρ_r+iρ_i)(cos+isin)] = ρ_r cos - ρ_i sin
                    // The stress G-derivative uses the same phase factor.
                    const re_rho_si = rho_val.r * cos_phase - rho_val.i * sin_phase;

                    vloc_rho_re += v_loc * re_rho_si;
                    dvloc_rho_re += dv_loc * re_rho_si;
                }

                evloc += vloc_rho_re;

                // Local stress contribution from this G:
                // From V_form(|G|) dependence: dV_form/dε_αβ = V'_form × d|G|/dε_αβ = V'_form × (-G_αG_β/|G|)
                // From 1/Ω (in ρ): -(1-Tr(ε)) gives -δ_αβ × E_loc at the end
                // So: σ_αβ += -(1/Ω) × dvloc_rho_re × G_αG_β/|G|
                const inv_gnorm = 1.0 / g_norm;
                for (0..3) |a| {
                    for (a..3) |b| {
                        sigma[a][b] -= dvloc_rho_re * gv[a] * gv[b] * inv_gnorm * inv_volume;
                    }
                }

                idx += 1;
            }
        }
    }

    // Diagonal: -(E_loc/Ω) δ_αβ, using internally-computed evloc from the augmented density.
    for (0..3) |a| {
        sigma[a][a] -= evloc * inv_volume;
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}

/// Nonlocal pseudopotential stress.
/// σ_αβ = -(E_nl/Ω) δ_αβ - (2 spin/Ω²) Σ_nk f w Re[Σ D conj(dp_αβ) p]
/// where dp_αβ = Σ_G (∂φ/∂q_α × q_β) × S(G) × c(G)
fn nonlocalStress(
    alloc: std.mem.Allocator,
    wavefunctions: ?scf.WavefunctionData,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    radial_tables_list: ?[]nonlocal.RadialTableSet,
    paw_dij_per_atom: ?[]const []const f64,
    paw_dij_m_per_atom: ?[]const []const f64,
    paw_tabs: ?[]const paw_mod.PawTab,
) !Stress3x3 {
    var sigma = zeroStress();
    const wf = wavefunctions orelse return sigma;
    const inv_volume = 1.0 / volume;
    const spin_factor: f64 = 2.0;

    for (wf.kpoints) |kp| {
        var basis = try plane_wave.generate(alloc, recip, wf.ecut_ry, kp.k_cart);
        defer basis.deinit(alloc);
        const gvecs = basis.gvecs;
        const n = gvecs.len;
        if (n != kp.basis_len) continue;

        for (atoms, 0..) |atom, atom_idx| {
            const sp = &species[atom.species_index];
            const upf = sp.upf;
            if (upf.beta.len == 0 or upf.dij.len == 0) continue;
            const nb = upf.beta.len;
            // Use PAW D_ij if available, otherwise UPF D^0_ij
            const dij_data: []const f64 = if (paw_dij_per_atom) |pda| pda[atom_idx] else upf.dij;
            // m-resolved D for PAW (preferred for stress consistency with Hamiltonian)
            const dij_m_data: ?[]const f64 = if (paw_dij_m_per_atom) |pda| pda[atom_idx] else null;
            // PAW overlap correction: q_ij = S_ij - δ_ij
            const qij_data: ?[]const f64 = if (paw_tabs) |tabs| if (atom.species_index < tabs.len) tabs[atom.species_index].sij else null else null;
            // Buffer for D_ij - ε_nk × q_ij (PAW overlap stress)
            const dij_eff_buf: ?[]f64 = if (qij_data != null) try alloc.alloc(f64, nb * nb) else null;
            defer if (dij_eff_buf) |buf| alloc.free(buf);

            const tables = if (radial_tables_list) |rtl| if (atom.species_index < rtl.len) rtl[atom.species_index] else null else null;

            // Pre-compute radial values and derivatives for each beta, G
            const radial_vals = try alloc.alloc(f64, nb * n);
            defer alloc.free(radial_vals);
            const radial_derivs = try alloc.alloc(f64, nb * n);
            defer alloc.free(radial_derivs);

            for (0..nb) |b_idx| {
                const l_val = upf.beta[b_idx].l orelse 0;
                _ = l_val;
                for (0..n) |g| {
                    const gmag = math.Vec3.norm(gvecs[g].kpg);
                    if (tables) |t| {
                        radial_vals[b_idx * n + g] = t.tables[b_idx].eval(gmag);
                        radial_derivs[b_idx * n + g] = t.tables[b_idx].evalDeriv(gmag);
                    } else {
                        const beta = upf.beta[b_idx];
                        const l = beta.l orelse 0;
                        radial_vals[b_idx * n + g] = nonlocal.radialProjector(beta.values, upf.r, upf.rab, l, gmag);
                        // Numerical derivative
                        const dg: f64 = 0.001;
                        const rp = nonlocal.radialProjector(beta.values, upf.r, upf.rab, l, gmag + dg);
                        const rm = nonlocal.radialProjector(beta.values, upf.r, upf.rab, l, if (gmag > dg) gmag - dg else 0.0);
                        radial_derivs[b_idx * n + g] = (rp - rm) / (2.0 * dg);
                    }
                }
            }

            // Count total m channels
            var m_total: usize = 0;
            for (0..nb) |b_idx| {
                const l_val = upf.beta[b_idx].l orelse 0;
                m_total += @as(usize, @intCast(2 * l_val + 1));
            }

            // Compute phase[g] = exp(+i G·R)
            const phase_buf = try alloc.alloc(math.Complex, n);
            defer alloc.free(phase_buf);
            for (gvecs, 0..) |gv, g| {
                phase_buf[g] = math.complex.expi(math.Vec3.dot(gv.cart, atom.position));
            }

            // Build m offsets
            const m_offsets = try alloc.alloc(usize, nb);
            defer alloc.free(m_offsets);
            const m_counts = try alloc.alloc(usize, nb);
            defer alloc.free(m_counts);
            {
                var off: usize = 0;
                for (0..nb) |b_idx| {
                    const l_val = upf.beta[b_idx].l orelse 0;
                    m_offsets[b_idx] = off;
                    m_counts[b_idx] = @as(usize, @intCast(2 * l_val + 1));
                    off += m_counts[b_idx];
                }
            }

            // Allocate work buffers
            const p_buf = try alloc.alloc(math.Complex, m_total);
            defer alloc.free(p_buf);
            // dp_buf[dir][bm] for 3 directions; q_β factor accumulated in stress loop
            const dp_buf = try alloc.alloc(math.Complex, 3 * m_total);
            defer alloc.free(dp_buf);
            // m-resolved D_eff buffer (PAW with m-resolved D)
            const dij_m_eff_buf: ?[]f64 = if (dij_m_data != null and qij_data != null) try alloc.alloc(f64, m_total * m_total) else null;
            defer if (dij_m_eff_buf) |buf| alloc.free(buf);

            for (0..kp.nbands) |band| {
                const occ = kp.occupations[band];
                if (occ <= 0.0) continue;
                const c = kp.coefficients[band * n .. (band + 1) * n];

                // PAW: compute D_ij^eff = D_ij - ε_nk × q_ij for this band
                // Use m-resolved D when available for consistency with Hamiltonian
                if (dij_m_eff_buf) |dm_buf| {
                    const eigenval = kp.eigenvalues[band];
                    const dm = dij_m_data.?;
                    const qij = qij_data.?;
                    // D_m_eff[bm*mt+jm] = D_m[bm*mt+jm] - ε × q_expanded[bm*mt+jm]
                    // q is diagonal in m: q_expanded[(i,m),(j,m')] = (q[i*nb+j] - δ_ij) × δ_{m,m'}
                    @memcpy(dm_buf, dm);
                    for (0..nb) |bi| {
                        for (0..nb) |bj| {
                            if ((upf.beta[bi].l orelse 0) != (upf.beta[bj].l orelse 0)) continue;
                            const q_ij = qij[bi * nb + bj] - (if (bi == bj) @as(f64, 1.0) else @as(f64, 0.0));
                            if (q_ij == 0.0) continue;
                            const mc = m_counts[bi];
                            for (0..mc) |mi| {
                                const bm = m_offsets[bi] + mi;
                                const jm = m_offsets[bj] + mi;
                                dm_buf[bm * m_total + jm] -= eigenval * q_ij;
                            }
                        }
                    }
                }
                const dij_for_band: []const f64 = if (dij_eff_buf) |buf| blk: {
                    const eigenval = kp.eigenvalues[band];
                    const qij = qij_data.?;
                    for (0..nb) |bi| {
                        for (0..nb) |bj| {
                            const idx_ij = bi * nb + bj;
                            const q_ij = qij[idx_ij] - (if (bi == bj) @as(f64, 1.0) else @as(f64, 0.0));
                            buf[idx_ij] = dij_data[idx_ij] - eigenval * q_ij;
                        }
                    }
                    break :blk buf;
                } else dij_data;
                // Use m-resolved D_eff for PAW stress when available
                const use_dij_m = dij_m_eff_buf != null;

                // Step A: Compute projections p_{bm} = Σ_G φ_{bm}(q) S(G) c(G)
                // and dp_{bm,α} = Σ_G (∂φ_{bm}/∂q_α) S(G) c(G)
                @memset(p_buf, math.complex.init(0, 0));
                @memset(dp_buf, math.complex.init(0, 0));

                for (0..nb) |b_idx| {
                    const l_val = upf.beta[b_idx].l orelse 0;
                    const m_count = m_counts[b_idx];
                    const m_off = m_offsets[b_idx];

                    for (0..m_count) |m_idx| {
                        const m = @as(i32, @intCast(m_idx)) - l_val;

                        for (0..n) |g| {
                            const q = gvecs[g].kpg;
                            const q_mag = math.Vec3.norm(q);
                            const radial = radial_vals[b_idx * n + g];
                            const ylm = nonlocal.realSphericalHarmonic(l_val, m, q.x, q.y, q.z);
                            const phi = 4.0 * std.math.pi * radial * ylm;

                            const pc = math.complex.mul(phase_buf[g], c[g]);
                            p_buf[m_off + m_idx] = math.complex.add(p_buf[m_off + m_idx], math.complex.scale(pc, phi));

                            // Compute ∂φ/∂q_α = 4π [R'(|q|) q_α/|q| Y + R(|q|) ∂Y/∂q_α]
                            if (q_mag < 1e-12) continue;
                            const dradial = radial_derivs[b_idx * n + g];
                            const inv_qmag = 1.0 / q_mag;
                            const nhat = [3]f64{ q.x * inv_qmag, q.y * inv_qmag, q.z * inv_qmag };

                            // ∂Y_lm/∂q_α = (1/|q|) [∂Y/∂n_α - n_α Σ_β n_β ∂Y/∂n_β]
                            const dy = dYlm_dq(l_val, m, q.x, q.y, q.z, q_mag);

                            for (0..3) |dir| {
                                const dphi = 4.0 * std.math.pi * (dradial * nhat[dir] * ylm + radial * dy[dir]);
                                dp_buf[dir * m_total + m_off + m_idx] = math.complex.add(
                                    dp_buf[dir * m_total + m_off + m_idx],
                                    math.complex.scale(pc, dphi),
                                );
                            }
                        }
                    }
                }

                // Step B: D-apply and accumulate stress
                // σ_αβ += -prefactor × Re[Σ_{bj,m} D_{bj} × conj(dp_{bm,α}) × q_β_component × p_{jm}]
                // But dp already has the (k+G) dependence baked in per-G, we need to handle q_β differently.
                // Actually, dp_buf stores Σ_G (∂φ/∂q_α) × S × c, and we need Σ_G (∂φ/∂q_α × q_β) × S × c.
                // These are different because q_β depends on G.
                // We need to compute dp_αβ = Σ_G (∂φ/∂q_α × q_β) × S × c for each (α,β) pair.

                // Redo: compute per-G contribution directly into stress
                const prefactor = 2.0 * occ * kp.weight * spin_factor * inv_volume * inv_volume;

                for (0..n) |g| {
                    const q = gvecs[g].kpg;
                    const q_mag = math.Vec3.norm(q);
                    const pc = math.complex.mul(phase_buf[g], c[g]);
                    if (q_mag < 1e-12) continue;
                    const dradial_all = radial_derivs;
                    _ = dradial_all;
                    const inv_qmag = 1.0 / q_mag;
                    const nhat = [3]f64{ q.x * inv_qmag, q.y * inv_qmag, q.z * inv_qmag };
                    const qv = [3]f64{ q.x, q.y, q.z };

                    // For this G, compute q(G) = Σ_{bm} conj(Dp_{bm}) × φ_{bm}(G)
                    // and dq_α(G) = Σ_{bm} conj(Dp_{bm}) × dφ_{bm}/dq_α(G)
                    // Then: σ_αβ += -prefactor × Re[dq_α(G) × q_β × S(G) × c(G)]
                    // But we need Dp_{bm} = Σ_j D_{bj} × p_{jm}

                    for (0..nb) |b_idx| {
                        const l_b = upf.beta[b_idx].l orelse 0;
                        const m_count_b = m_counts[b_idx];

                        for (0..m_count_b) |m_idx| {
                            const m = @as(i32, @intCast(m_idx)) - l_b;
                            const bm = m_offsets[b_idx] + m_idx;

                            // Compute Dp_{bm} = Σ_{j,m'} D^eff_{bm,jm'} × p_{jm'}
                            var dp_bm = math.complex.init(0, 0);
                            if (use_dij_m) {
                                // m-resolved D: sum over all (j, m') with l_j == l_b
                                const dm_eff = dij_m_eff_buf.?;
                                for (0..nb) |j_idx| {
                                    if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                                    const mj_count = m_counts[j_idx];
                                    for (0..mj_count) |mj| {
                                        const jm = m_offsets[j_idx] + mj;
                                        const d_val = dm_eff[bm * m_total + jm];
                                        if (d_val == 0) continue;
                                        dp_bm = math.complex.add(dp_bm, math.complex.scale(
                                            p_buf[jm],
                                            d_val,
                                        ));
                                    }
                                }
                            } else {
                                // Radial D: same D for all m
                                for (0..nb) |j_idx| {
                                    if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                                    const d_val = dij_for_band[b_idx * nb + j_idx];
                                    if (d_val == 0) continue;
                                    dp_bm = math.complex.add(dp_bm, math.complex.scale(
                                        p_buf[m_offsets[j_idx] + m_idx],
                                        d_val,
                                    ));
                                }
                            }
                            if (dp_bm.r == 0 and dp_bm.i == 0) continue;

                            const radial = radial_vals[b_idx * n + g];
                            const dradial_val = radial_derivs[b_idx * n + g];
                            const ylm = nonlocal.realSphericalHarmonic(l_b, m, q.x, q.y, q.z);
                            const dy = dYlm_dq(l_b, m, q.x, q.y, q.z, q_mag);

                            // z = conj(Dp) × S(G) × c(G)
                            const z = math.complex.mul(math.complex.conj(dp_bm), pc);

                            for (0..3) |a| {
                                const dphi_a = 4.0 * std.math.pi * (dradial_val * nhat[a] * ylm + radial * dy[a]);
                                for (a..3) |b| {
                                    // σ_αβ += -prefactor × Re[dphi_α × q_β × z]
                                    sigma[a][b] -= prefactor * dphi_a * qv[b] * z.r;
                                }
                            }
                        }
                    }
                }

                // Diagonal: from 1/Ω factor → -(E_nl_nk/Ω) δ contribution
                // For PAW: uses D^eff = D_ij - ε_nk × q_ij
                var e_nl_nk: f64 = 0.0;
                for (0..nb) |b_idx| {
                    const l_b = upf.beta[b_idx].l orelse 0;
                    const m_count_b = m_counts[b_idx];

                    for (0..m_count_b) |m_idx| {
                        const bm = m_offsets[b_idx] + m_idx;
                        var dp_bm_e = math.complex.init(0, 0);
                        if (use_dij_m) {
                            const dm_eff = dij_m_eff_buf.?;
                            for (0..nb) |j_idx| {
                                if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                                const mj_count = m_counts[j_idx];
                                for (0..mj_count) |mj| {
                                    const jm = m_offsets[j_idx] + mj;
                                    const d_val = dm_eff[bm * m_total + jm];
                                    if (d_val == 0) continue;
                                    dp_bm_e = math.complex.add(dp_bm_e, math.complex.scale(
                                        p_buf[jm],
                                        d_val,
                                    ));
                                }
                            }
                        } else {
                            for (0..nb) |j_idx| {
                                if ((upf.beta[j_idx].l orelse 0) != l_b) continue;
                                dp_bm_e = math.complex.add(dp_bm_e, math.complex.scale(
                                    p_buf[m_offsets[j_idx] + m_idx],
                                    dij_for_band[b_idx * nb + j_idx],
                                ));
                            }
                        }
                        const p_bm = p_buf[bm];
                        e_nl_nk += math.complex.mul(math.complex.conj(p_bm), dp_bm_e).r;
                    }
                }
                e_nl_nk *= inv_volume; // per-band nonlocal energy (without occ/weight/spin)
                const diag_contrib = -occ * kp.weight * spin_factor * e_nl_nk * inv_volume;
                for (0..3) |a| sigma[a][a] += diag_contrib;
            }
        }
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}

/// NLCC stress: contribution from core charge density dependence on strain.
/// σ_αβ = -(1/Ω) Σ_{G≠0} V_xc(G) × Σ_I ρ_core_form'(|G|) × G_αG_β/|G| × Re[S*_I(G)]
///        -(E_nlcc/Ω) δ_αβ  (volume scaling)
fn nlccStress(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_r: []const f64,
    rho_core: ?[]const f64,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    xc_func: xc_mod.Functional,
    ecutrho: f64,
) !Stress3x3 {
    var sigma = zeroStress();
    const core_tables = rho_core_tables orelse return sigma;

    // Check if any species has NLCC
    var has_nlcc = false;
    for (species) |sp| {
        if (sp.upf.nlcc.len > 0) {
            has_nlcc = true;
            break;
        }
    }
    if (!has_nlcc) return sigma;
    if (rho_core == null) return sigma;

    const inv_volume = 1.0 / grid.volume;

    // Compute V_xc(G) from V_xc(r)
    const n_grid = grid.nx * grid.ny * grid.nz;
    const rho_total = try alloc.alloc(f64, n_grid);
    defer alloc.free(rho_total);
    for (0..n_grid) |i| {
        rho_total[i] = rho_r[i];
        if (rho_core) |rc| rho_total[i] += rc[i];
    }

    const xc_fields = try xc_fields_mod.computeXcFields(alloc, grid_mod.Grid{
        .nx = grid.nx,
        .ny = grid.ny,
        .nz = grid.nz,
        .min_h = grid.min_h,
        .min_k = grid.min_k,
        .min_l = grid.min_l,
        .cell = grid.cell,
        .recip = grid.recip,
        .volume = grid.volume,
    }, rho_total, null, false, xc_func);
    defer {
        alloc.free(xc_fields.vxc);
        alloc.free(xc_fields.exc);
    }

    const fft_obj = grid_mod.Grid{
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
    const vxc_g = try fft_grid.realToReciprocal(alloc, fft_obj, xc_fields.vxc, false);
    defer alloc.free(vxc_g);

    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    var idx: usize = 0;
    var lz: usize = 0;
    while (lz < grid.nz) : (lz += 1) {
        var ky: usize = 0;
        while (ky < grid.ny) : (ky += 1) {
            var hx: usize = 0;
            while (hx < grid.nx) : (hx += 1) {
                const gh = grid.min_h + @as(i32, @intCast(hx));
                const gk = grid.min_k + @as(i32, @intCast(ky));
                const gl = grid.min_l + @as(i32, @intCast(lz));
                if (gh == 0 and gk == 0 and gl == 0) {
                    idx += 1;
                    continue;
                }
                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_norm = math.Vec3.norm(gvec);
                const g2_nlcc = gvec.x * gvec.x + gvec.y * gvec.y + gvec.z * gvec.z;
                if (g_norm < 1e-12 or g2_nlcc >= ecutrho) {
                    idx += 1;
                    continue;
                }

                const vxc_val = vxc_g[idx];
                const gv = [3]f64{ gvec.x, gvec.y, gvec.z };

                for (atoms) |atom| {
                    const si = atom.species_index;
                    if (species[si].upf.nlcc.len == 0) continue;

                    const drhoc = core_tables[si].evalDeriv(g_norm);
                    const phase = math.Vec3.dot(gvec, atom.position);
                    const cos_phase = std.math.cos(phase);
                    const sin_phase = std.math.sin(phase);
                    // Re[V_xc(G) × conj(exp(-iG·R))] = Re[V_xc × exp(+iGR)] = Vxc_r cos - Vxc_i sin
                    const re_vxc_si = vxc_val.r * cos_phase - vxc_val.i * sin_phase;

                    const factor = -drhoc * re_vxc_si / g_norm * inv_volume;
                    for (0..3) |a| {
                        for (a..3) |b| {
                            sigma[a][b] += factor * gv[a] * gv[b];
                        }
                    }
                }
                idx += 1;
            }
        }
    }

    // NLCC diagonal: -(∫V_xc × ρ_core dr)/Ω δ_αβ
    // The XC stress diagonal uses (E_xc - vxc_rho)/Ω where vxc_rho = ∫V_xc × ρ_aug (not ρ_total).
    // This contains an excess +∫V_xc × ρ_core/Ω that must be cancelled by this NLCC diagonal.
    {
        const dv = grid.volume / @as(f64, @floatFromInt(n_grid));
        var vxc_core: f64 = 0.0;
        for (0..n_grid) |i| {
            if (rho_core) |rc| {
                vxc_core += xc_fields.vxc[i] * rc[i] * dv;
            }
        }
        const nlcc_diag = -vxc_core * inv_volume;
        for (0..3) |a| sigma[a][a] += nlcc_diag;
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}

/// Build augmented density ρ̃ + n̂ for PAW stress calculation.
/// Same logic as addPawCompensationCharge in scf.zig.
fn buildAugmentedDensity(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho: []f64,
    paw_rhoij: []const []const f64,
    paw_tabs: []const paw_mod.PawTab,
    atoms: []const hamiltonian.AtomData,
    ecutrho: f64,
) !void {
    const total = grid.nx * grid.ny * grid.nz;
    const n_hat_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(n_hat_g);
    @memset(n_hat_g, math.complex.init(0.0, 0.0));

    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);
    const inv_omega = 1.0 / grid.volume;

    var idx: usize = 0;
    var il: usize = 0;
    while (il < grid.nz) : (il += 1) {
        var ik: usize = 0;
        while (ik < grid.ny) : (ik += 1) {
            var ih: usize = 0;
            while (ih < grid.nx) : (ih += 1) {
                const gh = grid.min_h + @as(i32, @intCast(ih));
                const gk = grid.min_k + @as(i32, @intCast(ik));
                const gl = grid.min_l + @as(i32, @intCast(il));
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_abs = math.Vec3.norm(gvec);
                const g2_aug = math.Vec3.dot(gvec, gvec);
                if (g2_aug >= ecutrho) {
                    idx += 1;
                    continue;
                }

                var sum_re: f64 = 0.0;
                var sum_im: f64 = 0.0;

                for (0..atoms.len) |a| {
                    const sp = atoms[a].species_index;
                    if (sp >= paw_tabs.len) continue;
                    const tab = &paw_tabs[sp];
                    if (tab.nbeta == 0) continue;
                    const nb = tab.nbeta;
                    const rij = paw_rhoij[a];
                    const pos = atoms[a].position;

                    const g_dot_r = math.Vec3.dot(gvec, pos);
                    const sf_re = @cos(g_dot_r);
                    const sf_im = -@sin(g_dot_r);

                    for (0..tab.n_qijl_entries) |e| {
                        const qidx = tab.qijl_indices[e];
                        if (qidx.l != 0) continue;
                        const i = qidx.first;
                        const j = qidx.second;
                        const rij_val = rij[i * nb + j];
                        if (@abs(rij_val) < 1e-30) continue;

                        const qijl_g = tab.evalQijlForm(e, g_abs);
                        if (@abs(qijl_g) < 1e-30) continue;

                        const ylm = 1.0 / @sqrt(4.0 * std.math.pi);
                        const gaunt = 1.0 / @sqrt(4.0 * std.math.pi);
                        const sym_factor: f64 = if (i != j) 2.0 else 1.0;
                        const contrib = rij_val * qijl_g * ylm * gaunt * sym_factor * inv_omega;
                        sum_re += contrib * sf_re;
                        sum_im += contrib * sf_im;
                    }
                }

                n_hat_g[idx].r += sum_re;
                n_hat_g[idx].i += sum_im;
                idx += 1;
            }
        }
    }

    // IFFT n_hat(G) → n_hat(r) and add to density
    const fft_obj = grid_mod.Grid{
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
    const n_hat_r = try fft_grid.reciprocalToReal(alloc, fft_obj, n_hat_g);
    defer alloc.free(n_hat_r);
    for (0..@min(rho.len, n_hat_r.len)) |i| {
        rho[i] += n_hat_r[i];
    }
}

/// PAW augmentation charge stress (off-diagonal only, matching QE's addusstress).
/// σ^aug_αβ = (1/Ω) Σ_{G≠0} Σ_{a,ij} ρ_ij × dQ_ij(|G|)/d|G| × V_eff(G) × S*_a(G) × (-G_αG_β/|G|) / Ω
fn augmentationStress(
    alloc: std.mem.Allocator,
    grid: Grid,
    potential_values: ?[]const math.Complex,
    paw_rhoij: ?[]const []const f64,
    paw_tabs: ?[]const paw_mod.PawTab,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
    ecutrho: f64,
) !Stress3x3 {
    var sigma = zeroStress();
    const tabs = paw_tabs orelse return sigma;
    const rhoij = paw_rhoij orelse return sigma;
    const pot_vals = potential_values orelse return sigma;
    _ = alloc;

    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);
    const ylm_gaunt = 1.0 / (4.0 * std.math.pi); // Y_00 × Gaunt for L=0

    var idx: usize = 0;
    var lz: usize = 0;
    while (lz < grid.nz) : (lz += 1) {
        var ky: usize = 0;
        while (ky < grid.ny) : (ky += 1) {
            var hx: usize = 0;
            while (hx < grid.nx) : (hx += 1) {
                const gh = grid.min_h + @as(i32, @intCast(hx));
                const gk = grid.min_k + @as(i32, @intCast(ky));
                const gl = grid.min_l + @as(i32, @intCast(lz));

                const gvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_norm = math.Vec3.norm(gvec);
                const g2_aug = gvec.x * gvec.x + gvec.y * gvec.y + gvec.z * gvec.z;
                if (g2_aug >= ecutrho) {
                    idx += 1;
                    continue;
                }

                const v_eff = pot_vals[idx];

                for (atoms, 0..) |atom, a| {
                    const si = atom.species_index;
                    if (si >= tabs.len) continue;
                    const tab = &tabs[si];
                    if (tab.nbeta == 0) continue;
                    const nb = tab.nbeta;
                    const rij = rhoij[a];

                    const g_dot_r = math.Vec3.dot(gvec, atom.position);
                    // S*_a(G) = exp(+iG·R)
                    const sf_re = @cos(g_dot_r);
                    const sf_im = @sin(g_dot_r);
                    // Re[V_eff(G) × exp(+iG·R)] = V_re cos(GR) - V_im sin(GR)
                    const re_vs = v_eff.r * sf_re - v_eff.i * sf_im;

                    for (0..tab.n_qijl_entries) |e| {
                        const qidx = tab.qijl_indices[e];
                        if (qidx.l != 0) continue;
                        const i = qidx.first;
                        const j = qidx.second;
                        const rij_val = rij[i * nb + j];
                        if (@abs(rij_val) < 1e-30) continue;

                        const sym_factor: f64 = if (i != j) 2.0 else 1.0;
                        const rij_sym = rij_val * sym_factor;

                        // Off-diagonal: dQ/dG derivative term (G≠0 only)
                        if (g_norm < 1e-12) continue;
                        const dqijl_g = tab.evalQijlFormDeriv(e, g_norm);
                        const factor = -rij_sym * dqijl_g * re_vs * ylm_gaunt * inv_volume / g_norm;
                        const gv = [3]f64{ gvec.x, gvec.y, gvec.z };
                        for (0..3) |a2| {
                            for (a2..3) |b| {
                                sigma[a2][b] += factor * gv[a2] * gv[b];
                            }
                        }
                    }
                }
                idx += 1;
            }
        }
    }

    // Symmetrize
    for (0..3) |a| {
        for (a + 1..3) |b| sigma[b][a] = sigma[a][b];
    }
    return sigma;
}

/// Compute ∂Y_lm(q̂)/∂q_α for real spherical harmonics.
/// Returns [3]f64 for (x, y, z) components.
fn dYlm_dq(l: i32, m: i32, qx: f64, qy: f64, qz: f64, q_mag: f64) [3]f64 {
    if (q_mag < 1e-15) return .{ 0, 0, 0 };
    const inv_q = 1.0 / q_mag;
    const inv_q2 = inv_q * inv_q;
    const nx = qx * inv_q;
    const ny = qy * inv_q;
    const nz = qz * inv_q;

    // ∂Y/∂q_α = (1/|q|) × [∂Y/∂n_α - n_α × (n · ∇_n Y)]
    // where ∇_n Y means gradient with respect to unit vector components

    if (l == 0) {
        return .{ 0, 0, 0 };
    }

    if (l == 1) {
        const c = @sqrt(3.0 / (4.0 * std.math.pi));
        // Y_{1,-1} = c ny, Y_{1,0} = c nz, Y_{1,1} = c nx
        // ∂Y/∂n_α: constant since Y is linear in n
        var dY_dn = [3]f64{ 0, 0, 0 };
        switch (m) {
            -1 => dY_dn = .{ 0, c, 0 },
            0 => dY_dn = .{ 0, 0, c },
            1 => dY_dn = .{ c, 0, 0 },
            else => {},
        }
        // n · ∇_n Y
        const n_dot_grad = nx * dY_dn[0] + ny * dY_dn[1] + nz * dY_dn[2];
        const nn = [3]f64{ nx, ny, nz };
        return .{
            (dY_dn[0] - nn[0] * n_dot_grad) * inv_q,
            (dY_dn[1] - nn[1] * n_dot_grad) * inv_q,
            (dY_dn[2] - nn[2] * n_dot_grad) * inv_q,
        };
    }

    if (l == 2) {
        // Y_{2,-2} = c2*2*nx*ny, Y_{2,-1} = c1*ny*nz, Y_{2,0} = c0*(3nz²-1)
        // Y_{2,1} = c1*nx*nz, Y_{2,2} = c2*(nx²-ny²)
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
        // Use numerical derivative for l=3
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
    std.debug.print("Stress {s:12} (GPa): {d:10.4} {d:10.4} {d:10.4} / {d:10.4} {d:10.4} {d:10.4} / {d:10.4} {d:10.4} {d:10.4}  P={d:.2}\n", .{
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

/// Symmetrize a stress tensor using crystal symmetry operations.
/// R_cart = cell × rot × cell^{-1} for each SymOp.
/// σ_sym = (1/N_ops) Σ R σ R^T
pub fn symmetrizeStress(sigma: Stress3x3, sym_ops: []const @import("../symmetry/symmetry.zig").SymOp, cell: math.Mat3) Stress3x3 {
    if (sym_ops.len == 0) return sigma;

    // Compute cell inverse for fractional→Cartesian rotation conversion
    // R_cart = cell^T × rot × (cell^T)^{-1}
    // Using row-vector convention: a_i are rows of cell, so position = frac × cell
    // Rotation in fractional: frac' = frac × rot
    // In Cartesian: x' = frac' × cell = frac × rot × cell = x × cell^{-1} × rot × cell
    // So R_cart = cell^{-1} × rot × cell (acting on column vectors from right)
    // For stress (rank-2 tensor): σ' = R σ R^T

    // Compute cell inverse (needed for fractional→Cartesian rotation conversion)
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
        // Convert fractional rotation to Cartesian rotation matrix.
        // Convention: frac' = R_frac × frac (column vector), x = C^T × frac
        // => R_cart = C^T × R_frac × (C^T)^{-1} = C^T × R_frac × (C^{-1})^T
        var r_cart: [3][3]f64 = undefined;

        // Step 1: ct_rot = C^T × R_frac
        // ct_rot[i][j] = Σ_k C^T[i][k] * R[k][j] = Σ_k C[k][i] * R[k][j]
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

        // Step 2: R_cart = ct_rot × (C^{-1})^T
        // R_cart[i][j] = Σ_k ct_rot[i][k] * c_inv^T[k][j] = Σ_k ct_rot[i][k] * c_inv[j][k]
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0;
                for (0..3) |k| {
                    s += ct_rot[i][k] * c_inv[j][k];
                }
                r_cart[i][j] = s;
            }
        }

        // σ' = R σ R^T
        var rs: [3][3]f64 = undefined; // R × σ
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0;
                for (0..3) |k| {
                    s += r_cart[i][k] * sigma[k][j];
                }
                rs[i][j] = s;
            }
        }
        // σ' = rs × R^T
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0;
                for (0..3) |k| {
                    s += rs[i][k] * r_cart[j][k]; // R^T_{kj} = R_{jk}
                }
                result[i][j] += s;
            }
        }
    }

    // Average
    const inv_n = 1.0 / @as(f64, @floatFromInt(sym_ops.len));
    for (0..3) |i| {
        for (0..3) |j| {
            result[i][j] *= inv_n;
        }
    }
    return result;
}

test "ewald stress finite difference" {
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

    const e0 = try ewald.ionIonEnergy(cell, recip_lat, &charges, &positions, params);
    const sigma = try ewald.ionIonStress(cell, recip_lat, &charges, &positions, params);

    // Fractional coordinates for Si diamond: (0,0,0) and (0.25, 0.25, 0.25)
    const frac = [_]math.Vec3{
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.25, .y = 0.25, .z = 0.25 },
    };

    const vol0 = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));

    // Finite difference: strain the cell and recompute energy
    const delta = 1e-5;
    for (0..3) |al| {
        for (al..3) |be| {
            // Apply Lagrangian strain ε_αβ = ±delta to cell: a'_iα = a_iα + Σ_β ε_αβ a_iβ
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

            // Convert fractional → Cartesian with strained cells
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

            const e_p = try ewald.ionIonEnergy(cell_p, recip_p, &charges, &pos_p, params);
            const e_m = try ewald.ionIonEnergy(cell_m, recip_m, &charges, &pos_m, params);

            const fd_factor: f64 = if (al == be) 1.0 else 2.0;
            const sigma_fd = (e_p - e_m) / (2.0 * delta * fd_factor) / vol0;

            try testing.expectApproxEqAbs(sigma[al][be], sigma_fd, @max(@abs(sigma_fd) * 1e-3, 1e-6));
        }
    }
    _ = e0;
}
