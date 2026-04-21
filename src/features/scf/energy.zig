const std = @import("std");
const config = @import("../config/config.zig");
const coulomb = @import("../coulomb/coulomb.zig");
const ewald = @import("../ewald/ewald.zig");
const term_mod = @import("../dft/term.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const gvec_iter = @import("gvec_iter.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const d3 = @import("../vdw/d3.zig");
const d3_params = @import("../vdw/d3_params.zig");
const xc_fields_mod = @import("xc_fields.zig");
const xc = @import("../xc/xc.zig");

const Grid = grid_mod.Grid;

const realToReciprocal = fft_grid.realToReciprocal;

const computeXcFields = xc_fields_mod.computeXcFields;
const computeXcFieldsSpin = xc_fields_mod.computeXcFieldsSpin;

pub const EnergyTerms = struct {
    total: f64,
    band: f64,
    hartree: f64,
    vxc_rho: f64,
    xc: f64,
    ion_ion: f64,
    psp_core: f64,
    double_counting: f64,
    local_pseudo: f64,
    nonlocal_pseudo: f64,
    entropy: f64 = 0.0, // -T*S term for smearing
    dispersion: f64 = 0.0, // DFT-D3 dispersion correction
    paw_onsite: f64 = 0.0, // PAW on-site energy correction (E^1 - Ẽ^1)
    paw_dxc_rhoij: f64 = 0.0, // -Σ D^xc_ij × ρ_ij (PAW double-counting correction)
};

/// Compute energy terms (Hartree + XC) from density.
/// If coulomb_r_cut is non-null, the cutoff Coulomb kernel is used for Hartree energy.
pub fn computeEnergyTerms(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    rho: []f64,
    rho_core: ?[]const f64,
    band_energy: f64,
    nonlocal_energy: f64,
    entropy_energy: f64,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    local_cfg: local_potential.LocalPotentialConfig,
    ewald_cfg: config.EwaldConfig,
    use_rfft: bool,
    xc_func: xc.Functional,
    quiet: bool,
    coulomb_r_cut: ?f64,
    vdw_cfg: config.VdwConfig,
    rho_aug: ?[]const f64,
    ecutrho: ?f64,
) !EnergyTerms {
    // For PAW, use augmented density (ρ̃ + n̂hat) for E_H and E_xc to ensure
    // variational consistency with the potential used in the SCF eigenvalue problem.
    // For NC pseudopotentials, rho_aug is null and we use rho (pseudo density).
    const rho_for_hxc = rho_aug orelse rho;

    // E_xc and V_xc use augmented density
    const xc_fields = try computeXcFields(alloc, grid, rho_for_hxc, rho_core, use_rfft, xc_func);
    defer {
        alloc.free(xc_fields.vxc);
        alloc.free(xc_fields.exc);
    }
    const vxc_r = xc_fields.vxc;

    // Hartree energy via Term contract (uses augmented density for PAW).
    const hartree_input = term_mod.EvalInput{
        .alloc = alloc,
        .io = io,
        .species = species,
        .atoms = atoms,
        .cell_bohr = grid.cell,
        .recip = grid.recip,
        .volume_bohr = grid.volume,
        .rho = rho_for_hxc,
        .grid = &grid,
    };
    const eh = try term_mod.termEnergy(.{ .hartree = .{
        .isolated = (coulomb_r_cut != null),
        .ecutrho = ecutrho,
    } }, hartree_input);

    // Local pseudopotential energy via Term contract (pseudo density).
    const e_local = try term_mod.termEnergy(.{ .atomic_local = .{
        .mode = local_cfg.mode,
        .explicit_alpha = local_cfg.alpha,
        .ecutrho = ecutrho,
    } }, .{
        .alloc = alloc,
        .io = io,
        .species = species,
        .atoms = atoms,
        .cell_bohr = grid.cell,
        .recip = grid.recip,
        .volume_bohr = grid.volume,
        .rho = rho,
        .grid = &grid,
    });

    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var vxc_rho: f64 = 0.0;
    const exc = try term_mod.termEnergy(.{ .xc = .{ .functional = xc_func } }, .{
        .alloc = alloc,
        .io = io,
        .species = species,
        .atoms = atoms,
        .cell_bohr = grid.cell,
        .recip = grid.recip,
        .volume_bohr = grid.volume,
        .rho = rho_for_hxc,
        .rho_core = rho_core,
        .grid = &grid,
    });
    // vxc_rho = ∫V_xc(ρ_aug+core) × ρ_aug dr  (for PAW)
    //         = ∫V_xc(ρ+core) × ρ dr           (for NC)
    // For PAW: D^hat adds ∫V_Hxc × n̂ to band energy, so the double-counting
    // must subtract ∫V_xc × ρ_aug (not ρ̃) for correct cancellation.
    for (rho_for_hxc, 0..) |value, i| {
        vxc_rho += vxc_r[i] * value * dv;
    }

    // Ion-ion energy: Ewald for periodic, direct Coulomb for isolated.
    // Periodic Ewald is routed through the Term contract; direct Coulomb
    // stays on its legacy helper for now. Both return Hartree → Rydberg.
    const ion = if (coulomb_r_cut != null)
        try computeDirectIonIonEnergy(alloc, species, atoms) * 2.0
    else blk: {
        const ewald_term: term_mod.Term = .{ .ewald = .{
            .alpha = ewald_cfg.alpha,
            .rcut = ewald_cfg.rcut,
            .gcut = ewald_cfg.gcut,
            .tol = ewald_cfg.tol,
            .quiet = quiet,
        } };
        const input = term_mod.EvalInput{
            .alloc = alloc,
            .io = io,
            .species = species,
            .atoms = atoms,
            .cell_bohr = grid.cell,
            .recip = grid.recip,
            .volume_bohr = grid.volume,
        };
        break :blk (try term_mod.termEnergy(ewald_term, input)) * 2.0;
    };
    // E_psp_core = (n_elec / Ω) × Σ_atoms epsatm  (Rydberg units)
    // This is the volume-dependent correction from the G=0 local potential.
    // For isolated systems, this correction is not needed since G=0 is handled
    // explicitly by the cutoff Coulomb kernel.
    var epsatm_sum: f64 = 0.0;
    var n_elec: f64 = 0.0;
    for (atoms) |atom| {
        epsatm_sum += species[atom.species_index].epsatm_ry;
        n_elec += species[atom.species_index].z_valence;
    }
    const psp_core = if (coulomb_r_cut != null) 0.0 else epsatm_sum * n_elec / grid.volume;
    const double_counting = -eh - vxc_rho + exc;
    // DFT-D3 dispersion correction
    const e_disp = if (vdw_cfg.enabled) try computeDispersionEnergy(alloc, species, atoms, grid.cell, vdw_cfg) else 0.0;
    // Entropy term (-T*S) is subtracted from total energy for smearing
    const total = band_energy + double_counting + ion + psp_core - entropy_energy + e_disp;
    return EnergyTerms{
        .total = total,
        .band = band_energy,
        .hartree = eh,
        .vxc_rho = vxc_rho,
        .xc = exc,
        .ion_ion = ion,
        .psp_core = psp_core,
        .double_counting = double_counting,
        .local_pseudo = e_local,
        .nonlocal_pseudo = nonlocal_energy,
        .entropy = entropy_energy,
        .dispersion = e_disp,
    };
}

/// Compute energy terms for spin-polarized calculation.
/// If coulomb_r_cut is non-null, the cutoff Coulomb kernel is used for Hartree energy.
pub fn computeEnergyTermsSpin(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_core: ?[]const f64,
    band_energy: f64,
    nonlocal_energy: f64,
    entropy_energy: f64,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    local_cfg: local_potential.LocalPotentialConfig,
    ewald_cfg: config.EwaldConfig,
    use_rfft: bool,
    xc_func: xc.Functional,
    quiet: bool,
    coulomb_r_cut: ?f64,
    vdw_cfg: config.VdwConfig,
    rho_aug_up: ?[]const f64,
    rho_aug_down: ?[]const f64,
    ecutrho: ?f64,
) !EnergyTerms {
    const total = grid.count();

    // For PAW, use augmented density (ρ̃ + n̂) for E_H and E_xc
    const rho_up_hxc = rho_aug_up orelse rho_up;
    const rho_down_hxc = rho_aug_down orelse rho_down;

    // Compute rho_total for Hartree
    const rho_total = try alloc.alloc(f64, total);
    defer alloc.free(rho_total);
    for (0..total) |i| {
        rho_total[i] = rho_up_hxc[i] + rho_down_hxc[i];
    }

    // rho_g uses augmented density for E_H
    const rho_g = try realToReciprocal(alloc, grid, rho_total, use_rfft);
    defer alloc.free(rho_g);

    // For e_local, use pseudo density (not augmented)
    const rho_pseudo_g = if (rho_aug_up != null) blk: {
        const rho_pseudo_total = try alloc.alloc(f64, total);
        defer alloc.free(rho_pseudo_total);
        for (0..total) |i| {
            rho_pseudo_total[i] = rho_up[i] + rho_down[i];
        }
        break :blk try realToReciprocal(alloc, grid, rho_pseudo_total, use_rfft);
    } else null;
    defer if (rho_pseudo_g) |g| alloc.free(g);
    const rho_g_for_eloc = rho_pseudo_g orelse rho_g;

    // Spin XC — use augmented density for PAW
    const xc_fields = try computeXcFieldsSpin(alloc, grid, rho_up_hxc, rho_down_hxc, rho_core, use_rfft, xc_func);
    defer {
        alloc.free(xc_fields.vxc_up);
        alloc.free(xc_fields.vxc_down);
        alloc.free(xc_fields.exc);
    }

    const inv_volume = 1.0 / grid.volume;
    var eh: f64 = 0.0;
    var e_local: f64 = 0.0;
    var it2 = gvec_iter.GVecIterator.init(grid);
    while (it2.next()) |g| {
        // ecutrho spherical cutoff: skip G-vectors beyond ecutrho
        const beyond_ecutrho = if (ecutrho) |ecut| g.g2 >= ecut else false;
        if (beyond_ecutrho) {
            continue;
        }
        const rho_val = rho_g[g.idx];
        const rho2 = rho_val.r * rho_val.r + rho_val.i * rho_val.i;
        // e_local uses pseudo density
        const rho_loc = rho_g_for_eloc[g.idx];
        if (coulomb_r_cut) |r_cut| {
            const g_mag = @sqrt(g.g2);
            const kernel = coulomb.cutoffCoulombEnergyKernel(g.g2, g_mag, r_cut);
            eh += 0.5 * kernel * rho2 * grid.volume;
        } else {
            if (g.gh == 0 and g.gk == 0 and g.gl == 0) {
                const vloc = try hamiltonian.ionicLocalPotential(g.gvec, species, atoms, inv_volume, local_cfg);
                e_local += rho_loc.r * vloc.r + rho_loc.i * vloc.i;
                continue;
            }
            if (g.g2 > 1e-12) {
                eh += 0.5 * 8.0 * std.math.pi * rho2 / g.g2 * grid.volume;
            }
        }
        const vloc = try hamiltonian.ionicLocalPotential(g.gvec, species, atoms, inv_volume, local_cfg);
        e_local += rho_loc.r * vloc.r + rho_loc.i * vloc.i;
    }
    e_local *= grid.volume;

    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var exc: f64 = 0.0;
    var vxc_rho: f64 = 0.0;
    for (xc_fields.exc) |e| {
        exc += e * dv;
    }
    // vxc_rho = ∫(V_xc_up * rho_up + V_xc_down * rho_down) dv — use augmented for PAW
    for (0..total) |i| {
        vxc_rho += (xc_fields.vxc_up[i] * rho_up_hxc[i] + xc_fields.vxc_down[i] * rho_down_hxc[i]) * dv;
    }

    const ion = if (coulomb_r_cut != null)
        try computeDirectIonIonEnergy(alloc, species, atoms) * 2.0
    else
        try computeIonIonEnergy(alloc, io, grid, species, atoms, ewald_cfg, quiet) * 2.0;
    var epsatm_sum: f64 = 0.0;
    var n_elec: f64 = 0.0;
    for (atoms) |atom| {
        epsatm_sum += species[atom.species_index].epsatm_ry;
        n_elec += species[atom.species_index].z_valence;
    }
    const psp_core = if (coulomb_r_cut != null) 0.0 else epsatm_sum * n_elec / grid.volume;
    const double_counting = -eh - vxc_rho + exc;
    const e_disp = if (vdw_cfg.enabled) try computeDispersionEnergy(alloc, species, atoms, grid.cell, vdw_cfg) else 0.0;
    const total_energy = band_energy + double_counting + ion + psp_core - entropy_energy + e_disp;
    return EnergyTerms{
        .total = total_energy,
        .band = band_energy,
        .hartree = eh,
        .vxc_rho = vxc_rho,
        .xc = exc,
        .ion_ion = ion,
        .psp_core = psp_core,
        .double_counting = double_counting,
        .local_pseudo = e_local,
        .nonlocal_pseudo = nonlocal_energy,
        .entropy = entropy_energy,
        .dispersion = e_disp,
    };
}

/// Compute ion-ion Ewald energy.
fn computeIonIonEnergy(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ewald_cfg: config.EwaldConfig,
    quiet: bool,
) !f64 {
    const count = atoms.len;
    if (count == 0) return 0.0;
    const charges = try alloc.alloc(f64, count);
    defer alloc.free(charges);
    const positions = try alloc.alloc(math.Vec3, count);
    defer alloc.free(positions);
    for (atoms, 0..) |atom, i| {
        charges[i] = species[atom.species_index].z_valence;
        positions[i] = atom.position;
    }
    const params = ewald.Params{
        .alpha = ewald_cfg.alpha,
        .rcut = ewald_cfg.rcut,
        .gcut = ewald_cfg.gcut,
        .tol = ewald_cfg.tol,
        .quiet = quiet,
    };
    return try ewald.ionIonEnergy(io, grid.cell, grid.recip, charges, positions, params);
}

/// Compute ion-ion energy by direct pairwise Coulomb sum (for isolated systems).
/// Returns energy in Hartree (same convention as Ewald).
fn computeDirectIonIonEnergy(
    alloc: std.mem.Allocator,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) !f64 {
    const count = atoms.len;
    if (count == 0) return 0.0;
    const charges = try alloc.alloc(f64, count);
    defer alloc.free(charges);
    const positions = try alloc.alloc(math.Vec3, count);
    defer alloc.free(positions);
    for (atoms, 0..) |atom, i| {
        charges[i] = species[atom.species_index].z_valence;
        positions[i] = atom.position;
    }
    return coulomb.directIonIonEnergy(charges, positions);
}

/// Compute <psi|V_nl|psi> for one band.
pub fn bandNonlocalEnergy(
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

/// Compute DFT-D3(BJ) dispersion energy from atomic data.
fn computeDispersionEnergy(
    alloc: std.mem.Allocator,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell: math.Mat3,
    vdw_cfg: config.VdwConfig,
) !f64 {
    const n_atoms = atoms.len;
    const atomic_numbers = try alloc.alloc(usize, n_atoms);
    defer alloc.free(atomic_numbers);
    const positions = try alloc.alloc(math.Vec3, n_atoms);
    defer alloc.free(positions);
    for (atoms, 0..) |atom, i| {
        atomic_numbers[i] = d3_params.atomicNumber(species[atom.species_index].symbol) orelse 0;
        positions[i] = atom.position;
    }

    const damping = getDampingParams(vdw_cfg);

    return try d3.computeEnergy(
        alloc,
        atomic_numbers,
        positions,
        cell,
        damping,
        vdw_cfg.cutoff_radius,
        vdw_cfg.cn_cutoff,
    );
}

/// Get damping parameters, applying user overrides if specified.
pub fn getDampingParams(vdw_cfg: config.VdwConfig) d3_params.DampingParams {
    var damping = d3_params.pbe_d3bj; // default
    if (vdw_cfg.s6) |v| damping.s6 = v;
    if (vdw_cfg.s8) |v| damping.s8 = v;
    if (vdw_cfg.a1) |v| damping.a1 = v;
    if (vdw_cfg.a2) |v| damping.a2 = v;
    return damping;
}
