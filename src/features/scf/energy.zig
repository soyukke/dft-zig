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
    species: []const hamiltonian.SpeciesEntry,
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

    const model = term_mod.Model{
        .species = species,
        .atoms = atoms,
        .cell_bohr = grid.cell,
        .recip = grid.recip,
        .volume_bohr = grid.volume,
    };
    const shared_input = term_mod.EvalInput{
        .alloc = alloc,
        .io = io,
        .model = &model,
        .grid = &grid,
    };

    // Hartree energy via Term contract (uses augmented density for PAW).
    var hartree_input = shared_input;
    hartree_input.rho = rho_for_hxc;
    const eh = try term_mod.termEnergy(.{ .hartree = .{
        .isolated = (coulomb_r_cut != null),
        .ecutrho = ecutrho,
    } }, hartree_input);

    // Local pseudopotential energy via Term contract (pseudo density).
    var local_input = shared_input;
    local_input.rho = rho;
    const e_local = try term_mod.termEnergy(.{ .atomic_local = .{
        .mode = local_cfg.mode,
        .explicit_alpha = local_cfg.alpha,
        .ecutrho = ecutrho,
    } }, local_input);

    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var vxc_rho: f64 = 0.0;
    // E_xc integrates the exc density from the computeXcFields call above.
    // Going through termEnergy(.xc) would run computeXcFields a second time
    // (measured: ~11% overhead on Cu/PBE). termEnergy(.xc) is kept for
    // standalone use (tests, future term-driven aggregation).
    var exc: f64 = 0.0;
    for (xc_fields.exc) |e| exc += e * dv;
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
    else (try term_mod.termEnergy(.{ .ewald = .{
        .alpha = ewald_cfg.alpha,
        .rcut = ewald_cfg.rcut,
        .gcut = ewald_cfg.gcut,
        .tol = ewald_cfg.tol,
        .quiet = quiet,
    } }, shared_input)) * 2.0;
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
    species: []const hamiltonian.SpeciesEntry,
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

    // Spin XC — use augmented density for PAW
    const xc_fields = try computeXcFieldsSpin(alloc, grid, rho_up_hxc, rho_down_hxc, rho_core, use_rfft, xc_func);
    defer {
        alloc.free(xc_fields.vxc_up);
        alloc.free(xc_fields.vxc_down);
        alloc.free(xc_fields.exc);
    }

    const model = term_mod.Model{
        .species = species,
        .atoms = atoms,
        .cell_bohr = grid.cell,
        .recip = grid.recip,
        .volume_bohr = grid.volume,
    };
    const shared_input = term_mod.EvalInput{
        .alloc = alloc,
        .io = io,
        .model = &model,
        .grid = &grid,
    };

    // Hartree uses the augmented total density.
    var hartree_input = shared_input;
    hartree_input.rho = rho_total;
    const eh = try term_mod.termEnergy(.{ .hartree = .{
        .isolated = (coulomb_r_cut != null),
        .ecutrho = ecutrho,
    } }, hartree_input);

    // Local pseudo uses the pseudo total density (not augmented).
    const rho_pseudo_total = if (rho_aug_up != null) blk: {
        const buf = try alloc.alloc(f64, total);
        for (0..total) |i| buf[i] = rho_up[i] + rho_down[i];
        break :blk buf;
    } else null;
    defer if (rho_pseudo_total) |buf| alloc.free(buf);
    var local_input = shared_input;
    local_input.rho = rho_pseudo_total orelse rho_total;
    const e_local = try term_mod.termEnergy(.{ .atomic_local = .{
        .mode = local_cfg.mode,
        .explicit_alpha = local_cfg.alpha,
        .ecutrho = ecutrho,
    } }, local_input);

    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var vxc_rho: f64 = 0.0;
    // Integrate exc directly from xc_fields (see computeEnergyTerms comment).
    var exc: f64 = 0.0;
    for (xc_fields.exc) |e| exc += e * dv;
    // vxc_rho = ∫(V_xc_up * rho_up + V_xc_down * rho_down) dv — use augmented for PAW
    for (0..total) |i| {
        vxc_rho += (xc_fields.vxc_up[i] * rho_up_hxc[i] + xc_fields.vxc_down[i] * rho_down_hxc[i]) * dv;
    }

    const ion = if (coulomb_r_cut != null)
        try computeDirectIonIonEnergy(alloc, species, atoms) * 2.0
    else (try term_mod.termEnergy(.{ .ewald = .{
        .alpha = ewald_cfg.alpha,
        .rcut = ewald_cfg.rcut,
        .gcut = ewald_cfg.gcut,
        .tol = ewald_cfg.tol,
        .quiet = quiet,
    } }, shared_input)) * 2.0;
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

/// Compute ion-ion energy by direct pairwise Coulomb sum (for isolated systems).
/// Returns energy in Hartree (same convention as Ewald).
fn computeDirectIonIonEnergy(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
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
    species: []const hamiltonian.SpeciesEntry,
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
