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

const DensityEnergyTerms = struct {
    hartree: f64,
    local_pseudo: f64,
    vxc_rho: f64,
    xc: f64,
};

const IonicEnergyTerms = struct {
    ion_ion: f64,
    psp_core: f64,
    dispersion: f64,
};

const EnergyTermContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: *const Grid,
    model: term_mod.Model,

    fn init(
        alloc: std.mem.Allocator,
        io: std.Io,
        grid: *const Grid,
        species: []const hamiltonian.SpeciesEntry,
        atoms: []const hamiltonian.AtomData,
    ) EnergyTermContext {
        return .{
            .alloc = alloc,
            .io = io,
            .grid = grid,
            .model = .{
                .species = species,
                .atoms = atoms,
                .cell_bohr = grid.cell,
                .recip = grid.recip,
                .volume_bohr = grid.volume,
            },
        };
    }

    fn evalInput(self: *const EnergyTermContext, rho: ?[]const f64) term_mod.EvalInput {
        return .{
            .alloc = self.alloc,
            .io = self.io,
            .model = &self.model,
            .rho = rho,
            .grid = self.grid,
        };
    }
};

/// Inputs to the energy aggregator. Density and geometry describe the
/// current SCF state; the remaining fields carry configuration knobs.
pub const EnergyInput = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho: []f64,
    /// Core density (NLCC). Present only when a species uses NLCC.
    rho_core: ?[]const f64 = null,
    /// Augmented density for PAW; null for NC where rho itself is used.
    rho_aug: ?[]const f64 = null,
    band_energy: f64,
    nonlocal_energy: f64,
    entropy_energy: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    ewald_cfg: config.EwaldConfig,
    vdw_cfg: config.VdwConfig,
    xc_func: xc.Functional,
    use_rfft: bool = false,
    quiet: bool = false,
    /// Real-space Coulomb cutoff for isolated BC; null for periodic.
    coulomb_r_cut: ?f64 = null,
    /// PAW augmented-density spherical G² cutoff; null outside PAW.
    ecutrho: ?f64 = null,
};

/// Spin-polarized variant of EnergyInput.
pub const EnergyInputSpin = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_core: ?[]const f64 = null,
    rho_aug_up: ?[]const f64 = null,
    rho_aug_down: ?[]const f64 = null,
    band_energy: f64,
    nonlocal_energy: f64,
    entropy_energy: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    ewald_cfg: config.EwaldConfig,
    vdw_cfg: config.VdwConfig,
    xc_func: xc.Functional,
    use_rfft: bool = false,
    quiet: bool = false,
    coulomb_r_cut: ?f64 = null,
    ecutrho: ?f64 = null,
};

/// Compute energy terms (Hartree + XC) from density.
/// If coulomb_r_cut is non-null, the cutoff Coulomb kernel is used for Hartree energy.
pub fn computeEnergyTerms(in: EnergyInput) !EnergyTerms {
    // For PAW, use augmented density (ρ̃ + n̂hat) for E_H and E_xc to ensure
    // variational consistency with the potential used in the SCF eigenvalue problem.
    // For NC pseudopotentials, rho_aug is null and we use rho (pseudo density).
    const rho_for_hxc = in.rho_aug orelse in.rho;
    const ctx = EnergyTermContext.init(in.alloc, in.io, &in.grid, in.species, in.atoms);
    const density = try computeScalarDensityEnergyTerms(
        &ctx,
        rho_for_hxc,
        in.rho,
        in.rho_core,
        in.local_cfg,
        in.use_rfft,
        in.xc_func,
        in.coulomb_r_cut,
        in.ecutrho,
    );
    const ionic = try computeIonicEnergyTerms(
        &ctx,
        in.ewald_cfg,
        in.vdw_cfg,
        in.quiet,
        in.coulomb_r_cut,
    );
    return assembleEnergyTerms(
        in.band_energy,
        in.nonlocal_energy,
        in.entropy_energy,
        density,
        ionic,
    );
}

/// Compute energy terms for spin-polarized calculation.
/// If coulomb_r_cut is non-null, the cutoff Coulomb kernel is used for Hartree energy.
pub fn computeEnergyTermsSpin(in: EnergyInputSpin) !EnergyTerms {
    // For PAW, use augmented density (ρ̃ + n̂) for E_H and E_xc
    const rho_up_hxc = in.rho_aug_up orelse in.rho_up;
    const rho_down_hxc = in.rho_aug_down orelse in.rho_down;
    const ctx = EnergyTermContext.init(in.alloc, in.io, &in.grid, in.species, in.atoms);
    const density = try computeSpinDensityEnergyTerms(
        &ctx,
        in.rho_up,
        in.rho_down,
        rho_up_hxc,
        rho_down_hxc,
        in.rho_core,
        in.rho_aug_up != null or in.rho_aug_down != null,
        in.local_cfg,
        in.use_rfft,
        in.xc_func,
        in.coulomb_r_cut,
        in.ecutrho,
    );
    const ionic = try computeIonicEnergyTerms(
        &ctx,
        in.ewald_cfg,
        in.vdw_cfg,
        in.quiet,
        in.coulomb_r_cut,
    );
    return assembleEnergyTerms(
        in.band_energy,
        in.nonlocal_energy,
        in.entropy_energy,
        density,
        ionic,
    );
}

fn computeScalarDensityEnergyTerms(
    ctx: *const EnergyTermContext,
    rho_for_hxc: []const f64,
    rho: []const f64,
    rho_core: ?[]const f64,
    local_cfg: local_potential.LocalPotentialConfig,
    use_rfft: bool,
    xc_func: xc.Functional,
    coulomb_r_cut: ?f64,
    ecutrho: ?f64,
) !DensityEnergyTerms {
    const xc_fields = try computeXcFields(
        ctx.alloc,
        ctx.grid.*,
        rho_for_hxc,
        rho_core,
        use_rfft,
        xc_func,
    );
    defer {
        ctx.alloc.free(xc_fields.vxc);
        ctx.alloc.free(xc_fields.exc);
    }

    const dv = ctx.grid.volume / @as(f64, @floatFromInt(ctx.grid.count()));
    return .{
        .hartree = try computeHartreeEnergy(ctx, rho_for_hxc, coulomb_r_cut, ecutrho),
        .local_pseudo = try computeLocalPseudoEnergy(ctx, rho, local_cfg, ecutrho),
        .vxc_rho = integrateScalarVxcRho(xc_fields.vxc, rho_for_hxc, dv),
        .xc = integrateExcDensity(xc_fields.exc, dv),
    };
}

fn computeSpinDensityEnergyTerms(
    ctx: *const EnergyTermContext,
    rho_up: []const f64,
    rho_down: []const f64,
    rho_up_hxc: []const f64,
    rho_down_hxc: []const f64,
    rho_core: ?[]const f64,
    has_augmented_density: bool,
    local_cfg: local_potential.LocalPotentialConfig,
    use_rfft: bool,
    xc_func: xc.Functional,
    coulomb_r_cut: ?f64,
    ecutrho: ?f64,
) !DensityEnergyTerms {
    const rho_total = try buildDensitySum(ctx.alloc, rho_up_hxc, rho_down_hxc);
    defer ctx.alloc.free(rho_total);

    const rho_pseudo_total = try maybeBuildPseudoDensitySum(
        ctx.alloc,
        rho_up,
        rho_down,
        has_augmented_density,
    );
    defer if (rho_pseudo_total) |buf| ctx.alloc.free(buf);

    const xc_fields = try computeXcFieldsSpin(
        ctx.alloc,
        ctx.grid.*,
        rho_up_hxc,
        rho_down_hxc,
        rho_core,
        use_rfft,
        xc_func,
    );
    defer {
        ctx.alloc.free(xc_fields.vxc_up);
        ctx.alloc.free(xc_fields.vxc_down);
        ctx.alloc.free(xc_fields.exc);
    }

    const dv = ctx.grid.volume / @as(f64, @floatFromInt(ctx.grid.count()));
    return .{
        .hartree = try computeHartreeEnergy(ctx, rho_total, coulomb_r_cut, ecutrho),
        .local_pseudo = try computeLocalPseudoEnergy(
            ctx,
            rho_pseudo_total orelse rho_total,
            local_cfg,
            ecutrho,
        ),
        .vxc_rho = integrateSpinVxcRho(
            xc_fields.vxc_up,
            xc_fields.vxc_down,
            rho_up_hxc,
            rho_down_hxc,
            dv,
        ),
        .xc = integrateExcDensity(xc_fields.exc, dv),
    };
}

fn computeHartreeEnergy(
    ctx: *const EnergyTermContext,
    rho: []const f64,
    coulomb_r_cut: ?f64,
    ecutrho: ?f64,
) !f64 {
    return try term_mod.termEnergy(.{ .hartree = .{
        .isolated = (coulomb_r_cut != null),
        .ecutrho = ecutrho,
    } }, ctx.evalInput(rho));
}

fn computeLocalPseudoEnergy(
    ctx: *const EnergyTermContext,
    rho: []const f64,
    local_cfg: local_potential.LocalPotentialConfig,
    ecutrho: ?f64,
) !f64 {
    return try term_mod.termEnergy(.{ .atomic_local = .{
        .mode = local_cfg.mode,
        .explicit_alpha = local_cfg.alpha,
        .ecutrho = ecutrho,
    } }, ctx.evalInput(rho));
}

fn computeIonicEnergyTerms(
    ctx: *const EnergyTermContext,
    ewald_cfg: config.EwaldConfig,
    vdw_cfg: config.VdwConfig,
    quiet: bool,
    coulomb_r_cut: ?f64,
) !IonicEnergyTerms {
    const ion_ion = if (coulomb_r_cut != null)
        try computeDirectIonIonEnergy(ctx.alloc, ctx.model.species, ctx.model.atoms) * 2.0
    else
        (try term_mod.termEnergy(.{ .ewald = .{
            .alpha = ewald_cfg.alpha,
            .rcut = ewald_cfg.rcut,
            .gcut = ewald_cfg.gcut,
            .tol = ewald_cfg.tol,
            .quiet = quiet,
        } }, ctx.evalInput(null))) * 2.0;
    const epsatm = sumAtomicPotentialOffsets(ctx.model.species, ctx.model.atoms);
    return .{
        .ion_ion = ion_ion,
        .psp_core = if (coulomb_r_cut != null)
            0.0
        else
            epsatm.epsatm_sum * epsatm.n_elec / ctx.grid.volume,
        .dispersion = if (vdw_cfg.enabled)
            try computeDispersionEnergy(
                ctx.alloc,
                ctx.model.species,
                ctx.model.atoms,
                ctx.grid.cell,
                vdw_cfg,
            )
        else
            0.0,
    };
}

fn assembleEnergyTerms(
    band_energy: f64,
    nonlocal_energy: f64,
    entropy_energy: f64,
    density: DensityEnergyTerms,
    ionic: IonicEnergyTerms,
) EnergyTerms {
    const double_counting = -density.hartree - density.vxc_rho + density.xc;
    const total = band_energy + double_counting + ionic.ion_ion + ionic.psp_core -
        entropy_energy + ionic.dispersion;
    return .{
        .total = total,
        .band = band_energy,
        .hartree = density.hartree,
        .vxc_rho = density.vxc_rho,
        .xc = density.xc,
        .ion_ion = ionic.ion_ion,
        .psp_core = ionic.psp_core,
        .double_counting = double_counting,
        .local_pseudo = density.local_pseudo,
        .nonlocal_pseudo = nonlocal_energy,
        .entropy = entropy_energy,
        .dispersion = ionic.dispersion,
    };
}

fn buildDensitySum(
    alloc: std.mem.Allocator,
    lhs: []const f64,
    rhs: []const f64,
) ![]f64 {
    const sum = try alloc.alloc(f64, lhs.len);
    for (0..lhs.len) |i| {
        sum[i] = lhs[i] + rhs[i];
    }
    return sum;
}

fn maybeBuildPseudoDensitySum(
    alloc: std.mem.Allocator,
    rho_up: []const f64,
    rho_down: []const f64,
    has_augmented_density: bool,
) !?[]f64 {
    if (!has_augmented_density) return null;
    return try buildDensitySum(alloc, rho_up, rho_down);
}

fn integrateExcDensity(exc_density: []const f64, dv: f64) f64 {
    // Integrate exc directly to avoid a second computeXcFields call via termEnergy(.xc).
    var exc: f64 = 0.0;
    for (exc_density) |value| exc += value * dv;
    return exc;
}

fn integrateScalarVxcRho(vxc: []const f64, rho: []const f64, dv: f64) f64 {
    // vxc_rho = ∫V_xc(ρ_aug+core) × ρ_aug dr  (for PAW)
    //         = ∫V_xc(ρ+core) × ρ dr           (for NC)
    var vxc_rho: f64 = 0.0;
    for (rho, 0..) |value, i| {
        vxc_rho += vxc[i] * value * dv;
    }
    return vxc_rho;
}

fn integrateSpinVxcRho(
    vxc_up: []const f64,
    vxc_down: []const f64,
    rho_up: []const f64,
    rho_down: []const f64,
    dv: f64,
) f64 {
    var vxc_rho: f64 = 0.0;
    for (0..rho_up.len) |i| {
        vxc_rho += (vxc_up[i] * rho_up[i] + vxc_down[i] * rho_down[i]) * dv;
    }
    return vxc_rho;
}

const AtomicPotentialOffsets = struct {
    epsatm_sum: f64,
    n_elec: f64,
};

fn sumAtomicPotentialOffsets(
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) AtomicPotentialOffsets {
    var epsatm_sum: f64 = 0.0;
    var n_elec: f64 = 0.0;
    for (atoms) |atom| {
        epsatm_sum += species[atom.species_index].epsatm_ry;
        n_elec += species[atom.species_index].z_valence;
    }
    return .{ .epsatm_sum = epsatm_sum, .n_elec = n_elec };
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
