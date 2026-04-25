const std = @import("std");
const apply = @import("apply.zig");
const config = @import("../config/config.zig");
const fft = @import("../fft/fft.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoint = @import("kpoint.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const potential_mod = @import("potential.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const util = @import("util.zig");

const Grid = grid_mod.Grid;
const KPoint = symmetry.KPoint;
const KpointCache = kpoint.KpointCache;
const KpointEigenData = kpoint.KpointEigenData;
const build_fft_index_map = fft_grid.build_fft_index_map;
const compute_kpoint_eigen_data = kpoint.solve.compute_kpoint_eigen_data;
const log_local_potential_mean = logging.log_local_potential_mean;
const has_nonlocal = util.has_nonlocal;
const has_qij = util.has_qij;
const has_paw = util.has_paw;
const total_electrons = util.total_electrons;

/// Wavefunction data for a single k-point.
pub const KpointWavefunction = struct {
    k_frac: math.Vec3,
    k_cart: math.Vec3,
    weight: f64,
    basis_len: usize,
    nbands: usize,
    eigenvalues: []f64,
    coefficients: []math.Complex,
    occupations: []f64,

    pub fn deinit(self: *KpointWavefunction, alloc: std.mem.Allocator) void {
        if (self.eigenvalues.len > 0) alloc.free(self.eigenvalues);
        if (self.coefficients.len > 0) alloc.free(self.coefficients);
        if (self.occupations.len > 0) alloc.free(self.occupations);
    }
};

/// Wavefunction data for all k-points (for force calculation).
pub const WavefunctionData = struct {
    kpoints: []KpointWavefunction,
    ecut_ry: f64,
    fermi_level: f64,

    pub fn deinit(self: *WavefunctionData, alloc: std.mem.Allocator) void {
        for (self.kpoints) |*kp| {
            kp.deinit(alloc);
        }
        if (self.kpoints.len > 0) alloc.free(self.kpoints);
    }
};

pub const WavefunctionResult = struct {
    wavefunctions: WavefunctionData,
    band_energy: f64,
    nonlocal_energy: f64,
};

const FinalWavefunctionSetup = struct {
    nocc: usize,
    local_cfg: local_potential.LocalPotentialConfig,
    use_iterative_config: bool,
    nonlocal_enabled: bool,
    fft_index_map: []usize,
    local_r: ?[]f64,
    iter_max_iter: usize,
    iter_tol: f64,

    fn deinit(self: *const FinalWavefunctionSetup, alloc: std.mem.Allocator) void {
        alloc.free(self.fft_index_map);
        if (self.local_r) |values| alloc.free(values);
    }
};

fn init_final_wavefunction_setup(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: Grid,
    ionic: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    potential: hamiltonian.PotentialGrid,
) !FinalWavefunctionSetup {
    const is_paw_wf = has_paw(species);
    const qij_enabled = has_qij(species) and !is_paw_wf;
    const use_iterative_config = (cfg.scf.solver == .iterative or
        cfg.scf.solver == .cg or
        cfg.scf.solver == .auto) and !qij_enabled;
    const fft_index_map = try build_fft_index_map(alloc, grid);
    var local_r: ?[]f64 = null;
    if (use_iterative_config) {
        local_r = try potential_mod.build_local_potential_real(alloc, grid, ionic, potential);
    }
    return .{
        .nocc = @as(usize, @intFromFloat(std.math.ceil(total_electrons(species, atoms) / 2.0))),
        .local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, grid.cell),
        .use_iterative_config = use_iterative_config,
        .nonlocal_enabled = cfg.scf.enable_nonlocal and has_nonlocal(species),
        .fft_index_map = fft_index_map,
        .local_r = local_r,
        .iter_max_iter = cfg.scf.iterative_max_iter,
        .iter_tol = cfg.scf.iterative_tol,
    };
}

fn maybe_log_final_wavefunction_potential_mean(
    io: std.Io,
    cfg: *const config.Config,
    potential: hamiltonian.PotentialGrid,
    local_r: ?[]const f64,
) !void {
    if (!cfg.scf.debug_fermi) return;
    const values = local_r orelse return;
    var sum: f64 = 0.0;
    for (values) |v| sum += v;
    const mean_local = sum / @as(f64, @floatFromInt(values.len));
    const pot_g0 = potential.value_at(0, 0, 0);
    try log_local_potential_mean(io, "scf", mean_local, "pot_g0", pot_g0.r);
}

fn fill_occupied_bands(
    occupations: []f64,
    eigen_data: KpointEigenData,
    kp: KPoint,
    nocc: usize,
    spin_factor: f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
) void {
    @memset(occupations, 0.0);
    var band: usize = 0;
    while (band < @min(nocc, eigen_data.nbands)) : (band += 1) {
        occupations[band] = 1.0;
        band_energy.* += kp.weight * spin_factor * eigen_data.values[band];
        if (eigen_data.nonlocal) |nl| {
            nonlocal_energy.* += kp.weight * spin_factor * nl[band];
        }
    }
}

fn compute_final_kpoint_eigen_data(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    setup: *const FinalWavefunctionSetup,
    kpoint_cache: *KpointCache,
    apply_cache: *apply.KpointApplyCache,
    wf_fft_plan: fft.Fft3dPlan,
) !KpointEigenData {
    return compute_kpoint_eigen_data(
        alloc,
        io,
        cfg,
        grid,
        kp,
        species,
        atoms,
        recip,
        volume,
        setup.local_cfg,
        potential,
        setup.local_r,
        setup.nocc,
        setup.use_iterative_config,
        has_qij(species) and !has_paw(species),
        setup.nonlocal_enabled,
        setup.fft_index_map,
        setup.iter_max_iter,
        setup.iter_tol,
        cfg.scf.iterative_reuse_vectors,
        kpoint_cache,
        null,
        wf_fft_plan,
        apply_cache,
        radial_tables,
        paw_tabs,
    );
}

fn compute_final_kpoint_wavefunction(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kp: KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    setup: *const FinalWavefunctionSetup,
    kpoint_cache: *KpointCache,
    apply_cache: *apply.KpointApplyCache,
    wf_fft_plan: fft.Fft3dPlan,
    spin_factor: f64,
    band_energy: *f64,
    nonlocal_energy: *f64,
) !KpointWavefunction {
    const eigen_data = try compute_final_kpoint_eigen_data(
        alloc,
        io,
        cfg,
        grid,
        kp,
        species,
        atoms,
        recip,
        volume,
        potential,
        radial_tables,
        paw_tabs,
        setup,
        kpoint_cache,
        apply_cache,
        wf_fft_plan,
    );
    errdefer {
        var ed = eigen_data;
        ed.deinit(alloc);
    }

    const occupations = try alloc.alloc(f64, eigen_data.nbands);
    errdefer alloc.free(occupations);

    fill_occupied_bands(
        occupations,
        eigen_data,
        kp,
        setup.nocc,
        spin_factor,
        band_energy,
        nonlocal_energy,
    );
    const wavefunction = KpointWavefunction{
        .k_frac = kp.k_frac,
        .k_cart = kp.k_cart,
        .weight = kp.weight,
        .basis_len = eigen_data.basis_len,
        .nbands = eigen_data.nbands,
        .eigenvalues = eigen_data.values,
        .coefficients = eigen_data.vectors,
        .occupations = occupations,
    };
    if (eigen_data.nonlocal) |nl| alloc.free(nl);
    return wavefunction;
}

fn find_wavefunction_fermi_level(kp_wavefunctions: []const KpointWavefunction) f64 {
    var fermi_level: f64 = -std.math.inf(f64);
    for (kp_wavefunctions) |kw| {
        for (kw.eigenvalues, 0..) |e, band| {
            if (kw.occupations[band] > 0.0) {
                fermi_level = @max(fermi_level, e);
            }
        }
    }
    return fermi_level;
}

pub const FinalWavefunctionParams = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    grid: Grid,
    kpoints: []KPoint,
    ionic: hamiltonian.PotentialGrid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    spin_factor: f64,
};

/// Compute final wavefunctions for force calculation.
pub fn compute_final_wavefunctions_with_spin_factor(
    params: FinalWavefunctionParams,
) !WavefunctionResult {
    var setup = try init_logged_final_wavefunction_setup(params);
    defer setup.deinit(params.alloc);

    var kp_wavefunctions = try params.alloc.alloc(KpointWavefunction, params.kpoints.len);
    var filled: usize = 0;
    errdefer {
        for (kp_wavefunctions[0..filled]) |*kw| {
            kw.deinit(params.alloc);
        }
        params.alloc.free(kp_wavefunctions);
    }
    var wf_fft_plan = try fft.Fft3dPlan.init_with_backend(
        params.alloc,
        params.io,
        params.grid.nx,
        params.grid.ny,
        params.grid.nz,
        params.cfg.scf.fft_backend,
    );
    defer wf_fft_plan.deinit(params.alloc);

    var band_energy: f64 = 0.0;
    var nonlocal_energy: f64 = 0.0;
    const fill_input = final_wavefunction_fill_input(
        params.alloc,
        params.io,
        &params.cfg,
        params.grid,
        params.kpoints,
        params.species,
        params.atoms,
        params.recip,
        params.volume,
        params.potential,
        params.radial_tables,
        params.paw_tabs,
        &setup,
        params.kpoint_cache,
        params.apply_caches,
        wf_fft_plan,
        params.spin_factor,
        kp_wavefunctions,
        &band_energy,
        &nonlocal_energy,
    );
    filled = try fill_final_kpoint_wavefunctions(fill_input);
    return WavefunctionResult{
        .wavefunctions = WavefunctionData{
            .kpoints = kp_wavefunctions,
            .ecut_ry = params.cfg.scf.ecut_ry,
            .fermi_level = find_wavefunction_fermi_level(kp_wavefunctions),
        },
        .band_energy = band_energy,
        .nonlocal_energy = nonlocal_energy,
    };
}

fn init_logged_final_wavefunction_setup(
    params: FinalWavefunctionParams,
) !FinalWavefunctionSetup {
    const setup = try init_final_wavefunction_setup(
        params.alloc,
        &params.cfg,
        params.grid,
        params.ionic,
        params.species,
        params.atoms,
        params.potential,
    );
    try maybe_log_final_wavefunction_potential_mean(
        params.io,
        &params.cfg,
        params.potential,
        setup.local_r,
    );
    return setup;
}

const FinalWavefunctionFill = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    setup: *const FinalWavefunctionSetup,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    wf_fft_plan: fft.Fft3dPlan,
    spin_factor: f64,
    wavefunctions: []KpointWavefunction,
    band_energy: *f64,
    nonlocal_energy: *f64,
};

fn final_wavefunction_fill_input(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: *const config.Config,
    grid: Grid,
    kpoints: []KPoint,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    potential: hamiltonian.PotentialGrid,
    radial_tables: ?[]const nonlocal_mod.RadialTableSet,
    paw_tabs: ?[]const paw_mod.PawTab,
    setup: *const FinalWavefunctionSetup,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    wf_fft_plan: fft.Fft3dPlan,
    spin_factor: f64,
    wavefunctions: []KpointWavefunction,
    band_energy: *f64,
    nonlocal_energy: *f64,
) FinalWavefunctionFill {
    return .{
        .alloc = alloc,
        .io = io,
        .cfg = cfg,
        .grid = grid,
        .kpoints = kpoints,
        .species = species,
        .atoms = atoms,
        .recip = recip,
        .volume = volume,
        .potential = potential,
        .radial_tables = radial_tables,
        .paw_tabs = paw_tabs,
        .setup = setup,
        .kpoint_cache = kpoint_cache,
        .apply_caches = apply_caches,
        .wf_fft_plan = wf_fft_plan,
        .spin_factor = spin_factor,
        .wavefunctions = wavefunctions,
        .band_energy = band_energy,
        .nonlocal_energy = nonlocal_energy,
    };
}

fn fill_final_kpoint_wavefunctions(input: FinalWavefunctionFill) !usize {
    var filled: usize = 0;
    for (input.kpoints, 0..) |kp, kidx| {
        input.wavefunctions[kidx] = try compute_final_kpoint_wavefunction(
            input.alloc,
            input.io,
            input.cfg,
            input.grid,
            kp,
            input.species,
            input.atoms,
            input.recip,
            input.volume,
            input.potential,
            input.radial_tables,
            input.paw_tabs,
            input.setup,
            &input.kpoint_cache[kidx],
            &input.apply_caches[kidx],
            input.wf_fft_plan,
            input.spin_factor,
            input.band_energy,
            input.nonlocal_energy,
        );
        filled += 1;
    }
    return filled;
}
