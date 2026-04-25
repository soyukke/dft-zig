const std = @import("std");
const apply = @import("apply.zig");
const config = @import("../config/config.zig");
const common_mod = @import("common.zig");
const core_density = @import("core_density.zig");
const density_mod = @import("density.zig");
const density_symmetry = @import("density_symmetry.zig");
const energy_mod = @import("energy.zig");
const final_wavefunction = @import("final_wavefunction.zig");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoint = @import("kpoint.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const mixing = @import("mixing.zig");
const model_mod = @import("../dft/model.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const paw_results_mod = @import("paw_results.zig");
const paw_scf = @import("paw_scf.zig");
const result_mod = @import("result.zig");
const run_state_mod = @import("run_state.zig");
const potential_mod = @import("potential.zig");
const xc_fields_mod = @import("xc_fields.zig");
const pw_grid_map = @import("pw_grid_map.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const thread_pool = @import("../thread_pool.zig");
const util = @import("util.zig");
const band_solver = @import("band_solver.zig");
const gvec_iter = @import("gvec_iter.zig");
const scf_spin = @import("scf_spin.zig");

pub const ThreadPool = thread_pool.ThreadPool;

const KPoint = symmetry.KPoint;

pub const Grid = grid_mod.Grid;

pub const ApplyContext = apply.ApplyContext;
pub const apply_hamiltonian = apply.apply_hamiltonian;
pub const apply_hamiltonian_batched = apply.apply_hamiltonian_batched;
pub const KpointApplyCache = apply.KpointApplyCache;
pub const NonlocalContext = apply.NonlocalContext;
pub const NonlocalSpecies = apply.NonlocalSpecies;
pub const apply_nonlocal_potential = apply.apply_nonlocal_potential;

pub const KpointCache = kpoint.KpointCache;
pub const kpoint_thread_count = kpoint.workers.kpoint_thread_count;

pub const real_to_reciprocal = fft_grid.real_to_reciprocal;
pub const reciprocal_to_real = fft_grid.reciprocal_to_real;
pub const fft_reciprocal_to_complex_in_place = fft_grid.fft_reciprocal_to_complex_in_place;
const fft_reciprocal_to_complex_in_place_mapped =
    fft_grid.fft_reciprocal_to_complex_in_place_mapped;
pub const fft_complex_to_reciprocal_in_place = fft_grid.fft_complex_to_reciprocal_in_place;
const fft_complex_to_reciprocal_in_place_mapped =
    fft_grid.fft_complex_to_reciprocal_in_place_mapped;
const index_to_freq = fft_grid.index_to_freq;

const mix_density = mixing.mix_density;
const mix_density_kerker = mixing.mix_density_kerker;
pub const ComplexPulayMixer = mixing.ComplexPulayMixer;

const ScfProfile = logging.ScfProfile;
const log_progress = logging.log_progress;
const log_iter_start = logging.log_iter_start;
const log_loop_profile = logging.log_loop_profile;
const log_energy_summary = logging.log_energy_summary;
const profile_start = logging.profile_start;
const profile_add = logging.profile_add;
const ScfLoopProfile = density_mod.ScfLoopProfile;

pub const PwGridMap = pw_grid_map.PwGridMap;

// gvec_iter re-exports
pub const GVecIterator = gvec_iter.GVecIterator;
const GVecItem = gvec_iter.GVecItem;

// xc_fields re-exports
const XcFields = xc_fields_mod.XcFields;
const XcFieldsSpin = xc_fields_mod.XcFieldsSpin;
const Gradient = xc_fields_mod.Gradient;
pub const compute_xc_fields = xc_fields_mod.compute_xc_fields;
const compute_xc_fields_spin = xc_fields_mod.compute_xc_fields_spin;
pub const gradient_from_real = xc_fields_mod.gradient_from_real;
pub const divergence_from_real = xc_fields_mod.divergence_from_real;

// potential re-exports
const build_potential_grid = potential_mod.build_potential_grid;
const build_potential_grid_spin = potential_mod.build_potential_grid_spin;
pub const build_ionic_potential_grid = potential_mod.build_ionic_potential_grid;
pub const build_local_potential_real = potential_mod.build_local_potential_real;
const filter_density_to_ecutrho = potential_mod.filter_density_to_ecutrho;
const SpinPotentialGrids = potential_mod.SpinPotentialGrids;

// core_density re-exports
pub const has_nlcc = core_density.has_nlcc;
pub const build_core_density = core_density.build_core_density;

pub const KpointWavefunction = final_wavefunction.KpointWavefunction;
pub const WavefunctionData = final_wavefunction.WavefunctionData;
pub const FinalWavefunctionParams = final_wavefunction.FinalWavefunctionParams;
pub const compute_final_wavefunctions_with_spin_factor =
    final_wavefunction.compute_final_wavefunctions_with_spin_factor;

pub const ScfResult = result_mod.ScfResult;

pub const EnergyTerms = energy_mod.EnergyTerms;

pub const DensityResult = density_mod.DensityResult;

// Band solver re-exports
pub const BandIterativeContext = band_solver.BandIterativeContext;
pub const BandVectorCache = band_solver.BandVectorCache;
pub const BandEigenOptions = band_solver.BandEigenOptions;
pub const init_band_iterative_context = band_solver.init_band_iterative_context;
pub const band_eigenvalues_iterative = band_solver.band_eigenvalues_iterative;
pub const band_eigenvalues_iterative_ext = band_solver.band_eigenvalues_iterative_ext;

pub const symmetrize_density = density_symmetry.symmetrize_density;

const ScfLoopProf = run_state_mod.ScfLoopProf;
const ScfRunCaches = run_state_mod.ScfRunCaches;
const ScfRunState = run_state_mod.ScfRunState;
const ScfRunContext = run_state_mod.ScfRunContext;
const ScfIterationDensity = run_state_mod.ScfIterationDensity;
const ScfIterationPotential = run_state_mod.ScfIterationPotential;
const build_scf_result = result_mod.build_scf_result;
const FinalPawResults = paw_results_mod.FinalPawResults;

/// Parameters for SCF calculation.
pub const ScfParams = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config.Config,
    model: *const model_mod.Model,
    initial_density: ?[]const f64 = null,
    initial_kpoint_cache: ?[]KpointCache = null,
    initial_apply_caches: ?[]apply.KpointApplyCache = null,
    ff_tables: ?[]const form_factor.LocalFormFactorTable = null,
};

pub const ScfCommon = common_mod.ScfCommon;

/// Potential-mixing branch: mixes V_in and V_out, frees potential_out.
fn mix_potential_mode(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: grid_mod.Grid,
    common: *ScfCommon,
    iterations: usize,
    rho: []f64,
    new_rho: []const f64,
    potential: *hamiltonian.PotentialGrid,
    potential_out: *hamiltonian.PotentialGrid,
    keep_potential_out: *bool,
) !void {
    const n_complex = potential.values.len;
    const n_f64 = n_complex * 2;
    const v_in: []f64 = @as([*]f64, @ptrCast(potential.values.ptr))[0..n_f64];

    if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
        const residual_g = try alloc.alloc(math.Complex, n_complex);
        for (0..n_complex) |idx| {
            residual_g[idx] = math.complex.sub(potential_out.values[idx], potential.values[idx]);
        }
        if (cfg.scf.diemac > 1.0) {
            mixing.apply_model_dielectric_preconditioner(
                grid,
                residual_g,
                cfg.scf.diemac,
                cfg.scf.dielng,
            );
        }
        const precond_f64: []f64 = @as([*]f64, @ptrCast(residual_g.ptr))[0..n_f64];
        try common.pulay_mixer.?.mix_with_residual(v_in, precond_f64, cfg.scf.mixing_beta);
    } else {
        const v_out_ptr = @as([*]const f64, @ptrCast(potential_out.values.ptr));
        const v_out: []const f64 = v_out_ptr[0..n_f64];
        mix_density(v_in, v_out, cfg.scf.mixing_beta);
    }

    @memcpy(rho, new_rho);
    potential_out.deinit(alloc);
    keep_potential_out.* = true;
}

/// Density-mixing branch: mixes rho and rebuilds potential.
fn mix_density_mode(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: grid_mod.Grid,
    common: *ScfCommon,
    iterations: usize,
    rho: []f64,
    new_rho: []const f64,
    potential: *hamiltonian.PotentialGrid,
    paw_ecutrho: ?f64,
) !void {
    if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
        if (cfg.scf.kerker_q0 > 0.0) {
            try common.pulay_mixer.?.mix_kerker_pulay(
                rho,
                new_rho,
                cfg.scf.mixing_beta,
                grid,
                cfg.scf.kerker_q0,
                cfg.scf.use_rfft,
            );
        } else {
            try common.pulay_mixer.?.mix(rho, new_rho, cfg.scf.mixing_beta);
        }
    } else if (cfg.scf.kerker_q0 > 0.0) {
        try mix_density_kerker(
            alloc,
            grid,
            rho,
            new_rho,
            cfg.scf.mixing_beta,
            cfg.scf.kerker_q0,
            cfg.scf.use_rfft,
        );
    } else {
        mix_density(rho, new_rho, cfg.scf.mixing_beta);
    }

    potential.deinit(alloc);
    potential.* = try potential_mod.build_potential_grid(
        alloc,
        grid,
        rho,
        common.rho_core,
        cfg.scf.use_rfft,
        cfg.scf.xc,
        null,
        common.coulomb_r_cut,
        paw_ecutrho,
    );
}

/// Update PAW D_ij matrix from the current mixed potential (no-op for non-PAW).
fn update_paw_dij_if_needed(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    grid: grid_mod.Grid,
    common: *ScfCommon,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    potential: hamiltonian.PotentialGrid,
    apply_caches: []apply.KpointApplyCache,
) !void {
    if (!common.is_paw) return;
    const tabs = common.paw_tabs orelse return;
    const gs_paw = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    const ecutrho_paw = cfg.scf.ecut_ry * gs_paw * gs_paw;
    try paw_scf.update_paw_dij(
        alloc,
        grid,
        common.ionic,
        potential,
        tabs,
        species,
        atoms,
        apply_caches,
        ecutrho_paw,
        &common.paw_rhoij.?,
        cfg.scf.xc,
        cfg.scf.symmetry,
        &common.paw_gaunt.?,
        false,
        null,
        1.0,
    );
}

fn prepare_scf_iteration_density(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
    prof: *ScfLoopProf,
) !ScfIterationDensity {
    const t_density_start = if (ctx.cfg.scf.profile) profile_start(ctx.io) else null;
    const density_result = try density_mod.compute_density(.{
        .alloc = ctx.alloc,
        .io = ctx.io,
        .cfg = ctx.cfg.*,
        .grid = ctx.common.grid,
        .kpoints = ctx.common.kpoints,
        .ionic = ctx.common.ionic,
        .species = ctx.species,
        .atoms = ctx.atoms,
        .recip = ctx.recip,
        .volume = ctx.volume_bohr,
        .potential = state.potential,
        .scf_iter = state.iterations,
        .kpoint_cache = caches.kpoint_cache,
        .apply_caches = caches.apply_caches,
        .radial_tables = ctx.common.radial_tables,
        .loop_profile = if (ctx.cfg.scf.profile) ScfLoopProfile{
            .build_local_r_ns = &prof.build_local_r_ns,
            .build_fft_map_ns = &prof.build_fft_map_ns,
        } else null,
        .paw_tabs = ctx.common.paw_tabs,
        .paw_rhoij = if (ctx.common.paw_rhoij) |*rij| rij else null,
    });
    if (ctx.cfg.scf.profile) profile_add(ctx.io, &prof.compute_density_ns, t_density_start);

    state.last_band_energy = density_result.band_energy;
    state.last_nonlocal_energy = density_result.nonlocal_energy;
    state.last_entropy_energy = density_result.entropy_energy;
    state.last_fermi_level = density_result.fermi_level;

    try paw_results_mod.symmetrize_density_and_rho_ij(
        ctx.alloc,
        ctx.cfg,
        ctx.common.grid,
        ctx.common,
        ctx.species,
        ctx.atoms,
        density_result.rho,
    );

    return .{
        .density_result = density_result,
        .rho_for_potential = try prepare_density_for_potential(ctx, density_result.rho),
    };
}

fn build_scf_iteration_potential(
    ctx: *const ScfRunContext,
    state: *ScfRunState,
    rho_for_potential: []const f64,
    prof: *ScfLoopProf,
) !ScfIterationPotential {
    if (state.vxc_r) |old| ctx.alloc.free(old);
    state.vxc_r = null;
    const vxc_r_ptr: ?*?[]f64 = if (ctx.cfg.relax.enabled) &state.vxc_r else null;
    const t_build_pot_start = if (ctx.cfg.scf.profile) profile_start(ctx.io) else null;
    const potential_out = try potential_mod.build_potential_grid(
        ctx.alloc,
        ctx.common.grid,
        rho_for_potential,
        ctx.common.rho_core,
        ctx.cfg.scf.use_rfft,
        ctx.cfg.scf.xc,
        vxc_r_ptr,
        ctx.common.coulomb_r_cut,
        ctx.paw_ecutrho,
    );
    if (ctx.cfg.scf.profile) profile_add(ctx.io, &prof.build_potential_ns, t_build_pot_start);

    const t_resid_start = if (ctx.cfg.scf.profile) profile_start(ctx.io) else null;
    state.last_potential_residual = try run_state_mod.record_potential_residual(
        ctx.alloc,
        ctx.common.grid,
        state.potential,
        potential_out,
        &state.vresid_last,
    );
    if (ctx.cfg.scf.profile) profile_add(ctx.io, &prof.residual_ns, t_resid_start);
    return .{ .potential_out = potential_out };
}

fn prepare_density_for_potential(
    ctx: *const ScfRunContext,
    rho: []f64,
) ![]const f64 {
    const ecutrho_comp = paw_results_mod.paw_ecutrho_compute(ctx.cfg);
    if (ctx.common.is_paw) {
        try paw_results_mod.filter_augmented_density_in_place(
            ctx.alloc,
            ctx.common.grid,
            rho,
            ecutrho_comp,
            ctx.cfg.scf.use_rfft,
        );
        return paw_results_mod.build_augmented_density(
            ctx.alloc,
            ctx.common.grid,
            rho,
            ctx.common,
            ctx.atoms,
            ecutrho_comp,
            ctx.common.grid.count(),
        );
    }
    return rho;
}

fn finish_scf_iteration(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
    density: *const ScfIterationDensity,
    potential_step: *ScfIterationPotential,
    prof: *ScfLoopProf,
) !bool {
    const diff = density_diff(state.rho, density.density_result.rho);
    const conv_value = switch (ctx.cfg.scf.convergence_metric) {
        .density => diff,
        .potential => state.last_potential_residual,
    };
    try log_scf_iteration_progress(ctx, state, diff);
    if (conv_value < ctx.cfg.scf.convergence) {
        return finish_converged_scf_iteration(ctx, state, density, potential_step);
    }

    const t_mix_start = if (ctx.cfg.scf.profile) profile_start(ctx.io) else null;
    if (ctx.cfg.scf.mixing_mode == .potential) {
        try mix_potential_mode(
            ctx.alloc,
            ctx.cfg,
            ctx.common.grid,
            ctx.common,
            state.iterations,
            state.rho,
            density.density_result.rho,
            &state.potential,
            &potential_step.potential_out,
            &potential_step.keep,
        );
    } else {
        try mix_density_mode(
            ctx.alloc,
            ctx.cfg,
            ctx.common.grid,
            ctx.common,
            state.iterations,
            state.rho,
            density.density_result.rho,
            &state.potential,
            ctx.paw_ecutrho,
        );
    }
    if (ctx.cfg.scf.profile) profile_add(ctx.io, &prof.mixing_ns, t_mix_start);
    try update_paw_dij_if_needed(
        ctx.alloc,
        ctx.cfg,
        ctx.common.grid,
        ctx.common,
        ctx.species,
        ctx.atoms,
        state.potential,
        caches.apply_caches,
    );
    return false;
}

fn log_scf_iteration_progress(
    ctx: *const ScfRunContext,
    state: *const ScfRunState,
    diff: f64,
) !void {
    try ctx.common.log.write_iter(
        state.iterations,
        diff,
        state.last_potential_residual,
        state.last_band_energy,
        state.last_nonlocal_energy,
    );
    if (!ctx.cfg.scf.quiet) {
        try log_progress(
            ctx.io,
            state.iterations,
            diff,
            state.last_potential_residual,
            state.last_band_energy,
            state.last_nonlocal_energy,
        );
    }
}

fn finish_converged_scf_iteration(
    ctx: *const ScfRunContext,
    state: *ScfRunState,
    density: *const ScfIterationDensity,
    potential_step: *ScfIterationPotential,
) bool {
    state.converged = true;
    @memcpy(state.rho, density.density_result.rho);
    state.potential.deinit(ctx.alloc);
    state.potential = potential_step.potential_out;
    potential_step.keep = true;
    return true;
}

fn run_scf_iterations(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
) !void {
    var prof = ScfLoopProf{};
    while (state.iterations < ctx.cfg.scf.max_iter) : (state.iterations += 1) {
        if (!ctx.cfg.scf.quiet) try log_iter_start(ctx.io, state.iterations);
        const density = try prepare_scf_iteration_density(ctx, caches, state, &prof);
        defer density.deinit(ctx.alloc, ctx.common.is_paw);

        var potential_step = try build_scf_iteration_potential(
            ctx,
            state,
            density.rho_for_potential,
            &prof,
        );
        defer potential_step.deinit(ctx.alloc);

        if (try finish_scf_iteration(ctx, caches, state, &density, &potential_step, &prof)) break;
    }
    if (ctx.cfg.scf.profile and !ctx.cfg.scf.quiet) {
        try log_loop_profile(
            ctx.io,
            prof.compute_density_ns,
            prof.build_potential_ns,
            prof.residual_ns,
            prof.mixing_ns,
            prof.build_local_r_ns,
            prof.build_fft_map_ns,
        );
    }
}

fn add_paw_onsite_energy_if_needed(
    ctx: *const ScfRunContext,
    energy_terms: *energy_mod.EnergyTerms,
) !void {
    if (!ctx.common.is_paw) return;
    const prij = ctx.common.paw_rhoij orelse return;
    const tabs = ctx.common.paw_tabs orelse return;
    energy_terms.paw_onsite = try paw_scf.compute_paw_onsite_energy_total(
        ctx.alloc,
        &prij,
        tabs,
        ctx.species,
        ctx.atoms,
        ctx.cfg.scf.xc,
        &ctx.common.paw_gaunt.?,
        null,
        null,
    );
    energy_terms.total += energy_terms.paw_onsite;
}

fn compute_final_scf_energy_terms(
    ctx: *const ScfRunContext,
    state: *ScfRunState,
) !energy_mod.EnergyTerms {
    const rho_aug_for_energy = try paw_results_mod.maybe_build_rho_aug_for_energy(
        ctx.alloc,
        ctx.cfg,
        ctx.common.grid,
        ctx.common,
        ctx.atoms,
        state.rho,
    );
    defer if (rho_aug_for_energy) |a| ctx.alloc.free(a);

    var energy_terms = try energy_mod.compute_energy_terms(.{
        .alloc = ctx.alloc,
        .io = ctx.io,
        .grid = ctx.common.grid,
        .species = ctx.species,
        .atoms = ctx.atoms,
        .rho = state.rho,
        .rho_core = ctx.common.rho_core,
        .rho_aug = rho_aug_for_energy,
        .band_energy = state.last_band_energy,
        .nonlocal_energy = state.last_nonlocal_energy,
        .entropy_energy = state.last_entropy_energy,
        .local_cfg = ctx.common.local_cfg,
        .ewald_cfg = ctx.cfg.ewald,
        .vdw_cfg = ctx.cfg.vdw,
        .xc_func = ctx.cfg.scf.xc,
        .use_rfft = ctx.cfg.scf.use_rfft,
        .quiet = ctx.cfg.scf.quiet,
        .coulomb_r_cut = ctx.common.coulomb_r_cut,
        .ecutrho = ctx.paw_ecutrho,
    });
    try add_paw_onsite_energy_if_needed(ctx, &energy_terms);
    return energy_terms;
}

fn maybe_compute_final_scf_wavefunctions(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
) !void {
    if (!(ctx.cfg.relax.enabled or ctx.cfg.dfpt.enabled or
        ctx.cfg.scf.compute_stress or ctx.cfg.dos.enabled))
    {
        return;
    }
    const wfn_result = try compute_final_wavefunctions_with_spin_factor(.{
        .alloc = ctx.alloc,
        .io = ctx.io,
        .cfg = ctx.cfg.*,
        .grid = ctx.common.grid,
        .kpoints = ctx.common.kpoints,
        .ionic = ctx.common.ionic,
        .species = ctx.species,
        .atoms = ctx.atoms,
        .recip = ctx.recip,
        .volume = ctx.volume_bohr,
        .potential = state.potential,
        .kpoint_cache = caches.kpoint_cache,
        .apply_caches = caches.apply_caches,
        .radial_tables = ctx.common.radial_tables,
        .paw_tabs = ctx.common.paw_tabs,
        .spin_factor = 2.0,
    });
    state.wavefunctions = wfn_result.wavefunctions;
    state.last_band_energy = wfn_result.band_energy;
    state.last_nonlocal_energy = wfn_result.nonlocal_energy;
}

fn extract_final_paw_results(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    energy_terms: *energy_mod.EnergyTerms,
) !FinalPawResults {
    var result: FinalPawResults = .{};
    if (!ctx.common.is_paw) return result;
    try paw_results_mod.extract_paw_results(
        ctx.alloc,
        ctx.cfg,
        ctx.species,
        ctx.atoms,
        caches.apply_caches,
        ctx.common,
        energy_terms,
        &result.paw_tabs,
        &result.paw_dij,
        &result.paw_dij_m,
        &result.paw_dxc,
        &result.paw_rhoij,
    );
    return result;
}

fn maybe_copy_ionic_g(
    alloc: std.mem.Allocator,
    common: *const ScfCommon,
) !?[]math.Complex {
    if (!common.is_paw) return null;
    const result_ionic_g = try alloc.alloc(math.Complex, common.ionic.values.len);
    @memcpy(result_ionic_g, common.ionic.values);
    return result_ionic_g;
}

fn log_final_scf_summary(
    ctx: *const ScfRunContext,
    state: *const ScfRunState,
    energy_terms: energy_mod.EnergyTerms,
) !void {
    if (!ctx.cfg.scf.quiet) {
        try log_energy_summary(
            ctx.io,
            density_symmetry.total_charge(state.rho, ctx.common.grid),
            ctx.common.ionic.value_at(0, 0, 0),
            state.potential.value_at(0, 0, 0),
            energy_terms,
        );
    }
    try ctx.common.log.write_result(
        state.converged,
        state.iterations,
        energy_terms.total,
        energy_terms.band,
        energy_terms.hartree,
        energy_terms.xc,
        energy_terms.ion_ion,
        energy_terms.psp_core,
    );
}

fn finalize_scf_run(
    ctx: *const ScfRunContext,
    caches: *const ScfRunCaches,
    state: *ScfRunState,
) !ScfResult {
    var energy_terms = try compute_final_scf_energy_terms(ctx, state);
    try maybe_compute_final_scf_wavefunctions(ctx, caches, state);
    const final_paw_results = try extract_final_paw_results(ctx, caches, &energy_terms);
    try log_final_scf_summary(ctx, state, energy_terms);
    return build_scf_result(.{
        .potential = state.potential,
        .density = state.rho,
        .iterations = state.iterations,
        .converged = state.converged,
        .energy = energy_terms,
        .fermi_level = state.last_fermi_level,
        .potential_residual = state.last_potential_residual,
        .wavefunctions = state.wavefunctions,
        .vresid = state.vresid_last,
        .grid = ctx.common.grid,
        .kpoint_cache = caches.kpoint_cache,
        .apply_caches = caches.apply_caches,
        .vxc_r = state.vxc_r,
        .paw_tabs = final_paw_results.paw_tabs,
        .paw_dij = final_paw_results.paw_dij,
        .paw_dij_m = final_paw_results.paw_dij_m,
        .paw_dxc = final_paw_results.paw_dxc,
        .paw_rhoij = final_paw_results.paw_rhoij,
        .ionic_g = try maybe_copy_ionic_g(ctx.alloc, ctx.common),
        .rho_core_copy = try result_mod.copy_rho_core_for_result(ctx.alloc, ctx.common.rho_core),
    });
}

pub fn run(params: ScfParams) !ScfResult {
    const alloc = params.alloc;
    const io = params.io;
    const cfg = params.cfg;
    const species = params.model.species;
    const atoms = params.model.atoms;
    const volume_bohr = params.model.volume_bohr;
    const initial_density = params.initial_density;
    const initial_kpoint_cache = params.initial_kpoint_cache;
    const initial_apply_caches = params.initial_apply_caches;
    if (!cfg.scf.enabled) return error.ScfDisabled;

    var common = try common_mod.init_scf_common(.{
        .alloc = params.alloc,
        .io = params.io,
        .cfg = params.cfg,
        .model = params.model,
        .ff_tables = params.ff_tables,
    });
    defer common.deinit();

    // Dispatch to spin-polarized SCF loop if nspin=2
    if (cfg.scf.nspin == 2) {
        return scf_spin.run_spin_polarized_loop(
            alloc,
            io,
            cfg,
            species,
            atoms,
            volume_bohr,
            &common,
        );
    }

    const paw_ecutrho = common_mod.ecutrho_for_paw(&cfg, common.is_paw);
    const ctx = run_state_mod.make_run_context(
        alloc,
        io,
        &cfg,
        params.model,
        &common,
        paw_ecutrho,
    );
    const caches = try run_state_mod.init_scf_run_caches(
        alloc,
        common.kpoints.len,
        initial_kpoint_cache,
        initial_apply_caches,
    );
    errdefer caches.deinit(alloc);

    var state = try run_state_mod.init_scf_run_state(alloc, &cfg, &common, initial_density, paw_ecutrho);
    errdefer state.deinit(alloc);

    try run_scf_iterations(&ctx, &caches, &state);
    return try finalize_scf_run(&ctx, &caches, &state);
}

pub const build_atomic_density = core_density.build_atomic_density;

pub const density_diff = util.density_diff;

pub const has_nonlocal = util.has_nonlocal;
pub const has_qij = util.has_qij;
pub const has_paw = util.has_paw;
pub const total_electrons = util.total_electrons;

test {
    _ = @import("band_solver.zig");
    _ = @import("gvec_iter.zig");
    _ = @import("paw_scf.zig");
    _ = @import("smearing.zig");
}

test "auto grid chooses fft-friendly size for aluminum" {
    const cell_ang = math.Mat3.from_rows(
        .{ .x = 0.0, .y = 2.025, .z = 2.025 },
        .{ .x = 2.025, .y = 0.0, .z = 2.025 },
        .{ .x = 2.025, .y = 2.025, .z = 0.0 },
    );
    const cell_bohr = cell_ang.scale(math.units_scale_to_bohr(.angstrom));
    const recip = math.reciprocal(cell_bohr);
    const grid = grid_mod.auto_grid(15.0, 1.0, recip);
    try std.testing.expect(grid[0] >= 3);
    try std.testing.expect(grid[1] >= 3);
    try std.testing.expect(grid[2] >= 3);
}
