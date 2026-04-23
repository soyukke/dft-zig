const std = @import("std");
const math = @import("../math/math.zig");
const config_mod = @import("../config/config.zig");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const form_factor_mod = @import("../pseudopotential/form_factor.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const timing_mod = @import("../runtime/timing.zig");
const runtime_logging = @import("../runtime/logging.zig");
const scf = @import("../scf/scf.zig");
const model_mod = @import("../dft/model.zig");
const forces_mod = @import("../forces/forces.zig");
pub const optimizer = @import("optimizer.zig");

const stress_mod = @import("../stress/stress.zig");

/// Result of structure relaxation.
pub const RelaxResult = struct {
    final_atoms: []hamiltonian.AtomData,
    final_energy: f64,
    final_forces: []math.Vec3,
    iterations: usize,
    converged: bool,
    trajectory: ?[]RelaxStep,
    // Caches for warmstarting final SCF after relax
    final_density: ?[]f64 = null,
    final_kpoint_cache: ?[]scf.KpointCache = null,
    final_apply_caches: ?[]scf.KpointApplyCache = null,
    final_potential: ?hamiltonian.PotentialGrid = null,
    // vc-relax: final cell parameters
    final_cell: ?math.Mat3 = null,
    final_recip: ?math.Mat3 = null,
    final_volume: ?f64 = null,

    pub fn deinit(self: *RelaxResult, alloc: std.mem.Allocator) void {
        if (self.final_atoms.len > 0) alloc.free(self.final_atoms);
        if (self.final_forces.len > 0) alloc.free(self.final_forces);
        if (self.trajectory) |traj| {
            for (traj) |*step| {
                step.deinit(alloc);
            }
            alloc.free(traj);
        }
        if (self.final_density) |d| alloc.free(d);
        if (self.final_kpoint_cache) |cache| {
            for (cache) |*c| c.deinit();
            alloc.free(cache);
        }
        if (self.final_apply_caches) |caches| {
            for (caches) |*ac| ac.deinit(alloc);
            alloc.free(caches);
        }
        if (self.final_potential) |*p| p.deinit(alloc);
    }
};

/// A single step in the relaxation trajectory.
pub const RelaxStep = struct {
    atoms: []hamiltonian.AtomData,
    energy: f64,
    forces: []math.Vec3,
    max_force: f64,

    pub fn deinit(self: *RelaxStep, alloc: std.mem.Allocator) void {
        if (self.atoms.len > 0) alloc.free(self.atoms);
        if (self.forces.len > 0) alloc.free(self.forces);
    }
};

const max_backtrack: usize = 5;

const RelaxTables = struct {
    local_cfg: local_potential.LocalPotentialConfig,
    force_radial_tables_buf: []nonlocal_mod.RadialTableSet,
    ff_tables_buf: []form_factor_mod.LocalFormFactorTable,
    rho_atom_tables_buf: []form_factor_mod.RadialFormFactorTable,
    rho_core_tables_buf: []form_factor_mod.RadialFormFactorTable,

    fn deinit(self: *RelaxTables, alloc: std.mem.Allocator) void {
        for (self.force_radial_tables_buf) |*t| {
            if (t.tables.len > 0) t.deinit(alloc);
        }
        alloc.free(self.force_radial_tables_buf);
        for (self.ff_tables_buf) |*t| t.deinit(alloc);
        alloc.free(self.ff_tables_buf);
        for (self.rho_atom_tables_buf) |*t| t.deinit(alloc);
        alloc.free(self.rho_atom_tables_buf);
        for (self.rho_core_tables_buf) |*t| t.deinit(alloc);
        alloc.free(self.rho_core_tables_buf);
    }

    fn force_radial_tables(self: *const RelaxTables) ?[]nonlocal_mod.RadialTableSet {
        return self.force_radial_tables_buf;
    }

    fn ff_tables(self: *const RelaxTables) ?[]const form_factor_mod.LocalFormFactorTable {
        return self.ff_tables_buf;
    }

    fn rho_atom_tables(self: *const RelaxTables) ?[]const form_factor_mod.RadialFormFactorTable {
        return self.rho_atom_tables_buf;
    }

    fn rho_core_tables(self: *const RelaxTables) ?[]const form_factor_mod.RadialFormFactorTable {
        return self.rho_core_tables_buf;
    }
};

const RelaxWarmstart = struct {
    prev_density: ?[]f64 = null,
    prev_kpoint_cache: ?[]scf.KpointCache = null,
    prev_apply_caches: ?[]scf.KpointApplyCache = null,
    final_potential_saved: ?hamiltonian.PotentialGrid = null,

    fn deinit(self: *RelaxWarmstart, alloc: std.mem.Allocator) void {
        if (self.prev_density) |pd| alloc.free(pd);
        if (self.prev_kpoint_cache) |cache| {
            for (cache) |*c| c.deinit();
            alloc.free(cache);
        }
        if (self.prev_apply_caches) |caches| {
            for (caches) |*ac| ac.deinit(alloc);
            alloc.free(caches);
        }
        if (self.final_potential_saved) |*p| p.deinit(alloc);
    }

    fn update_from_scf(
        self: *RelaxWarmstart,
        alloc: std.mem.Allocator,
        scf_result: *scf.ScfResult,
    ) !void {
        if (self.prev_density) |pd| alloc.free(pd);
        self.prev_density = try alloc.alloc(f64, scf_result.density.len);
        @memcpy(self.prev_density.?, scf_result.density);

        if (self.prev_kpoint_cache) |old_cache| {
            for (old_cache) |*c| c.deinit();
            alloc.free(old_cache);
        }
        self.prev_kpoint_cache = scf_result.kpoint_cache;
        scf_result.kpoint_cache = null;

        self.prev_apply_caches = scf_result.apply_caches;
        scf_result.apply_caches = null;
    }

    fn invalidate_for_cell_change(self: *RelaxWarmstart, alloc: std.mem.Allocator) void {
        if (self.prev_density) |pd| {
            alloc.free(pd);
            self.prev_density = null;
        }
        if (self.prev_kpoint_cache) |cache| {
            for (cache) |*c| c.deinit();
            alloc.free(cache);
            self.prev_kpoint_cache = null;
        }
        if (self.prev_apply_caches) |caches| {
            for (caches) |*ac| ac.deinit(alloc);
            alloc.free(caches);
            self.prev_apply_caches = null;
        }
    }

    fn save_final_potential(self: *RelaxWarmstart, scf_result: *scf.ScfResult) void {
        self.final_potential_saved = scf_result.potential;
        scf_result.potential = .{
            .nx = 0,
            .ny = 0,
            .nz = 0,
            .min_h = 0,
            .min_k = 0,
            .min_l = 0,
            .values = &[_]math.Complex{},
        };
    }
};

const RelaxState = struct {
    atoms: []hamiltonian.AtomData,
    opt: optimizer.Optimizer,
    trajectory: ?std.ArrayList(RelaxStep),
    current_cell: math.Mat3,
    current_recip: math.Mat3,
    current_volume: f64,
    alpha: f64,
    prev_positions: ?[]math.Vec3 = null,
    prev_forces: ?[]math.Vec3 = null,
    prev_energy: f64 = std.math.inf(f64),
    saved_positions: ?[]math.Vec3 = null,
    saved_displacement: ?[]math.Vec3 = null,
    backtrack_count: usize = 0,
    warmstart: RelaxWarmstart = .{},
    converged: bool = false,
    final_energy: f64 = 0.0,
    final_forces: []math.Vec3 = &[_]math.Vec3{},
    iterations: usize = 0,

    fn deinit(self: *RelaxState, alloc: std.mem.Allocator) void {
        if (self.atoms.len > 0) alloc.free(self.atoms);
        self.opt.deinit(alloc);
        if (self.trajectory) |*t| {
            for (t.items) |*step| step.deinit(alloc);
            t.deinit(alloc);
        }
        if (self.prev_positions) |pp| alloc.free(pp);
        if (self.prev_forces) |pf| alloc.free(pf);
        if (self.saved_positions) |sp| alloc.free(sp);
        if (self.saved_displacement) |sd| alloc.free(sd);
        if (self.final_forces.len > 0) alloc.free(self.final_forces);
        self.warmstart.deinit(alloc);
    }

    fn take_result(
        self: *RelaxState,
        alloc: std.mem.Allocator,
        cfg: config_mod.Config,
    ) !RelaxResult {
        var result_trajectory: ?[]RelaxStep = null;
        if (self.trajectory) |*traj| {
            result_trajectory = try traj.toOwnedSlice(alloc);
            self.trajectory = null;
        }

        const out_density = self.warmstart.prev_density;
        self.warmstart.prev_density = null;
        const out_kpoint_cache = self.warmstart.prev_kpoint_cache;
        self.warmstart.prev_kpoint_cache = null;
        const out_apply_caches = self.warmstart.prev_apply_caches;
        self.warmstart.prev_apply_caches = null;
        const out_potential = self.warmstart.final_potential_saved;
        self.warmstart.final_potential_saved = null;

        const result = RelaxResult{
            .final_atoms = self.atoms,
            .final_energy = self.final_energy,
            .final_forces = self.final_forces,
            .iterations = self.iterations,
            .converged = self.converged,
            .trajectory = result_trajectory,
            .final_density = out_density,
            .final_kpoint_cache = out_kpoint_cache,
            .final_apply_caches = out_apply_caches,
            .final_potential = out_potential,
            .final_cell = if (cfg.relax.cell_relax) self.current_cell else null,
            .final_recip = if (cfg.relax.cell_relax) self.current_recip else null,
            .final_volume = if (cfg.relax.cell_relax) self.current_volume else null,
        };
        self.atoms = &[_]hamiltonian.AtomData{};
        self.final_forces = &[_]math.Vec3{};
        return result;
    }
};

const RelaxContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    relax_cfg: config_mod.Config,
    species: []const hamiltonian.SpeciesEntry,
    tables: RelaxTables,

    fn deinit(self: *RelaxContext) void {
        self.tables.deinit(self.alloc);
    }
};

fn build_relax_config(
    io: std.Io,
    cfg: config_mod.Config,
    cell: math.Mat3,
) !config_mod.Config {
    var relax_cfg = cfg;
    if (cfg.scf.symmetry) {
        try log_warn(io, "scf.symmetry is enabled, but will be disabled during relaxation " ++
            "to ensure consistent k-point sets across iterations. " ++
            "Set scf.symmetry = false to suppress this warning.");
    }
    relax_cfg.scf.symmetry = false;
    if (cfg.relax.cell_relax) {
        relax_cfg.scf.compute_stress = true;
        relax_cfg.units = .bohr;
        relax_cfg.cell = cell;
    }
    return relax_cfg;
}

fn build_force_radial_tables(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    species: []const hamiltonian.SpeciesEntry,
) ![]nonlocal_mod.RadialTableSet {
    const g_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 1.5;
    const tables = try alloc.alloc(nonlocal_mod.RadialTableSet, species.len);
    for (species, 0..) |entry, si| {
        const upf = entry.upf.*;
        if (upf.beta.len == 0 or upf.dij.len == 0) {
            tables[si] = .{ .tables = &[_]nonlocal_mod.RadialTable{} };
            continue;
        }
        tables[si] = try nonlocal_mod.RadialTableSet.init(
            alloc,
            upf.beta,
            upf.r,
            upf.rab,
            g_max,
        );
    }
    return tables;
}

fn build_local_form_factor_tables(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    species: []const hamiltonian.SpeciesEntry,
    cell: math.Mat3,
) !struct {
    local_cfg: local_potential.LocalPotentialConfig,
    tables: []form_factor_mod.LocalFormFactorTable,
} {
    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, cell);
    const ff_q_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 3.0;
    const tables = try alloc.alloc(form_factor_mod.LocalFormFactorTable, species.len);
    for (species, 0..) |entry, si| {
        tables[si] = try form_factor_mod.LocalFormFactorTable.init(
            alloc,
            entry.upf.*,
            entry.z_valence,
            local_cfg,
            ff_q_max,
        );
    }
    return .{ .local_cfg = local_cfg, .tables = tables };
}

fn build_residual_form_factor_tables(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    species: []const hamiltonian.SpeciesEntry,
) !struct {
    rho_atom: []form_factor_mod.RadialFormFactorTable,
    rho_core: []form_factor_mod.RadialFormFactorTable,
} {
    const ff_q_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 3.0;
    const rho_atom = try alloc.alloc(form_factor_mod.RadialFormFactorTable, species.len);
    errdefer alloc.free(rho_atom);
    const rho_core = try alloc.alloc(form_factor_mod.RadialFormFactorTable, species.len);
    errdefer alloc.free(rho_core);

    for (species, 0..) |entry, si| {
        rho_atom[si] = try form_factor_mod.RadialFormFactorTable.init_rho_atom(
            alloc,
            entry.upf.*,
            ff_q_max,
        );
        rho_core[si] = try form_factor_mod.RadialFormFactorTable.init_rho_core(
            alloc,
            entry.upf.*,
            ff_q_max,
        );
    }
    return .{ .rho_atom = rho_atom, .rho_core = rho_core };
}

fn init_relax_tables(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    species: []const hamiltonian.SpeciesEntry,
    cell: math.Mat3,
) !RelaxTables {
    const force_radial_tables_buf = try build_force_radial_tables(alloc, cfg, species);
    errdefer {
        for (force_radial_tables_buf) |*t| {
            if (t.tables.len > 0) t.deinit(alloc);
        }
        alloc.free(force_radial_tables_buf);
    }

    const local = try build_local_form_factor_tables(alloc, cfg, species, cell);
    errdefer {
        for (local.tables) |*t| t.deinit(alloc);
        alloc.free(local.tables);
    }

    const residual = try build_residual_form_factor_tables(alloc, cfg, species);
    errdefer {
        for (residual.rho_atom) |*t| t.deinit(alloc);
        alloc.free(residual.rho_atom);
        for (residual.rho_core) |*t| t.deinit(alloc);
        alloc.free(residual.rho_core);
    }

    return .{
        .local_cfg = local.local_cfg,
        .force_radial_tables_buf = force_radial_tables_buf,
        .ff_tables_buf = local.tables,
        .rho_atom_tables_buf = residual.rho_atom,
        .rho_core_tables_buf = residual.rho_core,
    };
}

fn init_relax_context(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    species: []const hamiltonian.SpeciesEntry,
    cell: math.Mat3,
) !RelaxContext {
    return .{
        .alloc = alloc,
        .io = io,
        .cfg = cfg,
        .relax_cfg = try build_relax_config(io, cfg, cell),
        .species = species,
        .tables = try init_relax_tables(alloc, cfg, species, cell),
    };
}

fn init_relax_state(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    initial_atoms: []const hamiltonian.AtomData,
    cell: math.Mat3,
    recip: math.Mat3,
    volume: f64,
) !RelaxState {
    const atoms = try alloc.alloc(hamiltonian.AtomData, initial_atoms.len);
    @memcpy(atoms, initial_atoms);

    return .{
        .atoms = atoms,
        .opt = try optimizer.Optimizer.init(
            alloc,
            cfg.relax.algorithm,
            initial_atoms.len,
            initial_atoms,
            cell,
        ),
        .trajectory = if (cfg.relax.output_trajectory) .empty else null,
        .current_cell = cell,
        .current_recip = recip,
        .current_volume = volume,
        .alpha = if (cfg.ewald.alpha > 0.0) cfg.ewald.alpha else compute_default_alpha(cell),
    };
}

const RelaxStressInfo = struct {
    stress_converged: bool = true,
    cached_stress_total: ?stress_mod.Stress3x3 = null,
};

const ForceVxcInput = struct {
    avg: ?[]f64 = null,
    values: ?[]const f64 = null,

    fn deinit(self: *ForceVxcInput, alloc: std.mem.Allocator) void {
        if (self.avg) |v| alloc.free(v);
    }
};

fn build_step_model(
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    current_cell: math.Mat3,
    current_recip: math.Mat3,
    current_volume: f64,
) model_mod.Model {
    return .{
        .species = species,
        .atoms = atoms,
        .cell_bohr = current_cell,
        .recip = current_recip,
        .volume_bohr = current_volume,
    };
}

fn run_relax_scf(
    ctx: *const RelaxContext,
    state: *const RelaxState,
) !scf.ScfResult {
    const step_model = build_step_model(
        ctx.species,
        state.atoms,
        state.current_cell,
        state.current_recip,
        state.current_volume,
    );
    return try scf.run(.{
        .alloc = ctx.alloc,
        .io = ctx.io,
        .cfg = ctx.relax_cfg,
        .model = &step_model,
        .initial_density = state.warmstart.prev_density,
        .initial_kpoint_cache = state.warmstart.prev_kpoint_cache,
        .initial_apply_caches = state.warmstart.prev_apply_caches,
        .ff_tables = ctx.tables.ff_tables(),
    });
}

fn clear_relax_history(state: *RelaxState, alloc: std.mem.Allocator) void {
    if (state.prev_positions) |pp| {
        alloc.free(pp);
        state.prev_positions = null;
    }
    if (state.prev_forces) |pf| {
        alloc.free(pf);
        state.prev_forces = null;
    }
}

fn restore_backtracked_positions(
    atoms: []hamiltonian.AtomData,
    saved_positions: []const math.Vec3,
    saved_displacement: []const math.Vec3,
    scale: f64,
) void {
    for (0..atoms.len) |i| {
        atoms[i].position = math.Vec3.add(
            saved_positions[i],
            math.Vec3.scale(saved_displacement[i], scale),
        );
    }
}

fn handle_backtrack(
    ctx: *const RelaxContext,
    state: *RelaxState,
) !bool {
    const energy_increased = state.final_energy > state.prev_energy;
    const have_saved_step = state.saved_positions != null and state.saved_displacement != null;
    if (ctx.cfg.relax.cell_relax or !energy_increased or !have_saved_step) {
        state.backtrack_count = 0;
        state.prev_energy = state.final_energy;
        return false;
    }

    state.backtrack_count += 1;
    try log_backtrack(ctx.io, state.backtrack_count, state.final_energy, state.prev_energy);
    if (state.backtrack_count > max_backtrack) {
        state.opt.reset();
        state.backtrack_count = 0;
        state.prev_energy = state.final_energy;
        clear_relax_history(state, ctx.alloc);
        return false;
    }

    const scale = std.math.pow(f64, 0.5, @as(f64, @floatFromInt(state.backtrack_count)));
    restore_backtracked_positions(
        state.atoms,
        state.saved_positions.?,
        state.saved_displacement.?,
        scale,
    );
    return true;
}

fn average_forces_in_place(forces: []math.Vec3) void {
    var avg = math.Vec3{ .x = 0, .y = 0, .z = 0 };
    for (forces) |f| {
        avg.x += f.x;
        avg.y += f.y;
        avg.z += f.z;
    }
    const inv_n = 1.0 / @as(f64, @floatFromInt(forces.len));
    avg.x *= inv_n;
    avg.y *= inv_n;
    avg.z *= inv_n;
    for (forces) |*f| {
        f.x -= avg.x;
        f.y -= avg.y;
        f.z -= avg.z;
    }
}

fn build_force_vxc_input(
    alloc: std.mem.Allocator,
    scf_result: *const scf.ScfResult,
) !ForceVxcInput {
    const have_spin_vxc = scf_result.vxc_r_up != null and scf_result.vxc_r_down != null;
    if (!have_spin_vxc) return .{ .values = scf_result.vxc_r };

    const up = scf_result.vxc_r_up.?;
    const down = scf_result.vxc_r_down.?;
    const avg = try alloc.alloc(f64, up.len);
    for (0..up.len) |i| {
        avg[i] = (up[i] + down[i]) * 0.5;
    }
    return .{ .avg = avg, .values = avg };
}

fn paw_dij_slice(scf_result: *const scf.ScfResult) ?[]const []const f64 {
    if (scf_result.paw_dij) |dij| {
        return @as([]const []const f64, dij);
    }
    return null;
}

fn paw_rhoij_slice(scf_result: *const scf.ScfResult) ?[]const []const f64 {
    if (scf_result.paw_rhoij) |rij| {
        return @as([]const []const f64, rij);
    }
    return null;
}

fn compute_relax_forces(
    ctx: *const RelaxContext,
    state: *const RelaxState,
    scf_result: *const scf.ScfResult,
) !forces_mod.ForceTerms {
    const grid = forces_mod.Grid{
        .nx = scf_result.potential.nx,
        .ny = scf_result.potential.ny,
        .nz = scf_result.potential.nz,
        .min_h = scf_result.potential.min_h,
        .min_k = scf_result.potential.min_k,
        .min_l = scf_result.potential.min_l,
        .cell = state.current_cell,
        .recip = state.current_recip,
    };
    const rho_g = try density_to_reciprocal(
        ctx.alloc,
        ctx.io,
        grid,
        scf_result.density,
        ctx.cfg.scf.fft_backend,
    );
    defer ctx.alloc.free(rho_g);

    const coulomb_r_cut = if (ctx.cfg.boundary == .isolated)
        coulomb_mod.cutoff_radius(state.current_cell)
    else
        null;
    var vxc_input = try build_force_vxc_input(ctx.alloc, scf_result);
    defer vxc_input.deinit(ctx.alloc);

    const force_terms = try forces_mod.compute_forces(
        ctx.alloc,
        ctx.io,
        grid,
        rho_g,
        scf_result.potential.values,
        scf_result.ionic_g,
        ctx.species,
        state.atoms,
        state.current_cell,
        state.current_recip,
        state.current_volume,
        state.alpha,
        ctx.tables.local_cfg,
        scf_result.wavefunctions,
        if (scf_result.vresid) |vresid| vresid.values else null,
        ctx.cfg.scf.quiet,
        ctx.tables.force_radial_tables(),
        vxc_input.values,
        ctx.tables.rho_atom_tables(),
        ctx.tables.rho_core_tables(),
        ctx.tables.ff_tables(),
        coulomb_r_cut,
        ctx.cfg.vdw,
        scf_result.paw_tabs,
        paw_dij_slice(scf_result),
        paw_rhoij_slice(scf_result),
        scf_result.wavefunctions_down,
    );
    average_forces_in_place(force_terms.total);
    return force_terms;
}

fn maybe_compute_relax_stress(
    ctx: *const RelaxContext,
    state: *const RelaxState,
    scf_result: *const scf.ScfResult,
) !RelaxStressInfo {
    var info = RelaxStressInfo{};
    if (!ctx.cfg.relax.cell_relax) return info;

    const step_model = build_step_model(
        ctx.species,
        state.atoms,
        state.current_cell,
        state.current_recip,
        state.current_volume,
    );
    const stress_terms = try stress_mod.compute_stress_from_scf(
        ctx.alloc,
        ctx.io,
        @constCast(scf_result),
        ctx.relax_cfg,
        &step_model,
    );
    var sym_total = stress_terms.total;
    if (ctx.cfg.scf.symmetry) {
        const symmetry = @import("../symmetry/symmetry.zig");
        const sym_ops = try symmetry.get_symmetry_ops(
            ctx.alloc,
            state.current_cell,
            state.atoms,
            1e-5,
        );
        defer ctx.alloc.free(sym_ops);

        if (sym_ops.len > 1) {
            sym_total = stress_mod.symmetrize_stress(sym_total, sym_ops, state.current_cell);
        }
    }

    const ry_bohr3_to_gpa = 14710.507;
    if (ctx.cfg.relax.target_pressure != 0.0) {
        const p_offset = -ctx.cfg.relax.target_pressure / ry_bohr3_to_gpa;
        for (0..3) |aa| sym_total[aa][aa] += p_offset;
    }
    info.cached_stress_total = sym_total;

    var max_stress_gpa: f64 = 0.0;
    for (0..3) |aa| {
        for (0..3) |bb| {
            const dev = @abs(sym_total[aa][bb]) * ry_bohr3_to_gpa;
            if (dev > max_stress_gpa) max_stress_gpa = dev;
        }
    }
    info.stress_converged = max_stress_gpa < ctx.cfg.relax.stress_tol;
    const p = -(sym_total[0][0] + sym_total[1][1] + sym_total[2][2]) / 3.0;
    try log_vc_relax_iter(
        ctx.io,
        state.iterations - 1,
        p * ry_bohr3_to_gpa,
        max_stress_gpa,
        state.current_volume,
    );
    return info;
}

fn replace_final_forces(
    alloc: std.mem.Allocator,
    state: *RelaxState,
    forces: []const math.Vec3,
) !void {
    if (state.final_forces.len > 0) alloc.free(state.final_forces);
    state.final_forces = try alloc.alloc(math.Vec3, forces.len);
    @memcpy(state.final_forces, forces);
}

fn log_relax_step_profile(
    io: std.Io,
    iter: usize,
    scf_iterations: usize,
    relax_step_start: std.Io.Clock.Timestamp,
    relax_step_cpu_start: i64,
    scf_start: std.Io.Clock.Timestamp,
    scf_end: std.Io.Clock.Timestamp,
    force_start: std.Io.Clock.Timestamp,
    force_end: std.Io.Clock.Timestamp,
) void {
    const step_cpu_end = timing_mod.Timing.get_cpu_time_us();
    const scf_ns = scf_start.durationTo(scf_end).raw.nanoseconds;
    const force_ns = force_start.durationTo(force_end).raw.nanoseconds;
    const step_ns = relax_step_start.untilNow(io).raw.nanoseconds;
    const scf_ms = @as(f64, @floatFromInt(@as(u64, @intCast(scf_ns)))) / 1_000_000.0;
    const force_ms = @as(f64, @floatFromInt(@as(u64, @intCast(force_ns)))) / 1_000_000.0;
    const step_ms = @as(f64, @floatFromInt(@as(u64, @intCast(step_ns)))) / 1_000_000.0;
    const cpu_ms = @as(f64, @floatFromInt(step_cpu_end - relax_step_cpu_start)) / 1_000.0;
    var tbuf: [512]u8 = undefined;
    const msg = std.fmt.bufPrint(
        &tbuf,
        "relax_step_profile iter={d} scf_iters={d} scf_ms={d:.1}" ++
            " force_ms={d:.1} wall_ms={d:.1} cpu_ms={d:.1}\n",
        .{ iter, scf_iterations, scf_ms, force_ms, step_ms, cpu_ms },
    ) catch "";
    std.Io.File.stderr().writeStreamingAll(io, msg) catch {};
}

fn maybe_finish_relaxation(
    ctx: *const RelaxContext,
    state: *RelaxState,
    scf_result: *scf.ScfResult,
    max_force: f64,
    stress_info: RelaxStressInfo,
) !bool {
    if (max_force >= ctx.cfg.relax.force_tol or !stress_info.stress_converged) return false;
    state.converged = true;
    state.warmstart.save_final_potential(scf_result);
    try log_relax_converged(ctx.io, state.iterations - 1, state.final_energy, max_force);
    return true;
}

fn append_trajectory_step(
    alloc: std.mem.Allocator,
    trajectory: *?std.ArrayList(RelaxStep),
    atoms: []const hamiltonian.AtomData,
    forces: []const math.Vec3,
    energy: f64,
    max_force: f64,
) !void {
    if (trajectory.* == null) return;
    const step_atoms = try alloc.alloc(hamiltonian.AtomData, atoms.len);
    @memcpy(step_atoms, atoms);
    const step_forces = try alloc.alloc(math.Vec3, forces.len);
    @memcpy(step_forces, forces);
    try trajectory.*.?.append(alloc, .{
        .atoms = step_atoms,
        .energy = energy,
        .forces = step_forces,
        .max_force = max_force,
    });
}

fn ensure_vec3_buffer(
    alloc: std.mem.Allocator,
    slot: *?[]math.Vec3,
    len: usize,
) ![]math.Vec3 {
    if (slot.* == null) {
        slot.* = try alloc.alloc(math.Vec3, len);
    }
    return slot.*.?;
}

fn snapshot_positions(
    alloc: std.mem.Allocator,
    atoms: []const hamiltonian.AtomData,
) ![]math.Vec3 {
    const positions = try alloc.alloc(math.Vec3, atoms.len);
    for (atoms, 0..) |atom, i| {
        positions[i] = atom.position;
    }
    return positions;
}

fn remember_optimizer_history(
    alloc: std.mem.Allocator,
    state: *RelaxState,
    current_positions: []const math.Vec3,
    forces: []const math.Vec3,
) !void {
    const prev_positions = try ensure_vec3_buffer(
        alloc,
        &state.prev_positions,
        current_positions.len,
    );
    const prev_forces = try ensure_vec3_buffer(alloc, &state.prev_forces, forces.len);
    @memcpy(prev_positions, current_positions);
    @memcpy(prev_forces, forces);
}

fn save_backtrack_step(
    alloc: std.mem.Allocator,
    state: *RelaxState,
    displacement: []const math.Vec3,
) !void {
    const saved_positions = try ensure_vec3_buffer(alloc, &state.saved_positions, state.atoms.len);
    const saved_displacement = try ensure_vec3_buffer(
        alloc,
        &state.saved_displacement,
        displacement.len,
    );
    for (state.atoms, 0..) |atom, i| {
        saved_positions[i] = atom.position;
    }
    @memcpy(saved_displacement, displacement);
}

fn apply_displacement(
    atoms: []hamiltonian.AtomData,
    displacement: []const math.Vec3,
) void {
    for (0..atoms.len) |i| {
        atoms[i].position = math.Vec3.add(atoms[i].position, displacement[i]);
    }
}

fn maybe_update_cell_after_step(
    ctx: *RelaxContext,
    state: *RelaxState,
    stress_info: RelaxStressInfo,
) !void {
    if (!ctx.cfg.relax.cell_relax or stress_info.stress_converged) return;
    try update_cell_from_stress(
        &state.current_cell,
        &state.current_recip,
        &state.current_volume,
        stress_info.cached_stress_total.?,
        state.atoms,
        ctx.cfg.relax.cell_step,
    );
    ctx.relax_cfg.cell = state.current_cell;
    state.warmstart.invalidate_for_cell_change(ctx.alloc);
    if (ctx.cfg.ewald.alpha <= 0.0) {
        state.alpha = compute_default_alpha(state.current_cell);
    }
    state.opt.reset();
    clear_relax_history(state, ctx.alloc);
}

fn advance_relax_step(
    ctx: *RelaxContext,
    state: *RelaxState,
    forces: []const math.Vec3,
    stress_info: RelaxStressInfo,
) !void {
    const current_positions = try snapshot_positions(ctx.alloc, state.atoms);
    defer ctx.alloc.free(current_positions);

    if (state.prev_positions != null and state.prev_forces != null) {
        state.opt.update(state.prev_positions.?, current_positions, state.prev_forces.?, forces);
    }
    const displacement = try state.opt.step(ctx.alloc, forces, ctx.cfg.relax.max_step);
    defer ctx.alloc.free(displacement);

    try remember_optimizer_history(ctx.alloc, state, current_positions, forces);
    try save_backtrack_step(ctx.alloc, state, displacement);
    apply_displacement(state.atoms, displacement);
    try maybe_update_cell_after_step(ctx, state, stress_info);
}

fn relax_iterations(
    ctx: *RelaxContext,
    state: *RelaxState,
) !void {
    for (0..ctx.cfg.relax.max_iter) |iter| {
        state.iterations = iter + 1;
        const relax_step_start = std.Io.Clock.Timestamp.now(ctx.io, .awake);
        const relax_step_cpu_start = timing_mod.Timing.get_cpu_time_us();
        try log_relax_iter(ctx.io, iter, null, null);

        const scf_start = std.Io.Clock.Timestamp.now(ctx.io, .awake);
        var scf_result = try run_relax_scf(ctx, state);
        defer scf_result.deinit(ctx.alloc);

        const scf_end = std.Io.Clock.Timestamp.now(ctx.io, .awake);

        try state.warmstart.update_from_scf(ctx.alloc, &scf_result);
        state.final_energy = scf_result.energy.total;
        if (try handle_backtrack(ctx, state)) continue;

        const force_start = std.Io.Clock.Timestamp.now(ctx.io, .awake);
        var force_terms = try compute_relax_forces(ctx, state, &scf_result);
        defer force_terms.deinit(ctx.alloc);

        const force_end = std.Io.Clock.Timestamp.now(ctx.io, .awake);

        const stress_info = try maybe_compute_relax_stress(ctx, state, &scf_result);
        try replace_final_forces(ctx.alloc, state, force_terms.total);
        const max_force = forces_mod.max_force(force_terms.total);
        try log_relax_iter(ctx.io, iter, state.final_energy, max_force);
        log_relax_step_profile(
            ctx.io,
            iter,
            scf_result.iterations,
            relax_step_start,
            relax_step_cpu_start,
            scf_start,
            scf_end,
            force_start,
            force_end,
        );
        if (try maybe_finish_relaxation(ctx, state, &scf_result, max_force, stress_info)) break;
        try append_trajectory_step(
            ctx.alloc,
            &state.trajectory,
            state.atoms,
            force_terms.total,
            state.final_energy,
            max_force,
        );
        try advance_relax_step(ctx, state, force_terms.total, stress_info);
    }
}

/// Run structure relaxation.
pub fn run(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    species: []const hamiltonian.SpeciesEntry,
    initial_atoms: []const hamiltonian.AtomData,
    cell: math.Mat3, // Cell in Bohr units
    recip: math.Mat3,
    volume: f64,
) !RelaxResult {
    var ctx = try init_relax_context(alloc, io, cfg, species, cell);
    defer ctx.deinit();

    var state = try init_relax_state(alloc, cfg, initial_atoms, cell, recip, volume);
    defer state.deinit(alloc);

    try relax_iterations(&ctx, &state);
    if (!state.converged) {
        try log_relax_not_converged(
            io,
            state.iterations,
            state.final_energy,
            forces_mod.max_force(state.final_forces),
        );
    }
    return try state.take_result(alloc, cfg);
}

/// Update cell parameters based on stress tensor.
/// Uses steepest descent on the strain: ε_αβ = -step × σ_αβ / |σ|
/// Then cell' = (I + ε) × cell, and atoms are updated in fractional coordinates.
fn update_cell_from_stress(
    cell_ptr: *math.Mat3,
    recip_ptr: *math.Mat3,
    volume_ptr: *f64,
    sigma: stress_mod.Stress3x3,
    atoms: []hamiltonian.AtomData,
    max_strain: f64,
) !void {
    // Compute strain from stress: ε = -step × σ (simple steepest descent)
    // Scale so max strain component ≤ max_strain
    var max_s: f64 = 0.0;
    for (0..3) |a| {
        for (0..3) |b| {
            if (@abs(sigma[a][b]) > max_s) max_s = @abs(sigma[a][b]);
        }
    }
    if (max_s < 1e-15) return;

    // σ_αβ = (1/Ω) ∂E/∂ε_αβ
    // Steepest descent: Δε = -α × σ (move downhill in energy)
    // σ < 0 → expand, σ > 0 → contract
    const step = max_strain / max_s;
    var strain: [3][3]f64 = undefined;
    for (0..3) |a| {
        for (0..3) |b| {
            strain[a][b] = -step * sigma[a][b];
        }
    }
    // Symmetrize strain (should already be symmetric, but enforce)
    for (0..3) |a| {
        for (a + 1..3) |b| {
            const avg = (strain[a][b] + strain[b][a]) / 2.0;
            strain[a][b] = avg;
            strain[b][a] = avg;
        }
    }

    // Compute fractional coordinates of atoms BEFORE cell update
    // frac = pos × cell^{-1} (row-vector convention: pos_j = Σ_i frac_i × cell[i][j])
    const c = cell_ptr.m;
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

    // Update cell: cell' = (I + ε) × cell
    // cell'[i][j] = Σ_k (δ_ik + ε_ik) × cell[k][j]
    //             = cell[i][j] + Σ_k ε_ik × cell[k][j]
    var new_cell: [3][3]f64 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            var s: f64 = c[i][j]; // δ_ik × cell[k][j] = cell[i][j]
            for (0..3) |k| {
                s += strain[i][k] * c[k][j];
            }
            new_cell[i][j] = s;
        }
    }
    cell_ptr.* = math.Mat3{ .m = new_cell };

    // Recompute reciprocal lattice and volume
    recip_ptr.* = math.reciprocal(cell_ptr.*);
    const a1 = cell_ptr.row(0);
    const a2 = cell_ptr.row(1);
    const a3 = cell_ptr.row(2);
    volume_ptr.* = @abs(math.Vec3.dot(a1, math.Vec3.cross(a2, a3)));

    // Update Cartesian positions from fractional coordinates
    // pos' = frac × new_cell, where frac = pos × old_cell_inv
    for (atoms) |*atom| {
        const pos = atom.position;
        const frac = math.Vec3{
            .x = pos.x * c_inv[0][0] + pos.y * c_inv[1][0] + pos.z * c_inv[2][0],
            .y = pos.x * c_inv[0][1] + pos.y * c_inv[1][1] + pos.z * c_inv[2][1],
            .z = pos.x * c_inv[0][2] + pos.y * c_inv[1][2] + pos.z * c_inv[2][2],
        };
        atom.position = .{
            .x = frac.x * new_cell[0][0] + frac.y * new_cell[1][0] + frac.z * new_cell[2][0],
            .y = frac.x * new_cell[0][1] + frac.y * new_cell[1][1] + frac.z * new_cell[2][1],
            .z = frac.x * new_cell[0][2] + frac.y * new_cell[1][2] + frac.z * new_cell[2][2],
        };
    }
}

/// Compute default Ewald alpha parameter.
fn compute_default_alpha(cell: math.Mat3) f64 {
    const lmin = @min(
        @min(math.Vec3.norm(cell.row(0)), math.Vec3.norm(cell.row(1))),
        math.Vec3.norm(cell.row(2)),
    );
    return 5.0 / lmin;
}

/// Convert FFT index to frequency.
fn index_to_freq(i: usize, n: usize) i32 {
    const half = (n - 1) / 2;
    return if (i <= half) @as(i32, @intCast(i)) else @as(i32, @intCast(i)) - @as(i32, @intCast(n));
}

/// Convert real-space density to reciprocal space using FFT.
/// The output is reordered to match the grid's min_h, min_k, min_l layout
/// for consistency with SCF calculations.
fn density_to_reciprocal(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: forces_mod.Grid,
    density: []const f64,
    fft_backend: config_mod.FftBackend,
) ![]math.Complex {
    const fft = @import("../fft/fft.zig");

    const nx = grid.nx;
    const ny = grid.ny;
    const nz = grid.nz;
    const total = nx * ny * nz;

    if (density.len != total) return error.InvalidDensitySize;

    // Convert to complex and allocate output
    var data = try alloc.alloc(math.Complex, total);
    defer alloc.free(data);

    for (density, 0..) |d, i| {
        data[i] = math.complex.init(d, 0.0);
    }

    // 3D FFT in place using the specified backend
    var plan = try fft.Fft3dPlan.init_with_backend(alloc, io, nx, ny, nz, fft_backend);
    defer plan.deinit(alloc);

    plan.forward(data);

    // Scale by 1/N and reorder to match grid layout (min_h, min_k, min_l)
    const scale = 1.0 / @as(f64, @floatFromInt(total));
    var out = try alloc.alloc(math.Complex, total);

    var idx: usize = 0;
    var z: usize = 0;
    while (z < nz) : (z += 1) {
        var y: usize = 0;
        while (y < ny) : (y += 1) {
            var x: usize = 0;
            while (x < nx) : (x += 1) {
                const fh = index_to_freq(x, nx);
                const fk = index_to_freq(y, ny);
                const fl = index_to_freq(z, nz);
                const th = @as(usize, @intCast(fh - grid.min_h));
                const tk = @as(usize, @intCast(fk - grid.min_k));
                const tl = @as(usize, @intCast(fl - grid.min_l));
                const out_idx = th + nx * (tk + ny * tl);
                out[out_idx] = math.complex.scale(data[idx], scale);
                idx += 1;
            }
        }
    }

    return out;
}

fn log_warn(io: std.Io, msg: []const u8) !void {
    const logger = runtime_logging.stderr(io, .warn);
    try logger.print(.warn, "WARNING: {s}\n", .{msg});
}

fn log_relax_iter(io: std.Io, iter: usize, energy: ?f64, max_force: ?f64) !void {
    const logger = runtime_logging.stderr(io, .info);
    if (energy != null and max_force != null) {
        try logger.print(
            .info,
            "relax iter={d} energy={d:.8} max_force={d:.6}\n",
            .{ iter + 1, energy.?, max_force.? },
        );
    } else {
        try logger.print(.info, "relax iter={d} starting...\n", .{iter + 1});
    }
}

fn log_relax_converged(io: std.Io, iter: usize, energy: f64, max_force: f64) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(
        .info,
        "relax CONVERGED after {d} iterations, energy={d:.8} Ry," ++
            " max_force={d:.6} Ry/Bohr\n",
        .{ iter + 1, energy, max_force },
    );
}

fn log_backtrack(io: std.Io, count: usize, new_energy: f64, prev_energy_val: f64) !void {
    const scale = std.math.pow(f64, 0.5, @as(f64, @floatFromInt(count)));
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(
        .info,
        "relax BACKTRACK #{d}: E={d:.8} > E_prev={d:.8}" ++
            " (dE={d:.6}), scale={d:.4}\n",
        .{ count, new_energy, prev_energy_val, new_energy - prev_energy_val, scale },
    );
}

fn log_vc_relax_iter(
    io: std.Io,
    iter: usize,
    pressure_gpa: f64,
    max_stress_gpa: f64,
    volume: f64,
) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(
        .info,
        "vc-relax iter={d} P={d:.2} GPa  max_stress={d:.4} GPa" ++
            "  vol={d:.4} Bohr³\n",
        .{ iter, pressure_gpa, max_stress_gpa, volume },
    );
}

fn log_relax_not_converged(io: std.Io, iter: usize, energy: f64, max_force: f64) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(
        .info,
        "relax NOT CONVERGED after {d} iterations, energy={d:.8} Ry," ++
            " max_force={d:.6} Ry/Bohr\n",
        .{ iter, energy, max_force },
    );
}

/// Write relaxation results to output files.
pub fn write_output(
    alloc: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    result: *const RelaxResult,
    species: []const hamiltonian.SpeciesEntry,
    unit_scale_ang: f64,
) !void {
    // Write relax_status.txt
    {
        var file = try dir.createFile(io, "relax_status.txt", .{ .truncate = true });
        defer file.close(io);

        var buffer: [4096]u8 = undefined;
        var writer = file.writer(io, &buffer);
        const out = &writer.interface;

        try out.print("converged = {s}\n", .{if (result.converged) "true" else "false"});
        try out.print("iterations = {d}\n", .{result.iterations});
        try out.print("final_energy_ry = {d:.10}\n", .{result.final_energy});
        try out.print("final_energy_ev = {d:.10}\n", .{result.final_energy * 13.6057});

        if (result.final_forces.len > 0) {
            const max_force = forces_mod.max_force(result.final_forces);
            const rms_force = forces_mod.rms_force(result.final_forces);
            try out.print("max_force_ry_bohr = {d:.10}\n", .{max_force});
            try out.print("rms_force_ry_bohr = {d:.10}\n", .{rms_force});
        }
        try out.flush();
    }

    // Write relax_final.xyz - final structure in XYZ format
    {
        var file = try dir.createFile(io, "relax_final.xyz", .{ .truncate = true });
        defer file.close(io);

        var buffer: [8192]u8 = undefined;
        var writer = file.writer(io, &buffer);
        const out = &writer.interface;

        try out.print("{d}\n", .{result.final_atoms.len});
        try out.print("Final relaxed structure (Angstrom)\n", .{});

        for (result.final_atoms) |atom| {
            const symbol = species[atom.species_index].symbol;
            // Convert from Bohr to Angstrom
            const x = atom.position.x * unit_scale_ang;
            const y = atom.position.y * unit_scale_ang;
            const z = atom.position.z * unit_scale_ang;
            try out.print("{s} {d:.10} {d:.10} {d:.10}\n", .{ symbol, x, y, z });
        }
        try out.flush();
    }

    // Write relax_forces.csv - final forces
    if (result.final_forces.len > 0) {
        var file = try dir.createFile(io, "relax_forces.csv", .{ .truncate = true });
        defer file.close(io);

        var buffer: [8192]u8 = undefined;
        var writer = file.writer(io, &buffer);
        const out = &writer.interface;

        try out.writeAll("atom,symbol,fx_ry_bohr,fy_ry_bohr,fz_ry_bohr,f_mag_ry_bohr\n");

        for (result.final_forces, 0..) |f, i| {
            const symbol = species[result.final_atoms[i].species_index].symbol;
            const mag = math.Vec3.norm(f);
            try out.print(
                "{d},{s},{d:.10},{d:.10},{d:.10},{d:.10}\n",
                .{ i, symbol, f.x, f.y, f.z, mag },
            );
        }
        try out.flush();
    }

    // Write relax_trajectory.csv if trajectory is available
    if (result.trajectory) |traj| {
        var file = try dir.createFile(io, "relax_trajectory.csv", .{ .truncate = true });
        defer file.close(io);

        var buffer: [8192]u8 = undefined;
        var writer = file.writer(io, &buffer);
        const out = &writer.interface;

        try out.writeAll("iter,energy_ry,max_force_ry_bohr\n");

        for (traj, 0..) |step, i| {
            try out.print("{d},{d:.10},{d:.10}\n", .{ i + 1, step.energy, step.max_force });
        }
        try out.flush();
    }

    _ = alloc; // Reserved for future use
}

/// Write trajectory in extended XYZ format for visualization.
/// Compatible with ASE, OVITO, and other tools.
pub fn write_trajectory_xyz(
    io: std.Io,
    dir: std.Io.Dir,
    result: *const RelaxResult,
    species: []const hamiltonian.SpeciesEntry,
    cell: math.Mat3,
    unit_scale_ang: f64,
) !void {
    if (result.trajectory == null) return;
    const traj = result.trajectory.?;
    if (traj.len == 0) return;

    var file = try dir.createFile(io, "relax_trajectory.xyz", .{ .truncate = true });
    defer file.close(io);

    var buffer: [16384]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const out = &writer.interface;

    // Cell in Angstrom
    const c = cell.scale(unit_scale_ang);

    for (traj, 0..) |step, iter| {
        const n_atoms = step.atoms.len;

        // Number of atoms
        try out.print("{d}\n", .{n_atoms});

        // Extended XYZ comment line with lattice, energy, properties
        try out.print(
            "Lattice=\"{d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8}" ++
                " {d:.8} {d:.8} {d:.8}\" ",
            .{
                c.m[0][0], c.m[0][1], c.m[0][2],
                c.m[1][0], c.m[1][1], c.m[1][2],
                c.m[2][0], c.m[2][1], c.m[2][2],
            },
        );
        try out.print("Properties=species:S:1:pos:R:3:forces:R:3 ", .{});
        try out.print("energy={d:.10} ", .{step.energy * 13.6057}); // eV
        // eV/Angstrom
        try out.print("max_force={d:.10} ", .{step.max_force * 13.6057 / 0.529177});
        try out.print("iter={d} ", .{iter + 1});
        try out.print("pbc=\"T T T\"\n", .{});

        // Atom lines: symbol x y z fx fy fz
        for (step.atoms, 0..) |atom, i| {
            const sym = species[atom.species_index].symbol;
            // Position in Angstrom
            const x = atom.position.x * unit_scale_ang;
            const y = atom.position.y * unit_scale_ang;
            const z = atom.position.z * unit_scale_ang;
            // Force in eV/Angstrom
            const force_scale = 13.6057 / 0.529177; // Ry/Bohr -> eV/Angstrom
            const fx = step.forces[i].x * force_scale;
            const fy = step.forces[i].y * force_scale;
            const fz = step.forces[i].z * force_scale;
            try out.print(
                "{s} {d:.10} {d:.10} {d:.10} {d:.10} {d:.10} {d:.10}\n",
                .{ sym, x, y, z, fx, fy, fz },
            );
        }
        try out.flush();
    }

    // Also write final structure
    {
        const n_atoms = result.final_atoms.len;
        try out.print("{d}\n", .{n_atoms});
        try out.print(
            "Lattice=\"{d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8}" ++
                " {d:.8} {d:.8} {d:.8}\" ",
            .{
                c.m[0][0], c.m[0][1], c.m[0][2],
                c.m[1][0], c.m[1][1], c.m[1][2],
                c.m[2][0], c.m[2][1], c.m[2][2],
            },
        );
        try out.print("Properties=species:S:1:pos:R:3:forces:R:3 ", .{});
        try out.print("energy={d:.10} ", .{result.final_energy * 13.6057});
        try out.print("converged={s} ", .{if (result.converged) "true" else "false"});
        try out.print("pbc=\"T T T\"\n", .{});

        for (result.final_atoms, 0..) |atom, i| {
            const sym = species[atom.species_index].symbol;
            const x = atom.position.x * unit_scale_ang;
            const y = atom.position.y * unit_scale_ang;
            const z = atom.position.z * unit_scale_ang;
            const force_scale = 13.6057 / 0.529177;
            var fx: f64 = 0.0;
            var fy: f64 = 0.0;
            var fz: f64 = 0.0;
            if (result.final_forces.len > i) {
                fx = result.final_forces[i].x * force_scale;
                fy = result.final_forces[i].y * force_scale;
                fz = result.final_forces[i].z * force_scale;
            }
            try out.print(
                "{s} {d:.10} {d:.10} {d:.10} {d:.10} {d:.10} {d:.10}\n",
                .{ sym, x, y, z, fx, fy, fz },
            );
        }
        try out.flush();
    }
}
