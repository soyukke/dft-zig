const std = @import("std");
const math = @import("../math/math.zig");
const config_mod = @import("../config/config.zig");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const paw_mod = @import("../paw/paw.zig");
const scf = @import("../scf/scf.zig");
const forces_mod = @import("../forces/forces.zig");
pub const optimizer = @import("optimizer.zig");

const stress_mod = @import("../stress/stress.zig");
const output = @import("../dft/output.zig");

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

/// Run structure relaxation.
pub fn run(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    species: []hamiltonian.SpeciesEntry,
    initial_atoms: []hamiltonian.AtomData,
    cell: math.Mat3, // Cell in Bohr units
    recip: math.Mat3,
    volume: f64,
) !RelaxResult {
    const n_atoms = initial_atoms.len;

    // Disable symmetry during relaxation to ensure consistent k-point sets
    // across iterations. Atom displacements change symmetry, causing different
    // k-point reductions and energy discontinuities that break backtracking.
    var relax_cfg = cfg;
    if (cfg.scf.symmetry) {
        try logWarn(io, "scf.symmetry is enabled, but will be disabled during relaxation " ++
            "to ensure consistent k-point sets across iterations. " ++
            "Set scf.symmetry = false to suppress this warning.");
    }
    relax_cfg.scf.symmetry = false;

    // Enable stress computation for vc-relax
    if (cfg.relax.cell_relax) {
        relax_cfg.scf.compute_stress = true;
        // Override units to bohr since we'll update cell in Bohr
        relax_cfg.units = .bohr;
        relax_cfg.cell = cell; // cell is already in Bohr
    }

    // Mutable cell for vc-relax
    var current_cell = cell;
    var current_recip = recip;
    var current_volume = volume;

    // Copy initial atoms
    var atoms = try alloc.alloc(hamiltonian.AtomData, n_atoms);
    for (initial_atoms, 0..) |atom, i| {
        atoms[i] = atom;
    }

    // Initialize optimizer
    var opt = try optimizer.Optimizer.init(alloc, cfg.relax.algorithm, n_atoms, initial_atoms, cell);
    defer opt.deinit(alloc);

    // Trajectory storage
    var trajectory: ?std.ArrayList(RelaxStep) = if (cfg.relax.output_trajectory)
        .empty
    else
        null;
    defer if (trajectory) |*t| {
        for (t.items) |*step| {
            step.deinit(alloc);
        }
        t.deinit(alloc);
    };

    var prev_positions: ?[]math.Vec3 = null;
    defer if (prev_positions) |pp| alloc.free(pp);
    var prev_forces: ?[]math.Vec3 = null;
    defer if (prev_forces) |pf| alloc.free(pf);

    var converged = false;
    var final_energy: f64 = 0.0;
    var final_forces: []math.Vec3 = &[_]math.Vec3{};
    var iterations: usize = 0;

    // Backtracking line search state
    var prev_energy: f64 = std.math.inf(f64);
    var saved_positions: ?[]math.Vec3 = null;
    defer if (saved_positions) |sp| alloc.free(sp);
    var saved_displacement: ?[]math.Vec3 = null;
    defer if (saved_displacement) |sd| alloc.free(sd);
    var backtrack_count: usize = 0;
    const max_backtrack: usize = 5;

    // Density warmstart: reuse converged density from previous SCF
    var prev_density: ?[]f64 = null;
    defer if (prev_density) |pd| alloc.free(pd);

    // Wavefunction warmstart: reuse eigenvectors from previous SCF
    var prev_kpoint_cache: ?[]scf.KpointCache = null;
    defer if (prev_kpoint_cache) |cache| {
        for (cache) |*c| c.deinit();
        alloc.free(cache);
    };

    // NonlocalContext warmstart: reuse apply caches across relax steps
    var prev_apply_caches: ?[]scf.KpointApplyCache = null;
    defer if (prev_apply_caches) |caches| {
        for (caches) |*ac| ac.deinit(alloc);
        alloc.free(caches);
    };

    // Save final potential for band calculation (skip post-relax SCF)
    var final_potential_saved: ?hamiltonian.PotentialGrid = null;
    errdefer if (final_potential_saved) |*p| p.deinit(alloc);

    // Ewald alpha from config (or compute default); recomputed per step for vc-relax
    var alpha = if (cfg.ewald.alpha > 0.0) cfg.ewald.alpha else computeDefaultAlpha(cell);

    // Build radial lookup tables for nonlocal force (position-independent, reused across steps)
    const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
    const g_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 1.5;
    var force_radial_tables_buf = try alloc.alloc(nonlocal_mod.RadialTableSet, species.len);
    for (species, 0..) |entry, si| {
        const upf = entry.upf.*;
        if (upf.beta.len == 0 or upf.dij.len == 0) {
            force_radial_tables_buf[si] = .{ .tables = &[_]nonlocal_mod.RadialTable{} };
            continue;
        }
        force_radial_tables_buf[si] = try nonlocal_mod.RadialTableSet.init(alloc, upf.beta, upf.r, upf.rab, g_max);
    }
    defer {
        for (force_radial_tables_buf) |*t| {
            if (t.tables.len > 0) t.deinit(alloc);
        }
        alloc.free(force_radial_tables_buf);
    }
    const force_radial_tables: ?[]nonlocal_mod.RadialTableSet = force_radial_tables_buf;

    // Build local form factor lookup tables (position-independent, reused across steps)
    const form_factor_mod = @import("../pseudopotential/form_factor.zig");
    const local_alpha = if (cfg.ewald.alpha > 0.0) cfg.ewald.alpha else computeDefaultAlpha(cell);
    const ff_q_max = @sqrt(2.0 * cfg.scf.ecut_ry) * 3.0;
    var ff_tables_buf = try alloc.alloc(form_factor_mod.LocalFormFactorTable, species.len);
    for (species, 0..) |entry, si| {
        ff_tables_buf[si] = try form_factor_mod.LocalFormFactorTable.init(
            alloc,
            entry.upf.*,
            entry.z_valence,
            cfg.scf.local_potential,
            local_alpha,
            ff_q_max,
        );
    }
    defer {
        for (ff_tables_buf) |*t| t.deinit(alloc);
        alloc.free(ff_tables_buf);
    }
    const ff_tables: ?[]const form_factor_mod.LocalFormFactorTable = ff_tables_buf;

    // Build radial form factor tables for residual force (rhoAtomG and rhoCoreG)
    var rho_atom_tables_buf = try alloc.alloc(form_factor_mod.RadialFormFactorTable, species.len);
    var rho_core_tables_buf = try alloc.alloc(form_factor_mod.RadialFormFactorTable, species.len);
    for (species, 0..) |entry, si| {
        rho_atom_tables_buf[si] = try form_factor_mod.RadialFormFactorTable.initRhoAtom(alloc, entry.upf.*, ff_q_max);
        rho_core_tables_buf[si] = try form_factor_mod.RadialFormFactorTable.initRhoCore(alloc, entry.upf.*, ff_q_max);
    }
    defer {
        for (rho_atom_tables_buf) |*t| t.deinit(alloc);
        alloc.free(rho_atom_tables_buf);
        for (rho_core_tables_buf) |*t| t.deinit(alloc);
        alloc.free(rho_core_tables_buf);
    }
    const rho_atom_tables: ?[]const form_factor_mod.RadialFormFactorTable = rho_atom_tables_buf;
    const rho_core_tables: ?[]const form_factor_mod.RadialFormFactorTable = rho_core_tables_buf;

    for (0..cfg.relax.max_iter) |iter| {
        iterations = iter + 1;
        const relax_step_start = std.Io.Clock.Timestamp.now(io, .awake);
        const relax_step_cpu_start = output.Timing.getCpuTimeUs();

        // Log iteration start
        try logRelaxIter(io, iter, null, null);

        // Run SCF calculation (with density + wavefunction + nonlocal warmstart)
        const scf_start = std.Io.Clock.Timestamp.now(io, .awake);
        var scf_result = try scf.run(.{
            .alloc = alloc,
            .io = io,
            .cfg = relax_cfg,
            .species = species,
            .atoms = atoms,
            .recip = current_recip,
            .volume_bohr = current_volume,
            .initial_density = if (prev_density) |pd| pd else null,
            .initial_kpoint_cache = prev_kpoint_cache,
            .initial_apply_caches = prev_apply_caches,
            .ff_tables = ff_tables,
        });
        defer scf_result.deinit(alloc);
        const scf_end = std.Io.Clock.Timestamp.now(io, .awake);

        // Save converged density for warmstarting next SCF iteration
        {
            if (prev_density) |pd| alloc.free(pd);
            prev_density = try alloc.alloc(f64, scf_result.density.len);
            @memcpy(prev_density.?, scf_result.density);
        }

        // Save eigenvector cache for warmstarting next SCF iteration
        {
            if (prev_kpoint_cache) |old_cache| {
                for (old_cache) |*c| c.deinit();
                alloc.free(old_cache);
            }
            prev_kpoint_cache = scf_result.kpoint_cache;
            scf_result.kpoint_cache = null;
        }

        // Save apply caches (NonlocalContext + PwGridMap) for next relax step
        {
            // No need to free old - ownership was transferred to scf.run()
            prev_apply_caches = scf_result.apply_caches;
            scf_result.apply_caches = null;
        }

        final_energy = scf_result.energy.total;

        // Backtracking line search: reject step if energy increased
        // Disabled for vc-relax: cell changes invalidate caches causing energy jumps
        if (!cfg.relax.cell_relax and final_energy > prev_energy and saved_positions != null and saved_displacement != null) {
            backtrack_count += 1;
            try logBacktrack(io, backtrack_count, final_energy, prev_energy);

            if (backtrack_count > max_backtrack) {
                // Give up backtracking, reset Hessian and continue with current position
                opt.reset();
                backtrack_count = 0;
                prev_energy = final_energy;
                // Also reset prev_positions/prev_forces so BFGS starts fresh
                if (prev_positions) |pp| {
                    alloc.free(pp);
                    prev_positions = null;
                }
                if (prev_forces) |pf| {
                    alloc.free(pf);
                    prev_forces = null;
                }
            } else {
                // Restore to saved_positions + scale * saved_displacement
                const scale = std.math.pow(f64, 0.5, @as(f64, @floatFromInt(backtrack_count)));
                for (0..n_atoms) |i| {
                    atoms[i].position = math.Vec3.add(
                        saved_positions.?[i],
                        math.Vec3.scale(saved_displacement.?[i], scale),
                    );
                }
                continue;
            }
        } else {
            // Step accepted
            backtrack_count = 0;
            prev_energy = final_energy;
        }

        // Compute forces
        // First, get electron density in reciprocal space
        const grid = forces_mod.Grid{
            .nx = scf_result.potential.nx,
            .ny = scf_result.potential.ny,
            .nz = scf_result.potential.nz,
            .min_h = scf_result.potential.min_h,
            .min_k = scf_result.potential.min_k,
            .min_l = scf_result.potential.min_l,
            .cell = current_cell,
            .recip = current_recip,
        };

        // Get rho_g from density (need to FFT)
        const rho_g = try densityToReciprocal(alloc, grid, scf_result.density, cfg.scf.fft_backend);
        defer alloc.free(rho_g);

        const coulomb_r_cut: ?f64 = if (cfg.boundary == .isolated) coulomb_mod.cutoffRadius(current_cell) else null;

        // For spin-polarized NLCC force, compute averaged V_xc
        var vxc_avg: ?[]f64 = null;
        defer if (vxc_avg) |v| alloc.free(v);
        const vxc_for_force: ?[]const f64 = if (scf_result.vxc_r_up != null and scf_result.vxc_r_down != null) blk: {
            const up = scf_result.vxc_r_up.?;
            const down = scf_result.vxc_r_down.?;
            vxc_avg = try alloc.alloc(f64, up.len);
            for (0..up.len) |i| {
                vxc_avg.?[i] = (up[i] + down[i]) * 0.5;
            }
            break :blk vxc_avg.?;
        } else scf_result.vxc_r;

        const force_start = std.Io.Clock.Timestamp.now(io, .awake);
        // Build PAW S_ij per-species array for nonlocal force
        const paw_dij_slice: ?[]const []const f64 = if (scf_result.paw_dij) |dij| blk: {
            const s = @as([]const []const f64, dij);
            break :blk s;
        } else null;
        const paw_rhoij_slice: ?[]const []const f64 = if (scf_result.paw_rhoij) |rij| blk: {
            const s = @as([]const []const f64, rij);
            break :blk s;
        } else null;
        const paw_tabs_slice: ?[]const paw_mod.PawTab = if (scf_result.paw_tabs) |tabs| tabs else null;
        var force_terms = try forces_mod.computeForces(
            alloc,
            grid,
            rho_g,
            scf_result.potential.values,
            scf_result.ionic_g,
            species,
            atoms,
            current_cell,
            current_recip,
            current_volume,
            alpha,
            scf_result.wavefunctions,
            if (scf_result.vresid) |vresid| vresid.values else null,
            cfg.scf.quiet,
            force_radial_tables,
            vxc_for_force,
            rho_atom_tables,
            rho_core_tables,
            ff_tables,
            coulomb_r_cut,
            cfg.vdw,
            paw_tabs_slice,
            paw_dij_slice,
            paw_rhoij_slice,
            scf_result.wavefunctions_down,
        );
        defer force_terms.deinit(alloc);
        const force_end = std.Io.Clock.Timestamp.now(io, .awake);

        // Remove average force (eliminate net translational force)
        // In periodic systems, forces should satisfy Newton's third law (sum = 0)
        // but numerical noise can introduce a non-zero average
        {
            var avg = math.Vec3{ .x = 0, .y = 0, .z = 0 };
            for (force_terms.total) |f| {
                avg.x += f.x;
                avg.y += f.y;
                avg.z += f.z;
            }
            const inv_n: f64 = 1.0 / @as(f64, @floatFromInt(n_atoms));
            avg.x *= inv_n;
            avg.y *= inv_n;
            avg.z *= inv_n;
            for (force_terms.total) |*f| {
                f.x -= avg.x;
                f.y -= avg.y;
                f.z -= avg.z;
            }
        }

        // Copy forces for storage (after average removal so output matches convergence check)
        final_forces = try alloc.alloc(math.Vec3, n_atoms);
        for (force_terms.total, 0..) |f, i| {
            final_forces[i] = f;
        }

        // Check convergence
        const max_force = forces_mod.maxForce(force_terms.total);
        try logRelaxIter(io, iter, final_energy, max_force);

        // Relax step timing profile (unbuffered write to avoid buffer corruption)
        {
            const step_cpu_end = output.Timing.getCpuTimeUs();
            const scf_ms = @as(f64, @floatFromInt(@as(u64, @intCast(scf_start.durationTo(scf_end).raw.nanoseconds)))) / 1_000_000.0;
            const force_ms = @as(f64, @floatFromInt(@as(u64, @intCast(force_start.durationTo(force_end).raw.nanoseconds)))) / 1_000_000.0;
            const step_ms = @as(f64, @floatFromInt(@as(u64, @intCast(relax_step_start.untilNow(io).raw.nanoseconds)))) / 1_000_000.0;
            const cpu_ms = @as(f64, @floatFromInt(step_cpu_end - relax_step_cpu_start)) / 1_000.0;
            var tbuf: [512]u8 = undefined;
            const msg = std.fmt.bufPrint(&tbuf, "relax_step_profile iter={d} scf_iters={d} scf_ms={d:.1} force_ms={d:.1} wall_ms={d:.1} cpu_ms={d:.1}\n", .{ iter, scf_result.iterations, scf_ms, force_ms, step_ms, cpu_ms }) catch "";
            std.Io.File.stderr().writeStreamingAll(io, msg) catch {};
        }

        // vc-relax: compute stress and check stress convergence
        var stress_converged = true; // default true for non-vc-relax
        var max_stress_gpa: f64 = 0.0;
        var cached_stress_total: ?stress_mod.Stress3x3 = null;
        if (cfg.relax.cell_relax) {
            const stress_terms = try stress_mod.computeStressFromScf(alloc, &scf_result, relax_cfg, species, atoms);
            // Symmetrize stress using original symmetry (even though k-points are not reduced)
            var sym_total = stress_terms.total;
            if (cfg.scf.symmetry) {
                const symmetry = @import("../symmetry/symmetry.zig");
                const sym_ops = try symmetry.getSymmetryOps(alloc, current_cell, atoms, 1e-5);
                defer alloc.free(sym_ops);
                if (sym_ops.len > 1) {
                    sym_total = stress_mod.symmetrizeStress(sym_total, sym_ops, current_cell);
                }
            }
            // Subtract target pressure from diagonal (Pulay stress compensation)
            const ry_bohr3_to_gpa = 14710.507;
            if (cfg.relax.target_pressure != 0.0) {
                const p_offset = -cfg.relax.target_pressure / ry_bohr3_to_gpa; // convert GPa to Ry/Bohr³
                for (0..3) |aa| sym_total[aa][aa] += p_offset;
            }
            cached_stress_total = sym_total;
            const sigma = sym_total;
            // Check convergence: all stress components → 0 (target_pressure already subtracted)
            for (0..3) |aa| {
                for (0..3) |bb| {
                    const dev = @abs(sigma[aa][bb]) * ry_bohr3_to_gpa;
                    if (dev > max_stress_gpa) max_stress_gpa = dev;
                }
            }
            stress_converged = max_stress_gpa < cfg.relax.stress_tol;

            const p = -(sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0;
            try logVcRelaxIter(io, iter, p * ry_bohr3_to_gpa, max_stress_gpa, current_volume);
        }

        if (max_force < cfg.relax.force_tol and stress_converged) {
            converged = true;
            // Transfer potential ownership for band calculation (avoids post-relax SCF)
            final_potential_saved = scf_result.potential;
            scf_result.potential = .{
                .nx = 0,
                .ny = 0,
                .nz = 0,
                .min_h = 0,
                .min_k = 0,
                .min_l = 0,
                .values = &[_]math.Complex{},
            };
            try logRelaxConverged(io, iter, final_energy, max_force);
            break;
        }

        // Save to trajectory if enabled
        if (trajectory) |*traj| {
            var step_atoms = try alloc.alloc(hamiltonian.AtomData, n_atoms);
            for (atoms, 0..) |a, i| {
                step_atoms[i] = a;
            }
            var step_forces = try alloc.alloc(math.Vec3, n_atoms);
            for (force_terms.total, 0..) |f, i| {
                step_forces[i] = f;
            }
            try traj.append(alloc, RelaxStep{
                .atoms = step_atoms,
                .energy = final_energy,
                .forces = step_forces,
                .max_force = max_force,
            });
        }

        // Extract current positions for optimizer
        var current_positions = try alloc.alloc(math.Vec3, n_atoms);
        defer alloc.free(current_positions);
        for (atoms, 0..) |a, i| {
            current_positions[i] = a.position;
        }

        // Update optimizer state BEFORE computing step (so step uses updated Hessian)
        if (prev_positions != null and prev_forces != null) {
            opt.update(prev_positions.?, current_positions, prev_forces.?, force_terms.total);
        }

        // Compute displacement using optimizer (now with updated Hessian)
        const displacement = try opt.step(alloc, force_terms.total, cfg.relax.max_step);
        defer alloc.free(displacement);

        // Store current state for next iteration
        if (prev_positions == null) {
            prev_positions = try alloc.alloc(math.Vec3, n_atoms);
        }
        if (prev_forces == null) {
            prev_forces = try alloc.alloc(math.Vec3, n_atoms);
        }
        for (0..n_atoms) |i| {
            prev_positions.?[i] = current_positions[i];
        }
        for (force_terms.total, 0..) |f, i| {
            prev_forces.?[i] = f;
        }

        // Save positions and displacement for potential backtracking
        if (saved_positions == null) {
            saved_positions = try alloc.alloc(math.Vec3, n_atoms);
        }
        if (saved_displacement == null) {
            saved_displacement = try alloc.alloc(math.Vec3, n_atoms);
        }
        for (0..n_atoms) |i| {
            saved_positions.?[i] = atoms[i].position;
            saved_displacement.?[i] = displacement[i];
        }

        // Update positions
        for (0..n_atoms) |i| {
            atoms[i].position = math.Vec3.add(atoms[i].position, displacement[i]);
        }

        // vc-relax: update cell based on stress tensor (skip if stress already converged)
        if (cfg.relax.cell_relax and !stress_converged) {
            try updateCellFromStress(&current_cell, &current_recip, &current_volume, cached_stress_total.?, atoms, cfg.relax.cell_step);
            relax_cfg.cell = current_cell;

            // Invalidate ALL caches that depend on cell (k-points, FFT grids, density)
            // Density grid size may change with cell, so discard it
            if (prev_density) |pd| {
                alloc.free(pd);
                prev_density = null;
            }
            if (prev_kpoint_cache) |old_cache| {
                for (old_cache) |*c| c.deinit();
                alloc.free(old_cache);
                prev_kpoint_cache = null;
            }
            if (prev_apply_caches) |caches| {
                for (caches) |*ac| ac.deinit(alloc);
                alloc.free(caches);
                prev_apply_caches = null;
            }
            // Recompute Ewald alpha for new cell
            if (cfg.ewald.alpha <= 0.0) {
                alpha = computeDefaultAlpha(current_cell);
            }
            // Reset BFGS Hessian since cell changed
            opt.reset();
            if (prev_positions) |pp| {
                alloc.free(pp);
                prev_positions = null;
            }
            if (prev_forces) |pf| {
                alloc.free(pf);
                prev_forces = null;
            }
        }

        // Free forces allocated this iteration (will be reallocated next iteration)
        // But keep the last iteration's forces for output
        if (iter + 1 < cfg.relax.max_iter) {
            alloc.free(final_forces);
            final_forces = &[_]math.Vec3{};
        }
    }

    if (!converged) {
        try logRelaxNotConverged(io, iterations, final_energy, forces_mod.maxForce(final_forces));
    }

    // Build result
    var result_trajectory: ?[]RelaxStep = null;
    if (trajectory) |*traj| {
        result_trajectory = try traj.toOwnedSlice(alloc);
    }

    // Transfer cache ownership to result (prevent defer from freeing them)
    const out_density = prev_density;
    prev_density = null;
    const out_kpoint_cache = prev_kpoint_cache;
    prev_kpoint_cache = null;
    const out_apply_caches = prev_apply_caches;
    prev_apply_caches = null;
    const out_potential = final_potential_saved;
    final_potential_saved = null;

    return RelaxResult{
        .final_atoms = atoms,
        .final_energy = final_energy,
        .final_forces = final_forces,
        .iterations = iterations,
        .converged = converged,
        .trajectory = result_trajectory,
        .final_density = out_density,
        .final_kpoint_cache = out_kpoint_cache,
        .final_apply_caches = out_apply_caches,
        .final_potential = out_potential,
        .final_cell = if (cfg.relax.cell_relax) current_cell else null,
        .final_recip = if (cfg.relax.cell_relax) current_recip else null,
        .final_volume = if (cfg.relax.cell_relax) current_volume else null,
    };
}

/// Update cell parameters based on stress tensor.
/// Uses steepest descent on the strain: ε_αβ = -step × σ_αβ / |σ|
/// Then cell' = (I + ε) × cell, and atoms are updated in fractional coordinates.
fn updateCellFromStress(
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
fn computeDefaultAlpha(cell: math.Mat3) f64 {
    const lmin = @min(
        @min(math.Vec3.norm(cell.row(0)), math.Vec3.norm(cell.row(1))),
        math.Vec3.norm(cell.row(2)),
    );
    return 5.0 / lmin;
}

/// Convert FFT index to frequency.
fn indexToFreq(i: usize, n: usize) i32 {
    const half = (n - 1) / 2;
    return if (i <= half) @as(i32, @intCast(i)) else @as(i32, @intCast(i)) - @as(i32, @intCast(n));
}

/// Convert real-space density to reciprocal space using FFT.
/// The output is reordered to match the grid's min_h, min_k, min_l layout
/// for consistency with SCF calculations.
fn densityToReciprocal(
    alloc: std.mem.Allocator,
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
    var plan = try fft.Fft3dPlan.initWithBackend(alloc, io, nx, ny, nz, fft_backend);
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
                const fh = indexToFreq(x, nx);
                const fk = indexToFreq(y, ny);
                const fl = indexToFreq(z, nz);
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

fn logWarn(io: std.Io, msg: []const u8) !void {
    var buffer: [512]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    try out.print("WARNING: {s}\n", .{msg});
    try out.flush();
}

fn logRelaxIter(io: std.Io, iter: usize, energy: ?f64, max_force: ?f64) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    if (energy != null and max_force != null) {
        try out.print("relax iter={d} energy={d:.8} max_force={d:.6}\n", .{ iter + 1, energy.?, max_force.? });
    } else {
        try out.print("relax iter={d} starting...\n", .{iter + 1});
    }
    try out.flush();
}

fn logRelaxConverged(io: std.Io, iter: usize, energy: f64, max_force: f64) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    try out.print("relax CONVERGED after {d} iterations, energy={d:.8} Ry, max_force={d:.6} Ry/Bohr\n", .{ iter + 1, energy, max_force });
    try out.flush();
}

fn logBacktrack(io: std.Io, count: usize, new_energy: f64, prev_energy_val: f64) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    const scale = std.math.pow(f64, 0.5, @as(f64, @floatFromInt(count)));
    try out.print("relax BACKTRACK #{d}: E={d:.8} > E_prev={d:.8} (dE={d:.6}), scale={d:.4}\n", .{ count, new_energy, prev_energy_val, new_energy - prev_energy_val, scale });
    try out.flush();
}

fn logVcRelaxIter(io: std.Io, iter: usize, pressure_gpa: f64, max_stress_gpa: f64, volume: f64) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    try out.print("vc-relax iter={d} P={d:.2} GPa  max_stress={d:.4} GPa  vol={d:.4} Bohr³\n", .{ iter, pressure_gpa, max_stress_gpa, volume });
    try out.flush();
}

fn logRelaxNotConverged(io: std.Io, iter: usize, energy: f64, max_force: f64) !void {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.File.stderr().writer(io, &buffer);
    const out = &writer.interface;
    try out.print("relax NOT CONVERGED after {d} iterations, energy={d:.8} Ry, max_force={d:.6} Ry/Bohr\n", .{ iter, energy, max_force });
    try out.flush();
}

/// Write relaxation results to output files.
pub fn writeOutput(
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
            const max_force = forces_mod.maxForce(result.final_forces);
            const rms_force = forces_mod.rmsForce(result.final_forces);
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
            try out.print("{d},{s},{d:.10},{d:.10},{d:.10},{d:.10}\n", .{ i, symbol, f.x, f.y, f.z, mag });
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
pub fn writeTrajectoryXyz(
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
        try out.print("Lattice=\"{d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8}\" ", .{
            c.m[0][0], c.m[0][1], c.m[0][2],
            c.m[1][0], c.m[1][1], c.m[1][2],
            c.m[2][0], c.m[2][1], c.m[2][2],
        });
        try out.print("Properties=species:S:1:pos:R:3:forces:R:3 ", .{});
        try out.print("energy={d:.10} ", .{step.energy * 13.6057}); // eV
        try out.print("max_force={d:.10} ", .{step.max_force * 13.6057 / 0.529177}); // eV/Angstrom
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
            try out.print("{s} {d:.10} {d:.10} {d:.10} {d:.10} {d:.10} {d:.10}\n", .{ sym, x, y, z, fx, fy, fz });
        }
        try out.flush();
    }

    // Also write final structure
    {
        const n_atoms = result.final_atoms.len;
        try out.print("{d}\n", .{n_atoms});
        try out.print("Lattice=\"{d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8} {d:.8}\" ", .{
            c.m[0][0], c.m[0][1], c.m[0][2],
            c.m[1][0], c.m[1][1], c.m[1][2],
            c.m[2][0], c.m[2][1], c.m[2][2],
        });
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
            try out.print("{s} {d:.10} {d:.10} {d:.10} {d:.10} {d:.10} {d:.10}\n", .{ sym, x, y, z, fx, fy, fz });
        }
        try out.flush();
    }
}
