const std = @import("std");
const apply = @import("apply.zig");
const config = @import("../config/config.zig");
const energy_mod = @import("energy.zig");
const fft = @import("../fft/fft.zig");
const fft_grid = @import("fft_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoints_mod = @import("kpoint_parallel.zig");
const logging = @import("logging.zig");
const math = @import("../math/math.zig");
const mixing = @import("mixing.zig");
const paw_mod = @import("../paw/paw.zig");
const paw_scf = @import("paw_scf.zig");
const potential_mod = @import("potential.zig");
const xc_fields_mod = @import("xc_fields.zig");
const scf_mod = @import("scf.zig");
const symmetry = @import("../symmetry/symmetry.zig");

const Grid = scf_mod.Grid;
const KpointCache = kpoints_mod.KpointCache;
const KpointEigenData = kpoints_mod.KpointEigenData;
const ScfResult = scf_mod.ScfResult;
const WavefunctionData = scf_mod.WavefunctionData;
const EnergyTerms = energy_mod.EnergyTerms;

const computeKpointEigenData = kpoints_mod.computeKpointEigenData;
const findFermiLevelSpin = kpoints_mod.findFermiLevelSpin;
const accumulateKpointDensitySmearingSpin = kpoints_mod.accumulateKpointDensitySmearingSpin;
const buildFftIndexMap = fft_grid.buildFftIndexMap;
const mixDensity = mixing.mixDensity;
const mixDensityKerker = mixing.mixDensityKerker;
const logProgress = logging.logProgress;
const logIterStart = logging.logIterStart;

const KPoint = symmetry.KPoint;

/// Result from solveKpointsForSpin: eigendata and count.
const SpinEigenResult = struct {
    eigen_data: []KpointEigenData,
    filled: usize,
};

/// Solve eigenvalue problem for all k-points for a single spin channel.
fn solveKpointsForSpin(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    common: *scf_mod.ScfCommon,
    potential: hamiltonian.PotentialGrid,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    scf_iter: usize,
    shared_fft_plan: fft.Fft3dPlan,
) !SpinEigenResult {
    const grid = common.grid;
    const kpoints = common.kpoints;
    const species = common.species;
    const atoms = common.atoms;
    const recip = common.recip;
    const volume_bohr = common.volume_bohr;
    const radial_tables = common.radial_tables;

    // For spin-polarized SCF, each channel needs enough bands to accommodate
    // magnetic splitting: up channel may have more occupied than nelec/2.
    // Add 20% extra bands + 4 minimum buffer for partial occupations.
    const nocc_base = @as(usize, @intFromFloat(std.math.ceil(common.total_electrons / 2.0)));
    const nocc = nocc_base + @max(4, nocc_base / 5);
    const is_paw_spin = scf_mod.hasPaw(species);
    const has_qij = scf_mod.hasQij(species) and !is_paw_spin;
    const use_iterative_config = (cfg.scf.solver == .iterative or cfg.scf.solver == .cg or cfg.scf.solver == .auto) and !has_qij;
    const nonlocal_enabled = cfg.scf.enable_nonlocal and scf_mod.hasNonlocal(species);

    var local_r: ?[]f64 = null;
    if (use_iterative_config) {
        local_r = try potential_mod.buildLocalPotentialReal(alloc, grid, common.ionic, potential);
    }
    defer if (local_r) |values| alloc.free(values);

    const fft_index_map = try buildFftIndexMap(alloc, grid);
    defer alloc.free(fft_index_map);

    var iter_max_iter = cfg.scf.iterative_max_iter;
    var iter_tol = cfg.scf.iterative_tol;
    if (cfg.scf.iterative_warmup_steps > 0 and scf_iter < cfg.scf.iterative_warmup_steps) {
        iter_max_iter = cfg.scf.iterative_warmup_max_iter;
        iter_tol = cfg.scf.iterative_warmup_tol;
    }

    const eigen_data = try alloc.alloc(KpointEigenData, kpoints.len);
    var filled: usize = 0;
    errdefer {
        var ii: usize = 0;
        while (ii < filled) : (ii += 1) {
            eigen_data[ii].deinit(alloc);
        }
        alloc.free(eigen_data);
    }

    for (kpoints, 0..) |kp, kidx| {
        const ac_ptr: ?*apply.KpointApplyCache = &apply_caches[kidx];
        eigen_data[kidx] = try computeKpointEigenData(
            alloc,
            &cfg,
            grid,
            kp,
            species,
            atoms,
            recip,
            volume_bohr,
            potential,
            local_r,
            nocc,
            use_iterative_config,
            has_qij,
            nonlocal_enabled,
            fft_index_map,
            iter_max_iter,
            iter_tol,
            cfg.scf.iterative_reuse_vectors,
            &kpoint_cache[kidx],
            null,
            shared_fft_plan,
            ac_ptr,
            radial_tables,
            common.paw_tabs,
        );
        filled += 1;
    }

    return SpinEigenResult{
        .eigen_data = eigen_data,
        .filled = filled,
    };
}

// =========================================================================
// Spin-polarized SCF loop (nspin=2)
// =========================================================================

pub fn runSpinPolarizedLoop(
    alloc: std.mem.Allocator,
    cfg: config.Config,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
    volume_bohr: f64,
    common: *scf_mod.ScfCommon,
) !ScfResult {
    const grid = common.grid;
    const kpoints = common.kpoints;
    const total_electrons = common.total_electrons;
    const recip = common.recip;

    // Kpoint caches: separate for up and down
    const kpoint_cache_up = try alloc.alloc(KpointCache, kpoints.len);
    defer {
        for (kpoint_cache_up) |*cache| cache.deinit();
        alloc.free(kpoint_cache_up);
    }
    for (kpoint_cache_up) |*cache| cache.* = .{};

    const kpoint_cache_down = try alloc.alloc(KpointCache, kpoints.len);
    defer {
        for (kpoint_cache_down) |*cache| cache.deinit();
        alloc.free(kpoint_cache_down);
    }
    for (kpoint_cache_down) |*cache| cache.* = .{};

    const grid_count = grid.count();

    // Initial spin densities
    const rho_up = try alloc.alloc(f64, grid_count);
    errdefer alloc.free(rho_up);
    const rho_down = try alloc.alloc(f64, grid_count);
    errdefer alloc.free(rho_down);

    // Compute initial magnetization from spinat
    var m_total: f64 = 0.0;
    if (cfg.scf.spinat) |sa| {
        for (sa) |m| {
            m_total += m;
        }
    }
    // Clamp magnetization to be physical
    m_total = std.math.clamp(m_total, -total_electrons, total_electrons);

    // Build initial density from superposition of atomic densities, split by magnetization
    {
        const atomic_rho = try scf_mod.buildAtomicDensity(alloc, grid, common.species, atoms);
        defer alloc.free(atomic_rho);
        var sum: f64 = 0.0;
        const dv = grid.volume / @as(f64, @floatFromInt(grid_count));
        for (atomic_rho) |v| sum += v * dv;
        const scale = if (sum > 1e-10) total_electrons / sum else 1.0;
        const frac_up = (total_electrons + m_total) / (2.0 * total_electrons);
        const frac_down = (total_electrons - m_total) / (2.0 * total_electrons);
        for (0..grid_count) |gi| {
            const rho_scaled = atomic_rho[gi] * scale;
            rho_up[gi] = rho_scaled * frac_up;
            rho_down[gi] = rho_scaled * frac_down;
        }
    }

    if (!cfg.scf.quiet) {
        var buffer: [256]u8 = undefined;
        var writer = std.Io.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print("spin-scf: nspin=2, nelec={d:.1}, m_init={d:.2}\n", .{ total_electrons, m_total });
        try out.flush();
    }

    // Build initial potentials
    const spin_potentials = try potential_mod.buildPotentialGridSpin(alloc, grid, rho_up, rho_down, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, null, common.coulomb_r_cut);
    var potential_up = spin_potentials.up;
    errdefer potential_up.deinit(alloc);
    var potential_down = spin_potentials.down;
    errdefer potential_down.deinit(alloc);

    var iterations: usize = 0;
    var converged = false;
    var last_band_energy: f64 = 0.0;
    var last_nonlocal_energy: f64 = 0.0;
    var last_entropy_energy: f64 = 0.0;
    var last_fermi_level: f64 = std.math.nan(f64);
    var last_potential_residual: f64 = 0.0;

    const nelec = total_electrons;

    // Apply caches per spin channel
    const apply_caches_up = try alloc.alloc(apply.KpointApplyCache, kpoints.len);
    defer {
        for (apply_caches_up) |*ac| ac.deinit(alloc);
        alloc.free(apply_caches_up);
    }
    for (apply_caches_up) |*ac| ac.* = .{};

    const apply_caches_down = try alloc.alloc(apply.KpointApplyCache, kpoints.len);
    defer {
        for (apply_caches_down) |*ac| ac.deinit(alloc);
        alloc.free(apply_caches_down);
    }
    for (apply_caches_down) |*ac| ac.* = .{};

    // PAW ecutrho computation
    const gs_scf = if (cfg.scf.grid_scale > 0.0) cfg.scf.grid_scale else 1.0;
    const ecutrho_scf = cfg.scf.ecut_ry * gs_scf * gs_scf;

    // PAW spin-resolved rhoij: separate up/down accumulation
    var paw_rhoij_up: ?paw_mod.RhoIJ = if (common.paw_rhoij) |*prij|
        try prij.clone(alloc)
    else
        null;
    defer if (paw_rhoij_up) |*rij| rij.deinit(alloc);
    var paw_rhoij_down: ?paw_mod.RhoIJ = if (common.paw_rhoij) |*prij|
        try prij.clone(alloc)
    else
        null;
    defer if (paw_rhoij_down) |*rij| rij.deinit(alloc);

    while (iterations < cfg.scf.max_iter) : (iterations += 1) {
        if (!cfg.scf.quiet) {
            try logIterStart(params.io, iterations);
        }

        // PAW: reset rhoij before accumulation
        if (common.paw_rhoij) |*prij| {
            prij.reset();
        }
        if (paw_rhoij_up) |*rij| rij.reset();
        if (paw_rhoij_down) |*rij| rij.reset();

        var band_energy_total: f64 = 0.0;
        var nonlocal_energy_total: f64 = 0.0;
        var entropy_energy_total: f64 = 0.0;

        // Arrays for spin-channel eigendata
        const rho_out_up = try alloc.alloc(f64, grid_count);
        defer alloc.free(rho_out_up);
        @memset(rho_out_up, 0.0);
        const rho_out_down = try alloc.alloc(f64, grid_count);
        defer alloc.free(rho_out_down);
        @memset(rho_out_down, 0.0);

        // Solve spin-up and spin-down channels
        var shared_fft_plan = try fft.Fft3dPlan.initWithBackend(alloc, grid.nx, grid.ny, grid.nz, cfg.scf.fft_backend);
        defer shared_fft_plan.deinit(alloc);

        var result_up = try solveKpointsForSpin(alloc, cfg, common, potential_up, kpoint_cache_up, apply_caches_up, iterations, shared_fft_plan);
        var eigen_data_up = result_up.eigen_data;
        defer {
            for (eigen_data_up[0..result_up.filled]) |*entry| entry.deinit(alloc);
            alloc.free(eigen_data_up);
        }

        var result_down = try solveKpointsForSpin(alloc, cfg, common, potential_down, kpoint_cache_down, apply_caches_down, iterations, shared_fft_plan);
        var eigen_data_down = result_down.eigen_data;
        defer {
            for (eigen_data_down[0..result_down.filled]) |*entry| entry.deinit(alloc);
            alloc.free(eigen_data_down);
        }

        // PAW D_ij bootstrap: update D_ij from spin-averaged potential and re-solve.
        if (iterations == 0 and common.is_paw) {
            if (common.paw_tabs) |tabs| {
                // Bootstrap D_ij with spin-specific potentials
                try paw_scf.updatePawDij(alloc, grid, common.ionic, potential_up, tabs, species, atoms, apply_caches_up, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, true, null, 1.0);
                // Bootstrap down channel with potential_down (after first band solve creates ctx)
                try paw_scf.updatePawDij(alloc, grid, common.ionic, potential_down, tabs, species, atoms, apply_caches_down, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, true, null, 1.0);
                // Re-solve with bootstrapped D_ij
                for (kpoint_cache_up) |*cache| cache.deinit();
                for (kpoint_cache_up) |*cache| cache.* = .{};
                for (kpoint_cache_down) |*cache| cache.deinit();
                for (kpoint_cache_down) |*cache| cache.* = .{};
                for (eigen_data_up[0..result_up.filled]) |*entry| entry.deinit(alloc);
                alloc.free(eigen_data_up);
                const ru = try solveKpointsForSpin(alloc, cfg, common, potential_up, kpoint_cache_up, apply_caches_up, iterations, shared_fft_plan);
                result_up = ru;
                eigen_data_up = result_up.eigen_data;
                for (eigen_data_down[0..result_down.filled]) |*entry| entry.deinit(alloc);
                alloc.free(eigen_data_down);
                const rd = try solveKpointsForSpin(alloc, cfg, common, potential_down, kpoint_cache_down, apply_caches_down, iterations, shared_fft_plan);
                result_down = rd;
                eigen_data_down = result_down.eigen_data;
            }
        }

        // Find Fermi level(s). For PAW spin with initial magnetization, use FSM
        // with self-consistent m_total update from output occupation.
        const use_fsm = common.is_paw and @abs(m_total) > 0.1;
        const ne_up_target = (nelec + m_total) / 2.0;
        const ne_down_target = (nelec - m_total) / 2.0;
        const mu_up = if (use_fsm)
            findFermiLevelSpin(ne_up_target, cfg.scf.smear_ry, cfg.scf.smearing, eigen_data_up[0..result_up.filled], null, 1.0)
        else
            findFermiLevelSpin(nelec, cfg.scf.smear_ry, cfg.scf.smearing, eigen_data_up[0..result_up.filled], eigen_data_down[0..result_down.filled], 1.0);
        const mu_down = if (use_fsm)
            findFermiLevelSpin(ne_down_target, cfg.scf.smear_ry, cfg.scf.smearing, eigen_data_down[0..result_down.filled], null, 1.0)
        else
            mu_up;
        const mu = mu_up; // Use up channel Fermi level as reference
        last_fermi_level = mu;

        // Build fft_index_map for density accumulation
        const fft_index_map = try buildFftIndexMap(alloc, grid);
        defer alloc.free(fft_index_map);

        // Accumulate densities for each spin channel (FSM uses separate mu)
        for (eigen_data_up[0..result_up.filled], 0..) |entry, kidx| {
            try accumulateKpointDensitySmearingSpin(
                alloc,
                &cfg,
                grid,
                kpoints[kidx],
                entry,
                recip,
                volume_bohr,
                fft_index_map,
                mu_up,
                cfg.scf.smear_ry,
                rho_out_up,
                &band_energy_total,
                &nonlocal_energy_total,
                &entropy_energy_total,
                null,
                1.0,
                if (kidx < apply_caches_up.len) &apply_caches_up[kidx] else null,
                if (paw_rhoij_up) |*rij| rij else null,
                atoms,
            );
        }
        for (eigen_data_down[0..result_down.filled], 0..) |entry, kidx| {
            try accumulateKpointDensitySmearingSpin(
                alloc,
                &cfg,
                grid,
                kpoints[kidx],
                entry,
                recip,
                volume_bohr,
                fft_index_map,
                mu_down,
                cfg.scf.smear_ry,
                rho_out_down,
                &band_energy_total,
                &nonlocal_energy_total,
                &entropy_energy_total,
                null,
                1.0,
                if (kidx < apply_caches_down.len) &apply_caches_down[kidx] else null,
                if (paw_rhoij_down) |*rij| rij else null,
                atoms,
            );
        }

        // Combine spin-resolved rhoij into total: rhoij = rhoij_up + rhoij_down
        if (common.paw_rhoij) |*prij| {
            if (paw_rhoij_up) |*rij_up| {
                if (paw_rhoij_down) |*rij_down| {
                    for (0..prij.natom) |a| {
                        for (0..prij.values[a].len) |idx| {
                            prij.values[a][idx] = rij_up.values[a][idx] + rij_down.values[a][idx];
                        }
                    }
                }
            }
        }

        last_band_energy = band_energy_total;
        last_nonlocal_energy = nonlocal_energy_total;
        last_entropy_energy = entropy_energy_total;

        // Symmetrize spin densities
        if (common.sym_ops) |ops| {
            if (ops.len > 1) {
                try scf_mod.symmetrizeDensity(alloc, grid, rho_out_up, ops, cfg.scf.use_rfft);
                try scf_mod.symmetrizeDensity(alloc, grid, rho_out_down, ops, cfg.scf.use_rfft);
            }
        }

        // PAW: symmetrize rhoij between equivalent atoms
        if (common.paw_rhoij) |*prij| {
            if (cfg.scf.symmetry) {
                try paw_scf.symmetrizeRhoIJ(alloc, prij, species, atoms);
            }
        }

        // PAW: filter density to ecutrho sphere and build augmented density
        if (common.is_paw) {
            const filt_up = try potential_mod.filterDensityToEcutrho(alloc, grid, rho_out_up, ecutrho_scf, cfg.scf.use_rfft);
            defer alloc.free(filt_up);
            @memcpy(rho_out_up, filt_up);
            const filt_down = try potential_mod.filterDensityToEcutrho(alloc, grid, rho_out_down, ecutrho_scf, cfg.scf.use_rfft);
            defer alloc.free(filt_down);
            @memcpy(rho_out_down, filt_down);
        }

        // Build augmented density for potential construction (ρ̃ + n̂/2 per spin)
        var rho_aug_up: ?[]f64 = null;
        var rho_aug_down: ?[]f64 = null;
        defer if (rho_aug_up) |a| alloc.free(a);
        defer if (rho_aug_down) |a| alloc.free(a);
        if (common.is_paw) {
            if (common.paw_rhoij) |*prij| {
                // Compute n̂ from total rhoij into a temporary zero array
                const n_hat = try alloc.alloc(f64, grid_count);
                defer alloc.free(n_hat);
                @memset(n_hat, 0.0);
                try paw_scf.addPawCompensationCharge(alloc, grid, n_hat, prij, common.paw_tabs.?, atoms, ecutrho_scf, &common.paw_gaunt.?);
                // Split n̂ equally between up and down
                const aug_up = try alloc.alloc(f64, grid_count);
                const aug_down = try alloc.alloc(f64, grid_count);
                for (0..grid_count) |i| {
                    aug_up[i] = rho_out_up[i] + n_hat[i] * 0.5;
                    aug_down[i] = rho_out_down[i] + n_hat[i] * 0.5;
                }
                rho_aug_up = aug_up;
                rho_aug_down = aug_down;
            }
        }

        // Build new spin potentials (use augmented density for PAW)
        const pot_rho_up = rho_aug_up orelse rho_out_up;
        const pot_rho_down = rho_aug_down orelse rho_out_down;
        const new_potentials = try potential_mod.buildPotentialGridSpin(alloc, grid, pot_rho_up, pot_rho_down, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, null, common.coulomb_r_cut);
        var pot_out_up = new_potentials.up;
        var pot_out_down = new_potentials.down;
        var keep_pot_out = false;
        defer {
            if (!keep_pot_out) {
                pot_out_up.deinit(alloc);
                pot_out_down.deinit(alloc);
            }
        }

        // Compute residual from concatenated potentials
        {
            const nvals = potential_up.values.len;
            var sum_sq: f64 = 0.0;
            for (0..nvals) |idx| {
                const diff_up = math.complex.sub(pot_out_up.values[idx], potential_up.values[idx]);
                const diff_down = math.complex.sub(pot_out_down.values[idx], potential_down.values[idx]);
                sum_sq += diff_up.r * diff_up.r + diff_up.i * diff_up.i;
                sum_sq += diff_down.r * diff_down.r + diff_down.i * diff_down.i;
            }
            last_potential_residual = if (nvals > 0)
                std.math.sqrt(sum_sq / @as(f64, @floatFromInt(2 * nvals)))
            else
                0.0;
        }

        // Density diff for convergence check
        const rho_out_total = try alloc.alloc(f64, grid_count);
        defer alloc.free(rho_out_total);
        const rho_in_total = try alloc.alloc(f64, grid_count);
        defer alloc.free(rho_in_total);
        for (0..grid_count) |i| {
            rho_out_total[i] = rho_out_up[i] + rho_out_down[i];
            rho_in_total[i] = rho_up[i] + rho_down[i];
        }
        const diff = scf_mod.densityDiff(rho_in_total, rho_out_total);

        const conv_value = switch (cfg.scf.convergence_metric) {
            .density => diff,
            .potential => last_potential_residual,
        };

        try common.log.writeIter(iterations, diff, last_potential_residual, last_band_energy, last_nonlocal_energy);
        if (!cfg.scf.quiet) {
            try logProgress(params.io, iterations, diff, last_potential_residual, last_band_energy, last_nonlocal_energy);
        }

        if (conv_value < cfg.scf.convergence) {
            converged = true;
            @memcpy(rho_up, rho_out_up);
            @memcpy(rho_down, rho_out_down);
            potential_up.deinit(alloc);
            potential_up = pot_out_up;
            potential_down.deinit(alloc);
            potential_down = pot_out_down;
            keep_pot_out = true;
            break;
        }

        // For PAW spin, use density mixing (potential mixing kills magnetization
        // because V_xc splitting decays through mixing). Kerker preconditioning
        // stabilizes density mixing.
        const force_density_mixing = false; // density mixing unstable for PAW Fe
        if (cfg.scf.mixing_mode == .potential and !force_density_mixing) {
            const n_complex = potential_up.values.len;
            const n_f64 = n_complex * 2;
            // Mix up
            const v_in_up: []f64 = @as([*]f64, @ptrCast(potential_up.values.ptr))[0..n_f64];
            const v_out_up_f: []const f64 = @as([*]const f64, @ptrCast(pot_out_up.values.ptr))[0..n_f64];
            // Mix down
            const v_in_down: []f64 = @as([*]f64, @ptrCast(potential_down.values.ptr))[0..n_f64];
            const v_out_down_f: []const f64 = @as([*]const f64, @ptrCast(pot_out_down.values.ptr))[0..n_f64];

            // Concatenate for Pulay
            if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
                // Compute complex residual for up and down, concatenated
                const residual_concat = try alloc.alloc(f64, n_f64 * 2);
                // Ownership transfers to mixer via mixWithResidual — no defer free
                for (0..n_f64) |i| {
                    residual_concat[i] = @as([*]const f64, @ptrCast(pot_out_up.values.ptr))[i] - v_in_up[i];
                }
                for (0..n_f64) |i| {
                    residual_concat[n_f64 + i] = @as([*]const f64, @ptrCast(pot_out_down.values.ptr))[i] - v_in_down[i];
                }

                // Apply model dielectric preconditioner if enabled
                if (cfg.scf.diemac > 1.0) {
                    // Reinterpret as Complex slices and precondition each spin channel
                    const res_up_c: []math.Complex = @as([*]math.Complex, @ptrCast(@alignCast(residual_concat.ptr)))[0..n_complex];
                    const res_down_c: []math.Complex = @as([*]math.Complex, @ptrCast(@alignCast(residual_concat[n_f64..].ptr)))[0..n_complex];
                    mixing.applyModelDielectricPreconditioner(grid, res_up_c, cfg.scf.diemac, cfg.scf.dielng);
                    mixing.applyModelDielectricPreconditioner(grid, res_down_c, cfg.scf.diemac, cfg.scf.dielng);
                }

                // Concatenate v_in for Pulay
                const concat_in = try alloc.alloc(f64, n_f64 * 2);
                defer alloc.free(concat_in);
                @memcpy(concat_in[0..n_f64], v_in_up);
                @memcpy(concat_in[n_f64..], v_in_down);

                try common.pulay_mixer.?.mixWithResidual(concat_in, residual_concat, cfg.scf.mixing_beta);
                @memcpy(v_in_up, concat_in[0..n_f64]);
                @memcpy(v_in_down, concat_in[n_f64..]);
            } else {
                mixDensity(v_in_up, v_out_up_f, cfg.scf.mixing_beta);
                mixDensity(v_in_down, v_out_down_f, cfg.scf.mixing_beta);
            }

            @memcpy(rho_up, rho_out_up);
            @memcpy(rho_down, rho_out_down);
            pot_out_up.deinit(alloc);
            pot_out_down.deinit(alloc);
            keep_pot_out = true;
        } else {
            // Density mixing with Kerker preconditioning for PAW stability.
            // Kerker suppresses long-wavelength charge sloshing that causes
            // D_ij transients and kills magnetic order.
            if (force_density_mixing) {
                // PAW spin: Kerker density mixing with small beta for stability
                const paw_beta: f64 = 0.05;
                const paw_q0: f64 = 1.5;
                try mixDensityKerker(alloc, grid, rho_up, rho_out_up, paw_beta, paw_q0, cfg.scf.use_rfft);
                try mixDensityKerker(alloc, grid, rho_down, rho_out_down, paw_beta, paw_q0, cfg.scf.use_rfft);
            } else if (common.pulay_mixer != null and iterations >= cfg.scf.pulay_start) {
                const concat_in = try alloc.alloc(f64, grid_count * 2);
                defer alloc.free(concat_in);
                const concat_out = try alloc.alloc(f64, grid_count * 2);
                defer alloc.free(concat_out);
                @memcpy(concat_in[0..grid_count], rho_up);
                @memcpy(concat_in[grid_count..], rho_down);
                @memcpy(concat_out[0..grid_count], rho_out_up);
                @memcpy(concat_out[grid_count..], rho_out_down);
                try common.pulay_mixer.?.mix(concat_in, concat_out, cfg.scf.mixing_beta);
                @memcpy(rho_up, concat_in[0..grid_count]);
                @memcpy(rho_down, concat_in[grid_count..]);
            } else {
                mixDensity(rho_up, rho_out_up, cfg.scf.mixing_beta);
                mixDensity(rho_down, rho_out_down, cfg.scf.mixing_beta);
            }

            potential_up.deinit(alloc);
            potential_down.deinit(alloc);
            // PAW: rebuild potential from augmented density
            if (common.is_paw and common.paw_rhoij != null) {
                const n_hat_dm = try alloc.alloc(f64, grid_count);
                defer alloc.free(n_hat_dm);
                @memset(n_hat_dm, 0.0);
                try paw_scf.addPawCompensationCharge(alloc, grid, n_hat_dm, &common.paw_rhoij.?, common.paw_tabs.?, atoms, ecutrho_scf, &common.paw_gaunt.?);
                const dm_aug_up = try alloc.alloc(f64, grid_count);
                defer alloc.free(dm_aug_up);
                const dm_aug_down = try alloc.alloc(f64, grid_count);
                defer alloc.free(dm_aug_down);
                for (0..grid_count) |gi| {
                    dm_aug_up[gi] = rho_up[gi] + n_hat_dm[gi] * 0.5;
                    dm_aug_down[gi] = rho_down[gi] + n_hat_dm[gi] * 0.5;
                }
                const rebuilt = try potential_mod.buildPotentialGridSpin(alloc, grid, dm_aug_up, dm_aug_down, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, null, common.coulomb_r_cut);
                potential_up = rebuilt.up;
                potential_down = rebuilt.down;
            } else {
                const rebuilt = try potential_mod.buildPotentialGridSpin(alloc, grid, rho_up, rho_down, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc, null, null, common.coulomb_r_cut);
                potential_up = rebuilt.up;
                potential_down = rebuilt.down;
            }
        }

        // PAW: update D_ij with spin-resolved D^xc and D_ij mixing.
        // D_ij mixing (β = mixing_beta) prevents abrupt changes that kill Stoner feedback.
        if (common.is_paw) {
            if (common.paw_tabs) |tabs| {
                // D_ij mixing to smooth transients that kill Stoner feedback
                const dij_mix_beta: f64 = if (common.is_paw) cfg.scf.mixing_beta else 1.0;

                // Compute new D_ij with spin D^xc
                if (paw_rhoij_up) |*rij_up| {
                    try paw_scf.updatePawDij(alloc, grid, common.ionic, potential_up, tabs, species, atoms, apply_caches_up, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, rij_up, dij_mix_beta);
                } else {
                    try paw_scf.updatePawDij(alloc, grid, common.ionic, potential_up, tabs, species, atoms, apply_caches_up, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, null, dij_mix_beta);
                }
                if (paw_rhoij_down) |*rij_down| {
                    try paw_scf.updatePawDij(alloc, grid, common.ionic, potential_down, tabs, species, atoms, apply_caches_down, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, rij_down, dij_mix_beta);
                } else {
                    try paw_scf.updatePawDij(alloc, grid, common.ionic, potential_down, tabs, species, atoms, apply_caches_down, ecutrho_scf, &common.paw_rhoij.?, cfg.scf.xc, cfg.scf.symmetry, &common.paw_gaunt.?, false, null, dij_mix_beta);
                }
            }
        }
    }

    // Compute magnetization
    const dv = grid.volume / @as(f64, @floatFromInt(grid_count));
    var magnetization: f64 = 0.0;
    for (0..grid_count) |i| {
        magnetization += (rho_up[i] - rho_down[i]) * dv;
    }

    if (!cfg.scf.quiet) {
        var buffer: [256]u8 = undefined;
        var writer = std.Io.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print("spin-scf: magnetization = {d:.6} μ_B\n", .{magnetization});
        try out.flush();
    }

    // Compute total density for energy calculation
    const rho_total = try alloc.alloc(f64, grid_count);
    errdefer alloc.free(rho_total);
    for (0..grid_count) |i| {
        rho_total[i] = rho_up[i] + rho_down[i];
    }

    // PAW: build augmented density for energy computation (reuse ecutrho_scf)
    var rho_aug_up_for_energy: ?[]f64 = null;
    var rho_aug_down_for_energy: ?[]f64 = null;
    if (common.is_paw) {
        if (common.paw_rhoij) |*prij| {
            // Build n̂ from total rhoij
            const n_hat = try alloc.alloc(f64, grid_count);
            defer alloc.free(n_hat);
            @memset(n_hat, 0.0);
            try paw_scf.addPawCompensationCharge(alloc, grid, n_hat, prij, common.paw_tabs.?, atoms, ecutrho_scf, &common.paw_gaunt.?);
            // Subtract back rho=0 contribution to get pure n̂
            // (addPawCompensationCharge adds n̂ to rho, but we passed zeros)
            // rho_aug = rho + n̂/2 for each spin (n̂ split equally for non-magnetic)
            const aug_up = try alloc.alloc(f64, grid_count);
            const aug_down = try alloc.alloc(f64, grid_count);
            for (0..grid_count) |i| {
                aug_up[i] = rho_up[i] + n_hat[i] * 0.5;
                aug_down[i] = rho_down[i] + n_hat[i] * 0.5;
            }
            // Filter to ecutrho sphere
            const filt_up = try potential_mod.filterDensityToEcutrho(alloc, grid, aug_up, ecutrho_scf, cfg.scf.use_rfft);
            alloc.free(aug_up);
            const filt_down = try potential_mod.filterDensityToEcutrho(alloc, grid, aug_down, ecutrho_scf, cfg.scf.use_rfft);
            alloc.free(aug_down);
            rho_aug_up_for_energy = filt_up;
            rho_aug_down_for_energy = filt_down;
        }
    }
    defer if (rho_aug_up_for_energy) |a| alloc.free(a);
    defer if (rho_aug_down_for_energy) |a| alloc.free(a);

    const paw_ecutrho: ?f64 = if (common.is_paw) ecutrho_scf else null;

    // Compute energy terms using spin-polarized XC
    var energy_terms = try energy_mod.computeEnergyTermsSpin(
        alloc,
        grid,
        rho_up,
        rho_down,
        common.rho_core,
        last_band_energy,
        last_nonlocal_energy,
        last_entropy_energy,
        species,
        atoms,
        cfg.ewald,
        cfg.scf.use_rfft,
        cfg.scf.xc,
        cfg.scf.quiet,
        common.coulomb_r_cut,
        cfg.vdw,
        rho_aug_up_for_energy,
        rho_aug_down_for_energy,
        paw_ecutrho,
    );

    // Add PAW on-site energy correction (spin-resolved E_xc if rhoij_up/down available)
    if (common.is_paw) {
        if (common.paw_rhoij) |*prij| {
            if (common.paw_tabs) |tabs| {
                energy_terms.paw_onsite = try paw_scf.computePawOnsiteEnergyTotal(
                    alloc,
                    prij,
                    tabs,
                    species,
                    atoms,
                    cfg.scf.xc,
                    &common.paw_gaunt.?,
                    if (paw_rhoij_up) |*ru| ru else null,
                    if (paw_rhoij_down) |*rd| rd else null,
                );
                energy_terms.total += energy_terms.paw_onsite;
            }
        }
    }

    // PAW double-counting: -Σ (D^xc_up × ρ_up + D^xc_down × ρ_down + D^H × ρ_total)
    var result_paw_tabs: ?[]paw_mod.PawTab = null;
    var result_paw_dij: ?[][]f64 = null;
    var result_paw_dij_m: ?[][]f64 = null;
    var result_paw_dxc: ?[][]f64 = null;
    var result_paw_rhoij: ?[][]f64 = null;
    if (common.is_paw) {
        if (common.paw_rhoij) |prij| {
            if (common.paw_tabs) |tabs| {
                var dxc_list: std.ArrayList([]f64) = .empty;
                errdefer {
                    for (dxc_list.items) |d| alloc.free(d);
                    dxc_list.deinit(alloc);
                }
                var sum_dxc_rhoij: f64 = 0.0;
                for (atoms, 0..) |atom, ai| {
                    const si = atom.species_index;
                    const upf = species[si].upf;
                    const paw = upf.paw orelse {
                        const zero = try alloc.alloc(f64, 0);
                        try dxc_list.append(alloc, zero);
                        continue;
                    };
                    if (si >= tabs.len or tabs[si].nbeta == 0) {
                        const zero = try alloc.alloc(f64, 0);
                        try dxc_list.append(alloc, zero);
                        continue;
                    }
                    const mt = prij.m_total_per_atom[ai];
                    const sp_m_offsets = prij.m_offsets[ai];
                    const rhoij_m = prij.values[ai];

                    // D^xc double-counting: spin-resolved if available
                    if (paw_rhoij_up != null and paw_rhoij_down != null) {
                        const dxc_up = try alloc.alloc(f64, mt * mt);
                        defer alloc.free(dxc_up);
                        const dxc_down = try alloc.alloc(f64, mt * mt);
                        defer alloc.free(dxc_down);
                        try paw_mod.paw_xc.computePawDijXcAngularSpin(
                            alloc,
                            dxc_up,
                            dxc_down,
                            paw,
                            paw_rhoij_up.?.values[ai],
                            paw_rhoij_down.?.values[ai],
                            mt,
                            sp_m_offsets,
                            upf.r,
                            upf.rab,
                            paw.ae_core_density,
                            if (upf.nlcc.len > 0) upf.nlcc else null,
                            cfg.scf.xc,
                            &common.paw_gaunt.?,
                        );
                        // DC_xc = Σ (D^xc_up × ρ_up + D^xc_down × ρ_down)
                        const rij_up = paw_rhoij_up.?.values[ai];
                        const rij_dn = paw_rhoij_down.?.values[ai];
                        for (0..mt) |im| {
                            for (0..mt) |jm| {
                                sum_dxc_rhoij += dxc_up[im * mt + jm] * rij_up[im * mt + jm];
                                sum_dxc_rhoij += dxc_down[im * mt + jm] * rij_dn[im * mt + jm];
                            }
                        }
                        // Store D^xc_up for result (used in stress)
                        const dxc_m_copy = try alloc.alloc(f64, mt * mt);
                        @memcpy(dxc_m_copy, dxc_up);
                        try dxc_list.append(alloc, dxc_m_copy);
                    } else {
                        const dxc_m = try alloc.alloc(f64, mt * mt);
                        try paw_mod.paw_xc.computePawDijXcAngular(
                            alloc,
                            dxc_m,
                            paw,
                            rhoij_m,
                            mt,
                            sp_m_offsets,
                            upf.r,
                            upf.rab,
                            paw.ae_core_density,
                            if (upf.nlcc.len > 0) upf.nlcc else null,
                            cfg.scf.xc,
                            &common.paw_gaunt.?,
                        );
                        for (0..mt) |im| {
                            for (0..mt) |jm| {
                                sum_dxc_rhoij += dxc_m[im * mt + jm] * rhoij_m[im * mt + jm];
                            }
                        }
                        try dxc_list.append(alloc, dxc_m);
                    }

                    // D^H double-counting (uses total rhoij, spin-independent)
                    {
                        const dij_h_dc = try alloc.alloc(f64, mt * mt);
                        defer alloc.free(dij_h_dc);
                        try paw_mod.paw_xc.computePawDijHartreeMultiL(
                            alloc,
                            dij_h_dc,
                            paw,
                            rhoij_m,
                            mt,
                            sp_m_offsets,
                            upf.r,
                            upf.rab,
                            &common.paw_gaunt.?,
                        );
                        for (0..mt) |im| {
                            for (0..mt) |jm| {
                                sum_dxc_rhoij += dij_h_dc[im * mt + jm] * rhoij_m[im * mt + jm];
                            }
                        }
                    }
                }
                if (dxc_list.items.len > 0) {
                    result_paw_dxc = try dxc_list.toOwnedSlice(alloc);
                } else {
                    dxc_list.deinit(alloc);
                }
                energy_terms.paw_dxc_rhoij = -sum_dxc_rhoij;
                energy_terms.total += energy_terms.paw_dxc_rhoij;
            }
        }

        // Transfer paw_tabs ownership from common to result
        if (common.paw_tabs) |tabs| {
            result_paw_tabs = tabs;
            common.paw_tabs = null;
        }
        // Extract per-atom D_ij from first apply cache (up channel)
        if (apply_caches_up.len > 0) {
            if (apply_caches_up[0].nonlocal_ctx) |nc| {
                var dij_list: std.ArrayList([]f64) = .empty;
                errdefer {
                    for (dij_list.items) |d| alloc.free(d);
                    dij_list.deinit(alloc);
                }
                for (nc.species) |entry_sp| {
                    if (entry_sp.dij_per_atom) |dpa| {
                        for (dpa) |atom_dij| {
                            const copy = try alloc.alloc(f64, atom_dij.len);
                            @memcpy(copy, atom_dij);
                            try dij_list.append(alloc, copy);
                        }
                    }
                }
                if (dij_list.items.len > 0) {
                    result_paw_dij = try dij_list.toOwnedSlice(alloc);
                } else {
                    dij_list.deinit(alloc);
                }
                var dij_m_list: std.ArrayList([]f64) = .empty;
                errdefer {
                    for (dij_m_list.items) |d| alloc.free(d);
                    dij_m_list.deinit(alloc);
                }
                for (nc.species) |entry_sp| {
                    if (entry_sp.dij_m_per_atom) |dpa| {
                        for (dpa) |atom_dij_m| {
                            const copy = try alloc.alloc(f64, atom_dij_m.len);
                            @memcpy(copy, atom_dij_m);
                            try dij_m_list.append(alloc, copy);
                        }
                    }
                }
                if (dij_m_list.items.len > 0) {
                    result_paw_dij_m = try dij_m_list.toOwnedSlice(alloc);
                } else {
                    dij_m_list.deinit(alloc);
                }
            }
        }
        // Copy per-atom rhoij (contracted to radial basis)
        if (common.paw_rhoij) |*prij| {
            var rij_list: std.ArrayList([]f64) = .empty;
            errdefer {
                for (rij_list.items) |r| alloc.free(r);
                rij_list.deinit(alloc);
            }
            for (0..prij.natom) |a| {
                const nb = prij.nbeta_per_atom[a];
                const copy = try alloc.alloc(f64, nb * nb);
                prij.contractToRadial(a, copy);
                try rij_list.append(alloc, copy);
            }
            if (rij_list.items.len > 0) {
                result_paw_rhoij = try rij_list.toOwnedSlice(alloc);
            } else {
                rij_list.deinit(alloc);
            }
        }
    }

    if (!cfg.scf.quiet) {
        var buffer: [256]u8 = undefined;
        var writer = std.Io.File.stderr().writer(&buffer);
        const out = &writer.interface;
        try out.print("spin-scf: total_energy = {d:.10} Ry\n", .{energy_terms.total});
        try out.print("spin-scf: E_band={d:.8} E_H={d:.8} E_xc={d:.8} E_ion={d:.8}\n", .{ energy_terms.band, energy_terms.hartree, energy_terms.xc, energy_terms.ion_ion });
        try out.print("spin-scf: E_psp={d:.8} E_dc={d:.8} E_local={d:.8} E_nl={d:.8}\n", .{ energy_terms.psp_core, energy_terms.double_counting, energy_terms.local_pseudo, energy_terms.nonlocal_pseudo });
        if (common.is_paw) {
            try out.print("spin-scf: E_paw_onsite={d:.8} E_paw_dxc={d:.8}\n", .{ energy_terms.paw_onsite, energy_terms.paw_dxc_rhoij });
        }
        try out.flush();
    }

    try common.log.writeResult(
        converged,
        iterations,
        energy_terms.total,
        energy_terms.band,
        energy_terms.hartree,
        energy_terms.xc,
        energy_terms.ion_ion,
        energy_terms.psp_core,
    );

    // Compute final wavefunctions for force/stress/DOS/band calculation
    var wavefunctions_up: ?WavefunctionData = null;
    var wavefunctions_down_final: ?WavefunctionData = null;
    var vxc_r_up_result: ?[]f64 = null;
    var vxc_r_down_result: ?[]f64 = null;
    if (cfg.relax.enabled or cfg.dfpt.enabled or cfg.scf.compute_stress or cfg.dos.enabled) {
        const wfn_up = try scf_mod.computeFinalWavefunctionsWithSpinFactor(
            alloc, cfg, grid, kpoints, common.ionic, species, atoms, recip, volume_bohr,
            potential_up, kpoint_cache_up, apply_caches_up, common.radial_tables, common.paw_tabs, 1.0,
        );
        wavefunctions_up = wfn_up.wavefunctions;
        const wfn_down = try scf_mod.computeFinalWavefunctionsWithSpinFactor(
            alloc, cfg, grid, kpoints, common.ionic, species, atoms, recip, volume_bohr,
            potential_down, kpoint_cache_down, apply_caches_down, common.radial_tables, common.paw_tabs, 1.0,
        );
        wavefunctions_down_final = wfn_down.wavefunctions;

        // Update band energies from final wavefunctions (spin_factor=1.0 for each channel)
        last_band_energy = wfn_up.band_energy + wfn_down.band_energy;
        last_nonlocal_energy = wfn_up.nonlocal_energy + wfn_down.nonlocal_energy;

        // Store V_xc in real space for NLCC force
        const pot_rho_up_final = rho_aug_up_for_energy orelse rho_up;
        const pot_rho_down_final = rho_aug_down_for_energy orelse rho_down;
        const vxc_spin = try xc_fields_mod.computeXcFieldsSpin(alloc, grid, pot_rho_up_final, pot_rho_down_final, common.rho_core, cfg.scf.use_rfft, cfg.scf.xc);
        vxc_r_up_result = vxc_spin.vxc_up;
        vxc_r_down_result = vxc_spin.vxc_down;
        alloc.free(vxc_spin.exc);
    }
    errdefer if (wavefunctions_up) |*wf| wf.deinit(alloc);
    errdefer if (wavefunctions_down_final) |*wf| wf.deinit(alloc);
    errdefer if (vxc_r_up_result) |v| alloc.free(v);
    errdefer if (vxc_r_down_result) |v| alloc.free(v);

    return ScfResult{
        .potential = potential_up,
        .density = rho_total,
        .iterations = iterations,
        .converged = converged,
        .energy = energy_terms,
        .fermi_level = last_fermi_level,
        .potential_residual = last_potential_residual,
        .wavefunctions = wavefunctions_up,
        .vresid = null,
        .grid = grid,
        .density_up = rho_up,
        .density_down = rho_down,
        .potential_down = potential_down,
        .magnetization = magnetization,
        .wavefunctions_down = wavefunctions_down_final,
        .vxc_r_up = vxc_r_up_result,
        .vxc_r_down = vxc_r_down_result,
        .fermi_level_down = if (!std.math.isNan(last_fermi_level)) last_fermi_level else 0.0,
        .paw_tabs = result_paw_tabs,
        .paw_dij = result_paw_dij,
        .paw_dij_m = result_paw_dij_m,
        .paw_dxc = result_paw_dxc,
        .paw_rhoij = result_paw_rhoij,
        .rho_core = if (common.rho_core) |rc| blk: {
            const copy = try alloc.alloc(f64, rc.len);
            @memcpy(copy, rc);
            break :blk copy;
        } else null,
    };
}
