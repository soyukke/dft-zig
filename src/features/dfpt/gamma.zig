//! Γ-point DFPT phonon calculation.
//!
//! Contains the q=0 perturbation solver, dynmat construction,
//! diagnostics, and the top-level `run_phonon` entry point.

const std = @import("std");
const math = @import("../math/math.zig");
const xc = @import("../xc/xc.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../scf/scf.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const d3 = @import("../vdw/d3.zig");
const d3_params = @import("../vdw/d3_params.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const config_mod = @import("../config/config.zig");
const model_mod = @import("../dft/model.zig");

const symmetry_mod = @import("../symmetry/symmetry.zig");

const dfpt = @import("dfpt.zig");
const perturbation = dfpt.perturbation;
const sternheimer = dfpt.sternheimer;
const ewald2 = dfpt.ewald2;
const dynmat_mod = dfpt.dynmat;
const dynmat_contrib = dfpt.dynmat_contrib;

const GroundState = dfpt.GroundState;
const PreparedGroundState = dfpt.PreparedGroundState;
const DfptConfig = dfpt.DfptConfig;
const IonicData = dfpt.IonicData;
const PerturbationResult = dfpt.PerturbationResult;
const log_dfpt = dfpt.log_dfpt;
const log_dfpt_info = dfpt.log_dfpt_info;
const log_dfpt_warn = dfpt.log_dfpt_warn;

const Grid = scf_mod.Grid;

/// Result of a full phonon calculation.
pub const PhononResult = struct {
    /// Phonon frequencies in cm⁻¹
    frequencies_cm1: []f64,
    /// Eigenvalues ω² in Ry/(bohr²·amu)
    omega2: []f64,
    /// Eigenvectors (column-major, dim×dim)
    eigenvectors: []f64,
    /// Dimension (3 × n_atoms)
    dim: usize,

    pub fn deinit(self: *PhononResult, alloc: std.mem.Allocator) void {
        if (self.frequencies_cm1.len > 0) alloc.free(self.frequencies_cm1);
        if (self.omega2.len > 0) alloc.free(self.omega2);
        if (self.eigenvectors.len > 0) alloc.free(self.eigenvectors);
    }
};

const GammaPerturbationBuffers = struct {
    pert_results: []PerturbationResult,
    vloc1_gs: [][]math.Complex,
    rho1_core_gs: [][]math.Complex,

    fn init(alloc: std.mem.Allocator, dim: usize) !GammaPerturbationBuffers {
        const pert_results = try alloc.alloc(PerturbationResult, dim);
        errdefer alloc.free(pert_results);

        const vloc1_gs = try alloc.alloc([]math.Complex, dim);
        errdefer alloc.free(vloc1_gs);

        const rho1_core_gs = try alloc.alloc([]math.Complex, dim);
        errdefer alloc.free(rho1_core_gs);

        for (0..dim) |i| {
            pert_results[i] = .{ .rho1_g = &.{}, .psi1 = &.{} };
            vloc1_gs[i] = &.{};
            rho1_core_gs[i] = &.{};
        }
        return .{
            .pert_results = pert_results,
            .vloc1_gs = vloc1_gs,
            .rho1_core_gs = rho1_core_gs,
        };
    }

    fn deinit(self: *GammaPerturbationBuffers, alloc: std.mem.Allocator) void {
        for (self.pert_results) |*result| result.deinit(alloc);
        alloc.free(self.pert_results);
        for (self.vloc1_gs) |buf| {
            if (buf.len > 0) alloc.free(buf);
        }
        alloc.free(self.vloc1_gs);
        for (self.rho1_core_gs) |buf| {
            if (buf.len > 0) alloc.free(buf);
        }
        alloc.free(self.rho1_core_gs);
    }
};

const SolvePerturbationInputs = struct {
    vloc1_g: []math.Complex,
    rho1_core_g: []math.Complex,
    rho1_core_r: []f64,

    fn deinit(self: *SolvePerturbationInputs, alloc: std.mem.Allocator) void {
        alloc.free(self.rho1_core_r);
        alloc.free(self.rho1_core_g);
        alloc.free(self.vloc1_g);
    }
};

const NonlocalScfBuffers = struct {
    nl_out_buf: ?[]math.Complex = null,
    nl_phase_buf: ?[]math.Complex = null,
    nl_xphase_buf: ?[]math.Complex = null,
    nl_coeff_buf: ?[]math.Complex = null,
    nl_coeff2_buf: ?[]math.Complex = null,

    fn init(
        alloc: std.mem.Allocator,
        apply_ctx: *scf_mod.ApplyContext,
        n_pw: usize,
    ) !NonlocalScfBuffers {
        var buffers = NonlocalScfBuffers{};
        if (apply_ctx.nonlocal_ctx) |nl_ctx| {
            buffers.nl_out_buf = try alloc.alloc(math.Complex, n_pw);
            errdefer if (buffers.nl_out_buf) |buf| alloc.free(buf);

            buffers.nl_phase_buf = try alloc.alloc(math.Complex, n_pw);
            errdefer if (buffers.nl_phase_buf) |buf| alloc.free(buf);

            buffers.nl_xphase_buf = try alloc.alloc(math.Complex, n_pw);
            errdefer if (buffers.nl_xphase_buf) |buf| alloc.free(buf);

            buffers.nl_coeff_buf = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
            errdefer if (buffers.nl_coeff_buf) |buf| alloc.free(buf);

            buffers.nl_coeff2_buf = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
        }
        return buffers;
    }

    fn deinit(self: *NonlocalScfBuffers, alloc: std.mem.Allocator) void {
        if (self.nl_coeff2_buf) |buf| alloc.free(buf);
        if (self.nl_coeff_buf) |buf| alloc.free(buf);
        if (self.nl_xphase_buf) |buf| alloc.free(buf);
        if (self.nl_phase_buf) |buf| alloc.free(buf);
        if (self.nl_out_buf) |buf| alloc.free(buf);
    }
};

const GammaSymmetryData = struct {
    symops: []symmetry_mod.SymOp,
    indsym: [][]usize,
    tnons_shift: [][]math.Vec3,
    irr_info: dynmat_mod.IrreducibleAtomInfo,

    fn deinit(self: *GammaSymmetryData, alloc: std.mem.Allocator) void {
        self.irr_info.deinit(alloc);
        for (self.tnons_shift) |row| alloc.free(row);
        alloc.free(self.tnons_shift);
        for (self.indsym) |row| alloc.free(row);
        alloc.free(self.indsym);
        alloc.free(self.symops);
    }
};

fn build_all_gamma_analytic_perturbations(
    alloc: std.mem.Allocator,
    gs: GroundState,
    storage: *GammaPerturbationBuffers,
) !void {
    for (0..gs.atoms.len) |ia| {
        for (0..3) |dir| {
            const idx = 3 * ia + dir;
            storage.vloc1_gs[idx] = try perturbation.build_local_perturbation(
                alloc,
                gs.grid,
                gs.atoms[ia],
                gs.species,
                dir,
                gs.local_cfg,
                gs.ff_tables,
            );
            storage.rho1_core_gs[idx] = try perturbation.build_core_perturbation(
                alloc,
                gs.grid,
                gs.atoms[ia],
                gs.species,
                dir,
                gs.rho_core_tables,
            );
        }
    }
}

fn solve_sequential_gamma_perturbations(
    alloc: std.mem.Allocator,
    gs: GroundState,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    dfpt_cfg: DfptConfig,
    storage: *GammaPerturbationBuffers,
) !void {
    const dir_names = [_][]const u8{ "x", "y", "z" };
    for (irr_info.irr_atom_indices) |ia| {
        for (0..3) |dir| {
            const idx = 3 * ia + dir;
            log_dfpt(
                "dfpt: solving perturbation atom={d} dir={s} (irreducible)\n",
                .{ ia, dir_names[dir] },
            );
            storage.pert_results[idx] = try solve_perturbation(alloc, gs, ia, dir, dfpt_cfg);
        }
    }
}

fn build_irr_pert_indices(
    alloc: std.mem.Allocator,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]usize {
    const irr_pert_indices = try alloc.alloc(usize, irr_info.n_irr_atoms * 3);
    var pi: usize = 0;
    for (irr_info.irr_atom_indices) |ia| {
        for (0..3) |dir| {
            irr_pert_indices[pi] = 3 * ia + dir;
            pi += 1;
        }
    }
    return irr_pert_indices;
}

fn solve_parallel_gamma_perturbations(
    alloc: std.mem.Allocator,
    io: std.Io,
    gs: GroundState,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    dfpt_cfg: DfptConfig,
    storage: *GammaPerturbationBuffers,
    pert_thread_count: usize,
    dim: usize,
) !void {
    const irr_pert_indices = try build_irr_pert_indices(alloc, irr_info);
    defer alloc.free(irr_pert_indices);

    log_dfpt_info(
        "dfpt: using {d} threads for {d} perturbations ({d} irreducible)\n",
        .{ pert_thread_count, dim, irr_pert_indices.len },
    );

    var next_index = std.atomic.Value(usize).init(0);
    var stop = std.atomic.Value(u8).init(0);
    var worker_err: ?anyerror = null;
    var err_mutex = std.Io.Mutex.init;
    var log_mutex = std.Io.Mutex.init;
    var shared = GammaPertShared{
        .alloc = alloc,
        .io = io,
        .gs = &gs,
        .dfpt_cfg = &dfpt_cfg,
        .pert_results = storage.pert_results,
        .vloc1_gs = storage.vloc1_gs,
        .rho1_core_gs = storage.rho1_core_gs,
        .dim = irr_pert_indices.len,
        .irr_pert_indices = irr_pert_indices,
        .next_index = &next_index,
        .stop = &stop,
        .err = &worker_err,
        .err_mutex = &err_mutex,
        .log_mutex = &log_mutex,
    };
    var workers = try alloc.alloc(GammaPertWorker, pert_thread_count);
    defer alloc.free(workers);

    var threads = try alloc.alloc(std.Thread, pert_thread_count - 1);
    defer alloc.free(threads);

    for (0..pert_thread_count) |ti| {
        workers[ti] = .{ .shared = &shared, .thread_index = ti };
    }
    for (0..pert_thread_count - 1) |ti| {
        threads[ti] = try std.Thread.spawn(.{}, gamma_pert_worker_fn, .{&workers[ti + 1]});
    }
    gamma_pert_worker_fn(&workers[0]);
    for (threads) |t| t.join();
    if (worker_err) |e| return e;
}

fn maybe_run_gamma_diagnostics(
    alloc: std.mem.Allocator,
    gs: GroundState,
    storage: *const GammaPerturbationBuffers,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) !void {
    if (irr_info.n_irr_atoms == n_atoms) {
        try run_diagnostics(alloc, gs, storage.pert_results, storage.vloc1_gs, n_atoms);
        return;
    }
    log_dfpt_info(
        "dfpt: skipping diagnostics (symmetry-reduced: {d}/{d} irreducible atoms)\n",
        .{ irr_info.n_irr_atoms, n_atoms },
    );
}

fn init_gamma_symmetry_data(
    alloc: std.mem.Allocator,
    cell_bohr: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
) !GammaSymmetryData {
    const symops = try symmetry_mod.get_symmetry_ops(alloc, cell_bohr, atoms, 1e-5);
    errdefer alloc.free(symops);

    const sym_data = try dynmat_mod.build_indsym(alloc, symops, atoms, recip, 1e-5);
    errdefer {
        for (sym_data.indsym) |row| alloc.free(row);
        alloc.free(sym_data.indsym);
        for (sym_data.tnons_shift) |row| alloc.free(row);
        alloc.free(sym_data.tnons_shift);
    }

    var irr_info = try dynmat_mod.find_irreducible_atoms(
        alloc,
        symops,
        sym_data.indsym,
        atoms.len,
        .{ .x = 0, .y = 0, .z = 0 },
    );
    errdefer irr_info.deinit(alloc);

    return .{
        .symops = symops,
        .indsym = sym_data.indsym,
        .tnons_shift = sym_data.tnons_shift,
        .irr_info = irr_info,
    };
}

fn log_full_gamma_dynmat(dim: usize, dyn: []const f64) void {
    log_dfpt("dfpt: full dynmat (Ry/bohr², before ASR):\n", .{});
    for (0..dim) |i| {
        const ia = i / 3;
        const da = i % 3;
        for (0..dim) |j| {
            const jb = j / 3;
            const db = j % 3;
            log_dfpt(
                "  D(atom{d},{d}, atom{d},{d}) = {e:.10}\n",
                .{ ia, da, jb, db, dyn[i * dim + j] },
            );
        }
    }
}

fn direction_component(v: math.Vec3, dir: usize) f64 {
    return switch (dir) {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        else => 0.0,
    };
}

fn finalize_gamma_phonon_result(
    alloc: std.mem.Allocator,
    dyn: []f64,
    n_atoms: usize,
    masses: []const f64,
) !PhononResult {
    const dim = 3 * n_atoms;
    log_full_gamma_dynmat(dim, dyn);
    dynmat_mod.apply_asr(dyn, n_atoms);
    log_dfpt_info("dfpt: ASR applied\n", .{});
    dynmat_mod.mass_weight(dyn, n_atoms, masses);

    const result = try dynmat_mod.diagonalize(alloc, dyn, dim);
    log_dfpt_info("dfpt: phonon frequencies (cm⁻¹):\n", .{});
    for (result.frequencies_cm1) |f| log_dfpt_info("dfpt:   {d:.2}\n", .{f});
    return .{
        .frequencies_cm1 = result.frequencies_cm1,
        .omega2 = result.omega2,
        .eigenvectors = result.eigenvectors,
        .dim = result.dim,
    };
}

fn build_gamma_phonon_result(
    alloc: std.mem.Allocator,
    scf_result: *scf_mod.ScfResult,
    gs: GroundState,
    storage: *const GammaPerturbationBuffers,
    ionic: IonicData,
    symmetry: GammaSymmetryData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    cfg: config_mod.Config,
) !PhononResult {
    const n_atoms = gs.atoms.len;
    const rho0_g = try scf_mod.real_to_reciprocal(
        alloc,
        scf_result.grid,
        scf_result.density,
        false,
    );
    defer alloc.free(rho0_g);

    const dyn = try build_gamma_dynmat(
        alloc,
        gs,
        storage.pert_results,
        storage.vloc1_gs,
        storage.rho1_core_gs,
        rho0_g,
        ionic.charges,
        ionic.positions,
        cell_bohr,
        recip,
        volume,
        cfg,
        symmetry.irr_info,
    );
    defer alloc.free(dyn);

    if (symmetry.irr_info.n_irr_atoms < n_atoms) {
        dynmat_mod.reconstruct_dynmat_columns_real(
            dyn,
            n_atoms,
            symmetry.irr_info,
            symmetry.symops,
            symmetry.indsym,
            cell_bohr,
        );
    }
    return try finalize_gamma_phonon_result(alloc, dyn, n_atoms, ionic.masses);
}

fn maybe_compute_gamma_dielectric(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    gs: *GroundState,
    local_r: []const f64,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
) !void {
    if (!cfg.dfpt.compute_dielectric) return;

    log_dfpt_info("dfpt: computing dielectric tensor (ddk at all k-points)\n", .{});
    const electric = @import("electric.zig");
    const dielectric = try electric.compute_dielectric_all_k(
        alloc,
        io,
        cfg,
        gs,
        local_r,
        gs.species,
        gs.atoms,
        cell_bohr,
        recip,
        volume,
    );
    log_dfpt_info("dfpt: dielectric tensor ε∞:\n", .{});
    for (0..3) |i| {
        log_dfpt_info("  {d:.6} {d:.6} {d:.6}\n", .{
            dielectric.epsilon[i][0],
            dielectric.epsilon[i][1],
            dielectric.epsilon[i][2],
        });
    }

    var out_dir = std.Io.Dir.cwd().openDir(io, cfg.out_dir, .{}) catch null;
    defer if (out_dir) |*d| d.close(io);

    if (out_dir) |od| {
        electric.write_electric_results(io, od, dielectric) catch |err| {
            log_dfpt_warn("dfpt: warning: failed to write electric.dat: {}\n", .{err});
        };
    }
}

/// Run DFPT phonon calculation at the Γ-point.
/// Takes converged SCF results and returns phonon frequencies.
pub fn run_phonon(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    model: *const model_mod.Model,
) !PhononResult {
    const n_atoms = model.atoms.len;
    const dim = 3 * n_atoms;

    log_dfpt_info("dfpt: starting phonon calculation ({d} atoms, dim={d})\n", .{ n_atoms, dim });
    var prepared = try dfpt.prepare_ground_state(
        alloc,
        io,
        cfg,
        scf_result,
        model.species,
        model.atoms,
        model.volume_bohr,
        model.recip,
    );
    var gs = prepared.gs;
    const dfpt_cfg = DfptConfig.from_config(cfg);
    const pert_thread_count = dfpt.perturbation_thread_count(dim, dfpt_cfg.perturbation_threads);
    var symmetry = try init_gamma_symmetry_data(alloc, model.cell_bohr, model.atoms, model.recip);
    var storage = try GammaPerturbationBuffers.init(alloc, dim);
    const ionic = try IonicData.init(alloc, model.species, model.atoms);
    defer ionic.deinit(alloc);
    defer storage.deinit(alloc);
    defer symmetry.deinit(alloc);
    defer prepared.deinit();

    log_dfpt_info("dfpt: {d} symops, {d}/{d} irreducible atoms\n", .{
        symmetry.symops.len,
        symmetry.irr_info.n_irr_atoms,
        n_atoms,
    });
    try run_gamma_perturbation_solves(
        alloc,
        io,
        gs,
        symmetry.irr_info,
        dfpt_cfg,
        &storage,
        pert_thread_count,
        dim,
        n_atoms,
    );
    const result = try build_gamma_phonon_result(
        alloc,
        scf_result,
        gs,
        &storage,
        ionic,
        symmetry,
        model.cell_bohr,
        model.recip,
        model.volume_bohr,
        cfg,
    );
    try maybe_compute_gamma_dielectric_for_model(alloc, io, cfg, &gs, prepared.local_r, model);
    return result;
}

fn run_gamma_perturbation_solves(
    alloc: std.mem.Allocator,
    io: std.Io,
    gs: GroundState,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
    dfpt_cfg: DfptConfig,
    storage: *GammaPerturbationBuffers,
    pert_thread_count: usize,
    dim: usize,
    n_atoms: usize,
) !void {
    try build_all_gamma_analytic_perturbations(alloc, gs, storage);
    if (pert_thread_count <= 1) {
        try solve_sequential_gamma_perturbations(alloc, gs, irr_info, dfpt_cfg, storage);
    } else {
        try solve_parallel_gamma_perturbations(
            alloc,
            io,
            gs,
            irr_info,
            dfpt_cfg,
            storage,
            pert_thread_count,
            dim,
        );
    }
    try maybe_run_gamma_diagnostics(alloc, gs, storage, n_atoms, irr_info);
}

fn maybe_compute_gamma_dielectric_for_model(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    gs: *GroundState,
    local_r: []const f64,
    model: *const model_mod.Model,
) !void {
    try maybe_compute_gamma_dielectric(
        alloc,
        io,
        cfg,
        gs,
        local_r,
        model.cell_bohr,
        model.recip,
        model.volume_bohr,
    );
}

fn init_solve_perturbation_inputs(
    alloc: std.mem.Allocator,
    gs: GroundState,
    atom_index: usize,
    direction: usize,
) !SolvePerturbationInputs {
    const vloc1_g = try perturbation.build_local_perturbation(
        alloc,
        gs.grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        gs.local_cfg,
        gs.ff_tables,
    );
    errdefer alloc.free(vloc1_g);

    const rho1_core_g = try perturbation.build_core_perturbation(
        alloc,
        gs.grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        gs.rho_core_tables,
    );
    errdefer alloc.free(rho1_core_g);

    const rho1_core_g_for_ifft = try alloc.alloc(math.Complex, gs.grid.count());
    defer alloc.free(rho1_core_g_for_ifft);

    @memcpy(rho1_core_g_for_ifft, rho1_core_g);
    const rho1_core_r = try scf_mod.reciprocal_to_real(alloc, gs.grid, rho1_core_g_for_ifft);
    errdefer alloc.free(rho1_core_r);

    return .{
        .vloc1_g = vloc1_g,
        .rho1_core_g = rho1_core_g,
        .rho1_core_r = rho1_core_r,
    };
}

fn init_psi1_buffers(
    alloc: std.mem.Allocator,
    n_occ: usize,
    n_pw: usize,
) ![][]math.Complex {
    const psi1 = try alloc.alloc([]math.Complex, n_occ);
    var allocated: usize = 0;
    errdefer {
        for (0..allocated) |n| alloc.free(psi1[n]);
        alloc.free(psi1);
    }

    for (0..n_occ) |n| {
        psi1[n] = try alloc.alloc(math.Complex, n_pw);
        @memset(psi1[n], math.complex.init(0.0, 0.0));
        allocated = n + 1;
    }
    return psi1;
}

fn build_total_perturbation_potential(
    alloc: std.mem.Allocator,
    gs: GroundState,
    rho1_g: []const math.Complex,
    rho1_core_r: []const f64,
    vloc1_g: []const math.Complex,
) ![]f64 {
    const total = gs.grid.count();
    const vh1_g = try perturbation.build_hartree_perturbation(alloc, gs.grid, rho1_g);
    defer alloc.free(vh1_g);

    const rho1_g_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_g_copy);

    @memcpy(rho1_g_copy, rho1_g);
    const rho1_r = try scf_mod.reciprocal_to_real(alloc, gs.grid, rho1_g_copy);
    defer alloc.free(rho1_r);

    const rho1_total_r = try alloc.alloc(f64, total);
    defer alloc.free(rho1_total_r);

    for (0..total) |i| {
        rho1_total_r[i] = rho1_r[i] + rho1_core_r[i];
    }
    const vxc1_r = try perturbation.build_xc_perturbation_full(alloc, gs, rho1_total_r);
    defer alloc.free(vxc1_r);

    const vxc1_g = try scf_mod.real_to_reciprocal(alloc, gs.grid, vxc1_r, false);
    defer alloc.free(vxc1_g);

    const vtot1_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(vtot1_g);

    for (0..total) |i| {
        vtot1_g[i] = math.complex.add(
            math.complex.add(vloc1_g[i], vh1_g[i]),
            vxc1_g[i],
        );
    }
    const vtot1_g_for_ifft = try alloc.alloc(math.Complex, total);
    defer alloc.free(vtot1_g_for_ifft);

    @memcpy(vtot1_g_for_ifft, vtot1_g);
    return try scf_mod.reciprocal_to_real(alloc, gs.grid, vtot1_g_for_ifft);
}

fn add_nonlocal_perturbation_to_rhs(
    gs: GroundState,
    atom_index: usize,
    direction: usize,
    psi0: []const math.Complex,
    rhs: []math.Complex,
    nl_buffers: *const NonlocalScfBuffers,
) void {
    const nl_ctx = gs.apply_ctx.nonlocal_ctx orelse return;
    perturbation.apply_nonlocal_perturbation(
        gs.gvecs,
        gs.atoms,
        nl_ctx,
        atom_index,
        direction,
        1.0 / gs.grid.volume,
        psi0,
        nl_buffers.nl_out_buf.?,
        nl_buffers.nl_phase_buf.?,
        nl_buffers.nl_xphase_buf.?,
        nl_buffers.nl_coeff_buf.?,
        nl_buffers.nl_coeff2_buf.?,
    );
    for (0..rhs.len) |g| {
        rhs[g] = math.complex.add(rhs[g], nl_buffers.nl_out_buf.?[g]);
    }
}

fn update_psi1_for_all_bands(
    alloc: std.mem.Allocator,
    gs: GroundState,
    atom_index: usize,
    direction: usize,
    vtot1_r: []const f64,
    psi1: [][]math.Complex,
    cfg: DfptConfig,
    nl_buffers: *const NonlocalScfBuffers,
) !void {
    for (0..gs.n_occ) |n| {
        const rhs = try apply_v1_psi(
            alloc,
            gs.grid,
            gs.gvecs,
            vtot1_r,
            gs.wavefunctions[n],
            gs.apply_ctx,
        );
        defer alloc.free(rhs);

        add_nonlocal_perturbation_to_rhs(
            gs,
            atom_index,
            direction,
            gs.wavefunctions[n],
            rhs,
            nl_buffers,
        );
        for (0..gs.gvecs.len) |g| {
            rhs[g] = math.complex.scale(rhs[g], -1.0);
        }
        sternheimer.project_conduction(rhs, gs.wavefunctions, gs.n_occ);
        const result = try sternheimer.solve(
            alloc,
            gs.apply_ctx,
            rhs,
            gs.eigenvalues[n],
            gs.wavefunctions,
            gs.n_occ,
            gs.gvecs,
            .{
                .tol = cfg.sternheimer_tol,
                .max_iter = cfg.sternheimer_max_iter,
                .alpha_shift = cfg.alpha_shift,
            },
        );
        alloc.free(psi1[n]);
        psi1[n] = result.psi1;
    }
}

fn mix_density_response(
    rho1_g: []math.Complex,
    new_rho1_g: []const math.Complex,
    beta: f64,
) f64 {
    var diff_norm: f64 = 0.0;
    for (0..rho1_g.len) |i| {
        const dr = new_rho1_g[i].r - rho1_g[i].r;
        const di = new_rho1_g[i].i - rho1_g[i].i;
        diff_norm += dr * dr + di * di;
        rho1_g[i] = math.complex.add(
            math.complex.scale(rho1_g[i], 1.0 - beta),
            math.complex.scale(new_rho1_g[i], beta),
        );
    }
    return @sqrt(diff_norm);
}

fn log_rho1_consistency(
    final_rho1_g: []const math.Complex,
    mixed_rho1_g: []const math.Complex,
) void {
    var rho1_norm: f64 = 0.0;
    var rho1_diff: f64 = 0.0;
    for (0..final_rho1_g.len) |i| {
        rho1_norm += final_rho1_g[i].r * final_rho1_g[i].r +
            final_rho1_g[i].i * final_rho1_g[i].i;
        const dr = final_rho1_g[i].r - mixed_rho1_g[i].r;
        const di = final_rho1_g[i].i - mixed_rho1_g[i].i;
        rho1_diff += dr * dr + di * di;
    }
    log_dfpt(
        "dfpt_scf: final |rho1_g|={e:.6} |rho1_mixed - rho1_psi1|={e:.6}\n",
        .{ @sqrt(rho1_norm), @sqrt(rho1_diff) },
    );
}

/// Run DFPT SCF for a single perturbation (atom_index, direction).
/// Returns the converged first-order density and wavefunctions.
pub fn solve_perturbation(
    alloc: std.mem.Allocator,
    gs: GroundState,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
) !PerturbationResult {
    const total = gs.grid.count();
    var inputs = try init_solve_perturbation_inputs(alloc, gs, atom_index, direction);
    defer inputs.deinit(alloc);

    const rho1_g = try alloc.alloc(math.Complex, total);
    errdefer alloc.free(rho1_g);

    @memset(rho1_g, math.complex.init(0.0, 0.0));
    const psi1 = try init_psi1_buffers(alloc, gs.n_occ, gs.gvecs.len);
    var cleanup = PerturbationResult{ .rho1_g = rho1_g, .psi1 = psi1 };
    errdefer cleanup.deinit(alloc);

    var nl_buffers = try NonlocalScfBuffers.init(alloc, gs.apply_ctx, gs.gvecs.len);
    defer nl_buffers.deinit(alloc);

    var iter: usize = 0;
    while (iter < cfg.scf_max_iter) : (iter += 1) {
        const vtot1_r = try build_total_perturbation_potential(
            alloc,
            gs,
            rho1_g,
            inputs.rho1_core_r,
            inputs.vloc1_g,
        );
        defer alloc.free(vtot1_r);

        try update_psi1_for_all_bands(
            alloc,
            gs,
            atom_index,
            direction,
            vtot1_r,
            psi1,
            cfg,
            &nl_buffers,
        );
        const new_rho1_r = try compute_rho1(
            alloc,
            gs.grid,
            gs.gvecs,
            gs.wavefunctions,
            psi1,
            gs.n_occ,
            gs.apply_ctx,
        );
        defer alloc.free(new_rho1_r);

        const new_rho1_g = try scf_mod.real_to_reciprocal(alloc, gs.grid, new_rho1_r, false);
        const diff_norm = mix_density_response(rho1_g, new_rho1_g, cfg.mixing_beta);
        alloc.free(new_rho1_g);
        log_dfpt("dfpt_scf: iter={d} diff_norm={e:.6}\n", .{ iter, diff_norm });
        if (diff_norm < cfg.scf_tol) {
            log_dfpt("dfpt_scf: converged at iter={d}\n", .{iter});
            break;
        }
    }
    const final_rho1_r = try compute_rho1(
        alloc,
        gs.grid,
        gs.gvecs,
        gs.wavefunctions,
        psi1,
        gs.n_occ,
        gs.apply_ctx,
    );
    defer alloc.free(final_rho1_r);

    const final_rho1_g = try scf_mod.real_to_reciprocal(alloc, gs.grid, final_rho1_r, false);
    log_rho1_consistency(final_rho1_g, rho1_g);
    alloc.free(rho1_g);
    return .{ .rho1_g = final_rho1_g, .psi1 = psi1 };
}

/// Apply V^(1)(r)|ψ(r)⟩ → result in PW basis.
/// Uses FFT: ψ(r) = IFFT[scatter(ψ_G)], multiply by V^(1)(r), FFT back, gather.
pub fn apply_v1_psi(
    alloc: std.mem.Allocator,
    grid: Grid,
    gvecs: []const plane_wave.GVector,
    v1_r: []const f64,
    psi: []const math.Complex,
    ctx: *scf_mod.ApplyContext,
) ![]math.Complex {
    const n_pw = gvecs.len;
    const total = grid.count();

    // Scatter PW coefficients to full grid
    const work_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g);

    @memset(work_g, math.complex.init(0.0, 0.0));
    ctx.map.scatter(psi, work_g);

    // IFFT to real space
    const work_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r);

    try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, work_g, work_r, null);

    // Multiply by V^(1)(r)
    for (0..total) |i| {
        work_r[i] = math.complex.scale(work_r[i], v1_r[i]);
    }

    // FFT back to reciprocal space
    const work_g_out = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g_out);

    try scf_mod.fft_complex_to_reciprocal_in_place(alloc, grid, work_r, work_g_out, null);

    // Gather back to PW basis
    const result = try alloc.alloc(math.Complex, n_pw);
    ctx.map.gather(work_g_out, result);

    return result;
}

/// Compute first-order density response from wavefunctions.
/// ρ^(1)(r) = (4/Ω) × Σ_n Re[ψ_n^(0)*(r) × ψ_n^(1)(r)]
/// Factor 4 = 2 (spin degeneracy) × 2 (derivative of |ψ|²).
/// Factor 1/Ω from the PW normalization convention (ψ^grid = √Ω × ψ^physical).
pub fn compute_rho1(
    alloc: std.mem.Allocator,
    grid: Grid,
    gvecs: []const plane_wave.GVector,
    psi0: []const []const math.Complex,
    psi1: []const []const math.Complex,
    n_occ: usize,
    ctx: *scf_mod.ApplyContext,
) ![]f64 {
    const total = grid.count();
    const rho1_r = try alloc.alloc(f64, total);
    @memset(rho1_r, 0.0);

    const work_g0 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g0);

    const work_r0 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r0);

    const work_g1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g1);

    const work_r1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r1);

    // Weight: 4/Ω = 2(spin) × 2(d|ψ|²) × (1/Ω)(volume normalization)
    const weight = 4.0 / grid.volume;

    for (0..n_occ) |n| {
        // ψ_n^(0)(r) via IFFT
        @memset(work_g0, math.complex.init(0.0, 0.0));
        ctx.map.scatter(psi0[n], work_g0);
        try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, work_g0, work_r0, null);

        // ψ_n^(1)(r) via IFFT
        @memset(work_g1, math.complex.init(0.0, 0.0));
        ctx.map.scatter(psi1[n], work_g1);
        try scf_mod.fft_reciprocal_to_complex_in_place(alloc, grid, work_g1, work_r1, null);

        // ρ^(1)(r) += (4/Ω) × Re[ψ_n^(0)*(r) × ψ_n^(1)(r)]
        for (0..total) |i| {
            const conj0 = math.complex.conj(work_r0[i]);
            const prod = math.complex.mul(conj0, work_r1[i]);
            rho1_r[i] += weight * prod.r;
        }
    }
    _ = gvecs;

    return rho1_r;
}

/// Compute the electronic contribution to the dynamical matrix element.
/// D^elec_{Iα,Jβ} = Σ_G conj(V^(1)_{Iα}(G)) × ρ^(1)_{Jβ}(G) × Ω
pub fn compute_elec_dynmat_element(
    vloc1_g: []const math.Complex,
    rho1_g: []const math.Complex,
    volume: f64,
) f64 {
    var sum = math.complex.init(0.0, 0.0);
    for (0..vloc1_g.len) |i| {
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(vloc1_g[i]), rho1_g[i]));
    }
    return sum.r * volume;
}

/// Compute the nonlocal response contribution to the dynamical matrix.
/// C_nl_{Iα,Jβ} = 4 × Σ_n Re[⟨ψ_n|V_nl^(1)_{Iα}|δψ_n,Jβ⟩]
///
/// This accounts for the nonlocal pseudopotential's contribution to the
/// force constant via the first-order wavefunctions.
pub fn compute_nonlocal_response_dynmat(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]f64 {
    const dim = 3 * n_atoms;
    const n_pw = gs.gvecs.len;
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    const nl_ctx = gs.apply_ctx.nonlocal_ctx orelse return dyn;

    // Allocate work buffers
    const nl_out = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_out);

    const nl_phase = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_phase);

    const nl_xphase = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_xphase);

    const nl_coeff = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
    defer alloc.free(nl_coeff);

    const nl_coeff2 = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
    defer alloc.free(nl_coeff2);

    for (0..dim) |i| {
        const ia = i / 3;
        const dir_a = i % 3;
        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            var sum: f64 = 0.0;
            for (0..gs.n_occ) |n| {
                // Compute V_nl^(1)_{ia,dir_a} |δψ_n,j⟩
                perturbation.apply_nonlocal_perturbation(
                    gs.gvecs,
                    gs.atoms,
                    nl_ctx,
                    ia,
                    dir_a,
                    1.0 / gs.grid.volume,
                    pert_results[j].psi1[n],
                    nl_out,
                    nl_phase,
                    nl_xphase,
                    nl_coeff,
                    nl_coeff2,
                );
                // ⟨ψ_n|V_nl^(1)|δψ_n⟩ = Σ_G ψ*_n(G) × nl_out(G)
                var ip = math.complex.init(0.0, 0.0);
                for (0..n_pw) |g| {
                    ip = math.complex.add(ip, math.complex.mul(
                        math.complex.conj(gs.wavefunctions[n][g]),
                        nl_out[g],
                    ));
                }
                sum += ip.r;
            }
            dyn[i * dim + j] = 4.0 * sum;
        }
    }

    return dyn;
}

/// Compute the NLCC cross-term contribution to the dynamical matrix.
///
/// D_NLCC_cross(Iα,Jβ) = ∫ V_xc^(1)[ρ^(1)_core,I](r) × ρ^(1)_total,J(r) dr
///
/// For LDA this reduces to ∫ f_xc(r) × ρ^(1)_core,I(r) × ρ^(1)_total,J(r) dr.
/// For GGA, V_xc^(1) includes gradient-dependent terms via build_xc_perturbation_full.
pub fn compute_nlcc_cross_dynmat(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: GroundState,
    pert_results: []PerturbationResult,
    rho1_core_gs: []const []math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]f64 {
    const dim = 3 * n_atoms;
    const total = grid.count();
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    const dv = grid.volume / @as(f64, @floatFromInt(total));

    for (0..dim) |i| {
        // Get ρ^(1)_core,Iα(r)
        const rho1_core_i_g_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_core_i_g_copy);

        @memcpy(rho1_core_i_g_copy, rho1_core_gs[i]);
        const rho1_core_i_r = try scf_mod.reciprocal_to_real(alloc, grid, rho1_core_i_g_copy);
        defer alloc.free(rho1_core_i_r);

        // Build V_xc^(1)[ρ^(1)_core,I] using full GGA-aware kernel
        const vxc1_core_i = try perturbation.build_xc_perturbation_full(alloc, gs, rho1_core_i_r);
        defer alloc.free(vxc1_core_i);

        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            // Get ρ^(1)_total,J(r) = ρ^(1)_val,J + ρ^(1)_core,J
            const rho1_val_g_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_val_g_copy);

            @memcpy(rho1_val_g_copy, pert_results[j].rho1_g);
            const rho1_val_r = try scf_mod.reciprocal_to_real(alloc, grid, rho1_val_g_copy);
            defer alloc.free(rho1_val_r);

            const rho1_core_j_g_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_core_j_g_copy);

            @memcpy(rho1_core_j_g_copy, rho1_core_gs[j]);
            const rho1_core_j_r = try scf_mod.reciprocal_to_real(alloc, grid, rho1_core_j_g_copy);
            defer alloc.free(rho1_core_j_r);

            // D_NLCC(I,J) = ∫ V_xc^(1)[ρ^(1)_core,I](r) × ρ^(1)_total,J(r) dr
            var sum: f64 = 0.0;
            for (0..total) |r| {
                const rho1_total_j = rho1_val_r[r] + rho1_core_j_r[r];
                sum += vxc1_core_i[r] * rho1_total_j;
            }
            dyn[i * dim + j] = sum * dv;
        }
    }

    return dyn;
}

fn add_dynmat_contribution(dst: []f64, src: []const f64) void {
    for (0..dst.len) |i| dst[i] += src[i];
}

fn log_dynmat_sample(label: []const u8, dyn: []const f64) void {
    const off_diag = if (dyn.len > 3) dyn[3] else 0.0;
    log_dfpt("{s} D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ label, dyn[0], off_diag });
}

fn add_electronic_dynmat_contribution(
    dyn: []f64,
    vloc1_gs: []const []math.Complex,
    pert_results: []const PerturbationResult,
    volume: f64,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) void {
    const dim = irr_info.is_irreducible.len * 3;
    for (0..dim) |i| {
        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            dyn[i * dim + j] = compute_elec_dynmat_element(
                vloc1_gs[i],
                pert_results[j].rho1_g,
                volume,
            );
        }
    }
    log_dynmat_sample("dfpt: electronic", dyn);
}

fn add_ewald_dynmat_contribution(
    alloc: std.mem.Allocator,
    dyn: []f64,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    charges: []const f64,
    positions: []const math.Vec3,
) !void {
    const ewald_dyn = try ewald2.ewald_dynmat(alloc, cell_bohr, recip, charges, positions);
    defer alloc.free(ewald_dyn);

    log_dfpt(
        "dfpt: ewald D(0x,0x)={e:.10} D(0x,1x)={e:.10} (Ha)\n",
        .{ ewald_dyn[0], if (ewald_dyn.len > 3) ewald_dyn[3] else 0.0 },
    );
    for (0..dyn.len) |i| dyn[i] += ewald_dyn[i] * 2.0;
}

fn add_self_energy_dynmat_contribution(
    alloc: std.mem.Allocator,
    dyn: []f64,
    grid: Grid,
    gs: GroundState,
    rho0_g: []math.Complex,
) !void {
    const self_dyn = try dynmat_contrib.compute_self_energy_dynmat(
        alloc,
        grid,
        gs.species,
        gs.atoms,
        rho0_g,
        gs.local_cfg,
        gs.ff_tables,
    );
    defer alloc.free(self_dyn);

    log_dynmat_sample("dfpt: self-energy", self_dyn);
    add_dynmat_contribution(dyn, self_dyn);
}

fn add_nonlocal_dynmat_contributions(
    alloc: std.mem.Allocator,
    dyn: []f64,
    gs: GroundState,
    pert_results: []PerturbationResult,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) !void {
    const nl_resp_dyn = try compute_nonlocal_response_dynmat(
        alloc,
        gs,
        pert_results,
        n_atoms,
        irr_info,
    );
    defer alloc.free(nl_resp_dyn);

    log_dynmat_sample("dfpt: nl-response", nl_resp_dyn);
    add_dynmat_contribution(dyn, nl_resp_dyn);

    const nl_self_dyn = try dynmat_contrib.compute_nonlocal_self_energy_dynmat(alloc, gs, n_atoms);
    defer alloc.free(nl_self_dyn);

    log_dynmat_sample("dfpt: nl-self-energy", nl_self_dyn);
    add_dynmat_contribution(dyn, nl_self_dyn);
}

fn add_nlcc_dynmat_contributions(
    alloc: std.mem.Allocator,
    dyn: []f64,
    grid: Grid,
    gs: GroundState,
    pert_results: []PerturbationResult,
    rho1_core_gs: []const []math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) !void {
    if (gs.rho_core == null) return;

    const nlcc_cross_dyn = try compute_nlcc_cross_dynmat(
        alloc,
        grid,
        gs,
        pert_results,
        rho1_core_gs,
        n_atoms,
        irr_info,
    );
    defer alloc.free(nlcc_cross_dyn);

    log_dynmat_sample("dfpt: nlcc-cross", nlcc_cross_dyn);
    add_dynmat_contribution(dyn, nlcc_cross_dyn);

    const vxc_r_slice = gs.vxc_r.?;
    const vxc_r_copy = try alloc.alloc(f64, vxc_r_slice.len);
    defer alloc.free(vxc_r_copy);

    @memcpy(vxc_r_copy, vxc_r_slice);
    const vxc_g = try scf_mod.real_to_reciprocal(alloc, grid, vxc_r_copy, false);
    defer alloc.free(vxc_g);

    const nlcc_self_dyn = try dynmat_contrib.compute_nlcc_self_dynmat(
        alloc,
        grid,
        gs.species,
        gs.atoms,
        vxc_g,
        gs.rho_core_tables,
    );
    defer alloc.free(nlcc_self_dyn);

    log_dynmat_sample("dfpt: nlcc-self", nlcc_self_dyn);
    add_dynmat_contribution(dyn, nlcc_self_dyn);
}

fn add_d3_dynmat_contribution(
    alloc: std.mem.Allocator,
    dyn: []f64,
    gs: GroundState,
    cell_bohr: math.Mat3,
    cfg: config_mod.Config,
) !void {
    if (!cfg.vdw.enabled) return;

    const n_atoms = gs.atoms.len;
    const atomic_numbers = try alloc.alloc(usize, n_atoms);
    defer alloc.free(atomic_numbers);

    const atom_positions = try alloc.alloc(math.Vec3, n_atoms);
    defer alloc.free(atom_positions);

    for (gs.atoms, 0..) |atom, idx| {
        atomic_numbers[idx] =
            d3_params.atomic_number(gs.species[atom.species_index].symbol) orelse 0;
        atom_positions[idx] = atom.position;
    }
    var damping = d3_params.pbe_d3bj;
    if (cfg.vdw.s6) |v| damping.s6 = v;
    if (cfg.vdw.s8) |v| damping.s8 = v;
    if (cfg.vdw.a1) |v| damping.a1 = v;
    if (cfg.vdw.a2) |v| damping.a2 = v;

    const d3_dyn = try d3.compute_dynmat(
        alloc,
        atomic_numbers,
        atom_positions,
        cell_bohr,
        damping,
        cfg.vdw.cutoff_radius,
        cfg.vdw.cn_cutoff,
    );
    defer alloc.free(d3_dyn);

    log_dynmat_sample("dfpt: d3-disp", d3_dyn);
    add_dynmat_contribution(dyn, d3_dyn);
}

/// Build the Γ-point dynamical matrix from all contributions:
/// electronic, Ewald, self-energy, nonlocal response, nonlocal self-energy,
/// NLCC cross, and NLCC self.
/// Returns an owned slice of size dim×dim (caller must free).
fn build_gamma_dynmat(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    vloc1_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []math.Complex,
    charges: []const f64,
    positions: []const math.Vec3,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    cfg: config_mod.Config,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]f64 {
    const n_atoms = gs.atoms.len;
    const grid = gs.grid;
    const dyn = try alloc.alloc(f64, 9 * n_atoms * n_atoms);
    errdefer alloc.free(dyn);

    @memset(dyn, 0.0);
    add_electronic_dynmat_contribution(dyn, vloc1_gs, pert_results, volume, irr_info);
    try add_ewald_dynmat_contribution(alloc, dyn, cell_bohr, recip, charges, positions);
    try add_self_energy_dynmat_contribution(alloc, dyn, grid, gs, rho0_g);
    try add_nonlocal_dynmat_contributions(alloc, dyn, gs, pert_results, n_atoms, irr_info);
    try add_nlcc_dynmat_contributions(
        alloc,
        dyn,
        grid,
        gs,
        pert_results,
        rho1_core_gs,
        n_atoms,
        irr_info,
    );
    try add_d3_dynmat_contribution(alloc, dyn, gs, cell_bohr, cfg);
    log_dfpt("dfpt: total dynmat diagonal (Ry):", .{});
    for (0..3 * n_atoms) |i| {
        log_dfpt(" {e:.6}", .{dyn[i * 3 * n_atoms + i]});
    }
    log_dfpt("\n", .{});
    return dyn;
}

fn run_asr_band_diagnostic(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    n_atoms: usize,
    dir: usize,
    band: usize,
) !void {
    const n_pw = gs.gvecs.len;
    const sum_psi1 = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(sum_psi1);

    @memset(sum_psi1, math.complex.init(0.0, 0.0));
    for (0..n_atoms) |ia| {
        const j_idx = 3 * ia + dir;
        for (0..n_pw) |g| {
            sum_psi1[g] = math.complex.add(sum_psi1[g], pert_results[j_idx].psi1[band][g]);
        }
    }

    const expected = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(expected);

    for (0..n_pw) |g| {
        const g_d = direction_component(gs.gvecs[g].cart, dir);
        const psi = gs.wavefunctions[band][g];
        expected[g] = math.complex.init(psi.i * g_d, -psi.r * g_d);
    }
    sternheimer.project_conduction(expected, gs.wavefunctions, gs.n_occ);

    var diff_norm: f64 = 0.0;
    var exp_norm: f64 = 0.0;
    var sum_norm: f64 = 0.0;
    for (0..n_pw) |g| {
        const dr = sum_psi1[g].r - expected[g].r;
        const di = sum_psi1[g].i - expected[g].i;
        diff_norm += dr * dr + di * di;
        exp_norm += expected[g].r * expected[g].r + expected[g].i * expected[g].i;
        sum_norm += sum_psi1[g].r * sum_psi1[g].r + sum_psi1[g].i * sum_psi1[g].i;
    }
    const asr_rel = if (exp_norm > 1e-30) @sqrt(diff_norm / exp_norm) else 0.0;
    log_dfpt(
        "dfpt_asr: dir={d} band={d} |Σ_J ψ1|={e:.6} |expected|={e:.6}" ++
            " |diff|={e:.6} rel={e:.6}\n",
        .{ dir, band, @sqrt(sum_norm), @sqrt(exp_norm), @sqrt(diff_norm), asr_rel },
    );
}

fn run_asr_diagnostics(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    n_atoms: usize,
) !void {
    for (0..3) |dir| {
        for (0..gs.n_occ) |band| {
            try run_asr_band_diagnostic(alloc, gs, pert_results, n_atoms, dir, band);
        }
    }
}

fn run_nonlocal_commutator_band_diagnostic(
    alloc: std.mem.Allocator,
    gs: GroundState,
    n_atoms: usize,
    dir: usize,
    band: usize,
) !void {
    const nl_ctx = gs.apply_ctx.nonlocal_ctx.?;
    const n_pw = gs.gvecs.len;
    const nl_out1 = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_out1);

    const nl_out2 = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_out2);

    const nl_out3 = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_out3);

    const work_igpsi = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(work_igpsi);

    const nl_phase = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_phase);

    const nl_xphase = try alloc.alloc(math.Complex, n_pw);
    defer alloc.free(nl_xphase);

    const nl_coeff = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
    defer alloc.free(nl_coeff);

    const nl_coeff2 = try alloc.alloc(math.Complex, nl_ctx.max_m_total);
    defer alloc.free(nl_coeff2);

    @memset(nl_out1, math.complex.init(0.0, 0.0));
    for (0..n_atoms) |ia| {
        perturbation.apply_nonlocal_perturbation(
            gs.gvecs,
            gs.atoms,
            nl_ctx,
            ia,
            dir,
            1.0 / gs.grid.volume,
            gs.wavefunctions[band],
            nl_out2,
            nl_phase,
            nl_xphase,
            nl_coeff,
            nl_coeff2,
        );
        for (0..n_pw) |g| nl_out1[g] = math.complex.add(nl_out1[g], nl_out2[g]);
    }
    for (0..n_pw) |g| {
        const g_d = direction_component(gs.gvecs[g].cart, dir);
        const psi = gs.wavefunctions[band][g];
        work_igpsi[g] = math.complex.init(-psi.i * g_d, psi.r * g_d);
    }
    try scf_mod.apply_nonlocal_potential(gs.apply_ctx, work_igpsi, nl_out2);
    try scf_mod.apply_nonlocal_potential(gs.apply_ctx, gs.wavefunctions[band], nl_out3);
    for (0..n_pw) |g| {
        const g_d = direction_component(gs.gvecs[g].cart, dir);
        const v = nl_out3[g];
        nl_out3[g] = math.complex.init(-v.i * g_d, v.r * g_d);
    }

    var diff_norm: f64 = 0.0;
    var lhs_norm: f64 = 0.0;
    var rhs_norm: f64 = 0.0;
    for (0..n_pw) |g| {
        const expected_g = math.complex.sub(nl_out2[g], nl_out3[g]);
        const dr = nl_out1[g].r - expected_g.r;
        const di = nl_out1[g].i - expected_g.i;
        diff_norm += dr * dr + di * di;
        lhs_norm += nl_out1[g].r * nl_out1[g].r + nl_out1[g].i * nl_out1[g].i;
        rhs_norm += expected_g.r * expected_g.r + expected_g.i * expected_g.i;
    }
    const vnl_rel = if (rhs_norm > 1e-30) @sqrt(diff_norm / rhs_norm) else 0.0;
    log_dfpt(
        "dfpt_vnl_test: dir={d} band={d} |Σ V1_nl|ψ|={e:.6}" ++
            " |V_nl|iGψ⟩-iG V_nl|ψ⟩|={e:.6} |diff|={e:.6} rel={e:.6}\n",
        .{ dir, band, @sqrt(lhs_norm), @sqrt(rhs_norm), @sqrt(diff_norm), vnl_rel },
    );
}

fn run_nonlocal_commutator_diagnostics(
    alloc: std.mem.Allocator,
    gs: GroundState,
    n_atoms: usize,
) !void {
    if (gs.apply_ctx.nonlocal_ctx == null) return;
    for (0..1) |dir| {
        for (0..gs.n_occ) |band| {
            try run_nonlocal_commutator_band_diagnostic(alloc, gs, n_atoms, dir, band);
        }
    }
}

fn compute_wavefunction_dynmat_entry(
    alloc: std.mem.Allocator,
    gs: GroundState,
    vloc1_r: []const f64,
    psi1_set: []const []const math.Complex,
) !f64 {
    var value: f64 = 0.0;
    for (0..gs.n_occ) |band| {
        const vpsi = try apply_v1_psi(
            alloc,
            gs.grid,
            gs.gvecs,
            vloc1_r,
            psi1_set[band],
            gs.apply_ctx,
        );
        defer alloc.free(vpsi);

        var ip = math.complex.init(0.0, 0.0);
        for (0..gs.gvecs.len) |g| {
            ip = math.complex.add(
                ip,
                math.complex.mul(math.complex.conj(gs.wavefunctions[band][g]), vpsi[g]),
            );
        }
        value += 4.0 * ip.r;
    }
    return value;
}

fn run_electronic_wavefunction_diagnostics(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    vloc1_gs: []const []math.Complex,
) !void {
    const vloc1_0x_g_copy = try alloc.alloc(math.Complex, gs.grid.count());
    defer alloc.free(vloc1_0x_g_copy);

    @memcpy(vloc1_0x_g_copy, vloc1_gs[0]);
    const vloc1_0x_r = try scf_mod.reciprocal_to_real(alloc, gs.grid, vloc1_0x_g_copy);
    defer alloc.free(vloc1_0x_r);

    const d_wf_00 = try compute_wavefunction_dynmat_entry(
        alloc,
        gs,
        vloc1_0x_r,
        pert_results[0].psi1,
    );
    const d_wf_03 = if (pert_results.len > 3)
        try compute_wavefunction_dynmat_entry(alloc, gs, vloc1_0x_r, pert_results[3].psi1)
    else
        0.0;
    log_dfpt("dfpt: D_elec_wf D(0x,0x)={e:.10} D(0x,1x)={e:.10}\n", .{ d_wf_00, d_wf_03 });
}

/// Run diagnostic tests on the DFPT perturbation results at Γ-point.
/// This includes:
/// 1. ASR check: Σ_J ψ^(1)_{n,Jβ} vs P_c(-iG_β ψ^(0)_n)
/// 2. V_nl^(1) commutator test
/// 3. D_elec comparison via wavefunction inner product
fn run_diagnostics(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    vloc1_gs: []const []math.Complex,
    n_atoms: usize,
) !void {
    try run_asr_diagnostics(alloc, gs, pert_results, n_atoms);
    try run_nonlocal_commutator_diagnostics(alloc, gs, n_atoms);
    try run_electronic_wavefunction_diagnostics(alloc, gs, pert_results, vloc1_gs);
}

// =====================================================================
// Perturbation parallelism
// =====================================================================

const GammaPertShared = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    gs: *const GroundState,
    dfpt_cfg: *const DfptConfig,
    pert_results: []PerturbationResult,
    vloc1_gs: [][]math.Complex,
    rho1_core_gs: [][]math.Complex,
    dim: usize,
    irr_pert_indices: ?[]const usize = null,

    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

const GammaPertWorker = struct {
    shared: *GammaPertShared,
    thread_index: usize,
};

fn set_gamma_pert_error(shared: *GammaPertShared, e: anyerror) void {
    shared.err_mutex.lockUncancelable(shared.io);
    defer shared.err_mutex.unlock(shared.io);

    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

fn gamma_pert_worker_fn(worker: *GammaPertWorker) void {
    const shared = worker.shared;
    const gs = shared.gs.*;
    const alloc = shared.alloc;

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const work_idx = shared.next_index.fetchAdd(1, .acq_rel);
        if (work_idx >= shared.dim) break;

        // Map work index to actual perturbation index
        const idx = if (shared.irr_pert_indices) |indices| indices[work_idx] else work_idx;
        const ia = idx / 3;
        const dir = idx % 3;
        const dir_names = [_][]const u8{ "x", "y", "z" };

        {
            shared.log_mutex.lockUncancelable(shared.io);
            defer shared.log_mutex.unlock(shared.io);

            log_dfpt(
                "dfpt: [thread {d}] solving perturbation atom={d} dir={s} ({d}/{d})\n",
                .{ worker.thread_index, ia, dir_names[dir], work_idx + 1, shared.dim },
            );
        }

        // Solve DFPT SCF (vloc1/rho1_core already built by caller)
        shared.pert_results[idx] = solve_perturbation(
            alloc,
            gs,
            ia,
            dir,
            shared.dfpt_cfg.*,
        ) catch |e| {
            set_gamma_pert_error(shared, e);
            shared.stop.store(1, .release);
            break;
        };
    }
}
