//! Density Functional Perturbation Theory (DFPT) module.
//!
//! This module provides the common types, ground-state preparation, and
//! re-exports for the DFPT phonon calculation subsystem.
//!
//! Submodules:
//! - `gamma`: Γ-point phonon calculation (q=0)
//! - `phonon_q`: Finite-q phonon band structure
//! - `dynmat_contrib`: Shared dynamical matrix contributions
//! - `perturbation`, `sternheimer`, `ewald2`, `dynmat`: Core DFPT building blocks

pub const perturbation = @import("perturbation.zig");
pub const sternheimer = @import("sternheimer.zig");
pub const ewald2 = @import("ewald2.zig");
pub const dynmat = @import("dynmat.zig");
pub const atomic_data = @import("atomic_data.zig");
pub const dynmat_contrib = @import("dynmat_contrib.zig");
pub const gamma = @import("gamma.zig");
pub const phonon_q = @import("phonon_q.zig");
pub const electric = @import("electric.zig");

const std = @import("std");
const math = @import("../math/math.zig");
const xc = @import("../xc/xc.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../scf/scf.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const config_mod = @import("../config/config.zig");
const iterative = @import("../linalg/iterative.zig");
const linalg = @import("../linalg/linalg.zig");
const runtime_logging = @import("../runtime/logging.zig");

const Grid = scf_mod.Grid;

// =====================================================================
// Re-exports for external callers (preserves dfpt.runPhonon etc.)
// =====================================================================

pub const ifc = @import("ifc.zig");
pub const runPhonon = gamma.runPhonon;
pub const PhononResult = gamma.PhononResult;
pub const runPhononBand = phonon_q.runPhononBand;
pub const runPhononBandIFC = phonon_q.runPhononBandIFC;
pub const PhononBandResult = phonon_q.PhononBandResult;

// =====================================================================
// Common types
// =====================================================================

/// Ground-state input data for DFPT calculation.
pub const GroundState = struct {
    /// Occupied wavefunctions: [n_occ][n_pw] in PW basis
    wavefunctions: []const []const math.Complex,
    /// Number of occupied bands
    n_occ: usize,
    /// Eigenvalues (Ry) for each band
    eigenvalues: []const f64,
    /// Ground-state electron density in real space
    rho_r: []const f64,
    /// XC kernel f_xc(r) precomputed on the real-space grid
    fxc_r: []const f64,
    /// Plane wave basis (G-vectors)
    gvecs: []const plane_wave.GVector,
    /// FFT grid
    grid: Grid,
    /// Species data
    species: []const hamiltonian.SpeciesEntry,
    /// Atom data
    atoms: []const hamiltonian.AtomData,
    /// Local pseudopotential selection for this run
    local_cfg: local_potential.LocalPotentialConfig,
    /// Form factor tables
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    /// Apply context for H₀ application
    apply_ctx: *scf_mod.ApplyContext,
    /// NLCC core charge density on real-space grid (null if no NLCC)
    rho_core: ?[]const f64 = null,
    /// NLCC core charge form factor tables (for ρ^(1)_core computation)
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable = null,
    /// V_xc(r) on real-space grid (for NLCC dynmat self-energy term)
    vxc_r: ?[]const f64 = null,
    /// XC functional used for this ground state
    xc_func: xc.Functional = .lda_pz,
    /// GGA kernel fields (null for LDA)
    fxc_ns_r: ?[]const f64 = null,
    fxc_ss_r: ?[]const f64 = null,
    v_sigma_r: ?[]const f64 = null,
    grad_n0_x: ?[]const f64 = null,
    grad_n0_y: ?[]const f64 = null,
    grad_n0_z: ?[]const f64 = null,
};

/// Owns all resources needed to build a GroundState at the Γ-point.
/// Call `deinit` when done to free all owned memory.
pub const PreparedGroundState = struct {
    gs: GroundState,
    // Owned resources
    basis: plane_wave.Basis,
    ionic: hamiltonian.PotentialGrid,
    local_r: []f64,
    apply_ctx_ptr: *scf_mod.ApplyContext,
    eig: linalg.EigenDecomp,
    wf_2d: [][]math.Complex,
    wf_const: [][]const math.Complex,
    rho_core: ?[]f64,
    rho_core_tables_buf: ?[]form_factor.RadialFormFactorTable,
    vxc_r: ?[]f64,
    fxc_r: []f64,
    fxc_ns_r: ?[]f64 = null,
    fxc_ss_r: ?[]f64 = null,
    v_sigma_r: ?[]f64 = null,
    grad_n0_x: ?[]f64 = null,
    grad_n0_y: ?[]f64 = null,
    grad_n0_z: ?[]f64 = null,
    alloc: std.mem.Allocator,

    pub fn deinit(self: *PreparedGroundState) void {
        const a = self.alloc;
        a.free(self.fxc_r);
        if (self.fxc_ns_r) |v| a.free(v);
        if (self.fxc_ss_r) |v| a.free(v);
        if (self.v_sigma_r) |v| a.free(v);
        if (self.grad_n0_x) |v| a.free(v);
        if (self.grad_n0_y) |v| a.free(v);
        if (self.grad_n0_z) |v| a.free(v);
        if (self.vxc_r) |v| a.free(v);
        if (self.rho_core_tables_buf) |buf| {
            for (buf) |*t| t.deinit(a);
            a.free(buf);
        }
        if (self.rho_core) |rc| a.free(rc);
        for (self.wf_2d) |wf| a.free(wf);
        a.free(self.wf_2d);
        a.free(self.wf_const);
        self.eig.deinit(a);
        self.apply_ctx_ptr.deinit(a);
        a.destroy(self.apply_ctx_ptr);
        a.free(self.local_r);
        self.ionic.deinit(a);
        self.basis.deinit(a);
    }
};

/// DFPT solver configuration extracted from the global config.
pub const DfptConfig = struct {
    sternheimer_tol: f64 = 1e-8,
    sternheimer_max_iter: usize = 200,
    scf_tol: f64 = 1e-10,
    scf_max_iter: usize = 50,
    mixing_beta: f64 = 0.3,
    alpha_shift: f64 = 0.01,
    pulay_history: usize = 8,
    pulay_start: usize = 4,
    kpoint_threads: usize = 0,
    perturbation_threads: usize = 1,
    log_level: runtime_logging.Level = .info,

    /// Build DfptConfig from the global config.
    pub fn fromConfig(cfg: config_mod.Config) DfptConfig {
        return .{
            .sternheimer_tol = cfg.dfpt.sternheimer_tol,
            .sternheimer_max_iter = cfg.dfpt.sternheimer_max_iter,
            .scf_tol = cfg.dfpt.scf_tol,
            .scf_max_iter = cfg.dfpt.scf_max_iter,
            .mixing_beta = cfg.dfpt.mixing_beta,
            .alpha_shift = cfg.dfpt.alpha_shift,
            .pulay_history = cfg.dfpt.pulay_history,
            .pulay_start = cfg.dfpt.pulay_start,
            .kpoint_threads = cfg.dfpt.kpoint_threads,
            .perturbation_threads = cfg.dfpt.perturbation_threads,
            .log_level = cfg.dfpt.log_level,
        };
    }
};

/// Ionic data extracted from species/atoms for dynmat construction.
pub const IonicData = struct {
    charges: []f64,
    positions: []math.Vec3,
    masses: []f64,

    pub fn init(alloc: std.mem.Allocator, species: []const hamiltonian.SpeciesEntry, atoms: []const hamiltonian.AtomData) !IonicData {
        const n = atoms.len;
        const charges = try alloc.alloc(f64, n);
        errdefer alloc.free(charges);
        const positions = try alloc.alloc(math.Vec3, n);
        errdefer alloc.free(positions);
        const masses = try alloc.alloc(f64, n);
        errdefer alloc.free(masses);
        for (0..n) |i| {
            charges[i] = species[atoms[i].species_index].z_valence;
            positions[i] = atoms[i].position;
            masses[i] = atomic_data.atomicMass(species[atoms[i].species_index].symbol);
        }
        return .{ .charges = charges, .positions = positions, .masses = masses };
    }

    pub fn deinit(self: IonicData, alloc: std.mem.Allocator) void {
        alloc.free(self.masses);
        alloc.free(self.positions);
        alloc.free(self.charges);
    }
};

/// Result of a single perturbation DFPT SCF.
pub const PerturbationResult = struct {
    /// First-order density response ρ^(1)(G) in reciprocal space
    rho1_g: []math.Complex,
    /// First-order wavefunctions ψ^(1)_n for each occupied band
    psi1: [][]math.Complex,

    pub fn deinit(self: *PerturbationResult, alloc: std.mem.Allocator) void {
        if (self.rho1_g.len > 0) alloc.free(self.rho1_g);
        for (self.psi1) |p| {
            if (p.len > 0) alloc.free(p);
        }
        if (self.psi1.len > 0) alloc.free(self.psi1);
    }
};

// =====================================================================
// Ground-state preparation
// =====================================================================

/// Build a PreparedGroundState at the Γ-point from converged SCF results.
/// This extracts the common ground-state setup shared by `runPhonon` and `runPhononBand`.
pub fn prepareGroundState(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    volume: f64,
    recip: math.Mat3,
) !PreparedGroundState {
    setDfptLogLevel(cfg.dfpt.log_level);
    const grid = scf_result.grid;
    const local_cfg = local_potential.resolve(cfg.scf.local_potential, cfg.ewald.alpha, grid.cell);

    // Compute number of occupied bands
    const nelec = totalElectrons(species, atoms);
    const n_occ = @as(usize, @intFromFloat(std.math.ceil(nelec / 2.0)));

    // Generate Γ-point PW basis
    const k_gamma = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, k_gamma);
    errdefer basis.deinit(alloc);
    const gvecs = basis.gvecs;
    const n_pw = gvecs.len;

    logDfptInfo("dfpt: nelec={d:.1} n_occ={d} n_pw={d}\n", .{ nelec, n_occ, n_pw });

    // Build ionic potential grid
    var ionic = try scf_mod.buildIonicPotentialGrid(alloc, grid, species, @constCast(atoms), local_cfg, null, null);
    errdefer ionic.deinit(alloc);

    // Build local_r = IFFT(ionic + SCF potential)
    const local_r = try scf_mod.buildLocalPotentialReal(alloc, grid, ionic, scf_result.potential);
    errdefer alloc.free(local_r);

    // Build ApplyContext for H₀ at Γ-point (heap-allocated for stable pointer)
    // Use multiple workspaces to support perturbation parallelism
    const pert_threads = perturbationThreadCount(3 * atoms.len, cfg.dfpt.perturbation_threads);
    const apply_ctx_ptr = try alloc.create(scf_mod.ApplyContext);
    errdefer alloc.destroy(apply_ctx_ptr);
    apply_ctx_ptr.* = try scf_mod.ApplyContext.initWithWorkspaces(
        alloc,
        io,
        grid,
        @constCast(gvecs),
        local_r,
        null,
        species,
        atoms,
        1.0 / volume,
        true,
        null,
        null,
        cfg.scf.fft_backend,
        pert_threads,
    );
    errdefer apply_ctx_ptr.deinit(alloc);

    // Solve eigenvalue problem at Γ-point using LOBPCG
    logDfptInfo("dfpt: solving Gamma-point eigenvalues with LOBPCG\n", .{});
    const nbands_solve = @max(n_occ + 2, @as(usize, 8));
    const diag = try alloc.alloc(f64, n_pw);
    defer alloc.free(diag);
    for (gvecs, 0..) |g, i| {
        diag[i] = g.kinetic;
    }

    const op = iterative.Operator{
        .n = n_pw,
        .ctx = @ptrCast(apply_ctx_ptr),
        .apply = &scf_mod.applyHamiltonian,
        .apply_batch = &scf_mod.applyHamiltonianBatched,
    };
    var eig = try iterative.hermitianEigenDecompIterative(
        alloc,
        cfg.linalg_backend,
        op,
        diag,
        nbands_solve,
        .{ .max_iter = 100, .tol = 1e-8 },
    );
    errdefer eig.deinit(alloc);

    logDfptDebug("dfpt: Gamma eigenvalues (Ry):", .{});
    for (0..@min(n_occ + 2, eig.values.len)) |i| {
        logDfptDebug(" {d:.6}", .{eig.values[i]});
    }
    logDfptDebug("\n", .{});

    // Extract occupied wavefunctions as 2D array
    const wf_const = try alloc.alloc([]const math.Complex, n_occ);
    errdefer alloc.free(wf_const);
    const wf_2d = try alloc.alloc([]math.Complex, n_occ);
    errdefer {
        for (wf_2d) |wf| alloc.free(wf);
        alloc.free(wf_2d);
    }
    for (0..n_occ) |n| {
        wf_2d[n] = try alloc.alloc(math.Complex, n_pw);
        @memcpy(wf_2d[n], eig.vectors[n * n_pw .. (n + 1) * n_pw]);
        wf_const[n] = wf_2d[n];
    }

    // Build NLCC core charge density on grid (if any species has NLCC)
    var rho_core: ?[]f64 = null;
    if (scf_mod.hasNlcc(species)) {
        rho_core = try scf_mod.buildCoreDensity(alloc, grid, species, @constCast(atoms));
        logDfptDebug("dfpt: NLCC core charge built on grid\n", .{});
    }
    errdefer if (rho_core) |rc| alloc.free(rc);

    // Build rho_core form factor tables
    var rho_core_tables_buf: ?[]form_factor.RadialFormFactorTable = null;
    var rho_core_tables: ?[]const form_factor.RadialFormFactorTable = null;
    if (rho_core != null) {
        const ff_q_max = @sqrt(cfg.scf.ecut_ry) * 2.0 + 10.0;
        const buf = try alloc.alloc(form_factor.RadialFormFactorTable, species.len);
        for (species, 0..) |sp, si| {
            buf[si] = try form_factor.RadialFormFactorTable.initRhoCore(alloc, sp.upf.*, ff_q_max);
        }
        rho_core_tables_buf = buf;
        rho_core_tables = buf;
    }
    errdefer if (rho_core_tables_buf) |buf| {
        for (buf) |*t| t.deinit(alloc);
        alloc.free(buf);
    };

    // Compute V_xc(r) for NLCC self-energy dynmat term
    // For PBE, must use computeXcFields to properly include gradient corrections
    var vxc_r: ?[]f64 = null;
    if (rho_core != null) {
        if (cfg.scf.xc == .pbe) {
            const fields = try scf_mod.computeXcFields(alloc, grid, scf_result.density, rho_core, false, .pbe);
            alloc.free(fields.exc);
            vxc_r = fields.vxc;
        } else {
            const total_grid = grid.count();
            vxc_r = try alloc.alloc(f64, total_grid);
            for (0..total_grid) |i| {
                const core = if (rho_core) |rc| rc[i] else 0.0;
                const eval = xc.evalPoint(cfg.scf.xc, scf_result.density[i] + core, 0.0);
                vxc_r.?[i] = eval.df_dn;
            }
        }
    }
    errdefer if (vxc_r) |v| alloc.free(v);

    // Build f_xc(r) grid (includes GGA kernel terms for PBE)
    var fxc_result = try buildFxcGrid(alloc, grid, scf_result.density, rho_core, cfg.scf.xc);
    errdefer fxc_result.deinit(alloc);

    const gs = GroundState{
        .wavefunctions = wf_const,
        .n_occ = n_occ,
        .eigenvalues = eig.values,
        .rho_r = scf_result.density,
        .fxc_r = fxc_result.fxc_r,
        .gvecs = gvecs,
        .grid = grid,
        .species = species,
        .atoms = atoms,
        .local_cfg = local_cfg,
        .ff_tables = null,
        .apply_ctx = apply_ctx_ptr,
        .rho_core = rho_core,
        .rho_core_tables = rho_core_tables,
        .vxc_r = vxc_r,
        .xc_func = cfg.scf.xc,
        .fxc_ns_r = fxc_result.fxc_ns_r,
        .fxc_ss_r = fxc_result.fxc_ss_r,
        .v_sigma_r = fxc_result.v_sigma_r,
        .grad_n0_x = fxc_result.grad_n0_x,
        .grad_n0_y = fxc_result.grad_n0_y,
        .grad_n0_z = fxc_result.grad_n0_z,
    };

    return PreparedGroundState{
        .gs = gs,
        .basis = basis,
        .ionic = ionic,
        .local_r = local_r,
        .apply_ctx_ptr = apply_ctx_ptr,
        .eig = eig,
        .wf_2d = wf_2d,
        .wf_const = wf_const,
        .rho_core = rho_core,
        .rho_core_tables_buf = rho_core_tables_buf,
        .vxc_r = vxc_r,
        .fxc_r = fxc_result.fxc_r,
        .fxc_ns_r = fxc_result.fxc_ns_r,
        .fxc_ss_r = fxc_result.fxc_ss_r,
        .v_sigma_r = fxc_result.v_sigma_r,
        .grad_n0_x = fxc_result.grad_n0_x,
        .grad_n0_y = fxc_result.grad_n0_y,
        .grad_n0_z = fxc_result.grad_n0_z,
        .alloc = alloc,
    };
}

// =====================================================================
// Utility functions
// =====================================================================

/// Compute the number of threads to use for perturbation parallelism.
/// total: number of perturbations (3 * n_atoms)
/// cfg_threads: user-configured perturbation_threads (0 = auto)
pub fn perturbationThreadCount(total: usize, cfg_threads: usize) usize {
    if (total <= 1) return 1;
    if (cfg_threads > 0) return @min(total, cfg_threads);
    const cpu_count = std.Thread.getCpuCount() catch 1;
    if (cpu_count == 0) return 1;
    return @min(total, cpu_count);
}

/// Compute how many k-point threads each perturbation worker should use.
/// Divides available CPU cores among perturbation workers so total parallelism
/// ≈ pert_threads × kpt_threads ≈ available CPUs.
pub fn kpointThreadsForPertParallel(pert_threads: usize, cfg_kpoint_threads: usize) usize {
    if (pert_threads <= 1) return cfg_kpoint_threads; // no perturbation parallelism, keep original
    const cpu_count = std.Thread.getCpuCount() catch 1;
    if (cpu_count <= pert_threads) return 1; // not enough CPUs for k-point parallelism
    return @max(1, cpu_count / pert_threads);
}

/// Result of building the XC kernel grid.
pub const FxcGridResult = struct {
    fxc_r: []f64,
    fxc_ns_r: ?[]f64 = null,
    fxc_ss_r: ?[]f64 = null,
    v_sigma_r: ?[]f64 = null,
    grad_n0_x: ?[]f64 = null,
    grad_n0_y: ?[]f64 = null,
    grad_n0_z: ?[]f64 = null,

    pub fn deinit(self: *FxcGridResult, a: std.mem.Allocator) void {
        a.free(self.fxc_r);
        if (self.fxc_ns_r) |v| a.free(v);
        if (self.fxc_ss_r) |v| a.free(v);
        if (self.v_sigma_r) |v| a.free(v);
        if (self.grad_n0_x) |v| a.free(v);
        if (self.grad_n0_y) |v| a.free(v);
        if (self.grad_n0_z) |v| a.free(v);
    }
};

/// Build f_xc(r) = d²E_xc/dρ² on the real-space grid.
/// When NLCC is present, f_xc must be evaluated at the total density (ρ_val + ρ_core).
/// For PBE, also computes gradient-dependent kernel terms and ∇n₀.
pub fn buildFxcGrid(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_r: []const f64,
    rho_core: ?[]const f64,
    xc_func: xc.Functional,
) !FxcGridResult {
    const total = grid.count();
    // Build total density
    const density = try alloc.alloc(f64, total);
    defer alloc.free(density);
    for (0..total) |i| {
        const core = if (rho_core) |rc| rc[i] else 0.0;
        density[i] = rho_r[i] + core;
    }

    const fxc = try alloc.alloc(f64, total);
    errdefer alloc.free(fxc);

    if (xc_func == .lda_pz) {
        for (0..total) |i| {
            const kernel = xc.evalKernel(.lda_pz, density[i], 0.0);
            fxc[i] = kernel.fxc;
        }
        return .{ .fxc_r = fxc };
    }

    // PBE: compute ∇n₀ and gradient-dependent kernel
    var grad = try scf_mod.gradientFromReal(alloc, grid, density, false);
    // Transfer ownership of gradient arrays to result
    errdefer grad.deinit(alloc);

    const fxc_ns = try alloc.alloc(f64, total);
    errdefer alloc.free(fxc_ns);
    const fxc_ss = try alloc.alloc(f64, total);
    errdefer alloc.free(fxc_ss);
    const v_sigma = try alloc.alloc(f64, total);
    errdefer alloc.free(v_sigma);

    for (0..total) |i| {
        const g2 = grad.x[i] * grad.x[i] + grad.y[i] * grad.y[i] + grad.z[i] * grad.z[i];
        const kernel = xc.evalKernel(.pbe, density[i], g2);
        fxc[i] = kernel.fxc;
        fxc_ns[i] = kernel.f_ns;
        fxc_ss[i] = kernel.f_ss;
        v_sigma[i] = kernel.v_s;
    }

    return .{
        .fxc_r = fxc,
        .fxc_ns_r = fxc_ns,
        .fxc_ss_r = fxc_ss,
        .v_sigma_r = v_sigma,
        .grad_n0_x = grad.x,
        .grad_n0_y = grad.y,
        .grad_n0_z = grad.z,
    };
}

/// Compute total number of valence electrons.
fn totalElectrons(species: []const hamiltonian.SpeciesEntry, atoms: []const hamiltonian.AtomData) f64 {
    var total: f64 = 0.0;
    for (atoms) |atom| {
        total += species[atom.species_index].z_valence;
    }
    return total;
}

/// Log output for DFPT diagnostics.
pub fn logDfpt(comptime fmt: []const u8, args: anytype) void {
    logDfptDebug(fmt, args);
}

var dfpt_log_level = std.atomic.Value(u8).init(@intFromEnum(runtime_logging.Level.info));

pub fn setDfptLogLevel(level: runtime_logging.Level) void {
    dfpt_log_level.store(@intFromEnum(level), .release);
}

pub fn getDfptLogLevel() runtime_logging.Level {
    return @enumFromInt(dfpt_log_level.load(.acquire));
}

pub fn logDfptInfo(comptime fmt: []const u8, args: anytype) void {
    runtime_logging.debugPrint(getDfptLogLevel(), .info, fmt, args);
}

pub fn logDfptDebug(comptime fmt: []const u8, args: anytype) void {
    runtime_logging.debugPrint(getDfptLogLevel(), .debug, fmt, args);
}

pub fn logDfptWarn(comptime fmt: []const u8, args: anytype) void {
    runtime_logging.debugPrint(getDfptLogLevel(), .warn, fmt, args);
}

test {
    _ = perturbation;
    _ = sternheimer;
    _ = ewald2;
    _ = dynmat;
    _ = atomic_data;
    _ = dynmat_contrib;
    _ = phonon_q;
    _ = ifc;
    _ = electric;
}
