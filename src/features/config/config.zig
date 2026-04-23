const std = @import("std");
const builtin = @import("builtin");
const fft_sizing = @import("../../lib/fft/sizing.zig");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const runtime_logging = @import("../runtime/logging.zig");
const xc = @import("../xc/xc.zig");

pub const BandPathPoint = struct {
    label: []u8,
    k: math.Vec3,
};

pub const BandConfig = struct {
    points_per_segment: usize,
    nbands: usize,
    path: []BandPathPoint, // resolved from path_string at band calc time
    path_string: ?[]u8 = null, // "auto" or "G-X-W-K-G-L" style string
    solver: BandSolver,
    iterative_max_iter: usize,
    iterative_tol: f64,
    iterative_max_subspace: usize,
    iterative_block_size: usize,
    iterative_init_diagonal: bool,
    kpoint_threads: usize,
    iterative_reuse_vectors: bool,
    use_symmetry: bool,
    lobpcg_parallel: bool, // Use parallel LOBPCG (default: true when kpoint_threads=1)
};

pub const ScfConfig = struct {
    enabled: bool,
    solver: ScfSolver,
    xc: xc.Functional,
    smearing: SmearingMethod,
    smear_ry: f64,
    ecut_ry: f64,
    kmesh: [3]usize,
    kmesh_shift: [3]f64,
    grid: [3]usize,
    grid_scale: f64,
    mixing_beta: f64,
    max_iter: usize,
    convergence: f64,
    convergence_metric: ConvergenceMetric,
    profile: bool,
    quiet: bool,
    debug_nonlocal: bool,
    debug_local: bool,
    debug_fermi: bool,
    enable_nonlocal: bool,
    local_potential: LocalPotentialMode,
    symmetry: bool,
    time_reversal: bool,
    kpoint_threads: usize,
    iterative_max_iter: usize,
    iterative_tol: f64,
    iterative_max_subspace: usize,
    iterative_block_size: usize,
    iterative_init_diagonal: bool,
    iterative_warmup_steps: usize,
    iterative_warmup_max_iter: usize,
    iterative_warmup_tol: f64,
    iterative_reuse_vectors: bool,
    kerker_q0: f64, // Kerker preconditioning parameter (0 = disabled)
    diemac: f64, // Macroscopic dielectric constant for model preconditioner
    // (1.0 = disabled, ~12 for Si, 1e6 for metals)
    dielng: f64, // Thomas-Fermi screening length in Bohr for model preconditioner (default 1.0)
    pulay_history: usize, // Pulay/DIIS mixing history (0 = disabled, use linear mixing)
    pulay_start: usize, // Number of simple mixing iterations before Pulay kicks in
    mixing_mode: MixingMode, // density or potential mixing
    use_rfft: bool, // Use Real FFT for electron density (~2x faster)
    fft_backend: FftBackend, // FFT implementation backend (zig or vdsp)
    nspin: usize, // 1 = spin-unpolarized, 2 = collinear spin-polarized
    spinat: ?[]f64, // Initial magnetic moment per atom (μ_B units), length = natom
    compute_stress: bool,
    reference_json: ?[]u8,
    compare_reference_json: ?[]u8,
    comparison_json: ?[]u8,
    compare_tolerance_json: ?[]u8,
};

pub const MixingMode = enum {
    density,
    potential,
};

pub const ScfSolver = enum {
    dense,
    iterative,
    cg, // Band-by-band conjugate gradient
    auto, // Automatically choose based on basis size
};

pub const BandSolver = enum {
    dense,
    iterative,
    cg, // Band-by-band conjugate gradient
    auto, // Automatically choose based on basis size
};

pub const SmearingMethod = enum {
    none,
    fermi_dirac,
};

pub const ConvergenceMetric = enum {
    density,
    potential,
};

pub const LocalPotentialMode = local_potential.LocalPotentialMode;

pub const BoundaryCondition = enum {
    periodic, // Standard periodic boundary conditions (crystals)
    isolated, // Isolated (molecular/cluster) boundary conditions with cutoff Coulomb
};

pub fn boundary_condition_name(bc: BoundaryCondition) []const u8 {
    return switch (bc) {
        .periodic => "periodic",
        .isolated => "isolated",
    };
}

pub const FftBackend = enum {
    zig, // Pure Zig FFT implementation (sequential)
    zig_parallel, // Pure Zig FFT with internal parallelization
    zig_transpose, // Transpose-based parallel FFT (best cache efficiency)
    zig_comptime24, // Comptime-optimized parallel FFT for 24×24×24 grids
    vdsp, // Apple Accelerate vDSP (macOS only)
    fftw, // FFTW3 (requires linking with -Dfftw-include and -Dfftw-lib)
    metal, // Apple Metal GPU (macOS only)
};

/// Parse FFT backend string.
pub fn parse_fft_backend(value: []const u8) !FftBackend {
    if (std.mem.eql(u8, value, "zig")) return .zig;
    if (std.mem.eql(u8, value, "zig_parallel")) return .zig_parallel;
    if (std.mem.eql(u8, value, "zig_transpose")) return .zig_transpose;
    if (std.mem.eql(u8, value, "zig_comptime24")) return .zig_comptime24;
    if (std.mem.eql(u8, value, "fftw")) return .fftw;
    if (std.mem.eql(u8, value, "vdsp")) {
        if (comptime builtin.os.tag != .macos) {
            return error.VdspNotAvailable;
        }
        return .vdsp;
    }
    if (std.mem.eql(u8, value, "metal")) {
        if (comptime builtin.os.tag != .macos) {
            return error.MetalNotAvailable;
        }
        return .metal;
    }
    return error.InvalidFftBackend;
}

/// Return FFT backend name.
pub fn fft_backend_name(backend: FftBackend) []const u8 {
    return switch (backend) {
        .zig => "zig",
        .zig_parallel => "zig_parallel",
        .zig_transpose => "zig_transpose",
        .zig_comptime24 => "zig_comptime24",
        .vdsp => "vdsp",
        .fftw => "fftw",
        .metal => "metal",
    };
}

/// Return solver name.
pub fn scf_solver_name(solver: ScfSolver) []const u8 {
    return switch (solver) {
        .dense => "dense",
        .iterative => "iterative",
        .cg => "cg",
        .auto => "auto",
    };
}

pub fn band_solver_name(solver: BandSolver) []const u8 {
    return switch (solver) {
        .dense => "dense",
        .iterative => "iterative",
        .cg => "cg",
        .auto => "auto",
    };
}

pub fn xc_functional_name(xc_func: xc.Functional) []const u8 {
    return xc.functional_name(xc_func);
}

pub fn smearing_name(method: SmearingMethod) []const u8 {
    return switch (method) {
        .none => "none",
        .fermi_dirac => "fermi_dirac",
    };
}

pub fn convergence_metric_name(metric: ConvergenceMetric) []const u8 {
    return switch (metric) {
        .density => "density",
        .potential => "potential",
    };
}

pub fn local_potential_mode_name(mode: LocalPotentialMode) []const u8 {
    return local_potential.name(mode);
}

pub const EwaldConfig = struct {
    alpha: f64,
    rcut: f64,
    gcut: f64,
    tol: f64,
};

pub const VdwMethod = enum {
    none,
    d3bj,
};

pub fn vdw_method_name(method: VdwMethod) []const u8 {
    return switch (method) {
        .none => "none",
        .d3bj => "d3bj",
    };
}

pub const VdwConfig = struct {
    enabled: bool = false,
    method: VdwMethod = .none,
    cutoff_radius: f64 = 95.0, // Bohr
    cn_cutoff: f64 = 40.0, // Bohr
    s6: ?f64 = null,
    s8: ?f64 = null,
    a1: ?f64 = null,
    a2: ?f64 = null,
};

pub const RelaxAlgorithm = enum {
    steepest_descent,
    cg,
    bfgs,
};

pub fn relax_algorithm_name(algo: RelaxAlgorithm) []const u8 {
    return switch (algo) {
        .steepest_descent => "steepest_descent",
        .cg => "cg",
        .bfgs => "bfgs",
    };
}

pub const DfptConfig = struct {
    enabled: bool,
    sternheimer_tol: f64,
    sternheimer_max_iter: usize,
    scf_tol: f64,
    scf_max_iter: usize,
    mixing_beta: f64,
    alpha_shift: f64,
    qpath_npoints: usize, // 0=Γ点のみ(既存), >0=バンド構造(各セグメントのq点数)
    pulay_history: usize, // Pulay/DIIS history depth (0 = linear mixing only)
    pulay_start: usize, // Simple mixing iterations before Pulay kicks in
    kpoint_threads: usize, // 0=auto, 1=serial, N=N threads for k-point parallelism
    perturbation_threads: usize, // 0=auto (use all CPUs), 1=serial (default), N=N threads
    qgrid: ?[3]usize, // IFC q-grid (null=direct DFPT, e.g. [1,1,8] for 1D systems)
    qpath: []BandPathPoint, // Custom q-path for phonon band (empty=FCC auto)
    dos_qmesh: ?[3]usize = null, // Phonon DOS q-mesh (e.g. [20,20,20])
    dos_sigma: f64 = 5.0, // Phonon DOS Gaussian width (cm⁻¹)
    dos_nbin: usize = 500, // Number of frequency bins for phonon DOS
    compute_dielectric: bool = false, // Compute dielectric tensor ε∞
    log_level: runtime_logging.Level = .info,
    // warn=warnings only, info=summary, debug=full diagnostics
};

pub const DosConfig = struct {
    enabled: bool = false,
    sigma: f64 = 0.01, // Gaussian width in Ry
    npoints: usize = 1001,
    emin: ?f64 = null,
    emax: ?f64 = null,
    pdos: bool = false, // Compute projected DOS (atom/orbital resolved)
};

pub const OutputConfig = struct {
    cube: bool = false,
};

pub const RelaxConfig = struct {
    enabled: bool,
    algorithm: RelaxAlgorithm,
    max_iter: usize,
    force_tol: f64, // Ry/Bohr
    max_step: f64, // Bohr
    output_trajectory: bool,
    cell_relax: bool = false, // vc-relax: optimize cell shape/volume
    stress_tol: f64 = 0.5, // GPa, convergence threshold for stress
    cell_step: f64 = 0.01, // Maximum fractional cell strain per step
    target_pressure: f64 = 0.0, // GPa, target external pressure (compensates Pulay stress)
};

pub const ValidationSeverity = enum {
    err,
    warning,
    hint, // recommendations for better performance/convergence
};

pub const ValidationIssue = struct {
    severity: ValidationSeverity,
    section: []const u8,
    field: []const u8,
    message: []const u8, // heap-allocated
};

pub const ValidationResult = struct {
    issues: []ValidationIssue,
    allocator: std.mem.Allocator,

    pub fn has_errors(self: ValidationResult) bool {
        for (self.issues) |issue| {
            if (issue.severity == .err) return true;
        }
        return false;
    }

    pub fn deinit(self: *ValidationResult) void {
        for (self.issues) |issue| {
            self.allocator.free(issue.message);
        }
        self.allocator.free(self.issues);
    }
};

pub const Config = struct {
    title: []u8,
    xyz_path: []u8,
    out_dir: []u8,
    units: math.Units,
    linalg_backend: linalg.Backend,
    threads: usize, // 0=auto, 1=serial, N=N threads
    cell: math.Mat3,
    boundary: BoundaryCondition = .periodic, // periodic (default) or isolated (molecular)
    scf: ScfConfig,
    ewald: EwaldConfig,
    vdw: VdwConfig,
    band: BandConfig,
    relax: RelaxConfig,
    dfpt: DfptConfig,
    dos: DosConfig,
    output: OutputConfig,
    pseudopotentials: []pseudo.Spec,

    /// Free owned strings and arrays.
    pub fn deinit(self: *Config, alloc: std.mem.Allocator) void {
        alloc.free(self.title);
        alloc.free(self.xyz_path);
        alloc.free(self.out_dir);
        if (self.scf.spinat) |sa| {
            alloc.free(sa);
        }
        if (self.scf.reference_json) |path| {
            alloc.free(path);
        }
        if (self.scf.compare_reference_json) |path| {
            alloc.free(path);
        }
        if (self.scf.comparison_json) |path| {
            alloc.free(path);
        }
        if (self.scf.compare_tolerance_json) |path| {
            alloc.free(path);
        }
        if (self.band.path_string) |s| alloc.free(s);
        for (self.dfpt.qpath) |p| {
            alloc.free(p.label);
        }
        if (self.dfpt.qpath.len > 0) {
            alloc.free(self.dfpt.qpath);
        }
        for (self.pseudopotentials) |p| {
            alloc.free(p.element);
            alloc.free(p.path);
        }
        if (self.pseudopotentials.len > 0) {
            alloc.free(self.pseudopotentials);
        }
    }

    const CellValidation = struct {
        cell_bohr: math.Mat3,
        volume: f64,
    };

    /// Validate config after parsing. Returns all issues (errors + warnings).
    pub fn validate(self: *const Config, alloc: std.mem.Allocator) !ValidationResult {
        var issues: std.ArrayList(ValidationIssue) = .empty;
        errdefer {
            for (issues.items) |issue| alloc.free(issue.message);
            issues.deinit(alloc);
        }

        const cell_validation = try self.validate_cell_geometry(alloc, &issues);
        try self.validate_pseudopotentials(alloc, &issues);
        try self.validate_scf_config(alloc, &issues);
        try self.validate_boundary_consistency(alloc, &issues);
        try self.validate_relax_config(alloc, &issues);
        try self.validate_dfpt_config(alloc, &issues);
        try self.validate_dos_config(alloc, &issues);
        try self.validate_vdw_config(alloc, &issues);
        try self.validate_band_config(alloc, &issues);
        try self.validate_spin_consistency(alloc, &issues);
        try self.validate_grid_recommendation(cell_validation, alloc, &issues);
        try self.add_validation_hints(alloc, &issues);

        return .{
            .issues = try issues.toOwnedSlice(alloc),
            .allocator = alloc,
        };
    }

    fn validate_cell_geometry(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !CellValidation {
        const cell_bohr = self.cell.scale(math.units_scale_to_bohr(self.units));
        const a1 = cell_bohr.row(0);
        const a2 = cell_bohr.row(1);
        const a3 = cell_bohr.row(2);
        const volume = a1.dot(a2.cross(a3));
        if (@abs(volume) <= 1e-12) {
            try add_issue_literal(
                alloc,
                issues,
                .err,
                "cell",
                "a1,a2,a3",
                "cell has zero or near-zero volume (degenerate lattice vectors)",
            );
        }
        return .{ .cell_bohr = cell_bohr, .volume = volume };
    }

    fn validate_pseudopotentials(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.pseudopotentials.len != 0) return;
        try add_issue_literal(
            alloc,
            issues,
            .err,
            "pseudopotential",
            "",
            "at least one pseudopotential must be specified",
        );
    }

    fn validate_scf_config(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        try self.validate_scf_basis(alloc, issues);
        try self.validate_scf_convergence_controls(alloc, issues);
        try self.validate_scf_mixing(alloc, issues);
        try self.validate_scf_smearing(alloc, issues);
    }

    fn validate_scf_basis(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.scf.ecut_ry <= 0) {
            try add_issue(
                alloc,
                issues,
                .err,
                "scf",
                "ecut_ry",
                try std.fmt.allocPrint(alloc, "must be positive (got {d:.2})", .{self.scf.ecut_ry}),
            );
        } else if (self.scf.ecut_ry < 5.0) {
            try add_issue(
                alloc,
                issues,
                .warning,
                "scf",
                "ecut_ry",
                try std.fmt.allocPrint(
                    alloc,
                    "{d:.1} is unusually low; typical range is 15-100 Ry",
                    .{self.scf.ecut_ry},
                ),
            );
        }

        try validate_positive_mesh(alloc, issues, "scf", "kmesh", self.scf.kmesh);
        if (self.scf.grid_scale > 0) return;
        try add_issue_literal(alloc, issues, .err, "scf", "grid_scale", "must be positive");
    }

    fn validate_scf_convergence_controls(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.scf.max_iter == 0) {
            try add_issue_literal(alloc, issues, .err, "scf", "max_iter", "must be positive");
        }

        if (self.scf.convergence <= 0) {
            try add_issue_literal(alloc, issues, .err, "scf", "convergence", "must be positive");
            return;
        }

        if (self.scf.convergence <= 1e-3) return;
        try add_issue(
            alloc,
            issues,
            .warning,
            "scf",
            "convergence",
            try std.fmt.allocPrint(
                alloc,
                "{e} is loose; results may be inaccurate",
                .{self.scf.convergence},
            ),
        );
    }

    fn validate_scf_mixing(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.scf.mixing_beta <= 0 or self.scf.mixing_beta > 1.0) {
            try add_issue(
                alloc,
                issues,
                .err,
                "scf",
                "mixing_beta",
                try std.fmt.allocPrint(
                    alloc,
                    "must be in (0, 1] (got {d:.3})",
                    .{self.scf.mixing_beta},
                ),
            );
        } else if (self.scf.mixing_beta > 0.7) {
            try add_issue(
                alloc,
                issues,
                .warning,
                "scf",
                "mixing_beta",
                try std.fmt.allocPrint(
                    alloc,
                    "{d:.2} is high; may cause SCF oscillation",
                    .{self.scf.mixing_beta},
                ),
            );
        }

        if (self.scf.pulay_history == 0 or self.scf.pulay_start <= self.scf.pulay_history) return;
        try add_issue(
            alloc,
            issues,
            .err,
            "scf",
            "pulay_start",
            try std.fmt.allocPrint(
                alloc,
                "({d}) exceeds pulay_history ({d})",
                .{ self.scf.pulay_start, self.scf.pulay_history },
            ),
        );
    }

    fn validate_scf_smearing(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.scf.smearing != .none and self.scf.smear_ry <= 0) {
            try add_issue_literal(
                alloc,
                issues,
                .err,
                "scf",
                "smear_ry",
                "must be > 0 when smearing is enabled",
            );
        }
        if (self.scf.smearing == .none and self.scf.smear_ry > 0) {
            try add_issue_literal(
                alloc,
                issues,
                .warning,
                "scf",
                "smear_ry",
                "is set but smearing is disabled; value will be ignored",
            );
        }
    }

    fn validate_boundary_consistency(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.boundary != .isolated) return;
        for (self.scf.kmesh) |value| {
            if (value <= 1) continue;
            try add_issue_literal(
                alloc,
                issues,
                .warning,
                "scf",
                "kmesh",
                "isolated boundary with kmesh > [1,1,1]; only Gamma point is meaningful",
            );
            return;
        }
    }

    fn validate_relax_config(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (!self.relax.enabled) return;
        if (!self.scf.enabled) {
            try add_issue_literal(alloc, issues, .err, "relax", "", "requires scf to be enabled");
        }
        if (self.relax.max_iter == 0) {
            try add_issue_literal(alloc, issues, .err, "relax", "max_iter", "must be positive");
        }
        if (self.relax.force_tol <= 0) {
            try add_issue_literal(alloc, issues, .err, "relax", "force_tol", "must be positive");
        }
        if (self.relax.max_step <= 0) {
            try add_issue_literal(alloc, issues, .err, "relax", "max_step", "must be positive");
        }
        if (self.relax.cell_relax and !self.scf.compute_stress) {
            try add_issue_literal(
                alloc,
                issues,
                .warning,
                "relax",
                "cell_relax",
                "cell_relax is enabled but compute_stress is off; stress will be auto-enabled",
            );
        }
    }

    fn validate_dfpt_config(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (!self.dfpt.enabled) return;
        try self.validate_dfpt_requirements(alloc, issues);
        try self.validate_dfpt_controls(alloc, issues);
        try self.validate_dfpt_meshes(alloc, issues);
    }

    fn validate_dfpt_requirements(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (!self.scf.enabled) {
            try add_issue_literal(alloc, issues, .err, "dfpt", "", "requires scf to be enabled");
        }
        if (self.scf.smearing != .none) {
            try add_issue_literal(
                alloc,
                issues,
                .err,
                "dfpt",
                "",
                "DFPT is incompatible with Fermi-Dirac smearing",
            );
        }
        if (self.scf.nspin == 2) {
            try add_issue_literal(
                alloc,
                issues,
                .err,
                "dfpt",
                "",
                "spin-polarized DFPT (nspin=2) is not supported",
            );
        }
    }

    fn validate_dfpt_controls(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.dfpt.sternheimer_max_iter == 0) {
            try add_issue_literal(
                alloc,
                issues,
                .err,
                "dfpt",
                "sternheimer_max_iter",
                "must be positive",
            );
        }
        if (self.dfpt.sternheimer_tol <= 0) {
            try add_issue_literal(
                alloc,
                issues,
                .err,
                "dfpt",
                "sternheimer_tol",
                "must be positive",
            );
        }
        if (self.dfpt.scf_max_iter == 0) {
            try add_issue_literal(alloc, issues, .err, "dfpt", "scf_max_iter", "must be positive");
        }
        if (self.dfpt.scf_tol <= 0) {
            try add_issue_literal(alloc, issues, .err, "dfpt", "scf_tol", "must be positive");
        }
        if (self.dfpt.mixing_beta <= 0 or self.dfpt.mixing_beta > 1.0) {
            try add_issue(
                alloc,
                issues,
                .err,
                "dfpt",
                "mixing_beta",
                try std.fmt.allocPrint(
                    alloc,
                    "must be in (0, 1] (got {d:.3})",
                    .{self.dfpt.mixing_beta},
                ),
            );
        }
        if (self.dfpt.pulay_history == 0 or self.dfpt.pulay_start <= self.dfpt.pulay_history) {
            return;
        }
        try add_issue(
            alloc,
            issues,
            .err,
            "dfpt",
            "pulay_start",
            try std.fmt.allocPrint(
                alloc,
                "({d}) exceeds pulay_history ({d})",
                .{ self.dfpt.pulay_start, self.dfpt.pulay_history },
            ),
        );
    }

    fn validate_dfpt_meshes(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.dfpt.qgrid) |qgrid| {
            try validate_positive_mesh(alloc, issues, "dfpt", "qgrid", qgrid);
        }
        if (self.dfpt.dos_qmesh) |dos_qmesh| {
            try validate_positive_mesh(alloc, issues, "dfpt", "dos_qmesh", dos_qmesh);
        }
    }

    fn validate_dos_config(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (!self.dos.enabled) return;
        if (self.dos.sigma <= 0) {
            try add_issue_literal(alloc, issues, .err, "dos", "sigma", "must be positive");
        }
        if (self.dos.npoints == 0) {
            try add_issue_literal(alloc, issues, .err, "dos", "npoints", "must be positive");
        }
        if (self.dos.emin == null or self.dos.emax == null) return;
        if (self.dos.emin.? < self.dos.emax.?) return;
        try add_issue_literal(
            alloc,
            issues,
            .err,
            "dos",
            "emin/emax",
            "emin must be less than emax",
        );
    }

    fn validate_vdw_config(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (!self.vdw.enabled or self.vdw.method != .none) return;
        try add_issue_literal(
            alloc,
            issues,
            .err,
            "vdw",
            "method",
            "vdw enabled but method = \"none\"; set method = \"d3bj\"",
        );
    }

    fn validate_band_config(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.band.path.len == 0 and self.band.path_string == null) return;
        if (self.band.points_per_segment > 0) return;
        try add_issue_literal(alloc, issues, .err, "band", "points", "points must be positive");
    }

    fn validate_spin_consistency(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.scf.nspin != 2 or self.scf.spinat != null) return;
        try add_issue_literal(
            alloc,
            issues,
            .warning,
            "scf",
            "spinat",
            "nspin=2 but spinat not set; calculation may converge to non-magnetic solution",
        );
    }

    fn validate_grid_recommendation(
        self: *const Config,
        cell_validation: CellValidation,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.scf.grid[0] == 0 or self.scf.grid[1] == 0 or self.scf.grid[2] == 0) return;
        if (self.scf.ecut_ry <= 0 or @abs(cell_validation.volume) <= 1e-12) return;

        const recip = math.reciprocal(cell_validation.cell_bohr);
        const scale = if (self.scf.grid_scale > 0) self.scf.grid_scale else 1.0;
        const density_gmax = @max(2.0, scale) * @sqrt(self.scf.ecut_ry);
        const raw1 = @as(usize, @intFromFloat(
            std.math.ceil(density_gmax / math.Vec3.norm(recip.row(0))),
        )) * 2 + 1;
        const raw2 = @as(usize, @intFromFloat(
            std.math.ceil(density_gmax / math.Vec3.norm(recip.row(1))),
        )) * 2 + 1;
        const raw3 = @as(usize, @intFromFloat(
            std.math.ceil(density_gmax / math.Vec3.norm(recip.row(2))),
        )) * 2 + 1;
        if (self.scf.grid[0] >= raw1 and self.scf.grid[1] >= raw2 and self.scf.grid[2] >= raw3) {
            return;
        }

        try add_issue(
            alloc,
            issues,
            .warning,
            "scf",
            "grid",
            try std.fmt.allocPrint(
                alloc,
                "grid [{},{},{}] is smaller than recommended [{},{},{}]" ++
                    " for ecut_ry={d:.1}; use grid = [{},{},{}]" ++
                    " or set grid = [0,0,0] for auto",
                .{
                    self.scf.grid[0],
                    self.scf.grid[1],
                    self.scf.grid[2],
                    raw1,
                    raw2,
                    raw3,
                    self.scf.ecut_ry,
                    fft_sizing.next_fft_size(@max(raw1, 3)),
                    fft_sizing.next_fft_size(@max(raw2, 3)),
                    fft_sizing.next_fft_size(@max(raw3, 3)),
                },
            ),
        );
    }

    fn add_validation_hints(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        try self.add_scf_validation_hints(alloc, issues);
        try self.add_band_validation_hints(alloc, issues);
    }

    fn add_scf_validation_hints(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (!self.scf.enabled) return;
        if (self.scf.solver == .dense) {
            try add_issue_literal(
                alloc,
                issues,
                .hint,
                "scf",
                "solver",
                "solver = \"dense\" is slow for large systems;" ++
                    " consider solver = \"iterative\" (LOBPCG)",
            );
        }
        if (self.scf.fft_backend != .fftw) {
            try add_issue_literal(
                alloc,
                issues,
                .hint,
                "scf",
                "fft_backend",
                "fft_backend = \"fftw\" is recommended for production calculations",
            );
        }
        if (self.scf.diemac == 1.0 and self.scf.convergence < 1e-6) {
            try add_issue_literal(
                alloc,
                issues,
                .hint,
                "scf",
                "diemac",
                "diemac = 1.0 (disabled); for tight convergence (<1e-6)," ++
                    " set ~12 for semiconductors or ~1e6 for metals",
            );
        }
        if (self.scf.pulay_history == 0) {
            try add_issue_literal(
                alloc,
                issues,
                .hint,
                "scf",
                "pulay_history",
                "pulay_history = 0 (DIIS disabled); pulay_history = 8 with pulay_start = 4" ++
                    " typically improves SCF convergence",
            );
        }
        if (self.scf.mixing_mode == .density) {
            try add_issue_literal(
                alloc,
                issues,
                .hint,
                "scf",
                "mixing_mode",
                "mixing_mode = \"density\" can oscillate at large ecut;" ++
                    " mixing_mode = \"potential\" is recommended",
            );
        }
        if (self.scf.solver == .iterative and self.scf.iterative_tol < 1e-6) {
            try add_issue_literal(
                alloc,
                issues,
                .hint,
                "scf",
                "iterative_tol",
                "iterative_tol < 1e-6 increases eigensolver cost per SCF iteration;" ++
                    " 1e-4 is usually sufficient",
            );
        }
        if (!self.scf.symmetry) {
            try add_issue_literal(
                alloc,
                issues,
                .hint,
                "scf",
                "symmetry",
                "symmetry = false; enabling symmetry reduces k-points" ++
                    " and speeds up the calculation",
            );
        }
    }

    fn add_band_validation_hints(
        self: *const Config,
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
    ) !void {
        if (self.band.path.len == 0 and self.band.path_string == null) return;
        if (self.band.solver != .dense) return;
        try add_issue_literal(
            alloc,
            issues,
            .hint,
            "band",
            "solver",
            "solver = \"dense\" is slow for band structure;" ++
                " consider solver = \"iterative\"",
        );
    }

    fn validate_positive_mesh(
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
        section: []const u8,
        field: []const u8,
        mesh: [3]usize,
    ) !void {
        for (mesh, 0..) |value, i| {
            if (value != 0) continue;
            try add_issue(
                alloc,
                issues,
                .err,
                section,
                field,
                try std.fmt.allocPrint(alloc, "dimension [{d}] must be positive", .{i}),
            );
        }
    }

    /// Append a validation issue. Takes ownership of `message` (must be heap-allocated).
    /// Frees `message` if append fails.
    fn add_issue(
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
        severity: ValidationSeverity,
        section: []const u8,
        field: []const u8,
        message: []const u8,
    ) !void {
        errdefer alloc.free(message);
        try issues.append(alloc, .{
            .severity = severity,
            .section = section,
            .field = field,
            .message = message,
        });
    }

    /// Convenience: add_issue with a comptime string literal (dupes it).
    fn add_issue_literal(
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
        severity: ValidationSeverity,
        section: []const u8,
        field: []const u8,
        comptime message: []const u8,
    ) !void {
        const owned = try alloc.dupe(u8, message);
        errdefer alloc.free(owned);
        try add_issue(alloc, issues, severity, section, field, owned);
    }
};

const Section = enum {
    root,
    cell,
    scf,
    ewald,
    vdw,
    band,
    pseudopotential,
    relax,
    dfpt,
    dfpt_qpath,
    dos,
    output,
};

const default_load_scf = ScfConfig{
    .enabled = false,
    .solver = .dense,
    .xc = .lda_pz,
    .smearing = .none,
    .smear_ry = 0.0,
    .ecut_ry = 30.0,
    .kmesh = .{ 4, 4, 4 },
    .kmesh_shift = .{ 0.0, 0.0, 0.0 },
    .grid = .{ 0, 0, 0 },
    .grid_scale = 1.0,
    .mixing_beta = 0.3,
    .max_iter = 50,
    .convergence = 1e-6,
    .convergence_metric = .density,
    .profile = false,
    .quiet = false,
    .debug_nonlocal = false,
    .debug_local = false,
    .debug_fermi = false,
    .enable_nonlocal = true,
    .local_potential = .short_range,
    .symmetry = true,
    .time_reversal = true,
    .kpoint_threads = 0,
    .iterative_max_iter = 20,
    .iterative_tol = 1e-4,
    .iterative_max_subspace = 0,
    .iterative_block_size = 0,
    .iterative_init_diagonal = false,
    .iterative_warmup_steps = 2,
    .iterative_warmup_max_iter = 10,
    .iterative_warmup_tol = 1e-3,
    .iterative_reuse_vectors = true,
    .kerker_q0 = 0.0,
    .diemac = 1.0,
    .dielng = 1.0,
    .pulay_history = 8,
    .pulay_start = 4,
    .mixing_mode = .potential,
    .use_rfft = false,
    .fft_backend = .fftw,
    .nspin = 1,
    .spinat = null,
    .compute_stress = false,
    .reference_json = null,
    .compare_reference_json = null,
    .comparison_json = null,
    .compare_tolerance_json = null,
};

const default_load_ewald = EwaldConfig{
    .alpha = 0.0,
    .rcut = 0.0,
    .gcut = 0.0,
    .tol = 1e-8,
};

const default_load_band = BandConfig{
    .points_per_segment = 60,
    .nbands = 8,
    .path = &.{},
    .path_string = null,
    .solver = .dense,
    .iterative_max_iter = 40,
    .iterative_tol = 1e-6,
    .iterative_max_subspace = 0,
    .iterative_block_size = 0,
    .iterative_init_diagonal = false,
    .kpoint_threads = 0,
    .iterative_reuse_vectors = true,
    .use_symmetry = false,
    .lobpcg_parallel = false,
};

const default_load_relax = RelaxConfig{
    .enabled = false,
    .algorithm = .bfgs,
    .max_iter = 100,
    .force_tol = 1e-4,
    .max_step = 0.5,
    .output_trajectory = false,
};

const default_load_dfpt = DfptConfig{
    .enabled = false,
    .sternheimer_tol = 1e-8,
    .sternheimer_max_iter = 200,
    .scf_tol = 1e-10,
    .scf_max_iter = 50,
    .mixing_beta = 0.3,
    .alpha_shift = 0.01,
    .qpath_npoints = 0,
    .pulay_history = 8,
    .pulay_start = 4,
    .kpoint_threads = 0,
    .perturbation_threads = 1,
    .qgrid = null,
    .qpath = &.{},
};

const LoadState = struct {
    title: []u8,
    xyz_path: ?[]u8,
    out_dir: []u8,
    units: math.Units,
    linalg_backend: linalg.Backend,
    boundary: BoundaryCondition,
    a1: ?math.Vec3,
    a2: ?math.Vec3,
    a3: ?math.Vec3,
    scf: ScfConfig,
    ewald: EwaldConfig,
    vdw: VdwConfig,
    band: BandConfig,
    relax: RelaxConfig,
    dfpt: DfptConfig,
    dos: DosConfig,
    output: OutputConfig,
    top_threads: usize,
    scf_kpoint_threads_explicit: bool,
    band_kpoint_threads_explicit: bool,
    dfpt_kpoint_threads_explicit: bool,
    band_lobpcg_parallel_explicit: bool,
    current_section: Section,
    current_pseudo_index: ?usize,
    current_dfpt_qpath_index: ?usize,
    dfpt_qpath_list: std.ArrayList(BandPathPoint),
    pseudo_list: std.ArrayList(pseudo.Spec),

    fn init(alloc: std.mem.Allocator) !LoadState {
        const title = try alloc.dupe(u8, "dft_zig");
        errdefer alloc.free(title);

        const out_dir = try alloc.dupe(u8, "out");
        errdefer alloc.free(out_dir);

        return .{
            .title = title,
            .xyz_path = null,
            .out_dir = out_dir,
            .units = .angstrom,
            .linalg_backend = if (builtin.os.tag == .macos) .accelerate else .openblas,
            .boundary = .periodic,
            .a1 = null,
            .a2 = null,
            .a3 = null,
            .scf = default_load_scf,
            .ewald = default_load_ewald,
            .vdw = .{},
            .band = default_load_band,
            .relax = default_load_relax,
            .dfpt = default_load_dfpt,
            .dos = .{},
            .output = .{},
            .top_threads = 0,
            .scf_kpoint_threads_explicit = false,
            .band_kpoint_threads_explicit = false,
            .dfpt_kpoint_threads_explicit = false,
            .band_lobpcg_parallel_explicit = false,
            .current_section = .root,
            .current_pseudo_index = null,
            .current_dfpt_qpath_index = null,
            .dfpt_qpath_list = .empty,
            .pseudo_list = .empty,
        };
    }

    fn deinit(self: *LoadState, alloc: std.mem.Allocator) void {
        alloc.free(self.title);
        if (self.xyz_path) |path| alloc.free(path);
        alloc.free(self.out_dir);
        self.deinit_scf_owned_fields(alloc);
        if (self.band.path_string) |path| alloc.free(path);
        self.deinit_dfpt_qpath_list(alloc);
        self.deinit_pseudo_list(alloc);
    }

    fn deinit_scf_owned_fields(self: *LoadState, alloc: std.mem.Allocator) void {
        if (self.scf.spinat) |spinat| alloc.free(spinat);
        if (self.scf.reference_json) |path| alloc.free(path);
        if (self.scf.compare_reference_json) |path| alloc.free(path);
        if (self.scf.comparison_json) |path| alloc.free(path);
        if (self.scf.compare_tolerance_json) |path| alloc.free(path);
    }

    fn deinit_dfpt_qpath_list(self: *LoadState, alloc: std.mem.Allocator) void {
        for (self.dfpt_qpath_list.items) |point| {
            alloc.free(point.label);
        }
        self.dfpt_qpath_list.deinit(alloc);
    }

    fn deinit_pseudo_list(self: *LoadState, alloc: std.mem.Allocator) void {
        for (self.pseudo_list.items) |spec| {
            alloc.free(spec.element);
            alloc.free(spec.path);
        }
        self.pseudo_list.deinit(alloc);
    }

    fn enter_section(self: *LoadState, section: Section) void {
        self.current_section = section;
        self.current_pseudo_index = null;
        self.current_dfpt_qpath_index = null;
    }

    fn begin_dfpt_qpath(self: *LoadState, alloc: std.mem.Allocator) !void {
        const label = try alloc.dupe(u8, "");
        errdefer alloc.free(label);

        try self.dfpt_qpath_list.append(alloc, .{
            .label = label,
            .k = .{ .x = 0, .y = 0, .z = 0 },
        });
        self.current_section = .dfpt_qpath;
        self.current_dfpt_qpath_index = self.dfpt_qpath_list.items.len - 1;
        self.current_pseudo_index = null;
    }

    fn begin_pseudopotential(self: *LoadState, alloc: std.mem.Allocator) !void {
        const element = try alloc.dupe(u8, "");
        errdefer alloc.free(element);

        const path = try alloc.dupe(u8, "");
        errdefer alloc.free(path);

        try self.pseudo_list.append(alloc, .{
            .element = element,
            .path = path,
            .format = .upf,
        });
        self.current_section = .pseudopotential;
        self.current_pseudo_index = self.pseudo_list.items.len - 1;
        self.current_dfpt_qpath_index = null;
    }

    fn parse_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        switch (self.current_section) {
            .root => try self.parse_root_field(alloc, key, value),
            .cell => try self.parse_cell_field(key, value),
            .scf => try self.parse_scf_field(alloc, key, value),
            .ewald => try self.parse_ewald_field(key, value),
            .vdw => try self.parse_vdw_field(alloc, key, value),
            .band => try self.parse_band_field(alloc, key, value),
            .pseudopotential => try self.parse_pseudopotential_field(alloc, key, value),
            .relax => try self.parse_relax_field(alloc, key, value),
            .dfpt => try self.parse_dfpt_field(alloc, key, value),
            .dfpt_qpath => try self.parse_dfpt_qpath_field(alloc, key, value),
            .dos => try self.parse_dos_field(key, value),
            .output => try self.parse_output_field(key, value),
        }
    }

    fn parse_root_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        if (std.mem.eql(u8, key, "title")) {
            return try replace_owned_string(alloc, &self.title, value);
        }
        if (std.mem.eql(u8, key, "xyz")) {
            return try replace_optional_owned_string(alloc, &self.xyz_path, value);
        }
        if (std.mem.eql(u8, key, "out_dir")) {
            return try replace_owned_string(alloc, &self.out_dir, value);
        }
        if (std.mem.eql(u8, key, "units")) {
            const unit_str = try parse_string(alloc, value);
            defer alloc.free(unit_str);

            if (std.mem.eql(u8, unit_str, "angstrom")) {
                self.units = .angstrom;
                return;
            }
            if (std.mem.eql(u8, unit_str, "bohr")) {
                self.units = .bohr;
                return;
            }
            return error.InvalidUnits;
        }
        if (std.mem.eql(u8, key, "linalg_backend")) {
            const backend_str = try parse_string(alloc, value);
            defer alloc.free(backend_str);

            self.linalg_backend = try linalg.parse_backend(backend_str);
            return;
        }
        if (std.mem.eql(u8, key, "threads")) {
            self.top_threads = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "boundary")) {
            const boundary_str = try parse_string(alloc, value);
            defer alloc.free(boundary_str);

            if (std.mem.eql(u8, boundary_str, "periodic")) {
                self.boundary = .periodic;
                return;
            }
            if (std.mem.eql(u8, boundary_str, "isolated")) {
                self.boundary = .isolated;
                return;
            }
        }
        return error.UnsupportedToml;
    }

    fn parse_cell_field(self: *LoadState, key: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, key, "a1")) {
            self.a1 = try parse_vec3(value);
            return;
        }
        if (std.mem.eql(u8, key, "a2")) {
            self.a2 = try parse_vec3(value);
            return;
        }
        if (std.mem.eql(u8, key, "a3")) {
            self.a3 = try parse_vec3(value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_scf_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        if (try self.parse_scf_grid_field(key, value)) return;
        if (try self.parse_scf_enum_field(alloc, key, value)) return;
        if (try self.parse_scf_bool_field(key, value)) return;
        if (try self.parse_scf_numeric_field(key, value)) return;
        if (try self.parse_scf_owned_field(alloc, key, value)) return;
        return error.UnsupportedToml;
    }

    fn parse_scf_grid_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "ecut_ry")) {
            self.scf.ecut_ry = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "kmesh")) {
            const mesh = try parse_array_numbers(value, 3);
            self.scf.kmesh = .{
                try float_to_index(mesh[0]),
                try float_to_index(mesh[1]),
                try float_to_index(mesh[2]),
            };
            return true;
        }
        if (std.mem.eql(u8, key, "kmesh_shift")) {
            const shift = try parse_array_numbers(value, 3);
            self.scf.kmesh_shift = .{ shift[0], shift[1], shift[2] };
            return true;
        }
        if (std.mem.eql(u8, key, "grid")) {
            const grid = try parse_array_numbers(value, 3);
            self.scf.grid = .{
                try float_to_index(grid[0]),
                try float_to_index(grid[1]),
                try float_to_index(grid[2]),
            };
            return true;
        }
        if (std.mem.eql(u8, key, "grid_scale")) {
            self.scf.grid_scale = try parse_float(value);
            return true;
        }
        return false;
    }

    fn parse_scf_enum_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !bool {
        if (std.mem.eql(u8, key, "solver")) {
            const solver_str = try parse_string(alloc, value);
            defer alloc.free(solver_str);

            self.scf.solver = try parse_scf_solver(solver_str);
            return true;
        }
        if (std.mem.eql(u8, key, "xc")) {
            const xc_str = try parse_string(alloc, value);
            defer alloc.free(xc_str);

            self.scf.xc = try parse_xc_functional(xc_str);
            return true;
        }
        if (std.mem.eql(u8, key, "smearing")) {
            const smearing_str = try parse_string(alloc, value);
            defer alloc.free(smearing_str);

            self.scf.smearing = try parse_smearing_method(smearing_str);
            return true;
        }
        if (std.mem.eql(u8, key, "convergence_metric")) {
            const metric_str = try parse_string(alloc, value);
            defer alloc.free(metric_str);

            self.scf.convergence_metric = try parse_convergence_metric(metric_str);
            return true;
        }
        if (std.mem.eql(u8, key, "local_potential")) {
            const mode_str = try parse_string(alloc, value);
            defer alloc.free(mode_str);

            self.scf.local_potential = try parse_local_potential_mode(mode_str);
            return true;
        }
        if (std.mem.eql(u8, key, "mixing_mode")) {
            const trimmed = trim_toml_value(value);
            if (std.mem.eql(u8, trimmed, "density")) {
                self.scf.mixing_mode = .density;
                return true;
            }
            if (std.mem.eql(u8, trimmed, "potential")) {
                self.scf.mixing_mode = .potential;
                return true;
            }
        }
        if (std.mem.eql(u8, key, "fft_backend")) {
            self.scf.fft_backend = parse_fft_backend(trim_toml_value(value)) catch {
                return error.InvalidFftBackend;
            };
            return true;
        }
        return false;
    }

    fn parse_scf_bool_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "enabled")) {
            self.scf.enabled = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "profile")) {
            self.scf.profile = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "quiet")) {
            self.scf.quiet = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "debug_nonlocal")) {
            self.scf.debug_nonlocal = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "debug_local")) {
            self.scf.debug_local = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "debug_fermi")) {
            self.scf.debug_fermi = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "enable_nonlocal")) {
            self.scf.enable_nonlocal = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "symmetry")) {
            self.scf.symmetry = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "time_reversal")) {
            self.scf.time_reversal = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_init_diagonal")) {
            self.scf.iterative_init_diagonal = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_reuse_vectors")) {
            self.scf.iterative_reuse_vectors = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "use_rfft")) {
            self.scf.use_rfft = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "compute_stress")) {
            self.scf.compute_stress = try parse_bool(value);
            return true;
        }
        return false;
    }

    fn parse_scf_numeric_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "smear_ry")) {
            self.scf.smear_ry = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "mixing_beta")) {
            self.scf.mixing_beta = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "max_iter")) {
            self.scf.max_iter = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "convergence")) {
            self.scf.convergence = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "kpoint_threads")) {
            self.scf.kpoint_threads = try float_to_index(try parse_float(value));
            self.scf_kpoint_threads_explicit = true;
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_max_iter")) {
            self.scf.iterative_max_iter = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_tol")) {
            self.scf.iterative_tol = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_max_subspace")) {
            self.scf.iterative_max_subspace = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_block_size")) {
            self.scf.iterative_block_size = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_warmup_steps")) {
            self.scf.iterative_warmup_steps = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_warmup_max_iter")) {
            self.scf.iterative_warmup_max_iter = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_warmup_tol")) {
            self.scf.iterative_warmup_tol = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "kerker_q0")) {
            self.scf.kerker_q0 = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "diemac")) {
            const diemac = try parse_float(value);
            if (diemac < 1.0) return error.InvalidConfig;
            self.scf.diemac = diemac;
            return true;
        }
        if (std.mem.eql(u8, key, "dielng")) {
            const dielng = try parse_float(value);
            if (dielng <= 0.0) return error.InvalidConfig;
            self.scf.dielng = dielng;
            return true;
        }
        if (std.mem.eql(u8, key, "pulay_history")) {
            self.scf.pulay_history = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "pulay_start")) {
            self.scf.pulay_start = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "nspin")) {
            self.scf.nspin = try float_to_index(try parse_float(value));
            if (self.scf.nspin != 1 and self.scf.nspin != 2) return error.InvalidNspin;
            return true;
        }
        return false;
    }

    fn parse_scf_owned_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !bool {
        if (std.mem.eql(u8, key, "spinat")) {
            try replace_optional_float_array(alloc, &self.scf.spinat, value);
            return true;
        }
        if (std.mem.eql(u8, key, "reference_json")) {
            try replace_optional_owned_string(alloc, &self.scf.reference_json, value);
            return true;
        }
        if (std.mem.eql(u8, key, "compare_reference_json")) {
            try replace_optional_owned_string(alloc, &self.scf.compare_reference_json, value);
            return true;
        }
        if (std.mem.eql(u8, key, "comparison_json")) {
            try replace_optional_owned_string(alloc, &self.scf.comparison_json, value);
            return true;
        }
        if (std.mem.eql(u8, key, "compare_tolerance_json")) {
            try replace_optional_owned_string(alloc, &self.scf.compare_tolerance_json, value);
            return true;
        }
        return false;
    }

    fn parse_ewald_field(self: *LoadState, key: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, key, "alpha")) {
            self.ewald.alpha = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "rcut")) {
            self.ewald.rcut = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "gcut")) {
            self.ewald.gcut = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "tol")) {
            self.ewald.tol = try parse_float(value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_vdw_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        if (std.mem.eql(u8, key, "enabled")) {
            self.vdw.enabled = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "method")) {
            const method_str = try parse_string(alloc, value);
            defer alloc.free(method_str);

            if (std.mem.eql(u8, method_str, "d3bj")) {
                self.vdw.method = .d3bj;
                self.vdw.enabled = true;
                return;
            }
            if (std.mem.eql(u8, method_str, "none")) {
                self.vdw.method = .none;
                return;
            }
            return error.UnsupportedToml;
        }
        if (std.mem.eql(u8, key, "cutoff_radius")) {
            self.vdw.cutoff_radius = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "cn_cutoff")) {
            self.vdw.cn_cutoff = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "s6")) {
            self.vdw.s6 = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "s8")) {
            self.vdw.s8 = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "a1")) {
            self.vdw.a1 = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "a2")) {
            self.vdw.a2 = try parse_float(value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_relax_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        if (std.mem.eql(u8, key, "enabled")) {
            self.relax.enabled = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "algorithm")) {
            const algorithm_str = try parse_string(alloc, value);
            defer alloc.free(algorithm_str);

            if (std.mem.eql(u8, algorithm_str, "steepest_descent")) {
                self.relax.algorithm = .steepest_descent;
                return;
            }
            if (std.mem.eql(u8, algorithm_str, "cg")) {
                self.relax.algorithm = .cg;
                return;
            }
            if (std.mem.eql(u8, algorithm_str, "bfgs")) {
                self.relax.algorithm = .bfgs;
                return;
            }
            return error.UnsupportedToml;
        }
        if (std.mem.eql(u8, key, "max_iter")) {
            self.relax.max_iter = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "force_tol")) {
            self.relax.force_tol = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "max_step")) {
            self.relax.max_step = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "output_trajectory")) {
            self.relax.output_trajectory = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "cell_relax")) {
            self.relax.cell_relax = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "stress_tol")) {
            self.relax.stress_tol = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "cell_step")) {
            self.relax.cell_step = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "target_pressure")) {
            self.relax.target_pressure = try parse_float(value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_dfpt_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        if (std.mem.eql(u8, key, "enabled")) {
            self.dfpt.enabled = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "sternheimer_tol")) {
            self.dfpt.sternheimer_tol = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "sternheimer_max_iter")) {
            self.dfpt.sternheimer_max_iter = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "scf_tol")) {
            self.dfpt.scf_tol = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "scf_max_iter")) {
            self.dfpt.scf_max_iter = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "mixing_beta")) {
            self.dfpt.mixing_beta = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "alpha_shift")) {
            self.dfpt.alpha_shift = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "qpath_npoints")) {
            self.dfpt.qpath_npoints = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "pulay_history")) {
            self.dfpt.pulay_history = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "pulay_start")) {
            self.dfpt.pulay_start = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "kpoint_threads")) {
            self.dfpt.kpoint_threads = try float_to_index(try parse_float(value));
            self.dfpt_kpoint_threads_explicit = true;
            return;
        }
        if (std.mem.eql(u8, key, "perturbation_threads")) {
            self.dfpt.perturbation_threads = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "qgrid")) {
            const qgrid = try parse_array_numbers(value, 3);
            self.dfpt.qgrid = .{
                try float_to_index(qgrid[0]),
                try float_to_index(qgrid[1]),
                try float_to_index(qgrid[2]),
            };
            return;
        }
        if (std.mem.eql(u8, key, "dos_qmesh")) {
            const dos_qmesh = try parse_array_numbers(value, 3);
            self.dfpt.dos_qmesh = .{
                try float_to_index(dos_qmesh[0]),
                try float_to_index(dos_qmesh[1]),
                try float_to_index(dos_qmesh[2]),
            };
            return;
        }
        if (std.mem.eql(u8, key, "dos_sigma")) {
            self.dfpt.dos_sigma = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "dos_nbin")) {
            self.dfpt.dos_nbin = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "compute_dielectric")) {
            self.dfpt.compute_dielectric = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "log_level")) {
            const level_str = try parse_string(alloc, value);
            defer alloc.free(level_str);

            self.dfpt.log_level = try runtime_logging.parse_level(level_str);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_dfpt_qpath_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        const idx = self.current_dfpt_qpath_index orelse return error.InvalidToml;
        if (std.mem.eql(u8, key, "label")) {
            try replace_owned_string(alloc, &self.dfpt_qpath_list.items[idx].label, value);
            return;
        }
        if (std.mem.eql(u8, key, "k")) {
            self.dfpt_qpath_list.items[idx].k = try parse_vec3(value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_band_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        if (std.mem.eql(u8, key, "points")) {
            self.band.points_per_segment = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "nbands")) {
            self.band.nbands = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "solver")) {
            const solver_str = try parse_string(alloc, value);
            defer alloc.free(solver_str);

            self.band.solver = try parse_band_solver(solver_str);
            return;
        }
        if (std.mem.eql(u8, key, "kpoint_threads")) {
            self.band.kpoint_threads = try float_to_index(try parse_float(value));
            self.band_kpoint_threads_explicit = true;
            return;
        }
        if (std.mem.eql(u8, key, "iterative_max_iter")) {
            self.band.iterative_max_iter = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "iterative_tol")) {
            self.band.iterative_tol = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "iterative_max_subspace")) {
            self.band.iterative_max_subspace = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "iterative_block_size")) {
            self.band.iterative_block_size = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "iterative_init_diagonal")) {
            self.band.iterative_init_diagonal = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "iterative_reuse_vectors")) {
            self.band.iterative_reuse_vectors = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "use_symmetry")) {
            self.band.use_symmetry = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "lobpcg_parallel")) {
            self.band.lobpcg_parallel = try parse_bool(value);
            self.band_lobpcg_parallel_explicit = true;
            return;
        }
        if (std.mem.eql(u8, key, "path")) {
            try replace_optional_owned_string(alloc, &self.band.path_string, value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_dos_field(self: *LoadState, key: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, key, "enabled")) {
            self.dos.enabled = try parse_bool(value);
            return;
        }
        if (std.mem.eql(u8, key, "sigma")) {
            self.dos.sigma = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "npoints")) {
            self.dos.npoints = try float_to_index(try parse_float(value));
            return;
        }
        if (std.mem.eql(u8, key, "emin")) {
            self.dos.emin = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "emax")) {
            self.dos.emax = try parse_float(value);
            return;
        }
        if (std.mem.eql(u8, key, "pdos")) {
            self.dos.pdos = try parse_bool(value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_output_field(self: *LoadState, key: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, key, "cube")) {
            self.output.cube = try parse_bool(value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn parse_pseudopotential_field(
        self: *LoadState,
        alloc: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
    ) !void {
        const idx = self.current_pseudo_index orelse return error.InvalidToml;
        if (std.mem.eql(u8, key, "element")) {
            try replace_owned_string(alloc, &self.pseudo_list.items[idx].element, value);
            return;
        }
        if (std.mem.eql(u8, key, "path")) {
            try replace_owned_string(alloc, &self.pseudo_list.items[idx].path, value);
            return;
        }
        if (std.mem.eql(u8, key, "format")) {
            const format_str = try parse_string(alloc, value);
            defer alloc.free(format_str);

            self.pseudo_list.items[idx].format = try pseudo.parse_format(format_str);
            return;
        }
        if (std.mem.eql(u8, key, "core_energy_ry")) {
            _ = try parse_float(value);
            return;
        }
        return error.UnsupportedToml;
    }

    fn validate_required_fields(self: *const LoadState) !void {
        if (self.xyz_path == null) return error.MissingXyz;
        if (self.a1 == null or self.a2 == null or self.a3 == null) return error.MissingCell;
        for (self.pseudo_list.items) |spec| {
            if (spec.element.len == 0 or spec.path.len == 0) {
                return error.InvalidPseudopotential;
            }
        }
    }

    fn apply_thread_defaults(self: *LoadState) void {
        if (!self.scf_kpoint_threads_explicit) {
            self.scf.kpoint_threads = self.top_threads;
        }
        if (!self.band_kpoint_threads_explicit) {
            self.band.kpoint_threads = self.top_threads;
        }
        if (!self.dfpt_kpoint_threads_explicit) {
            self.dfpt.kpoint_threads = self.top_threads;
        }
        if (!self.band_lobpcg_parallel_explicit) {
            self.band.lobpcg_parallel = self.band.kpoint_threads == 1;
        }
    }

    fn finalize(self: *LoadState, alloc: std.mem.Allocator) !Config {
        try self.validate_required_fields();
        self.apply_thread_defaults();

        const dfpt_qpath = try self.dfpt_qpath_list.toOwnedSlice(alloc);
        errdefer alloc.free(dfpt_qpath);

        const pseudopotentials = try self.pseudo_list.toOwnedSlice(alloc);

        self.dfpt.qpath = dfpt_qpath;
        self.band.path = &.{};
        return .{
            .title = self.title,
            .xyz_path = self.xyz_path.?,
            .out_dir = self.out_dir,
            .units = self.units,
            .linalg_backend = self.linalg_backend,
            .threads = self.top_threads,
            .cell = math.Mat3.from_rows(self.a1.?, self.a2.?, self.a3.?),
            .boundary = self.boundary,
            .scf = self.scf,
            .ewald = self.ewald,
            .vdw = self.vdw,
            .band = self.band,
            .relax = self.relax,
            .dfpt = self.dfpt,
            .dos = self.dos,
            .output = self.output,
            .pseudopotentials = pseudopotentials,
        };
    }
};

/// Load config from a minimal TOML subset.
pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Config {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1024 * 1024));
    defer alloc.free(content);

    var state = try LoadState.init(alloc);
    errdefer state.deinit(alloc);

    try process_load_lines(alloc, content, &state);
    return try state.finalize(alloc);
}

fn process_load_lines(
    alloc: std.mem.Allocator,
    content: []const u8,
    state: *LoadState,
) !void {
    var it = std.mem.splitScalar(u8, content, '\n');
    while (it.next()) |raw_line| {
        try process_load_line(alloc, state, raw_line);
    }
}

fn process_load_line(
    alloc: std.mem.Allocator,
    state: *LoadState,
    raw_line: []const u8,
) !void {
    const line = std.mem.trim(u8, strip_comment(raw_line), " \t\r");
    if (line.len == 0) return;
    if (line[0] == '[') return try parse_section_line(alloc, state, line);

    const eq_index = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidToml;
    const key = std.mem.trim(u8, line[0..eq_index], " \t");
    const value = std.mem.trim(u8, line[eq_index + 1 ..], " \t");
    try state.parse_field(alloc, key, value);
}

fn parse_section_line(
    alloc: std.mem.Allocator,
    state: *LoadState,
    line: []const u8,
) !void {
    if (line.len < 3) return error.InvalidToml;
    if (line[1] == '[') return try parse_repeated_section(alloc, state, line);
    return try parse_standard_section(state, line);
}

fn parse_repeated_section(
    alloc: std.mem.Allocator,
    state: *LoadState,
    line: []const u8,
) !void {
    if (line.len < 5 or line[line.len - 2] != ']' or line[line.len - 1] != ']') {
        return error.InvalidToml;
    }

    const name = std.mem.trim(u8, line[2 .. line.len - 2], " \t");
    if (std.mem.eql(u8, name, "dfpt.qpath")) {
        return try state.begin_dfpt_qpath(alloc);
    }
    if (std.mem.eql(u8, name, "pseudopotential")) {
        return try state.begin_pseudopotential(alloc);
    }
    return error.UnsupportedToml;
}

fn parse_standard_section(state: *LoadState, line: []const u8) !void {
    if (line[line.len - 1] != ']') return error.InvalidToml;
    const name = std.mem.trim(u8, line[1 .. line.len - 1], " \t");
    if (std.mem.eql(u8, name, "cell")) return state.enter_section(.cell);
    if (std.mem.eql(u8, name, "scf")) return state.enter_section(.scf);
    if (std.mem.eql(u8, name, "ewald")) return state.enter_section(.ewald);
    if (std.mem.eql(u8, name, "vdw")) return state.enter_section(.vdw);
    if (std.mem.eql(u8, name, "band")) return state.enter_section(.band);
    if (std.mem.eql(u8, name, "relax")) return state.enter_section(.relax);
    if (std.mem.eql(u8, name, "dfpt")) return state.enter_section(.dfpt);
    if (std.mem.eql(u8, name, "dos")) return state.enter_section(.dos);
    if (std.mem.eql(u8, name, "output")) return state.enter_section(.output);
    return error.UnsupportedToml;
}

fn replace_owned_string(
    alloc: std.mem.Allocator,
    target: anytype,
    value: []const u8,
) !void {
    const target_info = @typeInfo(@TypeOf(target));
    comptime {
        if (target_info != .pointer) {
            @compileError("replace_owned_string target must be a pointer");
        }
        const child = target_info.pointer.child;
        if (child != []u8 and child != []const u8) {
            @compileError("replace_owned_string target must be *[]u8 or *[]const u8");
        }
    }
    const parsed = try parse_string(alloc, value);
    alloc.free(target.*);
    target.* = parsed;
}

fn replace_optional_owned_string(
    alloc: std.mem.Allocator,
    target: *?[]u8,
    value: []const u8,
) !void {
    const parsed = try parse_string(alloc, value);
    if (target.*) |old| alloc.free(old);
    target.* = parsed;
}

fn replace_optional_float_array(
    alloc: std.mem.Allocator,
    target: *?[]f64,
    value: []const u8,
) !void {
    const parsed = try parse_float_array(alloc, value);
    if (target.*) |old| alloc.free(old);
    target.* = parsed;
}

fn trim_toml_value(value: []const u8) []const u8 {
    return std.mem.trim(u8, value, " \t\"'");
}

/// Remove comments while respecting quoted strings.
fn strip_comment(line: []const u8) []const u8 {
    var in_string = false;
    var escaped = false;
    for (line, 0..) |c, i| {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (c == '#' and !in_string) {
            return line[0..i];
        }
    }
    return line;
}

/// Parse a quoted string with basic escapes.
fn parse_string(alloc: std.mem.Allocator, value: []const u8) ![]u8 {
    if (value.len < 2 or value[0] != '"' or value[value.len - 1] != '"') {
        return error.InvalidString;
    }
    const inner = value[1 .. value.len - 1];
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);

    var i: usize = 0;
    while (i < inner.len) : (i += 1) {
        const c = inner[i];
        if (c != '\\') {
            try out.append(alloc, c);
            continue;
        }
        if (i + 1 >= inner.len) return error.InvalidString;
        const next = inner[i + 1];
        switch (next) {
            '\\' => try out.append(alloc, '\\'),
            '"' => try out.append(alloc, '"'),
            'n' => try out.append(alloc, '\n'),
            't' => try out.append(alloc, '\t'),
            else => return error.InvalidString,
        }
        i += 1;
    }
    return try out.toOwnedSlice(alloc);
}

/// Parse a floating point value.
fn parse_float(value: []const u8) !f64 {
    return try std.fmt.parseFloat(f64, value);
}

/// Parse a boolean value.
fn parse_bool(value: []const u8) !bool {
    if (std.mem.eql(u8, value, "true")) return true;
    if (std.mem.eql(u8, value, "false")) return false;
    return error.InvalidBool;
}

/// Parse SCF solver mode.
fn parse_scf_solver(value: []const u8) !ScfSolver {
    if (std.mem.eql(u8, value, "dense")) return .dense;
    if (std.mem.eql(u8, value, "iterative")) return .iterative;
    if (std.mem.eql(u8, value, "cg")) return .cg;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    return error.InvalidScfSolver;
}

fn parse_xc_functional(value: []const u8) !xc.Functional {
    if (std.mem.eql(u8, value, "lda")) return .lda_pz;
    if (std.mem.eql(u8, value, "lda_pz")) return .lda_pz;
    if (std.mem.eql(u8, value, "pbe")) return .pbe;
    return error.InvalidXcFunctional;
}

fn parse_smearing_method(value: []const u8) !SmearingMethod {
    if (std.mem.eql(u8, value, "none")) return .none;
    if (std.mem.eql(u8, value, "fermi")) return .fermi_dirac;
    if (std.mem.eql(u8, value, "fermi_dirac")) return .fermi_dirac;
    return error.InvalidSmearingMethod;
}

fn parse_convergence_metric(value: []const u8) !ConvergenceMetric {
    if (std.mem.eql(u8, value, "density")) return .density;
    if (std.mem.eql(u8, value, "potential")) return .potential;
    if (std.mem.eql(u8, value, "vresid")) return .potential;
    return error.InvalidConvergenceMetric;
}

fn parse_local_potential_mode(value: []const u8) !LocalPotentialMode {
    if (std.mem.eql(u8, value, "tail")) return .tail;
    if (std.mem.eql(u8, value, "ewald")) return .ewald;
    if (std.mem.eql(u8, value, "short_range")) return .short_range;
    return error.InvalidLocalPotentialMode;
}

/// Parse band solver mode.
fn parse_band_solver(value: []const u8) !BandSolver {
    if (std.mem.eql(u8, value, "dense")) return .dense;
    if (std.mem.eql(u8, value, "iterative")) return .iterative;
    if (std.mem.eql(u8, value, "cg")) return .cg;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    return error.InvalidBandSolver;
}

/// Parse a three-element array into Vec3.
fn parse_vec3(value: []const u8) !math.Vec3 {
    const vals = try parse_array_numbers(value, 3);
    return .{ .x = vals[0], .y = vals[1], .z = vals[2] };
}

/// Parse a fixed-length numeric array.
fn parse_array_numbers(value: []const u8, expected_len: usize) ![3]f64 {
    if (value.len < 2 or value[0] != '[' or value[value.len - 1] != ']') {
        return error.InvalidArray;
    }
    const inner = value[1 .. value.len - 1];
    var parts = std.mem.splitScalar(u8, inner, ',');
    var out: [3]f64 = undefined;
    var count: usize = 0;
    while (parts.next()) |raw| {
        if (count >= expected_len) return error.InvalidArray;
        const token = std.mem.trim(u8, raw, " \t");
        out[count] = try std.fmt.parseFloat(f64, token);
        count += 1;
    }
    if (count != expected_len) return error.InvalidArray;
    return out;
}

/// Parse a variable-length float array from TOML, e.g. "[1.0, 2.0, 3.0]".
fn parse_float_array(alloc: std.mem.Allocator, value: []const u8) ![]f64 {
    if (value.len < 2 or value[0] != '[' or value[value.len - 1] != ']') {
        return error.InvalidArray;
    }
    const inner = value[1 .. value.len - 1];
    var list: std.ArrayList(f64) = .empty;
    errdefer list.deinit(alloc);

    var parts = std.mem.splitScalar(u8, inner, ',');
    while (parts.next()) |raw| {
        const token = std.mem.trim(u8, raw, " \t");
        if (token.len == 0) continue;
        const v = try std.fmt.parseFloat(f64, token);
        try list.append(alloc, v);
    }
    return try list.toOwnedSlice(alloc);
}

/// Convert a float to a rounded positive index.
fn float_to_index(value: f64) !usize {
    const rounded = std.math.floor(value + 0.5);
    if (rounded < 0.0) return error.InvalidArray;
    return @intFromFloat(rounded);
}

// --- Tests ---

/// Build a minimal valid Config for testing (all defaults are valid).
const default_test_config: Config = .{
    .title = @constCast("test"),
    .xyz_path = @constCast("test.xyz"),
    .out_dir = @constCast("out"),
    .units = .angstrom,
    .linalg_backend = .openblas,
    .threads = 0,
    .cell = math.Mat3.from_rows(
        .{ .x = 10.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 10.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 10.0 },
    ),
    .scf = .{
        .enabled = true,
        .solver = .iterative,
        .xc = .lda_pz,
        .smearing = .none,
        .smear_ry = 0.0,
        .ecut_ry = 30.0,
        .kmesh = .{ 4, 4, 4 },
        .kmesh_shift = .{ 0.0, 0.0, 0.0 },
        .grid = .{ 0, 0, 0 },
        .grid_scale = 1.0,
        .mixing_beta = 0.3,
        .max_iter = 50,
        .convergence = 1e-6,
        .convergence_metric = .density,
        .profile = false,
        .quiet = false,
        .debug_nonlocal = false,
        .debug_local = false,
        .debug_fermi = false,
        .enable_nonlocal = true,
        .local_potential = .short_range,
        .symmetry = true,
        .time_reversal = true,
        .kpoint_threads = 0,
        .iterative_max_iter = 20,
        .iterative_tol = 1e-4,
        .iterative_max_subspace = 0,
        .iterative_block_size = 0,
        .iterative_init_diagonal = false,
        .iterative_warmup_steps = 2,
        .iterative_warmup_max_iter = 10,
        .iterative_warmup_tol = 1e-3,
        .iterative_reuse_vectors = true,
        .kerker_q0 = 0.0,
        .diemac = 12.0,
        .dielng = 1.0,
        .pulay_history = 8,
        .pulay_start = 4,
        .mixing_mode = .potential,
        .use_rfft = false,
        .fft_backend = .fftw,
        .nspin = 1,
        .spinat = null,
        .compute_stress = false,
        .reference_json = null,
        .compare_reference_json = null,
        .comparison_json = null,
        .compare_tolerance_json = null,
    },
    .ewald = .{ .alpha = 0.0, .rcut = 0.0, .gcut = 0.0, .tol = 1e-8 },
    .vdw = .{},
    .band = .{
        .points_per_segment = 60,
        .nbands = 8,
        .path = &.{},
        .solver = .dense,
        .iterative_max_iter = 40,
        .iterative_tol = 1e-6,
        .iterative_max_subspace = 0,
        .iterative_block_size = 0,
        .iterative_init_diagonal = false,
        .kpoint_threads = 0,
        .iterative_reuse_vectors = true,
        .use_symmetry = false,
        .lobpcg_parallel = false,
    },
    .relax = .{
        .enabled = false,
        .algorithm = .bfgs,
        .max_iter = 100,
        .force_tol = 1e-4,
        .max_step = 0.5,
        .output_trajectory = false,
    },
    .dfpt = .{
        .enabled = false,
        .sternheimer_tol = 1e-8,
        .sternheimer_max_iter = 200,
        .scf_tol = 1e-10,
        .scf_max_iter = 50,
        .mixing_beta = 0.3,
        .alpha_shift = 0.01,
        .qpath_npoints = 0,
        .pulay_history = 8,
        .pulay_start = 4,
        .kpoint_threads = 0,
        .perturbation_threads = 1,
        .qgrid = null,
        .qpath = &.{},
    },
    .dos = .{},
    .output = .{},
    .pseudopotentials = @constCast(&[_]pseudo.Spec{.{
        .element = @constCast("Si"),
        .path = @constCast("Si.upf"),
        .format = .upf,
    }}),
};

fn test_default_config() Config {
    return default_test_config;
}

fn count_by_severity(issues: []const ValidationIssue, severity: ValidationSeverity) usize {
    var n: usize = 0;
    for (issues) |issue| {
        if (issue.severity == severity) n += 1;
    }
    return n;
}

test "load: pseudopotential section parses owned const strings" {
    const alloc = std.testing.allocator;
    const content =
        \\[[pseudopotential]]
        \\element = "Si"
        \\path = "pseudo/Si.upf"
        \\format = "upf"
        \\
    ;

    var state = try LoadState.init(alloc);
    defer state.deinit(alloc);

    try process_load_lines(alloc, content, &state);

    try std.testing.expectEqual(@as(usize, 1), state.pseudo_list.items.len);
    try std.testing.expectEqualStrings("Si", state.pseudo_list.items[0].element);
    try std.testing.expectEqualStrings("pseudo/Si.upf", state.pseudo_list.items[0].path);
    try std.testing.expectEqual(pseudo.Format.upf, state.pseudo_list.items[0].format);
}

test "validate: valid default config produces no issues" {
    const alloc = std.testing.allocator;
    const cfg = test_default_config();
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 0), result.issues.len);
    try std.testing.expect(!result.has_errors());
}

test "validate: ecut_ry = 0 is an error" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.ecut_ry = 0;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .err) >= 1);
}

test "validate: ecut_ry = 3.0 is a warning" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.ecut_ry = 3.0;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .warning) >= 1);
}

test "validate: mixing_beta out of range" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.mixing_beta = 1.5;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: smearing enabled with smear_ry = 0" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.smearing = .fermi_dirac;
    cfg.scf.smear_ry = 0.0;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: pulay_start exceeds pulay_history" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.pulay_history = 5;
    cfg.scf.pulay_start = 10;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: degenerate cell" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.cell = math.Mat3.from_rows(
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
    );
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: no pseudopotentials" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.pseudopotentials = &.{};
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: relax requires scf" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.relax.enabled = true;
    cfg.scf.enabled = false;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: isolated boundary with kmesh > 1 is a warning" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.boundary = .isolated;
    cfg.scf.kmesh = .{ 4, 4, 4 };
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .warning) >= 1);
}

test "validate: hint for dense solver" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.solver = .dense;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .hint) >= 1);
}

test "validate: hint for non-fftw backend" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.fft_backend = .zig;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .hint) >= 1);
}

test "validate: hint for diemac = 1.0 with tight convergence" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.diemac = 1.0;
    cfg.scf.convergence = 1e-8;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .hint) >= 1);
}

test "validate: no diemac hint with loose convergence" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.diemac = 1.0;
    cfg.scf.convergence = 1e-6; // loose enough that diemac doesn't matter
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    // diemac hint should NOT fire when convergence >= 1e-6
    var has_diemac_hint = false;
    for (result.issues) |issue| {
        if (issue.severity == .hint and std.mem.eql(u8, issue.field, "diemac")) {
            has_diemac_hint = true;
        }
    }
    try std.testing.expect(!has_diemac_hint);
}

test "validate: hint for density mixing" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.mixing_mode = .density;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .hint) >= 1);
}

test "validate: hint for tight iterative_tol" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.iterative_tol = 1e-8;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .hint) >= 1);
}

test "validate: hint for symmetry disabled" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.symmetry = false;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .hint) >= 1);
}

test "validate: no hints with recommended settings" {
    const alloc = std.testing.allocator;
    const cfg = test_default_config(); // default uses recommended settings
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expectEqual(@as(usize, 0), count_by_severity(result.issues, .hint));
}

test "validate: vdw enabled with method none" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.vdw.enabled = true;
    cfg.vdw.method = .none;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: dfpt with smearing is error" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.dfpt.enabled = true;
    cfg.scf.smearing = .fermi_dirac;
    cfg.scf.smear_ry = 0.01;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: dfpt with nspin=2 is error" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.dfpt.enabled = true;
    cfg.scf.nspin = 2;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: band points_per_segment = 0 is error" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.band.path_string = @constCast("G-X");
    cfg.band.points_per_segment = 0;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(result.has_errors());
}

test "validate: nspin=2 without spinat is warning" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.nspin = 2;
    cfg.scf.spinat = null;
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .warning) >= 1);
}

test "validate: grid too small for ecut is warning" {
    const alloc = std.testing.allocator;
    var cfg = test_default_config();
    cfg.scf.ecut_ry = 60.0;
    cfg.scf.grid = .{ 8, 8, 8 }; // way too small for ecut=60
    var result = try cfg.validate(alloc);
    defer result.deinit();

    try std.testing.expect(!result.has_errors());
    try std.testing.expect(count_by_severity(result.issues, .warning) >= 1);
}
