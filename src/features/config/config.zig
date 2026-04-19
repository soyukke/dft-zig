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
    diemac: f64, // Macroscopic dielectric constant for model preconditioner (1.0 = disabled, ~12 for Si, 1e6 for metals)
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

pub fn boundaryConditionName(bc: BoundaryCondition) []const u8 {
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
pub fn parseFftBackend(value: []const u8) !FftBackend {
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
pub fn fftBackendName(backend: FftBackend) []const u8 {
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
pub fn scfSolverName(solver: ScfSolver) []const u8 {
    return switch (solver) {
        .dense => "dense",
        .iterative => "iterative",
        .cg => "cg",
        .auto => "auto",
    };
}

pub fn bandSolverName(solver: BandSolver) []const u8 {
    return switch (solver) {
        .dense => "dense",
        .iterative => "iterative",
        .cg => "cg",
        .auto => "auto",
    };
}

pub fn xcFunctionalName(xc_func: xc.Functional) []const u8 {
    return xc.functionalName(xc_func);
}

pub fn smearingName(method: SmearingMethod) []const u8 {
    return switch (method) {
        .none => "none",
        .fermi_dirac => "fermi_dirac",
    };
}

pub fn convergenceMetricName(metric: ConvergenceMetric) []const u8 {
    return switch (metric) {
        .density => "density",
        .potential => "potential",
    };
}

pub fn localPotentialModeName(mode: LocalPotentialMode) []const u8 {
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

pub fn vdwMethodName(method: VdwMethod) []const u8 {
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

pub fn relaxAlgorithmName(algo: RelaxAlgorithm) []const u8 {
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
    log_level: runtime_logging.Level = .info, // warn=warnings only, info=summary, debug=full diagnostics
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

    pub fn hasErrors(self: ValidationResult) bool {
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

    /// Validate config after parsing. Returns all issues (errors + warnings).
    pub fn validate(self: *const Config, alloc: std.mem.Allocator) !ValidationResult {
        var issues: std.ArrayList(ValidationIssue) = .empty;
        errdefer {
            for (issues.items) |issue| alloc.free(issue.message);
            issues.deinit(alloc);
        }

        // --- Cell validity (convert to Bohr for grid checks) ---
        const unit_scale = math.unitsScaleToBohr(self.units);
        const cell_bohr = self.cell.scale(unit_scale);
        const a1 = cell_bohr.row(0);
        const a2 = cell_bohr.row(1);
        const a3 = cell_bohr.row(2);
        const volume = a1.dot(a2.cross(a3));
        if (@abs(volume) <= 1e-12) {
            try addIssueLiteral(alloc, &issues, .err, "cell", "a1,a2,a3", "cell has zero or near-zero volume (degenerate lattice vectors)");
        }

        // --- Pseudopotentials ---
        if (self.pseudopotentials.len == 0) {
            try addIssueLiteral(alloc, &issues, .err, "pseudopotential", "", "at least one pseudopotential must be specified");
        }

        // --- SCF checks ---
        if (self.scf.ecut_ry <= 0) {
            try addIssue(alloc, &issues, .err, "scf", "ecut_ry", try std.fmt.allocPrint(alloc, "must be positive (got {d:.2})", .{self.scf.ecut_ry}));
        } else if (self.scf.ecut_ry < 5.0) {
            try addIssue(alloc, &issues, .warning, "scf", "ecut_ry", try std.fmt.allocPrint(alloc, "{d:.1} is unusually low; typical range is 15-100 Ry", .{self.scf.ecut_ry}));
        }

        for (self.scf.kmesh, 0..) |k, i| {
            if (k == 0) {
                try addIssue(alloc, &issues, .err, "scf", "kmesh", try std.fmt.allocPrint(alloc, "dimension [{d}] must be positive", .{i}));
            }
        }

        if (self.scf.grid_scale <= 0) {
            try addIssueLiteral(alloc, &issues, .err, "scf", "grid_scale", "must be positive");
        }

        if (self.scf.max_iter == 0) {
            try addIssueLiteral(alloc, &issues, .err, "scf", "max_iter", "must be positive");
        }

        if (self.scf.convergence <= 0) {
            try addIssueLiteral(alloc, &issues, .err, "scf", "convergence", "must be positive");
        } else if (self.scf.convergence > 1e-3) {
            try addIssue(alloc, &issues, .warning, "scf", "convergence", try std.fmt.allocPrint(alloc, "{e} is loose; results may be inaccurate", .{self.scf.convergence}));
        }

        if (self.scf.mixing_beta <= 0 or self.scf.mixing_beta > 1.0) {
            try addIssue(alloc, &issues, .err, "scf", "mixing_beta", try std.fmt.allocPrint(alloc, "must be in (0, 1] (got {d:.3})", .{self.scf.mixing_beta}));
        } else if (self.scf.mixing_beta > 0.7) {
            try addIssue(alloc, &issues, .warning, "scf", "mixing_beta", try std.fmt.allocPrint(alloc, "{d:.2} is high; may cause SCF oscillation", .{self.scf.mixing_beta}));
        }

        if (self.scf.smearing != .none and self.scf.smear_ry <= 0) {
            try addIssueLiteral(alloc, &issues, .err, "scf", "smear_ry", "must be > 0 when smearing is enabled");
        }
        if (self.scf.smearing == .none and self.scf.smear_ry > 0) {
            try addIssueLiteral(alloc, &issues, .warning, "scf", "smear_ry", "is set but smearing is disabled; value will be ignored");
        }

        if (self.scf.pulay_history > 0 and self.scf.pulay_start > self.scf.pulay_history) {
            try addIssue(alloc, &issues, .err, "scf", "pulay_start", try std.fmt.allocPrint(alloc, "({d}) exceeds pulay_history ({d})", .{ self.scf.pulay_start, self.scf.pulay_history }));
        }

        // --- Boundary + kmesh consistency ---
        if (self.boundary == .isolated) {
            for (self.scf.kmesh) |k| {
                if (k > 1) {
                    try addIssueLiteral(alloc, &issues, .warning, "scf", "kmesh", "isolated boundary with kmesh > [1,1,1]; only Gamma point is meaningful");
                    break;
                }
            }
        }

        // --- Relax checks (when enabled) ---
        if (self.relax.enabled) {
            if (!self.scf.enabled) {
                try addIssueLiteral(alloc, &issues, .err, "relax", "", "requires scf to be enabled");
            }
            if (self.relax.max_iter == 0) {
                try addIssueLiteral(alloc, &issues, .err, "relax", "max_iter", "must be positive");
            }
            if (self.relax.force_tol <= 0) {
                try addIssueLiteral(alloc, &issues, .err, "relax", "force_tol", "must be positive");
            }
            if (self.relax.max_step <= 0) {
                try addIssueLiteral(alloc, &issues, .err, "relax", "max_step", "must be positive");
            }
            if (self.relax.cell_relax and !self.scf.compute_stress) {
                try addIssueLiteral(alloc, &issues, .warning, "relax", "cell_relax", "cell_relax is enabled but compute_stress is off; stress will be auto-enabled");
            }
        }

        // --- DFPT checks (when enabled) ---
        if (self.dfpt.enabled) {
            if (!self.scf.enabled) {
                try addIssueLiteral(alloc, &issues, .err, "dfpt", "", "requires scf to be enabled");
            }
            if (self.scf.smearing != .none) {
                try addIssueLiteral(alloc, &issues, .err, "dfpt", "", "DFPT is incompatible with Fermi-Dirac smearing");
            }
            if (self.scf.nspin == 2) {
                try addIssueLiteral(alloc, &issues, .err, "dfpt", "", "spin-polarized DFPT (nspin=2) is not supported");
            }
            if (self.dfpt.sternheimer_max_iter == 0) {
                try addIssueLiteral(alloc, &issues, .err, "dfpt", "sternheimer_max_iter", "must be positive");
            }
            if (self.dfpt.sternheimer_tol <= 0) {
                try addIssueLiteral(alloc, &issues, .err, "dfpt", "sternheimer_tol", "must be positive");
            }
            if (self.dfpt.scf_max_iter == 0) {
                try addIssueLiteral(alloc, &issues, .err, "dfpt", "scf_max_iter", "must be positive");
            }
            if (self.dfpt.scf_tol <= 0) {
                try addIssueLiteral(alloc, &issues, .err, "dfpt", "scf_tol", "must be positive");
            }
            if (self.dfpt.mixing_beta <= 0 or self.dfpt.mixing_beta > 1.0) {
                try addIssue(alloc, &issues, .err, "dfpt", "mixing_beta", try std.fmt.allocPrint(alloc, "must be in (0, 1] (got {d:.3})", .{self.dfpt.mixing_beta}));
            }
            if (self.dfpt.pulay_history > 0 and self.dfpt.pulay_start > self.dfpt.pulay_history) {
                try addIssue(alloc, &issues, .err, "dfpt", "pulay_start", try std.fmt.allocPrint(alloc, "({d}) exceeds pulay_history ({d})", .{ self.dfpt.pulay_start, self.dfpt.pulay_history }));
            }
            if (self.dfpt.qgrid) |qg| {
                for (qg, 0..) |q, i| {
                    if (q == 0) {
                        try addIssue(alloc, &issues, .err, "dfpt", "qgrid", try std.fmt.allocPrint(alloc, "dimension [{d}] must be positive", .{i}));
                    }
                }
            }
            if (self.dfpt.dos_qmesh) |qm| {
                for (qm, 0..) |q, i| {
                    if (q == 0) {
                        try addIssue(alloc, &issues, .err, "dfpt", "dos_qmesh", try std.fmt.allocPrint(alloc, "dimension [{d}] must be positive", .{i}));
                    }
                }
            }
        }

        // --- DOS checks (when enabled) ---
        if (self.dos.enabled) {
            if (self.dos.sigma <= 0) {
                try addIssueLiteral(alloc, &issues, .err, "dos", "sigma", "must be positive");
            }
            if (self.dos.npoints == 0) {
                try addIssueLiteral(alloc, &issues, .err, "dos", "npoints", "must be positive");
            }
            if (self.dos.emin != null and self.dos.emax != null and self.dos.emin.? >= self.dos.emax.?) {
                try addIssueLiteral(alloc, &issues, .err, "dos", "emin/emax", "emin must be less than emax");
            }
        }

        // --- VdW consistency ---
        if (self.vdw.enabled and self.vdw.method == .none) {
            try addIssueLiteral(alloc, &issues, .err, "vdw", "method", "vdw enabled but method = \"none\"; set method = \"d3bj\"");
        }

        // --- Band checks ---
        if (self.band.path.len > 0 or self.band.path_string != null) {
            if (self.band.points_per_segment == 0) {
                try addIssueLiteral(alloc, &issues, .err, "band", "points", "points must be positive");
            }
        }

        // --- Spin consistency ---
        if (self.scf.nspin == 2 and self.scf.spinat == null) {
            try addIssueLiteral(alloc, &issues, .warning, "scf", "spinat", "nspin=2 but spinat not set; calculation may converge to non-magnetic solution");
        }

        // --- Grid vs ecut consistency ---
        if (self.scf.grid[0] > 0 and self.scf.grid[1] > 0 and self.scf.grid[2] > 0 and
            self.scf.ecut_ry > 0 and @abs(volume) > 1e-12)
        {
            const recip = math.reciprocal(cell_bohr);
            const scale = if (self.scf.grid_scale > 0) self.scf.grid_scale else 1.0;
            const density_gmax = @max(2.0, scale) * @sqrt(self.scf.ecut_ry);
            const raw1 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / math.Vec3.norm(recip.row(0))))) * 2 + 1;
            const raw2 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / math.Vec3.norm(recip.row(1))))) * 2 + 1;
            const raw3 = @as(usize, @intFromFloat(std.math.ceil(density_gmax / math.Vec3.norm(recip.row(2))))) * 2 + 1;
            const rec1 = fft_sizing.nextFftSize(@max(raw1, 3));
            const rec2 = fft_sizing.nextFftSize(@max(raw2, 3));
            const rec3 = fft_sizing.nextFftSize(@max(raw3, 3));
            if (self.scf.grid[0] < raw1 or self.scf.grid[1] < raw2 or self.scf.grid[2] < raw3) {
                try addIssue(alloc, &issues, .warning, "scf", "grid", try std.fmt.allocPrint(alloc, "grid [{},{},{}] is smaller than recommended [{},{},{}] for ecut_ry={d:.1}; use grid = [{},{},{}] or set grid = [0,0,0] for auto", .{
                    self.scf.grid[0], self.scf.grid[1], self.scf.grid[2],
                    raw1,             raw2,             raw3,
                    self.scf.ecut_ry, rec1,             rec2,
                    rec3,
                }));
            }
        }

        // --- Recommendations (hints) ---

        // Solver: dense is very slow for non-trivial systems
        if (self.scf.enabled and self.scf.solver == .dense) {
            try addIssueLiteral(alloc, &issues, .hint, "scf", "solver", "solver = \"dense\" is slow for large systems; consider solver = \"iterative\" (LOBPCG)");
        }

        // FFT backend: fftw is significantly faster than pure Zig backends
        if (self.scf.enabled and self.scf.fft_backend != .fftw) {
            try addIssueLiteral(alloc, &issues, .hint, "scf", "fft_backend", "fft_backend = \"fftw\" is recommended for production calculations");
        }

        // Dielectric preconditioner: diemac=1.0 means disabled
        if (self.scf.enabled and self.scf.diemac == 1.0 and self.scf.convergence < 1e-6) {
            try addIssueLiteral(alloc, &issues, .hint, "scf", "diemac", "diemac = 1.0 (disabled); for tight convergence (<1e-6), set ~12 for semiconductors or ~1e6 for metals");
        }

        // Pulay/DIIS disabled
        if (self.scf.enabled and self.scf.pulay_history == 0) {
            try addIssueLiteral(alloc, &issues, .hint, "scf", "pulay_history", "pulay_history = 0 (DIIS disabled); pulay_history = 8 with pulay_start = 4 typically improves SCF convergence");
        }

        // Density mixing is known to oscillate for large ecut
        if (self.scf.enabled and self.scf.mixing_mode == .density) {
            try addIssueLiteral(alloc, &issues, .hint, "scf", "mixing_mode", "mixing_mode = \"density\" can oscillate at large ecut; mixing_mode = \"potential\" is recommended");
        }

        // iterative_tol too tight increases per-iteration cost
        if (self.scf.enabled and self.scf.solver == .iterative and self.scf.iterative_tol < 1e-6) {
            try addIssueLiteral(alloc, &issues, .hint, "scf", "iterative_tol", "iterative_tol < 1e-6 increases eigensolver cost per SCF iteration; 1e-4 is usually sufficient");
        }

        // Symmetry disabled
        if (self.scf.enabled and !self.scf.symmetry) {
            try addIssueLiteral(alloc, &issues, .hint, "scf", "symmetry", "symmetry = false; enabling symmetry reduces k-points and speeds up the calculation");
        }

        // Band solver recommendation
        if (self.band.path.len > 0 or self.band.path_string != null) {
            if (self.band.solver == .dense) {
                try addIssueLiteral(alloc, &issues, .hint, "band", "solver", "solver = \"dense\" is slow for band structure; consider solver = \"iterative\"");
            }
        }

        return .{
            .issues = try issues.toOwnedSlice(alloc),
            .allocator = alloc,
        };
    }

    /// Append a validation issue. Takes ownership of `message` (must be heap-allocated).
    /// Frees `message` if append fails.
    fn addIssue(
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

    /// Convenience: addIssue with a comptime string literal (dupes it).
    fn addIssueLiteral(
        alloc: std.mem.Allocator,
        issues: *std.ArrayList(ValidationIssue),
        severity: ValidationSeverity,
        section: []const u8,
        field: []const u8,
        comptime message: []const u8,
    ) !void {
        const owned = try alloc.dupe(u8, message);
        errdefer alloc.free(owned);
        try addIssue(alloc, issues, severity, section, field, owned);
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

/// Load config from a minimal TOML subset.
pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Config {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1024 * 1024));
    defer alloc.free(content);

    var title = try alloc.dupe(u8, "dft_zig");
    errdefer alloc.free(title);
    var xyz_path: ?[]u8 = null;
    errdefer if (xyz_path) |p| alloc.free(p);
    var out_dir = try alloc.dupe(u8, "out");
    errdefer alloc.free(out_dir);
    var units: math.Units = .angstrom;
    var linalg_backend: linalg.Backend = if (builtin.os.tag == .macos) .accelerate else .openblas;
    var boundary: BoundaryCondition = .periodic;
    var a1: ?math.Vec3 = null;
    var a2: ?math.Vec3 = null;
    var a3: ?math.Vec3 = null;

    var scf = ScfConfig{
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
        .diemac = 1.0, // 1.0 = disabled (no dielectric screening)
        .dielng = 1.0, // Thomas-Fermi screening length in Bohr
        .pulay_history = 8, // Pulay/DIIS mixing for faster SCF convergence
        .pulay_start = 4, // Simple mixing iterations before Pulay starts
        .mixing_mode = .potential, // Potential mixing for fast convergence (like ABINIT iscf=7)
        .use_rfft = false, // Disabled by default for now
        .fft_backend = .fftw, // FFTW is significantly faster
        .nspin = 1,
        .spinat = null,
        .compute_stress = false,
        .reference_json = null,
        .compare_reference_json = null,
        .comparison_json = null,
        .compare_tolerance_json = null,
    };
    errdefer {
        if (scf.spinat) |sa| {
            alloc.free(sa);
        }
        if (scf.reference_json) |stored_path| {
            alloc.free(stored_path);
        }
        if (scf.compare_reference_json) |stored_path| {
            alloc.free(stored_path);
        }
        if (scf.comparison_json) |stored_path| {
            alloc.free(stored_path);
        }
        if (scf.compare_tolerance_json) |stored_path| {
            alloc.free(stored_path);
        }
    }

    var ewald = EwaldConfig{
        .alpha = 0.0,
        .rcut = 0.0,
        .gcut = 0.0,
        .tol = 1e-8,
    };

    var vdw_cfg = VdwConfig{};

    var relax = RelaxConfig{
        .enabled = false,
        .algorithm = .bfgs,
        .max_iter = 100,
        .force_tol = 1e-4, // Ry/Bohr
        .max_step = 0.5, // Bohr
        .output_trajectory = false,
    };

    var dfpt_cfg = DfptConfig{
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

    var dfpt_qpath_list: std.ArrayList(BandPathPoint) = .empty;
    errdefer {
        for (dfpt_qpath_list.items) |p| {
            alloc.free(p.label);
        }
        dfpt_qpath_list.deinit(alloc);
    }
    var current_dfpt_qpath_index: ?usize = null;

    var dos_cfg = DosConfig{};
    var output_cfg = OutputConfig{};

    var top_threads: usize = 0;
    var scf_kpoint_threads_explicit = false;
    var band_kpoint_threads_explicit = false;
    var dfpt_kpoint_threads_explicit = false;
    var band_lobpcg_parallel_explicit = false;

    var band_points: usize = 60;
    var band_nbands: usize = 8;
    var band_solver: BandSolver = .dense;
    var band_iterative_max_iter: usize = 40;
    var band_iterative_tol: f64 = 1e-6; // 1e-6 is sufficient for SCF convergence
    var band_iterative_max_subspace: usize = 0;
    var band_iterative_block_size: usize = 0;
    var band_iterative_init_diagonal: bool = false;
    var band_kpoint_threads: usize = 0;
    var band_iterative_reuse_vectors: bool = true;
    var band_use_symmetry: bool = false; // TODO: proper symmetry-adapted basis transformation needed
    var band_lobpcg_parallel: bool = false; // k-point parallel is more efficient
    var band_path_string: ?[]u8 = null; // "auto" or "G-X-W-K-G-L" style path
    errdefer if (band_path_string) |s| alloc.free(s);

    var pseudo_list: std.ArrayList(pseudo.Spec) = .empty;
    errdefer {
        for (pseudo_list.items) |p| {
            alloc.free(p.element);
            alloc.free(p.path);
        }
        pseudo_list.deinit(alloc);
    }

    var current_section: Section = .root;
    var current_pseudo_index: ?usize = null;

    var it = std.mem.splitScalar(u8, content, '\n');
    while (it.next()) |raw_line| {
        const without_comment = stripComment(raw_line);
        const line = std.mem.trim(u8, without_comment, " \t\r");
        if (line.len == 0) continue;

        if (line[0] == '[') {
            if (line.len < 3) return error.InvalidToml;
            if (line[1] == '[') {
                if (line.len < 5 or line[line.len - 2] != ']' or line[line.len - 1] != ']') {
                    return error.InvalidToml;
                }
                const name = std.mem.trim(u8, line[2 .. line.len - 2], " \t");
                if (std.mem.eql(u8, name, "dfpt.qpath")) {
                    current_section = .dfpt_qpath;
                    current_dfpt_qpath_index = dfpt_qpath_list.items.len;
                    current_pseudo_index = null;
                    try dfpt_qpath_list.append(alloc, .{ .label = try alloc.dupe(u8, ""), .k = .{ .x = 0, .y = 0, .z = 0 } });
                } else if (std.mem.eql(u8, name, "pseudopotential")) {
                    current_section = .pseudopotential;
                    current_pseudo_index = pseudo_list.items.len;
                    current_dfpt_qpath_index = null;
                    try pseudo_list.append(alloc, .{
                        .element = try alloc.dupe(u8, ""),
                        .path = try alloc.dupe(u8, ""),
                        .format = .upf,
                    });
                } else {
                    return error.UnsupportedToml;
                }
            } else {
                if (line[line.len - 1] != ']') return error.InvalidToml;
                const name = std.mem.trim(u8, line[1 .. line.len - 1], " \t");
                if (std.mem.eql(u8, name, "cell")) {
                    current_section = .cell;
                    current_pseudo_index = null;
                } else if (std.mem.eql(u8, name, "scf")) {
                    current_section = .scf;
                    current_pseudo_index = null;
                } else if (std.mem.eql(u8, name, "ewald")) {
                    current_section = .ewald;
                    current_pseudo_index = null;
                } else if (std.mem.eql(u8, name, "vdw")) {
                    current_section = .vdw;
                    current_pseudo_index = null;
                } else if (std.mem.eql(u8, name, "band")) {
                    current_section = .band;
                    current_pseudo_index = null;
                } else if (std.mem.eql(u8, name, "relax")) {
                    current_section = .relax;
                    current_pseudo_index = null;
                } else if (std.mem.eql(u8, name, "dfpt")) {
                    current_section = .dfpt;
                    current_pseudo_index = null;
                } else if (std.mem.eql(u8, name, "dos")) {
                    current_section = .dos;
                    current_pseudo_index = null;
                } else if (std.mem.eql(u8, name, "output")) {
                    current_section = .output;
                    current_pseudo_index = null;
                } else {
                    return error.UnsupportedToml;
                }
            }
            continue;
        }

        const eq_index = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidToml;
        const key = std.mem.trim(u8, line[0..eq_index], " \t");
        const value = std.mem.trim(u8, line[eq_index + 1 ..], " \t");

        switch (current_section) {
            .root => {
                if (std.mem.eql(u8, key, "title")) {
                    alloc.free(title);
                    title = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "xyz")) {
                    if (xyz_path) |p| alloc.free(p);
                    xyz_path = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "out_dir")) {
                    alloc.free(out_dir);
                    out_dir = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "units")) {
                    const unit_str = try parseString(alloc, value);
                    defer alloc.free(unit_str);
                    if (std.mem.eql(u8, unit_str, "angstrom")) {
                        units = .angstrom;
                    } else if (std.mem.eql(u8, unit_str, "bohr")) {
                        units = .bohr;
                    } else {
                        return error.InvalidUnits;
                    }
                } else if (std.mem.eql(u8, key, "linalg_backend")) {
                    const backend_str = try parseString(alloc, value);
                    defer alloc.free(backend_str);
                    linalg_backend = try linalg.parseBackend(backend_str);
                } else if (std.mem.eql(u8, key, "threads")) {
                    top_threads = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "boundary")) {
                    const bc_str = try parseString(alloc, value);
                    defer alloc.free(bc_str);
                    if (std.mem.eql(u8, bc_str, "periodic")) {
                        boundary = .periodic;
                    } else if (std.mem.eql(u8, bc_str, "isolated")) {
                        boundary = .isolated;
                    } else {
                        return error.UnsupportedToml;
                    }
                } else {
                    return error.UnsupportedToml;
                }
            },
            .cell => {
                if (std.mem.eql(u8, key, "a1")) {
                    a1 = try parseVec3(value);
                } else if (std.mem.eql(u8, key, "a2")) {
                    a2 = try parseVec3(value);
                } else if (std.mem.eql(u8, key, "a3")) {
                    a3 = try parseVec3(value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .scf => {
                if (std.mem.eql(u8, key, "ecut_ry")) {
                    scf.ecut_ry = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "kmesh")) {
                    const km = try parseArrayNumbers(value, 3);
                    scf.kmesh = .{
                        try floatToIndex(km[0]),
                        try floatToIndex(km[1]),
                        try floatToIndex(km[2]),
                    };
                } else if (std.mem.eql(u8, key, "kmesh_shift")) {
                    const ks = try parseArrayNumbers(value, 3);
                    scf.kmesh_shift = .{ ks[0], ks[1], ks[2] };
                } else if (std.mem.eql(u8, key, "grid")) {
                    const grid = try parseArrayNumbers(value, 3);
                    scf.grid = .{
                        try floatToIndex(grid[0]),
                        try floatToIndex(grid[1]),
                        try floatToIndex(grid[2]),
                    };
                } else if (std.mem.eql(u8, key, "grid_scale")) {
                    scf.grid_scale = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "enabled")) {
                    scf.enabled = try parseBool(value);
                } else if (std.mem.eql(u8, key, "solver")) {
                    const solver_str = try parseString(alloc, value);
                    defer alloc.free(solver_str);
                    scf.solver = try parseScfSolver(solver_str);
                } else if (std.mem.eql(u8, key, "xc")) {
                    const xc_str = try parseString(alloc, value);
                    defer alloc.free(xc_str);
                    scf.xc = try parseXcFunctional(xc_str);
                } else if (std.mem.eql(u8, key, "smearing")) {
                    const smear_str = try parseString(alloc, value);
                    defer alloc.free(smear_str);
                    scf.smearing = try parseSmearingMethod(smear_str);
                } else if (std.mem.eql(u8, key, "smear_ry")) {
                    scf.smear_ry = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "mixing_beta")) {
                    scf.mixing_beta = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "max_iter")) {
                    scf.max_iter = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "convergence")) {
                    scf.convergence = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "convergence_metric")) {
                    const metric_str = try parseString(alloc, value);
                    defer alloc.free(metric_str);
                    scf.convergence_metric = try parseConvergenceMetric(metric_str);
                } else if (std.mem.eql(u8, key, "profile")) {
                    scf.profile = try parseBool(value);
                } else if (std.mem.eql(u8, key, "quiet")) {
                    scf.quiet = try parseBool(value);
                } else if (std.mem.eql(u8, key, "debug_nonlocal")) {
                    scf.debug_nonlocal = try parseBool(value);
                } else if (std.mem.eql(u8, key, "debug_local")) {
                    scf.debug_local = try parseBool(value);
                } else if (std.mem.eql(u8, key, "debug_fermi")) {
                    scf.debug_fermi = try parseBool(value);
                } else if (std.mem.eql(u8, key, "enable_nonlocal")) {
                    scf.enable_nonlocal = try parseBool(value);
                } else if (std.mem.eql(u8, key, "local_potential")) {
                    const mode_str = try parseString(alloc, value);
                    defer alloc.free(mode_str);
                    scf.local_potential = try parseLocalPotentialMode(mode_str);
                } else if (std.mem.eql(u8, key, "symmetry")) {
                    scf.symmetry = try parseBool(value);
                } else if (std.mem.eql(u8, key, "time_reversal")) {
                    scf.time_reversal = try parseBool(value);
                } else if (std.mem.eql(u8, key, "kpoint_threads")) {
                    scf.kpoint_threads = try floatToIndex(try parseFloat(value));
                    scf_kpoint_threads_explicit = true;
                } else if (std.mem.eql(u8, key, "iterative_max_iter")) {
                    scf.iterative_max_iter = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "iterative_tol")) {
                    scf.iterative_tol = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "iterative_max_subspace")) {
                    scf.iterative_max_subspace = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "iterative_block_size")) {
                    scf.iterative_block_size = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "iterative_init_diagonal")) {
                    scf.iterative_init_diagonal = try parseBool(value);
                } else if (std.mem.eql(u8, key, "iterative_warmup_steps")) {
                    scf.iterative_warmup_steps = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "iterative_warmup_max_iter")) {
                    scf.iterative_warmup_max_iter = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "iterative_warmup_tol")) {
                    scf.iterative_warmup_tol = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "iterative_reuse_vectors")) {
                    scf.iterative_reuse_vectors = try parseBool(value);
                } else if (std.mem.eql(u8, key, "kerker_q0")) {
                    scf.kerker_q0 = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "diemac")) {
                    const v = try parseFloat(value);
                    if (v < 1.0) return error.InvalidConfig;
                    scf.diemac = v;
                } else if (std.mem.eql(u8, key, "dielng")) {
                    const v = try parseFloat(value);
                    if (v <= 0.0) return error.InvalidConfig;
                    scf.dielng = v;
                } else if (std.mem.eql(u8, key, "pulay_history")) {
                    scf.pulay_history = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "pulay_start")) {
                    scf.pulay_start = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "mixing_mode")) {
                    const trimmed = std.mem.trim(u8, value, " \t\"'");
                    if (std.mem.eql(u8, trimmed, "density")) {
                        scf.mixing_mode = .density;
                    } else if (std.mem.eql(u8, trimmed, "potential")) {
                        scf.mixing_mode = .potential;
                    }
                } else if (std.mem.eql(u8, key, "use_rfft")) {
                    scf.use_rfft = try parseBool(value);
                } else if (std.mem.eql(u8, key, "compute_stress")) {
                    scf.compute_stress = try parseBool(value);
                } else if (std.mem.eql(u8, key, "fft_backend")) {
                    const trimmed = std.mem.trim(u8, value, " \t\"'");
                    scf.fft_backend = parseFftBackend(trimmed) catch return error.InvalidFftBackend;
                } else if (std.mem.eql(u8, key, "nspin")) {
                    scf.nspin = try floatToIndex(try parseFloat(value));
                    if (scf.nspin != 1 and scf.nspin != 2) return error.InvalidNspin;
                } else if (std.mem.eql(u8, key, "spinat")) {
                    if (scf.spinat) |old| alloc.free(old);
                    scf.spinat = try parseFloatArray(alloc, value);
                } else if (std.mem.eql(u8, key, "reference_json")) {
                    if (scf.reference_json) |old_path| {
                        alloc.free(old_path);
                    }
                    scf.reference_json = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "compare_reference_json")) {
                    if (scf.compare_reference_json) |old_path| {
                        alloc.free(old_path);
                    }
                    scf.compare_reference_json = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "comparison_json")) {
                    if (scf.comparison_json) |old_path| {
                        alloc.free(old_path);
                    }
                    scf.comparison_json = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "compare_tolerance_json")) {
                    if (scf.compare_tolerance_json) |old_path| {
                        alloc.free(old_path);
                    }
                    scf.compare_tolerance_json = try parseString(alloc, value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .ewald => {
                if (std.mem.eql(u8, key, "alpha")) {
                    ewald.alpha = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "rcut")) {
                    ewald.rcut = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "gcut")) {
                    ewald.gcut = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "tol")) {
                    ewald.tol = try parseFloat(value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .vdw => {
                if (std.mem.eql(u8, key, "enabled")) {
                    vdw_cfg.enabled = try parseBool(value);
                } else if (std.mem.eql(u8, key, "method")) {
                    const method_str = try parseString(alloc, value);
                    defer alloc.free(method_str);
                    if (std.mem.eql(u8, method_str, "d3bj")) {
                        vdw_cfg.method = .d3bj;
                        vdw_cfg.enabled = true;
                    } else if (std.mem.eql(u8, method_str, "none")) {
                        vdw_cfg.method = .none;
                    } else {
                        return error.UnsupportedToml;
                    }
                } else if (std.mem.eql(u8, key, "cutoff_radius")) {
                    vdw_cfg.cutoff_radius = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "cn_cutoff")) {
                    vdw_cfg.cn_cutoff = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "s6")) {
                    vdw_cfg.s6 = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "s8")) {
                    vdw_cfg.s8 = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "a1")) {
                    vdw_cfg.a1 = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "a2")) {
                    vdw_cfg.a2 = try parseFloat(value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .relax => {
                if (std.mem.eql(u8, key, "enabled")) {
                    relax.enabled = try parseBool(value);
                } else if (std.mem.eql(u8, key, "algorithm")) {
                    const algo_str = try parseString(alloc, value);
                    defer alloc.free(algo_str);
                    if (std.mem.eql(u8, algo_str, "steepest_descent")) {
                        relax.algorithm = .steepest_descent;
                    } else if (std.mem.eql(u8, algo_str, "cg")) {
                        relax.algorithm = .cg;
                    } else if (std.mem.eql(u8, algo_str, "bfgs")) {
                        relax.algorithm = .bfgs;
                    } else {
                        return error.UnsupportedToml;
                    }
                } else if (std.mem.eql(u8, key, "max_iter")) {
                    relax.max_iter = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "force_tol")) {
                    relax.force_tol = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "max_step")) {
                    relax.max_step = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "output_trajectory")) {
                    relax.output_trajectory = try parseBool(value);
                } else if (std.mem.eql(u8, key, "cell_relax")) {
                    relax.cell_relax = try parseBool(value);
                } else if (std.mem.eql(u8, key, "stress_tol")) {
                    relax.stress_tol = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "cell_step")) {
                    relax.cell_step = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "target_pressure")) {
                    relax.target_pressure = try parseFloat(value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .dfpt => {
                if (std.mem.eql(u8, key, "enabled")) {
                    dfpt_cfg.enabled = try parseBool(value);
                } else if (std.mem.eql(u8, key, "sternheimer_tol")) {
                    dfpt_cfg.sternheimer_tol = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "sternheimer_max_iter")) {
                    dfpt_cfg.sternheimer_max_iter = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "scf_tol")) {
                    dfpt_cfg.scf_tol = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "scf_max_iter")) {
                    dfpt_cfg.scf_max_iter = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "mixing_beta")) {
                    dfpt_cfg.mixing_beta = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "alpha_shift")) {
                    dfpt_cfg.alpha_shift = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "qpath_npoints")) {
                    dfpt_cfg.qpath_npoints = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "pulay_history")) {
                    dfpt_cfg.pulay_history = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "pulay_start")) {
                    dfpt_cfg.pulay_start = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "kpoint_threads")) {
                    dfpt_cfg.kpoint_threads = try floatToIndex(try parseFloat(value));
                    dfpt_kpoint_threads_explicit = true;
                } else if (std.mem.eql(u8, key, "perturbation_threads")) {
                    dfpt_cfg.perturbation_threads = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "qgrid")) {
                    const qg = try parseArrayNumbers(value, 3);
                    dfpt_cfg.qgrid = .{
                        try floatToIndex(qg[0]),
                        try floatToIndex(qg[1]),
                        try floatToIndex(qg[2]),
                    };
                } else if (std.mem.eql(u8, key, "dos_qmesh")) {
                    const dq = try parseArrayNumbers(value, 3);
                    dfpt_cfg.dos_qmesh = .{
                        try floatToIndex(dq[0]),
                        try floatToIndex(dq[1]),
                        try floatToIndex(dq[2]),
                    };
                } else if (std.mem.eql(u8, key, "dos_sigma")) {
                    dfpt_cfg.dos_sigma = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "dos_nbin")) {
                    dfpt_cfg.dos_nbin = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "compute_dielectric")) {
                    dfpt_cfg.compute_dielectric = try parseBool(value);
                } else if (std.mem.eql(u8, key, "log_level")) {
                    const level_str = try parseString(alloc, value);
                    defer alloc.free(level_str);
                    dfpt_cfg.log_level = try runtime_logging.parseLevel(level_str);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .dfpt_qpath => {
                const idx = current_dfpt_qpath_index orelse return error.InvalidToml;
                if (std.mem.eql(u8, key, "label")) {
                    alloc.free(dfpt_qpath_list.items[idx].label);
                    dfpt_qpath_list.items[idx].label = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "k")) {
                    dfpt_qpath_list.items[idx].k = try parseVec3(value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .band => {
                if (std.mem.eql(u8, key, "points")) {
                    band_points = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "nbands")) {
                    band_nbands = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "solver")) {
                    const solver_str = try parseString(alloc, value);
                    defer alloc.free(solver_str);
                    band_solver = try parseBandSolver(solver_str);
                } else if (std.mem.eql(u8, key, "kpoint_threads")) {
                    band_kpoint_threads = try floatToIndex(try parseFloat(value));
                    band_kpoint_threads_explicit = true;
                } else if (std.mem.eql(u8, key, "iterative_max_iter")) {
                    band_iterative_max_iter = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "iterative_tol")) {
                    band_iterative_tol = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "iterative_max_subspace")) {
                    band_iterative_max_subspace = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "iterative_block_size")) {
                    band_iterative_block_size = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "iterative_init_diagonal")) {
                    band_iterative_init_diagonal = try parseBool(value);
                } else if (std.mem.eql(u8, key, "iterative_reuse_vectors")) {
                    band_iterative_reuse_vectors = try parseBool(value);
                } else if (std.mem.eql(u8, key, "use_symmetry")) {
                    band_use_symmetry = try parseBool(value);
                } else if (std.mem.eql(u8, key, "lobpcg_parallel")) {
                    band_lobpcg_parallel = try parseBool(value);
                    band_lobpcg_parallel_explicit = true;
                } else if (std.mem.eql(u8, key, "path")) {
                    if (band_path_string) |old| alloc.free(old);
                    band_path_string = try parseString(alloc, value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .dos => {
                if (std.mem.eql(u8, key, "enabled")) {
                    dos_cfg.enabled = try parseBool(value);
                } else if (std.mem.eql(u8, key, "sigma")) {
                    dos_cfg.sigma = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "npoints")) {
                    dos_cfg.npoints = try floatToIndex(try parseFloat(value));
                } else if (std.mem.eql(u8, key, "emin")) {
                    dos_cfg.emin = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "emax")) {
                    dos_cfg.emax = try parseFloat(value);
                } else if (std.mem.eql(u8, key, "pdos")) {
                    dos_cfg.pdos = try parseBool(value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .output => {
                if (std.mem.eql(u8, key, "cube")) {
                    output_cfg.cube = try parseBool(value);
                } else {
                    return error.UnsupportedToml;
                }
            },
            .pseudopotential => {
                const idx = current_pseudo_index orelse return error.InvalidToml;
                if (std.mem.eql(u8, key, "element")) {
                    alloc.free(pseudo_list.items[idx].element);
                    pseudo_list.items[idx].element = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "path")) {
                    alloc.free(pseudo_list.items[idx].path);
                    pseudo_list.items[idx].path = try parseString(alloc, value);
                } else if (std.mem.eql(u8, key, "format")) {
                    const format_str = try parseString(alloc, value);
                    defer alloc.free(format_str);
                    pseudo_list.items[idx].format = try pseudo.parseFormat(format_str);
                } else if (std.mem.eql(u8, key, "core_energy_ry")) {
                    // Deprecated: epsatm is now auto-computed from UPF data.
                    // Accept and ignore for backward compatibility.
                    _ = try parseFloat(value);
                } else {
                    return error.UnsupportedToml;
                }
            },
        }
    }

    if (xyz_path == null) return error.MissingXyz;
    if (a1 == null or a2 == null or a3 == null) return error.MissingCell;

    for (pseudo_list.items) |p| {
        if (p.element.len == 0 or p.path.len == 0) return error.InvalidPseudopotential;
    }

    // Inherit top-level threads into scf/band/dfpt if not explicitly set
    if (!scf_kpoint_threads_explicit) {
        scf.kpoint_threads = top_threads;
    }
    if (!band_kpoint_threads_explicit) {
        band_kpoint_threads = top_threads;
    }
    if (!dfpt_kpoint_threads_explicit) {
        dfpt_cfg.kpoint_threads = top_threads;
    }
    // Auto-set lobpcg_parallel: true when threads==1 (single k-point thread, LOBPCG internal parallel)
    if (!band_lobpcg_parallel_explicit) {
        band_lobpcg_parallel = (if (band_kpoint_threads_explicit) band_kpoint_threads else top_threads) == 1;
    }

    const dfpt_qpath_slice = try dfpt_qpath_list.toOwnedSlice(alloc);
    dfpt_cfg.qpath = dfpt_qpath_slice;
    const cell = math.Mat3.fromRows(a1.?, a2.?, a3.?);
    const pseudo_slice = try pseudo_list.toOwnedSlice(alloc);
    return Config{
        .title = title,
        .xyz_path = xyz_path.?,
        .out_dir = out_dir,
        .units = units,
        .linalg_backend = linalg_backend,
        .threads = top_threads,
        .cell = cell,
        .boundary = boundary,
        .scf = scf,
        .ewald = ewald,
        .vdw = vdw_cfg,
        .relax = relax,
        .dfpt = dfpt_cfg,
        .dos = dos_cfg,
        .output = output_cfg,
        .band = .{
            .points_per_segment = band_points,
            .nbands = band_nbands,
            .path = &.{},
            .path_string = band_path_string,
            .solver = band_solver,
            .iterative_max_iter = band_iterative_max_iter,
            .iterative_tol = band_iterative_tol,
            .iterative_max_subspace = band_iterative_max_subspace,
            .iterative_block_size = band_iterative_block_size,
            .iterative_init_diagonal = band_iterative_init_diagonal,
            .kpoint_threads = band_kpoint_threads,
            .iterative_reuse_vectors = band_iterative_reuse_vectors,
            .use_symmetry = band_use_symmetry,
            .lobpcg_parallel = band_lobpcg_parallel,
        },
        .pseudopotentials = pseudo_slice,
    };
}

/// Remove comments while respecting quoted strings.
fn stripComment(line: []const u8) []const u8 {
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
fn parseString(alloc: std.mem.Allocator, value: []const u8) ![]u8 {
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
fn parseFloat(value: []const u8) !f64 {
    return try std.fmt.parseFloat(f64, value);
}

/// Parse a boolean value.
fn parseBool(value: []const u8) !bool {
    if (std.mem.eql(u8, value, "true")) return true;
    if (std.mem.eql(u8, value, "false")) return false;
    return error.InvalidBool;
}

/// Parse SCF solver mode.
fn parseScfSolver(value: []const u8) !ScfSolver {
    if (std.mem.eql(u8, value, "dense")) return .dense;
    if (std.mem.eql(u8, value, "iterative")) return .iterative;
    if (std.mem.eql(u8, value, "cg")) return .cg;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    return error.InvalidScfSolver;
}

fn parseXcFunctional(value: []const u8) !xc.Functional {
    if (std.mem.eql(u8, value, "lda")) return .lda_pz;
    if (std.mem.eql(u8, value, "lda_pz")) return .lda_pz;
    if (std.mem.eql(u8, value, "pbe")) return .pbe;
    return error.InvalidXcFunctional;
}

fn parseSmearingMethod(value: []const u8) !SmearingMethod {
    if (std.mem.eql(u8, value, "none")) return .none;
    if (std.mem.eql(u8, value, "fermi")) return .fermi_dirac;
    if (std.mem.eql(u8, value, "fermi_dirac")) return .fermi_dirac;
    return error.InvalidSmearingMethod;
}

fn parseConvergenceMetric(value: []const u8) !ConvergenceMetric {
    if (std.mem.eql(u8, value, "density")) return .density;
    if (std.mem.eql(u8, value, "potential")) return .potential;
    if (std.mem.eql(u8, value, "vresid")) return .potential;
    return error.InvalidConvergenceMetric;
}

fn parseLocalPotentialMode(value: []const u8) !LocalPotentialMode {
    if (std.mem.eql(u8, value, "tail")) return .tail;
    if (std.mem.eql(u8, value, "ewald")) return .ewald;
    if (std.mem.eql(u8, value, "short_range")) return .short_range;
    return error.InvalidLocalPotentialMode;
}

/// Parse band solver mode.
fn parseBandSolver(value: []const u8) !BandSolver {
    if (std.mem.eql(u8, value, "dense")) return .dense;
    if (std.mem.eql(u8, value, "iterative")) return .iterative;
    if (std.mem.eql(u8, value, "cg")) return .cg;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    return error.InvalidBandSolver;
}

/// Parse a three-element array into Vec3.
fn parseVec3(value: []const u8) !math.Vec3 {
    const vals = try parseArrayNumbers(value, 3);
    return .{ .x = vals[0], .y = vals[1], .z = vals[2] };
}

/// Parse a fixed-length numeric array.
fn parseArrayNumbers(value: []const u8, expected_len: usize) ![3]f64 {
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
fn parseFloatArray(alloc: std.mem.Allocator, value: []const u8) ![]f64 {
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
fn floatToIndex(value: f64) !usize {
    const rounded = std.math.floor(value + 0.5);
    if (rounded < 0.0) return error.InvalidArray;
    return @intFromFloat(rounded);
}

// --- Tests ---

/// Build a minimal valid Config for testing (all defaults are valid).
fn testDefaultConfig() Config {
    return .{
        .title = @constCast("test"),
        .xyz_path = @constCast("test.xyz"),
        .out_dir = @constCast("out"),
        .units = .angstrom,
        .linalg_backend = .openblas,
        .threads = 0,
        .cell = math.Mat3.fromRows(
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
        .pseudopotentials = @constCast(&[_]pseudo.Spec{.{ .element = @constCast("Si"), .path = @constCast("Si.upf"), .format = .upf }}),
    };
}

fn countBySeverity(issues: []const ValidationIssue, severity: ValidationSeverity) usize {
    var n: usize = 0;
    for (issues) |issue| {
        if (issue.severity == severity) n += 1;
    }
    return n;
}

test "validate: valid default config produces no issues" {
    const alloc = std.testing.allocator;
    const cfg = testDefaultConfig();
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), result.issues.len);
    try std.testing.expect(!result.hasErrors());
}

test "validate: ecut_ry = 0 is an error" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.ecut_ry = 0;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .err) >= 1);
}

test "validate: ecut_ry = 3.0 is a warning" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.ecut_ry = 3.0;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .warning) >= 1);
}

test "validate: mixing_beta out of range" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.mixing_beta = 1.5;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: smearing enabled with smear_ry = 0" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.smearing = .fermi_dirac;
    cfg.scf.smear_ry = 0.0;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: pulay_start exceeds pulay_history" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.pulay_history = 5;
    cfg.scf.pulay_start = 10;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: degenerate cell" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.cell = math.Mat3.fromRows(
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
    );
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: no pseudopotentials" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.pseudopotentials = &.{};
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: relax requires scf" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.relax.enabled = true;
    cfg.scf.enabled = false;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: isolated boundary with kmesh > 1 is a warning" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.boundary = .isolated;
    cfg.scf.kmesh = .{ 4, 4, 4 };
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .warning) >= 1);
}

test "validate: hint for dense solver" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.solver = .dense;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .hint) >= 1);
}

test "validate: hint for non-fftw backend" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.fft_backend = .zig;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .hint) >= 1);
}

test "validate: hint for diemac = 1.0 with tight convergence" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.diemac = 1.0;
    cfg.scf.convergence = 1e-8;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .hint) >= 1);
}

test "validate: no diemac hint with loose convergence" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.diemac = 1.0;
    cfg.scf.convergence = 1e-6; // loose enough that diemac doesn't matter
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
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
    var cfg = testDefaultConfig();
    cfg.scf.mixing_mode = .density;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .hint) >= 1);
}

test "validate: hint for tight iterative_tol" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.iterative_tol = 1e-8;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .hint) >= 1);
}

test "validate: hint for symmetry disabled" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.symmetry = false;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .hint) >= 1);
}

test "validate: no hints with recommended settings" {
    const alloc = std.testing.allocator;
    const cfg = testDefaultConfig(); // default uses recommended settings
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expectEqual(@as(usize, 0), countBySeverity(result.issues, .hint));
}

test "validate: vdw enabled with method none" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.vdw.enabled = true;
    cfg.vdw.method = .none;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: dfpt with smearing is error" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.dfpt.enabled = true;
    cfg.scf.smearing = .fermi_dirac;
    cfg.scf.smear_ry = 0.01;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: dfpt with nspin=2 is error" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.dfpt.enabled = true;
    cfg.scf.nspin = 2;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: band points_per_segment = 0 is error" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.band.path_string = @constCast("G-X");
    cfg.band.points_per_segment = 0;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(result.hasErrors());
}

test "validate: nspin=2 without spinat is warning" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.nspin = 2;
    cfg.scf.spinat = null;
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .warning) >= 1);
}

test "validate: grid too small for ecut is warning" {
    const alloc = std.testing.allocator;
    var cfg = testDefaultConfig();
    cfg.scf.ecut_ry = 60.0;
    cfg.scf.grid = .{ 8, 8, 8 }; // way too small for ecut=60
    var result = try cfg.validate(alloc);
    defer result.deinit();
    try std.testing.expect(!result.hasErrors());
    try std.testing.expect(countBySeverity(result.issues, .warning) >= 1);
}
