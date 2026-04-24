const std = @import("std");
const builtin = @import("builtin");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const runtime_logging = @import("../runtime/logging.zig");
const xc = @import("../xc/xc.zig");

pub const BandPathPoint = struct {
    label: []u8,
    k: math.Vec3,
};

pub const BandConfig = struct {
    points_per_segment: usize,
    nbands: usize,
    path: []BandPathPoint,
    path_string: ?[]u8 = null,
    solver: BandSolver,
    iterative_max_iter: usize,
    iterative_tol: f64,
    iterative_max_subspace: usize,
    iterative_block_size: usize,
    iterative_init_diagonal: bool,
    kpoint_threads: usize,
    iterative_reuse_vectors: bool,
    use_symmetry: bool,
    lobpcg_parallel: bool,
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
    kerker_q0: f64,
    diemac: f64,
    dielng: f64,
    pulay_history: usize,
    pulay_start: usize,
    mixing_mode: MixingMode,
    use_rfft: bool,
    fft_backend: FftBackend,
    nspin: usize,
    spinat: ?[]f64,
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
    cg,
    auto,
};

pub const BandSolver = enum {
    dense,
    iterative,
    cg,
    auto,
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
    periodic,
    isolated,
};

pub fn boundary_condition_name(bc: BoundaryCondition) []const u8 {
    return switch (bc) {
        .periodic => "periodic",
        .isolated => "isolated",
    };
}

pub const FftBackend = enum {
    zig,
    zig_parallel,
    zig_transpose,
    zig_comptime24,
    vdsp,
    fftw,
    metal,
};

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
    cutoff_radius: f64 = 95.0,
    cn_cutoff: f64 = 40.0,
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
    qpath_npoints: usize,
    pulay_history: usize,
    pulay_start: usize,
    kpoint_threads: usize,
    perturbation_threads: usize,
    qgrid: ?[3]usize,
    qpath: []BandPathPoint,
    dos_qmesh: ?[3]usize = null,
    dos_sigma: f64 = 5.0,
    dos_nbin: usize = 500,
    compute_dielectric: bool = false,
    log_level: runtime_logging.Level = .info,
};

pub const DosConfig = struct {
    enabled: bool = false,
    sigma: f64 = 0.01,
    npoints: usize = 1001,
    emin: ?f64 = null,
    emax: ?f64 = null,
    pdos: bool = false,
};

pub const OutputConfig = struct {
    cube: bool = false,
};

pub const RelaxConfig = struct {
    enabled: bool,
    algorithm: RelaxAlgorithm,
    max_iter: usize,
    force_tol: f64,
    max_step: f64,
    output_trajectory: bool,
    cell_relax: bool = false,
    stress_tol: f64 = 0.5,
    cell_step: f64 = 0.01,
    target_pressure: f64 = 0.0,
};

pub const ValidationSeverity = enum {
    err,
    warning,
    hint,
};

pub const ValidationIssue = struct {
    severity: ValidationSeverity,
    section: []const u8,
    field: []const u8,
    message: []const u8,
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
