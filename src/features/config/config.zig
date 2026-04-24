const std = @import("std");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const types = @import("types.zig");
const validation = @import("validation.zig");
const parser = @import("parser.zig");

pub const BandPathPoint = types.BandPathPoint;
pub const BandConfig = types.BandConfig;
pub const ScfConfig = types.ScfConfig;
pub const MixingMode = types.MixingMode;
pub const ScfSolver = types.ScfSolver;
pub const BandSolver = types.BandSolver;
pub const SmearingMethod = types.SmearingMethod;
pub const ConvergenceMetric = types.ConvergenceMetric;
pub const LocalPotentialMode = types.LocalPotentialMode;
pub const BoundaryCondition = types.BoundaryCondition;
pub const boundary_condition_name = types.boundary_condition_name;
pub const FftBackend = types.FftBackend;
pub const parse_fft_backend = types.parse_fft_backend;
pub const fft_backend_name = types.fft_backend_name;
pub const scf_solver_name = types.scf_solver_name;
pub const band_solver_name = types.band_solver_name;
pub const xc_functional_name = types.xc_functional_name;
pub const smearing_name = types.smearing_name;
pub const convergence_metric_name = types.convergence_metric_name;
pub const local_potential_mode_name = types.local_potential_mode_name;
pub const EwaldConfig = types.EwaldConfig;
pub const VdwMethod = types.VdwMethod;
pub const vdw_method_name = types.vdw_method_name;
pub const VdwConfig = types.VdwConfig;
pub const RelaxAlgorithm = types.RelaxAlgorithm;
pub const relax_algorithm_name = types.relax_algorithm_name;
pub const DfptConfig = types.DfptConfig;
pub const DosConfig = types.DosConfig;
pub const OutputConfig = types.OutputConfig;
pub const RelaxConfig = types.RelaxConfig;
pub const ValidationSeverity = types.ValidationSeverity;
pub const ValidationIssue = types.ValidationIssue;
pub const ValidationResult = types.ValidationResult;

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
        return validation.validate(self, alloc);
    }
};

/// Load config from a minimal TOML subset.
pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Config {
    return parser.load(Config, alloc, io, path);
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
