const std = @import("std");
const fft_sizing = @import("../../lib/fft/sizing.zig");
const math = @import("../math/math.zig");
const types = @import("types.zig");

const CellValidation = struct {
    cell_bohr: math.Mat3,
    volume: f64,
};

/// Validate config after parsing. Returns all issues (errors + warnings).
pub fn validate(cfg: anytype, alloc: std.mem.Allocator) !types.ValidationResult {
    var issues: std.ArrayList(types.ValidationIssue) = .empty;
    errdefer {
        for (issues.items) |issue| alloc.free(issue.message);
        issues.deinit(alloc);
    }

    const cell_validation = try validate_cell_geometry(cfg, alloc, &issues);
    try validate_pseudopotentials(cfg, alloc, &issues);
    try validate_scf_config(cfg, alloc, &issues);
    try validate_boundary_consistency(cfg, alloc, &issues);
    try validate_relax_config(cfg, alloc, &issues);
    try validate_dfpt_config(cfg, alloc, &issues);
    try validate_dos_config(cfg, alloc, &issues);
    try validate_vdw_config(cfg, alloc, &issues);
    try validate_band_config(cfg, alloc, &issues);
    try validate_spin_consistency(cfg, alloc, &issues);
    try validate_grid_recommendation(cfg, cell_validation, alloc, &issues);
    try add_validation_hints(cfg, alloc, &issues);

    return .{
        .issues = try issues.toOwnedSlice(alloc),
        .allocator = alloc,
    };
}

fn validate_cell_geometry(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !CellValidation {
    const cell_bohr = cfg.cell.scale(math.units_scale_to_bohr(cfg.units));
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
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.pseudopotentials.len != 0) return;
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
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    try validate_scf_basis(cfg, alloc, issues);
    try validate_scf_convergence_controls(cfg, alloc, issues);
    try validate_scf_mixing(cfg, alloc, issues);
    try validate_scf_smearing(cfg, alloc, issues);
}

fn validate_scf_basis(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.scf.ecut_ry <= 0) {
        try add_issue(
            alloc,
            issues,
            .err,
            "scf",
            "ecut_ry",
            try std.fmt.allocPrint(alloc, "must be positive (got {d:.2})", .{cfg.scf.ecut_ry}),
        );
    } else if (cfg.scf.ecut_ry < 5.0) {
        try add_issue(
            alloc,
            issues,
            .warning,
            "scf",
            "ecut_ry",
            try std.fmt.allocPrint(
                alloc,
                "{d:.1} is unusually low; typical range is 15-100 Ry",
                .{cfg.scf.ecut_ry},
            ),
        );
    }

    try validate_positive_mesh(alloc, issues, "scf", "kmesh", cfg.scf.kmesh);
    if (cfg.scf.grid_scale > 0) return;
    try add_issue_literal(alloc, issues, .err, "scf", "grid_scale", "must be positive");
}

fn validate_scf_convergence_controls(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.scf.max_iter == 0) {
        try add_issue_literal(alloc, issues, .err, "scf", "max_iter", "must be positive");
    }

    if (cfg.scf.convergence <= 0) {
        try add_issue_literal(alloc, issues, .err, "scf", "convergence", "must be positive");
        return;
    }

    if (cfg.scf.convergence <= 1e-3) return;
    try add_issue(
        alloc,
        issues,
        .warning,
        "scf",
        "convergence",
        try std.fmt.allocPrint(
            alloc,
            "{e} is loose; results may be inaccurate",
            .{cfg.scf.convergence},
        ),
    );
}

fn validate_scf_mixing(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.scf.mixing_beta <= 0 or cfg.scf.mixing_beta > 1.0) {
        try add_issue(
            alloc,
            issues,
            .err,
            "scf",
            "mixing_beta",
            try std.fmt.allocPrint(
                alloc,
                "must be in (0, 1] (got {d:.3})",
                .{cfg.scf.mixing_beta},
            ),
        );
    } else if (cfg.scf.mixing_beta > 0.7) {
        try add_issue(
            alloc,
            issues,
            .warning,
            "scf",
            "mixing_beta",
            try std.fmt.allocPrint(
                alloc,
                "{d:.2} is high; may cause SCF oscillation",
                .{cfg.scf.mixing_beta},
            ),
        );
    }

    if (cfg.scf.pulay_history == 0 or cfg.scf.pulay_start <= cfg.scf.pulay_history) return;
    try add_issue(
        alloc,
        issues,
        .err,
        "scf",
        "pulay_start",
        try std.fmt.allocPrint(
            alloc,
            "({d}) exceeds pulay_history ({d})",
            .{ cfg.scf.pulay_start, cfg.scf.pulay_history },
        ),
    );
}

fn validate_scf_smearing(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.scf.smearing != .none and cfg.scf.smear_ry <= 0) {
        try add_issue_literal(
            alloc,
            issues,
            .err,
            "scf",
            "smear_ry",
            "must be > 0 when smearing is enabled",
        );
    }
    if (cfg.scf.smearing == .none and cfg.scf.smear_ry > 0) {
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
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.boundary != .isolated) return;
    for (cfg.scf.kmesh) |value| {
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
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (!cfg.relax.enabled) return;
    if (!cfg.scf.enabled) {
        try add_issue_literal(alloc, issues, .err, "relax", "", "requires scf to be enabled");
    }
    if (cfg.relax.max_iter == 0) {
        try add_issue_literal(alloc, issues, .err, "relax", "max_iter", "must be positive");
    }
    if (cfg.relax.force_tol <= 0) {
        try add_issue_literal(alloc, issues, .err, "relax", "force_tol", "must be positive");
    }
    if (cfg.relax.max_step <= 0) {
        try add_issue_literal(alloc, issues, .err, "relax", "max_step", "must be positive");
    }
    if (cfg.relax.cell_relax and !cfg.scf.compute_stress) {
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
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (!cfg.dfpt.enabled) return;
    try validate_dfpt_requirements(cfg, alloc, issues);
    try validate_dfpt_controls(cfg, alloc, issues);
    try validate_dfpt_meshes(cfg, alloc, issues);
}

fn validate_dfpt_requirements(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (!cfg.scf.enabled) {
        try add_issue_literal(alloc, issues, .err, "dfpt", "", "requires scf to be enabled");
    }
    if (cfg.scf.smearing != .none) {
        try add_issue_literal(
            alloc,
            issues,
            .err,
            "dfpt",
            "",
            "DFPT is incompatible with Fermi-Dirac smearing",
        );
    }
    if (cfg.scf.nspin == 2) {
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
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.dfpt.sternheimer_max_iter == 0) {
        try add_issue_literal(
            alloc,
            issues,
            .err,
            "dfpt",
            "sternheimer_max_iter",
            "must be positive",
        );
    }
    if (cfg.dfpt.sternheimer_tol <= 0) {
        try add_issue_literal(
            alloc,
            issues,
            .err,
            "dfpt",
            "sternheimer_tol",
            "must be positive",
        );
    }
    if (cfg.dfpt.scf_max_iter == 0) {
        try add_issue_literal(alloc, issues, .err, "dfpt", "scf_max_iter", "must be positive");
    }
    if (cfg.dfpt.scf_tol <= 0) {
        try add_issue_literal(alloc, issues, .err, "dfpt", "scf_tol", "must be positive");
    }
    if (cfg.dfpt.mixing_beta <= 0 or cfg.dfpt.mixing_beta > 1.0) {
        try add_issue(
            alloc,
            issues,
            .err,
            "dfpt",
            "mixing_beta",
            try std.fmt.allocPrint(
                alloc,
                "must be in (0, 1] (got {d:.3})",
                .{cfg.dfpt.mixing_beta},
            ),
        );
    }
    if (cfg.dfpt.pulay_history == 0 or cfg.dfpt.pulay_start <= cfg.dfpt.pulay_history) {
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
            .{ cfg.dfpt.pulay_start, cfg.dfpt.pulay_history },
        ),
    );
}

fn validate_dfpt_meshes(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.dfpt.qgrid) |qgrid| {
        try validate_positive_mesh(alloc, issues, "dfpt", "qgrid", qgrid);
    }
    if (cfg.dfpt.dos_qmesh) |dos_qmesh| {
        try validate_positive_mesh(alloc, issues, "dfpt", "dos_qmesh", dos_qmesh);
    }
}

fn validate_dos_config(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (!cfg.dos.enabled) return;
    if (cfg.dos.sigma <= 0) {
        try add_issue_literal(alloc, issues, .err, "dos", "sigma", "must be positive");
    }
    if (cfg.dos.npoints == 0) {
        try add_issue_literal(alloc, issues, .err, "dos", "npoints", "must be positive");
    }
    if (cfg.dos.emin == null or cfg.dos.emax == null) return;
    if (cfg.dos.emin.? < cfg.dos.emax.?) return;
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
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (!cfg.vdw.enabled or cfg.vdw.method != .none) return;
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
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.band.path.len == 0 and cfg.band.path_string == null) return;
    if (cfg.band.points_per_segment > 0) return;
    try add_issue_literal(alloc, issues, .err, "band", "points", "points must be positive");
}

fn validate_spin_consistency(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.scf.nspin != 2 or cfg.scf.spinat != null) return;
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
    cfg: anytype,
    cell_validation: CellValidation,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.scf.grid[0] == 0 or cfg.scf.grid[1] == 0 or cfg.scf.grid[2] == 0) return;
    if (cfg.scf.ecut_ry <= 0 or @abs(cell_validation.volume) <= 1e-12) return;

    const recip = math.reciprocal(cell_validation.cell_bohr);
    const scale = if (cfg.scf.grid_scale > 0) cfg.scf.grid_scale else 1.0;
    const density_gmax = @max(2.0, scale) * @sqrt(cfg.scf.ecut_ry);
    const raw1 = @as(usize, @intFromFloat(
        std.math.ceil(density_gmax / math.Vec3.norm(recip.row(0))),
    )) * 2 + 1;
    const raw2 = @as(usize, @intFromFloat(
        std.math.ceil(density_gmax / math.Vec3.norm(recip.row(1))),
    )) * 2 + 1;
    const raw3 = @as(usize, @intFromFloat(
        std.math.ceil(density_gmax / math.Vec3.norm(recip.row(2))),
    )) * 2 + 1;
    if (cfg.scf.grid[0] >= raw1 and cfg.scf.grid[1] >= raw2 and cfg.scf.grid[2] >= raw3) {
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
                cfg.scf.grid[0],
                cfg.scf.grid[1],
                cfg.scf.grid[2],
                raw1,
                raw2,
                raw3,
                cfg.scf.ecut_ry,
                fft_sizing.next_fft_size(@max(raw1, 3)),
                fft_sizing.next_fft_size(@max(raw2, 3)),
                fft_sizing.next_fft_size(@max(raw3, 3)),
            },
        ),
    );
}

fn add_validation_hints(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    try add_scf_validation_hints(cfg, alloc, issues);
    try add_band_validation_hints(cfg, alloc, issues);
}

fn add_scf_validation_hints(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (!cfg.scf.enabled) return;
    try add_scf_hint_if(
        alloc,
        issues,
        cfg.scf.solver == .dense,
        "solver",
        "solver = \"dense\" is slow for large systems;" ++
            " consider solver = \"iterative\" (LOBPCG)",
    );
    try add_scf_hint_if(
        alloc,
        issues,
        cfg.scf.fft_backend != .fftw,
        "fft_backend",
        "fft_backend = \"fftw\" is recommended for production calculations",
    );
    try add_scf_hint_if(
        alloc,
        issues,
        cfg.scf.diemac == 1.0 and cfg.scf.convergence < 1e-6,
        "diemac",
        "diemac = 1.0 (disabled); for tight convergence (<1e-6)," ++
            " set ~12 for semiconductors or ~1e6 for metals",
    );
    try add_scf_hint_if(
        alloc,
        issues,
        cfg.scf.pulay_history == 0,
        "pulay_history",
        "pulay_history = 0 (DIIS disabled); pulay_history = 8 with pulay_start = 4" ++
            " typically improves SCF convergence",
    );
    try add_scf_hint_if(
        alloc,
        issues,
        cfg.scf.mixing_mode == .density,
        "mixing_mode",
        "mixing_mode = \"density\" can oscillate at large ecut;" ++
            " mixing_mode = \"potential\" is recommended",
    );
    try add_scf_hint_if(
        alloc,
        issues,
        cfg.scf.solver == .iterative and cfg.scf.iterative_tol < 1e-6,
        "iterative_tol",
        "iterative_tol < 1e-6 increases eigensolver cost per SCF iteration;" ++
            " 1e-4 is usually sufficient",
    );
    try add_scf_hint_if(
        alloc,
        issues,
        !cfg.scf.symmetry,
        "symmetry",
        "symmetry = false; enabling symmetry reduces k-points" ++
            " and speeds up the calculation",
    );
}

fn add_band_validation_hints(
    cfg: anytype,
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
) !void {
    if (cfg.band.path.len == 0 and cfg.band.path_string == null) return;
    if (cfg.band.solver != .dense) return;
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
    issues: *std.ArrayList(types.ValidationIssue),
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
    issues: *std.ArrayList(types.ValidationIssue),
    severity: types.ValidationSeverity,
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
    issues: *std.ArrayList(types.ValidationIssue),
    severity: types.ValidationSeverity,
    section: []const u8,
    field: []const u8,
    comptime message: []const u8,
) !void {
    const owned = try alloc.dupe(u8, message);
    errdefer alloc.free(owned);
    try add_issue(alloc, issues, severity, section, field, owned);
}

fn add_scf_hint_if(
    alloc: std.mem.Allocator,
    issues: *std.ArrayList(types.ValidationIssue),
    enabled: bool,
    field: []const u8,
    comptime message: []const u8,
) !void {
    if (!enabled) return;
    try add_issue_literal(alloc, issues, .hint, "scf", field, message);
}
