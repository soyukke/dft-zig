const std = @import("std");
const defaults = @import("defaults.zig");
const linalg = @import("../linalg/linalg.zig");
const math = @import("../math/math.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const runtime_logging = @import("../runtime/logging.zig");
const xc = @import("../xc/xc.zig");
const types = @import("types.zig");

const BandConfig = types.BandConfig;
const BandPathPoint = types.BandPathPoint;
const BandSolver = types.BandSolver;
const BoundaryCondition = types.BoundaryCondition;
const ConvergenceMetric = types.ConvergenceMetric;
const DfptConfig = types.DfptConfig;
const DosConfig = types.DosConfig;
const EwaldConfig = types.EwaldConfig;
const LocalPotentialMode = types.LocalPotentialMode;
const OutputConfig = types.OutputConfig;
const RelaxConfig = types.RelaxConfig;
const ScfConfig = types.ScfConfig;
const ScfSolver = types.ScfSolver;
const SmearingMethod = types.SmearingMethod;
const VdwConfig = types.VdwConfig;
const parse_fft_backend = types.parse_fft_backend;

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
            .linalg_backend = defaults.linalg_backend,
            .boundary = .periodic,
            .a1 = null,
            .a2 = null,
            .a3 = null,
            .scf = defaults.scf,
            .ewald = defaults.ewald,
            .vdw = .{},
            .band = defaults.band,
            .relax = defaults.relax,
            .dfpt = defaults.dfpt,
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
        if (try self.parse_scf_float_field(key, value)) return true;
        if (try self.parse_scf_index_field(key, value)) return true;
        if (std.mem.eql(u8, key, "kpoint_threads")) {
            self.scf.kpoint_threads = try float_to_index(try parse_float(value));
            self.scf_kpoint_threads_explicit = true;
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
        if (std.mem.eql(u8, key, "nspin")) {
            self.scf.nspin = try float_to_index(try parse_float(value));
            if (self.scf.nspin != 1 and self.scf.nspin != 2) return error.InvalidNspin;
            return true;
        }
        return false;
    }

    fn parse_scf_float_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "smear_ry")) {
            self.scf.smear_ry = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "mixing_beta")) {
            self.scf.mixing_beta = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "convergence")) {
            self.scf.convergence = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_tol")) {
            self.scf.iterative_tol = try parse_float(value);
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
        return false;
    }

    fn parse_scf_index_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "max_iter")) {
            self.scf.max_iter = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "iterative_max_iter")) {
            self.scf.iterative_max_iter = try float_to_index(try parse_float(value));
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
        if (std.mem.eql(u8, key, "pulay_history")) {
            self.scf.pulay_history = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "pulay_start")) {
            self.scf.pulay_start = try float_to_index(try parse_float(value));
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
        if (try self.parse_dfpt_bool_field(key, value)) return;
        if (try self.parse_dfpt_float_field(key, value)) return;
        if (try self.parse_dfpt_index_field(key, value)) return;
        if (try self.parse_dfpt_mesh_field(key, value)) return;
        if (std.mem.eql(u8, key, "kpoint_threads")) {
            self.dfpt.kpoint_threads = try float_to_index(try parse_float(value));
            self.dfpt_kpoint_threads_explicit = true;
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

    fn parse_dfpt_bool_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "enabled")) {
            self.dfpt.enabled = try parse_bool(value);
            return true;
        }
        if (std.mem.eql(u8, key, "compute_dielectric")) {
            self.dfpt.compute_dielectric = try parse_bool(value);
            return true;
        }
        return false;
    }

    fn parse_dfpt_float_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "sternheimer_tol")) {
            self.dfpt.sternheimer_tol = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "scf_tol")) {
            self.dfpt.scf_tol = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "mixing_beta")) {
            self.dfpt.mixing_beta = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "alpha_shift")) {
            self.dfpt.alpha_shift = try parse_float(value);
            return true;
        }
        if (std.mem.eql(u8, key, "dos_sigma")) {
            self.dfpt.dos_sigma = try parse_float(value);
            return true;
        }
        return false;
    }

    fn parse_dfpt_index_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "sternheimer_max_iter")) {
            self.dfpt.sternheimer_max_iter = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "scf_max_iter")) {
            self.dfpt.scf_max_iter = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "qpath_npoints")) {
            self.dfpt.qpath_npoints = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "pulay_history")) {
            self.dfpt.pulay_history = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "pulay_start")) {
            self.dfpt.pulay_start = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "perturbation_threads")) {
            self.dfpt.perturbation_threads = try float_to_index(try parse_float(value));
            return true;
        }
        if (std.mem.eql(u8, key, "dos_nbin")) {
            self.dfpt.dos_nbin = try float_to_index(try parse_float(value));
            return true;
        }
        return false;
    }

    fn parse_dfpt_mesh_field(self: *LoadState, key: []const u8, value: []const u8) !bool {
        if (std.mem.eql(u8, key, "qgrid")) {
            self.dfpt.qgrid = try parse_u3_array(value);
            return true;
        }
        if (std.mem.eql(u8, key, "dos_qmesh")) {
            self.dfpt.dos_qmesh = try parse_u3_array(value);
            return true;
        }
        return false;
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

    fn finalize(self: *LoadState, comptime ConfigType: type, alloc: std.mem.Allocator) !ConfigType {
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
pub fn load(
    comptime ConfigType: type,
    alloc: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !ConfigType {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1024 * 1024));
    defer alloc.free(content);

    var state = try LoadState.init(alloc);
    errdefer state.deinit(alloc);

    try process_load_lines(alloc, content, &state);
    return try state.finalize(ConfigType, alloc);
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

fn parse_u3_array(value: []const u8) ![3]usize {
    const vals = try parse_array_numbers(value, 3);
    return .{
        try float_to_index(vals[0]),
        try float_to_index(vals[1]),
        try float_to_index(vals[2]),
    };
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
