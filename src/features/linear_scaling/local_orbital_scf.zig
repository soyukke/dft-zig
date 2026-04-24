const std = @import("std");

const density_grid = @import("density_grid.zig");
const density_matrix = @import("density_matrix.zig");
const hartree_xc = @import("hartree_xc.zig");
const local_orbital_hamiltonian = @import("local_orbital_hamiltonian.zig");
const local_orbital_nonlocal = @import("local_orbital_nonlocal.zig");
const local_orbital_potential = @import("local_orbital_potential.zig");
const ionic_potential = @import("ionic_potential.zig");
const local_orbital = @import("local_orbital.zig");
const neighbor_list = @import("neighbor_list.zig");
const reference = @import("reference.zig");
const sparse = @import("sparse.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const xc = @import("../xc/xc.zig");
const math = @import("../math/math.zig");

pub const ScfModelOptions = struct {
    sigma: f64,
    cutoff: f64,
    local_potential: f64,
    kinetic_scale: f64 = 1.0,
    iterations: usize,
    threshold: f64,
    electrons: ?f64 = null,
    matrix_threshold: f64 = 0.0,
};

pub const ScfModelResult = struct {
    overlap: sparse.CsrMatrix,
    hamiltonian: sparse.CsrMatrix,
    density: sparse.CsrMatrix,
    energy: f64,
    iterations: usize,

    pub fn deinit(self: *ScfModelResult, alloc: std.mem.Allocator) void {
        self.overlap.deinit(alloc);
        self.hamiltonian.deinit(alloc);
        self.density.deinit(alloc);
        self.* = undefined;
    }
};

pub const ScfGridOptions = struct {
    sigma: f64,
    cutoff: f64,
    grid: local_orbital_potential.PotentialGrid,
    max_iter: usize,
    density_tol: f64,
    electrons: f64,
    xc: xc.Functional,
    ionic: ?[]const f64 = null,
    kinetic_scale: f64 = 1.0,
    matrix_threshold: f64 = 0.0,
    purification_iters: usize = 3,
    purification_threshold: f64 = 0.0,
    /// Ion sites for nonlocal pseudopotential (KB projectors)
    nonlocal_ions: ?[]const ionic_potential.IonSite = null,
    /// Nonlocal potential calculation options
    nonlocal_n_radial: usize = 100,
    nonlocal_r_max: f64 = 10.0,
    nonlocal_threshold: f64 = 0.0,
    /// Basis type for nonlocal: s_only or sp (includes p-orbitals)
    nonlocal_basis: local_orbital.BasisType = .s_only,
};

pub const ScfGridResult = struct {
    overlap: sparse.CsrMatrix,
    hamiltonian: sparse.CsrMatrix,
    density: sparse.CsrMatrix,
    energy: f64,
    energy_hartree: f64,
    energy_xc: f64,
    energy_vxc_rho: f64,
    energy_nonlocal: f64,
    iterations: usize,
    converged: bool,

    pub fn deinit(self: *ScfGridResult, alloc: std.mem.Allocator) void {
        self.overlap.deinit(alloc);
        self.hamiltonian.deinit(alloc);
        self.density.deinit(alloc);
        self.* = undefined;
    }
};

pub const ModelReference = struct {
    energy: f64,
    density: []f64,

    pub fn deinit(self: *ModelReference, alloc: std.mem.Allocator) void {
        if (self.density.len > 0) {
            alloc.free(self.density);
        }
        self.* = undefined;
    }

    pub fn as_reference_data(self: *const ModelReference) reference.ReferenceData {
        return .{ .energy = self.energy, .density = self.density };
    }
};

const ScfStepHamiltonian = struct {
    hamiltonian: sparse.CsrMatrix,
    energy_hartree: f64,
    energy_xc: f64,
    energy_vxc_rho: f64,
};

const LocalPotentialMatrix = struct {
    matrix: sparse.CsrMatrix,
    energy_hartree: f64,
    energy_xc: f64,
    energy_vxc_rho: f64,
};

const ScfGridState = struct {
    overlap: sparse.CsrMatrix,
    density: sparse.CsrMatrix,
    current_h: sparse.CsrMatrix,
};

fn validate_scf_grid_inputs(
    centers: []const math.Vec3,
    opts: ScfGridOptions,
) !void {
    if (centers.len == 0) return error.InvalidShape;
    const grid_count = opts.grid.count();
    if (grid_count == 0) return error.InvalidGrid;
    if (opts.ionic) |ionic| {
        if (ionic.len != grid_count) return error.InvalidGrid;
    }
}

fn build_scaled_kinetic_matrix(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    sigma: f64,
    cutoff: f64,
    pbc: neighbor_list.Pbc,
    cell: math.Mat3,
    kinetic_scale: f64,
) !sparse.CsrMatrix {
    var kinetic = try local_orbital.build_kinetic_csr_from_centers(
        alloc,
        centers,
        sigma,
        cutoff,
        pbc,
        cell,
    );
    if (kinetic_scale != 1.0) {
        sparse.scale_in_place(&kinetic, kinetic_scale);
    }
    return kinetic;
}

fn build_nonlocal_matrix(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: ScfGridOptions,
) !?sparse.CsrMatrix {
    const ions = opts.nonlocal_ions orelse return null;
    const nl_opts = local_orbital_nonlocal.NonlocalOptions{
        .sigma = opts.sigma,
        .cutoff = opts.cutoff,
        .n_radial = opts.nonlocal_n_radial,
        .r_max = opts.nonlocal_r_max,
        .threshold = opts.nonlocal_threshold,
        .basis = opts.nonlocal_basis,
    };
    return try local_orbital_nonlocal.build_nonlocal_csr(
        alloc,
        centers,
        ions,
        cell,
        pbc,
        nl_opts,
    );
}

fn build_local_potential_matrix(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    pbc: neighbor_list.Pbc,
    opts: ScfGridOptions,
    density: sparse.CsrMatrix,
) !LocalPotentialMatrix {
    const rho = try density_grid.build_density_grid_from_centers(
        alloc,
        centers,
        density,
        opts.sigma,
        opts.cutoff,
        pbc,
        opts.grid,
    );
    defer alloc.free(rho);

    var local = try hartree_xc.build_local_potential_grid(
        alloc,
        opts.grid,
        rho,
        opts.xc,
        opts.ionic,
    );
    defer local.deinit(alloc);

    const local_grid = local_orbital_potential.PotentialGrid{
        .cell = opts.grid.cell,
        .dims = opts.grid.dims,
        .values = local.values,
    };
    return .{
        .matrix = try local_orbital_potential.build_local_potential_csr_from_centers(
            alloc,
            centers,
            opts.sigma,
            opts.cutoff,
            pbc,
            local_grid,
        ),
        .energy_hartree = local.energy_hartree,
        .energy_xc = local.energy_xc,
        .energy_vxc_rho = local.energy_vxc_rho,
    };
}

fn build_scf_step_hamiltonian(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    pbc: neighbor_list.Pbc,
    opts: ScfGridOptions,
    density: sparse.CsrMatrix,
    kinetic: sparse.CsrMatrix,
    nonlocal: ?sparse.CsrMatrix,
) !ScfStepHamiltonian {
    var local = try build_local_potential_matrix(alloc, centers, pbc, opts, density);
    defer local.matrix.deinit(alloc);

    var h_kinetic_local = try sparse.add_scaled(
        alloc,
        kinetic,
        1.0,
        local.matrix,
        1.0,
        opts.matrix_threshold,
    );
    defer h_kinetic_local.deinit(alloc);

    return .{
        .hamiltonian = if (nonlocal) |nl|
            try sparse.add_scaled(
                alloc,
                h_kinetic_local,
                1.0,
                nl,
                1.0,
                opts.matrix_threshold,
            )
        else
            try sparse.clone(alloc, h_kinetic_local),
        .energy_hartree = local.energy_hartree,
        .energy_xc = local.energy_xc,
        .energy_vxc_rho = local.energy_vxc_rho,
    };
}

fn density_matrix_converged(
    alloc: std.mem.Allocator,
    density: sparse.CsrMatrix,
    next_density: sparse.CsrMatrix,
    density_tol: f64,
) !bool {
    const diag_old = try sparse.diagonal_values(alloc, density);
    defer alloc.free(diag_old);

    const diag_new = try sparse.diagonal_values(alloc, next_density);
    defer alloc.free(diag_new);

    var max_diff: f64 = 0.0;
    for (diag_old, 0..) |value, idx| {
        const diff = @abs(diag_new[idx] - value);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff < density_tol;
}

fn build_scf_grid_result(
    overlap: sparse.CsrMatrix,
    hamiltonian: sparse.CsrMatrix,
    density: sparse.CsrMatrix,
    energy: f64,
    energy_hartree: f64,
    energy_xc: f64,
    energy_vxc_rho: f64,
    energy_nonlocal: f64,
    iterations: usize,
    converged: bool,
) ScfGridResult {
    return .{
        .overlap = overlap,
        .hamiltonian = hamiltonian,
        .density = density,
        .energy = energy,
        .energy_hartree = energy_hartree,
        .energy_xc = energy_xc,
        .energy_vxc_rho = energy_vxc_rho,
        .energy_nonlocal = energy_nonlocal,
        .iterations = iterations,
        .converged = converged,
    };
}

fn init_scf_grid_state(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: ScfGridOptions,
    kinetic: sparse.CsrMatrix,
) !ScfGridState {
    var overlap = try local_orbital.build_overlap_csr_from_centers(
        alloc,
        centers,
        opts.sigma,
        opts.cutoff,
        pbc,
        cell,
    );
    errdefer overlap.deinit(alloc);

    var density = try density_matrix.density_from_hamiltonian(
        alloc,
        kinetic,
        overlap,
        opts.electrons,
        opts.purification_iters,
        opts.purification_threshold,
    );
    errdefer density.deinit(alloc);

    var current_h = try sparse.clone(alloc, kinetic);
    errdefer current_h.deinit(alloc);

    return .{
        .overlap = overlap,
        .density = density,
        .current_h = current_h,
    };
}

fn build_next_density(
    alloc: std.mem.Allocator,
    hamiltonian: sparse.CsrMatrix,
    overlap: sparse.CsrMatrix,
    opts: ScfGridOptions,
) !sparse.CsrMatrix {
    return try density_matrix.density_from_hamiltonian(
        alloc,
        hamiltonian,
        overlap,
        opts.electrons,
        opts.purification_iters,
        opts.purification_threshold,
    );
}

fn nonlocal_trace_product(
    nonlocal: ?sparse.CsrMatrix,
    density: sparse.CsrMatrix,
) !f64 {
    return if (nonlocal) |nl|
        try sparse.trace_product(nl, density)
    else
        0.0;
}

pub fn build_scf_model_from_centers(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: ScfModelOptions,
) !ScfModelResult {
    if (centers.len == 0) return error.InvalidShape;
    const ham_opts = local_orbital_hamiltonian.HamiltonianOptions{
        .sigma = opts.sigma,
        .cutoff = opts.cutoff,
        .local_potential = opts.local_potential,
        .kinetic_scale = opts.kinetic_scale,
        .threshold = opts.matrix_threshold,
    };
    var hamiltonian = try local_orbital_hamiltonian.build_hamiltonian_from_centers(
        alloc,
        centers,
        cell,
        pbc,
        ham_opts,
    );
    errdefer hamiltonian.deinit(alloc);

    var density = try density_matrix.mc_weeny_non_orthogonal(
        alloc,
        hamiltonian.overlap,
        hamiltonian.overlap,
        opts.iterations,
        opts.threshold,
    );
    errdefer density.deinit(alloc);
    if (opts.electrons) |target| {
        try density_matrix.normalize_trace_overlap(&density, hamiltonian.overlap, target);
    }
    const energy = try sparse.trace_product(hamiltonian.hamiltonian, density);
    return .{
        .overlap = hamiltonian.overlap,
        .hamiltonian = hamiltonian.hamiltonian,
        .density = density,
        .energy = energy,
        .iterations = opts.iterations,
    };
}

pub fn run_scf_with_grid(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: ScfGridOptions,
) !ScfGridResult {
    try validate_scf_grid_inputs(centers, opts);

    var kinetic = try build_scaled_kinetic_matrix(
        alloc,
        centers,
        opts.sigma,
        opts.cutoff,
        pbc,
        cell,
        opts.kinetic_scale,
    );
    defer kinetic.deinit(alloc);

    var nonlocal = try build_nonlocal_matrix(alloc, centers, cell, pbc, opts);
    defer if (nonlocal) |*nl| nl.deinit(alloc);

    var state = try init_scf_grid_state(alloc, centers, cell, pbc, opts, kinetic);
    errdefer {
        state.overlap.deinit(alloc);
        state.density.deinit(alloc);
        state.current_h.deinit(alloc);
    }

    var energy_hartree: f64 = 0.0;
    var energy_xc: f64 = 0.0;
    var energy_vxc_rho: f64 = 0.0;
    var converged = false;
    var iter: usize = 0;
    while (iter < opts.max_iter) : (iter += 1) {
        var step = try build_scf_step_hamiltonian(
            alloc,
            centers,
            pbc,
            opts,
            state.density,
            kinetic,
            nonlocal,
        );
        defer step.hamiltonian.deinit(alloc);

        energy_hartree = step.energy_hartree;
        energy_xc = step.energy_xc;
        energy_vxc_rho = step.energy_vxc_rho;

        var next_density = try build_next_density(alloc, step.hamiltonian, state.overlap, opts);
        errdefer next_density.deinit(alloc);

        converged = try density_matrix_converged(
            alloc,
            state.density,
            next_density,
            opts.density_tol,
        );

        state.density.deinit(alloc);
        state.density = next_density;
        state.current_h.deinit(alloc);
        state.current_h = try sparse.clone(alloc, step.hamiltonian);

        if (converged) break;
    }

    const energy = try sparse.trace_product(state.current_h, state.density);
    const energy_nonlocal = try nonlocal_trace_product(nonlocal, state.density);
    return build_scf_grid_result(
        state.overlap,
        state.current_h,
        state.density,
        energy,
        energy_hartree,
        energy_xc,
        energy_vxc_rho,
        energy_nonlocal,
        iter,
        converged,
    );
}

pub fn run_scf_with_grid_and_ions(
    alloc: std.mem.Allocator,
    centers: []const math.Vec3,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    ions: []const ionic_potential.IonSite,
    opts: ScfGridOptions,
) !ScfGridResult {
    const ionic = try ionic_potential.build_ionic_potential_grid(alloc, opts.grid, ions, pbc);
    defer alloc.free(ionic);

    var updated = opts;
    updated.ionic = ionic;
    // Pass ions for nonlocal potential if not already specified
    if (updated.nonlocal_ions == null) {
        updated.nonlocal_ions = ions;
    }
    return run_scf_with_grid(alloc, centers, cell, pbc, updated);
}

pub fn build_model_reference(
    alloc: std.mem.Allocator,
    model: *const ScfModelResult,
) !ModelReference {
    const density = try sparse.diagonal_values(alloc, model.density);
    return .{ .energy = model.energy, .density = density };
}

pub fn compare_model_to_reference(
    alloc: std.mem.Allocator,
    model: *const ScfModelResult,
    reference_data: reference.ReferenceData,
) !reference.ComparisonReport {
    var model_ref = try build_model_reference(alloc, model);
    defer model_ref.deinit(alloc);

    return reference.compare_reference(reference_data, model_ref.as_reference_data());
}

test "scf model returns normalized density" {
    const alloc = std.testing.allocator;
    const centers = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.5, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.5, .z = 0.0 },
    };
    const cell = math.Mat3.from_rows(
        .{ .x = 4.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 4.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 4.0 },
    );
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const opts = ScfModelOptions{
        .sigma = 0.5,
        .cutoff = 2.0,
        .local_potential = 0.1,
        .kinetic_scale = 1.0,
        .iterations = 2,
        .threshold = 0.0,
        .electrons = 2.0,
        .matrix_threshold = 0.0,
    };
    var result = try build_scf_model_from_centers(alloc, centers[0..], cell, pbc, opts);
    defer result.deinit(alloc);

    const trace_val = try density_matrix.trace_overlap(result.density, result.overlap);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), trace_val, 1e-8);
    try std.testing.expect(std.math.isFinite(result.energy));
}

test "model reference compares to itself" {
    const alloc = std.testing.allocator;
    const centers = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.5, .y = 0.0, .z = 0.0 },
    };
    const cell = math.Mat3.from_rows(
        .{ .x = 4.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 4.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 4.0 },
    );
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const opts = ScfModelOptions{
        .sigma = 0.5,
        .cutoff = 2.0,
        .local_potential = 0.2,
        .kinetic_scale = 1.0,
        .iterations = 2,
        .threshold = 0.0,
        .electrons = 2.0,
        .matrix_threshold = 0.0,
    };
    var result = try build_scf_model_from_centers(alloc, centers[0..], cell, pbc, opts);
    defer result.deinit(alloc);

    var model_ref = try build_model_reference(alloc, &result);
    defer model_ref.deinit(alloc);

    const report = try reference.compare_reference(
        model_ref.as_reference_data(),
        model_ref.as_reference_data(),
    );
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), report.energy.abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), report.density.max_abs, 1e-12);
}

test "run_scf_with_grid produces finite energy" {
    const alloc = std.testing.allocator;
    const centers = [_]math.Vec3{.{ .x = 1.5, .y = 1.5, .z = 1.5 }};
    const cell = math.Mat3.from_rows(
        .{ .x = 4.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 4.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 4.0 },
    );
    const grid = local_orbital_potential.PotentialGrid{
        .cell = cell,
        .dims = [3]usize{ 6, 6, 6 },
        .values = &[_]f64{},
    };
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const opts = ScfGridOptions{
        .sigma = 0.8,
        .cutoff = 3.0,
        .grid = grid,
        .max_iter = 1,
        .density_tol = 1e6,
        .electrons = 1.0,
        .xc = .lda_pz,
        .ionic = null,
        .kinetic_scale = 1.0,
        .matrix_threshold = 0.0,
        .purification_iters = 2,
        .purification_threshold = 0.0,
    };
    var result = try run_scf_with_grid(alloc, centers[0..], cell, pbc, opts);
    defer result.deinit(alloc);

    const trace_val = try density_matrix.trace_overlap(result.density, result.overlap);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), trace_val, 1e-8);
    try std.testing.expect(std.math.isFinite(result.energy));
    try std.testing.expect(result.converged);
}

test "run_scf_with_grid_and_ions shifts energy" {
    const alloc = std.testing.allocator;
    const centers = [_]math.Vec3{.{ .x = 1.5, .y = 1.5, .z = 1.5 }};
    const cell = math.Mat3.from_rows(
        .{ .x = 4.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 4.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 4.0 },
    );
    const grid = local_orbital_potential.PotentialGrid{
        .cell = cell,
        .dims = [3]usize{ 6, 6, 6 },
        .values = &[_]f64{},
    };
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const opts = ScfGridOptions{
        .sigma = 0.8,
        .cutoff = 3.0,
        .grid = grid,
        .max_iter = 1,
        .density_tol = 1e6,
        .electrons = 1.0,
        .xc = .lda_pz,
        .ionic = null,
        .kinetic_scale = 1.0,
        .matrix_threshold = 0.0,
        .purification_iters = 2,
        .purification_threshold = 0.0,
    };
    var baseline = try run_scf_with_grid(alloc, centers[0..], cell, pbc, opts);
    defer baseline.deinit(alloc);

    var r = try alloc.alloc(f64, 2);
    defer alloc.free(r);

    r[0] = 0.0;
    r[1] = 1.0;
    var rab = try alloc.alloc(f64, 2);
    defer alloc.free(rab);

    rab[0] = 0.0;
    rab[1] = 1.0;
    var v_local = try alloc.alloc(f64, 2);
    defer alloc.free(v_local);

    v_local[0] = 0.3;
    v_local[1] = 0.3;
    const upf = pseudo.UpfData{
        .r = r,
        .rab = rab,
        .v_local = v_local,
        .beta = &[_]pseudo.Beta{},
        .dij = &[_]f64{},
        .qij = &[_]f64{},
        .nlcc = &[_]f64{},
    };
    const ions = [_]ionic_potential.IonSite{.{ .position = centers[0], .upf = &upf }};
    var with_ions = try run_scf_with_grid_and_ions(alloc, centers[0..], cell, pbc, ions[0..], opts);
    defer with_ions.deinit(alloc);

    const diff = @abs(with_ions.energy - baseline.energy);
    try std.testing.expect(diff > 1e-4);
}
