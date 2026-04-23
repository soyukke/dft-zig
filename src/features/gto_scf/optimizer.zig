//! Geometry optimization for GTO-based Hartree-Fock calculations.
//!
//! Implements the BFGS quasi-Newton method to find equilibrium nuclear
//! positions by minimizing the total RHF energy.  At each step:
//!   1. Run SCF at the current geometry.
//!   2. Compute analytical RHF gradient (dE/dR).
//!   3. Update the inverse Hessian approximation (BFGS formula).
//!   4. Take a step: R_new = R_old - H_inv * grad (with optional line search).
//!   5. Check convergence (gradient RMS, max gradient, step size).
//!
//! Units: Hartree atomic units throughout (energy in Ha, distance in Bohr).

const std = @import("std");
const math = @import("../math/math.zig");
const Vec3 = math.Vec3;
const basis_mod = @import("../basis/basis.zig");
const ContractedShell = basis_mod.ContractedShell;
const gto_scf = @import("gto_scf.zig");
const ScfParams = gto_scf.ScfParams;
const ScfResult = gto_scf.ScfResult;
const gradient_mod = gto_scf.gradient;
const GradientResult = gradient_mod.GradientResult;
const kohn_sham = gto_scf.kohn_sham;
const KsParams = kohn_sham.KsParams;
const KsResult = kohn_sham.KsResult;
const XcFunctional = kohn_sham.XcFunctional;
const logging = @import("logging.zig");
const becke = @import("../grid/becke.zig");
const GridPoint = becke.GridPoint;

/// Parameters controlling the geometry optimization.
pub const OptParams = struct {
    /// Maximum number of geometry optimization steps.
    max_steps: usize = 100,
    /// Convergence threshold for RMS gradient (Ha/Bohr).
    grad_rms_threshold: f64 = 3e-4,
    /// Convergence threshold for max absolute gradient component (Ha/Bohr).
    grad_max_threshold: f64 = 4.5e-4,
    /// Convergence threshold for RMS step size (Bohr).
    step_rms_threshold: f64 = 1.2e-3,
    /// Convergence threshold for max step component (Bohr).
    step_max_threshold: f64 = 1.8e-3,
    /// Maximum step size for trust region (Bohr).
    max_step_size: f64 = 0.3,
    /// SCF parameters to use at each geometry.
    scf_params: ScfParams = .{},
    /// Print progress information.
    print_progress: bool = true,
};

/// Result of a geometry optimization.
pub const OptResult = struct {
    /// Optimized nuclear positions (Bohr).
    positions: []Vec3,
    /// Final total energy (Ha).
    energy: f64,
    /// Final gradient (Ha/Bohr).
    final_gradient: []Vec3,
    /// Number of optimization steps taken.
    steps: usize,
    /// Whether the optimization converged.
    converged: bool,
    /// Energy at each step.
    energy_history: []f64,

    pub fn deinit(self: *OptResult, alloc: std.mem.Allocator) void {
        if (self.positions.len > 0) alloc.free(self.positions);
        if (self.final_gradient.len > 0) alloc.free(self.final_gradient);
        if (self.energy_history.len > 0) alloc.free(self.energy_history);
    }
};

/// Compute the RMS of a flat f64 array.
fn vecRms(v: []const f64) f64 {
    if (v.len == 0) return 0.0;
    var sum: f64 = 0.0;
    for (v) |x| {
        sum += x * x;
    }
    return @sqrt(sum / @as(f64, @floatFromInt(v.len)));
}

/// Compute the max absolute value in a flat f64 array.
fn vecMaxAbs(v: []const f64) f64 {
    var mx: f64 = 0.0;
    for (v) |x| {
        const a = @abs(x);
        if (a > mx) mx = a;
    }
    return mx;
}

/// Scale a vector in-place if its norm exceeds `max_norm`, preserving direction.
fn trustRegionScale(step: []f64, max_norm: f64) void {
    var norm_sq: f64 = 0.0;
    for (step) |s| {
        norm_sq += s * s;
    }
    const norm = @sqrt(norm_sq);
    if (norm > max_norm) {
        const scale = max_norm / norm;
        for (step) |*s| {
            s.* *= scale;
        }
    }
}

/// Flatten an array of Vec3 into a contiguous f64 array [x0,y0,z0,x1,y1,z1,...].
fn flattenVec3(alloc: std.mem.Allocator, vecs: []const Vec3) ![]f64 {
    const n = vecs.len * 3;
    const flat = try alloc.alloc(f64, n);
    for (vecs, 0..) |v, i| {
        flat[i * 3 + 0] = v.x;
        flat[i * 3 + 1] = v.y;
        flat[i * 3 + 2] = v.z;
    }
    return flat;
}

/// Unflatten a contiguous f64 array back to Vec3 array.
fn unflattenToVec3(alloc: std.mem.Allocator, flat: []const f64) ![]Vec3 {
    const n_atoms = flat.len / 3;
    const vecs = try alloc.alloc(Vec3, n_atoms);
    for (0..n_atoms) |i| {
        vecs[i] = .{
            .x = flat[i * 3 + 0],
            .y = flat[i * 3 + 1],
            .z = flat[i * 3 + 2],
        };
    }
    return vecs;
}

/// Update shell centers to match new nuclear positions.
/// Each shell's center is set to the position of its parent atom.
fn updateShellCenters(
    shells: []ContractedShell,
    shell_to_atom: []const usize,
    new_positions: []const Vec3,
) void {
    for (shells, 0..) |*shell, i| {
        shell.center = new_positions[shell_to_atom[i]];
    }
}

/// Build a map from shell index to atom index.
/// This identifies which atom each shell belongs to by matching
/// the shell center to the closest nuclear position.
fn buildShellToAtomMap(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
    nuc_positions: []const Vec3,
) ![]usize {
    const map = try alloc.alloc(usize, shells.len);
    for (shells, 0..) |shell, i| {
        var best_atom: usize = 0;
        var best_dist: f64 = std.math.inf(f64);
        for (nuc_positions, 0..) |pos, a| {
            const dx = shell.center.x - pos.x;
            const dy = shell.center.y - pos.y;
            const dz = shell.center.z - pos.z;
            const dist = dx * dx + dy * dy + dz * dz;
            if (dist < best_dist) {
                best_dist = dist;
                best_atom = a;
            }
        }
        map[i] = best_atom;
    }
    return map;
}

/// Initialize the inverse Hessian to a scaled identity matrix.
/// Uses a diagonal value of 1/70 Ha/Bohr^2, which corresponds to
/// a typical bond stretch frequency of ~70 Ha/Bohr^2.
fn initInverseHessian(alloc: std.mem.Allocator, n: usize) ![]f64 {
    const h_inv = try alloc.alloc(f64, n * n);
    @memset(h_inv, 0.0);
    const diag_val: f64 = 1.0 / 70.0; // ~ reciprocal of typical Hessian eigenvalue
    for (0..n) |i| {
        h_inv[i * n + i] = diag_val;
    }
    return h_inv;
}

/// Perform BFGS update of the inverse Hessian approximation.
/// H_inv_{k+1} = (I - rho*s*y^T) * H_inv_k * (I - rho*y*s^T) + rho*s*s^T
/// where s = x_{k+1} - x_k, y = g_{k+1} - g_k, rho = 1/(y^T s).
fn bfgsUpdate(
    alloc: std.mem.Allocator,
    h_inv: []f64,
    s_vec: []const f64,
    y_vec: []const f64,
    n: usize,
) !void {
    // Compute y^T * s
    var ys: f64 = 0.0;
    for (0..n) |i| {
        ys += y_vec[i] * s_vec[i];
    }

    // Skip update if curvature condition not satisfied
    if (ys < 1e-10) return;

    const rho = 1.0 / ys;

    // Compute H_inv * y
    const hy = try alloc.alloc(f64, n);
    defer alloc.free(hy);

    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (0..n) |j| {
            sum += h_inv[i * n + j] * y_vec[j];
        }
        hy[i] = sum;
    }

    // Compute y^T * H_inv * y
    var yhy: f64 = 0.0;
    for (0..n) |i| {
        yhy += y_vec[i] * hy[i];
    }

    // BFGS update: H_inv += (ys + yHy) * rho^2 * (s s^T) - rho * (Hy s^T + s Hy^T)
    const factor = (ys + yhy) * rho * rho;
    for (0..n) |i| {
        for (0..n) |j| {
            h_inv[i * n + j] += factor * s_vec[i] * s_vec[j] -
                rho * (hy[i] * s_vec[j] + s_vec[i] * hy[j]);
        }
    }
}

/// Matrix-vector multiply: result = mat * vec.
fn matVecMul(alloc: std.mem.Allocator, mat: []const f64, vec: []const f64, n: usize) ![]f64 {
    const result = try alloc.alloc(f64, n);
    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (0..n) |j| {
            sum += mat[i * n + j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

/// Apply positions from flat `coords` to `nuc_positions` and update shell centers.
fn applyCoordsToPositions(
    coords: []const f64,
    nuc_positions: []Vec3,
    shells: []ContractedShell,
    shell_to_atom: []const usize,
) void {
    const n_atoms = nuc_positions.len;
    for (0..n_atoms) |i| {
        nuc_positions[i] = .{
            .x = coords[i * 3 + 0],
            .y = coords[i * 3 + 1],
            .z = coords[i * 3 + 2],
        };
    }
    updateShellCenters(shells, shell_to_atom, nuc_positions);
}

/// Flatten gradient Vec3 array into the provided `out` buffer.
fn flattenGradientInto(grad: []const Vec3, out: []f64) void {
    for (grad, 0..) |g, i| {
        out[i * 3 + 0] = g.x;
        out[i * 3 + 1] = g.y;
        out[i * 3 + 2] = g.z;
    }
}

/// Apply BFGS update, compute search direction, save history, and update coords.
fn applyBfgsStep(
    alloc: std.mem.Allocator,
    h_inv: []f64,
    coords: []f64,
    current_grad_flat: []const f64,
    prev_grad: *?[]f64,
    prev_step: *?[]f64,
    n3: usize,
    max_step_size: f64,
) !void {
    // BFGS update of inverse Hessian (requires both previous gradient and step)
    if (prev_grad.*) |pg| {
        if (prev_step.*) |ps| {
            // y = g_{k} - g_{k-1}
            const y_vec = try alloc.alloc(f64, n3);
            defer alloc.free(y_vec);

            for (0..n3) |i| {
                y_vec[i] = current_grad_flat[i] - pg[i];
            }
            // s = step taken in previous iteration (already saved)
            try bfgsUpdate(alloc, h_inv, ps, y_vec, n3);
        }
    }

    // Compute search direction: p = -H_inv * grad
    const search_dir = try matVecMul(alloc, h_inv, current_grad_flat, n3);
    defer alloc.free(search_dir);

    for (search_dir) |*d| {
        d.* = -d.*;
    }

    // Apply trust region scaling
    trustRegionScale(search_dir, max_step_size);

    // Save gradient for BFGS update in next iteration
    if (prev_grad.* == null) {
        prev_grad.* = try alloc.alloc(f64, n3);
    }
    @memcpy(prev_grad.*.?, current_grad_flat);

    // Save step for BFGS update in next iteration
    if (prev_step.* == null) {
        prev_step.* = try alloc.alloc(f64, n3);
    }
    @memcpy(prev_step.*.?, search_dir);

    // Take step: x_{k+1} = x_k + p_k
    for (0..n3) |i| {
        coords[i] += search_dir[i];
    }
}

/// Package the final optimization result from loop state.
fn packageOptResult(
    alloc: std.mem.Allocator,
    nuc_positions: []const Vec3,
    current_grad_flat: []const f64,
    current_energy: f64,
    steps: usize,
    converged: bool,
    energy_history: *std.ArrayList(f64),
) !OptResult {
    const n_atoms = nuc_positions.len;
    const final_positions = try alloc.alloc(Vec3, n_atoms);
    for (0..n_atoms) |i| {
        final_positions[i] = nuc_positions[i];
    }

    const final_gradient = try alloc.alloc(Vec3, n_atoms);
    for (0..n_atoms) |i| {
        final_gradient[i] = .{
            .x = current_grad_flat[i * 3 + 0],
            .y = current_grad_flat[i * 3 + 1],
            .z = current_grad_flat[i * 3 + 2],
        };
    }

    const energy_hist = try energy_history.toOwnedSlice(alloc);

    return .{
        .positions = final_positions,
        .energy = current_energy,
        .final_gradient = final_gradient,
        .steps = steps,
        .converged = converged,
        .energy_history = energy_hist,
    };
}

/// Run BFGS geometry optimization.
///
/// Takes the initial geometry (shells, nuclear positions/charges),
/// the number of electrons, and optimization parameters.
/// Returns the optimized geometry and energy.
///
/// The `shells` array is modified in-place during optimization
/// (centers are updated). The caller should provide a mutable copy
/// if they need to preserve the original.
fn runRhfScfAndGradient(
    alloc: std.mem.Allocator,
    shells: []ContractedShell,
    nuc_positions: []Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    n_occ: usize,
    scf_params: *ScfParams,
    prev_density: *?[]f64,
    current_grad_flat: []f64,
    current_energy_out: *f64,
) !void {
    // Seed SCF from previous density if available
    if (prev_density.*) |pd| {
        scf_params.initial_density = pd;
    }

    var scf_result = try gto_scf.runGeneralRhfScf(
        alloc,
        shells,
        nuc_positions,
        nuc_charges,
        n_electrons,
        scf_params.*,
    );
    defer scf_result.deinit(alloc);

    current_energy_out.* = scf_result.total_energy;

    // Save density matrix for next SCF
    if (prev_density.*) |pd| alloc.free(pd);
    const n_basis = scf_result.density_matrix.len;
    prev_density.* = try alloc.alloc(f64, n_basis);
    @memcpy(prev_density.*.?, scf_result.density_matrix);

    // Compute gradient
    var grad_result = try gradient_mod.computeRhfGradient(
        alloc,
        shells,
        nuc_positions,
        nuc_charges,
        scf_result.density_matrix,
        scf_result.orbital_energies,
        scf_result.mo_coefficients,
        n_occ,
    );
    defer grad_result.deinit(alloc);

    flattenGradientInto(grad_result.gradients, current_grad_flat);
}

/// Shared state owned by the BFGS loop. Callers must call `deinit`.
const BfgsLoopState = struct {
    shell_to_atom: []usize,
    coords: []f64,
    h_inv: []f64,
    energy_history: std.ArrayList(f64),
    prev_grad: ?[]f64,
    prev_step: ?[]f64,
    current_grad_flat: []f64,

    fn init(
        alloc: std.mem.Allocator,
        shells: []ContractedShell,
        nuc_positions: []const Vec3,
        n3: usize,
    ) !BfgsLoopState {
        const shell_to_atom = try buildShellToAtomMap(alloc, shells, nuc_positions);
        errdefer alloc.free(shell_to_atom);

        const coords = try flattenVec3(alloc, nuc_positions);
        errdefer alloc.free(coords);

        const h_inv = try initInverseHessian(alloc, n3);
        errdefer alloc.free(h_inv);

        const current_grad_flat = try alloc.alloc(f64, n3);
        errdefer alloc.free(current_grad_flat);

        return .{
            .shell_to_atom = shell_to_atom,
            .coords = coords,
            .h_inv = h_inv,
            .energy_history = .empty,
            .prev_grad = null,
            .prev_step = null,
            .current_grad_flat = current_grad_flat,
        };
    }

    fn deinit(self: *BfgsLoopState, alloc: std.mem.Allocator) void {
        alloc.free(self.shell_to_atom);
        alloc.free(self.coords);
        alloc.free(self.h_inv);
        self.energy_history.deinit(alloc);
        if (self.prev_grad) |pg| alloc.free(pg);
        if (self.prev_step) |ps| alloc.free(ps);
        alloc.free(self.current_grad_flat);
    }
};

const RhfOptContext = struct {
    alloc: std.mem.Allocator,
    shells: []ContractedShell,
    nuc_positions: []Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    n_occ: usize,
    shell_to_atom: []const usize,
    coords: []f64,
    h_inv: []f64,
    current_grad_flat: []f64,
    n3: usize,
    params: OptParams,
};

fn runRhfOptStep(
    ctx: RhfOptContext,
    scf_params: *ScfParams,
    prev_density: *?[]f64,
    prev_grad: *?[]f64,
    prev_step: *?[]f64,
    current_energy: *f64,
    energy_history: *std.ArrayList(f64),
    steps: usize,
) !bool {
    applyCoordsToPositions(ctx.coords, ctx.nuc_positions, ctx.shells, ctx.shell_to_atom);

    try runRhfScfAndGradient(
        ctx.alloc,
        ctx.shells,
        ctx.nuc_positions,
        ctx.nuc_charges,
        ctx.n_electrons,
        ctx.n_occ,
        scf_params,
        prev_density,
        ctx.current_grad_flat,
        current_energy,
    );
    try energy_history.append(ctx.alloc, current_energy.*);

    const grad_rms = vecRms(ctx.current_grad_flat);
    const grad_max = vecMaxAbs(ctx.current_grad_flat);

    logging.progress(
        ctx.params.print_progress,
        "  Opt step {d}: E = {d:.12} Ha, grad_rms = {e:10.3}, grad_max = {e:10.3}\n",
        .{ steps, current_energy.*, grad_rms, grad_max },
    );

    if (grad_rms < ctx.params.grad_rms_threshold and grad_max < ctx.params.grad_max_threshold) {
        return true;
    }

    try applyBfgsStep(
        ctx.alloc,
        ctx.h_inv,
        ctx.coords,
        ctx.current_grad_flat,
        prev_grad,
        prev_step,
        ctx.n3,
        ctx.params.max_step_size,
    );
    return false;
}

pub fn optimizeGeometry(
    alloc: std.mem.Allocator,
    shells: []ContractedShell,
    nuc_positions: []Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    params: OptParams,
) !OptResult {
    const n_atoms = nuc_positions.len;
    const n3 = n_atoms * 3;
    const n_occ = n_electrons / 2;

    var state = try BfgsLoopState.init(alloc, shells, nuc_positions, n3);
    defer state.deinit(alloc);

    // SCF params — enable density matrix reuse after first iteration
    var scf_params = params.scf_params;
    var prev_density: ?[]f64 = null;
    defer if (prev_density) |pd| alloc.free(pd);

    const ctx = RhfOptContext{
        .alloc = alloc,
        .shells = shells,
        .nuc_positions = nuc_positions,
        .nuc_charges = nuc_charges,
        .n_electrons = n_electrons,
        .n_occ = n_occ,
        .shell_to_atom = state.shell_to_atom,
        .coords = state.coords,
        .h_inv = state.h_inv,
        .current_grad_flat = state.current_grad_flat,
        .n3 = n3,
        .params = params,
    };

    var converged = false;
    var steps: usize = 0;
    var current_energy: f64 = 0.0;
    for (0..params.max_steps) |step| {
        steps = step + 1;
        if (try runRhfOptStep(
            ctx,
            &scf_params,
            &prev_density,
            &state.prev_grad,
            &state.prev_step,
            &current_energy,
            &state.energy_history,
            steps,
        )) {
            converged = true;
            break;
        }
    }

    return packageOptResult(
        alloc,
        nuc_positions,
        state.current_grad_flat,
        current_energy,
        steps,
        converged,
        &state.energy_history,
    );
}

/// Parameters controlling KS-DFT geometry optimization.
pub const KsOptParams = struct {
    /// Maximum number of geometry optimization steps.
    max_steps: usize = 100,
    /// Convergence threshold for RMS gradient (Ha/Bohr).
    grad_rms_threshold: f64 = 3e-4,
    /// Convergence threshold for max absolute gradient component (Ha/Bohr).
    grad_max_threshold: f64 = 4.5e-4,
    /// Convergence threshold for RMS step size (Bohr).
    step_rms_threshold: f64 = 1.2e-3,
    /// Convergence threshold for max step component (Bohr).
    step_max_threshold: f64 = 1.8e-3,
    /// Maximum step size for trust region (Bohr).
    max_step_size: f64 = 0.3,
    /// KS-DFT SCF parameters to use at each geometry.
    ks_params: KsParams = .{},
    /// Atomic numbers for Becke grid construction.
    atomic_numbers: []const u8 = &.{},
    /// Print progress information.
    print_progress: bool = true,
};

/// Run BFGS geometry optimization with KS-DFT (LDA or B3LYP).
///
/// Takes the initial geometry (shells, nuclear positions/charges),
/// the number of electrons, and optimization parameters.
/// Returns the optimized geometry and energy.
///
/// The `shells` array is modified in-place during optimization
/// (centers are updated). The caller should provide a mutable copy
/// if they need to preserve the original.
fn runKsDftAndGradient(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []ContractedShell,
    nuc_positions: []Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    n_occ: usize,
    params: KsOptParams,
    grid_config: becke.GridConfig,
    current_grad_flat: []f64,
    current_energy_out: *f64,
) !void {
    const n_atoms = nuc_positions.len;

    // Run KS-DFT SCF
    var ks_result = try kohn_sham.runKohnShamScf(
        alloc,
        io,
        shells,
        nuc_positions,
        nuc_charges,
        n_electrons,
        params.ks_params,
    );
    defer ks_result.deinit(alloc);

    current_energy_out.* = ks_result.total_energy;

    // Build molecular grid for gradient calculation
    // (must rebuild each step because atomic positions change)
    const becke_atoms = try alloc.alloc(becke.Atom, n_atoms);
    defer alloc.free(becke_atoms);

    for (0..n_atoms) |i| {
        becke_atoms[i] = .{
            .x = nuc_positions[i].x,
            .y = nuc_positions[i].y,
            .z = nuc_positions[i].z,
            .z_number = if (params.atomic_numbers.len > 0)
                @intCast(params.atomic_numbers[i])
            else
                @intFromFloat(nuc_charges[i]),
        };
    }
    const grid_points = try becke.buildMolecularGrid(alloc, becke_atoms, grid_config);
    defer alloc.free(grid_points);

    // Compute KS-DFT gradient
    var grad_result = try gradient_mod.computeKsDftGradient(
        alloc,
        shells,
        nuc_positions,
        nuc_charges,
        ks_result.density_matrix_result,
        ks_result.orbital_energies,
        ks_result.mo_coefficients,
        n_occ,
        grid_points,
        params.ks_params.xc_functional,
    );
    defer grad_result.deinit(alloc);

    flattenGradientInto(grad_result.gradients, current_grad_flat);
}

const KsOptContext = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []ContractedShell,
    nuc_positions: []Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    n_occ: usize,
    shell_to_atom: []const usize,
    coords: []f64,
    h_inv: []f64,
    current_grad_flat: []f64,
    n3: usize,
    params: KsOptParams,
    grid_config: becke.GridConfig,
};

fn runKsOptStep(
    ctx: KsOptContext,
    prev_grad: *?[]f64,
    prev_step: *?[]f64,
    current_energy: *f64,
    energy_history: *std.ArrayList(f64),
    steps: usize,
) !bool {
    applyCoordsToPositions(ctx.coords, ctx.nuc_positions, ctx.shells, ctx.shell_to_atom);

    try runKsDftAndGradient(
        ctx.alloc,
        ctx.io,
        ctx.shells,
        ctx.nuc_positions,
        ctx.nuc_charges,
        ctx.n_electrons,
        ctx.n_occ,
        ctx.params,
        ctx.grid_config,
        ctx.current_grad_flat,
        current_energy,
    );
    try energy_history.append(ctx.alloc, current_energy.*);

    const grad_rms = vecRms(ctx.current_grad_flat);
    const grad_max = vecMaxAbs(ctx.current_grad_flat);

    logging.progress(
        ctx.params.print_progress,
        "  KS-DFT opt step {d}: E = {d:.12} Ha, grad_rms = {e:10.3}, grad_max = {e:10.3}\n",
        .{ steps, current_energy.*, grad_rms, grad_max },
    );

    if (grad_rms < ctx.params.grad_rms_threshold and grad_max < ctx.params.grad_max_threshold) {
        return true;
    }

    try applyBfgsStep(
        ctx.alloc,
        ctx.h_inv,
        ctx.coords,
        ctx.current_grad_flat,
        prev_grad,
        prev_step,
        ctx.n3,
        ctx.params.max_step_size,
    );
    return false;
}

pub fn optimizeKsDftGeometry(
    alloc: std.mem.Allocator,
    io: std.Io,
    shells: []ContractedShell,
    nuc_positions: []Vec3,
    nuc_charges: []const f64,
    n_electrons: usize,
    params: KsOptParams,
) !OptResult {
    const n_atoms = nuc_positions.len;
    const n3 = n_atoms * 3;
    const n_occ = n_electrons / 2;

    var state = try BfgsLoopState.init(alloc, shells, nuc_positions, n3);
    defer state.deinit(alloc);

    const grid_config = becke.GridConfig{
        .n_radial = params.ks_params.n_radial,
        .n_angular = params.ks_params.n_angular,
        .prune = params.ks_params.prune,
    };

    const ctx = KsOptContext{
        .alloc = alloc,
        .io = io,
        .shells = shells,
        .nuc_positions = nuc_positions,
        .nuc_charges = nuc_charges,
        .n_electrons = n_electrons,
        .n_occ = n_occ,
        .shell_to_atom = state.shell_to_atom,
        .coords = state.coords,
        .h_inv = state.h_inv,
        .current_grad_flat = state.current_grad_flat,
        .n3 = n3,
        .params = params,
        .grid_config = grid_config,
    };

    var converged = false;
    var steps: usize = 0;
    var current_energy: f64 = 0.0;
    for (0..params.max_steps) |step| {
        steps = step + 1;
        if (try runKsOptStep(
            ctx,
            &state.prev_grad,
            &state.prev_step,
            &current_energy,
            &state.energy_history,
            steps,
        )) {
            converged = true;
            break;
        }
    }

    return packageOptResult(
        alloc,
        nuc_positions,
        state.current_grad_flat,
        current_energy,
        steps,
        converged,
        &state.energy_history,
    );
}

// Slow geometry optimization regression coverage lives in `regression_tests.zig`.
