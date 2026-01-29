const std = @import("std");
const math = @import("../math/math.zig");
const config = @import("../config/config.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");

/// Unified optimizer interface for structure relaxation.
/// New algorithms can be added to this union.
pub const Optimizer = union(enum) {
    bfgs: BFGS,
    steepest_descent: SteepestDescent,
    // Future: cg, lbfgs, fire, etc.

    /// Initialize optimizer from config.
    pub fn init(
        alloc: std.mem.Allocator,
        algorithm: config.RelaxAlgorithm,
        n_atoms: usize,
        atoms: []const hamiltonian.AtomData,
        cell: math.Mat3,
    ) !Optimizer {
        return switch (algorithm) {
            .bfgs => .{ .bfgs = try BFGS.init(alloc, n_atoms, atoms, cell) },
            .steepest_descent => .{ .steepest_descent = SteepestDescent.init() },
            .cg => return error.NotImplemented, // TODO
        };
    }

    /// Free optimizer resources.
    pub fn deinit(self: *Optimizer, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .bfgs => |*b| b.deinit(alloc),
            .steepest_descent => {},
        }
    }

    /// Compute step from forces.
    /// Returns displacement for each atom.
    pub fn step(
        self: *Optimizer,
        alloc: std.mem.Allocator,
        forces: []const math.Vec3,
        max_step: f64,
    ) ![]math.Vec3 {
        return switch (self.*) {
            .bfgs => |*b| try b.step(alloc, forces, max_step),
            .steepest_descent => |*sd| try sd.step(alloc, forces, max_step),
        };
    }

    /// Update optimizer state after a step.
    /// Call this after positions have been updated.
    pub fn update(
        self: *Optimizer,
        prev_positions: []const math.Vec3,
        new_positions: []const math.Vec3,
        prev_forces: []const math.Vec3,
        new_forces: []const math.Vec3,
    ) void {
        switch (self.*) {
            .bfgs => |*b| b.update(prev_positions, new_positions, prev_forces, new_forces),
            .steepest_descent => {},
        }
    }

    /// Reset optimizer state (e.g., after a line search failure).
    pub fn reset(self: *Optimizer) void {
        switch (self.*) {
            .bfgs => |*b| b.reset(),
            .steepest_descent => {},
        }
    }
};

/// BFGS quasi-Newton optimizer.
/// Uses an approximate inverse Hessian to compute steps.
pub const BFGS = struct {
    n_atoms: usize,
    n_dof: usize, // 3 * n_atoms
    h_inv: []f64, // Approximate inverse Hessian (n_dof x n_dof)
    initialized: bool,

    /// Initialize BFGS with model inverse Hessian based on nearest-neighbor distance.
    /// Uses a diagonal H_inv = (1/k) * I, where k is estimated from the nearest
    /// interatomic distance using Badger's rule: shorter bonds → stiffer springs.
    pub fn init(
        alloc: std.mem.Allocator,
        n_atoms: usize,
        atoms: []const hamiltonian.AtomData,
        cell: math.Mat3,
    ) !BFGS {
        const n_dof = 3 * n_atoms;
        const h_inv = try alloc.alloc(f64, n_dof * n_dof);

        // Estimate force constant from nearest-neighbor distance
        const h_inv_diag = estimateInvHessianDiag(n_atoms, atoms, cell);

        // Initialize as scaled identity matrix
        for (h_inv, 0..) |*h, i| {
            const row = i / n_dof;
            const col = i % n_dof;
            h.* = if (row == col) h_inv_diag else 0.0;
        }

        return BFGS{
            .n_atoms = n_atoms,
            .n_dof = n_dof,
            .h_inv = h_inv,
            .initialized = false,
        };
    }

    /// Estimate diagonal value for inverse Hessian from structure.
    /// Currently returns 1.0 (identity scaling) which is simple and robust.
    fn estimateInvHessianDiag(
        n_atoms: usize,
        atoms: []const hamiltonian.AtomData,
        cell: math.Mat3,
    ) f64 {
        _ = n_atoms;
        _ = atoms;
        _ = cell;
        return 1.0;
    }

    /// Compute distance between two positions using minimum image convention.
    fn minimumImageDistance(pos_a: math.Vec3, pos_b: math.Vec3, cell: math.Mat3) f64 {
        // Difference in Cartesian coordinates
        var dx = pos_b.x - pos_a.x;
        var dy = pos_b.y - pos_a.y;
        var dz = pos_b.z - pos_a.z;

        // Convert to fractional coordinates using Cramer's rule (3x3 inverse)
        // Cell rows are lattice vectors: r = f0*a0 + f1*a1 + f2*a2
        // where a_i = cell.row(i), so r = cell^T * f
        const a = cell.m;
        const det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
            a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
            a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if (@abs(det) < 1e-30) return @sqrt(dx * dx + dy * dy + dz * dz);
        const inv_det = 1.0 / det;

        // Compute inverse of cell^T (= (cell^{-1})^T), applied to (dx,dy,dz)
        const fx = inv_det * ((a[1][1] * a[2][2] - a[1][2] * a[2][1]) * dx +
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * dy +
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * dz);
        const fy = inv_det * ((a[1][2] * a[2][0] - a[1][0] * a[2][2]) * dx +
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * dy +
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * dz);
        const fz = inv_det * ((a[1][0] * a[2][1] - a[1][1] * a[2][0]) * dx +
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * dy +
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * dz);

        // Apply minimum image: wrap to [-0.5, 0.5]
        const fx_w = fx - @round(fx);
        const fy_w = fy - @round(fy);
        const fz_w = fz - @round(fz);

        // Convert back to Cartesian
        dx = a[0][0] * fx_w + a[1][0] * fy_w + a[2][0] * fz_w;
        dy = a[0][1] * fx_w + a[1][1] * fy_w + a[2][1] * fz_w;
        dz = a[0][2] * fx_w + a[1][2] * fy_w + a[2][2] * fz_w;

        return @sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// Free inverse Hessian matrix.
    pub fn deinit(self: *BFGS, alloc: std.mem.Allocator) void {
        if (self.h_inv.len > 0) {
            alloc.free(self.h_inv);
        }
    }

    /// Compute step from forces using inverse Hessian.
    /// step = H^{-1} * (-gradient) = H^{-1} * forces
    pub fn step(self: *BFGS, alloc: std.mem.Allocator, forces: []const math.Vec3, max_step: f64) ![]math.Vec3 {
        var displacements = try alloc.alloc(math.Vec3, self.n_atoms);

        // Convert forces to flat array
        var f_flat: [256 * 3]f64 = undefined; // Max 256 atoms
        if (self.n_atoms > 256) return error.TooManyAtoms;
        for (forces, 0..) |f, i| {
            f_flat[i * 3 + 0] = f.x;
            f_flat[i * 3 + 1] = f.y;
            f_flat[i * 3 + 2] = f.z;
        }

        // Compute displacement = H^{-1} * forces
        var d_flat: [256 * 3]f64 = undefined;
        for (0..self.n_dof) |i| {
            var sum: f64 = 0.0;
            for (0..self.n_dof) |j| {
                sum += self.h_inv[i * self.n_dof + j] * f_flat[j];
            }
            d_flat[i] = sum;
        }

        // Convert back to Vec3 and apply step limit
        var max_disp: f64 = 0.0;
        for (0..self.n_atoms) |i| {
            const dx = d_flat[i * 3 + 0];
            const dy = d_flat[i * 3 + 1];
            const dz = d_flat[i * 3 + 2];
            const mag = std.math.sqrt(dx * dx + dy * dy + dz * dz);
            if (mag > max_disp) max_disp = mag;
        }

        const scale = if (max_disp > max_step) max_step / max_disp else 1.0;
        for (0..self.n_atoms) |i| {
            displacements[i] = math.Vec3{
                .x = d_flat[i * 3 + 0] * scale,
                .y = d_flat[i * 3 + 1] * scale,
                .z = d_flat[i * 3 + 2] * scale,
            };
        }

        return displacements;
    }

    /// Update inverse Hessian using BFGS formula.
    /// H^{-1}_{k+1} = (I - ρ s y^T) H^{-1}_k (I - ρ y s^T) + ρ s s^T
    /// where s = x_{k+1} - x_k, y = g_{k+1} - g_k = -(f_{k+1} - f_k), ρ = 1/(y^T s)
    pub fn update(
        self: *BFGS,
        prev_positions: []const math.Vec3,
        new_positions: []const math.Vec3,
        prev_forces: []const math.Vec3,
        new_forces: []const math.Vec3,
    ) void {
        if (self.n_atoms > 256) return; // Safety check

        // Compute s = x_{k+1} - x_k (position change)
        var s: [256 * 3]f64 = undefined;
        for (0..self.n_atoms) |i| {
            s[i * 3 + 0] = new_positions[i].x - prev_positions[i].x;
            s[i * 3 + 1] = new_positions[i].y - prev_positions[i].y;
            s[i * 3 + 2] = new_positions[i].z - prev_positions[i].z;
        }

        // Compute y = g_{k+1} - g_k = -(f_{k+1} - f_k) (gradient change)
        // In our convention, force = -gradient
        var y: [256 * 3]f64 = undefined;
        for (0..self.n_atoms) |i| {
            y[i * 3 + 0] = -(new_forces[i].x - prev_forces[i].x);
            y[i * 3 + 1] = -(new_forces[i].y - prev_forces[i].y);
            y[i * 3 + 2] = -(new_forces[i].z - prev_forces[i].z);
        }

        // Compute ρ = 1/(y^T s)
        var ys: f64 = 0.0;
        for (0..self.n_dof) |i| {
            ys += y[i] * s[i];
        }

        // Skip update if curvature condition not satisfied
        if (ys <= 1e-10) {
            return;
        }

        const rho = 1.0 / ys;
        self.initialized = true;

        // BFGS update: H^{-1}_{k+1} = (I - ρ s y^T) H^{-1}_k (I - ρ y s^T) + ρ s s^T

        // Compute H^{-1} * y
        var h_y: [256 * 3]f64 = undefined;
        for (0..self.n_dof) |i| {
            var sum: f64 = 0.0;
            for (0..self.n_dof) |j| {
                sum += self.h_inv[i * self.n_dof + j] * y[j];
            }
            h_y[i] = sum;
        }

        // Compute y^T * H^{-1} * y
        var y_h_y: f64 = 0.0;
        for (0..self.n_dof) |i| {
            y_h_y += y[i] * h_y[i];
        }

        // Update H^{-1} using Sherman-Morrison formula variant
        const factor1 = (ys + y_h_y) * rho * rho;
        const factor2 = rho;

        for (0..self.n_dof) |i| {
            for (0..self.n_dof) |j| {
                self.h_inv[i * self.n_dof + j] +=
                    factor1 * s[i] * s[j] -
                    factor2 * (h_y[i] * s[j] + s[i] * h_y[j]);
            }
        }
    }

    /// Reset to identity inverse Hessian.
    pub fn reset(self: *BFGS) void {
        for (self.h_inv, 0..) |*h, i| {
            const row = i / self.n_dof;
            const col = i % self.n_dof;
            h.* = if (row == col) 1.0 else 0.0;
        }
        self.initialized = false;
    }
};

/// Simple steepest descent optimizer for debugging.
pub const SteepestDescent = struct {
    step_size: f64,

    pub fn init() SteepestDescent {
        return .{ .step_size = 0.1 };
    }

    /// Step in direction of forces with limited step size.
    pub fn step(self: *SteepestDescent, alloc: std.mem.Allocator, forces: []const math.Vec3, max_step: f64) ![]math.Vec3 {
        _ = self;
        var displacements = try alloc.alloc(math.Vec3, forces.len);

        // Find max force magnitude
        var max_force: f64 = 0.0;
        for (forces) |f| {
            const mag = math.Vec3.norm(f);
            if (mag > max_force) max_force = mag;
        }

        // Scale forces to respect max_step
        const scale = if (max_force > 0.0) @min(max_step / max_force, 1.0) else 0.0;

        for (forces, 0..) |f, i| {
            displacements[i] = math.Vec3.scale(f, scale);
        }

        return displacements;
    }
};

test "bfgs basic" {
    const testing = std.testing;
    const alloc = testing.allocator;

    // Create dummy atoms far apart so H_inv_diag ≈ 1/k with k from d_min
    const atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 0, .y = 0, .z = 0 }, .species_index = 0 },
        .{ .position = .{ .x = 5, .y = 0, .z = 0 }, .species_index = 0 },
    };
    // Large cubic cell (identity * 20 Bohr)
    const cell = math.Mat3{ .m = .{
        .{ 20, 0, 0 },
        .{ 0, 20, 0 },
        .{ 0, 0, 20 },
    } };

    var bfgs = try BFGS.init(alloc, 2, &atoms, cell);
    defer bfgs.deinit(alloc);

    const forces = [_]math.Vec3{
        math.Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 },
    };

    const disp = try bfgs.step(alloc, &forces, 10.0);
    defer alloc.free(disp);

    // Identity Hessian gives step = force
    try testing.expectApproxEqAbs(disp[0].x, 1.0, 1e-10);
    try testing.expectApproxEqAbs(disp[0].y, 0.0, 1e-10);
    try testing.expectApproxEqAbs(disp[1].x, 0.0, 1e-10);
    try testing.expectApproxEqAbs(disp[1].y, 1.0, 1e-10);
}

test "steepest descent basic" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var sd = SteepestDescent.init();

    const forces = [_]math.Vec3{
        math.Vec3{ .x = 2.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 },
    };

    const disp = try sd.step(alloc, &forces, 1.0);
    defer alloc.free(disp);

    // Max force is 2.0, max_step is 1.0, so scale = 0.5
    try testing.expectApproxEqAbs(disp[0].x, 1.0, 1e-10);
    try testing.expectApproxEqAbs(disp[1].y, 0.5, 1e-10);
}

test "optimizer interface" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 0, .y = 0, .z = 0 }, .species_index = 0 },
        .{ .position = .{ .x = 5, .y = 0, .z = 0 }, .species_index = 0 },
    };
    const cell = math.Mat3{ .m = .{
        .{ 20, 0, 0 },
        .{ 0, 20, 0 },
        .{ 0, 0, 20 },
    } };

    var opt = try Optimizer.init(alloc, .bfgs, 2, &atoms, cell);
    defer opt.deinit(alloc);

    const forces = [_]math.Vec3{
        math.Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 },
    };

    const disp = try opt.step(alloc, &forces, 10.0);
    defer alloc.free(disp);

    // Identity Hessian: step = force
    try testing.expectApproxEqAbs(disp[0].x, 1.0, 1e-10);
}
