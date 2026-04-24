//! Kleinman-Bylander projector construction.
//!
//! Given screened pseudopotentials V_ps,l(r) for each l-channel and
//! a chosen local potential V_loc(r), construct KB projectors:
//!
//!   |β_l⟩ = (V_ps,l - V_loc) |φ̃_l⟩
//!   D_l = ⟨φ̃_l| V_ps,l - V_loc |φ̃_l⟩
//!
//! The nonlocal pseudopotential is then:
//!   V_NL = Σ_l |β_l⟩ D_l^{-1} ⟨β_l|
//!
//! The local potential V_loc is typically chosen as V_ps,l_max (highest l channel)
//! or a weighted average.

const std = @import("std");
const RadialGrid = @import("radial_grid.zig").RadialGrid;
const tm_generator = @import("tm_generator.zig");

pub const KbProjector = struct {
    /// KB projector β_l(r) (unnormalized: (V_l - V_loc) * u_ps_l)
    beta: []f64,
    /// KB energy D_l = ⟨u_ps| V_l - V_loc |u_ps⟩ (Ry)
    d_ion: f64,
    /// Angular momentum
    l: u32,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *KbProjector) void {
        self.allocator.free(self.beta);
    }
};

pub const LocalPotential = struct {
    /// Ionic local potential: V_loc(r) = V_ps,l_loc(r) - V_H(r) - V_xc(r)
    /// For pseudopotential files, we need the unscreened ionic potential.
    v_local: []f64,
    /// The l-channel used as local
    l_local: u32,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *LocalPotential) void {
        self.allocator.free(self.v_local);
    }
};

/// Construct KB projector for a single l-channel.
///
/// v_ps_l: screened pseudopotential for this l (Ry)
/// v_local: local potential (Ry)
/// u_ps: pseudo wavefunction u_ps(r) = r*R(r)
pub fn build_projector(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    v_ps_l: []const f64,
    v_local: []const f64,
    u_ps: []const f64,
    l: u32,
) !KbProjector {
    const n = grid.n;

    // β_l(r) = (V_ps,l(r) - V_loc(r)) * u_ps(r)
    const beta = try allocator.alloc(f64, n);
    for (0..n) |i| {
        beta[i] = (v_ps_l[i] - v_local[i]) * u_ps[i];
    }

    // D_l = ∫ u_ps(r) * (V_ps,l(r) - V_loc(r)) * u_ps(r) dr
    //     = ∫ u_ps(r) * β_l(r) dr
    var d_ion: f64 = 0;
    for (1..n) |i| {
        const f_prev = u_ps[i - 1] * beta[i - 1] * grid.rab[i - 1];
        const f_curr = u_ps[i] * beta[i] * grid.rab[i];
        d_ion += 0.5 * (f_prev + f_curr);
    }

    return .{
        .beta = beta,
        .d_ion = d_ion,
        .l = l,
        .allocator = allocator,
    };
}

/// Unscreen the pseudopotential: remove Hartree and XC contributions
/// to get the bare ionic pseudopotential.
///
/// V_ion(r) = V_screened(r) - V_H[ρ_atom](r) - V_xc[ρ_atom](r)
///
/// This is needed for the UPF file, which stores the ionic (unscreened) potential.
pub fn unscreen(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    v_screened: []const f64,
    v_hartree: []const f64,
    v_xc: []const f64,
) ![]f64 {
    const n = grid.n;
    const v_ion = try allocator.alloc(f64, n);
    for (0..n) |i| {
        v_ion[i] = v_screened[i] - v_hartree[i] - v_xc[i];
    }
    return v_ion;
}

// ============================================================================
// Tests
// ============================================================================

const schrodinger = @import("schrodinger.zig");
const RadialGridMod = @import("radial_grid.zig");

test "KB projector: D_l nonzero for non-local channel" {
    // For H 1s: generate TM pseudo, use V_ps,s as local,
    // then there should be no nonlocal (D = 0) since it IS the local channel.
    const allocator = std.testing.allocator;
    var grid = try RadialGridMod.RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    var sol = try schrodinger.solve(allocator, &grid, v, .{ .n = 1, .l = 0 }, -0.5);
    defer sol.deinit();

    var pw = try tm_generator.generate(allocator, &grid, sol.u, v, sol.energy, 0, 1.5);
    defer pw.deinit();

    // When V_local = V_ps,l, the KB projector should give D = 0
    var kb = try build_projector(allocator, &grid, pw.v_ps, pw.v_ps, pw.u, 0);
    defer kb.deinit();

    try std.testing.expectApproxEqAbs(0.0, kb.d_ion, 1e-10);

    // beta should be zero everywhere
    for (0..grid.n) |i| {
        try std.testing.expectApproxEqAbs(0.0, kb.beta[i], 1e-10);
    }
}

test "KB projector: nonzero for different V_local" {
    // Generate s and p channels, use p as local, check s projector is nonzero
    const allocator = std.testing.allocator;
    var grid = try RadialGridMod.RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const v = try allocator.alloc(f64, grid.n);
    defer allocator.free(v);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        v[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
    }

    // s-channel
    var sol_s = try schrodinger.solve(allocator, &grid, v, .{ .n = 1, .l = 0 }, -0.5);
    defer sol_s.deinit();

    var pw_s = try tm_generator.generate(allocator, &grid, sol_s.u, v, sol_s.energy, 0, 1.5);
    defer pw_s.deinit();

    // p-channel (use 2p for hydrogen)
    var sol_p = try schrodinger.solve(allocator, &grid, v, .{ .n = 2, .l = 1 }, -0.2);
    defer sol_p.deinit();

    var pw_p = try tm_generator.generate(allocator, &grid, sol_p.u, v, sol_p.energy, 1, 1.5);
    defer pw_p.deinit();

    // Use p-channel as local, build s projector
    var kb_s = try build_projector(allocator, &grid, pw_s.v_ps, pw_p.v_ps, pw_s.u, 0);
    defer kb_s.deinit();

    // D_s should be nonzero (s and p potentials differ inside rc)
    try std.testing.expect(@abs(kb_s.d_ion) > 0.01);
}

test "unscreen: V_ion = V_screened - V_H - V_xc" {
    const allocator = std.testing.allocator;
    const n: usize = 100;
    var grid = try RadialGridMod.RadialGrid.init(allocator, n, 1e-6, 10.0);
    defer grid.deinit();

    const v_scr = try allocator.alloc(f64, n);
    defer allocator.free(v_scr);

    const v_h = try allocator.alloc(f64, n);
    defer allocator.free(v_h);

    const v_xc = try allocator.alloc(f64, n);
    defer allocator.free(v_xc);

    for (0..n) |i| {
        v_scr[i] = -1.0;
        v_h[i] = 0.3;
        v_xc[i] = -0.1;
    }

    const v_ion = try unscreen(allocator, &grid, v_scr, v_h, v_xc);
    defer allocator.free(v_ion);

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(-1.2, v_ion[i], 1e-15);
    }
}
