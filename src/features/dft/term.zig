//! Physical terms of the DFT Hamiltonian.
//!
//! Each Term contributes:
//!   (1) an energy value (possibly depending on the current density)
//!   (2) a per-k-point operator that sums into H|ψ⟩
//!
//! This is the dft-zig analogue of DFTK.jl's Term abstraction. The long-term
//! goal is to replace the monolithic buildHamiltonian / buildPotentialGrid
//! flow with an iteration over physics terms, each self-contained.
//!
//! This module (Step 2a) defines the types. Wiring to the SCF loop happens
//! in subsequent commits.

const std = @import("std");
const config_mod = @import("../config/config.zig");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const ewald_mod = @import("../ewald/ewald.zig");
const fft_grid = @import("../scf/fft_grid.zig");
const grid_mod = @import("../scf/pw_grid.zig");
const gvec_iter = @import("../scf/gvec_iter.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const xc_fields = @import("../scf/xc_fields.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const xc_mod = @import("../xc/xc.zig");

pub const Grid = grid_mod.Grid;

/// Kinetic energy: T|ψ⟩ = |G+k|²/2 |ψ⟩ (Rydberg).
/// Density-independent; operator-only (does not contribute to V_eff(r)).
pub const TermKinetic = struct {};

/// Local component of the ionic pseudopotential: V_loc(r) = Σ_a v^a_loc(r-R_a).
/// Depends on atomic positions (via Model), not on ρ. Contributes to V_eff(G).
///
/// mode and explicit_alpha parameterize the long-range splitting; resolved
/// against the cell at evaluation time (the effective alpha depends on geometry).
pub const TermAtomicLocal = struct {
    mode: local_potential.LocalPotentialMode = .short_range,
    explicit_alpha: f64 = 0.0,
};

/// Non-local pseudopotential: Σ_a Σ_ij |β^a_i⟩ D_ij ⟨β^a_j|.
/// Density-independent. Operator-only.
pub const TermAtomicNonlocal = struct {};

/// Hartree: V_H(G) = 8π/G² ρ(G) (Rydberg).
/// Uses a real-space Coulomb cutoff for isolated boundary conditions.
pub const TermHartree = struct {
    /// true for isolated boundary conditions (Coulomb cutoff applied at eval time).
    isolated: bool = false,
};

/// Exchange-correlation: V_xc[ρ](r), E_xc[ρ].
pub const TermXc = struct {
    functional: xc_mod.Functional,
};

/// Ion-ion Ewald sum. Energy-only (no contribution to H|ψ⟩).
pub const TermEwald = struct {
    alpha: f64,
    rcut: f64 = 0.0,
    gcut: f64 = 0.0,
    tol: f64 = 0.0,
    quiet: bool = false,
};

pub const Term = union(enum) {
    kinetic: TermKinetic,
    atomic_local: TermAtomicLocal,
    atomic_nonlocal: TermAtomicNonlocal,
    hartree: TermHartree,
    xc: TermXc,
    ewald: TermEwald,
};

/// Inputs passed to term energy evaluators.
///
/// Mandatory fields describe the physical system (geometry + species + config).
/// Optional fields carry state that only certain terms need (e.g. ρ for
/// Hartree/XC). Callers populate what they have; each term either uses it
/// or ignores it.
pub const EvalInput = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume_bohr: f64,
    rho: ?[]const f64 = null,
    grid: ?*const Grid = null,
};

/// Return the scalar energy contribution of a term.
///
/// Terms that do not yet route through this contract return 0 and are
/// still accumulated via the existing SCF code path. As each term is
/// wired, the corresponding legacy path is retired.
pub fn termEnergy(term: Term, input: EvalInput) !f64 {
    return switch (term) {
        .kinetic, .atomic_nonlocal => 0.0,
        .atomic_local => |t| try atomicLocalEnergy(t, input),
        .hartree => |t| try hartreeEnergy(t, input),
        .xc => |t| try xcEnergy(t, input),
        .ewald => |t| try ewaldEnergy(t, input),
    };
}

fn atomicLocalEnergy(term: TermAtomicLocal, input: EvalInput) !f64 {
    const grid = input.grid orelse return error.MissingGrid;
    const rho = input.rho orelse return error.MissingDensity;
    if (rho.len != grid.count()) return error.DensitySizeMismatch;

    const rho_g = try fft_grid.realToReciprocal(input.alloc, grid.*, rho, false);
    defer input.alloc.free(rho_g);

    const local_cfg = local_potential.resolve(term.mode, term.explicit_alpha, grid.cell);
    const inv_volume = 1.0 / grid.volume;

    var e_local: f64 = 0.0;
    var it = gvec_iter.GVecIterator.init(grid.*);
    while (it.next()) |g| {
        const rho_val = rho_g[g.idx];
        const vloc = try hamiltonian.ionicLocalPotential(g.gvec, input.species, input.atoms, inv_volume, local_cfg);
        e_local += rho_val.r * vloc.r + rho_val.i * vloc.i;
    }
    e_local *= grid.volume;
    return e_local;
}

fn xcEnergy(term: TermXc, input: EvalInput) !f64 {
    const grid = input.grid orelse return error.MissingGrid;
    const rho = input.rho orelse return error.MissingDensity;
    if (rho.len != grid.count()) return error.DensitySizeMismatch;

    const fields = try xc_fields.computeXcFields(input.alloc, grid.*, rho, null, false, term.functional);
    defer {
        input.alloc.free(fields.vxc);
        input.alloc.free(fields.exc);
    }

    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));
    var sum: f64 = 0.0;
    for (fields.exc) |e| sum += e * dv;
    return sum;
}

fn hartreeEnergy(term: TermHartree, input: EvalInput) !f64 {
    const grid = input.grid orelse return error.MissingGrid;
    const rho = input.rho orelse return error.MissingDensity;
    if (rho.len != grid.count()) return error.DensitySizeMismatch;

    const rho_g = try fft_grid.realToReciprocal(input.alloc, grid.*, rho, false);
    defer input.alloc.free(rho_g);

    const r_cut: ?f64 = if (term.isolated) coulomb_mod.cutoffRadius(grid.cell) else null;

    var eh: f64 = 0.0;
    var it = gvec_iter.GVecIterator.init(grid.*);
    while (it.next()) |g| {
        const rho_val = rho_g[g.idx];
        const rho2 = rho_val.r * rho_val.r + rho_val.i * rho_val.i;
        if (r_cut) |rc| {
            const g_mag = @sqrt(g.g2);
            const kernel = coulomb_mod.cutoffCoulombEnergyKernel(g.g2, g_mag, rc);
            eh += 0.5 * kernel * rho2 * grid.volume;
        } else {
            if (g.gh == 0 and g.gk == 0 and g.gl == 0) continue;
            if (g.g2 > 1e-12) {
                eh += 0.5 * 8.0 * std.math.pi * rho2 / g.g2 * grid.volume;
            }
        }
    }
    return eh;
}

fn ewaldEnergy(term: TermEwald, input: EvalInput) !f64 {
    const count = input.atoms.len;
    if (count == 0) return 0.0;
    const charges = try input.alloc.alloc(f64, count);
    defer input.alloc.free(charges);
    const positions = try input.alloc.alloc(math.Vec3, count);
    defer input.alloc.free(positions);
    for (input.atoms, 0..) |atom, i| {
        charges[i] = input.species[atom.species_index].z_valence;
        positions[i] = atom.position;
    }
    const params = ewald_mod.Params{
        .alpha = term.alpha,
        .rcut = term.rcut,
        .gcut = term.gcut,
        .tol = term.tol,
        .quiet = term.quiet,
    };
    return try ewald_mod.ionIonEnergy(input.io, input.cell_bohr, input.recip, charges, positions, params);
}

/// Build the canonical term list for a given configuration.
/// Caller owns the returned slice.
pub fn termsFromConfig(alloc: std.mem.Allocator, cfg: config_mod.Config) ![]Term {
    var list: std.ArrayList(Term) = .empty;
    errdefer list.deinit(alloc);

    try list.append(alloc, .{ .kinetic = .{} });
    try list.append(alloc, .{ .atomic_local = .{
        .mode = cfg.scf.local_potential,
        .explicit_alpha = cfg.ewald.alpha,
    } });
    if (cfg.scf.enable_nonlocal) {
        try list.append(alloc, .{ .atomic_nonlocal = .{} });
    }
    try list.append(alloc, .{ .hartree = .{ .isolated = (cfg.boundary == .isolated) } });
    try list.append(alloc, .{ .xc = .{ .functional = cfg.scf.xc } });
    try list.append(alloc, .{ .ewald = .{
        .alpha = cfg.ewald.alpha,
        .rcut = cfg.ewald.rcut,
        .gcut = cfg.ewald.gcut,
        .tol = cfg.ewald.tol,
        .quiet = cfg.scf.quiet,
    } });

    return list.toOwnedSlice(alloc);
}

test "termEnergy(.hartree) returns zero for uniform periodic density" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    const L = 8.0; // Bohr
    const cell = math.Mat3.fromRows(
        .{ .x = L, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = L, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = L },
    );
    const recip = math.reciprocal(cell);
    const volume = L * L * L;

    const nx: usize = 8;
    const grid = Grid{
        .nx = nx,
        .ny = nx,
        .nz = nx,
        .cell = cell,
        .recip = recip,
        .volume = volume,
        .min_h = grid_mod.minIndex(nx),
        .min_k = grid_mod.minIndex(nx),
        .min_l = grid_mod.minIndex(nx),
    };

    const n_points = grid.count();
    const rho = try alloc.alloc(f64, n_points);
    defer alloc.free(rho);
    @memset(rho, 1.0 / volume);

    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .species = &.{},
        .atoms = &.{},
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
        .rho = rho,
        .grid = &grid,
    };

    // Uniform ρ has non-zero Fourier component only at G=0, which
    // the periodic Hartree sum skips — so E_H must vanish.
    const eh = try termEnergy(.{ .hartree = .{} }, input);
    try testing.expectApproxEqAbs(eh, 0.0, 1e-10);
}

test "termEnergy(.hartree) is positive and deterministic for cosine density" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    const L = 8.0;
    const cell = math.Mat3.fromRows(
        .{ .x = L, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = L, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = L },
    );
    const recip = math.reciprocal(cell);
    const volume = L * L * L;

    const nx: usize = 8;
    const grid = Grid{
        .nx = nx,
        .ny = nx,
        .nz = nx,
        .cell = cell,
        .recip = recip,
        .volume = volume,
        .min_h = grid_mod.minIndex(nx),
        .min_k = grid_mod.minIndex(nx),
        .min_l = grid_mod.minIndex(nx),
    };

    const n_points = grid.count();
    const rho = try alloc.alloc(f64, n_points);
    defer alloc.free(rho);
    const twopi_L = 2.0 * std.math.pi / L;
    const rho0 = 1.0 / volume;
    const amp = 0.3 * rho0;
    for (0..nx) |ix| {
        const x = @as(f64, @floatFromInt(ix)) * L / @as(f64, @floatFromInt(nx));
        for (0..nx) |iy| {
            for (0..nx) |iz| {
                const idx = ix + nx * (iy + nx * iz);
                rho[idx] = rho0 + amp * @cos(twopi_L * x);
            }
        }
    }

    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .species = &.{},
        .atoms = &.{},
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
        .rho = rho,
        .grid = &grid,
    };

    const eh = try termEnergy(.{ .hartree = .{} }, input);
    // Positive by construction (cos density has real Fourier components at ±G₁).
    try testing.expect(eh > 0.0);
    // Deterministic: re-running gives the same value.
    const eh2 = try termEnergy(.{ .hartree = .{} }, input);
    try testing.expectApproxEqAbs(eh, eh2, 1e-14);
}

test "termEnergy(.atomic_local) is zero with no atoms" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    const L = 6.0;
    const cell = math.Mat3.fromRows(
        .{ .x = L, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = L, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = L },
    );
    const recip = math.reciprocal(cell);
    const volume = L * L * L;
    const nx: usize = 4;
    const grid = Grid{
        .nx = nx,
        .ny = nx,
        .nz = nx,
        .cell = cell,
        .recip = recip,
        .volume = volume,
        .min_h = grid_mod.minIndex(nx),
        .min_k = grid_mod.minIndex(nx),
        .min_l = grid_mod.minIndex(nx),
    };

    const rho = try alloc.alloc(f64, grid.count());
    defer alloc.free(rho);
    @memset(rho, 1.0 / volume);

    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .species = &.{},
        .atoms = &.{},
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
        .rho = rho,
        .grid = &grid,
    };

    const e = try termEnergy(.{ .atomic_local = .{} }, input);
    try testing.expectApproxEqAbs(e, 0.0, 1e-14);
}

test "termEnergy(.xc) matches computeXcFields integral (LDA)" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    const L = 6.0;
    const cell = math.Mat3.fromRows(
        .{ .x = L, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = L, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = L },
    );
    const recip = math.reciprocal(cell);
    const volume = L * L * L;

    const nx: usize = 6;
    const grid = Grid{
        .nx = nx,
        .ny = nx,
        .nz = nx,
        .cell = cell,
        .recip = recip,
        .volume = volume,
        .min_h = grid_mod.minIndex(nx),
        .min_k = grid_mod.minIndex(nx),
        .min_l = grid_mod.minIndex(nx),
    };

    const n_points = grid.count();
    const rho = try alloc.alloc(f64, n_points);
    defer alloc.free(rho);
    @memset(rho, 0.5 / volume);

    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .species = &.{},
        .atoms = &.{},
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
        .rho = rho,
        .grid = &grid,
    };

    const fields = try xc_fields.computeXcFields(alloc, grid, rho, null, false, .lda_pz);
    defer {
        alloc.free(fields.vxc);
        alloc.free(fields.exc);
    }
    const dv = volume / @as(f64, @floatFromInt(n_points));
    var expected: f64 = 0.0;
    for (fields.exc) |e| expected += e * dv;

    const actual = try termEnergy(.{ .xc = .{ .functional = .lda_pz } }, input);
    try testing.expectApproxEqRel(expected, actual, 1e-12);
}

test "termEnergy(.ewald) matches direct ionIonEnergy" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    // Graphene cell (Bohr) — reuse the established benchmark.
    const a = 4.6487262675;
    const c = 37.7945225;
    const cell = math.Mat3.fromRows(
        .{ .x = a, .y = 0.0, .z = 0.0 },
        .{ .x = a * 0.5, .y = a * std.math.sqrt(3.0) / 2.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = c },
    );
    const recip = math.reciprocal(cell);
    const volume = @abs(math.Vec3.dot(cell.row(0), math.Vec3.cross(cell.row(1), cell.row(2))));

    var species_arr = [_]hamiltonian.SpeciesEntry{
        .{ .symbol = "C", .upf = undefined, .z_valence = 4.0, .epsatm_ry = 0.0 },
    };
    const atoms = [_]hamiltonian.AtomData{
        .{ .position = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .species_index = 0 },
        .{ .position = .{ .x = 3.0991508450, .y = 2.6839433578, .z = 0.0 }, .species_index = 0 },
    };

    const charges = [_]f64{ 4.0, 4.0 };
    const positions = [_]math.Vec3{ atoms[0].position, atoms[1].position };
    const direct_params = ewald_mod.Params{ .alpha = 0.0, .rcut = 0.0, .gcut = 0.0, .tol = 0.0, .quiet = true };
    const e_direct = try ewald_mod.ionIonEnergy(io, cell, recip, &charges, &positions, direct_params);

    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .species = &species_arr,
        .atoms = &atoms,
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
    };
    const e_term = try termEnergy(.{ .ewald = .{ .alpha = 0.0, .quiet = true } }, input);

    try testing.expectApproxEqRel(e_direct, e_term, 1e-12);
}
