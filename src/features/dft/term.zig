//! Physical terms of the DFT Hamiltonian.
//!
//! Each Term carries the parameters for one density-dependent contribution
//! to the total energy (Hartree, XC, local pseudo, Ewald). Evaluation goes
//! through `term_energy(term, input)`.
//!
//! Wavefunction-dependent contributions (kinetic, nonlocal pseudopotential)
//! are not represented here — they are handled inside the band solver,
//! where ψ is available. When a ψ-aware Term contract is introduced, those
//! variants join this enum.

const std = @import("std");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const ewald_mod = @import("../ewald/ewald.zig");
const fft_grid = @import("../scf/fft_grid.zig");
const grid_mod = @import("../scf/pw_grid.zig");
const gvec_iter = @import("../scf/gvec_iter.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const xc_fields = @import("../scf/xc_fields.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const model_mod = @import("model.zig");
const xc_mod = @import("../xc/xc.zig");

pub const Grid = grid_mod.Grid;
pub const Model = model_mod.Model;

/// Local component of the ionic pseudopotential: V_loc(r) = Σ_a v^a_loc(r-R_a).
/// Depends on atomic positions (via Model), not on ρ. Contributes to V_eff(G).
///
/// mode and explicit_alpha parameterize the long-range splitting; resolved
/// against the cell at evaluation time (the effective alpha depends on geometry).
pub const TermAtomicLocal = struct {
    mode: local_potential.LocalPotentialMode = .short_range,
    explicit_alpha: f64 = 0.0,
    /// Optional spherical G² cutoff (PAW augmented-density regime).
    ecutrho: ?f64 = null,
};

/// Hartree: V_H(G) = 8π/G² ρ(G) (Rydberg).
/// Uses a real-space Coulomb cutoff for isolated boundary conditions.
pub const TermHartree = struct {
    /// true for isolated boundary conditions (Coulomb cutoff applied at eval time).
    isolated: bool = false,
    /// Optional spherical G² cutoff — PAW uses this to truncate the
    /// augmented-density sum when the density grid is denser than ecut.
    ecutrho: ?f64 = null,
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
    atomic_local: TermAtomicLocal,
    hartree: TermHartree,
    xc: TermXc,
    ewald: TermEwald,
};

/// Inputs passed to term energy evaluators.
///
/// `model` supplies the physical system (geometry, species). Optional
/// fields carry state that only certain terms need — callers populate
/// what they have; each term either uses it or returns an error.
pub const EvalInput = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    model: *const Model,
    rho: ?[]const f64 = null,
    /// Minority-spin density. When present, XC evaluation dispatches to
    /// the spin-polarized path (`rho` becomes the majority-spin density).
    rho_down: ?[]const f64 = null,
    /// Core density (NLCC) added to ρ for XC evaluation when present.
    rho_core: ?[]const f64 = null,
    grid: ?*const Grid = null,
    /// Use real-to-complex FFT for ρ→ρ(G) and gradient transforms.
    /// Matches the SCF driver's `use_rfft` so GGA gradients computed by
    /// the evaluator line up with the externally-held V_xc.
    use_rfft: bool = false,
};

/// Return the scalar energy contribution of a term.
///
/// Terms that do not yet route through this contract return 0 and are
/// still accumulated via the existing SCF code path. As each term is
/// wired, the corresponding legacy path is retired.
pub fn term_energy(term: Term, input: EvalInput) !f64 {
    return switch (term) {
        .atomic_local => |t| try atomic_local_energy(t, input),
        .hartree => |t| try hartree_energy(t, input),
        .xc => |t| try xc_energy(t, input),
        .ewald => |t| try ewald_energy(t, input),
    };
}

fn atomic_local_energy(term: TermAtomicLocal, input: EvalInput) !f64 {
    const grid = input.grid orelse return error.MissingGrid;
    const rho = input.rho orelse return error.MissingDensity;
    if (rho.len != grid.count()) return error.DensitySizeMismatch;

    const rho_g = try fft_grid.real_to_reciprocal(input.alloc, grid.*, rho, false);
    defer input.alloc.free(rho_g);

    const local_cfg = local_potential.resolve(term.mode, term.explicit_alpha, grid.cell);
    const inv_volume = 1.0 / grid.volume;

    var e_local: f64 = 0.0;
    var it = gvec_iter.GVecIterator.init(grid.*);
    while (it.next()) |g| {
        if (term.ecutrho) |ecut| {
            if (g.g2 >= ecut) continue;
        }
        const rho_val = rho_g[g.idx];
        const vloc = try hamiltonian.ionic_local_potential(
            g.gvec,
            input.model.species,
            input.model.atoms,
            inv_volume,
            local_cfg,
        );
        e_local += rho_val.r * vloc.r + rho_val.i * vloc.i;
    }
    e_local *= grid.volume;
    return e_local;
}

fn xc_energy(term: TermXc, input: EvalInput) !f64 {
    const grid = input.grid orelse return error.MissingGrid;
    const rho = input.rho orelse return error.MissingDensity;
    if (rho.len != grid.count()) return error.DensitySizeMismatch;

    // NOTE: the SCF energy path also calls compute_xc_fields for V_xc, so
    // this evaluator currently duplicates that work. A follow-up will
    // extend term_energy to expose V_xc alongside E_xc, letting the SCF
    // driver drop its own compute_xc_fields call.
    const dv = grid.volume / @as(f64, @floatFromInt(grid.count()));

    if (input.rho_down) |rho_down| {
        if (rho_down.len != grid.count()) return error.DensitySizeMismatch;
        const fields = try xc_fields.compute_xc_fields_spin(
            input.alloc,
            grid.*,
            rho,
            rho_down,
            input.rho_core,
            input.use_rfft,
            term.functional,
        );
        defer {
            input.alloc.free(fields.vxc_up);
            input.alloc.free(fields.vxc_down);
            input.alloc.free(fields.exc);
        }

        var sum: f64 = 0.0;
        for (fields.exc) |e| sum += e * dv;
        return sum;
    }

    const fields = try xc_fields.compute_xc_fields(
        input.alloc,
        grid.*,
        rho,
        input.rho_core,
        input.use_rfft,
        term.functional,
    );
    defer {
        input.alloc.free(fields.vxc);
        input.alloc.free(fields.exc);
    }

    var sum: f64 = 0.0;
    for (fields.exc) |e| sum += e * dv;
    return sum;
}

fn hartree_energy(term: TermHartree, input: EvalInput) !f64 {
    const grid = input.grid orelse return error.MissingGrid;
    const rho = input.rho orelse return error.MissingDensity;
    if (rho.len != grid.count()) return error.DensitySizeMismatch;

    const rho_g = try fft_grid.real_to_reciprocal(input.alloc, grid.*, rho, false);
    defer input.alloc.free(rho_g);

    const r_cut: ?f64 = if (term.isolated) coulomb_mod.cutoff_radius(grid.cell) else null;

    var eh: f64 = 0.0;
    var it = gvec_iter.GVecIterator.init(grid.*);
    while (it.next()) |g| {
        if (term.ecutrho) |ecut| {
            if (g.g2 >= ecut) continue;
        }
        const rho_val = rho_g[g.idx];
        const rho2 = rho_val.r * rho_val.r + rho_val.i * rho_val.i;
        if (r_cut) |rc| {
            const g_mag = @sqrt(g.g2);
            const kernel = coulomb_mod.cutoff_coulomb_energy_kernel(g.g2, g_mag, rc);
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

fn ewald_energy(term: TermEwald, input: EvalInput) !f64 {
    const atoms = input.model.atoms;
    const count = atoms.len;
    if (count == 0) return 0.0;
    const charges = try input.alloc.alloc(f64, count);
    defer input.alloc.free(charges);

    const positions = try input.alloc.alloc(math.Vec3, count);
    defer input.alloc.free(positions);

    for (atoms, 0..) |atom, i| {
        charges[i] = input.model.species[atom.species_index].z_valence;
        positions[i] = atom.position;
    }
    const params = ewald_mod.Params{
        .alpha = term.alpha,
        .rcut = term.rcut,
        .gcut = term.gcut,
        .tol = term.tol,
        .quiet = term.quiet,
    };
    return try ewald_mod.ion_ion_energy(
        input.io,
        input.model.cell_bohr,
        input.model.recip,
        charges,
        positions,
        params,
    );
}

test "term_energy(.hartree) returns zero for uniform periodic density" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    const L = 8.0; // Bohr
    const cell = math.Mat3.from_rows(
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
        .min_h = grid_mod.min_index(nx),
        .min_k = grid_mod.min_index(nx),
        .min_l = grid_mod.min_index(nx),
    };

    const n_points = grid.count();
    const rho = try alloc.alloc(f64, n_points);
    defer alloc.free(rho);

    @memset(rho, 1.0 / volume);

    const model = Model{
        .species = &.{},
        .atoms = &.{},
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
    };
    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .model = &model,
        .rho = rho,
        .grid = &grid,
    };

    // Uniform ρ has non-zero Fourier component only at G=0, which
    // the periodic Hartree sum skips — so E_H must vanish.
    const eh = try term_energy(.{ .hartree = .{} }, input);
    try testing.expectApproxEqAbs(eh, 0.0, 1e-10);
}

test "term_energy(.hartree) is positive and deterministic for cosine density" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    const L = 8.0;
    const cell = math.Mat3.from_rows(
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
        .min_h = grid_mod.min_index(nx),
        .min_k = grid_mod.min_index(nx),
        .min_l = grid_mod.min_index(nx),
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

    const model = Model{
        .species = &.{},
        .atoms = &.{},
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
    };
    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .model = &model,
        .rho = rho,
        .grid = &grid,
    };

    const eh = try term_energy(.{ .hartree = .{} }, input);
    // Positive by construction (cos density has real Fourier components at ±G₁).
    try testing.expect(eh > 0.0);
    // Deterministic: re-running gives the same value.
    const eh2 = try term_energy(.{ .hartree = .{} }, input);
    try testing.expectApproxEqAbs(eh, eh2, 1e-14);
}

test "term_energy(.atomic_local) is zero with no atoms" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    const L = 6.0;
    const cell = math.Mat3.from_rows(
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
        .min_h = grid_mod.min_index(nx),
        .min_k = grid_mod.min_index(nx),
        .min_l = grid_mod.min_index(nx),
    };

    const rho = try alloc.alloc(f64, grid.count());
    defer alloc.free(rho);

    @memset(rho, 1.0 / volume);

    const model = Model{
        .species = &.{},
        .atoms = &.{},
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
    };
    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .model = &model,
        .rho = rho,
        .grid = &grid,
    };

    const e = try term_energy(.{ .atomic_local = .{} }, input);
    try testing.expectApproxEqAbs(e, 0.0, 1e-14);
}

test "term_energy(.xc) matches compute_xc_fields integral (LDA)" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    const L = 6.0;
    const cell = math.Mat3.from_rows(
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
        .min_h = grid_mod.min_index(nx),
        .min_k = grid_mod.min_index(nx),
        .min_l = grid_mod.min_index(nx),
    };

    const n_points = grid.count();
    const rho = try alloc.alloc(f64, n_points);
    defer alloc.free(rho);

    @memset(rho, 0.5 / volume);

    const model = Model{
        .species = &.{},
        .atoms = &.{},
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
    };
    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .model = &model,
        .rho = rho,
        .grid = &grid,
    };

    const fields = try xc_fields.compute_xc_fields(alloc, grid, rho, null, false, .lda_pz);
    defer {
        alloc.free(fields.vxc);
        alloc.free(fields.exc);
    }

    const dv = volume / @as(f64, @floatFromInt(n_points));
    var expected: f64 = 0.0;
    for (fields.exc) |e| expected += e * dv;

    const actual = try term_energy(.{ .xc = .{ .functional = .lda_pz } }, input);
    try testing.expectApproxEqRel(expected, actual, 1e-12);
}

test "term_energy(.ewald) matches direct ion_ion_energy" {
    const testing = std.testing;
    const io = testing.io;
    const alloc = testing.allocator;

    // Graphene cell (Bohr) — reuse the established benchmark.
    const a = 4.6487262675;
    const c = 37.7945225;
    const cell = math.Mat3.from_rows(
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
    const direct_params = ewald_mod.Params{
        .alpha = 0.0,
        .rcut = 0.0,
        .gcut = 0.0,
        .tol = 0.0,
        .quiet = true,
    };
    const e_direct = try ewald_mod.ion_ion_energy(
        io,
        cell,
        recip,
        &charges,
        &positions,
        direct_params,
    );

    const model = Model{
        .species = &species_arr,
        .atoms = &atoms,
        .cell_bohr = cell,
        .recip = recip,
        .volume_bohr = volume,
    };
    const input = EvalInput{
        .alloc = alloc,
        .io = io,
        .model = &model,
    };
    const e_term = try term_energy(.{ .ewald = .{ .alpha = 0.0, .quiet = true } }, input);

    try testing.expectApproxEqRel(e_direct, e_term, 1e-12);
}
