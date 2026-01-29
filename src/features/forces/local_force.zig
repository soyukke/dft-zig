const std = @import("std");
const math = @import("../math/math.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");

/// Grid parameters for force calculation
pub const Grid = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    min_h: i32,
    min_k: i32,
    min_l: i32,
    cell: math.Mat3,
    recip: math.Mat3,
};

/// Compute local pseudopotential forces in reciprocal space.
/// F_loc(R_I) = Σ_G G × V_form(|G|) × [ρ_r sin(G·R) - ρ_i cos(G·R)]
///
/// Derived from Hellmann-Feynman theorem:
/// E_local = Σ_G ρ(G) × V_form(|G|) × S(G), where S(G) = Σ_I exp(-iG·R_I)
/// F_I = -∂E_local/∂R_I = Σ_G G × V_form(|G|) × Re[i × ρ(G) × exp(-iG·R_I)]
///
/// Returns forces in Rydberg/Bohr units.
pub fn localPseudoForces(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_g: []const math.Complex,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    _: f64, // volume (unused, kept for API compatibility)
    _: f64, // alpha (only needed for Ewald method)
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
) ![]math.Vec3 {
    const n_atoms = atoms.len;
    if (n_atoms == 0) return &[_]math.Vec3{};

    // Allocate force array
    var forces = try alloc.alloc(math.Vec3, n_atoms);
    for (forces) |*f| {
        f.* = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    }

    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    // Hellmann-Feynman force from local pseudopotential:
    //
    // Energy: E_local = Σ_G ρ(G) × V_form(|G|) × S(G)
    // where S(G) = Σ_I exp(-iG·R_I) is the structure factor.
    //
    // Force on atom I:
    // F_I = -∂E_local/∂R_I
    //     = Σ_G ρ(G) × V_form(|G|) × iG × exp(+iG·R_I)
    //     = Σ_G G × V_form(|G|) × Re[i × ρ(G) × exp(+iG·R_I)]
    //     = Σ_G G × V_form(|G|) × [ρ_r sin(G·R) + ρ_i cos(G·R)]
    // This matches ionicLocalPotential's phase exp(-iG·R) via E_loc = Re[ρ(G) V_loc(G)].

    // Loop over all G vectors
    var idx: usize = 0;
    var l: usize = 0;
    while (l < grid.nz) : (l += 1) {
        var k: usize = 0;
        while (k < grid.ny) : (k += 1) {
            var h: usize = 0;
            while (h < grid.nx) : (h += 1) {
                const gh = grid.min_h + @as(i32, @intCast(h));
                const gk = grid.min_k + @as(i32, @intCast(k));
                const gl = grid.min_l + @as(i32, @intCast(l));

                // Skip G=0 (no force contribution)
                if (gh == 0 and gk == 0 and gl == 0) {
                    idx += 1;
                    continue;
                }

                // Compute G vector
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_norm = math.Vec3.norm(gvec);

                // Get electron density at G
                const rho = rho_g[idx];

                // Loop over atoms
                for (atoms, 0..) |atom, atom_idx| {
                    // Get local pseudopotential form factor for this species
                    // Use the same form factor mode as SCF for consistency
                    const v_loc = if (ff_tables) |tables|
                        tables[atom.species_index].eval(g_norm)
                    else
                        hamiltonian.localFormFactor(&species[atom.species_index], g_norm);

                    // Phase: G·R
                    const phase = math.Vec3.dot(gvec, atom.position);
                    const cos_phase = std.math.cos(phase);
                    const sin_phase = std.math.sin(phase);

                    // Energy uses Re[ρ(G) × V_loc(G)*], so the force uses
                    // Re[i × ρ(G) × exp(+iG·R)] = ρ_r sin(G·R) + ρ_i cos(G·R)
                    const phase_factor = rho.r * sin_phase + rho.i * cos_phase;

                    // Force contribution: G × V_loc × phase_factor
                    const force_factor = v_loc * phase_factor;
                    const force_contrib = math.Vec3.scale(gvec, force_factor);

                    forces[atom_idx] = math.Vec3.add(forces[atom_idx], force_contrib);
                }

                idx += 1;
            }
        }
    }

    return forces;
}

test "local force finite difference" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var element_buf: [2]u8 = .{ 'S', 'i' };
    var path_buf: [24]u8 = undefined;
    const path_slice = "pseudo/Si.upf";
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..2],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(alloc, spec);
    defer parsed.deinit(alloc);

    var parsed_items = [_]pseudo.Parsed{parsed};
    const species = try hamiltonian.buildSpeciesEntries(alloc, parsed_items[0..]);
    defer {
        for (species) |*entry| {
            entry.deinit();
        }
        alloc.free(species);
    }

    // Create a minimal grid
    const a = 5.0;
    const grid = Grid{
        .nx = 4,
        .ny = 4,
        .nz = 4,
        .min_h = -1,
        .min_k = -1,
        .min_l = -1,
        .cell = math.Mat3{ .m = .{
            .{ a, 0.0, 0.0 },
            .{ 0.0, a, 0.0 },
            .{ 0.0, 0.0, a },
        } },
        .recip = math.Mat3{ .m = .{
            .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
            .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
            .{ 0.0, 0.0, 2.0 * std.math.pi / a },
        } },
    };
    const volume = a * a * a;

    const atom_pos = math.Vec3{ .x = 0.3, .y = 0.4, .z = 0.1 };
    const atoms = [_]hamiltonian.AtomData{
        .{ .position = atom_pos, .species_index = 0 },
    };

    // Create a simple reciprocal-space density with Hermitian symmetry
    var rho_g = try alloc.alloc(math.Complex, 64);
    defer alloc.free(rho_g);
    for (rho_g) |*r| {
        r.* = math.complex.init(0.0, 0.0);
    }

    const idx = struct {
        fn of(g: Grid, gh: i32, gk: i32, gl: i32) usize {
            const h = @as(usize, @intCast(gh - g.min_h));
            const k = @as(usize, @intCast(gk - g.min_k));
            const l = @as(usize, @intCast(gl - g.min_l));
            return h + g.nx * (k + g.ny * l);
        }
    };

    const rho_x = math.complex.init(0.01, 0.02);
    const rho_y = math.complex.init(0.015, -0.005);
    rho_g[idx.of(grid, 1, 0, 0)] = rho_x;
    rho_g[idx.of(grid, -1, 0, 0)] = math.complex.conj(rho_x);
    rho_g[idx.of(grid, 0, 1, 0)] = rho_y;
    rho_g[idx.of(grid, 0, -1, 0)] = math.complex.conj(rho_y);

    const forces = try localPseudoForces(alloc, grid, rho_g, species, atoms[0..], volume, 0.0, null);
    defer alloc.free(forces);

    const energyForPos = struct {
        fn eval(g: Grid, rho: []const math.Complex, species_entries: []const hamiltonian.SpeciesEntry, pos: math.Vec3) f64 {
            const b1 = g.recip.row(0);
            const b2 = g.recip.row(1);
            const b3 = g.recip.row(2);
            var energy: f64 = 0.0;
            var idx_local: usize = 0;
            var l: usize = 0;
            while (l < g.nz) : (l += 1) {
                var k: usize = 0;
                while (k < g.ny) : (k += 1) {
                    var h: usize = 0;
                    while (h < g.nx) : (h += 1) {
                        const gh = g.min_h + @as(i32, @intCast(h));
                        const gk = g.min_k + @as(i32, @intCast(k));
                        const gl = g.min_l + @as(i32, @intCast(l));
                        if (gh == 0 and gk == 0 and gl == 0) {
                            idx_local += 1;
                            continue;
                        }
                        const gvec = math.Vec3.add(
                            math.Vec3.add(
                                math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                                math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                            ),
                            math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                        );
                        const g_norm = math.Vec3.norm(gvec);
                        const rho_val = rho[idx_local];
                        const sp = &species_entries[0];
                        const v_loc = hamiltonian.localFormFactor(sp, g_norm);
                        const phase = math.Vec3.dot(gvec, pos);
                        const cos_phase = std.math.cos(phase);
                        const sin_phase = std.math.sin(phase);
                        energy += v_loc * (rho_val.r * cos_phase - rho_val.i * sin_phase);
                        idx_local += 1;
                    }
                }
            }
            return energy;
        }
    };

    const delta = 1e-5;
    const fx_num = blk: {
        const pos_plus = math.Vec3{ .x = atom_pos.x + delta, .y = atom_pos.y, .z = atom_pos.z };
        const pos_minus = math.Vec3{ .x = atom_pos.x - delta, .y = atom_pos.y, .z = atom_pos.z };
        const e_plus = energyForPos.eval(grid, rho_g, species, pos_plus);
        const e_minus = energyForPos.eval(grid, rho_g, species, pos_minus);
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };
    const fy_num = blk: {
        const pos_plus = math.Vec3{ .x = atom_pos.x, .y = atom_pos.y + delta, .z = atom_pos.z };
        const pos_minus = math.Vec3{ .x = atom_pos.x, .y = atom_pos.y - delta, .z = atom_pos.z };
        const e_plus = energyForPos.eval(grid, rho_g, species, pos_plus);
        const e_minus = energyForPos.eval(grid, rho_g, species, pos_minus);
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };
    const fz_num = blk: {
        const pos_plus = math.Vec3{ .x = atom_pos.x, .y = atom_pos.y, .z = atom_pos.z + delta };
        const pos_minus = math.Vec3{ .x = atom_pos.x, .y = atom_pos.y, .z = atom_pos.z - delta };
        const e_plus = energyForPos.eval(grid, rho_g, species, pos_plus);
        const e_minus = energyForPos.eval(grid, rho_g, species, pos_minus);
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };

    try testing.expectApproxEqAbs(forces[0].x, fx_num, 1e-5);
    try testing.expectApproxEqAbs(forces[0].y, fy_num, 1e-5);
    try testing.expectApproxEqAbs(forces[0].z, fz_num, 1e-5);
}
