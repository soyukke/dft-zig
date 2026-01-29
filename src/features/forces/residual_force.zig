const std = @import("std");
const math = @import("../math/math.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const local_force = @import("local_force.zig");

pub const Grid = local_force.Grid;

/// Residual-potential force correction using atomic valence density.
/// Uses ρ_atom(G) from PP_RHOATOM and v_resid(G) from SCF.
pub fn residualForces(
    alloc: std.mem.Allocator,
    grid: Grid,
    vresid_g: []const math.Complex,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho_atom_tables: ?[]const form_factor.RadialFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) ![]math.Vec3 {
    const n_atoms = atoms.len;
    const ngrid = grid.nx * grid.ny * grid.nz;
    if (vresid_g.len != ngrid) return error.InvalidGrid;

    var forces = try alloc.alloc(math.Vec3, n_atoms);
    for (forces) |*f| {
        f.* = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    }

    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    for (atoms, 0..) |atom, atom_index| {
        const entry = &species[atom.species_index];
        if (entry.upf.rho_atom.len == 0) continue;
        const pos = atom.position;

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
                    const gvec = math.Vec3.add(
                        math.Vec3.add(math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2, @as(f64, @floatFromInt(gk)))),
                        math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                    );
                    const g_norm = math.Vec3.norm(gvec);
                    if (g_norm > 1e-12) {
                        const rho_atom_g = if (rho_atom_tables) |tables|
                            tables[atom.species_index].eval(g_norm)
                        else
                            form_factor.rhoAtomG(entry.upf.*, g_norm);
                        const rho_core_g = if (rho_core_tables) |tables|
                            tables[atom.species_index].eval(g_norm)
                        else
                            form_factor.rhoCoreG(entry.upf.*, g_norm);
                        const rho_total_g = rho_atom_g + rho_core_g;
                        if (rho_total_g != 0.0) {
                            const vresid = vresid_g[idx];
                            const phase = math.Vec3.dot(gvec, pos);
                            const cos_phase = std.math.cos(phase);
                            const sin_phase = std.math.sin(phase);
                            // Uses Re[i * v_resid(G) * exp(-iG·R)] for E_res = Re[V_resid(G) ρ_atom(G) exp(-iG·R)].
                            const phase_factor = vresid.r * sin_phase - vresid.i * cos_phase;
                            const coeff = rho_total_g * phase_factor;
                            forces[atom_index] = math.Vec3.add(forces[atom_index], math.Vec3.scale(gvec, coeff));
                        }
                    }
                    idx += 1;
                }
            }
        }
    }

    return forces;
}
