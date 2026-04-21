const std = @import("std");
const math = @import("../math/math.zig");
const scf = @import("../scf/scf.zig");
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
    species: []const hamiltonian.SpeciesEntry,
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

    for (atoms, 0..) |atom, atom_index| {
        const entry = &species[atom.species_index];
        if (entry.upf.rho_atom.len == 0) continue;
        const pos = atom.position;

        var it = scf.GVecIterator.init(grid);
        while (it.next()) |g| {
            const g_norm = math.Vec3.norm(g.gvec);
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
                    const vresid = vresid_g[g.idx];
                    const phase = math.Vec3.dot(g.gvec, pos);
                    const cos_phase = std.math.cos(phase);
                    const sin_phase = std.math.sin(phase);
                    // Uses Re[i * v_resid(G) * exp(-iG·R)] for E_res = Re[V_resid(G) ρ_atom(G) exp(-iG·R)].
                    const phase_factor = vresid.r * sin_phase - vresid.i * cos_phase;
                    const coeff = rho_total_g * phase_factor;
                    forces[atom_index] = math.Vec3.add(forces[atom_index], math.Vec3.scale(g.gvec, coeff));
                }
            }
        }
    }

    return forces;
}
