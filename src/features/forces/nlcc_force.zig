const std = @import("std");
const math = @import("../math/math.zig");
const scf = @import("../scf/scf.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_force = @import("local_force.zig");

pub const Grid = local_force.Grid;

/// Parameters for the per-atom real-space NLCC accumulation loop.
const NlccRealLoopInputs = struct {
    grid: Grid,
    vxc_r: []const f64,
    vol_per_point: f64,
    two_pi: f64,
    a1: math.Vec3,
    a2: math.Vec3,
    a3: math.Vec3,
};

/// Accumulate the NLCC force on a single atom by summing over real-space grid points.
fn accumulate_nlcc_force_real(
    inputs: NlccRealLoopInputs,
    entry: *const hamiltonian.SpeciesEntry,
    pos: math.Vec3,
) math.Vec3 {
    var fx: f64 = 0.0;
    var fy: f64 = 0.0;
    var fz: f64 = 0.0;

    var idx: usize = 0;
    var iz: usize = 0;
    while (iz < inputs.grid.nz) : (iz += 1) {
        const frac_z = @as(f64, @floatFromInt(iz)) / @as(f64, @floatFromInt(inputs.grid.nz));
        var iy: usize = 0;
        while (iy < inputs.grid.ny) : (iy += 1) {
            const frac_y = @as(f64, @floatFromInt(iy)) / @as(f64, @floatFromInt(inputs.grid.ny));
            var ix: usize = 0;
            while (ix < inputs.grid.nx) : (ix += 1) {
                const frac_x = @as(f64, @floatFromInt(ix)) /
                    @as(f64, @floatFromInt(inputs.grid.nx));
                // Grid point in Cartesian coordinates
                const rvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(inputs.a1, frac_x),
                        math.Vec3.scale(inputs.a2, frac_y),
                    ),
                    math.Vec3.scale(inputs.a3, frac_z),
                );
                // Minimum-image displacement from atom to grid point
                const delta_raw = math.Vec3.sub(rvec, pos);
                const delta = minimum_image(
                    inputs.grid.cell,
                    inputs.grid.recip,
                    inputs.two_pi,
                    delta_raw,
                );
                const r = math.Vec3.norm(delta);

                if (r > 1e-10) {
                    const drho = sample_radial_derivative(entry.upf.r, entry.upf.nlcc, r);
                    if (drho != 0.0) {
                        // F_{I,α} = (Ω/N) Σ_n V_xc(r_n) × ρ'(r) × d_α/r
                        const coeff = inputs.vol_per_point * inputs.vxc_r[idx] * drho / r;
                        fx += coeff * delta.x;
                        fy += coeff * delta.y;
                        fz += coeff * delta.z;
                    }
                }
                idx += 1;
            }
        }
    }

    return math.Vec3{ .x = fx, .y = fy, .z = fz };
}

/// NLCC force correction computed in real space.
///
/// F_{I,α} = (Ω/N) Σ_n V_xc(r_n) × ρ'_core(|r_n - R_I|) × (r_n - R_I)_α / |r_n - R_I|
///
/// This real-space approach is consistent with how E_xc uses the grid-sampled
/// core density, avoiding aliasing issues with the reciprocal-space form factor.
pub fn nlcc_forces(
    alloc: std.mem.Allocator,
    grid: Grid,
    vxc_r: []const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    volume: f64,
) ![]math.Vec3 {
    const n_atoms = atoms.len;
    const ngrid = grid.nx * grid.ny * grid.nz;
    if (vxc_r.len != ngrid) return error.InvalidGrid;

    const forces = try alloc.alloc(math.Vec3, n_atoms);
    for (forces) |*f| {
        f.* = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    }

    const inputs = NlccRealLoopInputs{
        .grid = grid,
        .vxc_r = vxc_r,
        .vol_per_point = volume / @as(f64, @floatFromInt(ngrid)),
        .two_pi = 2.0 * std.math.pi,
        .a1 = grid.cell.row(0),
        .a2 = grid.cell.row(1),
        .a3 = grid.cell.row(2),
    };

    for (atoms, 0..) |atom, atom_index| {
        const entry = &species[atom.species_index];
        if (entry.upf.nlcc.len == 0) continue;
        forces[atom_index] = accumulate_nlcc_force_real(inputs, entry, atom.position);
    }

    return forces;
}

/// Sample the radial derivative dρ_core/dr using linear interpolation of the tabulated data.
fn sample_radial_derivative(r_mesh: []const f64, values: []const f64, r: f64) f64 {
    if (r_mesh.len < 2 or values.len < 2) return 0.0;
    if (r >= r_mesh[r_mesh.len - 1]) return 0.0;
    if (r <= r_mesh[0]) return 0.0;

    // Binary search for the interval containing r
    var low: usize = 0;
    var high: usize = r_mesh.len - 1;
    while (high - low > 1) {
        const mid = (low + high) / 2;
        if (r_mesh[mid] <= r) {
            low = mid;
        } else {
            high = mid;
        }
    }
    const r0 = r_mesh[low];
    const r1 = r_mesh[high];
    if (r1 - r0 < 1e-30) return 0.0;
    return (values[high] - values[low]) / (r1 - r0);
}

/// NLCC force computed in reciprocal space.
///
/// F_{a,α} = Σ_{G≠0} ρ_core_form(|G|) × G_α × [V_xc_r sin(G·R_a) + V_xc_i cos(G·R_a)]
///
/// This is derived from F = -dE_xc/dR_a = ∫ V_xc(r) × ∇ρ_core(r-R_a) dr.
/// Using Parseval's theorem with code convention f̃(G) = (1/N) Σ_n f(r_n) exp(-iGr_n):
///   E = Σ_G Ṽ_xc(G) × ρ_form(|G|) × exp(iG·R_a)
///   F_α = -dE/dR_α = Σ_G ρ_form × G_α × Re[-i Ṽ_xc(G) exp(iG·R)]
///       = Σ_G ρ_form × G_α × (V_r sin + V_i cos)
///
/// The G-space approach avoids aliasing between the bandwidth-limited V_xc and
/// the tabulated radial core charge derivative, which is critical for PAW
/// where the compensation charge n̂ makes V_xc more structured.
pub fn nlcc_forces_g_space(
    alloc: std.mem.Allocator,
    grid: Grid,
    vxc_g: []const math.Complex,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) ![]math.Vec3 {
    const n_atoms = atoms.len;
    const ngrid = grid.nx * grid.ny * grid.nz;
    if (vxc_g.len != ngrid) return error.InvalidGrid;

    var forces = try alloc.alloc(math.Vec3, n_atoms);
    for (forces) |*f| {
        f.* = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    }

    for (atoms, 0..) |atom, atom_index| {
        const entry = &species[atom.species_index];
        if (entry.upf.nlcc.len == 0) continue;
        const pos = atom.position;

        var fx: f64 = 0.0;
        var fy: f64 = 0.0;
        var fz: f64 = 0.0;

        var it = scf.GVecIterator.init(grid);
        while (it.next()) |g| {
            // Skip G=0
            if (g.gh == 0 and g.gk == 0 and g.gl == 0) {
                continue;
            }

            const g_norm = math.Vec3.norm(g.gvec);

            // Core charge form factor: ρ_core(G) = 4π ∫ r² ρ_core(r) j0(Gr) dr
            const rho_core_g = if (rho_core_tables) |tables|
                tables[atom.species_index].eval(g_norm)
            else
                form_factor.rho_core_g(entry.upf.*, g_norm);

            if (rho_core_g != 0.0) {
                const vxc = vxc_g[g.idx];
                const phase = math.Vec3.dot(g.gvec, pos);
                const cos_phase = std.math.cos(phase);
                const sin_phase = std.math.sin(phase);

                // Phase factor: V_r sin(G·R) + V_i cos(G·R)
                // = Re[-i × V̂_xc(G) × exp(iG·R)]
                const phase_factor = vxc.r * sin_phase + vxc.i * cos_phase;
                const coeff = rho_core_g * phase_factor;

                fx += g.gvec.x * coeff;
                fy += g.gvec.y * coeff;
                fz += g.gvec.z * coeff;
            }
        }

        forces[atom_index] = math.Vec3{ .x = fx, .y = fy, .z = fz };
    }

    return forces;
}

/// Apply minimum-image convention using reciprocal vectors.
fn minimum_image(cell: math.Mat3, recip: math.Mat3, two_pi: f64, delta: math.Vec3) math.Vec3 {
    var frac_x = math.Vec3.dot(recip.row(0), delta) / two_pi;
    var frac_y = math.Vec3.dot(recip.row(1), delta) / two_pi;
    var frac_z = math.Vec3.dot(recip.row(2), delta) / two_pi;
    frac_x -= @round(frac_x);
    frac_y -= @round(frac_y);
    frac_z -= @round(frac_z);
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    return math.Vec3.add(
        math.Vec3.add(math.Vec3.scale(a1, frac_x), math.Vec3.scale(a2, frac_y)),
        math.Vec3.scale(a3, frac_z),
    );
}
