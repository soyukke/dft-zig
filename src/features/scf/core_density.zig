const std = @import("std");
const grid_mod = @import("grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");

const Grid = grid_mod.Grid;

/// Check if any species provides NLCC data.
pub fn hasNlcc(species: []hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.nlcc.len > 0) return true;
    }
    return false;
}

/// Build core density on the real-space grid (assumes PP_NLCC stores rho_c(r)).
pub fn buildCoreDensity(
    alloc: std.mem.Allocator,
    grid: Grid,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
) ![]f64 {
    const total = grid.count();
    const rho_core = try alloc.alloc(f64, total);
    @memset(rho_core, 0.0);

    const a1 = grid.cell.row(0);
    const a2 = grid.cell.row(1);
    const a3 = grid.cell.row(2);

    var idx: usize = 0;
    var iz: usize = 0;
    while (iz < grid.nz) : (iz += 1) {
        const fz = @as(f64, @floatFromInt(iz)) / @as(f64, @floatFromInt(grid.nz));
        var iy: usize = 0;
        while (iy < grid.ny) : (iy += 1) {
            const fy = @as(f64, @floatFromInt(iy)) / @as(f64, @floatFromInt(grid.ny));
            var ix: usize = 0;
            while (ix < grid.nx) : (ix += 1) {
                const fx = @as(f64, @floatFromInt(ix)) / @as(f64, @floatFromInt(grid.nx));
                const rvec = math.Vec3.add(
                    math.Vec3.add(math.Vec3.scale(a1, fx), math.Vec3.scale(a2, fy)),
                    math.Vec3.scale(a3, fz),
                );
                var sum: f64 = 0.0;
                for (atoms) |atom| {
                    const entry = &species[atom.species_index];
                    if (entry.upf.nlcc.len == 0) continue;
                    const delta = minimumImage(grid.cell, grid.recip, math.Vec3.sub(rvec, atom.position));
                    const r = math.Vec3.norm(delta);
                    sum += sampleRadial(entry.upf.r, entry.upf.nlcc, r);
                }
                rho_core[idx] = sum;
                idx += 1;
            }
        }
    }

    return rho_core;
}

/// Sample radial function with linear interpolation.
fn sampleRadial(r_mesh: []f64, values: []f64, r: f64) f64 {
    if (r_mesh.len == 0 or values.len == 0) return 0.0;
    if (r <= r_mesh[0]) return values[0];
    if (r >= r_mesh[r_mesh.len - 1]) return 0.0;

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
    const t = (r - r0) / (r1 - r0);
    return values[low] * (1.0 - t) + values[high] * t;
}

/// Apply minimum-image convention using reciprocal vectors.
fn minimumImage(cell: math.Mat3, recip: math.Mat3, delta: math.Vec3) math.Vec3 {
    const two_pi = 2.0 * std.math.pi;
    var fx = math.Vec3.dot(recip.row(0), delta) / two_pi;
    var fy = math.Vec3.dot(recip.row(1), delta) / two_pi;
    var fz = math.Vec3.dot(recip.row(2), delta) / two_pi;
    fx -= @round(fx);
    fy -= @round(fy);
    fz -= @round(fz);
    const a1 = cell.row(0);
    const a2 = cell.row(1);
    const a3 = cell.row(2);
    return math.Vec3.add(
        math.Vec3.add(math.Vec3.scale(a1, fx), math.Vec3.scale(a2, fy)),
        math.Vec3.scale(a3, fz),
    );
}
