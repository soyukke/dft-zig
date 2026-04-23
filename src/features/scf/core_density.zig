const std = @import("std");
const fft_grid = @import("fft_grid.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const nonlocal_mod = @import("../pseudopotential/nonlocal.zig");
const UpfData = @import("../pseudopotential/pseudopotential.zig").UpfData;

const ctrapWeight = math.radial.ctrapWeight;
const reciprocalToReal = fft_grid.reciprocalToReal;

const Grid = grid_mod.Grid;

/// Check if any species provides NLCC data.
pub fn hasNlcc(species: []const hamiltonian.SpeciesEntry) bool {
    for (species) |entry| {
        if (entry.upf.nlcc.len > 0) return true;
    }
    return false;
}

/// Build core density on the real-space grid (assumes PP_NLCC stores rho_c(r)).
pub fn buildCoreDensity(
    alloc: std.mem.Allocator,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
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
                    const delta = minimumImage(
                        grid.cell,
                        grid.recip,
                        math.Vec3.sub(rvec, atom.position),
                    );
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

/// Compute atomic density form factor: ∫ rho_atom(r) × j₀(Gr) × rab(r) dr
/// UPF rho_atom includes 4πr² factor, so no extra r² needed.
fn rhoAtomFormFactor(upf: *const UpfData, g: f64) f64 {
    if (upf.rho_atom.len == 0) return 0.0;
    const n = @min(upf.rho_atom.len, @min(upf.r.len, upf.rab.len));
    var sum: f64 = 0.0;
    for (0..n) |i| {
        const x = g * upf.r[i];
        const j0 = nonlocal_mod.sphericalBessel(0, x);
        sum += upf.rho_atom[i] * j0 * upf.rab[i] * ctrapWeight(i, n);
    }
    return sum;
}

/// Build initial density from superposition of atomic pseudo-charge densities.
/// ρ_init(G) = (1/Ω) Σ_atom ρ_atom_form(|G|) × exp(-iG·R_atom)
/// Then inverse FFT to real space.
pub fn buildAtomicDensity(
    alloc: std.mem.Allocator,
    grid: Grid,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) ![]f64 {
    const total = grid.count();
    const inv_volume = 1.0 / grid.volume;
    const b1 = grid.recip.row(0);
    const b2 = grid.recip.row(1);
    const b3 = grid.recip.row(2);

    const rho_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho_g);

    var idx: usize = 0;
    var il: usize = 0;
    while (il < grid.nz) : (il += 1) {
        var ik: usize = 0;
        while (ik < grid.ny) : (ik += 1) {
            var ih: usize = 0;
            while (ih < grid.nx) : (ih += 1) {
                const gh = grid.min_h + @as(i32, @intCast(ih));
                const gk = grid.min_k + @as(i32, @intCast(ik));
                const gl = grid.min_l + @as(i32, @intCast(il));
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1, @as(f64, @floatFromInt(gh))),
                        math.Vec3.scale(b2, @as(f64, @floatFromInt(gk))),
                    ),
                    math.Vec3.scale(b3, @as(f64, @floatFromInt(gl))),
                );
                const g_norm = math.Vec3.norm(gvec);

                var sum_r: f64 = 0.0;
                var sum_i: f64 = 0.0;
                for (atoms) |atom| {
                    const rho_g_val = rhoAtomFormFactor(species[atom.species_index].upf, g_norm);
                    const g_dot_r = math.Vec3.dot(gvec, atom.position);
                    sum_r += rho_g_val * @cos(g_dot_r);
                    sum_i -= rho_g_val * @sin(g_dot_r);
                }
                rho_g[idx] = .{ .r = sum_r * inv_volume, .i = sum_i * inv_volume };
                idx += 1;
            }
        }
    }

    // Inverse FFT to real space
    const rho_r = try reciprocalToReal(alloc, grid, rho_g);
    return rho_r;
}
