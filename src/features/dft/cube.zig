const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const d3_params = @import("../vdw/d3_params.zig");

/// Emit the cube-format header: title, atom count/origin, and voxel vectors.
fn emit_cube_header(
    out: *std.Io.Writer,
    n_atoms: usize,
    grid: [3]usize,
    cell_bohr: math.Mat3,
) !void {
    const nx = grid[0];
    const ny = grid[1];
    const nz = grid[2];

    // Header lines
    try out.print("DFT-Zig electron density\n", .{});
    try out.print("density in electrons/Bohr^3\n", .{});

    // Number of atoms and origin
    try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
        n_atoms,
        @as(f64, 0.0),
        @as(f64, 0.0),
        @as(f64, 0.0),
    });

    // Voxel vectors: a_i / n_i
    const a1 = cell_bohr.row(0);
    const a2 = cell_bohr.row(1);
    const a3 = cell_bohr.row(2);
    const inv_nx: f64 = 1.0 / @as(f64, @floatFromInt(nx));
    const inv_ny: f64 = 1.0 / @as(f64, @floatFromInt(ny));
    const inv_nz: f64 = 1.0 / @as(f64, @floatFromInt(nz));

    try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
        nx, a1.x * inv_nx, a1.y * inv_nx, a1.z * inv_nx,
    });
    try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
        ny, a2.x * inv_ny, a2.y * inv_ny, a2.z * inv_ny,
    });
    try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
        nz, a3.x * inv_nz, a3.y * inv_nz, a3.z * inv_nz,
    });
}

/// Emit the atom block (atomic number, valence Z, Cartesian position in Bohr).
fn emit_cube_atoms(
    out: *std.Io.Writer,
    atoms: []const hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
) !void {
    for (atoms) |atom| {
        const sym = species[atom.species_index].symbol;
        const z_num = d3_params.atomic_number(sym) orelse 0;
        const z_val = species[atom.species_index].z_valence;
        try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
            z_num,
            z_val,
            atom.position.x,
            atom.position.y,
            atom.position.z,
        });
    }
}

/// Emit the volumetric density block in cube x-outer, y-middle, z-inner order.
/// DFT-Zig stores density as density[ix + nx*(iy + ny*iz)] where x is fastest in memory.
/// This means iz increments cause stride nx*ny jumps — acceptable for typical grid sizes
/// (32³ = 32K elements fits in L1 cache).
fn emit_cube_density(
    out: *std.Io.Writer,
    density: []const f64,
    grid: [3]usize,
) !void {
    const nx = grid[0];
    const ny = grid[1];
    const nz = grid[2];
    for (0..nx) |ix| {
        for (0..ny) |iy| {
            var count: usize = 0;
            for (0..nz) |iz| {
                const idx = ix + nx * (iy + ny * iz);
                // DFT-Zig density is already in electrons/Bohr³ (∫ρ dr = N_e)
                try out.print(" {e:>13.5}", .{density[idx]});
                count += 1;
                if (count % 6 == 0) {
                    try out.print("\n", .{});
                }
            }
            if (count % 6 != 0) {
                try out.print("\n", .{});
            }
        }
    }
}

/// Write electron density in Gaussian cube format.
/// Density is in electrons/Bohr³ (converted from Ry units).
/// Grid ordering: x-outer, y-middle, z-inner (Fortran convention).
pub fn write_cube_file(
    io: std.Io,
    dir: std.Io.Dir,
    filename: []const u8,
    density: []const f64,
    grid: [3]usize,
    cell_bohr: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
) !void {
    const file = try dir.createFile(io, filename, .{});
    defer file.close(io);

    var buf: [512]u8 = undefined;
    var writer = file.writer(io, &buf);
    const out = &writer.interface;

    try emit_cube_header(out, atoms.len, grid, cell_bohr);
    try emit_cube_atoms(out, atoms, species);
    try emit_cube_density(out, density, grid);
    try out.flush();
}
