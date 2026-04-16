const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const d3_params = @import("../vdw/d3_params.zig");

/// Write electron density in Gaussian cube format.
/// Density is in electrons/Bohr³ (converted from Ry units).
/// Grid ordering: x-outer, y-middle, z-inner (Fortran convention).
pub fn writeCubeFile(
    dir: std.Io.Dir,
    filename: []const u8,
    density: []const f64,
    grid: [3]usize,
    cell_bohr: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    species: []const hamiltonian.SpeciesEntry,
) !void {
    const file = try dir.createFile(filename, .{});
    defer file.close();
    var buf: [512]u8 = undefined;
    var writer = file.writer(&buf);
    const out = &writer.interface;

    const nx = grid[0];
    const ny = grid[1];
    const nz = grid[2];

    // Header lines
    try out.print("DFT-Zig electron density\n", .{});
    try out.print("density in electrons/Bohr^3\n", .{});

    // Number of atoms and origin
    try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
        atoms.len,
        @as(f64, 0.0),
        @as(f64, 0.0),
        @as(f64, 0.0),
    });

    // Voxel vectors: a_i / n_i
    const a1 = cell_bohr.row(0);
    const a2 = cell_bohr.row(1);
    const a3 = cell_bohr.row(2);

    try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
        nx, a1.x / @as(f64, @floatFromInt(nx)), a1.y / @as(f64, @floatFromInt(nx)), a1.z / @as(f64, @floatFromInt(nx)),
    });
    try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
        ny, a2.x / @as(f64, @floatFromInt(ny)), a2.y / @as(f64, @floatFromInt(ny)), a2.z / @as(f64, @floatFromInt(ny)),
    });
    try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
        nz, a3.x / @as(f64, @floatFromInt(nz)), a3.y / @as(f64, @floatFromInt(nz)), a3.z / @as(f64, @floatFromInt(nz)),
    });

    // Atom positions (in Bohr)
    for (atoms) |atom| {
        const sym = species[atom.species_index].symbol;
        const z_num = d3_params.atomicNumber(sym) orelse 0;
        const z_val = species[atom.species_index].z_valence;
        try out.print("{d:>5} {d:>12.6} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{
            z_num,
            z_val,
            atom.position.x,
            atom.position.y,
            atom.position.z,
        });
    }

    // Density values: x-outer, y-middle, z-inner (cube convention)
    // DFT-Zig stores density as density[ix + nx*(iy + ny*iz)] where x is fastest in memory.
    // For cube format we loop ix (outer), iy, iz (inner) and read density[ix + nx*(iy + ny*iz)].
    // This means iz increments cause stride nx*ny jumps — acceptable for typical grid sizes
    // (32³ = 32K elements fits in L1 cache).
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
    try out.flush();
}
