const std = @import("std");

const ionic_potential = @import("ionic_potential.zig");
const local_orbital = @import("local_orbital.zig");
const local_orbital_potential = @import("local_orbital_potential.zig");
const local_orbital_scf = @import("local_orbital_scf.zig");
const neighbor_list = @import("neighbor_list.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const xyz = @import("../structure/xyz.zig");
const xc = @import("../xc/xc.zig");
const math = @import("../math/math.zig");

pub const ScfRunOptions = struct {
    units: math.Units = .bohr,
    grid_dims: [3]usize,
    sigma: f64,
    cutoff: f64,
    max_iter: usize,
    density_tol: f64,
    electrons: f64,
    xc: xc.Functional,
    kinetic_scale: f64 = 2.0,
    matrix_threshold: f64 = 0.0,
    purification_iters: usize = 3,
    purification_threshold: f64 = 0.0,
    /// Enable nonlocal pseudopotential (KB projectors)
    use_nonlocal: bool = true,
    /// Number of radial integration points for nonlocal
    nonlocal_n_radial: usize = 100,
    /// Maximum radius for nonlocal integration (Bohr)
    nonlocal_r_max: f64 = 10.0,
    /// Threshold for dropping small nonlocal matrix elements
    nonlocal_threshold: f64 = 0.0,
    /// Basis type for nonlocal calculation: s_only or sp
    /// sp basis includes p-type orbitals for better p-wave projector overlap
    nonlocal_basis: local_orbital.BasisType = .s_only,
};

pub fn loadPseudos(
    alloc: std.mem.Allocator,
    io: std.Io,
    specs: []const pseudo.Spec,
) ![]pseudo.Parsed {
    var list: std.ArrayList(pseudo.Parsed) = .empty;
    errdefer {
        for (list.items) |*item| {
            item.deinit(alloc);
        }
        list.deinit(alloc);
    }
    for (specs) |spec| {
        const parsed = try pseudo.load(alloc, io, spec);
        try list.append(alloc, parsed);
    }
    return try list.toOwnedSlice(alloc);
}

pub fn deinitPseudos(alloc: std.mem.Allocator, pseudos: []pseudo.Parsed) void {
    for (pseudos) |*item| {
        item.deinit(alloc);
    }
    if (pseudos.len > 0) {
        alloc.free(pseudos);
    }
}

pub fn buildIonSitesFromAtoms(
    alloc: std.mem.Allocator,
    atoms: []const xyz.Atom,
    pseudos: []const pseudo.Parsed,
    scale_to_bohr: f64,
) ![]ionic_potential.IonSite {
    if (atoms.len == 0) return error.InvalidShape;
    const sites = try alloc.alloc(ionic_potential.IonSite, atoms.len);
    errdefer alloc.free(sites);
    for (atoms, 0..) |atom, idx| {
        const upf = findUpfForSymbol(pseudos, atom.symbol) orelse
            return error.MissingPseudopotential;
        sites[idx] = .{ .position = math.Vec3.scale(atom.position, scale_to_bohr), .upf = upf };
    }
    return sites;
}

pub fn runScfFromAtoms(
    alloc: std.mem.Allocator,
    atoms: []const xyz.Atom,
    pseudos: []const pseudo.Parsed,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: ScfRunOptions,
) !local_orbital_scf.ScfGridResult {
    if (opts.grid_dims[0] == 0 or opts.grid_dims[1] == 0 or opts.grid_dims[2] == 0) {
        return error.InvalidGrid;
    }
    const scale = math.unitsScaleToBohr(opts.units);
    const scaled_cell = cell.scale(scale);
    const sigma = opts.sigma * scale;
    const cutoff = opts.cutoff * scale;
    const grid = local_orbital_potential.PotentialGrid{
        .cell = scaled_cell,
        .dims = opts.grid_dims,
        .values = &[_]f64{},
    };

    const centers = try alloc.alloc(math.Vec3, atoms.len);
    defer alloc.free(centers);
    for (atoms, 0..) |atom, idx| {
        centers[idx] = math.Vec3.scale(atom.position, scale);
    }
    const sites = try buildIonSitesFromAtoms(alloc, atoms, pseudos, scale);
    defer alloc.free(sites);

    const scf_opts = local_orbital_scf.ScfGridOptions{
        .sigma = sigma,
        .cutoff = cutoff,
        .grid = grid,
        .max_iter = opts.max_iter,
        .density_tol = opts.density_tol,
        .electrons = opts.electrons,
        .xc = opts.xc,
        .ionic = null,
        .kinetic_scale = opts.kinetic_scale,
        .matrix_threshold = opts.matrix_threshold,
        .purification_iters = opts.purification_iters,
        .purification_threshold = opts.purification_threshold,
        .nonlocal_ions = if (opts.use_nonlocal) sites else null,
        .nonlocal_n_radial = opts.nonlocal_n_radial,
        .nonlocal_r_max = opts.nonlocal_r_max,
        .nonlocal_threshold = opts.nonlocal_threshold,
        .nonlocal_basis = opts.nonlocal_basis,
    };
    return local_orbital_scf.runScfWithGridAndIons(
        alloc,
        centers,
        scaled_cell,
        pbc,
        sites,
        scf_opts,
    );
}

pub fn runScfFromXyz(
    alloc: std.mem.Allocator,
    io: std.Io,
    xyz_path: []const u8,
    specs: []const pseudo.Spec,
    cell: math.Mat3,
    pbc: neighbor_list.Pbc,
    opts: ScfRunOptions,
) !local_orbital_scf.ScfGridResult {
    var atom_list = try xyz.load(alloc, io, xyz_path);
    defer atom_list.deinit(alloc);
    const pseudos = try loadPseudos(alloc, io, specs);
    defer deinitPseudos(alloc, pseudos);
    return runScfFromAtoms(alloc, atom_list.items, pseudos, cell, pbc, opts);
}

fn findUpfForSymbol(pseudos: []const pseudo.Parsed, symbol: []const u8) ?*const pseudo.UpfData {
    for (pseudos) |*item| {
        if (std.mem.eql(u8, item.spec.element, symbol)) {
            if (item.upf) |*data| return data;
            return null;
        }
        if (item.header.element) |elem| {
            if (std.mem.eql(u8, elem, symbol)) {
                if (item.upf) |*data| return data;
                return null;
            }
        }
    }
    return null;
}

test "buildIonSitesFromAtoms maps symbols" {
    const alloc = std.testing.allocator;
    const symbol_buf = [_]u8{ 'S', 'i' };
    const spec = pseudo.Spec{
        .element = symbol_buf[0..],
        .path = symbol_buf[0..],
        .format = .upf,
    };
    var r = try alloc.alloc(f64, 2);
    defer alloc.free(r);
    r[0] = 0.0;
    r[1] = 1.0;
    var rab = try alloc.alloc(f64, 2);
    defer alloc.free(rab);
    rab[0] = 0.0;
    rab[1] = 1.0;
    var v_local = try alloc.alloc(f64, 2);
    defer alloc.free(v_local);
    v_local[0] = 0.0;
    v_local[1] = 0.0;
    const upf = pseudo.UpfData{
        .r = r,
        .rab = rab,
        .v_local = v_local,
        .beta = &[_]pseudo.Beta{},
        .dij = &[_]f64{},
        .qij = &[_]f64{},
        .nlcc = &[_]f64{},
    };
    const parsed = pseudo.Parsed{
        .spec = spec,
        .header = .{ .element = null, .z_valence = null, .l_max = null, .mesh_size = null },
        .upf = upf,
    };
    const atoms = [_]xyz.Atom{
        .{ .symbol = symbol_buf[0..], .position = .{ .x = 1.0, .y = 2.0, .z = 3.0 } },
        .{ .symbol = symbol_buf[0..], .position = .{ .x = 2.0, .y = 3.0, .z = 4.0 } },
    };
    const pseudos = [_]pseudo.Parsed{parsed};
    const sites = try buildIonSitesFromAtoms(alloc, atoms[0..], pseudos[0..], 2.0);
    defer alloc.free(sites);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), sites[0].position.x, 1e-12);
    try std.testing.expect(sites[0].upf == &upf);
}
