//! End-to-end pseudopotential generation pipeline.
//!
//! atomic SCF → TM pseudization → KB projectors → UPF output → parser round-trip

const std = @import("std");
const dft_zig = @import("dft_zig");
const xc_mod = dft_zig.xc;
const pseudo_parser = dft_zig.pseudopotential;

const RadialGrid = @import("radial_grid.zig").RadialGrid;
const atomic_solver = @import("atomic_solver.zig");
const schrodinger = @import("schrodinger.zig");
const tm_generator = @import("tm_generator.zig");
const kb_projector = @import("kb_projector.zig");
const upf_writer = @import("upf_writer.zig");

pub const OrbitalDef = struct {
    n: u32,
    l: u32,
    occupation: f64,
};

pub const ChannelConfig = struct {
    n: u32,
    l: u32,
    occupation: f64,
    rc: f64, // cutoff radius (Bohr)
};

pub const GeneratorConfig = struct {
    z: f64,
    element: []const u8,
    xc: xc_mod.Functional,
    /// All orbitals for the all-electron calculation (core + valence)
    all_orbitals: []const OrbitalDef,
    /// Valence channels to pseudize (subset of all_orbitals)
    valence_channels: []const ChannelConfig,
    l_local: u32, // which l to use as local potential
};

const NonLocalData = struct {
    betas: std.ArrayListUnmanaged(upf_writer.BetaData) = .empty,
    dij: []f64,
    allocator: std.mem.Allocator,

    fn deinit(self: *NonLocalData) void {
        for (self.betas.items) |b| self.allocator.free(@constCast(b.values));
        self.betas.deinit(self.allocator);
        self.allocator.free(self.dij);
    }
};

/// Run the full pipeline and write UPF to the given writer.
pub fn generate_pseudopotential(
    allocator: std.mem.Allocator,
    config: GeneratorConfig,
    writer: anytype,
) !void {
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    const orb_configs = try build_orbital_configs(allocator, config.all_orbitals);
    defer allocator.free(orb_configs);

    var ae_result = try atomic_solver.solve(allocator, &grid, .{
        .z = config.z,
        .orbitals = orb_configs,
        .xc = config.xc,
    }, 300, 0.3, 1e-10);
    defer ae_result.deinit();

    const val_indices = try find_valence_indices(allocator, config);
    defer allocator.free(val_indices);

    const pw_list = try generate_pseudo_wavefunctions(
        allocator,
        &grid,
        config,
        &ae_result,
        val_indices,
    );
    defer deinit_pseudo_wavefunctions(allocator, pw_list);

    const rho_val = try build_valence_density(allocator, &grid, config.valence_channels, pw_list);
    defer allocator.free(rho_val);

    const v_h = try build_hartree_potential(allocator, &grid, rho_val);
    defer allocator.free(v_h);

    const v_xc = try build_valence_xc_potential(allocator, &grid, config.xc, rho_val);
    defer allocator.free(v_xc);

    const l_local_idx = find_local_channel_index(config.valence_channels, config.l_local);
    const v_local_screened = pw_list[l_local_idx].v_ps;
    const v_local_ion = try kb_projector.unscreen(allocator, &grid, v_local_screened, v_h, v_xc);
    defer allocator.free(v_local_ion);

    var nonlocal = try build_nonlocal_data(
        allocator,
        &grid,
        config.valence_channels,
        config.l_local,
        pw_list,
        v_local_screened,
    );
    defer nonlocal.deinit();

    const rho_atom = try build_atomic_rho(allocator, &grid, rho_val);
    defer allocator.free(rho_atom);

    try write_pseudopotential_data(
        writer,
        config,
        &grid,
        v_local_ion,
        nonlocal.betas.items,
        nonlocal.dij,
        rho_atom,
    );
}

fn write_pseudopotential_data(
    writer: anytype,
    config: GeneratorConfig,
    grid: *const RadialGrid,
    v_local_ion: []const f64,
    betas: []const upf_writer.BetaData,
    dij: []const f64,
    rho_atom: []const f64,
) !void {
    try upf_writer.write(.{
        .element = config.element,
        .z = config.z,
        .z_valence = total_valence_charge(config.valence_channels),
        .xc_functional = xc_name(config.xc),
        .r = grid.r,
        .rab = grid.rab,
        .v_local = v_local_ion,
        .betas = betas,
        .dij = dij,
        .rho_atom = rho_atom,
        .l_max = max_angular_momentum(config.valence_channels),
        .mesh_size = grid.n,
    }, writer);
}

fn xc_name(xc: xc_mod.Functional) []const u8 {
    return switch (xc) {
        .lda_pz => "PZ",
        .pbe => "PBE",
    };
}

fn build_orbital_configs(
    allocator: std.mem.Allocator,
    all_orbitals: []const OrbitalDef,
) ![]atomic_solver.OrbitalConfig {
    const orb_configs = try allocator.alloc(atomic_solver.OrbitalConfig, all_orbitals.len);
    for (all_orbitals, 0..) |orb, i| {
        orb_configs[i] = .{ .n = orb.n, .l = orb.l, .occupation = orb.occupation };
    }
    return orb_configs;
}

fn find_valence_indices(
    allocator: std.mem.Allocator,
    config: GeneratorConfig,
) ![]usize {
    const val_indices = try allocator.alloc(usize, config.valence_channels.len);
    for (config.valence_channels, 0..) |vch, vi| {
        val_indices[vi] = 0;
        for (config.all_orbitals, 0..) |orb, ai| {
            if (orb.n == vch.n and orb.l == vch.l) {
                val_indices[vi] = ai;
                break;
            }
        }
    }
    return val_indices;
}

fn generate_pseudo_wavefunctions(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    config: GeneratorConfig,
    ae_result: *const atomic_solver.AtomResult,
    val_indices: []const usize,
) ![]tm_generator.PseudoWavefunction {
    const pw_list =
        try allocator.alloc(tm_generator.PseudoWavefunction, config.valence_channels.len);
    errdefer allocator.free(pw_list);

    var initialized: usize = 0;
    errdefer {
        for (pw_list[0..initialized]) |*pw| pw.deinit();
    }

    for (config.valence_channels, 0..) |ch, i| {
        const ai = val_indices[i];
        pw_list[i] = try tm_generator.generate(
            allocator,
            grid,
            ae_result.wavefunctions[ai],
            ae_result.v_eff,
            ae_result.eigenvalues[ai],
            ch.l,
            ch.rc,
        );
        initialized += 1;
    }
    return pw_list;
}

fn deinit_pseudo_wavefunctions(
    allocator: std.mem.Allocator,
    pw_list: []tm_generator.PseudoWavefunction,
) void {
    for (pw_list) |*pw| pw.deinit();
    allocator.free(pw_list);
}

fn build_valence_density(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    valence_channels: []const ChannelConfig,
    pw_list: []const tm_generator.PseudoWavefunction,
) ![]f64 {
    const rho_val = try allocator.alloc(f64, grid.n);
    @memset(rho_val, 0);
    for (valence_channels, pw_list) |ch, pw| {
        const occ = ch.occupation;
        for (0..grid.n) |i| {
            const r = grid.r[i];
            const r2 = if (r > 1e-30) r * r else 1e-30;
            rho_val[i] += occ * pw.u[i] * pw.u[i] / (4.0 * std.math.pi * r2);
        }
    }
    return rho_val;
}

fn build_hartree_potential(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    rho_val: []const f64,
) ![]f64 {
    const v_h = try allocator.alloc(f64, grid.n);
    atomic_solver.radial_poisson(grid, rho_val, v_h);
    return v_h;
}

fn build_valence_xc_potential(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    xc: xc_mod.Functional,
    rho_val: []const f64,
) ![]f64 {
    const v_xc = try allocator.alloc(f64, grid.n);
    for (rho_val, 0..) |rho_i, i| {
        const xc_pt = xc_mod.eval_point(xc, rho_i, 0);
        v_xc[i] = xc_pt.df_dn;
    }
    return v_xc;
}

fn find_local_channel_index(valence_channels: []const ChannelConfig, l_local: u32) usize {
    for (valence_channels, 0..) |ch, i| {
        if (ch.l == l_local) return i;
    }
    return 0;
}

fn build_nonlocal_data(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    valence_channels: []const ChannelConfig,
    l_local: u32,
    pw_list: []const tm_generator.PseudoWavefunction,
    v_local_screened: []const f64,
) !NonLocalData {
    var betas: std.ArrayListUnmanaged(upf_writer.BetaData) = .empty;
    errdefer {
        for (betas.items) |b| allocator.free(@constCast(b.values));
        betas.deinit(allocator);
    }

    var dij_diag: std.ArrayListUnmanaged(f64) = .empty;
    defer dij_diag.deinit(allocator);

    for (valence_channels, pw_list) |ch, pw| {
        if (ch.l == l_local) continue;

        var kb = try kb_projector.build_projector(
            allocator,
            grid,
            pw.v_ps,
            v_local_screened,
            pw.u,
            ch.l,
        );
        defer kb.deinit();

        const beta_copy = try allocator.alloc(f64, grid.n);
        @memcpy(beta_copy, kb.beta);

        try betas.append(allocator, .{
            .l = ch.l,
            .values = beta_copy,
            .cutoff_index = tm_generator.find_grid_index(grid, ch.rc) + 1,
        });
        try dij_diag.append(allocator, kb.d_ion);
    }

    const dij = try build_diagonal_dij_matrix(allocator, dij_diag.items);
    return .{ .betas = betas, .dij = dij, .allocator = allocator };
}

fn build_diagonal_dij_matrix(
    allocator: std.mem.Allocator,
    diagonal: []const f64,
) ![]f64 {
    const nb = diagonal.len;
    const dij = try allocator.alloc(f64, nb * nb);
    @memset(dij, 0);
    for (diagonal, 0..) |value, i| {
        dij[i * nb + i] = value;
    }
    return dij;
}

fn build_atomic_rho(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    rho_val: []const f64,
) ![]f64 {
    const rho_atom = try allocator.alloc(f64, grid.n);
    for (0..grid.n) |i| {
        rho_atom[i] = rho_val[i] * 4.0 * std.math.pi * grid.r[i] * grid.r[i];
    }
    return rho_atom;
}

fn max_angular_momentum(valence_channels: []const ChannelConfig) u32 {
    var l_max: u32 = 0;
    for (valence_channels) |ch| {
        if (ch.l > l_max) l_max = ch.l;
    }
    return l_max;
}

fn total_valence_charge(valence_channels: []const ChannelConfig) f64 {
    var z_valence: f64 = 0;
    for (valence_channels) |ch| {
        z_valence += ch.occupation;
    }
    return z_valence;
}

// ============================================================================
// Tests
// ============================================================================

test "end-to-end: H atom s+p UPF round-trip" {
    const allocator = std.testing.allocator;

    const all_orbs = [_]OrbitalDef{
        .{ .n = 1, .l = 0, .occupation = 1.0 },
    };
    const val_channels = [_]ChannelConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0, .rc = 1.5 },
    };

    const io = std.testing.io;
    var buf: [1024 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);

    try generate_pseudopotential(allocator, .{
        .z = 1,
        .element = "H",
        .xc = .lda_pz,
        .all_orbitals = &all_orbs,
        .valence_channels = &val_channels,
        .l_local = 0,
    }, &writer);

    const output = writer.buffered();
    try std.testing.expect(output.len > 100);
    try std.testing.expect(std.mem.indexOf(u8, output, "element=\"H\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "z_valence=\"1.0\"") != null);

    // Write and parse back
    const tmp_path = "/tmp/ppgen_h_test.upf";
    const cwd = std.Io.Dir.cwd();
    {
        const file = try cwd.createFile(io, tmp_path, .{});
        defer file.close(io);

        try file.writeStreamingAll(io, output);
    }
    defer cwd.deleteFile(io, tmp_path) catch {};

    var parsed = try pseudo_parser.load(allocator, io, .{
        .element = "H",
        .path = tmp_path,
        .format = .upf,
    });
    defer parsed.deinit(allocator);

    try std.testing.expectApproxEqAbs(1.0, parsed.header.z_valence.?, 0.01);
    try std.testing.expectEqual(@as(usize, 4000), parsed.upf.?.r.len);
}
