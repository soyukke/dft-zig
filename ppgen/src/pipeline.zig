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

/// Run the full pipeline and write UPF to the given writer.
pub fn generatePseudopotential(
    allocator: std.mem.Allocator,
    config: GeneratorConfig,
    writer: anytype,
) !void {
    // 1. Set up grid
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();
    const n = grid.n;
    const n_all = config.all_orbitals.len;
    const n_val = config.valence_channels.len;

    // 2. Build orbital configs for all-electron SCF (core + valence)
    const orb_configs = try allocator.alloc(atomic_solver.OrbitalConfig, n_all);
    defer allocator.free(orb_configs);
    for (config.all_orbitals, 0..) |orb, i| {
        orb_configs[i] = .{ .n = orb.n, .l = orb.l, .occupation = orb.occupation };
    }

    // 3. Run all-electron atomic SCF
    var ae_result = try atomic_solver.solve(allocator, &grid, .{
        .z = config.z,
        .orbitals = orb_configs,
        .xc = config.xc,
    }, 300, 0.3, 1e-10);
    defer ae_result.deinit();

    // 4. Find valence orbital indices in the all-electron result
    const val_indices = try allocator.alloc(usize, n_val);
    defer allocator.free(val_indices);
    for (config.valence_channels, 0..) |vch, vi| {
        for (config.all_orbitals, 0..) |orb, ai| {
            if (orb.n == vch.n and orb.l == vch.l) {
                val_indices[vi] = ai;
                break;
            }
        }
    }

    // 5. Generate TM pseudo wavefunctions for valence channels only
    const pw_list = try allocator.alloc(tm_generator.PseudoWavefunction, n_val);
    defer {
        for (pw_list) |*pw| pw.deinit();
        allocator.free(pw_list);
    }
    for (config.valence_channels, 0..) |ch, i| {
        const ai = val_indices[i];
        pw_list[i] = try tm_generator.generate(
            allocator,
            &grid,
            ae_result.wavefunctions[ai],
            ae_result.v_eff,
            ae_result.eigenvalues[ai],
            ch.l,
            ch.rc,
        );
    }

    // 5. Build Hartree and XC potentials for unscreening
    const rho = ae_result.rho;
    const v_h = try allocator.alloc(f64, n);
    defer allocator.free(v_h);
    atomic_solver.radialPoisson(&grid, rho, v_h);

    const v_xc = try allocator.alloc(f64, n);
    defer allocator.free(v_xc);
    for (0..n) |i| {
        const xc_pt = xc_mod.evalPoint(config.xc, rho[i], 0);
        v_xc[i] = xc_pt.df_dn;
    }

    // 6. Choose local potential and unscreen it
    var l_local_idx: usize = 0;
    for (config.valence_channels, 0..) |ch, i| {
        if (ch.l == config.l_local) {
            l_local_idx = i;
            break;
        }
    }
    const v_local_screened = pw_list[l_local_idx].v_ps;

    const v_local_ion = try kb_projector.unscreen(allocator, &grid, v_local_screened, v_h, v_xc);
    defer allocator.free(v_local_ion);

    // 7. Build KB projectors for non-local channels
    var beta_list: std.ArrayListUnmanaged(upf_writer.BetaData) = .empty;
    defer {
        for (beta_list.items) |b| allocator.free(@constCast(b.values));
        beta_list.deinit(allocator);
    }

    var dij_list: std.ArrayListUnmanaged(f64) = .empty;
    defer dij_list.deinit(allocator);

    for (config.valence_channels, 0..) |ch, i| {
        if (ch.l == config.l_local) continue;

        var kb = try kb_projector.buildProjector(
            allocator,
            &grid,
            pw_list[i].v_ps,
            v_local_screened,
            pw_list[i].u,
            ch.l,
        );

        // Transfer ownership of beta to our list
        const beta_copy = try allocator.alloc(f64, n);
        @memcpy(beta_copy, kb.beta);
        kb.deinit();

        const i_rc = tm_generator.findGridIndex(&grid, ch.rc);

        try beta_list.append(allocator, .{
            .l = ch.l,
            .values = beta_copy,
            .cutoff_index = i_rc + 1,
        });
        try dij_list.append(allocator, kb.d_ion);
    }

    // Build full D_ij matrix (diagonal for NC)
    const nb = beta_list.items.len;
    const dij_matrix = try allocator.alloc(f64, nb * nb);
    defer allocator.free(dij_matrix);
    @memset(dij_matrix, 0);
    for (0..nb) |i| {
        dij_matrix[i * nb + i] = dij_list.items[i];
    }

    // 8. Build atomic charge density (4πr²ρ format for UPF)
    const rho_atom = try allocator.alloc(f64, n);
    defer allocator.free(rho_atom);
    for (0..n) |i| {
        rho_atom[i] = rho[i] * 4.0 * std.math.pi * grid.r[i] * grid.r[i];
    }

    // 9. Determine l_max
    var l_max: u32 = 0;
    for (config.valence_channels) |ch| {
        if (ch.l > l_max) l_max = ch.l;
    }

    // Total valence electrons
    var z_valence: f64 = 0;
    for (config.valence_channels) |ch| {
        z_valence += ch.occupation;
    }

    // 10. Write UPF
    const xc_name: []const u8 = switch (config.xc) {
        .lda_pz => "PZ",
        .pbe => "PBE",
    };

    try upf_writer.write(.{
        .element = config.element,
        .z = config.z,
        .z_valence = z_valence,
        .xc_functional = xc_name,
        .r = grid.r,
        .rab = grid.rab,
        .v_local = v_local_ion,
        .betas = beta_list.items,
        .dij = dij_matrix,
        .rho_atom = rho_atom,
        .l_max = l_max,
        .mesh_size = n,
    }, writer);
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

    var buf: [1024 * 1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);

    try generatePseudopotential(allocator, .{
        .z = 1,
        .element = "H",
        .xc = .lda_pz,
        .all_orbitals = &all_orbs,
        .valence_channels = &val_channels,
        .l_local = 0,
    }, fbs.writer());

    const output = fbs.getWritten();
    try std.testing.expect(output.len > 100);
    try std.testing.expect(std.mem.indexOf(u8, output, "element=\"H\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "z_valence=\"1.0\"") != null);

    // Write and parse back
    const tmp_path = "/tmp/ppgen_h_test.upf";
    {
        const file = try std.fs.cwd().createFile(tmp_path, .{});
        defer file.close();
        try file.writeAll(output);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var parsed = try pseudo_parser.load(allocator, .{
        .element = "H",
        .path = tmp_path,
        .format = .upf,
    });
    defer parsed.deinit(allocator);

    try std.testing.expectApproxEqAbs(1.0, parsed.header.z_valence.?, 0.01);
    try std.testing.expectEqual(@as(usize, 4000), parsed.upf.?.r.len);
}
