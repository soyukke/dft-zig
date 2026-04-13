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
    channels: []const ChannelConfig,
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

    // 2. Build orbital configs for atomic SCF
    const orb_configs = try allocator.alloc(atomic_solver.OrbitalConfig, config.channels.len);
    defer allocator.free(orb_configs);
    for (config.channels, 0..) |ch, i| {
        orb_configs[i] = .{ .n = ch.n, .l = ch.l, .occupation = ch.occupation };
    }

    // 3. Run all-electron atomic SCF
    var ae_result = try atomic_solver.solve(allocator, &grid, .{
        .z = config.z,
        .orbitals = orb_configs,
        .xc = config.xc,
    }, 200, 0.3, 1e-10);
    defer ae_result.deinit();

    // 4. Generate TM pseudo wavefunctions for each channel
    const pw_list = try allocator.alloc(tm_generator.PseudoWavefunction, config.channels.len);
    defer {
        for (pw_list) |*pw| pw.deinit();
        allocator.free(pw_list);
    }
    for (config.channels, 0..) |ch, i| {
        pw_list[i] = try tm_generator.generate(
            allocator,
            &grid,
            ae_result.wavefunctions[i],
            ae_result.v_eff,
            ae_result.eigenvalues[i],
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
    for (config.channels, 0..) |ch, i| {
        if (ch.l == config.l_local) {
            l_local_idx = i;
            break;
        }
    }
    const v_local_screened = pw_list[l_local_idx].v_ps;

    const v_local_ion = try kb_projector.unscreen(allocator, &grid, v_local_screened, v_h, v_xc);
    defer allocator.free(v_local_ion);

    // 7. Build KB projectors for non-local channels
    var beta_list = std.ArrayList(upf_writer.BetaData).init(allocator);
    defer {
        for (beta_list.items) |b| allocator.free(@constCast(b.values));
        beta_list.deinit(allocator);
    }

    var dij_list = std.ArrayList(f64).init(allocator);
    defer dij_list.deinit(allocator);

    for (config.channels, 0..) |ch, i| {
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
    for (config.channels) |ch| {
        if (ch.l > l_max) l_max = ch.l;
    }

    // Total valence electrons
    var z_valence: f64 = 0;
    for (config.channels) |ch| {
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

test "end-to-end: H atom single channel UPF generation" {
    const allocator = std.testing.allocator;

    const channels = [_]ChannelConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0, .rc = 1.5 },
    };

    var buf: [1024 * 1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);

    try generatePseudopotential(allocator, .{
        .z = 1,
        .element = "H",
        .xc = .lda_pz,
        .channels = &channels,
        .l_local = 0, // s as local (no nonlocal projectors)
    }, fbs.writer());

    const output = fbs.getWritten();

    // Verify UPF structure
    try std.testing.expect(output.len > 100);
    try std.testing.expect(std.mem.indexOf(u8, output, "<UPF version=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "element=\"H\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "z_valence=\"1.0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "number_of_proj=\"0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "<PP_LOCAL") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "</UPF>") != null);

    std.debug.print("\n  UPF file size: {d} bytes\n", .{output.len});
}

test "end-to-end: H atom s+p channels with KB projector" {
    const allocator = std.testing.allocator;

    const channels = [_]ChannelConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0, .rc = 1.5 },
        .{ .n = 2, .l = 1, .occupation = 0.0, .rc = 1.5 },
    };

    var buf: [1024 * 1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);

    try generatePseudopotential(allocator, .{
        .z = 1,
        .element = "H",
        .xc = .lda_pz,
        .channels = &channels,
        .l_local = 1, // p as local, s as nonlocal
    }, fbs.writer());

    const output = fbs.getWritten();

    // Should have 1 KB projector (s-channel)
    try std.testing.expect(std.mem.indexOf(u8, output, "number_of_proj=\"1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "<PP_BETA.1") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "angular_momentum=\"0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "<PP_DIJ") != null);

    std.debug.print("\n  UPF file size: {d} bytes (with 1 KB projector)\n", .{output.len});
}

test "end-to-end: UPF round-trip parse" {
    // Generate UPF and parse it back with DFT-Zig parser
    const allocator = std.testing.allocator;

    const channels = [_]ChannelConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0, .rc = 1.5 },
        .{ .n = 2, .l = 1, .occupation = 0.0, .rc = 1.5 },
    };

    var buf: [1024 * 1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);

    try generatePseudopotential(allocator, .{
        .z = 1,
        .element = "H",
        .xc = .lda_pz,
        .channels = &channels,
        .l_local = 1,
    }, fbs.writer());

    const output = fbs.getWritten();

    // Write to temp file and parse back
    const tmp_path = "/tmp/ppgen_test.upf";
    {
        const file = try std.fs.cwd().createFile(tmp_path, .{});
        defer file.close();
        try file.writeAll(output);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    // Parse with DFT-Zig parser
    var parsed = try pseudo_parser.load(allocator, .{
        .element = "H",
        .path = tmp_path,
        .format = .upf,
    });
    defer parsed.deinit(allocator);

    // Verify parsed data
    const header = parsed.header;
    try std.testing.expect(header.z_valence != null);
    try std.testing.expectApproxEqAbs(1.0, header.z_valence.?, 0.01);
    try std.testing.expect(header.l_max != null);
    try std.testing.expectEqual(@as(i32, 1), header.l_max.?);
    try std.testing.expect(header.mesh_size != null);
    try std.testing.expectEqual(@as(usize, 4000), header.mesh_size.?);

    const upf = parsed.upf.?;
    // Grid should match
    try std.testing.expectEqual(@as(usize, 4000), upf.r.len);
    try std.testing.expectEqual(@as(usize, 4000), upf.v_local.len);

    // Should have 1 beta projector
    try std.testing.expectEqual(@as(usize, 1), upf.beta.len);
    try std.testing.expectEqual(@as(i32, 0), upf.beta[0].l.?);

    // DIJ should be 1x1
    try std.testing.expectEqual(@as(usize, 1), upf.dij.len);
    try std.testing.expect(upf.dij[0] != 0);

    std.debug.print("\n  === Round-trip parse successful ===\n", .{});
    std.debug.print("  z_valence = {d:.1}\n", .{header.z_valence.?});
    std.debug.print("  l_max = {d}\n", .{header.l_max.?});
    std.debug.print("  mesh_size = {d}\n", .{header.mesh_size.?});
    std.debug.print("  n_beta = {d}\n", .{upf.beta.len});
    std.debug.print("  D_11 = {d:.6} Ry\n", .{upf.dij[0]});
}
