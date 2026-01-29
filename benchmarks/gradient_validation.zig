//! Gradient validation: B3LYP/6-31G(2df,p) analytical gradient vs PySCF reference.
//!
//! Validates that the KS-DFT analytical gradient implementation matches PySCF
//! for H2O at the same geometry used in qm9_validation.
//!
//! Usage: zig build run-gradient-test -Doptimize=ReleaseFast

const std = @import("std");
const dft_zig = @import("dft_zig");
const math_mod = dft_zig.math;
const basis_mod = dft_zig.basis;
const gto_scf = dft_zig.gto_scf;
const kohn_sham = gto_scf.kohn_sham;
const gradient_mod = gto_scf.gradient;
const grid_mod = dft_zig.grid;
const becke = grid_mod.becke;
const obara_saika = dft_zig.integrals_mod.obara_saika;

const ContractedShell = basis_mod.ContractedShell;
const Vec3 = math_mod.Vec3;
const KsParams = kohn_sham.KsParams;
const b631g2dfp = basis_mod.basis631g_2dfp;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("========================================================================\n", .{});
    std.debug.print("  Gradient Validation: B3LYP/6-31G(2df,p) H2O vs PySCF\n", .{});
    std.debug.print("  Grid: 50 radial, 302 angular (Lebedev), no pruning\n", .{});
    std.debug.print("========================================================================\n", .{});

    // === H2O geometry (same as qm9_validation) ===
    var nuc_positions = [_]Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.2217971552 }, // O
        .{ .x = 0.0, .y = 1.4308250325, .z = -0.8871886210 }, // H
        .{ .x = 0.0, .y = -1.4308250325, .z = -0.8871886210 }, // H
    };
    const nuc_charges = [_]f64{ 8.0, 1.0, 1.0 };
    const n_electrons: usize = 10;

    // Build basis
    const o_data = b631g2dfp.buildAtomShells(8, nuc_positions[0]).?;
    const h1_data = b631g2dfp.buildAtomShells(1, nuc_positions[1]).?;
    const h2_data = b631g2dfp.buildAtomShells(1, nuc_positions[2]).?;

    var all_shells: [b631g2dfp.MAX_SHELLS_PER_ATOM * 3]ContractedShell = undefined;
    var count: usize = 0;
    for (o_data.shells[0..o_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h1_data.shells[0..h1_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    for (h2_data.shells[0..h2_data.count]) |s| {
        all_shells[count] = s;
        count += 1;
    }
    const shells = all_shells[0..count];

    const n_basis = obara_saika.totalBasisFunctions(shells);
    std.debug.print("  Atoms:            {d}\n", .{nuc_charges.len});
    std.debug.print("  Electrons:        {d}\n", .{n_electrons});
    std.debug.print("  Basis functions:  {d}\n", .{n_basis});

    // Step 1: Run KS-DFT SCF
    std.debug.print("\n  [Step 1] Running KS-DFT SCF...\n", .{});
    var ks_result = try kohn_sham.runKohnShamScf(alloc, shells, &nuc_positions, &nuc_charges, n_electrons, .{
        .xc_functional = .b3lyp,
        .n_radial = 50,
        .n_angular = 302,
        .prune = false,
        .use_direct_scf = false,
        .schwarz_threshold = 1e-12,
        .use_diis = true,
        .verbose = true,
    });
    defer ks_result.deinit(alloc);

    std.debug.print("  SCF converged:    {}\n", .{ks_result.converged});
    std.debug.print("  Total energy:     {d:.12} Ha\n", .{ks_result.total_energy});

    // Step 2: Build grid for gradient
    std.debug.print("\n  [Step 2] Building molecular grid for gradient...\n", .{});
    const n_atoms = nuc_charges.len;
    const atoms = try alloc.alloc(becke.Atom, n_atoms);
    defer alloc.free(atoms);
    for (0..n_atoms) |i| {
        atoms[i] = .{
            .x = nuc_positions[i].x,
            .y = nuc_positions[i].y,
            .z = nuc_positions[i].z,
            .z_number = @intFromFloat(nuc_charges[i]),
        };
    }

    const grid_config = becke.GridConfig{
        .n_radial = 50,
        .n_angular = 302,
        .prune = false,
        .use_atomic_radii = true,
        .becke_hardness = 3,
    };
    const grid_points = try becke.buildMolecularGrid(alloc, atoms, grid_config);
    defer alloc.free(grid_points);
    std.debug.print("  Grid points:      {d}\n", .{grid_points.len});

    // Step 3: Compute analytical gradient
    std.debug.print("\n  [Step 3] Computing analytical gradient...\n", .{});
    const n_occ = n_electrons / 2;
    var timer = try std.time.Timer.start();
    var grad_result = try gradient_mod.computeKsDftGradient(
        alloc,
        shells,
        &nuc_positions,
        &nuc_charges,
        ks_result.density_matrix_result,
        ks_result.orbital_energies,
        ks_result.mo_coefficients,
        n_occ,
        grid_points,
        .b3lyp,
    );
    defer grad_result.deinit(alloc);
    const grad_time = @as(f64, @floatFromInt(timer.read())) / 1e9;
    std.debug.print("  Gradient time:    {d:.3} s\n", .{grad_time});

    // Step 4: Compare with PySCF reference
    // PySCF B3LYP/6-31G(2df,p) analytical gradient (Hartree/Bohr)
    // grid_response=False (DFT-Zig does not implement Becke weight derivatives)
    // atom_grid=(50,302), prune=None, cart=True
    // Geometry: same exact Bohr coordinates
    const pyscf_grad = [3]Vec3{
        .{ .x = 0.000000000000001, .y = 0.000000000000002, .z = -0.005122736516219 }, // O
        .{ .x = -0.000000000000000, .y = -0.003002252302951, .z = 0.002564563633082 }, // H
        .{ .x = 0.000000000000000, .y = 0.003002252302956, .z = 0.002564563633082 }, // H
    };

    std.debug.print("\n  Analytical gradient comparison (Hartree/Bohr):\n", .{});
    std.debug.print("  {s:>6} {s:>6} {s:>18} {s:>18} {s:>14}\n", .{ "Atom", "Coord", "DFT-Zig", "PySCF", "Diff" });
    std.debug.print("  {s:->74}\n", .{""});

    const atom_names = [_][]const u8{ "O", "H1", "H2" };
    const coord_names = [_][]const u8{ "x", "y", "z" };
    var max_err: f64 = 0.0;

    for (0..n_atoms) |i| {
        const zig_comps = [3]f64{ grad_result.gradients[i].x, grad_result.gradients[i].y, grad_result.gradients[i].z };
        const ref_comps = [3]f64{ pyscf_grad[i].x, pyscf_grad[i].y, pyscf_grad[i].z };
        for (0..3) |c| {
            const diff = @abs(zig_comps[c] - ref_comps[c]);
            if (diff > max_err) max_err = diff;
            std.debug.print("  {s:>6} {s:>6} {d:18.12} {d:18.12} {e:14.6}\n", .{ atom_names[i], coord_names[c], zig_comps[c], ref_comps[c], diff });
        }
    }

    std.debug.print("  {s:->74}\n", .{""});
    std.debug.print("  Max absolute error: {e:.6} Ha/Bohr\n", .{max_err});

    // Also compute finite-difference gradient for O z-component as sanity check
    std.debug.print("\n  [Step 4] Finite-difference validation for O z-component...\n", .{});
    const h: f64 = 1e-4; // Bohr
    // +h
    var pos_plus = nuc_positions;
    pos_plus[0].z += h;
    const shells_plus = buildShellsForGeometry(alloc, &pos_plus, &nuc_charges);
    var result_plus = try kohn_sham.runKohnShamScf(alloc, shells_plus, &pos_plus, &nuc_charges, n_electrons, .{
        .xc_functional = .b3lyp,
        .n_radial = 50,
        .n_angular = 302,
        .prune = false,
        .use_direct_scf = false,
        .schwarz_threshold = 1e-12,
        .use_diis = true,
        .verbose = false,
    });
    defer result_plus.deinit(alloc);

    // -h
    var pos_minus = nuc_positions;
    pos_minus[0].z -= h;
    const shells_minus = buildShellsForGeometry(alloc, &pos_minus, &nuc_charges);
    var result_minus = try kohn_sham.runKohnShamScf(alloc, shells_minus, &pos_minus, &nuc_charges, n_electrons, .{
        .xc_functional = .b3lyp,
        .n_radial = 50,
        .n_angular = 302,
        .prune = false,
        .use_direct_scf = false,
        .schwarz_threshold = 1e-12,
        .use_diis = true,
        .verbose = false,
    });
    defer result_minus.deinit(alloc);

    const fd_grad_oz = (result_plus.total_energy - result_minus.total_energy) / (2.0 * h);
    const analytical_oz = grad_result.gradients[0].z;
    const fd_diff = @abs(analytical_oz - fd_grad_oz);

    std.debug.print("  E(+h) = {d:.12} Ha\n", .{result_plus.total_energy});
    std.debug.print("  E(-h) = {d:.12} Ha\n", .{result_minus.total_energy});
    std.debug.print("  FD gradient O_z:  {d:.12} Ha/Bohr\n", .{fd_grad_oz});
    std.debug.print("  Analytical O_z:   {d:.12} Ha/Bohr\n", .{analytical_oz});
    std.debug.print("  |Analyt - FD|:    {e:.6} Ha/Bohr\n", .{fd_diff});

    // Summary
    std.debug.print("\n========================================================================\n", .{});
    const grad_tolerance = 1e-5; // 10 μHa/Bohr
    if (max_err < grad_tolerance) {
        std.debug.print("  PASS: Max gradient error vs PySCF = {e:.6} Ha/Bohr (< {e:.1})\n", .{ max_err, grad_tolerance });
    } else {
        std.debug.print("  FAIL: Max gradient error vs PySCF = {e:.6} Ha/Bohr (>= {e:.1})\n", .{ max_err, grad_tolerance });
    }
    const fd_tolerance = 1e-5;
    if (fd_diff < fd_tolerance) {
        std.debug.print("  PASS: |Analytical - FD| = {e:.6} Ha/Bohr (< {e:.1})\n", .{ fd_diff, fd_tolerance });
    } else {
        std.debug.print("  FAIL: |Analytical - FD| = {e:.6} Ha/Bohr (>= {e:.1})\n", .{ fd_diff, fd_tolerance });
    }
    std.debug.print("========================================================================\n\n", .{});

    if (max_err >= grad_tolerance or fd_diff >= fd_tolerance) {
        std.process.exit(1);
    }
}

/// Helper to rebuild shells for a displaced geometry.
fn buildShellsForGeometry(
    _: std.mem.Allocator,
    nuc_positions: []const Vec3,
    nuc_charges: []const f64,
) []const ContractedShell {
    // We need static storage since we can't return alloc'd shells easily
    const S = struct {
        var shells_buf: [b631g2dfp.MAX_SHELLS_PER_ATOM * 3]ContractedShell = undefined;
        var shell_count: usize = 0;
    };

    S.shell_count = 0;
    for (nuc_positions, nuc_charges) |pos, charge| {
        const z = @as(u32, @intFromFloat(charge));
        const data = b631g2dfp.buildAtomShells(z, pos).?;
        for (data.shells[0..data.count]) |s| {
            S.shells_buf[S.shell_count] = s;
            S.shell_count += 1;
        }
    }
    return S.shells_buf[0..S.shell_count];
}
