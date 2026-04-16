//! Density Fitting (RI-J/K) performance benchmark.
//!
//! Compares conventional (direct SCF) vs density fitting for QM9 molecules
//! with B3LYP/6-31G(2df,p).
//!
//! Usage: zig build run-df-bench -Doptimize=ReleaseFast

const std = @import("std");
const dft_zig = @import("dft_zig");
const math_mod = dft_zig.math;
const basis_mod = dft_zig.basis;
const gto_scf = dft_zig.gto_scf;
const kohn_sham = gto_scf.kohn_sham;
const molecule_mod = gto_scf.molecule;
const obara_saika = dft_zig.integrals_mod.obara_saika;

const KsParams = kohn_sham.KsParams;

const MolRef = struct {
    name: []const u8,
    xyz_file: []const u8,
    n_electrons: usize,
    n_basis: usize,
};

/// Molecules sorted by size (small → large)
const molecules = [_]MolRef{
    .{ .name = "H2", .xyz_file = "benchmarks/qm9_pipeline/xyz/H2.xyz", .n_electrons = 2, .n_basis = 10 },
    .{ .name = "H2O", .xyz_file = "benchmarks/qm9_pipeline/xyz/H2O.xyz", .n_electrons = 10, .n_basis = 41 },
    .{ .name = "CH4", .xyz_file = "benchmarks/qm9_pipeline/xyz/CH4.xyz", .n_electrons = 10, .n_basis = 51 },
    .{ .name = "N2", .xyz_file = "benchmarks/qm9_pipeline/xyz/N2.xyz", .n_electrons = 14, .n_basis = 62 },
    .{ .name = "C2H2", .xyz_file = "benchmarks/qm9_pipeline/xyz/C2H2.xyz", .n_electrons = 14, .n_basis = 72 },
    .{ .name = "CH2O", .xyz_file = "benchmarks/qm9_pipeline/xyz/CH2O.xyz", .n_electrons = 16, .n_basis = 72 },
    .{ .name = "C2H4", .xyz_file = "benchmarks/qm9_pipeline/xyz/C2H4.xyz", .n_electrons = 16, .n_basis = 82 },
    .{ .name = "CH3OH", .xyz_file = "benchmarks/qm9_pipeline/xyz/CH3OH.xyz", .n_electrons = 18, .n_basis = 82 },
    .{ .name = "C2H6", .xyz_file = "benchmarks/qm9_pipeline/xyz/C2H6.xyz", .n_electrons = 18, .n_basis = 92 },
};

const ks_params_conv = KsParams{
    .xc_functional = .b3lyp,
    .n_radial = 50,
    .n_angular = 302,
    .prune = false,
    .use_direct_scf = true,
    .schwarz_threshold = 1e-12,
    .use_diis = true,
    .verbose = false,
    .use_libcint = true,
    .use_density_fitting = false,
};

const ks_params_df = KsParams{
    .xc_functional = .b3lyp,
    .n_radial = 50,
    .n_angular = 302,
    .prune = false,
    .use_direct_scf = true,
    .schwarz_threshold = 1e-12,
    .use_diis = true,
    .verbose = false,
    .use_libcint = true,
    .use_density_fitting = true,
};

const RunResult = struct {
    energy: f64,
    iterations: usize,
    time_s: f64,
    converged: bool,
};

fn runScf(alloc: std.mem.Allocator, ref: MolRef, params: KsParams) !RunResult {
    var mol = try molecule_mod.loadXyzFile(alloc, ref.xyz_file, .@"6-31g_2dfp", 0);
    defer mol.deinit();

    var timer = try std.time.Timer.start();

    var result = try kohn_sham.runKohnShamScf(
        alloc,
        mol.shells,
        mol.positions,
        mol.charges,
        mol.n_electrons,
        params,
    );
    defer result.deinit(alloc);

    const elapsed = @as(f64, @floatFromInt(timer.read())) / 1e9;

    return .{
        .energy = result.total_energy,
        .iterations = result.iterations,
        .time_s = elapsed,
        .converged = result.converged,
    };
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("========================================================================\n", .{});
    std.debug.print("  Density Fitting Benchmark: Conventional vs RI-J/K\n", .{});
    std.debug.print("  B3LYP/6-31G(2df,p), aux: def2-universal-JKFIT\n", .{});
    std.debug.print("  Grid: 50 radial, 302 angular (Lebedev), no pruning\n", .{});
    std.debug.print("  Molecules: {d}\n", .{molecules.len});
    std.debug.print("========================================================================\n\n", .{});

    std.debug.print("  {s:<8} {s:>5} {s:>18} {s:>8} {s:>5} {s:>18} {s:>8} {s:>5} {s:>10} {s:>8}\n", .{
        "Molecule", "Nbas", "Conv E (Ha)", "Conv(s)", "Iter", "DF E (Ha)", "DF(s)", "Iter", "dE (Ha)", "Speedup",
    });
    std.debug.print("  {s:->110}\n", .{""});

    for (molecules) |ref| {
        // Conventional
        const conv = runScf(alloc, ref, ks_params_conv) catch |err| {
            std.debug.print("  {s:<8} Conv FAILED: {}\n", .{ ref.name, err });
            continue;
        };

        // Density Fitting
        const df = runScf(alloc, ref, ks_params_df) catch |err| {
            std.debug.print("  {s:<8} DF FAILED: {}\n", .{ ref.name, err });
            continue;
        };

        const diff = @abs(conv.energy - df.energy);
        const speedup = conv.time_s / df.time_s;

        std.debug.print("  {s:<8} {d:>5} {d:>18.10} {d:>8.2} {d:>5} {d:>18.10} {d:>8.2} {d:>5} {e:>10.2} {d:>7.2}x\n", .{
            ref.name,
            ref.n_basis,
            conv.energy,
            conv.time_s,
            conv.iterations,
            df.energy,
            df.time_s,
            df.iterations,
            diff,
            speedup,
        });
    }

    std.debug.print("  {s:->110}\n", .{""});
    std.debug.print("\n", .{});
}
