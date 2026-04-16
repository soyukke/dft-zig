const std = @import("std");
const dft_zig = @import("dft_zig");
const basis_mod = dft_zig.basis;
const rys_eri = dft_zig.integrals_mod.rys_eri;
const obara_saika = dft_zig.integrals_mod.obara_saika;

const ContractedShell = basis_mod.ContractedShell;
const PrimitiveGaussian = basis_mod.PrimitiveGaussian;

fn checkERI(
    label: []const u8,
    shell_a: ContractedShell,
    shell_b: ContractedShell,
    shell_c: ContractedShell,
    shell_d: ContractedShell,
    tol: f64,
) !void {
    const na = basis_mod.numCartesian(@as(u32, @intCast(shell_a.l)));
    const nb = basis_mod.numCartesian(@as(u32, @intCast(shell_b.l)));
    const nc = basis_mod.numCartesian(@as(u32, @intCast(shell_c.l)));
    const nd = basis_mod.numCartesian(@as(u32, @intCast(shell_d.l)));
    const total = na * nb * nc * nd;

    var output_rys: [10000]f64 = undefined;
    var output_os: [10000]f64 = undefined;

    _ = rys_eri.contractedShellQuartetERI(shell_a, shell_b, shell_c, shell_d, output_rys[0..total]);
    _ = obara_saika.contractedShellQuartetERI(shell_a, shell_b, shell_c, shell_d, output_os[0..total]);

    var max_err: f64 = 0.0;
    var max_val: f64 = 0.0;
    var n_mismatch: usize = 0;

    for (0..total) |i| {
        const err = @abs(output_rys[i] - output_os[i]);
        if (err > max_err) max_err = err;
        if (@abs(output_os[i]) > max_val) max_val = @abs(output_os[i]);
        if (err > tol) n_mismatch += 1;
    }

    const status = if (n_mismatch == 0) "PASS" else "FAIL";
    std.debug.print("{s}: {s}  total={d}  max_err={e:12.4}  max_val={e:12.4}  mismatches={d}\n", .{
        status, label, total, max_err, max_val, n_mismatch,
    });
    if (n_mismatch > 0) {
        std.debug.print("  FAIL: {d} ERIs differ by more than tolerance {e:8.1}\n", .{ n_mismatch, tol });
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    std.debug.print("=== Rys Quadrature ERI Validation ===\n", .{});

    // (ss|ss) - single primitive
    {
        const shell_a = ContractedShell{
            .l = 0,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 3.42525, .coeff = 1.0 },
            },
        };
        const shell_b = ContractedShell{
            .l = 0,
            .center = .{ .x = 0.0, .y = 0.0, .z = 1.4 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 3.42525, .coeff = 1.0 },
            },
        };
        try checkERI("(ss|ss) single prim", shell_a, shell_b, shell_a, shell_b, 1e-10);
    }

    // (sp|sp) - two primitives
    {
        const shell_s = ContractedShell{
            .l = 0,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 5.0, .coeff = 0.8 },
                .{ .alpha = 1.5, .coeff = 0.3 },
            },
        };
        const shell_p = ContractedShell{
            .l = 1,
            .center = .{ .x = 0.0, .y = 0.0, .z = 1.2 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 3.0, .coeff = 0.7 },
                .{ .alpha = 0.8, .coeff = 0.4 },
            },
        };
        try checkERI("(sp|sp) two prims", shell_s, shell_p, shell_s, shell_p, 1e-10);
    }

    // (pp|pp) - two primitives, off-axis centers
    {
        const shell_p1 = ContractedShell{
            .l = 1,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 4.0, .coeff = 0.6 },
                .{ .alpha = 1.0, .coeff = 0.4 },
            },
        };
        const shell_p2 = ContractedShell{
            .l = 1,
            .center = .{ .x = 0.5, .y = 0.3, .z = 1.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 3.5, .coeff = 0.5 },
                .{ .alpha = 0.9, .coeff = 0.5 },
            },
        };
        try checkERI("(pp|pp) off-axis", shell_p1, shell_p2, shell_p1, shell_p2, 1e-10);
    }

    // (dd|dd) - single primitive
    {
        const shell_d1 = ContractedShell{
            .l = 2,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 2.5, .coeff = 1.0 },
            },
        };
        const shell_d2 = ContractedShell{
            .l = 2,
            .center = .{ .x = 0.3, .y = -0.2, .z = 0.8 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 1.8, .coeff = 1.0 },
            },
        };
        try checkERI("(dd|dd) single prim", shell_d1, shell_d2, shell_d1, shell_d2, 1e-9);
    }

    // (sd|ps) - mixed angular momentum
    {
        const shell_s = ContractedShell{
            .l = 0,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 5.0, .coeff = 1.0 },
            },
        };
        const shell_p = ContractedShell{
            .l = 1,
            .center = .{ .x = 0.0, .y = 0.0, .z = 1.5 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 3.0, .coeff = 1.0 },
            },
        };
        const shell_d = ContractedShell{
            .l = 2,
            .center = .{ .x = 0.5, .y = 0.3, .z = 0.8 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 2.0, .coeff = 1.0 },
            },
        };
        try checkERI("(sd|ps) mixed AM", shell_s, shell_d, shell_p, shell_s, 1e-10);
    }

    // (ff|ff) - single primitive (high angular momentum)
    {
        const shell_f1 = ContractedShell{
            .l = 3,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 1.5, .coeff = 1.0 },
            },
        };
        const shell_f2 = ContractedShell{
            .l = 3,
            .center = .{ .x = 0.2, .y = -0.1, .z = 0.6 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 1.2, .coeff = 1.0 },
            },
        };
        try checkERI("(ff|ff) single prim", shell_f1, shell_f2, shell_f1, shell_f2, 1e-8);
    }

    // Timing comparison: (pp|pp) with multiple primitives
    {
        const shell_p_multi = ContractedShell{
            .l = 1,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 10.0, .coeff = 0.2 },
                .{ .alpha = 4.0, .coeff = 0.4 },
                .{ .alpha = 1.5, .coeff = 0.3 },
                .{ .alpha = 0.5, .coeff = 0.1 },
            },
        };
        const shell_p2_multi = ContractedShell{
            .l = 1,
            .center = .{ .x = 0.5, .y = 0.3, .z = 1.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 8.0, .coeff = 0.3 },
                .{ .alpha = 3.0, .coeff = 0.4 },
                .{ .alpha = 1.0, .coeff = 0.2 },
                .{ .alpha = 0.3, .coeff = 0.1 },
            },
        };

        var output_rys: [81]f64 = undefined;
        var output_os: [81]f64 = undefined;

        const n_repeats: usize = 10000;

        // Time Rys
        const t_start = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..n_repeats) |_| {
            _ = rys_eri.contractedShellQuartetERI(shell_p_multi, shell_p2_multi, shell_p_multi, shell_p2_multi, &output_rys);
        }
        const rys_ns: u64 = @intCast(t_start.untilNow(io).raw.nanoseconds);
        const t_start2 = std.Io.Clock.Timestamp.now(io, .awake);

        // Time OS
        for (0..n_repeats) |_| {
            _ = obara_saika.contractedShellQuartetERI(shell_p_multi, shell_p2_multi, shell_p_multi, shell_p2_multi, &output_os);
        }
        const os_ns: u64 = @intCast(t_start2.untilNow(io).raw.nanoseconds);

        const rys_us = @as(f64, @floatFromInt(rys_ns)) / 1000.0 / @as(f64, @floatFromInt(n_repeats));
        const os_us = @as(f64, @floatFromInt(os_ns)) / 1000.0 / @as(f64, @floatFromInt(n_repeats));
        std.debug.print("\n=== Timing: (pp|pp) 4-prim x {d} repeats ===\n", .{n_repeats});
        std.debug.print("  Rys:  {d:.2} us/quartet\n", .{rys_us});
        std.debug.print("  OS:   {d:.2} us/quartet\n", .{os_us});
        std.debug.print("  Speedup: {d:.1}x\n", .{os_us / rys_us});
    }

    // Timing comparison: (dd|dd) single primitive
    {
        const shell_d1 = ContractedShell{
            .l = 2,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 2.5, .coeff = 1.0 },
            },
        };
        const shell_d2 = ContractedShell{
            .l = 2,
            .center = .{ .x = 0.3, .y = -0.2, .z = 0.8 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 1.8, .coeff = 1.0 },
            },
        };
        const nd_cart = basis_mod.numCartesian(2);
        const total_dd = nd_cart * nd_cart * nd_cart * nd_cart;
        var output_rys2: [1296]f64 = undefined;
        var output_os2: [1296]f64 = undefined;

        const n_repeats2: usize = 10000;
        const t2_start = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..n_repeats2) |_| {
            _ = rys_eri.contractedShellQuartetERI(shell_d1, shell_d2, shell_d1, shell_d2, output_rys2[0..total_dd]);
        }
        const rys_ns2: u64 = @intCast(t2_start.untilNow(io).raw.nanoseconds);
        const t2_start2 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..n_repeats2) |_| {
            _ = obara_saika.contractedShellQuartetERI(shell_d1, shell_d2, shell_d1, shell_d2, output_os2[0..total_dd]);
        }
        const os_ns2: u64 = @intCast(t2_start2.untilNow(io).raw.nanoseconds);
        const rys_us2 = @as(f64, @floatFromInt(rys_ns2)) / 1000.0 / @as(f64, @floatFromInt(n_repeats2));
        const os_us2 = @as(f64, @floatFromInt(os_ns2)) / 1000.0 / @as(f64, @floatFromInt(n_repeats2));
        std.debug.print("\n=== Timing: (dd|dd) 1-prim x {d} repeats ===\n", .{n_repeats2});
        std.debug.print("  Rys:  {d:.2} us/quartet\n", .{rys_us2});
        std.debug.print("  OS:   {d:.2} us/quartet\n", .{os_us2});
        std.debug.print("  Speedup: {d:.1}x\n", .{os_us2 / rys_us2});
    }

    // Timing comparison: (ff|ff) single primitive
    {
        const shell_f1 = ContractedShell{
            .l = 3,
            .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 1.5, .coeff = 1.0 },
            },
        };
        const shell_f2 = ContractedShell{
            .l = 3,
            .center = .{ .x = 0.2, .y = -0.1, .z = 0.6 },
            .primitives = &[_]PrimitiveGaussian{
                .{ .alpha = 1.2, .coeff = 1.0 },
            },
        };
        const nf = basis_mod.numCartesian(3);
        const total_ff = nf * nf * nf * nf;
        var output_rys3: [10000]f64 = undefined;
        var output_os3: [10000]f64 = undefined;

        const n_repeats3: usize = 100;
        const t3_start = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..n_repeats3) |_| {
            _ = rys_eri.contractedShellQuartetERI(shell_f1, shell_f2, shell_f1, shell_f2, output_rys3[0..total_ff]);
        }
        const rys_ns3: u64 = @intCast(t3_start.untilNow(io).raw.nanoseconds);
        const t3_start2 = std.Io.Clock.Timestamp.now(io, .awake);
        // Note: OS (ff|ff) hits the per-integral fallback and is ~100x slower.
        // Only time 1 repeat for OS to avoid timeout.
        _ = obara_saika.contractedShellQuartetERI(shell_f1, shell_f2, shell_f1, shell_f2, output_os3[0..total_ff]);
        const os_ns3: u64 = @intCast(t3_start2.untilNow(io).raw.nanoseconds);
        const rys_us3 = @as(f64, @floatFromInt(rys_ns3)) / 1000.0 / @as(f64, @floatFromInt(n_repeats3));
        const os_us3 = @as(f64, @floatFromInt(os_ns3)) / 1000.0; // only 1 repeat
        std.debug.print("\n=== Timing: (ff|ff) 1-prim x {d} repeats ===\n", .{n_repeats3});
        std.debug.print("  Rys:  {d:.2} us/quartet\n", .{rys_us3});
        std.debug.print("  OS:   {d:.2} us/quartet\n", .{os_us3});
        std.debug.print("  Speedup: {d:.1}x\n", .{os_us3 / rys_us3});
    }

    std.debug.print("\nDone.\n", .{});
}
