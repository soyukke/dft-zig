const std = @import("std");
const math = @import("../math/math.zig");
const scf = @import("../scf/scf.zig");

/// Gaussian delta: (1/(σ√(2π))) × exp(-x²/(2σ²))
pub inline fn gaussianDelta(x: f64, inv_sigma_sqrt2pi: f64, inv_2sigma2: f64) f64 {
    return inv_sigma_sqrt2pi * @exp(-x * x * inv_2sigma2);
}

/// DOS computation result.
pub const DosResult = struct {
    energies: []f64,
    dos: []f64,

    pub fn deinit(self: *DosResult, alloc: std.mem.Allocator) void {
        if (self.energies.len > 0) alloc.free(self.energies);
        if (self.dos.len > 0) alloc.free(self.dos);
    }
};

/// Compute electronic DOS from SCF eigenvalues using Gaussian broadening.
/// DOS(E) = Σ_{n,k} w_k × spin_factor × (1/(σ√(2π))) × exp(-(E-ε_{n,k})²/(2σ²))
pub fn computeDos(
    alloc: std.mem.Allocator,
    wf_data: scf.WavefunctionData,
    sigma: f64,
    npoints: usize,
    emin_opt: ?f64,
    emax_opt: ?f64,
    nspin: usize,
) !DosResult {
    // Find energy range from eigenvalues
    var e_min: f64 = std.math.inf(f64);
    var e_max: f64 = -std.math.inf(f64);
    for (wf_data.kpoints) |kp| {
        for (kp.eigenvalues) |e| {
            if (e < e_min) e_min = e;
            if (e > e_max) e_max = e;
        }
    }

    // Extend range by 5σ or use user-specified range
    const emin = emin_opt orelse (e_min - 5.0 * sigma);
    const emax = emax_opt orelse (e_max + 5.0 * sigma);
    const de = (emax - emin) / @as(f64, @floatFromInt(npoints - 1));

    const energies = try alloc.alloc(f64, npoints);
    errdefer alloc.free(energies);
    const dos = try alloc.alloc(f64, npoints);
    errdefer alloc.free(dos);

    const inv_sigma_sqrt2pi = 1.0 / (sigma * std.math.sqrt(2.0 * std.math.pi));
    const inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    const spin_factor: f64 = if (nspin == 2) 1.0 else 2.0;
    const cutoff = 5.0 * sigma;

    for (0..npoints) |i| {
        const e = emin + @as(f64, @floatFromInt(i)) * de;
        energies[i] = e;

        var d: f64 = 0.0;
        for (wf_data.kpoints) |kp| {
            for (kp.eigenvalues) |ek| {
                const diff = e - ek;
                if (@abs(diff) <= cutoff) {
                    d += kp.weight * spin_factor * gaussianDelta(diff, inv_sigma_sqrt2pi, inv_2sigma2);
                }
            }
        }
        dos[i] = d;
    }

    return .{
        .energies = energies,
        .dos = dos,
    };
}

/// Write DOS to CSV file.
/// If fermi_level is NaN (e.g., insulator without smearing), shifted energies use 0.
pub fn writeDosCSV(io: std.Io, dir: std.Io.Dir, result: DosResult, fermi_level: f64) !void {
    return writeDosCSVNamed(io, dir, result, fermi_level, "dos.csv");
}

pub fn writeDosCSVNamed(io: std.Io, dir: std.Io.Dir, result: DosResult, fermi_level: f64, filename: []const u8) !void {
    const file = try dir.createFile(io, filename, .{});
    defer file.close(io);
    var buf: [256]u8 = undefined;
    var writer = file.writer(io, &buf);
    const out = &writer.interface;

    const ef = if (std.math.isNan(fermi_level)) @as(f64, 0.0) else fermi_level;
    try out.print("energy_ry,dos_states_per_ry,energy_shifted_ry\n", .{});
    for (0..result.energies.len) |i| {
        try out.print("{d:.8},{d:.8},{d:.8}\n", .{
            result.energies[i],
            result.dos[i],
            result.energies[i] - ef,
        });
    }
    try out.flush();
}

test "computeDos basic" {
    const alloc = std.testing.allocator;

    var eigenvalues1 = [_]f64{ 0.0, 0.5, 1.0 };
    var occupations1 = [_]f64{ 2.0, 2.0, 0.0 };
    var eigenvalues2 = [_]f64{ 0.1, 0.6, 1.1 };
    var occupations2 = [_]f64{ 2.0, 2.0, 0.0 };

    var kpoints = [_]scf.KpointWavefunction{
        .{
            .k_frac = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .k_cart = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .weight = 0.5,
            .basis_len = 10,
            .nbands = 3,
            .eigenvalues = &eigenvalues1,
            .coefficients = &.{},
            .occupations = &occupations1,
        },
        .{
            .k_frac = .{ .x = 0.5, .y = 0.0, .z = 0.0 },
            .k_cart = .{ .x = 0.5, .y = 0.0, .z = 0.0 },
            .weight = 0.5,
            .basis_len = 10,
            .nbands = 3,
            .eigenvalues = &eigenvalues2,
            .coefficients = &.{},
            .occupations = &occupations2,
        },
    };

    const wf_data = scf.WavefunctionData{
        .kpoints = &kpoints,
        .ecut_ry = 30.0,
        .fermi_level = 0.75,
    };

    var result = try computeDos(alloc, wf_data, 0.05, 101, null, null, 1);
    defer result.deinit(alloc);

    try std.testing.expect(result.energies.len == 101);
    for (result.dos) |d| {
        try std.testing.expect(d >= 0.0);
    }
    var integral: f64 = 0.0;
    const de = result.energies[1] - result.energies[0];
    for (result.dos) |d| {
        integral += d * de;
    }
    try std.testing.expectApproxEqAbs(integral, 6.0, 0.5);
}
