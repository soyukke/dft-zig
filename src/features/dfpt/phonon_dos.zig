const std = @import("std");
const math = @import("../math/math.zig");
const ifc_mod = @import("ifc.zig");
const dynmat_mod = @import("dynmat.zig");

/// Phonon DOS result.
pub const PhononDosResult = struct {
    frequencies: []f64, // cm⁻¹
    dos: []f64, // states per cm⁻¹

    pub fn deinit(self: *PhononDosResult, alloc: std.mem.Allocator) void {
        if (self.frequencies.len > 0) alloc.free(self.frequencies);
        if (self.dos.len > 0) alloc.free(self.dos);
    }
};

/// Compare function for lowerBound/upperBound: context is the target value.
fn compare_f64(context: f64, item: f64) std.math.Order {
    return std.math.order(context, item);
}

/// Gaussian delta: (1/(σ√(2π))) × exp(-x²/(2σ²))
inline fn gaussian_delta(x: f64, inv_sigma_sqrt2pi: f64, inv_2sigma2: f64) f64 {
    return inv_sigma_sqrt2pi * @exp(-x * x * inv_2sigma2);
}

/// Compute phonon DOS from IFC by interpolating on a dense q-mesh.
/// Uses Gaussian broadening: g(ω) = (1/N_q) Σ_{n,q} δ_σ(ω - ω_n(q))
pub fn compute_phonon_dos(
    alloc: std.mem.Allocator,
    ifc_data: *const ifc_mod.IFC,
    masses: []const f64,
    n_atoms: usize,
    dos_qmesh: [3]usize,
    sigma_cm1: f64,
    nbin: usize,
) !PhononDosResult {
    const dim = 3 * n_atoms;
    const nq1 = dos_qmesh[0];
    const nq2 = dos_qmesh[1];
    const nq3 = dos_qmesh[2];
    const n_q_total = nq1 * nq2 * nq3;

    // Collect all frequencies from the dense q-mesh
    const all_freqs = try alloc.alloc(f64, n_q_total * dim);
    defer alloc.free(all_freqs);

    var freq_idx: usize = 0;

    var f_min: f64 = std.math.inf(f64);
    var f_max: f64 = -std.math.inf(f64);

    for (0..nq1) |iq1| {
        for (0..nq2) |iq2| {
            for (0..nq3) |iq3| {
                const q_frac = math.Vec3{
                    .x = @as(f64, @floatFromInt(iq1)) / @as(f64, @floatFromInt(nq1)),
                    .y = @as(f64, @floatFromInt(iq2)) / @as(f64, @floatFromInt(nq2)),
                    .z = @as(f64, @floatFromInt(iq3)) / @as(f64, @floatFromInt(nq3)),
                };

                // Interpolate dynamical matrix at this q-point
                const dyn_q = try ifc_mod.interpolate(alloc, ifc_data, q_frac);
                defer alloc.free(dyn_q);

                // Mass-weight (in-place)
                dynmat_mod.mass_weight_complex(dyn_q, n_atoms, masses);

                // Eigenvalues only (no eigenvectors needed for DOS)
                const freq_cm1 = try dynmat_mod.eigenvalues_complex(alloc, dyn_q, dim);
                defer alloc.free(freq_cm1);

                for (0..dim) |m| {
                    all_freqs[freq_idx] = freq_cm1[m];
                    if (freq_cm1[m] < f_min) f_min = freq_cm1[m];
                    if (freq_cm1[m] > f_max) f_max = freq_cm1[m];
                    freq_idx += 1;
                }
            }
        }
    }

    // Set up frequency bins
    const margin = 5.0 * sigma_cm1;
    const bin_min = @min(f_min - margin, -margin);
    const bin_max = f_max + margin;
    const df = (bin_max - bin_min) / @as(f64, @floatFromInt(nbin - 1));

    const frequencies = try alloc.alloc(f64, nbin);
    errdefer alloc.free(frequencies);
    const dos = try alloc.alloc(f64, nbin);
    errdefer alloc.free(dos);

    const inv_sigma_sqrt2pi = 1.0 / (sigma_cm1 * std.math.sqrt(2.0 * std.math.pi));
    const inv_2sigma2 = 1.0 / (2.0 * sigma_cm1 * sigma_cm1);
    const inv_nq = 1.0 / @as(f64, @floatFromInt(n_q_total));
    const cutoff = 5.0 * sigma_cm1;

    // Sort frequencies for efficient cutoff evaluation via binary search
    std.mem.sort(f64, all_freqs[0..freq_idx], {}, std.sort.asc(f64));
    const sorted_freqs = all_freqs[0..freq_idx];

    for (0..nbin) |i| {
        const f = bin_min + @as(f64, @floatFromInt(i)) * df;
        frequencies[i] = f;

        // Binary search for [f - cutoff, f + cutoff] range
        const lo = std.sort.lowerBound(f64, sorted_freqs, f - cutoff, compare_f64);
        const hi = std.sort.upperBound(f64, sorted_freqs, f + cutoff, compare_f64);

        var d: f64 = 0.0;
        for (sorted_freqs[lo..hi]) |af| {
            d += gaussian_delta(f - af, inv_sigma_sqrt2pi, inv_2sigma2);
        }
        dos[i] = d * inv_nq;
    }

    return .{
        .frequencies = frequencies,
        .dos = dos,
    };
}

/// Write phonon DOS to CSV file.
pub fn write_phonon_dos_csv(io: std.Io, dir: std.Io.Dir, result: PhononDosResult) !void {
    const file = try dir.createFile(io, "phonon_dos.csv", .{});
    defer file.close(io);

    var buf: [256]u8 = undefined;
    var writer = file.writer(io, &buf);
    const out = &writer.interface;

    try out.print("frequency_cm-1,dos\n", .{});
    for (0..result.frequencies.len) |i| {
        try out.print("{d:.4},{e:.6}\n", .{ result.frequencies[i], result.dos[i] });
    }
    try out.flush();
}
