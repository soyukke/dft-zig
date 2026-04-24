const std = @import("std");

const compare = @import("compare.zig");
const energy_compare = @import("energy_compare.zig");
const scf_harness = @import("scf_harness.zig");

pub const ScfTolerance = struct {
    energy: compare.ComparisonTolerance,
    density: compare.ComparisonTolerance,
    energy_terms: energy_compare.EnergyTermsTolerance,
};

pub const ScfToleranceResult = struct {
    energy: bool,
    density: bool,
    energy_terms: energy_compare.EnergyTermsToleranceResult,
    all: bool,
};

pub fn within_scf_tolerance(
    report: scf_harness.ScfComparisonReport,
    tol: ScfTolerance,
) ScfToleranceResult {
    const energy_ok = compare.within_scalar_tolerance(report.total.energy, tol.energy);
    const density_ok = compare.within_tolerance(report.total.density, tol.density);
    const terms_ok = energy_compare.within_energy_tolerance(report.energy_terms, tol.energy_terms);
    const all = energy_ok and density_ok and terms_ok.all;
    return .{
        .energy = energy_ok,
        .density = density_ok,
        .energy_terms = terms_ok,
        .all = all,
    };
}

test "within_scf_tolerance aggregates results" {
    const report = scf_harness.ScfComparisonReport{
        .total = .{
            .energy = .{ .abs = 0.01, .rel = 0.001 },
            .density = .{
                .max_abs = 0.05,
                .mean_abs = 0.02,
                .rms = 0.03,
                .rel_max = 0.01,
                .rel_rms = 0.02,
            },
        },
        .energy_terms = .{
            .total = .{ .abs = 0.01, .rel = 0.001 },
            .band = .{ .abs = 0.01, .rel = 0.001 },
            .hartree = .{ .abs = 0.01, .rel = 0.001 },
            .vxc_rho = .{ .abs = 0.01, .rel = 0.001 },
            .xc = .{ .abs = 0.01, .rel = 0.001 },
            .ion_ion = .{ .abs = 0.01, .rel = 0.001 },
            .psp_core = .{ .abs = 0.01, .rel = 0.001 },
            .double_counting = .{ .abs = 0.01, .rel = 0.001 },
            .local_pseudo = .{ .abs = 0.01, .rel = 0.001 },
            .nonlocal_pseudo = .{ .abs = 0.01, .rel = 0.001 },
        },
    };
    const tol = ScfTolerance{
        .energy = .{ .abs = 0.1, .rel = 0.1 },
        .density = .{ .abs = 0.1, .rel = 0.1 },
        .energy_terms = .{
            .total = .{ .abs = 0.1, .rel = 0.1 },
            .band = .{ .abs = 0.1, .rel = 0.1 },
            .hartree = .{ .abs = 0.1, .rel = 0.1 },
            .vxc_rho = .{ .abs = 0.1, .rel = 0.1 },
            .xc = .{ .abs = 0.1, .rel = 0.1 },
            .ion_ion = .{ .abs = 0.1, .rel = 0.1 },
            .psp_core = .{ .abs = 0.1, .rel = 0.1 },
            .double_counting = .{ .abs = 0.1, .rel = 0.1 },
            .local_pseudo = .{ .abs = 0.1, .rel = 0.1 },
            .nonlocal_pseudo = .{ .abs = 0.1, .rel = 0.1 },
        },
    };
    const result = within_scf_tolerance(report, tol);
    try std.testing.expect(result.all);
}

test "within_scf_tolerance detects failed terms" {
    const report = scf_harness.ScfComparisonReport{
        .total = .{
            .energy = .{ .abs = 0.2, .rel = 0.2 },
            .density = .{
                .max_abs = 0.2,
                .mean_abs = 0.2,
                .rms = 0.2,
                .rel_max = 0.2,
                .rel_rms = 0.2,
            },
        },
        .energy_terms = .{
            .total = .{ .abs = 0.2, .rel = 0.2 },
            .band = .{ .abs = 0.2, .rel = 0.2 },
            .hartree = .{ .abs = 0.2, .rel = 0.2 },
            .vxc_rho = .{ .abs = 0.2, .rel = 0.2 },
            .xc = .{ .abs = 0.2, .rel = 0.2 },
            .ion_ion = .{ .abs = 0.2, .rel = 0.2 },
            .psp_core = .{ .abs = 0.2, .rel = 0.2 },
            .double_counting = .{ .abs = 0.2, .rel = 0.2 },
            .local_pseudo = .{ .abs = 0.2, .rel = 0.2 },
            .nonlocal_pseudo = .{ .abs = 0.2, .rel = 0.2 },
        },
    };
    const tol = ScfTolerance{
        .energy = .{ .abs = 0.1, .rel = 0.1 },
        .density = .{ .abs = 0.1, .rel = 0.1 },
        .energy_terms = .{
            .total = .{ .abs = 0.1, .rel = 0.1 },
            .band = .{ .abs = 0.1, .rel = 0.1 },
            .hartree = .{ .abs = 0.1, .rel = 0.1 },
            .vxc_rho = .{ .abs = 0.1, .rel = 0.1 },
            .xc = .{ .abs = 0.1, .rel = 0.1 },
            .ion_ion = .{ .abs = 0.1, .rel = 0.1 },
            .psp_core = .{ .abs = 0.1, .rel = 0.1 },
            .double_counting = .{ .abs = 0.1, .rel = 0.1 },
            .local_pseudo = .{ .abs = 0.1, .rel = 0.1 },
            .nonlocal_pseudo = .{ .abs = 0.1, .rel = 0.1 },
        },
    };
    const result = within_scf_tolerance(report, tol);
    try std.testing.expect(!result.all);
    try std.testing.expect(!result.energy);
    try std.testing.expect(!result.density);
}
