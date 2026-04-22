const std = @import("std");

const compare = @import("compare.zig");
const scf = @import("../scf/scf.zig");

pub const EnergyComparison = struct {
    total: compare.ScalarComparison,
    band: compare.ScalarComparison,
    hartree: compare.ScalarComparison,
    vxc_rho: compare.ScalarComparison,
    xc: compare.ScalarComparison,
    ion_ion: compare.ScalarComparison,
    psp_core: compare.ScalarComparison,
    double_counting: compare.ScalarComparison,
    local_pseudo: compare.ScalarComparison,
    nonlocal_pseudo: compare.ScalarComparison,
};

pub const EnergyTermsTolerance = struct {
    total: compare.ComparisonTolerance,
    band: compare.ComparisonTolerance,
    hartree: compare.ComparisonTolerance,
    vxc_rho: compare.ComparisonTolerance,
    xc: compare.ComparisonTolerance,
    ion_ion: compare.ComparisonTolerance,
    psp_core: compare.ComparisonTolerance,
    double_counting: compare.ComparisonTolerance,
    local_pseudo: compare.ComparisonTolerance,
    nonlocal_pseudo: compare.ComparisonTolerance,
};

pub const EnergyTermsToleranceResult = struct {
    total: bool,
    band: bool,
    hartree: bool,
    vxc_rho: bool,
    xc: bool,
    ion_ion: bool,
    psp_core: bool,
    double_counting: bool,
    local_pseudo: bool,
    nonlocal_pseudo: bool,
    all: bool,
};

pub fn compareEnergyTerms(reference: scf.EnergyTerms, candidate: scf.EnergyTerms) EnergyComparison {
    return .{
        .total = compare.compareScalar(reference.total, candidate.total),
        .band = compare.compareScalar(reference.band, candidate.band),
        .hartree = compare.compareScalar(reference.hartree, candidate.hartree),
        .vxc_rho = compare.compareScalar(reference.vxc_rho, candidate.vxc_rho),
        .xc = compare.compareScalar(reference.xc, candidate.xc),
        .ion_ion = compare.compareScalar(reference.ion_ion, candidate.ion_ion),
        .psp_core = compare.compareScalar(reference.psp_core, candidate.psp_core),
        .double_counting = compare.compareScalar(
            reference.double_counting,
            candidate.double_counting,
        ),
        .local_pseudo = compare.compareScalar(reference.local_pseudo, candidate.local_pseudo),
        .nonlocal_pseudo = compare.compareScalar(
            reference.nonlocal_pseudo,
            candidate.nonlocal_pseudo,
        ),
    };
}

pub fn withinEnergyTolerance(
    report: EnergyComparison,
    tol: EnergyTermsTolerance,
) EnergyTermsToleranceResult {
    const total = compare.withinScalarTolerance(report.total, tol.total);
    const band = compare.withinScalarTolerance(report.band, tol.band);
    const hartree = compare.withinScalarTolerance(report.hartree, tol.hartree);
    const vxc_rho = compare.withinScalarTolerance(report.vxc_rho, tol.vxc_rho);
    const xc = compare.withinScalarTolerance(report.xc, tol.xc);
    const ion_ion = compare.withinScalarTolerance(report.ion_ion, tol.ion_ion);
    const psp_core = compare.withinScalarTolerance(report.psp_core, tol.psp_core);
    const double_counting = compare.withinScalarTolerance(
        report.double_counting,
        tol.double_counting,
    );
    const local_pseudo = compare.withinScalarTolerance(report.local_pseudo, tol.local_pseudo);
    const nonlocal_pseudo = compare.withinScalarTolerance(
        report.nonlocal_pseudo,
        tol.nonlocal_pseudo,
    );
    const all = total and band and hartree and vxc_rho and xc and
        ion_ion and psp_core and double_counting and local_pseudo and nonlocal_pseudo;
    return .{
        .total = total,
        .band = band,
        .hartree = hartree,
        .vxc_rho = vxc_rho,
        .xc = xc,
        .ion_ion = ion_ion,
        .psp_core = psp_core,
        .double_counting = double_counting,
        .local_pseudo = local_pseudo,
        .nonlocal_pseudo = nonlocal_pseudo,
        .all = all,
    };
}

test "compareEnergyTerms uses scalar comparisons" {
    const ref = scf.EnergyTerms{
        .total = 1.0,
        .band = 2.0,
        .hartree = 3.0,
        .vxc_rho = 4.0,
        .xc = 5.0,
        .ion_ion = 6.0,
        .psp_core = 7.0,
        .double_counting = 8.0,
        .local_pseudo = 9.0,
        .nonlocal_pseudo = 10.0,
    };
    const cand = scf.EnergyTerms{
        .total = 1.1,
        .band = 2.0,
        .hartree = 2.5,
        .vxc_rho = 4.0,
        .xc = 4.9,
        .ion_ion = 6.0,
        .psp_core = 7.0,
        .double_counting = 8.0,
        .local_pseudo = 9.0,
        .nonlocal_pseudo = 10.0,
    };
    const report = compareEnergyTerms(ref, cand);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), report.total.abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), report.hartree.abs, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), report.xc.abs, 1e-12);
}

test "withinEnergyTolerance aggregates per-term checks" {
    const report = EnergyComparison{
        .total = .{ .abs = 0.1, .rel = 0.01 },
        .band = .{ .abs = 0.0, .rel = 0.0 },
        .hartree = .{ .abs = 0.2, .rel = 0.02 },
        .vxc_rho = .{ .abs = 0.0, .rel = 0.0 },
        .xc = .{ .abs = 0.0, .rel = 0.0 },
        .ion_ion = .{ .abs = 0.0, .rel = 0.0 },
        .psp_core = .{ .abs = 0.0, .rel = 0.0 },
        .double_counting = .{ .abs = 0.0, .rel = 0.0 },
        .local_pseudo = .{ .abs = 0.0, .rel = 0.0 },
        .nonlocal_pseudo = .{ .abs = 0.0, .rel = 0.0 },
    };
    const tol = EnergyTermsTolerance{
        .total = .{ .abs = 0.05, .rel = 0.1 },
        .band = .{ .abs = 0.01, .rel = 0.1 },
        .hartree = .{ .abs = 0.1, .rel = 0.1 },
        .vxc_rho = .{ .abs = 0.01, .rel = 0.1 },
        .xc = .{ .abs = 0.01, .rel = 0.1 },
        .ion_ion = .{ .abs = 0.01, .rel = 0.1 },
        .psp_core = .{ .abs = 0.01, .rel = 0.1 },
        .double_counting = .{ .abs = 0.01, .rel = 0.1 },
        .local_pseudo = .{ .abs = 0.01, .rel = 0.1 },
        .nonlocal_pseudo = .{ .abs = 0.01, .rel = 0.1 },
    };
    const result = withinEnergyTolerance(report, tol);
    try std.testing.expect(!result.total);
    try std.testing.expect(result.band);
    try std.testing.expect(!result.all);
}
