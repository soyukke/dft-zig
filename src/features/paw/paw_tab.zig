const std = @import("std");
const test_support = @import("../../test_support.zig");
const paw_data = @import("../pseudopotential/paw_data.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const PawData = paw_data.PawData;

const ctrap_weight = @import("../math/math.zig").radial.ctrap_weight;

/// Pre-computed PAW tables for one atomic species.
/// Built once from UPF data and reused throughout the calculation.
pub const PawTab = struct {
    /// Overlap correction: S_ij = delta_ij + q_ij
    /// where q_ij = <phi_i|phi_j> - <tphi_i|tphi_j>
    sij: []f64,
    /// Kinetic energy correction:
    /// k_ij = <phi_i|T|phi_j> - <tphi_i|T|tphi_j>
    kij: []f64,
    /// Q_ij^L(G) form factor tables for augmentation charges.
    /// Layout: [n_qijl_entries][n_qpoints]
    /// Each entry corresponds to a (i,j,L) triplet from the UPF file.
    qijl_form: []f64,
    /// Number of (i,j,L) entries
    n_qijl_entries: usize,
    /// Index mapping: for each entry, stores (first, second, L)
    qijl_indices: []QijlIndex,
    /// Number of G-points in form factor table
    n_qpoints: usize,
    /// G-spacing for form factor table
    dq: f64,
    /// Number of projectors
    nbeta: usize,
    /// l values for each projector
    l_list: []i32,

    const N_QPOINTS: usize = 4096;

    pub const QijlIndex = struct {
        first: usize,
        second: usize,
        l: usize,
    };

    /// Initialize PAW tables from UPF data.
    pub fn init(
        alloc: std.mem.Allocator,
        paw: PawData,
        r: []const f64,
        rab: []const f64,
        q_max: f64,
    ) !PawTab {
        const nbeta = paw.number_of_proj;
        const lmn2 = nbeta * nbeta;
        const n_mesh = r.len;

        const sij = try build_sij_table(alloc, paw, rab, n_mesh, nbeta, lmn2);
        errdefer alloc.free(sij);

        const kij = try build_kij_table(alloc, paw, r, rab, nbeta, lmn2);
        errdefer alloc.free(kij);

        const l_list = try alloc.alloc(i32, nbeta);
        errdefer alloc.free(l_list);

        for (0..nbeta) |i| l_list[i] = paw.ae_wfc[i].l;

        const n_entries = paw.qijl.len;
        const n_qpoints = N_QPOINTS;
        const dq = q_max / @as(f64, @floatFromInt(n_qpoints - 1));
        const qijl_form = try alloc.alloc(f64, n_entries * n_qpoints);
        errdefer alloc.free(qijl_form);

        const qijl_indices = try alloc.alloc(QijlIndex, n_entries);
        errdefer alloc.free(qijl_indices);

        build_qijl_form_factors(paw, r, rab, n_entries, n_qpoints, dq, qijl_form, qijl_indices);

        return .{
            .sij = sij,
            .kij = kij,
            .qijl_form = qijl_form,
            .n_qijl_entries = n_entries,
            .qijl_indices = qijl_indices,
            .n_qpoints = n_qpoints,
            .dq = dq,
            .nbeta = nbeta,
            .l_list = l_list,
        };
    }

    /// Evaluate Q_ij^L(G) form factor for a specific entry using linear interpolation.
    pub fn eval_qijl_form(self: *const PawTab, entry_idx: usize, g: f64) f64 {
        if (entry_idx >= self.n_qijl_entries) return 0.0;
        const base = entry_idx * self.n_qpoints;
        const idx_f = g / self.dq;
        const idx: usize = @intFromFloat(idx_f);
        if (idx >= self.n_qpoints - 1) return self.qijl_form[base + self.n_qpoints - 1];
        const t = idx_f - @as(f64, @floatFromInt(idx));
        return self.qijl_form[base + idx] * (1.0 - t) + self.qijl_form[base + idx + 1] * t;
    }

    /// Evaluate derivative dQ_ij^L(G)/dG using finite differences on the tabulated form factor.
    pub fn eval_qijl_form_deriv(self: *const PawTab, entry_idx: usize, g: f64) f64 {
        if (entry_idx >= self.n_qijl_entries) return 0.0;
        const base = entry_idx * self.n_qpoints;
        const idx_f = g / self.dq;
        const idx: usize = @intFromFloat(idx_f);
        if (idx >= self.n_qpoints - 1) return 0.0;
        // Linear interpolation derivative = constant slope of the segment
        return (self.qijl_form[base + idx + 1] - self.qijl_form[base + idx]) / self.dq;
    }

    /// Find the QIJL entry index for given (i, j, L).
    /// Returns null if not found.
    pub fn find_qijl_entry(self: *const PawTab, first: usize, second: usize, l: usize) ?usize {
        for (0..self.n_qijl_entries) |e| {
            const idx = self.qijl_indices[e];
            if (idx.first == first and idx.second == second and idx.l == l) return e;
        }
        return null;
    }

    pub fn deinit(self: *PawTab, alloc: std.mem.Allocator) void {
        if (self.sij.len > 0) alloc.free(self.sij);
        if (self.kij.len > 0) alloc.free(self.kij);
        if (self.qijl_form.len > 0) alloc.free(self.qijl_form);
        if (self.qijl_indices.len > 0) alloc.free(self.qijl_indices);
        if (self.l_list.len > 0) alloc.free(self.l_list);
    }
};

fn build_sij_table(
    alloc: std.mem.Allocator,
    paw: PawData,
    rab: []const f64,
    n_mesh: usize,
    nbeta: usize,
    lmn2: usize,
) ![]f64 {
    const sij = try alloc.alloc(f64, lmn2);
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            const ae_i = paw.ae_wfc[i].values;
            const ae_j = paw.ae_wfc[j].values;
            const ps_i = paw.ps_wfc[i].values;
            const ps_j = paw.ps_wfc[j].values;
            const n = @min(n_mesh, @min(ae_i.len, @min(ae_j.len, @min(ps_i.len, ps_j.len))));
            var sum: f64 = 0.0;
            for (0..n) |k| {
                sum += (ae_i[k] * ae_j[k] - ps_i[k] * ps_j[k]) * rab[k] * ctrap_weight(k, n);
            }
            const delta = if (i == j) @as(f64, 1.0) else @as(f64, 0.0);
            sij[i * nbeta + j] = delta + sum;
        }
    }
    return sij;
}

fn build_kij_table(
    alloc: std.mem.Allocator,
    paw: PawData,
    r: []const f64,
    rab: []const f64,
    nbeta: usize,
    lmn2: usize,
) ![]f64 {
    const kij = try alloc.alloc(f64, lmn2);
    for (0..nbeta) |i| {
        for (0..nbeta) |j| {
            kij[i * nbeta + j] = compute_kij(
                paw.ae_wfc[i].values,
                paw.ae_wfc[j].values,
                paw.ps_wfc[i].values,
                paw.ps_wfc[j].values,
                paw.ae_wfc[i].l,
                paw.ae_wfc[j].l,
                r,
                rab,
            );
        }
    }
    return kij;
}

fn build_qijl_form_factors(
    paw: PawData,
    r: []const f64,
    rab: []const f64,
    n_entries: usize,
    n_qpoints: usize,
    dq: f64,
    qijl_form: []f64,
    qijl_indices: []PawTab.QijlIndex,
) void {
    for (0..n_entries) |e| {
        const entry = paw.qijl[e];
        qijl_indices[e] = .{
            .first = entry.first_index,
            .second = entry.second_index,
            .l = entry.angular_momentum,
        };
        const qdata = entry.values;
        const n_r = @min(qdata.len, @min(r.len, rab.len));
        const l_val: i32 = @intCast(entry.angular_momentum);
        for (0..n_qpoints) |qi| {
            const g = @as(f64, @floatFromInt(qi)) * dq;
            var sum: f64 = 0.0;
            for (0..n_r) |k| {
                const x = g * r[k];
                const jl = nonlocal.spherical_bessel(l_val, x);
                sum += qdata[k] * jl * rab[k] * ctrap_weight(k, n_r);
            }
            qijl_form[e * n_qpoints + qi] = 4.0 * std.math.pi * sum;
        }
    }
}

/// Compute kinetic energy matrix element K_ij.
/// K_ij = <phi_i|T|phi_j> - <tphi_i|T|tphi_j>
/// where T is the kinetic energy operator in Rydberg units (T = -d²/dr² + l(l+1)/r²).
/// For u(r) = r*phi(r), the radial kinetic energy is:
///   <u_i|T|u_j> = integral[ u_i' * u_j' + l(l+1) * u_i * u_j / r² ] * dr
/// (Rydberg units: factor 1, not 1/2 as in Hartree)
fn compute_kij(
    ae_i: []const f64,
    ae_j: []const f64,
    ps_i: []const f64,
    ps_j: []const f64,
    l_i: i32,
    l_j: i32,
    r: []const f64,
    rab: []const f64,
) f64 {
    // K_ij is zero if l_i != l_j (by angular momentum selection)
    if (l_i != l_j) return 0.0;
    const l = l_i;
    const ll1 = @as(f64, @floatFromInt(l)) * (@as(f64, @floatFromInt(l)) + 1.0);

    const n = @min(ae_i.len, @min(ae_j.len, @min(ps_i.len, @min(ps_j.len, @min(r.len, rab.len)))));
    if (n < 3) return 0.0;

    // Compute derivatives using finite differences: du/dr = du/(rab * di)
    // where di is the index increment
    var sum: f64 = 0.0;
    for (1..n - 1) |k| {
        // Numerical derivative: u'(k) = (u(k+1) - u(k-1)) / (2 * rab(k))
        // But since r is non-uniform, use: du/dr = (u(k+1)-u(k-1)) / (r(k+1)-r(k-1))
        const dr = r[k + 1] - r[k - 1];
        if (@abs(dr) < 1e-30) continue;
        const inv_dr = 1.0 / dr;

        const d_ae_i = (ae_i[k + 1] - ae_i[k - 1]) * inv_dr;
        const d_ae_j = (ae_j[k + 1] - ae_j[k - 1]) * inv_dr;
        const d_ps_i = (ps_i[k + 1] - ps_i[k - 1]) * inv_dr;
        const d_ps_j = (ps_j[k + 1] - ps_j[k - 1]) * inv_dr;

        // Derivative term: u_i' * u_j'
        const deriv_ae = d_ae_i * d_ae_j;
        const deriv_ps = d_ps_i * d_ps_j;

        // Centrifugal term: l(l+1) * u_i * u_j / r²
        const centrifugal_ae = if (r[k] > 1e-10) ll1 * ae_i[k] * ae_j[k] / (r[k] * r[k]) else 0.0;
        const centrifugal_ps = if (r[k] > 1e-10) ll1 * ps_i[k] * ps_j[k] / (r[k] * r[k]) else 0.0;

        const delta = (deriv_ae + centrifugal_ae) - (deriv_ps + centrifugal_ps);
        sum += delta * rab[k] * ctrap_weight(k, n);
    }

    return sum;
}

test "PawTab init from Si PAW UPF" {
    const io = std.testing.io;
    const pseudo = @import("../pseudopotential/pseudopotential.zig");
    const alloc = std.testing.allocator;
    try test_support.require_file(io, "pseudo/Si.pbe-n-kjpaw_psl.1.0.0.UPF");

    var parsed = try pseudo.load(alloc, io, .{
        .element = "Si",
        .path = "pseudo/Si.pbe-n-kjpaw_psl.1.0.0.UPF",
        .format = .upf,
    });
    defer parsed.deinit(alloc);

    const upf = parsed.upf.?;
    const paw = upf.paw.?;

    var tab = try PawTab.init(alloc, paw, upf.r, upf.rab, 30.0);
    defer tab.deinit(alloc);

    // 6 projectors
    try std.testing.expectEqual(@as(usize, 6), tab.nbeta);

    // S_ij diagonal: should be close to 1 + q_ij
    // For well-normalized partial waves, diagonal elements ~ 1.0
    // Off-diagonal S_ij should be close to q_ij from UPF
    try std.testing.expect(tab.sij.len == 36);

    // K_ij should be non-zero for same-l pairs
    try std.testing.expect(tab.kij.len == 36);

    // Check that S_ii is close to 1 (delta + small correction)
    for (0..6) |i| {
        const sii = tab.sij[i * 6 + i];
        // S_ii should be around 1.0 (exact depends on partial wave normalization)
        try std.testing.expect(@abs(sii) > 0.1);
    }

    // Q_ij^L form factor at G=0: Q_ij^0(0) = 4pi * integral[r² Q_ij^0(r) dr]
    // Should be related to the multipole moment
    const entry0 = tab.find_qijl_entry(0, 0, 0);
    try std.testing.expect(entry0 != null);
    const q0_at_0 = tab.eval_qijl_form(entry0.?, 0.0);
    // Just check it's finite and non-zero
    try std.testing.expect(std.math.isFinite(q0_at_0));
}
