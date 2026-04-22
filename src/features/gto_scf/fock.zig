//! Fock matrix construction for Restricted Hartree-Fock.
//!
//! The Fock matrix is:
//!   F_μν = H_core_μν + G_μν(P)
//!
//! where H_core = T + V (kinetic + nuclear attraction), and G is the
//! two-electron part:
//!   G_μν = Σ_λσ P_λσ × [(μν|λσ) - ½(μλ|νσ)]
//!
//! The first term is the Coulomb (J) contribution and the second is
//! the exchange (K) contribution.
//!
//! Two modes are supported:
//!   1. ERI table: Pre-computed table with O(N^4) lookup (small basis sets).
//!   2. Direct SCF: On-the-fly ERI computation with Schwarz screening (large basis sets).

const std = @import("std");
const eri_mod = @import("../integrals/eri.zig");
const obara_saika = @import("../integrals/obara_saika.zig");
const rys_eri = @import("../integrals/rys_eri.zig");
const basis_mod = @import("../basis/basis.zig");
const ContractedShell = basis_mod.ContractedShell;
const AngularMomentum = basis_mod.AngularMomentum;
const EriTable = eri_mod.EriTable;
const GeneralEriTable = obara_saika.GeneralEriTable;

/// Build the Fock matrix F = H_core + G(P) (s-only, legacy).
pub fn buildFockMatrix(
    alloc: std.mem.Allocator,
    n: usize,
    h_core: []const f64,
    p: []const f64,
    eri_table: EriTable,
) ![]f64 {
    std.debug.assert(h_core.len == n * n);
    std.debug.assert(p.len == n * n);

    const f = try alloc.alloc(f64, n * n);
    updateFockMatrix(n, h_core, p, eri_table, f);
    return f;
}

/// Update the Fock matrix in-place: F = H_core + G(P) (s-only, legacy).
pub fn updateFockMatrix(
    n: usize,
    h_core: []const f64,
    p: []const f64,
    eri_table: EriTable,
    f: []f64,
) void {
    std.debug.assert(h_core.len == n * n);
    std.debug.assert(p.len == n * n);
    std.debug.assert(f.len == n * n);

    for (0..n) |mu| {
        for (0..n) |nu| {
            var g: f64 = 0.0;
            for (0..n) |lam| {
                for (0..n) |sig| {
                    const p_ls = p[lam * n + sig];
                    const j = eri_table.get(mu, nu, lam, sig);
                    const k = eri_table.get(mu, lam, nu, sig);
                    g += p_ls * (j - 0.5 * k);
                }
            }
            f[mu * n + nu] = h_core[mu * n + nu] + g;
        }
    }
}

/// Update the Fock matrix in-place using general ERI table (any angular momentum).
pub fn updateFockMatrixGeneral(
    n: usize,
    h_core: []const f64,
    p: []const f64,
    eri_table: GeneralEriTable,
    f: []f64,
) void {
    std.debug.assert(h_core.len == n * n);
    std.debug.assert(p.len == n * n);
    std.debug.assert(f.len == n * n);

    for (0..n) |mu| {
        for (0..n) |nu| {
            var g: f64 = 0.0;
            for (0..n) |lam| {
                for (0..n) |sig| {
                    const p_ls = p[lam * n + sig];
                    const j = eri_table.get(mu, nu, lam, sig);
                    const k = eri_table.get(mu, lam, nu, sig);
                    g += p_ls * (j - 0.5 * k);
                }
            }
            f[mu * n + nu] = h_core[mu * n + nu] + g;
        }
    }
}

// ============================================================================
// Schwarz screening and direct SCF
// ============================================================================

/// Maximum number of shells supported.
const MAX_SHELLS: usize = 128;

/// Schwarz screening data for shell pairs.
/// Q[i][j] = max over Cartesian components sqrt(|(ij|ij)|)
pub const SchwarzTable = struct {
    /// Schwarz upper bounds: Q[i * n_shells + j] for shell pair (i,j).
    q: []f64,
    /// Number of shells.
    n_shells: usize,
    /// Shell offsets (basis function index where each shell starts).
    shell_offsets: [MAX_SHELLS]usize,
    /// Number of Cartesian functions per shell.
    shell_sizes: [MAX_SHELLS]usize,
    /// Total number of basis functions.
    n_basis: usize,

    pub fn deinit(self: *SchwarzTable, alloc: std.mem.Allocator) void {
        if (self.q.len > 0) alloc.free(self.q);
    }

    /// Get Q value for shell pair (i, j).
    pub fn get(self: *const SchwarzTable, i: usize, j: usize) f64 {
        return self.q[i * self.n_shells + j];
    }
};

/// Build the Schwarz screening table.
/// For each shell pair (A,B), compute Q_AB = max_{a in A, b in B} sqrt(|(ab|ab)|).
/// Uses Rys quadrature shell-quartet ERI for efficiency.
pub fn buildSchwarzTable(
    alloc: std.mem.Allocator,
    shells: []const ContractedShell,
) !SchwarzTable {
    const n_shells = shells.len;
    std.debug.assert(n_shells <= MAX_SHELLS);

    // Compute shell offsets
    var shell_offsets: [MAX_SHELLS]usize = undefined;
    var shell_sizes: [MAX_SHELLS]usize = undefined;
    var offset: usize = 0;
    for (shells, 0..) |shell, si| {
        shell_offsets[si] = offset;
        const nc = shell.numCartesianFunctions();
        shell_sizes[si] = nc;
        offset += nc;
    }
    const n_basis = offset;

    const q = try alloc.alloc(f64, n_shells * n_shells);
    @memset(q, 0.0);

    // Stack buffer for batch ERI output
    const MAX_BATCH: usize = 15 * 15 * 15 * 15;
    var eri_buf: [MAX_BATCH]f64 = undefined;

    for (0..n_shells) |si| {
        const na = shell_sizes[si];
        for (si..n_shells) |sj| {
            const nb = shell_sizes[sj];

            // Compute (AB|AB) shell quartet using Rys ERI
            _ = rys_eri.contractedShellQuartetERI(
                shells[si],
                shells[sj],
                shells[si],
                shells[sj],
                &eri_buf,
            );

            // Find max |(ab|ab)| over all Cartesian components
            var max_val: f64 = 0.0;
            for (0..na) |ia| {
                for (0..nb) |ib| {
                    const val = eri_buf[ia * nb * na * nb + ib * na * nb + ia * nb + ib];
                    const abs_val = @abs(val);
                    if (abs_val > max_val) max_val = abs_val;
                }
            }
            const q_val = @sqrt(max_val);
            q[si * n_shells + sj] = q_val;
            q[sj * n_shells + si] = q_val;
        }
    }

    return .{
        .q = q,
        .n_shells = n_shells,
        .shell_offsets = shell_offsets,
        .shell_sizes = shell_sizes,
        .n_basis = n_basis,
    };
}

/// Build Fock matrix directly from shells using Schwarz screening (no ERI table).
/// This is the "direct SCF" approach.
///
/// Computes F = H_core + G(P) where:
///   G_μν = Σ_λσ P_λσ × [(μν|λσ) - hf_frac * (μλ|νσ)]
///
/// For RHF: hf_frac = 0.5
/// For KS-DFT with B3LYP: only call with the J part (hf_frac=0) and K part separately,
/// or use buildJKDirect.
pub fn buildFockDirect(
    n: usize,
    h_core: []const f64,
    p: []const f64,
    shells: []const ContractedShell,
    schwarz: *const SchwarzTable,
    threshold: f64,
    f: []f64,
) void {
    std.debug.assert(h_core.len == n * n);
    std.debug.assert(p.len == n * n);
    std.debug.assert(f.len == n * n);

    // Start with H_core
    @memcpy(f, h_core);

    // Build G directly via shell quartet loop
    buildGDirect(n, p, shells, schwarz, threshold, 0.5, f);
}

/// Build J and K matrices directly using Schwarz screening.
/// J_μν = Σ_λσ P_λσ (μν|λσ)
/// K_μν = Σ_λσ P_λσ (μλ|νσ)
///
/// Uses shell-quartet loops with Schwarz screening to skip negligible integrals.
/// This is a "direct SCF" approach: ERIs are computed on-the-fly, no ERI table needed.
///
/// Exploits 8-fold shell-quartet symmetry:
///   (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)
///
/// Loop: sa >= sb, sc >= sd, and combined pair(sa,sb) >= pair(sc,sd).
/// This reduces the number of unique shell quartets by ~8x.
///
/// Uses batch ERI computation (contractedShellQuartetERI) to build the theta table
/// once per primitive quartet and extract all Cartesian component ERIs at once.
pub fn buildJKDirect(
    n: usize,
    p: []const f64,
    shells: []const ContractedShell,
    schwarz: *const SchwarzTable,
    threshold: f64,
    j_mat: []f64,
    k_mat: []f64,
) void {
    std.debug.assert(p.len == n * n);
    std.debug.assert(j_mat.len == n * n);
    std.debug.assert(k_mat.len == n * n);

    @memset(j_mat, 0.0);
    @memset(k_mat, 0.0);

    const n_shells = schwarz.n_shells;

    // Stack buffer for batch ERI output.
    // Maximum size: MAX_CART^4 = 15^4 = 50625 (f|f|f|f case).
    const MAX_BATCH: usize = 15 * 15 * 15 * 15;
    var eri_buf: [MAX_BATCH]f64 = undefined;

    // 8-fold symmetric shell loop
    for (0..n_shells) |sa| {
        const na = schwarz.shell_sizes[sa];
        const off_a = schwarz.shell_offsets[sa];

        for (0..sa + 1) |sb| {
            const nb = schwarz.shell_sizes[sb];
            const off_b = schwarz.shell_offsets[sb];
            const q_ab = schwarz.get(sa, sb);

            const ab_pair = pairIndex(sa, sb);

            for (0..n_shells) |sc| {
                const nc = schwarz.shell_sizes[sc];
                const off_c = schwarz.shell_offsets[sc];

                for (0..sc + 1) |sd| {
                    const cd_pair = pairIndex(sc, sd);
                    if (cd_pair > ab_pair) continue;

                    const q_cd = schwarz.get(sc, sd);
                    if (q_ab * q_cd < threshold) continue;

                    const nd = schwarz.shell_sizes[sd];
                    const off_d = schwarz.shell_offsets[sd];

                    // Compute ALL ERIs for this shell quartet at once using Rys quadrature
                    _ = rys_eri.contractedShellQuartetERI(
                        shells[sa],
                        shells[sb],
                        shells[sc],
                        shells[sd],
                        &eri_buf,
                    );

                    // Determine symmetry flags for this shell quartet
                    const ab_same = (sa == sb);
                    const cd_same = (sc == sd);
                    const abcd_same = (ab_pair == cd_pair);

                    // Distribute batch ERIs to J and K matrices
                    for (0..na) |ia| {
                        const mu = off_a + ia;
                        for (0..nb) |ib| {
                            const nu = off_b + ib;
                            for (0..nc) |ic| {
                                const lam = off_c + ic;
                                for (0..nd) |id_d| {
                                    const sig = off_d + id_d;

                                    const idx_eri = ia * nb * nc * nd +
                                        ib * nc * nd + ic * nd + id_d;
                                    const eri = eri_buf[idx_eri];

                                    // Distribute ERI to J and K using all 8 permutational
                                    // symmetries

                                    // 1. (mu,nu | lam,sig) — original
                                    j_mat[mu * n + nu] += p[lam * n + sig] * eri;
                                    k_mat[mu * n + lam] += p[nu * n + sig] * eri;

                                    // 2. (nu,mu | lam,sig) — swap bra: only if sa != sb
                                    if (!ab_same) {
                                        j_mat[nu * n + mu] += p[lam * n + sig] * eri;
                                        k_mat[nu * n + lam] += p[mu * n + sig] * eri;
                                    }

                                    // 3. (mu,nu | sig,lam) — swap ket: only if sc != sd
                                    if (!cd_same) {
                                        j_mat[mu * n + nu] += p[sig * n + lam] * eri;
                                        k_mat[mu * n + sig] += p[nu * n + lam] * eri;
                                    }

                                    // 4. (nu,mu | sig,lam) — swap both bra and ket
                                    if (!ab_same and !cd_same) {
                                        j_mat[nu * n + mu] += p[sig * n + lam] * eri;
                                        k_mat[nu * n + sig] += p[mu * n + lam] * eri;
                                    }

                                    // 5. (lam,sig | mu,nu) — bra-ket exchange: only if ab != cd
                                    if (!abcd_same) {
                                        j_mat[lam * n + sig] += p[mu * n + nu] * eri;
                                        k_mat[lam * n + mu] += p[sig * n + nu] * eri;

                                        // 6. (sig,lam | mu,nu) — bra-ket + swap bra'
                                        if (!cd_same) {
                                            j_mat[sig * n + lam] += p[mu * n + nu] * eri;
                                            k_mat[sig * n + mu] += p[lam * n + nu] * eri;
                                        }

                                        // 7. (lam,sig | nu,mu) — bra-ket + swap ket'
                                        if (!ab_same) {
                                            j_mat[lam * n + sig] += p[nu * n + mu] * eri;
                                            k_mat[lam * n + nu] += p[sig * n + mu] * eri;
                                        }

                                        // 8. (sig,lam | nu,mu) — all three swaps
                                        if (!ab_same and !cd_same) {
                                            j_mat[sig * n + lam] += p[nu * n + mu] * eri;
                                            k_mat[sig * n + nu] += p[lam * n + mu] * eri;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Pair index for shell pairs: maps (a, b) with a >= b to a*(a+1)/2 + b.
fn pairIndex(a: usize, b: usize) usize {
    if (a >= b) return a * (a + 1) / 2 + b;
    return b * (b + 1) / 2 + a;
}

/// Build the G matrix directly: G = J - hf_frac * K.
/// Adds the result to f (which should already contain H_core).
fn buildGDirect(
    n: usize,
    p: []const f64,
    shells: []const ContractedShell,
    schwarz: *const SchwarzTable,
    threshold: f64,
    hf_frac: f64,
    f: []f64,
) void {
    const alloc = std.heap.page_allocator;
    const j_mat = alloc.alloc(f64, n * n) catch return;
    defer alloc.free(j_mat);
    const k_mat = alloc.alloc(f64, n * n) catch return;
    defer alloc.free(k_mat);

    buildJKDirect(n, p, shells, schwarz, threshold, j_mat, k_mat);

    for (0..n * n) |i| {
        f[i] += j_mat[i] - hf_frac * k_mat[i];
    }
}

test "Direct SCF J/K matches ERI table (H2 STO-3G)" {
    const testing = std.testing;
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");
    const math = @import("../math/math.zig");

    // H2 at R=1.4 bohr — only 2 s-shells, 2 basis functions
    const shells = [_]basis_mod.ContractedShell{
        .{ .center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 }, .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = math.Vec3{ .x = 1.4, .y = 0.0, .z = 0.0 }, .l = 0, .primitives = &sto3g.H_1s },
    };

    const n: usize = 2;

    // Simple density matrix (approximate — just needs to be nonzero for testing)
    const p_mat = [_]f64{ 0.6, 0.4, 0.4, 0.6 };

    // --- ERI table approach ---
    var eri_table = try obara_saika.buildEriTable(alloc, &shells);
    defer eri_table.deinit(alloc);

    var j_ref: [4]f64 = undefined;
    var k_ref: [4]f64 = undefined;

    for (0..n) |mu| {
        for (0..n) |nu| {
            var j_val: f64 = 0.0;
            var k_val: f64 = 0.0;
            for (0..n) |lam| {
                for (0..n) |sig| {
                    j_val += p_mat[lam * n + sig] * eri_table.get(mu, nu, lam, sig);
                    k_val += p_mat[lam * n + sig] * eri_table.get(mu, lam, nu, sig);
                }
            }
            j_ref[mu * n + nu] = j_val;
            k_ref[mu * n + nu] = k_val;
        }
    }

    // --- Direct SCF approach ---
    var schwarz = try buildSchwarzTable(alloc, &shells);
    defer schwarz.deinit(alloc);

    var j_direct: [4]f64 = undefined;
    var k_direct: [4]f64 = undefined;

    buildJKDirect(n, &p_mat, &shells, &schwarz, 1e-14, &j_direct, &k_direct);

    for (0..n * n) |i| {
        try testing.expectApproxEqAbs(j_ref[i], j_direct[i], 1e-10);
        try testing.expectApproxEqAbs(k_ref[i], k_direct[i], 1e-10);
    }
}

test "Fock matrix equals H_core when P is zero" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const n: usize = 2;
    const h_core = [_]f64{ -1.0, -0.5, -0.5, -1.0 };
    const p = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

    // Build a trivial ERI table (all zeros) — need to use the actual builder
    // but for zero density, G=0 regardless, so any ERI table works
    const sto3g = @import("../basis/sto3g.zig");
    const basis = @import("../basis/basis.zig");
    const math = @import("../math/math.zig");
    const shells = [_]basis.ContractedShell{
        .{ .center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 }, .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = math.Vec3{ .x = 1.4, .y = 0.0, .z = 0.0 }, .l = 0, .primitives = &sto3g.H_1s },
    };
    var eri_table = try eri_mod.buildEriTable(alloc, &shells);
    defer eri_table.deinit(alloc);

    const f = try buildFockMatrix(alloc, n, &h_core, &p, eri_table);
    defer alloc.free(f);

    // With P=0, F should equal H_core
    for (0..n * n) |i| {
        try testing.expectApproxEqAbs(f[i], h_core[i], 1e-12);
    }
}
