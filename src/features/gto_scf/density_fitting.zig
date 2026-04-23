//! Density Fitting (RI-J/K) for Coulomb and exchange matrix construction.
//!
//! Approximates the 4-center ERIs using an auxiliary basis:
//!   (μν|λσ) ≈ Σ_{PQ} (μν|P) (P|Q)^{-1} (Q|λσ)
//!
//! This reduces the J/K matrix build from O(N^4) to O(N^2 × N_aux + N_aux^3).
//!
//! Algorithm:
//!   1. Pre-compute 2-center Coulomb matrix (P|Q) and its Cholesky L
//!   2. Pre-compute 3-center integrals (μν|P) stored as eri3[μ*n+ν, P]
//!   3. J build: c = eri3^T · p_flat; solve L L^T d = c; J = eri3 · d
//!   4. K build: B = eri3 · L^{-T}; K_{μν} = Σ_P (B^P · P · B^P)_{μν}

const std = @import("std");
const basis_mod = @import("../basis/basis.zig");
const integrals = @import("../integrals/integrals.zig");
const obara_saika = integrals.obara_saika;
const eri_df = integrals.eri_df;
const blas = @import("../../lib/linalg/blas.zig");

const ContractedShell = basis_mod.ContractedShell;

fn buildCoulomb2Center(
    aux_shells: []const ContractedShell,
    n_aux: usize,
    coulomb_2c: []f64,
) void {
    @memset(coulomb_2c, 0.0);
    var shell_buf: [MAX_SHELL_CART * MAX_SHELL_CART]f64 = undefined;
    var off_p: usize = 0;
    for (aux_shells) |sp| {
        const np = basis_mod.numCartesian(sp.l);
        var off_q: usize = 0;
        for (aux_shells) |sq| {
            const nq = basis_mod.numCartesian(sq.l);
            _ = eri_df.contracted2CenterERI(sp, sq, &shell_buf);
            // Copy into full matrix
            for (0..np) |ip| {
                for (0..nq) |iq| {
                    const row = (off_p + ip) * n_aux + (off_q + iq);
                    coulomb_2c[row] = shell_buf[ip * nq + iq];
                }
            }
            off_q += nq;
        }
        off_p += np;
    }
}

fn zeroUpperTriangle(n_aux: usize, coulomb_2c: []f64) void {
    for (0..n_aux) |i| {
        for (i + 1..n_aux) |j| {
            coulomb_2c[i * n_aux + j] = 0.0;
        }
    }
}

fn build3CenterIntegrals(
    orbital_shells: []const ContractedShell,
    aux_shells: []const ContractedShell,
    n_basis: usize,
    n_aux: usize,
    eri3: []f64,
) void {
    @memset(eri3, 0.0);
    var shell_buf: [MAX_SHELL_CART * MAX_SHELL_CART * MAX_SHELL_CART]f64 = undefined;
    var off_a: usize = 0;
    for (orbital_shells) |sa| {
        const na_s = basis_mod.numCartesian(sa.l);
        var off_b: usize = 0;
        for (orbital_shells) |sb| {
            const nb_s = basis_mod.numCartesian(sb.l);
            var off_p: usize = 0;
            for (aux_shells) |sp| {
                const np_s = basis_mod.numCartesian(sp.l);
                _ = eri_df.contracted3CenterERI(sa, sb, sp, &shell_buf);
                // Copy: eri3[(off_a+ia)*n_basis+(off_b+ib), off_p+ip]
                for (0..na_s) |ia| {
                    for (0..nb_s) |ib| {
                        for (0..np_s) |ip| {
                            const mu = off_a + ia;
                            const nu = off_b + ib;
                            const p_idx = off_p + ip;
                            eri3[(mu * n_basis + nu) * n_aux + p_idx] =
                                shell_buf[ia * nb_s * np_s + ib * np_s + ip];
                        }
                    }
                }
                off_p += np_s;
            }
            off_b += nb_s;
        }
        off_a += na_s;
    }
}

pub const DensityFittingContext = struct {
    n_basis: usize,
    n_aux: usize,
    /// Cholesky factor L of (P|Q), row-major n_aux × n_aux (lower triangle)
    cholesky_l: []f64,
    /// 3-center integrals: eri3[mu*n_basis+nu, P] = (μν|P)
    /// Stored as n_basis² × n_aux row-major
    eri3: []f64,
    allocator: std.mem.Allocator,

    /// Build the DensityFittingContext from orbital and auxiliary shells.
    /// Computes (P|Q), its Cholesky factor, and all (μν|P) integrals.
    pub fn init(
        alloc: std.mem.Allocator,
        orbital_shells: []const ContractedShell,
        aux_shells: []const ContractedShell,
    ) !DensityFittingContext {
        const n_basis = obara_saika.totalBasisFunctions(orbital_shells);
        const n_aux = obara_saika.totalBasisFunctions(aux_shells);

        // Step 1: Build 2-center Coulomb matrix (P|Q) and Cholesky decompose
        const coulomb_2c = try alloc.alloc(f64, n_aux * n_aux);
        errdefer alloc.free(coulomb_2c);

        // Compute (P|Q) shell by shell
        buildCoulomb2Center(aux_shells, n_aux, coulomb_2c);

        // Cholesky decomposition: (P|Q) = L * L^T
        try blas.dpotrf(n_aux, coulomb_2c);
        // Zero the upper triangle (dpotrf leaves garbage there)
        zeroUpperTriangle(n_aux, coulomb_2c);

        // Step 2: Build 3-center integrals (μν|P)
        const eri3 = try alloc.alloc(f64, n_basis * n_basis * n_aux);
        errdefer alloc.free(eri3);

        build3CenterIntegrals(orbital_shells, aux_shells, n_basis, n_aux, eri3);

        return .{
            .n_basis = n_basis,
            .n_aux = n_aux,
            .cholesky_l = coulomb_2c,
            .eri3 = eri3,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *DensityFittingContext) void {
        self.allocator.free(self.cholesky_l);
        self.allocator.free(self.eri3);
    }

    /// Build the Coulomb (J) matrix using density fitting.
    ///
    /// Algorithm:
    ///   c_P = Σ_{λσ} P_{λσ} (λσ|P)     [contract density with 3c integrals]
    ///   Solve L * L^T * d = c            [Cholesky solve for fitting coefficients]
    ///   J_{μν} = Σ_P (μν|P) d_P          [expand with 3c integrals]
    pub fn buildJ(
        self: *const DensityFittingContext,
        alloc: std.mem.Allocator,
        p_mat: []const f64,
        j_mat: []f64,
    ) !void {
        const n = self.n_basis;
        const n_aux = self.n_aux;

        // c = eri3^T · p_flat  (n_aux vector)
        const c = try alloc.alloc(f64, n_aux);
        defer alloc.free(c);

        @memset(c, 0.0);

        // eri3 is (n*n) × n_aux, p_flat is n*n vector
        // c = eri3^T · p_flat
        blas.dgemv(.trans, n * n, n_aux, 1.0, self.eri3, n_aux, p_mat, 0.0, c);

        // Solve L * L^T * d = c
        // Forward: L * y = c
        blas.dtrsv(.lower, .no_trans, n_aux, self.cholesky_l, n_aux, c);
        // Backward: L^T * d = y
        blas.dtrsv(.lower, .trans, n_aux, self.cholesky_l, n_aux, c);
        // Now c contains d

        // J = eri3 · d
        blas.dgemv(.no_trans, n * n, n_aux, 1.0, self.eri3, n_aux, c, 0.0, j_mat);
    }

    /// Build the exchange (K) matrix using density fitting.
    ///
    /// Algorithm:
    ///   B_{μν,P} = Σ_Q (μν|Q) L^{-T}_{QP}   [solve triangular system]
    ///   K_{μν} = Σ_P Σ_λ B_{μλ,P} × (Σ_σ P_{λσ} B_{νσ,P})
    ///
    /// Equivalent to: K_{μν} = Σ_P (B^P · P · B^{P T})_{μν}
    pub fn buildK(
        self: *const DensityFittingContext,
        alloc: std.mem.Allocator,
        p_mat: []const f64,
        k_mat: []f64,
    ) !void {
        const n = self.n_basis;
        const n_aux = self.n_aux;

        // B = eri3 · L^{-T}  (n*n × n_aux)
        // eri3 is (n²) × n_aux, L is n_aux × n_aux
        // B_{μν,P} = Σ_Q eri3_{μν,Q} * (L^{-T})_{QP}
        // This is: B = eri3 · inv(L^T) = eri3 · L^{-T}
        // Solve: B · L^T = eri3, i.e., L · B^T = eri3^T
        // Using dtrsm: op(A) * X = alpha * B → L * B^T = eri3^T (column-by-column)
        // But in row-major, it's simpler to solve X * L^T = eri3, i.e.,
        // dtrsm(right, lower, trans, ...) on B
        const b_mat = try alloc.alloc(f64, n * n * n_aux);
        defer alloc.free(b_mat);

        @memcpy(b_mat, self.eri3);

        // Solve B * L^T = eri3 → B = eri3 * L^{-T}
        // dtrsm(right, lower, trans, m=n*n, n=n_aux, alpha=1, A=L, B=b_mat)
        blas.dtrsm(.right, .lower, .trans, n * n, n_aux, 1.0, self.cholesky_l, n_aux, b_mat, n_aux);

        // K_{μν} = Σ_P Σ_{λσ} B_{μλ,P} P_{λσ} B_{νσ,P}
        // = Σ_P (B^P P B^{PT})_{μν}
        // where B^P_{μλ} = B[μ*n+λ, P] = b_mat[(μ*n+λ)*n_aux + P]
        //
        // Efficient approach: for each P, extract B^P (n×n matrix), compute B^P · P,
        // then K += (B^P · P) · B^{PT}
        @memset(k_mat[0 .. n * n], 0.0);

        // Work buffer for B^P (n × n) and tmp = B^P · P (n × n)
        const bp_mat = try alloc.alloc(f64, n * n);
        defer alloc.free(bp_mat);

        const tmp = try alloc.alloc(f64, n * n);
        defer alloc.free(tmp);

        for (0..n_aux) |p_idx| {
            // Extract B^P: bp_mat[μ,λ] = b_mat[(μ*n+λ)*n_aux + p_idx]
            for (0..n) |mu| {
                for (0..n) |lam| {
                    bp_mat[mu * n + lam] = b_mat[(mu * n + lam) * n_aux + p_idx];
                }
            }

            // tmp = B^P · P  (n × n)
            blas.dgemm(.no_trans, .no_trans, n, n, n, 1.0, bp_mat, n, p_mat, n, 0.0, tmp, n);

            // K += tmp · B^{PT}  (n × n)
            blas.dgemm(.no_trans, .trans, n, n, n, 1.0, tmp, n, bp_mat, n, 1.0, k_mat, n);
        }
    }
};

const MAX_SHELL_CART: usize = basis_mod.MAX_CART;

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const math_mod = @import("../math/math.zig");

test "DF J matrix H2 STO-3G" {
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");
    const aux_basis = @import("../basis/aux_basis.zig");

    const r = 1.4;
    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = r, .y = 0.0, .z = 0.0 },
    };

    // Orbital basis
    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Simple even-tempered auxiliary basis for testing
    const aux1 = aux_basis.buildEvenTemperedAux(nuc_positions[0], 1, 4, 0.1, 3.0);
    const aux2 = aux_basis.buildEvenTemperedAux(nuc_positions[1], 1, 4, 0.1, 3.0);

    // Combine aux shells
    var all_aux: [aux_basis.MAX_AUX_SHELLS]ContractedShell = undefined;
    var aux_count: usize = 0;
    for (aux1.shells[0..aux1.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }
    for (aux2.shells[0..aux2.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }

    var df_ctx = try DensityFittingContext.init(alloc, &shells, all_aux[0..aux_count]);
    defer df_ctx.deinit();

    // Simple density matrix: P = [[1, 0.5], [0.5, 1]]
    const p_mat = [_]f64{ 1.0, 0.5, 0.5, 1.0 };
    var j_mat: [4]f64 = undefined;

    try df_ctx.buildJ(alloc, &p_mat, &j_mat);

    // J should be symmetric
    try testing.expectApproxEqAbs(j_mat[0 * 2 + 1], j_mat[1 * 2 + 0], 1e-10);

    // J should be positive (Coulomb repulsion)
    try testing.expect(j_mat[0] > 0.0);
    try testing.expect(j_mat[3] > 0.0);

    // Compare with conventional J
    var eri_table = try obara_saika.buildEriTable(alloc, &shells);
    defer eri_table.deinit(alloc);

    var j_ref: [4]f64 = undefined;
    const n: usize = 2;
    for (0..n) |mu| {
        for (0..n) |nu| {
            var j_val: f64 = 0.0;
            for (0..n) |lam| {
                for (0..n) |sig| {
                    j_val += p_mat[lam * n + sig] * eri_table.get(mu, nu, lam, sig);
                }
            }
            j_ref[mu * n + nu] = j_val;
        }
    }

    // DF J should approximate conventional J (tolerance depends on aux basis quality)
    for (0..4) |i| {
        try testing.expectApproxEqAbs(j_ref[i], j_mat[i], 0.02);
    }
}

test "DF K matrix H2 STO-3G" {
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");
    const aux_basis = @import("../basis/aux_basis.zig");

    const r = 1.4;
    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = r, .y = 0.0, .z = 0.0 },
    };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
    };

    const aux1 = aux_basis.buildEvenTemperedAux(nuc_positions[0], 1, 4, 0.1, 3.0);
    const aux2 = aux_basis.buildEvenTemperedAux(nuc_positions[1], 1, 4, 0.1, 3.0);

    var all_aux: [aux_basis.MAX_AUX_SHELLS]ContractedShell = undefined;
    var aux_count: usize = 0;
    for (aux1.shells[0..aux1.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }
    for (aux2.shells[0..aux2.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }

    var df_ctx = try DensityFittingContext.init(alloc, &shells, all_aux[0..aux_count]);
    defer df_ctx.deinit();

    const p_mat = [_]f64{ 1.0, 0.5, 0.5, 1.0 };
    var k_mat: [4]f64 = undefined;

    try df_ctx.buildK(alloc, &p_mat, &k_mat);

    // K should be symmetric
    try testing.expectApproxEqAbs(k_mat[0 * 2 + 1], k_mat[1 * 2 + 0], 1e-10);

    // Compare with conventional K
    var eri_table = try obara_saika.buildEriTable(alloc, &shells);
    defer eri_table.deinit(alloc);

    var k_ref: [4]f64 = undefined;
    const n: usize = 2;
    for (0..n) |mu| {
        for (0..n) |nu| {
            var k_val: f64 = 0.0;
            for (0..n) |lam| {
                for (0..n) |sig| {
                    k_val += p_mat[lam * n + sig] * eri_table.get(mu, lam, nu, sig);
                }
            }
            k_ref[mu * n + nu] = k_val;
        }
    }

    // DF K should approximate conventional K
    for (0..4) |i| {
        try testing.expectApproxEqAbs(k_ref[i], k_mat[i], 0.02);
    }
}

test "DF J/K symmetry" {
    const alloc = testing.allocator;
    const sto3g = @import("../basis/sto3g.zig");
    const aux_basis = @import("../basis/aux_basis.zig");

    const nuc_positions = [_]math_mod.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 1.4305226763, .z = 1.1092692351 },
        .{ .x = 0.0, .y = -1.4305226763, .z = 1.1092692351 },
    };

    const shells = [_]ContractedShell{
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_1s },
        .{ .center = nuc_positions[0], .l = 0, .primitives = &sto3g.O_2s },
        .{ .center = nuc_positions[0], .l = 1, .primitives = &sto3g.O_2p },
        .{ .center = nuc_positions[1], .l = 0, .primitives = &sto3g.H_1s },
        .{ .center = nuc_positions[2], .l = 0, .primitives = &sto3g.H_1s },
    };

    // Build def2-universal-jkfit auxiliary basis
    const aux_o = aux_basis.buildDef2UniversalJkfit(8, nuc_positions[0]).?;
    const aux_h1 = aux_basis.buildDef2UniversalJkfit(1, nuc_positions[1]).?;
    const aux_h2 = aux_basis.buildDef2UniversalJkfit(1, nuc_positions[2]).?;

    var all_aux: [aux_basis.MAX_AUX_SHELLS * 3]ContractedShell = undefined;
    var aux_count: usize = 0;
    for (aux_o.shells[0..aux_o.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }
    for (aux_h1.shells[0..aux_h1.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }
    for (aux_h2.shells[0..aux_h2.count]) |s| {
        all_aux[aux_count] = s;
        aux_count += 1;
    }

    var df_ctx = try DensityFittingContext.init(alloc, &shells, all_aux[0..aux_count]);
    defer df_ctx.deinit();

    const n = df_ctx.n_basis; // 7

    // Use identity-like density matrix
    const p_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(p_mat);

    @memset(p_mat, 0.0);
    for (0..n) |i| {
        p_mat[i * n + i] = 1.0;
    }

    const j_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(j_mat);

    const k_mat = try alloc.alloc(f64, n * n);
    defer alloc.free(k_mat);

    try df_ctx.buildJ(alloc, p_mat, j_mat);
    try df_ctx.buildK(alloc, p_mat, k_mat);

    // Check symmetry
    for (0..n) |mu| {
        for (mu + 1..n) |nu| {
            try testing.expectApproxEqAbs(j_mat[mu * n + nu], j_mat[nu * n + mu], 1e-10);
            try testing.expectApproxEqAbs(k_mat[mu * n + nu], k_mat[nu * n + mu], 1e-10);
        }
    }
}

// Slow density-fitting regression coverage lives in `regression_tests.zig`.
