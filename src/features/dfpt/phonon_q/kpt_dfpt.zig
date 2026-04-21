//! Multi-k-point DFPT data structures and builder.
//!
//! Given precomputed q-independent ground-state data (KPointGsData) and a
//! target q, builds the per-k-point data set the DFPT SCF solver needs:
//! the k+q PW basis, ApplyContext, and occupied wavefunctions at k+q
//! (solved from scratch when q≠0, reused from k when q=Γ).

const std = @import("std");
const math = @import("../../math/math.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../../scf/scf.zig");
const plane_wave = @import("../../plane_wave/basis.zig");
const config_mod = @import("../../config/config.zig");
const iterative = @import("../../linalg/iterative.zig");

const kpt_gs = @import("kpt_gs.zig");
const KPointGsData = kpt_gs.KPointGsData;

const Grid = scf_mod.Grid;

/// Data for a single k-point in the DFPT calculation.
/// Holds the PW basis, wavefunctions, eigenvalues, and apply context
/// for both the k-point and k+q-point.
pub const KPointDfptData = struct {
    /// K-point in fractional coordinates
    k_frac: math.Vec3,
    /// K-point in Cartesian coordinates
    k_cart: math.Vec3,
    /// IBZ weight for this k-point
    weight: f64,
    /// Number of occupied bands at this k-point
    n_occ: usize,
    /// Number of PW basis functions at k
    n_pw_k: usize,
    /// PW basis G-vectors at k
    basis_k: plane_wave.Basis,
    /// PwGridMap for k-basis
    map_k: scf_mod.PwGridMap,
    /// Apply context for H_k
    apply_ctx_k: *scf_mod.ApplyContext,
    /// Eigenvalues at k
    eigenvalues_k: []f64,
    /// Occupied wavefunctions at k: [n_occ][n_pw_k]
    wavefunctions_k: [][]math.Complex,
    /// Const view of wavefunctions at k
    wavefunctions_k_const: [][]const math.Complex,
    /// Number of PW basis functions at k+q
    n_pw_kq: usize,
    /// PW basis G-vectors at k+q
    basis_kq: plane_wave.Basis,
    /// PwGridMap for k+q-basis
    map_kq: scf_mod.PwGridMap,
    /// Apply context for H_{k+q}
    apply_ctx_kq: *scf_mod.ApplyContext,
    /// Occupied wavefunctions at k+q (for P_c projection): [n_occ_kq][n_pw_kq]
    occ_kq: [][]math.Complex,
    /// Const view
    occ_kq_const: [][]const math.Complex,
    /// Number of occupied bands at k+q
    n_occ_kq: usize,

    pub fn deinit(self: *KPointDfptData, alloc: std.mem.Allocator) void {
        self.deinitQOnly(alloc);
        for (self.wavefunctions_k) |w| alloc.free(w);
        alloc.free(self.wavefunctions_k);
        alloc.free(self.wavefunctions_k_const);
        alloc.free(self.eigenvalues_k);
        self.apply_ctx_k.deinit(alloc);
        alloc.destroy(self.apply_ctx_k);
        self.map_k.deinit(alloc);
        self.basis_k.deinit(alloc);
    }

    /// Deinit only the q-dependent data (k+q basis, wavefunctions, etc.).
    /// Used when k-data is owned by KPointGsData and shared.
    pub fn deinitQOnly(self: *KPointDfptData, alloc: std.mem.Allocator) void {
        for (self.occ_kq) |w| alloc.free(w);
        alloc.free(self.occ_kq);
        alloc.free(self.occ_kq_const);
        self.apply_ctx_kq.deinit(alloc);
        alloc.destroy(self.apply_ctx_kq);
        self.map_kq.deinit(alloc);
        self.basis_kq.deinit(alloc);
    }
};

/// Result of a multi-k-point perturbation SCF.
/// Stores the converged ρ^(1)(G) (summed over k-points) and per-k-point ψ^(1).
pub const MultiKPertResult = struct {
    /// First-order density response ρ^(1)(G) summed over all k-points
    rho1_g: []math.Complex,
    /// First-order wavefunctions per k-point: [n_kpts][n_occ][n_pw_kq]
    psi1_per_k: [][]const []math.Complex,

    pub fn deinit(self: *MultiKPertResult, alloc: std.mem.Allocator) void {
        if (self.rho1_g.len > 0) alloc.free(self.rho1_g);
        for (self.psi1_per_k) |psi1_k| {
            for (psi1_k) |p| {
                if (p.len > 0) alloc.free(@constCast(p));
            }
            alloc.free(@constCast(psi1_k));
        }
        alloc.free(@constCast(self.psi1_per_k));
    }
};

/// Build KPointDfptData array from precomputed ground-state data for a given q-point.
/// The k-point data (basis, wavefunctions, eigenvalues) is shared from KPointGsData.
/// Only the k+q data (basis_kq, occ_kq, etc.) is newly allocated per q-point.
pub fn buildKPointDfptDataFromGS(
    alloc: std.mem.Allocator,
    io: std.Io,
    kgs: []const KPointGsData,
    q_cart: math.Vec3,
    q_norm: f64,
    cfg: config_mod.Config,
    local_r: []const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    num_workspaces: usize,
) ![]KPointDfptData {
    const n_kpts = kgs.len;
    var kpts = try alloc.alloc(KPointDfptData, n_kpts);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| kpts[i].deinitQOnly(alloc);
        alloc.free(kpts);
    }

    for (kgs, 0..) |*kg, ik| {
        const kq_cart = math.Vec3{
            .x = kg.k_cart.x + q_cart.x,
            .y = kg.k_cart.y + q_cart.y,
            .z = kg.k_cart.z + q_cart.z,
        };
        const is_q_gamma = q_norm < 1e-10;

        // Build k+q basis
        var basis_kq = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kq_cart);
        errdefer basis_kq.deinit(alloc);
        const n_pw_kq = basis_kq.gvecs.len;

        var map_kq = try scf_mod.PwGridMap.init(alloc, @constCast(basis_kq.gvecs), grid);
        errdefer map_kq.deinit(alloc);

        const apply_ctx_kq = try alloc.create(scf_mod.ApplyContext);
        errdefer alloc.destroy(apply_ctx_kq);
        apply_ctx_kq.* = try scf_mod.ApplyContext.initWithWorkspaces(
            alloc,
            io,
            grid,
            @constCast(basis_kq.gvecs),
            local_r,
            null,
            species,
            atoms,
            1.0 / volume,
            true,
            null,
            null,
            cfg.scf.fft_backend,
            num_workspaces,
        );
        errdefer apply_ctx_kq.deinit(alloc);

        const n_occ_kq = kg.n_occ;
        const occ_kq = try alloc.alloc([]math.Complex, n_occ_kq);
        errdefer {
            for (occ_kq) |w| alloc.free(w);
            alloc.free(occ_kq);
        }
        const occ_kq_const = try alloc.alloc([]const math.Complex, n_occ_kq);
        errdefer alloc.free(occ_kq_const);

        if (is_q_gamma) {
            // q=0: k+q = k, reuse wavefunctions
            for (0..n_occ_kq) |n| {
                occ_kq[n] = try alloc.alloc(math.Complex, n_pw_kq);
                @memcpy(occ_kq[n], kg.wavefunctions_k[n]);
                occ_kq_const[n] = occ_kq[n];
            }
        } else {
            // q≠0: solve eigenvalue problem at k+q
            const diag_kq = try alloc.alloc(f64, n_pw_kq);
            defer alloc.free(diag_kq);
            for (basis_kq.gvecs, 0..) |g, i| {
                diag_kq[i] = g.kinetic;
            }

            const nbands_kq = @max(kg.n_occ + 2, @as(usize, 8));
            const op_kq = iterative.Operator{
                .n = n_pw_kq,
                .ctx = @ptrCast(apply_ctx_kq),
                .apply = &scf_mod.applyHamiltonian,
                .apply_batch = &scf_mod.applyHamiltonianBatched,
            };
            var eig_kq = try iterative.hermitianEigenDecompIterative(
                alloc,
                cfg.linalg_backend,
                op_kq,
                diag_kq,
                nbands_kq,
                .{ .max_iter = 100, .tol = 1e-8, .init_diagonal = true },
            );
            defer eig_kq.deinit(alloc);

            for (0..n_occ_kq) |n| {
                occ_kq[n] = try alloc.alloc(math.Complex, n_pw_kq);
                @memcpy(occ_kq[n], eig_kq.vectors[n * n_pw_kq .. (n + 1) * n_pw_kq]);
                occ_kq_const[n] = occ_kq[n];
            }
        }

        kpts[ik] = KPointDfptData{
            .k_frac = kg.k_frac,
            .k_cart = kg.k_cart,
            .weight = kg.weight,
            .n_occ = kg.n_occ,
            .n_pw_k = kg.n_pw_k,
            .basis_k = kg.basis_k, // shared, not owned
            .map_k = kg.map_k, // shared, not owned
            .apply_ctx_k = kg.apply_ctx_k, // shared, not owned
            .eigenvalues_k = kg.eigenvalues_k, // shared, not owned
            .wavefunctions_k = kg.wavefunctions_k, // shared, not owned
            .wavefunctions_k_const = kg.wavefunctions_k_const, // shared, not owned
            .n_pw_kq = n_pw_kq,
            .basis_kq = basis_kq,
            .map_kq = map_kq,
            .apply_ctx_kq = apply_ctx_kq,
            .occ_kq = occ_kq,
            .occ_kq_const = occ_kq_const,
            .n_occ_kq = n_occ_kq,
        };
        built = ik + 1;
    }

    return kpts;
}
