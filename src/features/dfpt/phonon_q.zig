//! Finite-q DFPT phonon band structure calculation.
//!
//! Contains the q≠0 perturbation solver, cross-basis operations,
//! complex dynamical matrix construction, q-path generation,
//! and the top-level `runPhononBand` entry point.

const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../scf/scf.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const d3 = @import("../vdw/d3.zig");
const d3_params = @import("../vdw/d3_params.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const config_mod = @import("../config/config.zig");
const iterative = @import("../linalg/iterative.zig");
const symmetry_mod = @import("../symmetry/symmetry.zig");

const dfpt = @import("dfpt.zig");
const perturbation = dfpt.perturbation;
const sternheimer = dfpt.sternheimer;
const ewald2 = dfpt.ewald2;
const dynmat_mod = dfpt.dynmat;
const dynmat_contrib = dfpt.dynmat_contrib;
const gamma = dfpt.gamma;

const GroundState = dfpt.GroundState;
const PreparedGroundState = dfpt.PreparedGroundState;
const DfptConfig = dfpt.DfptConfig;
const IonicData = dfpt.IonicData;
const PerturbationResult = dfpt.PerturbationResult;
const logDfpt = dfpt.logDfpt;

const kpoints_mod = @import("../kpoints/kpoints.zig");
const symmetry = @import("../symmetry/symmetry.zig");
const mesh_mod = @import("../kpoints/mesh.zig");
const reduction = @import("../kpoints/reduction.zig");
const wfn_rot = @import("../symmetry/wavefunction_rotation.zig");

const Grid = scf_mod.Grid;

// =====================================================================
// K-point ground-state data (q-independent)
// =====================================================================

/// Ground-state data for a single k-point, independent of q.
/// Holds PW basis, wavefunctions, eigenvalues, and apply context at k.
/// This data is precomputed once for the full BZ and reused across q-points.
pub const KPointGsData = struct {
    k_frac: math.Vec3,
    k_cart: math.Vec3,
    weight: f64,
    n_occ: usize,
    n_pw_k: usize,
    basis_k: plane_wave.Basis,
    map_k: scf_mod.PwGridMap,
    apply_ctx_k: *scf_mod.ApplyContext,
    eigenvalues_k: []f64,
    wavefunctions_k: [][]math.Complex,
    wavefunctions_k_const: [][]const math.Complex,

    pub fn deinit(self: *KPointGsData, alloc: std.mem.Allocator) void {
        for (self.wavefunctions_k) |w| alloc.free(w);
        alloc.free(self.wavefunctions_k);
        alloc.free(self.wavefunctions_k_const);
        alloc.free(self.eigenvalues_k);
        self.apply_ctx_k.deinit(alloc);
        alloc.destroy(self.apply_ctx_k);
        self.map_k.deinit(alloc);
        self.basis_k.deinit(alloc);
    }
};

/// Prepare full-BZ k-point ground-state data for DFPT.
/// Generates the full Monkhorst-Pack mesh (no symmetry reduction) and
/// solves the eigenvalue problem at each k-point using the SCF potential.
/// This is equivalent to ABINIT's kptopt=3 for DFPT calculations.
pub fn prepareFullBZKpoints(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    gs: *const GroundState,
    local_r: []const f64,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
) ![]KPointGsData {
    // Generate full BZ k-point mesh (no symmetry reduction)
    const kmesh = cfg.scf.kmesh;
    const shift = math.Vec3{
        .x = cfg.scf.kmesh_shift[0],
        .y = cfg.scf.kmesh_shift[1],
        .z = cfg.scf.kmesh_shift[2],
    };
    const full_kpts = try mesh_mod.generateKmesh(alloc, kmesh, recip, shift);
    defer alloc.free(full_kpts);

    const n_kpts = full_kpts.len;
    logDfpt("dfpt_band: generating full BZ k-mesh: {d}x{d}x{d} = {d} k-points\n", .{ kmesh[0], kmesh[1], kmesh[2], n_kpts });

    var kgs_data = try alloc.alloc(KPointGsData, n_kpts);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| kgs_data[i].deinit(alloc);
        alloc.free(kgs_data);
    }

    for (full_kpts, 0..) |kp, ik| {
        const k_frac = kp.k_frac;
        const k_cart = kp.k_cart;
        const wtk = kp.weight; // 1/N_total for full BZ

        // Generate PW basis at k
        var basis_k = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, k_cart);
        errdefer basis_k.deinit(alloc);
        const n_pw_k = basis_k.gvecs.len;

        // Build PwGridMap for k
        var map_k = try scf_mod.PwGridMap.init(alloc, @constCast(basis_k.gvecs), grid);
        errdefer map_k.deinit(alloc);

        // Build ApplyContext for H_k
        const apply_ctx_k = try alloc.create(scf_mod.ApplyContext);
        errdefer alloc.destroy(apply_ctx_k);
        apply_ctx_k.* = try scf_mod.ApplyContext.init(
            alloc,
                io,
            grid,
            @constCast(basis_k.gvecs),
            local_r,
            null,
            species,
            atoms,
            1.0 / volume,
            true,
            null,
            null,
            cfg.scf.fft_backend,
        );
        errdefer apply_ctx_k.deinit(alloc);

        // Solve eigenvalue problem at k
        const diag_k = try alloc.alloc(f64, n_pw_k);
        defer alloc.free(diag_k);
        for (basis_k.gvecs, 0..) |g, i| {
            diag_k[i] = g.kinetic;
        }

        const n_occ_k = gs.n_occ;
        const nbands_k = @max(n_occ_k + 2, @as(usize, 8));
        const op_k = iterative.Operator{
            .n = n_pw_k,
            .ctx = @ptrCast(apply_ctx_k),
            .apply = &scf_mod.applyHamiltonian,
            .apply_batch = &scf_mod.applyHamiltonianBatched,
        };
        var eig_k = try iterative.hermitianEigenDecompIterative(
            alloc,
            cfg.linalg_backend,
            op_k,
            diag_k,
            nbands_k,
            .{ .max_iter = 100, .tol = 1e-8, .init_diagonal = true },
        );
        defer eig_k.deinit(alloc);

        // Extract occupied eigenvalues and wavefunctions
        const eigenvalues_k = try alloc.alloc(f64, n_occ_k);
        errdefer alloc.free(eigenvalues_k);
        @memcpy(eigenvalues_k, eig_k.values[0..n_occ_k]);

        const wavefunctions_k = try alloc.alloc([]math.Complex, n_occ_k);
        errdefer {
            for (wavefunctions_k[0..built]) |w| alloc.free(w);
            alloc.free(wavefunctions_k);
        }
        const wavefunctions_k_const = try alloc.alloc([]const math.Complex, n_occ_k);
        errdefer alloc.free(wavefunctions_k_const);

        for (0..n_occ_k) |n| {
            wavefunctions_k[n] = try alloc.alloc(math.Complex, n_pw_k);
            @memcpy(wavefunctions_k[n], eig_k.vectors[n * n_pw_k .. (n + 1) * n_pw_k]);
            wavefunctions_k_const[n] = wavefunctions_k[n];
        }

        kgs_data[ik] = KPointGsData{
            .k_frac = k_frac,
            .k_cart = k_cart,
            .weight = wtk,
            .n_occ = n_occ_k,
            .n_pw_k = n_pw_k,
            .basis_k = basis_k,
            .map_k = map_k,
            .apply_ctx_k = apply_ctx_k,
            .eigenvalues_k = eigenvalues_k,
            .wavefunctions_k = wavefunctions_k,
            .wavefunctions_k_const = wavefunctions_k_const,
        };
        built = ik + 1;

        if (ik == 0 or (ik + 1) % 16 == 0 or ik + 1 == n_kpts) {
            logDfpt("dfpt_band: prepared k-point {d}/{d}\n", .{ ik + 1, n_kpts });
        }
    }

    return kgs_data;
}

/// Prepare full-BZ k-point ground-state data using IBZ expansion.
/// Solves eigenvalue problem only at IBZ k-points, then rotates wavefunctions
/// to the full BZ using symmetry operations. This ensures ε tensor isotropy
/// for cubic systems and reduces computation by the symmetry factor.
pub fn prepareFullBZKpointsFromIBZ(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    gs: *const GroundState,
    local_r: []const f64,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
) ![]KPointGsData {
    const kmesh = cfg.scf.kmesh;
    const shift = math.Vec3{
        .x = cfg.scf.kmesh_shift[0],
        .y = cfg.scf.kmesh_shift[1],
        .z = cfg.scf.kmesh_shift[2],
    };
    const total = kmesh[0] * kmesh[1] * kmesh[2];

    // Get symmetry operations
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    defer alloc.free(symops);
    logDfpt("dfpt_ibz: {d} symmetry operations\n", .{symops.len});

    // Filter symmetry ops compatible with k-mesh
    const kmesh_ops = try reduction.filterSymOpsForKmesh(alloc, symops, kmesh, shift, 1e-8);
    defer alloc.free(kmesh_ops);

    // Get IBZ→full BZ mapping
    var mapping = try reduction.reduceKmeshWithMapping(alloc, kmesh, shift, kmesh_ops, recip, true);
    defer mapping.deinit(alloc);
    const n_ibz = mapping.ibz_kpoints.len;
    logDfpt("dfpt_ibz: {d} IBZ k-points -> {d} full BZ k-points\n", .{ n_ibz, total });

    // Generate full BZ mesh for reference (to get k_frac/k_cart for each full point)
    const full_kpts = try mesh_mod.generateKmesh(alloc, kmesh, recip, shift);
    defer alloc.free(full_kpts);

    // Solve eigenvalue problem at each IBZ k-point
    const ibz_data = try alloc.alloc(IbzKData, n_ibz);
    var ibz_built: usize = 0;
    defer {
        for (0..ibz_built) |i| ibz_data[i].deinit(alloc);
        alloc.free(ibz_data);
    }

    for (mapping.ibz_kpoints, 0..) |kp, i_ibz| {
        const k_cart = kp.k_cart;
        var basis_k = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, k_cart);
        errdefer basis_k.deinit(alloc);
        const n_pw_k = basis_k.gvecs.len;

        // Build ApplyContext for H_k
        const apply_ctx_k = try alloc.create(scf_mod.ApplyContext);
        errdefer alloc.destroy(apply_ctx_k);
        apply_ctx_k.* = try scf_mod.ApplyContext.init(
            alloc,
                io,
            grid,
            @constCast(basis_k.gvecs),
            local_r,
            null,
            species,
            atoms,
            1.0 / volume,
            true,
            null,
            null,
            cfg.scf.fft_backend,
        );
        errdefer apply_ctx_k.deinit(alloc);

        // Solve eigenvalue problem
        const diag_k = try alloc.alloc(f64, n_pw_k);
        defer alloc.free(diag_k);
        for (basis_k.gvecs, 0..) |g, gi| {
            diag_k[gi] = g.kinetic;
        }
        const n_occ_k = gs.n_occ;
        const nbands_k = @max(n_occ_k + 2, @as(usize, 8));
        const op_k = iterative.Operator{
            .n = n_pw_k,
            .ctx = @ptrCast(apply_ctx_k),
            .apply = &scf_mod.applyHamiltonian,
            .apply_batch = &scf_mod.applyHamiltonianBatched,
        };
        var eig_k = try iterative.hermitianEigenDecompIterative(
            alloc,
            cfg.linalg_backend,
            op_k,
            diag_k,
            nbands_k,
            .{ .max_iter = 100, .tol = 1e-8, .init_diagonal = true },
        );
        defer eig_k.deinit(alloc);

        const eigenvalues_k = try alloc.alloc(f64, n_occ_k);
        errdefer alloc.free(eigenvalues_k);
        @memcpy(eigenvalues_k, eig_k.values[0..n_occ_k]);

        const wavefunctions_k = try alloc.alloc([]math.Complex, n_occ_k);
        var wf_built: usize = 0;
        errdefer {
            for (0..wf_built) |b| alloc.free(wavefunctions_k[b]);
            alloc.free(wavefunctions_k);
        }
        const wavefunctions_k_const = try alloc.alloc([]const math.Complex, n_occ_k);
        errdefer alloc.free(wavefunctions_k_const);

        for (0..n_occ_k) |n| {
            wavefunctions_k[n] = try alloc.alloc(math.Complex, n_pw_k);
            @memcpy(wavefunctions_k[n], eig_k.vectors[n * n_pw_k .. (n + 1) * n_pw_k]);
            wavefunctions_k_const[n] = wavefunctions_k[n];
            wf_built = n + 1;
        }

        // Clean up ApplyContext (we'll build new ones for full BZ points)
        apply_ctx_k.deinit(alloc);
        alloc.destroy(apply_ctx_k);

        ibz_data[i_ibz] = IbzKData{
            .basis = basis_k,
            .eigenvalues = eigenvalues_k,
            .wavefunctions = wavefunctions_k,
            .wavefunctions_const = wavefunctions_k_const,
            .n_occ = n_occ_k,
        };
        ibz_built = i_ibz + 1;

        logDfpt("dfpt_ibz: solved IBZ k-point {d}/{d} (n_pw={d})\n", .{ i_ibz + 1, n_ibz, n_pw_k });
    }

    // Now expand to full BZ
    var kgs_data = try alloc.alloc(KPointGsData, total);
    var full_built: usize = 0;
    errdefer {
        for (0..full_built) |i| kgs_data[i].deinit(alloc);
        alloc.free(kgs_data);
    }

    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);

    for (0..total) |i_full| {
        const i_ibz = mapping.full_to_ibz[i_full];
        const ibz = &ibz_data[i_ibz];

        const sk_frac = full_kpts[i_full].k_frac;
        const sk_cart = full_kpts[i_full].k_cart;
        const wtk = 1.0 / @as(f64, @floatFromInt(total));

        // Find the correct symop by checking k_rot * k_ibz_frac ≡ k_full_frac (mod 1)
        // The grid-based mapping in reduceKmeshWithMapping may assign wrong symops
        // for non-orthogonal reciprocal lattices (e.g., FCC).
        const found = findSymopForKpoint(symops, mapping.ibz_kpoints[i_ibz].k_frac, sk_frac, 1e-6);
        const symop = found.symop;
        const time_reversed = found.time_reversed;
        const ibz_frac = mapping.ibz_kpoints[i_ibz].k_frac;
        const sk_unwrapped = symop.k_rot.mulVec(ibz_frac);
        const sign_i: i32 = if (time_reversed) -1 else 1;

        // Build target PW basis by rotating IBZ basis G-vectors.
        // G_target = sign * (k_rot * G_ibz) + delta_hkl
        // This guarantees zero null mappings (exact 1:1 correspondence).
        const sign_f: f64 = if (time_reversed) -1.0 else 1.0;
        const delta_hkl = [3]i32{
            @as(i32, @intFromFloat(std.math.round(sign_f * sk_unwrapped.x - sk_frac.x))),
            @as(i32, @intFromFloat(std.math.round(sign_f * sk_unwrapped.y - sk_frac.y))),
            @as(i32, @intFromFloat(std.math.round(sign_f * sk_unwrapped.z - sk_frac.z))),
        };

        const n_pw_ibz = ibz.basis.gvecs.len;
        const gvecs_sk = try alloc.alloc(plane_wave.GVector, n_pw_ibz);
        errdefer alloc.free(gvecs_sk);

        for (ibz.basis.gvecs, 0..) |gv, gi| {
            const h0 = gv.h;
            const k0 = gv.k;
            const l0 = gv.l;
            const h1 = sign_i * (symop.k_rot.m[0][0] * h0 + symop.k_rot.m[0][1] * k0 + symop.k_rot.m[0][2] * l0) + delta_hkl[0];
            const k1 = sign_i * (symop.k_rot.m[1][0] * h0 + symop.k_rot.m[1][1] * k0 + symop.k_rot.m[1][2] * l0) + delta_hkl[1];
            const l1 = sign_i * (symop.k_rot.m[2][0] * h0 + symop.k_rot.m[2][1] * k0 + symop.k_rot.m[2][2] * l0) + delta_hkl[2];
            const g_cart = math.Vec3.add(
                math.Vec3.add(
                    math.Vec3.scale(b1, @as(f64, @floatFromInt(h1))),
                    math.Vec3.scale(b2, @as(f64, @floatFromInt(k1))),
                ),
                math.Vec3.scale(b3, @as(f64, @floatFromInt(l1))),
            );
            const kpg = math.Vec3.add(sk_cart, g_cart);
            gvecs_sk[gi] = .{
                .h = h1,
                .k = k1,
                .l = l1,
                .cart = g_cart,
                .kpg = kpg,
                .kinetic = math.Vec3.dot(kpg, kpg),
            };
        }

        var basis_sk = plane_wave.Basis{ .gvecs = gvecs_sk };
        errdefer basis_sk.deinit(alloc);
        const n_pw_sk = n_pw_ibz;

        // Build PwGridMap for S*k
        var map_sk = try scf_mod.PwGridMap.init(alloc, @constCast(basis_sk.gvecs), grid);
        errdefer map_sk.deinit(alloc);

        // Build ApplyContext for H_{S*k}
        const apply_ctx_sk = try alloc.create(scf_mod.ApplyContext);
        errdefer alloc.destroy(apply_ctx_sk);
        apply_ctx_sk.* = try scf_mod.ApplyContext.init(
            alloc,
                io,
            grid,
            @constCast(basis_sk.gvecs),
            local_r,
            null,
            species,
            atoms,
            1.0 / volume,
            true,
            null,
            null,
            cfg.scf.fft_backend,
        );
        errdefer apply_ctx_sk.deinit(alloc);

        // Rotate wavefunctions: mapping is trivial (identity permutation)
        // since target basis was built from IBZ basis in the same order.
        const rot_result = try wfn_rot.rotateWavefunctionsInPlace(
            alloc,
            ibz.wavefunctions_const,
            ibz.basis,
            symop,
            sk_unwrapped,
            time_reversed,
        );
        errdefer {
            for (rot_result.wfn) |w| alloc.free(w);
            alloc.free(rot_result.wfn);
            alloc.free(rot_result.wfn_const);
        }

        // Copy eigenvalues (same as IBZ point)
        const eigenvalues_sk = try alloc.alloc(f64, ibz.n_occ);
        errdefer alloc.free(eigenvalues_sk);
        @memcpy(eigenvalues_sk, ibz.eigenvalues);

        kgs_data[i_full] = KPointGsData{
            .k_frac = sk_frac,
            .k_cart = sk_cart,
            .weight = wtk,
            .n_occ = ibz.n_occ,
            .n_pw_k = n_pw_sk,
            .basis_k = basis_sk,
            .map_k = map_sk,
            .apply_ctx_k = apply_ctx_sk,
            .eigenvalues_k = eigenvalues_sk,
            .wavefunctions_k = rot_result.wfn,
            .wavefunctions_k_const = rot_result.wfn_const,
        };
        full_built = i_full + 1;

        if (i_full == 0 or (i_full + 1) % 16 == 0 or i_full + 1 == total) {
            logDfpt("dfpt_ibz: expanded k-point {d}/{d}\n", .{ i_full + 1, total });
        }
    }

    return kgs_data;
}

const SymopMatch = struct {
    symop: symmetry.SymOp,
    time_reversed: bool,
};

/// Find the symmetry operation that maps k_ibz to k_full: k_rot * k_ibz ≡ ±k_full (mod 1).
/// This is necessary because the grid-based mapIndex in reduceKmesh may assign incorrect
/// symops for non-orthogonal reciprocal lattices.
fn findSymopForKpoint(
    ops: []const symmetry.SymOp,
    k_ibz_frac: math.Vec3,
    k_full_frac: math.Vec3,
    tol: f64,
) SymopMatch {
    // Try without time reversal first
    for (ops) |op| {
        const sk = op.k_rot.mulVec(k_ibz_frac);
        const dx = sk.x - k_full_frac.x;
        const dy = sk.y - k_full_frac.y;
        const dz = sk.z - k_full_frac.z;
        if (@abs(dx - std.math.round(dx)) < tol and
            @abs(dy - std.math.round(dy)) < tol and
            @abs(dz - std.math.round(dz)) < tol)
        {
            return .{ .symop = op, .time_reversed = false };
        }
    }
    // Try with time reversal: -k_rot * k_ibz ≡ k_full (mod 1)
    for (ops) |op| {
        const sk = op.k_rot.mulVec(k_ibz_frac);
        const dx = -sk.x - k_full_frac.x;
        const dy = -sk.y - k_full_frac.y;
        const dz = -sk.z - k_full_frac.z;
        if (@abs(dx - std.math.round(dx)) < tol and
            @abs(dy - std.math.round(dy)) < tol and
            @abs(dz - std.math.round(dz)) < tol)
        {
            return .{ .symop = op, .time_reversed = true };
        }
    }
    // Fallback: identity (shouldn't happen if IBZ reduction is correct)
    return .{ .symop = ops[0], .time_reversed = false };
}

const IbzKData = struct {
    basis: plane_wave.Basis,
    eigenvalues: []f64,
    wavefunctions: [][]math.Complex,
    wavefunctions_const: [][]const math.Complex,
    n_occ: usize,

    fn deinit(self: *IbzKData, alloc: std.mem.Allocator) void {
        for (self.wavefunctions) |w| alloc.free(w);
        alloc.free(self.wavefunctions);
        alloc.free(self.wavefunctions_const);
        alloc.free(self.eigenvalues);
        self.basis.deinit(alloc);
    }
};

/// Build KPointDfptData array from precomputed ground-state data for a given q-point.
/// The k-point data (basis, wavefunctions, eigenvalues) is shared from KPointGsData.
/// Only the k+q data (basis_kq, occ_kq, etc.) is newly allocated per q-point.
fn buildKPointDfptDataFromGS(
    alloc: std.mem.Allocator,
    io: std.Io,
    kgs: []const KPointGsData,
    q_cart: math.Vec3,
    q_norm: f64,
    cfg: config_mod.Config,
    local_r: []const f64,
    species: []hamiltonian.SpeciesEntry,
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

// =====================================================================
// Multi-k-point data for DFPT
// =====================================================================

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

// =====================================================================
// Cross-basis operations for q≠0
// =====================================================================

/// Apply V^(1)(r)|ψ⟩ with complex V^(1) and cross-basis (k → k+q).
/// Scatters k-basis coefficients to grid, IFFTs, multiplies by V^(1)(r),
/// FFTs back, and gathers to k+q-basis.
fn applyV1PsiQ(
    alloc: std.mem.Allocator,
    grid: Grid,
    map_k: *const scf_mod.PwGridMap,
    map_kq: *const scf_mod.PwGridMap,
    v1_r_complex: []const math.Complex,
    psi_k: []const math.Complex,
    n_pw_k: usize,
    n_pw_kq: usize,
) ![]math.Complex {
    const total = grid.count();

    // Scatter k-basis PW coefficients to full grid
    const work_g = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g);
    @memset(work_g, math.complex.init(0.0, 0.0));
    map_k.scatter(psi_k, work_g);

    // IFFT to real space
    const work_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r);
    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work_g, work_r, null);

    // Multiply by V^(1)(r) (complex × complex)
    for (0..total) |i| {
        work_r[i] = math.complex.mul(work_r[i], v1_r_complex[i]);
    }

    // FFT back to reciprocal space
    const work_g_out = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g_out);
    try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, work_r, work_g_out, null);

    // Gather to k+q-basis
    const result = try alloc.alloc(math.Complex, n_pw_kq);
    map_kq.gather(work_g_out, result);
    _ = n_pw_k;

    return result;
}

/// Cached variant of applyV1PsiQ that uses pre-computed ψ^(0)(r) in real space.
/// Skips the scatter + IFFT step for ψ^(0), saving one FFT per band per SCF iteration.
pub fn applyV1PsiQCached(
    alloc: std.mem.Allocator,
    grid: Grid,
    map_kq: *const scf_mod.PwGridMap,
    v1_r_complex: []const math.Complex,
    psi0_r: []const math.Complex,
    n_pw_kq: usize,
) ![]math.Complex {
    const total = grid.count();

    // Multiply ψ^(0)(r) by V^(1)(r) (complex × complex)
    const work_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r);
    for (0..total) |i| {
        work_r[i] = math.complex.mul(psi0_r[i], v1_r_complex[i]);
    }

    // FFT back to reciprocal space
    const work_g_out = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g_out);
    try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, work_r, work_g_out, null);

    // Gather to k+q-basis
    const result = try alloc.alloc(math.Complex, n_pw_kq);
    map_kq.gather(work_g_out, result);

    return result;
}

/// Compute first-order density response for q≠0 from cross-basis wavefunctions.
/// ρ^(1)(r) = (4 × wtk / Ω) × Σ_n ψ^(0)*_{n,k}(r) × ψ^(1)_{n,k,q}(r)
/// Weight = 4×wtk/Ω: factor 2 from spin (occ=2), factor 2 from c.c. (2n+1 theorem),
/// wtk from k-point weight, 1/Ω from normalization.
/// Matches ABINIT's dfpt_mkrho convention:
/// weight = two * occ * wtk / ucvol.
/// For single k-point (Γ only), wtk=1.0 gives the original 4/Ω.
fn computeRho1Q(
    alloc: std.mem.Allocator,
    grid: Grid,
    map_k: *const scf_mod.PwGridMap,
    map_kq: *const scf_mod.PwGridMap,
    psi0_k: []const []const math.Complex,
    psi1_kq: []const []const math.Complex,
    n_occ: usize,
    n_pw_k: usize,
    n_pw_kq: usize,
    wtk: f64,
) ![]math.Complex {
    const total = grid.count();
    // ρ^(1)(r) is complex for q≠0
    const rho1_r = try alloc.alloc(math.Complex, total);
    @memset(rho1_r, math.complex.init(0.0, 0.0));

    const work_g0 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g0);
    const work_r0 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r0);
    const work_g1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g1);
    const work_r1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r1);

    // Weight: 4×wtk/Ω = 2(spin/occ) × 2(c.c./2n+1) × wtk(k-point) × (1/Ω)(normalization)
    // Matches ABINIT dfpt_mkrho: weight = two * occ_k * wtk_k / ucvol
    const weight = 4.0 * wtk / grid.volume;

    for (0..n_occ) |n| {
        // ψ^(0)(r) via IFFT using k-basis map
        @memset(work_g0, math.complex.init(0.0, 0.0));
        map_k.scatter(psi0_k[n], work_g0);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work_g0, work_r0, null);

        // ψ^(1)(r) via IFFT using k+q-basis map
        @memset(work_g1, math.complex.init(0.0, 0.0));
        map_kq.scatter(psi1_kq[n], work_g1);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work_g1, work_r1, null);

        // ρ^(1)(r) += (4×wtk/Ω) × ψ^(0)*(r) × ψ^(1)(r)  [complex]
        for (0..total) |i| {
            const conj0 = math.complex.conj(work_r0[i]);
            const prod = math.complex.mul(conj0, work_r1[i]);
            rho1_r[i] = math.complex.add(rho1_r[i], math.complex.scale(prod, weight));
        }
    }
    _ = n_pw_k;
    _ = n_pw_kq;

    return rho1_r;
}

/// Cached variant of computeRho1Q that uses pre-computed ψ^(0)(r) in real space.
/// Skips the scatter + IFFT for ψ^(0) each band, saving n_occ FFTs per call.
pub fn computeRho1QCached(
    alloc: std.mem.Allocator,
    grid: Grid,
    map_kq: *const scf_mod.PwGridMap,
    psi0_r_cache: []const []const math.Complex,
    psi1_kq: []const []const math.Complex,
    n_occ: usize,
    wtk: f64,
) ![]math.Complex {
    const total = grid.count();
    const rho1_r = try alloc.alloc(math.Complex, total);
    @memset(rho1_r, math.complex.init(0.0, 0.0));

    const work_g1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_g1);
    const work_r1 = try alloc.alloc(math.Complex, total);
    defer alloc.free(work_r1);

    const weight = 4.0 * wtk / grid.volume;

    for (0..n_occ) |n| {
        // ψ^(0)(r) — use cached version (no IFFT needed)
        const work_r0 = psi0_r_cache[n];

        // ψ^(1)(r) via IFFT using k+q-basis map
        @memset(work_g1, math.complex.init(0.0, 0.0));
        map_kq.scatter(psi1_kq[n], work_g1);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work_g1, work_r1, null);

        // ρ^(1)(r) += (4×wtk/Ω) × ψ^(0)*(r) × ψ^(1)(r)  [complex]
        for (0..total) |i| {
            const conj0 = math.complex.conj(work_r0[i]);
            const prod = math.complex.mul(conj0, work_r1[i]);
            rho1_r[i] = math.complex.add(rho1_r[i], math.complex.scale(prod, weight));
        }
    }

    return rho1_r;
}

/// Compute ρ^(1)(G) from ρ^(1)(r) complex via FFT.
fn complexRealToReciprocal(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho1_r: []math.Complex,
) ![]math.Complex {
    const total = grid.count();
    const out = try alloc.alloc(math.Complex, total);
    try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, rho1_r, out, null);
    return out;
}

// =====================================================================
// Complex dynmat element computations for q≠0
// =====================================================================

/// Compute the electronic contribution to the dynamical matrix element (complex, q≠0).
/// D^elec_{Iα,Jβ} = Σ_G conj(V^(1)_{Iα}(G)) × ρ^(1)_{Jβ}(G) × Ω
pub fn computeElecDynmatElementQ(
    vloc1_g: []const math.Complex,
    rho1_g: []const math.Complex,
    volume: f64,
) math.Complex {
    var sum = math.complex.init(0.0, 0.0);
    for (0..vloc1_g.len) |i| {
        sum = math.complex.add(sum, math.complex.mul(math.complex.conj(vloc1_g[i]), rho1_g[i]));
    }
    return math.complex.scale(sum, volume);
}

/// Compute the nonlocal response contribution to the dynamical matrix for q≠0.
/// D(Iα,Jβ) = 2 × Σ_n ⟨dV_nl_{Iα,q} ψ^(0)_n | δψ_{n,Jβ}⟩
/// Monochromatic convention: +q only. Hermitianization at total level.
pub fn computeNonlocalResponseDynmatQ(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    gvecs_kq: []const plane_wave.GVector,
    apply_ctx_kq: *scf_mod.ApplyContext,
    n_atoms: usize,
) ![]math.Complex {
    const dim = 3 * n_atoms;
    const n_pw_kq = gvecs_kq.len;
    const dyn = try alloc.alloc(math.Complex, dim * dim);
    @memset(dyn, math.complex.init(0.0, 0.0));

    const nl_ctx_k = gs.apply_ctx.nonlocal_ctx orelse return dyn;
    const nl_ctx_kq = apply_ctx_kq.nonlocal_ctx orelse return dyn;

    const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
    defer alloc.free(nl_out);

    // Compute D(Iα,Jβ) = 4 × Σ_n ⟨dV_nl_{Iα,q} ψ^(0)_n | δψ_{n,Jβ}⟩
    // Factor 4 = 2(spin) × 2(d²E/dτ₁dτ₂), matching q=0 convention.
    // All (I,J) pairs computed explicitly — no Hermitianization needed.
    for (0..dim) |i| {
        const ia = i / 3;
        const dir_a = i % 3;
        for (0..gs.n_occ) |n| {
            // Apply dV_nl_{Iα,q} to ψ^(0)_{n,k}: k-basis → k+q-basis
            try perturbation.applyNonlocalPerturbationQ(
                alloc,
                gs.gvecs,
                gvecs_kq,
                gs.atoms,
                nl_ctx_k,
                nl_ctx_kq,
                ia,
                dir_a,
                1.0 / gs.grid.volume,
                gs.wavefunctions[n],
                nl_out,
            );

            // Inner product with each δψ_{Jβ} in k+q space
            for (0..dim) |j| {
                var ip = math.complex.init(0.0, 0.0);
                for (0..n_pw_kq) |g| {
                    ip = math.complex.add(ip, math.complex.mul(
                        math.complex.conj(nl_out[g]),
                        pert_results[j].psi1[n][g],
                    ));
                }
                // Factor 4 = 2(spin) × 2(d²E/dτ₁dτ₂ = 2·E^(τ₁τ₂)),
                // matching ABINIT's d2nl convention: wtk × occ × two = 1 × 2 × 2 = 4.
                // All (I,J) pairs are computed explicitly, no Hermitianization needed.
                dyn[i * dim + j] = math.complex.add(dyn[i * dim + j], math.complex.scale(ip, 4.0));
            }
        }
    }

    return dyn;
}

/// Compute NLCC cross-term for q≠0.
/// D_NLCC_cross(Iα,Jβ) = ∫ V_xc^(1)[ρ^(1)_core,I](r) × conj(ρ^(1)_total,J(r)) dr
///
/// For LDA this reduces to ∫ f_xc × ρ^(1)_core,I × ρ^(1)_total,J dr.
/// For GGA, V_xc^(1) includes gradient-dependent terms.
/// For q≠0, ρ^(1)(r) is complex, so result is complex.
pub fn computeNlccCrossDynmatQ(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: GroundState,
    rho1_val_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    n_atoms: usize,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    const dim = 3 * n_atoms;
    const total = grid.count();
    const dyn = try alloc.alloc(math.Complex, dim * dim);
    @memset(dyn, math.complex.init(0.0, 0.0));

    const dv = grid.volume / @as(f64, @floatFromInt(total));

    for (0..dim) |i| {
        // Get ρ^(1)_core,Iα(r) in complex form
        const rho1_core_i_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_core_i_copy);
        @memcpy(rho1_core_i_copy, rho1_core_gs[i]);
        const rho1_core_i_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_core_i_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_i_copy, rho1_core_i_r, null);

        // Build V_xc^(1)[ρ^(1)_core,I] using GGA-aware kernel
        const vxc1_core_i = try perturbation.buildXcPerturbationFullComplex(alloc, gs, rho1_core_i_r);
        defer alloc.free(vxc1_core_i);

        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            // Get ρ^(1)_val,Jβ(r)
            const rho1_val_g_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_val_g_copy);
            @memcpy(rho1_val_g_copy, rho1_val_gs[j]);
            const work_r_j = try alloc.alloc(math.Complex, total);
            defer alloc.free(work_r_j);
            try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_val_g_copy, work_r_j, null);

            // Get ρ^(1)_core,Jβ(r)
            const rho1_core_j_copy = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_core_j_copy);
            @memcpy(rho1_core_j_copy, rho1_core_gs[j]);
            const rho1_core_j_r = try alloc.alloc(math.Complex, total);
            defer alloc.free(rho1_core_j_r);
            try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_j_copy, rho1_core_j_r, null);

            // D(I,J) = ∫ conj(V_xc^(1)[ρ^(1)_core,I]) × ρ^(1)_total,J dr
            var sum = math.complex.init(0.0, 0.0);
            for (0..total) |r| {
                const rho1_total_j = math.complex.add(work_r_j[r], rho1_core_j_r[r]);
                const prod = math.complex.mul(math.complex.conj(vxc1_core_i[r]), rho1_total_j);
                sum = math.complex.add(sum, prod);
            }
            dyn[i * dim + j] = math.complex.scale(sum, dv);
        }
    }

    return dyn;
}

// =====================================================================
// DFPT SCF solver for q≠0
// =====================================================================

/// Solve DFPT perturbation at q≠0 for a single perturbation (atom, direction).
/// Uses **potential mixing** (mix V^(1)_SCF, not ρ^(1)) for stable convergence.
/// Density mixing is unstable at finite q because V_H(G=0) = 8π/|q|² diverges.
pub fn solvePerturbationQ(
    alloc: std.mem.Allocator,
    gs: GroundState,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
    q_cart: math.Vec3,
    gvecs_kq: []const plane_wave.GVector,
    apply_ctx_kq: *scf_mod.ApplyContext,
    map_kq: *const scf_mod.PwGridMap,
    occ_kq: []const []const math.Complex,
    n_occ_kq: usize,
) !PerturbationResult {
    const n_pw_k = gs.gvecs.len;
    const n_pw_kq = gvecs_kq.len;
    const total = gs.grid.count();
    const n_occ = gs.n_occ;
    const grid = gs.grid;

    // Build V_ext^(1)_q(G) for this perturbation (bare, fixed)
    const vloc1_g = try perturbation.buildLocalPerturbationQ(
        alloc,
        grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        q_cart,
        gs.ff_tables,
    );
    defer alloc.free(vloc1_g);

    // Debug: verify vloc1_g depends on direction
    {
        var vloc_norm: f64 = 0.0;
        var vloc_sum_r: f64 = 0.0;
        var vloc_sum_i: f64 = 0.0;
        for (vloc1_g) |c| {
            vloc_norm += c.r * c.r + c.i * c.i;
            vloc_sum_r += c.r;
            vloc_sum_i += c.i;
        }
        logDfpt("dfptQ_vloc1: atom={d} dir={d} |vloc1_g|={e:.6} sum=({e:.6},{e:.6})\n", .{ atom_index, direction, @sqrt(vloc_norm), vloc_sum_r, vloc_sum_i });
        // Print first few G-point values
        const nshow = @min(vloc1_g.len, 5);
        for (0..nshow) |gi| {
            logDfpt("  vloc1_g[{d}]=({e:.8},{e:.8})\n", .{ gi, vloc1_g[gi].r, vloc1_g[gi].i });
        }
        // Print G=0 value (index where h=k=l=0)
        {
            const g0_h: usize = @intCast(-grid.min_h);
            const g0_k: usize = @intCast(-grid.min_k);
            const g0_l: usize = @intCast(-grid.min_l);
            const g0_idx = g0_l * grid.ny * grid.nx + g0_k * grid.nx + g0_h;
            logDfpt("  vloc1_g[G=0, idx={d}]=({e:.8},{e:.8}) grid=({d},{d},{d}) min=({d},{d},{d})\n", .{
                g0_idx,     vloc1_g[g0_idx].r, vloc1_g[g0_idx].i,
                grid.nx,    grid.ny,           grid.nz,
                grid.min_h, grid.min_k,        grid.min_l,
            });
        }
    }

    // Build ρ^(1)_core,q(G) (fixed, NLCC)
    const rho1_core_g = try perturbation.buildCorePerturbationQ(
        alloc,
        grid,
        gs.atoms[atom_index],
        gs.species,
        direction,
        q_cart,
        gs.rho_core_tables,
    );
    defer alloc.free(rho1_core_g);

    // IFFT ρ^(1)_core,q → real space (complex)
    const rho1_core_g_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_g_copy);
    @memcpy(rho1_core_g_copy, rho1_core_g);
    const rho1_core_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_r);
    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_g_copy, rho1_core_r, null);

    // Allocate first-order wavefunctions in k+q basis
    var psi1 = try alloc.alloc([]math.Complex, n_occ);
    for (0..n_occ) |n| {
        psi1[n] = try alloc.alloc(math.Complex, n_pw_kq);
        @memset(psi1[n], math.complex.init(0.0, 0.0));
    }

    // Map for k-basis
    const map_k_ptr: *const scf_mod.PwGridMap = &gs.apply_ctx.map;

    // Initialize V_SCF(G) = V_loc^(1)(G) [bare perturbation, no screening]
    var v_scf_g = try alloc.alloc(math.Complex, total);
    @memcpy(v_scf_g, vloc1_g);

    // Pre-compute ψ^(0)(r) for all occupied bands (invariant during SCF)
    const psi0_r_cache = try alloc.alloc([]math.Complex, n_occ);
    defer {
        for (psi0_r_cache) |band| alloc.free(band);
        alloc.free(psi0_r_cache);
    }
    for (0..n_occ) |n| {
        psi0_r_cache[n] = try alloc.alloc(math.Complex, total);
        const work = try alloc.alloc(math.Complex, total);
        defer alloc.free(work);
        @memset(work, math.complex.init(0.0, 0.0));
        map_k_ptr.scatter(gs.wavefunctions[n], work);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work, psi0_r_cache[n], null);
    }
    const psi0_r_const: []const []const math.Complex = @ptrCast(psi0_r_cache);

    // Pulay mixer for potential mixing
    var pulay = scf_mod.ComplexPulayMixer.init(alloc, cfg.pulay_history);
    defer pulay.deinit();

    // Pulay restart state
    var best_vresid: f64 = std.math.inf(f64);
    var best_v_scf: ?[]math.Complex = null;
    defer if (best_v_scf) |v| alloc.free(v);
    var pulay_active_since: usize = cfg.pulay_start;
    const restart_factor: f64 = 5.0;
    var force_converge: bool = false;

    // DFPT SCF loop — potential mixing
    var iter: usize = 0;
    while (iter < cfg.scf_max_iter) : (iter += 1) {
        // IFFT V_SCF(G) → V_SCF(r) [complex]
        const v_scf_g_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_scf_g_copy);
        @memcpy(v_scf_g_copy, v_scf_g);
        const v_scf_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_scf_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, v_scf_g_copy, v_scf_r, null);

        // Debug: on first iteration, print some v_scf values
        if (iter == 0) {
            logDfpt("dfptQ_vscf: atom={d} dir={d} iter=0 v_scf_r[0]=({e:.8},{e:.8}) v_scf_r[1]=({e:.8},{e:.8}) v_scf_r[100]=({e:.8},{e:.8})\n", .{
                atom_index,                                  direction,
                v_scf_r[0].r,                                v_scf_r[0].i,
                v_scf_r[1].r,                                v_scf_r[1].i,
                v_scf_r[@min(@as(usize, 100), total - 1)].r, v_scf_r[@min(@as(usize, 100), total - 1)].i,
            });
            logDfpt("dfptQ_vscf: v_scf_g[0]=({e:.8},{e:.8}) v_scf_g[1]=({e:.8},{e:.8}) v_scf_g[2]=({e:.8},{e:.8})\n", .{
                v_scf_g[0].r, v_scf_g[0].i,
                v_scf_g[1].r, v_scf_g[1].i,
                v_scf_g[2].r, v_scf_g[2].i,
            });
        }

        // Nonlocal contexts for V_nl^(1) (cross-basis: k → k+q)
        const nl_ctx_k_opt = gs.apply_ctx.nonlocal_ctx;
        const nl_ctx_kq_opt = apply_ctx_kq.nonlocal_ctx;

        // Solve Sternheimer for each occupied band
        for (0..n_occ) |n| {
            // RHS: -P_c^{k+q} × H^(1)|ψ^(0)_{n,k}⟩
            // H^(1)|ψ⟩ = V_SCF(r)|ψ(r)⟩ + V_nl^(1)|ψ⟩
            const rhs = try applyV1PsiQCached(alloc, grid, map_kq, v_scf_r, psi0_r_cache[n], n_pw_kq);
            defer alloc.free(rhs);

            // Add nonlocal perturbation: V_nl^(1)_{q}|ψ_k⟩ (cross-basis: k → k+q)
            if (nl_ctx_k_opt != null and nl_ctx_kq_opt != null) {
                const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
                defer alloc.free(nl_out);
                try perturbation.applyNonlocalPerturbationQ(
                    alloc,
                    gs.gvecs,
                    gvecs_kq,
                    gs.atoms,
                    nl_ctx_k_opt.?,
                    nl_ctx_kq_opt.?,
                    atom_index,
                    direction,
                    1.0 / grid.volume,
                    gs.wavefunctions[n],
                    nl_out,
                );
                for (0..n_pw_kq) |g| {
                    rhs[g] = math.complex.add(rhs[g], nl_out[g]);
                }
            }

            // Negate: rhs = -H^(1)|ψ⟩
            for (0..n_pw_kq) |g| {
                rhs[g] = math.complex.scale(rhs[g], -1.0);
            }

            // Project onto conduction band in k+q space
            sternheimer.projectConduction(rhs, occ_kq, n_occ_kq);

            // Solve: (H_{k+q} - ε_{n,k} + α·P_v^{k+q})|ψ^(1)⟩ = rhs
            const result = try sternheimer.solve(
                alloc,
                apply_ctx_kq,
                rhs,
                gs.eigenvalues[n],
                occ_kq,
                n_occ_kq,
                gvecs_kq,
                .{
                    .tol = cfg.sternheimer_tol,
                    .max_iter = cfg.sternheimer_max_iter,
                    .alpha_shift = cfg.alpha_shift,
                },
            );

            alloc.free(psi1[n]);
            psi1[n] = result.psi1;
        }

        // Compute ρ^(1)_{+q}(r) from cross-basis wavefunctions [weight=4×wtk/Ω]
        const psi1_const_view = try alloc.alloc([]const math.Complex, n_occ);
        defer alloc.free(psi1_const_view);
        for (0..n_occ) |n2| psi1_const_view[n2] = psi1[n2];
        const rho1_r = try computeRho1QCached(
            alloc,
            grid,
            map_kq,
            psi0_r_const,
            psi1_const_view,
            n_occ,
            1.0, // wtk=1.0 for single k-point backward compat
        );
        defer alloc.free(rho1_r);

        // FFT ρ^(1)(r) → ρ^(1)(G)
        const rho1_g = try complexRealToReciprocal(alloc, grid, rho1_r);
        defer alloc.free(rho1_g);

        // Diagnostic: ρ^(1) norm
        {
            var rho_norm: f64 = 0.0;
            for (0..total) |di| {
                rho_norm += rho1_g[di].r * rho1_g[di].r + rho1_g[di].i * rho1_g[di].i;
            }
            rho_norm = @sqrt(rho_norm);
            // D_elec from bare V^(1) and current ρ^(1)
            const d_elec_diag = computeElecDynmatElementQ(vloc1_g, rho1_g, grid.volume);
            logDfpt("dfptQ_diag: iter={d} |rho1|={e:.6} D_elec_bare(0)=({e:.6},{e:.6})\n", .{ iter, rho_norm, d_elec_diag.r, d_elec_diag.i });
        }

        // Build V_out(G) = V_loc^(1) + V_H^(1)[ρ] + V_xc^(1)[ρ]
        const vh1_g = try perturbation.buildHartreePerturbationQ(alloc, grid, rho1_g, q_cart);
        defer alloc.free(vh1_g);

        // V_xc^(1)(r) = f_xc(r) × ρ^(1)_total(r)  where ρ_total = ρ_val + ρ_core
        const rho1_g_copy2 = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_g_copy2);
        @memcpy(rho1_g_copy2, rho1_g);
        const rho1_val_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_val_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_g_copy2, rho1_val_r, null);

        const rho1_total_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_total_r);
        for (0..total) |i| {
            rho1_total_r[i] = math.complex.add(rho1_val_r[i], rho1_core_r[i]);
        }

        const vxc1_r = try perturbation.buildXcPerturbationFullComplex(alloc, gs, rho1_total_r);
        defer alloc.free(vxc1_r);

        // FFT V_xc^(1)(r) → V_xc^(1)(G)
        const vxc1_r_fft = try alloc.alloc(math.Complex, total);
        defer alloc.free(vxc1_r_fft);
        @memcpy(vxc1_r_fft, vxc1_r);
        const vxc1_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(vxc1_g);
        try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, vxc1_r_fft, vxc1_g, null);

        // V_out(G) = V_loc + V_H + V_xc
        const v_out_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_out_g);
        for (0..total) |i| {
            v_out_g[i] = math.complex.add(
                math.complex.add(vloc1_g[i], vh1_g[i]),
                vxc1_g[i],
            );
        }

        // Compute residual = V_out - V_SCF
        var residual_norm: f64 = 0.0;
        const residual = try alloc.alloc(math.Complex, total);
        for (0..total) |i| {
            residual[i] = math.complex.sub(v_out_g[i], v_scf_g[i]);
            residual_norm += residual[i].r * residual[i].r + residual[i].i * residual[i].i;
        }
        residual_norm = @sqrt(residual_norm);

        logDfpt("dfptQ_scf: iter={d} vresid={e:.6}\n", .{ iter, residual_norm });

        if (residual_norm < cfg.scf_tol or (force_converge and residual_norm < 10.0 * cfg.scf_tol)) {
            alloc.free(residual);
            logDfpt("dfptQ_scf: converged at iter={d} vresid={e:.6}\n", .{ iter, residual_norm });
            break;
        }

        // Track best residual and save corresponding V_SCF
        if (residual_norm < best_vresid) {
            best_vresid = residual_norm;
            if (best_v_scf == null) best_v_scf = try alloc.alloc(math.Complex, total);
            @memcpy(best_v_scf.?, v_scf_g);
        }

        // Pulay restart: if residual exceeds restart_factor × best, reset and restore
        if (iter >= pulay_active_since and residual_norm > restart_factor * best_vresid and best_vresid < 1.0) {
            if (best_v_scf) |v| @memcpy(v_scf_g, v);
            // If best is near convergence, force accept on next iteration
            if (best_vresid < 10.0 * cfg.scf_tol) {
                force_converge = true;
                logDfpt("dfptQ_scf: Pulay restart (near-converged) at iter={d} vresid={e:.6} best={e:.6}\n", .{ iter, residual_norm, best_vresid });
                alloc.free(residual);
                continue;
            }
            pulay.reset();
            pulay_active_since = iter + 1 + cfg.pulay_start;
            logDfpt("dfptQ_scf: Pulay restart at iter={d} vresid={e:.6} best={e:.6}\n", .{ iter, residual_norm, best_vresid });
            alloc.free(residual);
            continue;
        }

        // Mix V_SCF using Pulay (delayed start) or simple linear mixing
        if (cfg.pulay_history > 0 and iter >= pulay_active_since) {
            // Pulay/DIIS: ownership of residual transfers to mixer
            try pulay.mixWithResidual(v_scf_g, residual, cfg.mixing_beta);
        } else {
            // Simple linear mixing: V_SCF += β × residual
            const beta = cfg.mixing_beta;
            for (0..total) |i| {
                v_scf_g[i] = math.complex.add(v_scf_g[i], math.complex.scale(residual[i], beta));
            }
            alloc.free(residual);
        }
    }

    // Compute final ρ^(1)(G) from converged ψ^(1)
    const final_rho1_r = try computeRho1Q(
        alloc,
        grid,
        map_k_ptr,
        map_kq,
        gs.wavefunctions,
        psi1,
        n_occ,
        n_pw_k,
        n_pw_kq,
        1.0, // wtk=1.0 for single k-point backward compat
    );
    const final_rho1_g = try complexRealToReciprocal(alloc, grid, final_rho1_r);
    alloc.free(final_rho1_r);

    alloc.free(v_scf_g);

    return .{
        .rho1_g = final_rho1_g,
        .psi1 = psi1,
    };
}

// =====================================================================
// Multi-k-point DFPT perturbation solver
// =====================================================================


/// Shared data for parallel DFPT k-point processing within one SCF iteration.
const DfptKpointShared = struct {
    kpts: []KPointDfptData,
    v_scf_r: []const math.Complex,
    vloc1_g: []const math.Complex,
    cfg: DfptConfig,
    atom_index: usize,
    direction: usize,
    grid: Grid,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    /// Per-k-point ψ^(1) storage [n_kpts][n_occ][n_pw_kq] — each worker writes to its own ik
    psi1_per_k: [][][]math.Complex,
    /// Thread-local ρ^(1) buffers: rho1_locals[thread_index * total .. (thread_index+1) * total]
    rho1_locals: []math.Complex,
    total: usize,
    /// Pre-computed ψ^(0)_k(r) cache: [n_kpts][n_occ][total]
    psi0_r_cache: []const []const []const math.Complex,

    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

const DfptKpointWorker = struct {
    shared: *DfptKpointShared,
    thread_index: usize,
};

fn setDfptWorkerError(shared: *DfptKpointShared, e: anyerror) void {
    shared.err_mutex.lock();
    defer shared.err_mutex.unlock();
    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

fn dfptKpointWorkerFn(worker: *DfptKpointWorker) void {
    const shared = worker.shared;
    const thread_index = worker.thread_index;
    const total = shared.total;
    const start = thread_index * total;
    const rho1_local = shared.rho1_locals[start .. start + total];

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const ik = shared.next_index.fetchAdd(1, .acq_rel);
        if (ik >= shared.kpts.len) break;

        _ = arena.reset(.retain_capacity);
        const kalloc = arena.allocator();

        processOneKpointDfpt(
            kalloc,
            shared,
            ik,
            rho1_local,
        ) catch |e| {
            setDfptWorkerError(shared, e);
            shared.stop.store(1, .release);
            break;
        };
    }
}

/// Process a single k-point within one DFPT SCF iteration.
/// Solves Sternheimer for each occupied band, updates psi1_per_k[ik],
/// and accumulates ρ^(1) into the provided rho1_local buffer.
fn processOneKpointDfpt(
    alloc: std.mem.Allocator,
    shared: *DfptKpointShared,
    ik: usize,
    rho1_local: []math.Complex,
) !void {
    const kd = &shared.kpts[ik];
    const n_occ = kd.n_occ;
    const n_pw_kq = kd.n_pw_kq;
    const map_kq_ptr: *const scf_mod.PwGridMap = &kd.map_kq;
    const total = shared.total;

    // Nonlocal contexts
    const nl_ctx_k_opt = kd.apply_ctx_k.nonlocal_ctx;
    const nl_ctx_kq_opt = kd.apply_ctx_kq.nonlocal_ctx;

    // ψ^(0)(r) cache for this k-point
    const psi0_r_k = shared.psi0_r_cache[ik];

    // Solve Sternheimer for each occupied band at this k-point
    for (0..n_occ) |n| {
        // RHS: -P_c^{k+q} × H^(1)|ψ^(0)_{n,k}⟩
        const rhs = try applyV1PsiQCached(alloc, shared.grid, map_kq_ptr, shared.v_scf_r, psi0_r_k[n], n_pw_kq);
        defer alloc.free(rhs);

        // Add nonlocal perturbation
        if (nl_ctx_k_opt != null and nl_ctx_kq_opt != null) {
            const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
            defer alloc.free(nl_out);
            try perturbation.applyNonlocalPerturbationQ(
                alloc,
                kd.basis_k.gvecs,
                kd.basis_kq.gvecs,
                shared.atoms,
                nl_ctx_k_opt.?,
                nl_ctx_kq_opt.?,
                shared.atom_index,
                shared.direction,
                1.0 / shared.grid.volume,
                kd.wavefunctions_k_const[n],
                nl_out,
            );
            for (0..n_pw_kq) |g| {
                rhs[g] = math.complex.add(rhs[g], nl_out[g]);
            }
        }

        // Negate
        for (0..n_pw_kq) |g| {
            rhs[g] = math.complex.scale(rhs[g], -1.0);
        }

        // Project onto conduction band in k+q space
        sternheimer.projectConduction(rhs, kd.occ_kq_const, kd.n_occ_kq);

        // Solve Sternheimer
        const result = try sternheimer.solve(
            alloc,
            kd.apply_ctx_kq,
            rhs,
            kd.eigenvalues_k[n],
            kd.occ_kq_const,
            kd.n_occ_kq,
            kd.basis_kq.gvecs,
            .{
                .tol = shared.cfg.sternheimer_tol,
                .max_iter = shared.cfg.sternheimer_max_iter,
                .alpha_shift = shared.cfg.alpha_shift,
            },
        );

        // Copy result to psi1_per_k (fixed-size buffer, no reallocation needed)
        @memcpy(shared.psi1_per_k[ik][n], result.psi1);
        alloc.free(result.psi1);
    }

    // Compute ρ^(1) for this k-point (weighted by wtk)
    const psi1_const = try alloc.alloc([]const math.Complex, n_occ);
    defer alloc.free(psi1_const);
    for (0..n_occ) |n| psi1_const[n] = shared.psi1_per_k[ik][n];

    const rho1_k_r = try computeRho1QCached(
        alloc,
        shared.grid,
        map_kq_ptr,
        psi0_r_k,
        psi1_const,
        n_occ,
        kd.weight,
    );
    defer alloc.free(rho1_k_r);

    // Accumulate into thread-local buffer
    for (0..total) |i| {
        rho1_local[i] = math.complex.add(rho1_local[i], rho1_k_r[i]);
    }
}

/// Solve DFPT perturbation at q≠0 with multiple k-points.
/// For each k-point, solves Sternheimer equations and accumulates ρ^(1)
/// with the k-point weight. Uses potential mixing for stable convergence.
///
/// Returns the converged ρ^(1)(G) (summed over all k-points) and
/// per-k-point ψ^(1) wavefunctions.
pub fn solvePerturbationQMultiK(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    atom_index: usize,
    direction: usize,
    cfg: DfptConfig,
    q_cart: math.Vec3,
    grid: Grid,
    gs: GroundState,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) !MultiKPertResult {
    const total = grid.count();
    const n_kpts = kpts.len;

    // Build V_ext^(1)_q(G) for this perturbation (bare, fixed) — k-independent
    const vloc1_g = try perturbation.buildLocalPerturbationQ(
        alloc,
        grid,
        atoms[atom_index],
        species,
        direction,
        q_cart,
        ff_tables,
    );
    defer alloc.free(vloc1_g);

    // Build ρ^(1)_core,q(G) (fixed, NLCC) — k-independent
    const rho1_core_g = try perturbation.buildCorePerturbationQ(
        alloc,
        grid,
        atoms[atom_index],
        species,
        direction,
        q_cart,
        rho_core_tables,
    );
    defer alloc.free(rho1_core_g);

    // IFFT ρ^(1)_core,q → real space (complex)
    const rho1_core_g_copy = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_g_copy);
    @memcpy(rho1_core_g_copy, rho1_core_g);
    const rho1_core_r = try alloc.alloc(math.Complex, total);
    defer alloc.free(rho1_core_r);
    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_core_g_copy, rho1_core_r, null);

    // Allocate per-k-point first-order wavefunctions
    var psi1_per_k = try alloc.alloc([][]math.Complex, n_kpts);
    for (0..n_kpts) |ik| {
        const n_occ = kpts[ik].n_occ;
        const n_pw_kq = kpts[ik].n_pw_kq;
        psi1_per_k[ik] = try alloc.alloc([]math.Complex, n_occ);
        for (0..n_occ) |n| {
            psi1_per_k[ik][n] = try alloc.alloc(math.Complex, n_pw_kq);
            @memset(psi1_per_k[ik][n], math.complex.init(0.0, 0.0));
        }
    }

    // Initialize V_SCF(G) = V_loc^(1)(G) [bare perturbation, no screening]
    var v_scf_g = try alloc.alloc(math.Complex, total);
    @memcpy(v_scf_g, vloc1_g);

    // Pre-compute ψ^(0)_k(r) for all k-points (invariant during SCF)
    const psi0_r_cache = try alloc.alloc([][]math.Complex, n_kpts);
    defer {
        for (psi0_r_cache) |kc| {
            for (kc) |band| alloc.free(band);
            alloc.free(kc);
        }
        alloc.free(psi0_r_cache);
    }
    for (kpts, 0..) |*kd, ik| {
        psi0_r_cache[ik] = try alloc.alloc([]math.Complex, kd.n_occ);
        for (0..kd.n_occ) |n| {
            psi0_r_cache[ik][n] = try alloc.alloc(math.Complex, total);
            const work = try alloc.alloc(math.Complex, total);
            defer alloc.free(work);
            @memset(work, math.complex.init(0.0, 0.0));
            kd.map_k.scatter(kd.wavefunctions_k_const[n], work);
            try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work, psi0_r_cache[ik][n], null);
        }
    }

    // Pulay mixer for potential mixing
    var pulay = scf_mod.ComplexPulayMixer.init(alloc, cfg.pulay_history);
    defer pulay.deinit();

    // Pulay restart state
    var best_vresid: f64 = std.math.inf(f64);
    var best_v_scf: ?[]math.Complex = null;
    defer if (best_v_scf) |v| alloc.free(v);
    var pulay_active_since: usize = cfg.pulay_start;
    const restart_factor: f64 = 5.0;
    var force_converge: bool = false;

    // DFPT SCF loop — potential mixing, multi-k
    var iter: usize = 0;
    while (iter < cfg.scf_max_iter) : (iter += 1) {
        // IFFT V_SCF(G) → V_SCF(r) [complex]
        const v_scf_g_copy = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_scf_g_copy);
        @memcpy(v_scf_g_copy, v_scf_g);
        const v_scf_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_scf_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, v_scf_g_copy, v_scf_r, null);

        // Accumulate ρ^(1) over all k-points (parallel or sequential)
        const rho1_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_r);
        @memset(rho1_r, math.complex.init(0.0, 0.0));

        const thread_count = scf_mod.kpointThreadCount(n_kpts, cfg.kpoint_threads);

        if (thread_count <= 1) {
            // Sequential path — use cached ψ^(0)(r)
            for (kpts, 0..) |*kd, ik| {
                const n_occ = kd.n_occ;
                const n_pw_kq = kd.n_pw_kq;
                const map_kq_ptr: *const scf_mod.PwGridMap = &kd.map_kq;

                // Nonlocal contexts
                const nl_ctx_k_opt = kd.apply_ctx_k.nonlocal_ctx;
                const nl_ctx_kq_opt = kd.apply_ctx_kq.nonlocal_ctx;

                // Solve Sternheimer for each occupied band at this k-point
                for (0..n_occ) |n| {
                    // RHS: -P_c^{k+q} × H^(1)|ψ^(0)_{n,k}⟩
                    const rhs = try applyV1PsiQCached(alloc, grid, map_kq_ptr, v_scf_r, psi0_r_cache[ik][n], n_pw_kq);
                    defer alloc.free(rhs);

                    // Add nonlocal perturbation
                    if (nl_ctx_k_opt != null and nl_ctx_kq_opt != null) {
                        const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
                        defer alloc.free(nl_out);
                        try perturbation.applyNonlocalPerturbationQ(
                            alloc,
                            kd.basis_k.gvecs,
                            kd.basis_kq.gvecs,
                            atoms,
                            nl_ctx_k_opt.?,
                            nl_ctx_kq_opt.?,
                            atom_index,
                            direction,
                            1.0 / grid.volume,
                            kd.wavefunctions_k_const[n],
                            nl_out,
                        );
                        for (0..n_pw_kq) |g| {
                            rhs[g] = math.complex.add(rhs[g], nl_out[g]);
                        }
                    }

                    // Negate
                    for (0..n_pw_kq) |g| {
                        rhs[g] = math.complex.scale(rhs[g], -1.0);
                    }

                    // Project onto conduction band in k+q space
                    sternheimer.projectConduction(rhs, kd.occ_kq_const, kd.n_occ_kq);

                    // Solve Sternheimer
                    const result = try sternheimer.solve(
                        alloc,
                        kd.apply_ctx_kq,
                        rhs,
                        kd.eigenvalues_k[n],
                        kd.occ_kq_const,
                        kd.n_occ_kq,
                        kd.basis_kq.gvecs,
                        .{
                            .tol = cfg.sternheimer_tol,
                            .max_iter = cfg.sternheimer_max_iter,
                            .alpha_shift = cfg.alpha_shift,
                        },
                    );

                    @memcpy(psi1_per_k[ik][n], result.psi1);
                    alloc.free(result.psi1);
                }

                // Compute ρ^(1) for this k-point (weighted by wtk)
                const psi1_const = try alloc.alloc([]const math.Complex, n_occ);
                defer alloc.free(psi1_const);
                for (0..n_occ) |n| psi1_const[n] = psi1_per_k[ik][n];

                const psi0_r_k_const: []const []const math.Complex = @ptrCast(psi0_r_cache[ik]);
                const rho1_k_r = try computeRho1QCached(
                    alloc,
                    grid,
                    map_kq_ptr,
                    psi0_r_k_const,
                    psi1_const,
                    n_occ,
                    kd.weight,
                );
                defer alloc.free(rho1_k_r);

                // Accumulate
                for (0..total) |i| {
                    rho1_r[i] = math.complex.add(rho1_r[i], rho1_k_r[i]);
                }
            }
        } else {
            // Parallel path — spawn worker threads
            if (iter == 0) {
                logDfpt("dfptQ_mk: using {d} threads for {d} k-points\n", .{ thread_count, n_kpts });
            }

            // Allocate thread-local ρ^(1) buffers
            const rho1_locals = try alloc.alloc(math.Complex, total * thread_count);
            defer alloc.free(rho1_locals);
            @memset(rho1_locals, math.complex.init(0.0, 0.0));

            // Synchronization primitives
            var next_index = std.atomic.Value(usize).init(0);
            var stop = std.atomic.Value(u8).init(0);
            var worker_err: ?anyerror = null;
            var err_mutex = std.Io.Mutex.init;
            var log_mutex = std.Io.Mutex.init;

            // Build const view of psi0_r_cache for parallel workers
            const psi0_r_const_view = try alloc.alloc([]const []const math.Complex, n_kpts);
            defer alloc.free(psi0_r_const_view);
            for (0..n_kpts) |ik2| {
                psi0_r_const_view[ik2] = @ptrCast(psi0_r_cache[ik2]);
            }

            var shared = DfptKpointShared{
                .kpts = kpts,
                .v_scf_r = v_scf_r,
                .vloc1_g = vloc1_g,
                .cfg = cfg,
                .atom_index = atom_index,
                .direction = direction,
                .grid = grid,
                .species = species,
                .atoms = atoms,
                .psi1_per_k = psi1_per_k,
                .rho1_locals = rho1_locals,
                .total = total,
                .psi0_r_cache = psi0_r_const_view,
                .next_index = &next_index,
                .stop = &stop,
                .err = &worker_err,
                .err_mutex = &err_mutex,
                .log_mutex = &log_mutex,
            };

            // Create workers (thread 0 runs on main thread)
            var workers = try alloc.alloc(DfptKpointWorker, thread_count);
            defer alloc.free(workers);
            var threads = try alloc.alloc(std.Thread, thread_count - 1);
            defer alloc.free(threads);

            for (0..thread_count) |ti| {
                workers[ti] = .{ .shared = &shared, .thread_index = ti };
            }

            // Spawn worker threads (skip thread 0, it runs on main)
            for (0..thread_count - 1) |ti| {
                threads[ti] = try std.Thread.spawn(.{}, dfptKpointWorkerFn, .{&workers[ti + 1]});
            }

            // Run thread 0 on main thread
            dfptKpointWorkerFn(&workers[0]);

            // Join all threads
            for (threads) |t| {
                t.join();
            }

            // Check for errors
            if (worker_err) |e| return e;

            // Sum thread-local ρ^(1) into rho1_r
            for (0..thread_count) |ti| {
                const local_start = ti * total;
                const local = rho1_locals[local_start .. local_start + total];
                for (0..total) |i| {
                    rho1_r[i] = math.complex.add(rho1_r[i], local[i]);
                }
            }
        }

        // FFT ρ^(1)(r) → ρ^(1)(G)
        const rho1_g = try complexRealToReciprocal(alloc, grid, rho1_r);
        defer alloc.free(rho1_g);

        // Diagnostic
        {
            var rho_norm: f64 = 0.0;
            for (0..total) |di| {
                rho_norm += rho1_g[di].r * rho1_g[di].r + rho1_g[di].i * rho1_g[di].i;
            }
            rho_norm = @sqrt(rho_norm);
            const d_elec_diag = computeElecDynmatElementQ(vloc1_g, rho1_g, grid.volume);
            logDfpt("dfptQ_mk: iter={d} |rho1|={e:.6} D_elec_bare=({e:.6},{e:.6}) nk={d}\n", .{ iter, rho_norm, d_elec_diag.r, d_elec_diag.i, n_kpts });
        }

        // Build V_out(G) = V_loc^(1) + V_H^(1)[ρ] + V_xc^(1)[ρ]
        const vh1_g = try perturbation.buildHartreePerturbationQ(alloc, grid, rho1_g, q_cart);
        defer alloc.free(vh1_g);

        // V_xc^(1)(r) = f_xc(r) × ρ^(1)_total(r)
        const rho1_g_copy2 = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_g_copy2);
        @memcpy(rho1_g_copy2, rho1_g);
        const rho1_val_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_val_r);
        try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, rho1_g_copy2, rho1_val_r, null);

        const rho1_total_r = try alloc.alloc(math.Complex, total);
        defer alloc.free(rho1_total_r);
        for (0..total) |i| {
            rho1_total_r[i] = math.complex.add(rho1_val_r[i], rho1_core_r[i]);
        }

        const vxc1_r = try perturbation.buildXcPerturbationFullComplex(alloc, gs, rho1_total_r);
        defer alloc.free(vxc1_r);
        const vxc1_r_fft = try alloc.alloc(math.Complex, total);
        defer alloc.free(vxc1_r_fft);
        @memcpy(vxc1_r_fft, vxc1_r);
        const vxc1_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(vxc1_g);
        try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, vxc1_r_fft, vxc1_g, null);

        // V_out(G) = V_loc + V_H + V_xc
        const v_out_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(v_out_g);
        for (0..total) |i| {
            v_out_g[i] = math.complex.add(
                math.complex.add(vloc1_g[i], vh1_g[i]),
                vxc1_g[i],
            );
        }

        // Compute residual
        var residual_norm: f64 = 0.0;
        const residual = try alloc.alloc(math.Complex, total);
        for (0..total) |i| {
            residual[i] = math.complex.sub(v_out_g[i], v_scf_g[i]);
            residual_norm += residual[i].r * residual[i].r + residual[i].i * residual[i].i;
        }
        residual_norm = @sqrt(residual_norm);

        logDfpt("dfptQ_mk: iter={d} vresid={e:.6}\n", .{ iter, residual_norm });

        if (residual_norm < cfg.scf_tol or (force_converge and residual_norm < 10.0 * cfg.scf_tol)) {
            alloc.free(residual);
            logDfpt("dfptQ_mk: converged at iter={d} vresid={e:.6}\n", .{ iter, residual_norm });
            break;
        }

        // Track best residual and save corresponding V_SCF
        if (residual_norm < best_vresid) {
            best_vresid = residual_norm;
            if (best_v_scf == null) best_v_scf = try alloc.alloc(math.Complex, total);
            @memcpy(best_v_scf.?, v_scf_g);
        }

        // Pulay restart: if residual exceeds restart_factor × best, reset and restore
        if (iter >= pulay_active_since and residual_norm > restart_factor * best_vresid and best_vresid < 1.0) {
            if (best_v_scf) |v| @memcpy(v_scf_g, v);
            // If best is near convergence, force accept on next iteration
            if (best_vresid < 10.0 * cfg.scf_tol) {
                force_converge = true;
                logDfpt("dfptQ_mk: Pulay restart (near-converged) at iter={d} vresid={e:.6} best={e:.6}\n", .{ iter, residual_norm, best_vresid });
                alloc.free(residual);
                continue;
            }
            pulay.reset();
            pulay_active_since = iter + 1 + cfg.pulay_start;
            logDfpt("dfptQ_mk: Pulay restart at iter={d} vresid={e:.6} best={e:.6}\n", .{ iter, residual_norm, best_vresid });
            alloc.free(residual);
            continue;
        }

        // Mix
        if (cfg.pulay_history > 0 and iter >= pulay_active_since) {
            try pulay.mixWithResidual(v_scf_g, residual, cfg.mixing_beta);
        } else {
            const beta = cfg.mixing_beta;
            for (0..total) |i| {
                v_scf_g[i] = math.complex.add(v_scf_g[i], math.complex.scale(residual[i], beta));
            }
            alloc.free(residual);
        }
    }

    // Compute final ρ^(1)(G) from converged ψ^(1) (sum over all k-points)
    const final_rho1_r = try alloc.alloc(math.Complex, total);
    @memset(final_rho1_r, math.complex.init(0.0, 0.0));

    for (kpts, 0..) |*kd, ik| {
        const psi1_const = try alloc.alloc([]const math.Complex, kd.n_occ);
        defer alloc.free(psi1_const);
        for (0..kd.n_occ) |n| psi1_const[n] = psi1_per_k[ik][n];

        const rho1_k_r = try computeRho1Q(
            alloc,
            grid,
            &kd.map_k,
            &kd.map_kq,
            kd.wavefunctions_k_const,
            psi1_const,
            kd.n_occ,
            kd.n_pw_k,
            kd.n_pw_kq,
            kd.weight,
        );
        defer alloc.free(rho1_k_r);

        for (0..total) |i| {
            final_rho1_r[i] = math.complex.add(final_rho1_r[i], rho1_k_r[i]);
        }
    }

    const final_rho1_g = try complexRealToReciprocal(alloc, grid, final_rho1_r);
    alloc.free(final_rho1_r);

    alloc.free(v_scf_g);

    // Build result with const views
    const psi1_result = try alloc.alloc([]const []math.Complex, n_kpts);
    for (0..n_kpts) |ik| {
        const psi1_const = try alloc.alloc([]math.Complex, kpts[ik].n_occ);
        for (0..kpts[ik].n_occ) |n| psi1_const[n] = psi1_per_k[ik][n];
        psi1_result[ik] = psi1_const;
    }
    // Free the per-k inner slices (ownership transferred to psi1_result above)
    for (0..n_kpts) |ik| {
        alloc.free(psi1_per_k[ik]);
    }
    alloc.free(psi1_per_k);

    return .{
        .rho1_g = final_rho1_g,
        .psi1_per_k = psi1_result,
    };
}

// =====================================================================
// Complex dynamical matrix construction
// =====================================================================

/// Build the full complex dynamical matrix for a finite q-point.
/// Combines electronic, nonlocal, NLCC, Ewald, and self-energy contributions.
/// All (I,J) pairs are computed explicitly; no Hermitianization needed.
fn buildQDynmat(
    alloc: std.mem.Allocator,
    gs: GroundState,
    pert_results: []PerturbationResult,
    vloc1_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []math.Complex,
    charges: []const f64,
    positions: []const math.Vec3,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    q_cart: math.Vec3,
    gvecs_kq: []const plane_wave.GVector,
    apply_ctx_kq: *scf_mod.ApplyContext,
    vxc_g: ?[]const math.Complex,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    const n_atoms = gs.atoms.len;
    const dim = 3 * n_atoms;
    const grid = gs.grid;

    var dyn_q = try alloc.alloc(math.Complex, dim * dim);
    errdefer alloc.free(dyn_q);
    @memset(dyn_q, math.complex.init(0.0, 0.0));

    // ================================================================
    // Step 1: Accumulate monochromatic (+q) terms into dyn_q
    // These need Hermitianization D = D_raw + conj(D_raw^T)
    // ================================================================

    // Electronic contribution (monochromatic)
    for (0..dim) |i| {
        for (0..dim) |j| {
            dyn_q[i * dim + j] = computeElecDynmatElementQ(
                vloc1_gs[i],
                pert_results[j].rho1_g,
                volume,
            );
        }
    }
    logDfpt("dfptQ_dyn: D_elec(0x,0x)=({e:.6},{e:.6}) D_elec(0x,1x)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i, dyn_q[3].r, dyn_q[3].i });

    // Nonlocal response contribution (monochromatic)
    const nl_resp_q = try computeNonlocalResponseDynmatQ(alloc, gs, pert_results, gvecs_kq, apply_ctx_kq, n_atoms);
    defer alloc.free(nl_resp_q);
    logDfpt("dfptQ_dyn: D_nl_resp(0x,0x)=({e:.6},{e:.6}) D_nl_resp(0x,1x)=({e:.6},{e:.6})\n", .{ nl_resp_q[0].r, nl_resp_q[0].i, nl_resp_q[3].r, nl_resp_q[3].i });
    logDfpt("dfptQ_dyn: D_nl_resp(1x,0x)=({e:.6},{e:.6}) D_nl_resp(1x,1x)=({e:.6},{e:.6})\n", .{ nl_resp_q[3 * dim].r, nl_resp_q[3 * dim].i, nl_resp_q[3 * dim + 3].r, nl_resp_q[3 * dim + 3].i });
    // Print D_elec for atom1 block too
    logDfpt("dfptQ_dyn: D_elec(1x,0x)=({e:.6},{e:.6}) D_elec(1x,1x)=({e:.6},{e:.6})\n", .{ dyn_q[3 * dim].r, dyn_q[3 * dim].i, dyn_q[3 * dim + 3].r, dyn_q[3 * dim + 3].i });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], nl_resp_q[i]);
    }

    // NLCC cross contribution (monochromatic)
    if (gs.rho_core != null) {
        const rho1_val_gs = try alloc.alloc([]math.Complex, dim);
        defer alloc.free(rho1_val_gs);
        for (0..dim) |i| {
            rho1_val_gs[i] = pert_results[i].rho1_g;
        }

        const nlcc_cross = try computeNlccCrossDynmatQ(alloc, grid, gs, rho1_val_gs, rho1_core_gs, n_atoms, irr_info);
        defer alloc.free(nlcc_cross);
        logDfpt("dfptQ_dyn: D_nlcc_cross(0x,0x)=({e:.6},{e:.6})\n", .{ nlcc_cross[0].r, nlcc_cross[0].i });
        logDfpt("dfptQ_dyn: D_nlcc_cross(1x,1x)=({e:.6},{e:.6}) D_nlcc_cross(1x,0x)=({e:.6},{e:.6})\n", .{ nlcc_cross[3 * dim + 3].r, nlcc_cross[3 * dim + 3].i, nlcc_cross[3 * dim].r, nlcc_cross[3 * dim].i });
        for (0..dim * dim) |i| {
            dyn_q[i] = math.complex.add(dyn_q[i], nlcc_cross[i]);
        }
    }

    // ================================================================
    // Step 2: No Hermitianization needed.
    // All (I,J) pairs are computed explicitly, and rho1 includes
    // the full factor 4/Ω (matching ABINIT's dfpt_mkrho convention).
    // D_elec uses rho1 (factor 4/Ω included), D_nl_resp uses factor=4
    // directly (matching ABINIT's d2nl: wtk×occ×two = 4).
    // ABINIT's d2sym3 only does one-sided copy for missing elements;
    // since we compute all pairs, no symmetrization is needed.
    // ================================================================

    // ================================================================
    // Step 3: Add Hermitian (q-independent) terms directly
    // These are already the full contribution.
    // ================================================================

    // Ewald contribution (Ha → Ry: ×2)
    const ewald_dyn_q = try ewald2.ewaldDynmatQ(alloc, cell_bohr, recip, charges, positions, q_cart);
    defer alloc.free(ewald_dyn_q);
    logDfpt("dfptQ_dyn: D_ewald(0x,0x)=({e:.6},{e:.6}) D_ewald(0x,1x)=({e:.6},{e:.6}) [Ha]\n", .{ ewald_dyn_q[0].r, ewald_dyn_q[0].i, ewald_dyn_q[3].r, ewald_dyn_q[3].i });
    logDfpt("dfptQ_dyn: D_ewald(0x,0y)=({e:.6},{e:.6}) D_ewald(0y,0y)=({e:.6},{e:.6}) [Ha]\n", .{ ewald_dyn_q[1].r, ewald_dyn_q[1].i, ewald_dyn_q[dim + 1].r, ewald_dyn_q[dim + 1].i });
    logDfpt("dfptQ_dyn: D_ewald full [Ha]:\n", .{});
    for (0..dim) |row| {
        for (0..dim) |col| {
            logDfpt("  ({e:.6},{e:.6})", .{ ewald_dyn_q[row * dim + col].r, ewald_dyn_q[row * dim + col].i });
        }
        logDfpt("\n", .{});
    }
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.scale(ewald_dyn_q[i], 2.0));
    }

    // Self-energy contribution (local V_loc^(2)) — q-independent
    const self_dyn_real = try dynmat_contrib.computeSelfEnergyDynmat(alloc, grid, gs.species, gs.atoms, rho0_g, gs.ff_tables);
    defer alloc.free(self_dyn_real);
    logDfpt("dfptQ_dyn: D_self(0x,0x)={e:.6} D_self(0x,1x)={e:.6}\n", .{ self_dyn_real[0], self_dyn_real[3] });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(self_dyn_real[i], 0.0));
    }

    // Nonlocal self-energy contribution: V_nl^(2) (q-independent)
    const nl_self_real = try dynmat_contrib.computeNonlocalSelfEnergyDynmat(alloc, gs, n_atoms);
    defer alloc.free(nl_self_real);
    logDfpt("dfptQ_dyn: D_nl_self(0x,0x)={e:.6} D_nl_self(0x,1x)={e:.6}\n", .{ nl_self_real[0], nl_self_real[3] });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nl_self_real[i], 0.0));
    }

    // NLCC self contribution (q-independent)
    if (gs.rho_core != null) {
        if (vxc_g) |vg| {
            const nlcc_self_real = try dynmat_contrib.computeNlccSelfDynmat(alloc, grid, gs.species, gs.atoms, vg, gs.rho_core_tables);
            defer alloc.free(nlcc_self_real);
            logDfpt("dfptQ_dyn: D_nlcc_self(0x,0x)={e:.6}\n", .{nlcc_self_real[0]});
            for (0..dim * dim) |i| {
                dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nlcc_self_real[i], 0.0));
            }
        }
    }
    logDfpt("dfptQ_dyn: total(0x,0x)=({e:.6},{e:.6}) total(0x,1x)=({e:.6},{e:.6}) total(0x,1y)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i, dyn_q[3].r, dyn_q[3].i, dyn_q[4].r, dyn_q[4].i });

    return dyn_q;
}

// =====================================================================
// Multi-k-point dynamical matrix construction
// =====================================================================

/// Compute nonlocal response dynmat D_nl_resp summed over all k-points.
/// D(Iα,Jβ) = Σ_k wtk × 4 × Σ_n ⟨dV_nl_{Iα,q} ψ^(0)_{n,k} | δψ_{n,k,Jβ}⟩
pub fn computeNonlocalResponseDynmatQMultiK(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    pert_results: []MultiKPertResult,
    atoms: []const hamiltonian.AtomData,
    n_atoms: usize,
    volume: f64,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    const dim = 3 * n_atoms;
    const dyn = try alloc.alloc(math.Complex, dim * dim);
    @memset(dyn, math.complex.init(0.0, 0.0));

    for (kpts, 0..) |*kd, ik| {
        const n_pw_kq = kd.n_pw_kq;
        const nl_ctx_k = kd.apply_ctx_k.nonlocal_ctx orelse continue;
        const nl_ctx_kq = kd.apply_ctx_kq.nonlocal_ctx orelse continue;

        const nl_out = try alloc.alloc(math.Complex, n_pw_kq);
        defer alloc.free(nl_out);

        for (0..dim) |i| {
            const ia = i / 3;
            const dir_a = i % 3;
            for (0..kd.n_occ) |n| {
                try perturbation.applyNonlocalPerturbationQ(
                    alloc,
                    kd.basis_k.gvecs,
                    kd.basis_kq.gvecs,
                    atoms,
                    nl_ctx_k,
                    nl_ctx_kq,
                    ia,
                    dir_a,
                    1.0 / volume,
                    kd.wavefunctions_k_const[n],
                    nl_out,
                );

                for (0..dim) |j| {
                    if (!irr_info.is_irreducible[j / 3]) continue;
                    var ip = math.complex.init(0.0, 0.0);
                    for (0..n_pw_kq) |g| {
                        ip = math.complex.add(ip, math.complex.mul(
                            math.complex.conj(nl_out[g]),
                            pert_results[j].psi1_per_k[ik][n][g],
                        ));
                    }
                    // Factor 4 × wtk
                    dyn[i * dim + j] = math.complex.add(dyn[i * dim + j], math.complex.scale(ip, 4.0 * kd.weight));
                }
            }
        }
    }

    return dyn;
}

/// Compute nonlocal self-energy dynmat D_nl_self summed over all k-points.
/// This term only contributes to diagonal (I=J) blocks.
pub fn computeNonlocalSelfDynmatMultiK(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    atoms: []const hamiltonian.AtomData,
    n_atoms: usize,
    volume: f64,
) ![]f64 {
    const dim = 3 * n_atoms;
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);

    const inv_volume = 1.0 / volume;

    for (kpts) |*kd| {
        const n_pw = kd.n_pw_k;
        const nl_ctx = kd.apply_ctx_k.nonlocal_ctx orelse continue;

        const phase = try alloc.alloc(math.Complex, n_pw);
        defer alloc.free(phase);

        for (nl_ctx.species) |entry| {
            const g_count = entry.g_count;
            if (g_count != n_pw) continue;
            if (entry.m_total == 0) continue;

            const proj_std = try alloc.alloc(math.Complex, entry.m_total);
            defer alloc.free(proj_std);
            const proj_alpha = try alloc.alloc([3]math.Complex, entry.m_total);
            defer alloc.free(proj_alpha);
            const proj_alpha_beta = try alloc.alloc([3][3]math.Complex, entry.m_total);
            defer alloc.free(proj_alpha_beta);

            for (atoms, 0..) |atom, atom_idx| {
                if (atom.species_index != entry.species_index) continue;

                for (0..kd.n_occ) |n| {
                    const psi_n = kd.wavefunctions_k_const[n];

                    for (0..n_pw) |g| {
                        phase[g] = math.complex.expi(math.Vec3.dot(kd.basis_k.gvecs[g].cart, atom.position));
                    }

                    var b: usize = 0;
                    while (b < entry.beta_count) : (b += 1) {
                        const offset = entry.m_offsets[b];
                        const m_count = entry.m_counts[b];
                        var m_idx: usize = 0;
                        while (m_idx < m_count) : (m_idx += 1) {
                            const phi = entry.phi[(offset + m_idx) * g_count .. (offset + m_idx + 1) * g_count];
                            var p_std = math.complex.init(0.0, 0.0);
                            var p_a: [3]math.Complex = .{ math.complex.init(0.0, 0.0), math.complex.init(0.0, 0.0), math.complex.init(0.0, 0.0) };
                            var p_ab: [3][3]math.Complex = undefined;
                            for (0..3) |a| {
                                for (0..3) |bb| {
                                    p_ab[a][bb] = math.complex.init(0.0, 0.0);
                                }
                            }

                            for (0..n_pw) |g| {
                                const gvec = kd.basis_k.gvecs[g].cart;
                                const gc = [3]f64{ gvec.x, gvec.y, gvec.z };
                                const base = math.complex.scale(math.complex.mul(phase[g], psi_n[g]), phi[g]);
                                p_std = math.complex.add(p_std, base);
                                for (0..3) |a| {
                                    const weighted = math.complex.scale(base, gc[a]);
                                    p_a[a] = math.complex.add(p_a[a], math.complex.init(-weighted.i, weighted.r));
                                }
                                for (0..3) |a| {
                                    for (0..3) |bb| {
                                        p_ab[a][bb] = math.complex.add(p_ab[a][bb], math.complex.scale(base, -gc[a] * gc[bb]));
                                    }
                                }
                            }

                            proj_std[offset + m_idx] = p_std;
                            proj_alpha[offset + m_idx] = p_a;
                            proj_alpha_beta[offset + m_idx] = p_ab;
                        }
                    }

                    // Accumulate with wtk
                    b = 0;
                    while (b < entry.beta_count) : (b += 1) {
                        const l_b = entry.l_list[b];
                        const off_b = entry.m_offsets[b];
                        const mc_b = entry.m_counts[b];

                        var bp: usize = 0;
                        while (bp < entry.beta_count) : (bp += 1) {
                            if (entry.l_list[bp] != l_b) continue;
                            const dij = entry.coeffs[b * entry.beta_count + bp];
                            if (dij == 0.0) continue;
                            const off_bp = entry.m_offsets[bp];

                            var m_idx: usize = 0;
                            while (m_idx < mc_b) : (m_idx += 1) {
                                const p_std_bp = proj_std[off_bp + m_idx];

                                for (0..3) |alpha| {
                                    for (0..3) |beta| {
                                        const t1 = math.complex.mul(
                                            math.complex.conj(proj_alpha_beta[off_b + m_idx][alpha][beta]),
                                            p_std_bp,
                                        );
                                        const t2 = math.complex.mul(
                                            math.complex.conj(proj_alpha[off_b + m_idx][alpha]),
                                            proj_alpha[off_bp + m_idx][beta],
                                        );

                                        // 4/Ω × wtk × dij × Re(t1+t2)
                                        const val = 4.0 * inv_volume * kd.weight * dij * (t1.r + t2.r);
                                        const i_idx = 3 * atom_idx + alpha;
                                        const j_idx = 3 * atom_idx + beta;
                                        dyn[i_idx * dim + j_idx] += val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dyn;
}

/// Build the full complex dynamical matrix for a finite q-point with multiple k-points.
/// Combines electronic, nonlocal, NLCC, Ewald, and self-energy contributions.
fn buildQDynmatMultiK(
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    pert_results: []MultiKPertResult,
    vloc1_gs: []const []math.Complex,
    rho1_core_gs: []const []math.Complex,
    rho0_g: []math.Complex,
    gs: GroundState,
    charges: []const f64,
    positions: []const math.Vec3,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    q_cart: math.Vec3,
    grid: Grid,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    rho_core: ?[]const f64,
    vxc_g: ?[]const math.Complex,
    vdw_cfg: config_mod.VdwConfig,
    irr_info: dynmat_mod.IrreducibleAtomInfo,
) ![]math.Complex {
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;

    var dyn_q = try alloc.alloc(math.Complex, dim * dim);
    errdefer alloc.free(dyn_q);
    @memset(dyn_q, math.complex.init(0.0, 0.0));

    // ================================================================
    // Step 1: Electronic contribution (k-independent: uses total ρ^(1))
    // Only irreducible columns j are computed (others restored by symmetry)
    // ================================================================
    for (0..dim) |i| {
        for (0..dim) |j| {
            if (!irr_info.is_irreducible[j / 3]) continue;
            dyn_q[i * dim + j] = computeElecDynmatElementQ(
                vloc1_gs[i],
                pert_results[j].rho1_g,
                volume,
            );
        }
    }
    logDfpt("dfptQ_mk_dyn: D_elec(0x,0x)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i });

    // ================================================================
    // Step 2: Nonlocal response (k-dependent, summed over k-points)
    // Only irreducible columns j are computed
    // ================================================================
    const nl_resp_q = try computeNonlocalResponseDynmatQMultiK(alloc, kpts, pert_results, atoms, n_atoms, volume, irr_info);
    defer alloc.free(nl_resp_q);
    logDfpt("dfptQ_mk_dyn: D_nl_resp(0x,0x)=({e:.6},{e:.6})\n", .{ nl_resp_q[0].r, nl_resp_q[0].i });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], nl_resp_q[i]);
    }

    // ================================================================
    // Step 3: NLCC cross contribution (k-independent: uses total ρ^(1))
    // ================================================================
    if (rho_core != null) {
        const rho1_val_gs = try alloc.alloc([]math.Complex, dim);
        defer alloc.free(rho1_val_gs);
        for (0..dim) |i| {
            rho1_val_gs[i] = pert_results[i].rho1_g;
        }

        const nlcc_cross = try computeNlccCrossDynmatQ(alloc, grid, gs, rho1_val_gs, rho1_core_gs, n_atoms, irr_info);
        defer alloc.free(nlcc_cross);
        logDfpt("dfptQ_mk_dyn: D_nlcc_cross(0x,0x)=({e:.6},{e:.6})\n", .{ nlcc_cross[0].r, nlcc_cross[0].i });
        for (0..dim * dim) |i| {
            dyn_q[i] = math.complex.add(dyn_q[i], nlcc_cross[i]);
        }
    }

    // ================================================================
    // Step 4: k-independent terms
    // ================================================================

    // Ewald (Ha → Ry: ×2)
    const ewald_dyn_q = try ewald2.ewaldDynmatQ(alloc, cell_bohr, recip, charges, positions, q_cart);
    defer alloc.free(ewald_dyn_q);
    logDfpt("dfptQ_mk_dyn: D_ewald(0x,0x)=({e:.6},{e:.6}) [Ha]\n", .{ ewald_dyn_q[0].r, ewald_dyn_q[0].i });
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.scale(ewald_dyn_q[i], 2.0));
    }

    // Self-energy (local V_loc^(2))
    const self_dyn_real = try dynmat_contrib.computeSelfEnergyDynmat(alloc, grid, species, atoms, rho0_g, ff_tables);
    defer alloc.free(self_dyn_real);
    logDfpt("dfptQ_mk_dyn: D_self(0x,0x)={e:.6}\n", .{self_dyn_real[0]});
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(self_dyn_real[i], 0.0));
    }

    // Nonlocal self-energy: V_nl^(2) (k-dependent, summed over k-points)
    const nl_self_real = try computeNonlocalSelfDynmatMultiK(alloc, kpts, atoms, n_atoms, volume);
    defer alloc.free(nl_self_real);
    logDfpt("dfptQ_mk_dyn: D_nl_self(0x,0x)={e:.6}\n", .{nl_self_real[0]});
    for (0..dim * dim) |i| {
        dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nl_self_real[i], 0.0));
    }

    // NLCC self contribution
    if (rho_core != null) {
        if (vxc_g) |vg| {
            const nlcc_self_real = try dynmat_contrib.computeNlccSelfDynmat(alloc, grid, species, atoms, vg, rho_core_tables);
            defer alloc.free(nlcc_self_real);
            logDfpt("dfptQ_mk_dyn: D_nlcc_self(0x,0x)={e:.6}\n", .{nlcc_self_real[0]});
            for (0..dim * dim) |i| {
                dyn_q[i] = math.complex.add(dyn_q[i], math.complex.init(nlcc_self_real[i], 0.0));
            }
        }
    }

    // ================================================================
    // Step 5: D3 dispersion contribution
    // ================================================================
    if (vdw_cfg.enabled) {
        const atomic_numbers = try alloc.alloc(usize, n_atoms);
        defer alloc.free(atomic_numbers);
        for (atoms, 0..) |atom, i| {
            atomic_numbers[i] = d3_params.atomicNumber(species[atom.species_index].symbol) orelse 0;
        }
        var damping_params = d3_params.pbe_d3bj;
        if (vdw_cfg.s6) |v| damping_params.s6 = v;
        if (vdw_cfg.s8) |v| damping_params.s8 = v;
        if (vdw_cfg.a1) |v| damping_params.a1 = v;
        if (vdw_cfg.a2) |v| damping_params.a2 = v;
        const d3_dyn_q = try d3.computeDynmatQ(
            alloc,
            atomic_numbers,
            positions,
            cell_bohr,
            damping_params,
            vdw_cfg.cutoff_radius,
            vdw_cfg.cn_cutoff,
            q_cart,
        );
        defer alloc.free(d3_dyn_q);
        logDfpt("dfptQ_mk_dyn: D_d3(0x,0x)=({e:.6},{e:.6})\n", .{ d3_dyn_q[0].r, d3_dyn_q[0].i });
        for (0..dim * dim) |i| {
            dyn_q[i] = math.complex.add(dyn_q[i], d3_dyn_q[i]);
        }
    }

    logDfpt("dfptQ_mk_dyn: total(0x,0x)=({e:.6},{e:.6})\n", .{ dyn_q[0].r, dyn_q[0].i });

    return dyn_q;
}

// =====================================================================
// Q-path generation
// =====================================================================

/// Generate FCC q-path: Γ-X-W-K-Γ-L
pub fn generateFccQPath(
    alloc: std.mem.Allocator,
    recip: math.Mat3,
    npoints_per_seg: usize,
) !struct {
    q_points_frac: []math.Vec3,
    q_points_cart: []math.Vec3,
    distances: []f64,
    labels: [][]const u8,
    label_positions: []usize,
} {
    // FCC high-symmetry points in fractional (reduced) coordinates
    // Path: Γ-X-W-K-Γ-L (same as ABINIT anaddb)
    const points = [_]math.Vec3{
        math.Vec3{ .x = 0.000, .y = 0.000, .z = 0.000 }, // Γ
        math.Vec3{ .x = 0.500, .y = 0.000, .z = 0.500 }, // X
        math.Vec3{ .x = 0.500, .y = 0.250, .z = 0.750 }, // W
        math.Vec3{ .x = 0.375, .y = 0.375, .z = 0.750 }, // K
        math.Vec3{ .x = 0.000, .y = 0.000, .z = 0.000 }, // Γ
        math.Vec3{ .x = 0.500, .y = 0.500, .z = 0.500 }, // L
    };
    const labels = [_][]const u8{ "G", "X", "W", "K", "G", "L" };
    const n_segs = points.len - 1;
    const n_total = n_segs * npoints_per_seg + 1;

    var q_frac = try alloc.alloc(math.Vec3, n_total);
    errdefer alloc.free(q_frac);
    var q_cart = try alloc.alloc(math.Vec3, n_total);
    errdefer alloc.free(q_cart);
    var dists = try alloc.alloc(f64, n_total);
    errdefer alloc.free(dists);
    var label_list = try alloc.alloc([]const u8, points.len);
    errdefer alloc.free(label_list);
    var label_pos = try alloc.alloc(usize, points.len);
    errdefer alloc.free(label_pos);

    for (0..points.len) |i| {
        label_list[i] = labels[i];
        label_pos[i] = if (i == 0) 0 else i * npoints_per_seg;
    }

    var idx: usize = 0;
    var cum_dist: f64 = 0.0;
    for (0..n_segs) |seg| {
        const p0 = points[seg];
        const p1 = points[seg + 1];
        for (0..npoints_per_seg) |ip| {
            const t = @as(f64, @floatFromInt(ip)) / @as(f64, @floatFromInt(npoints_per_seg));
            const qf = math.Vec3{
                .x = p0.x + t * (p1.x - p0.x),
                .y = p0.y + t * (p1.y - p0.y),
                .z = p0.z + t * (p1.z - p0.z),
            };
            q_frac[idx] = qf;
            // Convert to Cartesian: q_cart = qf.x * b1 + qf.y * b2 + qf.z * b3
            q_cart[idx] = math.Vec3.add(
                math.Vec3.add(
                    math.Vec3.scale(recip.row(0), qf.x),
                    math.Vec3.scale(recip.row(1), qf.y),
                ),
                math.Vec3.scale(recip.row(2), qf.z),
            );
            if (idx == 0) {
                dists[idx] = 0.0;
            } else {
                const dq = math.Vec3.sub(q_cart[idx], q_cart[idx - 1]);
                cum_dist += math.Vec3.norm(dq);
                dists[idx] = cum_dist;
            }
            idx += 1;
        }
    }
    // Last point
    q_frac[idx] = points[n_segs];
    q_cart[idx] = math.Vec3.add(
        math.Vec3.add(
            math.Vec3.scale(recip.row(0), points[n_segs].x),
            math.Vec3.scale(recip.row(1), points[n_segs].y),
        ),
        math.Vec3.scale(recip.row(2), points[n_segs].z),
    );
    if (idx > 0) {
        const dq = math.Vec3.sub(q_cart[idx], q_cart[idx - 1]);
        cum_dist += math.Vec3.norm(dq);
    }
    dists[idx] = cum_dist;

    return .{
        .q_points_frac = q_frac,
        .q_points_cart = q_cart,
        .distances = dists,
        .labels = label_list,
        .label_positions = label_pos,
    };
}

/// Generate q-path from config-specified high-symmetry points.
/// Same format as generateFccQPath but with user-supplied points.
pub fn generateQPathFromConfig(
    alloc: std.mem.Allocator,
    qpath_points: []const config_mod.BandPathPoint,
    npoints_per_seg: usize,
    recip: math.Mat3,
) !struct {
    q_points_frac: []math.Vec3,
    q_points_cart: []math.Vec3,
    distances: []f64,
    labels: [][]const u8,
    label_positions: []usize,
} {
    const n_pts = qpath_points.len;
    if (n_pts < 2) return error.InvalidQPath;

    const n_segs = n_pts - 1;
    const n_total = n_segs * npoints_per_seg + 1;

    var q_frac = try alloc.alloc(math.Vec3, n_total);
    errdefer alloc.free(q_frac);
    var q_cart = try alloc.alloc(math.Vec3, n_total);
    errdefer alloc.free(q_cart);
    var dists = try alloc.alloc(f64, n_total);
    errdefer alloc.free(dists);
    var label_list = try alloc.alloc([]const u8, n_pts);
    errdefer alloc.free(label_list);
    var label_pos = try alloc.alloc(usize, n_pts);
    errdefer alloc.free(label_pos);

    for (0..n_pts) |i| {
        label_list[i] = qpath_points[i].label;
        label_pos[i] = if (i == 0) 0 else i * npoints_per_seg;
    }

    var idx: usize = 0;
    var cum_dist: f64 = 0.0;
    for (0..n_segs) |seg| {
        const p0 = qpath_points[seg].k;
        const p1 = qpath_points[seg + 1].k;
        for (0..npoints_per_seg) |ip| {
            const t = @as(f64, @floatFromInt(ip)) / @as(f64, @floatFromInt(npoints_per_seg));
            const qf = math.Vec3{
                .x = p0.x + t * (p1.x - p0.x),
                .y = p0.y + t * (p1.y - p0.y),
                .z = p0.z + t * (p1.z - p0.z),
            };
            q_frac[idx] = qf;
            q_cart[idx] = math.Vec3.add(
                math.Vec3.add(
                    math.Vec3.scale(recip.row(0), qf.x),
                    math.Vec3.scale(recip.row(1), qf.y),
                ),
                math.Vec3.scale(recip.row(2), qf.z),
            );
            if (idx == 0) {
                dists[idx] = 0.0;
            } else {
                const dq = math.Vec3.sub(q_cart[idx], q_cart[idx - 1]);
                cum_dist += math.Vec3.norm(dq);
                dists[idx] = cum_dist;
            }
            idx += 1;
        }
    }
    // Last point
    const last_pt = qpath_points[n_segs].k;
    q_frac[idx] = last_pt;
    q_cart[idx] = math.Vec3.add(
        math.Vec3.add(
            math.Vec3.scale(recip.row(0), last_pt.x),
            math.Vec3.scale(recip.row(1), last_pt.y),
        ),
        math.Vec3.scale(recip.row(2), last_pt.z),
    );
    if (idx > 0) {
        const dq = math.Vec3.sub(q_cart[idx], q_cart[idx - 1]);
        cum_dist += math.Vec3.norm(dq);
    }
    dists[idx] = cum_dist;

    return .{
        .q_points_frac = q_frac,
        .q_points_cart = q_cart,
        .distances = dists,
        .labels = label_list,
        .label_positions = label_pos,
    };
}

// =====================================================================
// Phonon band structure entry point
// =====================================================================

/// Result of phonon band structure calculation.
pub const PhononBandResult = struct {
    /// q-path distances for plotting
    distances: []f64,
    /// Frequencies [n_q][n_modes] in cm⁻¹
    frequencies: [][]f64,
    /// Number of modes (3 × n_atoms)
    n_modes: usize,
    /// Number of q-points
    n_q: usize,
    /// Labels for high-symmetry points
    labels: [][]const u8,
    /// Label positions (indices into distances)
    label_positions: []usize,

    pub fn deinit(self: *PhononBandResult, alloc: std.mem.Allocator) void {
        for (self.frequencies) |f| alloc.free(f);
        alloc.free(self.frequencies);
        alloc.free(self.distances);
        alloc.free(self.labels);
        alloc.free(self.label_positions);
    }
};

/// Run DFPT phonon band structure calculation.
/// Computes phonon frequencies along a q-path for the FCC lattice.
pub fn runPhononBand(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    npoints_per_seg: usize,
) !PhononBandResult {
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const grid = scf_result.grid;

    logDfpt("dfpt_band: starting phonon band calculation ({d} atoms)\n", .{n_atoms});

    // Prepare ground state (PW basis, eigenvalues, wavefunctions, NLCC, etc.)
    var prepared = try dfpt.prepareGroundState(alloc, io, cfg, scf_result, species, atoms, volume, recip);
    defer prepared.deinit();
    const gs = prepared.gs;

    // Ionic data for dynmat construction
    const ionic = try IonicData.init(alloc, species, atoms);
    defer ionic.deinit(alloc);

    // Ground-state density in G-space
    const rho0_g = try scf_mod.realToReciprocal(alloc, grid, scf_result.density, false);
    defer alloc.free(rho0_g);

    // V_xc(G) for NLCC self-energy (need mutable copy since realToReciprocal requires mutable input)
    var vxc_g: ?[]math.Complex = null;
    if (prepared.vxc_r) |v| {
        vxc_g = try scf_mod.realToReciprocal(alloc, grid, v, false);
    }
    defer if (vxc_g) |v| alloc.free(v);

    // Generate q-path
    const qpath = try generateFccQPath(alloc, recip, npoints_per_seg);
    defer alloc.free(qpath.q_points_frac);
    defer alloc.free(qpath.q_points_cart);

    const n_q = qpath.q_points_cart.len;
    logDfpt("dfpt_band: {d} q-points along path\n", .{n_q});

    // Build symmetry operations and atom mapping table for dynmat symmetrization
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    defer alloc.free(symops);
    logDfpt("dfpt_band: {d} symmetry operations found\n", .{symops.len});

    const sym_data = try dynmat_mod.buildIndsym(alloc, symops, atoms, recip, 1e-5);
    defer {
        for (sym_data.indsym) |row| alloc.free(row);
        alloc.free(sym_data.indsym);
        for (sym_data.tnons_shift) |row| alloc.free(row);
        alloc.free(sym_data.tnons_shift);
    }

    // Allocate result arrays
    var frequencies = try alloc.alloc([]f64, n_q);
    var freq_count: usize = 0;
    errdefer {
        for (0..freq_count) |i| alloc.free(frequencies[i]);
        alloc.free(frequencies);
    }

    const dfpt_cfg = DfptConfig.fromConfig(cfg);

    // ---------------------------------------------------------------
    // Prepare full-BZ k-point ground-state data (q-independent).
    // DFPT requires the full BZ sum over k-points. Even if SCF used
    // IBZ k-points, DFPT needs all k-points in the full BZ because
    // for q≠0, the function f(k, k+q) is not invariant under the
    // symmetry operations that map k to its star.
    // ---------------------------------------------------------------
    const kgs_data = try prepareFullBZKpoints(
        alloc, io,
        cfg,
        &gs,
        prepared.local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
    );
    defer {
        for (kgs_data) |*kg| kg.deinit(alloc);
        alloc.free(kgs_data);
    }
    const n_kpts = kgs_data.len;

    // q-point loop
    for (0..n_q) |iq| {
        const q_cart = qpath.q_points_cart[iq];
        const qf = qpath.q_points_frac[iq];
        const q_norm = math.Vec3.norm(q_cart);

        logDfpt("dfpt_band: q[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n", .{ iq, qf.x, qf.y, qf.z, q_norm });
        logDfpt("dfpt_band: q[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });

        // Find irreducible atoms for this q-point
        var irr_info = try dynmat_mod.findIrreducibleAtoms(alloc, symops, sym_data.indsym, n_atoms, qf);
        defer irr_info.deinit(alloc);
        logDfpt("dfpt_band: q[{d}] {d}/{d} irreducible atoms\n", .{ iq, irr_info.n_irr_atoms, n_atoms });

        // Find irreducible perturbations (atom+direction) for this q-point
        // Note: direction-level perturbation reduction (findIrreduciblePerturbations) is available
        // in dynmat_mod but not yet used for SCF solving — buildQDynmatMultiK requires all 3
        // directions per irreducible atom. Log atom-level reduction only.

        // Build KPointDfptData from precomputed ground-state data + q-dependent k+q data
        const pert_thread_count = dfpt.perturbationThreadCount(dim, dfpt_cfg.perturbation_threads);
        const kpts = try buildKPointDfptDataFromGS(
            alloc, io,
            kgs_data,
            q_cart,
            q_norm,
            cfg,
            prepared.local_r,
            species,
            atoms,
            recip,
            volume,
            grid,
            pert_thread_count,
        );
        defer {
            for (kpts) |*kd| kd.deinitQOnly(alloc);
            alloc.free(kpts);
        }

        // Solve perturbations for each atom and direction using multi-k solver
        var pert_results_mk = try alloc.alloc(MultiKPertResult, dim);
        var vloc1_gs = try alloc.alloc([]math.Complex, dim);
        var rho1_core_gs = try alloc.alloc([]math.Complex, dim);
        var pert_count_local: usize = 0;
        var vloc1_count_local: usize = 0;
        var rho1_core_count_local: usize = 0;
        defer {
            for (0..pert_count_local) |i| pert_results_mk[i].deinit(alloc);
            alloc.free(pert_results_mk);
            for (0..vloc1_count_local) |i| alloc.free(vloc1_gs[i]);
            alloc.free(vloc1_gs);
            for (0..rho1_core_count_local) |i| alloc.free(rho1_core_gs[i]);
            alloc.free(rho1_core_gs);
        }

        if (pert_thread_count <= 1) {
            // Phase 1: Build vloc1, rho1_core for ALL perturbations (cheap, analytic)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.ff_tables,
                    );
                    vloc1_count_local = pidx + 1;

                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                    rho1_core_count_local = pidx + 1;
                }
            }

            // Phase 2: Solve perturbation SCF for irreducible atoms only (expensive)
            // Direction-level reduction is used for dynmat reconstruction, not for SCF solving,
            // because buildQDynmatMultiK requires all 3 directions per irreducible atom.
            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
            }
            pert_count_local = dim;

            for (irr_info.irr_atom_indices) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    pert_results_mk[pidx] = try solvePerturbationQMultiK(
                        alloc,
                        kpts,
                        ia,
                        dir,
                        dfpt_cfg,
                        q_cart,
                        grid,
                        gs,
                        species,
                        atoms,
                        gs.ff_tables,
                        gs.rho_core_tables,
                    );

                    {
                        var rho1_norm: f64 = 0.0;
                        for (pert_results_mk[pidx].rho1_g) |c| {
                            rho1_norm += c.r * c.r + c.i * c.i;
                        }
                        logDfpt("dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n", .{ ia, dir, @sqrt(rho1_norm) });
                    }
                }
            }
        } else {
            // Parallel path — solve only irreducible atoms' perturbations concurrently
            const n_irr_perts = irr_info.n_irr_atoms * 3;
            var pert_dfpt_cfg = dfpt_cfg;
            pert_dfpt_cfg.kpoint_threads = dfpt.kpointThreadsForPertParallel(pert_thread_count, dfpt_cfg.kpoint_threads);

            logDfpt("dfpt_band: using {d} pert threads × {d} kpt threads for {d} perturbations ({d} irreducible)\n", .{ pert_thread_count, pert_dfpt_cfg.kpoint_threads, dim, n_irr_perts });

            // Initialize output arrays to safe defaults for cleanup
            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
                vloc1_gs[i] = &.{};
                rho1_core_gs[i] = &.{};
            }
            pert_count_local = dim;
            vloc1_count_local = dim;
            rho1_core_count_local = dim;

            // Build vloc1 and rho1_core for ALL perturbations (cheap, serial)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;
                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.ff_tables,
                    );
                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                }
            }

            var next_index = std.atomic.Value(usize).init(0);
            var stop_flag = std.atomic.Value(u8).init(0);
            var worker_err: ?anyerror = null;
            var err_mutex = std.Io.Mutex.init;
            var log_mutex = std.Io.Mutex.init;

            var qshared = QPointPertShared{
                .alloc = alloc,
                .kpts = kpts,
                .dfpt_cfg = &pert_dfpt_cfg,
                .q_cart = q_cart,
                .grid = grid,
                .gs = gs,
                .species = species,
                .atoms = atoms,
                .ff_tables = gs.ff_tables,
                .rho_core_tables = gs.rho_core_tables,
                .pert_results_mk = pert_results_mk,
                .vloc1_gs = vloc1_gs,
                .rho1_core_gs = rho1_core_gs,
                .dim = n_irr_perts,
                .irr_pert_indices = null,
                .next_index = &next_index,
                .stop = &stop_flag,
                .err = &worker_err,
                .err_mutex = &err_mutex,
                .log_mutex = &log_mutex,
            };

            // Build irr_pert_indices for atom-level reduction
            const irr_pert_indices = try alloc.alloc(usize, n_irr_perts);
            defer alloc.free(irr_pert_indices);
            {
                var pi: usize = 0;
                for (irr_info.irr_atom_indices) |ia| {
                    for (0..3) |dir_idx| {
                        irr_pert_indices[pi] = 3 * ia + dir_idx;
                        pi += 1;
                    }
                }
            }
            qshared.irr_pert_indices = irr_pert_indices;

            var workers = try alloc.alloc(QPointPertWorker, pert_thread_count);
            defer alloc.free(workers);
            var threads_arr = try alloc.alloc(std.Thread, pert_thread_count - 1);
            defer alloc.free(threads_arr);

            for (0..pert_thread_count) |ti| {
                workers[ti] = .{ .shared = &qshared, .thread_index = ti };
            }

            for (0..pert_thread_count - 1) |ti| {
                threads_arr[ti] = try std.Thread.spawn(.{}, qpointPertWorkerFn, .{&workers[ti + 1]});
            }

            qpointPertWorkerFn(&workers[0]);

            for (threads_arr) |t| {
                t.join();
            }

            if (worker_err) |e| return e;
        }

        // Build complex dynamical matrix from all contributions (multi-k version)
        const dyn_q = try buildQDynmatMultiK(
            alloc,
            kpts,
            pert_results_mk,
            vloc1_gs,
            rho1_core_gs,
            rho0_g,
            gs,
            ionic.charges,
            ionic.positions,
            cell_bohr,
            recip,
            volume,
            q_cart,
            grid,
            species,
            atoms,
            gs.ff_tables,
            gs.rho_core_tables,
            gs.rho_core,
            vxc_g,
            cfg.vdw,
            irr_info,
        );
        defer alloc.free(dyn_q);

        // Reconstruct non-irreducible columns from symmetry
        if (irr_info.n_irr_atoms < n_atoms) {
            dynmat_mod.reconstructDynmatColumnsComplex(dyn_q, n_atoms, irr_info, symops, sym_data.indsym, sym_data.tnons_shift, cell_bohr, qf);
        }

        // Apply acoustic sum rule at Γ-point (q=0)
        if (q_norm < 1e-10) {
            dynmat_mod.applyASRComplex(dyn_q, n_atoms);
        }

        // Mass-weight
        dynmat_mod.massWeightComplex(dyn_q, n_atoms, ionic.masses);

        // Diagonalize complex Hermitian matrix
        var result_q = try dynmat_mod.diagonalizeComplex(alloc, dyn_q, dim);
        defer result_q.deinit(alloc);

        frequencies[iq] = try alloc.alloc(f64, dim);
        @memcpy(frequencies[iq], result_q.frequencies_cm1);
        freq_count = iq + 1;

        logDfpt("dfpt_band: q[{d}] freqs:", .{iq});
        for (result_q.frequencies_cm1) |f| {
            logDfpt(" {d:.1}", .{f});
        }
        logDfpt("\n", .{});
    }

    return PhononBandResult{
        .distances = qpath.distances,
        .frequencies = frequencies,
        .n_modes = dim,
        .n_q = n_q,
        .labels = qpath.labels,
        .label_positions = qpath.label_positions,
    };
}

// =====================================================================
// Perturbation parallelism for q≠0
// =====================================================================

const QPointPertShared = struct {
    alloc: std.mem.Allocator,
    kpts: []KPointDfptData,
    dfpt_cfg: *const DfptConfig,
    q_cart: math.Vec3,
    grid: Grid,
    gs: GroundState,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    pert_results_mk: []MultiKPertResult,
    vloc1_gs: [][]math.Complex,
    rho1_core_gs: [][]math.Complex,
    dim: usize,
    irr_pert_indices: ?[]const usize = null,

    next_index: *std.atomic.Value(usize),
    stop: *std.atomic.Value(u8),
    err: *?anyerror,
    err_mutex: *std.Io.Mutex,
    log_mutex: *std.Io.Mutex,
};

const QPointPertWorker = struct {
    shared: *QPointPertShared,
    thread_index: usize,
};

fn setQPointPertError(shared: *QPointPertShared, e: anyerror) void {
    shared.err_mutex.lock();
    defer shared.err_mutex.unlock();
    if (shared.err.* == null) {
        shared.err.* = e;
    }
}

fn qpointPertWorkerFn(worker: *QPointPertWorker) void {
    const shared = worker.shared;
    const alloc = shared.alloc;

    while (true) {
        if (shared.stop.load(.acquire) != 0) break;
        const work_idx = shared.next_index.fetchAdd(1, .acq_rel);
        if (work_idx >= shared.dim) break;

        // Map work index to actual perturbation index
        const idx = if (shared.irr_pert_indices) |indices| indices[work_idx] else work_idx;
        const ia = idx / 3;
        const dir = idx % 3;

        {
            shared.log_mutex.lock();
            defer shared.log_mutex.unlock();
            const dir_names = [_][]const u8{ "x", "y", "z" };
            logDfpt("dfpt_band: [thread {d}] solving perturbation atom={d} dir={s} ({d}/{d})\n", .{ worker.thread_index, ia, dir_names[dir], work_idx + 1, shared.dim });
        }

        // Solve perturbation SCF with all k-points (vloc1/rho1_core already built by caller)
        shared.pert_results_mk[idx] = solvePerturbationQMultiK(
            alloc,
            shared.kpts,
            ia,
            dir,
            shared.dfpt_cfg.*,
            shared.q_cart,
            shared.grid,
            shared.gs,
            shared.species,
            shared.atoms,
            shared.ff_tables,
            shared.rho_core_tables,
        ) catch |e| {
            setQPointPertError(shared, e);
            shared.stop.store(1, .release);
            break;
        };

        // Debug: print rho1 norm
        {
            var rho1_norm: f64 = 0.0;
            for (shared.pert_results_mk[idx].rho1_g) |c| {
                rho1_norm += c.r * c.r + c.i * c.i;
            }
            shared.log_mutex.lock();
            defer shared.log_mutex.unlock();
            logDfpt("dfptQ_mk: pert({d},{d}) |rho1_g|={e:.6}\n", .{ ia, dir, @sqrt(rho1_norm) });
        }
    }
}

// =====================================================================
// IFC-interpolated phonon band structure
// =====================================================================

const ifc_mod = @import("ifc.zig");

/// Run DFPT phonon band structure using IFC interpolation.
/// 1. Compute D(q) on a coarse q-grid via DFPT
/// 2. Fourier transform to IFC: C(R)
/// 3. Interpolate D(q') at arbitrary q-path points
pub fn runPhononBandIFC(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    scf_result: *scf_mod.ScfResult,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
    volume: f64,
) !PhononBandResult {
    const n_atoms = atoms.len;
    const dim = 3 * n_atoms;
    const grid = scf_result.grid;
    const qgrid = cfg.dfpt.qgrid orelse return error.MissingQgrid;

    logDfpt("dfpt_ifc: starting IFC phonon band ({d} atoms, qgrid={d}x{d}x{d})\n", .{ n_atoms, qgrid[0], qgrid[1], qgrid[2] });

    // Prepare ground state
    var prepared = try dfpt.prepareGroundState(alloc, io, cfg, scf_result, species, atoms, volume, recip);
    defer prepared.deinit();
    const gs = prepared.gs;

    // Ionic data for dynmat construction
    const ionic = try IonicData.init(alloc, species, atoms);
    defer ionic.deinit(alloc);

    // Ground-state density in G-space
    const rho0_g = try scf_mod.realToReciprocal(alloc, grid, scf_result.density, false);
    defer alloc.free(rho0_g);

    // V_xc(G) for NLCC self-energy
    var vxc_g: ?[]math.Complex = null;
    if (prepared.vxc_r) |v| {
        vxc_g = try scf_mod.realToReciprocal(alloc, grid, v, false);
    }
    defer if (vxc_g) |v| alloc.free(v);

    // Build symmetry operations for dynmat symmetrization
    const symops = try symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5);
    defer alloc.free(symops);
    logDfpt("dfpt_ifc: {d} symmetry operations found\n", .{symops.len});

    const sym_data = try dynmat_mod.buildIndsym(alloc, symops, atoms, recip, 1e-5);
    defer {
        for (sym_data.indsym) |row| alloc.free(row);
        alloc.free(sym_data.indsym);
        for (sym_data.tnons_shift) |row| alloc.free(row);
        alloc.free(sym_data.tnons_shift);
    }

    const dfpt_cfg = DfptConfig.fromConfig(cfg);

    // Prepare full-BZ k-point ground-state data
    const kgs_data = try prepareFullBZKpoints(
        alloc, io,
        cfg,
        &gs,
        prepared.local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
    );
    defer {
        for (kgs_data) |*kg| kg.deinit(alloc);
        alloc.free(kgs_data);
    }
    const n_kpts = kgs_data.len;

    // =============================================================
    // Phase 1: Compute D(q) on the coarse q-grid
    // =============================================================
    const shift_zero = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const qgrid_points = try mesh_mod.generateKmesh(alloc, qgrid, recip, shift_zero);
    defer alloc.free(qgrid_points);
    const n_qgrid = qgrid_points.len;
    logDfpt("dfpt_ifc: {d} q-grid points for DFPT\n", .{n_qgrid});

    // Store fractional q-points and dynamical matrices
    var q_frac_grid = try alloc.alloc(math.Vec3, n_qgrid);
    defer alloc.free(q_frac_grid);
    var dynmat_grid = try alloc.alloc([]math.Complex, n_qgrid);
    var dynmat_count: usize = 0;
    defer {
        for (0..dynmat_count) |i| alloc.free(dynmat_grid[i]);
        alloc.free(dynmat_grid);
    }

    for (0..n_qgrid) |iq| {
        const q_cart = qgrid_points[iq].k_cart;
        const qf = qgrid_points[iq].k_frac;
        const q_norm = math.Vec3.norm(q_cart);

        q_frac_grid[iq] = qf;

        logDfpt("dfpt_ifc: q_grid[{d}] = ({d:.4},{d:.4},{d:.4}) |q|={d:.6}\n", .{ iq, qf.x, qf.y, qf.z, q_norm });
        logDfpt("dfpt_ifc: q_grid[{d}] using {d} k-points (full BZ)\n", .{ iq, n_kpts });

        // Find irreducible atoms for this q-point
        var irr_info = try dynmat_mod.findIrreducibleAtoms(alloc, symops, sym_data.indsym, n_atoms, qf);
        defer irr_info.deinit(alloc);
        logDfpt("dfpt_ifc: q_grid[{d}] {d}/{d} irreducible atoms\n", .{ iq, irr_info.n_irr_atoms, n_atoms });

        // Find irreducible perturbations (atom+direction) for this q-point

        // Build k+q data for this q-point
        const pert_thread_count = dfpt.perturbationThreadCount(dim, dfpt_cfg.perturbation_threads);
        const kpts = try buildKPointDfptDataFromGS(
            alloc, io,
            kgs_data,
            q_cart,
            q_norm,
            cfg,
            prepared.local_r,
            species,
            atoms,
            recip,
            volume,
            grid,
            pert_thread_count,
        );
        defer {
            for (kpts) |*kd| kd.deinitQOnly(alloc);
            alloc.free(kpts);
        }

        // Solve perturbations
        var pert_results_mk = try alloc.alloc(MultiKPertResult, dim);
        var vloc1_gs = try alloc.alloc([]math.Complex, dim);
        var rho1_core_gs = try alloc.alloc([]math.Complex, dim);
        var pert_count_local: usize = 0;
        var vloc1_count_local: usize = 0;
        var rho1_core_count_local: usize = 0;
        defer {
            for (0..pert_count_local) |i| pert_results_mk[i].deinit(alloc);
            alloc.free(pert_results_mk);
            for (0..vloc1_count_local) |i| alloc.free(vloc1_gs[i]);
            alloc.free(vloc1_gs);
            for (0..rho1_core_count_local) |i| alloc.free(rho1_core_gs[i]);
            alloc.free(rho1_core_gs);
        }

        if (pert_thread_count <= 1) {
            // Phase 1: Build vloc1, rho1_core for ALL perturbations (cheap)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.ff_tables,
                    );
                    vloc1_count_local = pidx + 1;

                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                    rho1_core_count_local = pidx + 1;
                }
            }

            // Phase 2: Solve perturbation SCF for irreducible atoms only
            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
            }
            pert_count_local = dim;

            for (irr_info.irr_atom_indices) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;

                    pert_results_mk[pidx] = try solvePerturbationQMultiK(
                        alloc,
                        kpts,
                        ia,
                        dir,
                        dfpt_cfg,
                        q_cart,
                        grid,
                        gs,
                        species,
                        atoms,
                        gs.ff_tables,
                        gs.rho_core_tables,
                    );
                }
            }
        } else {
            const n_irr_perts = irr_info.n_irr_atoms * 3;
            var pert_dfpt_cfg = dfpt_cfg;
            pert_dfpt_cfg.kpoint_threads = dfpt.kpointThreadsForPertParallel(pert_thread_count, dfpt_cfg.kpoint_threads);

            for (0..dim) |i| {
                pert_results_mk[i] = .{ .rho1_g = &.{}, .psi1_per_k = &.{} };
                vloc1_gs[i] = &.{};
                rho1_core_gs[i] = &.{};
            }
            pert_count_local = dim;
            vloc1_count_local = dim;
            rho1_core_count_local = dim;

            // Build vloc1 and rho1_core for ALL perturbations (cheap, serial)
            for (0..n_atoms) |ia| {
                for (0..3) |dir| {
                    const pidx = 3 * ia + dir;
                    vloc1_gs[pidx] = try perturbation.buildLocalPerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.ff_tables,
                    );
                    rho1_core_gs[pidx] = try perturbation.buildCorePerturbationQ(
                        alloc,
                        grid,
                        atoms[ia],
                        species,
                        dir,
                        q_cart,
                        gs.rho_core_tables,
                    );
                }
            }

            var next_index = std.atomic.Value(usize).init(0);
            var stop_flag = std.atomic.Value(u8).init(0);
            var worker_err: ?anyerror = null;
            var err_mutex = std.Io.Mutex.init;
            var log_mutex = std.Io.Mutex.init;

            var qshared = QPointPertShared{
                .alloc = alloc,
                .kpts = kpts,
                .dfpt_cfg = &pert_dfpt_cfg,
                .q_cart = q_cart,
                .grid = grid,
                .gs = gs,
                .species = species,
                .atoms = atoms,
                .ff_tables = gs.ff_tables,
                .rho_core_tables = gs.rho_core_tables,
                .pert_results_mk = pert_results_mk,
                .vloc1_gs = vloc1_gs,
                .rho1_core_gs = rho1_core_gs,
                .dim = n_irr_perts,
                .irr_pert_indices = null,
                .next_index = &next_index,
                .stop = &stop_flag,
                .err = &worker_err,
                .err_mutex = &err_mutex,
                .log_mutex = &log_mutex,
            };

            // Build irr_pert_indices for atom-level reduction
            const irr_pert_indices = try alloc.alloc(usize, n_irr_perts);
            defer alloc.free(irr_pert_indices);
            {
                var pi: usize = 0;
                for (irr_info.irr_atom_indices) |ia| {
                    for (0..3) |dir_idx| {
                        irr_pert_indices[pi] = 3 * ia + dir_idx;
                        pi += 1;
                    }
                }
            }
            qshared.irr_pert_indices = irr_pert_indices;

            var workers = try alloc.alloc(QPointPertWorker, pert_thread_count);
            defer alloc.free(workers);
            var threads_arr = try alloc.alloc(std.Thread, pert_thread_count - 1);
            defer alloc.free(threads_arr);

            for (0..pert_thread_count) |ti| {
                workers[ti] = .{ .shared = &qshared, .thread_index = ti };
            }

            for (0..pert_thread_count - 1) |ti| {
                threads_arr[ti] = try std.Thread.spawn(.{}, qpointPertWorkerFn, .{&workers[ti + 1]});
            }

            qpointPertWorkerFn(&workers[0]);

            for (threads_arr) |t| {
                t.join();
            }

            if (worker_err) |e| return e;
        }

        // Build dynamical matrix D(q) for this q-grid point
        const dyn_q = try buildQDynmatMultiK(
            alloc,
            kpts,
            pert_results_mk,
            vloc1_gs,
            rho1_core_gs,
            rho0_g,
            gs,
            ionic.charges,
            ionic.positions,
            cell_bohr,
            recip,
            volume,
            q_cart,
            grid,
            species,
            atoms,
            gs.ff_tables,
            gs.rho_core_tables,
            gs.rho_core,
            vxc_g,
            cfg.vdw,
            irr_info,
        );

        // Reconstruct non-irreducible columns from symmetry
        if (irr_info.n_irr_atoms < n_atoms) {
            dynmat_mod.reconstructDynmatColumnsComplex(dyn_q, n_atoms, irr_info, symops, sym_data.indsym, sym_data.tnons_shift, cell_bohr, qf);
        }

        dynmat_grid[iq] = dyn_q;
        dynmat_count = iq + 1;

        logDfpt("dfpt_ifc: q_grid[{d}] D(q) computed\n", .{iq});
    }

    // =============================================================
    // Phase 2: Compute IFC: C(R) = FT[D(q)]
    // =============================================================
    logDfpt("dfpt_ifc: computing IFC from {d} q-grid points\n", .{n_qgrid});

    // Cast dynmat_grid to const slices for computeIFC
    var dynmat_const = try alloc.alloc([]const math.Complex, n_qgrid);
    defer alloc.free(dynmat_const);
    for (0..n_qgrid) |i| {
        dynmat_const[i] = dynmat_grid[i];
    }

    var ifc_data = try ifc_mod.computeIFC(alloc, dynmat_const, q_frac_grid, qgrid, n_atoms);
    defer ifc_data.deinit(alloc);

    // Apply ASR in IFC space
    ifc_mod.applyASR(&ifc_data);
    logDfpt("dfpt_ifc: IFC ASR applied\n", .{});

    // =============================================================
    // Phase 3: Generate q-path and interpolate
    // =============================================================
    const npoints_per_seg = cfg.dfpt.qpath_npoints;

    // Use custom q-path if specified, otherwise FCC default
    var q_points_frac: []math.Vec3 = undefined;
    var q_points_cart: []math.Vec3 = undefined;
    var qpath_distances: []f64 = undefined;
    var qpath_labels: [][]const u8 = undefined;
    var qpath_label_positions: []usize = undefined;

    if (cfg.dfpt.qpath.len >= 2) {
        const qpath = try generateQPathFromConfig(alloc, cfg.dfpt.qpath, npoints_per_seg, recip);
        q_points_frac = qpath.q_points_frac;
        q_points_cart = qpath.q_points_cart;
        qpath_distances = qpath.distances;
        qpath_labels = qpath.labels;
        qpath_label_positions = qpath.label_positions;
    } else {
        const qpath = try generateFccQPath(alloc, recip, npoints_per_seg);
        q_points_frac = qpath.q_points_frac;
        q_points_cart = qpath.q_points_cart;
        qpath_distances = qpath.distances;
        qpath_labels = qpath.labels;
        qpath_label_positions = qpath.label_positions;
    }
    defer alloc.free(q_points_frac);
    defer alloc.free(q_points_cart);

    const n_q = q_points_cart.len;
    logDfpt("dfpt_ifc: interpolating {d} q-path points\n", .{n_q});

    // Allocate result arrays
    var frequencies = try alloc.alloc([]f64, n_q);
    var freq_count: usize = 0;
    errdefer {
        for (0..freq_count) |i| alloc.free(frequencies[i]);
        alloc.free(frequencies);
    }

    for (0..n_q) |iq| {
        const qf = q_points_frac[iq];
        const q_norm = math.Vec3.norm(q_points_cart[iq]);

        // Interpolate D(q') from IFC
        const dyn_interp = try ifc_mod.interpolate(alloc, &ifc_data, qf);
        defer alloc.free(dyn_interp);

        // Apply ASR at Gamma
        if (q_norm < 1e-10) {
            dynmat_mod.applyASRComplex(dyn_interp, n_atoms);
        }

        // Mass-weight
        dynmat_mod.massWeightComplex(dyn_interp, n_atoms, ionic.masses);

        // Diagonalize
        var result_q = try dynmat_mod.diagonalizeComplex(alloc, dyn_interp, dim);
        defer result_q.deinit(alloc);

        frequencies[iq] = try alloc.alloc(f64, dim);
        @memcpy(frequencies[iq], result_q.frequencies_cm1);
        freq_count = iq + 1;

        if (iq % 10 == 0 or iq == n_q - 1) {
            logDfpt("dfpt_ifc: q[{d}] freqs:", .{iq});
            for (result_q.frequencies_cm1) |f| {
                logDfpt(" {d:.1}", .{f});
            }
            logDfpt("\n", .{});
        }
    }

    // =============================================================
    // Phase 4 (optional): Phonon DOS from IFC interpolation
    // =============================================================
    if (cfg.dfpt.dos_qmesh) |dos_qmesh| {
        const phonon_dos_mod = @import("phonon_dos.zig");
        logDfpt("dfpt_ifc: computing phonon DOS on {d}x{d}x{d} mesh\n", .{ dos_qmesh[0], dos_qmesh[1], dos_qmesh[2] });

        var pdos = try phonon_dos_mod.computePhononDos(
            alloc,
            &ifc_data,
            ionic.masses,
            n_atoms,
            dos_qmesh,
            cfg.dfpt.dos_sigma,
            cfg.dfpt.dos_nbin,
        );
        defer pdos.deinit(alloc);

        // Write to out_dir
        var out_dir = try std.fs.cwd().openDir(cfg.out_dir, .{});
        defer out_dir.close();
        try phonon_dos_mod.writePhononDosCsv(out_dir, pdos);
        logDfpt("dfpt_ifc: phonon DOS written to phonon_dos.csv\n", .{});
    }

    return PhononBandResult{
        .distances = qpath_distances,
        .frequencies = frequencies,
        .n_modes = dim,
        .n_q = n_q,
        .labels = qpath_labels,
        .label_positions = qpath_label_positions,
    };
}
