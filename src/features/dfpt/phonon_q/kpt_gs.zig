//! K-point ground-state data for DFPT (q-independent).
//!
//! Precomputes PW basis, eigenvalues, and occupied wavefunctions at every
//! k-point of the full BZ (optionally via IBZ expansion using symmetry).
//! This data is expensive to build, so it is constructed once per SCF
//! potential and reused across every q-point of the phonon band structure.

const std = @import("std");
const math = @import("../../math/math.zig");
const hamiltonian = @import("../../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../../scf/scf.zig");
const plane_wave = @import("../../plane_wave/basis.zig");
const config_mod = @import("../../config/config.zig");
const iterative = @import("../../linalg/iterative.zig");
const symmetry_mod = @import("../../symmetry/symmetry.zig");
const symmetry = @import("../../symmetry/symmetry.zig");
const mesh_mod = @import("../../kpoints/mesh.zig");
const reduction = @import("../../kpoints/reduction.zig");
const wfn_rot = @import("../../symmetry/wavefunction_rotation.zig");

const dfpt = @import("../dfpt.zig");
const GroundState = dfpt.GroundState;
const logDfpt = dfpt.logDfpt;
const logDfptInfo = dfpt.logDfptInfo;

const Grid = scf_mod.Grid;

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
    species: []const hamiltonian.SpeciesEntry,
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
    logDfptInfo(
        "dfpt_band: generating full BZ k-mesh: {d}x{d}x{d} = {d} k-points\n",
        .{ kmesh[0], kmesh[1], kmesh[2], n_kpts },
    );

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
    species: []const hamiltonian.SpeciesEntry,
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
    logDfptInfo("dfpt_ibz: {d} symmetry operations\n", .{symops.len});

    // Filter symmetry ops compatible with k-mesh
    const kmesh_ops = try reduction.filterSymOpsForKmesh(alloc, symops, kmesh, shift, 1e-8);
    defer alloc.free(kmesh_ops);

    // Get IBZ→full BZ mapping
    var mapping = try reduction.reduceKmeshWithMapping(alloc, kmesh, shift, kmesh_ops, recip, true);
    defer mapping.deinit(alloc);
    const n_ibz = mapping.ibz_kpoints.len;
    logDfptInfo("dfpt_ibz: {d} IBZ k-points -> {d} full BZ k-points\n", .{ n_ibz, total });

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
            const kr = symop.k_rot.m;
            const rot0 = kr[0][0] * h0 + kr[0][1] * k0 + kr[0][2] * l0;
            const rot1 = kr[1][0] * h0 + kr[1][1] * k0 + kr[1][2] * l0;
            const rot2 = kr[2][0] * h0 + kr[2][1] * k0 + kr[2][2] * l0;
            const h1 = sign_i * rot0 + delta_hkl[0];
            const k1 = sign_i * rot1 + delta_hkl[1];
            const l1 = sign_i * rot2 + delta_hkl[2];
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
