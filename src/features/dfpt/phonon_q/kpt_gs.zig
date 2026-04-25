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
const log_dfpt = dfpt.log_dfpt;
const log_dfpt_info = dfpt.log_dfpt_info;

const Grid = scf_mod.Grid;

const OccupiedStates = struct {
    eigenvalues: []f64,
    wavefunctions: [][]math.Complex,
    wavefunctions_const: [][]const math.Complex,

    fn deinit(self: *OccupiedStates, alloc: std.mem.Allocator) void {
        for (self.wavefunctions) |w| alloc.free(w);
        alloc.free(self.wavefunctions);
        alloc.free(self.wavefunctions_const);
        alloc.free(self.eigenvalues);
    }
};

const RotatedBasis = struct {
    basis: plane_wave.Basis,
    sk_unwrapped: math.Vec3,
};

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

fn init_apply_context(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    grid: Grid,
    basis: *const plane_wave.Basis,
    local_r: []const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    volume: f64,
) !*scf_mod.ApplyContext {
    const apply_ctx = try alloc.create(scf_mod.ApplyContext);
    errdefer alloc.destroy(apply_ctx);

    apply_ctx.* = try scf_mod.ApplyContext.init_with_workspaces(
        alloc,
        io,
        grid,
        @constCast(basis.gvecs),
        local_r,
        null,
        species,
        atoms,
        1.0 / volume,
        true,
        null,
        null,
        cfg.scf.fft_backend,
        1,
    );
    return apply_ctx;
}

fn solve_occupied_states(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    gs: *const GroundState,
    basis: *const plane_wave.Basis,
    apply_ctx: *scf_mod.ApplyContext,
) !OccupiedStates {
    const n_pw = basis.gvecs.len;
    const diag = try alloc.alloc(f64, n_pw);
    defer alloc.free(diag);

    for (basis.gvecs, 0..) |g, i| diag[i] = g.kinetic;

    const n_occ = gs.n_occ;
    const op = iterative.Operator{
        .n = n_pw,
        .ctx = @ptrCast(apply_ctx),
        .apply = &scf_mod.apply_hamiltonian,
        .apply_batch = &scf_mod.apply_hamiltonian_batched,
    };
    var eig = try iterative.hermitian_eigen_decomp_iterative(
        alloc,
        cfg.linalg_backend,
        op,
        diag,
        @max(n_occ + 2, @as(usize, 8)),
        .{ .max_iter = 100, .tol = 1e-8, .init_diagonal = true },
    );
    defer eig.deinit(alloc);

    const eigenvalues = try alloc.alloc(f64, n_occ);
    errdefer alloc.free(eigenvalues);
    @memcpy(eigenvalues, eig.values[0..n_occ]);

    const wavefunctions = try alloc.alloc([]math.Complex, n_occ);
    var wf_built: usize = 0;
    errdefer {
        for (0..wf_built) |i| alloc.free(wavefunctions[i]);
        alloc.free(wavefunctions);
    }

    const wavefunctions_const = try alloc.alloc([]const math.Complex, n_occ);
    errdefer alloc.free(wavefunctions_const);

    for (0..n_occ) |n| {
        wavefunctions[n] = try alloc.alloc(math.Complex, n_pw);
        @memcpy(wavefunctions[n], eig.vectors[n * n_pw .. (n + 1) * n_pw]);
        wavefunctions_const[n] = wavefunctions[n];
        wf_built = n + 1;
    }

    return .{
        .eigenvalues = eigenvalues,
        .wavefunctions = wavefunctions,
        .wavefunctions_const = wavefunctions_const,
    };
}

fn solve_direct_kpoint(
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
    kp: symmetry.KPoint,
) !KPointGsData {
    var basis_k = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, kp.k_cart);
    errdefer basis_k.deinit(alloc);

    var map_k = try scf_mod.PwGridMap.init(alloc, @constCast(basis_k.gvecs), grid);
    errdefer map_k.deinit(alloc);

    const apply_ctx_k = try init_apply_context(
        alloc,
        io,
        cfg,
        grid,
        &basis_k,
        local_r,
        species,
        atoms,
        volume,
    );
    errdefer {
        apply_ctx_k.deinit(alloc);
        alloc.destroy(apply_ctx_k);
    }

    var occupied = try solve_occupied_states(alloc, cfg, gs, &basis_k, apply_ctx_k);
    errdefer occupied.deinit(alloc);

    return .{
        .k_frac = kp.k_frac,
        .k_cart = kp.k_cart,
        .weight = kp.weight,
        .n_occ = gs.n_occ,
        .n_pw_k = basis_k.gvecs.len,
        .basis_k = basis_k,
        .map_k = map_k,
        .apply_ctx_k = apply_ctx_k,
        .eigenvalues_k = occupied.eigenvalues,
        .wavefunctions_k = occupied.wavefunctions,
        .wavefunctions_k_const = occupied.wavefunctions_const,
    };
}

fn solve_ibz_kpoint(
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
    k_cart: math.Vec3,
) !IbzKData {
    var basis = try plane_wave.generate(alloc, recip, cfg.scf.ecut_ry, k_cart);
    errdefer basis.deinit(alloc);

    const apply_ctx = try init_apply_context(
        alloc,
        io,
        cfg,
        grid,
        &basis,
        local_r,
        species,
        atoms,
        volume,
    );
    errdefer {
        apply_ctx.deinit(alloc);
        alloc.destroy(apply_ctx);
    }

    var occupied = try solve_occupied_states(alloc, cfg, gs, &basis, apply_ctx);
    errdefer occupied.deinit(alloc);

    apply_ctx.deinit(alloc);
    alloc.destroy(apply_ctx);

    return .{
        .basis = basis,
        .eigenvalues = occupied.eigenvalues,
        .wavefunctions = occupied.wavefunctions,
        .wavefunctions_const = occupied.wavefunctions_const,
        .n_occ = gs.n_occ,
    };
}

fn build_rotated_basis(
    alloc: std.mem.Allocator,
    ibz_basis: plane_wave.Basis,
    symop: symmetry.SymOp,
    time_reversed: bool,
    ibz_frac: math.Vec3,
    sk_frac: math.Vec3,
    sk_cart: math.Vec3,
    recip: math.Mat3,
) !RotatedBasis {
    const b1 = recip.row(0);
    const b2 = recip.row(1);
    const b3 = recip.row(2);
    const sk_unwrapped = symop.k_rot.mul_vec(ibz_frac);
    const sign_i: i32 = if (time_reversed) -1 else 1;
    const sign_f: f64 = if (time_reversed) -1.0 else 1.0;
    const delta_hkl = [3]i32{
        @as(i32, @intFromFloat(std.math.round(sign_f * sk_unwrapped.x - sk_frac.x))),
        @as(i32, @intFromFloat(std.math.round(sign_f * sk_unwrapped.y - sk_frac.y))),
        @as(i32, @intFromFloat(std.math.round(sign_f * sk_unwrapped.z - sk_frac.z))),
    };

    const gvecs = try alloc.alloc(plane_wave.GVector, ibz_basis.gvecs.len);
    errdefer alloc.free(gvecs);

    for (ibz_basis.gvecs, 0..) |gv, gi| {
        const kr = symop.k_rot.m;
        const rot0 = kr[0][0] * gv.h + kr[0][1] * gv.k + kr[0][2] * gv.l;
        const rot1 = kr[1][0] * gv.h + kr[1][1] * gv.k + kr[1][2] * gv.l;
        const rot2 = kr[2][0] * gv.h + kr[2][1] * gv.k + kr[2][2] * gv.l;
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
        gvecs[gi] = .{
            .h = h1,
            .k = k1,
            .l = l1,
            .cart = g_cart,
            .kpg = kpg,
            .kinetic = math.Vec3.dot(kpg, kpg),
        };
    }

    return .{
        .basis = .{ .gvecs = gvecs },
        .sk_unwrapped = sk_unwrapped,
    };
}

fn expand_ibz_kpoint(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    local_r: []const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    symops: []const symmetry.SymOp,
    ibz_frac: math.Vec3,
    total: usize,
    ibz: *const IbzKData,
    full_kp: symmetry.KPoint,
) !KPointGsData {
    var setup = try init_expanded_kpoint_setup(
        alloc,
        io,
        cfg,
        local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
        symops,
        ibz_frac,
        ibz,
        full_kp,
    );
    errdefer setup.deinit(alloc);

    const rot_result = try wfn_rot.rotate_wavefunctions_in_place(
        alloc,
        ibz.wavefunctions_const,
        ibz.basis,
        setup.symop,
        setup.sk_unwrapped,
        setup.time_reversed,
    );
    errdefer {
        for (rot_result.wfn) |w| alloc.free(w);
        alloc.free(rot_result.wfn);
        alloc.free(rot_result.wfn_const);
    }

    const eigenvalues_k = try copy_kpoint_eigenvalues(alloc, ibz.eigenvalues, ibz.n_occ);
    errdefer alloc.free(eigenvalues_k);

    return .{
        .k_frac = full_kp.k_frac,
        .k_cart = full_kp.k_cart,
        .weight = 1.0 / @as(f64, @floatFromInt(total)),
        .n_occ = ibz.n_occ,
        .n_pw_k = ibz.basis.gvecs.len,
        .basis_k = setup.basis_k,
        .map_k = setup.map_k,
        .apply_ctx_k = setup.apply_ctx_k,
        .eigenvalues_k = eigenvalues_k,
        .wavefunctions_k = rot_result.wfn,
        .wavefunctions_k_const = rot_result.wfn_const,
    };
}

const ExpandedKPointSetup = struct {
    basis_k: plane_wave.Basis,
    sk_unwrapped: math.Vec3,
    symop: symmetry.SymOp,
    time_reversed: bool,
    map_k: scf_mod.PwGridMap,
    apply_ctx_k: *scf_mod.ApplyContext,

    fn deinit(self: *ExpandedKPointSetup, alloc: std.mem.Allocator) void {
        self.basis_k.deinit(alloc);
        self.map_k.deinit(alloc);
        self.apply_ctx_k.deinit(alloc);
        alloc.destroy(self.apply_ctx_k);
    }
};

fn init_expanded_kpoint_setup(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    local_r: []const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    symops: []const symmetry.SymOp,
    ibz_frac: math.Vec3,
    ibz: *const IbzKData,
    full_kp: symmetry.KPoint,
) !ExpandedKPointSetup {
    const found = find_symop_for_kpoint(symops, ibz_frac, full_kp.k_frac, 1e-6);
    var rotated = try build_rotated_basis(
        alloc,
        ibz.basis,
        found.symop,
        found.time_reversed,
        ibz_frac,
        full_kp.k_frac,
        full_kp.k_cart,
        recip,
    );
    errdefer rotated.basis.deinit(alloc);

    var map_k = try scf_mod.PwGridMap.init(alloc, @constCast(rotated.basis.gvecs), grid);
    errdefer map_k.deinit(alloc);

    const apply_ctx_k = try init_apply_context(
        alloc,
        io,
        cfg,
        grid,
        &rotated.basis,
        local_r,
        species,
        atoms,
        volume,
    );
    errdefer {
        apply_ctx_k.deinit(alloc);
        alloc.destroy(apply_ctx_k);
    }
    return .{
        .basis_k = rotated.basis,
        .sk_unwrapped = rotated.sk_unwrapped,
        .symop = found.symop,
        .time_reversed = found.time_reversed,
        .map_k = map_k,
        .apply_ctx_k = apply_ctx_k,
    };
}

fn copy_kpoint_eigenvalues(
    alloc: std.mem.Allocator,
    eigenvalues: []const f64,
    n_occ: usize,
) ![]f64 {
    const copied = try alloc.alloc(f64, n_occ);
    @memcpy(copied, eigenvalues);
    return copied;
}

/// Prepare full-BZ k-point ground-state data for DFPT.
/// Generates the full Monkhorst-Pack mesh (no symmetry reduction) and
/// solves the eigenvalue problem at each k-point using the SCF potential.
/// This is equivalent to ABINIT's kptopt=3 for DFPT calculations.
pub fn prepare_full_bz_kpoints(
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
    const kmesh = cfg.scf.kmesh;
    const shift = math.Vec3{
        .x = cfg.scf.kmesh_shift[0],
        .y = cfg.scf.kmesh_shift[1],
        .z = cfg.scf.kmesh_shift[2],
    };
    const full_kpts = try mesh_mod.generate_kmesh(alloc, kmesh, recip, shift);
    defer alloc.free(full_kpts);

    const n_kpts = full_kpts.len;
    log_dfpt_info(
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
        kgs_data[ik] = try solve_direct_kpoint(
            alloc,
            io,
            cfg,
            gs,
            local_r,
            species,
            atoms,
            recip,
            volume,
            grid,
            kp,
        );
        built = ik + 1;

        if (ik == 0 or (ik + 1) % 16 == 0 or ik + 1 == n_kpts) {
            log_dfpt("dfpt_band: prepared k-point {d}/{d}\n", .{ ik + 1, n_kpts });
        }
    }

    return kgs_data;
}

/// Prepare full-BZ k-point ground-state data using IBZ expansion.
/// Solves eigenvalue problem only at IBZ k-points, then rotates wavefunctions
/// to the full BZ using symmetry operations. This ensures ε tensor isotropy
/// for cubic systems and reduces computation by the symmetry factor.
pub fn prepare_full_bz_kpoints_from_ibz(
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
    var ctx = try init_ibz_expansion_context(alloc, cfg, atoms, cell_bohr, recip);
    defer ctx.deinit(alloc);

    const ibz_data = try solve_ibz_ground_states(
        alloc,
        io,
        cfg,
        gs,
        local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
        ctx.mapping.ibz_kpoints,
    );
    defer {
        for (ibz_data) |*data| data.deinit(alloc);
        alloc.free(ibz_data);
    }

    return try expand_full_bz_from_ibz(
        alloc,
        io,
        cfg,
        local_r,
        species,
        atoms,
        recip,
        volume,
        grid,
        ctx.symops,
        ctx.total,
        &ctx.mapping,
        ibz_data,
        ctx.full_kpts,
    );
}

const IbzExpansionContext = struct {
    total: usize,
    symops: []const symmetry.SymOp,
    kmesh_ops: []const reduction.KmeshOp,
    mapping: reduction.KmeshMapping,
    full_kpts: []const symmetry.KPoint,

    fn deinit(self: *IbzExpansionContext, alloc: std.mem.Allocator) void {
        alloc.free(self.symops);
        alloc.free(self.kmesh_ops);
        self.mapping.deinit(alloc);
        alloc.free(self.full_kpts);
    }
};

fn init_ibz_expansion_context(
    alloc: std.mem.Allocator,
    cfg: config_mod.Config,
    atoms: []const hamiltonian.AtomData,
    cell_bohr: math.Mat3,
    recip: math.Mat3,
) !IbzExpansionContext {
    const kmesh = cfg.scf.kmesh;
    const shift = math.Vec3{
        .x = cfg.scf.kmesh_shift[0],
        .y = cfg.scf.kmesh_shift[1],
        .z = cfg.scf.kmesh_shift[2],
    };
    const total = kmesh[0] * kmesh[1] * kmesh[2];
    const symops = try symmetry_mod.get_symmetry_ops(alloc, cell_bohr, atoms, 1e-5);
    errdefer alloc.free(symops);
    log_dfpt_info("dfpt_ibz: {d} symmetry operations\n", .{symops.len});

    const kmesh_ops = try reduction.filter_sym_ops_for_kmesh(alloc, symops, kmesh, shift, 1e-8);
    errdefer alloc.free(kmesh_ops);

    var mapping = try reduction.reduce_kmesh_with_mapping(
        alloc,
        kmesh,
        shift,
        kmesh_ops,
        recip,
        true,
    );
    errdefer mapping.deinit(alloc);
    log_dfpt_info(
        "dfpt_ibz: {d} IBZ k-points -> {d} full BZ k-points\n",
        .{ mapping.ibz_kpoints.len, total },
    );

    const full_kpts = try mesh_mod.generate_kmesh(alloc, kmesh, recip, shift);
    errdefer alloc.free(full_kpts);
    return .{
        .total = total,
        .symops = symops,
        .kmesh_ops = kmesh_ops,
        .mapping = mapping,
        .full_kpts = full_kpts,
    };
}

fn solve_ibz_ground_states(
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
    ibz_kpoints: []const symmetry.KPoint,
) ![]IbzKData {
    const ibz_data = try alloc.alloc(IbzKData, ibz_kpoints.len);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| ibz_data[i].deinit(alloc);
        alloc.free(ibz_data);
    }
    for (ibz_kpoints, 0..) |kp, i_ibz| {
        ibz_data[i_ibz] = try solve_ibz_kpoint(
            alloc,
            io,
            cfg,
            gs,
            local_r,
            species,
            atoms,
            recip,
            volume,
            grid,
            kp.k_cart,
        );
        built = i_ibz + 1;
        log_dfpt(
            "dfpt_ibz: solved IBZ k-point {d}/{d} (n_pw={d})\n",
            .{ i_ibz + 1, ibz_kpoints.len, ibz_data[i_ibz].basis.gvecs.len },
        );
    }
    return ibz_data;
}

fn expand_full_bz_from_ibz(
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: config_mod.Config,
    local_r: []const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    grid: Grid,
    symops: []const symmetry.SymOp,
    total: usize,
    mapping: *const reduction.KmeshMapping,
    ibz_data: []const IbzKData,
    full_kpts: []const symmetry.KPoint,
) ![]KPointGsData {
    var kgs_data = try alloc.alloc(KPointGsData, total);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| kgs_data[i].deinit(alloc);
        alloc.free(kgs_data);
    }
    for (0..total) |i_full| {
        const i_ibz = mapping.full_to_ibz[i_full];
        kgs_data[i_full] = try expand_ibz_kpoint(
            alloc,
            io,
            cfg,
            local_r,
            species,
            atoms,
            recip,
            volume,
            grid,
            symops,
            mapping.ibz_kpoints[i_ibz].k_frac,
            total,
            &ibz_data[i_ibz],
            full_kpts[i_full],
        );
        built = i_full + 1;
        if (i_full == 0 or (i_full + 1) % 16 == 0 or i_full + 1 == total) {
            log_dfpt("dfpt_ibz: expanded k-point {d}/{d}\n", .{ i_full + 1, total });
        }
    }
    return kgs_data;
}

const SymopMatch = struct {
    symop: symmetry.SymOp,
    time_reversed: bool,
};

/// Find the symmetry operation that maps k_ibz to k_full: k_rot * k_ibz ≡ ±k_full (mod 1).
/// This is necessary because the grid-based map_index in reduce_kmesh may assign incorrect
/// symops for non-orthogonal reciprocal lattices.
fn find_symop_for_kpoint(
    ops: []const symmetry.SymOp,
    k_ibz_frac: math.Vec3,
    k_full_frac: math.Vec3,
    tol: f64,
) SymopMatch {
    // Try without time reversal first
    for (ops) |op| {
        const sk = op.k_rot.mul_vec(k_ibz_frac);
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
        const sk = op.k_rot.mul_vec(k_ibz_frac);
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
