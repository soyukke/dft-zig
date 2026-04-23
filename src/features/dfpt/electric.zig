//! Electric field response: ddk perturbation for ε∞.
//!
//! Solves ddk Sternheimer at ALL k-points in the BZ.
//!
//! Self-consistent dielectric tensor (ABINIT rfelfd=3 equivalent):
//!   1) du/dk = -(H-ε)^{-1} P_c dH/dk |u⟩  (ddk Sternheimer)
//!   2) For each direction β, SCF loop:
//!      a) ψ¹_E = -(H-ε)^{-1} P_c (+i du/dk_β + V^(1)_Hxc |ψ⟩)
//!      b) ρ^(1) = Σ_k w_k × 2(spin) × 2 Re[ψ_k*(r) × ψ¹_E_k(r)]
//!      c) V^(1)_H = 8π ρ^(1)(G)/|G|²,  V^(1)_xc = f_xc × ρ^(1)(r)
//!      d) Mix V^(1)_Hxc, check convergence
//!   3) ε_αβ = δ_αβ - (16π/Ω) × occ × Σ w_k Re[<+i du/dk_α | ψ¹_{E,β}>]

const std = @import("std");
const math = @import("../math/math.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const scf_mod = @import("../scf/scf.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const config_mod = @import("../config/config.zig");

const dfpt = @import("dfpt.zig");
const sternheimer = dfpt.sternheimer;
const perturbation = dfpt.perturbation;
const phonon_q = dfpt.phonon_q;

const GroundState = dfpt.GroundState;
const DfptConfig = dfpt.DfptConfig;
const KPointGsData = phonon_q.KPointGsData;
const Grid = scf_mod.Grid;

const logDfpt = dfpt.logDfpt;
const logDfptInfo = dfpt.logDfptInfo;

pub const DielectricResult = struct {
    epsilon: [3][3]f64,
};

const BandSet = [][]math.Complex;
const DirectionBandSet = [3]BandSet;

const DdkResponse = struct {
    values: []DirectionBandSet,

    fn deinit(self: *DdkResponse, alloc: std.mem.Allocator) void {
        for (self.values) |dirs| deinitDirectionBandSet(alloc, dirs);
        alloc.free(self.values);
    }
};

const Psi0Cache = struct {
    values: []BandSet,

    fn deinit(self: *Psi0Cache, alloc: std.mem.Allocator) void {
        for (self.values) |bands| deinitBandSet(alloc, bands);
        alloc.free(self.values);
    }
};

const EfieldResponse = struct {
    values: []BandSet,

    fn deinit(self: *EfieldResponse, alloc: std.mem.Allocator) void {
        for (self.values) |bands| deinitBandSet(alloc, bands);
        alloc.free(self.values);
    }
};

const DdkNonlocalBuffers = struct {
    nl_out: []math.Complex,
    nl_phase: []math.Complex,
    nl_c1: []math.Complex,
    nl_c2: []math.Complex,
    nl_dc1: []math.Complex,
    nl_dc2: []math.Complex,

    fn init(
        alloc: std.mem.Allocator,
        n_pw: usize,
        max_m_total: usize,
    ) !DdkNonlocalBuffers {
        return .{
            .nl_out = try alloc.alloc(math.Complex, n_pw),
            .nl_phase = try alloc.alloc(math.Complex, n_pw),
            .nl_c1 = try alloc.alloc(math.Complex, @max(max_m_total, 1)),
            .nl_c2 = try alloc.alloc(math.Complex, @max(max_m_total, 1)),
            .nl_dc1 = try alloc.alloc(math.Complex, @max(max_m_total, 1)),
            .nl_dc2 = try alloc.alloc(math.Complex, @max(max_m_total, 1)),
        };
    }

    fn deinit(self: *DdkNonlocalBuffers, alloc: std.mem.Allocator) void {
        alloc.free(self.nl_out);
        alloc.free(self.nl_phase);
        alloc.free(self.nl_c1);
        alloc.free(self.nl_c2);
        alloc.free(self.nl_dc1);
        alloc.free(self.nl_dc2);
    }
};

const EfieldMixState = struct {
    best_vresid: f64 = std.math.inf(f64),
    best_v1: ?[]math.Complex = null,
    pulay_active_since: usize,
    force_converge: bool = false,

    fn init(dfpt_cfg: DfptConfig) EfieldMixState {
        return .{ .pulay_active_since = dfpt_cfg.pulay_start };
    }

    fn deinit(self: *EfieldMixState, alloc: std.mem.Allocator) void {
        if (self.best_v1) |v| alloc.free(v);
    }
};

const EfieldMixOutcome = enum {
    keep_iterating,
    converged,
};

/// Compute ε∞ by solving ddk Sternheimer at all k-points,
/// then self-consistently solving the efield Sternheimer with V^(1)_Hxc.
pub fn computeDielectricAllK(
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
) !DielectricResult {
    const grid = gs.grid;
    const dfpt_cfg = DfptConfig.fromConfig(cfg);
    const kgs = try phonon_q.prepareFullBZKpointsFromIBZ(
        alloc,
        io,
        cfg,
        gs,
        local_r,
        species,
        atoms,
        cell_bohr,
        recip,
        volume,
        grid,
    );
    defer deinitKgs(alloc, kgs);

    logDfptInfo("ddk: {d} k-points in full BZ\n", .{kgs.len});
    const radial_tables = try buildRadialTables(alloc, species, kgs);
    defer deinitRadialTables(alloc, radial_tables);

    var ddk = try solveDdkAllK(alloc, dfpt_cfg, kgs, atoms, radial_tables, volume);
    defer ddk.deinit(alloc);

    var psi0_cache = try cachePsi0RealSpace(alloc, grid, kgs);
    defer psi0_cache.deinit(alloc);

    var epsilon = try computeDielectricTensor(
        alloc,
        grid,
        gs,
        dfpt_cfg,
        kgs,
        ddk.values,
        psi0_cache.values,
        volume,
    );
    epsilon = try symmetrizeDielectricTensor(alloc, epsilon, cell_bohr, atoms, recip);
    return .{ .epsilon = epsilon };
}

fn deinitKgs(alloc: std.mem.Allocator, kgs: []KPointGsData) void {
    for (@constCast(kgs)) |*k| k.deinit(alloc);
    alloc.free(kgs);
}

fn deinitBandSet(alloc: std.mem.Allocator, bands: BandSet) void {
    for (bands) |band| alloc.free(band);
    alloc.free(bands);
}

fn deinitDirectionBandSet(
    alloc: std.mem.Allocator,
    directions: DirectionBandSet,
) void {
    for (0..3) |dir| deinitBandSet(alloc, directions[dir]);
}

fn deinitRadialTables(
    alloc: std.mem.Allocator,
    radial_tables: []nonlocal.RadialTableSet,
) void {
    for (radial_tables) |*table| table.deinit(alloc);
    alloc.free(radial_tables);
}

fn buildRadialTables(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    kgs: []const KPointGsData,
) ![]nonlocal.RadialTableSet {
    var g_max: f64 = 0.0;
    for (kgs) |kg| {
        for (kg.basis_k.gvecs) |gv| {
            g_max = @max(g_max, math.Vec3.norm(gv.kpg));
        }
    }
    g_max += 1.0;

    const radial_tables = try alloc.alloc(nonlocal.RadialTableSet, species.len);
    errdefer alloc.free(radial_tables);

    for (species, 0..) |entry, si| {
        radial_tables[si] = try nonlocal.RadialTableSet.init(
            alloc,
            entry.upf.beta,
            entry.upf.r,
            entry.upf.rab,
            g_max,
        );
    }
    return radial_tables;
}

fn solveDdkAllK(
    alloc: std.mem.Allocator,
    dfpt_cfg: DfptConfig,
    kgs: []const KPointGsData,
    atoms: []const hamiltonian.AtomData,
    radial_tables: []const nonlocal.RadialTableSet,
    volume: f64,
) !DdkResponse {
    const values = try alloc.alloc(DirectionBandSet, kgs.len);
    var kpts_built: usize = 0;
    errdefer {
        for (0..kpts_built) |ik| deinitDirectionBandSet(alloc, values[ik]);
        alloc.free(values);
    }

    for (kgs, 0..) |kg, ik| {
        const nl_ctx_opt = kg.apply_ctx_k.nonlocal_ctx;
        const max_m = if (nl_ctx_opt) |ctx| ctx.max_m_total else 0;
        var buffers = try DdkNonlocalBuffers.init(alloc, kg.n_pw_k, max_m);
        defer buffers.deinit(alloc);

        var dirs_built: usize = 0;
        errdefer for (0..dirs_built) |dir| deinitBandSet(alloc, values[ik][dir]);

        for (0..3) |dir| {
            values[ik][dir] = try solveDdkDirection(
                alloc,
                dfpt_cfg,
                &kg,
                atoms,
                radial_tables,
                volume,
                dir,
                &buffers,
            );
            dirs_built = dir + 1;
        }

        if (ik == 0 or (ik + 1) % 8 == 0 or ik + 1 == kgs.len) {
            logDfpt("ddk: k-point {d}/{d} done\n", .{ ik + 1, kgs.len });
        }
        kpts_built = ik + 1;
    }

    return .{ .values = values };
}

fn solveDdkDirection(
    alloc: std.mem.Allocator,
    dfpt_cfg: DfptConfig,
    kg: *const KPointGsData,
    atoms: []const hamiltonian.AtomData,
    radial_tables: []const nonlocal.RadialTableSet,
    volume: f64,
    dir: usize,
    buffers: *DdkNonlocalBuffers,
) !BandSet {
    const psi1 = try alloc.alloc([]math.Complex, kg.n_occ);
    var bands_built: usize = 0;
    errdefer {
        for (psi1[0..bands_built]) |band| alloc.free(band);
        alloc.free(psi1);
    }

    for (0..kg.n_occ) |n| {
        const h1psi = try alloc.alloc(math.Complex, kg.n_pw_k);
        defer alloc.free(h1psi);

        for (0..kg.n_pw_k) |g| {
            const kpg_dir = perturbation.gComponent(kg.basis_k.gvecs[g].kpg, dir);
            h1psi[g] = math.complex.scale(kg.wavefunctions_k[n][g], 2.0 * kpg_dir);
        }

        if (kg.apply_ctx_k.nonlocal_ctx) |nl_ctx| {
            applyDdkNonlocal(
                kg.basis_k.gvecs,
                atoms,
                nl_ctx,
                dir,
                1.0 / volume,
                kg.wavefunctions_k[n],
                radial_tables,
                buffers.nl_out,
                buffers.nl_phase,
                buffers.nl_c1,
                buffers.nl_c2,
                buffers.nl_dc1,
                buffers.nl_dc2,
            );
            for (0..kg.n_pw_k) |g| {
                h1psi[g] = math.complex.add(h1psi[g], buffers.nl_out[g]);
            }
        }

        const rhs = try alloc.alloc(math.Complex, kg.n_pw_k);
        defer alloc.free(rhs);

        for (0..kg.n_pw_k) |g| rhs[g] = math.complex.scale(h1psi[g], -1.0);
        sternheimer.projectConduction(rhs, kg.wavefunctions_k_const, kg.n_occ);

        const result = try sternheimer.solve(
            alloc,
            kg.apply_ctx_k,
            rhs,
            kg.eigenvalues_k[n],
            kg.wavefunctions_k_const,
            kg.n_occ,
            kg.basis_k.gvecs,
            .{
                .tol = dfpt_cfg.sternheimer_tol,
                .max_iter = dfpt_cfg.sternheimer_max_iter,
                .alpha_shift = dfpt_cfg.alpha_shift,
            },
        );
        psi1[n] = result.psi1;
        bands_built = n + 1;
    }

    return psi1;
}

fn cachePsi0RealSpace(
    alloc: std.mem.Allocator,
    grid: Grid,
    kgs: []const KPointGsData,
) !Psi0Cache {
    const total = grid.count();
    const values = try alloc.alloc(BandSet, kgs.len);
    var kpts_built: usize = 0;
    errdefer {
        for (0..kpts_built) |ik| deinitBandSet(alloc, values[ik]);
        alloc.free(values);
    }

    for (kgs, 0..) |kg, ik| {
        values[ik] = try alloc.alloc([]math.Complex, kg.n_occ);
        var bands_built: usize = 0;
        errdefer {
            for (values[ik][0..bands_built]) |band| alloc.free(band);
            alloc.free(values[ik]);
        }

        for (0..kg.n_occ) |n| {
            values[ik][n] = try alloc.alloc(math.Complex, total);
            const work = try alloc.alloc(math.Complex, total);
            defer alloc.free(work);

            @memset(work, math.complex.init(0.0, 0.0));
            kg.map_k.scatter(kg.wavefunctions_k_const[n], work);
            try scf_mod.fftReciprocalToComplexInPlace(
                alloc,
                grid,
                work,
                values[ik][n],
                null,
            );
            bands_built = n + 1;
        }
        kpts_built = ik + 1;
    }

    return .{ .values = values };
}

fn computeDielectricTensor(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: *const GroundState,
    dfpt_cfg: DfptConfig,
    kgs: []const KPointGsData,
    psi1_ddk: []const DirectionBandSet,
    psi0_r_cache: []const BandSet,
    volume: f64,
) ![3][3]f64 {
    var epsilon: [3][3]f64 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    for (0..3) |beta| {
        const column = try computeDielectricColumn(
            alloc,
            grid,
            gs,
            dfpt_cfg,
            kgs,
            psi1_ddk,
            psi0_r_cache,
            volume,
            beta,
        );
        for (0..3) |alpha| epsilon[alpha][beta] = column[alpha];
    }
    for (0..3) |i| epsilon[i][i] += 1.0;
    return epsilon;
}

fn computeDielectricColumn(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: *const GroundState,
    dfpt_cfg: DfptConfig,
    kgs: []const KPointGsData,
    psi1_ddk: []const DirectionBandSet,
    psi0_r_cache: []const BandSet,
    volume: f64,
    beta: usize,
) ![3]f64 {
    logDfptInfo("efield SCF: direction β={d}\n", .{beta});

    const v1_hxc_g = try alloc.alloc(math.Complex, grid.count());
    defer alloc.free(v1_hxc_g);

    @memset(v1_hxc_g, math.complex.init(0.0, 0.0));

    var psi1_ef = try initEfieldResponse(alloc, kgs);
    defer psi1_ef.deinit(alloc);

    var pulay = scf_mod.ComplexPulayMixer.init(alloc, dfpt_cfg.pulay_history);
    defer pulay.deinit();

    var mix_state = EfieldMixState.init(dfpt_cfg);
    defer mix_state.deinit(alloc);

    try runEfieldScfDirection(
        alloc,
        grid,
        gs,
        dfpt_cfg,
        kgs,
        psi1_ddk,
        psi0_r_cache,
        beta,
        v1_hxc_g,
        psi1_ef.values,
        &pulay,
        &mix_state,
    );
    return accumulateDielectricColumn(kgs, psi1_ddk, psi1_ef.values, volume);
}

fn initEfieldResponse(
    alloc: std.mem.Allocator,
    kgs: []const KPointGsData,
) !EfieldResponse {
    const values = try alloc.alloc(BandSet, kgs.len);
    var kpts_built: usize = 0;
    errdefer {
        for (0..kpts_built) |ik| deinitBandSet(alloc, values[ik]);
        alloc.free(values);
    }

    for (kgs, 0..) |kg, ik| {
        values[ik] = try alloc.alloc([]math.Complex, kg.n_occ);
        var bands_built: usize = 0;
        errdefer {
            for (values[ik][0..bands_built]) |band| alloc.free(band);
            alloc.free(values[ik]);
        }

        for (0..kg.n_occ) |n| {
            values[ik][n] = try alloc.alloc(math.Complex, kg.n_pw_k);
            @memset(values[ik][n], math.complex.init(0.0, 0.0));
            bands_built = n + 1;
        }
        kpts_built = ik + 1;
    }

    return .{ .values = values };
}

fn runEfieldScfDirection(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: *const GroundState,
    dfpt_cfg: DfptConfig,
    kgs: []const KPointGsData,
    psi1_ddk: []const DirectionBandSet,
    psi0_r_cache: []const BandSet,
    beta: usize,
    v1_hxc_g: []math.Complex,
    psi1_ef: []BandSet,
    pulay: *scf_mod.ComplexPulayMixer,
    mix_state: *EfieldMixState,
) !void {
    var iter: usize = 0;
    while (iter < dfpt_cfg.scf_max_iter) : (iter += 1) {
        const v1_hxc_r = try reciprocalToRealComplexCopy(alloc, grid, v1_hxc_g);
        defer alloc.free(v1_hxc_r);

        const rho1_r = try accumulateEfieldDensity(
            alloc,
            grid,
            dfpt_cfg,
            kgs,
            psi1_ddk,
            psi0_r_cache,
            beta,
            v1_hxc_r,
            psi1_ef,
        );
        defer alloc.free(rho1_r);

        const v_out_g = try buildEfieldOutputPotential(alloc, grid, gs, rho1_r);
        defer alloc.free(v_out_g);

        switch (try mixEfieldPotential(
            alloc,
            beta,
            iter,
            dfpt_cfg,
            v1_hxc_g,
            v_out_g,
            pulay,
            mix_state,
        )) {
            .converged => break,
            .keep_iterating => {},
        }
    }
}

fn reciprocalToRealComplexCopy(
    alloc: std.mem.Allocator,
    grid: Grid,
    values_g: []const math.Complex,
) ![]math.Complex {
    const work = try alloc.alloc(math.Complex, values_g.len);
    defer alloc.free(work);

    @memcpy(work, values_g);
    const values_r = try alloc.alloc(math.Complex, values_g.len);
    try scf_mod.fftReciprocalToComplexInPlace(alloc, grid, work, values_r, null);
    return values_r;
}

fn accumulateEfieldDensity(
    alloc: std.mem.Allocator,
    grid: Grid,
    dfpt_cfg: DfptConfig,
    kgs: []const KPointGsData,
    psi1_ddk: []const DirectionBandSet,
    psi0_r_cache: []const BandSet,
    beta: usize,
    v1_hxc_r: []const math.Complex,
    psi1_ef: []BandSet,
) ![]math.Complex {
    const rho1_r = try alloc.alloc(math.Complex, grid.count());
    @memset(rho1_r, math.complex.init(0.0, 0.0));

    errdefer alloc.free(rho1_r);

    for (kgs, 0..) |kg, ik| {
        try updateEfieldKpoint(
            alloc,
            grid,
            dfpt_cfg,
            &kg,
            psi1_ddk[ik][beta],
            psi0_r_cache[ik],
            v1_hxc_r,
            psi1_ef[ik],
            rho1_r,
        );
    }
    return rho1_r;
}

fn updateEfieldKpoint(
    alloc: std.mem.Allocator,
    grid: Grid,
    dfpt_cfg: DfptConfig,
    kg: *const KPointGsData,
    psi1_ddk_beta: BandSet,
    psi0_r_k: BandSet,
    v1_hxc_r: []const math.Complex,
    psi1_ef_k: BandSet,
    rho1_r: []math.Complex,
) !void {
    for (0..kg.n_occ) |n| {
        const psi1 = try solveEfieldBand(
            alloc,
            grid,
            dfpt_cfg,
            kg,
            psi1_ddk_beta[n],
            psi0_r_k[n],
            v1_hxc_r,
            n,
        );
        defer alloc.free(psi1);

        @memcpy(psi1_ef_k[n], psi1);
    }

    const psi1_const = try borrowConstBands(alloc, psi1_ef_k);
    defer alloc.free(psi1_const);

    const psi0_const = try borrowConstBands(alloc, psi0_r_k);
    defer alloc.free(psi0_const);

    const rho1_k_r = try phonon_q.computeRho1QCached(
        alloc,
        grid,
        &kg.map_k,
        psi0_const,
        psi1_const,
        kg.n_occ,
        kg.weight,
    );
    defer alloc.free(rho1_k_r);

    addDensityInPlace(rho1_r, rho1_k_r);
}

fn solveEfieldBand(
    alloc: std.mem.Allocator,
    grid: Grid,
    dfpt_cfg: DfptConfig,
    kg: *const KPointGsData,
    psi1_ddk_beta: []const math.Complex,
    psi0_r_band: []const math.Complex,
    v1_hxc_r: []const math.Complex,
    band_index: usize,
) ![]math.Complex {
    const rhs_ef = try alloc.alloc(math.Complex, kg.n_pw_k);
    defer alloc.free(rhs_ef);

    for (0..kg.n_pw_k) |g| {
        rhs_ef[g] = math.complex.init(-psi1_ddk_beta[g].i, psi1_ddk_beta[g].r);
    }

    const v1psi = try phonon_q.applyV1PsiQCached(
        alloc,
        grid,
        &kg.map_k,
        v1_hxc_r,
        psi0_r_band,
        kg.n_pw_k,
    );
    defer alloc.free(v1psi);

    for (0..kg.n_pw_k) |g| {
        rhs_ef[g] = math.complex.scale(
            math.complex.add(rhs_ef[g], v1psi[g]),
            -1.0,
        );
    }
    sternheimer.projectConduction(rhs_ef, kg.wavefunctions_k_const, kg.n_occ);

    const ef_result = try sternheimer.solve(
        alloc,
        kg.apply_ctx_k,
        rhs_ef,
        kg.eigenvalues_k[band_index],
        kg.wavefunctions_k_const,
        kg.n_occ,
        kg.basis_k.gvecs,
        .{
            .tol = dfpt_cfg.sternheimer_tol,
            .max_iter = dfpt_cfg.sternheimer_max_iter,
            .alpha_shift = dfpt_cfg.alpha_shift,
        },
    );
    return ef_result.psi1;
}

fn borrowConstBands(
    alloc: std.mem.Allocator,
    bands: BandSet,
) ![]const []const math.Complex {
    const const_bands = try alloc.alloc([]const math.Complex, bands.len);
    for (bands, 0..) |band, i| const_bands[i] = band;
    return const_bands;
}

fn addDensityInPlace(
    accum: []math.Complex,
    delta: []const math.Complex,
) void {
    for (accum, 0..) |value, i| {
        accum[i] = math.complex.add(value, delta[i]);
    }
}

fn buildEfieldOutputPotential(
    alloc: std.mem.Allocator,
    grid: Grid,
    gs: *const GroundState,
    rho1_r: []const math.Complex,
) ![]math.Complex {
    const rho1_r_copy = try alloc.alloc(math.Complex, rho1_r.len);
    defer alloc.free(rho1_r_copy);

    @memcpy(rho1_r_copy, rho1_r);
    const rho1_g = try alloc.alloc(math.Complex, rho1_r.len);
    defer alloc.free(rho1_g);

    try scf_mod.fftComplexToReciprocalInPlace(alloc, grid, rho1_r_copy, rho1_g, null);

    const vh1_g = try perturbation.buildHartreePerturbation(alloc, grid, rho1_g);
    defer alloc.free(vh1_g);

    const rho1_real = try alloc.alloc(f64, rho1_r.len);
    defer alloc.free(rho1_real);

    for (0..rho1_r.len) |i| rho1_real[i] = rho1_r[i].r;
    const vxc1_r = try perturbation.buildXcPerturbationFull(alloc, gs.*, rho1_real);
    defer alloc.free(vxc1_r);

    const vxc1_g = try scf_mod.realToReciprocal(alloc, grid, vxc1_r, false);
    defer alloc.free(vxc1_g);

    const v_out_g = try alloc.alloc(math.Complex, rho1_r.len);
    for (0..rho1_r.len) |i| v_out_g[i] = math.complex.add(vh1_g[i], vxc1_g[i]);
    return v_out_g;
}

fn mixEfieldPotential(
    alloc: std.mem.Allocator,
    beta: usize,
    iter: usize,
    dfpt_cfg: DfptConfig,
    v1_hxc_g: []math.Complex,
    v_out_g: []const math.Complex,
    pulay: *scf_mod.ComplexPulayMixer,
    mix_state: *EfieldMixState,
) !EfieldMixOutcome {
    var residual_norm: f64 = 0.0;
    const residual = try alloc.alloc(math.Complex, v_out_g.len);
    for (0..v_out_g.len) |i| {
        residual[i] = math.complex.sub(v_out_g[i], v1_hxc_g[i]);
        residual_norm += residual[i].r * residual[i].r + residual[i].i * residual[i].i;
    }
    residual_norm = @sqrt(residual_norm);
    logDfpt("efield SCF β={d}: iter={d} vresid={e:.6}\n", .{ beta, iter, residual_norm });

    const converged_strict = residual_norm < dfpt_cfg.scf_tol;
    const converged_forced = mix_state.force_converge and residual_norm < 10.0 * dfpt_cfg.scf_tol;
    if (converged_strict or converged_forced) {
        alloc.free(residual);
        logDfptInfo(
            "efield SCF β={d}: converged at iter={d} vresid={e:.6}\n",
            .{ beta, iter, residual_norm },
        );
        return .converged;
    }

    if (residual_norm < mix_state.best_vresid) {
        mix_state.best_vresid = residual_norm;
        if (mix_state.best_v1 == null) {
            mix_state.best_v1 = try alloc.alloc(math.Complex, v1_hxc_g.len);
        }
        @memcpy(mix_state.best_v1.?, v1_hxc_g);
    }

    const restart_factor: f64 = 5.0;
    const pulay_ready = iter >= mix_state.pulay_active_since;
    const residual_blew_up = residual_norm > restart_factor * mix_state.best_vresid;
    const best_is_good = mix_state.best_vresid < 1.0;
    if (pulay_ready and residual_blew_up and best_is_good) {
        if (mix_state.best_v1) |best| @memcpy(v1_hxc_g, best);
        if (mix_state.best_vresid < 10.0 * dfpt_cfg.scf_tol) {
            mix_state.force_converge = true;
            logDfpt(
                "efield SCF β={d}: Pulay restart (near-converged) iter={d}\n",
                .{ beta, iter },
            );
            alloc.free(residual);
            return .keep_iterating;
        }
        pulay.reset();
        mix_state.pulay_active_since = iter + 1 + dfpt_cfg.pulay_start;
        logDfpt(
            "efield SCF β={d}: Pulay restart iter={d} vresid={e:.6} best={e:.6}\n",
            .{ beta, iter, residual_norm, mix_state.best_vresid },
        );
        alloc.free(residual);
        return .keep_iterating;
    }

    if (dfpt_cfg.pulay_history > 0 and iter >= mix_state.pulay_active_since) {
        try pulay.mixWithResidual(v1_hxc_g, residual, dfpt_cfg.mixing_beta);
        return .keep_iterating;
    }

    for (0..v_out_g.len) |i| {
        const delta = math.complex.scale(residual[i], dfpt_cfg.mixing_beta);
        v1_hxc_g[i] = math.complex.add(v1_hxc_g[i], delta);
    }
    alloc.free(residual);
    return .keep_iterating;
}

fn accumulateDielectricColumn(
    kgs: []const KPointGsData,
    psi1_ddk: []const DirectionBandSet,
    psi1_ef: []const BandSet,
    volume: f64,
) [3]f64 {
    var column: [3]f64 = .{ 0.0, 0.0, 0.0 };
    const occ: f64 = 2.0;
    for (kgs, 0..) |kg, ik| {
        for (0..kg.n_occ) |n| {
            for (0..3) |alpha| {
                var overlap: f64 = 0.0;
                for (0..kg.n_pw_k) |g| {
                    const i_psi1_r = -psi1_ddk[ik][alpha][n][g].i;
                    const i_psi1_i = psi1_ddk[ik][alpha][n][g].r;
                    overlap += i_psi1_r * psi1_ef[ik][n][g].r;
                    overlap += i_psi1_i * psi1_ef[ik][n][g].i;
                }
                const prefactor = (16.0 * std.math.pi / volume) * occ * kg.weight;
                column[alpha] -= prefactor * overlap;
            }
        }
    }
    return column;
}

fn symmetrizeDielectricTensor(
    alloc: std.mem.Allocator,
    epsilon: [3][3]f64,
    cell_bohr: math.Mat3,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
) ![3][3]f64 {
    const symmetry_mod = @import("../symmetry/symmetry.zig");
    const symops_eps = symmetry_mod.getSymmetryOps(alloc, cell_bohr, atoms, 1e-5) catch null;
    if (symops_eps == null) return epsilon;

    const ops = symops_eps.?;
    defer alloc.free(ops);

    var eps_sym: [3][3]f64 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    const inv2pi = 1.0 / (2.0 * std.math.pi);
    for (ops) |op| {
        var rc: [3][3]f64 = undefined;
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0.0;
                for (0..3) |k2| {
                    for (0..3) |l2| {
                        const rot_val = @as(f64, @floatFromInt(op.rot.m[k2][l2]));
                        s += cell_bohr.m[k2][i] * rot_val * recip.m[l2][j] * inv2pi;
                    }
                }
                rc[i][j] = s;
            }
        }
        for (0..3) |i| {
            for (0..3) |j| {
                var val: f64 = 0.0;
                for (0..3) |a2| {
                    for (0..3) |b2| {
                        val += rc[a2][i] * epsilon[a2][b2] * rc[b2][j];
                    }
                }
                eps_sym[i][j] += val;
            }
        }
    }

    const nsym_f: f64 = @floatFromInt(ops.len);
    for (0..3) |i| {
        for (0..3) |j| eps_sym[i][j] /= nsym_f;
    }
    return eps_sym;
}

/// Compute nonlocal projector value: φ(q) = 4π f(|q|) Y_lm(q̂)
/// Uses nonlocal.zig's Y_lm to be consistent with computeDphiBeta.
fn computePhiBeta(kpg: math.Vec3, l: i32, m: i32, table: *const nonlocal.RadialTable) f64 {
    const r = math.Vec3.norm(kpg);
    const ylm = nonlocal.realSphericalHarmonic(l, m, kpg.x, kpg.y, kpg.z);
    return 4.0 * std.math.pi * table.eval(r) * ylm;
}

/// Analytical derivative of the nonlocal projector:
///   φ(q) = 4π f(|q|) Y_lm(q̂)
///   dφ/dq_α = 4π [f'(|q|) (q_α/|q|) Y_lm(q̂) + f(|q|) dY_lm(q̂)/dq_α]
///
/// where dY_lm(q̂)/dq_α = Σ_β (∂Y_lm/∂n_β)(δ_αβ - n_α n_β)/|q|
fn computeDphiBeta(
    kpg: math.Vec3,
    direction: usize,
    l: i32,
    m: i32,
    table: *const nonlocal.RadialTable,
) f64 {
    const four_pi = 4.0 * std.math.pi;
    const r2 = math.Vec3.dot(kpg, kpg);

    if (r2 < 1e-30) {
        // At q=0: only l=1 has non-zero derivative
        if (l == 1) {
            // At q=0 for l=1: analytical limit is singular, use FD fallback
            const delta: f64 = 1e-5;
            var qp = kpg;
            var qm = kpg;
            switch (direction) {
                0 => {
                    qp.x += delta;
                    qm.x -= delta;
                },
                1 => {
                    qp.y += delta;
                    qm.y -= delta;
                },
                2 => {
                    qp.z += delta;
                    qm.z -= delta;
                },
                else => {},
            }
            const ylm_p = nonlocal.realSphericalHarmonic(l, m, qp.x, qp.y, qp.z);
            const ylm_m = nonlocal.realSphericalHarmonic(l, m, qm.x, qm.y, qm.z);
            const phi_p = table.eval(math.Vec3.norm(qp)) * ylm_p;
            const phi_m = table.eval(math.Vec3.norm(qm)) * ylm_m;
            return four_pi * (phi_p - phi_m) / (2.0 * delta);
        }
        return 0.0;
    }

    const r = @sqrt(r2);
    const inv_r = 1.0 / r;
    const nx = kpg.x * inv_r;
    const ny = kpg.y * inv_r;
    const nz = kpg.z * inv_r;
    const n_alpha = switch (direction) {
        0 => nx,
        1 => ny,
        2 => nz,
        else => 0.0,
    };

    const f_val = table.eval(r);
    const fp_val = table.evalDeriv(r);
    const ylm = nonlocal.realSphericalHarmonic(l, m, kpg.x, kpg.y, kpg.z);

    // Term 1: f'(|q|) * (q_α/|q|) * Y_lm(q̂)
    const term1 = fp_val * n_alpha * ylm;

    // Term 2: f(|q|) * dY_lm(q̂)/dq_α
    // dY_lm/dq_α = (1/|q|) * Σ_β (∂Y_lm/∂n_β) * (δ_αβ - n_α * n_β)
    //            = (1/|q|) * [∂Y_lm/∂n_α - n_α * Σ_β n_β * ∂Y_lm/∂n_β]
    const grad = dYlm_dn(l, m, nx, ny, nz);
    const dot_n_grad = nx * grad[0] + ny * grad[1] + nz * grad[2];
    const dylm_dq = inv_r * (grad[direction] - n_alpha * dot_n_grad);
    const term2 = f_val * dylm_dq;

    return four_pi * (term1 + term2);
}

/// Partial derivatives of real spherical harmonics with respect to n_x, n_y, n_z.
/// Returns [∂Y_lm/∂n_x, ∂Y_lm/∂n_y, ∂Y_lm/∂n_z].
fn dYlm_dn(l: i32, m: i32, nx: f64, ny: f64, nz: f64) [3]f64 {
    const pi = std.math.pi;
    switch (l) {
        0 => return .{ 0.0, 0.0, 0.0 },
        1 => {
            // Y_1^{-1} = c*ny, Y_1^0 = c*nz, Y_1^1 = c*nx
            const c = @sqrt(3.0 / (4.0 * pi));
            return switch (m) {
                -1 => .{ 0.0, c, 0.0 },
                0 => .{ 0.0, 0.0, c },
                1 => .{ c, 0.0, 0.0 },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        2 => {
            const c0 = @sqrt(5.0 / (16.0 * pi));
            const c1 = @sqrt(15.0 / (4.0 * pi));
            const c2 = @sqrt(15.0 / (16.0 * pi));
            return switch (m) {
                // Y_2^{-2} = c2 * 2*nx*ny
                -2 => .{ c2 * 2.0 * ny, c2 * 2.0 * nx, 0.0 },
                // Y_2^{-1} = c1 * ny*nz
                -1 => .{ 0.0, c1 * nz, c1 * ny },
                // Y_2^0 = c0 * (3*nz²-1)
                0 => .{ 0.0, 0.0, c0 * 6.0 * nz },
                // Y_2^1 = c1 * nx*nz
                1 => .{ c1 * nz, 0.0, c1 * nx },
                // Y_2^2 = c2 * (nx²-ny²)
                2 => .{ c2 * 2.0 * nx, c2 * (-2.0) * ny, 0.0 },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        3 => {
            const c3m = @sqrt(35.0 / (32.0 * pi));
            const c2m = @sqrt(105.0 / (4.0 * pi));
            const c1m = @sqrt(21.0 / (32.0 * pi));
            const c0v = @sqrt(7.0 / (16.0 * pi));
            const c2p = @sqrt(105.0 / (16.0 * pi));
            const c3p = @sqrt(35.0 / (32.0 * pi));
            return switch (m) {
                // Y_3^{-3} = c3m * (3nx²-ny²)*ny
                -3 => .{
                    c3m * 6.0 * nx * ny,
                    c3m * (3.0 * nx * nx - 3.0 * ny * ny),
                    0.0,
                },
                // Y_3^{-2} = c2m * nx*ny*nz
                -2 => .{ c2m * ny * nz, c2m * nx * nz, c2m * nx * ny },
                // Y_3^{-1} = c1m * ny*(5nz²-1)
                -1 => .{ 0.0, c1m * (5.0 * nz * nz - 1.0), c1m * 10.0 * ny * nz },
                // Y_3^0 = c0v * (5nz³-3nz)
                0 => .{ 0.0, 0.0, c0v * (15.0 * nz * nz - 3.0) },
                // Y_3^1 = c1m * nx*(5nz²-1)  (same c as m=-1)
                1 => .{ c1m * (5.0 * nz * nz - 1.0), 0.0, c1m * 10.0 * nx * nz },
                // Y_3^2 = c2p * (nx²-ny²)*nz
                2 => .{ c2p * 2.0 * nx * nz, c2p * (-2.0) * ny * nz, c2p * (nx * nx - ny * ny) },
                // Y_3^3 = c3p * (nx²-3ny²)*nx
                3 => .{
                    c3p * (3.0 * nx * nx - 3.0 * ny * ny),
                    c3p * (-6.0) * nx * ny,
                    0.0,
                },
                else => .{ 0.0, 0.0, 0.0 },
            };
        },
        else => return .{ 0.0, 0.0, 0.0 },
    }
}

fn applyDdkNonlocal(
    gvecs: []const plane_wave.GVector,
    atoms: []const hamiltonian.AtomData,
    nl_ctx: scf_mod.NonlocalContext,
    direction: usize,
    inv_volume: f64,
    psi: []const math.Complex,
    radial_tables: []const nonlocal.RadialTableSet,
    out: []math.Complex,
    work_phase: []math.Complex,
    work_coeff: []math.Complex,
    work_coeff2: []math.Complex,
    work_dcoeff: []math.Complex,
    work_dcoeff2: []math.Complex,
) void {
    const n_pw = gvecs.len;
    @memset(out, math.complex.init(0.0, 0.0));

    for (nl_ctx.species) |sp| {
        const g_count = sp.g_count;
        if (g_count != n_pw) continue;
        if (sp.m_total == 0) continue;

        for (atoms) |atom| {
            if (atom.species_index != sp.species_index) continue;
            for (0..n_pw) |g| {
                work_phase[g] = math.complex.expi(math.Vec3.dot(gvecs[g].cart, atom.position));
            }

            var b: usize = 0;
            while (b < sp.beta_count) : (b += 1) {
                const l_val = sp.l_list[b];
                const offset = sp.m_offsets[b];
                const m_count = sp.m_counts[b];
                const table = &radial_tables[atom.species_index].tables[b];
                var m_idx: usize = 0;
                while (m_idx < m_count) : (m_idx += 1) {
                    const m_val = @as(i32, @intCast(m_idx)) - l_val;
                    var coeff = math.complex.init(0.0, 0.0);
                    var dcoeff = math.complex.init(0.0, 0.0);
                    for (0..n_pw) |g| {
                        const kpg = gvecs[g].kpg;
                        const xphase = math.complex.mul(psi[g], work_phase[g]);
                        // Use fresh phi consistent with computeDphiBeta's Y_lm
                        const phi_val = computePhiBeta(kpg, l_val, m_val, table);
                        coeff = math.complex.add(coeff, math.complex.scale(xphase, phi_val));
                        const dphi_val = computeDphiBeta(kpg, direction, l_val, m_val, table);
                        dcoeff = math.complex.add(dcoeff, math.complex.scale(xphase, dphi_val));
                    }
                    work_coeff[offset + m_idx] = math.complex.scale(coeff, inv_volume);
                    work_dcoeff[offset + m_idx] = math.complex.scale(dcoeff, inv_volume);
                }
            }
            applyDij(sp, work_coeff, work_coeff2);
            applyDij(sp, work_dcoeff, work_dcoeff2);

            b = 0;
            while (b < sp.beta_count) : (b += 1) {
                const l_val = sp.l_list[b];
                const offset = sp.m_offsets[b];
                const m_count = sp.m_counts[b];
                const table = &radial_tables[atom.species_index].tables[b];
                var m_idx: usize = 0;
                while (m_idx < m_count) : (m_idx += 1) {
                    const m_val = @as(i32, @intCast(m_idx)) - l_val;
                    const dc = work_coeff2[offset + m_idx];
                    const ddc = work_dcoeff2[offset + m_idx];
                    if (@abs(dc.r) + @abs(dc.i) + @abs(ddc.r) + @abs(ddc.i) < 1e-30) continue;
                    for (0..n_pw) |g| {
                        const phase_conj = math.complex.conj(work_phase[g]);
                        const phi_val = computePhiBeta(gvecs[g].kpg, l_val, m_val, table);
                        const dphi = computeDphiBeta(gvecs[g].kpg, direction, l_val, m_val, table);
                        out[g] = math.complex.add(out[g], math.complex.add(
                            math.complex.mul(math.complex.scale(phase_conj, dphi), dc),
                            math.complex.mul(math.complex.scale(phase_conj, phi_val), ddc),
                        ));
                    }
                }
            }
        }
    }
}

fn applyDij(sp: scf_mod.NonlocalSpecies, input: []const math.Complex, output: []math.Complex) void {
    var b: usize = 0;
    while (b < sp.beta_count) : (b += 1) {
        const l_val = sp.l_list[b];
        const offset = sp.m_offsets[b];
        const m_count = sp.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            var sum = math.complex.init(0.0, 0.0);
            var j: usize = 0;
            while (j < sp.beta_count) : (j += 1) {
                if (sp.l_list[j] != l_val) continue;
                const dij = sp.coeffs[b * sp.beta_count + j];
                if (dij == 0.0) continue;
                const scaled = math.complex.scale(input[sp.m_offsets[j] + m_idx], dij);
                sum = math.complex.add(sum, scaled);
            }
            output[offset + m_idx] = sum;
        }
    }
}

pub fn writeElectricResults(
    io: std.Io,
    dir: std.Io.Dir,
    dielectric: DielectricResult,
) !void {
    const file = try dir.createFile(io, "electric.dat", .{});
    defer file.close(io);

    var buf: [1024]u8 = undefined;
    var writer = file.writer(io, &buf);
    const out = &writer.interface;

    try out.print("# Dielectric tensor epsilon_inf\n", .{});
    for (0..3) |i| {
        try out.print("{d:12.6} {d:12.6} {d:12.6}\n", .{
            dielectric.epsilon[i][0], dielectric.epsilon[i][1], dielectric.epsilon[i][2],
        });
    }

    try out.flush();
}
