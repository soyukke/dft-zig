const std = @import("std");
const math = @import("../math/math.zig");
const config = @import("../config/config.zig");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const d3 = @import("../vdw/d3.zig");
const d3_params = @import("../vdw/d3_params.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const ewald = @import("../ewald/ewald.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
const runtime_logging = @import("../runtime/logging.zig");
const scf = @import("../scf/scf.zig");
pub const local_force = @import("local_force.zig");
pub const nonlocal_force = @import("nonlocal_force.zig");
pub const residual_force = @import("residual_force.zig");
pub const nlcc_force = @import("nlcc_force.zig");
pub const paw_dhat_force = @import("paw_dhat_force.zig");

/// Total forces on atoms from all contributions.
pub const ForceTerms = struct {
    total: []math.Vec3,
    ewald: []math.Vec3,
    local: []math.Vec3,
    nonlocal: ?[]math.Vec3,
    residual: ?[]math.Vec3,
    nlcc: ?[]math.Vec3,
    dispersion: ?[]math.Vec3,
    paw_dhat: ?[]math.Vec3 = null,

    pub fn deinit(self: *ForceTerms, alloc: std.mem.Allocator) void {
        if (self.total.len > 0) alloc.free(self.total);
        if (self.ewald.len > 0) alloc.free(self.ewald);
        if (self.local.len > 0) alloc.free(self.local);
        if (self.nonlocal) |nl| {
            if (nl.len > 0) alloc.free(nl);
        }
        if (self.residual) |resid| {
            if (resid.len > 0) alloc.free(resid);
        }
        if (self.nlcc) |values| {
            if (values.len > 0) alloc.free(values);
        }
        if (self.dispersion) |disp| {
            if (disp.len > 0) alloc.free(disp);
        }
        if (self.paw_dhat) |pd| {
            if (pd.len > 0) alloc.free(pd);
        }
    }
};

/// Grid parameters for force calculation
pub const Grid = local_force.Grid;

/// Compute ion-ion forces using either direct Coulomb (isolated) or Ewald (periodic).
/// Returns freshly allocated forces in Rydberg/Bohr units.
fn computeIonIonForces(
    alloc: std.mem.Allocator,
    charges: []const f64,
    positions: []const math.Vec3,
    cell: math.Mat3,
    recip: math.Mat3,
    alpha: f64,
    quiet: bool,
    coulomb_r_cut: ?f64,
) ![]math.Vec3 {
    const n_atoms = positions.len;
    const ewald_forces = try alloc.alloc(math.Vec3, n_atoms);
    if (coulomb_r_cut != null) {
        // Isolated system: direct pairwise Coulomb forces (in Hartree/Bohr)
        const direct_forces_ha = try coulomb_mod.directIonIonForces(alloc, charges, positions);
        defer alloc.free(direct_forces_ha);

        for (direct_forces_ha, 0..) |f_ha, i| {
            ewald_forces[i] = math.Vec3.scale(f_ha, 2.0);
        }
    } else {
        // Periodic system: Ewald forces (in Hartree/Bohr)
        const ewald_params = ewald.Params{
            .alpha = alpha,
            .rcut = 0.0, // auto
            .gcut = 0.0, // auto
            .tol = 1e-8,
            .quiet = quiet,
        };
        const ewald_forces_ha = try ewald.ionIonForces(
            alloc,
            cell,
            recip,
            charges,
            positions,
            ewald_params,
        );
        defer alloc.free(ewald_forces_ha);
        // Convert Ewald forces to Rydberg units
        for (ewald_forces_ha, 0..) |f_ha, i| {
            ewald_forces[i] = math.Vec3.scale(f_ha, 2.0);
        }
    }
    return ewald_forces;
}

/// Compute nonlocal forces including optional spin-down contribution.
fn computeNonlocalForcesTotal(
    alloc: std.mem.Allocator,
    wavefunctions: ?scf.WavefunctionData,
    wavefunctions_down: ?scf.WavefunctionData,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    radial_tables: ?[]nonlocal.RadialTableSet,
    paw_dij: ?[]const []const f64,
    paw_sij_list: ?[]const []const f64,
) !?[]math.Vec3 {
    const wf = wavefunctions orelse return null;
    const is_spin = wavefunctions_down != null;
    const sf: f64 = if (is_spin) 1.0 else 2.0;
    const nl_forces = try nonlocal_force.nonlocalForces(
        alloc,
        wf,
        species,
        atoms,
        recip,
        volume,
        radial_tables,
        paw_dij,
        paw_sij_list,
        sf,
    );
    // Add spin-down nonlocal forces if spin-polarized
    if (wavefunctions_down) |wf_down| {
        const nl_down = try nonlocal_force.nonlocalForces(
            alloc,
            wf_down,
            species,
            atoms,
            recip,
            volume,
            radial_tables,
            paw_dij,
            paw_sij_list,
            1.0,
        );
        defer alloc.free(nl_down);

        for (nl_forces, 0..) |*f, i| {
            f.* = math.Vec3.add(f.*, nl_down[i]);
        }
    }
    return nl_forces;
}

/// Extract V_xc(G) from V_eff(G) = V_H(G) + V_xc(G) by subtracting the Hartree component.
fn extractVxcFromPotential(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_g: []const math.Complex,
    potential_g: []const math.Complex,
) ![]math.Complex {
    const total = grid.nx * grid.ny * grid.nz;
    if (potential_g.len != total) return error.InvalidGrid;
    const vxc_g = try alloc.alloc(math.Complex, total);
    const b1x = grid.recip.row(0);
    const b2x = grid.recip.row(1);
    const b3x = grid.recip.row(2);
    var idx: usize = 0;
    var l: usize = 0;
    while (l < grid.nz) : (l += 1) {
        var k: usize = 0;
        while (k < grid.ny) : (k += 1) {
            var h: usize = 0;
            while (h < grid.nx) : (h += 1) {
                const gh = grid.min_h + @as(i32, @intCast(h));
                const gk = grid.min_k + @as(i32, @intCast(k));
                const gl = grid.min_l + @as(i32, @intCast(l));
                const gh_f = @as(f64, @floatFromInt(gh));
                const gk_f = @as(f64, @floatFromInt(gk));
                const gl_f = @as(f64, @floatFromInt(gl));
                const gvec = math.Vec3.add(
                    math.Vec3.add(
                        math.Vec3.scale(b1x, gh_f),
                        math.Vec3.scale(b2x, gk_f),
                    ),
                    math.Vec3.scale(b3x, gl_f),
                );
                const g2 = math.Vec3.dot(gvec, gvec);
                var vh = math.complex.init(0.0, 0.0);
                if (g2 > 1e-12) {
                    vh = math.complex.scale(rho_g[idx], 8.0 * std.math.pi / g2);
                }
                vxc_g[idx] = math.complex.sub(potential_g[idx], vh);
                idx += 1;
            }
        }
    }
    return vxc_g;
}

/// Compute NLCC forces from either a precomputed V_xc(r) or V_eff(G) + ρ(G).
fn computeNlccForcesIfAvailable(
    alloc: std.mem.Allocator,
    grid: Grid,
    rho_g: []const math.Complex,
    potential_g: ?[]const math.Complex,
    precomputed_vxc_r: ?[]const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    volume: f64,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) !?[]math.Vec3 {
    if (precomputed_vxc_r) |vxc_r| {
        // G-space NLCC force: FFT V_xc(r) → V_xc(G), then use form factor.
        // This avoids aliasing between bandwidth-limited V_xc and
        // the tabulated radial core charge derivative.
        const fft_grid_obj = scf.Grid{
            .nx = grid.nx,
            .ny = grid.ny,
            .nz = grid.nz,
            .min_h = grid.min_h,
            .min_k = grid.min_k,
            .min_l = grid.min_l,
            .cell = grid.cell,
            .recip = grid.recip,
            .volume = volume,
        };
        const vxc_g = try scf.realToReciprocal(alloc, fft_grid_obj, vxc_r, false);
        defer alloc.free(vxc_g);

        return try nlcc_force.nlccForcesGSpace(
            alloc,
            grid,
            vxc_g,
            species,
            atoms,
            rho_core_tables,
        );
    } else if (potential_g) |pot| {
        const vxc_g = try extractVxcFromPotential(alloc, grid, rho_g, pot);
        defer alloc.free(vxc_g);
        // Use G-space NLCC force directly (no need to FFT back to real space)
        return try nlcc_force.nlccForcesGSpace(
            alloc,
            grid,
            vxc_g,
            species,
            atoms,
            rho_core_tables,
        );
    }
    return null;
}

/// Compute D3(BJ) dispersion forces for the configured vdw settings.
fn computeDispersionForces(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell: math.Mat3,
    vdw_cfg: config.VdwConfig,
) !?[]math.Vec3 {
    if (!vdw_cfg.enabled) return null;
    const n_atoms = atoms.len;
    const atomic_numbers = try alloc.alloc(usize, n_atoms);
    defer alloc.free(atomic_numbers);

    const atom_positions = try alloc.alloc(math.Vec3, n_atoms);
    defer alloc.free(atom_positions);

    for (atoms, 0..) |atom, idx| {
        const symbol = species[atom.species_index].symbol;
        atomic_numbers[idx] = d3_params.atomicNumber(symbol) orelse 0;
        atom_positions[idx] = atom.position;
    }
    var damping = d3_params.pbe_d3bj;
    if (vdw_cfg.s6) |v| damping.s6 = v;
    if (vdw_cfg.s8) |v| damping.s8 = v;
    if (vdw_cfg.a1) |v| damping.a1 = v;
    if (vdw_cfg.a2) |v| damping.a2 = v;
    return try d3.computeForces(
        alloc,
        atomic_numbers,
        atom_positions,
        cell,
        damping,
        vdw_cfg.cutoff_radius,
        vdw_cfg.cn_cutoff,
    );
}

/// Sum per-component forces into the total force array.
fn sumForceComponents(
    total_forces: []math.Vec3,
    ewald_forces: []const math.Vec3,
    local_forces: []const math.Vec3,
    nl_forces: ?[]const math.Vec3,
    resid_forces: ?[]const math.Vec3,
    nlcc_forces: ?[]const math.Vec3,
    disp_forces: ?[]const math.Vec3,
    paw_dhat_forces: ?[]const math.Vec3,
) void {
    for (0..total_forces.len) |i| {
        total_forces[i] = math.Vec3.add(ewald_forces[i], local_forces[i]);
        if (nl_forces) |nl| {
            total_forces[i] = math.Vec3.add(total_forces[i], nl[i]);
        }
        if (resid_forces) |resid| {
            total_forces[i] = math.Vec3.add(total_forces[i], resid[i]);
        }
        if (nlcc_forces) |nlcc| {
            total_forces[i] = math.Vec3.add(total_forces[i], nlcc[i]);
        }
        if (disp_forces) |disp| {
            total_forces[i] = math.Vec3.add(total_forces[i], disp[i]);
        }
        if (paw_dhat_forces) |pd| {
            total_forces[i] = math.Vec3.add(total_forces[i], pd[i]);
        }
    }
}

/// Log force section timings when not in quiet mode.
fn logForceTimings(
    logger: runtime_logging.Logger,
    quiet: bool,
    t0: std.Io.Clock.Timestamp,
    after_ewald: std.Io.Clock.Timestamp,
    after_local: std.Io.Clock.Timestamp,
    after_nonlocal: std.Io.Clock.Timestamp,
    after_nlcc: std.Io.Clock.Timestamp,
) void {
    if (quiet) return;
    const ewald_ns = t0.durationTo(after_ewald).raw.nanoseconds;
    const local_ns = after_ewald.durationTo(after_local).raw.nanoseconds;
    const nl_ns = after_local.durationTo(after_nonlocal).raw.nanoseconds;
    const nlcc_ns = after_nonlocal.durationTo(after_nlcc).raw.nanoseconds;
    const total_ns = t0.durationTo(after_nlcc).raw.nanoseconds;
    const ewald_ms = @as(f64, @floatFromInt(ewald_ns)) / 1_000_000.0;
    const local_ms = @as(f64, @floatFromInt(local_ns)) / 1_000_000.0;
    const nl_ms = @as(f64, @floatFromInt(nl_ns)) / 1_000_000.0;
    const nlcc_ms = @as(f64, @floatFromInt(nlcc_ns)) / 1_000_000.0;
    const total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;
    logger.print(
        .info,
        "force_profile ewald_ms={d:.1} local_ms={d:.1} nonlocal_ms={d:.1}" ++
            " nlcc_ms={d:.1} total_ms={d:.1}\n",
        .{ ewald_ms, local_ms, nl_ms, nlcc_ms, total_ms },
    ) catch {};
}

/// Emit per-atom force component debug lines.
fn logForceComponents(
    logger: runtime_logging.Logger,
    ewald_forces: []const math.Vec3,
    local_forces: []const math.Vec3,
    nl_forces: ?[]const math.Vec3,
    resid_forces: ?[]const math.Vec3,
    nlcc_forces: ?[]const math.Vec3,
    total_forces: []const math.Vec3,
) void {
    logger.print(.debug, "\n=== Force Components (Ry/Bohr) ===\n", .{}) catch {};
    const zero_vec = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    for (0..total_forces.len) |i| {
        logForceComponentLine(
            logger,
            i,
            ewald_forces[i],
            local_forces[i],
            if (nl_forces) |nl| nl[i] else null,
            if (resid_forces) |resid| resid[i] else null,
            if (nlcc_forces) |nlcc| nlcc[i] else null,
            total_forces[i],
            zero_vec,
        );
    }
}

fn logForceComponentLine(
    logger: runtime_logging.Logger,
    i: usize,
    ew: math.Vec3,
    loc: math.Vec3,
    nlf: ?math.Vec3,
    rf: ?math.Vec3,
    cf: ?math.Vec3,
    tot: math.Vec3,
    zero_vec: math.Vec3,
) void {
    if (nlf) |nl_force| {
        logForceComponentWithNonlocal(
            logger,
            i,
            ew,
            loc,
            nl_force,
            rf,
            cf,
            tot,
            zero_vec,
        );
        return;
    }
    logForceComponentWithoutNonlocal(logger, i, ew, loc, rf, cf, tot, zero_vec);
}

fn logForceComponentWithNonlocal(
    logger: runtime_logging.Logger,
    i: usize,
    ew: math.Vec3,
    loc: math.Vec3,
    nl_force: math.Vec3,
    rf: ?math.Vec3,
    cf: ?math.Vec3,
    tot: math.Vec3,
    zero_vec: math.Vec3,
) void {
    if (rf != null or cf != null) {
        const resid_force = rf orelse zero_vec;
        const core_force = cf orelse zero_vec;
        logger.print(
            .debug,
            "Atom {d}: Ewald=({d:.6},{d:.6},{d:.6})" ++
                " Local=({d:.6},{d:.6},{d:.6})" ++
                " Nonlocal=({d:.6},{d:.6},{d:.6})" ++
                " Resid=({d:.6},{d:.6},{d:.6})" ++
                " NLCC=({d:.6},{d:.6},{d:.6})" ++
                " Total=({d:.6},{d:.6},{d:.6})\n",
            .{
                i,
                ew.x,
                ew.y,
                ew.z,
                loc.x,
                loc.y,
                loc.z,
                nl_force.x,
                nl_force.y,
                nl_force.z,
                resid_force.x,
                resid_force.y,
                resid_force.z,
                core_force.x,
                core_force.y,
                core_force.z,
                tot.x,
                tot.y,
                tot.z,
            },
        ) catch {};
        return;
    }
    logger.print(
        .debug,
        "Atom {d}: Ewald=({d:.6},{d:.6},{d:.6})" ++
            " Local=({d:.6},{d:.6},{d:.6})" ++
            " Nonlocal=({d:.6},{d:.6},{d:.6})" ++
            " Total=({d:.6},{d:.6},{d:.6})\n",
        .{
            i,
            ew.x,
            ew.y,
            ew.z,
            loc.x,
            loc.y,
            loc.z,
            nl_force.x,
            nl_force.y,
            nl_force.z,
            tot.x,
            tot.y,
            tot.z,
        },
    ) catch {};
}

fn logForceComponentWithoutNonlocal(
    logger: runtime_logging.Logger,
    i: usize,
    ew: math.Vec3,
    loc: math.Vec3,
    rf: ?math.Vec3,
    cf: ?math.Vec3,
    tot: math.Vec3,
    zero_vec: math.Vec3,
) void {
    if (rf != null or cf != null) {
        const resid_force = rf orelse zero_vec;
        const core_force = cf orelse zero_vec;
        logger.print(
            .debug,
            "Atom {d}: Ewald=({d:.6},{d:.6},{d:.6})" ++
                " Local=({d:.6},{d:.6},{d:.6})" ++
                " Resid=({d:.6},{d:.6},{d:.6})" ++
                " NLCC=({d:.6},{d:.6},{d:.6})" ++
                " Total=({d:.6},{d:.6},{d:.6})\n",
            .{
                i,
                ew.x,
                ew.y,
                ew.z,
                loc.x,
                loc.y,
                loc.z,
                resid_force.x,
                resid_force.y,
                resid_force.z,
                core_force.x,
                core_force.y,
                core_force.z,
                tot.x,
                tot.y,
                tot.z,
            },
        ) catch {};
        return;
    }
    logger.print(
        .debug,
        "Atom {d}: Ewald=({d:.6},{d:.6},{d:.6})" ++
            " Local=({d:.6},{d:.6},{d:.6})" ++
            " Total=({d:.6},{d:.6},{d:.6})\n",
        .{
            i,
            ew.x,
            ew.y,
            ew.z,
            loc.x,
            loc.y,
            loc.z,
            tot.x,
            tot.y,
            tot.z,
        },
    ) catch {};
}

/// Timestamp tuple captured across the major force-computation phases.
const ForceTimings = struct {
    t0: std.Io.Clock.Timestamp,
    after_ewald: std.Io.Clock.Timestamp,
    after_local: std.Io.Clock.Timestamp,
    after_nonlocal: std.Io.Clock.Timestamp,
    after_nlcc: std.Io.Clock.Timestamp,
};

const ForceComputeInput = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    rho_g: []const math.Complex,
    potential_g: ?[]const math.Complex,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    alpha: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    wavefunctions: ?scf.WavefunctionData,
    vresid_g: ?[]const math.Complex,
    quiet: bool,
    radial_tables: ?[]nonlocal.RadialTableSet,
    precomputed_vxc_r: ?[]const f64,
    rho_atom_tables: ?[]const form_factor.RadialFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    coulomb_r_cut: ?f64,
    vdw_cfg: config.VdwConfig,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_dij: ?[]const []const f64,
    paw_rhoij: ?[]const []const f64,
    wavefunctions_down: ?scf.WavefunctionData,
};

const IonData = struct {
    charges: []f64,
    positions: []math.Vec3,

    fn init(
        alloc: std.mem.Allocator,
        species: []const hamiltonian.SpeciesEntry,
        atoms: []const hamiltonian.AtomData,
    ) !IonData {
        var charges: []f64 = &[_]f64{};
        var positions: []math.Vec3 = &[_]math.Vec3{};
        try extractChargesAndPositions(alloc, species, atoms, &charges, &positions);
        return .{ .charges = charges, .positions = positions };
    }

    fn deinit(self: IonData, alloc: std.mem.Allocator) void {
        alloc.free(self.charges);
        alloc.free(self.positions);
    }
};

const PawSijData = struct {
    list: ?[]const []const f64 = null,
    buf: ?[][]const f64 = null,

    fn init(
        alloc: std.mem.Allocator,
        paw_tabs: ?[]const paw_mod.PawTab,
    ) !PawSijData {
        var data: PawSijData = .{};
        try buildPawSijList(alloc, paw_tabs, &data.list, &data.buf);
        return data;
    }

    fn deinit(self: PawSijData, alloc: std.mem.Allocator) void {
        if (self.buf) |buf| alloc.free(buf);
    }
};

const PrimaryForceComponents = struct {
    t0: std.Io.Clock.Timestamp,
    after_ewald: std.Io.Clock.Timestamp,
    after_local: std.Io.Clock.Timestamp,
    after_nonlocal: std.Io.Clock.Timestamp,
    ewald_forces: []math.Vec3,
    local_forces: []math.Vec3,
    nl_forces: ?[]math.Vec3,
};

const OptionalForceComponents = struct {
    after_nlcc: std.Io.Clock.Timestamp,
    resid_forces: ?[]math.Vec3,
    nlcc_forces: ?[]math.Vec3,
    disp_forces: ?[]math.Vec3,
    paw_dhat_forces: ?[]math.Vec3,
};

/// Assemble ForceTerms and emit timing/component logs.
fn finalizeForceTerms(
    alloc: std.mem.Allocator,
    io: std.Io,
    quiet: bool,
    timings: ForceTimings,
    ewald_forces: []math.Vec3,
    local_forces: []math.Vec3,
    nl_forces: ?[]math.Vec3,
    resid_forces: ?[]math.Vec3,
    nlcc_forces: ?[]math.Vec3,
    disp_forces: ?[]math.Vec3,
    paw_dhat_forces: ?[]math.Vec3,
) !ForceTerms {
    const n_atoms = ewald_forces.len;

    // Force timing profile (unbuffered write)
    const log_level: runtime_logging.Level = if (quiet) .warn else .info;
    const logger = runtime_logging.stderr(io, log_level);
    logForceTimings(
        logger,
        quiet,
        timings.t0,
        timings.after_ewald,
        timings.after_local,
        timings.after_nonlocal,
        timings.after_nlcc,
    );

    // Total forces
    const total_forces = try alloc.alloc(math.Vec3, n_atoms);
    sumForceComponents(
        total_forces,
        ewald_forces,
        local_forces,
        nl_forces,
        resid_forces,
        nlcc_forces,
        disp_forces,
        paw_dhat_forces,
    );

    // Debug output for force components
    if (runtime_logging.enabled(log_level, .debug)) {
        logForceComponents(
            logger,
            ewald_forces,
            local_forces,
            nl_forces,
            resid_forces,
            nlcc_forces,
            total_forces,
        );
    }

    return ForceTerms{
        .total = total_forces,
        .ewald = ewald_forces,
        .local = local_forces,
        .nonlocal = nl_forces,
        .residual = resid_forces,
        .nlcc = nlcc_forces,
        .dispersion = disp_forces,
        .paw_dhat = paw_dhat_forces,
    };
}

/// Extract ionic charges and positions for ion-ion force computation.
fn extractChargesAndPositions(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    charges_out: *[]f64,
    positions_out: *[]math.Vec3,
) !void {
    const n_atoms = atoms.len;
    charges_out.* = try alloc.alloc(f64, n_atoms);
    positions_out.* = try alloc.alloc(math.Vec3, n_atoms);
    for (atoms, 0..) |atom, i| {
        charges_out.*[i] = species[atom.species_index].z_valence;
        positions_out.*[i] = atom.position;
    }
}

/// Build the per-species S_ij list for PAW nonlocal forces. Caller owns the optional buffer.
fn buildPawSijList(
    alloc: std.mem.Allocator,
    paw_tabs: ?[]const paw_mod.PawTab,
    list_out: *?[]const []const f64,
    buf_out: *?[][]const f64,
) !void {
    list_out.* = null;
    buf_out.* = null;
    if (paw_tabs) |tabs| {
        const buf = try alloc.alloc([]const f64, tabs.len);
        for (tabs, 0..) |tab, si| {
            buf[si] = if (tab.nbeta > 0) tab.sij else &[_]f64{};
        }
        buf_out.* = buf;
        list_out.* = buf;
    }
}

/// Compute residual forces (from ∇V_Hxc correction) when vresid_g is provided.
fn computeResidualForcesIfAvailable(
    alloc: std.mem.Allocator,
    grid: Grid,
    vresid_g: ?[]const math.Complex,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    rho_atom_tables: ?[]const form_factor.RadialFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
) !?[]math.Vec3 {
    const vresid = vresid_g orelse return null;
    return try residual_force.residualForces(
        alloc,
        grid,
        vresid,
        species,
        atoms,
        rho_atom_tables,
        rho_core_tables,
    );
}

/// Compute PAW D^hat forces when all PAW inputs are provided.
fn computePawDhatForcesIfAvailable(
    alloc: std.mem.Allocator,
    grid: Grid,
    potential_g: ?[]const math.Complex,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_rhoij: ?[]const []const f64,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
) !?[]math.Vec3 {
    if (paw_tabs == null or paw_rhoij == null or potential_g == null) return null;
    return try paw_dhat_force.pawDhatForces(
        alloc,
        grid,
        potential_g.?,
        paw_tabs.?,
        paw_rhoij.?,
        species,
        atoms,
    );
}

/// Compute total forces on atoms.
/// Returns forces in Rydberg/Bohr units.
/// If coulomb_r_cut is non-null, uses direct Coulomb forces instead of Ewald
/// for the ion-ion contribution (isolated/molecular systems).
pub fn computeForces(
    alloc: std.mem.Allocator,
    io: std.Io,
    grid: Grid,
    rho_g: []const math.Complex,
    potential_g: ?[]const math.Complex,
    _: ?[]const math.Complex,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    alpha: f64,
    local_cfg: local_potential.LocalPotentialConfig,
    wavefunctions: ?scf.WavefunctionData,
    vresid_g: ?[]const math.Complex,
    quiet: bool,
    radial_tables: ?[]nonlocal.RadialTableSet,
    precomputed_vxc_r: ?[]const f64,
    rho_atom_tables: ?[]const form_factor.RadialFormFactorTable,
    rho_core_tables: ?[]const form_factor.RadialFormFactorTable,
    ff_tables: ?[]const form_factor.LocalFormFactorTable,
    coulomb_r_cut: ?f64,
    vdw_cfg: config.VdwConfig,
    paw_tabs: ?[]const paw_mod.PawTab,
    paw_dij: ?[]const []const f64,
    paw_rhoij: ?[]const []const f64,
    wavefunctions_down: ?scf.WavefunctionData,
) !ForceTerms {
    const force_input: ForceComputeInput = .{
        .alloc = alloc,
        .io = io,
        .grid = grid,
        .rho_g = rho_g,
        .potential_g = potential_g,
        .species = species,
        .atoms = atoms,
        .cell = cell,
        .recip = recip,
        .volume = volume,
        .alpha = alpha,
        .local_cfg = local_cfg,
        .wavefunctions = wavefunctions,
        .vresid_g = vresid_g,
        .quiet = quiet,
        .radial_tables = radial_tables,
        .precomputed_vxc_r = precomputed_vxc_r,
        .rho_atom_tables = rho_atom_tables,
        .rho_core_tables = rho_core_tables,
        .ff_tables = ff_tables,
        .coulomb_r_cut = coulomb_r_cut,
        .vdw_cfg = vdw_cfg,
        .paw_tabs = paw_tabs,
        .paw_dij = paw_dij,
        .paw_rhoij = paw_rhoij,
        .wavefunctions_down = wavefunctions_down,
    };
    const primary = try computePrimaryForceComponents(force_input);
    const optional = try computeOptionalForceComponents(force_input);
    const timings = ForceTimings{
        .t0 = primary.t0,
        .after_ewald = primary.after_ewald,
        .after_local = primary.after_local,
        .after_nonlocal = primary.after_nonlocal,
        .after_nlcc = optional.after_nlcc,
    };
    return try finalizeForceTerms(
        alloc,
        io,
        quiet,
        timings,
        primary.ewald_forces,
        primary.local_forces,
        primary.nl_forces,
        optional.resid_forces,
        optional.nlcc_forces,
        optional.disp_forces,
        optional.paw_dhat_forces,
    );
}

fn computePrimaryForceComponents(in: ForceComputeInput) !PrimaryForceComponents {
    const ion_data = try IonData.init(in.alloc, in.species, in.atoms);
    defer ion_data.deinit(in.alloc);

    const t0 = std.Io.Clock.Timestamp.now(in.io, .awake);
    const ewald_forces = try computeIonIonForces(
        in.alloc,
        ion_data.charges,
        ion_data.positions,
        in.cell,
        in.recip,
        in.alpha,
        in.quiet,
        in.coulomb_r_cut,
    );
    const after_ewald = std.Io.Clock.Timestamp.now(in.io, .awake);
    const local_forces = try local_force.localPseudoForces(
        in.alloc,
        in.grid,
        in.rho_g,
        in.species,
        in.atoms,
        in.volume,
        in.local_cfg,
        in.ff_tables,
    );
    const after_local = std.Io.Clock.Timestamp.now(in.io, .awake);
    const paw_sij = try PawSijData.init(in.alloc, in.paw_tabs);
    defer paw_sij.deinit(in.alloc);

    const nl_forces = try computeNonlocalForcesTotal(
        in.alloc,
        in.wavefunctions,
        in.wavefunctions_down,
        in.species,
        in.atoms,
        in.recip,
        in.volume,
        in.radial_tables,
        in.paw_dij,
        paw_sij.list,
    );
    return .{
        .t0 = t0,
        .after_ewald = after_ewald,
        .after_local = after_local,
        .after_nonlocal = std.Io.Clock.Timestamp.now(in.io, .awake),
        .ewald_forces = ewald_forces,
        .local_forces = local_forces,
        .nl_forces = nl_forces,
    };
}

fn computeOptionalForceComponents(in: ForceComputeInput) !OptionalForceComponents {
    const resid_forces = try computeResidualForcesIfAvailable(
        in.alloc,
        in.grid,
        in.vresid_g,
        in.species,
        in.atoms,
        in.rho_atom_tables,
        in.rho_core_tables,
    );
    const nlcc_forces = try computeNlccForcesIfAvailable(
        in.alloc,
        in.grid,
        in.rho_g,
        in.potential_g,
        in.precomputed_vxc_r,
        in.species,
        in.atoms,
        in.volume,
        in.rho_core_tables,
    );
    const after_nlcc = std.Io.Clock.Timestamp.now(in.io, .awake);
    return .{
        .after_nlcc = after_nlcc,
        .resid_forces = resid_forces,
        .nlcc_forces = nlcc_forces,
        .disp_forces = try computeDispersionForces(
            in.alloc,
            in.species,
            in.atoms,
            in.cell,
            in.vdw_cfg,
        ),
        .paw_dhat_forces = try computePawDhatForcesIfAvailable(
            in.alloc,
            in.grid,
            in.potential_g,
            in.paw_tabs,
            in.paw_rhoij,
            in.species,
            in.atoms,
        ),
    };
}

/// Compute maximum force magnitude.
pub fn maxForce(forces: []const math.Vec3) f64 {
    var max: f64 = 0.0;
    for (forces) |f| {
        const mag = math.Vec3.norm(f);
        if (mag > max) max = mag;
    }
    return max;
}

/// Compute RMS force.
pub fn rmsForce(forces: []const math.Vec3) f64 {
    if (forces.len == 0) return 0.0;
    var sum: f64 = 0.0;
    for (forces) |f| {
        sum += math.Vec3.dot(f, f);
    }
    return std.math.sqrt(sum / @as(f64, @floatFromInt(forces.len)));
}

test "force utilities" {
    const testing = std.testing;

    const forces = [_]math.Vec3{
        math.Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 },
        math.Vec3{ .x = 0.0, .y = 2.0, .z = 0.0 },
        math.Vec3{ .x = 0.0, .y = 0.0, .z = 3.0 },
    };

    const max = maxForce(&forces);
    try testing.expectApproxEqAbs(max, 3.0, 1e-10);

    const rms = rmsForce(&forces);
    // RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.16
    try testing.expectApproxEqAbs(rms, std.math.sqrt(14.0 / 3.0), 1e-10);
}
