const std = @import("std");
const math = @import("../math/math.zig");
const config = @import("../config/config.zig");
const coulomb_mod = @import("../coulomb/coulomb.zig");
const d3 = @import("../vdw/d3.zig");
const d3_params = @import("../vdw/d3_params.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const ewald = @import("../ewald/ewald.zig");
const form_factor = @import("../pseudopotential/form_factor.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const paw_mod = @import("../paw/paw.zig");
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
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    cell: math.Mat3,
    recip: math.Mat3,
    volume: f64,
    alpha: f64,
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
    const n_atoms = atoms.len;

    // Get ionic charges
    var charges = try alloc.alloc(f64, n_atoms);
    defer alloc.free(charges);
    for (atoms, 0..) |atom, i| {
        charges[i] = species[atom.species_index].z_valence;
    }

    // Get positions
    var positions = try alloc.alloc(math.Vec3, n_atoms);
    defer alloc.free(positions);
    for (atoms, 0..) |atom, i| {
        positions[i] = atom.position;
    }

    const t0 = std.Io.Clock.Timestamp.now(io, .awake);

    // Ion-ion forces: Ewald for periodic, direct Coulomb for isolated
    // Both return forces in Hartree/Bohr, multiply by 2 for Ry/Bohr
    var ewald_forces = try alloc.alloc(math.Vec3, n_atoms);
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
        const ewald_forces_ha = try ewald.ionIonForces(alloc, cell, recip, charges, positions, ewald_params);
        defer alloc.free(ewald_forces_ha);
        // Convert Ewald forces to Rydberg units
        for (ewald_forces_ha, 0..) |f_ha, i| {
            ewald_forces[i] = math.Vec3.scale(f_ha, 2.0);
        }
    }

    const after_ewald = std.Io.Clock.Timestamp.now(io, .awake);

    // Local pseudopotential forces
    const local_forces = try local_force.localPseudoForces(
        alloc,
        grid,
        rho_g,
        species,
        atoms,
        volume,
        alpha,
        ff_tables,
    );

    const after_local = std.Io.Clock.Timestamp.now(io, .awake);

    // Build per-species S_ij arrays for PAW overlap correction in nonlocal force
    var paw_sij_list: ?[]const []const f64 = null;
    var paw_sij_buf: ?[][]const f64 = null;
    if (paw_tabs) |tabs| {
        paw_sij_buf = try alloc.alloc([]const f64, tabs.len);
        for (tabs, 0..) |tab, si| {
            paw_sij_buf.?[si] = if (tab.nbeta > 0) tab.sij else &[_]f64{};
        }
        paw_sij_list = paw_sij_buf.?;
    }
    defer if (paw_sij_buf) |buf| alloc.free(buf);

    // Nonlocal forces (requires wavefunctions)
    var nl_forces: ?[]math.Vec3 = null;
    if (wavefunctions) |wf| {
        const is_spin = wavefunctions_down != null;
        const sf: f64 = if (is_spin) 1.0 else 2.0;
        nl_forces = try nonlocal_force.nonlocalForces(
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
            for (nl_forces.?, 0..) |*f, i| {
                f.* = math.Vec3.add(f.*, nl_down[i]);
            }
        }
    }

    const after_nonlocal = std.Io.Clock.Timestamp.now(io, .awake);

    var resid_forces: ?[]math.Vec3 = null;
    if (vresid_g) |vresid| {
        resid_forces = try residual_force.residualForces(
            alloc,
            grid,
            vresid,
            species,
            atoms,
            rho_atom_tables,
            rho_core_tables,
        );
    }

    var nlcc_forces: ?[]math.Vec3 = null;
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
        nlcc_forces = try nlcc_force.nlccForcesGSpace(
            alloc,
            grid,
            vxc_g,
            species,
            atoms,
            rho_core_tables,
        );
    } else if (potential_g) |pot| {
        const total = grid.nx * grid.ny * grid.nz;
        if (pot.len != total) return error.InvalidGrid;
        // Extract V_xc(G) = V_eff(G) - V_H(G)
        var vxc_g = try alloc.alloc(math.Complex, total);
        defer alloc.free(vxc_g);
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
                    const gvec = math.Vec3.add(
                        math.Vec3.add(math.Vec3.scale(b1x, @as(f64, @floatFromInt(gh))), math.Vec3.scale(b2x, @as(f64, @floatFromInt(gk)))),
                        math.Vec3.scale(b3x, @as(f64, @floatFromInt(gl))),
                    );
                    const g2 = math.Vec3.dot(gvec, gvec);
                    var vh = math.complex.init(0.0, 0.0);
                    if (g2 > 1e-12) {
                        vh = math.complex.scale(rho_g[idx], 8.0 * std.math.pi / g2);
                    }
                    vxc_g[idx] = math.complex.sub(pot[idx], vh);
                    idx += 1;
                }
            }
        }
        // Use G-space NLCC force directly (no need to FFT back to real space)
        nlcc_forces = try nlcc_force.nlccForcesGSpace(
            alloc,
            grid,
            vxc_g,
            species,
            atoms,
            rho_core_tables,
        );
    }

    const after_nlcc = std.Io.Clock.Timestamp.now(io, .awake);

    // D3 dispersion forces
    var disp_forces: ?[]math.Vec3 = null;
    if (vdw_cfg.enabled) {
        const atomic_numbers = try alloc.alloc(usize, n_atoms);
        defer alloc.free(atomic_numbers);
        const atom_positions = try alloc.alloc(math.Vec3, n_atoms);
        defer alloc.free(atom_positions);
        for (atoms, 0..) |atom, idx| {
            atomic_numbers[idx] = d3_params.atomicNumber(species[atom.species_index].symbol) orelse 0;
            atom_positions[idx] = atom.position;
        }
        var damping = d3_params.pbe_d3bj;
        if (vdw_cfg.s6) |v| damping.s6 = v;
        if (vdw_cfg.s8) |v| damping.s8 = v;
        if (vdw_cfg.a1) |v| damping.a1 = v;
        if (vdw_cfg.a2) |v| damping.a2 = v;
        disp_forces = try d3.computeForces(
            alloc,
            atomic_numbers,
            atom_positions,
            cell,
            damping,
            vdw_cfg.cutoff_radius,
            vdw_cfg.cn_cutoff,
        );
    }

    // PAW D^hat forces
    var paw_dhat_forces: ?[]math.Vec3 = null;
    if (paw_tabs != null and paw_rhoij != null and potential_g != null) {
        paw_dhat_forces = try paw_dhat_force.pawDhatForces(
            alloc,
            grid,
            potential_g.?,
            paw_tabs.?,
            paw_rhoij.?,
            species,
            atoms,
        );
    }

    // Force timing profile (unbuffered write)
    {
        const ewald_ms = @as(f64, @floatFromInt(t0.durationTo(after_ewald).raw.nanoseconds)) / 1_000_000.0;
        const local_ms = @as(f64, @floatFromInt(after_ewald.durationTo(after_local).raw.nanoseconds)) / 1_000_000.0;
        const nl_ms = @as(f64, @floatFromInt(after_local.durationTo(after_nonlocal).raw.nanoseconds)) / 1_000_000.0;
        const nlcc_ms = @as(f64, @floatFromInt(after_nonlocal.durationTo(after_nlcc).raw.nanoseconds)) / 1_000_000.0;
        const total_ms = @as(f64, @floatFromInt(t0.durationTo(after_nlcc).raw.nanoseconds)) / 1_000_000.0;
        var buf: [256]u8 = undefined;
        var writer = std.Io.File.stderr().writer(io, &buf);
        const out = &writer.interface;
        out.print("force_profile ewald_ms={d:.1} local_ms={d:.1} nonlocal_ms={d:.1} nlcc_ms={d:.1} total_ms={d:.1}\n", .{ ewald_ms, local_ms, nl_ms, nlcc_ms, total_ms }) catch {};
        out.flush() catch {};
    }

    // Total forces
    var total_forces = try alloc.alloc(math.Vec3, n_atoms);
    for (0..n_atoms) |i| {
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

    // Debug output for force components
    if (!quiet) {
        var buffer: [1024]u8 = undefined;
        var writer = std.Io.File.stderr().writer(io, &buffer);
        const out = &writer.interface;
        out.print("\n=== Force Components (Ry/Bohr) ===\n", .{}) catch {};
        for (0..n_atoms) |i| {
            const ew = ewald_forces[i];
            const loc = local_forces[i];
            const tot = total_forces[i];
            if (nl_forces) |nl| {
                const nlf = nl[i];
                if (resid_forces != null or nlcc_forces != null) {
                    const rf = if (resid_forces) |resid| resid[i] else math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
                    const cf = if (nlcc_forces) |nlcc| nlcc[i] else math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
                    out.print(
                        "Atom {d}: Ewald=({d:.6},{d:.6},{d:.6}) Local=({d:.6},{d:.6},{d:.6}) Nonlocal=({d:.6},{d:.6},{d:.6}) Resid=({d:.6},{d:.6},{d:.6}) NLCC=({d:.6},{d:.6},{d:.6}) Total=({d:.6},{d:.6},{d:.6})\n",
                        .{ i, ew.x, ew.y, ew.z, loc.x, loc.y, loc.z, nlf.x, nlf.y, nlf.z, rf.x, rf.y, rf.z, cf.x, cf.y, cf.z, tot.x, tot.y, tot.z },
                    ) catch {};
                } else {
                    out.print(
                        "Atom {d}: Ewald=({d:.6},{d:.6},{d:.6}) Local=({d:.6},{d:.6},{d:.6}) Nonlocal=({d:.6},{d:.6},{d:.6}) Total=({d:.6},{d:.6},{d:.6})\n",
                        .{ i, ew.x, ew.y, ew.z, loc.x, loc.y, loc.z, nlf.x, nlf.y, nlf.z, tot.x, tot.y, tot.z },
                    ) catch {};
                }
            } else {
                if (resid_forces != null or nlcc_forces != null) {
                    const rf = if (resid_forces) |resid| resid[i] else math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
                    const cf = if (nlcc_forces) |nlcc| nlcc[i] else math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
                    out.print(
                        "Atom {d}: Ewald=({d:.6},{d:.6},{d:.6}) Local=({d:.6},{d:.6},{d:.6}) Resid=({d:.6},{d:.6},{d:.6}) NLCC=({d:.6},{d:.6},{d:.6}) Total=({d:.6},{d:.6},{d:.6})\n",
                        .{ i, ew.x, ew.y, ew.z, loc.x, loc.y, loc.z, rf.x, rf.y, rf.z, cf.x, cf.y, cf.z, tot.x, tot.y, tot.z },
                    ) catch {};
                } else {
                    out.print(
                        "Atom {d}: Ewald=({d:.6},{d:.6},{d:.6}) Local=({d:.6},{d:.6},{d:.6}) Total=({d:.6},{d:.6},{d:.6})\n",
                        .{ i, ew.x, ew.y, ew.z, loc.x, loc.y, loc.z, tot.x, tot.y, tot.z },
                    ) catch {};
                }
            }
        }
        out.flush() catch {};
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
