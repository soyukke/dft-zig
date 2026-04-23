const std = @import("std");
const math = @import("../math/math.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const scf = @import("../scf/scf.zig");
const test_support = @import("../../test_support.zig");

/// Projector data for one species, used for analytical force computation.
const ForceProjectors = struct {
    species_index: usize,
    beta_count: usize,
    g_count: usize,
    l_list: []i32,
    coeffs: []const f64, // D_ij matrix (borrowed from UPF)
    m_offsets: []usize,
    m_counts: []usize,
    m_total: usize,
    phi: []f64, // phi[bm * g_count + g] = 4π β_l(|k+G|) Y_lm(k+G)

    fn deinit(self: *ForceProjectors, alloc: std.mem.Allocator) void {
        if (self.l_list.len > 0) alloc.free(self.l_list);
        if (self.m_offsets.len > 0) alloc.free(self.m_offsets);
        if (self.m_counts.len > 0) alloc.free(self.m_counts);
        if (self.phi.len > 0) alloc.free(self.phi);
    }
};

/// Fill per-beta radial projector values at each (k+G) magnitude, populate l_list /
/// m_offsets / m_counts, and return the total number of (b, m) channels.
fn fillRadialProjectors(
    upf: pseudo.UpfData,
    gvecs: []plane_wave.GVector,
    radial_tables: ?nonlocal.RadialTableSet,
    l_list: []i32,
    m_offsets: []usize,
    m_counts: []usize,
    radial_buf: []f64,
) usize {
    const beta_count = upf.beta.len;
    const g_count = gvecs.len;
    var total_m: usize = 0;
    var b: usize = 0;
    while (b < beta_count) : (b += 1) {
        const l_val = upf.beta[b].l orelse 0;
        l_list[b] = l_val;
        const m_count = @as(usize, @intCast(2 * l_val + 1));
        m_offsets[b] = total_m;
        m_counts[b] = m_count;
        total_m += m_count;
        if (radial_tables) |tables| {
            // Fast path: use pre-computed lookup table (O(1) per G-vector)
            for (0..g_count) |g| {
                const gmag = math.Vec3.norm(gvecs[g].kpg);
                radial_buf[b * g_count + g] = tables.tables[b].eval(gmag);
            }
        } else {
            // Slow path: direct radial projector computation (O(N_r) per G-vector)
            var g: usize = 0;
            while (g < g_count) : (g += 1) {
                const gmag = math.Vec3.norm(gvecs[g].kpg);
                radial_buf[b * g_count + g] = nonlocal.radialProjector(
                    upf.beta[b].values,
                    upf.r,
                    upf.rab,
                    l_val,
                    gmag,
                );
            }
        }
    }
    return total_m;
}

/// Build phi[bm * g_count + g] = 4π β_l(|k+G|) Y_lm(k+G) from radial values.
fn buildPhiFromRadial(
    gvecs: []plane_wave.GVector,
    l_list: []i32,
    m_offsets: []usize,
    m_counts: []usize,
    radial_buf: []f64,
    phi: []f64,
) void {
    const beta_count = l_list.len;
    const g_count = gvecs.len;
    var b: usize = 0;
    while (b < beta_count) : (b += 1) {
        const l_val = l_list[b];
        const m_count = m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const m = @as(i32, @intCast(m_idx)) - l_val;
            const offset = m_offsets[b] + m_idx;
            var g: usize = 0;
            while (g < g_count) : (g += 1) {
                const kpg = gvecs[g].kpg;
                const ylm = nonlocal.realSphericalHarmonic(l_val, m, kpg.x, kpg.y, kpg.z);
                const base = radial_buf[b * g_count + g];
                phi[offset * g_count + g] = 4.0 * std.math.pi * base * ylm;
            }
        }
    }
}

/// Build projector arrays for a species. Mirrors buildNonlocalSpeciesWithTables in apply.zig.
fn buildForceProjectors(
    alloc: std.mem.Allocator,
    species_index: usize,
    upf: pseudo.UpfData,
    gvecs: []plane_wave.GVector,
    radial_tables: ?nonlocal.RadialTableSet,
) !ForceProjectors {
    const beta_count = upf.beta.len;
    const g_count = gvecs.len;
    if (beta_count == 0 or upf.dij.len == 0) return error.InvalidPseudopotential;
    if (upf.dij.len != beta_count * beta_count) return error.InvalidPseudopotential;

    const l_list = try alloc.alloc(i32, beta_count);
    errdefer alloc.free(l_list);

    const m_offsets = try alloc.alloc(usize, beta_count);
    errdefer alloc.free(m_offsets);

    const m_counts = try alloc.alloc(usize, beta_count);
    errdefer alloc.free(m_counts);

    const radial_buf = try alloc.alloc(f64, beta_count * g_count);
    defer alloc.free(radial_buf);

    const total_m = fillRadialProjectors(
        upf,
        gvecs,
        radial_tables,
        l_list,
        m_offsets,
        m_counts,
        radial_buf,
    );

    const phi = try alloc.alloc(f64, total_m * g_count);
    errdefer alloc.free(phi);

    buildPhiFromRadial(gvecs, l_list, m_offsets, m_counts, radial_buf, phi);

    return ForceProjectors{
        .species_index = species_index,
        .beta_count = beta_count,
        .g_count = g_count,
        .l_list = l_list,
        .coeffs = upf.dij,
        .m_offsets = m_offsets,
        .m_counts = m_counts,
        .m_total = total_m,
        .phi = phi,
    };
}

/// Build projectors for each species for a single k-point (null entries signal
/// species without nonlocal channels).
fn buildSpeciesProjectors(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    gvecs: []plane_wave.GVector,
    radial_tables_list: ?[]nonlocal.RadialTableSet,
    projectors: []?ForceProjectors,
) !void {
    for (species, 0..) |entry, si| {
        const upf = entry.upf;
        if (upf.beta.len == 0 or upf.dij.len == 0) {
            projectors[si] = null;
            continue;
        }
        const tables = if (radial_tables_list) |rtl|
            if (si < rtl.len) rtl[si] else null
        else
            null;
        projectors[si] = try buildForceProjectors(alloc, si, upf.*, gvecs, tables);
    }
}

/// Step A: p_bm = Σ_G phi_bm(G) × phase(G) × c(G).
fn projectNonlocalBandCoeffs(
    proj: ForceProjectors,
    n: usize,
    phase_buf: []const math.Complex,
    c: []const math.Complex,
    coeff: []math.Complex,
) void {
    var b: usize = 0;
    while (b < proj.beta_count) : (b += 1) {
        const m_off = proj.m_offsets[b];
        const m_count = proj.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            const phi_row = proj.phi[(m_off + m_idx) * n .. (m_off + m_idx + 1) * n];
            var sum = math.complex.init(0.0, 0.0);
            var g: usize = 0;
            while (g < n) : (g += 1) {
                // phi is real, phase×c is complex
                const pc = math.complex.mul(phase_buf[g], c[g]);
                sum = math.complex.add(sum, math.complex.scale(pc, phi_row[g]));
            }
            coeff[m_off + m_idx] = sum;
        }
    }
}

/// Step B: Dp_bm = Σ_j (D_bj - ε_nk × q_bj) × p_jm (same l).
fn applyDMatrixToProjectors(
    proj: ForceProjectors,
    atom_idx: usize,
    species_index: usize,
    eigenvalue: f64,
    paw_dij: ?[]const []const f64,
    paw_sij: ?[]const []const f64,
    coeff: []const math.Complex,
    coeff2: []math.Complex,
) void {
    const nb = proj.beta_count;
    const use_paw_dij = paw_dij != null and atom_idx < paw_dij.?.len;
    const use_paw_sij = paw_sij != null and species_index < paw_sij.?.len;
    var b: usize = 0;
    while (b < nb) : (b += 1) {
        const l_val = proj.l_list[b];
        const m_off = proj.m_offsets[b];
        const m_count = proj.m_counts[b];
        var m_idx: usize = 0;
        while (m_idx < m_count) : (m_idx += 1) {
            var sum = math.complex.init(0.0, 0.0);
            var j: usize = 0;
            while (j < nb) : (j += 1) {
                if (proj.l_list[j] != l_val) continue;
                // D_ij: use per-atom PAW D_full or UPF D^0
                const d_val = if (use_paw_dij)
                    paw_dij.?[atom_idx][b * nb + j]
                else
                    proj.coeffs[b * nb + j];
                // q_ij = S_ij - δ_ij (overlap augmentation)
                const q_val = if (use_paw_sij) blk: {
                    const sij_arr = paw_sij.?[species_index];
                    const delta: f64 = if (b == j) 1.0 else 0.0;
                    break :blk sij_arr[b * nb + j] - delta;
                } else 0.0;
                const eff = d_val - eigenvalue * q_val;
                if (eff == 0.0) continue;
                sum = math.complex.add(
                    sum,
                    math.complex.scale(coeff[proj.m_offsets[j] + m_idx], eff),
                );
            }
            coeff2[m_off + m_idx] = sum;
        }
    }
}

/// Compute the maximum m_total across all non-null species projectors.
fn maxMTotal(projectors: []const ?ForceProjectors) usize {
    var max_m_total: usize = 0;
    for (projectors) |p| {
        if (p) |proj| {
            if (proj.m_total > max_m_total) max_m_total = proj.m_total;
        }
    }
    return max_m_total;
}

/// Aggregate inputs for accumulating nonlocal forces from one k-point.
const KpointForceInputs = struct {
    gvecs: []plane_wave.GVector,
    n: usize,
    kp_wf: scf.KpointWavefunction,
    projectors: []const ?ForceProjectors,
    phase_buf: []math.Complex,
    coeff_buf: []math.Complex,
    coeff2_buf: []math.Complex,
    paw_dij: ?[]const []const f64,
    paw_sij: ?[]const []const f64,
    spin_factor: f64,
    inv_volume: f64,
};

/// Accumulate nonlocal force contributions from a single atom at a single k-point.
fn accumulateAtomForceAtKpoint(
    inputs: KpointForceInputs,
    atom: hamiltonian.AtomData,
    atom_idx: usize,
    force_out: *math.Vec3,
) void {
    const proj_opt = inputs.projectors[atom.species_index];
    const proj = proj_opt orelse return;

    // Compute phase[g] = exp(+i G·R)
    for (inputs.gvecs, 0..) |gv, g| {
        inputs.phase_buf[g] = math.complex.expi(math.Vec3.dot(gv.cart, atom.position));
    }

    var band: usize = 0;
    while (band < inputs.kp_wf.nbands) : (band += 1) {
        const occ = inputs.kp_wf.occupations[band];
        if (occ <= 0.0) continue;

        const c = inputs.kp_wf.coefficients[band * inputs.n .. (band + 1) * inputs.n];
        const coeff = inputs.coeff_buf[0..proj.m_total];
        const coeff2 = inputs.coeff2_buf[0..proj.m_total];

        // Step A: Project p_bm = Σ_G phi_bm(G) × phase(G) × c(G)
        projectNonlocalBandCoeffs(proj, inputs.n, inputs.phase_buf, c, coeff);

        // Step B: D-apply Dp_bm = Σ_j (D_bj - ε_nk × q_bj) × p_jm (same l)
        // For PAW: D_bj = D_full (per-atom), q_bj = S_bj - δ_bj
        // For NCPP: D_bj = D^0 (from UPF), q_bj = 0
        applyDMatrixToProjectors(
            proj,
            atom_idx,
            atom.species_index,
            inputs.kp_wf.eigenvalues[band],
            inputs.paw_dij,
            inputs.paw_sij,
            coeff,
            coeff2,
        );

        // Steps C+D: Back-project q(G) and accumulate forces
        // q(G) = Σ_bm conj(Dp_bm) × phi_bm(G)
        // F_α += prefactor × Im[ G_α × q(G) × phase(G) × c(G) ]
        const prefactor = 2.0 * occ * inputs.kp_wf.weight *
            inputs.spin_factor * inputs.inv_volume;
        accumulateBandForce(
            proj,
            inputs.n,
            inputs.gvecs,
            inputs.phase_buf,
            c,
            coeff2,
            prefactor,
            force_out,
        );
    }
}

/// Steps C+D: Back-project q(G) and accumulate forces for a single band.
fn accumulateBandForce(
    proj: ForceProjectors,
    n: usize,
    gvecs: []const plane_wave.GVector,
    phase_buf: []const math.Complex,
    c: []const math.Complex,
    coeff2: []const math.Complex,
    prefactor: f64,
    force_out: *math.Vec3,
) void {
    var fx: f64 = 0.0;
    var fy: f64 = 0.0;
    var fz: f64 = 0.0;

    var g: usize = 0;
    while (g < n) : (g += 1) {
        // Compute q(G) = Σ_bm conj(Dp_bm) × phi_bm(G)
        var q = math.complex.init(0.0, 0.0);
        var b: usize = 0;
        while (b < proj.beta_count) : (b += 1) {
            const m_off = proj.m_offsets[b];
            const m_count = proj.m_counts[b];
            var m_idx: usize = 0;
            while (m_idx < m_count) : (m_idx += 1) {
                const phi_val = proj.phi[(m_off + m_idx) * n + g];
                const dp_conj = math.complex.conj(coeff2[m_off + m_idx]);
                q = math.complex.add(q, math.complex.scale(dp_conj, phi_val));
            }
        }

        // z = q(G) × phase(G) × c(G)
        const z = math.complex.mul(q, math.complex.mul(phase_buf[g], c[g]));

        // G_α × Im(z)
        const g_cart = gvecs[g].cart;
        fx += g_cart.x * z.i;
        fy += g_cart.y * z.i;
        fz += g_cart.z * z.i;
    }

    force_out.x += prefactor * fx;
    force_out.y += prefactor * fy;
    force_out.z += prefactor * fz;
}

/// Accumulate nonlocal force contributions from all atoms at a single k-point.
fn accumulateKpointForces(
    alloc: std.mem.Allocator,
    kp_wf: scf.KpointWavefunction,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    ecut_ry: f64,
    radial_tables_list: ?[]nonlocal.RadialTableSet,
    paw_dij: ?[]const []const f64,
    paw_sij: ?[]const []const f64,
    spin_factor: f64,
    inv_volume: f64,
    forces: []math.Vec3,
) !void {
    var basis = try plane_wave.generate(alloc, recip, ecut_ry, kp_wf.k_cart);
    defer basis.deinit(alloc);

    const gvecs = basis.gvecs;
    const n = gvecs.len;
    if (n != kp_wf.basis_len) return;

    // Build projectors per species
    const projectors = try alloc.alloc(?ForceProjectors, species.len);
    defer {
        for (projectors) |*p| {
            if (p.*) |*proj| proj.deinit(alloc);
        }
        alloc.free(projectors);
    }

    try buildSpeciesProjectors(alloc, species, gvecs, radial_tables_list, projectors);

    // Allocate work buffers
    const max_m_total = maxMTotal(projectors);

    const phase_buf = try alloc.alloc(math.Complex, n);
    defer alloc.free(phase_buf);

    const coeff_buf = try alloc.alloc(math.Complex, max_m_total);
    defer alloc.free(coeff_buf);

    const coeff2_buf = try alloc.alloc(math.Complex, max_m_total);
    defer alloc.free(coeff2_buf);

    const inputs = KpointForceInputs{
        .gvecs = gvecs,
        .n = n,
        .kp_wf = kp_wf,
        .projectors = projectors,
        .phase_buf = phase_buf,
        .coeff_buf = coeff_buf,
        .coeff2_buf = coeff2_buf,
        .paw_dij = paw_dij,
        .paw_sij = paw_sij,
        .spin_factor = spin_factor,
        .inv_volume = inv_volume,
    };

    for (atoms, 0..) |atom, atom_idx| {
        accumulateAtomForceAtKpoint(inputs, atom, atom_idx, &forces[atom_idx]);
    }
}

/// Compute analytical nonlocal pseudopotential forces (Hellmann-Feynman).
///
/// F_{I,α} = +2 × spin × Σ_{nk} f_{nk} w_k × inv_vol ×
///            Im[ Σ_G G_α × q(G) × phase(G) × c(G) ]
///
/// where q(G) = Σ_{bm} conj(Dp_{bm}) × phi_{bm}(G),
///       Dp_{bm} = Σ_j D_{bj} × p_{jm},
///       p_{bm} = Σ_G phi_{bm}(G) × phase(G) × c(G),
///       phase(G) = exp(+i G·R_I).
///
/// For PAW: D_ij → D_full (per-atom) and add eigenvalue-weighted overlap correction:
///   Dp_{bm} = Σ_j (D_bj - ε_nk × q_bj) × p_{jm}
///   where q_ij = S_ij - δ_ij (augmentation overlap).
///
/// Returns forces in Rydberg/Bohr units.
pub fn nonlocalForces(
    alloc: std.mem.Allocator,
    wavefunctions: scf.WavefunctionData,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    radial_tables_list: ?[]nonlocal.RadialTableSet,
    paw_dij: ?[]const []const f64,
    paw_sij: ?[]const []const f64,
    spin_factor: f64,
) ![]math.Vec3 {
    const n_atoms = atoms.len;
    if (n_atoms == 0) return &[_]math.Vec3{};

    const forces = try alloc.alloc(math.Vec3, n_atoms);
    errdefer alloc.free(forces);

    for (forces) |*f| {
        f.* = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    }

    const inv_volume = 1.0 / volume;

    for (wavefunctions.kpoints) |kp_wf| {
        try accumulateKpointForces(
            alloc,
            kp_wf,
            species,
            atoms,
            recip,
            wavefunctions.ecut_ry,
            radial_tables_list,
            paw_dij,
            paw_sij,
            spin_factor,
            inv_volume,
            forces,
        );
    }

    return forces;
}

test "nonlocal force basic" {
    const testing = std.testing;
    _ = testing;
}

test "nonlocal force analytical vs finite difference" {
    const io = std.testing.io;
    const testing = std.testing;
    const alloc = testing.allocator;
    try test_support.requireFile(io, "pseudo/Si.upf");

    const element_buf: [2]u8 = .{ 'S', 'i' };
    var path_buf: [24]u8 = undefined;
    const path_slice = "pseudo/Si.upf";
    @memcpy(path_buf[0..path_slice.len], path_slice);

    const spec = pseudo.Spec{
        .element = element_buf[0..2],
        .path = path_buf[0..path_slice.len],
        .format = .upf,
    };

    var parsed = try pseudo.load(alloc, io, spec);
    defer parsed.deinit(alloc);

    var parsed_items = [_]pseudo.Parsed{parsed};
    const species_entries = try hamiltonian.buildSpeciesEntries(alloc, parsed_items[0..]);
    defer {
        for (species_entries) |*entry| {
            entry.deinit();
        }
        alloc.free(species_entries);
    }

    const a = 8.0;
    const recip = math.Mat3{ .m = .{
        .{ 2.0 * std.math.pi / a, 0.0, 0.0 },
        .{ 0.0, 2.0 * std.math.pi / a, 0.0 },
        .{ 0.0, 0.0, 2.0 * std.math.pi / a },
    } };
    const volume = a * a * a;

    const atom_pos = math.Vec3{ .x = 0.3, .y = 0.4, .z = 0.2 };
    const atoms_arr = [_]hamiltonian.AtomData{
        .{ .position = atom_pos, .species_index = 0 },
    };

    const k_cart = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const ecut_ry = 6.0;
    var basis = try plane_wave.generate(alloc, recip, ecut_ry, k_cart);
    defer basis.deinit(alloc);

    const n = basis.gvecs.len;
    try testing.expect(n > 0);

    const eigenvalues = try alloc.alloc(f64, 1);
    defer alloc.free(eigenvalues);

    eigenvalues[0] = 0.0;

    const occupations = try alloc.alloc(f64, 1);
    defer alloc.free(occupations);

    occupations[0] = 1.0;

    const coefficients = try alloc.alloc(math.Complex, n);
    defer alloc.free(coefficients);

    for (coefficients, 0..) |*c_val, i| {
        const re = 0.05 * @as(f64, @floatFromInt(i + 1));
        const im = -0.03 * @as(f64, @floatFromInt(i + 2));
        c_val.* = math.complex.init(re, im);
    }

    const kpoints = try alloc.alloc(scf.KpointWavefunction, 1);
    defer alloc.free(kpoints);

    kpoints[0] = .{
        .k_frac = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .k_cart = k_cart,
        .weight = 1.0,
        .basis_len = n,
        .nbands = 1,
        .eigenvalues = eigenvalues,
        .coefficients = coefficients,
        .occupations = occupations,
    };

    const wf = scf.WavefunctionData{
        .kpoints = kpoints,
        .ecut_ry = ecut_ry,
        .fermi_level = 0.0,
    };

    // Analytical forces (NCPP mode: no PAW)
    const forces = try nonlocalForces(
        alloc,
        wf,
        species_entries,
        atoms_arr[0..],
        recip,
        volume,
        null,
        null,
        null,
        2.0,
    );
    defer alloc.free(forces);

    // Finite-difference reference
    const inv_volume = 1.0 / volume;
    const spin_factor = 2.0;
    const delta = 1e-5;

    const nonlocalEnergyEval = struct {
        fn eval(
            alloc_inner: std.mem.Allocator,
            gvecs: []plane_wave.GVector,
            sp: []hamiltonian.SpeciesEntry,
            atoms_local: []hamiltonian.AtomData,
            inv_vol: f64,
            psi: []const math.Complex,
            occ: f64,
        ) !f64 {
            const vnl = try hamiltonian.buildNonlocalMatrix(
                alloc_inner,
                gvecs,
                sp,
                atoms_local,
                inv_vol,
            );
            defer alloc_inner.free(vnl);

            var sum = math.complex.init(0.0, 0.0);
            var i: usize = 0;
            while (i < gvecs.len) : (i += 1) {
                const ci = psi[i];
                var tmp = math.complex.init(0.0, 0.0);
                var j: usize = 0;
                while (j < gvecs.len) : (j += 1) {
                    const cj = psi[j];
                    const hij = vnl[i + j * gvecs.len];
                    tmp = math.complex.add(tmp, math.complex.mul(hij, cj));
                }
                sum = math.complex.add(sum, math.complex.mul(math.complex.conj(ci), tmp));
            }
            return sum.r * occ * spin_factor;
        }
    };

    const fx_num = blk: {
        var atoms_plus = atoms_arr;
        var atoms_minus = atoms_arr;
        atoms_plus[0].position.x += delta;
        atoms_minus[0].position.x -= delta;
        const e_plus = try nonlocalEnergyEval.eval(
            alloc,
            basis.gvecs,
            species_entries,
            atoms_plus[0..],
            inv_volume,
            coefficients,
            occupations[0],
        );
        const e_minus = try nonlocalEnergyEval.eval(
            alloc,
            basis.gvecs,
            species_entries,
            atoms_minus[0..],
            inv_volume,
            coefficients,
            occupations[0],
        );
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };
    const fy_num = blk: {
        var atoms_plus = atoms_arr;
        var atoms_minus = atoms_arr;
        atoms_plus[0].position.y += delta;
        atoms_minus[0].position.y -= delta;
        const e_plus = try nonlocalEnergyEval.eval(
            alloc,
            basis.gvecs,
            species_entries,
            atoms_plus[0..],
            inv_volume,
            coefficients,
            occupations[0],
        );
        const e_minus = try nonlocalEnergyEval.eval(
            alloc,
            basis.gvecs,
            species_entries,
            atoms_minus[0..],
            inv_volume,
            coefficients,
            occupations[0],
        );
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };
    const fz_num = blk: {
        var atoms_plus = atoms_arr;
        var atoms_minus = atoms_arr;
        atoms_plus[0].position.z += delta;
        atoms_minus[0].position.z -= delta;
        const e_plus = try nonlocalEnergyEval.eval(
            alloc,
            basis.gvecs,
            species_entries,
            atoms_plus[0..],
            inv_volume,
            coefficients,
            occupations[0],
        );
        const e_minus = try nonlocalEnergyEval.eval(
            alloc,
            basis.gvecs,
            species_entries,
            atoms_minus[0..],
            inv_volume,
            coefficients,
            occupations[0],
        );
        break :blk -(e_plus - e_minus) / (2.0 * delta);
    };

    try testing.expectApproxEqAbs(forces[0].x, fx_num, 1e-4);
    try testing.expectApproxEqAbs(forces[0].y, fy_num, 1e-4);
    try testing.expectApproxEqAbs(forces[0].z, fz_num, 1e-4);
}
