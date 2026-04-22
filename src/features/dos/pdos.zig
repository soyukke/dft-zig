const std = @import("std");
const math = @import("../math/math.zig");
const scf = @import("../scf/scf.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const dos = @import("dos.zig");

const gaussianDelta = dos.gaussianDelta;

/// A single PDOS projection channel.
pub const PdosProjection = struct {
    atom_index: usize,
    l: i32,
    label: ?[]const u8,
    pdos: []f64,
};

/// Full PDOS result.
pub const PdosResult = struct {
    energies: []f64,
    projections: []PdosProjection,

    pub fn deinit(self: *PdosResult, alloc: std.mem.Allocator) void {
        if (self.energies.len > 0) alloc.free(self.energies);
        for (self.projections) |*p| {
            alloc.free(p.pdos);
        }
        if (self.projections.len > 0) alloc.free(self.projections);
    }
};

/// Compute projected DOS.
///
/// PDOS_{a,l}(E) = Σ_{n,k} w_k Σ_m |<φ_{a,l,m}|ψ_{nk}>|² × δ_σ(E - ε_{nk})
///
/// where φ_{a,l,m}(G) = (4π/√Ω) × i^l × R_l(|k+G|) × Y_{lm}(k̂+Ĝ) × exp(-i(k+G)·τ_a)
pub fn computePdos(
    alloc: std.mem.Allocator,
    wf_data: scf.WavefunctionData,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    recip: math.Mat3,
    volume: f64,
    sigma: f64,
    npoints: usize,
    emin_opt: ?f64,
    emax_opt: ?f64,
    nspin: usize,
) !PdosResult {
    // Find energy range
    var e_min: f64 = std.math.inf(f64);
    var e_max: f64 = -std.math.inf(f64);
    for (wf_data.kpoints) |kp| {
        for (kp.eigenvalues) |e| {
            if (e < e_min) e_min = e;
            if (e > e_max) e_max = e;
        }
    }
    const emin = emin_opt orelse (e_min - 5.0 * sigma);
    const emax = emax_opt orelse (e_max + 5.0 * sigma);
    const de = (emax - emin) / @as(f64, @floatFromInt(npoints - 1));

    const energies = try alloc.alloc(f64, npoints);
    errdefer alloc.free(energies);
    for (0..npoints) |i| {
        energies[i] = emin + @as(f64, @floatFromInt(i)) * de;
    }

    // Count total projections: one per (atom, wfc)
    var n_proj: usize = 0;
    for (atoms) |atom| {
        const upf = species[atom.species_index].upf;
        n_proj += upf.atomic_wfc.len;
    }

    if (n_proj == 0) {
        return PdosResult{
            .energies = energies,
            .projections = &[_]PdosProjection{},
        };
    }

    const projections = try alloc.alloc(PdosProjection, n_proj);
    errdefer alloc.free(projections);

    // Initialize projections
    var pi: usize = 0;
    for (atoms, 0..) |atom, ai| {
        const upf = species[atom.species_index].upf;
        for (upf.atomic_wfc) |wfc| {
            const pdos_arr = try alloc.alloc(f64, npoints);
            @memset(pdos_arr, 0.0);
            projections[pi] = .{
                .atom_index = ai,
                .l = wfc.l,
                .label = wfc.label,
                .pdos = pdos_arr,
            };
            pi += 1;
        }
    }

    // Build radial tables for atomic wavefunctions (one per species)
    const n_species = species.len;
    const wfc_tables = try alloc.alloc([]nonlocal.RadialTable, n_species);
    defer {
        for (wfc_tables) |tables| {
            for (tables) |*t| {
                @constCast(t).deinit(alloc);
            }
            if (tables.len > 0) alloc.free(tables);
        }
        alloc.free(wfc_tables);
    }

    // Compute g_max from ecut
    const g_max = std.math.sqrt(wf_data.ecut_ry) * 2.0; // generous margin

    for (0..n_species) |si| {
        const upf = species[si].upf;
        const n_wfc = upf.atomic_wfc.len;
        if (n_wfc == 0) {
            wfc_tables[si] = &[_]nonlocal.RadialTable{};
            continue;
        }
        const tables = try alloc.alloc(nonlocal.RadialTable, n_wfc);
        errdefer alloc.free(tables);
        for (0..n_wfc) |wi| {
            tables[wi] = try nonlocal.RadialTable.init(
                alloc,
                upf.atomic_wfc[wi].values,
                upf.r,
                upf.rab,
                upf.atomic_wfc[wi].l,
                g_max,
                2048,
            );
        }
        wfc_tables[si] = tables;
    }

    const inv_sigma_sqrt2pi = 1.0 / (sigma * std.math.sqrt(2.0 * std.math.pi));
    const inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    const spin_factor: f64 = if (nspin == 2) 1.0 else 2.0;
    const cutoff = 5.0 * sigma;

    // Projection normalization: <φ|ψ> = (1/Ω) Σ_G conj(φ_G) c_G
    // (same convention as nonlocal <β|ψ> in apply.zig)
    const inv_volume = 1.0 / volume;

    // For each k-point
    for (wf_data.kpoints) |kp| {
        // Generate basis for this k-point
        var basis = try plane_wave.generate(alloc, recip, wf_data.ecut_ry, kp.k_cart);
        defer basis.deinit(alloc);
        const gvecs = basis.gvecs;
        const n_pw = gvecs.len;

        // For each projection channel, compute |<φ|ψ_n>|² for all bands
        pi = 0;
        for (atoms) |atom| {
            const si = atom.species_index;
            const upf = species[si].upf;

            for (upf.atomic_wfc, 0..) |wfc, wi| {
                const l = wfc.l;
                const table = &wfc_tables[si][wi];

                // Sum over m = -l..l
                var m: i32 = -l;
                while (m <= l) : (m += 1) {
                    // Build φ_{a,l,m}(G) for this k-point
                    // φ(k+G) = 4π × i^l × R_l(|k+G|) × Y_{lm}(k̂+Ĝ) × exp(-i(k+G)·τ_a)
                    // i^l: l=0 -> 1, l=1 -> i, l=2 -> -1, l=3 -> -i
                    const il_r: f64 = switch (@mod(l, 4)) {
                        0 => 1.0,
                        2 => -1.0,
                        else => 0.0,
                    };
                    const il_i: f64 = switch (@mod(l, 4)) {
                        1 => 1.0,
                        3 => -1.0,
                        else => 0.0,
                    };

                    // For each band, compute <φ|ψ_n> = Σ_G conj(φ(k+G)) × c_n(G)
                    for (0..kp.nbands) |n| {
                        var proj_r: f64 = 0.0;
                        var proj_i: f64 = 0.0;

                        for (0..n_pw) |g| {
                            const kpg = gvecs[g].kpg;
                            const kpg_norm = math.Vec3.norm(kpg);
                            const radial = table.eval(kpg_norm);

                            // Spherical harmonic
                            var ylm: f64 = undefined;
                            if (kpg_norm < 1e-12) {
                                // At G=0: Y_00 = 1/√(4π), Y_{l>0} = 0
                                ylm = if (l == 0 and m == 0)
                                    nonlocal.realSphericalHarmonic(0, 0, 0, 0, 1.0)
                                else
                                    0.0;
                            } else {
                                ylm = nonlocal.realSphericalHarmonic(l, m, kpg.x, kpg.y, kpg.z);
                            }

                            const four_pi = 4.0 * std.math.pi;
                            // φ(k+G) = 4π × i^l × R_l × Y_{lm} × exp(-i(k+G)·τ_a)
                            const phase_angle = -math.Vec3.dot(kpg, atom.position);
                            const phase = math.complex.expi(phase_angle);
                            const factor = four_pi * radial * ylm;

                            // i^l × exp(-i(k+G)·τ_a)
                            const il_phase_r = il_r * phase.r - il_i * phase.i;
                            const il_phase_i = il_r * phase.i + il_i * phase.r;

                            // φ(k+G) = factor × (il_phase_r + i×il_phase_i)
                            const phi_r = factor * il_phase_r;
                            const phi_i = factor * il_phase_i;

                            // conj(φ) × c_n
                            const c = kp.coefficients[g + n * n_pw];
                            // conj(φ) = (phi_r - i×phi_i)
                            // conj(φ)×c = (phi_r×c.r + phi_i×c.i)
                            //           + i×(phi_r×c.i - phi_i×c.r)
                            proj_r += phi_r * c.r + phi_i * c.i;
                            proj_i += phi_r * c.i - phi_i * c.r;
                        }

                        // |<φ|ψ>|² for this m (with 1/Ω normalization)
                        const proj_sq = (proj_r * proj_r + proj_i * proj_i) * inv_volume;

                        // Add to PDOS with Gaussian broadening
                        const ek = kp.eigenvalues[n];
                        for (0..npoints) |ie| {
                            const diff = energies[ie] - ek;
                            if (@abs(diff) <= cutoff) {
                                projections[pi].pdos[ie] += kp.weight * spin_factor * proj_sq *
                                    gaussianDelta(diff, inv_sigma_sqrt2pi, inv_2sigma2);
                            }
                        }
                    }
                }
                pi += 1;
            }
        }
    }

    return PdosResult{
        .energies = energies,
        .projections = projections,
    };
}

/// Write PDOS to CSV file.
pub fn writePdosCSV(io: std.Io, dir: std.Io.Dir, result: PdosResult, fermi_level: f64) !void {
    const file = try dir.createFile(io, "pdos.csv", .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var writer = file.writer(io, &buf);
    const out = &writer.interface;

    const ef = if (std.math.isNan(fermi_level)) @as(f64, 0.0) else fermi_level;

    // Header
    try out.print("energy_ry,energy_shifted_ry", .{});
    for (result.projections) |p| {
        const l_name: []const u8 = switch (p.l) {
            0 => "s",
            1 => "p",
            2 => "d",
            3 => "f",
            else => "?",
        };
        if (p.label) |lbl| {
            try out.print(",atom{d}_{s}_{s}", .{ p.atom_index, lbl, l_name });
        } else {
            try out.print(",atom{d}_l{d}_{s}", .{ p.atom_index, p.l, l_name });
        }
    }
    try out.print("\n", .{});

    // Data
    for (0..result.energies.len) |i| {
        try out.print("{d:.8},{d:.8}", .{ result.energies[i], result.energies[i] - ef });
        for (result.projections) |p| {
            try out.print(",{d:.8}", .{p.pdos[i]});
        }
        try out.print("\n", .{});
    }
    try out.flush();
}
