const std = @import("std");
const math = @import("../math/math.zig");
const scf = @import("../scf/scf.zig");
const pseudo = @import("../pseudopotential/pseudopotential.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const dos = @import("dos.zig");

const gaussian_delta = dos.gaussian_delta;

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
const BroadeningParams = struct {
    inv_sigma_sqrt2pi: f64,
    inv_2sigma2: f64,
    spin_factor: f64,
    cutoff: f64,
    inv_volume: f64,
};

fn broadening_params(sigma: f64, volume: f64, nspin: usize) BroadeningParams {
    return .{
        .inv_sigma_sqrt2pi = 1.0 / (sigma * std.math.sqrt(2.0 * std.math.pi)),
        .inv_2sigma2 = 1.0 / (2.0 * sigma * sigma),
        .spin_factor = if (nspin == 2) 1.0 else 2.0,
        .cutoff = 5.0 * sigma,
        // Projection normalization: <φ|ψ> = (1/Ω) Σ_G conj(φ_G) c_G
        // (same convention as nonlocal <β|ψ> in apply.zig)
        .inv_volume = 1.0 / volume,
    };
}

/// Build an `npoints`-sized energy grid [emin, emax] covering all eigenvalues.
fn build_energy_grid(
    alloc: std.mem.Allocator,
    wf_data: scf.WavefunctionData,
    sigma: f64,
    npoints: usize,
    emin_opt: ?f64,
    emax_opt: ?f64,
) ![]f64 {
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
    return energies;
}

/// Allocate and initialize an empty PdosProjection per (atom, atomic_wfc).
fn init_projection_channels(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    npoints: usize,
) ![]PdosProjection {
    var n_proj: usize = 0;
    for (atoms) |atom| {
        const upf = species[atom.species_index].upf;
        n_proj += upf.atomic_wfc.len;
    }
    if (n_proj == 0) return &[_]PdosProjection{};

    const projections = try alloc.alloc(PdosProjection, n_proj);
    errdefer alloc.free(projections);

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
    return projections;
}

/// Build radial tables for every atomic wavefunction of every species.
fn build_wfc_radial_tables(
    alloc: std.mem.Allocator,
    species: []const hamiltonian.SpeciesEntry,
    g_max: f64,
) ![][]nonlocal.RadialTable {
    const n_species = species.len;
    const wfc_tables = try alloc.alloc([]nonlocal.RadialTable, n_species);
    errdefer alloc.free(wfc_tables);

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
    return wfc_tables;
}

/// Compute <φ_{a,l,m}|ψ_n> = Σ_G conj(φ(k+G)) × c_n(G) as (real, imag).
fn project_band_channel(
    coefficients: []const math.Complex,
    gvecs: []const plane_wave.GVector,
    atom: hamiltonian.AtomData,
    l: i32,
    m: i32,
    il_r: f64,
    il_i: f64,
    table: *const nonlocal.RadialTable,
    n_pw: usize,
    band: usize,
) struct { r: f64, i: f64 } {
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
                nonlocal.real_spherical_harmonic(0, 0, 0, 0, 1.0)
            else
                0.0;
        } else {
            ylm = nonlocal.real_spherical_harmonic(l, m, kpg.x, kpg.y, kpg.z);
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
        const c = coefficients[g + band * n_pw];
        // conj(φ) = (phi_r - i×phi_i)
        // conj(φ)×c = (phi_r×c.r + phi_i×c.i) + i×(phi_r×c.i - phi_i×c.r)
        proj_r += phi_r * c.r + phi_i * c.i;
        proj_i += phi_r * c.i - phi_i * c.r;
    }
    return .{ .r = proj_r, .i = proj_i };
}

/// Accumulate PDOS contribution of one (atom, l, m) channel over all bands of
/// a single k-point.
fn accumulate_channel_at_kpoint(
    kp: scf.KpointWavefunction,
    gvecs: []const plane_wave.GVector,
    atom: hamiltonian.AtomData,
    l: i32,
    m: i32,
    table: *const nonlocal.RadialTable,
    energies: []const f64,
    pdos: []f64,
    bp: BroadeningParams,
) void {
    const n_pw = gvecs.len;
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

    // For each band, compute <φ|ψ_n> and broaden onto the PDOS grid.
    for (0..kp.nbands) |n| {
        const proj = project_band_channel(
            kp.coefficients,
            gvecs,
            atom,
            l,
            m,
            il_r,
            il_i,
            table,
            n_pw,
            n,
        );

        // |<φ|ψ>|² for this m (with 1/Ω normalization)
        const proj_sq = (proj.r * proj.r + proj.i * proj.i) * bp.inv_volume;

        // Add to PDOS with Gaussian broadening
        const ek = kp.eigenvalues[n];
        for (0..energies.len) |ie| {
            const diff = energies[ie] - ek;
            if (@abs(diff) <= bp.cutoff) {
                pdos[ie] += kp.weight * bp.spin_factor * proj_sq *
                    gaussian_delta(diff, bp.inv_sigma_sqrt2pi, bp.inv_2sigma2);
            }
        }
    }
}

/// Accumulate PDOS contributions from a single k-point across all
/// (atom, wfc, m) projection channels.
fn accumulate_pdos_at_kpoint(
    alloc: std.mem.Allocator,
    kp: scf.KpointWavefunction,
    species: []const hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    wfc_tables: []const []nonlocal.RadialTable,
    recip: math.Mat3,
    ecut_ry: f64,
    projections: []PdosProjection,
    energies: []const f64,
    bp: BroadeningParams,
) !void {
    var basis = try plane_wave.generate(alloc, recip, ecut_ry, kp.k_cart);
    defer basis.deinit(alloc);

    const gvecs = basis.gvecs;

    var pi: usize = 0;
    for (atoms) |atom| {
        const si = atom.species_index;
        const upf = species[si].upf;

        for (upf.atomic_wfc, 0..) |wfc, wi| {
            const l = wfc.l;
            const table = &wfc_tables[si][wi];

            var m: i32 = -l;
            while (m <= l) : (m += 1) {
                accumulate_channel_at_kpoint(
                    kp,
                    gvecs,
                    atom,
                    l,
                    m,
                    table,
                    energies,
                    projections[pi].pdos,
                    bp,
                );
            }
            pi += 1;
        }
    }
}

fn free_wfc_tables(alloc: std.mem.Allocator, wfc_tables: [][]nonlocal.RadialTable) void {
    for (wfc_tables) |tables| {
        for (tables) |*t| {
            @constCast(t).deinit(alloc);
        }
        if (tables.len > 0) alloc.free(tables);
    }
    alloc.free(wfc_tables);
}

pub fn compute_pdos(
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
    const energies = try build_energy_grid(alloc, wf_data, sigma, npoints, emin_opt, emax_opt);
    errdefer alloc.free(energies);

    const projections = try init_projection_channels(alloc, species, atoms, npoints);
    if (projections.len == 0) {
        return PdosResult{ .energies = energies, .projections = projections };
    }
    errdefer alloc.free(projections);

    // Compute g_max from ecut (generous margin)
    const g_max = std.math.sqrt(wf_data.ecut_ry) * 2.0;
    const wfc_tables = try build_wfc_radial_tables(alloc, species, g_max);
    defer free_wfc_tables(alloc, wfc_tables);

    const bp = broadening_params(sigma, volume, nspin);

    // For each k-point: generate basis, project bands onto every channel.
    for (wf_data.kpoints) |kp| {
        try accumulate_pdos_at_kpoint(
            alloc,
            kp,
            species,
            atoms,
            wfc_tables,
            recip,
            wf_data.ecut_ry,
            projections,
            energies,
            bp,
        );
    }

    return PdosResult{
        .energies = energies,
        .projections = projections,
    };
}

/// Write PDOS to CSV file.
pub fn write_pdos_csv(io: std.Io, dir: std.Io.Dir, result: PdosResult, fermi_level: f64) !void {
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
