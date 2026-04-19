const std = @import("std");
const config = @import("../config/config.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const local_potential = @import("../pseudopotential/local_potential.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const plane_wave = @import("../plane_wave/basis.zig");
const runtime_logging = @import("../runtime/logging.zig");

pub const ScfLoopProfile = struct {
    build_local_r_ns: *u64,
    build_fft_map_ns: *u64,
};

pub fn logProgress(io: std.Io, iter: usize, diff: f64, vresid: f64, band_energy: f64, nonlocal_energy: f64) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(
        .info,
        "scf iter={d} diff={d:.6} vresid={d:.6} band={d:.6} nonlocal={d:.6}\n",
        .{ iter, diff, vresid, band_energy, nonlocal_energy },
    );
}

pub fn logIterStart(io: std.Io, iter: usize) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "scf iter={d} start\n", .{iter});
}

pub fn logKpoint(io: std.Io, index: usize, total: usize) !void {
    if (total == 0) return;
    if (index % 10 != 0 and index + 1 != total) return;
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "scf kpoint {d}/{d}\n", .{ index + 1, total });
}

pub const ScfLog = struct {
    file: std.Io.File,
    io: std.Io,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, out_dir: []const u8) !ScfLog {
        const cwd = std.Io.Dir.cwd();
        try cwd.createDirPath(io, out_dir);
        const log_path = try std.fs.path.join(alloc, &.{ out_dir, "scf.log" });
        defer alloc.free(log_path);
        const file = try cwd.createFile(io, log_path, .{ .truncate = true });
        return .{ .file = file, .io = io };
    }

    pub fn deinit(self: *ScfLog) void {
        self.file.close(self.io);
    }

    pub fn writeHeader(self: *ScfLog) !void {
        var buffer: [4096]u8 = undefined;
        var writer = self.file.writer(self.io, &buffer);
        const out = &writer.interface;
        try out.writeAll("iter,diff,vresid,band_energy,nonlocal_energy\n");
        try out.flush();
    }

    pub fn writeIter(
        self: *ScfLog,
        iter: usize,
        diff: f64,
        vresid: f64,
        band_energy: f64,
        nonlocal_energy: f64,
    ) !void {
        var buffer: [4096]u8 = undefined;
        var writer = self.file.writer(self.io, &buffer);
        const out = &writer.interface;
        try out.print(
            "{d},{d:.10},{d:.10},{d:.10},{d:.10}\n",
            .{ iter, diff, vresid, band_energy, nonlocal_energy },
        );
        try out.flush();
    }

    pub fn writeResult(
        self: *ScfLog,
        converged: bool,
        iterations: usize,
        energy_total: f64,
        band: f64,
        hartree: f64,
        xc: f64,
        ion_ion: f64,
        psp_core: f64,
    ) !void {
        var buffer: [4096]u8 = undefined;
        var writer = self.file.writer(self.io, &buffer);
        const out = &writer.interface;
        try out.print("# converged={s} iterations={d}\n", .{ if (converged) "true" else "false", iterations });
        try out.print(
            "# energy_total={d:.10} band={d:.10} hartree={d:.10} xc={d:.10} ion_ion={d:.10} psp_core={d:.10}\n",
            .{ energy_total, band, hartree, xc, ion_ion, psp_core },
        );
        try out.flush();
    }
};

pub const ScfProfile = struct {
    basis_ns: u64 = 0,
    eig_ns: u64 = 0,
    h_build_ns: u64 = 0,
    vnl_build_ns: u64 = 0,
    s_build_ns: u64 = 0,
    apply_h_ns: u64 = 0,
    apply_local_ns: u64 = 0,
    apply_nonlocal_ns: u64 = 0,
    density_ns: u64 = 0,
    apply_h_calls: usize = 0,
    // Local potential sub-timers
    local_scatter_ns: u64 = 0,
    local_ifft_ns: u64 = 0,
    local_vmul_ns: u64 = 0,
    local_fft_ns: u64 = 0,
    local_gather_ns: u64 = 0,
};

pub fn profileStart(io: std.Io) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.now(io, .awake);
}

pub fn profileAdd(io: std.Io, accum: *u64, start: ?std.Io.Clock.Timestamp) void {
    if (start) |t| {
        const ns: u64 = @intCast(t.untilNow(io).raw.nanoseconds);
        accum.* += ns;
    }
}

pub fn logProfile(io: std.Io, profile: ScfProfile, kpoints: usize) !void {
    const logger = runtime_logging.stderr(io, .info);
    const to_ms = 1.0 / @as(f64, @floatFromInt(std.time.ns_per_ms));
    try logger.print(
        .info,
        "profile kpoints={d} basis_ms={d:.3} eig_ms={d:.3} h_ms={d:.3} vnl_ms={d:.3} s_ms={d:.3} apply_ms={d:.3} density_ms={d:.3} local_ms={d:.3} nonlocal_ms={d:.3} apply_calls={d} scatter_ms={d:.3} ifft_ms={d:.3} vmul_ms={d:.3} fft_ms={d:.3} gather_ms={d:.3}\n",
        .{
            kpoints,
            @as(f64, @floatFromInt(profile.basis_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.eig_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.h_build_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.vnl_build_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.s_build_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.apply_h_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.density_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.apply_local_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.apply_nonlocal_ns)) * to_ms,
            profile.apply_h_calls,
            @as(f64, @floatFromInt(profile.local_scatter_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.local_ifft_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.local_vmul_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.local_fft_ns)) * to_ms,
            @as(f64, @floatFromInt(profile.local_gather_ns)) * to_ms,
        },
    );
}

pub fn logLoopProfile(
    io: std.Io,
    compute_density_ns: u64,
    build_potential_ns: u64,
    residual_ns: u64,
    mixing_ns: u64,
    build_local_r_ns: u64,
    build_fft_map_ns: u64,
) !void {
    const logger = runtime_logging.stderr(io, .info);
    const to_ms = 1.0 / @as(f64, @floatFromInt(std.time.ns_per_ms));
    try logger.print(
        .info,
        "scf_loop_profile compute_density_ms={d:.3} build_potential_ms={d:.3} residual_ms={d:.3} mixing_ms={d:.3} build_local_r_ms={d:.3} build_fft_map_ms={d:.3}\n",
        .{
            @as(f64, @floatFromInt(compute_density_ns)) * to_ms,
            @as(f64, @floatFromInt(build_potential_ns)) * to_ms,
            @as(f64, @floatFromInt(residual_ns)) * to_ms,
            @as(f64, @floatFromInt(mixing_ns)) * to_ms,
            @as(f64, @floatFromInt(build_local_r_ns)) * to_ms,
            @as(f64, @floatFromInt(build_fft_map_ns)) * to_ms,
        },
    );
}

pub fn logEnergySummary(io: std.Io, electron_count: f64, ionic_g0: math.Complex, pot_g0: math.Complex, energy_terms: anytype) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "scf: electron_count {d:.6}\n", .{electron_count});
    try logger.print(.info, "scf: ionic_g0 {d:.6} {d:.6}\n", .{ ionic_g0.r, ionic_g0.i });
    try logger.print(.info, "scf: hartree_xc_g0 {d:.6} {d:.6}\n", .{ pot_g0.r, pot_g0.i });
    try logger.print(
        .info,
        "scf: E_band={d:.8} E_H={d:.8} E_xc={d:.8} E_ion={d:.8}\n",
        .{ energy_terms.band, energy_terms.hartree, energy_terms.xc, energy_terms.ion_ion },
    );
    try logger.print(
        .info,
        "scf: E_psp={d:.8} E_dc={d:.8} E_local={d:.8} E_nl={d:.8}\n",
        .{
            energy_terms.psp_core,
            energy_terms.double_counting,
            energy_terms.local_pseudo,
            energy_terms.nonlocal_pseudo,
        },
    );
    try logger.print(
        .info,
        "scf: E_paw_onsite={d:.8} E_paw_dxc={d:.8} E_total={d:.8}\n",
        .{ energy_terms.paw_onsite, energy_terms.paw_dxc_rhoij, energy_terms.total },
    );
}

pub fn logSpinInit(io: std.Io, total_electrons: f64, magnetization: f64) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "spin-scf: nspin=2, nelec={d:.1}, m_init={d:.2}\n", .{ total_electrons, magnetization });
}

pub fn logSpinMagnetization(io: std.Io, magnetization: f64) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "spin-scf: magnetization = {d:.6} μ_B\n", .{magnetization});
}

pub fn logSpinEnergySummary(io: std.Io, energy_terms: anytype, include_paw: bool) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "spin-scf: total_energy = {d:.10} Ry\n", .{energy_terms.total});
    try logger.print(
        .info,
        "spin-scf: E_band={d:.8} E_H={d:.8} E_xc={d:.8} E_ion={d:.8}\n",
        .{ energy_terms.band, energy_terms.hartree, energy_terms.xc, energy_terms.ion_ion },
    );
    try logger.print(
        .info,
        "spin-scf: E_psp={d:.8} E_dc={d:.8} E_local={d:.8} E_nl={d:.8}\n",
        .{
            energy_terms.psp_core,
            energy_terms.double_counting,
            energy_terms.local_pseudo,
            energy_terms.nonlocal_pseudo,
        },
    );
    if (include_paw) {
        try logger.print(
            .info,
            "spin-scf: E_paw_onsite={d:.8} E_paw_dxc={d:.8}\n",
            .{ energy_terms.paw_onsite, energy_terms.paw_dxc_rhoij },
        );
    }
}

pub fn logLocalPotentialMean(io: std.Io, prefix: []const u8, mean_local: f64, g0_label: []const u8, g0_value: f64) !void {
    const logger = runtime_logging.stderr(io, .debug);
    try logger.print(
        .debug,
        "{s}: local_r mean={d:.6} {s}={d:.6}\n",
        .{ prefix, mean_local, g0_label, g0_value },
    );
}

pub fn logBandLocalPotentialMean(io: std.Io, mean_local: f64, ionic_g0: f64, extra_g0: f64) !void {
    const logger = runtime_logging.stderr(io, .debug);
    try logger.print(
        .debug,
        "band: local_r mean={d:.6} ionic_g0={d:.6} extra_g0={d:.6}\n",
        .{ mean_local, ionic_g0, extra_g0 },
    );
}

pub fn logEigenvalues(io: std.Io, prefix: []const u8, label: []const u8, values: []const f64, count: usize) !void {
    const limit = @min(count, 8);
    const logger = runtime_logging.stderr(io, .debug);
    try logger.print(.debug, "{s}: eig {s} nbands={d}", .{ prefix, label, count });
    var i: usize = 0;
    while (i < limit) : (i += 1) {
        try logger.print(.debug, " {d:.6}", .{values[i]});
    }
    if (count > limit) {
        try logger.writeAll(.debug, " ...");
    }
    try logger.writeAll(.debug, "\n");
}

pub fn logFermiDiag(
    io: std.Io,
    min_energy: f64,
    max_energy: f64,
    mu: f64,
    nelec: f64,
    min_nbands: usize,
    max_nbands: usize,
    smearing_name: []const u8,
    sigma: f64,
) !void {
    const outside = mu < min_energy or mu > max_energy;
    const logger = runtime_logging.stderr(io, .debug);
    try logger.print(
        .debug,
        "scf: fermi diag min={d:.6} max={d:.6} mu={d:.6} outside={s} nelec={d:.6} nbands={d}-{d} smear={s} sigma={d:.6}\n",
        .{
            min_energy,
            max_energy,
            mu,
            if (outside) "true" else "false",
            nelec,
            min_nbands,
            max_nbands,
            smearing_name,
            sigma,
        },
    );
}

pub fn logIterativeGridTooSmall(io: std.Io, nx: usize, ny: usize, nz: usize, sx: usize, sy: usize, sz: usize) !void {
    const logger = runtime_logging.stderr(io, .warn);
    try logger.print(
        .warn,
        "scf: iterative grid too small (need >= {d},{d},{d}, suggest {d},{d},{d})\n",
        .{ nx, ny, nz, sx, sy, sz },
    );
}

pub fn logIterativeSolverDisabled(io: std.Io, reason: []const u8) !void {
    const logger = runtime_logging.stderr(io, .info);
    try logger.print(.info, "scf: iterative solver disabled ({s})\n", .{reason});
}

pub fn mergeProfile(dest: *ScfProfile, src: ScfProfile) void {
    dest.basis_ns += src.basis_ns;
    dest.eig_ns += src.eig_ns;
    dest.h_build_ns += src.h_build_ns;
    dest.vnl_build_ns += src.vnl_build_ns;
    dest.s_build_ns += src.s_build_ns;
    dest.apply_h_ns += src.apply_h_ns;
    dest.apply_local_ns += src.apply_local_ns;
    dest.apply_nonlocal_ns += src.apply_nonlocal_ns;
    dest.density_ns += src.density_ns;
    dest.apply_h_calls += src.apply_h_calls;
    dest.local_scatter_ns += src.local_scatter_ns;
    dest.local_ifft_ns += src.local_ifft_ns;
    dest.local_vmul_ns += src.local_vmul_ns;
    dest.local_fft_ns += src.local_fft_ns;
    dest.local_gather_ns += src.local_gather_ns;
}

pub fn logNonlocalDiagnostics(
    alloc: std.mem.Allocator,
    io: std.Io,
    gvecs: []plane_wave.GVector,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    inv_volume: f64,
) !void {
    const g_count = gvecs.len;
    if (g_count == 0) return;

    var g_min = std.math.inf(f64);
    var g_max: f64 = 0.0;
    var g_min_idx: usize = 0;
    var g_max_idx: usize = 0;
    for (gvecs, 0..) |g, i| {
        const gmag = math.Vec3.norm(g.kpg);
        if (gmag < g_min) {
            g_min = gmag;
            g_min_idx = i;
        }
        if (gmag > g_max) {
            g_max = gmag;
            g_max_idx = i;
        }
    }

    const logger = runtime_logging.stderr(io, .debug);
    try logger.print(
        .debug,
        "scf: nonlocal diag g_count={d} g_min={d:.6} g_max={d:.6}\n",
        .{ g_count, g_min, g_max },
    );

    var s: usize = 0;
    while (s < species.len) : (s += 1) {
        const entry = &species[s];
        const upf = entry.upf.*;
        const beta_count = upf.beta.len;
        const coeffs = upf.dij;
        if (beta_count == 0 or coeffs.len == 0) continue;
        if (coeffs.len != beta_count * beta_count) return error.InvalidPseudopotential;

        var atom_count: usize = 0;
        for (atoms) |atom| {
            if (atom.species_index == s) atom_count += 1;
        }

        const radial = try alloc.alloc(f64, beta_count * g_count);
        defer alloc.free(radial);
        const l_list = try alloc.alloc(i32, beta_count);
        defer alloc.free(l_list);
        const angular = try alloc.alloc(f64, beta_count);
        defer alloc.free(angular);

        var b: usize = 0;
        while (b < beta_count) : (b += 1) {
            const l_val = upf.beta[b].l orelse 0;
            l_list[b] = l_val;
            angular[b] = nonlocal.angularFactor(l_val, 1.0);
            var g: usize = 0;
            while (g < g_count) : (g += 1) {
                const gmag = math.Vec3.norm(gvecs[g].kpg);
                radial[b * g_count + g] = nonlocal.radialProjector(
                    upf.beta[b].values,
                    upf.r,
                    upf.rab,
                    l_val,
                    gmag,
                );
            }
        }

        var min_val = std.math.inf(f64);
        var max_val = -std.math.inf(f64);
        var sum_val: f64 = 0.0;
        var sum_abs: f64 = 0.0;
        var neg_count: usize = 0;

        var g: usize = 0;
        while (g < g_count) : (g += 1) {
            var val: f64 = 0.0;
            var i: usize = 0;
            while (i < beta_count) : (i += 1) {
                const li = l_list[i];
                const ang = angular[i];
                var j: usize = 0;
                while (j < beta_count) : (j += 1) {
                    if (l_list[j] != li) continue;
                    const coeff = coeffs[i * beta_count + j];
                    if (coeff == 0.0) continue;
                    const ri = radial[i * g_count + g];
                    const rj = radial[j * g_count + g];
                    val += coeff * ang * ri * rj;
                }
            }

            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            if (val < 0.0) neg_count += 1;
            sum_val += val;
            sum_abs += @abs(val);
        }

        const count_f = @as(f64, @floatFromInt(g_count));
        const mean_val = sum_val / count_f;
        const mean_abs = sum_abs / count_f;
        const scale = inv_volume * @as(f64, @floatFromInt(atom_count));
        try logger.print(
            .debug,
            "scf: nonlocal diag species={s} atoms={d} min={d:.6} max={d:.6} mean={d:.6} mean_abs={d:.6} neg={d}/{d}\n",
            .{ entry.symbol, atom_count, min_val, max_val, mean_val, mean_abs, neg_count, g_count },
        );
        try logger.print(
            .debug,
            "scf: nonlocal diag scaled species={s} min={d:.6} max={d:.6} mean={d:.6}\n",
            .{ entry.symbol, min_val * scale, max_val * scale, mean_val * scale },
        );
        try logger.print(
            .debug,
            "scf: nonlocal radial sample species={s} g_min={d:.6} g_max={d:.6}\n",
            .{ entry.symbol, g_min, g_max },
        );
        b = 0;
        while (b < beta_count) : (b += 1) {
            const rmin = radial[b * g_count + g_min_idx];
            const rmax = radial[b * g_count + g_max_idx];
            try logger.print(
                .debug,
                "  beta[{d}] l={d} rmin={d:.6} rmax={d:.6}\n",
                .{ b, l_list[b], rmin, rmax },
            );
        }
    }
}

pub fn logLocalDiagnostics(
    io: std.Io,
    gvecs: []plane_wave.GVector,
    species: []hamiltonian.SpeciesEntry,
    atoms: []const hamiltonian.AtomData,
    local_cfg: local_potential.LocalPotentialConfig,
) !void {
    const g_count = gvecs.len;
    if (g_count == 0) return;

    var g_min = std.math.inf(f64);
    var g_max: f64 = 0.0;
    var g_min_nonzero = std.math.inf(f64);
    for (gvecs) |g| {
        const gmag = math.Vec3.norm(g.kpg);
        if (gmag < g_min) g_min = gmag;
        if (gmag > g_max) g_max = gmag;
        if (gmag > 1e-12 and gmag < g_min_nonzero) {
            g_min_nonzero = gmag;
        }
    }
    if (g_min_nonzero == std.math.inf(f64)) g_min_nonzero = 0.0;

    const logger = runtime_logging.stderr(io, .debug);
    try logger.print(
        .debug,
        "scf: local diag g_count={d} g_min={d:.6} g_min_nz={d:.6} g_max={d:.6}\n",
        .{ g_count, g_min, g_min_nonzero, g_max },
    );

    var s: usize = 0;
    while (s < species.len) : (s += 1) {
        const entry = &species[s];
        var atom_count: usize = 0;
        for (atoms) |atom| {
            if (atom.species_index == s) atom_count += 1;
        }
        if (atom_count == 0) continue;

        var min_val = std.math.inf(f64);
        var max_val = -std.math.inf(f64);
        var sum_val: f64 = 0.0;
        var sum_abs: f64 = 0.0;
        var neg_count: usize = 0;
        var count: usize = 0;
        for (gvecs) |g| {
            const gmag = math.Vec3.norm(g.kpg);
            if (gmag < 1e-12) continue;
            const vq = hamiltonian.localFormFactor(entry, gmag, local_cfg);
            if (vq < min_val) min_val = vq;
            if (vq > max_val) max_val = vq;
            if (vq < 0.0) neg_count += 1;
            sum_val += vq;
            sum_abs += @abs(vq);
            count += 1;
        }

        const count_f = @as(f64, @floatFromInt(count));
        const mean_val = if (count > 0) sum_val / count_f else 0.0;
        const mean_abs = if (count > 0) sum_abs / count_f else 0.0;
        if (count == 0) {
            min_val = 0.0;
            max_val = 0.0;
        }

        const vq0 = hamiltonian.localFormFactor(entry, 0.0, local_cfg);
        const vq_min = hamiltonian.localFormFactor(entry, g_min_nonzero, local_cfg);
        const vq_max = hamiltonian.localFormFactor(entry, g_max, local_cfg);
        try logger.print(
            .debug,
            "scf: local diag species={s} mode={s} alpha={d:.6} atoms={d} min={d:.6} max={d:.6} mean={d:.6} mean_abs={d:.6} neg={d}/{d}\n",
            .{
                entry.symbol,
                config.localPotentialModeName(local_cfg.mode),
                local_cfg.alpha,
                atom_count,
                min_val,
                max_val,
                mean_val,
                mean_abs,
                neg_count,
                count,
            },
        );
        try logger.print(
            .debug,
            "scf: local sample species={s} q0={d:.6} vq0={d:.6} q_min={d:.6} vq_min={d:.6} q_max={d:.6} vq_max={d:.6}\n",
            .{ entry.symbol, 0.0, vq0, g_min_nonzero, vq_min, g_max, vq_max },
        );
    }
}
