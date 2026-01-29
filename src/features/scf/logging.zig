const std = @import("std");
const config = @import("../config/config.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const math = @import("../math/math.zig");
const nonlocal = @import("../pseudopotential/nonlocal.zig");
const plane_wave = @import("../plane_wave/basis.zig");

pub fn logProgress(iter: usize, diff: f64, vresid: f64, band_energy: f64, nonlocal_energy: f64) !void {
    var buffer: [512]u8 = undefined;
    var writer = std.fs.File.stderr().writer(&buffer);
    const out = &writer.interface;
    try out.print(
        "scf iter={d} diff={d:.6} vresid={d:.6} band={d:.6} nonlocal={d:.6}\n",
        .{ iter, diff, vresid, band_energy, nonlocal_energy },
    );
    try out.flush();
}

pub fn logIterStart(iter: usize) !void {
    var buffer: [128]u8 = undefined;
    var writer = std.fs.File.stderr().writer(&buffer);
    const out = &writer.interface;
    try out.print("scf iter={d} start\n", .{iter});
    try out.flush();
}

pub fn logKpoint(index: usize, total: usize) !void {
    if (total == 0) return;
    if (index % 10 != 0 and index + 1 != total) return;
    var buffer: [128]u8 = undefined;
    var writer = std.fs.File.stderr().writer(&buffer);
    const out = &writer.interface;
    try out.print("scf kpoint {d}/{d}\n", .{ index + 1, total });
    try out.flush();
}

pub const ScfLog = struct {
    file: std.fs.File,

    pub fn init(alloc: std.mem.Allocator, out_dir: []const u8) !ScfLog {
        try std.fs.cwd().makePath(out_dir);
        const log_path = try std.fs.path.join(alloc, &.{ out_dir, "scf.log" });
        defer alloc.free(log_path);
        const file = try std.fs.cwd().createFile(log_path, .{ .truncate = true });
        return .{ .file = file };
    }

    pub fn deinit(self: *ScfLog) void {
        self.file.close();
    }

    pub fn writeHeader(self: *ScfLog) !void {
        var buffer: [4096]u8 = undefined;
        var writer = self.file.writer(&buffer);
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
        var writer = self.file.writer(&buffer);
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
        var writer = self.file.writer(&buffer);
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
};

pub fn profileStart() ?std.time.Instant {
    return std.time.Instant.now() catch null;
}

pub fn profileAdd(accum: *u64, start: ?std.time.Instant) void {
    if (start) |t0| {
        const t1 = std.time.Instant.now() catch return;
        accum.* += t1.since(t0);
    }
}

pub fn logProfile(profile: ScfProfile, kpoints: usize) !void {
    var buffer: [512]u8 = undefined;
    var writer = std.fs.File.stderr().writer(&buffer);
    const out = &writer.interface;
    const to_ms = 1.0 / @as(f64, @floatFromInt(std.time.ns_per_ms));
    try out.print(
        "profile kpoints={d} basis_ms={d:.3} eig_ms={d:.3} h_ms={d:.3} vnl_ms={d:.3} s_ms={d:.3} apply_ms={d:.3} density_ms={d:.3} local_ms={d:.3} nonlocal_ms={d:.3} apply_calls={d}\n",
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
        },
    );
    try out.flush();
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
}

pub fn logNonlocalDiagnostics(
    alloc: std.mem.Allocator,
    gvecs: []plane_wave.GVector,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
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

    var buffer: [512]u8 = undefined;
    var writer = std.fs.File.stderr().writer(&buffer);
    const out = &writer.interface;
    try out.print(
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
        try out.print(
            "scf: nonlocal diag species={s} atoms={d} min={d:.6} max={d:.6} mean={d:.6} mean_abs={d:.6} neg={d}/{d}\n",
            .{ entry.symbol, atom_count, min_val, max_val, mean_val, mean_abs, neg_count, g_count },
        );
        try out.print(
            "scf: nonlocal diag scaled species={s} min={d:.6} max={d:.6} mean={d:.6}\n",
            .{ entry.symbol, min_val * scale, max_val * scale, mean_val * scale },
        );
        try out.print(
            "scf: nonlocal radial sample species={s} g_min={d:.6} g_max={d:.6}\n",
            .{ entry.symbol, g_min, g_max },
        );
        b = 0;
        while (b < beta_count) : (b += 1) {
            const rmin = radial[b * g_count + g_min_idx];
            const rmax = radial[b * g_count + g_max_idx];
            try out.print(
                "  beta[{d}] l={d} rmin={d:.6} rmax={d:.6}\n",
                .{ b, l_list[b], rmin, rmax },
            );
        }
        try out.flush();
    }
}

pub fn logLocalDiagnostics(
    gvecs: []plane_wave.GVector,
    species: []hamiltonian.SpeciesEntry,
    atoms: []hamiltonian.AtomData,
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

    var buffer: [512]u8 = undefined;
    var writer = std.fs.File.stderr().writer(&buffer);
    const out = &writer.interface;
    try out.print(
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
            const vq = hamiltonian.localFormFactor(entry, gmag);
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

        const vq0 = hamiltonian.localFormFactor(entry, 0.0);
        const vq_min = hamiltonian.localFormFactor(entry, g_min_nonzero);
        const vq_max = hamiltonian.localFormFactor(entry, g_max);
        try out.print(
            "scf: local diag species={s} mode={s} alpha={d:.6} atoms={d} min={d:.6} max={d:.6} mean={d:.6} mean_abs={d:.6} neg={d}/{d}\n",
            .{
                entry.symbol,
                config.localPotentialModeName(entry.local_mode),
                entry.local_alpha,
                atom_count,
                min_val,
                max_val,
                mean_val,
                mean_abs,
                neg_count,
                count,
            },
        );
        try out.print(
            "scf: local sample species={s} q0={d:.6} vq0={d:.6} q_min={d:.6} vq_min={d:.6} q_max={d:.6} vq_max={d:.6}\n",
            .{ entry.symbol, 0.0, vq0, g_min_nonzero, vq_min, g_max, vq_max },
        );
    }
    try out.flush();
}
