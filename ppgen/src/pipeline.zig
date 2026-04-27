//! End-to-end pseudopotential generation pipeline.
//!
//! atomic SCF → TM pseudization → KB projectors → UPF output → parser round-trip

const std = @import("std");
const dft_zig = @import("dft_zig");
const xc_mod = dft_zig.xc;
const pseudo_parser = dft_zig.pseudopotential;

const RadialGrid = @import("radial_grid.zig").RadialGrid;
const atomic_solver = @import("atomic_solver.zig");
const schrodinger = @import("schrodinger.zig");
const tm_generator = @import("tm_generator.zig");
const kb_projector = @import("kb_projector.zig");
const upf_writer = @import("upf_writer.zig");

pub const OrbitalDef = struct {
    n: u32,
    l: u32,
    occupation: f64,
};

pub const ChannelConfig = struct {
    n: u32,
    l: u32,
    occupation: f64,
    rc: f64, // cutoff radius (Bohr)
    /// Fixed reference energy in Ry for unoccupied scattering channels.
    /// Null means use the corresponding all-electron bound orbital.
    reference_energy: ?f64 = null,
};

pub const GeneratorConfig = struct {
    z: f64,
    element: []const u8,
    xc: xc_mod.Functional,
    /// All orbitals for the all-electron calculation (core + valence)
    all_orbitals: []const OrbitalDef,
    /// Valence channels to pseudize (subset of all_orbitals)
    valence_channels: []const ChannelConfig,
    l_local: u32, // which l to use as local potential
    local_n: ?u32 = null,
    local_smooth_radius: ?f64 = null,
    nlcc: ?NlccConfig = null,
};

pub const NlccConfig = struct {
    charge: f64,
    radius: f64,
};

pub const LogDerivativeConfig = struct {
    r_match: f64,
    energies: []const f64,
};

pub const LogDerivativeSample = struct {
    channel_n: u32,
    l: u32,
    energy: f64,
    ae: f64,
    pseudo: f64,
    delta: f64,
    status: LogDerivativeStatus,
};

pub const LogDerivativeStatus = enum {
    ok,
    ae_error,
    pseudo_error,
};

pub const LogDerivativeReport = struct {
    samples: []LogDerivativeSample,
    max_abs_delta: f64,
    rms_delta: f64,
    valid_count: usize,
    invalid_count: usize,
    pole_mismatch_count: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LogDerivativeReport) void {
        self.allocator.free(self.samples);
    }
};

const NonLocalData = struct {
    betas: std.ArrayListUnmanaged(upf_writer.BetaData) = .empty,
    dij: []f64,
    allocator: std.mem.Allocator,

    fn deinit(self: *NonLocalData) void {
        for (self.betas.items) |b| self.allocator.free(@constCast(b.values));
        self.betas.deinit(self.allocator);
        self.allocator.free(self.dij);
    }
};

const ProjectorWork = struct {
    l: u32,
    u: []const f64,
    beta: []const f64,
};

/// Run the full pipeline and write UPF to the given writer.
pub fn generate_pseudopotential(
    allocator: std.mem.Allocator,
    config: GeneratorConfig,
    writer: anytype,
) !void {
    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    var ae_result = try solve_all_electron_atom(allocator, &grid, config);
    defer ae_result.deinit();

    const pw_list = try generate_pseudo_wavefunctions(allocator, &grid, config, &ae_result);
    defer deinit_pseudo_wavefunctions(allocator, pw_list);

    const rho_val = try build_valence_density(allocator, &grid, config.valence_channels, pw_list);
    defer allocator.free(rho_val);

    const nlcc = try build_optional_nlcc(allocator, &grid, config, &ae_result);
    defer if (nlcc) |values| allocator.free(values);

    const v_h = try build_hartree_potential(allocator, &grid, rho_val);
    defer allocator.free(v_h);

    const v_xc = try build_xc_potential(allocator, &grid, config.xc, rho_val, nlcc);
    defer allocator.free(v_xc);

    const l_local_idx = try find_local_channel_index(
        config.valence_channels,
        config.local_n,
        config.l_local,
    );
    const v_local_screened = try build_local_screened_potential(
        allocator,
        &grid,
        pw_list[l_local_idx].v_ps,
        config.local_smooth_radius,
    );
    defer allocator.free(v_local_screened);

    const v_local_ion = try kb_projector.unscreen(allocator, &grid, v_local_screened, v_h, v_xc);
    defer allocator.free(v_local_ion);

    var nonlocal = try build_nonlocal_data(
        allocator,
        &grid,
        config.valence_channels,
        l_local_idx,
        pw_list,
        v_local_screened,
    );
    defer nonlocal.deinit();

    const rho_atom = try build_atomic_rho(allocator, &grid, rho_val);
    defer allocator.free(rho_atom);

    try write_pseudopotential_data(
        writer,
        config,
        &grid,
        v_local_ion,
        nonlocal.betas.items,
        nonlocal.dij,
        rho_atom,
        nlcc,
    );
}

fn solve_all_electron_atom(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    config: GeneratorConfig,
) !atomic_solver.AtomResult {
    const orb_configs = try build_orbital_configs(allocator, config.all_orbitals);
    defer allocator.free(orb_configs);

    return try atomic_solver.solve(allocator, grid, .{
        .z = config.z,
        .orbitals = orb_configs,
        .xc = config.xc,
    }, 300, 0.3, 1e-10);
}

/// Compare all-electron and pseudized atomic scattering logarithmic derivatives.
pub fn compute_log_derivative_report(
    allocator: std.mem.Allocator,
    config: GeneratorConfig,
    diag: LogDerivativeConfig,
) !LogDerivativeReport {
    if (!std.math.isFinite(diag.r_match) or diag.r_match <= 0.0) {
        return error.InvalidGeneratorConfig;
    }
    if (diag.energies.len == 0) return error.InvalidGeneratorConfig;

    var grid = try RadialGrid.init(allocator, 4000, 1e-8, 80.0);
    defer grid.deinit();

    validate_log_derivative_radius(&grid, config.valence_channels, diag.r_match) catch |err| {
        return err;
    };

    var ae_result = try solve_all_electron_atom(allocator, &grid, config);
    defer ae_result.deinit();

    const pw_list = try generate_pseudo_wavefunctions(allocator, &grid, config, &ae_result);
    defer deinit_pseudo_wavefunctions(allocator, pw_list);

    return build_log_derivative_report(
        allocator,
        &grid,
        config.valence_channels,
        pw_list,
        ae_result.v_eff,
        diag,
    );
}

fn validate_log_derivative_radius(
    grid: *const RadialGrid,
    valence_channels: []const ChannelConfig,
    r_match: f64,
) !void {
    if (r_match <= grid.r[2] or r_match >= grid.r[grid.n - 2]) return error.InvalidGeneratorConfig;
    for (valence_channels) |ch| {
        if (!std.math.isFinite(ch.rc) or r_match <= ch.rc) return error.InvalidGeneratorConfig;
    }
}

fn build_log_derivative_report(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    valence_channels: []const ChannelConfig,
    pw_list: []const tm_generator.PseudoWavefunction,
    ae_v_eff: []const f64,
    diag: LogDerivativeConfig,
) !LogDerivativeReport {
    std.debug.assert(valence_channels.len == pw_list.len);
    const samples = try allocator.alloc(
        LogDerivativeSample,
        valence_channels.len * diag.energies.len,
    );
    errdefer allocator.free(samples);

    var stats = LogDerivativeStats{};
    var count: usize = 0;
    for (valence_channels, pw_list) |ch, pw| {
        for (diag.energies) |energy| {
            samples[count] = try compute_log_derivative_sample(
                allocator,
                grid,
                ch,
                pw.v_ps,
                ae_v_eff,
                energy,
                diag.r_match,
            );
            stats.add(samples[count]);
            count += 1;
        }
    }

    if (stats.valid_count == 0) return error.InvalidGeneratorConfig;
    return .{
        .samples = samples,
        .max_abs_delta = stats.max_abs_delta,
        .rms_delta = stats.rms(),
        .valid_count = stats.valid_count,
        .invalid_count = stats.invalid_count,
        .pole_mismatch_count = count_log_derivative_pole_mismatches(samples, diag.energies.len),
        .allocator = allocator,
    };
}

const LogDerivativeStats = struct {
    max_abs_delta: f64 = 0.0,
    sum_sq_delta: f64 = 0.0,
    valid_count: usize = 0,
    invalid_count: usize = 0,

    fn add(self: *LogDerivativeStats, sample: LogDerivativeSample) void {
        if (sample.status != .ok) {
            self.invalid_count += 1;
            return;
        }
        const delta = sample.delta;
        self.max_abs_delta = @max(self.max_abs_delta, @abs(delta));
        self.sum_sq_delta += delta * delta;
        self.valid_count += 1;
    }

    fn rms(self: LogDerivativeStats) f64 {
        std.debug.assert(self.valid_count > 0);
        return @sqrt(self.sum_sq_delta / @as(f64, @floatFromInt(self.valid_count)));
    }
};

fn count_log_derivative_pole_mismatches(
    samples: []const LogDerivativeSample,
    energy_count: usize,
) usize {
    std.debug.assert(energy_count > 1);
    std.debug.assert(samples.len % energy_count == 0);

    var mismatches: usize = 0;
    var start: usize = 0;
    while (start < samples.len) : (start += energy_count) {
        const block = samples[start .. start + energy_count];
        const ae_poles = count_log_derivative_poles(block, .ae);
        const ps_poles = count_log_derivative_poles(block, .pseudo);
        if (ae_poles != ps_poles) mismatches += 1;
    }
    return mismatches;
}

const LogDerivativeSide = enum {
    ae,
    pseudo,
};

fn count_log_derivative_poles(samples: []const LogDerivativeSample, side: LogDerivativeSide) usize {
    const pole_abs_threshold = 5.0;
    var count: usize = 0;
    for (1..samples.len) |i| {
        const left = log_derivative_value(samples[i - 1], side) orelse continue;
        const right = log_derivative_value(samples[i], side) orelse continue;
        if (left * right < 0.0 and @max(@abs(left), @abs(right)) >= pole_abs_threshold) {
            count += 1;
        }
    }
    return count;
}

fn log_derivative_value(sample: LogDerivativeSample, side: LogDerivativeSide) ?f64 {
    if (sample.status != .ok) return null;
    return switch (side) {
        .ae => sample.ae,
        .pseudo => sample.pseudo,
    };
}

fn compute_log_derivative_sample(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    channel: ChannelConfig,
    v_ps: []const f64,
    ae_v_eff: []const f64,
    energy: f64,
    r_match: f64,
) !LogDerivativeSample {
    if (!std.math.isFinite(energy)) return error.InvalidGeneratorConfig;

    const ae = schrodinger.fixed_energy_log_derivative(
        allocator,
        grid,
        ae_v_eff,
        .{ .n = channel.n, .l = channel.l },
        energy,
        r_match,
    ) catch return failed_log_derivative_sample(channel, energy, .ae_error);

    const pseudo = schrodinger.fixed_energy_log_derivative(
        allocator,
        grid,
        v_ps,
        .{ .n = channel.n, .l = channel.l },
        energy,
        r_match,
    ) catch return failed_log_derivative_sample(channel, energy, .pseudo_error);

    return .{
        .channel_n = channel.n,
        .l = channel.l,
        .energy = energy,
        .ae = ae,
        .pseudo = pseudo,
        .delta = pseudo - ae,
        .status = .ok,
    };
}

fn failed_log_derivative_sample(
    channel: ChannelConfig,
    energy: f64,
    status: LogDerivativeStatus,
) LogDerivativeSample {
    return .{
        .channel_n = channel.n,
        .l = channel.l,
        .energy = energy,
        .ae = 0.0,
        .pseudo = 0.0,
        .delta = 0.0,
        .status = status,
    };
}

fn write_pseudopotential_data(
    writer: anytype,
    config: GeneratorConfig,
    grid: *const RadialGrid,
    v_local_ion: []const f64,
    betas: []const upf_writer.BetaData,
    dij: []const f64,
    rho_atom: []const f64,
    nlcc: ?[]const f64,
) !void {
    try upf_writer.write(.{
        .element = config.element,
        .z = config.z,
        .z_valence = total_valence_charge(config.valence_channels),
        .xc_functional = xc_name(config.xc),
        .r = grid.r,
        .rab = grid.rab,
        .v_local = v_local_ion,
        .betas = betas,
        .dij = dij,
        .rho_atom = rho_atom,
        .nlcc = nlcc,
        .l_max = max_angular_momentum(config.valence_channels),
        .mesh_size = grid.n,
    }, writer);
}

fn xc_name(xc: xc_mod.Functional) []const u8 {
    return switch (xc) {
        .lda_pz => "PZ",
        .pbe => "PBE",
    };
}

fn build_orbital_configs(
    allocator: std.mem.Allocator,
    all_orbitals: []const OrbitalDef,
) ![]atomic_solver.OrbitalConfig {
    const orb_configs = try allocator.alloc(atomic_solver.OrbitalConfig, all_orbitals.len);
    for (all_orbitals, 0..) |orb, i| {
        orb_configs[i] = .{ .n = orb.n, .l = orb.l, .occupation = orb.occupation };
    }
    return orb_configs;
}

fn find_orbital_index(
    config: GeneratorConfig,
    channel: ChannelConfig,
) !usize {
    for (config.all_orbitals, 0..) |orb, ai| {
        if (orb.n == channel.n and orb.l == channel.l) {
            return ai;
        }
    }
    return error.InvalidGeneratorConfig;
}

fn generate_pseudo_wavefunctions(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    config: GeneratorConfig,
    ae_result: *const atomic_solver.AtomResult,
) ![]tm_generator.PseudoWavefunction {
    const pw_list =
        try allocator.alloc(tm_generator.PseudoWavefunction, config.valence_channels.len);
    errdefer allocator.free(pw_list);

    var initialized: usize = 0;
    errdefer {
        for (pw_list[0..initialized]) |*pw| pw.deinit();
    }

    for (config.valence_channels, 0..) |ch, i| {
        var owned_ref_u: ?[]f64 = null;
        defer if (owned_ref_u) |u| allocator.free(u);

        const ae_u: []const f64 = if (ch.reference_energy) |ref_energy| blk: {
            var ref_sol = try schrodinger.solve_fixed_energy_outward(
                allocator,
                grid,
                ae_result.v_eff,
                .{ .n = ch.n, .l = ch.l },
                ref_energy,
            );
            defer ref_sol.deinit();

            const u_copy = try allocator.alloc(f64, grid.n);
            errdefer allocator.free(u_copy);

            @memcpy(u_copy, ref_sol.u);
            owned_ref_u = u_copy;
            break :blk u_copy;
        } else blk: {
            const ai = try find_orbital_index(config, ch);
            break :blk ae_result.wavefunctions[ai];
        };
        const energy = if (ch.reference_energy) |ref_energy| blk: {
            break :blk ref_energy;
        } else blk: {
            const ai = try find_orbital_index(config, ch);
            break :blk ae_result.eigenvalues[ai];
        };

        pw_list[i] = try tm_generator.generate(
            allocator,
            grid,
            ae_u,
            ae_result.v_eff,
            energy,
            ch.l,
            ch.rc,
        );
        initialized += 1;
    }
    return pw_list;
}

fn deinit_pseudo_wavefunctions(
    allocator: std.mem.Allocator,
    pw_list: []tm_generator.PseudoWavefunction,
) void {
    for (pw_list) |*pw| pw.deinit();
    allocator.free(pw_list);
}

fn build_valence_density(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    valence_channels: []const ChannelConfig,
    pw_list: []const tm_generator.PseudoWavefunction,
) ![]f64 {
    const rho_val = try allocator.alloc(f64, grid.n);
    @memset(rho_val, 0);
    for (valence_channels, pw_list) |ch, pw| {
        const occ = ch.occupation;
        for (0..grid.n) |i| {
            const r = grid.r[i];
            const r2 = if (r > 1e-30) r * r else 1e-30;
            rho_val[i] += occ * pw.u[i] * pw.u[i] / (4.0 * std.math.pi * r2);
        }
    }
    return rho_val;
}

fn build_hartree_potential(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    rho_val: []const f64,
) ![]f64 {
    const v_h = try allocator.alloc(f64, grid.n);
    atomic_solver.radial_poisson(grid, rho_val, v_h);
    return v_h;
}

fn build_xc_potential(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    xc: xc_mod.Functional,
    rho_val: []const f64,
    rho_core: ?[]const f64,
) ![]f64 {
    if (rho_val.len != grid.n) return error.InvalidGeneratorConfig;
    if (rho_core) |core| {
        if (core.len != grid.n) return error.InvalidGeneratorConfig;
    }

    const v_xc = try allocator.alloc(f64, grid.n);
    errdefer allocator.free(v_xc);

    for (rho_val, 0..) |rho_i, i| {
        const core_i = if (rho_core) |core| core[i] else 0.0;
        const rho_total = rho_i + core_i;
        if (!std.math.isFinite(rho_total) or rho_total < 0.0) {
            return error.InvalidGeneratorConfig;
        }
        const xc_pt = xc_mod.eval_point(xc, rho_total, 0);
        v_xc[i] = xc_pt.df_dn;
    }
    return v_xc;
}

fn build_local_screened_potential(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    channel_potential: []const f64,
    smooth_radius: ?f64,
) ![]f64 {
    if (channel_potential.len != grid.n) return error.InvalidGeneratorConfig;

    const local = try allocator.alloc(f64, grid.n);
    errdefer allocator.free(local);

    if (smooth_radius) |radius| {
        try build_smooth_local_potential(grid, channel_potential, radius, local);
    } else {
        @memcpy(local, channel_potential);
    }
    return local;
}

fn build_smooth_local_potential(
    grid: *const RadialGrid,
    channel_potential: []const f64,
    smooth_radius: f64,
    local: []f64,
) !void {
    std.debug.assert(channel_potential.len == grid.n);
    std.debug.assert(local.len == grid.n);
    if (!std.math.isFinite(smooth_radius) or smooth_radius <= 0.0) {
        return error.InvalidGeneratorConfig;
    }

    const i_smooth = tm_generator.find_grid_index(grid, smooth_radius);
    if (i_smooth < 2 or i_smooth + 2 >= grid.n) return error.InvalidGeneratorConfig;

    const radius = grid.r[i_smooth];
    const inner_value = channel_potential[i_smooth];
    for (0..grid.n) |i| {
        if (i > i_smooth) {
            local[i] = channel_potential[i];
            continue;
        }
        const t = grid.r[i] / radius;
        const weight = smootherstep(t);
        local[i] = inner_value + weight * (channel_potential[i] - inner_value);
    }
}

fn smootherstep(t: f64) f64 {
    std.debug.assert(std.math.isFinite(t));
    std.debug.assert(t >= 0.0 and t <= 1.0);

    const t2 = t * t;
    const t3 = t2 * t;
    return t3 * (10.0 + t * (-15.0 + 6.0 * t));
}

fn find_local_channel_index(
    valence_channels: []const ChannelConfig,
    local_n: ?u32,
    l_local: u32,
) !usize {
    var found: ?usize = null;
    for (valence_channels, 0..) |ch, i| {
        if (ch.l == l_local and (local_n == null or ch.n == local_n.?)) {
            if (found != null) return error.InvalidGeneratorConfig;
            found = i;
        }
    }
    return found orelse error.InvalidGeneratorConfig;
}

fn build_nonlocal_data(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    valence_channels: []const ChannelConfig,
    local_channel_index: usize,
    pw_list: []const tm_generator.PseudoWavefunction,
    v_local_screened: []const f64,
) !NonLocalData {
    var betas: std.ArrayListUnmanaged(upf_writer.BetaData) = .empty;
    errdefer {
        for (betas.items) |b| allocator.free(@constCast(b.values));
        betas.deinit(allocator);
    }

    var projectors: std.ArrayListUnmanaged(ProjectorWork) = .empty;
    defer projectors.deinit(allocator);

    for (valence_channels, pw_list, 0..) |ch, pw, i| {
        if (i == local_channel_index) continue;

        var kb = try kb_projector.build_projector(
            allocator,
            grid,
            pw.v_ps,
            v_local_screened,
            pw.u,
            ch.l,
        );
        defer kb.deinit();

        const beta_copy = try allocator.alloc(f64, grid.n);
        @memcpy(beta_copy, kb.beta);

        try betas.append(allocator, .{
            .l = ch.l,
            .values = beta_copy,
            .cutoff_index = projector_cutoff_index(grid, ch.rc),
        });
        if (!std.math.isFinite(kb.d_ion) or @abs(kb.d_ion) <= 1e-14) {
            return error.InvalidGeneratorConfig;
        }
        try projectors.append(allocator, .{
            .l = ch.l,
            .u = pw.u,
            .beta = beta_copy,
        });
    }

    const dij = try build_kb_dij_matrix(allocator, grid, projectors.items);
    return .{ .betas = betas, .dij = dij, .allocator = allocator };
}

fn projector_cutoff_index(grid: *const RadialGrid, channel_rc: f64) usize {
    std.debug.assert(std.math.isFinite(channel_rc));
    const index = tm_generator.find_grid_index(grid, channel_rc) + 1;
    std.debug.assert(index <= grid.n);
    return index;
}

fn build_kb_dij_matrix(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    projectors: []const ProjectorWork,
) ![]f64 {
    const nb = projectors.len;
    const overlap = try allocator.alloc(f64, nb * nb);
    errdefer allocator.free(overlap);

    for (projectors, 0..) |row, i| {
        for (projectors, 0..) |col, j| {
            overlap[i * nb + j] = if (row.l == col.l)
                projector_overlap(grid, col.u, row.beta)
            else
                0.0;
        }
    }

    symmetrize_same_l_blocks(overlap, projectors);
    try invert_square_matrix(overlap, nb);
    return overlap;
}

fn symmetrize_same_l_blocks(values: []f64, projectors: []const ProjectorWork) void {
    const nb = projectors.len;
    std.debug.assert(values.len == nb * nb);

    for (0..nb) |i| {
        for (i + 1..nb) |j| {
            if (projectors[i].l != projectors[j].l) continue;
            const avg = 0.5 * (values[i * nb + j] + values[j * nb + i]);
            values[i * nb + j] = avg;
            values[j * nb + i] = avg;
        }
    }
}

fn projector_overlap(grid: *const RadialGrid, u: []const f64, beta: []const f64) f64 {
    std.debug.assert(u.len == grid.n);
    std.debug.assert(beta.len == grid.n);

    var sum: f64 = 0;
    for (1..grid.n) |i| {
        const f_prev = u[i - 1] * beta[i - 1] * grid.rab[i - 1];
        const f_curr = u[i] * beta[i] * grid.rab[i];
        sum += 0.5 * (f_prev + f_curr);
    }
    return sum;
}

fn invert_square_matrix(values: []f64, n: usize) !void {
    std.debug.assert(values.len == n * n);

    const max_n = 8;
    if (n > max_n) return error.InvalidGeneratorConfig;

    var inverse: [max_n * max_n]f64 = @splat(0.0);
    for (0..n) |i| inverse[i * n + i] = 1.0;

    for (0..n) |pivot_col| {
        const pivot_row = find_pivot_row(values, n, pivot_col);
        const pivot = values[pivot_row * n + pivot_col];
        if (!std.math.isFinite(pivot) or @abs(pivot) <= 1e-14) {
            return error.InvalidGeneratorConfig;
        }
        swap_matrix_rows(values, inverse[0 .. n * n], n, pivot_col, pivot_row);
        scale_matrix_row(values, inverse[0 .. n * n], n, pivot_col);
        eliminate_matrix_column(values, inverse[0 .. n * n], n, pivot_col);
    }

    @memcpy(values, inverse[0 .. n * n]);
}

fn find_pivot_row(values: []const f64, n: usize, pivot_col: usize) usize {
    var best = pivot_col;
    var best_abs = @abs(values[pivot_col * n + pivot_col]);
    for (pivot_col + 1..n) |row| {
        const candidate = @abs(values[row * n + pivot_col]);
        if (candidate > best_abs) {
            best_abs = candidate;
            best = row;
        }
    }
    return best;
}

fn swap_matrix_rows(
    values: []f64,
    inverse: []f64,
    n: usize,
    row_a: usize,
    row_b: usize,
) void {
    if (row_a == row_b) return;
    for (0..n) |col| {
        std.mem.swap(f64, &values[row_a * n + col], &values[row_b * n + col]);
        std.mem.swap(f64, &inverse[row_a * n + col], &inverse[row_b * n + col]);
    }
}

fn scale_matrix_row(values: []f64, inverse: []f64, n: usize, row: usize) void {
    const inv_pivot = 1.0 / values[row * n + row];
    for (0..n) |col| {
        values[row * n + col] *= inv_pivot;
        inverse[row * n + col] *= inv_pivot;
    }
}

fn eliminate_matrix_column(values: []f64, inverse: []f64, n: usize, pivot_col: usize) void {
    for (0..n) |row| {
        if (row == pivot_col) continue;
        const factor = values[row * n + pivot_col];
        if (factor == 0.0) continue;
        for (0..n) |col| {
            values[row * n + col] -= factor * values[pivot_col * n + col];
            inverse[row * n + col] -= factor * inverse[pivot_col * n + col];
        }
    }
}

fn build_atomic_rho(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    rho_val: []const f64,
) ![]f64 {
    const rho_atom = try allocator.alloc(f64, grid.n);
    for (0..grid.n) |i| {
        rho_atom[i] = rho_val[i] * 4.0 * std.math.pi * grid.r[i] * grid.r[i];
    }
    return rho_atom;
}

fn build_ae_core_nlcc(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    generator: GeneratorConfig,
    ae_result: *const atomic_solver.AtomResult,
    config: NlccConfig,
) ![]f64 {
    try validate_nlcc_config(config);

    const core_density = try build_ae_core_density(allocator, grid, generator, ae_result);
    defer allocator.free(core_density);

    const nlcc = try allocator.alloc(f64, grid.n);
    errdefer allocator.free(nlcc);

    for (nlcc, core_density, grid.r) |*rho, core, r| {
        rho.* = core * ae_core_smoothing_weight(r, config.radius);
    }
    try normalize_spherical_density(grid, nlcc, config.charge);
    return nlcc;
}

fn build_optional_nlcc(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    generator: GeneratorConfig,
    ae_result: *const atomic_solver.AtomResult,
) !?[]f64 {
    const nlcc_config = generator.nlcc orelse return null;
    return try build_ae_core_nlcc(allocator, grid, generator, ae_result, nlcc_config);
}

fn validate_nlcc_config(config: NlccConfig) !void {
    if (!std.math.isFinite(config.charge) or config.charge <= 0.0) {
        return error.InvalidGeneratorConfig;
    }
    if (!std.math.isFinite(config.radius) or config.radius <= 0.0) {
        return error.InvalidGeneratorConfig;
    }
}

fn build_ae_core_density(
    allocator: std.mem.Allocator,
    grid: *const RadialGrid,
    generator: GeneratorConfig,
    ae_result: *const atomic_solver.AtomResult,
) ![]f64 {
    if (generator.all_orbitals.len != ae_result.wavefunctions.len) {
        return error.InvalidGeneratorConfig;
    }

    const density = try allocator.alloc(f64, grid.n);
    @memset(density, 0.0);
    errdefer allocator.free(density);

    var core_occupation: f64 = 0.0;
    for (generator.all_orbitals, ae_result.wavefunctions) |orb, wave| {
        if (wave.len != grid.n) return error.InvalidGeneratorConfig;
        if (orb.occupation <= 0.0 or is_valence_orbital(generator.valence_channels, orb)) {
            continue;
        }
        core_occupation += orb.occupation;
        accumulate_orbital_density(grid, wave, orb.occupation, density);
    }
    if (core_occupation <= 0.0) return error.InvalidGeneratorConfig;
    return density;
}

fn is_valence_orbital(valence_channels: []const ChannelConfig, orb: OrbitalDef) bool {
    for (valence_channels) |channel| {
        if (channel.occupation <= 0.0) continue;
        if (channel.n == orb.n and channel.l == orb.l) return true;
    }
    return false;
}

fn accumulate_orbital_density(
    grid: *const RadialGrid,
    wave: []const f64,
    occupation: f64,
    density: []f64,
) void {
    std.debug.assert(wave.len == grid.n);
    std.debug.assert(density.len == grid.n);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        const r2 = if (r > 1e-30) r * r else 1e-30;
        density[i] += occupation * wave[i] * wave[i] / (4.0 * std.math.pi * r2);
    }
}

fn ae_core_smoothing_weight(r: f64, radius: f64) f64 {
    std.debug.assert(std.math.isFinite(r));
    std.debug.assert(std.math.isFinite(radius) and radius > 0.0);

    const x = r / radius;
    const x2 = x * x;
    const x4 = x2 * x2;
    return 1.0 - @exp(-x4);
}

fn normalize_spherical_density(
    grid: *const RadialGrid,
    rho: []f64,
    target_charge: f64,
) !void {
    std.debug.assert(rho.len == grid.n);

    const charge = integrate_spherical_density(grid, rho);
    if (!std.math.isFinite(charge) or charge <= 0.0) {
        return error.InvalidGeneratorConfig;
    }
    const scale = target_charge / charge;
    for (rho) |*value| value.* *= scale;
}

fn integrate_spherical_density(grid: *const RadialGrid, rho: []const f64) f64 {
    std.debug.assert(rho.len == grid.n);
    var sum: f64 = 0.0;
    for (0..grid.n) |i| {
        const weight: f64 = if (i == 0 or i + 1 == grid.n) 0.5 else 1.0;
        sum += weight * grid.r[i] * grid.r[i] * rho[i] * grid.rab[i];
    }
    return 4.0 * std.math.pi * sum;
}

fn max_angular_momentum(valence_channels: []const ChannelConfig) u32 {
    var l_max: u32 = 0;
    for (valence_channels) |ch| {
        if (ch.l > l_max) l_max = ch.l;
    }
    return l_max;
}

fn total_valence_charge(valence_channels: []const ChannelConfig) f64 {
    var z_valence: f64 = 0;
    for (valence_channels) |ch| {
        z_valence += ch.occupation;
    }
    return z_valence;
}

// ============================================================================
// Tests
// ============================================================================

test "end-to-end: H atom s+p UPF round-trip" {
    const allocator = std.testing.allocator;

    const all_orbs = [_]OrbitalDef{
        .{ .n = 1, .l = 0, .occupation = 1.0 },
    };
    const val_channels = [_]ChannelConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0, .rc = 1.5 },
    };

    const io = std.testing.io;
    var buf: [1024 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);

    try generate_pseudopotential(allocator, .{
        .z = 1,
        .element = "H",
        .xc = .lda_pz,
        .all_orbitals = &all_orbs,
        .valence_channels = &val_channels,
        .l_local = 0,
    }, &writer);

    const output = writer.buffered();
    try std.testing.expect(output.len > 100);
    try std.testing.expect(std.mem.indexOf(u8, output, "element=\"H\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "z_valence=\"1.0\"") != null);

    // Write and parse back
    const tmp_path = "/tmp/ppgen_h_test.upf";
    const cwd = std.Io.Dir.cwd();
    {
        const file = try cwd.createFile(io, tmp_path, .{});
        defer file.close(io);

        try file.writeStreamingAll(io, output);
    }
    defer cwd.deleteFile(io, tmp_path) catch {};

    var parsed = try pseudo_parser.load(allocator, io, .{
        .element = "H",
        .path = tmp_path,
        .format = .upf,
    });
    defer parsed.deinit(allocator);

    try std.testing.expectApproxEqAbs(1.0, parsed.header.z_valence.?, 0.01);
    try std.testing.expectEqual(@as(usize, 4000), parsed.upf.?.r.len);
}

test "KB D matrix inverts Hermitianized same-l projector overlap block" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 16, 1e-4, 4.0);
    defer grid.deinit();

    var wave0: [16]f64 = undefined;
    var wave1: [16]f64 = undefined;
    var beta0: [16]f64 = undefined;
    var beta1: [16]f64 = undefined;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        wave0[i] = r * @exp(-r);
        wave1[i] = r * r * @exp(-0.5 * r);
        beta0[i] = (1.0 + 0.1 * r) * wave0[i];
        beta1[i] = (0.7 - 0.05 * r) * wave1[i];
    }

    const projectors = [_]ProjectorWork{
        .{ .l = 1, .u = &wave0, .beta = &beta0 },
        .{ .l = 1, .u = &wave1, .beta = &beta1 },
    };
    const beta_phi = [_]f64{
        projector_overlap(&grid, &wave0, &beta0),
        projector_overlap(&grid, &wave1, &beta0),
        projector_overlap(&grid, &wave0, &beta1),
        projector_overlap(&grid, &wave1, &beta1),
    };
    const sym_beta_phi = [_]f64{
        beta_phi[0],
        0.5 * (beta_phi[1] + beta_phi[2]),
        0.5 * (beta_phi[1] + beta_phi[2]),
        beta_phi[3],
    };

    const dij = try build_kb_dij_matrix(allocator, &grid, &projectors);
    defer allocator.free(dij);

    try std.testing.expectApproxEqAbs(dij[1], dij[2], 1e-14);

    for (0..2) |row| {
        for (0..2) |col| {
            var value: f64 = 0.0;
            for (0..2) |k| value += dij[row * 2 + k] * sym_beta_phi[k * 2 + col];
            const expected: f64 = if (row == col) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, value, 1e-10);
        }
    }
}

test "local channel selection rejects ambiguous same-l channels" {
    const channels = [_]ChannelConfig{
        .{ .n = 3, .l = 1, .occupation = 2.0, .rc = 1.6 },
        .{ .n = 4, .l = 1, .occupation = 0.0, .rc = 1.6, .reference_energy = 0.0 },
    };
    try std.testing.expectError(
        error.InvalidGeneratorConfig,
        find_local_channel_index(&channels, null, 1),
    );
    try std.testing.expectEqual(@as(usize, 0), try find_local_channel_index(&channels, 3, 1));
}

test "projector cutoff uses channel support" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 64, 1e-4, 8.0);
    defer grid.deinit();

    const short_channel = projector_cutoff_index(&grid, 1.6);
    const long_channel = projector_cutoff_index(&grid, 2.2);
    const short_only = tm_generator.find_grid_index(&grid, 1.6) + 1;
    const channel_only = tm_generator.find_grid_index(&grid, 2.2) + 1;

    try std.testing.expectEqual(short_only, short_channel);
    try std.testing.expectEqual(channel_only, long_channel);
}

test "smooth local potential preserves cutoff and outer channel values" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 64, 1e-4, 8.0);
    defer grid.deinit();

    const radius = 1.2;
    const i_smooth = tm_generator.find_grid_index(&grid, radius);
    var channel: [64]f64 = undefined;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        channel[i] = -2.0 + 0.5 * r + 0.1 * r * r;
    }

    const local = try build_local_screened_potential(allocator, &grid, &channel, radius);
    defer allocator.free(local);

    try std.testing.expectApproxEqAbs(channel[i_smooth], local[i_smooth], 1e-14);
    for (i_smooth + 1..grid.n) |i| {
        try std.testing.expectApproxEqAbs(channel[i], local[i], 1e-14);
    }
    try std.testing.expectApproxEqAbs(channel[i_smooth], local[0], 1e-14);
}

test "XC potential uses valence plus NLCC density" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4, 1e-4, 1.0);
    defer grid.deinit();

    const rho_val = [_]f64{ 0.10, 0.20, 0.30, 0.40 };
    const rho_core = [_]f64{ 0.05, 0.04, 0.03, 0.02 };
    const v_xc = try build_xc_potential(allocator, &grid, .lda_pz, &rho_val, &rho_core);
    defer allocator.free(v_xc);

    for (rho_val, rho_core, 0..) |val, core, i| {
        const expected = xc_mod.eval_point(.lda_pz, val + core, 0.0).df_dn;
        try std.testing.expectApproxEqAbs(expected, v_xc[i], 1e-14);
    }
}

test "XC potential rejects mismatched NLCC density length" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 4, 1e-4, 1.0);
    defer grid.deinit();

    const rho_val = [_]f64{ 0.10, 0.20, 0.30, 0.40 };
    const rho_core = [_]f64{ 0.05, 0.04, 0.03 };
    try std.testing.expectError(
        error.InvalidGeneratorConfig,
        build_xc_potential(allocator, &grid, .lda_pz, &rho_val, &rho_core),
    );
}

test "AE core density excludes occupied valence orbitals" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 64, 1e-4, 8.0);
    defer grid.deinit();

    var core_wave: [64]f64 = undefined;
    var valence_wave: [64]f64 = undefined;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        core_wave[i] = r * @exp(-r);
        valence_wave[i] = 100.0 * r * @exp(-0.5 * r);
    }

    const all_orbitals = [_]OrbitalDef{
        .{ .n = 1, .l = 0, .occupation = 2.0 },
        .{ .n = 2, .l = 0, .occupation = 1.0 },
    };
    const valence_channels = [_]ChannelConfig{
        .{ .n = 2, .l = 0, .occupation = 1.0, .rc = 1.5 },
    };
    var eigenvalues = [_]f64{ -1.0, -0.1 };
    var rho = [_]f64{0.0} ** 64;
    var v_eff = [_]f64{0.0} ** 64;
    var wavefunctions = [_][]f64{ &core_wave, &valence_wave };
    const ae_result = atomic_solver.AtomResult{
        .total_energy = 0.0,
        .eigenvalues = &eigenvalues,
        .rho = &rho,
        .v_eff = &v_eff,
        .wavefunctions = &wavefunctions,
        .allocator = allocator,
    };

    const density = try build_ae_core_density(allocator, &grid, .{
        .z = 2.0,
        .element = "X",
        .xc = .lda_pz,
        .all_orbitals = &all_orbitals,
        .valence_channels = &valence_channels,
        .l_local = 0,
    }, &ae_result);
    defer allocator.free(density);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        const r2 = if (r > 1e-30) r * r else 1e-30;
        const expected = 2.0 * core_wave[i] * core_wave[i] / (4.0 * std.math.pi * r2);
        try std.testing.expectApproxEqAbs(expected, density[i], 1e-14);
    }
}

test "AE core NLCC integrates to configured partial charge" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 256, 1e-5, 12.0);
    defer grid.deinit();

    var core_wave: [256]f64 = undefined;
    var valence_wave: [256]f64 = undefined;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        core_wave[i] = r * @exp(-r);
        valence_wave[i] = r * r * @exp(-0.4 * r);
    }

    const all_orbitals = [_]OrbitalDef{
        .{ .n = 1, .l = 0, .occupation = 2.0 },
        .{ .n = 2, .l = 1, .occupation = 1.0 },
    };
    const valence_channels = [_]ChannelConfig{
        .{ .n = 2, .l = 1, .occupation = 1.0, .rc = 1.5 },
    };
    var eigenvalues = [_]f64{ -1.0, -0.1 };
    var rho = [_]f64{0.0} ** 256;
    var v_eff = [_]f64{0.0} ** 256;
    var wavefunctions = [_][]f64{ &core_wave, &valence_wave };
    const ae_result = atomic_solver.AtomResult{
        .total_energy = 0.0,
        .eigenvalues = &eigenvalues,
        .rho = &rho,
        .v_eff = &v_eff,
        .wavefunctions = &wavefunctions,
        .allocator = allocator,
    };
    const generator = GeneratorConfig{
        .z = 3.0,
        .element = "X",
        .xc = .lda_pz,
        .all_orbitals = &all_orbitals,
        .valence_channels = &valence_channels,
        .l_local = 1,
    };

    const nlcc = try build_ae_core_nlcc(allocator, &grid, generator, &ae_result, .{
        .charge = 0.5,
        .radius = 0.8,
    });
    defer allocator.free(nlcc);

    const charge = integrate_spherical_density(&grid, nlcc);
    try std.testing.expectApproxEqAbs(0.5, charge, 1e-6);
}

test "AE core NLCC rejects invalid parameters and missing core density" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 64, 1e-4, 8.0);
    defer grid.deinit();

    var valence_wave: [64]f64 = undefined;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        valence_wave[i] = r * @exp(-r);
    }
    const all_orbitals = [_]OrbitalDef{
        .{ .n = 1, .l = 0, .occupation = 1.0 },
    };
    const valence_channels = [_]ChannelConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0, .rc = 1.5 },
    };
    var eigenvalues = [_]f64{-1.0};
    var rho = [_]f64{0.0} ** 64;
    var v_eff = [_]f64{0.0} ** 64;
    var wavefunctions = [_][]f64{&valence_wave};
    const ae_result = atomic_solver.AtomResult{
        .total_energy = 0.0,
        .eigenvalues = &eigenvalues,
        .rho = &rho,
        .v_eff = &v_eff,
        .wavefunctions = &wavefunctions,
        .allocator = allocator,
    };
    const generator = GeneratorConfig{
        .z = 1.0,
        .element = "X",
        .xc = .lda_pz,
        .all_orbitals = &all_orbitals,
        .valence_channels = &valence_channels,
        .l_local = 0,
    };

    try std.testing.expectError(
        error.InvalidGeneratorConfig,
        build_ae_core_nlcc(allocator, &grid, generator, &ae_result, .{
            .charge = 0.0,
            .radius = 0.35,
        }),
    );
    try std.testing.expectError(
        error.InvalidGeneratorConfig,
        build_ae_core_nlcc(allocator, &grid, generator, &ae_result, .{
            .charge = 0.5,
            .radius = 0.0,
        }),
    );
    try std.testing.expectError(
        error.InvalidGeneratorConfig,
        build_ae_core_nlcc(allocator, &grid, generator, &ae_result, .{
            .charge = 0.5,
            .radius = 0.8,
        }),
    );
}

test "KB D matrix keeps different-l projector blocks decoupled" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 16, 1e-4, 4.0);
    defer grid.deinit();

    var wave0: [16]f64 = undefined;
    var wave1: [16]f64 = undefined;
    var beta0: [16]f64 = undefined;
    var beta1: [16]f64 = undefined;
    for (0..grid.n) |i| {
        const r = grid.r[i];
        wave0[i] = r * @exp(-r);
        wave1[i] = r * r * @exp(-0.5 * r);
        beta0[i] = 1.2 * wave0[i];
        beta1[i] = 0.8 * wave1[i];
    }

    const projectors = [_]ProjectorWork{
        .{ .l = 0, .u = &wave0, .beta = &beta0 },
        .{ .l = 1, .u = &wave1, .beta = &beta1 },
    };
    const dij = try build_kb_dij_matrix(allocator, &grid, &projectors);
    defer allocator.free(dij);

    try std.testing.expectApproxEqAbs(0.0, dij[1], 1e-14);
    try std.testing.expectApproxEqAbs(0.0, dij[2], 1e-14);
    try std.testing.expect(dij[0] > 0.0);
    try std.testing.expect(dij[3] > 0.0);
}

test "log derivative report is exact for identical channel potentials" {
    const allocator = std.testing.allocator;
    var grid = try RadialGrid.init(allocator, 512, 1e-6, 30.0);
    defer grid.deinit();

    const potential = try allocator.alloc(f64, grid.n);
    defer allocator.free(potential);

    const wave = try allocator.alloc(f64, grid.n);
    defer allocator.free(wave);

    for (0..grid.n) |i| {
        const r = grid.r[i];
        potential[i] = if (r > 1e-30) -2.0 / r else -2.0 / 1e-30;
        wave[i] = r * @exp(-r);
    }

    const channels = [_]ChannelConfig{
        .{ .n = 1, .l = 0, .occupation = 1.0, .rc = 1.5 },
    };
    const pw_list = [_]tm_generator.PseudoWavefunction{
        .{
            .u = wave,
            .v_ps = potential,
            .rc = 1.5,
            .l = 0,
            .allocator = allocator,
        },
    };
    const energies = [_]f64{ -0.5, 0.2 };

    var report = try build_log_derivative_report(
        allocator,
        &grid,
        &channels,
        &pw_list,
        potential,
        .{ .r_match = 3.0, .energies = &energies },
    );
    defer report.deinit();

    try std.testing.expectApproxEqAbs(0.0, report.max_abs_delta, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, report.rms_delta, 1e-12);
    try std.testing.expectEqual(@as(usize, 2), report.samples.len);
    try std.testing.expectEqual(@as(usize, 0), report.pole_mismatch_count);
}

test "log derivative pole mismatch flags ghost candidates" {
    const samples = [_]LogDerivativeSample{
        .{
            .channel_n = 3,
            .l = 1,
            .energy = 0.0,
            .ae = -12.0,
            .pseudo = -12.0,
            .delta = 0.0,
            .status = .ok,
        },
        .{
            .channel_n = 3,
            .l = 1,
            .energy = 0.4,
            .ae = 4.0,
            .pseudo = -4.0,
            .delta = 0.0,
            .status = .ok,
        },
        .{
            .channel_n = 4,
            .l = 2,
            .energy = 0.0,
            .ae = -8.0,
            .pseudo = -8.0,
            .delta = 0.0,
            .status = .ok,
        },
        .{
            .channel_n = 4,
            .l = 2,
            .energy = 0.4,
            .ae = 8.0,
            .pseudo = 9.0,
            .delta = 0.0,
            .status = .ok,
        },
    };

    try std.testing.expectEqual(
        @as(usize, 1),
        count_log_derivative_pole_mismatches(&samples, 2),
    );
}
