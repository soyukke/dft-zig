const std = @import("std");
const dft_zig = @import("dft_zig");
const xc_mod = dft_zig.xc;
pub const radial_grid = @import("radial_grid.zig");
pub const schrodinger = @import("schrodinger.zig");
pub const atomic_solver = @import("atomic_solver.zig");
pub const diagnostics = @import("diagnostics.zig");
pub const tm_generator = @import("tm_generator.zig");
pub const kb_projector = @import("kb_projector.zig");
pub const upf_writer = @import("upf_writer.zig");
pub const pipeline = @import("pipeline.zig");
pub const si_test = @import("si_test.zig");
pub const integration_test = @import("integration_test.zig");

const CliOptions = struct {
    output_path: []const u8,
    xc: xc_mod.Functional,
    rc_s: f64,
    rc_p: f64,
    rc_d: f64,
    p_energy: f64,
    d_energy: f64,
    d2_energy: f64,
    l_local: u32,
    local_n: u32,
    local_smooth_radius: ?f64,
    log_deriv_path: ?[]const u8,
    log_deriv_r: f64,
    log_deriv_min: f64,
    log_deriv_max: f64,
    log_deriv_step: f64,
    nlcc_charge: f64,
    nlcc_radius: f64,
    d2_energy_set: bool,
};

const CliParseState = struct {
    xc: xc_mod.Functional = .lda_pz,
    rc_s: f64 = 1.71,
    rc_p: f64 = 1.64,
    rc_d: f64 = 1.9,
    p_energy: f64 = 0.8,
    d_energy: f64 = 1.2,
    d2_energy: f64 = 2.0,
    d2_energy_set: bool = false,
    l_local: u32 = 2,
    local_n: u32 = 3,
    local_smooth_radius: ?f64 = 1.2,
    log_deriv_path: ?[]const u8 = null,
    log_deriv_r: f64 = 3.0,
    log_deriv_min: f64 = 0.0,
    log_deriv_max: f64 = 1.6,
    log_deriv_step: f64 = 0.4,
    nlcc_charge: f64 = 0.7241414335,
    nlcc_radius: f64 = 0.8,
    output_path: ?[]const u8 = null,

    fn finish(self: CliParseState) !CliOptions {
        try validate_log_deriv_energy_grid(
            self.log_deriv_min,
            self.log_deriv_max,
            self.log_deriv_step,
        );
        try validate_nlcc(self.nlcc_charge, self.nlcc_radius);
        return .{
            .output_path = self.output_path orelse return error.InvalidArguments,
            .xc = self.xc,
            .rc_s = self.rc_s,
            .rc_p = self.rc_p,
            .rc_d = self.rc_d,
            .p_energy = self.p_energy,
            .d_energy = self.d_energy,
            .d2_energy = self.d2_energy,
            .l_local = self.l_local,
            .local_n = self.local_n,
            .local_smooth_radius = self.local_smooth_radius,
            .log_deriv_path = self.log_deriv_path,
            .log_deriv_r = self.log_deriv_r,
            .log_deriv_min = self.log_deriv_min,
            .log_deriv_max = self.log_deriv_max,
            .log_deriv_step = self.log_deriv_step,
            .nlcc_charge = self.nlcc_charge,
            .nlcc_radius = self.nlcc_radius,
            .d2_energy_set = self.d2_energy_set,
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    var stderr_buffer: [256]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;

    var args_iter = try init.minimal.args.iterateAllocator(allocator);
    defer args_iter.deinit();

    _ = args_iter.next();
    const options = parse_args(&args_iter) catch |err| {
        try write_usage(stderr);
        try stderr.flush();
        return err;
    };

    var buf: [2 * 1024 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try generate_si_pseudopotential(allocator, options, &writer);

    const output = writer.buffered();
    const cwd = std.Io.Dir.cwd();
    try write_file(io, cwd, options.output_path, output);

    if (options.log_deriv_path) |path| {
        try write_si_log_derivatives(allocator, io, cwd, options, path);
    }

    try stdout.print("Written: {s}\n", .{options.output_path});
    try stdout.flush();
}

fn write_file(
    io: std.Io,
    cwd: std.Io.Dir,
    path: []const u8,
    contents: []const u8,
) !void {
    const file = try cwd.createFile(io, path, .{});
    defer file.close(io);

    try file.writeStreamingAll(io, contents);
}

fn write_si_log_derivatives(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    options: CliOptions,
    path: []const u8,
) !void {
    var report = try compute_si_log_derivatives(allocator, options);
    defer report.deinit();

    var buf: [512 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try write_log_derivative_report(&writer, report);
    try write_file(io, cwd, path, writer.buffered());
}

fn generate_si_pseudopotential(
    allocator: std.mem.Allocator,
    options: CliOptions,
    writer: anytype,
) !void {
    const all_orbs = si_all_orbitals();
    const d_local_channels = si_d_local_valence_channels(options);
    const d_local_multi_channels = si_d_local_multi_valence_channels(options);
    const multi_d_channels = si_multi_d_valence_channels(options);
    const val_channels: []const pipeline.ChannelConfig = if (options.l_local == 2) blk: {
        break :blk if (options.d2_energy_set) &d_local_multi_channels else &d_local_channels;
    } else &multi_d_channels;
    try pipeline.generate_pseudopotential(allocator, .{
        .z = 14,
        .element = "Si",
        .xc = options.xc,
        .all_orbitals = &all_orbs,
        .valence_channels = val_channels,
        .l_local = options.l_local,
        .local_n = options.local_n,
        .local_smooth_radius = options.local_smooth_radius,
        .nlcc = .{
            .charge = options.nlcc_charge,
            .radius = options.nlcc_radius,
        },
    }, writer);
}

fn compute_si_log_derivatives(
    allocator: std.mem.Allocator,
    options: CliOptions,
) !pipeline.LogDerivativeReport {
    const all_orbs = si_all_orbitals();
    const d_local_channels = si_d_local_valence_channels(options);
    const d_local_multi_channels = si_d_local_multi_valence_channels(options);
    const multi_d_channels = si_multi_d_valence_channels(options);
    const val_channels: []const pipeline.ChannelConfig = if (options.l_local == 2) blk: {
        break :blk if (options.d2_energy_set) &d_local_multi_channels else &d_local_channels;
    } else &multi_d_channels;
    const energies = try build_log_derivative_energy_grid(
        allocator,
        options.log_deriv_min,
        options.log_deriv_max,
        options.log_deriv_step,
    );
    defer allocator.free(energies);

    return pipeline.compute_log_derivative_report(allocator, .{
        .z = 14,
        .element = "Si",
        .xc = options.xc,
        .all_orbitals = &all_orbs,
        .valence_channels = val_channels,
        .l_local = options.l_local,
        .local_n = options.local_n,
    }, .{
        .r_match = options.log_deriv_r,
        .energies = energies,
    });
}

fn build_log_derivative_energy_grid(
    allocator: std.mem.Allocator,
    min_energy: f64,
    max_energy: f64,
    step: f64,
) ![]f64 {
    try validate_log_deriv_energy_grid(min_energy, max_energy, step);
    const sample_count = log_derivative_energy_sample_count(min_energy, max_energy, step);
    const energies = try allocator.alloc(f64, sample_count);
    errdefer allocator.free(energies);

    for (energies, 0..) |*energy, i| {
        energy.* = min_energy + @as(f64, @floatFromInt(i)) * step;
    }
    return energies;
}

fn validate_log_deriv_energy_grid(min_energy: f64, max_energy: f64, step: f64) !void {
    if (!std.math.isFinite(min_energy) or
        !std.math.isFinite(max_energy) or
        !std.math.isFinite(step))
    {
        return error.InvalidArguments;
    }
    if (min_energy < -5.0 or max_energy > 5.0 or min_energy > max_energy or step <= 0.0) {
        return error.InvalidArguments;
    }
    if (log_derivative_energy_sample_count(min_energy, max_energy, step) > 256) {
        return error.InvalidArguments;
    }
}

fn log_derivative_energy_sample_count(min_energy: f64, max_energy: f64, step: f64) usize {
    const span = max_energy - min_energy;
    const count_float = @floor(span / step + 1e-10) + 1.0;
    return @intFromFloat(count_float);
}

fn write_log_derivative_report(
    writer: anytype,
    report: pipeline.LogDerivativeReport,
) !void {
    try writer.print(
        "# max_abs_delta {e:.8} rms_delta {e:.8} valid {d} invalid {d} pole_mismatch {d}\n",
        .{
            report.max_abs_delta,
            report.rms_delta,
            report.valid_count,
            report.invalid_count,
            report.pole_mismatch_count,
        },
    );
    try writer.writeAll("channel_n\tl\tenergy_ry\tae\tpseudo\tdelta\tstatus\n");
    for (report.samples) |sample| {
        try writer.print("{d}\t{d}\t{d:.8}\t{e:.12}\t{e:.12}\t{e:.12}\t{s}\n", .{
            sample.channel_n,
            sample.l,
            sample.energy,
            sample.ae,
            sample.pseudo,
            sample.delta,
            log_derivative_status_name(sample.status),
        });
    }
}

fn log_derivative_status_name(status: pipeline.LogDerivativeStatus) []const u8 {
    return switch (status) {
        .ok => "ok",
        .ae_error => "ae_error",
        .pseudo_error => "pseudo_error",
    };
}

fn si_all_orbitals() [5]pipeline.OrbitalDef {
    return .{
        .{ .n = 1, .l = 0, .occupation = 2.0 },
        .{ .n = 2, .l = 0, .occupation = 2.0 },
        .{ .n = 2, .l = 1, .occupation = 6.0 },
        .{ .n = 3, .l = 0, .occupation = 2.0 },
        .{ .n = 3, .l = 1, .occupation = 2.0 },
    };
}

fn si_d_local_valence_channels(options: CliOptions) [4]pipeline.ChannelConfig {
    return .{
        .{ .n = 3, .l = 0, .occupation = 2.0, .rc = options.rc_s },
        .{ .n = 3, .l = 1, .occupation = 2.0, .rc = options.rc_p },
        .{
            .n = 4,
            .l = 1,
            .occupation = 0.0,
            .rc = options.rc_p,
            .reference_energy = options.p_energy,
        },
        .{
            .n = 3,
            .l = 2,
            .occupation = 0.0,
            .rc = options.rc_d,
            .reference_energy = options.d_energy,
        },
    };
}

fn si_d_local_multi_valence_channels(options: CliOptions) [5]pipeline.ChannelConfig {
    return .{
        .{ .n = 3, .l = 0, .occupation = 2.0, .rc = options.rc_s },
        .{ .n = 3, .l = 1, .occupation = 2.0, .rc = options.rc_p },
        .{
            .n = 4,
            .l = 1,
            .occupation = 0.0,
            .rc = options.rc_p,
            .reference_energy = options.p_energy,
        },
        .{
            .n = 3,
            .l = 2,
            .occupation = 0.0,
            .rc = options.rc_d,
            .reference_energy = options.d_energy,
        },
        .{
            .n = 4,
            .l = 2,
            .occupation = 0.0,
            .rc = options.rc_d,
            .reference_energy = options.d2_energy,
        },
    };
}

fn si_multi_d_valence_channels(options: CliOptions) [5]pipeline.ChannelConfig {
    return .{
        .{ .n = 3, .l = 0, .occupation = 2.0, .rc = options.rc_s },
        .{ .n = 3, .l = 1, .occupation = 2.0, .rc = options.rc_p },
        .{
            .n = 4,
            .l = 1,
            .occupation = 0.0,
            .rc = options.rc_p,
            .reference_energy = options.p_energy,
        },
        .{
            .n = 3,
            .l = 2,
            .occupation = 0.0,
            .rc = options.rc_d,
            .reference_energy = options.d_energy,
        },
        .{
            .n = 4,
            .l = 2,
            .occupation = 0.0,
            .rc = options.rc_d,
            .reference_energy = options.d2_energy,
        },
    };
}

fn parse_args(args_iter: anytype) !CliOptions {
    var state = CliParseState{};

    while (args_iter.next()) |arg| {
        if (try parse_named_arg(arg, args_iter, &state)) continue;
        if (std.mem.startsWith(u8, arg, "-")) return error.InvalidArguments;
        if (state.output_path != null) return error.InvalidArguments;
        state.output_path = arg;
    }

    return state.finish();
}

fn parse_named_arg(arg: []const u8, args_iter: anytype, state: *CliParseState) !bool {
    if (try parse_xc_arg(arg, args_iter, state)) return true;
    if (try parse_radius_arg(arg, args_iter, state)) return true;
    if (try parse_d_arg(arg, args_iter, state)) return true;
    if (try parse_reference_arg(arg, args_iter, state)) return true;
    if (try parse_local_arg(arg, args_iter, state)) return true;
    if (try parse_log_deriv_arg(arg, args_iter, state)) return true;
    if (try parse_nlcc_arg(arg, args_iter, state)) return true;
    return false;
}

fn parse_xc_arg(arg: []const u8, args_iter: anytype, state: *CliParseState) !bool {
    if (std.mem.eql(u8, arg, "--xc")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.xc = try parse_xc(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--xc=")) {
        state.xc = try parse_xc(arg["--xc=".len..]);
        return true;
    }
    return false;
}

fn parse_radius_arg(arg: []const u8, args_iter: anytype, state: *CliParseState) !bool {
    if (std.mem.eql(u8, arg, "--rc-s")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.rc_s = try parse_cutoff(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--rc-s=")) {
        state.rc_s = try parse_cutoff(arg["--rc-s=".len..]);
        return true;
    }
    if (std.mem.eql(u8, arg, "--rc-p")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.rc_p = try parse_cutoff(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--rc-p=")) {
        state.rc_p = try parse_cutoff(arg["--rc-p=".len..]);
        return true;
    }
    return false;
}

fn parse_d_arg(arg: []const u8, args_iter: anytype, state: *CliParseState) !bool {
    if (std.mem.eql(u8, arg, "--rc-d")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.rc_d = try parse_cutoff(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--rc-d=")) {
        state.rc_d = try parse_cutoff(arg["--rc-d=".len..]);
        return true;
    }
    if (std.mem.eql(u8, arg, "--d-energy-ry")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.d_energy = try parse_reference_energy(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--d-energy-ry=")) {
        state.d_energy = try parse_reference_energy(arg["--d-energy-ry=".len..]);
        return true;
    }
    return false;
}

fn parse_reference_arg(arg: []const u8, args_iter: anytype, state: *CliParseState) !bool {
    if (std.mem.eql(u8, arg, "--p-ref-energy-ry")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.p_energy = try parse_reference_energy(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--p-ref-energy-ry=")) {
        state.p_energy = try parse_reference_energy(arg["--p-ref-energy-ry=".len..]);
        return true;
    }
    if (std.mem.eql(u8, arg, "--d2-energy-ry")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.d2_energy = try parse_reference_energy(value);
        state.d2_energy_set = true;
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--d2-energy-ry=")) {
        state.d2_energy = try parse_reference_energy(arg["--d2-energy-ry=".len..]);
        state.d2_energy_set = true;
        return true;
    }
    return false;
}

fn parse_local_arg(arg: []const u8, args_iter: anytype, state: *CliParseState) !bool {
    if (std.mem.eql(u8, arg, "--local-l")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.l_local = try parse_local_l(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--local-l=")) {
        state.l_local = try parse_local_l(arg["--local-l=".len..]);
        return true;
    }
    if (std.mem.eql(u8, arg, "--local-n")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.local_n = try parse_local_n(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--local-n=")) {
        state.local_n = try parse_local_n(arg["--local-n=".len..]);
        return true;
    }
    if (std.mem.eql(u8, arg, "--local-smooth-radius")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.local_smooth_radius = try parse_local_smooth_radius(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--local-smooth-radius=")) {
        const value = arg["--local-smooth-radius=".len..];
        state.local_smooth_radius = try parse_local_smooth_radius(value);
        return true;
    }
    return false;
}

fn parse_log_deriv_arg(arg: []const u8, args_iter: anytype, state: *CliParseState) !bool {
    if (std.mem.eql(u8, arg, "--log-deriv")) {
        state.log_deriv_path = args_iter.next() orelse return error.InvalidArguments;
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--log-deriv=")) {
        state.log_deriv_path = arg["--log-deriv=".len..];
        return true;
    }
    if (std.mem.eql(u8, arg, "--log-deriv-r")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.log_deriv_r = try parse_log_deriv_radius(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--log-deriv-r=")) {
        state.log_deriv_r = try parse_log_deriv_radius(arg["--log-deriv-r=".len..]);
        return true;
    }
    if (std.mem.eql(u8, arg, "--log-deriv-min-ry")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.log_deriv_min = try parse_log_deriv_energy(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--log-deriv-min-ry=")) {
        const value = arg["--log-deriv-min-ry=".len..];
        state.log_deriv_min = try parse_log_deriv_energy(value);
        return true;
    }
    if (std.mem.eql(u8, arg, "--log-deriv-max-ry")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.log_deriv_max = try parse_log_deriv_energy(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--log-deriv-max-ry=")) {
        const value = arg["--log-deriv-max-ry=".len..];
        state.log_deriv_max = try parse_log_deriv_energy(value);
        return true;
    }
    if (std.mem.eql(u8, arg, "--log-deriv-step-ry")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.log_deriv_step = try parse_log_deriv_step(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--log-deriv-step-ry=")) {
        const value = arg["--log-deriv-step-ry=".len..];
        state.log_deriv_step = try parse_log_deriv_step(value);
        return true;
    }
    return false;
}

fn parse_nlcc_arg(arg: []const u8, args_iter: anytype, state: *CliParseState) !bool {
    if (std.mem.eql(u8, arg, "--nlcc-charge")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.nlcc_charge = try parse_nlcc_charge(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--nlcc-charge=")) {
        state.nlcc_charge = try parse_nlcc_charge(arg["--nlcc-charge=".len..]);
        return true;
    }
    if (std.mem.eql(u8, arg, "--nlcc-radius")) {
        const value = args_iter.next() orelse return error.InvalidArguments;
        state.nlcc_radius = try parse_nlcc_radius(value);
        return true;
    }
    if (std.mem.startsWith(u8, arg, "--nlcc-radius=")) {
        state.nlcc_radius = try parse_nlcc_radius(arg["--nlcc-radius=".len..]);
        return true;
    }
    return false;
}

fn parse_xc(value: []const u8) !xc_mod.Functional {
    if (std.mem.eql(u8, value, "lda") or std.mem.eql(u8, value, "lda_pz")) return .lda_pz;
    if (std.mem.eql(u8, value, "pbe")) return .pbe;
    return error.InvalidArguments;
}

fn parse_cutoff(value: []const u8) !f64 {
    const rc = std.fmt.parseFloat(f64, value) catch return error.InvalidArguments;
    if (!std.math.isFinite(rc) or rc < 0.5 or rc > 4.0) return error.InvalidArguments;
    return rc;
}

fn parse_local_l(value: []const u8) !u32 {
    const l = std.fmt.parseInt(u32, value, 10) catch return error.InvalidArguments;
    if (l > 2) return error.InvalidArguments;
    return l;
}

fn parse_local_n(value: []const u8) !u32 {
    const n = std.fmt.parseInt(u32, value, 10) catch return error.InvalidArguments;
    if (n < 1 or n > 8) return error.InvalidArguments;
    return n;
}

fn parse_local_smooth_radius(value: []const u8) !f64 {
    const radius = std.fmt.parseFloat(f64, value) catch return error.InvalidArguments;
    if (!std.math.isFinite(radius) or radius < 0.2 or radius > 4.0) {
        return error.InvalidArguments;
    }
    return radius;
}

fn parse_reference_energy(value: []const u8) !f64 {
    const energy = std.fmt.parseFloat(f64, value) catch return error.InvalidArguments;
    if (!std.math.isFinite(energy) or energy < -5.0 or energy > 5.0) {
        return error.InvalidArguments;
    }
    return energy;
}

fn parse_log_deriv_radius(value: []const u8) !f64 {
    const radius = std.fmt.parseFloat(f64, value) catch return error.InvalidArguments;
    if (!std.math.isFinite(radius) or radius < 2.1 or radius > 8.0) {
        return error.InvalidArguments;
    }
    return radius;
}

fn parse_log_deriv_energy(value: []const u8) !f64 {
    const energy = std.fmt.parseFloat(f64, value) catch return error.InvalidArguments;
    if (!std.math.isFinite(energy) or energy < -5.0 or energy > 5.0) {
        return error.InvalidArguments;
    }
    return energy;
}

fn parse_log_deriv_step(value: []const u8) !f64 {
    const step = std.fmt.parseFloat(f64, value) catch return error.InvalidArguments;
    if (!std.math.isFinite(step) or step <= 0.0 or step > 5.0) {
        return error.InvalidArguments;
    }
    return step;
}

fn parse_nlcc_charge(value: []const u8) !f64 {
    const charge = std.fmt.parseFloat(f64, value) catch return error.InvalidArguments;
    return validate_nlcc_charge(charge);
}

fn parse_nlcc_radius(value: []const u8) !f64 {
    const radius = std.fmt.parseFloat(f64, value) catch return error.InvalidArguments;
    return validate_nlcc_radius(radius);
}

fn validate_nlcc(charge: f64, radius: f64) !void {
    _ = try validate_nlcc_charge(charge);
    _ = try validate_nlcc_radius(radius);
}

fn validate_nlcc_charge(charge: f64) !f64 {
    if (!std.math.isFinite(charge) or charge <= 0.0 or charge > 4.0) {
        return error.InvalidArguments;
    }
    return charge;
}

fn validate_nlcc_radius(radius: f64) !f64 {
    if (!std.math.isFinite(radius) or radius < 0.1 or radius > 3.0) {
        return error.InvalidArguments;
    }
    return radius;
}

fn write_usage(stderr: anytype) !void {
    try stderr.writeAll(
        "Usage: ppgen [--xc lda_pz|pbe] [--rc-s bohr] [--rc-p bohr] " ++
            "[--rc-d bohr] [--p-ref-energy-ry ry] [--d-energy-ry ry] " ++
            "[--d2-energy-ry ry] [--local-l 0|1|2] [--local-n n] " ++
            "[--local-smooth-radius bohr] " ++
            "[--nlcc-charge e] [--nlcc-radius bohr] " ++
            "[--log-deriv path] [--log-deriv-r bohr] " ++
            "[--log-deriv-min-ry ry] [--log-deriv-max-ry ry] " ++
            "[--log-deriv-step-ry ry] <output.upf>\n",
    );
    try stderr.writeAll("  Generates Si norm-conserving pseudopotential.\n");
}

test {
    _ = radial_grid;
    _ = schrodinger;
    _ = atomic_solver;
    _ = diagnostics;
    _ = tm_generator;
    _ = kb_projector;
    _ = upf_writer;
    _ = pipeline;
    _ = si_test;
    _ = integration_test;
}

test "parse_xc accepts supported functionals" {
    try std.testing.expectEqual(xc_mod.Functional.lda_pz, try parse_xc("lda"));
    try std.testing.expectEqual(xc_mod.Functional.lda_pz, try parse_xc("lda_pz"));
    try std.testing.expectEqual(xc_mod.Functional.pbe, try parse_xc("pbe"));
    try std.testing.expectError(error.InvalidArguments, parse_xc("blyp"));
}

test "parse_cutoff rejects nonphysical values" {
    try std.testing.expectApproxEqAbs(1.7, try parse_cutoff("1.7"), 1e-12);
    try std.testing.expectError(error.InvalidArguments, parse_cutoff("nan"));
    try std.testing.expectError(error.InvalidArguments, parse_cutoff("0.1"));
    try std.testing.expectError(error.InvalidArguments, parse_cutoff("9.0"));
}

test "parse_local_l only allows generated channels" {
    try std.testing.expectEqual(@as(u32, 0), try parse_local_l("0"));
    try std.testing.expectEqual(@as(u32, 1), try parse_local_l("1"));
    try std.testing.expectEqual(@as(u32, 2), try parse_local_l("2"));
    try std.testing.expectError(error.InvalidArguments, parse_local_l("3"));
}

test "parse_local_n rejects nonphysical principal quantum numbers" {
    try std.testing.expectEqual(@as(u32, 3), try parse_local_n("3"));
    try std.testing.expectError(error.InvalidArguments, parse_local_n("0"));
    try std.testing.expectError(error.InvalidArguments, parse_local_n("9"));
}

test "parse_local_smooth_radius rejects invalid radii" {
    try std.testing.expectApproxEqAbs(1.2, try parse_local_smooth_radius("1.2"), 1e-12);
    try std.testing.expectError(error.InvalidArguments, parse_local_smooth_radius("nan"));
    try std.testing.expectError(error.InvalidArguments, parse_local_smooth_radius("0.1"));
    try std.testing.expectError(error.InvalidArguments, parse_local_smooth_radius("5.0"));
}

test "parse_reference_energy rejects invalid values" {
    try std.testing.expectApproxEqAbs(0.1, try parse_reference_energy("0.1"), 1e-12);
    try std.testing.expectError(error.InvalidArguments, parse_reference_energy("nan"));
    try std.testing.expectError(error.InvalidArguments, parse_reference_energy("9.0"));
}

test "parse_log_deriv_radius rejects invalid match radii" {
    try std.testing.expectApproxEqAbs(3.0, try parse_log_deriv_radius("3.0"), 1e-12);
    try std.testing.expectError(error.InvalidArguments, parse_log_deriv_radius("nan"));
    try std.testing.expectError(error.InvalidArguments, parse_log_deriv_radius("1.9"));
    try std.testing.expectError(error.InvalidArguments, parse_log_deriv_radius("9.0"));
}

test "parse_nlcc rejects invalid partial core parameters" {
    try std.testing.expectApproxEqAbs(0.5, try parse_nlcc_charge("0.5"), 1e-12);
    try std.testing.expectApproxEqAbs(0.35, try parse_nlcc_radius("0.35"), 1e-12);
    try std.testing.expectError(error.InvalidArguments, parse_nlcc_charge("0.0"));
    try std.testing.expectError(error.InvalidArguments, parse_nlcc_charge("nan"));
    try std.testing.expectError(error.InvalidArguments, parse_nlcc_charge("5.0"));
    try std.testing.expectError(error.InvalidArguments, parse_nlcc_radius("0.01"));
    try std.testing.expectError(error.InvalidArguments, parse_nlcc_radius("nan"));
    try std.testing.expectError(error.InvalidArguments, parse_nlcc_radius("5.0"));
}

test "parse_log_deriv_energy_grid rejects invalid ranges" {
    try std.testing.expectApproxEqAbs(1.2, try parse_log_deriv_energy("1.2"), 1e-12);
    try std.testing.expectApproxEqAbs(0.2, try parse_log_deriv_step("0.2"), 1e-12);
    try std.testing.expectError(error.InvalidArguments, parse_log_deriv_energy("nan"));
    try std.testing.expectError(error.InvalidArguments, parse_log_deriv_step("0.0"));
    try std.testing.expectError(
        error.InvalidArguments,
        validate_log_deriv_energy_grid(1.0, 0.0, 0.2),
    );
    try std.testing.expectError(
        error.InvalidArguments,
        validate_log_deriv_energy_grid(0.0, 5.0, 0.01),
    );
}

test "build_log_derivative_energy_grid uses explicit bounded samples" {
    const energies = try build_log_derivative_energy_grid(
        std.testing.allocator,
        0.0,
        0.6,
        0.2,
    );
    defer std.testing.allocator.free(energies);

    try std.testing.expectEqual(@as(usize, 4), energies.len);
    try std.testing.expectApproxEqAbs(0.0, energies[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.2, energies[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.4, energies[2], 1e-12);
    try std.testing.expectApproxEqAbs(0.6, energies[3], 1e-12);
}
