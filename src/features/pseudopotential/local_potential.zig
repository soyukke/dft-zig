const math = @import("../math/math.zig");

pub const LocalPotentialMode = enum {
    tail,
    ewald,
    short_range,
};

pub const LocalPotentialConfig = struct {
    mode: LocalPotentialMode,
    alpha: f64,

    pub fn init(mode: LocalPotentialMode, alpha: f64) LocalPotentialConfig {
        return .{ .mode = mode, .alpha = alpha };
    }
};

pub fn name(mode: LocalPotentialMode) []const u8 {
    return switch (mode) {
        .tail => "tail",
        .ewald => "ewald",
        .short_range => "short_range",
    };
}

pub fn defaultEwaldAlpha(cell_bohr: math.Mat3) f64 {
    const lmin = @min(
        @min(math.Vec3.norm(cell_bohr.row(0)), math.Vec3.norm(cell_bohr.row(1))),
        math.Vec3.norm(cell_bohr.row(2)),
    );
    return 5.0 / lmin;
}

pub fn resolve(
    mode: LocalPotentialMode,
    explicit_alpha: f64,
    cell_bohr: math.Mat3,
) LocalPotentialConfig {
    return .{
        .mode = mode,
        .alpha = if (mode == .ewald)
            (if (explicit_alpha > 0.0) explicit_alpha else defaultEwaldAlpha(cell_bohr))
        else
            0.0,
    };
}
