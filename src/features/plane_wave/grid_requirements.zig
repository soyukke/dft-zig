const basis = @import("basis.zig");

pub const GridRequirement = struct {
    nx: usize,
    ny: usize,
    nz: usize,
};

pub fn grid_requirement(gvecs: []basis.GVector) GridRequirement {
    var max_h: usize = 0;
    var max_k: usize = 0;
    var max_l: usize = 0;
    for (gvecs) |g| {
        const ah = @abs(g.h);
        const ak = @abs(g.k);
        const al = @abs(g.l);
        if (@as(usize, @intCast(ah)) > max_h) max_h = @intCast(ah);
        if (@as(usize, @intCast(ak)) > max_k) max_k = @intCast(ak);
        if (@as(usize, @intCast(al)) > max_l) max_l = @intCast(al);
    }
    return .{
        .nx = 2 * max_h + 1,
        .ny = 2 * max_k + 1,
        .nz = 2 * max_l + 1,
    };
}
