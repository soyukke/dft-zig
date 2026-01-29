const std = @import("std");

const pseudo = @import("../pseudopotential/pseudopotential.zig");
const local_orbital_potential = @import("local_orbital_potential.zig");
const neighbor_list = @import("neighbor_list.zig");
const math = @import("../math/math.zig");

pub const IonSite = struct {
    position: math.Vec3,
    upf: *const pseudo.UpfData,
};

pub fn buildIonicPotentialGrid(
    alloc: std.mem.Allocator,
    grid: local_orbital_potential.PotentialGrid,
    sites: []const IonSite,
    pbc: neighbor_list.Pbc,
) ![]f64 {
    const count = grid.count();
    if (count == 0) return error.InvalidGrid;
    if (grid.values.len != 0 and grid.values.len != count) return error.InvalidGrid;
    const values = try alloc.alloc(f64, count);
    errdefer alloc.free(values);

    const inv_cell = try local_orbital_potential.invertCell(grid.cell);
    var idx: usize = 0;
    var iz: usize = 0;
    while (iz < grid.dims[2]) : (iz += 1) {
        var iy: usize = 0;
        while (iy < grid.dims[1]) : (iy += 1) {
            var ix: usize = 0;
            while (ix < grid.dims[0]) : (ix += 1) {
                const point = grid.point(ix, iy, iz);
                var sum: f64 = 0.0;
                for (sites) |site| {
                    const delta = math.Vec3.sub(point, site.position);
                    const dvec = minimumImageDelta(grid.cell, inv_cell, pbc, delta);
                    const r = math.Vec3.norm(dvec);
                    sum += sampleLocalPotential(site.upf.*, r);
                }
                values[idx] = sum;
                idx += 1;
            }
        }
    }
    return values;
}

fn sampleLocalPotential(upf: pseudo.UpfData, r: f64) f64 {
    if (upf.r.len == 0) return 0.0;
    if (r <= upf.r[0]) return upf.v_local[0];
    const last = upf.r.len - 1;
    if (r >= upf.r[last]) return upf.v_local[last];

    var lo: usize = 0;
    var hi: usize = last;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (r < upf.r[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    const r0 = upf.r[lo];
    const r1 = upf.r[hi];
    if (r1 <= r0) return upf.v_local[lo];
    const t = (r - r0) / (r1 - r0);
    return upf.v_local[lo] * (1.0 - t) + upf.v_local[hi] * t;
}

fn wrapFrac(value: f64) f64 {
    return value - std.math.floor(value + 0.5);
}

fn minimumImageDelta(cell: math.Mat3, inv_cell: math.Mat3, pbc: neighbor_list.Pbc, delta: math.Vec3) math.Vec3 {
    var frac = inv_cell.mulVec(delta);
    if (pbc.x) frac.x = wrapFrac(frac.x);
    if (pbc.y) frac.y = wrapFrac(frac.y);
    if (pbc.z) frac.z = wrapFrac(frac.z);
    return cell.mulVec(frac);
}

test "ionic potential constant matches upf values" {
    const alloc = std.testing.allocator;
    var r = try alloc.alloc(f64, 2);
    defer alloc.free(r);
    r[0] = 0.0;
    r[1] = 1.0;
    var rab = try alloc.alloc(f64, 2);
    defer alloc.free(rab);
    rab[0] = 0.0;
    rab[1] = 1.0;
    var v_local = try alloc.alloc(f64, 2);
    defer alloc.free(v_local);
    v_local[0] = 1.2;
    v_local[1] = 1.2;
    const upf = pseudo.UpfData{
        .r = r,
        .rab = rab,
        .v_local = v_local,
        .beta = &[_]pseudo.Beta{},
        .dij = &[_]f64{},
        .qij = &[_]f64{},
        .nlcc = &[_]f64{},
    };

    const cell = math.Mat3.fromRows(
        .{ .x = 4.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 4.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 4.0 },
    );
    const dims = [3]usize{ 4, 4, 4 };
    const grid = local_orbital_potential.PotentialGrid{ .cell = cell, .dims = dims, .values = &[_]f64{} };
    const sites = [_]IonSite{.{ .position = .{ .x = 2.0, .y = 2.0, .z = 2.0 }, .upf = &upf }};
    const pbc = neighbor_list.Pbc{ .x = false, .y = false, .z = false };
    const values = try buildIonicPotentialGrid(alloc, grid, sites[0..], pbc);
    defer alloc.free(values);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2), values[0], 1e-12);
}
