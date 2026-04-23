//! Q-path generation for phonon band structure.
//!
//! Builds interpolated fractional/Cartesian q-points along high-symmetry
//! lines, plus cumulative distances and per-segment labels.

const std = @import("std");
const math = @import("../../math/math.zig");
const config_mod = @import("../../config/config.zig");

pub const GeneratedQPath = struct {
    q_points_frac: []math.Vec3,
    q_points_cart: []math.Vec3,
    distances: []f64,
    labels: [][]const u8,
    label_positions: []usize,
};

/// Generate FCC q-path: Γ-X-W-K-Γ-L
pub fn generate_fcc_q_path(
    alloc: std.mem.Allocator,
    recip: math.Mat3,
    npoints_per_seg: usize,
) !GeneratedQPath {
    // FCC high-symmetry points in fractional (reduced) coordinates
    // Path: Γ-X-W-K-Γ-L (same as ABINIT anaddb)
    const points = [_]math.Vec3{
        math.Vec3{ .x = 0.000, .y = 0.000, .z = 0.000 }, // Γ
        math.Vec3{ .x = 0.500, .y = 0.000, .z = 0.500 }, // X
        math.Vec3{ .x = 0.500, .y = 0.250, .z = 0.750 }, // W
        math.Vec3{ .x = 0.375, .y = 0.375, .z = 0.750 }, // K
        math.Vec3{ .x = 0.000, .y = 0.000, .z = 0.000 }, // Γ
        math.Vec3{ .x = 0.500, .y = 0.500, .z = 0.500 }, // L
    };
    const labels = [_][]const u8{ "G", "X", "W", "K", "G", "L" };
    const n_segs = points.len - 1;
    const n_total = n_segs * npoints_per_seg + 1;

    var q_frac = try alloc.alloc(math.Vec3, n_total);
    errdefer alloc.free(q_frac);
    var q_cart = try alloc.alloc(math.Vec3, n_total);
    errdefer alloc.free(q_cart);
    var dists = try alloc.alloc(f64, n_total);
    errdefer alloc.free(dists);
    var label_list = try alloc.alloc([]const u8, points.len);
    errdefer alloc.free(label_list);
    var label_pos = try alloc.alloc(usize, points.len);
    errdefer alloc.free(label_pos);

    for (0..points.len) |i| {
        label_list[i] = labels[i];
        label_pos[i] = if (i == 0) 0 else i * npoints_per_seg;
    }

    var idx: usize = 0;
    var cum_dist: f64 = 0.0;
    for (0..n_segs) |seg| {
        const p0 = points[seg];
        const p1 = points[seg + 1];
        for (0..npoints_per_seg) |ip| {
            const t = @as(f64, @floatFromInt(ip)) / @as(f64, @floatFromInt(npoints_per_seg));
            const qf = math.Vec3{
                .x = p0.x + t * (p1.x - p0.x),
                .y = p0.y + t * (p1.y - p0.y),
                .z = p0.z + t * (p1.z - p0.z),
            };
            q_frac[idx] = qf;
            // Convert to Cartesian: q_cart = qf.x * b1 + qf.y * b2 + qf.z * b3
            q_cart[idx] = math.Vec3.add(
                math.Vec3.add(
                    math.Vec3.scale(recip.row(0), qf.x),
                    math.Vec3.scale(recip.row(1), qf.y),
                ),
                math.Vec3.scale(recip.row(2), qf.z),
            );
            if (idx == 0) {
                dists[idx] = 0.0;
            } else {
                const dq = math.Vec3.sub(q_cart[idx], q_cart[idx - 1]);
                cum_dist += math.Vec3.norm(dq);
                dists[idx] = cum_dist;
            }
            idx += 1;
        }
    }
    // Last point
    q_frac[idx] = points[n_segs];
    q_cart[idx] = math.Vec3.add(
        math.Vec3.add(
            math.Vec3.scale(recip.row(0), points[n_segs].x),
            math.Vec3.scale(recip.row(1), points[n_segs].y),
        ),
        math.Vec3.scale(recip.row(2), points[n_segs].z),
    );
    if (idx > 0) {
        const dq = math.Vec3.sub(q_cart[idx], q_cart[idx - 1]);
        cum_dist += math.Vec3.norm(dq);
    }
    dists[idx] = cum_dist;

    return .{
        .q_points_frac = q_frac,
        .q_points_cart = q_cart,
        .distances = dists,
        .labels = label_list,
        .label_positions = label_pos,
    };
}

/// Generate q-path from config-specified high-symmetry points.
/// Same format as generate_fcc_q_path but with user-supplied points.
pub fn generate_q_path_from_config(
    alloc: std.mem.Allocator,
    qpath_points: []const config_mod.BandPathPoint,
    npoints_per_seg: usize,
    recip: math.Mat3,
) !GeneratedQPath {
    const n_pts = qpath_points.len;
    if (n_pts < 2) return error.InvalidQPath;

    const n_segs = n_pts - 1;
    const n_total = n_segs * npoints_per_seg + 1;

    var q_frac = try alloc.alloc(math.Vec3, n_total);
    errdefer alloc.free(q_frac);
    var q_cart = try alloc.alloc(math.Vec3, n_total);
    errdefer alloc.free(q_cart);
    var dists = try alloc.alloc(f64, n_total);
    errdefer alloc.free(dists);
    var label_list = try alloc.alloc([]const u8, n_pts);
    errdefer alloc.free(label_list);
    var label_pos = try alloc.alloc(usize, n_pts);
    errdefer alloc.free(label_pos);

    for (0..n_pts) |i| {
        label_list[i] = qpath_points[i].label;
        label_pos[i] = if (i == 0) 0 else i * npoints_per_seg;
    }

    var idx: usize = 0;
    var cum_dist: f64 = 0.0;
    for (0..n_segs) |seg| {
        const p0 = qpath_points[seg].k;
        const p1 = qpath_points[seg + 1].k;
        for (0..npoints_per_seg) |ip| {
            const t = @as(f64, @floatFromInt(ip)) / @as(f64, @floatFromInt(npoints_per_seg));
            const qf = math.Vec3{
                .x = p0.x + t * (p1.x - p0.x),
                .y = p0.y + t * (p1.y - p0.y),
                .z = p0.z + t * (p1.z - p0.z),
            };
            q_frac[idx] = qf;
            q_cart[idx] = math.Vec3.add(
                math.Vec3.add(
                    math.Vec3.scale(recip.row(0), qf.x),
                    math.Vec3.scale(recip.row(1), qf.y),
                ),
                math.Vec3.scale(recip.row(2), qf.z),
            );
            if (idx == 0) {
                dists[idx] = 0.0;
            } else {
                const dq = math.Vec3.sub(q_cart[idx], q_cart[idx - 1]);
                cum_dist += math.Vec3.norm(dq);
                dists[idx] = cum_dist;
            }
            idx += 1;
        }
    }
    // Last point
    const last_pt = qpath_points[n_segs].k;
    q_frac[idx] = last_pt;
    q_cart[idx] = math.Vec3.add(
        math.Vec3.add(
            math.Vec3.scale(recip.row(0), last_pt.x),
            math.Vec3.scale(recip.row(1), last_pt.y),
        ),
        math.Vec3.scale(recip.row(2), last_pt.z),
    );
    if (idx > 0) {
        const dq = math.Vec3.sub(q_cart[idx], q_cart[idx - 1]);
        cum_dist += math.Vec3.norm(dq);
    }
    dists[idx] = cum_dist;

    return .{
        .q_points_frac = q_frac,
        .q_points_cart = q_cart,
        .distances = dists,
        .labels = label_list,
        .label_positions = label_pos,
    };
}
