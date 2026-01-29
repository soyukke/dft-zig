const std = @import("std");

const dft_zig = @import("dft_zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const linear_scaling = dft_zig.linear_scaling;
    const math = dft_zig.math;
    const pseudo = dft_zig.pseudopotential;
    const local_orbital = linear_scaling.local_orbital;

    const specs = [_]pseudo.Spec{
        .{ .element = "Si", .path = "pseudo/Si.upf", .format = .upf },
    };

    const cell = math.Mat3.fromRows(
        .{ .x = 5.431, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 5.431, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 5.431 },
    );
    const pbc = linear_scaling.Pbc{ .x = true, .y = true, .z = true };

    // Parse command line args for basis type
    var args = std.process.args();
    _ = args.skip(); // skip program name
    const basis_arg = args.next();
    const use_sp = if (basis_arg) |arg| std.mem.eql(u8, arg, "sp") else false;

    const opts = linear_scaling.ScfRunOptions{
        .units = .angstrom,
        .grid_dims = [3]usize{ 16, 16, 16 },
        .sigma = 0.5,
        .cutoff = 3.0,
        .max_iter = 20,
        .density_tol = 1e-4,
        .electrons = 32.0,
        .xc = .lda_pz,
        .kinetic_scale = 1.0,
        .matrix_threshold = 0.0,
        .purification_iters = 5,
        .purification_threshold = 0.0,
        .nonlocal_basis = if (use_sp) local_orbital.BasisType.sp else local_orbital.BasisType.s_only,
    };

    var result = try linear_scaling.runScfFromXyz(
        alloc,
        "examples/silicon.xyz",
        specs[0..],
        cell,
        pbc,
        opts,
    );
    defer result.deinit(alloc);

    const basis_name = if (use_sp) "sp" else "s_only";
    std.debug.print("linear-scaling scf (basis={s}) converged={any} iterations={d}\n", .{ basis_name, result.converged, result.iterations });
    std.debug.print(
        "energy_total={d:.6} hartree={d:.6} xc={d:.6} vxc_rho={d:.6} nonlocal={d:.6}\n",
        .{ result.energy, result.energy_hartree, result.energy_xc, result.energy_vxc_rho, result.energy_nonlocal },
    );
}
