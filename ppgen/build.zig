const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const target_os = target.result.os.tag;

    // dft_zig dependency (reuse XC, math, sphericalBessel)
    const dft_zig_dep = b.dependency("dft_zig", .{
        .target = target,
        .optimize = optimize,
    });

    // ppgen executable
    const exe = b.addExecutable(.{
        .name = "ppgen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dft_zig", .module = dft_zig_dep.module("dft_zig") },
            },
        }),
    });
    linkLinearAlgebra(exe, target_os);
    b.installArtifact(exe);

    // run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run ppgen");
    run_step.dependOn(&run_cmd.step);

    // test step
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dft_zig", .module = dft_zig_dep.module("dft_zig") },
            },
        }),
    });
    linkLinearAlgebra(tests, target_os);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run ppgen tests");
    test_step.dependOn(&run_tests.step);
}

fn linkLinearAlgebra(c: *std.Build.Step.Compile, os: std.Target.Os.Tag) void {
    if (os == .macos) {
        c.linkFramework("Accelerate");
    } else {
        c.linkSystemLibrary("lapack");
        c.linkSystemLibrary("blas");
    }
    c.linkLibC();
}
