const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const target_os = target.result.os.tag;

    // FFTW3 options (default to environment variables when not provided)
    const fftw_include_opt = b.option([]const u8, "fftw-include", "Path to FFTW3 include directory");
    const fftw_lib_opt = b.option([]const u8, "fftw-lib", "Path to FFTW3 library directory");
    const fftw_include = fftw_include_opt orelse b.graph.environ_map.get("FFTW_INCLUDE");
    const fftw_lib = fftw_lib_opt orelse b.graph.environ_map.get("FFTW_LIB");
    const enable_fftw = fftw_include != null and fftw_lib != null;

    // Create FFTW options module (used by fftw_fft.zig for conditional compilation)
    const fftw_options = b.addOptions();
    fftw_options.addOption(bool, "enable_fftw", enable_fftw);

    // libcint options (default to environment variables when not provided)
    const libcint_include_opt = b.option([]const u8, "libcint-include", "Path to libcint include directory");
    const libcint_lib_opt = b.option([]const u8, "libcint-lib", "Path to libcint library directory");
    const libcint_include = libcint_include_opt orelse b.graph.environ_map.get("LIBCINT_INCLUDE");
    const libcint_lib = libcint_lib_opt orelse b.graph.environ_map.get("LIBCINT_LIB");
    const enable_libcint = libcint_include != null and libcint_lib != null;

    // Create libcint options module (used by libcint.zig for conditional compilation)
    const libcint_options = b.addOptions();
    libcint_options.addOption(bool, "enable_libcint", enable_libcint);

    // OpenBLAS options (default to environment variables when not provided)
    const openblas_include_opt = b.option([]const u8, "openblas-include", "Path to OpenBLAS include directory");
    const openblas_lib_opt = b.option([]const u8, "openblas-lib", "Path to OpenBLAS library directory");
    const openblas_include = openblas_include_opt orelse b.graph.environ_map.get("OPENBLAS_INCLUDE");
    const openblas_lib = openblas_lib_opt orelse b.graph.environ_map.get("OPENBLAS_LIB");

    // Netlib LAPACK (optional, for Linux where OpenBLAS LAPACK may have issues)
    const lapack_lib_opt = b.option([]const u8, "lapack-lib", "Path to LAPACK library directory");
    const lapack_lib = lapack_lib_opt orelse b.graph.environ_map.get("LAPACK_LIB");

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    const mod = b.addModule("dft_zig", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "fftw_options", .module = fftw_options.createModule() },
            .{ .name = "libcint_options", .module = libcint_options.createModule() },
        },
    });

    // Add Metal GPU bridge (Objective-C source) on macOS
    if (target_os == .macos) {
        mod.addCSourceFile(.{
            .file = b.path("src/lib/gpu/metal_bridge.m"),
            .flags = &.{"-fobjc-arc"},
        });
        mod.addIncludePath(b.path("src/lib/gpu"));
    }

    const exe = b.addExecutable(.{
        .name = "dft_zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dft_zig", .module = mod },
            },
        }),
    });

    linkLinearAlgebra(exe, target_os, openblas_include, openblas_lib, lapack_lib);

    // Link FFTW3 if paths are provided
    if (fftw_include) |inc| {
        exe.root_module.addIncludePath(.{ .cwd_relative = inc });
        mod.addIncludePath(.{ .cwd_relative = inc });
    }
    if (fftw_lib) |lib| {
        exe.root_module.addLibraryPath(.{ .cwd_relative = lib });
        exe.root_module.linkSystemLibrary("fftw3", .{});
    }

    // Link libcint if paths are provided
    if (libcint_include) |inc| {
        exe.root_module.addIncludePath(.{ .cwd_relative = inc });
        mod.addIncludePath(.{ .cwd_relative = inc });
    }
    if (libcint_lib) |lib| {
        exe.root_module.addLibraryPath(.{ .cwd_relative = lib });
        exe.root_module.linkSystemLibrary("cint", .{});
    }

    b.installArtifact(exe);

    // Linear scaling SCF example executable
    const linear_scaling_exe = b.addExecutable(.{
        .name = "linear_scaling_scf",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/linear_scaling_scf.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dft_zig", .module = mod },
            },
        }),
    });
    linkLinearAlgebra(linear_scaling_exe, target_os, openblas_include, openblas_lib, lapack_lib);
    b.installArtifact(linear_scaling_exe);

    const run_linear_scaling = b.addRunArtifact(linear_scaling_exe);
    run_linear_scaling.step.dependOn(b.getInstallStep());
    const linear_scaling_step = b.step("run-linear-scaling", "Run linear scaling SCF example");
    linear_scaling_step.dependOn(&run_linear_scaling.step);

    // FFT benchmark executable
    const fft_lib_mod = b.addModule("fft_lib", .{
        .root_source_file = b.path("src/lib/fft/fft.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "fftw_options", .module = fftw_options.createModule() },
            .{ .name = "libcint_options", .module = libcint_options.createModule() },
        },
    });
    if (fftw_include) |inc| {
        fft_lib_mod.addIncludePath(.{ .cwd_relative = inc });
    }
    // Add Metal GPU bridge to FFT library module on macOS
    if (target_os == .macos) {
        fft_lib_mod.addCSourceFile(.{
            .file = b.path("src/lib/gpu/metal_bridge.m"),
            .flags = &.{"-fobjc-arc"},
        });
        fft_lib_mod.addIncludePath(b.path("src/lib/gpu"));
    }
    // Comprehensive FFT all-backends benchmark executable
    const fft_all_bench_exe = b.addExecutable(.{
        .name = "bench_fft_all",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/fft/bench_all_backends.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fft_lib", .module = fft_lib_mod },
            },
        }),
    });
    linkLinearAlgebra(fft_all_bench_exe, target_os, openblas_include, openblas_lib, lapack_lib);
    b.installArtifact(fft_all_bench_exe);

    const run_fft_all_bench = b.addRunArtifact(fft_all_bench_exe);
    run_fft_all_bench.step.dependOn(b.getInstallStep());
    const fft_all_bench_step = b.step("bench-fft-all", "Run comprehensive FFT backend benchmark");
    fft_all_bench_step.dependOn(&run_fft_all_bench.step);

    // Rys ERI test executable
    const rys_test_exe = b.addExecutable(.{
        .name = "rys_eri_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/rys_eri_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dft_zig", .module = mod },
            },
        }),
    });
    linkLinearAlgebra(rys_test_exe, target_os, openblas_include, openblas_lib, lapack_lib);
    b.installArtifact(rys_test_exe);

    const run_rys_test = b.addRunArtifact(rys_test_exe);
    run_rys_test.step.dependOn(b.getInstallStep());
    const rys_test_step = b.step("run-rys-test", "Run Rys quadrature ERI validation tests");
    rys_test_step.dependOn(&run_rys_test.step);

    // Gradient validation executable (B3LYP/6-31G(2df,p) gradient vs PySCF)
    const grad_test_exe = b.addExecutable(.{
        .name = "gradient_validation",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/gradient_validation.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dft_zig", .module = mod },
            },
        }),
    });
    linkLinearAlgebra(grad_test_exe, target_os, openblas_include, openblas_lib, lapack_lib);
    b.installArtifact(grad_test_exe);

    const run_grad_test = b.addRunArtifact(grad_test_exe);
    run_grad_test.step.dependOn(b.getInstallStep());
    const grad_test_step = b.step("run-gradient-test", "Run gradient validation (B3LYP/6-31G(2df,p) vs PySCF)");
    grad_test_step.dependOn(&run_grad_test.step);

    // Density Fitting benchmark executable
    const df_bench_exe = b.addExecutable(.{
        .name = "df_benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/df_benchmark.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dft_zig", .module = mod },
            },
        }),
    });
    linkLinearAlgebra(df_bench_exe, target_os, openblas_include, openblas_lib, lapack_lib);
    b.installArtifact(df_bench_exe);

    const run_df_bench = b.addRunArtifact(df_bench_exe);
    run_df_bench.step.dependOn(b.getInstallStep());
    const df_bench_step = b.step("run-df-bench", "Run Density Fitting benchmark (conventional vs RI-J/K)");
    df_bench_step.dependOn(&run_df_bench.step);

    // Linear algebra benchmark executable
    const linalg_bench_exe = b.addExecutable(.{
        .name = "bench_linalg",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib/linalg/benchmark.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(linalg_bench_exe);

    const run_linalg_bench = b.addRunArtifact(linalg_bench_exe);
    run_linalg_bench.step.dependOn(b.getInstallStep());
    const linalg_bench_step = b.step("bench-linalg", "Run linear algebra SIMD benchmark");
    linalg_bench_step.dependOn(&run_linalg_bench.step);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Test filter option: zig build test -Dtest-filter="pattern"
    const test_filter = b.option([]const u8, "test-filter", "Filter tests by name");
    const test_filters: []const []const u8 = if (test_filter) |f| &.{f} else &.{};

    const mod_tests = b.addTest(.{
        .root_module = mod,
        .filters = test_filters,
    });
    linkLinearAlgebra(mod_tests, target_os, openblas_include, openblas_lib, lapack_lib);

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
        .filters = test_filters,
    });
    linkLinearAlgebra(exe_tests, target_os, openblas_include, openblas_lib, lapack_lib);

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Link FFTW3 to all executables that use the dft_zig module
    if (fftw_lib) |lib| {
        const fftw_lib_path = std.Build.LazyPath{ .cwd_relative = lib };

        linear_scaling_exe.root_module.addLibraryPath(fftw_lib_path);
        linear_scaling_exe.root_module.linkSystemLibrary("fftw3", .{});

        fft_all_bench_exe.root_module.addLibraryPath(fftw_lib_path);
        fft_all_bench_exe.root_module.linkSystemLibrary("fftw3", .{});

        df_bench_exe.root_module.addLibraryPath(fftw_lib_path);
        df_bench_exe.root_module.linkSystemLibrary("fftw3", .{});

        mod_tests.root_module.addLibraryPath(fftw_lib_path);
        mod_tests.root_module.linkSystemLibrary("fftw3", .{});

        exe_tests.root_module.addLibraryPath(fftw_lib_path);
        exe_tests.root_module.linkSystemLibrary("fftw3", .{});
    }

    // Link libcint to all executables that use the dft_zig module
    if (libcint_lib) |lib| {
        const libcint_lib_path = std.Build.LazyPath{ .cwd_relative = lib };

        linear_scaling_exe.root_module.addLibraryPath(libcint_lib_path);
        linear_scaling_exe.root_module.linkSystemLibrary("cint", .{});

        df_bench_exe.root_module.addLibraryPath(libcint_lib_path);
        df_bench_exe.root_module.linkSystemLibrary("cint", .{});

        rys_test_exe.root_module.addLibraryPath(libcint_lib_path);
        rys_test_exe.root_module.linkSystemLibrary("cint", .{});

        grad_test_exe.root_module.addLibraryPath(libcint_lib_path);
        grad_test_exe.root_module.linkSystemLibrary("cint", .{});

        mod_tests.root_module.addLibraryPath(libcint_lib_path);
        mod_tests.root_module.linkSystemLibrary("cint", .{});

        exe_tests.root_module.addLibraryPath(libcint_lib_path);
        exe_tests.root_module.linkSystemLibrary("cint", .{});
    }
}

/// Link linear algebra libraries (Accelerate on macOS, OpenBLAS+LAPACK on other platforms).
fn linkLinearAlgebra(
    step: *std.Build.Step.Compile,
    target_os: std.Target.Os.Tag,
    openblas_include: ?[]const u8,
    openblas_lib: ?[]const u8,
    lapack_lib_path: ?[]const u8,
) void {
    if (target_os == .macos) {
        step.root_module.linkFramework("Accelerate", .{});
        step.root_module.linkFramework("Metal", .{});
        step.root_module.linkFramework("MetalPerformanceShaders", .{});
        step.root_module.linkFramework("Foundation", .{});
    } else {
        // Link Netlib LAPACK first (takes priority for LAPACK symbols like dsygv_)
        if (lapack_lib_path) |lib| {
            step.root_module.addLibraryPath(.{ .cwd_relative = lib });
            step.root_module.linkSystemLibrary("lapack", .{});
        }
        // Then OpenBLAS for BLAS symbols
        if (openblas_lib) |lib| {
            step.root_module.addLibraryPath(.{ .cwd_relative = lib });
            step.root_module.linkSystemLibrary("openblas", .{});
        }
    }
    if (openblas_include) |inc| {
        step.root_module.addIncludePath(.{ .cwd_relative = inc });
    }
    step.root_module.link_libc = true;
}
