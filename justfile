# DFT-Zig Justfile
# Usage: just <recipe>
#
# For FFTW support, use `nix develop` to enter the development environment,
# which sets FFTW_INCLUDE and FFTW_LIB automatically.

# FFTW3 paths from environment variables (set by nix develop)
fftw_include := env("FFTW_INCLUDE", "")
fftw_lib := env("FFTW_LIB", "")

# Default recipe: show available commands
default:
    @just --list

# Build DFT-Zig with FFTW3 (ReleaseFast)
# Requires FFTW_INCLUDE and FFTW_LIB environment variables (use `nix develop`)
build:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "${FFTW_INCLUDE:-}" ] || [ -z "${FFTW_LIB:-}" ]; then
        echo "Error: FFTW_INCLUDE and FFTW_LIB must be set."
        echo "Run 'nix develop' to enter the development environment."
        exit 1
    fi
    zig build -Doptimize=ReleaseFast -Dfftw-include="$FFTW_INCLUDE" -Dfftw-lib="$FFTW_LIB"

# Run the default test suite (fast unit tests + regression tests)
test: test-unit test-regression

# Run fast unit tests
test-unit:
    zig build test

# Run fast unit tests plus the day-to-day GTO integration suite
test-unit-full:
    zig build test-full

# Run all Zig tests, including the slower GTO regression suite
test-unit-all:
    zig build test-all-zig

# Run the day-to-day GTO integration suite only
test-gto:
    zig build test-gto

# Run the slower GTO regression suite only
test-gto-regression:
    zig build test-gto-regression

# Run fast unit tests in ReleaseFast mode
test-unit-release-fast:
    zig build test -Doptimize=ReleaseFast

# Run the slower GTO integration suite in ReleaseFast mode
test-gto-release-fast:
    zig build test-gto -Doptimize=ReleaseFast

# Run the slower GTO regression suite in ReleaseFast mode
test-gto-regression-release-fast:
    zig build test-gto-regression -Doptimize=ReleaseFast

# Run the day-to-day full Zig test suite in ReleaseFast mode
test-unit-full-release-fast:
    zig build test-full -Doptimize=ReleaseFast

# Run all Zig tests in ReleaseFast mode
test-unit-all-release-fast:
    zig build test-all-zig -Doptimize=ReleaseFast

# Run fast unit tests with FFTW3
test-unit-fftw:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "${FFTW_INCLUDE:-}" ] || [ -z "${FFTW_LIB:-}" ]; then
        echo "Error: FFTW_INCLUDE and FFTW_LIB must be set."
        echo "Run 'nix develop' to enter the development environment."
        exit 1
    fi
    zig build test -Dfftw-include="$FFTW_INCLUDE" -Dfftw-lib="$FFTW_LIB"

# Run all regression tests (silicon + graphene + gaas + cu + fe + k-parallel + lobpcg-parallel + eos + dfpt + molecule + paw)
test-regression: test-silicon test-graphene test-gaas test-cu test-fe test-aluminum test-aluminum-11e test-silicon-kparallel test-lobpcg-parallel test-eos test-dfpt test-molecule test-paw
    @echo "All regression tests passed!"

# Run the full project suite (all Zig tests + regression tests)
test-full: test-unit-all test-regression

# Run the default suite with ReleaseFast binaries
test-release-fast: test-unit-release-fast test-regression

# Run the full project suite with ReleaseFast binaries
test-full-release-fast: test-unit-all-release-fast test-regression

# Run silicon regression test (uses saved ABINIT baseline)
test-silicon: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/silicon
    source ../../.venv/bin/activate
    echo "Running Silicon regression test..."
    python3 plot_comparison.py --run-check
    echo "Silicon test passed!"

# Run graphene regression test (uses saved ABINIT baseline)
test-graphene: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/graphene
    source ../../.venv/bin/activate
    echo "Running Graphene regression test..."
    python3 plot_comparison.py --run-check
    echo "Graphene test passed!"

# Run GaAs regression test (uses saved ABINIT baseline)
test-gaas: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/gaas
    source ../../.venv/bin/activate
    echo "Running GaAs regression test..."
    python3 plot_comparison.py --run-check
    echo "GaAs test passed!"

# Run Cu regression test (uses saved ABINIT baseline)
test-cu: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/cu
    source ../../.venv/bin/activate
    echo "Running Cu regression test..."
    python3 plot_comparison.py --run-check
    echo "Cu test passed!"

# Run Fe BCC regression test (spin-polarized, uses saved ABINIT baseline)
test-fe: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/fe
    source ../../.venv/bin/activate
    echo "Running Fe BCC regression test..."
    python3 plot_comparison.py --run-check
    echo "Fe BCC test passed!"

# Run Aluminum regression test (uses saved ABINIT baseline)
test-aluminum: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/aluminum
    source ../../.venv/bin/activate
    echo "Running Aluminum regression test..."
    python3 plot_comparison.py --run-check
    echo "Aluminum test passed!"

# Run Aluminum 11e regression test (semi-core, uses saved ABINIT baseline)
test-aluminum-11e: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/aluminum_11e
    source ../../.venv/bin/activate
    echo "Running Aluminum 11e regression test..."
    python3 plot_comparison.py --run-check
    echo "Aluminum 11e test passed!"

# Run silicon k-parallel regression test (compare parallel vs serial)
test-silicon-kparallel: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/silicon
    source ../../.venv/bin/activate
    echo "Running Silicon k-parallel regression test..."
    python3 test_kparallel.py --run
    echo "Silicon k-parallel test passed!"

# Run LOBPCG parallel regression test (compare parallel vs serial LOBPCG)
test-lobpcg-parallel: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/silicon
    source ../../.venv/bin/activate
    echo "Running LOBPCG parallel regression test..."
    python3 test_lobpcg_parallel.py --run
    echo "LOBPCG parallel test passed!"

# Run silicon EOS regression test (total energy vs ABINIT at 3 volumes)
test-eos: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/silicon_eos
    source ../../.venv/bin/activate
    echo "Running Silicon EOS regression test..."
    python3 test_eos.py --run
    echo "Silicon EOS test passed!"

# Run DFPT phonon regression test (phonon frequencies vs ABINIT at Gamma, X, L)
test-dfpt: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/silicon_dfpt
    source ../../.venv/bin/activate
    echo "Running DFPT phonon regression test..."
    python3 test_dfpt.py --run
    echo "DFPT phonon test passed!"

# Run silicon PAW regression test (band structure vs saved baseline)
test-paw: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/silicon_paw
    source ../../.venv/bin/activate
    echo "Running Silicon PAW regression test..."
    python3 test_paw.py --run
    echo "Silicon PAW test passed!"

# Run molecule SCF regression test (conventional + DF vs PySCF baseline)
test-molecule: build
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/molecule
    source ../../.venv/bin/activate
    echo "Running molecule regression test..."
    python3 test_molecule.py --run
    echo "Molecule test passed!"

# Plot silicon bands (without running calculations)
plot-silicon:
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/silicon
    source ../../.venv/bin/activate
    python3 plot_comparison.py

# Plot graphene bands (without running calculations)
plot-graphene:
    #!/usr/bin/env bash
    set -euo pipefail
    cd benchmarks/graphene
    source ../../.venv/bin/activate
    python3 plot_comparison.py

# Format all Zig source files
fmt:
    zig fmt src scripts

# Path to the shared style checker in ~/dotfiles/zig-tools. Override with
# ZIG_STYLE_CHECKER=/some/other/check_style.zig if needed.
style_checker := env("ZIG_STYLE_CHECKER", env("HOME") + "/dotfiles/zig-tools/check_style.zig")

# Check formatting without rewriting (reports diffs; not part of `lint`).
fmt-check:
    zig fmt --check src scripts

# Run the style checker against the ratcheting baseline.
# Fails only when a file has *more* violations than scripts/style_baseline.txt.
lint:
    zig run {{style_checker}} -- --root src

# Report every violation, ignoring the baseline (fails on any).
lint-strict:
    zig run {{style_checker}} -- --root src --strict

# Regenerate the style baseline from current violations (run after cleanup).
lint-update-baseline:
    zig run {{style_checker}} -- --root src --update-baseline

# Clean build artifacts and caches
clean:
    rm -rf zig-out zig-cache .zig-cache
    rm -f stderr.log stderr_paw.log stdout.log

# Run DFT-Zig with a config file
run config:
    ./zig-out/bin/dft_zig {{config}}
