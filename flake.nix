{
  description = "DFT-Zig: Density Functional Theory implementation in Zig";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        zig_0_16 = let
          zigBin = {
            "aarch64-darwin" = {
              url = "https://ziglang.org/download/0.16.0/zig-aarch64-macos-0.16.0.tar.xz";
              hash = "0yqiq1nrjfawh1k24mf969q1w9bhwfbwqi2x8f9zklca7bsyza26";
            };
            "x86_64-darwin" = {
              url = "https://ziglang.org/download/0.16.0/zig-x86_64-macos-0.16.0.tar.xz";
              hash = "0dibmghlqrr8qi5cqs9n0nl25qdnb5jvr542dyljfqdyy2bzzh2x";
            };
            "x86_64-linux" = {
              url = "https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz";
              hash = "1kgamnyy7vsw5alb5r4xk8nmgvmgbmxkza5hs7b51x6dbgags1h6";
            };
            "aarch64-linux" = {
              url = "https://ziglang.org/download/0.16.0/zig-aarch64-linux-0.16.0.tar.xz";
              hash = "12gf4d1rjncc8r4i32sfdmnwdl0d6hg717hb3801zxjlmzmpsns0";
            };
          }.${system};
          src = builtins.fetchTarball {
            url = zigBin.url;
            sha256 = zigBin.hash;
          };
        in nixpkgs.legacyPackages.${system}.stdenv.mkDerivation {
          pname = "zig";
          version = "0.16.0";
          inherit src;
          dontBuild = true;
          installPhase = ''
            mkdir -p $out
            cp -r ./* $out/
            mkdir -p $out/bin
            ln -sf $out/zig $out/bin/zig
          '';
        };

        pkgs = nixpkgs.legacyPackages.${system};

        # FFTW3 library (double precision)
        fftw = pkgs.fftw;

        # libcint: general GTO integrals for quantum chemistry
        libcint = pkgs.libcint;

        # OpenBLAS for BLAS, Netlib LAPACK for LAPACK routines (dsygv etc.)
        # OpenBLAS 0.3.30 has LAPACK issues (dsygv SIGABRT in workspace query)
        openblas = pkgs.openblas;
        lapack = pkgs.lapack-reference;

        # Python environment for plotting and analysis
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy
          matplotlib
          scipy
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            zig_0_16
            fftw
            libcint
            pythonEnv
            pkgs.ffmpeg
            pkgs.just
            pkgs.git
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.apple-sdk_15
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            openblas
            lapack
          ];

          shellHook = ''
            export FFTW_INCLUDE="${fftw.dev}/include"
            export FFTW_LIB="${fftw}/lib"
            export LIBCINT_INCLUDE="${libcint}/include"
            export LIBCINT_LIB="${libcint}/lib"
          '' + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export OPENBLAS_INCLUDE="${openblas.dev}/include"
            export OPENBLAS_LIB="${openblas}/lib"
            export LAPACK_LIB="${lapack}/lib"
          '' + ''
            # Ensure Zig can write to cache
            export ZIG_GLOBAL_CACHE_DIR="$HOME/.cache/zig"
            # Clear NIX_CFLAGS to avoid confusing Zig's build system
            unset NIX_CFLAGS_COMPILE
            unset NIX_LDFLAGS
            echo "DFT-Zig development environment"
            echo "  Zig: $(zig version)"
            echo "  FFTW: ${fftw.version}"
            echo "  libcint: ${libcint.version}"
            echo ""
            echo "Build commands:"
            echo "  just build   - Build with FFTW (uses \$FFTW_INCLUDE, \$FFTW_LIB)"
            echo "  just test         - Run all tests"
          '';
        };

        # Package for installation
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "dft-zig";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = [ zig_0_16 ];
          buildInputs = [ fftw libcint ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.apple-sdk_15
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            openblas
          ];

          buildPhase = ''
            export HOME=$TMPDIR
            zig build -Doptimize=ReleaseFast \
              -Dfftw-include=${fftw.dev}/include \
              -Dfftw-lib=${fftw}/lib \
              -Dlibcint-include=${libcint}/include \
              -Dlibcint-lib=${libcint}/lib \
          '' + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              -Dopenblas-include=${openblas.dev}/include \
              -Dopenblas-lib=${openblas}/lib \
          '' + ''
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp zig-out/bin/dft_zig $out/bin/
          '';
        };
      }
    );
}
