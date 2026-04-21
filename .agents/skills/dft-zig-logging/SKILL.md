---
name: dft-zig-logging
description: Use when working on logging in the dft-zig repo: defining log policy, designing new runtime logs, reducing noisy runtime output, adding or changing log levels and sinks, replacing raw std.debug.print or stderr writes, or refactoring SCF/DFPT/Band/Relax/FFT logging toward the shared runtime logger.
---

# DFT Zig Logging

## Overview

Use this skill when changing logging behavior in `dft-zig`. The goal is to keep event emission close to the feature code, while centralizing log policy, level handling, and sink decisions.

## When To Use

- The user asks to clean up, redesign, or standardize logging in `dft-zig`
- Runtime output is too noisy or the wrong messages appear on `stderr`
- A feature adds new SCF, DFPT, band, relax, k-point, force, or FFT diagnostics
- The code uses raw `std.debug.print` or `std.Io.File.stderr().writer(...)` in production paths

## Workflow

1. Inventory existing outputs before editing.
   Use the search commands in [references/policy.md](references/policy.md) to find raw prints, feature helpers, and config gates.
2. Classify each message before moving it.
   Decide whether it is `err`, `warn`, `info`, `debug`, `profile`, file artifact, or test-only output.
3. Route policy through shared logging.
   Prefer `src/features/runtime/logging.zig` for levels and sink behavior, and thin feature-specific helpers for domain semantics.
4. Keep console output small.
   Normal runs should show start, convergence, summary, and real warnings. Heavy diagnostics belong behind `debug` or in output files.
5. Verify behavior, not just compilation.
   Run `zig fmt`, `zig build`, and `zig build test --summary all`. If the change is user-visible, inspect representative runtime output too.

## Adding New Logs

When introducing a new log message, decide these points before writing code:

1. What kind of signal is this?
   Is it a failure, a warning, normal progress, detailed diagnosis, profile data, or an artifact that should be written to a file?
2. Who needs it during a normal run?
   If most users do not need to see it every run, it should probably be `debug` or a file artifact.
3. Where should the event live?
   Emit the event near the feature logic that knows the context.
4. Where should policy live?
   Route level and sink handling through `src/features/runtime/logging.zig` and feature helpers.
5. Does it need rate limiting?
   Repeated per-k-point or per-iteration logs should usually be sampled or gated.

For new logs, prefer this pattern:

- add or extend a semantic feature helper
- pick the narrowest useful level
- keep console text short and stable
- move large numeric dumps to files

## Required Rules

- Do not add raw `std.debug.print` in production runtime paths.
- Do not add raw `stderr().writer(...)` in production runtime paths unless you are editing the shared runtime logger itself.
- Keep feature call sites semantic.
  Prefer helpers such as `logProgress`, `logRelaxIter`, or DFPT-specific wrappers over ad hoc strings scattered through solver code.
- Preserve true warnings and errors.
  Reduce noisy success-path output first.
- When a dump is large or repetitive, move it to an artifact file instead of the console.

## References

- Read [references/policy.md](references/policy.md) for the detailed policy, migration checklist, and repo-specific hotspots.
