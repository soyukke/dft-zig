# DFT Zig Logging Policy

## Intent

This repo should treat logging as a cross-cutting runtime concern.

- Event sources live in feature code.
- Level policy and sink behavior live in shared runtime logging.
- Heavy numeric diagnostics should prefer files over console output.

The main smell to remove is feature code choosing its own output backend with raw `std.debug.print` or direct `stderr` writers.

## Level Model

- `err`: execution cannot proceed or a command should fail
- `warn`: execution continues, but the user should notice
- `info`: concise progress and summaries for normal runs
- `debug`: detailed diagnostics, iteration internals, value dumps

Separate these from:

- `profile`: timing and counters
- `artifact`: CSV, JSON, matrices, trajectories, other output files
- `test-only`: diagnostic output used only by tests or benchmarks

## Normal Runtime Policy

- Keep `info` short.
  Examples: calculation start, thread counts, convergence, final frequencies, final stress, final band summary.
- Keep iteration internals in `debug`.
  Examples: per-iteration residuals, dense matrix elements, perturbation-by-perturbation traces, full eigenvalue dumps.
- Keep `warn` and `err` visible even in quiet runs.
- Avoid success-path logs on `stderr` that look like failures in `zig build test`.

## Structural Rules

- Production paths should not call raw `std.debug.print`.
- Production paths should not create raw `std.Io.File.stderr().writer(...)` writers outside the shared runtime logger.
- Use `src/features/runtime/logging.zig` for shared level handling.
- Use thin feature wrappers for semantic events.
  Examples:
  - `src/features/scf/logging.zig`
  - DFPT wrappers in `src/features/dfpt/dfpt.zig`
- If a feature needs a new config gate, add it in config and interpret it centrally.

## Migration Pattern

1. Search the affected area.

```sh
rg -n "std\\.debug\\.print|std\\.Io\\.File\\.stderr\\(\\)\\.writer" src
rg -n "logDfpt|quiet|debug_|profile" src/features
```

2. Classify each message.

- keep as `warn` or `err`
- keep as concise `info`
- demote to `debug`
- move to artifact file
- leave as test-only or benchmark-only

3. Centralize the backend.

- Extend `src/features/runtime/logging.zig` if you need new shared behavior.
- Prefer semantic helpers over call-site strings where practical.

4. Reduce noisy summaries before touching real warnings.

5. Verify:

```sh
zig fmt <changed-files>
zig build
zig build test --summary all
```

## Adding New Log Events

When adding a brand-new log, use this checklist:

1. Decide the class first:
   - `err`
   - `warn`
   - `info`
   - `debug`
   - `profile`
   - `artifact`
2. Prefer semantic helpers over inline strings.
   Good: `logRelaxIter(...)`, `logProgress(...)`, DFPT-specific wrappers.
   Worse: ad hoc strings emitted directly from solver loops.
3. Keep `info` compact.
   One line for start, convergence, summary, or major phase change is usually enough.
4. Treat repeated inner-loop signals as `debug` by default.
5. If the message includes lots of numeric data, ask whether it should become a file in `out_dir`.
6. If a warning can fire many times, consider deduplication or rate limiting.

Useful default choices in this repo:

- start/end of a major phase: `info`
- convergence reached: `info`
- fallback path taken but still valid: `warn` or `info`, depending on importance
- per-iteration residuals: `debug`
- full matrix / full spectrum dumps: `artifact` or `debug`
- environment/backend discovery chatter: `info` if rare and useful, otherwise `debug`

## Current Repo Priorities

These areas are especially worth checking when doing logging cleanup:

- `src/features/kpoints`
- `src/features/forces`
- `src/lib/fft/metal_fft.zig`
- remaining DFPT `debug` noise if normal runs are still too chatty
- test and benchmark files that intentionally print diagnostics

## Practical Guidelines

- Do not log and then immediately rethrow unless the log adds context the caller cannot add later.
- Rate-limit repeated progress logs.
  Examples: every N k-points, first/last iteration, convergence transitions only.
- Prefer stable field ordering in structured or semi-structured text output.
- If a message is primarily for machine comparison, write a file instead of a console line.
- If a warning is expected in tests, keep it explicit and scoped to tests.

## Example Decisions

- `unknown element 'Xx', using mass=1.0 AMU`
  Keep as `warn`.
- `Metal FFT: using GPU device ...`
  Usually `info` or `debug`, depending on how often it appears and whether it matters to a normal user.
- `dfptQ_mk: iter=... vresid=...`
  `debug`.
- `phonon frequencies`
  `info`.
- `full dynmat`
  artifact file or `debug`, not default console output.
