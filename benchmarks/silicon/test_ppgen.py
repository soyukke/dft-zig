#!/usr/bin/env python3
"""Generate ppgen Si PBE UPF, run Si SCF+bands, and compare with ABINIT."""

import csv
import json
import math
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT_DIR = os.path.join(SCRIPT_DIR, "out_ppgen")
UPF_PATH = os.path.join(OUT_DIR, "Si_ppgen_PBE.upf")
LOG_DERIV_PATH = os.path.join(OUT_DIR, "Si_ppgen_logderiv.tsv")
DIAGNOSTICS_PATH = os.path.join(OUT_DIR, "ppgen_diagnostics.json")
CONFIG = "dft_zig_ppgen.toml"
BASELINE_DIR = os.path.join(SCRIPT_DIR, "baseline")
RY_TO_EV = 13.6057
GAP_DIFF_THRESHOLD_MEV = 75.0
AVG_MSE_THRESHOLD_MEV2 = 6_500.0
MAX_ABS_DIJ_THRESHOLD = 20_000.0
MAX_DIJ_ASYM_THRESHOLD = 1e-8
NLCC_CHARGE_THRESHOLD = 1e-3
LOG_DERIV_MAX_THRESHOLD = 0.75
LOG_DERIV_RMS_THRESHOLD = 0.17
LOG_DERIV_INVALID_THRESHOLD = 0
LOG_DERIV_POLE_MISMATCH_THRESHOLD = 0


def run(cmd, cwd):
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def build_and_run():
    os.makedirs(OUT_DIR, exist_ok=True)
    run(["zig", "build", "-Doptimize=ReleaseFast"], os.path.join(ROOT, "ppgen"))
    run([
        os.path.join(ROOT, "ppgen", "zig-out", "bin", "ppgen"),
        "--xc",
        "pbe",
        "--rc-s",
        "1.71",
        "--rc-p",
        "1.64",
        "--local-l",
        "2",
        "--p-ref-energy-ry",
        "0.8",
        "--d-energy-ry",
        "1.2",
        "--nlcc-charge",
        "0.7241414335",
        "--nlcc-radius",
        "0.80",
        "--log-deriv",
        LOG_DERIV_PATH,
        "--log-deriv-min-ry",
        "0.0",
        "--log-deriv-max-ry",
        "1.6",
        "--log-deriv-step-ry",
        "0.2",
        UPF_PATH,
    ], ROOT)
    run(["zig", "build", "-Doptimize=ReleaseFast"], ROOT)
    run([os.path.join(ROOT, "zig-out", "bin", "dft_zig"), CONFIG], SCRIPT_DIR)
    run(["python3", "diagnose_ppgen.py"], SCRIPT_DIR)


def parse_status():
    status_path = os.path.join(OUT_DIR, "status.txt")
    values = {}
    with open(status_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def load_bands():
    path = os.path.join(OUT_DIR, "band_energies.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise AssertionError("band_energies.csv is empty")

    band_keys = [key for key in rows[0].keys() if key.startswith("band")]
    bands = []
    for key in band_keys:
        values = [float(row[key]) * RY_TO_EV for row in rows]
        if not all(math.isfinite(v) for v in values):
            raise AssertionError(f"{key} contains non-finite values")
        bands.append(values)
    return rows, bands


def load_abinit_baseline():
    path = os.path.join(BASELINE_DIR, "abinit_bands.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise AssertionError("abinit_bands.csv is empty")

    band_keys = [key for key in rows[0].keys() if key.startswith("band")]
    bands = []
    for key in band_keys:
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(v) for v in values):
            raise AssertionError(f"ABINIT {key} contains non-finite values")
        bands.append(values)
    return bands


def align_to_vbm(bands):
    vbm = max(bands[3])
    return [[value - vbm for value in band] for band in bands]


def band_gap(bands):
    aligned = align_to_vbm(bands)
    return min(min(band) for band in aligned[4:])


def mean_square_errors(candidate, reference):
    if len(candidate) != len(reference):
        raise AssertionError(f"band count mismatch: {len(candidate)} vs {len(reference)}")
    errors = []
    for idx, (cand_band, ref_band) in enumerate(zip(candidate, reference)):
        if len(cand_band) != len(ref_band):
            raise AssertionError(f"band {idx} k-point count mismatch")
        diff2 = [(c - r) * 1000.0 for c, r in zip(cand_band, ref_band)]
        errors.append(sum(value * value for value in diff2) / len(diff2))
    return errors


def load_log_derivatives():
    with open(LOG_DERIV_PATH) as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))
    if not rows:
        raise AssertionError("Si_ppgen_logderiv.tsv is empty")
    return rows


def load_max_abs_dij():
    with open(UPF_PATH) as f:
        text = f.read()
    match = re.search(r"<PP_DIJ[^>]*>(.*?)</PP_DIJ>", text, re.DOTALL)
    if not match:
        raise AssertionError("PP_DIJ is missing from generated UPF")
    values = [float(token) for token in match.group(1).split()]
    if not values:
        raise AssertionError("PP_DIJ is empty")
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("PP_DIJ contains non-finite values")
    return max(abs(value) for value in values)


def load_diagnostics():
    with open(DIAGNOSTICS_PATH) as f:
        diagnostics = json.load(f)
    assert_finite_tree(diagnostics)
    return diagnostics


def assert_finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("ppgen diagnostics contain a non-finite value")


def check_log_derivatives():
    rows = load_log_derivatives()
    valid_deltas = []
    invalid = 0
    for row in rows:
        status = row["status"]
        if status == "ok":
            delta = float(row["delta"])
            if not math.isfinite(delta):
                raise AssertionError("log-derivative delta contains non-finite value")
            valid_deltas.append(delta)
        else:
            invalid += 1

    if not valid_deltas:
        raise AssertionError("no valid log-derivative samples")

    max_abs = max(abs(delta) for delta in valid_deltas)
    rms = math.sqrt(sum(delta * delta for delta in valid_deltas) / len(valid_deltas))
    if max_abs > LOG_DERIV_MAX_THRESHOLD:
        raise AssertionError(f"log-derivative max delta too large: {max_abs:.4f}")
    if rms > LOG_DERIV_RMS_THRESHOLD:
        raise AssertionError(f"log-derivative RMS delta too large: {rms:.4f}")
    if invalid > LOG_DERIV_INVALID_THRESHOLD:
        raise AssertionError(f"too many invalid log-derivative samples: {invalid}")
    pole_mismatches = count_log_derivative_pole_mismatches(rows)
    if pole_mismatches > LOG_DERIV_POLE_MISMATCH_THRESHOLD:
        raise AssertionError(f"log-derivative pole mismatch count: {pole_mismatches}")
    return max_abs, rms, invalid, pole_mismatches


def count_log_derivative_pole_mismatches(rows):
    grouped = {}
    for row in rows:
        key = (int(row["channel_n"]), int(row["l"]))
        grouped.setdefault(key, []).append(row)

    mismatches = 0
    for block in grouped.values():
        block.sort(key=lambda row: float(row["energy_ry"]))
        ae_poles = count_log_derivative_poles(block, "ae")
        pseudo_poles = count_log_derivative_poles(block, "pseudo")
        if ae_poles != pseudo_poles:
            mismatches += 1
    return mismatches


def count_log_derivative_poles(rows, key):
    pole_abs_threshold = 5.0
    count = 0
    for left, right in zip(rows, rows[1:]):
        if left["status"] != "ok" or right["status"] != "ok":
            continue
        left_value = float(left[key])
        right_value = float(right[key])
        if left_value * right_value < 0.0 and max(abs(left_value), abs(right_value)) >= pole_abs_threshold:
            count += 1
    return count


def check_results():
    status = parse_status()
    if status.get("scf_converged") != "true":
        raise AssertionError("SCF did not converge")

    iterations = int(status["scf_iterations"])
    if iterations > 80:
        raise AssertionError(f"SCF took too many iterations: {iterations}")

    total_energy = float(status["scf_energy_total"])
    if not (-25.0 < total_energy < -10.0):
        raise AssertionError(f"unexpected primitive-cell total energy: {total_energy} Ry")

    rows, bands = load_bands()
    abinit_bands = load_abinit_baseline()
    if len(bands) != 8:
        raise AssertionError(f"expected 8 bands, got {len(bands)}")
    if len(rows) != 101:
        raise AssertionError(f"expected 101 band k-points, got {len(rows)}")

    gap_ev = band_gap(bands)
    valence_width_ev = max(max(band) for band in bands[:4]) - min(min(band) for band in bands[:4])
    abinit_gap_ev = band_gap(abinit_bands)
    gap_diff_mev = abs(gap_ev - abinit_gap_ev) * 1000.0
    mse_per_band = mean_square_errors(align_to_vbm(bands), align_to_vbm(abinit_bands))
    avg_mse = sum(mse_per_band) / len(mse_per_band)

    if not (0.05 < gap_ev < 3.0):
        raise AssertionError(f"unexpected Si gap from ppgen UPF: {gap_ev:.4f} eV")
    if not (5.0 < valence_width_ev < 20.0):
        raise AssertionError(f"unexpected valence width: {valence_width_ev:.4f} eV")
    if gap_diff_mev > GAP_DIFF_THRESHOLD_MEV:
        raise AssertionError(f"gap differs from ABINIT by {gap_diff_mev:.1f} meV")
    if avg_mse > AVG_MSE_THRESHOLD_MEV2:
        raise AssertionError(f"band MSE vs ABINIT is too large: {avg_mse:.1f} meV^2")
    max_abs_dij = load_max_abs_dij()
    if max_abs_dij > MAX_ABS_DIJ_THRESHOLD:
        raise AssertionError(f"generated PP_DIJ is poorly conditioned: {max_abs_dij:.3e}")
    log_max_abs, log_rms, log_invalid, log_pole_mismatches = check_log_derivatives()
    diagnostics = load_diagnostics()
    if diagnostics["dij"]["max_abs"] > MAX_ABS_DIJ_THRESHOLD:
        raise AssertionError(f"diagnostic D_ij is poorly conditioned: {diagnostics['dij']['max_abs']:.3e}")
    if diagnostics["dij"]["max_asym"] > MAX_DIJ_ASYM_THRESHOLD:
        raise AssertionError(f"diagnostic D_ij is not Hermitian: {diagnostics['dij']['max_asym']:.3e}")
    if not diagnostics["form_factor"]["beta_channels"]:
        raise AssertionError("ppgen form-factor diagnostics did not compare any beta channels")
    if "objective" not in diagnostics["form_factor"]:
        raise AssertionError("ppgen form-factor objective is missing")
    if abs(diagnostics["nlcc"]["ppgen"]["charge"] - 0.7241414335) > NLCC_CHARGE_THRESHOLD:
        raise AssertionError(f"unexpected NLCC charge: {diagnostics['nlcc']['ppgen']['charge']:.6f}")
    if not diagnostics["logderiv_by_l"]:
        raise AssertionError("ppgen log-derivative diagnostics are empty")

    print("ppgen Si PBE comparison:")
    print(f"  SCF iterations: {iterations}")
    print(f"  Total energy:   {total_energy:.8f} Ry")
    print(f"  Band gap:       {gap_ev:.4f} eV")
    print(f"  ABINIT gap:     {abinit_gap_ev:.4f} eV")
    print(f"  Gap diff:       {gap_diff_mev:.1f} meV")
    print(f"  Valence width:  {valence_width_ev:.4f} eV")
    print(f"  Avg band MSE:   {avg_mse:.1f} meV^2")
    print("  Per-band MSE:   " + ", ".join(f"{mse:.1f}" for mse in mse_per_band))
    print(f"  Max |D_ij|:     {max_abs_dij:.3e}")
    print(f"  NLCC charge:    {diagnostics['nlcc']['ppgen']['charge']:.4f}")
    print(f"  Logderiv max:   {log_max_abs:.4f}")
    print(f"  Logderiv RMS:   {log_rms:.4f}")
    print(f"  Logderiv bad:   {log_invalid}")
    print(f"  Pole mismatch:  {log_pole_mismatches}")
    print(f"  Local FF RMS:   {diagnostics['form_factor']['local_sr_rms_delta']:.3e}")
    print_form_factor_objective(diagnostics["form_factor"])


def print_form_factor_objective(form_factor):
    objective = form_factor["objective"]
    local = form_factor["local_sr_windows"]
    print(f"  Local FF 0-5:   {objective['local_solid_q_rms']:.3e}")
    print(f"  Beta FF 0-5:    {objective['beta_mean_solid_q_rms']:.3e}")
    print(f"  FF score:       {objective['absolute_score']:.3e}")
    print(
        "  Local FF bins:  " +
        ", ".join(
            f"{name}={local[name]['rms_delta']:.3e}"
            for name in ["low", "mid", "high"]
        )
    )


if __name__ == "__main__":
    if "--run" in sys.argv:
        build_and_run()
    check_results()
