#!/usr/bin/env python3
"""Write ppgen Si diagnostics against the local ONCV/ABINIT reference."""

import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = SCRIPT_DIR / "out_ppgen"
PPGEN_UPF = OUT_DIR / "Si_ppgen_PBE.upf"
REFERENCE_UPF = ROOT / "pseudo" / "Si.upf"
LOG_DERIV_PATH = OUT_DIR / "Si_ppgen_logderiv.tsv"
BAND_PATH = OUT_DIR / "band_energies.csv"
KPOINT_PATH = OUT_DIR / "band_kpoints.csv"
BASELINE_BANDS = SCRIPT_DIR / "baseline" / "abinit_bands.csv"
DIAGNOSTICS_JSON = OUT_DIR / "ppgen_diagnostics.json"
FORM_FACTOR_CSV = OUT_DIR / "ppgen_form_factors.csv"
BAND_ERROR_CSV = OUT_DIR / "ppgen_band_errors.csv"
RY_TO_EV = 13.6057
Q_WINDOWS = (
    ("low", 0.0, 2.0),
    ("mid", 2.0, 5.0),
    ("high", 5.0, 8.0),
)


def read_text(path):
    with open(path) as f:
        return f.read()


def floats(text):
    return [float(token) for token in text.split()]


def tag_body(text, name):
    match = re.search(rf"<{re.escape(name)}(?:\s[^>]*)?>(.*?)</{re.escape(name)}>", text, re.S)
    if not match:
        raise AssertionError(f"{name} is missing")
    return match.group(1)


def tag_attrs(opening_tag):
    attrs = {}
    for key, value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"', opening_tag):
        attrs[key] = value.strip()
    return attrs


def parse_upf(path):
    text = read_text(path)
    header_open = re.search(r"<PP_HEADER\b[^>]*>", text, re.S)
    if not header_open:
        raise AssertionError(f"{path} has no PP_HEADER")
    header = tag_attrs(header_open.group(0))
    r = np.array(floats(tag_body(text, "PP_R")), dtype=float)
    rab = np.array(floats(tag_body(text, "PP_RAB")), dtype=float)
    local = np.array(floats(tag_body(text, "PP_LOCAL")), dtype=float)
    nlcc_match = re.search(r"<PP_NLCC(?:\s[^>]*)?>(.*?)</PP_NLCC>", text, re.S)
    nlcc = np.array(floats(nlcc_match.group(1)), dtype=float) if nlcc_match else np.array([], dtype=float)
    betas = []
    for match in re.finditer(r"<PP_BETA\.(\d+)\b([^>]*)>(.*?)</PP_BETA\.\1>", text, re.S):
        attrs = tag_attrs(match.group(2))
        values = np.array(floats(match.group(3)), dtype=float)
        l = int(attrs["angular_momentum"])
        betas.append({
            "index": int(match.group(1)),
            "l": l,
            "cutoff_radius_index": int(attrs.get("cutoff_radius_index", len(values))),
            "values": values,
        })
    dij = np.array(floats(tag_body(text, "PP_DIJ")), dtype=float)
    return {
        "core_correction": header.get("core_correction", "F") == "T",
        "z_valence": float(header.get("z_valence", "4.0")),
        "r": r,
        "rab": rab,
        "local": local,
        "nlcc": nlcc,
        "betas": betas,
        "dij": dij,
    }


def spherical_bessel(l, x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-7
    if l == 0:
        out[small] = 1.0 - x[small] * x[small] / 6.0
        out[~small] = np.sin(x[~small]) / x[~small]
    elif l == 1:
        out[small] = x[small] / 3.0
        xs = x[~small]
        out[~small] = np.sin(xs) / (xs * xs) - np.cos(xs) / xs
    elif l == 2:
        out[small] = x[small] * x[small] / 15.0
        xs = x[~small]
        out[~small] = (3.0 / (xs * xs) - 1.0) * np.sin(xs) / xs - 3.0 * np.cos(xs) / (xs * xs)
    else:
        raise AssertionError(f"unsupported l={l}")
    return out


def radial_integral(r, rab, values, q, l, power):
    n = min(len(r), len(rab), len(values))
    weights = np.ones(n)
    if n > 1:
        weights[0] = 0.5
        weights[-1] = 0.5
    return float(np.sum(weights * rab[:n] * (r[:n] ** power) * values[:n] * spherical_bessel(l, q * r[:n])))


def local_short_range_form(upf, q):
    n = min(len(upf["r"]), len(upf["rab"]), len(upf["local"]))
    r = upf["r"][:n]
    values = upf["local"][:n].copy()
    nonzero = r > 0.0
    values[nonzero] += 2.0 * upf["z_valence"] / r[nonzero]
    if not nonzero[0]:
        values[0] = values[1] if n > 1 else 0.0
    return 4.0 * math.pi * radial_integral(r, upf["rab"][:n], values, q, 0, 2)


def beta_form(upf, beta, q):
    return radial_integral(upf["r"], upf["rab"], beta["values"], q, beta["l"], 1)


def load_bands(path, ev):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    keys = [key for key in rows[0] if key.startswith("band")]
    scale = RY_TO_EV if ev == "ry" else 1.0
    return [[float(row[key]) * scale for row in rows] for key in keys]


def align_to_vbm(bands):
    vbm = max(bands[3])
    return [[value - vbm for value in band] for band in bands]


def band_gap(bands):
    aligned = align_to_vbm(bands)
    return min(min(band) for band in aligned[4:])


def load_kpoint_labels():
    if not KPOINT_PATH.exists():
        return []
    with open(KPOINT_PATH) as f:
        rows = list(csv.DictReader(f))
    return [row.get("label", "") for row in rows]


def write_band_errors(candidate, reference):
    labels = load_kpoint_labels()
    candidate_aligned = align_to_vbm(candidate)
    reference_aligned = align_to_vbm(reference)
    with open(BAND_ERROR_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k_index", "label", "band", "candidate_ev", "reference_ev", "delta_mev"])
        for band_index, (cand_band, ref_band) in enumerate(zip(candidate_aligned, reference_aligned)):
            for k_index, (cand, ref) in enumerate(zip(cand_band, ref_band)):
                label = labels[k_index] if k_index < len(labels) else ""
                writer.writerow([k_index, label, band_index, cand, ref, (cand - ref) * 1000.0])


def band_summary():
    candidate = load_bands(BAND_PATH, "ry")
    reference = load_bands(BASELINE_BANDS, "ev")
    write_band_errors(candidate, reference)
    candidate_aligned = align_to_vbm(candidate)
    reference_aligned = align_to_vbm(reference)
    per_band = []
    for cand_band, ref_band in zip(candidate_aligned, reference_aligned):
        deltas = [(cand - ref) * 1000.0 for cand, ref in zip(cand_band, ref_band)]
        per_band.append(sum(delta * delta for delta in deltas) / len(deltas))
    dominant = sorted(
        (
            {"band": band, "mse_mev2": mse}
            for band, mse in enumerate(per_band)
        ),
        key=lambda row: row["mse_mev2"],
        reverse=True,
    )
    return {
        "gap_ev": band_gap(candidate),
        "abinit_gap_ev": band_gap(reference),
        "gap_diff_mev": abs(band_gap(candidate) - band_gap(reference)) * 1000.0,
        "avg_mse_mev2": sum(per_band) / len(per_band),
        "per_band_mse_mev2": per_band,
        "dominant_bands": dominant[:3],
    }


def logderiv_summary():
    by_l = {}
    with open(LOG_DERIV_PATH) as f:
        rows = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        for row in rows:
            l = row["l"]
            block = by_l.setdefault(l, {"valid": 0, "invalid": 0, "max_abs_delta": 0.0, "rms_acc": 0.0})
            if row["status"] != "ok":
                block["invalid"] += 1
                continue
            delta = float(row["delta"])
            block["valid"] += 1
            block["max_abs_delta"] = max(block["max_abs_delta"], abs(delta))
            block["rms_acc"] += delta * delta
    for block in by_l.values():
        block["rms_delta"] = math.sqrt(block.pop("rms_acc") / block["valid"]) if block["valid"] else math.nan
    return by_l


def dij_summary(upf):
    nproj = len(upf["betas"])
    matrix = upf["dij"].reshape((nproj, nproj))
    blocks = []
    for l in sorted({beta["l"] for beta in upf["betas"]}):
        indices = [i for i, beta in enumerate(upf["betas"]) if beta["l"] == l]
        block = matrix[np.ix_(indices, indices)]
        blocks.append({
            "l": l,
            "indices": indices,
            "condition": float(np.linalg.cond(block)),
            "max_abs": float(np.max(np.abs(block))),
            "max_asym": float(np.max(np.abs(block - block.T))),
        })
    return {
        "max_abs": float(np.max(np.abs(matrix))),
        "max_asym": float(np.max(np.abs(matrix - matrix.T))),
        "blocks": blocks,
    }


def nlcc_summary(ppgen, reference):
    return {
        "ppgen": one_nlcc_summary(ppgen),
        "reference": one_nlcc_summary(reference),
    }


def one_nlcc_summary(upf):
    nlcc = upf["nlcc"]
    if len(nlcc) == 0:
        return {
            "core_correction": upf["core_correction"],
            "charge": 0.0,
            "max": 0.0,
            "rms_radius": 0.0,
        }
    n = min(len(upf["r"]), len(upf["rab"]), len(nlcc))
    r = upf["r"][:n]
    rab = upf["rab"][:n]
    weights = np.ones(n)
    if n > 1:
        weights[0] = 0.5
        weights[-1] = 0.5
    radial_weight = 4.0 * math.pi * weights * r * r * rab
    charge = float(np.sum(radial_weight * nlcc[:n]))
    r2 = float(np.sum(radial_weight * r * r * nlcc[:n]))
    return {
        "core_correction": upf["core_correction"],
        "charge": charge,
        "max": float(np.max(nlcc[:n])),
        "rms_radius": math.sqrt(r2 / charge) if charge > 0.0 else 0.0,
    }


def form_factor_summary(ppgen, reference):
    q_grid = np.linspace(0.0, 8.0, 161)
    rows = []
    local_pp = np.array([local_short_range_form(ppgen, q) for q in q_grid])
    local_ref = np.array([local_short_range_form(reference, q) for q in q_grid])
    local_delta = local_pp - local_ref
    for q, pp, ref in zip(q_grid, local_pp, local_ref):
        rows.append(["local_sr", "", q, pp, ref, pp - ref])

    beta_summaries = []
    pp_by_l = group_betas(ppgen["betas"])
    ref_by_l = group_betas(reference["betas"])
    for l in sorted(set(pp_by_l) & set(ref_by_l)):
        for ordinal, (pp_beta, ref_beta) in enumerate(zip(pp_by_l[l], ref_by_l[l])):
            pp_values = np.array([beta_form(ppgen, pp_beta, q) for q in q_grid])
            ref_values = np.array([beta_form(reference, ref_beta, q) for q in q_grid])
            delta = pp_values - ref_values
            channel = f"l{l}_ord{ordinal}"
            for q, pp, ref, diff in zip(q_grid, pp_values, ref_values, delta):
                rows.append(["beta", channel, q, pp, ref, diff])
            beta_summaries.append({
                "channel": channel,
                "ppgen_index": pp_beta["index"],
                "reference_index": ref_beta["index"],
                "l": l,
                "rms_delta": float(math.sqrt(np.mean(delta * delta))),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "windows": q_window_summary(q_grid, delta, ref_values),
            })

    with open(FORM_FACTOR_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "channel", "q_bohr_inv", "ppgen", "reference", "delta"])
        writer.writerows(rows)

    beta_solid_q = [
        channel["windows"]["solid"]["rms_delta"]
        for channel in beta_summaries
    ]
    local_windows = q_window_summary(q_grid, local_delta, local_ref)
    beta_mean_solid_q = float(np.mean(beta_solid_q)) if beta_solid_q else math.nan

    return {
        "q_min": float(q_grid[0]),
        "q_max": float(q_grid[-1]),
        "q_count": int(len(q_grid)),
        "local_sr_rms_delta": float(math.sqrt(np.mean(local_delta * local_delta))),
        "local_sr_max_abs_delta": float(np.max(np.abs(local_delta))),
        "local_sr_windows": local_windows,
        "beta_channels": beta_summaries,
        "objective": {
            "local_solid_q_rms": local_windows["solid"]["rms_delta"],
            "beta_mean_solid_q_rms": beta_mean_solid_q,
            "absolute_score": local_windows["solid"]["rms_delta"] + beta_mean_solid_q,
        },
    }


def q_window_summary(q_grid, delta, reference):
    result = {}
    for name, q_min, q_max in Q_WINDOWS:
        result[name] = q_window_metrics(q_grid, delta, reference, q_min, q_max)
    result["solid"] = q_window_metrics(q_grid, delta, reference, 0.0, 5.0)
    return result


def q_window_metrics(q_grid, delta, reference, q_min, q_max):
    mask = (q_grid >= q_min) & (q_grid <= q_max)
    if not np.any(mask):
        raise AssertionError("empty q window")
    window_delta = delta[mask]
    window_ref = reference[mask]
    rms = math.sqrt(np.mean(window_delta * window_delta))
    ref_scale = math.sqrt(np.mean(window_ref * window_ref))
    relative = rms / ref_scale if ref_scale > 1e-30 else 0.0
    return {
        "q_min": q_min,
        "q_max": q_max,
        "rms_delta": float(rms),
        "relative_rms_delta": float(relative),
        "max_abs_delta": float(np.max(np.abs(window_delta))),
    }


def group_betas(betas):
    grouped = {}
    for beta in betas:
        grouped.setdefault(beta["l"], []).append(beta)
    for block in grouped.values():
        block.sort(key=lambda beta: beta["index"])
    return grouped


def hardness_summary(upf):
    q_grid = np.linspace(0.0, 10.0, 201)
    result = []
    for beta in upf["betas"]:
        values = np.array([beta_form(upf, beta, q) for q in q_grid])
        weight = np.trapezoid(values * values, q_grid)
        high = np.trapezoid(values[q_grid >= 6.0] * values[q_grid >= 6.0], q_grid[q_grid >= 6.0])
        q2 = np.trapezoid(q_grid * q_grid * values * values, q_grid)
        result.append({
            "index": beta["index"],
            "l": beta["l"],
            "q_rms": float(math.sqrt(q2 / weight)) if weight > 0.0 else math.nan,
            "high_q_ratio_ge_6": float(high / weight) if weight > 0.0 else math.nan,
        })
    return result


def assert_finite_tree(value, path="diagnostics"):
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            assert_finite_tree(child, f"{path}[{idx}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"{path} is not finite")


def main():
    ppgen = parse_upf(PPGEN_UPF)
    reference = parse_upf(REFERENCE_UPF)
    diagnostics = {
        "band": band_summary(),
        "logderiv_by_l": logderiv_summary(),
        "dij": dij_summary(ppgen),
        "nlcc": nlcc_summary(ppgen, reference),
        "form_factor": form_factor_summary(ppgen, reference),
        "hardness": {
            "ppgen": hardness_summary(ppgen),
            "reference": hardness_summary(reference),
        },
    }
    assert_finite_tree(diagnostics)
    with open(DIAGNOSTICS_JSON, "w") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {DIAGNOSTICS_JSON}")
    print(f"Wrote {FORM_FACTOR_CSV}")
    print(f"Wrote {BAND_ERROR_CSV}")


if __name__ == "__main__":
    main()
