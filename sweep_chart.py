from __future__ import annotations

import json
import time
from pathlib import Path
import math 

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import (
    _normalize_model_output,
    time_sample,
    time_sample_batch,
    time_vmap_overhead,
    model_1_random_slopes, model_1_random_slopes_vmapped,
    model_3_bayesian_matrix_factorization, model_3_bayesian_matrix_factorization_vmapped,
    model_4_fir_time_series, model_4_fir_time_series_vmapped,
    model_5_poisson_claims_with_exposure, model_5_poisson_claims_with_exposure_vmapped,
    model_9_crossed_effects_multiplicative, model_9_crossed_effects_multiplicative_vmapped,
)

OUT_DIR = Path("scaling_figures")
OUT_DIR.mkdir(exist_ok=True)


NITER = 1000 


BASE_KWARGS = {
    "crossed_effects_multiplicative": {"n_subj": 16, "n_item": 16},
    "bayesian_matrix_factorization": {"n_users": 10, "n_items": 9, "d": 3},
    "poisson_claims_with_exposure": {"n": 72},
    "fir_time_series": {"T": 64, "K": 4},
    "random_slopes_mixed_effects": {"n_groups": 10, "n_per_group": 13},
}


SCALE_KEYS = {
    "crossed_effects_multiplicative": ["n_subj", "n_item"],
    "bayesian_matrix_factorization": ["n_users", "n_items"],
    "poisson_claims_with_exposure": ["n"],
    "fir_time_series": ["T"],
    "random_slopes_mixed_effects": ["n_per_group"],
}


def _scale_kwargs(name, multiplier):
    base = BASE_KWARGS[name]
    out = dict(base)
    mult = math.pow(multiplier, 1/len(SCALE_KEYS[name]))
    for k in SCALE_KEYS[name]:
        out[k] = max(1, round(base[k] * mult))
    return out


def _total_size(name, multiplier):
    kw = _scale_kwargs(name, multiplier)
    if name == "crossed_effects_multiplicative":
        return 2 * kw["n_subj"] * kw["n_item"]
    if name == "bayesian_matrix_factorization":
        return 2 * kw["d"] * kw["n_users"] * kw["n_items"]
    if name == "poisson_claims_with_exposure":
        return 7 * kw["n"]
    if name == "fir_time_series":
        return 2 * kw["K"] * kw["T"]
    if name == "random_slopes_mixed_effects":
        return 4 * BASE_KWARGS[name]["n_groups"] * kw["n_per_group"]
    raise ValueError(name)


MULTIPLIERS = {
    "crossed_effects_multiplicative": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "bayesian_matrix_factorization": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "poisson_claims_with_exposure": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "fir_time_series": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "random_slopes_mixed_effects": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}



SCALING_MODELS = [
    dict(
        name="crossed_effects_multiplicative",
        raw_builder=model_9_crossed_effects_multiplicative,
        vmap_builder=model_9_crossed_effects_multiplicative_vmapped,
        x_label="~graph nodes = 2 * n_subj * n_item",

        raw_cutoff_mult=5,
    ),
    dict(
        name="bayesian_matrix_factorization",
        raw_builder=model_3_bayesian_matrix_factorization,
        vmap_builder=model_3_bayesian_matrix_factorization_vmapped,
        x_label="~graph nodes = 2d * n_users * n_items, d=3 fixed",
        raw_cutoff_mult=5,
    ),
    dict(
        name="poisson_claims_with_exposure",
        raw_builder=model_5_poisson_claims_with_exposure,
        vmap_builder=model_5_poisson_claims_with_exposure_vmapped,
        x_label="~graph nodes = 7 * n",
        raw_cutoff_mult=5,
    ),
    dict(
        name="fir_time_series",
        raw_builder=model_4_fir_time_series,
        vmap_builder=model_4_fir_time_series_vmapped,
        x_label="~graph nodes = 2K * T, K=4 fixed",
        raw_cutoff_mult=5,
    ),
    dict(
        name="random_slopes_mixed_effects",
        raw_builder=model_1_random_slopes,
        vmap_builder=model_1_random_slopes_vmapped,
        x_label="~graph nodes = 4 * n_groups * n_per_group",
        raw_cutoff_mult=5,
    ),
]


def measure_timing(raw_builder, vmap_builder, kwargs, niter):
    raw_targets, given_vars, given_vals = _normalize_model_output(raw_builder(**kwargs))
    vmap_targets, vmap_given_vars, vmap_given_vals = _normalize_model_output(vmap_builder(**kwargs))

    raw_time = time_sample(raw_targets, niter, given_vars, given_vals)
    mapped_time = time_sample_batch(raw_targets, niter, given_vars, given_vals)
    handvmap_time = time_sample(vmap_targets, niter, vmap_given_vars, vmap_given_vals)

    return raw_time, mapped_time, handvmap_time


import multiprocessing as mp
import queue as queue_module

HEARTBEAT_S = 60      


def _measure_timing_worker(raw_builder, vmap_builder, kwargs, niter, result_queue, skip_raw=False):
    try:
        raw_targets, given_vars, given_vals = _normalize_model_output(raw_builder(**kwargs))
        vmap_targets, vmap_given_vars, vmap_given_vals = _normalize_model_output(vmap_builder(**kwargs))

        if skip_raw:
            result_queue.put(("progress", "raw (unfused): SKIPPED (past raw_cutoff_mult for this model)"))
            raw_time = None
        else:
            result_queue.put(("progress", "raw (unfused): starting..."))
            t0 = time.perf_counter()
            raw_time = time_sample(raw_targets, niter, given_vars, given_vals)
            result_queue.put(("progress", f"raw (unfused): done in {time.perf_counter()-t0:.1f}s"))

        result_queue.put(("progress", "engine (VmapEngine): starting..."))
        t0 = time.perf_counter()
        mapped_time = time_sample_batch(raw_targets, niter, given_vars, given_vals)
        result_queue.put(("progress", f"engine (VmapEngine): done in {time.perf_counter()-t0:.1f}s"))

        result_queue.put(("progress", "engine (run_all_vmaps overhead): starting..."))
        t0 = time.perf_counter()
        overhead_time = time_vmap_overhead(raw_targets, given_vars, given_vals)
        result_queue.put(("progress", f"engine (run_all_vmaps overhead): done in {time.perf_counter()-t0:.1f}s"))

        result_queue.put(("progress", "hand_vmap (oracle): starting..."))
        t0 = time.perf_counter()
        handvmap_time = time_sample(vmap_targets, niter, vmap_given_vars, vmap_given_vals)
        result_queue.put(("progress", f"hand_vmap (oracle): done in {time.perf_counter()-t0:.1f}s"))

        result_queue.put(("ok", (raw_time, mapped_time, handvmap_time, overhead_time)))
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


def measure_timing_with_timeout(raw_builder, vmap_builder, kwargs, niter,
                                 heartbeat_s=HEARTBEAT_S, skip_raw=False):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_measure_timing_worker,
                     args=(raw_builder, vmap_builder, kwargs, niter, q, skip_raw))
    p.start()

    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        try:
            status, payload = q.get(timeout=heartbeat_s)
        except queue_module.Empty:
            print(f"      ... still running [{elapsed/60:.1f} min elapsed, "
                  f"no stage update in last {heartbeat_s}s]", flush=True)
            continue

        if status == "progress":
            print(f"      [{elapsed/60:.1f} min] {payload}", flush=True)
            continue
        elif status == "error":
            p.join()
            raise RuntimeError(payload)
        elif status == "ok":
            p.join()
            return payload


def run_sweep():
    mult_table =  MULTIPLIERS
    results = []
    by_model = {cfg["name"]: [] for cfg in SCALING_MODELS}

    try:
        for cfg in SCALING_MODELS:
            name = cfg["name"]
            mults = mult_table[name]
            print(f"\n{'='*70}\n{name}  (benchmark.py default: {BASE_KWARGS[name]})\n{'='*70}")

            for mult in mults:
                kwargs = _scale_kwargs(name, mult)
                x_value = _total_size(name, mult)
                print(f"  multiplier={mult}x (kwargs={kwargs}, x={x_value}) ...", flush=True)

                raw_cutoff = cfg.get("raw_cutoff_mult")
                skip_raw = raw_cutoff is not None and mult > raw_cutoff

                try:
                    t0 = time.perf_counter()
                    raw_time, mapped_time, handvmap_time, overhead_time = measure_timing_with_timeout(
                        cfg["raw_builder"], cfg["vmap_builder"], kwargs, NITER, skip_raw=skip_raw
                    )
                    raw_str = f"{raw_time:.3f}s" if raw_time is not None else "SKIPPED"
                    speedup_str = f"{raw_time/mapped_time:.2f}x" if raw_time is not None else "n/a (raw skipped)"
                    print(f"    timing: raw={raw_str}  engine={mapped_time:.3f}s  "
                          f"engine_overhead={overhead_time:.3f}s  "
                          f"hand_vmap={handvmap_time:.3f}s  speedup={speedup_str}  "
                          f"[{time.perf_counter()-t0:.2f}s]")
                except Exception as e:
                    print(f"    SKIPPING {name} @ {mult}x -- {type(e).__name__}: {e}")
                    continue

                row = dict(
                    model=name, multiplier=mult, x_value=x_value,
                    raw_time_s=raw_time, mapped_time_s=mapped_time, handvmap_time_s=handvmap_time,
                    overhead_time_s=overhead_time,
                )
                results.append(row)
                by_model[name].append(row)

            _plot_model_if_ready(name, cfg["x_label"], by_model[name])
            _plot_combined_if_ready(by_model)
    finally:
        for cfg in SCALING_MODELS:
            _plot_model_if_ready(cfg["name"], cfg["x_label"], by_model[cfg["name"]])
        _plot_combined_if_ready(by_model)

    return results, by_model


def _plot_model_if_ready(name, x_label, rows):
    if len(rows) < 2:
        print(f"  (skipping plots for {name}: need >=2 completed multipliers, got {len(rows)})")
        return
    plot_time_scaling(name, x_label, rows)
    plot_speedup(name, x_label, rows)
    print(f"  saved {name}_time_scaling.png, {name}_speedup.png")


def _plot_combined_if_ready(by_model):
    ready = {name: rows for name, rows in by_model.items() if len(rows) >= 2}
    if len(ready) < 1:
        return
    plot_combined_speedup(ready)
    print(f"  updated combined_speedup.png ({len(ready)}/{len(by_model)} models so far)")


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def _fit_loglog_slope(xs, ys):
    """Fit log(y) = a*log(x) + b, return the exponent a (the 'O(x^a)' you
    can annotate a chart with)."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    mask = (xs > 0) & (ys > 0)
    if mask.sum() < 2:
        return float("nan")
    a, b = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
    return a


def _fit_loglog_curve(xs, ys):
    """Fit log(y) = a*log(x) + b. Returns (slope_a, predict_fn) where
    predict_fn(x) gives the fitted y at any x (used to extrapolate beyond
    the measured points)."""
    xs_arr, ys_arr = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    mask = (xs_arr > 0) & (ys_arr > 0)
    if mask.sum() < 2:
        return float("nan"), lambda x: float("nan")
    a, b = np.polyfit(np.log(xs_arr[mask]), np.log(ys_arr[mask]), 1)
    return a, (lambda x: np.exp(b) * np.asarray(x, dtype=float) ** a)


def plot_time_scaling(model_name, x_label, rows):
    xs = [r["x_value"] for r in rows]
    mapped = [r["mapped_time_s"] for r in rows]
    handvmap = [r["handvmap_time_s"] for r in rows]
    overhead = [r["overhead_time_s"] for r in rows]

    raw_xs_measured = [x for x, r in zip(xs, rows) if r["raw_time_s"] is not None]
    raw_ys_measured = [r["raw_time_s"] for r in rows if r["raw_time_s"] is not None]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for ys, label, marker in [(mapped, "VmapEngine", "s"),
                               (handvmap, "hand-vmapped (oracle)", "^"),
                               (overhead, "VmapEngine overhead (run_all_vmaps)", "d")]:
        slope = _fit_loglog_slope(xs, ys)
        ax.plot(xs, ys, marker=marker, label=f"{label}  (O(x^{slope:.2f}))")

    if raw_xs_measured:
        slope, predict = _fit_loglog_curve(raw_xs_measured, raw_ys_measured)
        ax.plot(raw_xs_measured, raw_ys_measured, marker="o", linestyle="-",
                 color="tab:blue", label=f"raw (unfused), measured  (O(x^{slope:.2f}))")

        raw_xs_skipped = [x for x, r in zip(xs, rows) if r["raw_time_s"] is None]
        if raw_xs_skipped and not np.isnan(slope):
            # Extend the fitted curve from the last measured point through
            # every skipped (larger) x, so the dashed segment visibly picks
            # up where the solid measured line leaves off.
            fit_xs = sorted([raw_xs_measured[-1]] + raw_xs_skipped)
            fit_ys = predict(fit_xs)
            ax.plot(fit_xs, fit_ys, marker="o", linestyle="--", color="tab:blue",
                     markerfacecolor="none", alpha=0.6,
                     label="raw (unfused), extrapolated fit -- not measured")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"wall time, s (median of 3 @ niter={NITER})")
    ax.set_title(f"{model_name}: wall time vs. model size")
    ax.legend(fontsize=7.5)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_time_scaling.png", dpi=200)
    plt.close(fig)


def plot_speedup(model_name, x_label, rows):
    measured = [r for r in rows if r["raw_time_s"] is not None]
    xs = [r["x_value"] for r in measured]
    speedup = [r["raw_time_s"] / r["mapped_time_s"] for r in measured]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, speedup, marker="o", color="darkgreen")
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("speedup  (raw time / engine time)")
    title = f"{model_name}: engine speedup vs. model size"
    if len(measured) < len(rows):
        title += f"\n(only {len(measured)}/{len(rows)} points shown -- raw not measured past raw_cutoff_mult)"
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_speedup.png", dpi=200)
    plt.close(fig)


def plot_combined_speedup(rows_by_model):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows in rows_by_model.items():
        measured = [r for r in rows if r["raw_time_s"] is not None]
        xs = [r["x_value"] for r in measured]
        speedup = [r["raw_time_s"] / r["mapped_time_s"] for r in measured]
        ax.plot(xs, speedup, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("model size (see per-model figures for exact x-axis definition)")
    ax.set_ylabel("speedup  (raw time / engine time)")
    ax.set_title("VmapEngine speedup vs. model size, across model families")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_speedup.png", dpi=200)
    plt.close(fig)


def main():
    results, by_model = run_sweep()
    n_plotted = sum(1 for rows in by_model.values() if len(rows) >= 2)
    print(f"\nDone. Figures saved to {OUT_DIR}/ "
          f"({n_plotted}/{len(by_model)} models had >=2 completed multipliers and got plotted).")


if __name__ == "__main__":
    main()