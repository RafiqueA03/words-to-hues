"""
Aggregation checks for the lexical-neighbourhood centroid
(paper Appendix B, E and F; Llama-3.1-8B, British English in the paper).

  count     Neighbour count (App B). Cumulative centroids over the first k
            neighbours in generation order, k = 1..5. Per run: mean of positions
            1..k. "avg": each position averaged across runs, then positions 1..k
            averaged. Mean-centred RSA at every layer.

  position  Single neighbour positions (App F). Each position p = 1..5 on its
            own. Per run: position p. "avg": position p averaged across runs.
            Mean-centred RSA at every layer.

  median    Median vs mean aggregation (App E). Within each run the neighbour
            vectors are combined with the component-wise median instead of the
            mean; the run-averaged centroid is the mean of the per-run medians.
            RSA (raw and mean-centred) at every layer.

Usage
-----
    python analyses/aggregation.py count    --lang en --model llama8b --sweep all
    python analyses/aggregation.py position --lang en --model llama8b --sweep all
    python analyses/aggregation.py median   --lang en --model llama8b --sweep all

Output  results/<lang>/<model>/<sweep>/aggregation/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from common import (config, human_matrix, layers_of, mean_centre,
                    neighbours_by_position, neighbours_for_median,
                    pearson_rsm, run_dir_for, upper_triangle)
from rsa import rsa_for_tag

N_POS = config.N_NEIGHBOURS


def centred_tau(vectors: dict, human: np.ndarray, order: list) -> float:
    present = [c for c in order if c in vectors and np.linalg.norm(vectors[c]) > 0]
    idx = [order.index(c) for c in present]
    rsm = pearson_rsm(mean_centre({c: vectors[c] for c in present}), present)
    return kendalltau(upper_triangle(rsm), upper_triangle(human[np.ix_(idx, idx)]))[0]


def _per_position(run_dir: Path, order: list) -> tuple:
    """{colour: {run: {pos: (n_layers, d) float64}}} and the sorted run list."""
    raw = neighbours_by_position(run_dir, layer_idx=None)
    data = {c: {r: {p: v.astype(np.float64) for p, v in pos.items()}
                for r, pos in raw.get(c.replace(" ", "_"), {}).items()} for c in order}
    runs = sorted({r for c in data for r in data[c]})
    return data, runs


# =============================================================================
# APPENDIX B / F
# =============================================================================

def neighbour_count(run_dir: Path, lang: str) -> pd.DataFrame:
    human, order = human_matrix(lang)
    data, runs = _per_position(run_dir, order)
    rows = []
    for li, layer in enumerate(layers_of(run_dir)):
        for k in range(1, N_POS + 1):
            for run in runs:
                vecs = {c: np.mean([data[c][run][p][li] for p in range(1, k + 1)
                                    if p in data[c].get(run, {})], axis=0)
                        for c in order if any(p in data[c].get(run, {}) for p in range(1, k + 1))}
                rows.append({"tag": f"run{run}", "k": k, "layer": layer,
                             "tau": centred_tau(vecs, human, order)})
            vecs = {}
            for c in order:
                per_pos = [np.mean([data[c][r][p][li] for r in runs if p in data[c].get(r, {})], axis=0)
                           for p in range(1, k + 1)
                           if any(p in data[c].get(r, {}) for r in runs)]
                if per_pos:
                    vecs[c] = np.mean(per_pos, axis=0)
            rows.append({"tag": "avg", "k": k, "layer": layer, "tau": centred_tau(vecs, human, order)})
    return pd.DataFrame(rows)


def neighbour_position(run_dir: Path, lang: str) -> pd.DataFrame:
    human, order = human_matrix(lang)
    data, runs = _per_position(run_dir, order)
    rows = []
    for li, layer in enumerate(layers_of(run_dir)):
        for p in range(1, N_POS + 1):
            for run in runs:
                vecs = {c: data[c][run][p][li] for c in order if p in data[c].get(run, {})}
                rows.append({"tag": f"run{run}", "position": p, "layer": layer,
                             "tau": centred_tau(vecs, human, order)})
            vecs = {c: np.mean([data[c][r][p][li] for r in runs if p in data[c].get(r, {})], axis=0)
                    for c in order if any(p in data[c].get(r, {}) for r in runs)}
            rows.append({"tag": "avg", "position": p, "layer": layer,
                         "tau": centred_tau(vecs, human, order)})
    return pd.DataFrame(rows)


# =============================================================================
# APPENDIX E
# =============================================================================

def median_aggregation(run_dir: Path, lang: str, out_dir: Path) -> pd.DataFrame:
    human, order = human_matrix(lang)
    cent_dir = out_dir / "median_centroids"
    cent_dir.mkdir(parents=True, exist_ok=True)
    runs_found = set()
    for colour, runs in neighbours_for_median(run_dir).items():
        medians = []
        for run in sorted(runs):
            m = np.median(runs[run], axis=0)
            np.save(cent_dir / f"{colour}_CENTROID_run{run}.npy", m)
            medians.append(m)
            runs_found.add(run)
        np.save(cent_dir / f"{colour}_CENTROID_avg.npy", np.mean(medians, axis=0))

    layers = layers_of(run_dir)
    frames = [rsa_for_tag(f"run{r}", cent_dir, human, order, layers) for r in sorted(runs_found)]
    frames.append(rsa_for_tag("avg", cent_dir, human, order, layers))
    df = pd.concat(frames, ignore_index=True)
    return df[df.colour == "OVERALL"].drop(columns=["method", "significant"])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregation checks (App B, E, F)")
    parser.add_argument("check", choices=["count", "position", "median"])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args()

    run_dir = run_dir_for(args.lang, args.model, args.sweep)
    out_dir = run_dir / "aggregation"
    out_dir.mkdir(exist_ok=True)
    print(f"{args.check}: {args.lang} {args.model} {args.sweep}")

    if args.check == "count":
        df = neighbour_count(run_dir, args.lang)
        df.to_csv(out_dir / "neighbour_count.csv", index=False)
        key = "k"
    elif args.check == "position":
        df = neighbour_position(run_dir, args.lang)
        df.to_csv(out_dir / "neighbour_position.csv", index=False)
        key = "position"
    else:
        df = median_aggregation(run_dir, args.lang, out_dir)
        df.to_csv(out_dir / "median_rsa.csv", index=False)
        c = df[(df.tag == "avg") & (df.condition == "centred")]
        best = c.loc[c.tau.idxmax()]
        print(f"  median, centred, avg: l* = {int(best.layer)}  tau = {best.tau:.4f}")
        print(f"  Output: {out_dir}")
        return

    avg = df[df.tag == "avg"]
    for v in sorted(avg[key].unique()):
        d = avg[avg[key] == v]
        best = d.loc[d.tau.idxmax()]
        print(f"  {key}={v}: best layer {int(best.layer)}  tau {best.tau:.4f}")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
