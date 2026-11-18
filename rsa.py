"""
Step 3 - Layer-wise Representational Similarity Analysis (RSA).

For each layer, a model representational similarity matrix (RSM) is built from
Pearson correlations between colour-term centroids, and compared with the human
colour-term similarity matrix using Kendall's tau over the upper triangles.
Paper: Section 3.4 and Section 4.1.

  conditions : "raw" and "centred" (centroids mean-centred across terms at each
               layer, to reduce anisotropy). The paper reports "centred".
  tags       : one RSA per run ("run1".."run5") and one on the run-averaged
               centroids ("avg"). The best layer l* is the layer with the
               highest centred tau on the "avg" centroids.

Usage
-----
    python rsa.py --lang en --model llama8b --sweep all
    python rsa.py --all

Input   results/<lang>/<model>/<sweep>/centroids/, run_info.json
        data/human_similarity/<Language>_similarity_matrix.csv
Output  results/<lang>/<model>/<sweep>/rsa/
            per_run/run<N>/rsa_run<N>_full.csv     overall + per-colour tau
            per_run/all_runs_combined.csv
            averaged/rsa_avg_full.csv
            cross_run_summary.csv                  mean/SD/min/max tau over runs
            best_layers.json                       l* per condition ("avg")
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

import config

CONDITIONS = ["raw", "centred"]
REPORTED_CONDITION = "centred"


# =============================================================================
# INPUTS
# =============================================================================

def normalise(name: str) -> str:
    """Match colour names across files: 'dark_blue' == 'Dark blue'."""
    return str(name).replace("_", " ").lower().strip()


def human_similarity_file(lang: str) -> Path:
    return config.HUMAN_SIM_DIR / f"{config.LANGUAGES[lang]['file_stem']}_similarity_matrix.csv"


def load_human_similarity(path: Path, colours: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Human similarity sub-matrix for the lexicon, in lexicon order."""
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    df.columns = [normalise(c) for c in df.columns]
    df.index   = [normalise(i) for i in df.index]
    order   = [normalise(c) for c in colours]
    present = [c for c in order if c in df.index]
    missing = [c for c in order if c not in df.index]
    if missing:
        print(f"  Human matrix: not found, dropped: {missing}")
    if not present:
        raise ValueError(f"No lexicon colours found in {path}")
    return df.loc[present, present].values.astype(np.float64), present


def detect_runs(centroid_dir: Path, colour: str) -> List[int]:
    runs = []
    for f in centroid_dir.glob(f"{colour.replace(' ', '_')}_CENTROID_run*.npy"):
        part = f.stem.split("_run")[-1]
        if part.isdigit():
            runs.append(int(part))
    return sorted(runs)


def load_centroids(centroid_dir: Path, colours: List[str], tag: str) -> Dict[str, np.ndarray]:
    """All centroids for one tag, shape (n_layers, hidden_size) per colour."""
    out = {}
    for colour in colours:
        path = centroid_dir / f"{colour.replace(' ', '_')}_CENTROID_{tag}.npy"
        if path.exists():
            out[colour] = np.load(path).astype(np.float64)
        else:
            print(f"  Missing: {path.name}")
    return out


# =============================================================================
# RSA
# =============================================================================

def mean_centre(vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = np.mean(np.stack(list(vectors.values())), axis=0)
    return {c: v - mu for c, v in vectors.items()}


def pearson_rsm(vectors: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
    n = len(order)
    rsm = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rsm[i, j] = rsm[j, i] = pearsonr(vectors[order[i]], vectors[order[j]])[0]
    return rsm


def upper_triangle(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], k=1)]


def compare(model_rsm: np.ndarray, human: np.ndarray, order: List[str]) -> List[Dict]:
    """Overall tau, plus a per-colour tau over each term's row."""
    tau, p = kendalltau(upper_triangle(model_rsm), upper_triangle(human))
    rows = [{"colour": "OVERALL", "tau": tau, "p_value": p}]
    for i, colour in enumerate(order):
        mrow = np.delete(model_rsm[i], i)
        hrow = np.delete(human[i], i)
        t, pv = kendalltau(mrow, hrow)
        rows.append({"colour": colour, "tau": t, "p_value": pv})
    return rows


def rsa_for_tag(tag: str, centroid_dir: Path, human: np.ndarray,
                order: List[str], layers: List[int]) -> pd.DataFrame:
    """RSA at every layer and condition for one tag ('run1', ..., 'avg')."""
    centroids = load_centroids(centroid_dir, order, tag)
    rows = []
    for layer_idx, layer in enumerate(layers):
        vecs = {c: centroids[c][layer_idx] for c in order
                if c in centroids and np.linalg.norm(centroids[c][layer_idx]) > 0}
        if not vecs:
            continue
        present = [c for c in order if c in vecs]
        if len(present) < len(order):
            print(f"  [{tag}] layer {layer}: using {len(present)}/{len(order)} colours")
        idx = [order.index(c) for c in present]
        human_sub = human[np.ix_(idx, idx)]
        for condition in CONDITIONS:
            v = mean_centre(vecs) if condition == "centred" else vecs
            for r in compare(pearson_rsm(v, present), human_sub, present):
                rows.append({"tag": tag, "layer": layer, "condition": condition,
                             "method": "centroid", **r,
                             "significant": r["p_value"] < 0.05})
    return pd.DataFrame(rows)


# =============================================================================
# ONE EXTRACTION RUN
# =============================================================================

def run_rsa(run_dir: Path, lang: str) -> None:
    centroid_dir = run_dir / "centroids"
    out_root     = run_dir / "rsa"
    info_file    = run_dir / "run_info.json"
    if not centroid_dir.exists() or not info_file.exists():
        print(f"  Missing centroids/ or run_info.json in {run_dir} - skipped")
        return

    layers = json.loads(info_file.read_text(encoding="utf-8"))["layers"]
    human, order = load_human_similarity(human_similarity_file(lang),
                                         config.LANGUAGES[lang]["colours"])
    runs = detect_runs(centroid_dir, order[0])
    print(f"  Layers {layers[0]}-{layers[-1]} ({len(layers)})  |  colours {len(order)}  |  runs {runs}")

    per_run = []
    for run in runs:
        tag = f"run{run}"
        df = rsa_for_tag(tag, centroid_dir, human, order, layers)
        if df.empty:
            continue
        d = out_root / "per_run" / tag
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f"rsa_{tag}_full.csv", index=False, encoding="utf-8-sig")
        per_run.append(df)

    avg = rsa_for_tag("avg", centroid_dir, human, order, layers)
    (out_root / "averaged").mkdir(parents=True, exist_ok=True)
    avg.to_csv(out_root / "averaged" / "rsa_avg_full.csv", index=False, encoding="utf-8-sig")

    # Cross-run summary (SD is the population SD, as in the original analysis).
    if per_run:
        combined = pd.concat(per_run, ignore_index=True)
        combined.to_csv(out_root / "per_run" / "all_runs_combined.csv",
                        index=False, encoding="utf-8-sig")
        runs_overall = combined[combined["colour"] == "OVERALL"]
        avg_overall  = avg[avg["colour"] == "OVERALL"]
        summary = []
        for condition in CONDITIONS:
            for layer in layers:
                t = runs_overall[(runs_overall["condition"] == condition) &
                                 (runs_overall["layer"] == layer)]["tau"].values
                a = avg_overall[(avg_overall["condition"] == condition) &
                                (avg_overall["layer"] == layer)]["tau"].values
                summary.append({
                    "condition": condition, "method": "centroid", "layer": layer,
                    "mean_tau_runs": float(np.mean(t)) if len(t) else np.nan,
                    "std_tau_runs":  float(np.std(t))  if len(t) else np.nan,
                    "min_tau_runs":  float(np.min(t))  if len(t) else np.nan,
                    "max_tau_runs":  float(np.max(t))  if len(t) else np.nan,
                    "tau_avg_hs":    float(a[0]) if len(a) else np.nan,
                })
        pd.DataFrame(summary).to_csv(out_root / "cross_run_summary.csv",
                                     index=False, encoding="utf-8-sig")

    # Best layer per condition, on the run-averaged centroids.
    best = {}
    avg_overall = avg[avg["colour"] == "OVERALL"]
    for condition in CONDITIONS:
        c = avg_overall[avg_overall["condition"] == condition]
        if not c.empty:
            row = c.loc[c["tau"].idxmax()]
            best[condition] = {"layer": int(row["layer"]), "tau": float(row["tau"]),
                               "p_value": float(row["p_value"])}
    with open(out_root / "best_layers.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    if REPORTED_CONDITION in best:
        b = best[REPORTED_CONDITION]
        print(f"  Best layer ({REPORTED_CONDITION}, avg): l* = {b['layer']}  tau = {b['tau']:.3f}")
    print(f"  Output: {out_root}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lang", choices=sorted(config.LANGUAGES))
    parser.add_argument("--model", choices=sorted(config.MODELS))
    parser.add_argument("--sweep", choices=["all", "coarse", "fine"])
    parser.add_argument("--all", action="store_true", help="every run with centroids")
    args = parser.parse_args()

    if args.all:
        targets = [(p.parent.relative_to(config.RESULTS_DIR).parts[0], p.parent)
                   for p in sorted(config.RESULTS_DIR.glob("*/*/*/centroids"))]
    elif args.lang and args.model and args.sweep:
        targets = [(args.lang, config.output_dir(args.lang, args.model, args.sweep))]
    else:
        parser.error("give --lang, --model and --sweep, or use --all")

    if not targets:
        print(f"No centroids found under {config.RESULTS_DIR}")
    for lang, run_dir in targets:
        print(f"\n{'=' * 70}\n{run_dir.relative_to(config.ROOT_DIR)}\n{'=' * 70}")
        run_rsa(run_dir, lang)


if __name__ == "__main__":
    main()
