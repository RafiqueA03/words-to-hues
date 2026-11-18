"""
RSA robustness to the similarity measures and the RSA statistic
(paper appendix "Robustness to Similarity Measures and RSA Statistic").

  metric  Model side: the layer-wise RSA is recomputed with a cosine-similarity
          RSM in place of Pearson correlation (raw and mean-centred, run-averaged
          centroids). Reports, per condition, the peak tau and layer under both
          metrics and the absolute change in peak tau.

  grid    Human side x RSA statistic (18-cell grid, Llama-3.1-8B in the paper):
          human similarity 1 - sqrt(JSD) (reported), 1 - JSD and 1 - Hellinger,
          crossed with Kendall's tau (reported) and Spearman's rho. Model RSM is
          fixed: mean-centred Pearson on run-averaged centroids. Reports the peak
          value and layer of every cell.

Human-similarity files for the grid (data/human_similarity/):
    <Language>_similarity_matrix_1mjsd.csv       1 - JSD
    <Language>_similarity_matrix_hellinger.csv   1 - Hellinger distance
    <Language>_similarity_matrix_1msqrtjsd.csv   1 - sqrt(JSD); if absent, the
                                                 main <Language>_similarity_matrix.csv
Each file must already contain similarities (1 - divergence/distance).

Usage
-----
    python analyses/rsa_robustness.py metric --lang en --model llama8b --sweep all
    python analyses/rsa_robustness.py grid   --lang en --model llama8b --sweep all

Output  results/<lang>/<model>/<sweep>/robustness/
"""

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from common import (config, human_matrix, layers_of, load_avg_centroids,
                    mean_centre, pearson_rsm, run_dir_for, upper_triangle)

HUMAN_VARIANTS = ["1msqrtjsd", "1mjsd", "hellinger"]
STATISTICS     = {"kendall": kendalltau, "spearman": spearmanr}


def cosine_rsm(vectors: dict, order: list) -> np.ndarray:
    n = len(order)
    rsm = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = vectors[order[i]], vectors[order[j]]
            denom = np.linalg.norm(vi) * np.linalg.norm(vj)
            rsm[i, j] = rsm[j, i] = float(np.dot(vi, vj) / denom) if denom else 0.0
    return rsm


# =============================================================================
# MODEL SIDE: PEARSON vs COSINE
# =============================================================================

def metric_comparison(run_dir: Path, lang: str) -> tuple:
    human, order = human_matrix(lang)
    rows = []
    for idx, layer in enumerate(layers_of(run_dir)):
        vecs = load_avg_centroids(run_dir, order, idx)
        present = [c for c in order if c in vecs]
        sel = [order.index(c) for c in present]
        h_upper = upper_triangle(human[np.ix_(sel, sel)])
        for condition in ("raw", "centred"):
            v = mean_centre({c: vecs[c] for c in present}) if condition == "centred" else vecs
            for metric, build in (("pearson", pearson_rsm), ("cosine", cosine_rsm)):
                tau = kendalltau(upper_triangle(build(v, present)), h_upper)[0]
                rows.append({"layer": layer, "condition": condition, "metric": metric, "tau": tau})
    layerwise = pd.DataFrame(rows)

    summary = []
    for condition in ("raw", "centred"):
        peak = {}
        for metric in ("pearson", "cosine"):
            d = layerwise[(layerwise.condition == condition) & (layerwise.metric == metric)]
            best = d.loc[d.tau.idxmax()]
            peak[metric] = (int(best.layer), float(best.tau))
        summary.append({"condition": condition,
                        "pearson_layer": peak["pearson"][0], "pearson_tau": peak["pearson"][1],
                        "cosine_layer": peak["cosine"][0], "cosine_tau": peak["cosine"][1],
                        "abs_delta_peak_tau": abs(peak["cosine"][1] - peak["pearson"][1]),
                        "same_peak_layer": peak["pearson"][0] == peak["cosine"][0]})
    return pd.DataFrame(summary), layerwise


# =============================================================================
# HUMAN SIDE x RSA STATISTIC (18-cell grid)
# =============================================================================

def _normalise_accents(name: str) -> str:
    """Name matching for the variant files, some of which lost accents."""
    name = str(name).replace("_", " ").lower().strip()
    return "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")


def variant_file(lang: str, variant: str) -> Path:
    stem = config.LANGUAGES[lang]["file_stem"]
    path = config.HUMAN_SIM_DIR / f"{stem}_similarity_matrix_{variant}.csv"
    if not path.exists() and variant == "1msqrtjsd":
        path = config.HUMAN_SIM_DIR / f"{stem}_similarity_matrix.csv"
    return path


def similarity_grid(run_dir: Path, lang: str) -> pd.DataFrame:
    colours = config.LANGUAGES[lang]["colours"]
    layer_rsms = {}
    for idx, layer in enumerate(layers_of(run_dir)):
        vecs = load_avg_centroids(run_dir, [c.replace("_", " ") for c in colours], idx)
        present = list(vecs)
        if present:
            layer_rsms[layer] = (pearson_rsm(mean_centre(vecs), present), present)

    rows = []
    for variant in HUMAN_VARIANTS:
        path = variant_file(lang, variant)
        if not path.exists():
            print(f"  skipped {variant}: {path.name} not found")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
        df.index   = [_normalise_accents(i) for i in df.index]
        df.columns = [_normalise_accents(c) for c in df.columns]
        for stat, fn in STATISTICS.items():
            best_val, best_layer = None, None
            for layer, (rsm, present) in layer_rsms.items():
                keys = [_normalise_accents(c) for c in present]
                h = df.loc[keys, keys].values.astype(np.float64)
                val = fn(upper_triangle(rsm), upper_triangle(h))[0]
                if best_val is None or val > best_val:
                    best_val, best_layer = val, layer
            rows.append({"language": config.LANGUAGES[lang]["label"], "human_variant": variant,
                         "human_file": path.name, "statistic": stat,
                         "peak_value": best_val, "best_layer": best_layer})
            print(f"  {variant:10s} {stat:9s} peak {best_val:.4f} at layer {best_layer}")
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RSA robustness checks")
    parser.add_argument("check", choices=["metric", "grid"])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args()

    run_dir = run_dir_for(args.lang, args.model, args.sweep)
    out_dir = run_dir / "robustness"
    out_dir.mkdir(exist_ok=True)
    print(f"{args.check}: {args.lang} {args.model} {args.sweep}")

    if args.check == "metric":
        summary, layerwise = metric_comparison(run_dir, args.lang)
        summary.to_csv(out_dir / "pearson_vs_cosine.csv", index=False)
        layerwise.to_csv(out_dir / "pearson_vs_cosine_layerwise.csv", index=False)
        for _, r in summary.iterrows():
            print(f"  {r.condition:8s} Pearson l*={r.pearson_layer} tau={r.pearson_tau:.6f} | "
                  f"cosine l*={r.cosine_layer} tau={r.cosine_tau:.6f} | "
                  f"|delta|={r.abs_delta_peak_tau:.6f}")
    else:
        similarity_grid(run_dir, args.lang).to_csv(out_dir / "similarity_grid.csv", index=False)
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
