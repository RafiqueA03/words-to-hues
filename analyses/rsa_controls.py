"""
RSA correspondence controls (paper Appendix H).

Each control tests whether RSA alignment depends on the correct correspondence
between colour terms and their representations. All run at the best layer l*
(override with --layer), with mean-centred Pearson RSMs and Kendall's tau.

  permutation  Label permutation (G.1). One random permutation is applied to the
               rows and columns of the human matrix; tau is recomputed against
               the fixed model RSM. 10,000 permutations, seed 0.
               p = (#null >= observed + 1) / (n + 1).

  scramble     Cross-colour scrambling (G.2). Each colour's centroid is rebuilt
               from 5 neighbour vectors drawn (without replacement) from OTHER
               colours. Neighbour vectors are pooled by word across runs.
               1,000 iterations, seed 0. Same p-value formula as above.

  ablation     Graded real-content ablation (G.3). Each centroid uses n of the
               colour's own neighbour vectors and 5 - n from other colours,
               n = 1..4 (pooled over runs and positions). 1,000 iterations per
               level, seed n. Reports the distribution of tau at each level.

Usage
-----
    python analyses/rsa_controls.py permutation --lang en --model llama8b --sweep all
    python analyses/rsa_controls.py scramble    --lang en --model llama70b --sweep fine
    python analyses/rsa_controls.py ablation    --lang en --model llama8b --sweep all

Output  results/<lang>/<model>/<sweep>/controls/<control>.csv  (+ null distribution)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from common import (best_layer, centred_rsm, human_matrix, load_avg_centroids,
                    neighbours_by_position, neighbours_by_word, normalise,
                    resolve_layer, run_dir_for, upper_triangle)

N_PERMUTATIONS = 10_000
N_ITERATIONS   = 1_000
N_TOTAL        = 5


def _restrict(human: np.ndarray, order: list, keep: list) -> np.ndarray:
    idx = [order.index(c) for c in keep]
    return human[np.ix_(idx, idx)]


# =============================================================================
# G.1  LABEL PERMUTATION
# =============================================================================

def label_permutation(run_dir: Path, lang: str, layer_idx: int, seed: int = 0) -> tuple:
    human, order = human_matrix(lang)
    vecs  = load_avg_centroids(run_dir, order, layer_idx)
    keep  = [c for c in order if c in vecs]
    human = _restrict(human, order, keep)
    model_upper = upper_triangle(centred_rsm(vecs, keep))

    tau_obs = kendalltau(upper_triangle(human), model_upper)[0]
    rng, n = np.random.default_rng(seed), len(keep)
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        perm = rng.permutation(n)
        null[i] = kendalltau(upper_triangle(human[np.ix_(perm, perm)]), model_upper)[0]

    result = {"n_colours": n, "tau_obs": tau_obs, "null_mean": null.mean(),
              "null_sd": null.std(), "null_max": null.max(),
              "p_value": (np.sum(null >= tau_obs) + 1) / (N_PERMUTATIONS + 1)}
    return result, null


# =============================================================================
# G.2  CROSS-COLOUR SCRAMBLING
# =============================================================================

def cross_colour_scramble(run_dir: Path, lang: str, layer_idx: int, seed: int = 0) -> tuple:
    human, order = human_matrix(lang)
    pool_raw = neighbours_by_word(run_dir, layer_idx)                 # keys: file names
    by_norm  = {normalise(k): k for k in pool_raw}
    common   = [c for c in order if c in by_norm]
    keys     = [by_norm[c] for c in common]
    human    = _restrict(human, order, common)
    human_upper = upper_triangle(human)

    # word-level vectors: mean over the runs in which each word was generated
    word_level = {k: {w: np.mean(v, axis=0) for w, v in pool_raw[k].items() if v} for k in keys}

    # Observed tau for this pooling (centroid = mean of word-level vectors).
    real = {c: np.mean(list(word_level[k].values()), axis=0) for c, k in zip(common, keys)}
    tau_real = kendalltau(upper_triangle(centred_rsm(real, common)), human_upper)[0]

    rng = np.random.default_rng(seed)
    null = np.empty(N_ITERATIONS)
    for it in range(N_ITERATIONS):
        scrambled = {}
        for c, k in zip(common, keys):
            others = np.stack([v for c2, k2 in zip(common, keys) if c2 != c
                               for v in word_level[k2].values()], axis=0)
            scrambled[c] = others[rng.choice(len(others), size=N_TOTAL, replace=False)].mean(axis=0)
        null[it] = kendalltau(upper_triangle(centred_rsm(scrambled, common)), human_upper)[0]

    result = {"n_colours": len(common), "tau_real": tau_real, "null_mean": null.mean(),
              "null_sd": null.std(), "null_max": null.max(),
              "p_value": (np.sum(null >= tau_real) + 1) / (N_ITERATIONS + 1)}
    return result, null


# =============================================================================
# G.3  GRADED REAL-CONTENT ABLATION
# =============================================================================

def graded_ablation(run_dir: Path, lang: str, layer_idx: int, base_seed: int = 0) -> tuple:
    human, order = human_matrix(lang)
    by_pos = neighbours_by_position(run_dir, layer_idx)
    # flat pool per colour: runs in order, positions 1..5 within each run
    flat = {}
    for c in order:
        runs = by_pos.get(c.replace(" ", "_"), {})
        flat[c] = [v.astype(np.float64) for r in sorted(runs)
                   for p in range(1, N_TOTAL + 1)
                   if p in runs[r] and np.linalg.norm(runs[r][p]) > 0
                   for v in [runs[r][p]]]
    human_upper = upper_triangle(human)

    rows, summary = [], []
    for n_real in (1, 2, 3, 4):
        rng = np.random.default_rng(base_seed + n_real)
        taus = np.empty(N_ITERATIONS)
        for it in range(N_ITERATIONS):
            cents = {}
            for c in order:
                own = flat[c]
                real = [own[i] for i in rng.choice(len(own), size=n_real, replace=False)]
                other = [v for c2 in order if c2 != c for v in flat[c2]]
                fake = [other[i] for i in rng.choice(len(other), size=N_TOTAL - n_real, replace=False)]
                cents[c] = np.mean(real + fake, axis=0)
            taus[it] = kendalltau(upper_triangle(centred_rsm(cents, order)), human_upper)[0]
        rows += [{"n_real": n_real, "iteration": i + 1, "tau": t} for i, t in enumerate(taus)]
        summary.append({"n_real": n_real, "n_total": N_TOTAL, "tau_mean": taus.mean(),
                        "tau_sd": taus.std(), "tau_min": taus.min(), "tau_max": taus.max()})
        print(f"  n_real={n_real}: mean tau {taus.mean():.3f} (SD {taus.std():.3f})")
    return pd.DataFrame(summary), pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RSA correspondence controls (Appendix H)")
    parser.add_argument("control", choices=["permutation", "scramble", "ablation"])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--layer", type=int, help="default: best RSA layer l*")
    args = parser.parse_args()

    run_dir = run_dir_for(args.lang, args.model, args.sweep)
    layer, layer_idx = resolve_layer(run_dir, args.layer)
    out_dir = run_dir / "controls"
    out_dir.mkdir(exist_ok=True)
    print(f"{args.control}: {args.lang} {args.model} {args.sweep}  layer {layer}")

    if args.control == "ablation":
        summary, draws = graded_ablation(run_dir, args.lang, layer_idx)
        summary.insert(0, "layer", layer)
        summary.to_csv(out_dir / "ablation.csv", index=False)
        draws.to_csv(out_dir / "ablation_draws.csv", index=False)
    else:
        fn = label_permutation if args.control == "permutation" else cross_colour_scramble
        result, null = fn(run_dir, args.lang, layer_idx)
        pd.DataFrame([{"layer": layer, **result}]).to_csv(out_dir / f"{args.control}.csv", index=False)
        np.save(out_dir / f"{args.control}_null.npy", null)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in result.items()))
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
