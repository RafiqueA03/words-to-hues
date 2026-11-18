"""
Direct-term RSA baselines (paper Appendix D; Llama-3.1-8B in the paper).

Compares the lexical-neighbourhood centroid with two representations of the
target colour term itself:

  prompt target     mean hidden state of the target term's tokens in the
                    question line of the prompt ("... for {colour}?").
                    Requires extraction with --prompt-states
                    (hidden_states/<colour>_hfill_occ1_run<r>.npy).
  generated target  mean hidden state over the generated tokens before the
                    first '-' token, i.e. the repeated target term at the start
                    of the answer ("{colour} - ...").

Each is averaged across runs ("avg"), and layer-wise RSA is computed exactly as
in rsa.py (raw and mean-centred). The centroid results are read from rsa/.

Usage
-----
    python analyses/direct_term.py --lang en --model llama8b --sweep all

Output  results/<lang>/<model>/<sweep>/direct_term/
            prompt_target/, generated_target/     vectors (<colour>_CENTROID_<tag>.npy)
            rsa_layerwise.csv                     tau per method x layer x condition
            summary.csv                           best layer and tau per method
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import (_entries_with_states, config, human_matrix, layers_of,
                    run_dir_for)
from rsa import rsa_for_tag


def generated_target_vectors(run_dir: Path) -> dict:
    """{colour: {run: (n_layers, d)}}: mean state over tokens before the first '-'."""
    out = defaultdict(dict)
    for e, _, hs in _entries_with_states(run_dir):
        idx = []
        for i, (tok, _) in enumerate(e["token_probs"]):
            if tok.strip() == "-":
                break
            idx.append(i)
        if idx:
            out[e["colour"]][e["run"]] = np.mean(hs[idx], axis=0)
    return out


def prompt_target_vectors(run_dir: Path, colours: list) -> dict:
    """{colour: {run: (n_layers, d)}} from hidden_states/<colour>_hfill_occ1_run<r>.npy."""
    out = defaultdict(dict)
    for colour in colours:
        for f in (run_dir / "hidden_states").glob(f"{colour}_hfill_occ1_run*.npy"):
            run = f.stem.split("_run")[-1]
            if run.isdigit():
                out[colour][int(run)] = np.load(f)
    return out


def save_tags(vectors: dict, out_dir: Path) -> list:
    """Save per-run and run-averaged vectors with the centroid file naming."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = set()
    for colour, per_run in vectors.items():
        for run, v in per_run.items():
            np.save(out_dir / f"{colour}_CENTROID_run{run}.npy", v)
            runs.add(run)
        np.save(out_dir / f"{colour}_CENTROID_avg.npy",
                np.mean([per_run[r] for r in sorted(per_run)], axis=0))
    return sorted(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-term RSA baselines (Appendix D)")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args()

    run_dir = run_dir_for(args.lang, args.model, args.sweep)
    out = run_dir / "direct_term"
    human, order = human_matrix(args.lang)
    layers = layers_of(run_dir)
    colours = config.LANGUAGES[args.lang]["colours"]
    print(f"direct-term baselines: {args.lang} {args.model} {args.sweep}")

    frames = []
    for method, vecs in (("prompt_target", prompt_target_vectors(run_dir, colours)),
                         ("generated_target", generated_target_vectors(run_dir))):
        if not vecs:
            print(f"  {method}: no vectors found - skipped"
                  + (" (run extraction with --prompt-states)" if method == "prompt_target" else ""))
            continue
        save_tags(vecs, out / method)
        df = rsa_for_tag("avg", out / method, human, order, layers)
        frames.append(df[df.colour == "OVERALL"].assign(method=method))

    centroid = pd.read_csv(run_dir / "rsa" / "averaged" / "rsa_avg_full.csv")
    frames.append(centroid[centroid.colour == "OVERALL"].assign(method="centroid"))
    layerwise = pd.concat(frames, ignore_index=True)[["method", "tag", "layer", "condition", "tau", "p_value"]]
    layerwise.to_csv(out / "rsa_layerwise.csv", index=False)

    rows = []
    for (method, condition), d in layerwise.groupby(["method", "condition"]):
        best = d.loc[d.tau.idxmax()]
        rows.append({"method": method, "condition": condition,
                     "best_layer": int(best.layer), "tau": float(best.tau)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    for _, r in summary[summary.condition == "centred"].iterrows():
        print(f"  {r.method:17s} centred: best layer {r.best_layer}  tau {r.tau:.3f}")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
