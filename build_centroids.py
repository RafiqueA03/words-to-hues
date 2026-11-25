"""
Step 2 - Build lexical-neighbourhood centroids from the extraction outputs.

For every colour term and run, the generated response
    "<source> - <neighbour1>, <neighbour2>, ..., <neighbour5>"
is parsed, each neighbour is located in the generated tokens, and its hidden
state is the mean over its tokens. The per-run centroid is the mean over the
neighbours found. Paper: Section 3.3.

Run-averaged centroid (as in the original analysis code)
------------------------------------------------------
Neighbours are pooled across runs by word: each distinct neighbour word is first
averaged over the runs in which it was generated, and the centroid is the mean
of these word vectors. A word generated in several runs therefore counts once.

Usage
-----
    python build_centroids.py --lang en --model llama8b --sweep all
    python build_centroids.py --all              # every extraction run found
    python build_centroids.py --all --overwrite  # rebuild existing outputs

Input   results/<lang>/<model>/<sweep>/raw_responses.txt, hidden_states/
Output  results/<lang>/<model>/<sweep>/centroids/
            <colour>_CENTROID_run<r>.npy   per-run centroid  (n_layers, hidden_size)
            <colour>_CENTROID_avg.npy      run-averaged centroid
            neighbours.csv                 every parsed neighbour, per run
            summary.json                   counts of parsed / skipped entries
"""

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

# Tokens that never count as part of a neighbour word.
PUNCTUATION_TOKENS = (',', '.', '-', '\n', ' ')


# =============================================================================
# READING THE RAW RESPONSES
# =============================================================================

def read_raw_responses(raw_file: Path) -> List[Dict]:
    """Parse raw_responses.txt into one entry per (run, colour)."""
    content = raw_file.read_text(encoding="utf-8-sig")
    entries = []
    for block in content.split("-" * 70):
        if "RUN:" not in block:
            continue
        run_m    = re.search(r"RUN: (\d+)", block)
        colour_m = re.search(r"COLOUR: (\S+)", block)
        resp_m   = re.search(r"RESPONSE:\n(.*?)\n\nTOKEN PROBABILITIES:", block, re.DOTALL)
        tok_m    = re.search(r"TOKEN PROBABILITIES:\n(.*?)(?:\n\nHIDDEN|\n\n|$)", block, re.DOTALL)
        if not (run_m and colour_m and resp_m and tok_m):
            continue
        token_probs = []
        for line in tok_m.group(1).strip().split("\n"):
            m = re.search(r"\[\s*\d+\]\s+(.+?)\s+(\d+\.\d+)$", line)
            if m:
                token_probs.append((ast.literal_eval(m.group(1)), float(m.group(2))))
        entries.append({
            "run":         int(run_m.group(1)),
            "colour":      colour_m.group(1),
            "response":    resp_m.group(1).strip(),
            "token_probs": token_probs,
        })
    return entries


def parse_response(response: str) -> Tuple[Optional[str], List[str]]:
    """Split '<source> - a, b, c, d, e' (first line only)."""
    first_line = response.strip().split("\n")[0].strip()
    m = re.search(r"^(\S+)\s+-\s+(.+)$", first_line)
    if not m:
        return None, []
    return m.group(1), [s.strip() for s in m.group(2).split(",")]


# =============================================================================
# LOCATING NEIGHBOURS IN THE GENERATED TOKENS
# =============================================================================

def parse_neighbour_list(response: str) -> Tuple[Optional[str], List[str]]:
    """Base-model responses: a plain comma-separated list (no '<source> -')."""
    first_line = response.strip().split("\n")[0].strip()
    neighbours = [s.strip() for s in first_line.split(",") if s.strip()]
    return ("", neighbours) if neighbours else (None, [])


def map_neighbours(
    neighbours: List[str],
    token_probs: List[Tuple[str, float]],
    hidden_states: np.ndarray,
    skip_source_prefix: bool = True,
) -> Dict[str, Dict]:
    """Return, per neighbour word, its token indices and mean hidden state.

    The search starts after the first ' -' (skipping the repeated source term;
    set skip_source_prefix=False for base-model lists) and moves left to right,
    so each word is matched after the previous one. Repeated words keep their
    first occurrence. If a word is not found as written, it is searched without
    spaces (e.g. 'sky blue' vs 'skyblue').
    """
    token_text  = "".join(t for t, _ in token_probs)
    first_dash  = token_text.find(" -") if skip_source_prefix else -1
    current_pos = first_dash + 2 if first_dash != -1 else 0
    zero        = np.zeros(hidden_states.shape[1:])

    result: Dict[str, Dict] = {}
    for word in neighbours:
        if word in result:
            continue
        start = token_text.find(word, current_pos)
        if start == -1:
            start = token_text.find(word.replace(" ", ""), current_pos)
        if start == -1:
            result[word] = {"found": False, "token_indices": [], "tokens": [],
                            "confidence": 0.0, "hidden_state": zero}
            continue
        end = start + len(word)

        indices, tokens, probs = [], [], []
        pos = 0
        for i, (tok, prob) in enumerate(token_probs):
            tok_end = pos + len(tok)
            if pos < end and tok_end > start and tok.strip() and tok not in PUNCTUATION_TOKENS:
                indices.append(i)
                tokens.append(tok)
                probs.append(prob)
            pos = tok_end

        if probs:
            confidence = float(np.prod(probs) ** (1 / len(probs)))
            hs = np.mean(hidden_states[indices], axis=0)
        else:
            # Kept as in the original code: counted as found, with a zero vector.
            print(f"      WARNING: '{word}' matched no usable tokens - zero vector used")
            confidence, hs = 0.0, zero

        result[word] = {"found": True, "token_indices": indices, "tokens": tokens,
                        "confidence": confidence, "hidden_state": hs}
        current_pos = end
    return result


# =============================================================================
# BUILDING CENTROIDS FOR ONE EXTRACTION RUN
# =============================================================================

def build_centroids(run_dir: Path, overwrite: bool = False, base_model: bool = False) -> None:
    """base_model=True: responses are plain neighbour lists (base-model prompt)."""
    parse = parse_neighbour_list if base_model else parse_response
    raw_file   = run_dir / "raw_responses.txt"
    hidden_dir = run_dir / "hidden_states"
    out_dir    = run_dir / "centroids"

    if not raw_file.exists():
        print(f"  No raw_responses.txt in {run_dir} - skipped")
        return
    if (out_dir / "summary.json").exists() and not overwrite:
        print(f"  Centroids already exist in {out_dir} - skipped (use --overwrite)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = read_raw_responses(raw_file)
    print(f"  Parsed {len(entries)} entries "
          f"({len({e['colour'] for e in entries})} colours x {len({e['run'] for e in entries})} runs)")

    per_colour: Dict[str, List[Dict]] = defaultdict(list)   # colour -> list of runs
    skipped: List[Dict] = []
    rows: List[List] = []

    for e in entries:
        colour, run = e["colour"], e["run"]
        source, neighbours = parse(e["response"])
        if source is None:
            skipped.append({"colour": colour, "run": run, "reason": "unparseable response"})
            continue
        hs_file = hidden_dir / f"{colour}_run{run}.npy"
        if not hs_file.exists():
            skipped.append({"colour": colour, "run": run, "reason": "hidden states missing"})
            continue
        hidden_states = np.load(hs_file)
        if len(e["token_probs"]) != hidden_states.shape[0]:
            skipped.append({"colour": colour, "run": run,
                            "reason": f"{len(e['token_probs'])} tokens vs "
                                      f"{hidden_states.shape[0]} hidden-state steps"})
            continue

        words = map_neighbours(neighbours, e["token_probs"], hidden_states,
                               skip_source_prefix=not base_model)
        per_colour[colour].append({"run": run, "words": words})

        for rank, (word, d) in enumerate(words.items(), 1):
            rows.append([run, colour, source, rank, word, d["found"],
                         d["token_indices"], d["tokens"], f"{d['confidence']:.6f}"])

        found = [d["hidden_state"] for d in words.values() if d["found"]]
        if found:
            np.save(out_dir / f"{colour}_CENTROID_run{run}.npy", np.mean(found, axis=0))
        else:
            skipped.append({"colour": colour, "run": run, "reason": "no neighbour found"})

    # Run-averaged centroid: pool by word across runs, then average the words.
    n_avg = 0
    for colour, runs in per_colour.items():
        word_vectors = []
        for word in sorted({w for r in runs for w in r["words"]}):
            vecs = [r["words"][word]["hidden_state"] for r in runs
                    if word in r["words"] and r["words"][word]["found"]]
            if vecs:
                word_vectors.append(np.mean(vecs, axis=0))
        if word_vectors:
            np.save(out_dir / f"{colour}_CENTROID_avg.npy", np.mean(word_vectors, axis=0))
            n_avg += 1

    with open(out_dir / "neighbours.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "colour", "source", "rank", "neighbour", "found",
                    "token_indices", "tokens", "confidence"])
        w.writerows(rows)

    summary = {
        "entries_parsed":      len(entries),
        "colours_with_avg":    n_avg,
        "skipped":             skipped,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  Averaged centroids: {n_avg} colours  |  skipped entries: {len(skipped)}")
    print(f"  Output: {out_dir}")


# =============================================================================
# MAIN
# =============================================================================

def find_all_run_dirs() -> List[Path]:
    """Every results/<lang>/<model>/<sweep>/ folder that has raw responses."""
    return sorted(p.parent for p in config.RESULTS_DIR.glob("*/*/*/raw_responses.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lang", choices=sorted(config.LANGUAGES))
    parser.add_argument("--model", choices=sorted(config.MODELS))
    parser.add_argument("--sweep", choices=["all", "coarse", "fine"])
    parser.add_argument("--all", action="store_true", help="process every extraction run found")
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing centroids")
    parser.add_argument("--base-model", action="store_true",
                        help="responses are plain neighbour lists (base-model comparison, App J)")
    args = parser.parse_args()

    if args.all:
        run_dirs = find_all_run_dirs()
    elif args.lang and args.model and args.sweep:
        run_dirs = [config.output_dir(args.lang, args.model, args.sweep)]
    else:
        parser.error("give --lang, --model and --sweep, or use --all")

    if not run_dirs:
        print(f"No extraction outputs found under {config.RESULTS_DIR}")
    for run_dir in run_dirs:
        print(f"\n{'=' * 70}\n{run_dir.relative_to(config.ROOT_DIR)}\n{'=' * 70}")
        build_centroids(run_dir, overwrite=args.overwrite, base_model=args.base_model)


if __name__ == "__main__":
    main()
