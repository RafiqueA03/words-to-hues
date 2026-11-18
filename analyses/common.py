"""
Shared helpers for the additional analyses (appendix controls, baselines, ...).

The main pipeline (extract_hidden_states.py -> build_centroids.py -> rsa.py ->
hue_geometry.py) must have been run first. These helpers reuse its parsing and
RSA functions so that every analysis works on the same matrices.

Lexical-neighbour vectors
-------------------------
The original analyses used two neighbour-extraction rules besides the one in
build_centroids.py. Both are reproduced here exactly:

  by word      (cross-colour scrambling): repeated words kept once; words not
               found, or matching no usable tokens, are skipped.
  by position  (graded ablation, neighbour count, single positions): every
               listed neighbour keeps its position 1..5, repeats included; words
               not found, or matching no usable tokens, are skipped.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config                                                    # noqa: E402
from build_centroids import PUNCTUATION_TOKENS, parse_response, read_raw_responses  # noqa: E402
from rsa import (human_similarity_file, load_human_similarity,   # noqa: E402
                 mean_centre, normalise, pearson_rsm, upper_triangle)


# =============================================================================
# RUN FOLDERS AND LAYERS
# =============================================================================

def run_dir_for(lang: str, model: str, sweep: str) -> Path:
    return config.output_dir(lang, model, sweep)


def layers_of(run_dir: Path) -> List[int]:
    return json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))["layers"]


def best_layer(run_dir: Path, condition: str = "centred") -> int:
    """l* from rsa.py (highest centred tau on the run-averaged centroids)."""
    best = json.loads((run_dir / "rsa" / "best_layers.json").read_text(encoding="utf-8"))
    return int(best[condition]["layer"])


def resolve_layer(run_dir: Path, layer: Optional[int]) -> Tuple[int, int]:
    """Return (layer number, index into the saved arrays); default layer = l*."""
    layers = layers_of(run_dir)
    layer = best_layer(run_dir) if layer is None else layer
    if layer not in layers:
        raise ValueError(f"layer {layer} not extracted in {run_dir} (have {layers[0]}-{layers[-1]})")
    return layer, layers.index(layer)


# =============================================================================
# CENTROIDS AND HUMAN MATRIX
# =============================================================================

def load_avg_centroids(run_dir: Path, colours: List[str], layer_idx: int) -> Dict[str, np.ndarray]:
    """Run-averaged centroids at one layer (zero vectors dropped), as in rsa.py."""
    out = {}
    for c in colours:
        path = run_dir / "centroids" / f"{c.replace(' ', '_')}_CENTROID_avg.npy"
        if not path.exists():
            print(f"  Missing: {path.name}")
            continue
        v = np.load(path).astype(np.float64)[layer_idx]
        if np.linalg.norm(v) > 0:
            out[c] = v
    return out


def human_matrix(lang: str) -> Tuple[np.ndarray, List[str]]:
    return load_human_similarity(human_similarity_file(lang), config.LANGUAGES[lang]["colours"])


def centred_rsm(vectors: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
    return pearson_rsm(mean_centre({c: vectors[c] for c in order}), order)


# =============================================================================
# NEIGHBOUR VECTORS FROM THE RAW GENERATIONS
# =============================================================================

def _locate(word: str, token_text: str, current_pos: int,
            token_probs: List[Tuple[str, float]], use_matched_length: bool):
    """Character span search and usable token indices for one neighbour word."""
    start, length = token_text.find(word, current_pos), len(word)
    if start == -1:
        stripped = word.replace(" ", "")
        start = token_text.find(stripped, current_pos)
        if use_matched_length:
            length = len(stripped)
    if start == -1:
        return None, []
    end = start + length
    indices, pos = [], 0
    for i, (tok, _) in enumerate(token_probs):
        tok_end = pos + len(tok)
        if pos < end and tok_end > start and tok.strip() and tok not in PUNCTUATION_TOKENS:
            indices.append(i)
        pos = tok_end
    return end, indices


def _entries_with_states(run_dir: Path):
    """Yield (entry, neighbours, hidden_states) for every parseable generation."""
    for e in read_raw_responses(run_dir / "raw_responses.txt"):
        source, neighbours = parse_response(e["response"])
        if source is None:
            continue
        hs_file = run_dir / "hidden_states" / f"{e['colour']}_run{e['run']}.npy"
        if not hs_file.exists():
            continue
        hs = np.load(hs_file)
        if len(e["token_probs"]) != hs.shape[0]:
            continue
        yield e, neighbours, hs


def neighbours_by_word(run_dir: Path, layer_idx: int) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """{colour: {word: [vector per run]}} at one layer (scrambling-control rule)."""
    by_word: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for e, neighbours, hs in _entries_with_states(run_dir):
        text = "".join(t for t, _ in e["token_probs"])
        dash = text.find(" -")
        current_pos = dash + 2 if dash != -1 else 0
        seen = set()
        for word in neighbours:
            if word in seen:
                continue
            end, idx = _locate(word, text, current_pos, e["token_probs"], use_matched_length=True)
            if end is None:
                continue
            if idx:
                seen.add(word)
                by_word[e["colour"]][word].append(hs[idx][:, layer_idx].mean(axis=0))
            current_pos = end
    return {c: dict(w) for c, w in by_word.items()}


def neighbours_by_position(run_dir: Path, layer_idx: Optional[int],
                           max_position: int = config.N_NEIGHBOURS) -> Dict[str, Dict[int, Dict[int, np.ndarray]]]:
    """{colour: {run: {position: vector}}} (per-position rule).

    layer_idx=None keeps every layer: vectors of shape (n_layers, hidden_size).
    """
    layer_sel = slice(None) if layer_idx is None else layer_idx
    out: Dict[str, Dict[int, Dict[int, np.ndarray]]] = defaultdict(dict)
    for e, neighbours, hs in _entries_with_states(run_dir):
        text = "".join(t for t, _ in e["token_probs"])
        dash = text.find(" -")
        current_pos = dash + 2 if dash != -1 else 0
        found = {}
        for position, word in enumerate(neighbours[:max_position], 1):
            end, idx = _locate(word, text, current_pos, e["token_probs"], use_matched_length=False)
            if end is None or not idx:
                continue
            found[position] = hs[idx][:, layer_sel].mean(axis=0)
            current_pos = end
        out[e["colour"]][e["run"]] = found
    return dict(out)


def _parse_after_first_dash(response: str) -> List[str]:
    """Response parser of the median analysis: split the first line at its first '-'."""
    first_line = response.strip().split("\n")[0].strip()
    if "-" not in first_line:
        return []
    return [s.strip() for s in first_line.split("-", 1)[1].split(",") if s.strip()]


def neighbours_for_median(run_dir: Path) -> Dict[str, Dict[int, np.ndarray]]:
    """{colour: {run: array (n_found, n_layers, hidden_size)}} (median-analysis rule).

    Search starts after the first '-'; repeated words are kept once; words not
    found or matching no usable tokens are dropped; vectors keep generation order.
    """
    out: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
    for e in read_raw_responses(run_dir / "raw_responses.txt"):
        neighbours = _parse_after_first_dash(e["response"])
        if not neighbours:
            continue
        hs_file = run_dir / "hidden_states" / f"{e['colour']}_run{e['run']}.npy"
        if not hs_file.exists():
            continue
        hs = np.load(hs_file)
        if len(e["token_probs"]) != hs.shape[0]:
            continue
        text = "".join(t for t, _ in e["token_probs"])
        dash = text.find("-")
        current_pos = dash + 1 if dash != -1 else 0
        seen, vecs = set(), []
        for word in neighbours:
            if word in seen:
                continue
            seen.add(word)
            end, idx = _locate(word, text, current_pos, e["token_probs"], use_matched_length=False)
            if end is None:
                continue
            if idx:
                vecs.append(hs[idx].mean(axis=0))
            current_pos = end
        if vecs:
            out[e["colour"]][e["run"]] = np.stack(vecs, axis=0)
    return dict(out)
