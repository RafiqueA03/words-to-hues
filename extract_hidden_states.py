"""
Step 1 - Lexical-neighbour generation and hidden-state extraction (NDIF).

For each colour term, the instruction-tuned model is prompted to generate five
related colour names (lexical neighbours; the prompt asks for "synonyms").
Hidden states are recorded at every generated token for the requested layers.
Paper: Section 3.3 and Appendix A.

Usage
-----
    python extract_hidden_states.py --lang en --model llama8b --sweep all
    python extract_hidden_states.py --lang es --model qwen72b --sweep coarse
    python extract_hidden_states.py --lang fr --model llama70b --sweep fine

    # optional: also save prompt-side states of the target term
    # (used only by the direct-term baselines, Appendix D)
    python extract_hidden_states.py --lang en --model llama8b --sweep all --prompt-states

Requires NDIF access configured for nnsight (see the nnsight documentation).

Outputs  (results/<lang>/<model>/<sweep>/)
-------
    raw_responses.txt               generated text + per-token probabilities
    token_probabilities.csv         the same probabilities in tabular form
    hidden_states/<colour>_run<r>.npy
        shape (n_generated_tokens, n_layers, hidden_size); step 0 is the
        prefill position, later steps are decode positions
    hidden_states/<colour>_hfill_occ{1,2}_run<r>.npy   (only with --prompt-states)
        shape (n_layers, hidden_size)
    all_hidden_states_run<r>.npz    all colours of one run
    run_info.json                   model, layers, settings, failures
"""

import argparse
import csv
import functools
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nnsight import LanguageModel

import config


# =============================================================================
# UTILITIES
# =============================================================================

def retry(max_retries: int = 3, base_wait: int = 10, backoff: int = 2):
    """Retry a remote call with exponential back-off (10 s, 20 s, ...)."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_retries:
                        raise
                    wait = base_wait * (backoff ** (attempt - 1))
                    print(f"  [{fn.__name__}] attempt {attempt}/{max_retries} "
                          f"failed: {exc} - retrying in {wait}s")
                    time.sleep(wait)
        return wrapper
    return decorator


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def shuffled_colours(colours: List[str], seed: int) -> List[str]:
    """Colour order for one run (re-seeded so the order is reproducible)."""
    order = list(colours)
    random.seed(seed)
    random.shuffle(order)
    return order


def build_prompt(model: LanguageModel, lang_cfg: dict, colour: str) -> str:
    messages = [
        {"role": "system", "content": lang_cfg["system_prompt"]},
        {"role": "user",   "content": lang_cfg["query_prompt"].format(colour=colour)},
    ]
    return model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =============================================================================
# GENERATION + HIDDEN STATES
# =============================================================================

@retry()
def generate_response(
    model: LanguageModel,
    formatted: str,
    run_seed: int,
    layers: List[int],
) -> Tuple[str, List[Tuple[str, float]], np.ndarray]:
    """Generate the neighbour list and return text, token probs, hidden states.

    `model.tokenizer.encode(formatted)` includes BOS and matches the length of
    nnsight's `sequences`, so no offset correction is needed here.
    """
    prompt_len = len(model.tokenizer.encode(formatted))

    with model.generate(
        formatted,
        **config.GENERATION,
        pad_token_id=model.tokenizer.eos_token_id,
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        remote=True,
    ):
        torch.manual_seed(run_seed)
        gen_output = model.generator.output.save()

    generated_ids = gen_output.sequences[0, prompt_len:].tolist()
    response_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

    token_probs: List[Tuple[str, float]] = []
    for t, gid in enumerate(generated_ids):
        probs = F.softmax(gen_output.scores[t][0].float(), dim=-1)
        token_probs.append((model.tokenizer.decode([gid]), probs[gid].item()))

    # hidden_states[step][n] is transformer block n (index 0 = embeddings).
    # Step 0 holds the whole prompt: take its last position. Later steps hold
    # a single decode position.
    steps = []
    for step_idx in range(len(generated_ids)):
        per_layer = []
        for layer in layers:
            hs  = gen_output.hidden_states[step_idx][layer]
            vec = hs[0, -1, :] if step_idx == 0 else hs[0, 0, :]
            per_layer.append(vec.cpu().float().numpy())
        steps.append(np.stack(per_layer, axis=0))

    return response_text, token_probs, np.stack(steps, axis=0)


# =============================================================================
# OPTIONAL: PROMPT-SIDE STATES OF THE TARGET TERM  (Appendix D baselines)
# =============================================================================

def _char_span_to_token_indices(start: int, end: int,
                                offsets: List[Tuple[int, int]]) -> List[int]:
    return [i for i, (s, e) in enumerate(offsets) if e > start and s < end]


@retry()
def extract_prompt_states(
    model: LanguageModel,
    formatted: str,
    colour: str,
    layers: List[int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Mean hidden state of the target term at its two prompt occurrences.

    occ1 = the question ("... for {colour}?"), occ2 = the answer-format line
    ("{colour} - synonym1, ..."). The last two occurrences are used, because
    some terms (e.g. azul_claro) also appear in the fixed in-context example.

    Token indices are shifted by +1 because nnsight's generate() prepends a BOS
    token that `tokenizer(..., add_special_tokens=False)` does not include.
    """
    offsets = model.tokenizer(formatted, return_offsets_mapping=True,
                              add_special_tokens=False)["offset_mapping"]
    spans = [(m.start(), m.end()) for m in
             re.finditer(r"\b" + re.escape(colour) + r"\b", formatted)]
    if len(spans) < 2:
        print(f"      prompt states: expected >= 2 occurrences, found {len(spans)} - skipped")
        return None, None

    idx1 = [i + 1 for i in _char_span_to_token_indices(*spans[-2], offsets)]
    idx2 = [i + 1 for i in _char_span_to_token_indices(*spans[-1], offsets)]
    if not idx1 or not idx2:
        print("      prompt states: empty token indices - skipped")
        return None, None

    with model.generate(
        formatted,
        max_new_tokens=1,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
        remote=True,
    ):
        out = model.generator.output.save()

    prefill = out.hidden_states[0]

    def mean_at(token_idx: List[int]) -> np.ndarray:
        return np.stack(
            [prefill[layer][0, token_idx, :].cpu().float().numpy().mean(axis=0)
             for layer in layers],
            axis=0,
        )

    return mean_at(idx1), mean_at(idx2)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(lang: str, model_key: str, sweep: str,
                   num_runs: int, prompt_states: bool) -> None:
    lang_cfg   = config.LANGUAGES[lang]
    model_info = config.MODELS[model_key]
    layers     = config.sweep_layers(lang, model_key, sweep)

    out_dir    = config.output_dir(lang, model_key, sweep)
    hidden_dir = out_dir / "hidden_states"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    raw_file   = out_dir / "raw_responses.txt"
    csv_file   = out_dir / "token_probabilities.csv"

    print(f"\n{'=' * 70}")
    print(f"Language : {lang_cfg['label']}  ({len(lang_cfg['colours'])} terms)")
    print(f"Model    : {model_info['hf_id']}")
    print(f"Sweep    : {sweep}  layers {layers}")
    print(f"Runs     : {num_runs}  (seeds {config.BASE_SEED}-{config.BASE_SEED + num_runs - 1})")
    print(f"Output   : {out_dir}")
    print(f"{'=' * 70}\n")

    model = LanguageModel(model_info["hf_id"])

    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(f"Colour Naming Experiment - {lang_cfg['label']} - NDIF Remote\n")
        f.write(f"Model     : {model_info['hf_id']}\n")
        f.write(f"Runs      : {num_runs}\n")
        f.write(f"Layers    : {layers}\n")
        f.write(f"Base seed : {config.BASE_SEED}\n")
        f.write(f"{'=' * 70}\n\n")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["run", "colour", "seed", "token_index", "token", "probability"])

    failures: List[Dict] = []

    for run in range(1, num_runs + 1):
        run_seed = config.BASE_SEED + (run - 1)
        set_seed(run_seed)
        colours = shuffled_colours(lang_cfg["colours"], run_seed)
        print(f"\n  Run {run}/{num_runs}  (seed={run_seed})")

        run_hidden: Dict[str, np.ndarray] = {}
        for i, colour in enumerate(colours, 1):
            print(f"    [{i:2d}/{len(colours)}] {colour}")
            formatted = build_prompt(model, lang_cfg, colour)

            if prompt_states:
                try:
                    occ1, occ2 = extract_prompt_states(model, formatted, colour, layers)
                    if occ1 is not None:
                        np.save(hidden_dir / f"{colour}_hfill_occ1_run{run}.npy", occ1)
                        np.save(hidden_dir / f"{colour}_hfill_occ2_run{run}.npy", occ2)
                except Exception as exc:
                    print(f"      prompt states failed: {exc}")
                    failures.append({"run": run, "colour": colour, "stage": "prompt_states",
                                     "error": str(exc)})

            try:
                response, token_probs, hidden = generate_response(
                    model, formatted, run_seed, layers)
            except Exception as exc:
                print(f"      generation failed: {exc}")
                failures.append({"run": run, "colour": colour, "stage": "generation",
                                 "error": str(exc)})
                continue

            # Block format is read by build_centroids.py - keep it unchanged.
            with open(raw_file, "a", encoding="utf-8") as f:
                f.write(f"RUN: {run} | COLOUR: {colour} | SEED: {run_seed}\n")
                f.write(f"RESPONSE:\n{response}\n\n")
                f.write("TOKEN PROBABILITIES:\n")
                for idx, (tok, prob) in enumerate(token_probs):
                    f.write(f"  [{idx:3d}] {repr(tok):20s} {prob:.6f}\n")
                f.write(f"\nHIDDEN STATES shape : {hidden.shape}\n")
                f.write(f"  Saved to          : hidden_states/{colour}_run{run}.npy\n")
                f.write(f"{'-' * 70}\n\n")
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for idx, (tok, prob) in enumerate(token_probs):
                    w.writerow([run, colour, run_seed, idx, tok, prob])

            np.save(hidden_dir / f"{colour}_run{run}.npy", hidden)
            run_hidden[colour] = hidden
            print(f"      hidden states {hidden.shape}")
            time.sleep(0.1)

        np.savez_compressed(out_dir / f"all_hidden_states_run{run}.npz", **run_hidden)

    run_info = {
        "language": lang, "model": model_info["hf_id"], "sweep": sweep,
        "layers": layers,
        "layer_numbering": "transformer blocks counted from 1 (hidden_states index)",
        "generation": config.GENERATION, "base_seed": config.BASE_SEED,
        "num_runs": num_runs, "prompt_states": prompt_states,
        "failures": failures,
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {out_dir}")
    if failures:
        print(f"WARNING: {len(failures)} failed call(s) - see run_info.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lang", required=True, choices=sorted(config.LANGUAGES))
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--sweep", required=True, choices=["all", "coarse", "fine"],
                        help="'all' for 7-8B models; 'coarse' or 'fine' for 70-72B models")
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS)
    parser.add_argument("--prompt-states", action="store_true",
                        help="also save prompt-side states of the target term (Appendix D)")
    args = parser.parse_args()

    if args.model == "llama8b_base":
        parser.error("the base model uses a different (plain-text) prompt: "
                     "use analyses/extract_base_model.py")
    config.sweep_layers(args.lang, args.model, args.sweep)   # validate before connecting
    run_experiment(args.lang, args.model, args.sweep, args.runs, args.prompt_states)


if __name__ == "__main__":
    main()
