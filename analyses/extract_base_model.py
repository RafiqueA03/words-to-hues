"""
Base-model extraction for the instruction-tuning comparison (paper Appendix I).

The base (non-instruction-tuned) Llama-3.1-8B does not follow chat templates, so
it receives a plain-text few-shot completion prompt: five demonstration lines
"<source> - n1, n2, n3, n4, n5" followed by "<target> -". Demonstrations that
contain the target term are left out, so the target occurs exactly once.

Differences from extract_hidden_states.py (everything else is the same: sampling
settings, seeds, colour order, hidden-state rule, output format):
  - plain-text prompt instead of the chat template;
  - generation also stops at the first newline, because a base model otherwise
    keeps inventing further "<colour> - ..." lines (this keeps answers to one
    line, comparable to the instruct model);
  - raw_responses.txt also records the prompt;
  - with --prompt-states, one prompt-side vector per term (single occurrence):
    hidden_states/<colour>_hfill_run<r>.npy

British English only, as in the paper.

Usage
-----
    python analyses/extract_base_model.py
    python build_centroids.py --lang en --model llama8b_base --sweep all --base-model
    python rsa.py --lang en --model llama8b_base --sweep all

Output  results/en/llama8b_base/all/
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nnsight import LanguageModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config                                                          # noqa: E402
from extract_hidden_states import retry, set_seed, shuffled_colours   # noqa: E402

LANG, MODEL_KEY, SWEEP = "en", "llama8b_base", "all"
N_EXAMPLES = 5

# Candidate demonstrations; any containing the exact target term is skipped.
FEW_SHOT_EXAMPLES: Sequence[Tuple[str, Sequence[str]]] = (
    ("cyan", ("turquoise", "teal", "aqua", "sky_blue", "light_blue")),
    ("navy", ("dark_blue", "midnight_blue", "navy_blue", "deep_blue", "ink_blue")),
    ("scarlet", ("red", "crimson", "vermilion", "ruby", "cherry_red")),
    ("ivory", ("cream", "off_white", "eggshell", "bone", "vanilla")),
    ("charcoal", ("dark_grey", "graphite", "slate", "smoke", "anthracite")),
    ("burgundy", ("maroon", "wine", "claret", "oxblood", "dark_red")),
    ("mauve", ("lilac", "lavender", "dusty_purple", "pale_purple", "heather")),
    ("amber", ("gold", "honey", "ochre", "mustard", "yellow_orange")),
)


# =============================================================================
# PROMPT
# =============================================================================

def term_pattern(term: str) -> "re.Pattern[str]":
    """The term as a complete, underscore-aware lexical item."""
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])")


def create_prompt(colour: str) -> str:
    pattern = term_pattern(colour)
    lines = [f"{src} - {', '.join(syns)}" for src, syns in FEW_SHOT_EXAMPLES
             if pattern.search(f"{src} - {', '.join(syns)}") is None]
    if len(lines) < N_EXAMPLES:
        raise ValueError(f"only {len(lines)} demonstrations without {colour!r}")
    prompt = ("British English colour-name synonym completions.\n"
              "Each line gives one source colour followed by exactly five distinct "
              "synonym colour names. Use lowercase words, joining multiword names "
              "with underscores.\n\n"
              + "\n".join(lines[:N_EXAMPLES]) + f"\n{colour} -")
    if len(pattern.findall(prompt)) != 1:
        raise ValueError(f"{colour!r} must occur exactly once in the prompt")
    return prompt


# =============================================================================
# GENERATION
# =============================================================================

def stop_token_ids(model: LanguageModel) -> List[int]:
    """End-of-text plus the newline token (if '\\n' is a single token)."""
    ids = [model.tokenizer.eos_token_id]
    newline = model.tokenizer.encode("\n", add_special_tokens=False)
    if len(newline) == 1:
        ids.append(newline[0])
    else:
        print(f"    WARNING: '\\n' is {len(newline)} tokens - newline stop disabled")
    return ids


@retry()
def generate_response(model: LanguageModel, prompt: str, run_seed: int,
                      layers: List[int]) -> Tuple[str, List[Tuple[str, float]], np.ndarray]:
    prompt_len = len(model.tokenizer.encode(prompt, add_special_tokens=True))
    with model.generate(
        prompt,
        **config.GENERATION,
        pad_token_id=model.tokenizer.eos_token_id,
        eos_token_id=stop_token_ids(model),
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        remote=True,
    ):
        torch.manual_seed(run_seed)
        out = model.generator.output.save()

    generated_ids = out.sequences[0, prompt_len:].tolist()
    text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
    token_probs = [(model.tokenizer.decode([g]),
                    F.softmax(out.scores[t][0].float(), dim=-1)[g].item())
                   for t, g in enumerate(generated_ids)]
    steps = []
    for step in range(len(generated_ids)):
        per_layer = []
        for layer in layers:
            hs = out.hidden_states[step][layer]
            per_layer.append((hs[0, -1, :] if step == 0 else hs[0, 0, :]).cpu().float().numpy())
        steps.append(np.stack(per_layer, axis=0))
    return text, token_probs, np.stack(steps, axis=0)


@retry()
def extract_prompt_state(model: LanguageModel, prompt: str, colour: str,
                         layers: List[int]) -> Optional[np.ndarray]:
    """Mean state of the target term's tokens at its single prompt occurrence."""
    m = term_pattern(colour).search(prompt)
    offsets = model.tokenizer(prompt, return_offsets_mapping=True,
                              add_special_tokens=True)["offset_mapping"]
    idx = [i for i, (s, e) in enumerate(offsets) if e > m.start() and s < m.end()]
    if not idx:
        return None
    with model.generate(prompt, max_new_tokens=1, do_sample=False,
                        pad_token_id=model.tokenizer.eos_token_id,
                        output_hidden_states=True, return_dict_in_generate=True,
                        remote=True):
        out = model.generator.output.save()
    return np.stack([out.hidden_states[0][layer][0, idx, :].cpu().float().numpy().mean(axis=0)
                     for layer in layers], axis=0)


# =============================================================================
# RUNNER
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Base-model extraction (Appendix I)")
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS)
    parser.add_argument("--prompt-states", action="store_true")
    args = parser.parse_args()

    info = config.MODELS[MODEL_KEY]
    layers = list(range(1, info["n_layers"] + 1))
    colours = config.LANGUAGES[LANG]["colours"]
    out_dir = config.output_dir(LANG, MODEL_KEY, SWEEP)
    hidden_dir = out_dir / "hidden_states"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    raw_file, csv_file = out_dir / "raw_responses.txt", out_dir / "token_probabilities.csv"

    model = LanguageModel(info["hf_id"])
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write("Colour Naming Experiment - British English - NDIF Remote\n")
        f.write("Model type : base pretrained model (not instruction tuned)\n")
        f.write(f"Model      : {info['hf_id']}\n")
        f.write(f"Runs       : {args.runs}\nLayers     : {layers}\n")
        f.write(f"Base seed  : {config.BASE_SEED}\n{'=' * 70}\n\n")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["run", "colour", "seed", "token_index", "token", "probability"])

    failures = []
    for run in range(1, args.runs + 1):
        run_seed = config.BASE_SEED + run - 1
        set_seed(run_seed)
        print(f"\n  Run {run}/{args.runs} (seed={run_seed})")
        run_hidden = {}
        for i, colour in enumerate(shuffled_colours(colours, run_seed), 1):
            print(f"    [{i:2d}/{len(colours)}] {colour}")
            prompt = create_prompt(colour)
            if args.prompt_states:
                try:
                    v = extract_prompt_state(model, prompt, colour, layers)
                    if v is not None:
                        np.save(hidden_dir / f"{colour}_hfill_run{run}.npy", v)
                except Exception as exc:
                    failures.append({"run": run, "colour": colour, "stage": "prompt_state",
                                     "error": str(exc)})
            try:
                text, token_probs, hidden = generate_response(model, prompt, run_seed, layers)
            except Exception as exc:
                print(f"      generation failed: {exc}")
                failures.append({"run": run, "colour": colour, "stage": "generation",
                                 "error": str(exc)})
                continue
            with open(raw_file, "a", encoding="utf-8") as f:
                f.write(f"RUN: {run} | COLOUR: {colour} | SEED: {run_seed}\n")
                f.write(f"PROMPT:\n{prompt}\n\n")
                f.write(f"RESPONSE:\n{text}\n\nTOKEN PROBABILITIES:\n")
                for idx, (tok, prob) in enumerate(token_probs):
                    f.write(f"  [{idx:3d}] {repr(tok):20s} {prob:.6f}\n")
                f.write(f"\nHIDDEN STATES shape : {hidden.shape}\n")
                f.write(f"  Saved to          : hidden_states/{colour}_run{run}.npy\n")
                f.write(f"{'-' * 70}\n\n")
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([run, colour, run_seed, idx, tok, prob]
                                        for idx, (tok, prob) in enumerate(token_probs))
            np.save(hidden_dir / f"{colour}_run{run}.npy", hidden)
            run_hidden[colour] = hidden
            time.sleep(0.1)
        np.savez_compressed(out_dir / f"all_hidden_states_run{run}.npz", **run_hidden)

    json.dump({"language": LANG, "model": info["hf_id"], "sweep": SWEEP, "layers": layers,
               "layer_numbering": "transformer blocks counted from 1 (hidden_states index)",
               "generation": config.GENERATION, "stop_on_newline": True,
               "base_seed": config.BASE_SEED, "num_runs": args.runs,
               "prompt_states": args.prompt_states, "failures": failures},
              open(out_dir / "run_info.json", "w", encoding="utf-8"), indent=2)
    print(f"\nDone: {out_dir}" + (f"  ({len(failures)} failures)" if failures else ""))


if __name__ == "__main__":
    main()
