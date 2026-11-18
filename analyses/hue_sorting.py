"""
Hue-sorting probe (paper Section 4.4, Appendix L).

The model receives a shuffled list of colour terms (BCT or non-BCT chromatic
terms) and sorts them in circular hue order, starting and ending at an anchor
(the term with the lowest CAM16-UCS hue angle). Five runs, seeds 42-46,
temperature 0.6, top-p 0.9, up to 200 new tokens.

Evaluation: Spearman's rho between the model's order and (i) the CAM16-UCS hue
order (rho_ref) and (ii) the model's own Procrustes-aligned MDS hue order from
hue_geometry.py (rho_MDS). Both directions around the circle are tried (anchor
fixed) and the better rho is kept. Means over runs are reported.

Ported from the original analysis code; not re-verified against the original
outputs.

Usage
-----
    python analyses/hue_sorting.py generate --lang en --model llama8b          # needs NDIF
    python analyses/hue_sorting.py evaluate --lang en --model llama8b --sweep all

Output  results/<lang>/<model>/sorting/{responses,evaluation,summary}.csv
"""

import argparse
import random
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import config, normalise, run_dir_for
from hue_geometry import load_focal_colours

MAX_TOKENS = 200

SYSTEM = {
    "en": ("Your task is to sort the colour names in British English based on their hue angles. "
           "I will give you the list of colour names. You must sort them on the hue circle.\n\n"
           "Example:\nQ: Sort these colours in circular hue order starting and ending "
           "at crimson: [gold, amber, crimson]\nA: crimson, amber, gold, crimson\n\n"
           "CRITICAL: No additional text, explanations. Just return a comma-separated "
           "list of hue sorted colour names."),
    "es": ("Tu tarea es ordenar los nombres de colores en español según sus ángulos de tono. "
           "Te daré la lista de nombres de colores. Debes ordenarlos en el círculo de tono.\n\n"
           "Ejemplo:\nP: Ordena estos colores en orden circular de tono empezando y terminando "
           "en carmesí: [oro, ámbar, carmesí]\nR: carmesí, ámbar, oro, carmesí\n\n"
           "CRÍTICO: Sin texto adicional, sin explicaciones. Solo devuelve una lista de "
           "nombres de colores separados por comas, ordenados por tono."),
    "fr": ("Votre tâche est de trier les noms de couleurs en français selon leurs angles "
           "de teinte. Je vous donnerai la liste des noms de couleurs. Vous devez les "
           "trier sur le cercle de teinte.\n\nExemple :\nQ : Triez ces couleurs dans l'ordre "
           "circulaire de teinte en commençant et en terminant par cramoisi : [or, ambre, cramoisi]\n"
           "R : cramoisi, ambre, or, cramoisi\n\nCRITIQUE : Aucun texte supplémentaire, aucune "
           "explication. Renvoyez uniquement une liste de noms de couleurs séparés par des "
           "virgules, triés par teinte."),
}
QUERY = {
    "en": "Sort these colours in circular hue order, starting and ending at {c1}: [{colours}]",
    "es": "Ordena estos colores en orden circular de tono, empezando y terminando en {c1}: [{colours}]",
    "fr": "Triez ces couleurs dans l'ordre circulaire de teinte, en commençant et en terminant par {c1} : [{colours}]",
}


def groups(lang: str) -> dict:
    """Chromatic BCT and non-BCT terms, each sorted by reference hue angle."""
    L = config.LANGUAGES[lang]
    focal = load_focal_colours(lang)
    terms = [normalise(c) for c in L["colours"]]
    bct = [c for c in terms if c in L["bct_chromatic"]]
    non = [c for c in terms if c not in L["bct_chromatic"] and c not in L["achromatic"]]
    return {g: sorted([c for c in words if c in focal], key=lambda c: focal[c]["hue"])
            for g, words in (("bct", bct), ("non_bct", non))}


# =============================================================================
# GENERATE (NDIF)
# =============================================================================

def generate(lang: str, model_key: str) -> None:
    import torch
    from nnsight import LanguageModel
    from extract_hidden_states import set_seed

    model = LanguageModel(config.MODELS[model_key]["hf_id"])
    hue_orders, rows = groups(lang), []
    for run in range(1, config.NUM_RUNS + 1):
        seed = config.BASE_SEED + run - 1
        set_seed(seed)
        for group, order in hue_orders.items():
            shuffled = order.copy()
            random.shuffle(shuffled)
            messages = [{"role": "system", "content": SYSTEM[lang]},
                        {"role": "user", "content": QUERY[lang].format(c1=order[0], colours=", ".join(shuffled))}]
            prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            n = len(model.tokenizer.encode(prompt))
            try:
                with model.generate(prompt, max_new_tokens=MAX_TOKENS, do_sample=True,
                                    temperature=config.GENERATION["temperature"],
                                    top_p=config.GENERATION["top_p"],
                                    return_dict_in_generate=True, remote=True):
                    torch.manual_seed(seed)
                    out = model.generator.output.save()
                response = model.tokenizer.decode(out.sequences[0, n:].tolist(),
                                                  skip_special_tokens=True).strip()
            except Exception as exc:
                print(f"  run {run} {group}: {exc}")
                response = "ERROR"
            rows.append({"run": run, "seed": seed, "group": group, "anchor": order[0],
                         "gt_order": ", ".join(order), "response": response})
            time.sleep(0.5)
    out_dir = config.RESULTS_DIR / lang / model_key / "sorting"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "responses.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# PARSE AND EVALUATE
# =============================================================================

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def parse(response: str, expected: list) -> list:
    """Map comma-separated tokens to lexicon terms (direct, parenthetical, embedded)."""
    exp, out = set(expected), []
    for tok in (t.strip() for t in str(response).split(",") if t.strip()):
        t = _norm(tok)
        m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", t)
        cands = [t] + ([_norm(m.group(1)), _norm(m.group(2))] if m else [])
        hit = next((c for c in cands if c in exp), None)
        if hit is None:
            hit = next((c for c in sorted(exp, key=len, reverse=True)
                        if re.search(r"\b" + re.escape(c) + r"\b", t)), None)
        if hit and hit not in out:
            out.append(hit)
    return out


def spearman_both_ways(ref: list, model_ranks: list) -> float:
    """Best of forward and reversed (anchor kept first) Spearman's rho."""
    fwd = spearmanr(ref, model_ranks)[0]
    rev = spearmanr(ref, [model_ranks[0]] + model_ranks[1:][::-1])[0]
    return round(float(max(fwd, rev)), 4)


def evaluate(lang: str, model_key: str, sweep: str) -> None:
    run_dir = run_dir_for(lang, model_key, sweep)
    sort_dir = config.RESULTS_DIR / lang / model_key / "sorting"
    resp = pd.read_csv(sort_dir / "responses.csv", encoding="utf-8-sig")
    rows = []
    for _, r in resp.iterrows():
        gt = [c.strip() for c in r.gt_order.split(",")]
        model = parse(r.response, gt)
        rank = {c: i for i, c in enumerate(model)}
        common = [c for c in gt if c in rank]
        row = {"run": r.run, "group": r.group, "n_terms": len(gt), "n_returned": len(common),
               "rho_ref": np.nan, "rho_mds": np.nan}
        if len(common) >= 2:
            row["rho_ref"] = spearman_both_ways([gt.index(c) for c in common], [rank[c] for c in common])
            cd = pd.read_csv(run_dir / "hue_geometry" / r.group / "circular_displacement.csv")
            llm = dict(zip(cd.colour.map(normalise), cd.llm_rank))
            if gt[0] in llm:                                   # rotate so the anchor is rank 0
                llm = {c: (v - llm[gt[0]]) % len(llm) for c, v in llm.items()}
            both = [c for c in common if c in llm]
            if len(both) >= 2:
                row["rho_mds"] = spearman_both_ways([llm[c] for c in both], [rank[c] for c in both])
        rows.append(row)
    ev = pd.DataFrame(rows)
    ev.to_csv(sort_dir / "evaluation.csv", index=False)
    summary = ev.groupby("group")[["rho_ref", "rho_mds"]].agg(["mean", "sem"])
    summary.to_csv(sort_dir / "summary.csv")
    print(summary.round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description="Hue-sorting probe (Section 4.4)")
    p.add_argument("step", choices=["generate", "evaluate"])
    p.add_argument("--lang", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sweep", default="all", help="run whose hue geometry is used for rho_MDS")
    a = p.parse_args()
    generate(a.lang, a.model) if a.step == "generate" else evaluate(a.lang, a.model, a.sweep)


if __name__ == "__main__":
    main()
