"""
Lexical frequency and hue stability (paper Section 4.3, Appendix K).

Correlates each colour term's Google Books Ngram frequency (mean 1900-2020)
with its hue-reconstruction error (circular displacement, CD) from
hue_geometry.py. BCT and non-BCT chromatic terms are pooled. Spearman's rho.
The paper uses Llama-3.1-8B.

Two steps, so the correlation does not depend on the live Ngram service:

  fetch      Query the Google Books Ngram API once and save the frequencies to
             data/ngram_frequencies/<Language>_ngram_frequencies.csv.
             Each term is tried as an adjective-tagged form ("blue_ADJ"),
             capitalised, case-insensitive, then untagged; the first variant
             with data is kept. Corpora (2019): English 26, French 30, Spanish 32.
  correlate  Spearman correlations between the saved frequencies (mean, recent
             10 years, max) and CD / angular error.

Usage
-----
    python analyses/frequency.py fetch     --lang en
    python analyses/frequency.py correlate --lang en --model llama8b --sweep all

Output  results/<lang>/<model>/<sweep>/frequency/{by_colour,spearman}.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import config, normalise, run_dir_for

API_URL    = "https://books.google.com/ngrams/json"
CORPUS     = {"en": 26, "fr": 30, "es": 32}
YEARS      = (1900, 2020)
FOUND_MIN  = 1e-10
FREQ_COLS  = ["ngram_mean_freq", "ngram_recent_freq", "ngram_max_freq"]
ERROR_COLS = ["displacement", "angular_error", "angular_error_signed"]


def frequency_file(lang: str) -> Path:
    stem = config.LANGUAGES[lang]["file_stem"]
    return config.DATA_DIR / "ngram_frequencies" / f"{stem}_ngram_frequencies.csv"


# =============================================================================
# FETCH
# =============================================================================

def _query(content: str, corpus: int, case_insensitive: bool):
    import requests
    params = {"content": content, "year_start": YEARS[0], "year_end": YEARS[1],
              "corpus": corpus, "smoothing": 0, "case_insensitive": case_insensitive}
    try:
        r = requests.get(API_URL, params=params, timeout=60)
        return r.json() if r.status_code == 200 and r.json() else None
    except Exception as exc:
        print(f"  API error: {exc}")
        return None


def _variants(term: str):
    words = term.split()
    cap = [words[0].capitalize()] + words[1:]
    tag = lambda w: " ".join(w[:-1] + [f"{w[-1]}_ADJ"])
    return [(tag(words), "ADJ", False), (tag(cap), "ADJ-cap", False), (tag(words), "ADJ-ci", True),
            (" ".join(words), "bare", False), (" ".join(cap), "bare-cap", False),
            (" ".join(words), "bare-ci", True)]


def fetch_one(term: str, corpus: int) -> dict:
    for content, label, ci in _variants(term):
        data = _query(content, corpus, ci)
        for item in data or []:
            ts = np.array(item["timeseries"], dtype=float)
            if np.any(ts >= FOUND_MIN):
                return {"colour": term, "ngram_query": content, "ngram_variant": label,
                        "ngram_mean_freq": ts.mean(), "ngram_recent_freq": ts[-10:].mean(),
                        "ngram_max_freq": ts.max()}
        time.sleep(0.5)
    return {"colour": term, "ngram_query": "", "ngram_variant": "failed",
            **{c: np.nan for c in FREQ_COLS}}


def fetch(lang: str) -> None:
    L = config.LANGUAGES[lang]
    terms = sorted(normalise(c) for c in L["colours"] if normalise(c) not in L["achromatic"])
    rows = []
    for t in terms:
        row = fetch_one(t, CORPUS[lang])
        for _ in range(3):                      # retry failed terms
            if row["ngram_variant"] != "failed":
                break
            time.sleep(10)
            row = fetch_one(t, CORPUS[lang])
        print(f"  {t:15s} {row['ngram_variant']}")
        rows.append(row)
        time.sleep(2)
    path = frequency_file(lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


# =============================================================================
# CORRELATE
# =============================================================================

def hue_errors(run_dir: Path) -> pd.DataFrame:
    """Per-colour CD and angular error, BCT and non-BCT pooled."""
    parts = []
    for group in ("bct", "non_bct"):
        g = run_dir / "hue_geometry" / group
        cd = pd.read_csv(g / "circular_displacement.csv")[["colour", "displacement"]]
        ang = pd.read_csv(g / "angular_error.csv")
        parts.append(cd.merge(ang, on="colour"))
    df = pd.concat(parts, ignore_index=True)
    df["colour"] = df["colour"].map(normalise)
    return df.groupby("colour", as_index=False)[ERROR_COLS].mean()


def correlate(lang: str, run_dir: Path, freq_path: Path) -> None:
    freq = pd.read_csv(freq_path, encoding="utf-8-sig")
    freq["colour"] = freq["colour"].map(normalise)
    keep = ["colour"] + [c for c in ("ngram_query", "ngram_variant") if c in freq] + FREQ_COLS
    merged = hue_errors(run_dir).merge(freq[keep], on="colour", how="left")
    rows = []
    valid_all = merged.dropna(subset=FREQ_COLS)
    for e in ERROR_COLS:
        for f in FREQ_COLS:
            v = valid_all.dropna(subset=[f, e])
            rho, p = spearmanr(v[f], v[e]) if v[f].nunique() > 1 and v[e].nunique() > 1 else (np.nan, np.nan)
            rows.append({"error_metric": e, "freq_metric": f, "spearman_rho": rho,
                         "p_value": p, "n": len(v)})
    out = run_dir / "frequency"
    out.mkdir(exist_ok=True)
    merged.to_csv(out / "by_colour.csv", index=False, encoding="utf-8-sig")
    res = pd.DataFrame(rows)
    res.to_csv(out / "spearman.csv", index=False)
    r = res[(res.error_metric == "displacement") & (res.freq_metric == "ngram_mean_freq")].iloc[0]
    print(f"  CD vs mean frequency: rho = {r.spearman_rho:.2f}, p = {r.p_value:.3f}, n = {r.n}")
    print(f"  Output: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexical frequency analysis (Section 4.3)")
    parser.add_argument("step", choices=["fetch", "correlate"])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--model", default="llama8b")
    parser.add_argument("--sweep", default="all")
    parser.add_argument("--frequencies", type=Path, help="frequency CSV (default: data/ngram_frequencies/)")
    args = parser.parse_args()
    if args.step == "fetch":
        fetch(args.lang)
    else:
        correlate(args.lang, run_dir_for(args.lang, args.model, args.sweep),
                  args.frequencies or frequency_file(args.lang))


if __name__ == "__main__":
    main()
