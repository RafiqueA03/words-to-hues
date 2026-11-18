"""
Step 4 - Hue reconstruction: MDS, Procrustes alignment and hue metrics.

At the best RSA layer l* (mean-centred, run-averaged centroids), basic (BCT)
and non-basic (non-BCT) chromatic terms are analysed separately:

  1. Mean-centre the centroids, build a Pearson RSM, convert to angular
     dissimilarities d_ij = arccos(r_ij) / pi, and apply classical MDS (3D).
  2. Align the first two MDS dimensions to the reference CAM16-UCS (a, b)
     coordinates of each term's focal colour with Procrustes analysis
     (translation, uniform scaling, rotation/reflection). Report the disparity
     and a 10,000-permutation test.
  3. Hue Kendall's tau over pairwise circular hue distances, and mean circular
     displacement (CD) in hue-rank positions, after the best circular shift.
  4. BCT group only: achromatic terms are projected into the MDS space with the
     Nystrom extension, for residual plots.

Paper: Section 3.5, Section 4.2, Table 1, Appendix J.

Usage
-----
    python hue_geometry.py --lang en --model llama8b --sweep all
    python hue_geometry.py --all

Input   results/<lang>/<model>/<sweep>/centroids/, run_info.json, rsa/best_layers.json
        data/focal_colours/<Language>_rgb_focals.csv
Output  results/<lang>/<model>/<sweep>/hue_geometry/
            summary.csv                      one row per group (Table 1 values)
            <group>/procrustes_residuals.csv per-colour residuals (+ achromatic, BCT)
            <group>/circular_displacement.csv
            <group>/angular_error.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

import colour as colour_lib
from colour import CCS_ILLUMINANTS, XYZ_to_CAM16UCS, sRGB_to_XYZ

import config

N_PERMUTATIONS   = 10_000
PERMUTATION_SEED = 42
RSA_CONDITION    = "centred"


def normalise(name: str) -> str:
    return str(name).replace("_", " ").lower().strip()


# =============================================================================
# REFERENCE GEOMETRY: focal colours in CAM16-UCS
# =============================================================================
# Kept exactly as in the analysis reported in the paper: sRGB values in [0, 1]
# are converted to XYZ in [0, 1], with the white point XYZ_w also in [0, 1].
# Note: colour-science documents CAM16 inputs on a [0, 100] scale; using that
# scale shifts hue angles by up to about 2.5 degrees for saturated colours.

VIEWING_CONDITIONS = {
    "XYZ_w":    np.array([95.047, 100.000, 108.883]) / 100,
    "L_A":      64,
    "Y_b":      20,
    "surround": colour_lib.VIEWING_CONDITIONS_CAM16["Average"],
}


def rgb_to_cam16ucs(r: float, g: float, b: float) -> np.ndarray:
    xyz = sRGB_to_XYZ(np.array([r, g, b]) / 255.0,
                      illuminant=CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    return XYZ_to_CAM16UCS(xyz, **VIEWING_CONDITIONS)


def focal_colours_file(lang: str) -> Path:
    return config.FOCAL_COLOURS_DIR / f"{config.LANGUAGES[lang]['file_stem']}_rgb_focals.csv"


def load_focal_colours(lang: str) -> Dict[str, Dict[str, float]]:
    """CAM16-UCS J, a, b and hue angle of each term's focal colour."""
    code = config.LANGUAGES[lang]["code"]
    df = pd.read_csv(focal_colours_file(lang), encoding="utf-8-sig")
    out = {}
    for _, row in df.iterrows():
        J, a, b = (float(x) for x in rgb_to_cam16ucs(
            row[f"{code}_Red"], row[f"{code}_Green"], row[f"{code}_Blue"]))
        out[normalise(row[code])] = {"J": J, "a": a, "b": b,
                                     "hue": np.degrees(np.arctan2(b, a)) % 360}
    return out


# =============================================================================
# MDS
# =============================================================================

def mean_centre(vectors: Dict[str, np.ndarray], order: List[str]) -> Dict[str, np.ndarray]:
    mu = np.stack([vectors[c] for c in order]).mean(axis=0)
    return {c: vectors[c] - mu for c in order}


def pearson_rsm(vectors: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
    n = len(order)
    rsm = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rsm[i, j] = rsm[j, i] = pearsonr(vectors[order[i]], vectors[order[j]])[0]
    return rsm


def classical_mds(dissim: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Torgerson MDS. Also returns what the Nystrom extension needs."""
    n    = dissim.shape[0]
    d_sq = dissim ** 2
    H    = np.eye(n) - np.ones((n, n)) / n
    B    = -0.5 * H @ d_sq @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pos = eigvals > 0
    k = min(n_components, int(pos.sum()))
    vecs, vals = eigvecs[:, pos][:, :k], eigvals[pos][:k]
    model = {"eigvecs": vecs, "eigvals": vals,
             "col_means_sq": d_sq.mean(axis=0), "grand_mean_sq": d_sq.mean()}
    return vecs * np.sqrt(vals), eigvals, model


def mds_from_centroids(vectors: Dict[str, np.ndarray], order: List[str]):
    v = mean_centre(vectors, order)
    dissim = np.arccos(np.clip(pearson_rsm(v, order), -1, 1)) / np.pi
    np.fill_diagonal(dissim, 0.0)
    return classical_mds(dissim, n_components=3)


def nystrom_project(new_vectors: Dict[str, np.ndarray], new_order: List[str],
                    ref_vectors: Dict[str, np.ndarray], ref_order: List[str],
                    mds_model: Dict) -> np.ndarray:
    """Project new terms into an existing MDS space (Nystrom extension)."""
    D = np.zeros((len(new_order), len(ref_order)))
    for i, ci in enumerate(new_order):
        for j, cj in enumerate(ref_order):
            D[i, j] = np.arccos(np.clip(pearsonr(new_vectors[ci], ref_vectors[cj])[0], -1, 1)) / np.pi
    d_sq = D ** 2
    b_new = -0.5 * (d_sq - d_sq.mean(axis=1, keepdims=True)
                    - mds_model["col_means_sq"] + mds_model["grand_mean_sq"])
    return (b_new @ mds_model["eigvecs"]) / np.sqrt(mds_model["eigvals"])


# =============================================================================
# PROCRUSTES
# =============================================================================
# Both configurations are centred and scaled to unit Frobenius norm; the
# rotation/reflection R comes from an SVD. Disparity = sum of squared
# differences after alignment (as in the analysis reported in the paper).

def fit_procrustes(ref_2d: np.ndarray, llm_2d: np.ndarray) -> Dict:
    ref_mu, llm_mu = ref_2d.mean(axis=0), llm_2d.mean(axis=0)
    ref_c,  llm_c  = ref_2d - ref_mu,     llm_2d - llm_mu
    ref_norm, llm_norm = np.linalg.norm(ref_c), np.linalg.norm(llm_c)
    ref_s = ref_c / ref_norm
    u, _, vt = np.linalg.svd(ref_s.T @ (llm_c / llm_norm))
    return {"ref_mu": ref_mu, "ref_norm": ref_norm, "ref_s": ref_s,
            "llm_mu": llm_mu, "llm_norm": llm_norm, "R": vt.T @ u.T}


def align(llm_2d: np.ndarray, p: Dict) -> np.ndarray:
    return (llm_2d - p["llm_mu"]) / p["llm_norm"] @ p["R"]


def disparity(ref_2d: np.ndarray, llm_2d: np.ndarray) -> float:
    p = fit_procrustes(ref_2d, llm_2d)
    return float(np.sum((p["ref_s"] - align(llm_2d, p)) ** 2))


def permutation_p(ref_2d: np.ndarray, llm_2d: np.ndarray,
                  n_perm: int = N_PERMUTATIONS, seed: int = PERMUTATION_SEED) -> float:
    """Share of label permutations with disparity <= observed (as in the paper)."""
    rng = np.random.default_rng(seed)
    obs = disparity(ref_2d, llm_2d)
    count = sum(disparity(ref_2d, llm_2d[rng.permutation(len(llm_2d))]) <= obs
                for _ in range(n_perm))
    return count / n_perm


# =============================================================================
# HUE METRICS
# =============================================================================

def pairwise_circular_distance(angles_deg: np.ndarray) -> np.ndarray:
    diff = np.abs(angles_deg[:, None] - angles_deg[None, :]) % 360
    return np.minimum(diff, 360 - diff)


def hue_tau(llm_hue: np.ndarray, ref_hue: np.ndarray) -> Tuple[float, float]:
    """Kendall's tau over pairwise circular hue distances (rotation/reflection invariant)."""
    iu = np.triu_indices(len(llm_hue), k=1)
    return kendalltau(pairwise_circular_distance(llm_hue)[iu],
                      pairwise_circular_distance(ref_hue)[iu])


def circular_displacement(llm_hue: np.ndarray, ref_hue: np.ndarray,
                          order: List[str]) -> Tuple[pd.DataFrame, float]:
    """Mean circular rank displacement after the best circular shift of the LLM order."""
    n = len(order)
    ref_rank = {c: i for i, c in enumerate([order[i] for i in np.argsort(ref_hue)])}
    llm_order = [order[i] for i in np.argsort(llm_hue)]
    best_rows, best_mean = None, np.inf
    for shift in range(n):
        shifted = llm_order[shift:] + llm_order[:shift]
        llm_rank = {c: i for i, c in enumerate(shifted)}
        rows = []
        for i, c in enumerate(order):
            d = abs(ref_rank[c] - llm_rank[c])
            rows.append({"colour": c, "ref_rank": ref_rank[c], "llm_rank": llm_rank[c],
                         "displacement": min(d, n - d), "shift": shift,
                         "ref_hue": ref_hue[i], "llm_hue": llm_hue[i]})
        mean = np.mean([r["displacement"] for r in rows])
        if mean < best_mean:
            best_rows, best_mean = rows, mean
    df = pd.DataFrame(best_rows).sort_values("displacement", ascending=False)
    return df.reset_index(drop=True), float(best_mean)


# =============================================================================
# ONE GROUP (BCT or non-BCT)
# =============================================================================

def load_centroids_at(centroid_dir: Path, colours: List[str], layer_idx: int) -> Dict[str, np.ndarray]:
    out = {}
    for c in colours:
        path = centroid_dir / f"{c.replace(' ', '_')}_CENTROID_avg.npy"
        if not path.exists():
            print(f"  Missing: {path.name}")
            continue
        v = np.load(path).astype(np.float64)[layer_idx]
        if np.linalg.norm(v) > 0:
            out[c] = v
    return out


def analyse_group(group: str, words: List[str], centroid_dir: Path, layer_idx: int,
                  focal: Dict, achromatic: List[str], out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = load_centroids_at(centroid_dir, words, layer_idx)
    usable = sorted(set(vectors) & set(focal))
    if len(usable) < 3:
        print(f"  {group}: too few colours ({len(usable)}) - skipped")
        return {}

    ref_2d = np.array([[focal[c]["a"], focal[c]["b"]] for c in usable])
    coords, eigvals, mds_model = mds_from_centroids(vectors, usable)
    llm_2d = coords[:, :2]

    # Procrustes
    p = fit_procrustes(ref_2d, llm_2d)
    aligned = align(llm_2d, p)
    disp = float(np.sum((p["ref_s"] - aligned) ** 2))
    perm_p = permutation_p(ref_2d, llm_2d)
    resid = pd.DataFrame({"colour": usable,
                          "residual": np.sqrt(np.sum((p["ref_s"] - aligned) ** 2, axis=1)),
                          "chromatic": True})

    # Achromatic terms (BCT only): Nystrom projection, then the same alignment.
    if group == "bct":
        ach_vec = load_centroids_at(centroid_dir, achromatic, layer_idx)
        ach = sorted(set(ach_vec) & set(focal))
        if ach:
            mu = np.stack([vectors[c] for c in usable]).mean(axis=0)
            proj = nystrom_project({c: ach_vec[c] - mu for c in ach}, ach,
                                   {c: vectors[c] - mu for c in usable}, usable, mds_model)
            ach_ref = (np.array([[focal[c]["a"], focal[c]["b"]] for c in ach]) - p["ref_mu"]) / p["ref_norm"]
            ach_res = np.sqrt(np.sum((ach_ref - align(proj[:, :2], p)) ** 2, axis=1))
            resid = pd.concat([resid, pd.DataFrame({"colour": ach, "residual": ach_res,
                                                    "chromatic": False})], ignore_index=True)
    resid.sort_values("residual", ascending=False).to_csv(
        out_dir / "procrustes_residuals.csv", index=False, encoding="utf-8-sig")

    # Hue tau: from the MDS plane (invariant to rotation/reflection).
    ref_hue = np.array([focal[c]["hue"] for c in usable])
    tau, tau_p = hue_tau(np.degrees(np.arctan2(llm_2d[:, 1], llm_2d[:, 0])) % 360, ref_hue)

    # Circular displacement and angular error: from the aligned coordinates.
    llm_hue = np.degrees(np.arctan2(aligned[:, 1], aligned[:, 0])) % 360
    cd_df, mean_cd = circular_displacement(llm_hue, ref_hue, usable)
    cd_df.to_csv(out_dir / "circular_displacement.csv", index=False, encoding="utf-8-sig")

    signed = (llm_hue - ref_hue) % 360
    signed = np.where(signed <= 180, signed, signed - 360)
    pd.DataFrame({"colour": usable, "angular_error": np.abs(signed),
                  "angular_error_signed": signed}).to_csv(
        out_dir / "angular_error.csv", index=False, encoding="utf-8-sig")

    pos = eigvals[eigvals > 0]
    print(f"  {group:8s} n={len(usable):2d}  disparity={disp:.3f}  p={perm_p:.4f}  "
          f"hue tau={tau:.3f}  CD={mean_cd:.2f}")
    return {"group": group, "n_terms": len(usable), "disparity": disp,
            "permutation_p": perm_p, "hue_tau": tau, "hue_tau_p": tau_p,
            "mean_circular_displacement": mean_cd,
            "mds_top3_variance": float(pos[:3].sum() / pos.sum())}


# =============================================================================
# ONE EXTRACTION RUN
# =============================================================================

def run_hue_geometry(run_dir: Path, lang: str) -> None:
    centroid_dir = run_dir / "centroids"
    best_file    = run_dir / "rsa" / "best_layers.json"
    info_file    = run_dir / "run_info.json"
    for f in (centroid_dir, best_file, info_file):
        if not f.exists():
            print(f"  Missing {f.relative_to(run_dir)} - run the earlier steps first")
            return

    layers = json.loads(info_file.read_text(encoding="utf-8"))["layers"]
    best   = json.loads(best_file.read_text(encoding="utf-8"))[RSA_CONDITION]["layer"]
    layer_idx = layers.index(best)
    print(f"  Best RSA layer ({RSA_CONDITION}): l* = {best}")

    L = config.LANGUAGES[lang]
    bct, achromatic = L["bct_chromatic"], L["achromatic"]
    non_bct = [c for c in (normalise(x) for x in L["colours"])
               if c not in bct and c not in achromatic]
    focal = load_focal_colours(lang)

    out_root = run_dir / "hue_geometry"
    rows = []
    for group, words in (("bct", bct), ("non_bct", non_bct)):
        r = analyse_group(group, words, centroid_dir, layer_idx, focal, achromatic,
                          out_root / group)
        if r:
            rows.append({"layer": best, **r})
    if rows:
        pd.DataFrame(rows).to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"  Output: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lang", choices=sorted(config.LANGUAGES))
    parser.add_argument("--model", choices=sorted(config.MODELS))
    parser.add_argument("--sweep", choices=["all", "coarse", "fine"])
    parser.add_argument("--all", action="store_true", help="every run with RSA results")
    args = parser.parse_args()

    if args.all:
        targets = [(p.parent.parent.relative_to(config.RESULTS_DIR).parts[0], p.parent.parent)
                   for p in sorted(config.RESULTS_DIR.glob("*/*/*/rsa/best_layers.json"))]
    elif args.lang and args.model and args.sweep:
        targets = [(args.lang, config.output_dir(args.lang, args.model, args.sweep))]
    else:
        parser.error("give --lang, --model and --sweep, or use --all")

    if not targets:
        print(f"No RSA results found under {config.RESULTS_DIR}")
    for lang, run_dir in targets:
        print(f"\n{'=' * 70}\n{run_dir.relative_to(config.ROOT_DIR)}\n{'=' * 70}")
        run_hue_geometry(run_dir, lang)


if __name__ == "__main__":
    main()
