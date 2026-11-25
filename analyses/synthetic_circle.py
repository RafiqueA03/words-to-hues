"""
Synthetic circle-detector calibration (paper Appendix K.1).

Checks that the geometry-recovery stage distinguishes circular from
non-circular structure. No model, no hidden states: point sets are generated
directly and passed through Euclidean distances -> metric MDS (2D) -> Procrustes
alignment to a reference circle -> permutation test.

  positive_circle  points on a circle plus Gaussian noise (sigma 0.12)
  negative_cloud   points uniform in a square
  negative_line    points along a noisy line; its 1D structure should still be
                   recovered on the leading axis (|Spearman rho| with the true order)

N = 8, 21, 29 (BrEn BCT, non-BCT, all chromatic terms), 30 draws per N,
2,000 permutations per draw; draw d uses seed d.

Note: this calibration uses scikit-learn's metric MDS on Euclidean distances and
scipy's Procrustes disparity, as in the original analysis; it is not identical
to hue_geometry.py (classical MDS on angular dissimilarities).

Usage
-----
    python analyses/synthetic_circle.py
    python analyses/synthetic_circle.py --n 29

Output  results/synthetic_circle/summary.csv, draws.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, spearmanr
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

N_VALUES  = [8, 21, 29]
N_DRAWS   = 30
N_PERM    = 2000
BASE_SEED = 0


def circular_distances(angles: np.ndarray) -> np.ndarray:
    diff = np.abs(angles[:, None] - angles[None, :])
    return np.minimum(diff, 2 * np.pi - diff)


def circular_displacement(ref_angles: np.ndarray, rec_angles: np.ndarray) -> float:
    """Mean circular rank displacement, best over circular shifts and reflection."""
    n = len(ref_angles)
    ref_rank = np.argsort(np.argsort(ref_angles))
    rec_rank = np.argsort(np.argsort(rec_angles))
    best = None
    for variant in (rec_rank, (n - rec_rank) % n):
        for shift in range(n):
            s = (variant + shift) % n
            d = np.minimum(np.abs(s - ref_rank), n - np.abs(s - ref_rank)).mean()
            best = d if best is None or d < best else best
    return best


def analyse(coords: np.ndarray, ref_coords: np.ndarray, seed: int):
    n = coords.shape[0]
    embedding = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
                    n_init=4, normalized_stress=False).fit_transform(squareform(pdist(coords)))
    m_ref, m_rec, disparity = procrustes(ref_coords, embedding)
    rec = np.arctan2(m_rec[:, 1], m_rec[:, 0]) % (2 * np.pi)
    ref = np.arctan2(m_ref[:, 1], m_ref[:, 0]) % (2 * np.pi)
    iu = np.triu_indices(n, 1)
    tau = kendalltau(circular_distances(ref)[iu], circular_distances(rec)[iu])[0]
    disp = circular_displacement(ref, rec)
    rng = np.random.default_rng(seed)
    null = np.array([procrustes(ref_coords[rng.permutation(n)], embedding)[2] for _ in range(N_PERM)])
    p = (np.sum(null <= disparity) + 1) / (N_PERM + 1)
    return disparity, tau, disp, p, embedding


def reference_circle(n: int):
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)]), angles


def noisy_circle(angles, sigma, rng):
    return np.column_stack([np.cos(angles), np.sin(angles)]) + rng.normal(0, sigma, size=(len(angles), 2))


def uniform_cloud(n, rng, half_width=1.2):
    return rng.uniform(-half_width, half_width, size=(n, 2))


def noisy_line(n, sigma, rng):
    x = np.linspace(-1, 1, n)
    y = rng.normal(0, sigma, size=n)
    x = x + rng.normal(0, sigma, size=n)
    perm = rng.permutation(n)
    return np.column_stack([x, y])[perm], perm


def run_for_n(n: int) -> list:
    ref_coords, ref_angles = reference_circle(n)
    rows = []
    for draw in range(N_DRAWS):
        seed = BASE_SEED + draw
        rng = np.random.default_rng(seed)
        # Same order of random draws as the original: circle, cloud, line.
        for name in ("positive_circle", "negative_cloud", "negative_line"):
            true_order = None
            if name == "positive_circle":
                coords = noisy_circle(ref_angles, 0.12, rng)
            elif name == "negative_cloud":
                coords = uniform_cloud(n, rng)
            else:
                coords, true_order = noisy_line(n, 0.02, rng)
            disparity, tau, disp, p, emb = analyse(coords, ref_coords, seed)
            row = {"n": n, "draw": draw, "condition": name, "disparity": disparity,
                   "tau": tau, "circular_displacement": disp, "p_value": p}
            if true_order is not None:
                c = emb - emb.mean(axis=0)
                pc1 = c @ np.linalg.eigh(c.T @ c)[1][:, -1]
                row["line_axis_abs_rho"] = abs(spearmanr(pc1, true_order)[0])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic circle calibration (Appendix K.1)")
    parser.add_argument("--n", type=int, nargs="*", default=N_VALUES)
    args = parser.parse_args()

    draws = pd.DataFrame([r for n in args.n for r in run_for_n(n)])
    summary = (draws.groupby(["n", "condition"])
                    .agg(disparity=("disparity", "mean"), tau=("tau", "mean"),
                         circular_displacement=("circular_displacement", "mean"),
                         p_mean=("p_value", "mean"),
                         share_p_below_001=("p_value", lambda p: float(np.mean(p < 0.001))),
                         line_axis_abs_rho=("line_axis_abs_rho", "mean"))
                    .reset_index())
    out = config.RESULTS_DIR / "synthetic_circle"
    out.mkdir(parents=True, exist_ok=True)
    draws.to_csv(out / "draws.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
