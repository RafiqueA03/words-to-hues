"""
Independent validation on German basic colour terms (paper Appendix J.4).

The BrEn BCT hue reconstruction is repeated with an independent reference: the
CAM16-UCS centroids of German BCTs from Saji et al. (2025), matched through their
English glosses. The model side is unchanged: BrEn run-averaged centroids at the
same layer l* as the main analysis. Grey is projected as the achromatic term.

German reference: 13 centroids in CAM16-UCS (J, a, b), derived from the raw
German data of Saji et al. (2025) (D65, Bradford adaptation). They are listed
below as used in the paper; the derivation from the raw data is not part of
this repository.

Usage
-----
    python analyses/german_validation.py --model llama8b --sweep all

Output  results/en/<model>/<sweep>/german_validation/summary.csv
        (+ per-colour files, as in hue_geometry.py), and the BrEn-reference
        values from hue_geometry/summary.csv for comparison.
"""

import argparse

import numpy as np
import pandas as pd

from common import best_layer, layers_of, run_dir_for
from hue_geometry import analyse_group

# Saji et al. (2025), German raw data -> CAM16-UCS centroids, English glosses.
GERMAN_CENTROIDS = {
    "red":        (40.065758,  38.506551,   9.752878),
    "orange":     (79.216374,  10.884025,  30.664865),
    "yellow":     (87.264418,  -4.550325,  34.583030),
    "green":      (68.064576, -21.321526,   8.380043),
    "blue":       (26.808953, -11.047469, -27.915445),
    "brown":      (38.428982,   9.877991,  19.636857),
    "purple":     (23.288306,  16.216433, -10.690141),
    "pink":       (45.552337,  38.483308,  -2.090795),
    "grey":       (59.024424,  -0.916819,  -0.530651),
    "light blue": (57.729075, -23.552013, -19.696858),
    "dark green": (31.794779, -19.087008,   6.511147),
    "turquoise":  (77.483808, -22.885792,  -2.542911),
    "peach":      (70.009079,  17.601205,  13.065966),
}
SHARED_BCT = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]


def german_reference() -> dict:
    return {name: {"J": J, "a": a, "b": b, "hue": float(np.degrees(np.arctan2(b, a)) % 360)}
            for name, (J, a, b) in GERMAN_CENTROIDS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="German BCT validation (Appendix J.4)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args()

    run_dir = run_dir_for("en", args.model, args.sweep)
    layer = best_layer(run_dir)
    layer_idx = layers_of(run_dir).index(layer)
    out = run_dir / "german_validation"
    print(f"German validation: en {args.model} {args.sweep}  layer {layer}")

    result = analyse_group("bct", SHARED_BCT, run_dir / "centroids", layer_idx,
                           german_reference(), ["grey"], out / "bct")
    row = {"layer": layer, "reference": "German (Saji et al. 2025)", **result}

    own = run_dir / "hue_geometry" / "summary.csv"
    if own.exists():
        bren = pd.read_csv(own).set_index("group").loc["bct"]
        row["bren_reference_hue_tau"] = bren["hue_tau"]
        row["bren_reference_cd"] = bren["mean_circular_displacement"]
    pd.DataFrame([row]).to_csv(out / "summary.csv", index=False)
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
