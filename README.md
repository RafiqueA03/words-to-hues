# From Words to Hues

<!-- TITLE: update once the final camera-ready title is confirmed -->
**How LLMs Encode Perceptual Colour Space** · EACL 2027 (Main Conference)

Rafique Ahmed¹, Dimitris Mylonas¹, Giovanni Luca Masala², Alexandros Koliousis¹
¹Northeastern University London, UK · ²University of Kent, UK

[Project page](https://rafiquea03.github.io/words-to-hues/) · Paper (link after publication) · [Citation](#citation)

---

Do the hidden states of text-only language models recover the geometry of
human colour naming? For each colour term, a model generates five related
colour names (*lexical neighbours*). Their hidden states are averaged into a
**lexical-neighbourhood centroid**, which is compared with human colour-naming
similarity (layer-wise RSA) and with the CAM16-UCS hue circle (MDS +
Procrustes). We study British English, Spanish and French indispensable colour
lexicons with Llama-3.1 (8B, 70B) and Qwen-2.5 (7B, 72B).

## Repository layout

```
config.py                  all settings: languages, prompts, lexicons, models, layers, paths
extract_hidden_states.py   step 1  generation + hidden states (NDIF)
build_centroids.py         step 2  lexical-neighbourhood centroids
rsa.py                     step 3  layer-wise RSA
hue_geometry.py            step 4  hue reconstruction (Table 1)
analyses/                  appendix analyses (see table below)
data/                      human reference data
docs/                      project page (GitHub Pages)
results/                   outputs (created by the scripts; not tracked)
```

## Setup

```bash
pip install -r requirements.txt
```

`scikit-learn` is pinned below 1.10 because 1.10
changes the MDS defaults used in `analyses/synthetic_circle.py`.

### NDIF access (step 1 only)

Hidden states are extracted remotely through [NDIF](https://ndif.us) with
[nnsight](https://nnsight.net). This needs an NDIF account and API key.
Configure the key in nnsight on your own machine (see the nnsight
documentation). **Do not put API keys in this repository**: the scripts only
call `remote=True` and read the key from nnsight's configuration. Steps 2–4
and most analyses run locally from saved hidden states and need no key.

## Data

```
data/human_similarity/   <Language>_similarity_matrix.csv   human colour-term similarity (1 − √JSD)
data/focal_colours/      <Language>_rgb_focals.csv          focal colour of each term (sRGB)
data/ngram_frequencies/  <Language>_ngram_frequencies.csv   Google Books Ngram frequencies (§4.3)
```

British English files are included. **Spanish and French files will be added
shortly.** The British English and French naming data were published by
Mylonas et al. (2026); the Spanish data were obtained through a data-sharing
agreement with the ColorNaming project, and the Spanish indispensable lexicon
was derived by us using the same procedure.

## Running the pipeline

```bash
python extract_hidden_states.py --lang en --model llama8b --sweep all   # 1. generation + hidden states (NDIF)
python build_centroids.py --all                                         # 2. lexical-neighbourhood centroids
python rsa.py --all                                                     # 3. layer-wise RSA
python hue_geometry.py --all                                            # 4. hue reconstruction (Table 1)
```

- `--lang {en,es,fr}`, `--model {llama8b,llama70b,qwen7b,qwen72b}`
- Use `--sweep all` for the 7–8B models; for the 70–72B models run
  `--sweep coarse`, then `--sweep fine` around the coarse peak.
- Outputs go to `results/<lang>/<model>/<sweep>/`.
- Layers are numbered from 1, as in the paper.

Each script's header documents its inputs, options and outputs.

## Paper sections and scripts

| Paper | Script |
|---|---|
| §3.3, App. A | `extract_hidden_states.py`, `build_centroids.py` |
| §3.4, §4.1 | `rsa.py` |
| §3.5, §4.2, Table 1, App. J | `hue_geometry.py` |
| §4.3, App. K | `analyses/frequency.py` |
| §4.4, App. L | `analyses/hue_sorting.py` |
| App. B, E, F | `analyses/aggregation.py` (`count`, `median`, `position`) |
| App. D | `analyses/direct_term.py` (extraction with `--prompt-states`) |
| App. G | `analyses/rsa_controls.py` (`permutation`, `scramble`, `ablation`) |
| Robustness appendix | `analyses/rsa_robustness.py` (`metric`, `grid`) |
| App. I | `analyses/extract_base_model.py`, then `build_centroids.py --base-model` and `rsa.py` |
| App. J.1 | `analyses/synthetic_circle.py` |
| App. J.4 | `analyses/german_validation.py` |

<!-- Section numbers: check against the final camera-ready before release -->

## Project page

The page in `docs/` is a static site (no build step). It contains interactive
hue wheels for the three lexicons and a hue-shift explorer that reproduces the
paper's circular-displacement figures for every language, model and term
group. Its data (`docs/static/data.js`) are taken from the outputs behind the
paper's figures, and every circular-displacement value matches Table 1.

- **View locally:** open `docs/index.html` in a browser.
- **Publish:** in the GitHub repository, *Settings → Pages → Deploy from a
  branch → `main` / `docs`*. The page is then served at
  `https://rafiquea03.github.io/words-to-hues/`. On a free GitHub account,
  Pages requires the repository to be public.

## Notes

- Generation runs remotely on NDIF with sampling, so regenerated neighbours
  may differ slightly from those used in the paper.
- The German reference centroids (App. J.4) are taken from Saji et al. (2025)
  and listed in the script.

## Citation

<!-- Replace with the official ACL Anthology entry after publication -->
```bibtex
@inproceedings{ahmed2027words,
  title     = {From Words to Hues: How LLMs Encode Perceptual Colour Space},
  author    = {Ahmed, Rafique and Mylonas, Dimitris and Masala, Giovanni Luca and Koliousis, Alexandros},
  booktitle = {Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year      = {2027}
}
```

## Acknowledgements

We thank the National Deep Inference Fabric (NDIF) for their kind provision of
compute resources. This work was supported by the Leverhulme Trust
[SCHEME / GRANT NO.].

## Licence

This repository (code and data) is released under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](LICENSE).
