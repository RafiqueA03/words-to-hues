# From Words to Hues

Code for the EACL 2027 paper *From Words to Hues* (title to be finalised).

For each colour term, a language model generates five related colour names
(lexical neighbours). Their hidden states are averaged into a
lexical-neighbourhood centroid. Centroids are compared with human colour-naming
similarity (layer-wise RSA) and with the hue circle (MDS + Procrustes).
British English, Spanish and French; Llama-3.1 (8B, 70B) and Qwen-2.5 (7B, 72B).

## Setup

```bash
pip install -r requirements.txt
```

Hidden states are extracted remotely through [NDIF](https://ndif.us) with
[nnsight](https://nnsight.net); this needs NDIF access and an API key configured for nnsight.

## Data

```
data/human_similarity/   <Language>_similarity_matrix.csv   (human colour-term similarity)
data/focal_colours/      <Language>_rgb_focals.csv          (focal colour of each term)
data/ngram_frequencies/  <Language>_ngram_frequencies.csv   (Google Books Ngram, §4.3)
```

## Main pipeline

```bash
python extract_hidden_states.py --lang en --model llama8b --sweep all   # 1. generation + hidden states (NDIF)
python build_centroids.py --all                                         # 2. lexical-neighbourhood centroids
python rsa.py --all                                                     # 3. layer-wise RSA
python hue_geometry.py --all                                            # 4. hue reconstruction (Table 1)
```

`--lang {en,es,fr}`, `--model {llama8b,llama70b,qwen7b,qwen72b}`. Use `--sweep all`
for 7-8B models, and `--sweep coarse` then `--sweep fine` for 70-72B models.
Outputs go to `results/<lang>/<model>/<sweep>/`. Settings are in `config.py`;
layers are numbered from 1, as in the paper.

## Paper sections

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

Each script's header describes its inputs, options and outputs.

## Notes

- Generation runs remotely on NDIF, so regenerated neighbours may differ
  slightly from those used in the paper.
- The German reference centroids (App. J.4) are taken from Saji et al. (2025)
  and listed in the script.

## Citation

*(to be added after publication)*

## Acknowledgements

We thank the National Deep Inference Fabric (NDIF) for their kind provision of
compute resources.
