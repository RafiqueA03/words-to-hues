# From Words to Hues

**How LLM Representations Align with Human Colour-Naming Geometry** · EACL 2027 (Main Conference)

Rafique Ahmed¹, Dimitris Mylonas¹, Giovanni Luca Masala², Alexandros Koliousis¹
¹Northeastern University London, UK · ²University of Kent, UK

[Project page](https://rafiquea03.github.io/words-to-hues/) · Paper (link after publication) · [Citation](#citation)

---

Code for the paper. For each colour term, a language model generates related
colour names; their hidden states are averaged into a lexical-neighbourhood
centroid and compared with human colour-naming similarity (RSA) and with the
CAM16-UCS hue circle (MDS + Procrustes). Languages: British English, Spanish,
French. Models: Llama-3.1 (8B, 70B) and Qwen-2.5 (7B, 72B). See the paper and
the project page for the method and results.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12.13; exact package versions are pinned in
`requirements.txt` (scikit-learn must be below 1.10).

Hidden-state extraction (step 1) runs remotely on [NDIF](https://ndif.us) via
[nnsight](https://nnsight.net) and needs an NDIF API key, configured in nnsight
on your own machine. Do not put API keys in this repository. All other steps
run locally.

## Usage

```bash
python extract_hidden_states.py --lang en --model llama8b --sweep all   # 1. generation + hidden states (NDIF)
python build_centroids.py --all                                         # 2. centroids
python rsa.py --all                                                     # 3. layer-wise RSA
python hue_geometry.py --all                                            # 4. hue reconstruction
```

Options: `--lang {en,es,fr}`, `--model {llama8b,llama70b,qwen7b,qwen72b}`.
Settings are in `config.py`; appendix analyses are in `analyses/`. Each
script's header documents its inputs and outputs. Generation uses sampling, so
regenerated outputs may differ slightly from those in the paper.

## Data

`data/` contains the human reference files (similarity matrices, focal
colours, Ngram frequencies). British English files are included; Spanish and
French files will be added.

## Citation

```bibtex
@inproceedings{ahmed2027words,
  title     = {From Words to Hues: How LLM Representations Align with Human Colour-Naming Geometry},
  author    = {Ahmed, Rafique and Mylonas, Dimitris and Masala, Giovanni Luca and Koliousis, Alexandros},
  booktitle = {Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year      = {2027}
}
```

## Acknowledgements

We thank the National Deep Inference Fabric (NDIF) for their kind provision of
compute resources. This work was supported in part by the Leverhulme Trust
Research Project Grant *Cones to Words* (RPG-2024-096).

## Licence

Code and data are released under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](LICENSE).
