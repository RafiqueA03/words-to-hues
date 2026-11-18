"""
Shared configuration for the "From Words to Hues" pipeline.

All experimental settings live here so that every script uses the same
languages, prompts, colour lexicons, models, layer ranges and paths.
Values follow the paper (EACL 2027), Sections 3.1-3.5 and Appendix A.

Layer numbering
---------------
Layers are given as transformer-block numbers counted from 1
(1 = first block, L = last block), which is how the paper reports layers
(e.g. Llama-3.1-8B peaks at layer 25). In Hugging Face `hidden_states`,
index 0 is the embedding output, so block n is `hidden_states[n]`.
The original scripts stored 0-based indices and read `hidden_states[i + 1]`;
the ranges below are those same layers, written in block numbers.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR          = Path(__file__).resolve().parent
DATA_DIR          = ROOT_DIR / "data"
HUMAN_SIM_DIR     = DATA_DIR / "human_similarity"   # *_similarity_matrix.csv
FOCAL_COLOURS_DIR = DATA_DIR / "focal_colours"      # *_rgb_focals.csv, *_rgb_centroids.csv
RESULTS_DIR       = ROOT_DIR / "results"            # all generated outputs


def output_dir(lang: str, model: str, sweep: str) -> Path:
    """Directory for one extraction run: results/<lang>/<model>/<sweep>/"""
    return RESULTS_DIR / lang / model / sweep


# =============================================================================
# GENERATION SETTINGS  (paper Section 3.3)
# =============================================================================

GENERATION = {
    "max_new_tokens": 150,
    "do_sample":      True,
    "temperature":    0.6,
    "top_p":          0.9,
}
BASE_SEED = 42          # run r uses seed BASE_SEED + (r - 1): 42, 43, 44, 45, 46
NUM_RUNS  = 5
N_NEIGHBOURS = 5        # lexical neighbours requested per colour term


# =============================================================================
# MODELS  (paper Section 3.3)
# =============================================================================

MODELS = {
    "llama8b":  {"hf_id": "meta-llama/Llama-3.1-8B-Instruct",  "n_layers": 32, "size": "small"},
    "llama70b": {"hf_id": "meta-llama/Llama-3.1-70B-Instruct", "n_layers": 80, "size": "large"},
    "qwen7b":   {"hf_id": "Qwen/Qwen2.5-7B-Instruct",          "n_layers": 28, "size": "small"},
    "qwen72b":  {"hf_id": "Qwen/Qwen2.5-72B-Instruct",         "n_layers": 80, "size": "large"},
    # Base (non-instruction-tuned) model, used only for the comparison in Appendix I.
    "llama8b_base": {"hf_id": "meta-llama/Meta-Llama-3.1-8B",  "n_layers": 32, "size": "small"},
}


# =============================================================================
# LAYER SWEEPS
# =============================================================================
# small models : "all"    -> every layer (1..L)
# large models : "coarse" -> every 10th layer (1, 11, ..., 71)
#                "fine"   -> contiguous range around the coarse-sweep peak

COARSE_LAYERS = list(range(1, 80, 10))   # [1, 11, 21, 31, 41, 51, 61, 71]

FINE_LAYERS = {
    ("en", "llama70b"): list(range(51, 72)),   # 51-71
    ("en", "qwen72b"):  list(range(61, 81)),   # 61-80
    ("es", "llama70b"): list(range(41, 72)),   # 41-71
    ("es", "qwen72b"):  list(range(61, 81)),   # 61-80
    ("fr", "llama70b"): list(range(51, 72)),   # 51-71
    ("fr", "qwen72b"):  list(range(61, 81)),   # 61-80
}


def sweep_layers(lang: str, model: str, sweep: str) -> list:
    """Return the transformer-block numbers to extract for one run."""
    info = MODELS[model]
    if info["size"] == "small":
        if sweep != "all":
            raise ValueError(f"{model} is a small model: use --sweep all")
        return list(range(1, info["n_layers"] + 1))
    if sweep == "coarse":
        return COARSE_LAYERS
    if sweep == "fine":
        return FINE_LAYERS[(lang, model)]
    raise ValueError(f"{model} is a large model: use --sweep coarse or --sweep fine")


# =============================================================================
# LANGUAGES: lexicons, BCT sets and prompts  (paper Sections 3.1-3.2, 3.5, Appendix A)
# =============================================================================
# `code` and `file_stem` match the names used by the released data files.

LANGUAGES = {
    "en": {
        "code": "BE",
        "label": "British English",
        "file_stem": "British_English",
        "colours": [
            'beige', 'black', 'blue', 'brown', 'cream', 'dark_blue', 'dark_green',
            'dark_purple', 'green', 'grey', 'light_blue', 'light_green', 'lilac',
            'lime_green', 'magenta', 'maroon', 'mustard', 'ochre', 'olive', 'orange',
            'pale_green', 'peach', 'pink', 'purple', 'red', 'salmon', 'salmon_pink',
            'teal', 'terracotta', 'turquoise', 'white', 'yellow',
        ],
        # Basic colour terms (paper Section 3.5): 8 chromatic + 3 achromatic.
        # Every other lexicon term is a non-basic colour term (non-BCT).
        "bct_chromatic": ['blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'yellow'],
        "achromatic": ['black', 'white', 'grey'],
        "system_prompt": """Your task is to identify synonym colour names in British English.
I will give you one colour name at a time. You must provide five synonym colour names.

Format:
sourcecolour - synonym1, synonym2, synonym3, synonym4, synonym5

Example:
cyan - turquoise, teal, aqua, sky_blue, light_blue

CRITICAL: No additional text, explanations, or repeated colours. All five synonyms must be different words.""",
        "query_prompt": """What are the synonym colours for {colour}?

Provide exactly:
{colour} - synonym1, synonym2, synonym3, synonym4, synonym5
""",
    },
    "es": {
        "code": "ES",
        "label": "Spanish",
        "file_stem": "Spanish",
        "colours": [
            'amarillo', 'azul', 'azul_cielo', 'azul_claro', 'azul_marino', 'beige',
            'blanco', 'caqui', 'carne', 'celeste', 'celeste_claro', 'crema', 'fucsia',
            'granate', 'gris', 'lila', 'marrón', 'morado', 'naranja', 'negro', 'ocre',
            'rojo', 'rosa', 'rosa_claro', 'salmón', 'turquesa', 'verde', 'verde_claro',
            'verde_oscuro', 'verde_pastel', 'violeta',
        ],
        # Basic colour terms (paper Section 3.5): 8 chromatic + 3 achromatic.
        # Every other lexicon term is a non-basic colour term (non-BCT).
        "bct_chromatic": ['amarillo', 'azul', 'marrón', 'morado', 'naranja', 'rojo', 'rosa', 'verde'],
        "achromatic": ['negro', 'blanco', 'gris'],
        "system_prompt": """Tu tarea es identificar nombres de colores sinónimos en español.
Te daré un nombre de color a la vez. Debes proporcionar cinco nombres de colores sinónimos.

Formato:
colorbase - sinonimo1, sinonimo2, sinonimo3, sinonimo4, sinonimo5

Ejemplo:
cian - turquesa, aguamarina, azul_cielo, celeste, azul_claro

CRÍTICO: Sin texto adicional, explicaciones ni colores repetidos. Los cinco sinónimos deben ser palabras diferentes.""",
        "query_prompt": """¿Cuáles son los colores sinónimos de {colour}?

Proporciona exactamente:
{colour} - sinonimo1, sinonimo2, sinonimo3, sinonimo4, sinonimo5
""",
    },
    "fr": {
        "code": "FR",
        "label": "French",
        "file_stem": "French",
        "colours": [
            'beige', 'blanc', 'bleu', 'bleu_clair', 'bleu_foncé', 'bleu_gris',
            'bordeaux', 'gris', 'gris_foncé', 'jaune', 'jaune_pâle', 'kaki',
            'marron', 'marron_clair', 'noir', 'orange', 'rose', 'rose_pâle',
            'rouge', 'turquoise', 'vert', 'vert_clair', 'vert_foncé', 'vert_kaki',
            'violet', 'violet_clair', 'violet_foncé',
        ],
        # Basic colour terms (paper Section 3.5): 8 chromatic + 3 achromatic.
        # Every other lexicon term is a non-basic colour term (non-BCT).
        "bct_chromatic": ['bleu', 'jaune', 'marron', 'orange', 'rose', 'rouge', 'vert', 'violet'],
        "achromatic": ['noir', 'blanc', 'gris'],
        "system_prompt": """Votre tâche est d'identifier des noms de couleurs synonymes en français.
Je vous donnerai un nom de couleur à la fois. Vous devez fournir cinq noms de couleurs synonymes.

Format :
couleur_source - synonyme1, synonyme2, synonyme3, synonyme4, synonyme5

Exemple :
cyan - turquoise, teal, aqua, bleu_ciel, bleu_clair

CRITIQUE : Aucun texte supplémentaire, explication ou couleur répétée. Les cinq synonymes doivent être des mots différents.""",
        "query_prompt": """Quels sont les synonymes de couleur pour {colour} ?

Fournissez exactement :
{colour} - synonyme1, synonyme2, synonyme3, synonyme4, synonyme5
""",
    },
}
