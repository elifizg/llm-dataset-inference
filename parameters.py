# parameters.py
# Central configuration file. All settings are managed here.
# To change any parameter, only this file needs to be modified.

import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "EleutherAI/pythia-410m-deduped"  # 410m | 1.4b | 6.9b | 12b
TORCH_DTYPE = "float16"                          # float16 (GPU) | float32 (CPU)
DEVICE      = "cuda"                             # cuda | cpu

# ── Data ──────────────────────────────────────────────────────────────────────
SUBSET      = "wikipedia"        # wikipedia | arxiv | github
N_SAMPLES   = 2000               # full dataset (2000 per split)
SEQ_LENGTH  = 512                # maximum sequence length in tokens
RANDOM_SEED = 42

# ── MIA Scores ────────────────────────────────────────────────────────────────
COMPUTE_PERPLEXITY   = True
COMPUTE_ZLIB         = True
COMPUTE_MIN_K        = True
COMPUTE_MAX_K        = True
COMPUTE_PERTURBATION = False  # Phase 2
COMPUTE_REFERENCE    = False  # Phase 2

K_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# ── Training ──────────────────────────────────────────────────────────────────
OUTLIER_PERCENTILE = 2.5
A_SPLIT_RATIO      = 0.5
N_SEEDS            = 10

# ── Testing ───────────────────────────────────────────────────────────────────
P_VALUE_THRESHOLD = 0.1
FP_P_VALUE_MIN    = 0.5

# ── Batch ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 16  # 16-32 for A100; 8 for T4

# ── Paths (auto-detected) ─────────────────────────────────────────────────────
# Detects whether running in Google Colab or locally and sets paths accordingly.
# Override these manually if your Drive path is different.

def _detect_base():
    """Return the appropriate base directory for the current environment."""
    # Google Colab
    colab_base = "/content/drive/MyDrive/llm_dataset_inference"
    if os.path.exists("/content/drive/MyDrive"):
        return colab_base
    # Local fallback — creates directories next to this file
    local_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    return local_base

_BASE          = _detect_base()
CHECKPOINT_DIR = os.path.join(_BASE, "checkpoints")
RESULTS_DIR    = os.path.join(_BASE, "results")