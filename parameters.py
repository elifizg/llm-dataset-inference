# parameters.py
# Central configuration file. All settings are managed here.
# Works in Google Colab, Jupyter, and local Python environments.
# Model and dataset are downloaded automatically from HuggingFace on first run.

import os
import torch

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "EleutherAI/pythia-410m-deduped"  # 410m | 1.4b | 6.9b | 12b

# Auto-detect GPU availability
if torch.cuda.is_available():
    DEVICE      = "cuda"
    TORCH_DTYPE = "float16"   # float16 saves VRAM on GPU
    BATCH_SIZE  = 16          # safe for A100/T4
elif torch.backends.mps.is_available():
    DEVICE      = "mps"       # Apple Silicon
    TORCH_DTYPE = "float32"   # float16 not fully supported on MPS
    BATCH_SIZE  = 8
else:
    DEVICE      = "cpu"
    TORCH_DTYPE = "float32"   # float16 not supported on CPU
    BATCH_SIZE  = 4           # CPU is slow; keep batch small

# ── Data ──────────────────────────────────────────────────────────────────────
# Dataset: pratyushmaini/llm_dataset_inference (auto-downloaded from HuggingFace)
# Model:   EleutherAI/pythia-410m-deduped     (auto-downloaded from HuggingFace)
SUBSET      = "wikipedia"   # wikipedia | arxiv | github
N_SAMPLES   = 2000          # full dataset (2000 per split)
SEQ_LENGTH  = 512           # maximum sequence length in tokens
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

# ── Paths (auto-detected) ─────────────────────────────────────────────────────
# Priority:
#   1. LLM_DI_BASE environment variable (manual override)
#   2. Google Colab  → /content/llm_dataset_inference/
#   3. Local         → <repo_root>/output/

def _detect_base() -> str:
    """
    Detect the runtime environment and return the appropriate base directory
    for checkpoints and results.

    Priority order:
        1. LLM_DI_BASE environment variable (explicit override)
        2. Google Colab  (/content/ exists) → /content/llm_dataset_inference/
        3. Local fallback → <repo_root>/output/

    Local usage example:
        export LLM_DI_BASE="/path/to/my/checkpoints"
        python main.py
    """
    # 1. Explicit override
    env_base = os.environ.get("LLM_DI_BASE")
    if env_base:
        return env_base

    # 2. Google Colab
    if os.path.exists("/content"):
        return "/content/llm_dataset_inference"

    # 3. Local fallback
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(repo_root, "output")


_BASE          = _detect_base()
CHECKPOINT_DIR = os.path.join(_BASE, "checkpoints")
RESULTS_DIR    = os.path.join(_BASE, "results")

# ── Runtime info ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device        : {DEVICE}")
    print(f"Dtype         : {TORCH_DTYPE}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Results dir   : {RESULTS_DIR}")