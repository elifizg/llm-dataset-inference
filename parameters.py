# parameters.py
# Central configuration file. All settings are managed here.
# To change any parameter, only this file needs to be modified.

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "EleutherAI/pythia-410m-deduped"  # 410m-deduped | 1.4b-deduped | 6.9b-deduped | 12b-deduped
TORCH_DTYPE = "float16"                          # float16 (GPU) | float32 (CPU)
DEVICE      = "cuda"                             # cuda | cpu

# ── Data ──────────────────────────────────────────────────────────────────────
SUBSET      = "wikipedia_(en)"   # wikipedia_(en) | arxiv | github | pile_cc | pubmed_central
N_SAMPLES   = 1000               # number of examples for each of train and val splits
SEQ_LENGTH  = 512                # maximum sequence length in tokens
RANDOM_SEED = 42

# ── MIA Scores ────────────────────────────────────────────────────────────────
# Controls which MIA groups are computed.
COMPUTE_PERPLEXITY   = True
COMPUTE_ZLIB         = True
COMPUTE_MIN_K        = True
COMPUTE_MAX_K        = True
COMPUTE_PERTURBATION = False  # disabled for Phase 1; enabled in Phase 2
COMPUTE_REFERENCE    = False  # reference models are time-intensive; Phase 2

K_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # k values for min-k and max-k

# ── Training (Linear Regressor) ───────────────────────────────────────────────
OUTLIER_PERCENTILE = 2.5  # top and bottom percentile clipped before training
A_SPLIT_RATIO      = 0.5  # fraction of data used for A-split (regressor training)
N_SEEDS            = 10   # number of random seeds for repeated t-tests

# ── Testing ───────────────────────────────────────────────────────────────────
P_VALUE_THRESHOLD = 0.1  # p-value below this threshold → dataset was in training
FP_P_VALUE_MIN    = 0.5  # false positive test: p-value must exceed this value

# ── Batch ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 16  # 16-32 for A100; use 8 for T4

# ── Checkpoints and Results ───────────────────────────────────────────────────
DRIVE_BASE     = "/content/drive/MyDrive/llm_dataset_inference"
CHECKPOINT_DIR = f"{DRIVE_BASE}/checkpoints"
RESULTS_DIR    = f"{DRIVE_BASE}/results"
