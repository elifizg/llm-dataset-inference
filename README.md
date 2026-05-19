# LLM Dataset Inference: Did You Train on My Dataset?

A replication, extension, and deep learning contribution building on [Maini et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.06443).

---

## Overview

Large language models (LLMs) are trained on massive web scrapes, raising serious copyright and privacy concerns. This project addresses a core question: **can we statistically prove that a specific dataset was used to train an LLM?**

We replicate the Dataset Inference method of Maini et al. (2024), extend it with three novel deep learning contributions, and benchmark all methods against state-of-the-art baselines including Min-k%++ (ICLR 2025).

### Key Finding

Individual Membership Inference Attacks (MIAs) fail under IID conditions (AUC ≈ 0.50). Our adaptive ensemble methods — especially the Hidden State MLP and Gradient-based MLP — achieve statistically significant detection (p < 0.001) with zero false positives.

---

## Method Comparison

| Method | Input | Combined p-value | Detected | FP OK |
|--------|-------|-----------------|----------|-------|
| B1: Perplexity | MIA score | >0.1 | ✗ | — |
| B1: zlib Ratio | MIA score | >0.1 | ✗ | — |
| B2: Min-k% (k=0.1) | MIA score | >0.1 | ✗ | — |
| B3: Min-k%++ | MIA score | ~0.000 | ✓* | ✗ |
| M1: Linear Ensemble | 16 MIA scores | 0.0941 | ✓ | ✓ |
| M2: Baseline MLP | 16 MIA scores | 0.0001 | ✓ | ✓ |
| M3: Improved MLP | 16 MIA scores | 0.000016 | ✓ | ✓ |
| M4: Hidden State MLP | 1024-dim hidden states | ~0.000 | ✓ | ✓ |
| M5: Gradient-based MLP | 147 gradient norms | 0.000491 | ✓ | ✓ |

*Min-k%++ detects but fails the false positive test (FP p=0.275 < 0.5).

---

## Project Structure

```
llm-dataset-inference/
├── main.py                  # Full pipeline orchestration
├── train.py                 # Regressor training on A-split
├── test.py                  # T-test and false positive evaluation
├── parameters.py            # All configuration in one place
│
├── data/
│   └── loader.py            # Data loading from official Pile dataset
│
├── models/
│   ├── pythia.py            # Pythia model wrapper + token loss extraction
│   ├── regressor.py         # Linear regressor (M1)
│   ├── mlp.py               # Improved MLP with BCE, BatchNorm, early stopping
│   ├── hidden_states.py     # Hidden state extraction (M4)
│   └── gradient_features.py # Gradient norm extraction (M5)
│
├── mia/
│   ├── scores.py            # 16 MIA score computation
│   └── perturbation.py      # Perturbation functions (Phase 2)
│
└── utils/
    ├── checkpoint.py        # Google Drive save/load
    └── visualization.py     # Result plots
```

---

## Dataset

We use the official dataset released by Maini et al.:

```python
from datasets import load_dataset
ds_train = load_dataset("pratyushmaini/llm_dataset_inference",
                         name="wikipedia", split="train")
ds_val   = load_dataset("pratyushmaini/llm_dataset_inference",
                         name="wikipedia", split="val")
```

- **Train split** (member): 2,000 texts from Pythia's training data
- **Val split** (non-member): 2,000 texts from the same domain and time period
- **IID condition**: both splits come from the same distribution — no temporal shift

---

## Model

We use [Pythia-410M-deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped) from EleutherAI. Pythia is the ideal evaluation model because its exact training data (The Pile) and training/validation splits are publicly known.

---

## Reproducing Results

### 1. Setup

```python
# In Google Colab
!pip install transformers datasets accelerate scikit-learn scipy matplotlib huggingface_hub -q
!git clone https://github.com/elifizg/llm-dataset-inference.git

import sys
sys.path.insert(0, '/content/llm-dataset-inference')
```

### 2. Mount Drive and configure paths

```python
from google.colab import drive
drive.mount('/content/drive')

import parameters, os
parameters.CHECKPOINT_DIR = '/content/drive/MyDrive/llm_dataset_inference/checkpoints'
parameters.RESULTS_DIR    = '/content/drive/MyDrive/llm_dataset_inference/results'
os.makedirs(parameters.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(parameters.RESULTS_DIR,    exist_ok=True)
```

### 3. Fix random seeds (reproducibility)

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

### 4. Run the full pipeline

```python
from main import run
results = run(subset="wikipedia_(en)", model_tag="410m")
```

All intermediate results (loss lists, feature matrices) are checkpointed to Drive. If the session disconnects, re-running the pipeline will load from cache automatically.

---

## Deep Learning Contributions

### M3: Improved MLP
Replaces the linear regressor with a 2-layer MLP using:
- **BCE loss** instead of MSE (appropriate for binary targets)
- **BatchNorm1d** for training stability across seeds
- **ReduceLROnPlateau** scheduler — halves LR when loss plateaus
- **Early stopping** — restores best weights, prevents overfitting

### M4: Hidden State MLP
Instead of surface MIA scores, extracts Pythia's **internal last-layer hidden states** (1024-dim) via mean pooling over non-padding tokens. Trains a classifier on these representations — a genuine "look inside the model" approach.

### M5: Gradient-based MLP (Novel)
Computes **per-layer gradient L2 norms** for each text via backpropagation. Training examples produce characteristically smaller gradients (the model is near a local minimum for its training data). This feature is not explored in Maini et al. (2024).

---

## Requirements

```
torch>=2.0
transformers>=4.35
datasets>=2.14
scikit-learn>=1.3
scipy>=1.11
matplotlib>=3.7
huggingface_hub>=0.19
```

**Compute:** Google Colab Pro (A100 40GB) recommended. Pythia-410M fits on T4 (16GB).

---

## References

- Maini, P., Jia, H., Papernot, N., & Dziedzic, A. (2024). *LLM Dataset Inference: Did you train on my dataset?* NeurIPS 2024. [arXiv:2406.06443](https://arxiv.org/abs/2406.06443)
- Shi, W. et al. (2024). *Detecting pretraining data from large language models.* ICLR 2024.
- Zhang, H. et al. (2025). *Min-K%++: Improved baseline for detecting pre-training data of LLMs.* ICLR 2025 (Spotlight).
- Biderman, S. et al. (2023). *Pythia: A suite for analyzing large language models across training and scaling.* ICML 2023.
- Gao, L. et al. (2020). *The Pile: An 800GB dataset of diverse text for language modeling.* arXiv:2101.00027.
