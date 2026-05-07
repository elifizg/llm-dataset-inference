# utils/checkpoint.py
# Save and load intermediate results to Google Drive.
# All expensive computations (loss lists, feature matrices) are checkpointed
# so that a session restart does not require recomputation from scratch.

import os
import json
import numpy as np
import parameters as P


def _ensure_dirs() -> None:
    """Create checkpoint and results directories if they do not exist."""
    os.makedirs(P.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(P.RESULTS_DIR,    exist_ok=True)


def save_loss_list(loss_list: list,
                   split: str,
                   subset: str,
                   model_tag: str) -> None:
    """
    Save per-token loss lists to a JSON file on Drive.

    Args:
        loss_list:  List of per-token loss lists produced by get_token_losses().
        split:      Data split identifier, either 'train' or 'val'.
        subset:     Dataset subset name (e.g. 'wikipedia_en', 'arxiv').
        model_tag:  Short model identifier (e.g. '410m', '1.4b').
    """
    _ensure_dirs()
    path = os.path.join(P.CHECKPOINT_DIR, f"loss_{split}_{subset}_{model_tag}.json")
    with open(path, "w") as f:
        json.dump(loss_list, f)
    print(f"[checkpoint] Saved: {path}")


def load_loss_list(split: str,
                   subset: str,
                   model_tag: str) -> list | None:
    """
    Load a previously saved per-token loss list from Drive.

    Args:
        split:      Data split identifier, either 'train' or 'val'.
        subset:     Dataset subset name.
        model_tag:  Short model identifier.

    Returns:
        The loaded loss list, or None if the file does not exist.
    """
    path = os.path.join(P.CHECKPOINT_DIR, f"loss_{split}_{subset}_{model_tag}.json")
    if not os.path.exists(path):
        print(f"[checkpoint] Not found: {path}")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    print(f"[checkpoint] Loaded: {path} ({len(data)} texts)")
    return data


def save_features(features: np.ndarray,
                  split: str,
                  subset: str,
                  model_tag: str) -> None:
    """
    Save an MIA feature matrix to a .npy file on Drive.

    Args:
        features:   Feature matrix of shape (n_samples, n_scores).
        split:      Data split identifier, either 'train' or 'val'.
        subset:     Dataset subset name.
        model_tag:  Short model identifier.
    """
    _ensure_dirs()
    path = os.path.join(P.CHECKPOINT_DIR, f"features_{split}_{subset}_{model_tag}.npy")
    np.save(path, features)
    print(f"[checkpoint] Features saved: {path} — shape {features.shape}")


def load_features(split: str,
                  subset: str,
                  model_tag: str) -> np.ndarray | None:
    """
    Load a previously saved MIA feature matrix from Drive.

    Args:
        split:      Data split identifier, either 'train' or 'val'.
        subset:     Dataset subset name.
        model_tag:  Short model identifier.

    Returns:
        The loaded feature matrix as a numpy array, or None if not found.
    """
    path = os.path.join(P.CHECKPOINT_DIR, f"features_{split}_{subset}_{model_tag}.npy")
    if not os.path.exists(path):
        print(f"[checkpoint] Not found: {path}")
        return None
    data = np.load(path)
    print(f"[checkpoint] Features loaded: {path} — shape {data.shape}")
    return data


def save_results(results: dict,
                 subset: str,
                 model_tag: str) -> None:
    """
    Save t-test results and AUC values to a JSON file on Drive.

    Args:
        results:   Dictionary containing p-values, AUCs, and other metrics.
        subset:    Dataset subset name.
        model_tag: Short model identifier.
    """
    _ensure_dirs()
    path = os.path.join(P.RESULTS_DIR, f"results_{subset}_{model_tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[checkpoint] Results saved: {path}")
