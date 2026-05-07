# data/loader.py
# Loads member and non-member text samples from The Pile dataset.
# Member texts are from the Pythia training split; non-member texts are from
# the validation split of the same domain and time period (IID condition).

import numpy as np
from datasets import load_dataset
import parameters as P


# Mapping from short parameter names to the full domain names used in the dataset.
SUBSET_MAP = {
    "wikipedia_(en)": "Wikipedia (en)",
    "arxiv":          "ArXiv",
    "github":         "Github",
    "pile_cc":        "Pile-CC",
    "pubmed_central": "PubMed Central",
    "hackernews":     "HackerNews",
    "dm_mathematics": "DM Mathematics",
    "freelaw":        "FreeLaw",
}


def load_pile_iid(subset: str = "Wikipedia (en)",
                  n_samples: int = P.N_SAMPLES,
                  seed: int = P.RANDOM_SEED) -> tuple[list, list]:
    """
    Load member and non-member texts from ArmelR/the-pile-splitted.

    This dataset provides The Pile split into 22 domain-specific subsets in
    Parquet format (no loading script required). The train split contains
    texts that were used to train Pythia (members); the test split contains
    texts that were not used (non-members). Both splits come from the same
    domain and time period, satisfying the IID condition required for valid
    dataset inference evaluation.

    Args:
        subset:    Full domain name as used in the dataset (e.g. 'Wikipedia (en)').
        n_samples: Number of texts to collect for each of train and val.
        seed:      Random seed for reproducibility.

    Returns:
        train_texts: List of member texts (from the training split).
        val_texts:   List of non-member texts (from the test split).
    """
    print(f"[loader] Loading Pile subset: {subset}")

    ds_train = load_dataset(
        "ArmelR/the-pile-splitted",
        split="train",
        streaming=True
    ).filter(lambda x: x["domain"] == subset)

    ds_val = load_dataset(
        "ArmelR/the-pile-splitted",
        split="test",
        streaming=True
    ).filter(lambda x: x["domain"] == subset)

    np.random.seed(seed)
    train_texts: list[str] = []
    val_texts:   list[str] = []

    for ex in ds_train:
        if len(train_texts) >= n_samples:
            break
        text = ex["text"].strip()
        if len(text.split()) >= 20:
            train_texts.append(text)

    for ex in ds_val:
        if len(val_texts) >= n_samples:
            break
        text = ex["text"].strip()
        if len(text.split()) >= 20:
            val_texts.append(text)

    print(f"[loader] Train: {len(train_texts)} | Val: {len(val_texts)}")
    _sanity_check(train_texts, val_texts)
    return train_texts, val_texts


def load_mimir(subset: str = P.SUBSET,
               n_samples: int = P.N_SAMPLES,
               seed: int = P.RANDOM_SEED,
               **kwargs) -> tuple[list, list]:
    """
    Main data loading function. Converts the short subset name from
    parameters.py to the full domain name and delegates to load_pile_iid.

    Args:
        subset:    Short subset name defined in parameters.py (e.g. 'wikipedia_(en)').
        n_samples: Number of texts to collect for each split.
        seed:      Random seed for reproducibility.

    Returns:
        train_texts: List of member texts.
        val_texts:   List of non-member texts.
    """
    full_name = SUBSET_MAP.get(subset, subset)
    return load_pile_iid(subset=full_name, n_samples=n_samples, seed=seed)


def _sanity_check(train_texts: list, val_texts: list) -> None:
    """
    Run basic sanity checks on loaded texts.

    Checks for short texts (fewer than 20 words) and approximate overlap
    between train and val sets based on the first 50 characters of each text.
    Prints warnings if issues are found.

    Args:
        train_texts: List of member texts.
        val_texts:   List of non-member texts.
    """
    short_train = sum(1 for t in train_texts if len(t.split()) < 20)
    short_val   = sum(1 for t in val_texts   if len(t.split()) < 20)

    if short_train > 0:
        print(f"[loader] WARNING: {short_train} short texts in train split (<20 words)")
    if short_val > 0:
        print(f"[loader] WARNING: {short_val} short texts in val split (<20 words)")

    train_prefixes = set(t[:50] for t in train_texts)
    val_prefixes   = set(t[:50] for t in val_texts)
    overlap = len(train_prefixes & val_prefixes)

    if overlap > 0:
        print(f"[loader] WARNING: {overlap} overlapping texts between train and val splits!")
    else:
        print("[loader] OK: No overlap between train and val splits.")
