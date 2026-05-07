# mia/perturbation.py
# Perturbation-based MIA scores — Phase 2 addition.
# Currently contains only function stubs with full docstrings.
# These will be activated by setting COMPUTE_PERTURBATION = True in parameters.py.

import random
import re


def whitespace_perturbation(text: str) -> str:
    """
    Replace single spaces with a random number of spaces (1–3).

    The hypothesis is that a model trained on the original text will assign
    lower perplexity to the original than to this whitespace-perturbed version.

    Args:
        text: Input string to perturb.

    Returns:
        Perturbed string with randomized whitespace.
    """
    return re.sub(r' +', lambda m: ' ' * random.randint(1, 3), text)


def underscore_trick(text: str) -> str:
    """
    Replace all spaces with underscores.

    Args:
        text: Input string to perturb.

    Returns:
        Perturbed string with underscores instead of spaces.
    """
    return text.replace(" ", "_")


def random_deletion(text: str, p: float = 0.1) -> str:
    """
    Randomly delete each word with probability p.

    Args:
        text: Input string to perturb.
        p:    Probability of deleting each word.

    Returns:
        Perturbed string with random words removed.
    """
    words = text.split()
    kept  = [w for w in words if random.random() > p]
    return " ".join(kept) if kept else text


def change_char_case(text: str) -> str:
    """
    Randomly flip the case of individual characters with probability 0.1.

    Args:
        text: Input string to perturb.

    Returns:
        Perturbed string with random case flips.
    """
    return "".join(
        c.upper() if c.islower() and random.random() < 0.1
        else c.lower() if c.isupper() and random.random() < 0.1
        else c
        for c in text
    )


def butter_fingers(text: str, p: float = 0.05) -> str:
    """
    Simulate typing errors by replacing characters with adjacent keyboard keys.

    Args:
        text: Input string to perturb.
        p:    Probability of replacing each character.

    Returns:
        Perturbed string with simulated typos.
    """
    keyboard = {
        'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'eft', 't': 'rgy',
        'a': 'qsz', 's': 'awdx', 'd': 'sefc', 'f': 'drgv', 'g': 'fthb',
    }
    result = []
    for c in text:
        if c.lower() in keyboard and random.random() < p:
            result.append(random.choice(keyboard[c.lower()]))
        else:
            result.append(c)
    return "".join(result)


# Registry of all perturbation functions available for Phase 2.
PERTURBATION_FUNCTIONS = {
    "whitespace_perturbation": whitespace_perturbation,
    "underscore_trick":        underscore_trick,
    "random_deletion":         random_deletion,
    "change_char_case":        change_char_case,
    "butter_fingers":          butter_fingers,
    # synonym_substitution: requires NLTK WordNet; added in Phase 2
}


def apply_all_perturbations(texts: list) -> dict:
    """
    Apply all registered perturbation functions to a list of texts.

    Args:
        texts: List of input strings to perturb.

    Returns:
        Dictionary mapping perturbation name to the list of perturbed strings.
        Example: {'whitespace_perturbation': [...], 'underscore_trick': [...]}
    """
    return {
        name: [fn(t) for t in texts]
        for name, fn in PERTURBATION_FUNCTIONS.items()
    }
