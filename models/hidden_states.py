# models/hidden_states.py
# Extracts internal hidden state representations from Pythia.
# Instead of surface MIA scores, uses the model's last hidden layer
# as a richer membership signal.

import torch
import numpy as np
import parameters as P


def extract_hidden_states(model,
                           tokenizer,
                           texts: list[str],
                           batch_size: int = P.BATCH_SIZE,
                           pooling: str = "mean") -> np.ndarray:
    """
    Extract mean-pooled hidden states from the last transformer layer.

    For each text, the model produces a hidden state tensor of shape
    (seq_len, hidden_dim). We reduce this to a single vector of shape
    (hidden_dim,) using mean pooling over non-padding tokens.

    This provides a richer membership signal than surface MIA scores:
    instead of asking 'how surprised is the model by this text?', we ask
    'how does the model internally represent this text?'

    Args:
        model:      Loaded Pythia model in eval mode (output_hidden_states
                    will be enabled inside this function).
        tokenizer:  Corresponding tokenizer with pad_token set.
        texts:      List of input strings.
        batch_size: Number of texts processed per forward pass.
        pooling:    Pooling strategy over token dimension.
                    'mean' — average over non-padding tokens (default).
                    'first' — use only the first token (CLS-like).

    Returns:
        representations: np.ndarray of shape (n_texts, hidden_dim).
                         Each row is the pooled hidden state for one text.
    """
    representations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=P.SEQ_LENGTH
        )
        attention_mask = inputs["attention_mask"]

        if P.DEVICE == "cuda":
            inputs         = {k: v.cuda() for k, v in inputs.items()}
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Last hidden layer: shape (batch, seq_len, hidden_dim)
        last_hidden = outputs.hidden_states[-1]

        if pooling == "mean":
            # Mask padding tokens before averaging
            mask         = attention_mask.unsqueeze(-1).float()
            sum_hidden   = (last_hidden * mask).sum(dim=1)
            count        = mask.sum(dim=1).clamp(min=1)
            pooled       = (sum_hidden / count).cpu().numpy()
        else:
            # First token only
            pooled = last_hidden[:, 0, :].cpu().numpy()

        representations.append(pooled)

        n_done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0:
            print(f"[hidden] Processed {n_done}/{len(texts)} texts")

    return np.vstack(representations)