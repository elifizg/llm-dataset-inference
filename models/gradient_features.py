# models/gradient_features.py
# Gradient-based membership signal for dataset inference.
# Computes per-layer gradient norms for each input text.
# Training examples (members) produce systematically different gradient
# patterns compared to validation examples (non-members).
# This feature is not explored in Maini et al. (2024) and represents
# a novel deep learning contribution.

import torch
import numpy as np
import parameters as P


def compute_gradient_features(model,
                               tokenizer,
                               texts: list,
                               batch_size: int = 4) -> np.ndarray:
    """
    Compute per-layer gradient norm features for each input text.

    For each text, we perform a forward pass, compute the cross-entropy
    loss, and then backpropagate to obtain gradients with respect to all
    model parameters. We then extract the L2 norm of gradients for each
    named parameter group (layer), producing a fixed-size feature vector
    regardless of the model size.

    The intuition is that the loss landscape around training examples
    differs systematically from that around unseen examples: gradients
    for members tend to be smaller (the model is already near a local
    minimum for these examples) compared to non-members.

    Note: This function processes one text at a time due to memory
    constraints of storing per-text gradient information. batch_size
    controls how many texts are processed before clearing GPU cache.

    Args:
        model:      Loaded Pythia model. Must support gradient computation
                    (not in torch.no_grad() context).
        tokenizer:  Corresponding tokenizer with pad_token set.
        texts:      List of input strings.
        batch_size: Number of texts to process before clearing GPU cache.

    Returns:
        features: np.ndarray of shape (n_texts, n_layers).
                  Each row contains the gradient L2 norm per layer
                  for one input text.
    """
    loss_fct = torch.nn.CrossEntropyLoss()
    all_features = []

    # Get parameter group names once
    param_names = [name for name, param in model.named_parameters()
                   if param.requires_grad and "weight" in name]

    model.train()  # enable gradient computation

    for i, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=P.SEQ_LENGTH
        )
        if P.DEVICE == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Zero gradients
        model.zero_grad()

        # Forward pass
        outputs = model(**inputs)
        logits  = outputs.logits

        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Backward pass — compute gradients
        loss.backward()

        # Extract gradient norms per layer
        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name:
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
                else:
                    grad_norms.append(0.0)

        all_features.append(grad_norms)

        # Clear gradients and cache periodically
        model.zero_grad()
        if (i + 1) % batch_size == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[gradient] Processed {i + 1}/{len(texts)} texts")

    model.eval()  # restore eval mode

    features = np.array(all_features)
    print(f"[gradient] Done. Feature shape: {features.shape}")
    return features