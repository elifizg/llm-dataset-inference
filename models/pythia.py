# models/pythia.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import parameters as P

def load_model(model_name: str = P.MODEL_NAME) -> tuple:
    """Load Pythia model and tokenizer. Returns (model, tokenizer)."""
    print(f"[pythia] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if P.TORCH_DTYPE == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=P.DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[pythia] Loaded — {n_params:.0f}M parameters")
    if P.DEVICE == "cuda":
        print(f"[pythia] GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer

def get_token_losses(model, tokenizer, texts: list, batch_size: int = P.BATCH_SIZE) -> list:
    """
    Compute per-token cross-entropy loss for each input text.

    Returns a list of lists — each inner list contains float loss values
    for the non-padding tokens of one text.
    """
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=P.SEQ_LENGTH)
        if P.DEVICE == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        shifted_logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        loss = loss.view(labels.size(0), labels.size(1) - 1)
        for entry in loss.tolist():
            clean = [x for x in entry if x != 0.0]
            if clean:
                loss_list.append(clean)
        if (i // batch_size) % 10 == 0:
            print(f"[pythia] Processed {min(i+batch_size, len(texts))}/{len(texts)}")
    return loss_list
