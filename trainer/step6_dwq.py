import torch, random, json
from ..gpt2_switchable import GPT2Switchable
from ..data import get_squad_loader
from ..utils import log
from transformers.models.gpt2.modeling_gpt2 import Conv1D
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BITS = [8, 4, 2]

def rand_cfg():
    return {n: random.choice(BITS) for n, _ in model.named_modules() if isinstance(_, (torch.nn.Linear, Conv1D))}

def main():
    global model
    model = GPT2Switchable(cfg_bits=BITS).to(DEVICE)
    loader = get_squad_loader()
    atk_success = 0; total = 0  # dummy robustness metric
    for step, batch in enumerate(loader):
        if step >= 100:
            break
        # Random precision per DWQ
        model.set_precision_config(rand_cfg())
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        _ = model(**batch)  # forward pass
        total += 1
    log("Step‑6 DWQ run complete – random precision applied every step ✅")

if __name__ == "__main__":
    main()