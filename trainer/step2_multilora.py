"""Step 2 – attach Multi‑LoRA & routing test."""
import torch, json, os
from ..gpt2_switchable import GPT2Switchable
from ..utils import log
from transformers.models.gpt2.modeling_gpt2 import Conv1D
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = GPT2Switchable().to(DEVICE)
    # Quick smoke test – toggle a random layer to 4‑bit + LoRA‑4
    target = next(n for n, m in model.named_modules() if isinstance(m, (torch.nn.Linear, Conv1D)))
    model.set_precision_config({target: 4})
    # forward dummy input
    tok = model.gpt2_tokenizer = __import__("transformers").AutoTokenizer.from_pretrained("gpt2")
    inp = tok("hello world", return_tensors="pt").to(DEVICE)
    _ = model(**inp)
    # --- requirement check
    ok = hasattr(dict(model.named_modules())[target], "lora")
    log("Step‑2 requirement check → " + json.dumps({"multi_lora_present": ok}))
    model.save_pretrained("checkpoints/step2_multilora")

if __name__ == "__main__":
    main()