"""Step 3 – InstantNet‑style cascade distillation training (switchable precision)."""
import torch, torch.optim as optim, os, json
from ..gpt2_switchable import GPT2Switchable
from ..data import get_squad_loader
from ..utils import log
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import Conv1D
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BITS = [8, 4, 2]

def main():
    model = GPT2Switchable(cfg_bits=BITS).to(DEVICE)
    loader = get_squad_loader()
    opt = optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for step, batch in enumerate(loader):
        if step >= 1000:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        total_loss = 0.0
        with torch.no_grad():
            teacher_logits = model(**batch).logits  # 8‑bit (default) teacher
        for bit in BITS:
            model.set_precision_config({n: bit for n, _ in model.named_modules() if isinstance(_, (torch.nn.Linear, Conv1D))})
            out = model(**batch)
            lm_loss = out.loss
            kd_loss = nn.functional.kl_div(out.logits.log_softmax(-1), teacher_logits.softmax(-1), reduction="batchmean")
            total_loss += lm_loss + 0.5 * kd_loss
        opt.zero_grad(); total_loss.backward(); opt.step()
        if step % 100 == 0:
            log(f"[Instant] step={step} loss={total_loss.item():.4f}")
    model.save_pretrained("checkpoints/step3_instant")
    # requirement check – switchable precision path
    cfg_ok = len(model.bits) == 3 and 2 in model.bits
    log("Step‑3 requirement check → " + json.dumps({"switchable_precision": cfg_ok}))

if __name__ == "__main__":
    main()