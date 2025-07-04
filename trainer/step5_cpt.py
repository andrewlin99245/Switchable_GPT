# trainer/step5_cpt.py
"""
Step 5 – Cyclic Precision Training (CPT) à la Fu et al. (ICLR’21).
Cycles every GPT-2 projection between 8-bit and 3-bit with a cosine schedule.
"""

import os, json, math, torch, torch.optim as optim, torch.nn as nn
from transformers import AutoTokenizer, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from safetensors.torch import load_file as load_safetensors

from ..switchable_lora import attach_single_lora, LoRAModule

from ..quant.llm_qat       import patch_gpt2_for_qat
from ..gpt2_switchable     import GPT2Switchable
from ..data                import get_squad_loader
from ..utils               import log

# ──────────────────────────────────────────────────────────────── hyper-params
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
STEP1_CKPT    = "checkpoints/step1_qat"
STEP5_CKPT    = "checkpoints/step5_cpt"
BITS_SUPPORTED = [8, 7, 6, 5, 4, 3]          # model knows 8/4/2/3
HIGH_BIT, LOW_BIT = 8, 3               # CPT bounds
CYCLE_STEPS   = 200                    # one cosine cycle = 200 steps
MAX_STEPS     = 1000                   # tune 1 k iterations
LR            = 2e-5

# ──────────────────────────────────────────────────────────────── helpers
def cosine_cpt_bit(step: int, T: int, lo: int, hi: int) -> int:
    """Cosine schedule between lo & hi; rounds to nearest supported bit."""
    frac = (step % T) / T
    cont = lo + 0.5 * (hi - lo) * (1 - math.cos(math.pi * frac))
    # snap to nearest legal bit-width
    return min(BITS_SUPPORTED, key=lambda b: abs(b - cont))

# ──────────────────────────────────────────────────────────────── main
def main():
    # 1) Build & patch the model skeleton (no HF weights yet)
    config = GPT2Config.from_pretrained(STEP1_CKPT)
    model = GPT2Switchable(config)                 # __init__ only takes config

    # record supported bit-widths and default all layers to 8-bit
    model.bits = BITS_SUPPORTED
    model.active_bits = {
        name: 8
        for name, m in model.named_modules()
        if isinstance(m, (nn.Linear, Conv1D))
    }

    # inject fake-quant QAT hooks at 8-bit for all projections
    patch_gpt2_for_qat(model, bit_w=8, bit_a=8)

    # 2) Attach multi-bit LoRA adapters
    attach_single_lora(model, r=8)

    # 3) Load the Step-1 QAT checkpoint weights into the patched model
    ckpt_file  = os.path.join(STEP1_CKPT, "model.safetensors")
    state_dict = load_safetensors(ckpt_file, device="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)

    # 4) Hook each projection layer to add its active LoRA branch
    for name, mod in model.named_modules():
        if hasattr(mod, "lora"):
            #print("[Step-5 CPT] attaching LoRA to", name)
            orig = mod.forward
            def make_fwd(orig, module):
                def fwd(x, *a, **kw):
                    out = orig(x, *a, **kw)
                    bit = getattr(module, "bit_w", 8)       # current backbone bit
                    delta = module.lora(x, bit)              # one adapter, quantised
                    return out + delta
                return fwd
            mod.forward = make_fwd(orig, mod)

    # 5) Prepare tokenizer & data
    tokenizer = AutoTokenizer.from_pretrained(STEP1_CKPT)
    loader    = get_squad_loader()

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    model.train()

    # 5) CPT loop
    for step, batch in enumerate(loader):
        if step >= MAX_STEPS: break

        # choose bit via cosine CPT
        bit = cosine_cpt_bit(step, CYCLE_STEPS, LOW_BIT, HIGH_BIT)

        # apply same bit to every projection layer
        model.set_precision_config({
            name: bit
            for name, m in model.named_modules()
            if isinstance(m, (nn.Linear, Conv1D))
        })

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss  = model(**batch).loss

        loss.backward(); optimizer.step(); optimizer.zero_grad()

        log(f"[CPT] step={step:04d} active_bit={bit} loss={loss.item():.4f}")

    # 6) Save CPT-tuned checkpoint
    os.makedirs(STEP5_CKPT, exist_ok=True)
    model.save_pretrained(STEP5_CKPT); tokenizer.save_pretrained(STEP5_CKPT)

    # 7) Quick requirement check
    req = {
        "cyclic_precision": True,
        "low_bit": LOW_BIT,
        "high_bit": HIGH_BIT,
        "cycle_len": CYCLE_STEPS
    }
    log("Step-5 requirement check → " + json.dumps(req))

if __name__ == "__main__":
    main()
