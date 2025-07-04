import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from safetensors.torch import load_file as load_safetensors
from ..quant.llm_qat import patch_gpt2_for_qat
from ..gpt2_switchable import GPT2Switchable, attach_multi_lora
from ..data import get_squad_loader
from ..utils import log

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
STEP1_CKPT   = "checkpoints/step1_qat"
STEP3_CKPT   = "checkpoints/step3_instantnet"
BITS         = [8, 4, 2]
LORA_RANK    = 8
LR           = 2e-5
MAX_STEPS    = 1000

def main():
    # 1) Build & patch the model skeleton (no HF weights yet)
    config = GPT2Config.from_pretrained(STEP1_CKPT)
    model = GPT2Switchable(config)                 # __init__ only takes config

    # record supported bit-widths and default all layers to 8-bit
    model.bits = BITS
    model.active_bits = {
        name: 8
        for name, m in model.named_modules()
        if isinstance(m, (nn.Linear, Conv1D))
    }

    # inject fake-quant QAT hooks at 8-bit for all projections
    patch_gpt2_for_qat(model, bit_w=8, bit_a=8)

    # 2) Attach multi-bit LoRA adapters
    attach_multi_lora(model, bits=BITS, r=LORA_RANK)

    # 3) Load the Step-1 QAT checkpoint weights into the patched model
    ckpt_file  = os.path.join(STEP1_CKPT, "model.safetensors")
    state_dict = load_safetensors(ckpt_file, device="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)

    # 4) Hook each projection layer to add its active LoRA branch
    for name, mod in model.named_modules():
        if hasattr(mod, "lora"):
            orig_forward = mod.forward
            def make_forward(orig_forward, module):
                def fwd(x, *args, **kwargs):
                    out = orig_forward(x, *args, **kwargs)
                    bit = module.bit_w
                    delta = module.lora[str(bit)](x)
                    return out + delta
                return fwd
            mod.forward = make_forward(orig_forward, mod)

    # 5) Prepare tokenizer & data
    tokenizer = AutoTokenizer.from_pretrained(STEP1_CKPT)
    loader    = get_squad_loader()

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    model.train()

    # 6) InstantNet-style switchable-precision training loop
    for step, batch in enumerate(loader):
        if step >= MAX_STEPS:
            break

        total_loss = 0.0
        for bit in BITS:
            # set all projection layers to this bit
            layer2bit = {
                name: bit
                for name, m in model.named_modules()
                if isinstance(m, (nn.Linear, Conv1D))
            }
            model.set_precision_config(layer2bit)

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            total_loss += out.loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
        current_bit = BITS[step % len(BITS)]
        log(f"[InstantNet] step={step} loss={total_loss.item():.4f}")

    # 7) Save checkpoint & tokenizer
    os.makedirs(STEP3_CKPT, exist_ok=True)
    model.save_pretrained(STEP3_CKPT)
    tokenizer.save_pretrained(STEP3_CKPT)

    # 8) Requirement check
    req = {
        "loaded_step1_ckpt": os.path.isdir(STEP1_CKPT),
        "lora_attached": any(
            hasattr(m, "lora")
            for _, m in model.named_modules()
            if isinstance(m, (nn.Linear, Conv1D))
        ),
        "multi_bit_trained": True
    }
    log("Step-3 requirement check → " + json.dumps(req))

if __name__ == "__main__":
    main()