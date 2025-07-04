# lora.py  ───────────────────────────────────────────────────────────────
import math, torch, torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from .quant.fake_quant import fake_quant           # <- you already have this

class LoRAModule(nn.Module):
    """
    ONE rank-r LoRA adapter.  We quantise A & B each forward according to the
    *parent* projection layer’s bit_w.
    """
    def __init__(self, in_dim: int, out_dim: int, r: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.scale = 1.0 / math.sqrt(r)

    def forward(self, x: torch.Tensor, bit: int) -> torch.Tensor:
        # fake-quant LoRA weights to <bit> bits
        A_q = fake_quant(self.A, bit)
        B_q = fake_quant(self.B, bit)

        orig = x.shape                      # [..., in_dim]
        x_f  = x.reshape(-1, orig[-1])      # [N, in_dim]
        down = x_f  @ A_q.t()               # [N, r]
        up   = down @ B_q.t()               # [N, out_dim]
        delta = up.view(*orig[:-1], up.size(-1)) * self.scale
        return delta
# ------------------------------------------------------------------------
def attach_single_lora(model: nn.Module, r: int = 8):
    """
    Give EVERY Linear / Conv1D exactly **one** LoRA adapter.
    """
    for m in model.modules():
        if isinstance(m, (nn.Linear, Conv1D)):
            in_d, out_d = (m.in_features, m.out_features)    \
                          if isinstance(m, nn.Linear) else   \
                          m.weight.shape
            m.lora = LoRAModule(in_d, out_d, r=r)
    return model
