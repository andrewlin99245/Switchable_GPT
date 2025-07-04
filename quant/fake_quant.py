from __future__ import annotations
import torch
from torch.autograd.function import Function

class FakeQuant(Function):
    """Symmetric MinMax fake‑quant with STE – used by LLM‑QAT."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, bit: int):
        if bit == 32:
            return x  # FP32 bypass
        qmax = 2 ** (bit - 1) - 1
        scale = x.abs().max() / qmax + 1e-8
        ctx.save_for_backward(scale)
        return (x / scale).round().clamp(-qmax, qmax) * scale

    @staticmethod
    def backward(ctx, grad_out):
        (scale,) = ctx.saved_tensors
        return grad_out.clone(), None

def fake_quant(x: torch.Tensor, bit: int):
    return FakeQuant.apply(x, bit)