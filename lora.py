import math
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import Conv1D

class LoRAModule(nn.Module):
    """Rank-r LoRA adapter for Linear/Conv1D layers, handles 2D or 3D inputs."""
    def __init__(self, in_dim: int, out_dim: int, r: int = 8):
        super().__init__()
        self.r = r
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.scale = 1.0 / math.sqrt(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, in_dim] or [batch, in_dim]
        orig_shape = x.shape
        in_dim = orig_shape[-1]
        # flatten leading dims
        x_flat = x.reshape(-1, in_dim)         # [N, in_dim] where N = batch*seq_len or batch
        # down-project, up-project
        down = x_flat @ self.A.t()             # [N, r]
        up   = down   @ self.B.t()             # [N, out_dim]
        # restore original batch shape
        out = up.view(*orig_shape[:-1], up.size(-1))  # [..., out_dim]
        return out * self.scale


def attach_multi_lora(model: nn.Module, bits=[8, 4, 2], r=8):
    """Attach a per-bit ModuleDict of LoRA adapters to every Linear/Conv1D."""
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, Conv1D)):
            # determine in/out dims
            if isinstance(mod, nn.Linear):
                in_dim, out_dim = mod.in_features, mod.out_features
            else:
                # Conv1D.weight is [in_dim, out_dim]
                in_dim, out_dim = mod.weight.shape
            # create adapters
            mod.lora = nn.ModuleDict({
                str(b): LoRAModule(in_dim, out_dim, r=r)
                for b in bits
            })
    return model
