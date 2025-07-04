import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from .fake_quant import fake_quant

class QATLinear(nn.Linear):
    """Linear layer with weight/activation/KV fake-quant."""
    def __init__(self, in_f, out_f, bias=True, bit_w=8, bit_a=8):
        super().__init__(in_f, out_f, bias)
        self.bit_w, self.bit_a = bit_w, bit_a

    def forward(self, x):
        w_q = fake_quant(self.weight, self.bit_w)
        x_q = fake_quant(x, self.bit_a)
        return nn.functional.linear(x_q, w_q, self.bias)

def patch_gpt2_for_qat(model, bit_w=8, bit_a=8):
    """
    Replace every nn.Linear and Conv1D with QATLinear, preserving weights & bias.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Linear, Conv1D)):
            # 1) Figure out dims & original tensors
            if isinstance(module, nn.Linear):
                in_f, out_f = module.in_features, module.out_features
                weight_data = module.weight.data
                bias_data   = module.bias.data if module.bias is not None else None
            else:  # Conv1D: weight.shape = [in_f, out_f]
                in_f, out_f = module.weight.size(0), module.weight.size(1)
                # QATLinear(in_f, out_f) expects weight shape [out_f, in_f],
                # so we must transpose Conv1D.weight
                weight_data = module.weight.data.transpose(0, 1).contiguous()
                bias_data   = module.bias.data

            # 2) Build QATLinear
            qat = QATLinear(in_f, out_f, bias=(bias_data is not None),
                            bit_w=bit_w, bit_a=bit_a)
            # copy over parameters
            qat.weight.data.copy_(weight_data)
            if bias_data is not None:
                qat.bias.data.copy_(bias_data)

            # 3) Swap it into the model
            parent, attr = model, name.split(".")[-1]
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p)
            setattr(parent, attr, qat)

    return model
