import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from .quant.llm_qat import patch_gpt2_for_qat
from .lora import attach_multi_lora
import torch.nn as nn

class GPT2Switchable(GPT2LMHeadModel):
    """GPT-2 wrapper that supports per-layer bit configs *and* multi-bit LoRA."""
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        cfg_bits: list[int] = [8, 4, 2],
        lora_rank: int | None = 8,
        layer2bit: dict[str, int] | None = None,
        **kwargs
    ) -> "GPT2Switchable":
        # 1. Load base GPT-2 weights
        model: GPT2Switchable = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # 2. Store precision config
        model.bits = cfg_bits
        model.active_bits = {
            name: 8
            for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Linear, Conv1D))
        }
        print(model.active_bits)  # Debug: print initial active bits
        # 3. Patch all linears for QAT at full 8-bit initially
        patch_gpt2_for_qat(model, bit_w=8, bit_a=8)

        # 3a. If user provided a per-layer map, apply it immediately
        if layer2bit is not None:
            model.set_precision_config(layer2bit)

        # 4. Attach LoRA branches if requested
        if lora_rank is not None:
            attach_multi_lora(model, bits=cfg_bits, r=lora_rank)

        return model

    def set_precision_config(self, layer2bit: dict[str, int]):
        """Update per-layer weight/activation bit-widths."""
        for name, bit in layer2bit.items():
            if name not in self.active_bits:
                raise KeyError(f"Layer '{name}' not found in model.")
            self.active_bits[name] = bit
            mod = dict(self.named_modules())[name]
            if isinstance(mod, (torch.nn.Linear, Conv1D)):
                # update fake-quant settings
                mod.bit_w = bit
                mod.bit_a = bit
            if hasattr(mod, "lora"):
                #print(f"[GPT2Switchable] Setting LoRA bit-width for {name} to {bit}")
                lora = mod.lora
                # case A: multi-LoRA (ModuleDict of branches)
                if isinstance(lora, nn.ModuleDict):
                    for branch in lora.values():
                        branch.bit_w = bit
                        branch.bit_a = bit
                # case B: single LoRA adapter
                else:
                    lora.bit_w = bit
                    lora.bit_a = bit

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
