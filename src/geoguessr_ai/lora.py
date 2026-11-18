"""Minimal LoRA for the HF CLIP vision tower (no peft dependency).

y = W x + b + (alpha / r) * B A x, with A (r x in) Kaiming-uniform and B (out x r) zero, so a
wrapped model computes exactly the base model at init. Only A and B are trainable (fp32, also
when the base weights are fp16); everything else is frozen by add_lora.

Blocks are vision_model.encoder.layers[i]; `blocks` = K wraps only the last K blocks (0 = all),
so the frozen lower blocks build no autograd graph (their input never requires grad).
"""

import math

import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        dx = x.to(self.lora_a.dtype) if not torch.is_autocast_enabled() else x
        return self.base(x) + (self.scale * (dx @ self.lora_a.t() @ self.lora_b.t())).to(x.dtype)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """The base Linear with B A folded into its weight (for export); summed in fp32, so small
        deltas are not rounded away before the add."""
        w = (self.base.weight.float() + self.scale * self.lora_b.float() @ self.lora_a.float()).to(self.base.weight.dtype)
        out = nn.Linear(self.base.in_features, self.base.out_features, self.base.bias is not None,
                        device=w.device, dtype=w.dtype)
        out.weight.copy_(w)
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def add_lora(vision: nn.Module, rank: int, alpha: float | None = None,
             targets=DEFAULT_TARGETS, blocks: int = 0) -> list[str]:
    """Freeze `vision`, wrap the `targets` Linears of the last `blocks` blocks (0 = all).
    Returns the wrapped module names."""
    for p in vision.parameters():
        p.requires_grad_(False)
    layers = vision.vision_model.encoder.layers
    first = 0 if blocks <= 0 else max(0, len(layers) - blocks)
    wrapped = []
    for i in range(first, len(layers)):
        for name, mod in list(layers[i].named_modules()):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in targets and isinstance(mod, nn.Linear):
                parent = layers[i].get_submodule(name.rsplit(".", 1)[0]) if "." in name else layers[i]
                setattr(parent, leaf, LoRALinear(mod, rank, alpha or rank))
                wrapped.append(f"layers.{i}.{name}")
    if not wrapped:
        raise ValueError(f"no Linear named {targets} in blocks {first}..{len(layers) - 1}")
    return wrapped


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in module.named_parameters() if n.endswith(("lora_a", "lora_b"))]
