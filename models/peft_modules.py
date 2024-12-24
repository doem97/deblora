import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F


class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(
            6.0 / float(3 * reduce(mul, patch_size, 1) + emb_dim)
        )
        nn.init.uniform_(self.prompt, -init_val, init_val)

    @property
    def dtype(self):
        return self.prompt.dtype

    def forward(self, x):
        x = x[:, : self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x


class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x


class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, alpha=1, dtype=None):
        super().__init__()
        self.lora_A = nn.Parameter(
            torch.zeros(in_dim, bottle_dim, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(bottle_dim, in_dim, dtype=dtype)
        )
        self.alpha = alpha
        self.scaling = self.alpha / bottle_dim

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class MLPLoRA(LoRA):
    def __init__(self, in_dim, bottle_dim, out_dim, alpha=1, dtype=None):
        super().__init__(in_dim, bottle_dim, alpha=alpha, dtype=dtype)
        self.lora_B = nn.Parameter(
            torch.zeros(bottle_dim, out_dim, dtype=dtype)
        )
        nn.init.zeros_(self.lora_B)


class FLoRA(nn.Module):
    """
    Fine-grained LoRA provides control over position-wise rank, alpha, and lr.

    Args:
        in_dim (int): Input dimension.
        rank (int): Rank of the low-rank approximation.
        out_dim (int, optional): Output dimension. If None, it's set to in_dim. Default: None.
        dtype (torch.dtype, optional): Data type of the parameters. Default: None.
    """

    def __init__(self, in_dim, rank, alpha=None, out_dim=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.out_dim = out_dim if out_dim is not None else in_dim

        self.lora_A = nn.Parameter(
            torch.zeros(self.in_dim, self.rank, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.rank, self.out_dim, dtype=dtype)
        )
        self.alpha = rank if alpha is None else alpha
        # alpha controls the magnitude of LoRA updates relative to
        # the original weights (higher alpha -> stronger adaptation)
        # Usually alpha is set to rank, i.e., scaling = 1
        # If alpha > rank, scaling > 1, i.e., stronger adaptation
        # If alpha < rank, scaling < 1, i.e., weaker adaptation
        self.scaling = self.alpha / self.rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(in_dim, dtype=dtype))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, x):
        if len(x.shape) == 4:  # for CNN
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(
                1, -1, 1, 1
            )
        else:
            return x * self.scale + self.shift


class MaskedLinear(nn.Module):
    def __init__(self, weight, bias, ratio=0.0, generator=None):
        super().__init__()
        # weight: (out_dim, in_dim)
        # bias: (out_dim)
        out_dim, in_dim = weight.shape
        num_params = out_dim * in_dim + out_dim
        ratio = float(eval(ratio)) if isinstance(ratio, str) else float(ratio)
        num_masked = int(num_params * ratio)

        # randomly select the optimized parameters
        masked_indexs = torch.randperm(num_params, generator=generator)[
            :num_masked
        ]
        mask = torch.zeros(num_params, dtype=bool).scatter(
            dim=0, index=masked_indexs, value=True
        )
        mask = mask.reshape(out_dim, in_dim + 1)
        self.mask_weight = mask[:, :-1]
        self.mask_bias = mask[:, -1]

        self.optimized_weight = nn.Parameter(
            torch.masked_select(weight.detach(), mask=self.mask_weight)
        )
        self.optimized_bias = nn.Parameter(
            torch.masked_select(bias.detach(), mask=self.mask_bias)
        )

    def forward(self, x, weight, bias):
        self.mask_weight = self.mask_weight.to(weight.device)
        self.mask_bias = self.mask_bias.to(bias.device)

        if self.mask_weight.sum() > 0:
            weight = torch.masked_scatter(
                weight, mask=self.mask_weight, source=self.optimized_weight
            )
        if self.mask_bias.sum() > 0:
            bias = torch.masked_scatter(
                bias, mask=self.mask_bias, source=self.optimized_bias
            )
        return F.linear(x, weight, bias)
