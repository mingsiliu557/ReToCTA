# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from .moe import KeepTopK,BiasedDropout
from easydict import EasyDict
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL




class Cotta_Adapter(nn.Module):
    def __init__(self,num_experts,dim,k=4,drop="our"):
        super(Cotta_Adapter, self).__init__()
        TopK_Function = KeepTopK(top_k=int(k)) 
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(dim, num_experts),
            TopK_Function,
            nn.Softmax(dim=-1)
        )
        self.router_2 = nn.Sequential(
            nn.Linear(dim, num_experts),
            TopK_Function,
            nn.Softmax(dim=-1)
        )
        self.adaptmlp = nn.ModuleList([Adapter(config=EasyDict(d_model=dim), dropout=0.1, bottleneck=int(192),init_option="lora",adapter_scalar="0.8",adapter_layernorm_option="None",drop=drop) for i in range(self.num_experts)])
        self.biaseddrop = BiasedDropout(0.5)

    def forward(self, x, spatial_hw=None):
        """spatial_hw: optional (H, W) for pooled map before flatten; used when saving expert maps as [C,H,W]."""
        weights = self.router(x)  # [b, n, d] -> [b, n, e]
        weights_2 = self.router_2(self.biaseddrop(x, None, 0.5, True))
        high_expert_value = 0
        low_expert_value = 0
        expert_value = 0
        num_expert = self.num_experts
        v_list = [1/4, 1/2, 1/4, 1/2]
        for i in range(num_expert):
            weight_idx = weights[:, :, i].unsqueeze(dim=-1)
            biased_threshold = torch.mean(weight_idx)
            weight_idx_2 = weights_2[:, :, i].unsqueeze(dim=-1)

            if i <= 1:
                low_expert_value += weight_idx_2 * self.adaptmlp[i](x, None, v_list[i]+0.1*biased_threshold, largest=False)
            else:
                high_expert_value += weight_idx_2 * self.adaptmlp[i](x, None, v_list[i]+0.1*biased_threshold)

        expert_value = low_expert_value + high_expert_value
        self.last_spatial_hw = spatial_hw
        self.last_low_expert_value = low_expert_value.detach()
        self.last_high_expert_value = high_expert_value.detach()
        self.last_expert_value = expert_value.detach()
        return expert_value

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 drop="our"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        # self.dwconv = DWConv(self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.drop = drop

        self.ourdropout = BiasedDropout(p=0.5)
        # self.ourdropout = nn.Dropout(p=0.5)
        # if drop=="our":
        #     self.ourdroput = BiasedDropout(p=0.5)
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x,p1,p2, largest=False,add_residual=False, residual=None):
        residual = x

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        # down = self.dwconv(down, H, W)
        
        down = self.non_linear_func(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        down = self.ourdropout(down, p1, p2,largest)
        # down = self.ourdropout(down)
        
        up = self.up_proj(down)
        
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        output = up

        return output

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

        # 初始化权重为0
        init.constant_(self.dwconv.weight, 0)

        # 如果有偏置项，也初始化为0
        if self.dwconv.bias is not None:
            init.constant_(self.dwconv.bias, 0)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x





# import math
# from typing import Callable, Dict, List, Optional, Tuple

# import numpy as np
# import PIL
# import torch
# import torch.nn.functional as F

# import torch.nn as nn

# class ViDAInjectedLinear(nn.Module):
#     def __init__(self, in_features, out_features, bias=False, r=4, r2 = 64):
#         super().__init__()

#         self.linear_vida = nn.Linear(in_features, out_features, bias)
#         self.vida_down = nn.Linear(in_features, r, bias=False)
#         self.vida_up = nn.Linear(r, out_features, bias=False)
#         self.vida_down2 = nn.Linear(in_features, r2, bias=False)
#         self.vida_up2 = nn.Linear(r2, out_features, bias=False)
#         self.scale1 = 1.0
#         self.scale2 = 1.0

#         nn.init.normal_(self.vida_down.weight, std=1 / r**2)
#         nn.init.zeros_(self.vida_up.weight)

#         nn.init.normal_(self.vida_down2.weight, std=1 / r2**2)
#         nn.init.zeros_(self.vida_up2.weight)

#     def forward(self, input):
#         return self.linear_vida(input) + self.vida_up(self.vida_down(input)) * self.scale1 + self.vida_up2(self.vida_down2(input)) * self.scale2



# def inject_trainable_vida(
#     model: nn.Module,
#     target_replace_module: List[str] = ["CrossAttention", "Attention"],
#     r: int = 4,
#     r2: int = 16,
# ):
#     """
#     inject vida into model, and returns vida parameter groups.
#     """

#     require_grad_params = []
#     names = []

#     for _module in model.modules():
#         if _module.__class__.__name__ in target_replace_module:

#             for name, _child_module in _module.named_modules():
#                 if _child_module.__class__.__name__ == "Linear":

#                     weight = _child_module.weight
#                     bias = _child_module.bias
#                     _tmp = ViDAInjectedLinear(
#                         _child_module.in_features,
#                         _child_module.out_features,
#                         _child_module.bias is not None,
#                         r,
#                         r2,
#                     )
#                     _tmp.linear_vida.weight = weight
#                     if bias is not None:
#                         _tmp.linear_vida.bias = bias

#                     # switch the module
#                     _module._modules[name] = _tmp

#                     require_grad_params.extend(
#                         list(_module._modules[name].vida_up.parameters())
#                     )
#                     require_grad_params.extend(
#                         list(_module._modules[name].vida_down.parameters())
#                     )
#                     _module._modules[name].vida_up.weight.requires_grad = True
#                     _module._modules[name].vida_down.weight.requires_grad = True

#                     require_grad_params.extend(
#                         list(_module._modules[name].vida_up2.parameters())
#                     )
#                     require_grad_params.extend(
#                         list(_module._modules[name].vida_down2.parameters())
#                     )
#                     _module._modules[name].vida_up2.weight.requires_grad = True
#                     _module._modules[name].vida_down2.weight.requires_grad = True                    
#                     names.append(name)

#     return require_grad_params, names

# class BufferLayer(nn.Module):
#     def __init__(self, channels=64):
#         super(BufferLayer, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.alpha = nn.Parameter(torch.tensor(1e-4))
#         self.beta = nn.Parameter(torch.tensor(1e-4))

#     def forward(self, x):
#         out1 = self.conv1(x)  # 1x1 conv
#         out2 = self.conv2(x)  # 3x3 conv with padding=1
#         return self.alpha * out1 + self.beta * out2 + x