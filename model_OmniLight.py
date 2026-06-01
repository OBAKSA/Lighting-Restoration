# ------------------------------------------------------------------------
# NAFNet: Copyright (c) 2022 megvii-model. All Rights Reserved.
#  MIT License.
#
#  Modifications made by Youngjin Oh
# ------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# This file incorporates work from Restormer, FFTformer, covered by the following
# copyright and permission notice:
#
#   Restormer : Copyright (c) 2022 Syed Waqas Zamir and contributors
#   FFTformer : Copyright (c) 2023 kkkls
#   Licensed under the MIT License.
#   
#   Modifications made by Youngjin Oh
# ---------------------------------------------------------------------------
# This file incorporates work from MoCE-IR covered by the following
# copyright and permission notice:
#
#   MoCE-IR : Copyright (c) 2025 Computer Vision Lab, University of Wurzburg Licensed under CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International) (the "License"); 
#   you may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/legalcode
#   The code is released for academic research use only. For commercial use, please contact Computer Vision Lab, University of Wurzburg. 
#   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#   See the License for the specific language governing permissions and limitations under the License.
#
#   Original code at: https://github.com/eduardzamfir/MoCE-IR
#
#   Modifications made by Youngjin Oh
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import numbers
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal

device = torch.device('cuda')


#### LAYERNORM ver 2 ####
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


## Resizing modules ##
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


## SimpleGate from NAFNet
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


## Feature Fusion (Pixelwise Softmax) DINOv2
class FeatureFusion(nn.Module):
    def __init__(self, dim_dino=768, dim=32):
        super().__init__()
        self.predictor_shallow = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=dim_dino, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.predictor_middle = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=dim_dino, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.predictor_final = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=dim_dino, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv_dino = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=dim_dino, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.norm_dino = LayerNorm2d(dim)

    def forward(self, feature_dino):
        feature_dino = feature_dino.float()
        shallow, middle, final = feature_dino.chunk(3, dim=1)

        f_shallow = self.predictor_shallow(shallow)
        f_middle = self.predictor_middle(middle)
        f_final = self.predictor_final(final)

        f_dino = torch.cat([f_shallow, f_middle, f_final], dim=1)

        weight = F.softmax(f_dino, dim=1)

        fused_feature = weight[:, 0, :, :].unsqueeze(1) * shallow + weight[:, 1, :, :].unsqueeze(1) * middle \
                        + weight[:, 2, :, :].unsqueeze(1) * final

        fused_feature = self.norm_dino(self.conv_dino(fused_feature))

        return fused_feature


class FeatureFusionV2(nn.Module):
    def __init__(self, dim_dino=768, dim=36):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=dim_dino * 3,
                      out_channels=3,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False,
                      groups=3
                      )
        )
        self.conv_dino = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=dim_dino, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.norm_dino = LayerNorm2d(dim)

    def forward(self, feature_dino):
        feature_dino = feature_dino.float()
        shallow, middle, final = feature_dino.chunk(3, dim=1)

        weights = self.predictor(feature_dino)
        weights = F.softmax(weights, dim=1)

        fused_feature = weights[:, 0:1, :, :] * shallow + weights[:, 1:2, :, :] * middle + weights[:, 2:3, :, :] * final

        fused_feature = self.norm_dino(self.conv_dino(fused_feature))

        return fused_feature


##########################################################################
## Feed-Forward Network (FFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim * ffn_expansion_factor, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dim * ffn_expansion_factor // 2, out_channels=dim, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)

        self.sg = SimpleGate()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.sg(x)
        x = self.conv2(x)
        return x


## Gated-Dconv Feed-Forward Network (GDFN)
class GatedDconvFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention + auxiliary Cross-Attention w/ DINOv2 (Spatial domain)
class Attention_Spat_Fuse(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature_self = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_cross = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvq_x = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False)
        self.qkvq_x_dwconv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4, bias=False)

        self.kv_dino = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.kv_dino_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                        bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)

    def forward(self, x, dino):
        b, c, h, w = x.shape

        qkvq_x = self.qkvq_x_dwconv(self.qkvq_x(x))
        q, k, v, q_ = qkvq_x.chunk(4, dim=1)

        kv_dino = self.kv_dino_dwconv(self.kv_dino(dino))
        k_dino, v_dino = kv_dino.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_ = rearrange(q_, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k_dino = rearrange(k_dino, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_dino = rearrange(v_dino, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # self-attention
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        self_attn = (q @ k.transpose(-2, -1)) * self.temperature_self
        self_attn = self_attn.softmax(dim=-1)

        out_self = (self_attn @ v)
        out_self = rearrange(out_self, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # cross-attention
        q_ = torch.nn.functional.normalize(q_, dim=-1)
        k_dino = torch.nn.functional.normalize(k_dino, dim=-1)

        cross_attn = (q_ @ k_dino.transpose(-2, -1)) * self.temperature_cross
        cross_attn = cross_attn.softmax(dim=-1)

        out_cross = (cross_attn @ v_dino)
        out_cross = rearrange(out_cross, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        alpha = torch.sigmoid(self.alpha)
        out = out_self + out_cross * alpha

        out = self.project_out(out)
        return out


## Multi-DConv Head FFT-based Self-Attention + auxiliary Cross-Attention w/ DINOv2 (Frequency Domain)
class Attention_Freq_Fuse(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature_self = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_cross = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvq_x = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False)
        self.qkvq_x_dwconv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4, bias=False)

        self.kv_dino = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.kv_dino_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                        bias=False)

        self.norm_self = LayerNorm2d(dim // num_heads)
        self.norm_cross = LayerNorm2d(dim // num_heads)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)

        self.patch_size = 8

    def forward(self, x, dino):
        b, c, h, w = x.shape

        qkvq_x = self.qkvq_x_dwconv(self.qkvq_x(x))
        q, k, v, q_ = qkvq_x.chunk(4, dim=1)

        kv_dino = self.kv_dino_dwconv(self.kv_dino(dino))
        k_dino, v_dino = kv_dino.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                      patch1=self.patch_size, patch2=self.patch_size)
        k = rearrange(k, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                      patch1=self.patch_size, patch2=self.patch_size)

        q_ = rearrange(q_, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                       patch1=self.patch_size, patch2=self.patch_size)
        k_dino = rearrange(k_dino, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2',
                           head=self.num_heads, patch1=self.patch_size, patch2=self.patch_size)

        # self-attention in frequency
        q_fft = torch.fft.rfft2(q.float(), norm='ortho')
        k_fft = torch.fft.rfft2(k.float(), norm='ortho')

        self_attn_fft = q_fft * k_fft
        self_attn = torch.fft.irfft2(self_attn_fft, s=(self.patch_size, self.patch_size), norm='ortho')
        out_self = rearrange(self_attn, 'b (head c) h w patch1 patch2 -> b head c (h patch1 w patch2)',
                             head=self.num_heads, patch1=self.patch_size, patch2=self.patch_size)
        out_self = out_self * self.temperature_self

        out_self = rearrange(out_self, 'b head c (h w) -> b c head (h w)',
                             head=self.num_heads, h=h, w=w)
        out_self = self.norm_self(out_self)

        out_self = rearrange(out_self, 'b c head (h w) -> b (head c) h w',
                             head=self.num_heads, h=h, w=w)
        out_self = out_self * v

        # cross-attention in frequency
        q__fft = torch.fft.rfft2(q_.float(), norm='ortho')
        k_dino_fft = torch.fft.rfft2(k_dino.float(), norm='ortho')

        cross_attn_fft = q__fft * k_dino_fft
        cross_attn = torch.fft.irfft2(cross_attn_fft, s=(self.patch_size, self.patch_size), norm='ortho')
        out_cross = rearrange(cross_attn, 'b (head c) h w patch1 patch2 -> b head c (h patch1 w patch2)',
                              head=self.num_heads, patch1=self.patch_size, patch2=self.patch_size)
        out_cross = out_cross * self.temperature_cross

        out_cross = rearrange(out_cross, 'b head c (h w) -> b c head (h w)',
                              head=self.num_heads, h=h, w=w)
        out_cross = self.norm_cross(out_cross)

        out_cross = rearrange(out_cross, 'b c head (h w) -> b (head c) h w',
                              head=self.num_heads, h=h, w=w)
        out_cross = out_cross * v_dino

        alpha = torch.sigmoid(self.alpha)
        out = out_self + out_cross * alpha

        out = self.project_out(out)
        return out


####################################################################################################################################################
## -- OmniLight : Visual Prior Guided Mixture-of-Experts -- ##
def dwt_init(x):
    # x: (B, C, H, W)
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)


def idwt_init(x_LL, x_High):
    C = x_LL.shape[1]
    x_HL = x_High[:, 0:C, :, :]
    x_LH = x_High[:, C:2 * C, :, :]
    x_HH = x_High[:, 2 * C:3 * C, :, :]

    x1 = (x_LL - x_HL - x_LH + x_HH) / 2
    x2 = (x_LL - x_HL + x_LH - x_HH) / 2
    x3 = (x_LL + x_HL - x_LH - x_HH) / 2
    x4 = (x_LL + x_HL + x_LH + x_HH) / 2

    B, C, H_half, W_half = x_LL.shape
    output = torch.zeros((B, C, H_half * 2, W_half * 2), device=x_LL.device)

    output[:, :, 0::2, 0::2] = x1
    output[:, :, 1::2, 0::2] = x2
    output[:, :, 0::2, 1::2] = x3
    output[:, :, 1::2, 1::2] = x4

    return output


class DepthWiseConv(nn.Module):
    """
    kernel-size controllable Depth-wise Conv
    """

    def __init__(self, dim, kernel_size, bias):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride=1, padding=kernel_size // 2, groups=dim, bias=bias)

    def forward(self, x):
        return self.conv(x)


## --- LL Expert Block (Restormer Style) --- ##
class Expert_LL(nn.Module):
    def __init__(self, dim, internal_dim, num_heads, kernel_size):
        super().__init__()
        self.internal_dim_in = nn.Conv2d(dim, internal_dim, 1, bias=False)
        self.internal_dim_out = nn.Conv2d(internal_dim, dim, 1, bias=False)

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(internal_dim, internal_dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = DepthWiseConv(internal_dim * 3, kernel_size=kernel_size, bias=False)
        self.project_out = nn.Conv2d(internal_dim, internal_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.internal_dim_in(x)
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.internal_dim_out(out)
        return out


## --- High (LH,HL,HH) Expert Block (CNN/NAFNet Style) --- ##
class Expert_High(nn.Module):
    def __init__(self, dim, internal_dim, kernel_size):
        super().__init__()
        self.internal_dim_in = nn.Conv2d(dim, internal_dim, 1, groups=1, bias=False)
        self.internal_dim_out = nn.Conv2d(internal_dim, dim, 1, groups=1, bias=False)

        self.conv1 = nn.Conv2d(internal_dim, internal_dim * 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dwconv = DepthWiseConv(internal_dim * 2, kernel_size=kernel_size, bias=True)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(internal_dim, internal_dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=internal_dim, out_channels=internal_dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        )

    def forward(self, x):
        x = self.internal_dim_in(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        out = self.conv2(x) * self.sca(x)
        out = self.internal_dim_out(out)
        return out


## --- Routing Algorithm + Wavelet Domain Mixture-of-Experts Module & Block --- ##
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3), requires_grad=True,
                            device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def to_spatial(self, x, x_shape):
        h, w = x_shape
        amp, phase = x.chunk(2, dim=1)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        x = real + 1j * imag
        x = torch.fft.ifft2(x, s=(h, w), norm="backward").real
        return x

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class RoutingFunction(nn.Module):
    def __init__(self, dim, num_experts, k, complexity, complexity_bias=True):
        super().__init__()

        self.encoder_avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            Rearrange('b c 1 1 -> b c'),  # (B, C)
        )
        self.encoder_maxpool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # (B, C, 1, 1)
            Rearrange('b c 1 1 -> b c'),  # (B, C)
        )
        self.dino_avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            Rearrange('b c 1 1 -> b c'),  # (B, C)
        )
        self.dino_maxpool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # (B, C, 1, 1)
            Rearrange('b c 1 1 -> b c'),  # (B, C)
        )
        self.gate = nn.Linear(dim * 4, num_experts, bias=False)  # (B, Num_Experts)

        if complexity_bias:
            # ex) weights for 3x3=1, 5x5=1.5, 7x7=2
            complexity = complexity / complexity.max()  # Normalize
        else:
            complexity = torch.ones(num_experts)

        self.register_buffer('complexity', complexity)

        self.k = k  # Top-k selection
        self.tau = 1.0  # Temperature for Importance
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0  # Noise Standard Deviation
        self.use_complexity_bias = complexity_bias

    def forward(self, feat_encoder, feat_dino):
        encoder_avg = self.encoder_avgpool(feat_encoder)
        encoder_max = self.encoder_maxpool(feat_encoder)
        dino_avg = self.dino_avgpool(feat_dino)
        dino_max = self.dino_maxpool(feat_dino)
        global_vec = torch.cat([encoder_avg, encoder_max, dino_avg, dino_max], dim=1)
        logits = self.gate(global_vec)  # (B, num_experts)

        loss_imp = 0
        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))

        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        gating_scores = noisy_logits.float().softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0

        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values.to(logits.dtype))

        return gates, top_k_indices, top_k_values, aux_loss

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)

        if self.use_complexity_bias:
            importance = importance * (self.complexity * self.tau)

        imp_mean = importance.mean()
        imp_std = importance.std()
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2
        return loss_imp

    def load_loss(self, logits, logits_noisy, noise_std):
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]

        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )

        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std

        normal_dist = Normal(0, 1)
        # device matching
        if logits.is_cuda:
            pass

        p = 1. - normal_dist.cdf(noise_required_to_win)

        p_mean = p.mean(dim=0)

        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2

        return loss_load


class MoEModule(nn.Module):
    def __init__(self, dim, expert_class, expert_args_list, k, complexity, router_dim=None):
        super().__init__()
        self.num_experts = len(expert_args_list)

        actual_router_dim = router_dim if router_dim is not None else dim

        # For advanced routing with complexity bias, set use_complexity_bias=True
        self.router = RoutingFunction(actual_router_dim, self.num_experts, k=k, complexity=complexity, complexity_bias=True)

        self.experts = nn.ModuleList([
            expert_class(dim, **args) for args in expert_args_list
        ])

    def forward(self, x, feat_encoder, feat_dino):
        # gates: (B, num_experts)
        # indices: (B, k)
        # aux_loss: Scalar
        gates, indices, values, aux_loss = self.router(feat_encoder, feat_dino)

        if x.size(0) == 1 and self.router.k == 1:
            idx = indices.item()
            expert_out = self.experts[idx](x)
            gate_val = values[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).to(x.dtype)
            out = expert_out * gate_val
            return out, aux_loss

        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_outputs = [self.experts[exp](expert_inputs[exp]) for exp in range(len(self.experts))]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)

        else:
            selected_experts = [self.experts[i] for i in indices.squeeze(0)]
            expert_outputs = torch.stack([expert(x) for expert in selected_experts], dim=1)
            gates = gates.gather(1, indices)
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs
            out = weighted_outputs.sum(dim=1)

        return out, aux_loss


class SFTLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sft_net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels, channels * 2, 1)
        )
        nn.init.constant_(self.sft_net[-1].weight, 0)
        nn.init.constant_(self.sft_net[-1].bias, 0)

    def forward(self, condition, main):
        params = self.sft_net(condition)
        scale, shift = params.chunk(2, dim=1)
        return main * (1 + scale) + shift


class WDMoEModule(nn.Module):
    def __init__(self, dim, num_heads, k=1, expert_type='freq'):
        super().__init__()

        # 1. Visual Prior Expert
        if expert_type == 'freq':
            self.visual_prior_expert = Attention_Freq_Fuse(dim, num_heads)
        else:
            self.visual_prior_expert = Attention_Spat_Fuse(dim, num_heads)

        # 2. LL MoE (3 Experts: channel)
        self.moe_LL = MoEModule(
            dim=dim,
            router_dim=dim,
            expert_class=Expert_LL,
            expert_args_list=[
                {'internal_dim': dim // 4, 'num_heads': num_heads, 'kernel_size': 3},
                {'internal_dim': dim // 2, 'num_heads': num_heads, 'kernel_size': 3},
                {'internal_dim': dim, 'num_heads': num_heads, 'kernel_size': 3}
            ],
            k=k,
            complexity=torch.tensor([0.25, 0.5, 1.0])
        )

        # 3. High MoE (3 Experts: k=3, 5, 7)
        self.moe_High = MoEModule(
            dim=dim * 3,
            router_dim=dim,
            expert_class=Expert_High,
            expert_args_list=[
                {'internal_dim': dim // 2, 'kernel_size': 3},
                {'internal_dim': dim // 2, 'kernel_size': 5},
                {'internal_dim': dim // 2, 'kernel_size': 7}
            ],
            k=k,
            complexity=torch.tensor([9.0, 25.0, 49.0])
        )

        # 4. SFT Fusion
        self.norm = LayerNorm2d(dim)
        self.sft = SFTLayer(dim)

    def forward(self, x, feat_encoder, feat_dino):
        # Visual Prior Expert
        x_vp = self.visual_prior_expert(x, feat_dino)

        # DWT
        x_LL, x_High = dwt_init(x)

        # MoE Routing
        out_LL, loss_LL = self.moe_LL(x_LL, feat_encoder, feat_dino)
        out_High, loss_High = self.moe_High(x_High, feat_encoder, feat_dino)

        # IDWT
        x_dwt_branch = idwt_init(out_LL, out_High)
        x_dwt_branch = self.norm(x_dwt_branch)

        # Fusion
        out_fused = self.sft(condition=x_dwt_branch, main=x_vp)  # DWT conditions DINO

        return out_fused, loss_LL + loss_High


class WDMoEBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, topk=1, block_idx=0):
        super().__init__()

        expert_type = 'freq' if block_idx % 2 == 0 else 'spat'

        self.norm1 = LayerNorm2d(dim)
        self.attn = WDMoEModule(dim, num_heads, k=topk, expert_type=expert_type)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDconvFeedForward(dim, ffn_expansion_factor, bias=False)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x, feat_encoder, feat_dino):
        temp = x
        x, aux_loss = self.attn(self.norm1(x), feat_encoder, feat_dino)
        x = temp + self.beta * x
        x = x + self.gamma * self.ffn(self.norm2(x))

        return x, aux_loss


## -- DINOLight Block -- ##
class SFDINOBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, block_idx=0):
        super().__init__()

        expert_type = 'freq' if block_idx % 2 == 0 else 'spat'

        self.norm1 = LayerNorm2d(dim)

        if expert_type == 'freq':
            self.attn = Attention_Freq_Fuse(dim, num_heads)
        else:
            self.attn = Attention_Spat_Fuse(dim, num_heads)

        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDconvFeedForward(dim, ffn_expansion_factor, bias=False)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x, dino):
        x = x + self.beta * self.attn(self.norm1(x), dino)
        x = x + self.gamma * self.ffn(self.norm2(x))

        return x


## Encoder and Refinement Block ##
## Multi-DConv Head Transposed Self-Attention (Channel Domain)
class Attention_Restormer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


## Multi-DConv Head FFT-based Self-Attention (Frequency Domain)
class Attention_FFTformer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)

        self.norm_self1 = LayerNorm2d(dim // num_heads)
        self.norm_self2 = LayerNorm2d(dim // num_heads)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)

        self.patch_size = 8

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                      patch1=self.patch_size, patch2=self.patch_size)
        k = rearrange(k, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                      patch1=self.patch_size, patch2=self.patch_size)

        # self-attention in frequency
        q_fft = torch.fft.rfft2(q.float(), norm='ortho')
        k_fft = torch.fft.rfft2(k.float(), norm='ortho')

        attn_fft = q_fft * k_fft
        attn = torch.fft.irfft2(attn_fft, s=(self.patch_size, self.patch_size), norm='ortho')
        out = rearrange(attn, 'b (head c) h w patch1 patch2 -> b head c (h patch1 w patch2)', head=self.num_heads,
                        patch1=self.patch_size, patch2=self.patch_size)
        out = out * self.temperature

        out = rearrange(out, 'b head c (h w) -> b c head (h w)', head=self.num_heads, h=h, w=w)
        out = self.norm_self1(out)

        out = rearrange(out, 'b c head (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out * v

        out = self.project_out(out)
        return out


## FFTformer and Restormer in sequence (Dual Domain)
class RefinementBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn1 = Attention_FFTformer(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.attn2 = Attention_Restormer(dim, num_heads)

        self.norm3 = LayerNorm2d(dim)
        self.ffn1 = GatedDconvFeedForward(dim, ffn_expansion_factor, bias=False)
        self.norm4 = LayerNorm2d(dim)
        self.ffn2 = GatedDconvFeedForward(dim, ffn_expansion_factor, bias=False)

        self.beta1 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.beta1 * self.attn1(self.norm1(x))
        x = x + self.gamma1 * self.ffn1(self.norm3(x))

        x = x + self.beta2 * self.attn2(self.norm2(x))
        x = x + self.gamma2 * self.ffn2(self.norm4(x))

        return x


####################################################################################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#####################
## -- OmniLight -- ##
class OmniLight(nn.Module):
    def __init__(self, channels=32, dim_dino=768, num_blocks=[4, 6, 6, 8, 6, 6, 4], refine_num_blocks=2, topk=1):
        super().__init__()

        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.ff_enc1 = FeatureFusionV2(dim_dino=dim_dino, dim=channels * 1)
        self.ff_enc2 = FeatureFusionV2(dim_dino=dim_dino, dim=channels * 2)
        self.ff_enc3 = FeatureFusionV2(dim_dino=dim_dino, dim=channels * 4)
        self.ff_mid = FeatureFusionV2(dim_dino=dim_dino, dim=channels * 8)
        self.ff_dec3 = FeatureFusionV2(dim_dino=dim_dino, dim=channels * 4)
        self.ff_dec2 = FeatureFusionV2(dim_dino=dim_dino, dim=channels * 2)
        self.ff_dec1 = FeatureFusionV2(dim_dino=dim_dino, dim=channels * 1)

        self.reduce_dim3 = nn.Conv2d(channels * 8, channels * 4, 1)
        self.reduce_dim2 = nn.Conv2d(channels * 8, channels * 2, 1)
        self.reduce_dim1 = nn.Conv2d(channels * 8, channels * 1, 1)

        self.Transformer_enc1 = nn.ModuleList(
            [SFDINOBlock(dim=channels, num_heads=1, ffn_expansion_factor=2, block_idx=i) for i in range(num_blocks[0])])
        self.Transformer_enc2 = nn.ModuleList(
            [SFDINOBlock(dim=channels * 2, num_heads=2, ffn_expansion_factor=2, block_idx=i) for i in range(num_blocks[1])])
        self.Transformer_enc3 = nn.ModuleList(
            [SFDINOBlock(dim=channels * 4, num_heads=4, ffn_expansion_factor=2, block_idx=i) for i in range(num_blocks[2])])

        self.Transformer_mid = nn.ModuleList(
            [WDMoEBlock(dim=channels * 8, num_heads=8, ffn_expansion_factor=2, topk=topk, block_idx=i) for i in range(num_blocks[3])])
        self.Transformer_dec3 = nn.ModuleList(
            [WDMoEBlock(dim=channels * 4, num_heads=4, ffn_expansion_factor=2, topk=topk, block_idx=i) for i in range(num_blocks[4])])
        self.Transformer_dec2 = nn.ModuleList(
            [WDMoEBlock(dim=channels * 2, num_heads=2, ffn_expansion_factor=2, topk=topk, block_idx=i) for i in range(num_blocks[5])])
        self.Transformer_dec1 = nn.ModuleList(
            [WDMoEBlock(dim=channels, num_heads=1, ffn_expansion_factor=2, topk=topk, block_idx=i) for i in range(num_blocks[6])])

        self.Transformer_refinement = nn.ModuleList(
            [RefinementBlock(dim=channels, num_heads=1) for i in range(refine_num_blocks)])

        self.down1_2 = Downsample(channels)
        self.down2_3 = Downsample(channels * 2)
        self.down3_4 = Downsample(channels * 4)

        self.up4_3 = Upsample(channels * 8)
        self.channel_reduce3 = nn.Conv2d(channels * 8, channels * 4, kernel_size=1, bias=False)
        self.up3_2 = Upsample(channels * 4)
        self.channel_reduce2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False)
        self.up2_1 = Upsample(channels * 2)
        self.channel_reduce1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

        self.aux_loss = None
        self.num_moe_blocks = len(self.Transformer_mid) + len(self.Transformer_dec3) + len(self.Transformer_dec2) + len(self.Transformer_dec1)

    def forward(self, inp, dino):
        # shallow
        self.aux_loss = 0
        x = self.conv_in(inp)

        # --- Encoder --- #
        ff_enc1 = self.ff_enc1(dino)
        ff_enc1 = F.interpolate(ff_enc1, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_enc1:
            x = checkpoint.checkpoint(block, x, ff_enc1, use_reentrant=False)
        down1 = x
        x = self.down1_2(x)

        ff_enc2 = self.ff_enc2(dino)
        ff_enc2 = F.interpolate(ff_enc2, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_enc2:
            x = checkpoint.checkpoint(block, x, ff_enc2, use_reentrant=False)
        down2 = x
        x = self.down2_3(x)

        ff_enc3 = self.ff_enc3(dino)
        ff_enc3 = F.interpolate(ff_enc3, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_enc3:
            x = block(x, ff_enc3)
        down3 = x
        x = self.down3_4(x)

        feat_encoder = x

        # --- Bottleneck --- #
        ff_mid = self.ff_mid(dino)
        ff_mid = F.interpolate(ff_mid, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_mid:
            x, aux_loss = block(x, feat_encoder, ff_mid)
            self.aux_loss += aux_loss

        # --- Decoder --- #
        x = self.up4_3(x)
        x = torch.cat([x, down3], 1)
        x = self.channel_reduce3(x)

        feat_encoder_dec3 = self.reduce_dim3(feat_encoder)
        ff_dec3 = self.ff_dec3(dino)
        ff_dec3 = F.interpolate(ff_dec3, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_dec3:
            x, aux_loss = block(x, feat_encoder_dec3, ff_dec3)
            self.aux_loss += aux_loss

        x = self.up3_2(x)
        x = torch.cat([x, down2], 1)
        x = self.channel_reduce2(x)

        feat_encoder_dec2 = self.reduce_dim2(feat_encoder)
        ff_dec2 = self.ff_dec2(dino)
        ff_dec2 = F.interpolate(ff_dec2, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_dec2:
            x, aux_loss = block(x, feat_encoder_dec2, ff_dec2)
            self.aux_loss += aux_loss

        x = self.up2_1(x)
        x = torch.cat([x, down1], 1)
        x = self.channel_reduce1(x)

        feat_encoder_dec1 = self.reduce_dim1(feat_encoder)
        ff_dec1 = self.ff_dec1(dino)
        ff_dec1 = F.interpolate(ff_dec1, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_dec1:
            x, aux_loss = block(x, feat_encoder_dec1, ff_dec1)
            self.aux_loss += aux_loss

        # --- Refinement --- #
        for block in self.Transformer_refinement:
            x = checkpoint.checkpoint(block, x, use_reentrant=False)

        # long skip connection
        # x = x + temp
        x = self.conv_out(x) + inp
        self.aux_loss /= self.num_moe_blocks
        return x, self.aux_loss
