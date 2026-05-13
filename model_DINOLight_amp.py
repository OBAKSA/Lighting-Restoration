import torch
import torch.nn as nn
from torch.nn import functional as F
import numbers
import numpy as np
from einops import rearrange
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.utils.checkpoint as checkpoint

device = torch.device('cuda')


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, eps=eps, elementwise_affine=True, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x_ = x.float()
            x_ = x_.permute(0, 2, 3, 1)  # (N, H, W, C)
            x_ = self.layer_norm(x_)  # C only
            x_ = x_.permute(0, 3, 1, 2)  # (N, C, H, W)로 복원
        return x_.to(orig_dtype)


## Resizing modules ##
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

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
            nn.SiLU(),
            nn.Conv2d(in_channels=dim_dino, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.predictor_middle = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=dim_dino, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.predictor_final = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=dim_dino, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv_dino = nn.Sequential(
            nn.SiLU(),
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


##########################################################################
## FFN
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2):
        super(FeedForward, self).__init__()

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


##########################################################################
## Multi-DConv Head Transposed Self-Attention + auxiliary Self-Attention (Spatial Domain)
class Attention_Spat(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention_Spat, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v, q_, k_, v_ = qkv.chunk(6, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_ = rearrange(q_, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_ = rearrange(k_, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_ = rearrange(v_, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature1
        attn = attn.softmax(dim=-1)

        out1 = (attn @ v)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_ = torch.nn.functional.normalize(q_, dim=-1)
        k_ = torch.nn.functional.normalize(k_, dim=-1)

        attn_ = (q_ @ k_.transpose(-2, -1)) * self.temperature2
        attn_ = attn_.softmax(dim=-1)

        out2 = (attn_ @ v_)

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        alpha = torch.sigmoid(self.alpha)
        out = out1 + out2 * alpha

        out = self.project_out(out)
        return out


## Multi-DConv Head Transposed Self-Attention + auxiliary Self-Attention (Frequency Domain)
class Attention_Freq(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention_Freq, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=False)

        self.norm_self1 = LayerNorm2d(dim // num_heads)
        self.norm_self2 = LayerNorm2d(dim // num_heads)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)

        self.patch_size = 8

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v, q_, k_, v_ = qkv.chunk(6, dim=1)

        q = rearrange(q, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                      patch1=self.patch_size, patch2=self.patch_size)
        k = rearrange(k, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                      patch1=self.patch_size, patch2=self.patch_size)

        q_ = rearrange(q_, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                       patch1=self.patch_size, patch2=self.patch_size)
        k_ = rearrange(k_, 'b (head c) (h patch1) (w patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                       patch1=self.patch_size, patch2=self.patch_size)

        # self-attention in frequency
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())

        attn_fft1 = q_fft * k_fft
        # attn1 = torch.fft.irfft2(attn_fft1.float(), s=(self.patch_size, self.patch_size))
        attn1 = torch.fft.irfft2(attn_fft1, s=(self.patch_size, self.patch_size))
        out1 = rearrange(attn1, 'b (head c) h w patch1 patch2 -> b head c (h patch1 w patch2)', head=self.num_heads,
                         patch1=self.patch_size, patch2=self.patch_size)
        out1 = out1 * self.temperature1

        out1 = rearrange(out1, 'b head c (h w) -> b c head (h w)', head=self.num_heads, h=h, w=w)
        out1 = self.norm_self1(out1)

        out1 = rearrange(out1, 'b c head (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out1 = out1 * v

        # self-attention in frequency
        q__fft = torch.fft.rfft2(q_.float())
        k__fft = torch.fft.rfft2(k_.float())

        attn_fft2 = q__fft * k__fft
        # attn2 = torch.fft.irfft2(attn_fft2.float(), s=(self.patch_size, self.patch_size))
        attn2 = torch.fft.irfft2(attn_fft2, s=(self.patch_size, self.patch_size))
        out2 = rearrange(attn2, 'b (head c) h w patch1 patch2 -> b head c (h patch1 w patch2)', head=self.num_heads,
                         patch1=self.patch_size, patch2=self.patch_size)
        out2 = out2 * self.temperature2

        out2 = rearrange(out2, 'b head c (h w) -> b c head (h w)', head=self.num_heads, h=h, w=w)
        out2 = self.norm_self2(out2)

        out2 = rearrange(out2, 'b c head (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = out2 * v_

        alpha = torch.sigmoid(self.alpha)
        out = out1 + out2 * alpha

        out = self.project_out(out)
        return out


## Multi-DConv Head Transposed Self-Attention + auxiliary Cross-Attention w/ DINOv2 (Spatial domain)
class Attention_Spat_Fuse(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention_Spat_Fuse, self).__init__()
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


## Multi-DConv Head Transposed Self-Attention + auxiliary Cross-Attention w/ DINOv2 (Frequency Domain)
class Attention_Freq_Fuse(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention_Freq_Fuse, self).__init__()
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
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())

        self_attn_fft = q_fft * k_fft
        # self_attn = torch.fft.irfft2(self_attn_fft.float(), s=(self.patch_size, self.patch_size))
        self_attn = torch.fft.irfft2(self_attn_fft, s=(self.patch_size, self.patch_size))
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
        q__fft = torch.fft.rfft2(q_.float())
        k_dino_fft = torch.fft.rfft2(k_dino.float())

        cross_attn_fft = q__fft * k_dino_fft
        # cross_attn = torch.fft.irfft2(cross_attn_fft.float(), s=(self.patch_size, self.patch_size))
        cross_attn = torch.fft.irfft2(cross_attn_fft, s=(self.patch_size, self.patch_size))
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


##########################################################################
## Blocks with Self-Attention + auxiliary Self-Attention ##
class TransformerBlock_SF(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2):
        super(TransformerBlock_SF, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.attn1 = Attention_Freq(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.attn2 = Attention_Spat(dim, num_heads)

        self.norm3 = LayerNorm2d(dim)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor)
        self.norm4 = LayerNorm2d(dim)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor)

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


## Blocks with Self-Attention + auxiliary Cross-Attention w/ DINOv2 ##
class TransformerBlock_SF_Fuse(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2):
        super(TransformerBlock_SF_Fuse, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.attn1 = Attention_Freq_Fuse(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.attn2 = Attention_Spat_Fuse(dim, num_heads)

        self.norm3 = LayerNorm2d(dim)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor)
        self.norm4 = LayerNorm2d(dim)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor)

        self.beta1 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x, dino):
        x = x + self.beta1 * self.attn1(self.norm1(x), dino)
        x = x + self.gamma1 * self.ffn1(self.norm3(x))

        x = x + self.beta2 * self.attn2(self.norm2(x), dino)
        x = x + self.gamma2 * self.ffn2(self.norm4(x))

        return x


####################################################################################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#########################
## -- DINOLight_amp -- ##
class DINOLight_amp(nn.Module):
    def __init__(self, channels=32, dim_dino=768, num_blocks=[2, 2, 2, 4], refine_num_blocks=2):
        super(DINOLight_amp, self).__init__()

        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.ff_enc1 = FeatureFusion(dim_dino=dim_dino, dim=channels * 1)
        self.ff_enc2 = FeatureFusion(dim_dino=dim_dino, dim=channels * 2)
        self.ff_enc3 = FeatureFusion(dim_dino=dim_dino, dim=channels * 4)
        self.ff_mid = FeatureFusion(dim_dino=dim_dino, dim=channels * 8)
        self.ff_dec3 = FeatureFusion(dim_dino=dim_dino, dim=channels * 4)
        self.ff_dec2 = FeatureFusion(dim_dino=dim_dino, dim=channels * 2)
        self.ff_dec1 = FeatureFusion(dim_dino=dim_dino, dim=channels * 1)

        self.Transformer_enc1 = nn.ModuleList(
            [TransformerBlock_SF_Fuse(dim=channels, num_heads=1) for i in range(num_blocks[0])])
        self.Transformer_enc2 = nn.ModuleList(
            [TransformerBlock_SF_Fuse(dim=channels * 2, num_heads=2) for i in range(num_blocks[1])])
        self.Transformer_enc3 = nn.ModuleList(
            [TransformerBlock_SF_Fuse(dim=channels * 4, num_heads=4) for i in range(num_blocks[2])])
        self.Transformer_mid = nn.ModuleList(
            [TransformerBlock_SF_Fuse(dim=channels * 8, num_heads=8) for i in range(num_blocks[3])])
        self.Transformer_dec3 = nn.ModuleList(
            [TransformerBlock_SF_Fuse(dim=channels * 4, num_heads=4) for i in range(num_blocks[2])])
        self.Transformer_dec2 = nn.ModuleList(
            [TransformerBlock_SF_Fuse(dim=channels * 2, num_heads=2) for i in range(num_blocks[1])])
        self.Transformer_dec1 = nn.ModuleList(
            [TransformerBlock_SF_Fuse(dim=channels, num_heads=1) for i in range(num_blocks[0])])

        self.Transformer_refinement = nn.ModuleList(
            [TransformerBlock_SF(dim=channels, num_heads=1) for i in range(refine_num_blocks)])

        self.down1_2 = Downsample(channels)
        self.down2_3 = Downsample(channels * 2)
        self.down3_4 = Downsample(channels * 4)

        self.up4_3 = Upsample(channels * 8)
        self.channel_reduce3 = nn.Conv2d(channels * 8, channels * 4, kernel_size=1, bias=False)
        self.up3_2 = Upsample(channels * 4)
        self.channel_reduce2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False)
        self.up2_1 = Upsample(channels * 2)
        self.channel_reduce1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, inp, dino):
        # network
        x = self.conv_in(inp)

        # CS-Attn + DINO
        ff_enc1 = self.ff_enc1(dino)
        ff_enc1 = F.interpolate(ff_enc1, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_enc1:
            x = block(x, ff_enc1)
        down1 = x
        x = self.down1_2(x)

        ff_enc2 = self.ff_enc2(dino)
        ff_enc2 = F.interpolate(ff_enc2, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_enc2:
            x = block(x, ff_enc2)
        down2 = x
        x = self.down2_3(x)

        ff_enc3 = self.ff_enc3(dino)
        ff_enc3 = F.interpolate(ff_enc3, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_enc3:
            x = block(x, ff_enc3)
        down3 = x
        x = self.down3_4(x)

        ff_mid = self.ff_mid(dino)
        ff_mid = F.interpolate(ff_mid, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_mid:
            x = block(x, ff_mid)

        x = self.up4_3(x)
        x = torch.cat([x, down3], 1)
        x = self.channel_reduce3(x)

        ff_dec3 = self.ff_dec3(dino)
        ff_dec3 = F.interpolate(ff_dec3, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_dec3:
            x = block(x, ff_dec3)

        x = self.up3_2(x)
        x = torch.cat([x, down2], 1)
        x = self.channel_reduce2(x)

        ff_dec2 = self.ff_dec2(dino)
        ff_dec2 = F.interpolate(ff_dec2, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_dec2:
            x = block(x, ff_dec2)

        x = self.up2_1(x)
        x = torch.cat([x, down1], 1)
        x = self.channel_reduce1(x)

        ff_dec1 = self.ff_dec1(dino)
        ff_dec1 = F.interpolate(ff_dec1, (x.shape[2], x.shape[3]), mode='bilinear')
        for block in self.Transformer_dec1:
            x = block(x, ff_dec1)

        for block in self.Transformer_refinement:
            x = block(x)

        # long skip connection
        # x = x + temp
        x = self.conv_out(x) + inp
        return x
