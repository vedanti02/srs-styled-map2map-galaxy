"""3D UNet velocity network for 64^3 patches with periodic (circular) convolutions.

Design (DDPM/EDM-style, sized for ~12k training patches):
  * every conv is kernel-3 with circular padding -> exactly shift-equivariant on the torus
    (matches the deployed GAN, which also used periodic padding on the 64^3 core)
  * time t in [0,1] -> sinusoidal embedding -> MLP -> per-block scale/shift (FiLM on GroupNorm)
  * levels 64 -> 32 -> 16 -> 8 with channels base*mults; 2 res blocks per level
  * one self-attention block at the 8^3 bottleneck (512 tokens) for global context, which
    matters for count conservation
  * input channels = x_t (+ LF conditioning channel), output = 1 velocity channel
  * final conv zero-initialised so the initial velocity is ~0
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cpad(x, p=1):
    return F.pad(x, (p,) * 6, mode="circular")


class PConv3(nn.Module):
    """3x3x3 conv with circular padding (stride 1 or 2). Output size = N / stride for even N."""
    def __init__(self, cin, cout, stride=1, zero_init=False):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, stride=stride, padding=0)
        if zero_init:
            nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(cpad(x, 1))


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """t in [0,1] (B,) -> (B, dim). Scaled by 1000 like DDPM timesteps."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def gn(ch, groups=8):
    return nn.GroupNorm(min(groups, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_ch, dropout=0.0):
        super().__init__()
        self.norm1 = gn(cin)
        self.conv1 = PConv3(cin, cout)
        self.emb = nn.Linear(emb_ch, 2 * cout)
        self.norm2 = gn(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = PConv3(cout, cout, zero_init=True)
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.norm = gn(ch)
        self.qkv = nn.Conv3d(ch, 3 * ch, 1)
        self.proj = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, D * H * W).unbind(1)
        q, k, v = (a.transpose(-1, -2) for a in (q, k, v))          # (B, heads, N, d)
        h = F.scaled_dot_product_attention(q, k, v)                   # (B, heads, N, d)
        h = h.transpose(-1, -2).reshape(B, C, D, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = PConv3(ch, ch, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = PConv3(ch, ch)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet3D(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32, mults=(1, 2, 4, 8), num_res=2,
                 attn_levels=(3,), dropout=0.0, heads=4):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.base = base
        emb_ch = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))
        self.stem = PConv3(in_ch, base)

        chs = [base * m for m in mults]
        self.down = nn.ModuleList()
        skip_chs = [base]
        ch = base
        for lvl, cout in enumerate(chs):
            for _ in range(num_res):
                blk = nn.ModuleList([ResBlock(ch, cout, emb_ch, dropout)])
                if lvl in attn_levels:
                    blk.append(Attention(cout, heads))
                self.down.append(blk)
                ch = cout
                skip_chs.append(ch)
            if lvl < len(chs) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_ch, dropout), Attention(ch, heads),
                                  ResBlock(ch, ch, emb_ch, dropout)])

        self.up = nn.ModuleList()
        for lvl, cout in reversed(list(enumerate(chs))):
            for i in range(num_res + 1):
                blk = nn.ModuleList([ResBlock(ch + skip_chs.pop(), cout, emb_ch, dropout)])
                if lvl in attn_levels:
                    blk.append(Attention(cout, heads))
                ch = cout
                if i == num_res and lvl > 0:
                    blk.append(Upsample(ch))
                self.up.append(blk)
        assert not skip_chs

        self.out_norm = gn(ch)
        self.out_conv = PConv3(ch, out_ch, zero_init=True)

    def forward(self, x, t, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        assert x.shape[1] == self.in_ch, (x.shape, self.in_ch)
        emb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.stem(x)
        hs = [h]
        for blk in self.down:
            for m in blk:
                h = m(h, emb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        for blk in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for m in blk:
                h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        assert not hs
        return self.out_conv(F.silu(self.out_norm(h)))


def build_unet(args_or_dict, in_ch):
    a = args_or_dict if isinstance(args_or_dict, dict) else vars(args_or_dict)
    mults = tuple(int(m) for m in str(a["ch_mult"]).split(","))
    attn = tuple(int(l) for l in str(a["attn_levels"]).split(",") if l != "")
    return UNet3D(in_ch=in_ch, out_ch=1, base=int(a["base_ch"]), mults=mults, num_res=int(a["num_res"]),
                  attn_levels=attn, dropout=float(a["dropout"]))
