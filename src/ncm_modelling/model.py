from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .quantizer import MultiScaleResidualVQ


def _gn(c):
    return nn.GroupNorm(32, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = _gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def _channels(base, n_down):
    return [base * min(2 ** i, 4) for i in range(n_down + 1)]


class Encoder(nn.Module):
    def __init__(self, base, n_down, latent_dim):
        super().__init__()
        ch = _channels(base, n_down)
        layers = [nn.Conv2d(3, ch[0], 3, padding=1)]
        for i in range(n_down):
            layers += [ResBlock(ch[i], ch[i]), Down(ch[i], ch[i + 1])]
        layers += [ResBlock(ch[-1], ch[-1]), ResBlock(ch[-1], ch[-1]),
                   _gn(ch[-1]), nn.SiLU(), nn.Conv2d(ch[-1], latent_dim, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, base, n_down, latent_dim):
        super().__init__()
        rev = _channels(base, n_down)[::-1]
        layers = [nn.Conv2d(latent_dim, rev[0], 3, padding=1),
                  ResBlock(rev[0], rev[0]), ResBlock(rev[0], rev[0])]
        for i in range(n_down):
            layers += [Up(rev[i], rev[i + 1]), ResBlock(rev[i + 1], rev[i + 1])]
        layers += [_gn(rev[-1]), nn.SiLU(), nn.Conv2d(rev[-1], 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class NCMImageTokenizer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        n_down = int(math.log2(cfg.downsample))
        self.enc = Encoder(cfg.base_ch, n_down, cfg.latent_dim)
        self.quant = MultiScaleResidualVQ(cfg.latent_dim, cfg.scales, cfg.codebook_size,
                                          cfg.crop // cfg.downsample, cfg.adaptive_codebooks, cfg.beta)
        self.dec = Decoder(cfg.base_ch, n_down, cfg.latent_dim)
        self.crop = cfg.crop

    def forward(self, x, cuts):
        """cuts: 1-based level counts to decode, e.g. [K, 3]. Returns {cut: recon}, codes, vq_loss."""
        f = self.enc(x)
        f_hats, codes, vq = self.quant(f)
        return {k: self.dec(f_hats[k - 1]) for k in cuts}, codes, vq

    @torch.no_grad()
    def encode(self, x):                                      # (B,3,H,W) in [-1,1] -> (B, num_codes) int32
        return self.quant.encode_flat(self.enc(x))

    def decode(self, codes, upto=None):                       # (B, num_codes) -> (B,3,H,W) in [-1,1]
        return self.dec(self.quant.decode_latent(codes, upto)).clamp(-1.0, 1.0)

    def random_codes(self, batch=1, device="cpu"):
        parts = [torch.randint(0, b.size, (batch, s * s), device=device)
                 for b, s in zip(self.quant.books, self.quant.scales)]
        return torch.cat(parts, 1).int()