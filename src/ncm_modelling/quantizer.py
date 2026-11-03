from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, size: int, dim: int):
        super().__init__()
        self.size, self.dim = size, dim
        self.emb = nn.Embedding(size, dim)
        nn.init.uniform_(self.emb.weight, -1.0 / size, 1.0 / size)
        self.register_buffer("usage", torch.zeros(size))                 # EMA of per-code counts
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))
        self._ready = False

    @torch.no_grad()
    def nearest(self, x: torch.Tensor) -> torch.Tensor:                  # x: (N, d) fp32
        e = self.emb.weight.float()
        d2 = x.pow(2).sum(1, keepdim=True) - 2.0 * x @ e.t() + e.pow(2).sum(1)[None]
        return d2.argmin(1)

    @torch.no_grad()
    def maybe_init(self, x: torch.Tensor):
        """Data-dependent init from the first batch's residuals."""
        if self._ready:
            return
        if bool(self.initted):
            self._ready = True
            return
        n = x.shape[0]
        pick = (torch.randint(0, n, (self.size,), device=x.device) if n < self.size
                else torch.randperm(n, device=x.device)[: self.size])
        self.emb.weight.copy_(x[pick] + 0.01 * torch.randn(self.size, self.dim, device=x.device))
        self.initted.fill_(True)
        self._ready = True

    @torch.no_grad()
    def track(self, idx: torch.Tensor):
        cnt = torch.bincount(idx, minlength=self.size).float()
        self.usage.mul_(0.99).add_(0.01 * cnt)

    @torch.no_grad()
    def reinit_dead(self, x: torch.Tensor, ratio: float) -> int:
        m = self.usage.mean()
        if float(m) <= 0 or x.shape[0] == 0:
            return 0
        dead = self.usage < ratio * m
        n = int(dead.sum())
        if n == 0:
            return 0
        pick = torch.randint(0, x.shape[0], (n,), device=x.device)
        self.emb.weight[dead] = x[pick] + 0.01 * torch.randn(n, self.dim, device=x.device)
        self.usage[dead] = m
        return n


def codebook_sizes(scales, base: int, adaptive: bool):
    if not adaptive:
        return [base] * len(scales)
    return [min(base, max(128, 128 * s)) for s in scales]               # coarse levels -> small vocab


class MultiScaleResidualVQ(nn.Module):
    """
    Level k quantizes the residual (f - f_hat_{k-1}) pooled to an s_k x s_k grid,
    and adds phi_k(upsample(code vectors)) to the cumulative latent reconstruction.
    Mirrors NCM Eq. (7)-(8) with 2D grids (VAR-style next-scale tokenization).
    """

    def __init__(self, dim, scales, base_size, latent_size, adaptive=True, beta=0.25):
        super().__init__()
        self.d, self.scales, self.L, self.beta = dim, tuple(scales), latent_size, beta
        self.sizes = codebook_sizes(self.scales, base_size, adaptive)
        self.books = nn.ModuleList(Codebook(v, dim) for v in self.sizes)
        self.phi = nn.ModuleList(nn.Conv2d(dim, dim, 3, padding=1) for _ in self.scales)
        sq = [s * s for s in self.scales]
        self.offsets = [0]
        for q in sq:
            self.offsets.append(self.offsets[-1] + q)
        self.num_codes = self.offsets[-1]
        self.K = len(self.scales)
        self._last = {}

    # ---- helpers (static shapes -> ONNX/TensorRT friendly)
    def _down(self, x, s):
        return x if s == self.L else F.adaptive_avg_pool2d(x, (s, s))

    def _up(self, x, s):
        return x if s == self.L else F.interpolate(x, size=(self.L, self.L), mode="bilinear",
                                                   align_corners=False)

    # ---- training / encoding path
    def forward(self, f: torch.Tensor):
        with torch.autocast(f.device.type, enabled=False):
            f = f.float()
            B, d = f.shape[0], f.shape[1]
            f_hat = torch.zeros_like(f)
            f_hats, codes, commit, cb = [], [], 0.0, 0.0
            for k, s in enumerate(self.scales):
                u = self._down(f - f_hat.detach(), s)                       # (B,d,s,s)
                flat = u.permute(0, 2, 3, 1).reshape(-1, d)
                book = self.books[k]
                if self.training:
                    book.maybe_init(flat.detach())
                idx = book.nearest(flat.detach())
                e = book.emb(idx).view(B, s, s, d).permute(0, 3, 1, 2)
                if self.training:
                    book.track(idx)
                    self._last[k] = flat.detach()
                commit = commit + F.mse_loss(u, e.detach())
                cb = cb + F.mse_loss(e, u.detach())
                e_st = u + (e - u).detach()                                  # straight-through
                f_hat = f_hat + self.phi[k](self._up(e_st, s))
                f_hats.append(f_hat)
                codes.append(idx.view(B, s, s))
            vq_loss = (self.beta * commit + cb) / self.K
        return f_hats, codes, vq_loss

    @torch.no_grad()
    def encode_flat(self, f):
        _, codes, _ = self.forward(f)
        return torch.cat([c.flatten(1) for c in codes], 1).int()            # (B, num_codes)

    def reinit_dead(self, ratio):
        return [b.reinit_dead(self._last[k], ratio) if k in self._last else 0
                for k, b in enumerate(self.books)]

    # ---- decode / generator-side path (what the Transformer stage + TensorRT use)
    def lookup(self, k, idx):                                                # idx: (B, s*s)
        s = self.scales[k]
        return self.books[k].emb(idx).transpose(1, 2).reshape(-1, self.d, s, s)

    def add_level(self, f_hat, k, idx):
        """f_hat_k = f_hat_{k-1} + phi_k(up(e_k)).  idx: (B, s_k^2) integer codes."""
        s = self.scales[k]
        return f_hat + self.phi[k](self._up(self.lookup(k, idx.long()), s))

    def decode_latent(self, codes, upto=None):                               # codes: (B, num_codes)
        upto = upto or self.K
        f_hat = None
        for k in range(upto):
            idx = codes[:, self.offsets[k]:self.offsets[k + 1]]
            step = self.phi[k](self._up(self.lookup(k, idx.long()), self.scales[k]))
            f_hat = step if f_hat is None else f_hat + step
        return f_hat

    def level_input(self, f_hat, k):
        """Resize the cumulative reconstruction to level k's grid (Transformer input, NCM Eq. 4).
        Bilinear on purpose: exports cleanly to TensorRT. Use the same op in training and inference."""
        s = self.scales[k]
        return f_hat if s == f_hat.shape[-1] else F.interpolate(
            f_hat, size=(s, s), mode="bilinear", align_corners=False)