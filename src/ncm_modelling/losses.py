from __future__ import annotations
import torch
import torch.nn as nn


def weighted_l1(pred, tgt, w):
    """pred/tgt (B,3,H,W), w (B,1,H,W) >= 1 : object pixels (esp. small ones) count more."""
    return ((pred - tgt).abs() * w).sum() / (w.sum() * pred.shape[1])


def psnr_per_sample(pred, tgt, mask=None):
    """Inputs in [-1,1]; PSNR in dB w.r.t. a [0,1] range. mask (B,1,H,W) restricts to a region."""
    se = ((pred - tgt) / 2).pow(2)
    if mask is None:
        mse = se.mean((1, 2, 3))
    else:
        mse = (se * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * pred.shape[1]).clamp_min(1.0)
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


class Perceptual(nn.Module):
    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled, self.net = False, None
        if not enabled:
            return
        try:
            import lpips
            self.net = lpips.LPIPS(net="alex", verbose=False).eval().requires_grad_(False)
            self.enabled = True
        except Exception as e:                                  # missing package / no weights download
            print(f"[warn] LPIPS disabled: {e}")

    def forward(self, pred, tgt):
        with torch.autocast(pred.device.type, enabled=False):
            return self.net(pred.float(), tgt.float()).mean()