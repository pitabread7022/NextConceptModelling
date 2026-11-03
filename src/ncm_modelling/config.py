from __future__ import annotations
import argparse
import math
from dataclasses import dataclass, fields


@dataclass
class Config:
    # ---- data ----
    data_root: str = "data/dota8"
    out_dir: str = "runs/ncm_img"
    crop: int = 256
    batch_size: int = 16
    virtual_epoch: int = 512          # DOTA8 is tiny -> sample random crops
    num_workers: int = 0              # 0 is safest on Windows
    obj_crop_prob: float = 0.5        # prob. of a class-balanced, object-centred crop
    obj_weight: float = 4.0           # loss-weight boost for object pixels
    ref_obj_area: float = 1024.0      # objects smaller than this (px^2) get extra boost
    max_weight: float = 12.0
    # ---- model ----
    downsample: int = 8               # 256 -> 32x32 latent (keeps small objects alive)
    base_ch: int = 64
    latent_dim: int = 64
    scales: tuple = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)   # last == crop // downsample
    codebook_size: int = 1024
    adaptive_codebooks: bool = True   # smaller codebooks for coarse levels
    beta: float = 0.25
    # ---- optim ----
    steps: int = 6000
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup: int = 300
    grad_clip: float = 1.0
    amp: str = "bf16"                 # bf16 | fp16 | none
    seed: int = 42
    # ---- loss ----
    lambda_cut: float = 0.5           # weight of the random-truncation loss
    lambda_vq: float = 1.0
    lambda_lpips: float = 0.1         # 0 disables LPIPS
    # ---- codebook upkeep ----
    reinit_every: int = 100
    reinit_until: float = 0.7         # fraction of training during which dead codes are reset
    dead_ratio: float = 0.02
    # ---- logging ----
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000
    resume: str = ""


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    for f in fields(Config):
        d = f.default
        if isinstance(d, tuple):
            p.add_argument(f"--{f.name}", default=d,
                           type=lambda s: tuple(int(v) for v in s.split(",")))
        else:
            p.add_argument(f"--{f.name}", default=d, type=type(d))
    return Config(**vars(p.parse_args(argv)))


def validate(cfg: Config) -> None:
    assert cfg.downsample & (cfg.downsample - 1) == 0, "downsample must be a power of 2"
    assert cfg.crop % cfg.downsample == 0
    assert cfg.scales[-1] == cfg.crop // cfg.downsample, \
        f"last scale must equal latent size {cfg.crop // cfg.downsample}"
    assert list(cfg.scales) == sorted(cfg.scales) and len(cfg.scales) >= 2
    assert cfg.amp in ("bf16", "fp16", "none")
    _ = int(math.log2(cfg.downsample))