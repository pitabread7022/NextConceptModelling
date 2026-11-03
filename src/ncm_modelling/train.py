from __future__ import annotations
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .config import Config, parse_args, validate
from .data import DotaSplit, TileDataset, TrainCrops
from .losses import Perceptual, psnr_per_sample, weighted_l1
from .model import NCMImageTokenizer


def cycle(dl):
    while True:
        for b in dl:
            yield b


@torch.no_grad()
def evaluate(model, ds, cfg, dev, amp, tag, out_dir=None, step=0):
    model.eval()
    K, q = model.quant.K, model.quant
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    psnr, psnr_obj = torch.zeros(K), torch.zeros(K)
    n, n_obj = 0, 0
    counts = [torch.zeros(b.size) for b in q.books]
    grid_rows = None
    show = sorted({1, max(1, K // 4), K // 2, (3 * K) // 4, K})
    for bi, batch in enumerate(dl):
        x, w = batch["image"].to(dev), batch["weight"].to(dev)
        with amp():
            f = model.enc(x)
        f_hats, codes, _ = q(f)
        mask = (w > 1.5).float()
        has = mask.sum((1, 2, 3)) > 0
        n += x.shape[0]
        n_obj += int(has.sum())
        recs = {}
        for k in range(K):
            with amp():
                r = model.dec(f_hats[k]).float().clamp(-1, 1)
            psnr[k] += psnr_per_sample(r, x).sum().item()
            if has.any():
                psnr_obj[k] += psnr_per_sample(r, x, mask)[has].sum().item()
            if bi == 0 and (k + 1) in show:
                recs[k + 1] = r[:8]
            counts[k] += torch.bincount(codes[k].flatten().cpu(), minlength=q.sizes[k]).float()
        if bi == 0:
            grid_rows = [x[:8]] + [recs[k] for k in show]
    util = [float((c > 0).float().mean()) for c in counts]
    ppl = []
    for c in counts:
        p = c / c.sum().clamp_min(1)
        ppl.append(float(torch.exp(-(p * torch.log(p.clamp_min(1e-10))).sum())))
    res = dict(psnr=(psnr / max(n, 1)).tolist(),
               psnr_obj=(psnr_obj / max(n_obj, 1)).tolist(), util=util, ppl=ppl)
    print(f"\n[eval:{tag}] step {step}  tiles={n}  tiles_with_objects={n_obj}")
    print("  lvl scale  PSNR  objPSNR  util%   ppl")
    for k in range(K):
        print(f"  {k+1:>3} {q.scales[k]:>5} {res['psnr'][k]:>6.2f} {res['psnr_obj'][k]:>8.2f} "
              f"{100*util[k]:>6.1f} {ppl[k]:>6.0f}")
    if out_dir is not None and grid_rows is not None:
        grid = torch.cat(grid_rows, 0).add(1).div(2).clamp(0, 1)
        save_image(grid, Path(out_dir) / "samples" / f"{tag}_step{step:06d}.png", nrow=grid_rows[0].shape[0])
    model.train()
    return res


def main():
    cfg: Config = parse_args()
    validate(cfg)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out = Path(cfg.out_dir); (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # ---------------- data
    mk = lambda split: DotaSplit(cfg.data_root, split, cfg.crop, cfg.obj_weight, cfg.ref_obj_area, cfg.max_weight)
    train_split = mk("train")
    try:
        val_split = mk("val")
    except FileNotFoundError:
        val_split = None
    print(f"train images={len(train_split)}  classes with objects={sorted(train_split.pool)}")
    dl = DataLoader(TrainCrops(train_split, cfg.crop, cfg.virtual_epoch, cfg.obj_crop_prob),
                    batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.num_workers, pin_memory=dev.type == "cuda",
                    persistent_workers=cfg.num_workers > 0)
    it = cycle(dl)
    eval_sets = {"train": TileDataset(train_split, cfg.crop)}
    if val_split is not None:
        eval_sets["val"] = TileDataset(val_split, cfg.crop)

    # ---------------- model / optim
    model = NCMImageTokenizer(cfg).to(dev)
    K = model.quant.K
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | levels={K} "
          f"| codes/image={model.quant.num_codes} | codebook sizes={model.quant.sizes}")
    perc = Perceptual(cfg.lambda_lpips > 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    use_amp = cfg.amp != "none" and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if (cfg.amp == "bf16" and torch.cuda.is_available()
                                   and torch.cuda.is_bf16_supported()) else torch.float16
    amp = lambda: torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp)
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    start, best = 0, -1.0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck.get("best", -1.0)
        print(f"resumed from {cfg.resume} @ step {start}")

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * s / max(cfg.warmup, 1)
        p = (s - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def recon_loss(pred, x, w):
        loss = weighted_l1(pred.float(), x, w)
        if perc.enabled:
            loss = loss + cfg.lambda_lpips * perc(pred, x)
        return loss

    def save(name, step):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), cfg=asdict(cfg),
                        step=step, best=best), out / name)

    # ---------------- loop
    model.train()
    t0, log = time.time(), out / "metrics.jsonl"
    for step in range(start + 1, cfg.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next(it)
        x, w = batch["image"].to(dev, non_blocking=True), batch["weight"].to(dev, non_blocking=True)

        k_rand = random.randint(1, K - 1)                       # dual-cut nested dropout
        with amp():
            recons, _, vq = model(x, [K, k_rand])
        l_full = recon_loss(recons[K], x, w)
        l_cut = recon_loss(recons[k_rand], x, w)
        loss = l_full + cfg.lambda_cut * l_cut + cfg.lambda_vq * vq

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt); scaler.update()

        if cfg.reinit_every and step % cfg.reinit_every == 0 and step < cfg.reinit_until * cfg.steps:
            n_re = model.quant.reinit_dead(cfg.dead_ratio)
            if sum(n_re):
                print(f"  [dead-code reinit] per level: {n_re}")

        if step % cfg.log_every == 0:
            rec = dict(step=step, loss=loss.item(), full=l_full.item(), cut=l_cut.item(),
                       cut_levels=k_rand, vq=float(vq), lr=lr_at(step),
                       it_s=cfg.log_every / (time.time() - t0))
            t0 = time.time()
            print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

        if step % cfg.eval_every == 0 or step == cfg.steps:
            res = {t: evaluate(model, ds, cfg, dev, amp, t, out, step) for t, ds in eval_sets.items()}
            key = res.get("val", res["train"])["psnr"][-1]
            if key > best:
                best = key; save("best.pt", step)
        if step % cfg.save_every == 0 or step == cfg.steps:
            save("last.pt", step)
    print(f"done. best full-level PSNR = {best:.2f} dB  -> {out/'best.pt'}")


if __name__ == "__main__":
    main()