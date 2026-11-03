"""Tokenise a split into (codes, cls_hot) arrays for training the Transformer generator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import DotaSplit, TileDataset
from .model import NCMImageTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ncm_img/best.pt")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--dihedral", action="store_true", help="add all 8 flips/rotations")
    ap.add_argument("--out", default="runs/ncm_img/codes")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = NCMImageTokenizer(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for split in a.splits.split(","):
        sp = DotaSplit(cfg.data_root, split, cfg.crop, cfg.obj_weight, cfg.ref_obj_area, cfg.max_weight)
        ds = TileDataset(sp, cfg.crop, a.stride, tuple(range(8)) if a.dihedral else (0,))
        codes, hots, metas = [], [], []
        for b in DataLoader(ds, batch_size=32):
            codes.append(model.encode(b["image"].to(dev)).cpu().numpy().astype(np.int16))
            hots.append(b["cls_hot"].numpy()); metas.append(b["meta"].numpy())
        np.savez_compressed(out / f"{split}.npz", codes=np.concatenate(codes),
                            cls_hot=np.concatenate(hots), meta=np.concatenate(metas))
        print(f"{split}: {sum(len(c) for c in codes)} samples")
    q = model.quant
    (out / "meta.json").write_text(json.dumps(dict(
        scales=q.scales, offsets=q.offsets, codebook_sizes=q.sizes, num_codes=q.num_codes,
        latent_size=q.L, latent_dim=q.d, crop=cfg.crop), indent=2))


if __name__ == "__main__":
    main()