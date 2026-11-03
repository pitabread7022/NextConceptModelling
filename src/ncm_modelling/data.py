from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

DOTA_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
]
NUM_CLASSES = len(DOTA_CLASSES)
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------- labels
def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def parse_obb(path: Path, W: int, H: int):
    """Ultralytics DOTA-OBB line: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalised). Returns [(cls, pts(4,2) px)]."""
    objs = []
    if not path.exists():
        return objs
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) < 9:
            continue
        pts = np.asarray(t[1:9], dtype=np.float32).reshape(4, 2)
        if pts.max() <= 1.5:                       # normalised -> pixels
            pts = pts * np.array([W, H], dtype=np.float32)
        objs.append((int(float(t[0])), pts))
    return objs


def build_weight_map(objs, W, H, obj_weight, ref_area, max_w):
    """Per-pixel loss weight: 1 on background, larger on (especially small) objects."""
    canvas = Image.new("F", (W, H), 1.0)
    draw = ImageDraw.Draw(canvas)
    polys = []
    for cls, pts in objs:
        area = max(_poly_area(pts), 16.0)
        w = float(np.clip(1.0 + obj_weight * np.sqrt(ref_area / area), 1.0, max_w))
        polys.append((w, pts, cls))
    for w, pts, _ in sorted(polys, key=lambda t: t[0]):     # heavier objects drawn last
        draw.polygon([(float(a), float(b)) for a, b in pts], fill=w)
    centers = [(float(p[:, 0].mean()), float(p[:, 1].mean()), cls) for _, p, cls in polys]
    return np.asarray(canvas, dtype=np.float32).astype(np.float16), centers


# ----------------------------------------------------------------------------- split cache
class DotaSplit:
    """Loads one split fully into RAM (DOTA8 is tiny) with weight maps + object centres."""

    def __init__(self, root, split, crop, obj_weight, ref_area, max_w):
        root = Path(root)
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.exists():
            raise FileNotFoundError(img_dir)
        self.items = []
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in _IMG_EXT:
                continue
            img = np.asarray(Image.open(p).convert("RGB"))
            H, W = img.shape[:2]
            objs = parse_obb(lbl_dir / f"{p.stem}.txt", W, H)
            wmap, centers = build_weight_map(objs, W, H, obj_weight, ref_area, max_w)
            ph, pw = max(crop - H, 0), max(crop - W, 0)
            if ph or pw:
                img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
                wmap = np.pad(wmap, ((0, ph), (0, pw)), constant_values=1)
            self.items.append(dict(name=p.stem, img=img, w=wmap, centers=centers))
        # class -> [(item_idx, cx, cy)] for class-balanced object-centred crops
        self.pool = {}
        for i, it in enumerate(self.items):
            for cx, cy, cls in it["centers"]:
                self.pool.setdefault(cls, []).append((i, cx, cy))

    def __len__(self):
        return len(self.items)


def _hot(centers, x0, y0, c):
    hot = np.zeros(NUM_CLASSES, np.float32)
    for cx, cy, cls in centers:
        if x0 <= cx < x0 + c and y0 <= cy < y0 + c and 0 <= cls < NUM_CLASSES:
            hot[cls] = 1.0
    return hot


def _dihedral(a: np.ndarray, t: int) -> np.ndarray:
    a = np.rot90(a, t % 4)
    if t >= 4:
        a = a[:, ::-1]
    return np.ascontiguousarray(a)


def _to_sample(img, w, hot):
    return {
        "image": torch.from_numpy(img).permute(2, 0, 1).float().div_(127.5).sub_(1.0),  # [-1,1]
        "weight": torch.from_numpy(w.astype(np.float32))[None],
        "cls_hot": torch.from_numpy(hot),
    }


# ----------------------------------------------------------------------------- datasets
class TrainCrops(Dataset):
    def __init__(self, split: DotaSplit, crop: int, length: int, obj_crop_prob: float):
        self.split, self.crop, self.length, self.p = split, crop, length, obj_crop_prob

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = np.random.default_rng()
        c, s = self.crop, self.split
        if s.pool and rng.random() < self.p:
            cls = list(s.pool.keys())[rng.integers(len(s.pool))]          # class-balanced
            ii, cx, cy = s.pool[cls][rng.integers(len(s.pool[cls]))]
            jit = c // 4
            x0 = int(cx - c / 2 + rng.integers(-jit, jit + 1))
            y0 = int(cy - c / 2 + rng.integers(-jit, jit + 1))
        else:
            ii = int(rng.integers(len(s.items)))
            x0 = y0 = None
        it = s.items[ii]
        H, W = it["img"].shape[:2]
        if x0 is None:
            x0, y0 = int(rng.integers(0, W - c + 1)), int(rng.integers(0, H - c + 1))
        x0, y0 = min(max(x0, 0), W - c), min(max(y0, 0), H - c)
        img, w = it["img"][y0:y0 + c, x0:x0 + c], it["w"][y0:y0 + c, x0:x0 + c]
        hot = _hot(it["centers"], x0, y0, c)
        t = int(rng.integers(8))
        return _to_sample(_dihedral(img, t), _dihedral(w, t), hot)


class TileDataset(Dataset):
    """Deterministic tiling (eval / code caching). stride<crop gives overlap."""

    def __init__(self, split: DotaSplit, crop: int, stride: int | None = None, dihedral=(0,)):
        self.split, self.crop = split, crop
        stride = stride or crop
        self.index = []
        for i, it in enumerate(split.items):
            H, W = it["img"].shape[:2]
            xs = list(range(0, W - crop + 1, stride))
            ys = list(range(0, H - crop + 1, stride))
            if xs[-1] != W - crop:
                xs.append(W - crop)
            if ys[-1] != H - crop:
                ys.append(H - crop)
            for y0 in ys:
                for x0 in xs:
                    for t in dihedral:
                        self.index.append((i, x0, y0, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, x0, y0, t = self.index[k]
        it, c = self.split.items[i], self.crop
        img, w = it["img"][y0:y0 + c, x0:x0 + c], it["w"][y0:y0 + c, x0:x0 + c]
        hot = _hot(it["centers"], x0, y0, c)
        out = _to_sample(_dihedral(img, t), _dihedral(w, t), hot)
        out["meta"] = torch.tensor([i, x0, y0, t])
        return out