from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .model import NCMImageTokenizer


class DecoderCodes(nn.Module):          # (B,N) int32 codes -> image   [main inference artifact]
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, codes): return self.m.decode(codes)


class DecoderLatent(nn.Module):         # (B,d,L,L) latent -> image
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, z): return self.m.dec(z).clamp(-1.0, 1.0)


class LevelStep(nn.Module):             # (f_hat, codes_k) -> f_hat'  (cumulative reconstruction update)
    def __init__(self, m, k): super().__init__(); self.q, self.k = m.quant, k
    def forward(self, f_hat, idx): return self.q.add_level(f_hat, self.k, idx)


class EncoderCodes(nn.Module):          # image -> (B,N) int32 codes (needs scales dividing the latent size)
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m.quant.encode_flat(self.m.enc(x))


def export(mod, args, path, in_names, out_names, dyn, opset):
    kw = dict(input_names=in_names, output_names=out_names, dynamic_axes=dyn, opset_version=opset)
    try:
        torch.onnx.export(mod, args, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(mod, args, str(path), **kw)
    print("exported", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ncm_img/best.pt")
    ap.add_argument("--out", default="runs/ncm_img/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--level_steps", action="store_true")
    ap.add_argument("--encoder", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    m = NCMImageTokenizer(cfg); m.load_state_dict(ck["model"]); m.eval()
    q, out = m.quant, Path(a.out); out.mkdir(parents=True, exist_ok=True)

    codes = m.random_codes(2)
    export(DecoderCodes(m), (codes,), out / "decoder_codes.onnx", ["codes"], ["image"],
           {"codes": {0: "b"}, "image": {0: "b"}}, a.opset)
    z = torch.randn(1, q.d, q.L, q.L)
    export(DecoderLatent(m), (z,), out / "decoder_latent.onnx", ["latent"], ["image"],
           {"latent": {0: "b"}, "image": {0: "b"}}, a.opset)
    if a.level_steps:
        for k, s in enumerate(q.scales):
            export(LevelStep(m, k), (z, torch.randint(0, q.sizes[k], (1, s * s)).int()),
                   out / f"level_step_{k:02d}.onnx", ["f_hat", "idx"], ["f_hat_out"],
                   {"f_hat": {0: "b"}, "idx": {0: "b"}, "f_hat_out": {0: "b"}}, a.opset)
    if a.encoder:
        try:
            export(EncoderCodes(m), (torch.randn(1, 3, cfg.crop, cfg.crop),), out / "encoder_codes.onnx",
                   ["image"], ["codes"], {"image": {0: "b"}, "codes": {0: "b"}}, a.opset)
        except Exception as e:
            print(f"[warn] encoder export failed (adaptive pooling to non-divisor grids?): {e}\n"
                  "       use scales that divide the latent size, e.g. 1,2,4,8,16,32")

    if a.verify:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out / "decoder_codes.onnx"), providers=["CPUExecutionProvider"])
        ref = m.decode(codes).detach().numpy()
        got = sess.run(None, {"codes": codes.numpy()})[0]
        print("max |torch - onnx| =", abs(ref - got).max())

    N = q.num_codes
    print(f"\nTensorRT:\n  trtexec --onnx={out/'decoder_codes.onnx'} --saveEngine={out/'decoder_codes.plan'} "
          f"--fp16 --minShapes=codes:1x{N} --optShapes=codes:8x{N} --maxShapes=codes:32x{N}")


if __name__ == "__main__":
    main()