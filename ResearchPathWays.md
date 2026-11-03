NCM is a good fit for solo projects. The paper's models are about 100M parameters, the TinyStories setup is small (a 36M tokenizer plus an 88M generator, trained for 50k steps), and the code is released at github.com/HazyResearch/ncm. These are the projects I'd consider, roughly from most to least practical.

## Strong starter projects

**1. Reproduce and extend the ablations (TinyStories)**
Re-run the tokenizer and generator end to end, then push on the open questions the authors list: input-dependent level schedules (adaptive K per example), corruption based on the model's actual errors instead of uniform random replacement, and end-to-end training of the tokenizer and generator. It's cheap, and any of these could become a small paper or workshop submission.

**2. Constraint-satisfaction and puzzle solvers**
The paper's Sudoku result (a 25M model, 12 NFE, 86% on hard puzzles) suggests coarse-to-fine refinement suits global-constraint problems. Try Latin squares, Kakuro, N-Queens, small scheduling or SAT instances, or maze paths. The vocabularies are tiny, so tokenizer reconstruction can be exact. Since these tasks have verifiers, you can compare against AR and masked-diffusion baselines directly.

**3. Verifier-guided test-time scaling**
Sampling is cheap, so you can spend the savings on search. The paper uses MBR decoding, but with a real checker (puzzle validity, unit tests, a parser) you can do best-of-K, or prune candidates at the coarse levels and only refine the survivors, which is a form of beam search over levels. I haven't seen this done and it's easy to test.

## Application projects

**4. Structured or code generation**
Coarse levels might capture the skeleton (the shape of a JSON schema, SQL query, config file, or parametric CAD script) and finer levels fill in the details. Exact reconstruction matters here. The paper reports 99.8–99.99% token accuracy on text, but a single wrong token breaks code, so measuring validity rate is the central question. You could test it on OpenSCAD, since the structure and the parameters separate naturally.

**5. Niche fast translation or summarization**
Train a small NCM on a language pair or domain you care about (English–Hindi, say, or domain-specific summaries) and benchmark wall-clock latency against a small AR model on your own GPU. The paper's speed claims come from H100 runs against diffusion baselines, so independent measurements on different hardware would be valuable.

## More experimental

**6. Controllable generation and an editing demo**
The paper shows that fixing coarse codes preserves a story's theme, mixing coarse codes blends two stories, and perturbing two early codes changes characters and events. You could build an interactive tool for steering text by editing coarse codes, and probe which levels control topic, entities, and style.

**7. Multiscale residual VQ for time series**
This goes beyond the paper, so treat it as speculative. The same recipe (nested dropout plus coarse-to-fine generation) could apply to telemetry such as solar irradiance or inverter power. Coarse levels would capture the daily or seasonal shape and fine levels the transients. Uses include synthetic data generation, imputation, and anomaly detection via per-level reconstruction error.

**8. NCM as a drafter or planner for an AR model**
Use the fast coarse levels to produce a plan or draft that a larger AR model conditions on or verifies, similar in spirit to diffusion drafters for speculative decoding. This is the hardest option and the least certain to pay off.

## Caveats

- All results are at about 100M parameters. The scaling evidence (up to 725M on LM1B) is preliminary.
- The tokenizer makes rare errors on named entities, numerals, and diacritics, so check how that affects your task.
- The paper's samples show fluency artifacts such as repeated words, and they note artifacts at coarse-level transitions.
- The 4–5× lower training-token budget counts token exposure, not FLOPs.

If you want one place to start, I'd pick #2 or #3. Both have clear verifiers and cheap compute, and they build on the paper's strongest result. I can sketch an experiment plan for any of them if you tell me your GPU budget.

# apply NCM VQ VAE to dota8 dataset

1. **The NCM tokenizer itself is text-specific.** Its encoder embeds tokens, its decoder predicts token logits with cross-entropy, and the repo reflects that. For images, the same idea already exists as the multiscale residual VQ-VAE in VAR (next-scale prediction), which the NCM paper cites as its inspiration. So you would be applying the NCM/VAR *recipe* to images, not NCM's code.
2. **Eight images is far too few to train a VQ-VAE.** It would memorize them. Use DOTA8 only as a smoke test (does the pipeline run, can it overfit). Train on random crops from full DOTA or your own satellite tiles. Fine-tuning a pretrained VAR tokenizer is more realistic than training from scratch.

## What changes from text to images

| NCM (text) | Image version |
|---|---|
| Token embedding encoder, L×d latents | Conv encoder to a d×16×16 latent grid |
| Level k has ℓ_k codes | Level k is an s_k×s_k grid, e.g. (1,2,3,4,6,8,12,16) |
| Down_k / R_k on 1D sequences | `interpolate` down, then up plus a small conv |
| Decoder with cross-entropy over tokens | Conv decoder with L1 + LPIPS (optionally GAN) loss |
| Nested dropout over levels | Same, unchanged |
| Loss at the sampled cutoff | Same: commitment + codebook loss on the cumulative reconstruction |

## Quantizer sketch (PyTorch, untested)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MSResidualVQ(nn.Module):
    def __init__(self, d=256, V=4096, scales=(1,2,3,4,6,8,12,16), beta=0.25):
        super().__init__()
        self.scales, self.beta = scales, beta
        self.codebooks = nn.ModuleList(nn.Embedding(V, d) for _ in scales)
        self.phi = nn.ModuleList(nn.Conv2d(d, d, 3, padding=1) for _ in scales)  # R_k

    def forward(self, f, k_cut=None):            # f: B,d,H,W from the encoder
        B, d, H, W = f.shape
        K = len(self.scales); k_cut = k_cut or K  # nested dropout: sample k_cut in training
        f_hat, codes = torch.zeros_like(f), []
        for k in range(k_cut):
            s = self.scales[k]
            u = F.interpolate((f - f_hat).detach(), size=(s, s), mode='area')   # Down_k
            flat = u.permute(0,2,3,1).reshape(-1, d)
            idx = torch.cdist(flat, self.codebooks[k].weight).argmin(1)
            e = self.codebooks[k](idx).view(B, s, s, d).permute(0,3,1,2)
            up = F.interpolate(e, size=(H, W), mode='bicubic')
            f_hat = f_hat + self.phi[k](up)                                      # R_k
            codes.append(idx.view(B, s, s))
        loss = self.beta * F.mse_loss(f_hat.detach(), f) + F.mse_loss(f_hat, f.detach())
        f_st = f + (f_hat - f).detach()          # straight-through to the encoder
        return f_st, codes, loss
```

Training step: `z = enc(img)`, then `k = random.randint(1, K)` (with some probability use the full K), then `z_q, codes, vq_loss = quant(z, k)`, `recon = dec(z_q)`, and `loss = l1(recon, img) + λ·lpips(recon, img) + vq_loss`. Add EMA codebook updates and dead-code resets if utilization is low (the paper reports 62–90% utilization).

## Steps

1. **Data:** Cut DOTA (or your own imagery) into 256×256 crops with random flips and rotations. Ignore the oriented-box labels for the tokenizer.
2. **Base:** Start from VAR's open-source multiscale VQ-VAE (check the repo for its exact scale config) and fine-tune it on your crops, or train the sketch above from scratch.
3. **Smoke test on DOTA8:** Overfit the 8 images and confirm that reconstruction improves with more levels.
4. **Evaluate:**
   - PSNR/LPIPS after each level.
   - Codebook utilization per level.
   - Visual reconstructions per level.
   - Small-object survival. DOTA has tiny vehicles that a 16× downsample can wipe out, so try 8× (a 32×32 grid for 256 crops) if they vanish.
5. **Optional second stage:** Train a VAR-style transformer that predicts level k from the cumulative reconstruction of levels 1..k−1. NCM's text-specific parts (prefix-LM conditioning, block-diagonal attention over 1D blocks) need adapting to 2D.

## Limits

- A VQ-VAE doesn't do detection. It gives you compression, generation, or coarse-to-fine features. If your goal is detection or segmentation of rooftops, a pretrained detector will serve you better, and the codes would only be an auxiliary representation.
- None of this has been validated on aerial imagery, so treat it as an experiment.

What are you hoping to get out of it: generation, compression, or features for a downstream detector?