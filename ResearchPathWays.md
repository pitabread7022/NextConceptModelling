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