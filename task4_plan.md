# Task 4 Plan — Style-Conditioned Face-to-Sketch Generation (Conditional GAN)

> Scope: this plan covers **Task 4** of Assignment #1 (Generative AI) in full — dataset, architecture, losses,
> training, Optuna HPO, experiment tracking, evaluation, ONNX export, application workspace, backend API,
> Docker deployment, and the report/video/repo deliverables that apply to this task. Cross-cutting assignment
> rules (Optuna, MLflow/W&B, ONNX, Docker, Stitch, report format, AI-use appendix, submission logistics) are
> included wherever the spec applies them to "each task" / "all four tasks."
>
> ⚠️ **Date check:** the assignment PDF states a deadline of **March 16, 2024**, which is clearly a stale/template
> date relative to when this plan is being written. Confirm the real due date with the instructor / Google
> Classroom before finalizing a schedule — everything below is organized by phase/dependency, not by calendar date.

---

## 0. Pre-Implementation & Ongoing Research Plan

The assignment requires every non-trivial technical choice to be evidence-backed, not defaulted into. This
section is the reading/investigation plan that feeds those decisions, tiered by when it blocks progress.
Each row names the source, which plan section/decision it backs, and what to capture as "evidence" for the
report (the assignment explicitly wants alternatives investigated + why one was chosen + what difficulties
came up).

### 0.1 Tier A — Blocking (read before writing any training code)

| # | Topic / source | Backs (decision) | Evidence to capture for the report |
|---|---|---|---|
| A1 | **pix2pix** — P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros, *"Image-to-Image Translation with Conditional Adversarial Networks,"* CVPR 2017. This is the base recipe the assignment describes (U-Net G, PatchGAN D, L1+adversarial objective, λ=100 convention). Read the method **and** the ablation section (Sec. 5.1/5.2 on loss and generator/discriminator variants). | §3 (architecture), §4 (loss objective), §6 (Optuna λ_L1 range) | Note which of pix2pix's own ablations you're reusing vs. re-testing yourself on FS2K |
| A2 | **FS2K dataset paper** — D.-P. Fan, Z. Huang, P. Zheng, H. Liu, X. Qin, L. Van Gool, *"Facial-Sketch Synthesis: A New Challenge,"* Machine Intelligence Research, vol. 19, pp. 257–287, 2022 (arXiv:2112.15439). Confirms the 3 style categories, dataset composition, and the authors' own FSGAN baseline. | §2 (dataset/splits), report's related-work section | Record how FS2K defines its official train/test split and style labels in metadata — this is what your split code must match exactly |
| A3 | **Data audit** (empirical, not a paper) — inspect actual FS2K files: sketch channel count (grayscale vs RGB), value range, filename-based pairing scheme. | §2, §3.1 (generator output channels) | A short written note of what you found — this becomes a one-paragraph "dataset characteristics" subsection in the report |

### 0.2 Tier B — Needed before finalizing architecture (research in parallel with a throwaway baseline)

| # | Topic / source | Backs (decision) | Evidence to capture |
|---|---|---|---|
| B1 | **Conditioning mechanisms** — compare (a) naive channel-concatenation of the broadcasted style embedding, (b) **FiLM**: E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville, *"FiLM: Visual Reasoning with a General Conditioning Layer,"* AAAI 2018, and (c) **projection discriminator** (for the D side): T. Miyato and M. Koyama, *"cGANs with Projection Discriminator,"* ICLR 2018. | §3.3 (style-embedding fusion), §15 Q1 | A small side-by-side: same photo, 3 styles, 3 outputs, for each conditioning method you actually tried |
| B2 | **PatchGAN receptive field** — reread pix2pix's own supplementary ablation (1×1 / 16×16 / 70×70 / 286×286 receptive fields) as the starting reference for patch size selection. | §3.2, §15 Q2 | State the receptive field you chose and cite the pix2pix finding it's based on |

### 0.3 Tier C — Needed before Optuna / full training (research while the baseline trains)

| # | Topic / source | Backs (decision) | Evidence to capture |
|---|---|---|---|
| C1 | **Small-dataset GAN stability** — **DiffAugment**: S. Zhao, Z. Liu, J. Lin, J.-Y. Zhu, S. Han, *"Differentiable Augmentation for Data-Efficient GAN Training,"* NeurIPS 2020; and **spectral normalization**: T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida, *"Spectral Normalization for Generative Adversarial Networks,"* ICLR 2018. FS2K has only 2,104 pairs — expect to need at least one stabilizer. | §15 (GAN-stability question), §6 (Optuna extra knob if adopted) | Loss/sample-quality comparison with vs. without the stabilizer you adopt |
| C2 | **Evaluation metrics** — **LPIPS**: R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, *"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,"* CVPR 2018; **FID**: M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, S. Hochreiter, *"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,"* NeurIPS 2017. Note FID's reliability typically assumes thousands of samples — with FS2K's small test set this may not hold, which is itself worth reporting as a limitation. | §9 (evaluation plan), §15 Q5 | Explicit justification for which metric(s) you kept and why FID was/wasn't trustworthy at this sample size |
| C3 | **Optuna pruning for adversarial objectives** — GAN loss doesn't monotonically improve, so pruning directly on raw D/G loss is misleading; check Optuna's pruner docs for guidance on using a periodic validation proxy (e.g., val L1/LPIPS at fixed epochs) instead. | §6 (Optuna study design) | State the pruning signal used and why raw adversarial loss was rejected as a pruning metric |

### 0.4 Tier D — Just-in-time (look up when you hit the problem, non-blocking)

| # | Topic | Backs |
|---|---|---|
| D1 | ONNX opset compatibility for `nn.Embedding` + transposed-conv upsampling paths | §8 (ONNX export) |
| D2 | `getUserMedia` webcam-capture pattern in React | §9 (frontend workspace) |

### 0.5 Draft Reference List (refine into full IEEE format for the report)

```
[1] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional
    adversarial networks," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 1125–1134.
[2] D.-P. Fan, Z. Huang, P. Zheng, H. Liu, X. Qin, and L. Van Gool, "Facial-sketch synthesis: A new
    challenge," Mach. Intell. Res., vol. 19, pp. 257–287, 2022.
[3] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a
    general conditioning layer," in Proc. AAAI Conf. Artif. Intell., 2018.
[4] T. Miyato and M. Koyama, "cGANs with projection discriminator," in Proc. Int. Conf. Learn.
    Represent. (ICLR), 2018.
[5] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral normalization for generative
    adversarial networks," in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.
[6] S. Zhao, Z. Liu, J. Lin, J.-Y. Zhu, and S. Han, "Differentiable augmentation for data-efficient
    GAN training," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.
[7] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of
    deep features as a perceptual metric," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018.
[8] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two
    time-scale update rule converge to a local Nash equilibrium," in Proc. Adv. Neural Inf. Process.
    Syst. (NeurIPS), 2017.
```
*(Verify each entry against the actual published version before submission — venues/page numbers can drift
across preprint vs. camera-ready copies.)*

---

## 1. Requirement Traceability Checklist

Use this as the master checklist. Every line is a literal obligation from the assignment text (Task 4 section
plus the cross-cutting sections that name "each task" / "all four tasks").

### 1.1 Dataset & Splits
- [ ] Use the **FS2K Facial Sketch Synthesis Dataset** (2,104 paired photo/sketch samples, 3 sketch styles). Repo: https://github.com/DengPingFan/FS2K
- [ ] Use FS2K's **official train/test split** — do not re-split the official test set.
- [ ] From the **official training portion**, reserve **15%** as a validation set.
- [ ] Validation split uses **random seed 42**.
- [ ] Validation split must be **stratified by sketch style** (proportional representation of the 3 styles).
- [ ] Official test set is **not touched** during training or hyperparameter selection (Optuna).
- [ ] Resize **both** photographs and sketches to **128×128**.
- [ ] Preserve **correct pairing** between every photo and its target sketch through every transform.

### 1.2 Augmentation
- [ ] Any spatial augmentation (crop, flip, rotate, resize) is applied **identically** to the photo and its
      paired sketch in the same call (shared random state), never independently.

### 1.3 Architecture
- [ ] Generator: **U-Net-style encoder–decoder**.
- [ ] Generator input: facial photograph **+ selected style condition** → output: sketch.
- [ ] Discriminator: **PatchGAN** — judges local patches of a (photo, sketch) pair as real/fake, not the whole image.
- [ ] Style condition implemented as a **learned categorical embedding** over the 3 FS2K styles.
- [ ] The embedding is **fused into both the generator and the discriminator** internals — not just used as a
      dataset label / interface tag.
- [ ] Discriminator receives **photograph + style condition + (real OR generated) sketch**.

### 1.4 Losses & Objective
- [ ] Discriminator trained to score real pairs as real, generated pairs as fake.
- [ ] Generator trained to (a) fool the discriminator and (b) stay close to the paired ground-truth sketch.
- [ ] Generator objective = **adversarial loss + reconstruction loss**.
- [ ] Discriminator / adversarial terms **may use BCE-with-logits**.
- [ ] Initial reconstruction weight **λ_L1 = 100** (pix2pix convention) as a *starting point only*.
- [ ] Final λ_L1 must be **selected via Optuna**, not accepted by default.

### 1.5 Hyperparameter Optimization (Optuna — mandatory for every task)
- [ ] Optuna search tunes **at least**: generator LR, discriminator LR, batch size, base channel count,
      dropout rate, style-embedding dimension, reconstruction-loss weight (λ_L1).
- [ ] Because GAN training is expensive, **trials may use a reduced epoch budget**.
- [ ] The **best trial's config is retrained for the full training schedule** afterward.
- [ ] Report: complete search space, number of completed trials, best trial, final chosen configuration.

### 1.6 Training Logging (this task specifically + cross-cutting tracking rule)
- [ ] Log, **separately**, per step/epoch: discriminator real loss, discriminator fake loss, generator
      adversarial loss, generator reconstruction loss, and validation measurements.
- [ ] At **fixed intervals**, log generated samples for the **same fixed set of validation photographs**, so
      generator progress is visible over time.
- [ ] All of the above (hyperparameters, losses, eval results, checkpoints, visual outputs) recorded in
      **MLflow or Weights & Biases** (assignment-wide requirement, applies to Task 4 too).

### 1.7 Export
- [ ] Export the **final generator only** to **ONNX** (discriminator is training-only, not exported).
- [ ] Verify ONNX generator output is numerically consistent with the PyTorch/TF generator output
      (assignment-wide ONNX-parity requirement).

### 1.8 Application — "Face-to-Sketch Generator" workspace
- [ ] Workspace is one of the app's 4 clearly-accessible workspaces (single browser app, React + Tailwind
      frontend, FastAPI backend — assignment-wide).
- [ ] User can **upload a photo OR capture one via webcam**.
- [ ] User selects **Style 1 / Style 2 / Style 3** (fixed categorical options — free-text prompting is *not*
      required or expected).
- [ ] "Generate" produces the sketch via the exported ONNX generator.
- [ ] UI shows **original photo and generated sketch side by side**.
- [ ] UI provides a **download** option for the result.
- [ ] Backend exposes at minimum a **face-to-sketch** inference operation (alongside health-check,
      universal-restoration, hard-routing, soft-mixture from Tasks 1–3).
- [ ] Visual design for this workspace is **first designed in Google Stitch**; evidence of that design is
      captured for the report.
- [ ] Works end-to-end inside **Docker Compose** (frontend + backend containers, one documented startup command,
      no VS Code / manual script running needed by the evaluator).
- [ ] Must run correctly on **previously unseen photographs** supplied live during evaluation.

### 1.9 Repository (assignment-wide, Task-4-relevant parts)
- [ ] Task 4 source code, config, data-prep scripts, training script, Optuna study, evaluation script, ONNX
      export script all committed.
- [ ] Dockerfile(s) covering the backend that serves the Task 4 model; docker-compose entry wired in.
- [ ] README section with exact commands to reproduce Task 4 data prep → training → Optuna → export → app.
- [ ] **No full dataset or large checkpoints committed directly** — use a documented download link or Git LFS.

### 1.10 Report, Video, AI-use (assignment-wide, Task-4-relevant parts)
- [ ] IEEE-format LaTeX report section for Task 4: architecture, losses, training procedure, Optuna results,
      experimental results, visual outputs, failure cases, application design, conclusions.
- [ ] Includes: generator/discriminator architecture diagram, training curves (4 loss curves + validation),
      Optuna results, quantitative result table(s), generated-image grids, error maps, application screenshots,
      Stitch design evidence, meaningful failure cases — each one **interpreted in text**, not just pasted in.
- [ ] 5–7 minute demo video (uploaded to YouTube, link only in report) shows, among the other tasks,
      **face-to-sketch generation, result downloading, and experiment-tracking records**.
- [ ] AI-use appendix entry: which AI tools were used for Task 4 (research/coding/design/debugging/docs), what
      for, and how outputs were verified/corrected.
- [ ] Be ready to justify the architecture, explain any research-informed decision, interpret a result, modify
      part of the implementation, or run Task 4 on unseen images live.

---

## 2. Dataset Pipeline — FS2K

| Item | Spec |
|---|---|
| Source | FS2K GitHub repo (photos, sketches, per-sample style label, official train/test list) |
| Size | 2,104 paired samples, 3 sketch styles |
| Split | Official train / official test (test untouched until final eval) |
| Validation | 15% of official train, seed 42, **stratified by style** |
| Resolution | 128×128, both modalities |
| Color | Photo: RGB. Sketch: keep native channel count from FS2K but resize identically; confirm during data audit whether sketches are RGB or single-channel and document the choice |
| Pairing | Every transform (resize, crop, flip, rotation) applied to the (photo, sketch) tuple with one shared random draw, never sampled twice |

**Build steps**
1. `download_fs2k.py` — fetch/verify the dataset, checksum the file counts against the paper's 2,104 figure.
2. Parse FS2K's provided train/test annotation lists (do not re-derive from folder structure only).
3. `dataset.py` — `FS2KPairedDataset`: loads photo, sketch, style label; returns `(photo, sketch, style_id)`.
4. Stratified 85/15 split of the official train list only, `sklearn.model_selection.train_test_split(..., stratify=style_labels, random_state=42)`.
5. `augmentations.py` — a **paired transform wrapper** (e.g., a single `random.Random` seed per sample, or
   Albumentations' `additional_targets` mechanism) so flip/rotate/crop/resize is identical across the pair.
6. Sanity-check script: visualize N random pairs post-augmentation to confirm alignment wasn't broken.

---

## 3. Model Architecture

### 3.1 Generator — conditional U-Net
- Encoder: strided conv blocks (Conv→Norm→LeakyReLU), progressively downsampling 128→64→32→16→8→...
- Decoder: transposed-conv / upsample+conv blocks with **U-Net skip connections** from matching encoder
  resolutions (skip connections are appropriate here — Task 4 is not a bottleneck-autoencoder task like
  Tasks 1–3, so no "meaningful bottleneck" justification is required for this task).
- Output: `tanh` activation to a 128×128 sketch image (channel count matches whatever the dataset audit in
  §2 determines for sketches).

### 3.2 Discriminator — PatchGAN
- Series of strided conv blocks (no global pooling / FC head) producing an N×N grid of real/fake logits, each
  logit corresponding to a receptive-field patch of the input pair.
- Input: photograph concatenated with the (real or generated) sketch, plus the style conditioning (see §4).
- Patch size (receptive field) is an architectural choice to investigate (e.g., 70×70 vs 30×30 receptive
  field, per pix2pix-style ablations) — record the chosen value and why in the report.

### 3.3 Style Embedding
- `nn.Embedding(num_styles=3, embedding_dim=d)` — `d` itself is an Optuna-tunable hyperparameter (§1.5 / §8).
- Fusion strategy to select and justify with evidence (this is a required research decision, not a default):
  - Option A: broadcast the embedding spatially and **concatenate as extra channels** at the generator input
    (and again at the discriminator input).
  - Option B: inject via **FiLM / conditional normalization** at one or more encoder-decoder layers.
  - Option C: inject only at the bottleneck.
- Whichever is chosen must demonstrably affect generation (ablation: same photo, 3 styles → 3 visibly
  different sketches) — this is how "incorporated rather than used only as an interface label" gets proven.

---

## 4. Loss Functions & Training Objective

Reconstructing the (partially garbled in the source PDF) formulas in standard pix2pix-style notation:

- Generator: `ŷ = G(x, s)` — `x` = input photo, `s` = style condition, `ŷ` = generated sketch.
- Discriminator: `D(x, s, y) → patch-wise real/fake logits`, evaluated once on the real pair `(x, s, y)` and
  once on the fake pair `(x, s, ŷ)`.
- Adversarial loss (BCE-with-logits form):
  - `L_D = BCE(D(x, s, y), 1) + BCE(D(x, s, ŷ), 0)`
  - `L_adv(G) = BCE(D(x, s, ŷ), 1)`
- Reconstruction loss: `L_L1 = || y − ŷ ||₁`
- Full generator objective: `L_G = L_adv(G) + λ_L1 · L_L1`
- λ_L1 initial value 100 (pix2pix convention), **final value chosen by Optuna** (§1.4/§1.5).

Open research item: whether to add an auxiliary perceptual/SSIM term on top of L1 (not mandated by the spec,
but worth investigating and justifying either way given the "investigate alternative loss functions"
requirement in the assignment's general instructions).

---

## 5. Training Procedure

1. Standard adversarial training loop: alternate one discriminator step and one generator step per batch
   (or an investigated `n_critic` ratio — record if changed from 1:1 and why).
2. Optimizers: Adam with β1≈0.5, β2≈0.999 (pix2pix/DCGAN convention) — β values and separate G/D learning
   rates are open Optuna/research decisions.
3. Per step, log **D real loss, D fake loss, G adversarial loss, G reconstruction loss** as four separate
   series (not merged) — required by the spec.
4. Per epoch, compute validation measurements (see §9) on the held-out validation split.
5. At fixed intervals (e.g., every K epochs), run the generator on the **same fixed batch of validation
   photographs** across all 3 styles and log the resulting image grid — required so progression is visible.
6. Log all of the above to **MLflow or W&B**, including config, checkpoints, and the periodic image grids.
7. Train the Optuna-selected best config for the **full schedule** (more epochs than trial runs) to produce
   the final model used for evaluation, ONNX export, and the app.

---

## 6. Hyperparameter Optimization (Optuna)

| Hyperparameter | Role | Suggested search space (to refine with pilot runs) |
|---|---|---|
| Generator learning rate | G optimizer | log-uniform, e.g. 1e-5 – 5e-4 |
| Discriminator learning rate | D optimizer | log-uniform, e.g. 1e-5 – 5e-4 |
| Batch size | data loader | categorical {4, 8, 16, 32} (bounded by GPU memory + small dataset size) |
| Base channel count | G/D width | categorical {32, 48, 64, 96} |
| Dropout rate | G/D regularization | uniform 0.0 – 0.5 |
| Style-embedding dimension | conditioning | categorical {4, 8, 16, 32} |
| Reconstruction-loss weight λ_L1 | objective | log-uniform, e.g. 10 – 200 (centered on 100) |

- Objective/validation metric to optimize: propose a combination such as validation L1 (and/or an
  investigated perceptual metric, see §9) — must be explicitly chosen and justified, since the spec does not
  fix one for Task 4 the way it does for Task 1.
- Use **reduced epoch budget per trial**; consider Optuna pruning (e.g., `MedianPruner`) on runs showing GAN
  instability (mode collapse, exploding D loss, dead generator output).
- Deliverable for the report: full search space table, number of trials completed, best trial parameters +
  score, and the final configuration actually used for full training.

---

## 7. Evaluation Plan

The spec does not fix Task-4 metrics explicitly (unlike Task 1's PSNR/SSIM-by-severity structure), so metric
selection is itself a research decision to investigate and justify:

- **Quantitative** (candidates to evaluate and choose from, with citations in the report):
  - Pixel-level: L1 / MAE between generated and ground-truth sketch.
  - Structural: SSIM.
  - Perceptual: LPIPS (better correlation with human judgment for sketch-style transfer than raw pixel
    losses).
  - Distributional (only if sample count justifies it): FID between generated and real sketch sets, per
    style.
- **Per-style breakdown**: report all chosen metrics separately for Style 1 / Style 2 / Style 3, since the
  model is explicitly conditioned on style.
- **Qualitative**: generated-image grids (same validation photos across training time, per §5.5), side-by-side
  photo/ground-truth-sketch/generated-sketch/error-map panels, covering all three styles.
- **Failure cases**: identify and discuss meaningful failure modes (e.g., a style the generator undershoots,
  extreme poses/lighting, background clutter) — mirrors the "meaningful failure cases" expectation applied to
  every task's results discussion.
- **Error maps**: pixel-wise absolute difference between generated and ground-truth sketch, for the report's
  visual-outputs requirement.
- Run final evaluation on the **untouched official test set** only after Optuna/model selection is frozen.

---

## 8. ONNX Export & Verification

1. Export only the **final trained generator** (`G(x, s) → ŷ`) to ONNX — discriminator is not exported.
2. Handle the style condition as an explicit ONNX input (e.g., an integer/one-hot tensor) so the app backend
   can pass Style 1/2/3 at inference time without needing the training-time embedding lookup table separately.
3. Run the same validation batch through the PyTorch/TF generator and the ONNX Runtime generator; compare
   outputs (max abs diff / cosine similarity per pixel or per feature) and document the tolerance achieved.
4. Store the `.onnx` file via the documented download link / Git LFS path referenced from the README (per the
   "no large files directly in GitHub" rule).

---

## 9. Application Integration — "Face-to-Sketch Generator" Workspace

**Frontend (React + Tailwind, styled per the Google Stitch design produced for this workspace):**
- Image input control: file upload **and** webcam capture (e.g., `getUserMedia` + capture-to-file).
- Style selector: 3 fixed options — Style 1 / Style 2 / Style 3 (buttons or radio group, not free text).
- "Generate" action → calls backend inference endpoint.
- Result view: original photo and generated sketch **side by side**, plus reported inference time.
- "Download" control for the generated sketch.
- Loading/error states for invalid uploads, oversized files, backend errors.

**Backend (FastAPI):**
- `POST /api/face-to-sketch` — multipart photo + `style` field (1/2/3) → validates file type/size, preprocesses
  (RGB, resize/crop to 128×128 consistent with training-time preprocessing), runs ONNX Runtime inference,
  returns the generated sketch (base64 or binary) + inference time in ms + model version/tag.
- `GET /api/health` — shared health-check endpoint across all four workspaces.
- Shared model-loading layer that loads the ONNX generator once at startup (not per-request).

---

## 10. Docker & Deployment

- Backend Dockerfile: Python base image, install `onnxruntime`, `fastapi`, `uvicorn`, copy backend code + the
  exported `.onnx` generator (or a startup step that downloads it per the README-documented link).
- Frontend Dockerfile: Node build stage → static assets served (e.g., via nginx) or served by a small Node
  server; Tailwind build included.
- `docker-compose.yml`: wires frontend + backend (+ the other three tasks' backends/services as applicable)
  behind one `docker compose up` command; exposes the app on a documented local port.
- Verify: a fresh clone + documented model-download step + `docker compose up` reaches a working
  Face-to-Sketch workspace with no manual Python/VS Code steps.

---

## 11. Repository & Documentation Deliverables (Task 4 slice)

```
repo/
├── src/task4_face2sketch/
│   ├── data/
│   │   ├── download_fs2k.py
│   │   ├── dataset.py            # FS2KPairedDataset, stratified 85/15 split, seed 42
│   │   └── augmentations.py      # paired-safe spatial transforms
│   ├── models/
│   │   ├── style_embedding.py
│   │   ├── generator_unet.py
│   │   └── discriminator_patchgan.py
│   ├── losses.py                 # adversarial (BCE-logits) + L1
│   ├── train.py                  # full training loop + MLflow/W&B logging
│   ├── optuna_search.py          # HPO study (reduced-epoch trials)
│   ├── evaluate.py               # metrics + qualitative grids + error maps
│   ├── export_onnx.py            # generator export + parity check
│   └── configs/task4.yaml
├── backend/app/routers/face_to_sketch.py
├── frontend/src/workspaces/FaceToSketchGenerator/
├── docker/ (Dockerfiles referenced by docker-compose.yml)
└── README.md  # exact reproduction commands for Task 4 end-to-end
```
- README: dataset download/build → training → Optuna study → full retrain → evaluation → ONNX export →
  app startup, as copy-pasteable commands.
- No full FS2K copy or large checkpoints committed — documented download link or Git LFS entry instead.

---

## 12. Technical Report — Task 4 Section Outline

1. Task introduction & relation to paired image-to-image translation (pix2pix-style conditional GANs).
2. Related work: conditional GANs, PatchGAN discriminators, style-conditioning strategies (embeddings vs
   FiLM vs AdaIN), sketch-generation-specific literature.
3. Dataset preparation: FS2K description, split methodology (85/15, seed 42, stratified), preprocessing.
4. Architecture diagrams: generator (U-Net + conditioning point), discriminator (PatchGAN + conditioning point).
5. Loss functions and final objective, with the reconstructed formulas from §4 above.
6. Training procedure, including the warm-up-free straightforward adversarial loop and sample-logging cadence.
7. Optuna: search space table, trial count, best trial, final configuration — reference §6.
8. Results: quantitative table(s) per style (§7), training/validation curves (4 separate loss series + val
   metric), generated-image grids across training checkpoints, error maps, failure-case discussion.
9. Application design: Stitch mockup evidence, screenshots of the working workspace, ONNX-parity note.
10. Limitations & conclusion for this task.

---

## 13. Demonstration Video — Task 4 Portion

Within the shared 5–7 minute video, the Task 4 segment must show:
- Uploading a photo (and/or webcam capture).
- Selecting each style and generating a sketch.
- Downloading the generated result.
- A glimpse of the MLflow/W&B experiment-tracking dashboard showing Task 4 runs (losses, sample grids).

---

## 14. AI-Use Appendix Notes (Task 4)

Log, as work proceeds: which AI tools were used (e.g., for researching pix2pix variants, drafting boilerplate
training/export code, debugging shape mismatches, writing docs), what each was used for, and how the output
was verified (unit tests, manual inspection of generated grids, ONNX parity check, etc.) — required per the
assignment's AI-use appendix rule.

---

## 15. Research Questions Requiring Evidence-Backed Decisions

These are the "must investigate, don't just accept a default" items specific to Task 4. Each links back to the
sourced reading in §0 that should inform the answer — the report needs the *decision*, the *alternative(s)
considered*, and *why*, not just the final choice.
- [ ] **Q1 — Style-embedding fusion** (concat vs FiLM vs projection-discriminator vs bottleneck-only): see §0.2 B1
      (Perez et al. 2018; Miyato & Koyama 2018). Justify with an ablation showing style actually changes output.
- [ ] **Q2 — PatchGAN receptive-field size** and its effect on sketch texture/sharpness: see §0.2 B2 (pix2pix
      supplementary ablation).
- [ ] **Q3 — Whether to add a perceptual/SSIM term** alongside L1, and why: see §0.1 A1 (pix2pix ablations) as
      the starting reference, extend empirically if you deviate.
- [ ] **Q4 — Final λ_L1 from Optuna** vs the pix2pix-convention default of 100 — explain the delta if any (§6,
      §0.1 A1).
- [ ] **Q5 — Quantitative metric(s)** for sketch quality (L1/SSIM/LPIPS/FID) given only 2,104 total pairs: see
      §0.3 C2 (Zhang et al. 2018; Heusel et al. 2017) — including whether FID is even trustworthy at this scale.
- [ ] **Q6 — GAN-stability techniques** worth adopting given the small dataset (augmentation strength, spectral
      normalization, label smoothing, `n_critic` ratio): see §0.3 C1 (Zhao et al. 2020; Miyato et al. 2018) —
      evaluate empirically, don't assume.
- [ ] **Q7 — Optuna's validation objective** definition for this task (single metric vs weighted combination),
      and the pruning signal used: see §0.3 C3.

---

## 16. Suggested Execution Phases (dependency-ordered, not calendar-dated)

1. **Data**: download FS2K, build pairing dataset, stratified split, paired augmentation, visual sanity check.
2. **Baseline model**: implement U-Net generator + PatchGAN discriminator + style embedding, get a training
   loop running end-to-end on a small subset (overfit test) before trusting it on full data.
3. **Optuna HPO**: reduced-epoch trials across §6's search space; select best config.
4. **Full training**: retrain best config for the full schedule with complete MLflow/W&B logging.
5. **Evaluation**: compute chosen metrics per style, build qualitative grids, error maps, identify failures.
6. **Export**: ONNX export + parity verification.
7. **App integration**: backend endpoint, frontend workspace (post-Stitch design), Docker wiring.
8. **Reporting**: write Task 4 report section, capture screenshots, record the Task 4 video segment.
9. **Final QA**: fresh-clone + `docker compose up` test; run on genuinely unseen photos to rehearse the live
   evaluation scenario.

---

## 17. Risks & Open Assumptions

- FS2K is small (2,104 pairs) for adversarial training — expect to lean on augmentation and possibly reduced
  model capacity; document instability if/when it appears.
- Sketch channel format (grayscale vs RGB) in FS2K needs to be confirmed during the data audit before locking
  the generator's output channel count.
- The exact validation objective for Optuna is undefined by the spec and must be chosen and justified.
- Deadline in the source document (March 16, 2024) is stale — confirm the real date before scheduling.
