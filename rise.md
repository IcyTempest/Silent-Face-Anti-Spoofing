# Face anti-spoofing — project handoff

Read this first. `claude/CDCN-port-notes.md` has the low-level CDCN implementation
detail; this document is the decision record and the state of play.

---

## The actual problem

A shipping mobile app does face liveness **on-device**, front camera, across a wide
device range — flagships down to cheap Android handsets with poor cameras.

Current production model: **MiniFASNetV2**, a third-party 128×128 retrain (not
minivision's original 80×80). It fails on **high-PPI OLED screen replay** — a Pixel 9
or S23 displaying a face, filmed clean and edgeless, reads as live.

The workaround in production is a decision threshold cranked to **0.99**, which
catches those attacks at the cost of rejecting genuine users. That is the problem to
solve: not "detect spoofs" but "detect clean high-PPI replays without wrecking the
false-reject rate on bad cameras."

---

## Decision: do NOT deploy CDCN. Fine-tune MiniFASNetV2 instead.

CDCN (CVPR'20 Central Difference Convolutional Network) was investigated in depth.
It is **not viable on-device**. Measured from the layer definitions:

| model | params | GMACs/image |
|---|---|---|
| MiniFASNetV2 (approx) | 0.4 M | **0.05** |
| MobileNetV3-Small | 2.5 M | 0.06 |
| MobileNetV2 @224 | 3.4 M | 0.30 |
| ResNet-50 @224 | 25.6 M | 4.10 |
| CDCN @256 | 2.2 M | **52.60** |
| DC-CDN (IJCAI'21) | 4.5 M | **105.19** |

CDCN is ~**1000×** the compute of MiniFASNetV2. DC-CDN, despite the paper's
"five ninths the parameters" claim, is **twice as expensive as CDCN** — that claim
is about kernel sparsity inside each C-CDC op, while the architecture runs two full
backbones in parallel. Dead end for mobile.

Note the params column: CDCN is only 2.2M, *smaller* than MobileNetV2. The 9MB
checkpoint was never the problem. The cost is architectural — CDCN holds 128–196
channels at full 256×256 through Block1, which alone is **72.7%** of total compute.

**The useful insight: CDC is an operator, not an architecture.** `Conv2d_cd` is a
drop-in for `nn.Conv2d` (a 3×3 conv plus a 1×1 built from the summed kernel, then a
subtraction). Dropping it into a mobile backbone costs ~**11%** over a plain conv.

### Recommended order of work

1. **Fine-tune MiniFASNetV2 on our own data.** One variable changed (the data), model
   already ships, latency/export/quantization already known. Directly shippable.
2. If insufficient: **swap `nn.Conv2d` → `Conv2d_cd` inside MiniFASNetV2**, same
   backbone, same export path, ~11% more compute. This tests whether the central
   difference operator actually helps *our* attack, in a deployable form.
3. MobileNetV3 only if both fail. It is a lateral move — same cost class, no reason
   to expect a generic ImageNet backbone to beat a purpose-built FAS network with FAS
   pretraining.

Change one thing at a time. Data and architecture together = uninterpretable result.

---

## Final CDCN evaluation — the evidence for abandoning it

Scored after fine-tuning (CelebA-Spoof pretrain + 15% own live-only frames), on held-out
images including **new subjects not in training**. Sorted by score:

```
0.0050  SPOOF  (early set)
0.0126  SPOOF  (early set)
0.0696  SPOOF  (early set)
0.2889  LIVE
0.3055  LIVE
0.3157  LIVE
0.3232  SPOOF  (new)   <-- ranks above three live faces
0.3554  LIVE
0.3878  SPOOF  (new)
0.3920  SPOOF  (new)
```

**No threshold separates these.** AUC = 13/24 = **0.54** (0.50 is a coin flip). At 0.32
you reject three of four genuine users and still pass a replay; below 0.29 half the
spoofs walk through.

Two details make this decisive rather than just disappointing:

- **The new spoofs had clearly visible pixel grid and moiré**, filmed at various angles —
  and still scored mid-range. The model is not using screen texture at all.
- **New live faces (0.29–0.36) and new spoofs (0.32–0.39) occupy the same band.** The
  score tracks *unfamiliarity*, not class. Anything outside the training distribution
  gets ~0.3 regardless of what it is. The early spoofs score low because they resemble
  CelebA's spoofs, not because the model understands screens.

The earlier promising result — a clean gap between spoof ≤0.07 and live ≥0.12 — did not
survive contact with new subjects. Live-only fine-tuning shifted the distribution
without creating separation.

Caveat: n=10. But three spoofs ranking above three live faces is structural overlap,
not sampling noise.

**Keep these ten images as a hard test set.** Especially the three new visible-grid
spoofs. Run MiniFASNetV2 against them before and after fine-tuning — if the current
production model already separates them, that is itself an important data point.

---

## The data asset (the real output of this work)

`~/codes/python/CDCN/real_world_data_crop/subNN/{live,spoof}/*.png`

- **749 live / 543 spoof**, 256×256 crops at **1.4× the face box**, BGR on disk
- 30 subject directories (`sub0`–`sub29`); **19 have both classes**, 11 are live-only
- Front-camera selfie video, ~20s per clip, frames extracted at 1 fps
- Covers bright and dark conditions, varied phones (participants used their own)
- Spoof = replays of the participants' own live clips shown on screens

Upstream stages, all name-sanitised:
```
real_world_data_raw/        original videos
real_world_data_filter/     subNN/{live,spoof}/  <- videos sorted by class
real_world_data_processed/  subNN/{live,spoof}/  <- 1 fps JPEG frames
real_world_data_crop/       subNN/{live,spoof}/  <- 256×256 face crops (USE THIS)
```

Prep script: `one_process_data.py`. Videos → frames (ffmpeg, honours iPhone .MOV
rotation) → RetinaFace crop → `sanitize_tree()` pass.

### Known gaps in the data

- Only ~2 display types used for replays; too narrow to prove generalisation
- 11 subjects have no spoof pass
- No deliberate degraded-optics set (cracked protector, smudged lens)
- Some early test images were shot on the **back** camera — invalid, production is front

---

## Findings that cost real time

**CelebA-Spoof teaches the wrong boundary.** It is built on CelebA = web-scraped
celebrity photos. Inspected samples: the *live* class is professional press and concert
photography (DSLR, sharp, stage lighting); the *spoof* class is someone holding a phone
in front of a houseplant. The learned boundary is **"professional photograph vs amateur
phone snapshot"**, not "real face vs screen". Our phone-video frames sit on the spoof
side of it.

**Moiré may not exist in the attack.** On a MacBook Retina (~227 PPI) filmed close with
a phone front camera, the pixel grid is not resolvable even zoomed to max — the display
pitch falls below the camera's sampling resolution, so there is no aliasing and no
moiré. Any approach relying on high-frequency screen texture has nothing to detect on
high-PPI panels at close range. **Flatness (geometry) is the cue that survives**, which
is display-independent: a screen is flat at any PPI. (Note: the final evaluation showed
the model failing even where moiré *was* visible, so texture was never being used.)

**CDCN without depth supervision is not CDCN.** Its distinguishing property is
regressing a real 3D depth map for live faces and zeros for flat spoofs. CelebA-Spoof
ships no depth ground truth, so our loader substituted all-ones/all-zeros maps. That
deletes the geometry signal entirely and makes the contrastive-depth loss idle against
a uniform target. What remained was a texture classifier with a 32×32 output. If CDC is
ever revisited, generate pseudo-depth for live frames first (Depth Anything V2,
`depth-anything/Depth-Anything-V2-Small-hf`), spoofs stay zero, and apply the same
horizontal flip to the map as to the image.

**Architecture never rescues a mismatched training distribution.** MiniFASNetV2 beats
a CelebA-trained CDCN on our data purely because minivision collected purpose-built
phone-liveness data. This was the central lesson of the whole exercise.

---

## Measurement methodology (reusable, independent of architecture)

- **APCER** = attacks wrongly accepted as live → security failure
- **BPCER** = real faces wrongly rejected → usability failure
- **ACER** = (APCER + BPCER)/2

**Never read ACER alone.** 4% ACER can be 1%/7% or 7%/1% — completely different
products. Production biometrics fixes APCER at a policy level (e.g. ≤1%) and reports
BPCER at that operating point. A model with 4% ACER may have 25% BPCER at APCER 1%,
which is unshippable.

- **Split by identity, never by frame.** ~20 frames come from each 20s clip and are
  near-identical; frame-level splits leak and produce meaningless validation.
- **Derive the threshold from a val split of our own data** (equal-error point), never
  from a public benchmark's. Domains shift the whole distribution.
- **Hold out an unseen display type as well as unseen faces** — that is what predicts
  production.
- **Always test on subjects not in training.** The CDCN result looked promising until
  new identities were introduced, at which point separation vanished entirely.
- Select checkpoints on **own-val ACER**, not test ACER (test leakage).

Loss diagnostics for depth-map style training: with constant target maps the absolute
loss floor at "learned only the class prior" is `p·(1−p)` — about 0.22 at 34% live.
Plateau there = nothing learned.

---

## Environment gotchas

- **`RetinaFace.extract_faces()` returns RGB, not BGR.** Calling
  `cv2.cvtColor(..., COLOR_BGR2RGB)` on it swaps a second time and feeds the model BGR.
  Symptom: every image scores low.
- **`extract_faces()` is not reliably ordered by confidence.** A small background face
  can win. Use `detect_faces()` and pick the **largest-area** box.
- **`expand_face_area` is a percentage, not a multiplier.** `13` = 1.13×, not 1.3×.
- Kaggle rejects archive members containing `'`, `(`, `)`. `sanitize_tree()` in
  `one_process_data.py` handles this; it is idempotent and collision-safe.
- Kaggle free tier: P100 or 2×T4, 30 h/week, 12 h/session. Worth it only with AMP.
- AMP (`autocast`/`GradScaler`) is CUDA-only in practice; gate on `device.type == 'cuda'`
  or it errors/warns on MPS.
- Training CDCN was GPU-bound on Apple Silicon (~10 img/s) while the dataloader managed
  70.7 img/s — do not optimise the loader for that class of model.

---

## Immediate next steps

**Before writing any training code, extract five facts from the production app:**

1. Shipped artifact format — `.pth`, `.onnx`, or `.tflite`?
2. Input size (believed 128×128, unconfirmed)
3. **Crop scale** — what multiple of the face box does inference cut? Our crops are
   1.4×; minivision's original pipeline uses ~2.7× and ~4.0×. If they differ, the 1,292
   images need a re-crop pass from `real_world_data_processed` (no re-shoot needed).
4. Normalization (`/255`, `(x−127.5)/128`, or ImageNet mean/std)
5. Single model, or a fusion of two crop scales?

If only ONNX/TFLite exists, fine-tuning is impractical — find the repo whose
architecture matches those weights and load them into PyTorch. Match **the variant we
ship**, not minivision's original.

Then: fine-tune on our data, calibrate the threshold on a held-out identity split, and
report APCER/BPCER separately against the current production model as baseline. Run the
ten-image hard test set above as a first sanity check before anything else.

---

## Licensing

The CDCN repo README states *"It is just for research purpose, and commercial use is
not allowed."* CelebA-Spoof carries its own non-commercial terms. Neither matters if
we ship MiniFASNet fine-tuned on our own data, but clear it before any CDCN-derived
code reaches production.
