# -*- coding: utf-8 -*-
"""Score the anti-spoof models IN ISOLATION and as a fusion, side by side.

Unlike test.py (which globs the model folder and always fuses), this scores each
selected model on its own AND the fused combination, then prints one comparison
table so you can see what each crop scale contributes.

Score convention: prob_real = softmax[1] (class 1 = real). A live face should
score high, a spoof low. Fusion = mean of the selected models' prob_real.

    # score every model in the folder, individually + fused
    python test_isolated.py --data real_world_data_processed

    # just the two of interest
    python test_isolated.py --models 2.7_80x80_MiniFASNetV2.pth 4_0_0_80x80_MiniFASNetV1SE.pth

Metrics (all on prob_real):
  AUC     separability, P(live scores > spoof). 0.5 = useless, 1.0 = perfect.
  EER     error rate where APCER == BPCER (threshold-free summary).
  @thr    APCER / BPCER at a fixed decision threshold (default 0.5; try 0.99).
"""
import os
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

MODEL_DIR = "./resources/anti_spoof_models"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def find_images(data_dir):
    """Return (path, label) with label 1=live, 0=spoof. Handles either
    <data>/<subject>/{live,spoof}/ or <data>/{live,spoof}/ layouts."""
    out = []
    for cls, label in (("live", 1), ("spoof", 0)):
        # both nested (subject/live) and flat (live) layouts
        for p in glob.glob(os.path.join(data_dir, "**", cls, "*"), recursive=True):
            if p.lower().endswith(IMG_EXT):
                out.append((p, label))
    return out


def auc(scores, y):
    """Mann-Whitney AUC: P(score[live] > score[spoof]). y: 1=live, 0=spoof."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = int(y.sum())            # live
    n0 = len(y) - n1             # spoof
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_points(scores, y):
    """Sweep every threshold -> (FPR, TPR). y: 1=live(positive), 0=spoof.
    TPR = live correctly accepted; FPR = spoof wrongly accepted as live."""
    order = np.argsort(-scores)          # most real-looking first
    yy = y[order]
    P, N = yy.sum(), (yy == 0).sum()
    tpr = np.concatenate([[0], np.cumsum(yy) / P])
    fpr = np.concatenate([[0], np.cumsum(yy == 0) / N])
    return fpr, tpr


def eer(scores, y):
    """Equal error rate: sweep thresholds, pick the one minimizing |APCER - BPCER|.
    Returns (eer, threshold)."""
    gap, res = 1e9, (1.0, 0.5)
    for t in np.unique(scores):
        accept = scores >= t
        bpcer = float((~accept[y == 1]).mean()) if (y == 1).any() else 0.0  # live rejected
        apcer = float((accept[y == 0]).mean()) if (y == 0).any() else 0.0   # spoof accepted
        if abs(apcer - bpcer) < gap:
            gap, res = abs(apcer - bpcer), ((apcer + bpcer) / 2, float(t))
    return res


def rates_at(scores, y, thr):
    """APCER (spoof accepted) and BPCER (live rejected) at a threshold on prob_real."""
    accept = scores >= thr
    apcer = float(accept[y == 0].mean()) if (y == 0).any() else float("nan")
    bpcer = float((~accept[y == 1]).mean()) if (y == 1).any() else float("nan")
    return apcer, bpcer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="real_world_data_processed")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--models", nargs="*", default=None,
                    help="checkpoint filenames to use (default: all .pth in --model-dir)")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="decision threshold on prob_real for the @thr column (try 0.99)")
    ap.add_argument("--limit", type=int, default=0, help="cap images (0 = all), for a quick run")
    ap.add_argument("--plot", nargs="?", const="roc_models.png", default=None,
                    help="save a ROC curve of every config to this path (default roc_models.png)")
    args = ap.parse_args()

    models = args.models or sorted(f for f in os.listdir(args.model_dir) if f.endswith(".pth"))
    if not models:
        raise SystemExit(f"no .pth models in {args.model_dir}")

    data = find_images(args.data)
    if args.limit:
        rng = np.random.default_rng(0)
        rng.shuffle(data)
        data = data[:args.limit]
    if not data:
        raise SystemExit(f"no images found under {args.data}")
    n_live = sum(l for _, l in data)
    print(f"{len(data)} images | live={n_live} spoof={len(data) - n_live} | models: {models}\n")

    predictor = AntiSpoofPredict(0)
    cropper = CropImage()

    # prob_real per model, per image
    scores = {m: np.zeros(len(data)) for m in models}
    y = np.array([lbl for _, lbl in data], dtype=int)

    for i, (path, _) in enumerate(data):
        frame = cv2.imread(path)
        if frame is None:
            continue
        bbox = predictor.get_bbox(frame)                 # detect once, shared across models
        for m in models:
            h, w, _, scale = parse_model_name(m)
            img = cropper.crop(org_img=frame, bbox=bbox, scale=scale,
                               out_w=w, out_h=h, crop=scale is not None)
            prob = predictor.predict(img, os.path.join(args.model_dir, m))  # (1,3) softmax
            scores[m][i] = float(prob[0][1])             # class 1 = real
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(data)}")

    # add the fusion (mean prob_real) if more than one model
    configs = list(models)
    if len(models) > 1:
        scores["FUSION(mean)"] = np.mean([scores[m] for m in models], axis=0)
        configs.append("FUSION(mean)")

    # report
    print(f"\n{'config':<34s} {'AUC':>7s} {'EER':>7s} {'EERthr':>7s} "
          f"{'APCER@' + str(args.thr):>10s} {'BPCER@' + str(args.thr):>10s}")
    print("-" * 80)
    for c in configs:
        s = scores[c]
        a = auc(s, y)
        e, ethr = eer(s, y)
        ap_, bp = rates_at(s, y, args.thr)
        print(f"{c:<34s} {a:>7.4f} {e * 100:>6.2f}% {ethr:>7.3f} "
              f"{ap_ * 100:>9.2f}% {bp * 100:>9.2f}%")
    print("\nreminder: on confounded data a high AUC means the model separates the "
          "capture rigs, not necessarily live vs spoof.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
        fig, ax = plt.subplots(figsize=(6, 6))
        for c, col in zip(configs, colors):
            fpr, tpr = roc_points(scores[c], y)
            ax.plot(fpr, tpr, lw=2, color=col, label=f"{c}  (AUC={auc(scores[c], y):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)   # coin-flip diagonal
        ax.set_xlabel("spoof wrongly accepted as live  (FPR = APCER)")
        ax.set_ylabel("live correctly accepted  (TPR = 1 - BPCER)")
        ax.set_title("Anti-spoof model ROC")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nROC curve -> {args.plot}")


if __name__ == "__main__":
    main()
