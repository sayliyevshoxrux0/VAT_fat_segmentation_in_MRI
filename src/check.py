"""
Quantitative Grad-CAM analysis: where does each encoder's heat fall?

For every (architecture, seed, severity, test patient) it computes Grad-CAM on
the last encoder stage (same setup as gradcam_compare.py), then splits the heat
into three mutually exclusive regions:
    VAT         = expert fat mask
    other torso = inside the body but not VAT
    background  = outside the body
and reports the fraction of total heat in each, plus an "enrichment" ratio
(heat fraction / area fraction). Enrichment 1.0 = heat spread evenly over the
image, >1 = the model concentrates heat there more than chance.

Aggregation matches the rest of your results: mean over the 10 patients within
a seed, then mean +/- sample std (n-1) across the 3 seeds.

Run as a file (python src/check.py), and open figures/torso_check.png first to
confirm the body masks look right.
"""

import csv
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from models import build_unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCHS = ["resnet34", "mit_b0", "efficientnet-b0", "tu-mobilevit_s"]
SEEDS = [0, 1, 2]
SEVERITIES = [0, 4]
PATIENTS = list(range(59, 69))


def torso_mask(img):
    """Approximate body mask: Otsu threshold, close gaps, fill the largest outer contour."""
    u8 = (img * 255).astype(np.uint8)
    _, bw = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bw)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=-1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))  # trim thin bridges (arm contact, streaks)
    return mask > 0


def load_model(arch, seed, severity):
    model = build_unet(arch).to(device)
    path = f"weights/best_model_{arch}_seed_{seed}_severity_{severity}.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def get_target_layer(model, arch):
    enc = model.encoder
    if arch == "resnet34":
        return enc.layer4[-1]
    if arch == "efficientnet-b0":
        return enc._blocks[-1]
    if arch == "mit_b0":
        return enc.block4[-1]
    if arch == "tu-mobilevit_s":
        return enc.model.stages_4
    raise ValueError(arch)


# ---- load images / expert masks / body masks once ----
data = []
for p in PATIENTS:
    img = np.array(Image.open(f"data/images/PATIENT_{p}.tif")).astype(np.float32) / 255.0
    expert = np.array(Image.open(f"data/expert/MANUAL_VAT_PATIENT_{p}.tif")).astype(np.float32) / 255.0
    vat = expert > 0.5
    torso = torso_mask(img) | vat          # make sure VAT is always inside the torso
    data.append(dict(
        patient=p, img=img, vat=vat, torso=torso,
        other=torso & ~vat, bg=~torso,
        x=torch.tensor(img)[None, None].to(device),
    ))

# visual check of the body masks -- look at this before trusting any numbers
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, d in zip(axes.ravel(), data):
    ax.imshow(d["img"], cmap="gray")
    ax.contour(d["torso"].astype(float), levels=[0.5], colors="red", linewidths=0.8)
    ax.set_title(f"patient {d['patient']}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("figures/torso_check.png", dpi=100)
plt.close()

chance_vat = np.mean([d["vat"].mean() for d in data])
chance_torso = np.mean([d["torso"].mean() for d in data])
print(f"Chance level (area fractions): VAT {chance_vat:.3f} | torso {chance_torso:.3f} | background {1 - chance_torso:.3f}\n")

# ---- main loop ----
rows = []
for arch in ARCHS:
    for seed in SEEDS:
        for severity in SEVERITIES:
            model = load_model(arch, seed, severity)
            cam = GradCAM(model=model, target_layers=[get_target_layer(model, arch)])

            for d in data:
                with torch.no_grad():
                    pred = (torch.sigmoid(model(d["x"])) > 0.5)[0, 0].cpu().numpy().astype(np.float32)

                row = dict(arch=arch, seed=seed, severity=severity, patient=d["patient"],
                           f_vat=np.nan, f_torso=np.nan, f_bg=np.nan, enrich_vat=np.nan, enrich_torso=np.nan)

                if pred.sum() > 0:  # empty prediction -> empty target -> no heatmap
                    heat = cam(input_tensor=d["x"], targets=[SemanticSegmentationTarget(0, pred)])[0]
                    total = heat.sum()
                    if total > 1e-8:
                        f_vat = heat[d["vat"]].sum() / total
                        f_oth = heat[d["other"]].sum() / total
                        row.update(
                            f_vat=f_vat,
                            f_torso=f_vat + f_oth,
                            f_bg=heat[d["bg"]].sum() / total,
                            enrich_vat=f_vat / d["vat"].mean(),
                            enrich_torso=(f_vat + f_oth) / d["torso"].mean(),
                        )
                rows.append(row)

            cam.activations_and_grads.release()
            print(f"done: {arch}, seed {seed}, severity {severity}")

with open("results/gradcam_quant_rows.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def summarize(arch, severity, key):
    per_seed = []
    for seed in SEEDS:
        vals = [r[key] for r in rows if r["arch"] == arch and r["severity"] == severity and r["seed"] == seed]
        per_seed.append(np.nanmean(vals))
    return np.mean(per_seed), np.std(per_seed, ddof=1)


keys = ["f_vat", "f_torso", "f_bg", "enrich_vat", "enrich_torso"]
print("\n=== Grad-CAM heat distribution (mean +/- std across 3 seeds) ===")
print(f"{'arch':>16} {'sev':>3} | " + " | ".join(f"{k:>15}" for k in keys))
for arch in ARCHS:
    for severity in SEVERITIES:
        cells = []
        for k in keys:
            m, s = summarize(arch, severity, k)
            cells.append(f"{m:6.3f} +/- {s:5.3f}")
        print(f"{arch:>16} {severity:>3} | " + " | ".join(cells))

n_skipped = sum(1 for r in rows if np.isnan(r["f_vat"]))
print(f"\nSkipped (empty prediction or empty heatmap): {n_skipped} of {len(rows)}")