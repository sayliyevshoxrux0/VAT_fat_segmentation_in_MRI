import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
import os
from expert_eval_loaders import test_loader # Uses your existing expert loader
from models import build_unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the grid of experiments
architectures = ["resnet34", "efficientnet-b0", "mit_b0", "tu-mobilevit_s"]
seeds = [0, 1, 2]

# Map label sources to their checkpoint directories and severities
experiments = {
    "otsu": {"dir": "otsu_weights", "severities": [0, 2, 3, 4]},
    "kmeans": {"dir": "kmeans_weights", "severities": [0, 4]},
    "fcm": {"dir": "fuzzy_means", "severities": [0, 4]}
}

results = []

def get_checkpoint_path(label_source, arch, seed, severity):
    base_dir = experiments[label_source]["dir"]
    
    # Handle the specific K-means MobileViT naming exception mentioned in the handover doc
    if (label_source == "kmeans" or label_source == "otsu") and arch == "tu-mobilevit_s":
        filename = f"best_model_tu-mobilevit_seed_{seed}_severity_{severity}.pth"
    else:
        filename = f"best_model_{arch}_seed_{seed}_severity_{severity}.pth"
        
    return os.path.join("weights", base_dir, filename)

# Disable gradient calculation for inference
with torch.no_grad():
    for label_source, config in experiments.items():
        for arch in architectures:
            for severity in config["severities"]:
                for seed in seeds:
                    ckpt_path = get_checkpoint_path(label_source, arch, seed, severity)
                    
                    if not os.path.exists(ckpt_path):
                        print(f"Skipping missing checkpoint: {ckpt_path}")
                        continue
                        
                    print(f"Evaluating: {label_source} | {arch} | Sev: {severity} | Seed: {seed}")
                    
                    # Initialize and load model
                    model = build_unet(arch).to(device)
                    
                    model.load_state_dict(torch.load(ckpt_path, map_location=device))
                    model.eval()

                    # Track patient ID (assuming test_loader yields patients 59-68 sequentially)
                    patient_id = 59 
                    
                    for images, masks in test_loader:
                        images = images.to(device)
                        masks = masks.to(device)
                        
                        predictions = model(images)
                        predictions = torch.sigmoid(predictions)
                        predictions = (predictions > 0.5).float()
                        
                        # Calculate Dice per image in the batch
                        for j in range(images.shape[0]):
                            pred = predictions[j]
                            target = masks[j]
                            
                            intersection = (pred * target).sum()
                            dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
                            
                            results.append({
                                "label_source": label_source,
                                "arch": arch,
                                "severity": severity,
                                "seed": seed,
                                "patient_id": patient_id,
                                "test_dice": dice.item()
                            })
                            patient_id += 1

# Export to CSV
csv_filename = os.path.join("results", "per_patient_test_dice.csv")
with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["label_source", "arch", "severity", "seed", "patient_id", "test_dice"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nSuccessfully exported per-patient Dice scores to {csv_filename}")