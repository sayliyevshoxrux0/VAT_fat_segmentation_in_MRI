import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import NoisyMRIDataset
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
import csv
import os
from expert_eval_loaders import valid_loader, test_loader
from models import build_unet

device = torch.device("cuda")

# 1. Dataset

image_paths = [
    f"data/images/PATIENT_{i}.jpg" if i in (5, 6, 7) else f"data/images/PATIENT_{i}.tif"
    for i in range(1, 49)
]   

mask_paths = [
    f"data/fuzzy_mask/FUZZY_C_MEANS_VAT_PATIENT_{i}.tif" if i not in (5, 6, 7) else f"data/fuzzy_mask/FUZZY_C_MEANS_VAT_PATIENT_{i}.jpg"
    for i in range(1, 49)
]



# 1.1 Data Augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
])


SEEDS = [0, 1, 2]
severities = [0, 4]

results = []  # list of dicts -- one row per (seed, severity)


def dice_loss(prediction, target):
    prediction = torch.sigmoid(prediction)
    smooth = 1e-6
    intersection = (prediction * target).sum()
    dice = (2 * intersection + smooth) / (prediction.sum() + target.sum() + smooth)
    return 1 - dice


def dice_score(prediction, target):
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float()
    target = target.float()
    smooth = 1e-6
    intersection = (prediction * target).sum()
    dice = (2 * intersection + smooth) / (prediction.sum() + target.sum() + smooth)
    return dice.item()


bce = nn.BCEWithLogitsLoss()
epochs = 100

for seed in SEEDS:
    torch.manual_seed(seed)

    for severity in severities:

        dataset = NoisyMRIDataset(
            image_paths, mask_paths,
            severity=severity,
            seed_offset=seed,
            transform=train_transform,
        )

        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = build_unet("mit_b0", encoder_weights="imagenet").to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_dice = 0.0
        checkpoint_path = f"weights/fuzzy_means/best_model_mit_b0_seed_{seed}_severity_{severity}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        for epoch in range(epochs):

            model.train()
            train_loss = 0

            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                predictions = model(images)
                loss = bce(predictions, masks) + dice_loss(predictions, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            valid_loss = 0
            total_dice = 0

            with torch.no_grad():
                for images, masks in valid_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    predictions = model(images)
                    loss = bce(predictions, masks) + dice_loss(predictions, masks)

                    valid_loss += loss.item()
                    total_dice += dice_score(predictions, masks)

            valid_loss /= len(valid_loader)
            valid_dice = total_dice / len(valid_loader)

            if valid_dice > best_dice:
                best_dice = valid_dice
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[seed={seed}, severity={severity}] New best model saved! (Valid Dice: {valid_dice:.4f})")

            print(
                f"[seed={seed}, severity={severity}] Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"Valid Dice: {valid_dice:.4f}"
            )

        # TEST -- runs ONCE, after all epochs finish, against EXPERT labels now
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        test_dices = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                predictions = model(images)
                predictions = torch.sigmoid(predictions)
                predictions = (predictions > 0.5).float()

                for j in range(images.shape[0]):
                    pred = predictions[j]
                    target = masks[j]

                    intersection = (pred * target).sum()
                    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
                    test_dices.append(dice.item())

        mean_test_dice = sum(test_dices) / len(test_dices)
        results.append({"arch": "mit_b0", "seed": seed, "severity": severity, "test_dice": mean_test_dice})

        print(f"[seed={seed}, severity={severity}] Mean Test Dice (vs expert labels): {mean_test_dice:.4f}\n")

print("=== Summary (evaluated against EXPERT labels) ===")
for row in results:
    print(f"seed={row['seed']}, severity={row['severity']}: mean test Dice={row['test_dice']:.4f}")

with open("results/fuzzy_mit_b0.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["arch", "seed", "severity", "test_dice"])
    writer.writeheader()
    writer.writerows(results)






