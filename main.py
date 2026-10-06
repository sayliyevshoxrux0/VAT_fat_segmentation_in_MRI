import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import MRIDataset
from unet import UNet
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

device = torch.device("cuda")

# 1. Dataset

image_paths = [
    f"data/images/PATIENT_{i}.jpg" if i in (5, 6, 7) else f"data/images/PATIENT_{i}.tif"
    for i in range(1, 49)
]

mask_paths = [
    f"data/masks/OTSU_VAT_PATIENT_ID_{i}.tif" if i not in (5, 6, 7) else f"data/masks/OTSU_VAT_PATIENT_ID_{i}.jpg"
    for i in range(1, 49)
]

image_valid = [
    f"data/images/PATIENT_{i}.tif"
    for i in range(49, 59)
]

mask_valid = [
    f"data/masks/OTSU_VAT_PATIENT_ID_{i}.tif"
    for i in range(49, 59)
]

image_test = [
    f"data/images/PATIENT_{i}.tif"
    for i in range(59, 69)
]

mask_test = [
    f"data/masks/OTSU_VAT_PATIENT_ID_{i}.tif"
    for i in range(59, 69)
]

cardiac = [
    f"data/cardiac/cardiac-mri-4-chamber-cine-image-{i}.webp"
    for i in range(1, 8)
]


# 1.1 Data Augmentation

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    #A.GaussNoise(var_limit=(5.0, 20.0), p=0.2) on 100 epoch, approach near best result but still less than best and might be overfitting
])


dataset = MRIDataset(image_paths, mask_paths, transform=train_transform)
validation = MRIDataset(image_valid, mask_valid)
test = MRIDataset(image_test, mask_test)

cardiac = MRIDataset(cardiac, mask_test[0:8])


train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

valid_loader = DataLoader(
    validation,
    batch_size=4,
    shuffle=False
)

test_loader = DataLoader(
    test,
    batch_size=4,
    shuffle=False
)


# 2. Model

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
).to(device)


# 3. Loss function

def dice_loss(prediction, target):

    prediction = torch.sigmoid(prediction)

    smooth = 1e-6

    intersection = (prediction * target).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum() + target.sum() + smooth
    )

    return 1 - dice
def dice_score(prediction, target):

    # Convert logits → probabilities
    prediction = torch.sigmoid(prediction)

    # Convert probabilities → 0/1 mask
    prediction = prediction > 0.5

    prediction = prediction.float()
    target = target.float()

    smooth = 1e-6

    intersection = (prediction * target).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum() + target.sum() + smooth
    )

    return dice.item()

bce = nn.BCEWithLogitsLoss()


# 4. Optimizer


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# TRAIN


best_dice = 0.0
epochs = 80

for epoch in range(epochs):

    model.train()

    train_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        predictions = model(images)

        bce_loss = bce(predictions, masks)
        d_loss = dice_loss(predictions, masks)

        loss = bce_loss + d_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)



    # VALIDATION


    model.eval()

    valid_loss = 0
    total_dice = 0

    with torch.no_grad():

        for images, masks in valid_loader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            bce_loss = bce(predictions, masks)
            d_loss = dice_loss(predictions, masks)

            loss = bce_loss + d_loss

            valid_loss += loss.item()

            # Dice for this batch
            dice = dice_score(predictions, masks)
            total_dice += dice


    valid_loss /= len(valid_loader)
    valid_dice = total_dice / len(valid_loader)

    if valid_dice > best_dice:
        best_dice = valid_dice

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(" New best model found")

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Valid Loss: {valid_loss:.4f} | "
        f"Valid Dice: {valid_dice:.4f}"
    )



# TEST

 
model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()

test_dices = []

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        predictions = model(images)

        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()

        for i in range(images.shape[0]):

            pred = predictions[i]
            target = masks[i]

            intersection = (pred * target).sum()

            dice = (
                2 * intersection + 1e-6
            ) / (
                pred.sum() + target.sum() + 1e-6
            )

            test_dices.append(dice.item())


print("Individual test Dice scores:")

for i, dice in enumerate(test_dices):
    print(f"Patient {59 + i}: {dice:.4f}")

print(
    "Mean Test Dice:",
    sum(test_dices) / len(test_dices)
)

torch.save(
    model.state_dict(),
    "best_model.pth"
)




# VISUALISE


# Take one image from the dataset
# image, mask = test[0]
import numpy as np

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image_paths = [
    f"data/images/PATIENT_{i}.jpg"
    for i in range(1, 8)
]

images = [
    Image.open(path).convert('L')   
    for path in image_paths
]

for i in range(len(images)):
    images[i] = transform(images[i]).unsqueeze(0).to(device)
  




# Turn off gradient calculation
with torch.no_grad():
    prediction = [
        model(image)
        for image in images
    ]

# Convert model output to probability
prediction = torch.sigmoid(prediction)

# Remove batch/channel dimensions
prediction = prediction[0, 0].cpu().numpy()
image = image[0, 0].cpu().numpy()
# mask = mask[0]

# Turn probability into binary mask
prediction_binary = prediction > 0.5

# Plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("original")
plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(mask, cmap="gray")
# plt.title("Otsu mask")
# plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image, cmap="gray")
plt.imshow(prediction_binary, cmap='Reds', alpha = 0.4)
plt.title("prediction")
plt.axis("off")

plt.show()



