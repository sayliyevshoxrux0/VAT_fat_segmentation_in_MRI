from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from noise_generator import add_noise

'''
below class is used to augment dataset, transform to necessary format, and add noise
'''
class NoisyMRIDataset(Dataset):

    def __init__(self, image_paths, mask_paths, severity=0, seed_offset=0, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.severity = severity
        self.seed_offset = seed_offset  # different offset per run -> different noise pattern, same image
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]))
        mask = np.array(Image.open(self.mask_paths[index]))

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        if self.severity > 0:
            mask = add_noise(mask, self.severity, seed=self.seed_offset * 1000 + index)
            mask = mask.astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask