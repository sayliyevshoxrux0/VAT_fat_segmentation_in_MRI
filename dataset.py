from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        # Loading MRI
        image = np.array(
            Image.open(self.image_paths[index])
        )

        # Loading Otsu mask
        mask = np.array(
            Image.open(self.mask_paths[index])
        )

        # Normalizing MRI
        image = image.astype(np.float32) / 255.0

        # Converting from 0/255 to 0/1
        mask = mask.astype(np.float32) / 255.0

        # Data Augmentation 
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Converting to PyTorch tensors
        image = torch.tensor(image)
        mask = torch.tensor(mask)

        # Adding dimension
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return image, mask
