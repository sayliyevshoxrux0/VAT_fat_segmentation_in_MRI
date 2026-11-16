import torch
from torch.utils.data import DataLoader
from dataset import NoisyMRIDataset

# 1. Define paths for validation and test images
image_valid = [f"data/images/PATIENT_{i}.tif" for i in range(49, 59)]
image_test = [f"data/images/PATIENT_{i}.tif" for i in range(59, 69)]

# 2. Point EXPLICITLY to the EXPERT manual masks
mask_valid = [f"data/expert/MANUAL_VAT_PATIENT_{i}.tif" for i in range(49, 59)]
mask_test = [f"data/expert/MANUAL_VAT_PATIENT_{i}.tif" for i in range(59, 69)]

# 3. Create Dataset instances (Severity is 0 for eval)
validation = NoisyMRIDataset(image_valid, mask_valid, severity=0) 
test = NoisyMRIDataset(image_test, mask_test, severity=0)       

# 4. Create the DataLoaders that will be imported
valid_loader = DataLoader(validation, batch_size=4, shuffle=False)
test_loader = DataLoader(test, batch_size=4, shuffle=False)