import cv2
import numpy as np


def add_noise(mask, severity, seed=None):
    rng = np.random.default_rng(seed)

    mask = mask.astype(np.uint8)
    mode = rng.choice(["dilate", "erode"])
    kernel_size = rng.integers(1, severity + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if mode == "dilate":
        return cv2.dilate(mask, kernel, iterations=1)
    return cv2.erode(mask, kernel, iterations=1)
