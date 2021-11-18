from typing import Dict, Tuple
import numpy as np


def kmeans_image_compress(image: np.array, k: int, max_error: float = 0.0) -> np.array:
    shape = image.shape
    pixels = np.reshape(np.copy(image), (-1, 3))
    means = pixels[np.random.randint(0, len(pixels), k),:]
    y = np.zeros(len(pixels))
    while True:
        diff = np.array([np.linalg.norm(pixels - mean, axis=1) for mean in means])
        y_new = np.argmin(diff, axis=0)

        count = np.count_nonzero(y != y_new)
        y = y_new

        means = np.array([np.mean(pixels[y == i], axis=0) for i in range(k)])

        if count / len(y) <= max_error:
            break

    for i, v in enumerate(means):
        pixels[y == i] = v

    return pixels.reshape(shape)
