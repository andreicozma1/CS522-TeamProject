import numpy as np


def wta_classify(images: np.array, k: int, lp: float = 0.001) -> (np.array, np.array):
    shape = images.shape
    images_flattened = np.reshape(np.copy(images), (shape[0], -1))

    y = np.random.randint(0, k, shape[0])
    centers = np.array([np.mean(images_flattened[y == i], axis=0)
                        for i in range(k)])
    for image in images_flattened:
        closer = np.argmin([np.linalg.norm(image - center)
                           for center in centers])
        centers[closer] += lp * (image - centers[closer])

    diff = np.array(
        [np.linalg.norm(images_flattened - center, axis=1) for center in centers])
    y = np.argmin(diff, axis=0)

    return y, centers
