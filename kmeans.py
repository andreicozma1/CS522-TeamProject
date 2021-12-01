import numpy as np


def kmeans_image_compress(image: np.array, k: int, max_error: float = 0.0) -> (np.array, np.array):
    pixels = np.reshape(np.copy(image), (-1, 3))
    means = pixels[np.random.randint(0, len(pixels), k), :]
    y = np.zeros(len(pixels))
    while True:
        diff = np.array([np.linalg.norm(pixels - mean, axis=1)
                        for mean in means])
        y_new = np.argmin(diff, axis=0)

        count = np.count_nonzero(y != y_new)
        y = y_new

        means = np.array([np.mean(pixels[y == i], axis=0) for i in range(k)])

        if count / len(y) <= max_error:
            break

    return y, means


def kmeans_image_colorize(image: np.array, means: np.array) -> np.array:
    new_image = np.zeros((image.shape[0], 3))
    for i, v in enumerate(means):
        new_image[image == i] = v

    return new_image


def kmeans_classify(images: np.array, k: int, max_error: float = 0.0) -> np.array:
    shape = images.shape
    images_flattened = np.reshape(np.copy(images), (shape[0], -1))

    y = np.random.randint(0, k, shape[0])
    means = np.array([np.mean(images_flattened[y == i], axis=0)
                     for i in range(k)])
    while True:
        diff = np.array(
            [np.linalg.norm(images_flattened - mean, axis=1) for mean in means])
        y_new = np.argmin(diff, axis=0)

        count = np.count_nonzero(y != y_new)
        y = y_new

        means = np.array([np.mean(images_flattened[y == i], axis=0)
                         for i in range(k)])

        if count / len(y) <= max_error:
            break

    return y, means


def get_ideal(images: np.array, y: np.array) -> (np.array, np.array):
    means = np.array([np.mean(images[y == i], axis=0) for i in range(2)])
    diff = np.array(
        [np.linalg.norm(images - mean, axis=1) for mean in means])
    y_new = np.argmin(diff, axis=0)

    return y_new, means
