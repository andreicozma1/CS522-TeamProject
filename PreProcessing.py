import PIL
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


class PreProcessing:
    """
    # helper functions to resize images
    """

    @staticmethod
    def resize_image(x:np.array, size: tuple):
        return np.asarray(Image.fromarray(x).resize(size, resample=PIL.Image.LANCZOS)).astype(np.uint8)

    @staticmethod
    def resize_images(X, size:tuple):
        X_new = np.zeros((len(X), *size))
        size = size[:-1] if len(size) == 3 else size
        for i, x in tqdm(enumerate(X)):
            X_new[i] = PreProcessing.resize_image(x, size)
        return X_new

    @staticmethod
    def img2grayscale(img: np.array):
        return np.asarray(ImageOps.grayscale(Image.fromarray(img.astype(np.uint8),'RGB'))).astype(np.uint8)

    @staticmethod
    def dataset2grayscale(X):
        X_new = []
        for x in tqdm(X):
            X_new.append(PreProcessing.img2grayscale(x))
        return X_new

    @staticmethod
    def min_max_scale_img(x:np.array):
        x = x.astype("float64")
        x /= 255
        return x

    @staticmethod
    def min_max_scale_dataset(X):
        # min-max scale images
        X_new = []
        for x in tqdm(X):
            X_new.append(PreProcessing.min_max_scale_img(x))
        return X_new