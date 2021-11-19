import os
from typing import Tuple
from PIL import Image, ImageOps
import PIL
import numpy as np
import pickle


def get_dataset_fn(dataset_dir: str, type: str) -> Tuple[np.array, np.array]:
    """
    given the directory name of the entire dataset and whether it is the train, validation, or test set
    return X and y where X is a list of filenames and y is the target labels
    """
    # get base dir path (account for different operating systems)
    base_dir = os.path.join(dataset_dir, type)
    # get directories for each class in each folder
    no_mask_dir = os.path.join(base_dir, "WithoutMask")
    mask_dir = os.path.join(base_dir, "WithMask")
    # get filenames
    no_mask_fn = list(map(lambda fn: os.path.join(
        no_mask_dir, fn), os.listdir(no_mask_dir)))
    mask_fn = list(map(lambda fn: os.path.join(
        mask_dir, fn), os.listdir(mask_dir)))
    # create X with associated y labels
    y = np.hstack((np.zeros(len(no_mask_fn)), np.ones(len(mask_fn))))
    X = np.hstack((np.array(no_mask_fn), np.array(mask_fn)))
    return X, y


def load_img(fn: str) -> np.array:
    """
    load an image into a numpy array from the given filename
    # converts to grayscale
    """
    return np.array(Image.open(fn))


def load_dataset(X_fn: np.array) -> list:
    """
    given a list of filenames, load in each image into a numpy array and return a list of the images
    """
    X = []
    # iterate through each filename and load in each image
    for i in range(len(X_fn)):
        X.append(load_img(X_fn[i]))

    # returns X as a list because images are different sizes
    return X


def to_categorical(y) -> np.array:
    """
    One hot encode the target labels
    """
    Y = np.zeros((len(y), len(np.unique(y))))
    for i in range(len(y)):
        Y[i, int(y[i])] = 1
    return Y

# helper functions to resize images


def resize_img(x: np.array, size: tuple):
    return np.asarray(Image.fromarray(x).resize(size, resample=PIL.Image.LANCZOS)).astype(np.uint8)


def resize_dataset(X, size: tuple):
    X_new = np.zeros((len(X), *size))
    size = size[:-1] if len(size) == 3 else size
    for i, x in enumerate(X):
        X_new[i] = resize_img(x, size)
    return X_new

# helper functions to turn images to grayscale


def img2grayscale(img: np.array):
    return np.asarray(ImageOps.grayscale(Image.fromarray(img.astype(np.uint8), 'RGB'))).astype(np.uint8)


def dataset2grayscale(X):
    X_new = []
    for x in X:
        X_new.append(img2grayscale(x))
    return X_new

# min-max scale images


def min_max_scale_img(x: np.array):
    x = x.astype("float64")
    x /= 255
    return x


def min_max_scale_dataset(X):
    X_new = []
    for x in X:
        X_new.append(min_max_scale_img(x))
    return X_new


def get_conv_dataset(dataset, fn):
    if os.path.exists(fn):
        with open(fn, 'rb') as f:
            conv = pickle.load(f)
    else:
        conv = np.array([model.feedforward(x) for x in dataset])
        with open(fn, 'wb') as f:
            pickle.dump(conv, f)
    return conv
