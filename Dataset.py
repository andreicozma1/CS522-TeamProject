import gzip
import os
import pickle
from copy import deepcopy

import numpy as np
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


class Data:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __str__(self):
        # Print out information about the data
        return "X: " + str(len(self.X)) + " y: " + str(len(self.y))


def format_percentage(num, deno):
    return f"{round((num / deno) * 100, 2)}%"


class Dataset:

    def __init__(self, dataset_dir=None):
        self.dataset_dir = dataset_dir
        self.train = None
        self.validation = None
        self.test = None
        self.transforms = []
        print("# Dataset Init:", self.dataset_dir)

    def __str__(self):
        # Print out information about the train, test, and validation sets
        len_train = len(self.train.y)
        len_validation = len(self.validation.y)
        len_test = len(self.test.y)
        len_total = len_train + len_validation + len_test
        return f"# Dataset Info: {self.dataset_dir}\n" \
               f"\t- Train: {len_train} ({format_percentage(len_train, len_total)})\n" \
               f"\t- Validation: {len_validation} ({format_percentage(len_validation, len_total)})\n" \
               f"\t- Test: {len_test} ({format_percentage(len_test, len_total)})\n" \
               f"\t- Total: {len_total}"

    def get_dataset_fn(self, dataset_dir: str, type: str) -> (np.array, np.array):
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

        len_mask = len(mask_fn)
        len_nomask = len(no_mask_fn)
        len_total = len_mask + len_nomask
        print(f"\t=> {type.upper()}\n"
              f"\t\t- With Mask: {len_mask} ({format_percentage(len_mask, len_total)})\n"
              f"\t\t- No Mask: {len_nomask} ({format_percentage(len_nomask, len_total)})\n"
              f"\t\t- Total: {len_total}")
        return X, y

    def load_img(self, fn: str) -> np.array:
        """
        load an image into a numpy array from the given filename
        # converts to grayscale
        """
        return np.array(Image.open(fn))

    def load_files(self, X_fn: np.array) -> list:
        """
        given a list of filenames, load in each image into a numpy array and return a list of the images
        """
        X = []
        # iterate through each filename and load in each image
        for i in tqdm(range(len(X_fn))):
            X.append(self.load_img(X_fn[i]))

        # returns X as a list because images are different sizes
        return X

    def to_categorical(self, y) -> np.array:
        """
        One hot encode the target labels
        """
        Y = np.zeros((len(y), len(np.unique(y))))
        for i in range(len(y)):
            Y[i, int(y[i])] = 1
        return Y

    def load_all(self) -> tuple:

        print("\t- Loading Datasets...")
        # get dataset of filenames and labels
        X_train_fn, y_train = self.get_dataset_fn(self.dataset_dir, "train")
        X_validation_fn, y_validation = self.get_dataset_fn(
            self.dataset_dir, "validation")
        X_test_fn, y_test = self.get_dataset_fn(self.dataset_dir, "test")

        # one hot encode target labels
        y_train = self.to_categorical(y_train)
        y_validation = self.to_categorical(y_validation)
        y_test = self.to_categorical(y_test)

        # load in images
        print("# Loading Train Set...")
        X_train = self.load_files(X_train_fn)
        print("# Loading Validation Set...")
        X_validation = self.load_files(X_validation_fn)
        print("# Loading Test Set...")
        X_test = self.load_files(X_test_fn)

        self.train = Data(X_train, y_train)
        self.validation = Data(X_validation, y_validation)
        self.test = Data(X_test, y_test)
        print("\t- Datasets Loaded!")
        return self.train, self.validation, self.test

    def transform(self, x_routine, overwrite=False):
        # Keep track of the transformations performed and skip transforming if already done
        if x_routine.__name__ in self.transforms and not overwrite:
            raise ValueError(
                f"Dataset has already been transformed with {x_routine}!\nRe-run with overwrite=True to overwrite.")

        print(f"# Performing Feature Set Transformation: {x_routine.__name__}")
        sets = [self.train, self.validation, self.test]
        for i, data in enumerate(sets):
            print(f"\t- {i + 1}/{len(sets)}")
            data.X = x_routine(data.X)

        self.transforms.append(x_routine.__name__)
        # print("\t- (1/3) Transforming Train Set...", flush=True)
        # self.train.X = x_routine(self.train.X)
        # print("\t- (2/3) Transforming Train Set...", flush=True)
        # self.validation.X = x_routine(self.validation.X)
        # print("\t- (3/3) Transforming Train Set...", flush=True)
        # self.test.X = x_routine(self.test.X)

    def copy_transform(self, x_routine):
        copy = deepcopy(self)
        copy.transform(x_routine)
        return copy

    def print_transforms(self):
        print("# Transformations:")
        for i, transform in enumerate(self.transforms):
            print(f"\t- {i+1}. {transform.__name__}")

    def save_gzip(self, dir_path, file_name, overwrite=False):
        os.makedirs(dir_path, exist_ok=True)
        fullpath = os.path.join(dir_path, file_name)

        # If the file already exists output an error and prompt the user to use the overwrite argument
        if os.path.isfile(fullpath) and not overwrite:
            raise FileExistsError(
                f"File already exists: {fullpath}! Re-run with overwrite=True to overwrite.")

        print("# Saving to Gzip Pickle File:", fullpath)
        with gzip.open(fullpath, 'wb') as f:
            pickle.dump(self, f)

        print(" => Done!")

    @staticmethod
    def load_gzip(dir_path, file_name):
        fullpath = os.path.join(dir_path, file_name)
        # print("# Loading from Gzip Pickle File:", fullpath)

        if not os.path.isfile(fullpath):
            raise FileNotFoundError(
                f"File not found: {fullpath}")

        with gzip.open(fullpath, 'rb') as f:
            new = pickle.load(f)
            return new
