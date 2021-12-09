import numpy as np
from itertools import permutations, repeat
from operator import iconcat
from functools import reduce


class NaiveBayesFuser:
    def __init__(self):
        self.matrices = []
        self.fused_hypercube = None

    def add_matrix(self, matrix: np.array):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")

        matrix = np.copy(matrix.T).astype(float)
        matrix /= np.sum(matrix, axis=0)  # Divide rows by their sum

        if self.fused_hypercube is None:
            self.matrices.append(matrix)
            self.fused_hypercube = matrix
            return

        if matrix.shape[0] != self.fused_hypercube.shape[0]:
            raise ValueError("Matrix sizes must match")

        self.matrices.append(matrix)

        new_hypercube = np.zeros(
            self.fused_hypercube.shape + (matrix.shape[0],))
        for indices in np.ndindex(self.fused_hypercube.shape):
            hypercube_index = indices[:-1]
            matrix_index = indices[-1]

            new_hypercube[indices] = self.fused_hypercube[hypercube_index] * \
                                     matrix[matrix_index]

        self.fused_hypercube = new_hypercube

    def classify(self, predictions: np.array) -> int:
        return np.argmax(self.fused_hypercube[predictions])

    def get_readable_fused_hypercube(self) -> np.array:
        axes = list(range(self.fused_hypercube.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]  # Swap only the last and second-to-last axes

        return np.transpose(self.fused_hypercube, axes=axes)


class BKS:
    def __init__(self, y_true: np.array):
        self.n_classifiers = 0
        self.training_pred = []
        self.y_true = y_true
        self.N = len(y_true)
        self.classes = np.unique(y_true)
        self.n_classes = len(self.classes)
        self.lookup = {}

    def add_train_pred(self, y_pred: np.array):
        assert self.N == len(y_pred)
        self.n_classifiers += 1
        self.training_pred.append(y_pred)
        return self

    def compile(self):

        # create list of tuples of all possible classification options
        combinations = list(
            set(permutations(sorted(reduce(iconcat, list(repeat(self.classes, times=self.n_classifiers)), [])),
                             self.n_classifiers)))

        # for each combination of predictions find the actual ground truths
        for c in combinations:
            # find indices where classifiers predicted the current combination of labels
            a = np.ones(self.N, dtype=bool)
            for m in range(self.n_classifiers):
                # get all the true labels for the classes predictions
                a = np.logical_and(a, (self.training_pred[m] == c[m]))

            # get all the unique values of the ground truths
            uniq = np.unique(self.y_true[a], return_counts=True)

            # assign the counts of the unique values to the correct indices in counts
            # Note: this accounts for when there are less than n_classes results in np.unique
            counts = np.zeros(self.n_classes)
            for i, n in enumerate(uniq[0]):
                # get index of value in the classes array
                j = np.where(self.classes == n)[0]
                if len(j) == 0:
                    continue
                counts[j[0]] = uniq[1][i]

            # get indices that have the max value
            mask = np.where(counts == np.max(counts))

            # get classes that have those max values
            decision = self.classes[mask]

            # assign the decisions to the lookup table
            self.lookup[c] = decision

        return self

    def predict(self, *args):
        # check input data
        assert len(args) > 0
        assert len(args) == self.n_classifiers

        # set initial values
        n = len(args[0])
        pred = np.zeros(n)

        # for each prediction and each classifier pick one of the choices
        for i in range(n):
            l = []
            for j in range(len(args)):
                l.append(args[j][i])
            # if there is only one decision it will pick that decision
            # if there is a tie in decisions pick a random one
            pred[i] = np.random.choice(self.lookup[tuple(l)], 1)[0]
        return pred
