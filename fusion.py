import numpy as np


class NaiveBayesFuser:
    def __init__(self):
        self.matrices = []
        self.fused_hypercube = None

    def add_matrix(self, matrix: np.array):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")

        matrix = np.copy(matrix.T).astype(float)
        matrix /= np.sum(matrix, axis=0)    # Divide rows by their sum

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
        axes[-2], axes[-1] = axes[-1], axes[-2] # Swap only the last and second-to-last axes

        return np.transpose(self.fused_hypercube, axes=axes)
