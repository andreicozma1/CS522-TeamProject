#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter
import numpy as np


def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.Xtrain = X
        self.ytrain = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distances = [euc_dist(x, xtrain) for xtrain in self.Xtrain]

        # get k nearest samples, labels

        k_indices = np.argsort(distances)[:self.k]
        k_nearst_labels = [self.ytrain[i] for i in k_indices]

        # majority vote, most common class label
        most_common = Counter(k_nearst_labels).most_common(1)
        return most_common[0][0]