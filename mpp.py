import numpy as np


class MPPCase1:
    def __init__(self, priors: list):
        self.priors = np.array(priors)
        self.k = len(priors)
        self.means = {}
        self.var = 0.0

    def fit(self, X: np.array, y: np.array):
        X = np.reshape(np.copy(X), (X.shape[0], -1))
        for k in range(self.k):
            self.var += np.mean(np.diagonal(np.cov(X[y == k].T)))
            self.means[k] = np.mean(X[y == k], axis=0)

        self.var /= self.k

    def predict(self, X: np.array) -> np.array:
        g = np.array([[-0.5 * (x - self.means[k]).dot(x - self.means[k]) / self.var + np.log(self.priors[k])
                     for x in X] for k in range(self.k)])
        g /= self.var
        for k in range(self.k):
            g[k] += np.log(self.priors[k])

        return np.argmax(g, axis=0)


class MPPCase2:
    def __init__(self, priors: list):
        self.priors = np.array(priors)
        self.k = len(priors)
        self.means = {}
        self.var = None
        self.inv_var = None

    def fit(self, X: np.array, y: np.array):
        X = np.reshape(np.copy(X), (X.shape[0], -1))
        self.var = np.zeros((X.shape[1], X.shape[1]))
        for k in range(self.k):
            self.var += np.cov(X[y == k].T)
            self.means[k] = np.mean(X[y == k], axis=0)

        self.var /= self.k
        self.inv_var = np.linalg.inv(self.var)

    def predict(self, X: np.array) -> np.array:
        g = np.array([
            [-0.5 * (x - self.means[k]).T.dot(self.inv_var).dot(x - self.means[k])
             + np.log(self.priors[k]) for x in X]
            for k in range(self.k)])

        return np.argmax(g, axis=0)


class MPPCase3:
    def __init__(self, priors: list):
        self.priors = np.array(priors)
        self.k = len(priors)
        self.means = {}
        self.vars = {}
        self.inv_vars = {}
        self.det_vars = {}

    def fit(self, X: np.array, y: np.array):
        X = np.reshape(np.copy(X), (X.shape[0], -1))
        for k in range(self.k):
            self.vars[k] = np.cov(X[y == k].T)
            self.inv_vars[k] = np.linalg.inv(self.vars[k])
            _, self.det_vars[k] = np.linalg.slogdet(self.vars[k])
            self.means[k] = np.mean(X[y == k], axis=0)

    def predict(self, X: np.array) -> np.array:
        g = np.array([
            [-0.5 * (x - self.means[k]).T.dot(self.inv_vars[k]).dot(x - self.means[k])
             - 0.5 * self.det_vars[k] + np.log(self.priors[k]) for x in X]
            for k in range(self.k)])

        return np.argmax(g, axis=0)
