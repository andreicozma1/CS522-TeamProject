import numpy as np


class MPPCase1:
    def __init__(self, priors: list):
        self.priors = np.array(priors)
        self.k = len(priors)
        self.means = {}
        self.var = None
    
    def fit(self, X: np.array, y: np.array):
        X = np.reshape(np.copy(X), (X.shape[0], -1))
        self.var = np.mean(np.diagonal(np.cov(X.T)))
        for k in range(self.k):
            self.means[k] = np.mean(X[y == k], axis=0)

    def predict(self, X: np.array) -> np.array:
        g = np.array([[np.dot(x - self.means[k], x - self.means[k]) for x in X] for k in range(self.k)])
        g /= self.var
        for k in range(self.k):
            g[k] += np.log(self.priors[k])
        
        return np.argmin(g, axis=0)