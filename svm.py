"""
Code from:
ftp://ftp.ai.mit.edu/pub/users/tlp/projects/svm/svm-smo/smo.pdf
"""
import numba
from numba import njit
import numpy as np
# import cupy as np


class SVM:
    def __init__(self, kernel='linear', C=10000.0, max_iter=100000, degree=3, gamma=1, tolerance=0.001, epsilon=0.001):
        self.is_linear = True if kernel == 'linear' else False

        @njit(parallel=True, fastmath=True)
        def rbf(x,y):
            return np.exp(-gamma * np.linalg.norm(y-x) ** 2)

        @njit(parallel=True, fastmath=True)
        def poly(x,y):
            return np.dot(x, y.T) ** degree

        @njit(parallel=True, fastmath=True)
        def linear(x,y):
            return np.dot(x, y.T)
        self.K = {'poly': poly,
                  'rbf': rbf,
                  'linear': linear}[kernel]

        self.C = C
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.alphas:np.array = None  # lagrangian multipliers
        self._X:np.array = None
        self._y:np.array = None
        self.w:np.array = None
        self.b:np.int = 0
        self.N:np.int = 0

        if self.is_linear:
            self.learned_func = self.learned_func_linear
            self.predict_func = self.predict_func_linear
        else:
            self.learned_func = self.learned_func_nonlinear
            self.predict_func = self.predict_func_nonlinear

    def fit(self, X, y, Te_X=None, Te_y=None):
        assert (Te_X is None and Te_y is None) or (Te_X is not None and Te_y is not None)
        # convert y labels to -1 an 1
        self._X = X
        self._y = np.where(y <= 0, -1, 1)
        self.N, self.n_features = self._X.shape

        self.E = np.zeros(self.N)  # errors
        self.alphas = np.zeros(self.N)  # lagrangian multipliers
        self.weights = np.zeros(self.n_features)  # only for linear kernels
        self.b = 0  # threshold

        num_changed = 0
        examine_all = 1
        print('starting training')
        m = 0
        while num_changed > 0 or examine_all:
            print('starting epoch #',m)
            num_changed = 0

            if examine_all:
                print('examine all')
                for k in range(self.N):
                    num_changed += self.examine_example(k)
                    print(f'k={k}, num_changed={k}')
            else:
                print('examine all else')

                for k in range(self.N):
                    if self.alphas[k] != 0 and self.alphas[k] != self.C:
                        num_changed += self.examine_example(k)
                    print(f'k={k}, num_changed={k}')

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1
            print('finished epoch #', m)
            if Te_X is not None:
                print('Test error rate:', self.error_rate(Te_X, Te_y))

        print('finished training')

    def examine_example(self, i1):
        print('examine example ',i1)
        y1 = self._y[i1]
        alph1 = self.alphas[i1]

        if alph1 > 0 and alph1 < self.C:
            E1 = self.E[i1]
        else:
            E1 = self.learned_func(i1) - y1

        r1 = y1 * E1
        if (r1 < -self.tolerance and alph1 < self.C) or (r1 > self.tolerance and alph1 > self.C):
            # try to find i2 in 3 ways return 1 if any are successful
            if self.try_E1_E2(i1, E1):
                return 1
            if self.try_non_bound_examples(i1):
                return 1

            if self.try_all_examples(i1):
                return 1
        return 0

    def try_E1_E2(self, i1, E1):
        i2 = -1
        t_max = 0
        # find k index with biggest error margin from first point
        for k in range(self.N):
            # alpha must be larger than 0 and less than C
            if self.alphas[k] > 0 and self.alphas[k] < self.C:
                E2 = self.E[k]
                tmp = np.abs(E1 - E2)
                if tmp > t_max:
                    # save the current largest error and index
                    t_max = tmp
                    i2 = k
        # if found
        if i2 >= 0 and self.take_step(i1, i2):
            return 1
        return 0

    def try_non_bound_examples(self, i1):
        k0 = np.random.randint(0, self.N)
        for k in range(k0, k0 + self.N):
            i2 = k % self.N
            if self.alphas[i2] > 0 and self.alphas[i2] < self.C:
                if self.take_step(i1, i2):
                    return 1
        return 0

    def try_all_examples(self, i1):
        k0 = np.random.randint(0, self.N)
        for k in range(k0, k0 + self.N):
            i2 = k % self.N
            if not (self.alphas[i2] > 0 and self.alphas[i2] < self.C):
                if self.take_step(i1, i2):
                    return 1

    def take_step(self, i1, i2):
        # print('Takestep Start: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))
        # if indices are the same return 0
        if i1 == i2:
            return 0

        # look up needed params
        alph1, y1, E1 = self.get_params(i1)
        alph2, y2, E2 = self.get_params(i2)
        # print('Got Params: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

        s = y1 * y2

        # compute L and H
        L, H = self.get_LH(y1, alph1, y2, alph2)
        if L == H:
            return 0
        # print('Got LH: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

        k11, k12, k22, eta = self.get_eta(i1, i2)
        # print('Got ETA: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

        if eta < 0:
            a2 = alph2 + y2 * (E2 - E1) / eta
            if a2 < L:
                a2 = L
            else:
                a2 = H
        else:
            # compute objective function at a2=L and a2=H
            Lobj, Hobj = self.get_objective_LH(E1, y2, alph2, E2, eta, L, H)
            # print('Got Obj LH: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

            if Lobj > Hobj + self.epsilon:
                a2 = L
            elif Lobj < Hobj - self.epsilon:
                a2 = H
            else:
                a2 = alph2

        if np.abs(a2 - alph2) < self.epsilon * (a2 + alph2 + self.epsilon):
            return 0

        a1 = alph1 - s * (a2 - alph2)
        if a1 < 0:
            a2 += s * a1
            a1 = 0
        elif a1 > self.C:
            a2 += s * (a1 - self.C)
            a1 = self.C

        # print('Start Update: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))
        # update threshold to reflect change in Lagrangian multipliers
        self.update_threshold(y1, alph1, a1, E1, y2, a2, alph2, E2, k11, k12, k22)

        # update weight vectors to reflect change in a1 and a2, if linear SVM
        self.update_weights(i1, y1, alph1, a1, i2, y2, a2, alph2)

        # update error cache using new Lagrange Multiplier
        self.update_errors(i1, y1, alph1, a1, i2, y2, a2, alph2)
        # print('Finish Update: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

        # store the new alpha values
        self.alphas[i1] = a1
        self.alphas[i2] = a2

        return 1

    def predict(self, X):
        print('starting predicting')
        y_pred = self.predict_func(X)
        print('finished predicting')
        return y_pred

    def get_params(self, i):
        alph = self.alphas[i]
        y = self._y[i]
        if alph > 0 and alph < self.C:
            E = self.E[i]
        else:
            E = self.learned_func(i) - y
        return alph, y, E

    def get_LH(self, y1, alph1, y2, alph2):
        if y1 == y2:
            gamma = alph1 + alph2
            if gamma > self.C:
                L = gamma - self.C
                H = self.C
            else:
                L = 0
                H = gamma
        else:
            gamma = alph1 - alph2
            if gamma > 0:
                L = 0
                H = self.C - gamma
            else:
                L = -gamma
                H = self.C
        return L, H

    def get_eta(self, i1, i2):
        k11 = self.K(self._X[i1], self._X[i1])
        k12 = self.K(self._X[i1], self._X[i2])
        k22 = self.K(self._X[i2], self._X[i2])
        eta = 2 * k12 - k11 - k22
        return k11, k12, k22, eta

    def get_objective_LH(self, E1, y2, alph2, E2, eta, L, H):
        c1 = eta / 2
        c2 = y2 * (E1 - E2) - eta * alph2
        Lobj = c1 * L * L + c2 * L
        Hobj = c1 * H * H + c2 * H
        return Lobj, Hobj

    def update_threshold(self, y1, alph1, a1, E1, y2, a2, alph2, E2, k11, k12, k22):
        if a1 > 0 and a1 < self.C:
            b_new = self.b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12
        else:
            if a2 > 0 and a2 < self.C:
                b_new = self.b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22
            else:
                b1 = self.b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12
                b2 = self.b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22
                b_new = (b1 + b2) / 2
        self.delta_b = b_new - self.b
        self.b = b_new

    def update_weights(self, i1, y1, alph1, a1, i2, y2, a2, alph2):
        # assumes data is not sparse and it is binary
        if self.is_linear:
            t1 = y1 * (a1 - alph1)
            t2 = y2 * (a2 - alph2)
            self.w += self._X[i1] * t1 + self._X[i2] * t2

    def update_errors(self, i1, y1, alph1, a1, i2, y2, a2, alph2):
        t1 = y1 * (a1 - alph1)
        t2 = y2 * (a2 - alph2)
        for i in range(self.N):
            if 0 < self.alphas[i] < self.C:
                self.E += t1 * self.K(i1,i) + t2 * self.K(i2,i) - self.delta_b
        self.E[i1] = 0
        self.E[i2] = 0

    @njit(parallel=True, fastmath=True)
    def learned_func_linear(self,k):
        return self.w.dot(self._X[k]) - self.b

    # @njit(parallel=True, fastmath=True)
    def learned_func_nonlinear(self, k: np.int) -> np.float:
        s:np.float = 0
        for i in np.arange(0, self.N):
            if self.alphas[i] > 0:
                s += self.alphas[i]*self._y[i]*self.K(self._X[i], self._X[k])
        # s -= self.b
        s = np.subtract(s, self.b)
        return s

    def predict_func_linear(self,X):
        return X.dot(self.w) - self.b

    def predict_func_nonlinear(self,X):
        s = np.zeros(len(X))
        for k in range(len(X)):
            for i,x in enumerate(self._X):
                if self.alphas[i] > 0:
                    s[k] += self.alphas[i]*self._y[i]*self.K(self._X[i], X[k])
        return s - self.b

    def error_rate(self,X,y):
        y_pred = self.predict_func(X)
        n_total = len(y)
        n_error = n_total-np.sum(y == y_pred)
        return float(n_error) / float(n_total)