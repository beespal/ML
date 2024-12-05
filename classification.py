import numpy as np


class Classification_step:
    """
    Classification model with step activation function
    """

    def __init__(self, n_epoch=10, learning_rate=None):
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate

    def fit(self, X, y):
        if self.learning_rate is None:
            self.learning_rate = 3 * 10**-3
        if len(X.shape) == 1:
            self.N = X.shape[0]
            self.M = 1
        else:
            self.N = X.shape[0]
            self.M = X.shape[1]
        self.W = np.random.randn(self.M)
        self.b = np.random.randn(1)[0]
        for i in range(self.n_epoch):
            for j in range(self.N):
                object_index = np.random.randint(0, self.N)
                object = X[object_index]
                target = y[object_index]
                score = object @ self.W + self.b
                if score >= 0:
                    prediction = 1
                else:
                    prediction = 0
                if prediction == target:
                    continue
                for k in range(self.M):
                    grad_w = np.sign(score) * object[k]
                    self.W[k] = self.W[k] - self.learning_rate * grad_w
                grad_b = np.sign(score)
                self.b = self.b - self.learning_rate * grad_b

    def predict(self, X):
        score = X @ self.W + self.b
        res = (score >= 0).astype(int)
        return res


class Classification_ReLU:
    """
    Classification model with ReLU activation function
    """

    def __init__(self, n_epoch=10, learning_rate=None):
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate

    def fit(self, X, y):
        if self.learning_rate is None:
            self.learning_rate = 3 * 10**-3
        if len(X.shape) == 1:
            self.N = X.shape[0]
            self.M = 1
        else:
            self.N = X.shape[0]
            self.M = X.shape[1]
        self.W = np.random.randn(self.M)
        self.b = np.random.randn(1)[0]
        for i in range(self.n_epoch):
            for j in range(self.N):
                object_index = np.random.randint(0, self.N)
                object = X[object_index]
                target = y[object_index]
                score = object @ self.W + self.b
                if score >= 0:
                    prediction = 1
                else:
                    prediction = 0
                if prediction == target:
                    continue
                for k in range(self.M):
                    grad_w = (
                        -target * int(-score >= 0) * object[k]
                        + (1 - target) * int(score >= 0) * object[k]
                    )
                    self.W[k] = self.W[k] - self.learning_rate * grad_w
                grad_b = -target * int(-score >= 0) + (1 - target) * int(score >= 0)
                self.b = self.b - self.learning_rate * grad_b

    def predict(self, X):
        score = X @ self.W + self.b
        res = (score >= 0).astype(int)
        return res


class Logistic_Regression:
    """
    Classification model with sigmoid activation function
    """

    def __init__(self, n_epoch=10, learning_rate=None):
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate

    def fit(self, X, y):
        if self.learning_rate is None:
            self.learning_rate = 3 * 10**-3
        if len(X.shape) == 1:
            self.N = X.shape[0]
            self.M = 1
        else:
            self.N = X.shape[0]
            self.M = X.shape[1]
        self.W = np.random.randn(self.M)
        self.b = np.random.randn(1)[0]
        for i in range(self.n_epoch):
            for j in range(self.N):
                object_index = np.random.randint(0, self.N)
                object = X[object_index]
                target = y[object_index]
                score = object @ self.W + self.b
                prediction = 1 / (1 + np.exp(score))
                for k in range(self.M):
                    grad_w = (target - prediction) * object[k]
                    self.W[k] = self.W[k] - self.learning_rate * grad_w
                grad_b = target - prediction
                self.b = self.b - self.learning_rate * grad_b

    def predict_proba(self, X):
        score = X @ self.W + self.b
        return 1 / (1 + np.exp(score))

    def predict(self, X):
        score = X @ self.W + self.b
        return (1 / (1 + np.exp(score)) >= 1 / 2).astype(int)
