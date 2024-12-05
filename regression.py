import numpy as np


class Linear_reg_mae:
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
                prediction = object @ self.W + self.b
                error = prediction - target
                for k in range(self.M):
                    grad_w = np.sign(error) * object[k]
                    grad_b = np.sign(error)
                    self.W[k] = self.W[k] - self.learning_rate * grad_w
                self.b = self.b - self.learning_rate * grad_b

    def predict(self, X):
        return X @ self.W + self.b


class Linear_reg_mse:
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
                prediction = object @ self.W + self.b
                error = prediction - target
                for k in range(self.M):
                    grad_w = 2 * error * object[k]
                    grad_b = 2 * error
                    self.W[k] = self.W[k] - self.learning_rate * grad_w
                self.b = self.b - self.learning_rate * grad_b
            self.learning_rate = self.learning_rate / (i + 1)

    def predict(self, X):
        return X @ self.W + self.b


def R2_score(y_true, y_pred):
    return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
