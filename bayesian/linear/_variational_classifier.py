import numpy as np


from bayesian.linear._classifier import Classifier


class VariationalClassifier(Classifier):

    def __init__(self, a0: float, b0: float, feature=None):
        self.a0 = a0
        self.b0 = b0
        self.feature = feature

    @property
    def hyperparameters(self):
        return (self.a0, self.b0)

    def fit(self, x, y, iter_max: int = 100):
        x = self._preprocess(x)
        y = np.asarray(y)
        n, d = x.shape
        self.a = self.a0 + 0.5 * d
        self.b = self.b0
        self.alpha = self.a / self.b
        xi = np.random.uniform(-1, 1, size=n)
        eye = np.eye(d)
        prev = np.copy(xi)
        for _ in range(iter_max):
            lambda_ = np.tanh(xi) * 0.25 / xi
            self.w_precision = self.alpha * eye + 2 * (lambda_ * x.T) @ x
            self.w_mean = np.linalg.solve(
                self.w_precision, np.sum(x.T * (y - 0.5), axis=1))
            xi = np.sqrt(np.sum(
                x @ (self.w_mean * self.w_mean[:, None] + np.linalg.inv(
                    self.w_precision)) * x,
                axis=-1))
            if np.allclose(xi, prev):
                break
            else:
                prev = np.copy(xi)
