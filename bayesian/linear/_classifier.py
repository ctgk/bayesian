import numpy as np

from bayesian._model import Model


class Classifier(Model):
    """
    # Bayesian Logistic Regression

    ### Prior distribution of model parameter

    ### Likelihood function of model parameter

    ### Posterior distribution of model parameter

    ### Predictive distribution
    """

    def __init__(self, alpha: float, feature=None):
        super().__init__()
        self.alpha = alpha
        self.feature = feature
        self.hyperparameters = (alpha,)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.hyperparameters == other.hyperparameters) and (
            self.feature == other.feature)

    @staticmethod
    def _sigmoid(a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, x, y, iter_max: int = 100):
        x = self._preprocess(x)
        y = np.asarray(y)
        w = np.zeros((np.size(x, 1)))
        eye = np.eye(np.size(x, 1))
        for _ in range(iter_max):
            w_prev = np.copy(w)
            p = self._sigmoid(x @ w)
            grad = x.T @ (p - y) + self.alpha * w
            hessian = (x.T * p * (1 - p)) @ x + self.alpha * eye
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w_mean = w
        self.w_precision = hessian

    def proba(self, x: np.ndarray) -> np.ndarray:
        x = self._preprocess(x)
        mu = x @ self.w_mean
        var = np.sum(np.linalg.solve(self.w_precision, x.T).T * x, axis=1)
        return self._sigmoid(mu / np.sqrt(1 + np.pi * var / 8))
