import numpy as np

from bayesian.preprocess._feature import Feature


class PolynomialFeature(Feature):

    def __init__(self, degree: int):
        self.degree = degree

    def __eq__(self, other):
        if not isinstance(other, PolynomialFeature):
            return False
        return self.degree == other.degree

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        return np.power(x, self.degree)
