import typing as tp

import numpy as np

from bayesian.preprocess._feature import Feature


class GaussianFeature(Feature):

    def __init__(self, loc: tp.List[float], scale: float):
        loc = np.asarray(loc)
        self.loc = loc
        self.scale = scale
        self.var = scale ** 2

    def __eq__(self, other):
        if not isinstance(other, GaussianFeature):
            return False
        return np.allclose(
            self.loc, other.loc) and np.isclose(self.scale, other.scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        return np.exp(-0.5 * np.square((x - self.loc).sum(
            axis=-1, keepdims=True)) / self.var)
