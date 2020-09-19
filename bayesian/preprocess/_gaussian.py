import numpy as np


class GaussianFeatures(object):

    def __init__(self, loc: np.ndarray, scale: float):
        loc = np.asarray(loc)
        if loc.ndim == 1:
            loc = loc[..., None]
        self.loc = loc
        self.var = scale ** 2
        self.ndim = len(loc) + 1

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[..., None]
        basis = [np.ones(len(x))]
        for loc in self.loc:
            basis.append(
                np.exp(-0.5 * np.square((x - loc).sum(axis=-1)) / self.var))
        return np.asarray(basis).T
