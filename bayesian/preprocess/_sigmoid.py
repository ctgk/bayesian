import numpy as np


class SigmoidalFeatures(object):

    def __init__(self, loc: np.ndarray, scale: float):
        loc = np.asarray(loc)
        if loc.ndim == 1:
            loc = loc[..., None]
        self.loc = loc
        self.scale = scale
        self.ndim = len(loc) + 1

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[..., None]
        basis = [np.ones(len(x))]
        for loc in self.loc:
            basis.append(
                np.tanh((x - loc).sum(axis=-1) * self.scale * 0.5) * 0.5 + 0.5)
        return np.asarray(basis).T
