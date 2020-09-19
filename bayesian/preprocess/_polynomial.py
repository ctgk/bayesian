import typing as tp
import numpy as np


class PolynomialFeatures(object):

    def __init__(self, use_nth_deg: tp.List[bool]):
        self.use_nth_deg = use_nth_deg
        self.ndim = sum(use_nth_deg)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[..., None]
        matrix = [
            x ** i if i != 0 else np.ones((len(x), 1))
            for i, use in enumerate(self.use_nth_deg) if use
        ]
        return np.concatenate(matrix, axis=1)
