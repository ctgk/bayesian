import numpy as np

from bayesian.preprocess._feature import Feature


class BiasFeature(Feature):

    def __eq__(self, other):
        return isinstance(other, BiasFeature)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.ones((len(x), 1))
