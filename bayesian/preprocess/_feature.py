import abc
from itertools import zip_longest

import numpy as np


class Feature(abc.ABC):

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def transform(self, x) -> np.ndarray:
        pass


class StackedFeatures(Feature):

    def __init__(self, *features: Feature):
        self.features = features

    def __eq__(self, other):
        if not isinstance(other, StackedFeatures):
            return False
        return all(f1 == f2 for f1, f2 in zip_longest(
            self.features, other.features))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([f.transform(x) for f in self.features], axis=-1)
