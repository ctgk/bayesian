import numpy as np
import pytest

from bayesian.preprocess import Feature, StackedFeatures


class Derived(Feature):

    def __init__(self, arg):
        self.arg = arg

    def __eq__(self, other):
        return self.arg == other.arg

    def transform(self, x: np.ndarray):
        return np.zeros((len(x), 1))


def test_feature_init():
    with pytest.raises(TypeError):
        Feature()


def test_stacked_features_init():
    StackedFeatures(Derived(1), Derived(2))


def test_stacked_features_eq():
    a = StackedFeatures(Derived(1), Derived(2))
    b = StackedFeatures(Derived(2), Derived(1))
    c = StackedFeatures(Derived(1), Derived(2))
    assert a != b
    assert a == c
    assert b != c


if __name__ == "__main__":
    pytest.main([__file__])
