import numpy as np
import pytest

from bayesian.preprocess import SigmoidalFeatures


def test_init():
    feature = SigmoidalFeatures([-1, 0, 1], 1)
    assert feature.ndim == 4


def test_transform():
    feature = SigmoidalFeatures(np.random.rand(5), 1)
    actual = feature.transform(np.random.rand(10))
    assert actual.shape == (10, 6)


if __name__ == "__main__":
    pytest.main([__file__])
