import numpy as np
import pytest

from bayesian.preprocess import PolynomialFeatures


def test_init():
    feature = PolynomialFeatures([True, True, False, True])
    assert feature.ndim == 3


def test_transform():
    feature = PolynomialFeatures([True, True, True])
    actual = feature.transform([
        [2, -3],
        [3, 4],
    ])
    assert np.allclose(actual, [
        [1, 2, -3, 4, 9],
        [1, 3, 4, 9, 16],
    ])


if __name__ == "__main__":
    pytest.main([__file__])
