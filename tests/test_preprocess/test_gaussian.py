import numpy as np
import pytest

from bayesian.preprocess import GaussianFeature


def test_init():
    GaussianFeature([-1, 0, 1], 1)


@pytest.mark.parametrize('f1, f2, expected', [
    (GaussianFeature([1, 2], 0.3), GaussianFeature([1, 2], 0.3), True),
    (GaussianFeature([1, 2], 0.3), GaussianFeature([1, 2], 0.2), False),
    (GaussianFeature([1, 2], 0.3), GaussianFeature([3, 2], 0.3), False),
    (GaussianFeature([1, 2], 0.3), GaussianFeature([1], 0.3), False),
])
def test_eq(f1, f2, expected):
    assert (f1 == f2) == expected
    assert (f1 != f2) != expected


@pytest.mark.parametrize('loc, scale, x, expected', [
    (
        [1, -1], 1.,
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],
        [[np.exp(-2)], [np.exp(-4)], [1.], [np.exp(-2)]],
    ),
])
def test_transform(loc, scale, x, expected):
    feature = GaussianFeature(loc, scale)
    actual = feature.transform(x)
    assert np.allclose(actual, expected)


if __name__ == "__main__":
    pytest.main([__file__])
