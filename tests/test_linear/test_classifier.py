import numpy as np
import pytest

from bayesian.linear import Classifier
from bayesian.preprocess import GaussianFeature, StackedFeatures


def test_fit():
    model = Classifier(1e-5)
    model.fit(
        [[1, 0], [1, 1], [1, 2], [1, 3], [1, 5], [1, 6], [1, 7], [1, 8]],
        [1, 1, 1, 0, 1, 0, 0, 0])
    assert np.isclose(model.proba([[1, 4]])[0], 0.5, rtol=0, atol=1e-4)
    assert model.proba([[1, -1000]])[0] > 0.9
    assert model.proba([[1, 1000]])[0] < 0.1


@pytest.mark.parametrize('a1, a2, x, y, feature', [
    (0.1, 1., [[1, 3], [1, 5]], [0, 1], None),
    (10., 1., [[1, 2], [1, 10]], [1, 0], None),
    (
        10., 1., [2, 10], [1, 0],
        StackedFeatures(GaussianFeature(3, 0.1), GaussianFeature(5, 0.1))
    ),
])
def test_hyperparameters(a1: float, a2: float, x: list, y: list, feature):
    m1 = Classifier(a1, feature=feature)
    m1.fit(x, y)
    m2 = Classifier(a2, feature=feature)
    m2.fit(x, y)
    assert (np.linalg.norm(m1.w_mean) > np.linalg.norm(m2.w_mean)) == (a1 < a2)
    assert (
        (np.linalg.det(m1.w_precision) < np.linalg.det(m2.w_precision))
        == (a1 < a2))


if __name__ == "__main__":
    pytest.main([__file__])
