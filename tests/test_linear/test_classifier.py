import numpy as np
import pytest

from bayesian.linear import Classifier


def test_fit():
    model = Classifier(0.)
    model.fit(
        [[1, 0], [1, 1], [1, 2], [1, 3], [1, 5], [1, 6], [1, 7], [1, 8]],
        [1, 1, 1, 0, 1, 0, 0, 0])
    assert np.isclose(model.proba([[1, 4]])[0], 0.5)
    assert model.proba([[1, -1000]])[0] > 0.9
    assert model.proba([[1, 1000]])[0] < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
