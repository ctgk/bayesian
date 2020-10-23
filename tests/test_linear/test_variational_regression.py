import numpy as np
import pytest

from bayesian.linear import VariationalRegression


def test_fit():
    model = VariationalRegression(1, 1, 100)
    model.fit([-1, 1], [-1, 1])
    assert np.allclose(model.predict([-1, 1])[0], [-1, 1], rtol=0., atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
