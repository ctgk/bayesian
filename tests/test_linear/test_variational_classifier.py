import pytest

from bayesian.linear import VariationalClassifier


def test_fit():
    model = VariationalClassifier(a0=1e-5, b0=0.1)
    model.fit([[1, -1]] * 10 + [[1, 1]] * 10, [0] * 10 + [1] * 10)
    assert model.proba([[1, -1000]])[0] < 0.05
    assert model.proba([[1, 1000]])[0] > 0.95


if __name__ == "__main__":
    pytest.main([__file__])
