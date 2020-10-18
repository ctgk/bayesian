import pytest

from bayesian.preprocess import PolynomialFeature


def test_init():
    PolynomialFeature(1)


@pytest.mark.parametrize('f1, f2, expected', [
    (PolynomialFeature(2), PolynomialFeature(2), True),
    (PolynomialFeature(2), PolynomialFeature(1), False),
    (PolynomialFeature(2), PolynomialFeature(3), False),
])
def test_eq(f1, f2, expected):
    assert (f1 == f2) is expected
    assert (f1 != f2) is not expected


if __name__ == "__main__":
    pytest.main([__file__])
