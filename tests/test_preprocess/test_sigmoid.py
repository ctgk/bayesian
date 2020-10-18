import pytest

from bayesian.preprocess import SigmoidalFeature


def test_init():
    SigmoidalFeature([-1, 0, 1], 1)


@pytest.mark.parametrize('f1, f2, expected', [
    (SigmoidalFeature([1, 2], 0.3), SigmoidalFeature([1, 2], 0.3), True),
    (SigmoidalFeature([1, 2], 0.3), SigmoidalFeature([1, 2], 0.2), False),
    (SigmoidalFeature([1, 2], 0.3), SigmoidalFeature([3, 2], 0.3), False),
    (SigmoidalFeature([1, 2], 0.3), SigmoidalFeature([1], 0.3), False),
])
def test_eq(f1, f2, expected):
    assert (f1 == f2) == expected
    assert (f1 != f2) != expected


if __name__ == "__main__":
    pytest.main([__file__])
