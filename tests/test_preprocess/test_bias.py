import pytest

from bayesian.preprocess import BiasFeature


def test_init():
    BiasFeature()


def test_eq():
    BiasFeature() == BiasFeature()


if __name__ == "__main__":
    pytest.main([__file__])
