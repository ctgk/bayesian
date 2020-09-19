from bayesian.preprocess._gaussian import GaussianFeatures
from bayesian.preprocess._polynomial import PolynomialFeatures
from bayesian.preprocess._sigmoid import SigmoidalFeatures


_classes = [
    GaussianFeatures,
    PolynomialFeatures,
    SigmoidalFeatures,
]


for _cls in _classes:
    _cls.__module__ = 'bayesian.preprocess'


__all__ = [_cls.__name__ for _cls in _classes]
