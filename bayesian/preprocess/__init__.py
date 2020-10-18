from bayesian.preprocess._bias import BiasFeature
from bayesian.preprocess._feature import Feature, StackedFeatures
from bayesian.preprocess._gaussian import GaussianFeature
from bayesian.preprocess._polynomial import PolynomialFeature
from bayesian.preprocess._sigmoid import SigmoidalFeature


_classes = [
    BiasFeature,
    Feature,
    GaussianFeature,
    PolynomialFeature,
    SigmoidalFeature,
    StackedFeatures,
]


for _cls in _classes:
    _cls.__module__ = 'bayesian.preprocess'


__all__ = [_cls.__name__ for _cls in _classes]
