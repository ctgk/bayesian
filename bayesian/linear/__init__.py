from bayesian.linear._regression import Regression
from bayesian.linear._variational_regression import VariationalRegression


_classes = [
    Regression,
    VariationalRegression
]


for _cls in _classes:
    _cls.__module__ = 'bayesian.linear'


__all__ = [_cls.__name__ for _cls in _classes]
