from bayesian.linear._regression import Regression


_classes = [
    Regression,
]


for _cls in _classes:
    _cls.__module__ = 'bayesian.linear'


__all__ = [_cls.__name__ for _cls in _classes]
