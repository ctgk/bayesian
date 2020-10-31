import abc

import numpy as np


class Model(abc.ABC):

    def __eq__(self, other: object):
        if type(self) != type(other):
            return False
        return (self.hyperparameters == other.hyperparameters) and (
            self.feature == other.feature
        )

    @property
    @abc.abstractmethod
    def hyperparameters(self):
        pass

    def _preprocess(self, x):
        if self.feature is None:
            x = np.asarray(x)
            if x.ndim == 1:
                x = x[:, None]
            return x
        return self.feature.transform(x)
