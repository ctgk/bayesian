import abc

import numpy as np


class Model(abc.ABC):

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    def _preprocess(self, x):
        if self.feature is None:
            x = np.asarray(x)
            if x.ndim == 1:
                x = x[:, None]
            return x
        return self.feature.transform(x)
