import numpy as np


from bayesian.linear._classifier import Classifier


class VariationalClassifier(Classifier):
    r"""
    ### Prior distribution of model parameter

    $$
    \begin{aligned}
    p({\alpha}|a_0,b_0) &= {\rm Gam}(\alpha|a_0, b_0)\\
    p({\boldsymbol w}|\alpha)
        &= \mathcal{N}({\boldsymbol 0}, \alpha^{-1}{\bf I})
    \end{aligned}
    $$

    ### Likelihood function of model parameter

    $$
    \begin{aligned}
    p({\boldsymbol y}|{\boldsymbol w}, {\bf X})
        &= \prod_i p(y_i|{\boldsymbol w}, {\boldsymbol x}_i)\\
    &= \prod_i {\rm Bern}(y_i
        |{\boldsymbol w}^{\top}{\boldsymbol x}_i)
    \end{aligned}
    $$

    ### Variational posterior distribution of model parameter

    $$
    \begin{aligned}
    p({\boldsymbol w},\alpha|{\boldsymbol y},{\bf X},a_0,b_0)
        &\approx q({\boldsymbol w})q(\alpha)\\
    &= \mathcal{N}({\boldsymbol w}|{\boldsymbol m}_N, {\bf S}_N)
        {\rm Gam}(\alpha|a,b)
    \end{aligned}
    $$

    ### Variational predictive distribution

    $$
    \begin{aligned}
    p(y|{\boldsymbol x},{\bf y},{\bf X},a_0,b_0)
        &= \iint p(y|{\boldsymbol w},{\boldsymbol x})
            p({\boldsymbol w},\alpha|{\boldsymbol y},{\bf X},a_0,b_0)
            {\rm d}{\boldsymbol w}{\rm d}\alpha\\
    &\approx \int p(y|{\boldsymbol w},{\boldsymbol x})
        q({\boldsymbol w}){\rm d}{\boldsymbol w}
    \end{aligned}
    $$
    """

    def __init__(self, a0: float, b0: float, feature=None):
        self.a0 = a0
        self.b0 = b0
        self.feature = feature

    @property
    def hyperparameters(self):
        return (self.a0, self.b0)

    def fit(self, x, y, iter_max: int = 100):
        x = self._preprocess(x)
        y = np.asarray(y)
        n, d = x.shape
        self.a = self.a0 + 0.5 * d
        self.b = self.b0
        self.alpha = self.a / self.b
        xi = np.random.uniform(-1, 1, size=n)
        eye = np.eye(d)
        prev = np.copy(xi)
        for _ in range(iter_max):
            lambda_ = np.tanh(xi) * 0.25 / xi
            self.w_precision = self.alpha * eye + 2 * (lambda_ * x.T) @ x
            self.w_mean = np.linalg.solve(
                self.w_precision, np.sum(x.T * (y - 0.5), axis=1))
            xi = np.sqrt(np.sum(
                x @ (self.w_mean * self.w_mean[:, None] + np.linalg.inv(
                    self.w_precision)) * x,
                axis=-1))
            if np.allclose(xi, prev):
                break
            else:
                prev = np.copy(xi)
