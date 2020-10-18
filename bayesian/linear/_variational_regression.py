import numpy as np

from bayesian.linear._regression import Regression


class VariationalRegression(Regression):
    r"""
    # Variational Bayesian Linear Regression

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
        p({\boldsymbol y}|{\boldsymbol w}, {\bf X}, \beta)
            &= \prod_i p(y_i|{\boldsymbol w}, {\boldsymbol x}_i,\beta^{-1})\\
        &= \prod_i \mathcal{N}(y_i
            |{\boldsymbol w}^{\top}{\boldsymbol x}_i,\beta^{-1})
    \end{aligned}
    $$

    ### Variational posterior distribution of model parameter

    $$
    \begin{aligned}
    p({\boldsymbol w},\alpha|{\boldsymbol y},{\bf X},\beta,a_0,b_0)
        &\approx q({\boldsymbol w})q(\alpha)\\
    &= \mathcal{N}({\boldsymbol w}|{\boldsymbol m}_N, {\bf S}_N)
        {\rm Gam}(\alpha|a,b)
    \end{aligned}
    $$

    ### Variational predictive distribution

    $$
    \begin{aligned}
    p(y|{\boldsymbol x},{\bf y},{\bf X},a_0,b_0,\beta)
        &= \iint p(y|{\boldsymbol w},{\boldsymbol x},\beta)
            p({\boldsymbol w},\alpha|{\boldsymbol y},{\bf X},\beta,a_0,b_0)
            {\rm d}{\boldsymbol w}{\rm d}\alpha\\
    &\approx \int p(y|{\boldsymbol w},{\boldsymbol x},\beta)
        q({\boldsymbol w}){\rm d}{\boldsymbol w}
    \end{aligned}
    $$
    """

    def __init__(
            self,
            a0: float,
            b0: float,
            beta: float,
            iter_max: int = 100,
            feature=None):
        self.a0 = a0
        self.b0 = b0
        self.beta = beta
        self.iter_max = iter_max
        self.feature = feature
        self.hyperparameters = [a0, b0, beta]

    def __eq__(self, other):
        if not isinstance(other, VariationalRegression):
            return False
        return (self.hyperparameters == other.hyperparameters) and (
            self.feature == other.feature)

    def fit(self, x, y):
        x = self._preprocess(x)
        y = np.asarray(y)
        self.a = self.a0 + np.size(x, -1)
        self.b = self.b0
        eye = np.eye(np.size(x, -1))
        xx = x.T @ x
        bxy = self.beta * x.T @ y
        for _ in range(self.iter_max):
            b_prev = self.b
            self.w_cov = np.linalg.inv(self.a * eye + self.beta * xx)
            self.w_mean = self.w_cov @ bxy
            self.b = self.b0 + 0.5 * (
                np.sum(self.w_mean ** 2) + np.trace(self.w_cov))
            if np.allclose(self.b, b_prev):
                break

    def predict(self, x):
        return super().predict(x)
