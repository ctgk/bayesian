import numpy as np

from bayesian._model import Model


class Regression(Model):
    r"""
    # Bayesian Linear Regression

    ### Prior disribution of model parameter

    $$
    p({\boldsymbol w} | \alpha)
        = \mathcal{N}({\boldsymbol 0}, \alpha^{-1}{\bf I})
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

    ### Posterior distribution of model parameter

    $$
    \begin{aligned}
        p({\boldsymbol w}|{\boldsymbol y},{\bf X},\alpha,\beta)
            &\propto p({\boldsymbol y}|{\boldsymbol w}, {\bf X}, \beta)
                p({\boldsymbol w} | \alpha)\\
        &=\mathcal{N}({\boldsymbol w}|{\boldsymbol m}_N, {\bf S}_N)\\
        &=\mathcal{N}({\boldsymbol w}|
            \beta{\bf S}_N{\bf X}^\top{\boldsymbol y},
            (\alpha{\bf I} + \beta{\bf X}^{\top}{\bf X})^{-1})\\
    \end{aligned}
    $$

    ### Predictive distribution

    $$
    \begin{aligned}
        p(y|{\boldsymbol x},{\bf y},{\bf X},\alpha,\beta)
            &=\int p(y|{\boldsymbol w},{\boldsymbol x},\beta)p({\boldsymbol w}
                |{\boldsymbol y},{\bf X},\alpha,\beta){\rm d}{\boldsymbol w}\\
        &=\mathcal{N}(y|{\boldsymbol m}_N^{\top}{\boldsymbol x},
            {1\over\beta}+{\boldsymbol x}^{\top}{\bf S}_N{\boldsymbol x})
    \end{aligned}
    $$
    """

    def __init__(
            self, alpha: float, beta: float, bias: bool = True, feature=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.feature = feature
        self.hyperparameters = [alpha, beta]

    def __eq__(self, other):
        if not isinstance(other, Regression):
            return False
        return (self.hyperparameters == other.hyperparameters) and (
            self.feature == other.feature)

    def fit(self, x, y):
        x = self._preprocess(x)
        y = np.asarray(y)
        self.w_cov = np.linalg.inv(
            np.eye(np.size(x, -1)) * self.alpha + self.beta * x.T @ x)
        self.w_mean = self.beta * self.w_cov @ x.T @ y
        self.w = {'mean': self.w_mean, 'cov': self.w_cov}

    def predict(self, x):
        x = self._preprocess(x)
        y = x @ self.w_mean
        y_var = 1 / self.beta + np.sum(x @ self.w_cov * x, axis=1)
        y_std = np.sqrt(y_var)
        return y, y_std
