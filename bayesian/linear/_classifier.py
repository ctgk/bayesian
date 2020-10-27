import numpy as np

from bayesian._model import Model


class Classifier(Model):
    r"""
    ### Prior distribution of model parameter

    $$
    p({\boldsymbol w} | \alpha)
        = \mathcal{N}({\boldsymbol 0}, \alpha^{-1}{\bf I})
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

    ### Posterior distribution of model parameter

    $$
    \begin{aligned}
        p({\boldsymbol w}|{\boldsymbol y},{\bf X},\alpha)
            &\propto p({\boldsymbol y}|{\boldsymbol w}, {\bf X})
                p({\boldsymbol w} | \alpha)\\
        &\approx \mathcal{N}(
            {\boldsymbol w}|{\boldsymbol w}_{\rm MAP}, {\bf S}_N)\\
    \end{aligned}
    $$

    Using the Laplace approximation of the posterior distribution,
    ${\bf S}_N$ is the Hessian matrix of the log posterior distribution at
    ${\boldsymbol w} = {\boldsymbol w}_{\rm MAP}$.

    ### Predictive distribution

    $$
    \begin{aligned}
        p(y|{\boldsymbol x}, {\bf y}, {\bf X}, \alpha)
            &= \int p(y|{\boldsymbol w},{\boldsymbol x})
                p({\boldsymbol w}|{\boldsymbol y},{\bf X},\alpha)
                    {\rm d}{\boldsymbol w}\\
            &\approx \int {\rm Bern}(y|{\boldsymbol w},{\boldsymbol x})
                \mathcal{N}({\boldsymbol w}|{\boldsymbol m}_{\rm MAP},
                    {\bf S}_N){\rm d}{\boldsymbol w}\\
    \end{aligned}
    $$
    """

    def __init__(self, alpha: float, feature=None):
        super().__init__()
        self.alpha = alpha
        self.feature = feature

    @staticmethod
    def _sigmoid(a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def _log_prior(self, w):
        return 0.5 * (
            np.log(self.alpha)
            - self.alpha * np.square(w).sum()
            - np.log(2 * np.pi))

    @staticmethod
    def _log_likelihood(labels, logits):
        return np.sum(
            labels * logits
            - np.maximum(logits, 0)
            - np.log1p(np.exp(-np.abs(logits)))
        )

    def _log_posterior(self, x, y, w):
        return self._log_likelihood(y, x @ w) + self._log_prior(w)

    @property
    def hyperparameters(self):
        return (self.alpha,)

    def fit(self, x, y, iter_max: int = 100):
        x = self._preprocess(x)
        y = np.asarray(y)
        w = np.zeros((np.size(x, 1)))
        eye = np.eye(np.size(x, 1))
        score = -np.inf
        for _ in range(iter_max):
            p = self._sigmoid(x @ w)
            grad = x.T @ (p - y) + self.alpha * w
            hessian = (x.T * p * (1 - p)) @ x + self.alpha * eye
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            score_current = self._log_posterior(x, y, w)
            assert score_current > score
            if score_current - score > 1e-7:
                break
            score = score_current
        self.w_mean = w
        self.w_precision = hessian

    def proba(self, x: np.ndarray) -> np.ndarray:
        x = self._preprocess(x)
        mu = x @ self.w_mean
        var = np.sum(np.linalg.solve(self.w_precision, x.T).T * x, axis=1)
        return self._sigmoid(mu / np.sqrt(1 + np.pi * var / 8))
