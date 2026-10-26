"""Additional potentials not available in odl."""
import numpy as np
import scipy.special
from odl.operator import Operator
from odl.solvers.functional import Functional


class MaskedKullbackLeibler(Functional):
    r"""Poisson negative log-likelihood on observed pixels plus non-negativity.

    .. math::
        F(x) = \sum_{i : m_i = 1} \left(x_i - y_i + y_i \log(y_i / x_i)\right)
               + \iota_{\geq 0}(x)

    with a binary mask :math:`m` (1 = observed) and data :math:`y`. Up to a
    constant this is the negative log-likelihood of :math:`y_i \sim
    \mathrm{Poisson}(x_i)` on the observed pixels. The proximal operator is

    .. math::
        \mathrm{prox}_{\tau F}(v)_i =
        \begin{cases}
            \frac12\left(v_i - \tau + \sqrt{(v_i - \tau)^2 + 4\tau y_i}\right)
                & m_i = 1, \\
            \max(v_i, 0) & m_i = 0.
        \end{cases}

    Parameters
    ----------
    space : `DiscretizedSpace` or `TensorSpace`
        Domain of the functional.
    mask : ``space`` element
        Binary mask, 1 for observed pixels.
    data : ``space`` element
        Observed counts :math:`y`, non-negative.
    """

    def __init__(self, space, mask, data):
        self.mask = mask
        self.data = data
        super().__init__(space, linear=False, grad_lipschitz=np.nan)

    def _call(self, x):
        if np.any(np.asarray(x) < 0):
            return np.inf
        with np.errstate(invalid='ignore', divide='ignore'):
            vals = x - self.data + scipy.special.xlogy(self.data, self.data / x)
            res = (self.mask * vals).inner(self.domain.one())
        return res if np.isfinite(res) else np.inf

    @property
    def gradient(self):
        func = self

        class MaskedKLGradient(Operator):
            def __init__(self):
                super().__init__(func.domain, func.domain, linear=False)

            def _call(self, x):
                return func.mask * (1 - func.data / x)

        return MaskedKLGradient()

    @property
    def proximal(self):
        func = self

        def factory(tau):
            tau = float(tau)

            class MaskedKLProximal(Operator):
                def __init__(self):
                    super().__init__(func.domain, func.domain, linear=False)

                def _call(self, v):
                    diff = v - tau
                    prox_kl = 0.5 * (diff + (diff ** 2 + 4 * tau * func.data).ufuncs.sqrt())
                    result = v + func.mask * (prox_kl - v)
                    result.ufuncs.maximum(0, out=result)
                    return result

            return MaskedKLProximal()

        return factory
