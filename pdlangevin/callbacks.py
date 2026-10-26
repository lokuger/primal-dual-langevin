"""Callbacks for computing statistics of Markov chains on the fly."""
import numpy as np
import odl
from odl.solvers import Callback


class RunningMoments:
    """Running mean and variance of a sequence of samples (Welford's algorithm).

    Works for odl space elements as well as numpy arrays.

    References
    ----------
    B. P. Welford. Note on a method for calculating corrected sums of squares
    and products. Technometrics 4(3), 1962.
    """

    def __init__(self):
        self.iter = 0
        self.mean = None
        self.S = None

    def update(self, x):
        self.iter += 1
        if self.iter == 1:
            if isinstance(x, odl.set.space.LinearSpaceElement):
                self.mean = x.copy()
                self.S = x.space.zero()
            else:
                self.mean = np.array(x, dtype=float)
                self.S = np.zeros_like(self.mean)
        else:
            mean_new = self.mean + (x - self.mean) / self.iter
            self.S += (x - self.mean) * (x - mean_new)
            self.mean = mean_new

    @property
    def var(self):
        """Unbiased sample variance, zero after a single sample."""
        if self.iter == 0:
            return None
        if self.iter == 1:
            return 0 * self.S
        return self.S / (self.iter - 1)

    @property
    def std(self):
        var = self.var
        if var is None:
            return None
        if isinstance(var, odl.set.space.LinearSpaceElement):
            return var.ufuncs.sqrt()
        return np.sqrt(var)


class CallbackRunningMoments(Callback):
    """Update a `RunningMoments` instance with every iterate."""

    def __init__(self, running_moments=None):
        self.running_moments = RunningMoments() if running_moments is None else running_moments

    def __call__(self, x):
        self.running_moments.update(x)

    def reset(self):
        self.running_moments = RunningMoments()


class CallbackAfterBurnIn(Callback):
    """Forward iterates to other callbacks only after ``burn_in`` iterations."""

    def __init__(self, *callbacks, burn_in=0):
        self.callbacks = callbacks
        self.burn_in = int(burn_in)
        self.iter = 0

    def __call__(self, x):
        self.iter += 1
        if self.iter > self.burn_in:
            for c in self.callbacks:
                c(x)

    def reset(self):
        self.iter = 0
        for c in self.callbacks:
            c.reset()


class CallbackEvery(Callback):
    """Call ``function(x)`` at iterations ``step, 2*step, ...`` and store the results."""

    def __init__(self, function, step):
        self.function = function
        self.step = int(step)
        self.iter = 0
        self.results = []

    def __call__(self, x):
        self.iter += 1
        if self.iter % self.step == 0:
            self.results.append(self.function(x))

    def reset(self):
        self.iter = 0
        self.results = []
