"""Langevin sampling algorithms for densities of the form exp(-f(Kx) - g(x)).

All samplers update the initial state ``x`` in place and report the iterates
through callbacks; they never store samples themselves.
"""
import numpy as np
import odl
from odl.operator import Operator
from odl.solvers.functional import Functional


def _check_potentials(x, f, K, g):
    if not isinstance(f, Functional):
        raise TypeError('`f` {!r} is not a `Functional`'.format(f))
    if not isinstance(g, Functional):
        raise TypeError('`g` {!r} is not a `Functional`'.format(g))
    if not isinstance(K, Operator):
        raise TypeError('`K` {!r} is not an `Operator`'.format(K))
    if x not in K.domain:
        raise TypeError('`x` {!r} is not in the domain of `K` {!r}'.format(x, K.domain))
    if g.domain != K.domain:
        raise TypeError('`g.domain` {!r} must equal `K.domain` {!r}'.format(g.domain, K.domain))
    if K.range != f.domain:
        raise TypeError('`K.range` {!r} must equal `f.domain` {!r}'.format(K.range, f.domain))


def _standard_normal(space, rng):
    """Draw an element of ``space`` with i.i.d. standard normal entries."""
    if isinstance(space, odl.ProductSpace):  # e.g. the joint (u, v) space for TGV
        return space.element(tuple(rng.normal(size=s.shape) for s in space.spaces))
    return space.element(rng.normal(size=space.shape))


def primal_dual_langevin(x, f, K, g, tau, sigma, theta, niter, callback=None,
                         dual_callback=None, noise_inside_prox=False, **kwargs):
    """Unadjusted Langevin primal-dual algorithm (ULPDA).

    Samples approximately from the density proportional to exp(-f(Kx) - g(x))
    with convex, possibly non-smooth ``f`` and ``g``. One iteration reads

        y_{n+1}  = prox_{sigma f*}(y_n + sigma K x_theta_n)
        x_{n+1}  = prox_{tau g}(x_n - tau K^T y_{n+1}) + sqrt(2 tau) xi_{n+1}
        x_theta_{n+1} = x_{n+1} + theta (x_{n+1} - x_n)

    With ``noise_inside_prox=True``, the diffusion is added inside the
    proximal argument instead:

        x_{n+1}  = prox_{tau g}(x_n - tau K^T y_{n+1} + sqrt(2 tau) xi_{n+1})

    Parameters
    ----------
    x : ``K.domain`` element
        Initial state of the Markov chain, updated in place.
    f : `Functional`
        Dualized potential, needs ``f.convex_conj.proximal``.
    K : linear `Operator`
        Operator inside ``f``.
    g : `Functional`
        Primal potential, needs ``g.proximal``.
    tau, sigma : positive float
        Primal and dual step sizes. Their ratio ``sigma / tau`` is the
        parameter lambda of the paper.
    theta : float
        Over-relaxation parameter.
    niter : int
        Number of iterations.
    callback, dual_callback : callable, optional
        Called with the primal and dual iterate after each iteration.
    noise_inside_prox : bool, optional
        Add the diffusion before (True) or after (False, default) the
        proximal step in ``g``.

    Other Parameters
    ----------------
    x_relax : ``K.domain`` element, optional
        Initial over-relaxed primal variable. Default: copy of ``x``.
    y : ``K.range`` element, optional
        Initial dual variable. Default: zero.
    rng : `numpy.random.Generator`, optional
        Random number generator, pass a seeded one for reproducible chains.
    """
    _check_potentials(x, f, K, g)
    space = g.domain

    tau, sigma, theta = float(tau), float(sigma), float(theta)
    r = np.sqrt(2 * tau)

    x_relax = kwargs.pop('x_relax', None)
    if x_relax is None:
        x_relax = x.copy()
    elif x_relax not in K.domain:
        raise TypeError('`x_relax` {!r} is not in the domain of `K` {!r}'.format(x_relax, K.domain))

    y = kwargs.pop('y', None)
    if y is None:
        y = K.range.zero()
    elif y not in K.range:
        raise TypeError('`y` {!r} is not in the range of `K` {!r}'.format(y, K.range))

    rng = kwargs.pop('rng', None)
    if rng is None:
        rng = np.random.default_rng()
    if kwargs:
        raise TypeError('unexpected keyword argument(s): {}'.format(kwargs))

    prox_tau_g = g.proximal(tau)
    prox_sigma_fconj = f.convex_conj.proximal(sigma)

    x_old = space.element()
    dual_tmp = K.range.element()
    primal_tmp = space.element()

    for _ in range(niter):
        x_old.assign(x)

        # dual step: y = prox_{sigma f*}(y + sigma K x_relax)
        K(x_relax, out=dual_tmp)
        dual_tmp.lincomb(1, y, sigma, dual_tmp)
        prox_sigma_fconj(dual_tmp, out=y)

        # primal step with diffusion
        K.adjoint(y, out=primal_tmp)
        primal_tmp.lincomb(1, x, -tau, primal_tmp)
        if noise_inside_prox:
            primal_tmp.lincomb(1, primal_tmp, r, _standard_normal(space, rng))
            prox_tau_g(primal_tmp, out=x)
        else:
            prox_tau_g(primal_tmp, out=x)
            x.lincomb(1, x, r, _standard_normal(space, rng))

        # over-relaxation
        x_relax.lincomb(1 + theta, x, -theta, x_old)

        if callback is not None:
            callback(x)
        if dual_callback is not None:
            dual_callback(y)


def prox_sub_langevin(x, f, K, g, tau, niter, callback=None, dual_callback=None,
                      noise_inside_prox=False, **kwargs):
    """Proximal-subgradient Langevin algorithm (Prox-Sub) of [HHP2024].

    Samples approximately from the density proportional to exp(-f(Kx) - g(x)).
    The subgradient of ``f`` is taken from ``f.gradient``. One iteration reads

        theta_n = f.gradient(K x_n)
        x_{n+1} = prox_{tau g}(x_n - tau K^T theta_n) + sqrt(2 tau) xi_{n+1}

    or, with ``noise_inside_prox=True``,

        x_{n+1} = prox_{tau g}(x_n - tau K^T theta_n + sqrt(2 tau) xi_{n+1}).

    It is the formal limit lambda -> infinity of `primal_dual_langevin`.

    Parameters
    ----------
    x : ``K.domain`` element
        Initial state of the Markov chain, updated in place.
    f : `Functional`
        Potential composed with ``K``, needs ``f.gradient`` (a subgradient).
    K : linear `Operator`
        Operator inside ``f``.
    g : `Functional`
        Potential with ``g.proximal``.
    tau : positive float
        Step size.
    niter : int
        Number of iterations.
    callback, dual_callback : callable, optional
        Called with ``x_n`` and the subgradient ``theta_n`` after each iteration.
    noise_inside_prox : bool, optional
        Add the diffusion before (True) or after (False, default) the
        proximal step in ``g``.

    Other Parameters
    ----------------
    rng : `numpy.random.Generator`, optional
        Random number generator, pass a seeded one for reproducible chains.

    References
    ----------
    [HHP2024] A. Habring, M. Holler, T. Pock. Subgradient Langevin methods for
        sampling from nonsmooth potentials. SIAM J. Math. Data Sci., 2024.
    """
    _check_potentials(x, f, K, g)
    space = g.domain

    tau = float(tau)
    r = np.sqrt(2 * tau)

    rng = kwargs.pop('rng', None)
    if rng is None:
        rng = np.random.default_rng()
    if kwargs:
        raise TypeError('unexpected keyword argument(s): {}'.format(kwargs))

    prox_tau_g = g.proximal(tau)
    grad_f = f.gradient

    for _ in range(niter):
        y = grad_f(K(x))
        tmp = x - tau * K.adjoint(y)
        if noise_inside_prox:
            tmp.lincomb(1, tmp, r, _standard_normal(space, rng))
            x.assign(prox_tau_g(tmp))
        else:
            x.lincomb(1, prox_tau_g(tmp), r, _standard_normal(space, rng))

        if callback is not None:
            callback(x)
        if dual_callback is not None:
            dual_callback(y)


def forward_backward_langevin(x, f, g, gamma, niter, callback=None, **kwargs):
    """Forward-backward Langevin algorithm for exp(-f(x) - g(x)).

    ``f`` is smooth (``f.gradient``), ``g`` has a proximal operator. One
    iteration takes a gradient step in ``f``, adds the diffusion and then takes
    a proximal step in ``g``:

        x_{n+1} = prox_{gamma g}(x_n - gamma grad f(x_n) + sqrt(2 gamma) xi_{n+1})

    Parameters
    ----------
    x : ``f.domain`` element
        Initial state of the Markov chain, updated in place.
    f : `Functional`
        Smooth potential, needs ``f.gradient``.
    g : `Functional`
        Potential with ``g.proximal``.
    gamma : positive float
        Step size.
    niter : int
        Number of iterations.
    callback : callable, optional
        Called with the current iterate after each iteration.

    Other Parameters
    ----------------
    rng : `numpy.random.Generator`, optional
        Random number generator, pass a seeded one for reproducible chains.
    """
    if not isinstance(f, Functional):
        raise TypeError('`f` {!r} is not a `Functional`'.format(f))
    if not isinstance(g, Functional):
        raise TypeError('`g` {!r} is not a `Functional`'.format(g))
    if f.domain != g.domain:
        raise TypeError('`f.domain` {!r} must equal `g.domain` {!r}'.format(f.domain, g.domain))
    space = f.domain
    if x not in space:
        raise TypeError('`x` {!r} is not in the domain of `f` {!r}'.format(x, space))

    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError('`gamma` must be positive, got {}'.format(gamma))
    r = np.sqrt(2 * gamma)

    rng = kwargs.pop('rng', None)
    if rng is None:
        rng = np.random.default_rng()
    if kwargs:
        raise TypeError('unexpected keyword argument(s): {}'.format(kwargs))

    grad_f = f.gradient
    prox_gamma_g = g.proximal(gamma)

    for _ in range(niter):
        x.lincomb(1, x, -gamma, grad_f(x))
        x.lincomb(1, x, r, _standard_normal(space, rng))
        prox_gamma_g(x, out=x)

        if callback is not None:
            callback(x)
