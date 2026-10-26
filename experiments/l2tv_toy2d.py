"""Two-pixel TV denoising toy example (Sec. 5.2.1 of the paper).

The posterior on R^2 has density proportional to
    exp(-||x - x_noisy||^2 / (2 sigma^2) - alpha |x_2 - x_1|).
We run N_CHAINS parallel chains of the primal-dual sampler for several lambda
and of Prox-Sub (lambda = infinity), and track the Wasserstein-2 distance of
the empirical distribution of the chains to the posterior, which in 2D can be
computed by numerical integration on a grid.

Outputs (results/l2tv-2d/):
    scatter_plots.png            Fig. 3
    wasserstein_plots_time.pdf   Fig. 4
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import odl
import ot

from pdlangevin import primal_dual_langevin, prox_sub_langevin, CallbackEvery
from pdlangevin.plotting import result_dir

X_TRUE = [3.0, 1.0]
STD_NOISE = 0.5
ALPHA = 10.0             # TV regularization parameter
TAU = 1e-3 * STD_NOISE**2
N_ITER = 10**4           # simulated time T = N_ITER * TAU
N_W2 = 20                # number of W2 evaluations, equidistant in [0, T]
LAMBDAS = [10, 100, 1000]
N_CHAINS = 10**4
GRID = (-1, 4, 100)      # posterior is evaluated on GRID x GRID
SEED = 12345678


def setup(n_chains, rng):
    x_noisy = np.array(X_TRUE) + STD_NOISE * rng.normal(size=2)
    # n_chains independent copies of the 2-pixel problem, stacked along axis 0
    space = odl.uniform_discr(min_pt=[1, 0], max_pt=[n_chains, 1], shape=[n_chains, 2],
                              dtype='float32', nodes_on_bdry=True)
    g = 1 / (2 * STD_NOISE**2) * odl.solvers.L2NormSquared(space).translated(
        space.element(np.tile(x_noisy, (n_chains, 1))))
    K = odl.PartialDerivative(space, axis=1, pad_mode='symmetric')
    f = ALPHA * odl.solvers.L1Norm(K.range)
    return x_noisy, space, f, K, g


def posterior_on_grid(x_noisy):
    t = np.linspace(GRID[0], GRID[1], GRID[2])
    x1, x2 = np.meshgrid(t, t, indexing='ij')
    nld = (((x1 - x_noisy[0])**2 + (x2 - x_noisy[1])**2) / (2 * STD_NOISE**2)
           + ALPHA * np.abs(x2 - x1))
    dens = np.exp(-nld)
    dens /= dens.sum()
    return x1, x2, nld, dens


def run_name(lamda):
    return 'proxsub' if np.isinf(lamda) else f'lambda{lamda:g}'


def sample(lamda, n_chains):
    """Run one sampler (Prox-Sub for lamda = inf) and track W2 to the posterior."""
    x_noisy, space, f, K, g = setup(n_chains, np.random.default_rng(SEED))
    x1, x2, _, dens = posterior_on_grid(x_noisy)
    grid_points = np.stack([x1.ravel(), x2.ravel()], axis=1)
    grid_weights = dens.ravel()
    name = run_name(lamda)

    def w2_to_posterior(x):
        x = x.asarray().astype(float)
        w2 = np.sqrt(ot.emd2(np.full(len(x), 1 / len(x)), grid_weights,
                             ot.dist(x, grid_points), numItermax=10**6))
        print(f'  {name}: W2 = {w2:.4f}', flush=True)
        return w2

    # one generator per run, so that runs can be computed independently
    rng = np.random.default_rng([SEED, (LAMBDAS + [np.inf]).index(lamda)])
    print(f'{name}: {N_ITER} steps with {n_chains} chains...', flush=True)
    x = space.zero()
    cb = CallbackEvery(w2_to_posterior, N_ITER // N_W2)
    if np.isinf(lamda):
        prox_sub_langevin(x, f, K, g, TAU, N_ITER, callback=cb, rng=rng)
    else:
        print(f'  tau*sigma*||K||^2 = {TAU * lamda * TAU * K.norm(estimate=True)**2:.2e} (<= 1 required)')
        primal_dual_langevin(x, f, K, g, TAU, lamda * TAU, 1, N_ITER, callback=cb, rng=rng)
    return {'x_noisy': x_noisy, 'samples': x.asarray(), 'w2': np.array(cb.results)}


def principal_axes(cov):
    ev, pc = np.linalg.eigh(cov)
    s = np.sign(pc[1, :])  # point the arrows upwards in the plot
    return s * np.sqrt(ev) * pc[0, :], s * np.sqrt(ev) * pc[1, :]


def plot_scatter(res, path):
    x1, x2, nld, dens = posterior_on_grid(res['x_noisy'])
    pts = np.stack([x1.ravel(), x2.ravel()], axis=1)
    m = dens.ravel() @ pts
    C = (pts - m).T @ ((pts - m) * dens.ravel()[:, None])

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
    names = [f'lambda{lamda}' for lamda in LAMBDAS] + ['proxsub']
    titles = [rf'Primal-dual, $\lambda$={lamda}' for lamda in LAMBDAS] + [r'Prox-Sub ("$\lambda=\infty$")']
    for a, name, title in zip(ax, names, titles):
        s = res[f'{name}_samples']
        sm = s.mean(axis=0)
        a.contourf(x1, x2, nld, levels=20)
        a.scatter(s[:, 0], s[:, 1], s=1, c='darkgray')
        a.scatter(*sm, s=50, c='black', marker='*')
        a.quiver([sm[0]] * 2, [sm[1]] * 2, *principal_axes(np.cov(s, rowvar=False)), color='black', scale=2.5)
        a.scatter(*m, s=50, c='blue', marker='*')
        a.quiver([m[0]] * 2, [m[1]] * 2, *principal_axes(C), color='blue', scale=2.5)
        a.scatter(*res['x_noisy'], s=50, c='y', marker='^')
        a.set_xlim(GRID[:2])
        a.set_ylim(GRID[:2])
        a.set_title(title)
        a.set_aspect(1)
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_w2(res, path):
    fig = plt.figure(figsize=(4, 4))
    markers = ['*', 's', '^', 'x']
    t = TAU * (N_ITER // N_W2) * np.arange(1, N_W2 + 1)
    plt.semilogy(t, res['proxsub_w2'], marker=markers[-1], label=r'Prox-Sub ("$\lambda = \infty$")')
    for lamda, marker in zip(LAMBDAS, markers):
        plt.semilogy(t, res[f'lambda{lamda}_w2'], marker=marker, label=rf'$\lambda =$ {lamda}')
    plt.legend()
    plt.xlabel(r'SDE time $t$')
    plt.ylabel(r'$W_2(\hat \mu_t,\mu^\ast)$')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', type=float, default=LAMBDAS + [np.inf],
                        help='lambda values to compute, "inf" for Prox-Sub (default: all)')
    parser.add_argument('--force', action='store_true', help='resample even if cached results exist')
    args = parser.parse_args()

    out = result_dir('l2tv-2d')
    for lamda in args.runs:
        cache = os.path.join(out, f'results_{run_name(lamda)}.npz')
        if args.force or not os.path.exists(cache):
            np.savez(cache, **sample(lamda, N_CHAINS))

    # plot once all runs are available
    caches = {run_name(l): os.path.join(out, f'results_{run_name(l)}.npz') for l in LAMBDAS + [np.inf]}
    if not all(os.path.exists(c) for c in caches.values()):
        return
    res = {}
    for name, cache in caches.items():
        r = np.load(cache)
        res['x_noisy'] = r['x_noisy']
        res[f'{name}_samples'], res[f'{name}_w2'] = r['samples'], r['w2']
    plot_scatter(res, os.path.join(out, 'scatter_plots.png'))
    plot_w2(res, os.path.join(out, 'wasserstein_plots_time.pdf'))
    print(f'Figures written to {out}')


if __name__ == '__main__':
    main()
