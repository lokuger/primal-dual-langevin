"""1D Gaussian toy example (Fig. 1 and Sec. 5.1 of the paper).

f(y) = y^2 / (2 c_f), g(x) = x^2 / (2 c_g), K = k, so the target is
N(0, c_f c_g / (c_g k^2 + c_f)). We run the primal-dual sampler for several
lambda = sigma / tau with sigma tau k^2 = c, and forward-backward Langevin
(the limit lambda = infinity) for comparison.

Outputs (results/gaussians1d/):
    histograms_xmarg_no_stationaries.pdf   Fig. 1
    joint_dist_lambda{1,10,100}.pdf        Fig. 2
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import odl
from matplotlib import cm

from pdlangevin import primal_dual_langevin, forward_backward_langevin
from pdlangevin.plotting import result_dir

C_F, C_G, K_SCALAR = 1.0, 2.0, 1.5
LAMBDAS = [1, 10, 100]
C_STEP = 1e-4            # sigma * tau * k^2; c <= 1 is needed for stability
N_SAMPLES = 10**6
SEED = 123456

# variances of target and of the stationary solution of the primal-dual SDE
C_H = C_F * C_G / (C_G * K_SCALAR**2 + C_F)
C_H_DUAL = K_SCALAR**2 / C_F**2 * C_H


def c_stat_primal(lamda):
    return ((lamda * C_F**2 * C_G**2 + C_G * (C_F + C_G * K_SCALAR**2))
            / ((C_F + C_G * K_SCALAR**2) * (1 + lamda * C_F * C_G)))


def c_stat_dual(lamda):
    return (K_SCALAR**2 * lamda * C_G**2) / ((C_G * K_SCALAR**2 + C_F) * (1 + lamda * C_F * C_G))


def normal_density(t, var):
    return np.exp(-t**2 / (2 * var)) / np.sqrt(2 * np.pi * var)


class StoreScalar:
    """Write the scalar iterate into a preallocated array."""

    def __init__(self, n):
        self.values = np.empty(n)
        self.i = 0

    def __call__(self, x):
        self.values[self.i] = x[0]
        self.i += 1


def sample(n_samples):
    rng = np.random.default_rng(SEED)
    space = odl.uniform_discr([0], [1], (1,))
    K = K_SCALAR * odl.IdentityOperator(space)
    f = 1 / (2 * C_F) * odl.solvers.L2NormSquared(space)
    g = 1 / (2 * C_G) * odl.solvers.L2NormSquared(space)

    samples = {}
    for lamda in LAMBDAS:
        tau, sigma = np.sqrt(C_STEP / lamda) / K_SCALAR, np.sqrt(C_STEP * lamda) / K_SCALAR
        primal, dual = StoreScalar(n_samples), StoreScalar(n_samples)
        primal_dual_langevin(space.zero(), f, K, g, tau, sigma, 1, n_samples,
                             callback=primal, dual_callback=dual, rng=rng)
        samples[f'lambda{lamda}'] = np.stack([primal.values, dual.values])
        print(f'lambda = {lamda}: tau = {tau:.2e}, sigma = {sigma:.2e}, '
              f'sample var = {primal.values.var():.4f} (target {C_H:.4f}, stationary {c_stat_primal(lamda):.4f})')

    # lambda = infinity: forward step in f(Kx), backward step in g
    primal = StoreScalar(n_samples)
    forward_backward_langevin(space.zero(), f * K, g, C_STEP * K_SCALAR**2, n_samples,
                              callback=primal, rng=rng)
    samples['ula'] = np.stack([primal.values, K_SCALAR / C_F * primal.values])
    return samples


def plot_histograms(samples, out):
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    t = np.linspace(-4, 4, 200)
    for j, lamda in enumerate(LAMBDAS):
        ax[j].hist(samples[f'lambda{lamda}'][0], 50, density=True, color='gray')
        ax[j].set_title(r'$\lambda = \sigma/\tau$ = {}'.format(lamda))
    ax[-1].hist(samples['ula'][0], 50, density=True, color='gray')
    ax[-1].set_title('ULA')
    for a in ax:
        a.plot(t, normal_density(t, C_H), color='orangered', linewidth=2)
        a.set_xlim(-4, 4)
    fig.savefig(os.path.join(out, 'histograms_xmarg_no_stationaries.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_joint(s, lamda, path, ylabel=True):
    """Scatter plot of (X_n, Y_n) with marginal histograms and densities."""
    plt.rcParams.update({'font.size': 13.5})
    xmin, xmax, ymin, ymax, nbins = -2, 2, -3, 3, 50
    xx = np.linspace(xmin, xmax, 200)
    yy = np.linspace(ymin, ymax, 200)
    color = 'darkslategray'

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=(2, 1), height_ratios=(1, 3), wspace=0.03, hspace=0.03)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_c)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_c)
    ax_c.set_xlim([xmin - 0.1, xmax + 0.1])
    ax_c.set_ylim([ymin - 0.1, ymax + 0.1])

    # concentrated joint target: all mass on the line y = k x / c_f (zero elsewhere)
    ax_c.set_facecolor(cm.coolwarm(0.0))
    ax_c.scatter(s[0, ::1000], s[1, ::1000], c=color, s=10)  # every 1000th sample
    ax_c.plot(xx, K_SCALAR / C_F * xx, c='white', linewidth=1.5)

    # target marginals (dashed) and stationary marginals of the primal-dual SDE (solid)
    ax_x.plot(xx, normal_density(xx, C_H), '--', c='orangered', label=r'$\mu^\ast$')
    ax_y.plot(normal_density(yy, C_H_DUAL), yy, '--', c='orangered', label=r'$\nu^\ast$')
    ax_x.plot(xx, normal_density(xx, c_stat_primal(lamda)), c='k', label=r'$\mu^\lambda$')
    ax_y.plot(normal_density(yy, c_stat_dual(lamda)), yy, c='k', label=r'$\nu^\lambda$')

    # sample histograms, rescaled for the mass outside the plotted window
    Ix = (s[0] > xmin) & (s[0] < xmax)
    Iy = (s[1] > ymin) & (s[1] < ymax)
    hx, bx = np.histogram(s[0, Ix], bins=nbins, range=(xmin, xmax), density=True)
    hy, by = np.histogram(s[1, Iy], bins=nbins, range=(ymin, ymax), density=True)
    ax_x.bar(bx[:-1], Ix.mean() * hx, width=bx[1] - bx[0], alpha=0.6, align='edge', color=color, label=r'$X_k$')
    ax_y.barh(by[:-1], width=Iy.mean() * hy, height=by[1] - by[0], alpha=0.6, align='edge', color=color, label=r'$Y_k$')

    ax_x.legend(loc='upper right')
    ax_y.legend(loc='upper right')
    ax_x.tick_params(axis='x', labelbottom=False)
    ax_y.tick_params(axis='y', labelleft=False)
    ax_x.set_yticks([])
    ax_y.set_xticks([])
    ax_c.set_xticks([xmin, xmax])
    ax_c.set_yticks([ymin, ymax])
    ax_c.set_xlabel('Primal X')
    if ylabel:
        ax_c.set_ylabel('Dual Y')
    ax_x.set_title(rf'$\lambda = {lamda}$')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true', help='resample even if cached results exist')
    args = parser.parse_args()

    out = result_dir('gaussians1d')
    cache = os.path.join(out, 'samples.npz')
    if args.force or not os.path.exists(cache):
        np.savez(cache, **sample(N_SAMPLES))
    samples = dict(np.load(cache))

    plot_histograms(samples, out)
    for lamda in LAMBDAS:
        plot_joint(samples[f'lambda{lamda}'], lamda, os.path.join(out, f'joint_dist_lambda{lamda}.pdf'),
                   ylabel=lamda == LAMBDAS[0])
    print(f'Figures written to {out}')


if __name__ == '__main__':
    main()
