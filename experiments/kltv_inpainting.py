"""Poisson inpainting with TV prior and non-negativity (Sec. 5.4 of the paper).

Half of the pixels are removed, the rest is observed with Poisson noise. The
potential is g(x) = masked KL(x) + indicator(x >= 0) and f(Kx) = alpha TV(x).
We compare four samplers:
    (A) primal-dual, diffusion after the prox      ('primal-dual-noise-outside')
    (B) primal-dual, diffusion inside the prox     ('primal-dual-noise-inside')
    (C) Prox-Sub, diffusion after the prox         ('prox-sub-noise-outside')
    (D) Prox-Sub, diffusion inside the prox        ('prox-sub-noise-inside')

Outputs (results/kltv-inpainting/):
    ground_truth.pdf, observed.pdf, <algo>_{sample,mean,logstd}.pdf,
    cbar_mean.pdf, cbar_logstd.pdf                                       Fig. 8
    negative_pixels.txt   fraction of negative pixels in the last sample (subcaptions of Fig. 8)
"""
import argparse
import os

import numpy as np
import odl
from skimage import io, transform

from pdlangevin import (primal_dual_langevin, prox_sub_langevin, RunningMoments,
                        CallbackRunningMoments, CallbackAfterBurnIn, MaskedKullbackLeibler)
from pdlangevin.plotting import save_image, save_colorbar, result_dir, data_path

IMAGE = 'cow-scratch.png'
N = 128
INTENSITY = 1000         # image is scaled to [1, INTENSITY] Poisson rates
REMOVE_FRAC = 0.5
ALPHA = 5e-4             # TV regularization parameter
TAU = 100.0
LAMBDA = 5e-10           # sigma = LAMBDA * TAU, so that tau * sigma * ||K||^2 <= 1
N_ITER = 10**4
BURN_IN = N_ITER // 10
SEED_DATA = 2345
SEED_CHAINS = 1

ALGOS = {   # name: (primal-dual?, noise inside prox?)
    'primal-dual-noise-outside': (True, False),
    'primal-dual-noise-inside': (True, True),
    'prox-sub-noise-outside': (False, False),
    'prox-sub-noise-inside': (False, True),
}


def problem():
    rng = np.random.default_rng(SEED_DATA)
    rgb = transform.resize(io.imread(data_path(IMAGE)), [N, N])
    im = 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]
    space = odl.uniform_discr([0, 0], [1, 1], im.shape)
    x_true = np.maximum(INTENSITY * (im - im.min()) / (im.max() - im.min()), 1.0)
    mask = rng.choice([0.0, 1.0], size=im.shape, p=[REMOVE_FRAC, 1 - REMOVE_FRAC])
    y_obs = np.maximum(1.0, mask * rng.poisson(x_true))
    x_true, mask, y_obs = space.element(x_true), space.element(mask), space.element(y_obs)

    g = MaskedKullbackLeibler(space, mask, y_obs)
    K = odl.Gradient(space, pad_mode='symmetric')
    f = ALPHA * odl.solvers.GroupL1Norm(K.range, exponent=2)
    return x_true, y_obs, f, K, g


def sample(niter):
    x_true, y_obs, f, K, g = problem()
    print(f'primal-dual: tau*sigma*||K||^2 = {TAU * LAMBDA * TAU * K.norm(estimate=True)**2:.2e}')
    res = {'x_true': x_true.asarray(), 'y_obs': y_obs.asarray()}
    for i, (name, (primal_dual, inside)) in enumerate(ALGOS.items()):
        print(f'{name}: {niter} steps...')
        rng = np.random.default_rng([SEED_CHAINS, i])
        rm = RunningMoments()
        callback = CallbackAfterBurnIn(CallbackRunningMoments(rm), burn_in=BURN_IN)
        x = y_obs.copy()
        if primal_dual:
            primal_dual_langevin(x, f, K, g, TAU, LAMBDA * TAU, 1, niter, callback,
                                 noise_inside_prox=inside, rng=rng)
        else:
            prox_sub_langevin(x, f, K, g, TAU, niter, callback, noise_inside_prox=inside, rng=rng)
        res[f'{name}_sample'] = x.asarray()
        res[f'{name}_mean'] = rm.mean.asarray()
        res[f'{name}_std'] = rm.std.asarray()
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true', help='resample even if cached results exist')
    args = parser.parse_args()

    out = result_dir('kltv-inpainting')
    cache = os.path.join(out, 'results.npz')
    if args.force or not os.path.exists(cache):
        np.savez(cache, **sample(N_ITER))
    r = dict(np.load(cache))

    save_image(r['x_true'], os.path.join(out, 'ground_truth.pdf'), 0, INTENSITY)
    save_image(r['y_obs'], os.path.join(out, 'observed.pdf'), 0, INTENSITY)
    logstd = {name: np.log(r[f'{name}_std']) for name in ALGOS}
    slo, shi = min(v.min() for v in logstd.values()), max(v.max() for v in logstd.values())
    lines = []
    for name in ALGOS:
        s = r[f'{name}_sample']
        lines.append(f'{name}: {(s < 0).mean() * 100:.2f}% negative pixels ({(s < 0).sum()} of {s.size})')
        save_image(s, os.path.join(out, f'{name}_sample.pdf'), 0, INTENSITY)
        save_image(r[f'{name}_mean'], os.path.join(out, f'{name}_mean.pdf'), 0, INTENSITY)
        save_image(logstd[name], os.path.join(out, f'{name}_logstd.pdf'), slo, shi)
    # intensity colorbar, shared by ground truth, observation, samples and means
    save_colorbar(os.path.join(out, 'cbar_mean.pdf'), 0, INTENSITY)
    save_colorbar(os.path.join(out, 'cbar_logstd.pdf'), slo, shi)
    with open(os.path.join(out, 'negative_pixels.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Figures written to {out}')


if __name__ == '__main__':
    main()
