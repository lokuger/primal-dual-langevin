"""TV-based image denoising (Sec. 5.2.2 of the paper).

Posterior proportional to exp(-||x - x_noisy||^2 / (2 sigma^2) - alpha TV(x))
for a 128 x 128 phantom with Gaussian noise. We draw N_ITER samples with the
primal-dual sampler for lambda in {1, 10, 100} and with Prox-Sub
(lambda = infinity), all with the same step size tau, and report sample mean
and pixel-wise standard deviation of the primal and dual iterates.

Each run is cached separately, so runs can be started in parallel via --runs.

Outputs (results/l2tv-images/):
    {primaldual,proxsub}_{mean,logstd,duallogstd}_<spec>.pdf    Figs. 5 and 6
    cbar_mmse.pdf, cbar_logstd.pdf, cbar_duallogstd.pdf
"""
import argparse
import os

import numpy as np
import odl
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr

from pdlangevin import (primal_dual_langevin, prox_sub_langevin, RunningMoments,
                        CallbackRunningMoments, CallbackAfterBurnIn)
from pdlangevin.plotting import save_image, save_colorbar, result_dir, data_path

IMAGE = 'phantom128.png'
STD_NOISE = 0.25
ALPHA = 0.1              # TV regularization parameter
LOG_TAU = -4.0
LOG_LAMBDAS = [0.0, 1.0, 2.0, np.inf]   # inf = Prox-Sub
LOG_N_ITER = 6.0
BURN_IN = 10**3          # as stated in the paper; the original figures used no burn-in
SEED_DATA = 2345
SEED_CHAINS = 1


def run_name(log_lamda):
    if np.isinf(log_lamda):
        return f'proxsub_{{}}_logtau{LOG_TAU}_logsamples{LOG_N_ITER}'
    return f'primaldual_{{}}_loglambda{log_lamda}_logtau{LOG_TAU}_logsamples{LOG_N_ITER}'


def problem():
    im = io.imread(data_path(IMAGE)).astype(float)
    space = odl.uniform_discr([0, 0], [1, 1], im.shape)
    x_true = space.element((im - im.min()) / (im.max() - im.min()))
    rng = np.random.default_rng(SEED_DATA)
    x_noisy = x_true + space.element(STD_NOISE * rng.normal(size=x_true.shape))
    g = 1 / (2 * STD_NOISE**2) * odl.solvers.L2NormSquared(space).translated(x_noisy)
    K = odl.Gradient(space, pad_mode='symmetric')
    f = ALPHA * odl.solvers.GroupL1Norm(K.range)
    return x_true, x_noisy, f, K, g


def sample(log_lamda, niter):
    x_true, x_noisy, f, K, g = problem()
    tau = 10**LOG_TAU
    # one generator per run, so that runs can be computed independently
    rng = np.random.default_rng([SEED_CHAINS, 0 if np.isinf(log_lamda) else 1 + int(10 * log_lamda)])
    rm, rm_dual = RunningMoments(), RunningMoments()
    callback = CallbackAfterBurnIn(CallbackRunningMoments(rm), burn_in=BURN_IN)
    dual_callback = CallbackAfterBurnIn(CallbackRunningMoments(rm_dual), burn_in=BURN_IN)
    x = x_noisy.copy()
    if np.isinf(log_lamda):
        print(f'Prox-Sub: tau * L_grad(g) = {tau * g.grad_lipschitz:.2e} (< 1 required)')
        prox_sub_langevin(x, f, K, g, tau, niter, callback, dual_callback, rng=rng)
    else:
        sigma = 10**log_lamda * tau
        print(f'Primal-dual, lambda = {10**log_lamda:g}: tau*sigma*||K||^2 = '
              f'{tau * sigma * K.norm(estimate=True)**2:.2e} (<= 1 required)')
        primal_dual_langevin(x, f, K, g, tau, sigma, 1, niter, callback, dual_callback, rng=rng)
    return {'x_true': x_true.asarray(), 'x_noisy': x_noisy.asarray(),
            'mean': rm.mean.asarray(), 'std': rm.std.asarray(),
            'dual_std': np.stack([p.asarray() for p in rm_dual.std])}


def dual_logstd(dual_std):
    """Log of the pixel-wise l2 norm over the (horizontal, vertical) components."""
    r = np.log(np.sqrt(np.sum(dual_std**2, axis=0)))
    # Prox-Sub: the subgradient is constant in one corner due to the boundary handling
    r[np.isneginf(r)] = r[np.isfinite(r)].min()
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', type=float, default=LOG_LAMBDAS,
                        help='log10(lambda) values to compute, "inf" for Prox-Sub (default: all)')
    parser.add_argument('--force', action='store_true', help='resample even if cached results exist')
    args = parser.parse_args()

    out = result_dir('l2tv-images')

    for log_lamda in args.runs:
        cache = os.path.join(out, run_name(log_lamda).format('results') + '.npz')
        if args.force or not os.path.exists(cache):
            np.savez(cache, **sample(log_lamda, int(10**LOG_N_ITER)))

    # plot all runs that are available, with shared color ranges
    res = {}
    for log_lamda in LOG_LAMBDAS:
        cache = os.path.join(out, run_name(log_lamda).format('results') + '.npz')
        if os.path.exists(cache):
            res[log_lamda] = dict(np.load(cache))
    if not res:
        return
    first = next(iter(res.values()))
    print(f'noisy PSNR = {psnr(first["x_true"], first["x_noisy"], data_range=1):.2f} dB')
    save_image(first['x_true'], os.path.join(out, 'ground_truth.pdf'))
    save_image(first['x_noisy'], os.path.join(out, 'noisy_observation.pdf'))

    logstd = {k: np.log(r['std']) for k, r in res.items()}
    duallogstd = {k: dual_logstd(r['dual_std']) for k, r in res.items()}
    lo, hi = min(v.min() for v in logstd.values()), max(v.max() for v in logstd.values())
    dlo, dhi = min(v.min() for v in duallogstd.values()), max(v.max() for v in duallogstd.values())
    for log_lamda, r in res.items():
        name = run_name(log_lamda)
        print(f'{name.format("mean")}: MMSE PSNR = {psnr(r["x_true"], r["mean"], data_range=1):.2f} dB')
        save_image(r['mean'], os.path.join(out, name.format('mean') + '.pdf'))
        save_image(logstd[log_lamda], os.path.join(out, name.format('logstd') + '.pdf'), lo, hi)
        save_image(duallogstd[log_lamda], os.path.join(out, name.format('duallogstd') + '.pdf'), dlo, dhi)
    save_colorbar(os.path.join(out, 'cbar_mmse.pdf'), 0, 1)
    save_colorbar(os.path.join(out, 'cbar_logstd.pdf'), lo, hi)
    save_colorbar(os.path.join(out, 'cbar_duallogstd.pdf'), dlo, dhi)
    print(f'Figures written to {out}')


if __name__ == '__main__':
    main()
