"""TGV-based image denoising (Sec. 5.3 of the paper).

We sample jointly in x = (u, v) from the density proportional to
    exp(-||u - u_noisy||^2 / (2 sigma^2)
        - alpha_1 ||grad u - v||_{2,1} - alpha_0 ||E v||_{2,1}),
with the primal-dual sampler, dualizing both TGV terms.

Outputs (results/l2tgv-images/):
    {phantom,tortoise}_noisy_observation.pdf, mean_<spec>.pdf,
    logstd_<spec>.pdf, cbar_logstd_<spec>.pdf                           Fig. 7
"""
import argparse
import os

import numpy as np
import odl
from odl.solvers import L2NormSquared, GroupL1Norm, SeparableSum, ZeroFunctional
from skimage import io, transform
from skimage.metrics import peak_signal_noise_ratio as psnr

from pdlangevin import primal_dual_langevin, RunningMoments, CallbackRunningMoments
from pdlangevin.plotting import save_image, save_colorbar, result_dir, data_path

N = 256
# per image: noise standard deviation, TGV regularization parameter alpha_1 and the
# prefix of the ground truth / observation file names
IMAGES = {
    'phantom': dict(std_noise=0.1, alpha=10.0, prefix='phantom'),
    'tortoise_noisy': dict(std_noise=0.25, alpha=2.0, prefix='tortoise', file='tortoise2.png'),
}
BETA = 1.5               # alpha_0 = BETA * alpha_1
LOG_LAMBDA = 4.0
LOG_N_ITER = 5.0
SEED_DATA = 2345
SEED_CHAINS = 1


def problem(name):
    p = IMAGES[name]
    U = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[N, N])
    if 'file' in p:
        im = transform.resize(io.imread(data_path(p['file']), as_gray=True), (N, N))
        x_true = U.element((im - im.min()) / (im.max() - im.min()))
    else:
        x_true = odl.phantom.tgv_phantom(U)
    rng = np.random.default_rng(SEED_DATA)
    x_noisy = x_true + U.element(p['std_noise'] * rng.normal(size=x_true.shape))

    G = odl.Gradient(U, method='forward', pad_mode='symmetric')
    V = G.range
    Dx = odl.PartialDerivative(U, 0, method='backward', pad_mode='symmetric')
    Dy = odl.PartialDerivative(U, 1, method='backward', pad_mode='symmetric')
    E = odl.operator.ProductSpaceOperator([[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx], [0.5 * Dy, 0.5 * Dx]])
    domain = odl.ProductSpace(U, V)
    # K(u, v) = (grad u - v, E v)
    K = odl.BroadcastOperator(odl.ReductionOperator(G, odl.ScalingOperator(V, -1)),
                              E * odl.ComponentProjection(domain, 1))
    f = SeparableSum(p['alpha'] * GroupL1Norm(V), p['alpha'] * BETA * GroupL1Norm(E.range))
    g = SeparableSum(1 / (2 * p['std_noise']**2) * L2NormSquared(U).translated(x_noisy),
                     ZeroFunctional(V))
    return x_true, x_noisy, f, K, g, domain, G


def sample(name, niter):
    x_true, x_noisy, f, K, g, domain, G = problem(name)
    op_norm = 1.1 * odl.power_method_opnorm(K)
    lamda = 10**LOG_LAMBDA
    tau, sigma = 1 / np.sqrt(lamda) / op_norm, np.sqrt(lamda) / op_norm
    rng = np.random.default_rng([SEED_CHAINS, list(IMAGES).index(name)])
    rm = RunningMoments()
    z = domain.element((x_noisy.copy(), G(x_noisy)))
    print(f'{name}: lambda = {lamda:g}, tau = {tau:.2e}, sigma = {sigma:.2e}, {niter} steps...')
    primal_dual_langevin(z, f, K, g, tau, sigma, 1, niter, callback=CallbackRunningMoments(rm), rng=rng)
    return {'x_true': x_true.asarray(), 'x_noisy': x_noisy.asarray(),
            'mean': rm.mean[0].asarray(), 'std': rm.std[0].asarray()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', nargs='+', choices=list(IMAGES), default=list(IMAGES))
    parser.add_argument('--force', action='store_true', help='resample even if cached results exist')
    args = parser.parse_args()

    out = result_dir('l2tgv-images')
    for name in args.images:
        spec = f'{name}_loglambda{LOG_LAMBDA}_logsamples{LOG_N_ITER}'
        cache = os.path.join(out, spec + '.npz')
        if args.force or not os.path.exists(cache):
            np.savez(cache, **sample(name, int(10**LOG_N_ITER)))
        r = dict(np.load(cache))

        print(f'{name}: noisy PSNR = {psnr(r["x_true"], r["x_noisy"], data_range=1):.2f} dB, '
              f'MMSE PSNR = {psnr(r["x_true"], r["mean"], data_range=1):.2f} dB')
        logstd = np.log(r['std'])
        prefix = IMAGES[name]['prefix']
        save_image(r['x_true'], os.path.join(out, f'{prefix}_ground_truth.pdf'))
        save_image(r['x_noisy'], os.path.join(out, f'{prefix}_noisy_observation.pdf'))
        save_image(r['mean'], os.path.join(out, f'mean_{spec}.pdf'))
        save_image(logstd, os.path.join(out, f'logstd_{spec}.pdf'), logstd.min(), logstd.max())
        save_colorbar(os.path.join(out, f'cbar_logstd_{spec}.pdf'), logstd.min(), logstd.max())
    print(f'Figures written to {out}')


if __name__ == '__main__':
    main()
