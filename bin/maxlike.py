"""
Author: LRP
Date: 24-05-2016
Description:
Maximum-likelihood parameter extimation

"""
from __future__ import print_function

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interpn

# Saving space on print statements
o = str('[INFO] ')
w = str('[WARNING] ')
e = str('[ERROR] ')


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def lnprior(theta, priors):
    # Priors need to be specified in the control file
    mt, z, g, t = theta
    # Priors are also implemented by the logg clipping for the chisq
    # calculation
    # priors = np.array(([1.0, 5.0, -1., 0., -1.0, 1.0, 3400, 4400]))
    pmt, pz, pg, pt = priors.reshape(len(priors)/2, 2)

    if pmt[0] < mt < pmt[1] and\
       pz[0] < z < pz[1] and \
       pg[0] < g < pg[1] and \
       pt[0] < t < pt[1]:
        return 0.0
    return -np.inf


def lnlike(theta, spec, sn, mod, chigrid):
    mt, z, g, t = theta
    points = (mod.mt, mod.z, mod.g, mod.t)
    # Index the chisq grid
    # chi = bfobj.vchi
    chi = chigrid

    # Nearest neighbour
    chidx = [find_nearest(i, j) for i, j in zip(points, theta)]
    like_check = -(chi[chidx[0], chidx[1], chidx[2], chidx[3]])/2.
    # Interpolate the grid
    if like_check is np.ma.masked:
        like = -np.inf
    else:
        like = -(interpn(points, chi, theta, fill_value=np.inf))/2.

    if not np.isfinite(like) or np.abs(like) == 0.0:
        return -np.inf
    return np.sum(like)


def lnprob(theta, spec, sn, mod, chigrid, priors):
    lp = lnprior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    # Prior + likelihood function
    return lp + lnlike(theta, spec, sn, mod, chigrid)


def run_mcmc(spec, sn, mod, chigrid, priors, iguess,
             ndim, nwalkers, burnin, nsteps, nout):
    np.random.seed(123)

    # Set up the sampler
    # average micro, Z=LMC, logg=-0.1, Teff=4000
    # Z and logg cannot be set ==0.0 as pos will also be zero, always
    # iguess = np.array([3.0, -0.1, -0.1, 4000])
    iguess = np.array(iguess) + 0.001
    print(o + 'Initial guess:', iguess)

    # Initial positions of the walkers in parameter space
    # Make sure this iguess is somewhere physical!
    pos = np.array([iguess + 0.001*np.array(iguess)*np.random.randn(ndim)
                    for i in range(nwalkers)])

    # lnprob - A function that takes a vector in the parameter space as input
    # and returns the natural logarithm of the posterior probability
    # for that position.
    # args - (optional) A list of extra positional arguments for lnprob.
    # lnprob will be called with the sequence lnprob(p, *args, **kwargs).
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(spec, sn, mod, chigrid, priors),
                                    threads=1)

    # Clear and run the production chain.
    print(o + 'Running MCMC...')
    then = time.time()
    state = None
    while sampler.iterations < nsteps:
        pos, lnp, state = sampler.run_mcmc(pos, nout, rstate0=state)

    print(o + 'Time taken: {}s'.format(round(time.time() - then, 3)))
    print(o + 'Mean acceptance frac:', np.mean(sampler.acceptance_fraction))
    print(o + 'Autocorrelation time for each parameters:', sampler.acor)
    print(o + 'If autocorrelation time < run time increase samples')
    return sampler, pos, lnp


def make_plots(sampler, ndim, burnin, pos, lnp):
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 8))
    label = [r'$\xi$ (km/s)', '[Z] (dex)',
             'log g (c.g.s)', r'T$_{\rm eff}$ (K)']

    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(label[i])

    fig.tight_layout(h_pad=0.0)
    # Make the triangle plot.
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    # # Compute the quantiles.
    # mt_mcmc, z_mcmc, \
    #     g_mcmc, t_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    #                          zip(*np.percentile(samples, [16, 50, 84],
    #                              axis=0)))

    fig = corner.corner(samples, labels=label,
                        truths=['NaN', 'NaN', 'NaN', 'NaN'],
                        figsize=(8, 8))
    return


def run_fit(spec, sn, mod, chigrid, priors, iguess):
    # ndim     = number of parameters
    # nwalkers = number of walkers
    # burnin   = number of burnin in steps
    # nsteps   = total number of steps
    # nout     = output every nout

    ndim, nwalkers, burnin, nsteps, nout = 4, 300, 300, 600, 10
    print(o + 'ndim, nwalkers, burnin, nsteps, nout =',
          ndim, nwalkers, burnin, nsteps, nout)
    print(o + 'Basic Priors')
    print('Micro Turb.: {}'.format(priors[0:2]))
    print('Metallicity: {}'.format(priors[2:4]))
    print('Surface grav.: {}'.format(priors[4:6]))
    print('Temperature.: {}'.format(priors[6:]))

    sampler, pos, lnp = run_mcmc(spec, sn, mod, chigrid, priors, iguess,
                                 ndim, nwalkers, burnin, nsteps, nout)

    # make_plots(sampler, ndim, burnin, pos, lnp)

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    # Compute the quantiles.
    results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    print('[INFO] param median +1sig -1sig')
    print('Micro Turb.: {}'.format(np.round(results[0], 3)))
    print('Metallicity: {}'.format(np.round(results[1], 3)))
    print('Surface grav.: {}'.format(np.round(results[2], 3)))
    print('Temperature.: {}'.format(np.round(results[3], 3)))
    return sampler, pos, lnp, results
