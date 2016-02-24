"""
    Author: LRP
    Date: 03-02-2016
    Description:

"""

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def chisq(obs, err, mod):
    return np.sum(((obs - mod)**2) / err**2)

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def lnprior(theta):
    mt, z, g, t = theta
    # Start with no prior on logg for simplicity
    # Even though it isn't stated here, there is a prior on logg as it gets
    # clipped before it is passes to this function
    if 1. < mt < 5. and -1. < z < 1. and -1. < g < 1. and 3400 < t < 4400:
        return 0.0
    return -np.inf


def lnlike(theta, ospeccc, sn):
    mt, z, g, t = theta
    parameters = (mod.mt, mod.z, mod.g, mod.t)
    chidx = [find_nearest(i, j) for i, j in zip(parameters, theta)]
    # Or just index the chisq grid
    chi = bfobj.vchi
    # like = np.exp(-chi)
    # lnlike = -chi
    like = -chi[chidx[0], chidx[1], chidx[2], chidx[3]]

    # lnlike = -chisq(ospeccc, 1/sn, model)

    # return np.sum(np.log(like))
    # return np.sum(lnlike)
    if not np.isfinite(like):
        return -np.inf
    return like


def lnprob(theta, ospeccc, sn):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    # Prior + likelihood function
    return lp + lnlike(theta, ospeccc, 1./sn)


def run_mcmc(ospeccc, sn, ndim, nwalkers, burnin, nsteps, nout):

    np.random.seed(123)

    # Set up the sampler.
    # average micro, Z=LMC, logg=0.0, Teff=4000
    guess = [3.5, -0.3, 0.2, 4000]

    # How much should the initial positions alter??
    # Initial positions of the walkers in parameter space
    pos = np.array([guess + 0.01*np.array(guess)*np.random.randn(ndim)
                    for i in range(nwalkers)])

    # lnprob - A function that takes a vector in the parameter space as input
    # and returns the natural logarithm of the posterior probability
    # for that position.
    # args - (optional) A list of extra positional arguments for lnprob.
    # lnprob will be called with the sequence lnprob(p, *args, **kwargs).
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(ospeccc, 1./sn), threads=1)

    # Clear and run the production chain.
    print("Running MCMC...")

    state = None
    while sampler.iterations < nsteps:
        pos, lnp, state = sampler.run_mcmc(pos, nout, rstate0=state)

    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
    return sampler, pos, lnp


def make_plots(sampler, ndim, burnin, pos, lnp):
    # pl.clf()

    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 8))
    label = ['xi (km/s)', '[Z]', 'log g', 'T_{eff}']

    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(label[i])

    fig.tight_layout(h_pad=0.0)
    # out1 = 'NGC2100_line-time.png'
    # fig.savefig(out1)

    # Make the triangle plot.
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=['xi (km/s)', '[Z]', 'log g', 'T_{eff}'],
                        truths=['NaN', 'NaN', 'NaN', 'NaN'],
                        figsize=(8, 8))
    # out2 = 'NGC2100_line-triangle-v2.png'
    # fig.savefig(out2)

    # Compute the quantiles.
    mt_mcmc, z_mcmc, \
        g_mcmc, t_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                 axis=0)))
    print('[INFO] param median +1sig -1sig 1sig_upper 2sig_upper  3sig_upper')
    print(np.round(mt_mcmc, 3))
    print(np.round(z_mcmc, 3))
    print(np.round(g_mcmc, 3))
    print(np.round(t_mcmc, 3))


def run_fit(ospeccc, sn):
    # ndim     = number of parameters
    # nwalkers = number of walkers
    # burnin   = number of burnin in steps
    # nsteps   = total number of steps
    # nout     = output every nout

    ndim, nwalkers, burnin, nsteps, nout = 4, 100, 100, 400, 100
    print('ndim, nwalkers, burnin, nsteps, nout =',
          ndim, nwalkers, burnin, nsteps, nout)

    sampler, pos, lnp = run_mcmc(ospeccc, sn, ndim, nwalkers,
                                 burnin, nsteps, nout)

    make_plots(sampler, ndim, burnin, pos, lnp)
    return sampler, pos, lnp

