
"""

Author: LRP
Date: 22-01-1015

Description:
Full artillery for running the J-band Analysis technique on RSGs

All dependencies are contained within astropy
Code is* written to conform with PEP8 style guide

Usage:
within ipython:
    run rsjanal.py

Outside the interactive environment:

    python rsjanal.py

References:
Davies et al. (2010)
Gazak (2014) Thesis

*has been attempted to be

TODO:
-- Better modulisation of this parent script
-- Improve portability (particularly w.r.t model grids)

"""
from __future__ import print_function

import sys
sys.path.append("/home/lee/Work/RSG-JAnal/bin/.")

import matplotlib.pyplot as plt
import numpy as np
import time

from uncertainties import ufloat
from uncertainties import unumpy

import bestfit
import cc
import chisq
import contfit
import readdata
import resolution as res

nom = unumpy.nominal_values

# MCMC stuff:
import emcee
import corner
# import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from matplotlib.ticker import MaxNLocator

# Saving space on print statements
o = str('[INFO] ')
w = str('[WARNING] ')
e = str('[ERROR] ')


def trimspec(w1, w2, s2):
    """Trim s2 and w2 to match w1"""
    roi = np.where((w2 > w1.min()) & (w2 < w1.max()))[0]
    return w2[roi], s2[roi]


def defidx(w1):
    """
    Define regions/lines for diagnostic lines to compute chisq over
    This function is now twinned with chisq.chireg, any changes made here
    must be relected in chisq.chireg
    """
    # Only the cores of individual lines (with main regions)
    # idx = [np.where((w1 > 1.18205) & (w1 < 1.18330))[0]]  # MgI
    # # idx = [np.where((w1 > 1.18780) & (w1 < 1.18880))[0]]  # FeI
    # idx.append(np.where((w1 > 1.18780) & (w1 < 1.18990))[0])  # FeI & TiI
    # idx.append(np.where((w1 > 1.19447) & (w1 < 1.19530))[0])  # TiI
    # idx.append(np.where((w1 > 1.19650) & (w1 < 1.19965))[0])  # FeI, Si, Si
    # idx.append(np.where((w1 > 1.20250) & (w1 < 1.20350))[0])  # SiI
    # idx.append(np.where((w1 > 1.20780) & (w1 < 1.20870))[0])  # Mg
    # idx.append(np.where((w1 > 1.21000) & (w1 < 1.21070))[0])  # SiI

    # # Only the cores of individual lines
    idx = [np.where((w1 > 1.18205) & (w1 < 1.18330))[0]]  # MgI
    # idx = [np.where((w1 > 1.18780) & (w1 < 1.18880))[0]]  # FeI
    idx.append(np.where((w1 > 1.18780) & (w1 < 1.18880))[0])  # FeI
    idx.append(np.where((w1 > 1.18880) & (w1 < 1.18990))[0])  # TiI
    idx.append(np.where((w1 > 1.19447) & (w1 < 1.19530))[0])  # TiI
    idx.append(np.where((w1 > 1.19650) & (w1 < 1.19790))[0])  # FeI
    idx.append(np.where((w1 > 1.19780) & (w1 < 1.19878))[0])  # SiI
    idx.append(np.where((w1 > 1.19878) & (w1 < 1.19965))[0])  # SiI
    idx.append(np.where((w1 > 1.20250) & (w1 < 1.20350))[0])  # SiI
    idx.append(np.where((w1 > 1.21000) & (w1 < 1.21070))[0])  # SiI
    idx.append(np.where((w1 > 1.20780) & (w1 < 1.20870))[0])  # Mg
    # Plus some continuum:
    # idx.append(np.where((w1 > 1.2114) & (w1 < 1.21820))[0])  # continuum

    # # Main regions containing lines
    # idx = [np.where((w1 > 1.18000) & (w1 < 1.19200))[0]]  # MgI, FeI, TiI
    # idx.append(np.where((w1 > 1.19400) & (w1 < 1.20600))[0])  # TiI, FeI, SiIx3
    # idx.append(np.where((w1 > 1.20600) & (w1 < 1.21200))[0])  # SiI, MgI

    # Lines & continuum
    # idx = [np.where((w1 > 1.1879) & (w1 < 1.1899))[0]]
    # idx = [np.where((w1 > 1.18780) & (w1 < 1.18990))[0]]  # FeI & TiI
    # idx = [np.where((w1 > 1.18780) & (w1 < 1.18990))[0]]  # FeI & TiI
    # idx.append(np.where((w1 > 1.18780) & (w1 < 1.18880))[0])  # FeI
    # idx.append(np.where((w1 > 1.19447) & (w1 < 1.19986))[0])  # TiI, FeI, SiIx2
    # idx.append(np.where((w1 > 1.19447) & (w1 < 1.19528))[0])  # TiI
    # idx.append(np.where((w1 > 1.1965) & (w1 < 1.19986))[0])  # FeI, SiI & SiI
    # idx.append(np.where((w1 > 1.20700) & (w1 < 1.21150))[0])  # Mg & Si
    return idx


def bfcontour(chi, mpar, idx, bfpars):
    """Show contours for fit results."""
    """This shouldn't be in this main script."""
    xbf, zbf, gbf, tbf = bfpars
    xii, zi, gi, ti = idx
    xi, z, g, t = mpar
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    min1 = chi[xii, :, gi, :].min()
    n = (min1 + 1, min1 + 2, min1 + 3, min1 + 5, min1 + 10)
    ax1.contour(t, z, chi[xii, :, gi, :], n)
    ax1.scatter(tbf, zbf, marker='x', color='r', s=50)
    ax1.set_xlabel('Teff', fontsize=10)
    ax1.set_ylabel('[Z]', fontsize=10)

    min2 = chi[:, :, gi, ti].min()
    n = (min2 + 1, min2 + 2, min2 + 3, min2 + 5, min2 + 10)
    ax2.contour(xi, z, chi[:, :, gi, ti].T, n)
    ax2.scatter(xbf, zbf, marker='x', color='r', s=50)
    ax2.set_xlabel(r'$\xi$', fontsize=10)
    ax2.set_ylabel('[Z]', fontsize=10)

    min3 = chi[xii, :, :, ti].min()
    n = (min3 + 1, min3 + 2, min3 + 3, min3 + 5, min3 + 10)
    ax3.contour(g, z, chi[xii, :, :, ti], n)
    ax3.scatter(gbf, zbf, marker='x', color='r', s=50)
    ax3.set_xlabel('log (g)', fontsize=10)
    ax3.set_ylabel('[Z]', fontsize=10)


def outfiles():
    """Generate out-files for analysis."""
    t = time.gmtime()
    date = str(t[0]) + '-' + str(t[1]).zfill(2) + '-' + str(t[2]).zfill(2)
    # Write files with owave, ospec, mspec with names as filenames:
    for i, j in enumerate(bfspec):
        desc = '\nDescription\nObserved spctrum and bestfit model for '\
               + odata.id[i]
        head = ('Author: LRP\nDate: ' + date + desc)
        tmp = np.column_stack((owave, ospeccc[i], j))
        fname = 'output/' + odata.id[i]
        np.savetxt(fname + '.dat', tmp, header=head, fmt='%6.6f')
        print(o + 'Spectra written to file: {}.dat'.format(fname))

    # Save final parameters to text file:
    pname = 'output/' + raw_input('[INFO] Please enter an appropriate \
    filename in the format xxx-pars.dat\n') + '-pars'
    while pname == 'output/-pars':
        print(o + 'Please include an appropriate file name:')
        pname = 'output/' + raw_input('[INFO] Please enter an appropriate \
    filename in the format xxx-pars.dat\n') + '-pars'
    desc = '\nDescription\nBestfit parameters for input spectra\n\
    ID, micro, [Z], logg, Teff, err_micro, err_Z, err_logg, err_teff'
    head = ('Author: LRP\nDate: ' + date + desc)
    opars = np.column_stack((odata.id, np.round(pars, 5)))
    pars_mcmc_ = pars_mcmc.reshape(pars_mcmc.shape[0],
                                   pars_mcmc.shape[1]*pars_mcmc.shape[2])
    opars_mcmc = np.column_stack((odata.id, np.round(pars_mcmc_, 5)))
    np.savetxt(pname + '.dat', opars, header=head, fmt='%s')
    np.savetxt(pname + '-mcmc.dat', opars_mcmc, header=head, fmt='%s')
    print(o + 'Please note that this file needs a more descriptive header')
    return

# MCMC stuff:


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def lnprior(theta):
    mt, z, g, t = theta
    # These priors are a fantasy as I already implement priors for the chisq
    # calculation
    if 1. < mt < 5. and -1. < z < 0. and -1. < g < 1. and 3400 < t < 4400:
        return 0.0
    return -np.inf


def lnlike(theta, spec, sn):
    mt, z, g, t = theta
    points = (mod.mt, mod.z, mod.g, mod.t)
    # Index the chisq grid
    chi = bfobj.vchi

    # Nearest neighbour
    chidx = [find_nearest(i, j) for i, j in zip(points, theta)]
    like_check = -(chi[chidx[0], chidx[1], chidx[2], chidx[3]])/2.
    # Interpolate the grid
    if like_check is np.ma.masked:
        like = -np.inf
    else:
        like = -(interpn(points, chi, theta, fill_value=-np.inf))/2.

    # like = -(interpn(points, chi, theta, fill_value=-np.inf))/2.
    # print(like)
    # Compute the chisq in house:
    # model = mgrid[chidx[0], chidx[1], chidx[2], chidx[3]]
    # if not np.isfinite(model):
    #     return -np.inf

    # # prepare model
    # cft = contfit.contfit(resi, owave, model, spec)[0]
    # mscale = model / cft(owave)
    # mcc, s = cc.ccshift(spec, mscale, owave)

    # like = -chisq(spec, 1/sn, mcc)

    # return np.sum(np.log(like))
    # return np.sum(lnlike)
    # return like
    if not np.isfinite(like) or np.abs(like) == 0.0:
        return -np.inf
    return np.sum(like)


def lnprob(theta, spec, sn):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    # Prior + likelihood function
    return lp + lnlike(theta, spec, sn)


def run_mcmc(spec, sn, ndim, nwalkers, burnin, nsteps, nout):

    np.random.seed(123)

    # Set up the sampler.
    # average micro, Z=LMC, logg=-0.1, Teff=4000
    # Z and logg cannot be set ==0.0 as pos will also be zero, always
    guess = [3.5, -0.3, -0.1, 3900]

    # Initial positions of the walkers in parameter space
    # Make sure this guess is somewhere physical!
    pos = np.array([guess + 0.001*np.array(guess)*np.random.randn(ndim)
                    for i in range(nwalkers)])

    # lnprob - A function that takes a vector in the parameter space as input
    # and returns the natural logarithm of the posterior probability
    # for that position.
    # args - (optional) A list of extra positional arguments for lnprob.
    # lnprob will be called with the sequence lnprob(p, *args, **kwargs).
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(spec, sn), threads=1)

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
    # pl.clf()

    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 8))
    label = [r'$\xi$ (km/s)', '[Z] (dex)',
             'log g (c.g.s)', r'T$_{\rm eff}$ (K)']

    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(label[i])

    fig.tight_layout(h_pad=0.0)
    # out1 = 'NGC2100_line-time.png'
    # fig.savefig(out1)

    # Make the triangle plot.
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
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
    fig = corner.corner(samples, labels=label,
                        truths=['NaN', 'NaN', 'NaN', 'NaN'],
                        figsize=(8, 8))
    # out2 = 'NGC2100_line-triangle-v2.png'
    # fig.savefig(out2)

    return mt_mcmc, z_mcmc, g_mcmc, t_mcmc


def run_fit(spec, sn):
    # ndim     = number of parameters
    # nwalkers = number of walkers
    # burnin   = number of burnin in steps
    # nsteps   = total number of steps
    # nout     = output every nout

    ndim, nwalkers, burnin, nsteps, nout = 4, 300, 500, 1000, 10
    print(o + 'ndim, nwalkers, burnin, nsteps, nout =',
          ndim, nwalkers, burnin, nsteps, nout)

    sampler, pos, lnp = run_mcmc(spec, sn, ndim, nwalkers,
                                 burnin, nsteps, nout)

    results = make_plots(sampler, ndim, burnin, pos, lnp)
    return sampler, pos, lnp, results

# Start proceedings:
print('[INFO] Reading in model grid ...')
then = time.time()
# mod = readdata.ReadMod(
#     'models/MODELSPEC_2013sep12_nLTE_R10000_J_turb_abun_grav_temp-int.sav')
mod = readdata.ReadMod(
    'models/MODELSPEC_251114_J_nlte_R10000_J_turb_abun_grav_temp-int.sav')

print('[INFO] Time taken: {}s'.format(round(time.time() - then, 3)))

print('[INFO] Read observed spectra from file:')
print('[INFO] Please ensure all files are ordered similarly!')

# These files should be user input

# odata = readdata.ReadObs('../ngc6822/Spectra/N6822-spec-24AT-sam-norm.txt',
#                          'input/NGC6822-janal-input.txt',
#                          mu=ufloat(23.3, 0.05))
# odata = readdata.ReadObs('input/NGC6822-janal-nspec.v1.txt',
#                          'input/NGC6822-janal-input.txt',
#                          mu=ufloat(23.3, 0.05))

# n6822 = readdata.ReadObs('input/NGC300-janal-nspec-lum.v2.txt',
#                          'input/NGC300-janal-info.txt',
#                          mu=ufloat(23.5, 0.05))

# odata = readdata.ReadObs('input/NGC2100-janal-nspec.v3.txt',
#                          'input/NGC2100-janal-info.txt',
#                          mu=ufloat(18.5, 0.05))

# odata = readdata.ReadObs('input/NGC2100-bad1-nspec.txt',
#                          'input/NGC2100-janal-info-bad1.txt',
#                          mu=ufloat(18.5, 0.05))

# odata = readdata.ReadObs('input/NGC55-nspec-galspec-v1.txt',
#                          'input/NGC55-janal-info-gal.txt',
#                          mu=ufloat(26.58, 0.11))  # Tanka et al. 2011

# odata = readdata.ReadObs('input/ngc55-35-r5.txt',
#                          'input/NGC55-janal-info-ngc35.txt',
#                          mu=ufloat(26.7, 0.11))  # Rolf's email
odata = readdata.ReadObs('input/NGC55-nspec-sample-v5.txt',
                         'input/NGC55-janal-info-sample.txt',
                         mu=ufloat(26.7, 0.11))  # Rolf's email

# Cluster spectrum:
# odata = readdata.ReadObs('input/NGC2100-nspec-cspec.v4.txt',
#                          'input/NGC2100-janal-info-cluster.txt',
#                          mu=ufloat(18.5, 0.05))

# Fake Input:
# odata = readdata.ReadObs('input/Fake-spec-Fakespec-t1.txt',
#                          'input/Fake-info-Fakespec-t1.txt',
#                          mu=ufloat(23.3, 0.05))

# odata = readdata.ReadObs('input/Fake-spec-Fakespec-tres-sn150-all.txt',
#                          'input/Fake-info-Fakespec-tres-sn150-all.txt',
#                          mu=ufloat(23.3, 0.05))
# odata = readdata.ReadObs('input/Fake-spec-NGC6822-t1sn150.txt',
#                          'input/Fake-info-NGC6822-t1sn150.txt',
#                          mu=ufloat(23.3, 0.05))

print('[INFO] Observations and models trimed to the 1.155-1.225mu region')
owave, ospec = trimspec(mod.twave, odata.wave, odata.spec)
ospec = ospec.T
# For testing purposes:
# ospec = ospec.T[0:1]

print('[INFO] Resampling model grid ...')
then = time.time()
mssam = contfit.specsam(mod.twave, mod.tgrid, owave)
print('[INFO] Time taken: {}s'.format(round(time.time() - then, 3)))

# In this for loop we need more than one spectrum! -- change this!

# Initialise things we want to keep:
# This should be one object for each target
bfclass = [0]*len(ospec)
ospeccc = [0]*len(ospec)
mscale = [0]*len(ospec)
chi = [0]*len(ospec)
cft = [0]*len(ospec)
pars_mcmc = [0]*len(ospec)
for i, j in enumerate(ospec):
    print('[INFO] Observed spectrum: {}'.format(odata.id[i]))
    print('[INFO] Degrading model grid to resolution of observation')
    then = time.time()
    # clip s/n at 150.
    sn = 150. if odata.sn[i] >= 150. else odata.sn[i]
    resi = float(odata.res[i])
    mdeg = res.degrade(owave, mssam, mod.res, resi)
    print('[INFO] Time taken: {}s'.format(round(time.time() - then, 3)))

    print('[INFO] Shift spectrum onto rest wavelength:')
    mspec1 = mdeg[10, 4, 4, 6]
    s1, arr_ = cc.crossc(mspec1, j, width=40)
    print('[INFO] Cross-Correlation shift = {}'.format(s1))
    # tmp_spec, s1 = cc.ccshift(mspec1, j, owave, quiet=False, width=40)
    s = odata.spec[:, i]  # Why is 'j' not used here?
    srest, s2 = cc.ccshift(mspec1, s, odata.wave, shift1=s1, quiet=False)
    owave1, spec = trimspec(mod.twave, odata.wave, srest)
    # spec, s1 = cc.ccshift(mspec1, j, owave, quiet=False, width=40)

    # Restrict logg range
    mgrid = bestfit.clipg(mdeg, mod.t, mod.g, nom(odata.L[i]))
    # mgrid = bestfit.clipmt(mgrid, mod.mt)
    # mgrid = mdeg

    # Index over which to calculate chi-sqared
    line_idx = defidx(owave)
    print('[INFO] Compute chi-squared grid ...')
    then = time.time()
    # Need to pass the the model grid and a observed spectrum class:
    chi[i], mscale[i], cft[i] = chisq.chigrid(mgrid, spec, owave,
                                              resi, line_idx, sn)
    chii = chi[i]  # / len(line_idx)
    print('[INFO] Time taken: {}s'.format(round(time.time() - then, 3)))

    print('------------------------------------')
    print('[INFO] Calcualte bestfit parameters ...')
    then = time.time()
    bfobj = bestfit.BestFit(chii, mod.head, odata.id[i])
    bfobj.showmin()
    bfobj.showavpar()
    # bfcontour(chii, (mod.mt, mod.z, mod.g, mod.t), bfobj.fi)
    # bfobj.showfin()
    print('[INFO] Time taken in seconds:', time.time() - then)
    print('------------------------------------')
    ospeccc[i] = spec
    bfclass[i] = bfobj
    sampler, pos, lnp, pars_mcmc[i] = run_fit(spec, sn)

bfspec = np.array([mscale[i][j.fi] for i, j in enumerate(bfclass)])
pars = np.array([(i.wpar, i.err) for i in bfclass])
pars = pars.reshape(len(pars), 8)
pars_mcmc = np.array(pars_mcmc)

# End game
outfiles()

# Unfinished and/or unused:


def bestfitoned():
    # bestfit parameters an average over the whole grid weighted by chisq's?
    # np.meshgrid?
    # x, y, w, z = np.meshgrid(mod.mt, mod.z, mod.g, mod.t)
    xichi = bfobj.fchi[:, 0, 5, 2]
    xxifine, yxifine = bestfit.onedfit(mod.head[0][0], xichi, 2, 10)
    print(xxifine[yxifine.argmin()])

    zchi = bfobj.fchi[5, :, 5, 2]
    xzfine, yzfine = bestfit.onedfit(mod.head[0][1], zchi, 2, 10)
    print(xzfine[yzfine.argmin()])

    gchi = bfobj.fchi[5, 0, :, 2]
    xgfine, ygfine = bestfit.onedfit(mod.head[0][2], gchi, 3, 10)
    print(xgfine[ygfine.argmin()])

    tchi = bfobj.fchi[5, 0, 5, :]
    xtfine, ytfine = bestfit.onedfit(mod.head[0][3], tchi, 3, 10)
    print(xtfine[ytfine.argmin()])


def checkcontfit():
    idx = bfclass[1].fi
    goodidx = (13, 2, 3, 5)
    # f = np.poly1d(cft[1][idx])
    mspec = mscale[1][idx]
    o = ospec[1]
    ox = owave
    plt.plot(ox, o, 'black')
    plt.plot(ox, mspec, 'r')
    plt.plot(ox, mscale[1][goodidx], 'blue')


def plot2(r1, in1, out1, n1, r2, in2, out2, n2):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    ax[0].scatter(r1[in1], r1[out1])
    ax[0].set_xlabel(n1 + ' In')
    ax[0].set_ylabel(n1 + ' Out')
    ax[1].scatter(r2[in2], r2[out2])
    ax[1].set_xlabel(n2 + ' In')
    ax[1].set_ylabel(n2 + ' Out')


def chivspar(vchi, (mt, z, g, t), (mtr, zr, gr, tr)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    ax1.plot(mtr, vchi[:, z, g, t])
    ax1.set_xlabel(r'$\xi$')
    ax2.plot(zr, vchi[mt, :, g, t])
    ax2.set_xlabel(r'[Z]')
    ax3.plot(gr, vchi[mt, z, :, t])
    ax3.set_xlabel(r'$log g$')
    ax4.plot(tr, vchi[mt, z, g, :])
    ax4.set_xlabel(r'$T_{eff}$')
    plt.show()

# zvst = [mod.t[bfclass[-1].vchi[6, z, 8, :].argmin()]
#         for z in xrange(len(mod.z))]
# zvsmt = [mod.mt[bfclass[-1].vchi[:, z, 8, 4].argmin()]
#          for z in xrange(len(mod.z))]
