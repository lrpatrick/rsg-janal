
"""

Author: LRP
Date: 10-06-2016

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
Gazak (2014)

*has been attempted to be

TODO:
-- Include model path in config files
-- Create class for "all of the things we want to keep" in the loop I have
    i.e. more object orientated
-- Config files should include luminosity not photometry
-- Improve portability (particularly w.r.t model grids)
"""
from __future__ import print_function

import sys
sys.path.append("/home/lee/Work/RSG-JAnal/bin/.")

import matplotlib.pyplot as plt
import numpy as np
import time
# from uncertainties import ufloat
from uncertainties import unumpy

import bestfit
import cc
import chisq
import contfit
import maxlike
import readdata
import resolution as res

nom = unumpy.nominal_values

# MCMC stuff: (Should be located elsewhere)
# import emcee
# import corner
# # import matplotlib.pyplot as plt
# from scipy.interpolate import interpn
# from matplotlib.ticker import MaxNLocator

# Saving space on print statements
o = str('[INFO] ')
w = str('[WARNING] ')
e = str('[ERROR] ')


def outfiles(config, owave, ospec, mspec, pars_mcmc):
    """Generate out-files for analysis."""
    t = time.gmtime()
    date = str(t[0]) + '-' + str(t[1]).zfill(2) + '-' + str(t[2]).zfill(2)

    # Write files with owave, ospec, mspec with names as filenames:
    desc = '\nDescription\nObserved spctrum and bestfit model for '\
           + config.out_name
    head = ('Author: LRP\nDate: ' + date + desc)
    tmp = np.column_stack((owave, ospec, mspec))
    fout = config.out_dir + '/' + config.out_name
    print(fout)
    np.savetxt(fout + '.dat', tmp, header=head, fmt='%6.6f')
    print(o + 'Spectra written to file: {}.dat'.format(fout))

    # Save final parameters to text file:
    desc = '\nDescription\nBestfit parameters for' + config.out_name + '\n\
    micro, err_up, err_down [Z], err_up, err_down\
    logg, err_up, err_down Teff, err_up, err_down'
    head = ('Author: LRP\nDate: ' + date + desc)

    # X^2-min
    # opars = np.column_stack((config.out_name, np.round(pars, 5)))
    # np.savetxt(fout + '.dat', opars, header=head, fmt='%s')

    # MCMC/max-likelihood
    opars_mcmc = np.ravel(pars_mcmc)
    np.savetxt(fout + '-mcmc.dat', opars_mcmc, header=head, delimiter=' ')
    return


def between(x, low, up):
    """Return points in x between low and up"""
    idx = [np.where((x > low) & (x < up))[0]]
    return idx


def cut_grid(grid, pars, priors):
    """Prime the grid with the priors i.e. replace spectra with np.nan's"""
    p = np.reshape(priors, (len(priors)/2, 2))
    cutgrid = np.copy(grid)

    for count, (par, prior) in enumerate(zip(pars, p)):
        throw = np.where((par < prior[0]) | (par > prior[1]))[0]
        sl = [slice(None)]*count + [throw]

        if np.shape(throw)[0] != 0:
            cutgrid[sl] = np.nan

    return cutgrid


def rsgparams(fconfig):
    """
    Main function to estimate stellar parameters from RSGs
    Routine:
    1. Read in configuration file
    2. Read in models
    3. ...
    """
    # Read configuration file
    print(o + 'Congifuration file used: {}'.format(fconfig))
    config = readdata.ReadConfig(fconfig)
    print(o + 'Output name for spectrum: {}'.format(config.out_name))
    # Read in model
    print('[INFO] Reading in model grid ...')
    then = time.time()
    mod = readdata.ReadMod(config.mod_path)
    print('[INFO] Time taken: {}s'.format(round(time.time() - then, 3)))

    print(o + 'Observations and models trimed to the 1.155-1.225mu region')
    owave, ospec = contfit.trimspec(mod.twave, config.wave, config.spec)

    print(o + 'Resampling model grid ...')
    then = time.time()
    # Would it not make more sense to do this when required?
    pars = (mod.mt, mod.z, mod.g, mod.t)
    cgrid = cut_grid(mod.tgrid, pars, config.priors)
    mssam = contfit.specsam(mod.twave, cgrid, owave)
    print(o + 'Time taken: {}s'.format(round(time.time() - then, 3)))

    print(o + 'Degrading model grid to resolution of observation')
    then = time.time()
    # clip s/n at 150.
    sn = 150. if config.snr >= 150. else config.snr
    resi = float(config.res)
    mdeg = res.degrade(owave, mssam, mod.res, resi)
    print(o + 'Time taken: {}s'.format(round(time.time() - then, 3)))
    # Remove large scale wiggles:
    # s = contfit.wiggles(config.wave, config.spec)

    print(o + 'Shift spectrum onto rest wavelength:')
    mspec1 = mdeg[10, 4, 4, 6]
    s1, arr_ = cc.crossc(mspec1, ospec, width=40)
    print(o + 'Cross-Correlation shift = {}'.format(s1))

    # Apply the shift from the trimmed specturm j to whole observed spectrum
    # to avioid interpolation residuals at ends of spectra
    # Only implement if significant shift is detected

    if np.abs(s1) < 0.01:
        print(o + 'Cross-Correlation consistent with zero. No shift applied')
        srest = config.spec
    else:
        srest, s2 = cc.ccshift(mspec1, config.spec, config.wave,
                               shift1=s1, quiet=False)

    owave1, spec = contfit.trimspec(mod.twave, config.wave, srest)

    # Restrict logg range
    print(o + 'Implement prior based on luminosity and mass of target')
    mgrid = bestfit.clipg(mdeg, mod.t, mod.g, nom(config.L),
                          config.mlow, config.mhigh)

    low, high = config.fit_regions.reshape(len(config.fit_regions)/2., 2).T
    line_idx = [between(owave, l, h) for l, h in zip(low, high)]

    lowc, highc = config.cc_regions.reshape(len(config.cc_regions)/2., 2).T
    cc_idx = [between(owave, l, h) for l, h in zip(lowc, highc)]
    print(o + 'Compute chi-squared grid ...')
    then = time.time()
    # Need to pass the the model grid and a observed spectrum class:
    cfitdof = 3
    chi, mscale, cft = chisq.chigrid(mgrid, spec, owave, resi, line_idx, sn,
                                     cc_idx, config.cfit, cfitdof)
    print(o + 'Time taken: {}s'.format(round(time.time() - then, 3)))

    print('------------------------------------')
    print(o + 'Calcualte bestfit parameters ...')
    then = time.time()
    bfobj = bestfit.BestFit(chi, mod.head, config.out_name)
    bfobj.showmin()
    bfobj.showavpar()
    # bfobj.bfcontour()
    # bfobj.showfin()
    print(o + 'Time taken in seconds:', time.time() - then)
    print('------------------------------------')
    sampler, pos, lnp, pars_mcmc = maxlike.run_fit(spec, sn, mod, bfobj.vchi,
                                                   config.priors, bfobj.fipar)

    theta = np.array(pars_mcmc)[:, 0]
    points = (mod.mt, mod.z, mod.g, mod.t)
    bfidx = [maxlike.find_nearest(i, j) for i, j in zip(points, theta)]
    bfspec = mscale[tuple(bfidx)]
    # pars = np.array(bfobj.wpar, bfobj.err)
    # pars = pars.reshape(len(pars), 8)
    outfiles(config, owave, spec, bfspec, pars_mcmc)
    return bfspec, config, owave, spec, pars_mcmc, bfobj  # , sampler, pos, lnp, mgrid

# fconfig = sys.argv[1]
# bfspec, config, owave, spec, bfspec, pars_mcmc, bfobj = rsgparams(fconfig)
