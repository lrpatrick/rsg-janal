"""
Author: LRP
Date: 22-01-1015
Description:
Full artillery for running the J-band Analysis technique on RSGs

All dependencies are contained within astropy
Code is* written to conform with PEP8 style guide

References:
Davies et al. (2010)
Gazak (2014) Thesis

*has been attempted to be

TODO:
-- Go through routines and find everything which needs to be updated

"""
from __future__ import print_function

import sys
sys.path.append("/home/lee/Work/RSG-JAnal/bin/.")
import time
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy

import bestfit
import cc
import chisq
import contfit
import readdata
import resolution as res

nom = unumpy.nominal_values
stddev = unumpy.std_devs


def trimspec(w1, w2, s2):
    """Trim s2 and w2 to match w1"""
    roi = np.where((w2 > w1.min()) & (w2 < w1.max()))[0]
    return w2[roi], s2[roi]


def defidx(w1):
    """Define regions/lines for diagnostic lines to compute chisq over"""
    idx = [np.where((w1 > 1.1879) & (w1 < 1.1899))[0]]
    idx.append(np.where((w1 > 1.19447) & (w1 < 1.19528))[0])
    idx.append(np.where((w1 > 1.1965) & (w1 < 1.19986))[0])
    idx.append(np.where((w1 > 1.20253) & (w1 < 1.20359))[0])
    idx.append(np.where((w1 > 1.20986) & (w1 < 1.21078))[0])
    return idx


# def plotspec(owave, ospec, mwave, mspec):
#     """Plot spectra in a paper-ready way"""
#     plt.ion()
#     plt.figure()
#     plt.plot(mwave, ospec, 'black', label='Obs')
#     plt.plot(mwave, mspec, 'r', label='BF model')
#     plt.legend(loc=4)
#     plt.xlabel(r'Wavelength ($\mu$m)')
#     plt.ylabel('Norm. Flux')
#     plt.show()

# Start the proceedings:
print('[INFO] Reading in model grid ...')
then = time.time()
mod = readdata.ReadMod(
    'models/MODELSPEC_2013sep12_nLTE_R10000_J_turb_abun_grav_temp-int.sav')
print('[INFO] Time taken: {}s'.format(round(time.time() - then, 3)))

print('[INFO] Read observed spectra from file:')
print('[INFO] Please ensure all files are ordered similarly!')
# n6822 = readdata.ReadObs('../ngc6822/Spectra/N6822-spec-24AT.v2-sam.txt',
#                          'input/NGC6822-janal-input.txt',
#                          mu=ufloat(23.3, 0.05))

n6822 = readdata.ReadObs('input/Fake-spec-NGC6822-test1.txt',
                         'input/Fake-info-NGC6822-test1.txt',
                         mu=ufloat(23.3, 0.05))

# Prep:
print('[INFO] Observations and models trimed to the 1.165-1.215mu region')
owave, ospec = trimspec(mod.twave, n6822.wave, n6822.nspec)
ospec = ospec.T

print('[INFO] Resampling model grid ...')
then = time.time()
mssam = contfit.specsam(mod.twave, mod.tgrid, owave)
print('[INFO] Time taken: {}s'.format(round(time.time() - then, 3)))

# In this for loop we need more than one spectrum! -- change this!

bfclass = []
chi = ([0]*np.shape(ospec)[0])
mscale = ([0]*np.shape(ospec)[0])
cft = ([0]*np.shape(ospec)[0])
ospeccc = ([0]*np.shape(ospec)[0])
for i, j in enumerate(ospec):
    print('[INFO] Degrading model grid ...')
    then = time.time()
    # clip s/n at 150.
    sn = 150. if n6822.sn[i] >= 150. else n6822.sn[i]
    resi = float(n6822.res[i])
    mdeg = res.degrade(owave, mssam, mod.res, resi)
    print('[INFO] Time taken:{}s'.format(round(time.time() - then, 3)))

    print('[INFO] Shift spectrum onto rest wavelength:')
    mspec1 = mdeg[0, 0, 0, 0]
    spec, s1 = cc.ccshift(mspec1, j, owave, quiet=False)
    ospeccc[i] = spec
    # Using diag. lines only:
    idx = defidx(owave)
    mgrid = readdata.cliptg(mdeg, mod.t.astype(float), mod.g, nom(n6822.L[i]))
    owavem = owave

    print('[INFO] Compute chi-squared grid ...')
    then = time.time()
    # Need to pass the the model grid and a observed spectrum class:
    chi[i], mscale[i], cft[i] = chisq.chigrid(mgrid, spec, owavem,
                                              resi, idx, sn)
    chii = chi[i]  # / 8.
    print('[INFO] Time taken: {}s'.format(round(time.time() - then, 3)))

    # Constrain the grid to reject unphysical log g's
    # -0.25 should be the step between grid values
    # gstep = np.abs(mod.g[0] - mod.g[1])
    # glow = nom(n6822.glow[1]) - 0.25
    # gup = nom(n6822.gup[1]) + 0.25
    # glow = nom(n6822.glow[i]) - gstep - 0.3  # test3
    # gup = nom(n6822.gup[i]) + gstep  # test3
    # mod.parlimit(glow, gup, 'GRAVS')
    # vfchi = chii[:, :, mod.parcut]
    vfchi = chii
    print('------------------------------------')
    print('[INFO] Calcualte bestfit parameters ...')
    then = time.time()
    bfobj = bestfit.BestFit(vfchi, mod.head)
    bfobj.showmin()
    # bfobj.showfin()
    print('[INFO] Time taken in seconds:', time.time() - then)
    print('------------------------------------')
    bfclass.append(bfobj)

bfspec = [mscale[i][j.fi] for i, j in enumerate(bfclass)]

# End game
t = time.gmtime()
date = str(t[0]) + '-' + str(t[1]).zfill(2) + '-' + str(t[2]).zfill(2)
# head = 'Author: LRP\nDate:' + date

# out1 = np.append(owave, np.array(ospeccc)).reshape(12, len(ospeccc[0])).T
# np.savetxt('obs-outspec.txt', out1, header=head)

# out2 = np.append(owave, np.array(bfspec)).reshape(12, len(bfspec[0])).T
# np.savetxt('mod-outspec.txt', out2, header=head)

# Unfinished and unused:


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


# Errors:
# import errors
# mspec = mcheat
# Add noise characteristic of the S/N ratio of the observations
# Guess at 0.01 for now ...
# Need a function to measure noise in observations and replicate it


# def err(mspec, mgrid, owave, ores):
#     """
#         Need to pass this everything that chisq.chigrid needs
#     """
#     epar = np.zeros((10, 4))
#     for i in xrange(10):
#         nmod = mspec + np.random.normal(0, 0.01, mspec.shape[0])
#         echi, eoscale = chisq.chigrid(mgrid, nmod, owave, ores)
#         epar[i] = bestfit.bf(echi, mod.par, quiet=True)
#     return epar

# then = time.time()
# epar = err(mraw, mod.grid, mod.wave, mod.res)
# print('[INFO] Time taken in seconds:', time.time() - then)
