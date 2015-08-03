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
-- update sigma in chisq calculation
-- optimise chisq calculation
-- Update the Teff in logg limit calculation
    -- Ben fits Teff first and uses that result to constrain logg
-- Go through routines and find everything which needs to be updated

"""
from __future__ import print_function

import sys
sys.path.append("/home/lee/Work/RSG-JAnal/bin/.")
import time
# import glob
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
t1 = time.time() - then
print('[INFO] Time taken in seconds:', t1)

print('[INFO] Read observed spectra from file:')
print('[INFO] Please ensure all files are ordered similarly!')
# n6822 = readdata.ReadObs('../ngc6822/Spectra/N6822-spec-24AT.v2-sam.txt',
#                          'input/NGC6822-janal-input.txt',
#                          mu=ufloat(23.3, 0.05))

n6822 = readdata.ReadObs('test/Fake-spec.txt',
                         'test/Fake-info.txt',
                         mu=ufloat(23.3, 0.05))

# Prep:
print('[INFO] Observations and models trimed to the 1.165-1.215mu region')
owave, ospec = trimspec(mod.twave, n6822.wave, n6822.nspec)
ospec = ospec.T


print('[INFO] Resampling model grid ...')
then = time.time()
mssam = contfit.specsam(mod.twave, mod.tgrid, owave)
print('[INFO] Time taken in seconds:', time.time() - then)

# ############################################################################
# Test2:
# print('[INFO] Degrading model grid ...')
# then = time.time()
# resi = float(n6822.res[0])
# mdeg = res.degrade(owave, mssam, float(mod.res), resi)
# print('[INFO] Time taken in seconds:', time.time() - then)
# ############################################################################
# In this for loop we need more than one spectrum! -- change this!

bfclass = []
chi = ([0]*np.shape(ospec)[0])
mscale = ([0]*np.shape(ospec)[0])
cft = ([0]*np.shape(ospec)[0])
ospeccc = ([0]*np.shape(ospec)[0])
for i, j in enumerate(ospec):
    print('[INFO] Degrading model grid ...')
    then = time.time()
    # resi = 10000  # test 1
    # clip s/n at 150.
    sn = 150. if n6822.sn[i] >= 150. else n6822.sn[i]
    resi = float(nom(n6822.res[i]))  # test 3
    mdeg = res.degrade(owave, mssam, float(mod.res), resi)  # test 3
    print('[INFO] Time taken:', round(time.time() - then, 3), 's')  # test3

    print('[INFO] Shift spectrum onto rest wavelength:')
    mspec1 = mdeg[0, 0, 0, 0]
    spec, s1 = cc.ccshift(mspec1, j, owave, quiet=False)
    ospeccc[i] = spec
    # Using diag. lines only:
    idx = defidx(owave)  # test 2 & 3
    mgrid = readdata.cliptg(mdeg, mod.t.astype(float), mod.g, nom(n6822.L[0]))
    owavem = owave

    print('[INFO] Compute chi-squared grid ...')
    then = time.time()
    # Need to pass the the model grid and a observed spectrum class:
    chi[i], mscale[i], cft[i] = chisq.chigrid(mgrid, spec, owavem,
                                              resi, idx, sn)
    chii = chi[i]  # / 8.
    print('[INFO] Time taken in seconds:', time.time() - then)

    # Constrain the grid to reject unphysical log g's
    # -0.25 should be the step between grid values
    gstep = np.abs(mod.g[0] - mod.g[1])
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
# np.savetxt('n6822-outspec.txt', out1, header=head)

# out2 = np.append(owave, np.array(bfspec)).reshape(12, len(bfspec[0])).T
# np.savetxt('mod-outspec.txt', out2, header=head)
# Find best few models:


def bestfew(grid, n):
    g1 = grid
    bfall = []
    for i in xrange(n):
        bfi = np.unravel_index(np.argmin(g1), g1.shape)
        # print(bfi, g1.min())
        bfall.append(bfi)
        g1 = np.ma.masked_equal(g1, g1[bfi])
    return bfall


def wparams(grid):
    """Define parameters using weights defined by chi-squared values"""
    mtv, zv, gv, tv = np.meshgrid(mod.head[0][0], mod.head[0][1],
                                  mod.head[0][2], mod.head[0][3],
                                  indexing='ij')
    # np.average(mtv, weights=np.exp(-vfchi/2))
    # # To recover parameters:
    # mtv[14, 18, 6, 5], zv[14, 18, 6, 5], gv[14, 18, 6, 5], tv[14, 18, 6, 5]
    n = 100
    bftop5 = bestfew(grid, n)
    mttop, ztop, gtop, ttop, chival = np.array([(mtv[i], zv[i], gv[i], tv[i],
                                                grid[i]) for i in bftop5]).T
    w = [np.exp(-i/2) for i in chival]
    wi = w / np.sum(w)
    wpars = [np.average(i[0], weights=wi)
             for i in zip((mttop, ztop, gtop, ttop))]
    # std = [np.std(i[0]) for i in zip((mttop, ztop, gtop, ttop))]
    # This method underestimates the errors as I think it only works with
    # the reduced chi-squared
    # for i, par in enumerate(zip((mttop, ztop, gtop, ttop))):
    #     tmp = n*np.sum(wi*(par - wpars[i])**2) / ((n - 1)*np.sum(wi))
    #     print(wpars[i], np.sqrt(tmp))
    return wpars

# x = np.zeros((len(n6822.glow), 4))
# errx = np.zeros((len(n6822.glow), 4))
# for i in xrange(len(n6822.glow)):
#     glow = np.log10(nom(n6822.glow[i])) - 0.25  # test3
#     gup = np.log10(nom(n6822.gup[i])) + 0.25  # test3
#     mod.parlimit(glow, gup, 'GRAVS')
#     x[i] = wparams(bfclass[i].vchi)
#     test1 = np.ma.masked_greater_equal(bfclass[i].fchi, bfclass[i].fchi.min() + 3.)
#     m = np.array(np.where(~test1.mask))
#     print(m.shape)
#     errx[i] = [np.std(mod.head[0][k][l]) for k, l in enumerate(m)]
#     maxi = [mod.head[0][k][l].max() for k, l in enumerate(m)]
#     mini = [mod.head[0][k][l].min() for k, l in enumerate(m)]
#     print(maxi, mini)


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


# Unfinished:
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
