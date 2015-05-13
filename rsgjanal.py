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
"""
from __future__ import print_function

import sys
sys.path.append("/home/lee/Work/RSG-JAnal/bin/.")
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
from uncertainties import ufloat
from uncertainties import unumpy

import contfit
import bestfit
import resolution as res

nom = unumpy.nominal_values
stddev = unumpy.std_devs


class ReadMod(object):
    """
        Simple class to read in model grid and alter the shape of it

        ... find out who desinged this grid! -- and destroy them?
        How could someone make the names and the grid a different order!!!

    """
    def __init__(self, savfile):
        self.all = readsav(savfile)
        self.grid = self.all['modelspec'][0][0]
        self.par = self.all['modelspec'][0][1]
        self.wave = self.all['modelspec'][0][2]
        self.pnames = self.par.dtype.names
        self.res = 10000.
        self.mt, self.z, self.g, self.t = self.gridorder()

    def gridorder(self):
        """Sort the model grid and get something more well structured out!"""
        # Order:
        teff = self.par.field('TEMPS')[0]  # 11
        abuns = self.par.field('ABUNS')[0]  # 19
        logg = self.par.field('GRAVS')[0]  # 9
        mt = self.par.field('TURBS')[0]  # 21
        return mt, abuns, logg, teff
        # return teff, abuns, logg, mt

    def parlimit(self, low, high, name):
        """Restrict a given model parameter, usually used with logg"""
        # Reset parameter first:
        par = self.findpar(name)
        self.par.field(name)[0] = par
        self.parcut = np.where((par > low) & (par < high))[0]
        self.par.field(name)[0] = self.par.field(name)[0][self.parcut]
        print('[INFO] ' + name + ' range constrained by ReadMod.parlimit')

    def findpar(self, name):
        """Look-up table style function using the names of parameters"""
        if name == 'GRAVS':
            par = self.g
        if name == 'ABUNS':
            par = self.z
        if name == 'TEMPS':
            par == self.t
        if name == 'TURBS':
            par = self.mt
        return par


class ReadObs(object):
    """
        ReadObs assumes a file with columns: 1. Wavelength 2-N: Spectra
        If observations are normalised used self.spec,
        if not, a simple median normalisation is apllied in self.nspec
    """
    def __init__(self, fspec, fphot, mu):
        self.fspec = fspec
        self.fphot = fphot
        self.mu = mu
        # self.fres = fres
        self.fall = np.genfromtxt(fspec)
        self.wave = self.fall[:, 0]
        self.spec = self.fall[:, 1:]
        self.nspec = self.spec / np.median(self.spec, axis=0)
        # self.wave, self.spec, self.misc = self.readspec(fspec)
        # Should read in a text file:
        # self.res = np.genfromtxt(fres)
        self.res = 3700.  # IFU 14: res at Ar1.21430: 3736 +/- 142
        self.phot = np.genfromtxt(fphot)
        self.mk = unumpy.uarray(self.phot[:, 15], self.phot[:, 16])
        self.L = self.luminosity()
        self.gup, self.glow = self.glimits()

    def glimits(self):
        """Set gravity limits based on some good assumptions about mass"""
        lsun = 3.846*10**26
        l = 10**self.L * lsun
        grav = 6.67*10**-11
        sb = 5.67*10**-8
        mup = 40*2*10**30
        mlow = 8*2*10**30
        t = 4000
        g = lambda M, T, L: ((4*np.pi*sb*grav*M*T**4) / L)*10**2
        return g(mup, t, l), g(mlow, t, l)

    def luminosity(self):
        """Calculate Luminosity based on Davies et al. (2013) correction"""
        a = ufloat(0.90, 0.11)
        b = ufloat(-0.40, 0.01)
        l = a + b*(self.mk - self.mu)
        return l

    def readspec(self, fspec):
        """Read in a set of spectra (unused)"""
        sfiles = glob.glob(fspec + '/*.dat')
        sfiles.sort()
        wave = [None]*len(sfiles)
        spec = [None]*len(sfiles)
        misc = [None]*len(sfiles)
        for i, j in enumerate(sfiles):
            f = np.genfromtxt(j)
            wave[i] = np.nan_to_num(f[:, 0])
            spec[i] = np.nan_to_num(f[:, 1])
            if f.shape[1] > 2:
                print('[INFO] File containing spectrum contains more than'),
                print('just wavelength and spectrum!')
                print('[INFO] Everything else written to self.misc')
            misc[i] = f[:, 2:]
        return wave, spec, misc


def plotspec(owave, ospec, mwave, mspec):
    """Plot spectra in a paper-ready way"""
    plt.ion()
    plt.figure()
    plt.plot(mwave, ospec, 'black', label='Obs')
    plt.plot(mwave, mspec, 'r', label='BF model')
    plt.legend(loc=4)
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('Norm. Flux')
    plt.show()


def trimspec(w1, w2, s2):
    """Trim s2 and w2 to match w1"""
    roi = np.where((w2 > w1.min()) & (w2 < w1.max()))[0]
    return w2[roi], s2[roi]

lines = np.genfromtxt('lib/lines.txt')[:, 1]
import fit


def maskline(wave, spec):
    """Mask regions around lines"""
    wid = 0.0005  # microns
    idxall = []
    for l in lines:
        idx = np.where((wave > l - wid) & (wave < l + wid))[0]
        x = wave[idx]
        y = spec[idx]*-1 + 1.
        guess = [0.0001, l, 0.0001]  # amp, cen, wid
        robs = fit.fitline(x, y, (y / np.std(y)), guess)
        wid = robs.values.values()[1]*6  # 3sigma * 2
        idx = np.where((wave > l - wid) & (wave < l + wid))[0]
        idxall.append(idx)
        # print(spec[idx].std())
    return idxall

# Start the proceedings:
print('[INFO] Reading in model grid ...')
then = time.time()
mod = ReadMod(
    'models/MODELSPEC_2013sep12_nLTE_R10000_J_turb_abun_grav_temp-int.sav')
t1 = time.time() - then
print('[INFO] Time taken in seconds:', t1)

# Testing:
# Pick a model to test the routines on!
# 0.01 is a reasonable level of noise (comparing this to observations)
nlevel = 0.001
msize = mod.grid[0, 0, 0, 0].shape[0]
# RSG1 RSG2 RSG4 RSG7 RSG8 RSG9 RSG10 RSG11 RSG14
mrsg1 = mod.grid[12, 3, 4, 4] + np.random.normal(0, nlevel, msize)
mrsg2 = mod.grid[12, 1, 6, 4] + np.random.normal(0, nlevel, msize)
mrsg4 = mod.grid[15, 7, 4, 5] + np.random.normal(0, nlevel, msize)
mrsg7 = mod.grid[15, 3, 6, 6] + np.random.normal(0, nlevel, msize)
mrsg8 = mod.grid[10, 3, 7, 5] + np.random.normal(0, nlevel, msize)
mrsg9 = mod.grid[13, 5, 4, 6] + np.random.normal(0, nlevel, msize)
mrsg10 = mod.grid[13, 2, 3, 5] + np.random.normal(0, nlevel, msize)
mrsg11 = mod.grid[16, 4, 2, 5] + np.random.normal(0, nlevel, msize)
mrsg14 = mod.grid[12, 7, 2, 5] + np.random.normal(0, nlevel, msize)

# Real spectra
# Trim to match models:
print('[INFO] Read observed spectra from file:')
print('[INFO] Please ensure all files are ordered similarly!')
# n6822 = ReadObs('../ngc6822/Spectra/N6822-spec-24AT.v1.txt')

n6822 = ReadObs('../ngc6822/Spectra/N6822-spec-24AT.v2-sam.txt',
                '../ngc6822/Photometry/N6822-phot-KMOS-sam-err.txt',
                mu=ufloat(23.3, 0.05))
# Prep:
print('[INFO] Trim observations to match models:')
owave, ospec = trimspec(mod.wave, n6822.wave, n6822.nspec)
# Bestfit mod from Ben:
# r10bf = np.genfromtxt('../ngc6822/fits/bestfit/specfit_N6822_24AT_v1_30.dat')

# test:
ospec = ospec.T
# ospec = (mrsg4, mrsg9, mrsg11)

# Fix 2 parameters:
# Micro & logg
fixi = ((12, 4), (12, 6), (15, 4), (15, 6), (10, 7),
        (13, 4), (13, 3), (16, 2), (12, 2))
# Input parameters for Z & teff
fixin = ((3, 4), (1, 4), (7, 5), (3, 6), (3, 5), (5, 6),
         (2, 5), (4, 5), (7, 5))
# Reverse the parameters:
# Fix Z & teff
# fixi = ((3, 4), (1, 4), (7, 5), (3, 6), (3, 5), (5, 6),
#         (2, 5), (4, 5), (7, 5))
# # Input parameters for Micro & logg
# fixin = ((12, 4), (12, 6), (15, 4), (15, 6), (10, 7),
#          (13, 4), (13, 3), (16, 2), (12, 2))

# fixi = ((15, 4), (13, 4), (16, 2))
# ospec = (mrsg1, mrsg2, mrsg4, mrsg7, mrsg8, mrsg9, mrsg10, mrsg11, mrsg14)
# ossam = contfit.specsam(mod.wave, ospec, owave)
# ospec = res.degrade(owave, ossam, mod.res, n6822.res)
# -----------
params = np.zeros((np.shape(ospec)[0], 4))
fixout = np.zeros(np.shape(fixi)).astype(int)
# In this for loop we need more than one spectrum! -- change this!
# Going from testing to real observations should be simpler than this!
# currently:
# ospec.T --> ospec
# owave --> mod.wave
# mdeg --> mod.grid

# Resample:
print('[INFO] Resampling model grid ...')
then = time.time()
mssam = contfit.specsam(mod.wave, mod.grid, owave)
print('[INFO] Time taken in seconds:', time.time() - then)

# Degrade:
print('[INFO] Degrading model grid ...')
then = time.time()
mdeg = res.degrade(owave, mssam, mod.res, n6822.res)
print('[INFO] Time taken in seconds:', time.time() - then)

for i, spec in enumerate(ospec):
    # Find indicies around lines:
    idx = maskline(owave, spec)  # Observation at Obs. resolution
    # idx = maskline(mod.wave, spec)  # Model at mod. resolution
    # Constrain the grid to reject unphysical log g's
    glow = np.log10(nom(n6822.glow[i])) - 0.25
    gup = np.log10(nom(n6822.gup[i])) + 0.25
    mod.parlimit(glow, gup, 'GRAVS')
    mgrid = mdeg[:, :, mod.parcut]
    # mgrid = mod.grid[:, :, mod.parcut]
    # Compute chisq:
    print('[INFO] Compute chi-squared grid ...')
    then = time.time()
    chi, oscale, cft = bestfit.chigrid(mgrid, spec, owave, n6822.res, idx)
    # test1 : use model at degraded res & sampling
    # chi, oscale, cft = bestfit.chigrid(mgrid, spec, owave, n6822.res)
    # test2 : use maodel at orig. res & sampling
    # chi, oscale, cft = bestfit.chigrid(mgrid, spec, mod.wave, mod.res, idx)
    vchi = np.ma.masked_where(chi == 0.0, chi, copy=False)
    print('[INFO] Time taken in seconds:', time.time() - then)

    print('------------------------------------')
    print('[INFO] Calcualte bestfit parameters ...')
    then = time.time()
    # mod.par.field('TEMPS')[0] = mod.par.field('TEMPS')[0][5]
    # (xiav, zav, gav, tav), mini = bestfit.bf(chi, mod.par)
    # params[i], mini = bestfit.bf(chi, mod.par)
    bfobj = bestfit.BestFit(vchi, mod.par)
    bfobj.showinit()
    bfobj.showfin()
    print('[INFO] Time taken in seconds:', time.time() - then)
    print('------------------------------------')
    # fixout[i] = bestfit.fix2(vchi[fixi[i][0], :, fixi[i][1]],
    #                          mod.par[0][0], mod.par[0][1], 'Teff', '[Z]')
    params[i] = bfobj.bf
    # fixout[i] = bestfit.fix2(vchi[:, fixi[i][0], :, fixi[i][1]],
    #                          mod.par[0][2], mod.par[0][3])
    # Plot the bestfit model to the observed data:
    # Have to change the order of bfobs.min w.r.t the grid!!!

    # bfspec = mod.grid[bfobj.min[0, 0], bfobj.min[1, 0],
    #                   bfobj.min[2, 0], bfobj.min[3, 0]]
    bfspec = mod.grid[bfobj.min[3, 0], bfobj.min[1, 0],
                      bfobj.min[2, 0], bfobj.min[0, 0]]
    # plotspec(owavem, spec, mod.wave, bfspec)
# -----------
tout, zout = np.column_stack(fixout)
tin, zin = np.column_stack(fixin)
# mtout, gout = np.column_stack(fixout)
# mtin, gin = np.column_stack(fixin)

# plt.scatter(bfobj.prange[0][tin], bfobj.prange[0][tout])
# f, ax = plt.subplots(1, 2)
# ax[0].scatter(bfobj.prange[0][tin], bfobj.prange[0][tout])
# ax[0].set_xlabel('Teff In')
# ax[0].set_ylabel('Teff Out')
# ax[1].scatter(bfobj.prange[1][zin], bfobj.prange[1][zout])
# ax[1].set_xlabel('[Z] In')
# ax[1].set_ylabel('[Z] Out')


def plot2(r1, in1, out1, n1, r2, in2, out2, n2):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    ax[0].scatter(r1[in1], r1[out1])
    ax[0].set_xlabel(n1 + ' In')
    ax[0].set_ylabel(n1 + ' Out')
    ax[1].scatter(r2[in2], r2[out2])
    ax[1].set_xlabel(n2 + ' In')
    ax[1].set_ylabel(n2 + ' Out')

# plot2(bfobj.prange[2], gin, gout, 'log g',
#       bfobj.prange[3], mtin, mtout, 'MicroTurb')
# plot2(bfobj.prange[1], tin, tout, 'Teff',
#       bfobj.prange[0], zin, zout, '[Z]')


# Unfinished:
# Errors:
# import errors
# mspec = mcheat
# Add noise characteristic of the S/N ratio of the observations
# Guess at 0.01 for now ...
# Need a function to measure noise in observations and replicate it


def err(mspec, mgrid, owave, ores):
    """
        Need to pass this everything that bestfit.chigrid needs
    """
    epar = np.zeros((10, 4))
    for i in xrange(10):
        nmod = mspec + np.random.normal(0, 0.01, mspec.shape[0])
        echi, eoscale = bestfit.chigrid(mgrid, nmod, owave, ores)
        epar[i] = bestfit.bf(echi, mod.par, quiet=True)
    return epar

# then = time.time()
# epar = err(mraw, mod.grid, mod.wave, mod.res)
# print('[INFO] Time taken in seconds:', time.time() - then)


def errper(a):
    low = np.percentile(a, 15.9)
    high = np.percentile(a, 84.1)
    (high - low) / 2.

# Just to get us started, crudely take the std:
# et, eg, exi, ez = np.round(np.std(epar, axis=0), 3)
# print('------------------------------------')
# print('[INFO] Bestfit parameters:')
# print('------------------------------------')
# print(tav, '+/-', et, gav, '+/-', eg, xiav, '+/-', exi, zav, '+/-', ez)
# print('------------------------------------')

# Mask MgI lines


# def maregion(wave, s, w1, w2):
#     idx = np.where((owave > mg[0] - wid) & (owave < mg[0] + wid))[0]
#     guess = [0.0001, mg[0], 0.0001]
#     x = owave[idx]
#     y = ospec[idx]*-1 + 1.
#     rmg1 = fit.fitline(x, y, w=None, guess=guess)
#     wid = rmg1.values.values()[1]*3
#     onew = np.ma.masked_where(((owave > mg[0] - wid) &
#                                (owave < mg[0] + wid)), owave)
#     return onew


def maskmg(owave, ospec):
    wid = 0.0005
    mg = np.array([1.1828, 1.2083])
    idx = np.where((owave > mg[0] - wid) & (owave < mg[0] + wid))[0]
    guess = [0.0001, mg[0], 0.0001]
    x = owave[idx]
    y = ospec[idx]*-1 + 1.
    rmg1 = fit.fitline(x, y, w=None, guess=guess)
    wid = rmg1.values.values()[1]*3
    onew = np.ma.masked_where(((owave > mg[0] - wid) &
                               (owave < mg[0] + wid)), owave)
    return onew
