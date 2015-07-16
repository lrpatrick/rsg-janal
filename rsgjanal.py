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
from scipy.io.idl import readsav
from uncertainties import ufloat
from uncertainties import unumpy

import contfit
import bestfit
import resolution as res
import cc

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
        self.head = self.changehead()
        self.res = 10000.
        self.mt, self.z, self.g, self.t = self.gridorder()
        self.trim = maregion(self.wave, 1.165, 1.215)
        self.tgrid = self.grid[:, :, :, :, self.trim]
        self.twave = self.wave[self.trim]

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
        par, idx = self.findpar(name)
        self.head[0][idx] = par
        self.parcut = np.where((par > low) & (par < high))[0]
        self.head[0][idx] = self.head[0][idx][self.parcut]
        print('[INFO] ' + name + ' range constrained by ReadMod.parlimit')
        print('[INFO] Range is now:', self.head[0][int(idx)])

    def findpar(self, name):
        """Look-up table style function using the names of parameters"""
        if name == 'TURBS':
            par = self.mt
            idx = 0
        if name == 'ABUNS':
            par = self.z
            idx = 1
        if name == 'GRAVS':
            par = self.g
            idx = 2
        if name == 'TEMPS':
            par == self.t
            idx = 3
        return par, idx

    def changehead(self):
        """Sort the parameters part of the grid out!"""
        # Should be more general!
        x = [self.par[0][3], self.par[0][1], self.par[0][2], self.par[0][0]]
        y = self.pnames[3], self.pnames[1], self.pnames[2], self.pnames[0]
        return (x, y)


class ReadObs(object):
    """
        ReadObs assumes a file with columns: 1. Wavelength 2-N: Spectra
        If observations are normalised used self.spec,
        if not, a simple median normalisation is apllied in self.nspec
    """
    def __init__(self, fspec, fphot, fres, fsn, mu):
        self.fspec = fspec
        self.fphot = fphot
        self.res = fres
        self.sn = fsn
        self.mu = mu
        self.fall = np.genfromtxt(fspec)
        self.wave = self.fall[:, 0]
        self.spec = self.fall[:, 1:]
        # trimmed:
        # self.trim = maregion(self.wave, 1.165, 1.215)
        # self.twave = self.wave[self.trim]
        # self.tspec = self.spec[self.trim]
        self.nspec = self.spec / np.median(self.spec, axis=0)
        self.phot = np.genfromtxt(fphot)
        self.mk = unumpy.uarray(self.phot[:, 15], self.phot[:, 16])
        self.L = self.luminosity()
        self.gup, self.glow = self.glimits()
        # Should read in a text file:
        # self.res = 3700.  # IFU 14: res at Ar1.21430: 3736 +/- 142

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

    # def readspec(self, fspec):
    #     """Read in a set of spectra (unused)"""
    #     sfiles = glob.glob(fspec + '/*.dat')
    #     sfiles.sort()
    #     wave = [None]*len(sfiles)
    #     spec = [None]*len(sfiles)
    #     misc = [None]*len(sfiles)
    #     for i, j in enumerate(sfiles):
    #         f = np.genfromtxt(j)
    #         wave[i] = np.nan_to_num(f[:, 0])
    #         spec[i] = np.nan_to_num(f[:, 1])
    #         if f.shape[1] > 2:
    #             print('[INFO] File containing spectrum contains more than'),
    #             print('just wavelength and spectrum!')
    #             print('[INFO] Everything else written to self.misc')
    #         misc[i] = f[:, 2:]
    #     return wave, spec, misc


def coarsesam(grid, samp):
    """
        5x in MicroTurb
        2(.5)x in Z
        2x in logg
        1x in Teff
        How do I sample 2.5x in [Z]???
    """
    if samp == (5, 2, 2, 1):
        return grid[0::5, 1::2, 0::2]
    else:
        print('[WARNING] Sampling changed hack required in bestfit.coursesam')


def maregion(wave, w1, w2):
    """Filter a region between w1 & w2"""
    return (wave > w1) & (wave < w2)


def maskobs(wl):
    """Mask observations using maregion"""
    omask = maregion(wl, 1.1915, 1.1935)
    omask += maregion(wl, 1.182, 1.184)
    omask += maregion(wl, 1.2080, 1.2087)
    return ~omask


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
lines.sort()
# import fit
from lmfit.models import GaussianModel


def maskline(wave, spec, sn):
    """Mask regions around lines"""
    plt.figure()
    plt.plot(wave, spec)
    idxall = []
    # stderr = []
    for l in lines:
        wid = 0.0005  # microns
        idx = np.where((wave > l - wid) & (wave < l + wid))[0]
        x = wave[idx]
        y = spec[idx]*-1 + 1.
        # guess = [0.0001, l, 0.0001]  # amp, cen, wid
        # robs = fit.fitline(x, y, np.sqrt(y), guess)
        mod = GaussianModel()
        pars = mod.guess(y, x=x, weights=1/sn)
        out = mod.fit(y, pars, x=x, weights=1/sn)
        print(out.fit_report(min_correl=0.25))
        # plt.figure()
        # plt.plot(x, out.best_fit*-1. + 1.)
        wid = out.values.values()[1]*3  # 3*FWHM
        idx = np.where((wave > l - wid) & (wave < l + wid))[0]
        plt.plot(wave[idx], spec[idx])
        idxall.append(idx)
        # stderr.append(out.params.values()[3].stderr)
        # print(spec[idx])
    return idxall


def defidx(w1):
    """Define regions/lines for diagnostic lines to compute chisq over"""
    idx = [np.where((w1 > 1.1879) & (w1 < 1.1899))[0]]
    idx.append(np.where((w1 > 1.19447) & (w1 < 1.19528))[0])
    idx.append(np.where((w1 > 1.1965) & (w1 < 1.19986))[0])
    idx.append(np.where((w1 > 1.20253) & (w1 < 1.20359))[0])
    idx.append(np.where((w1 > 1.20986) & (w1 < 1.21078))[0])
    return idx

# Start the proceedings:
print('[INFO] Reading in model grid ...')
then = time.time()
mod = ReadMod(
    'models/MODELSPEC_2013sep12_nLTE_R10000_J_turb_abun_grav_temp-int.sav')
t1 = time.time() - then
print('[INFO] Time taken in seconds:', t1)

# Fake spectra:
mrsg01 = mod.tgrid[12, 3, 4, 4]
mrsg02 = mod.tgrid[12, 1, 6, 4]
mrsg04 = mod.tgrid[15, 7, 4, 5]
mrsg07 = mod.tgrid[15, 3, 6, 6]
mrsg08 = mod.tgrid[10, 3, 7, 5]
mrsg09 = mod.tgrid[13, 5, 4, 6]
mrsg10 = mod.tgrid[13, 2, 3, 5]
mrsg11 = mod.tgrid[16, 4, 2, 5]
mrsg14 = mod.tgrid[12, 7, 2, 5]
mrsg17 = mod.tgrid[10, 5, 4, 5]
mrsg18 = mod.tgrid[6, 2, 6, 4]

lmc05 = mod.tgrid[8, 9, 4, 5]
lmc07 = mod.tgrid[16, 6, 2, 7]

n300010 = mod.tgrid[17, 9, 0, 7]
n300022 = mod.tgrid[11, 8, 1, 1]
n300037 = mod.tgrid[10, 4, 6, 5]
n300126 = mod.tgrid[16, 7, 2, 9]
n300134 = mod.tgrid[20, 9, 3, 10]

f1 = mod.tgrid[13, 12, 3, 6]
f2 = mod.tgrid[14, 14, 6, 5]
f3 = mod.tgrid[14, 18, 6, 5]

# Real spectra
# Trim to match models:
print('[INFO] Read observed spectra from file:')
print('[INFO] Please ensure all files are ordered similarly!')
# n6822 = ReadObs('../ngc6822/Spectra/N6822-spec-24AT.v1.txt')
n6822res = np.genfromtxt('../ngc6822/Catalogues/N6822-cat-res.txt')[:, 2]

# Giving the class files assumes that the files will be standardised!
# Better option would be to give the class the spectra and photometry
n6822 = ReadObs('../ngc6822/Spectra/N6822-spec-24AT.v2-sam.txt',
                '../ngc6822/Photometry/N6822-phot-KMOS-sam-err.txt', n6822res,
                [150]*len(n6822res), mu=ufloat(23.3, 0.05))
n6822.sn[1] = 120
n6822.sn[4] = 103
n6822.sn[-1] = 104

# Prep:
print('[INFO] Observations and models trimed to the 1.165-1.215mu region')
owave, ospec = trimspec(mod.twave, n6822.wave, n6822.nspec)
# Select first 4 for speed
ospec = ospec.T[0:]
# Mask the observations
omask = maskobs(owave)

# Bestfit mod from Ben:
# r10bf = np.genfromtxt('../ngc6822/fits/bestfit/specfit_N6822_24AT_v1_30.dat')
# Fix 2 parameters:
# Micro & logg
# fixi = ((12, 4), (12, 6), (15, 4), (15, 6), (10, 7),
#         (13, 4), (13, 3), (16, 2), (12, 2), (), ())
# # Input parameters for Z & teff
# fixin = ((3, 4), (1, 4), (7, 5), (3, 6), (3, 5), (5, 6),
#          (2, 5), (4, 5), (7, 5), (), ())
# Reverse the parameters:
# Fix Z & teff
# fixi = ((3, 4), (1, 4), (7, 5), (3, 6), (3, 5), (5, 6),
#         (2, 5), (4, 5), (7, 5))
# # Input parameters for Micro & logg
# fixin = ((12, 4), (12, 6), (15, 4), (15, 6), (10, 7),
#          (13, 4), (13, 3), (16, 2), (12, 2))

# fixi = ((15, 4), (13, 4), (16, 2))

# Using models instead of observations
# Tests 1 & 2
# mspec = [mrsg01, mrsg02]
mspec = [mrsg01, mrsg02, mrsg04, mrsg07, mrsg08,
         mrsg09, mrsg10, mrsg11, mrsg14, mrsg17, mrsg18,
         lmc05, lmc07, n300010, n300022, n300037, n300126, n300134,
         f1, f2, f3]
# mspec = [mrsg02]*10
# owave = mod.twave  # test 1
# omask = maskobs(owave)
# # mrsg01 = mod.tgrid[12, 3, 4, 4]
# Test 2:
# Degrade & Resample:
# osam = contfit.specsam(mod.twave, mspec, owave)
# odeg = res.degrade(owave, osam, float(mod.res), float(n6822.res[0]))
# Noise added after degrading and resampling
nlevel = 0.01
# msize = mod.tgrid.shape[-1]  # test 1
msize = np.shape(owave)[0]  # test 2
# noise = np.random.normal(0, nlevel, (msize, np.shape(mspec)[0]))  # test 1
# noise = np.random.normal(0, nlevel, (msize, np.shape(odeg)[0]))  # test 2
# ospec = mspec + noise.T  # test 1
# mdeg = mod.tgrid  # test 1
# ospec = odeg + noise.T  # test 2

# fixout = np.zeros(np.shape(fixi)).astype(int)
# In this for loop we need more than one spectrum! -- change this!

print('[INFO] Resampling model grid ...')
then = time.time()
mssam = contfit.specsam(mod.twave, mod.tgrid, owave)
print('[INFO] Time taken in seconds:', time.time() - then)
# Test2:
# print('[INFO] Degrading model grid ...')
# then = time.time()
# resi = float(n6822.res[0])
# mdeg = res.degrade(owave, mssam, float(mod.res), resi)
# print('[INFO] Time taken in seconds:', time.time() - then)
# -----------
# for i in bfclass:
#     print i.fipar
#     minidx = np.unravel_index(np.argmin(i.fchi), i.fchi.shape)
#     # print minidx
#     for i, j in enumerate(minidx):
#         print mod.head[0][i][j],
#     print
# # Get mean parameters and errors from multiple runs:
# [(np.mean(full[i::15], axis=0), np.std(full[i::15], axis=0)) for i in xrange(15)]

# -----------
bfclass = []
params = np.zeros((np.shape(ospec)[0], 4))
chi = ([0]*np.shape(ospec)[0])
mscale = ([0]*np.shape(ospec)[0])
cft = ([0]*np.shape(ospec)[0])
ospeccc = ([0]*np.shape(ospec)[0])
for i, j in enumerate(ospec):
    print('[INFO] Degrading model grid ...')
    then = time.time()
    # resi = 10000  # test 1
    # resi = float(n6822.res[0])  # test 2
    sn = n6822.sn[i]
    resi = float(n6822.res[i])  # test 3
    mdeg = res.degrade(owave, mssam, float(mod.res), resi)  # test 3
    print('[INFO] Time taken:', round(time.time() - then, 3), 's')  # test3

    print('[INFO] Shift spectrum onto rest wavelength:')
    mspec1 = mdeg[0, 0, 0, 0]
    spec, s1 = cc.ccshift(mspec1, j, owave, quiet=False)
    ospeccc[i] = spec
    # Using diag. lines only:
    idx = defidx(owave)  # test 2 & 3
    # idx = maskline(owave, spec, sn)  # test 2 & 3
    # idx = maskline(mod.twave, spec, sn)  # Model at mod. resolution
    mgrid = mdeg
    owavem = owave
    # Using entire region with masks
    # Mask model spectra
    # mgrid = mdeg[:, :, :, :, omask]
    # spec = spec[omask]
    # owavem = owave[omask]

    print('[INFO] Compute chi-squared grid ...')
    then = time.time()
    # Need to pass the the model grid and a observed spectrum class:
    # chi, mscale, cft = bestfit.chigrid(mgrid, spec, owavem, resi)
    # chi[i], mscale[i], cft[i] = bestfit.chigrid(mgrid, spec, owavem, resi)
    # chi, mscale, cft = bestfit.chigrid(mgrid, spec, owavem, resi, idx, sn)
    chi[i], mscale[i], cft[i] = bestfit.chigrid(mgrid, spec, owavem, resi, idx, sn)
    # chi, oscale, cft = bestfit.chigrid(mgrid, spec, owavem, 10000)
    chii = chi[i]  # / 8.
    # chii = chi
    vchi = np.ma.masked_where(chii == 0.0, chii, copy=False)
    print('[INFO] Time taken in seconds:', time.time() - then)

    # Constrain the grid to reject unphysical log g's
    # -0.25 should be the step between grid values
    # glow = np.log10(nom(n6822.glow[1])) - 0.25
    # gup = np.log10(nom(n6822.gup[1])) + 0.25
    glow = np.log10(nom(n6822.glow[i])) - 0.25  # test3
    gup = np.log10(nom(n6822.gup[i])) + 0.25  # test3
    mod.parlimit(glow, gup, 'GRAVS')
    vfchi = vchi[:, :, mod.parcut]
    # Filter coarse grid:
    vcoarse = coarsesam(vchi, (5, 2, 2, 1))
    # Constrain coarse grid:
    # gcoarse = (mod.g[mod.parcut][0] <= mod.g[0::2]) & \
    #           (mod.g[mod.parcut][-1] >= mod.g[0::2])
    # vcchi = vcoarse[:, :, gcoarse]
    # No g-range restrictions
    # vfchi = vchi
    vcchi = vcoarse
    print('------------------------------------')
    print('[INFO] Calcualte bestfit parameters ...')
    then = time.time()
    bfobj = bestfit.BestFit(vfchi, vcchi, mod.head)
    # bfobj.showinit()
    # bfobj.showfin()
    print('[INFO] Time taken in seconds:', time.time() - then)
    print('------------------------------------')
    params[i] = bfobj.bf
    bfclass.append(bfobj)
    # fixout[i] = bestfit.fix2(vchi[fixi[i][0], :, fixi[i][1]],
    #                          mod.par[0][0], mod.par[0][1], 'Teff', '[Z]')
    # fixout[i] = bestfit.fix2(vchi[:, fixi[i][0], :, fixi[i][1]],
    #                          mod.par[0][2], mod.par[0][3])
    # Plot the bestfit model to the observed data:
    # Have to change the order of bfobs.min w.r.t the grid!!!

    # bfspec = mod.grid[bfobj.min[0, 0], bfobj.min[1, 0],
    #                   bfobj.min[2, 0], bfobj.min[3, 0]]
    # bfspec = mgrid[bfobj.min[3, 0], bfobj.min[1, 0],
    #                bfobj.min[2, 0], bfobj.min[0, 0]]
    # plotspec(owavem, spec, mod.wave, bfspec)
# -----------
bfspec = [mscale[i][j.fi] for i, j in enumerate(bfclass)]

# tout, zout = np.column_stack(fixout)
# tin, zin = np.column_stack(fixin)
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

x = np.zeros((len(n6822.glow), 4))
errx = np.zeros((len(n6822.glow), 4))
for i in xrange(len(n6822.glow)):
    glow = np.log10(nom(n6822.glow[i])) - 0.25  # test3
    gup = np.log10(nom(n6822.gup[i])) + 0.25  # test3
    mod.parlimit(glow, gup, 'GRAVS')
    x[i] = wparams(bfclass[i].fchi)
    test1 = np.ma.masked_greater_equal(bfclass[i].fchi, bfclass[i].fchi.min() + 3.)
    m = np.array(np.where(~test1.mask))
    print(m.shape)
    errx[i] = [np.std(mod.head[0][k][l]) for k, l in enumerate(m)]
    maxi = [mod.head[0][k][l].max() for k, l in enumerate(m)]
    mini = [mod.head[0][k][l].min() for k, l in enumerate(m)]
    print(maxi, mini)


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


# def err(mspec, mgrid, owave, ores):
#     """
#         Need to pass this everything that bestfit.chigrid needs
#     """
#     epar = np.zeros((10, 4))
#     for i in xrange(10):
#         nmod = mspec + np.random.normal(0, 0.01, mspec.shape[0])
#         echi, eoscale = bestfit.chigrid(mgrid, nmod, owave, ores)
#         epar[i] = bestfit.bf(echi, mod.par, quiet=True)
#     return epar

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
