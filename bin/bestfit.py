"""
Author: LRP
Date: 10-02-2015
Description:
Find best fit parameters for a set of model grid and an observation

All dependencies are contained within astropy

References:
Gazak (2014) Thesis

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import contfit
import cc
import fit
# from wrap import rsgjanal

lines = np.genfromtxt('lib/lines.txt')[:, 1]


class BestFit(object):
    """ BestFit:
        Calculate bestfit parameters from a grid of chisq values

        Parameters:
        chi :
            Full chisq-array
        mpar :
            model parameters

        BestFit parameters contained within self.bf
    """
    def __init__(self, chi, mpar):
        self.chi = chi
        self.mpar = mpar
        self.pnames = self.mpar.dtype.names
        self.prange = self.mpar[0]
        self.vchi = np.ma.masked_equal(chi, 0.0, copy=False)
        self.c = coarsesam(self.vchi, (5, 2, 2, 1))
        self.cidx = np.unravel_index(np.argmin(self.c), self.c.shape)
        self.v = np.ma.masked_where(self.vchi != self.c[self.cidx], chi,
                                    copy=False)
        self.fi = np.unravel_index(self.v.argmax(), self.vchi.shape)
        self.min = minima(self.vchi, self.fi, self.prange, quiet=False)
        self.bf = [np.mean(self.prange[i][j]) for i, j in enumerate(self.min)]

    def showinit(self):
        print('[INFO] Initial bestfit parameters:')
        print('[INFO] Teff, [Z], log g, MicroTurb:')
        print('[INFO]', end=' ')
        for i, j in enumerate(self.cidx):
            print(self.mpar[0][i][j], end=' ')
        print('')

    def showfin(self):
        print('[INFO] Final bestfit parameters:')
        print('[INFO] Teff, [Z], log g, MicroTurb:')
        print('[INFO]', end=' ')
        for i, j in enumerate(self.min):
            avpar = np.mean(self.prange[i][j])
            print(avpar, end=' ')
        print('')


def coarsesam(grid, samp):
    """
        5x in MicroTurb
        2x in logg
        2(.5)x in Z
        1x in Teff
        How do I sample 2.5x in [Z]???
    """
    if samp == (5, 2, 2, 1):
        return grid[0::5, 1::2, 0::2, :]
    else:
        print('[WARNING] Sampling changed hack required in bestfit.coursesam')


def fix2(chi, x, y, xname, yname):
    """
    "A good plan violently executed now
    is better than a perfect plan executed next week."
    George S. Patton
    """
    f, ax = plt.subplots(1)
    xmin, ymin = minidx(chi)
    cmin = chi.min()
    n = (cmin + 1, cmin + 2, cmin + 3, cmin + 5, cmin + 10)
    ax.contour(x, y, chi, n)
    ax.set_xlabel(xname, fontsize=10)
    ax.set_ylabel(yname, fontsize=10)
    plt.show()
    return (xmin, ymin)


def minima(chi, idx, mpar, quiet=False):
    """
    Define the set of three minima for each parameter

    Parameters:
    chi : numpy.ma.core.MaskedArray
        chisq-array with 0.0's masked out
    idx : tuple
        4-D index of chi in order:
        MicroTurb
        [Z]
        log (g)
        Teff
    Returns:
        ximin : numpy.ndarray
            Three microturbulence indicies
        zmin : numpy.ndarray
            Three [Z] indicies
        gmin : numpy.ndarray
            Three log (g) indicies
        tmin : numpy.ndarray
            Three Teff indicies
    """
    xii, zi, gi, ti = idx
    # xi, z, g, t = mpar
    t, z, g, xi = mpar
    ximin = np.zeros(3).astype(int)
    zmin = np.zeros(3).astype(int)
    gmin = np.zeros(3).astype(int)
    tmin = np.zeros(3).astype(int)

    gmin[0], tmin[0] = minidx(chi[xii, zi, :, :])
    zmin[0], tmin[1] = minidx(chi[xii, :, gi, :])
    zmin[1], gmin[1] = minidx(chi[xii, :, :, ti])
    ximin[0], tmin[2] = minidx(chi[:, zi, gi, :])
    ximin[1], gmin[2] = minidx(chi[:, zi, :, ti])
    ximin[2], zmin[2] = minidx(chi[:, :, gi, ti])

    # Diagnostic plots:
    if quiet is False:
        # cmin = chi.min()
        # n = (cmin + 1, cmin + 2, cmin + 3, cmin + 5, cmin + 10)
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6))\
            = plt.subplots(2, 3, figsize=(16, 10))
        min1 = chi[xii, :, gi, :].min()
        n = (min1 + 1, min1 + 2, min1 + 3, min1 + 5, min1 + 10)
        ax1.contour(t, z, chi[xii, :, gi, :], n)
        ax1.set_xlabel('Teff', fontsize=10)
        ax1.set_ylabel('[Z]', fontsize=10)

        min2 = chi[:, :, gi, ti].min()
        n = (min2 + 1, min2 + 2, min2 + 3, min2 + 5, min2 + 10)
        ax2.contour(xi, z, chi[:, :, gi, ti].T, n)
        ax2.set_xlabel('Xi', fontsize=10)
        ax2.set_ylabel('[Z]', fontsize=10)

        min3 = chi[xii, :, :, ti].min()
        n = (min3 + 1, min3 + 2, min3 + 3, min3 + 5, min3 + 10)
        ax3.contour(g, z, chi[xii, :, :, ti], n)
        ax3.set_xlabel('log (g)', fontsize=10)
        ax3.set_ylabel('[Z]', fontsize=10)
        min4 = chi[:, zi, gi, :].min()
        n = (min4 + 1, min4 + 2, min4 + 3, min4 + 5, min4 + 10)
        ax4.contour(t, xi, chi[:, zi, gi, :], n)
        ax4.set_xlabel('Teff', fontsize=10)
        ax4.set_ylabel('Xi', fontsize=10)

        min5 = chi[xii, zi, :, :].min()
        n = (min5 + 1, min5 + 2, min5 + 3, min5 + 5, min5 + 10)
        ax5.contour(t, g, chi[xii, zi, :, :], n)
        ax5.set_xlabel('Teff', fontsize=10)
        ax5.set_ylabel('log (g)', fontsize=10)

        min6 = chi[:, zi, :, ti].min()
        n = (min6 + 1, min6 + 2, min6 + 3, min6 + 5, min6 + 10)
        ax6.contour(g, xi, chi[:, zi, :, ti], n)
        ax6.set_xlabel('log (g)', fontsize=10)
        ax6.set_ylabel('Xi', fontsize=10)

        plt.show()

    return np.vstack((tmin, zmin, gmin, ximin))


def minidx(arr):
    """Find min of a N-D array return the non-flat index"""
    idx = np.unravel_index(arr.argmin(), arr.shape)
    return idx


def chigrid(mgrid, ospec, owave, ores):
    chi = np.zeros(mgrid.shape[0:4])
    oscale = np.zeros(mgrid.shape)
    cft = np.zeros(np.append(mgrid.shape[0:4], 4))
    for i in xrange(mgrid.shape[3]):
        for j in xrange(mgrid.shape[2]):
            for k in xrange(mgrid.shape[1]):
                for l in xrange(mgrid.shape[0]):
                    # print(i, j, k, l)
                    mspec = mgrid[l][k][j][i]
                    # Is this check helping?
                    if np.isnan(mspec.max()) == False:
                        # I need to find a way to create oscale & chi without
                        # this ridiculous 4D for loop ..
                        oscale[l, k, j, i], \
                            chi[l, k, j, i], \
                            cft[l, k, j, i] = rsgjanal(ospec, owave, ores,
                                                       mspec, quiet=False)
    return chi, oscale, cft


def chigrid2(mgrid, ospec, owave, ores):
    oscale, chi, cft = [rsgjanal(ospec, owave, ores, mgrid[l][k][j][i],
                        quiet=False)
                        for i in xrange(mgrid.shape[3])
                        for j in xrange(mgrid.shape[2])
                        for k in xrange(mgrid.shape[1])
                        for l in xrange(mgrid.shape[0])
                        if ~np.isnan(mgrid[l][k][j][i].max())]
    return chi, oscale, cft


def rsgjanal(ospec, owave, ores, mspec, quiet=True):
    """
        Trimmed spectra, nothing more to do than run the routines!
        Full artillery!
        Make an input file for filenames and resolution for each spectrum

    """

    # 1. Find spectral resolution - lets start by assuimng I know this
    # 2. Resample models onto ooresbservations
    # 3. Degrade model spectrum to match resolution of observations
    # -- use Jesus' routine
    # 4. Fit the continuum
    # 5. Derive stellar parameters ... -- done outside function
    # 6. Define the errors ... -- done outside function

    # Many of the steps are done before this function ...
    # This should be the function to execute all of that!
    # It can't be run within a 4D for loop and
    # 1.:

    # 2.: Resample
    # mssam = contfit.specsam(mwave, mspec, owave)
    # 3.:
    # mdeg = degrader(owave, mspec, mres, ores)
    # 4.:
    cft = contfit.contfit2(ores, owave, mspec, ospec)

    # oscale = ospec * cft(owave)
    oscale = ospec * cft(owave)
    # Cross-correlate
    mcc = cc.ccshift(oscale, mspec, owave)
    # Calculate Chisq
    chi = chicalc(owave, oscale, mcc)
    # chi = chisq(oscale, np.std(oscale), mcc)

    if quiet is not True:
        return oscale, chi, cft
    return oscale, chi


def chicalc(wave, ospec, mspec):
    """
        Mask regions around lines
        Fit Gaussian profile to each line

        !This is the rate determining step in calculating the chisq grid!
    """
    chi = 0.0
    # Need to define the width from the fitting process
    wid = 0.0005  # microns
    for l in lines:
        idx = np.where((wave > l - wid) & (wave < l + wid))[0]
        # Observed
        x = wave[idx]
        y = ospec[idx]*-1 + 1.
        guess = [0.0001, l, 0.0001]  # amp, cen, wid
        robs = fit.fitline(x, y, (y / np.std(y)), guess)
        wid = robs.values.values()[1]*6  # 3sigma * 2
        idx = np.where((wave > l - wid) & (wave < l + wid))[0]
        # Model
        # ymod = mspec[idx]*-1 + 1.
        # rmod = fit.fitline(x, ymod, (ymod / np.std(ymod)), guess)
        # How to define the "error on the line strength"?
        # A first guess from fitting the line ...
        # err = 2.11e-06
        # err = robs.params.values()[0].stderr
        # err = y - robs.best_fit
        err = np.std(ospec[idx])
        # print err
        chi += chisq(ospec[idx], err, mspec[idx])
        # chi += chisq(robs.best_fit, err, rmod.best_fit)
        # Testing:
        # print(robs.fit_report())
        # plt.plot(x, y, 'bo')
        plt.plot(x, robs.best_fit*-1 + 1., 'r-')
    return chi


def chisq(obs, err, mod):
    return np.sum(((obs - mod)**2) / err**2)


def cplot(x, y, z, n, xname, yname):
    """
        Wrap matplotlib's contout plot
        np.meshgrid is a good way to do this!

        Parameters:
        x, y : x & y coordniates for figure
        z : z-coordinate for contours
    """
    X, Y = np.meshgrid(x, y)
    plt.ion()
    plt.figure()
    cs = plt.contour(X, Y, z, n, colors='k')
    plt.clabel(cs, inline=0, fontsize=10)
    plt.title(xname + ' vs. ' + yname)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()

import itertools


def mintest(chi, idx):
    """

    """
    n = len(idx)
    y = lambda n: (n**2 - n) / 2
    nfigs = y(n)
    f, axs = plt.subplots(2, nfigs / 2, figsize=(16, 10))

    exc = itertools.combinations(range(n), n - 2)
    inc = list(itertools.combinations(range(n), 2))[::-1]
    pars = np.zeros(np.shape(inc)).astype(int)

    for count, i in enumerate(exc):
        sl = [slice(None)]*i[0] + [idx[i[0]]]

        for j in i[1:]:
            sl += [slice(None)]*(j - i[0] - 1) + [idx[j]]

        sl += [slice(None)]*(len(idx) - i[len(i) - 1] - 1)
        pars[count] = np.unravel_index(chi[sl].argmin(), chi[sl].shape)
        # minx = chi[sl].min()
        # ncont = (minx + 1, minx + 2, minx + 3, minx + 5, minx + 10)
        # axidx = changeindx(count, nfigs)
        # axs[0, 1].contour(x, y, chi[sl], ncont)
    avpar = [pars[np.ma.masked_equal(inc, k).mask] for k in range(n)]
    return np.asarray(avpar)


def changeindx(i, n):
    if i < n / 2.:
        newi = [0] + [i]
    else:
        newi = [1] + [i - n / 2]
    return newi

# # sl for len(idx) = 3
# for i in itertools.combinations(range(3), 1):
#     sl = [slice(None)]*i + [idx[i]] + [slice(None)]*(len(idx) - i - 1)
#     print(chi[sl].shape)

#     # sl for len(idx) = 4
# for i, j in itertools.combinations(range(4), 2):
#     sl = [slice(None)]*i + [idx[i]] + [slice(None)]*(j - i - 1) \
#         + [idx[j]] + [slice(None)]*(len(idx) - j - 1)
#     print sl

#     # sl for len(idx) = 5
# for i, j, k in itertools.combinations(range(5), 3):
#     print i, j, k
#     sl = [slice(None)]*i + [idx[i]] + [slice(None)]*(j - i - 1) \
#         + [idx[j]] + [slice(None)]*(k - j - 1) \
#         + [idx[k]] + [slice(None)]*(len(idx) - k - 1)
#     print sl

#     # sl for len(idx) = n where n > 2
# for i in itertools.combinations(range(n), n - 2):
#     print i
#     sl = [slice(None)]*i[0] + [idx[i[0]]]
#     for j in i[1:]:
#         sl += [slice(None)]*(j - i[0] - 1)\
#             + [idx[j]]
#     sl += [slice(None)]*(len(idx) - i[len(i) - 1] - 1)
#     print(sl)
#     print(chi[sl].shape)


#         print(chi[sl].shape)
#         midx = np.unravel_index(chi[sl].argmin(), chi[sl].shape)
#        midx = minidx(chi[sl])


# Not used:


def bf(chi, mpar, quiet=False):
    """
        Calculate bestfit parameters from a grid of chisq values

        Parameters:
        chi :
            Full chisq-array
        mpar :
            model parameters

        Returns:
        tav, gav, xiv, zav : floats
            Average bestfit parameter for effective temperature,
            sufrace gravity, microturbulence and metallicity
    """
    # Make mpar more readable!
    xi = mpar.field('TURBS')[0]  # 21
    z = mpar.field('ABUNS')[0]  # 19
    g = mpar.field('GRAVS')[0]  # 9
    t = mpar.field('TEMPS')[0]  # 11

    # Mask out obsolete 0.0's:
    vchi = np.ma.masked_equal(chi, 0.0, copy=False)
    coarse = coarsesam(vchi, (5, 2, 2, 1))
    idx = np.unravel_index(np.argmin(coarse), coarse.shape)
    xii, zi, gi, ti = idx
    print('[INFO] Initial bestfit parameters:')
    print('[INFO] Teff, log g, MicroTurb, [Z]:')
    print('[INFO]', t[ti], g[gi], xi[xii], z[zi])
    print('[INFO] Chisq min = ', coarse.min())

    # Coarse index on fine grid:
    v = np.ma.masked_where(vchi != coarse[idx], chi, copy=False)
    fi = np.unravel_index(v.argmax(), vchi.shape)
    # Mask out obsolete 0.0's:
    # vchi = np.ma.masked_equal(chi, 0.0, copy=False)
    # Fix the params and find minima:
    mini = minima(vchi, fi, (xi, z, g, t), quiet=quiet)
    # # Average values:
    xiav = np.mean(xi[mini[0]])
    zav = np.mean(z[mini[1]])
    gav = np.mean(g[mini[2]])
    tav = np.mean(t[mini[3]])

    print('------------------------------------------------')
    print('[INFO] Calculated bestfit parameters:')
    print('[INFO] Teff, logg, MickyTurb, [Z]')
    print('[INFO]', tav, gav, xiav, zav)
    print('------------------------------------------------')
    if quiet is False:
        print('[INFO] Prepare for some plots!')

    return (xiav, zav, gav, tav), mini


def contour(x, y, z, n):
    """
        Wrap matplotlib's contout plot
        np.meshgrid is a good way to do this!
        Return a simple contour plot rather than plotting it

        Parameters:
        x, y : x & y coordniates for figure
        z : z-coordinate for contours
    """
    X, Y = np.meshgrid(x, y)
    cs = plt.contour(X, Y, z, n, colors='k')
    # plt.clabel(cs, inline=0, fontsize=10)
    return cs


def modelinterp():
    """
        inerpolate between two models
        -- How to implement this?
            Average the two?
    """
    return
