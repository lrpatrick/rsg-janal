"""
Author: LRP
Date: 17-07-2015
Description:
Find best fit parameters from a chisq-grid
"""
from __future__ import print_function

import astropy.constants as c
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Saving space on print statements
o = str('[INFO] ')
w = str('[WARNING] ')
e = str('[ERROR] ')


class BestFit(object):
    """Calculate bestfit parameters from a grid of chisq values"""

    def __init__(self, fchi, mhead, name):
        """Init"""
        self.fchi = fchi
        self.mhead = mhead
        self.id = name

        self.pnames = self.mhead[1]
        self.prange = self.mhead[0]

        # Mask 0.0's:
        self.vchi = np.ma.masked_where(fchi == 0.0, fchi, copy=False)
        self.fi = np.unravel_index(np.argmin(self.vchi), self.vchi.shape)
        self.fipar = [self.prange[i][j] for i, j in enumerate(self.fi)]
        self.wpar = self.wparams()
        self.err = self.errparam()

    def showmin(self):
        """Print minimum of chisq-grid and the parameters"""
        print('[INFO] Minimum of chisq-grid:', self.vchi.min())
        print('[INFO] Parameters')
        print('[INFO] MicroTurb, [Z], log g, Teff:')
        print('[INFO]', self.fipar)

    def showavpar(self):
        """Print weigthed average parameters"""
        print('[INFO] Weighted average parameters')
        print('[INFO] MicroTurb, [Z], log g, Teff:')
        print('[INFO]', np.around(self.wpar, 2))

    def bestfew(self, n):
        """Select top n bestfit models"""
        g1 = self.vchi
        bfall = []
        for i in xrange(n):
            bfi = np.unravel_index(np.argmin(g1), g1.shape)
            # print(bfi, g1.min())
            bfall.append(bfi)
            g1 = np.ma.masked_equal(g1, g1[bfi])
        return bfall

    def wparams(self):
        """Define parameters using weights defined by chi-squared values"""
        mtv, zv, gv, tv = np.meshgrid(self.prange[0], self.prange[1],
                                      self.prange[2], self.prange[3],
                                      indexing='ij')
        n = 100
        bftop = self.bestfew(n)
        mttop, ztop, gtop, \
            ttop, chival = np.array([(mtv[i], zv[i], gv[i], tv[i],
                                     self.vchi[i]) for i in bftop]).T
        w = [np.exp(-i/2.) for i in chival]
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

    def errparam(self):
        """Define errors for parameters where <chi-sq-min + 3"""
        dchi = np.ma.masked_greater_equal(self.vchi, self.vchi.min() + 3.)
        mask = np.array(np.where(~dchi.mask))
        err = [np.std(self.prange[k][m]) for k, m in enumerate(mask)]
        return err

    def bfcontour(self):
        """Show contours for fit results"""
        chi = self.vchi
        mpar = self.prange
        xbf, zbf, gbf, tbf = self.fipar
        xii, zi, gi, ti = self.fi
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


def masslims():
    """Not Used"""
    print('[INFO] Mass restrictions for this data set?')
    while True:
        try:
            mhigh = int(raw_input('[INFO] Upper limit: Default 40 M_sun:\n') or
                        int(40))
            break
        except ValueError:
            print('[ERROR] Fool! Input not int. Try again...')
            print('[INFO] Please select an integer between 8 and 40')

    while True:
        try:
            mlow = int(raw_input('[INFO] Lower limit: Default 8 M_sun:\n') or
                       int(8))
            break
        except ValueError:
            print('[ERROR] Fool! Input not int. Try again...')
            print('[INFO] Please select an integer between 8 and 40')
    return mhigh*c.M_sun, mlow*c.M_sun


def clipg(grid, trange, grange, lum, mlow, mhigh):
    """
    Clip the unphysical areas of the grid based on luminosity.
    Assumes a grid with axes: micro, Z, logg, Teff
    Insert np.nan's into the grid where the range is clipped
    """
    if lum == 0.0:
        print('[INFO] No restrictions based on L or M implemented')
        return grid
    else:
        newgrid = np.copy(grid)
        mhigh = mhigh*c.M_sun
        mlow = mlow*c.M_sun
        # mhigh, mlow = masslims()
        lsi = 10**lum*c.L_sun
        # g = lambda T, M: np.log10(((const*M*T**4) / lsi).cgs.value)
        gstep = grange[1] - grange[0]
        ghigh = grav(trange*u.K, mhigh, lsi) + gstep
        glow = grav(trange*u.K, mlow, lsi) - 0.3 - gstep

        print('[INFO] Rejected surface gravity models:')
        for ti in xrange(len(trange)):
            grej = np.where((glow[ti] > grange) | (ghigh[ti] < grange))[0]
            newgrid[:, :, grej, ti] = np.nan
            print(r'[INFO] Teff {}K log g {}'.format(trange[ti], grange[grej]))
        return newgrid


def grav(t, m, lum):
    k = 4*np.pi*c.sigma_sb*c.G
    return np.log10(((k*m*t**4) / lum).cgs.value)


def clipmt(grid, mtrange):
    """Restrict MicroTurb range."""
    newgrid = np.copy(grid)
    # mhigh = 40.*c.M_sun
    newgrid[0:9, :, :] = np.nan
    newgrid[12:, :, :] = np.nan
    return newgrid

# Estimated parameters: vo, sigma
# Observed data: (rv, e_rv)
