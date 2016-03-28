"""
Author: LRP
Date: 17-07-2015
Description:
Find best fit parameters from a chisq-grid
"""
from __future__ import print_function

import astropy.constants as c
import astropy.units as u
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
        m = np.array(np.where(~dchi.mask))
        err = [np.std(self.prange[k][l]) for k, l in enumerate(m)]
        return err


def masslims():
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


def clipg(grid, trange, grange, l):
    """
    Clip the unphysical areas of the grid based on luminosity.
    Assumes a grid with axes: micro, Z, logg, Teff
    Insert np.nan's into the grid where the range is clipped
    """
    newgrid = np.copy(grid)
    # mhigh = 40.*c.M_sun
    # mlow = 8.*c.M_sun
    mhigh, mlow = masslims()
    const = 4*np.pi*c.sigma_sb*c.G
    lsi = 10**l*c.L_sun
    g = lambda T, M: np.log10(((const*M*T**4) / lsi).cgs.value)

    gstep = grange[1] - grange[0]
    ghigh = g(trange*u.K, mhigh) + gstep
    glow = g(trange*u.K, mlow) - 0.3 - gstep

    print('[INFO] Rejected surface gravity models:')
    for ti in xrange(len(trange)):
        grej = np.where((glow[ti] > grange) | (ghigh[ti] < grange))[0]
        newgrid[:, :, grej, ti] = np.nan
        print(r'[INFO] Teff {}K log g {}'.format(trange[ti], grange[grej]))
    return newgrid


def clipmt(grid, mtrange):
    """Restrict MicroTurb range."""
    newgrid = np.copy(grid)
    # mhigh = 40.*c.M_sun
    newgrid[0:9, :, :] = np.nan
    newgrid[12:, :, :] = np.nan
    return newgrid

# Estimated parameters: vo, sigma
# Observed data: (rv, e_rv)
