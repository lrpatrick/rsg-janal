"""
Author: LRP
Date: 17-07-2015
Description:
Find best fit parameters from a chisq-grid
"""
from __future__ import print_function

import numpy as np


class BestFit(object):
    """ Calculate bestfit parameters from a grid of chisq values"""
    def __init__(self, fchi, mhead):
        self.fchi = fchi
        self.mhead = mhead

        self.pnames = self.mhead[1]
        self.prange = self.mhead[0]

        # Remove 0.0's:
        self.vchi = np.ma.masked_where(fchi == 0.0, fchi, copy=False)
        self.fi = np.unravel_index(np.argmin(self.vchi), self.vchi.shape)
        self.fipar = [self.prange[i][j] for i, j in enumerate(self.fi)]
        self.wpar = self.wparams()
        self.err = self.errparam()

    def showmin(self):
        print('[INFO] Minimum of chisq-grid:', self.vchi.min())
        print('[INFO] Parameters')
        print('[INFO] MicroTurb, [Z], log g, Teff:')
        print('[INFO]', self.fipar)

    def bestfew(self, n):
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
        n = 10
        bftop = self.bestfew(n)
        mttop, ztop, gtop, \
            ttop, chival = np.array([(mtv[i], zv[i], gv[i], tv[i],
                                     self.vchi[i]) for i in bftop]).T
        w = [np.exp(-i) for i in chival]
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
