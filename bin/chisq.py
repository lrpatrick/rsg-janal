"""
Author: LRP
Date: 16-07-2015
Description:
Calcaulte chi-squared grid

References:
Gazak (2014) Thesis

"""
import itertools
import numpy as np

import contfit
import cc


def chigrid(mgrid, ospec, owave, ores, idx, snr):
    """This function could take in oclass and mspec"""
    chi = np.zeros(mgrid.shape[0:-1])
    mscale = np.zeros(mgrid.shape)
    cft = np.zeros(np.append(mgrid.shape[0:-1], 4))

    for i, j, k, l in itertools.product(*map(xrange, (mgrid.shape[0:-1]))):
        # print(i, j, k, l)
        mspec = mgrid[i, j, k, l]

        if ~np.isnan(mspec.max()):
            chi[i, j, k, l], mscale[i, j, k, l], cft[i, j, k, l]\
                = chiprep(ospec, owave, ores, mspec, idx, snr)

    return chi, mscale, cft


def chiprep(ospec, owave, ores, mspec, idx, snr):
    """Prep for chisq calculation
        This function could take in a oclass and mspec
    """
    cft = contfit.contfit(ores, owave, mspec, ospec)[0]
    mscale = mspec / cft(owave)
    mcc, shift = cc.ccshift(ospec, mscale, owave)
    # Calculate Chisq
    chi = chicalc(owave, ospec, mcc, idx, snr)
    return chi, mcc, cft


def chicalc(owave, ospec, mspec, idx, snr):
    """Calculate chisq for each line"""
    chi = np.sum([chisq(ospec[i], 1. / snr, mspec[i]) / len(i) for i in idx])
    return chi


def chisq(obs, err, mod):
    return np.sum(((obs - mod)**2) / err**2)
