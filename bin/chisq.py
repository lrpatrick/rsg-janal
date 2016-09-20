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

# TODO: chigrid and chiprep be one class?


def chigrid(mgrid, ospec, owave, ores, idx, snr, ccidx, cfit, cfitdof):
    """Function could take in oclass and mspec"""
    chi = np.zeros(mgrid.shape[:-1])
    mscale = np.zeros(mgrid.shape)
    cft = np.zeros(np.append(mgrid.shape[:-1], cfitdof + 1))

    for i, j, k, l in itertools.product(*map(xrange, (mgrid.shape[:-1]))):
        mspec = mgrid[i, j, k, l]

        if ~np.isnan(mspec.max()):
            chi[i, j, k, l], mscale[i, j, k, l], cft[i, j, k, l]\
                = chiprep(ospec, owave, ores, mspec,
                          idx, snr, ccidx, cfit, cfitdof)

    return chi, mscale, cft


def chiprep(ospec, owave, ores, mspec, idx, snr, ccidx, cfit, cfitdof):
    """
    Prep for chisq calculation
    This function could take in a oclass and mspec
    This function is now twinned with rsganal.defidx, any changes made here
    must be relected in chisq.ccregions
    """
    if cfit == 'simple':
        # print('[INFO] simple c-fitting')
        cft = contfit.alt_contfit(ores, owave, mspec, ospec, cfitdof)
    else:
        # print('[INFO] regular c-fitting')
        cft = contfit.contfit(ores, owave, mspec, ospec, cfitdof)

    mscale = mspec / cft(owave)
    mcc, s = cc.ccshift(ospec, mscale, owave, quiet=True)
    # Calculate Chisq
    ccregs = ccidx
    chi = [0]*len(idx)
    ndof = 4
    for i, reg in enumerate(ccregs):

        test = False

        if test is False:
            oreg = ospec[reg]
            wreg = owave[reg]
            mreg_tmp, sreg = cc.ccshift(oreg, mscale[reg], wreg, quiet=True)
            mreg, sreg1 = cc.ccshift(owave, mscale, owave, shift1=sreg)
            chi[i] = chisq(ospec[idx[i]], 1./snr, mreg[idx[i]]) / (len(idx[i][0]) - ndof)
        else:
            chi[i] = chisq(ospec[idx[i]], 1./snr, mspec[idx[i]]) / (len(idx[i][0]) - ndof)

    chisum = np.sum(chi)
    return chisum, mcc, cft


def chisq(obs, err, mod):
    residuals = (obs - mod)/err
    chi = np.sum(residuals**2)
    return chi
    # return np.sum(((obs - mod)**2) / err**2)
