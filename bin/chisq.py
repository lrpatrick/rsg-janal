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
    """Function could take in oclass and mspec"""
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
    """
    Prep for chisq calculation
    This function could take in a oclass and mspec
    This function is now twinned with rsganal.defidx, any changes made here
    must be relected in chisq.chireg
    """
    cft = contfit.contfit(ores, owave, mspec, ospec)[0]
    mscale = mspec / cft(owave)
    mcc, s = cc.ccshift(ospec, mscale, owave)
    # Calculate Chisq
    ccregs = chireg(owave)
    chi = [0]*len(idx)
    for i, reg in enumerate(ccregs):

        oreg = ospec[reg]
        wreg = owave[reg]
        mreg_tmp, sreg = cc.ccshift(oreg, mscale[reg], wreg, quiet=True)
        mreg, sreg1 = cc.ccshift(owave, mscale, owave, shift1=sreg)
        # plt.plot(owave[reg], mreg[reg], 'b')
        chi[i] = chisq(ospec[idx[i]], 1./snr, mreg[idx[i]]) / len(idx[i])
        # print('[INFO] chisq value for region {} = {}'.format(i, chi[i]))

        # chi = chicalc(owave, ospec, mcc, idx, snr)
    chisum = np.sum(chi)
    # print('[INFO] Total chisq = {}'.format(chisum))
    return chisum, mcc, cft


def chireg(w1):
    """Note: This must match with defidx in rsganal.py"""
    # For "Main regions containing lines" in rsganal.defidx
    # ccreg = [np.where((w1 > 1.18000) & (w1 < 1.19200))[0]]  # MgI, FeI, TiI
    # ccreg.append(np.where((w1 > 1.19400) & (w1 < 1.20600))[0])  # Ti,Fe,Six3
    # ccreg.append(np.where((w1 > 1.20600) & (w1 < 1.21200))[0])  # SiI, MgI

    # For "Only the cores of individual lines" in rsganal.defidx
    # but using the "Main regions containing lines" regions to cross-correlate
    # ccreg = [np.where((w1 > 1.18000) & (w1 < 1.19200))[0]]  # MgI, FeI, TiI
    # ccreg.append(np.where((w1 > 1.18000) & (w1 < 1.19200))[0])  # MgI,FeI,TiI
    # ccreg.append(np.where((w1 > 1.18000) & (w1 < 1.19200))[0])  # MgI,FeI,TiI
    # ccreg.append(np.where((w1 > 1.18000) & (w1 < 1.19200))[0])  # MgI,FeI,TiI

    # ccreg.append(np.where((w1 > 1.19400) & (w1 < 1.20600))[0])  # Ti,Fe,Six3
    # ccreg.append(np.where((w1 > 1.19400) & (w1 < 1.20600))[0])  # Ti,Fe,Six3
    # ccreg.append(np.where((w1 > 1.19400) & (w1 < 1.20600))[0])  # Ti,Fe,Six3
    # ccreg.append(np.where((w1 > 1.19400) & (w1 < 1.20600))[0])  # Ti,Fe,Six3
    # ccreg.append(np.where((w1 > 1.19400) & (w1 < 1.20600))[0])  # Ti,Fe,Six3

    # ccreg.append(np.where((w1 > 1.20600) & (w1 < 1.21200))[0])  # SiI, MgI
    # ccreg.append(np.where((w1 > 1.20600) & (w1 < 1.21200))[0])  # SiI, MgI

    # # For "Only the cores of individual lines" in rsganal.defidx
    # ccreg = [np.where((w1 > 1.1870) & (w1 < 1.190))[0]]  # FeI & TiI
    ccreg = [np.where((w1 > 1.1815) & (w1 < 1.1850))[0]]  # MgI
    ccreg.append(np.where((w1 > 1.1870) & (w1 < 1.190))[0])  # FeI & TiI
    ccreg.append(np.where((w1 > 1.1870) & (w1 < 1.190))[0])  # FeI & TiI
    ccreg.append(np.where((w1 > 1.19350) & (w1 < 1.19600))[0])  # TiI,FeI,SiIx2
    ccreg.append(np.where((w1 > 1.19550) & (w1 < 1.19800))[0])  # TiI,FeI,SiIx2
    ccreg.append(np.where((w1 > 1.19550) & (w1 < 1.19986))[0])  # TiI,FeI,SiIx2
    ccreg.append(np.where((w1 > 1.19550) & (w1 < 1.19986))[0])  # TiI,FeI,SiIx2
    ccreg.append(np.where((w1 > 1.20100) & (w1 < 1.20500))[0])  # SiI
    ccreg.append(np.where((w1 > 1.20700) & (w1 < 1.21150))[0])  # Mg & Si
    ccreg.append(np.where((w1 > 1.20700) & (w1 < 1.21150))[0])  # Mg & Si
    # ccreg.append(np.where((w1 > 1.2114) & (w1 < 1.21820))[0])  # continuum

    # For "Only the cores of individual lines" in rsganal.defidx
    # ccreg = [np.where((w1 > 1.1870) & (w1 < 1.190))[0]]  # FeI & TiI
    # ccreg = [np.where((w1 > 1.1815) & (w1 < 1.1850))[0]]  # MgI
    # ccreg.append(np.where((w1 > 1.1870) & (w1 < 1.190))[0])  # FeI & TiI
    # ccreg.append(np.where((w1 > 1.19350) & (w1 < 1.19600))[0])  # Ti,Fe,Six2
    # ccreg.append(np.where((w1 > 1.19550) & (w1 < 1.19800))[0])  # Ti,Fe,Six2
    # ccreg.append(np.where((w1 > 1.20100) & (w1 < 1.20500))[0])  # SiI
    # ccreg.append(np.where((w1 > 1.20700) & (w1 < 1.21150))[0])  # Mg & Si
    # ccreg.append(np.where((w1 > 1.20700) & (w1 < 1.21150))[0])  # Mg & Si
    return ccreg


def chicalc(owave, ospec, mspec, idx, snr):
    """Calculate chisq for each line"""
    chi = np.sum([chisq(ospec[i], 1. / snr, mspec[i]) / len(i) for i in idx])
    return chi


def chisq(obs, err, mod):
    return np.sum(((obs - mod)**2) / err**2)
