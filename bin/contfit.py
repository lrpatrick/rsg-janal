"""
Author: LRP
Date: 22-01-2015
Description:
Routine to fit the continuum for a spectrum

All dependencies shoule be contained within astropy

References:
Gazak (2014) Thesis

"""
import numpy as np
from scipy.interpolate import interp1d


# def contfit(res, wave, mspec, ospec):
#     """

#         What do I want this function to be able to do?
#         1. Read in two spectra over the same wavelength range
#         2. Scale the flux level of the model to match that of the data
#         3. Return the scaled model spectrum

#         Arguments:
#         res : float
#             Resolution of the observations and model
#         wave : numpy.ndarray
#             Wavelength axis for model and observations
#         mspec : numpy.ndarray
#             Model spectrum (same size as wave & ospec)
#         ospec : numpy.ndarray
#             Observed spectrum

#         Returns:
#         cft : numpy.lib.polynomial.poly1d
#             Function which defines the scaling for the model/obs. spectra
#     """

#     # Checks:
#     if wave.shape[0] != mspec.shape[0]:
#         print '[ERROR] confit.confit()'
#         print '[ERROR]',
#         print 'Wavelength axis and model spectra are not of same length'
#         print '[ERROR]',
#         print 'Exit now'
#         raise SystemExit

#     if ospec.shape[0] != mspec.shape[0]:
#         print '[ERROR] confit.confit()'
#         print '[ERROR]',
#         print 'Observed and model spectra are not of same length'
#         print '[ERROR]',
#         print 'Exit now'
#         raise SystemExit

#     if res > 10000.:
#         print '[WARNING] Resolution greater than 10000!'
#         print '[WARNING] As of Feburary 2015 model resolution is 10000'

#     # Define continuum width
#     s = 0.5
#     cw = 1.20 * s / res
#     nele = np.round(cw / (wave[1] - wave[0]), 0).astype(int)
#     # Identify 'continuum' points and remove outliers
#     midx = [x + np.argmax(mspec[x:x + nele])
#             for x in np.arange(0, ospec.shape[0], nele)]

#     c1 = np.column_stack((wave[midx], mspec[midx]))
#     c2idx = np.array(midx)[np.where(np.abs(c1[:, 1] - np.median(c1[:, 1]))
#                                     < 3 * c1[:, 1].std())[0]]

#     # Correction function:
#     # Take ratio at 'continuum' points:
#     r2 = mspec[c2idx] / ospec[c2idx]
#     cf = np.poly1d(np.polyfit(wave[c2idx], r2, 3))
#     # Remove outliers
#     c3idx = c2idx[np.where(np.abs(cf(wave[c2idx]) - r2)
#                            < 3 * r2.std())[0]]

#     # Correction function tuned:
#     # Repeat previous step with outliers removed
#     r3 = mspec[c3idx] / ospec[c3idx]
#     cft = np.poly1d(np.polyfit(wave[c3idx], r3, 3))
#     return cft


def contfit(res, wave, mspec, ospec):
    """
        TODO:
        Read in the full model grid and compute a cft for each model

        What do I want this function to be able to do?
        1. Read in two spectra over the same wavelength range
        2. Scale the flux level of the model to match that of the data
        3. Return the scaling function

        Arguments:
        res : float
            Resolution of the observations and model
        wave : numpy.ndarray
            Wavelength axis for model and observations
        mspec : numpy.ndarray
            Model spectrum (same size as wave & ospec)
        ospec : numpy.ndarray
            Observed spectrum

        Returns:
        cft : numpy.lib.polynomial.poly1d
            Function which defines the scaling for the model/obs. spectra
    """
    # Checks
    if res > 10000.:
        print '[WARNING] Resolution greater than 10000!'
        print '[WARNING] As of Feburary 2015 model resolution is 10000'

    # Define continuum width
    s = 1.0
    cw = 1.20*s / res
    n = np.ceil((cw / (wave[1] - wave[0]))).astype(int)
    # print(n)
    # Identify 'continuum' points and remove outliers
    y = np.arange(0, ospec.shape[0], n)
    c1idx = np.array([x + np.argmax(mspec[x:x + n]) for x in y])
    # c1 = np.column_stack((wave[c1idx], mspec[c1idx]))
    sig1 = mspec[c1idx].std()
    mcont = mspec[c1idx]
    c2idx = c1idx[np.where(np.abs(mcont - np.mean(mcont)) < 3*sig1)[0]]

    # Correction function:
    # Take ratio at 'continuum' points:
    r1cont = mspec[c2idx] / ospec[c2idx]
    cf1 = np.poly1d(np.polyfit(wave[c2idx], r1cont, 3))
    # Remove outliers
    r1contsig = r1cont.std()
    c3idx = c2idx[np.where(np.abs(cf1(wave[c2idx]) - r1cont) < 3*r1contsig)[0]]

    # Correction function tuned:
    # Repeat previous step with outliers removed
    r2cont = mspec[c3idx] / ospec[c3idx]
    return np.poly1d(np.polyfit(wave[c3idx], r2cont, 3)), c1idx, c2idx, c3idx


def specsam(win, inspec, wnew):
    """
        Change sampling of the model to match the observations by means of a
        linear interpolation using scipy.interpolate.interp1d

        Arguments:
        win : numpy.ndarray
            Initial wavelength axis
        inspec : numpy.ndarray
            Input spectrum associated with win
        wnew : numpy.ndarray

        Output:
        newspec : numpy.ndarray
            inspec resampled onto wnew
    """
    i1d = interp1d(win, inspec)
    return i1d(wnew)

# Stuff for testing
# from degrade import degrader

# from scipy.io.idl import readsav
# # model spec
# mod = readsav(
#     '../models/MODELSPEC_2013sep12_nLTE_R10000_J_turb_abun_grav_temp-int.sav')
# mgrid = mod['modelspec'][0][0]
# mspec = mgrid[13][2][3][5]
# # Simulated observed spec, at 10,000 resolution
# ospec_sim = mgrid[0][1][0][0] + np.random.normal(0, 0.01, mspec.shape[0])
# # Teff, [Z], log g, Xi
# mrange = mod['modelspec'][0][1]
# mwave = mod['modelspec'][0][2]

# # observed spec
# n6822_spec = np.genfromtxt('../../ngc6822/Spectra/N6822-spec-24AT.v1.txt')
# rsg_obs = np.column_stack((n6822_spec[:, 0], n6822_spec[:, 3]))
# # RSG 10:
# # Teff    log g    Xi    [Z]
# # 3900    -0.3    3.7   -0.67

# ##############################################################################
# # Let the testing begin:
# ##############################################################################
# # Try confit on real oberved spectrum:
# # Resample model:

# # c2idx = contfit(mwave, 10000, mwave, mspec, 0, 0)
# # cft = contfit(mwave, 10000, mwave, mspec, mwave, ospec_sim)
# cft = contfit(10000, mwave, mspec, ospec_sim)
# # Try with degraded spectra
# # First resample:
# ospec3000 = degrader(mwave, ospec_sim, 10000, 3000, quick=4)
# mod3000 = degrader(mwave, mspec, 10000, 3000, quick=4)
# # z = np.polyfit(mwave[c2idx], ratio, 3)
# # fit = np.poly1d(z)
# # fcorr = fit(mwave)
# mod4000 = degrader(mwave, mspec, 10000, 4000, quick=4)
# mod5000 = degrader(mwave, mspec, 10000, 5000, quick=4)
# mod6000 = degrader(mwave, mspec, 10000, 6000, quick=4)
# mod7000 = degrader(mwave, mspec, 10000, 7000, quick=4)
# mod8000 = degrader(mwave, mspec, 10000, 8000, quick=4)

# import matplotlib.pyplot as plt
# plt.legend(loc=4)
# plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'Normalised Flux')
# plt.plot(mwave, mod3000, c='black', label='R = 3000')
# plt.plot(mwave, mod4000, c='r', label='R = 4000')
# plt.plot(mwave, mod5000, c='g', label='R = 5000')
# plt.plot(mwave, mod6000, c='b', label='R = 6000')

# New functions:
