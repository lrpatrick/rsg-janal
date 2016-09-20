"""
Author: LRP
Date: 03-08-2015
Description:
Routine to fit the continuum for a spectrum

"""
from __future__ import print_function

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt


def alt_contfit(res, wave, mspec, ospec, cfitdof):
    """
    Default method only really works for good S/N data
    For poorer S/N data we'll do something a little more simple

    1. Divide observed spectrum by model spec
        (assuming wavelength solution is consistent)
    2. Median filter the result (filter width = 7 i.e. extreme!)
    3. Fit third order polynomial to the reusult
    """
    residual = mspec / ospec
    mf7 = medfilt(residual, 7)
    return np.poly1d(np.polyfit(wave, mf7, cfitdof))


def contfit(res, wave, mspec, ospec, cfitdof):
    """
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
        print('[WARNING] Resolution greater than 10000!')
        print('[WARNING] As of Feburary 2015 model resolution is 10000')

    # Define continuum width
    s = 1.0
    # cw = 1.2*s / res
    cw = wave[len(wave)/2]*s / res
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
    return np.poly1d(np.polyfit(wave[c3idx], r2cont, cfitdof))


def trimspec(w1, w2, s2):
    """Trim s2 and w2 to match w1"""
    roi = np.where((w2 > w1.min()) & (w2 < w1.max()))[0]
    return w2[roi], s2[roi]


def specsam(win, inspec, wnew):
    """Update spectral sampling using scipy.interpolate.interp1d"""
    if np.all(win == wnew):
        return inspec
    else:
        i1d = interp1d(win, inspec)
        return i1d(wnew)


# def wiggles(wave, s):
#     """
#     Remove any large scale wiggles from the spectrum using a simple
#     polynomial function
#     """
#     wf = np.poly1d(np.polyfit(wave, s, 3))
#     return s / wf(wave)
