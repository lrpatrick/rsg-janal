"""
Author: LRP
Date: 03-08-2015
Description:
Routine to fit the continuum for a spectrum

"""
from __future__ import print_function

import numpy as np
from scipy.interpolate import interp1d


def contfit(res, wave, mspec, ospec):
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
