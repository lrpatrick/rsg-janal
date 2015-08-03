"""
    Author: LRP
    Date: 08-04-2015
    Description:
    Functions to fit spectral lines using LMFIT's 'Model' function
"""
from __future__ import print_function

from lmfit import Model
import numpy as np


def fitline(x, y, w, guess):
    """Simple Gaussian model for a line where initial guesses are provided"""
    g = Model(gaussian)
    result = g.fit(y, x=x, weights=w, amp=guess[0], cen=guess[1], wid=guess[2])
    return result


def fitlines(lines, wave, spec):
    """
        Wrap fitline() for a set of lines
        Parameters:
        lines : numpy.ndarray
            Line list
        wave : numpy.ndarray
            wavelength axis
        spec : numpy.ndarray
            Spectrum covering at least all of the spectral features in 'lines'

        Returns:
        resutlt : numpy.ndarray
            Array of result from fitline
    """
    result = np.empty(0)
    wid = 0.0005
    for i, l in enumerate(lines):
        idx = np.where((wave > l - wid) & (wave < l + wid))[0]
        x = wave[idx]
        y = spec[idx]*-1 + 1.
        guess = [0.0001, l, 0.0001]  # amp, cen, wid
        rtmp = fitline(x, y, guess)
        result = np.append(result, rtmp)
        # Tests:
        # print(rtmp.fit_report())
        # plt.plot(xline, oline, 'bo')
        # plt.plot(xline, rtmp.best_fit, 'r-')
    return result


def gaussian(x, amp, cen, wid):
    """
        1-d gaussian: gaussian(x, amp, cen, wid)
    """
    return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2/(2*wid**2))
