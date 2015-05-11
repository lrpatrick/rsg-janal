# Fit the lines:
"""
    Author: LRP
    Date: 08-04-2015
    Description:
    Functions to fit spectral lines using LMFIT's 'Model' function
"""
from lmfit import Model
import numpy as np


def fitline(x, y, w, guess):
    """Simple Gaussian model for a line where initial guesses are provided"""
    g = Model(gaussian)
    result = g.fit(y, x=x, weights=w, amp=guess[0], cen=guess[1], wid=guess[2])
    # result = g.fit(y, x=x, weights=w, amp=guess[0], cen=guess[1], wid=guess[2])
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

# np.random.seed(0)
# x = np.linspace(-5., 5., 200)
# y = 3 * np.exp(-0.5 * (x - 1.3)**2 / 0.8**2)
# y += np.random.normal(0., 0.2, x.shape)
# gmod = Model(gaussian)
# # amp, cen, wid
# guess = [5, 5, 1]
# result = gmod.fit(y, x=x, amp=guess[0], cen=guess[1], wid=guess[2])
# # one line in the model
# wid = 0.0005
# idx = np.where((owave > lines[0] - wid) & (owave < lines[0] + wid))[0]
# # idx = np.where((mwave > l - wid) & (mwave < l + wid))[0]
# idx = np.array([1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053,
#                 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063])
# xline = mwave[idx]
# mline = mspec[idx]*-1 + 1.
# xline = owave[idx]
# oline = ospec[idx]*-1 + 1.
# guess = [0.0001, 1.188285, 0.0001]
# result = gmod.fit(mline, x=xline, amp=guess[0], cen=guess[1], wid=guess[2])
# vals = result.best_values.values()
