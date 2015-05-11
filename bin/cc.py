"""
    Author: LRP
    Date: 06-11-2014
    File Description:
    Cross-correlation module, stolen and slightly ammended from an
    IDL conversion
"""
import numpy as np
from scipy.interpolate import UnivariateSpline


def ccshift(s1, s2, x1, shift1=0.0, width=None, quiet=True):
    """
        ccshift performs a cross-correlation and implements a shift based on
        the results
        Spectra must initially be on same x-axis for this to work effectively

        Arguments:
        s1 : numpy.ndarray
            master spectrum
        s2 : numpy.ndarray
            spectrum to be corrected (same shape as s1)
        x1 : numpy.ndarray
            x-axis for the two spectra
        width : int
            Width for the cross-correlation
            Default None == 15 (resolution elements)
        shift : float
            Shift to implement if known
            Default == 0.0 which allows the module to define the shift

        Output:
        s2cc_fin : numpy.ndarray
            Cross-correlated spectrum with the shift implemented,
                    rebinned onto input x-axis
        shift1 : float
            Shift used to correct s1
    """
    if shift1 == 0.:
        shift1, corr_array1 = crossc(s1, s2, width=width)

    # Define the xaxis step:
    step = np.abs(x1[1] - x1[0])
    xcorr = x1 + shift1 * step

    s2_mod = UnivariateSpline(x1, s2, s=0.)
    s2_corr = s2_mod(x1)
    # Now interpolate back onto original axis
    s2_corr_mod = UnivariateSpline(xcorr, s2_corr, s=0.)
    s2cc_fin = s2_corr_mod(x1)
    # Test to make sure correlation has worked
    shift2, corr_array2 = crossc(s1, s2cc_fin)

    if quiet is False:
        print '[INFO] Output from ccshift'
        print '[INFO] Cross-Correlation shift = ', shift1
        print '[INFO] After correction shift = ', shift2
        return s2cc_fin, shift1

    if abs(shift2) > abs(shift1):
        print '[WARNING] Cross-Correlation not effective.'
        print '[WARNING] No correction applied'
        s2cc_fin = s2
        pass

    return s2cc_fin
    pass


def crossc(s1, s2, ishift=None, width=None, i1=None, i2=None):
    """
    Normalized mean and covariance cross correlation offset between
    two input vectors or the same length.

    Required Arguments:

        s1: numpy.ndarray
        first spectrum
        s2: numpy.ndarray
        second spectrum
        ishift: float
        approximate offset (default = 0)
        width: float
        search width (default = 15)
        i1, i2: float
        region in first spectrum containing the feature(s)
        (default  i1=0, i2=n_elements(s2)-1)

    Output:

        offset: numpy.ndarray
        offset of s2 from s1 in data points
        corr: numpy.ndarray
        output correlation vector

    Note: Output is given in the form of a tuple (offset, corr)
    """

    # s1 - first spectrum
    # s2 - second spectrum

    # ishift - approximate offset (default = 0)
    # width - search width (default = 15)
    # i1,i2 - region in first spectrum containing the feature(s)
    #         (default  i1=0, i2=n_elements(s2)-1)

    # outputs:
    # offset: offset of s2 from s1 in data points
    # corr: output correlation vector

    if ishift is None:
        ishift = 0.0
    approx = int((ishift + 100000.5) - 100000)

    if width is None:
        width = 15

    ns = len(s1)

    if i1 is None:
        i1 = 0
    if i2 is None:
        i2 = ns - 1

    # ns2 = ns / 2
    width2 = width / 2
    it2_start = (i1 - approx + width2) if (i1 - approx + width2) > 0\
        else 0
    it2_end = (i2 - approx - width2) if (i2 - approx - width2) < (ns - 1)\
        else (ns - 1)
    nt = it2_end - it2_start + 1

    if nt < 1.0:
        print '[WARNING]',
        print 'modules.crossc',
        print 'Exception: region too small, ',
        print 'width too large, or ishift too large'
        # raise Exception("cross correlate - region too small,
        # width too large, or ishift too large")

    template2 = s2[it2_start:(it2_end + 1)]

    corr = np.zeros((width))
    mean2 = template2.sum() / nt
    sig2 = np.sqrt(np.sum((template2 - mean2) ** 2))
    diff2 = template2 - mean2

    for i in range(width):
        it1_start = it2_start - width2 + approx + i
        it1_end = it1_start + nt - 1
        template1 = s1[it1_start:(it1_end + 1)]
        mean1 = template1.sum() / nt
        sig1 = np.sqrt(np.sum((template1 - mean1) ** 2))
        diff1 = template1 - mean1

        if (sig1 == 0) or (sig2 == 0):
            print '[WARNING]',
            print 'modules.crossc',
            print 'Exception: zero variance computed'
            # raise Exception("cross correlate - zero variance computed")

        corr[i] = np.sum(diff1 * diff2) / (sig1 * sig2)

    # maxc = corr.max()
    k = np.where(corr == corr.max())
    k = k[0][0]

    if (k == 0) or (k == (width - 1.0)):
        print '[WARNING]',
        print 'modules.crossc',
        print 'Exception: maximum on edge of search area'
        offset = 0.
        corr = 0.
        return (offset, corr)
        # raise Exception("cross correlate - maximum on edge of search area")

    kmin = (corr[k - 1] - corr[k]) / \
        (corr[k - 1] + corr[k + 1] - 2.0 * corr[k]) - 0.5
    offset = k + kmin - width2 + approx

    return (offset, corr)