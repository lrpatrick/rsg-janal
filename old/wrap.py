"""
    Author: LRP
    Date: 13-03-2015
"""
import contfit
# from degradespec import degrader
import bestfit
import numpy as np
# This is supposed to be the only function to call!
# Then this function calls everything from it


def rsgjanal(ospec, owave, ores, mspec, quiet=True):
    """
        Trimmed spectra, nothing more to do than run the routines!
        Full artillery!
        Make an input file for filenames and resolution for each spectrum

    """

    # 1. Find spectral resolution - lets start by assuimng I know this
    # 2. Resample models onto ooresbservations
    # 3. Degrade model spectrum to match resolution of observations
    # -- use Jesus' routine
    # 4. Fit the continuum
    # 5. Derive stellar parameters ... -- done outside function
    # 6. Define the errors ... -- done outside function

    # Many of the steps are done before this function ...
    # This should be the function to execute all of that!
    # It can't be run within a 4D for loop and
    # 1.:

    # 2.: Resample
    # mssam = contfit.specsam(mwave, mspec, owave)
    # 3.:
    # mdeg = degrader(owave, mspec, mres, ores)
    # 4.:
    # Test: is contfit to blame for our mismatched parameters?
    cft = contfit.contfit2(ores, owave, mspec, ospec)

    oscale = ospec * cft(owave)
    # Calculate Chisq
    chi = bestfit.chicalc(owave, oscale, mspec)
    # chi = bestfit.chisq(oscale, np.std(oscale), mspec)

    if quiet is not True:
        return oscale, chi, cft
    return oscale, chi
