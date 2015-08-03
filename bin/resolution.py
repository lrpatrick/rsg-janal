"""
Author: LRP
Date: 30-01-2015
Description:
Different routines regarding the resolution of the spectra and how to measure
this effectively

All dependencies are contained within astropy
Code is written to conform with PEP8 style guide

References:
Gazak (2014) Thesis

"""

import numpy as np
import sys


def degrade(wl, f, rin, rout):
    """
        Adapted to apply to the entire model grid by introducting slicing
        QUICK option has also been removed

        Adaptation of:
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
     Adapted to Python by A Bostroem May 24, 2013

     DEGRADER v. 0.1, 18 January 2012
     Jesus Maiz Apellaniz, IAA

     This function degrades the spectral resolution of a spectrum.

     Positional parameters:
     wl:        Wavelength in .
     f:        Normalized flux.
     rin:      Input R.
     rout:     Output R.

     Changes:
     v0.1:     QUICK changed from simple flag to number of pivot points.
               Change of definition of R from sigma-based to FWHM-based.
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    """
    if np.max(rout - rin) == 0:
        return f
    if np.max(rout - rin) > 0:
        sys.exit('Rout has to be smaller than Rin')
    if type(rin) != float and type(rin) != int:
        if len(rin) != 1 and len(rin) != len(wl):
            sys.exit('Incompatible wl and rin')

    if type(rout) != float and type(rout) != int:
        if len(rout) != 1 and len(rout) != len(wl):
            sys.exit('Incompatible wl and rout')

    reff = 1.0 / np.sqrt(1.0 / float(rout)**2 - 1.0 / float(rin)**2)
    ff = np.zeros(np.shape(f))
    s2fwhm = 2.0*np.sqrt(2.0*np.log(2))
    dleff = wl / (s2fwhm*reff)
    for indx in xrange(len(wl)):
        num = np.exp(-0.5*(wl - wl[indx])**2 / dleff[indx]**2)
        kern = num / np.sum(num)
        # Altered for a ND grid of spectra
        i = [slice(None)]*(len(f.shape) - 1) + [indx]
        # print i
        ff[i] = np.sum(f*kern, axis=len(f.shape) - 1)
        # ff[:, :, :, :, indx] = np.sum(f*kern, axis=len(f.shape) - 1)
    return ff
