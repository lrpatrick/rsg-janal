import numpy as np
import sys
import scipy


def degrader(wl, f, rin, rout, quick=False):
    '''
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ; Adapted to Python by A Bostroem May 24, 2013

    ; DEGRADER v. 0.1, 18 January 2012
    ; Jesus Maiz Apellaniz, IAA
    ;
    ; This function degrades the spectral resolution of a spectrum.
    ;
    ; Positional parameters:
    ; wl:        Wavelength in .
    ; f:        Normalized flux.
    ; rin:      Input R.
    ; rout:     Output R.
    ;
    ; Keyword parameters:
    ; QUICK:    Flag to do a quick degradation (less accurate).
    ;           If a number > 1,
    ;           it uses that as the number of pivot points.
    ;
    ; Changes:
    ; v0.1:     QUICK changed from simple flag to number of pivot points.
    ;           Change of definition of R from sigma-based to FWHM-based.
    ;
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    '''

    if np.max(rout - rin) == 0:
        return f
    if np.max(rout - rin) > 0:
        sys.exit('Rout has to be smaller than Rin')
    # if len(f) != len(wl):
    #     sys.exit('Incompatible wl and f')

    if type(rin) != float and type(rin) != int:
        if len(rin) != 1 and len(rin) != len(wl):
            sys.exit('Incompatible wl and rin')
    if type(rout) != float and type(rout) != int:
        if len(rout) != 1 and len(rout) != len(wl):
            sys.exit('Incompatible wl and rout')
    reff = 1.0 / np.sqrt(1.0/float(rout)**2 - 1.0/float(rin)**2)
    ff = np.zeros(np.shape(f))
    s2fwhm = 2.0*np.sqrt(2.0*np.log(2))
    if quick == False:
        dleff = wl/(s2fwhm*reff)
        for indx in range(len(wl)):
            kern = np.exp(-0.5*(wl - wl[indx])**2/dleff[indx]**2)
            kern = kern/np.sum(kern)
            ff[indx] = np.sum(f*kern)
    else:
        if type(rin) != float and type(rin) != int:
            if len(rin) != 1 or len(rout) != 1:
                sys.exit('variables rin and rout not implemented for quick option')
        if ((wl[1:] - wl[:-1]) < 0).any():
            sys.exit('wl has to be monotonically increasing')
        if quick < 2:
            quick = 2
        else:
            quick = round(quick)

        ilp = np.int_(np.round((len(wl) - 1)*np.float_(np.arange(quick))/(quick - 1)))
        ddlm = np.median(wl[1:] - wl[:-1])

        dlmin = wl[ilp[0:quick-1]]/(s2fwhm*reff)
        dlmax = wl[ilp[1:quick]]/(s2fwhm*reff)
        npxmin = np.ceil(3.0*dlmin/ddlm)
        npxmax = np.ceil(3.0*dlmax/ddlm)
        for indx in range(int(quick) - 1):
            w = (wl[ilp[indx+1]]-wl[ilp[indx]:ilp[indx+1]+1])/(wl[ilp[indx+1]] - wl[ilp[indx]])
            kmin = np.exp(-0.5*(-3.0+3.0*np.float_(np.arange(2*npxmax[indx]+1))/npxmin[indx])**2)
            kmin = kmin/np.sum(kmin)
            fmin = scipy.ndimage.filters.convolve(f, kmin, mode='reflect')
            kmax = np.exp(-0.5*(-3.0+3.0*np.float_(np.arange(2*npxmax[indx]+1))/npxmax[indx])**2)
            kmax = kmax/np.sum(kmax)
            fmax = scipy.ndimage.filters.convolve(f, kmax, mode='reflect')
            ff[ilp[indx]:ilp[indx+1]+1] = w*fmin[ilp[indx]:ilp[indx+1]+1]+(1 - w)*fmax[ilp[indx]:ilp[indx+1]+1]
    return ff
