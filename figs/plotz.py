import matplotlib.pyplot as plt
import numpy as np
from scipy.io.idl import readsav

import sys
sys.path.append("/home/lee/Work/RSG-JAnal/bin/.")

import contfit
import resolution as res


def trimspec(w1, w2, s2):
    """Trim s2 and w2 to match w1"""
    roi = np.where((w2 > w1.min()) & (w2 < w1.max()))[0]
    return w2[roi], s2[roi]

mod = readsav(
    '../models/MODELSPEC_2013sep12_nLTE_R10000_J_turb_abun_grav_temp-int.sav')

grid = mod['modelspec'][0][0]
par = mod['modelspec'][0][1]
wave = mod['modelspec'][0][2]

n6822 = np.genfromtxt('../../ngc6822/Spectra/N6822-spec-24AT.v2-sam.txt')
nspec = n6822[:, 1:] / np.median(n6822[:, 1:])
owave, ospec = trimspec(wave, n6822[:, 0], nspec)

mssam = contfit.specsam(wave, grid, owave)
mdeg = res.degrade(owave, mssam, 10000, 3000)

# ############################################################################
# vary-micro:
f, ax = plt.subplots(2, 2, figsize=(12, 12))
# ax[0][0].set_xlabel(r'Wavelength ($\mu$m)')
ax[0][0].set_ylabel(r'Normalised Flux')
ax[0][0].plot(owave, mdeg[0, 4, 4, 5], label=r'$\xi$ = 1.0')
ax[0][0].plot(owave, mdeg[10, 4, 4, 5], label=r'$\xi$ = 3.0')
ax[0][0].plot(owave, mdeg[20, 4, 4, 5], label=r'$\xi$ = 5.0')
ax[0][0].axis((1.184, 1.193, 0.62, 1.01))
# ax[0].legend(loc=4)
ax[0][0].get_xaxis().get_major_formatter().set_useOffset(False)
ax[0][0].minorticks_on()

# ax[0][1].set_xlabel(r'Wavelength ($\mu$m)')
# ax[0][1].set_ylabel(r'Normalised Flux')
ax[0][1].plot(owave, mdeg[0, 4, 4, 5], label=r'$\xi$ = 1.0')
ax[0][1].plot(owave, mdeg[10, 4, 4, 5], label=r'$\xi$ = 3.0')
ax[0][1].plot(owave, mdeg[20, 4, 4, 5], label=r'$\xi$ = 5.0')
ax[0][1].axis((1.19349, 1.2025, 0.62, 1.01))
# ax[1].legend(loc=4)
ax[0][1].get_xaxis().get_major_formatter().set_useOffset(False)
ax[0][1].minorticks_on()

ax[1][0].set_xlabel(r'Wavelength ($\mu$m)')
ax[1][0].set_ylabel(r'Normalised Flux')
ax[1][0].plot(owave, mdeg[0, 4, 4, 5], label=r'$\xi$ = 1.0')
ax[1][0].plot(owave, mdeg[10, 4, 4, 5], label=r'$\xi$ = 3.0')
ax[1][0].plot(owave, mdeg[20, 4, 4, 5], label=r'$\xi$ = 5.0')
ax[1][0].axis((1.19909, 1.20817, 0.62, 1.01))
# ax[2].legend(loc=4)
ax[1][0].get_xaxis().get_major_formatter().set_useOffset(False)
ax[1][0].minorticks_on()

ax[1][1].set_xlabel(r'Wavelength ($\mu$m)')
# ax[1][1].set_ylabel(r'Normalised Flux')
ax[1][1].plot(owave, mdeg[0, 4, 4, 5], label=r'$\xi$ = 1.0')
ax[1][1].plot(owave, mdeg[10, 4, 4, 5], label=r'$\xi$ = 3.0')
ax[1][1].plot(owave, mdeg[20, 4, 4, 5], label=r'$\xi$ = 5.0')
ax[1][1].axis((1.206, 1.215, 0.62, 1.01))
ax[1][1].legend(loc=4)
ax[1][1].get_xaxis().get_major_formatter().set_useOffset(False)
ax[1][1].minorticks_on()

plt.tight_layout()

# ############################################################################
# Vary Z
f, ax = plt.subplots(2, 2, figsize=(12, 12))
# ax[0][0].set_xlabel(r'Wavelength ($\mu$m)')
ax[0][0].set_ylabel(r'Normalised Flux')
ax[0][0].plot(owave, mdeg[14, 0, 4, 5], label='[Z] = -1.0')
ax[0][0].plot(owave, mdeg[14, 4, 4, 5], label='[Z] = -0.5')
ax[0][0].plot(owave, mdeg[14, 9, 4, 5], label='[Z] =  0.0')
ax[0][0].axis((1.184, 1.193, 0.65, 1.01))
# ax[0].legend(loc=4)
ax[0][0].get_xaxis().get_major_formatter().set_useOffset(False)
ax[0][0].minorticks_on()

# ax[0][1].set_xlabel(r'Wavelength ($\mu$m)')
# ax[0][1].set_ylabel(r'Normalised Flux')
ax[0][1].plot(owave, mdeg[14, 0, 4, 5], label='[Z] = -1.0')
ax[0][1].plot(owave, mdeg[14, 4, 4, 5], label='[Z] = -0.5')
ax[0][1].plot(owave, mdeg[14, 9, 4, 5], label='[Z] =  0.0')
ax[0][1].axis((1.19349, 1.2025, 0.65, 1.01))
# ax[1].legend(loc=4)
ax[0][1].get_xaxis().get_major_formatter().set_useOffset(False)
ax[0][1].minorticks_on()

ax[1][0].set_xlabel(r'Wavelength ($\mu$m)')
ax[1][0].set_ylabel(r'Normalised Flux')
ax[1][0].plot(owave, mdeg[14, 0, 4, 5], label='[Z] = -1.0')
ax[1][0].plot(owave, mdeg[14, 4, 4, 5], label='[Z] = -0.5')
ax[1][0].plot(owave, mdeg[14, 9, 4, 5], label='[Z] =  0.0')
ax[1][0].axis((1.19909, 1.20817, 0.65, 1.01))
# ax[2].legend(loc=4)
ax[1][0].get_xaxis().get_major_formatter().set_useOffset(False)
ax[1][0].minorticks_on()

ax[1][1].set_xlabel(r'Wavelength ($\mu$m)')
# ax[1][1].set_ylabel(r'Normalised Flux')
ax[1][1].plot(owave, mdeg[14, 0, 4, 5], label='[Z] = -1.0')
ax[1][1].plot(owave, mdeg[14, 4, 4, 5], label='[Z] = -0.5')
ax[1][1].plot(owave, mdeg[14, 9, 4, 5], label='[Z] =  0.0')
ax[1][1].axis((1.206, 1.215, 0.65, 1.01))
ax[1][1].legend(loc=4)
ax[1][1].get_xaxis().get_major_formatter().set_useOffset(False)
ax[1][1].minorticks_on()

plt.tight_layout()
# ############################################################################
# Vary logg
f, ax = plt.subplots(2, 2, figsize=(12, 12))
# ax[0][0].set_xlabel(r'Wavelength ($\mu$m)')
ax[0][0].set_ylabel(r'Normalised Flux')
ax[0][0].plot(owave, mdeg[14, 4, 0, 5], label=r'log$g$ = -1.0')
ax[0][0].plot(owave, mdeg[14, 4, 4, 5], label=r'log$g$ =  0.0')
ax[0][0].plot(owave, mdeg[14, 4, 8, 5], label=r'log$g$ =  1.0')
ax[0][0].axis((1.184, 1.193, 0.65, 1.01))
# ax[0].legend(loc=4)
ax[0][0].get_xaxis().get_major_formatter().set_useOffset(False)
ax[0][0].minorticks_on()

# ax[0][1].set_xlabel(r'Wavelength ($\mu$m)')
# ax[0][1].set_ylabel(r'Normalised Flux')
ax[0][1].plot(owave, mdeg[14, 4, 0, 5], label=r'log$g$ = -1.0')
ax[0][1].plot(owave, mdeg[14, 4, 4, 5], label=r'log$g$ =  0.0')
ax[0][1].plot(owave, mdeg[14, 4, 8, 5], label=r'log$g$ =  1.0')
ax[0][1].axis((1.19349, 1.2025, 0.65, 1.01))
# ax[1].legend(loc=4)
ax[0][1].get_xaxis().get_major_formatter().set_useOffset(False)
ax[0][1].minorticks_on()

ax[1][0].set_xlabel(r'Wavelength ($\mu$m)')
ax[1][0].set_ylabel(r'Normalised Flux')
ax[1][0].plot(owave, mdeg[14, 4, 0, 5], label=r'log$g$ = -1.0')
ax[1][0].plot(owave, mdeg[14, 4, 4, 5], label=r'log$g$ =  0.0')
ax[1][0].plot(owave, mdeg[14, 4, 8, 5], label=r'log$g$ =  1.0')
ax[1][0].axis((1.19909, 1.20817, 0.65, 1.01))
# ax[2].legend(loc=4)
ax[1][0].get_xaxis().get_major_formatter().set_useOffset(False)
ax[1][0].minorticks_on()

ax[1][1].set_xlabel(r'Wavelength ($\mu$m)')
# ax[1][1].set_ylabel(r'Normalised Flux')
ax[1][1].plot(owave, mdeg[14, 4, 0, 5], label=r'log$g$ = -1.0')
ax[1][1].plot(owave, mdeg[14, 4, 4, 5], label=r'log$g$ =  0.0')
ax[1][1].plot(owave, mdeg[14, 4, 8, 5], label=r'log$g$ =  1.0')
ax[1][1].axis((1.206, 1.215, 0.65, 1.01))
ax[1][1].legend(loc=4)
ax[1][1].get_xaxis().get_major_formatter().set_useOffset(False)
ax[1][1].minorticks_on()

plt.tight_layout()
# ############################################################################
# Vary T
f, ax = plt.subplots(2, 2, figsize=(12, 12))
# ax[0][0].set_xlabel(r'Wavelength ($\mu$m)')
ax[0][0].set_ylabel(r'Normalised Flux')
ax[0][0].plot(owave, mdeg[14, 4, 4, 0], label=r'T$_{eff}$ = 3400')
ax[0][0].plot(owave, mdeg[14, 4, 4, 5], label=r'T$_{eff}$ = 3900')
ax[0][0].plot(owave, mdeg[14, 4, 4, 10], label=r'T$_{eff}$ = 4400')
ax[0][0].axis((1.184, 1.193, 0.65, 1.01))
# ax[0].legend(loc=4)
ax[0][0].get_xaxis().get_major_formatter().set_useOffset(False)
ax[0][0].minorticks_on()

# ax[0][1].set_xlabel(r'Wavelength ($\mu$m)')
# ax[0][1].set_ylabel(r'Normalised Flux')
ax[0][1].plot(owave, mdeg[14, 4, 4, 0], label=r'T$_{eff}$ = 3400')
ax[0][1].plot(owave, mdeg[14, 4, 4, 5], label=r'T$_{eff}$ = 3900')
ax[0][1].plot(owave, mdeg[14, 4, 4, 10], label=r'T$_{eff}$ = 4400')
ax[0][1].axis((1.19349, 1.2025, 0.65, 1.01))
# ax[1].legend(loc=4)
ax[0][1].get_xaxis().get_major_formatter().set_useOffset(False)
ax[0][1].minorticks_on()

ax[1][0].set_xlabel(r'Wavelength ($\mu$m)')
ax[1][0].set_ylabel(r'Normalised Flux')
ax[1][0].plot(owave, mdeg[14, 4, 4, 0], label=r'T$_{eff}$ = 3400')
ax[1][0].plot(owave, mdeg[14, 4, 4, 5], label=r'T$_{eff}$ = 3900')
ax[1][0].plot(owave, mdeg[14, 4, 4, 10], label=r'T$_{eff}$ = 4400')
ax[1][0].axis((1.19909, 1.20817, 0.65, 1.01))
# ax[2].legend(loc=4)
ax[1][0].get_xaxis().get_major_formatter().set_useOffset(False)
ax[1][0].minorticks_on()

ax[1][1].set_xlabel(r'Wavelength ($\mu$m)')
# ax[1][1].set_ylabel(r'Normalised Flux')
ax[1][1].plot(owave, mdeg[14, 4, 4, 0], label=r'T$_{eff}$ = 3400')
ax[1][1].plot(owave, mdeg[14, 4, 4, 5], label=r'T$_{eff}$ = 3900')
ax[1][1].plot(owave, mdeg[14, 4, 4, 10], label=r'T$_{eff}$ = 4400')
ax[1][1].axis((1.206, 1.215, 0.65, 1.01))
ax[1][1].legend(loc=4)
ax[1][1].get_xaxis().get_major_formatter().set_useOffset(False)
ax[1][1].minorticks_on()

plt.tight_layout()
# ############################################################################
