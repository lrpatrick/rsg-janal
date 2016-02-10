"""
Author: LRP
Date: 31-07-1015
Description:
Classes to read in necessary input files and order them correctly
"""
from __future__ import print_function

import sys
sys.path.append("/home/lee/Work/RSG-JAnal/bin/.")

import numpy as np
from scipy.io.idl import readsav
from uncertainties import ufloat
from uncertainties import unumpy


nom = unumpy.nominal_values
stddev = unumpy.std_devs


class ReadMod(object):
    """
        Simple class to read in model grid and alter the shape of it

        ... find out who desinged this grid! -- and destroy them?
        How could someone make the names and the grid a different order!!!

    """
    def __init__(self, savfile):
        self.all = readsav(savfile)
        self.grid = self.all['modelspec'][0][0]
        self.par = self.all['modelspec'][0][1]
        self.wave = self.all['modelspec'][0][2]
        self.pnames = self.par.dtype.names
        self.head = self.changehead()
        self.res = float(10000.)
        self.mt, self.z, self.g, self.t = self.gridorder()
        self.trim = maregion(self.wave, 1.1575, 1.22)
        self.tgrid = self.grid[:, :, :, :, self.trim]
        self.twave = self.wave[self.trim]

    def gridorder(self):
        """Sort the model grid and get something more well structured out!"""
        # Order:
        teff = self.par.field('TEMPS')[0].astype(float)
        abuns = self.par.field('ABUNS')[0]
        logg = self.par.field('GRAVS')[0]
        mt = self.par.field('TURBS')[0]
        # Return ordered similarlaly to the grid:
        return mt, abuns, logg, teff

    def parlimit(self, low, high, name):
        """Restrict a given model parameter, usually used with logg"""
        # Reset parameter first:
        par, idx = self.findpar(name)
        self.head[0][idx] = par
        self.parcut = np.where((par > low) & (par < high))[0]
        self.head[0][idx] = self.head[0][idx][self.parcut]
        print('[INFO] ' + name + ' range constrained by ReadMod.parlimit')
        print('[INFO] Range is now:', self.head[0][int(idx)])

    def findpar(self, name):
        """Look-up table style function using the names of parameters"""
        if name == 'TURBS':
            par = self.mt
            idx = 0
        if name == 'ABUNS':
            par = self.z
            idx = 1
        if name == 'GRAVS':
            par = self.g
            idx = 2
        if name == 'TEMPS':
            par == self.t
            idx = 3
        return par, idx

    def changehead(self):
        """Sort the parameters part of the grid out!"""
        # Should be more general!
        x = [self.par[0][3], self.par[0][1], self.par[0][2], self.par[0][0]]
        y = self.pnames[3], self.pnames[1], self.pnames[2], self.pnames[0]
        return (x, y)


class ReadObs(object):
    """
        ReadObs takes in:
        1. File containing spectra: fspec
        2. File containing info from spectra: fsinfo
        3. Distance modulus of galaxy: mu

        For this input it assumes the following structures:
        fspec: 1. Wavelength 2-N: Spectra
        fsinfo:
        0: ID
        1-9: Photometry:B V R I J err H err K err
        10-11: res. err
        12: S/N

        Notes:
        If observations are normalised used self.spec,
        if not, a simple median normalisation is apllied in self.nspec
    """
    def __init__(self, fspec, fsinfo, mu):
        self.fspec = fspec
        self.fsinfo = fsinfo
        self.mu = mu

        # Star info:
        self.info = np.genfromtxt(self.fsinfo)
        self.id = np.genfromtxt(self.fsinfo, usecols=0, dtype='S')
        self.phot = self.info[:, 1:15]
        self.res = self.info[:, 15]
        self.eres = self.info[:, 16]
        self.sn = self.info[:, 17]

        self.wavenspec = np.genfromtxt(self.fspec)
        self.wave = self.wavenspec[:, 0]
        self.spec = self.wavenspec[:, 1:]
        self.nspec = self.spec / np.median(self.spec, axis=0)

        # self.mk = unumpy.uarray(self.phot[:, 10], self.phot[:, 11])
        self.L = self.luminosity()
        # self.gup, self.glow = self.glimits()

    def luminosity(self):
        """Calculate Luminosity based on Davies et al. (2013) correction"""
        print('[INFO] Please enter filter to compute bolometric correction')
        band = raw_input('[INFO] Options [V, R, I, J, H, K]\n')
        options = ['V', 'R', 'I', 'J', 'H', 'K']
        while band not in options:
            print('[INFO] You fool! {} is not a valid option!'.format(band))
            print('[INFO] Please select a valid filter:')
            band = raw_input('[INFO] Options [V, R, I, J, H, K]\n')

        d = {'V': (ufloat(3.12, 0.06), ufloat(-0.29, 0.01), 2),
             'R': (ufloat(2.44, 0.07), ufloat(-0.34, 0.01), 4),
             'I': (ufloat(1.90, 0.08), ufloat(-0.37, 0.01), 6),
             'J': (ufloat(1.30, 0.09), ufloat(-0.39, 0.01), 8),
             'H': (ufloat(0.97, 0.10), ufloat(-0.40, 0.01), 10),
             'K': (ufloat(0.90, 0.11), ufloat(-0.40, 0.01), 12)}
        a, b, c = d[band]
        # a = d[band][0]
        # b = d[band][1]
        # c = d[band][2]
        self.band = unumpy.uarray(self.phot[:, c], self.phot[:, c + 1])
        if band == 'K':
            ak = 0.06  # For NGC2100
            print('[INFO] IS Extinction taken as Ak=0.06 for NGC2100')
            l = a + b*(self.band - self.mu - ak)
        l = a + b*(self.band - self.mu)
        return l


def maregion(wave, w1, w2):
    """Filter a region between w1 & w2"""
    return (wave > w1) & (wave < w2)


def maskobs(wl):
    """Mask observations using maregion"""
    omask = maregion(wl, 1.1915, 1.1935)
    omask += maregion(wl, 1.182, 1.184)
    omask += maregion(wl, 1.2080, 1.2087)
    return ~omask
