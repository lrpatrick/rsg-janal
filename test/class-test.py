""" An example of a class!"""
from scipy.io.idl import readsav


def gridorder(mpar):
    """
        sort the model grid and get something more well structured out!
    """
    # Order:
    mt = mpar.field('TURBS')[0]  # 21
    abuns = mpar.field('ABUNS')[0]  # 19
    logg = mpar.field('GRAVS')[0]  # 9
    teff = mpar.field('TEMPS')[0]  # 11
    return mt, abuns, logg, teff


class ReadGrid:
    """docstring for ReadGrid"""
    def __init__(self, savfile):
        self.all = readsav(savfile)
        self.grid = self.all['modelspec'][0][0]
        self.mpar = self.all['modelspec'][0][1]
        self.wave = self.all['modelspec'][0][2]
        self.mt, self.z, self.g, self.t = gridorder(self.mpar)


test2 = ReadGrid('models/MODELSPEC_2013sep12_nLTE_R10000_J_turb_abun_grav_temp-int.sav')
