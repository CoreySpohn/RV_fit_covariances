import astropy.io.fits
import astropy.units as u
import numpy as np
import pandas as pd

from systemobject import SystemObject


class System:
    """
    Class for the whole stellar system
    """

    def __init__(self, infile):
        star_ext = 3
        planet_ext = 4
        h = astropy.io.fits.getheader(infile, ext=0)  # read header of first extension
        n_ext = h["N_EXT"]  # get the largest extension
        nplanets = n_ext - 3

        self.star = SystemObject(infile, star_ext)
        self.planets = []
        for i in range(nplanets):  # loop over all planets
            self.planets.append(SystemObject(infile, planet_ext + i))

    def get_rv(self):
        """
        Return the RV data for the star and the planets
        """
        star_rv = np.array(self.star.vz)
        planet_rv = np.zeros((len(self.star.t), len(self.planets)))
        for i, planet in enumerate(self.planets):
            planet_rv[:, i] = planet.vz

        star_rv = star_rv * self.star.vz.unit
        planet_rv = planet_rv * planet.vz.unit
        return star_rv, planet_rv
