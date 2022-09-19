import astropy.io.fits
import astropy.units as u
import numpy as np
import pandas as pd


class SystemObject:
    """
    Class for the planets and stars in the exoVista systems
    """

    def __init__(self, infile, obj_ext):

        # Get the object's data from the fits file
        obj_data, obj_header = astropy.io.fits.getdata(infile, ext=obj_ext, header=True)

        # Time data
        self.t = obj_data[:, 0] * u.yr

        # Position data
        self.x = (obj_data[:, 9] * u.AU,)
        self.y = (obj_data[:, 10] * u.AU,)
        self.z = (obj_data[:, 11] * u.AU,)

        # Velocity data
        self.vx = obj_data[:, 12] * u.AU / u.yr
        self.vy = obj_data[:, 13] * u.AU / u.yr
        self.vz = obj_data[:, 14] * u.AU / u.yr

        if "Planet data array" == obj_header["COMMENT"]:
            self.is_planet = True
            self.is_star = False
        else:
            self.is_planet = False
            self.is_star = True

        if self.is_planet:
            # Assign the planet's keplerian orbital elements
            self.a = obj_header["A"] * u.AU
            self.e = obj_header["E"]
            self.i = obj_header["I"] * u.deg
            self.W = obj_header["LONGNODE"] * u.deg
            self.w = obj_header["ARGPERI"] * u.deg

            # Assign the planet's mass/radius information
            self.mass = obj_header["M"] * u.M_earth
            self.radius = obj_header["R"] * u.R_earth

            # Assign the planet's time-varying mean anomaly and contrast
            self.M = obj_data[:, 8] * u.deg
            self.contrast = obj_data[:, 15]

        if self.is_star:
            # System identifiers
            self.exoVista_id = obj_header["STARID"]
            self.HIP_id = obj_header["HIP"]

            # System midplane information
            self.midplane_PA = obj_header["PA"] * u.deg  # Position angle
            self.midplane_I = obj_header["I"] * u.deg  # Inclination

            # Proper motion
            self.PMRA = obj_header["PMRA"] * u.mas / u.yr
            self.PMDEC = obj_header["PMDEC"] * u.mas / u.yr

            # Celestial coordinates
            self.RA = obj_header["RA"]
            self.DEC = obj_header["DEC"]
            self.dist = obj_header["DIST"] * u.pc

            # Spectral properties
            self.spectral_type = obj_header["TYPE"]
            self.MV = obj_header["M_V"]  # Absolute V band mag
            self.Bmag = obj_header["BMAG"]
            self.Vmag = obj_header["VMAG"]
            self.Rmag = obj_header["RMAG"]
            self.Imag = obj_header["IMAG"]
            self.Jmag = obj_header["JMAG"]
            self.Hmag = obj_header["HMAG"]
            self.Kmag = obj_header["KMAG"]

            # Stellar properties
            self.Lstar = obj_header["LSTAR"] * u.Lsun  # Bolometric luminosity
            self.Teff = obj_header["TEFF"] * u.K  # Effective temperature
            self.angdiam = obj_header["ANGDIAM"] * u.K  # Angular diameter
            self.mass = obj_header["MASS"] * u.M_sun
            self.radius = obj_header["RSTAR"] * u.R_sun


def get_object_data(inputfile, planet_ext, nplanets):
    """
    Gets all planet data into an easy to parse format

    Args:
        inputfile (Path):
            Path to exoVista fits file
        planet_ext (int):
            Extension that corresponds to the first planet data entry in the fits file
        nplanets (int):
            How many planets are included in the system

    Returns:
        planets_df (pandas.DataFrame):
            A dataframe containing the static keplerian orbital elements,
            planet mass, planet radius, mean anomaly, and barycentric velocity
    """
    planet_dicts = {}
    for i in range(nplanets):  # loop over all planets
        planet_data, planet_header = astropy.io.fits.getdata(
            inputfile, ext=planet_ext + i, header=True
        )
        elements = load_planet_elements(planet_data, planet_header)
        planet_dicts[i] = elements
    planets_df = pd.DataFrame.from_dict(planet_dicts, orient="index")
    return planets_df


def load_planet_elements(data, header):
    """
    This will extract the keplerian orbital elements for a planet

    Args:
        data (numpy.ndarray):
            ExoVista planet data array
        data (astropy fits header):
            ExoVista planet header

    Returns:
        a (astropy Quantity):
            Semi-major axis
        e (astropy Quantity):
            Eccentricity
        i (astropy Quantity):
            Orbital inclination
        W (astropy Quantity):
            Longitude of the ascending node
        w (astropy Quantity):
            Argument of pericenter
        M (astropy Quantity):
            Mean anomaly
        Mp (astropy Quantity):
            Planet mass
        Rp (astropy Quantity):
            Planet radius
        vz (astropy Quantity):
            Planet barycentric velocity
    """
    elements = {
        "a": header["A"] * u.AU,
        "e": header["E"],
        "i": header["I"] * u.deg,
        "W": header["LONGNODE"] * u.deg,
        "w": header["ARGPERI"] * u.deg,
        "Mp": header["M"] * u.M_earth,
        "Rp": header["R"] * u.R_earth,
        "t": data[:, 0] * u.yr,
        "M": data[:, 8] * u.deg,
        "x": data[:, 9] * u.AU,
        "y": data[:, 10] * u.AU,
        "z": data[:, 11] * u.AU,
        "vx": data[:, 12] * u.AU / u.yr,
        "vy": data[:, 13] * u.AU / u.yr,
        "vz": data[:, 14] * u.AU / u.yr,
        "contrast": data[:, 15],
    }

    return elements
