import astropy.io.fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
from tqdm import tqdm

from system import System
from systemobject import SystemObject

"""
Next step:
    Having a class called system which contains the solar system properties
    seems like the way to go.
"""


def load_system(inputfile):
    """
    This will pull relevant data from the fits file

    Args:
        inputfile (Path):
            ExoVista fits file for a star

    Returns:
        TODO
    """
    system = System(inputfile)
    star_rv, planet_rv = system.get_rv()
    breakpoint()

    # planet_data = get_all_planet_data(inputfile, planet_ext, nplanets)
    # cmap = plt.get_cmap("viridis")
    # system_video(star_data, planet_data, cmap)


def system_video(star_data, planet_data, cmap, lim=10):
    """
    Function to visualize the movement of the planets

    """
    for i, time in enumerate(tqdm(planet_data["t"][0])):
        fig, (ax_xy, ax_zy) = plt.subplots(ncols=2)
        # Create x, y plot
        x_data = [data["x"][i] for _, data in planet_data.iterrows()]
        y_data = [data["y"][i] for _, data in planet_data.iterrows()]
        z_data = [data["z"][i] for _, data in planet_data.iterrows()]
        all_z_data = [data["z"] for _, data in planet_data.iterrows()]
        radii = planet_data["Rp"]
        ax_xy = plot_all_planet_locations(
            ax_xy, x_data, y_data, z_data, all_z_data, radii, cmap
        )
        ax_xy.scatter(star_data[0][i], star_data[1][i], color="k")
        ax_xy.set_xlabel("x (AU)")
        ax_xy.set_ylabel("y (AU)")
        ax_xy.set_xlim([-lim, lim])
        ax_xy.set_ylim([-lim, lim])
        ax_xy.set_aspect("equal")

        ax_zy = plot_all_planet_locations(
            ax_zy, z_data, y_data, x_data, all_z_data, radii, cmap
        )
        ax_zy.scatter(star_data[0][i], star_data[1][i], color="k")
        ax_zy.set_xlabel("z (AU)")
        ax_zy.set_ylabel("y (AU)")
        ax_zy.set_xlim([-lim, lim])
        ax_zy.set_ylim([-lim, lim])
        ax_zy.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(f"figures/system_video/img_{i:03}.png")
        plt.close()


def plot_all_planet_locations(
    ax, x_data, y_data, z_data, all_z_data, planet_radii, cmap
):
    """
    Put each planet in the scatter plot
    """
    for n, i, x, y, z, all_z, Rp in zip(
        np.arange(0, len(x_data) + 1),
        np.linspace(0, 1, len(x_data)),
        x_data,
        y_data,
        z_data,
        all_z_data,
        planet_radii,
    ):
        color = cmap(i)
        ms = planet_marker_size(
            z, all_z, base_size=5 + Rp.to(u.R_earth).value, factor=0.1
        )
        ax.scatter(x, y, label=f"Planet {n}", color=color, s=ms)
    return ax


def planet_marker_size(z, all_z, base_size=5, factor=0.5):
    """
    Make the planet marker smaller when the planet is behind the star in its orbit
    """
    z_range = np.abs(max(all_z) - min(all_z))

    # Want being at max z to correspond to a factor of 1 and min z to be a factor of negative 1
    scaled_z = (
        2
        * (z.decompose().value - min(all_z).decompose().value)
        / z_range.decompose().value
        - 1
    )

    marker_size = base_size * (1 + factor * scaled_z)

    return marker_size


def get_star_data(data):
    x = data[:, 9]
    y = data[:, 10]
    z = data[:, 11]
    return x, y, z


def load_rv(data):
    """
    This will extract the RV signal of a full system

    Args:
        data (numpy.ndarray):
            ExoVista star data array

    Returns:
        rv_signal (array):
            System's true radial velocity
        rv_times (array):
            Times associated with each radial velocity measurement

    """
    rv_signal = data[:, 14] * u.AU / u.yr
    rv_times = data[:, 0] * u.yr

    return rv_signal, rv_times


def get_all_planet_data(inputfile, planet_ext, nplanets):
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
        # planet = SystemObject(inputfile, planet_ext + i)
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


def load_scene(inputfile, time=0):
    """
    This routine reads the output .fits file produced by exoVista
    and converts all quantities to flux at the same spectral resolution.
    Pixel coordinates of all objects are returned. Note that pixel coordinates
    are in 00LL format, where (0,0) is the lower-left corner of the lower-left
    pixel (and (0.5,0.5) is the center of the lower-left pixel). The star
    should be located at exactly (npix/2,npix/2)--the intersection of
    the central 4 pixels.

    --- INPUTS ---
    inputfile = filename and path of fits file containing scene
    time = desired time (default = 0)

    --- RETURNS ---
    lambda: wavelength vector (microns)
    xystar: location of star (pixels)
    fstar: stellar flux vector (Jy)
    xyplanet: Nplanets x 2 array of planet locations (pixels)
    fplanet: Nplanets x nlambda array of planet fluxes (Jy)
    diskimage: npix x npix x nlambda disk image cube (Jy per pix)
    angdiam: angular diameter of star (mas)
    pixscale: pixel scale (mas)
    """

    # Define extension numbers
    lam_ext = 0
    disklam_ext = 1
    disk_ext = 2
    star_ext = 3
    planet_ext = 4  # first planet extension
    h = astropy.io.fits.getheader(inputfile, ext=0)  # read header of first extension
    n_ext = h["N_EXT"]  # get the largest extension #

    # Get wavelength array
    lambdas, h = astropy.io.fits.getdata(
        inputfile, ext=lam_ext, header=True
    )  # read wavelength extension
    nlambda = len(lambdas)

    # STEP 1: STAR
    # Need to determine x, y, and fstar
    xystar = np.zeros(2)
    fstar = np.zeros(nlambda)
    d, h = astropy.io.fits.getdata(inputfile, ext=star_ext, header=True)
    angdiam = h["ANGDIAM"]
    pixscale = h["PXSCLMAS"]

    if d.ndim == 1:
        d = np.expand_dims(d, 1)
    t = d[:, 0]  # time vector
    x = d[:, 1]  # heliocentric x location vector (pix)
    y = d[:, 2]  # heliocentric y location vector (pix)
    xystar[0] = x[0]  # pick the first entry by default
    xystar[1] = y[0]
    fstar = d[0, 15 : 15 + nlambda]  # grab the stellar flux of first time entry
    # If the fits file contains a vector of times, interpolate...
    if len(t) > 1:
        x_interp = scipy.interpolate.interp1d(t, x, kind="quadratic")
        y_interp = scipy.interpolate.interp1d(t, y, kind="quadratic")
        xystar[0] = x_interp(time)
        xystar[1] = y_interp(time)
    # if fits file contains a vector of times, interpolate...
    if len(t) > 1:
        for ii in range(nlambda):
            fstar_interp = scipy.interpolate.interp1d(
                t, d[:, 15 + ii], kind="quadratic"
            )
            fstar[ii] = fstar_interp(time)

    # STEP 2: PLANETS
    # ;Need to determine x, y, and fplanet
    nplanets = n_ext - 3
    xyplanet = np.zeros((nplanets, 2))
    fplanet = np.zeros((nplanets, nlambda))
    for ip in range(nplanets):  # loop over all planets
        d, h = astropy.io.fits.getdata(inputfile, ext=planet_ext + ip, header=True)
        if d.ndim == 1:
            d = np.expand_dims(d, 1)
        t = d[:, 0]  # time vector
        x = d[:, 1]  # heliocentric x position vector (pix)
        y = d[:, 2]  # heliocentric y position vector (pix)
        xyplanet[ip, 0] = x[0]  # pick the first entry by default
        xyplanet[ip, 1] = y[0]
        contrast = d[0, 15 : 15 + nlambda]
        fplanet[ip, :] = contrast * fstar  # convert to flux

        if len(t) > 1:
            for ii in range(nlambda):
                contrast_interp = scipy.interpolate.interp1d(
                    t, d[:, 15 + ii], kind="quadratic"
                )
                contrast = contrast_interp(time)
                fplanet[ip, ii] = contrast * fstar[ii]

    # STEP 3: DISK
    lambdas_disk = astropy.io.fits.getdata(
        inputfile, ext=disklam_ext
    )  # disk wavelengths
    nlambdas_disk = len(lambdas_disk)
    temp = astropy.io.fits.getdata(inputfile, ext=disk_ext)
    contrast = temp[0:nlambdas_disk, :, :]  # 3D contrast data cube
    # cprecision = temp[nlambdas_disk, :, :]  # 2D contrast precision

    # Interpolate the disk image cube to the desired wavelength spacing
    lambda_indices = np.searchsorted(lambdas_disk, lambdas) - 1

    # index in log lambda space (have to add on fractional indices)
    frac_lambda_indices = lambda_indices + (
        np.log(lambdas) - np.log(lambdas_disk[lambda_indices])
    ) / (
        np.log(lambdas_disk[lambda_indices + 1]) - np.log(lambdas_disk[lambda_indices])
    )

    contrast_interp = scipy.interpolate.interp1d(
        np.arange(len(lambdas_disk)), contrast, axis=0, kind="cubic"
    )
    diskimage = np.multiply(contrast_interp(frac_lambda_indices).T, fstar).T

    return (lambdas, xystar, fstar, xyplanet, fplanet, diskimage, angdiam, pixscale)
