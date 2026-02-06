"""HEALPix chunking and streaming I/O utilities."""

import healpy as hp
import numpy as np


def get_healpix_pixels_in_region(nside, ra_min, ra_max, dec_min, dec_max):
    """Get all HEALPix pixel indices that overlap a rectangular sky region.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.
    ra_min, ra_max : float
        RA range in degrees.
    dec_min, dec_max : float
        Dec range in degrees.

    Returns
    -------
    pixels : np.ndarray
        Sorted array of HEALPix pixel indices (RING ordering).
    """
    # Generate a grid of test points covering the region
    n_ra = max(int((ra_max - ra_min) * 10), 100)
    n_dec = max(int((dec_max - dec_min) * 10), 100)
    ra_grid = np.linspace(ra_min, ra_max, n_ra)
    dec_grid = np.linspace(dec_min, dec_max, n_dec)
    ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)

    # Convert to theta, phi for healpy (colatitude, longitude in radians)
    theta = np.radians(90.0 - dec_mesh.ravel())
    phi = np.radians(ra_mesh.ravel())

    pixels = hp.ang2pix(nside, theta, phi, nest=False)
    return np.unique(pixels)


def get_all_healpix_pixels(nside):
    """Get all HEALPix pixel indices for a given nside.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.

    Returns
    -------
    pixels : np.ndarray
        Array of all pixel indices [0, npix).
    """
    return np.arange(hp.nside2npix(nside))


def pixel_boundaries(nside, pixel):
    """Get approximate RA/Dec boundaries for a HEALPix pixel.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.
    pixel : int
        Pixel index (RING ordering).

    Returns
    -------
    ra_center, dec_center, radius : float
        Center coordinates and approximate radius in degrees.
    """
    theta, phi = hp.pix2ang(nside, pixel, nest=False)
    ra_center = np.degrees(phi)
    dec_center = 90.0 - np.degrees(theta)
    # Approximate pixel radius
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    radius = np.sqrt(pixel_area / np.pi)
    return ra_center, dec_center, radius


def radec_to_pixel(nside, ra, dec):
    """Convert RA/Dec to HEALPix pixel index.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.
    ra, dec : float or array
        Coordinates in degrees.

    Returns
    -------
    pixel : int or array
        HEALPix pixel index (RING ordering).
    """
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=False)
