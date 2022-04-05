#!/usr/bin/env python
""" Fast convolution in the UV domain """
__author__ = "Wasim Raja"

from typing import Tuple
import numpy as np
import astropy.units as units
from radio_beam import Beam
from astropy import units as u
import racs_tools.gaussft as gaussft


def convolve(
    image: np.ndarray, old_beam: Beam, new_beam: Beam, dx: u.Quantity, dy: u.Quantity
) -> Tuple[np.ndarray, float]:
    """Convolve by X-ing in the Fourier domain.
        - convolution with Gaussian kernels only 
        - no need for generation of a kernel image
        - direct computation of the FT of the kernel
        - dimension of FT = dimension of image

    Args:
        image (np.ndarray): The image to be convolved.
        old_beam (Beam): Current image PSF.
        new_beam (Beam): Target image PSF.
        dx (u.Quantity): Grid size in x (e.g. CDELT1)
        dy (u.Quantity): Grid size in y (e.g. CDELT2)

    Returns:
        Tuple[np.ndarray, float]: (convolved image, scaling factor)
    """
    nx = image.shape[0]
    ny = image.shape[1]

    # The coordinates in FT domain:
    u_image = np.fft.fftfreq(nx, d=dx.to(units.rad).value)
    v_image = np.fft.fftfreq(ny, d=dy.to(units.rad).value)

    g_final = np.zeros((nx, ny), dtype=float)
    [g_final, g_ratio] = gaussft.gaussft(
        bmin_in=old_beam.minor.to(units.deg).value,
        bmaj_in=old_beam.major.to(units.deg).value,
        bpa_in=old_beam.pa.to(units.deg).value,
        bmin=new_beam.minor.to(units.deg).value,
        bmaj=new_beam.major.to(units.deg).value,
        bpa=new_beam.pa.to(units.deg).value,
        u=u_image,
        v=v_image,
        nx=nx,
        ny=ny,
    )
    # Perform the x-ing in the FT domain
    im_f = np.fft.fft2(image)

    # Now convolve with the desired Gaussian:
    M = np.multiply(im_f, g_final)
    im_conv = np.fft.ifft2(M)
    im_conv = np.real(im_conv)

    return im_conv, g_ratio
