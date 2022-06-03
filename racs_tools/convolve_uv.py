#!/usr/bin/env python
""" Fast convolution in the UV domain """
__author__ = "Wasim Raja"

from typing import Tuple
import numpy as np
import astropy.units as units
from radio_beam import Beam
from astropy import units as u
from astropy import convolution
import scipy.signal
import racs_tools.gaussft as gaussft
import logging as log

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


def smooth(
    image: np.ndarray,
    old_beam: Beam,
    final_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    sfactor: float,
    conbeam: Beam = None,
    conv_mode: str = "robust",
) -> np.ndarray:
    """Apply smoothing to image

    Args:
        image (np.ndarray): 2D image array.
        old_beam (Beam): Current beam.
        final_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        sfactor (float): Scaling factor.
        conbeam (Beam, optional): Convoling beam to use. Defaults to None.
        conv_mode (str, optional): Convolution mode to use. Defaults to "robust".

    Returns:
        np.ndarray: Smoothed image.
    """
    if np.isnan(sfactor):
        log.warning("Beam larger than cutoff -- blanking")

        newim = np.ones_like(image) * np.nan
        return newim
    if conbeam is None:
        conbeam = final_beam.deconvolve(old_beam)
    if np.isnan(conbeam):
        return image * np.nan
    if np.isnan(image).all():
        return image
    if conbeam == Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg) and sfactor == 1:
        return image
    # using Beams package
    log.debug(f"Old beam is {old_beam!r}")
    log.debug(f"Using convolving beam {conbeam!r}")
    log.debug(f"Target beam is {final_beam!r}")
    log.debug(f"Using scaling factor {sfactor}")
    pix_scale = dy

    gauss_kern = conbeam.as_kernel(pix_scale)
    conbm1 = gauss_kern.array / gauss_kern.array.max()
    fac = sfactor
    if conv_mode == "robust":
        newim, fac = convolve(
            image.astype("f8"), old_beam, final_beam, dx, dy,
        )
        # keep the new sfactor computed by this method
        sfactor = fac
    if conv_mode == "scipy":
        newim = scipy.signal.convolve(image.astype("f8"), conbm1, mode="same")
    elif conv_mode == "astropy":
        newim = convolution.convolve(image.astype("f8"), conbm1, normalize_kernel=False,)
    elif conv_mode == "astropy_fft":
        newim = convolution.convolve_fft(
            image.astype("f8"), conbm1, normalize_kernel=False, allow_huge=True,
        )
    log.debug(f"Using scaling factor {fac}")
    if np.any(np.isnan(newim)):
        log.warning(f"{np.isnan(newim).sum()} NaNs present in smoothed output")

    newim *= fac
    return newim