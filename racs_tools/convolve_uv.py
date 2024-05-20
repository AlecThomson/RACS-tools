#!/usr/bin/env python3
""" Fast convolution in the UV domain """
__author__ = "Wasim Raja"


from typing import Optional, Tuple

import astropy.units as units
import numpy as np
import scipy.signal
from astropy import convolution
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError

import racs_tools.gaussft as gaussft
from racs_tools import au2
from racs_tools.logging import logger


def round_up(n: float, decimals: int = 0) -> float:
    """Round to number of decimals

    Args:
        n (float): Number to round.
        decimals (int, optional): Number of decimals. Defaults to 0.

    Returns:
        float: Rounded number.
    """
    multiplier = 10**decimals
    return np.ceil(n * multiplier) / multiplier


def my_ceil(a: float, precision: float = 0.0) -> float:
    """Modified ceil function to round up to precision

    Args:
        a (float): Number to round.
        precision (float, optional): Precision of rounding. Defaults to 0.

    Returns:
        float: Rounded number.
    """
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def get_nyquist_beam(
    target_beam: Beam,
    target_header: fits.Header,
    beams: Beams,
) -> Beam:
    wcs = WCS(target_header)
    pixelscales = proj_plane_pixel_scales(wcs)

    dx = pixelscales[0] * u.deg
    dy = pixelscales[1] * u.deg
    if not np.isclose(dx, dy):
        raise ValueError(f"GRID MUST BE SAME IN X AND Y. Got {dx=} and {dy=}")
    grid = dy
    # Get the minor axis of the convolving beams
    minorcons = []
    for beam in beams:
        try:
            minorcons += [target_beam.deconvolve(beam).minor.to(u.arcsec).value]
        except BeamError as err:
            logger.error(err)
            logger.warning(
                f"Could not deconvolve. New: {target_beam!r}, Old: {beam!r} - will set convolving beam to 0.0"
            )
            minorcons += [
                target_beam.deconvolve(beam, failure_returns_pointlike=True)
                .minor.to(u.arcsec)
                .value
            ]
    minorcons = np.array(minorcons) * u.arcsec
    samps = minorcons / grid.to(u.arcsec)
    # Check that convolving beam will be Nyquist sampled
    if any(samps.value < 2):
        # Set the convolving beam to be Nyquist sampled
        nyq_con_beam = Beam(major=grid * 2, minor=grid * 2, pa=0 * u.deg)
        # Find new target based on common beam * Nyquist beam
        # Not sure if this is best - but it works
        nyq_beam = target_beam.convolve(nyq_con_beam)
        nyq_beam = Beam(
            major=my_ceil(nyq_beam.major.to(u.arcsec).value, precision=1) * u.arcsec,
            minor=my_ceil(nyq_beam.minor.to(u.arcsec).value, precision=1) * u.arcsec,
            pa=round_up(nyq_beam.pa.to(u.deg), decimals=2),
        )
        logger.info(f"Smallest common Nyquist sampled beam is: {nyq_beam!r}")
        if target_beam is not None:
            if target_beam < nyq_beam:
                logger.warning("TARGET BEAM WILL BE UNDERSAMPLED!")
                raise ValueError("CAN'T UNDERSAMPLE BEAM - EXITING")
        else:
            logger.warning("COMMON BEAM WILL BE UNDERSAMPLED!")
            logger.warning("SETTING COMMON BEAM TO NYQUIST BEAM")
            target_beam = nyq_beam

    return target_beam


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

    nanflag = np.isnan(image).any()

    if nanflag:
        # Create a mask for the NaNs
        mask = np.isnan(image).astype(int)
        logger.warning(
            f"Image contains {mask.sum()} ({mask.sum()/ mask.size *100 :0.1f}%) NaNs"
        )
        image = np.nan_to_num(image)

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
    )
    # Perform the x-ing in the FT domain
    im_f = np.fft.fft2(image)

    # Now convolve with the desired Gaussian:
    M = np.multiply(im_f, g_final)
    im_conv = np.fft.ifft2(M)
    im_conv = np.real(im_conv)

    if nanflag:
        # Convert the mask to the FT domain
        mask_f = np.fft.fft2(mask)
        # Multiply the mask by the FT of the Gaussian
        M = np.multiply(mask_f, g_final)
        # Invert the FT of the mask
        mask_conv = np.fft.ifft2(M)
        mask_conv = np.real(mask_conv)
        # Use approx values to find the NaNs
        # Need this to get around numerical issues
        mask_conv = ~(mask_conv + 1 < 2)
        logger.warning(
            f"Convolved image contains {mask_conv.sum()} ({mask_conv.sum()/ mask_conv.size *100 :0.1f}%) NaNs"
        )
        im_conv[mask_conv > 0] = np.nan

    return im_conv, g_ratio


CONVOLUTION_FUNCTIONS = {
    "robust": convolve,
    "scipy": convolution.convolve,
    "astropy": convolution.convolve,
    "astropy_fft": convolution.convolve_fft,
}


def parse_conv_mode(conv_mode: str) -> str:
    logger.info(f"Convolution mode: {conv_mode}")
    if conv_mode not in CONVOLUTION_FUNCTIONS.keys():
        raise ValueError(
            f"Please select valid convolution method! Expected one of {list(CONVOLUTION_FUNCTIONS.keys())}, got {conv_mode}"
        )

    logger.info(f"Using convolution method {conv_mode}")
    if conv_mode == "robust":
        logger.info("This is the most robust method. And fast!")
    elif conv_mode == "scipy":
        logger.info("This fast, but not robust to NaNs or small PSF changes")
    else:
        logger.info("This is slower, but robust to NaNs, but not to small PSF changes")

    return conv_mode


def get_convolving_beam(
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    cutoff: Optional[float] = None,
) -> Tuple[Beam, float]:
    """Get the beam to use for smoothing

    Args:
        old_beam (Beam): Current beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        cutoff (float, optional): Cutoff for beamsize in arcsec. Defaults to None.

    Raises:
        err: If beam deconvolution fails.

    Returns:
        Tuple[Beam, float]: Convolving beam and scaling factor.
    """
    logger.info(f"Current beam is {old_beam!r}")

    if cutoff is not None and old_beam.major.to(u.arcsec) > cutoff * u.arcsec:
        return (
            Beam(
                major=np.nan * u.deg,
                minor=np.nan * u.deg,
                pa=np.nan * u.deg,
            ),
            np.nan,
        )

    if new_beam == old_beam:
        conbm = Beam(
            major=0 * u.deg,
            minor=0 * u.deg,
            pa=0 * u.deg,
        )
        fac = 1.0
        logger.warning(
            f"New beam {new_beam!r} and old beam {old_beam!r} are the same. Won't attempt convolution."
        )
        return conbm, fac
    try:
        conbm = new_beam.deconvolve(old_beam)
    except BeamError as err:
        logger.error(err)
        logger.warning(
            f"Could not deconvolve. New: {new_beam!r}, Old: {old_beam!r} - will set convolving beam to 0.0"
        )
        conbm = new_beam.deconvolve(old_beam, failure_returns_pointlike=True)
    fac, _, _, _, _ = au2.gauss_factor(
        beamConv=[
            conbm.major.to(u.arcsec).value,
            conbm.minor.to(u.arcsec).value,
            conbm.pa.to(u.deg).value,
        ],
        beamOrig=[
            old_beam.major.to(u.arcsec).value,
            old_beam.minor.to(u.arcsec).value,
            old_beam.pa.to(u.deg).value,
        ],
        dx1=dx.to(u.arcsec).value,
        dy1=dy.to(u.arcsec).value,
    )

    return conbm, fac


def smooth(
    image: np.ndarray,
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    conv_mode: str = "robust",
    cutoff: Optional[float] = None,
) -> np.ndarray:
    """Apply smoothing to image

    Args:
        image (np.ndarray): 2D image array.
        old_beam (Beam): Current beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        sfactor (float): Scaling factor.
        conbeam (Beam, optional): Convoling beam to use. Defaults to None.
        conv_mode (str, optional): Convolution mode to use. Defaults to "robust".

    Returns:
        np.ndarray: Smoothed image.

    """

    conbeam, sfactor = get_convolving_beam(
        old_beam=old_beam,
        new_beam=new_beam,
        dx=dx,
        dy=dy,
        cutoff=cutoff,
    )

    if np.isnan(sfactor):
        logger.warning("Beam larger than cutoff -- blanking")
        newim = np.ones_like(image) * np.nan
        return newim
    if conbeam is None:
        conbeam = new_beam.deconvolve(old_beam)
    if np.isnan(conbeam):
        return image * np.nan
    if np.isnan(image).all():
        return image
    if conbeam == Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg) and sfactor == 1:
        return image
    # using Beams package
    logger.debug(f"Old beam is {old_beam!r}")
    logger.debug(f"Using convolving beam {conbeam!r}")
    logger.debug(f"Target beam is {new_beam!r}")
    logger.debug(f"Using scaling factor {sfactor}")
    pix_scale = dy

    gauss_kern = conbeam.as_kernel(pix_scale)
    conbm1 = gauss_kern.array / gauss_kern.array.max()
    fac = sfactor
    if conv_mode == "robust":
        newim, fac = convolve(
            image.astype("f8"),
            old_beam,
            new_beam,
            dx,
            dy,
        )
        # keep the new sfactor computed by this method
        sfactor = fac
    if conv_mode == "scipy":
        newim = scipy.signal.convolve(image.astype("f8"), conbm1, mode="same")
    elif conv_mode == "astropy":
        newim = convolution.convolve(
            image.astype("f8"),
            conbm1,
            normalize_kernel=False,
        )
    elif conv_mode == "astropy_fft":
        newim = convolution.convolve_fft(
            image.astype("f8"),
            conbm1,
            normalize_kernel=False,
            allow_huge=True,
        )
    logger.debug(f"Using scaling factor {fac}")
    if np.any(np.isnan(newim)):
        logger.warning(f"{np.isnan(newim).sum()} NaNs present in smoothed output")

    newim *= fac

    # Ensure the output data-type is the same as the input
    return newim.astype(image.dtype)
