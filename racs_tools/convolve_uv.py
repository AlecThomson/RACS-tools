#!/usr/bin/env python3
"""Fast convolution in the UV domain"""

__author__ = "Wasim Raja"

import gc
from typing import Literal, NamedTuple

import numpy as np
import scipy.fft
import scipy.signal
from astropy import convolution, units
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError

from racs_tools import au2, gaussft
from racs_tools.logging import logger

ZERO_BEAM = Beam(0 * u.deg)
NAN_BEAM = Beam(np.nan * u.deg)


class ConvolutionResult(NamedTuple):
    """Result of convolution"""

    image: np.ndarray
    """Convolved image"""
    scaling_factor: float
    """Scaling factor for the image in Jy/beam"""


def round_up(n: float, decimals: int = 0) -> float:
    """Round to number of decimals

    Args:
        n (float): Number to round.
        decimals (int, optional): Number of decimals. Defaults to 0.

    Returns:
        float: Rounded number.
    """
    multiplier = 10**decimals
    # Ignore round-off below 1e-9 of the last place
    eps = 1e-9 * getattr(n, "unit", 1)
    return np.ceil(n * multiplier - eps) / multiplier


def my_ceil(a: float, precision: float = 0.0) -> float:
    """Modified ceil function to round up to precision

    Args:
        a (float): Number to round.
        precision (float, optional): Precision of rounding. Defaults to 0.

    Returns:
        float: Rounded number.
    """
    # Ignore round-off below 1e-9 of the last place (30.500000000000004 -> 30.5)
    multiplier = 10**precision
    return np.ceil(a * multiplier - 1e-9) / multiplier


def beam_contains(big_beam: Beam, small_beam: Beam) -> bool:
    """Check if small_beam can be convolved up to big_beam

    Unlike Beam.deconvolve, this allows a kernel that is a line or a point.

    Args:
        big_beam (Beam): Target beam.
        small_beam (Beam): Input beam.

    Returns:
        bool: True if small_beam fits inside big_beam.
    """

    def cov(beam: Beam) -> np.ndarray:
        major = beam.major.to(u.arcsec).value
        minor = beam.minor.to(u.arcsec).value
        pa = beam.pa.to(u.rad).value
        rot = np.array([[np.cos(pa), -np.sin(pa)], [np.sin(pa), np.cos(pa)]])
        return rot @ np.diag([major**2, minor**2]) @ rot.T

    big = cov(big_beam)
    residual = big - cov(small_beam)
    if not np.isfinite(residual).all():
        return False
    # Round-off is relative to the beam size. Beams are often read from
    # float32 columns (CASA beam tables, cube headers written from them), so
    # allow a few float32 epsilons: a kernel that small is no kernel at all.
    tol = 4 * np.finfo(np.float32).eps * np.trace(big)
    return bool(np.linalg.eigvalsh(residual)[0] >= -tol)


def round_common_beam(common_beam: Beam, beams: Beams | list[Beam]) -> Beam:
    """Round a computed common beam up to 0.1 arcsec and 0.01 deg

    If rounding the PA leaves an input outside, step the axes up by 0.1 arcsec.

    Args:
        common_beam (Beam): Common beam.
        beams (Beams | list[Beam]): Beams it must contain.

    Returns:
        Beam: Rounded beam.
    """
    for step in (0.0, 0.1):
        rounded = Beam(
            major=my_ceil(common_beam.major.to(u.arcsec).value + step, precision=1)
            * u.arcsec,
            minor=my_ceil(common_beam.minor.to(u.arcsec).value + step, precision=1)
            * u.arcsec,
            pa=round_up(common_beam.pa.to(u.deg), decimals=2),
        )
        if all(beam_contains(rounded, beam) for beam in beams):
            break
    return rounded


def check_square_pixels(dx: u.Quantity, dy: u.Quantity) -> None:
    """Check that pixels are square, as the image-plane kernel modes assume

    Args:
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.

    Raises:
        ValueError: If |dx| != |dy|.
    """
    if not np.isclose(abs(dx), abs(dy)):
        raise ValueError(
            f"Pixels must be square for this convolution mode, got {dx=} and {dy=}. "
            "Use conv_mode='robust'."
        )


def get_nyquist_beam(
    target_beam: Beam,
    target_header: fits.Header,
    beams: Beams,
) -> Beam:
    """Get the Nyquist sampled beam

    Args:
        target_beam (Beam): Target beam.
        target_header (fits.Header): Target header.
        beams (Beams): All the beams to be convolved.

    Raises:
        ValueError: If grid is not same in x and y.
        ValueError: If beam is undersampled.

    Returns:
        Beam: Nyquist sampled beam.
    """
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


def kernel_reach(old_beam: Beam, new_beam: Beam, dx: u.Quantity, dy: u.Quantity) -> int:
    """The reach in pixels of the kernel from ``old_beam`` to ``new_beam``: 6.1
    standard deviations along its major axis, plus one. Beyond it the kernel
    holds under 1e-8 of its sum."""
    fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))

    def covariance(beam: Beam) -> np.ndarray:
        """In pixels, along (rows, columns), PA from the row axis."""
        major = beam.major.to(units.rad).value * fwhm_to_sigma
        minor = beam.minor.to(units.rad).value * fwhm_to_sigma
        t = beam.pa.to(units.rad).value
        c, s = np.cos(t), np.sin(t)
        rotation = np.array([[c, -s], [s, c]])
        return rotation @ np.diag([major**2, minor**2]) @ rotation.T

    steps = np.array([abs(dy.to(units.rad).value), abs(dx.to(units.rad).value)])
    kernel = (covariance(new_beam) - covariance(old_beam)) / np.outer(steps, steps)
    return int(np.ceil(6.1 * np.sqrt(max(np.linalg.eigvalsh(kernel).max(), 0)))) + 1


def convolve(
    image: np.ndarray,
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    cutoff: float | None = None,
    nan_fraction: float = 0.5,
) -> ConvolutionResult:
    """Convolve by X-ing in the Fourier domain.
        - convolution with Gaussian kernels only
        - no need for generation of a kernel image
        - direct computation of the FT of the kernel

    The sky beyond the image's edges and under its NaN pixels is missing data.
    The image is padded by the kernel's reach, so the convolution is linear
    rather than circular, and each pixel is divided by the share of its
    kernel's weight that fell on data. A pixel with more than ``nan_fraction``
    of that weight on missing data is blanked.

    Args:
        image (np.ndarray): The image to be convolved.
        old_beam (Beam): Current image PSF.
        new_beam (Beam): Target image PSF.
        dx (u.Quantity): Grid size in x (e.g. CDELT1)
        dy (u.Quantity): Grid size in y (e.g. CDELT2)
        cutoff (Optional[float], optional): Dummy cutoff for beamszie in arcsec. Defaults to None. NOT USED.
        nan_fraction (float, optional): Blank pixels with more than this share of their kernel's weight on missing data. Defaults to 0.5.

    Returns:
        ConvolutionResult: convolved image, scaling factor
    """

    ### These values aren't used in convolution, but are needed for santiy checks
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
        return ConvolutionResult(newim, sfactor)
    if conbeam is None:
        conbeam = new_beam.deconvolve(old_beam)
    if np.isnan(conbeam):
        return ConvolutionResult(image * np.nan, sfactor)
    if np.isnan(image).all():
        # Same factor gaussft gives a finite image
        g_ratio = np.sqrt(
            (new_beam.major * new_beam.minor / (old_beam.major * old_beam.minor))
            .decompose()
            .value
        )
        return ConvolutionResult(image, g_ratio)
    if conbeam == ZERO_BEAM and sfactor == 1:
        return ConvolutionResult(image, sfactor)
    ###

    # Now we do the convolution
    nan_mask = np.isnan(image)
    if nan_mask.any():
        logger.warning(
            f"Image contains {nan_mask.sum()} "
            f"({nan_mask.sum() / nan_mask.size * 100:0.1f}%) NaNs"
        )

    nx = image.shape[0]
    ny = image.shape[1]

    # The image is real, so half of its spectrum is redundant. Transforming in
    # the image's own precision and keeping only that half is what makes this
    # affordable on a wide image: a single-precision plane never gets promoted
    # to complex128, and the last axis is stored once rather than twice.
    #
    # Matched on kind and width rather than identity, because FITS data arrives
    # big-endian (">f4") and `np.dtype(">f4") == np.float32` is False, which
    # would send every real image down the double-precision path.
    real_dtype = (
        np.float32
        if image.dtype.kind == "f" and image.dtype.itemsize <= 4
        else np.float64
    )

    beams = {
        "bmin_in": old_beam.minor.to(units.deg).value,
        "bmaj_in": old_beam.major.to(units.deg).value,
        "bpa_in": old_beam.pa.to(units.deg).value,
        "bmin": new_beam.minor.to(units.deg).value,
        "bmaj": new_beam.major.to(units.deg).value,
        "bpa": new_beam.pa.to(units.deg).value,
    }
    # Padding by the kernel's reach makes the convolution linear: without it
    # flux convolved off one edge would come back on the opposite one.
    reach = kernel_reach(old_beam, new_beam, dx, dy)
    px = scipy.fft.next_fast_len(nx + reach, real=True)
    py = scipy.fft.next_fast_len(ny + reach, real=True)

    # The coordinates in FT domain. Only the non-redundant v are needed, to
    # match the output of the real-input transform below.
    # Rows run along y and columns along x. A negative CDELT would mirror the PA.
    u_image = np.fft.fftfreq(px, d=abs(dy.to(units.rad).value)).astype(real_dtype)
    v_image = np.fft.rfftfreq(py, d=abs(dx.to(units.rad).value)).astype(real_dtype)
    [g_final, g_ratio] = gaussft.gaussft(u=u_image, v=v_image, **beams)

    # On an even axis the Nyquist bin holds +1/2 and -1/2 cycles per pixel at
    # once, and the taper differs between them when the kernel is rotated.
    # Convolving the image's band-limited interpolant exactly averages the
    # two. The taper is a quadratic form, so taper(-u, v) = taper(u, -v):
    # averaging over the sign of u alone is exact, corner included.
    if px % 2 == 0:
        row = px // 2
        flipped, _ = gaussft.gaussft(u=-u_image[row : row + 1], v=v_image, **beams)
        g_final[row] = (g_final[row] + flipped[0]) / 2
    if py % 2 == 0:
        rows = np.arange(px) != px // 2 if px % 2 == 0 else slice(None)
        flipped, _ = gaussft.gaussft(u=u_image, v=-v_image[-1:], **beams)
        g_final[rows, -1] = (g_final[rows, -1] + flipped[rows, 0]) / 2
    del u_image
    del v_image

    def convolve_padded(
        plane: np.ndarray, fill: float, taper: np.ndarray
    ) -> np.ndarray:
        """``plane`` in the corner of a (px, py) grid of ``fill``, through the taper, cropped back."""
        padded = np.full((px, py), fill, dtype=real_dtype)
        padded[:nx, :ny] = plane
        spectrum = scipy.fft.rfft2(padded, overwrite_x=True)
        del padded
        spectrum *= taper
        return scipy.fft.irfft2(spectrum, s=(px, py), overwrite_x=True)[:nx, :ny]

    # Missing pixels are zero-filled, and a mask of them, the padding
    # included, is convolved the same way to give the share of each pixel's
    # kernel that fell on missing data.
    im_conv = convolve_padded(np.where(nan_mask, 0, image), 0, g_final).astype(
        np.float32, copy=False
    )
    del image
    missing = convolve_padded(nan_mask, 1, g_final) / g_ratio
    del nan_mask, g_final

    # Divided by the share on data, so a pixel near missing data keeps its
    # flux scale. Shares under 1e-8 are the kernel's truncation; a coarsely
    # sampled kernel rings, so near an edge the share can dip below zero.
    blank = missing > max(nan_fraction, 1e-8)
    near = (np.abs(missing) > 1e-8) & ~blank
    im_conv[near] /= 1 - missing[near]
    im_conv[blank] = np.nan
    logger.debug(
        f"Convolved image contains {blank.sum()} ({blank.sum() / blank.size * 100:0.1f}%) NaNs"
    )

    return ConvolutionResult(image=im_conv, scaling_factor=g_ratio)


def convolve_scipy(
    image: np.ndarray,
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    cutoff: float | None = None,
) -> ConvolutionResult:
    """Convolve using scipy's convolution

    Args:
        image (np.ndarray): Image to be convolved.
        old_beam (Beam): Current beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        cutoff (Optional[float], optional): Cutoff for beamszie in arcsec. Defaults to None.

    Returns:
        ConvolutionResult: Convolved image, scaling factor.
    """
    check_square_pixels(dx, dy)
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
        return ConvolutionResult(newim, sfactor)
    if conbeam is None:
        conbeam = new_beam.deconvolve(old_beam)
    if np.isnan(conbeam):
        return ConvolutionResult(image * np.nan, sfactor)
    if np.isnan(image).all():
        return ConvolutionResult(image, sfactor)
    if conbeam == ZERO_BEAM and sfactor == 1:
        return ConvolutionResult(image, sfactor)

    gauss_kern = conbeam.as_kernel(dy)
    conbm1 = gauss_kern.array / gauss_kern.array.max()
    newim = scipy.signal.convolve(image.astype("f8"), conbm1, mode="same")

    return ConvolutionResult(newim, sfactor)


def convolve_astropy(
    image: np.ndarray,
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    cutoff: float | None = None,
) -> ConvolutionResult:
    """Convolve using astropy's convolution

    Args:
        image (np.ndarray): Image to be convolved.
        old_beam (Beam): Current beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        cutoff (Optional[float], optional): Cutoff for beamszie in arcsec. Defaults to None.

    Returns:
        ConvolutionResult: Convolved image, scaling factor.
    """
    check_square_pixels(dx, dy)
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
        return ConvolutionResult(newim, sfactor)
    if conbeam is None:
        conbeam = new_beam.deconvolve(old_beam)
    if np.isnan(conbeam):
        return ConvolutionResult(image * np.nan, sfactor)
    if np.isnan(image).all():
        return ConvolutionResult(image, sfactor)
    if conbeam == ZERO_BEAM and sfactor == 1:
        return ConvolutionResult(image, sfactor)
    gauss_kern = conbeam.as_kernel(dy)
    conbm1 = gauss_kern.array / gauss_kern.array.max()
    newim = convolution.convolve(
        image.astype("f8"),
        conbm1,
        normalize_kernel=False,
    )

    return ConvolutionResult(newim, sfactor)


def convolve_astropy_fft(
    image: np.ndarray,
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    cutoff: float | None = None,
) -> ConvolutionResult:
    """Convolve using astropy's FFT convolution

    Args:
        image (np.ndarray): Image to be convolved.
        old_beam (Beam): Current beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        cutoff (Optional[float], optional): Cutoff for beamszie in arcsec. Defaults to None.

    Returns:
        ConvolutionResult: Convolved image, scaling factor.
    """
    check_square_pixels(dx, dy)
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
        return ConvolutionResult(newim, sfactor)
    if conbeam is None:
        conbeam = new_beam.deconvolve(old_beam)
    if np.isnan(conbeam):
        return ConvolutionResult(image * np.nan, sfactor)
    if np.isnan(image).all():
        return ConvolutionResult(image, sfactor)
    if conbeam == ZERO_BEAM and sfactor == 1:
        return ConvolutionResult(image, sfactor)
    gauss_kern = conbeam.as_kernel(dy)
    conbm1 = gauss_kern.array / gauss_kern.array.max()
    newim = convolution.convolve_fft(
        image.astype("f8"),
        conbm1,
        normalize_kernel=False,
        allow_huge=True,
    )

    return ConvolutionResult(newim, sfactor)


CONVOLUTION_FUNCTIONS = {
    "robust": convolve,
    "scipy": convolve_scipy,
    "astropy": convolve_astropy,
    "astropy_fft": convolve_astropy_fft,
}


def parse_conv_mode(
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"],
) -> str:
    """Parse convolution mode

    Args:
        conv_mode (str): Convolution mode.

    Returns:
        str: Convolution mode.
    """
    logger.info(f"Convolution mode: {conv_mode}")
    if conv_mode not in CONVOLUTION_FUNCTIONS:
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
    cutoff: float | None = None,
) -> tuple[Beam, float]:
    """Get the beam to use for smoothing

    Args:
        old_beam (Beam): Current beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        cutoff (float, optional): Cutoff for beamsize in arcsec. Defaults to None.

    Raises:
        ValueError: If the input beam has zero size or is larger than the target.

    Returns:
        Tuple[Beam, float]: Convolving beam and scaling factor.
    """
    logger.info(f"Current beam is {old_beam!r}")

    if cutoff is not None and old_beam.major.to(u.arcsec) > cutoff * u.arcsec:
        return (
            NAN_BEAM,
            np.nan,
        )

    for beam in (old_beam, new_beam):
        if not np.isfinite([beam.major.value, beam.minor.value, beam.pa.value]).all():
            logger.warning(f"Beam {beam!r} is not finite -- blanking")
            return NAN_BEAM, np.nan

    if not (old_beam.major > 0 and old_beam.minor > 0):
        raise ValueError(f"Input beam {old_beam!r} has zero size")

    if new_beam == old_beam:
        conbm = ZERO_BEAM
        fac = 1.0
        logger.warning(
            f"New beam {new_beam!r} and old beam {old_beam!r} are the same. Won't attempt convolution."
        )
        return conbm, fac
    if not beam_contains(new_beam, old_beam):
        raise ValueError(
            f"Target beam {new_beam!r} cannot be reached by convolving {old_beam!r}"
        )
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
    logger.debug(f"Old beam is {old_beam!r}")
    logger.debug(f"Using convolving beam {conbm!r}")
    logger.debug(f"Target beam is {new_beam!r}")
    logger.debug(f"Using scaling factor {fac}")
    return conbm, fac


def smooth(
    image: np.ndarray,
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"] = "robust",
    cutoff: float | None = None,
) -> np.ndarray:
    """Apply smoothing to image in Jy/beam

    Args:
        image (np.ndarray): 2D image array.
        old_beam (Beam): Current beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        conv_mode (str, optional): Convolution mode to use. Defaults to "robust".
        cutoff (Optional[float], optional): Cutoff for beamsize in arcsec. Defaults to None.

    Returns:
        np.ndarray: Smoothed image.

    """
    out_dtype = image.dtype
    conv_func = CONVOLUTION_FUNCTIONS.get(conv_mode, convolve)
    new_image, fac = conv_func(
        image=image,
        old_beam=old_beam,
        new_beam=new_beam,
        dx=dx,
        dy=dy,
        cutoff=cutoff,
    )
    del image
    gc.collect()
    logger.debug(f"Using scaling factor {fac}")
    if np.any(np.isnan(new_image)):
        logger.warning(f"{np.isnan(new_image).sum()} NaNs present in smoothed output")

    new_image = fac * new_image

    # Ensure the output data-type is the same as the input
    return new_image.astype(out_dtype)
