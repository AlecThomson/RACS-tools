#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import shutil
import subprocess as sp
from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from radio_beam import Beam

from racs_tools import au2, beamcon_2D
from racs_tools.convolve_uv import smooth
from racs_tools.logging import logger

logger.setLevel(logging.DEBUG)


class TestImage(NamedTuple):
    path: Path
    beam: Beam
    data: np.ndarray
    pix_scale: u.Quantity


@pytest.fixture
def make_2d_image(tmpdir) -> TestImage:
    """Make a fake 2D image from with a Gaussian beam.

    Args:
        beam (Beam): Gaussian beam.

    Returns:
        str: Name of the output FITS file.
    """
    pix_scale = 2.5 * u.arcsec

    beam = Beam(20 * u.arcsec, 10 * u.arcsec, 10 * u.deg)

    data = beam.as_kernel(pixscale=pix_scale, x_size=100, y_size=100).array
    data /= data.max()

    hdu = fits.PrimaryHDU(data=data)

    hdu.header = beam.attach_to_header(hdu.header)
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.header["CDELT1"] = -pix_scale.to(u.deg).value
    hdu.header["CDELT2"] = pix_scale.to(u.deg).value
    hdu.header["CRPIX1"] = 50
    hdu.header["CRPIX2"] = 50
    hdu.header["CRVAL1"] = 0
    hdu.header["CRVAL2"] = 0
    hdu.header["CTYPE1"] = "RA---SIN"
    hdu.header["CTYPE2"] = "DEC--SIN"
    hdu.header["CUNIT1"] = "deg"
    hdu.header["CUNIT2"] = "deg"
    hdu.header["EQUINOX"] = 2000.0
    hdu.header["RADESYS"] = "FK5"
    hdu.header["LONPOLE"] = 180.0
    hdu.header["LATPOLE"] = 0.0

    outf = Path(tmpdir) / "2d.fits"
    hdu.writeto(outf, overwrite=True)

    yield TestImage(
        path=outf,
        beam=beam,
        data=data,
        pix_scale=pix_scale,
    )

    outf.unlink()


@pytest.fixture
def mirsmooth(make_2d_image: TestImage) -> TestImage:
    """Smooth a FITS image to a target beam using MIRIAD.

    Args:
        outf (str): FITS image to smooth.
        target_beam (Beam): Target beam.

    Returns:
        Tuple[str, str, str]: Names of the output images.
    """
    target_beam = Beam(40 * u.arcsec, 40 * u.arcsec, 0 * u.deg)
    outim = make_2d_image.path.with_suffix(".im")
    cmd = f"fits op=xyin in={make_2d_image.path.as_posix()} out={outim.as_posix()}"
    sp.run(cmd.split())

    smoothim = make_2d_image.path.with_suffix(".smooth.im")
    cmd = f"convol map={outim} fwhm={target_beam.major.to(u.arcsec).value},{target_beam.minor.to(u.arcsec).value} pa={target_beam.pa.to(u.deg).value} options=final out={smoothim}"
    sp.run(cmd.split())

    smoothfits = outim.with_suffix(".mirsmooth.fits")
    cmd = f"fits op=xyout in={smoothim} out={smoothfits}"
    sp.run(cmd.split())

    yield TestImage(
        path=smoothfits,
        beam=target_beam,
        data=fits.getdata(smoothfits),
        pix_scale=make_2d_image.pix_scale,
    )

    for f in (outim, smoothim, smoothfits):
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()


def check_images(image_1: Path, image_2: Path) -> bool:
    """Compare two FITS images.

    Args:
        fname_1 (str): Image 1.
        fname_2 (str): Image 2.

    Returns:
        bool: True if the images are the same.
    """
    data_1 = fits.getdata(image_1)
    data_2 = fits.getdata(image_2)

    logger.debug(f"{np.nanmean(data_1)=}")
    logger.debug(f"{np.nanmean(data_2)=}")
    logger.info(f"{data_1.shape=}")
    logger.info(f"{data_2.shape=}")

    logger.debug(f"{np.nanmean(data_1-data_2)=}")

    return np.allclose(data_1, data_2, atol=1e-3)


def test_robust(make_2d_image: TestImage, mirsmooth: TestImage):
    """Test the robust convolution."""
    beamcon_2D.smooth_fits_files(
        infile_list=[make_2d_image.path],
        suffix="robust",
        conv_mode="robust",
        bmaj=mirsmooth.beam.major.to(u.arcsec).value,
        bmin=mirsmooth.beam.minor.to(u.arcsec).value,
        bpa=mirsmooth.beam.pa.to(u.deg).value,
    )

    fname_beamcon = make_2d_image.path.with_suffix(".robust.fits")
    assert check_images(mirsmooth.path, fname_beamcon), "Beamcon does not match miriad"


def test_astropy(make_2d_image: TestImage, mirsmooth: TestImage):
    """Test the astropy convolution."""
    beamcon_2D.smooth_fits_files(
        infile_list=[make_2d_image.path],
        suffix="astropy",
        conv_mode="astropy",
        bmaj=mirsmooth.beam.major.to(u.arcsec).value,
        bmin=mirsmooth.beam.minor.to(u.arcsec).value,
        bpa=mirsmooth.beam.pa.to(u.deg).value,
    )

    fname_beamcon = make_2d_image.path.with_suffix(".astropy.fits")
    assert check_images(mirsmooth.path, fname_beamcon), "Beamcon does not match miriad"


def test_scipy(make_2d_image: TestImage, mirsmooth: TestImage):
    """Test the scipy convolution."""
    beamcon_2D.smooth_fits_files(
        infile_list=[make_2d_image.path],
        suffix="scipy",
        conv_mode="scipy",
        bmaj=mirsmooth.beam.major.to(u.arcsec).value,
        bmin=mirsmooth.beam.minor.to(u.arcsec).value,
        bpa=mirsmooth.beam.pa.to(u.deg).value,
    )

    fname_beamcon = make_2d_image.path.with_suffix(".scipy.fits")
    assert check_images(mirsmooth.path, fname_beamcon), "Beamcon does not match miriad"


def test_smooth(make_2d_image: TestImage, mirsmooth: TestImage):
    """Test the smoothing function."""
    logger.debug(f"Testing smooth with {make_2d_image.path}")
    target_beam = mirsmooth.beam
    old_beam = make_2d_image.beam
    dx = make_2d_image.pix_scale
    dy = make_2d_image.pix_scale
    for conv_mode in ("robust", "astropy", "scipy"):
        if conv_mode != "robust":
            conbm = target_beam.deconvolve(old_beam)
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
        else:
            fac = 1
        smooth_data = smooth(
            image=make_2d_image.data,
            old_beam=old_beam,
            new_beam=target_beam,
            dx=make_2d_image.pix_scale,
            dy=make_2d_image.pix_scale,
            conv_mode=conv_mode,
        )
        assert np.allclose(
            smooth_data, mirsmooth.data, atol=1e-5
        ), f"Smooth with {conv_mode} does not match miriad"
