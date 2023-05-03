#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess as sp
import unittest
from typing import List, Tuple, Union

import astropy.units as u
import numpy as np
import schwimmbad
from astropy.io import fits
from radio_beam import Beam

from racs_tools import beamcon_2D


def make_2d_image(beam: Beam) -> str:
    """Make a fake 2D image from with a Gaussian beam.

    Args:
        beam (Beam): Gaussian beam.

    Returns:
        str: Name of the output FITS file.
    """
    pix_scale = 2.5 * u.arcsec

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

    outf = "2d.fits"
    hdu.writeto(outf, overwrite=True)

    return outf


def mirsmooth(outf: str, target_beam: Beam) -> Tuple[str, str, str]:
    """Smooth a FITS image to a target beam using MIRIAD.

    Args:
        outf (str): FITS image to smooth.
        target_beam (Beam): Target beam.

    Returns:
        Tuple[str, str, str]: Names of the output images.
    """
    outim = outf.replace(".fits", ".im")
    cmd = f"fits op=xyin in={outf} out={outim}"
    sp.run(cmd.split())

    smoothim = outim.replace(".im", ".smooth.im")
    cmd = f"convol map={outim} fwhm={target_beam.major.to(u.arcsec).value},{target_beam.minor.to(u.arcsec).value} pa={target_beam.pa.to(u.deg).value} options=final out={smoothim}"
    sp.run(cmd.split())

    smoothfits = outim.replace(".im", ".mirsmooth.fits")
    cmd = f"fits op=xyout in={outim.replace('.im', '.smooth.im')} out={smoothfits}"
    sp.run(cmd.split())

    return outim, smoothim, smoothfits


def check_images(fname_1: str, fname_2: str) -> bool:
    """Compare two FITS images.

    Args:
        fname_1 (str): Image 1.
        fname_2 (str): Image 2.

    Returns:
        bool: True if the images are the same.
    """
    data_1 = fits.getdata(fname_1)
    data_2 = fits.getdata(fname_2)

    return np.allclose(data_1, data_2, atol=1e-5)


def cleanup(files: List[str]):
    """Remove files.

    Args:
        files (List[str]): List of files to remove.
    """
    for f in files:
        sp.run(f"rm -rfv {f}".split())


class test_Beamcon2D(unittest.TestCase):
    """Test the 2D beam convolution."""

    def setUp(self) -> None:
        """Set up the test."""
        self.orginal_beam = Beam(20 * u.arcsec, 10 * u.arcsec, 10 * u.deg)
        test_image = make_2d_image(self.orginal_beam)

        self.test_image = test_image
        self.target_beam = Beam(40 * u.arcsec, 40 * u.arcsec, 0 * u.deg)
        mirfile, mirfile_smooth, fname_mir = mirsmooth(test_image, self.target_beam)
        self.test_mir = fname_mir

        self.files = [
            test_image,
            mirfile,
            mirfile_smooth,
            fname_mir,
        ]

    def test_robust(self):
        """Test the robust convolution."""
        with schwimmbad.SerialPool() as pool:
            beamcon_2D.main(
                pool=pool,
                infile=[self.test_image],
                suffix="robust",
                conv_mode="robust",
                bmaj=self.target_beam.major.to(u.arcsec).value,
                bmin=self.target_beam.minor.to(u.arcsec).value,
                bpa=self.target_beam.pa.to(u.deg).value,
            )

        fname_beamcon = self.test_image.replace(".fits", ".robust.fits")
        self.files.append(fname_beamcon)
        assert check_images(
            self.test_mir, fname_beamcon
        ), "Beamcon does not match miriad"

    def test_astropy(self):
        """Test the astropy convolution."""
        print(f"{self.test_image=}")
        with schwimmbad.SerialPool() as pool:
            beamcon_2D.main(
                pool=pool,
                infile=[self.test_image],
                suffix="astropy",
                conv_mode="astropy",
                bmaj=self.target_beam.major.to(u.arcsec).value,
                bmin=self.target_beam.minor.to(u.arcsec).value,
                bpa=self.target_beam.pa.to(u.deg).value,
            )

        fname_beamcon = self.test_image.replace(".fits", ".astropy.fits")
        self.files.append(fname_beamcon)
        assert check_images(
            self.test_mir, fname_beamcon
        ), "Beamcon does not match miriad"

    def test_scipy(self):
        """Test the scipy convolution."""
        with schwimmbad.SerialPool() as pool:
            beamcon_2D.main(
                pool=pool,
                infile=[self.test_image],
                suffix="scipy",
                conv_mode="scipy",
                bmaj=self.target_beam.major.to(u.arcsec).value,
                bmin=self.target_beam.minor.to(u.arcsec).value,
                bpa=self.target_beam.pa.to(u.deg).value,
            )

        fname_beamcon = self.test_image.replace(".fits", ".scipy.fits")
        self.files.append(fname_beamcon)
        assert check_images(
            self.test_mir, fname_beamcon
        ), "Beamcon does not match miriad"

    def tearDown(self) -> None:
        """Clean up."""
        cleanup(self.files)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    suite = unittest.TestLoader().loadTestsFromTestCase(test_Beamcon2D)
    unittest.TextTestRunner(verbosity=1).run(suite)
