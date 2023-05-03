#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess as sp
import unittest

import astropy.units as u
import numpy as np
import schwimmbad
from astropy.io import fits
from astropy.table import Table
from radio_beam import Beam, Beams
from test_2d import check_images, cleanup, mirsmooth

from racs_tools import beamcon_3D


def smoothcube(outf: str, target_beam: Beam) -> str:
    """Smooth a FITS cube to a target beam.

    Args:
        outf (str): FITS cube to smooth.
        target_beam (Beam): Target beam.

    Returns:
        str: Output FITS cube.
    """
    cube = np.squeeze(fits.getdata(outf))
    header = fits.getheader(outf)
    with fits.open(outf) as hdulist:
        beams = Beams.from_fits_bintable(hdulist[1])

    smoothcube = []

    for image, beam in zip(cube, beams):
        hdu = fits.PrimaryHDU(data=image, header=header)
        hdu.header = beam.attach_to_header(hdu.header)
        hdu.writeto("tmp.fits", overwrite=True)
        outim, smoothim, smoothfits = mirsmooth("tmp.fits", target_beam)
        smoothcube.append(fits.getdata(smoothfits))
        os.remove("tmp.fits")
        os.remove(smoothfits)
        shutil.rmtree(outim)
        shutil.rmtree(smoothim)

    smoothcube = np.array(smoothcube)
    cube_hdu = fits.PrimaryHDU(data=smoothcube, header=header)
    cube_hdu.header = target_beam.attach_to_header(cube_hdu.header)

    smooth_outf = "smoothcube.fits"
    cube_hdu.writeto(smooth_outf, overwrite=True)

    return smooth_outf


def make_3d_image(beams: Beams) -> str:
    """Make a fake 3D image from with a Gaussian beam.

    Args:
        beams (Beams): Gaussian beams.

    Returns:
        str: FITS cube filename.
    """
    pix_scale = 2.5 * u.arcsec

    freqs = np.linspace(1, 2, len(beams)) * u.GHz

    cube = []
    for beam in beams:
        data = beam.as_kernel(pixscale=pix_scale, x_size=100, y_size=100).array
        data /= data.max()
        cube.append(data)

    cube = np.array(cube)[np.newaxis]

    hdu = fits.PrimaryHDU(data=cube)
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.header["CDELT1"] = -pix_scale.to(u.deg).value
    hdu.header["CDELT2"] = pix_scale.to(u.deg).value
    hdu.header["CDELT3"] = (freqs[1] - freqs[0]).to(u.Hz).value
    hdu.header["CDELT4"] = 1
    hdu.header["CRPIX1"] = 50
    hdu.header["CRPIX2"] = 50
    hdu.header["CRPIX3"] = 1
    hdu.header["CRPIX4"] = 1
    hdu.header["CRVAL1"] = 0
    hdu.header["CRVAL2"] = 0
    hdu.header["CRVAL3"] = freqs[0].to(u.Hz).value
    hdu.header["CRVAL4"] = 1
    hdu.header["CTYPE1"] = "RA---SIN"
    hdu.header["CTYPE2"] = "DEC--SIN"
    hdu.header["CTYPE3"] = "FREQ"
    hdu.header["CTYPE4"] = "STOKES"
    hdu.header["CUNIT1"] = "deg"
    hdu.header["CUNIT2"] = "deg"
    hdu.header["CUNIT3"] = "Hz"
    hdu.header["CUNIT4"] = ""
    hdu.header["EQUINOX"] = 2000.0
    hdu.header["RADESYS"] = "FK5"
    hdu.header["LONPOLE"] = 180.0
    hdu.header["LATPOLE"] = 0.0
    hdu.header["CASAMBM"] = True
    tiny = np.finfo(np.float32).tiny
    chans = np.arange(len(freqs))
    pols = np.zeros_like(chans)  # Zeros because we take the first one
    beam_table = Table(
        data=[
            # Replace NaNs with np.finfo(np.float32).tiny - this is the smallest
            # positive number that can be represented in float32
            # We use this to keep CASA happy
            np.nan_to_num(beams.major.to(u.arcsec), nan=tiny * u.arcsec),
            np.nan_to_num(beams.minor.to(u.arcsec), nan=tiny * u.arcsec),
            np.nan_to_num(beams.pa.to(u.deg), nan=tiny * u.deg),
            chans,
            pols,
        ],
        names=["BMAJ", "BMIN", "BPA", "CHAN", "POL"],
        dtype=["f4", "f4", "f4", "i4", "i4"],
    )
    tab_hdu = fits.table_to_hdu(beam_table)
    tab_header = tab_hdu.header
    tab_header["EXTNAME"] = "BEAMS"
    tab_header["NCHAN"] = len(freqs)
    tab_header["NPOL"] = 1  # Only one pol for now
    new_hdulist = fits.HDUList([hdu, tab_hdu])

    outf = "3d.fits"
    new_hdulist.writeto(outf, overwrite=True)

    return outf


class test_Beamcon3D(unittest.TestCase):
    """Test the beamcon_3D script."""

    def setUp(self) -> None:
        """Set up the test."""
        self.orginal_beams = Beams(
            major=np.linspace(50, 10, 10) * u.arcsec,
            minor=np.linspace(10, 10, 10) * u.arcsec,
            pa=np.random.uniform(0, 180, 10) * u.deg,
        )
        test_image = make_3d_image(self.orginal_beams)

        self.test_image = test_image
        self.target_beam = Beam(60 * u.arcsec, 60 * u.arcsec, 0 * u.deg)
        smoothfits = smoothcube(test_image, self.target_beam)
        self.test_cube = smoothfits

        self.files = [
            test_image,
            smoothfits,
        ]

    def test_robust(self):
        """Test the robust mode."""
        beamcon_3D.main(
            infile=[self.test_image],
            suffix="robust",
            conv_mode="robust",
            mode="total",
            bmaj=self.target_beam.major.to(u.arcsec).value,
            bmin=self.target_beam.minor.to(u.arcsec).value,
            bpa=self.target_beam.pa.to(u.deg).value,
        )

        fname_beamcon = self.test_image.replace(".fits", ".robust.fits")
        self.files.append(fname_beamcon)
        check_images(self.test_cube, fname_beamcon), "Beamcon does not match Miriad"

    def test_astropy(self):
        """Test the astropy mode."""
        beamcon_3D.main(
            infile=[self.test_image],
            suffix="astropy",
            conv_mode="astropy",
            mode="total",
            bmaj=self.target_beam.major.to(u.arcsec).value,
            bmin=self.target_beam.minor.to(u.arcsec).value,
            bpa=self.target_beam.pa.to(u.deg).value,
        )

        fname_beamcon = self.test_image.replace(".fits", ".astropy.fits")
        self.files.append(fname_beamcon)
        check_images(self.test_cube, fname_beamcon), "Beamcon does not match Miriad"

    def test_scipy(self):
        """Test the scipy mode."""
        beamcon_3D.main(
            infile=[self.test_image],
            suffix="scipy",
            conv_mode="scipy",
            mode="total",
            bmaj=self.target_beam.major.to(u.arcsec).value,
            bmin=self.target_beam.minor.to(u.arcsec).value,
            bpa=self.target_beam.pa.to(u.deg).value,
        )
        fname_beamcon = self.test_image.replace(".fits", ".scipy.fits")
        self.files.append(fname_beamcon)
        check_images(self.test_cube, fname_beamcon), "Beamcon does not match Miriad"

    def tearDown(self) -> None:
        """Tear down the test."""
        cleanup(self.files)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    suite = unittest.TestLoader().loadTestsFromTestCase(test_Beamcon3D)
    unittest.TextTestRunner(verbosity=1).run(suite)
