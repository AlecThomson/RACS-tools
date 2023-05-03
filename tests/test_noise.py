#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess as sp
import unittest

import astropy.units as u
import numpy as np
from astropy.io import fits
from test_2d import cleanup

from racs_tools import getnoise_list


def make_noise_cube(noise_per_chan: np.ndarray) -> np.ndarray:
    """Make a noise cube."""
    noise_cube = np.zeros((len(noise_per_chan), 100, 100))
    for i, noise in enumerate(noise_per_chan):
        noise_cube[i] = np.random.normal(loc=0, scale=noise, size=(100, 100))

    return noise_cube


def write_cube(cube_data: np.ndarray, outfile: str) -> None:
    """Write a cube to a FITS file."""
    hdu = fits.PrimaryHDU(data=cube_data)
    pix_scale = 2.5 * u.arcsec
    freqs = np.linspace(1e9, 2e9, len(cube_data)) * u.Hz
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
    hdu.writeto(outfile, overwrite=True)


class test_Noise(unittest.TestCase):
    """Test the 2D beam convolution."""

    def setUp(self) -> None:
        nchan = 100
        qnoise = np.linspace(1, 11, nchan)
        # Add spikes to the noise
        qnoise[10] = 1e3
        self.qnoise = qnoise
        unoise = np.linspace(2, 12, nchan)
        unoise[20] = 1e3
        self.unoise = unoise
        self.bad_idx = [10, 20]

        qcube = make_noise_cube(qnoise)
        ucube = make_noise_cube(unoise)

        qfile = "qcube.fits"
        self.qfile = qfile
        ufile = "ucube.fits"
        self.ufile = ufile

        write_cube(qcube, qfile)
        write_cube(ucube, ufile)

        self.files = [qfile, ufile]

    def test_main(self):
        getnoise_list.main(
            qfile=self.qfile,
            ufile=self.ufile,
            blank=False,
            save_noise=True,
        )
        qnoise_file = self.qfile.replace(".fits", ".noise.txt")
        unoise_file = self.ufile.replace(".fits", ".noise.txt")
        qnoise = np.loadtxt(qnoise_file)
        unoise = np.loadtxt(unoise_file)
        assert np.allclose(
            qnoise, self.qnoise, rtol=1e-1
        ), "Measured Stokes Q noise values does not match input values."
        assert np.allclose(
            unoise, self.unoise, rtol=1e-1
        ), "Measured Stokes U noise values does not match input values."

        self.files.append(qnoise_file)
        self.files.append(unoise_file)

    def test_blank(self):
        getnoise_list.main(
            qfile=self.qfile,
            ufile=self.ufile,
            blank=True,
            outfile="noise.txt",
            save_noise=False,
        )
        self.files.append("noise.txt")
        bad_chans = np.loadtxt("noise.txt").astype(bool)
        assert np.all(bad_chans[self.bad_idx]), "Blanking failed."

        # Check for NaNs in the cubes
        blank_q = self.qfile.replace(".fits", ".blanked.fits")
        self.files.append(blank_q)
        blank_u = self.ufile.replace(".fits", ".blanked.fits")
        self.files.append(blank_u)
        qcube = fits.getdata(blank_q)
        ucube = fits.getdata(blank_u)
        # We expect NaNs in the bad channels
        assert np.all(np.isnan(qcube[self.bad_idx])), "NaNs not found in Stokes Q cube."
        assert np.all(np.isnan(ucube[self.bad_idx])), "NaNs not found in Stokes U cube."

    def tearDown(self) -> None:
        cleanup(self.files)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    suite = unittest.TestLoader().loadTestsFromTestCase(test_Noise)
    unittest.TextTestRunner(verbosity=1).run(suite)
