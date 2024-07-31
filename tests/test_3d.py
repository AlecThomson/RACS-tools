#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import subprocess as sp
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from radio_beam import Beam, Beams

from racs_tools import beamcon_3D

from .test_2d import TestImage, check_images


@pytest.fixture
def make_3d_image(
    tmpdir: str,
    beams: Beams = Beams(
        major=[
            50,
        ]
        * 100
        * u.arcsec,
        minor=[
            10,
        ]
        * 100
        * u.arcsec,
        pa=[
            0,
        ]
        * 100
        * u.deg,
    ),
) -> TestImage:
    """Make a fake 3D image from with a Gaussian beam.

    Args:
        beams (Beams): Gaussian beams.

    Returns:
        str: FITS cube filename.
    """
    pix_scale = 2.5 * u.arcsec

    freqs = np.linspace(1, 2, len(beams)) * u.GHz

    cube = []
    peaks = np.random.uniform(0.1, 10, len(beams))
    for beam, peak in zip(beams, peaks):
        data = beam.as_kernel(pixscale=pix_scale, x_size=100, y_size=100).array
        data /= data.max()
        data *= peak
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
    beam = beams[0]
    hdu.header = beam.attach_to_header(hdu.header)
    tab_header["EXTNAME"] = "BEAMS"
    tab_header["NCHAN"] = len(freqs)
    tab_header["NPOL"] = 1  # Only one pol for now
    new_hdulist = fits.HDUList([hdu, tab_hdu])

    outf = Path(tmpdir) / "3d.fits"
    new_hdulist.writeto(outf, overwrite=True)

    yield TestImage(
        path=outf,
        beam=beams,
        data=cube,
        pix_scale=pix_scale,
    )
    outf.unlink()


@pytest.fixture
def mirsmooth_3d(make_3d_image: TestImage) -> TestImage:
    """Smooth a FITS image to a target beam using MIRIAD.

    Args:
        outf (str): FITS image to smooth.
        target_beam (Beam): Target beam.

    Returns:
        Tuple[str, str, str]: Names of the output images.
    """
    target_beam = Beam(60 * u.arcsec, 60 * u.arcsec, 0 * u.deg)
    outim = make_3d_image.path.with_suffix(".im")
    cmd = f"fits op=xyin in={make_3d_image.path.as_posix()} out={outim.as_posix()}"
    sp.run(cmd.split())

    smoothim = make_3d_image.path.with_suffix(".smooth.im")
    cmd = f"convol map={outim} fwhm={target_beam.major.to(u.arcsec).value},{target_beam.minor.to(u.arcsec).value} pa={target_beam.pa.to(u.deg).value} options=cube out={smoothim}"
    sp.run(cmd.split())

    smoothfits = outim.with_suffix(".mirsmooth_3d.fits")
    cmd = f"fits op=xyout in={smoothim} out={smoothfits}"
    sp.run(cmd.split())

    yield TestImage(
        path=smoothfits,
        beam=target_beam,
        data=fits.getdata(smoothfits),
        pix_scale=make_3d_image.pix_scale,
    )

    for f in (outim, smoothim, smoothfits):
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()


def test_robust_3d(make_3d_image, mirsmooth_3d):
    """Test the robust mode."""
    beamcon_3D.smooth_fits_cube(
        infiles_list=[make_3d_image.path],
        suffix="robust",
        conv_mode="robust",
        mode="total",
        bmaj=mirsmooth_3d.beam.major.to(u.arcsec).value,
        bmin=mirsmooth_3d.beam.minor.to(u.arcsec).value,
        bpa=mirsmooth_3d.beam.pa.to(u.deg).value,
    )

    fname_beamcon = make_3d_image.path.with_suffix(".robust.fits")
    assert check_images(
        mirsmooth_3d.path, fname_beamcon
    ), "Beamcon does not match Miriad"


# def test_astropy(self):
#     """Test the astropy mode."""
#     beamcon_3D.main(
#         infile=[self.test_image],
#         suffix="astropy",
#         conv_mode="astropy",
#         mode="total",
#         bmaj=self.target_beam.major.to(u.arcsec).value,
#         bmin=self.target_beam.minor.to(u.arcsec).value,
#         bpa=self.target_beam.pa.to(u.deg).value,
#     )

#     fname_beamcon = self.test_image.replace(".fits", ".astropy.fits")
#     self.files.append(fname_beamcon)
#     check_images(self.test_cube, fname_beamcon), "Beamcon does not match Miriad"

# def test_scipy(self):
#     """Test the scipy mode."""
#     beamcon_3D.main(
#         infile=[self.test_image],
#         suffix="scipy",
#         conv_mode="scipy",
#         mode="total",
#         bmaj=self.target_beam.major.to(u.arcsec).value,
#         bmin=self.target_beam.minor.to(u.arcsec).value,
#         bpa=self.target_beam.pa.to(u.deg).value,
#     )
#     fname_beamcon = self.test_image.replace(".fits", ".scipy.fits")
#     self.files.append(fname_beamcon)
#     check_images(self.test_cube, fname_beamcon), "Beamcon does not match Miriad"

#     def tearDown(self) -> None:
#         """Tear down the test."""
#         cleanup(self.files)


# if __name__ == "__main__":
#     unittest.TestLoader.sortTestMethodsUsing = None
#     suite = unittest.TestLoader().loadTestsFromTestCase(test_Beamcon3D)
#     unittest.TextTestRunner(verbosity=1).run(suite)
