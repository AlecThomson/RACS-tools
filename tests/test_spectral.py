import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from racs_tools.beamcon_3D import initfiles
from radio_beam import Beams


@pytest.fixture
def spec_axis() -> WCS:
    header = fits.Header()
    header["NAXIS"] = 1
    header["NAXIS1"] = 288
    header["CTYPE1"] = "FREQ"
    header["CRVAL1"] = 800e6
    header["CRPIX1"] = 1
    header["CDELT1"] = 1e6
    return WCS(header).spectral


def test_crpix(spec_axis: WCS) -> None:
    crpix = (
        int(np.squeeze(spec_axis.wcs.crpix))
        if not np.isscalar(spec_axis.wcs.crpix)
        else int(spec_axis.wcs.crpix)
    )
    assert crpix == 1


def test_nchan(spec_axis: WCS) -> None:
    nchans = spec_axis.array_shape[0]
    assert nchans == 288


def _write_cube(path, ctypes: list[str]) -> None:
    """Write a small cube whose axes (FITS order) have the given CTYPEs.

    An empty CTYPE leaves that axis as a plain linear axis, so the header has
    no spectral or no Stokes axis for WCS to find.
    """
    shape = {3: (3, 8, 8), 4: (1, 3, 8, 8)}[len(ctypes)]
    hdu = fits.PrimaryHDU(data=np.ones(shape, dtype=np.float32))
    for i, ctype in enumerate(ctypes, start=1):
        if ctype:
            hdu.header[f"CTYPE{i}"] = ctype
        hdu.header[f"CRPIX{i}"] = 2 if ctype == "FREQ" else 1
        hdu.header[f"CDELT{i}"] = 1e6 if ctype == "FREQ" else 1
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.writeto(path)


@pytest.fixture
def commonbeams() -> Beams:
    return Beams(
        major=[10, 11, 12] * u.arcsec,
        minor=[8, 9, 10] * u.arcsec,
        pa=[0, 0, 0] * u.deg,
    )


def test_initfiles_attaches_reference_channel_beam(tmp_path, commonbeams):
    infile = tmp_path / "cube.fits"
    _write_cube(infile, ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"])

    outfile = initfiles(infile, commonbeams, tmp_path, mode="total", suffix="sm")

    header = fits.getheader(outfile)
    # CRPIX3 = 2 selects the second channel (index 1)
    assert header["BMAJ"] == pytest.approx(commonbeams[1].major.to(u.deg).value)


@pytest.mark.parametrize(
    ("ctypes", "missing"),
    [
        (["RA---SIN", "DEC--SIN", "", "STOKES"], "spectral"),
        (["RA---SIN", "DEC--SIN", "FREQ"], "Stokes"),
    ],
)
def test_initfiles_missing_axis_raises(tmp_path, commonbeams, ctypes, missing):
    """A header without the axis gives a clear ValueError, not a TypeError."""
    infile = tmp_path / "cube.fits"
    _write_cube(infile, ctypes)

    with pytest.raises(ValueError, match=f"No {missing} axis found"):
        initfiles(infile, commonbeams, tmp_path, mode="total", suffix="sm")
