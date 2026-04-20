from astropy.wcs import WCS
from astropy.io import fits
import numpy as np

import pytest

@pytest.fixture
def spec_axis() -> WCS:
    header = fits.Header()
    header["NAXIS"] = 1
    header["NAXIS1"] = 288
    header["CTYPE1"] = "FREQ"
    header["CRVAL1"] = 800e6
    header["CRPIX1"] = 1
    header["CDELT1"] = 1e6
    return  WCS(header).spectral

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