#!/usr/bin/env python3
"""Tests for the UV-domain ("robust") convolution.

These pin the properties that keep `convolve` affordable on a wide image: it
transforms in the image's own precision, and it keeps only the non-redundant
half of the spectrum. A regression in either quietly triples peak memory,
which is how a single 19533x17000 plane came to need ~17 GB.
"""

import numpy as np
import pytest
import scipy.fft
from astropy import units as u
from astropy.io import fits
from racs_tools import gaussft
from racs_tools.convolve_uv import (
    beam_contains,
    convolve,
    my_ceil,
    round_common_beam,
    round_up,
    smooth,
)
from radio_beam import Beam

OLD_BEAM = Beam(12 * u.arcsec, 10 * u.arcsec, 20 * u.deg)
NEW_BEAM = Beam(18 * u.arcsec, 15 * u.arcsec, 35 * u.deg)
PIX = (2.5 * u.arcsec).to(u.deg)
BEAM_KWARGS = {
    "bmin_in": OLD_BEAM.minor.to(u.deg).value,
    "bmaj_in": OLD_BEAM.major.to(u.deg).value,
    "bpa_in": OLD_BEAM.pa.to(u.deg).value,
    "bmin": NEW_BEAM.minor.to(u.deg).value,
    "bmaj": NEW_BEAM.major.to(u.deg).value,
    "bpa": NEW_BEAM.pa.to(u.deg).value,
}


def make_image(shape=(129, 97), dtype=np.float32, nans=False) -> np.ndarray:
    """A reproducible test image, optionally with a NaN blank."""
    rng = np.random.default_rng(1234)
    image = rng.normal(size=shape).astype(dtype)
    image[shape[0] // 2 :, : shape[1] // 3] += 5.0
    if nans:
        image[:10, :15] = np.nan
    return image


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussft_taper_is_real_and_follows_the_grid(dtype):
    """The taper is real, and single precision in gives single precision out."""
    u_image = np.fft.fftfreq(65, d=1.2e-5).astype(dtype)
    v_image = np.fft.rfftfreq(48, d=1.2e-5).astype(dtype)

    taper, ratio = gaussft.gaussft(u=u_image, v=v_image, **BEAM_KWARGS)

    assert taper.dtype == dtype
    assert taper.shape == (u_image.size, v_image.size)
    assert np.isfinite(ratio)


def test_gaussft_matches_a_double_precision_reference():
    """The single pass agrees with the same expression built as whole arrays."""
    u_image = np.fft.fftfreq(65, d=1.2e-5)
    v_image = np.fft.rfftfreq(48, d=1.2e-5)

    taper, ratio = gaussft.gaussft(u=u_image, v=v_image, **BEAM_KWARGS)

    fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
    sx, sy = (
        np.deg2rad(BEAM_KWARGS["bmaj"]) * fwhm_to_sigma,
        np.deg2rad(BEAM_KWARGS["bmin"]) * fwhm_to_sigma,
    )
    sx_in, sy_in = (
        np.deg2rad(BEAM_KWARGS["bmaj_in"]) * fwhm_to_sigma,
        np.deg2rad(BEAM_KWARGS["bmin_in"]) * fwhm_to_sigma,
    )
    bpa, bpa_in = np.deg2rad(BEAM_KWARGS["bpa"]), np.deg2rad(BEAM_KWARGS["bpa_in"])

    def arg(uu, vv, s_maj, s_min, angle):
        ur = uu[:, None] * np.cos(angle) - vv[None, :] * np.sin(angle)
        vr = uu[:, None] * np.sin(angle) + vv[None, :] * np.cos(angle)
        return -2 * np.pi**2 * ((s_maj * ur) ** 2 + (s_min * vr) ** 2)

    expected_ratio = np.sqrt(2 * np.pi * sx * sy) / np.sqrt(2 * np.pi * sx_in * sy_in)
    expected = expected_ratio * np.exp(
        arg(u_image, v_image, sx, sy, bpa) - arg(u_image, v_image, sx_in, sy_in, bpa_in)
    )

    assert np.isclose(ratio, expected_ratio)
    assert np.allclose(taper, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("shape", [(128, 96), (129, 97)])
@pytest.mark.parametrize("nans", [False, True])
def test_convolve_agrees_across_precision(shape, nans):
    """A single precision plane convolves to the same answer as a double one.

    Also covers odd axis lengths, which a half-spectrum transform has to be
    told the original shape of to invert.
    """
    single = convolve(
        image=make_image(shape, np.float32, nans),
        old_beam=OLD_BEAM,
        new_beam=NEW_BEAM,
        dx=PIX,
        dy=PIX,
    )
    double = convolve(
        image=make_image(shape, np.float64, nans),
        old_beam=OLD_BEAM,
        new_beam=NEW_BEAM,
        dx=PIX,
        dy=PIX,
    )

    assert single.image.shape == shape
    assert np.isclose(single.scaling_factor, double.scaling_factor)
    assert np.array_equal(np.isnan(single.image), np.isnan(double.image))
    assert np.isnan(single.image).any() == nans

    finite = ~np.isnan(double.image)
    assert np.allclose(
        single.image[finite],
        double.image[finite],
        rtol=0,
        atol=1e-4 * double.image[finite].std(),
    )


def test_convolve_keeps_single_precision_for_fits_data(tmp_path, monkeypatch):
    """A plane read from a BITPIX -32 file stays in single precision.

    `fits.getdata` hands back a big-endian ">f4" array. That is not equal to
    `np.float32`, so a dtype identity check here would send every real image
    down the double precision path and undo the saving.
    """
    path = tmp_path / "plane.fits"
    fits.PrimaryHDU(data=make_image(dtype=np.float32)).writeto(path)
    image = fits.getdata(path)
    assert image.dtype == np.dtype(">f4"), "expected big-endian FITS data"

    spectra = []
    real_rfft2 = scipy.fft.rfft2

    def spy(*args, **kwargs):
        result = real_rfft2(*args, **kwargs)
        spectra.append(result)
        return result

    monkeypatch.setattr(scipy.fft, "rfft2", spy)
    convolve(image=image, old_beam=OLD_BEAM, new_beam=NEW_BEAM, dx=PIX, dy=PIX)

    assert spectra, "the image should have been transformed"
    for spectrum in spectra:
        assert spectrum.dtype == np.complex64, (
            f"spectrum promoted to {spectrum.dtype}, which doubles peak memory"
        )
        # Half spectrum only: the redundant conjugate half is never stored
        assert spectrum.shape == (image.shape[0], image.shape[1] // 2 + 1)


def test_convolve_leaves_the_input_alone():
    """The caller's array is not consumed, with or without NaNs to fill."""
    for nans in (False, True):
        image = make_image(nans=nans)
        before = image.copy()
        convolve(image=image, old_beam=OLD_BEAM, new_beam=NEW_BEAM, dx=PIX, dy=PIX)
        assert np.array_equal(image, before, equal_nan=True)


def test_rounding_keeps_exact_values():
    """Round-off from unit conversion does not bump a value up a step."""
    for k in range(100, 400):
        arcsec = (k / 10 * u.arcsec).to(u.deg).to(u.arcsec).value
        assert np.isclose(my_ceil(arcsec, precision=1), k / 10)
    assert np.isclose(my_ceil(30.51, precision=1), 30.6)
    assert np.isclose(round_up(10.01 * u.deg, decimals=2).value, 10.01)
    assert np.isclose(round_up(10.011 * u.deg, decimals=2).value, 10.02)


def test_round_common_beam_contains_inputs():
    """Rounding the PA must not leave an input outside the common beam."""
    beam = Beam(20 * u.arcsec, 10 * u.arcsec, 10.005 * u.deg)
    rounded = round_common_beam(beam, [beam])
    assert beam_contains(rounded, beam)


def test_beam_contains_one_dimensional_kernel():
    target = Beam(20 * u.arcsec, 20 * u.arcsec, 0 * u.deg)
    for pa in (0, 37, 90, -45):
        assert beam_contains(target, Beam(20 * u.arcsec, 10 * u.arcsec, pa * u.deg))
    assert not beam_contains(target, Beam(21 * u.arcsec, 10 * u.arcsec, 0 * u.deg))


@pytest.mark.parametrize("sign", [(-1, 1), (1, -1), (-1, -1)])
def test_negative_pixel_size_keeps_pa(sign):
    """A negative CDELT1, as in most headers, must not mirror the PA."""
    image = make_image(dtype=np.float64)
    want = convolve(image, OLD_BEAM, NEW_BEAM, PIX, PIX).image
    got = convolve(image, OLD_BEAM, NEW_BEAM, sign[0] * PIX, sign[1] * PIX).image
    assert np.allclose(got, want)


def test_non_square_pixels():
    """Rows are spaced by dy and columns by dx."""
    old, new = Beam(6 * u.arcsec), Beam(12 * u.arcsec)
    dx, dy = 1.5, 3.0
    image = np.zeros((64, 96))
    image[32, 48] = 1
    out = convolve(image, old, new, dx * u.arcsec, dy * u.arcsec).image
    rows, cols = np.indices(out.shape)
    total = out.sum()
    var_y = (out * (rows - (out * rows).sum() / total) ** 2).sum() / total * dy**2
    var_x = (out * (cols - (out * cols).sum() / total) ** 2).sum() / total * dx**2
    want = (12**2 - 6**2) / (8 * np.log(2))
    assert np.isclose(var_x, want, rtol=1e-3)
    assert np.isclose(var_y, want, rtol=1e-3)


def test_all_nan_scaling_factor():
    """An all-NaN image gets the same scaling factor as a finite one."""
    old, new = Beam(10 * u.arcsec), Beam(20 * u.arcsec)
    nan = np.full((16, 16), np.nan, dtype=np.float32)
    result = convolve(nan, old, new, PIX, PIX)
    assert np.isnan(result.image).all()
    assert np.isclose(result.scaling_factor, 2.0)


def test_near_equal_beams_keep_nan_block():
    """A solid NaN region stays blank when the beams are nearly equal."""
    old = Beam(10 * u.arcsec)
    new = Beam(10.000001 * u.arcsec, 10 * u.arcsec, 0 * u.deg)
    image = make_image(shape=(256, 256))
    image[64:192, 64:192] = np.nan
    out = smooth(image, old, new, PIX, PIX)
    assert np.array_equal(np.isnan(out), np.isnan(image))


@pytest.mark.parametrize("conv_mode", ["robust", "scipy", "astropy"])
def test_unreachable_and_zero_beams_raise(conv_mode):
    """No silent amplification, inf or NaN output for impossible kernels."""
    image = make_image(shape=(16, 16), dtype=np.float64)
    small = Beam(10 * u.arcsec)
    big = Beam(20 * u.arcsec, 10 * u.arcsec, 0 * u.deg)
    with pytest.raises(ValueError):
        smooth(image, big, small, PIX, PIX, conv_mode=conv_mode)
    with pytest.raises(ValueError):
        smooth(image, Beam(0 * u.deg), small, PIX, PIX, conv_mode=conv_mode)
    # A one-dimensional kernel is fine
    out = smooth(image, big, Beam(20 * u.arcsec), PIX, PIX, conv_mode=conv_mode)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("conv_mode", ["scipy", "astropy", "astropy_fft"])
def test_kernel_modes_reject_non_square_pixels(conv_mode):
    """The image-plane kernel modes assume square pixels."""
    image = make_image(shape=(32, 32), dtype=np.float64)
    with pytest.raises(ValueError, match="square"):
        smooth(image, OLD_BEAM, NEW_BEAM, PIX, 2 * PIX, conv_mode=conv_mode)
    # The sign of CDELT1 does not matter
    out = smooth(image, OLD_BEAM, NEW_BEAM, -PIX, PIX, conv_mode=conv_mode)
    assert np.isfinite(out).all()
