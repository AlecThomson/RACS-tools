#!/usr/bin/env python3
"""
Code to generate FT of final 2D-Gaussian to be used
for convolving an image. The code deconvolves the input
psf. The intrinsic psf must be specified.

Python version of gaussft.f by Wasim Raja
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def gaussft(
    bmin_in: float,
    bmaj_in: float,
    bpa_in: float,
    bmin: float,
    bmaj: float,
    bpa: float,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute the Fourier transform of a 2D Gaussian for convolution.

    The returned taper is real and takes the dtype of ``u`` and ``v``, so a
    single-precision grid gives a single-precision taper. It is filled in one
    pass, meaning only the taper itself is ever allocated at image size.

    Parameters:
        bmin_in (float): Intrinsic psf BMIN (degrees)
        bmaj_in (float): Intrinsic psf BMAJ (degrees)
        bpa_in (float): Intrinsic psf BPA (degrees)
        bmin (float): Final psf BMIN (degrees)
        bmaj (float): Final psf BMAJ (degrees)
        bpa (float): Final psf BPA (degrees)
        u (np.ndarray): Fourier coordinates corresponding to image coord x
        v (np.ndarray): Fourier coordinates corresponding to image coord y

    Returns:
        g_final (np.ndarray): Final array to be multiplied to FT(image) for convolution
            in the FT domain.
        g_ratio (float): Factor for flux scaling
    """
    deg2rad = np.pi / 180.0

    bmaj_in_rad, bmin_in_rad, bpa_in_rad = (
        bmaj_in * deg2rad,
        bmin_in * deg2rad,
        bpa_in * deg2rad,
    )
    bmaj_rad, bmin_rad, bpa_rad = bmaj * deg2rad, bmin * deg2rad, bpa * deg2rad

    sx, sy = (
        bmaj_rad / (2 * np.sqrt(2.0 * np.log(2.0))),
        bmin_rad / (2 * np.sqrt(2.0 * np.log(2.0))),
    )
    sx_in, sy_in = (
        bmaj_in_rad / (2.0 * np.sqrt(2.0 * np.log(2.0))),
        bmin_in_rad / (2.0 * np.sqrt(2.0 * np.log(2.0))),
    )

    g_amp = np.sqrt(2.0 * np.pi * sx * sy)

    dg_amp = np.sqrt(2.0 * np.pi * sx_in * sy_in)

    g_ratio = g_amp / dg_amp

    cos_bpa, sin_bpa = np.cos(bpa_rad), np.sin(bpa_rad)
    cos_bpa_in, sin_bpa_in = np.cos(bpa_in_rad), np.sin(bpa_in_rad)
    two_pi_sq = 2.0 * np.pi**2

    g_final = np.empty((u.size, v.size), dtype=u.dtype)
    for i in range(u.size):
        u_cosbpa, u_sinbpa = u[i] * cos_bpa, u[i] * sin_bpa
        u_cosbpa_in, u_sinbpa_in = u[i] * cos_bpa_in, u[i] * sin_bpa_in
        for j in range(v.size):
            ur = u_cosbpa - v[j] * sin_bpa
            vr = u_sinbpa + v[j] * cos_bpa
            g_arg = -two_pi_sq * ((sx * ur) ** 2 + (sy * vr) ** 2)

            ur_in = u_cosbpa_in - v[j] * sin_bpa_in
            vr_in = u_sinbpa_in + v[j] * cos_bpa_in
            dg_arg = -two_pi_sq * ((sx_in * ur_in) ** 2 + (sy_in * vr_in) ** 2)

            g_final[i, j] = g_ratio * np.exp(g_arg - dg_arg)

    return g_final, g_ratio
