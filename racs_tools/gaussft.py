#!/usr/bin/env python3
"""
Code to generate FT of final 2D-Gaussian to be used
for convolving an image. The code deconvolves the input
psf. The intrinsic psf must be specified.

Python version of gaussft.f by Wasim Raja
"""
from typing import Tuple

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
) -> Tuple[np.ndarray, float]:
    """
    Compute the Fourier transform of a 2D Gaussian for convolution.

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

    u_cosbpa, u_sinbpa = u * np.cos(bpa_rad), u * np.sin(bpa_rad)
    u_cosbpa_in, u_sinbpa_in = u * np.cos(bpa_in_rad), u * np.sin(bpa_in_rad)

    v_cosbpa, v_sinbpa = v * np.cos(bpa_rad), v * np.sin(bpa_rad)
    v_cosbpa_in, v_sinbpa_in = v * np.cos(bpa_in_rad), v * np.sin(bpa_in_rad)

    g_amp = np.sqrt(2.0 * np.pi * sx * sy)

    dg_amp = np.sqrt(2.0 * np.pi * sx_in * sy_in)

    g_ratio = g_amp / dg_amp

    # Vectorized calculation of ur, vr, g_arg, and dg_arg
    ur = u_cosbpa[:, np.newaxis] - v_sinbpa[np.newaxis, :]
    vr = u_sinbpa[:, np.newaxis] + v_cosbpa[np.newaxis, :]
    g_arg = -2.0 * np.pi**2 * ((sx * ur) ** 2 + (sy * vr) ** 2)

    ur_in = u_cosbpa_in[:, np.newaxis] - v_sinbpa_in[np.newaxis, :]
    vr_in = u_sinbpa_in[:, np.newaxis] + v_cosbpa_in[np.newaxis, :]
    dg_arg = -2.0 * np.pi**2 * ((sx_in * ur_in) ** 2 + (sy_in * vr_in) ** 2)

    # Vectorized calculation of g_final
    g_final = g_ratio * np.exp(g_arg - dg_arg)

    return g_final, g_ratio
