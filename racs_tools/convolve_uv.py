#!/usr/bin/env python
""" Fast convolution in the UV domain """
__author__ = "Wasim Raja"

import numpy as np
import astropy.units as units
import racs_tools.gaussft as gaussft

def convolve(image, old_beam, new_beam, dx, dy):
    """Convolve by X-ing in the Fourier domain.
        - convolution with Gaussian kernels only 
        - no need for generation of a kernel image
        - direct computation of the FT of the kernel
        - dimension of FT = dimension of image

    Args:
        image (2D array): The image to be convolved.
        old_beam (radio_beam.Beam): Current image PSF.
        new_beam (radio_beam.Beam): Target image PSF.
        dx (float): Grid size in x in degrees (e.g. CDELT1)
        dy (float): Grid size in y in degrees (e.g. CDELT2)

    Returns:
        tuple: (convolved image, scaling factor)
    """
    nx = image.shape[0]
    ny = image.shape[1]

    # The coordinates in FT domain:
    u = np.fft.fftfreq(nx, d=dx.to(units.rad).value)
    v = np.fft.fftfreq(ny, d=dy.to(units.rad).value)

    g_final = np.zeros((nx, ny), dtype=float)
    [g_final, g_ratio] = gaussft.gaussft(bmin_in=old_beam.minor.to(units.deg).value,
                                         bmaj_in=old_beam.major.to(units.deg).value,
                                         bpa_in=old_beam.pa.to(units.deg).value,
                                         bmin=new_beam.minor.to(units.deg).value,
                                         bmaj=new_beam.major.to(units.deg).value,
                                         bpa=new_beam.pa.to(units.deg).value,
                                         u=u, v=v,
                                         nx=nx, ny=ny)
    # Perform the x-ing in the FT domain
    im_f = np.fft.fft2(image)

    # Now convolve with the desired Gaussian:
    M = np.multiply(im_f, g_final)
    im_conv = np.fft.ifft2(M)
    im_conv = np.real(im_conv)

    # print("factor: %f" % g_ratio)
    # print("dx: %s" % dx)
    # print("dy: %s" % dy)
    # tmp = old_beam.minor.to(units.deg).value
    # print("bMaj psf: %f , %f" % (tmp,tmp*3600))
    # tmp = bmin_in
    # print("bMaj psf: %f , %f" % (tmp,tmp*3600.0))
    # tmp = bpa_in
    # print("bPA psf: %f " % tmp)
    # tmp = bmaj
    # print("bMaj desired: %f, %f" % (tmp,tmp*3600.0))
    # tmp = bmin
    # print("bMin desired: %f, %f" % (tmp,tmp*3600.0))
    # tmp = bpa
    # print("bPA desired: %f " % tmp)
    return im_conv, g_ratio
