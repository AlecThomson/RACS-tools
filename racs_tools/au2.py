#!/usr/bin/env python
""" For getting fluxes right in Jy/beam """
__author__ = "Tessa Vernstrom"

from scipy import *
import numpy as np
import math



def gaussianDeconvolve(smaj, smin, spa, bmaj, bmin, bpa):
    """'s' as in 'source', 'b' as in 'beam'. All arguments in
    radians. (Well, major and minor axes can be in any units, so long
    as they're consistent.)

    Returns dmaj, dmin, dpa, status
    Return units are consistent with the inputs.
    status is one of 'ok', 'pointlike', 'fail'

    Derived from miriad gaupar.for:GauDfac()

    We currently don't do a great job of dealing with pointlike
    sources. I've added extra code ensure smaj >= bmaj, smin >= bmin,
    and increased coefficient in front of "limit" from 0.1 to
    0.5. Feel a little wary about that first change.
    """

    from numpy import cos, sin, sqrt, min, abs, arctan2
    import numpy as np
    
    spa=np.radians(spa)
    bpa=np.radians(bpa)
    if smaj < bmaj:
        smaj = bmaj
    if smin < bmin:
        smin = bmin

    alpha = ((smaj * cos (spa))**2 + (smin * sin (spa))**2 -
             (bmaj * cos (bpa))**2 - (bmin * sin (bpa))**2)
    beta = ((smaj * sin (spa))**2 + (smin * cos (spa))**2 -
            (bmaj * sin (bpa))**2 - (bmin * cos (bpa))**2)
    gamma = 2 * ((smin**2 - smaj**2) * sin (spa) * cos (spa) -
                 (bmin**2 - bmaj**2) * sin (bpa) * cos (bpa))
#    print smaj,smin
#    print alpha,beta,gamma
    s = alpha + beta
    t = sqrt ((alpha - beta)**2 + gamma**2)
#    print s,t
    dmaj = sqrt (0.5 * (s + t))
    if s>t:
        dmin = sqrt (0.5 * (s - t))
    else:
        dmin= 0
#    print dmaj,dmin
    if alpha < 0 or beta < 0:
        dmaj = dmin = dpa = 0
    
#    if(smaj>bmaj):
#        dmaj= sqrt (0.5 * (s + t))
    if abs (gamma) + abs (alpha - beta) == 0:
        dpa = 0
    else:
        dpa=0.5 * arctan2 (-gamma, alpha - beta)
#    if((s>=t)&(bmin!=smin)):
#        dmin=sqrt (0.5 * (s - t))

    return dmaj, dmin, np.degrees(dpa)





def gauss_factor(beamConv, beamOrig=None, dx1=1, dy1=1):
    """
    Calculates the scaling factor to be applied after convolving
    a map in Jy/beam with a gaussian to get fluxes in Jy/beam again.

    This function is a copy of the FORTRAN gaufac function from the Miriad
    package, which determine the Gaussian parameters resulting from
    convolving two gaussians. This function yields the same result as
    the MIRIAD gaufac function.

    Parameters
    ----------
    beamConv : list
        A list of the [major axis, minor axis, position_angle]
        of the gaussion used for convolution.
    beamOrig :
        Same format as beamConv but giving the parameters of the original
        beam of the map. As Default the self.resolution list is used.
    dx1, dy1 : floats
        Being the pixel size in both dimensions of the map.
        By default the ``CDELT1`` and ``CDELT2`` keywords from the
        fits header are used.

    Returns
    -------
    fac :
        Factor for the output Units.
    amp :
        Amplitude of resultant gaussian.
    bmaj, bmin :
        Major and minor axes of resultant gaussian.
    bpa :
        Position angle of the resulting gaussian.
    """
    # include 'mirconst.h'
    # Define cosine and Sinus of the position Angles of the
    # Gaussians
    arcsecInGrad=1#(1./3600)*(np.pi/180.)
    deg2Grad=(np.pi/180)
    bmaj2, bmin2, bpa2 = beamConv
    bmaj2, bmin2, bpa2 = (bmaj2 * arcsecInGrad, bmin2 *
                          arcsecInGrad, bpa2 * deg2Grad)
    #if beamOrig is None:
    bmaj1, bmin1, bpa1 = beamOrig
    bmaj1, bmin1, bpa1 = (bmaj1 * arcsecInGrad,
                              bmin1 * arcsecInGrad,
                              bpa1 * deg2Grad)
    #if dx1 is None:
    dx1 = dx1 * arcsecInGrad
    #if dy1 is None:
    dy1 = dy1 * arcsecInGrad
    cospa1 = math.cos(bpa1)
    cospa2 = math.cos(bpa2)
    sinpa1 = math.sin(bpa1)
    sinpa2 = math.sin(bpa2)
    alpha = ((bmaj1 * cospa1) ** 2
             + (bmin1 * sinpa1) ** 2
             + (bmaj2 * cospa2) ** 2
             + (bmin2 * sinpa2) ** 2)
    beta = ((bmaj1 * sinpa1) ** 2
            + (bmin1 * cospa1) ** 2
            + (bmaj2 * sinpa2) ** 2
            + (bmin2 * cospa2) ** 2)
    gamma = (2 * ((bmin1 ** 2 - bmaj1 ** 2)
                  * sinpa1 * cospa1
                  + (bmin2 ** 2 - bmaj2 ** 2)
                  * sinpa2 * cospa2))
    s = alpha + beta
    t = math.sqrt((alpha - beta) ** 2 + gamma ** 2)
    bmaj = math.sqrt(0.5 * (s + t))
    bmin = math.sqrt(0.5 * (s - t))
    if (abs(gamma) + abs(alpha - beta)) == 0:
        bpa = 0.0
    else:
        bpa = 0.5 * np.arctan2(-1 * gamma, alpha - beta)
        #print alpha,beta,gamma
    amp = (math.pi / (4.0 * math.log(2.0)) * bmaj1 * bmin1 * bmaj2 * bmin2
           / math.sqrt(alpha * beta - 0.25 * gamma * gamma))
    fac = ((math.sqrt(dx1 ** 2) * math.sqrt(dy1 ** 2))) / amp

    return fac, amp, bmaj , bmin , np.degrees(bpa)
