#!/usr/bin/env python
""" Find bad channels by checking statistics of each channel image. """

import argparse
import warnings
from typing import List, Tuple, Union

import astropy.units as u
import numpy as np
from astropy.stats import mad_std
from spectral_cube import SpectralCube
from spectral_cube.utils import SpectralCubeWarning

warnings.filterwarnings(action="ignore", category=SpectralCubeWarning, append=True)


#############################################
####### ADAPTED FROM SCRIPT BY G. HEALD #####
#############################################
def getcube(filename: str) -> SpectralCube:
    """Read FITS file as SpectralCube
    Masks out 0Jy/beam pixels

    Args:
        filename (str): Filename

    Returns:
        SpectralCube: Data cube
    """
    cube = SpectralCube.read(filename)
    mask = ~(cube == 0 * u.jansky / u.beam)
    cube = cube.with_mask(mask)
    return cube


def getbadchans(
    qcube: SpectralCube,
    ucube: SpectralCube,
    cliplev: float = 5,
) -> np.ndarray:
    """Find bad channels in Stokes Q and U cubes

    Args:
        qcube (SpectralCube): Stokes Q data
        ucube (SpectralCube): Stokes U data
        cliplev (float, optional): Number stddev above median to clip. Defaults to 5.

    Returns:
        np.ndarray: Bad channel boolean array
    """
    assert len(ucube.spectral_axis) == len(qcube.spectral_axis)

    qnoisevals = (
        qcube.apply_function_parallel_spatial(
            function=mad_std,
            parallel=False,
            use_memmap=True,
            ignore_nan=True,
        )[:, 0, 0].unitless_filled_data[:]
        * qcube.unit
    )
    unoisevals = (
        ucube.apply_function_parallel_spatial(
            function=mad_std,
            parallel=False,
            use_memmap=True,
            ignore_nan=True,
        )[:, 0, 0].unitless_filled_data[:]
        * ucube.unit
    )
    qmeannoise = np.nanmedian(qnoisevals)
    qstdnoise = mad_std(qnoisevals, ignore_nan=True)
    print(
        f"Median Q noise=({qmeannoise.value:0.3f}±{qstdnoise.value:0.3f}) / ({qmeannoise.unit})"
    )
    umeannoise = np.nanmedian(unoisevals)
    ustdnoise = mad_std(unoisevals, ignore_nan=True)
    print(
        f"Median U noise=({umeannoise.value:0.3f}±{ustdnoise.value:0.3f}) / ({umeannoise.unit})"
    )
    qbadones = np.logical_or(
        qnoisevals > (qmeannoise + cliplev * qstdnoise), ~np.isfinite(qnoisevals)
    )
    ubadones = np.logical_or(
        unoisevals > (umeannoise + cliplev * ustdnoise), ~np.isfinite(unoisevals)
    )
    print(f"{qbadones.sum()} of {len(qcube.spectral_axis)} are bad in Q")
    print(f"{ubadones.sum()} of {len(ucube.spectral_axis)} are bad in U")
    total_bad = np.logical_or(qbadones, ubadones)
    print(f"{total_bad.sum()} of {len(qcube.spectral_axis)} are bad in Q -or- U")
    return total_bad


def blankchans(
    qcube: SpectralCube, ucube: SpectralCube, totalbad: np.ndarray, blank: bool = False
) -> Tuple[SpectralCube, SpectralCube]:
    """Mask out bad channels

    Args:
        qcube (SpectralCube): Stokes Q data
        ucube (SpectralCube): Stokes U data
        totalbad (np.ndarray): Flagged bad channels
        blank (bool, optional): Print flags to screen. Defaults to False.

    Returns:
        Tuple[SpectralCube, SpectralCube]: _description_
    """
    chans = np.arange(len(qcube.spectral_axis))
    badchans = chans[totalbad]
    badfreqs = qcube.spectral_axis[totalbad]
    if not blank:
        print(
            "Nothing will be blanked, but these are the channels/frequencies that would be with the -b option activated:"
        )
    print(f"Bad channels are {badchans}")
    print(f"Bad frequencies are {badfreqs}")
    totalgood = ~totalbad
    q_msk = qcube.mask_channels(totalgood)
    u_msk = ucube.mask_channels(totalgood)
    return q_msk, u_msk


def writefits(qcube: SpectralCube, ucube: SpectralCube, qfile: str, ufile: str) -> None:
    """Write output to disk

    Args:
        qcube (SpectralCube): Stokes Q data
        ucube (SpectralCube): Stokes U data
        qfile (str): Original Q file
        ufile (str): Original U file
    """
    outfile = qfile.replace(".fits", ".blanked.fits")
    print(f"Writing to {outfile}")
    qcube.write(outfile, format="fits", overwrite=True)
    outfile = ufile.replace(".fits", ".blanked.fits")
    print(f"Writing to {outfile}")
    ucube.write(outfile, format="fits", overwrite=True)


def main(
    qfile: str,
    ufile: str,
    blank: bool = False,
    cliplev: float = 5,
    iterate: int = 1,
    outfile: str = None,
) -> None:
    """Flag bad channels in Stokes Q and U cubes

    Args:
        qfile (str): Stokes Q fits file
        ufile (str): Stokes U fits file
        blank (bool, optional): Flag bad data and save to disk. Defaults to False.
        cliplev (float, optional): Number of stddev above median to flag. Defaults to 5.
        iterate (int, optional): Number of flagging iterations. Defaults to 1.
        outfile (str, optional): File to write flagged channels to. Defaults to None.
    """
    qcube = getcube(qfile)
    ucube = getcube(ufile)

    assert len(ucube.spectral_axis) == len(
        qcube.spectral_axis
    ), "Cubes have different number of channels"

    # Iterate
    for i in range(iterate):
        print(f"Flagging iteration {i+1} of {iterate}")
        totalbad = getbadchans(
            qcube,
            ucube,
            cliplev=cliplev,
        )
        qcube, ucube = blankchans(qcube, ucube, totalbad, blank=blank)

    if blank:
        writefits(qcube, qcube, qfile, ufile)

    if outfile is not None:
        print(f"Saving bad files to {outfile}")
        np.savetxt(outfile, totalbad)


def cli() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Cube is a Stokes cube, returning spectral cube for I component",
    )
    warnings.filterwarnings("ignore", message="Invalid value encountered in less")
    warnings.filterwarnings("ignore", message="Invalid value encountered in greater")
    warnings.filterwarnings("ignore", message="overflow encountered in square")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("qfile", type=str, help="Stokes Q fits file")
    parser.add_argument("ufile", type=str, help="Stokes U fits file")
    parser.add_argument(
        "-b",
        "--blank",
        help="Blank bad channels? [default False - just print out bad channels]",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--cliplev",
        help="Clip level in sigma, make this number lower to be more aggressive [default 5]",
        default=5.0,
        type=float,
    )

    parser.add_argument(
        "-i",
        "--iterate",
        help="Iterate flagging check N times [dafult 1 -- one pass only]",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-f",
        "--file",
        help="Filename to write bad channel indices to file [None --  do not write]",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    main(
        qfile=args.qfile,
        ufile=args.ufile,
        blank=args.blank,
        cliplev=args.cliplev,
        iterate=args.iterate,
        outfile=args.file,
    )


if __name__ == "__main__":
    cli()
