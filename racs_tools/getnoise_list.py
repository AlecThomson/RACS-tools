#!/usr/bin/env python

import functools
import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube
from tqdm import tqdm, trange
import schwimmbad
import time
import warnings
import sys
from spectral_cube.utils import SpectralCubeWarning

warnings.filterwarnings(action="ignore", category=SpectralCubeWarning, append=True)
import psutil

print = functools.partial(print, f"[{psutil.Process().cpu_num()}]", flush=True)


#############################################
####### ADAPTED FROM SCRIPT BY G. HEALD #####
#############################################


def myfit(x, y, fn):
    # Find the width of a Gaussian distribution by computing the second moment of
    # the data (y) on a given axis (x)

    w = np.sqrt(abs(sum(x ** 2 * y) / sum(y)))

    # Or something like this. Probably a Gauss plus a power-law with a lower cutoff is necessary
    # func = lambda x, a, b: np.exp(0.5*x**2/a**2) + x**(-1.*b)

    # [popt, pvar] = curve_fit(func, x, y)
    # Need a newer version of scipy for this to work...

    # a = popt[0]
    # b = popt[1]
    if fn != "":
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.semilogy(x, y, "+", label="Data")
        # plt.semilogy(x, np.exp(-0.5*x**2/a**2) + x**(-1.*b), label="Noise fit")
        plt.semilogy(x, np.exp(-0.5 * x ** 2 / w ** 2), label="Noise fit")
        plt.title("Normalized pixel distribution")
        plt.ylabel("Rel. Num. of Pixels")
        plt.xlabel("Pixel brightness (Jy/beam)")
        plt.legend(loc=3)
        plt.savefig(fn)
        # plt.show()
        plt.close()
    return w


def calcnoise(args):
    """Get noise in plane from cube.
    """
    i, file, totalbad, update = args
    if update:
        print(f"Checking channel {i}")
    if totalbad is not None and totalbad[i]:
        return -1
    else:
        cube = getcube(file)
        plane = cube.unmasked_data[i]
        imsize = plane.shape
        assert len(imsize) == 2
        nx = imsize[-1]
        ny = imsize[-2]
        Id = plane[ny // 3 : 2 * ny // 3, nx // 3 : 2 * nx // 3].flatten()
        if len(Id[np.isnan(Id)]) == len(Id):
            return -1.0
        else:
            rms = np.std(Id)
            mval = np.mean(Id)
            Id = Id[np.logical_and(Id < mval + 3.0 * rms, Id > mval - 3.0 * rms)]
            # print mval,rms,len(Id)

            # hrange = (-1,1)
            # , range=hrange) # 0 = values, 1 = left bin edges
            Ih = np.histogram(Id, bins=100)
            if max(Ih[0]) == 0.0:
                return -1.0
            Ix = Ih[1][:-1] + 0.5 * (Ih[1][1] - Ih[1][0])
            Iv = Ih[0] / float(max(Ih[0]))
            Inoise = myfit(Ix, Iv, "")
            return Inoise


def getcube(filename):
    """Read FITS file as SpectralCube

    Masks out 0Jy/beam pixels

    """
    cube = SpectralCube.read(filename)
    mask = ~(cube == 0 * u.jansky / u.beam)
    cube = cube.with_mask(mask)
    return cube


def getbadchans(
    pool, qcube, ucube, ufile, qfile, totalbad=None, cliplev=5, update=False
):
    """Find deviated channels
    """
    assert len(ucube.spectral_axis) == len(qcube.spectral_axis)
    inputs = [[i, qfile, totalbad, update] for i in range(len(qcube.spectral_axis))]
    if pool.__class__.__name__=="MPIPool" or pool.__class__.__name__=="SerialPool":
        print(f"Checking Q...")
        tic = time.perf_counter()
        qnoisevals = list(pool.map(calcnoise, inputs))
        toc = time.perf_counter()
        print(f"Time taken was {toc - tic}s")

    elif pool.__class__.__name__=="MultiPool":
        qnoisevals = list(
            tqdm(
                pool.imap_unordered(calcnoise, inputs),
                total=len(ucube.spectral_axis),
                desc="Checking Q",
            )
        )
    qnoisevals = np.array(qnoisevals)

    inputs = [[i, ufile, totalbad, update] for i in range(len(ucube.spectral_axis))]
    if pool.__class__.__name__=="MPIPool" or pool.__class__.__name__=="SerialPool":
        print(f"Checking U...")
        tic = time.perf_counter()
        unoisevals = list(pool.map(calcnoise, inputs))
        toc = time.perf_counter()
        print(f"Time taken was {toc - tic}s")

    elif pool.__class__.__name__=="MultiPool":
        unoisevals = list(
            tqdm(
                pool.imap_unordered(calcnoise, inputs),
                total=len(ucube.spectral_axis),
                desc="Checking U",
            )
        )
    unoisevals = np.array(unoisevals)
    qmeannoise = np.median(qnoisevals[abs(qnoisevals) < 1.0])
    qstdnoise = np.std(qnoisevals[abs(qnoisevals) < 1.0])
    print("Q median, std:", qmeannoise, qstdnoise)
    umeannoise = np.median(unoisevals[abs(unoisevals) < 1.0])
    ustdnoise = np.std(unoisevals[abs(unoisevals) < 1.0])
    print("U median, std:", umeannoise, ustdnoise)
    qbadones = np.logical_or(
        qnoisevals > (qmeannoise + cliplev * qstdnoise), qnoisevals == -1.0
    )
    ubadones = np.logical_or(
        unoisevals > (umeannoise + cliplev * ustdnoise), unoisevals == -1.0
    )
    print(
        sum(np.asarray(qbadones, dtype=int)),
        "of",
        len(qcube.spectral_axis),
        "are bad (Q)",
    )
    print(
        sum(np.asarray(ubadones, dtype=int)),
        "of",
        len(ucube.spectral_axis),
        "are bad (U)",
    )
    totalbad = np.logical_or(qbadones, ubadones)
    print(
        sum(np.asarray(totalbad, dtype=int)),
        "of",
        len(qcube.spectral_axis),
        "are bad in Q -or- U",
    )
    return totalbad


def blankchans(qcube, ucube, totalbad, blank=False):
    """Mask out bad chans
    """
    chans = np.array([i for i, chan in enumerate(qcube.spectral_axis)])
    badchans = chans[totalbad]
    badfreqs = qcube.spectral_axis[totalbad]
    if not blank:
        print(
            "Nothing will be blanked, but these are the channels/frequencies that would be with the -b option activated:"
        )
    print(f"Bad channels are {badchans}")
    print(f"Bad frequencies are {badfreqs}")
    totalgood = [not bad for bad in totalbad]
    q_msk = qcube.mask_channels(totalgood)
    u_msk = ucube.mask_channels(totalgood)
    return q_msk, u_msk


def writefits(qcube, ucube, clargs):
    """Write output to disk
    """
    outfile = clargs.qfitslist.replace(".fits", ".blanked.fits")
    print(f"Writing to {outfile}")
    qcube.write(outfile, format="fits", overwrite=True)
    outfile = clargs.ufitslist.replace(".fits", ".blanked.fits")
    print(f"Writing to {outfile}")
    ucube.write(outfile, format="fits", overwrite=True)


def main(pool, clargs):
    qcube = getcube(clargs.qfitslist)
    ucube = getcube(clargs.ufitslist)

    totalbad = None

    if clargs.mpi or clargs.n_cores == 1:
        update = True
    else:
        update = False

    totalbad = getbadchans(
        pool,
        qcube,
        ucube,
        clargs.qfitslist,
        clargs.ufitslist,
        totalbad=totalbad,
        cliplev=clargs.cliplev,
        update=update,
    )
    # print fitslist[np.asarray(badones,dtype=int)]

    q_msk, u_msk = blankchans(qcube, ucube, totalbad, blank=clargs.blank)

    if clargs.iterate is not None:
        print(f"Iterating {clargs.iterate} additional time(s)...")
        for i in range(clargs.iterate):
            totalbad = getbadchans(
                pool,
                q_msk,
                u_msk,
                clargs.qfitslist,
                clargs.ufitslist,
                totalbad=totalbad,
                cliplev=clargs.cliplev,
                update=update,
            )
            q_msk, u_msk = blankchans(q_msk, u_msk, totalbad, blank=clargs.blank)

    if clargs.blank:
        writefits(q_msk, u_msk, clargs)

    if clargs.file is not None:
        print(f"Saving bad files to {clargs.file}")
        np.savetxt(clargs.file, totalbad)


def cli():
    import argparse

    descStr = """
    Find bad channels by checking statistics of each channel image.

    """
    warnings.filterwarnings(
        "ignore",
        message="Cube is a Stokes cube, returning spectral cube for I component",
    )
    warnings.filterwarnings("ignore", message="Invalid value encountered in less")
    warnings.filterwarnings("ignore", message="Invalid value encountered in greater")
    warnings.filterwarnings("ignore", message="overflow encountered in square")

    # Parse the command line options
    ap = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("qfitslist", help="Wildcard list of Q fits files")
    ap.add_argument("ufitslist", help="Wildcard list of U fits files")
    ap.add_argument(
        "--blank",
        "-b",
        help="Blank bad channel maps? [default False]",
        default=False,
        action="store_true",
    )
    ap.add_argument(
        "--cliplev",
        "-c",
        help="Clip level in sigma, make this number lower to be more aggressive [default 5]",
        default=5.0,
        type=float,
    )

    ap.add_argument(
        "--iterate",
        "-i",
        help="Iterate flagging check N additional times [None -- one pass only]",
        default=None,
        type=int,
    )

    ap.add_argument(
        "--file",
        "-f",
        help="Filename to write bad channel indices to file [None --  do not write]",
        default=None,
        type=str,
    )

    group = ap.add_mutually_exclusive_group()

    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi", dest="mpi", default=False, action="store_true", help="Run with MPI."
    )

    args = ap.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    if args.mpi:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    main(pool, args)

    pool.close()


if __name__ == "__main__":
    cli()
