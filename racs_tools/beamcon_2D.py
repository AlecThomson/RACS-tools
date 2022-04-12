#!/usr/bin/env python
""" Convolve ASKAP images to common resolution """
__author__ = "Alec Thomson"

from functools import partial
from hashlib import new
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
from astropy import units as u
from astropy.io import fits, ascii
import astropy.wcs
from astropy.table import Table
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from racs_tools import au2
from racs_tools.convolve_uv import smooth
import schwimmbad
import psutil
from tqdm import tqdm
import logging as log

#############################################
#### ADAPTED FROM SCRIPT BY T. VERNSTROM ####
#############################################


def round_up(n: float, decimals: int = 0) -> float:
    """Round to number of decimals 

    Args:
        n (float): Number to round.
        decimals (int, optional): Number of decimals. Defaults to 0.

    Returns:
        float: Rounded number.
    """
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier


def my_ceil(a: float, precision: float = 0) -> float:
    """Modified ceil function to round up to precision

    Args:
        a (float): Number to round.
        precision (float, optional): Precision of rounding. Defaults to 0.

    Returns:
        float: Rounded number.
    """
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def getbeam(
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    cutoff: float = None,
) -> Tuple[Beam, float]:
    """Get the beam to use for smoothing

    Args:
        old_beam (Beam): Current beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x.
        dy (u.Quantity): Pixel size in y.
        cutoff (float, optional): Cutoff for beamsize in arcsec. Defaults to None.

    Raises:
        err: If beam deconvolution fails.

    Returns:
        Tuple[Beam, float]: Convolving beam and scaling factor.
    """
    log.info(f"Current beam is {old_beam!r}")

    if cutoff is not None and old_beam.major.to(u.arcsec) > cutoff * u.arcsec:
        return np.nan, np.nan

    if new_beam == old_beam:
        conbm = Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg,)
        fac = 1.0
        log.warning(
            f"New beam {new_beam!r} and old beam {old_beam!r} are the same. Won't attempt convolution."
        )
        return conbm, fac
    try:
        conbm = new_beam.deconvolve(old_beam)
    except Exception as err:
        log.warning(f"Could not deconvolve. New: {new_beam!r}, Old: {old_beam!r}")
        raise err
    fac, amp, outbmaj, outbmin, outbpa = au2.gauss_factor(
        beamConv=[
            conbm.major.to(u.arcsec).value,
            conbm.minor.to(u.arcsec).value,
            conbm.pa.to(u.deg).value,
        ],
        beamOrig=[
            old_beam.major.to(u.arcsec).value,
            old_beam.minor.to(u.arcsec).value,
            old_beam.pa.to(u.deg).value,
        ],
        dx1=dx.to(u.arcsec).value,
        dy1=dy.to(u.arcsec).value,
    )

    return conbm, fac


def getimdata(cubenm: str) -> dict:
    """Get image data from FITS file

    Args:
        cubenm (str): File name of image.

    Returns:
        dict: Data and metadata.
    """
    log.info(f"Getting image data from {cubenm}")
    with fits.open(cubenm, memmap=True, mode="denywrite") as hdu:

        w = astropy.wcs.WCS(hdu[0])
        pixelscales = astropy.wcs.utils.proj_plane_pixel_scales(w)

        dxas = pixelscales[0] * u.deg
        dyas = pixelscales[1] * u.deg

        if len(hdu[0].data.shape) == 4:
            # has spectral, polarization axes
            data = hdu[0].data[0, 0]
        else:
            data = hdu[0].data
        nx, ny = data.shape[-1], data.shape[-2]

        old_beam = Beam.from_fits_header(hdu[0].header)

        datadict = {
            "filename": os.path.basename(cubenm),
            "image": data,
            "4d": (len(hdu[0].data.shape) == 4),
            "header": hdu[0].header,
            "old_beam": old_beam,
            "nx": nx,
            "ny": ny,
            "dx": dxas,
            "dy": dyas,
        }
    return datadict



def savefile(
    newimage: np.ndarray,
    filename: str,
    header: fits.Header,
    final_beam: Beam,
    outdir: str = ".",
) -> None:
    """Save smoothed image to FITS file

    Args:
        newimage (np.ndarray): Smoothed image.
        filename (str): File name.
        header (fits.Header): FITS header.
        final_beam (Beam): New beam.
        outdir (str, optional): Output directory. Defaults to ".".
    """
    outfile = f"{outdir}/{filename}"
    log.info(f"Saving to {outfile}")
    beam = final_beam
    header = beam.attach_to_header(header)
    fits.writeto(outfile, newimage.astype(np.float32), header=header, overwrite=True)


def worker(
    file:str, 
    outdir:str, 
    new_beam:Beam, 
    conv_mode:str, 
    suffix:str="", 
    prefix:str="", 
    cutoff:float=None,
    dryrun:bool=False,
) -> dict:
    """Parallel worker function

    Args:
        file (str): FITS file to smooth.
        outdir (str): Output directory.
        new_beam (Beam): Target PSF.
        conv_mode (str): Convolving mode.
        suffix (str, optional): Filename suffix. Defaults to "".
        prefix (str, optional): Filename prefix. Defaults to "".
        cutoff (float, optional): PSF cutoff. Defaults to None.
        dryrun (bool, optional): Do a dryrun. Defaults to False.

    Returns:
        dict: Output data.
    """
    log.info(f"Working on {file}")

    if outdir is None:
        outdir = os.path.dirname(file)

    if outdir == "":
        outdir = "."

    outfile = os.path.basename(file)
    outfile = outfile.replace(".fits", f".{suffix}.fits")
    if prefix is not None:
        outfile = prefix + outfile
    datadict = getimdata(file)

    conbeam, sfactor = getbeam(
        old_beam=datadict["old_beam"], 
        new_beam=new_beam,
        dx=datadict["dx"],
        dy=datadict["dy"],
        cutoff=cutoff,
    )

    datadict.update({"conbeam": conbeam, "final_beam": new_beam, "sfactor": sfactor})
    if not dryrun:
        if (
            conbeam == Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg)
            and sfactor == 1
        ):
            newim = datadict["image"]
        else:
            newim = smooth(
                image=datadict["image"],
                old_beam=datadict["old_beam"],
                final_beam=datadict["final_beam"],
                dx=datadict["dx"],
                dy=datadict["dy"],
                sfactor=datadict["sfactor"],
                conbeam=datadict["conbeam"],
                conv_mode=conv_mode,
            )
        if datadict["4d"]:
            # make it back into a 4D image
            newim = np.expand_dims(np.expand_dims(newim, axis=0), axis=0)
        datadict.update(
            {"newimage": newim,}
        )
        savefile(
            newimage=datadict["newimage"],
            filename=outfile,
            header=datadict["header"],
            final_beam=datadict["final_beam"],
            outdir=outdir,
        )

    return datadict


def getmaxbeam(
    files: List[str],
    conv_mode: str = "robust",
    target_beam: Beam = None,
    cutoff: float = None,
    tolerance: float = 0.0001,
    nsamps: float = 200,
    epsilon: float = 0.0005,
) -> Tuple[Beam, Beams]:
    """Get the smallest common beam.

    Args:
        files (List[str]): List of FITS files.
        conv_mode (str, optional): Convolution mode. Defaults to "robust".
        target_beam (Beam, optional): Target PSF. Defaults to None.
        cutoff (float, optional): Cutoff PSF BMAJ in arcsec. Defaults to None.
        tolerance (float, optional): Common beam tolerance. Defaults to 0.0001.
        nsamps (float, optional): Common beam samples. Defaults to 200.
        epsilon (float, optional): Commonn beam epsilon. Defaults to 0.0005.

    Raises:
        Exception: X and Y pixel sizes are not the same.
        Exception: Convolving beam will be undersampled on pixel grid.

    Returns:
        Tuple[Beam, Beams]: Common beam and all beams.
    """
    beams_list = []
    for file in files:
        header = fits.getheader(file, memmap=True)
        beam = Beam.from_fits_header(header)
        beams_list.append(beam)

    beams = Beams(
        [beam.major.to(u.deg).value for beam in beams_list] * u.deg,
        [beam.minor.to(u.deg).value for beam in beams_list] * u.deg,
        [beam.pa.to(u.deg).value for beam in beams_list] * u.deg,
    )
    if cutoff is not None:
        flags = beams.major > cutoff * u.arcsec
    else:
        flags = np.array([False for beam in beams])
    try:
        cmn_beam = beams[~flags].common_beam(
            tolerance=tolerance, epsilon=epsilon, nsamps=nsamps
        )
    except BeamError:
        log.warning(
            "Couldn't find common beam with defaults\nTrying again with smaller tolerance"
        )
        cmn_beam = beams[~flags].common_beam(
            tolerance=tolerance * 0.1, epsilon=epsilon, nsamps=nsamps
        )

    # Round up values
    cmn_beam = Beam(
        major=my_ceil(cmn_beam.major.to(u.arcsec).value, precision=1) * u.arcsec,
        minor=my_ceil(cmn_beam.minor.to(u.arcsec).value, precision=1) * u.arcsec,
        pa=round_up(cmn_beam.pa.to(u.deg), decimals=2),
    )
    target_header = header
    w = astropy.wcs.WCS(target_header)
    pixelscales = astropy.wcs.utils.proj_plane_pixel_scales(w)

    dx = pixelscales[0] * u.deg
    dy = pixelscales[1] * u.deg
    if not dx == dy:
        raise Exception("GRID MUST BE SAME IN X AND Y")
    grid = dy
    if conv_mode != "robust":
        # Get the minor axis of the convolving beams
        minorcons = []
        for beam in beams:
            minorcons += [cmn_beam.deconvolve(beam).minor.to(u.arcsec).value]
        minorcons = np.array(minorcons) * u.arcsec
        samps = minorcons / grid.to(u.arcsec)
        # Check that convolving beam will be Nyquist sampled
        if any(samps.value < 2):
            # Set the convolving beam to be Nyquist sampled
            nyq_con_beam = Beam(major=grid * 2, minor=grid * 2, pa=0 * u.deg)
            # Find new target based on common beam * Nyquist beam
            # Not sure if this is best - but it works
            nyq_beam = cmn_beam.convolve(nyq_con_beam)
            nyq_beam = Beam(
                major=my_ceil(nyq_beam.major.to(u.arcsec).value, precision=1)
                * u.arcsec,
                minor=my_ceil(nyq_beam.minor.to(u.arcsec).value, precision=1)
                * u.arcsec,
                pa=round_up(nyq_beam.pa.to(u.deg), decimals=2),
            )
            log.info(f"Smallest common Nyquist sampled beam is: {nyq_beam!r}")
            if target_beam is not None:
                if target_beam < nyq_beam:
                    log.warning("TARGET BEAM WILL BE UNDERSAMPLED!")
                    raise Exception("CAN'T UNDERSAMPLE BEAM - EXITING")
            else:
                log.warning("COMMON BEAM WILL BE UNDERSAMPLED!")
                log.warning("SETTING COMMON BEAM TO NYQUIST BEAM")
                cmn_beam = nyq_beam

    return cmn_beam, beams


def writelog(output: List[Dict], commonbeam_log: str):
    """Write beamlog file.

    Args:
        output (List[Dict]): List of dictionaries containing output.
        commonbeam_log (str): Name of log file.
    """
    commonbeam_tab = Table()
    commonbeam_tab.add_column([out["filename"] for out in output], name="FileName")
    # Origina
    commonbeam_tab.add_column(
        [out["old_beam"].major.to(u.deg).value for out in output] * u.deg,
        name="Original BMAJ",
    )
    commonbeam_tab.add_column(
        [out["old_beam"].minor.to(u.deg).value for out in output] * u.deg,
        name="Original BMIN",
    )
    commonbeam_tab.add_column(
        [out["old_beam"].pa.to(u.deg).value for out in output] * u.deg,
        name="Original BPA",
    )
    # Target
    commonbeam_tab.add_column(
        [out["final_beam"].major.to(u.deg).value for out in output] * u.deg,
        name="Target BMAJ",
    )
    commonbeam_tab.add_column(
        [out["final_beam"].minor.to(u.deg).value for out in output] * u.deg,
        name="Target BMIN",
    )
    commonbeam_tab.add_column(
        [out["final_beam"].pa.to(u.deg).value for out in output] * u.deg,
        name="Target BPA",
    )
    # Convolving
    commonbeam_tab.add_column(
        [out["conbeam"].major.to(u.deg).value for out in output] * u.deg,
        name="Convolving BMAJ",
    )
    commonbeam_tab.add_column(
        [out["conbeam"].minor.to(u.deg).value for out in output] * u.deg,
        name="Convolving BMIN",
    )
    commonbeam_tab.add_column(
        [out["conbeam"].pa.to(u.deg).value for out in output] * u.deg,
        name="Convolving BPA",
    )

    # Write to log file
    units = ""
    for col in commonbeam_tab.colnames:
        unit = commonbeam_tab[col].unit
        unit = str(unit)
        units += unit + " "
    commonbeam_tab.meta["comments"] = [units]
    ascii.write(
        commonbeam_tab, output=commonbeam_log, format="commented_header", overwrite=True
    )
    log.info(f"Convolving log written to {commonbeam_log}")


def main(pool, args):
    """Main script.

    Args:
        pool (method): Multiprocessing or schwimmbad Pool.
        args (Namespace): Commandline args.

    Raises:
        Exception: If no files are found.
        Exception: If invalid convolution mode is specified.
        Exception: If partial target beam is specified.
        Exception: If target beam cannot be used.
    """    
    if args.dryrun:
        log.info("Doing a dry run -- no files will be saved")
    # Fix up outdir
    outdir = args.outdir
    if outdir is not None:
        if outdir[-1] == "/":
            outdir = outdir[:-1]
    else:
        outdir = None

    # Get file list
    files = sorted(args.infile)
    if files == []:
        raise Exception("No files found!")

    # Parse args

    conv_mode = args.conv_mode
    log.info(f"Convolution mode: {conv_mode}")
    if not conv_mode in ["robust", "scipy", "astropy", "astropy_fft"]:
        raise Exception("Please select valid convolution method!")

    log.info(f"Using convolution method {conv_mode}")
    if conv_mode == "robust":
        log.info("This is the most robust method. And fast!")
    elif conv_mode == "scipy":
        log.info("This fast, but not robust to NaNs or small PSF changes")
    else:
        log.info("This is slower, but robust to NaNs, but not to small PSF changes")

    bmaj = args.bmaj
    bmin = args.bmin
    bpa = args.bpa

    nonetest = [test is None for test in [bmaj, bmin, bpa]]

    if all(nonetest):
        target_beam = None

    elif not all(nonetest) and any(nonetest):
        raise Exception("Please specify all target beam params!")

    elif not all(nonetest) and not any(nonetest):
        target_beam = Beam(bmaj * u.arcsec, bmin * u.arcsec, bpa * u.deg)
        log.info(f"Target beam is {target_beam!r}")

    # Find smallest common beam
    big_beam, allbeams = getmaxbeam(
        files,
        conv_mode=conv_mode,
        target_beam=target_beam,
        cutoff=args.cutoff,
        tolerance=args.tolerance,
        nsamps=args.nsamps,
        epsilon=args.epsilon,
    )

    if target_beam is not None:
        log.info("Checking that target beam will deconvolve...")

        mask_count = 0
        failed = []
        for i, (beam, file) in enumerate(
            tqdm(
                zip(allbeams, files),
                total=len(allbeams),
                desc="Deconvolving",
                disable=(log.root.level > log.INFO),
            )
        ):
            try:
                target_beam.deconvolve(beam)
            except ValueError:
                mask_count += 1
                failed.append(file)
        if mask_count > 0:
            log.warning("The following images could not reach target resolution:")
            log.warning(failed)

            raise Exception("Please choose a larger target beam!")

        else:
            new_beam = target_beam

    else:
        new_beam = big_beam

    if args.circularise:
        log.info("Circular beam requested, setting BMIN=BMAJ and BPA=0")
        new_beam = Beam(major=new_beam.major, minor=new_beam.major, pa=0 * u.deg,)

    log.info(f"Final beam is {new_beam!r}")

    output = list(
        pool.map(
            partial(
                worker,
                outdir=outdir,
                new_beam=new_beam,
                conv_mode=conv_mode,
                suffix=args.suffix,
                prefix=args.prefix,
                cutoff=args.cutoff,
                dryrun=args.dryrun,
            ), 
            files
        )
    )

    if args.log is not None:
        writelog(output, args.log)

    log.info("Done!")


def cli():
    """Command-line interface
    """
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Smooth a field of 2D images to a common resolution.

    - Parallelisation can run using multiprocessing or MPI.

    - Default names of output files are /path/to/beamlog{infile//.fits/.{SUFFIX}.fits}

    - By default, the smallest common beam will be automatically computed.
    - Optionally, you can specify a target beam to use.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "infile",
        metavar="infile",
        type=str,
        help="Input FITS image(s) to smooth (can be a wildcard) - beam info must be in header.",
        nargs="+",
    )

    parser.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        type=str,
        default=None,
        help="Add prefix to output filenames.",
    )

    parser.add_argument(
        "-s",
        "--suffix",
        dest="suffix",
        type=str,
        default="sm",
        help="Add suffix to output filenames [sm].",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default=None,
        help="Output directory of smoothed FITS image(s) [same as input file].",
    )

    parser.add_argument(
        "--conv_mode",
        dest="conv_mode",
        type=str,
        choices=["robust", "scipy", "astropy", "astropy_fft"],
        default="robust",
        help="""Which method to use for convolution [robust].
        'robust' computes the analytic FT of the convolving Gaussian.
        Note this mode cannot handle NaNs in the data.
        Can also be 'scipy', 'astropy', or 'astropy_fft'.
        Note these other methods cannot cope well with small convolving beams.
        """,
    )

    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )

    parser.add_argument(
        "-d",
        "--dryrun",
        dest="dryrun",
        action="store_true",
        help="Compute common beam and stop [False].",
    )

    parser.add_argument(
        "--bmaj",
        dest="bmaj",
        type=float,
        default=None,
        help="Target BMAJ (arcsec) to convolve to [None].",
    )

    parser.add_argument(
        "--bmin",
        dest="bmin",
        type=float,
        default=None,
        help="Target BMIN (arcsec) to convolve to [None].",
    )

    parser.add_argument(
        "--bpa",
        dest="bpa",
        type=float,
        default=None,
        help="Target BPA (deg) to convolve to [None].",
    )

    parser.add_argument(
        "--log",
        dest="log",
        type=str,
        default=None,
        help="Name of beamlog file. If provided, save beamlog data to a file [None - not saved].",
    )

    parser.add_argument(
        "--logfile", default=None, type=str, help="Save logging output to file",
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        dest="cutoff",
        type=float,
        default=None,
        help="Cutoff BMAJ value (arcsec) -- Blank channels with BMAJ larger than this [None -- no limit]",
    )

    parser.add_argument(
        "--circularise",
        action="store_true",
        help="Circularise the final PSF -- Sets the BMIN = BMAJ, and BPA=0.",
    )

    parser.add_argument(
        "-t",
        "--tolerance",
        dest="tolerance",
        type=float,
        default=0.0001,
        help="tolerance for radio_beam.commonbeam.",
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        dest="epsilon",
        type=float,
        default=0.0005,
        help="epsilon for radio_beam.commonbeam.",
    )

    parser.add_argument(
        "-n",
        "--nsamps",
        dest="nsamps",
        type=int,
        default=200,
        help="nsamps for radio_beam.commonbeam.",
    )

    group = parser.add_mutually_exclusive_group()

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

    args = parser.parse_args()
    if args.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        myPE = comm.Get_rank()
    else:
        try:
            myPE = psutil.Process().cpu_num()
        except AttributeError:
            myPE = 0
    if args.verbosity == 1:
        log.basicConfig(
            filename=args.logfile,
            level=log.INFO,
            format=f"[{myPE}] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    elif args.verbosity >= 2:
        log.basicConfig(
            filename=args.logfile,
            level=log.DEBUG,
            format=f"[{myPE}] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    if args.mpi:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    main(pool, args)
    pool.close()


if __name__ == "__main__":
    cli()
