#!/usr/bin/env python
""" Convolve ASKAP images to common resolution """
__author__ = "Alec Thomson"

import logging
import sys
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from tqdm import tqdm

from racs_tools.convolve_uv import (
    get_convolving_beam,
    get_nyquist_beam,
    my_ceil,
    parse_conv_mode,
    round_up,
    smooth,
)
from racs_tools.logging import (
    init_worker,
    log_listener,
    log_queue,
    logger,
    set_verbosity,
)
from racs_tools.parallel import get_executor

# logger = setup_logger()


#############################################
#### ADAPTED FROM SCRIPT BY T. VERNSTROM ####
#############################################
class ImageData(NamedTuple):
    """Image data and metadata"""

    filename: Path
    """Filename of FITS file"""
    image: np.ndarray
    """Image data"""
    four_d: bool
    """Does the image have spectral and polarization axes?"""
    header: fits.Header
    """FITS header"""
    old_beam: Beam
    """Original beam"""
    nx: int
    """Number of pixels in x"""
    ny: int
    """Number of pixels in y"""
    dx: u.Quantity
    """Pixel size in x"""
    dy: u.Quantity
    """Pixel size in y"""


class BeamLogInfo(NamedTuple):
    """Beam log info"""

    filename: Path
    """Filename of FITS file"""
    old_beam: Beam
    """Original beam"""
    new_beam: Beam
    """Target beam"""
    conv_beam: Beam
    """Convolving beam"""


def check_target_beam(
    target_beam: Beam,
    all_beams: Beams,
    files: List[Path],
    cutoff: Optional[float] = None,
) -> bool:
    """Check that target beam will deconvolve

    Args:
        target_beam (Beam): Target beam.
        all_beams (Beams): All the beams to check.
        files (List[Path]): All the FITS files to check.
        cutoff (Optional[float], optional): Cutoff of beam in arcsec. Defaults to None.

    Raises:
        BeamError: If beam deconvolution fails.

    Returns:
        bool: If the target beam will deconvolve.
    """
    logger.info("Checking that target beam will deconvolve...")
    mask_count = 0
    failed = []
    for i, (beam, file) in enumerate(
        tqdm(
            zip(all_beams, files),
            total=len(all_beams),
            desc="Deconvolving",
            disable=(logger.level > logging.INFO),
        )
    ):
        if cutoff is not None and beam.major.to(u.arcsec) > cutoff * u.arcsec:
            continue
        try:
            target_beam.deconvolve(beam)
        except ValueError:
            mask_count += 1
            failed.append(file)
        except BeamError as be:
            # BeamError should not be raised if beams are equal
            if target_beam != beam:
                raise BeamError(be)
    is_good = mask_count == 0
    if not is_good:
        logger.error("The following images could not reach target resolution:")
        logger.error(failed)

    return is_good


def getimdata(cubenm: Path) -> ImageData:
    """Get image data from FITS file

    Args:
        cubenm (Path): File name of image.

    Returns:
        ImageData: Data and metadata.
    """
    logger.info(f"Getting image data from {cubenm}")
    with fits.open(cubenm, memmap=True, mode="denywrite") as hdu:
        header = hdu[0].header
        wcs = WCS(hdu[0])
        pixelscales = proj_plane_pixel_scales(wcs)

        dx = pixelscales[0] * u.deg
        dy = pixelscales[1] * u.deg

        if len(hdu[0].data.shape) == 4:
            # has spectral, polarization axes
            data = hdu[0].data[0, 0]
        else:
            data = hdu[0].data
        nx, ny = data.shape[-1], data.shape[-2]

        old_beam = Beam.from_fits_header(hdu[0].header)
        is_4d = len(hdu[0].data.shape) == 4
    return ImageData(
        filename=Path(cubenm),
        image=data,
        four_d=is_4d,
        header=header,
        old_beam=old_beam,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
    )


def savefile(
    newimage: np.ndarray,
    outfile: Path,
    header: fits.Header,
    new_beam: Beam,
) -> None:
    """Save smoothed image to FITS file

    Args:
        newimage (np.ndarray): Smoothed image.
        outfile (Path): File name.
        header (fits.Header): FITS header.
        new_beam (Beam): New beam.

    Raises:
        FileNotFoundError: If file is not saved.
    """
    logger.info(f"Saving to {outfile.absolute()}")
    beam = new_beam
    header = beam.attach_to_header(header)
    fits.writeto(
        outfile.absolute(), newimage.astype(np.float32), header=header, overwrite=True
    )

    if not outfile.exists():
        raise FileNotFoundError(f"File {outfile} not saved!")


def beamcon_2d_on_fits(
    file: Path,
    new_beam: Beam,
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"],
    suffix: str = "",
    prefix: str = "",
    outdir: Optional[Path] = None,
    cutoff: Optional[float] = None,
    dryrun: bool = False,
) -> BeamLogInfo:
    """Run beamcon_2d on a FITS file

    Args:
        file (Path): FITS file to smooth.
        new_beam (Beam): Target beam.
        conv_mode (str): Convolution mode.
        suffix (str, optional): Filename suffix. Defaults to "".
        prefix (str, optional): Filename prefix. Defaults to "".
        outdir (Optional[Path], optional): Ouput directory. Defaults to None (will be same as input).
        cutoff (Optional[float], optional): Cutoff for beamsize in arcsec. Defaults to None.
        dryrun (bool, optional): Don't save any images. Defaults to False.

    Returns:
        BeamLogInfo: Beamlog information.
    """
    logger.info(f"Working on {file}")

    outfile = Path(file.name)
    if suffix:
        outfile = outfile.with_suffix(f".{suffix}.fits")
    if prefix:
        outfile = Path(prefix + outfile.name)

    if outdir is not None:
        outdir = outdir.absolute()
    else:
        outdir = file.parent.absolute()

    image_data = getimdata(file)

    conv_beam, _ = get_convolving_beam(
        old_beam=image_data.old_beam,
        new_beam=new_beam,
        dx=image_data.dx,
        dy=image_data.dy,
        cutoff=cutoff,
    )

    if dryrun:
        return BeamLogInfo(
            filename=outfile,
            old_beam=image_data.old_beam,
            new_beam=new_beam,
            conv_beam=conv_beam,
        )

    new_image = smooth(
        image=image_data.image,
        old_beam=image_data.old_beam,
        new_beam=new_beam,
        dx=image_data.dx,
        dy=image_data.dy,
        conv_mode=conv_mode,
        cutoff=cutoff,
    )

    if image_data.four_d:
        # make it back into a 4D image
        new_image = np.expand_dims(np.expand_dims(new_image, axis=0), axis=0)

    savefile(
        newimage=new_image,
        outfile=outdir / outfile,
        header=image_data.header,
        new_beam=new_beam,
    )
    del new_image

    return BeamLogInfo(
        filename=outfile,
        old_beam=image_data.old_beam,
        new_beam=new_beam,
        conv_beam=conv_beam,
    )


def get_common_beam(
    files: List[Path],
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"] = "robust",
    target_beam: Optional[Beam] = None,
    cutoff: Optional[float] = None,
    tolerance: float = 0.0001,
    nsamps: float = 200,
    epsilon: float = 0.0005,
) -> Tuple[Beam, Beams]:
    """Get the smallest common beam.

    Args:
        files (List[Path]): FITS files to convolve.
        conv_mode (Literal["robust", "scipy", "astropy", "astropy_fft"], optional): _description_. Defaults to "robust".
        target_beam (Optional[Beam], optional): Target beam. Defaults to None.
        cutoff (Optional[float], optional): Cutoff for beamsize in arcse. Defaults to None.
        tolerance (float, optional): Radio beam tolerance. Defaults to 0.0001.
        nsamps (float, optional): Radio beam nsamps. Defaults to 200.
        epsilon (float, optional): Radio beam epsilon. Defaults to 0.0005.

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
        if np.all(flags):
            logger.critical(
                "All beams are larger than cutoff. All outputs will be blanked!"
            )
            nan_beam = Beam(np.nan * u.deg, np.nan * u.deg, np.nan * u.deg)
            nan_beams = Beams(
                [np.nan for beam in beams_list] * u.deg,
                [np.nan for beam in beams_list] * u.deg,
                [np.nan for beam in beams_list] * u.deg,
            )
            return nan_beam, nan_beams
    else:
        flags = np.array([False for beam in beams])

    if target_beam is None:
        # Find the common beam
        try:
            target_beam = beams[~flags].common_beam(
                tolerance=tolerance, epsilon=epsilon, nsamps=nsamps
            )
        except BeamError:
            logger.warning(
                "Couldn't find common beam with defaults\nTrying again with smaller tolerance"
            )
            target_beam = beams[~flags].common_beam(
                tolerance=tolerance * 0.1, epsilon=epsilon, nsamps=nsamps
            )

        # Round up values
        target_beam = Beam(
            major=my_ceil(target_beam.major.to(u.arcsec).value, precision=1) * u.arcsec,
            minor=my_ceil(target_beam.minor.to(u.arcsec).value, precision=1) * u.arcsec,
            pa=round_up(target_beam.pa.to(u.deg), decimals=2),
        )

    if conv_mode != "robust":
        target_beam = get_nyquist_beam(
            target_beam=target_beam, target_header=header, beams=beams
        )

    return target_beam, beams


def writelog(output: List[BeamLogInfo], commonbeam_log: Path):
    """Write beamlog file.

    Args:
        output (List[BeamLogInfo]): List of beamlog information.
        commonbeam_log (Path): Name of log file.
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
        [out["new_beam"].major.to(u.deg).value for out in output] * u.deg,
        name="Target BMAJ",
    )
    commonbeam_tab.add_column(
        [out["new_beam"].minor.to(u.deg).value for out in output] * u.deg,
        name="Target BMIN",
    )
    commonbeam_tab.add_column(
        [out["new_beam"].pa.to(u.deg).value for out in output] * u.deg,
        name="Target BPA",
    )
    # Convolving
    commonbeam_tab.add_column(
        [out["conv_beam"].major.to(u.deg).value for out in output] * u.deg,
        name="Convolving BMAJ",
    )
    commonbeam_tab.add_column(
        [out["conv_beam"].minor.to(u.deg).value for out in output] * u.deg,
        name="Convolving BMIN",
    )
    commonbeam_tab.add_column(
        [out["conv_beam"].pa.to(u.deg).value for out in output] * u.deg,
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
    logger.info(f"Convolving log written to {commonbeam_log}")


def smooth_fits_files(
    infile_list: List[Path] = [],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    outdir: Optional[Path] = None,
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"] = "robust",
    dryrun: bool = False,
    bmaj: Optional[float] = None,
    bmin: Optional[float] = None,
    bpa: Optional[float] = None,
    log: Optional[str] = None,
    circularise: bool = False,
    cutoff: Optional[float] = None,
    listfile: bool = False,
    tolerance: float = 0.0001,
    nsamps: int = 200,
    epsilon: float = 0.0005,
    ncores: Optional[int] = None,
    executor_type: Literal["thread", "process", "mpi"] = "thread",
    verbosity: int = 0,
) -> Beam:
    """Smooth a field of 2D images to a common resolution.

    Args:
        infile_list (List[Path], optional): List of FITS files to convolve. Defaults to [].
        prefix (Optional[str], optional): Output filename prefix. Defaults to None.
        suffix (Optional[str], optional): Output filename suffix. Defaults to None.
        outdir (Optional[Path], optional): Output directory. Defaults to None - same as input.
        conv_mode (Literal["robust", "scipy", "astropy", "astropy_fft"], optional): Convolution mode. Defaults to "robust".
        dryrun (bool, optional): Don't save any images. Defaults to False.
        bmaj (Optional[float], optional): Target beam major axis in arcsec. Defaults to None.
        bmin (Optional[float], optional): Target beam minor axis in arcsec. Defaults to None.
        bpa (Optional[float], optional): Target beam poistion angle in deg. Defaults to None.
        log (Optional[str], optional): Ouput logfile. Defaults to None.
        circularise (bool, optional): Set minor axis to same as major. Defaults to False.
        cutoff (Optional[float], optional): Cutoff for beamsize in arcse. Defaults to None.
        tolerance (float, optional): Radio beam tolerance. Defaults to 0.0001.
        nsamps (int, optional): Radio beam nsamp. Defaults to 200.
        epsilon (float, optional): Radio beam epsilon. Defaults to 0.0005.
        ncores (Optional[int], optional): Maximum number of cores to use. Defaults to None.
        executor_type (Literal["thread", "process", "mpi"], optional): Executor to use. Defaults to "thread".

    Raises:
        FileNotFoundError: If no files are found.
        ValueError: If target beam is not specified completely.
        BeamError: If target beam is too small.

    Returns:
        Beam: Common beam used.
    """
    # Required for multiprocessing logging
    log_listener.start()
    if dryrun:
        logger.info("Doing a dry run -- no files will be saved")

    # Check early as can fail
    Executor = get_executor(executor_type)

    # Get file list
    if listfile:
        assert len(infile_list) == 1, "Only one list file can be provided!"
        with open(infile_list[0]) as f:
            infile_list = [Path(line) for line in f.read().splitlines()]
    files = sorted(infile_list)
    if len(files) == 0:
        raise FileNotFoundError("No files found!")

    conv_mode = parse_conv_mode(conv_mode)

    nonetest = [param is None for param in (bmaj, bmin, bpa)]
    if all(nonetest):
        target_beam = None
    elif any(nonetest):
        raise ValueError("Please specify all target beam params!")
    else:
        target_beam = Beam(bmaj * u.arcsec, bmin * u.arcsec, bpa * u.deg)
        logger.info(f"Target beam is {target_beam!r}")

    # Find smallest common beam
    common_beam, all_beams = get_common_beam(
        files,
        conv_mode=conv_mode,
        target_beam=target_beam,
        cutoff=cutoff,
        tolerance=tolerance,
        nsamps=nsamps,
        epsilon=epsilon,
    )

    if target_beam is not None:
        if not check_target_beam(target_beam, all_beams, files, cutoff):
            raise BeamError("Please choose a larger target beam!")

        common_beam = target_beam

    if circularise:
        logger.info("Circular beam requested, setting BMIN=BMAJ and BPA=0")
        common_beam = Beam(
            major=common_beam.major,
            minor=common_beam.major,
            pa=0 * u.deg,
        )

    logger.info(f"Final beam is {common_beam!r}")
    with Executor(
        max_workers=ncores, initializer=init_worker, initargs=(log_queue, verbosity)
    ) as executor:
        futures = []
        for file in files:
            future = executor.submit(
                beamcon_2d_on_fits,
                file=file,
                outdir=outdir,
                new_beam=common_beam,
                conv_mode=conv_mode,
                suffix=suffix,
                prefix=prefix,
                cutoff=cutoff,
                dryrun=dryrun,
            )
            futures.append(future)

    beam_log_list = [future.result() for future in futures]
    if log is not None:
        writelog(beam_log_list, log)

    logger.info("Done!")
    log_listener.enqueue_sentinel()
    return common_beam


def cli():
    """Command-line interface"""
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
        description=descStr, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "infile",
        metavar="infile",
        type=Path,
        help="Input FITS image(s) to smooth (can be a wildcard) - beam info must be in header.",
        nargs="+",
    )

    parser.add_argument(
        "--listfile",
        action="store_true",
        help="Switch to assume `infile` is a text file list of images.",
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
        type=Path,
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
        Note this mode can now handle NaNs in the data.
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
        "--logfile",
        default=None,
        type=str,
        help="Save logging output to file",
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

    parser.add_argument(
        "--ncores",
        type=int,
        default=None,
        help="Number of cores to use for parallelisation. If None, use all available cores.",
    )

    parser.add_argument(
        "--executor",
        type=str,
        choices=["thread", "process", "mpi"],
        default="thread",
        help="Executor to use for parallelisation",
    )

    args = parser.parse_args()

    nonetest = [param is None for param in (args.bmaj, args.bmin, args.bpa)]
    if not all(nonetest) and any(nonetest):
        parser.error("Please specify all target beam params!")

    set_verbosity(
        logger=logger,
        verbosity=args.verbosity,
    )

    _ = smooth_fits_files(
        infile_list=args.infile,
        prefix=args.prefix,
        suffix=args.suffix,
        outdir=args.outdir,
        conv_mode=args.conv_mode,
        dryrun=args.dryrun,
        bmaj=args.bmaj,
        bmin=args.bmin,
        bpa=args.bpa,
        log=args.log,
        circularise=args.circularise,
        cutoff=args.cutoff,
        tolerance=args.tolerance,
        nsamps=args.nsamps,
        epsilon=args.epsilon,
        ncores=args.ncores,
        executor_type=args.executor,
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    sys.exit(cli())
