#!/usr/bin/env python
""" Convolve ASKAP cubes to common resolution """
__author__ = "Alec Thomson"

import logging
import os
from pathlib import Path
import stat
import sys
import warnings
from typing import Dict, List, Tuple, NamedTuple, Union

import numpy as np
import scipy.signal
from astropy import units as u
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from spectral_cube import SpectralCube
from spectral_cube.utils import SpectralCubeWarning
from tqdm import tqdm, trange

from racs_tools import au2
from racs_tools.beamcon_2D import my_ceil, round_up
from racs_tools.convolve_uv import smooth

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mpiSwitch = False
if (
    os.environ.get("OMPI_COMM_WORLD_SIZE") is not None
    or int(os.environ.get("SLURM_NTASKS") or 1) > 1
):
    mpiSwitch = True

if mpiSwitch:
    try:
        from mpi4py import MPI
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Script called with mpiexec/mpirun/srun, but mpi4py not installed"
        )
    # Get the processing environment
    comm = MPI.COMM_WORLD
    nPE = comm.Get_size()
    myPE = comm.Get_rank()
else:
    nPE = 1
    myPE = 0

warnings.filterwarnings(action="ignore", category=SpectralCubeWarning, append=True)
warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

#############################################
#### ADAPTED FROM SCRIPT BY T. VERNSTROM ####
#############################################

class CubeData(NamedTuple):
    """Cube data structure"""

    filename: Path
    outdir: Path
    dx: u.Quantity
    dy: u.Quantity
    beamlog: Path
    beam: Beam
    nchan: int


class MaskCubeData(NamedTuple):
    """Masked cube data"""

    mask: np.ndarray
    beams: Beams


class BeamLogData(NamedTuple):
    """Beam log data"""

    facs: np.ndarray
    convbeams: Beams
    commonbeams: Beams
    commonbeam_log: Path


class Error(OSError):
    pass


class SameFileError(Error):
    """Raised when source and destination are the same file."""


class SpecialFileError(OSError):
    """Raised when trying to do a kind of operation (e.g. copying) which is
    not supported on a special file (e.g. a named pipe)"""


class ExecError(OSError):
    """Raised when a command could not be executed"""


class ReadError(OSError):
    """Raised when an archive cannot be read"""


class RegistryError(Exception):
    """Raised when a registry operation with the archiving
    and unpacking registeries fails"""


def _samefile(src, dst):
    # Macintosh, Unix.
    if hasattr(os.path, "samefile"):
        try:
            return os.path.samefile(src, dst)
        except OSError:
            return False


def copyfile(src, dst, *, follow_symlinks=True):
    """Copy data from src to dst.

    If follow_symlinks is not set and src is a symbolic link, a new
    symlink will be created instead of copying the file it points to.

    """
    if _samefile(src, dst):
        raise SameFileError(f"{src!r} and {dst!r} are the same file")

    for fn in [src, dst]:
        try:
            st = os.stat(fn)
        except OSError:
            # File most likely does not exist
            pass
        else:
            # XXX What about other special files? (sockets, devices...)
            if stat.S_ISFIFO(st.st_mode):
                raise SpecialFileError("`%s` is a named pipe" % fn)

    if not follow_symlinks and os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        with open(src, "rb") as fsrc:
            with open(dst, "wb") as fdst:
                copyfileobj(fsrc, fdst)
    return dst


def copyfileobj(fsrc, fdst, length=16 * 1024):
    # copied = 0
    total = os.fstat(fsrc.fileno()).st_size
    with tqdm(
        total=total,
        disable=(logger.root.level > logging.INFO),
        unit_scale=True,
        desc="Copying file",
    ) as pbar:
        while True:
            buf = fsrc.read(length)
            if not buf:
                break
            fdst.write(buf)
            copied = len(buf)
            pbar.update(copied)


def getbeams(file: Path, header: fits.Header) -> Tuple[Table, int, Path]:
    """Get beam information from a fits file or beamlog.

    Args:
        file (str): FITS filename.
        header (fits.Header): FITS header.

    Returns:
        Tuple[Table, int, str]: Table of beams, number of beams, and beamlog filename.
    """
    # Add beamlog info to dict just in case
    dirname = file.parent
    basename = file.name
    beamlog = (dirname / f"beamlog.{basename}").with_suffix(".txt")

    # First check for CASA beams
    try:
        headcheck = header["CASAMBM"]
    except KeyError:
        headcheck = False
    if headcheck:
        logger.info(
            "CASA beamtable found in header - will use this table for beam calculations"
        )
        with fits.open(file) as hdul:
            hdu = hdul.pop("BEAMS")
            beams = Table.read(hdu)

    # Otherwise use beamlog file
    else:
        logger.info("No CASA beamtable found in header - looking for beamlogs")
        logger.info(f"Getting beams from {beamlog}")

        beams = Table.read(beamlog, format="ascii.commented_header")
        # Header looks like:
        # colnames=['Channel', 'BMAJarcsec', 'BMINarcsec', 'BPAdeg']
        # But needs some fix up - astropy can't read the header properly
        for col in beams.colnames:
            idx = col.find("[")
            if idx == -1:
                new_col = col
                unit = u.Unit("")
            else:
                new_col = col[:idx]
                unit = u.Unit(col[idx + 1 : -1])
            beams[col].unit = unit
            beams[col].name = new_col
    nchan = len(beams)
    return beams, nchan, beamlog


def getfacs(
    beams: Beams, convbeams: Beams, dx: u.Quantity, dy: u.Quantity
) -> np.ndarray:
    """Get the conversion factors for each beam.

    Args:
        beams (Beams): Old beams.
        convbeams (Beams): Convolving beams.
        dx (u.Quantity): Pixel size in x direction.
        dy (u.Quantity): Pixel size in y direction.

    Returns:
        np.ndarray: Conversion factors.
    """
    facs_list = []
    for conbm, old_beam in zip(convbeams, beams):
        if conbm == Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg):
            fac = 1.0
        else:
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
        facs_list.append(fac)
    facs = np.array(facs_list)
    return facs


def cpu_to_use(max_cpu: int, count: int) -> int:
    """Find number of cpus to use.
    Find the right number of cpus to use when dividing up a task, such
    that there are no remainders.
    Args:
        max_cpu (int): Maximum number of cores to use for a process.
        count (int): Number of tasks.

    Returns:
        int: Maximum number of cores to be used that divides into the number
        of tasks.
    """
    factors_list = []
    for i in range(1, count + 1):
        if count % i == 0:
            factors_list.append(i)
    factors = np.array(factors_list)
    return max(factors[factors <= max_cpu])


def worker(
    filename: str,
    idx: int,
    **kwargs,
) -> np.ndarray:
    """Smooth worker function.

    Extracts a single image from a FITS cube and smooths it.

    Args:
        filename (str): FITS cube filename.
        idx (int): Channel index.

    Kwargs:
        Passed to :func:`smooth`.

    Returns:
        np.ndarray: Smoothed channel image.
    """
    cube = SpectralCube.read(filename)
    # Get spectral axis
    wcs = cube.wcs
    axis_type_dict = wcs.get_axis_types()[::-1]  # Reverse order for fits
    axis_names = [i["coordinate_type"] for i in axis_type_dict]
    spec_idx = axis_names.index("spectral")
    slicer = [slice(None)] * len(cube.unmasked_data.shape)
    slicer[spec_idx] = idx
    slicer = tuple(slicer)

    plane = cube.unmasked_data[slicer].value.astype(np.float32)
    logger.debug(f"Size of plane is {(plane.nbytes*u.byte).to(u.MB)}")
    newim = smooth(image=plane, **kwargs)
    return newim


def makedata(files: List[Path], outdirs: List[Path]) -> List[CubeData]:
    """Create data dictionary.

    Args:
        files (List[str]): List of filenames.
        outdir (str): Output directory.

    Raises:
        Exception: If pixel grid in X and Y is not the same.

    Returns:
        Dict[Dict]: Data and metadata for each channel and image.
    """
    datalist = []  # type: List[CubeData]
    for i, (file, out) in enumerate(zip(files, outdirs)):
        # Get metadata
        header = fits.getheader(file)
        w = WCS(header)
        pixelscales = proj_plane_pixel_scales(w)

        dxas = pixelscales[0] * u.deg
        dyas = pixelscales[1] * u.deg

        if not dxas == dyas:
            raise Exception("GRID MUST BE SAME IN X AND Y")
        # Get beam info
        beam, nchan, beamlog = getbeams(file=file, header=header)

        cube_data = CubeData(
            filename=file,
            outdir=out,
            dx=dxas,
            dy=dyas,
            beamlog=beamlog,
            beam=beam,
            nchan=nchan,
        )
        datalist.append(cube_data)

    return datalist


def natural_commonbeamer(
    datalist_masked: List[MaskCubeData],
    grids: List[u.Quantity],
    nchans: int,
    conv_mode: str = "robust",
    tolerance: float = 0.0001,
    nsamps: int = 200,
    epsilon: float = 0.0005,
) -> Beams:
    """Find common beam for per channel.

    Args:
        datalist_masked (List[MaskCubeData]): List of masked data.
        grids (List[u.Quantity]): List of pixel grids.
        nchans (int): Number of channels.
        conv_mode (str, optional): Convolution mode. Defaults to "robust".
        tolerance (float, optional): Tolerance for convergence. Defaults to 0.0001.
        nsamps (int, optional): Number of samples. Defaults to 200.
        epsilon (float, optional): Epsilon. Defaults to 0.0005.

    Returns:
        Beams: Common beams.
    """
    ### Natural mode ###
    big_beams = []
    for n in trange(
        nchans,
        desc="Constructing beams",
        disable=(logger.root.level > logging.INFO),
    ):
        majors_list = []
        minors_list = []
        pas_list = []
        for d in datalist_masked:
            major = d.beams[n].major
            minor = d.beams[n].minor
            pa = d.beams[n].pa
            if d.mask[n]:
                major *= np.nan
                minor *= np.nan
                pa *= np.nan
            majors_list.append(major.to(u.arcsec).value)
            minors_list.append(minor.to(u.arcsec).value)
            pas_list.append(pa.to(u.deg).value)

        majors = np.array(majors_list)
        minors = np.array(minors_list)
        pas = np.array(pas_list)

        majors *= u.arcsec
        minors *= u.arcsec
        pas *= u.deg
        big_beams.append(Beams(major=majors, minor=minors, pa=pas))

    # Find common beams
    bmaj_common = []
    bmin_common = []
    bpa_common = []
    for i, beams in tqdm(
        enumerate(big_beams),
        desc="Finding common beam per channel",
        disable=(logger.root.level > logging.INFO),
        total=nchans,
    ):
        if all(np.isnan(beams)):
            commonbeam = Beam(
                major=np.nan * u.deg, minor=np.nan * u.deg, pa=np.nan * u.deg
            )
        else:
            try:
                commonbeam = beams[~np.isnan(beams)].common_beam(
                    tolerance=tolerance,
                    nsamps=nsamps,
                    epsilon=epsilon,
                )
            except BeamError:
                logger.warn("Couldn't find common beam with defaults")
                logger.warn("Trying again with smaller tolerance")

                commonbeam = beams[~np.isnan(beams)].common_beam(
                    tolerance=tolerance * 0.1,
                    nsamps=nsamps,
                    epsilon=epsilon,
                )
            # Round up values
            commonbeam = Beam(
                major=my_ceil(commonbeam.major.to(u.arcsec).value, precision=1)
                * u.arcsec,
                minor=my_ceil(commonbeam.minor.to(u.arcsec).value, precision=1)
                * u.arcsec,
                pa=round_up(commonbeam.pa.to(u.deg), decimals=2),
            )

            grid = grids[i]
            if conv_mode != "robust":
                # Get the minor axis of the convolving beams
                minorcons = []
                for beam in beams[~np.isnan(beams)]:
                    minorcons += [
                        commonbeam.deconvolve(beam).minor.to(u.arcsec).value
                    ]
                minorcons = np.array(minorcons) * u.arcsec
                samps = minorcons / grid.to(u.arcsec)
                # Check that convolving beam will be Nyquist sampled
                if any(samps.value < 2):
                    # Set the convolving beam to be Nyquist sampled
                    nyq_con_beam = Beam(
                        major=grid * 2, minor=grid * 2, pa=0 * u.deg
                    )
                    # Find new target based on common beam * Nyquist beam
                    # Not sure if this is best - but it works
                    nyq_beam = commonbeam.convolve(nyq_con_beam)
                    nyq_beam = Beam(
                        major=my_ceil(
                            nyq_beam.major.to(u.arcsec).value, precision=1
                        )
                        * u.arcsec,
                        minor=my_ceil(
                            nyq_beam.minor.to(u.arcsec).value, precision=1
                        )
                        * u.arcsec,
                        pa=round_up(nyq_beam.pa.to(u.deg), decimals=2),
                    )
                    logger.info(
                        f"Smallest common Nyquist sampled beam is: {nyq_beam!r}"
                    )

                    logger.warn("COMMON BEAM WILL BE UNDERSAMPLED!")
                    logger.warn("SETTING COMMON BEAM TO NYQUIST BEAM")
                    commonbeam = nyq_beam

        bmaj_common.append(commonbeam.major.to(u.arcsec).value)
        bmin_common.append(commonbeam.minor.to(u.arcsec).value)
        bpa_common.append(commonbeam.pa.to(u.deg).value)

    bmaj_common *= u.arcsec
    bmin_common *= u.arcsec
    bpa_common *= u.deg

    # Make Beams object
    commonbeams = Beams(major=bmaj_common, minor=bmin_common, pa=bpa_common)

    return commonbeams

def total_commonbeamer(
    datalist_masked: List[MaskCubeData],
    grids: List[u.Quantity],
    nchans: int,
    target_beam: Union[Beam, None] = None,
    conv_mode: str = "robust",
    tolerance: float = 0.0001,
    nsamps: int = 200,
    epsilon: float = 0.0005,
):
    majors_list = []
    minors_list = []
    pas_list = []
    for d in datalist_masked:
        major = d.beams.major
        minor = d.beams.minor
        pa = d.beams.pa
        major[d.mask] *= np.nan
        minor[d.mask] *= np.nan
        pa[d.mask] *= np.nan
        majors_list.append(major.to(u.arcsec).value)
        minors_list.append(minor.to(u.arcsec).value)
        pas_list.append(pa.to(u.deg).value)

    majors = np.array(majors_list).ravel()
    minors = np.array(minors_list).ravel()
    pas = np.array(pas_list).ravel()

    majors *= u.arcsec
    minors *= u.arcsec
    pas *= u.deg
    big_beams = Beams(major=majors, minor=minors, pa=pas)

    logger.info("Finding common beam across all channels")
    logger.info("This may take some time...")

    try:
        commonbeam = big_beams[~np.isnan(big_beams)].common_beam(
            tolerance=tolerance, nsamps=nsamps, epsilon=epsilon
        )
    except BeamError:
        logger.warn("Couldn't find common beam with defaults")
        logger.warn("Trying again with smaller tolerance")

        commonbeam = big_beams[~np.isnan(big_beams)].common_beam(
            tolerance=tolerance * 0.1, nsamps=nsamps, epsilon=epsilon
        )
    if target_beam:
        commonbeam = target_beam
    else:
        # Round up values
        commonbeam = Beam(
            major=my_ceil(commonbeam.major.to(u.arcsec).value, precision=1)
            * u.arcsec,
            minor=my_ceil(commonbeam.minor.to(u.arcsec).value, precision=1)
            * u.arcsec,
            pa=round_up(commonbeam.pa.to(u.deg), decimals=2),
        )
    if conv_mode != "robust":
        # Get the minor axis of the convolving beams
        grid = grids[0]
        minorcons = []
        for beam in big_beams[~np.isnan(big_beams)]:
            minorcons += [commonbeam.deconvolve(beam).minor.to(u.arcsec).value]
        minorcons = np.array(minorcons) * u.arcsec
        samps = minorcons / grid.to(u.arcsec)
        # Check that convolving beam will be Nyquist sampled
        if any(samps.value < 2):
            # Set the convolving beam to be Nyquist sampled
            nyq_con_beam = Beam(major=grid * 2, minor=grid * 2, pa=0 * u.deg)
            # Find new target based on common beam * Nyquist beam
            # Not sure if this is best - but it works
            nyq_beam = commonbeam.convolve(nyq_con_beam)
            nyq_beam = Beam(
                major=my_ceil(nyq_beam.major.to(u.arcsec).value, precision=1)
                * u.arcsec,
                minor=my_ceil(nyq_beam.minor.to(u.arcsec).value, precision=1)
                * u.arcsec,
                pa=round_up(nyq_beam.pa.to(u.deg), decimals=2),
            )
            logger.info(f"Smallest common Nyquist sampled beam is: {nyq_beam!r}")
            if target_beam is not None:
                commonbeam = target_beam
                if target_beam < nyq_beam:
                    logger.warn("TARGET BEAM WILL BE UNDERSAMPLED!")
                    raise Exception("CAN'T UNDERSAMPLE BEAM - EXITING")
            else:
                logger.warn("COMMON BEAM WILL BE UNDERSAMPLED!")
                logger.warn("SETTING COMMON BEAM TO NYQUIST BEAM")
                commonbeam = nyq_beam

    # Make Beams object
    commonbeams = Beams(
        major=[commonbeam.major] * nchans * commonbeam.major.unit,
        minor=[commonbeam.minor] * nchans * commonbeam.minor.unit,
        pa=[commonbeam.pa] * nchans * commonbeam.pa.unit,
    )
    return commonbeams


def make_beamlogs(
    commonbeams: Beams,
    datalist: List[CubeData],
    datalist_masked: List[MaskCubeData],
    suffix: str,
    nchans: int,
    circularise: bool = False,
) -> List[BeamLogData]:
    if circularise:
        logger.info("Circular beam requested, setting BMIN=BMAJ and BPA=0")
        commonbeams = Beams(
            major=commonbeams.major,
            minor=commonbeams.major,
            pa=commonbeams.pa * 0,
        )

    logger.info("Final beams are:")
    for i, commonbeam in enumerate(commonbeams):
        logger.info(f"Channel {i}: {commonbeam!r}")

    beamlog_data_list = [] # type: List[BeamLogData]
    for dm, d in tqdm(
        zip(
            datalist_masked,
            datalist,
        ),
        desc="Getting convolution data",
        disable=(logger.root.level > logging.INFO),
    ):
        # Get convolving beams
        conv_bmaj = []
        conv_bmin = []
        conv_bpa = []
        old_beams = dm.beams
        masks = dm.mask
        for commonbeam, old_beam, mask in zip(commonbeams, old_beams, masks):
            if mask:
                convbeam = Beam(
                    major=np.nan * u.deg, minor=np.nan * u.deg, pa=np.nan * u.deg
                )
            else:
                old_beam_check = Beam(
                    major=my_ceil(old_beam.major.to(u.arcsec).value, precision=1)
                    * u.arcsec,
                    minor=my_ceil(old_beam.minor.to(u.arcsec).value, precision=1)
                    * u.arcsec,
                    pa=round_up(old_beam.pa.to(u.deg), decimals=2),
                )
                if commonbeam == old_beam_check:
                    convbeam = Beam(
                        major=0 * u.deg,
                        minor=0 * u.deg,
                        pa=0 * u.deg,
                    )
                    logger.warn(
                        f"New beam {commonbeam!r} and old beam {old_beam_check!r} are the same. Won't attempt convolution."
                    )

                else:
                    convbeam = commonbeam.deconvolve(old_beam)

            conv_bmaj.append(convbeam.major.to(u.arcsec).value)
            conv_bmin.append(convbeam.minor.to(u.arcsec).value)
            conv_bpa.append(convbeam.pa.to(u.deg).value)

        conv_bmaj *= u.arcsec
        conv_bmin *= u.arcsec
        conv_bpa *= u.deg

        # Construct beams object
        convbeams = Beams(major=conv_bmaj, minor=conv_bmin, pa=conv_bpa)

        # Get gaussian beam factors
        facs = getfacs(
            beams=dm.beams,
            convbeams=convbeams,
            dx=d.dx,
            dy=d.dy,
        )

        # Setup conv beamlog
        commonbeam_log = d.beamlog.with_suffix(f".{suffix}.txt")

        commonbeam_tab = Table()
        # Save target
        commonbeam_tab.add_column(np.arange(nchans), name="Channel")
        commonbeam_tab.add_column(commonbeams.major, name="Target BMAJ")
        commonbeam_tab.add_column(commonbeams.minor, name="Target BMIN")
        commonbeam_tab.add_column(commonbeams.pa, name="Target BPA")
        # Save convolving beams
        commonbeam_tab.add_column(convbeams.major, name="Convolving BMAJ")
        commonbeam_tab.add_column(convbeams.minor, name="Convolving BMIN")
        commonbeam_tab.add_column(convbeams.pa, name="Convolving BPA")
        # Save facs
        commonbeam_tab.add_column(facs, name="Convolving factor")

        # Write to log file
        units = ""
        for col in commonbeam_tab.colnames:
            unit = commonbeam_tab[col].unit
            unit = str(unit)
            units += unit + " "
        commonbeam_tab.meta["comments"] = [units]
        ascii.write(
            commonbeam_tab,
            output=commonbeam_log,
            format="commented_header",
            overwrite=True,
        )
        logger.info(f"Convolving log written to {commonbeam_log}")

        beamlog_data = BeamLogData(
            facs=facs,
            convbeams=convbeams,
            commonbeams=commonbeams,
            commonbeam_log=commonbeam_log,
        )
        beamlog_data_list.append(beamlog_data)

    return beamlog_data_list


def masking(datalist_mask: List[MaskCubeData], cutoff: Union[u.Quantity, None] = None) -> List[MaskCubeData]:
    """Apply masking to data.

    Args:
        nchans (int): Number of channels in cubes.
        datadict (dict): Data dictionary.
        cutoff (None, optional): Cutoff BMAJ size for masking. Defaults to None.

    Returns:
        dict: Updated data dictionary.
    """
    if cutoff:
        for data_mask in datalist_mask:
            cutmask = data_mask.beams.major > cutoff
            data_mask.mask += cutmask

    # Check for pipeline masking
    nullbeam = Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg)
    tiny = np.finfo(np.float32).tiny  # Smallest positive number - used to mask
    smallbeam = Beam(major=tiny * u.deg, minor=tiny * u.deg, pa=tiny * u.deg)

    for data_mask in datalist_mask:
        nullmask = np.logical_or(
            data_mask.beams == nullbeam,
            data_mask.beams == smallbeam,
        )
        data_mask.mask += nullmask
    return datalist_mask


def initfiles(
    filename: Path,
    commonbeams: Beams,
    outdir: Path,
    mode: str,
    suffix=None,
    prefix=None,
    ref_chan=None,
) -> Path:
    """Initialise output files

    Args:
        datadict (dict): Main data dict - indexed
        mode (str): 'total' or 'natural'

    Returns:
        datadict: Updated datadict
    """
    logger.debug(f"Reading {filename}")
    with fits.open(filename, memmap=True, mode="denywrite") as hdulist:
        primary_hdu = hdulist[0]
        data = primary_hdu.data
        header = primary_hdu.header
        wcs = WCS(header)

    ## Header
    spec_axis = wcs.spectral
    crpix = int(spec_axis.wcs.crpix)
    nchans = spec_axis.array_shape[0]
    assert nchans == len(
        commonbeams
    ), "Number of channels in header and commonbeams do not match"
    chans = np.arange(nchans)
    if ref_chan is None:
        # Get reference channel, and attach PSF there
        crindex = crpix - 1  # For python!
    elif ref_chan == "first":
        crindex = 0
    elif ref_chan == "last":
        crindex = -1
    elif ref_chan == "mid":
        # Locate mid Channel
        # In python's 0-based index, the following will set the mid Channel to
        # the upper-mid value for even number of channels. For odd-number of
        # channels, the mid value is unique.
        crindex = nchans // 2
    ref_psf = commonbeams[crindex]
    # Check the Stokes
    stokes_axis = wcs.sub(["stokes"])
    if stokes_axis.array_shape == ():
        raise ValueError("No Stokes axis found")
    nstokes = stokes_axis.array_shape[0]
    if nstokes > 1:
        logger.critical(
            f"More than one Stokes parameter in header. Only the first one will be used."
        )
    pols = np.zeros_like(chans)  # Zeros because we take the first one
    if any(
        (
            np.isnan(ref_psf.major.value),
            np.isnan(ref_psf.minor.value),
            np.isnan(ref_psf.pa.value),
        )
    ):
        logger.warning("Reference PSF is NaN - replacing with 0 in the header")
        ref_psf = Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg)
        header["COMMENT"] = "Reference PSF is NaN"
        header["COMMENT"] = "- This is likely because the reference channel is masked."
        header["COMMENT"] = "- It has been replaced with 0 to keep FITS happy."
    header = ref_psf.attach_to_header(header)
    primary_hdu = fits.PrimaryHDU(data=data, header=header)
    if mode == "natural":
        # Make a CASA beamtable
        header["CASAMBM"] = True
        header["COMMENT"] = "The PSF in each image plane varies."
        header[
            "COMMENT"
        ] = "Full beam information is stored in the second FITS extension."
        tiny = np.finfo(np.float32).tiny
        beam_table = Table(
            data=[
                # Replace NaNs with np.finfo(np.float32).tiny - this is the smallest
                # positive number that can be represented in float32
                # We use this to keep CASA happy
                np.nan_to_num(commonbeams.major.to(u.arcsec), nan=tiny * u.arcsec),
                np.nan_to_num(commonbeams.minor.to(u.arcsec), nan=tiny * u.arcsec),
                np.nan_to_num(commonbeams.pa.to(u.deg), nan=tiny * u.deg),
                chans,
                pols,
            ],
            names=["BMAJ", "BMIN", "BPA", "CHAN", "POL"],
            dtype=["f4", "f4", "f4", "i4", "i4"],
        )
        header["COMMENT"] = f"The value '{tiny}' repsenents a NaN PSF in the beamtable."
        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        tab_hdu = fits.table_to_hdu(beam_table)
        tab_header = tab_hdu.header
        tab_header["EXTNAME"] = "BEAMS"
        tab_header["NCHAN"] = nchans
        tab_header["NPOL"] = 1  # Only one pol for now
        new_hdulist = fits.HDUList([primary_hdu, tab_hdu])

    elif mode == "total":
        new_hdulist = fits.HDUList([primary_hdu])

    # Set up output file
    if not suffix:
        suffix = mode
    outname = filename.with_suffix(f".{suffix}.fits")
    if prefix:
        outname = Path(prefix + outname.as_posix())

    outfile = outdir / outname
    logger.info(f"Initialising to {outfile}")
    new_hdulist.writeto(outfile, overwrite=True)

    return outfile


def readlogs(commonbeam_log: Path) -> BeamLogData:
    """Read convolving log files

    Args:
        commonbeam_log (str): Filename of the common beam log

    Raises:
        Exception: If the log file is not found

    Returns:
        BeamLogData: Common beams, convolving beams, and scaling factors
    """
    logger.info(f"Reading from {commonbeam_log}")
    try:
        commonbeam_tab = Table.read(commonbeam_log, format="ascii.commented_header")
    except FileNotFoundError:
        raise Exception("beamlogConvolve must be co-located with image")
    # Convert to Beams
    commonbeams = Beams(
        major=commonbeam_tab["Target BMAJ"] * u.arcsec,
        minor=commonbeam_tab["Target BMIN"] * u.arcsec,
        pa=commonbeam_tab["Target BPA"] * u.deg,
    )
    convbeams = Beams(
        major=commonbeam_tab["Convolving BMAJ"] * u.arcsec,
        minor=commonbeam_tab["Convolving BMIN"] * u.arcsec,
        pa=commonbeam_tab["Convolving BPA"] * u.deg,
    )
    facs = np.array(commonbeam_tab["Convolving factor"])
    logger.info("Final beams are:")
    for i, commonbeam in enumerate(commonbeams):
        logger.info(f"Channel {i}: {commonbeam!r}")

    return BeamLogData(
        facs=facs,
        convbeams=convbeams,
        commonbeams=commonbeams,
        commonbeam_log=commonbeam_log,
    )


def main(
    infile: list,
    uselogs: bool = False,
    mode: str = "natural",
    conv_mode: str = "robust",
    dryrun: bool = False,
    prefix: str = None,
    suffix: str = None,
    outdir: str = None,
    bmaj: float = None,
    bmin: float = None,
    bpa: float = None,
    cutoff: float = None,
    circularise: bool = False,
    ref_chan: int = None,
    tolerance: float = 0.0001,
    epsilon: float = 0.0005,
    nsamps: int = 200,
):
    """main script

    Args:
        args (args): Command line args

    """

    if myPE == 0:
        logger.info(f"Total number of MPI ranks = {nPE}")
        # Parse args
        if dryrun:
            logger.info("Doing a dry run -- no files will be saved")

        # Check mode
        logger.info(f"Mode is {mode}")
        if mode == "natural" and mode == "total":
            raise Exception("'mode' must be 'natural' or 'total'")
        if mode == "natural":
            logger.info("Smoothing each channel to a common resolution")
        if mode == "total":
            logger.info("Smoothing all channels to a common resolution")

        # Check cutoff
        if cutoff:
            cutoff *= u.arcsec
            logger.info(f"Cutoff is: {cutoff}")

        # Check target
        logger.debug(conv_mode)
        if (
            not conv_mode == "robust"
            and not conv_mode == "scipy"
            and not conv_mode == "astropy"
            and not conv_mode == "astropy_fft"
        ):
            raise Exception("Please select valid convolution method!")

        logger.info(f"Using convolution method {conv_mode}")
        if conv_mode == "robust":
            logger.info("This is the most robust method. And fast!")
        elif conv_mode == "scipy":
            logger.info("This fast, but not robust to NaNs or small PSF changes")
        else:
            logger.info(
                "This is slower, but robust to NaNs, but not to small PSF changes"
            )

        nonetest = [test is None for test in [bmaj, bmin, bpa]]

        if not all(nonetest) and mode != "total":
            raise Exception("Only specify a target beam in 'total' mode")

        if all(nonetest):
            target_beam = None

        elif not all(nonetest) and any(nonetest):
            raise Exception("Please specify all target beam params!")

        elif not all(nonetest) and not any(nonetest):
            target_beam = Beam(bmaj * u.arcsec, bmin * u.arcsec, bpa * u.deg)
            logger.info(f"Target beam is {target_beam!r}")

        files = sorted(infile)
        if files == []:
            raise Exception("No files found!")

        if outdir:
            outdir = Path(outdir)
            outdirs = [outdir] * len(files)
        else:
            outdirs = []
            for f in files:
                out = Path(f)
                outdirs += [out]

        datalist = makedata(files, outdirs)

        # Sanity check channel counts
        nchans = np.array([d.nchan for d in datalist])
        check = all(nchans == nchans[0])

        if not check:
            raise Exception("Unequal number of spectral channels!")

        else:
            nchans = nchans[0]

        # Check suffix
        if not suffix:
            suffix = mode

        # Construct Beams objects
        datalist_mask = [] # Type: List[MaskCubeData]
        for d in datalist:
            beam = d.beam
            bmaj = np.array(beam["BMAJ"]) * beam["BMAJ"].unit
            bmin = np.array(beam["BMIN"]) * beam["BMIN"].unit
            bpa = np.array(beam["BPA"]) * beam["BPA"].unit
            beams = Beams(major=bmaj, minor=bmin, pa=bpa)
            datalist_mask.append(
                MaskCubeData(
                    mask = np.array([False] * nchans),
                    beams = beams,
                )
            )

        # Apply some masking
        datalist_masked = masking(datalist_mask=datalist_mask, cutoff=cutoff)
        # Get the grid size
        grids = [d.dy for d in datalist_masked]

        if not uselogs:
            if not suffix:
                suffix = mode
            if mode == "natural":
                commonbeams = natural_commonbeamer(
                    datalist_masked=datalist_masked,
                    grids=grids,
                    nchans=nchans,
                    conv_mode=conv_mode,
                    tolerance=tolerance,
                    nsamps=nsamps,
                    epsilon=epsilon,
                )
            elif mode == "total":
                commonbeams = total_commonbeamer(
                    datalist_masked=datalist_masked,
                    grids=grids,
                    nchans=nchans,
                    target_beam=target_beam,
                    conv_mode=conv_mode,
                    tolerance=tolerance,
                    nsamps=nsamps,
                    epsilon=epsilon,
                )

            beamlog_data_list = make_beamlogs(
                commonbeams=commonbeams,
                datalist=datalist,
                datalist_masked=datalist_masked,
                suffix=suffix,
                nchans=nchans,
                circularise=circularise,
            )
        else:
            logger.info("Reading from convolve beamlog files")
            beamlog_data_list = []
            for d in datalist:
                commonbeam_log = d.beamlog.with_suffix(
                    f".{suffix}.txt"
                )
                bemlog_data = readlogs(commonbeam_log)
                beamlog_data_list.append(bemlog_data)
    else:
        if not dryrun:
            files = None
            datalist = None
            datalist_masked = None
            beamlog_data_list = None
            nchans = None

    if mpiSwitch:
        comm.Barrier()

    # Init the files in parallel
    if not dryrun:
        if myPE == 0:
            logger.info("Initialising output files")
        if mpiSwitch:
            files = comm.bcast(files, root=0)
            datalist = comm.bcast(datalist, root=0)
            datalist_masked = comm.bcast(datalist_masked, root=0)
            beamlog_data_list = comm.bcast(beamlog_data_list, root=0)
            nchans = comm.bcast(nchans, root=0)

        indices = list(datadict.keys())
        dims = len(indices)

        if nPE > dims:
            my_start = myPE
            my_end = myPE

        else:
            count = dims // nPE
            rem = dims % nPE
            if myPE < rem:
                # The first 'remainder' ranks get 'count + 1' tasks each
                my_start = myPE * (count + 1)
                my_end = my_start + count

            else:
                # The remaining 'size - remainder' ranks get 'count' task each
                my_start = myPE * count + rem
                my_end = my_start + (count - 1)

        if myPE == 0:
            logger.info(f"There are {dims} files to init")
        logger.debug(f"My start is {my_start}, my end is {my_end}")

        # Init output files and retrieve file names
        outfile_dict = {} # Use a dict to preserve order
        for idx in indices[my_start : my_end + 1]:
            outfile = initfiles(
                filename=datalist[idx].filename,
                commonbeams=beamlog_data_list[idx].commonbeams,
                outdir=datalist[idx].outdir,
                mode=mode,
                suffix=suffix,
                prefix=prefix,
                ref_chan=ref_chan,
            )
            outfile_dict.update({idx: outfile})

        if mpiSwitch:
            # Send to master proc
            outlist = comm.gather(outfile_dict, root=0)
        else:
            outlist = [outfile_dict]

        if mpiSwitch:
            comm.Barrier()

        # Now do the convolution in parallel
        if myPE == 0:
            # Conver list to dict and save to main dict
            outlist_dict = {}
            for d in outlist:
                outlist_dict.update(d)
            # Also make inputs list
            inputs = []
            for key in datadict.keys():
                datadict[key]["outfile"] = outlist_dict[key]
                for chan in range(nchans):
                    inputs.append((key, chan))

        else:
            datadict = None
            inputs = None
        if mpiSwitch:
            comm.Barrier()
        if mpiSwitch:
            inputs = comm.bcast(inputs, root=0)
            datadict = comm.bcast(datadict, root=0)

        dims = len(files) * nchans
        assert len(inputs) == dims
        count = dims // nPE
        rem = dims % nPE
        if myPE < rem:
            # The first 'remainder' ranks get 'count + 1' tasks each
            my_start = myPE * (count + 1)
            my_end = my_start + count

        else:
            # The remaining 'size - remainder' ranks get 'count' task each
            my_start = myPE * count + rem
            my_end = my_start + (count - 1)
        if myPE == 0:
            logger.info(f"There are {nchans} channels, across {len(files)} files")
        logger.debug(f"My start is {my_start}, my end is {my_end}")

        for inp in inputs[my_start : my_end + 1]:
            key, chan = inp
            outfile = datadict[key]["outfile"]
            logger.debug(f"{outfile}  - channel {chan} - Started")

            cubedict = datadict[key]
            newim = worker(
                filename=cubedict["filename"],
                idx=chan,
                dx=cubedict["dx"],
                dy=cubedict["dy"],
                old_beam=cubedict["beams"][chan],
                final_beam=cubedict["commonbeams"][chan],
                conbeam=cubedict["convbeams"][chan],
                sfactor=cubedict["facs"][chan],
                conv_mode=conv_mode,
            )

            with fits.open(outfile, mode="update", memmap=True) as outfh:
                # Find which axis is the spectral and stokes
                wcs = WCS(outfh[0])
                axis_type_dict = wcs.get_axis_types()[::-1]  # Reverse order for fits
                axis_names = [i["coordinate_type"] for i in axis_type_dict]
                spec_idx = axis_names.index("spectral")
                stokes_idx = axis_names.index("stokes")
                slicer = [slice(None)] * len(outfh[0].data.shape)
                slicer[spec_idx] = chan
                slicer[stokes_idx] = 0  # only do single stokes
                slicer = tuple(slicer)

                outfh[0].data[slicer] = newim.astype(
                    np.float32
                )  # make sure data is 32-bit
                outfh.flush()
            logger.info(f"{outfile}  - channel {chan} - Done")

    logger.info("Done!")
    return datadict


def cli():
    """Command-line interface"""
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Smooth a field of 3D cubes to a common resolution.

    - Parallelisation is done using MPI.

    - Default names of output files are /path/to/beamlog{infile//.fits/.{SUFFIX}.fits}

    - By default, the smallest common beam will be automatically computed.
    - Optionally, you can specify a target beam to use.

    - It is currently assumed that cubes will be 4D with a dummy Stokes axis.
    - Iterating over Stokes axis is not yet supported.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "infile",
        metavar="infile",
        type=str,
        help="""Input FITS image(s) to smooth (can be a wildcard)
        - CASA beamtable will be used if present i.e. if CASAMBM = T
        - Otherwise beam info must be in co-located beamlog files.
        - beamlog must have the name /path/to/beamlog{infile//.fits/.txt}
        """,
        nargs="+",
    )

    parser.add_argument(
        "--uselogs",
        dest="uselogs",
        action="store_true",
        help="Get convolving information from previous run [False].",
    )

    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="natural",
        help="""Common resolution mode [natural].
        natural -- allow frequency variation.
        total -- smooth all plans to a common resolution.
        """,
    )

    parser.add_argument(
        "--conv_mode",
        dest="conv_mode",
        type=str,
        default="robust",
        choices=["robust", "scipy", "astropy", "astropy_fft"],
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
        "--logfile",
        default=None,
        type=str,
        help="Save logging output to file",
    )

    parser.add_argument(
        "-d",
        "--dryrun",
        dest="dryrun",
        action="store_true",
        help="Compute common beam and stop [False].",
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
        default=None,
        help="Add suffix to output filenames [{MODE}].",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default=None,
        help="Output directory of smoothed FITS image(s) [None - same as input].",
    )

    parser.add_argument(
        "--bmaj",
        dest="bmaj",
        type=float,
        default=None,
        help="BMAJ to convolve to [max BMAJ from given image(s)].",
    )

    parser.add_argument(
        "--bmin",
        dest="bmin",
        type=float,
        default=None,
        help="BMIN to convolve to [max BMAJ from given image(s)].",
    )

    parser.add_argument(
        "--bpa", dest="bpa", type=float, default=None, help="BPA to convolve to [0]."
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
        "--ref_chan",
        dest="ref_chan",
        type=str,
        default=None,
        choices=["first", "last", "mid"],
        help="""Reference psf for header [None].
            first  -- use psf for first frequency channel.
            last -- use psf for the last frequency channel.
            mid -- use psf for the centre frequency channel.
            Will use the CRPIX channel if not set.
            """,
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

    args = parser.parse_args()

    if args.verbosity == 1:
        logger.basicConfig(
            filename=args.logfile,
            level=logging.INFO,
            format=f"[{myPE}] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    elif args.verbosity >= 2:
        logger.basicConfig(
            filename=args.logfile,
            level=logger.DEBUG,
            format=f"[{myPE}] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    arg_dict = vars(args)
    # Pop the verbosity argument
    _ = arg_dict.pop("verbosity")
    # Pop the logfile argument
    _ = arg_dict.pop("logfile")

    _ = main(**arg_dict)


if __name__ == "__main__":
    sys.exit(cli())
