#!/usr/bin/env python
""" Convolve ASKAP cubes to common resolution """
__author__ = "Alec Thomson"

from typing import Dict, List, Tuple
from racs_tools.beamcon_2D import my_ceil, round_up
from racs_tools.convolve_uv import smooth
from spectral_cube.utils import SpectralCubeWarning
import warnings
from astropy.utils.exceptions import AstropyWarning
import os
import stat
import sys
import numpy as np
import scipy.signal
from astropy import units as u
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import Table
from spectral_cube import SpectralCube
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from tqdm import tqdm, trange
from racs_tools import au2
import logging as log

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
        disable=(log.root.level > log.INFO),
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


def getbeams(file: str, header: fits.Header) -> Tuple[Table, int, str]:
    """Get beam information from a fits file or beamlog.

    Args:
        file (str): FITS filename.
        header (fits.Header): FITS header.

    Returns:
        Tuple[Table, int, str]: Table of beams, number of beams, and beamlog filename.
    """
    # Add beamlog info to dict just in case
    dirname = os.path.dirname(file)
    basename = os.path.basename(file)
    if dirname == "":
        dirname = "."
    beamlog = f"{dirname}/beamlog.{basename}".replace(".fits", ".txt")

    # First check for CASA beams
    try:
        headcheck = header["CASAMBM"]
    except KeyError:
        headcheck = False
    if headcheck:
        log.info(
            "CASA beamtable found in header - will use this table for beam calculations"
        )
        with fits.open(file) as hdul:
            hdu = hdul.pop("BEAMS")
            beams = Table(hdu.data)

    # Otherwise use beamlog file
    else:
        log.info("No CASA beamtable found in header - looking for beamlogs")
        log.info(f"Getting beams from {beamlog}")

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
    plane = cube.unmasked_data[idx].value.astype(np.float32)
    log.debug(f"Size of plane is {(plane.nbytes*u.byte).to(u.MB)}")
    newim = smooth(
        image=plane,
        **kwargs
    )
    return newim


def makedata(files: List[str], outdir: str) -> Dict[str, dict]:
    """Create data dictionary.

    Args:
        files (List[str]): List of filenames.
        outdir (str): Output directory.

    Raises:
        Exception: If pixel grid in X and Y is not the same.

    Returns:
        Dict[Dict]: Data and metadata for each channel and image.
    """
    datadict = {}  # type: Dict[str,dict]
    for i, (file, out) in enumerate(zip(files, outdir)):
        # Set up files
        datadict[f"cube_{i}"] = {}
        datadict[f"cube_{i}"]["filename"] = file
        datadict[f"cube_{i}"]["outdir"] = out
        # Get metadata
        header = fits.getheader(file)
        w = WCS(header)
        pixelscales = proj_plane_pixel_scales(w)

        dxas = pixelscales[0] * u.deg
        dyas = pixelscales[1] * u.deg

        datadict[f"cube_{i}"]["dx"] = dxas
        datadict[f"cube_{i}"]["dy"] = dyas
        if not dxas == dyas:
            raise Exception("GRID MUST BE SAME IN X AND Y")
        # Get beam info
        beam, nchan, beamlog = getbeams(file=file, header=header)
        datadict[f"cube_{i}"]["beamlog"] = beamlog
        datadict[f"cube_{i}"]["beam"] = beam
        datadict[f"cube_{i}"]["nchan"] = nchan
    return datadict


def commonbeamer(
    datadict: Dict[str, dict],
    nchans: int,
    conv_mode: str = "robust",
    mode: str = "natural",
    suffix: str = None,
    target_beam: Beam = None,
    circularise: bool = False,
    tolerance: float = 0.0001,
    nsamps: int = 200,
    epsilon: float = 0.0005,
) -> Dict[str, dict]:
    """Find common beam for all channels.
    Computed beams will be written to convolving beam log.

    Args:
        datadict (Dict[str, dict]): Main data dictionary.
        nchans (int): Number of channels.
        conv_mode (str, optional): Convolution mode. Defaults to "robust".
        mode (str, optional): Frequency mode. Defaults to "natural".
        target_beam (Beam, optional): Target PSF. Defaults to None.
        circularise (bool, optional): Circularise PSF. Defaults to False.
        tolerance (float, optional): Common beam tolerance. Defaults to 0.0001.
        nsamps (int, optional): Common beam samples. Defaults to 200.
        epsilon (float, optional): Common beam epsilon. Defaults to 0.0005.

    Raises:
        Exception: If convolving beam will be undersampled on pixel grid.

    Returns:
        Dict[str, dict]: Updated data dictionary.
    """
    if suffix is None:
        suffix = mode
    ### Natural mode ###
    if mode == "natural":
        big_beams = []
        for n in trange(
            nchans, desc="Constructing beams", disable=(log.root.level > log.INFO)
        ):
            majors_list = []
            minors_list = []
            pas_list = []
            for key in datadict.keys():
                major = datadict[key]["beams"][n].major
                minor = datadict[key]["beams"][n].minor
                pa = datadict[key]["beams"][n].pa
                if datadict[key]["mask"][n]:
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
        for beams in tqdm(
            big_beams,
            desc="Finding common beam per channel",
            disable=(log.root.level > log.INFO),
            total=nchans,
        ):
            if all(np.isnan(beams)):
                commonbeam = Beam(
                    major=np.nan * u.deg, minor=np.nan * u.deg, pa=np.nan * u.deg
                )
            else:
                try:
                    commonbeam = beams[~np.isnan(beams)].common_beam(
                        tolerance=tolerance, nsamps=nsamps, epsilon=epsilon,
                    )
                except BeamError:
                    log.warn("Couldn't find common beam with defaults")
                    log.warn("Trying again with smaller tolerance")

                    commonbeam = beams[~np.isnan(beams)].common_beam(
                        tolerance=tolerance * 0.1, nsamps=nsamps, epsilon=epsilon,
                    )
                # Round up values
                commonbeam = Beam(
                    major=my_ceil(commonbeam.major.to(u.arcsec).value, precision=1)
                    * u.arcsec,
                    minor=my_ceil(commonbeam.minor.to(u.arcsec).value, precision=1)
                    * u.arcsec,
                    pa=round_up(commonbeam.pa.to(u.deg), decimals=2),
                )

                grid = datadict[key]["dy"]
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
                        log.info(
                            f"Smallest common Nyquist sampled beam is: {nyq_beam!r}"
                        )

                        log.warn("COMMON BEAM WILL BE UNDERSAMPLED!")
                        log.warn("SETTING COMMON BEAM TO NYQUIST BEAM")
                        commonbeam = nyq_beam

            bmaj_common.append(commonbeam.major.to(u.arcsec).value)
            bmin_common.append(commonbeam.minor.to(u.arcsec).value)
            bpa_common.append(commonbeam.pa.to(u.deg).value)

        bmaj_common *= u.arcsec
        bmin_common *= u.arcsec
        bpa_common *= u.deg

        # Make Beams object
        commonbeams = Beams(major=bmaj_common, minor=bmin_common, pa=bpa_common)

    elif mode == "total":
        majors_list = []
        minors_list = []
        pas_list = []
        for key in datadict.keys():
            major = datadict[key]["beams"].major
            minor = datadict[key]["beams"].minor
            pa = datadict[key]["beams"].pa
            major[datadict[key]["mask"]] *= np.nan
            minor[datadict[key]["mask"]] *= np.nan
            pa[datadict[key]["mask"]] *= np.nan
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

        log.info("Finding common beam across all channels")
        log.info("This may take some time...")

        try:
            commonbeam = big_beams[~np.isnan(big_beams)].common_beam(
                tolerance=tolerance, nsamps=nsamps, epsilon=epsilon
            )
        except BeamError:
            log.warn("Couldn't find common beam with defaults")
            log.warn("Trying again with smaller tolerance")

            commonbeam = big_beams[~np.isnan(big_beams)].common_beam(
                tolerance=tolerance * 0.1, nsamps=nsamps, epsilon=epsilon
            )
        if target_beam is not None:
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
            grid = datadict[key]["dy"]
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
                log.info(f"Smallest common Nyquist sampled beam is: {nyq_beam!r}")
                if target_beam is not None:
                    commonbeam = target_beam
                    if target_beam < nyq_beam:
                        log.warn("TARGET BEAM WILL BE UNDERSAMPLED!")
                        raise Exception("CAN'T UNDERSAMPLE BEAM - EXITING")
                else:
                    log.warn("COMMON BEAM WILL BE UNDERSAMPLED!")
                    log.warn("SETTING COMMON BEAM TO NYQUIST BEAM")
                    commonbeam = nyq_beam

        # Make Beams object
        commonbeams = Beams(
            major=[commonbeam.major] * nchans * commonbeam.major.unit,
            minor=[commonbeam.minor] * nchans * commonbeam.minor.unit,
            pa=[commonbeam.pa] * nchans * commonbeam.pa.unit,
        )

    if circularise:
        log.info("Circular beam requested, setting BMIN=BMAJ and BPA=0")
        commonbeams = Beams(
            major=commonbeams.major, minor=commonbeams.major, pa=commonbeams.pa * 0,
        )

    log.info("Final beams are:")
    for i, commonbeam in enumerate(commonbeams):
        log.info(f"Channel {i}: {commonbeam!r}")

    for key in tqdm(
        datadict.keys(),
        desc="Getting convolution data",
        disable=(log.root.level > log.INFO),
    ):
        # Get convolving beams
        conv_bmaj = []
        conv_bmin = []
        conv_bpa = []
        old_beams = datadict[key]["beams"]
        masks = datadict[key]["mask"]
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
                    convbeam = Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg,)
                    log.warn(
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
            beams=datadict[key]["beams"],
            convbeams=convbeams,
            dx=datadict[key]["dx"],
            dy=datadict[key]["dy"],
        )
        datadict[key]["facs"] = facs

        # Setup conv beamlog
        datadict[key]["convbeams"] = convbeams
        commonbeam_log = datadict[key]["beamlog"].replace(".txt", f".{suffix}.txt")
        datadict[key]["commonbeams"] = commonbeams
        datadict[key]["commonbeamlog"] = commonbeam_log

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
        log.info(f"Convolving log written to {commonbeam_log}")

    return datadict


def masking(nchans:int, datadict: dict, cutoff: u.Quantity=None) -> dict:
    """Apply masking to data.

    Args:
        nchans (int): Number of channels in cubes.
        datadict (dict): Data dictionary.
        cutoff (None, optional): Cutoff BMAJ size for masking. Defaults to None.

    Returns:
        dict: Updated data dictionary.
    """    
    for key in datadict.keys():
        mask = np.array([False] * nchans)
        datadict[key]["mask"] = mask
    if cutoff is not None:
        for key in datadict.keys():
            majors = datadict[key]["beams"].major
            cutmask = majors > cutoff
            datadict[key]["mask"] += cutmask

    # Check for pipeline masking
    nullbeam = Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg)
    for key in datadict.keys():
        nullmask = datadict[key]["beams"] == nullbeam
        datadict[key]["mask"] += nullmask
    return datadict


def initfiles(filename: str, commonbeams: Beams, outdir:str, mode:str, suffix=None, prefix=None, ref_chan=None):
    """Initialise output files

    Args:
        datadict (dict): Main data dict - indexed
        mode (str): 'total' or 'natural'

    Returns:
        datadict: Updated datadict
    """
    log.debug(f"Reading {filename}")
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
        log.critical(
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
        log.warning("Reference PSF is NaN - replacing with 0 in the header")
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
        beam_table = Table(
            data=[
                commonbeams.major.to(u.arcsec),
                commonbeams.minor.to(u.arcsec),
                commonbeams.pa.to(u.deg),
                chans,
                pols,
            ],
            names=["BMAJ", "BMIN", "BPA", "CHAN", "POL"],
            dtype=["f4", "f4", "f4", "i4", "i4"],
        )
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
    if suffix is None:
        suffix = mode
    outname = os.path.basename(filename)
    outname = outname.replace(".fits", f".{suffix}.fits")
    if prefix is not None:
        outname = prefix + outname

    outfile = os.path.join(outdir,outname)
    log.info(f"Initialising to {outfile}")
    new_hdulist.writeto(outfile, overwrite=True)

    return outfile


def readlogs(commonbeam_log: str) -> Tuple[Beams, Beams, np.ndarray]:
    """Read convolving log files

    Args:
        commonbeam_log (str): Filename of the common beam log

    Raises:
        Exception: If the log file is not found

    Returns:
        Tuple[Beams, Beams, np.ndarray]: Common beams, convolving beams, and scaling factors
    """    
    log.info(f"Reading from {commonbeam_log}")
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
    log.info("Final beams are:")
    for i, commonbeam in enumerate(commonbeams):
        log.info(f"Channel {i}: {commonbeam!r}")
    return commonbeams, convbeams, facs


def main(args):
    """main script

    Args:
        args (args): Command line args

    """

    if myPE == 0:
        log.info(f"Total number of MPI ranks = {nPE}")
        # Parse args
        if args.dryrun:
            log.info("Doing a dry run -- no files will be saved")

        # Check mode
        mode = args.mode
        log.info(f"Mode is {mode}")
        if mode == "natural" and mode == "total":
            raise Exception("'mode' must be 'natural' or 'total'")
        if mode == "natural":
            log.info("Smoothing each channel to a common resolution")
        if mode == "total":
            log.info("Smoothing all channels to a common resolution")

        # Check cutoff
        cutoff = args.cutoff
        if args.cutoff is not None:
            cutoff = args.cutoff * u.arcsec
            log.info(f"Cutoff is: {cutoff}")

        # Check target
        conv_mode = args.conv_mode
        log.debug(conv_mode)
        if (
            not conv_mode == "robust"
            and not conv_mode == "scipy"
            and not conv_mode == "astropy"
            and not conv_mode == "astropy_fft"
        ):
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

        if not all(nonetest) and mode != "total":
            raise Exception("Only specify a target beam in 'total' mode")

        if all(nonetest):
            target_beam = None

        elif not all(nonetest) and any(nonetest):
            raise Exception("Please specify all target beam params!")

        elif not all(nonetest) and not any(nonetest):
            target_beam = Beam(bmaj * u.arcsec, bmin * u.arcsec, bpa * u.deg)
            log.info(f"Target beam is {target_beam!r}")

        files = sorted(args.infile)
        if files == []:
            raise Exception("No files found!")

        outdir = args.outdir
        if outdir is not None:
            if outdir[-1] == "/":
                outdir = outdir[:-1]
            outdir = [outdir] * len(files)
        else:
            outdir = []
            for f in files:
                out = os.path.dirname(f)
                if out == "":
                    out = "."
                outdir += [out]

        datadict = makedata(files, outdir)

        # Sanity check channel counts
        nchans = np.array([datadict[key]["nchan"] for key in datadict.keys()])
        check = all(nchans == nchans[0])

        if not check:
            raise Exception("Unequal number of spectral channels!")

        else:
            nchans = nchans[0]

        # Check suffix
        suffix = args.suffix
        if suffix is None:
            suffix = mode

        # Construct Beams objects
        for key in datadict.keys():
            beam = datadict[key]["beam"]
            bmaj = np.array(beam["BMAJ"]) * beam["BMAJ"].unit
            bmin = np.array(beam["BMIN"]) * beam["BMIN"].unit
            bpa = np.array(beam["BPA"]) * beam["BPA"].unit
            beams = Beams(major=bmaj, minor=bmin, pa=bpa)
            datadict[key]["beams"] = beams

        # Apply some masking
        datadict = masking(nchans=nchans, datadict=datadict, cutoff=cutoff)

        if not args.uselogs:
            datadict = commonbeamer(
                datadict=datadict,
                nchans=nchans,
                conv_mode=conv_mode,
                target_beam=target_beam,
                mode=mode,
                suffix=suffix,
                circularise=args.circularise,
                tolerance=args.tolerance,
                nsamps=args.nsamps,
                epsilon=args.epsilon,
            )
        else:
            log.info("Reading from convolve beamlog files")
            for key in datadict.keys():
                commonbeam_log = datadict[key]["beamlog"].replace(".txt", f".{suffix}.txt")
                commonbeams, convbeams, facs = readlogs(commonbeam_log)
                # Save to datadict
                datadict[key]["facs"] = facs
                datadict[key]["convbeams"] = convbeams
                datadict[key]["commonbeams"] = commonbeams
                datadict[key]["commonbeamlog"] = commonbeam_log
    else:
        if not args.dryrun:
            files = None
            datadict = None
            nchans = None

    if mpiSwitch:
        comm.Barrier()

    # Init the files in parallel
    if not args.dryrun:
        if myPE == 0:
            log.info("Initialising output files")
        if mpiSwitch:
            files = comm.bcast(files, root=0)
            datadict = comm.bcast(datadict, root=0)
            nchans = comm.bcast(nchans, root=0)

        conv_mode = args.conv_mode
        inputs = list(datadict.keys())
        dims = len(inputs)

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
            log.info(f"There are {dims} files to init")
        log.debug(f"My start is {my_start}, my end is {my_end}")

        # Init output files and retrieve file names
        outfile_dict = {}
        for inp in inputs[my_start : my_end + 1]:
            outfile = initfiles(
                filename=datadict[inp]["filename"],
                commonbeams=datadict[inp]["commonbeams"],
                outdir=datadict[inp]["outdir"],
                mode=args.mode,
                suffix=suffix,
                prefix=args.prefix,
                ref_chan=args.ref_chan,
            )
            outfile_dict.update({inp: outfile})

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
            log.info(f"There are {nchans} channels, across {len(files)} files")
        log.debug(f"My start is {my_start}, my end is {my_end}")

        for inp in inputs[my_start : my_end + 1]:
            key, chan = inp
            outfile = datadict[key]["outfile"]
            log.debug(f"{outfile}  - channel {chan} - Started")

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
                conv_mode=conv_mode
            )

            with fits.open(outfile, mode="update", memmap=True) as outfh:
                outfh[0].data[chan, 0, :, :] = newim.astype(
                    np.float32
                )  # make sure data is 32-bit
                outfh.flush()
            log.info(f"{outfile}  - channel {chan} - Done")

    log.info("Done!")


def cli():
    """Command-line interface
    """
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
        Note this mode cannot handle NaNs in the data.
        Can also be 'scipy', 'astropy', or 'astropy_fft'.
        Note these other methods cannot cope well with small convolving beams.
        """,
    )

    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )

    parser.add_argument(
        "--logfile", default=None, type=str, help="Save logging output to file",
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

    main(args)


if __name__ == "__main__":
    sys.exit(cli())
