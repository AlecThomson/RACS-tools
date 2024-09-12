#!/usr/bin/env python3
""" Convolve ASKAP cubes to common resolution """
__author__ = "Alec Thomson"

import logging
import sys
import warnings
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional, Tuple

import numpy as np
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
from racs_tools.convolve_uv import parse_conv_mode, smooth
from racs_tools.logging import (
    init_worker,
    log_listener,
    log_queue,
    logger,
    set_verbosity,
)
from racs_tools.parallel import get_executor

warnings.filterwarnings(action="ignore", category=SpectralCubeWarning, append=True)
warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

#############################################
#### ADAPTED FROM SCRIPT BY T. VERNSTROM ####
#############################################


class CubeBeams(NamedTuple):
    """Cube beams"""

    beam_table: Table
    """Beam table"""
    nchan: int
    """Number of channels"""
    beamlog: Path
    """Beamlog filename"""


class CubeData(NamedTuple):
    """Cube data and metadata"""

    filename: Path
    """Cube filename"""
    outdir: Path
    """Output directory"""
    dx: u.Quantity
    """Pixel size in x direction"""
    dy: u.Quantity
    """Pixel size in y direction"""
    beamlog: Path
    """Beamlog filename"""
    beam_table: Table
    """Beam table"""
    beams: Beams
    """Beams object"""
    nchan: int
    """Number of channels"""
    mask: np.ndarray
    """Mask array"""

    def with_options(self, **kwargs):
        """Create a new CubeData instance with keywords updated

        Returns:
            CubeData: New CubeData instance with updated attributes
        """
        # TODO: Update the signature to have the actual attributes to
        # help keep mypy and other linters happy
        as_dict = self._asdict()
        as_dict.update(kwargs)

        return CubeData(**as_dict)


class CommonBeamData(NamedTuple):
    """Common beam data"""

    commonbeams: Beams
    """Common beams"""
    convbeams: Beams
    """Convolving beams"""
    facs: np.ndarray
    """Scaling factors"""
    commonbeamlog: Path
    """Common beamlog filename"""


def get_beams(file: Path, header: fits.Header) -> CubeBeams:
    """Get beam information from a fits file or beamlog.

    Args:
        file (Path): FITS filename.
        header (fits.Header): FITS headesr.

    Raises:
        e: If beam information cannot be extracted.

    Returns:
        CubeBeams: Table of beams, number of beams, and beamlog filename.
    """
    # Add beamlog info to dict just in case
    dirname = file.parent
    basename = file.name
    beamlog = (dirname / f"beamlog.{basename}").with_suffix(".txt")

    # First check for CASA beams
    headcheck = header.get("CASAMBM", False)
    if headcheck:
        logger.info(
            "CASA beamtable found in header - will use this table for beam calculations"
        )
        with fits.open(file) as hdul:
            hdu = hdul.pop("BEAMS")
            beams = Table.read(hdu)

    # Otherwise use beamlog file
    else:
        if beamlog.exists():
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
        else:
            logger.warning("No beamlog found")
            logger.warning("Using header beam - assuming a constant beam")
            try:
                beam: Beam = Beam.from_fits_header(header)
                wcs = WCS(header)
                spec_axis = wcs.spectral
                _nchan = spec_axis.array_shape[0]
                beams = Table()
                beams.add_column(np.arange(_nchan), name="Channel")
                beams.add_column([beam.major.to(u.arcsec)] * _nchan, name="BMAJ")
                beams.add_column(
                    [beam.minor.to(u.arcsec)] * _nchan,
                    name="BMIN",
                )
                beams.add_column([beam.pa.to(u.deg)] * _nchan, name="BPA")

            except Exception as e:
                logger.error("Couldn't get beam from header")
                raise e

    nchan = len(beams)
    return CubeBeams(beams, nchan, beamlog)


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
            fac, _, _, _, _ = au2.gauss_factor(
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


def smooth_plane(
    filename: Path,
    idx: int,
    old_beam: Beam,
    new_beam: Beam,
    dx: u.Quantity,
    dy: u.Quantity,
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"] = "robust",
    mask: bool = False,
) -> np.ndarray:
    """Smooth a single plane of a cube.

    Args:
        filename (Path): FITS filename.
        idx (int): Channel index.
        old_beam (Beam): Old beam.
        new_beam (Beam): Target beam.
        dx (u.Quantity): Pixel size in x direction.
        dy (u.Quantity): Pixel size in y direction.
        conv_mode (Literal["robust", "scipy", "astropy", "astropy_fft"], optional): Convolution mode. Defaults to "robust".
        mask (bool, optional): Mask the channel. Defaults to False.

    Returns:
        np.ndarray: Convolved plane.
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

    if mask:
        logger.info(f"Masking channel {idx}")
        newim = plane * np.nan
    else:
        newim = smooth(
            image=plane,
            old_beam=old_beam,
            new_beam=new_beam,
            dx=dx,
            dy=dy,
            conv_mode=conv_mode,
        )
    del plane
    return newim


def make_data(files: List[Path], outdir: List[Path]) -> List[CubeData]:
    """Create data dictionary.

    Args:
        files (List[Path]): List of filenames.
        outdir (List[Path]): Output directories.

    Raises:
        Exception: If pixel grid in X and Y is not the same.

    Returns:
        List[CubeData]: Data and metadata for each channel and image.
    """
    cube_data_list: List[CubeData] = []
    for _, (file, out) in enumerate(zip(files, outdir)):
        # Get metadata
        header = fits.getheader(file)
        w = WCS(header)
        pixelscales = proj_plane_pixel_scales(w)

        dxas = pixelscales[0] * u.deg
        dyas = pixelscales[1] * u.deg
        if not np.isclose(dxas, dyas):
            raise ValueError(f"GRID MUST BE SAME IN X AND Y, got {dxas} and {dyas}")
        # Get beam info
        beam_table, nchan, beamlog = get_beams(file=file, header=header)
        # Construct beams
        bmaj = np.array(beam_table["BMAJ"]) * beam_table["BMAJ"].unit
        bmin = np.array(beam_table["BMIN"]) * beam_table["BMIN"].unit
        bpa = np.array(beam_table["BPA"]) * beam_table["BPA"].unit
        beams = Beams(major=bmaj, minor=bmin, pa=bpa)
        cube_data = CubeData(
            filename=file,
            outdir=out,
            dx=dxas,
            dy=dyas,
            beamlog=beamlog,
            beam_table=beam_table,
            beams=beams,
            nchan=nchan,
            mask=np.array([False] * nchan),
        )
        cube_data_list.append(cube_data)
    return cube_data_list


def commonbeamer(
    cube_data_list: List[CubeData],
    nchans: int,
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"] = "robust",
    mode: Literal["natural", "total"] = "natural",
    suffix: Optional[str] = None,
    target_beam: Optional[Beam] = None,
    circularise: bool = False,
    tolerance: float = 0.0001,
    nsamps: int = 200,
    epsilon: float = 0.0005,
) -> List[CommonBeamData]:
    """Find common beam for all channels.
    Computed beams will be written to convolving beam logger.

    Args:
        cube_data_list (List[CubeData]): List of cube data.
        nchans (int): Number of channels.
        conv_mode (Literal["robust", "scipy", "astropy", "astropy_fft"], optional): Convolution mode. Defaults to "robust".
        mode (Literal["natural", "total"], optional): Frequency mode. Defaults to "natural".
        target_beam (Beam, optional): Target PSF. Defaults to None.
        circularise (bool, optional): Circularise PSF. Defaults to False.
        tolerance (float, optional): Common beam tolerance. Defaults to 0.0001.
        nsamps (int, optional): Common beam samples. Defaults to 200.
        epsilon (float, optional): Common beam epsilon. Defaults to 0.0005.

    Raises:
        Exception: If convolving beam will be undersampled on pixel grid.

    Returns:
        List[CommonBeamData]: Common beam data for each channel and cube.
    """
    if suffix is None:
        suffix = mode
    ### Natural mode ###
    if mode == "natural":
        big_beams = []
        for n in trange(
            nchans,
            desc="Constructing beams",
            disable=(logger.level > logging.INFO),
        ):
            majors_list = []
            minors_list = []
            pas_list = []
            for cube_data in cube_data_list:
                major = cube_data.beams[n].major
                minor = cube_data.beams[n].minor
                pa = cube_data.beams[n].pa
                if cube_data.mask[n]:
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
            disable=(logger.level > logging.INFO),
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

                grid = cube_data.dy
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

    elif mode == "total":
        majors_list = []
        minors_list = []
        pas_list = []
        for cube_data in cube_data_list:
            major = cube_data.beams.major
            minor = cube_data.beams.minor
            pa = cube_data.beams.pa
            major[cube_data.mask] *= np.nan
            minor[cube_data.mask] *= np.nan
            pa[cube_data.mask] *= np.nan
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
            grid = cube_data.dy
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

    common_beam_data_list: List[CommonBeamData] = []
    for cube_data in tqdm(
        cube_data_list,
        desc="Getting convolution data",
        disable=(logger.level > logging.INFO),
    ):
        # Get convolving beams
        conv_bmaj = []
        conv_bmin = []
        conv_bpa = []
        old_beams = cube_data.beams
        masks = cube_data.mask
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
            beams=cube_data.beams,
            convbeams=convbeams,
            dx=cube_data.dx,
            dy=cube_data.dy,
        )

        # Setup conv beamlog
        commonbeam_log = cube_data.beamlog.with_suffix(f".{suffix}.txt")
        common_beam_data = CommonBeamData(
            commonbeams=commonbeams,
            convbeams=convbeams,
            facs=facs,
            commonbeamlog=commonbeam_log,
        )
        common_beam_data_list.append(common_beam_data)

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

    return common_beam_data_list


def masking(
    cube_data_list: List[CubeData], cutoff: Optional[u.Quantity] = None
) -> List[CubeData]:
    """Apply masking to cubes

    Args:
        cube_data_list (List[CubeData]): List of cube data.
        cutoff (Optional[u.Quantity], optional): Cutoff for PSF. Defaults to None.

    Returns:
        List[CubeData]: List of masked cube data.
    """
    # Check for pipeline masking
    nullbeam = Beam(major=0 * u.deg, minor=0 * u.deg, pa=0 * u.deg)
    tiny = np.finfo(np.float32).tiny  # Smallest positive number - used to mask
    smallbeam = Beam(major=tiny * u.deg, minor=tiny * u.deg, pa=tiny * u.deg)
    masked_cube_data_list: List[CubeData] = []
    for cube_data in cube_data_list:
        mask = cube_data.mask
        if cutoff is not None:
            majors = cube_data.beams.major
            cutmask = majors > cutoff
            mask += cutmask
        nullmask = np.logical_or(
            cube_data.beams == nullbeam,
            cube_data.beams == smallbeam,
        )
        mask += nullmask
        masked_cube_data = cube_data.with_options(mask=mask)
        masked_cube_data_list.append(masked_cube_data)
    return masked_cube_data_list


def initfiles(
    filename: Path,
    commonbeams: Beams,
    outdir: Path,
    mode: Literal["natural", "total"],
    suffix="",
    prefix="",
    ref_chan: Optional[int] = None,
) -> Path:
    """Initialise output files

    Args:
        filename (Path): Original filename
        commonbeams (Beams): Common beams for each channel
        outdir (Path): Output directory
        mode (Literal["natural", "total"]): Frequency mode - natural or total
        suffix (str, optional): Output suffix. Defaults to "".
        prefix (str, optional): Output prefix. Defaults to "".
        ref_chan (Optional[int], optional): Reference channel index. Defaults to None.

    Raises:
        ValueError: If no Stokes axis is found in the header

    Returns:
        Path: Output filename
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
            "More than one Stokes parameter in header. Only the first one will be used."
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
        header["COMMENT"] = (
            "Full beam information is stored in the second FITS extension."
        )
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
    if suffix is None:
        suffix = mode
    outfile = Path(filename.name)
    if suffix:
        outfile = outfile.with_suffix(f".{suffix}.fits")
    if prefix:
        outfile = Path(prefix + outfile.name)

    outfile = outdir / outfile.name
    logger.info(f"Initialising to {outfile}")
    new_hdulist.writeto(outfile, overwrite=True)

    return outfile


def readlogs(commonbeam_log: Path) -> CommonBeamData:
    """Read convolving log files

    Args:
        commonbeam_log (Path): Filename of the common beam log

    Raises:
        Exception: If the log file is not found

    Returns:
        CommonBeamData: Common beams, convolving beams, and scaling factors
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
    return CommonBeamData(
        commonbeams=commonbeams,
        convbeams=convbeams,
        facs=facs,
        commonbeamlog=commonbeam_log,
    )


def smooth_and_write_plane(
    chan: int,
    cube_data: CubeData,
    common_beam_data: CommonBeamData,
    outfile: Path,
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"] = "robust",
) -> None:
    """Smooth a single plane of a cube and write to a output file.

    Args:
        chan (int): Channel index.
        cube_data (CubeData): Cube data.
        common_beam_data (CommonBeamData): Common beam data.
        outfile (Path): Output filename.
        conv_mode (Literal["robust", "scipy", "astropy", "astropy_fft"], optional): _description_. Defaults to "robust".
    """
    logger.debug(f"{outfile}  - channel {chan} - Started")

    newim = smooth_plane(
        filename=cube_data.filename,
        idx=chan,
        old_beam=cube_data.beams[chan],
        new_beam=common_beam_data.commonbeams[chan],
        dx=cube_data.dx,
        dy=cube_data.dy,
        conv_mode=conv_mode,
        mask=cube_data.mask[chan],
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

        outfh[0].data[slicer] = newim.astype(np.float32)  # make sure data is 32-bit
        outfh.flush()
        del newim

    logger.info(f"{outfile}  - channel {chan} - Done")


def smooth_fits_cube(
    infiles_list: List[Path],
    uselogs: bool = False,
    mode: Literal["natural", "total"] = "natural",
    conv_mode: Literal["robust", "scipy", "astropy", "astropy_fft"] = "robust",
    dryrun: bool = False,
    prefix: str = None,
    suffix: str = None,
    outdir: Optional[Path] = None,
    bmaj: Optional[float] = None,
    bmin: Optional[float] = None,
    bpa: Optional[float] = None,
    cutoff: Optional[float] = None,
    circularise: bool = False,
    ref_chan: Optional[int] = None,
    tolerance: float = 0.0001,
    epsilon: float = 0.0005,
    nsamps: int = 200,
    ncores: Optional[int] = None,
    executor_type: Literal["thread", "process", "mpi"] = "thread",
    verbosity: int = 0,
) -> Tuple[List[CubeData], List[CommonBeamData]]:
    """Convolve a set of FITS cubes to a common beam.

    Args:
        infiles_list (List[Path]): FITS files to convolve.
        uselogs (bool, optional): Use beamlogs. Defaults to False.
        mode (Literal["natural", "total"], optional): Frequency mode. Defaults to "natural".
        conv_mode (Literal["robust", "scipy", "astropy", "astropy_fft"], optional): Convolution mode. Defaults to "robust".
        dryrun (bool, optional): Do not write any images. Defaults to False.
        prefix (str, optional): Output filename prefix. Defaults to None.
        suffix (str, optional): Output filename suffix. Defaults to None.
        outdir (Optional[Path], optional): Output directory. Defaults to None.
        bmaj (Optional[float], optional): Target beam major axis in arcsec. Defaults to None.
        bmin (Optional[float], optional): Target beam minor axis in arcsec. Defaults to None.
        bpa (Optional[float], optional): Target beam position angle in deg. Defaults to None.
        cutoff (Optional[float], optional): Beam cutoff in arcsec. Defaults to None.
        circularise (bool, optional): Set minor axis to major axis. Defaults to False.
        ref_chan (Optional[int], optional): Reference channel for PSF in header. Defaults to None.
        tolerance (float, optional): Radio beam tolerance. Defaults to 0.0001.
        epsilon (float, optional): Radio beam epsilon. Defaults to 0.0005.
        nsamps (int, optional): Radio beam nsamps. Defaults to 200.
        ncores (Optional[int], optional): Radio beam ncores. Defaults to None.
        executor_type (Literal["thread", "process", "mpi"], optional): Executor type. Defaults to "thread".

    Raises:
        ValueError: If mode is not 'natural' or 'total'.
        ValueError: If target beam is not fully specified.
        FileNotFoundError: If no files are found.
        ValueError: If number of channels are not equal.

    Returns:
        Tuple[List[CubeData], List[CommonBeamData]]: Cube data and common beam data.
    """
    # Required for multiprocessing logging
    log_listener.start()
    if dryrun:
        logger.info("Doing a dry run -- no files will be saved")

    # Check early as can fail
    Executor = get_executor(executor_type)

    # Check mode
    logger.info(f"Mode is {mode}")
    if mode == "natural":
        logger.info("Smoothing each channel to a common resolution")
    elif mode == "total":
        logger.info("Smoothing all channels to a common resolution")
    else:
        raise ValueError(f"Mode must be 'natural' or 'total', not '{mode}'")

    # Check cutoff
    if cutoff is not None:
        cutoff *= u.arcsec
        logger.info(f"Cutoff is: {cutoff}")

    # Check target
    conv_mode = parse_conv_mode(conv_mode)

    nonetest = [param is None for param in (bmaj, bmin, bpa)]
    if all(nonetest):
        target_beam = None
    elif any(nonetest):
        raise ValueError("Please specify all target beam params!")
    else:
        target_beam = Beam(bmaj * u.arcsec, bmin * u.arcsec, bpa * u.deg)
        logger.info(f"Target beam is {target_beam!r}")

    files = sorted(infiles_list)
    if len(files) == 0:
        raise FileNotFoundError("No files found!")

    outdir_list: List[Path] = (
        [outdir] * len(files) if outdir is not None else [f.parent for f in files]
    )

    cube_data_list = make_data(files, outdir_list)

    # Sanity check channel counts
    nchans = np.array([cube_data.nchan for cube_data in cube_data_list])
    check = all(nchans == nchans[0])

    if not check:
        raise ValueError(f"Unequal number of spectral channels! Got {nchans}")

    nchans = nchans[0]

    # Check suffix
    if suffix is None:
        suffix = mode

    # Apply some masking
    cube_data_list = masking(cube_data_list, cutoff=cutoff)

    if not uselogs:
        common_beam_data_list = commonbeamer(
            cube_data_list=cube_data_list,
            nchans=nchans,
            conv_mode=conv_mode,
            target_beam=target_beam,
            mode=mode,
            suffix=suffix,
            circularise=circularise,
            tolerance=tolerance,
            nsamps=nsamps,
            epsilon=epsilon,
        )
    else:
        logger.info("Reading from convolve beamlog files")
        common_beam_data_list: List[CommonBeamData] = []
        for cube_data in cube_data_list:
            commonbeam_log = cube_data.beamlog.with_suffix(f".{suffix}.txt")
            common_beam_data = readlogs(commonbeam_log)
            common_beam_data_list.append(common_beam_data)

    if dryrun:
        logger.info("Doing a dryrun so all done!")
        return cube_data_list, common_beam_data_list

    # Init the files in parallel
    logger.info("Initialising output files")
    # Init output files and retrieve file names
    with Executor(
        max_workers=ncores, initializer=init_worker, initargs=(log_queue, verbosity)
    ) as executor:
        futures = []
        for cube_data, common_beam_data in zip(cube_data_list, common_beam_data_list):
            future = executor.submit(
                initfiles,
                filename=cube_data.filename,
                commonbeams=common_beam_data.commonbeams,
                outdir=cube_data.outdir,
                mode=mode,
                suffix=suffix,
                prefix=prefix,
                ref_chan=ref_chan,
            )
            futures.append(future)
    outfiles = [future.result() for future in futures]

    with Executor(
        max_workers=ncores, initializer=init_worker, initargs=(log_queue, verbosity)
    ) as executor:
        futures = []
        for cube_data, common_beam_data, outfile in zip(
            cube_data_list, common_beam_data_list, outfiles
        ):
            for chan in range(nchans):
                future = executor.submit(
                    smooth_and_write_plane,
                    chan=chan,
                    cube_data=cube_data,
                    common_beam_data=common_beam_data,
                    outfile=outfile,
                    conv_mode=conv_mode,
                )
                futures.append(future)
        _ = [future.result() for future in futures]

    logger.info("Done!")

    log_listener.enqueue_sentinel()

    return cube_data_list, common_beam_data_list


def cli():
    """Command-line interface"""
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Smooth a field of 3D cubes to a common resolution.

    - Default names of output files are /path/to/beamlog{infile//.fits/.{SUFFIX}.fits}

    - By default, the smallest common beam will be automatically computed.
    - Optionally, you can specify a target beam to use.

    - It is currently assumed that cubes will be 4D with a dummy Stokes axis.
    - Iterating over Stokes axis is not yet supported.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "infile",
        metavar="infile",
        type=Path,
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
        help="Compute common beam and stop.",
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
        type=Path,
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

    parser.add_argument(
        "--ncores",
        type=int,
        default=None,
        help="Number of cores to use for parallelisation. If None, use all available cores.",
    )

    parser.add_argument(
        "--executor_type",
        type=str,
        default="thread",
        choices=["thread", "process", "mpi"],
        help="Executor type for parallelisation.",
    )

    args = parser.parse_args()

    # Set up logging
    set_verbosity(
        logger=logger,
        verbosity=args.verbosity,
    )

    _ = smooth_fits_cube(
        infiles_list=args.infile,
        uselogs=args.uselogs,
        mode=args.mode,
        conv_mode=args.conv_mode,
        dryrun=args.dryrun,
        prefix=args.prefix,
        suffix=args.suffix,
        outdir=args.outdir,
        bmaj=args.bmaj,
        bmin=args.bmin,
        bpa=args.bpa,
        cutoff=args.cutoff,
        circularise=args.circularise,
        ref_chan=args.ref_chan,
        tolerance=args.tolerance,
        epsilon=args.epsilon,
        ncores=args.ncores,
        nsamps=args.nsamps,
        executor_type=args.executor_type,
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    sys.exit(cli())
