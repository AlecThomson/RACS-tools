#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

try:
    from mpi4py import MPI

    myPE = MPI.COMM_WORLD.Get_rank()
except ImportError:
    myPE = 0

logging.captureWarnings(True)
logger = logging.getLogger("racs_tools")
logger.setLevel(logging.WARNING)

formatter = logging.Formatter(
    fmt=f"[{myPE}] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logger(
    verbosity: int = 0,
    filename: Optional[str] = None,
):
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
