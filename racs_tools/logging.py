#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
from typing import Optional, Tuple

logging.captureWarnings(True)

# Following guide from gwerbin/multiprocessing_logging.py
# https://gist.github.com/gwerbin/e9ab7a88fef03771ab0bf3a11cf921bc


def setup_logger(
    filename: Optional[str] = None,
) -> Tuple[logging.Logger, QueueListener, mp.Queue]:
    """Setup a logger

    Args:
        filename (Optional[str], optional): Output log file. Defaults to None.

    Returns:
        Tuple[logging.Logger, QueueListener, mp.Queue]: Logger, log listener and log queue
    """
    logger = logging.getLogger("racs_tools")
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        fmt="[%(threadName)s] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    log_queue = mp.Queue()
    log_listener = QueueListener(log_queue, ch)

    return logger, log_listener, log_queue


def set_verbosity(logger: logging.Logger, verbosity: int) -> None:
    """Set the logger verbosity

    Args:
        logger (logging.Logger): The logger
        verbosity (int): Verbosity level
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logger.setLevel(level)


def init_worker(log_queue: mp.Queue, verbosity: int = 0) -> None:
    """Initialise a worker process with a logger

    Args:
        log_queue (mp.Queue): The log queue
        verbosity (int, optional): Verbosity level. Defaults to 0.
    """
    logger = logging.getLogger("racs_tools")

    set_verbosity(logger, verbosity)

    handler = QueueHandler(log_queue)
    logger.addHandler(handler)


logger, log_listener, log_queue = setup_logger()
