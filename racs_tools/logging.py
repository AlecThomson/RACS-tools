#!/usr/bin/env python3

import logging
import multiprocessing as mp
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import QueueHandler, QueueListener

logging.captureWarnings(True)

# Following guide from gwerbin/multiprocessing_logging.py
# https://gist.github.com/gwerbin/e9ab7a88fef03771ab0bf3a11cf921bc


def setup_logger(
    filename: str | None = None,
) -> tuple[logging.Logger, QueueListener, mp.Queue]:
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

    # Keep exactly one queue handler per process. With a thread executor every
    # worker shares this logger, so adding unconditionally fans each record out
    # once per worker and leaves the handlers attached for later calls - where
    # they keep filling the queue after the listener has stopped consuming it.
    for stale in [h for h in logger.handlers if isinstance(h, QueueHandler)]:
        logger.removeHandler(stale)

    logger.addHandler(QueueHandler(log_queue))


logger, log_listener, log_queue = setup_logger()

_listener_lock = threading.Lock()
_listener_users = 0
"""Number of ``running_log_listener`` blocks currently active."""


@contextmanager
def running_log_listener() -> Iterator[None]:
    """Run the module-level log listener for the duration of the block.

    On Python 3.13, ``QueueListener.start()`` raises ``RuntimeError`` if the
    listener's monitor thread is still set from a previous run. Calling
    ``enqueue_sentinel()`` alone (without ``stop()``) leaves the thread
    reference set even after the monitor has exited, so the listener must
    always be stopped -- on every exit path -- before it can be started
    again.

    The listener is a module-level singleton, so nested or concurrent blocks
    share the one run: it is started by the first block to enter and stopped
    by the last to leave. Without that, an inner ``stop()`` would enqueue a
    sentinel that the outer block's monitor thread consumes, and the
    ``join()`` inside ``stop()`` would then never return.
    """
    global _listener_users

    with _listener_lock:
        if _listener_users == 0:
            log_listener.start()
        _listener_users += 1
    try:
        yield
    finally:
        with _listener_lock:
            _listener_users -= 1
            if _listener_users == 0:
                log_listener.stop()
