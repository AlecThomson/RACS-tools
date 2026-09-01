#!/usr/bin/env python3
"""Regression tests for the multiprocessing log listener.

On Python 3.13, ``logging.handlers.QueueListener.start()`` raises
``RuntimeError: Listener already started`` if it is called a second time
without an intervening ``stop()``. ``beamcon_3D.smooth_fits_cube`` and
``beamcon_2D.smooth_fits_files`` used to start the shared module-level
listener and only ever call ``enqueue_sentinel()`` on the way out, so the
listener's monitor thread reference was never cleared. Python 3.11/3.12
happen to tolerate that state, which is why this only showed up on 3.13.
"""

import logging
import multiprocessing as mp
import threading
from logging.handlers import QueueHandler, QueueListener

import pytest
from racs_tools import beamcon_3D
from racs_tools.logging import (
    init_worker,
    log_listener,
    log_queue,
    logger,
    running_log_listener,
)

from .test_3d import make_3d_image  # noqa: F401  (shared fixture)


class _ListHandler(logging.Handler):
    """A handler that just records what it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture()
def guard_queue_listener_start(monkeypatch):
    """Install Python 3.13's "already started" guard on ``QueueListener``.

    On 3.11/3.12 ``QueueListener.start()`` silently tolerates being called
    while a previous monitor thread reference is still set, so without this
    guard these regression tests would be a no-op on those versions.
    """
    original_start = QueueListener.start

    def start_once(self):
        if getattr(self, "_thread", None) is not None:
            raise RuntimeError("Listener already started")
        original_start(self)

    monkeypatch.setattr(QueueListener, "start", start_once)


def _assert_listener_reset() -> None:
    assert log_listener._thread is None
    assert log_queue.empty()


@pytest.mark.usefixtures("guard_queue_listener_start")
def test_smooth_fits_cube_twice(make_3d_image):  # noqa: F811
    """Two dry runs in the same process must both succeed."""
    for _ in range(2):
        beamcon_3D.smooth_fits_cube(
            infiles_list=[make_3d_image.path],
            mode="total",
            bmaj=60.0,
            bmin=60.0,
            bpa=0.0,
            dryrun=True,
        )
        _assert_listener_reset()


@pytest.mark.usefixtures("guard_queue_listener_start")
def test_smooth_fits_cube_dryrun_then_real(make_3d_image):  # noqa: F811
    """Match flint's call order: a dry run followed by a real convolution."""
    beamcon_3D.smooth_fits_cube(
        infiles_list=[make_3d_image.path],
        mode="total",
        bmaj=60.0,
        bmin=60.0,
        bpa=0.0,
        dryrun=True,
    )
    _assert_listener_reset()

    beamcon_3D.smooth_fits_cube(
        infiles_list=[make_3d_image.path],
        suffix="robust",
        mode="total",
        bmaj=60.0,
        bmin=60.0,
        bpa=0.0,
        dryrun=False,
    )
    _assert_listener_reset()

    outfile = make_3d_image.path.with_suffix(".robust.fits")
    assert outfile.exists()


@pytest.mark.usefixtures("guard_queue_listener_start")
def test_smooth_fits_cube_exception_then_success(make_3d_image):  # noqa: F811
    """An early raise must not leave the listener started for the next call."""
    with pytest.raises(FileNotFoundError):
        beamcon_3D.smooth_fits_cube(infiles_list=[])
    _assert_listener_reset()

    beamcon_3D.smooth_fits_cube(
        infiles_list=[make_3d_image.path],
        mode="total",
        bmaj=60.0,
        bmin=60.0,
        bpa=0.0,
        dryrun=True,
    )
    _assert_listener_reset()


def _log_from_worker(queue: mp.Queue, message: str) -> None:
    init_worker(queue, verbosity=2)
    logger.warning(message)


@pytest.mark.usefixtures("guard_queue_listener_start")
def test_running_log_listener_relays_records_across_calls(monkeypatch):
    """A stale sentinel would silently stop records from reaching the handler.

    This drives ``running_log_listener`` directly (rather than through
    ``smooth_fits_cube``) so it can swap in a handler that records what it
    receives, and repeats the start/stop cycle enough times that a leaked
    sentinel from an earlier call -- the bug described above -- would cause
    the monitor thread to exit immediately and silently drop later messages.
    """
    list_handler = _ListHandler()
    monkeypatch.setattr(log_listener, "handlers", (list_handler,))
    # Start from a clean handler list so the assertion counts one record per
    # message regardless of what earlier tests left attached to the logger.
    monkeypatch.setattr(logger, "handlers", [])

    for i in range(3):
        with running_log_listener():
            proc = mp.Process(target=_log_from_worker, args=(log_queue, f"message {i}"))
            proc.start()
            proc.join()
        _assert_listener_reset()

    messages = [record.getMessage() for record in list_handler.records]
    assert messages == ["message 0", "message 1", "message 2"]


@pytest.mark.usefixtures("guard_queue_listener_start")
def test_running_log_listener_nests():
    """Nested blocks share one listener run rather than deadlocking.

    The listener is a process-wide singleton, so an inner ``stop()`` used to
    enqueue a sentinel that the outer block's monitor thread consumed, leaving
    the inner ``join()`` waiting forever.
    """
    with running_log_listener():
        assert log_listener._thread is not None
        with running_log_listener():
            assert log_listener._thread is not None
        # The inner block must not have torn down the shared listener.
        assert log_listener._thread is not None
    _assert_listener_reset()


@pytest.mark.usefixtures("guard_queue_listener_start")
def test_running_log_listener_concurrent():
    """Concurrent blocks in different threads must not hang or crash."""
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with running_log_listener():
                barrier.wait(timeout=10)
        except BaseException as exc:  # noqa: BLE001 - reported via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive(), "running_log_listener deadlocked"

    assert errors == []
    _assert_listener_reset()


def test_init_worker_keeps_one_queue_handler():
    """``init_worker`` must not stack a queue handler per worker.

    With a thread executor every worker shares this process's logger, so
    adding unconditionally fanned each record out once per worker and left the
    handlers attached -- still filling the queue once the listener had stopped
    consuming it.
    """
    original_handlers = list(logger.handlers)
    try:
        for _ in range(3):
            init_worker(log_queue, verbosity=0)
            queue_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, QueueHandler)
            ]
            assert len(queue_handlers) == 1
    finally:
        logger.handlers = original_handlers
