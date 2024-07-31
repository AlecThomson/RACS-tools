#!/usr/bin/env python3
""" Utilities for parallelism """
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, Literal, TypeVar

from racs_tools.logging import logger

# logger = setup_logger()

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError:
    logger.debug("MPI not available")
    MPIPoolExecutor = None

Executor = TypeVar("Executor", ThreadPoolExecutor, ProcessPoolExecutor, MPIPoolExecutor)

EXECUTORS: Dict[Literal["thread", "process", "mpi"], Executor] = {
    "thread": ThreadPoolExecutor,
    "process": ProcessPoolExecutor,
    "mpi": MPIPoolExecutor,
}


def get_executor(
    executor_type: Literal["thread", "process", "mpi"] = "thread",
) -> Executor:
    """Get an executor based on the type

    Args:
        executor_type (Literal["thread", "process", "mpi"], optional): Type of executor. Defaults to "thread".

    Raises:
        ValueError: If the executor type is not available

    Returns:
        Executor: Executor class
    """
    ExecutorClass: Executor = EXECUTORS.get(executor_type, ThreadPoolExecutor)
    if ExecutorClass is None:
        raise ValueError(f"Executor type {executor_type} is not available!")
    logger.info(f"Using {ExecutorClass.__name__} for parallelisation")
    return ExecutorClass
