#!/usr/bin/env python
"""
Parallel Hello World
from: https://github.com/erdc/mpi4py/blob/master/demo/helloworld.py
"""

import sys

from mpi4py import MPI

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

msg = f"Hello, World! I am process {rank:d} of {size:d} on {name:s}.\n"
sys.stdout.write(msg)
