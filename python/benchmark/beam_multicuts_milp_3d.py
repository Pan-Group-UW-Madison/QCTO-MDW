from pytopmulticuts import SimpOptimizer, MulticutsOptimizer, LinearElasticity
from dolfinx.mesh import create_box, CellType, create_cell_partitioner
from dolfinx import mesh
from mpi4py import MPI
import numpy as np
from hashlib import sha256
import time
from datetime import datetime
from dolfinx.graph import partitioner_scotch
import argparse

import os
import sys

os.system("clear")
MPI.COMM_WORLD.barrier()

parser = argparse.ArgumentParser()
parser.add_argument("--Nx", type=int, default=50, help="Number of elements in x direction")
parser.add_argument("--log", action='store_true', help="Log file")

args, unknown = parser.parse_known_args()

Nx = Nz = args.Nx
Ny = 3 * Nx

hashtag_local = sha256(str(time.time()).encode()).hexdigest()
hashtag = MPI.COMM_WORLD.bcast(hashtag_local, root=0)

hashtag_short = hashtag[:8]

if args.log:
    output_filename = "log/beam_multicuts_milp_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + "_" + str(hashtag_short) + ".log"
    error_filename = "log/beam_multicuts_milp_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + "_" + str(hashtag_short) + ".err"
    sys.stdout = open(output_filename, 'w')
    sys.stderr = open(error_filename, 'w')

MPI.COMM_WORLD.barrier()

if MPI.COMM_WORLD.rank == 0:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Start time: " + date_time_str, flush=True)
    print("Hashtag: " + str(hashtag), flush=True)

mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 30, 10]],
                [Nx, Ny, Nz], CellType.hexahedron)

descriptor = {
    "prefix": "result/",
    "problem name": "beam_multicuts_milp_"+str(Nx)+"x"+str(Ny)+"x"+str(Nz)+"_"+str(hashtag_short),
    "mesh": mesh,
    "young's modulus": [210.0],
    "poisson's ratio": [0.29],
    "disp_bc": lambda x: np.isclose(x[1], 0) & (np.less(x[0], 1.5) | np.greater(x[0], 8.5)),
    "traction_bcs": [[(0, 0, -2.0),
                    lambda x: np.isclose(x[1], 30) & (
                        np.greater(x[0], 4.5) & np.less(x[0], 5.5)
                        & np.greater(x[2], 4.5) & np.less(x[2], 5.5))]],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-6,
        "ksp_max_it": 500,
        "ksp_gmres_restart": 100,
    },
    "objective": "compliance",
    "interpolation": "discrete",
}

problem = LinearElasticity(descriptor)

multicuts_descriptor = {
    "subproblem_solver": "milp",
    "max_iter": 200,
    "opt_tol": 5e-3,
    "initial_trust_region": 0.3,
    "filter_radius": 10 / Nx * 2.5,
    "vol_frac": 0.08,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
}

multicuts_optimizer = MulticutsOptimizer(multicuts_descriptor, problem)
multicuts_optimizer.solve()