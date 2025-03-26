from pytopmulticuts import SimpOptimizer, MulticutsOptimizer, LinearElasticity
from dolfinx.mesh import create_box, CellType
from mpi4py import MPI
import numpy as np

import os
import sys
import argparse
from hashlib import sha256
import time
from datetime import datetime

os.system("clear")
MPI.COMM_WORLD.barrier()

parser = argparse.ArgumentParser()
parser.add_argument("--Nx", type=int, default=50, help="Number of elements in x direction")
parser.add_argument("--log", action='store_true', help="Log file")

args, unknown = parser.parse_known_args()

Nx = Nz = args.Nx
Ny = 4 * Nx

hashtag_local = sha256(str(time.time()).encode()).hexdigest()
hashtag = MPI.COMM_WORLD.bcast(hashtag_local, root=0)

hashtag_short = hashtag[:8]

if args.log:
    output_filename = "log/bridge_simp_dw_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + "_" + str(hashtag_short) + ".log"
    error_filename = "log/bridge_simp_dw_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + "_" + str(hashtag_short) + ".err"
    sys.stdout = open(output_filename, 'w')
    sys.stderr = open(error_filename, 'w')

MPI.COMM_WORLD.barrier()

if MPI.COMM_WORLD.rank == 0:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Start time: " + date_time_str, flush=True)
    print("Hashtag: " + str(hashtag), flush=True)

mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 40, 10]],
                [Nx, Ny, Nz], CellType.hexahedron)

descriptor = {
    "prefix": "result/",
    "problem name": "bridge_simp_"+str(Nx)+"x"+str(Ny)+"x"+str(Nz)+"_"+str(hashtag_short),
    "mesh": mesh,
    "young's modulus": [210.0],
    "poisson's ratio": [0.29],
    "disp_bc": lambda x: (np.isclose(x[2], 0) & np.less(x[1], 2)) | (np.isclose(x[2], 0) & np.greater(x[1], 38)),
    "traction_bcs": [[(0, 0, -0.1),
                     lambda x: np.isclose(x[2], 10)]],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 500,
        "ksp_gmres_restart": 100,
    },
    "objective": "compliance",
    "interpolation": "continuous",
}

problem = LinearElasticity(descriptor)

simp_descriptor = {
    "subproblem_solver": "oc",
    "max_iter": 500,
    "opt_tol": 1e-5,
    "filter_radius": 0.6,
    "vol_frac": 0.12,
    "beta_interval": 50,
    "beta_max": 128,
    "move": 0.02,
    "penalty": 3,
    "epsilon": 1e-6,
    "solid_zone": lambda x: np.greater(x[2], 9.6),
    "void_zone": lambda x: np.full(x.shape[1], False),
}

simp_optimizer = SimpOptimizer(simp_descriptor, problem)
simp_optimizer.solve()