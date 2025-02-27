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
Ny = 4 * Nx

hashtag_local = sha256(str(time.time()).encode()).hexdigest()
hashtag = MPI.COMM_WORLD.bcast(hashtag_local, root=0)

hashtag_short = hashtag[:8]

project_name = os.path.basename(__file__).split(".")[0]

if args.log:
    output_filename = "log/" + project_name + "_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + "_" + str(hashtag_short) + ".log"
    error_filename = "log/" + project_name + "_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + "_" + str(hashtag_short) + ".err"
    sys.stdout = open(output_filename, 'w')
    sys.stderr = open(error_filename, 'w')

MPI.COMM_WORLD.barrier()

if MPI.COMM_WORLD.rank == 0:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Start time: " + date_time_str, flush=True)
    print("Hashtag: " + str(hashtag), flush=True)

partitioner = create_cell_partitioner(partitioner_scotch())
mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 40, 10]],
                [Nx, Ny, Nz], CellType.hexahedron, ghost_mode=mesh.GhostMode.none, partitioner=partitioner)

descriptor = {
    "prefix": "intermediate/",
    # "problem_name": project_name+"_"+str(Nx)+"x"+str(Ny)+"x"+str(Nz)+"_"+str(hashtag_short),
    "problem_name": project_name,
    "mesh": mesh,
    # "young's modulus": [44, 73, 100, 210],
    # "density": [1.74, 2.70, 4.50, 7.80],
    "young's modulus": [44, 73, 210],
    "density": [1.74, 2.70, 7.8],
    # "young's modulus": [73, 210],
    # "density": [2.70, 7.80],
    "poisson's ratio": 0.3,
    "disp_bc": lambda x: (np.isclose(x[2], 0) & np.less(x[1], 2)) | (np.isclose(x[2], 0) & np.greater(x[1], 38)),
    "traction_bcs": [[(0, 0, -0.1),
                     lambda x: np.isclose(x[2], 10)]],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "gmres",
        "ksp_rtol": 1e-6,
        "ksp_max_it": 500,
        "ksp_gmres_restart": 100,
        # "ksp_monitor": None,
        "pc_type": "gamg",
    },
    "objective": "compliance",
    "interpolation": "discrete",
}

problem = LinearElasticity(descriptor)

multicuts_descriptor = {
    "subproblem_solver": "dw",
    "max_iter": 100,
    "opt_tol": 5e-3,
    "initial_trust_region": 0.3,
    "filter_radius": 10 / Nx * 2.5,
    "mass": 3120,
    # "initial_mass": 800,
    # "num_stages": 3,
    # "solid_zone": [lambda x: np.full(x.shape[1], False), lambda x: np.greater(x[2], 9.8)],
    # "void_zone": [lambda x: np.greater(x[2], 9.8), lambda x: np.full(x.shape[1], False)],
    "solid_zone": [lambda x: np.full(x.shape[1], False),  lambda x: np.full(x.shape[1], False), lambda x: np.greater(x[2], 9.8)],
    "void_zone": [lambda x: np.greater(x[2], 9.8),  lambda x: np.greater(x[2], 9.8), lambda x: np.full(x.shape[1], False)],
    # "solid_zone": [lambda x: np.full(x.shape[1], False), lambda x: np.full(x.shape[1], False), lambda x: np.full(x.shape[1], False), lambda x: np.greater(x[2], 9.8)],
    # "void_zone": [lambda x: np.greater(x[2], 9.8), lambda x: np.greater(x[2], 9.8), lambda x: np.greater(x[2], 9.8), lambda x: np.full(x.shape[1], False)],
    "num_divisions": 100,
    "solver_type": "quantum-simulated-subproblem",
}

multicuts_optimizer = MulticutsOptimizer(multicuts_descriptor, problem)
multicuts_optimizer.solve()