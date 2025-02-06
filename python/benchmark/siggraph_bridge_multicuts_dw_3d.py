from pytopmulticuts import SimpOptimizer, MulticutsOptimizer, LinearElasticity
from dolfinx.mesh import create_box, CellType
from mpi4py import MPI
import numpy as np
from hashlib import sha256
import time
from datetime import datetime

import os
import sys

os.system("clear")
MPI.COMM_WORLD.barrier()

Nx = Nz = int(sys.argv[1])
Ny = 4 * Nx

hashtag_local = sha256(str(time.time()).encode()).hexdigest()
hashtag = MPI.COMM_WORLD.bcast(hashtag_local, root=0)

hashtag_short = hashtag[:8]

output_filename = "log/bridge_multicuts_dw_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + "_" + str(hashtag_short) + ".log"
error_filename = "log/bridge_multicuts_dw_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + "_" + str(hashtag_short) + ".err"
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
    "problem_name": "bridge_multicuts_dw_"+str(Nx)+"x"+str(Ny)+"x"+str(Nz)+"_"+str(hashtag_short),
    "mesh": mesh,
    "young's modulus": [100.0],
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[1], 0) | np.isclose(x[1], 40),
    "traction_bcs": [[(0, 0, -0.1),
                     lambda x: np.isclose(x[2], 0)]],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-6,
        "ksp_max_it": 500,
        "ksp_gmres_restart": 120,
    },
    "objective": "compliance",
    "interpolation": "discrete",
}

problem = LinearElasticity(descriptor)

multicuts_descriptor = {
    "subproblem_solver": "dw",
    "max_iter": 100,
    "opt_tol": 1e-2,
    "initial_trust_region": 0.3,
    "filter_radius": 10 / Nx * 2.5,
    "vol_frac": 0.08,
    "solid_zone": lambda x: np.less(x[2], 0.2),
    "void_zone": lambda x: np.full(x.shape[1], False),
    "num_divisions": 100,
    "solver_type": "quantum-simulated-subproblem",
}

multicuts_optimizer = MulticutsOptimizer(multicuts_descriptor, problem)
multicuts_optimizer.solve()