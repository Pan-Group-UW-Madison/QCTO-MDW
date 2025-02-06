from pytopmulticuts import SimpOptimizer, MulticutsOptimizer, LinearElasticity
from dolfinx.mesh import create_box, CellType
from mpi4py import MPI
import numpy as np

import os
import sys

os.system("clear")

Nx = Nz = 75 # 322
Ny = 4 * Nx

output_filename = "log/bridge_multicuts_milp_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + ".log"
sys.stdout = open(output_filename, 'w')

mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 40, 10]],
                [Nx, Ny, Nz], CellType.hexahedron)

descriptor = {
    "prefix": "result/",
    "problem_name": "bridge_multicuts_milp_"+str(Nx)+"x"+str(Ny)+"x"+str(Nz),
    "mesh": mesh,
    "young's modulus": [100.0],
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[1], 0) | np.isclose(x[1], 40),
    "traction_bcs": [[(0, 0, -1.0),
                     lambda x: np.isclose(x[2], 0)]],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-6,
    },
    "objective": "compliance",
    "interpolation": "discrete",
}

problem = LinearElasticity(descriptor)

multicuts_descriptor = {
    "subproblem_solver": "milp",
    "max_iter": 100,
    "opt_tol": 1e-2,
    "initial_trust_region": 0.3,
    "filter_radius": 10 / Nx * 2.5,
    "vol_frac": 0.08,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
}

multicuts_optimizer = MulticutsOptimizer(multicuts_descriptor, problem)
multicuts_optimizer.solve()