from pytopmulticuts import SimpOptimizer, MulticutsOptimizer, LinearElasticity
from dolfinx.mesh import create_box, CellType
from mpi4py import MPI
import numpy as np

import os
import sys

os.system("clear")

Nx = Nz = 100 # 322
Ny = 3 * Nx

output_filename = "log/output_simp_" + str(Nx) + "x" + str(Ny) + "x" + str(Nz) + ".log"
sys.stdout = open(output_filename, 'w')

mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 30, 10]],
                [Nx, Ny, Nz], CellType.hexahedron)

descriptor = {
    "prefix": "result/",
    "problem_name": "beam_simp_"+str(Nx)+"x"+str(Ny)+"x"+str(Nz),
    "mesh": mesh,
    "young's modulus": [100.0],
    "poisson's ratio": 0.25,
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
    },
    "objective": "compliance",
    "interpolation": "continuous",
}

problem = LinearElasticity(descriptor)

simp_descriptor = {
    "subproblem_solver": "oc",
    "max_iter": 400,
    "opt_tol": 1e-5,
    "filter_radius": 0.6,
    "vol_frac": 0.08,
    "beta_interval": 50,
    "beta_max": 128,
    "use_oc": True,
    "move": 0.02,
    "penalty": 3,
    "epsilon": 1e-6,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
}

simp_optimizer = SimpOptimizer(simp_descriptor, problem)
simp_optimizer.solve()