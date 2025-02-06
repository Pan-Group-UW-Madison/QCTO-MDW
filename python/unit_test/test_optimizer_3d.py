from pytopmulticuts import SimpOptimizer, MulticutsOptimizer, LinearElasticity
from dolfinx.mesh import create_box, CellType
from mpi4py import MPI
import numpy as np

import os

# os.system("clear")

Nx = Nz = 75
Ny = 225

mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 30, 10]],
                  [Nx, Ny, Nz], CellType.hexahedron)

descriptor = {
    "prefix": "result/",
    "problem_name": "linear_elasticity_test",
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
    "interpolation": "discrete",
}

problem = LinearElasticity(descriptor)

# simp_descriptor = {
#     "subproblem_solver": "oc",
#     "max_iter": 400,
#     "opt_tol": 1e-5,
#     "filter_radius": 0.6,
#     "vol_frac": 0.08,
#     "beta_interval": 50,
#     "beta_max": 128,
#     "use_oc": True,
#     "move": 0.02,
#     "penalty": 3,
#     "epsilon": 1e-6,
#     "solid_zone": lambda x: np.full(x.shape[1], False),
#     "void_zone": lambda x: np.full(x.shape[1], False),
# }

# simp_optimizer = SimpOptimizer(simp_descriptor, problem)
# simp_optimizer.solve()

multicuts_descriptor = {
    "subproblem_solver": "dw",
    "max_iter": 100,
    "opt_tol": 1e-2,
    "initial_trust_region": 0.3,
    "filter_radius": 10 / Nx * 2,
    "vol_frac": 0.08,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
}

multicuts_optimizer = MulticutsOptimizer(multicuts_descriptor, problem)
multicuts_optimizer.solve()