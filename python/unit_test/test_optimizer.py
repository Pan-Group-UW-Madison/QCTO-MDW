from pytopmulticuts import SimpOptimizer, MulticutsOptimizer, LinearElasticity
from dolfinx.mesh import create_rectangle, CellType
from mpi4py import MPI
import numpy as np

import os

mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [60, 40]],
                        [120, 80], CellType.quadrilateral)

descriptor = {
    "prefix": "result/",
    "problem_name": "linear_elasticity_test",
    "mesh": mesh,
    "young's modulus": [1.0],
    "density": 1.0,
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[0], 0),
    "traction_bcs": [[(0, -1.0),
                      lambda x: (np.isclose(x[0], 60) & np.greater(x[1], 18) & np.less(x[1], 22))]],
    "body_force": (0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-6,
    },
    "objective": "compliance",
    "interpolation": "discrete",
}

problem = LinearElasticity(descriptor)

# simp_descriptor = {
#     "subproblem_solver": "oc",
#     "max_iter": 300,
#     "opt_tol": 1e-5,
#     "filter_radius": 0.6,
#     "vol_frac": 0.3,
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
    "max_iter": 50,
    "opt_tol": 1e-2,
    "initial_trust_region": 0.3,
    "filter_radius": 60 / 120 * 3,
    "vol_frac": 0.3,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
}

multicuts_optimizer = MulticutsOptimizer(multicuts_descriptor, problem)
multicuts_optimizer.solve()