from pytopmulticuts import MulticutsOptimizer, LinearElasticity
from dolfinx.mesh import create_rectangle, create_box, CellType
from mpi4py import MPI
import numpy as np

import os

os.system("clear")

mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [60, 40]],
                        [120, 80], CellType.quadrilateral)

descriptor = {
    "prefix": "result/",
    "problem_name": "beam_multicuts_2d",
    "mesh": mesh,
    "young's modulus": 1.0,
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

multicuts_descriptor = {
    "subproblem_solver": "milp",
    "max_iter": 20,
    "opt_tol": 5e-3,
    "filter_radius": 0.6,
    "vol_frac": 0.5,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
}

multicuts_optimizer = MulticutsOptimizer(multicuts_descriptor, problem)
multicuts_optimizer.solve()