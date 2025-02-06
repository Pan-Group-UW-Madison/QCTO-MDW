import os
os.system("clear")

from pytopmulticuts import LinearElasticity
from mpi4py import MPI
import numpy as np
from dolfinx.mesh import create_rectangle, create_box, CellType

# 2D
mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [60, 40]],
                        [120, 80], CellType.quadrilateral)

descriptor = {
    "prefix": "result/",
    "problem_name": "linear_elasticity_test",
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

problem.summary()

problem.solve_prime()

# 3D
mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 20, 10]],
                  [16, 32, 16], CellType.hexahedron)

descriptor = {
    "prefix": "result/",
    "problem_name": "linear_elasticity_test",
    "mesh": mesh,
    "young's modulus": [100.0],
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[1], 0) & (np.less(x[0], 1.5) | np.greater(x[0], 8.5)),
    "traction_bcs": [[(0, 0, -2.0),
                     lambda x: np.isclose(x[1], 20) & (
                         np.greater(x[0], 4.5) & np.less(x[0], 5.5)
                         & np.greater(x[2], 4.5) & np.less(x[2], 5.5))]],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
    "objective": "compliance",
    "interpolation": "discrete",
}

problem = LinearElasticity(descriptor)

problem.summary()