from pytopmulticuts import RadiusFilter
from dolfinx.mesh import create_rectangle, create_box, CellType
from mpi4py import MPI
import numpy as np
import os
import time

os.system("clear")

Nx = Ny = 40

mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [40, 40]],
                        [Nx, Ny], CellType.quadrilateral)

R = 40.0 / Nx * 5

start = time.time()
radiusFilter = RadiusFilter(mesh, R)
end = time.time()

if MPI.COMM_WORLD.rank == 0:
    print(f"Time taken: {end-start:.4f}s")

# Nx = Ny = Nz = 40

# start = time.time()
# mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 10, 10]],
#                   [Nx, Nx, Nx], CellType.hexahedron)
# end = time.time()
# if MPI.COMM_WORLD.rank == 0:
#     print(f"Time taken for generating mesh: {end-start:.4f}s")

# R = 10.0 / Nx * 5

# start = time.time()
# radiusFilter = RadiusFilter(mesh, R)
# end = time.time()

# if MPI.COMM_WORLD.rank == 0:
#     print(f"Time taken: {end-start:.4f}s")