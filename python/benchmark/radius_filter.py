from pytopmulticuts import RadiusFilter
from dolfinx.mesh import create_box, CellType
from mpi4py import MPI
import numpy as np
import os
import time

def test_function(nx, ny, nz, output):
    num_elements = nx * ny * nz
    
    start = time.time()
    mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 10, 10]],
                        [nx, ny, nz], CellType.hexahedron)
    end = time.time()
    mesh_generation_time = end-start
    
    R = 10.0 / nx * 3
    
    start = time.time()
    radiusFilter = RadiusFilter(mesh, R)
    end = time.time()
    radius_filter_time = end-start
    
    if MPI.COMM_WORLD.rank == 0 and output:
        print(f"Number of elements: {num_elements}")
        print(f"  Mesh generation time: {mesh_generation_time:.4f}s")
        print(f"  Radius filter time: {radius_filter_time:.4f}s")
        print("", flush=True)

os.system("clear")

n_list = [50, 150, 200, 250, 300, 350, 400, 450, 500]

# warm up
test_function(50, 50, 50, False)

for n in n_list:
    test_function(n, n, n, True)