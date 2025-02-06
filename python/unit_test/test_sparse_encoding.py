import os
os.system("clear")

from pytopmulticuts.optimizer.quantum import sparse_encoding_blp

import numpy as np
import time

constraint = np.linspace(1, 10, 1000000).repeat(2).reshape(2, -1)
obj = np.linspace(1, 10, 1000000)

rhs = 10

start = time.perf_counter()
row, col, val = sparse_encoding_blp(obj, constraint, rhs, 2)
end = time.perf_counter()

print(f"Time elapsed: {end - start:.4f} s")