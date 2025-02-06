import numpy as np
import math

def sparse_encoding_blp(obj, dense_constraints, rhs, clustering = 2):
    if isinstance(dense_constraints, np.ndarray):
        if dense_constraints.ndim == 1:
            dense_constraints = dense_constraints.reshape(1, -1)
        row, col, val = sparse_encoding_constraint(dense_constraints, rhs, clustering)
    else:
        raise ValueError("The constraint must be a numpy array.")
    
    return row, col, val
    
def sparse_encoding_mblp(bin_obj, con_obj, bin_dense_constraint, con_constraint, rhs, clustering = 2):
    sparse_encoding_constraint(bin_constraint, rhs, clustering)

def sparse_encoding_constraint(constraints, rhs, clustering = 2):    
    L = 0
    num_new_var = 0;
    num_old_var = constraints.shape[1]
    l = num_old_var
    
    while l > clustering:
        L += 1
        l = (l + clustering - 1) // clustering
        num_new_var += l
    
    num_con = constraints.shape[0]
    num_new_var *= num_con
    
    num_var = num_old_var + num_new_var
    
    offset_by_level = np.zeros(L + 1, dtype=int)
    l = num_old_var
    for i in range(L):
        l = (l + clustering - 1) // clustering
        offset_by_level[i + 1] = offset_by_level[i] + L
    
    l1 = num_old_var
    l2 = num_old_var // clustering
    nnz = l1 + l2
    num_encoding_con = l2
    for i in range(L - 1):
        l1 = l2
        l2 = l2 // clustering
        nnz += l1 + l2
        num_encoding_con += l2
    
    nnz += clustering
    num_encoding_con += 1
    
    row = np.zeros(nnz * num_con, dtype=int)
    col = np.zeros(nnz * num_con, dtype=int)
    val = np.zeros(nnz * num_con, dtype=float)
    
    print(nnz)
    
    return row, col, val