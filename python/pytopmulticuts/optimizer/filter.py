from mpi4py import MPI
import numpy as np
from dolfinx.mesh import locate_entities, compute_midpoints
from dolfinx.geometry import bb_tree
from scipy.spatial import KDTree
from numba import njit
from petsc4py import PETSc
   
@njit
def compute_filter_values(target_centers, source_centers, neighbors_i, neighbors_j, radius):
    num_cell = target_centers.shape[0]
    filter_values = np.zeros(neighbors_j.shape[0], dtype=np.float64)
    for i in range(num_cell):
        for j in range(neighbors_i[i], neighbors_i[i+1]):
            neighbor = neighbors_j[j]
            dist = np.linalg.norm(target_centers[i, :] - source_centers[neighbor, :])
            filter_values[j] = max(0, radius - dist)
        filter_values[neighbors_i[i]:neighbors_i[i+1]] = \
            filter_values[neighbors_i[i]:neighbors_i[i+1]] / \
                np.sum(filter_values[neighbors_i[i]:neighbors_i[i+1]])
    
    return filter_values

class RadiusFilter:
    def __init__(self, mesh, radius):
        self.mesh = mesh
        self.comm = mesh.comm
        self.radius = radius
        
        self.comm.Barrier()
        
        # compute element centers
        num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
        self.num_cells_local = num_cells_local
        cell_centers_local = compute_midpoints(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32))
        self.coords = cell_centers_local
        
        local_tree = KDTree(cell_centers_local)
        
        # communicate ghosts indexes
        ghosts = mesh.topology.index_map(mesh.topology.dim).ghosts
        ghosts_owner = mesh.topology.index_map(mesh.topology.dim).owners
        filter_comm_rank = np.unique(ghosts_owner)
        self.filter_comm_rank = filter_comm_rank
        query_global_idx = []
        for rank in filter_comm_rank:
            query_global_idx.append(np.sort(ghosts[np.where(ghosts_owner == rank)[0]]).astype(np.int64))
        
        send_requests = []
        recv_requests = []
        query_idx_length = []
        for i, rank in enumerate(filter_comm_rank):
            send_requests.append(self.comm.isend(query_global_idx[i].shape[0], dest=rank, tag=0))
            recv_requests.append(self.comm.irecv(source=rank, tag=0))
        
        self.comm.Barrier()
        
        MPI.Request.Waitall(send_requests)
        
        for i in range(len(recv_requests)):
            query_idx_length.append(recv_requests[i].wait())
        
        send_requests = []
        recv_requests = []
        query_idx = []
        for i, rank in enumerate(filter_comm_rank):
            send_requests.append(self.comm.Isend([query_global_idx[i], MPI.LONG], dest=rank, tag=1))
            recv_idx = np.empty(query_idx_length[i], dtype=np.int64)
            query_idx.append(recv_idx)
            recv_requests.append(self.comm.Irecv([recv_idx, MPI.LONG], source=rank, tag=1))
        
        self.comm.Barrier()
        
        MPI.Request.Waitall(send_requests)
        MPI.Request.Waitall(recv_requests)
        
        query_local_idx = []
        for i, idx in enumerate(query_idx):
            query_local_idx.append(mesh.topology.index_map(mesh.topology.dim).global_to_local(idx))
        
        # find neighbors for each ghost cell
        query_neighbor_coords = []
        self.query_neighbors = []
        for idx in query_local_idx:
            query_points = cell_centers_local[idx, :]
            neighbors = np.unique(np.concatenate(local_tree.query_ball_point(query_points, radius)))
            self.query_neighbors.append(neighbors)
            query_neighbor_coords.append(cell_centers_local[neighbors, :].flatten())
        
        # communicate neighbors
        send_requests = []
        recv_requests = []
        recv_neighbor_num = []
        for i, rank in enumerate(filter_comm_rank):
            send_size = query_neighbor_coords[i].shape[0] / 3
            send_requests.append(self.comm.isend(send_size, dest=rank, tag=2))
            recv_requests.append(self.comm.irecv(source=rank, tag=2))        
        
        self.comm.Barrier()
        
        MPI.Request.Waitall(send_requests)
        
        for i in range(len(recv_requests)):
            recv_neighbor_num.append(recv_requests[i].wait())
        
        self.recv_neighbor_offset = np.cumsum([0] + recv_neighbor_num, dtype=np.int32)
        recv_neighbor_offset = self.recv_neighbor_offset
        
        send_requests = []
        recv_requests = []
        recv_neighbor_coords = np.zeros((recv_neighbor_offset[-1]*3), dtype=np.float64)
        for i, rank in enumerate(filter_comm_rank):
            send_requests.append(self.comm.Isend([query_neighbor_coords[i], MPI.DOUBLE], dest=rank, tag=3))
            recv_requests.append(self.comm.Irecv([recv_neighbor_coords[3*recv_neighbor_offset[i]:3*recv_neighbor_offset[i+1]], MPI.DOUBLE], source=rank, tag=3))
        
        self.comm.Barrier()
        
        MPI.Request.Waitall(send_requests)
        MPI.Request.Waitall(recv_requests)
        
        recv_neighbor_coords = recv_neighbor_coords.reshape(-1, 3)
        num_ghost = recv_neighbor_coords.shape[0]
        
        source_coords = np.concatenate([cell_centers_local, recv_neighbor_coords])
        num_cells_total = source_coords.shape[0]
        
        full_tree = KDTree(source_coords)
        full_neighbors = full_tree.query_ball_point(cell_centers_local, radius-1e-6)
        
        self.filter_i = np.cumsum([0] + [len(neighbors) for neighbors in full_neighbors]).astype(np.int32)
        self.filter_j = np.concatenate(full_neighbors).astype(np.int32)
        self.filter_v = compute_filter_values(cell_centers_local, source_coords, 
                                         self.filter_i, self.filter_j, radius)
        
        self.filter_mat = PETSc.Mat().createAIJWithArrays(
            (num_cells_local, num_cells_total), (self.filter_i, self.filter_j, self.filter_v),
            comm=MPI.COMM_SELF)
        
        self.rhs = PETSc.Vec().createSeq(num_cells_total, comm=MPI.COMM_SELF)
        self.lhs = PETSc.Vec().createSeq(num_cells_local, comm=MPI.COMM_SELF)
        
        # symmetry
        self.cell_symmetry_neighbor_rank = []
        self.cell_symmetry_local_idx = []
        self.cell_symmetry_send_idx = []
        
        x_coords = mesh.geometry.x[:, 0]
        x_max, x_min = np.max(x_coords), np.min(x_coords)
        x_max = MPI.COMM_WORLD.allreduce(x_max, op=MPI.MAX)
        x_min = MPI.COMM_WORLD.allreduce(x_min, op=MPI.MIN)
        # symmetry in x = 5
        length_x = x_max - x_min
        cell_centers_symmetry = cell_centers_local.copy()
        cell_centers_symmetry[:, 0] = length_x - cell_centers_symmetry[:, 0]
        
        self.prepare_symmetry(cell_centers_local, cell_centers_symmetry)
        
        y_coords = mesh.geometry.x[:, 1]
        y_max, y_min = np.max(y_coords), np.min(y_coords)
        y_max = MPI.COMM_WORLD.allreduce(y_max, op=MPI.MAX)
        y_min = MPI.COMM_WORLD.allreduce(y_min, op=MPI.MIN)
        # symmetry in y = 20
        length_y = y_max - y_min
        cell_centers_symmetry = cell_centers_local.copy()
        cell_centers_symmetry[:, 1] = length_y - cell_centers_symmetry[:, 1]
        
        self.prepare_symmetry(cell_centers_local, cell_centers_symmetry)
    
    def filter(self, field):
        # symmetry
        for n in range(len(self.cell_symmetry_neighbor_rank)):
            cell_symmetry_neighbor_rank = self.cell_symmetry_neighbor_rank[n]
            cell_symmetry_local_idx = self.cell_symmetry_local_idx[n]
            cell_symmetry_send_idx = self.cell_symmetry_send_idx[n]
            
            recv_fields = []
            send_requests = []
            recv_requests = []
            for i, rank in enumerate(cell_symmetry_neighbor_rank):
                send_field = field[cell_symmetry_send_idx[i]]
                send_requests.append(self.comm.Isend([send_field, MPI.DOUBLE], dest=rank, tag=7))
                recv_field = np.zeros(cell_symmetry_local_idx[i].shape[0], dtype=np.float64)
                recv_fields.append(recv_field)
                recv_requests.append(self.comm.Irecv([recv_field, MPI.DOUBLE], source=rank, tag=7))
            
            self.comm.Barrier()
            
            MPI.Request.Waitall(send_requests)
            MPI.Request.Waitall(recv_requests)
            
            for i, idx in enumerate(cell_symmetry_local_idx):
                field[idx] = 0.5 * (field[idx] + recv_fields[i])
        
        # radius filter
        self.rhs.array[:self.num_cells_local] = field
        
        send_requests = []
        recv_requests = []
        recv_fields = []
        for i, neighbors in enumerate(self.query_neighbors):
            send_field = field[neighbors]
            send_requests.append(self.comm.Isend([send_field, MPI.DOUBLE], dest=self.filter_comm_rank[i], tag=4))
            recv_field = np.zeros(self.recv_neighbor_offset[i+1]-self.recv_neighbor_offset[i], dtype=np.float64)
            recv_fields.append(recv_field)
            recv_requests.append(self.comm.Irecv([recv_field, MPI.DOUBLE], source=self.filter_comm_rank[i], tag=4))
        
        self.comm.Barrier()
        
        MPI.Request.Waitall(send_requests)
        MPI.Request.Waitall(recv_requests)
        
        for i, neighbors in enumerate(self.query_neighbors):
            idx = np.arange(self.recv_neighbor_offset[i], self.recv_neighbor_offset[i+1])+self.num_cells_local
            self.rhs.array[idx] = recv_fields[i]
        
        self.filter_mat.mult(self.rhs, self.lhs)
        values = self.lhs.array.copy()
        
        return values
    
    def prepare_symmetry(self, cell_centers_original, cell_centers_symmetry):
        cell_symmetry_neighbor_rank = []
        cell_symmetry_local_idx = []
        cell_symmetry_recv_size = []
        cell_symmetry_send_idx = []
        
        cell_symmetry_remote_idx = []
        for i in range(MPI.COMM_WORLD.size):
            if MPI.COMM_WORLD.rank == i:
                cell_centers_rank = cell_centers_original.copy()
                cell_centers_rank = cell_centers_rank.reshape(-1, 1)
            else:
                cell_centers_rank = None
            
            MPI.COMM_WORLD.Barrier()
            cell_centers_rank = MPI.COMM_WORLD.bcast(cell_centers_rank, root=i)
            cell_centers_rank = cell_centers_rank.reshape(-1, 3)
            
            if MPI.COMM_WORLD.rank != i:
                rank_tree = KDTree(cell_centers_rank)
                rank_neighbors = rank_tree.query_ball_point(cell_centers_symmetry, 1e-3)
                
                non_empty = [i for i, neighbors in enumerate(rank_neighbors) if len(neighbors) > 0]
                if len(non_empty) > 0:
                    cell_symmetry_neighbor_rank.append(i)
                    cell_symmetry_local_idx.append(np.array(non_empty, dtype=np.int32))
                    cell_symmetry_remote_idx.append(np.concatenate(rank_neighbors).astype(np.int32))
        
        cell_symmetry_send_size = []
        
        send_requests = []
        recv_requests = []
        for i, rank in enumerate(cell_symmetry_neighbor_rank):
            send_size = cell_symmetry_remote_idx[i].shape[0]
            send_requests.append(self.comm.isend(send_size, dest=rank, tag=5))
            recv_requests.append(self.comm.irecv(source=rank, tag=5))
        
        self.comm.Barrier()
        
        MPI.Request.Waitall(send_requests)
        for i in range(len(recv_requests)):
            cell_symmetry_recv_size.append(recv_requests[i].wait())
        
        send_requests = []
        recv_requests = []
        for i, rank in enumerate(cell_symmetry_neighbor_rank):
            send_requests.append(self.comm.Isend([cell_symmetry_remote_idx[i], MPI.INT], dest=rank, tag=6))
            recv_idx = np.empty(cell_symmetry_recv_size[i], dtype=np.int32)
            cell_symmetry_send_idx.append(recv_idx)
            recv_requests.append(self.comm.Irecv([recv_idx, MPI.INT], source=rank, tag=6))
        
        self.comm.Barrier()
        
        MPI.Request.Waitall(send_requests)
        MPI.Request.Waitall(recv_requests)
        
        self.cell_symmetry_neighbor_rank.append(cell_symmetry_neighbor_rank)
        self.cell_symmetry_local_idx.append(cell_symmetry_local_idx)
        self.cell_symmetry_send_idx.append(cell_symmetry_send_idx)