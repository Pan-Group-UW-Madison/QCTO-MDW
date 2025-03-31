from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
import math

import gurobipy as gp

from .optimizer import SubOptimizer
from .quantum import sparse_encoding_blp

class Cluster():
    def __init__(self, num_local_size, num_materials=1):
        self.comm = MPI.COMM_WORLD
        self.num_materials = num_materials
        self.num_local_size = num_local_size
        self.num_global_size = int(self.comm.allreduce(num_local_size))
    
    def prepare_cluster(self, rho, nD):
        self.nD = nD
        
        rho_stacked = np.zeros((self.num_local_size))
        for i in range(self.num_materials):
            rho_stacked += (i+1) * rho[i]
        unique_rho_value_local = np.unique(rho_stacked)
        
        unique_rho_value_global = self.comm.allgather(unique_rho_value_local)
        unique_rho_value_global = np.unique(np.concatenate(unique_rho_value_global))
        
        local_unique_rho = []
        for i in range(unique_rho_value_global.size):
            local_unique_rho.append(np.where(rho_stacked == unique_rho_value_global[i])[0])
        
        local_unique_size = np.array([len(local_unique_rho[i]) for i in range(unique_rho_value_global.size)])
        global_unique_size = self.comm.allreduce(local_unique_size)
        global_unique_size_offset = np.cumsum([0] + global_unique_size.tolist())
        division_unique_size_offset_by_rank = np.zeros((unique_rho_value_global.size, self.comm.size+1), dtype=int)
        for i in range(unique_rho_value_global.size):
            unique_value_size = self.comm.allgather(local_unique_size[i])
            division_unique_size_offset_by_rank[i, :] = np.cumsum([0] + unique_value_size)
        
        offset = np.linspace(0, self.num_global_size, nD+1, dtype=int)
        
        self.offset_source = []
        self.offset_target = []
        for i in range(len(local_unique_rho)):
            idx = local_unique_rho[i]
            if len(idx) == 0:
                continue
            start_global_idx = int(global_unique_size_offset[i]+division_unique_size_offset_by_rank[i, self.comm.rank])
            end_global_idx = int(global_unique_size_offset[i]+division_unique_size_offset_by_rank[i, self.comm.rank] + local_unique_size[i])
            start_division_idx = int(np.searchsorted(offset, start_global_idx, side='left'))
            end_division_idx = int(np.searchsorted(offset, end_global_idx, side='left'))
            
            if start_global_idx < offset[start_division_idx]:
                start_division_idx -= 1
            idx_range = np.zeros((end_division_idx-start_division_idx+1), dtype=int)
            idx_range[0] = 0
            idx_range[-1] = len(idx)
            
            for j in range(1, idx_range.size-1):
                idx_range[j] = min(offset[start_division_idx+j] - start_global_idx, len(idx))
            
            for j in range(idx_range.size-1):
                self.offset_source.append(start_division_idx+j)
                self.offset_target.append(idx[idx_range[j]:idx_range[j+1]])
    
    def apply_cluster(self, vector):
        result = np.zeros(self.nD*self.num_materials)
        
        for i in range(self.num_materials):
            local_result = np.zeros(self.nD)
            for j in range(len(self.offset_source)):
                local_result[self.offset_source[j]] = vector[self.offset_target[j]+self.num_local_size*i].sum()
            result[i*self.nD:(i+1)*self.nD] = self.comm.reduce(local_result, root=0)
        
        return result
    
    def apply_backward_cluster(self, vector):
        result = np.zeros(self.num_local_size*self.num_materials)
        
        for i in range(self.num_materials):
            for j in range(len(self.offset_source)):
                result[self.offset_target[j]+self.num_local_size*i] = vector[i*self.nD+self.offset_source[j]]
        
        return result

class DWOptimizer(SubOptimizer):
    def __init__(self, problem, num_free_size, num_local_size, num_materials=1, num_divisions=1, solver_type="classical"):
        super().__init__(problem)
        
        self.rho_local_size = num_free_size
        self.rho_global_size = int(self.comm.allreduce(num_local_size))
        
        rho_size_by_rank = self.comm.allgather(self.rho_local_size)
        self.rho_offset = np.cumsum([0] + rho_size_by_rank)
        
        self.num_materials = num_materials
        
        self.options = {
            "outputflag": 0,
            "MIPGap": 1e-9,
            "FeasibilityTol": 1e-9,
            "OptimalityTol": 1e-9,
            "Threads": 1
        }
        
        self.env = gp.Env(params=self.options)
        self.sub_problem_model = gp.Model(env=self.env)
        
        self.x = self.sub_problem_model.addMVar((self.rho_local_size*self.num_materials, ), vtype=gp.GRB.BINARY, lb=0, ub=1, name="x")
        
        if self.num_materials > 1:
            for i in range(self.rho_local_size):
                self.sub_problem_model.addConstr(self.x[i::self.rho_local_size].sum() <= 1)
        
        rho_field = self.problem.rho_field[0]
        num_elems = rho_field.x.petsc_vec.array.size
        self.centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems]
        
        self.nD = num_divisions
        
        if solver_type == "classical":
            self.subproblem_quantum_simulated = False
        elif solver_type == "quantum-simulated-subproblem":
            self.subproblem_quantum_simulated = True
        
        self.cluster = Cluster(self.rho_local_size, self.num_materials)
    
    def set_dQdrho(self, dQdrho):
        self.dQdrho = np.zeros((1, self.rho_local_size*self.num_materials))
        for i in range(self.num_materials):
            self.dQdrho[0, i*self.rho_local_size:(i+1)*self.rho_local_size] = dQdrho[i].copy()
        
    def update(self, rho, obj, weight, quantity, d):
        if isinstance(obj, float):
            n = 1
        else:
            n = len(obj)
        
        self.lagrange_multiplier_list = []
        self.rho_list = []
        self.cost_list = []
        
        compare_milp_solution = True
        check_milp_solution = False
        
        if check_milp_solution or compare_milp_solution:
            milp_start = MPI.Wtime()
            rho_milp, cost_milp, lagrange_multiplier_milp = self.milp_solution(rho, obj, weight, quantity, d)
            milp_end = MPI.Wtime()
        
        if compare_milp_solution:
            dw_start = MPI.Wtime()
        
        nD = self.nD
        if n == 1:
            obj_coeff_original = np.zeros((self.rho_local_size*self.num_materials))
            for i in range(self.num_materials):
                obj_coeff_original[i*self.rho_local_size:(i+1)*self.rho_local_size] = weight[i]
        
            rho_stacked = np.zeros((self.rho_local_size))
            for j in range(self.num_materials):
                rho_stacked += rho[j][:self.rho_local_size]
            trust_region_coeff_stack = (1 - 2*rho_stacked) / self.rho_global_size
            trust_region_coeff = np.zeros((self.rho_local_size*self.num_materials))
            for i in range(self.num_materials):
                trust_region_coeff[i*self.rho_local_size:(i+1)*self.rho_local_size] = trust_region_coeff_stack
            
            constraints_coeff = np.vstack([self.dQdrho, trust_region_coeff])
            rho_total = self.comm.allreduce(np.sum(rho_stacked**2))
            rhs = [quantity, -rho_total/self.rho_global_size + d]
        else:
            constraints_coeff = []
            rhs = []
            
            for i in range(n):
                cut_coeff = np.zeros((self.rho_local_size*self.num_materials))
                for j in range(self.num_materials):
                    cut_coeff[j*self.rho_local_size:(j+1)*self.rho_local_size] = weight[i][j]
                
                rho_stacked = np.zeros((self.rho_local_size))
                for j in range(self.num_materials):
                    rho_stacked += rho[i][j][:self.rho_local_size]
                trust_region_coeff_stack = (1 - 2*rho_stacked) / self.rho_global_size
                trust_region_coeff = np.zeros((self.rho_local_size*self.num_materials))
                for j in range(self.num_materials):
                    trust_region_coeff[j*self.rho_local_size:(j+1)*self.rho_local_size] = trust_region_coeff_stack
                
                constraints_coeff.append(cut_coeff)
                constraints_coeff.append(trust_region_coeff)
                
                rho_flattened = np.zeros((self.rho_local_size*self.num_materials))
                for j in range(self.num_materials):
                    rho_flattened[j*self.rho_local_size:(j+1)*self.rho_local_size] = rho[i][j][:self.rho_local_size]
                
                rhs_local = cut_coeff@rho_flattened
                rhs_local = self.comm.allreduce(rhs_local)
                rhs.append(-obj[i]+rhs_local)
                rho_total = self.comm.allreduce(np.sum(rho_stacked**2))
                rhs.append(-rho_total/self.rho_global_size + d[i])
            
            constraints_coeff.append(self.dQdrho)
            rhs.append(quantity)
            
            constraints_coeff = np.vstack(constraints_coeff)
        
        self.comm.Barrier()
        start_time = MPI.Wtime()
        
        feasibility = 0
        while True:
            if n == 1:
                rho_init, lagrange_multipliers_init, feasibility = self.initialize_single_cut(obj_coeff_original, rho, constraints_coeff, rhs, nD, self.rho_local_size)
            else:
                rho_init, lagrange_multipliers_init, feasibility = self.initialize_multicuts(rho, constraints_coeff, rhs, nD, self.rho_local_size, n)
            if feasibility == 1:
                break
            else:
                if n > 1:
                    for i in range(n):
                        rhs[2*i+1] = rhs[2*i+1] + d[i]
                else:
                    rhs[1] = rhs[1] + d
        self.lagrange_multiplier_list.append(lagrange_multipliers_init)
        self.rho_list.append(rho_init)
        
        if nD > self.nD:
            nD = self.nD
            self.cluster.prepare_cluster(rho, nD)
        
        end_time = MPI.Wtime()
        if self.comm.rank == 0:
            print(f"  Initialization time: {end_time - start_time:.4f} s, nD: {nD}", flush=True)
        
        sub_problem_time = 0
        master_problem_time = 0
        self.master_milp_problem_time = 0
        
        if self.subproblem_quantum_simulated:
            self.sub_problem_casting_time = 0
        
        if n == 1:
            rho_flattened = np.zeros((self.rho_local_size*self.num_materials))
            for i in range(self.num_materials):
                rho_flattened[i*self.rho_local_size:(i+1)*self.rho_local_size] = rho[i][:self.rho_local_size]
        else:
            rho_flattened = []
            for i in range(n):
                rho_flattened.append(np.zeros((self.rho_local_size*self.num_materials)))
                for j in range(self.num_materials):
                    rho_flattened[i][j*self.rho_local_size:(j+1)*self.rho_local_size] = rho[i][j][:self.rho_local_size]
        
        for i in range(50):
            start_time = MPI.Wtime()
            if n == 1:
                rho_dw, cost = self.subproblem_single_cut(obj_coeff_original, obj, constraints_coeff, self.lagrange_multiplier_list[-1], rho_flattened)
            else:
                rho_dw, cost = self.subproblem_multicuts(obj, constraints_coeff, self.lagrange_multiplier_list[-1], rho_flattened, n)
            end_time = MPI.Wtime()
            sub_problem_time += end_time - start_time
            self.rho_list.append(rho_dw.copy())
            self.cost_list.append(cost)
            
            start_time = MPI.Wtime()
            if n == 1:
                lagrange_multipliers = self.master_problem_single_cut(obj_coeff_original, constraints_coeff, rhs, nD)
            else:
                lagrange_multipliers = self.master_problem_multicuts(obj, constraints_coeff, rhs, nD, n)
            end_time = MPI.Wtime()
            master_problem_time += end_time - start_time
            
            if check_milp_solution:
                if self.comm.rank == 0:
                    print(lagrange_multipliers, flush=True)
            
            stop_criteria = 0
            for j in range(i):
                lagrange_diff = np.linalg.norm(lagrange_multipliers - self.lagrange_multiplier_list[j])
                if lagrange_diff / np.linalg.norm(lagrange_multipliers) < 1e-9:
                    stop_criteria = 1
                    rho_new = self.rho_list[j]
                    cost = self.cost_list[j]
                    break
            
            if stop_criteria == 1:
                break
            
            self.lagrange_multiplier_list.append(lagrange_multipliers)
        
        if compare_milp_solution:
            dw_end = MPI.Wtime()
        
        if self.comm.rank == 0:
            print(f"  Converged at iteration {i}", flush=True)
            print(f"  Subproblem time: {sub_problem_time:.4f} s", flush=True)
            print(f"  Master problem time: {master_problem_time:.4f} s", flush=True)
            print(f"  Master MILP problem time: {self.master_milp_problem_time:.4f} s", flush=True)
            
            if self.subproblem_quantum_simulated:
                print(f"  Construction time of quantum subproblem: {self.sub_problem_casting_time:.4f} s", flush=True)
            
            if compare_milp_solution:
                print(f"  DW solution time: {dw_end - dw_start:.4f} s", flush=True)
                print(f"  MILP solution time: {milp_end - milp_start:.4f} s", flush=True)
        
        rho_result = []
        if check_milp_solution:
            for i in range(self.num_materials):
                rho_result.append(rho_milp[i*self.rho_local_size:(i+1)*self.rho_local_size].copy())
            cost = cost_milp
        else:
            for i in range(self.num_materials):
                rho_result.append(rho_dw[i*self.rho_local_size:(i+1)*self.rho_local_size].copy())
        
        return rho_result, cost
    
    def initialize_single_cut(self, obj, rho, constraints, rhs, nD, rho_size):
        self.cluster.prepare_cluster(rho, nD)
        
        obj_reduced_global = self.cluster.apply_cluster(obj)
        quantity_reduced_global = self.cluster.apply_cluster(constraints[0, :])
        trust_region_reduced_global = self.cluster.apply_cluster(constraints[1, :])
        
        if self.comm.rank == 0:
            env = gp.Env(params=self.options)
            model = gp.Model(env=env)
            
            x = model.addMVar(nD * self.num_materials, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
            
            obj_func = obj_reduced_global @ x
            
            model.setObjective(obj_func, gp.GRB.MINIMIZE)
            
            model.addConstr(quantity_reduced_global @ x <= rhs[0])
            model.addConstr(trust_region_reduced_global @ x <= rhs[1])
            
            for i in range(nD):
                model.addConstr(x[i::nD].sum() <= 1)
            
            model.optimize()
            
            status = model.status
            if status == gp.GRB.OPTIMAL:
                x_reduced_global = x.getAttr('X').copy()
                lagrange_multipliers = model.getAttr('Pi')[0:2].copy()
                optimize_result = 1
            else:
                lagrange_multipliers = None
                optimize_result = 0
        else:
            optimize_result = 0
            x_reduced_global = None
            lagrange_multipliers = None
        
        self.comm.Barrier()
        
        optimize_result = self.comm.bcast(optimize_result, root=0)
        if optimize_result == 0:
            feasibility = 0
            x_local = None
            lagrange_multipliers = None
        else:
            feasibility = 1
            
            x_reduced_global = np.array(self.comm.bcast(x_reduced_global, root=0))
            lagrange_multipliers = np.array(self.comm.bcast(lagrange_multipliers, root=0))
            
            x_local = self.cluster.apply_backward_cluster(x_reduced_global)
            lagrange_multipliers = lagrange_multipliers.reshape((-1, 2))
        
        return x_local, lagrange_multipliers, feasibility
    
    def initialize_multicuts(self, rho, constraints, rhs, nD, rho_size, n):
        multicuts_rho = []
        for i in range(self.num_materials):
            multicuts_rho_cut = np.zeros(rho_size)
            for j in range(n):
                multicuts_rho_cut = multicuts_rho_cut + rho[j][i] + 2*j
            multicuts_rho.append(multicuts_rho_cut)
            
        self.cluster.prepare_cluster(multicuts_rho, nD)
        
        quantity_reduced_global = self.cluster.apply_cluster(constraints[0, :])
        trust_region_reduced_global = []
        cut_reduced_global = []
        
        for i in range(n):
            cut_reduced_global.append(self.cluster.apply_cluster(constraints[2*i+1, :]))
            trust_region_reduced_global.append(self.cluster.apply_cluster(constraints[2*i+2, :]))
        
        trust_region_reduced_global = np.array(trust_region_reduced_global)
        cut_reduced_global = np.array(cut_reduced_global)
        
        if self.comm.rank == 0:
            env = gp.Env(params=self.options)
            model = gp.Model(env=env)
            
            x = model.addMVar(nD * self.num_materials, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
            eta = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=-math.inf, ub=math.inf, name="eta")
            
            obj_func = eta
            
            model.setObjective(obj_func, gp.GRB.MINIMIZE)
            
            for i in range(n):
                model.addConstr(cut_reduced_global[i, :] @ x <= rhs[i*2] + eta)
                model.addConstr(trust_region_reduced_global[i, :] @ x <= rhs[i*2+1])
            
            model.addConstr(quantity_reduced_global @ x <= rhs[-1])
            
            model.optimize()
            
            status = model.status
            if status == gp.GRB.OPTIMAL:
                x_reduced_global = x.getAttr('X').copy()
                lagrange_multipliers = model.getAttr('Pi')[0:2*n+1].copy()
                optimize_result = 1
            else:
                lagrange_multipliers = None
                optimize_result = 0
        else:
            optimize_result = 0
            x_reduced_global = None
            lagrange_multipliers = None
            
        self.comm.Barrier()
        
        optimize_result = self.comm.bcast(optimize_result, root=0)
        if optimize_result == 0:
            feasibility = 0
            x_local = None
        else:
            feasibility = 1
            
            x_reduced_global = self.comm.bcast(x_reduced_global, root=0)
            lagrange_multipliers = np.array(self.comm.bcast(lagrange_multipliers, root=0))
            
            x_local = self.cluster.apply_backward_cluster(x_reduced_global)
            lagrange_multipliers = lagrange_multipliers.reshape((-1, 2*n+1))
            
        return x_local, lagrange_multipliers, feasibility
    
    def milp_solution(self, rho, obj, weight, quantity, d):
        if isinstance(obj, float):
            n = 1
        else:
            n = len(obj)
        
        if n == 1:
            rho_size = self.rho_offset[-1]
            rho_global = np.zeros(rho_size*self.num_materials)
            weight_global = np.zeros(rho_size*self.num_materials)
            dQdrho_global = np.zeros(rho_size*self.num_materials)
            for i in range(self.num_materials):
                rho_gather = self.comm.gather(rho[i], root=0)
                weight_gather = self.comm.gather(weight[i], root=0)
                dQdrho_gather = self.comm.gather(self.dQdrho[0, i*self.rho_local_size:(i+1)*self.rho_local_size], root=0)
                if self.comm.rank == 0:
                    rho_global[i*rho_size:(i+1)*rho_size] = np.concatenate(rho_gather)
                    weight_global[i*rho_size:(i+1)*rho_size] = np.concatenate(weight_gather)
                    dQdrho_global[i*rho_size:(i+1)*rho_size] = np.concatenate(dQdrho_gather)
        else:
            rho_size = self.rho_offset[-1]
            rho_global = np.zeros((n, rho_size*self.num_materials))
            weight_global = np.zeros((n, rho_size*self.num_materials))
            dQdrho_global = np.zeros(rho_size*self.num_materials)
            
            for i in range(self.num_materials):
                for j in range(n):
                    rho_gather = self.comm.gather(rho[j][i], root=0)
                    weight_gather = self.comm.gather(weight[j][i], root=0)
                    if self.comm.rank == 0:
                        rho_global[j, i*rho_size:(i+1)*rho_size] = np.concatenate(rho_gather)
                        weight_global[j, i*rho_size:(i+1)*rho_size] = np.concatenate(weight_gather)
                
                dQdrho_gather = self.comm.gather(self.dQdrho[0, i*self.rho_local_size:(i+1)*self.rho_local_size], root=0)
                if self.comm.rank == 0:
                    dQdrho_global[i*rho_size:(i+1)*rho_size] = np.concatenate(dQdrho_gather)
        
        if self.comm.rank == 0:            
            with gp.Env(params=self.options) as env, gp.Model(env=env) as model:
                x = model.addMVar(rho_size*self.num_materials, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
                
                if n == 1:
                    obj_func = weight_global @ x
                    
                    # cut
                    model.setObjective(obj_func, gp.GRB.MINIMIZE)
                    # mass/volume constraint
                    quantity_constraint = dQdrho_global @ x - quantity
                    model.addConstr(quantity_constraint <= 0)
                    # trust region constraint
                    rho_stacked = np.zeros(rho_size)
                    for i in range(self.num_materials):
                        rho_stacked += rho_global[i*rho_size:(i+1)*rho_size]
                    trust_region_coeff = np.repeat((1 - 2*rho_stacked) / self.rho_global_size, self.num_materials)
                    trust_region_const = trust_region_coeff @ x + np.sum(rho_stacked**2) / self.rho_global_size - d
                    model.addConstr(trust_region_const <= 0)
                    # material usage constraint
                    if self.num_materials > 1:
                        for i in range(rho_size):
                            model.addConstr(x[i:self.num_materials*rho_size:rho_size].sum() <= 1)
                else:
                    eta = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=-math.inf, ub=math.inf, name="eta")
                    
                    obj_func = eta
                    model.setObjective(obj_func, gp.GRB.MINIMIZE)
                    
                    for i in range(n):
                        # cut
                        cut_coeff = weight_global[i, :].copy()
                        cut_const = obj[i] - cut_coeff @ rho_global[i, :]
                        model.addConstr(cut_coeff @ x + cut_const <= eta)
                        # trust region constraint
                        rho_stacked = np.zeros(rho_size)
                        for j in range(self.num_materials):
                            rho_stacked += rho_global[i, j*rho_size:(j+1)*rho_size]
                        trust_region_coeff = np.repeat((1 - 2*rho_stacked) / self.rho_global_size, self.num_materials)
                        trust_region_const = trust_region_coeff @ x + np.sum(rho_stacked**2) / self.rho_global_size - d[i]
                        model.addConstr(trust_region_const <= 0)
                    
                    # mass/volume constraint
                    quantity_constraint = dQdrho_global @ x - quantity
                    model.addConstr(quantity_constraint <= 0)
                    # material usage constraint
                    if self.num_materials > 1:
                        for i in range(rho_size):
                            model.addConstr(x[i:self.num_materials*rho_size:rho_size].sum() <= 1)
                
                model.optimize()
            
                status = model.status
                if status == gp.GRB.OPTIMAL:
                    rho_global_new = x.getAttr('X').copy()
                    if n == 1:
                        lagrange_multipliers = model.getAttr('Pi')[0:2].copy()
                    else:
                        lagrange_multipliers = model.getAttr('Pi')[0:2*n+1].copy()
                    cost = model.objVal
                    optimize_result = 1
                else:
                    optimize_result = 0
        else:
            optimize_result = 0
            cost = None
            rho_global_new = None
            lagrange_multipliers = None
        
        self.comm.Barrier()
        
        optimize_result = self.comm.bcast(optimize_result, root=0)
        if optimize_result == 0:
            rho_new = None
            cost = None
            lagrange_multipliers = None
        else:
            rho_global_new = self.comm.bcast(rho_global_new, root=0)
            lagrange_multipliers = np.array(self.comm.bcast(lagrange_multipliers, root=0))
            
            rho_new = np.zeros(self.rho_local_size*self.num_materials)
            for i in range(self.num_materials):
                rho_new[i*self.rho_local_size:(i+1)*self.rho_local_size] = rho_global_new[self.rho_offset[self.comm.rank]+self.rho_local_size*i:self.rho_offset[self.comm.rank+1]+self.rho_local_size*i].copy()
            
            if n == 1:
                if self.comm.rank == 0:
                    cost = np.array([np.dot(weight_global, rho_global_new-rho_global)], dtype='d')
                self.comm.Barrier()
                cost = self.comm.bcast(cost, root=0)
                cost = obj + cost[0]
            else:
                cost = self.comm.bcast(cost, root=0)
        
        return rho_new, cost, lagrange_multipliers
    
    def master_problem_single_cut(self, obj_coeff, constraint_coeff, rhs, nD):
        n = len(self.rho_list)
        
        obj_local = np.zeros(n*nD, dtype='d')
        constraints_local = np.zeros((2, n*nD), dtype='d')
        
        for i in range(n):
            for j in range(self.num_materials):
                for k in range(len(self.cluster.offset_source)):
                    offset_source = i*nD + self.cluster.offset_source[k]
                    offset_target = self.cluster.offset_target[k]+self.rho_local_size*j
                    obj_local[offset_source] += np.dot(obj_coeff[offset_target], self.rho_list[i][offset_target])
                    constraints_local[:, offset_source] += np.dot(constraint_coeff[:, offset_target], self.rho_list[i][offset_target])
        
        obj_global = self.comm.reduce(obj_local, root=0)
        constraints_global = self.comm.reduce(constraints_local, root=0)
        
        start = MPI.Wtime()
        if self.comm.rank == 0:
            with gp.Env(params=self.options) as env, gp.Model(env=env) as model:
                x = model.addMVar(n*nD, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
                
                obj_func = obj_global @ x
                
                model.setObjective(obj_func, gp.GRB.MINIMIZE)
                
                for i in range(2):
                    model.addConstr(constraints_global[i, :] @ x <= rhs[i])
                
                row_nD = np.zeros(n*nD)
                col_nD = np.zeros(n*nD)
                val_nD = np.ones(n*nD)
                rhs_nD = np.ones(nD)
                
                for i in range(nD):
                    row_nD[n*i:n*(i+1)] = i
                    col_nD[n*i:n*(i+1)] = np.arange(i, n*nD, nD)
                
                mat = sp.csr_matrix((val_nD, (row_nD, col_nD)), shape=(nD, n*nD))
                
                model.addConstr(mat @ x <= rhs_nD)
                
                model.optimize()
                
                status = model.status
                if status == gp.GRB.OPTIMAL:
                    lagrange_multipliers = model.getAttr('Pi')[0:2].copy()
                    optimize_result = 1
                else:
                    optimize_result = 0
        else:
            optimize_result = 0
            lagrange_multipliers = None
            
        self.comm.Barrier()
        end = MPI.Wtime()
        if self.comm.rank == 0:
            self.master_milp_problem_time += end - start
        lagrange_multipliers = np.array(self.comm.bcast(lagrange_multipliers, root=0))
        
        return lagrange_multipliers
    
    def master_problem_multicuts(self, obj, constraint_coeff, rhs, nD, num_cuts):
        n = len(self.rho_list)
        
        constraints_local = np.zeros((2*num_cuts+1, n*nD), dtype='d')
        
        for i in range(n):
            for m in range(self.num_materials):
                for j in range(len(self.cluster.offset_source)):
                    offset_source = i*nD + self.cluster.offset_source[j]
                    offset_target = self.cluster.offset_target[j]+self.rho_local_size*m
                    for k in range(num_cuts):
                        constraints_local[2*k, offset_source] += np.dot(constraint_coeff[k*2, offset_target], self.rho_list[i][offset_target])
                        constraints_local[2*k+1, offset_source] += np.dot(constraint_coeff[k*2+1, offset_target], self.rho_list[i][offset_target])
                    constraints_local[-1, offset_source] += np.dot(constraint_coeff[-1, offset_target], self.rho_list[i][offset_target])
        
        constraints_global = self.comm.reduce(constraints_local, root=0)
        
        start = MPI.Wtime()
        if self.comm.rank == 0:            
            with gp.Env(params=self.options) as env, gp.Model(env=env) as model:
                x = model.addMVar(n*nD, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
                eta = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=-math.inf, ub=math.inf, name="eta")
                
                obj_func = eta
                
                model.setObjective(eta, gp.GRB.MINIMIZE)
                
                # cuts and trust region constraint
                for i in range(num_cuts):
                    model.addConstr(constraints_global[2*i, :] @ x - eta <= rhs[2*i])
                    model.addConstr(constraints_global[2*i+1, :] @ x <= rhs[2*i+1])
                # mass/volume constraint
                model.addConstr(constraints_global[-1, :] @ x <= rhs[-1])
                
                row_nD = np.zeros(n*nD)
                col_nD = np.zeros(n*nD)
                val_nD = np.ones(n*nD)
                rhs_nD = np.ones(nD)
                
                for i in range(nD):
                    row_nD[n*i:n*(i+1)] = i
                    col_nD[n*i:n*(i+1)] = np.arange(i, n*nD, nD)
                
                mat = sp.csr_matrix((val_nD, (row_nD, col_nD)), shape=(nD, n*nD))
                
                model.addConstr(mat @ x <= rhs_nD)
                
                model.optimize()
                
                status = model.status
                if status == gp.GRB.OPTIMAL:
                    lagrange_multipliers = model.getAttr('Pi')[0:2*num_cuts+1].copy()
                    optimize_result = 1
                else:
                    optimize_result = 0
        else:
            optimize_result = 0
            lagrange_multipliers = None
        
        self.comm.Barrier()
        end = MPI.Wtime()
        if self.comm.rank == 0:
            self.master_milp_problem_time += end - start
        lagrange_multipliers = np.array(self.comm.bcast(lagrange_multipliers, root=0))
        
        return lagrange_multipliers
    
    def subproblem_single_cut(self, obj, obj0, constraints, lagrange_multipliers, rho):
        obj_func = (obj - lagrange_multipliers @ constraints) @ self.x
        self.sub_problem_model.setObjective(obj_func, gp.GRB.MINIMIZE)
        self.sub_problem_model.update()
        self.sub_problem_model.optimize()
        
        rho_new = self.x.getAttr('X').copy()
        
        cost = np.array([np.dot(obj, rho_new-rho)], dtype='d')
        self.comm.Allreduce(MPI.IN_PLACE, cost, op=MPI.SUM)
        
        cost = obj0 + cost[0]
        
        if self.subproblem_quantum_simulated:
            start_time = MPI.Wtime()
            if self.num_materials == 1:
                qubo_linear = obj - lagrange_multipliers @ constraints
            else:
                qubo_linear = obj - lagrange_multipliers @ constraints
                qubo_quadratic_row = np.zeros((self.rho_local_size*self.num_materials))
            end_time = MPI.Wtime()
            self.sub_problem_casting_time += end_time - start_time
        
        return rho_new, cost
    
    def subproblem_multicuts(self, obj0, constraints, lagrange_multipliers, rho, n):
        obj_func = (-lagrange_multipliers @ constraints) @ self.x
        self.sub_problem_model.setObjective(obj_func, gp.GRB.MINIMIZE)
        self.sub_problem_model.update()
        self.sub_problem_model.optimize()
        
        rho_new = self.x.getAttr('X').copy()
        
        cuts = constraints[::2]
        cost_local = np.zeros(n)
        for i in range(n):
            cost_local[i] = np.dot(cuts[i], rho_new - rho[i])
        
        self.comm.Barrier()
        
        cost = self.comm.allreduce(cost_local)
        cost = obj0 + cost
        
        cost = max(cost)
        
        if self.subproblem_quantum_simulated:
            start_time = MPI.Wtime()
            qubo_linear = obj_func.copy()
            if self.num_materials > 1:
                pass
            end_time = MPI.Wtime()
            self.sub_problem_casting_time += end_time - start_time
        
        return rho_new, cost