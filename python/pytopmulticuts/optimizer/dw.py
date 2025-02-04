from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
import math

import gurobipy as gp

from .optimizer import SubOptimizer

class DWOptimizer(SubOptimizer):
    def __init__(self, problem, num_divisions=1, solver_type="classical"):
        super().__init__(problem)
        
        self.rho_local_size = problem.rho_field.x.petsc_vec.array.size
        self.rho_global_size = int(self.comm.allreduce(self.rho_local_size))
        
        rho_size_by_rank = self.comm.allgather(self.rho_local_size)        
        self.rho_offset = np.cumsum([0] + rho_size_by_rank)
        
        self.options = {
            "outputflag": 0,
            "MIPGap": 1e-9,
            "FeasibilityTol": 1e-9,
            "OptimalityTol": 1e-9,
            "Threads": 1
        }
        
        self.env = gp.Env(params=self.options)
        self.sub_problem_model = gp.Model(env=self.env)
        
        self.x = self.sub_problem_model.addMVar((self.rho_local_size, ), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
        
        rho_field = self.problem.rho_field
        num_elems = rho_field.x.petsc_vec.array.size
        self.centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems]
        
        self.nD = num_divisions
        
        if solver_type == "classical":
            self.subproblem_quantum_simulated = False
        elif solver_type == "quantum-simulated-subproblem":
            self.subproblem_quantum_simulated = True
        
    def update(self, rho, obj, weight, vol_frac, d):
        if isinstance(obj, float):
            n = 1
        else:
            n = len(obj)
        
        self.lagrange_multiplier_list = []        
        self.rho_list = []
        self.cost_list = []
        
        nD = self.nD
        if n == 1:
            obj_coeff_original = weight
            volume_coeff = np.ones(self.rho_local_size) / self.rho_global_size
            trust_region_coeff = 1 - 2*rho
            
            constraints_coeff = np.vstack([volume_coeff, trust_region_coeff])
            rho_total = self.comm.allreduce(np.sum(rho**2))
            rhs = [vol_frac, -rho_total + d*self.rho_global_size]
        else:
            volume_coeff = np.ones(self.rho_local_size) / self.rho_global_size
            
            constraints_coeff = []
            rhs = []
            
            for i in range(n):
                cut_coeff = weight[i]
                trust_region_coeff = 1 - 2*rho[i]
                
                constraints_coeff.append(cut_coeff)
                constraints_coeff.append(trust_region_coeff)
                
                rhs_local = weight[i]@rho[i]
                rhs_local = self.comm.allreduce(rhs_local)
                rhs.append(-obj[i]+rhs_local)
                rho_total = self.comm.allreduce(np.sum(rho[i]**2))
                rhs.append(-rho_total + d[i]*self.rho_global_size)
            
            constraints_coeff.append(volume_coeff)
            rhs.append(vol_frac)
            
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
                nD *= 2
        self.lagrange_multiplier_list.append(lagrange_multipliers_init)
        self.rho_list.append(rho_init)
        
        end_time = MPI.Wtime()
        if self.comm.rank == 0:
            print(f"  Initialization time: {end_time - start_time:.4f} s, nD: {nD}", flush=True)
        
        sub_problem_time = 0
        master_problem_time = 0
        
        if self.subproblem_quantum_simulated:
            self.sub_problem_casting_time = 0
        
        for i in range(30):
            start_time = MPI.Wtime()
            if n == 1:
                rho_dw, cost = self.subproblem_single_cut(obj_coeff_original, obj, constraints_coeff, self.lagrange_multiplier_list[-1], rho)
            else:
                rho_dw, cost = self.subproblem_multicuts(obj, constraints_coeff, self.lagrange_multiplier_list[-1], rho, n)
            end_time = MPI.Wtime()
            sub_problem_time += end_time - start_time
            self.rho_list.append(rho_dw)
            self.cost_list.append(cost)
            
            start_time = MPI.Wtime()
            if n == 1:
                lagrange_multipliers = self.master_problem_single_cut(obj_coeff_original, constraints_coeff, rhs, nD)
            else:
                lagrange_multipliers = self.master_problem_multicuts(obj, constraints_coeff, rhs, nD, n)
            end_time = MPI.Wtime()
            master_problem_time += end_time - start_time
            
            vol = np.sum(rho_dw)
            vol = self.comm.allreduce(vol)
            vol /= self.rho_global_size
            
            stop_criteria = 0
            for j in range(i):
                lagrange_diff = np.linalg.norm(lagrange_multipliers - self.lagrange_multiplier_list[j])
                if lagrange_diff / np.linalg.norm(lagrange_multipliers) < 1e-9:
                    stop_criteria = 1
                    rho_new = self.rho_list[j]
                    cost = self.cost_list[j]
                    break
            
            if stop_criteria == 1:
                if self.comm.rank == 0:
                    print(f"  Converged at iteration {i}", flush=True)
                    print(f"  Subproblem time: {sub_problem_time:.4f} s", flush=True)
                    print(f"  Master problem time: {master_problem_time:.4f} s", flush=True)
                    
                    if self.subproblem_quantum_simulated:
                        print(f"  Construction time of quantum subproblem: {self.sub_problem_casting_time:.4f} s", flush=True)
                break
            
            self.lagrange_multiplier_list.append(lagrange_multipliers)
        
        return rho_dw, cost
    
    def calculate_offset(self, rho, nD):
        self.offset = []
        unique_rho_value = np.unique(rho)
        unique_rho = []
        
        for i in range(unique_rho_value.size):
            unique_rho.append(np.where(rho == unique_rho_value[i])[0])
        
        unique_rho = np.concatenate(unique_rho)
        
        offset = np.linspace(0, unique_rho.size, nD+1, dtype=int)
        for i in range(nD):
            self.offset.append(unique_rho[offset[i]:offset[i+1]])
    
    def initialize_single_cut(self, obj, rho, constraints, rhs, nD, rho_size):
        self.calculate_offset(rho, nD)
        
        obj_reduced_local = np.zeros(nD)
        volume_reduced_local = np.zeros(nD)
        trust_region_reduced_local = np.zeros(nD)
        
        for i in range(nD):
            obj_reduced_local[i] = obj[self.offset[i]].sum()
            volume_reduced_local[i] = constraints[0][self.offset[i]].sum()
            trust_region_reduced_local[i] = constraints[1][self.offset[i]].sum()
        
        obj_reduced_global = self.comm.gather(obj_reduced_local, root=0)
        volume_reduced_global = self.comm.gather(volume_reduced_local, root=0)
        trust_region_reduced_global = self.comm.gather(trust_region_reduced_local, root=0)
        
        if self.comm.rank == 0:
            env = gp.Env(params=self.options)
            model = gp.Model(env=env)
            
            x = model.addMVar(nD * self.comm.size, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
            
            obj_reduced_global = np.concatenate(obj_reduced_global)
            obj_func = obj_reduced_global @ x
            
            model.setObjective(obj_func, gp.GRB.MINIMIZE)
            volume_reduce_global = np.concatenate(volume_reduced_global)
            trust_region_reduced_global = np.concatenate(trust_region_reduced_global)
            
            model.addConstr(volume_reduce_global @ x <= rhs[0])
            model.addConstr(trust_region_reduced_global @ x <= rhs[1])
            
            model.optimize()
            
            status = model.status
            if status == gp.GRB.OPTIMAL:
                x_reduced_global = x.getAttr('X').copy()
                lagrange_multipliers = model.getAttr('Pi').copy()
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
            
            x_reduced_local = x_reduced_global[self.comm.rank*nD:(self.comm.rank+1)*nD]
            x_local = np.zeros(rho_size)
            for i in range(nD):
                x_local[self.offset[i]] = x_reduced_local[i]
            lagrange_multipliers = lagrange_multipliers.reshape((-1, 2))
        
        return x_local, lagrange_multipliers, feasibility
    
    def initialize_multicuts(self, rho, constraints, rhs, nD, rho_size, n):
        multicuts_rho = np.zeros(rho_size)
        for i in range(n):
            multicuts_rho += rho[i] + 2 * i
        
        self.calculate_offset(multicuts_rho, nD)
        
        volume_reduced_local = np.zeros(nD)
        trust_region_reduced_local = np.zeros((nD, n))
        cut_reduced_local = np.zeros((nD, n))
        
        for i in range(nD):
            for j in range(n):
                trust_region_reduced_local[i, j] = constraints[j*2+1][self.offset[i]].sum()
                cut_reduced_local[i, j] = constraints[j*2][self.offset[i]].sum()
            
            volume_reduced_local[i] = constraints[-1][self.offset[i]].sum()
        
        volume_reduced_global = self.comm.gather(volume_reduced_local, root=0)
        trust_region_reduced_global = self.comm.gather(trust_region_reduced_local, root=0)
        cut_reduced_global = self.comm.gather(cut_reduced_local, root=0)
        
        self.comm.Barrier()
        
        if self.comm.rank == 0:
            env = gp.Env(params=self.options)
            model = gp.Model(env=env)
            
            x = model.addMVar(nD * self.comm.size, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
            eta = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=-math.inf, ub=math.inf, name="eta")
            
            obj_func = eta
            
            model.setObjective(obj_func, gp.GRB.MINIMIZE)
            
            volume_reduced_global = np.concatenate(volume_reduced_global)
            trust_region_reduced_global = np.concatenate(trust_region_reduced_global)
            cut_reduced_global = np.concatenate(cut_reduced_global)
            
            for i in range(n):
                model.addConstr(cut_reduced_global[:, i] @ x <= rhs[i*2] + eta)
                model.addConstr(trust_region_reduced_global[:, i] @ x <= rhs[i*2+1])
            
            model.addConstr(volume_reduced_global @ x <= rhs[-1])
            
            model.optimize()
            
            status = model.status
            if status == gp.GRB.OPTIMAL:
                x_reduced_global = x.getAttr('X').copy()
                lagrange_multipliers = model.getAttr('Pi').copy()
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
            
            x_reduced_local = x_reduced_global[self.comm.rank*nD:(self.comm.rank+1)*nD]
            x_local = np.zeros(rho_size)
            for i in range(nD):
                x_local[self.offset[i]] = x_reduced_local[i]
            lagrange_multipliers = lagrange_multipliers.reshape((-1, 2*n+1))
            
        return x_local, lagrange_multipliers, feasibility
    
    def milp_solution(self, rho, obj, weight, vol_frac, d):
        if isinstance(obj, float):
            n = 1
        else:
            n = len(obj)
        
        if n == 1:        
            rho_global = self.comm.gather(rho, root=0)
            weight_global = self.comm.gather(weight, root=0)
            
            if self.comm.rank == 0:
                rho_global = np.concatenate(rho_global)
                weight_global = np.array(np.concatenate(weight_global))
                rho_size = rho_global.size
        else:
            rho_global = []
            weight_global = []
            
            for i in range(n):
                rho_global_cut = self.comm.gather(rho[i], root=0)
                weight_global_cut = self.comm.gather(weight[i], root=0)
                
                if self.comm.rank == 0:
                    rho_global_cut = np.concatenate(rho_global_cut)
                    weight_global_cut = np.array(np.concatenate(weight_global_cut))
                    
                    rho_global.append(rho_global_cut)
                    weight_global.append(weight_global_cut)
                
                self.comm.Barrier()
            
            if self.comm.rank == 0:
                rho_size = len(rho_global[0])
        
        if self.comm.rank == 0:            
            with gp.Env(params=self.options) as env, gp.Model(env=env) as model:
                x = model.addMVar(rho_size, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
                
                if n == 1:
                    obj_func = weight_global @ x
                    
                    # cut
                    model.setObjective(obj_func, gp.GRB.MINIMIZE)
                    # mass/volume constraint
                    volume_coeff = np.ones(rho_size)
                    volume_const = volume_coeff @ x - vol_frac*rho_size
                    model.addConstr(volume_const <= 0)
                    # trust region constraint
                    trust_region_coeff = 1 - 2*rho_global
                    trust_region_const = trust_region_coeff @ x + np.sum(rho_global**2) - d*rho_size
                    model.addConstr(trust_region_const <= 0)
                else:
                    eta = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=-math.inf, ub=math.inf, name="eta")
                    
                    obj_func = eta
                    model.setObjective(obj_func, gp.GRB.MINIMIZE)
                    
                    for i in range(n):
                        # cut
                        cut_coeff = weight_global[i].copy()
                        cut_const = obj[i] - cut_coeff @ rho_global[i]
                        model.addConstr(cut_coeff @ x + cut_const <= eta)
                        # trust region constraint
                        trust_region_coeff = 1 - 2*rho_global[i]
                        trust_region_const = trust_region_coeff @ x + np.sum(rho_global[i]**2) - d[i]*rho_size
                        model.addConstr(trust_region_const <= 0)
                    
                    # mass/volume constraint
                    volume_coeff = np.ones(rho_size)/rho_size
                    volume_const = volume_coeff @ x - vol_frac
                    model.addConstr(volume_const <= 0)
                
                model.optimize()
            
                status = model.status
                if status == gp.GRB.OPTIMAL:
                    rho_global_new = x.getAttr('X').copy()
                    lagrange_multipliers = model.getAttr('Pi').copy()
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
            exit(1)
        rho_global_new = self.comm.bcast(rho_global_new, root=0)
        lagrange_multipliers = np.array(self.comm.bcast(lagrange_multipliers, root=0))
        
        rho_new = rho_global_new[self.rho_offset[self.comm.rank]:self.rho_offset[self.comm.rank+1]]
        
        if n == 1:
            cost = np.array([np.dot(weight, rho_new-rho)], dtype='d')
            self.comm.Allreduce(MPI.IN_PLACE, cost, op=MPI.SUM)
            cost = obj + cost[0]
        else:
            cost = self.comm.bcast(cost, root=0)
        
        return rho_new, cost, lagrange_multipliers
    
    def master_problem_single_cut(self, obj_coeff, constraint_coeff, rhs, nD):
        n = len(self.rho_list)
        
        obj_local = np.zeros(n*nD, dtype='d')
        constraints_local = np.zeros((2, n*nD), dtype='d')
        
        for i in range(nD):
            for j in range(n):
                obj_local[i*n + j] += np.dot(obj_coeff[self.offset[i]], self.rho_list[j][self.offset[i]])
                constraints_local[:, i*n + j] = np.dot(constraint_coeff[:, self.offset[i]], self.rho_list[j][self.offset[i]])
        
        obj_global = self.comm.gather(obj_local, root=0)
        constraints_global = self.comm.gather(constraints_local, root=0)
        
        if self.comm.rank == 0:
            obj_global = np.concatenate(obj_global)
            
            with gp.Env(params=self.options) as env, gp.Model(env=env) as model:
                x = model.addMVar(n*nD*self.comm.size, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
                
                obj_func = obj_global @ x
                
                model.setObjective(obj_func, gp.GRB.MINIMIZE)
                
                constraints_global_reshape = np.zeros((2, n*nD*self.comm.size))
                for i in range(self.comm.size):
                    constraints_global_reshape[:, i*n*nD:(i+1)*n*nD] = constraints_global[i]
                for i in range(2):
                    model.addConstr(constraints_global_reshape[i, :] @ x <= rhs[i])
                
                row_nD = np.zeros(n*nD*self.comm.size)
                col_nD = np.zeros(n*nD*self.comm.size)
                val_nD = np.ones(n*nD*self.comm.size)
                rhs_nD = np.ones(nD*self.comm.size)
                
                for i in range(nD*self.comm.size):
                    row_nD[n*i:n*(i+1)] = i
                    col_nD[n*i:n*(i+1)] = np.arange(i*n, (i+1)*n)
                
                mat = sp.csr_matrix((val_nD, (row_nD, col_nD)), shape=(nD*self.comm.size, n*nD*self.comm.size))
                
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
        lagrange_multipliers = np.array(self.comm.bcast(lagrange_multipliers, root=0))
        
        return lagrange_multipliers
    
    def master_problem_multicuts(self, obj, constraint_coeff, rhs, nD, num_cuts):
        n = len(self.rho_list)
        
        constraints_local = np.zeros((2*num_cuts+1, n*nD), dtype='d')
        
        for i in range(nD):
            for j in range(n):
                for k in range(num_cuts):
                    constraints_local[2*k, i*n + j] = np.dot(constraint_coeff[k*2, self.offset[i]], self.rho_list[j][self.offset[i]])
                    constraints_local[2*k+1, i*n + j] = np.dot(constraint_coeff[k*2+1, self.offset[i]], self.rho_list[j][self.offset[i]])
                constraints_local[-1, i*n + j] = np.dot(constraint_coeff[-1, self.offset[i]], self.rho_list[j][self.offset[i]])
        
        constraints_global = self.comm.gather(constraints_local, root=0)
        
        if self.comm.rank == 0:            
            with gp.Env(params=self.options) as env, gp.Model(env=env) as model:
                x = model.addMVar(n*nD*self.comm.size, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
                eta = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=-math.inf, ub=math.inf, name="eta")
                
                obj_func = eta
                
                model.setObjective(eta, gp.GRB.MINIMIZE)
                
                constraints_global_reshape = np.zeros((2*num_cuts+1, n*nD*self.comm.size))
                for i in range(self.comm.size):
                    constraints_global_reshape[:, i*n*nD:(i+1)*n*nD] = constraints_global[i]
                
                for i in range(num_cuts):
                    model.addConstr(constraints_global_reshape[2*i, :] @ x - eta <= rhs[2*i])
                    model.addConstr(constraints_global_reshape[2*i+1, :] @ x <= rhs[2*i+1])
                model.addConstr(constraints_global_reshape[-1, :] @ x <= rhs[-1])
                
                row_nD = np.zeros(n*nD*self.comm.size)
                col_nD = np.zeros(n*nD*self.comm.size)
                val_nD = np.ones(n*nD*self.comm.size)
                rhs_nD = np.ones(nD*self.comm.size)
                
                for i in range(nD*self.comm.size):
                    row_nD[n*i:n*(i+1)] = i
                    col_nD[n*i:n*(i+1)] = np.arange(i*n, (i+1)*n)
                
                mat = sp.csr_matrix((val_nD, (row_nD, col_nD)), shape=(nD*self.comm.size, n*nD*self.comm.size))
                
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
        lagrange_multipliers = np.array(self.comm.bcast(lagrange_multipliers, root=0))
        
        return lagrange_multipliers
    
    def subproblem_single_cut(self, obj, obj0, constraints, lagrangian_multipliers, rho):
        obj_func = (obj - lagrangian_multipliers @ constraints) @ self.x
        self.sub_problem_model.setObjective(obj_func, gp.GRB.MINIMIZE)
        self.sub_problem_model.update()
        self.sub_problem_model.optimize()
        
        rho_new = self.x.getAttr('X').copy()
        
        cost = np.array([np.dot(obj, rho_new-rho)], dtype='d')
        self.comm.Allreduce(MPI.IN_PLACE, cost, op=MPI.SUM)
        
        cost = obj0 + cost[0]
        
        num_material = 1
        
        if self.subproblem_quantum_simulated:
            start_time = MPI.Wtime()
            if num_material == 1:
                qubo_linear = obj_func.copy()
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
        
        num_material = 1
        
        if self.subproblem_quantum_simulated:
            start_time = MPI.Wtime()
            if num_material == 1:
                qubo_linear = obj_func.copy()
            end_time = MPI.Wtime()
            self.sub_problem_casting_time += end_time - start_time
        
        return rho_new, cost