from mpi4py import MPI
import numpy as np
import math

import gurobipy as gp

from .optimizer import SubOptimizer

class MilpOptimizer(SubOptimizer):
    def __init__(self, problem, num_free_size, num_local_size, num_materials=1, solver_type="classical"):
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
            "TimeLimit": 300,
            "Threads": 1,
        }
    
    def set_dQdrho(self, dQdrho):
        self.dQdrho = np.zeros((1, self.rho_local_size*self.num_materials))
        for i in range(self.num_materials):
            self.dQdrho[0, i*self.rho_local_size:(i+1)*self.rho_local_size] = dQdrho[i].copy()
        
    def update(self, rho, obj, weight, quantity, d):
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
        
        start_time = MPI.Wtime()
        if self.comm.rank == 0:            
            run = 1            
            while run <= 2:
                with gp.Env(params=self.options) as env, gp.Model(env=env) as model:
                    # Formulate problem
                    # single cut
                    if run == 1:
                        x = model.addMVar(rho_size*self.num_materials, vtype=gp.GRB.BINARY, lb=0, ub=1, name="x")
                    else:
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
                        # material usage constraint
                        model.addConstr(trust_region_const <= 0)
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
                        cost = model.objVal
                        optimize_result = 1
                        break
                    elif run == 1:
                        run = 2
                    else:
                        optimize_result = 0
                        break
        else:
            optimize_result = 0
            cost = None
            rho_global_new = None
        
        self.comm.Barrier()
        end_time = MPI.Wtime()
        
        optimize_result = self.comm.bcast(optimize_result, root=0)
        if optimize_result == 0:
            rho_new = None
            cost = None
        else:
            rho_global_new = self.comm.bcast(rho_global_new, root=0)
            
            rho_new = rho_global_new[self.rho_offset[self.comm.rank]:self.rho_offset[self.comm.rank+1]]
            
            if n == 1:
                if self.comm.rank == 0:
                    cost = np.array([np.dot(weight_global, rho_global_new-rho_global)], dtype='d')
                self.comm.Barrier()
                cost = self.comm.bcast(cost, root=0)
                cost = obj + cost[0]
            else:
                cost = self.comm.bcast(cost, root=0)
        
        if self.comm.rank == 0:
            print("  MILP optimization time: {0:.2f} s".format(end_time-start_time))
        
        return rho_new, cost