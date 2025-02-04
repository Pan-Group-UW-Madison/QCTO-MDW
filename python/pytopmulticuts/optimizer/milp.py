from mpi4py import MPI
import numpy as np
import math

import gurobipy as gp

from .optimizer import SubOptimizer

class MilpOptimizer(SubOptimizer):
    def __init__(self, problem):
        super().__init__(problem)
        
        rho_local_size = problem.rho_field.x.petsc_vec.array.size
        rho_size_by_rank = self.comm.allgather(rho_local_size)
        
        self.rho_offset = np.cumsum([0] + rho_size_by_rank)
        
        self.options = {
            "outputflag": 0,
            "MIPGap": 1e-9,
            "FeasibilityTol": 1e-9,
            "OptimalityTol": 1e-9,
            "TimeLimit": 100,
            "Threads": 1,
        }
        
    def update(self, rho, obj, weight, vol_frac, d):
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
            run = 1            
            while run <= 2:
                with gp.Env(params=self.options) as env, gp.Model(env=env) as model:
                    # Formulate problem
                    # single cut
                    if run == 1:
                        x = model.addMVar(rho_size, vtype=gp.GRB.BINARY, lb=0, ub=1, name="x")
                    else:
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
                        volume_coeff = np.ones(rho_size)
                        volume_const = volume_coeff @ x - vol_frac*rho_size
                        model.addConstr(volume_const <= 0)
                    
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
        
        optimize_result = self.comm.bcast(optimize_result, root=0)
        if optimize_result == 0:
            return None, math.inf
        rho_global_new = self.comm.bcast(rho_global_new, root=0)
        
        rho_new = rho_global_new[self.rho_offset[self.comm.rank]:self.rho_offset[self.comm.rank+1]]
        
        if n == 1:
            cost = np.array([np.dot(weight, rho_new-rho)], dtype='d')
            self.comm.Allreduce(MPI.IN_PLACE, cost, op=MPI.SUM)
            cost = obj + cost[0]
        else:
            cost = self.comm.bcast(cost, root=0)
        
        return rho_new, cost