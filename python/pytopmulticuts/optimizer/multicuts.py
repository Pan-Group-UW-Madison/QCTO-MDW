from .optimizer import Optimizer
from .filter import RadiusFilter
from .sensitivity import Sensitivity

from .milp import MilpOptimizer
from .dw import DWOptimizer

import time
import numpy as np
import math
from itertools import combinations
from mpi4py import MPI

class Cuts:
    def __init__(self):
        self.solution = [[] for _ in range(6)]
        self.lambdas_history = [[] for _ in range(6)]
        self.x_history = [[] for _ in range(6)]
        self.c0_history = []
        self.weight_history = []
        self.d_history = []
        self.use_flag_history = [[] for _ in range(6)]
        
        self.num_cuts = int(0)
    
    def add_solution(self, x, obj_func, c0, weight, d, lambdas = None):
        if lambdas is None:
            lambdas = [self.num_cuts]
            self.c0_history.append(c0)
            self.weight_history.append(weight)
            self.d_history.append(d)
            
            self.num_cuts += 1
        n = len(lambdas) - 1
        
        self.solution[n].append(obj_func)
        self.lambdas_history[n].append(lambdas)
        self.x_history[n].append(x)
        self.use_flag_history[n].append(False)
    
    def check_flag(self, lambdas):
        n = len(lambdas) - 1
        
        for i, lamb in enumerate(self.lambdas_history[n]):
            if np.linalg.norm(np.array(lamb) - np.array(lambdas)) < 1e-6:
                return self.use_flag_history[n][i]
        
        return False

    def check_solution(self, lambdas):
        n = len(lambdas) - 1
        
        for i, lamb in enumerate(self.lambdas_history[n]):
            if np.linalg.norm(np.array(lamb) - np.array(lambdas)) < 1e-6:
                return True, self.x_history[n][i], self.solution[n][i]
        
        return False, None, None
    
    def estimate_cost(self, lambdas):
        n = len(lambdas)
        
        estimated_cost = math.inf
        
        for i in range(1, n):
            candidates = list(combinations(lambdas, i))
            for candidate in candidates:
                flag, x, cost = self.check_solution(candidate)
                if flag == True:
                    if estimated_cost == math.inf:
                        estimated_cost = cost
                    else:
                        estimated_cost = max(estimated_cost, cost)
        
        return estimated_cost

    def flag_solution(self, lambdas):
        n = len(lambdas) - 1
        
        for i, lamb in enumerate(self.lambdas_history[n]):
            if np.linalg.norm(np.array(lamb) - np.array(lambdas)) < 1e-6:
                self.use_flag_history[n][i] = True
                break
    
    def get_milp_operator(self, lambdas):
        x = []
        c0 = []
        weights = []
        d = []
        for lamb in lambdas:
            x.append(self.x_history[0][lamb])
            c0.append(self.c0_history[lamb])
            weights.append(self.weight_history[lamb])
            d.append(self.d_history[lamb])
        return x, c0, weights, d
    
    def clear(self):
        self.solution = [[] for _ in range(6)]
        self.lambdas_history = [[] for _ in range(6)]
        self.x_history = [[] for _ in range(6)]
        self.c0_history = []
        self.weight_history = []
        self.d_history = []
        self.use_flag_history = [[] for _ in range(6)]
        
        self.num_cuts = int(0)

class MulticutsOptimizer(Optimizer):
    def __init__(self, descriptor, problem):
        super().__init__(descriptor, problem)
        
        self.optimizer_name = "multicuts-" + descriptor["subproblem_solver"]
        
        self.max_iter = descriptor["max_iter"]
        self.opt_tol = descriptor["opt_tol"]
        self.vol_frac = descriptor["vol_frac"]
        self.radius = descriptor["filter_radius"]
        self.d = descriptor["initial_trust_region"]
        
        self.verbose = 1
        
        num_consts = 1 if self.problem.objective == "compliance" else 2
        
        rho_field = self.problem.rho_field
        num_elems = rho_field.x.petsc_vec.array.size
        centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
        solid, void = descriptor["solid_zone"](centers), descriptor["void_zone"](centers)
        rho_ini = np.full(num_elems, descriptor["vol_frac"])
        rho_ini[solid], rho_ini[void] = 1.0, 1e-4
        rho_field.x.petsc_vec.array[:] = rho_ini
        
        self.free = np.where((solid == False) & (void == False))
        self.num_free = len(self.free[0])
        
        num_fixed = num_elems - self.num_free
        num_fixed_global = self.comm.allreduce(num_fixed, op=MPI.SUM)
        num_elems_global = self.comm.allreduce(num_elems, op=MPI.SUM)
        
        self.vol_frac -= num_fixed_global / num_elems_global
        
        self.num_fem = 0
        
        self.cuts = Cuts()
        
        if descriptor["subproblem_solver"] == "milp":
            self.sub_optimizer = MilpOptimizer(problem)
        elif descriptor["subproblem_solver"] == "dw":
            self.sub_optimizer = DWOptimizer(problem, self.num_free, num_elems, descriptor["num_divisions"], descriptor["solver_type"])
        else:
            raise ValueError("Invalid subproblem_solver")
            exit(1)
        
    def solve_prime(self):
        self.problem.solve_prime()
        
        [C_value, V_value, U_value], [dCdrho, dVdrho, dUdrho] = self.sens_problem.evaluate()
        if self.problem.objective == "compliance":
            dJdrho = dCdrho
        else:
            dJdrho = dUdrho
        
        dJdrho = dJdrho.array * (self.problem.rho_field.x.petsc_vec.array + self.problem.eps.value)
        dJdrho = self.sens_filter.filter(dJdrho)
        
        self.problem.sensitivity.x.petsc_vec.array = dJdrho.copy()
        
        self.num_fem += 1
            
        return C_value, V_value, dJdrho
    
    def update_trust_region(self, c0, c, cost, d):
        omega = (c0 - c) / (c0 - cost)
        if omega < 1 and omega >= 0:
            factor = max(0.75 * d, 1e-3)
        elif omega < 0:
            factor = max(0.5 * d, 1e-3)
        else:
            factor = min(1.5 * d, 1.0)
        
        return factor
            
    def solve(self):
        self.problem.summary()
        
        self.sens_problem = Sensitivity(self.problem)
        self.sens_filter = RadiusFilter(self.problem.mesh, self.radius)
        
        running_timer = time.perf_counter()
        
        self.analysis_time, self.optimization_time = 0, 0
        
        self.num_iter = 0
        
        vol_frac_list = [self.vol_frac]
        if self.problem.mesh.topology.dim == 3:
            if self.vol_frac < 0.2:
                n = 7
                A = -(n - 1) / math.log(self.vol_frac / 0.3)
                vol_frac_list = np.exp(-np.arange(n) / A) * 0.3
                vol_frac_list = np.round(vol_frac_list, decimals=4)
                vol_frac_list = np.append(vol_frac_list, self.vol_frac)
                vol_frac_list = np.append(vol_frac_list, self.vol_frac)
                eps_list = np.ones(n+2) * 1e-2
                eps_list[-2] = 1e-3
                eps_list[-1] = 1e-4
            else:
                vol_frac_list = [self.vol_frac, self.vol_frac]
                eps_list = np.array([1e-2, 1e-4])
        
        self.problem.rho_field.x.petsc_vec.array[self.free] = vol_frac_list[0]
        
        self.stage = 0
        
        while self.stage < len(vol_frac_list) and self.num_fem < self.max_iter:
            if self.comm.rank == 0:
                print(f"Stage: {self.stage}, Vol frac: {vol_frac_list[self.stage]:.4f}", flush=True)
            self.vol_frac = vol_frac_list[self.stage]
            self.problem.eps.value = eps_list[self.stage]
            # jump start
            if self.stage == 0 or self.stage == len(vol_frac_list) - 1:
                fem_sen_time = time.perf_counter()
                C_value, V_value, dJdrho = self.solve_prime()
                fem_sen_time = time.perf_counter() - fem_sen_time
                self.analysis_time += fem_sen_time
                
                opt_time = time.perf_counter()
                rho_values = self.problem.rho_field.x.petsc_vec.array[self.free].copy()
                if self.stage == 0:
                    rho_new, cost = self.multi_cuts(rho_values, C_value, dJdrho[self.free], self.vol_frac, 1.0)
                else:
                    rho_new, cost = self.multi_cuts(rho_values, C_value, dJdrho[self.free], self.vol_frac, self.d)
                self.problem.rho_field.x.petsc_vec.array[self.free] = rho_new.copy()
                opt_time = time.perf_counter() - opt_time
                self.optimization_time += opt_time
                
                if self.comm.rank == 0 and self.verbose > 0:
                    print(f"Iter: {self.num_iter:3d}, "\
                        f"analysis time: {fem_sen_time:9.4f} s, "\
                        f"optimization time: {opt_time:9.4f} s, "\
                        f"C: {C_value:6.3f}, Cost: {cost:8.4f}, "\
                        f"V: {V_value:4.3f}", \
                        flush=True)
                
                self.cuts.clear()
                upper_bound = math.inf
                rho_optimal = None
            else:
                opt_time = time.perf_counter()
                rho_values = self.problem.rho_field.x.petsc_vec.array[self.free].copy()
                if self.stage == 0:
                    rho_new, cost = self.multi_cuts(rho_values, C_value, dJdrho[self.free], self.vol_frac, 1.0)
                else:
                    rho_new, cost = self.multi_cuts(rho_values, C_value, dJdrho[self.free], self.vol_frac, self.d)
                self.problem.rho_field.x.petsc_vec.array[self.free] = rho_new.copy()
                opt_time = time.perf_counter() - opt_time
                self.optimization_time += opt_time
                
                self.cuts.clear()
                
                if self.comm.rank == 0 and self.verbose > 0:
                    print(f"Iter: {self.num_iter:3d}, "\
                        f"optimization time: {opt_time:9.4f} s, "\
                        f"C: {C_value:6.3f}, Cost: {cost:8.4f}, "\
                        f"V: {V_value:4.3f}", \
                        flush=True)
                
                upper_bound = math.inf
                rho_optimal = None
                    
            stack_ite = 1
            num_inner_iter = 0
            
            while self.num_fem < self.max_iter:
                num_inner_iter += 1
                
                fem_sen_time = time.perf_counter()
                C_value, V_value, dJdrho = self.solve_prime()
                fem_sen_time = time.perf_counter() - fem_sen_time
                self.analysis_time += fem_sen_time
                
                # adjust trust region according to the merit function
                if num_inner_iter > 1:
                    self.d = self.update_trust_region(old_c, C_value, cost, self.d)
                
                # stack the cuts
                if num_inner_iter == 1:
                    c = C_value
                    old_c = c
                else:
                    old_c = c
                    c = C_value
                sens = dJdrho.copy()
                
                # stop condition
                condition1 = abs(C_value - upper_bound) / abs(upper_bound) < self.opt_tol
                condition2 = abs(C_value - cost) / abs(upper_bound) < self.opt_tol
                condition3 = (C_value > upper_bound) and (cost > upper_bound)
                
                # branch over the cuts
                opt_time = time.perf_counter()
                rho_values = self.problem.rho_field.x.petsc_vec.array[self.free].copy()
                rho_new, cost = self.multi_cuts(rho_values, c, sens[self.free], self.vol_frac, self.d)
                self.problem.rho_field.x.petsc_vec.array[self.free] = rho_new.copy()
                opt_time = time.perf_counter() - opt_time
                self.optimization_time += opt_time
                
                if self.comm.rank == 0 and self.verbose > 0:
                    print(f"Iter: {self.num_iter+num_inner_iter:3d}, "\
                        f"analysis time: {fem_sen_time:9.4f} s, "\
                        f"optimization time: {opt_time:9.4f} s, "\
                        f"C: {C_value:6.3f}, Cost: {cost:8.4f}, "\
                        f"Upper: {upper_bound:6.3f}, ", \
                        f"V: {V_value:4.3f}, ", \
                        f"Trust region: {self.d:4.3f}, ", \
                        f"Con1: {abs(C_value - upper_bound) / abs(upper_bound):5.3f}, ", \
                        f"Con2: {abs(C_value - cost) / abs(upper_bound):5.3f}", \
                        flush=True)
                
                if upper_bound > C_value:
                    upper_bound = C_value
                    rho_optimal = rho_values.copy()
                    weight_optimal = dJdrho.copy()
                    d_optimal = self.d
                    stack_ite = 0
                else:
                    stack_ite += 1
                
                if (condition1 and condition2) or condition3 or stack_ite > 3:
                    break
            
            self.stage += 1
            
            self.problem.rho_field.x.petsc_vec.array[self.free] = rho_optimal.copy()
            
            self.num_iter += num_inner_iter
            
            if self.stage < len(vol_frac_list):
                rho_values = rho_optimal
                C_value = upper_bound
                dJdrho = weight_optimal.copy()
                self.d = max(vol_frac_list[self.stage - 1] - vol_frac_list[self.stage] + 5e-3, d_optimal)
                
                if self.comm.rank == 0:
                    print("\n", flush=True)
            
            self.cuts.clear()
        
        self.running_time = time.perf_counter() - running_timer
        super().summary()
        
        if self.comm.rank == 0:
            print(f"  Result: {upper_bound:6.3f}", flush=True)
        
        self.problem.save_results()
    
    def multi_cuts(self, rho_values, c_value, sens, vol_frac, d):
        # single cut
        rho_new, cost = self.sub_optimizer.update(rho_values, c_value, sens, vol_frac, d)
        
        self.cuts.add_solution(rho_new, cost, c_value, sens, d)
        
        # multi cuts
        if self.cuts.num_cuts > 1:
            if cost < min(self.cuts.solution[0][:-1]):
                self.cuts.flag_solution([self.cuts.num_cuts - 1])
            else:
                # check multi cuts
                min_cost_last_cut = cost
                for i in range(2, 3):
                    candidates = list(combinations(range(0, self.cuts.num_cuts-1), i))
                    # estimate lower bound
                    estimated_costs = []
                    reduced_candidates = []
                    for candidate in candidates:
                        lambdas = list(candidate)
                        flag = self.cuts.check_flag(lambdas)
                        if flag == False:
                            reduced_candidates.append(lambdas)
                            flag, rho_solution, estimated_cost = self.cuts.check_solution(lambdas)
                            if flag == True:
                                estimated_costs.append(estimated_cost)
                            else:
                                estimated_costs.append(self.cuts.estimate_cost(lambdas))
                    
                    if len(estimated_costs) == 0:
                        break
                    
                    estimated_min_cost = min(estimated_costs)
                    
                    if estimated_min_cost > min_cost_last_cut:
                        if self.comm.rank == 0:
                            print(f"  No multicuts check", flush=True)
                        break
                    
                    if self.comm.rank == 0:
                        print(f"  Single cut cost: {min_cost_last_cut:8.4f}", flush=True)
                    
                    idx = np.argsort(estimated_costs)
                    estimated_costs = [estimated_costs[i] for i in idx]
                    sorted_candidates = [reduced_candidates[i] for i in idx]
                    
                    best_candidate_cost = math.inf
                    best_candidate_rho = None
                    best_candidate = None
                    for i, candidate in enumerate(sorted_candidates):
                        flag, candidate_rho_multicuts, candidate_solution_cost = self.cuts.check_solution(candidate)
                        if flag == False:
                            if self.comm.rank == 0:
                                print(f"  Checking candidate: {candidate}", flush=True)
                            rho_multicuts, c_multicuts, sens_multicuts, d_multicuts = self.cuts.get_milp_operator(candidate)
                            candidate_rho_multicuts, candidate_solution_cost = self.sub_optimizer.update(rho_multicuts, c_multicuts, sens_multicuts, vol_frac, d_multicuts)
                            self.cuts.add_solution(candidate_rho_multicuts, candidate_solution_cost, c_multicuts, sens_multicuts, d_multicuts, candidate)
                        
                        if best_candidate_cost > candidate_solution_cost:
                            best_candidate_cost = candidate_solution_cost
                            best_candidate_rho = candidate_rho_multicuts.copy()
                            best_candidate = candidate.copy()
                        
                        min_estimated_cost = math.inf
                        if i < len(sorted_candidates) - 1:
                            min_estimated_cost = min(estimated_costs[i+1:None])
                        
                        if self.comm.rank == 0:
                            print(f"  Candidate: {candidate}, cost: {candidate_solution_cost:8.4f}, min estimated cost: {min_estimated_cost:8.4f}", flush=True)
                        
                        if (best_candidate_cost < min_estimated_cost) or min(best_candidate_cost, min_estimated_cost) > min_cost_last_cut:
                            # no need to check further
                            break
                        
                    if best_candidate_cost < min_cost_last_cut:
                        cost = best_candidate_cost
                        rho_new = best_candidate_rho.copy()
                        self.cuts.flag_solution(best_candidate)
                        if self.comm.rank == 0:
                            print(f"  Found multicuts {best_candidate}", flush=True)
        else:
            self.cuts.flag_solution([0])
        
        return rho_new, cost