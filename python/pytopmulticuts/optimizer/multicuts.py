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
from petsc4py import PETSc

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
        self.radius = descriptor["filter_radius"]
        self.d0 = self.d = descriptor["initial_trust_region"]
        
        # initialize vol/mass
        if "vol_frac" in descriptor:
            target_quantity = descriptor["vol_frac"]
        if "mass" in descriptor:
            self.mass = descriptor["mass"]
        if "initial_vol_frac" in descriptor:
            self.initial_vol_frac = descriptor["initial_vol_frac"]
        else:
            self.initial_vol_frac = None
        if "initial_mass" in descriptor:
            self.initial_mass = descriptor["initial_mass"]
        else:
            self.initial_mass = None
        if "num_stages" in descriptor:
            self.num_stages = descriptor["num_stages"]
        else:
            self.num_stages = 1
        
        self.symmetry = descriptor["filter symmetry"]
        
        self.min_eps_factor = 1e-4
        
        self.verbose = 1
        
        num_consts = 1 if self.problem.objective == "compliance" else 2
        
        num_elems = self.problem.rho_field[0].x.petsc_vec.array.size
        centers = self.problem.rho_field[0].function_space.tabulate_dof_coordinates()[:num_elems].T        
        if self.problem.num_materials == 1:
            solid, void = [descriptor["solid_zone"](centers)], [descriptor["void_zone"](centers)]
            self.quantity_constraint = descriptor["vol_frac"]
            if self.initial_vol_frac is None:
                self.initial_quantity_constraint = self.quantity_constraint
            else:
                self.initial_quantity_constraint = self.initial_vol_frac
        else:
            if not isinstance(descriptor["solid_zone"], list) or not isinstance(descriptor["void_zone"], list):
                raise ValueError("Invalid solid_zone or void_zone")
                exit(1)
            solid = []
            void = []
            for i in range(self.problem.num_materials):
                solid.append(descriptor["solid_zone"][i](centers))
                void.append(descriptor["void_zone"][i](centers))
            
            self.quantity_constraint = descriptor["mass"]
            if self.initial_mass is None:
                self.initial_quantity_constraint = self.quantity_constraint
            else:
                self.initial_quantity_constraint = self.initial_mass
        
        self.free = np.where((solid[0] == False) & (void[0] == False))[0]
        self.num_free = len(self.free)
        for i in range(self.problem.num_materials):
            rho_ini = np.zeros(num_elems)
            rho_ini[solid[i]] = 1.0
            
            free = np.where((solid[i] == False) & (void[i] == False))
            if np.linalg.norm(free[0] - self.free) > 1e-6:
                raise ValueError("Invalid solid_zone or void_zone")
                exit(1)
            
            self.problem.rho_field[i].x.petsc_vec.array = rho_ini.copy()
        
        self.num_fem = 0
        
        self.cuts = Cuts()
        
        if descriptor["subproblem_solver"] == "milp":
            self.sub_optimizer = MilpOptimizer(problem, self.num_free, num_elems, self.problem.num_materials, descriptor["solver_type"])
        elif descriptor["subproblem_solver"] == "dw":
            self.sub_optimizer = DWOptimizer(problem, self.num_free, num_elems, self.problem.num_materials, descriptor["num_divisions"], descriptor["solver_type"])
        else:
            raise ValueError("Invalid subproblem_solver")
            exit(1)
        
    def solve_prime(self):
        self.problem.solve_prime()
        
        [J, quantity], dJdrho = self.sens_problem.evaluate()
        
        dJdrho_base = np.zeros(self.problem.rho_field[0].x.petsc_vec.array.size)
        rho_mask = np.zeros(self.problem.rho_field[0].x.petsc_vec.array.size)
        for i in range(self.problem.num_materials):
            dJdrho_base[self.problem.rho_field[i].x.petsc_vec.array > 0] = \
                dJdrho[i].array[self.problem.rho_field[i].x.petsc_vec.array > 0].copy()
            rho_mask += self.problem.rho_field[i].x.petsc_vec.array.copy()
        dJdrho_base[rho_mask < 1e-3] = self.problem.eps.value * \
            dJdrho[-1].array[rho_mask < 1e-3]
        
        for i in range(self.problem.num_materials):
            dJdrho[i] = dJdrho[i].array.copy()
            idx = np.where(self.problem.rho_field[i].x.petsc_vec.array > 0)
            dJdrho[i][idx] = dJdrho[i][idx] * self.problem.rho_field[i].x.petsc_vec.array[idx]
            idx = np.where(self.problem.rho_field[i].x.petsc_vec.array == 0)
            dJdrho[i][idx] = \
                2.0 * np.divide(np.multiply(dJdrho[i][idx], dJdrho_base[idx]), dJdrho[i][idx] + dJdrho_base[idx])
            dJdrho[i][rho_mask < 1e-3] = dJdrho_base[rho_mask < 1e-3]
            dJdrho[i] = self.sens_filter.filter(dJdrho[i])
        
        for i in range(self.problem.num_materials):
            self.problem.sensitivity[i].x.petsc_vec.array = dJdrho[i]
        
        self.num_fem += 1
        
        # self.problem.save_results("_" + str(self.num_fem))
            
        return J, quantity, dJdrho
    
    def update_trust_region(self, c0, c, cost, d):
        omega = (c0 - c) / (c0 - cost)
        if omega < 1 and omega >= 0:
            factor = max(0.75 * d, 1e-3)
        elif omega < 0:
            factor = max(0.5 * d, 1e-3)
        else:
            factor = min(2 * d, 1.0)
        
        if self.comm.rank == 0 and self.verbose > 0:
            print(f"  Trust region: {d:4.3f} -> {factor:4.3f} with omega {omega:4.3f}", flush=True)
        
        return factor
            
    def solve(self):
        self.problem.summary()
        
        self.sens_problem = Sensitivity(self.problem)
        self.sens_filter = RadiusFilter(self.problem.mesh, self.radius, self.symmetry)
        
        running_timer = time.perf_counter()
        
        self.analysis_time, self.optimization_time = 0, 0
        
        self.num_iter = 0
        
        if self.num_stages > 1:
            n = self.num_stages
            A = -(n - 1) / math.log(self.quantity_constraint / self.initial_quantity_constraint)
            quantity_constraint_list = np.exp(-np.arange(n) / A) * self.initial_quantity_constraint
            quantity_constraint_list = np.round(quantity_constraint_list, decimals=4)
            if self.problem.num_materials == 1:
                quantity_constraint_list = np.append(quantity_constraint_list, self.quantity_constraint)
            quantity_constraint_list = np.append(quantity_constraint_list, self.quantity_constraint)
        else:
            quantity_constraint_list = np.array([self.quantity_constraint, self.quantity_constraint, self.quantity_constraint])
        
        eps_list = np.ones(self.num_stages+2, dtype=float) * 1e-2
        eps_list[-2] = 1e-3
        eps_list[-1] = 1e-4
        
        quantity_local_offset = 0.0
        dQdrho = self.sens_problem.evaluate_quantity()
        free_quantity_local = np.zeros(self.problem.num_materials)
        for i in range(self.problem.num_materials):
            quantity_local_offset += np.dot(dQdrho[i].array, self.problem.rho_field[i].x.petsc_vec.array)
            free_quantity_local[i] = np.sum(dQdrho[i].array[self.free])
        quantity_constraint_offset = self.comm.allreduce(quantity_local_offset, op=MPI.SUM)
        free_quantity = np.array(self.comm.allreduce(free_quantity_local, op=MPI.SUM))
        
        total_free_quantity = np.sum(free_quantity)
        allowable_quantity = quantity_constraint_list[0] - quantity_constraint_offset
        initial_quantity = allowable_quantity / total_free_quantity
        for i in range(self.problem.num_materials):
            self.problem.rho_field[i].x.petsc_vec.array[self.free] = initial_quantity
        
        for i in range(self.problem.num_materials):
            dQdrho[i] = dQdrho[i].array[self.free]
        self.sub_optimizer.set_dQdrho(dQdrho)
        
        self.stage = 0
        min_inner_iter = 0
        
        while self.stage < len(quantity_constraint_list) and self.num_fem < self.max_iter:
            if self.comm.rank == 0:
                if self.problem.num_materials == 1:
                    print(f"Stage: {self.stage}, Vol frac: {quantity_constraint_list[self.stage]:.4f}", flush=True)
                else:
                    print(f"Stage: {self.stage}, Mass: {quantity_constraint_list[self.stage]:.4f}", flush=True)
            target_quantity = quantity_constraint_list[self.stage] - quantity_constraint_offset
            self.problem.eps.value = eps_list[self.stage]
        
            # jump start
            if self.stage == 0 or self.stage >= self.num_stages:
                fem_sen_time = time.perf_counter()
                J, quantity, dJdrho = self.solve_prime()
                fem_sen_time = time.perf_counter() - fem_sen_time
                self.analysis_time += fem_sen_time
                
                opt_time = time.perf_counter()
                rho_values, sens = [], []
                for i in range(self.problem.num_materials):
                    rho_values.append(self.problem.rho_field[i].x.petsc_vec.array[self.free].copy())
                    sens.append(dJdrho[i][self.free].copy())
                if self.stage == 0:
                    rho_new, cost = self.multi_cuts(rho_values, J, sens, target_quantity, 1.0)
                else:
                    rho_new, cost = self.multi_cuts(rho_values, J, sens, target_quantity, self.d)
                for i in range(self.problem.num_materials):
                    self.problem.rho_field[i].x.petsc_vec.array[self.free] = rho_new[i].copy()
                opt_time = time.perf_counter() - opt_time
                self.optimization_time += opt_time
                
                self.cuts.clear()
                upper_bound = math.inf
                rho_optimal = None
            else:
                opt_time = time.perf_counter()
                rho_values, sens = [], []
                for i in range(self.problem.num_materials):
                    rho_values.append(self.problem.rho_field[i].x.petsc_vec.array[self.free].copy())
                    sens.append(dJdrho[i][self.free].copy())
                rho_new, cost = self.multi_cuts(rho_values, J, sens, target_quantity, self.d)
                for i in range(self.problem.num_materials):
                    self.problem.rho_field[i].x.petsc_vec.array[self.free] = rho_new[i].copy()
                opt_time = time.perf_counter() - opt_time
                self.optimization_time += opt_time
                
                self.cuts.clear()
                
                upper_bound = math.inf
                rho_optimal = None
                
            if self.comm.rank == 0 and self.verbose > 0:
                if self.problem.num_materials == 1:
                    print(f"Iter: {self.num_iter:3d}, "\
                        f"analysis time: {fem_sen_time:9.4f} s, "\
                        f"optimization time: {opt_time:9.4f} s, "\
                        f"C: {J:6.3f}, Cost: {cost:8.4f}, "\
                        f"V: {quantity:4.3f}", \
                        flush=True)
                else:
                    print(f"Iter: {self.num_iter:3d}, "\
                        f"analysis time: {fem_sen_time:9.4f} s, "\
                        f"optimization time: {opt_time:9.4f} s, "\
                        f"C: {J:6.3f}, Cost: {cost:8.4f}, "\
                        f"M: {quantity:4.3f}", \
                        flush=True)
                    
            stack_ite = 1
            num_inner_iter = 0
            
            while self.num_fem < self.max_iter:
                num_inner_iter += 1
                
                fem_sen_time = time.perf_counter()
                J, quantity, dJdrho = self.solve_prime()
                fem_sen_time = time.perf_counter() - fem_sen_time
                self.analysis_time += fem_sen_time
                
                # adjust trust region according to the merit function
                if num_inner_iter > 1:
                    self.d = self.update_trust_region(old_c, J, cost, self.d)
                
                # stack the cuts
                if num_inner_iter == 1:
                    c = J
                    old_c = c
                else:
                    old_c = c
                    c = J
                
                # stop condition
                condition1 = round(abs(J - upper_bound) / abs(upper_bound), 3) <= self.opt_tol
                condition2 = round(abs(J - cost) / abs(upper_bound), 3) <= self.opt_tol
                condition3 = (J > upper_bound) and (cost > upper_bound)
                
                # branch over the cuts
                opt_time = time.perf_counter()
                rho_values, sens = [], []
                for i in range(self.problem.num_materials):
                    rho_values.append(self.problem.rho_field[i].x.petsc_vec.array[self.free].copy())
                    sens.append(dJdrho[i][self.free].copy())
                rho_new, cost = self.multi_cuts(rho_values, J, sens, target_quantity, self.d)
                for i in range(self.problem.num_materials):
                    self.problem.rho_field[i].x.petsc_vec.array[self.free] = rho_new[i].copy()
                opt_time = time.perf_counter() - opt_time
                self.optimization_time += opt_time
                
                if self.comm.rank == 0 and self.verbose > 0:
                    if self.problem.num_materials == 1:
                        print(f"Iter: {self.num_iter+num_inner_iter:3d}, "\
                            f"analysis time: {fem_sen_time:9.4f} s, "\
                            f"optimization time: {opt_time:9.4f} s, "\
                            f"C: {J:6.3f}, Cost: {cost:8.4f}, "\
                            f"Upper: {upper_bound:6.3f}, ", \
                            f"V: {quantity:4.3f}, ", \
                            f"Trust region: {self.d:4.3f}, ", \
                            f"Con1: {abs(J - upper_bound) / abs(upper_bound):5.3f}, ", \
                            f"Con2: {abs(J - cost) / abs(upper_bound):5.3f}", \
                            flush=True)
                    else:
                        print(f"Iter: {self.num_iter+num_inner_iter:3d}, "\
                            f"analysis time: {fem_sen_time:9.4f} s, "\
                            f"optimization time: {opt_time:9.4f} s, "\
                            f"C: {J:6.3f}, Cost: {cost:8.4f}, "\
                            f"Upper: {upper_bound:6.3f}, ", \
                            f"M: {quantity:4.3f}, ", \
                            f"Trust region: {self.d:4.3f}, ", \
                            f"Con1: {abs(J - upper_bound) / abs(upper_bound):5.3f}, ", \
                            f"Con2: {abs(J - cost) / abs(upper_bound):5.3f}", \
                            flush=True)
                
                if upper_bound > J:
                    upper_bound = J
                    rho_optimal = rho_values.copy()
                    weight_optimal = dJdrho.copy()
                    d_optimal = self.d
                    stack_ite = 0
                else:
                    stack_ite += 1
                
                if (condition1 and condition2) or stack_ite > 5:
                    break
            
            self.stage += 1
            
            for i in range(self.problem.num_materials):
                self.problem.rho_field[i].x.petsc_vec.array[self.free] = rho_optimal[i].copy()
            
            self.num_iter += num_inner_iter
            
            if self.stage < len(quantity_constraint_list):
                rho_values = rho_optimal
                J = upper_bound
                dJdrho = weight_optimal.copy()
                if self.stage < self.num_stages:
                    self.d = max(quantity_constraint_list[self.stage - 1] - quantity_constraint_list[self.stage] + 5e-3, d_optimal)
                else:
                    self.d = max(self.d0, d_optimal)
                
                min_inner_iter = 0
                
                # self.problem.save_results("_" + str(self.stage))
                
                if self.comm.rank == 0:
                    print("\n", flush=True)
            
            self.cuts.clear()
        
        self.running_time = time.perf_counter() - running_timer
        super().summary()
        
        if self.comm.rank == 0:
            print(f"  Result: {upper_bound:6.3f}", flush=True)
        
        self.problem.save_results()
    
    def multi_cuts(self, rho_values, c_value, sens, quantity, d):
        # single cut
        rho_new, cost = self.sub_optimizer.update(rho_values, c_value, sens, quantity, d)
        
        self.cuts.add_solution(rho_new, cost, c_value, sens, d)
        
        # multi cuts
        if self.cuts.num_cuts > 1:
            if cost < min(self.cuts.solution[0][:-1]):
                self.cuts.flag_solution([self.cuts.num_cuts - 1])
            else:
                unused_single_cut = []
                unused_single_cut_idx = []
                for i in range(len(self.cuts.solution[0])-1):
                    if self.cuts.use_flag_history[0][i] == False:
                        unused_single_cut.append(self.cuts.solution[0][i])
                        unused_single_cut_idx.append(i)
                if len(unused_single_cut) > 0:
                    idx = np.argmin(np.array(unused_single_cut))
                    _, rho_new, cost = self.cuts.check_solution([unused_single_cut_idx[idx]])
                    self.cuts.flag_solution([unused_single_cut_idx[idx]])
                    
                    return rho_new, cost
                
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
                        self.cuts.flag_solution([self.cuts.num_cuts - 1])
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
                            candidate_rho_multicuts, candidate_solution_cost = self.sub_optimizer.update(rho_multicuts, c_multicuts, sens_multicuts, quantity, d_multicuts)
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
                        self.cuts.flag_solution([self.cuts.num_cuts - 1])
        else:
            self.cuts.flag_solution([0])
        
        return rho_new, cost