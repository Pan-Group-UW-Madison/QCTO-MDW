"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

'''
Modified by:
- Zisheng Ye (ye57@wisc.edu)
- Wenxiao Pan (wpan9@wisc.edu)
'''

import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from petsc4py import PETSc

class Sensitivity():
    def __init__(self, problem):
        self.comm = comm = MPI.COMM_WORLD
        self.num_materials = len(problem.rho_field)
        # Compliance
        self.opt_compliance = problem.objective == "compliance"
        if self.opt_compliance:
            self.C_form = form(problem.J)
        if problem.interpolation == "continuous":
            self.dCdrho_form, self.dCdrho_vec = [], []
            for i in range(self.num_materials):
                self.dCdrho_form.append(form(-ufl.derivative(problem.J, problem.rho_phys_field)))
                self.dCdrho_vec.append(create_vector(self.dCdrho_form[-1]))
        else:
            self.dCdrho_form, self.dCdrho_vec = [], []
            for i in range(self.num_materials):
                self.dCdrho_form.append(form(-ufl.derivative(problem.J, problem.rho_field[i])))
                self.dCdrho_vec.append(create_vector(self.dCdrho_form[-1]))

        # Volume
        self.total_volume = comm.allreduce(
            assemble_scalar(form(problem.total_volume)), op=MPI.SUM)
        self.V_form = form(problem.volume)
        if problem.interpolation == "continuous":
            dVdrho_form = form(ufl.derivative(problem.volume, problem.rho_phys_field))
        else:
            dVdrho_form = form(ufl.derivative(problem.volume, problem.rho_field[0]))
        self.dVdrho_vec = create_vector(dVdrho_form)
        assemble_vector(self.dVdrho_vec, dVdrho_form)
        self.dVdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dVdrho_vec /= self.total_volume
        
        # Mass
        if self.num_materials > 1:
            self.M_form = form(problem.mass)
            self.dMdrho_vec = []
            for i in range(self.num_materials):
                dMdrho_form = form(ufl.derivative(problem.mass, problem.rho_field[i]))
                dMdrho_vec = create_vector(dMdrho_form)
                assemble_vector(dMdrho_vec, dMdrho_form)
                dMdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                self.dMdrho_vec.append(dMdrho_vec)

    def evaluate(self):
        # Compliance
        if self.opt_compliance:
            C_value = self.comm.allreduce(assemble_scalar(self.C_form), op=MPI.SUM)
        else:
            self.problem.lhs_mat.mult(self.u_field.x.petsc_vec, self.prod_vec)
            C_value = self.u_field.x.petsc_vec.dot(self.prod_vec)
        for i in range(self.num_materials):
            with self.dCdrho_vec[i].localForm() as loc:
                loc.set(0)
            assemble_vector(self.dCdrho_vec[i], self.dCdrho_form[i])
            self.dCdrho_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        if self.num_materials > 1:
            quantity = self.comm.allreduce(assemble_scalar(self.M_form), op=MPI.SUM)
        else:
            quantity = self.comm.allreduce(assemble_scalar(self.V_form), op=MPI.SUM) / self.total_volume

        func_values = [C_value, quantity]
        sensitivities = self.dCdrho_vec.copy()
        return func_values, sensitivities

    def evaluate_quantity(self):
        if self.num_materials > 1:
            return self.dMdrho_vec
        else:
            return [self.dVdrho_vec.copy()]