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

import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from petsc4py import PETSc


class Sensitivity():
    def __init__(self, problem):
        self.comm = comm = MPI.COMM_WORLD
        # Compliance
        self.opt_compliance = problem.objective == "compliance"
        if self.opt_compliance:
            self.C_form = form(problem.compliance)
        if problem.interpolation == "continuous":
            self.dCdrho_form = form(-ufl.derivative(problem.compliance, problem.rho_phys_field))
        else:
            self.dCdrho_form = form(-ufl.derivative(problem.compliance, problem.rho_field))
        self.dCdrho_vec = create_vector(self.dCdrho_form)

        # Volume
        self.total_volume = comm.allreduce(
            assemble_scalar(form(problem.total_volume)), op=MPI.SUM)
        self.V_form = form(problem.volume)
        if problem.interpolation == "continuous":
            dVdrho_form = form(ufl.derivative(problem.volume, problem.rho_phys_field))
        else:
            dVdrho_form = form(ufl.derivative(problem.volume, problem.rho_field))
        self.dVdrho_vec = create_vector(dVdrho_form)
        assemble_vector(self.dVdrho_vec, dVdrho_form)
        self.dVdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dVdrho_vec /= self.total_volume

        # Displacement
        if not self.opt_compliance:
            self.dfdrho_form = form(ufl.adjoint(ufl.derivative(problem.f_int, problem.rho_phys_field)))
            self.dfdrho_mat = create_matrix(self.dfdrho_form)
            self.problem = problem
            self.u_field, self.lambda_field = u_field, lambda_field
            self.dUdrho_vec = problem.rho_phys_field.x.petsc_vec.copy()
            self.prod_vec = u_field.x.petsc_vec.copy()

    def __del__(self):
        if not self.opt_compliance:
            self.prod_vec.destroy()

    def evaluate(self):
        # Compliance
        if self.opt_compliance:
            C_value = self.comm.allreduce(assemble_scalar(self.C_form), op=MPI.SUM)
        else:
            self.problem.lhs_mat.mult(self.u_field.x.petsc_vec, self.prod_vec)
            C_value = self.u_field.x.petsc_vec.dot(self.prod_vec)
        with self.dCdrho_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(self.dCdrho_vec, self.dCdrho_form)
        self.dCdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        actual_volume = self.comm.allreduce(assemble_scalar(self.V_form), op=MPI.SUM)
        V_value = actual_volume / self.total_volume
        self.dVdrho_vec_copy = self.dVdrho_vec.copy()

        func_values = [C_value, V_value]
        sensitivities = [self.dCdrho_vec, self.dVdrho_vec_copy]
        return func_values, sensitivities
