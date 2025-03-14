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

import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import FunctionSpace, Function, Constant, dirichletbc, locate_dofs_topological, functionspace

from petsc4py import PETSc
import dolfinx.io
from dolfinx.fem import form, Function
from dolfinx import la
from dolfinx.fem.petsc import (create_vector, create_matrix,
                               assemble_vector, assemble_matrix, set_bc)

from mpi4py import MPI

from ..common import bcolors

class Problem:
    def __init__(self, descriptor):
        if descriptor["mesh"] is None:
            if MPI.COMM_WORLD.rank == 0:
                raise ValueError("Mesh is not provided.")
            exit(1)
        else:
            self.mesh = descriptor["mesh"]
        
        if descriptor["prefix"] is None:
            self.prefix = "result/"
        else:
            self.prefix = descriptor["prefix"]
            
        if descriptor["problem name"] is None:
            self.problem_name = "problem"
        else:
            self.problem_name = descriptor["problem name"]
        
        self.comm = self.mesh.comm
        
        self.V = None
        self.S0 = None
        self.S = None
        
        self.objective = descriptor["objective"]
        
        num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cell_tags = np.zeros(num_cells, dtype=np.int32)
        self.materials = meshtags(self.mesh, self.mesh.topology.dim, np.zeros(num_cells, dtype=np.int32), cell_tags)
        
        self.solver = None
        self.lhs_mat = None
        self.rhs_vec = None
        self.u_wrap = None
        
    def set_solver(self, petsc_options):
        self.lhs_mat = create_matrix(self.lhs_form)
        self.rhs_vec = create_vector(self.rhs_form)
        self.u_wrap = la.create_petsc_vector_wrap(self.u_field.x)
        
        self.solver = PETSc.KSP().create(self.u_field.function_space.mesh.comm)
        self.solver.setOperators(self.lhs_mat)
        prefix = f"linear_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)
        
        # Apply PETSc options
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        for var in [self.lhs_mat, self.rhs_vec]:
            if var is not None:
                var.setOptionsPrefix(prefix)
                var.setFromOptions()
        
    def solve_prime(self):
        """Solve K*x=F."""
        self.lhs_mat.zeroEntries()
        start = MPI.Wtime()
        assemble_matrix(self.lhs_mat, self.lhs_form, bcs=self.bcs)
        end = MPI.Wtime()
        if self.comm.rank == 0:
            print(f"Assembly time: {end-start:.4f} s", flush=True)
        self.lhs_mat.assemble()
        self.solver.solve(self.rhs_vec, self.u_wrap)
        self.u_field.x.scatter_forward()
        
    def solve_adjoint(self):
        """Solve K*lambda=-L."""
        pass
        
    def __del__(self):
        if self.solver is not None:
            self.solver.destroy()
            self.lhs_mat.destroy()
            self.rhs_vec.destroy()
            self.u_wrap.destroy()
        
class LinearElasticity(Problem):
    def __init__(self, descriptor):
        super().__init__(descriptor)
        
        if descriptor["problem name"] is None:
            self.problem_name = "linear_elasticity"
        
        self.dim = self.mesh.topology.dim            
        self.V = functionspace(self.mesh, ("CG", 1, (self.dim,)))
        self.S0 = functionspace(self.mesh, ("DG", 0))
        self.u, self.v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        self.u_field = Function(self.V)
        self.u_field.name = "displacement"
        self.rho_field = []
        self.material = Function(self.S0)
        self.material.name = "material"
        self.rank = Function(self.S0)
        self.rank.x.petsc_vec.set(self.comm.rank)
        self.rank.name = "rank"
        self.sensitivity = []
        
        if descriptor["interpolation"] == "continuous":
            self.S = functionspace(self.mesh, ("CG", 1))
            self.rho_phys_field = Function(self.S)
        
        if isinstance(descriptor["young's modulus"], (int, float)):
            self.E_list = np.array([descriptor["young's modulus"]], dtype=np.float64)
        elif isinstance(descriptor["young's modulus"], (list, tuple)):
            self.E_list = np.array(descriptor["young's modulus"], dtype=np.float64)
        elif isinstance(descriptor["young's modulus"], np.ndarray):
            self.E_list = descriptor["young's modulus"]
        else:
            raise ValueError("Young's modulus is not in the correct format.")
            exit(1)
        if np.size(self.E_list, 0) > 1:
            if isinstance(descriptor["density"], (list, tuple)):
                self.density_list = np.array(descriptor["density"], dtype=np.float64)
            elif isinstance(descriptor["density"], np.ndarray):
                self.density_list = descriptor["density"]
            elif descriptor["density"] is None:
                raise ValueError("Density is not provided.")
                exit(1)
            else:
                raise ValueError("Density is not in the correct format.")
                exit(1)
        else:
            self.density_list = np.array([1.0], dtype=np.float64)
        
        self.num_materials = np.size(self.E_list, 0)
        
        if isinstance(descriptor["poisson's ratio"], (int, float)):
            if self.num_materials > 1:
                raise ValueError("Poisson's ratio is not provided for each material.")
                exit(1)
            else:
                self.nu_list = np.array(descriptor["poisson's ratio"], dtype=np.float64)
        elif isinstance(descriptor["poisson's ratio"], (list, tuple)):
            self.nu_list = np.array(descriptor["poisson's ratio"], dtype=np.float64)
            if np.size(self.nu_list, 0) != self.num_materials:
                raise ValueError("Poisson's ratio is not provided for each material.")
                exit(1)
        elif isinstance(descriptor["poisson's ratio"], np.ndarray):
            self.nu_list = descriptor["poisson's ratio"]
            if np.size(self.nu_list, 0) != self.num_materials:
                raise ValueError("Poisson's ratio is not provided for each material.")
                exit(1)
        else:
            raise ValueError("Poisson's ratio is not in the correct format.")
            exit(1)
        
        for i in range(self.num_materials):
            self.rho_field.append(Function(self.S0))
            self.rho_field[-1].name = f"material_{i}"
            self.sensitivity.append(Function(self.S0))
            self.sensitivity[-1].name = f"sensitivity_{i}"
        
        if descriptor["interpolation"] == "discrete":
            self.interpolation = "discrete"
            self.eps = Constant(self.mesh, 1e-2)
        else:
            self.interpolation = "continuous"
            self.eps = Constant(self.mesh, 1e-6)
        
        # Kinematics
        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(v):
            λ, μ = self.lambda_mu
            return 2.0 * μ * ufl.sym(ufl.grad(v)) + λ * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v))
        
        self.disp_facets = locate_entities_boundary(self.mesh, self.dim-1, descriptor["disp_bc"])
        self.bcs = [dirichletbc(Constant(self.mesh, np.full(self.dim, 0.0)), locate_dofs_topological(self.V, self.dim-1, self.disp_facets), self.V)]
        
        tractions, facets, markers = [], [], []
        for marker, (traction, traction_bc) in enumerate(descriptor["traction_bcs"]):
            tractions.append(Constant(self.mesh, np.array(traction, dtype=float)))
            current_facets = locate_entities_boundary(self.mesh, self.dim-1, traction_bc)
            facets.extend(current_facets)
            markers.extend([marker,]*len(current_facets))
        facets = np.array(facets, dtype=np.int32)
        markers = np.array(markers, dtype=np.int32)
        _, unique_indices = np.unique(facets, return_index=True)
        facets, markers = facets[unique_indices], markers[unique_indices]
        sorted_indices = np.argsort(facets)
        facet_tags = meshtags(self.mesh, self.dim-1, facets[sorted_indices], markers[sorted_indices])
        
        metadata = {"quadrature_degree": descriptor["quadrature_degree"]}
        self.dx = ufl.Measure("dx", metadata=metadata)
        self.ds = ufl.Measure("ds", domain=self.mesh, metadata=metadata, subdomain_data=facet_tags)
        b = Constant(self.mesh, np.array(descriptor["body_force"], dtype=float))
        
        lhs = ufl.inner(sigma(self.u), epsilon(self.v))*self.dx
        rhs = ufl.dot(b, self.v)*self.dx
        for marker, t in enumerate(tractions):
            rhs += ufl.dot(t, self.v)*self.ds(marker)
        self.lhs_form = form(lhs)
        self.rhs_form = form(rhs)
            
        super().set_solver(descriptor["petsc_options"])
        
        assemble_vector(self.rhs_vec, self.rhs_form)
        self.rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.rhs_vec, self.bcs)        
        
        # Define optimization-related variables
        # - minimize compliance
        # - minimize target displacement
        if self.objective == "compliance":
            self.J = ufl.inner(sigma(self.u_field), epsilon(self.u_field))*self.dx
        if self.objective == "target displacement":
            if isinstance(descriptor["target displacement"], (list, tuple)):
                if isinstance(descriptor["target displacement"][0], (list, tuple)):
                    for target in enumerate(descriptor["target displacement"]):
                        target_disp, location = target[0], target[1]
                        facets = locate_entities_boundary(self.mesh, self.dim-1, location)
                else:
                    target_disp_func, location = descriptor["target displacement"][0], descriptor["target displacement"][1]
                    target_disp = Function(self.V).sub(2)
                    target_disp.interpolate(target_disp_func)
                    facets = locate_entities_boundary(self.mesh, self.dim-1, location)
                    target_dofs = np.unique(self.mesh.topology.connectivity(self.mesh.topology.dim - 1, 0).array[facets])
                    d_target = ufl.Measure("ds", domain=self.mesh, subdomain_data=facets)
            
            self.J = ufl.inner(self.u_field.sub(2) - target_disp, self.u_field.sub(2) - target_disp) * d_target
            
        if self.interpolation == "discrete":
            self.volume = 0
            self.mass = 0
            for i in range(self.num_materials):
                self.volume += self.rho_field[i]*self.dx
                self.mass += self.density_list[i]*self.rho_field[i]*self.dx            
        else:
            self.volume = self.rho_phys_field*self.dx
        self.total_volume = Constant(self.mesh, 1.0)*self.dx
    
    @property
    def E(self):
        if self.interpolation == "discrete":
            base_E = self.eps * max(self.E_list)
            val = self.eps * max(self.E_list)
            for i in range(self.num_materials):
                val += (self.E_list[i] - self.eps * max(self.E_list)) * self.rho_field[i]
            return val
        else:
            return (self.eps + (1 - self.eps) * self.rho_phys_field**3) * self.E_list[0]
    
    @property
    def ν(self):
        if self.interpolation == "discrete":
            val = 0
            for i in range(self.num_materials):
                val += self.nu_list[i] * self.rho_field[i]
            if val == 0:
                val = 0.3
            return val
        else:
            return self.nu_list[0]
    
    @property
    def lambda_mu(self):
        return self.E * self.ν / ((1.0 + self.ν) * (1.0 - 2.0 * self.ν)), self.E / (2.0 * (1.0 + self.ν))
        
    def summary(self):
        if self.comm.rank == 0:
            print("Problem name: ", self.problem_name)
            print("  Number of ranks: ", self.comm.size)
            if self.mesh.topology.dim == 2:
                print("  Number of cells: ", self.mesh.topology.index_map(2).size_global)
                print("  Number of vertices: ", self.mesh.topology.index_map(0).size_global)
                print("  Number of dofs: ", 2*self.V.dofmap.index_map.size_global)
            elif self.mesh.topology.dim == 3:
                print("  Number of cells: ", self.mesh.topology.index_map(3).size_global)
                print("  Number of vertices: ", self.mesh.topology.index_map(0).size_global)
                print("  Number of dofs: ", 3*self.V.dofmap.index_map.size_global)
            print("  Number of materials: ", np.size(self.E_list, 0), flush=True)
    
    def save_results(self, suffix=""):
        with dolfinx.io.XDMFFile(self.mesh.comm, self.prefix+self.problem_name+suffix+".xdmf", "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            if self.interpolation == "discrete":
                xdmf.write_function(self.u_field)
                self.material.x.petsc_vec.set(0)
                for i in range(self.num_materials):
                    self.material.x.petsc_vec.array += self.rho_field[i].x.petsc_vec.array * (i+1)
                xdmf.write_function(self.material)
                
                # for i in range(self.num_materials):
                #     xdmf.write_function(self.sensitivity[i])
                # xdmf.write_function(self.rank)
            else:
                self.rho_phys_field.name = "density"
                xdmf.write_function(self.rho_phys_field)
                self.rho_field[0].name = "density0"
                xdmf.write_function(self.rho_field[0])