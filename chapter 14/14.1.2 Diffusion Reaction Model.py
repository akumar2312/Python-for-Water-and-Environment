"""
Program to simulate the diffusion reaction
system in the domain realized using the
aquifer geometry of the Navier-Stokes simulation.
"""


# We first load the necessary libraries
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, nls, io
from ufl import (dot, dx, grad)
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import read_from_msh


"""
File handling
"""
simulationName = "Diff_React-2Dp"
meshName = "aquifer2D"
meshPath = "./meshes_gmsh/" + meshName + ".msh"
resultPath = "./result/" + simulationName + "/"


"""
Load external mesh that was generated using GMSH.
The external mesh has dimension=3 although the
z-coordinates are just zeros. Therefore, we use
gdim=2, to specify the actual dimension of the 
problem.
"""
domain, cell_tags, ft = io.gmshio.read_from_msh(
    filename=meshPath,
    comm=MPI.COMM_WORLD,
    rank=0,
    gdim=2)
gdim = domain.topology.dim  # domain dimension
fdim = gdim - 1  # facet dimension

"""
We define the start and stop time of the
simulation. The values are same as that 
used in the Navier-Stokes to help make a
comparison.
"""
t_start = 0.0
t_stop = 3.0
delta_t = 1 / 1500
n_steps = int((t_stop - t_start) / delta_t)  # N steps

"""
These constants are defined as they are
to be used in the weak form of the PDE.
"""
dt_inv = fem.Constant(
    domain=domain,
    c=PETSc.ScalarType(1 / delta_t))
Rate = fem.Constant(
    domain=domain,
    c=PETSc.ScalarType(0.01))
D_coeff = fem.Constant(
    domain=domain,
    c=PETSc.ScalarType(0.3))

"""
Here, we define the element type
and the function space to solve 
for the concentration of the 
reactant.
"""
P_elem = ufl.FiniteElement(
    family="Lagrange",
    cell=domain.ufl_cell(),
    degree=1)
V = fem.FunctionSpace(
    mesh=domain,
    element=P_elem)
v1 = ufl.TestFunction(function_space=V)
u1 = fem.Function(V=V)
u1n = fem.Function(V=V)
u1.name = "concentration"
u1n.name = "concentration"


class SourceExpression:
    """
    We define the source location and its
    magnitude that would go into the weak
    form. The (x, y) location is specified
    using 'ptSrc_xy' argument.
    The magnitude is specified as 10.0.
    """

    def __init__(self, t_, ptSrc_xy):
        self.t = t_
        self.ptSrc_xy = ptSrc_xy

    def eval(self, x):
        values = np.full(x.shape[1], 0.0)
        idx = ((x[0] - self.ptSrc_xy[0]) ** 2) + \
              ((x[1] - self.ptSrc_xy[1]) ** 2) < 0.05 ** 2
        values[idx] = 100.0  # Strength of the source
        return values


# Inheriting from the source function class definition
source = SourceExpression(t_=0.0,
                          ptSrc_xy=[0.50, 0.75])

# Interpolating into a function
f = fem.Function(V=V)
f.interpolate(u=source.eval)

"""
We define the weak formulation of the 
set of partial differential equation
concerning the diffusion and reactive 
process.
"""
F = ((u1 - u1n) * dt_inv) * v1 * dx
F = F + D_coeff * dot(grad(u1), grad(v1)) * dx
F = F + Rate * u1 * v1 * dx
F = F - f * v1 * dx

"""
The current diffusion reaction equation,
although solvable by a linear solver, we 
deliberately do away with a nonlinear
solver because one may wish to model a 
case where the rate of decomposition of 
the contaminant may at times be proportional 
to a nonlinear expression of the concentrations
of the reactants. Newton solver is good in 
such cases.
"""
problem = fem.petsc.NonlinearProblem(
    F=F,
    u=u1)
solver = nls.petsc.NewtonSolver(
    comm=MPI.COMM_WORLD,
    problem=problem)
solver.rtol = 1e-6
solver.report = True

"""
We use the open file format '.xdmf' to store 
the results of the simulation. We first write 
the mesh followed by the function. The mesh is
written just once; then in the time loop we 
only write the function values as the mesh can
now be shared during visualization.
"""
xdmfu = XDMFFile(
    comm=domain.comm,
    filename=resultPath + simulationName + ".xdmf",
    file_mode="w",
    encoding=XDMFFile.Encoding.HDF5)
xdmfu.write_mesh(mesh=domain)
xdmfu.write_function(
    u=u1,
    t=t_start)

"""
There is no velocity term in this system. We only
model the contaminant spread due to diffusion. 
"""

t = t_start
for n in range(n_steps):
    print("Doing {}/{} step".format(n, n_steps))
    t = t + delta_t
    r = solver.solve(u=u1)
    data_ = u1.vector.array
    data_[data_ < 0] = 0
    u1.vector.array = data_
    u1n.x.array[:] = u1.vector.array
    xdmfu.write_function(
        u=u1,
        t=t)
xdmfu.close()
print("Done!")
