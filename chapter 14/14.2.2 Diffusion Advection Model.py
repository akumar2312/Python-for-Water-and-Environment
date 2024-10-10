"""
Program to solve diffusion, advection
equation for modelling the contaminant
transport problem. Given the location
and strength (concentration) of the
source function, we study the concentration
of the reactant as a function of space and
time. To make the visualizations appealing
we load the velocity data from the
Navier-Stokes simulation output.
"""

# We first load the necessary libraries
import h5py
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, nls, io
from ufl import (dot, dx, grad)
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import read_from_msh
import tqdm.autonotebook

"""
File handling block
"""
simulationName = "Diff_Adv-2D"
meshName = "aquifer2D"
meshPath = "./meshes_gmsh/" + meshName + ".msh"
velocityDataPath = "./result/" \
                   "Stream_NS-2D/velocity_timeseries.h5"
resultPath = "./result/" + simulationName + "/"
resultPath = "/home/anil/Desktop/py/result/" + \
             simulationName + "/"

"""
We load the same mesh, generated externally
using GMSH which was also used for the
Navier-Stokes simulation.
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
used in the Navier-Stokes because we want
to read the velocity data it.
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
D_coeff = fem.Constant(
    domain=domain,
    c=PETSc.ScalarType(0.03))

"""
Here, we define the element type
and the function space to solve 
for the concentration of the 
reactant.
"""
P1 = ufl.FiniteElement(
    family="Lagrange",
    cell=domain.ufl_cell(),
    degree=1)
V = fem.FunctionSpace(
    mesh=domain,
    element=P1)
v1 = ufl.TestFunction(function_space=V)
u1 = fem.Function(V=V)
u1n = fem.Function(V=V)
u1.name = "concentration"
u1n.name = "concentration"

"""
Here, we define a vector element
and a corresponding function space,
function for loading the 
velocity (a vector) data from the 
NS simulation.
"""
Pvec = ufl.VectorElement(
    family="Lagrange",
    cell=domain.ufl_cell(),
    degree=2)
W = fem.FunctionSpace(
    mesh=domain,
    element=Pvec)
w = fem.Function(V=W)


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
                          ptSrc_xy=[0.25, 0.75])

# Interpolating into a function
f = fem.Function(V=V)
f.interpolate(u=source.eval)

"""
Here, we define the weak formulation
of the partial differential equation.
'w' is a function defined on a vector
function space 'W' that holds the value
of the velocity field. This velocity
is taken from a previous Navier-Stokes
simulation. 
"""
F = ((u1 - u1n) * dt_inv) * v1 * dx
F = F + dot(w, grad(u1)) * v1 * dx
F = F + D_coeff * dot(grad(u1), grad(v1)) * dx
# F = F + Rate * u1 * v1 * dx
F = F - f * v1 * dx

"""
The current advection reaction equation,
although solvable by a linear solver, we 
deliberately do away with a nonlinear
solver because one may wish to model a 
case where the rate of decomposition of 
the contaminant may at times be nonlinear.
Newton solver is good with them.
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
written just once; in the time loop we only
write the function values as the mesh gets 
shared during visualization.
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
We load the velocity data file from the Navier-Stokes
simulation. The velocity vector (u_i, u_j) is used
to realize the advection part of the equation. 
Because the data was stored in HDF5 format, we employ
h5py library to load the data. This is done just before
'solving' the equation.
"""
with h5py.File(
        name=velocityDataPath, mode='r') as fr:
    print("Name of the velocity variable is {}".format(
        list(fr.keys())))

    # Time stepping and solving
    progress = tqdm.autonotebook.tqdm(
        desc="Solving nonlinear system",
        total=n_steps
    )
    t = t_start
    for n in range(n_steps):
        print("Doing {}/{} step".format(n, n_steps))
        t = t + delta_t
        u_data = fr['u'][n]
        w.vector.array = u_data
        r = solver.solve(u=u1)
        u1n.x.array[:] = u1.x.array
        xdmfu.write_function(
            u=u1,
            t=t)
    xdmfu.close()
fr.close()
print("Done!")
