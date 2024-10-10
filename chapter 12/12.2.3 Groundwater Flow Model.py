"""
Program to solve unsteady groundwater
flow equation using the finite-element
method.
"""
import numpy as np
from dolfinx import plot
from dolfinx.fem import (FunctionSpace, Function,
                         dirichletbc, locate_dofs_topological,
                         form, petsc, Constant)
from dolfinx.io.gmshio import read_from_msh
from ufl import (FiniteElement, TestFunction, TrialFunction,
                 grad, dot, dx, FacetNormal)
from mpi4py import MPI
import pyvista as pv
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
pv.set_plot_theme(pv.themes.DocumentTheme())


"""
File I/O
"""
simulationName = "Groundwater_flow-2D"
meshName = "groundwater_2D"
meshPath = "../meshes_gmsh/" + meshName + ".msh"
resultPath = "./result/"


"""
Running parameters
"""
t_start = 0
t_stop = 1
Nt = 100
dt = (t_stop - t_start) / Nt


"""
Computational domain
"""
domain, cell_tags, facet_tags = read_from_msh(
    filename=meshPath,
    comm=MPI.COMM_WORLD,
    rank=0,
    gdim=2)

n = FacetNormal(domain)
surround = int(8)


"""
Model parameters
"""
S = Constant(
    domain=domain, c=3.5)  # Storage
K = Constant(
    domain=domain, c=0.05)  # Hydraulic conductivity


"""
Finite element space
"""
fe_elem = FiniteElement(family="CG",
                        cell=domain.ufl_cell(),
                        degree=2)
fe_space = FunctionSpace(mesh=domain,
                         element=fe_elem)


"""
Visualize
"""
ocean = plt.cm.get_cmap("viridis")
ocean_modified = LinearSegmentedColormap.from_list(
    "ocean_modified",
    [ocean(i) for i in range(int(0), int(192))])

def visualize_and_save(
        h_, t, name_,
        grid = pv.UnstructuredGrid(
            *plot.create_vtk_mesh(fe_space))):

    p = pv.Plotter(off_screen=True)

    # Initialize grid values
    grid.point_data[f"h({t})"] = h_.x.array.real

    # Warp by values
    warped = grid.warp_by_scalar(
        scalars=f"h({t})", factor=1.0)

    # Set angle and show
    p.add_mesh(warped, cmap=ocean_modified,
               show_scalar_bar=True,
               lighting='three lights')
    # p.show_grid()
    p.show_axes()
    # p.camera.view_angle = 15.0
    p.camera.azimuth = 120
    # p.camera.elevation = -15
    p.camera.position = (-13.0, 15.0, 10.0)

    p.show(title="Groundwater flow simulation",
           screenshot=resultPath + name_)
    p.close()


"""
Initial condition
"""
def initial_head(x):
    return np.full(x.shape[1], 0.0)


h_n = Function(V=fe_space, name="h_initial")
h_n.interpolate(initial_head)
h_ = Function(V=fe_space, name="h_solution")
h_.interpolate(initial_head)


"""
Boundary condition
"""
fdim = domain.topology.dim - 1
bc = dirichletbc(value=ScalarType(0.0),
                 dofs=locate_dofs_topological(
                     V=fe_space,
                     entity_dim=fdim,
                     entities=facet_tags.find(surround)),
                 V=fe_space)

# Store first field value
name_ = "h_{:.04f}_.png"
visualize_and_save(h_, t_start, name_.format(t_start))


"""
Forcing function
"""
def sink(x):
    vals = np.full(x.shape[1], 0.0)
    well_1 = (x[0] - 2.5) ** 2 + (x[1] - 1.25) ** 2 < 0.15
    well_2 = (x[0] - 8.75) ** 2 + (x[1] - 7.5) ** 2 < 0.15
    vals[well_1] = -20.0  # deeper water level
    vals[well_2] = -10.0  # shallower water level
    return vals


"""
Variation formulation
"""
# Trial and Test functions
h = TrialFunction(function_space=fe_space)
v = TestFunction(function_space=fe_space)

# Source function
depth_fun = Function(V=fe_space)
depth_fun.interpolate(lambda x: sink(x))

# Variational form
a = S*h * v * dx
a += dt*K * dot(grad(h), grad(v)) * dx
L = (S*h_n + dt*depth_fun) * v * dx

# Bilinear and linear forms
bil_form, lin_form = form(a), form(L)


"""
Solver config
"""
# assemble
A = petsc.assemble_matrix(bil_form, bcs=[bc])
b = petsc.create_vector(lin_form)
A.assemble()

# solver
linear_solver = PETSc.KSP().create(domain.comm)
linear_solver.setOperators(A)
linear_solver.setType(PETSc.KSP.Type.PREONLY)
linear_solver.getPC().setType(PETSc.PC.Type.LU)


"""
Time-marching through solution
"""
t = t_start
for i in range(Nt):

    # Update time
    t = t + dt

    # Set all the entries of the local vector b to zero
    # and then assemble the global vector linear_form
    # using the values from the local vector.
    with b.localForm() as loc_b:
        loc_b.set(0)
    petsc.assemble_vector(b, lin_form)

    # Apply boundary conditions to the right-hand side
    # vector 'b' of the linear system by modifying its
    # values according to the specified boundary conditions.
    # The 'ghostUpdate' operation ensures that the values of
    # 'b' are correctly communicated and updated across parallel
    # processes.
    petsc.apply_lifting(b, [bil_form], [[bc]])
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    # Solves a linear system of equations represented by the
    # matrix equation Ax = b. Solves the system
    # and updates the solution vector h_.vector. The
    # h_.x.scatter_forward() operation distributes the
    # updated values of h_ across the distributed mesh for further
    # processing or visualization.
    linear_solver.solve(b, h_.vector)
    h_.x.scatter_forward()

    # Assign previous to the current
    h_n.x.array[:] = h_.x.array

    # Store subsequent field values
    name_ = "h_{:.04f}_.png"
    if i % 1 == 0:
        visualize_and_save(h_, t, name_.format(t))

    print("Done step {}".format(t))

print("Done!")

