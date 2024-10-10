"""
Program to simulate a steady seepage
flow using finite element method.
"""
import numpy as np
from dolfinx import fem, plot
from dolfinx.io.gmshio import read_from_msh
from ufl import (FiniteElement, MixedElement,
                 TestFunctions, TrialFunctions,
                 grad, dot, inner,
                 ds, dx, FacetNormal)
from mpi4py import MPI
import pyvista as pv
from petsc4py.PETSc import ScalarType


"""
File IO
"""
simulationName = "Seepage_flow-2D"
meshName = "seepage_2D"
meshPath = "./meshes_gmsh/" + meshName + ".msh"
resultPath = "./result/" + simulationName + ".png"


"""
Computational domain - We load an external
mesh suitable for demonstrating seepage
flow
"""
domain, cell_tags, facet_tags = read_from_msh(
    filename=meshPath,
    comm=MPI.COMM_WORLD,
    rank=0,
    gdim=2)

# Facet dimension where the boundary condition needs
# to be applied is 1 less than the domain's dimension
facetdim = domain.topology.dim - 1
n = FacetNormal(domain)


"""
Finite element space and functions
"""
# Define mixed elements for (velocity, pressure) pair
# that are stable (from literature)
DRT = FiniteElement(
    family="DRT",
    cell=domain.ufl_cell(),
    degree=2)
CG = FiniteElement(
    family="CG",
    cell=domain.ufl_cell(),
    degree=3)
W = fem.FunctionSpace(
    mesh=domain,
    element=MixedElement([DRT, CG]))


"""
Trial, Test functions and 
variational formulation
"""
# Derive trial and test functions from the mixed element space
(sigma, u) = TrialFunctions(function_space=W)
(tau, v) = TestFunctions(function_space=W)
a = (dot(sigma, tau) +
     dot(grad(u), tau) +
     dot(sigma, grad(v))) * dx
L = -inner(n, sigma) * v * ds


"""
Boundary conditions - this sets up the 
necessary boundary conditions for the 
Darcy flow simulation by assigning specific 
pressure values at the inlet and outlet 
boundaries, ensuring accurate modeling 
of the flow behavior.
"""
# Dirichlet boundary condition on pressure (u)
spc = 1  # signifies 2nd space where pressure variable dwells
Q, _ = W.sub(spc).collapse()
dof_inlet = fem.locate_dofs_topological(
    V=(W.sub(spc), Q),
    entity_dim=facetdim,
    entities=facet_tags.find(12))
dof_outlet = fem.locate_dofs_topological(
    V=(W.sub(spc), Q),
    entity_dim=facetdim,
    entities=facet_tags.find(13))

p_inlet = fem.Function(Q)
p_inlet.interpolate(
    lambda xd:
    ScalarType(100.0) * np.ones((1, xd.shape[1])))
bc_pressure_inlet = fem.dirichletbc(
    value=p_inlet, dofs=dof_inlet, V=W)

p_outlet = fem.Function(Q)
p_outlet.interpolate(
    lambda xd:
    ScalarType(10.0) * np.ones((1, xd.shape[1])))
bc_pressure_outlet = fem.dirichletbc(
    value=p_outlet, dofs=dof_outlet, V=W)


"""
Solver configuration - We create a linear
solver object for this model. The petsc
options configures the PETSc library to use
a LU-based preconditioner with MUMPS as the 
underlying solver. These choices can significantly 
impact the performance and accuracy of the numerical 
computations performed using PETSc.
"""
# Solve
problem = fem.petsc.LinearProblem(
    a, L, bcs=[bc_pressure_inlet,
               bc_pressure_outlet],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"})

w_h = problem.solve()
velocity_h, pressure_h = w_h.split()

print("Solved! See the plots.")


"""
Visualization - We use PyVista Plotter object 
with a specific size and settings for line and 
polygon smoothing. It then creates two subplots 
within the plotter canvas to display the pressure 
and velocity fields separately.
Then, for each subplot, the code projects the 
simulation results onto a suitable PyVista data 
structure, which consists of cells, cell types, 
and coordinates. The projected solution is assigned
to the grid and visualized using different visualizations.
"""
fontsize = 16
zoom = 1.6

# Setting pyvista plotter's backend
pv.set_plot_theme(pv.themes.DocumentTheme())
pv.set_jupyter_backend('None')

canvas_length, canvas_breadth = 1200, 1200

# Initialize a pyvista canvas with 1 row and 2 columns.
p = pv.Plotter(shape=(2, 1),
               window_size=(canvas_length, canvas_breadth),
               multi_samples=8,
               line_smoothing=True,
               polygon_smoothing=True, border=False)
# p.add_title("Darcy flow simulation")

# Project the solutions suitable with pyvista.
V0_h = fem.FunctionSpace(domain, ("CG", 2))
uh = fem.Function(V0_h, dtype=np.float64)
uh.interpolate(pressure_h)

# Get the cells, types and coordinates from the mesh.
cells, cell_types, x_ = plot.create_vtk_mesh(V0_h)

# Define an unstructured grid to store the data.
grid = pv.UnstructuredGrid(cells, cell_types, x_)

# Assign the projected solution onto the grid.
grid.point_data["u"] = uh.x.array.reshape(
    x_.shape[0], V0_h.dofmap.index_map_bs)

# Select the 1st panel in the canvas.
p.subplot(0, 0)
p.add_text('Pressure',
           position=(0.45, 0.75),
           viewport=True,
           shadow=True,
           font_size=fontsize)

# Set color map, zoom level, and other settings.
p.add_mesh(
    mesh=grid.warp_by_scalar(
        scalars="u", factor=0.0),
    cmap="turbo",
    scalar_bar_args={'title': "(Pa)",
                     'label_font_size': fontsize + 8,
                     'fmt': '%10.2f',
                     'position_x': 0.25,
                     'position_y': 0.01,
                     'bold': False,
                     'width': 0.5,
                     'height': 0.2,
                     'n_labels': 4}
)

# p.add_bounding_box()
p.show_bounds(ticks='both',
              xlabel='length (m)',
              ylabel='height (m)',
              use_2d=True,
              all_edges=True,
              font_size=fontsize + 4)

# Display the XY plane
p.view_xy()

# Define the zoom level
p.camera.zoom(value=zoom)

V0_sigma = fem.VectorFunctionSpace(
    domain, ("CG", 2))
usigma = fem.Function(
    V0_sigma, dtype=np.float64)
usigma.interpolate(velocity_h)
cells, cell_types, x_ = plot.create_vtk_mesh(
    V0_sigma)
grid = pv.UnstructuredGrid(cells, cell_types, x_)
grid.point_data["u"] = uh.x.array.reshape(
    x_.shape[0], V0_h.dofmap.index_map_bs)

# Rearranging data for glyphs
points2d = usigma.x.array.reshape(
    x_.shape[0], V0_sigma.dofmap.index_map_bs)
points3d = np.hstack(
    [points2d, np.zeros([points2d.shape[0], 1])])
grid["vectors"] = points3d
grid.set_active_vectors("vectors")
glyphs = grid.glyph(
    scale=1, orient="vectors", tolerance=0.025,
    factor=0.5, geom=pv.Arrow().scale(
        xyz=(1, 1, 0.07), inplace=True))

p.subplot(1, 0)
p.add_mesh(mesh=glyphs,
           cmap="coolwarm", color="white",
           show_scalar_bar=False)

p.subplot(1, 0)
p.add_text('Velocity',
           position=(0.45, 0.75),
           viewport=True,
           shadow=True,
           font_size=fontsize)
p.add_mesh(
    mesh=grid.warp_by_scalar(
        scalars="u", factor=0.0),
    cmap="turbo",
    scalar_bar_args={'title': "(m/s)",
                     'label_font_size': fontsize + 8,
                     'fmt': '%10.2f',
                     'position_x': 0.25,
                     'position_y': 0.01,
                     'bold': False,
                     'width': 0.5,
                     'height': 0.2,
                     'n_labels': 4})

p.show_bounds(ticks='both',
              xlabel='length (m)',
              ylabel='height (m)',
              use_2d=True,
              all_edges=True,
              font_size=fontsize + 4,
              location='outer')

# Display the XY plane, set zoom level
# and show a title on the window
p.view_xy()
p.camera.zoom(value=zoom)
p.show(title="Seepage flow simulation",
       screenshot=resultPath)


print("Done!")
