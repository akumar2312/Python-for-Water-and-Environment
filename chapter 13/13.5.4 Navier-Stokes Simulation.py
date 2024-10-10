"""
Program for Navier-Stokes simulation
on a 2D domain with 1 inlet and 1 outlet
"""
import os
import sys
import ufl
from ufl import (div, dot, dx, inner,
                 grad, nabla_grad,
                 lhs, rhs)
from dolfinx.io.gmshio import read_from_msh
from dolfinx.io import XDMFFile
from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI
import tqdm.autonotebook
import numpy as np
import h5py

"""
File handling block: We load an externally generated
mesh from GMSH. We also specify the location of the 
mesh (.msh) file and the location where the results
would be stored.
"""
simulationName = "Stream_NS-2D"
meshName = "aquifer2D"
meshPath = "./meshes_gmsh/" + meshName + ".msh"
# resultPath = "./result/" + simulationName + "/"
resultPath = "/home/anil/Desktop/py/result/" + \
             simulationName + "/"

if os.path.isfile(meshPath):
    """
    Load external mesh that was generated using GMSH.
    The external mesh has dimension=3 although the
    z-coordinates are just zeros. Therefore, we use
    gdim=2, to specify the actual dimension of the 
    problem.
    """
    gdim = 2
    domain, cell_tags, facet_tags = read_from_msh(
        filename=meshPath,
        comm=MPI.COMM_WORLD,
        rank=0, gdim=gdim)

    # Integer markers are set during mesh creation in GMSH
    inlet_marker = 13
    outlet_marker = 14
    noflow_marker = 15
    island_marker = 16

    # Facet dimension for boundary condition
    fdim = domain.topology.dim - 1

    # -------------------------------------------------------
    """
    Setting material properties and simulation times. 
    Kinematic viscosity and density are the only
    parameters that are specified.
    """
    # Representative Water Kinematic viscosity
    mu_water_kinetic = fem.Constant(domain=domain,
                                    c=PETSc.ScalarType(0.0089))
    # Representative Water Density
    rho_water = fem.Constant(domain=domain,
                             c=PETSc.ScalarType(1.0))
    t_initial = 0  # Time initial
    t_final = 3  # Time final
    dt = 1 / 1500  # Stepping
    n_steps = int((t_final - t_initial) / dt)  # N steps
    k = fem.Constant(domain=domain,  # Convert to PETSc type
                     c=PETSc.ScalarType(dt))

    """
    Finite elements and Function spaces: We specify
    two element types; vector Lagrange element for the
    velocity field and a simple Lagrange element for 
    the pressure field. 
    """
    LG2_elem = ufl.VectorElement(family="Lagrange",
                                 cell=domain.ufl_cell(),
                                 degree=2)
    LG1_elem = ufl.FiniteElement(family="Lagrange",
                                 cell=domain.ufl_cell(),
                                 degree=1)
    Vspace = fem.FunctionSpace(mesh=domain, element=LG2_elem)
    Qspace = fem.FunctionSpace(mesh=domain, element=LG1_elem)


    class VelocityProfileAtInlet:
        """
        Time dependent Boundary Condition: We intend to
        vary the inlet velocity to have a profile
        according to a given equation. This is realized
        using a time-dependent boundary condition helping
        to change the velocity as a function of time.
        """

        def __init__(self, t):
            self.t = t

        def __call__(self, x):
            velocity_vector = np.zeros((2, x.shape[1]),
                                       dtype=PETSc.ScalarType)
            velocity_vector[0] = 10.0 * (1.001 + np.sin(
                2 * self.t * np.pi / 3
            )) * x[1] * (3.5 - x[1]) / (3.5 ** 2)
            return velocity_vector


    # BC at Inlet
    u_at_inlet = fem.Function(V=Vspace)
    inlet_velocity = VelocityProfileAtInlet(t=t_initial)
    u_at_inlet.interpolate(u=inlet_velocity)

    # checking dof during assignment of velocity BC in SWE34
    doff = fem.locate_dofs_topological(
            V=Vspace,
            entity_dim=fdim,
            entities=facet_tags.find(inlet_marker))

    bc_u_inflow = fem.dirichletbc(
        value=u_at_inlet,
        dofs=doff
    )

    # No-slip (at Walls)
    u_noslip = np.array(
        tuple([0, ]) * gdim, dtype=PETSc.ScalarType)
    bc_u_walls = fem.dirichletbc(
        value=u_noslip,
        dofs=fem.locate_dofs_topological(
            V=Vspace, entity_dim=fdim,
            entities=facet_tags.find(noflow_marker)
        ),
        V=Vspace
    )

    # No-slip (on Islands)
    bc_u_island = fem.dirichletbc(
        value=u_noslip,  # as this vector is already created
        dofs=fem.locate_dofs_topological(
            V=Vspace,
            entity_dim=fdim,
            entities=facet_tags.find(island_marker)
        ),
        V=Vspace)
    bc_on_velocity = [bc_u_inflow,
                      bc_u_island,
                      bc_u_walls]

    # BC at Outlet
    bc_p_outlet = fem.dirichletbc(
        value=PETSc.ScalarType(0.0),
        dofs=fem.locate_dofs_topological(
            V=Qspace,
            entity_dim=fdim,
            entities=facet_tags.find(outlet_marker)
        ),
        V=Qspace)
    bc_on_pressure = [bc_p_outlet]

    """
    Defining Test and Trial Functions: We define 
    the trial and test functions to be used in the
    weak form.
    """
    u = ufl.TrialFunction(function_space=Vspace)
    v = ufl.TestFunction(function_space=Vspace)

    u_ = fem.Function(V=Vspace)
    u_.name = "velocity"

    u_s = fem.Function(V=Vspace)

    u_n = fem.Function(V=Vspace)
    u_n1 = fem.Function(V=Vspace)

    p = ufl.TrialFunction(function_space=Qspace)
    q = ufl.TestFunction(function_space=Qspace)
    p_ = fem.Function(V=Qspace)
    p_.name = "pressure"

    phi = fem.Function(V=Qspace)

    # To be used in the weak-form
    f = fem.Constant(
        domain=domain,
        c=PETSc.ScalarType((0.0, 0.0)))

    """
    Specifying the variational formulation
    for the 3 steps in Chorin's method
    """
    # First weak form in Chorin's method
    F = rho_water / k * dot(
        u - u_n, v) * dx
    F = F + inner(
        dot(
            (3 / 2) * u_n - (1 / 2) * u_n1,
            (1 / 2) * nabla_grad(u + u_n)
        ), v
    ) * dx
    F = F + (1 / 2) * mu_water_kinetic * inner(
        grad(u + u_n), grad(v)) * dx
    F = F - dot(
        p_, div(v)) * dx
    F = F + dot(
        f, v) * dx

    a1 = fem.form(lhs(F))
    L1 = fem.form(rhs(F))

    # Second weak form in Chorin's method
    a2 = fem.form(dot(grad(p),
                      grad(q)) * dx)
    L2 = fem.form(-rho_water / k * dot(div(u_s),
                                           q) * dx)

    # Third weak form in Chorin's method
    a3 = fem.form(rho_water * dot(u, v) * dx)
    L3 = fem.form(rho_water * dot(
        u_s, v) * dx - k * dot(nabla_grad(phi),
                               v) * dx)

    """
    Forming linear system
    """
    A1 = fem.petsc.create_matrix(a=a1)
    b1 = fem.petsc.create_vector(L=L1)

    A2 = fem.petsc.assemble_matrix(a2, bc_on_pressure)
    b2 = fem.petsc.create_vector(L=L2)

    A3 = fem.petsc.assemble_matrix(a=a3)
    b3 = fem.petsc.create_vector(L=L3)

    A2.assemble()
    A3.assemble()

    """
    Configuring solvers for Steps 1, 2 and 3
    """
    STEP1_solver = PETSc.KSP().create(domain.comm)
    STEP1_solver.setOperators(A1)
    STEP1_solver.setType(PETSc.KSP.Type.BCGS)
    STEP1_solver.getPC().setType(PETSc.PC.Type.JACOBI)

    STEP2_solver = PETSc.KSP().create(domain.comm)
    STEP2_solver.setOperators(A2)
    STEP2_solver.setType(PETSc.KSP.Type.MINRES)
    STEP2_solver.getPC().setHYPREType("boomeramg")
    STEP2_solver.getPC().setType(PETSc.PC.Type.HYPRE)

    STEP3_solver = PETSc.KSP().create(domain.comm)
    STEP3_solver.setOperators(A3)
    STEP3_solver.setType(PETSc.KSP.Type.CG)
    STEP3_solver.getPC().setType(PETSc.PC.Type.SOR)

    """
    Creating file objects for storing results
    """
    xdmfu = XDMFFile(comm=domain.comm,  # velocity result
                     filename=resultPath + "stream_u.xdmf",
                     file_mode="w",
                     encoding=XDMFFile.Encoding.HDF5)
    xdmfu.write_mesh(mesh=domain)
    xdmfu.write_function(u=u_,
                         t=t_initial)

    xdmfp = XDMFFile(comm=domain.comm,  # pressure result
                     filename=resultPath + "stream_p.xdmf",
                     file_mode="w",
                     encoding=XDMFFile.Encoding.HDF5)
    xdmfp.write_mesh(mesh=domain)
    xdmfp.write_function(u=p_,
                         t=t_initial)

    # for storing the time series velocity data
    usize = u_.x.array.shape[0]
    with h5py.File(
            name=resultPath + 'velocity_timeseries.h5',
            mode='w') as fr:
        dset = fr.create_dataset(name="u",
                                 shape=[n_steps, usize],
                                 dtype=np.float32)
        progress = tqdm.autonotebook.tqdm(
            desc="Solving PDE system",
            total=n_steps
        )

        """
        Time stepping through solutions
        """
        t_at = 0
        for i in range(n_steps):
            """
            Following operations are done:
            1) Update the new time
            2) Get the new velocity at the source
            3) Compute the tentative velocity
            4) Do the pressure correction
            5) Do the velocity correction
            6) Save the solutions
            7) Store the result for next time step
            """

            # 0) Tracking progress at every 10 steps
            progress.update(n=1)

            # 1) Update current time step
            t_at = t_at + dt

            # 2) Update inlet velocity
            inlet_velocity.t = t_at
            u_at_inlet.interpolate(u=inlet_velocity)

            # 3) Estimating velocity
            A1.zeroEntries()
            fem.petsc.assemble_matrix(
                A1, a1, bcs=bc_on_velocity)
            A1.assemble()

            with b1.localForm() as loc:
                loc.set(0)
            fem.petsc.assemble_vector(b1, L1)
            fem.petsc.apply_lifting(
                b=b1, a=[a1], bcs=[bc_on_velocity])
            b1.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(
                b=b1, bcs=bc_on_velocity)
            STEP1_solver.solve(b1, u_s.vector)
            u_s.x.scatter_forward()

            # 4): Doing Pressure correction
            with b2.localForm() as loc:
                loc.set(0)
            fem.petsc.assemble_vector(b2, L2)
            fem.petsc.apply_lifting(
                b=b2, a=[a2], bcs=[bc_on_pressure])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                           mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(
                b=b2, bcs=bc_on_pressure)
            STEP2_solver.solve(b2, phi.vector)
            phi.x.scatter_forward()

            p_.vector.axpy(1, phi.vector)
            p_.x.scatter_forward()

            # 5) Doing Velocity correction
            with b3.localForm() as loc:
                loc.set(0)
            fem.petsc.assemble_vector(b3, L3)
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                           mode=PETSc.ScatterMode.REVERSE)
            STEP3_solver.solve(b3, u_.vector)
            u_.x.scatter_forward()

            # 6) Storing the solutions
            xdmfu.write_function(u_, t_at)
            xdmfp.write_function(p_, t_at)
            u_data = u_.x.array
            dset[i] = u_data

            # 7) Assign future with current state
            with u_.vector.localForm() as loc_, \
                    u_n.vector.localForm() as loc_n, \
                    u_n1.vector.localForm() as loc_n1:
                loc_n.copy(loc_n1)
                loc_.copy(loc_n)

        # Close the file handles
        xdmfu.close()
        xdmfp.close()
    fr.close()
    print("Done!")

else:
    """
    1) Check if (.msh) file exists or not?
    2) Recommend to put GMSH file in meshPath.
    """
    if not os.path.exists(meshPath):
        print("\n\nMsg: Source path created!")
        os.makedirs(meshPath)

    if not os.path.exists(resultPath + "/"):
        """
        1) Create result folder if required.
        """
        os.makedirs(resultPath + "/")
        print("Msg: Result folder created!")

    if not os.path.isfile(meshPath + meshName + ".msh"):
        print("Msg: Check if GMSH mesh file (*.msh) exists?")
        sys.exit()
