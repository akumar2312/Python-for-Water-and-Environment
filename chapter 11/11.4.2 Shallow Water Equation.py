"""
Shallow water simulation using the
Finite Volume Method (FVM) in
rectangular domain
"""

import numpy as np
import pyvista as pv

"""
File I/O
"""
simulationName = "swe_FVM"
resultPath = "./result/" + simulationName + "/"


"""
Computational domain creation
"""
xlen = 2
ylen = 7
divs = 60
dx = xlen / divs
dy = ylen / divs

x = np.arange(-1 - dx, 1 + dx, dx)
y = np.arange(-1 - dy, 1 + dy, dy)
xx, yy = np.meshgrid(x, y)
print("Min {}, {}".format(np.min(xx), np.min(yy)))
print("Max {}, {}".format(np.max(xx), np.max(yy)))


"""
Model parameters
"""
g = 9.8
cfl = 0.5


# Source
def eta_initial(y, x):
    """Intial eta values function."""
    Amp = 8.0
    μx, μy = -0.1, -0.9
    σ = 0.3
    return Amp * np.exp(
        -1 * (0.02*(x - μx) ** 2 +
              (y - μy) ** 2) / (σ ** 2))


# Base depth + Gaussian amplitude
eta = np.ones_like(xx)
eta = eta + eta_initial(yy, xx)

# Solution variable initializing as [mesh_size x 3] shape
Solution = np.zeros([xx.shape[0], xx.shape[1], 3])
Solution[:, :, 0] = eta  # Zero index corresponds to etas
u = np.zeros_like(xx)
v = np.copy(u)

# Shifting the indices by 1 for easing the BC application
p_p1_x = np.roll(np.arange(len(x)), 1)
p_p1_y = np.roll(np.arange(len(y)), 1)
p_m1_x = np.roll(np.arange(len(x)), -1)
p_m1_y = np.roll(np.arange(len(y)), -1)

# Running parameters
t_curr = 0.0
dt = 0.0
t_stop = 2.0

# Solution variables
etas = list()
times = list()
times.append(t_curr)
Solution_old = Solution  # old solution variable
counter = 0

"""
Time-marching
"""
while t_curr < t_stop:
    # Save files
    hsol = Solution[:, :, 0]
    grid = pv.StructuredGrid(xx * xlen, yy * ylen, hsol)
    grid.point_data["height"] = hsol.flatten(order="F")

    top = grid.points.copy()
    bottom = grid.points.copy()
    bottom[:, -1] = -5.0  # Bottom plane
    vol = pv.StructuredGrid()
    vol.points = np.vstack([top, bottom])
    vol.dimensions = [*grid.dimensions[0:2], 2]

    z_above = hsol.flatten(order="F")
    zlevels = vol.points[:, 2]
    zlevels[zlevels < np.min(z_above)] = -0.5
    vol.point_data["height"] = zlevels

    write_format = resultPath + \
                   "height_" + \
                   "{:04d}.vtk".format(counter)
    vol.save(write_format)


    # Initialising variables
    u_old = u  # old x-velocity
    v_old = v  # old y-velocity

    t_curr = t_curr + dt  # incrementing time

    times.append(t_curr)

    # calculate alpha = abs(u) + sqrt(gh) used for finding flux
    alpha_u = 0.5 * np.abs(u_old +
                           u_old[:, p_m1_x]) + \
              np.sqrt(
        g * 0.5 * (Solution_old[:, :, 0] +
                   Solution_old[:, p_m1_x, 0]))
    alpha_v = 0.5 * np.abs(v_old +
                           v_old[p_m1_y, :]) + \
              np.sqrt(
        g * 0.5 * (Solution_old[:, :, 0] +
                   Solution_old[p_m1_y, :, 0]))

    # compute maximum alpha
    alpha_max = np.linalg.norm(
        x=np.hstack([np.ravel(alpha_u), np.ravel(alpha_v)]),
        ord=np.inf)

    # computing dt on the fly for stability
    dt = np.min([cfl * (dx / alpha_max),
                 cfl * (dy / alpha_max)])

    # pre-computing some terms
    huv = Solution_old[:, :, 1] * Solution_old[:, :, 2] / \
          Solution_old[:, :, 0]
    gh_sqr = 0.5 * g * Solution_old[:, :, 0] ** 2

    # compute (hu, hu ** 2 + gh ** 2 / 2, huv)
    LFFlux_u = np.stack([Solution_old[:, :, 1],
                         Solution_old[:, :, 1] ** 2 /
                         Solution_old[:, :, 0] + gh_sqr,
                         huv],
                        2)

    # compute (hv, huv, hv ** 2 + gh ** 2 / 2)
    LFFlux_v = np.stack([Solution_old[:, :, 2],
                         huv,
                         Solution_old[:, :, 2] ** 2. /
                         Solution_old[:, :, 0] + gh_sqr],
                        2)

    # compute fluxes in x, y direction
    flux_x = np.zeros_like(LFFlux_u)
    flux_y = np.zeros_like(LFFlux_u)
    for ii in range(flux_x.shape[2]):
        temp_LFFluxu = LFFlux_u[:, p_m1_x, :]
        temp_Uold = Solution_old[:, p_m1_x, :] - \
                    Solution_old
        flux_x[:, :, ii] = 0.5 * (
                LFFlux_u[:, :, ii] +
                temp_LFFluxu[:, :, ii]) - \
                           0.5 * np.multiply(
            temp_Uold[:, :, ii], alpha_u)

    for ii in range(flux_y.shape[2]):
        temp_LFFluxv = LFFlux_v[p_m1_y, :, :]
        temp_Uold = Solution_old[p_m1_y, :, :] - \
                    Solution_old
        flux_y[:, :, ii] = 0.5 * (
                LFFlux_v[:, :, ii] +
                temp_LFFluxv[:, :, ii]) - \
                           0.5 * np.multiply(
            temp_Uold[:, :, ii], alpha_v)

    # assiging to solution variable
    Solution = Solution_old - \
               (dt / dx) * (flux_x - flux_x[:, p_p1_x, :]) -\
               (dt / dy) * (flux_y - flux_y[p_p1_y, :, :])

    # imposing no-slip boundary conditions on eta, hu and hv,
    # respectively
    Solution[0:, -1, 0] = Solution[0:, -2, 0]
    Solution[0:, 0, 0] = Solution[0:, 1, 0]
    Solution[-1, 0:, 0] = Solution[-2, 0:, 0]
    Solution[0, 0:, 0] = Solution[1, 0:, 0]

    Solution[0:, -1, 1] = -Solution[0:, -2, 1]
    Solution[0:, 0, 1] = -Solution[0:, 1, 1]
    Solution[-1, 0:, 1] = Solution[-2, 0:, 1]
    Solution[0, 0:, 1] = Solution[1, 0:, 1]

    Solution[0:, -1, 2] = Solution[0:, -2, 2]
    Solution[0:, 0, 2] = Solution[0:, 1, 2]
    Solution[-1, 0:, 2] = -Solution[-2, 0:, 2]
    Solution[0, 0:, 2] = -Solution[1, 0:, 2]

    # Retrieving u and v
    u = np.divide(Solution[:, :, 1], Solution[:, :, 0])
    v = np.divide(Solution[:, :, 2], Solution[:, :, 0])

    print("Done timestep {:.04f}".format(t_curr))
    etas.append(Solution[:, :, 0])
    Solution_old = Solution  # Assigning new solution to old
    counter = counter + 1

print("Done!")
