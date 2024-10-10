"""
Program to simulate the kinematic wave equation
for a circular channel
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

"""
Simulation parameters
"""
x_start = 0  # (m)
x_stop = 100  # (m)
NX = 50  # number of spatial grid point
x, dx = np.linspace(x_start, x_stop, NX, retstep=True)

t_start = 0  # (s)
t_stop = 40  # (s)
NT = 100  # number of temporal grid points
dt = (t_stop - t_start) / NT

"""
Courant_Friedrichs Lewy (CFL) condition
"""
CFL = 0.5
v = 1.0
dt_cfl = CFL * dx / np.abs(v)

if dt <= dt_cfl:
    print("dt < dt_CFL")
    print(dt, "<", dt_cfl)
    print("Algorithm is stable - proceed")

    # Setting model parameters
    sim_name = "Circ"
    n = 0.025  # roughness coefficient
    S = 0.01  # slope of channel
    D = 2.00  # diameter of channel
    alpha = (0.804/n) * S**(1/2) * D**(1/6)
    beta = 5 / 4

    # Rainfall or lateral inflow
    inflow = 0.1  # (m^3 / s)
    time_to_inflow = 10  # (s)

    # Solution variables
    A = np.zeros([NX, NT])  # time along the columns
    t_vals = np.zeros([NT, ])  # storing in actual units

    # Initial condition
    A[:, 0] = 0.005

    # Time loop
    for t in range(0, NT - 1):
        # Storing the time values in seconds
        t_vals[t] = t * dt

        # Lower inflow to occur after some time
        if t < time_to_inflow:
            q = inflow
        else:
            q = inflow / 3

        # Space loop
        for x in range(1, NX):
            A_mean = 0.5 * (A[x, t] + A[x - 1, t + 1])
            A[x, t + 1] = np.divide(dx*q + (dx/dt)
                                    * A_mean +
                                    alpha*beta*A_mean*
                                    A[x - 1, t + 1],
                                    (dx/dt) +
                                    alpha*beta*A_mean)

    # Computing discharge based on area
    Q = alpha * A**beta

    # Plot
    t_vals[-1] = t_vals[-2] + dt
    plt.plot(t_vals, Q[int(10), :], "-x",
             label="x=" + str(int(10)) + "m")
    plt.plot(t_vals, Q[int(NX / 2), :], "-s",
             label="x=" + str(int(NX / 2)) + "m")
    plt.plot(t_vals, Q[int(NX / 2) + 10, :], "-+",
             label="x=" + str(int(NX / 2) + 10) + "m")
    plt.xlabel("Time $(s)$")
    plt.ylabel("Discharge $(m^3/s)$")
    plt.ylim([-0.1, 3.5])
    plt.legend()
    plt.grid(ls="--")
    plt.tight_layout()
    plt.savefig("./result/" + sim_name + "_.png", dpi=300)

else:
    print("dt > dt_CFL")
    print(dt, ">", dt_cfl)
    print("Unstable - reduce dt!")
    print("Program exiting!")
    exit()

print("Done!")

