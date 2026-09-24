"""
Worksheet 1, Problem 1  --  Python 3 corrected version of p1.py

Integrate  d2(theta)/dt2 = -k*theta  with the Euler and the Euler-Cromer
methods and plot the total energy against time for three different initial
conditions (three different initial energies).

Changes made relative to the original p1.py
-------------------------------------------
1. Arrays are allocated with np.zeros() and the initial value written
   explicitly.  The original used np.linspace(b, 1000, 1000), which happens
   to put the right number in element 0 but says nothing about intent.
2. The three initial conditions are looped over instead of two of them being
   commented out at the bottom of the file.
3. plt.legend() is only called on axes that actually carry labelled curves,
   so the run is warning-free.
4. Prints the relative energy drift of each method, which is the quantity
   the exercise is really about.

Algorithms are hand written: no ODE solver is imported.

Euler          :  v_{n+1} = v_n - k x_n dt      x_{n+1} = x_n + v_n     dt
Euler-Cromer   :  v_{n+1} = v_n - k x_n dt      x_{n+1} = x_n + v_{n+1} dt

The single difference is which velocity drives the position update.  Euler is
not symplectic and the energy grows without bound; Euler-Cromer is, and the
energy merely oscillates about its initial value.
"""

import numpy as np
import matplotlib.pyplot as plt


def integrate(theta0, omega0, k, t_end=10.0, n_steps=1000):
    """Return (t, E_euler, E_cromer) for one initial condition."""
    t = np.linspace(0.0, t_end, n_steps)
    dt = t[1] - t[0]

    # --- Euler-Cromer ----------------------------------------------------
    x_c = np.zeros(n_steps)
    v_c = np.zeros(n_steps)
    E_c = np.zeros(n_steps)
    x_c[0], v_c[0] = theta0, omega0
    E_c[0] = 0.5 * (omega0 ** 2 + k * theta0 ** 2)

    # --- plain Euler -----------------------------------------------------
    x_e = np.zeros(n_steps)
    v_e = np.zeros(n_steps)
    E_e = np.zeros(n_steps)
    x_e[0], v_e[0] = theta0, omega0
    E_e[0] = E_c[0]

    for i in range(1, n_steps):
        # Euler-Cromer: the *new* velocity moves the position
        v_c[i] = v_c[i - 1] - k * x_c[i - 1] * dt
        x_c[i] = x_c[i - 1] + v_c[i] * dt
        E_c[i] = 0.5 * (v_c[i] ** 2 + k * x_c[i] ** 2)

        # plain Euler: the *old* velocity moves the position
        v_e[i] = v_e[i - 1] - k * x_e[i - 1] * dt
        x_e[i] = x_e[i - 1] + v_e[i - 1] * dt
        E_e[i] = 0.5 * (v_e[i] ** 2 + k * x_e[i] ** 2)

    return t, E_e, E_c


def main():
    k = 0.2
    # (theta0, omega0) -- three different initial energies
    conditions = [(1.0, 2.0), (2.0, 3.0), (2.0, 5.0)]

    fig, axes = plt.subplots(len(conditions), 2, figsize=(11, 9), sharex=True)

    for row, (theta0, omega0) in enumerate(conditions):
        t, E_e, E_c = integrate(theta0, omega0, k)
        E0 = E_c[0]

        drift_e = (E_e[-1] - E0) / E0 * 100.0
        drift_c = (E_c[-1] - E0) / E0 * 100.0
        print("theta0 = %5.2f  omega0 = %5.2f  E0 = %7.4f   "
              "drift: Euler %+8.2f %%   Euler-Cromer %+6.2f %%"
              % (theta0, omega0, E0, drift_e, drift_c))

        axes[row, 0].plot(t, E_e, label="Euler")
        axes[row, 0].axhline(E0, ls="--", lw=0.8, color="grey", label="E(0)")
        axes[row, 0].set_ylabel("Energy")
        axes[row, 0].set_title(r"Euler   $\theta_0$=%.1f, $\dot\theta_0$=%.1f, $E_0$=%.3f"
                               % (theta0, omega0, E0))
        axes[row, 0].legend(fontsize=8)

        axes[row, 1].plot(t, E_c, color="tab:orange", label="Euler-Cromer")
        axes[row, 1].axhline(E0, ls="--", lw=0.8, color="grey", label="E(0)")
        axes[row, 1].set_title(r"Euler-Cromer   $E_0$=%.3f" % E0)
        axes[row, 1].legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.suptitle("Total energy vs time,  k = %.2f" % k)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
