"""
Worksheet 5, Problem 2(a)  --  Python 3 corrected version of p2E.py

Solve the 1-d diffusion equation

        d(rho)/dt = D d2(rho)/dx2,      D = (dx)^2 / (2 dt)

by the EULER method, with the initial condition set at t = 1 to

        rho(x) = exp(-x^2 / (2 sigma^2)) / sqrt(2 pi sigma^2),   sigma^2 = 2 D t

on x in [-2, 2] with dx = 0.01.  Plot rho(x, t) at three times.

Changes made relative to the original p2E.py
--------------------------------------------
1. The right-hand side was
        (rho_plus + rho_minus - 2 rho) / (2.0 * t1)
   with t1 = t[i], the CURRENT TIME, where it should be the time STEP.
   With D = dx^2/(2 dt) the discrete Laplacian
        D (rho_+ + rho_- - 2 rho) / dx^2
   reduces exactly to  (rho_+ + rho_- - 2 rho) / (2 dt);  the original
   divided by t instead of dt, which makes the diffusion coefficient decay
   like 1/t.  Corrected.
2. The initial condition computed  si = t * dx^2 / dt,  which IS sigma^2,
   and then substituted it into  exp(-x^2/(2 si^2)) / sqrt(2 pi si^2)  -- so
   sigma^2 was used where sigma belonged.  With the original numbers that
   made sigma = 0.001 on a grid of spacing dx = 0.01: the initial profile was
   a single-cell spike, not a resolved Gaussian.  Corrected to use sigma^2
   directly.
3. dt is now chosen so that the initial width sigma = 0.2 is comfortably
   resolved by the grid and the profile stays inside the box at the latest
   plotted time.  The unused variable M (number of particles) is dropped.
4. The three output times are named constants instead of hard-coded array
   indices 15, 500 and N-1.
5. The result is compared against the exact Gaussian solution at each output
   time, and the conserved mass is reported.

Algorithms are hand written: the finite-difference Laplacian and the Euler
step are both implemented here.

A note on the diffusion number
------------------------------
With D = dx^2/(2 dt) the diffusion number is

        r = D dt / dx^2 = 1/2

exactly, and the Euler update collapses to

        rho_new(x) = ( rho(x + dx) + rho(x - dx) ) / 2

which is precisely the lattice random walk of Problem 1.  That is not a
coincidence -- it is the discrete statement that diffusion is the continuum
limit of the random walk.  r = 1/2 is also the stability limit of explicit
Euler for this equation, so this scheme sits exactly on the boundary: stable,
but with no damping of the shortest wavelength.
"""

import numpy as np
import matplotlib.pyplot as plt


# --- grid and physical parameters ----------------------------------------
DX = 0.01
X_MIN, X_MAX = -2.0, 2.0
DT = 0.0025
D = DX ** 2 / (2.0 * DT)        # as prescribed by the worksheet -> 0.02

T_START = 1.0                   # the worksheet sets the initial condition here
OUTPUT_TIMES = [1.0, 4.0, 9.0]


def initial_profile(x, t):
    """
    Exact solution at time t:  a Gaussian of variance sigma^2 = 2 D t.

    NOTE the distinction the original lost: `var` below is sigma SQUARED, so
    it appears as  exp(-x^2 / (2 var)) / sqrt(2 pi var)  -- not var^2.
    """
    var = 2.0 * D * t
    return np.exp(-x ** 2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def laplacian(rho):
    """
    D * d2(rho)/dx2 by the three-point central difference, with rho held at
    zero on the two boundaries (Dirichlet).  Returns an array the same shape
    as rho whose first and last entries are zero.
    """
    out = np.zeros_like(rho)
    out[1:-1] = D * (rho[2:] + rho[:-2] - 2.0 * rho[1:-1]) / DX ** 2
    return out


def main():
    x = np.arange(X_MIN, X_MAX + 0.5 * DX, DX)
    rho = initial_profile(x, T_START)

    n_steps = int(round((OUTPUT_TIMES[-1] - T_START) / DT))
    want = {int(round((t - T_START) / DT)): t for t in OUTPUT_TIMES}

    snapshots = []

    print("dx = %.4f   dt = %.4f   D = dx^2/(2 dt) = %.4f" % (DX, DT, D))
    print("diffusion number r = D dt / dx^2 = %.4f  (stability limit 0.5)\n"
          % (D * DT / DX ** 2))
    print("%-8s %-14s %-16s %-16s"
          % ("t", "sigma", "mass (exact 1)", "max |rho - exact|"))

    def report(t, profile):
        exact = initial_profile(x, t)
        print("%-8.2f %-14.4f %-16.6f %-16.3e"
              % (t, np.sqrt(2.0 * D * t), np.sum(profile) * DX,
                 np.max(np.abs(profile - exact))))

    if 0 in want:
        snapshots.append((want[0], rho.copy()))
        report(want[0], rho)

    for step in range(1, n_steps + 1):
        # ---- explicit Euler: rho_{n+1} = rho_n + dt * L[rho_n] ----------
        rho = rho + DT * laplacian(rho)

        if step in want:
            t = want[step]
            snapshots.append((t, rho.copy()))
            report(t, rho)

    fig, ax = plt.subplots(figsize=(8, 5))
    for t, profile in snapshots:
        line, = ax.plot(x, profile, lw=1.4, label="Euler, t = %.1f" % t)
        ax.plot(x, initial_profile(x, t), "--", lw=1.0,
                color=line.get_color())
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho(x,t)$")
    ax.set_title("1-d diffusion by explicit Euler\n"
                 "(dashed: exact Gaussian of variance 2Dt)")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
