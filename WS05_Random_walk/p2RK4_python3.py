"""
Worksheet 5, Problem 2(b)  --  Python 3 corrected version of p2RK4.py

Solve the 1-d diffusion equation

        d(rho)/dt = D d2(rho)/dx2,      D = (dx)^2 / (2 dt)

by RK4, with the initial condition set at t = 1 to

        rho(x) = exp(-x^2 / (2 sigma^2)) / sqrt(2 pi sigma^2),   sigma^2 = 2 D t

on x in [-2, 2] with dx = 0.01.  Plot rho(x, t) at three times.

Changes made relative to the original p2RK4.py
----------------------------------------------
1. The right-hand side divided by t1 = t[i], the CURRENT TIME, where the
   TIME STEP belonged.  Same bug as in p2E.py; see that file for the
   algebra.  Corrected.
2. The initial condition used sigma^2 where sigma belonged, planting a
   Gaussian of width 0.001 on a grid of spacing 0.01 -- a one-cell spike.
   Corrected.
3. The RK4 stages were not RK4.  The original wrote
        k2 = dt * f1(p1 + k1/2, p2, p3, t + 0.5)
   which advances only the CENTRE point between stages and leaves the two
   neighbours p2, p3 frozen at their stage-0 values.  Because the spatial
   operator couples neighbouring cells, every stage has to advance the whole
   vector.  RK4 is now applied to the full state array.
4. The stage times were hard-coded as t+0.5 and t+1 instead of t+dt/2 and
   t+dt.  Here the operator has no explicit time dependence so this made no
   numerical difference, but it would break the moment a source term were
   added.  Corrected anyway.
5. The three output times are named constants instead of array indices
   15, 500 and N-1.
6. The result is compared against the exact Gaussian at each output time.

Algorithms are hand written: the Laplacian and all four RK4 stages are
implemented here.

Stability
---------
With D = dx^2/(2 dt) the diffusion number is r = 1/2, so the eigenvalues of
the discrete Laplacian satisfy  lambda * dt  in  [-4r, 0] = [-2, 0].  RK4 is
stable on the negative real axis out to about -2.785, so this scheme is
comfortably inside its stability region -- unlike explicit Euler, which sits
exactly on its own limit at r = 1/2.
"""

import numpy as np
import matplotlib.pyplot as plt


# --- grid and physical parameters ----------------------------------------
DX = 0.01
X_MIN, X_MAX = -2.0, 2.0
DT = 0.0025
D = DX ** 2 / (2.0 * DT)        # as prescribed by the worksheet -> 0.02

T_START = 1.0
OUTPUT_TIMES = [1.0, 4.0, 9.0]


def exact_profile(x, t):
    """Exact solution at time t: a Gaussian of VARIANCE sigma^2 = 2 D t."""
    var = 2.0 * D * t
    return np.exp(-x ** 2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def rhs(rho):
    """
    D * d2(rho)/dx2 by the three-point central difference, with rho held at
    zero on the two boundaries.  Acts on the WHOLE state vector -- this is
    the part the original RK4 got wrong.
    """
    out = np.zeros_like(rho)
    out[1:-1] = D * (rho[2:] + rho[:-2] - 2.0 * rho[1:-1]) / DX ** 2
    return out


def rk4_step(rho, dt):
    """One classic RK4 step of the full vector state."""
    k1 = dt * rhs(rho)
    k2 = dt * rhs(rho + 0.5 * k1)
    k3 = dt * rhs(rho + 0.5 * k2)
    k4 = dt * rhs(rho + k3)
    return rho + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def main():
    x = np.arange(X_MIN, X_MAX + 0.5 * DX, DX)
    rho = exact_profile(x, T_START)

    n_steps = int(round((OUTPUT_TIMES[-1] - T_START) / DT))
    want = {int(round((t - T_START) / DT)): t for t in OUTPUT_TIMES}

    snapshots = []

    print("dx = %.4f   dt = %.4f   D = dx^2/(2 dt) = %.4f" % (DX, DT, D))
    print("diffusion number r = D dt / dx^2 = %.4f\n" % (D * DT / DX ** 2))
    print("%-8s %-14s %-16s %-16s"
          % ("t", "sigma", "mass (exact 1)", "max |rho - exact|"))

    def report(t, profile):
        exact = exact_profile(x, t)
        print("%-8.2f %-14.4f %-16.6f %-16.3e"
              % (t, np.sqrt(2.0 * D * t), np.sum(profile) * DX,
                 np.max(np.abs(profile - exact))))

    if 0 in want:
        snapshots.append((want[0], rho.copy()))
        report(want[0], rho)

    for step in range(1, n_steps + 1):
        rho = rk4_step(rho, DT)

        if step in want:
            t = want[step]
            snapshots.append((t, rho.copy()))
            report(t, rho)

    fig, ax = plt.subplots(figsize=(8, 5))
    for t, profile in snapshots:
        line, = ax.plot(x, profile, lw=1.4, label="RK4, t = %.1f" % t)
        ax.plot(x, exact_profile(x, t), "--", lw=1.0,
                color=line.get_color())
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho(x,t)$")
    ax.set_title("1-d diffusion by RK4\n"
                 "(dashed: exact Gaussian of variance 2Dt)")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
