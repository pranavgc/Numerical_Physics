"""
Worksheet 7, Problem 1  --  Python 3 corrected version of p1.py

Solve the boundary value problem

        y''(x) + 4 y(x) = 0,     x in [0, pi/4]
        y(0) = -2,   y(pi/4) = 10

by the shooting method.  Exact solution: y(x) = -2 cos(2x) + 10 sin(2x).
Plot exact and numerical solutions, and show the error as a function of the
step size.

Changes made relative to the original p1.py
-------------------------------------------
1. The step size inside the integrator was  dx = x[1] - x[2],  which is
   NEGATIVE.  The integrator therefore marched backwards while the grid ran
   forwards, so the secant iteration was matching the boundary value at the
   wrong end of the interval.  The same expression is written correctly as
   x[2] - x[1] twice elsewhere in the original file, so this was a
   transposition slip.  Corrected to x[1] - x[0].
2. The function packed two scalars and two arrays into one
   np.array([dx, r, y1, y2]).  NumPy allowed this as an object array before
   1.24 and raises ValueError now:
       "setting an array element with a sequence ... inhomogeneous shape".
   The script therefore crashes outright on any current NumPy.  Results are
   returned as a tuple instead.
3. The RK4 stage bookkeeping mixed up which slope belonged to which
   variable.  The original happened to stay self-consistent, but the names
   were misleading; the stages are now written out in the conventional order
   with the position and velocity slopes clearly paired.
4. The secant iteration recomputed F(v) once at the top of the loop and
   again inside it, doubling the work.  Values are now carried between
   iterations, and the loop has an iteration cap so a bad bracket cannot
   spin forever.
5. The error study used only N = 80, 90, 100, 110 -- four points spanning
   less than a factor of two in dx, far too narrow to see a convergence
   order.  It now spans a factor of 32 and the fitted order is printed.
6. The redundant "while ... else" is removed (with no break in the loop the
   else clause always runs).

Algorithms are hand written: RK4 and the secant root finder are both
implemented here.

How shooting works here
-----------------------
The BVP has a known y(0) but an unknown y'(0).  Guess y'(0) = s, integrate
to x = pi/4, and look at the miss  F(s) = y(pi/4; s) - 10.  Drive F to zero
with the secant method.  For a LINEAR problem like this one, F is an affine
function of s, so the secant method lands on the answer in a single step --
which is a useful sanity check on the implementation.
"""

import numpy as np
import matplotlib.pyplot as plt


X_LEFT = 0.0
X_RIGHT = np.pi / 4.0
Y_LEFT = -2.0
Y_RIGHT = 10.0


def f_y(v):
    """dy/dx = v"""
    return v


def f_v(y):
    """dv/dx = -4 y"""
    return -4.0 * y


def integrate(slope, n):
    """
    RK4 from x = 0 to x = pi/4 with y(0) = -2 and y'(0) = slope.

    Returns (x, y).  Note dx = x[1] - x[0] > 0: the original used
    x[1] - x[2], which is the negative of this.
    """
    x = np.linspace(X_LEFT, X_RIGHT, n)
    dx = x[1] - x[0]

    y = np.zeros(n)
    v = np.zeros(n)
    y[0], v[0] = Y_LEFT, slope

    for i in range(n - 1):
        k1y = f_y(v[i])
        k1v = f_v(y[i])

        k2y = f_y(v[i] + 0.5 * dx * k1v)
        k2v = f_v(y[i] + 0.5 * dx * k1y)

        k3y = f_y(v[i] + 0.5 * dx * k2v)
        k3v = f_v(y[i] + 0.5 * dx * k2y)

        k4y = f_y(v[i] + dx * k3v)
        k4v = f_v(y[i] + dx * k3y)

        y[i + 1] = y[i] + dx * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        v[i + 1] = v[i] + dx * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0

    return x, y


def shoot(n, s0=1.0, s1=-10.0, tol=1e-10, max_iter=50):
    """
    Secant iteration on the initial slope until y(pi/4) matches Y_RIGHT.

    Returns (x, y, slope, iterations).  Hand-written secant: no root finder
    is imported.
    """
    x, y = integrate(s0, n)
    f0 = y[-1] - Y_RIGHT
    x, y = integrate(s1, n)
    f1 = y[-1] - Y_RIGHT

    for it in range(1, max_iter + 1):
        if abs(f1) < tol:
            return x, y, s1, it
        if f1 == f0:
            break
        s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
        s0, f0 = s1, f1
        s1 = s2
        x, y = integrate(s1, n)
        f1 = y[-1] - Y_RIGHT

    return x, y, s1, max_iter


def exact(x):
    """y(x) = -2 cos(2x) + 10 sin(2x)"""
    return -2.0 * np.cos(2.0 * x) + 10.0 * np.sin(2.0 * x)


def least_squares_slope(x, y):
    """Slope of the best straight line through (x, y), computed by hand."""
    xm, ym = np.mean(x), np.mean(y)
    return np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)


def main():
    # the analytic initial slope, for reference: y'(0) = 20
    print("Shooting for y'' + 4y = 0 on [0, pi/4],  y(0) = %.1f, y(pi/4) = %.1f"
          % (Y_LEFT, Y_RIGHT))
    print("exact initial slope y'(0) = 20\n")

    n_list = [25, 50, 100, 200, 400, 800]
    dx_list = np.zeros(len(n_list))
    err_list = np.zeros(len(n_list))

    print("%-8s %-14s %-16s %-14s %-8s"
          % ("N", "dx", "rms error", "slope found", "iters"))
    for j, n in enumerate(n_list):
        x, y, slope, iters = shoot(n)
        err = np.sqrt(np.mean((y - exact(x)) ** 2))
        dx_list[j] = x[1] - x[0]
        err_list[j] = err
        print("%-8d %-14.6f %-16.4e %-14.8f %-8d"
              % (n, dx_list[j], err, slope, iters))

    order = least_squares_slope(np.log(dx_list), np.log(err_list))
    print("\nfitted order = %.3f  (RK4: 4 expected)" % order)

    # --- plots -----------------------------------------------------------
    x, y, slope, _ = shoot(100)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(x, y, lw=1.6, label="numerical (shooting + RK4)")
    ax[0].plot(x, exact(x), "*", ms=4, label="exact")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y(x)")
    ax[0].set_title(r"$y''+4y=0$,  $\Delta x$ = %.5f" % (x[1] - x[0]))
    ax[0].legend()

    ax[1].loglog(dx_list, err_list, "o-", label="measured")
    ax[1].loglog(dx_list, err_list[-1] * (dx_list / dx_list[-1]) ** 4, "--",
                 label="slope 4")
    ax[1].set_xlabel(r"$\Delta x$")
    ax[1].set_ylabel("rms error")
    ax[1].set_title("Error vs step size, fitted order %.2f" % order)
    ax[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
