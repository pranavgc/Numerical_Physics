"""
Worksheet 7, Problem 2(b)  --  Python 3 corrected version of p2b.py

Solve the nonlinear boundary value problem

        y''(x) = -y(x) + 2 (y'(x))^2 / y(x),     x in [-1, 1]
        y(1) = y(-1) = 1/(e + 1/e) = y0

by shooting from BOTH ends and matching at x = 0.
Exact solution: y(x) = 1 / (e^x + e^-x).

Changes made relative to the original p2b.py
--------------------------------------------
1. The function packed two scalars and two arrays into one
   np.array([y[-4], y2[-4], y, y2]).  NumPy allowed this as an object array
   before 1.24 and raises ValueError now, so the script crashes outright on
   any current NumPy.  Results are returned as a tuple instead.
2. The matching was degenerate.  The left branch was launched with slope +c
   and the right branch with slope -c -- a single unknown -- and the ODE is
   invariant under x -> -x, so the two branches were exact mirror images for
   EVERY c and their values at the midpoint agreed automatically.  The
   condition being "solved" was satisfied identically, so the iteration was
   not determining anything.
   This version treats the two slopes as INDEPENDENT unknowns (sL, sR) and
   matches BOTH the value and the derivative at x = 0 -- two equations in
   two unknowns, which is what double-sided shooting means.  It is solved
   with a hand-written 2x2 Newton iteration using a finite-difference
   Jacobian.
3. The match was taken at index -4 rather than at the midpoint, for no
   stated reason.  Both branches now stop exactly at x = 0.
4. Only the value was matched, never the derivative, so the reconstructed
   solution could have had a kink at the join.  Both are matched now, and
   the residual jump in y and y' at x = 0 is printed.
5. y sits in a denominator, so a trial trajectory crossing zero makes the
   right-hand side diverge.  The integrator now detects that instead of
   silently returning inf or nan.

Algorithms are hand written: RK4, the 2x2 linear solve and the Newton
iteration are all implemented here.

Why match two conditions
------------------------
Each half of the interval is an initial value problem with one free
parameter (the slope at its own outer boundary).  Two free parameters need
two matching conditions, and continuity of the solution at the join requires
both  y_L(0) = y_R(0)  and  y_L'(0) = y_R'(0).  Matching only the value
would leave the derivative free and admit a kinked "solution".
"""

import numpy as np
import matplotlib.pyplot as plt


X_LEFT, X_RIGHT, X_MATCH = -1.0, 1.0, 0.0
Y_BOUNDARY = 1.0 / (np.exp(1.0) + np.exp(-1.0))
N_HALF = 201          # grid points on each half of the interval


def f_y(v):
    """dy/dx = v"""
    return v


def f_v(y, v):
    """dv/dx = -y + 2 v^2 / y"""
    return -y + 2.0 * v ** 2 / y


def integrate(x_start, x_end, slope, n=N_HALF):
    """
    RK4 from x_start to x_end with y(x_start) = y0 and y'(x_start) = slope.

    dx is signed, so this integrates rightwards when x_end > x_start and
    leftwards when x_end < x_start -- which is exactly what the right-hand
    branch needs.

    Returns (x, y, v, ok); ok is False if y came too close to zero.
    """
    x = np.linspace(x_start, x_end, n)
    dx = x[1] - x[0]

    y = np.zeros(n)
    v = np.zeros(n)
    y[0], v[0] = Y_BOUNDARY, slope

    for i in range(n - 1):
        if abs(y[i]) < 1e-12:
            return x, y, v, False

        k1y = f_y(v[i])
        k1v = f_v(y[i], v[i])

        k2y = f_y(v[i] + 0.5 * dx * k1v)
        k2v = f_v(y[i] + 0.5 * dx * k1y, v[i] + 0.5 * dx * k1v)

        k3y = f_y(v[i] + 0.5 * dx * k2v)
        k3v = f_v(y[i] + 0.5 * dx * k2y, v[i] + 0.5 * dx * k2v)

        k4y = f_y(v[i] + dx * k3v)
        k4v = f_v(y[i] + dx * k3y, v[i] + dx * k3v)

        y[i + 1] = y[i] + dx * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        v[i + 1] = v[i] + dx * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0

    return x, y, v, True


def branches(sL, sR):
    """Integrate both halves inward to x = 0 with the given outer slopes."""
    xL, yL, vL, okL = integrate(X_LEFT, X_MATCH, sL)
    xR, yR, vR, okR = integrate(X_RIGHT, X_MATCH, sR)
    return (xL, yL, vL), (xR, yR, vR), (okL and okR)


def mismatch(sL, sR):
    """
    The two matching conditions at x = 0, as a length-2 vector:
        [ y_L(0) - y_R(0),  y_L'(0) - y_R'(0) ]
    Driving both to zero is what double-sided shooting has to achieve.
    """
    (xL, yL, vL), (xR, yR, vR), ok = branches(sL, sR)
    if not ok:
        return np.array([np.nan, np.nan])
    return np.array([yL[-1] - yR[-1], vL[-1] - vR[-1]])


def solve_2x2(a, b):
    """
    Solve the 2x2 system a @ z = b by Cramer's rule, written out by hand.
    No linear algebra routine is imported.
    """
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    if det == 0.0:
        raise RuntimeError("singular Jacobian in the Newton step")
    z0 = (b[0] * a[1, 1] - a[0, 1] * b[1]) / det
    z1 = (a[0, 0] * b[1] - b[0] * a[1, 0]) / det
    return np.array([z0, z1])


def newton_2d(sL, sR, tol=1e-12, max_iter=60, h=1e-7):
    """
    Two-dimensional Newton iteration on (sL, sR) with a finite-difference
    Jacobian.  Hand written; no root finder is imported.
    """
    for it in range(1, max_iter + 1):
        F = mismatch(sL, sR)
        if not np.all(np.isfinite(F)):
            raise RuntimeError("a trial trajectory diverged at "
                               "(sL, sR) = (%.6g, %.6g)" % (sL, sR))
        if np.max(np.abs(F)) < tol:
            return sL, sR, F, it

        # finite-difference Jacobian, column by column
        J = np.zeros((2, 2))
        J[:, 0] = (mismatch(sL + h, sR) - F) / h
        J[:, 1] = (mismatch(sL, sR + h) - F) / h

        step = solve_2x2(J, -F)
        sL += step[0]
        sR += step[1]

    return sL, sR, mismatch(sL, sR), max_iter


def exact(x):
    """y(x) = 1 / (e^x + e^-x)"""
    return 1.0 / (np.exp(x) + np.exp(-x))


def main():
    # start from slopes of opposite sign, which is what the symmetry of the
    # problem suggests -- but they are free to move independently from here
    sL, sR, F, iters = newton_2d(0.2, -0.2)

    (xL, yL, vL), (xR, yR, vR), _ = branches(sL, sR)

    # stitch the two halves: the right branch was integrated leftwards, so
    # it is reversed, and its first point duplicates the join
    x = np.concatenate((xL, xR[::-1][1:]))
    y = np.concatenate((yL, yR[::-1][1:]))

    h = 1e-6
    slope_exact_L = (exact(X_LEFT + h) - exact(X_LEFT - h)) / (2.0 * h)

    print("Shooting from both ends,  y'' = -y + 2 y'^2 / y  on [-1, 1]")
    print("boundary value y(+-1)  = %.10f" % Y_BOUNDARY)
    print("points per half        = %d  (dx = %.6f)" % (N_HALF, xL[1] - xL[0]))
    print("Newton iterations      = %d" % iters)
    print("left  slope y'(-1)     = %+.10f   (exact %+.10f)"
          % (sL, slope_exact_L))
    print("right slope y'(+1)     = %+.10f   (exact %+.10f)"
          % (sR, -slope_exact_L))
    print("mismatch at x = 0:  dy = %.3e   dy' = %.3e" % (F[0], F[1]))
    print("rms error vs exact     = %.3e"
          % np.sqrt(np.mean((y - exact(x)) ** 2)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(xL, yL, lw=1.8, label="left branch  (from x = -1)")
    ax[0].plot(xR, yR, lw=1.8, label="right branch (from x = +1)")
    ax[0].axvline(X_MATCH, ls="--", lw=0.8, color="grey", label="match point")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y(x)")
    ax[0].set_title("The two branches meeting at x = 0")
    ax[0].legend(fontsize=8)

    ax[1].plot(x, y, lw=1.8, label="numerical (stitched)")
    ax[1].plot(x[::20], exact(x[::20]), "*", ms=7,
               label="exact  $1/(e^x+e^{-x})$")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y(x)")
    ax[1].set_title("Double-sided shooting, nonlinear BVP")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
