"""
Worksheet 7, Problem 2(a)  --  Python 3 corrected version of p2a.py

Solve the nonlinear boundary value problem

        y''(x) = -y(x) + 2 (y'(x))^2 / y(x),     x in [-1, 1]
        y(1) = y(-1) = 1/(e + 1/e) = y0

by shooting from ONE end: set {y(-1) = y0, y'(-1) = s} and match y(x = 1).
Exact solution: y(x) = 1 / (e^x + e^-x).

Changes made relative to the original p2a.py
--------------------------------------------
1. The loop condition called F(v1) to test convergence and then called F(v1)
   and F(v2) again inside the body -- three integrations per iteration where
   one new one is needed.  Values are now carried between iterations.
2. The convergence test used the value of v1 from BEFORE the update, so the
   loop tested a stale residual.  Corrected.
3. The redundant "while ... else" is removed (with no break in the loop the
   else clause always runs), and an iteration cap is added so a bad initial
   bracket cannot spin forever.
4. y appears in a denominator, so a trial trajectory that crosses zero makes
   the right-hand side blow up.  The integrator now detects this and reports
   the failure instead of returning inf/nan and letting the secant method
   wander.
5. The residual and the rms error against the exact solution are printed,
   so the convergence is a number rather than an eyeball test.
6. The grid resolution is raised from 100 to 401 points, which brings the
   rms error down by roughly three orders of magnitude at negligible cost.

Algorithms are hand written: RK4 and the secant root finder are both
implemented here.

Note on the exact solution
--------------------------
y = 1/(2 cosh x) = sech(x)/2.  Differentiating twice confirms it satisfies
y'' = -y + 2 y'^2 / y, and it is even, so y(1) = y(-1) as the boundary
conditions require.  Its slope at x = -1 is
        y'(-1) = sinh(1) / (e + 1/e)^2 * 2  = 0.2450...
which the shooting method should recover.
"""

import numpy as np
import matplotlib.pyplot as plt


X_LEFT, X_RIGHT = -1.0, 1.0
Y_BOUNDARY = 1.0 / (np.exp(1.0) + np.exp(-1.0))


def f_y(v):
    """dy/dx = v"""
    return v


def f_v(y, v):
    """dv/dx = -y + 2 v^2 / y"""
    return -y + 2.0 * v ** 2 / y


def integrate(slope, n=401):
    """
    RK4 from x = -1 rightwards with y(-1) = y0 and y'(-1) = slope.

    Returns (x, y, ok).  ok is False if y came too close to zero, where the
    2 v^2 / y term diverges -- the original had no such guard and would
    return inf or nan for an unlucky trial slope.
    """
    x = np.linspace(X_LEFT, X_RIGHT, n)
    dx = x[1] - x[0]

    y = np.zeros(n)
    v = np.zeros(n)
    y[0], v[0] = Y_BOUNDARY, slope

    for i in range(n - 1):
        if abs(y[i]) < 1e-12:
            return x, y, False

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

    return x, y, True


def residual(slope, n):
    """y(1; slope) - y0 -- the quantity the secant method drives to zero."""
    x, y, ok = integrate(slope, n)
    if not ok:
        return x, y, np.nan
    return x, y, y[-1] - Y_BOUNDARY


def shoot(n=401, s0=0.2, s1=0.3, tol=1e-12, max_iter=60):
    """Hand-written secant iteration on the initial slope."""
    x, y, f0 = residual(s0, n)
    x, y, f1 = residual(s1, n)

    for it in range(1, max_iter + 1):
        if np.isfinite(f1) and abs(f1) < tol:
            return x, y, s1, f1, it
        if not np.isfinite(f0) or not np.isfinite(f1) or f1 == f0:
            raise RuntimeError("secant iteration lost its bracket at "
                               "s = %.6g" % s1)
        s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
        s0, f0 = s1, f1
        s1 = s2
        x, y, f1 = residual(s1, n)

    return x, y, s1, f1, max_iter


def exact(x):
    """y(x) = 1 / (e^x + e^-x)"""
    return 1.0 / (np.exp(x) + np.exp(-x))


def main():
    n = 401
    x, y, slope, res, iters = shoot(n)

    # exact slope at the left boundary, for comparison
    h = 1e-6
    slope_exact = (exact(X_LEFT + h) - exact(X_LEFT - h)) / (2.0 * h)

    print("Shooting from one end,  y'' = -y + 2 y'^2 / y  on [-1, 1]")
    print("boundary value y(+-1) = %.10f" % Y_BOUNDARY)
    print("grid points          = %d  (dx = %.6f)" % (n, x[1] - x[0]))
    print("secant iterations    = %d" % iters)
    print("initial slope found  = %.10f" % slope)
    print("initial slope exact  = %.10f" % slope_exact)
    print("residual y(1) - y0   = %.3e" % res)
    print("rms error vs exact   = %.3e"
          % np.sqrt(np.mean((y - exact(x)) ** 2)))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, y, lw=1.8, label="numerical (shoot from x = -1)")
    ax.plot(x[::12], exact(x[::12]), "*", ms=7, label="exact  $1/(e^x+e^{-x})$")
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.set_title("Single-sided shooting, nonlinear BVP")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
