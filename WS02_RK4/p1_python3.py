"""
Worksheet 2, Problem 1  --  Python 3 corrected version of p1.py

Integrate the damped oscillator

        d2(theta)/dt2 + 2 lambda w0 d(theta)/dt + w0^2 theta = 0

with RK4, vary lambda and numerically determine the critical damping value
lambda_c.

Changes made relative to the original p1.py
-------------------------------------------
1. The RK4 combination was written
        x1[i+1] = x1[i] + dt*(k11 + 2*k21*+2*k31 + k41)/6
   The "*+" makes  2*k21 * (+2*k31)  -- a PRODUCT of the two middle stages
   instead of their sum.  The integrator was therefore not RK4 and had no
   defined order of accuracy.  Fixed to  k11 + 2*k21 + 2*k31 + k41.
2. "print 'lambda critical=', ..." was Python 2 syntax and stopped the file
   from parsing at all under Python 3.
3. The lambda sweep tested c1[i-1], which wraps to the last element of the
   array when i == 0 and would have reported a meaningless value.  The search
   is now a coarse scan followed by bisection, so lambda_c is resolved to
   1e-6 instead of to the 0.06 spacing of the original grid.
4. sys.exit() in the middle of a loop is replaced by an ordinary return.
5. The result is compared against the analytic answer, lambda_c = 1.

Algorithms are hand written: no ODE solver or root finder is imported.

Criterion used
--------------
Underdamped motion (lambda < 1) oscillates and therefore crosses zero.
Critically and overdamped motion (lambda >= 1) released from rest never
crosses zero.  lambda_c is the boundary between the two behaviours.
"""

import numpy as np
import matplotlib.pyplot as plt


def f_pos(v):
    """d(theta)/dt = v"""
    return v


def f_vel(x, v, lam, w0):
    """d(v)/dt = -2 lambda w0 v - w0^2 x"""
    return -(2.0 * lam * w0 * v + x * w0 ** 2)


def rk4(lam, w0, t_end=100.0, n_steps=10000, x0=0.1, v0=0.0):
    """Classic fourth-order Runge-Kutta on the (theta, v) pair."""
    t = np.linspace(0.0, t_end, n_steps)
    dt = t[1] - t[0]
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0], v[0] = x0, v0

    for i in range(n_steps - 1):
        k1x = f_pos(v[i])
        k1v = f_vel(x[i], v[i], lam, w0)

        k2x = f_pos(v[i] + 0.5 * dt * k1v)
        k2v = f_vel(x[i] + 0.5 * dt * k1x, v[i] + 0.5 * dt * k1v, lam, w0)

        k3x = f_pos(v[i] + 0.5 * dt * k2v)
        k3v = f_vel(x[i] + 0.5 * dt * k2x, v[i] + 0.5 * dt * k2v, lam, w0)

        k4x = f_pos(v[i] + dt * k3v)
        k4v = f_vel(x[i] + dt * k3x, v[i] + dt * k3v, lam, w0)

        x[i + 1] = x[i] + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        v[i + 1] = v[i] + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0

    return t, x, v


def oscillates(lam, w0, t_end=400.0, n_steps=40000):
    """
    True if the trajectory crosses zero, i.e. the motion is underdamped.

    The first zero crossing of an underdamped oscillator released from rest
    occurs at  t = pi / (w0 sqrt(1 - lambda^2)),  which diverges as lambda
    approaches 1.  A short observation window therefore reports "no crossing"
    slightly below the true lambda_c.  t_end = 400 keeps that bias near 1e-4;
    lengthening it further shrinks the bias but costs proportionally more
    RK4 steps.
    """
    _, x, _ = rk4(lam, w0, t_end=t_end, n_steps=n_steps)
    return bool(np.any(x < 0.0))


def find_lambda_c(w0, lo=0.1, hi=2.0, tol=1e-6):
    """
    Bracket lambda_c between an oscillating and a non-oscillating lambda,
    then bisect.  Hand-written bisection -- no root finder imported.
    """
    if not oscillates(lo, w0):
        raise RuntimeError("lower bracket is already non-oscillatory")
    if oscillates(hi, w0):
        raise RuntimeError("upper bracket still oscillates")

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if oscillates(mid, w0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    w0 = 0.5

    lam_c = find_lambda_c(w0)
    print("lambda critical = %.6f   (analytic value 1.0, error %.2e)"
          % (lam_c, abs(lam_c - 1.0)))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    for lam in [0.1, 0.3, 0.6, 1.0, 1.6]:
        t, x, _ = rk4(lam, w0, t_end=60.0, n_steps=6000)
        ax[0].plot(t, x, label=r"$\lambda$ = %.1f" % lam)
    ax[0].axhline(0.0, lw=0.8, color="grey")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel(r"$\theta(t)$")
    ax[0].set_title(r"Damped oscillator, $\omega_0$ = %.1f" % w0)
    ax[0].legend(fontsize=8)

    # how far below zero the trajectory dips, as a function of lambda
    lams = np.linspace(0.1, 1.5, 60)
    dips = np.zeros(len(lams))
    for j, lam in enumerate(lams):
        _, x, _ = rk4(lam, w0, t_end=60.0, n_steps=6000)
        dips[j] = -np.min(x)
    ax[1].plot(lams, dips, "o-", ms=3)
    ax[1].axvline(lam_c, ls="--", color="red",
                  label=r"$\lambda_c$ = %.4f" % lam_c)
    ax[1].axhline(0.0, lw=0.8, color="grey")
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$-\min_t\,\theta(t)$  (overshoot)")
    ax[1].set_title("Overshoot vanishes at critical damping")
    ax[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
