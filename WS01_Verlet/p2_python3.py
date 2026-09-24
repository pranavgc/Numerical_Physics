"""
Worksheet 1, Problem 2  --  Python 3 corrected version of p2.py

Integrate the pendulum

        d2(theta)/dt2 = -k sin(theta)

with the Verlet algorithm and show that the error in the velocity is O(dt^2).

Changes made relative to the original p2.py
-------------------------------------------
1. The original integrated  -k*theta  (the linear oscillator), not
   -k*sin(theta).  This version integrates the pendulum that was actually
   set.  The linear case is kept as a separate check because it has a closed
   form to validate the error machinery against.
2. The RMS error was computed as  (sum(v - z)**2 / N)**0.5  -- the residuals
   were summed and *then* squared, so positive and negative errors cancelled
   and the result was not an error norm at all.  Corrected to
   sqrt(mean((v - z)**2)).
3. The dt plotted on the x axis came from np.linspace(0, 2, j) while the
   integration used np.linspace(0, 20, u), so the axis was wrong by a factor
   of ten.  There is now a single dt, returned by the integrator.
4. y[0] = x[1]/2*dt evaluated as (x[1]/2)*dt instead of x[1]/(2*dt).
   The endpoint velocities are now handled with one-sided differences.
5. The convergence order is fitted and printed instead of being left for the
   reader to eyeball off a plot.

The nonlinear pendulum has no elementary closed-form solution, so the
reference velocity is a Verlet run on a much finer grid.  The fine grid step
divides every coarse step exactly, so the reference is sampled by striding --
no interpolation is needed and no interpolation error is introduced.

Algorithms are hand written: no ODE solver is imported.

Position Verlet:   x_{n+1} = 2 x_n - x_{n-1} + a(x_n) dt^2
Velocity by central difference:   v_n = (x_{n+1} - x_{n-1}) / (2 dt)
"""

import numpy as np
import matplotlib.pyplot as plt


def accel(x, k, nonlinear=True):
    """Angular acceleration of the pendulum (or of the linear oscillator)."""
    if nonlinear:
        return -k * np.sin(x)
    return -k * x


def verlet(k, dt, n_steps, x0, v0, nonlinear=True):
    """
    Position-Verlet integration.  Returns (t, x, v).

    The first step is taken with the Taylor expansion
        x_1 = x_0 + v_0 dt + 0.5 a(x_0) dt^2
    which is accurate to the same order as the Verlet recursion itself.
    """
    t = np.arange(n_steps + 1) * dt
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)

    x[0] = x0
    x[1] = x0 + v0 * dt + 0.5 * accel(x0, k, nonlinear) * dt ** 2

    for i in range(1, n_steps):
        x[i + 1] = 2.0 * x[i] - x[i - 1] + accel(x[i], k, nonlinear) * dt ** 2

    # central differences in the interior, one-sided at the two ends
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return t, x, v


def rms(a, b):
    """Root-mean-square difference of two equally long arrays."""
    return np.sqrt(np.mean((a - b) ** 2))


def convergence_study(k, x0, v0, t_end, nonlinear, ref_refine=64):
    """
    Integrate at a ladder of step sizes and measure the velocity error.

    For the nonlinear pendulum the reference is a Verlet run whose step is
    ref_refine times smaller than the finest step under test.  Every coarse
    grid is a subset of the fine grid, so the comparison is exact.
    For the linear oscillator the analytic velocity is used instead.
    """
    # step counts chosen so each divides the next: strides stay integers
    n_list = np.array([250, 500, 1000, 2000, 4000, 8000])
    dt_list = t_end / n_list

    n_ref = n_list[-1] * ref_refine
    dt_ref = t_end / n_ref
    if nonlinear:
        _, _, v_ref = verlet(k, dt_ref, n_ref, x0, v0, nonlinear=True)

    errors = np.zeros(len(n_list))
    for j, n in enumerate(n_list):
        t, x, v = verlet(k, dt_list[j], n, x0, v0, nonlinear)
        if nonlinear:
            stride = n_ref // n
            v_exact = v_ref[::stride]
        else:
            w = np.sqrt(k)
            v_exact = -x0 * w * np.sin(w * t) + v0 * np.cos(w * t)
        # drop the two endpoints: they use one-sided differences, which are
        # only first-order accurate and would mask the interior O(dt^2)
        errors[j] = rms(v[1:-1], v_exact[1:-1])

    return dt_list, errors


def fit_order(dt, err):
    """Least-squares slope of log(err) against log(dt), done by hand."""
    X = np.log(dt)
    Y = np.log(err)
    Xm, Ym = X.mean(), Y.mean()
    return np.sum((X - Xm) * (Y - Ym)) / np.sum((X - Xm) ** 2)


def main():
    k = 0.34
    x0, v0 = 1.0, 0.0      # released from rest at 1 rad -- genuinely nonlinear
    t_end = 20.0

    dt_nl, err_nl = convergence_study(k, x0, v0, t_end, nonlinear=True)
    dt_li, err_li = convergence_study(k, x0, v0, t_end, nonlinear=False)

    print("Pendulum   d2(theta)/dt2 = -k sin(theta),  k = %.2f" % k)
    print("  %-12s %-14s" % ("dt", "rms velocity error"))
    for d, e in zip(dt_nl, err_nl):
        print("  %-12.6f %-14.3e" % (d, e))
    print("  fitted order = %.3f  (expected 2)" % fit_order(dt_nl, err_nl))
    print()
    print("Linear check  d2(theta)/dt2 = -k theta  (analytic reference)")
    print("  fitted order = %.3f  (expected 2)" % fit_order(dt_li, err_li))

    # --- one trajectory, to show the pendulum is actually being solved ----
    t, x, v = verlet(k, t_end / 4000, 4000, x0, v0, nonlinear=True)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].plot(t, x, label=r"$\theta(t)$")
    ax[0].plot(t, v, label=r"$\dot\theta(t)$")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Angle / angular velocity")
    ax[0].set_title(r"Verlet, $\ddot\theta=-k\sin\theta$")
    ax[0].legend()

    ax[1].plot(dt_nl, err_nl, "o-", label="pendulum")
    ax[1].plot(dt_li, err_li, "s--", label="linear")
    ax[1].set_xlabel(r"$\Delta t$")
    ax[1].set_ylabel("rms velocity error")
    ax[1].set_title(r"Error vs $\Delta t$")
    ax[1].legend()

    ax[2].plot(dt_nl ** 2, err_nl, "o-", label="pendulum")
    ax[2].plot(dt_li ** 2, err_li, "s--", label="linear")
    ax[2].set_xlabel(r"$\Delta t^2$")
    ax[2].set_ylabel("rms velocity error")
    ax[2].set_title(r"Error vs $\Delta t^2$ -- straight line $\Rightarrow$ $O(\Delta t^2)$")
    ax[2].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
