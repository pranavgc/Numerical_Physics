"""
Worksheet 2, Problem 2  --  Python 3 corrected version of p2.py

Damped and driven oscillator

    d2(theta)/dt2 + 2 lambda w0 d(theta)/dt + w0^2 theta = (F0/m) sin(w t)

Integrate, plot the steady-state amplitude against the relative frequency
w/w0 for different damping, and numerically determine the resonant frequency
    w_r = w0 sqrt(1 - 2 lambda^2).

The original p2.py was already correct; this version is a tidy-up.

Changes made relative to the original p2.py
-------------------------------------------
1. f1 and f2 were redefined on every pass of a 600-iteration nested loop.
   They are now module-level functions taking their parameters as arguments,
   so nothing closes over a loop variable.
2. n = np.where(A == np.amax(A)) sat inside the inner loop, where only its
   last value was ever used.  The argmax is now taken once, after the sweep.
3. The measured resonance is compared against the analytic
   w_r = w0 sqrt(1 - 2 lambda^2), and the case 2 lambda^2 >= 1 (no resonance
   peak at all) is handled instead of silently reporting the grid edge.
4. A finer frequency grid and parabolic interpolation through the three
   points around the peak, so the resonance is resolved below the grid
   spacing instead of being quantised to it.
5. plt.legend() is called once at the end rather than once per curve.

Algorithms are hand written: no ODE solver is imported.
"""

import numpy as np
import matplotlib.pyplot as plt


def f_pos(v):
    """d(theta)/dt = v"""
    return v


def f_vel(x, v, t, lam, w0, w_drive, F):
    """d(v)/dt = -2 lambda w0 v - w0^2 x + F sin(w t)"""
    return -2.0 * lam * w0 * v - w0 ** 2 * x + F * np.sin(w_drive * t)


def rk4_driven(lam, w0, w_drive, F, t_end=300.0, n_steps=3000,
               x0=0.0, v0=4.0):
    """Classic RK4 with correct half-step stage times."""
    t = np.linspace(0.0, t_end, n_steps)
    dt = t[1] - t[0]
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0], v[0] = x0, v0

    for i in range(n_steps - 1):
        ti = t[i]

        k1x = f_pos(v[i])
        k1v = f_vel(x[i], v[i], ti, lam, w0, w_drive, F)

        k2x = f_pos(v[i] + 0.5 * dt * k1v)
        k2v = f_vel(x[i] + 0.5 * dt * k1x, v[i] + 0.5 * dt * k1v,
                    ti + 0.5 * dt, lam, w0, w_drive, F)

        k3x = f_pos(v[i] + 0.5 * dt * k2v)
        k3v = f_vel(x[i] + 0.5 * dt * k2x, v[i] + 0.5 * dt * k2v,
                    ti + 0.5 * dt, lam, w0, w_drive, F)

        k4x = f_pos(v[i] + dt * k3v)
        k4v = f_vel(x[i] + dt * k3x, v[i] + dt * k3v,
                    ti + dt, lam, w0, w_drive, F)

        x[i + 1] = x[i] + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        v[i + 1] = v[i] + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0

    return t, x, v


def steady_amplitude(lam, w0, w_drive, F, transient_fraction=0.5):
    """
    Peak displacement after the transient has decayed.  The first half of the
    trajectory is discarded, which is what "steady state" means here.
    """
    _, x, _ = rk4_driven(lam, w0, w_drive, F)
    start = int(len(x) * transient_fraction)
    return np.max(np.abs(x[start:]))


def parabolic_peak(xs, ys, i):
    """
    Refine the location of a maximum by fitting a parabola through the three
    samples around index i.  Hand-written; no curve-fitting routine used.
    """
    if i == 0 or i == len(xs) - 1:
        return xs[i]
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return xs[i]
    # vertex offset in units of the (uniform) grid spacing
    delta = 0.5 * (y0 - y2) / denom
    return xs[i] + delta * (xs[i + 1] - xs[i])


def main():
    w0 = 3.0
    F = 10.0
    w_grid = np.linspace(0.5, 10.0, 120)
    lambdas = np.arange(0.1, 0.7, 0.1)

    fig, ax = plt.subplots(figsize=(8, 5))

    print("%-10s %-14s %-14s %-10s" %
          ("lambda", "w_r measured", "w_r analytic", "rel. error"))

    for lam in lambdas:
        A = np.zeros(len(w_grid))
        for j, w_drive in enumerate(w_grid):
            A[j] = steady_amplitude(lam, w0, w_drive, F)

        peak = int(np.argmax(A))
        w_r = parabolic_peak(w_grid, A, peak)

        if 1.0 - 2.0 * lam ** 2 > 0.0:
            w_r_exact = w0 * np.sqrt(1.0 - 2.0 * lam ** 2)
            rel = abs(w_r - w_r_exact) / w_r_exact
            print("%-10.2f %-14.4f %-14.4f %-10.2e"
                  % (lam, w_r, w_r_exact, rel))
        else:
            print("%-10.2f %-14.4f %-14s %-10s"
                  % (lam, w_r, "none", "-- overdamped, no peak"))

        ax.plot(w_grid / w0, A, label=r"$\lambda$ = %.1f" % lam)

    ax.set_xlabel(r"$\omega/\omega_0$")
    ax.set_ylabel(r"$A(\omega)$")
    ax.set_title("Steady-state amplitude of the driven damped oscillator")
    ax.grid(True, lw=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
