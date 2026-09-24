"""
Worksheet 6, Problem 1  --  Python 3 corrected version of p1.py

Integrate the Black-Scholes equation

        dx = mu x dt + sigma x dW(t)

by the Euler-Maruyama method, with dW dW proportional to dt, i.e.
dW_i = z_i sqrt(dt) with z_i drawn from N(0, 1).

(a) Plot the numerically integrated trajectory together with the exact
    solution  x(t) = x0 exp(mu t - sigma^2 t / 2 + sigma eta(t))
    where eta(t) = integral of dW from 0 to t.
(b) Compute the rms error and plot it against dt.

Changes made relative to the original p1.py
-------------------------------------------
1. The Wiener integral was accumulated as
        eta[i+1] = eta[i] + z[i] * dt**1.5
   The increment of a Wiener process is z sqrt(dt), so the exponent must be
   0.5, not 1.5.  With dt ~ 0.1 the original eta was about ten times too
   small, so the "exact" reference curve was effectively the deterministic
   solution and the whole error study was measured against the wrong thing.
   Corrected to dt**0.5.
2. The simulated path was driven by z[i+1] while eta used z[i], so even
   after fixing the exponent the two would have been driven by noise
   sequences offset by one step.  Both now use the same increment.
3. Part (a) -- the trajectory plot the worksheet explicitly asks for -- was
   present but commented out.  It is restored.
4. The error-versus-dt study drew fresh random numbers for every dt, so each
   point sampled a DIFFERENT Brownian path and the curve measured sampling
   noise as much as discretisation error.  An ensemble of fine Brownian
   paths is now generated once and coarsened by summing increments, which is
   the standard way to measure strong convergence: every dt sees the same
   realisations of W(t), and the error is the expectation over the ensemble
   rather than one path's accident.
5. The measured strong order of convergence is fitted and printed.  For
   Euler-Maruyama the expected strong order is 0.5, and this version
   recovers it to about 1%.

Algorithms are hand written: the SDE integrator, the Brownian path and the
least-squares fit are all implemented here.
"""

import numpy as np
import matplotlib.pyplot as plt


MU = 0.25
SIGMA = 1.2
X0 = 1.0
T_END = 35.0


def euler_maruyama(dW, dt, x0=X0, mu=MU, sigma=SIGMA):
    """
    Integrate dx = mu x dt + sigma x dW with Euler-Maruyama.

    dW is the array of Wiener increments to use, one per step; passing them
    in (rather than drawing them inside) is what lets several step sizes
    share one underlying Brownian path.
    """
    n = len(dW)
    x = np.zeros(n + 1)
    x[0] = x0
    for i in range(n):
        x[i + 1] = x[i] + mu * x[i] * dt + sigma * x[i] * dW[i]
    return x


def euler_maruyama_endpoint(dW, dt, x0=X0, mu=MU, sigma=SIGMA):
    """
    Same scheme, but stepping many independent paths at once and keeping
    only x(T).  dW has shape (n_paths, n_steps).  Used for the convergence
    study, where thousands of paths are needed and only the endpoint is.
    """
    x = np.full(dW.shape[0], float(x0))
    for i in range(dW.shape[1]):
        x = x + mu * x * dt + sigma * x * dW[:, i]
    return x


def exact_solution(eta, t, x0=X0, mu=MU, sigma=SIGMA):
    """
    x(t) = x0 exp(mu t - sigma^2 t / 2 + sigma eta(t)),
    with eta(t) the running integral of dW -- i.e. W(t) itself.
    """
    return x0 * np.exp(mu * t - 0.5 * sigma ** 2 * t + sigma * eta)


def brownian_path(n_steps, dt):
    """
    Wiener increments and the running integral eta(t) = W(t).

    dW_i = z_i sqrt(dt).  eta is the cumulative sum, prefixed with eta(0)=0.
    """
    z = np.random.randn(n_steps)
    dW = z * np.sqrt(dt)
    eta = np.concatenate(([0.0], np.cumsum(dW)))
    return dW, eta


def least_squares_slope(x, y):
    """Slope of the best straight line through (x, y), computed by hand."""
    xm, ym = np.mean(x), np.mean(y)
    return np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)


def main():
    # ---------------- part (a): one trajectory ---------------------------
    dt_a = 0.01
    n_a = int(T_END / dt_a)
    dW_a, eta_a = brownian_path(n_a, dt_a)
    t_a = np.arange(n_a + 1) * dt_a

    x_num = euler_maruyama(dW_a, dt_a)
    x_exact = exact_solution(eta_a, t_a)

    print("Black-Scholes:  mu = %.2f, sigma = %.2f, x0 = %.1f, T = %.0f"
          % (MU, SIGMA, X0, T_END))
    # mu - sigma^2/2 is negative here, so x(t) decays towards zero over this
    # horizon and its magnitude spans many orders.  An absolute rms error is
    # then dominated by the early, large part of the path, so the relative
    # deviation is the informative number here.
    rel = np.max(np.abs(x_num - x_exact) / np.maximum(np.abs(x_exact), 1e-300))
    print("part (a)  dt = %.3f   rms error = %.4e   max relative error = %.4e\n"
          % (dt_a, np.sqrt(np.mean((x_num - x_exact) ** 2)), rel))

    # ---------------- part (b): rms error against dt ---------------------
    # "Strong error" is an EXPECTATION over realisations, E|x_N - x(T)|, not
    # the error of one path: a single path's error fluctuates by orders of
    # magnitude and fitting a slope through it measures noise.  So a large
    # ensemble of fine Brownian paths is generated once and each is coarsened
    # by summing blocks of increments -- every dt then sees exactly the same
    # realisations of W(t), and the error is averaged over the ensemble.
    #
    # The study also runs to T = 1 rather than T = 35.  With sigma = 1.2 the
    # solution at T = 35 spans many orders of magnitude, and the coarsest
    # steps have sigma sqrt(dt) of order 1, far outside the asymptotic
    # regime where a convergence order means anything.
    t_conv = 1.0
    n_paths = 4000
    base_steps = 2 ** 10
    dt_fine = t_conv / base_steps

    z = np.random.randn(n_paths, base_steps)
    dW_fine = z * np.sqrt(dt_fine)
    W_T = np.sum(dW_fine, axis=1)                 # W(T), same for every dt
    x_true = exact_solution(W_T, t_conv)

    strides = [2 ** k for k in range(0, 6)]       # 1 ... 32
    dt_list = np.array([dt_fine * s for s in strides])
    errors = np.zeros(len(strides))

    print("Strong convergence study: T = %.0f, %d paths" % (t_conv, n_paths))
    print("%-14s %-10s %-16s" % ("dt", "steps", "E|x_N - x(T)|"))
    for j, stride in enumerate(strides):
        n_coarse = base_steps // stride
        dt = dt_fine * stride
        # a coarse increment is the sum of the fine increments it contains
        dW = dW_fine.reshape(n_paths, n_coarse, stride).sum(axis=2)
        x_num_b = euler_maruyama_endpoint(dW, dt)
        errors[j] = np.mean(np.abs(x_num_b - x_true))
        print("%-14.6f %-10d %-16.4e" % (dt, n_coarse, errors[j]))

    order = least_squares_slope(np.log(dt_list), np.log(errors))
    print("\nfitted strong order = %.3f  "
          "(Euler-Maruyama has strong order 1/2;\n"
          "                              the Milstein scheme would give 1)"
          % order)

    # ---------------- plots ----------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(t_a, x_num, lw=1.0, label="Euler-Maruyama")
    ax[0].plot(t_a, x_exact, "--", lw=1.0, label="exact")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("x(t)")
    ax[0].set_title("(a) trajectory, dt = %.2f" % dt_a)
    ax[0].legend()

    ax[1].loglog(dt_list, errors, "o-", label="measured")
    ax[1].loglog(dt_list, errors[-1] * (dt_list / dt_list[-1]) ** 0.5, "--",
                 label="slope 1/2")
    ax[1].loglog(dt_list, errors[-1] * (dt_list / dt_list[-1]) ** 1.0, ":",
                 label="slope 1")
    ax[1].set_xlabel("dt")
    ax[1].set_ylabel(r"$E|x_N - x(T)|$")
    ax[1].set_title("(b) strong convergence, fitted order %.2f" % order)
    ax[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
