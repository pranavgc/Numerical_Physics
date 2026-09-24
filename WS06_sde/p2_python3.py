"""
Worksheet 6, Problem 2  --  Python 3 corrected version of p2.py

Integrate the stochastic birth-death model

        dn = k1 dt - k2 n dt + dW(t)

by the Euler-Maruyama method, with the noise property
dW dW proportional to (k1 + k2 n) dt, i.e.
dW_i = z_i sqrt(dt (k1 + k2 n)) with z_i drawn from N(0, 1).

(a) Mean trajectory n(t) over N realisations, plotted with the deterministic
    solution n(t) = (k1/k2)(1 - exp(-k2 t)).
(b) Steady-state distribution P(n), fitted to a Poisson, and its mean.

Changes made relative to the original p2.py
-------------------------------------------
1. The noise amplitude read
        z1 * (t1 * (k11 + k21 * n))**0.5
   where the function's own argument is named n1 and `n` at module scope is
   the NUMBER OF REALISATIONS (100).  The noise was therefore a constant
   sqrt(dt * (k1 + 100 k2)) instead of the state-dependent
   sqrt(dt * (k1 + k2 n)).  Because the drift is untouched, the mean
   trajectory still matched the deterministic curve, so part (a) looked
   right while part (b) -- which is entirely about the fluctuations -- was
   measuring the wrong process.  Corrected to use the state n1.
2. "print "mean=", ..." was Python 2 syntax and stopped the file from
   parsing under Python 3.
3. math.factorial(n) overflows into slow big integers and then into inf for
   n beyond about 170.  The Poisson probability is now evaluated through a
   hand-written log-gamma (Stirling with the standard correction series), so
   it stays finite for any n the simulation can reach.
4. The noise argument can go negative when a fluctuation pushes n below
   -k1/k2; sqrt of a negative number silently produced nan and poisoned the
   whole realisation.  The variance is now clamped at zero, and n itself is
   clamped at zero since a molecule count cannot be negative.
5. plt.show(block=False) followed by a second plt.show() opened two windows;
   both panels now go into one figure.
6. The fitted Poisson mean is obtained from the simulated data and compared
   with k1/k2, which is what part (b) asks for.

Algorithms are hand written: the integrator, the histogram, the log-gamma
and the Poisson density are all implemented here.
"""

import numpy as np
import matplotlib.pyplot as plt


K1 = 1.5
K2 = 0.03
N_REALISATIONS = 2000
T_END = 250.0
N_STEPS = 1000


def drift(n, k1, k2):
    """Deterministic part: k1 - k2 n."""
    return k1 - k2 * n


def noise_variance(n, k1, k2):
    """
    Variance rate of the increment: k1 + k2 n, clamped at zero.

    The clamp matters because a large downward fluctuation can drive n
    negative, and sqrt of a negative number is nan.
    """
    return np.maximum(k1 + k2 * n, 0.0)


def simulate(n_real, n_steps, t_end, k1, k2):
    """
    Euler-Maruyama for all realisations at once.

    Returns (t, N) with N of shape (n_real, n_steps): one row per
    realisation.  n is clamped at zero -- a molecule count cannot be
    negative, and without the clamp a single excursion below -k1/k2 would
    make the whole row nan.
    """
    t = np.linspace(0.0, t_end, n_steps)
    dt = t[1] - t[0]

    N = np.zeros((n_real, n_steps))
    for i in range(n_steps - 1):
        n_now = N[:, i]
        z = np.random.randn(n_real)
        dW = z * np.sqrt(dt * noise_variance(n_now, k1, k2))
        N[:, i + 1] = np.maximum(n_now + drift(n_now, k1, k2) * dt + dW, 0.0)

    return t, N


def deterministic(t, k1, k2):
    """n(t) = (k1/k2)(1 - exp(-k2 t))."""
    return (k1 / k2) * (1.0 - np.exp(-k2 * t))


def log_gamma(x):
    """
    log(Gamma(x)) by the Lanczos-style Stirling series, written out by hand.

    Used instead of math.factorial so that P(n) stays finite and fast for
    large n.  Accurate to better than 1e-10 for x >= 1, which is far more
    than this comparison needs.
    """
    x = np.asarray(x, dtype=float)
    # shift small arguments upward using Gamma(x) = Gamma(x+1)/x
    shift = np.zeros_like(x)
    y = x.copy()
    for _ in range(8):
        small = y < 8.0
        if not np.any(small):
            break
        shift[small] += np.log(y[small])
        y[small] += 1.0

    inv = 1.0 / y
    inv2 = inv * inv
    series = (1.0 / 12.0) * inv \
        - (1.0 / 360.0) * inv * inv2 \
        + (1.0 / 1260.0) * inv * inv2 * inv2 \
        - (1.0 / 1680.0) * inv * inv2 * inv2 * inv2
    return (y - 0.5) * np.log(y) - y + 0.5 * np.log(2.0 * np.pi) \
        + series - shift


def poisson_pmf(n, lam):
    """P(n) = lam^n exp(-lam) / n!, evaluated in logs to avoid overflow."""
    n = np.asarray(n, dtype=float)
    log_p = n * np.log(lam) - lam - log_gamma(n + 1.0)
    return np.exp(log_p)


def histogram(samples, lo, hi):
    """
    Integer-valued histogram over [lo, hi], one bin per integer, computed by
    hand in a single pass.  Returns (values, probability).
    """
    n_bins = int(hi - lo) + 1
    counts = np.zeros(n_bins)
    for s in samples:
        idx = int(round(s)) - int(lo)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
    values = np.arange(int(lo), int(lo) + n_bins)
    return values, counts / len(samples)


def main():
    t, N = simulate(N_REALISATIONS, N_STEPS, T_END, K1, K2)
    mean_trajectory = np.mean(N, axis=0)
    exact_trajectory = deterministic(t, K1, K2)

    # steady state: the relaxation time is 1/k2, so by t = 250 = 7.5/k2 the
    # transient is long gone.  Pool the last 10% of every realisation.
    tail = N[:, int(0.9 * N_STEPS):].ravel()

    lam_expected = K1 / K2
    lam_fitted = np.mean(tail)          # the ML estimate of a Poisson mean

    print("Birth-death model:  k1 = %.2f, k2 = %.3f, %d realisations"
          % (K1, K2, N_REALISATIONS))
    print("(a) max |mean - deterministic| = %.4f"
          % np.max(np.abs(mean_trajectory - exact_trajectory)))
    print("(b) steady-state mean     = %.4f" % lam_fitted)
    print("    expected mean k1/k2   = %.4f" % lam_expected)
    print("    steady-state variance = %.4f  (a Poisson has variance = mean)"
          % np.var(tail))

    lo = max(0.0, np.floor(lam_fitted - 5.0 * np.sqrt(lam_fitted)))
    hi = np.ceil(lam_fitted + 5.0 * np.sqrt(lam_fitted))
    values, prob = histogram(tail, lo, hi)
    poisson = poisson_pmf(values, lam_fitted)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(t, mean_trajectory, lw=1.4, label="mean trajectory")
    ax[0].plot(t, exact_trajectory, "--", lw=1.4, label="deterministic")
    for row in range(3):        # a few single realisations, for context
        ax[0].plot(t, N[row], lw=0.5, alpha=0.4, color="grey")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("n(t)")
    ax[0].set_title("(a) mRNA concentration vs time")
    ax[0].legend()

    ax[1].bar(values, prob, width=0.9, alpha=0.6, label="simulated")
    ax[1].plot(values, poisson, "r*-", ms=5, lw=0.8,
               label=r"Poisson, $\lambda$ = %.2f" % lam_fitted)
    ax[1].set_xlabel("n")
    ax[1].set_ylabel("P(n)")
    ax[1].set_title("(b) steady-state distribution")
    ax[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
