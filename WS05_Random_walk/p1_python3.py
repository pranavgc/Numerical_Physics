"""
Worksheet 5, Problem 1  --  Python 3 corrected version of p1.py

A random walker on a 1-d lattice hops left or right with equal probability,
starting from the origin.  Store the final position after N steps, repeat for
an ensemble of M walkers, plot the normalised distribution of final
positions, compute the mean and the variance, and show that the variance
grows linearly with N.

Changes made relative to the original p1.py
-------------------------------------------
1. M = 100 walkers is not "M >> 1"; the variance estimate it produces is so
   noisy that the linear growth is hard to see.  Raised to 20000.
2. The mean position was computed into z1 and then never used, although the
   worksheet asks for it.  It is now reported alongside the variance.
3. The histogram shown was only the final ensemble (N = 999); every earlier
   ensemble was overwritten and thrown away.  Distributions for three
   different N are now kept and plotted together.
4. The inner walk was a Python loop calling random.choice once per step,
   which is about a hundred times slower than drawing the steps as an array.
   The walk is still built explicitly from +-1 steps -- nothing is imported
   to do it -- but the steps are drawn in one go and summed.
5. The variance is fitted against N by hand and the slope compared with the
   analytic value 1 (Var = N for unit steps with p = q = 1/2).
6. The final-position histogram is compared with the Gaussian limit
   N(0, N) that the central limit theorem predicts.

Algorithms are hand written: the walk, the histogram and the least-squares
fit are all implemented here.
"""

import numpy as np
import matplotlib.pyplot as plt


def walk_endpoints(n_steps, n_walkers):
    """
    Final position of `n_walkers` independent walkers after `n_steps` hops.

    Each step is +1 or -1 with probability 1/2.  The steps are drawn as a
    (n_walkers, n_steps) array of uniforms, mapped to +-1, and summed along
    the step axis -- exactly the same walk the original for-loop performed,
    just done a row at a time.
    """
    steps = np.where(np.random.rand(n_walkers, n_steps) < 0.5, -1, 1)
    return np.sum(steps, axis=1)


def pdf(samples, lo, hi, bin_width=2.0):
    """
    Probability density of `samples`, computed by hand in a single pass.

    After an even number of steps the endpoint is always even (and after an
    odd number, always odd), so the lattice only populates every second
    integer.  A bin width of 2 puts exactly one reachable site in each bin
    and avoids the empty-bin comb that a width of 1 would produce.
    """
    n_bins = int(np.ceil((hi - lo) / bin_width))
    counts = np.zeros(n_bins)

    for s in samples:
        idx = int((s - lo) / bin_width)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0

    centres = lo + (np.arange(n_bins) + 0.5) * bin_width
    density = counts / (len(samples) * bin_width)
    return centres, density


def least_squares_slope(x, y):
    """Slope and intercept of the best straight line, computed by hand."""
    xm, ym = np.mean(x), np.mean(y)
    slope = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    return slope, ym - slope * xm


def gaussian(x, mu, var):
    return np.exp(-((x - mu) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def main():
    n_walkers = 20000
    N_grid = np.arange(100, 1001, 50)

    means = np.zeros(len(N_grid))
    variances = np.zeros(len(N_grid))

    for k, N in enumerate(N_grid):
        ends = walk_endpoints(N, n_walkers)
        means[k] = np.mean(ends)
        variances[k] = np.var(ends)

    slope, intercept = least_squares_slope(N_grid.astype(float), variances)

    print("M = %d walkers per ensemble" % n_walkers)
    print("mean position averaged over all N : %+.4f   (exact 0)"
          % np.mean(means))
    print("Var(x) = %.4f * N %+.4f      (exact Var = 1.0 * N)"
          % (slope, intercept))

    # --- distributions at three values of N ------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    for N in [100, 400, 900]:
        ends = walk_endpoints(N, n_walkers)
        lim = 4.0 * np.sqrt(N)
        centres, density = pdf(ends, -lim, lim)
        line, = ax[0].plot(centres, density, lw=1.2, label="N = %d" % N)
        ax[0].plot(centres, gaussian(centres, 0.0, float(N)), "--",
                   lw=1.0, color=line.get_color())

    ax[0].set_xlabel("End position")
    ax[0].set_ylabel("P(end position)")
    ax[0].set_title("Normalised distribution of final positions\n"
                    "(dashed: Gaussian limit with variance N)")
    ax[0].legend(fontsize=8)

    ax[1].plot(N_grid, variances, "o", ms=4, label="measured")
    ax[1].plot(N_grid, slope * N_grid + intercept, "-", color="red",
               label="fit: %.3f N %+.2f" % (slope, intercept))
    ax[1].set_xlabel("N (number of steps)")
    ax[1].set_ylabel("Var(x)")
    ax[1].set_title(r"Var(x) $\propto$ N")
    ax[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
