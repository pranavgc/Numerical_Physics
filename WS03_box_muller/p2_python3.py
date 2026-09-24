"""
Worksheet 3, Problem 2  --  Python 3 corrected version of p2.py

Generate N >> 1 random numbers drawn from the normal distribution N[mu,
sigma^2] with the Box-Muller transform, and verify that the probability
distribution function is

        P(x) = 1/sqrt(2 pi sigma^2) exp(-(x - mu)^2 / (2 sigma^2))

Plot the PDF for three different (mu, sigma).

Changes made relative to the original p2.py
-------------------------------------------
1. Only the cosine branch of the Box-Muller pair was used, throwing away
   half of every generated pair.  Both branches are now kept, so the same
   number of uniforms yields twice as many normals.
2. The rescaling before binning was
        b = (a + abs(min(a))) / (max(a) - min(a))
   which only maps to [0,1] when min(a) < 0.  It happened to hold for the
   three parameter sets used, and would have silently produced garbage for a
   strictly positive sample.  Corrected to (a - min) / (max - min); in fact
   the binning helper now takes the range directly and no rescaling is
   needed at all.
3. The binning used  z.count(i)  in a loop over 1000 bins -- quadratic in
   the sample size.  Replaced by a single O(N) pass, still hand written.
4. plt.subplot(2,2,i+1) left an empty fourth panel for three curves.
5. sigma = 16 in the original was labelled "STDDEV" but the third case is
   now (mu, sigma) = (2, 4), a value where the curve is still visible on the
   same axes as the other two.
6. The maximum absolute deviation between the measured and the analytic PDF
   is printed, so the verification the worksheet asks for is a number rather
   than an eyeball test.

Algorithms are hand written: the Gaussian generator and the histogram are
both implemented here, not taken from a library.

Box-Muller:  given u1, u2 uniform on (0,1],
    z0 = sqrt(-2 ln u1) cos(2 pi u2)
    z1 = sqrt(-2 ln u1) sin(2 pi u2)
are two independent standard normals.  x = mu + sigma z.
"""

import numpy as np
import matplotlib.pyplot as plt


def box_muller(mu, sigma, n_pairs):
    """
    Return 2 * n_pairs normal variates with mean mu and standard deviation
    sigma, using both branches of the Box-Muller transform.

    u1 is drawn on (0, 1] rather than [0, 1) so that log(u1) never overflows.
    """
    u1 = 1.0 - np.random.rand(n_pairs)      # (0, 1]
    u2 = np.random.rand(n_pairs)            # [0, 1)

    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2

    z0 = radius * np.cos(angle)
    z1 = radius * np.sin(angle)
    return mu + sigma * np.concatenate((z0, z1))


def gaussian(x, mu, sigma):
    """The analytic density the worksheet asks us to verify against."""
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)) \
        / np.sqrt(2.0 * np.pi * sigma ** 2)


def pdf(samples, n_bins, lo, hi):
    """
    Probability density of `samples` over [lo, hi], computed by hand in a
    single pass.  Returns (centres, density).
    """
    width = (hi - lo) / n_bins
    counts = np.zeros(n_bins)
    inside = 0

    for s in samples:
        idx = int((s - lo) / width)
        if idx == n_bins:
            idx = n_bins - 1
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
            inside += 1

    centres = lo + (np.arange(n_bins) + 0.5) * width
    # normalise over the whole sample, not just the part inside the window,
    # so that clipped tails show up as missing area rather than being hidden
    density = counts / (len(samples) * width)
    return centres, density


def main():
    n_pairs = 200000        # -> 400000 variates per parameter set
    n_bins = 120

    params = [(0.0, 1.0), (2.0, 1.0), (2.0, 4.0)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    print("%-14s %-12s %-12s %-14s" %
          ("(mu, sigma)", "mean", "std", "max |dP|"))

    for ax, (mu, sigma) in zip(axes, params):
        x = box_muller(mu, sigma, n_pairs)

        lo, hi = mu - 5.0 * sigma, mu + 5.0 * sigma
        centres, density = pdf(x, n_bins, lo, hi)
        exact = gaussian(centres, mu, sigma)

        print("(%4.1f, %4.1f)  %-12.5f %-12.5f %-14.3e"
              % (mu, sigma, np.mean(x), np.std(x),
                 np.max(np.abs(density - exact))))

        ax.plot(centres, density, lw=1.2, label="generated")
        ax.plot(centres, exact, "--", lw=1.6, label="analytic")
        ax.set_title(r"$\mu$ = %.1f,  $\sigma$ = %.1f" % (mu, sigma))
        ax.set_xlabel("X")
        ax.set_ylabel("P(X)")
        ax.legend(fontsize=8)

    fig.suptitle("Box-Muller normal variates, N = %d per panel"
                 % (2 * n_pairs))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
