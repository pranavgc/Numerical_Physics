"""
Worksheet 3, Problem 3  --  Python 3 corrected version of p3.py

Generate N >> 1 random numbers with a Lorentzian density

        P(x) = gamma / (pi (gamma^2 + x^2))

and plot the PDF for three different gamma.

Changes made relative to the original p3.py
-------------------------------------------
1. The clamping of the uniform variate away from 0 and 1 was done with a
   Python for loop over every sample; it is now a single vectorised line.
2. The binning used  z.count(i)  in a loop over 1000 bins -- quadratic in
   the sample size.  Replaced by a single O(N) pass, still hand written.
3. The rescaling  (a + abs(min(a))) / (max(a) - min(a))  only mapped to
   [0,1] when min(a) < 0.  The histogram helper now takes an explicit
   window, so no rescaling is needed.
4. The original binned over the full sampled range, which for a Lorentzian
   is dominated by a handful of extreme tail values -- almost every bin ends
   up empty and the plotted curve is a spike.  The histogram window is now
   set to a few gamma around the origin, where the density is worth looking
   at, and the fraction of samples falling outside is reported.
5. plt.subplot(2,2,i+1) left an empty fourth panel for three curves.
6. The maximum deviation from the analytic density is printed.

Algorithms are hand written: the generator and the histogram are both
implemented here.

Inverse transform sampling
--------------------------
The Lorentzian CDF is  F(x) = 1/2 + arctan(x/gamma)/pi,  so inverting it,
    x = gamma tan(pi (u - 1/2)),     u uniform on (0, 1)
gives Lorentzian variates.  u is kept strictly inside (0,1) because the
tangent diverges at both ends.

The Lorentzian has no finite mean or variance, so unlike the Gaussian case
there is nothing useful to check the sample moments against; the comparison
is made against the density itself.
"""

import numpy as np
import matplotlib.pyplot as plt


def lorentzian_samples(gamma, n, eps=1e-12):
    """
    n Lorentzian variates of half-width gamma, by inverse transform.

    u is clamped to [eps, 1-eps] rather than to the original [0.01, 0.99]:
    the wide clamp truncated the distribution at the 1st and 99th
    percentiles, which is a visible bias.  eps = 1e-12 only removes the
    samples that would literally overflow the tangent.
    """
    u = np.random.rand(n)
    u = np.clip(u, eps, 1.0 - eps)
    return gamma * np.tan(np.pi * (u - 0.5))


def lorentzian_pdf(x, gamma):
    """The analytic density the worksheet asks us to verify against."""
    return gamma / (np.pi * (gamma ** 2 + x ** 2))


def pdf(samples, n_bins, lo, hi):
    """
    Probability density of `samples` over the window [lo, hi], computed by
    hand in a single pass.  Samples outside the window are counted but not
    binned, and the density is normalised by the *total* sample count so the
    truncated tails show up honestly as missing area.

    Returns (centres, density, fraction_outside).
    """
    width = (hi - lo) / n_bins
    counts = np.zeros(n_bins)
    outside = 0

    for s in samples:
        if s < lo or s >= hi:
            outside += 1
            continue
        idx = int((s - lo) / width)
        if idx == n_bins:
            idx = n_bins - 1
        counts[idx] += 1.0

    centres = lo + (np.arange(n_bins) + 0.5) * width
    density = counts / (len(samples) * width)
    return centres, density, outside / float(len(samples))


def main():
    n_samples = 400000
    n_bins = 150
    gammas = [2.0, 3.0, 6.0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    print("%-8s %-14s %-16s %-14s" %
          ("gamma", "window", "frac. in tails", "max |dP|"))

    for ax, gamma in zip(axes, gammas):
        x = lorentzian_samples(gamma, n_samples)

        lo, hi = -8.0 * gamma, 8.0 * gamma
        centres, density, outside = pdf(x, n_bins, lo, hi)
        exact = lorentzian_pdf(centres, gamma)

        print("%-8.1f [%5.1f,%5.1f]  %-16.4f %-14.3e"
              % (gamma, lo, hi, outside, np.max(np.abs(density - exact))))

        ax.plot(centres, density, lw=1.2, label="generated")
        ax.plot(centres, exact, "--", lw=1.6, label="analytic")
        ax.set_title(r"$\gamma$ = %.1f" % gamma)
        ax.set_xlabel("X")
        ax.set_ylabel("P(X)")
        ax.legend(fontsize=8)

    fig.suptitle("Lorentzian variates by inverse transform, N = %d per panel"
                 % n_samples)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
