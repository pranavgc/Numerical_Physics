"""
Worksheet 3, Problem 1  --  Python 3 corrected version of p1.py

Generate N >> 1 uniformly distributed random numbers in [0, 1] and write a
function to compute their probability distribution function.

Changes made relative to the original p1.py
-------------------------------------------
1. "print 'area under the graph', v" was Python 2 syntax and stopped the file
   from parsing under Python 3.
2. The binning used  z.count(i)  inside a loop over 10000 bins, i.e. 10^8
   list scans -- quadratic in the sample size.  Replaced by a single pass
   over the data that increments one counter per sample, which is O(N) and
   is still written out by hand.
3. The number of bins was tied to the number of samples (10000 bins for
   10000 samples, so roughly one sample per bin).  Bin count and sample
   count are now separate, and the default of 50 bins for 10^5 samples
   gives a PDF you can actually read.
4. "import collections" was never used and has been dropped.
5. plt.legend() was called with no labelled artists, producing a warning.
6. The normalisation check now also reports the mean and the variance
   against their analytic values 1/2 and 1/12.

Algorithms are hand written: the histogram is not taken from a library.
"""

import numpy as np
import matplotlib.pyplot as plt


def pdf(samples, n_bins, lo=None, hi=None):
    """
    Probability density of `samples`, computed by hand.

    Returns (centres, density) where sum(density) * bin_width == 1.

    One pass over the data: each sample is mapped to a bin index and that
    counter is incremented.  Samples landing exactly on the upper edge are
    folded into the last bin.
    """
    lo = float(np.min(samples)) if lo is None else float(lo)
    hi = float(np.max(samples)) if hi is None else float(hi)
    width = (hi - lo) / n_bins

    counts = np.zeros(n_bins)
    for s in samples:
        idx = int((s - lo) / width)
        if idx == n_bins:          # a sample sitting exactly on the top edge
            idx = n_bins - 1
        if 0 <= idx < n_bins:
            counts[idx] += 1.0

    centres = lo + (np.arange(n_bins) + 0.5) * width
    density = counts / (np.sum(counts) * width)
    return centres, density


def main():
    n_samples = 100000
    n_bins = 50

    x = np.random.rand(n_samples)
    centres, density = pdf(x, n_bins, lo=0.0, hi=1.0)
    width = centres[1] - centres[0]

    area = np.sum(density) * width
    print("samples              = %d" % n_samples)
    print("bins                 = %d" % n_bins)
    print("area under the PDF   = %.6f   (exact 1)" % area)
    print("mean                 = %.6f   (exact %.6f)" % (np.mean(x), 0.5))
    print("variance             = %.6f   (exact %.6f)"
          % (np.var(x), 1.0 / 12.0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(centres, density, width=width * 0.9, alpha=0.6,
           label="measured PDF")
    ax.axhline(1.0, color="red", ls="--",
               label="analytic P(x) = 1 on [0,1]")
    ax.set_xlabel("X")
    ax.set_ylabel("P(X)")
    ax.set_ylim(0.0, 1.4)
    ax.set_title("Normalised uniform distribution, N = %d" % n_samples)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
