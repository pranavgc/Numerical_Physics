"""
Generate the figures the worksheets ask for, from the corrected scripts.

Runs each pN_python3.py and saves the figure it produces as <stem>.png in the
same folder.  The scripts are NOT modified: matplotlib's show() is patched to
save instead of blocking, and each script is executed with runpy exactly as
if it had been run from the command line.

The random seed is fixed before each script so the stochastic figures
(WS03, WS04, WS05, WS06) are reproducible.  Re-running this file regenerates
byte-identical images.

Usage:  MPLBACKEND=Agg python3 make_figures.py [output_root]
"""

import os
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 20260922
DPI = 150

SCRIPTS = [
    "WS01_Verlet/p1_python3.py",
    "WS01_Verlet/p2_python3.py",
    "WS02_RK4/p1_python3.py",
    "WS02_RK4/p2_python3.py",
    "WS03_box_muller/p1_python3.py",
    "WS03_box_muller/p2_python3.py",
    "WS03_box_muller/p3_python3.py",
    "WS04_Kuramoto/p1_python3.py",
    "WS05_Random_walk/p1_python3.py",
    "WS05_Random_walk/p2E_python3.py",
    "WS05_Random_walk/p2RK4_python3.py",
    "WS06_sde/p1_python3.py",
    "WS06_sde/p2_python3.py",
    "WS07-shooting/p1_python3.py",
    "WS07-shooting/p2a_python3.py",
    "WS07-shooting/p2b_python3.py",
    "WS08-Numerov/p1_python3.py",
    "WS08-Numerov/p2_python3.py",
    # WS09 asks only for the eigenvalues and eigenvectors, which the scripts
    # print.  The worksheet requests no plot, so none is generated.
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else here

    saved = []
    real_show = plt.show

    for rel in SCRIPTS:
        folder, fname = os.path.split(rel)
        stem = os.path.splitext(fname)[0]
        out_dir = os.path.join(out_root, folder)
        os.makedirs(out_dir, exist_ok=True)

        written = []

        def fake_show(*args, **kwargs):
            """Save every open figure instead of opening a window."""
            nums = plt.get_fignums()
            for k, num in enumerate(nums):
                fig = plt.figure(num)
                name = stem if len(nums) == 1 else "%s_%d" % (stem, k + 1)
                path = os.path.join(out_dir, name + ".png")
                fig.savefig(path, dpi=DPI, bbox_inches="tight",
                            facecolor="white")
                written.append(path)
            plt.close("all")

        plt.show = fake_show
        np.random.seed(SEED)
        plt.close("all")

        try:
            runpy.run_path(os.path.join(here, rel), run_name="__main__")
        finally:
            plt.show = real_show
            plt.close("all")

        for p in written:
            size = os.path.getsize(p)
            print("  %-56s %6.1f kB" % (os.path.relpath(p, out_root),
                                        size / 1024.0))
            saved.append(p)

    print("\n%d figures written under %s" % (len(saved), out_root))


if __name__ == "__main__":
    main()
